from __future__ import annotations

import copy
import dataclasses
import functools
import importlib
import json
import math
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import toml
import torch
from mpl_toolkits.mplot3d import Axes3D

from utils.misc import PathUtils, log


@dataclasses.dataclass(kw_only=True)
class CalibrationPlane:
    """
    A world-space plane ``n . p + d == 0`` plus everything needed to estimate it from a
    CaptureStudio session (segmentation, RGB-D loading, RANSAC fit and visualization).

    ``normal`` points "up", i.e. towards the side the cameras are on.
    """
    normal: torch.Tensor  # (3,) unit normal
    offset: torch.Tensor  # () scalar d
    corners: Optional[torch.Tensor] = None  # (4, 3) quad, ccw around the normal
    metadata: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = dataclasses.field(default=None, repr=False, compare=False)

    # ---- configuration / lazily-loaded segmentor (shared by all instances) -------------
    SEGMENTOR_NAME = 'nvidia/segformer-b4-finetuned-ade-512-512'
    SEGMENTOR = None  # (processor, model)
    FLOOR_IDS = None  # ADE20k class ids matching FLOOR_TOKENS
    FLOOR_TOKENS = ('floor', 'ground', 'road', 'sidewalk', 'pavement')

    # ==================================================================================
    # basic API
    # ==================================================================================
    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        """Height above the plane (positive on the camera side). points: (..., 3)."""
        return points @ self.normal.to(points) + self.offset.to(points)

    def project(self, points: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection of points onto the plane."""
        return points - self.signed_distance(points)[..., None] * self.normal.to(points)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            normal=self.normal.detach().cpu().flatten().tolist(),
            offset=float(self.offset.detach().cpu().flatten()[0]),
            corners=None if self.corners is None else self.corners.detach().cpu().reshape(-1, 3).tolist(),
            metadata=self.metadata,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationPlane':
        corners = data.get('corners')
        return cls(
            normal=torch.tensor(data['normal'], dtype=torch.float32).reshape(3),
            offset=torch.tensor(float(data['offset']), dtype=torch.float32),
            corners=None if corners is None else torch.tensor(corners, dtype=torch.float32).reshape(-1, 3),
            metadata=data.get('metadata'),
        )

    # ==================================================================================
    # small numeric helpers
    # ==================================================================================
    @staticmethod
    def _as_np(x) -> Optional[np.ndarray]:
        if x is None:
            return None
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    @staticmethod
    def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        return v / (np.linalg.norm(v) + eps)

    @staticmethod
    def _weighted_fit(points: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """Weighted total-least-squares plane through `points`."""
        w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
        w /= w.sum()
        mu = (w[:, None] * points).sum(0)
        _, _, vt = np.linalg.svd(np.sqrt(w)[:, None] * (points - mu), full_matrices=False)
        n = CalibrationPlane._unit(vt[-1])
        return n, float(-n @ mu)

    @classmethod
    def _ransac(cls, rng, points: np.ndarray, tau: float, iters: int = 2000,
                score_points: int = 25_000, chunk: int = 256) -> Tuple[np.ndarray, float]:
        """Plane RANSAC; candidates are scored on a random subset to stay O(iters * subset)."""
        sub = points if points.shape[0] <= score_points else points[rng.choice(points.shape[0], score_points, replace=False)]
        tri = rng.integers(0, sub.shape[0], size=(iters, 3))
        a, b, c = sub[tri[:, 0]], sub[tri[:, 1]], sub[tri[:, 2]]
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        ok = norms > 1e-9
        if not ok.any():
            return cls._weighted_fit(points, np.ones(points.shape[0]))
        normals = normals[ok] / norms[ok, None]
        offsets = -np.einsum('ij,ij->i', normals, a[ok])

        best = (-1, normals[0], float(offsets[0]))
        for s in range(0, normals.shape[0], chunk):
            ns, ds = normals[s:s + chunk], offsets[s:s + chunk]
            cnt = (np.abs(sub @ ns.T + ds[None, :]) <= tau).sum(0)
            j = int(cnt.argmax())
            if int(cnt[j]) > best[0]:
                best = (int(cnt[j]), ns[j], float(ds[j]))
        return best[1], best[2]

    @classmethod
    def _quad_from_points(cls, points: np.ndarray, normal: np.ndarray, pct: float = 2.0) -> Optional[np.ndarray]:
        """Robust in-plane bounding rectangle of `points` (PCA axes, percentile extents)."""
        if points.shape[0] < 3:
            return None
        mu = points.mean(0)
        x0 = points - mu
        _, ev = np.linalg.eigh((x0.T @ x0) / max(points.shape[0] - 1, 1))
        e1 = ev[:, 2] - normal * (ev[:, 2] @ normal)
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
        e1 = cls._unit(e1)
        e2 = cls._unit(np.cross(normal, e1))
        p = float(np.clip(pct, 0.0, 25.0))
        (umin, umax), (vmin, vmax) = np.percentile(x0 @ e1, [p, 100 - p]), np.percentile(x0 @ e2, [p, 100 - p])
        quad = np.stack([mu + umin * e1 + vmin * e2, mu + umax * e1 + vmin * e2,
                         mu + umax * e1 + vmax * e2, mu + umin * e1 + vmax * e2], 0)
        if np.cross(quad[1] - quad[0], quad[3] - quad[0]) @ normal < 0:  # keep winding along the normal
            quad = quad[[0, 3, 2, 1]]
        return quad.astype(np.float64)

    @classmethod
    def _quad_from_cameras(cls, centers: np.ndarray, normal: np.ndarray, offset: float, margin: float) -> np.ndarray:
        """Fallback floor quad: camera centres projected onto the plane, padded by `margin`."""
        proj = centers - (centers @ normal + offset)[:, None] * normal
        mu = proj.mean(0)
        x0 = proj - mu
        if x0.shape[0] >= 2 and np.linalg.norm(x0) > 1e-9:
            _, _, vt = np.linalg.svd(x0, full_matrices=False)
            e1 = vt[0] - normal * (vt[0] @ normal)
        else:
            e1 = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
        e1 = cls._unit(e1)
        e2 = cls._unit(np.cross(normal, e1))
        u, v = x0 @ e1, x0 @ e2
        umin, umax = float(u.min()) - margin, float(u.max()) + margin
        vmin, vmax = float(v.min()) - margin, float(v.max()) + margin
        return np.stack([mu + umin * e1 + vmin * e2, mu + umax * e1 + vmin * e2,
                         mu + umax * e1 + vmax * e2, mu + umin * e1 + vmax * e2], 0)

    @staticmethod
    def _rot_from_z(target: np.ndarray) -> np.ndarray:
        """Rotation taking +Z to `target`."""
        z, t = np.array([0.0, 0.0, 1.0]), CalibrationPlane._unit(np.asarray(target, dtype=np.float64))
        v, c = np.cross(z, t), float(z @ t)
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + vx + vx @ vx / (1.0 + c)

    # ==================================================================================
    # floor segmentation
    # ==================================================================================
    @classmethod
    def _segmentor(cls, device: Optional[str] = None):
        if cls.SEGMENTOR is None:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
            processor = AutoImageProcessor.from_pretrained(cls.SEGMENTOR_NAME)
            try:
                model = AutoModelForSemanticSegmentation.from_pretrained(cls.SEGMENTOR_NAME, dtype=dtype)
            except TypeError:  # older transformers
                model = AutoModelForSemanticSegmentation.from_pretrained(cls.SEGMENTOR_NAME, torch_dtype=dtype)
            model = model.to(device).eval()
            id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
            cls.FLOOR_IDS = [k for k, v in id2label.items() if any(t in v for t in cls.FLOOR_TOKENS)] or [3]
            cls.SEGMENTOR = (processor, model)
            log(f'[CalibrationPlane] floor classes: {[id2label[i] for i in cls.FLOOR_IDS]}', 'debug')
        return cls.SEGMENTOR

    @staticmethod
    def auto_brighten(rgb: np.ndarray) -> np.ndarray:
        """Linear-light exposure lift, so that dark studio floors still segment well."""
        rgb = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
        x = rgb.astype(np.float32) / 255.0
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        luma = lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        valid = luma > 1e-5
        if np.count_nonzero(valid) < 64:
            return rgb.copy()
        gain = float(np.clip(0.75 / max(float(np.percentile(luma[valid], 90.0)), 1e-6), 1.0, 6.0))
        lin = lin * (1.0 + (gain - 1.0) * (luma / (luma + 0.035))[..., None])  # protect deep shadows
        lin = lin / (1.0 + 0.08 * lin)  # soft highlight shoulder
        out = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.maximum(lin, 1e-8) ** (1 / 2.4) - 0.055)
        return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)

    @classmethod
    def floor_probability(cls, rgb: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """HxW float32 floor probability in [0, 1]."""
        processor, model = cls._segmentor(device)
        rgb = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
        inputs = {k: (v.to(model.device, dtype=model.dtype) if torch.is_floating_point(v) else v.to(model.device))
                  for k, v in processor(images=rgb, return_tensors='pt').items()}
        with torch.inference_mode():
            logits = model(**inputs).logits.float()
            logits = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode='bilinear', align_corners=False)[0]
            prob = logits.softmax(dim=0)[cls.FLOOR_IDS].sum(dim=0).cpu().numpy()
        return np.clip(prob, 0.0, 1.0).astype(np.float32)

    # ==================================================================================
    # RGB-D view loading
    # ==================================================================================
    @classmethod
    def _camera_dir(cls, session_path: Path, cam_name: str) -> Optional[Path]:
        short = cam_name.split('/')[-1]
        for c in (session_path / 'orbbec' / short, session_path / short):
            if c.is_dir() and (c / 'color').is_dir():
                return c
        return None

    @staticmethod
    def _timestamps(directory: Path, suffixes: Sequence[str]) -> List[int]:
        out = []
        for p in directory.iterdir():
            if p.suffix.lower() in suffixes:
                try:
                    out.append(int(p.stem))
                except ValueError:
                    pass
        return sorted(out)

    @classmethod
    def load_session_views(cls, session_path: Union[str, Path], cam_names: Sequence[str],
                           intrinsics, extrinsics_c2w, image_size=None,
                           max_views: int = 5,
                           max_side: int = 1024,
                           prerotation: Optional[str] = None,
                           frame_fraction: float = 0.5,
                           min_valid_depth_ratio: float = 0.02) -> List[Dict[str, Any]]:
        """
        Loads one (color, depth) pair for a few cameras spread around the rig and unprojects
        them to world points.

        The per-frame ``mask/`` images are *foreground/person* masks and are deliberately NOT
        applied -- they would remove exactly the floor pixels we are after.
        """
        session_path = Path(session_path).expanduser()
        K_all, c2w_all = cls._as_np(intrinsics).astype(np.float64), cls._as_np(extrinsics_c2w).astype(np.float64)
        size_all = cls._as_np(image_size)

        candidates = [i for i, n in enumerate(cam_names) if cls._camera_dir(session_path, n) is not None]
        if not candidates:
            log(f'[CalibrationPlane::load_session_views] no camera folders with color frames under {session_path}', 'warning')
            return []
        order = list(
            dict.fromkeys(
                candidates[j] for j in
                np.linspace(0, len(candidates) - 1, num=min(len(candidates), 2 * max_views)).round().astype(int).tolist()
            )
        )
        views: List[Dict[str, Any]] = []
        for cam_i in order:
            if len(views) >= max_views:
                break
            name = cam_names[cam_i]
            cam_dir = cls._camera_dir(session_path, name)
            color_dir = cam_dir / 'color'
            depth_dir = cam_dir / 'depth_aligned' if (cam_dir / 'depth_aligned').is_dir() else None
            if depth_dir is None:
                log(f'[CalibrationPlane::load_session_views] {name}: no depth folder, skipped', 'warning')
                continue
            color_ts = cls._timestamps(color_dir, ('.jpg', '.jpeg', '.png'))
            depth_ts = cls._timestamps(depth_dir, ('.png', '.npy'))
            if not color_ts or not depth_ts:
                log(f'[CalibrationPlane::load_session_views] {name}: no frames, skipped', 'warning')
                continue
            c_ts = color_ts[min(int(len(color_ts) * float(np.clip(frame_fraction, 0.0, 0.999))), len(color_ts) - 1)]
            d_ts = min(depth_ts, key=lambda t: abs(t - c_ts))
            if abs(d_ts - c_ts) > 100:
                log(f'[CalibrationPlane::load_session_views] {name}: color/depth ts differ by {abs(d_ts - c_ts)}', 'warning')
            color_path = next((p for p in (color_dir / f'{c_ts}{s}' for s in ('.jpg', '.jpeg', '.png')) if p.is_file()), None)
            depth_path = next((p for p in (depth_dir / f'{d_ts}{s}' for s in ('.png', '.npy')) if p.is_file()), None)
            if color_path is None or depth_path is None:
                continue
            bgr = PathUtils.read_file(color_path)
            depth = PathUtils.read_file(depth_path, png_type='depth').astype(np.float32).squeeze() / 1000.0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            K, c2w = K_all[cam_i].copy(), c2w_all[cam_i].copy()
            if prerotation:  # calibration already refers to the rotated image -> rotate pixels only
                flag = getattr(cv2, f'ROTATE_{prerotation.upper()}')
                rgb, depth = cv2.rotate(rgb, flag), cv2.rotate(depth, flag)

            h, w = rgb.shape[:2]
            if size_all is not None:  # calibration resolution -> actual image resolution
                ch, cw = int(size_all[cam_i][0]), int(size_all[cam_i][1])
                if ch > 0 and cw > 0 and (ch, cw) != (h, w):
                    K[0, :] *= w / float(cw)
                    K[1, :] *= h / float(ch)
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            if (s := min(1.0, max_side / max(h, w))) < 1.0:  # downscale for speed
                nw, nh = int(round(w * s)), int(round(h * s))
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
                K[0, :] *= nw / float(w)
                K[1, :] *= nh / float(h)
                h, w = nh, nw

            valid = np.isfinite(depth) & (depth > 0)
            if float(valid.mean()) < min_valid_depth_ratio:
                log(f'[CalibrationPlane::load_session_views] {name}: only {valid.mean() * 100:.1f}% valid depth, skipped', 'warning')
                continue
            med = float(np.median(depth[valid]))
            if not 0.2 <= med <= 15.0:
                log(f'[CalibrationPlane::load_session_views] {name}: suspicious median depth {med:.2f} m', 'warning')

            uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            d64 = depth.astype(np.float64)
            cam_pts = np.stack([(uu - K[0, 2]) * d64 / K[0, 0], (vv - K[1, 2]) * d64 / K[1, 1], d64], axis=-1)
            points_world = cam_pts @ c2w[:3, :3].T + c2w[:3, 3]
            points_world[~valid] = np.nan

            views.append(dict(name=name, rgb=rgb, depth=depth.astype(np.float32), valid=valid,
                              points_world=points_world, K=K, c2w=c2w, cam_index=cam_i))
            log(f'[CalibrationPlane::load_session_views] {name}: {color_path.name} + {depth_path.name} '
                f'({w}x{h}, median depth {med:.2f} m)', 'debug')
        return views

    # ==================================================================================
    # estimation
    # ==================================================================================
    @staticmethod
    def _debug_tile(rgb: np.ndarray, prob: np.ndarray, mask: np.ndarray, max_side: int):
        h, w = rgb.shape[:2]
        if (s := min(1.0, max_side / max(h, w))) < 1.0:
            size = (int(round(w * s)), int(round(h * s)))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            prob = cv2.resize(prob, size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        return rgb.copy(), (np.clip(prob, 0, 1) * 255).astype(np.uint8), mask.astype(bool)

    @classmethod
    def estimate(
            cls,
            views: List[Dict[str, Any]],
            prob_threshold: float = 0.9995,
            min_points_per_view: int = 512,
            depth_range: Tuple[float, float] = (0.2, 6.0),
            max_points: int = 300_000,
            ransac_iters: int = 2000,
            tau_m: Optional[float] = None,
            max_normal_deviation_deg: float = 25.0,
            corner_percentile: float = 2.0,
            brighten: bool = True,
            keep_debug: bool = True,
            debug_max_side: int = 720,
            device: Optional[str] = None,
            seed: int = 42
    ) -> Optional['CalibrationPlane']:
        """
        Per-view weighted plane -> cross-view consistency filter -> global RANSAC + weighted
        refits -> normal oriented towards the cameras -> robust corner quad.
        """
        if not views:
            return None
        rng = np.random.default_rng(seed)

        per_view: List[Dict[str, Any]] = []
        for v in views:
            prob = cls.floor_probability(cls.auto_brighten(v['rgb']) if brighten else v['rgb'], device=device)
            depth = v['depth']
            m = (prob >= prob_threshold) & np.isfinite(depth) & (depth > depth_range[0]) & (depth <= depth_range[1])
            m &= np.isfinite(v['points_world']).all(-1)
            if v.get('valid') is not None:
                m &= v['valid']
            e = dict(name=v.get('name', '?'), num_points=int(m.sum()), coverage=float(m.mean()),
                     mean_prob=float(prob[m].mean()) if m.any() else 0.0,
                     used=False, angle_deg=None, inlier_ratio=None, normal=None)
            if e['num_points'] >= min_points_per_view:
                e['points'], e['weights'] = v['points_world'][m].astype(np.float64), prob[m].astype(np.float64)
                e['normal'], _ = cls._weighted_fit(e['points'], e['weights'])
            else:
                log(f"[CalibrationPlane::estimate] {e['name']}: only {e['num_points']} floor points, skipped", 'warning')
            if keep_debug:
                e['debug_rgb'], e['debug_prob'], e['debug_mask'] = cls._debug_tile(v['rgb'], prob, m, debug_max_side)
            per_view.append(e)

        usable = [e for e in per_view if e['normal'] is not None]
        if not usable:
            log('[CalibrationPlane::estimate] no view produced enough floor points', 'warning')
            return None

        # cross-view consistency: drop views whose normal disagrees with the median one
        normals = np.stack([e['normal'] for e in usable], 0)
        signs = np.where(normals @ normals[0] < 0, -1.0, 1.0)
        med = cls._unit(np.median(normals * signs[:, None], axis=0))
        for e, n_v in zip(usable, normals):
            e['angle_deg'] = math.degrees(math.acos(float(np.clip(abs(n_v @ med), -1.0, 1.0))))
            e['used'] = e['angle_deg'] <= max_normal_deviation_deg
        kept = [e for e in usable if e['used']] or usable
        for e in usable:
            e['used'] = e in kept
            if not e['used']:
                log(f"[CalibrationPlane::estimate] {e['name']}: normal off by {e['angle_deg']:.1f} deg, rejected", 'warning')

        points = np.concatenate([e['points'] for e in kept], 0)
        weights = np.concatenate([e['weights'] for e in kept], 0)
        if points.shape[0] > max_points:
            sel = rng.choice(points.shape[0], max_points, replace=False)
            points, weights = points[sel], weights[sel]

        scale = float(np.median(np.linalg.norm(points - points.mean(0), axis=1))) or 1.0
        tau = float(tau_m) if tau_m else max(0.005, 0.01 * scale)
        normal, offset = cls._ransac(rng, points, tau, iters=ransac_iters)
        for _ in range(2):  # weighted refit + re-selection of inliers
            inl = np.abs(points @ normal + offset) <= tau
            if int(inl.sum()) < 3:
                break
            normal, offset = cls._weighted_fit(points[inl], weights[inl])
        inl = np.abs(points @ normal + offset) <= tau

        centers = np.stack([cls._as_np(v['c2w'])[:3, 3] for v in views], 0)
        if float(np.mean(centers @ normal + offset)) < 0:  # normal must point towards the cameras
            normal, offset = -normal, -offset
        heights = centers @ normal + offset
        residuals = points[inl] @ normal + offset
        corners = cls._quad_from_points(points[inl], normal, corner_percentile)

        for e in per_view:  # per-view agreement with the final plane (for the overlay), then free memory
            if e.get('points') is not None:
                e['inlier_ratio'] = float((np.abs(e['points'] @ normal + offset) <= tau).mean())
            e.pop('points', None)
            e.pop('weights', None)

        metadata = dict(
            views=[e['name'] for e in per_view], views_used=[e['name'] for e in per_view if e['used']],
            num_points=int(points.shape[0]), num_inliers=int(inl.sum()), inlier_ratio=float(inl.mean()),
            rms_m=float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else float('nan'), tau_m=tau,
            angle_to_world_y_deg=math.degrees(math.acos(float(np.clip(abs(normal[1]), -1.0, 1.0)))),
            camera_heights_m=[float(h) for h in heights],
        )
        log(f"[CalibrationPlane::estimate] n={np.round(normal, 4).tolist()} d={offset:+.4f} | "
            f"inliers {metadata['inlier_ratio'] * 100:.1f}% of {metadata['num_points']} pts | "
            f"rms {metadata['rms_m'] * 1000:.1f} mm | cameras {heights.mean():.2f} m above the floor "
            f"(min {heights.min():.2f}, max {heights.max():.2f})", 'info')
        if not 0.3 <= float(heights.mean()) <= 5.0:
            log('[CalibrationPlane::estimate] camera height looks implausible -- check the depth unit / calibration scale', 'warning')

        return cls(
            normal=torch.tensor(normal, dtype=torch.float32),
            offset=torch.tensor(float(offset), dtype=torch.float32),
            corners=None if corners is None else torch.tensor(corners, dtype=torch.float32),
            metadata=metadata,
            debug=dict(per_view=per_view) if keep_debug else None,
        )

    # ==================================================================================
    # visualization
    # ==================================================================================
    @staticmethod
    def _put_label(img: np.ndarray, lines: Sequence[str], color=(255, 255, 255)) -> None:
        h, w = img.shape[:2]
        scale = max(0.45, min(w, h) / 900.0)
        thick, y = max(1, int(round(scale * 2))), 10
        for line in lines:
            (tw, th), bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.rectangle(img, (8, y), (14 + tw, y + th + bl + 8), (0, 0, 0), -1)
            cv2.putText(img, line, (11, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            y += th + bl + 12

    def visualize_probabilities(self, out_path: Union[str, Path], alpha: float = 0.6, max_cols: int = 2,
                                max_side: int = 3840, draw_contour: bool = True) -> bool:
        """(1) Grid of the input views with the floor probability heat-map blended on top."""
        per_view = [e for e in (self.debug or {}).get('per_view', []) if e.get('debug_rgb') is not None]
        if not per_view:
            log('[CalibrationPlane::visualize_probabilities] no debug data (estimate with keep_debug=True)', 'warning')
            return False
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tiles = []
        for e in per_view:
            prob = (e['debug_prob'].astype(np.float32) / 255.0)[..., None]
            bgr = cv2.cvtColor(e['debug_rgb'], cv2.COLOR_RGB2BGR).astype(np.float32)
            heat = cv2.applyColorMap(e['debug_prob'], cv2.COLORMAP_TURBO).astype(np.float32)
            tile = np.clip(bgr * (1.0 - alpha * prob) + heat * (alpha * prob), 0, 255).astype(np.uint8)
            if draw_contour and e['debug_mask'].any():
                contours, _ = cv2.findContours(e['debug_mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tile, contours, -1, (0, 255, 0), max(1, tile.shape[0] // 400))
            lines = [f"{e['name']}  [{'USED' if e['used'] else 'REJECTED'}]",
                     f"floor coverage {e['coverage'] * 100:.1f}%  |  mean p {e['mean_prob']:.3f}"]
            if e['angle_deg'] is not None:
                lines.append(f"normal vs median: {e['angle_deg']:.1f} deg" +
                             (f"  |  on final plane: {e['inlier_ratio'] * 100:.1f}%" if e['inlier_ratio'] is not None else ''))
            self._put_label(tile, lines, color=(255, 255, 255) if e['used'] else (80, 80, 255))
            tiles.append(tile)

        th, tw = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, tw - t.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
        cols = max(1, min(int(max_cols), len(tiles)))
        rows = [np.hstack((tiles[r:r + cols] + [np.zeros_like(tiles[0])] * cols)[:cols]) for r in range(0, len(tiles), cols)]
        grid = np.vstack(rows)
        if (s := min(1.0, max_side / max(grid.shape[:2]))) < 1.0:
            grid = cv2.resize(grid, (int(grid.shape[1] * s), int(grid.shape[0] * s)), interpolation=cv2.INTER_AREA)

        meta = self.metadata or {}
        band = np.zeros((max(56, int(0.08 * grid.shape[0])), grid.shape[1], 3), dtype=np.uint8)
        self._put_label(band, [
            f"floor plane: n={np.round(self._as_np(self.normal), 4).tolist()}  d={float(self.offset):+.4f}",
            f"inliers {meta.get('inlier_ratio', float('nan')) * 100:.1f}%  |  rms {meta.get('rms_m', float('nan')) * 1000:.1f} mm"
            f"  |  tau {meta.get('tau_m', float('nan')) * 1000:.0f} mm"])
        cv2.imwrite(str(out_path), np.vstack([band, grid]))
        log(f'[CalibrationPlane::visualize_probabilities] wrote {out_path}', 'info')
        return True

    def build_scene_mesh(self, intrinsics, extrinsics_c2w, image_size, frustum_depth: float = 0.3,
                         floor_cells: int = 24, floor_margin: float = 1.0, add_world_axes: bool = True,
                         add_normal_arrow: bool = True, out_path: Optional[Union[str, Path]] = None,
                         show: bool = False) -> o3d.geometry.TriangleMesh:
        """
        (2) One mesh with a checkerboard quad on the estimated plane, one colored pyramid per
        camera, and a magenta arrow along the normal. Write to ``.ply`` to keep vertex colors.
        """
        n, d = self._unit(self._as_np(self.normal).reshape(3).astype(np.float64)), float(self._as_np(self.offset).reshape(-1)[0])
        K_all, c2w_all = self._as_np(intrinsics).astype(np.float64), self._as_np(extrinsics_c2w).astype(np.float64)
        size_all, centers = self._as_np(image_size), c2w_all[:, :3, 3]

        quad = self._as_np(self.corners)
        if quad is None or quad.shape[0] < 4:
            quad = self._quad_from_cameras(centers, n, d, floor_margin)
        else:
            quad, mu = quad[:4].astype(np.float64), quad[:4].astype(np.float64).mean(0)
            quad = mu + (quad - mu) * (1.0 + 2.0 * floor_margin / max(1e-6, float(np.linalg.norm(quad[1] - quad[0]))))

        # floor: independent quad per cell so the checkerboard stays crisp
        cells = max(2, int(floor_cells))
        uu, vv = np.meshgrid(np.linspace(0, 1, cells + 1), np.linspace(0, 1, cells + 1), indexing='ij')
        p0, p1, p2, p3 = quad
        grid = (((1 - uu) * (1 - vv))[..., None] * p0 + (uu * (1 - vv))[..., None] * p1 +
                (uu * vv)[..., None] * p2 + ((1 - uu) * vv)[..., None] * p3)
        verts, tris, cols = [], [], []
        light, dark = np.array([0.78, 0.78, 0.80]), np.array([0.32, 0.32, 0.36])
        for i in range(cells):
            for j in range(cells):
                b = len(verts)
                verts += [grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]]
                tris += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
                cols += [light if (i + j) % 2 == 0 else dark] * 4
        scene = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
                                          o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32)))
        scene.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))

        # cameras: pyramid from the optical centre through the image corners
        num_cams = c2w_all.shape[0]
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]], dtype=np.int32)
        for i in range(num_cams):
            K, c2w = K_all[i], c2w_all[i]
            h, w = (int(size_all[i][0]), int(size_all[i][1])) if size_all is not None else (1024, 1024)
            pix = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
            dirs = np.stack([(pix[:, 0] - K[0, 2]) / K[0, 0], (pix[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)], -1)
            world_v = np.vstack([np.zeros((1, 3)), dirs * float(frustum_depth)]) @ c2w[:3, :3].T + c2w[:3, 3]
            col = cv2.applyColorMap(np.uint8([[int(255 * i / max(1, num_cams - 1))]]), cv2.COLORMAP_TURBO)[0, 0]
            frustum = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world_v), o3d.utility.Vector3iVector(faces))
            frustum.vertex_colors = o3d.utility.Vector3dVector(np.tile((col[::-1] / 255.0)[None, :], (5, 1)))
            scene += frustum

        span = float(np.linalg.norm(quad[2] - quad[0])) or 1.0
        if add_normal_arrow:
            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.012 * span, cone_radius=0.028 * span,
                                                           cylinder_height=0.18 * span, cone_height=0.06 * span)
            arrow.rotate(self._rot_from_z(n), center=np.zeros(3))
            arrow.translate(centers.mean(0) - (centers.mean(0) @ n + d) * n)
            arrow.paint_uniform_color([1.0, 0.15, 0.6])
            scene += arrow
        if add_world_axes:
            scene += o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15 * span)
        scene.compute_vertex_normals()

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(str(out_path), scene, write_ascii=False)
            log(f'[CalibrationPlane::build_scene_mesh] wrote {out_path} ({len(scene.vertices)} verts, {num_cams} cameras)', 'info')
        if show:
            try:
                o3d.visualization.draw_geometries([scene], window_name='floor plane + cameras')
            except Exception as e:  # headless
                log(f'[CalibrationPlane::build_scene_mesh] cannot open a window: {e}', 'warning')
        return scene


@dataclasses.dataclass(kw_only=True)
class CalibrationData:
    intrinsics: torch.Tensor  # (N, 3, 3)
    extrinsics_c2w: torch.Tensor  # (N, 4, 4), c2w
    dists: torch.Tensor  # (N, 5) in the following order (k1, k2, p1, p2, k3)
    rotmats: torch.Tensor  # (N, 3, 3), c2w
    tvecs: torch.Tensor  # (N, 3), in world coordinates
    cam_names: List[str]  # (N,)
    image_size: torch.Tensor  # (N, 2) in (H, W) format
    _preprocessing_transforms: Optional[List[Callable]] = None
    _is_prerotated: bool = False
    _prerotation: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None
    floor_plane: Optional[CalibrationPlane] = None

    FLOOR_PLANE_CACHE_NAME = 'floor_plane.json'

    def __getitem__(self, item) -> 'CalibrationData':
        # select index in transform arguments
        new_transforms = copy.deepcopy(self._preprocessing_transforms)
        for ti, t in enumerate(new_transforms or []):
            if isinstance(t, partial) and t.keywords is not None:
                new_keywords = copy.deepcopy(t.keywords)
                for k in t.keywords:
                    if isinstance(t.keywords[k], (np.ndarray, torch.Tensor)) and t.keywords[k].shape[0] == self.intrinsics.shape[0]:
                        new_keywords[k] = t.keywords[k][item]
                new_transforms[ti] = partial(t.func, *t.args, **new_keywords)
        return CalibrationData(
            intrinsics=self.intrinsics[item],
            extrinsics_c2w=self.extrinsics_c2w[item],
            dists=self.dists[item],
            rotmats=self.rotmats[item],
            tvecs=self.tvecs[item],
            cam_names=self.cam_names[item] if isinstance(item, int) else [self.cam_names[i] for i in item],
            image_size=self.image_size[item],
            _preprocessing_transforms=new_transforms,
            _is_prerotated=self._is_prerotated,
            _prerotation=self._prerotation,
            floor_plane=self.floor_plane,  # world-space: unaffected by camera subsetting
        )

    def create_cameras(self, mean_idx: List[int] | Literal['all'] = 'all', device: torch.device | str = 'cpu', lib: str = 'pytorch3d') -> Tuple[Any, torch.Tensor, torch.Tensor]:
        assert lib in ['open3d', 'pytorch3d'], f"Invalid library: {lib}. Choose 'pytorch3d' or 'kaolin'"
        from utils.vis import VisUtils
        if lib == 'open3d':
            cameras = VisUtils.create_cameras_o3d(
                intrinsics=self.intrinsics,
                extrinsics=self.extrinsics_c2w,
                image_size=tuple(self.image_size[0].cpu().tolist()),
                is_c2w=True,
            )
        else:
            cameras = VisUtils.create_cameras_p3d(
                intrinsics=self.intrinsics,
                rotmats=self.rotmats,
                tvecs=self.tvecs,
                image_size=tuple(self.image_size[0].cpu().tolist()),
                is_c2w=True,
            ).to(device=device)
        mean_look_at_point, mean_up_vector = VisUtils.compute_mean_look_at_and_up(
            tvecs=self.tvecs[mean_idx] if mean_idx != 'all' else self.tvecs,
            rotmats=self.rotmats[mean_idx] if mean_idx != 'all' else self.rotmats,
        )
        return cameras, mean_look_at_point, mean_up_vector

    def create_camera_plane(self, plane_idx: List[int] | Literal['all'] = 'all', device: torch.device | str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fits a plane to the camera translation vectors using SVD.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            plane_normal (3,) and offset d such that n.dot(p) + d == 0 on the plane.
        """
        tvecs = self.tvecs.clone().to(device=device)
        if plane_idx != 'all':
            # noinspection PyTypeChecker
            tvecs = tvecs[plane_idx]
        dev, dtype = tvecs.device, tvecs.dtype
        if tvecs.shape[0] < 3:
            return torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype), torch.tensor(0.0, device=dev, dtype=dtype)

        centroid = torch.mean(tvecs, dim=0)
        try:
            # rows of Vh are principal directions (high-to-low variance) -> last one is the normal
            _, _, Vh = torch.linalg.svd(tvecs - centroid, full_matrices=False)
            plane_normal = Vh[-1, :]
        except torch.linalg.LinAlgError:  # e.g. all points identical
            plane_normal = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype)
        if torch.linalg.norm(plane_normal) < 1e-6:
            plane_normal = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype)
        return plane_normal, -torch.dot(plane_normal, centroid)

    @functools.cached_property
    def extrinsics_w2c(self) -> torch.Tensor:
        """Returns the extrinsics in world-to-camera format."""
        return self.extrinsics_c2w.inverse()

    def get_preprocessing_transforms(self) -> List[Callable]:
        """
        Returns the preprocessing transforms if they exist, otherwise an empty list.

        >>> calibration_data = CalibrationData(...)
        >>> for transform in calibration_data.get_preprocessing_transforms():
        >>>     image, mask = transform(image, mask, cam_idx_s0_all=0)  # cam idx wrt all cam indices
        """
        return self._preprocessing_transforms or []

    def resize(self, *new_size_hw: int, apply_intrinsics_fix: bool = False) -> 'CalibrationData':
        """
        Returns a new CalibrationData object with resized intrinsics and image_size.
        Extrinsics are not resized, as they are assumed to be in world coordinates.

        Parameters
        ----------
        new_size_hw : int
            The new height and width. If a single value is given it is used for both.
        apply_intrinsics_fix : bool, optional
            If True, brings the intrinsics to the GPS training intrinsics. Default is False.
        """
        if len(new_size_hw) == 1:
            new_size_hw = new_size_hw[0], new_size_hw[0]
        elif len(new_size_hw) != 2:
            raise ValueError(f"Invalid new size: {new_size_hw}. Expected a single integer or a tuple of two integers (height, width).")

        # resize
        src_h, src_w = int(self.image_size[0][0]), int(self.image_size[0][1])
        tgt_h, tgt_w = new_size_hw
        scale = max(tgt_h / src_h, tgt_w / src_w)
        new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))

        # center crop
        top = (new_h - tgt_h) // 2
        left = (new_w - tgt_w) // 2

        # adjust intrinsics
        intrinsic_new = self.intrinsics.clone()
        intrinsic_new[..., 0, 0] *= scale
        intrinsic_new[..., 1, 1] *= scale
        intrinsic_new[..., 0, 2] = intrinsic_new[..., 0, 2] * scale - left
        intrinsic_new[..., 1, 2] = intrinsic_new[..., 1, 2] * scale - top

        # adjust image size
        image_size_new = self.image_size.clone()
        image_size_new[:, 0] = tgt_h
        image_size_new[:, 1] = tgt_w
        resize_transform = partial(CalibrationData.resize_and_center_crop_transform, src_image_size_hw=(src_h, src_w), target_image_size_hw=(tgt_h, tgt_w))
        transforms = (self._preprocessing_transforms or []) + [resize_transform]

        if apply_intrinsics_fix:
            all_remaps_x, all_remaps_y = [], []
            authors_h, authors_w = 1024, 1024
            authors_fx, authors_fy = authors_w * 0.6, authors_h * 0.6
            authors_cx, authors_cy = authors_w // 2, authors_h // 2
            author_u, author_v = np.meshgrid(np.arange(authors_w), np.arange(authors_h))  # target pixel grid
            for intri in intrinsic_new:
                calib_fx, calib_fy, calib_cx, calib_cy = intri[0, 0].item(), intri[1, 1].item(), intri[0, 2].item(), intri[1, 2].item()
                all_remaps_x.append((calib_fx / authors_fx * (author_u - authors_cx) + calib_cx).astype(np.float32))
                all_remaps_y.append((calib_fy / authors_fy * (author_v - authors_cy) + calib_cy).astype(np.float32))
            transforms.append(partial(CalibrationData.remap_transform, remaps_x=np.stack(all_remaps_x, axis=0), remaps_y=np.stack(all_remaps_y, axis=0)))
            intrinsic_new[:, 0, 0] = authors_fx
            intrinsic_new[:, 1, 1] = authors_fy
            intrinsic_new[:, 0, 2] = authors_cx
            intrinsic_new[:, 1, 2] = authors_cy
            image_size_new[:, 0] = authors_h
            image_size_new[:, 1] = authors_w

        return CalibrationData(
            intrinsics=intrinsic_new,
            extrinsics_c2w=self.extrinsics_c2w.clone(),
            dists=self.dists.clone(),
            rotmats=self.rotmats.clone(),
            tvecs=self.tvecs.clone(),
            cam_names=self.cam_names.copy(),
            image_size=image_size_new,
            _preprocessing_transforms=transforms,
            _is_prerotated=self._is_prerotated,
            _prerotation=self._prerotation,
            floor_plane=self.floor_plane,
        )

    def rotate(self, rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None) -> 'CalibrationData':
        """
        Returns a new CalibrationData object with rotated intrinsics and image_size.
        Extrinsics are rotated in the camera frame only, so world coordinates are unchanged.
        """
        if rotate is None or self._is_prerotated:
            return self

        K = self.intrinsics.clone()  # (N,3,3)
        Tw2c = self.extrinsics_w2c.clone()  # (N,4,4)
        image_size = self.image_size.clone()  # (N,2) [H,W]
        device, dtype, N = K.device, K.dtype, K.shape[0]

        def _rz_torch(angle_deg: float) -> torch.Tensor:
            a = torch.tensor(angle_deg * math.pi / 180.0, dtype=dtype, device=device)
            c, s = torch.cos(a), torch.sin(a)
            R = torch.eye(3, dtype=dtype, device=device)
            R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
            return R

        def _Himg_torch(W: torch.Tensor, H: torch.Tensor, which: str) -> torch.Tensor:
            """
            Pixel-space homography p' = Himg p using 0-based integer pixel centers.
            (switch offW/offH to -0.5 if your pipeline is half-pixel-centered)
            """
            offW, offH = W - 1.0, H - 1.0
            Himg = torch.eye(3, dtype=dtype, device=device)
            if which == '90_CLOCKWISE':  # u' = -v + (H-1), v' = u
                Himg[0, 0], Himg[0, 1], Himg[0, 2] = 0.0, -1.0, offH
                Himg[1, 0], Himg[1, 1], Himg[1, 2] = 1.0, 0.0, 0.0
            elif which == '90_COUNTERCLOCKWISE':  # u' = v, v' = -u + (W-1)
                Himg[0, 0], Himg[0, 1], Himg[0, 2] = 0.0, 1.0, 0.0
                Himg[1, 0], Himg[1, 1], Himg[1, 2] = -1.0, 0.0, offW
            elif which == '180':  # u' = -u + (W-1), v' = -v + (H-1)
                Himg[0, 0], Himg[0, 1], Himg[0, 2] = -1.0, 0.0, offW
                Himg[1, 0], Himg[1, 1], Himg[1, 2] = 0.0, -1.0, offH
            else:
                raise ValueError("rotate must be '90_COUNTERCLOCKWISE'|'90_CLOCKWISE'|'180'")
            return Himg

        # this transform rotates the image / mask / depth during loading
        transforms = [partial(CalibrationData.rotate_transform, rotate=rotate)] + (self._preprocessing_transforms or [])

        # camera-frame Z rotation (keeps fx,fy positive for typical K)
        Rz = _rz_torch({'90_CLOCKWISE': +90.0, '90_COUNTERCLOCKWISE': -90.0, '180': 180.0}[rotate])
        Mz = torch.eye(4, dtype=dtype, device=device)
        Mz[:3, :3] = Rz

        K_new, Tw2c_new, image_size_new = torch.empty_like(K), torch.empty_like(Tw2c), image_size.clone()
        for i in range(N):
            Hi, Wi = image_size[i, 0].to(dtype=dtype), image_size[i, 1].to(dtype=dtype)
            K_new[i] = _Himg_torch(Wi, Hi, rotate) @ K[i] @ Rz.transpose(0, 1)  # K' = Himg K Rz^T
            Tw2c_new[i] = Mz @ Tw2c[i]  # T'w2c = blkdiag(Rz,1) Tw2c
            if rotate in ('90_CLOCKWISE', '90_COUNTERCLOCKWISE'):
                image_size_new[i, 0], image_size_new[i, 1] = image_size[i, 1], image_size[i, 0]

        Tc2w_new = torch.linalg.inv(Tw2c_new)
        return CalibrationData(
            intrinsics=K_new,
            extrinsics_c2w=Tc2w_new,
            dists=self.dists.clone(),
            rotmats=Tc2w_new[:, :3, :3],
            tvecs=Tc2w_new[:, :3, 3],
            cam_names=self.cam_names.copy(),
            image_size=image_size_new,
            _preprocessing_transforms=transforms,
            _is_prerotated=self._is_prerotated,
            _prerotation=self._prerotation,
            floor_plane=self.floor_plane,
        )

    def plot_camera_centers(self) -> None:
        """Plots camera centers extracted from the extrinsic matrices."""
        extrinsics = self.extrinsics_c2w.cpu().numpy()
        assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4)
        centers_np = extrinsics[:, :3, 3]

        fig = plt.figure()
        # noinspection PyTypeChecker
        ax: Axes3D = fig.add_subplot(111, projection='3d')
        ax.scatter(centers_np[:, 0], centers_np[:, 1], centers_np[:, 2], c='blue', marker='o')
        for i, c in enumerate(centers_np.tolist()):
            ax.text(c[0], c[1], c[2], f"{i}", fontsize=10)
        ax.set_title("Camera Centers")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()

    def export_to_unity(self, json_path: Path) -> None:
        with open(json_path, 'w') as f:
            unity_c2w = self.extrinsics_c2w @ torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0]))[None]
            transs = unity_c2w[:, :3, 3]
            rots = kornia.geometry.rotation_matrix_to_quaternion(unity_c2w[:, :3, :3])[..., (1, 2, 3, 0)].numpy()
            json_data = []
            for cam_name, trans, rot, intri, (h, w) in zip(self.cam_names, transs, rots, self.intrinsics, self.image_size):
                cam_name = cam_name.split('/')[-1]
                json_data.append({
                    "id": cam_name,
                    "index": str(cam_name).split(' ')[0].lower().replace('cam', ''),
                    'position': trans.flatten().tolist(),
                    'rotation': rot.flatten().tolist(),
                    "intrinsics": {
                        "fx": intri[0][0].item(),
                        "fy": intri[1][1].item(),
                        "cx": intri[0][2].item(),
                        "cy": intri[1][2].item(),
                        "w": w.int().item(),
                        "h": h.int().item()
                    }
                })
            json.dump(dict(cameras=json_data), f, indent=2)

    def export_to_holomit(self, json_path: Path) -> None:
        with open(json_path, 'w') as f:
            unity_c2w = self.extrinsics_c2w @ torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0]))[None]
            calibration_data, processing_data = {}, {}
            for cam_name, c2w, intri, dist, (h, w) in zip(self.cam_names, unity_c2w.squeeze().detach().cpu(), self.intrinsics.detach().cpu(), self.dists.detach().cpu(), self.image_size.detach().cpu()):
                dist = dist.flatten().tolist()
                cam_name = cam_name.split('/')[-1]
                cam_idx = str(cam_name).split(' ')[0].lower().replace('cam', '')
                cam_key = f'SN{int(cam_idx):010d}'
                calibration_data[cam_key] = {
                    "trafo": c2w.tolist(),
                    "dScale": 0.001,
                    "color_shape": {
                        "height": int(h),
                        "width": int(w),
                        "numChannels": 1,
                        "channelSize": 2
                    },
                    "color_intrinsics": {
                        "fx": intri[0][0].item(),
                        "fy": intri[1][1].item(),
                        "cx": intri[0][2].item(),
                        "cy": intri[1][2].item(),
                    },
                    "color_distortion": {
                        "k1": dist[0] if len(dist) > 1 else 0.0,
                        "k2": dist[1] if len(dist) > 1 else 0.0,
                        "k3": dist[4] if len(dist) > 1 else 0.0,
                        "k4": 0.0,
                        "k5": 0.0,
                        "k6": 0.0,
                        "p1": dist[2] if len(dist) > 1 else 0.0,
                        "p2": dist[3] if len(dist) > 1 else 0.0,
                        "codx": 0.0,
                        "cody": 0.0,
                        "metric_radius": 0.0
                    }
                }
                processing_data[cam_key] = {"threshold_near": 0.5, "threshold_far": 5.0, "mask_color": False}
            processing_data["general"] = {"bounding_box": {"xMin": 0.0, "xMax": 0.0, "yMin": 0.0, "yMax": 0.0, "zMin": 0.0, "zMax": 0.0}}
            json.dump(dict(version=2, fps=30, ts_unit="ms", calibration=calibration_data, processing=processing_data), f, indent=2)

    # ==================================================================================
    # floor plane
    # ==================================================================================
    def _try_estimate_floor_plane(
            self,
            session_path: Path,
            max_views: int = 4,
            use_cache: bool = True,
            force: bool = False,
            keep_debug: bool = True,
            loader_kwargs: Optional[Dict[str, Any]] = None,
            estimator_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[CalibrationPlane]:
        """
        Estimates the world-space floor plane from a few RGB-D frames of the session and stores
        it in ``self.floor_plane``. Never raises -- on failure it logs and leaves it as None.

        ``idx`` is accepted for API symmetry with ``from_session``; ``self`` is already restricted
        to those cameras, so the views are picked from ``self.cam_names``. The result is cached
        at ``<session>/__calib__/floor_plane.json`` (the cache holds no debug overlays).
        """
        session_path = Path(session_path).expanduser()
        cache_path = session_path / '__calib__' / self.FLOOR_PLANE_CACHE_NAME
        if use_cache and not force and cache_path.is_file():
            try:
                self.floor_plane = CalibrationPlane.from_dict(json.loads(cache_path.read_text()))
                log(f'[CalibrationData::_try_estimate_floor_plane] Loaded cached floor plane from {cache_path}', 'debug')
                return self.floor_plane
            except Exception as e:
                log(f'[CalibrationData::_try_estimate_floor_plane] Ignoring unreadable cache {cache_path}: {e}', 'warning')

        try:
            t0 = time.time()
            views = CalibrationPlane.load_session_views(
                session_path,
                cam_names=self.cam_names,
                intrinsics=self.intrinsics,
                extrinsics_c2w=self.extrinsics_c2w,
                image_size=self.image_size,
                max_views=max_views,
                prerotation=self._prerotation if self._is_prerotated else None, **(loader_kwargs or {})
            )
            if not views:
                log('[CalibrationData::_try_estimate_floor_plane] No usable RGB-D frames in this session, skipping', 'warning')
                return None

            plane = CalibrationPlane.estimate(views, keep_debug=keep_debug, **(estimator_kwargs or {}))
            if plane is None:
                log('[CalibrationData::_try_estimate_floor_plane] Floor plane estimation returned nothing', 'warning')
                return None
            plane.metadata = dict(plane.metadata or {}, session=str(session_path), seconds=round(time.time() - t0, 2))
            self.floor_plane = plane

            if use_cache:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(plane.to_dict(), indent=2))
                    log(f'[CalibrationData::_try_estimate_floor_plane] Cached floor plane at {cache_path}', 'debug')
                except Exception as e:
                    log(f'[CalibrationData::_try_estimate_floor_plane] Could not write cache: {e}', 'warning')
            return plane
        except Exception as e:
            log(f'[CalibrationData::_try_estimate_floor_plane] Failed: {type(e).__name__}: {e}', 'warning', exc_info=e)
            return None

    def visualize_floor_probabilities(self, out_path: Union[str, Path], session_path: Optional[Path] = None, **kwargs) -> bool:
        """
        (1) Input views with the per-pixel floor probability on top. Needs the estimator debug
        data; if it is missing (e.g. the plane came from the cache) pass ``session_path`` and it
        is re-estimated.
        """
        if (self.floor_plane is None or self.floor_plane.debug is None) and session_path is not None:
            self._try_estimate_floor_plane(session_path, force=True, keep_debug=True)
        if self.floor_plane is None:
            log('[CalibrationData::visualize_floor_probabilities] No floor plane estimated', 'warning')
            return False
        return self.floor_plane.visualize_probabilities(out_path, **kwargs)

    def visualize_floor_plane_3d(self, out_path: Optional[Union[str, Path]] = None, show: bool = False, **kwargs):
        """(2) Mesh with the cameras and the inferred floor plane."""
        if self.floor_plane is None:
            log('[CalibrationData::visualize_floor_plane_3d] No floor plane estimated', 'warning')
            return None
        return self.floor_plane.build_scene_mesh(
            intrinsics=self.intrinsics, extrinsics_c2w=self.extrinsics_c2w, image_size=self.image_size,
            out_path=out_path, show=show, **kwargs)

    # ==================================================================================
    # image transforms (applied at load time)
    # ==================================================================================
    @staticmethod
    def remap_transform(image: np.ndarray, mask: np.ndarray, depth: Optional[np.ndarray] = None, remaps_x: np.ndarray = None, remaps_y: np.ndarray = None, cam_idx_s0: int = -1) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Warps image/mask/depth with the provided remap_x, remap_y coordinates."""
        assert remaps_x is not None and remaps_y is not None, "remaps_x and remaps_y must be provided for remapping."
        assert cam_idx_s0 >= 0, f"Invalid camera index: {cam_idx_s0}. Must be non-negative."
        warped_image = cv2.remap(image, remaps_x[cam_idx_s0], remaps_y[cam_idx_s0], interpolation=cv2.INTER_LINEAR)
        warped_mask = cv2.remap(mask.astype(np.float32), remaps_x[cam_idx_s0], remaps_y[cam_idx_s0], interpolation=cv2.INTER_NEAREST).astype(mask.dtype)
        if depth is None:
            return warped_image, warped_mask
        warped_depth = cv2.remap(depth.astype(np.float32), remaps_x[cam_idx_s0], remaps_y[cam_idx_s0], interpolation=cv2.INTER_NEAREST)
        return warped_image, warped_mask, warped_depth

    @staticmethod
    def resize_and_center_crop_transform(image: np.ndarray, mask: np.ndarray, depth: Optional[np.ndarray] = None,
                                         src_image_size_hw: Tuple[int, int] = (-1, -1),
                                         target_image_size_hw: Tuple[int, int] = (-1, -1),
                                         **ignored_kwargs) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        src_h, src_w = src_image_size_hw
        tgt_h, tgt_w = target_image_size_hw
        assert src_h > 0 and src_w > 0, f"Invalid source image size: {src_image_size_hw}. Must be positive integers."
        assert tgt_h > 0 and tgt_w > 0, f"Invalid target image size: {target_image_size_hw}. Must be positive integers."
        # resize
        scale = max(tgt_h / src_h, tgt_w / src_w)
        new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))
        image_rs = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth_rs = None if depth is None else cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # center crop
        top, left = (new_h - tgt_h) // 2, (new_w - tgt_w) // 2
        bot, right = top + tgt_h, left + tgt_w
        img_cp, mask_cp = image_rs[top:bot, left:right, :], mask_rs[top:bot, left:right]
        if depth_rs is None:
            return img_cp, mask_cp
        return img_cp, mask_cp, depth_rs[top:bot, left:right]

    @staticmethod
    def rotate_transform(image: np.ndarray, mask: np.ndarray, depth: Optional[np.ndarray] = None,
                         rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None,
                         **ignored_kwargs) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if rotate:
            rot_flag = getattr(cv2, f'ROTATE_{rotate.upper()}')
            image = cv2.rotate(image, rot_flag)
            mask = cv2.rotate(mask.astype(np.uint8), rot_flag).astype(mask.dtype)
            depth = None if depth is None else cv2.rotate(depth, rot_flag)
        return image, mask, depth

    # ==================================================================================
    # constructors
    # ==================================================================================
    @classmethod
    def from_pkl(cls, data: List[Dict[str, Union[list, float, int]]]) -> 'CalibrationData':
        calibration_data = dict(
            intrinsics=torch.stack([torch.tensor(d['intrinsic']) for d in data]).float(),
            extrinsics=torch.stack([torch.tensor(d['extrinsic']) for d in data]).float(),
            image_size=torch.stack([torch.tensor([d['height'], d['width']]) for d in data]).int(),
        )
        num_cameras = len(calibration_data['intrinsics'])
        calibration_data['dists'] = torch.zeros((num_cameras, 5), dtype=torch.float32)
        calibration_data['rotmats'] = calibration_data['extrinsics'][:, :3, :3]
        calibration_data['tvecs'] = calibration_data['extrinsics'][:, :3, 3]
        calibration_data['cam_names'] = [f'cam{c + 1:02d}' for c in range(num_cameras)]
        return cls(**calibration_data)

    @classmethod
    def from_session(cls, session_path: Path, idx: Literal['all'] | List[int] = 'all', estimate_floor: bool = True) -> Union['CalibrationData', None]:
        def ensure_rotmat_is_orthonormal(rotmat: torch.Tensor) -> torch.Tensor:
            U, _, V = torch.linalg.svd(rotmat)
            return U.T @ V

        def ensure_rotmat_is_valid(rotmat: torch.Tensor) -> torch.Tensor:
            rotmat = rotmat.clone()
            for i in range(rotmat.shape[0]):
                if torch.linalg.det(rotmat[i]) < 0:
                    rotmat[i, 0] *= -1
                if (rotmat[i].T @ rotmat[i] - torch.eye(rotmat.shape[-1])).abs().max() > 1e-6:
                    rotmat[i] = ensure_rotmat_is_orthonormal(rotmat[i])
            return rotmat

        session_path = Path(session_path).expanduser()
        if not session_path.exists():
            for d in PathUtils.capturestudio_cache_path().glob('Captures_*'):
                if d.is_dir() and (d / session_path.name).exists():
                    session_path = d / session_path.name
                    break
        assert session_path.exists(), f"Session path does not exist: {session_path}"

        log(f"Reading calibration from capture session: {session_path}", 'info')
        calibration_dir = session_path / '__calib__'
        if not calibration_dir.exists() or (not (calibration_dir / 'caliscope').exists() and not (calibration_dir / 'multicamcalib').exists()):
            log(f"Calibration directory does not exist: {calibration_dir}", 'error')
            return None
        # read calibration data
        module = importlib.import_module('utils.calib')
        method = 'Caliscope' if (calibration_dir / 'caliscope').exists() else 'MultiCamCalib'
        calib_reader = getattr(module, f'{method}Reader')(session_path)
        calib_data = calib_reader.read()
        if idx == 'all':
            idx = list(range(len(calib_data)))
        values = [v for vi, v in enumerate(calib_data.values()) if vi in idx]

        extri_4x4 = torch.eye(4, dtype=torch.float32)[None].repeat(len(idx), 1, 1)
        extri_4x4[:, :3, :] = torch.stack([
            torch.tensor(v['extri'] if v.get('extri') is not None else np.eye(4)[:3], dtype=torch.float32) for v in values
        ], dim=0)
        extri_4x4[:, :3, :3] = ensure_rotmat_is_valid(extri_4x4[:, :3, :3])
        extri_4x4 = extri_4x4.inverse()  # w2c (viewmatrix) --> c2w (calibration matrix)
        extri_4x4[:, :3, :3] = ensure_rotmat_is_valid(extri_4x4[:, :3, :3])

        cd = CalibrationData(
            intrinsics=torch.stack([torch.tensor(v['intri']['K'], dtype=torch.float32) for v in values], dim=0).reshape(-1, 3, 3),
            dists=torch.stack([torch.tensor(v['intri']['dist'], dtype=torch.float32) for v in values], dim=0),
            rotmats=extri_4x4[:, :3, :3],
            tvecs=extri_4x4[:, :3, 3],
            extrinsics_c2w=extri_4x4,
            cam_names=[v['cam_name'] for v in values],
            image_size=torch.stack([torch.tensor(v['image_size'], dtype=torch.int) for v in values], dim=0),
            _is_prerotated=getattr(calib_reader, 'is_prerotated', False),
            _prerotation=getattr(calib_reader, 'prerotation', None),
        )
        if cd._is_prerotated and cd._prerotation is not None:
            log(f'[CalibrationData::from_session] Found prerotated calibration (rotate={cd._prerotation})', 'debug')
            cd._preprocessing_transforms = [partial(CalibrationData.rotate_transform, rotate=cd._prerotation)]
        if estimate_floor:
            cd._try_estimate_floor_plane(session_path)
        return cd

    @classmethod
    def from_session_folder(cls, session_folder: Path, idx: Literal['all'] | List[int] = 'all') -> Union['CalibrationData', None]:
        """Reads calibration data from a session folder (per-camera parameters/*.npy)."""
        session_folder = Path(session_folder).expanduser()
        if not session_folder.exists():
            log(f"Session folder does not exist: {session_folder}", 'error')
            return None
        cam_names, intrinsic_files, extrinsic_w2c_files = [], [], []
        image_size_hw = None
        for cam in sorted([_ for _ in (session_folder / 'orbbec').iterdir() if _.is_dir() and _.name.startswith('cam')], key=lambda x: int(x.name.replace('cam', ''))):
            cam_names.append(f'orbbec/{cam.name}')
            intrinsic_files.append(cam / 'parameters' / 'intrinsic.npy')
            extrinsic_w2c_files.append(cam / 'parameters' / 'extrinsic.npy')
            if image_size_hw is None:
                image_size_hw = cv2.imread(str(next((cam / 'color').glob('*.jpg')))).shape[:2]  # (H, W)
        assert all(p.exists() for p in intrinsic_files) and all(p.exists() for p in extrinsic_w2c_files)
        intrinsics = torch.stack([torch.tensor(np.load(str(p)), dtype=torch.float32) for p in intrinsic_files], dim=0).float()
        extri_4x4 = torch.stack([torch.tensor(np.load(str(p)), dtype=torch.float32) for p in extrinsic_w2c_files], dim=0).float()
        if idx == 'all':
            idx = list(range(len(cam_names)))
        return CalibrationData(
            intrinsics=intrinsics.reshape(-1, 3, 3)[idx],
            dists=torch.zeros((len(idx), 5), dtype=torch.float32),
            rotmats=extri_4x4[idx, :3, :3],
            tvecs=extri_4x4[idx, :3, 3],
            extrinsics_c2w=extri_4x4[idx],
            cam_names=[c for i, c in enumerate(cam_names) if i in idx],
            image_size=torch.tensor(image_size_hw, dtype=torch.int)[None].repeat(len(idx), 1),
        )


class ColmapReader:
    def __init__(self, capture_path: Path, experiment_name: str = 'for_sp_nm1'):
        self.capture_path = capture_path
        self.experiment_name = experiment_name
        self.colmap_path = capture_path / 'colmap' / experiment_name
        self.colmap_out_path = self.colmap_path / 'colmap_out'
        self.colmap_recon_path = sorted(self.colmap_out_path.iterdir(), key=lambda x: int(x.name))[-1]

    def read(self):
        log(f'[{self.__class__.__name__}::read] Reading COLMAP data (run={self.experiment_name}, path={self.colmap_recon_path})', 'debug')
        with open(self.colmap_recon_path / 'cameras.pkl', 'rb') as f:
            return pickle.load(f)


class CaliscopeReader:
    def __init__(self, capture_path: Path, override_cam_mapping: Optional[str] = None, which_calib_method: str = 'caliscope'):  # 'cam_7-->cam_8, cam_8-->cam_7'
        self.capture_path = capture_path
        if (capture_path / 'raw_color').exists():
            self.camera_names = sorted([d.name for d in (capture_path / 'raw_color').iterdir() if d.is_dir()], key=lambda x: int(x.split(' ')[0]))
        else:
            # new capture structure (v2)
            orbbec_path = capture_path / 'orbbec'
            self.camera_names = sorted([f'orbbec/{d.name}' for d in (orbbec_path.glob('cam*') if len(list(orbbec_path.glob('cam*'))) > 0 else (orbbec_path / 'raw_color').glob('cam*')) if d.is_dir()], key=lambda x: int(x.replace('orbbec/cam', '')))
            if (capture_path / 'sony').exists():
                self.camera_names.append('sony')
            if (capture_path / 'apple').exists():
                self.camera_names.append('apple')
        self.camera_indices_s0 = [int(_.split(' ')[0].replace('orbbec/cam', '').replace('sony', '13').replace('apple', '14')) - 1 for _ in self.camera_names]
        if (capture_path / '__calib__').exists() and (capture_path / '__calib__' / which_calib_method.lower()).exists():
            self.calib_root = capture_path / '__calib__' / which_calib_method.lower()
        elif (capture_path / 'caliscope').exists():
            self.calib_root = capture_path / which_calib_method.lower()
        else:
            raise FileNotFoundError(f'Calibration path not found in {capture_path} (tried {capture_path / "__calib__" / which_calib_method} and {capture_path / which_calib_method})')
        self.override_cam_mapping = dict()
        if override_cam_mapping is not None:
            for pair in override_cam_mapping.split(','):
                pair = pair.strip().split('-->')
                self.override_cam_mapping[pair[0].strip()] = pair[1].strip()
        self.is_prerotated = False
        self.prerotation = None

    def read(self, only_indices: Optional[List[int]] = None):
        log(f'[{self.__class__.__name__}::read] Reading Caliscope data (path={self.calib_root})', 'debug')
        file = (self.calib_root / 'camera_array.toml') if (self.calib_root / 'camera_array.toml').is_file() else (self.calib_root / 'config.toml')
        with open(file, 'r') as f:
            caliscope_data = toml.load(f)

        self.is_prerotated = caliscope_data.get('prerotated', False)
        self.prerotation = caliscope_data.get('prerotation', None)
        if 'cameras' in caliscope_data:
            caliscope_data = caliscope_data['cameras']
        assert caliscope_data is not None and caliscope_data.get('camera_count', len(caliscope_data)) <= len(self.camera_names)

        cam_dict = {}
        for name, data in caliscope_data.items():
            # Thanos: skip non-camera entries
            if name in ['camera_count', 'capture_id', 'capture_date', 'creation_date']:
                continue
            if name.isdigit():
                name = f'cam_{int(name):02d}'
            try:
                cam_index_s0 = int(self.override_cam_mapping.get(name, name).replace("cam_", "")) - 1
            except ValueError:
                continue
            if cam_index_s0 not in self.camera_indices_s0 or (only_indices is not None and (cam_index_s0 + 1) not in only_indices):
                log(f'[{self.__class__.__name__}::read] {name} --> {cam_index_s0} not in camera_indices_s0 or not in only_indices', 'warning')
                continue
            camera_name = self.camera_names[self.camera_indices_s0.index(cam_index_s0)]
            if name in self.override_cam_mapping and self.override_cam_mapping[name] != name:
                log(f'[{self.__class__.__name__}::read] {name} --> {cam_index_s0} --> {camera_name}', 'warning')
            cam_dict[camera_name] = dict(
                cam_index_s0=cam_index_s0,
                cam_index=cam_index_s0 + 1,
                cam_name=camera_name,
                cam_model='SIMPLE_RADIAL',
                caliscope_key=name,
                intri=dict(K=np.array(data['matrix']), dist=np.array(data['distortions']).reshape(1, 5), H=data['size'][1], W=data['size'][0]),
                rotation=cv2.Rodrigues(np.array(data['rotation']).reshape(3, 1))[0] if isinstance(data['rotation'], list) else data['rotation'],
                translation=np.asarray(data['translation']).flatten() if isinstance(data['translation'], list) else data['translation'],
                extri=None if (data['rotation'] == 'null' or data['translation'] == 'null') else np.hstack([
                    data['rotation'] if isinstance(data['rotation'], np.ndarray) else cv2.Rodrigues(np.array(data['rotation']).reshape(3, 1))[0],
                    np.atleast_2d(data['translation']).reshape((-1, 1))
                ]),
                image_size=(data['size'][1], data['size'][0]),  # H, W
            )
        return dict(sorted(cam_dict.items(), key=lambda item: item[1]['cam_index_s0']))

    def create_files(self, cam_dict):
        file = (self.calib_root / 'camera_array.toml') if (self.calib_root / 'camera_array.toml').is_file() else (self.calib_root / 'config.toml')
        with open(file, 'r') as f:
            caliscope_data = toml.load(f)
        assert caliscope_data is not None and caliscope_data['camera_count'] == len(self.camera_names)
        for name, data in caliscope_data.items():
            if not name.startswith('cam_'):
                cam_dict[name] = data
                continue
            cam_index_s0 = int(self.override_cam_mapping.get(name, name).replace('cam_', '')) - 1
            camera_name = self.camera_names[cam_index_s0]
            assert cam_dict[camera_name]['caliscope_key'] == name
            data['matrix'] = cam_dict[camera_name]['intri']['K'].tolist()
            data['distortions'] = cam_dict[camera_name]['intri']['dist'].flatten().tolist()
            data['translation'] = cam_dict[camera_name]['extri_translation'].flatten().tolist()
            data['rotation'] = cam_dict[camera_name]['extri_rotation'].tolist()
            data['extri_matrix'] = cam_dict[camera_name]['extri'].tolist()
        log(f'[{self.__class__.__name__}::create_files] Writing to {file}', 'debug')
        with open(file, 'w') as f:
            toml.dump(caliscope_data, f)
        with open(self.calib_root / 'cameras.pkl', 'wb') as f:
            pickle.dump(cam_dict, f)


class MultiCamCalibReader(CaliscopeReader):
    def __init__(self, capture_path: Path, override_cam_mapping: Optional[str] = None):
        super().__init__(capture_path, override_cam_mapping, which_calib_method='multicamcalib')

    def read(self, only_indices: Optional[List[int]] = None):
        log(f'[{self.__class__.__name__}::read] Reading MultiCamCalib data (path={self.calib_root})', 'debug')
        with open(self.calib_root / 'cam_params_final.json', 'r') as f:
            mcc_data = json.load(f)
        assert mcc_data is not None and len(mcc_data) <= len(self.camera_names)
        cam_dict = {}
        for name, data in mcc_data.items():
            cam_index_s0 = int(self.override_cam_mapping.get(name, name).replace('cam_', ''))
            if cam_index_s0 not in self.camera_indices_s0 or (only_indices is not None and (cam_index_s0 + 1) not in only_indices):
                log(f'[{self.__class__.__name__}::read] {name} --> {cam_index_s0} not in camera_indices_s0 or not in only_indices', 'warning')
                continue
            camera_name = self.camera_names[self.camera_indices_s0.index(cam_index_s0)]
            if name in self.override_cam_mapping and self.override_cam_mapping[name] != name:
                log(f'[{self.__class__.__name__}::read] {name} --> {cam_index_s0} --> {camera_name}', 'warning')
            extri = None if (data['rvec'] == 'null' or data['tvec'] == 'null') else np.hstack([
                data['rvec'] if isinstance(data['rvec'], np.ndarray) else cv2.Rodrigues(np.array(data['rvec']).reshape(3, 1))[0],
                np.atleast_2d(data['tvec']).reshape((-1, 1))
            ])
            if extri is not None:
                extri[:, 3] = extri[:, 3] / 1000  # mm --> m
                rot_y_90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])  # MulticamCalib output is rotated +90 deg around y
                extri[:3, :3] = extri[:3, :3] @ rot_y_90
                extri_4x4 = np.eye(4, dtype=extri.dtype)
                extri_4x4[:3, :4] = extri[:3, :4]
                extri = extri_4x4[:3, :4]
            cam_dict[camera_name] = dict(
                cam_index_s0=cam_index_s0,
                cam_index=cam_index_s0 + 1,
                cam_name=camera_name,
                cam_model='SIMPLE_RADIAL',
                multicamcalib_key=name,
                intri=dict(
                    K=np.array([[data['fx'], 0, data['cx']], [0, data['fy'], data['cy']], [0, 0, 1]]),
                    dist=np.array([data['k1'], data['k2'], data['p1'], data['p2'], data['k3']]).reshape(1, 5),
                    H=data['size'][1],
                    W=data['size'][0]
                ),
                rotation=extri[:3, :3] if extri is not None else None,
                translation=extri[:3, 3] if extri is not None else None,
                extri=extri,
                image_size=(data['size'][1], data['size'][0]),  # H, W TODO data['size'] = (3840, 2160) if 'size' not in data else data['size']
            )
        return dict(sorted(cam_dict.items(), key=lambda item: item[1]['cam_index_s0']))


if __name__ == '__main__':
    # session_ = 'Captures_Cagliari_Nov_2025/Cagliari_1_Calib_6'
    session_ = 'Captures_Cagliari_Jun_2026/Cagliari_2_5cams_Calib_2'
    capture_path_ = Path.home() / 'CAPTURESTUDIO_CACHE' / session_
    out_dir_ = Path.cwd()
    stem_ = session_.split('/')[1].lower()

    calib_data_ = CalibrationData.from_session(capture_path_)

    # ---- floor plane -------------------------------------------------------------
    if calib_data_.floor_plane is None or calib_data_.floor_plane.debug is None:
        # from_session may have served the plane from the json cache (no probability maps kept)
        calib_data_._try_estimate_floor_plane(capture_path_, max_views=4, force=True)

    if calib_data_.floor_plane is not None:
        print('floor normal :', calib_data_.floor_plane.normal.tolist())
        print('floor offset :', float(calib_data_.floor_plane.offset))
        print('cam heights  :', [round(h, 3) for h in calib_data_.floor_plane.signed_distance(calib_data_.tvecs).tolist()])

        # (1) floor probabilities on the images
        calib_data_.visualize_floor_probabilities(out_dir_ / f'{stem_}_floor_probs.png', session_path=capture_path_)
        # (2) mesh with the cameras + the inferred floor plane
        calib_data_.visualize_floor_plane_3d(out_dir_ / f'{stem_}_floor_scene.ply', show=False, frustum_depth=0.3, floor_margin=1.0)

    # calib_data_.export_to_unity(out_dir_ / f'{stem_}_unity.json')
    # calib_data_.export_to_holomit(out_dir_ / f'{stem_}_holomit.json')
