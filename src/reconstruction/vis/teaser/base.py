import abc
import dataclasses
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional, List, Literal, Union, Dict, Iterator, Type

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from plyfile import PlyData

from reconstruction.data.capturestudio import MultiSessionDataset
from reconstruction.primitive.pcd import RGBDImage
from reconstruction.primitive.splat import GSImage
from reconstruction.vis.cam_orbit import CameraOrbit
from reconstruction.vis.dataset_visualizer import DatasetVisualizer
from utils.misc import PathUtils, log


class CapturestudioVirtualBackgroundFloorWallEstimator:
    SEGMENTOR_MODEL = None
    SEGMENTOR_MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
    SEGMENTOR_MODEL_DEVICE = 'cuda'

    def __init__(self,
                 wall_overshoot_m: float = 2.0,
                 wall_height_m: float = 3.0,
                 wall_pad_width_m: float = 1.0,
                 wall_show_logo: bool = False,
                 floor_depth_scale: float = 1.5,
                 floor_min_valid_points_per_view: int = 25,
                 floor_max_depth_m: float = 5.0,
                 seed: int = 42):
        self.wall_overshoot_m = wall_overshoot_m
        self.wall_height_m = wall_height_m
        self.wall_pad_width_m = wall_pad_width_m
        self.wall_show_logo = wall_show_logo
        self.floor_depth_scale = floor_depth_scale
        self.min_valid_points_per_view = floor_min_valid_points_per_view
        self.max_depth_for_floor_m = floor_max_depth_m
        self._rng = np.random.default_rng(seed=seed)

    @staticmethod
    def basis_from_corners(corners):
        _normalize = lambda v: v / (np.linalg.norm(v) + 1e-12)
        _c0, _c1, _c2, _c3 = corners[:4]
        _t = _c1 - _c0  # width axis (left->right)
        _u = _c3 - _c0  # depth (floor) or vertical (wall)
        _width, _height = float(np.linalg.norm(_t)), float(np.linalg.norm(_u))
        _ex = _normalize(_t)
        _ey = _normalize(_u - _ex * float(np.dot(_ex, _u)))
        _ez = _normalize(np.cross(_ex, _ey))  # local +Z normal
        _ey = _normalize(np.cross(_ez, _ex))  # re-orthogonalize
        _R = np.stack([_ex, _ey, _ez], axis=1)  # columns
        _ctr = 0.25 * (_c0 + _c1 + _c2 + _c3)
        return _R, _width, _height, _ctr

    def _segment(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.__class__.SEGMENTOR_MODEL is None:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            processor = AutoImageProcessor.from_pretrained(self.__class__.SEGMENTOR_MODEL_NAME, use_fast=True)
            model = AutoModelForSemanticSegmentation.from_pretrained(self.__class__.SEGMENTOR_MODEL_NAME, dtype=torch.float16).to(self.__class__.SEGMENTOR_MODEL_DEVICE)
            self.__class__.SEGMENTOR_MODEL = (processor, model)
        processor, model = self.__class__.SEGMENTOR_MODEL
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        floor_ids = [k for k, v in id2label.items()
                     if any(t in v.lower() for t in ("floor", "ground", "road", "sidewalk", "pavement"))] or [0]
        rgb = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
        rgb_input = {k: v.to(model.device, dtype=model.dtype) for k, v in processor(images=rgb, return_tensors="pt").items()}
        with torch.inference_mode():
            logits = model(**rgb_input).logits
            logits = torch.nn.functional.interpolate(
                logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
            )[0]
            p = logits.softmax(dim=0)[floor_ids].sum(dim=0).float().detach().cpu().numpy()
        floor_prob_map = np.clip(p, 0, 1).astype(np.float32)
        floor_mask = floor_prob_map >= 0.5
        return floor_mask, floor_prob_map

    def _estimate_floor_plane(self, rgb: np.ndarray, depth: np.ndarray, points_w: np.ndarray) -> Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
        floor_mask, floor_prob = self._segment(rgb)
        depth_mask = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth_for_floor_m)
        points_w_mask = np.isfinite(points_w).all(-1)
        mask = floor_mask & depth_mask & points_w_mask
        # Weighted-PCA
        if np.count_nonzero(mask) >= self.min_valid_points_per_view:
            points_w_valid = points_w[mask].astype(np.float64)
            point_weights = floor_prob[mask].astype(np.float64) / (floor_prob[mask].astype(np.float64).sum() + 1e-12)
            mu = (point_weights[:, None] * points_w_valid).sum(0)
            _, _, Vt = np.linalg.svd((points_w_valid - mu) * np.sqrt(point_weights)[:, None], full_matrices=False)
            normal_world = Vt[-1]  # least variance direction --> plane normal
            offset_world = -normal_world @ mu
            return normal_world.astype(np.float64), float(offset_world), points_w_valid, point_weights, floor_mask
        return None

    def _estimate_floor_planes(self, views: List[RGBDImage]) -> List[Dict[str, Union[np.ndarray, float]]]:
        planes = []
        for view in views:
            plane_i_data = self._estimate_floor_plane(view.rgb, view.depth, view.points_world)
            if plane_i_data is not None:
                planes.append(dict(
                    normal_world=plane_i_data[0],
                    offset_world=plane_i_data[1],
                    floor_points_w=plane_i_data[2],
                    floor_point_weights=plane_i_data[3],
                    floor_mask=plane_i_data[4],
                ))
        return planes

    def _fit_floor_to_planes(self, view_planes: List[Dict[str, Union[np.ndarray, float]]]) -> Tuple[np.ndarray, float, Dict[str, Union[np.ndarray, float]]]:
        all_floor_points_w = np.concatenate([
            _['floor_points_w']
            for _ in view_planes
        ], axis=0).reshape(-1, 3)
        all_floor_point_weights = np.concatenate([
            _['floor_point_weights']
            for _ in view_planes
        ], axis=0).reshape(-1)
        all_floor_points_mask = np.isfinite(all_floor_points_w).all(-1)
        all_floor_points_w = all_floor_points_w[all_floor_points_mask]
        all_floor_point_weights = all_floor_point_weights[all_floor_points_mask]
        if len(all_floor_points_w) > 1_000_000:
            idx = self._rng.choice(len(all_floor_points_w), 1_000_000, replace=False, p=all_floor_point_weights / (all_floor_point_weights.sum() + 1e-12))
            all_floor_points_w, all_floor_point_weights = all_floor_points_w[idx], all_floor_point_weights[idx]

        # RANSAC
        scale = float(np.median(np.linalg.norm(all_floor_points_w, axis=1)))
        tau = max(0.005, 0.01 * scale)
        iters = 3000
        best_inl = None
        best_cnt = -1
        p_full = all_floor_point_weights / (all_floor_point_weights.sum() + 1e-12) if np.isfinite(all_floor_point_weights).all() and all_floor_point_weights.sum() > 0 else None
        for _ in range(iters):
            idx = self._rng.choice(len(all_floor_points_w), 3, replace=False, p=p_full)
            a, b, c = all_floor_points_w[idx]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-9: continue
            n /= nn
            d = -n @ a
            dist = np.abs(all_floor_points_w @ n + d)
            inl = dist <= tau
            cnt = int(inl.sum())
            if cnt > best_cnt:
                best_cnt = cnt
                best_inl = inl
        if best_inl is None:
            mu = (all_floor_point_weights[:, None] * all_floor_points_w).sum(0)
            X0 = all_floor_points_w - mu
            Xw = (np.sqrt(np.maximum(all_floor_point_weights, 1e-12))[:, None] * X0)
            _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
            n_final = Vt[-1]
            d_final = -n_final @ mu
            inliers = np.abs(all_floor_points_w @ n_final + d_final) <= tau
        else:
            Xr = all_floor_points_w[best_inl]
            wr = (all_floor_point_weights[best_inl] + 1e-6)
            wr = wr / (wr.sum() + 1e-12)
            mu = (wr[:, None] * Xr).sum(0)
            X0 = Xr - mu
            Xw = (np.sqrt(wr)[:, None] * X0)
            _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
            n_final = Vt[-1]
            d_final = -n_final @ mu
            inliers = best_inl
        floor_normal = n_final.astype(np.float64)
        floor_offset = float(d_final)

        # Identify floor "corners" using PCA
        all_floor_points_w_inliers = all_floor_points_w[inliers]
        if all_floor_points_w_inliers.shape[0] >= 3:
            mu_f = all_floor_points_w_inliers.mean(0)
            X0f = all_floor_points_w_inliers - mu_f
            Cf = (X0f.T @ X0f) / max(all_floor_points_w_inliers.shape[0] - 1, 1)
            _, evf = np.linalg.eigh(Cf)
            e1f = evf[:, 2]
            e1f = e1f - floor_normal * (e1f @ floor_normal)
            e1f /= (np.linalg.norm(e1f) + 1e-12)
            e2f = np.cross(floor_normal, e1f)
            e2f /= (np.linalg.norm(e2f) + 1e-12)
            U = X0f @ e1f
            V = X0f @ e2f
            umin0, umax0 = float(U.min()), float(U.max())
            vmin0, vmax0 = float(V.min()), float(V.max())
            pca_rect = np.stack([
                mu_f + umin0 * e1f + vmin0 * e2f,
                mu_f + umax0 * e1f + vmin0 * e2f,
                mu_f + umax0 * e1f + vmax0 * e2f,
                mu_f + umin0 * e1f + vmax0 * e2f,
            ], 0).astype(np.float64)
        else:
            pca_rect = None

        return floor_normal, floor_offset, dict(floor_corners_pca=pca_rect, floor_points_w_inliers=all_floor_points_w_inliers, floor_scale=scale)

    def _estimate_wall_plane(self, extrinsic_w2cs: List[np.ndarray], floor_normal: np.ndarray, floor_offset: float) -> Optional[Tuple[np.ndarray, float, np.ndarray, Dict[str, Union[float, Optional[np.ndarray]]]]]:
        if len(extrinsic_w2cs) < 2:
            raise NotImplementedError('Wall plane estimation with less than 2 cameras is currently not implemented.')

        c2ws = [np.linalg.inv(_).astype(np.float64) for _ in extrinsic_w2cs]
        Cw = np.stack([E[:3, 3] for E in c2ws], 0)

        # Compute mean lookat point of all cameras
        Rw = np.stack([E[:3, :3] for E in c2ws], 0)
        fwd = Rw[:, :, 2]
        fwd /= (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12)
        I = np.eye(3, dtype=np.float64)
        M = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros((3,), dtype=np.float64)
        for Cw_i, fwd_i in zip(Cw, fwd):
            fi = fwd_i
            A = I - np.outer(fi, fi)  # rank-1 projector to ray + remove component along ray direction = get direction perpendicular to ray
            M += A
            b += A @ Cw_i  # project to the plane perpendicular to the ray
        try:
            mean_lookat = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            print(f'[{self.__class__.__name__}::_estimate_wall_plane] Solving for mean_lookat point failed.]')
            return None

        # Compute arc chord
        arc_end0 = Cw[0]
        arc_end1 = Cw[-1]
        chord = arc_end1 - arc_end0
        clen = np.linalg.norm(chord)
        if clen < 1e-9:
            t_hat = np.cross(floor_normal, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(t_hat) < 1e-6:
                t_hat = np.cross(floor_normal, np.array([0.0, 1.0, 0.0]))
            t_hat /= (np.linalg.norm(t_hat) + 1e-12)
        else:
            t_hat = chord / clen

        # Compute wall up direction
        n_up = floor_normal / (np.linalg.norm(floor_normal) + 1e-12)
        s_look = float(np.dot(n_up, mean_lookat) + floor_offset)
        if s_look < 0.0:
            n_up = -n_up

        chord_mid = 0.5 * (arc_end0 + arc_end1)
        v_to_look = mean_lookat - chord_mid
        v_norm = np.linalg.norm(v_to_look)
        if v_norm < 1e-9:
            v_dir = np.cross(t_hat, n_up)
            if np.linalg.norm(v_dir) < 1e-9: v_dir = np.array([1.0, 0.0, 0.0])
            v_dir /= (np.linalg.norm(v_dir) + 1e-12)
        else:
            v_dir = v_to_look / v_norm

        wall_back_offset_m = float(self.wall_overshoot_m)
        anchor_pre = chord_mid - wall_back_offset_m * v_dir
        h_anchor = float(np.dot(floor_normal, anchor_pre) + floor_offset)
        anchor_on_floor = anchor_pre - h_anchor * floor_normal

        # Compute wall normal
        a1w = t_hat / (np.linalg.norm(t_hat) + 1e-12)  # width
        a2w = n_up  # vertical
        n_wall_est = np.cross(a2w, a1w)
        n_wall_est /= (np.linalg.norm(n_wall_est) + 1e-12)  # into room (horizontal)
        d_wall_est = -float(np.dot(n_wall_est, anchor_on_floor))

        def proj_to_plane(P):
            return P - (np.dot(n_wall_est, P) + d_wall_est) * n_wall_est

        # Get wall corners
        P0p = proj_to_plane(arc_end0)
        P1p = proj_to_plane(arc_end1)
        s0 = float(np.dot(P0p - anchor_on_floor, a1w))
        s1 = float(np.dot(P1p - anchor_on_floor, a1w))
        s_min = min(s0, s1) - self.wall_pad_width_m
        s_max = max(s0, s1) + self.wall_pad_width_m
        wall_corners = np.stack([
            anchor_on_floor + s_min * a1w + 0.0 * a2w,
            anchor_on_floor + s_max * a1w + 0.0 * a2w,
            anchor_on_floor + s_max * a1w + self.wall_height_m * a2w,
            anchor_on_floor + s_min * a1w + self.wall_height_m * a2w,
        ], 0).astype(np.float64)
        if wall_corners.shape[0] >= 3:
            nw = np.cross(wall_corners[1] - wall_corners[0], wall_corners[3] - wall_corners[0])
            if np.linalg.norm(nw) > 1e-12:
                nw = nw / np.linalg.norm(nw)
                n_wall = nw.astype(np.float64)
                d_wall = float(-nw @ wall_corners[0])
            else:
                n_wall = n_wall_est
                d_wall = d_wall_est
        else:
            n_wall = n_wall_est
            d_wall = d_wall_est

        # Correct wall normal: towards the inside of the scene
        if mean_lookat is not None:
            v_ref = mean_lookat - anchor_on_floor
            v_ref = v_ref - n_up * (v_ref @ n_up)
            if np.dot(n_wall, v_ref) < 0:
                n_wall = -n_wall

        return n_wall, d_wall, wall_corners, dict(a1w=a1w, s_min=s_min, s_max=s_max, mean_lookat=mean_lookat)

    def _align_floor_and_wall(self,
                              floor_normal: np.ndarray,
                              floor_offset: np.ndarray,
                              wall_normal: np.ndarray,
                              wall_corners: np.ndarray,
                              floor_corners_pca: np.ndarray,
                              floor_points_w_inliers: np.ndarray,
                              floor_scale: float,
                              a1w: Optional[float] = None,
                              s_min: Optional[float] = None,
                              s_max: Optional[float] = None,
                              mean_lookat: Optional[np.ndarray] = None) -> np.ndarray:
        if wall_corners is not None and a1w is not None and s_min is not None and s_max is not None:
            w0 = wall_corners[0]  # left-bottom
            w1 = wall_corners[1]  # right-bottom
            e_t = w1 - w0
            Lw = float(np.linalg.norm(e_t))
            if Lw < 1e-12:
                floor_corners = floor_corners_pca
            else:
                e_t /= Lw
                e_n = (wall_normal if wall_normal is not None else np.cross(floor_normal, e_t))
                e_n = e_n / (np.linalg.norm(e_n) + 1e-12)
                if mean_lookat is not None:
                    v_ref = mean_lookat - w0
                    v_ref = v_ref - floor_normal * (v_ref @ floor_normal)
                    if np.dot(e_n, v_ref) < 0: e_n = -e_n

                V_all = (floor_points_w_inliers - w0) @ e_n
                V_pos = V_all[V_all > 0]
                if len(V_pos) >= 10:
                    vmax = float(np.percentile(V_pos, 90))
                elif len(V_pos) > 0:
                    vmax = float(np.max(V_pos))
                else:
                    if floor_corners_pca is not None:
                        v_candidates = (floor_corners_pca - w0) @ e_n
                        vmax = float(np.max(v_candidates)) - float(np.min(v_candidates))
                    else:
                        vmax = 0.3 * max(1.0, floor_scale)

                vmax = float(max(vmax, 0.1 * max(1.0, floor_scale)) * max(self.floor_depth_scale, 0.0))  # extend floor along the "depth" dimension
                c00 = w0
                c10 = w1
                c11 = w1 + vmax * e_n
                c01 = w0 + vmax * e_n
                floor_corners = np.stack([c00, c10, c11, c01], 0).astype(np.float64)
        else:
            floor_corners = floor_corners_pca

        if floor_corners is not None and floor_corners.shape[0] >= 3:
            nf = np.cross(floor_corners[1] - floor_corners[0], floor_corners[3] - floor_corners[0])
            if np.linalg.norm(nf) > 1e-12:
                nf = nf / np.linalg.norm(nf)
                floor_normal = nf.astype(np.float64)
                floor_offset = float(-nf @ floor_corners[0])

        return floor_normal, floor_offset, floor_corners

    def __call__(self, views: List[RGBDImage], vis_2d_path: Optional[Path] = None, vis_3d_path: Optional[Path] = None) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray]:
        # Floor plane estimation
        floor_plane_candidates = self._estimate_floor_planes(views)
        if not floor_plane_candidates:
            raise RuntimeError('No floor planes could be estimated.')
        floor_normal, floor_offset, floor_alignment_data = self._fit_floor_to_planes(floor_plane_candidates)

        # Wall plane + corners estimation
        wall_data = self._estimate_wall_plane([_.extrinsic_w2c for _ in views], floor_normal, floor_offset)
        if wall_data is None:
            raise RuntimeError('No wall planes could be estimated.')
        wall_normal, wall_offset, wall_corners, wall_alignment_data = wall_data

        # Alignment > Floor corners
        floor_normal, floor_offset, floor_corners = self._align_floor_and_wall(floor_normal, floor_offset, wall_normal, wall_corners, **(floor_alignment_data | wall_alignment_data))

        # Visualization
        if vis_2d_path is not None:
            if not self.__class__.visualize_floor_and_wall_2d(
                    views=views,
                    floor_masks=[_['floor_mask'] for _ in floor_plane_candidates],
                    floor_normal=floor_normal,
                    floor_offset=floor_offset,
                    floor_points_w_inliers=floor_alignment_data['floor_points_w_inliers'],
                    floor_scale=floor_alignment_data['floor_scale'],
                    out_path=vis_2d_path
            ):
                print(f'[{self.__class__.__name__}::__call__] Visualizing floor and wall planes in 2d: FAILed', file=sys.stderr)
            else:
                print(f'[{self.__class__.__name__}::__call__] Floor and wall visualization in 2d (overlays): {vis_2d_path}')

        if vis_3d_path is not None:
            if not self.__class__.visualize_floor_and_wall_3d(
                    views=views,
                    floor_corners=floor_corners,
                    floor_scale=floor_alignment_data['floor_scale'],
                    wall_corners=wall_corners,
                    out_path=vis_3d_path
            ):
                print(f'[{self.__class__.__name__}::__call__] Visualizing floor and wall planes in 3d: FAILed', file=sys.stderr)
            else:
                print(f'[{self.__class__.__name__}::__call__] Floor and wall visualization in 3d: {vis_3d_path}')

        return floor_normal, floor_offset, floor_corners, wall_normal, wall_offset, wall_corners

    @staticmethod
    def visualize_floor_and_wall_2d(views: List[RGBDImage], floor_masks: List[np.ndarray], floor_normal: np.ndarray, floor_offset: float, floor_points_w_inliers: np.ndarray, floor_scale: float, out_path: Path) -> bool:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _project_points(_K: np.ndarray, _w2c: np.ndarray, _Xw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            _Xw = np.asarray(_Xw, dtype=np.float64)
            _Rt = _w2c[:3, :3]
            _t = _w2c[:3, 3]
            _Xc = (_Rt @ _Xw.T).T + _t
            _z = _Xc[:, 2]
            _uv = np.empty((_Xc.shape[0], 2), dtype=np.float64)
            _uv[:, 0] = (_K[0, 0] * (_Xc[:, 0] / np.maximum(_z, 1e-6))) + _K[0, 2]
            _uv[:, 1] = (_K[1, 1] * (_Xc[:, 1] / np.maximum(_z, 1e-6))) + _K[1, 2]
            return _uv, _z > 1e-6

        if floor_points_w_inliers.shape[0] >= 3:
            mu_f = floor_points_w_inliers.mean(0)
            X0 = floor_points_w_inliers - mu_f
            C = (X0.T @ X0) / max(floor_points_w_inliers.shape[0] - 1, 1)
            _, ev = np.linalg.eigh(C)
            e1 = ev[:, 2]
            e1 = e1 - floor_normal * (e1 @ floor_normal)
            e1 /= (np.linalg.norm(e1) + 1e-12)
            e2 = np.cross(floor_normal, e1)
            e2 /= (np.linalg.norm(e2) + 1e-12)
            U = X0 @ e1
            V = X0 @ e2
            umin, umax = U.min(), U.max()
            vmin, vmax = V.min(), V.max()
            Nu, Nv = 10, 8
            u_line = np.linspace(umin, umax, Nu)
            v_line = np.linspace(vmin, vmax, Nv)
        else:
            u_line = np.linspace(-floor_scale, floor_scale, 10)
            v_line = np.linspace(-floor_scale, floor_scale, 8)
            mu_f = -floor_normal * floor_offset
            e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            e1 = e1 - floor_normal * (e1 @ floor_normal)
            e1 /= (np.linalg.norm(e1) + 1e-12)
            e2 = np.cross(floor_normal, e1)
            e2 /= (np.linalg.norm(e2) + 1e-12)

        grid_images = []
        orange = np.array([255, 165, 0], dtype=np.float32)
        for vi, (floor_mask, view) in enumerate(zip(floor_masks, views)):
            K, w2c, rgb = view.intrinsic, view.extrinsic_w2c, view.rgb
            H, W = rgb.shape[:2]
            ov = rgb.copy().astype(np.float32)
            for c in range(3):
                ch = ov[..., c]
                ch[floor_mask] = orange[c] * 0.35 + ch[floor_mask] * 0.65
                ov[..., c] = ch
            canvas_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)

            def plane_uv_to_xyz(_u, _v):
                return mu_f + _u * e1 + _v * e2

            v_samp = np.linspace(v_line.min(), v_line.max(), num=64)
            for u in u_line:
                X_line = np.stack([plane_uv_to_xyz(u, vv) for vv in v_samp], axis=0)
                uv, valid = _project_points(K, w2c, X_line)
                uv = uv[valid]
                if uv.shape[0] >= 2:
                    pts = np.round(uv).astype(np.int32)
                    inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
                    pts = pts[inside]
                    if pts.shape[0] >= 2:
                        cv2.polylines(canvas_bgr, [pts.reshape(-1, 1, 2)], False, (255, 255, 0), 1, cv2.LINE_AA)

            u_samp = np.linspace(u_line.min(), u_line.max(), num=64)
            for v in v_line:
                X_line = np.stack([plane_uv_to_xyz(uu, v) for uu in u_samp], axis=0)
                uv, valid = _project_points(K, w2c, X_line)
                uv = uv[valid]
                if uv.shape[0] >= 2:
                    pts = np.round(uv).astype(np.int32)
                    inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
                    pts = pts[inside]
                    if pts.shape[0] >= 2:
                        cv2.polylines(canvas_bgr, [pts.reshape(-1, 1, 2)], False, (255, 255, 0), 1, cv2.LINE_AA)

            grid_images.append(canvas_bgr)
        if len(grid_images) % 2 == 1:
            grid_images.append(np.zeros_like(grid_images[-1]))

        grid = np.vstack([np.hstack(grid_images[:len(grid_images) // 2]), np.hstack(grid_images[len(grid_images) // 2:])])
        grid = cv2.resize(grid, (int(grid.shape[1] * min(1.0, 3840 / max(grid.shape[:2]))), int(grid.shape[0] * min(1.0, 3840 / max(grid.shape[:2])))))
        cv2.imwrite(str(out_path), grid.clip(0, 255).astype(np.uint8))
        return True

    @staticmethod
    def visualize_floor_and_wall_3d(views: List[RGBDImage], floor_corners: np.ndarray, floor_scale: float, wall_corners: np.ndarray, out_path: Path) -> bool:
        import open3d as o3d
        pts_all, cols_all = [], []
        ref_idx = next((i for i, im in enumerate(views) if im is not None and im.depth is not None), None)
        if ref_idx is not None:
            ref_img = views[ref_idx]
            ref_pcd = ref_img.unproject().open3d
            pts_ref = np.asarray(ref_pcd.points, dtype=np.float64)
            cols_ref = np.asarray(ref_pcd.colors, dtype=np.float64)
            if pts_ref.size:
                pts_all.append(pts_ref)
                cols_all.append(cols_ref)

        def sample_sphere_points(center, radius, n_pts):
            phi = (1.0 + 5 ** 0.5) / 2.0
            i = np.arange(n_pts, dtype=np.float64)
            z = 1.0 - 2.0 * (i + 0.5) / n_pts
            theta = 2.0 * np.pi * i / phi
            rxy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
            unit = np.stack([rxy * np.cos(theta), rxy * np.sin(theta), z], axis=1)
            return center[None, :] + radius * unit

        if floor_corners is not None and floor_corners.shape[0] >= 3:
            r_small = max(0.02 * floor_scale, 0.02)
            n_per = 256
            pink = np.array([[1.0, 0.2, 0.8]], dtype=np.float64)
            for c in floor_corners:
                sp = sample_sphere_points(c, r_small, n_per)
                pts_all.append(sp)
                cols_all.append(np.repeat(pink, sp.shape[0], axis=0))

        if wall_corners is not None:
            r_small = max(0.02 * floor_scale, 0.02)
            n_per = 256
            green = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
            for c in wall_corners:
                sp = sample_sphere_points(c, r_small, n_per)
                pts_all.append(sp)
                cols_all.append(np.repeat(green, sp.shape[0], axis=0))

        def sample_quad(corners: np.ndarray, nu: int, nv: int):
            corners = np.asarray(corners, dtype=np.float64)
            if corners.shape[0] == 3:
                p0, p1, p2 = corners
                us = np.linspace(0.0, 1.0, nu)
                P = []
                for u in us:
                    vmax = 1.0 - u
                    vv = np.linspace(0.0, vmax, nv)
                    A = (1 - u - vv)[:, None] * p0[None, :] + u * p1[None, :] + vv[:, None] * p2[None, :]
                    P.append(A)
                return np.vstack(P) if P else np.zeros((0, 3), dtype=np.float64)
            else:
                p0, p1, p2, p3 = corners[:4]
                us = np.linspace(0.0, 1.0, nu)
                vs = np.linspace(0.0, 1.0, nv)
                uu, vv = np.meshgrid(us, vs)
                P = ((1 - uu) * (1 - vv))[:, :, None] * p0 + (uu * (1 - vv))[:, :, None] * p1 + \
                    (uu * vv)[:, :, None] * p2 + ((1 - uu) * vv)[:, :, None] * p3
                return P.reshape(-1, 3)

        if floor_corners is not None and floor_corners.shape[0] >= 3:
            Pf = sample_quad(floor_corners, 200, 200)
            if Pf.size:
                pts_all.append(Pf)
                cols_all.append(np.tile(np.array([[1.0, 0.2, 0.8]], dtype=np.float64), (Pf.shape[0], 1)))

        if wall_corners is not None and wall_corners.shape[0] >= 3:
            Pw = sample_quad(wall_corners, 160, 160)
            if Pw.size:
                pts_all.append(Pw)
                cols_all.append(np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (Pw.shape[0], 1)))

        pts_all = np.vstack(pts_all) if pts_all else np.zeros((0, 3), dtype=np.float64)
        cols_all = np.vstack(cols_all) if cols_all else np.zeros((0, 3), dtype=np.float64)
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(pts_all)
        pcd_all.colors = o3d.utility.Vector3dVector(cols_all)
        o3d.io.write_point_cloud(str(out_path), pcd_all)
        return True


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualBackground:
    floor_normal: np.ndarray
    floor_offset: float
    floor_corners: np.ndarray
    floor_texture_path: Optional[Path] = None
    wall_normal: Optional[np.ndarray] = None
    wall_offset: Optional[float] = None
    wall_corners: Optional[np.ndarray] = None
    wall_texture_path: Optional[Path] = None
    static_ply_path: Optional[Path] = None
    static_ply: Optional[trimesh.Trimesh | trimesh.Scene] = None
    bg_type: Literal['ply:pcd', 'ply:mesh', 'ply:splat', 'floor_wall'] = 'floor_wall'
    R_scene: Optional[np.ndarray] = None  # 3x3
    t_scene: Optional[np.ndarray] = None  # 1x3

    @classmethod
    def infer_scene_tf(cls, floor_corners: np.ndarray, wall_corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Infer a global scene transform (R_scene, t_scene):
        #   - floor is at the y=0 plane
        #   - floor's midpoint is at the origin
        #   - the wall's width direction is parallel to x-axis (viser)
        #   - the wall is at negative Z's (i.e. the entire scene is rotated so that when it loads the user "looks" at the inner of the scene not behind the wall)
        _normalize = lambda v: np.asarray(v, dtype=np.float64) / (np.linalg.norm(np.asarray(v, dtype=np.float64)) + 1e-12)

        def _ensure_quad(c):
            c = np.asarray(c, dtype=np.float64)
            if c is None or c.shape[0] < 3:
                return None
            if c.shape[0] == 3:
                p0, p1, p2 = c
                p3 = p2 - (p1 - p0)
                return np.stack([p0, p1, p2, p3], 0)
            return c[:4].astype(np.float64)

        def _rot_axis_angle(axis, theta):
            axis = _normalize(axis)
            K = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]], dtype=np.float64)
            I = np.eye(3, dtype=np.float64)
            return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

        def _rot_a_to_b(a, b):
            a = _normalize(a)
            b = _normalize(b)
            v = np.cross(a, b)
            s = np.linalg.norm(v)
            c = float(a @ b)
            if s < 1e-10:
                if c > 0.0:
                    return np.eye(3, dtype=np.float64)
                # 180° around any axis ⟂ a
                axis = _normalize(np.cross(a, np.array([1.0, 0.0, 0.0])) if abs(a[0]) < 0.9
                                  else np.cross(a, np.array([0.0, 1.0, 0.0])))
                return _rot_axis_angle(axis, np.pi)
            K = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]], dtype=np.float64)
            I = np.eye(3, dtype=np.float64)
            return I + K + K @ K * ((1.0 - c) / (s * s))

        # Conventions: Open3D-like world -> Viser world pre-rotation
        Rx = np.diag([1.0, -1.0, -1.0])  # forward -Z,+Y,+X  ->  +Z,-Y,+X
        up_vis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        _fc = _ensure_quad(floor_corners)
        _wc = _ensure_quad(wall_corners)
        R_scene = np.eye(3, dtype=np.float64)
        t_scene = np.zeros(3, dtype=np.float64)
        if _fc is not None:
            # --- Step 1: align floor normal to Viser up (−Y) ---
            n_floor = _normalize(np.cross(_fc[1] - _fc[0], _fc[3] - _fc[0]))  # world
            n_floor_vis0 = Rx @ n_floor
            R1 = _rot_a_to_b(n_floor_vis0, up_vis)
            Rtmp = R1 @ Rx

            # --- Step 2: yaw so wall width is parallel to +X ---
            if _wc is not None:
                width_vec_w = _wc[1] - _wc[0]  # bottom edge (left->right)
            else:
                # fallback: use floor's left->right
                width_vec_w = _fc[1] - _fc[0]
            width_vec_v = Rtmp @ width_vec_w
            # project to ground (XZ) plane
            width_vec_v = width_vec_v - up_vis * float(width_vec_v @ up_vis)
            if np.linalg.norm(width_vec_v) < 1e-9:
                yaw = 0.0
            else:
                yaw = -np.arctan2(width_vec_v[2], width_vec_v[0])  # aim to +X
            R2 = _rot_axis_angle(up_vis, yaw)

            # Compose
            R_scene = R2 @ Rtmp

            # --- Step 3: ensure wall sits at negative Z (scene faces user) ---
            wall_center_w = (_wc.mean(0) if _wc is not None else _fc.mean(0))
            z_after = (R_scene @ wall_center_w)[2]
            if z_after > 0.0:
                Rflip = _rot_axis_angle(up_vis, np.pi)  # 180° yaw
                R_scene = Rflip @ R_scene

            # --- Step 4: translate so floor midpoint is at origin (=> plane y=0) ---
            floor_center_w = _fc.mean(0)
            t_scene = -(R_scene @ floor_center_w)

        return R_scene, t_scene

    @classmethod
    def from_capturestudio_dataset(cls,
                                   dataset: MultiSessionDataset, t: int = 0,
                                   estimator_cls: Type[CapturestudioVirtualBackgroundFloorWallEstimator] = CapturestudioVirtualBackgroundFloorWallEstimator,
                                   **estimator_kwargs) -> 'CapturestudioVirtualBackground':
        from reconstruction.vis.dataset_visualizer import DatasetVisualizer
        dataset_t = dataset[t]
        rgbd_raw_images = DatasetVisualizer.sft_format_to_rgbd_images(dataset_t)

        estimator = estimator_cls(**estimator_kwargs)
        floor_normal, floor_offset, floor_corners, wall_normal, wall_offset, wall_corners = estimator(rgbd_raw_images)

        R_scene, t_scene = cls.infer_scene_tf(floor_corners, wall_corners)

        return CapturestudioVirtualBackground(
            floor_normal=floor_normal,
            floor_offset=floor_offset,
            floor_corners=floor_corners,
            floor_texture_path=PathUtils.resources_path() / 'backdrops' / f'floor_dark.jpg',
            wall_normal=wall_normal,
            wall_offset=wall_offset,
            wall_corners=wall_corners,
            wall_texture_path=PathUtils.resources_path() / 'backdrops' / f'curtain_dark{"_hslu" if estimator.wall_show_logo else ""}.png',
            R_scene=R_scene,
            t_scene=t_scene,
        )

    @classmethod
    def _load_metadata(cls, json_path: Path, floor_corners: Optional[np.ndarray] = None, wall_corners: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Load metadata
        if json_path.exists():
            with open(json_path, "r") as f:
                metadata = json.load(f)
            floor_corners = np.array(metadata['floor_corners']).astype(np.double)
            wall_corners = np.array(metadata['wall_corners']).astype(np.double)
            if 'viser_tf' in metadata:
                from scipy.spatial.transform import Rotation as R
                tf = np.array(metadata['viser_tf'])
                m = np.concatenate([floor_corners, wall_corners], axis=0).mean(axis=0)[None]
                floor_corners = R.from_matrix(tf[:3, :3]).apply(floor_corners - m) + m + tf[:3, 3][None]
                wall_corners = R.from_matrix(tf[:3, :3]).apply(wall_corners - m) + m + tf[:3, 3][None]
            if 'scale' in metadata:
                scale = metadata['scale']  # Scale object
                scale_arr = np.asarray(scale, dtype=np.float32).reshape(-1)
                if scale_arr.size == 1:
                    scale_xyz = np.repeat(scale_arr, 3).astype(np.float32)
                elif scale_arr.size == 3:
                    scale_xyz = scale_arr.astype(np.float32)
                else:
                    raise ValueError(f"Expected scalar or 3-vector scale, got shape {scale_arr.shape}")
            else:
                scale_xyz = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            log(f'[{cls.__name__}::_load_metadata] No json metadata file provided!', 'warning')
            assert floor_corners is not None
            assert wall_corners is not None
            scale_xyz = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        return floor_corners, wall_corners, scale_xyz

    @classmethod
    def _load_trimesh(cls, obj_path: str, floor_corners: Optional[np.ndarray] = None, wall_corners: Optional[np.ndarray] = None) -> Tuple[trimesh.Geometry, np.ndarray, np.ndarray]:
        if Path(obj_path).suffix not in ['.ply', '.obj']:
            raise RuntimeError

        # load object
        obj = cls.check_trimesh_can_read(obj_path)
        if obj is None:
            raise RuntimeError('Trimesh could not read the object at: ' + obj_path)

        # load metadata
        floor_corners, wall_corners, scale_xyz = cls._load_metadata(Path(obj_path).with_suffix(".json"), floor_corners, wall_corners)

        # process
        obj.apply_scale(tuple(scale_xyz.tolist()))

        return obj, floor_corners, wall_corners

    @classmethod
    def _load_gs(cls, file_path: str, floor_corners: Optional[np.ndarray] = None, wall_corners: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:

        def _load_splat_file(center: bool = False) -> Dict[str, np.ndarray]:
            """Load an antimatter15-style splat file."""
            start_time = time.time()
            splat_buffer = file_path.read_bytes()
            bytes_per_gaussian = (
                # Each Gaussian is serialized as:
                # - position (vec3, float32)
                    3 * 4
                    # - xyz (vec3, float32)
                    + 3 * 4
                    # - rgba (vec4, uint8)
                    + 4
                    # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
                    + 4
            )
            assert len(splat_buffer) % bytes_per_gaussian == 0
            num_gaussians = len(splat_buffer) // bytes_per_gaussian

            # Reinterpret cast to dtypes that we want to extract.
            splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
                (num_gaussians, bytes_per_gaussian)
            )
            scales = splat_uint8[:, 12:24].copy().view(np.float32)
            wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
            from viser import transforms as tf
            Rs = tf.SO3(wxyzs).as_matrix()
            covariances = np.einsum(
                "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
            )
            centers = splat_uint8[:, 0:12].copy().view(np.float32)
            if center:
                centers -= np.mean(centers, axis=0, keepdims=True)
            print(
                f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
            )
            return {
                "centers": centers,
                # Colors should have shape (N, 3).
                "rgbs": splat_uint8[:, 24:27] / 255.0,
                "opacities": splat_uint8[:, 27:28] / 255.0,
                # Covariances should have shape (N, 3, 3).
                "covariances": covariances,
                # No SH coefficients in the splat file.
                "sh_coeffs": np.concatenate([splat_uint8[:, 24:27] / 255.0, np.zeros((num_gaussians, 45), dtype=np.float32)], axis=-1),
            }

        def _load_ply_file(center: bool = False, subsample_factor: float = 1.0) -> Dict[str, np.ndarray]:
            """Load Gaussians stored in a PLY file."""
            start_time = time.time()
            SH_C0 = 0.28209479177387814

            plydata = PlyData.read(file_path)
            v = plydata["vertex"]

            # subsampling
            num_gaussians = len(v)
            if 0.0 < subsample_factor < 1.0:
                target_num = max(1, int(num_gaussians * subsample_factor))
                # Random uniform sampling for an even reduction
                indices = np.random.choice(num_gaussians, target_num, replace=False)
                v = v[indices]
                num_gaussians = target_num

            positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
            scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
            wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
            wxyzs = wxyzs / (np.linalg.norm(wxyzs, axis=-1, keepdims=True) + 1e-12)
            colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
            opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))
            dc_coeffs = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)

            # Rest coefficients 0-14 belongs to RED channel, 15-29 to GREEN, 30-44 to BLUE
            # Due to spherical harmonic calculations calculating a triplet at a time
            # we need to stack them by (0,15,30), (1,16,31), ..., (14,29,44)
            rest_coeffs = []
            for i in range(15):
                rest_coeffs.append(v[f"f_rest_{i}"])
                rest_coeffs.append(v[f"f_rest_{i + 15}"])
                rest_coeffs.append(v[f"f_rest_{i + 30}"])
            rest_coeffs = np.stack(rest_coeffs, axis=1)
            sh_coeffs = np.concatenate([dc_coeffs, rest_coeffs], axis=1)
            from viser import transforms as tf
            Rs = tf.SO3(wxyzs).as_matrix()
            covariances = np.einsum(
                "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
            )

            if center:
                positions -= np.mean(positions, axis=0, keepdims=True)

            log(f"[{cls.__name__}::_load_gs::_load_ply] PLY file with {num_gaussians=} loaded in {time.time() - start_time:.3f} seconds")

            return {
                "centers": positions,
                "rgbs": colors,
                "opacities": opacities,
                "covariances": covariances,
                "sh_coeffs": sh_coeffs,
            }

        # load object
        file_path = Path(file_path)
        if file_path.suffix == ".splat":
            splat_data = _load_splat_file(center=True)
        elif file_path.suffix == ".ply":
            splat_data = _load_ply_file(center=True)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")

        # load metadata
        floor_corners, wall_corners, scale_xyz = cls._load_metadata(Path(file_path).with_suffix(".json"), floor_corners, wall_corners)

        # Scale centers and covariances consistently: C' = A C A^T.
        A = np.diag(scale_xyz).astype(np.float32)
        centers_mean = splat_data["centers"].mean(axis=0)[None]
        splat_data["centers"] = (np.asarray(splat_data["centers"] - centers_mean, dtype=np.float32) * scale_xyz[None, :]).astype(np.float32) + centers_mean
        splat_data["covariances"] = np.einsum("ij,njk,lk->nil", A, np.asarray(splat_data["covariances"], dtype=np.float32), A).astype(np.float32)

        return splat_data, floor_corners, wall_corners

    @classmethod
    def from_object(cls, obj_path: str, floor_corners: Optional[np.ndarray] = None, wall_corners: Optional[np.ndarray] = None, is_gs: bool = False) -> 'CapturestudioVirtualBackground':
        try:
            if is_gs:
                raise RuntimeError
            obj, floor_corners, wall_corners = cls._load_trimesh(obj_path, floor_corners=floor_corners, wall_corners=wall_corners)
            bg_type = 'ply:mesh' if isinstance(obj, trimesh.Trimesh) else 'ply:pcd'
        except RuntimeError:
            obj, floor_corners, wall_corners = cls._load_gs(obj_path, floor_corners=floor_corners, wall_corners=wall_corners)
            bg_type = 'ply:splat'

        def compute_plane(corners: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
            """Helper to compute normal and offset from 4 planar corner points."""
            if corners is None or corners.shape[0] < 3:
                return None, None
            # Cross product of two adjacent edges (p1-p0 and p3-p0)
            n = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                d = float(-n @ corners[0])
                return n.astype(np.float64), d
            return None, None

        floor_normal, floor_offset = compute_plane(floor_corners)
        wall_normal, wall_offset = compute_plane(wall_corners)

        R_scene, t_scene = cls.infer_scene_tf(floor_corners, wall_corners)

        background = CapturestudioVirtualBackground(
            floor_corners=floor_corners,
            floor_normal=floor_normal,
            floor_offset=floor_offset,
            floor_texture_path=None,
            wall_corners=wall_corners,
            wall_normal=wall_normal,
            wall_offset=wall_offset,
            static_ply_path=Path(obj_path),
            static_ply=obj,
            bg_type=bg_type,
            R_scene=R_scene,
            t_scene=t_scene
        )
        return background

    @staticmethod
    def check_trimesh_can_read(obj_path: str) -> Optional[trimesh.Trimesh]:
        from trimesh.resolvers import FilePathResolver

        obj_path = os.path.abspath(obj_path)
        obj_dir = os.path.dirname(obj_path)
        print(f"[check] OBJ: {obj_path}")
        print(f"[check] Exists: {os.path.exists(obj_path)}")
        print(f"[check] Dir: {obj_dir}\n")

        try:
            # Ensure relative texture paths in MTL are resolved against the OBJ folder.
            resolver = FilePathResolver(obj_dir)
            scene_or_mesh = trimesh.load(
                obj_path,
                force='scene',  # prefer a Scene still OK if it returns a Mesh
                process=False,  # don't alter geometry when validating
                resolver=resolver
            )
        except Exception as e:
            print(f"[error] Failed to load with trimesh.load(..., force='scene'): {str(e)}")
            traceback.print_exc()
            return None

        def describe_mesh(_name, _mesh: trimesh.Trimesh):
            print(f"  - geometry: {_name}")
            print(f"      V/F: {len(_mesh.vertices)}/{len(_mesh.faces)}")
            _kind = getattr(_mesh.visual, "kind", "none")
            print(f"      visual.kind: {_kind}")
            if _kind == "texture":
                _mat = getattr(_mesh.visual, "material", None)
                _img = getattr(_mat, "image", None)
                print(f"      texture image: {'yes' if _img is not None else 'no'}")

        print("[ok] Loaded type:", type(scene_or_mesh).__name__)
        if isinstance(scene_or_mesh, trimesh.Scene):
            print(f"[ok] Scene has {len(scene_or_mesh.geometry)} geometries")
            for name, geom in scene_or_mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    describe_mesh(name, geom)
                else:
                    print(f"  - geometry: {name} (type {type(geom).__name__})")
        elif isinstance(scene_or_mesh, trimesh.Trimesh):
            describe_mesh(os.path.basename(obj_path), scene_or_mesh)
        else:
            print("[warn] Unexpected type:", type(scene_or_mesh))

        return scene_or_mesh


@dataclasses.dataclass
class CapturestudioVirtualLight:
    t_world: np.ndarray
    rotation_world: Optional[np.ndarray]
    color: Tuple[int, int, int]
    intensity: int


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualCameras:
    gt_extrinsics_c2ws: np.ndarray  # C x 4 x 4
    gt_intrinsics: List[np.ndarray]  # C x 3 x 3
    virtual_extrinsics_c2ws: List[np.ndarray]  # t_total x 4 x 4
    virtual_intrinsics: List[np.ndarray]  # t_total x 3 x 3
    virtual_assignment: List[int]  # t_total
    camera_orbit_type: Literal['interpolated', 'audience']
    camera_orbit: CameraOrbit
    image_size_hw: Tuple[int, int]
    t_total: int
    z_near: float = 0.1
    z_far: float = 1_000.0
    frustum_scale: float = 0.1
    traverse_velocity: float = 0.5  # arc meters / second
    data_fps: int = 30
    camera_show_gt_frusta: bool = False
    camera_show_virtual_frusta: bool = False
    orbit_offset_m: float = 0.0  # user control to push the orbit outwards (meters)
    _t_current: int = 0
    _t_start: int = 0
    _R_scene: Optional[np.ndarray] = None
    _t_scene: Optional[np.ndarray] = None
    # cached look-at and roll alignment
    _virt_lookat_w: Optional[np.ndarray] = None
    _roll_delta_rad: Optional[float] = None

    @property
    def current_camera(self) -> Dict[str, np.ndarray]:
        # Prepared, roll-fixed, (optionally) offset pose:
        c2w = self._prepare_virtual_c2w(self._t_current, self._R_scene, self._t_scene)
        K = self.virtual_intrinsics[self._t_current]
        return dict(extrinsic_c2w=c2w, intrinsics=K, z_near=self.z_near, z_far=self.z_far)

    @property
    def gt_cam_index(self) -> int:
        return self.virtual_assignment[self._t_current]

    # noinspection PyUnusedLocal
    def tick(self, **kwargs) -> Optional[Tuple[int, int]]:
        if self.t_total <= 0 or len(self.virtual_extrinsics_c2ws) == 0:
            return None

        nV = len(self.virtual_extrinsics_c2ws)
        t_prev = int(self._t_current) % nV
        t_next = (t_prev + 1) % nV

        self._t_current = t_next
        return t_prev, t_next

    @classmethod
    def from_capturestudio_dataset(cls,
                                   dataset: MultiSessionDataset,
                                   background: Optional[CapturestudioVirtualBackground],
                                   camera_orbit_type: Literal['interpolated', 'audience'],
                                   t_total: int,
                                   camera_orbit_start_idx: Union[int, float] = 0,
                                   **dataclass_kwargs) -> 'CapturestudioVirtualCameras':
        floor_wall_kwargs = {}
        if background is not None:
            floor_wall_kwargs = dict(floor_normal=background.floor_normal, floor_offset=background.floor_offset,
                                     wall_normal=background.wall_normal, wall_offset=background.wall_offset)

        camera_orbit = dataset.get_camera_orbit(camera_orbit_type, **floor_wall_kwargs)
        data_fps = dataclass_kwargs.get('data_fps', 30)
        traverse_velocity = dataclass_kwargs.pop('camera_traverse_velocity', 0.5)
        camera_orbit_traversor = camera_orbit.traverse(velocity=traverse_velocity, data_fps=data_fps)

        virtual_data = [(K_, c2w_, idx_closest_) for i, (K_, c2w_, idx_closest_, _) in
                        zip(range(0, t_total, 30 // data_fps), camera_orbit_traversor)]

        if camera_orbit_start_idx != 0:
            # Determine the sequence starting from camera_orbit_start_idx to the end
            n = len(virtual_data)
            if isinstance(camera_orbit_start_idx, float):
                camera_orbit_start_idx = int(camera_orbit_start_idx * n)
            start = camera_orbit_start_idx % n
            forward_part = virtual_data[start:]

            # Fill the remaining length by traversing backwards from the end.
            # We start the reversed slice at index 1 to avoid duplicating the 'peak' element.
            needed_len = n - len(forward_part)
            backward_part = virtual_data[::-1][1: 1 + needed_len]

            virtual_data = forward_part + backward_part

        virtual_intrinsics: List[np.ndarray] = [_[0] for _ in virtual_data]
        virtual_c2ws: List[np.ndarray] = [_[1] for _ in virtual_data]
        virtual_assignment: List[int] = [_[2] for _ in virtual_data]
        gt_c2ws = camera_orbit.gt_extrinsics_c2w[camera_orbit.gt_cam_idx_s0]

        orbit_offset_m = float(dataclass_kwargs.pop('camera_orbit_offset_m', dataclass_kwargs.pop('orbit_offset_m', 0.0)))
        return cls(
            gt_extrinsics_c2ws=gt_c2ws,
            gt_intrinsics=camera_orbit.gt_intrinsics[camera_orbit.gt_cam_idx_s0],
            camera_orbit_type=camera_orbit_type,
            camera_orbit=camera_orbit,
            t_total=t_total,
            virtual_extrinsics_c2ws=virtual_c2ws,
            virtual_intrinsics=virtual_intrinsics,
            virtual_assignment=virtual_assignment,
            traverse_velocity=traverse_velocity,
            orbit_offset_m=orbit_offset_m,
            image_size_hw=dataset.target_image_size_hw,
            **dataclass_kwargs
        )

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, np.float64)
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    @staticmethod
    def _axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
        a = CapturestudioVirtualCameras._normalize(axis)
        x, y, z = a
        c, s, C = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
        return np.array([
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c]
        ], dtype=np.float64)

    def _compute_lookat_if_needed(self, R_scene: Optional[np.ndarray], t_scene: Optional[np.ndarray]) -> None:
        """Least-squares intersection point of all virtual forward rays (after scene transform)."""
        if self._virt_lookat_w is not None:
            return
        if len(self.virtual_extrinsics_c2ws) == 0:
            self._virt_lookat_w = np.zeros(3, np.float64)
            return

        def _apply_scene(_c2w: np.ndarray) -> np.ndarray:
            if R_scene is None or t_scene is None:
                return _c2w
            _out = np.eye(4, dtype=np.float64)
            _out[:3, :3] = R_scene @ _c2w[:3, :3]
            _out[:3, 3] = (R_scene @ _c2w[:3, 3]) + t_scene
            return _out

        I3 = np.eye(3, dtype=np.float64)
        M = np.zeros((3, 3), np.float64)
        b = np.zeros((3,), np.float64)
        for c2w in self.virtual_extrinsics_c2ws:
            A = _apply_scene(c2w)
            C = A[:3, 3]
            f = self._normalize(A[:3, 2])  # +Z forward (OpenCV/Viser)
            P = I3 - np.outer(f, f)  # projector ⟂ f
            M += P
            b += P @ C
        try:
            self._virt_lookat_w = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            # Fallback: mean of ray midpoints
            self._virt_lookat_w = b / (np.trace(M) / 2.0 + 1e-12)

    def _roll_align_first_to_gt(self, R_scene: Optional[np.ndarray], t_scene: Optional[np.ndarray]) -> None:
        """Compute a single roll delta so that the FIRST virtual camera matches the FIRST GT camera."""
        if self._roll_delta_rad is not None:
            return
        if len(self.virtual_extrinsics_c2ws) == 0 or self.gt_extrinsics_c2ws is None or self.gt_extrinsics_c2ws.shape[0] == 0:
            self._roll_delta_rad = 0.0
            return

        def _apply_scene(c2w: np.ndarray) -> np.ndarray:
            if R_scene is None or t_scene is None:
                return c2w
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = R_scene @ c2w[:3, :3]
            out[:3, 3] = (R_scene @ c2w[:3, 3]) + t_scene
            return out

        # scene-transformed
        V0 = _apply_scene(self.virtual_extrinsics_c2ws[0])
        G0 = _apply_scene(self.gt_extrinsics_c2ws[0])

        f = self._normalize(V0[:3, 2])  # virtual forward
        down_world = np.array([0.0, 1.0, 0.0], np.float64)  # Viser world DOWN (+Y)
        # Base upright down (zero roll)
        d_base = self._normalize(down_world - f * float(down_world @ f))

        # Target down: GT0 down projected ⟂ virtual forward
        d_gt_raw = self._normalize(G0[:3, 1])  # +Y is down in OpenCV/Viser
        d_tgt = self._normalize(d_gt_raw - f * float(d_gt_raw @ f))
        # Angle to rotate around f: d_base -> d_tgt
        sin_a = float(np.dot(f, np.cross(d_base, d_tgt)))
        cos_a = float(np.dot(d_base, d_tgt))
        self._roll_delta_rad = float(np.arctan2(sin_a, cos_a))

    def _prepare_virtual_c2w(self, idx: int, R_scene: Optional[np.ndarray], t_scene: Optional[np.ndarray]) -> np.ndarray:
        """Apply scene transform, orbit offset, look-at aim, and zero-roll + global roll delta."""
        # --- scene transform of the raw virtual camera ---
        c2w = np.asarray(self.virtual_extrinsics_c2ws[idx], np.float64)
        if R_scene is not None and t_scene is not None:
            Rw = c2w[:3, :3]
            tw = c2w[:3, 3]
            c2w_v = np.eye(4, dtype=np.float64)
            c2w_v[:3, :3] = R_scene @ Rw
            c2w_v[:3, 3] = (R_scene @ tw) + t_scene
        else:
            c2w_v = c2w.copy()

        C = c2w_v[:3, 3]
        f = self._normalize(c2w_v[:3, 2])  # base forward

        # --- ensure we have a look-at if needed ---
        self._compute_lookat_if_needed(R_scene, t_scene)
        L = self._virt_lookat_w if self._virt_lookat_w is not None else (C + f)

        # --- optional orbit offset: push radially from look-at and re-aim at L ---
        if float(self.orbit_offset_m) != 0.0:
            v = C - L
            dist = float(np.linalg.norm(v))
            dir_ = self._normalize(v if dist > 1e-9 else f)
            C = L + (dist + float(self.orbit_offset_m)) * dir_
            f = self._normalize(L - C)  # re-aim to look-at

        # --- zero-roll: make camera "down" as close as possible to world down ---
        down_world = np.array([0.0, 1.0, 0.0], np.float64)  # OpenCV/Viser +Y = down
        d0 = down_world - f * float(down_world @ f)
        if np.linalg.norm(d0) < 1e-9:
            # forward almost vertical pick an arbitrary horizontal down
            d0 = np.array([1.0, 0.0, 0.0], np.float64) - f * float(f[0])
        d = self._normalize(d0)
        r = self._normalize(np.cross(d, f))
        d = self._normalize(np.cross(f, r))  # re-orthogonalize

        R_upright = np.stack([r, d, f], axis=1)

        # --- global roll delta so the first virtual matches first GT exactly ---
        self._roll_align_first_to_gt(R_scene, t_scene)
        if abs(float(self._roll_delta_rad or 0.0)) > 1e-12:
            R_roll = self._axis_angle(f, float(self._roll_delta_rad))
            R_final = R_roll @ R_upright
        else:
            R_final = R_upright

        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R_final
        out[:3, 3] = C
        return out


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualDynamicForeground:
    image_generator: Optional[Iterator[Union[List[Union[RGBDImage, GSImage, o3d.geometry.PointCloud]], o3d.geometry.PointCloud]]] = None
    index: int = 0
    _active_cam_idx: int = 0
    _last_vis_args = ()

    _blending_strategy: Literal['swap', 'blend:blend_pcr', 'merge:naive', 'merge:tsdf'] = 'swap'

    # noinspection PyUnusedLocal
    def tick(self, new_active_cam_idx: Optional[int] = None, **kwargs) -> None:
        if new_active_cam_idx is not None:
            self._active_cam_idx = new_active_cam_idx

    @classmethod
    def from_capturestudio_dataset(cls,
                                   dataset: MultiSessionDataset,
                                   t_start: int = 0,
                                   t_total: int = -1,
                                   index: int = 0,
                                   active_cam_idx: int = 0,
                                   use_gs: bool = False,
                                   blending_strategy: Literal['swap', 'blend:blend_pcr', 'merge:naive', 'merge:tsdf'] = 'swap',
                                   write_ply_files: bool = False,
                                   write_ply_root: str = '/mnt/d/TEASER_{gs_or_pcd}/{session}') -> 'CapturestudioVirtualDynamicForeground':
        class SessionImagesGenerator:
            def __init__(self, _dataset, _use_gs, _t_start, _t_total):
                self._dataset = _dataset
                self._use_gs = _use_gs
                self._t_start = _t_start
                self._t_total = _t_total if _t_total > 0 else len(_dataset) - abs(_t_total) - _t_start
                self._t_current = _t_start
                self._write_ply_files = write_ply_files
                self._ply_out_dir = Path(write_ply_root.format(gs_or_pcd="GS" if use_gs else "PCD", session=dataset.session_names[0].lower()))
                if self._write_ply_files:
                    self._ply_out_dir.mkdir(parents=True, exist_ok=True)
                    for cam_idx_s0 in dataset.cam_indices_s0:
                        (self._ply_out_dir / f'cam{(int(cam_idx_s0) + 1):02d}').mkdir(exist_ok=True)
                    log(f'[{cls.__name__}::from_capturestudio_dataset] Writing ply files to {self._ply_out_dir}', 'info')

            def __iter__(self):
                return self

            def __next__(self):
                if self._t_start <= self._t_current < (self._t_start + self._t_total):
                    # print(f'\t[CapturestudioVirtualDynamicForeground::from_capturestudio_dataset][SessionImagesGenerator::next] Loading data at t={self._t_current} (use_gs={self._use_gs})')
                    _dataset_t = self._dataset[self._t_current]
                    _images_t = DatasetVisualizer.sft_format_to_rgbd_images(_dataset_t)
                    if self._use_gs:
                        _images_t = [GSImage.from_rgbd_image(_) for _ in _images_t]
                    if self._write_ply_files:
                        for cam_idx_s0, img in zip(self._dataset.cam_indices_s0, _images_t):
                            ply_path = self._ply_out_dir / f'cam{(int(cam_idx_s0) + 1):02d}' / f'{(self._t_current - self._t_start):06d}.ply'
                            if not ply_path.exists():
                                img.unproject().save_ply(ply_path)
                    self._t_current += 1
                    return _images_t
                raise StopIteration()

        return cls(
            image_generator=SessionImagesGenerator(dataset, use_gs, t_start, t_total).__iter__(),
            index=index,
            _active_cam_idx=active_cam_idx,
            _blending_strategy=blending_strategy,
        )

    @classmethod
    def from_merged_ply_files(cls,
                              ply_root: Path,
                              ply_format: str = '{ply_index:06d}.ply',
                              t_start: int = 0,
                              t_total: int = -1,
                              index: int = 0) -> 'CapturestudioVirtualDynamicForeground':
        class PlyGenerator:
            def __init__(self, _ply_root, _ply_format, _t_start, _t_total):
                self._ply_root = Path(_ply_root)
                self._ply_format = _ply_format
                self._t_start = _t_start
                self._t_total = _t_total if _t_total > 0 else (len(list(self._ply_root.glob('*.ply'))) - abs(_t_total) - _t_start + 1)
                self._t_current = _t_start
                self._ply_files = [
                    Path(self._ply_root) / self._ply_format.format(ply_index=_)
                    for _ in range(t_start, self._t_total)
                ]

            def __iter__(self):
                return self

            def __next__(self) -> o3d.geometry.PointCloud:
                if self._t_start <= self._t_current < (self._t_start + self._t_total):
                    log(f'\t[CapturestudioVirtualDynamicForeground::from_merged_ply_files][PlyGenerator::next] Loading data at t={self._t_current}', 'debug')
                    _ply_t = self._ply_files[self._t_current]
                    _ply_t_o3d = o3d.io.read_point_cloud(str(_ply_t))
                    self._t_current += 1
                    return _ply_t_o3d
                raise StopIteration()

        return cls(
            image_generator=PlyGenerator(ply_root, ply_format, t_start, t_total).__iter__(),
            index=index,
            _active_cam_idx=0,
            _blending_strategy='merge:naive',
        )


class CapturestudioVirtualScene(metaclass=abc.ABCMeta):
    def __init__(self,
                 background: Optional[CapturestudioVirtualBackground] = None,
                 foregrounds: Optional[Union[CapturestudioVirtualDynamicForeground, List[CapturestudioVirtualDynamicForeground]]] = None,
                 cameras: Optional[CapturestudioVirtualCameras] = None,
                 lights: Optional[CapturestudioVirtualLight] = None):
        self._background = background
        self._foregrounds = foregrounds if isinstance(foregrounds, list) else [foregrounds]
        self._cameras = cameras
        self._lights = lights

        self._R_scene = None
        self._t_scene = None

        if foregrounds is not None or background is not None or cameras is not None or lights is not None:
            self.on_scene_updated(f'{("|" + foregrounds) if foregrounds is not None else ""}{("|" + background) if background is not None else ""}{("|" + cameras) if cameras is not None else ""}{("|" + lights) if lights is not None else ""}'.lstrip('|'))

    @abc.abstractmethod
    def on_scene_updated(self, what_updated: str, set_not_unset: bool = True) -> None:
        """
        Updated viewer once scene elements have been updated.

        Parameters
        ----------
        what_updated: str
            One or more of 'foregrounds', 'background', 'cameras', 'lights'. If more than one, they will be joined with "|".
        set_not_unset: bool
            If True, the element(s) were set, otherwise they were unset.
        """
        raise NotImplementedError

    def flush(self):
        pass

    def tick(self):
        pass

    def set_background(self, background: Optional[CapturestudioVirtualBackground]) -> 'CapturestudioVirtualScene':
        if background is None:
            return self

        if self._background is not None:
            self.unset_background()
        self._background = background
        self._R_scene = background.R_scene
        self._t_scene = background.t_scene
        self.on_scene_updated('background')
        return self

    def set_foregrounds(self, foregrounds: Union[CapturestudioVirtualDynamicForeground, List[CapturestudioVirtualDynamicForeground]]) -> 'CapturestudioVirtualScene':
        if self._foregrounds is not None:
            self.unset_foregrounds()
        self._foregrounds = foregrounds if isinstance(foregrounds, list) else [foregrounds]
        self.on_scene_updated('foregrounds')
        return self

    def set_cameras(self, cameras: List[CapturestudioVirtualCameras]) -> 'CapturestudioVirtualScene':
        if self._cameras is not None:
            self.unset_cameras()
        self._cameras = cameras
        self.on_scene_updated('cameras')
        return self

    def set_lights(self, lights: List[CapturestudioVirtualLight]) -> 'CapturestudioVirtualScene':
        self._lights = lights
        self.on_scene_updated('lights')
        return self

    def unset_background(self) -> 'CapturestudioVirtualScene':
        del self._background
        self._background = None
        self._R_scene = None
        self._t_scene = None
        self.on_scene_updated('background', set_not_unset=False)
        time.sleep(0.2)
        gc.collect()
        return self

    def unset_foregrounds(self) -> 'CapturestudioVirtualScene':
        del self._foregrounds
        self._foregrounds = None
        self.on_scene_updated('foregrounds', set_not_unset=False)
        time.sleep(0.2)
        gc.collect()
        return self

    def unset_cameras(self) -> 'CapturestudioVirtualScene':
        del self._cameras
        self._cameras = None
        self.on_scene_updated('cameras', set_not_unset=False)
        time.sleep(0.2)
        gc.collect()
        return self

    def unset_lights(self) -> 'CapturestudioVirtualScene':
        del self._lights
        self._lights = None
        self.on_scene_updated('lights', set_not_unset=False)
        time.sleep(0.2)
        gc.collect()
        return self


@dataclasses.dataclass(kw_only=True)
class TeaserGeneratorRenderConfig(object):
    use_gs: bool
    image_size_hw: Tuple[int, int]

    # Background
    floor_depth_scale: float
    wall_overshoot_m: float
    wall_pad_width_m: float

    # Cameras
    camera_show_gt_frusta: bool
    camera_show_virtual_frusta: bool
    camera_orbit_type: Literal['interpolated', 'audience']
    camera_traverse_velocity: float  # m/s
    camera_orbit_offset_m: float

    @classmethod
    def for_apr_may_2025(cls, use_gs: bool, image_size_hw: Tuple[int, int], show_gt_frusta: bool = False, camera_traverse_velocity: float = 0.6, camera_orbit_offset_m:float=0.4) -> 'TeaserGeneratorRenderConfig':
        return TeaserGeneratorRenderConfig(
            use_gs=use_gs,
            image_size_hw=image_size_hw,
            # Background
            floor_depth_scale=3.5,
            wall_overshoot_m=-4.0,
            wall_pad_width_m=3.0,
            # Cameras
            camera_show_gt_frusta=show_gt_frusta,
            camera_show_virtual_frusta=False,
            camera_orbit_type='interpolated',
            camera_traverse_velocity=camera_traverse_velocity,
            camera_orbit_offset_m=camera_orbit_offset_m,
        )


class TeaserGenerator(metaclass=abc.ABCMeta):
    def __init__(
            self,
            session_perf: Union[str, List[Union[str, Path]]],
            session_calib: str,
            calib_method: Literal['Caliscope', 'MultiCamCalib'],
            depth_source: Literal['bilateral_spatial', 'bilateral_temporal', 'aligned'],
            cam_idx_perf: List[int],
            cam_idx_raw: List[int],
            render_config: TeaserGeneratorRenderConfig,
            t_start: int = 0,
            t_total: int = 300,
            debug: bool = False,
            out_video_path: str = '{session}_{start_idx:03d}_{pcd_or_gs}.mp4',
            **scene_kwargs
    ):
        sessions_perf = [session_perf] if isinstance(session_perf, (str, Path)) else session_perf
        if len(sessions_perf) > 1 and isinstance(sessions_perf[1], Path):
            # first --> raw, then --> merged ply dirs
            self.session_raw = sessions_perf.pop(0)
        else:
            self.session_raw = sessions_perf[0]
        self.sessions_perf = sessions_perf
        self.session_calib = session_calib
        self.calib_method = calib_method
        self.depth_source = depth_source
        self.cam_idx_perf = cam_idx_perf
        self.cam_idx_raw = cam_idx_raw
        self.render_config = render_config
        self.t_start = t_start
        self.t_total = t_total
        self.debug = debug
        self.out_video_path = out_video_path.format(session='+'.join([(_ if isinstance(_, str) else str(_.name)).lower() for _ in self.sessions_perf]), start_idx=t_start, pcd_or_gs='pcd' if not render_config.use_gs else 'gs')
        self._scene_kwargs = scene_kwargs
        self._scene = None
        self._init_scene()

    @property
    def dataset_raw(self) -> MultiSessionDataset:
        return MultiSessionDataset(
            calibration_session_name=self.session_calib,
            calibration_method=self.calib_method,
            session_names=[self.session_raw],
            cam_indices=self.cam_idx_raw,
            n_cams_per_sample=-1,
            use_stereo=False,
            depth_filter='aligned',
            target_image_size_hw=self.render_config.image_size_hw,
            apply_intrinsics_fix=False,
            return_of=False,
        )

    @property
    def datasets_vis(self) -> Union[List[MultiSessionDataset], List[Path]]:
        if isinstance(self.sessions_perf[0], Path):
            return self.sessions_perf  # for merged data roots

        return [
            MultiSessionDataset(
                calibration_session_name=self.session_calib,
                calibration_method=self.calib_method,
                session_names=[session_perf_i],
                cam_indices=self.cam_idx_perf,
                n_cams_per_sample=-1,
                use_stereo=self.depth_source == 'stereo',
                depth_filter=None if self.depth_source == 'stereo' else self.depth_source,
                target_image_size_hw=self.render_config.image_size_hw,
                apply_intrinsics_fix=False,
                return_of=False,
            )
            for session_perf_i in self.sessions_perf
        ]

    @abc.abstractmethod
    def _init_scene(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> Path:
        raise NotImplementedError
