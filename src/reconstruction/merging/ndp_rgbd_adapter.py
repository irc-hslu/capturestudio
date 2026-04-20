from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Hashable, Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from reconstruction.primitive.pcd import RGBDImage


def _normalize_np(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _sample_indices_far_from_center(
    uv: np.ndarray,
    k: int,
    center_uv: np.ndarray,
    far_ratio: float = 0.7,
) -> np.ndarray:
    """
    Fast deterministic sampler:
    - takes most points from the far periphery
    - keeps some uniform coverage for stability
    """
    n = uv.shape[0]
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)

    k_far = int(round(k * far_ratio))
    k_far = max(0, min(k_far, k))
    k_uni = k - k_far

    # radial distance from optical center / principal point
    r2 = np.sum((uv - center_uv[None, :]) ** 2, axis=1)

    # deterministic: take farthest k_far points
    if k_far > 0:
        idx_far = np.argpartition(r2, n - k_far)[-k_far:]
    else:
        idx_far = np.zeros((0,), dtype=np.int64)

    # fill the rest with evenly spaced points for stability / coverage
    if k_uni > 0:
        mask = np.ones(n, dtype=bool)
        mask[idx_far] = False
        pool = np.flatnonzero(mask)
        if pool.size <= k_uni:
            idx_uni = pool
        else:
            idx_uni = pool[np.linspace(0, pool.size - 1, num=k_uni, dtype=np.int64)]
    else:
        idx_uni = np.zeros((0,), dtype=np.int64)

    idx = np.concatenate([idx_far, idx_uni], axis=0)
    return idx.astype(np.int64)

def _sample_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return np.random.choice(n, size=k, replace=False).astype(np.int64)


def _extract_valid_points_colors(rgbd: RGBDImage) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull a stable valid mask from our PixelPoints implementation.

    Accessing normals triggers the same validity cleanup that our codebase uses
    for Open3D export (boundary trim + outlier rejection).
    """
    pp = rgbd.unproject()
    _ = pp.open3d
    valid = np.asarray(pp.valid, dtype=bool)
    flat_idx = np.flatnonzero(valid.reshape(-1))

    H, W = valid.shape
    vv = flat_idx // W
    uu = flat_idx % W
    uv = np.stack([uu, vv], axis=1).astype(np.float32)

    pts = np.asarray(pp.points, dtype=np.float32).reshape(-1, 3)[flat_idx]
    cols = np.asarray(pp.colors, dtype=np.float32).reshape(-1, 3)[flat_idx]
    return pts, cols, uv


def _camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    r = w2c[:3, :3]
    t = w2c[:3, 3]
    return (-r.T @ t).astype(np.float32)


def _robust_nn_error(
    src: np.ndarray,
    tgt: np.ndarray,
    max_samples: int = 4096,
    clip_m: float = 0.03,
) -> float:
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return np.inf

    src_idx = _sample_indices(src.shape[0], max_samples)
    tgt_idx = _sample_indices(tgt.shape[0], max_samples)

    xs = np.asarray(src[src_idx], dtype=np.float64)
    ys = np.asarray(tgt[tgt_idx], dtype=np.float64)

    tree = cKDTree(ys)
    dists, _ = tree.query(xs, k=1)
    dists = np.minimum(dists, clip_m)
    return float(np.mean(dists))


def _truncated_chamfer_distance(
        x: torch.Tensor,
        y: torch.Tensor,
        trunc: float,
) -> torch.Tensor:
    """
    Symmetric truncated Chamfer on unbatched clouds x:(N,3), y:(M,3).
    """
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(float("inf"), device=x.device if x.numel() else y.device)
    d = torch.cdist(x[None], y[None]).squeeze(0)  # (N, M)
    dx = d.min(dim=1).values.clamp(max=trunc)
    dy = d.min(dim=0).values.clamp(max=trunc)
    return 0.5 * (dx.mean() + dy.mean())


class MLP(nn.Module):
    def __init__(self, depth: int, width: int) -> None:
        super().__init__()
        layers = []
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NDPLayer(nn.Module):
    """
    Minimal Deformation Pyramid layer adapted from the public repo, but
    specialized for our setting:

    - default motion='sflow' because our views are already calibrated;
      we want a small corrective warp, not free global re-registration
    - identity-biased outputs, matching the repo's small-MLP-scale idea
    """

    def __init__(
            self,
            depth: int,
            width: int,
            k0: int,
            level_idx: int,
            *,
            motion: str = "sflow",
            mlp_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        assert motion in {"sflow", "SE3", "Sim3"}
        self.motion = motion
        self.k0 = int(k0)
        self.level_idx = int(level_idx)
        self.mlp_scale = float(mlp_scale)

        self.input = nn.Sequential(nn.Linear(6, width), nn.ReLU(inplace=True))
        self.mlp = MLP(depth=depth, width=width)
        self.trn_branch = nn.Linear(width, 3)

        if motion in {"SE3", "Sim3"}:
            self.rot_branch = nn.Linear(width, 3)  # Euler
        if motion == "Sim3":
            self.scale_branch = nn.Linear(width, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def posenc(self, x: torch.Tensor) -> torch.Tensor:
        # Same simplified encoding as their public code: one frequency per level
        # with increasing frequency deeper in the pyramid.
        freq = 2.0 ** (self.level_idx + self.k0)
        x_pos = x[:, 0:1]
        y_pos = x[:, 1:2]
        z_pos = x[:, 2:3]
        return torch.cat(
            [
                torch.sin(freq * x_pos), torch.cos(freq * x_pos),
                torch.sin(freq * y_pos), torch.cos(freq * y_pos),
                torch.sin(freq * z_pos), torch.cos(freq * z_pos),
            ],
            dim=1,
        )

    @staticmethod
    def euler_to_rotmat(e: torch.Tensor) -> torch.Tensor:
        # e: (N,3)
        rx, ry, rz = e[:, 0], e[:, 1], e[:, 2]
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        one = torch.ones_like(rx)
        zero = torch.zeros_like(rx)

        Rx = torch.stack([
            torch.stack([one, zero, zero], dim=1),
            torch.stack([zero, cx, -sx], dim=1),
            torch.stack([zero, sx, cx], dim=1),
        ], dim=1)

        Ry = torch.stack([
            torch.stack([cy, zero, sy], dim=1),
            torch.stack([zero, one, zero], dim=1),
            torch.stack([-sy, zero, cy], dim=1),
        ], dim=1)

        Rz = torch.stack([
            torch.stack([cz, -sz, zero], dim=1),
            torch.stack([sz, cz, zero], dim=1),
            torch.stack([zero, zero, one], dim=1),
        ], dim=1)

        return Rz @ Ry @ Rx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        f = self.posenc(x)
        f = self.input(f)
        f = self.mlp(f)

        t = self.mlp_scale * self.trn_branch(f)

        if self.motion == "sflow":
            x_out = x + t
            aux = {"translation": t}
            return x_out, aux

        R = self.euler_to_rotmat(self.mlp_scale * self.rot_branch(f))
        x_rot = (R @ x[..., None]).squeeze(-1)

        if self.motion == "SE3":
            x_out = x_rot + t
            aux = {"translation": t, "rotation": R}
            return x_out, aux

        s = 1.0 + self.mlp_scale * self.scale_branch(f)
        x_out = s * x_rot + t
        aux = {"translation": t, "rotation": R, "scale": s}
        return x_out, aux


class DeformationPyramid(nn.Module):
    def __init__(
            self,
            n_hierarchy: int,
            depth: int,
            width: int,
            k0: int,
            *,
            motion: str = "sflow",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                NDPLayer(depth=depth, width=width, k0=k0, level_idx=i + 1, motion=motion)
                for i in range(n_hierarchy)
            ]
        )

    @property
    def n_hierarchy(self) -> int:
        return len(self.layers)

    def gradient_setup(self, optimized_level: int) -> None:
        for i, layer in enumerate(self.layers):
            req = (i == optimized_level) or (optimized_level < 0)
            for p in layer.parameters():
                p.requires_grad = req

    def warp(
            self,
            x: torch.Tensor,
            *,
            max_level: Optional[int] = None,
            min_level: int = 0,
    ) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
        if max_level is None:
            max_level = self.n_hierarchy - 1
        data: dict[int, dict[str, torch.Tensor]] = {}
        for i in range(min_level, max_level + 1):
            x, aux = self.layers[i](x)
            data[i] = aux
        return x, data


@dataclass
class NDPConfig:
    samples: int = 4000
    motion: str = "sflow"  # better for already-calibrated RGBD
    depth: int = 3
    width: int = 128
    levels: int = 6
    k0: int = -5
    lr: float = 5e-3
    iters: int = 120
    max_break_count: int = 8
    break_threshold_ratio: float = 5e-4
    trunc_m: float = 0.03
    w_disp: float = 0.02  # keep warp small
    w_level_decay: float = 0.5  # stronger regularization on finer levels
    accept_improvement_ratio: float = 0.98
    device: str = "cuda"


@dataclass
class NDPWarmStart:
    state_dict: dict[str, Any]
    config: dict[str, Any]


class NDPTTemporalCache:
    def __init__(self) -> None:
        self._cache: dict[Hashable, NDPWarmStart] = {}

    def get(self, key: Hashable) -> Optional[NDPWarmStart]:
        return self._cache.get(key)

    def put(self, key: Hashable, state: NDPWarmStart) -> None:
        self._cache[key] = state

    def clear(self) -> None:
        self._cache.clear()


class RGBDDeformationPyramidRegistrar:
    """
    Deformation-Pyramid-style registration adapted to our RGBD multiview case.

    Key differences from the shape_transfer.py file:
    - no independent centering of source and target (preserves calibration)
    - point-cloud input comes from RGBDImage.unproject()
    - default motion is small scene flow, not free Sim(3)
    - a do-no-harm acceptance gate prevents making fusion worse
    - optional warm-start reuse across timesteps
    """

    def __init__(
            self,
            source_rgbd: RGBDImage,
            target_rgbd: RGBDImage,
            *,
            config: Optional[NDPConfig] = None,
    ) -> None:
        self.config = config or NDPConfig()
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        self.src_points_np, self.src_colors_np, self.src_uv = _extract_valid_points_colors(source_rgbd)
        self.src_center_uv = np.asarray(
            [source_rgbd.intrinsic[0, 2], source_rgbd.intrinsic[1, 2]],
            dtype=np.float32,
        )
        self.tgt_points_np, _, self.tgt_uv = _extract_valid_points_colors(target_rgbd)
        self.tgt_center_uv = np.asarray(
            [target_rgbd.intrinsic[0, 2], target_rgbd.intrinsic[1, 2]],
            dtype=np.float32,
        )

        # Shared origin for numerical stability without destroying the existing alignment.
        if self.tgt_points_np.shape[0] > 0:
            self.origin_np = self.tgt_points_np.mean(axis=0, keepdims=True).astype(np.float32)
        else:
            self.origin_np = np.zeros((1, 3), dtype=np.float32)

        self.src_points0_np = (self.src_points_np - self.origin_np).astype(np.float32)
        self.tgt_points0_np = (self.tgt_points_np - self.origin_np).astype(np.float32)

        self.src_points0_t = torch.from_numpy(self.src_points0_np).to(self.device)
        self.tgt_points0_t = torch.from_numpy(self.tgt_points0_np).to(self.device)

        self.model = DeformationPyramid(
            n_hierarchy=self.config.levels,
            depth=self.config.depth,
            width=self.config.width,
            k0=self.config.k0,
            motion=self.config.motion,
        ).to(self.device)

    # def _sample_pair(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     src_idx = _sample_indices(src.shape[0], self.config.samples)
    #     tgt_idx = _sample_indices(tgt.shape[0], self.config.samples)
    #     return src[src_idx], tgt[tgt_idx]

    def _sample_pair(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Bias source sampling toward the image periphery, where your distortion is stronger.
        src_idx = _sample_indices_far_from_center(
            self.src_uv,
            self.config.samples,
            self.src_center_uv,
            far_ratio=0.7,
        )

        # Keep target simple and stable.
        tgt_idx = _sample_indices_far_from_center(
            self.tgt_uv,
            self.config.samples,
            self.tgt_center_uv,
            far_ratio=0.7,
        )

        return src[src_idx], tgt[tgt_idx]

    def _fit(self, warm_start: Optional[NDPWarmStart] = None, verbose: int = 0) -> dict[str, Any]:
        if warm_start is not None:
            try:
                self.model.load_state_dict(warm_start.state_dict, strict=True)
            except Exception:
                pass

        s_sample, t_sample = self._sample_pair(self.src_points0_t, self.tgt_points0_t)

        history: list[float] = []
        for level in range(self.model.n_hierarchy):
            self.model.gradient_setup(optimized_level=level)
            optimizer = torch.optim.AdamW(self.model.layers[level].parameters(), lr=self.config.lr)

            break_counter = 0
            loss_prev = 1e10

            for it in range(self.config.iters):
                s_warped, aux = self.model.warp(s_sample, max_level=level, min_level=level)
                loss_cd = _truncated_chamfer_distance(s_warped, t_sample, trunc=self.config.trunc_m)

                # Small-displacement prior. Stronger on finer levels.
                disp = s_warped - s_sample
                level_scale = self.config.w_level_decay ** max(0, level)
                loss_disp = (disp.square().sum(dim=1).mean())

                loss = loss_cd + (self.config.w_disp / max(level_scale, 1e-6)) * loss_disp

                loss_val = float(loss.detach().cpu())
                history.append(loss_val)

                if verbose and (it == 0 or (it + 1) % 20 == 0 or (it + 1) == self.config.iters):
                    print(f"[NDP][L{level}] iter {it + 1}/{self.config.iters} "
                          f"loss={loss_val:.6f} cd={float(loss_cd.detach().cpu()):.6f} "
                          f"disp={float(loss_disp.detach().cpu()):.6f}")

                if loss_val < 1e-5:
                    break
                if abs(loss_prev - loss_val) < max(loss_prev, 1e-8) * self.config.break_threshold_ratio:
                    break_counter += 1
                else:
                    break_counter = 0
                if break_counter >= self.config.max_break_count:
                    break
                loss_prev = loss_val

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # Carry the current warped samples to the next, finer level.
            s_sample = s_warped.detach()

        self.model.gradient_setup(optimized_level=-1)
        warped_all0_t, _ = self.model.warp(self.src_points0_t)
        warped_all_np = warped_all0_t.detach().cpu().numpy().astype(np.float32) + self.origin_np

        warm = NDPWarmStart(
            state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            config=asdict(self.config),
        )
        return {
            "warped_points": warped_all_np,
            "loss_history": history,
            "warm_start": warm,
        }

    def register(
            self,
            *,
            warm_start: Optional[NDPWarmStart] = None,
            cache: Optional[NDPTTemporalCache] = None,
            cache_key: Optional[Hashable] = None,
            verbose: int = 0,
    ) -> tuple[o3d.geometry.PointCloud, dict[str, Any]]:
        if warm_start is None and cache is not None and cache_key is not None:
            warm_start = cache.get(cache_key)

        result = self._fit(warm_start=warm_start, verbose=verbose)
        warped_points = np.asarray(result["warped_points"], dtype=np.float32)

        # Do-no-harm gate.
        before = _robust_nn_error(self.src_points_np, self.tgt_points_np)
        after = _robust_nn_error(warped_points, self.tgt_points_np)
        accepted = after < self.config.accept_improvement_ratio * before

        final_points = warped_points if accepted else self.src_points_np
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd.colors = o3d.utility.Vector3dVector(self.src_colors_np)

        result["accepted"] = accepted
        result["nn_error_before"] = before
        result["nn_error_after"] = after if accepted else before

        if cache is not None and cache_key is not None:
            cache.put(cache_key, result["warm_start"])

        return pcd, result


def warp_view_to_reference_ndp(
        view: RGBDImage,
        ref_view: RGBDImage,
        *,
        config: Optional[NDPConfig] = None,
        warm_start: Optional[NDPWarmStart] = None,
        cache: Optional[NDPTTemporalCache] = None,
        cache_key: Optional[Hashable] = None,
        verbose: int = 0,
) -> tuple[o3d.geometry.PointCloud, dict[str, Any]]:
    registrar = RGBDDeformationPyramidRegistrar(view, ref_view, config=config)
    return registrar.register(
        warm_start=warm_start,
        cache=cache,
        cache_key=cache_key,
        verbose=verbose,
    )
