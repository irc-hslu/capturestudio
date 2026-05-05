from __future__ import annotations

from typing import Sequence, Optional, Literal

import numpy as np
import open3d as o3d

from reconstruction.primitive.pcd import RGBDImage
from reconstruction.vis.teaser.gco_nonrigid import (
    FastGCOTemporalCache,
    warp_view_to_reference_fast,
)
from reconstruction.vis.teaser.ndp_rgbd_adapter import (
    NDPConfig,
    NDPTTemporalCache,
    warp_view_to_reference_ndp,
)

_NDP_CACHE = NDPTTemporalCache()
_GCO_CACHE = FastGCOTemporalCache()


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    """Return camera center in world coordinates from a world-to-camera matrix."""
    r = w2c[:3, :3]
    t = w2c[:3, 3]
    return (-r.T @ t).astype(np.float32)


_t = 0

def _select_reference_view_index(
    views: Sequence[RGBDImage],
    target_w2c: np.ndarray,
    *,
    distance_weight: float = 1.0,
    angle_weight: float = 0.25,
) -> int:
    """
    Pick the source view whose pose is closest to the target pose.

    Score = position distance + angle penalty
    where angle penalty is based on the camera forward direction.
    """
    target_w2c = np.asarray(target_w2c, dtype=np.float32)
    target_c2w = np.linalg.inv(target_w2c)
    target_pos = target_c2w[:3, 3]
    target_fwd = target_c2w[:3, 2]
    target_fwd = _normalize(target_fwd[None])[0]

    best_i = 0
    best_score = np.inf

    for i, view in enumerate(views):
        w2c = np.asarray(view.extrinsic_w2c, dtype=np.float32)
        if w2c.shape != (4, 4):
            continue

        c2w = np.linalg.inv(w2c)
        pos = c2w[:3, 3]
        fwd = c2w[:3, 2]
        fwd = _normalize(fwd[None])[0]

        pos_dist = float(np.linalg.norm(pos - target_pos))
        cos_sim = float(np.clip(np.dot(fwd, target_fwd), -1.0, 1.0))
        angle_penalty = 1.0 - cos_sim

        score = distance_weight * pos_dist + angle_weight * angle_penalty
        if score < best_score:
            best_score = score
            best_i = i

    return best_i


def blend_point_cloud(
        views: Sequence[RGBDImage],
        target_w2c: np.ndarray,
        *,
        voxel_size_m: float = 0.01,
        angle_power: float = 2.0,
        refine_registration: Optional[Literal['pauly', 'deformation_pyramid']] = None,
) -> o3d.geometry.PointCloud:
    """Fuse synchronized RGB-D views into one blended point cloud.

    Blending is weighted by angular agreement between the source-view ray and
    the target-view ray for each 3D point. Points are then merged per voxel
    using weighted averages of position and color.

    Returns:
        Open3D tensor point cloud with:
            - positions: (N, 3) float32
            - colors:    (N, 3) float32
    """
    global _t

    views = list(views)
    if not views:
        raise ValueError("views cannot be empty")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be > 0")

    target_w2c = np.asarray(target_w2c, dtype=np.float32)
    if target_w2c.shape != (4, 4):
        raise ValueError("target_w2c must have shape (4, 4)")

    target_pos = _camera_center_from_w2c(target_w2c)

    points_chunks: list[np.ndarray] = []
    colors_chunks: list[np.ndarray] = []
    weights_chunks: list[np.ndarray] = []

    # FIX: as reference use the view that has the smallest distance with the target view
    # view_i_ref = len(views) // 2
    view_i_ref = _select_reference_view_index(views, target_w2c)
    for view_i in range(len(views)):
        # Always keep original source observation
        pcd_src = views[view_i].unproject().open3d
        orig_points = np.asarray(pcd_src.points, dtype=np.float32)
        orig_colors = np.asarray(pcd_src.colors, dtype=np.float32)

        if orig_points.ndim != 2 or orig_points.shape[1] != 3:
            raise ValueError("Unprojected point cloud points must have shape (N, 3)")
        if orig_points.shape[0] == 0:
            continue
        if orig_colors.shape != orig_points.shape:
            raise ValueError("Unprojected point cloud colors must have shape (N, 3)")

        # Default: no refinement
        final_points = orig_points
        final_colors = orig_colors

        if view_i != view_i_ref and refine_registration == "pauly":
            pcd_warped, result = warp_view_to_reference_fast(
                views[view_i],
                views[view_i_ref],
                # cache=_GCO_CACHE,
                # cache_key=(view_i, view_i_ref),
                node_stride=32,
                steps_per_stage=(16, 10, 6),
                lrs=(1e-2, 5e-3, 2e-3),
                uv_window_px=4.0,
                use_global_rigid=False,
                device="cuda",
                verbose=0,
            )
            final_points = np.asarray(pcd_warped.points, dtype=np.float32)
            final_colors = np.asarray(pcd_warped.colors, dtype=np.float32)

            _t += 1

        elif view_i != view_i_ref and refine_registration == "deformation_pyramid":
            pcd_warped, result = warp_view_to_reference_ndp(
                views[view_i],
                views[view_i_ref],
                config=NDPConfig(
                    motion="sflow",
                    levels=8,
                    samples=4000,
                    lr=1e-3,
                    iters=100,
                    trunc_m=0.04,
                    w_disp=0.05,
                    device="cuda",
                ),
                # cache=_NDP_CACHE,
                # cache_key=(view_i, view_i_ref),
                verbose=0,
            )
            final_points = np.asarray(pcd_warped.points, dtype=np.float32)
            final_colors = np.asarray(pcd_warped.colors, dtype=np.float32)

            # _t += 1

        # if _t % 10:
        #     _GCO_CACHE.clear()
        #     _NDP_CACHE.clear()

        extrinsic_w2c = np.asarray(views[view_i].extrinsic_w2c, dtype=np.float32)
        if extrinsic_w2c.shape != (4, 4):
            raise ValueError("view.extrinsic_w2c must have shape (4, 4)")

        src_pos = _camera_center_from_w2c(extrinsic_w2c)

        # IMPORTANT:
        # source ray from ORIGINAL measured geometry
        src_dirs = _normalize(orig_points - src_pos[None, :])

        # target ray from FINAL geometry that will be fused
        tgt_dirs = _normalize(final_points - target_pos[None, :])

        cos_sim = np.sum(src_dirs * tgt_dirs, axis=1)
        angular_weight = np.clip((cos_sim + 1.0) * 0.5, 0.0, 1.0) ** angle_power

        points_chunks.append(final_points)
        colors_chunks.append(final_colors)
        weights_chunks.append(angular_weight.astype(np.float32))

    if not points_chunks:
        return o3d.t.geometry.PointCloud()

    points_all = np.concatenate(points_chunks, axis=0)
    colors_all = np.concatenate(colors_chunks, axis=0)
    weights_all = np.concatenate(weights_chunks, axis=0)

    voxel_ids = np.floor(points_all / voxel_size_m).astype(np.int64)

    # inverse maps each point to its compact voxel index in [0, num_voxels)
    _, inverse = np.unique(voxel_ids, axis=0, return_inverse=True)
    num_voxels = int(inverse.max()) + 1

    weighted_points = points_all * weights_all[:, None]
    weighted_colors = colors_all * weights_all[:, None]

    # bincount-based grouped reduction
    weight_sums = np.bincount(inverse, weights=weights_all, minlength=num_voxels).astype(np.float32)
    counts = np.bincount(inverse, minlength=num_voxels).astype(np.int32)

    point_sums = np.stack(
        [
            np.bincount(inverse, weights=weighted_points[:, dim], minlength=num_voxels)
            for dim in range(3)
        ],
        axis=1,
    ).astype(np.float32)

    color_sums = np.stack(
        [
            np.bincount(inverse, weights=weighted_colors[:, dim], minlength=num_voxels)
            for dim in range(3)
        ],
        axis=1,
    ).astype(np.float32)

    valid = weight_sums > 1e-8
    if not np.any(valid):
        return o3d.t.geometry.PointCloud()

    out_points = point_sums[valid] / weight_sums[valid, None]
    out_colors = color_sums[valid] / weight_sums[valid, None]
    out_weights = (weight_sums[valid] / counts[valid]).astype(np.float32)[:, None]

    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3d.core.Tensor(out_points, dtype=o3d.core.Dtype.Float32)
    pcd.point["colors"] = o3d.core.Tensor(out_colors, dtype=o3d.core.Dtype.Float32)
    pcd.point["weights"] = o3d.core.Tensor(out_weights, dtype=o3d.core.Dtype.Float32)
    return pcd.to_legacy()
