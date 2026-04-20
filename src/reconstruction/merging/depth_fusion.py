from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reconstruction.primitive.pcd import RGBDImage


class DepthFusion(nn.Module):
    """
    Simple differentiable multi-view depth fusion layer.

    This layer refines and inpaints per-view depth maps by reprojecting all
    neighboring views into each reference view and fusing only the source
    observations that pass a small set of geometric and photometric checks.

    The implementation is intentionally simple and interpretable:

    1. Unproject every source-view pixel with its source depth.
    2. Transform the 3D point into the reference camera.
    3. Project the point into the reference image.
    4. Sample reference-view depth and RGB at the projected location.
    5. Reject or downweight contributions that are likely unreliable:
       - invalid source depth,
       - projection outside the image,
       - source pixel near a depth discontinuity,
       - source pixel near a strong RGB edge,
       - target location near a depth discontinuity,
       - target location near a strong RGB edge,
       - large RGB mismatch,
       - large depth mismatch when the reference depth is already valid,
       - source point landing behind the reference surface.
    6. Bilinearly splat the surviving source depths into the reference view.
    7. Normalize by the accumulated support weights.

    The fusion operator is a soft-weighted average rather than a hard median.
    This is deliberate: it keeps the layer differentiable with respect to the
    input depth maps while remaining easy to inspect and debug.

    Parameters
    ----------
    rgb_threshold : float
        User-facing photometric relaxation parameter in [0, 1].

        - 0.0 means very strict RGB agreement and strong suppression near RGB
          edges, so almost no fusion happens.
        - 1.0 means more relaxed RGB agreement and weaker suppression near RGB
          edges, so fusion is allowed more often.

    depth_threshold : float
        User-facing geometric relaxation parameter in [0, 1].

        - 0.0 means very strict depth agreement, strong discontinuity
          suppression, strong occlusion rejection, and high minimum support.
        - 1.0 means more relaxed geometric checks and lower minimum support.

    Attributes
    ----------
    rgb_threshold : float
        Stored user-facing RGB relaxation parameter.

    depth_threshold : float
        Stored user-facing depth relaxation parameter.
    """

    def __init__(self, rgb_threshold: float, depth_threshold: float) -> None:
        super().__init__()

        self.rgb_threshold = float(max(0.0, min(1.0, rgb_threshold)))
        self.depth_threshold = float(max(0.0, min(1.0, depth_threshold)))

        # Internal constants derived only from the 2 public thresholds.
        self.eps = 1e-8

        # RGB checks
        self.max_rgb_difference = 0.02 + 0.20 * self.rgb_threshold
        self.max_rgb_edge = 0.03 + 0.18 * self.rgb_threshold

        # Depth checks
        self.max_depth_relative_difference = 0.005 + 0.08 * self.depth_threshold
        self.max_depth_absolute_difference = 0.002 + 0.02 * self.depth_threshold
        self.max_depth_edge = 0.005 + 0.05 * self.depth_threshold
        self.max_occlusion_relative = 0.002 + 0.02 * self.depth_threshold
        self.max_occlusion_absolute = 0.001 + 0.01 * self.depth_threshold

        # Support requirement:
        # strict mode -> require more evidence,
        # relaxed mode -> allow weaker evidence.
        self.min_support_weight = 0.75 - 0.60 * self.depth_threshold

        # Fixed sharpness values keep the implementation simple.
        self.rgb_gate_scale = 40.0
        self.depth_gate_scale = 40.0
        self.edge_gate_scale = 40.0
        self.occlusion_gate_scale = 80.0

    def _make_pixel_grid(self, batch_size: int, image_height: int, image_width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ys, xs = torch.meshgrid(
            torch.arange(image_height, device=device, dtype=dtype),
            torch.arange(image_width, device=device, dtype=dtype),
            indexing="ij",
        )
        xs = xs.unsqueeze(0).expand(batch_size, image_height, image_width)
        ys = ys.unsqueeze(0).expand(batch_size, image_height, image_width)
        ones = torch.ones_like(xs)
        return xs, ys, ones

    def _make_sampling_grid(self, u: torch.Tensor, v: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
        u_normalized = 2.0 * u / max(image_width - 1, 1) - 1.0
        v_normalized = 2.0 * v / max(image_height - 1, 1) - 1.0
        return torch.stack([u_normalized, v_normalized], dim=-1)

    def _sample_map_nearest(self, input_map: torch.Tensor, sampling_grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            input_map,
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)

    def _sample_map_bilinear(self, input_map: torch.Tensor, sampling_grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            input_map,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)

    def _compute_depth_edge_map(self, depth_map: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        depth_dx = F.pad(depth_map[:, :, 1:] - depth_map[:, :, :-1], (0, 1, 0, 0))
        depth_dy = F.pad(depth_map[:, 1:, :] - depth_map[:, :-1, :], (0, 0, 0, 1))

        valid_dx = F.pad(valid_mask[:, :, 1:] * valid_mask[:, :, :-1], (0, 1, 0, 0))
        valid_dy = F.pad(valid_mask[:, 1:, :] * valid_mask[:, :-1, :], (0, 0, 0, 1))

        depth_dx = depth_dx.abs() * valid_dx
        depth_dy = depth_dy.abs() * valid_dy

        depth_edge = torch.maximum(depth_dx, depth_dy)
        depth_edge = depth_edge / depth_map.clamp_min(self.eps)
        depth_edge = torch.where(valid_mask > 0.0, depth_edge, torch.zeros_like(depth_edge))
        return depth_edge

    def _compute_rgb_edge_map(self, rgb_map: torch.Tensor) -> torch.Tensor:
        rgb_dx = F.pad(rgb_map[:, :, :, 1:] - rgb_map[:, :, :, :-1], (0, 1, 0, 0))
        rgb_dy = F.pad(rgb_map[:, :, 1:, :] - rgb_map[:, :, :-1, :], (0, 0, 0, 1))

        rgb_edge_x = rgb_dx.abs().mean(dim=1)
        rgb_edge_y = rgb_dy.abs().mean(dim=1)
        return torch.maximum(rgb_edge_x, rgb_edge_y)

    def _bilinear_splat(self, values: torch.Tensor, u: torch.Tensor, v: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, image_height, image_width = values.shape

        u0 = torch.floor(u)
        v0 = torch.floor(v)
        u1 = u0 + 1.0
        v1 = v0 + 1.0

        weight_u1 = u - u0
        weight_v1 = v - v0
        weight_u0 = 1.0 - weight_u1
        weight_v0 = 1.0 - weight_v1

        neighbors = [
            (u0, v0, weight_u0 * weight_v0),
            (u1, v0, weight_u1 * weight_v0),
            (u0, v1, weight_u0 * weight_v1),
            (u1, v1, weight_u1 * weight_v1),
        ]

        value_sum = torch.zeros(
            batch_size,
            image_height * image_width,
            device=values.device,
            dtype=values.dtype,
        )
        weight_sum = torch.zeros(
            batch_size,
            image_height * image_width,
            device=weights.device,
            dtype=weights.dtype,
        )

        for u_neighbor, v_neighbor, bilinear_weight in neighbors:
            u_index = u_neighbor.long()
            v_index = v_neighbor.long()

            in_bounds = (
                    (u_index >= 0) & (u_index < image_width) &
                    (v_index >= 0) & (v_index < image_height)
            )

            total_weight = weights * bilinear_weight * in_bounds.to(weights.dtype)

            u_index = u_index.clamp(0, image_width - 1)
            v_index = v_index.clamp(0, image_height - 1)
            linear_index = (v_index * image_width + u_index).view(batch_size, -1)

            value_sum.scatter_add_(
                1,
                linear_index,
                (values * total_weight).view(batch_size, -1),
            )
            weight_sum.scatter_add_(
                1,
                linear_index,
                total_weight.view(batch_size, -1),
            )

        return (
            value_sum.view(batch_size, image_height, image_width),
            weight_sum.view(batch_size, image_height, image_width),
        )

    def forward(self, depths: torch.Tensor, rgbs: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse depth maps across views.

        Parameters
        ----------
        depths : torch.Tensor
            Depth maps of shape ``(B, V, H, W)`` in metric units. Values less
            than or equal to zero are treated as invalid. NaNs and infinities
            are sanitized to zero.

        rgbs : torch.Tensor
            RGB images of shape ``(B, V, 3, H, W)``. Expected to be in
            ``[0, 1]``.

        intrinsics : torch.Tensor
            Camera intrinsics of shape ``(B, V, 3, 3)``.

        extrinsics : torch.Tensor
            Camera-to-world transforms of shape ``(B, V, 4, 4)``.

        Returns
        -------
        fused_depths : torch.Tensor
            Refined depth maps of shape ``(B, V, H, W)``.

        support_weights : torch.Tensor
            Total fusion support weights of shape ``(B, V, H, W)``. Larger
            values indicate stronger multi-view evidence.
        """
        batch_size, num_views, image_height, image_width = depths.shape
        device = depths.device
        dtype = depths.dtype

        depths = torch.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)
        rgbs = rgbs.to(device=device, dtype=dtype)
        intrinsics = intrinsics.to(device=device, dtype=dtype)
        extrinsics = extrinsics.to(device=device, dtype=dtype)

        world_to_camera = torch.linalg.inv(extrinsics)
        xs, ys, ones = self._make_pixel_grid(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
            device=device,
            dtype=dtype,
        )

        valid_depth = (depths > 0.0).to(dtype)

        depth_edges = []
        rgb_edges = []
        for view_idx in range(num_views):
            depth_edges.append(
                self._compute_depth_edge_map(depths[:, view_idx], valid_depth[:, view_idx])
            )
            rgb_edges.append(
                self._compute_rgb_edge_map(rgbs[:, view_idx])
            )

        depth_edges = torch.stack(depth_edges, dim=1)  # (B, V, H, W)
        rgb_edges = torch.stack(rgb_edges, dim=1)  # (B, V, H, W)

        fused_depths = torch.zeros_like(depths)
        support_weights = torch.zeros_like(depths)

        for reference_view_idx in range(num_views):
            reference_depth = depths[:, reference_view_idx]
            reference_rgb = rgbs[:, reference_view_idx]
            reference_valid = valid_depth[:, reference_view_idx]
            reference_depth_edge = depth_edges[:, reference_view_idx]
            reference_rgb_edge = rgb_edges[:, reference_view_idx]

            reference_depth_map = reference_depth.unsqueeze(1)
            reference_valid_map = reference_valid.unsqueeze(1)
            reference_depth_edge_map = reference_depth_edge.unsqueeze(1)
            reference_rgb_edge_map = reference_rgb_edge.unsqueeze(1)
            reference_rgb_map = reference_rgb

            depth_sum = reference_depth * reference_valid
            weight_sum = reference_valid.clone()

            for source_view_idx in range(num_views):
                if source_view_idx == reference_view_idx:
                    continue

                source_depth = depths[:, source_view_idx]
                source_rgb = rgbs[:, source_view_idx]
                source_valid = valid_depth[:, source_view_idx]
                source_depth_edge = depth_edges[:, source_view_idx]
                source_rgb_edge = rgb_edges[:, source_view_idx]

                if not source_valid.bool().any():
                    continue

                source_intrinsics = intrinsics[:, source_view_idx]
                reference_intrinsics = intrinsics[:, reference_view_idx]
                source_c2w = extrinsics[:, source_view_idx]
                reference_w2c = world_to_camera[:, reference_view_idx]

                source_fx = source_intrinsics[:, 0, 0].view(batch_size, 1, 1)
                source_fy = source_intrinsics[:, 1, 1].view(batch_size, 1, 1)
                source_cx = source_intrinsics[:, 0, 2].view(batch_size, 1, 1)
                source_cy = source_intrinsics[:, 1, 2].view(batch_size, 1, 1)

                source_x = (xs - source_cx) / source_fx * source_depth
                source_y = (ys - source_cy) / source_fy * source_depth
                source_points = torch.stack([source_x, source_y, source_depth, ones], dim=-1)

                source_to_reference = reference_w2c @ source_c2w
                reference_points = (
                        source_points.view(batch_size, -1, 4) @ source_to_reference.mT
                ).view(batch_size, image_height, image_width, 4)

                reference_x = reference_points[..., 0]
                reference_y = reference_points[..., 1]
                reference_z = reference_points[..., 2]

                reference_z_safe = torch.where(
                    reference_z.abs() > self.eps,
                    reference_z,
                    torch.full_like(reference_z, self.eps),
                )

                reference_fx = reference_intrinsics[:, 0, 0].view(batch_size, 1, 1)
                reference_fy = reference_intrinsics[:, 1, 1].view(batch_size, 1, 1)
                reference_cx = reference_intrinsics[:, 0, 2].view(batch_size, 1, 1)
                reference_cy = reference_intrinsics[:, 1, 2].view(batch_size, 1, 1)

                inverse_z = reference_z_safe.reciprocal()
                projected_u = reference_fx * (reference_x * inverse_z) + reference_cx
                projected_v = reference_fy * (reference_y * inverse_z) + reference_cy

                finite_projection = (
                        torch.isfinite(projected_u) &
                        torch.isfinite(projected_v) &
                        torch.isfinite(reference_z)
                ).to(dtype)

                projection_in_front = (reference_z > self.eps).to(dtype)

                sampling_grid = self._make_sampling_grid(
                    u=projected_u,
                    v=projected_v,
                    image_height=image_height,
                    image_width=image_width,
                )

                sampled_reference_depth = self._sample_map_nearest(reference_depth_map, sampling_grid)
                sampled_reference_valid = self._sample_map_nearest(reference_valid_map, sampling_grid)
                sampled_reference_depth_edge = self._sample_map_nearest(reference_depth_edge_map, sampling_grid)
                sampled_reference_rgb_edge = self._sample_map_nearest(reference_rgb_edge_map, sampling_grid)

                sampled_reference_rgb = F.grid_sample(
                    reference_rgb_map,
                    sampling_grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=True,
                )

                rgb_difference = (source_rgb - sampled_reference_rgb).abs().mean(dim=1)
                rgb_gate = torch.sigmoid(
                    (self.max_rgb_difference - rgb_difference) * self.rgb_gate_scale
                )

                source_depth_edge_gate = torch.sigmoid(
                    (self.max_depth_edge - source_depth_edge) * self.edge_gate_scale
                )
                source_rgb_edge_gate = torch.sigmoid(
                    (self.max_rgb_edge - source_rgb_edge) * self.edge_gate_scale
                )

                reference_depth_edge_gate = torch.sigmoid(
                    (self.max_depth_edge - sampled_reference_depth_edge) * self.edge_gate_scale
                )
                reference_rgb_edge_gate = torch.sigmoid(
                    (self.max_rgb_edge - sampled_reference_rgb_edge) * self.edge_gate_scale
                )

                depth_difference_threshold = torch.clamp(
                    self.max_depth_relative_difference * sampled_reference_depth.abs(),
                    min=self.max_depth_absolute_difference,
                )
                depth_difference = (reference_z - sampled_reference_depth).abs()

                # If the reference depth is valid, require geometric agreement.
                # If the reference depth is invalid, allow source-driven fill-in.
                depth_gate = torch.sigmoid(
                    (depth_difference_threshold - depth_difference) * self.depth_gate_scale
                )
                depth_gate = sampled_reference_valid * depth_gate + (1.0 - sampled_reference_valid)

                # Strong one-sided occlusion check:
                # if the projected source point lies behind the known reference surface,
                # suppress it aggressively.
                occlusion_threshold = torch.clamp(
                    self.max_occlusion_relative * sampled_reference_depth.abs(),
                    min=self.max_occlusion_absolute,
                )
                behind_distance = reference_z - sampled_reference_depth
                occlusion_gate = torch.sigmoid(
                    (occlusion_threshold - behind_distance) * self.occlusion_gate_scale
                )
                occlusion_gate = sampled_reference_valid * occlusion_gate + (1.0 - sampled_reference_valid)

                source_weight = (
                        source_valid *
                        finite_projection *
                        projection_in_front *
                        source_depth_edge_gate *
                        source_rgb_edge_gate *
                        reference_depth_edge_gate *
                        reference_rgb_edge_gate *
                        rgb_gate *
                        depth_gate *
                        occlusion_gate
                )

                splatted_depth_sum, splatted_weight_sum = self._bilinear_splat(
                    values=reference_z,
                    u=projected_u,
                    v=projected_v,
                    weights=source_weight,
                )

                depth_sum = depth_sum + splatted_depth_sum
                weight_sum = weight_sum + splatted_weight_sum

            fused_depth = depth_sum / weight_sum.clamp_min(self.eps)

            # Keep original valid pixels if support is weak.
            enough_support = weight_sum > self.min_support_weight
            fused_depth = torch.where(
                enough_support,
                fused_depth,
                reference_depth * reference_valid,
            )

            fused_depth = torch.where(
                reference_valid > 0.0,
                fused_depth,
                torch.where(enough_support, fused_depth, torch.zeros_like(fused_depth)),
            )

            final_support = torch.where(enough_support, weight_sum, torch.zeros_like(weight_sum))

            fused_depths[:, reference_view_idx] = fused_depth
            support_weights[:, reference_view_idx] = final_support

        return fused_depths, support_weights


DEPTH_FUSION_ = None


def fuse_depth_maps(images: List[RGBDImage]) -> List[RGBDImage]:
    global DEPTH_FUSION_
    if DEPTH_FUSION_ is None:
        DEPTH_FUSION_ = DepthFusion(
            rgb_threshold=0.25,  # a bit looser with color mismatches
            depth_threshold=0.1,  # stricter around discontinuities
        )

    depth_fused, _ = DEPTH_FUSION_.forward(
        depths=torch.stack([torch.from_numpy(_.depth) for _ in images], dim=0)[None].cuda(),
        rgbs=torch.stack([torch.from_numpy(_.rgb) for _ in images], dim=0).permute(0, 3, 1, 2)[None].cuda(),
        intrinsics=torch.stack([torch.from_numpy(_.intrinsic) for _ in images], dim=0)[None].cuda(),
        extrinsics=torch.stack([torch.from_numpy(np.linalg.inv(_.extrinsic_w2c)) for _ in images], dim=0)[None].cuda(),
    )

    depth_fused = depth_fused[0].detach().cpu().numpy().squeeze()
    for i, image in enumerate(images):
        image.depth = depth_fused[i]

    return images
