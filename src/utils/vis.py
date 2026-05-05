import copy
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union, Literal

import PIL
import cv2
import easydict
import numpy as np
import open3d as o3d
import torch
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from kornia.geometry import quaternion_to_rotation_matrix
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt, cm
from matplotlib import rcParams
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from torch import nn
from torchvision.transforms import transforms, ToPILImage
from torchvision.utils import flow_to_image

from utils.misc import PathUtils, log, Str

# Set the font to JetBrains Mono
font_paths_ = [
    "/usr/share/fonts/truetype/JetBrainsMono-Regular.ttf",
    str(PathUtils.resources_path() / "fonts/JetBrainsMono-Regular.ttf"),
]
for font_path_ in font_paths_:
    if Path(font_path_).exists():
        fm.fontManager.addfont(font_path_)
        font_prop_ = fm.FontProperties(fname=font_path_)
        rcParams["font.family"] = font_prop_.get_name()
        log(f"[matplotlib::font_manager] Using font: {font_path_}", "debug")
        break
logging.getLogger("matplotlib.backends").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
ColorT = str or Tuple[int, int, int] or Tuple[int, int, int, int]


class VisUtils:
    """Visualization utilities for optical flow."""

    @classmethod
    def striped_teaser_image(
            cls,
            color: np.ndarray,  # (H,W,3) uint8
            depth: np.ndarray,  # (H,W,3) uint8
            normals: np.ndarray,  # (H,W,3) uint8
            *,
            widths=(0.06, 0.06),  # fractional widths along TR→BL (first, second)
            spacing=0.02,  # fractional gap between stripes (0 = adjacent)
            center=0.50,  # 0..1 position along s (0=TR, 1=BL)
            feather_px=2.0,  # soft stripe edges (0 = hard)
            order=("depth", "normals"),  # which modality is nearer TR, then the other
            # boundary controls
            draw_boundaries=True,
            boundary_color=(40, 40, 40),  # dark gray (BGR/RGB agnostic for pure drawing)
            boundary_thickness_px=2.0,  # line thickness (across s)
            boundary_len_frac=0.3,  # length along r as a fraction of diagonal
            boundary_center_frac=0.50,  # where to center the ticks along r (0=TL, 1=BR)
            boundary_opacity=1.0  # 0..1
    ) -> np.ndarray:
        """
        Compose a single image with two diagonal stripes (TR→BL) over a color base,
        plus short boundary ticks at:
          • color → first stripe (TR side)
          • between first & second stripes (mid boundary)
          • second stripe → color (BL side)

        If `spacing == 0.0`, the mid boundary is drawn ONCE (not doubled),
        edges are still drawn.

        Returns:
            (H,W,3) uint8 image.
        """
        assert color.shape == depth.shape == normals.shape and color.ndim == 3 and color.shape[2] == 3
        H, W, _ = color.shape

        # ---- coordinate grids
        # s runs along the stripe direction (0 at top-right → L at bottom-left)
        # r runs along the orthogonal diagonal (0 at top-left → R at bottom-right)
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        s = (W - 1 - x) + y
        r = x + y
        L = float(W + H - 2)  # max value of s (and r)
        R = L

        # ---- stripe layout along s
        w1_px = max(0.0, float(widths[0]) * L)
        w2_px = max(0.0, float(widths[1]) * L)
        gap_px = max(0.0, float(spacing) * L)
        block_px = w1_px + gap_px + w2_px
        c = np.clip(center, 0.0, 1.0) * L
        start = c - block_px / 2.0

        # intervals [a,b] along s for the two stripes
        a1, b1 = start, start + w1_px  # first stripe (nearer TR)
        a2, b2 = b1 + gap_px, b1 + gap_px + w2_px  # second stripe (toward BL)

        # ---- soft band masks (0..1)
        def soft_band(sv, a, b, feather):
            if feather <= 0:
                return ((sv >= a) & (sv <= b)).astype(np.float32)
            f = float(feather)
            left = np.clip((sv - (a - f)) / f, 0.0, 1.0)
            right = np.clip(((b + f) - sv) / f, 0.0, 1.0)
            return (left * right).astype(np.float32)

        mask_first = soft_band(s, a1, b1, feather_px)
        mask_second = soft_band(s, a2, b2, feather_px)

        # ---- assign modalities to stripes by order
        first, second = order
        assert first in ("depth", "normals") and second in ("depth", "normals") and first != second
        src = {"depth": depth.astype(np.float32), "normals": normals.astype(np.float32)}

        mask_depth = mask_first if first == "depth" else mask_second
        mask_normals = mask_first if first == "normals" else mask_second
        mask_color = 1.0 - np.maximum(mask_depth, mask_normals)

        # ---- composite
        out = (mask_color[..., None] * color.astype(np.float32) +
               mask_depth[..., None] * src["depth"] +
               mask_normals[..., None] * src["normals"])

        # ---- boundaries (ticks), short segments centered along r
        if draw_boundaries and boundary_thickness_px > 0 and boundary_opacity > 0:
            # Always draw edge boundaries: TR edge at s=a1, BL edge at s=b2
            s_lines = [a1, b2]

            # Mid boundary logic:
            #   spacing == 0.0  → exactly one boundary at s=b1 (== a2)
            #   spacing  > 0.0  → two boundaries at s=b1 and s=a2 (around the gap)
            if gap_px <= 1e-9:
                s_lines += [b1]  # single mid line
            else:
                s_lines += [b1, a2]  # double mid lines flanking the gap

            # Limit ticks to a short window along r around r_center
            r_center = np.clip(boundary_center_frac, 0.0, 1.0) * R
            if boundary_len_frac <= 1.0:
                half_len = 0.5 * max(1.0, boundary_len_frac * R)  # fraction of diag
            else:
                half_len = 0.5 * boundary_len_frac  # interpret as pixels if >1
            r_window = np.abs(r - r_center) <= half_len

            # Thickness across s (± half_thick)
            half_thick = max(0.5, float(boundary_thickness_px) / 2.0)

            # Keep ticks confined to the overall stripe block (so they don't run into color-only area elsewhere)
            in_block = (s >= (a1 - half_thick)) & (s <= (b2 + half_thick))

            # Build boolean mask for boundaries
            boundary_mask = np.zeros((H, W), dtype=bool)
            for s0 in s_lines:
                boundary_mask |= ((np.abs(s - s0) <= half_thick) & r_window & in_block)

            if boundary_mask.any():
                bc = np.array(boundary_color, dtype=np.float32).reshape(1, 1, 3)
                bm = boundary_mask.astype(np.float32)[..., None] * float(boundary_opacity)
                out = (1.0 - bm) * out + bm * bc

        return np.clip(out, 0, 255).astype(np.uint8)

    @classmethod
    def color_tensor(cls, shape: List[int], color: ColorT, norm: bool = False) -> torch.Tensor:
        """Create a tensor of a single color.

        Parameters:
        ----------
        shape: list
            The shape of the tensor.
        color: str or tuple
            The color of the tensor in hex or RGB(A) format. If the color has an alpha channel, the tensor will have
            an alpha channel as well.
        norm: bool
            Whether to normalize the tensor to [-1, 1]. Default: False.

        Returns:
        -------
        torch.Tensor
            The tensor of the given color and shape.
        """
        if isinstance(color, str):
            color = Str(color).rgb()
        if len(shape) == 2:
            shape = [len(color), *shape]
        assert shape[0] == len(color), f'Color dimension {len(color)} does not match tensor dimension {shape[0]}'
        color_tensor = torch.tensor(color)
        if norm:
            color_tensor = color_tensor / 255.0
        return (torch.ones(*shape).transpose(0, -1).contiguous() * color_tensor).transpose(0, -1).contiguous()

    @classmethod
    def color_pick(cls, img: torch.Tensor, coords: Tuple[int or float, int or float] = (0, 0)) -> ColorT:
        """Pick the color of a pixel in an image.

        Parameters:
        ----------
        img: torch.Tensor
            The image tensor, with shape [3, height, width].
        coords: tuple
            The coordinates of the pixel to pick, with shape [2]. If the coordinates are floats, they will be treated
            as relative coordinates and multiplied by the image size. Default: (0, 0).

        Returns:
        -------
        tuple
            The color of the pixel in RGB format.
        """
        if isinstance(coords[0], float) or isinstance(coords[1], float):
            coords = (
                int(coords[0] * img.shape[1]),
                int(coords[1] * img.shape[2])
            )
        color = img[:, coords[0], coords[1]] if img.ndim == 3 else img[0, :, coords[0], coords[1]]
        if color.max() <= 1:
            if color.min() < 0:
                color = (color + 1) / 2
            color = color * 255
        return tuple(color.int().tolist())

    @classmethod
    def compute_mean_look_at_and_up(cls, tvecs: torch.Tensor, rotmats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a robust mean look-at point and up vector from a set of cameras.

        :param tvecs: (N,3) tensor of camera centers in world coordinates.
        :param rotmats: (N,3,3) tensor of camera-to-world rotation matrices.
        :return: (mean_look_at_point, mean_up_vector)
            mean_look_at_point: (3,) tensor
            mean_up_vector: (3,) tensor (normalized)
        """
        device = tvecs.device
        tvecs = tvecs.to(dtype=torch.float, device=device)
        rotmats = rotmats.to(dtype=torch.float, device=device)
        N = tvecs.shape[0]

        # 1) Compute forward and up vectors in world coordinates
        # c2w = R^T for each camera
        # c2w = rotmats  # (N,3,3)

        # Forward direction in camera coords is (0,0,-1), so in world:
        # forward_i = c2w * [0,0,-1] = -c2w[:, :, 2]
        forward_vectors = -rotmats[:, :, 2]  # shape (N, 3)
        forward_vectors = forward_vectors / forward_vectors.norm(dim=1, keepdim=True)

        # Up direction in camera coords is (0,1,0), so in world:
        # up_i = c2w[:, :, 1]
        up_vectors = rotmats[:, :, 1]  # shape (N, 3)
        up_mean = up_vectors.mean(dim=0)
        up_mean = up_mean / up_mean.norm()

        # 2) Compute the best-fit point P that minimizes sum of squared distances to each line:
        # Each line: P(t) = C_i + lambda_i * forward_i
        # We want a point P that is closest on average to all lines.
        #
        # The normal equation for this problem:
        # sum_i (I - f_i f_i^T)(P - C_i) = 0
        # sum_i (I - f_i f_i^T) P = sum_i (I - f_i f_i^T) C_i
        # Let M = sum_i (I - f_i f_i^T) and b = sum_i (I - f_i f_i^T) C_i
        # Then P = M^-1 b

        eye = torch.eye(3, device=device, dtype=torch.float32)
        M = torch.zeros((3, 3), device=device, dtype=torch.float32)
        b = torch.zeros((3,), device=device, dtype=torch.float32)

        for i in range(N):
            f = forward_vectors[i]
            # outer product f f^T
            f_outer = f.unsqueeze(1) @ f.unsqueeze(0)  # (3,3)
            A = eye - f_outer
            M += A
            b += A @ tvecs[i]

        # Solve M P = b
        mean_look_at_point = torch.linalg.solve(M, b)  # shape (3,)

        return mean_look_at_point, up_mean

    @classmethod
    def get_world2view_matrix_3dgs(cls, rotmats: torch.Tensor, tvecs: torch.Tensor) -> torch.Tensor:
        rt = torch.zeros(rotmats.shape[0], 4, 4, device=rotmats.device, dtype=rotmats.dtype)
        rt[:, :3, :3] = rotmats.transpose(-1, -2)
        rt[:, :3, 3:] = -rotmats.transpose(-1, -2) @ tvecs[..., None]
        rt[:, 3, 3] = 1.0
        return rt


    @classmethod
    def focal2fov(cls, focal: float, pixels: int) -> float:
        return 2 * math.atan(pixels / (2 * focal))

    @classmethod
    def create_cameras_3dgs(cls, intrinsics: torch.Tensor, rotmats: torch.Tensor, tvecs: torch.Tensor, image_size: (int, int), znear: float = 0.01, zfar: float = 100.0) -> List[easydict.EasyDict]:
        H, W = image_size
        focal_x, focal_y = intrinsics[..., 0, 0].detach().cpu().tolist(), intrinsics[..., 1, 1].detach().cpu().tolist()
        fov_x, fov_y = [cls.focal2fov(focal_x_i, W) for focal_x_i in focal_x], [cls.focal2fov(focal_y_i, H) for focal_y_i in focal_y]
        world_view_transform = cls.get_world2view_matrix_3dgs(rotmats, tvecs).transpose(-2, -1).cuda()
        from utils.splat import GSUtils
        projection_matrix = torch.from_numpy(GSUtils.get_projection_matrix_3dgs(intrinsics.squeeze().detach().cpu().numpy(), image_size_hw=image_size, znear=znear, zfar=zfar)).transpose(-2, -1).cuda()
        full_proj_transform = world_view_transform.bmm(projection_matrix)
        camera_center = world_view_transform.inverse()[:, 3, :3]
        return [
            easydict.EasyDict(
                world_view_transform=world_view_transform[_],
                projection_matrix=projection_matrix[_],
                full_proj_transform=full_proj_transform[_],
                camera_center=camera_center[_],
                fov_x=fov_x[_],
                fov_y=fov_y[_],
                H=H,
                W=W
            )
            for _ in range(len(intrinsics))
        ]

    @classmethod
    def create_cameras_o3d(cls,
                           intrinsics: Union[torch.Tensor, np.ndarray],
                           extrinsics: Union[torch.Tensor, np.ndarray],
                           image_size: (int, int),
                           is_c2w: bool = True) -> List[o3d.camera.PinholeCameraParameters]:
        """
        Create Open3D compatible cameras from intrinsics and extrinsics.

        Parameters
        ----------
        intrinsics : torch.Tensor
            Shape (B, 3, 3), camera intrinsic matrices.
        extrinsics : torch.Tensor
            Shape (B, 4, 4), camera-to-world extrinsics.
            Kaolin expects world-to-camera (R,T).
        image_size : (H, W)
            The height and width of the image.
        is_c2w : bool
            If True, the input extrinsics are camera-to-world (R,T).

        Returns
        -------
        cameras : list of o3d.camera.PinholeCameraParameters
        """
        cameras = []
        H, W = image_size
        extrinsics = extrinsics if is_c2w else extrinsics.inverse()
        for intrinsic, extrinsic in zip(intrinsics, extrinsics):
            cam = o3d.camera.PinholeCameraParameters()
            cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=W, height=H,
                fx=intrinsic[0, 0].item(),
                fy=intrinsic[1, 1].item(),
                cx=intrinsic[0, 2].item(),
                cy=intrinsic[1, 2].item()
            )
            cam.extrinsic = extrinsic.cpu().numpy() if isinstance(extrinsic, torch.Tensor) else extrinsic
            cameras.append(cam)
        return cameras

    @classmethod
    def create_cameras_o3d_from_p3d(cls, cameras: PerspectiveCameras) -> List[o3d.camera.PinholeCameraParameters]:
        """
        Create Open3D compatible cameras from PyTorch3D cameras.

        Parameters
        ----------
        cameras : pytorch3d.renderer.PerspectiveCameras
            The PyTorch3D cameras to convert.

        Returns
        -------
        cameras : list of o3d.camera.PinholeCameraParameters
            The Open3D compatible cameras.
        """
        intrinsics = torch.eye(3)[None].repeat(len(cameras), 1, 1)
        intrinsics[:, :2, :2] = torch.diag_embed(cameras.focal_length).cpu()
        intrinsics[:, :2, 2] = cameras.principal_point.cpu()
        extrinsics = cameras.get_world_to_view_transform().inverse().get_matrix().transpose(-1, -2).cpu()
        image_size = (cameras.image_size[0][0].item(), cameras.image_size[0][1].item())
        return cls.create_cameras_o3d(intrinsics, extrinsics, image_size, is_c2w=True)

    @classmethod
    def create_cameras_p3d(cls, intrinsics: torch.Tensor, rotmats: torch.Tensor, tvecs: torch.Tensor, image_size: (int, int), is_c2w: bool = True):
        """
        Create PyTorch3D PerspectiveCameras from intrinsics and extrinsics.

        Parameters
        ----------
        intrinsics : torch.Tensor
            Shape (B, 3, 3), camera intrinsic matrices.
        rotmats : torch.Tensor
            Shape (B, 3, 3), camera-to-world rotations.
            PyTorch3D expects world-to-camera (R,T).
        tvecs : torch.Tensor
            Shape (B, 3), camera-to-world translations.
            PyTorch3D expects world-to-camera (R,T).
        image_size : (H, W)
            The height and width of the image.
        is_c2w : bool
            If True, the input extrinsics are camera-to-world (R,T).

        Returns
        -------
        cameras : PerspectiveCameras
        """
        if rotmats.shape[-1] != 3:
            assert rotmats.ndim == 2 and rotmats.shape[1] == 4
            rotmats = quaternion_to_rotation_matrix(rotmats)
        image_size_torch = torch.tensor(image_size, device=intrinsics.device).expand(intrinsics.shape[0], 2)

        # Normalize focal lengths and principal points to [-1,1] space used by PyTorch3D
        # PyTorch3D uses NDC convention:
        # focal_length is normalized by image size
        # principal_point is in normalized image coords
        # focal_length_x_ndc = fx * 2 / W
        # focal_length_y_ndc = fy * 2 / H
        focal_length = torch.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], dim=-1)  # (B,2)
        principal_point = torch.stack([intrinsics[:, 0, 2], intrinsics[:, 1, 2]], dim=-1)  # (B,2)

        # Convert extrinsics to world-to-camera if needed
        # If extrinsics are in camera-to-world format (R_c2w, t_c2w),:
        #   world_to_camera = extrinsics^-1
        if is_c2w:
            R_w2c = rotmats.transpose(-1, -2)  # R_w2c = R_c2w^T
            t_w2c = -torch.bmm(R_w2c, tvecs[..., None])[..., 0]  # t_w2c = -R_w2c * t_c2w
        else:
            R_w2c = rotmats.transpose(-1, -2)
            t_w2c = tvecs

        cameras = PerspectiveCameras(
            focal_length=focal_length.flip(dims=[-1]),
            principal_point=principal_point,
            R=R_w2c.transpose(-1, -2),
            T=t_w2c,
            in_ndc=False,
            device=intrinsics.device,
            image_size=image_size_torch,
        )
        return cameras

    @classmethod
    def torch_binom(cls, n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Computes the binomial coefficient C(n, k) using the log-gamma function for numerical stability.
        Similar to scipy.special.comb but for PyTorch.

        :param n: Tensor of non-negative integers (n >= k).
        :param k: Tensor of non-negative integers (k >= 0).
        :return: Tensor of binomial coefficients C(n, k).
        """
        mask = n.detach() >= k.detach()
        n = mask * n
        k = mask * k
        a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
        return torch.exp(a) * mask

    @classmethod
    def bernstein_basis(cls, degree: int, j: int, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the j-th Bernstein basis polynomial of a given degree evaluated at t.

        Args:
            degree (int): The degree of the polynomial (d).
            j (int): The index of the basis polynomial (0 <= j <= degree).
            t (torch.Tensor): Tensor of parameter values (usually in [0, 1]).

        Returns:
            torch.Tensor: The value of the Bernstein basis polynomial B_{j,d}(t).
                           Has the same shape as t.
        """
        if not (0 <= j <= degree):
            raise ValueError("j must be between 0 and degree, inclusive.")

        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)

        # Compute binomial coefficient C(degree, j)
        binom_coeff = cls.torch_binom(torch.tensor(degree, dtype=t.dtype, device=t.device),
                                      torch.tensor(j, dtype=t.dtype, device=t.device))

        t_clamped = torch.clamp(t, 0.0, 1.0)
        one_minus_t_clamped = 1.0 - t_clamped

        if j == 0:
            term1 = torch.ones_like(t_clamped)
        else:
            term1 = torch.pow(t_clamped, j)

        if j == degree:
            term2 = torch.ones_like(t_clamped)
        else:
            term2 = torch.pow(one_minus_t_clamped, degree - j)

        return binom_coeff * term1 * term2

    @classmethod
    def evaluate_bezier(cls, control_points: torch.Tensor, t_values: torch.Tensor) -> torch.Tensor:
        """
        Evaluates a Bézier curve at given parameter values t.

        Args:
            control_points (torch.Tensor): Tensor of control points, shape (d+1, 3),
                                            where d is the degree.
            t_values (torch.Tensor): Tensor of parameter values t to evaluate at,
                                     shape (S,).

        Returns:
            torch.Tensor: Evaluated points on the curve, shape (S, 3).
        """
        degree = control_points.shape[0] - 1
        n_samples = t_values.shape[0]
        n_dims = control_points.shape[1]  # Should be 3

        evaluated_points = torch.zeros((n_samples, n_dims),
                                       dtype=control_points.dtype,
                                       device=control_points.device)

        for j in range(degree + 1):
            basis_vals = cls.bernstein_basis(degree, j, t_values)  # Shape (S,)
            evaluated_points += basis_vals.unsqueeze(1) * control_points[j]

        return evaluated_points

    @classmethod
    def fit_polynomial_bezier(cls, points: torch.Tensor, degree: int, num_samples: int) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """
        Fits a 3D polynomial curve of a given degree to N points using a Bézier curve formulation.

        The curve interpolates the first and last points and minimizes least-squares error
        to the intermediate points. It also calculates the assignment of each sampled curve
        point to the closest original input point.

        Args:
            points (torch.Tensor): Input ground truth (GT) points, shape (N, 3). N must be >= 2.
            degree (int): Desired degree of the polynomial (d). Must be >= 1.
                          If d >= N-1, the curve will interpolate all points.
            num_samples (int): Number of points (S) to sample along the fitted curve.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
                - fitted_curve_points: Sampled points (S, 3).
                - assignments_closest_gt: Assignment based on closest GT point (S,).
                - snapped_assignments: Assignment with transitions snapped to anchors (S,).
                  Returns None for any if fitting fails.
        """
        n_points = points.shape[0]
        n_dims = points.shape[1]

        # --- Input Validation ---
        assert n_dims == 3, "Input points tensor must have shape (N, 3)."
        assert n_points >= 2, "Need at least 2 points to define a curve."
        assert n_points > degree >= 1, "Degree must be at least 1 and less than N-1."

        # --- Handle Simple Cases (N=2 or degree=1 -> Straight line) ---
        if n_points == 2 or degree == 1:
            t_sample = torch.linspace(0, 1, num_samples, dtype=points.dtype, device=points.device)
            fitted_curve_points = points[0].unsqueeze(0) * (1 - t_sample.unsqueeze(1)) + \
                                  points[1].unsqueeze(0) * t_sample.unsqueeze(1)
            # Assignments for a line segment
            assignments_closest_gt = torch.round(t_sample * (n_points - 1)).long()
            assignments_closest_gt = torch.clamp(assignments_closest_gt, 0, n_points - 1)
            snapped_assignments = assignments_closest_gt.clone()
            # # Camera transition: segment 0...s1 gets color 1. Here s1 is the end.
            # camera_transition_assignments = torch.ones_like(assignments_closest_gt)  # Color of point 1
            # Maybe assign first point to first camera? Let's make it simple: all points get color 1.
            # A better definition might be needed for the N=2 case specific to user's intent.
            # Assigning based on t > 0.5 might be closer?
            camera_transition_assignments = (t_sample > 0.0).long()  # Assign 1 if t > 0, else 0? Let's stick to index 1.
            camera_transition_assignments[:] = 1  # Assign color of the second camera (index 1)

            return fitted_curve_points, assignments_closest_gt, snapped_assignments

        # --- Parameterization (Chord Length) ---
        diffs = torch.diff(points, dim=0)
        distances = torch.linalg.norm(diffs, dim=1)
        t_values_unnormalized = torch.zeros(n_points, dtype=points.dtype, device=points.device)
        t_values_unnormalized[1:] = torch.cumsum(distances, dim=0)
        total_length = t_values_unnormalized[-1]
        if total_length < 1e-9:
            log("Total chord length is near zero. Using uniform parameterization.", 'warning')
            t_values = torch.linspace(0, 1, n_points, dtype=points.dtype, device=points.device)
        else:
            t_values = t_values_unnormalized / total_length

        # --- Set up Least Squares for Internal Control Points ---
        p_start = points[0]
        p_end = points[-1]
        intermediate_points = points[1:-1]
        intermediate_t = t_values[1:-1]

        num_intermediate = n_points - 2
        num_unknown_cps = degree - 1

        if num_unknown_cps <= 0:  # Should be caught by degree=1 check, but safety first
            log("No internal control points (degree=1 case should be handled).", 'error')
            return None, None, None  # Should not happen

        M = torch.zeros((num_intermediate, num_unknown_cps), dtype=points.dtype, device=points.device)
        for i in range(num_intermediate):
            for k in range(num_unknown_cps):
                basis_idx = k + 1
                M[i, k] = cls.bernstein_basis(degree, basis_idx, intermediate_t[i])

        basis_0_vals = cls.bernstein_basis(degree, 0, intermediate_t)
        basis_d_vals = cls.bernstein_basis(degree, degree, intermediate_t)
        R = intermediate_points - basis_0_vals.unsqueeze(1) * p_start - basis_d_vals.unsqueeze(1) * p_end

        # --- Solve the System ---
        try:
            lstsq_result = torch.linalg.lstsq(M, R)
            internal_control_points = lstsq_result.solution
        except Exception as e:
            log(f"\tError during torch.linalg.lstsq: {e}", 'error')
            return None, None, None

        # --- Assemble Full Control Points ---
        all_control_points = torch.zeros((degree + 1, 3), dtype=points.dtype, device=points.device)
        all_control_points[0] = p_start
        all_control_points[1:degree] = internal_control_points
        all_control_points[degree] = p_end

        # --- Evaluate Curve ---
        t_sample = torch.linspace(0, 1, num_samples, dtype=points.dtype, device=points.device)
        fitted_curve_points = cls.evaluate_bezier(all_control_points, t_sample)

        # --- Calculate Assignment 1: Closest GT for each Curve Point ---
        distances_curve_to_gt = torch.cdist(fitted_curve_points, points)
        assignments_closest_gt = torch.argmin(distances_curve_to_gt, dim=1)

        # --- Calculate Assignment 2: Snapped Assignment ---
        distances_gt_to_curve = torch.cdist(points, fitted_curve_points)
        gt_anchor_indices = torch.argmin(distances_gt_to_curve, dim=1)  # Shape (N,)

        # Sort unique anchors by curve index: list of (anchor_curve_idx, original_gt_idx)
        # noinspection PyUnresolvedReferences
        anchor_pairs = sorted([(idx.item(), gt_idx) for gt_idx, idx in enumerate(gt_anchor_indices)])
        unique_anchor_map = {}
        for anchor_idx, gt_idx in anchor_pairs:
            if anchor_idx not in unique_anchor_map: unique_anchor_map[anchor_idx] = gt_idx
        sorted_unique_anchors = sorted(unique_anchor_map.items())
        num_unique_anchors = len(sorted_unique_anchors)

        # --- Calculate Assignment 2: Snapped Assignment (Midpoint) ---
        snapped_assignments = torch.zeros_like(assignments_closest_gt)
        if num_unique_anchors == 0:
            snapped_assignments = assignments_closest_gt.clone()  # Fallback
        elif num_unique_anchors == 1:
            snapped_assignments[:] = sorted_unique_anchors[0][1]
        else:
            midpoints = [(sorted_unique_anchors[k][0] + sorted_unique_anchors[k + 1][0]) // 2 for k in range(num_unique_anchors - 1)]
            start_interval = 0
            current_gt_idx = sorted_unique_anchors[0][1]
            for k in range(num_unique_anchors - 1):
                end_interval = midpoints[k]
                snapped_assignments[start_interval: end_interval + 1] = current_gt_idx
                start_interval = end_interval + 1
                current_gt_idx = sorted_unique_anchors[k + 1][1]
            snapped_assignments[start_interval:] = sorted_unique_anchors[-1][1]

        # --- Calculate Assignment 3: Camera Transition Assignment ---
        camera_transition_assignments = torch.zeros_like(assignments_closest_gt)
        if num_unique_anchors == 0:
            camera_transition_assignments = assignments_closest_gt.clone()  # Fallback
        elif num_unique_anchors == 1:
            camera_transition_assignments[:] = sorted_unique_anchors[0][1]
        else:
            start_curve_idx = 0
            for k in range(num_unique_anchors - 1):
                # Segment up to next_anchor_idx gets color of next_anchor's original GT index
                next_anchor_idx, next_original_gt_idx = sorted_unique_anchors[k + 1]

                # Assign color of the *next* camera to the segment starting from the current anchor
                color_index = next_original_gt_idx
                # The segment runs from start_curve_idx up to (but not including) next_anchor_idx
                # Handle case where start_curve_idx might >= next_anchor_idx if anchors are identical after unique filter
                if next_anchor_idx > start_curve_idx:
                    camera_transition_assignments[start_curve_idx: next_anchor_idx] = color_index

                # Update start index for the next segment
                start_curve_idx = next_anchor_idx

            # Assign the last segment (from the last anchor index onwards)
            # Color should be that of the last camera/anchor
            last_anchor_idx, last_original_gt_idx = sorted_unique_anchors[-1]
            if start_curve_idx < num_samples:  # Ensure there are points left to assign
                camera_transition_assignments[start_curve_idx:] = last_original_gt_idx

        return fitted_curve_points, assignments_closest_gt, camera_transition_assignments

    @classmethod
    def export_trajectory_ply(cls, filename: Union[str, Path], gt_points: torch.Tensor, curve_points: torch.Tensor, assignments: torch.Tensor, gt_colors: np.ndarray):
        """
        Exports GT points and curve points to a colored PLY file.

        Args:
            filename (str): The output PLY filename.
            gt_points (torch.Tensor): Ground truth points (N, 3).
            curve_points (torch.Tensor): Sampled curve points (S, 3).
            assignments (torch.Tensor): Assignment index (0 to N-1) for each curve point (S,).
            gt_colors (np.ndarray): Colors (0-255 uint8) for GT points (N, 3).
        """
        n_gt = gt_points.shape[0]
        n_curve = curve_points.shape[0]

        # Ensure data is on CPU and NumPy for PLY export
        gt_points_np = gt_points.cpu().numpy().astype('f4')
        curve_points_np = curve_points.cpu().numpy().astype('f4')
        assignments_np = assignments.cpu().numpy()
        gt_colors_uint8 = gt_colors.astype('u1')

        # Assign colors to curve points based on assignments
        curve_colors_uint8 = gt_colors_uint8[assignments_np]

        # Create vertex data structured array
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        all_verts = np.zeros(n_gt + n_curve, dtype=vertex_dtype)

        # Fill GT points
        all_verts[:n_gt]['x'] = gt_points_np[:, 0]
        all_verts[:n_gt]['y'] = gt_points_np[:, 1]
        all_verts[:n_gt]['z'] = gt_points_np[:, 2]
        all_verts[:n_gt]['red'] = gt_colors_uint8[:, 0]
        all_verts[:n_gt]['green'] = gt_colors_uint8[:, 1]
        all_verts[:n_gt]['blue'] = gt_colors_uint8[:, 2]

        # Fill Curve points
        all_verts[n_gt:]['x'] = curve_points_np[:, 0]
        all_verts[n_gt:]['y'] = curve_points_np[:, 1]
        all_verts[n_gt:]['z'] = curve_points_np[:, 2]
        all_verts[n_gt:]['red'] = curve_colors_uint8[:, 0]
        all_verts[n_gt:]['green'] = curve_colors_uint8[:, 1]
        all_verts[n_gt:]['blue'] = curve_colors_uint8[:, 2]

        # Create PlyElement and PlyData
        from plyfile import PlyElement, PlyData
        # noinspection PyTypeChecker
        el = PlyElement.describe(all_verts, 'vertex')
        try:
            PlyData([el]).write(str(filename))
            log(f"\tSuccessfully exported 3D trajectory to {filename}")
        except Exception as e:
            log(f"\tError exporting PLY file: {e}", 'critical', exc_info=e)

    @classmethod
    def create_cameras(
            cls,
            intrinsics_selected: torch.Tensor,
            tvecs_selected: torch.Tensor,
            rotmats_selected: torch.Tensor,
            tvecs_all: torch.Tensor,
            rotmats_all: torch.Tensor,
            H: int,
            W: int,
            num_virtual_cameras: int = 360,
            vis_only_idx: Optional[List[int]] = None,
            assignment_only_idx: Optional[List[int]] = None,
            out_path: Optional[Path] = None,
            use_bezier: bool = False,
            lib: Union[Literal['kaolin'] | Literal['pytorch3d'] | Literal['open3d']] = 'pytorch3d',
    ) -> Tuple[PerspectiveCameras, PerspectiveCameras, Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[int], List[int]]:
        cameras = cls.create_cameras_p3d(
            intrinsics=intrinsics_selected,
            rotmats=rotmats_selected,
            tvecs=tvecs_selected,
            image_size=(H, W),
            is_c2w=True,
        )
        virtual_cameras, assignment_closest, assignment_stereo = cls.create_virtual_cameras(
            intrinsics=intrinsics_selected,
            tvecs=tvecs_all,
            rotmats=rotmats_all,
            vis_only_idx=vis_only_idx,
            assignment_only_idx=assignment_only_idx,
            H=H,
            W=W,
            num_cameras=num_virtual_cameras,
            out_path=out_path,
            use_bezier=use_bezier,
        )
        if use_bezier and vis_only_idx is not None and assignment_only_idx is not None:
            cameras = cameras[[vis_only_idx.index(_) for _ in assignment_only_idx]]
        inverse_assignment_closest = []
        for cam_idx, virtual_cam_idx in assignment_closest.items():
            inverse_assignment_closest.extend([cam_idx] * len(virtual_cam_idx))
        inverse_assignment_stereo = []
        for cam_idx, virtual_cam_idx in assignment_stereo.items():
            inverse_assignment_stereo.extend([cam_idx] * len(virtual_cam_idx))
        if lib == 'open3d':
            cameras = cls.create_cameras_o3d_from_p3d(cameras)
            virtual_cameras = cls.create_cameras_o3d_from_p3d(virtual_cameras)
        return cameras, virtual_cameras, assignment_closest, assignment_stereo, inverse_assignment_closest, inverse_assignment_stereo

    @classmethod
    def create_trajectory_swaying(cls, all_cams: List[int], all_ts: List[int], t_start_idx: int = 0, t_stop_idx: int = -1, cam_start_idx: int = 0, fixed_ts: Optional[List[int]] = None) -> Dict[int, List[int]]:
        # get all[1 times
        t_start_idx_s0 = all_ts.index(t_start_idx)
        if t_stop_idx < 0:
            t_stop_idx_s0 = len(all_ts) + t_stop_idx + 1
        else:
            t_stop_idx_s0 = min(len(all_ts), all_ts.index(t_stop_idx) + 1)
        all_ts = all_ts[t_start_idx_s0:t_stop_idx_s0]
        ts = copy.deepcopy(all_ts)
        # get all cameras
        cam_start_idx_s0 = all_cams.index(cam_start_idx)
        cams = all_cams[cam_start_idx_s0:]
        if cam_start_idx_s0 != 0:
            cams += all_cams[::-1]
        # make cams at least as big as times via cloning
        sort_dir = -1
        while len(cams) < len(ts):
            cams += all_cams[::sort_dir]
            sort_dir *= -1
        cams = cams[:len(ts)]
        # trajectory = {ti: [cami] for ti, cami in zip(ts, cams)}
        if fixed_ts is not None:
            fixed_ts = [(len(all_ts) + _) if _ < 0 else _ for _ in fixed_ts]
            fixed_ts_idx_s0 = [(all_ts.index(fixed_ti) - t_start_idx_s0) for fixed_ti in fixed_ts]
            idx_s0_offset = 0
            for fixed_ti, fixed_ti_idx_s0 in zip(fixed_ts, fixed_ts_idx_s0):
                if (fixed_ti_idx_s0 + idx_s0_offset) == 0 or cams[fixed_ti_idx_s0 + idx_s0_offset] > cams[fixed_ti_idx_s0 + idx_s0_offset - 1]:
                    cams[fixed_ti_idx_s0 + idx_s0_offset:fixed_ti_idx_s0 + idx_s0_offset] = all_cams[all_cams.index(cams[fixed_ti_idx_s0 + idx_s0_offset]):] + all_cams[::-1] + all_cams[:all_cams.index(cams[fixed_ti_idx_s0 + idx_s0_offset])]
                else:
                    cams[fixed_ti_idx_s0 + idx_s0_offset:fixed_ti_idx_s0 + idx_s0_offset] = all_cams[:all_cams.index(cams[fixed_ti_idx_s0 + idx_s0_offset]) + 1][::-1] + all_cams + all_cams[all_cams.index(cams[fixed_ti_idx_s0 + idx_s0_offset]):][::-1]
                ts[fixed_ti_idx_s0 + idx_s0_offset:fixed_ti_idx_s0 + idx_s0_offset] = [fixed_ti_idx_s0] * len(all_cams) * 2
                idx_s0_offset += len(all_cams) * 2
        ts_cam = list(zip(ts, cams))
        # remove duplicates
        ts_cam_dedup = []
        for idx, pair in enumerate(ts_cam):
            # If this is the first element, or it differs from the previous one, keep it
            if idx == 0 or pair != ts_cam[idx - 1]:
                ts_cam_dedup.append(pair)
        # group by time
        grouped = defaultdict(list)
        for t_idx, cam_idx in ts_cam_dedup:
            grouped[t_idx].append(cam_idx)
        return dict(grouped)

    @classmethod
    def create_virtual_cameras(cls, intrinsics: torch.Tensor, tvecs: torch.Tensor, rotmats: torch.Tensor, H: int, W: int, out_path: Optional[Path] = None, vis_only_idx: Optional[List[int]] = None, assignment_only_idx: Optional[List[int]] = None, use_bezier: bool = False, num_cameras: int = 360) -> Tuple[PerspectiveCameras, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        # create a virtual trajectory around the pointcloud
        #   - get current look ats from current cameras
        #   - find the common lookat point / new lookat point
        #   - create an ellipsoidal curve that best fits the cameras
        #   - create a video, with how the gaussian PCD is rasterized at every virtual camera position (for each virtual camera there is a useful function to get the projected/rasterized PCD:
        #       projected_images, projected_depths, valid_pixels = gs_pcds.project(
        #         cameras=cameras,
        #       )
        #     which return a batch of projected images, depths and valid pixels, one for each camera
        # Find the common look-at point by averaging the current look-at points
        mean_look_at_point, mean_up_vector = cls.compute_mean_look_at_and_up(
            tvecs=tvecs,
            rotmats=rotmats
        )
        # Define a circle of virtual camera positions around the point cloud (best fitting circle on camera positions, plus a radius offset)
        virtual_cam_T, assignment_middle, assignment_camtrans = cls.virtual_tvecs(
            tvecs=tvecs,
            vis_only_idx=vis_only_idx,
            num_points=num_cameras,
            assignment_only_idx=assignment_only_idx,
            radius_offset=0.5,
            use_bezier=use_bezier
        )
        # Create virtual cameras
        R, T = look_at_view_transform(
            eye=virtual_cam_T.to(dtype=torch.float32),
            at=mean_look_at_point.unsqueeze(0).expand(virtual_cam_T.shape[0], -1),
            up=mean_up_vector.unsqueeze(0).expand(virtual_cam_T.shape[0], -1),
            device=tvecs.device
        )
        virtual_intrinsics = intrinsics.mean(0).expand(virtual_cam_T.shape[0], 3, 3)
        # virtual_intrinsics = intrinsics[list(assignment_middle.keys())].repeat_interleave(torch.tensor(list(map(lambda x: len(x), assignment_middle.values())), dtype=torch.long, device=intrinsics.device), dim=0)
        virtual_cameras = cls.create_cameras_p3d(
            intrinsics=virtual_intrinsics,
            rotmats=R,
            tvecs=T,
            image_size=(H, W),
            is_c2w=False,
        )
        # Visualize the virtual cameras
        if out_path is not None:
            cams_w2c = cls.create_cameras_p3d(
                intrinsics=intrinsics[vis_only_idx] if vis_only_idx is not None else intrinsics,
                rotmats=rotmats[vis_only_idx] if vis_only_idx is not None else rotmats,
                tvecs=tvecs[vis_only_idx] if vis_only_idx is not None else tvecs,
                image_size=(H, W),
                is_c2w=True,
            ).get_world_to_view_transform().inverse().get_matrix().transpose(-1, -2)
            virtual_w2c = virtual_cameras.get_world_to_view_transform().inverse().get_matrix().transpose(-1, -2)
            out_path = (Path(out_path) / f'cams_vis.pdf') if out_path is not None else None
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    num_existing_files = len(list(out_path.parent.glob(f"{out_path.stem}*{out_path.suffix}")))
                    out_path = out_path.with_name(out_path.stem + f'_{num_existing_files + 1}' + out_path.suffix)
            cls.visualize_virtual_cameras(
                cam_T=cams_w2c[..., :3, 3],
                cam_R=cams_w2c[..., :3, :3],
                virtual_cam_T=virtual_w2c[..., :3, 3],
                virtual_cam_R=virtual_w2c[..., :3, :3],  # bug fix
                mean_look_at_point=mean_look_at_point,
                assignment_closest=assignment_middle,
                assignment_camtrans=assignment_camtrans,
                save_path=out_path
            )
        return virtual_cameras, assignment_middle, assignment_camtrans

    @classmethod
    def extract_extrinsics_from_p3d(cls, cam: PerspectiveCameras) -> torch.Tensor:
        out = cam.get_world_to_view_transform().get_matrix().inverse().transpose(-2, -1)
        return out.contiguous()

    @classmethod
    def extract_from_p3d(cls, cameras: PerspectiveCameras) -> List[Dict[str, torch.Tensor]]:
        return [
            dict(
                extrinsics=cls.extract_extrinsics_from_p3d(cam_),
                intrinsics=cls.extract_intrinsics_from_p3d(cam_),
                height=cam_.image_size[0, 0].item(),
                width=cam_.image_size[0, 1].item(),
            )
            for cam_ in cameras
        ]

    @classmethod
    def extract_intrinsics_from_p3d(cls, cam: PerspectiveCameras) -> torch.Tensor:
        out = cam.get_projection_transform().get_matrix().transpose(-2, -1)[..., [0, 1, 3], :3]
        out[..., :2, :2] = out[..., [1, 0], :2][..., :2, [1, 0]]
        return out.contiguous()

    @classmethod
    def vis_disparity(cls, disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output=None):
        """
        from FoundationStereo repo
        @disp: np array (H,W)
        @invalid_thres: > thres is invalid
        """
        disp = disp.copy()
        H, W = disp.shape[:2]
        invalid_mask = disp >= invalid_thres
        if (invalid_mask == 0).sum() == 0:
            if other_output is not None:
                other_output['min_val'] = None
                other_output['max_val'] = None
            return np.zeros((H, W, 3))
        if min_val is None:
            min_val = disp[invalid_mask == 0].min()
        if max_val is None:
            max_val = disp[invalid_mask == 0].max()
        if other_output is not None:
            other_output['min_val'] = min_val
            other_output['max_val'] = max_val
        vis = ((disp - min_val) / (max_val - min_val)).clip(0, 1) * 255
        if cmap is None:
            vis_image_bgr = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)
        else:
            vis_image_bgr = cmap(vis.astype(np.uint8))[..., :3] * 255
        if invalid_mask.any():
            vis_image_bgr[invalid_mask] = 0
        return vis_image_bgr.astype(np.uint8)

    @classmethod
    def vis_depth(cls, depth, min_val=1.0, max_val=3.5, invalid_thres=np.inf, color_map=cv2.COLORMAP_INFERNO):
        depth_vis = ((depth.clip(min_val, max_val) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        valid_mask = (depth_vis > 0.0) & (depth_vis < invalid_thres)
        depth_in_mask = depth_vis[valid_mask]
        depth_in_mask = 255 - depth_in_mask
        depth_vis[valid_mask] = depth_in_mask
        depth_image_bgr = cv2.applyColorMap(depth_vis, color_map)
        return depth_image_bgr.astype(np.uint8)

    # noinspection PyTypeChecker
    @classmethod
    def visualize_virtual_cameras(cls,
                                  cam_T: torch.Tensor,
                                  cam_R: torch.Tensor,
                                  virtual_cam_T: torch.Tensor,
                                  virtual_cam_R: torch.Tensor,
                                  mean_look_at_point: torch.Tensor,
                                  assignment_closest: torch.Tensor | Dict[int, torch.Tensor],
                                  assignment_camtrans: torch.Tensor | Dict[int, torch.Tensor],
                                  offset_factor: float = 1.0,
                                  save_path: str | None = None):
        """
        Creates a 2D visualization with the two different assignments.
        - Projects points onto the best-fit plane (PCA on GT points).
        - Plots GT and virtual camera locations and optical axis.
        - Plots curve points colored by 'closest GT' assignment (circles).
        - Plots curve points colored by 'camera transition' assignment, offset slightly along the local curve normal (crosses).

        Args:
            cam_T (torch.Tensor): Original ground truth points (N, 3).
            cam_R (torch.Tensor): Original ground truth rotations (N, 3, 3).
            virtual_cam_T (torch.Tensor): Fitted curve points (S, 3).
            virtual_cam_R (torch.Tensor): Fitted curve rotations (S, 3, 3).
            mean_look_at_point (torch.Tensor): The common target point (3,).
            assignment_closest (torch.Tensor): Assignment indices (0...N-1) for curve points
                                               based on the closest GT point (S,).
            assignment_camtrans (torch.Tensor): Assignment indices (0...N-1) for curve points
                                                based on the camera transition logic (S,).
            offset_factor (float): Multiplier for the average point spacing to determine
                                   the offset distance for the 'x' markers. Adjust for visual preference.
            save_path (str | None): If provided, saves the plot to this file path.
        """
        # 1. Ensure data is on CPU and NumPy
        cam_T_np = cam_T.detach().cpu().numpy().astype(np.float64)
        virtual_cam_T_np = virtual_cam_T.detach().cpu().numpy().astype(np.float64)
        look_vectors_np = cam_R.detach().cpu()[:, :, 2].numpy().astype(np.float64)  # old (view) wrt new (world)
        virtual_look_vectors_np = virtual_cam_R.detach().cpu()[:, :, 2].numpy().astype(np.float64)
        if isinstance(assignment_closest, dict):
            # dict contains GT cam index as key and list of virtual cam indices as value
            # need to transform it to a 1D tensor, where in each element is the GT cam index for that virtual camera index
            assignment_closest = torch.cat([torch.full((len(assignment_closest[cam_idx]),), cam_idx) for cam_idx in assignment_closest.keys()])
        if isinstance(assignment_camtrans, dict):
            assignment_camtrans = torch.cat([torch.full((len(assignment_camtrans[cam_idx]),), cam_idx) for cam_idx in assignment_camtrans.keys()])
        assignment_closest_np = assignment_closest.detach().cpu().numpy()
        assignment_camtrans_np = assignment_camtrans.detach().cpu().numpy()

        n_gt = cam_T_np.shape[0]
        n_curve = virtual_cam_T_np.shape[0]

        # 2. PCA Projection
        assert n_curve >= 2 and n_gt >= 3, "PCA requires at least 3 points to define a plane."
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, svd_solver='full')
        # Fit PCA *only* on the ground truth points to define the plane
        pca.fit(cam_T_np)
        # Transform both sets of points into the 2D plane
        cam_T_2d = pca.transform(cam_T_np)
        # mean_look_at_2d = pca.transform(mean_look_at_point.reshape(1, -1).numpy())[0]
        virtual_cam_T_2d = pca.transform(virtual_cam_T_np)
        # Project look vectors by projecting their endpoints
        vector_endpoints_3d = cam_T_np + look_vectors_np
        vector_endpoints_2d = pca.transform(vector_endpoints_3d)
        look_vectors_2d = vector_endpoints_2d - cam_T_2d
        virtual_vector_endpoints_3d = virtual_cam_T_np + virtual_look_vectors_np
        virtual_vector_endpoints_2d = pca.transform(virtual_vector_endpoints_3d)
        virtual_look_vectors_2d = virtual_vector_endpoints_2d - virtual_cam_T_2d

        # 3. Calculate Approximate Normals and Offset Points
        # Estimate tangent components (dx, dy) using gradient
        dx = np.gradient(virtual_cam_T_2d[:, 0])
        dy = np.gradient(virtual_cam_T_2d[:, 1])
        # Normal vector is perpendicular to tangent: (-dy, dx)
        normals_approx = np.stack([-dy, dx], axis=-1)  # Shape (S, 2)
        # Normalize the normal vectors
        magnitudes = np.linalg.norm(normals_approx, axis=1, keepdims=True)
        epsilon = 1e-9
        normals_normalized = normals_approx / (magnitudes + epsilon)
        # Calculate dynamic offset distance
        avg_spacing = np.mean(np.linalg.norm(np.diff(virtual_cam_T_2d, axis=0), axis=1))
        offset_distance = avg_spacing * offset_factor
        # Calculate offset points
        offset_curve_points_2d = virtual_cam_T_2d + normals_normalized * offset_distance

        # 4. Prepare Colors
        cmap = plt.cm.get_cmap('tab20')
        color_indices = np.arange(n_gt) % 20
        gt_colors = cmap(color_indices / 19.0)
        if n_gt > 20:
            log(f"\tN GT points > 20 ({n_gt}). 'tab20' colors will repeat.", 'warning')

        # 5. Clip assignments (safety)
        assignment_closest_np = np.clip(assignment_closest_np, 0, n_gt - 1)
        assignment_camtrans_np = np.clip(assignment_camtrans_np, 0, n_gt - 1)

        # 6. Plotting
        fig, ax = plt.subplots(figsize=(10, 10))

        # --- Plot GT points ---
        cam_T_2d_active = np.copy(cam_T_2d)
        gt_colors_active = np.copy(gt_colors)
        look_vectors_2d_active = np.copy(look_vectors_2d)
        cam_T_2d_inactive = []
        look_vectors_2d_inactive = []
        for i, gt_point in enumerate(cam_T_2d):
            if np.sum(assignment_closest_np == i) == 0:
                cam_T_2d_inactive.append(gt_point)
                np.delete(cam_T_2d_active, i, axis=0)
                np.delete(gt_colors_active, i, axis=0)
                look_vectors_2d_inactive.append(look_vectors_2d[i])
                np.delete(look_vectors_2d_active, i, axis=0)
        ax.scatter(cam_T_2d_active[:, 0], cam_T_2d_active[:, 1], c=gt_colors_active, marker='^', s=180, edgecolors='k', label='GT Cameras (recon)', zorder=3)
        look_vectors_2d_active = look_vectors_2d_active / (np.linalg.norm(look_vectors_2d_active, axis=1, keepdims=True) + 1e-9) * 0.1
        ax.quiver(cam_T_2d_active[:, 0], cam_T_2d_active[:, 1],
                  look_vectors_2d_active[:, 0], look_vectors_2d_active[:, 1],
                  color='black', angles='xy', scale_units='xy', width=0.002, headwidth=1, headlength=0)
        # Plot inactive GT points with a grayed-out color
        if cam_T_2d_inactive:
            cam_T_2d_inactive = np.stack(cam_T_2d_inactive, axis=0)
            look_vectors_2d_inactive = np.stack(look_vectors_2d_inactive, axis=0)
            ax.scatter(cam_T_2d_inactive[:, 0], cam_T_2d_inactive[:, 1], c='gray', marker='^', s=180, edgecolors='darkgray', label='GT Cameras (vis)', zorder=3)
            look_vectors_2d_inactive = look_vectors_2d_inactive / (np.linalg.norm(look_vectors_2d_inactive, axis=1, keepdims=True) + 1e-9) * 0.1
            ax.quiver(cam_T_2d_inactive[:, 0], cam_T_2d_inactive[:, 1],
                      look_vectors_2d_inactive[:, 0], look_vectors_2d_inactive[:, 1],
                      color='gray', angles='xy', scale_units='xy', width=0.003, headwidth=1, headlength=0)

        # --- Plot Curve (Closest Assignment) - Circles at original position ---
        colors_closest = gt_colors[assignment_closest_np]
        ax.scatter(virtual_cam_T_2d[:, 0], virtual_cam_T_2d[:, 1], c=colors_closest, marker='o', s=35, label='Virtual (Closest Assign.)', alpha=0.8, zorder=1)
        virtual_look_vectors_2d = virtual_look_vectors_2d / (np.linalg.norm(virtual_look_vectors_2d, axis=1, keepdims=True) + 1e-9) * 0.1
        ax.quiver(virtual_cam_T_2d[:, 0], virtual_cam_T_2d[:, 1],
                  virtual_look_vectors_2d[:, 0], virtual_look_vectors_2d[:, 1],
                  color=colors_closest, angles='xy', scale_units='xy', width=0.003, headwidth=1, headlength=0)

        # Look at point
        if mean_look_at_point is not None:
            # project mean look-at point to 2D
            mean_look_at_point_2d = pca.transform(mean_look_at_point.cpu().numpy().reshape(1, -1))
            ax.scatter(mean_look_at_point_2d[:, 0], mean_look_at_point_2d[:, 1], c='black', marker='o', s=100, label='Look At (mean)', alpha=0.8, zorder=4)

        # --- Plot Curve (Camera Transition Assignment) - Crosses at OFFSET position ---
        colors_camtrans = gt_colors[assignment_camtrans_np]
        ax.scatter(offset_curve_points_2d[:, 0], offset_curve_points_2d[:, 1], c=colors_camtrans, marker='x', s=70, label='Virtual (CamTrans Assign.)', alpha=0.8, zorder=2)
        ax.set_title("Virtual Camera Trajectory with Assignments")
        ax.set_xlabel(f"PC 1 (Explains {pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC 2 (Explains {pca.explained_variance_ratio_[1] * 100:.1f}%)")
        # make axis equal, i.e. xlim == ylim
        ax.set_aspect('equal', adjustable='box')
        # Set axis limits based on GT points
        xlim = np.array([np.min(cam_T_2d[:, 0]), np.max(cam_T_2d[:, 0])])
        ylim = np.array([np.min(cam_T_2d[:, 1]), np.max(cam_T_2d[:, 1])])
        aggr_lim = np.minimum(xlim[0], ylim[0]), np.maximum(xlim[1], ylim[1])
        ax.set_xlim([aggr_lim[0] - 0.2, aggr_lim[1] + 0.2])
        ax.set_ylim([aggr_lim[0] - 0.2, aggr_lim[1] + 0.2])
        ax.legend()
        plt.tight_layout()

        # 8. Save or show
        if save_path:
            plt.savefig(save_path, dpi=150)
            log(f"\tSaved 2D Bezier assignment plot to {save_path}")
        else:
            plt.show()

    @classmethod
    def virtual_tvecs(cls, tvecs: torch.Tensor, num_points: int = 360, vis_only_idx: Optional[List[int]] = None, assignment_only_idx: Optional[List[int]] = None, radius_offset: float = 0.5, use_bezier: bool = False, bezier_degree: int = -1, vis: Optional[Path] = None) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        :param tvecs: torch.Tensor, camera positions in world coordinates. Tensor of shape (N,3).
        :param num_points: int, number of points to sample in the fitted circle.
        :param vis_only_idx: List[int], optional, after computing the circle center / radius use only those cameras to compute the closest idx
        :param assignment_only_idx: List[int], optional, after computing the closest idx wrt to virtual cameras, update the assignments to be wrt to assignment_only_idx
        :param radius_offset: float, an offset added to the computed radius.
        :param use_bezier: bool, if True, use Bézier curve fitting instead of circle fitting.
        :param bezier_degree: int, degree of the Bézier curve to fit. If -1, it will be set to min(3, len(vis_only_idx) - 1).
        :param vis: Path, optional, if set it will prodice a visualization of the 2D circle, projected cameras, and assignments.
        :return: Tuple[torch.Tensor, Dict[int, torch.Tensor]]
            - circle_points_3d: torch.Tensor, the circle points (i.e. the positions of the virtual cameras) in world coordinates.
            - closest_idx: Dict[int, torch.Tensor], assignment dictionary where for each ground-truth camera (key) an index tensor to the virtual cameras is given (value).
        """
        if vis_only_idx is None:
            vis_only_idx = list(range(len(tvecs)))
        if assignment_only_idx is None:
            assignment_only_idx = vis_only_idx

        if use_bezier:
            # Use Bézier curve fitting
            assert bezier_degree == -1 or bezier_degree >= 1, "bezier_degree must be -1 or a positive integer."
            virtual_tvecs_bezier, assignment_closest, assignment_camtrans = cls.fit_polynomial_bezier(
                points=tvecs,
                num_samples=num_points,
                degree=bezier_degree if bezier_degree > 0 else min(2, len(tvecs) - 1),
            )
            # Map each assignment index to the vis_only_idx whose GT tvec is closest to the corresponding virtual tvec
            device = assignment_closest.device
            vt = torch.as_tensor(virtual_tvecs_bezier, dtype=torch.float32, device=device)  # virtual tvecs
            gt = torch.as_tensor(tvecs, dtype=torch.float32, device=device)  # GT tvecs
            vis_only_idx_tensor = torch.as_tensor(vis_only_idx, dtype=torch.long, device=device)

            def map_by_tvec_distance(assign_idx: torch.Tensor) -> torch.Tensor:
                vpos = vt  # (N, 3) virtual positions for these assignments
                candidates = gt[vis_only_idx_tensor]  # (M, 3) GT positions to choose from
                d = torch.cdist(vpos, candidates)  # (N, M) Euclidean distances
                j = torch.argmin(d, dim=1)  # nearest candidate per assignment
                return vis_only_idx_tensor[j]  # map to GT indices

            assignment_closest = map_by_tvec_distance(assignment_closest).detach().cpu()
            assignment_camtrans = map_by_tvec_distance(assignment_camtrans).detach().cpu()
            assignment_closest_dict = defaultdict(list)
            assignment_camtrans_dict = defaultdict(list)
            for curve_idx in range(num_points):
                assignment_closest_dict[assignment_closest[curve_idx].item()].append(curve_idx)
                assignment_camtrans_dict[assignment_camtrans[curve_idx].item()].append(curve_idx)
            # Create a 2D visualization of the Bézier curve and assignments
            if set(assignment_only_idx) != set(vis_only_idx):
                # 1. Identify Kept/Removed Relative Indices
                #
                removed_relative_indices = [i for i in vis_only_idx if i not in set(assignment_only_idx)]
                if removed_relative_indices:
                    log(f"\tRemoved relative indices: {removed_relative_indices}")
                    kept_points_3d = tvecs[assignment_only_idx]
                    removed_points_3d = tvecs[removed_relative_indices]
                    # find Closest Kept point for Each Removed point
                    distances_removed_to_kept = torch.cdist(removed_points_3d, kept_points_3d)
                    closest_kept_subset_indices = [vis_only_idx.index(assignment_only_idx[_.item()]) for _ in torch.argmin(distances_removed_to_kept, dim=1)]  # Shape (num_removed,)
                    # merge dictionaries
                    for removed_idx, closest_kept_idx in zip(removed_relative_indices, closest_kept_subset_indices):
                        assignment_closest_dict[closest_kept_idx] += assignment_closest_dict[removed_idx]
                        assignment_closest_dict[closest_kept_idx] = sorted(assignment_closest_dict[closest_kept_idx])
                        del assignment_closest_dict[removed_idx]
                        assignment_camtrans_dict[closest_kept_idx] += assignment_camtrans_dict[removed_idx]
                        assignment_camtrans_dict[closest_kept_idx] = sorted(assignment_camtrans_dict[closest_kept_idx])
                        del assignment_camtrans_dict[removed_idx]
                        assignment_closest[assignment_closest == removed_idx] = closest_kept_idx
                        assignment_camtrans[assignment_camtrans == removed_idx] = closest_kept_idx
                        if closest_kept_idx - removed_idx == 1:
                            # fix camtrans assignment (as the first of the pair is removed): assign all to the closest of the closest
                            assignment_only_idx_except_closest = [_ for _ in assignment_only_idx if _ != closest_kept_idx]
                            distances_closest_kept_to_other_kept = torch.cdist(tvecs[[closest_kept_idx]], tvecs[assignment_only_idx_except_closest])
                            closest_of_closest_kept_idx = assignment_only_idx_except_closest[torch.argmin(distances_closest_kept_to_other_kept, dim=1).item()]
                            assignment_camtrans_dict[closest_of_closest_kept_idx] += assignment_camtrans_dict[closest_kept_idx]
                            assignment_camtrans_dict[closest_of_closest_kept_idx] = sorted(assignment_camtrans_dict[closest_of_closest_kept_idx])
                            del assignment_camtrans_dict[closest_kept_idx]
                            assignment_camtrans[assignment_camtrans == closest_kept_idx] = closest_of_closest_kept_idx
            # noinspection PyTypeChecker
            return virtual_tvecs_bezier, {k: v for k, v in assignment_closest_dict.items()}, {k: v for k, v in assignment_camtrans_dict.items()}

        # 1) Fit a plane to camera positions
        plane_origin = tvecs.mean(0)
        vht = torch.linalg.svd(tvecs - plane_origin)[2]
        plane_normal = vht[2, :]
        plane_normal /= plane_normal.norm()

        # 2) Project points onto the plane
        tvecs_centered = tvecs - plane_origin[None]
        tvec_dists = torch.sum(tvecs_centered @ plane_normal[:, None], dim=1)
        projected_points = tvecs - tvec_dists[:, None] @ plane_normal[None]  # (N,3)

        # 3) Transform points in the plane coordinate system (using the rows 0 and 1 of `vht` span the plane)
        u = vht[0, :]
        v = vht[1, :]
        projected_2d = torch.stack([
            torch.sum(projected_points * u[None], dim=1),
            torch.sum(projected_points * v[None], dim=1)
        ], dim=1)  # Shape: (N, 2)

        # 4) Fit plane in 2D (e.g. using torch's LBFGS implementation)
        mean_2d = projected_2d.mean(0)
        radius_init = torch.mean(torch.sqrt(torch.sum((projected_2d - mean_2d[None]) ** 2, dim=1)))
        params = torch.tensor([mean_2d[0], mean_2d[1], radius_init], device=tvecs.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([params], line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            projected_2d_dists = torch.sqrt((projected_2d[:, 0] - params[0]) ** 2 + (projected_2d[:, 1] - params[1]) ** 2)
            loss = torch.sum((projected_2d_dists - params[2]) ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)
        cx, cy, radius = params.detach()
        radius += radius_offset

        # 5) Translate circle center and radius back to 3D coordinates (taking into account that the camera positions had been centered)
        circle_center_2d = torch.tensor([cx, cy], device=tvecs.device, dtype=tvecs.dtype)
        # circle_center_3d = plane_origin + circle_center_2d[0] * u + circle_center_2d[1] * v

        # 6) Generate points on the circle (in world coordinates)
        theta = torch.linspace(0, 2 * torch.pi, num_points, device=tvecs.device)
        circle_2d_points = torch.stack([
            cx + torch.cos(theta) * radius,
            cy + torch.sin(theta) * radius
        ], dim=1)
        circle_points_3d = plane_origin + circle_2d_points[:, 0:1] * u + circle_2d_points[:, 1:2] * v

        # 7) Generate assignment dict
        def unblock_idx(idx):
            idx = torch.sort(idx).values
            diff = idx[1:] - idx[:-1]
            # noinspection PyUnresolvedReferences
            split_points = (diff != 1).nonzero(as_tuple=True)[0] + 1
            # tensor_split by indices, similar to np.split
            blocks = torch.tensor_split(idx, split_points.tolist())
            # Sort blocks by their first element in descending order
            blocks = sorted(blocks, key=lambda b: b[0].item(), reverse=True)
            return torch.cat(blocks)

        distances_2d = torch.linalg.norm(projected_2d[:, torch.newaxis] - circle_2d_points[None], dim=2)
        closest_idx_middle_values = torch.argmin(distances_2d, dim=0)  # Shape: (num_points,)
        closest_idx_middle = {int(k.item()): unblock_idx((closest_idx_middle_values == k).nonzero(as_tuple=True)[0])
                              for k in torch.unique(closest_idx_middle_values)}  # Dict[int, torch.LongTensor]
        closest_idx_middle = dict(sorted(closest_idx_middle.items(), key=lambda item: item[1].max().item()))

        # 8) Rearrange the circle indices into C-1 bins based on C-2 break points (for the stereo case)
        closest_idx_exact = {}
        internal_breaks = torch.stack([
            assigned_idx[torch.argmin(torch.linalg.norm(projected_2d[cam][None] - circle_2d_points[assigned_idx], dim=-1)).item()]
            for cam, assigned_idx in closest_idx_middle.items()
        ])
        all_idx = torch.cat(list(closest_idx_middle.values())).unique(sorted=True).to(dtype=torch.long, device=distances_2d.device)
        for i, (cam, assigned_idx) in enumerate(closest_idx_middle.items()):
            if i < len(internal_breaks) - 1:
                # noinspection PyTypeChecker
                closest_idx_exact[cam] = torch.arange(internal_breaks[i], internal_breaks[i + 1], device=internal_breaks.device)
            else:
                # noinspection PyTypeChecker
                closest_idx_exact[cam] = torch.cat([torch.arange(internal_breaks[-1], all_idx.max(), device=internal_breaks.device), torch.arange(0, internal_breaks[0], device=internal_breaks.device)])

        # 9) Select subset of cameras
        if vis_only_idx != list(range(len(tvecs))) or True:
            closest_idx_middle = {k: v for k, v in closest_idx_middle.items() if k in vis_only_idx}
            closest_idx_exact = {k: v for k, v in closest_idx_exact.items() if k in vis_only_idx}
            # fix edges
            closest_idx_exact_keys = list(closest_idx_exact.keys())
            closest_idx_exact[closest_idx_exact_keys[0]] = torch.cat((closest_idx_exact[closest_idx_exact_keys[0]], closest_idx_middle[closest_idx_exact_keys[0]])).unique()
            if len(closest_idx_exact_keys) > 1:
                closest_idx_exact[closest_idx_exact_keys[-1]] = closest_idx_exact[closest_idx_exact_keys[-1]][(closest_idx_exact[closest_idx_exact_keys[-1]].view(1, -1) == closest_idx_middle[closest_idx_exact_keys[-1]].view(-1, 1)).any(dim=0)]
                if len(closest_idx_exact_keys) >= 2:
                    closest_idx_exact[closest_idx_exact_keys[-2]] = torch.cat([closest_idx_exact[closest_idx_exact_keys[-1]], closest_idx_exact[closest_idx_exact_keys[-2]]]).unique()
                del closest_idx_exact[closest_idx_exact_keys[-1]]
            else:
                closest_idx_exact[closest_idx_exact_keys[0]] = closest_idx_exact[closest_idx_exact_keys[0]][(closest_idx_exact[closest_idx_exact_keys[0]].view(1, -1) == closest_idx_middle[closest_idx_exact_keys[0]].view(-1, 1)).any(dim=0)]
            all_idx = torch.cat([v for k, v in closest_idx_middle.items()]).unique(sorted=True).to(dtype=torch.long, device=distances_2d.device)
            assert torch.equal(all_idx, torch.cat([v for k, v in closest_idx_exact.items()]).unique(sorted=True).to(dtype=torch.long, device=distances_2d.device))
            # remap idx
            remap_dict = dict(zip(all_idx.cpu().tolist(), range(len(all_idx))))
            closest_idx_middle = {
                vis_only_idx.index(k): torch.tensor([remap_dict[_.cpu().item()] for _ in v], dtype=v.dtype, device=v.device)
                for k, v in closest_idx_middle.items()
            }
            closest_idx_exact = {
                vis_only_idx.index(k): torch.tensor([remap_dict[_.cpu().item()] for _ in v], dtype=v.dtype, device=v.device)
                for k, v in closest_idx_exact.items()
            }
            projected_2d = projected_2d[vis_only_idx]
            circle_2d_points = circle_2d_points[all_idx]
            circle_points_3d = circle_points_3d[all_idx]

        # Do another round of assignment, if the assignment_only_idx != vis_only_idx:
        #  Vis indices were used to generate the virtual cameras and their assignments. Assignment indices, will update the assignments, only in the direction of reassigning the points to fewer cameras.
        #  This is useful, in the case that I want to use e.g. 2 cameras' partial PCDs but visualize them to the virtual cameras corresponding e.g. to 4 cameras, keeping the trajectory the same while visualizing fewer cameras.
        if assignment_only_idx != vis_only_idx:
            # Compute distances between circle points and assignment_only_idx cameras
            assignment_only_idx_rel = [vis_only_idx.index(a) for a in assignment_only_idx]
            distances_2d_assignment = torch.linalg.norm(projected_2d[assignment_only_idx_rel][:, torch.newaxis] - circle_2d_points[None], dim=2)
            closest_idx_middle_values_assignment = torch.argmin(distances_2d_assignment, dim=0)  # Shape: (num_points,)
            # Map the closest_idx_middle_values_assignment to the original camera indices
            closest_idx_middle_values_assignment = torch.tensor(assignment_only_idx_rel, device=tvecs.device)[closest_idx_middle_values_assignment]
            # Update closest_idx_middle and closest_idx_exact based on the new assignments
            closest_idx_middle = {int(k.item()): unblock_idx((closest_idx_middle_values_assignment == k).nonzero(as_tuple=True)[0])
                                  for k in torch.unique(closest_idx_middle_values_assignment)}
            closest_idx_middle = dict(sorted(closest_idx_middle.items(), key=lambda item: item[1].max().item()))
            # Update closest_idx_exact
            if len(assignment_only_idx) == 1:
                closest_idx_exact = {}
            else:
                remaining_exact_keys = [_ for _ in closest_idx_exact.keys() if _ in closest_idx_middle and (_ + 1) % len(vis_only_idx) in closest_idx_middle]
                removed_exact_keys = [_ for _ in closest_idx_exact.keys() if _ not in remaining_exact_keys]  # Cameras removed from assignment_only_idx
                closest_idx_exact_keys = list(closest_idx_exact.keys())
                # Redistribute virtual cameras assigned to removed cameras to the remaining cameras
                for removed_key in removed_exact_keys:
                    # Find the closest remaining camera based on projected_2d positions
                    distances_to_remaining = torch.linalg.norm(projected_2d[removed_key][None] - projected_2d[remaining_exact_keys], dim=1)
                    closest_remaining_key = remaining_exact_keys[torch.argmin(distances_to_remaining).item()]
                    # Append the virtual cameras assigned to the removed camera to the closest remaining camera
                    exact_idx_of_removed = closest_idx_exact[removed_key]
                    exact_idx_of_closest_remaining = closest_idx_exact[closest_remaining_key]
                    closest_idx_exact[closest_remaining_key] = torch.cat([exact_idx_of_closest_remaining, exact_idx_of_removed]) if closest_idx_exact_keys.index(closest_remaining_key) < closest_idx_exact_keys.index(removed_key) \
                        else torch.cat([exact_idx_of_removed, exact_idx_of_closest_remaining])
                    del closest_idx_exact[removed_key]

        # 10) Plot projected camera positions, circle center, and circle points
        if vis is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter([cx.item()], [cy.item()], color='black', s=50, label="Circle Center")
            colors = cm.get_cmap('tab10', len(tvecs))
            for camera_idx, assigned in closest_idx_middle.items():
                tensor_idx = camera_idx
                ax.scatter(*projected_2d[tensor_idx].cpu().unbind(), color=colors(vis_only_idx[camera_idx]), edgecolor='black', s=150)
                ax.text(*(projected_2d[tensor_idx].cpu() + 0.03).unbind(), s=f'{vis_only_idx[camera_idx]:02d}')
                ax.scatter(*circle_2d_points[assigned].cpu().unbind(1), color=colors(vis_only_idx[camera_idx]), label=f"Virtual -> Camera {vis_only_idx[camera_idx]}")
            circle_2d_points_offset = torch.stack([
                cx + torch.cos(theta) * (radius * 1.05),
                cy + torch.sin(theta) * (radius * 1.05)
            ], dim=1)[all_idx]
            closest_idx_keys = list(closest_idx_middle.keys())
            for camera_idx, assigned in closest_idx_exact.items():
                ax.scatter(*circle_2d_points_offset[assigned].cpu().unbind(1), color=colors(vis_only_idx[camera_idx]), marker='x', label=f"Virtual -> Stereo {vis_only_idx[camera_idx]}-{vis_only_idx[closest_idx_keys[(closest_idx_keys.index(camera_idx) + 1) % len(closest_idx_keys)]]}")
            # compute arc length
            arc_angle = torch.rad2deg(torch.arccos(torch.dot(circle_2d_points[closest_idx_middle[closest_idx_keys[0]][0]] - circle_center_2d, circle_2d_points[closest_idx_middle[closest_idx_keys[-1]][-1]] - circle_center_2d) / radius ** 2))
            ax.set_title(f'Projected Camera Positions and Fitted Circle (arc angle = {arc_angle.item():.1f}°)')
            ax.set_xlabel("Plane X")
            ax.set_ylabel("Plane Y")
            ax.axis("equal")
            ax.legend()
            plt.savefig(str(vis))
            log(f'[virtual_tvecs] Plot saved at {vis}')

        return circle_points_3d, closest_idx_middle, closest_idx_exact

    @classmethod
    def pil(cls, img: torch.Tensor or Image, save_path: Optional[Path or str] = None) -> PIL.Image:
        """Convert an image tensor to a PIL image. Optionally, remove the background using the rembg library, and save
        the image to disk.

        Parameters:
        ----------
        img: torch.Tensor or PIL.Image
            The image tensor or PIL image.
        save_path: Path or str or None
            The path to save the image to. Default: None (don't save).

        Returns:
        -------
        PIL.Image
            The PIL image.
        """
        if isinstance(img, torch.Tensor):
            img = (img.squeeze() if img.ndim == 4 else img).cpu().numpy().transpose(1, 2, 0)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            if np.max(img) <= 1:
                if np.min(img) < 0:
                    img = (img + 1.0) / 2.0
                img = (img * 255).astype(np.uint8)
            elif np.max(img) <= 255:
                if np.min(img) < 0:
                    img = img + 127.5
                img = img.astype(np.uint8)
            # print(img.shape)
            # img = transforms.ToPILImage()(img.astype(np.uint8))
            img = PIL.Image.fromarray(img.squeeze(), mode='RGBA' if img.shape[2] == 4 else ('RGB' if img.shape[2] == 3 else 'L'))
        if save_path is not None:
            img.save(save_path)
        return img

    @classmethod
    def grid_pil(cls, imgs: List[Image or torch.Tensor], rows: int = 1, cols: int or None = None) -> Image:
        if cols is None:
            cols = len(imgs) // rows
            if len(imgs) % rows > 0:
                cols += 1
        assert len(imgs) == rows * cols

        if isinstance(imgs[0], torch.Tensor):
            to_pil = ToPILImage()
            imgs = [to_pil(img) for img in imgs]

        w, h = imgs[0].size
        grid = PIL.Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    @classmethod
    def reconstruction(cls, x: torch.Tensor, model: nn.Module, is_flow: bool = False) -> Image:
        # Get the reconstructed images
        with torch.no_grad():
            x_hat, _ = model(x)
            x_hat = x_hat.detach().cpu()
            x = x.detach().cpu()
        # Split tensor into separate tensors
        imgs_tensor = [
            t.squeeze()
            for t in torch.split(x, split_size_or_sections=1, dim=0)
        ]
        imgs_hat_tensor = [
            t.squeeze()
            for t in torch.split(x_hat, split_size_or_sections=1, dim=0)
        ]
        # Convert to PIL images
        to_pil = ToPILImage() if not is_flow else flow_to_image
        imgs = [to_pil(t) for t in imgs_tensor]
        imgs_hat = [to_pil(t) for t in imgs_hat_tensor]
        # Create grid
        grid = cls.grid_pil(imgs + imgs_hat, rows=2, cols=len(imgs))
        return grid

    @classmethod
    def text(cls, img: torch.Tensor, text: str, color: ColorT = '#000000', font: str or None = 'JetBrainsMono-Regular',
             font_size: int = 24, offset_v: int = 0, offset_h: int = 0) -> torch.Tensor:
        """Add text to an image.

        Parameters:
        ----------
        text: str
            The text to add to the image.
        img: torch.Tensor
            The image tensor.
        color: ColorT
            The text color.
        font: str
            The font to use. Default: 'JetBrainsMono-Regular' (will be automatically located if present in Linux).
        font_size: int
            The font size. Default: 24.
        margin: int
            The margin around the text. Default: 0.

        Returns:
        -------
        torch.Tensor
            The image tensor with the text.
        """
        # Get the image as a PIL image
        if img.ndim == 4 and img.shape[0] > 1:
            return torch.stack([
                cls.text(_, text=text, color=color, font=font, font_size=font_size, offset_v=offset_v, offset_h=offset_h)
                for _ in img
            ], dim=0)

        device = img.device
        img_pil = cls.pil(img.detach().cpu())
        # Setup drawer
        try:
            if font is not None:
                font_path = PathUtils.font(font)
                font = ImageFont.truetype(font=str(font_path), size=font_size)
            else:
                font = ImageFont.load_default(size=font_size)
        except OSError:
            font = ImageFont.load_default(size=font_size)
        drawer = ImageDraw.Draw(img_pil)
        tw, th = drawer.textbbox(xy=(0, 0), text=text, font=font)[2:]
        iw, ih = img_pil.width, img_pil.height
        assert tw < iw and th < ih, f'[VisUtils::text] text too large for image: {tw}x{th} > ' \
                                    f'{iw}x{ih} pixels'
        # Draw text
        drawer.text(((iw - tw) // 2 + offset_h, (ih - th) // 2 + offset_v), text, fill=color, font=font)
        # Convert back to tensor
        img_tensor = transforms.PILToTensor()(img_pil).to(device)
        # Set alpha channel
        if img.shape[0] == 4 and img_tensor.shape[0] == 3:
            img_tensor = torch.cat((img_tensor, img[-1].detach().unsqueeze(0)), dim=0)
        # Normalize
        if img.max() <= 1 < img_tensor.max():
            img_tensor = img_tensor / 255.0
            if img.min() < 0:
                img_tensor = img_tensor * 2.0 - 1.0
        return img_tensor

if __name__ == '__main__':
    # a = VisUtils.arrow_tensor(width_px=5, length_px=100, color='#E3003B', bg_color='#FFFFFF00')
    # a_pil = VisUtils.pil(a, save_path='arrow.png')
    # plt.imshow(a_pil)
    # # plt.imshow(transforms.ToPILImage()((a - 127.5)/127.5))
    # plt.show()

    # imgs = VisUtils.grid(
    #     2 * torch.rand(10, 3, 512, 512) - 1,
    #     2 * torch.rand(10, 3, 512, 512) - 1,
    #     2 * torch.rand(10, 2, 512, 512) - 1,
    #     2 * torch.rand(10, 2, 512, 512) - 1,
    #     color='#E3003B', bg_color='#FFFFFF00', padding_px=100, border_px=10)
    # imgs_pil = VisUtils.pil(imgs[0], save_path='grid.png')

    padding_ = VisUtils.color_tensor([100, 500], color='#FFFFFFFF')
    padding_ = VisUtils.text(padding_, 'Hello World!', font_size=50)
    padding_pil = VisUtils.pil(padding_, save_path='grid.png')
    plt.imshow(padding_pil)
    plt.show()
