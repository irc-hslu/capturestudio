# COPYRIGHT 2024 by Athanasios Charisoudis <athanasios.charisoudis@ieee.org>
# Licensed under the Apache License, Version 2.0 (the "License");
# Original source: https://github.com/charisoudis/flowforge

import os
from pathlib import Path
from typing import Optional, Union, List

import cv2
import numpy as np
import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as F


class FlowUtils:
    @classmethod
    def size_to_divisible_by(cls, H, W, by=32):
        # Calculate the aspect ratio
        aspect_ratio = W / H

        # Find the maximum height divisible by `by`
        new_H = (H // by) * by
        # Calculate the corresponding width to maintain aspect ratio
        new_W = int(new_H * aspect_ratio)
        # Ensure the new width is also divisible by 32
        new_W = (new_W // by) * by

        # Alternatively, find the maximum width divisible by 32
        # and calculate the corresponding height
        alt_W = (W // by) * by
        alt_H = int(alt_W / aspect_ratio)
        alt_H = (alt_H // by) * by

        # Choose the larger of the two options to maximize the size
        if (new_H * new_W) > (alt_H * alt_W):
            return new_H, new_W
        return alt_H, alt_W

    @classmethod
    def make_colorwheel(cls, device=None):
        """
        Generate color wheel according to the Middlebury color code.
        Returns:
            torch.Tensor: Color wheel of shape [55, 3] on the specified device.
        """
        # Color transitions
        RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = torch.zeros([ncols, 3], device=device)
        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
        col += RY
        # YG
        colorwheel[col:col + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
        colorwheel[col:col + YG, 1] = 255
        col += YG
        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
        col += GC
        # CB
        colorwheel[col:col + CB, 1] = 255 - torch.floor(255 * torch.arange(0, CB) / CB)
        colorwheel[col:col + CB, 2] = 255
        col += CB
        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
        col += BM
        # MR
        colorwheel[col:col + MR, 2] = 255 - torch.floor(255 * torch.arange(0, MR) / MR)
        colorwheel[col:col + MR, 0] = 255
        return colorwheel

    @classmethod
    def flow_uv_to_colors(cls, u, v, convert_to_bgr=False):
        """
        Convert batched flow U and V components to RGB images using the color wheel.

        Args:
            u (torch.Tensor): Horizontal flow of shape [B, H, W]
            v (torch.Tensor): Vertical flow of shape [B, H, W]
            convert_to_bgr (bool, optional): Convert output images to BGR. Defaults to False.

        Returns:
            torch.Tensor: Flow visualization images of shape [B, 3, H, W]
        """
        B, H, W = u.shape
        nan_mask = torch.isnan(u) | torch.isnan(v)
        u = u.clone()
        v = v.clone()
        u[nan_mask] = 0
        v[nan_mask] = 0

        # Ensure colorwheel is on the same device as u and v
        colorwheel = cls.make_colorwheel(device=u.device)  # Shape [ncols, 3]
        ncols = colorwheel.shape[0]

        rad = torch.sqrt(u ** 2 + v ** 2)  # Shape [B, H, W]
        a = torch.atan2(-v, -u) / torch.pi  # Shape [B, H, W]

        fk = (a + 1) / 2 * (ncols - 1)  # Shape [B, H, W]
        k0 = torch.floor(fk).long()  # Shape [B, H, W]
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0.float()

        # Normalize colorwheel to [0, 1]
        colorwheel = colorwheel / 255.0  # Shape [ncols, 3]

        # Flatten indices for indexing
        k0 = k0.view(-1)  # Shape [B*H*W]
        k1 = k1.view(-1)
        f = f.view(-1, 1)

        # Index colorwheel
        col0 = colorwheel[k0]  # Shape [B*H*W, 3]
        col1 = colorwheel[k1]  # Shape [B*H*W, 3]

        # Interpolate between neighboring colors
        col = (1 - f) * col0 + f * col1  # Shape [B*H*W, 3]
        col = col.view(B, H, W, 3)  # Shape [B, H, W, 3]

        # Adjust saturation
        rad_max = rad.view(B, -1).max(dim=1)[0].view(B, 1, 1) + 1e-5  # Shape [B, 1, 1]
        rad_norm = rad / rad_max  # Shape [B, H, W]

        # Create masks
        idx = rad_norm <= 1  # Shape [B, H, W]
        idx = idx.unsqueeze(-1)  # Shape [B, H, W, 1]

        # Adjust colors based on radius
        col = torch.where(
            idx,
            1 - rad_norm.unsqueeze(-1) * (1 - col),
            col * 0.75
        )

        # Set NaNs to zero
        nan_mask = nan_mask.unsqueeze(-1).expand_as(col)
        col = torch.where(nan_mask, torch.zeros_like(col), col)

        if convert_to_bgr:
            col = col[..., [2, 1, 0]]

        # Transpose to get [B, 3, H, W]
        col = col.permute(0, 3, 1, 2).contiguous()

        return col.mul(255).add_(0.5).clamp_(0, 255).byte()

    @classmethod
    def flow_to_image(cls, flow_uv, clip_flow: Optional[float] = None, convert_to_bgr: bool = False, resize_to: Optional[Union[int, List[int]]] = None):
        """
        Converts a batch of flow tensors to visual RGB images.

        Args:
            flow_uv (torch.Tensor): Flow UV images of shape [B, 2, H, W]
            clip_flow (float, optional): Clip maximum flow values. Defaults to None.
            convert_to_bgr (bool, optional): Convert output images to BGR. Defaults to False.
            resize_to (Union[int, Tuple[int, int]], optional): Resize flow to this size. Defaults to None.

        Returns:
            torch.Tensor: Flow visualization images of shape [B, 3, H, W]
        """
        assert flow_uv.dim() == 4, 'Input flow must have four dimensions [B, 2, H, W]'
        assert flow_uv.shape[1] == 2, 'Input flow must have shape [B, 2, H, W]'

        if clip_flow is not None:
            flow_uv = torch.clamp(flow_uv, -clip_flow, clip_flow)

        if resize_to is not None:
            flow_uv = F.resize(flow_uv, size=resize_to if isinstance(resize_to, (list, tuple)) else (resize_to, resize_to))

        u = flow_uv[:, 0, :, :]  # Shape [B, H, W]
        v = flow_uv[:, 1, :, :]  # Shape [B, H, W]

        rad = torch.sqrt(u ** 2 + v ** 2)  # Shape [B, H, W]
        rad_max = rad.view(rad.size(0), -1).max(dim=1)[0].view(-1, 1, 1)  # Shape [B, 1, 1]
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)

        return cls.flow_uv_to_colors(u, v, convert_to_bgr)

    @classmethod
    def read_flow_from_flo(cls, path: Path or str):
        """ Read .flo file in Middlebury format"""
        assert Path(path).suffix == '.flo'
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                raise ValueError('Magic number incorrect. Invalid .flo file')
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return torch.from_numpy(np.resize(data, (int(h), int(w), 2))).moveaxis(-1, 0)

    @classmethod
    def write_flow_to_flo(cls, uv: torch.Tensor, path: Path or str, v=None):
        """
        flow: (H, W, 2)
        """
        nBands = 2
        if isinstance(uv, torch.Tensor) and uv.shape[0] == 2:
            uv = uv.permute(1, 2, 0)  # [2, H, W] --> [H, W, 2]
        if v is None:
            assert (uv.ndim == 3)
            assert (uv.shape[2] == 2)
            u = uv[:, :, 0]
            v = uv[:, :, 1]
        else:
            u = uv
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        assert (u.shape == v.shape)
        height, width = u.shape
        f = open(str(path), 'wb')
        # write the header
        f.write(np.array([202021.25], np.float32))
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        f.close()

    @staticmethod
    def read_flow_from_png(path: Path or str):
        """
        Reads optical flow from a PNG file in the KITTI benchmark format.

        Args:
            path (Path or str): Input filename

        Returns:
            torch.Tensor: Optical flow tensor of shape [2, H, W]
        """
        assert Path(path).suffix == '.png'

        # Read the image using OpenCV
        I_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if I_bgr is None:
            raise ValueError(f"Image file {path} could not be read")

        # Convert to RGB format
        I = I_bgr[:, :, ::-1]

        # Extract U, V, and valid mask
        u_scaled = I[:, :, 0].astype(np.float32)
        v_scaled = I[:, :, 1].astype(np.float32)
        valid_mask = I[:, :, 2].astype(np.uint8)
        valid_mask = np.minimum(valid_mask, 1)

        # Recover flow components
        u = (u_scaled - 2 ** 15) / 64.0
        v = (v_scaled - 2 ** 15) / 64.0

        # Apply valid mask
        u[valid_mask == 0] = 0
        v[valid_mask == 0] = 0

        # Stack to get flow tensor of shape [2, H, W]
        flow = np.stack((u, v), axis=0)

        # Convert to PyTorch tensor
        flow_tensor = torch.from_numpy(flow)

        return flow_tensor

    @classmethod
    def write_flow_to_png(cls, flow: torch.Tensor, path: Path or str):
        """
        Writes optical flow to a PNG file in the KITTI benchmark format.

        Args:
            flow (torch.Tensor): Optical flow tensor of shape [2, H, W]
            path (Path or str): Output filename
        """
        assert Path(path).suffix == '.png'
        assert flow.ndim == 3 and flow.shape[0] == 2, "Flow must have shape [2, H, W]"

        # Extract U and V components
        u = flow[0, :, :].cpu().numpy()
        v = flow[1, :, :].cpu().numpy()

        # Scale and shift
        u_scaled = (u * 64.0 + 2 ** 15).clip(0, 2 ** 16 - 1)
        v_scaled = (v * 64.0 + 2 ** 15).clip(0, 2 ** 16 - 1)

        # Convert to uint16
        u_scaled = u_scaled.astype(np.uint16)
        v_scaled = v_scaled.astype(np.uint16)

        # Valid mask (assumed to be valid everywhere)
        valid_mask = np.ones_like(u_scaled, dtype=np.uint16)

        # Stack channels to form (H, W, 3)
        I = np.stack((u_scaled, v_scaled, valid_mask), axis=2)

        # OpenCV expects images in BGR format
        I_bgr = I[:, :, ::-1]  # Reverse the channels to BGR

        # Write the image using OpenCV
        cv2.imwrite(path, I_bgr)

    @staticmethod
    def read_flow_from_png_custom(path: Path or str):
        """
        Reads optical flow from a PNG file in a modified KITTI benchmark format.

        Args:
            path (Path or str): Input filename

        Returns:
            torch.Tensor: Optical flow tensor of shape [2, H, W]
        """
        assert Path(path).suffix == '.png'

        # Read the image using OpenCV
        I_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if I_bgr is None:
            raise ValueError(f"Image file {path} could not be read")

        # Convert to RGB format
        I = I_bgr[:, :, ::-1]

        # Extract U, V, and valid mask
        u_scaled = I[:, :, 0].astype(np.float32)
        v_scaled = I[:, :, 1].astype(np.float32)
        valid_mask = I[:, :, 2].astype(np.uint8)
        valid_mask = np.minimum(valid_mask, 1)

        # Recover flow components
        u = (u_scaled - 2 ** 14) / 512.0
        v = (v_scaled - 2 ** 14) / 512.0

        # Apply valid mask
        u[valid_mask == 0] = 0
        v[valid_mask == 0] = 0

        # Stack to get flow tensor of shape [2, H, W]
        flow = np.stack((u, v), axis=0)

        # Convert to PyTorch tensor
        flow_tensor = torch.from_numpy(flow)

        return flow_tensor

    @classmethod
    def write_flow_to_png_custom(cls, flow: torch.Tensor, path: Optional[Union[Path, str]] = None, return_bytes=True):
        """
        Writes optical flow to a PNG file in a modified KITTI benchmark format.

        Args:
            flow (torch.Tensor): Optical flow tensor of shape [2, H, W]
            path (Path or str): Output filename
            return_bytes (bool): Return bytes instead of str
        """
        assert return_bytes or (path is not None and Path(path).suffix == '.png')
        assert flow.ndim == 3 and flow.shape[0] == 2, "Flow must have shape [2, H, W]"

        # Extract U and V components
        u = flow[0, :, :].cpu().numpy()
        v = flow[1, :, :].cpu().numpy()

        # Scale and shift
        u_scaled = (u * 512.0 + 2 ** 14).clip(0, 2 ** 16 - 1)
        v_scaled = (v * 512.0 + 2 ** 14).clip(0, 2 ** 16 - 1)

        # Convert to uint16
        u_scaled = u_scaled.astype(np.uint16)
        v_scaled = v_scaled.astype(np.uint16)

        # Valid mask (assumed to be valid everywhere)
        valid_mask = np.ones_like(u_scaled, dtype=np.uint16)

        # Stack channels to form (H, W, 3)
        I = np.stack((u_scaled, v_scaled, valid_mask), axis=2)

        # OpenCV expects images in BGR format
        I_bgr = I[:, :, ::-1]  # Reverse the channels to BGR

        # Write the image using OpenCV (robust implementation)
        # cv2.imwrite(path, I_bgr)
        is_success, buffer = cv2.imencode('.png', I_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not is_success:
            raise RuntimeError("cv2.imencode failed while writing flow to path: " + str(path))
        if return_bytes:
            return buffer

        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            with open(temp_path, "wb") as f:
                f.write(buffer)
                f.flush()
                os.fsync(f.fileno())
            os.rename(temp_path, path)
        except Exception as e:
            if temp_path.exists():
                os.remove(temp_path)
            raise e

    @staticmethod
    def read_flow_from_png_full(path: Path or str) -> torch.Tensor:
        assert Path(path).suffix == '.png'

        I_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if I_bgr is None:
            raise ValueError(f"Could not read image {path}")

        I = I_bgr[:, :, ::-1]
        u_scaled = I[:, :, 0].astype(np.float32)
        v_scaled = I[:, :, 1].astype(np.float32)
        valid_mask = I[:, :, 2]
        H, W = u_scaled.shape

        def decode_minmax(scaled_val, bound):
            norm = scaled_val / 65535.0
            return (2 * norm - 1) * bound

        u_min = decode_minmax(u_scaled[0, 0], H)
        u_max = decode_minmax(u_scaled[0, 1], H)
        v_min = decode_minmax(v_scaled[0, 0], W)
        v_max = decode_minmax(v_scaled[0, 1], W)

        denom_u = u_max - u_min
        denom_v = v_max - v_min

        u_norm = u_scaled / 65535.0
        v_norm = v_scaled / 65535.0
        u = u_norm * denom_u + u_min
        v = v_norm * denom_v + v_min

        u[valid_mask == 0] = 0
        v[valid_mask == 0] = 0

        flow = np.stack((u, v), axis=0)
        return torch.from_numpy(flow)

    @classmethod
    def write_flow_to_png_full(cls, flow: torch.Tensor, path: Path or str) -> None:
        assert Path(path).suffix == '.png'
        assert flow.ndim == 3 and flow.shape[0] == 2

        u = flow[0].cpu().numpy()
        v = flow[1].cpu().numpy()

        u_min, u_max = np.min(u), np.max(u)
        v_min, v_max = np.min(v), np.max(v)

        denom_u = u_max - u_min
        denom_v = v_max - v_min
        u_norm = np.zeros_like(u)
        v_norm = np.zeros_like(v)
        if denom_u > 1e-6:
            u_norm = (u - u_min) / denom_u
        if denom_v > 1e-6:
            v_norm = (v - v_min) / denom_v

        u_scaled = (u_norm * 65535).astype(np.uint16)
        v_scaled = (v_norm * 65535).astype(np.uint16)

        # Store min/max values encoded into [0,1]
        def encode_minmax(val, bound):
            norm = ((val / bound) + 1) / 2
            return np.clip(norm * 65535, 0, 65535).astype(np.uint16)

        u_scaled[0, 0] = encode_minmax(u_min, u.shape[0])
        u_scaled[0, 1] = encode_minmax(u_max, u.shape[0])
        v_scaled[0, 0] = encode_minmax(v_min, v.shape[1])
        v_scaled[0, 1] = encode_minmax(v_max, v.shape[1])

        valid_mask = np.ones_like(u_scaled, dtype=np.uint16)
        valid_mask[0, :2] = 0  # reserve pixels for min/max

        I = np.stack((u_scaled, v_scaled, valid_mask), axis=2)
        cv2.imwrite(str(path), I[:, :, ::-1])

    @classmethod
    def warp_frame_from_flow(cls, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp `frame` backward using `flow`.

        Args:
            frame: Tensor of shape [1, C, H, W]
            flow:  Tensor of shape [1, 2, H, W] — flow from frame_t to frame_t+1

        Returns:
            Warped frame of shape [1, C, H, W]
        """
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)
        if flow.ndim == 3:
            flow = flow.unsqueeze(0)
        B, C, H, W = frame.shape

        # Create normalized mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, H, W]
        grid = grid.to(frame.device)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]

        # Invert the flow: move pixels backward
        sampling_grid = grid + flow
        sampling_grid[:, 0, :, :] = 2.0 * sampling_grid[:, 0, :, :] / (W - 1) - 1.0  # x: normalize to [-1, 1]
        sampling_grid[:, 1, :, :] = 2.0 * sampling_grid[:, 1, :, :] / (H - 1) - 1.0  # y

        # Rearrange to [B, H, W, 2] for grid_sample
        sampling_grid = sampling_grid.permute(0, 2, 3, 1)

        # Sample from input image
        warped = torch.nn.functional.grid_sample(frame, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

    @classmethod
    def resize_crop_flow(
            cls,
            flow: torch.Tensor,
            out_h: int = 1024,
            out_w: int = 1024,
            keep_aspect: bool = True,
            crop_mode: str = "center",  # "center" or ("top","left") ints via crop_offset
            crop_offset: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Resize a (2, H, W) optical flow and crop to (out_h, out_w).

        Args:
            flow: (2, H, W) float tensor, flow[0]=u (x), flow[1]=v (y) in *pixels*.
            out_h, out_w: desired output size.
            keep_aspect: if True, isotropic scale so both dims >= target, then crop.
                         if False, anisotropic resize directly to (out_h, out_w).
            crop_mode: "center" or use crop_offset (top,left) explicitly.
            crop_offset: (top, left) if you want a specific crop (only used when keep_aspect=True).

        Returns:
            (2, out_h, out_w) float tensor.
        """
        assert flow.ndim == 3 and flow.shape[0] == 2, "flow must be (2, H, W)"
        _, H, W = flow.shape
        flow = flow.unsqueeze(0)  # (1,2,H,W)

        if keep_aspect:
            # scale so that the resized frame covers the target in both dims
            s = max(out_h / H, out_w / W)  # isotropic scale
            H1 = max(out_h, int(round(H * s)))
            W1 = max(out_w, int(round(W * s)))

            # resize with bilinear; align_corners=True keeps scaling consistent
            flow_rs = nnF.interpolate(flow, size=(H1, W1), mode="bilinear", align_corners=True)
            # scale both components by the same factor (isotropic)
            flow_rs[:, 0] *= s  # u (x)
            flow_rs[:, 1] *= s  # v (y)

            # crop
            if crop_offset is not None:
                top, left = crop_offset
            elif crop_mode == "center":
                top = (H1 - out_h) // 2
                left = (W1 - out_w) // 2
            else:
                raise ValueError("Unsupported crop_mode; use 'center' or supply crop_offset.")

            top = max(0, min(top, H1 - out_h))
            left = max(0, min(left, W1 - out_w))

            flow_out = flow_rs[:, :, top:top + out_h, left:left + out_w].squeeze(0)

        else:
            # anisotropic resize directly to the target, then scale u/v per axis
            flow_rs = nnF.interpolate(flow, size=(out_h, out_w), mode="bilinear", align_corners=True)
            sx = out_w / W
            sy = out_h / H
            flow_rs[:, 0] *= sx  # u (x) scales with width
            flow_rs[:, 1] *= sy  # v (y) scales with height
            flow_out = flow_rs.squeeze(0)

        return flow_out
