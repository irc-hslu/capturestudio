"""
CUDA-accelerated Orbbec-style Depth-to-Color alignment.

This module intentionally implements D2C only. C2D was removed to keep the
runtime path small and optimized for batch/offline depth rendering.

The expensive part of high-quality D2C is forward rasterization of each depth
pixel footprint into the color image. This implementation pushes that operation
into a custom PyTorch CUDA extension:

    one CUDA thread per depth pixel
    project the four pixel corners into color space
    conservative quad rasterization over the projected footprint
    atomicMin into a per-frame z-buffer
    optional tiny-hole fill kernel

Depth units:
    depth_buffer is uint16 in depth units.
    depth_unit_mm is millimeters per depth unit.
    extrinsic.trans is assumed to be in millimeters. Internally translation is
    divided by depth_unit_mm so the output depth remains in input depth units.

Important distortion note:
    Set add_target_distortion=True only when your RGB target image is still in
    raw/distorted color-camera coordinates. If the RGB stream is already
    rectified/undistorted by the SDK/ISP, set add_target_distortion=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

EPSILON = 1e-6
UINT16_MAX = np.iinfo(np.uint16).max


class DistortionModel(Enum):
    OB_DISTORTION_NONE = 0
    OB_DISTORTION_BROWN_CONRADY = 1
    OB_DISTORTION_BROWN_CONRADY_K6 = 2
    OB_DISTORTION_KANNALA_BRANDT4 = 3

    @classmethod
    def coerce(cls, value: Union["DistortionModel", int, str]) -> "DistortionModel":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            if value in cls.__members__:
                return cls[value]
            upper = value.upper()
            if upper in cls.__members__:
                return cls[upper]
        raise ValueError(f"Unsupported distortion model: {value!r}")


@dataclass(frozen=True)
class OBCameraIntrinsic:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self, name: str = "intrinsic") -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{name}: width and height must be positive")
        if self.width * self.height > 100_000_000:
            raise ValueError(f"{name}: dimensions are unreasonably large")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(f"{name}: fx and fy must be positive")


@dataclass(frozen=True)
class OBCameraDistortion:
    model: DistortionModel = DistortionModel.OB_DISTORTION_NONE
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", DistortionModel.coerce(self.model))

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.p1, self.p2],
            dtype=np.float32,
        )

    def has_nonzero_coefficients(self) -> bool:
        return bool(np.any(np.abs(self.as_array()) > 0.0))


@dataclass(frozen=True)
class OBExtrinsic:
    """Depth-to-color extrinsic: P_color = R * P_depth + t."""

    rot: np.ndarray
    trans: np.ndarray

    def normalized(self) -> "OBExtrinsic":
        rot = np.asarray(self.rot, dtype=np.float32)
        trans = np.asarray(self.trans, dtype=np.float32)

        if rot.shape == (9,):
            rot = rot.reshape(3, 3)
        if rot.shape != (3, 3):
            raise ValueError(f"extrinsic.rot must be shape (3,3) or (9,), got {rot.shape}")

        trans = trans.reshape(-1)
        if trans.shape != (3,):
            raise ValueError(f"extrinsic.trans must be shape (3,), got {trans.shape}")

        return OBExtrinsic(rot=rot.copy(), trans=trans.copy())


def add_distortion_vectorized(disto: OBCameraDistortion, xy: np.ndarray) -> np.ndarray:
    """Apply Orbbec/OpenCV-style distortion to normalized coordinates."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape[-1] != 2:
        raise ValueError("xy must have last dimension 2")

    if disto.model == DistortionModel.OB_DISTORTION_NONE or not disto.has_nonzero_coefficients():
        return xy.copy()

    x = xy[..., 0]
    y = xy[..., 1]
    r2 = x * x + y * y

    if disto.model in (DistortionModel.OB_DISTORTION_BROWN_CONRADY, DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6):
        r4 = r2 * r2
        r6 = r4 * r2

        if disto.model == DistortionModel.OB_DISTORTION_BROWN_CONRADY:
            radial = 1.0 + disto.k1 * r2 + disto.k2 * r4 + disto.k3 * r6
        else:
            numerator = 1.0 + disto.k1 * r2 + disto.k2 * r4 + disto.k3 * r6
            denominator = 1.0 + disto.k4 * r2 + disto.k5 * r4 + disto.k6 * r6
            radial = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan, dtype=np.float32),
                where=np.abs(denominator) > EPSILON,
            )

        two_xy = 2.0 * x * y
        x_tangential = disto.p2 * (r2 + 2.0 * x * x) + disto.p1 * two_xy
        y_tangential = disto.p1 * (r2 + 2.0 * y * y) + disto.p2 * two_xy
        xd = x * radial + x_tangential
        yd = y * radial + y_tangential
        return np.stack([xd, yd], axis=-1).astype(np.float32, copy=False)

    if disto.model == DistortionModel.OB_DISTORTION_KANNALA_BRANDT4:
        r = np.sqrt(r2)
        theta = np.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        theta_d = theta * (1.0 + disto.k1 * theta2 + disto.k2 * theta4 + disto.k3 * theta6 + disto.k4 * theta8)
        scale = np.divide(theta_d, r, out=np.ones_like(r, dtype=np.float32), where=r > EPSILON)
        return np.stack([x * scale, y * scale], axis=-1).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported distortion model: {disto.model}")


def remove_distortion_vectorized(disto: OBCameraDistortion, xy_distorted: np.ndarray, iterations: int = 12) -> np.ndarray:
    """Iteratively invert distortion for normalized coordinates."""
    xy_distorted = np.asarray(xy_distorted, dtype=np.float32)
    if disto.model == DistortionModel.OB_DISTORTION_NONE or not disto.has_nonzero_coefficients():
        return xy_distorted.copy()

    und = xy_distorted.copy()
    target = xy_distorted
    for _ in range(iterations):
        redistorted = add_distortion_vectorized(disto, und)
        und = und - (redistorted - target)
        bad = ~np.isfinite(und[..., 0]) | ~np.isfinite(und[..., 1])
        if np.any(bad):
            und[bad] = target[bad]
            break
    return und.astype(np.float32, copy=False)


def _load_extension():
    """Import installed extension, or optionally JIT-build it from local sources."""
    import os
    from torch.utils.cpp_extension import load
    from utils.misc import PathUtils, env_get

    # Compile and load the CUDA extension
    if env_get('CC', None) is not None:
        os.environ["CC"] = env_get('CC')
    if env_get('CXX', None) is not None:
        os.environ["CXX"] = env_get('CXX')
    if env_get('CUDA_INCLUDE_PATH', None) is not None:
        extra_include_paths = [env_get('CUDA_INCLUDE_PATH')]
    else:
        extra_include_paths = []
    # Compile and load the CUDA extension with the new filenames
    return load(
        name="orbbec_d2c_cuda_jit",
        sources=[
            str(PathUtils.torch_extension_path('orbbec_d2c') / "d2c.cpp"),
            str(PathUtils.torch_extension_path('orbbec_d2c') / "d2c_kernel.cu"),
        ],
        extra_cflags=["-O3", '-std=c++17'],
        extra_cuda_cflags=["-O3", '--use_fast_math'],
        extra_include_paths=extra_include_paths,
    )


class AlignD2CCUDA:
    """CUDA-only D2C aligner.

    The class accepts one depth frame with shape (H, W) or a batch with shape
    (B, H, W). Passing a CUDA torch tensor avoids CPU->GPU copy overhead.
    """

    _CORNER_OFFSETS: Tuple[Tuple[float, float], ...] = (
        (-0.5, -0.5),
        (+0.5, -0.5),
        (+0.5, +0.5),
        (-0.5, +0.5),
        (0.0, 0.0),  # center, used for z value and optional fallback behavior
    )

    def __init__(self, device: Optional[Union[str, "torch.device"]] = None, compile_extension: bool = True) -> None:
        self.initialized = False
        self.depth_unit_mm = 1.0
        self.depth_intric: Optional[OBCameraIntrinsic] = None
        self.depth_disto: Optional[OBCameraDistortion] = None
        self.rgb_intric: Optional[OBCameraIntrinsic] = None
        self.rgb_disto: Optional[OBCameraDistortion] = None
        self.transform: Optional[OBExtrinsic] = None
        self.add_target_distortion = False
        self.k6_r2_limit: Optional[float] = None
        self.need_to_undistort_depth = False
        self.scaled_trans_np = np.zeros(3, dtype=np.float32)

        self.device = device
        self._coeffs_by_device: Dict[str, "torch.Tensor"] = {}
        self._trans_by_device: Dict[str, "torch.Tensor"] = {}
        self._dist_by_device: Dict[str, "torch.Tensor"] = {}
        self._ext = _load_extension() if compile_extension else None

    def initialize(
            self,
            depth_intri: OBCameraIntrinsic,
            depth_dist: OBCameraDistortion,
            color_intri: OBCameraIntrinsic,
            color_dist: OBCameraDistortion,
            depth2color_extri: OBExtrinsic,
            depth_unit_mm: float = 1.0,
            add_target_distortion: bool = False,
            k6_r2_limit: Optional[float] = None,
    ) -> "AlignD2CCUDA":
        if depth_unit_mm <= 0.0:
            raise ValueError("depth_unit_mm must be positive")
        depth_intri.validate("depth_intri")
        color_intri.validate("color_intri")
        extri = depth2color_extri.normalized()

        self.depth_intric = depth_intri
        self.depth_disto = depth_dist
        self.rgb_intric = color_intri
        self.rgb_disto = color_dist
        self.transform = extri
        self.depth_unit_mm = float(depth_unit_mm)
        self.add_target_distortion = bool(add_target_distortion)
        self.k6_r2_limit = k6_r2_limit
        self.scaled_trans_np = (extri.trans.astype(np.float32) / np.float32(self.depth_unit_mm)).astype(np.float32)
        self.need_to_undistort_depth = (
                depth_dist.model != DistortionModel.OB_DISTORTION_NONE and depth_dist.has_nonzero_coefficients()
        )

        self._coeffs_by_device.clear()
        self._trans_by_device.clear()
        self._dist_by_device.clear()
        self.initialized = True
        return self

    def _compute_coefficients_np(self, du: float, dv: float) -> np.ndarray:
        if self.depth_intric is None or self.depth_disto is None or self.transform is None:
            raise RuntimeError("AlignD2CCUDA is not initialized")

        h = self.depth_intric.height
        w = self.depth_intric.width
        yy, xx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        x = ((xx + np.float32(du)) - np.float32(self.depth_intric.cx)) / np.float32(self.depth_intric.fx)
        y = ((yy + np.float32(dv)) - np.float32(self.depth_intric.cy)) / np.float32(self.depth_intric.fy)
        xy = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1)

        if self.need_to_undistort_depth:
            xy = remove_distortion_vectorized(self.depth_disto, xy)

        R = self.transform.rot.astype(np.float32)
        coeff_x = R[0, 0] * xy[:, 0] + R[0, 1] * xy[:, 1] + R[0, 2]
        coeff_y = R[1, 0] * xy[:, 0] + R[1, 1] * xy[:, 1] + R[1, 2]
        coeff_z = R[2, 0] * xy[:, 0] + R[2, 1] * xy[:, 1] + R[2, 2]
        return np.stack([coeff_x, coeff_y, coeff_z], axis=-1).astype(np.float32, copy=False)

    def _coeffs_np(self) -> np.ndarray:
        return np.stack([self._compute_coefficients_np(du, dv) for du, dv in self._CORNER_OFFSETS], axis=0)

    def _resolve_device(self, depth_buffer=None):
        if depth_buffer is not None and torch.is_tensor(depth_buffer):
            if not depth_buffer.is_cuda:
                if self.device is not None:
                    return torch.device(self.device)
                return torch.device("cuda")
            return depth_buffer.device
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda")

    def _device_key(self, device) -> str:
        return str(device)

    def _ensure_extension(self):
        if self._ext is None:
            self._ext = _load_extension()
        return self._ext

    def _get_cached_tensors(self, device):
        if not self.initialized or self.rgb_disto is None:
            raise RuntimeError("AlignD2CCUDA is not initialized")

        key = self._device_key(device)
        if key not in self._coeffs_by_device:
            coeffs = self._coeffs_np()
            self._coeffs_by_device[key] = torch.from_numpy(coeffs).to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            self._trans_by_device[key] = torch.from_numpy(self.scaled_trans_np).to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            self._dist_by_device[key] = torch.from_numpy(self.rgb_disto.as_array()).to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        return self._coeffs_by_device[key], self._trans_by_device[key], self._dist_by_device[key]

    def prepare(self, device: Optional[Union[str, "torch.device"]] = None) -> None:
        """Precompute/upload coefficient tensors before entering a frame-processing loop."""
        dev = torch.device(device) if device is not None else self._resolve_device()
        self._get_cached_tensors(dev)

    def D2C(
            self,
            depth_buffer: Union[np.ndarray, "torch.Tensor"],
            *,
            out_depth: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
            return_torch: Optional[bool] = None,
            device: Optional[Union[str, "torch.device"]] = None,
            max_depth_value: int = 65535,
            max_footprint_px: int = 32,
            conservative_raster: bool = True,
            center_fallback: bool = True,
            fill_holes: bool = False,
            hole_radius: int = 2,
            hole_max_depth_delta: int = 80,
            hole_min_valid_neighbors: int = 10,
            hole_fill_iterations: int = 1,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Align depth to the color camera image plane using CUDA.

        Args:
            depth_buffer: uint16 depth image, shape (H,W), or batch shape (B,H,W).
                Passing a CUDA torch tensor avoids host-device transfer.
            out_depth: optional destination buffer. If NumPy, result is copied back to CPU.
                If torch CUDA tensor, result is copied on-device.
            return_torch: default is True for torch input, False for NumPy input.
            max_depth_value: output values are clipped to [1, max_depth_value].
            max_footprint_px: reject absurd projected footprints wider/taller than this.
            conservative_raster: also tests pixel corners, reducing pinholes versus strict
                center-sample rasterization.
            center_fallback: writes the projected center pixel if a tiny quad covers no
                sampled target pixel. This removes sparse black speckles with minimal dilation.
            fill_holes: optional CUDA local median fill for residual tiny zero pixels.
            hole_radius: local fill radius, clamped to [1,3] inside the CUDA kernel.
            hole_max_depth_delta: fill only when local valid depths are coherent.
            hole_min_valid_neighbors: minimum valid neighbors required to fill a zero pixel.
            hole_fill_iterations: usually 1. Use 2 for slightly stronger speckle fill.
        """
        if not self.initialized:
            raise RuntimeError("AlignD2CCUDA is not initialized")
        if self.depth_intric is None or self.rgb_intric is None or self.rgb_disto is None:
            raise RuntimeError("AlignD2CCUDA is missing calibration")

        ext = self._ensure_extension()
        dev = torch.device(device) if device is not None else self._resolve_device(depth_buffer)
        if dev.type != "cuda":
            raise RuntimeError("D2C CUDA requires a CUDA device")

        input_was_torch = torch.is_tensor(depth_buffer)
        if return_torch is None:
            return_torch = input_was_torch

        if input_was_torch:
            depth_t = depth_buffer.to(device=dev, non_blocking=True)
        else:
            arr = np.asarray(depth_buffer)
            if arr.dtype != np.uint16:
                raise TypeError("depth_buffer must be uint16")
            depth_t = torch.from_numpy(np.ascontiguousarray(arr)).to(device=dev, non_blocking=True)

        if depth_t.dtype != torch.uint16:
            raise TypeError(f"depth_buffer must be torch.uint16, got {depth_t.dtype}")
        if depth_t.ndim not in (2, 3):
            raise ValueError("depth_buffer must have shape (H,W) or (B,H,W)")
        if tuple(depth_t.shape[-2:]) != (self.depth_intric.height, self.depth_intric.width):
            raise ValueError(
                f"depth shape {tuple(depth_t.shape[-2:])} does not match initialized depth "
                f"shape {(self.depth_intric.height, self.depth_intric.width)}"
            )
        depth_t = depth_t.contiguous()

        coeffs, trans, dist_coeffs = self._get_cached_tensors(dev)
        k6_limit = -1.0 if self.k6_r2_limit is None else float(self.k6_r2_limit)

        result_t = ext.d2c_forward_cuda(
            depth_t,
            coeffs,
            trans,
            int(self.rgb_intric.height),
            int(self.rgb_intric.width),
            float(self.rgb_intric.fx),
            float(self.rgb_intric.fy),
            float(self.rgb_intric.cx),
            float(self.rgb_intric.cy),
            int(self.rgb_disto.model.value),
            bool(self.add_target_distortion),
            dist_coeffs,
            float(k6_limit),
            int(max_depth_value),
            int(max_footprint_px),
            bool(conservative_raster),
            bool(center_fallback),
            bool(fill_holes),
            int(hole_radius),
            int(hole_max_depth_delta),
            int(hole_min_valid_neighbors),
            int(hole_fill_iterations),
        )

        if out_depth is not None:
            if torch.is_tensor(out_depth):
                if out_depth.shape != result_t.shape:
                    raise ValueError(f"out_depth shape {tuple(out_depth.shape)} does not match result {tuple(result_t.shape)}")
                if out_depth.dtype != torch.uint16:
                    raise TypeError("out_depth torch tensor must be torch.uint16")
                out_depth.copy_(result_t.to(device=out_depth.device), non_blocking=True)
                return out_depth if return_torch else out_depth.cpu().numpy()

            out_arr = np.asarray(out_depth)
            expected = tuple(result_t.shape)
            if out_arr.shape != expected:
                raise ValueError(f"out_depth shape {out_arr.shape} does not match result {expected}")
            if out_arr.dtype != np.uint16:
                raise TypeError("out_depth NumPy array must be uint16")
            out_arr[...] = result_t.detach().cpu().numpy()
            return out_arr if not return_torch else torch.from_numpy(out_arr).to(device=dev, non_blocking=True)

        if return_torch:
            return result_t
        return result_t.detach().cpu().numpy()


# Backward-compatible class name for code that used AlignImpl before.
AlignImpl = AlignD2CCUDA

__all__ = [
    "AlignD2CCUDA",
    "AlignImpl",
    "DistortionModel",
    "OBCameraIntrinsic",
    "OBCameraDistortion",
    "OBExtrinsic",
    "add_distortion_vectorized",
    "remove_distortion_vectorized",
]
