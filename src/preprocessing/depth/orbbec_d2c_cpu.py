"""
High-quality Depth<->Color alignment for Orbbec-style calibration parameters.

This module intentionally separates three operations that are often mixed together:

1. Projection of depth pixels into the color camera image plane.
2. Depth-to-color forward rendering with a z-buffer.
3. Color-to-depth backward sampling with subpixel float maps and bilinear interpolation.

Quality defaults:
    * C2D uses float projection maps + bilinear interpolation.
    * D2C uses projected pixel-footprint quad rasterization + z-buffer when possible.
    * A faster 2x2 z-buffer splat is also available.

Important calibration note:
    Set add_target_distortion=True only when the target RGB image is in the raw/distorted
    color-camera coordinate system. If the RGB frame is already rectified/undistorted by
    the SDK/ISP, use add_target_distortion=False. Double-applying color distortion is a
    common cause of curved black invalid bands/lines.

Coordinate/units convention:
    * depth_buffer is uint16 in depth units.
    * depth_unit_mm is millimeters per depth unit.
    * extrinsic.trans is assumed to be in millimeters, matching common Orbbec/OpenNI-style
      calibration. Internally translation is divided by depth_unit_mm so output aligned
      depth is expressed in the same units as depth_buffer.

Dependencies:
    Required: numpy
    Optional: opencv-python for fastest CPU C2D remap
    Optional: torch for GPU C2D remap
    Optional: numba for fast high-quality D2C quad rasterization
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np

EPSILON = 1e-6
UINT16_MAX = np.iinfo(np.uint16).max

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    _HAS_CV2 = False

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    F = None
    _HAS_TORCH = False

try:
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    _HAS_NUMBA = False


class DistortionModel(Enum):
    """Subset of Orbbec-style distortion models used by this implementation."""

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
            # Support short names used in some codebases.
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
        if self.fx <= 0 or self.fy <= 0:
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

    def has_nonzero_coefficients(self) -> bool:
        coeffs = (self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.p1, self.p2)
        return bool(np.any(np.abs(np.asarray(coeffs, dtype=np.float64)) > 0.0))


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

        if trans.shape != (3,):
            trans = trans.reshape(-1)
        if trans.shape != (3,):
            raise ValueError(f"extrinsic.trans must be shape (3,), got {trans.shape}")

        return OBExtrinsic(rot=rot.copy(), trans=trans.copy())


def _intrinsic_signature(intri: OBCameraIntrinsic) -> Tuple[object, ...]:
    return (
        int(intri.width),
        int(intri.height),
        float(intri.fx),
        float(intri.fy),
        float(intri.cx),
        float(intri.cy),
    )


def _distortion_signature(dist: OBCameraDistortion) -> Tuple[object, ...]:
    return (
        dist.model.value,
        float(dist.k1),
        float(dist.k2),
        float(dist.k3),
        float(dist.k4),
        float(dist.k5),
        float(dist.k6),
        float(dist.p1),
        float(dist.p2),
    )


def _extrinsic_signature(extri: OBExtrinsic) -> Tuple[object, ...]:
    e = extri.normalized()
    return tuple(float(x) for x in e.rot.reshape(-1)) + tuple(float(x) for x in e.trans.reshape(-1))


def add_distortion_vectorized(disto: OBCameraDistortion, xy: np.ndarray) -> np.ndarray:
    """
    Apply distortion to normalized camera coordinates.

    xy shape: (..., 2), values are normalized pinhole coordinates.
    Returns same shape as xy.

    Brown-Conrady uses OpenCV-style coefficients:
        x_d = x * radial + tangential_x
        y_d = y * radial + tangential_y

    Brown-Conrady K6 uses the rational radial model:
        radial = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)
    """
    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape[-1] != 2:
        raise ValueError("xy must have last dimension 2")

    model = disto.model
    if model == DistortionModel.OB_DISTORTION_NONE or not disto.has_nonzero_coefficients():
        return xy.copy()

    x = xy[..., 0]
    y = xy[..., 1]

    if model in (DistortionModel.OB_DISTORTION_BROWN_CONRADY, DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        if model == DistortionModel.OB_DISTORTION_BROWN_CONRADY:
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

    if model == DistortionModel.OB_DISTORTION_KANNALA_BRANDT4:
        # Fisheye model commonly represented as theta_d = theta * (1 + k1*t^2 + ... + k4*t^8).
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        theta_d = theta * (1.0 + disto.k1 * theta2 + disto.k2 * theta4 + disto.k3 * theta6 + disto.k4 * theta8)
        scale = np.divide(theta_d, r, out=np.ones_like(r, dtype=np.float32), where=r > EPSILON)
        xd = x * scale
        yd = y * scale
        return np.stack([xd, yd], axis=-1).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported distortion model: {model}")


def remove_distortion_vectorized(
    disto: OBCameraDistortion,
    xy_distorted: np.ndarray,
    iterations: int = 12,
) -> np.ndarray:
    """
    Iteratively invert distortion for normalized coordinates.

    This is deliberately conservative and robust. It is slower than a closed-form
    approximation but avoids most corner-case drift.
    """
    xy_distorted = np.asarray(xy_distorted, dtype=np.float32)
    if disto.model == DistortionModel.OB_DISTORTION_NONE or not disto.has_nonzero_coefficients():
        return xy_distorted.copy()

    und = xy_distorted.copy()
    target = xy_distorted

    for _ in range(iterations):
        redistorted = add_distortion_vectorized(disto, und)
        error = redistorted - target
        und = und - error
        # Stop runaway updates when K6 denominator or fisheye model produces invalid values.
        bad = ~np.isfinite(und[..., 0]) | ~np.isfinite(und[..., 1])
        if np.any(bad):
            und[bad] = target[bad]
            break

    return und.astype(np.float32, copy=False)


def _is_integer_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.integer)


def _clip_round_to_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if _is_integer_dtype(dtype):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


def _remap_numpy(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Pure NumPy bilinear remap fallback."""
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = map_x.shape
    src_dtype = image.dtype

    if image.ndim == 2:
        out = np.zeros((dst_h, dst_w), dtype=np.float32)
    elif image.ndim == 3:
        out = np.zeros((dst_h, dst_w, image.shape[2]), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim}")

    inside = valid & (map_x >= 0.0) & (map_x <= src_w - 1.0) & (map_y >= 0.0) & (map_y <= src_h - 1.0)
    if not np.any(inside):
        return out.astype(src_dtype)

    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)

    xv = map_x[inside]
    yv = map_y[inside]
    x0v = x0[inside]
    x1v = x1[inside]
    y0v = y0[inside]
    y1v = y1[inside]

    wx = xv - x0v.astype(np.float32)
    wy = yv - y0v.astype(np.float32)

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    if image.ndim == 2:
        values = (
            image[y0v, x0v].astype(np.float32) * w00
            + image[y0v, x1v].astype(np.float32) * w10
            + image[y1v, x0v].astype(np.float32) * w01
            + image[y1v, x1v].astype(np.float32) * w11
        )
        out[inside] = values
    else:
        values = (
            image[y0v, x0v].astype(np.float32) * w00[:, None]
            + image[y0v, x1v].astype(np.float32) * w10[:, None]
            + image[y1v, x0v].astype(np.float32) * w01[:, None]
            + image[y1v, x1v].astype(np.float32) * w11[:, None]
        )
        out[inside] = values

    return _clip_round_to_dtype(out, src_dtype)


def _remap_torch(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """Torch grid_sample remap. Useful when map/image are large and a GPU is available."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")

    src_h, src_w = image.shape[:2]
    dst_h, dst_w = map_x.shape
    src_dtype = image.dtype

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img_f = image.astype(np.float32, copy=False)
    if image.ndim == 2:
        tensor = torch.from_numpy(img_f[None, None, :, :]).to(device)
    elif image.ndim == 3:
        tensor = torch.from_numpy(np.transpose(img_f, (2, 0, 1))[None, :, :, :]).to(device)
    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim}")

    mx = map_x.astype(np.float32, copy=True)
    my = map_y.astype(np.float32, copy=True)
    mx[~valid] = -2.0
    my[~valid] = -2.0

    # align_corners=True maps -1->0 and +1->W-1/H-1 exactly.
    gx = 2.0 * mx / max(src_w - 1, 1) - 1.0
    gy = 2.0 * my / max(src_h - 1, 1) - 1.0
    grid_np = np.stack([gx, gy], axis=-1)[None, :, :, :]
    grid = torch.from_numpy(grid_np).to(device)

    sampled = F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    out = sampled.detach().cpu().numpy()[0]
    if image.ndim == 2:
        out_np = out[0]
    else:
        out_np = np.transpose(out, (1, 2, 0))
    return _clip_round_to_dtype(out_np, src_dtype)



# ----------------------------- Hole filling -----------------------------


def fill_small_depth_holes(
    depth: np.ndarray,
    invalid_value: int = 0,
    max_hole_area: int = 8,
    radius: int = 2,
    max_depth_delta: int = 80,
    iterations: int = 1,
) -> np.ndarray:
    """
    Conservatively fill tiny isolated holes in a uint16 depth image.

    This is meant for residual one/few-pixel forward-warp holes after D2C. It should
    not be used as a general depth completion algorithm. Large zero regions are kept
    untouched because they usually correspond to true disocclusions, invalid source
    depth, or out-of-FOV areas.

    Args:
        depth: uint16 depth image. Zero is treated as invalid by default.
        invalid_value: invalid depth marker, usually 0.
        max_hole_area: only connected invalid components with area <= this are filled.
            Use 4..16 for sparse speckles. Larger values can smear depth boundaries.
        radius: local neighborhood radius used to collect candidate valid depths.
        max_depth_delta: candidate valid depths must be mutually consistent within this
            range. This prevents filling across strong foreground/background edges.
        iterations: repeat count. Usually 1 is enough; 2 can close slightly larger cracks.

    Returns:
        A new uint16 depth image with small holes filled.
    """
    if not isinstance(depth, np.ndarray) or depth.dtype != np.uint16:
        raise TypeError("depth must be a uint16 NumPy array")
    if depth.ndim != 2:
        raise ValueError("depth must be a single-channel 2D image")
    if max_hole_area <= 0 or radius <= 0 or iterations <= 0:
        return depth.copy()

    filled = depth.copy()
    h, w = filled.shape

    for _ in range(int(iterations)):
        invalid = filled == np.uint16(invalid_value)
        if not np.any(invalid):
            break

        # Prefer OpenCV connected components when available so we only touch tiny holes.
        if _HAS_CV2:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                invalid.astype(np.uint8), connectivity=8
            )

            changed = False
            for label in range(1, num_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area <= 0 or area > int(max_hole_area):
                    continue

                left = int(stats[label, cv2.CC_STAT_LEFT])
                top = int(stats[label, cv2.CC_STAT_TOP])
                width = int(stats[label, cv2.CC_STAT_WIDTH])
                height = int(stats[label, cv2.CC_STAT_HEIGHT])

                x0 = max(0, left - radius)
                y0 = max(0, top - radius)
                x1 = min(w, left + width + radius)
                y1 = min(h, top + height + radius)

                patch = filled[y0:y1, x0:x1]
                patch_valid = patch[patch != invalid_value]
                if patch_valid.size < 3:
                    continue

                # Robust edge guard: only fill from a locally coherent surface.
                lo, hi = np.percentile(patch_valid.astype(np.float32), [20.0, 80.0])
                if (hi - lo) > float(max_depth_delta):
                    continue

                value = np.uint16(np.clip(np.rint(np.median(patch_valid)), 1, UINT16_MAX))
                ys, xs = np.where(labels == label)
                filled[ys, xs] = value
                changed = True

            if not changed:
                break

        else:
            # Dependency-free fallback: fill isolated invalid pixels whose local valid
            # neighborhood is coherent. This does not component-label larger holes, so
            # keep it intentionally stricter than the OpenCV path.
            ys, xs = np.where(invalid)
            candidates = []
            values = []
            max_neighbors = (2 * radius + 1) * (2 * radius + 1) - 1

            for y, x in zip(ys, xs):
                y0 = max(0, int(y) - radius)
                y1 = min(h, int(y) + radius + 1)
                x0 = max(0, int(x) - radius)
                x1 = min(w, int(x) + radius + 1)
                patch = filled[y0:y1, x0:x1]
                patch_valid = patch[patch != invalid_value]
                # Require most of the neighborhood to be valid to avoid filling large holes.
                if patch_valid.size < max(3, int(0.65 * max_neighbors)):
                    continue
                lo, hi = np.percentile(patch_valid.astype(np.float32), [20.0, 80.0])
                if (hi - lo) > float(max_depth_delta):
                    continue
                candidates.append((int(y), int(x)))
                values.append(np.uint16(np.clip(np.rint(np.median(patch_valid)), 1, UINT16_MAX)))

            if not candidates:
                break
            for (y, x), value in zip(candidates, values):
                filled[y, x] = value

    return filled

# ----------------------------- D2C rasterizers -----------------------------


def _point_in_triangle(px: float, py: float, ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-12:
        return False
    a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    c = 1.0 - a - b
    eps = -1e-5
    return a >= eps and b >= eps and c >= eps


def _rasterize_quads_python(
    quads_xy: np.ndarray,
    depth_u32: np.ndarray,
    valid: np.ndarray,
    zbuf: np.ndarray,
    max_footprint_px: int,
) -> None:
    h, w = zbuf.shape
    n = quads_xy.shape[0]

    for i in range(n):
        if not valid[i]:
            continue

        q = quads_xy[i]
        if not np.all(np.isfinite(q)):
            continue

        min_x_f = float(np.min(q[:, 0]))
        max_x_f = float(np.max(q[:, 0]))
        min_y_f = float(np.min(q[:, 1]))
        max_y_f = float(np.max(q[:, 1]))

        min_x = max(0, int(np.floor(min_x_f)))
        max_x = min(w - 1, int(np.ceil(max_x_f)))
        min_y = max(0, int(np.floor(min_y_f)))
        max_y = min(h - 1, int(np.ceil(max_y_f)))

        if min_x > max_x or min_y > max_y:
            continue

        if (max_x - min_x + 1) > max_footprint_px or (max_y - min_y + 1) > max_footprint_px:
            # Usually indicates invalid calibration/distortion or very extreme projection.
            continue

        z = depth_u32[i]

        ax, ay = float(q[0, 0]), float(q[0, 1])
        bx, by = float(q[1, 0]), float(q[1, 1])
        cx, cy = float(q[2, 0]), float(q[2, 1])
        dx, dy = float(q[3, 0]), float(q[3, 1])

        for yy in range(min_y, max_y + 1):
            for xx in range(min_x, max_x + 1):
                inside = _point_in_triangle(xx, yy, ax, ay, bx, by, cx, cy) or _point_in_triangle(
                    xx, yy, ax, ay, cx, cy, dx, dy
                )
                if inside and z < zbuf[yy, xx]:
                    zbuf[yy, xx] = z


if _HAS_NUMBA:

    @njit(cache=True)  # type: ignore[misc]
    def _point_in_triangle_numba(px, py, ax, ay, bx, by, cx, cy):  # pragma: no cover - compiled path
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-12:
            return False
        a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
        b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
        c = 1.0 - a - b
        eps = -1e-5
        return a >= eps and b >= eps and c >= eps

    @njit(cache=True)  # type: ignore[misc]
    def _rasterize_quads_numba(quads_xy, depth_u32, valid, zbuf, max_footprint_px):  # pragma: no cover - compiled path
        h = zbuf.shape[0]
        w = zbuf.shape[1]
        n = quads_xy.shape[0]

        for i in range(n):
            if not valid[i]:
                continue

            # Read points explicitly; numba support for np.all/isfinite/min across axes is less portable.
            ax = quads_xy[i, 0, 0]
            ay = quads_xy[i, 0, 1]
            bx = quads_xy[i, 1, 0]
            by = quads_xy[i, 1, 1]
            cx = quads_xy[i, 2, 0]
            cy = quads_xy[i, 2, 1]
            dx = quads_xy[i, 3, 0]
            dy = quads_xy[i, 3, 1]

            if not (
                np.isfinite(ax)
                and np.isfinite(ay)
                and np.isfinite(bx)
                and np.isfinite(by)
                and np.isfinite(cx)
                and np.isfinite(cy)
                and np.isfinite(dx)
                and np.isfinite(dy)
            ):
                continue

            min_x_f = min(min(ax, bx), min(cx, dx))
            max_x_f = max(max(ax, bx), max(cx, dx))
            min_y_f = min(min(ay, by), min(cy, dy))
            max_y_f = max(max(ay, by), max(cy, dy))

            min_x = int(np.floor(min_x_f))
            max_x = int(np.ceil(max_x_f))
            min_y = int(np.floor(min_y_f))
            max_y = int(np.ceil(max_y_f))

            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0
            if max_x > w - 1:
                max_x = w - 1
            if max_y > h - 1:
                max_y = h - 1

            if min_x > max_x or min_y > max_y:
                continue

            if (max_x - min_x + 1) > max_footprint_px or (max_y - min_y + 1) > max_footprint_px:
                continue

            z = depth_u32[i]
            for yy in range(min_y, max_y + 1):
                for xx in range(min_x, max_x + 1):
                    inside = _point_in_triangle_numba(xx, yy, ax, ay, bx, by, cx, cy) or _point_in_triangle_numba(
                        xx, yy, ax, ay, cx, cy, dx, dy
                    )
                    if inside and z < zbuf[yy, xx]:
                        zbuf[yy, xx] = z

else:
    _rasterize_quads_numba = None


class AlignImpl:
    """
    High-quality Orbbec-style D2C/C2D aligner.

    Public API intentionally remains close to the user's original port:
        initialize(...)
        D2C(depth_buffer, out_depth=None, map_out=None, method="quad")
        C2D(depth_buffer, rgb_buffer, out_rgb, backend="auto")

    Recommended first test for black curved invalid bands:
        initialize(..., add_target_distortion=False)
    """

    def __init__(self) -> None:
        self.initialized = False
        self.depth_unit_mm = 1.0

        self.depth_intric: Optional[OBCameraIntrinsic] = None
        self.depth_disto: Optional[OBCameraDistortion] = None
        self.rgb_intric: Optional[OBCameraIntrinsic] = None
        self.rgb_disto: Optional[OBCameraDistortion] = None
        self.transform: Optional[OBExtrinsic] = None

        self.add_target_distortion = True
        self.need_to_undistort_depth = False
        self.scaled_trans = np.zeros(3, dtype=np.float32)

        # Optional hard limit for K6/rational model. Default None avoids artificial curved invalid boundaries.
        self.k6_r2_limit: Optional[float] = None

        # Projection coefficient cache keyed by depth-pixel offset (du, dv).
        self._coeff_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._signature: Optional[Tuple[object, ...]] = None

    def initialize(
        self,
        depth_intri: OBCameraIntrinsic,
        depth_dist: OBCameraDistortion,
        color_intri: OBCameraIntrinsic,
        color_dist: OBCameraDistortion,
        depth2color_extri: OBExtrinsic,
        depth_unit_mm: float = 1.0,
        add_target_distortion: bool = True,
        k6_r2_limit: Optional[float] = None,
    ) -> "AlignImpl":
        """
        Initialize the alignment processor.

        Args:
            depth_intri: Depth camera intrinsics.
            depth_dist: Depth camera distortion.
            color_intri: Color camera intrinsics.
            color_dist: Color camera distortion.
            depth2color_extri: Depth-to-color extrinsic: P_color = R * P_depth + t.
            depth_unit_mm: Millimeters per depth unit in depth_buffer.
            add_target_distortion: Apply color distortion before color pixel projection. Use False
                when the RGB image is already rectified/undistorted.
            k6_r2_limit: Optional safety radius squared for rational K6 distortion. Keeping
                this None avoids curved invalidation bands unless your calibration specifically
                requires a cutoff.
        """
        if not isinstance(depth_intri, OBCameraIntrinsic) or not isinstance(color_intri, OBCameraIntrinsic):
            raise TypeError("depth_intri and color_intri must be OBCameraIntrinsic")
        if not isinstance(depth_dist, OBCameraDistortion) or not isinstance(color_dist, OBCameraDistortion):
            raise TypeError("depth_dist and color_dist must be OBCameraDistortion")
        if not isinstance(depth2color_extri, OBExtrinsic):
            raise TypeError("depth2color_extri must be OBExtrinsic")
        if depth_unit_mm <= 0:
            raise ValueError("depth_unit_mm must be positive")

        depth_intri.validate("depth_intri")
        color_intri.validate("color_intri")
        extri = depth2color_extri.normalized()

        signature: Tuple[object, ...] = (
            _intrinsic_signature(depth_intri),
            _distortion_signature(depth_dist),
            _intrinsic_signature(color_intri),
            _distortion_signature(color_dist),
            _extrinsic_signature(extri),
            float(depth_unit_mm),
            bool(add_target_distortion),
            None if k6_r2_limit is None else float(k6_r2_limit),
        )

        if self.initialized and self._signature == signature:
            return self

        self.depth_intric = depth_intri
        self.depth_disto = depth_dist
        self.rgb_intric = color_intri
        self.rgb_disto = color_dist
        self.transform = extri
        self.depth_unit_mm = float(depth_unit_mm)
        self.scaled_trans = extri.trans.astype(np.float32) / np.float32(self.depth_unit_mm)
        self.add_target_distortion = bool(add_target_distortion)
        self.k6_r2_limit = k6_r2_limit

        self.need_to_undistort_depth = (
            depth_dist.model != DistortionModel.OB_DISTORTION_NONE and depth_dist.has_nonzero_coefficients()
        )

        self._coeff_cache.clear()

        self._signature = signature
        self.initialized = True

        # Build center coefficients immediately; corners are lazily built when quad D2C is requested.
        self._coefficients_for_offset(0.0, 0.0)
        return self

    def reset(self) -> None:
        self.initialized = False
        self.depth_intric = None
        self.depth_disto = None
        self.rgb_intric = None
        self.rgb_disto = None
        self.transform = None
        self.scaled_trans = np.zeros(3, dtype=np.float32)
        self.k6_r2_limit = None
        self._coeff_cache.clear()
        self._signature = None

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("AlignImpl is not initialized")
        assert self.depth_intric is not None
        assert self.depth_disto is not None
        assert self.rgb_intric is not None
        assert self.rgb_disto is not None
        assert self.transform is not None

    def _coefficients_for_offset(self, du: float, dv: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_initialized()
        assert self.depth_intric is not None
        assert self.depth_disto is not None
        assert self.transform is not None

        key = (float(du), float(dv))
        if key in self._coeff_cache:
            return self._coeff_cache[key]

        h = self.depth_intric.height
        w = self.depth_intric.width

        u, v = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
            indexing="xy",
        )

        x = ((u + np.float32(du)) - np.float32(self.depth_intric.cx)) / np.float32(self.depth_intric.fx)
        y = ((v + np.float32(dv)) - np.float32(self.depth_intric.cy)) / np.float32(self.depth_intric.fy)

        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        if self.need_to_undistort_depth:
            xy = np.stack([x_flat, y_flat], axis=-1)
            xy_ud = remove_distortion_vectorized(self.depth_disto, xy)
            x_flat = xy_ud[:, 0]
            y_flat = xy_ud[:, 1]

        R = self.transform.rot.astype(np.float32, copy=False)
        coeff_x = R[0, 0] * x_flat + R[0, 1] * y_flat + R[0, 2]
        coeff_y = R[1, 0] * x_flat + R[1, 1] * y_flat + R[1, 2]
        coeff_z = R[2, 0] * x_flat + R[2, 1] * y_flat + R[2, 2]

        result = (
            coeff_x.astype(np.float32, copy=False),
            coeff_y.astype(np.float32, copy=False),
            coeff_z.astype(np.float32, copy=False),
        )
        self._coeff_cache[key] = result
        return result

    def _apply_target_distortion(
        self,
        tx: np.ndarray,
        ty: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_initialized()
        assert self.rgb_disto is not None

        valid = np.isfinite(tx) & np.isfinite(ty)

        if self.rgb_disto.model == DistortionModel.OB_DISTORTION_NONE or not self.rgb_disto.has_nonzero_coefficients():
            return tx.astype(np.float32, copy=False), ty.astype(np.float32, copy=False), valid

        if self.rgb_disto.model == DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6 and self.k6_r2_limit is not None:
            r2 = tx * tx + ty * ty
            valid &= r2 < float(self.k6_r2_limit)

        out_x = np.full_like(tx, np.nan, dtype=np.float32)
        out_y = np.full_like(ty, np.nan, dtype=np.float32)

        if np.any(valid):
            xy = np.stack([tx[valid], ty[valid]], axis=-1)
            xy_d = add_distortion_vectorized(self.rgb_disto, xy)
            finite = np.isfinite(xy_d[:, 0]) & np.isfinite(xy_d[:, 1])

            valid_indices = np.flatnonzero(valid)
            finite_indices = valid_indices[finite]
            out_x[finite_indices] = xy_d[finite, 0]
            out_y[finite_indices] = xy_d[finite, 1]

            valid[:] = False
            valid[finite_indices] = True

        return out_x, out_y, valid

    def project_depth_to_color_float(
        self,
        depth_buffer: np.ndarray,
        du: float = 0.0,
        dv: float = 0.0,
        add_target_distortion: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project depth pixels into color image coordinates.

        Args:
            depth_buffer: uint16 depth frame, shape (depth_h, depth_w).
            du, dv: depth-pixel offsets. Use (0,0) for pixel center; +/-0.5 for corners.
            add_target_distortion: override for this call. None uses the initialized setting.

        Returns:
            px: float32 color x coordinate per depth pixel, shape (H,W).
            py: float32 color y coordinate per depth pixel, shape (H,W).
            zc: float32 depth in color-camera frame, in depth units, shape (H,W).
            valid: bool mask, shape (H,W).
        """
        self._require_initialized()
        assert self.depth_intric is not None
        assert self.rgb_intric is not None

        if not isinstance(depth_buffer, np.ndarray) or depth_buffer.dtype != np.uint16:
            raise TypeError("depth_buffer must be a NumPy uint16 array")
        if depth_buffer.shape != (self.depth_intric.height, self.depth_intric.width):
            raise ValueError(
                f"depth_buffer shape {depth_buffer.shape} does not match initialized depth "
                f"shape {(self.depth_intric.height, self.depth_intric.width)}"
            )

        if add_target_distortion is None:
            add_target_distortion = self.add_target_distortion

        coeff_x, coeff_y, coeff_z = self._coefficients_for_offset(float(du), float(dv))
        depth_flat = depth_buffer.reshape(-1).astype(np.float32)

        xc = depth_flat * coeff_x + self.scaled_trans[0]
        yc = depth_flat * coeff_y + self.scaled_trans[1]
        zc = depth_flat * coeff_z + self.scaled_trans[2]

        valid = (depth_flat > EPSILON) & (zc > EPSILON)

        px = np.full(depth_flat.shape, np.nan, dtype=np.float32)
        py = np.full(depth_flat.shape, np.nan, dtype=np.float32)

        valid_idx = np.flatnonzero(valid)
        if valid_idx.size:
            tx = xc[valid_idx] / zc[valid_idx]
            ty = yc[valid_idx] / zc[valid_idx]

            if add_target_distortion:
                tx, ty, distortion_valid = self._apply_target_distortion(tx, ty)
                valid_idx = valid_idx[distortion_valid]
                tx = tx[distortion_valid]
                ty = ty[distortion_valid]

            px[valid_idx] = tx * np.float32(self.rgb_intric.fx) + np.float32(self.rgb_intric.cx)
            py[valid_idx] = ty * np.float32(self.rgb_intric.fy) + np.float32(self.rgb_intric.cy)

            # Reset final validity to exactly the points that survived projection/distortion.
            valid[:] = False
            valid[valid_idx] = True

        valid &= np.isfinite(px) & np.isfinite(py) & np.isfinite(zc)

        h = self.depth_intric.height
        w = self.depth_intric.width
        return (
            px.reshape(h, w),
            py.reshape(h, w),
            zc.reshape(h, w).astype(np.float32, copy=False),
            valid.reshape(h, w),
        )

    def make_depth_to_color_map(
        self,
        depth_buffer: np.ndarray,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a depth-indexed map into the color frame.

        map_xy[y_depth, x_depth] = (x_color_float, y_color_float).
        Invalid locations are NaN for floating dtype and -1 for integer dtype.
        """
        px, py, _, valid = self.project_depth_to_color_float(depth_buffer, du=0.0, dv=0.0)
        h, w = depth_buffer.shape
        map_xy = np.empty((h, w, 2), dtype=dtype)

        if np.issubdtype(np.dtype(dtype), np.floating):
            map_xy.fill(np.nan)
            map_xy[..., 0][valid] = px[valid].astype(dtype)
            map_xy[..., 1][valid] = py[valid].astype(dtype)
        else:
            map_xy.fill(-1)
            u = np.floor(px + 0.5).astype(np.int32)
            v = np.floor(py + 0.5).astype(np.int32)
            assert self.rgb_intric is not None
            inside = valid & (u >= 0) & (u < self.rgb_intric.width) & (v >= 0) & (v < self.rgb_intric.height)
            map_xy[..., 0][inside] = u[inside].astype(dtype)
            map_xy[..., 1][inside] = v[inside].astype(dtype)

        return map_xy, valid

    def C2D(
        self,
        depth_buffer: np.ndarray,
        rgb_buffer: np.ndarray,
        out_rgb: np.ndarray,
        backend: Literal["auto", "opencv", "torch", "numpy"] = "auto",
        torch_device: Optional[str] = None,
    ) -> int:
        """
        Color-to-depth alignment: sample the RGB/color image at each projected depth pixel.

        This is a backward sampling problem and therefore should use a float map and bilinear
        interpolation. This implementation does that.

        Args:
            depth_buffer: uint16 depth image, shape (depth_h, depth_w).
            rgb_buffer: color image, shape (color_h, color_w) or (color_h, color_w, C).
            out_rgb: preallocated output image, shape (depth_h, depth_w) or (depth_h, depth_w, C).
            backend: "auto" chooses OpenCV if available, otherwise NumPy. "torch" can be faster
                on GPU for large frames.
            torch_device: e.g. "cuda", "cpu". Only used with backend="torch".
        """
        self._require_initialized()
        assert self.depth_intric is not None
        assert self.rgb_intric is not None

        if not isinstance(depth_buffer, np.ndarray) or depth_buffer.dtype != np.uint16:
            print("Error: depth_buffer must be a NumPy array of type uint16.")
            return -1
        if depth_buffer.shape != (self.depth_intric.height, self.depth_intric.width):
            print("Error: depth_buffer shape does not match initialized depth intrinsics.")
            return -1
        if not isinstance(rgb_buffer, np.ndarray) or not isinstance(out_rgb, np.ndarray):
            print("Error: rgb_buffer and out_rgb must be NumPy arrays.")
            return -1
        if rgb_buffer.shape[:2] != (self.rgb_intric.height, self.rgb_intric.width):
            print("Error: rgb_buffer shape does not match initialized color intrinsics.")
            return -1
        if out_rgb.shape[:2] != depth_buffer.shape:
            print("Error: out_rgb spatial shape must match depth_buffer.")
            return -1
        if rgb_buffer.ndim != out_rgb.ndim:
            print("Error: rgb_buffer and out_rgb must have matching dimensionality.")
            return -1
        if rgb_buffer.ndim == 3 and rgb_buffer.shape[2] != out_rgb.shape[2]:
            print("Error: rgb_buffer and out_rgb channel counts must match.")
            return -1
        if rgb_buffer.dtype != out_rgb.dtype:
            print("Error: rgb_buffer and out_rgb dtypes must match.")
            return -1

        px, py, _, valid = self.project_depth_to_color_float(depth_buffer, du=0.0, dv=0.0)

        map_x = px.astype(np.float32, copy=True)
        map_y = py.astype(np.float32, copy=True)

        # Border checks are stricter for the NumPy fallback because it needs x+1/y+1.
        inside = valid & (map_x >= 0.0) & (map_x <= self.rgb_intric.width - 1.0) & (map_y >= 0.0) & (
            map_y <= self.rgb_intric.height - 1.0
        )

        selected_backend = backend
        if selected_backend == "auto":
            selected_backend = "opencv" if _HAS_CV2 else "numpy"

        try:
            if selected_backend == "opencv":
                if not _HAS_CV2:
                    raise RuntimeError("OpenCV is not available")
                map_x[~inside] = -1.0
                map_y[~inside] = -1.0
                warped = cv2.remap(
                    rgb_buffer,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                np.copyto(out_rgb, warped)
                return 0

            if selected_backend == "torch":
                warped = _remap_torch(rgb_buffer, map_x, map_y, inside, device=torch_device)
                np.copyto(out_rgb, warped)
                return 0

            if selected_backend == "numpy":
                warped = _remap_numpy(rgb_buffer, map_x, map_y, inside)
                np.copyto(out_rgb, warped)
                return 0

            print(f"Error: unsupported C2D backend: {backend}")
            return -1

        except Exception as exc:
            print(f"Error during C2D remap with backend={selected_backend}: {exc}")
            return -1

    def _d2c_splat_2x2(
        self,
        px: np.ndarray,
        py: np.ndarray,
        zc: np.ndarray,
        valid: np.ndarray,
        zbuf: np.ndarray,
        max_depth_value: int,
    ) -> None:
        """Fast z-buffered 2x2 splat using floor(px), not biased round+right+down."""
        assert self.rgb_intric is not None
        rgb_h, rgb_w = self.rgb_intric.height, self.rgb_intric.width

        x = px[valid]
        y = py[valid]
        z = zc[valid]

        if x.size == 0:
            return

        z_u32 = np.clip(np.rint(z), 1, max_depth_value).astype(np.uint32)
        u0 = np.floor(x).astype(np.int32)
        v0 = np.floor(y).astype(np.int32)
        flat = zbuf.reshape(-1)

        for oy in (0, 1):
            for ox in (0, 1):
                u = u0 + ox
                v = v0 + oy
                inside = (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h)
                if np.any(inside):
                    idx = v[inside] * rgb_w + u[inside]
                    np.minimum.at(flat, idx, z_u32[inside])

    def _d2c_quad(
        self,
        depth_buffer: np.ndarray,
        zbuf: np.ndarray,
        max_depth_value: int,
        max_footprint_px: int,
    ) -> None:
        """
        High-quality D2C: project the four corners of each depth pixel and rasterize its
        footprint as a quadrilateral with a z-buffer.
        """
        px_center, _, zc_center, valid_center = self.project_depth_to_color_float(depth_buffer, 0.0, 0.0)
        _ = px_center  # center x is not needed except for final validity through zc_center.

        corners = [
            self.project_depth_to_color_float(depth_buffer, -0.5, -0.5),
            self.project_depth_to_color_float(depth_buffer, +0.5, -0.5),
            self.project_depth_to_color_float(depth_buffer, +0.5, +0.5),
            self.project_depth_to_color_float(depth_buffer, -0.5, +0.5),
        ]

        valid = valid_center.copy()
        quad_xy = np.empty((depth_buffer.size, 4, 2), dtype=np.float32)

        for i, (px, py, _, valid_corner) in enumerate(corners):
            valid &= valid_corner
            quad_xy[:, i, 0] = px.reshape(-1)
            quad_xy[:, i, 1] = py.reshape(-1)

        valid_flat = valid.reshape(-1)
        z_u32 = np.clip(np.rint(zc_center.reshape(-1)), 1, max_depth_value).astype(np.uint32)

        if _HAS_NUMBA and _rasterize_quads_numba is not None:
            _rasterize_quads_numba(quad_xy, z_u32, valid_flat, zbuf, int(max_footprint_px))
        else:
            _rasterize_quads_python(quad_xy, z_u32, valid_flat, zbuf, int(max_footprint_px))

    def D2C(
        self,
        depth_buffer: np.ndarray,
        out_depth: Optional[np.ndarray] = None,
        map_out: Optional[np.ndarray] = None,
        method: Literal["quad", "splat"] = "quad",
        max_footprint_px: int = 16,
        fill_holes: bool = False,
        hole_max_area: int = 8,
        hole_radius: int = 2,
        hole_max_depth_delta: int = 80,
        hole_fill_iterations: int = 1,
    ) -> Union[int, np.ndarray]:
        """
        Depth-to-color alignment.

        Args:
            depth_buffer: uint16 depth image, shape (depth_h, depth_w).
            out_depth: optional preallocated uint16 output, shape (color_h, color_w).
            map_out: optional map from depth pixels to color coordinates, shape (depth_h, depth_w, 2).
                Use float32 for high-quality subpixel maps. int32 maps are supported for legacy code
                and receive rounded coordinates.
            method:
                "quad"  - high-quality pixel-footprint rasterization + z-buffer.
                "splat" - faster 2x2 center splat + z-buffer.
            max_footprint_px: skip projected quads whose bounding box exceeds this size in either
                dimension. This avoids catastrophic smears from bad calibration or invalid distortion.
            fill_holes: if True, run a conservative post-pass that fills only tiny isolated zero
                islands. This is useful for sparse black speckles after forward warping.
            hole_max_area: maximum connected zero-component area to fill. Start with 4..8.
            hole_radius: local neighborhood radius used to estimate replacement depth.
            hole_max_depth_delta: local valid depths must be this coherent, in depth units,
                otherwise the hole is left untouched to avoid foreground/background bleeding.
            hole_fill_iterations: repeat count. Usually 1; use 2 for tiny cracks.

        Returns:
            If out_depth is None, returns the new aligned depth image.
            If out_depth is provided, writes in-place and returns 0 on success or -1 on error.
        """
        self._require_initialized()
        assert self.depth_intric is not None
        assert self.rgb_intric is not None

        if not isinstance(depth_buffer, np.ndarray) or depth_buffer.dtype != np.uint16:
            print("Error: depth_buffer must be a NumPy array of type uint16.")
            return -1
        if depth_buffer.shape != (self.depth_intric.height, self.depth_intric.width):
            print("Error: depth_buffer shape does not match initialized depth intrinsics.")
            return -1

        created_output = out_depth is None
        if out_depth is None:
            out_depth = np.zeros((self.rgb_intric.height, self.rgb_intric.width), dtype=np.uint16)
        else:
            if not isinstance(out_depth, np.ndarray) or out_depth.dtype != np.uint16:
                print("Error: out_depth must be a NumPy array of type uint16.")
                return -1
            if out_depth.shape != (self.rgb_intric.height, self.rgb_intric.width):
                print("Error: out_depth shape does not match initialized color intrinsics.")
                return -1

        if map_out is not None:
            if not isinstance(map_out, np.ndarray):
                print("Error: map_out must be a NumPy array.")
                return -1
            if map_out.shape != (self.depth_intric.height, self.depth_intric.width, 2):
                print("Error: map_out shape must be (depth_h, depth_w, 2).")
                return -1
            if not (
                np.issubdtype(map_out.dtype, np.floating) or np.issubdtype(map_out.dtype, np.integer)
            ):
                print("Error: map_out dtype must be floating or integer.")
                return -1

        try:
            # Always produce center map when requested. This is the correct map for C2D/diagnostics.
            if map_out is not None:
                px, py, _, valid = self.project_depth_to_color_float(depth_buffer, 0.0, 0.0)
                if np.issubdtype(map_out.dtype, np.floating):
                    map_out.fill(np.nan)
                    map_out[..., 0][valid] = px[valid].astype(map_out.dtype)
                    map_out[..., 1][valid] = py[valid].astype(map_out.dtype)
                else:
                    map_out.fill(-1)
                    u = np.floor(px + 0.5).astype(np.int32)
                    v = np.floor(py + 0.5).astype(np.int32)
                    inside = valid & (u >= 0) & (u < self.rgb_intric.width) & (v >= 0) & (v < self.rgb_intric.height)
                    map_out[..., 0][inside] = u[inside].astype(map_out.dtype)
                    map_out[..., 1][inside] = v[inside].astype(map_out.dtype)

            sentinel = np.iinfo(np.uint32).max
            zbuf = np.full((self.rgb_intric.height, self.rgb_intric.width), sentinel, dtype=np.uint32)
            max_depth_value = int(UINT16_MAX)

            if method == "quad":
                self._d2c_quad(depth_buffer, zbuf, max_depth_value, max_footprint_px=max_footprint_px)
            elif method == "splat":
                px, py, zc, valid = self.project_depth_to_color_float(depth_buffer, 0.0, 0.0)
                self._d2c_splat_2x2(px, py, zc, valid, zbuf, max_depth_value=max_depth_value)
            else:
                print(f"Error: unsupported D2C method: {method}")
                return -1

            out_depth.fill(0)
            mask = zbuf != sentinel
            out_depth[mask] = zbuf[mask].astype(np.uint16)

            if fill_holes:
                filled = fill_small_depth_holes(
                    out_depth,
                    invalid_value=0,
                    max_hole_area=hole_max_area,
                    radius=hole_radius,
                    max_depth_delta=hole_max_depth_delta,
                    iterations=hole_fill_iterations,
                )
                np.copyto(out_depth, filled)

            if created_output:
                return out_depth
            return 0

        except Exception as exc:
            print(f"Error during D2C: {exc}")
            return -1


__all__ = [
    "AlignImpl",
    "DistortionModel",
    "OBCameraIntrinsic",
    "OBCameraDistortion",
    "OBExtrinsic",
    "add_distortion_vectorized",
    "remove_distortion_vectorized",
    "fill_small_depth_holes",
]