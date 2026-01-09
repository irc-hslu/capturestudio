from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Literal

import numpy as np
from tqdm import tqdm

from utils.misc import PathUtils

# Constants
EPSILON = 1e-6  # Use a smaller epsilon consistent with C++


class DistortionModel(Enum):
    OB_DISTORTION_NONE = 0  # Added for completeness
    OB_DISTORTION_BROWN_CONRADY = 1
    OB_DISTORTION_BROWN_CONRADY_K6 = 2
    OB_DISTORTION_KANNALA_BRANDT4 = 3


@dataclass(kw_only=True)
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

    def to_orbbec(self):
        """Convert to Orbbec SDK intrinsic format."""
        from pyorbbecsdk import OBCameraDistortion as OBCameraDistortionSDK
        out = OBCameraDistortionSDK()
        out.k1 = self.k1
        out.k2 = self.k2
        out.k3 = self.k3
        out.k4 = self.k4
        out.k5 = self.k5
        out.k6 = self.k6
        out.p1 = self.p1
        out.p2 = self.p2
        # out.model = self.model.value
        return out


@dataclass(kw_only=True)
class OBCameraIntrinsic:
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    def to_orbbec(self):
        """Convert to Orbbec SDK intrinsic format."""
        from pyorbbecsdk import OBCameraIntrinsic as OBCameraIntrinsicSDK
        out = OBCameraIntrinsicSDK()
        out.width = self.width
        out.height = self.height
        out.fx = self.fx
        out.fy = self.fy
        out.cx = self.cx
        out.cy = self.cy
        return out


@dataclass(kw_only=True)
class OBExtrinsic:
    # Default to identity rotation and zero translation
    rot: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    trans: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def to_orbbec(self):
        """Convert to Orbbec SDK extrinsic format."""
        from pyorbbecsdk import OBExtrinsic as OBExtrinsicSDK
        out = OBExtrinsicSDK()
        out.rot = self.rot.tolist()
        out.transform = self.trans.tolist()
        return out


# --- Distortion Functions (Vectorized NumPy versions) ---

def add_distortion_vectorized(distort_param: OBCameraDistortion, pt_ud: np.ndarray) -> np.ndarray:
    """Applies distortion to an array of undistorted points (N, 2)."""
    assert pt_ud.ndim == 2 and pt_ud.shape[1] == 2, "Input pt_ud must be shape (N, 2)"
    pt_d = np.zeros_like(pt_ud)
    k1, k2, k3 = distort_param.k1, distort_param.k2, distort_param.k3
    k4, k5, k6 = distort_param.k4, distort_param.k5, distort_param.k6
    p1, p2 = distort_param.p1, distort_param.p2

    x_ud = pt_ud[:, 0]
    y_ud = pt_ud[:, 1]
    r2 = x_ud ** 2 + y_ud ** 2

    if distort_param.model in [DistortionModel.OB_DISTORTION_BROWN_CONRADY, DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6]:
        r4 = r2 ** 2
        r6 = r4 * r2
        k_radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        if distort_param.model == DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6:
            # Add epsilon to prevent division by zero, although ideally k4,k5,k6 are such that this isn't needed
            denominator = 1 + k4 * r2 + k5 * r4 + k6 * r6
            k_radial /= (denominator + EPSILON * (denominator == 0))  # Avoid division by zero

        # Tangential distortion
        t_x = p2 * (r2 + 2 * x_ud ** 2) + 2 * p1 * x_ud * y_ud
        t_y = p1 * (r2 + 2 * y_ud ** 2) + 2 * p2 * x_ud * y_ud

        pt_d[:, 0] = x_ud * k_radial + t_x
        pt_d[:, 1] = y_ud * k_radial + t_y

    elif distort_param.model == DistortionModel.OB_DISTORTION_KANNALA_BRANDT4:
        r = np.sqrt(r2)
        # Handle r=0 case to avoid division by zero and NaN in atan
        r_is_zero = (r < EPSILON)
        r_safe = np.where(r_is_zero, 1.0, r)  # Use 1.0 when r is zero, scale will be theta=0

        theta = np.arctan(r_safe)
        theta2 = theta ** 2
        theta4 = theta2 * theta2
        theta6 = theta2 * theta4
        theta8 = theta4 * theta4

        k_radial = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

        # Scale: k_radial / r. If r is zero, theta is zero, so k_radial is zero. Scale should be 1?
        # In the C++ code, if r is zero, pt_d = pt_ud * 0 = 0.
        # If r is non-zero, scale is k_radial / r
        # Let's follow C++: if r is near zero, result is near zero.
        scale = np.where(r_is_zero, 0.0, k_radial / r_safe)

        pt_d[:, 0] = scale * x_ud
        pt_d[:, 1] = scale * y_ud
    elif distort_param.model == DistortionModel.OB_DISTORTION_NONE:
        pt_d = pt_ud  # No distortion
    else:
        raise ValueError(f"Unsupported distortion model: {distort_param.model}")

    return pt_d


def remove_distortion_vectorized(distort_param: OBCameraDistortion, pt_d: np.ndarray) -> np.ndarray:
    """Removes distortion from an array of distorted points (N, 2) using iterative method."""
    assert pt_d.ndim == 2 and pt_d.shape[1] == 2, "Input pt_d must be shape (N, 2)"

    if distort_param.model == DistortionModel.OB_DISTORTION_NONE:
        return pt_d  # No distortion to remove

    max_iteration = 20
    tolerance = 1e-6  # Match C++ epsilon

    # Initial guess: undistorted point is the distorted point
    pt_ud_estimate = pt_d.copy()

    for _ in range(max_iteration):
        pt_d_reprojected = add_distortion_vectorized(distort_param, pt_ud_estimate)
        # Error is the difference between target distorted point and reprojected estimate
        error = pt_d - pt_d_reprojected
        # Update the estimate by adding the error
        pt_ud_estimate += error

        # Check for convergence
        mean_abs_error = np.mean(np.abs(error))
        if mean_abs_error < tolerance:
            break
        # print(f"Iter {_}: Mean abs error = {mean_abs_error}") # DEBUG

    # One final refinement might be closer to the C++ version's specific update rule
    # Let's recalculate the reprojection based on the final estimate
    # pt_d_final_reprojected = add_distortion_vectorized(distort_param, pt_ud_estimate)
    # pt_ud_final = pt_ud_estimate + (pt_d - pt_d_final_reprojected) # Apply final correction
    # return pt_ud_final
    return pt_ud_estimate  # Return the last estimate


# --- Helper Functions ---

def polynomial(x: float, a: float, b: float, c: float, d: float) -> float:
    # Corrected polynomial according to C++ (k6=a, k5=b, k4=c, 1=d)
    return a * x ** 3 + b * x ** 2 + c * x + d


def binary_search(left: float, right: float, a: float, b: float, c: float, d: float, tolerance: float = 1e-4) -> float:
    # Needs polynomial(left,...) != 0
    poly_left = polynomial(left, a, b, c, d)
    if abs(poly_left) < tolerance: return left  # Avoid issues if root is at the start

    while (right - left) > tolerance:
        mid = (left + right) / 2.0
        f_mid = polynomial(mid, a, b, c, d)
        if abs(f_mid) < tolerance:
            return mid
        elif f_mid * poly_left < 0:  # Root is in [left, mid]
            right = mid
        else:  # Root is in [mid, right]
            left = mid
            poly_left = f_mid  # Update the boundary value
    return (left + right) / 2.0


def estimate_inflection_point(depth_intr: OBCameraIntrinsic, rgb_intr: OBCameraIntrinsic, disto: OBCameraDistortion) -> float:
    result = 0.0
    if disto.model == DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6 and (disto.k4 != 0 or disto.k5 != 0 or disto.k6 != 0):
        r2_vals = []
        for intr in [depth_intr, rgb_intr]:
            # Calculate max radial distance in normalized coords for each camera
            corners_norm = np.array([
                [0, 0], [intr.width, 0], [0, intr.height], [intr.width, intr.height]
            ], dtype=np.float32)
            corners_norm[:, 0] = (corners_norm[:, 0] - intr.cx) / intr.fx
            corners_norm[:, 1] = (corners_norm[:, 1] - intr.cy) / intr.fy
            r2_vals.extend(list(np.sum(corners_norm ** 2, axis=1)))

        r2_max = max(r2_vals) if r2_vals else 0.0
        # Search for root of denominator polynomial: 1 + k4*r2 + k5*r4 + k6*r6 = 0
        # Let x = r2. Search for root of k6*x^3 + k5*x^2 + k4*x + 1 = 0
        a, b, c, d = disto.k6, disto.k5, disto.k4, 1.0

        # Check if polynomial is always positive/negative in the range [0, r2_max]
        # Simple check: evaluate at endpoints and potentially midpoint
        if polynomial(0, a, b, c, d) > 0 and \
                polynomial(r2_max, a, b, c, d) > 0 and \
                polynomial(r2_max / 2, a, b, c, d) > 0:
            # Likely no root in range (this isn't foolproof but covers simple cases)
            return 0.0
        if polynomial(0, a, b, c, d) < 0 and \
                polynomial(r2_max, a, b, c, d) < 0 and \
                polynomial(r2_max / 2, a, b, c, d) < 0:
            # Likely no root in range
            return 0.0

        # More robust search needed. C++ version uses step search then binary search.
        # Let's refine the search range slightly. We only care about positive r2.
        search_min = 0.0
        search_max = r2_max
        step = max(search_max / 100.0, 1e-5)  # Avoid tiny steps if r2_max is small

        prev_x = search_min
        prev_f = polynomial(prev_x, a, b, c, d)

        current_x = prev_x + step
        found_root = False
        while current_x <= search_max:
            current_f = polynomial(current_x, a, b, c, d)
            if prev_f * current_f <= 0:  # Sign change indicates root between prev_x and current_x
                # Add small epsilon to prev_x if prev_f is zero to avoid issues in binary search
                search_left = prev_x + EPSILON if abs(prev_f) < EPSILON else prev_x
                result = binary_search(search_left, current_x, a, b, c, d)
                found_root = True
                break
            prev_x, prev_f = current_x, current_f
            current_x += step

        # If no root found by stepping, double-check the final interval if search_max > prev_x
        if not found_root and search_max > prev_x:
            current_f = polynomial(search_max, a, b, c, d)
            if prev_f * current_f <= 0:
                search_left = prev_x + EPSILON if abs(prev_f) < EPSILON else prev_x
                result = binary_search(search_left, search_max, a, b, c, d)

    # Return the r^2 value where the inflection/denominator issue occurs
    return result if result > EPSILON else 0.0


# noinspection PyPep8Naming,PyMethodMayBeStatic
class AlignImpl:
    def __init__(self):
        self.initialized = False
        self.depth_unit_mm = 1.0
        self.r2_max_loc = 0.0  # r^2 limit for K6 model
        self.auto_down_scale = 1.0  # Not directly used, scale derived if use_scale is True
        self.depth_intric: Optional[OBCameraIntrinsic] = None
        self.depth_disto: Optional[OBCameraDistortion] = None
        self.rgb_intric: Optional[OBCameraIntrinsic] = None
        self.rgb_disto: Optional[OBCameraDistortion] = None
        self.transform: Optional[OBExtrinsic] = None  # Stores depth-to-color extrinsic
        self.add_target_distortion = True
        self.gap_fill_copy = True  # If True, simple 4-pixel fill; if False, use corner coords (mimics C++ channel=2)
        self.use_scale = False
        self.need_to_undistort_depth = False
        self.scaled_trans = np.zeros(3, dtype=np.float32)

        # LUTs storing precomputed coefficients (numpy arrays)
        # Key: (depth_width, depth_height)
        # Value: List of numpy arrays [channel0, channel1] (channel1 only used if gap_fill_copy=False)
        self.rot_coeff_ht_x: Dict[Tuple[int, int], List[np.ndarray]] = {}
        self.rot_coeff_ht_y: Dict[Tuple[int, int], List[np.ndarray]] = {}
        self.rot_coeff_ht_z: Dict[Tuple[int, int], List[np.ndarray]] = {}

        # ROI limits (used for clipping color coordinates) - updated in set_limit_roi
        self.x_limit = np.array([0.0, 0.0], dtype=np.float32)
        self.y_limit = np.array([0.0, 0.0], dtype=np.float32)

    def initialize(self,
                   depth_intri: OBCameraIntrinsic,
                   depth_dist: OBCameraDistortion,
                   color_intri: OBCameraIntrinsic,
                   color_dist: OBCameraDistortion,
                   depth2color_extri: OBExtrinsic,
                   depth_unit_mm: float = 1.0,
                   add_target_distortion: bool = True,
                   gap_fill_copy: bool = True,
                   use_scale: bool = False) -> 'AlignImpl':
        """
        Initializes the alignment processor with camera parameters.

        Args:
            depth_intri: Intrinsics of the depth camera.
            depth_dist: Distortion parameters of the depth camera.
            color_intri: Intrinsics of the color camera.
            color_dist: Distortion parameters of the color camera.
            depth2color_extri: Extrinsic transformation from depth camera to color camera coordinates.
            depth_unit_mm: The value of one depth unit in millimeters (e.g., 1.0 if depth is in mm).
            add_target_distortion: Apply color camera distortion to the projected points.
            gap_fill_copy: Method for filling gaps. True uses 4-pixel replication, False uses corner calculation.
            use_scale: Adjust color intrinsics to match depth pixel pitch if scales differ significantly.
        """
        # Basic parameter validation
        if not all([isinstance(p, OBCameraIntrinsic) for p in [depth_intri, color_intri]]) or \
                not all([isinstance(p, OBCameraDistortion) for p in [depth_dist, color_dist]]) or \
                not isinstance(depth2color_extri, OBExtrinsic):
            raise TypeError("Invalid input parameter types")
        if depth_unit_mm <= 0:
            raise ValueError("depth_unit_mm must be positive")
        if not (0 < depth_intri.width * depth_intri.height <= 1e8 and 0 < color_intri.width * color_intri.height <= 1e8):  # Sanity check size
            raise ValueError("Invalid camera dimensions")
        if not (depth_intri.fx > 0 and depth_intri.fy > 0 and color_intri.fx > 0 and color_intri.fy > 0):
            raise ValueError("Focal lengths must be positive")

        # Check if re-initialization is needed (simple object comparison)
        # Note: Numpy array comparison needs np.array_equal
        needs_reinit = not self.initialized or \
                       self.depth_intric != depth_intri or \
                       self.depth_disto != depth_dist or \
                       self.rgb_intric != color_intri or \
                       self.rgb_disto != color_dist or \
                       not np.array_equal(self.transform.rot, depth2color_extri.rot) or \
                       not np.array_equal(self.transform.trans, depth2color_extri.trans) or \
                       self.depth_unit_mm != depth_unit_mm or \
                       self.add_target_distortion != add_target_distortion or \
                       self.gap_fill_copy != gap_fill_copy or \
                       self.use_scale != use_scale

        if needs_reinit:
            # print("Re-initializing AlignImpl...")
            self.depth_intric = depth_intri
            self.depth_disto = depth_dist
            self.rgb_intric = color_intri
            self.rgb_disto = color_dist
            self.transform = depth2color_extri  # OBExtrinsic contains numpy arrays already
            self.add_target_distortion = add_target_distortion
            self.gap_fill_copy = gap_fill_copy
            self.use_scale = use_scale  # Store the original request

            # Check if depth distortion needs removal during LUT calculation
            self.need_to_undistort_depth = (depth_dist.model != DistortionModel.OB_DISTORTION_NONE and
                                            (np.any([depth_dist.k1, depth_dist.k2, depth_dist.k3, depth_dist.k4, depth_dist.k5, depth_dist.k6]) or np.any([depth_dist.p1, depth_dist.p2])))

            self.depth_unit_mm = depth_unit_mm
            # Scale translation by depth unit
            self.scaled_trans = self.transform.trans / self.depth_unit_mm

            # Prepare LUTs for the *current* depth resolution
            # Note: prepare_depth_resolution handles self.use_scale internally if needed for LUT calc
            # but D2C handles the actual scaling of intrinsics during the process call
            self.prepare_depth_resolution()

            # Set ROI limits based on *current* (potentially scaled if we use_scale was true *during init*) rgb intrinsics
            self.set_limit_roi()
            self.initialized = True
            # print("Initialization complete.")

        return self

    def reset(self):
        """Resets the alignment processor and clears cached data."""
        self.clear_matrix_cache()
        self.initialized = False
        self.depth_intric = None
        self.depth_disto = None
        self.rgb_intric = None
        self.rgb_disto = None
        self.transform = None
        # print("AlignImpl reset.")

    def prepare_depth_resolution(self):
        """Precomputes rotation coefficient LUTs for the current depth resolution."""
        if not self.depth_intric or not self.transform:
            raise RuntimeError("Cannot prepare resolution, not initialized properly.")

        key = (self.depth_intric.width, self.depth_intric.height)
        if key in self.rot_coeff_ht_x:
            # print(f"LUTs for resolution {key} already exist.")
            return  # Already computed for this resolution

        # print(f"Preparing LUTs for depth resolution: {key}...")
        self.clear_matrix_cache()  # Clear cache before creating new LUTs

        # Estimate inflection point for K6 model safety check *before* scaling
        if self.add_target_distortion and self.rgb_disto.model == DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6:
            self.r2_max_loc = estimate_inflection_point(self.depth_intric, self.rgb_intric, self.rgb_disto)
            # print(f"Estimated K6 r^2 max location: {self.r2_max_loc}")
        else:
            self.r2_max_loc = 0.0  # No limit needed

        # Determine number of channels based on gap fill strategy
        num_channels = 1 if self.gap_fill_copy else 2
        height, width = self.depth_intric.height, self.depth_intric.width
        # coeff_num = height * width

        # Prepare output lists for LUTs
        rot_coeff_list_x = []
        rot_coeff_list_y = []
        rot_coeff_list_z = []

        # Create coordinate grid for depth pixels
        v_coords, u_coords = np.meshgrid(np.arange(height, dtype=np.float32),
                                         np.arange(width, dtype=np.float32),
                                         indexing='ij')  # Matrix indexing ij -> v, u

        for i in range(num_channels):
            multiplier = 0.0 if self.gap_fill_copy else (0.5 if i == 1 else -0.5)  # Offset for corner sampling

            # Calculate normalized coordinates (x, y) for each depth pixel (potentially offset)
            # Apply offsets for corner sampling if not gap_fill_copy
            u_offset = u_coords + multiplier
            v_offset = v_coords + multiplier

            x = (u_offset - self.depth_intric.cx) / self.depth_intric.fx
            y = (v_offset - self.depth_intric.cy) / self.depth_intric.fy

            # Flatten for processing
            x_flat = x.flatten()
            y_flat = y.flatten()
            pt_norm_flat = np.stack([x_flat, y_flat], axis=-1)  # Shape (N, 2)

            # Undistort points if necessary
            if self.need_to_undistort_depth:
                # print(f"Applying depth undistortion for channel {i}...")
                # This is iterative, so applying it vectorized is complex but possible
                pt_undistorted_flat = remove_distortion_vectorized(self.depth_disto, pt_norm_flat)
                x_ud_flat = pt_undistorted_flat[:, 0]
                y_ud_flat = pt_undistorted_flat[:, 1]
            else:
                # print(f"Skipping depth undistortion for channel {i}.")
                x_ud_flat = x_flat
                y_ud_flat = y_flat

            # Apply rotation matrix (Extrinsic R component)
            # Result = R @ [x_ud, y_ud, 1]^T
            # We need coefficients for: depth * (coeff) + trans
            # Point in depth cam = [x_ud*depth, y_ud*depth, depth] (approx, using pinhole)
            # Point in color cam = R @ [x_ud*depth, y_ud*depth, depth]^T + T
            # Point in color cam = depth * (R @ [x_ud, y_ud, 1]^T) + T
            # So the coefficients are the columns of R multiplied by [x_ud, y_ud, 1]

            # Coeffs for X_color = depth * coeff_x + trans_x
            # Coeffs for Y_color = depth * coeff_y + trans_y
            # Coeffs for Z_color = depth * coeff_z + trans_z

            # R is 3x3, shape (3,3)
            # Input vectors are effectively [x_ud, y_ud, 1], shape (N, 3)
            # We want R[0,:] dot [x_ud, y_ud, 1], R[1,:] dot [x_ud, y_ud, 1], R[2,:] dot [x_ud, y_ud, 1]

            R = self.transform.rot  # Shape (3, 3)

            rot_coeff_x = R[0, 0] * x_ud_flat + R[0, 1] * y_ud_flat + R[0, 2]  # * 1
            rot_coeff_y = R[1, 0] * x_ud_flat + R[1, 1] * y_ud_flat + R[1, 2]  # * 1
            rot_coeff_z = R[2, 0] * x_ud_flat + R[2, 1] * y_ud_flat + R[2, 2]  # * 1

            rot_coeff_list_x.append(rot_coeff_x.astype(np.float32))
            rot_coeff_list_y.append(rot_coeff_y.astype(np.float32))
            rot_coeff_list_z.append(rot_coeff_z.astype(np.float32))

        # Store the computed LUTs
        self.rot_coeff_ht_x[key] = rot_coeff_list_x
        self.rot_coeff_ht_y[key] = rot_coeff_list_y
        self.rot_coeff_ht_z[key] = rot_coeff_list_z
        # print(f"Finished preparing LUTs for resolution {key}.")

    def clear_matrix_cache(self):
        """Clears the cached rotation coefficient LUTs."""
        self.rot_coeff_ht_x.clear()
        self.rot_coeff_ht_y.clear()
        self.rot_coeff_ht_z.clear()
        # print("Matrix cache cleared.")

    def set_limit_roi(self):
        """Sets the valid coordinate limits based on RGB camera intrinsics."""
        if not self.rgb_intric:
            # This can happen if reset() is called after init but before D2C
            # Or if called directly without init
            # print("Warning: Cannot set ROI limits, RGB intrinsics not available.")
            self.x_limit = np.array([0.0, 0.0], dtype=np.float32)
            self.y_limit = np.array([0.0, 0.0], dtype=np.float32)
            return

        # Inclusive limits [min, max]
        self.x_limit[0] = 0.0
        self.x_limit[1] = float(self.rgb_intric.width - 1)
        self.y_limit[0] = 0.0
        self.y_limit[1] = float(self.rgb_intric.height - 1)
        # print(f"ROI limits set: x={self.x_limit}, y={self.y_limit}")

    def _apply_target_distortion(self, tx: np.ndarray, ty: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Applies the configured RGB distortion model to normalized coordinates."""
        pt_norm = np.stack([tx, ty], axis=-1)
        valid_mask = np.ones(tx.shape, dtype=bool)  # Start assuming all are valid

        if self.rgb_disto.model == DistortionModel.OB_DISTORTION_BROWN_CONRADY:
            pt_distorted = add_distortion_vectorized(self.rgb_disto, pt_norm)
            tx_d, ty_d = pt_distorted[:, 0], pt_distorted[:, 1]
        elif self.rgb_disto.model == DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6:
            # Check r2 limit before applying distortion
            r2 = tx ** 2 + ty ** 2
            if self.r2_max_loc > 0:
                # Mark points beyond the safe radius as invalid
                valid_mask = r2 < self.r2_max_loc
                print(f"K6 Check: {np.sum(~valid_mask)} points invalidated by r2 > {self.r2_max_loc}")

            # Apply distortion only to valid points (or all if no limit)
            pt_norm_valid = np.stack([tx[valid_mask], ty[valid_mask]], axis=-1)
            pt_distorted_valid = add_distortion_vectorized(self.rgb_disto, pt_norm_valid)

            # Initialize output arrays
            tx_d = np.zeros_like(tx)
            ty_d = np.zeros_like(ty)

            # Fill in the results for valid points
            tx_d[valid_mask] = pt_distorted_valid[:, 0]
            ty_d[valid_mask] = pt_distorted_valid[:, 1]

        elif self.rgb_disto.model == DistortionModel.OB_DISTORTION_KANNALA_BRANDT4:
            pt_distorted = add_distortion_vectorized(self.rgb_disto, pt_norm)
            tx_d, ty_d = pt_distorted[:, 0], pt_distorted[:, 1]

        elif self.rgb_disto.model == DistortionModel.OB_DISTORTION_NONE:
            # No distortion applied
            tx_d, ty_d = tx, ty
        else:
            raise ValueError(f"Unsupported RGB distortion model for D2C: {self.rgb_disto.model}")

        return tx_d, ty_d, valid_mask

    def _d2c_vectorized(self, depth_buffer: np.ndarray,
                        out_depth: Optional[np.ndarray],
                        map_out: Optional[np.ndarray],
                        rgb_intric_proc: OBCameraIntrinsic,  # Intrinsics possibly scaled
                        coeff_x: List[np.ndarray],
                        coeff_y: List[np.ndarray],
                        coeff_z: List[np.ndarray]) -> None:
        """Core D2C logic implemented with NumPy vectorization."""

        depth_h, depth_w = depth_buffer.shape
        rgb_h, rgb_w = rgb_intric_proc.height, rgb_intric_proc.width
        num_channels = 1 if self.gap_fill_copy else 2
        total_depth_pixels = depth_h * depth_w

        # Flatten depth buffer for processing
        depth_flat = depth_buffer.flatten().astype(np.float32)  # Ensure float for calculations

        # --- Coordinate Calculation (Channels combined if num_channels > 1) ---
        all_pixel_x_f = []
        all_pixel_y_f = []
        all_depth_color_cam = []
        all_valid_masks = []
        all_original_indices = np.arange(total_depth_pixels)  # Keep track of original pixel index

        for i in range(num_channels):
            # Calculate 3D point coordinates in the *color* camera frame
            # X_c = d * coeff_x + T_x
            dst_x = depth_flat * coeff_x[i] + self.scaled_trans[0]
            dst_y = depth_flat * coeff_y[i] + self.scaled_trans[1]
            dst_z = depth_flat * coeff_z[i] + self.scaled_trans[2]  # This is depth in color cam frame

            # --- Validity Check 1: Input depth and calculated Z ---
            # Points with zero input depth or non-positive Z in color frame are invalid
            valid_mask_initial = (depth_flat > EPSILON) & (dst_z > EPSILON)

            # Filter out initially invalid points to avoid division by zero etc.
            valid_indices = np.where(valid_mask_initial)[0]
            if len(valid_indices) == 0:
                # print(f"Channel {i}: No valid depth points found.")
                all_pixel_x_f.append(np.array([], dtype=np.float32))
                all_pixel_y_f.append(np.array([], dtype=np.float32))
                all_depth_color_cam.append(np.array([], dtype=np.float32))
                all_valid_masks.append(np.array([], dtype=bool))
                continue  # Skip to next channel if no valid points

            # Process only valid points from now on
            original_indices_valid = all_original_indices[valid_indices]
            dst_x_valid = dst_x[valid_indices]
            dst_y_valid = dst_y[valid_indices]
            dst_z_valid = dst_z[valid_indices]  # Depth in color camera frame for valid points

            # Normalize coordinates (project onto image plane at Z=1)
            tx_valid = dst_x_valid / dst_z_valid
            ty_valid = dst_y_valid / dst_z_valid

            # --- Apply target (color) distortion if enabled ---
            valid_mask_distortion = np.ones(len(valid_indices), dtype=bool)  # Assume valid unless K6 fails
            if self.add_target_distortion:
                tx_distorted_valid, ty_distorted_valid, valid_mask_distortion = \
                    self._apply_target_distortion(tx_valid, ty_valid)
            else:
                tx_distorted_valid = tx_valid
                ty_distorted_valid = ty_valid
                # valid_mask_distortion remains all True

            # Update overall valid mask based on distortion check
            overall_valid_mask = valid_mask_distortion  # Boolean array matching _valid arrays

            # --- Project to pixel coordinates ---
            # Apply color camera intrinsics (potentially scaled)
            pixel_x_f_valid = tx_distorted_valid * rgb_intric_proc.fx + rgb_intric_proc.cx
            pixel_y_f_valid = ty_distorted_valid * rgb_intric_proc.fy + rgb_intric_proc.cy

            # Store results for this channel, only for the points that passed all checks
            final_valid_indices_in_channel = np.where(overall_valid_mask)[0]

            all_pixel_x_f.append(pixel_x_f_valid[final_valid_indices_in_channel])
            all_pixel_y_f.append(pixel_y_f_valid[final_valid_indices_in_channel])
            # Store the depth in the color camera frame (dst_z) for the valid points
            all_depth_color_cam.append(dst_z_valid[final_valid_indices_in_channel])
            # Store the original depth pixel index for mapping later
            all_valid_masks.append(original_indices_valid[final_valid_indices_in_channel])

        # --- Combine results and fill output buffers ---
        if not any(len(arr) > 0 for arr in all_pixel_x_f):
            # print("No valid points found across all channels.")
            # Ensure output buffers are zeroed/initialized if they exist
            if out_depth is not None: out_depth.fill(0)
            if map_out is not None: map_out.fill(-1)  # Or appropriate invalid marker
            return

        # --- Mapping to output buffers ---
        # We need to handle the two gap filling strategies

        if self.gap_fill_copy:  # num_channels == 1
            pixel_x_f = all_pixel_x_f[0]
            pixel_y_f = all_pixel_y_f[0]
            depth_values = all_depth_color_cam[0]  # Depth in color cam space
            original_indices = all_valid_masks[0]  # Original flat index of the depth pixel

            if len(pixel_x_f) == 0: return  # No valid points

            # Round coordinates to get integer pixel locations
            u_rgb = np.round(pixel_x_f).astype(int)
            v_rgb = np.round(pixel_y_f).astype(int)

            # --- Validity Check 2: Check if projected points are within color image bounds ---
            valid_bounds_mask = (u_rgb >= 0) & (u_rgb < rgb_w) & \
                                (v_rgb >= 0) & (v_rgb < rgb_h)

            # Filter coordinates and depths by bounds
            u_rgb_valid = u_rgb[valid_bounds_mask]
            v_rgb_valid = v_rgb[valid_bounds_mask]
            depth_values_valid = depth_values[valid_bounds_mask]
            original_indices_valid = original_indices[valid_bounds_mask]

            if len(u_rgb_valid) == 0: return  # No points within bounds

            # --- Update Output Depth Buffer (Handling Overwrites with min depth) ---
            if out_depth is not None:
                out_depth.fill(65535)  # Initialize with max value
                # Method: Create linear indices, sort by depth, keep first unique index
                linear_indices = v_rgb_valid * rgb_w + u_rgb_valid
                # Sort primarily by index, secondarily by depth (to keep the smallest depth for ties)
                sort_order = np.lexsort((depth_values_valid, linear_indices))
                u_rgb_sorted = u_rgb_valid[sort_order]
                v_rgb_sorted = v_rgb_valid[sort_order]
                depth_sorted = depth_values_valid[sort_order]
                linear_indices_sorted = linear_indices[sort_order]

                # Find unique indices (first occurrence corresponds to min depth due to sorting)
                unique_indices, first_occurrence_indices = np.unique(linear_indices_sorted, return_index=True)

                # Update output depth buffer at the unique locations
                out_depth.flat[unique_indices] = depth_sorted[first_occurrence_indices].astype(np.uint16)

                # --- Gap Filling (simple 4-pixel replication) ---
                # Apply to the unique points that were just written
                u_unique = u_rgb_sorted[first_occurrence_indices]
                v_unique = v_rgb_sorted[first_occurrence_indices]
                depth_unique = out_depth.flat[unique_indices]  # Get the depths we just wrote

                # Check boundaries for neighbors
                can_go_right = (u_unique + 1) < rgb_w
                can_go_down = (v_unique + 1) < rgb_h

                # Update right neighbor
                idx_r = (v_unique[can_go_right] * rgb_w + (u_unique[can_go_right] + 1))
                np.minimum(out_depth.flat[idx_r], depth_unique[can_go_right], out=out_depth.flat[idx_r])

                # Update bottom neighbor
                idx_b = ((v_unique[can_go_down] + 1) * rgb_w + u_unique[can_go_down])
                np.minimum(out_depth.flat[idx_b], depth_unique[can_go_down], out=out_depth.flat[idx_b])

                # Update bottom-right neighbor
                can_go_br = can_go_right & can_go_down
                idx_br = ((v_unique[can_go_br] + 1) * rgb_w + (u_unique[can_go_br] + 1))
                np.minimum(out_depth.flat[idx_br], depth_unique[can_go_br], out=out_depth.flat[idx_br])

            # --- Update Map Output Buffer ---
            if map_out is not None:
                map_out.fill(-1)  # Initialize map
                # Map the valid original depth indices to their (u, v) in the color image
                map_out[original_indices_valid, 0] = u_rgb_valid
                map_out[original_indices_valid, 1] = v_rgb_valid

        else:  # num_channels == 2 (gap_fill_copy = False) -> Fill rectangles
            pixel_x_f0, pixel_x_f1 = all_pixel_x_f
            pixel_y_f0, pixel_y_f1 = all_pixel_y_f
            depth_values0, depth_values1 = all_depth_color_cam
            original_indices0, original_indices1 = all_valid_masks  # Indices matching the valid points in each channel

            # Find the common original depth pixels that were valid in *both* channels
            common_original_indices, comm_idx0, comm_idx1 = np.intersect1d(
                original_indices0, original_indices1, return_indices=True)

            if len(common_original_indices) == 0:
                # print("Warning: No depth pixels were valid in both corner channels.")
                return  # Cannot form rectangles

            # Get coords and depths corresponding to the common pixels
            px0, py0 = pixel_x_f0[comm_idx0], pixel_y_f0[comm_idx0]
            px1, py1 = pixel_x_f1[comm_idx1], pixel_y_f1[comm_idx1]
            d0, d1 = depth_values0[comm_idx0], depth_values1[comm_idx1]

            # Round coordinates
            u0_r, v0_r = np.round(px0).astype(int), np.round(py0).astype(int)
            u1_r, v1_r = np.round(px1).astype(int), np.round(py1).astype(int)

            # Determine rectangle boundaries (ensure u0 <= u1, v0 <= v1)
            u_min = np.minimum(u0_r, u1_r)
            u_max = np.maximum(u0_r, u1_r)
            v_min = np.minimum(v0_r, v1_r)
            v_max = np.maximum(v0_r, v1_r)

            # Clip boundaries to image dimensions
            u_min_clip = np.maximum(0, u_min)
            v_min_clip = np.maximum(0, v_min)
            u_max_clip = np.minimum(rgb_w - 1, u_max)
            v_max_clip = np.minimum(rgb_h - 1, v_max)

            # Determine the depth value for the rectangle (minimum of the two corner depths)
            rect_depth = np.minimum(d0, d1).astype(np.uint16)

            # --- Update Output Depth Buffer (Fill Rectangles) ---
            if out_depth is not None:
                out_depth.fill(65535)  # Initialize
                # This requires iterating through the rectangles, as NumPy doesn't have a direct
                # vectorized "fill-rectangle-with-minimum" operation easily.
                print(f"Filling {len(common_original_indices)} rectangles (looping)...")  # DEBUG/Performance Warning
                for i in range(len(common_original_indices)):
                    u_start, u_end = u_min_clip[i], u_max_clip[i]
                    v_start, v_end = v_min_clip[i], v_max_clip[i]
                    current_rect_depth = rect_depth[i]

                    if u_start <= u_end and v_start <= v_end:  # Check if rectangle is valid after clipping
                        # Get the slice and update using minimum
                        rect_slice = out_depth[v_start:v_end + 1, u_start:u_end + 1]
                        np.minimum(rect_slice, current_rect_depth, out=rect_slice)

            # --- Update Map Output Buffer ---
            if map_out is not None:
                map_out.fill(-1)  # Initialize map
                # Map to the top-left corner (channel 0's rounded coordinate)
                # Check if the top-left corner is within bounds before mapping
                map_mask = (u0_r >= 0) & (u0_r < rgb_w) & \
                           (v0_r >= 0) & (v0_r < rgb_h)
                valid_map_indices = common_original_indices[map_mask]
                map_out[valid_map_indices, 0] = u0_r[map_mask]
                map_out[valid_map_indices, 1] = v0_r[map_mask]

    def _d2c_post_process(self, src_depth: np.ndarray, scale: float, out_w: int, out_h: int) -> np.ndarray:
        """Upsamples the aligned depth map using nearest-neighbor interpolation."""
        in_h, in_w = src_depth.shape
        # print(f"Post-processing (upsampling) {in_w}x{in_h} -> {out_w}x{out_h} with scale {scale:.2f}")

        # Create coordinate grid for the output image
        out_y_coords, out_x_coords = np.meshgrid(np.arange(out_h, dtype=np.float32),
                                                 np.arange(out_w, dtype=np.float32),
                                                 indexing='ij')

        # Calculate corresponding coordinates in the source (scaled-down) image
        # Add 0.5 for nearest neighbor centering before dividing by scale
        src_x_coords = (out_x_coords + 0.5) / scale - 0.5
        src_y_coords = (out_y_coords + 0.5) / scale - 0.5

        # Round to get nearest integer coordinates in the source image
        src_x_indices = np.round(src_x_coords).astype(int)
        src_y_indices = np.round(src_y_coords).astype(int)

        # Clip indices to be within the bounds of the source image
        np.clip(src_x_indices, 0, in_w - 1, out=src_x_indices)
        np.clip(src_y_indices, 0, in_h - 1, out=src_y_indices)

        # Sample from the source depth buffer using the calculated indices
        out_depth_scaled = src_depth[src_y_indices, src_x_indices]

        return out_depth_scaled

    def D2C(self, depth_buffer: np.ndarray,
            out_depth: Optional[np.ndarray] = None,
            map_out: Optional[np.ndarray] = None) -> Union[int, np.ndarray]:
        """
        Performs Depth-to-Color (D2C) alignment.

        Projects the depth frame onto the color camera's image plane.

        Args:
            depth_buffer: Input depth frame (H, W) as a NumPy array (uint16).
                          Units should correspond to depth_unit_mm set during initialization.
            out_depth: A pre-allocated NumPy array (color_h, color_w, uint16) to store the
                       aligned depth map. If None, depth output is skipped.
            map_out: A pre-allocated NumPy array (depth_h, depth_w, 2, int32) to store the
                     mapping from each depth pixel to its corresponding (u, v) coordinate
                     in the color image. Invalid mappings are marked with -1. If None,
                     map output is skipped.

        Returns:
            0 on success, -1 on failure (e.g., not initialized, bad parameters), the aligned depth map if out_depth is None.
        """
        if not self.initialized:
            print("Error: AlignImpl not initialized.")
            return -1

        if not isinstance(depth_buffer, np.ndarray) or depth_buffer.dtype != np.uint16:
            print("Error: depth_buffer must be a NumPy array of type uint16.")
            return -1

        if out_depth is None:
            out_depth = np.zeros((self.rgb_intric.height, self.rgb_intric.width), dtype=np.uint16)

        depth_h, depth_w = depth_buffer.shape
        if depth_w != self.depth_intric.width or depth_h != self.depth_intric.height:
            print(f"Error: Input depth dimensions ({depth_w}x{depth_h}) do not match "
                  f"initialized depth dimensions ({self.depth_intric.width}x{self.depth_intric.height}).")
            # Option: Could try to call prepare_depth_resolution here if allowed
            # self.prepare_depth_resolution() # Requires modifying initialize logic slightly
            return -1

        # Check output buffer dimensions if provided
        if out_depth is not None:
            if not isinstance(out_depth, np.ndarray) or out_depth.dtype != np.uint16:
                print("Error: out_depth must be a NumPy array of type uint16.")
                return -1
            if out_depth.shape != (self.rgb_intric.height, self.rgb_intric.width):
                print(f"Error: out_depth dimensions ({out_depth.shape[1]}x{out_depth.shape[0]}) do not match "
                      f"initialized color dimensions ({self.rgb_intric.width}x{self.rgb_intric.height}).")
                return -1
        if map_out is not None:
            if not isinstance(map_out, np.ndarray) or map_out.dtype != np.int32:
                print("Error: map_out must be a NumPy array of type int32.")
                return -1
            if map_out.shape != (depth_h, depth_w, 2):
                print(f"Error: map_out dimensions ({map_out.shape}) do not match expected ({depth_h}, {depth_w}, 2).")
                return -1

        # --- Get precomputed coefficients ---
        key = (depth_w, depth_h)
        if key not in self.rot_coeff_ht_x:
            print(f"Error: LUTs for depth resolution {key} not found. Was prepare_depth_resolution called?")
            # Or potentially call it here if auto-preparation is desired
            # self.prepare_depth_resolution()
            # if key not in self.rot_coeff_ht_x: return -1 # Failed to prepare
            return -1

        coeff_x = self.rot_coeff_ht_x[key]
        coeff_y = self.rot_coeff_ht_y[key]
        coeff_z = self.rot_coeff_ht_z[key]

        # --- Handle Scaling ---
        rgb_intric_processing = self.rgb_intric  # Use original by default
        scale = 1.0
        temp_out_depth = None  # Temporary buffer if scaling is used

        if self.use_scale:
            # Calculate scale factor (ensure fx is not zero)
            if abs(self.depth_intric.fx) < EPSILON or abs(self.rgb_intric.fx) < EPSILON:
                print("Warning: Cannot calculate scale factor due to zero focal length. Disabling scaling.")
                self.use_scale = False  # Disable for this run
            else:
                scale = self.rgb_intric.fx / self.depth_intric.fx
                if abs(scale - 1.0) > EPSILON:  # Only scale if significantly different
                    # print(f"Applying scaling factor: {scale:.2f}")
                    # Create temporary scaled intrinsics for processing
                    scaled_width = int(round(self.rgb_intric.width / scale))
                    scaled_height = int(round(self.rgb_intric.height / scale))
                    if scaled_width <= 0 or scaled_height <= 0:
                        print(f"Error: Invalid scaled dimensions ({scaled_width}x{scaled_height}). Disabling scaling.")
                        self.use_scale = False  # Disable scaling
                    else:
                        rgb_intric_processing = OBCameraIntrinsic(
                            width=scaled_width,
                            height=scaled_height,
                            fx=self.rgb_intric.fx / scale,
                            fy=self.rgb_intric.fy / scale,
                            cx=self.rgb_intric.cx / scale,
                            cy=self.rgb_intric.cy / scale
                        )
                        # If output depth is requested, we need a temporary buffer of the scaled size
                        if out_depth is not None:
                            temp_out_depth = np.zeros((scaled_height, scaled_width), dtype=np.uint16)
                else:
                    self.use_scale = False  # Scale is close to 1, no need to scale

        # Determine which output buffer to use for the core processing
        process_out_depth = temp_out_depth if self.use_scale and out_depth is not None else out_depth

        # --- Run the core vectorized D2C logic ---
        try:
            self._d2c_vectorized(depth_buffer, process_out_depth, map_out, rgb_intric_processing,
                                 coeff_x, coeff_y, coeff_z)
        except Exception as e:
            print(f"Error during D2C vectorization: {e}")
            import traceback
            traceback.print_exc()
            return -1

        # --- Post-processing for scaling ---
        if self.use_scale and out_depth is not None and temp_out_depth is not None:
            # print("Applying D2C post-processing (scaling)...")
            final_scaled_depth = self._d2c_post_process(temp_out_depth, scale,
                                                        self.rgb_intric.width, self.rgb_intric.height)
            # Copy result to the final output buffer
            np.copyto(out_depth, final_scaled_depth)

        # --- Final cleanup: Replace placeholder 65535 with 0 ---
        if out_depth is not None:
            out_depth[out_depth == 65535] = 0

        return out_depth

    def C2D(self, depth_buffer: np.ndarray, rgb_buffer: np.ndarray, out_rgb: np.ndarray) -> int:
        """
        Performs Color-to-Depth (C2D) alignment.

        Warps the color image to match the depth image perspective.

        Args:
            depth_buffer: Input depth frame (H, W, uint16).
            rgb_buffer: Input color frame (Hc, Wc, Channels) or (Hc, Wc). Must match color
                       intrinsics dimensions. Supported formats inferred from shape/dtype.
            out_rgb: Pre-allocated NumPy array (H, W, Channels) or (H, W) to store the
                     warped color image. Must match depth dimensions.

        Returns:
             0 on success, -1 on failure.
        """
        if not self.initialized:
            print("Error: AlignImpl not initialized.")
            return -1

        # --- Input Validation ---
        if not isinstance(depth_buffer, np.ndarray) or depth_buffer.dtype != np.uint16:
            print("Error: depth_buffer must be a NumPy array of type uint16.")
            return -1
        if not isinstance(rgb_buffer, np.ndarray):
            print("Error: rgb_buffer must be a NumPy array.")
            return -1
        if not isinstance(out_rgb, np.ndarray):
            print("Error: out_rgb must be a NumPy array.")
            return -1

        depth_h, depth_w = depth_buffer.shape
        if out_rgb.shape[:2] != (depth_h, depth_w):
            print(f"Error: out_rgb shape {out_rgb.shape} does not match depth shape {(depth_h, depth_w)}.")
            return -1
        if rgb_buffer.shape[:2] != (self.rgb_intric.height, self.rgb_intric.width):
            print(f"Error: rgb_buffer shape {rgb_buffer.shape} does not match color intrinsic dimensions "
                  f"({self.rgb_intric.height}x{self.rgb_intric.width}).")
            return -1
        # Check channel/dtype consistency
        if rgb_buffer.ndim != out_rgb.ndim or \
                (rgb_buffer.ndim == 3 and rgb_buffer.shape[2] != out_rgb.shape[2]) or \
                rgb_buffer.dtype != out_rgb.dtype:
            print("Error: rgb_buffer and out_rgb must have matching dimensions (channels) and dtype.")
            return -1

        # --- Get the D2C mapping ---
        # Create map buffer: (depth_h, depth_w, 2) for [u, v] coords in color image
        depth_xy_map = np.full((depth_h, depth_w, 2), -1, dtype=np.int32)

        # Call D2C, requesting only the map
        ret = self.D2C(depth_buffer, out_depth=None, map_out=depth_xy_map)

        if ret != 0:
            print("Error: Failed to compute D2C map for C2D.")
            return -1

        # --- Map Pixels using the generated map ---
        # print("Mapping color pixels to depth frame...")
        # Flatten map for easier indexing
        map_flat = depth_xy_map.reshape(-1, 2)  # Shape (N, 2) where N = depth_h * depth_w

        # Create indices for the destination (output) buffer
        dst_indices = np.arange(depth_h * depth_w)

        # Extract source u, v coordinates from the map
        src_u = map_flat[:, 0]
        src_v = map_flat[:, 1]

        # Create a mask for valid map entries
        valid_map_mask = (src_u >= 0)  # & (src_v >= 0) implicitly checked by u>=0

        # Filter destination indices and source coordinates using the mask
        dst_indices_valid = dst_indices[valid_map_mask]
        src_u_valid = src_u[valid_map_mask]
        src_v_valid = src_v[valid_map_mask]

        # Calculate linear indices for the source rgb_buffer
        src_linear_indices = src_v_valid * self.rgb_intric.width + src_u_valid

        # --- Perform the gather operation ---
        # Handles both 2D (grayscale/Y16) and 3D (RGB/BGR/RGBA/BGRA) arrays
        if rgb_buffer.ndim == 2:
            out_rgb.fill(0)  # Initialize output
            out_rgb.flat[dst_indices_valid] = rgb_buffer.flat[src_linear_indices]
        elif rgb_buffer.ndim == 3:
            num_channels = rgb_buffer.shape[2]
            # Reshape buffers to (N, C) for easier indexing
            rgb_buffer_flat_channels = rgb_buffer.reshape(-1, num_channels)
            out_rgb.fill(0)  # Initialize output
            out_rgb_flat_channels = out_rgb.reshape(-1, num_channels)
            # Gather pixel values
            out_rgb_flat_channels[dst_indices_valid] = rgb_buffer_flat_channels[src_linear_indices]
        else:
            print(f"Error: Unsupported number of dimensions in rgb_buffer: {rgb_buffer.ndim}")
            return -1

        # print("C2D mapping complete.")
        return 0


def align_depth_to_color_for_cam(depth_dir: str, out_dir: str, parameters_dir: str, start_offset: int = 0, total_frames: int = -1, depth_format: Literal['npy', 'png'] = 'npy', force: bool = False, celery_app=None):
    """
    Aligns color frames to depth frames for a given session.

    Parameters
    ----------
    depth_dir : str
        Path to the directory containing raw depth frames.
    out_dir : str
        Path to the directory where aligned depth frames will be saved.
    parameters_dir : str
        Path to the directory containing camera parameters (intrinsics, extrinsics, distortions).
    start_offset : int
        Start offset for the frames to process
    total_frames : int
        Total number of frames to process. If -1, process all frames.
    depth_format : Literal['npy', 'png']
        Format of the depth frames to save. 'npy' for numpy arrays, 'png' for 16-bit PNG images.
    force : bool
        Whether to force re-generation of depth frames when they exist.
    celery_app : Optional[celery.Celery]
        The Celery app instance for task management, by default None. If provided, the task progress will be tracked and tqdm will be disabled.
    """
    depth_dir = Path(depth_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir = Path(parameters_dir)
    color_dir = depth_dir.parent / 'color'
    color_size_hw = PathUtils.read_file(Path(str(next(iter(color_dir.glob('*.jpg')))))).shape[:2]

    # align depth frames to color frames
    depth_dir = Path(depth_dir)
    depth_files_npy = sorted(depth_dir.glob('*.npy'), key=lambda x: int(x.stem))  # global indexing
    if len(depth_files_npy) == 0:
        assert depth_format != 'npy', "No .npy depth files found in the directory. Please provide a valid depth directory with .npy files."
        depth_files = sorted(depth_dir.glob(f'*.{depth_format}'), key=lambda x: int(x.stem))  # global indexing
    else:
        depth_files = depth_files_npy
    if total_frames < 0:
        total_frames = len(depth_files) + total_frames - start_offset + 1
    all_depth_files = depth_files[start_offset:start_offset + total_frames]
    dest_depth_files = [out_dir / f.with_suffix(f'.{depth_format}').name for f in all_depth_files]

    # ATTN: some of the existing files were corrupted, so we force checking of each file
    if all(f.exists() and depth_format == 'png' and PathUtils.verify_file(f) for f in dest_depth_files) and not force:
        return None

    # load parameters
    c_intri = np.load(parameters_dir / 'color_intri.npy')
    d_intri = np.load(parameters_dir / 'depth_intri.npy')
    c_dist = np.load(parameters_dir / 'color_dist.npy')
    d_dist = np.load(parameters_dir / 'depth_dist.npy')
    extri = np.load(parameters_dir / 'depth_extri2color.npy')
    first_depth = PathUtils.read_file(all_depth_files[0], png_type='depth')
    aligner_kwargs = dict(
        depth_intri=OBCameraIntrinsic(
            width=first_depth.shape[-1],
            height=first_depth.shape[-2],
            fx=d_intri[0, 0].item(), fy=d_intri[1, 1].item(),
            cx=d_intri[0, 2].item(), cy=d_intri[1, 2].item()
        ),
        depth_dist=OBCameraDistortion(
            model=DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6,
            k1=d_dist[0].item(), k2=d_dist[1].item(),
            p1=d_dist[2].item(), p2=d_dist[3].item(),
            k3=d_dist[4].item(), k4=d_dist[5].item(), k5=d_dist[6].item(), k6=d_dist[7].item()
        ),
        color_intri=OBCameraIntrinsic(
            width=color_size_hw[1],
            height=color_size_hw[0],
            fx=c_intri[0, 0].item(), fy=c_intri[1, 1].item(),
            cx=c_intri[0, 2].item(), cy=c_intri[1, 2].item()
        ),
        color_dist=OBCameraDistortion(
            model=DistortionModel.OB_DISTORTION_KANNALA_BRANDT4,
            k1=c_dist[0].item(), k2=c_dist[1].item(),
            p1=c_dist[2].item(), p2=c_dist[3].item(),
            k3=c_dist[4].item(), k4=c_dist[5].item(), k5=c_dist[6].item(), k6=c_dist[7].item()
        ),
        depth2color_extri=OBExtrinsic(
            rot=extri[:3, :3],
            trans=extri[:3, 3]
        ),
        depth_unit_mm=1.0,
        add_target_distortion=False,
        gap_fill_copy=True,
        use_scale=True
    )
    aligner = AlignImpl().initialize(**aligner_kwargs).D2C
    for i, (depth_path, aligned_depth_path) in tqdm(enumerate(zip(all_depth_files, dest_depth_files)), desc=f'Aligning depth frames (cam: {depth_dir.parent.name})', disable=celery_app is not None):
        if aligned_depth_path.exists() and depth_format == 'png' and PathUtils.verify_file(aligned_depth_path) and not force:
            continue

        # Load depth frame
        depth = PathUtils.read_file(depth_path, png_type='depth')
        # check if already aligned (size matches color size)
        if depth.shape[0] != color_size_hw[0] or depth.shape[1] != color_size_hw[1]:
            # Align depth frame to color frame
            aligned_depth = aligner(depth)
        else:
            aligned_depth = depth
        # Save aligned depth frame
        if not aligned_depth_path.exists():
            PathUtils.write_file(aligned_depth_path, aligned_depth, png_type='depth')

    return True