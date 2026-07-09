import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Literal

import cv2
import numpy as np
import toml
from scipy.optimize import least_squares

from tasks import app, AutoRetryTask
from utils.misc import PathUtils, log

MAX_BOARD_RMSE = 0.040


@dataclass
class CameraObservation3D:
    """3D observation of a ChArUco board from a single camera and frame.

    Parameters
    ----------
    ids : ndarray, shape (K,)
        Corner ids (board corner indices).
    points_cam : ndarray, shape (K, 3)
        3D points in camera coordinates.
    """
    ids: np.ndarray
    points_cam: np.ndarray


@dataclass
class Cluster3D:
    """Synchronized multi-camera 3D ChArUco observations.

    Parameters
    ----------
    frame_index : int
        Index of the synchronized frame.
    detections : dict
        Mapping from camera index to `CameraObservation3D`.
    """
    frame_index: int
    detections: Dict[int, CameraObservation3D]


def _rigid_transform_3d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rigid transform (R, t) aligning A to B.

    Solves for R, t such that R @ A_i + t ≈ B_i in least-squares sense.

    Parameters
    ----------
    A : ndarray, shape (N, 3)
        Source 3D points.
    B : ndarray, shape (N, 3)
        Target 3D points.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    t : ndarray, shape (3,)
        Translation vector.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.shape[1] != 3:
        raise ValueError("A and B must both have shape (N, 3)")

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1.0
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


def _depth_to_3d(intrinsics: np.ndarray, corners_2d: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Backproject 2D points with depth to 3D camera space.

    Parameters
    ----------
    intrinsics : np.ndarray, shape (3, 3)
        Camera intrinsics.
    corners_2d : ndarray, shape (K, 2)
        Pixel coordinates (u, v).
    depths : ndarray, shape (K,)
        Depth values in meters.

    Returns
    -------
    points_cam : ndarray, shape (K, 3)
        3D points in camera coordinates.
    """
    K = intrinsics.astype(np.double)
    dist = None  # data are already undistorted

    pts = corners_2d.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)

    z = depths.reshape(-1, 1)
    x = undist[:, 0:1] * z
    y = undist[:, 1:2] * z

    points_cam = np.concatenate([x, y, z], axis=1)
    return points_cam.astype(np.float64)


@app.task(name="calibration.detect_corners_2d", base=AutoRetryTask)
def detect_corners_2d(color_dir: str, out_dir: str, session_metadata_path: str, total_detections: int = -1, force: bool = False):
    """Detect full ChArUco boards in color images and store 2D corners.

    For each color image in `color_dir`, this task creates a JSON file with
    the same stem in `our_dir`. If detection is successful (all board corners
    detected and detection limit not exceeded), the JSON contains:
    - ids: list of corner ids
    - corners_2d: list of [u, v] pixel coordinates
    - board: dict with board configuration parameters

    Otherwise, the JSON contains an empty object {}.

    Parameters
    ----------
    color_dir : str
        Directory containing color images (jpg).
    out_dir : str
        Directory where detection JSON files will be written.
    session_metadata_path : int
        Path to session_metadata.json file from which to load the charuco profile key.
    total_detections : int, optional
        Maximum number of successful detections to store. If negative,
        all successful detections are stored.
    """
    color_dir_path = Path(color_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    color_files = sorted(color_dir_path.glob("*.jpg"), key=lambda p: p.stem)
    if not color_files:
        raise RuntimeError(f"No color images found in {color_dir_path}")

    # Load charuco profile
    if Path(session_metadata_path).exists():
        with open(session_metadata_path, 'r') as fp:
            calibration_pattern = json.load(fp).get('calibration_pattern', 'charuco_6x4_a2')
    else:
        log('[calibration.detect_corners_2d] session_metadata.json not found, using default charuco_6x4_a2', 'warning')
        calibration_pattern = 'charuco_6x4_a2'
    with open(PathUtils.resources_path() / 'calibration_patterns' / calibration_pattern / 'charuco_info.json', 'r') as fp:
        charuco_profile = json.load(fp)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, charuco_profile['dictionary']))
    board = cv2.aruco.CharucoBoard(
        size=[int(charuco_profile['columns']), int(charuco_profile['rows'])],
        squareLength=float(charuco_profile['square_size_overide_cm']) * 0.01,
        markerLength=float(charuco_profile['square_size_overide_cm']) * float(charuco_profile['aruco_scale']) * 0.01,
        dictionary=aruco_dict,
    )
    board_num_corners = int(np.asarray(board.getChessboardCorners()).shape[0])

    # detect corners
    detector_params = cv2.aruco.DetectorParameters()
    det_count = 0
    for color_path in color_files:
        json_path = out_dir_path / f"{color_path.stem}.json"
        if json_path.exists() and not force:
            continue

        if det_count >= total_detections > 0:
            with open(json_path, "w") as f:
                json.dump({}, f)
            continue

        img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if img is None:
            with open(json_path, "w") as f:
                json.dump({}, f)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )
        if ids is None or len(ids) == 0:
            with open(json_path, "w") as f:
                json.dump({}, f)
            continue
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board,
        )
        if charuco_ids is None or charuco_corners is None or charuco_ids.shape[0] != board_num_corners:
            with open(json_path, "w") as f:
                json.dump({}, f)
            continue

        ids_flat = charuco_ids.flatten().astype(int)
        corners_2d = charuco_corners[:, 0, :].astype(float)

        order = np.argsort(ids_flat)
        ids_sorted = ids_flat[order]
        corners_sorted = corners_2d[order]

        with open(json_path, "w") as f:
            json.dump({
                "ids": ids_sorted.tolist(),
                "corners_2d": corners_sorted.tolist(),
                "board": {
                    "squaresX": int(charuco_profile['columns']),
                    "squaresY": int(charuco_profile['rows']),
                    "squareLength": float(charuco_profile['square_size_overide_cm']) * 0.01,
                    "markerLength": float(charuco_profile['square_size_overide_cm']) * float(charuco_profile['aruco_scale']) * 0.01,
                    "dictionary": int(getattr(cv2.aruco, charuco_profile['dictionary'])),
                },
            }, f)
        det_count += 1

    log(f'[calibration::detect_corners_3d] In {color_dir_path.parent.name}/{color_dir_path.name} OpenCV detected: {det_count} boards, ')
    return True


@app.task(name="calibration.lift_corners_3d", base=AutoRetryTask)
def lift_corners_3d(corners_dir: str, depth_dir: str, parameters_dir: str, out_dir: str, force: bool = False):
    """Lift 2D ChArUco corners to 3D camera coordinates using depth.

    For each JSON file in `corners_dir` (produced by `detect_corners_opencv`),
    this task loads the corresponding depth file (by index order) and intrinsics
    from `parameters_dir`. It then:

    - For each 2D corner, looks at an 11x11 window around the rounded pixel.
    - Gathers all non-zero depth values in this window and takes their median.
    - If all depths are valid, backprojects to 3D camera space.
    - Writes a JSON with:
      - ids
      - corners_2d
      - corners_3d
      - board (copied from input JSON)

    If no valid full-depth detection is possible, an empty JSON {} is written.

    Parameters
    ----------
    corners_dir : str
        Directory containing 2D corner JSON files.
    depth_dir : str
        Directory containing depth frames (png or npy).
    parameters_dir : str
        Directory containing camera intrinsics (color_intri.npy).
    out_dir : str
        Directory where 3D corner JSON files will be written.
    """
    corners_dir_path = Path(corners_dir)
    depth_dir_path = Path(depth_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    corner_files = sorted(corners_dir_path.glob("*.json"), key=lambda p: p.stem)
    depth_files = sorted(
        list(depth_dir_path.glob("*.png")) + list(depth_dir_path.glob("*.npy")),
        key=lambda p: p.stem,
    )

    if len(corner_files) == 0:
        raise RuntimeError(f"No corner JSON files found in {corners_dir_path}")
    if len(corner_files) != len(depth_files):
        raise RuntimeError(
            f"Mismatched corner/depth file counts: {len(corner_files)} vs {len(depth_files)}"
        )

    intr = np.load(str(Path(parameters_dir) / "color_intri.npy"))

    valid_count = 0
    window_radius = 5  # 11x11 window

    for idx, corner_path in enumerate(corner_files):
        depth_path = depth_files[idx]
        out_path = out_dir_path / corner_path.name
        if out_path.exists() and not force:
            continue

        with open(corner_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

        if not data or "ids" not in data or "corners_2d" not in data:
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        ids = np.asarray(data["ids"], dtype=int)
        corners_2d = np.asarray(data["corners_2d"], dtype=np.float32)
        if ids.size == 0 or corners_2d.shape[0] == 0:
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        if depth_path.suffix == ".npy":
            depth_raw = np.load(depth_path)
        else:
            depth_raw = PathUtils.read_file(depth_path, png_type="depth")

        if depth_raw is None:
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        if depth_raw.dtype == np.uint16:
            depth_raw = depth_raw.astype(np.float32) / 1000.0

        depth = depth_raw.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]

        h, w = depth.shape[:2]
        u_px = np.rint(corners_2d[:, 0]).astype(int)
        v_px = np.rint(corners_2d[:, 1]).astype(int)

        if not (np.all((u_px >= 0) & (u_px < w)) and np.all((v_px >= 0) & (v_px < h))):
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        depths = np.zeros(len(u_px), dtype=np.float32)
        valid_depths = True

        for i, (u, v) in enumerate(zip(u_px, v_px)):
            u0 = max(0, u - window_radius)
            u1 = min(w, u + window_radius + 1)
            v0 = max(0, v - window_radius)
            v1 = min(h, v + window_radius + 1)

            patch = depth[v0:v1, u0:u1]
            mask = (patch > 0.0) & np.isfinite(patch)

            if not np.any(mask):
                valid_depths = False
                break

            depths[i] = np.median(patch[mask])

        if not valid_depths or np.any(depths <= 0.0) or np.any(~np.isfinite(depths)):
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        points_cam = _depth_to_3d(intr, corners_2d, depths)
        out_data = {
            "ids": ids.tolist(),
            "corners_2d": corners_2d.astype(float).tolist(),
            "corners_3d": points_cam.astype(float).tolist(),
        }
        if "board" in data:
            out_data["board"] = data["board"]

        with open(out_path, "w") as f:
            json.dump(out_data, f)
        valid_count += 1

    return valid_count


class _MultiViewCalibratorFull:
    """Multi-camera 3D ChArUco calibration with bundle adjustment."""

    def __init__(
            self,
            clusters: List[Cluster3D],
            board,
            camera_names: Sequence[str],
            active_cams: Sequence[int],
            min_cams_per_cluster_ba: int = 3,
    ) -> None:
        self.clusters = clusters
        self.board = board
        self.camera_names = list(camera_names)
        self.active_cams = sorted(set(active_cams))
        self.min_cams_per_cluster_ba = int(min_cams_per_cluster_ba)

        self.num_cameras = len(self.camera_names)
        self.num_frames = len(self.clusters)

        self.board_points_all = np.asarray(
            self.board.getChessboardCorners(), dtype=np.float64
        )
        self.board_num_corners = int(self.board_points_all.shape[0])

        self.cam_index_map = {idx: i for i, idx in enumerate(self.active_cams)}

    def _initial_guess(self) -> np.ndarray:
        """Compute initial guess for camera extrinsics and board poses."""
        root_cam_idx = self.active_cams[0]
        root_internal_idx = self.cam_index_map[root_cam_idx]

        # Camera extrinsics: world = root camera
        R_wc = [None] * self.num_cameras
        t_wc = [None] * self.num_cameras

        R_wc[root_cam_idx] = np.eye(3, dtype=np.float64)
        t_wc[root_cam_idx] = np.zeros(3, dtype=np.float64)

        # Board pose per frame (cluster)
        R_wb_list: List[Optional[np.ndarray]] = [None] * self.num_frames
        t_wb_list: List[Optional[np.ndarray]] = [None] * self.num_frames

        # Iterative propagation over the camera graph via overlapping clusters
        known_cams = {root_cam_idx}
        updated = True
        iteration = 0

        while updated:
            updated = False
            iteration += 1

            for frame_idx, cluster in enumerate(self.clusters):
                # Find any known camera in this cluster
                known_in_cluster = [
                    c for c in cluster.detections.keys() if c in known_cams
                ]
                if not known_in_cluster:
                    continue

                # Estimate board pose in world if not yet estimated
                if R_wb_list[frame_idx] is None:
                    ref_cam = known_in_cluster[0]
                    obs = cluster.detections[ref_cam]

                    ids = obs.ids.astype(int)
                    order = np.argsort(ids)
                    P_b = self.board_points_all[ids[order]]
                    P_c = obs.points_cam[order]

                    R_wc_ref = R_wc[ref_cam]
                    t_wc_ref = t_wc[ref_cam]
                    Pw = (R_wc_ref @ P_c.T + t_wc_ref[:, None]).T

                    R_wb, t_wb = _rigid_transform_3d(P_b, Pw)
                    R_wb_list[frame_idx] = R_wb
                    t_wb_list[frame_idx] = t_wb
                    updated = True

                R_wb = R_wb_list[frame_idx]
                t_wb = t_wb_list[frame_idx]

                Pw_frame = None
                if R_wb is not None:
                    # Precompute board points in world for this frame
                    # (Same for all cameras in this cluster)
                    ids_any = (
                        next(iter(cluster.detections.values())).ids.astype(int)
                    )
                    order_any = np.argsort(ids_any)
                    P_b_any = self.board_points_all[ids_any[order_any]]
                    Pw_frame = (R_wb @ P_b_any.T + t_wb[:, None]).T

                for cam_idx, obs in cluster.detections.items():
                    if cam_idx in known_cams:
                        continue

                    # Use the same ordering as above to align P_c with P_w
                    ids = obs.ids.astype(int)
                    order = np.argsort(ids)
                    P_c = obs.points_cam[order]

                    # Map board points via ids to this camera order
                    P_b = self.board_points_all[ids[order]]
                    if R_wb is None:
                        continue
                    Pw = (R_wb @ P_b.T + t_wb[:, None]).T

                    R, t = _rigid_transform_3d(P_c, Pw)
                    R_wc[cam_idx] = R
                    t_wc[cam_idx] = t
                    known_cams.add(cam_idx)
                    updated = True

        # Only active cameras that ended up known are used
        active_known = [c for c in self.active_cams if R_wc[c] is not None]
        if active_known[0] != root_cam_idx:
            raise RuntimeError("Root camera lost during initialization.")

        self.active_cams = active_known
        self.cam_index_map = {idx: i for i, idx in enumerate(self.active_cams)}

        # Ensure all clusters have board pose; fallback using any known camera
        for frame_idx, cluster in enumerate(self.clusters):
            if R_wb_list[frame_idx] is not None:
                continue
            known_in_cluster = [
                c for c in cluster.detections.keys() if c in self.active_cams
            ]
            if not known_in_cluster:
                continue
            ref_cam = known_in_cluster[0]
            obs = cluster.detections[ref_cam]

            ids = obs.ids.astype(int)
            order = np.argsort(ids)
            P_b = self.board_points_all[ids[order]]
            P_c = obs.points_cam[order]

            R_wc_ref = R_wc[ref_cam]
            t_wc_ref = t_wc[ref_cam]
            Pw = (R_wc_ref @ P_c.T + t_wc_ref[:, None]).T

            R_wb, t_wb = _rigid_transform_3d(P_b, Pw)
            R_wb_list[frame_idx] = R_wb
            t_wb_list[frame_idx] = t_wb

        # Pack parameters: cameras (excluding root) then board poses
        num_active_cams = len(self.active_cams)
        num_cam_params = 6 * (num_active_cams - 1)
        num_frame_params = 6 * self.num_frames
        x0 = np.zeros(num_cam_params + num_frame_params, dtype=np.float64)

        # Camera extrinsics
        cam_param_offset = 0
        for cam_idx in self.active_cams[1:]:
            R = R_wc[cam_idx]
            t = t_wc[cam_idx]
            rvec, _ = cv2.Rodrigues(R)
            x0[cam_param_offset: cam_param_offset + 3] = rvec.flatten()
            x0[cam_param_offset + 3: cam_param_offset + 6] = t
            cam_param_offset += 6

        # Board poses per frame
        for frame_idx in range(self.num_frames):
            base = num_cam_params + 6 * frame_idx
            R_wb = R_wb_list[frame_idx]
            t_wb = t_wb_list[frame_idx]
            if R_wb is None or t_wb is None:
                # Default if not available
                R_wb = np.eye(3, dtype=np.float64)
                t_wb = np.zeros(3, dtype=np.float64)
            rvec, _ = cv2.Rodrigues(R_wb)
            x0[base: base + 3] = rvec.flatten()
            x0[base + 3: base + 6] = t_wb

        return x0

    def _unpack_camera_params(self, x: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Unpack camera extrinsics from parameter vector."""
        num_active_cams = len(self.active_cams)
        cam_params: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Root camera is identity
        root_cam_idx = self.active_cams[0]
        cam_params[root_cam_idx] = (np.eye(3, dtype=np.float64), np.zeros(3))

        cam_param_offset = 0
        for cam_idx in self.active_cams[1:]:
            rvec = x[cam_param_offset: cam_param_offset + 3]
            t = x[cam_param_offset + 3: cam_param_offset + 6]
            R, _ = cv2.Rodrigues(rvec)
            cam_params[cam_idx] = (R, t)
            cam_param_offset += 6

        return cam_params

    def _bundle_residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute bundle adjustment residuals."""
        num_active_cams = len(self.active_cams)
        num_cam_params = 6 * (num_active_cams - 1)

        cam_params = self._unpack_camera_params(x)
        residuals = []

        for frame_idx, cluster in enumerate(self.clusters):
            # Count how many active cameras are present in this cluster
            active_in_cluster = [
                cam_idx for cam_idx in cluster.detections.keys()
                if cam_idx in self.active_cams
            ]
            if len(active_in_cluster) < self.min_cams_per_cluster_ba:
                continue  # use cluster for graph/init, but not for BA

            base_frame = num_cam_params + 6 * frame_idx
            rvec_wb = x[base_frame: base_frame + 3]
            t_wb = x[base_frame + 3: base_frame + 6]
            R_wb, _ = cv2.Rodrigues(rvec_wb)

            for cam_idx in active_in_cluster:
                obs = cluster.detections[cam_idx]

                ids = obs.ids.astype(int)
                order = np.argsort(ids)
                ids_sorted = ids[order]
                P_c_meas = obs.points_cam[order]

                P_b = self.board_points_all[ids_sorted]
                Pw = (R_wb @ P_b.T + t_wb[:, None]).T

                R_wc, t_wc = cam_params[cam_idx]
                R_cw = R_wc.T
                t_cw = -R_cw @ t_wc

                P_c_pred = (R_cw @ Pw.T + t_cw[:, None]).T

                residuals.append((P_c_pred - P_c_meas).ravel())

        if not residuals:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(residuals)

    def _compute_error_stats(self, x_opt: np.ndarray, num_bins: int = 20) -> dict:
        """Compute global RMSE, per-cluster RMSE and error histogram."""
        num_active_cams = len(self.active_cams)
        num_cam_params = 6 * (num_active_cams - 1)

        cam_params = self._unpack_camera_params(x_opt)

        all_errors: List[float] = []
        per_cluster_sq_sum = np.zeros(self.num_frames, dtype=np.float64)
        per_cluster_counts = np.zeros(self.num_frames, dtype=np.int64)

        for frame_idx, cluster in enumerate(self.clusters):
            active_in_cluster = [
                cam_idx for cam_idx in cluster.detections.keys()
                if cam_idx in self.active_cams
            ]
            if len(active_in_cluster) < self.min_cams_per_cluster_ba:
                continue

            base_frame = num_cam_params + 6 * frame_idx
            rvec_wb = x_opt[base_frame: base_frame + 3]
            t_wb = x_opt[base_frame + 3: base_frame + 6]
            R_wb, _ = cv2.Rodrigues(rvec_wb)

            for cam_idx in active_in_cluster:
                obs = cluster.detections[cam_idx]

                ids = obs.ids.astype(int)
                order = np.argsort(ids)
                ids_sorted = ids[order]
                P_c_meas = obs.points_cam[order]

                P_b = self.board_points_all[ids_sorted]
                Pw = (R_wb @ P_b.T + t_wb[:, None]).T

                R_wc, t_wc = cam_params[cam_idx]
                R_cw = R_wc.T
                t_cw = -R_cw @ t_wc

                P_c_pred = (R_cw @ Pw.T + t_cw[:, None]).T

                errs = np.linalg.norm(P_c_pred - P_c_meas, axis=1)
                all_errors.extend(errs.tolist())
                per_cluster_sq_sum[frame_idx] += np.sum(errs ** 2)
                per_cluster_counts[frame_idx] += errs.shape[0]

        all_errors = np.asarray(all_errors, dtype=np.float64)
        if all_errors.size == 0:
            return {
                "rmse": float("nan"),
                "per_cluster_rmse": [],
                "error_histogram": {"bin_edges": [], "counts": []},
            }

        global_rmse = float(np.sqrt(np.mean(all_errors ** 2)))

        per_cluster_rmse = []
        for frame_idx in range(self.num_frames):
            n = per_cluster_counts[frame_idx]
            if n == 0:
                continue
            rmse = float(np.sqrt(per_cluster_sq_sum[frame_idx] / n))
            per_cluster_rmse.append(
                {
                    "frame_index": int(self.clusters[frame_idx].frame_index),
                    "rmse": rmse,
                    "num_points": int(n),
                }
            )

        counts, bin_edges = np.histogram(all_errors, bins=num_bins)
        error_histogram = {
            "bin_edges": bin_edges.astype(float).tolist(),
            "counts": counts.astype(int).tolist(),
        }

        return {
            "rmse": global_rmse,
            "per_cluster_rmse": per_cluster_rmse,
            "error_histogram": error_histogram,
        }

    def plot_camera_poses(self, poses: Dict[str, np.ndarray], out_path: str | Path) -> None:
        """Plot calibrated camera centers and look directions.

        Parameters
        ----------
        poses : dict
            Mapping from camera name to 4x4 homogeneous transform T_wc (world-to-camera).
        out_path : str or Path
            Output path where the plot image will be saved (e.g. .png).
        """

        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for server/worker
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

        out_path = Path(out_path)

        if not poses:
            return

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        centers = []
        dirs = []
        labels = []

        for name, T in poses.items():
            T = np.asarray(T, dtype=np.float64)
            R_wc = T[:3, :3]
            t_wc = T[:3, 3]

            # Camera center in world frame: X_w = R_wc^T (X_c - t_wc), so for X_c=0: C_w = -R_wc^T t_wc
            C_w = -R_wc.T @ t_wc

            # Forward/look direction: world vector corresponding to camera +Z axis
            f_w = R_wc.T @ np.array([0.0, 0.0, 1.0])

            centers.append(C_w)
            dirs.append(f_w)
            labels.append(name)

        centers = np.vstack(centers)
        dirs = np.vstack(dirs)

        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker="o")

        # Arrow length; adjust to taste relative to scene scale
        arrow_len = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)) * 0.1
        if not np.isfinite(arrow_len) or arrow_len <= 0:
            arrow_len = 0.1

        for C_w, f_w, name in zip(centers, dirs, labels):
            p2 = C_w + arrow_len * f_w
            ax.plot(
                [C_w[0], p2[0]],
                [C_w[1], p2[1]],
                [C_w[2], p2[2]],
            )
            ax.text(C_w[0], C_w[1], C_w[2], name)

        # Make axes roughly equal
        max_range = (centers.max(axis=0) - centers.min(axis=0)).max()
        if max_range <= 0:
            max_range = 1.0
        mid = centers.mean(axis=0)
        for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
            axis(m - max_range / 2.0, m + max_range / 2.0)

        ax.set_xlabel("X (world)")
        ax.set_ylabel("Y (world)")
        ax.set_zlabel("Z (world)")
        ax.set_title("Calibrated Camera Poses")

        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    def compute_pixel_rmse(
            self,
            per_cam_data: List[List[Optional[dict]]],
            cam_dirs: List[Path],
            camera_names: Sequence[str],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute 3D->2D reprojection RMSE in pixels per camera and globally.

        For each camera and each frame where we have:
        - 'corners_3d' : 3D points in that camera's coordinates
        - 'corners_2d' : detected 2D Charuco corners
        - intrinsics   : color_intri.npy under the camera parameters dir

        we:
        - project corners_3d into the image using intrinsics
        - compare to corners_2d (in pixels)
        - accumulate squared pixel errors

        Parameters
        ----------
        per_cam_data : list
            per_cam_data[cam_idx][frame_idx] is the JSON dict for that
            camera/frame, as loaded earlier in calibrate_hslu.
        cam_dirs : list of Path
            Directories containing 3D corner JSON files, one per camera.
        camera_names : sequence of str
            Names of cameras, index-aligned with cam_dirs.

        Returns
        -------
        rmse_global_px : float
            Global RMSE over all cameras and frames, in pixels.
        rmse_per_cam_px : dict
            Mapping camera_name -> RMSE in pixels (NaN if no valid measurements).
        """
        # Load intrinsics per camera
        intrinsics_by_cam: Dict[int, np.ndarray] = {}
        for cam_idx, cam_dir in enumerate(cam_dirs):
            # cam_dir is the 3D-corners dir; its parent is the camera dir (e.g. cam01)
            cam_root = cam_dir.parent
            param_dir = cam_root / "parameters"
            K_path = param_dir / "color_intri.npy"
            if not K_path.exists():
                log(f"[calibration::calibrate_hslu] Intrinsics not found for {camera_names[cam_idx]} at {K_path}, "
                    f"skipping pixel RMSE for this camera.", "warning")
                continue
            K = np.load(str(K_path))
            intrinsics_by_cam[cam_idx] = K

        per_cam_sq_err: Dict[int, float] = {i: 0.0 for i in range(len(cam_dirs))}
        per_cam_count: Dict[int, int] = {i: 0 for i in range(len(cam_dirs))}

        for cam_idx, cam_data in enumerate(per_cam_data):
            if cam_idx not in intrinsics_by_cam:
                continue
            K = intrinsics_by_cam[cam_idx]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            for frame_data in cam_data:
                if (
                        not frame_data
                        or "corners_3d" not in frame_data
                        or "corners_2d" not in frame_data
                ):
                    continue

                P_c = np.asarray(frame_data["corners_3d"], dtype=np.float64)  # (N, 3)
                uv_meas = np.asarray(frame_data["corners_2d"], dtype=np.float64)  # (N, 2)

                if P_c.shape[0] == 0 or P_c.shape[0] != uv_meas.shape[0]:
                    continue

                X = P_c[:, 0]
                Y = P_c[:, 1]
                Z = P_c[:, 2]

                valid = Z > 0.0
                if not np.any(valid):
                    continue

                X = X[valid]
                Y = Y[valid]
                Z = Z[valid]
                uv_meas_valid = uv_meas[valid]

                # Pinhole projection (no distortion, already undistorted)
                u_pred = fx * (X / Z) + cx
                v_pred = fy * (Y / Z) + cy
                uv_pred = np.stack([u_pred, v_pred], axis=1)

                err = uv_pred - uv_meas_valid
                se = float(np.sum(err ** 2))
                n = err.shape[0]

                per_cam_sq_err[cam_idx] += se
                per_cam_count[cam_idx] += n

        # Global RMSE
        global_sq = sum(per_cam_sq_err.values())
        global_n = sum(per_cam_count.values())
        if global_n > 0:
            rmse_global_px = float(np.sqrt(global_sq / global_n))
        else:
            rmse_global_px = float("nan")

        # Per-camera RMSE
        rmse_per_cam_px: Dict[str, float] = {}
        for cam_idx, name in enumerate(camera_names):
            n = per_cam_count.get(cam_idx, 0)
            if n > 0:
                rmse = float(np.sqrt(per_cam_sq_err[cam_idx] / n))
            else:
                rmse = float("nan")
            rmse_per_cam_px[name] = rmse

        return rmse_global_px, rmse_per_cam_px

    def calibrate(self) -> Tuple[Dict[str, np.ndarray], dict]:
        """Run bundle adjustment and return camera poses and error stats.

        Returns
        -------
        poses : dict
            Mapping from camera name to 4x4 homogeneous transform T_wc
            (world-to-camera) for each active camera.
        stats : dict
            Error statistics (global RMSE, per-cluster RMSE, histogram).
        """
        if not self.clusters:
            raise RuntimeError("No clusters available for calibration.")

        x0 = self._initial_guess()
        result = least_squares(
            self._bundle_residuals,
            x0,
            method="trf",
            jac="2-point",
            loss="linear",
            x_scale="jac",
            verbose=2,
            max_nfev=200,
        )
        x_opt = result.x

        cam_params = self._unpack_camera_params(x_opt)
        poses: Dict[str, np.ndarray] = {}

        for cam_idx in self.active_cams:
            R_wc, t_wc = cam_params[cam_idx]
            T_wc = np.eye(4, dtype=np.float64)
            T_wc[:3, :3] = R_wc
            T_wc[:3, 3] = t_wc

            name = self.camera_names[cam_idx]
            poses[name] = T_wc

        stats = self._compute_error_stats(x_opt)

        return poses, stats


class _MultiViewCalibrator:
    """Multi-camera calibration using a camera pose graph built from RGBD ChArUco clusters.

    This implementation does NOT optimize per-frame board poses. Instead it:

    - For each cluster, estimates board->camera transforms for all cameras
      that see the whole board.
    - For each pair of such cameras, computes a relative camera–camera
      transform T_ij (cam j -> cam i) and a quality weight.
    - Accumulates these relative constraints across all clusters.
    - Solves a global pose graph in SE(3) for camera poses only, with a
      fixed root camera (index 0).
    """

    def __init__(
            self,
            clusters: List[Cluster3D],
            board,
            camera_names: Sequence[str],
            active_cams: Sequence[int],
            min_common_views_for_edge: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        clusters : list of Cluster3D
            Multi-camera 3D ChArUco observations per frame.
        board : cv2.aruco_CharucoBoard
            The CharUco board model (used to get 3D board corner positions).
        camera_names : sequence of str
            Names of all cameras (by index).
        active_cams : sequence of int
            Indices of cameras that belong to the connected component
            containing the root; only these get calibrated.
        min_common_views_for_edge : int, optional
            Minimum number of shared clusters required before keeping an
            edge between a camera pair in the pose graph. This is used to
            prune very weak connections in large graphs.
        """
        self.clusters = clusters
        self.board = board
        self.camera_names = list(camera_names)
        self.active_cams = sorted(set(active_cams))
        self.min_common_views_for_edge = int(min_common_views_for_edge)

        self.num_cameras = len(self.camera_names)
        self.num_frames = len(self.clusters)

        # Board model points in a consistent order (by board corner index)
        self.board_points_all = np.asarray(
            self.board.getChessboardCorners(), dtype=np.float64
        )
        self.board_num_corners = int(self.board_points_all.shape[0])

        self.cam_index_map = {idx: i for i, idx in enumerate(self.active_cams)}

        # Relative pose measurements between cameras, derived from all clusters
        # Each entry: dict with keys:
        #   'i', 'j' (camera indices),
        #   'T_ij' (4x4, cam j -> cam i),
        #   'weight' (float),
        #   'frame_index' (int)
        self.measurements: List[Dict[str, object]] = []
        # Count how many clusters contributed a measurement to each camera pair
        self.edge_counts: Dict[Tuple[int, int], int] = {}

        self._build_relative_measurements()

    # --------------------------------------------------------------------- #
    # Relative measurements (2/3/4-uples per cluster -> camera–camera edges)
    # --------------------------------------------------------------------- #

    def _estimate_board_to_cam(
            self, obs: CameraObservation3D
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Estimate board->camera transform and RMS error for a single camera.

        Parameters
        ----------
        obs : CameraObservation3D
            Observed 3D board corners in camera coordinates for one camera.

        Returns
        -------
        R_cb : ndarray, shape (3, 3)
            Rotation matrix board->camera.
        t_cb : ndarray, shape (3,)
            Translation vector board->camera.
        rmse : float
            RMS alignment error in camera coordinates (same units as depth).
        """
        ids = obs.ids.astype(int)
        order = np.argsort(ids)
        ids_sorted = ids[order]
        P_c = obs.points_cam[order]  # (K, 3) in camera frame

        # Model board points for these ids
        P_b = self.board_points_all[ids_sorted]  # (K, 3) in board frame

        R_cb, t_cb = _rigid_transform_3d(P_b, P_c)
        P_c_fit = (R_cb @ P_b.T + t_cb[:, None]).T
        errs = np.linalg.norm(P_c_fit - P_c, axis=1)
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        return R_cb, t_cb, rmse

    def _build_relative_measurements(self) -> None:
        """Build pairwise relative camera pose measurements from all clusters.

        For each cluster:
        - Estimate board->camera for all active cameras that see the board.
        - For each pair (i, j) in that set, compute T_ij (cam j -> cam i) and
          append a measurement.
        """
        for frame_idx, cluster in enumerate(self.clusters):
            # Restrict to active cams present in this cluster
            cams_in_cluster = [
                cam_idx
                for cam_idx in cluster.detections.keys()
                if cam_idx in self.active_cams
            ]
            if len(cams_in_cluster) < 2:
                continue

            # Estimate board->camera for each camera in this cluster
            board_to_cam: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
            for cam_idx in cams_in_cluster:
                obs = cluster.detections[cam_idx]
                R_cb, t_cb, rmse = self._estimate_board_to_cam(obs)
                board_to_cam[cam_idx] = (R_cb, t_cb, rmse)

            cams_in_cluster = sorted(board_to_cam.keys())
            for a_idx in range(len(cams_in_cluster)):
                for b_idx in range(a_idx + 1, len(cams_in_cluster)):
                    i = cams_in_cluster[a_idx]
                    j = cams_in_cluster[b_idx]
                    R_cb_i, t_cb_i, rmse_i = board_to_cam[i]
                    R_cb_j, t_cb_j, rmse_j = board_to_cam[j]

                    # Homogeneous transforms board->camera
                    T_cb_i = np.eye(4, dtype=np.float64)
                    T_cb_i[:3, :3] = R_cb_i
                    T_cb_i[:3, 3] = t_cb_i

                    T_cb_j = np.eye(4, dtype=np.float64)
                    T_cb_j[:3, :3] = R_cb_j
                    T_cb_j[:3, 3] = t_cb_j

                    # Relative transform cam j -> cam i
                    T_ij = T_cb_i @ np.linalg.inv(T_cb_j)

                    if rmse_i > MAX_BOARD_RMSE or rmse_j > MAX_BOARD_RMSE:
                        continue

                    # Simple weight: inverse of sum of RMSEs (clipped)
                    sigma = rmse_i + rmse_j
                    weight = 1.0 / max(1e-6, sigma)

                    key = (min(i, j), max(i, j))
                    self.edge_counts[key] = self.edge_counts.get(key, 0) + 1

                    self.measurements.append(
                        {
                            "i": i,
                            "j": j,
                            "T_ij": T_ij,
                            "weight": weight,
                            "frame_index": cluster.frame_index,
                        }
                    )

        # Optionally prune very weak edges (pairs seen together too few times)
        if self.min_common_views_for_edge > 1:
            kept = []
            for m in self.measurements:
                i = m["i"]
                j = m["j"]
                key = (min(i, j), max(i, j))
                if self.edge_counts.get(key, 0) >= self.min_common_views_for_edge:
                    kept.append(m)
            self.measurements = kept

    # --------------------------------------------------------------------- #
    # Parameterization: camera->world poses, root fixed to identity
    # --------------------------------------------------------------------- #

    def _initial_guess(self) -> np.ndarray:
        """Compute initial guess for camera->world poses via graph propagation.

        Returns
        -------
        x0 : ndarray, shape (6 * (num_active_cams - 1),)
            SE(3) parameters (rvec, t) for all active cameras except the root.
        """
        root_cam_idx = self.active_cams[0]

        # Camera->world poses T_cw[k] (4x4). Root is identity.
        T_cw: Dict[int, Optional[np.ndarray]] = {
            cam_idx: None for cam_idx in self.active_cams
        }
        T_cw[root_cam_idx] = np.eye(4, dtype=np.float64)

        updated = True
        while updated:
            updated = False
            for m in self.measurements:
                i = int(m["i"])
                j = int(m["j"])
                T_ij = m["T_ij"]  # cam j -> cam i

                Ti = T_cw[i]
                Tj = T_cw[j]

                if Ti is not None and Tj is None:
                    # T_cw_j = T_cw_i * T_ij  (cam j -> world)
                    T_cw[j] = Ti @ T_ij
                    updated = True
                elif Ti is None and Tj is not None:
                    # T_cw_i = T_cw_j * inv(T_ij)
                    T_cw[i] = Tj @ np.linalg.inv(T_ij)
                    updated = True

        # Keep only cameras that ended up with a pose
        active_known = [c for c in self.active_cams if T_cw[c] is not None]
        if not active_known or active_known[0] != root_cam_idx:
            raise RuntimeError("Root camera lost or graph disconnected during initialization.")

        self.active_cams = active_known
        self.cam_index_map = {idx: i for i, idx in enumerate(self.active_cams)}

        # Pack parameters (rvec, t) for all active cams except root
        num_active_cams = len(self.active_cams)
        x0 = np.zeros(6 * (num_active_cams - 1), dtype=np.float64)

        offset = 0
        for cam_idx in self.active_cams[1:]:
            T = T_cw[cam_idx]
            if T is None:
                R_cw = np.eye(3, dtype=np.float64)
                t_cw = np.zeros(3, dtype=np.float64)
            else:
                R_cw = T[:3, :3]
                t_cw = T[:3, 3]
            rvec, _ = cv2.Rodrigues(R_cw)
            x0[offset: offset + 3] = rvec.flatten()
            x0[offset + 3: offset + 6] = t_cw
            offset += 6

        return x0

    def _unpack_camera_params(self, x: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Unpack camera->world extrinsics from parameter vector.

        Returns
        -------
        cam_params : dict
            Mapping cam_idx -> (R_cw, t_cw) for all active cameras.
        """
        cam_params: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Root camera is identity
        root_cam_idx = self.active_cams[0]
        cam_params[root_cam_idx] = (np.eye(3, dtype=np.float64), np.zeros(3))

        offset = 0
        for cam_idx in self.active_cams[1:]:
            rvec = x[offset: offset + 3]
            t_cw = x[offset + 3: offset + 6]
            R_cw, _ = cv2.Rodrigues(rvec)
            cam_params[cam_idx] = (R_cw, t_cw)
            offset += 6

        return cam_params

    # --------------------------------------------------------------------- #
    # Pose-graph residuals: SE(3) error on camera–camera edges
    # --------------------------------------------------------------------- #

    def _pose_graph_residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute residuals for the camera pose graph.

        Each measurement contributes a 6D residual:
        - 3 for rotation (axis-angle of the error)
        - 3 for translation

        The error is defined as:
            Delta = inv(T_ij_meas) @ T_ij_pred
        where:
            T_ij_meas : measured cam j -> cam i
            T_ij_pred : predicted cam j -> cam i from global poses.
        """
        cam_params = self._unpack_camera_params(x)
        residuals = []

        for m in self.measurements:
            i = int(m["i"])
            j = int(m["j"])
            T_ij_meas = m["T_ij"]
            w = float(m["weight"])

            if i not in cam_params or j not in cam_params:
                continue

            R_cw_i, t_cw_i = cam_params[i]
            R_cw_j, t_cw_j = cam_params[j]

            # Camera->world homogeneous transforms
            T_cw_i = np.eye(4, dtype=np.float64)
            T_cw_i[:3, :3] = R_cw_i
            T_cw_i[:3, 3] = t_cw_i

            T_cw_j = np.eye(4, dtype=np.float64)
            T_cw_j[:3, :3] = R_cw_j
            T_cw_j[:3, 3] = t_cw_j

            # Predicted relative transform cam j -> cam i
            T_ij_pred = np.linalg.inv(T_cw_i) @ T_cw_j

            # Error transform in cam i frame
            Delta = np.linalg.inv(T_ij_meas) @ T_ij_pred

            R_err = Delta[:3, :3]
            t_err = Delta[:3, 3]

            rvec_err, _ = cv2.Rodrigues(R_err)
            rvec_err = rvec_err.flatten()

            # Optionally scale rotation vs translation (here: simple unit scale)
            res = np.concatenate([rvec_err, 2.0 * t_err])
            residuals.append(np.sqrt(w) * res)

        if not residuals:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(residuals)

    # --------------------------------------------------------------------- #
    # Error statistics for diagnostics (edge-level SE(3) errors)
    # --------------------------------------------------------------------- #

    def _compute_error_stats(self, x_opt: np.ndarray, num_bins: int = 20) -> dict:
        """Compute global RMSE and histogram over edge translation errors.

        Here we define the "error" per measurement as the Euclidean norm of the
        translation part of the SE(3) error:
            ||t_err|| from Delta = inv(T_ij_meas) @ T_ij_pred.

        This is different from per-corner 3D error, but directly reflects how
        well the relative camera poses are matched by the final solution.
        """
        cam_params = self._unpack_camera_params(x_opt)

        trans_errors: List[float] = []
        per_edge_sq_sum: Dict[int, float] = {}
        per_edge_counts: Dict[int, int] = {}

        for idx, m in enumerate(self.measurements):
            i = int(m["i"])
            j = int(m["j"])
            T_ij_meas = m["T_ij"]

            if i not in cam_params or j not in cam_params:
                continue

            R_cw_i, t_cw_i = cam_params[i]
            R_cw_j, t_cw_j = cam_params[j]

            T_cw_i = np.eye(4, dtype=np.float64)
            T_cw_i[:3, :3] = R_cw_i
            T_cw_i[:3, 3] = t_cw_i

            T_cw_j = np.eye(4, dtype=np.float64)
            T_cw_j[:3, :3] = R_cw_j
            T_cw_j[:3, 3] = t_cw_j

            T_ij_pred = np.linalg.inv(T_cw_i) @ T_cw_j
            Delta = np.linalg.inv(T_ij_meas) @ T_ij_pred
            t_err = Delta[:3, 3]
            e = float(np.linalg.norm(t_err))

            trans_errors.append(e)
            per_edge_sq_sum[idx] = per_edge_sq_sum.get(idx, 0.0) + e ** 2
            per_edge_counts[idx] = per_edge_counts.get(idx, 0) + 1

        if not trans_errors:
            return {
                "rmse": float("nan"),
                "per_cluster_rmse": [],
                "error_histogram": {"bin_edges": [], "counts": []},
            }

        trans_errors = np.asarray(trans_errors, dtype=np.float64)
        global_rmse = float(np.sqrt(np.mean(trans_errors ** 2)))

        # Here "per_cluster_rmse" is not literally per frame anymore; we keep
        # the same key for compatibility but interpret it as per-measurement
        # index. You can adapt this to aggregate per frame if desired.
        per_cluster_rmse = []
        for idx, sq_sum in per_edge_sq_sum.items():
            n = per_edge_counts[idx]
            rmse = float(np.sqrt(sq_sum / n))
            frame_index = int(self.measurements[idx]["frame_index"])
            per_cluster_rmse.append(
                {
                    "frame_index": frame_index,
                    "rmse": rmse,
                    "num_points": 1,
                }
            )

        counts, bin_edges = np.histogram(trans_errors, bins=num_bins)
        error_histogram = {
            "bin_edges": bin_edges.astype(float).tolist(),
            "counts": counts.astype(int).tolist(),
        }

        return {
            "rmse": global_rmse,
            "per_cluster_rmse": per_cluster_rmse,
            "error_histogram": error_histogram,
        }

    def compute_pixel_rmse(
            self,
            per_cam_data: List[List[Optional[dict]]],
            cam_dirs: List[Path],
            camera_names: Sequence[str],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute 3D->2D reprojection RMSE in pixels per camera and globally.

        For each camera and each frame where we have:
        - 'corners_3d' : 3D points in that camera's coordinates
        - 'corners_2d' : detected 2D Charuco corners
        - intrinsics   : color_intri.npy under the camera parameters dir

        we:
        - project corners_3d into the image using intrinsics
        - compare to corners_2d (in pixels)
        - accumulate squared pixel errors

        Parameters
        ----------
        per_cam_data : list
            per_cam_data[cam_idx][frame_idx] is the JSON dict for that
            camera/frame, as loaded earlier in calibrate_hslu.
        cam_dirs : list of Path
            Directories containing 3D corner JSON files, one per camera.
        camera_names : sequence of str
            Names of cameras, index-aligned with cam_dirs.

        Returns
        -------
        rmse_global_px : float
            Global RMSE over all cameras and frames, in pixels.
        rmse_per_cam_px : dict
            Mapping camera_name -> RMSE in pixels (NaN if no valid measurements).
        """
        # Load intrinsics per camera
        intrinsics_by_cam: Dict[int, np.ndarray] = {}
        for cam_idx, cam_dir in enumerate(cam_dirs):
            # cam_dir is the 3D-corners dir; its parent is the camera dir (e.g. cam01)
            cam_root = cam_dir.parent
            param_dir = cam_root / "parameters"
            K_path = param_dir / "color_intri.npy"
            if not K_path.exists():
                log(f"[calibration::calibrate_hslu] Intrinsics not found for {camera_names[cam_idx]} at {K_path}, "
                    f"skipping pixel RMSE for this camera.", "warning")
                continue
            K = np.load(str(K_path))
            intrinsics_by_cam[cam_idx] = K

        per_cam_sq_err: Dict[int, float] = {i: 0.0 for i in range(len(cam_dirs))}
        per_cam_count: Dict[int, int] = {i: 0 for i in range(len(cam_dirs))}

        for cam_idx, cam_data in enumerate(per_cam_data):
            if cam_idx not in intrinsics_by_cam:
                continue
            K = intrinsics_by_cam[cam_idx]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            for frame_data in cam_data:
                if (
                        not frame_data
                        or "corners_3d" not in frame_data
                        or "corners_2d" not in frame_data
                ):
                    continue

                P_c = np.asarray(frame_data["corners_3d"], dtype=np.float64)  # (N, 3)
                uv_meas = np.asarray(frame_data["corners_2d"], dtype=np.float64)  # (N, 2)

                if P_c.shape[0] == 0 or P_c.shape[0] != uv_meas.shape[0]:
                    continue

                X = P_c[:, 0]
                Y = P_c[:, 1]
                Z = P_c[:, 2]

                valid = Z > 0.0
                if not np.any(valid):
                    continue

                X = X[valid]
                Y = Y[valid]
                Z = Z[valid]
                uv_meas_valid = uv_meas[valid]

                # Pinhole projection (no distortion, already undistorted)
                u_pred = fx * (X / Z) + cx
                v_pred = fy * (Y / Z) + cy
                uv_pred = np.stack([u_pred, v_pred], axis=1)

                err = uv_pred - uv_meas_valid
                se = float(np.sum(err ** 2))
                n = err.shape[0]

                per_cam_sq_err[cam_idx] += se
                per_cam_count[cam_idx] += n

        # Global RMSE
        global_sq = sum(per_cam_sq_err.values())
        global_n = sum(per_cam_count.values())
        if global_n > 0:
            rmse_global_px = float(np.sqrt(global_sq / global_n))
        else:
            rmse_global_px = float("nan")

        # Per-camera RMSE
        rmse_per_cam_px: Dict[str, float] = {}
        for cam_idx, name in enumerate(camera_names):
            n = per_cam_count.get(cam_idx, 0)
            if n > 0:
                rmse = float(np.sqrt(per_cam_sq_err[cam_idx] / n))
            else:
                rmse = float("nan")
            rmse_per_cam_px[name] = rmse

        return rmse_global_px, rmse_per_cam_px

    def plot_camera_poses(self, poses: Dict[str, np.ndarray], out_path: str | Path) -> None:
        """Plot calibrated camera centers and look directions.

        Parameters
        ----------
        poses : dict
            Mapping from camera name to 4x4 homogeneous transform T_wc (world-to-camera).
        out_path : str or Path
            Output path where the plot image will be saved (e.g. .png).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = Path(out_path)

        if not poses or len(poses) < 2:
            return

        # Collect camera centers in world coordinates
        labels = []
        centers = []
        for name, T in poses.items():
            T = np.asarray(T, dtype=np.float64)
            R_wc = T[:3, :3]
            t_wc = T[:3, 3]
            C_w = -R_wc.T @ t_wc  # camera center in world frame
            labels.append(name)
            centers.append(C_w)

        centers = np.vstack(centers)  # (N, 3)
        N = centers.shape[0]

        # Robust plane fit with simple RANSAC
        if N >= 3:
            rng = np.random.default_rng(0)
            best_inliers = None
            best_count = 0

            # Scale-dependent distance threshold
            bbox = centers.max(axis=0) - centers.min(axis=0)
            scale = float(np.linalg.norm(bbox)) if np.isfinite(bbox).all() else 1.0
            dist_thresh = 0.02 * max(scale, 1e-3)  # ~2% of rig size

            num_iters = min(500, N * 10)

            for _ in range(num_iters):
                idx = rng.choice(N, size=3, replace=False)
                p1, p2, p3 = centers[idx]

                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-8:
                    continue
                n /= n_norm

                d = np.abs((centers - p1) @ n)
                inliers = d < dist_thresh
                count = int(inliers.sum())
                if count > best_count:
                    best_count = count
                    best_inliers = inliers

            if best_inliers is None or best_count < 3:
                # Fall back to all points if RANSAC failed
                inliers = np.ones(N, dtype=bool)
            else:
                inliers = best_inliers
        else:
            inliers = np.ones(N, dtype=bool)

        centers_in = centers[inliers]
        if centers_in.shape[0] < 3:
            centers_in = centers

        # Refine plane with PCA on inliers
        centroid = centers_in.mean(axis=0)
        X = centers_in - centroid
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        n = Vt[-1]  # normal is last PC
        n /= np.linalg.norm(n)

        # Construct an orthonormal basis (u, v) on the plane
        # Start from world x-axis or y-axis if nearly parallel
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(axis @ n)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])

        u = axis - (axis @ n) * n
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-8:
            u = np.array([1.0, 0.0, 0.0])  # fallback
            u_norm = 1.0
        u /= u_norm

        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        # Project camera centers onto plane basis
        rel = centers - centroid  # (N, 3)
        x_coords = rel @ u  # (N,)
        y_coords = rel @ v  # (N,)

        # Plot in 2D
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x_coords, y_coords, marker="o")

        for x, y, name in zip(x_coords, y_coords, labels):
            ax.text(x, y, name, fontsize=8, ha="center", va="center")

        # Equal aspect
        xmin, xmax = float(x_coords.min()), float(x_coords.max())
        ymin, ymax = float(y_coords.min()), float(y_coords.max())
        dx = xmax - xmin
        dy = ymax - ymin
        span = max(dx, dy)
        if span <= 0:
            span = 1.0
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        ax.set_xlim(cx - 0.6 * span, cx + 0.6 * span)
        ax.set_ylim(cy - 0.6 * span, cy + 0.6 * span)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("Plane X")
        ax.set_ylabel("Plane Y")
        ax.set_title("Camera centers projected onto fitted plane")

        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    # --------------------------------------------------------------------- #
    # Public entry point
    # --------------------------------------------------------------------- #

    def calibrate(self) -> Tuple[Dict[str, np.ndarray], dict]:
        """Run pose-graph optimization and return camera poses and error stats.

        Returns
        -------
        poses : dict
            Mapping from camera name to 4x4 homogeneous transform T_wc
            (world-to-camera) for each active camera.
        stats : dict
            Error statistics (global RMSE, per-measurement RMSE, histogram).
        """
        if not self.clusters or not self.measurements:
            raise RuntimeError("No clusters or relative measurements available for calibration.")

        x0 = self._initial_guess()
        result = least_squares(
            self._pose_graph_residuals,
            x0,
            method="trf",
            jac="2-point",
            loss="linear",
            x_scale="jac",
            verbose=2,
            max_nfev=200,
        )
        x_opt = result.x

        cam_params = self._unpack_camera_params(x_opt)
        poses: Dict[str, np.ndarray] = {}

        for cam_idx in self.active_cams:
            R_cw, t_cw = cam_params[cam_idx]

            # T_cw: camera->world
            T_cw = np.eye(4, dtype=np.float64)
            T_cw[:3, :3] = R_cw
            T_cw[:3, 3] = t_cw

            # We return world->camera (T_wc) for downstream consistency
            T_wc = np.linalg.inv(T_cw)

            name = self.camera_names[cam_idx]
            poses[name] = T_wc

        stats = self._compute_error_stats(x_opt)
        return poses, stats


def _visualize_camera_graph(
        adjacency: Dict[int, set],
        edge_weights: Dict[Tuple[int, int], int],
        camera_names: Sequence[str],
        out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import networkx as nx
            use_nx = True
        except ImportError:
            use_nx = False
    except ImportError:
        log("[calibration::calibrate_hslu] Matplotlib not available, skipping graph plot.", "warning")
        return

    if not adjacency:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    if use_nx:
        G = nx.Graph()
        for cam_idx in range(len(camera_names)):
            G.add_node(cam_idx, label=camera_names[cam_idx])

        for (a, b), w in edge_weights.items():
            G.add_edge(a, b, weight=w)

        pos = nx.spring_layout(G, seed=0)
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, width=[0.5 + 0.1 * w for w in weights], ax=ax)
        nx.draw_networkx_labels(
            G,
            pos,
            labels={i: camera_names[i] for i in range(len(camera_names))},
            font_size=8,
            ax=ax,
        )
    else:
        # Simple circular layout fallback
        import math
        n = len(camera_names)
        angles = [2 * math.pi * i / n for i in range(n)]
        coords = {
            i: (math.cos(ang), math.sin(ang))
            for i, ang in enumerate(angles)
        }

        for (a, b), w in edge_weights.items():
            xa, ya = coords[a]
            xb, yb = coords[b]
            ax.plot([xa, xb], [ya, yb], "k-", linewidth=0.5 + 0.1 * w)

        for i, name in enumerate(camera_names):
            x, y = coords[i]
            ax.scatter([x], [y], s=100)
            ax.text(x, y, name, fontsize=8, ha="center", va="center")

        ax.axis("equal")
        ax.axis("off")

    ax.set_title("Camera Co-visibility Graph")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log(f"[calibration::calibrate_hslu] Camera graph plot saved to {out_path}", "debug")


def _write_open3d_pose_graph(
        adjacency: Dict[int, set],
        edge_weights: Dict[Tuple[int, int], int],
        camera_names: Sequence[str],
        out_path: Path,
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        log("[calibration::calibrate_hslu] Open3D not available, skipping pose graph export.", "debug")
        return

    try:
        if hasattr(o3d, "pipelines") and hasattr(o3d.pipelines, "registration"):
            reg = o3d.pipelines.registration
        elif hasattr(o3d, "registration"):
            reg = o3d.registration
        else:
            log("[calibration::calibrate_hslu] Open3D registration module not found, skipping pose graph export.", "debug")
            return

        PoseGraph = reg.PoseGraph
        PoseGraphNode = reg.PoseGraphNode
        PoseGraphEdge = reg.PoseGraphEdge
    except Exception:
        log("[calibration::calibrate_hslu] Failed to access Open3D PoseGraph API, skipping pose graph export.", "debug")
        return

    pose_graph = PoseGraph()

    # Add one node per camera with identity pose (placeholder)
    for cam_idx in range(len(camera_names)):
        T = np.eye(4)
        pose_graph.nodes.append(PoseGraphNode(T))

    # Add edges with identity relative pose but information scaled by co-visibility count
    for (a, b), w in edge_weights.items():
        T_ab = np.eye(4)
        info = np.eye(6) * float(max(1, w))
        pose_graph.edges.append(
            PoseGraphEdge(int(a), int(b), T_ab, info, uncertain=True)
        )

    try:
        o3d.io.write_pose_graph(str(out_path), pose_graph)
        log(f"[calibration::calibrate_hslu] Open3D pose graph written to {out_path}", "debug")
    except Exception as e:
        log(f"[calibration::calibrate_hslu] Failed to write Open3D pose graph: {e}", "warning")


@app.task(name="calibration.calibrate_hslu", base=AutoRetryTask)
def calibrate_hslu(
        *all_corner_3d_dirs: str,
        out_dir: str,
        min_cams_per_cluster: int = 2,
        min_clusters: int = 20,
        caliscope_toml_path: Optional[str] = None,
        force: bool = False,
        rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None,
):
    """Perform multi-camera calibration from RGBD ChArUco 3D corners.

    This task consumes, for each camera, a directory of 3D corner JSON files
    produced by `lift_corners_3d`. It:

    - Reads the JSON files in each directory in sync (sorted by stem).
    - For each frame, builds a multi-view cluster of cameras that see the board
      with all corners and have 3D points.
    - Builds a camera co-visibility graph where nodes are cameras and edges
      connect cameras that jointly observe the board in at least one cluster.
    - Visualizes this camera graph and (if Open3D is available) writes an
      Open3D pose graph representation.
    - Keeps only cameras in the connected component of the root camera (index 0).
    - Runs global bundle adjustment using `_MultiViewCalibrator`, with BA
      residuals coming only from clusters where at least
      `min_cams_per_cluster` cameras are present (but all clusters with >=2
      cams are used to propagate initial poses).
    - Outputs camera poses, RMSE statistics, histogram of 3D errors, a camera
      pose plot, and a camera graph plot.

    Parameters
    ----------
    *all_corner_3d_dirs : str
        Directories containing 3D corner JSON files, one per camera.
        Each directory is assumed to contain a time-synchronized sequence of
        JSON files produced by `lift_corners_3d`.
    out_dir : str
        Output directory for calibration results.
    min_cams_per_cluster : int, optional
        Minimum number of cameras required in a cluster to contribute residuals
        to bundle adjustment. Clusters with only 2 cameras are still used to
        build the camera graph and propagate initial poses, but they do not
        contribute residuals unless this threshold is <= 2.
    min_clusters : int, optional
        Minimum number of valid BA clusters (with at least
        `min_cams_per_cluster` cameras) required for calibration.
    force : bool, optional
        If True, override an existing calibration.json in out_dir. If False and
        calibration.json exists and is readable, it will be returned and no
        calibration is performed.

    Returns
    -------
    dict
        A dictionary with:
        - "poses": mapping from camera name (directory basename) to 4x4
          homogeneous transform T_wc (world-to-camera).
        - "rmse": global RMSE over all 3D residuals used in BA.
        - "per_cluster_rmse": list of per-cluster RMSE dictionaries.
        - "error_histogram": histogram of per-point 3D errors.
        - "camera_pose_plot": path to the saved camera pose plot image.
        - "camera_graph_plot": path to the saved camera graph plot image.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    calib_path = out_dir_path / "calibration.json"

    # If calibration exists and force is False, return the existing result
    if calib_path.exists() and not force:
        try:
            with open(calib_path, "r") as f:
                existing = json.load(f)
            log(f"[calibration::calibrate_hslu] Using existing calibration at {calib_path}", "debug")
            return existing
        except json.JSONDecodeError:
            log(f"[calibration::calibrate_hslu] Existing calibration.json at {calib_path} is invalid. Recomputing.", "warning")
    if len(all_corner_3d_dirs) < 2:
        raise ValueError("At least two cameras are required for calibration.")

    cam_dirs = [Path(d) for d in all_corner_3d_dirs]

    # Use parent directory name as camera name (e.g. cam01)
    camera_names = [p.parent.name for p in cam_dirs]
    num_cams = len(cam_dirs)
    log(f"[calibration::calibrate_hslu] Starting calibration with {num_cams} cameras.", "debug")

    # Load and align 3D JSON files per camera
    per_cam_files: List[List[Path]] = []
    per_cam_data: List[List[Optional[dict]]] = []
    for cam_dir in cam_dirs:
        files = sorted(cam_dir.glob("*.json"), key=lambda p: p.stem)
        if not files:
            raise RuntimeError(f"No 3D corner JSON files found in {cam_dir}")
        per_cam_files.append(files)
        per_cam_data.append([None] * len(files))
    num_frames_set = {len(files) for files in per_cam_files}
    if len(num_frames_set) != 1:
        raise RuntimeError(
            f"All cameras must have the same number of frames, got {num_frames_set}"
        )
    num_frames = next(iter(num_frames_set))
    log(f"[calibration::calibrate_hslu] Found {num_frames} synchronized frames per camera.", "debug")

    # Read JSON data
    for cam_idx, files in enumerate(per_cam_files):
        for i, path in enumerate(files):
            with open(path, "r") as f:
                try:
                    per_cam_data[cam_idx][i] = json.load(f)
                except json.JSONDecodeError:
                    per_cam_data[cam_idx][i] = {}

    # Recover board configuration from first non-empty file
    board_config: Optional[dict] = None
    for cam_data in per_cam_data:
        for data in cam_data:
            if data and "board" in data:
                board_config = data["board"]
                break
        if board_config is not None:
            break
    if board_config is None:
        raise RuntimeError("Could not find board configuration in any 3D JSON file.")
    squaresX = int(board_config["squaresX"])
    squaresY = int(board_config["squaresY"])
    squareLength = float(board_config["squareLength"])
    markerLength = float(board_config["markerLength"])
    dictionary_id = int(board_config["dictionary"])
    log(f"[calibration::calibrate_hslu] Board configuration: {squaresX}x{squaresY}, "
        f"squareLength={squareLength}, markerLength={markerLength}, dict={dictionary_id}", "debug")
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        size=[squaresX, squaresY],
        squareLength=squareLength,
        markerLength=markerLength,
        dictionary=aruco_dict,
    )
    board_points_all = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    board_num_corners = int(board_points_all.shape[0])

    # Build clusters and camera adjacency graph
    all_clusters: List[Cluster3D] = []  # clusters with >= 2 cameras (graph clusters)
    adjacency: Dict[int, set] = {i: set() for i in range(num_cams)}
    edge_weights: Dict[Tuple[int, int], int] = {}  # (a,b) -> number of shared clusters (co-visibility)
    ba_candidate_indices: List[int] = []  # indices in all_clusters usable for BA
    for frame_idx in range(num_frames):
        detections: Dict[int, CameraObservation3D] = {}
        for cam_idx in range(num_cams):
            data = per_cam_data[cam_idx][frame_idx]
            if not data or "ids" not in data or "corners_3d" not in data:
                continue

            ids = np.asarray(data["ids"], dtype=int)
            pts3d = np.asarray(data["corners_3d"], dtype=np.float64)

            if ids.size != board_num_corners or pts3d.shape[0] != board_num_corners:
                continue
            if not np.array_equal(np.sort(ids), np.arange(board_num_corners)):
                continue

            detections[cam_idx] = CameraObservation3D(ids=ids, points_cam=pts3d)

        if len(detections) == 0:
            continue

        # Graph clusters: frames with at least 2 cameras
        if len(detections) >= 2:
            all_clusters.append(Cluster3D(frame_index=frame_idx, detections=detections))
            cams_in_cluster = sorted(detections.keys())
            # Update adjacency and edge weights
            for i in range(len(cams_in_cluster)):
                for j in range(i + 1, len(cams_in_cluster)):
                    a = cams_in_cluster[i]
                    b = cams_in_cluster[j]
                    adjacency[a].add(b)
                    adjacency[b].add(a)
                    key = (min(a, b), max(a, b))
                    edge_weights[key] = edge_weights.get(key, 0) + 1

            if len(detections) >= int(min_cams_per_cluster):
                # Candidate for BA (final decision later after active camera filtering)
                ba_candidate_indices.append(len(all_clusters) - 1)

    if not all_clusters:
        raise RuntimeError("No clusters with at least 2 cameras; camera graph is empty.")
    log(f"[calibration::calibrate_hslu] Built {len(all_clusters)} graph clusters ({len(ba_candidate_indices)} candidates with >= {min_cams_per_cluster} cams).", "debug")
    # Log adjacency / degrees
    degrees = {cam_idx: len(neigh) for cam_idx, neigh in adjacency.items()}
    degree_str = "\n\t\t ".join(
        f"{camera_names[i]}: deg={deg}" for i, deg in sorted(degrees.items())
    )
    log(f"[calibration::calibrate_hslu] Camera graph degrees: {degree_str}", "debug")

    # # Visualize camera graph (2D)
    # graph_plot_path = out_dir_path / "camera_graph.png"
    # _visualize_camera_graph(adjacency, edge_weights, camera_names, graph_plot_path)
    # pose_graph_path = out_dir_path / "camera_graph_open3d.json"
    # _write_open3d_pose_graph(adjacency, edge_weights, camera_names, pose_graph_path)

    # Determine connected component containing root camera (index 0)
    root_cam = 0
    visited = set()
    stack = [root_cam]
    while stack:
        c = stack.pop()
        if c in visited:
            continue
        visited.add(c)
        stack.extend(adjacency[c] - visited)
    active_cams = sorted(visited)
    if len(active_cams) < 1:
        raise RuntimeError("Root camera is not connected to any other camera.")
    active_names = [camera_names[i] for i in active_cams]
    log(f"[calibration::calibrate_hslu] Active cameras in root component ({len(active_cams)}): {', '.join(active_names)}", "debug")

    # Filter clusters to those involving at least one active camera
    filtered_clusters: List[Cluster3D] = []
    for cl in all_clusters:
        active_dets = {
            cam_idx: det
            for cam_idx, det in cl.detections.items()
            if cam_idx in active_cams
        }
        if active_dets:
            filtered_clusters.append(
                Cluster3D(frame_index=cl.frame_index, detections=active_dets)
            )

    # Check how many filtered clusters are usable for BA
    ba_filtered_indices: List[int] = []
    for idx, cl in enumerate(filtered_clusters):
        if len(cl.detections) >= int(min_cams_per_cluster):
            ba_filtered_indices.append(idx)

    if len(ba_filtered_indices) < int(min_clusters):
        raise RuntimeError(
            f"Not enough valid BA clusters after filtering active cameras: "
            f"{len(ba_filtered_indices)} < {min_clusters}"
        )

    log(f"[calibration::calibrate_hslu] Using {len(filtered_clusters)} filtered clusters "
        f"({len(ba_filtered_indices)} with >= {min_cams_per_cluster} cams) for calibration.", "debug")

    # Cap the number of BA clusters inside the calibrator to keep the problem size manageable
    # (still uses all clusters for graph-based pose propagation)
    # max_ba_clusters = max(min_clusters, 200)
    # exit(0)

    calibrator = _MultiViewCalibrator(
        clusters=filtered_clusters,
        board=board,
        camera_names=camera_names,
        active_cams=active_cams,
        min_common_views_for_edge=10,
    )
    poses, stats = calibrator.calibrate()

    if rotate is not None:
        # TODO: implement the rotation of the camera poses to account
        pass

    poses_serializable = {name: pose.tolist() for name, pose in poses.items()}
    log(
        f"[calibration::calibrate_hslu] Pose-graph translation RMSE: "
        f"{stats['rmse']:.4f} m",
        "debug",
    )
    # 3D->2D reprojection RMSE in pixels (per camera and global)
    rmse_px_global, rmse_px_per_cam = calibrator.compute_pixel_rmse(
        per_cam_data=per_cam_data,
        cam_dirs=cam_dirs,
        camera_names=camera_names,
    )
    log(
        f"[calibration::calibrate_hslu] 3D->2D reprojection RMSE (global): "
        f"{rmse_px_global:.2f} px",
        "debug",
    )
    for cam_name, v in rmse_px_per_cam.items():
        log(
            f"[calibration::calibrate_hslu] 3D->2D reprojection RMSE for {cam_name}: "
            f"{v:.2f} px",
            "debug",
        )

    # Plot camera poses (in 3D)
    pose_plot_path = out_dir_path / "camera_poses.png"
    calibrator.plot_camera_poses(poses, pose_plot_path)

    # Save calibration result
    result = {
        "poses": poses_serializable,
        "rmse": stats["rmse"],  # translation RMSE in meters (pose-graph)
        "per_cluster_rmse": stats["per_cluster_rmse"],
        "error_histogram": stats["error_histogram"],
        "camera_pose_plot": str(pose_plot_path),
        "rmse_pixels_global": rmse_px_global,
        "rmse_pixels_per_camera": rmse_px_per_cam,
    }
    with open(calib_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[calibration::calibrate_hslu] Calibration completed and saved to {calib_path}", "debug")

    if caliscope_toml_path is not None:
        cfg_path = Path(caliscope_toml_path)
        if not cfg_path.exists():
            log(
                f"[calibration::calibrate_hslu] Caliscope config TOML not found at "
                f"{cfg_path}, skipping override.",
                "warning",
            )
        else:
            try:
                cfg = toml.load(str(cfg_path))
            except Exception as e:
                log(
                    f"[calibration::calibrate_hslu] Failed to load Caliscope TOML at "
                    f"{cfg_path}: {e}. Skipping override.",
                    "warning",
                )
                cfg = None

            if cfg is not None:
                for cam_name, T_wc in poses.items():
                    # cam_name is e.g. "cam01" -> section "cam_1"
                    digits = "".join(ch for ch in cam_name if ch.isdigit())
                    if not digits:
                        log(
                            f"[calibration::calibrate_hslu] Could not parse index from "
                            f"camera name '{cam_name}', skipping.",
                            "warning",
                        )
                        continue
                    cam_idx = int(digits)
                    section = f"cam_{cam_idx}"

                    if section not in cfg:
                        log(
                            f"[calibration::calibrate_hslu] Section '{section}' not "
                            f"found in Caliscope config, skipping.",
                            "warning",
                        )
                        continue

                    T_wc = np.asarray(T_wc, dtype=np.float64)
                    # Convert world->camera to camera->world for config
                    # T_cw = np.linalg.inv(T_wc)
                    T_cw = T_wc
                    R_cw = T_cw[:3, :3]
                    t_cw = T_cw[:3, 3]

                    rvec, _ = cv2.Rodrigues(R_cw)
                    rvec = rvec.flatten()

                    cfg[section]["translation"] = t_cw.astype(float).tolist()
                    cfg[section]["rotation"] = rvec.astype(float).tolist()

                try:
                    with open(cfg_path, "w") as f:
                        toml.dump(cfg, f)
                    log(
                        f"[calibration::calibrate_hslu] Updated Caliscope config "
                        f"extrinsics at {cfg_path}",
                        "debug",
                    )
                except Exception as e:
                    log(
                        f"[calibration::calibrate_hslu] Failed to write Caliscope "
                        f"config TOML at {cfg_path}: {e}",
                        "warning",
                    )

    return result


@app.task(name="calibration.generate_caliscope_config", base=AutoRetryTask)
def generate_caliscope_config(capturestudio_cache_root: str, rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None):
    """
    Generate a configuration file for Caliscope-based calibration from a Capture Studio session directory.

    Parameters
    ----------
    capturestudio_cache_root: str
        Path to the session folder containing camera directories, e.g. "/root/CAPTURESTUDIO_CACHE/Vlad_1_Calib_1".
    rotate: Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180'], optional
        If provided, the intrinsic parameters will be rotated accordingly. `rotate` is the rotation needed to *unrotate* the inputs so that the (physical) floor is at the bottom of the image and the ceiling at the top.
    """
    # Define output directory for caliscope-based calibration
    capturestudio_cache_root = Path(capturestudio_cache_root)
    caliscope_dir = capturestudio_cache_root / '__calib__' / 'caliscope'
    caliscope_dir.mkdir(parents=True, exist_ok=True)

    # 1) camera_array.toml
    cam_array_file = caliscope_dir / 'camera_array.toml'
    if not cam_array_file.exists():
        cam_array_config = {}

        # Load camera profiles
        cam_count = 0
        for cam_dir in sorted(capturestudio_cache_root.glob('orbbec/cam*'), key=lambda x: int(x.name.replace('orbbec/cam', '').replace('cam', ''))):
            if not cam_dir.is_dir():
                continue
            cam_idx_s1 = int(cam_dir.name.replace('orbbec/cam', '').replace('cam', ''))
            color_dist = np.load(str(cam_dir / 'parameters' / 'color_dist.npy'))
            color_intri = np.load(str(cam_dir / 'parameters' / 'color_intri.npy'))
            first_frame = sorted(cam_dir.glob('color/*.jpg'), key=lambda x: int(x.stem))[0]
            first_frame_size_hw = tuple(cv2.imread(str(first_frame), cv2.IMREAD_UNCHANGED).shape[:2])
            if rotate is not None:
                height, width = first_frame_size_hw  # (H, W)

                # Update the intrinsic parameters (upper-triangular form) for the rotation applied to the images.
                # Note: This keeps K upper-triangular; resulting 3D will generally be rotated about camera Z accordingly.
                color_intri = color_intri.astype(np.float64, copy=False)

                fx = float(color_intri[0, 0])
                fy = float(color_intri[1, 1])
                cx = float(color_intri[0, 2])
                cy = float(color_intri[1, 2])

                if rotate == "90_CLOCKWISE":
                    # Image mapping: u' = (H-1) - v ; v' = u
                    # Upper-triangular K' parameters:
                    # fx' = fy, fy' = fx
                    # cx' = (H-1) - cy, cy' = cx
                    fx_p = fy
                    fy_p = fx
                    cx_p = (height - 1.0) - cy
                    cy_p = cx
                    new_w, new_h = height, width

                elif rotate == "90_COUNTERCLOCKWISE":
                    # Image mapping: u' = v ; v' = (W-1) - u
                    # Upper-triangular K' parameters:
                    # fx' = fy, fy' = fx
                    # cx' = cy, cy' = (W-1) - cx
                    fx_p = fy
                    fy_p = fx
                    cx_p = cy
                    cy_p = (width - 1.0) - cx
                    new_w, new_h = height, width

                elif rotate == "180":
                    # Image mapping: u' = (W-1) - u ; v' = (H-1) - v
                    # Upper-triangular K' parameters:
                    # fx' = fx, fy' = fy
                    # cx' = (W-1) - cx, cy' = (H-1) - cy
                    fx_p = fx
                    fy_p = fy
                    cx_p = (width - 1.0) - cx
                    cy_p = (height - 1.0) - cy
                    new_w, new_h = width, height

                else:
                    raise ValueError(f"Unsupported rotate value: {rotate!r}")
                color_intri = np.array(
                    [[fx_p, 0.0, cx_p],
                     [0.0, fy_p, cy_p],
                     [0.0, 0.0, 1.0]],
                    dtype=np.float32
                )
                first_frame_size_hw = (new_h, new_w)

            cam_array_config[f'cameras.{cam_idx_s1}'] = dict(
                cam_id=cam_idx_s1,
                physical_index=cam_idx_s1 - 1,
                rotation_count=0,
                error=0.01,
                grid_count=20,
                size=(first_frame_size_hw[1], first_frame_size_hw[0]),  # width, height
                matrix=color_intri.tolist(),  # 3x3 list
                distortions=color_dist.tolist()[:5],  # k1, k2, p1, p2, k3
            )
            cam_count += 1

        with open(cam_array_file, mode="w") as f:
            toml.dump(cam_array_config, f)
        log(f"[calibration::generate_caliscope_config] Generated camera config file at {cam_array_file}", 'debug')

    # 2) Targets / intrinsic_charuco.toml
    Path(caliscope_dir / 'calibration' / 'targets').mkdir(parents=True, exist_ok=True)
    if not (caliscope_dir / 'calibration' / 'targets' / 'intrinsic_charuco.toml').is_file():
        # Load charuco profile
        if (capturestudio_cache_root / 'orbbec' / 'session_metadata.json').exists():
            with open(capturestudio_cache_root / 'orbbec' / 'session_metadata.json', 'r') as fp:
                calibration_pattern = json.load(fp).get('calibration_pattern', 'charuco_6x4_a2')
        else:
            log('[calibration.generate_caliscope_config] session_metadata.json not found, using default charuco_6x4_a2', 'warning')
            calibration_pattern = 'charuco_6x4_a2'
        with open(PathUtils.resources_path() / 'calibration_patterns' / calibration_pattern / 'charuco_info.json', 'r') as fp:
            charuco_profile = json.load(fp)

        # Generate cam_array_config file using intrinsic and distortion parameters
        with open(caliscope_dir / 'calibration' / 'targets' / 'intrinsic_charuco.toml', 'w') as fp:
            toml.dump(charuco_profile, fp)
        log(f"[calibration.generate_caliscope_config] Charuco configuration file generated to {caliscope_dir / 'calibration' / 'targets' / 'intrinsic_charuco.toml'}", 'debug')

    # 3) Targets / config.toml
    if not (caliscope_dir / 'calibration' / 'targets' / 'config.toml').is_file():
        targets_config = dict(
            intrinsic_target_type="charuco",
            extrinsic_target_type="charuco",
            extrinsic_charuco_same_as_intrinsic=True
        )
        with open(caliscope_dir / 'calibration' / 'targets' / 'config.toml', 'w') as f:
            toml.dump(targets_config, f)
        log(f"[calibration.generate_caliscope_config] targets/config.toml generated to {caliscope_dir / 'calibration' / 'targets' / 'config.toml'}", 'debug')

    return True


@app.task(name="calibration.generate_caliscope_videos", base=AutoRetryTask)
def generate_caliscope_videos(capturestudio_cache_root: str, cam_name: str, start_offset: int = 0, total_frames: int = -1, fps: int = 30, rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None):
    """
    Generate videos for Caliscope-based calibration from a Capture Studio session directory.
    ATTN: Assumes that the frames have already been synchronized.

    Parameters
    ----------
    capturestudio_cache_root: str
        Path to the session folder containing camera directories, e.g. "/root/CAPTURESTUDIO_CACHE/Vlad_1_Calib_1".
    cam_name : str
        The camera dir name, e.g. "cam01".
    start_offset : int
        The starting index of the color frames to process. Default is 0.
    total_frames : int
        The total number of color frames to process. If -1, all frames from the start offset will be processed.
    fps : int
        The frame rate for the output videos. Default is 30.
    rotate: Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180'], optional
        If provided, the videos will be rotated accordingly.
    """
    # Define output directory for caliscope-based calibration
    capturestudio_cache_root = Path(capturestudio_cache_root)
    caliscope_dir = capturestudio_cache_root / '__calib__' / 'caliscope'
    caliscope_dir.mkdir(parents=True, exist_ok=True)
    extrinsic_videos_dir = caliscope_dir / 'calibration' / 'extrinsic'
    extrinsic_videos_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_videos_dir = caliscope_dir / 'calibration' / 'intrinsic'
    intrinsic_videos_dir.mkdir(parents=True, exist_ok=True)

    # 2) Generate caliscope videos
    cam_dir = capturestudio_cache_root / 'orbbec' / cam_name
    cam_idx_s1 = int(cam_dir.name.replace('orbbec/cam', '').replace('cam', ''))
    cam_color_dir = cam_dir / 'color'
    assert cam_color_dir.exists() and cam_color_dir.is_dir(), f"Color directory {cam_color_dir} does not exist"
    if total_frames < 0:
        total_frames = len(list(cam_color_dir.glob('*.jpg'))) - start_offset
    video_path = extrinsic_videos_dir / f'cam_{cam_idx_s1}.mp4'
    if video_path.exists() and PathUtils.verify_file(video_path):
        log(f"[calibration.generate_caliscope_videos] \tCaliscope video already exists at {video_path}. Skipping video generation.", 'debug')
        return True

    # Create video
    from preprocessing.generate_video import frames_to_video
    written_video_path = frames_to_video(cam_color_dir, start_offset=start_offset, total_frames=total_frames, fps=fps, rotate=rotate)
    # move to extrinsic_videos_dir
    shutil.move(written_video_path, video_path)
    # copy to intrinsic_videos_dir
    shutil.copy(extrinsic_videos_dir / f'cam_{cam_idx_s1}.mp4', intrinsic_videos_dir / f'cam_{cam_idx_s1}.mp4')
    return True
