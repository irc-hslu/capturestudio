import abc
from pathlib import Path
from typing import List, Literal, Union, Iterator, Tuple, Iterable, Dict, Sequence, Optional

import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from pytorch3d.renderer import look_at_view_transform
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation as ScipyRotation, Slerp

from utils.misc import log
from utils.vis import VisUtils


class CameraOrbit(metaclass=abc.ABCMeta):
    def __init__(self, gt_intrinsics: np.ndarray, gt_extrinsics_w2c: np.ndarray, gt_image_size_hw: Tuple[int, int], trajectory_idx: Union[Literal['all'], Sequence[int]] = 'all', reconstruction_idx: Union[Literal['all'], Sequence[int]] = 'all', num_points: int = 1_000, **trajectory_kwargs):
        if trajectory_idx == 'all':
            trajectory_idx = list(range(gt_extrinsics_w2c.shape[0]))
        else:
            trajectory_idx = [int(_) - 1 for _ in trajectory_idx]
        if reconstruction_idx == 'all':
            reconstruction_idx = list(range(len(trajectory_idx)))
        else:
            reconstruction_idx = [int(_) - 1 for _ in reconstruction_idx]
        self.gt_intrinsics = gt_intrinsics
        self.gt_extrinsics_w2c = gt_extrinsics_w2c
        self.gt_image_size_hw = gt_image_size_hw
        self.gt_extrinsics_c2w = np.linalg.inv(gt_extrinsics_w2c)
        self.gt_cam_centers = self.gt_extrinsics_c2w[:, :3, 3]
        self.gt_cam_idx_s0 = reconstruction_idx
        self.num_points = num_points

        # Create virtual cameras along the trajectory
        virtual_intrinsics, virtual_extrinsics_w2c, mean_lookat_point, mean_up_vector, assignment_closest, assignment_middle = self.create_virtual_cameras(
            gt_intrinsics=gt_intrinsics[trajectory_idx],
            gt_extrinsics_c2w=self.gt_extrinsics_c2w[trajectory_idx],
            assignment_idx=[trajectory_idx.index(_) for _ in reconstruction_idx],
            **trajectory_kwargs
        )
        assignment_closest = {trajectory_idx[k]: v for k, v in assignment_closest.items()}
        assignment_middle = {trajectory_idx[k]: v for k, v in assignment_middle.items()}

        # Store the virtual camera parameters
        self.virtual_intrinsic = virtual_intrinsics
        self.virtual_extrinsic_w2c = virtual_extrinsics_w2c
        self.virtual_extrinsic_c2w = np.linalg.inv(virtual_extrinsics_w2c)
        self.mean_look_at_point = mean_lookat_point
        self.mean_up_vector = mean_up_vector

        # Create interpolator functions for rotation and translation
        self._rot_interp = self.__class__.create_rotation_interpolator(self.virtual_extrinsic_c2w[:, :3, :3])
        self._t_interp = self.__class__.create_translation_interpolator(self.virtual_extrinsic_c2w[:, :3, 3])
        self._inv_assignment_closest = self.__class__.build_virtual_to_gt_mapping(assignment_closest, reconstruction_idx)
        self._inv_assignment_middle = self.__class__.build_virtual_to_gt_mapping(assignment_middle, reconstruction_idx)

    @abc.abstractmethod
    def create_virtual_cameras(self, gt_intrinsics: np.ndarray, gt_extrinsics_c2w: np.ndarray, assignment_idx: List[int], **trajectory_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Iterable[int]], Dict[int, Iterable[int]]]:
        """
        Create virtual cameras along the trajectory defined by the ground truth extrinsics.
        This method computes the mean look-at point and up vector, and assigns virtual cameras
        to the ground truth cameras.

        Parameters
        ----------
        gt_intrinsics : np.ndarray
            Ground truth camera intrinsic matrix of shape (N, 3, 3).
        gt_extrinsics_c2w : np.ndarray
            Ground truth camera extrinsics in c2w format of shape (N, 4, 4).
        assignment_idx : List[int]
            Indices of the ground truth cameras to be used during the virtual-to-gt assignment process.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Iterable[int]], Dict[int, Iterable[int]]]
            A tuple containing:
            - virtual_intrinsics: The intrinsic matrix for the virtual cameras, shape (M, 3, 3).
            - virtual_extrinsics_w2c: The extrinsics of the virtual cameras in w2c format, shape (M, 4, 4).
            - mean_lookat_point: The mean look-at point in world coordinates, shape (3,).
            - mean_up_vector: The mean up vector in world coordinates, shape (3,).
            - assignment_closest: A dictionary mapping virtual camera indices to the closest ground truth camera indices.
            - assignment_middle: A dictionary mapping virtual camera indices to the middle ground truth camera indices.
        """
        raise NotImplementedError

    def traverse(self, velocity: float, loop: bool = False, data_fps: float = 30, debug_mode: bool = False) -> Iterator[Tuple[np.ndarray, np.ndarray, int, int]]:
        """
        Traverse through the virtual cameras and yield the intrinsic, extrinsic_c2w matrices, the closest, and the middle assignments.

        Parameters
        ----------
        velocity : float
            The speed of traversal in meters per second. This is the distance moved per frame.
        loop : bool
            If True, the traversal will loop back to the start after reaching the end. If False, it will swap direction and move back to the start.
        data_fps : float
            The frames per second of the data, used to calculate the time step for traversal.
        debug_mode : bool
            If True, debug information will be returned instead of the intrinsic and extrinsic matrices.

        Yields
        ------
        Iterator[Tuple[np.ndarray, np.ndarray]]
            A tuple containing the rotation matrix and translation vector for each virtual camera.
        """
        velocity_meters_per_frame = velocity / data_fps  # Convert velocity to distance per frame
        num_samples = self.num_points
        t_samples = np.linspace(0, 1, num_samples)
        segment_lengths = np.linalg.norm(np.diff(self.virtual_extrinsic_c2w[:, :3, 3], axis=0), axis=1)
        arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = arc_lengths[-1]
        log(f'Traversing trajectory of {total_length:.2f} meters at {velocity:.2f} m/s', 'debug')
        s_to_t = interp1d(arc_lengths / total_length, t_samples, kind='linear', fill_value="extrapolate")
        s = 0.0  # length along the trajectory in meters
        direction = 1
        while True:
            s_norm = s / total_length
            s_norm = np.clip(s_norm, 0, 1)
            t = s_to_t(s_norm)
            closest_virtual_idx = min(int(t * self.num_points), self.num_points - 1)

            # virtual camera extrinsic in c2w format
            t_extrinsic_c2w = np.eye(4)
            t_extrinsic_c2w[:3, :3] = self._rot_interp(t)
            t_extrinsic_c2w[:3, 3] = self._t_interp(t)

            # virtual camera intrinsic
            t_intrinsic = np.asarray(self.virtual_intrinsic[closest_virtual_idx])  # mean over all gt cameras

            # assignments
            gt_idx_closest = self._inv_assignment_closest[closest_virtual_idx]
            gt_idx_middle = self._inv_assignment_middle[closest_virtual_idx] - 1  # FIX: -1 as we need to return the first camera index of the stereo pair

            if debug_mode:
                yield t, s, gt_idx_closest, gt_idx_middle
            else:
                yield t_intrinsic, t_extrinsic_c2w, gt_idx_closest, gt_idx_middle

            s += direction * velocity_meters_per_frame  # velocity is distance per frame
            if s >= total_length:
                if loop:
                    s = 0
                else:
                    s = total_length
                    direction = -1
            elif s <= 0:
                if loop:
                    s = total_length
                else:
                    s = 0
                    direction = 1

    def export_poses(self, scale: float = 0.4, output_path: Union[Path, str] = 'camera_poses.png') -> None:
        """
        Exports a sequence of camera poses to a PLY file as Open3D coordinate frame meshes.

        Parameters
        ----------
        scale : float
            Scale of each coordinate frame (default: 0.1).
        output_path : Union[Path, str]
            Path to save the rendered image of the camera poses (default: 'camera_poses.png').
        """
        # Merge all frames
        scene = o3d.geometry.TriangleMesh()
        for c2w in self.gt_extrinsics_c2w:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=c2w[:3, 3])
            frame.rotate(c2w[:3, :3])
            scene += frame

        # Create a pink sphere at the look-at point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.5)
        sphere.paint_uniform_color([1.0, 0.2, 0.8])  # pink
        sphere.translate(self.mean_look_at_point)
        sphere.compute_vertex_normals()
        scene += sphere

        # Uniformly select 20 virtual cameras for visualization as orange spheres
        cmap = plt.cm.get_cmap('tab20')
        n_gt = len(self.gt_extrinsics_c2w)
        color_indices = np.arange(n_gt) % 20
        gt_colors = cmap(color_indices / 19.0)
        for virtual_idx in range(0, self.num_points, max(1, self.num_points // 40)):
            virtual_c2w = self.virtual_extrinsic_c2w[virtual_idx]
            virtual_center = virtual_c2w[:3, 3]
            # virtual_center = self.virtual_cam_centers[virtual_idx]
            virtual_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.1)
            virtual_color = gt_colors[self._inv_assignment_closest[virtual_idx]][:3]
            virtual_sphere.paint_uniform_color(virtual_color)
            virtual_sphere.translate(virtual_center)
            virtual_sphere.compute_vertex_normals()
            # virtual_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.5, origin=virtual_center)
            # virtual_frame.rotate(virtual_c2w[:3, :3])
            scene += virtual_sphere
            # scene += virtual_frame

        output_ply_path = Path(output_path).with_suffix('.ply')
        o3d.io.write_triangle_mesh(output_ply_path, scene, write_ascii=True)
        log(f"[{self.__class__.__name__}::export_poses] Camera poses saved to {output_ply_path.resolve().parent.name}/{output_ply_path.name}", 'debug')

        # Offscreen render
        renderer = o3d.visualization.rendering.OffscreenRenderer(1024, 1024)
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Set background to black
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("scene", scene, mat)

        # Set up camera to view the whole scene
        bounds = scene.get_axis_aligned_bounding_box()
        center = bounds.get_center().astype(np.float32)  # (3,)
        extent = bounds.get_extent().max()
        cam_pos = (center + np.array([0.8, 0.8, 0.8]) * extent).astype(np.float32)
        up = np.array([-0.9, -0.6, 0.2], dtype=np.float32)

        renderer.setup_camera(60.0, center, cam_pos, up)
        img = renderer.render_to_image()
        output_image_path = Path(output_path).with_suffix('.png')
        o3d.io.write_image(output_image_path, img)
        log(f"[{self.__class__.__name__}::export_poses] Camera poses render saved to {output_image_path.resolve().parent.name}/{output_image_path.name}", 'debug')

    @staticmethod
    def compute_mean_look_at_and_up(cam_centers: np.ndarray, cam_rotmats_c2w: np.ndarray):
        """
        Compute the mean look-at point and up vector from camera centers and rotation matrices.

        Args:
            cam_centers (np.ndarray): Camera centers in world coordinates of shape (N, 3).
            cam_rotmats_c2w (np.ndarray): Camera rotation matrices of shape (N, 3, 3).

        Returns:
            tuple: Mean look-at point and up vector.
        """
        N = cam_centers.shape[0]

        # 1) Compute forward (lookat) and up vectors in world coordinates
        # Forward direction (-Z axis) of each camera in world coords:
        # forward_i = <last column of cam-to-world rotmats> = <last row of world-to-cam rotmats>
        forward_vectors = -cam_rotmats_c2w[:, :, 2]  # (N, 3)
        forward_vectors = forward_vectors / np.linalg.norm(forward_vectors, axis=1, keepdims=True)
        # Up direction in camera coords is (0,1,0), so in world:
        # up_i = <second column of cam-to-world rotmats> = <second row of world-to-cam rotmats>
        up_vectors = cam_rotmats_c2w[:, :, 1]  # (N, 3)
        up_vectors = up_vectors / np.linalg.norm(up_vectors, axis=1, keepdims=True)
        up_mean = up_vectors.mean(0)  # (3,)
        up_mean = up_mean / np.linalg.norm(up_mean)

        # 2) Compute the best-fit point P that minimizes sum of squared distances to each line:
        # Each line: P(t) = C_i + lambda_i * forward_i
        # We want a point P that is closest on average to all look at linelines.
        # The normal equation for this problem:
        # sum_i (I - f_i f_i^T)(P - C_i) = 0
        # sum_i (I - f_i f_i^T) P = sum_i (I - f_i f_i^T) C_i
        # Let M = sum_i (I - f_i f_i^T) and b = sum_i (I - f_i f_i^T) C_i
        # Then P = M^-1 b
        eye = np.eye(3, dtype=np.float32)
        M = np.zeros((3, 3), dtype=np.float32)
        b = np.zeros((3,), dtype=np.float32)
        for i in range(N):
            f = forward_vectors[i]
            # outer product f f^T
            f_outer = f[:, None] @ f[None, :]  # (3,3)
            A = eye - f_outer
            M += A
            b += A @ cam_centers[i]
        # Solve M P = b
        mean_look_at_point = np.linalg.solve(M, b)  # shape (3,)

        return mean_look_at_point, up_mean

    @staticmethod
    def create_rotation_interpolator(rot_matrices: np.ndarray):
        """
        Create a spherical linear interpolation (SLERP) function for a sequence of rotation matrices.

        Parameters
        ----------
        rot_matrices : np.ndarray
            Array of shape (N, 3, 3) containing rotation matrices in c2w format.

        Returns
        -------
        function
            A function that takes a parameter t (0 <= t <= 1) and returns the interpolated rotation matrix.
        """
        N = len(rot_matrices)
        t_points = np.linspace(0, 1, N)
        key_rots = ScipyRotation.from_matrix(rot_matrices)
        slerp = Slerp(t_points, key_rots)

        def interpolate_rotation(t):
            return slerp(t).as_matrix()

        return interpolate_rotation

    @staticmethod
    def create_translation_interpolator(points: np.ndarray):
        """
        Create a cubic spline interpolation function for a sequence of 3D points.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N, 3) containing 3D points in world coordinates.

        Returns
        -------
        function
            A function that takes a parameter t (0 <= t <= 1) and returns the interpolated 3D position.
        """
        N = len(points)
        t_points = np.linspace(0, 1, N)
        splines = [CubicSpline(t_points, points[:, i]) for i in range(3)]

        def interpolate_position(t):
            return np.array([s(t) for s in splines])

        return interpolate_position

    @staticmethod
    def build_virtual_to_gt_mapping(gt_to_virtuals: dict[int, Iterable[int]], indices: Optional[List[int]] = None) -> dict[int, int]:
        virtual_to_gt = {}
        for gt_idx, virtuals in gt_to_virtuals.items():
            for v in virtuals:
                virtual_to_gt[int(v)] = int(indices.index(int(gt_idx)) if indices is not None else int(gt_idx))
        return virtual_to_gt


class InterpolatedCameraOrbit(CameraOrbit):
    def create_virtual_cameras(self, gt_intrinsics: np.ndarray, gt_extrinsics_c2w: np.ndarray, assignment_idx: List[int], mean_lookat=None, mean_up=None, **trajectory_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Iterable[int]], Dict[int, Iterable[int]]]:
        # go to torch to reuse our current utils
        if mean_up is None or mean_lookat is None:
            mean_lookat_point, mean_up_vector = VisUtils.compute_mean_look_at_and_up(
                torch.from_numpy(gt_extrinsics_c2w[:, :3, 3]),  # Camera centers
                torch.from_numpy(gt_extrinsics_c2w[:, :3, :3])  # Camera rotation matrices
            )
        else:
            mean_lookat_point = torch.from_numpy(mean_lookat).float()
            mean_up_vector = torch.from_numpy(mean_up).float()
        virtual_cam_centers, assignment_closest, assignment_middle = VisUtils.virtual_tvecs(
            tvecs=torch.from_numpy(gt_extrinsics_c2w[:, :3, 3]),
            vis_only_idx=assignment_idx,
            num_points=self.num_points,  # need a multiple of 100 to map t from [0, 1] to [0, N]
            use_bezier=True,
            **trajectory_kwargs
        )
        # Create virtual cameras
        virtual_w2c_rot, virtual_w2c_t = look_at_view_transform(
            eye=virtual_cam_centers,
            at=mean_lookat_point.unsqueeze(0).expand(virtual_cam_centers.shape[0], -1),
            up=mean_up_vector.unsqueeze(0).expand(virtual_cam_centers.shape[0], -1),
            device=virtual_cam_centers.device
        )
        virtual_intrinsics = torch.from_numpy(gt_intrinsics).mean(0).expand(virtual_w2c_t.shape[0], 3, 3)
        virtual_extrinsics_w2c = torch.eye(4)[None].repeat(virtual_w2c_t.shape[0], 1, 1)
        virtual_extrinsics_w2c[..., :3, :3] = virtual_w2c_rot.transpose(-1, -2)
        virtual_extrinsics_w2c[..., :3, 3] = virtual_w2c_t
        return virtual_intrinsics.cpu().numpy(), virtual_extrinsics_w2c.cpu().numpy(), mean_lookat_point.cpu().numpy(), mean_up_vector.cpu().numpy(), assignment_closest, assignment_middle

    @classmethod
    def from_session(cls,
                     calibration_session: Union[Path, str],
                     calibration_method: Literal['MultiCamCalib', 'Caliscope'] = 'MultiCamCalib',
                     reconstruction_idx: Union[Literal['all'], Sequence[int]] = 'all',
                     image_size_hw: Optional[Tuple[int, int]] = None,
                     calibration_data_from_folder: Optional[Path] = None,
                     **trajectory_kwargs) -> 'InterpolatedCameraOrbit':
        """
        Create an InterpolatedCameraOrbit instance from a calibration session directory.

        Parameters
        ----------
        calibration_session : Union[Path, str]
            Path to the calibration session directory.
        calibration_method : Literal['MultiCamCalib', 'Caliscope']
            The calibration method used for the session. See `utils.calib.CalibrationData` for details.
        reconstruction_idx : Union[Literal['all'], List[int]]
            Indices of the reconstruction cameras to use for the virtual-to-gt camera assignment.
        image_size_hw : Optional[Tuple[int, int]]
            If provided, the image size to resize the ground truth images to. If None, the original size is used.

        Returns
        -------
        InterpolatedCameraOrbit
            An instance of InterpolatedCameraOrbit.
        """
        from utils.calib import CalibrationData
        if calibration_data_from_folder is None:
            calibration_data = CalibrationData.from_session(calibration_session, method=calibration_method)  #
        else:
            calibration_data = CalibrationData.from_session_folder(calibration_data_from_folder)
        if image_size_hw is not None:
            calibration_data = calibration_data.resize(*image_size_hw)
        return cls(
            gt_intrinsics=calibration_data.intrinsics.cpu().numpy(),
            gt_extrinsics_w2c=calibration_data.extrinsics_w2c.cpu().numpy(),
            reconstruction_idx=[int(_) for _ in reconstruction_idx] if not isinstance(reconstruction_idx, str) else reconstruction_idx,
            gt_image_size_hw=(int(calibration_data.image_size[0][0]), int(calibration_data.image_size[0][1])),
            **trajectory_kwargs
        )


if __name__ == "__main__":
    from utils.calib import CalibrationData

    # read session data
    orbit_ = InterpolatedCameraOrbit.from_session('Thanos_2_Calib_1', reconstruction_idx=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], bezier_degree=8)
    orbit_.export_poses()
    counter_ = 0
    for t_, s_, gtic_, gtim_ in orbit_.traverse(velocity=3.0, loop=True, debug_mode=True):
        print(f"t: {t_:.3f}, s: {s_:.2f} (gt={gtic_:02d}|{gtim_:02d})")
        if counter_ > 100:
            break
        counter_ += 1
