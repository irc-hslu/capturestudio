import abc
from pathlib import Path
from typing import List, Literal, Union, Iterator, Tuple, Iterable, Dict, Sequence, Optional

import numpy as np
import torch
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

        # Store floor and wall data (if present)
        self._floor_wall_data = {k: v for k, v in trajectory_kwargs.items() if k.startswith('floor') or k.startswith('wall')}
        self._trajectory_kwargs = {k: v for k, v in trajectory_kwargs.items() if not k.startswith('floor') and not k.startswith('wall')}

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

    def export_poses(self, scale: float = 0.4, output_path: Union[Path, str] = 'camera_poses.png', visualize_traversal: bool = False) -> None:
        import numpy as np
        import open3d as o3d
        from matplotlib import pyplot as plt
        from utils.misc import log
        from pathlib import Path

        def _make_plane_mesh(n: np.ndarray, d: float, center_hint: np.ndarray, extent: float, color_rgb=(0.5, 0.5, 0.5)) -> o3d.geometry.TriangleMesh:
            n = np.asarray(n, dtype=np.float64)
            n = n / (np.linalg.norm(n) + 1e-12)
            a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(a, n)) > 0.95:
                a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            e1 = np.cross(n, a);
            e1 /= (np.linalg.norm(e1) + 1e-12)
            e2 = np.cross(n, e1);
            e2 /= (np.linalg.norm(e2) + 1e-12)
            c0 = center_hint - (np.dot(n, center_hint) + d) * n
            s = float(extent)
            corners = np.stack([c0 + (-s) * e1 + (-s) * e2,
                                c0 + (s) * e1 + (-s) * e2,
                                c0 + (s) * e1 + (s) * e2,
                                c0 + (-s) * e1 + (s) * e2], axis=0).astype(np.float64)
            tri = o3d.geometry.TriangleMesh()
            tri.vertices = o3d.utility.Vector3dVector(corners)
            tri.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
            tri.compute_vertex_normals()
            tri.paint_uniform_color(color_rgb)
            return tri

        scene = o3d.geometry.TriangleMesh()
        for c2w in self.gt_extrinsics_c2w:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=c2w[:3, 3])
            frame.rotate(c2w[:3, :3])
            scene += frame

        sph0 = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.5)
        sph0.paint_uniform_color([1.0, 0.2, 0.8])
        sph0.translate(self.mean_look_at_point)
        sph0.compute_vertex_normals()
        scene += sph0

        if visualize_traversal:
            order = []
            seen = set()
            steps = max(self.num_points, 200)
            fps = 30
            single_vcam_velocity = np.linalg.norm(np.diff(self.virtual_extrinsic_c2w[:, :3, 3], axis=0), axis=1).mean() * fps
            it = self.traverse(velocity=single_vcam_velocity, loop=False, data_fps=fps, debug_mode=True)
            for _ in range(steps):
                try:
                    t, s, gt_idx_closest, gt_idx_middle = next(it)
                except StopIteration:
                    break
                vidx = min(int(t * self.num_points), self.num_points - 1)
                if len(order) == 0 or order[-1] != vidx:
                    order.append(vidx)
                    seen.add(vidx)
                if len(seen) >= self.num_points:
                    break
            assert len(order) > 0, 'No virtual cams yielded during traversal'
            cmap = plt.cm.get_cmap('viridis')
            r0, r1 = scale * 0.05, scale * 0.22
            M = len(order)
            for k, vidx in enumerate(order):
                alpha = 0.0 if M <= 1 else k / (M - 1)
                col = cmap(alpha)[:3]
                r = r0 + (r1 - r0) * alpha
                vc = self.virtual_extrinsic_c2w[vidx][:3, 3]
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(r))
                sph.paint_uniform_color([float(col[0]), float(col[1]), float(col[2])])
                sph.translate(vc)
                sph.compute_vertex_normals()
                scene += sph
        else:
            cmap = plt.cm.get_cmap('tab20')
            n_gt = len(self.gt_extrinsics_c2w)
            color_indices = np.arange(n_gt) % 20
            gt_colors = cmap(color_indices / 19.0)
            for virtual_idx in range(0, self.num_points, max(1, self.num_points // 40)):
                virtual_c2w = self.virtual_extrinsic_c2w[virtual_idx]
                vc = virtual_c2w[:3, 3]
                # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=vc)
                # frame.rotate(virtual_c2w[:3, :3])
                # scene += frame
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.1)
                col = gt_colors[self._inv_assignment_closest.get(virtual_idx, 0) % 20][:3]
                sph.paint_uniform_color(col)
                sph.translate(vc)
                sph.compute_vertex_normals()
                scene += sph

        fw = getattr(self, "_floor_wall_data", {}) or {}
        floor_n = fw.get("floor_normal", None)
        floor_d = fw.get("floor_offset", None)
        all_pts = np.vstack([self.gt_extrinsics_c2w[:, :3, 3], self.virtual_extrinsic_c2w[:, :3, 3]])
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(all_pts.astype(np.float64)))
        extent = 0.6 * np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound())
        center_hint = aabb.get_center().astype(np.float64)
        # if (isinstance(floor_n, np.ndarray)) and (floor_d is not None):
        #     floor_mesh = _make_plane_mesh(floor_n, float(floor_d), center_hint, extent, color_rgb=(0.35, 0.35, 0.35))
        #     scene += floor_mesh

        output_ply_path = Path(output_path).with_suffix('.ply')
        o3d.io.write_triangle_mesh(output_ply_path, scene, write_ascii=True)
        log(f"[{self.__class__.__name__}::export_poses] Camera poses saved to {output_ply_path.resolve().parent.name}/{output_ply_path.name}", 'debug')

        renderer = o3d.visualization.rendering.OffscreenRenderer(1024, 1024)
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("scene", scene, mat)
        bounds = scene.get_axis_aligned_bounding_box()
        center = bounds.get_center().astype(np.float32)
        extent_len = bounds.get_extent().max()
        cam_pos = (center + np.array([0.8, 0.8, 0.8]) * extent_len).astype(np.float32)
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
            **{k: v for k, v in trajectory_kwargs.items() if not k.startswith('floor') and not k.startswith('wall')}
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
                     rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None,
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
        rotate: Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180'], optional
            Rotation to apply to intrinsics to unrotate camera data.

        Returns
        -------
        InterpolatedCameraOrbit
            An instance of InterpolatedCameraOrbit.
        """
        from utils.calib import CalibrationData
        if calibration_data_from_folder is None:
            calibration_data = CalibrationData.from_session(calibration_session, method=calibration_method)
        else:
            calibration_data = CalibrationData.from_session_folder(calibration_data_from_folder)
        if image_size_hw is not None:
            calibration_data = (calibration_data
                                .rotate(rotate)
                                .resize(*image_size_hw))
        else:
            calibration_data = calibration_data.rotate(rotate)
        return cls(
            gt_intrinsics=calibration_data.intrinsics.cpu().numpy(),
            gt_extrinsics_w2c=calibration_data.extrinsics_w2c.cpu().numpy(),
            reconstruction_idx=[int(_) for _ in reconstruction_idx] if not isinstance(reconstruction_idx, str) else reconstruction_idx,
            gt_image_size_hw=(int(calibration_data.image_size[0][0]), int(calibration_data.image_size[0][1])),
            **trajectory_kwargs
        )


class AudienceViewAnchoredCameraOrbit_bak(CameraOrbit):
    def create_virtual_cameras(self,
                               gt_intrinsics: np.ndarray,
                               gt_extrinsics_c2w: np.ndarray,
                               assignment_idx: List[int],
                               mean_lookat: Optional[np.ndarray] = None,
                               mean_up: Optional[np.ndarray] = None,
                               **trajectory_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Iterable[int]], Dict[int, Iterable[int]]]:
        import torch
        from pytorch3d.renderer import look_at_view_transform

        def _normalize(v):
            v = torch.as_tensor(v, dtype=torch.float32)
            n = torch.linalg.norm(v)
            return v / (n + 1e-12)

        Cw = torch.from_numpy(gt_extrinsics_c2w[:, :3, 3]).float()
        Rw = torch.from_numpy(gt_extrinsics_c2w[:, :3, :3]).float()

        if mean_up is None or mean_lookat is None:
            mean_lookat_point, mean_up_vector = VisUtils.compute_mean_look_at_and_up(Cw, Rw)
        else:
            mean_lookat_point = torch.from_numpy(mean_lookat).float()
            mean_up_vector = torch.from_numpy(mean_up).float()

        floor_wall_data = {k: v for k, v in trajectory_kwargs.items() if k.startswith('floor') or k.startswith('wall')}
        n_floor = floor_wall_data.get("floor_normal", None)
        n_floor_t = _normalize(mean_up_vector) if n_floor is None else _normalize(torch.from_numpy(n_floor))
        if torch.dot(n_floor_t, _normalize(mean_up_vector)) < 0:
            n_floor_t = -n_floor_t

        n_wall = floor_wall_data.get("wall_normal", None)
        n_wall_t = None if n_wall is None else -_normalize(torch.from_numpy(n_wall))

        # anchors
        i0, i1 = int(assignment_idx[0]), int(assignment_idx[-1])
        p0 = Cw[i0].clone()
        p1 = Cw[i1].clone()

        # arc plane parallel to floor (height = average of anchors)
        h0 = torch.dot(n_floor_t, p0)
        h1 = torch.dot(n_floor_t, p1)
        h = 0.5 * (h0 + h1)
        p0p = p0 + (h - h0) * n_floor_t
        p1p = p1 + (h - h1) * n_floor_t

        chord = p1p - p0p
        chord_len = torch.linalg.norm(chord)
        if chord_len.item() < 1e-6:
            any_in_plane = _normalize(torch.cross(n_floor_t, torch.tensor([1.0, 0.0, 0.0])))
            if torch.linalg.norm(any_in_plane) < 1e-3:
                any_in_plane = _normalize(torch.cross(n_floor_t, torch.tensor([0.0, 1.0, 0.0])))
            p1p = p0p + 0.25 * any_in_plane
            chord = p1p - p0p
            chord_len = torch.linalg.norm(chord)

        t_hat = chord / (chord_len + 1e-12)
        v_hat = _normalize(torch.cross(n_floor_t, t_hat))  # in-plane, perpendicular to chord

        # TODO: Wall estimation is unreliable. Select solution that is the furthest from the (mean) lookat point.
        # facing the wall (bulge towards wall): project wall onto plane to set sign; fallback to look-at
        if n_wall_t is not None:
            w_in_plane = n_wall_t - torch.dot(n_wall_t, n_floor_t) * n_floor_t
            if torch.linalg.norm(w_in_plane) < 1e-9:
                w_in_plane = v_hat
            sign_pref = torch.sign(torch.dot(_normalize(w_in_plane), v_hat))
        else:
            mid = 0.5 * (p0p + p1p)
            to_scene = (mean_lookat_point - mid)
            w_in_plane = to_scene - torch.dot(to_scene, n_floor_t) * n_floor_t
            sign_pref = -torch.sign(torch.dot(_normalize(w_in_plane), v_hat)) if torch.linalg.norm(w_in_plane) > 1e-9 else torch.tensor(1.0)

        # -------- project all GT cams to the arc plane, derive extent & peak location --------
        h_all = torch.matmul(Cw, n_floor_t)  # [N]
        Cproj = Cw + (h - h_all)[:, None] * n_floor_t[None, :]  # onto <n,x>=h
        rel = Cproj - p0p[None, :]
        s = torch.sum(rel * t_hat[None, :], dim=1)  # along-chord
        y = torch.sum(rel * v_hat[None, :], dim=1)  # signed perpendicular

        on_strip = (s >= 0.0) & (s <= chord_len + 1e-6)
        idx_pool = torch.nonzero(on_strip, as_tuple=False).squeeze(-1) if torch.any(on_strip) else torch.arange(Cproj.shape[0])

        # furthest camera (by |y|)
        idx_far_local = torch.argmax(torch.abs(y[idx_pool]))
        idx_far = idx_pool[idx_far_local]
        s_far = s[idx_far]
        y_far = y[idx_far]

        # extent & min-curvature logic
        y_extent = torch.max(torch.abs(y[idx_pool])) if idx_pool.numel() > 0 else torch.tensor(0.0)
        y_extent_ratio_thresh = float(trajectory_kwargs.get('y_extent_ratio_thresh', 0.06))
        min_sag_ratio = float(trajectory_kwargs.get('min_sag_ratio', 0.12))
        small_extent = (y_extent < y_extent_ratio_thresh * chord_len)

        reach = float(trajectory_kwargs.get('reach_factor', 1.9))
        clearance = float(trajectory_kwargs.get('clearance_ratio', 0.10)) * chord_len
        sag_min = 0.05 * chord_len
        sag_max = 0.85 * chord_len

        if small_extent:
            sag = torch.clamp(min_sag_ratio * chord_len, min=sag_min, max=sag_max)
            dir_sign = sign_pref  # use wall-facing direction
            t_peak = 0.5
        else:
            sag_target = torch.abs(y_far) * reach - clearance
            sag = torch.clamp(sag_target, min=sag_min, max=sag_max)
            dir_sign = torch.sign(y_far) if torch.abs(y_far) > 1e-9 else sign_pref
            t_peak = float(torch.clamp(s_far / (chord_len + 1e-12), 0.0, 1.0))

        bend_dir = dir_sign * v_hat

        # -------- build Bézier curve (degree 5 by default) with softened peak --------
        degree = int(trajectory_kwargs.get('degree', 5))
        half_w = float(trajectory_kwargs.get('bulge_half_width', 0.6))
        peak_soft = float(trajectory_kwargs.get('peak_softness', 1.00))  # 0..1, larger spreads/softens the bump
        t_vals = torch.linspace(0.0, 1.0, steps=self.num_points).unsqueeze(1)

        if degree == 3:
            # cubic: two controls around the peak
            beta = max(0.04, min(0.96, t_peak - half_w))
            gamma = max(0.04, min(0.96, t_peak + half_w))
            c0 = (1.0 - beta) * p0p + beta * p1p + sag * bend_dir
            c1 = (1.0 - gamma) * p0p + gamma * p1p + sag * bend_dir
            B = ((1 - t_vals) ** 3) * p0p[None, :] \
                + 3 * ((1 - t_vals) ** 2) * t_vals * c0[None, :] \
                + 3 * (1 - t_vals) * (t_vals ** 2) * c1[None, :] \
                + (t_vals ** 3) * p1p[None, :]
        else:
            # quintic: four controls; inner ones near peak at full sag, outer ones reduced by peak_soft
            t1 = max(0.02, min(0.98, t_peak - 2 * half_w))
            t2 = max(0.02, min(0.98, t_peak - half_w))
            t3 = max(0.02, min(0.98, t_peak + half_w))
            t4 = max(0.02, min(0.98, t_peak + 2 * half_w))
            # ensure monotonic
            t1, t2, t3, t4 = sorted([t1, t2, t3, t4])
            # chord points
            L = lambda t: (1.0 - t) * p0p + t * p1p
            # control deflections (outer softened)
            a_out = max(0.0, min(1.0, 0.5 * peak_soft))  # ~0.3 by default
            a_in = 1.0  # full sag at inner controls
            P0 = p0p
            P1 = L(t1) + (a_out * sag) * bend_dir
            P2 = L(t2) + (a_in * sag) * bend_dir
            P3 = L(t3) + (a_in * sag) * bend_dir
            P4 = L(t4) + (a_out * sag) * bend_dir
            P5 = p1p
            # binomial coefficients for n=5: [1,5,10,10,5,1]
            one = (1 - t_vals);
            t = t_vals
            B = (one ** 5) * P0[None, :] \
                + 5 * (one ** 4) * t * P1[None, :] \
                + 10 * (one ** 3) * (t ** 2) * P2[None, :] \
                + 10 * (one ** 2) * (t ** 3) * P3[None, :] \
                + 5 * one * (t ** 4) * P4[None, :] \
                + (t ** 5) * P5[None, :]

        eye = B
        up = n_floor_t.expand_as(eye)

        # camera frames
        R_w2c, T_w2c = look_at_view_transform(
            eye=eye, at=mean_lookat_point.expand_as(eye), up=up, device=eye.device
        )
        K_mean = torch.from_numpy(gt_intrinsics).float().mean(0, keepdim=True).expand(eye.shape[0], -1, -1)

        extr = torch.eye(4, dtype=torch.float32)[None].repeat(eye.shape[0], 1, 1)
        extr[:, :3, :3] = R_w2c.transpose(-1, -2)
        extr[:, :3, 3] = T_w2c

        # ---------------- assignments (unchanged) ----------------
        idx_tensor = torch.tensor(assignment_idx, dtype=torch.long)
        C_assign = Cw[idx_tensor]
        dists = torch.cdist(eye, C_assign)
        nearest = torch.argmin(dists, dim=1)

        assignment_closest: Dict[int, Iterable[int]] = {}
        for j_cam_in_list in range(C_assign.shape[0]):
            frame_ids = torch.nonzero(nearest == j_cam_in_list, as_tuple=False).squeeze(-1).cpu().tolist()
            if frame_ids:
                gt_idx = int(assignment_idx[j_cam_in_list])
                assignment_closest[gt_idx] = frame_ids

        L = len(assignment_idx)
        assignment_middle: Dict[int, Iterable[int]] = {}
        Tn = eye.shape[0]
        if L == 1:
            assignment_middle[int(assignment_idx[0])] = list(range(Tn))
        elif L == 2:
            assignment_middle[int(assignment_idx[1])] = list(range(Tn))
        else:
            interior_min_idx = []
            for i_mid in range(1, L - 1):
                dist_mid = dists[:, i_mid]
                t_star = int(torch.argmin(dist_mid).item())
                interior_min_idx.append(t_star)
            boundaries = [0] + sorted(interior_min_idx) + [Tn]
            for seg_i in range(L - 1):
                start = int(boundaries[seg_i])
                end = int(boundaries[seg_i + 1])
                if end <= start: continue
                right_key = int(assignment_idx[seg_i + 1])
                assignment_middle[right_key] = list(range(start, end))

        return (
            K_mean.cpu().numpy(),
            extr.cpu().numpy(),
            mean_lookat_point.cpu().numpy(),
            up[0].cpu().numpy(),
            assignment_closest,
            assignment_middle
        )

    @classmethod
    def from_session(cls,
                     calibration_session: Union[Path, str],
                     calibration_method: Literal['MultiCamCalib', 'Caliscope'] = 'MultiCamCalib',
                     reconstruction_idx: Union[Literal['all'], Sequence[int]] = 'all',
                     image_size_hw: Optional[Tuple[int, int]] = None,
                     calibration_data_from_folder: Optional[Path] = None,
                     rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None,
                     **trajectory_kwargs) -> 'AudienceViewAnchoredCameraOrbit':
        from utils.calib import CalibrationData
        if calibration_data_from_folder is None:
            calibration_data = CalibrationData.from_session(calibration_session, method=calibration_method)
        else:
            calibration_data = CalibrationData.from_session_folder(calibration_data_from_folder)
        if image_size_hw is not None:
            calibration_data = calibration_data.rotate(rotate).resize(*image_size_hw)
        else:
            calibration_data =  calibration_data.rotate(rotate)
        return cls(
            gt_intrinsics=calibration_data.intrinsics.cpu().numpy(),
            gt_extrinsics_w2c=calibration_data.extrinsics_w2c.cpu().numpy(),
            reconstruction_idx=[int(_) for _ in reconstruction_idx] if not isinstance(reconstruction_idx, str) else reconstruction_idx,
            gt_image_size_hw=(int(calibration_data.image_size[0][0]), int(calibration_data.image_size[0][1])),
            **trajectory_kwargs
        )

class AudienceViewAnchoredCameraOrbit(CameraOrbit):
    def create_virtual_cameras(self,
                               gt_intrinsics: np.ndarray,
                               gt_extrinsics_c2w: np.ndarray,
                               assignment_idx: List[int],
                               mean_lookat: Optional[np.ndarray] = None,
                               mean_up: Optional[np.ndarray] = None,
                               **trajectory_kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Iterable[int]], Dict[int, Iterable[int]]]:
        """
        Build an 'audience' orbit anchored at two GT cameras, lying on the (estimated) floor plane.

        Fix: ensure the bulge direction goes AWAY from the wall by default.
        You can override with:
          - bulge_mode in {"away","toward","auto"} (default "away")
              "away"   -> always bend away from wall
              "toward" -> always bend toward wall
              "auto"   -> follow legacy heuristic (may pick either side)
        """
        import torch
        from pytorch3d.renderer import look_at_view_transform

        bulge_mode: str = str(trajectory_kwargs.get("bulge_mode", "away")).lower()
        assert bulge_mode in {"away", "toward", "auto"}

        def _normalize(v):
            v = torch.as_tensor(v, dtype=torch.float32)
            n = torch.linalg.norm(v)
            return v / (n + 1e-12)

        Cw = torch.from_numpy(gt_extrinsics_c2w[:, :3, 3]).float()
        Rw = torch.from_numpy(gt_extrinsics_c2w[:, :3, :3]).float()

        if mean_up is None or mean_lookat is None:
            mean_lookat_point, mean_up_vector = VisUtils.compute_mean_look_at_and_up(Cw, Rw)
        else:
            mean_lookat_point = torch.from_numpy(mean_lookat).float()
            mean_up_vector = torch.from_numpy(mean_up).float()

        # Floor / wall priors
        floor_wall_data = {k: v for k, v in trajectory_kwargs.items() if k.startswith('floor') or k.startswith('wall')}
        n_floor = floor_wall_data.get("floor_normal", None)
        n_floor_t = _normalize(mean_up_vector) if n_floor is None else _normalize(torch.from_numpy(n_floor))
        if torch.dot(n_floor_t, _normalize(mean_up_vector)) < 0:
            n_floor_t = -n_floor_t

        n_wall = floor_wall_data.get("wall_normal", None)
        # IMPORTANT: do NOT flip the wall normal here; use as provided by estimator.
        n_wall_t = None if n_wall is None else _normalize(torch.from_numpy(n_wall))

        # Anchors
        i0, i1 = int(assignment_idx[0]), int(assignment_idx[-1])
        p0 = Cw[i0].clone()
        p1 = Cw[i1].clone()

        # Arc plane parallel to floor (height = average of anchors)
        h0 = torch.dot(n_floor_t, p0)
        h1 = torch.dot(n_floor_t, p1)
        h = 0.5 * (h0 + h1)
        p0p = p0 + (h - h0) * n_floor_t
        p1p = p1 + (h - h1) * n_floor_t

        chord = p1p - p0p
        chord_len = torch.linalg.norm(chord)
        if chord_len.item() < 1e-6:
            any_in_plane = _normalize(torch.cross(n_floor_t, torch.tensor([1.0, 0.0, 0.0])))
            if torch.linalg.norm(any_in_plane) < 1e-3:
                any_in_plane = _normalize(torch.cross(n_floor_t, torch.tensor([0.0, 1.0, 0.0])))
            p1p = p0p + 0.25 * any_in_plane
            chord = p1p - p0p
            chord_len = torch.linalg.norm(chord)

        t_hat = chord / (chord_len + 1e-12)
        # In-plane, perpendicular to chord (sign to be chosen)
        v_hat = _normalize(torch.cross(n_floor_t, t_hat))

        # ---------------------------------------------------------------------
        # Pick bulge direction sign. Define "toward wall" vector within floor:
        #   w_in_plane = (I - n_floor n_floor^T) * n_wall   (if wall known)
        # If wall is unknown, infer using look-at direction from chord mid.
        # Then:
        #   sign_wall_vs_vhat = sign( dot( normalize(w_in_plane), v_hat ) )
        # "Away" from wall => use -sign_wall_vs_vhat; "Toward" => +sign_wall_vs_vhat.
        # Legacy "auto" keeps prior heuristic (may go either way).
        # ---------------------------------------------------------------------
        if n_wall_t is not None:
            w_in_plane = n_wall_t - torch.dot(n_wall_t, n_floor_t) * n_floor_t
            if torch.linalg.norm(w_in_plane) < 1e-9:
                w_in_plane = v_hat  # degenerate, fallback
            sign_wall_vs_vhat = torch.sign(torch.dot(_normalize(w_in_plane), v_hat))
        else:
            # Fallback: direction from chord midpoint toward mean look-at, projected to floor
            mid = 0.5 * (p0p + p1p)
            to_scene = (mean_lookat_point - mid)
            w_in_plane = to_scene - torch.dot(to_scene, n_floor_t) * n_floor_t
            if torch.linalg.norm(w_in_plane) < 1e-9:
                sign_wall_vs_vhat = torch.tensor(1.0, dtype=torch.float32)
            else:
                sign_wall_vs_vhat = torch.sign(torch.dot(_normalize(w_in_plane), v_hat))

        # Preferred sign given mode
        # bulge_mode = "toward" # TODO SIMONE33 COMMENT THIS LINE, FOR SINGER CLOSE TO MICROPHONE bulge_mode = "toward"
        if bulge_mode == "away":
            base_sign = -sign_wall_vs_vhat  # enforce away from wall
        elif bulge_mode == "toward":
            base_sign = sign_wall_vs_vhat   # enforce toward wall
        else:  # "auto" legacy: same as original 'sign_pref'
            base_sign = sign_wall_vs_vhat

        # -------- project all GT cams to the arc plane, derive extent & peak location --------
        h_all = torch.matmul(Cw, n_floor_t)  # [N]
        Cproj = Cw + (h - h_all)[:, None] * n_floor_t[None, :]  # onto <n,x>=h
        rel = Cproj - p0p[None, :]
        s = torch.sum(rel * t_hat[None, :], dim=1)  # along-chord
        y = torch.sum(rel * v_hat[None, :], dim=1)  # signed perpendicular (w.r.t. current v_hat)

        on_strip = (s >= 0.0) & (s <= chord_len + 1e-6)
        idx_pool = torch.nonzero(on_strip, as_tuple=False).squeeze(-1) if torch.any(on_strip) else torch.arange(Cproj.shape[0])

        # furthest camera (by |y|)
        if idx_pool.numel() > 0:
            idx_far_local = torch.argmax(torch.abs(y[idx_pool]))
            idx_far = idx_pool[idx_far_local]
            s_far = s[idx_far]
            y_far = y[idx_far]
        else:
            idx_far = torch.tensor(0, dtype=torch.long)
            s_far = torch.tensor(0.5 * chord_len)
            y_far = torch.tensor(0.0)

        # extent & min-curvature logic (same magnitudes as before)
        y_extent = torch.max(torch.abs(y[idx_pool])) if idx_pool.numel() > 0 else torch.tensor(0.0)
        y_extent_ratio_thresh = float(trajectory_kwargs.get('y_extent_ratio_thresh', 0.06))
        min_sag_ratio = float(trajectory_kwargs.get('min_sag_ratio', 0.12))
        small_extent = (y_extent < y_extent_ratio_thresh * chord_len) and False

        reach = float(trajectory_kwargs.get('reach_factor', 1.9))
        clearance = float(trajectory_kwargs.get('clearance_ratio', 0.10)) * chord_len
        sag_min = 0.05 * chord_len
        sag_max = 0.85 * chord_len

        if small_extent:
            sag = torch.clamp(min_sag_ratio * chord_len, min=sag_min, max=sag_max)
            dir_sign = base_sign
            t_peak = 0.5
        else:
            sag_target = torch.abs(y_far) * reach - clearance
            sag = torch.clamp(sag_target, min=sag_min, max=sag_max)
            if bulge_mode in {"away", "toward"}:
                # Force direction per user-selected mode
                dir_sign = base_sign
            else:
                # Legacy auto: use the sign implied by GT spread
                dir_sign = torch.sign(y_far) if torch.abs(y_far) > 1e-9 else base_sign
            t_peak = float(torch.clamp(s_far / (chord_len + 1e-12), 0.0, 1.0))

        bend_dir = dir_sign * v_hat

        # -------- build Bézier curve (degree 5 by default) with softened peak --------
        degree = int(trajectory_kwargs.get('degree', 5))
        half_w = float(trajectory_kwargs.get('bulge_half_width', 0.6))
        peak_soft = float(trajectory_kwargs.get('peak_softness', 1.00))  # 0..1, larger spreads/softens the bump
        t_vals = torch.linspace(0.0, 1.0, steps=self.num_points).unsqueeze(1)

        if degree == 3:
            # cubic: two controls around the peak
            beta = max(0.04, min(0.96, t_peak - half_w))
            gamma = max(0.04, min(0.96, t_peak + half_w))
            c0 = (1.0 - beta) * p0p + beta * p1p + sag * bend_dir
            c1 = (1.0 - gamma) * p0p + gamma * p1p + sag * bend_dir
            B = ((1 - t_vals) ** 3) * p0p[None, :] \
                + 3 * ((1 - t_vals) ** 2) * t_vals * c0[None, :] \
                + 3 * (1 - t_vals) * (t_vals ** 2) * c1[None, :] \
                + (t_vals ** 3) * p1p[None, :]
        else:
            # quintic: four controls; inner ones near peak at full sag, outer ones reduced by peak_soft
            t1 = max(0.02, min(0.98, t_peak - 2 * half_w))
            t2 = max(0.02, min(0.98, t_peak - half_w))
            t3 = max(0.02, min(0.98, t_peak + half_w))
            t4 = max(0.02, min(0.98, t_peak + 2 * half_w))
            t1, t2, t3, t4 = sorted([t1, t2, t3, t4])

            L = lambda t: (1.0 - t) * p0p + t * p1p
            a_out = max(0.0, min(1.0, 0.5 * peak_soft))  # ~0.5*peak_soft
            a_in = 1.0
            P0 = p0p
            P1 = L(t1) + (a_out * sag) * bend_dir
            P2 = L(t2) + (a_in * sag) * bend_dir
            P3 = L(t3) + (a_in * sag) * bend_dir
            P4 = L(t4) + (a_out * sag) * bend_dir
            P5 = p1p

            one = (1 - t_vals)
            t = t_vals
            B = (one ** 5) * P0[None, :] \
                + 5 * (one ** 4) * t * P1[None, :] \
                + 10 * (one ** 3) * (t ** 2) * P2[None, :] \
                + 10 * (one ** 2) * (t ** 3) * P3[None, :] \
                + 5 * one * (t ** 4) * P4[None, :] \
                + (t ** 5) * P5[None, :]

        eye = B
        up = n_floor_t.expand_as(eye)

        # Camera frames
        R_w2c, T_w2c = look_at_view_transform(
            eye=eye, at=mean_lookat_point.expand_as(eye), up=up, device=eye.device
        )
        K_mean = torch.from_numpy(gt_intrinsics).float().mean(0, keepdim=True).expand(eye.shape[0], -1, -1)

        extr = torch.eye(4, dtype=torch.float32)[None].repeat(eye.shape[0], 1, 1)
        extr[:, :3, :3] = R_w2c.transpose(-1, -2)
        extr[:, :3, 3] = T_w2c

        # ---------------- assignments (unchanged) ----------------
        idx_tensor = torch.tensor(assignment_idx, dtype=torch.long)
        C_assign = Cw[idx_tensor]
        dists = torch.cdist(eye, C_assign)
        nearest = torch.argmin(dists, dim=1)

        assignment_closest: Dict[int, Iterable[int]] = {}
        for j_cam_in_list in range(C_assign.shape[0]):
            frame_ids = torch.nonzero(nearest == j_cam_in_list, as_tuple=False).squeeze(-1).cpu().tolist()
            if frame_ids:
                gt_idx = int(assignment_idx[j_cam_in_list])
                assignment_closest[gt_idx] = frame_ids

        Ln = len(assignment_idx)
        assignment_middle: Dict[int, Iterable[int]] = {}
        Tn = eye.shape[0]
        if Ln == 1:
            assignment_middle[int(assignment_idx[0])] = list(range(Tn))
        elif Ln == 2:
            assignment_middle[int(assignment_idx[1])] = list(range(Tn))
        else:
            interior_min_idx = []
            for i_mid in range(1, Ln - 1):
                dist_mid = dists[:, i_mid]
                t_star = int(torch.argmin(dist_mid).item())
                interior_min_idx.append(t_star)
            boundaries = [0] + sorted(interior_min_idx) + [Tn]
            for seg_i in range(Ln - 1):
                start = int(boundaries[seg_i])
                end = int(boundaries[seg_i + 1])
                if end <= start:
                    continue
                right_key = int(assignment_idx[seg_i + 1])
                assignment_middle[right_key] = list(range(start, end))

        return (
            K_mean.cpu().numpy(),
            extr.cpu().numpy(),
            mean_lookat_point.cpu().numpy(),
            up[0].cpu().numpy(),
            assignment_closest,
            assignment_middle
        )

    @classmethod
    def from_session(cls,
                     calibration_session: Union[Path, str],
                     calibration_method: Literal['MultiCamCalib', 'Caliscope'] = 'MultiCamCalib',
                     reconstruction_idx: Union[Literal['all'], Sequence[int]] = 'all',
                     image_size_hw: Optional[Tuple[int, int]] = None,
                     calibration_data_from_folder: Optional[Path] = None,
                     rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None,
                     **trajectory_kwargs) -> 'AudienceViewAnchoredCameraOrbit':
        from utils.calib import CalibrationData
        if calibration_data_from_folder is None:
            calibration_data = CalibrationData.from_session(calibration_session, method=calibration_method)
        else:
            calibration_data = CalibrationData.from_session_folder(calibration_data_from_folder)
        if image_size_hw is not None:
            calibration_data = calibration_data.rotate(rotate).resize(*image_size_hw)
        else:
            calibration_data = calibration_data.rotate(rotate)
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
