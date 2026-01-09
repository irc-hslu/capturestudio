# Synthetic data generation for THuman2.1 models
import copy
import json
import math
import pickle
import random
from pathlib import Path
from typing import Optional, Tuple, List, Literal, Union, Dict

import cv2
import numpy as np
import open3d as o3d
import pytorch3d.transforms
import torch
from pytorch3d.transforms import RotateAxisAngle, Rotate
from tqdm import tqdm

from utils.calib import CalibrationData
from utils.misc import PathUtils, Gender
from utils.vis import VisUtils

THUMAN_ROOT = Path('/media/charisoudis/nas_transmixr/Simone/Volumetric_Video/Human Datasets/THuman2_1')
CAPTURES_ROOT = PathUtils.project_path().parent / 'capturestudio' / 'out' / 'Captures_Apr_May_2025'
CALIBRATION_SESSION_NAME = 'Thanos_2_Calib_1'
CALIBRATION_METHOD: Literal['MultiCamCalib', 'Caliscope'] = 'MultiCamCalib'
FOCAL_RATIO = 0.5859


class THumanUtilsO3d:
    MeshTransformMatrix = RotateAxisAngle(180, 'Y').compose(RotateAxisAngle(180, 'X')).compose(RotateAxisAngle(45, 'X')).get_matrix().squeeze().transpose(-1, -2)  # 4x4
    SMPLX = None

    @classmethod
    def read_model(cls, model_idx: int, is_train: bool = True, thuman_root: Optional[Path | str] = None) -> Tuple[Path, Path | None]:
        """
        Read the model file for a given index.
        """
        if thuman_root is None:
            thuman_root = PathUtils.dataset_path('THuman2_1')
        models_path = (Path(thuman_root) / 'model') if (Path(thuman_root) / 'model').exists() else Path(thuman_root)
        model_path = models_path / ('train' if is_train else 'val') / f'{model_idx:04d}' / f'{model_idx:04d}.obj'
        if model_path.with_stem(f'{model_idx:04d}_original').exists():
            model_path = model_path.with_stem(f'{model_idx:04d}_original')
        # SMPL model
        if models_path.name == 'model':
            smpl_model_path = Path(thuman_root).parent / 'THuman_2_1_smplx' / 'smplx'
        else:
            smpl_model_path = thuman_root / 'THuman2.0_Smpl_X_Paras'
        smpl_model_path = smpl_model_path / f'{model_idx:04d}' / 'smplx_param.pkl'
        assert smpl_model_path.exists(), f"SMPL params not found at {smpl_model_path}"
        # if not smpl_model_path.exists():
        #     log(f"SMPL params not found at {smpl_model_path}", 'warning')
        #     return model_path, None
        return model_path, smpl_model_path

    @classmethod
    def smpl_forward(cls, smpl_parameters: Dict[str, np.ndarray], return_spline_dir: bool = False) -> Union[o3d.geometry.TriangleMesh, Tuple[o3d.geometry.TriangleMesh, np.ndarray]]:
        """
        Forward pass through the SMPL model to get the mesh.

        Args:
            smpl_parameters (Dict[str, np.ndarray]): SMPL parameters including 'betas', 'expression', 'thetas', 'global_orient',
                'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose'.
            return_spline_dir (bool): If True, also return the spline direction.

        Returns:
            o3d.geometry.TriangleMesh: The SMPL mesh.
        """
        # Get SMPL pointcloud
        if cls.SMPLX is None:
            from reconstruction.data.synthetic.smpl import Smplx
            cls.SMPLX = Smplx().cuda()
        with torch.no_grad():
            smpl_out = cls.SMPLX.forward(
                gender=Gender.NEUTRAL,
                betas=torch.from_numpy(smpl_parameters['betas'].reshape(1, 10)).float().cuda(),
                expression=torch.from_numpy(smpl_parameters['expression'].reshape(1, 10)).float().cuda(),
                thetas=torch.from_numpy(np.asarray(smpl_parameters['body_pose']).reshape(1, 21, 3)).float().cuda(),
                global_orient=torch.from_numpy(np.asarray(smpl_parameters['global_orient']).reshape(1, 3)).float().cuda(),
                left_hand_thetas=torch.from_numpy(smpl_parameters['left_hand_pose'].reshape(1, 15, 3)).float().cuda(),
                right_hand_thetas=torch.from_numpy(smpl_parameters['right_hand_pose'].reshape(1, 15, 3)).float().cuda(),
                jaw_pose=torch.from_numpy(smpl_parameters['jaw_pose'].reshape(1, 3)).float().cuda(),
                leye_pose=torch.from_numpy(smpl_parameters['leye_pose'].reshape(1, 3)).float().cuda(),
                reye_pose=torch.from_numpy(smpl_parameters['reye_pose'].reshape(1, 3)).float().cuda(),
                return_verts=True,
                return_shaped=True,
                center_on_pelvis=False
            )
        smpl_mesh = o3d.geometry.TriangleMesh()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(smpl_out.vertices.squeeze(0).cpu().numpy())
        smpl_mesh.triangles = o3d.utility.Vector3iVector(cls.SMPLX.get_faces(Gender.NEUTRAL).cpu().numpy())
        # smpl_mesh.compute_vertex_normals()
        smpl_mesh.paint_uniform_color([0.8, 0.6, 0.7])  # Light pink color for SMPL mesh
        if return_spline_dir:
            spline_dir = smpl_out.joints.cpu().numpy()[0][6] - smpl_out.joints.cpu().numpy()[0][3]
            spline_dir = spline_dir / np.linalg.norm(spline_dir)
            return smpl_mesh, spline_dir
        return smpl_mesh

    # noinspection PyIncorrectDocstring
    @classmethod
    def create_camera_frustum(cls,
                              cam: o3d.camera.PinholeCameraParameters,
                              mesh: Optional[o3d.geometry.Geometry] = None,
                              image_size: Tuple[int, int] = (1024, 1024),
                              color: List[float] = [1.0, 0.0, 0.0],
                              line: Literal['solid', 'dashed'] = 'solid',
                              scale: float = 0.2,
                              dash_length: float = 0.02,
                              gap_length: float = 0.01) -> List[o3d.geometry.Geometry]:
        """
        Build an Open3D LineSet for a camera, so you can see its position
        & view-frustum in world space.

        Args:
            cam (o3d.camera.PinholeCameraParameters): Camera parameters.
            mesh (Optional[o3d.geometry.TriangleMesh]): Mesh to align the frustum with.
            image_size (Tuple[int, int]): Size of the images (width, height).
            color (List[float]): RGB color for the frustum.
            line (Literal['solid', 'dashed']): Line style.
            scale (float): Scale factor for frustum size.
            dash_length (float): Length of dash segments (if dashed).
            gap_length (float): Gap between dashes (if dashed).

        Returns:
            o3d.geometry.LineSet: The camera frustum as a LineSet.
        """
        if mesh is not None and line == 'solid':
            render = cls.render_mesh_to_cameras(mesh, [cam], image_size=image_size)['rgb'][0]  # (H, W, 3) float with max 1.0
        else:
            render = None
        intrinsics, extrinsic = cam.intrinsic.intrinsic_matrix, cam.extrinsic
        w, h = image_size
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        d = scale

        # corners in camera frame: center + 4 near-plane points
        corners_cam = np.array([
            [0, 0, 0],  # camera center
            [(0 - cx) / fx * d, (0 - cy) / fy * d, d],  # tl
            [(w - cx) / fx * d, (0 - cy) / fy * d, d],  # tr
            [(w - cx) / fx * d, (h - cy) / fy * d, d],  # br
            [(0 - cx) / fx * d, (h - cy) / fy * d, d],  # bl
        ])
        # transform to world
        corners_h = np.hstack([corners_cam, np.ones((5, 1))])  # (5,4)
        world_pts = (extrinsic @ corners_h.T).T[:, :3]

        # connect center→corners & corners→corners
        line_indices = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1],
        ]

        frustum = o3d.geometry.LineSet()
        frustum.points = o3d.utility.Vector3dVector(world_pts)
        if line == 'solid':
            frustum.lines = o3d.utility.Vector2iVector(line_indices)
            frustum.colors = o3d.utility.Vector3dVector([color for _ in line_indices])
        elif line == 'dashed':
            dashed_points, dashed_lines = [], []
            for start_idx, end_idx in line_indices:
                start, end = world_pts[start_idx], world_pts[end_idx]
                line_vec = end - start
                line_length = np.linalg.norm(line_vec)
                num_dashes = int(np.floor(line_length / (dash_length + gap_length)))
                dash_dir = line_vec / line_length

                for i in range(num_dashes):
                    seg_start = start + dash_dir * (i * (dash_length + gap_length))
                    seg_end = seg_start + dash_dir * dash_length

                    dashed_points.extend([seg_start, seg_end])
                    dashed_lines.append([len(dashed_points) - 2, len(dashed_points) - 1])

            frustum.points = o3d.utility.Vector3dVector(dashed_points)
            frustum.lines = o3d.utility.Vector2iVector(dashed_lines)
            frustum.colors = o3d.utility.Vector3dVector([color for _ in dashed_lines])

        geometries = [frustum]

        if render is not None:
            # build a textured plane at the near‐plane (corners_cam[1:5])
            plane_verts = corners_cam[1:5]
            plane_mesh = o3d.geometry.TriangleMesh()
            plane_mesh.vertices = o3d.utility.Vector3dVector(plane_verts)
            plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])

            # UVs for two triangles: tl=(0,0), tr=(1,0), br=(1,1), bl=(0,1)
            uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
            tri_uvs = [uvs[0], uvs[1], uvs[2], uvs[0], uvs[2], uvs[3]]
            plane_mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
            plane_mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])

            # convert render to uint8 image
            img = (np.clip(render, 0.0, 1.0) * 255).astype(np.uint8)
            tex = o3d.geometry.Image(img)
            plane_mesh.textures = [tex]

            # transform plane into world coords
            plane_mesh.transform(extrinsic)
            geometries.append(plane_mesh)

        return geometries

    @classmethod
    def create_cameras_from_calibration(cls,
                                        calibration_session_path: Path,
                                        calibration_method: Literal['Caliscope', 'MultiCamCalib'] = 'MultiCamCalib',
                                        image_size: Tuple[int, int] = (1024, 1024),
                                        cam_idx: Union[Literal['all'], List['int']] = 'all',
                                        align_to_yup: bool = False,
                                        lookat_point: Optional[Tuple[float, float, float]] = None,
                                        return_calibration_data: bool = False,
                                        override_calibration: Optional[Path] = None
                                        ) -> Union[List[o3d.camera.PinholeCameraParameters], Tuple[List[o3d.camera.PinholeCameraParameters], CalibrationData, torch.Tensor]]:
        """
        Create Open3D camera parameters from a calibration session.

        Args:
            calibration_session_path (Path): Path to the calibration session directory.
            calibration_method (Literal['Caliscope', 'MultiCamCalib']): Method used for calibration.
            image_size (Tuple[int, int]): Size of the images to be used for the cameras (width, height).
            cam_idx (Union[Literal['all'], List[int]]): Indices of cameras to use. If 'all', use all cameras.
            align_to_yup (bool): If True, align cameras to Y-up orientation.
            lookat_point (Optional[Tuple[float, float, float]]): If provided, the cameras will be aligned to look at this point.
            return_calibration_data (bool): If True, return the calibration data along with the camera parameters.
            override_calibration (Optional[Path]): Path to a custom calibration file to override the default calibration.

        Returns:
            List[o3d.camera.PinholeCameraParameters]: List of camera parameters.
        """
        from utils.calib import CalibrationData
        # create cameras
        if override_calibration is not None:
            with open(override_calibration, 'rb') as f:
                data = pickle.load(f)
            calibration_data = CalibrationData.from_pkl(data)
        else:
            calibration_data = CalibrationData.from_session(
                calibration_session_path,
                idx=cam_idx,
                method=calibration_method
            )
        # alter the intrinsics to be 1K
        calibration_data.intrinsics[:, 0, 0] = image_size[1] * FOCAL_RATIO
        calibration_data.intrinsics[:, 1, 1] = image_size[0] * FOCAL_RATIO
        calibration_data.intrinsics[:, 0, 2] = image_size[1] * 0.5
        calibration_data.intrinsics[:, 1, 2] = image_size[0] * 0.5 + 25
        calibration_data.image_size = torch.tensor((image_size,)).repeat(len(cam_idx), 1).int()
        gt_cameras, mean_look_at_point, _ = calibration_data.create_cameras(mean_idx=[_ for _ in cam_idx if _ != 5], lib='open3d')
        if align_to_yup:
            # Align cameras to Y-up orientation
            if cam_idx == 'all':
                cam_idx = list(range(len(gt_cameras)))
            plane_normal = cls.create_camera_plane(gt_cameras, plane_idx=[_ for _ in cam_idx if _ not in [5, 6]])[0]
            aligned_extrinsics = cls.align_cameras_via_mean_look(
                extrinsics=[cam.extrinsic for cam in gt_cameras],
                mean_look=np.asarray(mean_look_at_point).squeeze(),
                plane_normal=np.asarray(plane_normal).squeeze(),
                lookat_point=np.asarray(lookat_point).squeeze() if lookat_point is not None else np.array(mean_look_at_point).squeeze()
            )
            # Update the camera parameters with the new extrinsics
            for i, aligned_extrinsic in enumerate(aligned_extrinsics):
                gt_cameras[i].extrinsic = aligned_extrinsic
                calibration_data.extrinsics_c2w[i] = torch.from_numpy(aligned_extrinsic).float()
        new_lookat_point = np.asarray(lookat_point).squeeze() if lookat_point is not None else np.asarray(mean_look_at_point).squeeze()
        if return_calibration_data:
            return gt_cameras, calibration_data, torch.from_numpy(new_lookat_point).float()
        return gt_cameras

    @classmethod
    def rotate_cameras_to_front(cls, extrinsics, mean_look, ref_idx=None):
        """
        Rotate your cameras around the mean look-at point so that the “middle” camera
        lies on the +X axis in the ground plane.

        Args:
            extrinsics : list of (4×4) numpy arrays, each camera→world matrix
            mean_look  : (3,) numpy array, the common look-at point in world coords
            ref_idx    : int or None, index of the reference camera (default: middle one)

        Returns:
            new_extrs  : list of (4×4) numpy arrays, the rotated extrinsics
        """
        from scipy.spatial.transform import Rotation as ScipyRotation
        if ref_idx is None:
            # pick the middle camera if none specified
            ref_idx = len(extrinsics) // 2

        # 1) Get reference camera position and form the ground-plane vector
        E_ref = extrinsics[ref_idx]
        t_ref = E_ref[:3, 3]
        v = t_ref - mean_look
        v[1] = 0  # project onto y=0 plane
        v /= np.linalg.norm(v)  # normalize
        theta = np.arctan2(v[0], v[1])  # angle from +X

        # 2) Build a rotation about Y by –θ
        R_align = ScipyRotation.from_rotvec([0, -2 * theta, 0]).as_matrix()

        # 3) Apply to each camera: rotate around mean_look
        new_extrs = []
        for E in extrinsics:
            R_cam = E[:3, :3]
            t_cam = E[:3, 3]

            # rotate orientation
            R_new = R_align.dot(R_cam)
            # rotate position about mean_look
            t_new = R_align.dot(t_cam - mean_look) + mean_look

            E2 = np.eye(4)
            E2[:3, :3] = R_new
            E2[:3, 3] = t_new
            new_extrs.append(E2)

        return new_extrs

    @classmethod
    def align_cameras_via_mean_look(cls, extrinsics, mean_look, plane_normal, lookat_point: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Align cameras by:
            1) Center cameras at mean_look
            2) Rotate so plane_normal → +Y
            3) Translate so mean_look → desired_look

        Args:
            extrinsics (List[np.ndarray]): List of (4x4) camera→rig-world matrices.
            mean_look (np.ndarray): Mean look-at point in rig-world coordinates (shape: (3,)).
            plane_normal (np.ndarray): Normal vector of the fitted plane in rig-world coordinates (shape: (3,)).
            lookat_point (Optional[np.ndarray]): Desired look-at point in rig-world coordinates (shape: (3,)). If None, uses mean_look.
        Returns:
            List[np.ndarray]: List of aligned camera extrinsics (4x4 matrices).
        """
        n = plane_normal / np.linalg.norm(plane_normal)
        up = np.array([0.0, -1.0, 0.0])
        v = np.cross(n, up)
        s = np.linalg.norm(v)
        c = np.dot(n, up)
        if s < 1e-8:
            R_align = np.eye(3)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R_align = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s * s))

        new_extrs = []
        for E in extrinsics:
            R_cam = E[:3, :3]
            t_cam = E[:3, 3]

            # 1) center at mean_look
            t0 = t_cam - mean_look

            # 2) rotate both orientation and centered translation
            R2 = R_align.dot(R_cam)
            t1 = R_align.dot(t0)

            # 3) translate so that mean_look (which is now at origin) → desired_look
            t2 = t1 + mean_look
            if lookat_point is not None:
                t2 += lookat_point - mean_look

            E2 = np.eye(4)
            E2[:3, :3] = R2
            E2[:3, 3] = t2
            new_extrs.append(E2)

        # Ensure cameras are looking at the desired look-at point from the +x
        new_extrs_front = cls.rotate_cameras_to_front(new_extrs, lookat_point if lookat_point is not None else mean_look, ref_idx=len(new_extrs) // 2)
        return new_extrs_front

    @classmethod
    def create_plane_mesh(cls,
                          cam_plane_normal: np.ndarray,
                          cam_plane_d: np.ndarray,
                          width: float = 2.0,
                          height: float = 2.0,
                          normal_length: float = 0.5,
                          plane_color: list = [0.0, 1.0, 0.0],
                          normal_color: list = [1.0, 0.0, 0.0]):
        """
        Visualize a finite patch of the plane defined by
            a·x + b·y + c·z + d = 0
        and draw its normal vector.

        Args:
            cam_plane_normal: (3,) array [a, b, c]
            cam_plane_d:      scalar d
            width:            patch width (meters)
            height:           patch height (meters)
            normal_length:    length of the normal arrow
            plane_color:      [r,g,b] for the plane
            normal_color:     [r,g,b] for the normal
        Returns:
            plane_mesh, normal_vec  (Open3D geometries)
        """
        # 1) Unpack and normalize the normal
        n = cam_plane_normal.astype(float)
        norm2 = np.dot(n, n)
        if norm2 == 0:
            raise ValueError("Zero plane normal")
        # compute a point on the plane: solve n·P + d = 0  ⇒  P = -(d / ||n||²) * n
        P0 = -(cam_plane_d / norm2) * n
        u = n.copy()

        # 2) Build two in‐plane orthonormal axes (u, v)
        # pick a tangent direction not parallel to n
        if abs(n[0]) < abs(n[1]) and abs(n[0]) < abs(n[2]):
            tangent = np.array([1.0, 0.0, 0.0])
        elif abs(n[1]) < abs(n[2]):
            tangent = np.array([0.0, 1.0, 0.0])
        else:
            tangent = np.array([0.0, 0.0, 1.0])
        u = np.cross(n, tangent)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        # 3) Scale u, v to half‐width/height
        u *= (width / 2.0)
        v *= (height / 2.0)

        # 4) Corners of the rectangular patch
        corners = np.vstack([
            P0 + u + v,
            P0 + u - v,
            P0 - u - v,
            P0 - u + v,
        ])

        # 5) Make the TriangleMesh for the plane
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(corners)
        plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
        plane_mesh.paint_uniform_color(plane_color)

        # 6) Build the normal arrow as a LineSet
        p1 = P0 + (n / np.linalg.norm(n)) * normal_length
        line_pts = np.vstack([P0, p1])
        normal_vec = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_pts),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        normal_vec.colors = o3d.utility.Vector3dVector([normal_color])
        return plane_mesh, normal_vec

    @classmethod
    def visualize_mesh(cls,
                       mesh: o3d.geometry.TriangleMesh,
                       image_path: Path,
                       smpl_parameters: Optional[Dict[str, np.ndarray]] = None,
                       cameras: Optional[List[o3d.camera.PinholeCameraParameters]] = None,
                       width: int = 1024, height: int = 1024,
                       window_visible: bool = False,
                       visualize_authors_cams_from: Optional[Path] = None) -> Optional[List[o3d.camera.PinholeCameraParameters]]:
        """
        Visualize the alignment of an Open3D mesh and SMPL vertices, and save the view as an image.

        Args:
            mesh (o3d.geometry.TriangleMesh): The scanned mesh.
            image_path (str): Local path where the rendered image will be saved (e.g., "output.png").
            smpl_parameters (Dict[str, np.ndarray], optional): SMPL parameters including 'betas', 'expression', 'thetas', 'global_orient',
                'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose'. If None, SMPL mesh will not be rendered.
            cameras (List[o3d.camera.PinholeCameraParameters], optional): List of camera parameters for visualization. The cameras will be represented as frustums in the scene.
            width (int): Width of the offscreen render window in pixels.
            height (int): Height of the offscreen render window in pixels.
            window_visible (bool): If True, the Open3D window will be visible during rendering.
            visualize_authors_cams_from (Path, optional): If provided, will visualize the cameras from the specified path.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=window_visible, width=width, height=height)
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        if smpl_parameters is not None:
            smpl_mesh = cls.smpl_forward(smpl_parameters, return_spline_dir=False)
            smpl_mesh = cls.process_mesh(smpl_mesh, human_height_std=0.0, random_rotation=False, random_translation=False)
            vis.add_geometry(smpl_mesh)
        vis.add_geometry(mesh)
        if cameras is not None:
            # vis cameras
            for cam_idx_s0, cam in enumerate(cameras):
                cam_idx = cam_idx_s0 + 1
                frustum_geometries = cls.create_camera_frustum(cam, mesh=mesh, image_size=(width, height), line='solid' if cam_idx in [10, 11] else 'dashed')
                for g in frustum_geometries:
                    vis.add_geometry(g)
            # # vis cameras plane
            # cam_plane_normal, cam_plane_ofs = cls.create_camera_plane(cameras, plane_idx='all')  # Create a camera plane for visualization
            # plane_mesh, normal_vec = cls.create_plane_mesh(
            #     cam_plane_normal=np.asarray(cam_plane_normal).squeeze(),
            #     cam_plane_d=np.asarray(cam_plane_ofs).squeeze().item(),
            #     width=5.0, height=5.0,
            #     normal_length=1.0,
            #     plane_color=[1.0, 1.0, 0.0],
            #     normal_color=[0.0, 1.0, 0.0]
            # )
            # vis.add_geometry(plane_mesh)
            # vis.add_geometry(normal_vec)
        author_cams = None
        if visualize_authors_cams_from is not None:
            # read cameras
            authors_cam_dirs = sorted((visualize_authors_cams_from / 'train' / 'parm').glob('0000_*'), key=lambda x: int(x.stem.split('_')[1]))
            authors_intrinsics = [np.load(acd / '0_intrinsic.npy') for acd in authors_cam_dirs]
            authors_extrinsics = [np.load(acd / '0_extrinsic.npy') for acd in authors_cam_dirs]
            # convert to Open3D cameras
            author_cams = []
            for authors_cam_idx, (intri, extri) in enumerate(zip(authors_intrinsics, authors_extrinsics)):
                ac = o3d.camera.PinholeCameraParameters()
                ac.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=1024, height=1024,
                    fx=intri[0, 0].item(), fy=intri[1, 1].item(),
                    cx=intri[0, 2].item(), cy=intri[1, 2].item()
                )
                extri_4x4 = np.eye(4)
                extri_4x4[:3, :3] = extri[:3, :3]
                extri_4x4[:3, 3] = extri[:3, 3]
                ac.extrinsic = np.linalg.inv(extri_4x4)
                author_cams.append(ac)
                frustum_geometries = cls.create_camera_frustum(ac, mesh=mesh, image_size=(width, height), color=[0.0, 0.0, 1.0], line='solid' if authors_cam_idx in [14, 15] else 'dashed')
                for g in frustum_geometries:
                    vis.add_geometry(g)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0,  # length of each axis arrow
            origin=[0, 0, 0]
        )
        vis.add_geometry(axes)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(image_path), do_render=True)
        if window_visible:
            vis.run()
        vis.destroy_window()
        return author_cams

    # noinspection PyTypeChecker
    @classmethod
    def read_mesh(cls, canonicalize: bool = False, return_full_smpl_params: bool = False, **reader_kwargs) -> Tuple[o3d.geometry.TriangleMesh, torch.Tensor | None]:
        """
        Read the mesh file using Kaolin.
        """
        model_path, smpl_path = cls.read_model(**reader_kwargs)
        mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(str(model_path), True)
        if smpl_path is None:
            return mesh, None
        # SMPL model
        with open(smpl_path, 'rb') as f:
            smpl_params = pickle.load(f)
        if canonicalize:
            from scipy.spatial.transform import Rotation as ScipyRotation
            # 1) Compute the inverse SMPL rotation
            rot_vec = smpl_params['global_orient'].squeeze()  # (3,) axis-angle
            R_mat = ScipyRotation.from_rotvec(rot_vec).as_matrix()  # rotates rest→posed
            R_inv = R_mat.T  # posed -> rest
            # 2) Rotate mesh about its center to make it upright
            mesh.rotate(R_inv, center=mesh.get_center())
            smpl_params['global_orient'] = np.zeros_like(smpl_params['global_orient'])  # reset global orientation
            # fix remaining offset from +Y axis
            smpl_mesh, spline_dir = cls.smpl_forward(smpl_params, return_spline_dir=True)
            vz, vy = spline_dir[2], spline_dir[1]
            theta = np.arctan2(vz, vy)  # small tilt angle in radians
            R_align = ScipyRotation.from_rotvec([-0.5 * theta, 0.0, 0.0]).as_matrix()
            mesh.rotate(R_align, center=mesh.get_center())
            smpl_params['global_orient'] = ScipyRotation.from_matrix(R_align).as_rotvec()
            # spline_dir2 = cls.smpl_forward(smpl_params2, return_spline_dir=True)[1]
            # print(spline_dir, spline_dir2)
            # exit(0)
            # 3) Scale/translate the mesh to fit with SMPL
            scan_bb = mesh.get_axis_aligned_bounding_box()
            smpl_bb = smpl_mesh.get_axis_aligned_bounding_box()
            scan_size = scan_bb.get_extent()  # [dx, dy, dz]
            smpl_size = smpl_bb.get_extent()
            mesh.scale(smpl_size.max() / scan_size.max(), scan_bb.get_center())
            scan_bb = mesh.get_axis_aligned_bounding_box()  # recalculate bounding box after scaling
            mesh.translate(smpl_bb.get_center() - scan_bb.get_center())
        # mesh.compute_vertex_normals()
        if not return_full_smpl_params:
            global_orient = torch.from_numpy(smpl_params['global_orient']).float().reshape(1, 3)
            return mesh, global_orient
        return mesh, smpl_params

    @classmethod
    def process_mesh(cls,
                     mesh: o3d.geometry.TriangleMesh,
                     human_height_mean: float = 1.80,
                     human_height_std: float = 0.1,
                     random_rotation: bool = False,
                     random_translation: bool = False) -> o3d.geometry.TriangleMesh:
        """
        Process the mesh using Kaolin.
        """
        # Get vertices and normalize
        verts = torch.from_numpy(np.asarray(mesh.vertices)).float()
        #   - height normalization
        vy_min, vy_max = verts[:, 1].min(), verts[:, 1].max()
        human_height = human_height_mean + (2 * random.random() - 1) * human_height_std  # Random height between 1.70 and 1.90
        verts /= (vy_max - vy_min) / human_height
        verts[:, 1] -= verts[:, 1].min()
        verts = verts.cpu().numpy()
        # randomly move the scan
        if random_translation:
            move_range = 0.1 if human_height < 1.80 else 0.05
            delta_x = np.max(verts[:, 0]) - np.min(verts[:, 0])
            delta_z = np.max(verts[:, 2]) - np.min(verts[:, 2])
            if delta_x > 1.0 or delta_z > 1.0:
                move_range = 0.01
            verts[:, 0] += np.random.uniform(-move_range, move_range, 1)
            verts[:, 2] += np.random.uniform(-move_range, move_range, 1)
        # randomly rotate the scan
        if random_rotation:
            angle = np.random.uniform(-np.pi / 4, np.pi / 4)  # Random rotation angle between -45 and 45 degrees
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.array([[cos_angle, 0, sin_angle],
                                        [0, 1, 0],
                                        [-sin_angle, 0, cos_angle]])
            verts = verts @ rotation_matrix.T
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        return mesh

    @classmethod
    def create_camera_plane(cls, cameras: List[o3d.camera.PinholeCameraParameters], plane_idx: List[int] | Literal['all'] = 'all') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fits a plane to the camera translation vectors using SVD.

        Assumes self.tvecs is a torch.Tensor of shape (N, 3).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - plane_normal: Normalized normal vector of the best-fit plane (shape: (3,)).
                - offset: Offset 'd' such that n.dot(p) + d = 0 for points p on the plane (scalar).
        """
        tvecs = torch.stack([torch.from_numpy(cam.extrinsic[..., :3, 3]).squeeze() for cam in cameras], dim=0).float()
        if plane_idx != 'all':
            # noinspection PyTypeChecker
            tvecs = tvecs[plane_idx]
        if tvecs is None or tvecs.shape[0] < 3:
            # Return a default plane (e.g., Z=0) if not enough points
            dev = tvecs.device if tvecs is not None else torch.device('cpu')
            dtype = tvecs.dtype if tvecs is not None else torch.float32
            return torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype), torch.tensor(0.0, device=dev, dtype=dtype)

        centroid = torch.mean(tvecs, dim=0)
        centered_tvecs = tvecs - centroid

        try:
            # SVD on centered points: U S Vh = X_centered
            # The rows of Vh are principal directions, sorted high-to-low variance.
            # The last row corresponds to the smallest singular value / minimum variance direction (plane normal).
            U, S, Vh = torch.linalg.svd(centered_tvecs, full_matrices=False)
            plane_normal = Vh[-1, :]
        except torch.linalg.LinAlgError:
            # Handle SVD failure (e.g., all points identical)
            dev = tvecs.device
            dtype = tvecs.dtype
            plane_normal = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype)  # Default normal

        # Ensure the normal has non-zero magnitude before calculating offset
        if torch.linalg.norm(plane_normal) < 1e-6:
            dev = tvecs.device
            dtype = tvecs.dtype
            plane_normal = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype)  # Default normal

        # Calculate offset d = -n . centroid
        offset = -torch.dot(plane_normal, centroid)
        return plane_normal, offset

    @classmethod
    def align_mesh_with_cameras(cls, mesh: o3d.geometry.TriangleMesh, global_orient: Optional[torch.Tensor], cameras: List[o3d.camera.PinholeCameraParameters], plane_idx: List[int] | Literal['all'] = 'all', mean_look_at_point: Optional[torch.Tensor] = None, over_525: bool = False) -> o3d.geometry.TriangleMesh:
        """
        Align the mesh with the cameras.
        """
        verts_processed = torch.from_numpy(np.asarray(mesh.vertices)).float()
        # if over_525:
        #     from pytorch3d.transforms import RotateAxisAngle
        #     rotator = RotateAxisAngle(90, 'Z').compose(RotateAxisAngle(45, 'X'))
        #     verts_processed = rotator.transform_points(verts_processed[None])[0]
        # if global_orient is not None:
        #     global_orient = rotator.transform_points(global_orient[None])[0]
        #  - make vertices look forward
        verts_processed = torch.cat([
            verts_processed, torch.ones(verts_processed.shape[0], 1, device=verts_processed.device)
        ], dim=1) @ cls.MeshTransformMatrix
        denom = verts_processed[..., 3:]  # denominator
        denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
        denom = denom_sign * torch.clamp(denom.abs(), 1e-8)
        verts_processed = verts_processed[..., :3] / denom
        #   - rotate mesh to y-orientation
        z_sign = 1
        if global_orient is not None:
            verts_processed = Rotate(pytorch3d.transforms.axis_angle_to_matrix(global_orient).transpose(-1, -2)).transform_points(verts_processed)
        if global_orient.squeeze()[1] > math.pi / 2:
            z_sign = -1
        #  - get camera plane normal
        camera_plane_normal = cls.create_camera_plane(cameras, plane_idx=plane_idx)[0]
        # - get torso vertices
        # #  - get torso plane normal
        # torso_plane_normal = cls.fit_torso_plane(verts_processed)[0]
        # #  - align mesh so that it is vertically upright wrt camera plane but also looking towards camera 5 (cameras[4])
        # verts_processed = cls.align_mesh_orthogonal_and_lookat(verts_processed, torso_plane_normal, camera_plane_normal, torch.from_numpy(cameras[4].extrinsic[:3, 3]).to(device=verts_processed.device))
        # verts_processed = cls.align_mesh_to_vector(verts_processed, vector=torso_plane_normal)
        # #  - align mesh to camera plane normal
        verts_processed = cls.align_mesh_to_vector(verts_processed, z_sign * camera_plane_normal, mesh_axis=0, over_525=over_525)
        # #   - align tetriary axis with camera's up vector
        # verts_processed = cls.align_mesh`_tta_to_cams_out(verts_processed, cameras)
        if mean_look_at_point is not None:
            #   - place mesh at the cameras' look-at point
            verts_processed += mean_look_at_point.to(verts_processed.device) - torch.mean(verts_processed, dim=0)
            # #   - bring mesh closer to the cameras
            # offset_vector = mean_look_at_point.to(verts_processed.device) - torch.from_numpy(np.mean([cameras[c].extrinsic[:3, 3] for c in [3, 4, 6, 7]], axis=0)).to(device=verts_processed.device)
            # offset_vector = offset_vector / torch.linalg.norm(offset_vector)
            # verts_processed -= 0.5 * offset_vector
        mesh.vertices = o3d.utility.Vector3dVector(verts_processed.cpu().numpy())
        return mesh

    @classmethod
    def align_mesh_tta_to_cams_out(cls, verts_processed: torch.Tensor, cameras: List[o3d.camera.PinholeCameraParameters]):
        # Get the camera vector, pointing outwards into the scene
        tvecs = torch.stack([torch.from_numpy(cam.extrinsic[..., :3, 3]).squeeze() for cam in cameras], dim=0).float()
        cams_out_end = (tvecs[0] + tvecs[-1]) / 2
        cams_out_start = tvecs[4]
        cams_out_axis = cams_out_end - cams_out_start
        cams_out_axis = cams_out_axis / (torch.linalg.norm(cams_out_axis) + 1e-8)  # Normalize
        # Get the tetriary axis of the mesh vertices
        return cls.align_mesh_to_vector(verts_processed, vector=cams_out_axis, mesh_axis=1)

    @classmethod
    def render_mesh_to_cameras(cls, mesh: o3d.geometry.Geometry, cameras: List[o3d.camera.PinholeCameraParameters], image_size: Tuple[int, int] = (2048, 2048), vis=None) -> dict:
        H, W = image_size
        if vis is None:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=W, height=H, visible=False)

        vis.clear_geometries()
        vis.add_geometry(mesh)

        # Initial dummy render
        vis.poll_events()
        vis.update_renderer()

        results = {'rgb': [], 'depth': [], 'mask': [], 'intrinsic': [], 'extrinsic': []}
        # close_kernel = np.ones((3, 3), np.uint8)
        for cam in cameras:
            # Clone and modify intrinsics
            cam = o3d.camera.PinholeCameraParameters(cam)
            cam.intrinsic.width = int(W)
            cam.intrinsic.height = int(H)
            extri = np.linalg.inv(np.asarray(cam.extrinsic))
            cam.extrinsic = extri

            vis.reset_view_point()
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

            # Force hard reset
            vis.poll_events()
            vis.update_renderer()

            results['rgb'].append(np.asarray(vis.capture_screen_float_buffer()))
            results['depth'].append(np.asarray(vis.capture_depth_float_buffer()))
            mask = np.logical_and(results['depth'][-1] > 0, results['depth'][-1] < 1e4).astype(np.float32)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
            results['mask'].append(mask)
            results['intrinsic'].append(cam.intrinsic.intrinsic_matrix)
            results['extrinsic'].append(cam.extrinsic)
        return results

    @classmethod
    def render_mesh_to_cameras_offscreen(cls, mesh: o3d.geometry.TriangleMesh, cameras: List[o3d.camera.PinholeCameraParameters], image_size: Tuple[int, int] = (1024, 1024), renderer=None) -> dict:
        H, W = image_size
        if renderer is None:
            renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)

        # Set up material and rendering
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mtl)
        renderer.scene.set_background([0, 0, 0, 0])  # RGBA

        results = {'rgb': [], 'depth': [], 'mask': [], 'intrinsic': [], 'extrinsic': []}
        for cam in cameras:
            # Clone and modify intrinsics
            cam = o3d.camera.PinholeCameraParameters(cam)
            cam.intrinsic.width = W
            cam.intrinsic.height = H
            extri = np.linalg.inv(np.asarray(cam.extrinsic))
            cam.extrinsic = extri

            # Set camera parameters
            renderer.setup_camera(cam.intrinsic, cam.extrinsic)

            # Render
            img = renderer.render_to_image()
            depth = renderer.render_to_depth_image()

            # Convert to numpy arrays
            rgb_array = np.asarray(img)
            depth_array = np.asarray(depth)

            # Process results
            results['rgb'].append(rgb_array)
            results['depth'].append(depth_array)
            mask = np.logical_and(depth_array > 0, depth_array < 1e4).astype(np.float32)
            results['mask'].append(mask)
            results['intrinsic'].append(cam.intrinsic.intrinsic_matrix)
            results['extrinsic'].append(cam.extrinsic)
        # Clean up
        renderer.scene.clear_geometry()
        return results

    @classmethod
    def copy_camera_to_hr(cls, cameras: List[o3d.camera.PinholeCameraParameters], image_size: Tuple[int, int] = (2048, 2048)) -> List[o3d.camera.PinholeCameraParameters]:
        """
        Copy the camera parameters to a higher resolution.
        """
        new_cameras = []
        H, W = image_size
        for cam in cameras:
            new_cam = o3d.camera.PinholeCameraParameters()
            new_cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=W, height=H,
                fx=W * FOCAL_RATIO,
                fy=H * FOCAL_RATIO,
                cx=W * 0.5,
                cy=H * 0.5 + 25
            )
            new_cam.extrinsic = cam.extrinsic.copy()
            new_cameras.append(new_cam)
        return new_cameras

    @classmethod
    def depth2pts(cls, depth, extrinsic, intrinsic):
        # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
        rot = extrinsic[:3, :3]
        trans = extrinsic[:3, 3:]
        S, S = depth.shape

        y, x = np.meshgrid(np.linspace(0.5, S - 0.5, S), np.linspace(0.5, S - 0.5, S), indexing='ij')
        pts_2d = np.stack([x, y, np.ones_like(x)], axis=-1)  # H W 3

        pts_2d[..., 2] = 1.0 / (depth + 1e-8)
        pts_2d[..., 0] -= intrinsic[0, 2]
        pts_2d[..., 1] -= intrinsic[1, 2]
        pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
        pts_2d = np.concatenate([pts_2d_xy, pts_2d[..., 2:]], axis=-1)

        pts_2d[..., 0] /= intrinsic[0, 0]
        pts_2d[..., 1] /= intrinsic[1, 1]
        pts_2d = pts_2d.reshape(-1, 3).T
        pts = rot.T @ pts_2d - rot.T @ trans
        return pts.T.reshape((S, S, 3))

    @classmethod
    def pts2depth(cls, ptsmap, extrinsic, intrinsic):
        S, S, _ = ptsmap.shape
        pts = ptsmap.reshape((-1, 3)).T
        calib = intrinsic @ extrinsic
        pts = calib[:3, :3] @ pts
        pts = pts + calib[:3, 3:4]
        pts[:2, :] /= (pts[2:, :] + 1e-8)
        depth = 1.0 / (pts[2, :].reshape((S, S)) + 1e-8)
        return depth

    @classmethod
    def stereo_pts2flow(cls, pts0, pts1, rectify0, rectify1, Tf_x):
        new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y = rectify0
        new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y = rectify1
        new_depth0 = cls.pts2depth(pts0, new_extr0, new_intr0)
        new_depth1 = cls.pts2depth(pts1, new_extr1, new_intr1)
        new_depth0 = cv2.remap(new_depth0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        new_depth1 = cv2.remap(new_depth1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

        offset0 = new_intr1[0, 2] - new_intr0[0, 2]
        disparity0 = -new_depth0 * Tf_x
        flow0 = offset0 - disparity0

        offset1 = new_intr0[0, 2] - new_intr1[0, 2]
        disparity1 = -new_depth1 * (-Tf_x)
        flow1 = offset1 - disparity1

        flow0[new_depth0 < 0.05] = 0
        flow1[new_depth1 < 0.05] = 0

        return flow0, flow1

    @classmethod
    def get_rectified_stereo_data(cls, main_view_data, ref_view_data):
        img0, mask0, intr0, extr0, pts0 = main_view_data
        img1, mask1, intr1, extr1, pts1 = ref_view_data

        H, W = img0.shape[:2]
        r0, t0 = extr0[:3, :3], extr0[:3, 3:]
        r1, t1 = extr1[:3, :3], extr1[:3, 3:]

        E0 = np.eye(4)
        E0[:3, :3], E0[:3, 3:] = r0.T, -r0.T @ t0
        E1 = np.eye(4)
        E1[:3, :3], E1[:3, 3:] = r1, t1
        E_01 = E1 @ E0
        R_01, T_01 = E_01[:3, :3], E_01[:3, 3]
        R_to_use, T_to_use = R_01, T_01.flatten()

        dist0 = np.zeros(5, dtype=np.float32)
        dist1 = np.zeros(5, dtype=np.float32)

        intr_1, dist_1 = intr0, dist0
        intr_2, dist_2 = intr1, dist1
        R_rect0, R_rect1, P_rect0, P_rect1, _, _, _ = cv2.stereoRectify(
            intr_1, dist_1, intr_2, dist_2, (W, H), R_to_use, T_to_use, flags=0
        )

        new_extr0 = R_rect0 @ extr0
        new_intr0 = P_rect0[:3, :3]
        new_extr1 = R_rect1 @ extr1
        new_intr1 = P_rect1[:3, :3]
        Tf_x = P_rect1[0, 3]

        camera = {
            'intr0': new_intr0,
            'intr1': new_intr1,
            'extr0': new_extr0,
            'extr1': new_extr1,
            'Tf_x': Tf_x
        }

        rectify_mat0_x, rectify_mat0_y = cv2.initUndistortRectifyMap(intr0, dist0, R_rect0, P_rect0, (W, H), cv2.CV_32FC1)
        new_img0 = cv2.remap(img0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        new_mask0 = cv2.remap(mask0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_NEAREST)

        rectify_mat1_x, rectify_mat1_y = cv2.initUndistortRectifyMap(intr1, dist1, R_rect1, P_rect1, (W, H), cv2.CV_32FC1)
        new_img1 = cv2.remap(img1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
        new_mask1 = cv2.remap(mask1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_NEAREST)

        stereo_data = {
            'img0': new_img0,
            'mask0': new_mask0,
            'img1': new_img1,
            'mask1': new_mask1,
            'camera': camera,
        }

        if pts0 is not None:
            rectify0 = new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y
            rectify1 = new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y
            flow0, flow1 = cls.stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x)
            kernel = np.ones((3, 3), dtype=np.uint8)
            flow_eroded, valid_eroded = [], []
            for (flow, new_mask) in [(flow0, new_mask0), (flow1, new_mask1)]:
                valid = (new_mask.copy() / 255.0).astype(np.float32)
                if valid.ndim == 3:
                    valid = valid[..., 0]
                valid = cv2.erode(valid, kernel, 1)
                valid[valid >= 0.66] = 1.0
                valid[valid < 0.66] = 0.0
                flow *= valid
                valid *= 255.0
                flow_eroded.append(flow)
                valid_eroded.append(valid)
            stereo_data.update({
                'flow0': flow_eroded[0],
                'valid0': valid_eroded[0].astype(np.uint8),
                'flow1': flow_eroded[1],
                'valid1': valid_eroded[1].astype(np.uint8)
            })
        return stereo_data

    @classmethod
    def save_stereo_data(cls, stereo_data, save_path: Path, model_idx: int, camera_pair_idx: int) -> None:
        cv2.imwrite(str(save_path / 'img' / f'{model_idx:04}_{camera_pair_idx:03}' / '0.jpg'), stereo_data['img0'])
        cv2.imwrite(str(save_path / 'img' / f'{model_idx:04}_{camera_pair_idx:03}' / '1.jpg'), stereo_data['img1'])
        cv2.imwrite(str(save_path / 'mask' / f'{model_idx:04}_{camera_pair_idx:03}' / '0.png'), stereo_data['mask0'])
        cv2.imwrite(str(save_path / 'mask' / f'{model_idx:04}_{camera_pair_idx:03}' / '1.png'), stereo_data['mask1'])
        np.save(str(save_path / 'flow' / f'{model_idx:04}_{camera_pair_idx:03}' / '0.npy'), stereo_data['flow0'].astype(np.float16))
        np.save(str(save_path / 'flow' / f'{model_idx:04}_{camera_pair_idx:03}' / '1.npy'), stereo_data['flow1'].astype(np.float16))
        cv2.imwrite(str(save_path / 'valid' / f'{model_idx:04}_{camera_pair_idx:03}' / '0.png'), stereo_data['valid0'])
        cv2.imwrite(str(save_path / 'valid' / f'{model_idx:04}_{camera_pair_idx:03}' / '1.png'), stereo_data['valid1'])
        with open(save_path / 'parm' / f'{model_idx:04}_{camera_pair_idx:03}' / '0_1.json', 'w') as json_fp:
            camera_data = copy.deepcopy(stereo_data['camera'])
            for key in camera_data.keys():
                camera_data[key] = camera_data[key].tolist()
            json.dump(camera_data, json_fp, indent=1)

    @classmethod
    def render_data(cls,
                    model_dir: Path,
                    output_dir: Path,
                    gt_cameras_all: List[o3d.camera.PinholeCameraParameters],
                    cameras_l_virtual_r_all: List[List[o3d.camera.PinholeCameraParameters]],
                    mean_look_at_point: torch.Tensor,
                    plane_idx: List[int] | Literal['all'] = 'all',
                    create_renders: bool = True,
                    create_rectified: bool = False,
                    store_inverse_depth: bool = True,
                    vis=None) -> None:
        """
        Render the mesh data.
        """
        # Read and process the mesh
        model_idx = int(model_dir.name)
        train_dir_name = model_dir.parent.name
        mesh, smpl_parameters = THumanUtilsO3d.read_mesh(
            model_idx=model_idx,
            is_train=train_dir_name == 'train',
            thuman_root=THUMAN_ROOT,
            return_full_smpl_params=True,
            canonicalize=True,
        )
        mesh = THumanUtilsO3d.process_mesh(mesh, random_translation=True, random_rotation=True)

        # Create cameras list
        # renderer_1k = o3d.visualization.rendering.OffscreenRenderer(width=gt_cameras_all[0].intrinsic.width, height=gt_cameras_all[0].intrinsic.height)
        # renderer_2k = o3d.visualization.rendering.OffscreenRenderer(width=hr_image_size[1], height=hr_image_size[0])
        for lri, (cameras_l_virtual_r) in enumerate(cameras_l_virtual_r_all):
            if create_renders:
                # get the rendering cameras
                #   - the first and last cameras are the source views
                render_cams = [cameras_l_virtual_r[0], cameras_l_virtual_r[-1]]
                #   - three novel viewpoints between source views
                virtual_cameras_lr = cameras_l_virtual_r[1:-1]
                render_cams += [
                    virtual_cameras_lr[np.random.choice(np.arange(0, len(virtual_cameras_lr) // 2), 1).item()],
                    virtual_cameras_lr[len(virtual_cameras_lr) // 2],
                    virtual_cameras_lr[np.random.choice(np.arange(len(virtual_cameras_lr) // 2, len(virtual_cameras_lr)), 1).item()]
                ]
                # render_cams_hr = cls.copy_camera_to_hr(render_cams[2:], image_size=hr_image_size)

                # render
                # renderings = THumanUtilsO3d.render_mesh_to_cameras_offscreen(mesh, render_cams, image_size=(gt_cameras_all[0].intrinsic.height, gt_cameras_all[0].intrinsic.width), renderer=renderer_1k)
                # renderings_hr = THumanUtilsO3d.render_mesh_to_cameras_offscreen(mesh, render_cams_hr, image_size=hr_image_size, renderer=renderer_2k)
                renderings = cls.render_mesh_to_cameras(mesh, render_cams, image_size=(gt_cameras_all[0].intrinsic.height, gt_cameras_all[0].intrinsic.width), vis=vis)
                # renderings_hr = cls.render_mesh_to_cameras(mesh, render_cams_hr, image_size=hr_image_size)

                # save
                for subdir in ['img', 'depth', 'mask', 'parm']:
                    (output_dir / train_dir_name / subdir / f'{model_idx:04d}_{lri:03d}').mkdir(parents=True, exist_ok=True)
                for c in range(len(render_cams)):
                    cv2.imwrite(str(output_dir / train_dir_name / 'img' / f'{model_idx:04d}_{lri:03d}' / f'{c}.jpg'),
                                (np.clip(renderings['rgb'][c] * renderings['mask'][c][..., None], 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1])
                    cv2.imwrite(str(output_dir / train_dir_name / 'mask' / f'{model_idx:04d}_{lri:03d}' / f'{c}.png'),
                                (np.clip(renderings['mask'][c], 0, 1) * 255.0 + 0.5).astype(np.uint8))
                    depth_c = renderings['depth'][c]
                    if store_inverse_depth:
                        depth_c[depth_c > 1e-8] = 1.0 / depth_c[depth_c > 1e-8]
                    else:
                        depth_c = np.clip(depth_c, 0, 2.0)  # clip to [0, 2] range, as we will multiply by 2**15 and store as uint16
                    cv2.imwrite(str(output_dir / train_dir_name / 'depth' / f'{model_idx:04d}_{lri:03d}' / f'{c}.png'),
                                (depth_c * 2**15).astype(np.uint16))
                    np.save(str(output_dir / train_dir_name / 'parm' / f'{model_idx:04d}_{lri:03d}' / f'{c}_extrinsic.npy'), renderings['extrinsic'][c])
                    np.save(str(output_dir / train_dir_name / 'parm' / f'{model_idx:04d}_{lri:03d}' / f'{c}_intrinsic.npy'), renderings['intrinsic'][c])
                # for c in range(len(render_cams_hr)):
                #     cv2.imwrite(str(output_dir / train_dir_name / 'img' / f'{model_idx:04d}_{lri:03d}' / f'{c + 2}_hr.jpg'),
                #                 (np.clip(renderings_hr['rgb'][c] * renderings_hr['mask'][c][..., None], 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1])

            # Save rectified data
            if create_rectified:
                rendered_output_dir = output_dir / train_dir_name
                rectified_output_dir = output_dir / 'rectified_local' / train_dir_name
                for subdir in ['img', 'mask', 'flow', 'valid', 'parm']:
                    (rectified_output_dir / subdir / f'{model_idx:04d}_{lri:03d}').mkdir(parents=True, exist_ok=True)
                # Setup paths
                img_path0 = rendered_output_dir / 'img' / f'{model_idx:04}_{lri:03}' / '0.jpg'
                img_path1 = rendered_output_dir / 'img' / f'{model_idx:04}_{lri:03}' / '1.jpg'
                mask_path0 = rendered_output_dir / 'mask' / f'{model_idx:04}_{lri:03}' / '0.png'
                mask_path1 = rendered_output_dir / 'mask' / f'{model_idx:04}_{lri:03}' / '1.png'
                depth_path0 = rendered_output_dir / 'depth' / f'{model_idx:04}_{lri:03}' / '0.png'
                depth_path1 = rendered_output_dir / 'depth' / f'{model_idx:04}_{lri:03}' / '1.png'
                intri_path0 = rendered_output_dir / 'parm' / f'{model_idx:04}_{lri:03}' / '0_intrinsic.npy'
                intri_path1 = rendered_output_dir / 'parm' / f'{model_idx:04}_{lri:03}' / '1_intrinsic.npy'
                extri_path0 = rendered_output_dir / 'parm' / f'{model_idx:04}_{lri:03}' / '0_extrinsic.npy'
                extri_path1 = rendered_output_dir / 'parm' / f'{model_idx:04}_{lri:03}' / '1_extrinsic.npy'
                # Load data
                img0 = cv2.imread(str(img_path0))
                img1 = cv2.imread(str(img_path1))
                mask0 = cv2.imread(str(mask_path0), cv2.IMREAD_GRAYSCALE)
                mask1 = cv2.imread(str(mask_path1), cv2.IMREAD_GRAYSCALE)
                depth0 = cv2.imread(str(depth_path0), cv2.IMREAD_UNCHANGED) / 10_000.0
                depth1 = cv2.imread(str(depth_path1), cv2.IMREAD_UNCHANGED) / 10_000.0
                intri0 = np.load(str(intri_path0))
                intri1 = np.load(str(intri_path1))
                extri0 = np.load(str(extri_path0))[:3, :4]
                extri1 = np.load(str(extri_path1))[:3, :4]
                # Create PCDs
                pts0 = cls.depth2pts(depth0, extri0, intri0)
                pts1 = cls.depth2pts(depth1, extri1, intri1)
                # Rectify
                stereo_data = cls.get_rectified_stereo_data((img0, mask0, intri0, extri0, pts0), (img1, mask1, intri1, extri1, pts1))
                # Save
                cls.save_stereo_data(stereo_data, rectified_output_dir, model_idx=model_idx, camera_pair_idx=lri)


# if __name__ == '__main__':
#     assert THUMAN_ROOT.exists(), f"THuman2.1 root path does not exist: {THUMAN_ROOT}"
#     assert CAPTURES_ROOT.exists(), f"Capture root path does not exist: {CAPTURES_ROOT}"
#
#     cameras_ = THumanUtilsO3d.create_cameras_from_calibration(
#         calibration_session_path=CAPTURES_ROOT / 'Aggregated' / CALIBRATION_SESSION_NAME,
#         calibration_method=CALIBRATION_METHOD,
#         image_size=(1024, 1024),
#         cam_idx=list(range(17)),
#         align_to_yup=True,
#         lookat_point=(0.0, 0.75, 0.0),
#         override_calibration=Path('authors_cams.pkl')
#     )
#     for model_idx_ in tqdm([0], desc='Visualizing THuman/SMPL Alignment'):
#         models_root_ = (THUMAN_ROOT / 'model') if (THUMAN_ROOT / 'model').exists() else THUMAN_ROOT
#         if not (models_root_ / 'train' / f'{model_idx_:04d}').exists() and not (models_root_ / 'val' / f'{model_idx_:04d}').exists():
#             print(f'Model {model_idx_:04d} does not exist in {models_root_}', file=sys.stderr)
#             continue
#         is_train_ = (models_root_ / 'train' / f'{model_idx_:04d}').exists()
#         mesh_, smpl_parameters_ = THumanUtilsO3d.read_mesh(
#             model_idx=model_idx_,
#             is_train=is_train_,
#             thuman_root=THUMAN_ROOT,
#             return_full_smpl_params=True,
#             canonicalize=True,
#         )
#         mesh_ = THumanUtilsO3d.process_mesh(mesh_, human_height_std=0.0, random_translation=False, random_rotation=False)
#         authors_cams_ = THumanUtilsO3d.visualize_mesh(
#             mesh=mesh_,
#             smpl_parameters=smpl_parameters_,
#             cameras=cameras_,
#             image_path=models_root_ / ('train' if is_train_ else 'val') / f'{model_idx_:04d}' / 'mesh_smpl_alignment.jpg',
#             window_visible=True,
#             # visualize_authors_cams_from=THUMAN_ROOT / 'rendered@245'
#         )
#         if not os.path.exists('authors_cams.pkl'):
#             print('Saving authors cameras to authors_cams.pkl')
#             with open('authors_cams.pkl', 'wb') as f:
#                 pickle.dump([
#                     dict(
#                         intrinsic=c.intrinsic.intrinsic_matrix.tolist(),
#                         extrinsic=c.extrinsic.tolist(),
#                         width=c.intrinsic.width,
#                         height=c.intrinsic.height
#                     )
#                     for c in authors_cams_
#                 ], f)
#         # render
#         renderings_ = THumanUtilsO3d.render_mesh_to_cameras(mesh_, cameras_, image_size=(1024, 1024))
#         # save
#         import torchvision
#
#         rendered_images_ = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in renderings_['rgb']], dim=0)
#         torchvision.utils.save_image(rendered_images_.cpu(), str(models_root_ / ('train' if is_train_ else 'val') / f'{model_idx_:04d}' / 'renders_rgb.jpg'), nrow=4, normalize=True, scale_each=True)
#         rendered_depths = torch.stack([torch.from_numpy(r)[None] for r in renderings_['depth']], dim=0)
#         torchvision.utils.save_image(rendered_depths.cpu(), str(models_root_ / ('train' if is_train_ else 'val') / f'{model_idx_:04d}' / 'renders_depth.jpg'), nrow=4, normalize=True, scale_each=True)
#         rendered_masks = torch.stack([torch.from_numpy(r)[None] for r in renderings_['mask']], dim=0)
#         torchvision.utils.save_image(rendered_masks.cpu(), str(models_root_ / ('train' if is_train_ else 'val') / f'{model_idx_:04d}' / 'renders_mask.jpg'), nrow=4, normalize=True, scale_each=True)
#
#         if authors_cams_ is not None:
#             authors_renderings_ = THumanUtilsO3d.render_mesh_to_cameras(mesh_, authors_cams_, image_size=(1024, 1024))
#             authors_rendered_images_ = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in authors_renderings_['rgb']], dim=0)
#             torchvision.utils.save_image(authors_rendered_images_.cpu(), str(models_root_ / ('train' if is_train_ else 'val') / f'{model_idx_:04d}' / 'authors_renders_rgb.jpg'), nrow=4, normalize=True, scale_each=True)
#             authors_rendered_depths_ = torch.stack([torch.from_numpy(r)[None] for r in renderings_['depth']], dim=0)
#             torchvision.utils.save_image(authors_rendered_depths_.cpu(), str(models_root_ / ('train' if is_train_ else 'val') / f'{model_idx_:04d}' / 'authors_renders_depth.jpg'), nrow=4, normalize=True, scale_each=True)
#             authors_rendered_masks_ = torch.stack([torch.from_numpy(r)[None] for r in renderings_['mask']], dim=0)
#             torchvision.utils.save_image(authors_rendered_masks_.cpu(), str(models_root_ / ('train' if is_train_ else 'val') / f'{model_idx_:04d}' / 'authors_renders_mask.jpg'), nrow=4, normalize=True, scale_each=True)
#     exit(0)

if __name__ == '__main__':
    assert THUMAN_ROOT.exists(), f"THuman2.1 root path does not exist: {THUMAN_ROOT}"
    assert CAPTURES_ROOT.exists(), f"Capture root path does not exist: {CAPTURES_ROOT}"

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # create cameras
    target_image_size_ = (1024, 1024)  # (H, W)
    target_image_size_hr_ = (1420, 1420)  # (H, W)
    cam_idx_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cam_subset_idx_ = [_ for _ in cam_idx_ if _ not in [5, 6]]  #
    # cam_idx_ = list(range(17))  # all cameras
    # cam_subset_idx_ = cam_idx_
    cameras_, calibration_data_, mean_look_at_point_ = THumanUtilsO3d.create_cameras_from_calibration(
        calibration_session_path=CAPTURES_ROOT / CALIBRATION_SESSION_NAME,
        calibration_method=CALIBRATION_METHOD,
        image_size=(1024, 1024),
        cam_idx=cam_idx_,
        align_to_yup=True,
        lookat_point=(0.0, 0.75, 0.0),
        return_calibration_data=True,
        # override_calibration=Path('authors_cams.pkl'),
    )

    # 1) create virtual cameras
    all_cams = []
    for li, mi, ri in zip(range(len(cam_subset_idx_) - 2), range(1, len(cam_subset_idx_) - 1), range(2, len(cam_subset_idx_))):
        cams_idx_this = [cam_subset_idx_[li], cam_subset_idx_[mi], cam_subset_idx_[ri]]
        cams, virtual_cams = VisUtils.create_cameras(
            intrinsics_selected=calibration_data_.intrinsics[cams_idx_this],
            rotmats_selected=calibration_data_.extrinsics[cams_idx_this, :3, :3],
            tvecs_selected=calibration_data_.extrinsics[cams_idx_this, :3, 3],
            rotmats_all=calibration_data_.extrinsics[cams_idx_this, :3, :3],
            tvecs_all=calibration_data_.extrinsics[cams_idx_this, :3, 3],
            vis_only_idx=list(range(3)),
            assignment_only_idx=list(range(3)),
            H=calibration_data_.image_size[0, 0].item(),
            W=calibration_data_.image_size[0, 1].item(),
            # out_path=Path(__file__).parent,
            use_bezier=True,
            num_virtual_cameras=200,
            lib='open3d'
        )[:2]
        all_cams.append([cams[0]] + virtual_cams[:100] + [cams[1]])
        if ri == len(cam_subset_idx_) - 1:
            all_cams.append([cams[1]] + virtual_cams[100:] + [cams[2]])

    # create mesh
    models_root = (THUMAN_ROOT / 'model') if (THUMAN_ROOT / 'model').exists() else THUMAN_ROOT
    pbar = tqdm(total=sum(len(list((models_root / subset_).iterdir())) for subset_ in ['train', 'val']), desc='Rendering Model')
    vis_ = o3d.visualization.Visualizer()
    vis_.create_window(width=target_image_size_[1], height=target_image_size_[0], visible=False)
    for subset_ in ['train', 'val']:
        for model_dir_ in (models_root / subset_).iterdir():
            if not model_dir_.is_dir():
                continue
            THumanUtilsO3d.render_data(
                model_dir=model_dir_,
                output_dir=Path(THUMAN_ROOT / f'rendered_{CALIBRATION_SESSION_NAME.lower()}_depth_fix_no_lights'),
                gt_cameras_all=cameras_,
                cameras_l_virtual_r_all=all_cams,
                mean_look_at_point=mean_look_at_point_,
                plane_idx=cam_subset_idx_,
                # hr_image_size=target_image_size_hr_,
                create_renders=True,
                create_rectified=True,
                store_inverse_depth=True,
                vis=vis_,
            )
            pbar.update(1)

            # mesh_, smpl_parameters_ = THumanUtilsO3d.read_mesh(
            #     model_idx=0,
            #     is_train=True,
            #     thuman_root=THUMAN_ROOT,
            #     return_full_smpl_params=True,
            #     canonicalize=True
            # )
            # mesh_ = THumanUtilsO3d.process_mesh(mesh_)
            #
            # # render
            # renderings = THumanUtilsO3d.render_mesh_to_cameras(mesh_, cameras_, image_size=target_image_size_)
            #
            # # save
            # import torchvision
            # rendered_images = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in renderings['rgb']], dim=0)
            # torchvision.utils.save_image(rendered_images.cpu(), 'test_o3d.png', nrow=4, normalize=True, scale_each=True)
            # rendered_depths = torch.stack([torch.from_numpy(r)[None] for r in renderings['depth']], dim=0)
            # torchvision.utils.save_image(rendered_depths.cpu(), 'test_o3d_depth.png', nrow=4, normalize=True, scale_each=True)
            # rendered_masks = torch.stack([torch.from_numpy(r)[None] for r in renderings['mask']], dim=0)
            # torchvision.utils.save_image(rendered_masks.cpu(), 'test_o3d_mask.png', nrow=4, normalize=True, scale_each=True)
            #
            # # show
            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.subplots(1, 3, figsize=(32, 8))
            # plt.suptitle(f'Open3d Renders (model_idx: {0:04d})', fontsize=20)
            # plt.subplot(1, 3, 1)
            # rendered_image_grid = cv2.cvtColor(cv2.imread('test_o3d.png'), cv2.COLOR_BGR2RGB)
            # plt.imshow(rendered_image_grid)
            # plt.axis('off')
            # plt.title('Images')
            # plt.subplot(1, 3, 2)
            # rendered_depth_grid = cv2.imread('test_o3d_depth.png')
            # plt.imshow(rendered_depth_grid)
            # plt.title('Depths (Normalized)')
            # plt.axis('off')
            # plt.subplot(1, 3, 3)
            # rendered_mask_grid = cv2.imread('test_o3d_mask.png')
            # plt.imshow(rendered_mask_grid)
            # plt.title('Masks')
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
            # exit(0)
    pbar.close()
    # Clean up
    vis_.clear_geometries()
    vis_.destroy_window()
