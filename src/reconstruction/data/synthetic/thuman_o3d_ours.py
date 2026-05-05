import os
import sys

import imageio

os.environ["OPEN3D_HEADLESS_RENDERING"] = "1"
os.environ['OPEN3D_RENDERING_ENGINE'] = 'osmesa'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import copy
import math
import pickle
import random
from pathlib import Path
from typing import Optional, Tuple, List, Literal, Union, Dict
from collections import OrderedDict

import cv2
import numpy as np
import open3d as o3d
import pytorch3d.transforms
import torch
from pytorch3d.transforms import RotateAxisAngle, Rotate
from tqdm import tqdm

import pyrender
import trimesh

from reconstruction.vis.cam_orbit import InterpolatedCameraOrbit
from utils.calib import CalibrationData
from utils.misc import PathUtils, Gender
from utils.vis import VisUtils

THUMAN_ROOT = Path('/media/charisoudis/nas_transmixr/Simone/Volumetric_Video/Human Datasets/THuman2_1')
CAPTURES_ROOT = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025'
CALIBRATION_SESSION_NAME = 'Thanos_2_Calib_1'
OUT_ROOT = Path('/root/DATASETS/THuman2_1')
CALIBRATION_METHOD: Literal['MultiCamCalib', 'Caliscope'] = 'MultiCamCalib'
N_VIRTUAL_PER_PAIR = 5  # Number of virtual cameras per pair of real cameras


def _ensure_double_sided(pr_mesh: pyrender.Mesh):
    for p in pr_mesh.primitives:
        if getattr(p, "material", None) is not None:
            p.material.doubleSided = True


def _o3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    V = np.asarray(o3d_mesh.vertices)
    F = np.asarray(o3d_mesh.triangles)
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if o3d_mesh.has_vertex_colors():
        C = np.asarray(o3d_mesh.vertex_colors)
        if C.max() <= 1.0:
            C = (C * 255).astype(np.uint8)
        tm.visual.vertex_colors = C
    return tm


def _add_mesh_to_scene(scene: pyrender.Scene, mesh) -> list:
    """
    Adds the given mesh to the scene and returns a list of node(s).
    Accepts pyrender.Mesh, trimesh.Trimesh/Scene, or o3d TriangleMesh.
    """
    nodes = []
    if isinstance(mesh, pyrender.Mesh):
        _ensure_double_sided(mesh)
        nodes.append(scene.add(mesh))
    elif isinstance(mesh, trimesh.Trimesh):
        pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        _ensure_double_sided(pr)
        nodes.append(scene.add(pr))
    elif isinstance(mesh, trimesh.Scene):
        for name, geom in mesh.geometry.items():
            pr = pyrender.Mesh.from_trimesh(geom, smooth=True)
            _ensure_double_sided(pr)
            T = mesh.graph.get(name)
            nodes.append(scene.add(pr, pose=T))
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        tm = _o3d_to_trimesh(mesh)
        pr = pyrender.Mesh.from_trimesh(tm, smooth=True)
        _ensure_double_sided(pr)
        nodes.append(scene.add(pr))
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")
    return nodes


def _add_textured_obj_to_scene(scene: pyrender.Scene, obj_path: Path):
    """Load OBJ+MTL (+textures) and add all parts with their transforms."""
    loaded = trimesh.load(obj_path, skip_materials=False, maintain_order=True, process=False)
    nodes = []
    if isinstance(loaded, trimesh.Scene):
        for name, geom in loaded.geometry.items():
            pr = pyrender.Mesh.from_trimesh(geom, smooth=True)
            _ensure_double_sided(pr)
            T = loaded.graph.get(name)
            nodes.append(scene.add(pr, pose=T))
    else:
        pr = pyrender.Mesh.from_trimesh(loaded, smooth=True)
        _ensure_double_sided(pr)
        nodes.append(scene.add(pr))
    return nodes


def _add_any_mesh_to_scene(scene: pyrender.Scene, mesh_or_path, mesh_obj_path: Path | None):
    """
    Accepts: pyrender.Mesh, trimesh.Trimesh/Scene, o3d TriangleMesh,
             or uses mesh_obj_path if provided.
    Returns list of added node(s).
    """
    nodes = []
    if mesh_obj_path is not None and Path(mesh_obj_path).suffix.lower() == ".obj":
        return _add_textured_obj_to_scene(scene, Path(mesh_obj_path))

    m = mesh_or_path
    if isinstance(m, pyrender.Mesh):
        _ensure_double_sided(m)
        nodes.append(scene.add(m))
    elif isinstance(m, trimesh.Trimesh):
        pr = pyrender.Mesh.from_trimesh(m, smooth=True)
        _ensure_double_sided(pr)
        nodes.append(scene.add(pr))
    elif isinstance(m, trimesh.Scene):
        for name, geom in m.geometry.items():
            pr = pyrender.Mesh.from_trimesh(geom, smooth=True)
            _ensure_double_sided(pr)
            T = m.graph.get(name)
            nodes.append(scene.add(pr, pose=T))
    elif isinstance(m, o3d.geometry.TriangleMesh):
        tm = _o3d_to_trimesh(m)
        pr = pyrender.Mesh.from_trimesh(tm, smooth=True)
        _ensure_double_sided(pr)
        nodes.append(scene.add(pr))
    else:
        raise TypeError(f"Unsupported mesh type: {type(m)}")
    return nodes


def _rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float32)


def _rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def _pose_from_direction(direction_world: np.ndarray) -> np.ndarray:
    """
    Build a pose whose local -Z axis points along `direction_world`.
    (pyrender DirectionalLight uses the node orientation; translation is irrelevant.)
    """
    d = np.asarray(direction_world, dtype=np.float32)
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("direction vector too small")
    d /= n

    f = -d  # forward = local +Z maps to world f, so local -Z aligns with +d
    up_guess = np.array([0, 1, 0], dtype=np.float32)
    if abs(np.dot(f, up_guess)) > 0.999:  # avoid near-colinearity
        up_guess = np.array([1, 0, 0], dtype=np.float32)

    right = np.cross(up_guess, f);
    right /= np.linalg.norm(right)
    up = np.cross(f, right)  # already normalized

    R = np.stack([right, up, f], axis=1)  # columns are the local axes in world space
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    return T


def add_taichi_ring_lights(scene: pyrender.Scene,
                           num_lights: int = 6,
                           pitch_deg_range: float = 30.0,
                           base_dir: np.ndarray = np.array([0, 0, 1], dtype=np.float32),
                           color=(1.0, 1.0, 1.0),
                           intensity: float = 3.5,
                           rng: np.random.Generator | None = None):
    """
    Mimics the Taichi snippet:
        rotateX(uniform[-30,30]) @ rotateY(360/num * l) @ base_dir
    Then adds pyrender.DirectionalLight for each direction.
    """
    if rng is None:
        rng = np.random.default_rng()

    base_dir = np.asarray(base_dir, dtype=np.float32)
    for l in range(num_lights):
        yaw = np.deg2rad(360.0 / num_lights * l)
        pitch = np.deg2rad(rng.uniform(-pitch_deg_range, pitch_deg_range))
        R = _rot_x(pitch) @ _rot_y(yaw)
        dir_world = (R @ base_dir).astype(np.float32)

        light = pyrender.DirectionalLight(color=np.array(color, dtype=np.float32),
                                          intensity=float(intensity))
        pose = _pose_from_direction(dir_world)
        scene.add(light, pose=pose)


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
            smpl_model_path = Path(thuman_root).parent / 'THuman2.1_Release Smpl-X Paras' / 'smplx'
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
                                        calibration_session_name: str,
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
            calibration_session_name (str): Name of the calibration session.
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
                calibration_session_name,
                idx=cam_idx,
                method=calibration_method
            )
        # alter the intrinsics to be 1K
        calibration_data = calibration_data.resize(*image_size)
        gt_cameras, mean_look_at_point, _ = calibration_data.create_cameras(mean_idx=[_ for _ in cam_idx if _ != 5] if cam_idx != 'all' else 'all', lib='open3d')
        plane_normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)  # Default plane normal for Y-up alignment
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
            return gt_cameras, calibration_data, torch.tensor(mean_look_at_point).float(), torch.from_numpy(new_lookat_point).float(), torch.tensor(plane_normal).float()
        return gt_cameras

    @classmethod
    def rotate_cameras_to_front(cls, extrinsics, mean_look, ref_idx=None, extra_yaw_deg=-90.0):
        """
        Rotate cameras around the mean_look point so the reference camera lies on +X,
        then apply an additional yaw (default +90°) about Y to all cameras.

        Args:
            extrinsics : list of (4x4) cam->world matrices
            mean_look  : (3,) world-space point to rotate about
            ref_idx    : which camera to treat as "middle" (default: middle index)
            extra_yaw_deg : additional yaw about +Y applied AFTER alignment (default +90°)

        Returns:
            new_extrs  : list of (4x4) cam->world matrices after rotation
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as ScipyRotation

        exts = [np.array(E, dtype=np.float64, copy=True) for E in extrinsics]
        if ref_idx is None:
            ref_idx = len(exts) // 2

        # --- 1) Direction of the reference camera in the ground plane (XZ) ---
        E_ref = exts[ref_idx]
        t_ref = E_ref[:3, 3]
        v = t_ref - np.asarray(mean_look, dtype=np.float64).reshape(3)
        v[1] = 0.0  # project to ground (y=0)
        n = np.linalg.norm(v)
        if n < 1e-12:
            theta = 0.0
        else:
            v /= n
            # yaw from +X toward +Z; atan2(z, x)
            theta = np.arctan2(v[2], v[0])

        # --- 2) Align ref direction to +X (rotate world by -theta about Y) ---
        R_align = ScipyRotation.from_rotvec([0.0, -theta, 0.0]).as_matrix()

        # --- 3) Then rotate everyone by extra yaw (default +90°) about Y ---
        R_extra = ScipyRotation.from_euler('y', float(extra_yaw_deg), degrees=True).as_matrix()

        # Total rotation: align first, then extra yaw
        R_total = R_extra @ R_align

        mean_look = np.asarray(mean_look, dtype=np.float64).reshape(3)
        A = np.eye(4)
        A[:3, :3] = R_total
        A[:3, 3] = mean_look - R_total @ mean_look  # T(mean) * R * T(-mean)

        # --- 4) Apply to all cam->world extrinsics by pre-multiplication ---
        new_extrs = [A @ E for E in exts]
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
    def read_mesh(
            cls,
            canonicalize: bool = False,
            return_full_smpl_params: bool = False,
            **reader_kwargs
    ) -> Tuple[List[pyrender.Mesh], Optional[torch.Tensor | Dict]]:
        """
        Read the textured OBJ via trimesh, keep materials, optionally canonicalize,
        and return pyrender.Mesh objects (ready to add to a pyrender.Scene).

        Returns:
            (meshes, smpl) where:
              - meshes: List[pyrender.Mesh] (textured, double-sided)
              - smpl:   torch.Tensor of shape (1, 3) with global_orient (default),
                        or the full smpl_params dict if return_full_smpl_params=True,
                        or None if no SMPL file.
        """
        model_path, smpl_path = cls.read_model(**reader_kwargs)
        model_path = Path(model_path)

        # -------------------------------
        # Load OBJ+MTL(+textures) via trimesh
        # -------------------------------
        loaded = trimesh.load(
            model_path,
            skip_materials=False,
            maintain_order=True,
            process=False
        )

        # Normalize to a list of (Trimesh, local_to_world 4x4) pairs
        if isinstance(loaded, trimesh.Scene):
            geoms: List[Tuple[trimesh.Trimesh, np.ndarray]] = []
            for name, geom in loaded.geometry.items():
                T = loaded.graph.get(name)  # node-to-world
                geoms.append((geom, T))
        else:
            geoms = [(loaded, np.eye(4))]

        # -------------------------------
        # Optionally canonicalize w.r.t. SMPL
        # -------------------------------
        smpl_params = None
        if smpl_path is not None:
            with open(smpl_path, "rb") as f:
                smpl_params = pickle.load(f)

        if canonicalize and smpl_params is not None:
            from scipy.spatial.transform import Rotation as ScipyRotation

            def _stack_vertices(glist: List[Tuple[trimesh.Trimesh, np.ndarray]]) -> np.ndarray:
                pts = []
                for g, T in glist:
                    v = g.vertices
                    v_h = np.c_[v, np.ones((len(v), 1))]
                    pts.append((v_h @ T.T)[:, :3])
                return np.vstack(pts) if pts else np.zeros((0, 3))

            def _center_extent(glist: List[Tuple[trimesh.Trimesh, np.ndarray]]):
                V = _stack_vertices(glist)
                vmin = V.min(axis=0)
                vmax = V.max(axis=0)
                center = 0.5 * (vmin + vmax)
                extent = (vmax - vmin)
                return center, extent

            def _T_about_center(R: np.ndarray, c: np.ndarray) -> np.ndarray:
                T = np.eye(4)
                T[:3, :3] = R
                T_pre = np.eye(4);
                T_pre[:3, 3] = -c
                T_post = np.eye(4);
                T_post[:3, 3] = c
                return T_post @ T @ T_pre

            def _S_about_center(s: float, c: np.ndarray) -> np.ndarray:
                S = np.eye(4);
                S[0, 0] = S[1, 1] = S[2, 2] = s
                T_pre = np.eye(4);
                T_pre[:3, 3] = -c
                T_post = np.eye(4);
                T_post[:3, 3] = c
                return T_post @ S @ T_pre

            # Running global transform applied to ALL meshes (left-multiplied)
            T_total = np.eye(4)

            # 1) Remove global SMPL rotation: rotate scan by R_inv about its center
            rot_vec = smpl_params['global_orient'].squeeze()  # (3,)
            R_mat = ScipyRotation.from_rotvec(rot_vec).as_matrix()
            R_inv = R_mat.T

            center0, _ = _center_extent(geoms)
            T_total = _T_about_center(R_inv, center0) @ T_total

            # 2) Small tilt alignment using SMPL spline_dir
            smpl_mesh, spline_dir = cls.smpl_forward(smpl_params, return_spline_dir=True)
            vz, vy = spline_dir[2], spline_dir[1]
            theta = np.arctan2(vz, vy)
            R_align = ScipyRotation.from_rotvec([-0.5 * theta, 0.0, 0.0]).as_matrix()
            # update smpl_params' global_orient to this small align (as in original)
            smpl_params['global_orient'] = ScipyRotation.from_matrix(R_align).as_rotvec()

            # apply R_align about current center
            center1, _ = _center_extent([(g, T_total @ T) for g, T in geoms])
            T_total = _T_about_center(R_align, center1) @ T_total

            # 3) Scale + translate to match SMPL AABB
            # SMPL bbox (Open3D)
            smpl_bb = smpl_mesh.get_axis_aligned_bounding_box()
            smpl_center = np.asarray(smpl_bb.get_center())
            smpl_extent = np.asarray(smpl_bb.get_extent())

            # Scan bbox after current T_total
            center2, extent2 = _center_extent([(g, T_total @ T) for g, T in geoms])
            s = (smpl_extent.max() / max(extent2.max(), 1e-9))

            # scale about center, then translate centers
            T_total = _S_about_center(s, center2) @ T_total
            # recompute center after scaling
            center3, _ = _center_extent([(g, T_total @ T) for g, T in geoms])
            T_translate = np.eye(4);
            T_translate[:3, 3] = (smpl_center - center3)
            T_total = T_translate @ T_total

            # Bake T_total into each geom (compose with existing node transform)
            baked_geoms: List[trimesh.Trimesh] = []
            for g, T in geoms:
                g2 = g.copy()
                g2.apply_transform(T_total @ T)
                baked_geoms.append(g2)
            geoms = [(g2, np.eye(4)) for g2 in baked_geoms]

        # -------------------------------
        # Convert to pyrender.Mesh (preserve textures/UVs)
        # -------------------------------
        def _to_pyrender_meshes(glist: List[Tuple[trimesh.Trimesh, np.ndarray]]) -> List[pyrender.Mesh]:
            out = []
            for g, T in glist:
                # bake any remaining local transform
                if not np.allclose(T, np.eye(4)):
                    g = g.copy()
                    g.apply_transform(T)
                pr = pyrender.Mesh.from_trimesh(g, smooth=True)
                # make backfaces visible (scans often have flipped normals)
                for prim in pr.primitives:
                    if getattr(prim, "material", None) is not None:
                        prim.material.doubleSided = True
                out.append(pr)
            return out

        pr_meshes = _to_pyrender_meshes(geoms)[0]

        # -------------------------------
        # Return SMPL in requested shape
        # -------------------------------
        if smpl_params is None:
            return pr_meshes, None

        if not return_full_smpl_params:
            go = torch.from_numpy(smpl_params['global_orient']).float().reshape(1, 3)
            return pr_meshes, go

        return pr_meshes, smpl_params

    @classmethod
    def process_mesh(cls,
                     mesh,
                     human_height_mean: float = 1.5,
                     human_height_std: float = 0.1,
                     random_rotation: bool = False,
                     random_translation: bool = False):
        """
        Same logic as original: height normalize by y-extent, set min(y)=0, optional XY jitter, optional Y-rotation.
        Supports: pyrender.Mesh (preferred), trimesh.Trimesh, or o3d.geometry.TriangleMesh.
        Returns the same object type (mutated in place).
        """

        # --- Helpers to read/write vertices for each supported type ---
        def read_vertices(m):
            if isinstance(m, pyrender.Mesh):
                Vs = [prim.positions.copy() for prim in m.primitives if prim.positions is not None]
                return Vs  # list of (Ni,3)
            elif isinstance(m, trimesh.Trimesh):
                return [m.vertices.copy()]
            elif isinstance(m, o3d.geometry.TriangleMesh):
                return [np.asarray(m.vertices).copy()]
            else:
                raise TypeError("mesh must be pyrender.Mesh, trimesh.Trimesh, or open3d TriangleMesh")

        def write_vertices(m, Vlist):
            if isinstance(m, pyrender.Mesh):
                i = 0
                for prim in m.primitives:
                    if prim.positions is None:
                        continue
                    prim.positions = Vlist[i]
                    i += 1
            elif isinstance(m, trimesh.Trimesh):
                m.vertices = Vlist[0]
                m.update_vertices(mask=None)
            elif isinstance(m, o3d.geometry.TriangleMesh):
                m.vertices = o3d.utility.Vector3dVector(Vlist[0])
            return m

        # Read all vertex blocks (for pyrender.Mesh there can be multiple primitives)
        Vblocks = read_vertices(mesh)
        Vall = np.concatenate(Vblocks, axis=0)

        # --- Original logic starts here ---
        verts_t = torch.from_numpy(Vall).float()

        # height normalization
        vy_min, vy_max = verts_t[:, 1].min(), verts_t[:, 1].max()
        human_height = human_height_mean + (2 * random.random() - 1) * human_height_std
        scale = human_height / (float(vy_max - vy_min) + 1e-12)
        verts_t *= scale

        # floor to y=0 (use global min across all blocks)
        y_min_after = verts_t[:, 1].min()
        verts_t[:, 1] -= y_min_after

        Vall = verts_t.cpu().numpy()

        # split back into blocks
        sizes = [v.shape[0] for v in Vblocks]
        cum = np.cumsum([0] + sizes)
        Vblocks = [Vall[cum[i]:cum[i + 1], :] for i in range(len(sizes))]

        # random translation (x,z) using extents after scaling/flooring
        if random_translation:
            delta_x = Vall[:, 0].max() - Vall[:, 0].min()
            delta_z = Vall[:, 2].max() - Vall[:, 2].min()
            move_range = 0.1 if human_height < 1.40 else 0.05
            if delta_x > 1.0 or delta_z > 1.0:
                move_range = 0.01
            dx = float(np.random.uniform(-move_range, move_range))
            dz = float(np.random.uniform(-move_range, move_range))
            for V in Vblocks:
                V[:, 0] += dx
                V[:, 2] += dz

        # random rotation around Y
        if random_rotation:
            angle = float(np.random.uniform(-np.pi / 4, np.pi / 4))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([[cos_a, 0, sin_a],
                          [0, 1, 0],
                          [-sin_a, 0, cos_a]], dtype=np.float32)
            for V in Vblocks:
                V[:] = V @ R.T

        # Write back and return same type
        return write_vertices(mesh, Vblocks)

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

    PYRENDER_CACHE = {}

    @classmethod
    def render_mesh_to_cameras(cls,
                               mesh,  # accepts pyrender.Mesh, trimesh, or o3d mesh
                               cameras: list,
                               image_size: tuple = (2048, 2048)) -> dict:
        H, W = image_size
        if (H, W) in cls.PYRENDER_CACHE:
            r, scene = cls.PYRENDER_CACHE[(H, W)]
        else:
            r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.2, 0.2, 0.2, 1.0])
            cls.PYRENDER_CACHE[(H, W)] = (r, scene)

        # clear scene
        add_taichi_ring_lights(scene,
                               num_lights=6,
                               pitch_deg_range=30.0,
                               base_dir=np.array([0, 0, 1]),
                               color=(1.0, 1.0, 1.0),
                               intensity=3.5)

        # Add the mesh (any supported type)
        _add_mesh_to_scene(scene, mesh)

        results = {'rgb': [], 'depth': [], 'mask': [], 'intrinsic': [], 'extrinsic': []}
        for cam in cameras:
            K = cam.intrinsic.intrinsic_matrix
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            pyr_cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=1_000)
            T_wc = np.asarray(cam.extrinsic, copy=True)
            T_wc[:3, 1:3] *= -1  # flip Y and Z axes
            T_wc[3, :] = np.array([0.0, 0.0, 0.0, 1.0])  # ensure homogeneous
            cam_node = scene.add(pyr_cam, pose=T_wc.astype(np.float32))
            scene.main_camera_node = cam_node

            # Render
            color, depth = r.render(scene)

            # Collect (keep original mask logic: 0 < d < 1e4)
            depth_f32 = depth.astype(np.float32, copy=False)
            results['rgb'].append(color.copy())  # uint8 (H, W, 3)
            results['depth'].append(depth_f32.copy())
            mask = np.logical_and(depth_f32 > 0, depth_f32 < 1e4).astype(np.float32)
            results['mask'].append(mask)
            results['intrinsic'].append(K.copy())
            results['extrinsic'].append(np.asarray(cam.extrinsic, copy=True))

            # Remove camera for next iteration
            scene.remove_node(cam_node)

        scene.clear()
        return results

    @classmethod
    def visualize_mesh(cls,
                       mesh,  # can be pyrender.Mesh, trimesh, or o3d mesh
                       image_path: Path,
                       smpl_parameters=None,
                       cameras=None,
                       width: int = 1024, height: int = 1024,
                       visualize_authors_cams_from: Path = None,
                       mesh_obj_path: Path = None) -> None:
        """
        Headless render of a mesh (textured if mesh_obj_path is given) with optional SMPL overlay.
        Saves an image to `image_path` using pyrender.
        """
        # Scene + light
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.25, 0.25, 0.25, 1.0])
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=np.eye(4))

        # Main mesh
        _add_any_mesh_to_scene(scene, mesh, mesh_obj_path)

        # Optional SMPL overlay (keeps your processing)
        if smpl_parameters is not None:
            smpl_o3d = cls.smpl_forward(smpl_parameters, return_spline_dir=False)
            smpl_o3d = cls.process_mesh(smpl_o3d, human_height_std=0.0,
                                        random_rotation=False, random_translation=False)
            smpl_tm = _o3d_to_trimesh(smpl_o3d)
            smpl_pr = pyrender.Mesh.from_trimesh(smpl_tm, smooth=True)
            _ensure_double_sided(smpl_pr)
            scene.add(smpl_pr)

        # Camera
        r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

        if cameras and len(cameras) > 0:
            cam0 = cameras[0]
            K = cam0.intrinsic.intrinsic_matrix
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            pyr_cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=1e-3, zfar=1e4)

            # Open3D extrinsic is world->cam; pyrender wants cam->world
            T_wc = np.asarray(cam0.extrinsic, copy=True)
            T_wc[:3, 1:3] *= -1  # flip Y and Z axes
            T_wc[3, :] = np.array([0.0, 0.0, 0.0, 1.0])  # ensure homogeneous
            cam_node = scene.add(pyr_cam, pose=T_wc.astype(np.float32))
            scene.main_camera_node = cam_node
        # else: auto camera

        # Render & save
        color, _ = r.render(scene)
        imageio.imwrite(str(image_path), color)
        r.delete()

    @classmethod
    def render_data_in_capturestudio_raw_fmt(
            cls,
            model_dir: Path,
            output_session_dir: Path,  # e.g. /root/DATASETS/THuman2_1_Thanos_2_Calib_1
            all_cams: Dict[str, o3d.camera.PinholeCameraParameters],
            image_size_hw: Tuple[int, int],
    ):
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

        # Render to all cams
        all_renders = cls.render_mesh_to_cameras(
            mesh,
            list(all_cams.values()),
            image_size=image_size_hw
        )

        for j, cam_name in enumerate(all_cams.keys()):

            # Create the camera directory
            output_cam_dir = output_session_dir / 'orbbec' / cam_name
            output_cam_dir.mkdir(parents=True, exist_ok=True)
            for subdir in ['color', 'depth_aligned', 'mask', 'parameters']:
                (output_cam_dir / subdir).mkdir(parents=True, exist_ok=True)

            # Save the rendered images, depth maps, masks, and parameters
            cv2.imwrite(str(output_cam_dir / 'color' / f'{model_idx:04d}.jpg'), cv2.cvtColor(all_renders['rgb'][j], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_cam_dir / 'mask' / f'{model_idx:04d}.jpg'), (np.clip(all_renders['mask'][j], 0, 1) * 255.0 + 0.5).astype(np.uint8))
            cv2.imwrite(str(output_cam_dir / 'depth_aligned' / f'{model_idx:04d}.png'), (np.clip(all_renders['depth'][j], 0, 5.0) * 1e3 * 13).astype(np.uint16))
            if not cam_name.startswith('cam'):
                np.save(str(output_cam_dir / 'parameters' / f'{model_idx:04d}_extrinsic.npy'), all_renders['extrinsic'][j])
                np.save(str(output_cam_dir / 'parameters' / f'{model_idx:04d}_intrinsic.npy'), all_renders['intrinsic'][j])
            elif not (output_cam_dir / 'parameters' / 'intrinsic.npy').exists() or not (output_cam_dir / 'parameters' / 'extrinsic.npy').exists() or model_idx == 0:
                np.save(str(output_cam_dir / 'parameters' / 'intrinsic.npy'), all_renders['intrinsic'][j])
                np.save(str(output_cam_dir / 'parameters' / 'extrinsic.npy'), all_renders['extrinsic'][j])


# if __name__ == '__main__':
#     assert THUMAN_ROOT.exists(), f"THuman2.1 root path does not exist: {THUMAN_ROOT}"
#     assert CAPTURES_ROOT.exists(), f"Capture root path does not exist: {CAPTURES_ROOT}"
#
#     cameras_ = THumanUtilsO3d.create_cameras_from_calibration(
#         calibration_session_name=CALIBRATION_SESSION_NAME,
#         calibration_method=CALIBRATION_METHOD,
#         image_size=(1024, 1024),
#         cam_idx='all',
#         align_to_yup=True,
#         lookat_point=(0.0, 0.75, 0.0),
#         # override_calibration=Path('authors_cams.pkl')
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
#         THumanUtilsO3d.visualize_mesh(
#             mesh=mesh_,
#             smpl_parameters=smpl_parameters_,
#             cameras=copy.deepcopy(cameras_),
#             image_path='mesh_smpl_alignment.jpg',
#             # visualize_authors_cams_from=THUMAN_ROOT / 'rendered@245'
#         )
#         # if not os.path.exists('authors_cams.pkl'):
#         #     print('Saving authors cameras to authors_cams.pkl')
#         #     with open('authors_cams.pkl', 'wb') as f:
#         #         pickle.dump([
#         #             dict(
#         #                 intrinsic=c.intrinsic.intrinsic_matrix.tolist(),
#         #                 extrinsic=c.extrinsic.tolist(),
#         #                 width=c.intrinsic.width,
#         #                 height=c.intrinsic.height
#         #             )
#         #             for c in authors_cams_
#         #         ], f)
#         # render
#         renderings_ = THumanUtilsO3d.render_mesh_to_cameras(mesh_, cameras_, image_size=(1024, 1024))
#         # save
#         import torchvision
#
#         rendered_images_ = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in renderings_['rgb']], dim=0)
#         torchvision.utils.save_image(rendered_images_.cpu() / 255.0, str('/root/capturestudio2/src/reconstruction/data/synthetic/renders_rgb.jpg'), nrow=4, normalize=True, scale_each=True)
#         # print('done')
#         # rendered_depths = torch.stack([torch.from_numpy(r)[None] for r in renderings_['depth']], dim=0)
#         # torchvision.utils.save_image(rendered_depths.cpu(), str('/root/capturestudio2/src/reconstruction/data/synthetic/renders_depth.jpg'), nrow=4, normalize=True, scale_each=True)
#         # print('done')
#         # rendered_masks = torch.stack([torch.from_numpy(r)[None] for r in renderings_['mask']], dim=0)
#         # torchvision.utils.save_image(rendered_masks.cpu(), str('/root/capturestudio2/src/reconstruction/data/synthetic/renders_mask.jpg'), nrow=4, normalize=True, scale_each=True)
#         # print('done')
#
#         # if authors_cams_ is not None:
#         #     authors_renderings_ = THumanUtilsO3d.render_mesh_to_cameras_offscreen(mesh_, authors_cams_, image_size=(1024, 1024))
#         #     authors_rendered_images_ = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in authors_renderings_['rgb']], dim=0)
#         #     torchvision.utils.save_image(authors_rendered_images_.cpu(), str('/root/capturestudio2/src/reconstruction/data/synthetic/authors_renders_rgb.jpg'), nrow=4, normalize=True, scale_each=True)
#         #     authors_rendered_depths_ = torch.stack([torch.from_numpy(r)[None] for r in renderings_['depth']], dim=0)
#         #     torchvision.utils.save_image(authors_rendered_depths_.cpu(), str('/root/capturestudio2/src/reconstruction/data/synthetic/authors_renders_depth.jpg'), nrow=4, normalize=True, scale_each=True)
#         #     authors_rendered_masks_ = torch.stack([torch.from_numpy(r)[None] for r in renderings_['mask']], dim=0)
#         #     torchvision.utils.save_image(authors_rendered_masks_.cpu(), str('/root/capturestudio2/src/reconstruction/data/synthetic/authors_renders_mask.jpg'), nrow=4, normalize=True, scale_each=True)
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
    cam_idx_reconstruction_ = [_ for _ in cam_idx_ if _ not in [5]]
    gt_cameras_, calibration_data_, mean_look_at_ori_, mean_lookat_, plane_normal_ = THumanUtilsO3d.create_cameras_from_calibration(
        CALIBRATION_SESSION_NAME,
        cam_idx=cam_idx_,
        image_size=target_image_size_,
        return_calibration_data=True,
        align_to_yup=True,
        lookat_point=(0.0, 0.75, 0.0),
    )
    # camera_orbit_ = InterpolatedCameraOrbit(
    #     gt_intrinsics=calibration_data_.intrinsics.cpu().numpy(),
    #     gt_extrinsics_w2c=calibration_data_.extrinsics_w2c.cpu().numpy(),
    #     reconstruction_idx=[int(_) for _ in cam_idx_reconstruction_] if not isinstance(cam_idx_reconstruction_, str) else cam_idx_reconstruction_,
    #     gt_image_size_hw=(int(calibration_data_.image_size[0][0]), int(calibration_data_.image_size[0][1])),
    #     bezier_degree=8,
    # )
    cam_orbits_lr_all_ = InterpolatedCameraOrbit(
        gt_intrinsics=calibration_data_.intrinsics.numpy(),
        gt_extrinsics_w2c=calibration_data_.extrinsics_w2c.numpy(),
        reconstruction_idx=cam_idx_reconstruction_,
        gt_image_size_hw=(int(calibration_data_.image_size[0][0]), int(calibration_data_.image_size[0][1])),
        bezier_degree=8,
    )
    cam_orbits_aligned_extrinsics = THumanUtilsO3d.align_cameras_via_mean_look(
        extrinsics=[_ for _ in cam_orbits_lr_all_.virtual_extrinsic_c2w],
        mean_look=mean_look_at_ori_.numpy().squeeze(),
        plane_normal=plane_normal_.numpy().squeeze(),
        lookat_point=mean_lookat_.numpy().squeeze(),
    )
    # Update the camera parameters with the new extrinsics
    for i, __ in enumerate(cam_orbits_aligned_extrinsics):
        cam_orbits_lr_all_.virtual_extrinsic_c2w[i] = __
        cam_orbits_lr_all_.virtual_extrinsic_w2c[i] = np.linalg.inv(__)
    gt_cam_names_ = [c.split('/', 1)[-1] for c in calibration_data_.cam_names]
    gt_cams_dict_ = dict(zip(gt_cam_names_, gt_cameras_))

    # create mesh
    models_root = (THUMAN_ROOT / 'model') if (THUMAN_ROOT / 'model').exists() else THUMAN_ROOT
    model_dirs_ = dict(train=sorted((models_root / 'train').iterdir(), key=lambda x: int(x.name)), val=sorted((models_root / 'val').iterdir(), key=lambda x: int(x.name)))
    pbar = tqdm(total=sum(len(model_dirs_[subset_]) for subset_ in ['train', 'val']), desc='Rendering Model')

    for subset_ in ['train', 'val']:
        for model_dir_ in model_dirs_[subset_]:
            if not model_dir_.is_dir() or int(model_dir_.name) > 525:
                continue

            # # select virtual cameras
            # virtual_cam_names_, virtual_cams_lr_ = [], []
            # for pair_idx_ in range(len(cam_idx_) - 1):
            #     # create cams
            #     s, e = pair_idx_ * cam_orbits_lr_all_.num_points // len(cam_idx_), (pair_idx_ + 1) * cam_orbits_lr_all_.num_points // len(cam_idx_)
            #     virtual_idx = [
            #         *random.sample(range(s, (s + e) // 2), (N_VIRTUAL_PER_PAIR - 1) // 2),
            #         (s + e) // 2,
            #         *random.sample(range((s + e) // 2, e), (N_VIRTUAL_PER_PAIR - 1) // 2)
            #     ]
            #     virtual_cams_lr_.extend(
            #         VisUtils.create_cameras_o3d(
            #             intrinsics=cam_orbits_lr_all_.virtual_intrinsic[virtual_idx],
            #             extrinsics=cam_orbits_lr_all_.virtual_extrinsic_c2w[virtual_idx],
            #             image_size=cam_orbits_lr_all_.gt_image_size_hw,
            #             is_c2w=True
            #         )
            #     )
            #     virtual_cam_names_.extend(
            #         [f'v{gt_cam_names_[pair_idx_]}_{gt_cam_names_[pair_idx_ + 1].replace("cam", "")}_{v_idx:02d}' for v_idx in range(len(virtual_cams_lr_))]
            #     )

            THumanUtilsO3d.render_data_in_capturestudio_raw_fmt(
                model_dir=model_dir_,
                output_session_dir=OUT_ROOT / f'rendered_{CALIBRATION_SESSION_NAME.lower()}',
                all_cams=gt_cams_dict_,  # | dict(zip(virtual_cam_names_, virtual_cams_lr_)),
                image_size_hw=target_image_size_,
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
            # renderings = THumanUtilsO3d.render_mesh_to_cameras_offscreen(mesh_, gt_cameras_, image_size=target_image_size_)
            #
            # # save
            # import torchvision
            #
            # rendered_images = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in renderings['rgb']], dim=0)
            # torchvision.utils.save_image(rendered_images.cpu(), 'test_o3d.png', nrow=4, normalize=True, scale_each=True)
            # rendered_depths = torch.stack([torch.from_numpy(r)[None] for r in renderings['depth']], dim=0)
            # torchvision.utils.save_image(rendered_depths.cpu(), 'test_o3d_depth.png', nrow=4, normalize=True, scale_each=True)
            # rendered_masks = torch.stack([torch.from_numpy(r)[None] for r in renderings['mask']], dim=0)
            # torchvision.utils.save_image(rendered_masks.cpu(), 'test_o3d_mask.png', nrow=4, normalize=True, scale_each=True)
            #
            # # show
            # import matplotlib.pyplot as plt
            #
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

    for r, scene in THumanUtilsO3d.PYRENDER_CACHE.values():
        scene.clear()
        r.delete()
