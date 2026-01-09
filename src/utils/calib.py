import copy
import dataclasses
import functools
import importlib
import json
import pickle
from functools import partial
from pathlib import Path
from typing import Optional, List, Callable
from typing import Tuple, Union, Dict, Literal, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
from mpl_toolkits.mplot3d import Axes3D

from utils.misc import log, PathUtils


@dataclasses.dataclass(kw_only=True)
class CalibrationData:
    intrinsics: torch.Tensor  # (N, 3, 3)
    extrinsics_c2w: torch.Tensor  # (N, 4, 4), c2w
    dists: torch.Tensor  # (N, 5) in the following order (k1, k2, p1, p2, k3)
    rotmats: torch.Tensor  # (N, 3, 3), c2w
    tvecs: torch.Tensor  # (N, 3), in world coordinates
    cam_names: List[str]  # (N,)
    image_size: torch.Tensor  # (N, 2) in (H, W) format
    _preprocessing_transforms: Optional[List[Callable]] = None

    def __getitem__(self, item) -> 'CalibrationData':
        # select index in transform arguments
        new_transforms = copy.deepcopy(self._preprocessing_transforms)
        for ti, t in enumerate(new_transforms or []):
            if isinstance(t, partial) and t.keywords is not None:
                new_keywords = copy.deepcopy(t.keywords)
                for k in t.keywords:
                    if isinstance(t.keywords[k], (np.ndarray, torch.Tensor)) and t.keywords[k].shape[0] == self.intrinsics.shape[0]:
                        new_keywords[k] = t.keywords[k][item]
                new_transforms[ti] = partial(t.func, *t.args, **new_keywords)
        return CalibrationData(
            intrinsics=self.intrinsics[item],
            extrinsics_c2w=self.extrinsics_c2w[item],
            dists=self.dists[item],
            rotmats=self.rotmats[item],
            tvecs=self.tvecs[item],
            cam_names=self.cam_names[item] if isinstance(item, int) else [self.cam_names[i] for i in item],
            image_size=self.image_size[item],
            _preprocessing_transforms=new_transforms
        )

    def create_cameras(self, mean_idx: List[int] | Literal['all'] = 'all', device: torch.device | str = 'cpu', lib: str = 'pytorch3d') -> Tuple[Any, torch.Tensor, torch.Tensor]:
        assert lib in ['open3d', 'pytorch3d'], f"Invalid library: {lib}. Choose 'pytorch3d' or 'kaolin'"
        from utils.vis import VisUtils
        if lib == 'open3d':
            cameras = VisUtils.create_cameras_o3d(
                intrinsics=self.intrinsics,
                extrinsics=self.extrinsics_c2w,
                image_size=tuple(self.image_size[0].cpu().tolist()),
                is_c2w=True,
            )
        elif lib == 'pytorch3d':
            cameras = VisUtils.create_cameras_p3d(
                intrinsics=self.intrinsics,
                rotmats=self.rotmats,
                tvecs=self.tvecs,
                image_size=tuple(self.image_size[0].cpu().tolist()),
                is_c2w=True,
            ).to(device=device)
        mean_look_at_point, mean_up_vector = VisUtils.compute_mean_look_at_and_up(
            tvecs=self.tvecs[mean_idx] if mean_idx != 'all' else self.tvecs,
            rotmats=self.rotmats[mean_idx] if mean_idx != 'all' else self.rotmats,
        )
        return cameras, mean_look_at_point, mean_up_vector

    def create_camera_plane(self, plane_idx: List[int] | Literal['all'] = 'all', device: torch.device | str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fits a plane to the camera translation vectors using SVD.

        Assumes self.tvecs is a torch.Tensor of shape (N, 3).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - plane_normal: Normalized normal vector of the best-fit plane (shape: (3,)).
                - offset: Offset 'd' such that n.dot(p) + d = 0 for points p on the plane (scalar).
        """
        tvecs = self.tvecs.clone().to(device=device)
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

    @functools.cached_property
    def extrinsics_w2c(self) -> torch.Tensor:
        """
        Returns the extrinsics in world-to-camera format.
        """
        return self.extrinsics_c2w.inverse()

    def get_preprocessing_transforms(self) -> List[Callable]:
        """
        Returns the preprocessing transforms if they exist, otherwise returns an empty list.
        Example usage:

        >>> calibration_data = CalibrationData(...)
        >>> transforms = calibration_data.get_preprocessing_transforms()
        >>> for transform in transforms:
        >>>     image, mask = transform(image, mask, cam_idx_s0_all=0) # the cam idx wrt to all cam indices

        Returns
        -------
        List[Callable]
            A list of preprocessing transforms to be applied to the images. If no preprocessing transforms are set, returns an empty list.
        """
        if self._preprocessing_transforms is None:
            return []
        return self._preprocessing_transforms

    def resize(self, *new_size_hw: int, apply_intrinsics_fix: bool = False) -> 'CalibrationData':
        """
        Returns a new CalibrationData object with resized intrinsics and image_size.
        Extrinsics are not resized, as they are assumed to be in world coordinates.

        Parameters
        ----------
        new_size_hw : int
            The new height and width for the images. If single value is provided, it is used for both height and width.
        apply_intrinsics_fix : bool, optional
            If True, applies a fix to the intrinsics to bring them to GPS training intrinsics. Default is False.

        Returns
        -------
        CalibrationData
            A new CalibrationData object with resized intrinsics and image_size.
        """
        if len(new_size_hw) == 1:
            new_size_hw = new_size_hw[0], new_size_hw[0]
        elif len(new_size_hw) != 2:
            raise ValueError(f"Invalid new size: {new_size_hw}. Expected a single integer or a tuple of two integers (height, width).")

        # resize
        src_h, src_w = int(self.image_size[0][0]), int(self.image_size[0][1])
        tgt_h, tgt_w = new_size_hw
        scale = max(tgt_h / src_h, tgt_w / src_w)
        new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))

        # center crop
        top = (new_h - tgt_h) // 2
        left = (new_w - tgt_w) // 2

        # adjust intrinsics
        intrinsic_new = self.intrinsics.clone()
        intrinsic_new[..., 0, 0] *= scale
        intrinsic_new[..., 1, 1] *= scale
        intrinsic_new[..., 0, 2] = intrinsic_new[..., 0, 2] * scale - left
        intrinsic_new[..., 1, 2] = intrinsic_new[..., 1, 2] * scale - top

        # adjust image size
        image_size_new = self.image_size.clone()
        image_size_new[:, 0] = tgt_h
        image_size_new[:, 1] = tgt_w
        resize_transform = partial(CalibrationData.resize_and_center_crop_transform, src_image_size_hw=(src_h, src_w), target_image_size_hw=(tgt_h, tgt_w))
        transforms = (self._preprocessing_transforms or []) + [resize_transform]

        if apply_intrinsics_fix:
            all_remaps_x, all_remaps_y = [], []
            authors_h, authors_w = 1024, 1024
            authors_fx, authors_fy = authors_w * 0.6, authors_h * 0.6
            authors_cx, authors_cy = authors_w // 2, authors_h // 2
            author_u, author_v = np.meshgrid(np.arange(authors_w), np.arange(authors_h))  # target pixel grid
            for intri in intrinsic_new:
                calib_fx, calib_fy, calib_cx, calib_cy = intri[0, 0].item(), intri[1, 1].item(), intri[0, 2].item(), intri[1, 2].item()
                map_x = calib_fx / authors_fx * (author_u - authors_cx) + calib_cx
                map_y = calib_fy / authors_fy * (author_v - authors_cy) + calib_cy
                all_remaps_x.append(map_x.astype(np.float32))
                all_remaps_y.append(map_y.astype(np.float32))
            transforms.append(partial(CalibrationData.remap_transform, remaps_x=np.stack(all_remaps_x, axis=0), remaps_y=np.stack(all_remaps_y, axis=0)))
            intrinsic_new[:, 0, 0] = authors_fx
            intrinsic_new[:, 1, 1] = authors_fy
            intrinsic_new[:, 0, 2] = authors_cx
            intrinsic_new[:, 1, 2] = authors_cy
            image_size_new[:, 0] = authors_h
            image_size_new[:, 1] = authors_w

        return CalibrationData(
            intrinsics=intrinsic_new,
            extrinsics_c2w=self.extrinsics_c2w.clone(),
            dists=self.dists.clone(),
            rotmats=self.rotmats.clone(),
            tvecs=self.tvecs.clone(),
            cam_names=self.cam_names.copy(),
            image_size=image_size_new,
            _preprocessing_transforms=transforms
        )

    def rotate(self, rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None) -> 'CalibrationData':
        """
        Returns a new CalibrationData object with rotated intrinsics and image_size.
        Extrinsics are not rotated, as they are assumed to be in world coordinates.

        Parameters
        ----------
        rotate : str, optional
            The rotation to apply. Must be one of '90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180'. Default is None.

        Returns
        -------
        CalibrationData
            A new CalibrationData object with rotated intrinsics and image_size.
        """
        if rotate is None:
            return self

        K = self.intrinsics.clone()
        Hpix = self.image_size[:, 0].to(dtype=K.dtype, device=K.device)  # H
        Wpix = self.image_size[:, 1].to(dtype=K.dtype, device=K.device)  # W

        # pull standard parameters (assumes zero skew)
        fx = K[:, 0, 0].clone()
        fy = K[:, 1, 1].clone()
        cx = K[:, 0, 2].clone()
        cy = K[:, 1, 2].clone()

        K_new = torch.zeros_like(K)
        K_new[:, 2, 2] = 1.0  # homogeneous coord

        if rotate == '90_COUNTERCLOCKWISE':
            # (u', v') = (v, W-1 - u), new size (H', W') = (W, H)
            K_new[:, 0, 0] = fy
            K_new[:, 1, 1] = fx
            K_new[:, 0, 2] = cy
            K_new[:, 1, 2] = (Wpix - 1.0) - cx
            image_size_new = torch.stack([self.image_size[:, 1], self.image_size[:, 0]], dim=1)

        elif rotate == '90_CLOCKWISE':
            # (u', v') = (H-1 - v, u), new size (H', W') = (W, H)
            K_new[:, 0, 0] = fy
            K_new[:, 1, 1] = fx
            K_new[:, 0, 2] = (Hpix - 1.0) - cy
            K_new[:, 1, 2] = cx
            image_size_new = torch.stack([self.image_size[:, 1], self.image_size[:, 0]], dim=1)

        elif rotate == '180':
            # (u', v') = (W-1 - u, H-1 - v), size unchanged
            K_new[:, 0, 0] = fx
            K_new[:, 1, 1] = fy
            K_new[:, 0, 2] = (Wpix - 1.0) - cx
            K_new[:, 1, 2] = (Hpix - 1.0) - cy
            image_size_new = self.image_size.clone()

        else:
            raise ValueError("rotate must be one of '90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180' or None")

        # preserve (potential) skew term only for 180°, where it’s invariant under the flip.
        # for 90° rotations we assume zero skew (OpenCV default). if you truly have nonzero skew,
        # switch to an RQ-based update.
        if rotate == '180':
            K_new[:, 0, 1] = K[:, 0, 1]

        rotate_transform = partial(CalibrationData.rotate_transform, rotate=rotate)
        transforms = (self._preprocessing_transforms or []) + [rotate_transform]

        return CalibrationData(
            intrinsics=K_new,
            extrinsics_c2w=self.extrinsics_c2w.clone(),  # c2w left untouched
            dists=self.dists.clone(),
            rotmats=self.rotmats.clone(),
            tvecs=self.tvecs.clone(),
            cam_names=self.cam_names.copy(),
            image_size=image_size_new,
            _preprocessing_transforms=transforms
        )

    def plot_camera_centers(self) -> None:
        """
        Plots camera centers extracted from extrinsic matrices.
        """
        extrinsics = self.extrinsics_c2w.cpu().numpy()
        is_w2c = False  # by
        assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4)

        centers: List[Tuple[float, float, float]] = []
        for extrinsic in extrinsics:
            if is_w2c:
                extrinsic = np.linalg.inv(extrinsic)
            center = extrinsic[:3, 3]  # Extract translation
            centers.append(center.tolist())
        centers_np = np.array(centers)

        fig = plt.figure()
        # noinspection PyTypeChecker
        ax: Axes3D = fig.add_subplot(111, projection='3d')
        ax.scatter(centers_np[:, 0], centers_np[:, 1], centers_np[:, 2], c='blue', marker='o')

        for i, c in enumerate(centers):
            ax.text(c[0], c[1], c[2], f"{i}", fontsize=10)

        ax.set_title("Camera Centers")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def remap_transform(image: np.ndarray, mask: np.ndarray, depth: Optional[np.ndarray] = None, remaps_x: np.ndarray = None, remaps_y: np.ndarray = None, cam_idx_s0: int = -1) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Applies a warp transformation to the image, mask, and depth map using the provided warp_x and warp_y coordinates.
        """
        assert remaps_x is not None and remaps_y is not None, "remaps_x and remaps_y must be provided for remapping."
        assert cam_idx_s0 >= 0, f"Invalid camera index: {cam_idx_s0}. Must be non-negative."
        warped_image = cv2.remap(image, remaps_x[cam_idx_s0], remaps_y[cam_idx_s0], interpolation=cv2.INTER_LINEAR)
        warped_mask = cv2.remap(mask.astype(np.float32), remaps_x[cam_idx_s0], remaps_y[cam_idx_s0], interpolation=cv2.INTER_NEAREST).astype(mask.dtype)
        if depth is None:
            return warped_image, warped_mask
        warped_depth = cv2.remap(depth.astype(np.float32), remaps_x[cam_idx_s0], remaps_y[cam_idx_s0], interpolation=cv2.INTER_NEAREST)
        return warped_image, warped_mask, warped_depth

    @staticmethod
    def resize_and_center_crop_transform(
            image: np.ndarray,
            mask: np.ndarray,
            depth: Optional[np.ndarray] = None,
            src_image_size_hw: Tuple[int, int] = (-1, -1),
            target_image_size_hw: Tuple[int, int] = (-1, -1),
            **ignored_kwargs) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        src_h, src_w = src_image_size_hw
        tgt_h, tgt_w = target_image_size_hw
        assert src_h > 0 and src_w > 0, f"Invalid source image size: {src_image_size_hw}. Must be positive integers."
        assert tgt_h > 0 and tgt_w > 0, f"Invalid target image size: {target_image_size_hw}. Must be positive integers."
        # resize
        scale = max(tgt_h / src_h, tgt_w / src_w)
        new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))
        image_rs = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        depth_rs = None if depth is None else cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # center crop
        top = (new_h - tgt_h) // 2
        left = (new_w - tgt_w) // 2
        bot = top + tgt_h
        right = left + tgt_w
        img_cp = image_rs[top:bot, left:right, :]
        mask_cp = mask_rs[top:bot, left:right]
        if depth_rs is None:
            return img_cp, mask_cp
        depth_cp = depth_rs[top:bot, left:right]
        return img_cp, mask_cp, depth_cp

    @staticmethod
    def rotate_transform(
            image: np.ndarray,
            mask: np.ndarray,
            depth: Optional[np.ndarray] = None,
            rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None,
            **ignored_kwargs) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if rotate:
            image = cv2.rotate(image, getattr(cv2, f'ROTATE_{rotate.upper()}'))
            mask_dtype = mask.dtype
            mask = cv2.rotate(mask.astype(np.uint8), getattr(cv2, f'ROTATE_{rotate.upper()}')).astype(mask_dtype)
            depth = cv2.rotate(depth, getattr(cv2, f'ROTATE_{rotate.upper()}'))
        return image, mask, depth

    @classmethod
    def from_pkl(cls, data: List[Dict[str, Union[list, float, int]]]) -> 'CalibrationData':
        calibration_data = dict(
            intrinsics=torch.stack([torch.tensor(d['intrinsic']) for d in data]).float(),
            extrinsics=torch.stack([torch.tensor(d['extrinsic']) for d in data]).float(),
            image_size=torch.stack([torch.tensor([d['height'], d['width']]) for d in data]).int(),
        )
        num_cameras = len(calibration_data['intrinsics'])
        calibration_data['dists'] = torch.zeros((num_cameras, 5), dtype=torch.float32)
        calibration_data['rotmats'] = calibration_data['extrinsics'][:, :3, :3]
        calibration_data['tvecs'] = calibration_data['extrinsics'][:, :3, 3]
        calibration_data['cam_names'] = [f'cam{c + 1:02d}' for c in range(num_cameras)]
        return cls(**calibration_data)

    @classmethod
    def from_session(cls, session_path: Path, idx: Literal['all'] | List[int] = 'all', method: Literal['Caliscope'] | Literal['MultiCamCalib'] = 'MultiCamCalib') -> Union['CalibrationData', None]:
        def ensure_rotmat_is_orthonormal(rotmat: torch.Tensor) -> torch.Tensor:
            """Ensure that the rotation matrix is orthonormal"""
            U, _, V = torch.linalg.svd(rotmat)
            return U.T @ V

        def ensure_rotmat_is_valid(rotmat: torch.Tensor) -> torch.Tensor:
            """Ensure that the rotation matrix is valid"""
            rotmat = rotmat.clone()
            for i in range(rotmat.shape[0]):
                if torch.linalg.det(rotmat[i]) < 0:
                    rotmat[i, 0] *= -1
                if (rotmat[i].T @ rotmat[i] - torch.eye(rotmat.shape[-1])).abs().max() > 1e-6:
                    rotmat[i] = ensure_rotmat_is_orthonormal(rotmat[i])
            return rotmat

        session_path = Path(session_path)
        if not session_path.exists():
            for d in PathUtils.capturestudio_cache_path().glob('Captures_*'):
                if d.is_dir() and (d/session_path.name).exists():
                    session_path = d/session_path.name
                    break
        assert session_path.exists(), f"Session path does not exist: {session_path}"

        log(f"Reading calibration from capture session: {session_path}", 'info')
        calibration_dir = session_path / '__calib__'
        if not calibration_dir.exists() or (not (calibration_dir / 'caliscope').exists() and not (calibration_dir / 'multicamcalib').exists()):
            log(f"Calibration directory does not exist: {calibration_dir}", 'error')
            return None
        # read calibration data
        module = importlib.import_module('utils.calib')
        method = 'Caliscope' if (calibration_dir / 'caliscope').exists() else 'MultiCamCalib'
        reader_cls = getattr(module, f'{method}Reader')
        caliscope_data = reader_cls(session_path).read()
        if idx == 'all':
            idx = list(range(len(caliscope_data)))
        extri_4x4 = torch.eye(4, dtype=torch.float32)[None].repeat(len(idx), 1, 1)
        extri_4x4[:, :3, :] = torch.stack([
            torch.tensor(v['extri'], dtype=torch.float32)
            for vi, v in enumerate(caliscope_data.values())
            if vi in idx
        ], dim=0)
        extri_4x4[:, :3, :3] = ensure_rotmat_is_valid(extri_4x4[:, :3, :3])
        extri_4x4 = extri_4x4.inverse()  # w2c (viewmatrix) --> c2w (calibration matrix)
        extri_4x4[:, :3, :3] = ensure_rotmat_is_valid(extri_4x4[:, :3, :3])
        return CalibrationData(
            intrinsics=torch.stack([
                torch.tensor(v['intri']['K'], dtype=torch.float32)
                for vi, v in enumerate(caliscope_data.values())
                if vi in idx
            ], dim=0).reshape(-1, 3, 3),
            dists=torch.stack([
                torch.tensor(v['intri']['dist'], dtype=torch.float32)
                for vi, v in enumerate(caliscope_data.values())
                if vi in idx
            ], dim=0),
            rotmats=extri_4x4[:, :3, :3],
            tvecs=extri_4x4[:, :3, 3],
            extrinsics_c2w=extri_4x4,
            cam_names=[
                v['cam_name']
                for vi, v in enumerate(caliscope_data.values())
                if vi in idx
            ],
            image_size=torch.stack([
                torch.tensor(v['image_size'], dtype=torch.int)
                for vi, v in enumerate(caliscope_data.values())
                if vi in idx
            ], dim=0)
        )

    @classmethod
    def from_session_folder(cls, session_folder: Path, idx: Literal['all'] | List[int] = 'all') -> Union['CalibrationData', None]:
        """
        Reads calibration data from a session folder.
        """
        session_folder = Path(session_folder)
        if not session_folder.exists():
            log(f"Session folder does not exist: {session_folder}", 'error')
            return None
        cam_names, intrinsic_files, extrinsic_w2c_files = [], [], []
        image_size_hw = None
        for cam in sorted([_ for _ in (session_folder / 'orbbec').iterdir() if _.is_dir() and _.name.startswith('cam')], key=lambda x: int(x.name.replace('cam', ''))):
            cam_names.append(f'orbbec/{cam.name}')
            intrinsic_files.append(cam / 'parameters' / 'intrinsic.npy')
            extrinsic_w2c_files.append(cam / 'parameters' / 'extrinsic.npy')
            if image_size_hw is None:
                image_size_hw = cv2.imread(next((cam / 'color').glob('*.jpg'))).shape[:2]  # (H, W)
        assert all(p.exists() for p in intrinsic_files) and all(p.exists() for p in extrinsic_w2c_files)
        intrinsics = torch.stack([torch.tensor(np.load(str(p)), dtype=torch.float32) for p in intrinsic_files], dim=0).float()
        extri_4x4 = torch.stack([torch.tensor(np.load(str(p)), dtype=torch.float32) for p in extrinsic_w2c_files], dim=0).float()
        if idx == 'all':
            idx = list(range(len(cam_names)))
        return CalibrationData(
            intrinsics=intrinsics.reshape(-1, 3, 3)[idx],
            dists=torch.stack([
                torch.zeros((5,), dtype=torch.float32)
                for _ in range(len(cam_names))
                if _ in idx
            ], dim=0),
            rotmats=extri_4x4[idx, :3, :3],
            tvecs=extri_4x4[idx, :3, 3],
            extrinsics_c2w=extri_4x4[idx],
            cam_names=[c for _, c in enumerate(cam_names) if _ in idx],
            image_size=torch.stack([
                torch.tensor(image_size_hw, dtype=torch.int)
                for _ in range(len(cam_names))
                if _ in idx
            ], dim=0)
        )


class ColmapReader:
    def __init__(self, capture_path: Path, experiment_name: str = 'for_sp_nm1'):
        self.capture_path = capture_path
        self.experiment_name = experiment_name
        self.colmap_path = capture_path / 'colmap' / experiment_name
        self.colmap_out_path = self.colmap_path / 'colmap_out'
        self.colmap_recon_path = sorted(self.colmap_out_path.iterdir(), key=lambda x: int(x.name))[-1]

    def read(self):
        log(f'[{self.__class__.__name__}::read] Reading COLMAP data (run={self.experiment_name}, path={self.colmap_recon_path})', 'debug')
        with open(self.colmap_recon_path / 'cameras.pkl', 'rb') as f:
            all_cam_data = pickle.load(f)
        return all_cam_data


class CaliscopeReader:
    def __init__(self, capture_path: Path, override_cam_mapping: Optional[str] = None, which_calib_method: str = 'caliscope'):  # 'cam_7-->cam_8, cam_8-->cam_7'
        self.capture_path = capture_path
        if (capture_path / 'raw_color').exists():
            self.camera_names = sorted([d.name for d in (capture_path / 'raw_color').iterdir() if d.is_dir()], key=lambda x: int(x.split(' ')[0]))
        else:
            # new capture structure (v2)
            orbbec_path = capture_path / 'orbbec'
            self.camera_names = sorted([f'orbbec/{d.name}' for d in (orbbec_path.glob('cam*') if len(list(orbbec_path.glob('cam*'))) > 0 else (orbbec_path / 'raw_color').glob('cam*')) if d.is_dir()], key=lambda x: int(x.replace('orbbec/cam', '')))
            if (capture_path / 'sony').exists():
                self.camera_names.append('sony')
            if (capture_path / 'apple').exists():
                self.camera_names.append('apple')
        self.camera_indices_s0 = [int(_.split(' ')[0].replace('orbbec/cam', '').replace('sony', '13').replace('apple', '14')) - 1 for _ in self.camera_names]
        if (capture_path / '__calib__').exists() and (capture_path / '__calib__' / which_calib_method.lower()).exists():
            self.calib_root = capture_path / '__calib__' / which_calib_method.lower()
        elif (capture_path / 'caliscope').exists():
            self.calib_root = capture_path / which_calib_method.lower()
        else:
            raise FileNotFoundError(f'Calibration path not found in {capture_path} (tried {capture_path / "__calib__" / which_calib_method} and {capture_path / which_calib_method})')
        self.override_cam_mapping = dict()
        if override_cam_mapping is not None:
            for pair in override_cam_mapping.split(','):
                pair = pair.strip().split('-->')
                self.override_cam_mapping[pair[0].strip()] = pair[1].strip()

    def read(self, only_indices: Optional[List[int]] = None):
        log(f'[{self.__class__.__name__}::read] Reading Caliscope data (path={self.calib_root})', 'debug')
        with open(self.calib_root / 'config.toml', 'r') as f:
            caliscope_data = toml.load(f)
        assert caliscope_data is not None and caliscope_data['camera_count'] <= len(self.camera_names)
        cam_dict = {}
        for name, data in caliscope_data.items():
            # Thanos: skip non-camera entries
            if name in ['camera_count', 'capture_id', 'capture_date', 'creation_date']:
                continue

            try:
                cam_index_s0 = int(self.override_cam_mapping.get(name, name))
            except ValueError:
                continue
            if cam_index_s0 not in self.camera_indices_s0 or (only_indices is not None and (cam_index_s0 + 1) not in only_indices):
                log(f'[{self.__class__.__name__}::read] {name} --> {cam_index_s0} not in camera_indices_s0 or not in only_indices', 'warning')
                continue
            camera_name = self.camera_names[self.camera_indices_s0.index(cam_index_s0)]
            if name in self.override_cam_mapping and self.override_cam_mapping[name] != name:
                log(f'[{self.__class__.__name__}::read] {name} --> {cam_index_s0} --> {camera_name}', 'warning')
            cam_dict[camera_name] = dict(
                cam_index_s0=cam_index_s0,
                cam_index=cam_index_s0 + 1,
                cam_name=camera_name,
                cam_model='SIMPLE_RADIAL',
                caliscope_key=name,
                intri=dict(K=np.array(data['matrix']), dist=np.array(data['distortions']).reshape(1, 5), H=data['size'][1], W=data['size'][0]),
                rotation=cv2.Rodrigues(np.array(data['rotation']).reshape(3, 1))[0] if isinstance(data['rotation'], list) else data['rotation'],
                translation=np.asarray(data['translation']).flatten() if isinstance(data['translation'], list) else data['translation'],
                extri=None if (data['rotation'] == 'null' or data['translation'] == 'null') else np.hstack([
                    data['rotation'] if isinstance(data['rotation'], np.ndarray) else cv2.Rodrigues(np.array(data['rotation']).reshape(3, 1))[0],
                    np.atleast_2d(data['translation']).reshape((-1, 1))
                ]),
                image_size=(data['size'][1], data['size'][0]),  # H, W
            )
        # sort cam_dict by cam_index_s0
        cam_dict = dict(sorted(cam_dict.items(), key=lambda item: item[1]['cam_index_s0']))
        return cam_dict

    def create_files(self, cam_dict):
        with open(self.calib_root / 'config.toml', 'r') as f:
            caliscope_data = toml.load(f)
        assert caliscope_data is not None and caliscope_data['camera_count'] == len(self.camera_names)
        for name, data in caliscope_data.items():
            if not name.startswith('cam_'):
                cam_dict[name] = data
                continue
            cam_index_s0 = int(self.override_cam_mapping.get(name, name).replace('cam_', '')) - 1
            camera_name = self.camera_names[cam_index_s0]
            assert cam_dict[camera_name]['caliscope_key'] == name
            data['matrix'] = cam_dict[camera_name]['intri']['K'].tolist()
            data['distortions'] = cam_dict[camera_name]['intri']['dist'].flatten().tolist()
            data['translation'] = cam_dict[camera_name]['extri_translation'].flatten().tolist()
            data['rotation'] = cam_dict[camera_name]['extri_rotation'].tolist()
            data['extri_matrix'] = cam_dict[camera_name]['extri'].tolist()
        log(f'[{self.__class__.__name__}::create_files] Writing to {self.calib_root / "config.toml"}', 'debug')
        with open(self.calib_root / 'config.toml', 'w') as f:
            toml.dump(caliscope_data, f)
        with open(self.calib_root / 'cameras.pkl', 'wb') as f:
            pickle.dump(cam_dict, f)


class MultiCamCalibReader(CaliscopeReader):
    def __init__(self, capture_path: Path, override_cam_mapping: Optional[str] = None):
        super().__init__(capture_path, override_cam_mapping, which_calib_method='multicamcalib')

    def read(self, only_indices: Optional[List[int]] = None):
        log(f'[{self.__class__.__name__}::read] Reading MultiCamCalib data (path={self.calib_root})', 'debug')
        with open(self.calib_root / 'cam_params_final.json', 'r') as f:
            mcc_data = json.load(f)
        assert mcc_data is not None and len(mcc_data) <= len(self.camera_names)
        cam_dict = {}
        for name, data in mcc_data.items():
            cam_index_s0 = int(self.override_cam_mapping.get(name, name).replace('cam_', ''))
            if cam_index_s0 not in self.camera_indices_s0 or (only_indices is not None and (cam_index_s0 + 1) not in only_indices):
                log(f'[{self.__class__.__name__}::read] {name} --> {cam_index_s0} not in camera_indices_s0 or not in only_indices', 'warning')
                continue
            camera_name = self.camera_names[self.camera_indices_s0.index(cam_index_s0)]
            if name in self.override_cam_mapping and self.override_cam_mapping[name] != name:
                log(f'[{self.__class__.__name__}::read] {name} --> {cam_index_s0} --> {camera_name}', 'warning')
            data['size'] = (3840, 2160) if 'size' not in data else data['size']
            extri = None if (data['rvec'] == 'null' or data['tvec'] == 'null') else np.hstack([
                data['rvec'] if isinstance(data['rvec'], np.ndarray) else cv2.Rodrigues(np.array(data['rvec']).reshape(3, 1))[0],
                np.atleast_2d(data['tvec']).reshape((-1, 1))
            ])
            if extri is not None:
                extri[:, 3] = extri[:, 3] / 1000  # convert from mm to m
                rot_y_90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])  # MulticamCalib's output is rotated by 90 degrees around y-axis, so rotate by -90 degrees along y-axis
                extri[:3, :3] = extri[:3, :3] @ rot_y_90
                extri_4x4 = np.eye(4, dtype=extri.dtype)
                extri_4x4[:3, :4] = extri[:3, :4]
                extri = extri_4x4[:3, :4]
            cam_dict[camera_name] = dict(
                cam_index_s0=cam_index_s0,
                cam_index=cam_index_s0 + 1,
                cam_name=camera_name,
                cam_model='SIMPLE_RADIAL',
                multicamcalib_key=name,
                intri=dict(
                    K=np.array([
                        [data['fx'], 0, data['cx']],
                        [0, data['fy'], data['cy']],
                        [0, 0, 1]
                    ]),
                    dist=np.array([
                        data['k1'], data['k2'], data['p1'], data['p2'], data['k3']
                    ]).reshape(1, 5),
                    H=data['size'][1],
                    W=data['size'][0]
                ),
                rotation=extri[:3, :3] if extri is not None else None,
                translation=extri[:3, 3] if extri is not None else None,
                extri=extri,
                image_size=(data['size'][1], data['size'][0]),  # H, W
            )
        # sort cam_dict by cam_index_s0
        cam_dict = dict(sorted(cam_dict.items(), key=lambda item: item[1]['cam_index_s0']))
        return cam_dict


if __name__ == '__main__':
    from pathlib import Path

    # SUBJECT = 'Thobias'
    # PERF = True
    # CALIB = False
    # DATE = datetime(2024, 9, 17, 8, 43, 24)
    # capture_path_ = CaptureSession.capture_path(subject=SUBJECT, calib=CALIB, perf=PERF, date=DATE)
    # caliscope_data_ = CaliscopeReader(capture_path_, override_cam_mapping='cam_7-->cam_8, cam_8-->cam_7').read()
    # colmap_data_ = ColmapReader(capture_path_).read()
    # print(caliscope_data_)

    capture_path_ = Path('/home/charisoudis/PyCharmMiscProject/irc-hslu/capturestudio/out/Captures_Apr_May_2025/Aggregated/Thanos_2_Calib_1')
    calib_data_ = CalibrationData.from_session(capture_path_)
    print(calib_data_)
