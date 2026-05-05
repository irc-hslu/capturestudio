import itertools
from bisect import bisect_right
from pathlib import Path
from typing import List, Literal, Union, Tuple, Dict, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from reconstruction.primitive.stereo import StereoUtils
from utils.flow import FlowUtils
from utils.misc import log, PathUtils


class SingleSessionDataset(Dataset):
    """
    Dataset for a single session (ALL cameras in a session).
    """

    def __init__(self,
                 session_root: Union[Path, str],
                 cam_indices: List[int],
                 return_numpy: bool = False,
                 depth_filter: Optional[Literal['aligned', 'bilateral_spatial', 'bilateral_temporal']] = 'bilateral_spatial',
                 transforms: Optional[List[Callable]] = None,
                 return_of: bool = False,
                 rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None):
        self.session_root = Path(session_root)
        self.session_name = self.session_root.name
        self.cam_indices = cam_indices
        self.camera_folders = [session_root / 'orbbec' / f'cam{index:02d}' for index in self.cam_indices]
        self.rgb_file_paths = [
            sorted((self.camera_folders[i] / 'color').glob('*.jpg'), key=lambda x: int(x.stem))
            for i in range(len(self.camera_folders))
        ]
        assert all(len(rgb_files) > 0 for rgb_files in self.rgb_file_paths), "No RGB files found in the specified camera folders."
        assert all(len(rgb_files) == len(self.rgb_file_paths[0]) for rgb_files in self.rgb_file_paths), f"All cameras must have the same number of RGB files, got {[len(_) for _ in self.rgb_file_paths]} cameras with varying file counts."
        if depth_filter is not None:
            assert depth_filter in ['aligned', 'bilateral_spatial', 'bilateral_temporal']
            self.depth_file_paths = [
                sorted((self.camera_folders[i] / ('depth_aligned' if depth_filter == 'aligned' else f'depth_filtering_{depth_filter}')).glob('*.png'), key=lambda x: int(x.stem))
                for i in range(len(self.camera_folders))
            ]
            # assert all(len(depth_files) > 0 for depth_files in self.depth_file_paths), "No depth files found in the specified camera folders."
            self.n_frames = min(len(r) for r in self.rgb_file_paths)
            # self._ensure_depth_maps_exist(depth_filter=depth_filter)
            self.use_depth = True
        else:
            self.use_depth = False
            self.n_frames = min(len(r) for r in self.rgb_file_paths)

        self.return_of = return_of
        self.rotate = rotate

        # assert all(len(depth_files) == len(self.depth_file_paths[0]) for depth_files in self.depth_file_paths), f"All cameras must have the same number of depth files, got {[len(_) for _ in self.depth_file_paths]} cameras with varying file counts."
        # assert len(self.rgb_file_paths) == len(self.depth_file_paths), "Number of RGB and depth cameras must match."
        # assert len(self.depth_file_paths[0]) == len(self.rgb_file_paths[0]), "Number of depth frames must match the number of RGB frames ({} != {})".format(len(self.depth_file_paths[0]), len(self.rgb_file_paths[0]))
        self.return_numpy = return_numpy

        first_src_image = cv2.imread(str(self.rgb_file_paths[0][0]))
        if self.rotate:
            first_src_image = cv2.rotate(first_src_image, getattr(cv2, f'ROTATE_{self.rotate.upper()}'))
        self.src_image_size_hw = first_src_image.shape[:2]
        self.transforms = transforms if transforms is not None else []

        self._ensure_segmentation_masks_exist()
        self._depth_dirname = 'depth_aligned' if depth_filter is None or depth_filter == 'aligned' else f'depth_filtering_{depth_filter}'
        log(f'[{self.__class__.__name__}::__init__] Number of frames in dataset: {self.n_frames}', 'info')

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        """
        Returns a tensors with RGB, masks, and depth images for each camera at the specified index.
        """
        rgb_file_paths = [
            self.rgb_file_paths[i][idx]
            for i in range(len(self.camera_folders))
        ]
        depth_file_paths = [
            self.depth_file_paths[i][idx] if self.depth_file_paths[i] and self.depth_file_paths[i][idx].exists() else None
            for i in range(len(self.camera_folders))
        ] if self.use_depth else [None] * len(self.camera_folders)
        rgb_images, masks, depths, cam_indices_rel = [], [], [], []
        optical_flows = []
        # cam_indices_global = []
        for cam_idx_rel, (cam_idx_global, rgb_path, depth_path) in enumerate(zip(self.cam_indices, rgb_file_paths, depth_file_paths)):
            rgb_image = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            mask_path = Path(str(rgb_path).replace('color', 'mask'))
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if depth_path is not None:
                depth_map = PathUtils.read_file(depth_path, png_type='depth').astype(np.float32) * 1e-3
            else:
                depth_map = np.zeros(rgb_image.shape[:2], dtype=np.float32)
            for transform in self.transforms:
                rgb_image, mask_image, depth_map = transform(rgb_image, mask_image, depth_map, cam_idx_s0=cam_idx_rel)
            rgb_images.append(rgb_image)
            masks.append(mask_image)
            depths.append(depth_map)
            if self.return_of:
                of_path = Path(str(rgb_path).replace('color', 'flow_bwd')).with_suffix('.png')
                of = FlowUtils.read_flow_from_png_custom(of_path).cpu()
                if self.rotate:
                    of = FlowUtils.rotate_flow(of, self.rotate)
                if self.transforms:
                    of = FlowUtils.resize_crop_flow(of, *rgb_image.shape[:2])
                optical_flows.append(of.numpy())
            # cam_indices_global.append(cam_idx_global)
            cam_indices_rel.append(cam_idx_rel)  # 0-based index
        # create numpy data
        img, mask, depth, cam_idx_rel = np.stack(rgb_images, axis=0), np.stack(masks, axis=0), np.stack(depths, axis=0), np.array(cam_indices_rel, dtype=np.int64)
        if optical_flows:
            optical_flows = np.stack(optical_flows, axis=0)
        if self.return_numpy:
            if self.return_of:
                return img, mask, depth, optical_flows, cam_idx_rel
            return img, mask, depth, cam_idx_rel

        # to torch
        # @formatter:off
        img_torch, mask_torch, depth_torch, cam_indices_rel = \
            torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0, \
            (torch.from_numpy(mask).unsqueeze(1).float() / 255.0) > 0.5, \
            torch.from_numpy(depth).unsqueeze(1), \
            torch.from_numpy(cam_idx_rel)
        if self.return_of:
            optical_flows_torch = torch.from_numpy(optical_flows)
            return img_torch, mask_torch, depth_torch, cam_indices_rel, optical_flows_torch
        # @formatter:on
        return img_torch, mask_torch, depth_torch, cam_indices_rel

    def _ensure_segmentation_masks_exist(self):
        for camera_folder in self.camera_folders:
            mask_folder = camera_folder / 'mask'
            if not mask_folder.exists() or not any(mask_folder.glob('*.jpg')) or len(list(mask_folder.glob('*.jpg'))) < self.n_frames - 1:  # -1 because frame extraction could result in one less frame
                raise NotImplementedError(
                    f"Segmentation masks are missing for camera {camera_folder.name} in session {self.session_name}. "
                    "Please generate segmentation masks in irc-hslu/capturestudio."
                )
            elif len(list(mask_folder.glob('*.jpg'))) == self.n_frames - 1:
                self.n_frames -= 1

    def _ensure_depth_maps_exist(self, depth_filter: Optional[str] = None):
        should_align = False
        should_filter = [False] * len(self.camera_folders)
        session_name = None
        for i, cam_dir in enumerate(self.camera_folders):
            depth_aligned_folder = cam_dir / 'depth_aligned'
            if not depth_aligned_folder.exists() or not any(depth_aligned_folder.glob('*.png')) or len(list(depth_aligned_folder.glob('*.png'))) < self.n_frames:  # -1 because frame extraction could result in one less frame
                should_align = True
                should_filter = True
            elif depth_filter in ['bilateral_spatial', 'bilateral_temporal'] and len(list((cam_dir / f"depth_filtering_{depth_filter}").glob('*.png'))) < self.n_frames:
                should_filter[i] = True
            if session_name is None:
                session_name = cam_dir.parent.parent.name
        if should_align:
            raise NotImplementedError("Aligned depth maps generation is not implemented. Please use irc-hslu/capturestudio.")
        if any(should_filter):
            raise NotImplementedError("Filtering depth maps is not implemented. Please use irc-hslu/capturestudio.")


class MultiSessionDataset(Dataset):
    """
    Encapsulates multiple session datasets.
    Also loads calibration data and provides stereo rectification if needed.
    """
    CALIBRATION_DATA = None

    def __init__(self,
                 calibration_session_name: str,
                 calibration_method: Union[Literal['Caliscope'], Literal['MultiCamCalib']],
                 cam_indices: List[int],
                 session_names: List[str],
                 depth_filter: Optional[Literal['bilateral_spatial', 'bilateral_temporal', 'aligned']] = 'bilateral_spatial',
                 use_stereo: bool = False,
                 stereo_order_matters: bool = True,
                 n_cams_per_sample: int = -1,
                 target_image_size_hw: Tuple[int, int] = (1024, 1024),
                 apply_intrinsics_fix: bool = False,
                 is_cam_indices_s0: bool = False,
                 return_of: bool = False,
                 rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None):
        self.depth_filter = depth_filter
        self.calibration_session_name = calibration_session_name
        self.calibration_method = calibration_method
        self.session_names = session_names
        self.session_roots = []
        for session_name in session_names:
            _root = PathUtils.capturestudio_cache_path()
            for _candidate in _root.glob('Captures_*'):
                if (_candidate / session_name).exists() and (_candidate / session_name).is_dir():
                    self.session_roots.append(_candidate / session_name)
                    break
            else:
                raise FileNotFoundError(f"Session {session_name} not found")
        self.cam_indices = cam_indices
        self.cam_indices_s0 = cam_indices if is_cam_indices_s0 else [_ - 1 for _ in cam_indices]  # convert to 0-based indices
        self.is_cam_indices_s0 = is_cam_indices_s0
        self.target_image_size_hw = target_image_size_hw
        self.rotate = rotate
        self.use_stereo = use_stereo
        if use_stereo:
            assert n_cams_per_sample in [-1, 2]
            if n_cams_per_sample == -1:
                n_cams_per_sample = 2
        elif n_cams_per_sample == -1:
            n_cams_per_sample = len(cam_indices)
        self.n_cams_per_sample = n_cams_per_sample
        self.cam_indices_tuples = list(filter(lambda x: np.all(np.abs(np.diff(x)) == 1.0), getattr(itertools, 'permutations' if use_stereo and stereo_order_matters else 'combinations')(range(len(cam_indices)), n_cams_per_sample)))  # in stereo order maters (LR != RL)

        # read calibration data
        self.apply_intrinsics_fix = apply_intrinsics_fix
        if self.apply_intrinsics_fix:
            log('[MultiSessionDataset::__init__] Applying intrinsics fix for calibration data.', 'info')
        from utils.calib import CalibrationData
        _root = PathUtils.capturestudio_cache_path()
        for _candidate in _root.glob('Captures_*'):
            if (_candidate / self.calibration_session_name).exists() and (_candidate / self.calibration_session_name).is_dir():
                _calibration_session_path = _candidate / self.calibration_session_name
                break
        else:
            raise FileNotFoundError(f"Calibration session {self.calibration_session_name} not found")
        all_calibration_data = CalibrationData.from_session(
            session_path=_calibration_session_path,
            method=calibration_method
        )
        if target_image_size_hw is not None and rotate is not None:
            all_calibration_data = (all_calibration_data
                                    .rotate(rotate)
                                    .resize(*target_image_size_hw, apply_intrinsics_fix=apply_intrinsics_fix))
        elif target_image_size_hw is None and rotate is not None:
            all_calibration_data = all_calibration_data.rotate(rotate)
        elif target_image_size_hw is not None and rotate is None:
            all_calibration_data = all_calibration_data.resize(*target_image_size_hw, apply_intrinsics_fix=apply_intrinsics_fix)
        cam_indices_s0 = [_ - 1 for _ in list(cam_indices)]
        self.calibration_data = all_calibration_data[cam_indices_s0]

        # read training and validation data
        self.training_datasets = {
            len(ds) * len(self.cam_indices_tuples): ds
            for ds in [
                SingleSessionDataset(
                    session_root,
                    cam_indices,
                    return_numpy=use_stereo,
                    depth_filter=depth_filter,
                    transforms=self.calibration_data.get_preprocessing_transforms(),
                    return_of=return_of,
                    rotate=rotate
                )
                for session_root in self.session_roots
            ]
        }
        self.return_of = return_of
        assert len(self.training_datasets) > 0, "No training datasets found for the specified sessions."
        self.src_image_size_hw = self.training_datasets[next(iter(self.training_datasets))].src_image_size_hw
        assert all(ds.src_image_size_hw == self.src_image_size_hw for ds in self.training_datasets.values()), "All datasets must have the same source image size."
        self.dataset_lengths = [len(ds) * len(self.cam_indices_tuples) for ds in self.training_datasets.values()]
        self.dataset_lengths_cum = np.cumsum(self.dataset_lengths).tolist()
        self.datasets = list(self.training_datasets.values())

        # setup stereo rectification if needed
        self.rectification_data: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}  # (left_cam_index, right_cam_index) -> {'ref_intrinsic': ..., 'other_intrinsic': ..., 'ref_extrinsic_w2c': ..., 'other_extrinsic_w2c': ..., 'tf_x': ..., 'ref_rectify_mat_x': ..., 'ref_rectify_mat_y': ..., 'other_rectify_mat_x': ..., 'other_rectify_mat_y': ...}
        if use_stereo:
            self._create_rectification_data()

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, idx):
        """
        Returns a tuple of RGB images, masks, and depth images for the specified index.
        The index is relative to the concatenated dataset of all sessions.

        Parameters:
        ----------
        idx: int
            The index of the item to retrieve from the combined dataset.

        Returns:
        -------
        tuple
            A tuple containing:
            - RGB images (torch.Tensor): Shape (N, C, H, W) where N is the number of cameras, C is the number of channels (3 for RGB), H is height, and W is width. If stereo, N=2, else N=1.
            - Masks (torch.Tensor): Shape (N, 1, H, W)
            - Depth images (torch.Tensor): Shape (N, 1, H, W) (in meters).
            If stereo, the following additional data is included:
            - Intrinsics (torch.Tensor): Shape (N, 3, 3). If stereo, it contains the rectified intrinsics for the stereo pair.
            - Extrinsics (torch.Tensor): Shape (N, 4, 4). If rectified extrinsics for the stereo pair.
            - Horizontal translation factor for stereo pair (torch.Tensor): Shape (N,).
            EndIf
            - Camera indices (torch.Tensor): Shape (N,). The indices of the cameras in the stereo pair, from 0 to (C-1), where C is the number of cameras.
            - Frame index (torch.Tensor): Shape (N,). The index of the frame in the dataset, from 0 to (T-1), where T is the total number of frames in the dataset.
        """
        # TODO: erase me
        if idx >= self.dataset_lengths[0]:
            idx = self.dataset_lengths[0] - 1 # clip
        dataset_index = bisect_right(self.dataset_lengths_cum, idx)
        if not 0 <= dataset_index < len(self.training_datasets):
            raise IndexError(f"Index out of range for the combined dataset: {dataset_index}")
        dataset = self.datasets[dataset_index]
        sample_idx = (idx - sum(self.dataset_lengths[:dataset_index])) // len(self.cam_indices_tuples)
        cam_indices = list(self.cam_indices_tuples[sample_idx % len(self.cam_indices_tuples)])
        try:
            all_cam_data = dataset[sample_idx]
        except (cv2.error,):
            log(f"Error reading sample {sample_idx} from dataset {dataset.session_name}. Skipping this sample.", 'warning')
            return self.__getitem__(idx + 1)
        cam_data = [
            cd[cam_indices]
            for cd in all_cam_data
        ]
        cam_out_data = [[] for _ in range(5 + (1 if self.return_of else 0))]
        for c in range(len(cam_indices)):
            img, mask, depth, intri, extri = cam_data[0][c], cam_data[1][c], cam_data[2][c], self.calibration_data.intrinsics[cam_indices[c]].clone(), self.calibration_data.extrinsics_c2w[cam_indices[c]].clone()
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
            if not isinstance(depth, torch.Tensor):
                depth = torch.from_numpy(depth).unsqueeze(0).float()
            cam_out_data[0].append(img)
            cam_out_data[1].append(mask)
            cam_out_data[2].append(depth)
            cam_out_data[3].append(intri)
            cam_out_data[4].append(extri)
            if self.return_of:
                of = cam_data[-1][c]
                if not isinstance(of, torch.Tensor):
                    of = torch.from_numpy(of)
                cam_out_data[5].append(of)
        if self.return_of:
            cam_out_data = [
                torch.stack(cam_out_data[0], dim=0),  # RGB images
                torch.stack(cam_out_data[1], dim=0),  # masks
                torch.stack(cam_out_data[2], dim=0),  # depth maps
                torch.stack(cam_out_data[3], dim=0),  # intrinsics
                torch.stack(cam_out_data[4], dim=0),  # extrinsics
                torch.tensor(cam_indices).long(),  # camera indices
                torch.stack(cam_out_data[5], dim=0),  # OFs
            ]
        else:
            cam_out_data = [
                torch.stack(cam_out_data[0], dim=0),  # RGB images
                torch.stack(cam_out_data[1], dim=0),  # masks
                torch.stack(cam_out_data[2], dim=0),  # depth maps
                torch.stack(cam_out_data[3], dim=0),  # intrinsics
                torch.stack(cam_out_data[4], dim=0),  # extrinsics
                torch.tensor(cam_indices).long(),  # camera indices
            ]
        # frame = torch.tensor([sample_idx] * len(cam_out_data[0]), dtype=torch.int64)  # frame index for stereo pair
        # cam_out_data.append(frame)

        if self.use_stereo:
            img_l, mask_l, depth_l = cam_data[0][0], cam_data[1][0], cam_data[2][0]
            img_r, mask_r, depth_r = cam_data[0][1], cam_data[1][1], cam_data[2][1]

            # rectify the data for stereo pair
            rectification_data_lr = self.rectification_data[(cam_indices[0], cam_indices[1])]
            img_l, img_r, mask_l, mask_r, depth_l, depth_r = StereoUtils.rectify(
                img_l, img_r,
                rectification_data_lr,
                mask_l, mask_r,
                depth_l, depth_r
            )
            intri_l_torch, extri_l_torch = torch.from_numpy(rectification_data_lr['ref_intrinsic'].copy()), torch.from_numpy(np.linalg.inv(rectification_data_lr['ref_extrinsic_w2c'].copy()))
            intri_r_torch, extri_r_torch = torch.from_numpy(rectification_data_lr['other_intrinsic'].copy()), torch.from_numpy(np.linalg.inv(rectification_data_lr['other_extrinsic_w2c'].copy()))
            # convert to tensors
            img_l_torch, img_r_torch = torch.from_numpy(img_l).permute(2, 0, 1).float() / 255.0, torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
            mask_l_torch, mask_r_torch = torch.from_numpy(mask_l).unsqueeze(0).float() / 255.0, torch.from_numpy(mask_r).unsqueeze(0).float() / 255.0
            depth_l_torch, depth_r_torch = torch.from_numpy(depth_l).unsqueeze(0).float(), torch.from_numpy(depth_r).unsqueeze(0).float()
            tfx_l_tensor, tfx_r_tensor = torch.tensor(rectification_data_lr['tf_x']).float(), torch.tensor(rectification_data_lr['tf_x']).float()
            # append rectified data to the output
            cam_out_data += [
                torch.stack([img_l_torch, img_r_torch], dim=0),  # rectified RGB images
                torch.stack([mask_l_torch, mask_r_torch], dim=0),  # rectified  masks
                torch.stack([depth_l_torch, depth_r_torch], dim=0),  # rectified depth maps
                torch.stack([intri_l_torch, intri_r_torch], dim=0).float(),  # rectified intrinsics
                torch.stack([extri_l_torch, extri_r_torch], dim=0).float(),  # rectified extrinsics
                torch.stack([tfx_l_tensor, tfx_r_tensor], dim=0),  # translation factor of rectified pair
            ]

        return tuple(cam_out_data)

    @property
    def intrinsics_target(self):
        # adjust intrinsics to the target image size
        intrinsics = self.calibration_data.intrinsics.clone().to(dtype=torch.float32)
        scale = max(self.target_image_size_hw[0] / self.src_image_size_hw[0], self.target_image_size_hw[1] / self.src_image_size_hw[1])
        intrinsics[:, 0, 0] *= scale
        intrinsics[:, 1, 1] *= scale
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * scale - (self.target_image_size_hw[1] - self.src_image_size_hw[1]) // 2
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * scale - (self.target_image_size_hw[0] - self.src_image_size_hw[0]) // 2
        return intrinsics

    @property
    def rotmats(self):
        return self.extrinsics[:, :3, :3]

    @property
    def tvecs(self):
        return self.calibration_data.extrinsics_c2w[:, :3, 3]

    @property
    def extrinsics(self):
        return self.calibration_data.extrinsics_c2w

    def _create_rectification_data(self) -> None:
        from reconstruction.primitive.stereo import StereoUtils

        # create rectified stereo pairs
        self.rectification_data = {}
        intrinsics_ori, extrinsics_w2c_ori = self.calibration_data.intrinsics.cpu().numpy(), self.calibration_data.extrinsics_w2c.cpu().numpy()
        for il, ir in self.cam_indices_tuples:
            self.rectification_data[(il, ir)] = StereoUtils.compute_rectification_data(
                ref_intrinsic=intrinsics_ori[il].copy(),
                ref_extrinsic_w2c=extrinsics_w2c_ori[il].copy(),
                other_intrinsic=intrinsics_ori[ir].copy(),
                other_extrinsic_w2c=extrinsics_w2c_ori[ir].copy(),
                image_size_hw=self.src_image_size_hw,
                use_cache=True,
                cache_key_prefix=f"{self.calibration_session_name}_{self.calibration_method}".lower()
            )
            if (ir, il) not in self.rectification_data:
                self.rectification_data[(ir, il)] = StereoUtils.compute_rectification_data(
                    ref_intrinsic=intrinsics_ori[ir].copy(),
                    ref_extrinsic_w2c=extrinsics_w2c_ori[ir].copy(),
                    other_intrinsic=intrinsics_ori[il].copy(),
                    other_extrinsic_w2c=extrinsics_w2c_ori[il].copy(),
                    image_size_hw=self.src_image_size_hw,
                    use_cache=True,
                    cache_key_prefix=f"{self.calibration_session_name}_{self.calibration_method}".lower()
                )

    def get_camera_orbit(self, orbit_type: Literal['interpolated', 'audience'] = 'interpolated', floor_wall_data: Optional[dict] = None, **trajectory_kwargs):
        # Estimate floor and wall
        from reconstruction.primitive.pcd import RGBDImage
        from reconstruction.vis.dataset_visualizer import DatasetVisualizer
        ds_with_unfiltered_depth = self if self.depth_filter == 'aligned' else self.__class__(
            calibration_session_name=self.calibration_session_name,
            calibration_method=self.calibration_method,
            cam_indices=self.cam_indices,
            session_names=self.session_names,
            depth_filter='aligned',
            use_stereo=self.use_stereo,
            target_image_size_hw=self.target_image_size_hw,
            apply_intrinsics_fix=self.apply_intrinsics_fix,
            is_cam_indices_s0=self.is_cam_indices_s0,
            return_of=False,
            rotate=self.rotate
        )
        first_rgbd_images = DatasetVisualizer.sft_format_to_rgbd_images(ds_with_unfiltered_depth[0])
        # first_pcd = PixelPoints.from_partials(*[_.unproject() for _ in first_rgbd_images]).save_ply(f'/root/capturestudio2/src/{self.session_names[0].split("_")[0].lower()}_first_pcd_stitched.ply')
        # print('done')
        # exit(0)

        if floor_wall_data is None:
            floor_wall_data = RGBDImage.estimate_floor(
                *first_rgbd_images,
                export=False,
                export_path=f'/root/capturestudio2/src/{self.session_names[0].lower()}_floor_wall.png',
                wall_overshoot_m=trajectory_kwargs.pop('wall_overshoot_m', 2.0),
            )
        # FIX: if wall was estimated "in front of" the chord, make the bulging "toward" the wall so that it ends up being away from it
        wall_overshoot_m = floor_wall_data.pop('wall_overshoot_m', 2.0)
        if wall_overshoot_m < 0 and 'bulge_mode' not in trajectory_kwargs:
            trajectory_kwargs['bulge_mode'] = 'toward'

        if orbit_type == 'interpolated':
            from reconstruction.vis.cam_orbit import InterpolatedCameraOrbit
            floor_wall_data = {}
            return InterpolatedCameraOrbit.from_session(
                calibration_session=self.calibration_session_name,
                calibration_method=self.calibration_method,
                trajectory_idx=self.cam_indices,
                reconstruction_idx=self.cam_indices,
                image_size_hw=self.target_image_size_hw,
                rotate=self.rotate,
                **(floor_wall_data | trajectory_kwargs)
            )
        if orbit_type == 'audience':
            # Create a trajectory anchored on the first and last's camera position, and that is curved and in a plane parallel to the floor
            from reconstruction.vis.cam_orbit import AudienceViewAnchoredCameraOrbit
            return AudienceViewAnchoredCameraOrbit.from_session(
                calibration_session=self.calibration_session_name,
                calibration_method=self.calibration_method,
                trajectory_idx=self.cam_indices,
                reconstruction_idx=self.cam_indices,
                image_size_hw=self.target_image_size_hw,
                rotate=self.rotate,
                **(floor_wall_data | trajectory_kwargs)
            )
        raise ValueError('orbit_type not supported: {}'.format(orbit_type))


class MultiSessionDataLoader(torch.utils.data.DataLoader):
    """
    DataLoader for MultiSessionDataset.
    We do not convert data to torch; potentially we only convert to RGBD images. This dataloader is used for parallel data fetching and conversion to RGBD images.
    """

    @classmethod
    def for_eval(cls, convert_to_rgbd_images: bool = True, batch_size: int = 4, num_workers: int = 8, persistent_workers: bool = False, **ds_kwargs) -> 'MultiSessionDataLoader':
        def no_collate_fn(batch):
            return batch

        def collate_to_rgbd_images(batch):
            from reconstruction.vis.dataset_visualizer import DatasetVisualizer
            return [DatasetVisualizer.sft_format_to_rgbd_images(b) for b in batch]

        collate_fn = collate_to_rgbd_images if convert_to_rgbd_images else no_collate_fn
        dataset = MultiSessionDataset(**ds_kwargs)
        return cls(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers
        )


if __name__ == "__main__":
    session_name_ = "Thanos_2_Perf_1"
    calibration_session_name_ = "Thanos_2_Calib_1"
    cam_indices_ = [4, 5, 7, 8]
    dataset_ = MultiSessionDataset(calibration_session_name_, 'MultiCamCalib', cam_indices_, [session_name_], use_stereo=False)
    print(f"Dataset length: {len(dataset_)}")
    print(f"Dataset intrinsics: {dataset_.intrinsics_target.shape}")
    print(f"Dataset rotmats: {dataset_.rotmats.shape}")
    print(f"Dataset tvecs: {dataset_.tvecs.shape}")
    for i_ in range(len(dataset_)):
        if dataset_.use_stereo:
            rgb_, mask_, depth_, intri_, extri_, tf_x_, cam_indices_rel_ = dataset_[i_]
        else:
            rgb_, mask_, depth_, intri_, extri_, cam_indices_rel_ = dataset_[i_]
            tf_x_ = torch.zeros(len(dataset_.cam_indices), dtype=torch.float32)
        print(f"[{i_:04d}] RGB {rgb_.shape} ({rgb_.min():.1f}, {rgb_.max():.1f}) | Mask {mask_.shape} ({mask_.min():.1f}, {mask_.max():.1f}) | Depth {depth_.shape} ({depth_.min():.1f}, {depth_.max():.1f}) | In {intri_.shape} Ex {extri_.shape} TFx {tf_x_.shape} | C={cam_indices_rel_.tolist()}")
        if i_ == 10:
            break
        break
    """
    # Output example:
    [0000] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 7.8) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[0, 0] C=[0, 1]
    [0001] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.7) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[0, 0] C=[1, 2]
    [0002] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.8) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[0, 0] C=[2, 3]
    [0003] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 7.8) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[0, 0] C=[1, 0]
    [0004] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.7) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[0, 0] C=[2, 1]
    [0005] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.8) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[0, 0] C=[3, 2]
    [0006] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.8) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[1, 1] C=[0, 1]
    [0007] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.8) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[1, 1] C=[1, 2]
    [0008] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.3) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[1, 1] C=[2, 3]
    [0009] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.8) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[1, 1] C=[1, 0]
    [0010] RGB torch.Size([2, 3, 2160, 3840]) (0.0, 1.0) | Mask torch.Size([2, 1, 2160, 3840]) (0.0, 1.0) | Depth torch.Size([2, 1, 2160, 3840]) (0.0, 6.8) | In torch.Size([2, 3, 3]) Ex torch.Size([2, 4, 4]) TFx torch.Size([2]) | T=[1, 1] C=[2, 1]
    """
    exit(0)
