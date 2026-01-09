import itertools
from bisect import bisect_right
from pathlib import Path
from typing import List, Literal, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from reconstruction.primitive.stereo import StereoUtils
from utils.misc import PathUtils, log

THUMAN_ROOT = Path('/root/DATASETS/THuman2_1')
CALIBRATION_CAPTURES_ROOT = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025'




class MultiSessionDataset(Dataset):
    """
    Encapsulates multiple session datasets.
    Also loads calibration data and provides stereo rectification if needed.
    """
    CALIBRATION_DATA = None

    def __init__(self,
                 calibration_session_name: str,
                 calibration_method: Literal['Caliscope', 'MultiCamCalib'],
                 cam_indices: List[int],
                 depth_filter: Optional[Literal['aligned']],
                 use_stereo: bool = False,
                 stereo_order_matters: bool = True,
                 n_cams_per_sample: int = -1,
                 target_image_size_hw: Tuple[int, int] = (1024, 1024),
                 is_cam_indices_s0: bool = False):
        assert depth_filter is None or depth_filter == 'aligned', "Only 'aligned' depth filter is supported or None."
        self.calibration_session_name = calibration_session_name
        self.calibration_method = calibration_method
        self.session_roots = [THUMAN_ROOT / f'rendered_{calibration_session_name.lower()}']
        self.cam_indices = cam_indices
        self.cam_indices_s0 = cam_indices if is_cam_indices_s0 else [_ - 1 for _ in cam_indices]  # convert to 0-based indices
        self.is_cam_indices_s0 = is_cam_indices_s0
        self.target_image_size_hw = target_image_size_hw
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
        if self.__class__.CALIBRATION_DATA is None:
            from utils.calib import CalibrationData
            self.__class__.CALIBRATION_DATA = CalibrationData.from_session_folder(
                session_folder=THUMAN_ROOT / f'rendered_{calibration_session_name.lower()}'
            )
        all_calibration_data = self.__class__.CALIBRATION_DATA.resize(*target_image_size_hw, apply_intrinsics_fix=True)
        self.calibration_data = all_calibration_data[self.cam_indices_s0]

        # read training and validation data
        from reconstruction.data.capturestudio import SingleSessionDataset
        self.training_datasets = {
            len(ds) * len(self.cam_indices_tuples): ds
            for ds in [
                SingleSessionDataset(session_root, cam_indices, return_numpy=use_stereo, depth_filter=depth_filter, transforms=self.calibration_data.get_preprocessing_transforms())
                for session_root in self.session_roots
            ]
        }
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
        dataset_index = bisect_right(self.dataset_lengths_cum, idx)
        if not 0 <= dataset_index < len(self.training_datasets):
            raise IndexError("Index out of range for the combined dataset.")
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
        cam_out_data = [[] for _ in range(5)]
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

    def get_camera_orbit(self, **trajectory_kwargs):
        from reconstruction.vis.cam_orbit import InterpolatedCameraOrbit
        return InterpolatedCameraOrbit.from_session(
            calibration_session=self.calibration_session_name,
            calibration_method=self.calibration_method,
            reconstruction_idx=self.cam_indices_s0,
            image_size_hw=self.target_image_size_hw,
            calibration_data_from_folder=THUMAN_ROOT / f'rendered_{self.calibration_session_name.lower()}',
            **trajectory_kwargs
        )


class MultiSessionDataLoader(DataLoader):
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

