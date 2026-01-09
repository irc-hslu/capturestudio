import gc
import sys
from pathlib import Path

src_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_root))
sys.path.append(str(src_root.parent / 'resources' / 'submodules' / 'foundationstereo'))

from dotenv import load_dotenv

load_dotenv(str(src_root.parent / '.env'))

from itertools import chain
from pathlib import Path
from typing import Union, Literal, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from reconstruction.primitive.pcd import RGBDImage
from reconstruction.primitive.splat import GSImage
from reconstruction.primitive.stereo import StereoImage, StereoGSImage
from reconstruction.vis.cam_orbit import CameraOrbit
from reconstruction.vis.frame_creator import FrameCreator, N4FrameCreator
from utils.misc import PathUtils, log


class DatasetVisualizer:
    def __init__(self, dataset: Dataset, camera_orbit: CameraOrbit, frame_creators: Union[FrameCreator, List[FrameCreator]], data_fps: int = 30):
        self.dataset = dataset
        self.camera_orbit = camera_orbit
        self.frame_creators = frame_creators if isinstance(frame_creators, list) else [frame_creators]
        self.data_fps = data_fps

    def run(self,
            recon_type: Literal['pcd', 'gs', 'stereo', 'stereo+gs'],
            output_dir: Union[Path, str],
            gs_regressor_model: Literal['gps'] = 'gps',
            gs_regressor_checkpoint: Union[Path, str] = None,
            disparity_estimator_model: Literal['raftstereo', 'foundationstereo'] = 'gps',
            disparity_estimator_checkpoint: Union[Path, str] = None,
            camera_orbit_velocity: float = 1.0,
            t_start: int = 0,
            t_total: Optional[int] = -1,
            split_stereo: bool = True,
            save_ply: bool = False,
            force: bool = False) -> None:
        """
        Visualize the dataset by creating frames for each sample and saving them to a video.

        Parameters
        ----------
        recon_type : Literal['pcd', 'gs', 'stereo', 'stereo+gs']
            The type of reconstruction to perform. 'pcd' for point cloud, 'gs' for Gaussian splatting, 'stereo' for point cloud with depth from stereo, and 'stereo+gs' for Gaussian splatting with depth from stereo.
        output_dir : Union[Path, str]
            The directory where the output video(s) and/or ply files will be saved. The video will be named based on the dataset and reconstruction type.
        gs_regressor_model : Literal['gps'], optional
            The Gaussian splatting regressor to use. Default is 'gps'.
        gs_regressor_checkpoint : Union[Path, str], optional
            The checkpoint path for the Gaussian splatting regressor. Default is None.
        disparity_estimator_model : Literal['raftstereo', 'foundationstereo'], optional
            The disparity estimator model to use. Default is 'gps'.
        disparity_estimator_checkpoint : Union[Path, str], optional
            The checkpoint path for the disparity estimator model. Default is None.
        camera_orbit_velocity : float, optional
            The velocity of the camera orbit in arc length meters / second. Default is 1.0.
        t_start : int, optional
            The starting index of the dataset to visualize. Default is 0.
        t_total : Optional[int], optional
            The total number of frames to visualize. If -1, it will visualize until the end of the dataset. Default is -1.
        split_stereo : bool, optional
            If True, the depth will be computed using stereo, but during rendering the images will be split into left and right and PCD will not be stitched. Default is False.
        save_ply : bool, optional
            If True, the point cloud for each frame and each will be saved as a PLY file (in `output_dir / cam{idx:02d} / <timestamp>.ply`). Default is False.
        force : bool, optional
            If True, the output video will be overwritten if it already exists. If False, it will skip writing if the file already exists. Default is False.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video_paths = [
            output_dir / (f'{frame_creator.main_mode_modality}_{frame_creator.satellite_mode_modality}_{frame_creator.satellite_mode_src.replace("gt", "")}'.rstrip('_') + f'_{t_start:04d}_{t_total:03d}.mp4')
            for frame_creator in self.frame_creators
        ]
        if all(p.exists() for p in output_video_paths) and not force:
            log(f'[{self.__class__.__name__}::run] Output video files already exists: {output_video_paths}. Skipping dataset visualization.', 'debug')
            return None

        camera_orbit_traversor = self.camera_orbit.traverse(velocity=camera_orbit_velocity, loop=False, data_fps=self.data_fps)
        for frame_creator in self.frame_creators:
            if hasattr(frame_creator, 'evaluator') and frame_creator.evaluator is not None:
                frame_creator.evaluator.reset()
        if t_total is None or t_total < 0:
            # noinspection PyTypeChecker
            t_total = len(self.dataset) - t_start
        video_writers = [None] * len(self.frame_creators)
        pbar = tqdm(zip(range(t_start, t_start + t_total), camera_orbit_traversor), desc=f"Visualizing dataset to {output_dir.parent.name}/{output_dir.name}", total=t_total)
        for t, (t_cam_intrinsic, t_cam_extrinsic_c2w, t_cam_gt_idx_closest, t_cam_gt_idx_middle) in pbar:
            for frame_creator in self.frame_creators:
                if hasattr(frame_creator, 'evaluator') and frame_creator.evaluator is not None:
                    mean_metrics = frame_creator.evaluator.gather()
                    break
            else:
                mean_metrics = {}
            pbar.set_postfix({'idx_closest': t_cam_gt_idx_closest, 'idx_middle': t_cam_gt_idx_middle, 't': t, **mean_metrics})
            t_images = self.sft_format_to_rgbd_images(self.dataset[t])
            if recon_type == 'gs':
                t_images = [GSImage.from_rgbd_image(_, gs_regressor_model=gs_regressor_model, gs_regressor_checkpoint=gs_regressor_checkpoint) for _ in t_images]
            elif 'stereo' in recon_type:
                has_gs = 'gs' in recon_type
                t_images = StereoImage.from_rgb_images(*t_images, return_last_as_rgbd=not has_gs, disparity_estimator_model=disparity_estimator_model, disparity_estimator_checkpoint=disparity_estimator_checkpoint)
                if has_gs:
                    # stereo + gs
                    stereo_gs_images = []
                    for stereo_image in t_images:
                        rgbd_image_l, rgbd_image_r = stereo_image.split_lr()
                        gs_image_l = GSImage.from_rgbd_image(rgbd_image_l, gs_regressor_model=gs_regressor_model, gs_regressor_checkpoint=gs_regressor_checkpoint)
                        gs_image_r = GSImage.from_rgbd_image(rgbd_image_r, gs_regressor_model=gs_regressor_model, gs_regressor_checkpoint=gs_regressor_checkpoint)
                        stereo_gs_image = StereoGSImage.from_gs_images(gs_image_l, gs_image_r)
                        stereo_gs_images.append(stereo_gs_image)
                    stereo_gs_images.append(t_images[-1])
                    t_images = stereo_gs_images
                if split_stereo:
                    # split stereo images (i.e. no stitching) by using the closest part of each LR pair to the current position in the orbit
                    t_images_split = []
                    all_sub_images = list(chain.from_iterable([_.split_lr() for _ in t_images[:-1]]))
                    t_images_split.append(all_sub_images[0])
                    for si in range(1, 2 * len(t_images[:-1]) - 1, 2):
                        image_l, image_r = all_sub_images[si], all_sub_images[si + 1]
                        t_images_split.append(image_r if (t_cam_gt_idx_middle + 1) > (si // 2 + 1) else image_l)
                    t_images_split.append(all_sub_images[-1])
                    t_images = t_images_split
                    t_cam_gt_idx_middle = t_cam_gt_idx_closest

            if save_ply:
                if 'stereo' not in recon_type:
                    for cam_idx, t_image_cam in zip(self.camera_orbit.gt_cam_idx_s0, t_images):
                        cam_dir = output_dir / f'cam{(cam_idx + 1):02d}'
                        cam_dir.mkdir(parents=True, exist_ok=True)
                        if (cam_dir / f'{t:06d}.ply').exists():
                            continue
                        t_image_cam.unproject().save_ply(cam_dir / f'{t:06d}.ply')
                elif not split_stereo:
                    for cam_idx, t_image_cam in zip(self.camera_orbit.gt_cam_idx_s0[:-1], t_images[:-1]):
                        cam_dir = output_dir / f'cam{(cam_idx + 1):02d}_{(cam_idx + 2):02d}'
                        cam_dir.mkdir(parents=True, exist_ok=True)
                        if (cam_dir / f'{t:06d}.ply').exists():
                            continue
                        t_image_cam.unproject().save_ply(cam_dir / f'{t:06d}.ply')
                else:
                    c = 0
                    for cam_idx in self.camera_orbit.gt_cam_idx_s0[:-1]:
                        for side in ['l', 'r']:
                            cam_dir = output_dir / f'cam{(cam_idx + 1):02d}_{(cam_idx + 2):02d}_{side}'
                            cam_dir.mkdir(parents=True, exist_ok=True)
                            if (cam_dir / f'{t:06d}.ply').exists():
                                continue
                            t_image_cam = all_sub_images[c]
                            t_image_cam.unproject().save_ply(cam_dir / f'{t:06d}.ply')
                            c += 1

            for fci, frame_creator in enumerate(self.frame_creators):
                video_frame = frame_creator.forward(
                    gt_images=t_images,
                    virtual_intrinsic=t_cam_intrinsic,
                    virtual_extrinsic=t_cam_extrinsic_c2w,
                    is_c2w=True,
                    assignment=[t_cam_gt_idx_closest] if 'stereo' not in recon_type else [t_cam_gt_idx_middle],
                    virtual_image_size_hw=self.camera_orbit.gt_image_size_hw,
                )

                if video_writers[fci] is None:
                    video_writers[fci] = cv2.VideoWriter(
                        str(output_video_paths[fci]),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        self.data_fps,  # Frame rate
                        (video_frame.shape[1], video_frame.shape[0])
                    )
                video_writers[fci].write(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))
        for frame_creator, video_writer in zip(self.frame_creators, video_writers):
            if video_writer is not None:
                video_writer.release()
            if hasattr(frame_creator, 'evaluator') and frame_creator.evaluator is not None:
                mean_metrics = frame_creator.evaluator.gather()
                print(mean_metrics)
        # clean offscreen renderers
        from reconstruction.primitive.pcd import PixelPoints
        for pyrender_renderer, pyrender_scene in PixelPoints.PYRENDER_RENDERER_CACHE.values():
            pyrender_scene.clear()
            if hasattr(pyrender_renderer, 'close'):
                pyrender_renderer.close()
            if hasattr(pyrender_renderer, 'delete'):
                pyrender_renderer.delete()
            else:
                del pyrender_renderer
            gc.collect()
        for o3d_renderer in PixelPoints.O3D_VISUALIZER_CACHE.values():
            if hasattr(o3d_renderer, 'scene') and hasattr(o3d_renderer.scene, 'clear_geometry'):
                o3d_renderer.scene.clear_geometry()
                del o3d_renderer
            gc.collect()
        return None

    @staticmethod
    def sft_format_to_rgbd_images(torch_data: List[torch.Tensor]) -> List[RGBDImage]:
        """
        Convert the data from the format used in the SFT to RGBDImage objects.

        Parameters
        ----------
        torch_data : List[torch.Tensor]
            A list of tensors containing the data in the SFT format:
            - torch_data[0]: RGB (N, 3, H, W), where N is the number of cameras
            - torch_data[1]: Mask (N, 1, H, W)
            - torch_data[1]: Depth (N, 1, H, W)
            - torch_data[2]: Intrinsics (N, 3, 3)
            - torch_data[3]: Extrinsics (N, 4, 4) in c2w format
            - torch_data[4]: Camera indices (N,)

        Returns
        -------
        List[RGBDImage]
            A list of RGBDImage objects created from the input tensors.
        """
        rgb, mask, depth, intrinsics, extrinsics, cam_indices = torch_data[:6]
        rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()
        mask = mask.squeeze(1).cpu().float().numpy() > 0.5  # Convert mask to boolean
        depth = depth.squeeze(1).cpu().numpy()
        intrinsics = intrinsics.cpu().numpy()
        extrinsics = extrinsics.cpu().numpy()
        cam_indices = cam_indices.cpu().numpy()
        rgbd_images = []
        for i in range(rgb.shape[0]):
            rgbd_images.append(
                RGBDImage(
                    rgb=np.ascontiguousarray((rgb[i] * 255.0 + 0.5).astype(np.uint8)),
                    mask=np.ascontiguousarray(mask[i]),
                    depth=np.ascontiguousarray(depth[i]),
                    intrinsic=np.ascontiguousarray(intrinsics[i]),
                    extrinsic_w2c=np.ascontiguousarray(np.linalg.inv(extrinsics[i])),
                )
            )
        return [rgbd_image for _, rgbd_image in sorted(zip(cam_indices.tolist(), rgbd_images), key=lambda x: x[0])]


if __name__ == "__main__":
    from reconstruction.data.capturestudio import MultiSessionDataset
    from reconstruction.eval.metrics import MetricsAggregator
    from reconstruction.vis.teaser import create_teaser_video, create_teaser_grid

    # --------------------------------------------------------------------------------------------------------
    session_name = 'Simone_2_Perf_3'
    calibration_session_name_ = 'Thanos_2_Calib_1'
    output_dir_ = PathUtils().out_path() / 'evaluation' / session_name
    output_dir_.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------------------------------------------------------
    image_size_hw_ = (1024, 1024)
    cam_idx_ = [4, 5, 7, 8, 9]
    start_frame_  = 1200
    total_frames_ = 300
    gs_regressor_model_ = 'gps'  # only 'gps' is supported for now
    # gps_regressor_ckpt_ = 'gpsgaussian/RGBD2GS_Regression_latest_0708_psnr_38_91.pth'  # 'neptune://85/best'  # 'gpsgaussian/gps_21_stage2_final.pth', 'neptune://85/best'
    # gps_regressor_ckpt_ = 'neptune://154/best'  # 'neptune://85/best'  # 'gpsgaussian/gps_21_stage2_final.pth', 'neptune://85/best'
    disparity_estimator_model_ = 'raftstereo'  # 'raftstereo' or 'foundationstereo'
    # disparity_estimator_ckpt_ = gps_regressor_ckpt_ if disparity_estimator_model_ == 'raftstereo' else 'vitl'  # for foundation stereo: 'vits', 'vitl'  |  for gps: <same as gps_regressor_ckpt_>
    frame_creators_ = [
        FrameCreator.from_num_cameras(len(cam_idx_), mode=f'{vis_data}_recon|{vis_data}_gt', main_size_hw=image_size_hw_, evaluator=MetricsAggregator.for_psnr())
        for vis_data in ['rgb', 'depth', 'feat']
    ]
    # --------------------------------------------------------------------------------------------------------

    def create_teasers_and_grid(dirs: List[Path]) -> None:
        for d in dirs:
            create_teaser_video({
                "paths": {
                    "rgb": str(d / f"rgb_rgb_{start_frame_:04d}_{total_frames_:03d}.mp4"),
                    "depth": str(d / f"depth_depth_{start_frame_:04d}_{total_frames_:03d}.mp4"),
                    "norm": str(d / f"feat_feat_{start_frame_:04d}_{total_frames_:03d}.mp4"),
                },
                "output": str(d / f"teaser_{start_frame_:04d}_{total_frames_:03d}.mp4"),
                "fps": 30,
                "target_height": 720,
                "bitrate": "15M",
                # timeline (seconds) ─ names are free; order matters
                "durations": {
                    "rgb_only": 1.5,  # A
                    "rgb_to_depth": 1.0,  # B  (transition)
                    "depth_hold": 1.0,  # NEW  ← depth full-screen pause
                    "depth_to_norm": 1.5,  # C  (transition)
                    "reveal_bands": 1.5,  # D
                    "hold": 1.5,  # E
                    "collapse": 1.0,  # F
                },
                # diagonal proportions (must sum ≤ 1.0; remaining goes to RGB corners)
                "bands": {
                    "rgb_corner": 0.35,  # each corner
                    "normal": 0.15,
                    "depth": 0.15,
                },
            })
        create_teaser_grid({
            # Six input videos, row-major order: [row0-col0, row0-col1, …]
            "paths": [str(d / f'rgb_rgb_{start_frame_:04d}_{total_frames_:03d}.mp4') for d in dirs],

            # Short labels that will be burned into the clips (same order)
            "labels": [d.name.upper().replace("_", " ") for d in dirs],

            # Font & styling for overlay text
            "font": "JetBrainsMono-Regular.ttf",
            "font_size": 22,
            "font_color": "white",
            "stroke_color": "black",
            "stroke_width": 0,

            # Output file & encoding
            "output": str(dirs[0].parent / f'teaser_grid_{start_frame_:04d}_{total_frames_:03d}.mp4'),
            "fps": 30,
            "bitrate": "15M",
        })

    # ================================
    # ====  PCD vs GS Evaluation  ====
    # ================================
    # [start] ---------------------------------------------------
    output_dir_this_ = output_dir_ / 'pcd_vs_gs'
    output_dirs_this_ = []
    gs_regressor_ckpt_this_ = 'neptune://154/best'
    for depth_filter_ in ['bilateral_temporal', 'stereo|raftstereo']:
        # noinspection PyTypeChecker
        dataset_ = MultiSessionDataset(
            calibration_session_name=calibration_session_name_,
            calibration_method='MultiCamCalib',
            session_names=[session_name],
            cam_indices=cam_idx_,
            n_cams_per_sample=-1,
            use_stereo=False,
            depth_filter=depth_filter_ if not depth_filter_.startswith('stereo') else None,
            target_image_size_hw=image_size_hw_,
            apply_intrinsics_fix=False,  # not ('best' in Path(gps_regressor_ckpt_).stem and int(re.split(r"[-/]", str(gps_regressor_ckpt_))[-2]) > 21),
        )
        camera_orbit_ = dataset_.get_camera_orbit(bezier_degree=3)
        dataset_visualizer_ = DatasetVisualizer(dataset=dataset_, camera_orbit=camera_orbit_, frame_creators=frame_creators_)

        # noinspection PyTypeChecker
        for recon_type_ in ['pcd', 'gs']:
            if depth_filter_.startswith('stereo'):
                recon_type_ = f'stereo{"+gs" if recon_type_ == "gs" else ""}'
                disparity_estimator_model_ = depth_filter_.split('|')[-1]
                assert disparity_estimator_model_ in ['raftstereo', 'foundationstereo']

            # noinspection PyTypeChecker
            output_dir_this_i_ = output_dir_this_ / f'{"gs" if "gs" in recon_type_ else "pcd"}_{depth_filter_.split("|")[-1]}'
            dataset_visualizer_.run(
                recon_type=recon_type_,
                output_dir=output_dir_this_i_,
                t_start=start_frame_,
                t_total=total_frames_,
                disparity_estimator_model=disparity_estimator_model_,
                disparity_estimator_checkpoint=gs_regressor_ckpt_this_ if disparity_estimator_model_ == 'raftstereo' else 'vitl',
                gs_regressor_model=gs_regressor_model_,
                gs_regressor_checkpoint=gs_regressor_ckpt_this_,
                camera_orbit_velocity=0.5,  # m/s (arc length)
                split_stereo=True,
            )
            output_dirs_this_.append(output_dir_this_i_)
    # create teaser video and grid
    create_teasers_and_grid(output_dirs_this_)
    # ----------------------------------------------------- [end]
    # --------------------------------------------------------------------------------------------------------

    # ===============================================================
    # ====  Vanilla GPS evaluation: depth only from raft stereo  ====
    # ===============================================================
    disparity_estimator_model_ = 'raftstereo'
    # ------------------------------------------------------------
    #   i) b/a camera change: Vanilla vs GPS-21 (authors vs ours setup)
    # -----------------------------------------------------------
    # [start] ---------------------------------------------------
    dataset_ = MultiSessionDataset(
        calibration_session_name=calibration_session_name_,
        calibration_method='MultiCamCalib',
        session_names=[session_name],
        cam_indices=cam_idx_,
        n_cams_per_sample=-1,
        use_stereo=False,
        depth_filter=None,
        target_image_size_hw=image_size_hw_,
        apply_intrinsics_fix=True,
    )
    camera_orbit_ = dataset_.get_camera_orbit(bezier_degree=3)
    dataset_visualizer_ = DatasetVisualizer(dataset=dataset_, camera_orbit=camera_orbit_, frame_creators=frame_creators_)
    output_dir_this_ = output_dir_ / 'vanilla_gps_evaluation' / 'cam_authors_vs_ours'
    output_dirs_this_ = []
    for gs_regressor_ckpt_this_, label_ in [('gpsgaussian/gps_17cams_authors_stage2_final.pth', 'authors'), ('gpsgaussian/gps_21_stage2_final.pth', 'ours')]:
        output_dir_this_i_ = output_dir_this_ / label_
        dataset_visualizer_.run(
            recon_type='stereo+gs',
            output_dir=output_dir_this_i_,
            t_start=start_frame_,
            t_total=total_frames_,
            disparity_estimator_model=disparity_estimator_model_,
            disparity_estimator_checkpoint=gs_regressor_ckpt_this_ if disparity_estimator_model_ == 'raftstereo' else 'vitl',
            gs_regressor_model=gs_regressor_model_,
            gs_regressor_checkpoint=gs_regressor_ckpt_this_,
            camera_orbit_velocity=0.5,  # m/s (arc length)
            split_stereo=True,
        )
        output_dirs_this_.append(output_dir_this_i_)
    # create teaser videos and grid
    create_teasers_and_grid(output_dirs_this_)
    # ----------------------------------------------------- [end]

    # ------------------------------------------------------------
    #   ii) b/a intrinsics/extrinsics bug fixes: Vanilla vs Simone (authors setup)
    # -----------------------------------------------------------
    # [start] ---------------------------------------------------
    dataset_rs_ = MultiSessionDataset(
        calibration_session_name=calibration_session_name_,
        calibration_method='MultiCamCalib',
        session_names=[session_name],
        cam_indices=cam_idx_,
        n_cams_per_sample=-1,
        use_stereo=False,
        depth_filter=None,
        target_image_size_hw=image_size_hw_,
        apply_intrinsics_fix=True,
    )
    dataset_bft_ = MultiSessionDataset(
        calibration_session_name=calibration_session_name_,
        calibration_method='MultiCamCalib',
        session_names=[session_name],
        cam_indices=cam_idx_,
        n_cams_per_sample=-1,
        use_stereo=False,
        depth_filter='bilateral_temporal',
        target_image_size_hw=image_size_hw_,
        apply_intrinsics_fix=True,
    )
    camera_orbit_ = dataset_bft_.get_camera_orbit(bezier_degree=3)
    dataset_visualizer_rs_ = DatasetVisualizer(dataset=dataset_rs_, camera_orbit=camera_orbit_, frame_creators=frame_creators_)
    dataset_visualizer_bft_ = DatasetVisualizer(dataset=dataset_bft_, camera_orbit=camera_orbit_, frame_creators=frame_creators_)
    output_dir_this_ = output_dir_ / 'vanilla_gps_evaluation' / 'vanilla_vs_simone'
    output_dirs_this_ = []
    for gs_regressor_ckpt_this_, label_ in [('gpsgaussian/gps_17cams_authors_stage2_final.pth', 'authors'), ('gpsgaussian/RGBD2GS_Regression_final_bug.pth', 'simone_bug'), ('gpsgaussian/RGBD2GS_Regression_final_thanos.pth', 'simone_thanos'), ('gpsgaussian/RGBD2GS_Regression_latest_0708_psnr_38_91.pth', 'simone_simone')]:
        output_dir_this_i_ = output_dir_this_ / label_
        (dataset_visualizer_rs_ if label_ == 'authors' else dataset_visualizer_bft_).run(
            recon_type='stereo+gs' if label_ == 'authors' else 'gs',
            output_dir=output_dir_this_i_,
            t_start=start_frame_,
            t_total=total_frames_,
            disparity_estimator_model=disparity_estimator_model_,
            disparity_estimator_checkpoint=gs_regressor_ckpt_this_ if disparity_estimator_model_ == 'raftstereo' else 'vitl',
            gs_regressor_model=gs_regressor_model_,
            gs_regressor_checkpoint=gs_regressor_ckpt_this_,
            camera_orbit_velocity=0.5,  # m/s (arc length)
            split_stereo=True,
        )
        output_dirs_this_.append(output_dir_this_i_)
    # create teaser videos and grid
    create_teasers_and_grid(output_dirs_this_)
    # ----------------------------------------------------- [end]

    # ------------------------------------------------------------
    #   iii) b/a SFT: GPS-21 vs GPS-85 vs. GPS-154  (our setup)
    # -----------------------------------------------------------
    # [start] ---------------------------------------------------
    dataset_rs_ = MultiSessionDataset(
        calibration_session_name=calibration_session_name_,
        calibration_method='MultiCamCalib',
        session_names=[session_name],
        cam_indices=cam_idx_,
        n_cams_per_sample=-1,
        use_stereo=False,
        depth_filter=None,
        target_image_size_hw=image_size_hw_,
        apply_intrinsics_fix=True,
    )
    dataset_bft_ = MultiSessionDataset(
        calibration_session_name=calibration_session_name_,
        calibration_method='MultiCamCalib',
        session_names=[session_name],
        cam_indices=cam_idx_,
        n_cams_per_sample=-1,
        use_stereo=False,
        depth_filter='bilateral_temporal',
        target_image_size_hw=image_size_hw_,
        apply_intrinsics_fix=False,
    )
    camera_orbit_ = dataset_bft_.get_camera_orbit(bezier_degree=3)
    dataset_visualizer_rs_ = DatasetVisualizer(dataset=dataset_rs_, camera_orbit=camera_orbit_, frame_creators=frame_creators_)
    dataset_visualizer_bft_ = DatasetVisualizer(dataset=dataset_bft_, camera_orbit=camera_orbit_, frame_creators=frame_creators_)
    output_dir_this_ = output_dir_ / 'vanilla_gps_evaluation' / 'sft'
    output_dirs_this_ = []
    for gs_regressor_ckpt_this_, label_ in [('gpsgaussian/gps_21_stage2_final.pth', 'ours'), ('gpsgaussian/RGBD2GS_Regression_latest_0708_psnr_38_91.pth', 'simone_simone'), ('gpsgaussian/gps-85-best.pth', 'ours+sft'), ('gpsgaussian/gps-154-best.pth', 'ours+sft+instancenorm')]:
        output_dir_this_i_ = output_dir_this_ / label_
        (dataset_visualizer_rs_ if '+' not in label_ else dataset_visualizer_bft_).run(
            recon_type='stereo+gs' if '+' not in label_ else 'gs',
            output_dir=output_dir_this_i_,
            t_start=start_frame_,
            t_total=total_frames_,
            disparity_estimator_model=disparity_estimator_model_,
            disparity_estimator_checkpoint=gs_regressor_ckpt_this_ if disparity_estimator_model_ == 'raftstereo' else 'vitl',
            gs_regressor_model=gs_regressor_model_,
            gs_regressor_checkpoint=gs_regressor_ckpt_this_,
            camera_orbit_velocity=0.5,  # m/s (arc length)
            split_stereo=True,
        )
        output_dirs_this_.append(output_dir_this_i_)
    # create teaser videos and grid
    create_teasers_and_grid(output_dirs_this_)

    # ----------------------------------------------------- [end]

    #
    # # - GPS-21: Intrinsics Fix / RaftStereo
    # gps_regressor_ckpt_this_ = 'gpsgaussian/gps_21_stage2_final.pth'
    # disparity_estimator_model_ = 'raftstereo'
    # disparity_estimator_checkpoint_ = gps_regressor_ckpt_
    # output_dir_ = PathUtils().out_path() / 'evaluation' / 'gps' / 'B_A SFT_ GPS-21 vs GPS-85 _our setup_'
    # output_dir_.mkdir(parents=True, exist_ok=True)
    # dataset_ = MultiSessionDataset(
    #     calibration_session_name='Thanos_2_Calib_1',
    #     calibration_method='MultiCamCalib',
    #     session_names=['Thanos_2_Perf_1'],
    #     cam_indices=[4, 5, 7, 8],
    #     n_cams_per_sample=-1,
    #     use_stereo=False,
    #     depth_filter=None,
    #     target_image_size_hw=image_size_hw_,
    #     apply_intrinsics_fix=True,
    # )
    # dataset_visualizer_ = DatasetVisualizer(
    #     dataset=dataset_,
    #     camera_orbit=dataset_.get_camera_orbit(bezier_degree=3),
    #     frame_creators=N4FrameCreator(mode=f'{vis_data}_recon|{vis_data}_gt', main_size_hw=image_size_hw_, evaluator=MetricsAggregator.for_psnr()))
    # dataset_visualizer_.run(
    #     t_start=1200,
    #     t_total=300,
    #     recon_type='stereo+gs',
    #     output_dir=output_dir_,
    #     disparity_estimator_model=disparity_estimator_model_,
    #     disparity_estimator_checkpoint=disparity_estimator_checkpoint_,
    #     gs_regressor_model=gs_regressor_model_,
    #     gs_regressor_checkpoint=gps_regressor_ckpt_,
    #     camera_orbit_velocity=0.3,  # m/s (arc length)
    #     split_stereo=True,
    # )
    # # - GPS-85-RS: Intrinsics Fix / RaftStereo
    # gs_regressor_model_ = 'gps'
    # gps_regressor_ckpt_ = 'neptune://85/best'
    # disparity_estimator_model_ = 'raftstereo'
    # disparity_estimator_checkpoint_ = gps_regressor_ckpt_
    # output_video_path_ = PathUtils().out_path() / 'results' / 'Thanos_2_Perf_1' / 'vanilla_gps_evaluation' / f'gps_85_rs.mp4'
    # output_video_path_.parent.mkdir(parents=True, exist_ok=True)
    # dataset_ = MultiSessionDataset(
    #     calibration_session_name='Thanos_2_Calib_1',
    #     calibration_method='MultiCamCalib',
    #     session_names=['Thanos_2_Perf_1'],
    #     cam_indices=[4, 5, 7, 8],
    #     n_cams_per_sample=-1,
    #     use_stereo=False,
    #     depth_filter=None,
    #     target_image_size_hw=image_size_hw_,
    #     apply_intrinsics_fix=True,
    # )
    # for vis_data in ['rgb', 'depth', 'feat']:
    #     dataset_visualizer_ = DatasetVisualizer(dataset=dataset_, camera_orbit=dataset_.get_camera_orbit(bezier_degree=3), frame_creators=N4FrameCreator(mode=f'{vis_data}_recon|{vis_data}_gt', main_size_hw=image_size_hw_, evaluator=MetricsAggregator.for_psnr()))
    #     dataset_visualizer_.run(
    #         t_start=1200,
    #         t_total=300,
    #         recon_type='stereo+gs',
    #         output_video_path=output_video_path_,
    #         disparity_estimator_model=disparity_estimator_model_,
    #         disparity_estimator_checkpoint=disparity_estimator_checkpoint_,
    #         gs_regressor_model=gs_regressor_model_,
    #         gs_regressor_checkpoint=gps_regressor_ckpt_,
    #         camera_orbit_velocity=0.3,  # m/s (arc length)
    #         split_stereo=True,
    #     )
    # # - GPS-85: Depth BFS
    # gs_regressor_model_ = 'gps'
    # gps_regressor_ckpt_ = 'neptune://85/best'
    # output_video_path_ = PathUtils().out_path() / 'results' / 'Thanos_2_Perf_1' / 'vanilla_gps_evaluation' / f'gps_85.mp4'
    # output_video_path_.parent.mkdir(parents=True, exist_ok=True)
    # dataset_ = MultiSessionDataset(
    #     calibration_session_name='Thanos_2_Calib_1',
    #     calibration_method='MultiCamCalib',
    #     session_names=['Thanos_2_Perf_1'],
    #     cam_indices=[4, 5, 7, 8],
    #     n_cams_per_sample=-1,
    #     use_stereo=False,
    #     depth_filter='bilateral_spatial',
    #     target_image_size_hw=image_size_hw_,
    #     apply_intrinsics_fix=False,
    # )
    # for vis_data in ['rgb', 'depth', 'feat']:
    #     dataset_visualizer_ = DatasetVisualizer(dataset=dataset_, camera_orbit=dataset_.get_camera_orbit(bezier_degree=3), frame_creators=N4FrameCreator(mode=f'{vis_data}_recon|{vis_data}_gt', main_size_hw=image_size_hw_, evaluator=MetricsAggregator.for_psnr()))
    #     dataset_visualizer_.run(
    #         t_start=1200,
    #         t_total=300,
    #         recon_type='gs',
    #         output_video_path=output_video_path_,
    #         gs_regressor_model=gs_regressor_model_,
    #         gs_regressor_checkpoint=gps_regressor_ckpt_,
    #         camera_orbit_velocity=0.3,  # m/s (arc length)
    #         split_stereo=True,
    #     )
    #
    # # ------------------------------------------------------------
    # #   iii) b/a SFT: GPS-21 vs GPS-154 (our setup; best SFT: rot bug fix v2, scales tanh + learnable per axis affine)
    # # -----------------------------------------------------------
    # # - GPS-154-RS: Intrinsics Fix / RaftStereo
    # gs_regressor_model_ = 'gps'
    # gps_regressor_ckpt_ = 'neptune://154/best'
    # disparity_estimator_model_ = 'raftstereo'
    # disparity_estimator_checkpoint_ = gps_regressor_ckpt_
    # output_video_path_ = PathUtils().out_path() / 'results' / 'Thanos_2_Perf_1' / 'vanilla_gps_evaluation' / f'gps_154_rs.mp4'
    # output_video_path_.parent.mkdir(parents=True, exist_ok=True)
    # dataset_ = MultiSessionDataset(
    #     calibration_session_name='Thanos_2_Calib_1',
    #     calibration_method='MultiCamCalib',
    #     session_names=['Thanos_2_Perf_1'],
    #     cam_indices=[4, 5, 7, 8],
    #     n_cams_per_sample=-1,
    #     use_stereo=False,
    #     depth_filter=None,
    #     target_image_size_hw=image_size_hw_,
    #     apply_intrinsics_fix=True,
    # )
    # for vis_data in ['rgb', 'depth', 'feat']:
    #     dataset_visualizer_ = DatasetVisualizer(dataset=dataset_, camera_orbit=dataset_.get_camera_orbit(bezier_degree=3), frame_creators=N4FrameCreator(mode=f'{vis_data}_recon|{vis_data}_gt', main_size_hw=image_size_hw_, evaluator=MetricsAggregator.for_psnr()))
    #     dataset_visualizer_.run(
    #         t_start=1200,
    #         t_total=300,
    #         recon_type='stereo+gs',
    #         output_video_path=output_video_path_,
    #         disparity_estimator_model=disparity_estimator_model_,
    #         disparity_estimator_checkpoint=disparity_estimator_checkpoint_,
    #         gs_regressor_model=gs_regressor_model_,
    #         gs_regressor_checkpoint=gps_regressor_ckpt_,
    #         camera_orbit_velocity=0.3,  # m/s (arc length)
    #         split_stereo=True,
    #     )
    # # - GPS-154: Depth BFS
    # gs_regressor_model_ = 'gps'
    # gps_regressor_ckpt_ = 'neptune://154/best'
    # output_video_path_ = PathUtils().out_path() / 'results' / 'Thanos_2_Perf_1' / 'vanilla_gps_evaluation' / f'gps_154.mp4'
    # output_video_path_.parent.mkdir(parents=True, exist_ok=True)
    # dataset_ = MultiSessionDataset(
    #     calibration_session_name='Thanos_2_Calib_1',
    #     calibration_method='MultiCamCalib',
    #     session_names=['Thanos_2_Perf_1'],
    #     cam_indices=[4, 5, 7, 8],
    #     n_cams_per_sample=-1,
    #     use_stereo=False,
    #     depth_filter='bilateral_spatial',
    #     target_image_size_hw=image_size_hw_,
    #     apply_intrinsics_fix=False,
    # )
    # for vis_data in ['rgb', 'depth', 'feat']:
    #     dataset_visualizer_ = DatasetVisualizer(dataset=dataset_, camera_orbit=dataset_.get_camera_orbit(bezier_degree=3), frame_creators=N4FrameCreator(mode=f'{vis_data}_recon|{vis_data}_gt', main_size_hw=image_size_hw_, evaluator=MetricsAggregator.for_psnr()))
    #     dataset_visualizer_.run(
    #         t_start=1200,
    #         t_total=300,
    #         recon_type='gs',
    #         output_video_path=output_video_path_,
    #         gs_regressor_model=gs_regressor_model_,
    #         gs_regressor_checkpoint=gps_regressor_ckpt_,
    #         camera_orbit_velocity=0.3,  # m/s (arc length)
    #         split_stereo=True,
    #     )
    #
    # # dataset_visualizer_.run(
    # #     t_start=1200,
    # #     t_total=30,
    # #     recon_type='stereo',
    # #     output_video_path=output_video_path_,
    # #     disparity_estimator_model=disparity_estimator_model_,
    # #     camera_orbit_velocity=2.3,  # m/s (arc length)
    # #     split_stereo=False,
    # # )
    # # dataset_visualizer_.run(
    # #     t_start=1200,
    # #     t_total=30,
    # #     recon_type='gs',
    # #     output_video_path=output_video_path_,
    # #     disparity_estimator_model=disparity_estimator_model_,
    # #     disparity_estimator_checkpoint=disparity_estimator_ckpt_,
    # #     gs_regressor_model=gs_regressor_model_,
    # #     gs_regressor_checkpoint=gps_regressor_ckpt_,
    # #     camera_orbit_velocity=2.3,  # m/s (arc length)
    # #     split_stereo=False,
    # # )
