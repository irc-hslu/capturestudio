import json
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Literal, List, Tuple, Optional

import numpy as np
import pytorch3d.transforms.rotation_conversions
import torch
from scipy.interpolate import CubicSpline
from transformations import quaternion_slerp

from tasks import app, AutoRetryTask

manifest_file_lock = multiprocessing.Lock()


def _reconstruction(
        capturestudio_cache_root: str,
        calibration_session_name: str,
        excel_data: dict,
        recon_type: Literal['pcd', 'gs'],
        start_frame: int,
        total_frames: int,
        cam_idx: List[int] = [4, 5, 7, 8, 9],
        depth_source: Literal['bilateral_spatial', 'bilateral_temporal', 'stereo'] = 'bilateral_temporal',
        image_size_hw: Tuple[int, int] = (1024, 1280),
        disparity_estimator_model: str = 'raftstereo',
        disparity_estimator_checkpoint: str = 'neptune://154/best',
        gs_regressor_model: str = 'gps',
        gs_regressor_checkpoint: str = 'neptune://154/best',
        orbit_type: Literal['audience', 'interpolated'] = 'interpolated',
        force: bool = False,
        calibration_method: Literal['MultiCamCalib', 'Caliscope'] = 'Caliscope',
        rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None,
        camera_orbit_velocity: float = 0.3,
        save_ply: bool = False):
    from reconstruction.data.capturestudio import MultiSessionDataset
    from reconstruction.eval.metrics import MetricsAggregator
    from reconstruction.vis.dataset_visualizer import DatasetVisualizer
    from reconstruction.vis.frame_creator import FrameCreator

    capturestudio_cache_root = Path(capturestudio_cache_root)
    output_dir = capturestudio_cache_root / f'reconstruction'
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_ = MultiSessionDataset(
        calibration_session_name=calibration_session_name,
        calibration_method=calibration_method,
        session_names=[capturestudio_cache_root.name],
        cam_indices=cam_idx,
        n_cams_per_sample=-1,
        use_stereo=False,
        depth_filter=None if depth_source.startswith('stereo') else depth_source,
        target_image_size_hw=image_size_hw,
        apply_intrinsics_fix=depth_source.startswith('stereo'),
    )
    dataset_visualizer_ = DatasetVisualizer(
        dataset=dataset_,
        camera_orbit=dataset_.get_camera_orbit(
            bezier_degree=3,
            orbit_type=orbit_type
        ),
        frame_creators=[
            FrameCreator.from_num_cameras(
                num_cameras=len(cam_idx),
                mode=f'{vis_modality}_recon|{vis_modality}_gt',
                main_size_hw=image_size_hw,
                evaluator=MetricsAggregator.for_psnr(),
                rotate=rotate
            )
            for vis_modality in ['rgb', 'depth', 'feat']
        ]
    )
    if recon_type == 'gs' and depth_source.startswith('stereo'):
        recon_type = 'stereo+gs'
    elif depth_source.startswith('stereo'):
        recon_type = 'stereo'
    dataset_visualizer_.run(
        t_start=start_frame,
        t_total=total_frames,
        recon_type=recon_type,
        output_dir=output_dir / f'{"gs" if "gs" in recon_type else "pcd"}_{disparity_estimator_model if depth_source == "stereo" else depth_source}',
        disparity_estimator_model=disparity_estimator_model,
        disparity_estimator_checkpoint=disparity_estimator_checkpoint,
        gs_regressor_model=gs_regressor_model,
        gs_regressor_checkpoint=gs_regressor_checkpoint,
        camera_orbit_velocity=camera_orbit_velocity,  # m/s (arc length)
        split_stereo=True,
        save_ply=save_ply,
        force=force
    )

    # Create manifest file
    manifest_file = output_dir / 'manifest.json'
    with manifest_file_lock:
        if not manifest_file.exists() or manifest_file.stat().st_size == 0:
            cameras_info_single = [
                {
                    "id": (d.cam_names[0] if isinstance(d.cam_names, list) else d.cam_names).split('/')[-1],
                    "index": (d.cam_names[0] if isinstance(d.cam_names, list) else d.cam_names).split('/')[-1].replace('cam', ''),
                    "position": d.extrinsics_c2w[..., :3, 3].squeeze().tolist(),
                    "rotation": pytorch3d.transforms.rotation_conversions.matrix_to_quaternion(d.extrinsics_c2w[..., :3, :3].squeeze()[None])[0].tolist(),
                    "intrinsics": {
                        "fx": d.intrinsics.squeeze()[0, 0].item(),
                        "fy": d.intrinsics.squeeze()[1, 1].item(),
                        "cx": d.intrinsics.squeeze()[0, 2].item(),
                        "cy": d.intrinsics.squeeze()[1, 2].item(),
                        "w": int(d.image_size.squeeze()[0]),
                        "h": int(d.image_size.squeeze()[1]),
                    }
                }
                for d in dataset_.calibration_data
            ]
            # Fit a polynomial to the camera's positions and do slerp for rotations
            #  - stereo: for each successive camera pair, add to the `cameras_info_stereo` list a camera in the middle (along the polynomial fit) of the two cameras' positions and with a mean rotation (according to slerp)
            #  - stereo_split: for each successive camera pair, compute two other cameras at 1/4 and 3/4 of the distance between the two cameras' positions, and with a corresponding mix of the rotations
            t = np.arange(len(cameras_info_single))
            cs = CubicSpline(t, np.array([c["position"] for c in cameras_info_single]), axis=0)
            Q = torch.tensor([c["rotation"] for c in cameras_info_single], dtype=torch.float32)
            cameras_info_stereo, cameras_info_stereo_split = [], []
            for i, (cam_l, cam_r) in enumerate(zip(cameras_info_single[:-1], cameras_info_single[1:])):
                def make(alpha, tag):
                    return {
                        "id": f"cam{cam_l['index']}_{cam_r['index']}{tag}",
                        "index": f"{cam_l['index']}_{cam_r['index']}{tag}",
                        "position": cs(i + alpha).tolist(),
                        "rotation": quaternion_slerp(Q[i], Q[i + 1], torch.tensor(alpha)).tolist(),
                        "intrinsics": cam_l["intrinsics"].copy(),
                    }

                cameras_info_stereo.append(make(0.5, ""))
                cameras_info_stereo_split += [make(0.25, "_l"), make(0.75, "_r")]

            with manifest_file.open('w') as f:
                json.dump({
                    "sessionId": capturestudio_cache_root.name,
                    "title": capturestudio_cache_root.name.replace("_", " ").title(),
                    "performers": [_.strip() for _ in excel_data['Subject'].split(',')],
                    "location": "Interaction Space, IRC HSLU",
                    "date": datetime.strptime(excel_data['Session ID'].split('@')[1], '%d_%m_%Y_%H_%M_%S').strftime('%Y-%m-%d'),
                    "time": datetime.strptime(excel_data['Session ID'].split('@')[1], '%d_%m_%Y_%H_%M_%S').strftime('%H:%M'),
                    "fps": int(excel_data['Color FPS']),
                    "cameras": cameras_info_single + cameras_info_stereo + cameras_info_stereo_split,
                    "reconstructions": [],
                    "pointBudget": 300_000
                }, f)
        depth_source_info = {
            "bilateral_spatial": {"label": "camera depth (S)", "dir": "pcd_bilateral_spatial"},
            "bilateral_temporal": {"label": "camera depth", "dir": "pcd_bilateral_temporal"},
            "raftstereo": {"label": "depth from stereo", "dir": "pcd_raftstereo"},
            "foundationstereo": {"label": "depth from stereo (FS)", "dir": "pcd_foundationstereo"}
        }
        with manifest_file.open('r') as f:
            manifest = json.load(f)
        manifest_reconstructions = manifest['reconstructions']
        if depth_source == 'stereo':
            depth_source = disparity_estimator_model
        recon_type = 'gs' if 'gs' in recon_type else 'pcd'
        for mr in manifest_reconstructions:
            if mr['frameStart'] == start_frame and mr['totalFrames'] == total_frames:
                if depth_source not in mr['depthSources']:
                    # Add new depth source to existing reconstruction entry
                    mr['depthSources'][depth_source] = depth_source_info[depth_source]
                if recon_type not in mr['recon_types']:
                    # Add new reconstruction type to existing entry
                    mr['recon_types'].append(recon_type)
                break
        else:
            # Create a new reconstruction entry
            manifest_reconstructions.append({
                "frameStart": start_frame,
                "totalFrames": total_frames,
                "depthSources": {depth_source: depth_source_info[depth_source]},
                "recon_types": [recon_type],
                "defaultDepthSource": depth_source,
            })
        manifest['reconstructions'] = manifest_reconstructions
        with manifest_file.open('w') as f:
            json.dump(manifest, f, indent=4)
    return None


@app.task(name="reconstruction.pcd_reconstruction", base=AutoRetryTask)
def pcd_reconstruction(capturestudio_cache_root: str,
                       calibration_session_name: str,
                       excel_data: dict,
                       start_frame: int,
                       total_frames: int,
                       cam_idx: List[int] = [4, 5, 7, 8],
                       depth_source: Literal['bilateral_spatial', 'bilateral_temporal', 'stereo'] = 'bilateral_temporal',
                       image_size_hw: Tuple[int, int] = (1024, 1024),
                       disparity_estimator_model: str = 'raftstereo',
                       disparity_estimator_checkpoint: str = 'neptune://154/best',
                        orbit_type: Literal['audience', 'interpolated'] = 'interpolated',
                       force: bool = False,
                       calibration_method: Literal['MultiCamCalib', 'Caliscope'] = 'MultiCamCalib',
                       rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None,
                       camera_orbit_velocity: float = 0.3,
                       save_ply: bool = False):
    return _reconstruction(
        capturestudio_cache_root=capturestudio_cache_root,
        calibration_session_name=calibration_session_name,
        excel_data=excel_data,
        recon_type='pcd',
        start_frame=start_frame,
        total_frames=total_frames,
        cam_idx=cam_idx,
        depth_source=depth_source,
        image_size_hw=image_size_hw,
        disparity_estimator_model=disparity_estimator_model,
        disparity_estimator_checkpoint=disparity_estimator_checkpoint,
        orbit_type=orbit_type,
        force=force,
        calibration_method=calibration_method,
        rotate=rotate,
        camera_orbit_velocity=camera_orbit_velocity,
        save_ply=save_ply
    )


@app.task(name="reconstruction.gs_reconstruction", base=AutoRetryTask)
def gs_reconstruction(capturestudio_cache_root: str,
                      calibration_session_name: str,
                      excel_data: dict,
                      start_frame: int,
                      total_frames: int,
                      cam_idx: List[int] = [4, 5, 7, 8],
                      depth_source: Literal['bilateral_spatial', 'bilateral_temporal', 'stereo'] = 'bilateral_temporal',
                      image_size_hw: Tuple[int, int] = (1024, 1024),
                      disparity_estimator_model: str = 'raftstereo',
                      disparity_estimator_checkpoint: str = 'neptune://154/best',
                      gs_regressor_model: str = 'gps',
                      gs_regressor_checkpoint: str = 'neptune://154/best',
                      orbit_type: Literal['audience', 'interpolated'] = 'interpolated',
                      force: bool = False,
                      calibration_method: Literal['MultiCamCalib', 'Caliscope'] = 'MultiCamCalib',
                      rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None,
                      camera_orbit_velocity: float = 0.3,
                      save_ply: bool = False):
    return _reconstruction(
        capturestudio_cache_root=capturestudio_cache_root,
        calibration_session_name=calibration_session_name,
        excel_data=excel_data,
        recon_type='gs',
        start_frame=start_frame,
        total_frames=total_frames,
        cam_idx=cam_idx,
        depth_source=depth_source,
        image_size_hw=image_size_hw,
        disparity_estimator_model=disparity_estimator_model,
        disparity_estimator_checkpoint=disparity_estimator_checkpoint,
        gs_regressor_model=gs_regressor_model,
        gs_regressor_checkpoint=gs_regressor_checkpoint,
        orbit_type=orbit_type,
        force=force,
        calibration_method=calibration_method,
        rotate=rotate,
        camera_orbit_velocity=camera_orbit_velocity,
        save_ply=save_ply
    )


@app.task(name="reconstruction.generate_teaser_video", base=AutoRetryTask)
def generate_teaser_video(capturestudio_cache_root: str,
                          recon_type: Literal['pcd', 'gs'],
                          depth_source: Literal['bilateral_temporal', 'stereo'],
                          disparity_estimator_model: Literal['raftstereo', 'foundationstereo'],
                          start_frame: int,
                          total_frames: int,
                          force: bool = False):
    from reconstruction.vis.teaser import create_teaser_video

    capturestudio_cache_root = Path(capturestudio_cache_root)
    all_reconstructions_dir = capturestudio_cache_root / f'reconstruction'
    reconstruction_dir = all_reconstructions_dir / f"{recon_type}_{disparity_estimator_model if depth_source == 'stereo' else depth_source}"

    cfg = {
        # source files
        "paths": {
            "rgb": str(reconstruction_dir / f'rgb_rgb_{start_frame:04d}_{total_frames:03d}.mp4'),
            "depth": str(reconstruction_dir / f'depth_depth_{start_frame:04d}_{total_frames:03d}.mp4'),
            "norm": str(reconstruction_dir / f'feat_feat_{start_frame:04d}_{total_frames:03d}.mp4'),
        },

        # output & global video params
        "output": str(reconstruction_dir / f'teaser_{start_frame:04d}_{total_frames:03d}.mp4'),
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
    }
    create_teaser_video(cfg, force=force)
    return None


@app.task(name="reconstruction.generate_teaser_grid_video", base=AutoRetryTask)
def generate_teaser_grid_video(capturestudio_cache_root: str,
                               recon_types: List[Literal['pcd', 'gs']],
                               depth_sources: List[Literal['bilateral_temporal', 'stereo']],
                               disparity_estimator_model: Literal['raftstereo', 'foundationstereo'],
                               start_frame: int,
                               total_frames: int,
                               force: bool = False):
    from reconstruction.vis.teaser import create_teaser_grid

    capturestudio_cache_root = Path(capturestudio_cache_root)
    all_reconstructions_dir = capturestudio_cache_root / f'reconstruction'
    reconstruction_dirs = [
        all_reconstructions_dir / f"{recon_type}_{disparity_estimator_model if depth_source == 'stereo' else depth_source}"
        for depth_source in depth_sources
        for recon_type in recon_types
    ]

    grid_cfg = {
        # Six input videos, row-major order: [row0-col0, row0-col1, …]
        "paths": [str(d / f'teaser_{start_frame:04d}_{total_frames:03d}.mp4') for d in reconstruction_dirs],

        # Short labels that will be burned into the clips (same order)
        "labels": [d.name.upper().replace("_", " ") for d in reconstruction_dirs],

        # Font & styling for overlay text
        "font": "JetBrainsMono-Regular.ttf",
        "font_size": 18,
        "font_color": "white",
        "stroke_color": "black",
        "stroke_width": 0,

        # Output file & encoding
        "output": str(all_reconstructions_dir / f'teaser_grid_{start_frame:04d}_{total_frames:03d}.mp4'),
        "fps": 30,
        "bitrate": "15M",
    }
    create_teaser_grid(grid_cfg, force=force)
    return None


@app.task(name="reconstruction.link_to_webapp", base=AutoRetryTask)
def link_to_webapp(capturestudio_cache_root: str, webapp_public_root: str):
    capturestudio_cache_root = Path(capturestudio_cache_root)
    all_reconstructions_dir = capturestudio_cache_root / f'reconstruction'

    webapp_public_root = Path(webapp_public_root)
    webapp_all_reconstructions_dir = webapp_public_root / 'reconstructions'
    webapp_all_reconstructions_dir.mkdir(parents=True, exist_ok=True)
    webapp_session_reconstructions_dir = webapp_all_reconstructions_dir / capturestudio_cache_root.name

    # Create a folder named after the session name inside the webapp reconstructions directory, and make it point to the "reconstruction" directory in capturestudio cache
    if not (webapp_session_reconstructions_dir.exists() or webapp_session_reconstructions_dir.is_symlink()):
        webapp_session_reconstructions_dir.symlink_to(all_reconstructions_dir, target_is_directory=True)
