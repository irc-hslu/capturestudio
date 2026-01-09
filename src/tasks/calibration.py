import json
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import toml

from tasks import app, AutoRetryTask
from utils.misc import PathUtils, log


@app.task(name="calibration.generate_caliscope_config", base=AutoRetryTask)
def generate_caliscope_config(capturestudio_cache_root: str):
    """
    Generate a configuration file for Caliscope-based calibration from a Capture Studio session directory.

    Parameters
    ----------
    capturestudio_cache_root: str
        Path to the session folder containing camera directories, e.g. "/root/CAPTURESTUDIO_CACHE/Vlad_1_Calib_1".
    """
    # Define output directory for caliscope-based calibration
    capturestudio_cache_root = Path(capturestudio_cache_root)
    caliscope_dir = capturestudio_cache_root / '__calib__' / 'caliscope'
    caliscope_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate caliscope configuration file
    config_path = caliscope_dir / 'config.toml'
    if not config_path.exists():
        config = {}

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
            config[f'cam_{cam_idx_s1}'] = dict(
                port=cam_idx_s1,
                physical_index=cam_idx_s1 - 1,
                serial_number=f'SERIAL_NUMBER_{cam_idx_s1}',  # dummy
                rotation_count=0,
                error=0.01,
                translation="null",
                rotation="null",
                exposure="null",
                grid_count=20,
                size=(first_frame_size_hw[1], first_frame_size_hw[0]),  # width, height
                matrix=color_intri.tolist(),  # 3x3 list
                distortions=color_dist.tolist()[:5],  # k1, k2, p1, p2, k3
            )
            cam_count += 1

        # Load charuco profile
        if (capturestudio_cache_root / 'orbbec' / 'session_metadata.json').exists():
            with open(capturestudio_cache_root / 'orbbec' / 'session_metadata.json', 'r') as fp:
                calibration_pattern = json.load(fp).get('calibration_pattern', 'charuco_6x4_a2')
        else:
            log('[calibration.generate_caliscope_config] session_metadata.json not found, using default charuco_6x4_a2', 'warning')
            calibration_pattern = 'charuco_6x4_a2'
        with open(PathUtils.resources_path() / 'calibration_patterns' / calibration_pattern / 'charuco_info.json', 'r') as fp:
            charuco_profile = json.load(fp)

        # Generate config file using intrinsic and distortion parameters
        with open(config_path, 'w') as fp:
            toml.dump(
                dict(
                    camera_count=cam_count,
                    creation_date=datetime.now().isoformat(),
                    save_tracked_points_video=True,
                    **OrderedDict(sorted(config.items(), key=lambda x: int(x[0].split('_')[1]))),
                    charuco=charuco_profile,
                ),
                fp
            )
        log(f"[calibration.generate_caliscope_config] Caliscope configuration file generated to {config_path}", 'debug')

    return True


@app.task(name="calibration.generate_caliscope_videos", base=AutoRetryTask)
def generate_caliscope_videos(capturestudio_cache_root: str, cam_name: str, start_offset: int = 0, total_frames: int = -1, fps: int = 30):
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
    """
    # Define output directory for caliscope-based calibration
    capturestudio_cache_root = Path(capturestudio_cache_root)
    caliscope_dir = capturestudio_cache_root / '__calib__' / 'caliscope'
    caliscope_dir.mkdir(parents=True, exist_ok=True)
    extrinsic_videos_dir = caliscope_dir / 'calibration' / 'extrinsic'
    extrinsic_videos_dir.mkdir(parents=True, exist_ok=True)

    # 2) Generate caliscope videos
    cam_dir = capturestudio_cache_root / 'orbbec' / cam_name
    cam_idx_s1 = int(cam_dir.name.replace('orbbec/cam', '').replace('cam', ''))
    cam_color_dir = cam_dir / 'color'
    assert cam_color_dir.exists() and cam_color_dir.is_dir(), f"Color directory {cam_color_dir} does not exist"
    if total_frames < 0:
        total_frames = len(list(cam_color_dir.glob('*.jpg'))) - start_offset
    video_path = extrinsic_videos_dir / f'port_{cam_idx_s1}.mp4'
    if video_path.exists() and PathUtils.verify_file(video_path):
        log(f"[calibration.generate_caliscope_videos] \tCaliscope video already exists at {video_path}. Skipping video generation.", 'debug')
        return True

    # Create video
    from preprocessing.generate_video import frames_to_video
    written_video_path = frames_to_video(cam_color_dir, start_offset=start_offset, total_frames=total_frames, fps=fps)
    # move to extrinsic_videos_dir
    shutil.move(written_video_path, video_path)
    return True
