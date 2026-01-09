import os
import re
import subprocess
from pathlib import Path
from typing import Union, Literal

import cv2
import numpy as np

from utils.misc import log, PathUtils


def frames_to_video(frame_dir: Union[Path, str], start_offset: int = 0, total_frames: int = -1, fps: int = 30, celery_app=None):
    frame_dir = Path(frame_dir)
    # Get all color files
    color_files = sorted(frame_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    if total_frames == -1:
        total_frames = len(color_files) - start_offset
    assert (start_offset + total_frames) <= len(color_files), f"Start offset {start_offset} + total frames {total_frames} exceeds the number of available color files {len(color_files)} in {frame_dir}."
    # Write all the frame paths to a text file
    video_path = frame_dir / f'video-{start_offset:06d}-{total_frames:06d}.mp4'
    frames_txt_path = frame_dir / f'video-{start_offset:06d}-{total_frames:06d}.txt'
    lines = []
    for img_path in color_files[start_offset:start_offset + total_frames]:
        lines.append(f"file '{str(img_path.resolve())}'")
    payload = "\n".join(lines) + "\n"
    if frames_txt_path.exists() and video_path.exists() and frames_txt_path.stat().st_size == len(payload.encode('utf-8')) and PathUtils.verify_file(video_path, required_frames=total_frames):
        return None

    # Write frames.txt to a temp file in the same directory, then atomically replace
    tmp_path = frames_txt_path.with_suffix(f".txt.tmp")
    out_dir_fd = os.open(str(frames_txt_path.parent), os.O_DIRECTORY)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)  # single write
            f.flush()
            os.fsync(f.fileno())  # force file data+metadata to disk

        # Atomic rename -> readers never see a partial file
        os.replace(tmp_path, frames_txt_path)

        # fsync the directory so the rename is durable
        os.fsync(out_dir_fd)
    finally:
        os.close(out_dir_fd)
        # Best-effort cleanup if something failed before replace
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

    # Generate the video460870
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-r", str(fps),
        "-i", str(video_path.with_suffix('.txt')),
        "-c:v", "h264_nvenc",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(video_path)
    ]
    process = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE if celery_app is not None else subprocess.DEVNULL,
        universal_newlines=True,
        bufsize=1
    )
    # duration = None
    # for line in process.stderr:
    #     line = line.strip()
    #     # Extract duration
    #     if duration is None:
    #         match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", line)
    #         if match:
    #             h, m, s = map(float, match.groups())
    #             duration = h * 3600 + m * 60 + s
    #     # Extract current progress timestamp
    #     match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
    #     if match and duration:
    #         h, m, s = map(float, match.groups())
    #         current = h * 3600 + m * 60 + s
    #         progress = current / duration
    process.wait()
    return video_path


def generate_multiview_video(capturestudio_cache_path: str, subfolder: Literal['color', 'depth', 'depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal'], file_extension: Literal['jpg', 'png', 'npy']):
    """
    Generate multi-camera video for a given capture session.
    ATTN: This task assumes that the frames have already been synchronized.

    Parameters
    ----------
    capturestudio_cache_path: str
        Path to the session folder containing camera directories, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".
    subfolder: str, optional
        The depth modality to use for the multiview video. Options are 'depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal'. Default is 'depth_aligned'.
    file_extension: str, optional
        The format of the depth images. Options are 'png' or 'jpg'. Default is 'png'.
    """
    capturestudio_cache_path = Path(capturestudio_cache_path)
    output_video_path = capturestudio_cache_path / 'orbbec' / f'multiview_{subfolder}.mp4'
    if output_video_path.exists():
        log(f"Multiview video {output_video_path} already exists. Skipping generation.", 'debug')
        return None

    is_depth = 'depth' in str(subfolder)
    all_cam_dirs = sorted([d for d in (capturestudio_cache_path / 'orbbec').iterdir() if d.is_dir() and d.name.startswith('cam') and (d / subfolder).exists()], key=lambda d: int(d.name.split('cam')[1].split(' ')[0]))
    all_file_paths = {
        cam_dir.name: sorted(cam_dir.glob(f'{subfolder}/*.{file_extension}'), key=lambda x: int(x.stem))
        for cam_dir in all_cam_dirs
    }
    total_len = len(all_file_paths[all_cam_dirs[0].name])
    main_idx = -1
    image_size = PathUtils.read_file(all_file_paths[all_cam_dirs[0].name][0], png_type='depth' if is_depth else 'color').shape[:2]  # (height, width)
    if image_size == (640, 576) or image_size == (576, 640):
        # Orbbec Fempto Bolt depth
        main_height, main_width = int(image_size[0] * 1.5), int(image_size[1] * 1.5)
        crop_width = 0
    else:
        main_height, main_width = 1024, 1820
        crop_width = 270
    small_height = main_height // (5 if len(all_cam_dirs) == 5 else 4)
    small_width = (main_width - 2 * crop_width) // (5 if len(all_cam_dirs) == 5 else 4)
    video_writer = None
    dummy_frame1 = np.zeros((small_height, small_width, 3), dtype=np.uint8)
    dummy_frame3 = np.zeros((small_height * 3, small_width, 3), dtype=np.uint8)
    dummy_frame4 = np.zeros((small_height * 4, small_width, 3), dtype=np.uint8)
    for t in range(total_len):
        if t % 100 == 0:
            main_idx = (main_idx + 1) % len(all_cam_dirs)  # cycle through cameras every 100 frames
        frames = [
            ((PathUtils.read_file(all_file_paths[cam_dir.name][t], png_type='depth').clip(200, 4_000).astype(np.float32) - 200) / 3_800) if is_depth else
            PathUtils.read_file(all_file_paths[cam_dir.name][t])
            for cam_dir in all_cam_dirs
        ]
        for fi, (frame, frame_cam_dir) in enumerate(zip(frames, all_cam_dirs)):
            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                log(f"Frame {t} in camera {frame_cam_dir.name} is empty. Replacing it with previous frame in folder in os...", 'warning')
                current_frame_path = all_file_paths[frame_cam_dir.name][t]
                prev_frame = PathUtils.read_file(all_file_paths[frame_cam_dir.name][t - 1], png_type='depth' if is_depth else 'generic')
                PathUtils.write_file(current_frame_path, prev_frame, png_type='depth' if is_depth else 'generic')
                log(f"\t {all_file_paths[frame_cam_dir.name][t]} <-- {all_file_paths[frame_cam_dir.name][t - 1]}.", 'warning')
                if is_depth:
                    frames[fi] = (prev_frame.clip(200, 4_000).astype(np.float32) - 200) / 3_800
                else:
                    frames[fi] = prev_frame
        # resize to have height of 256px and keep aspect ratio
        resized_frames = [
            cv2.resize(cv2.resize(frame, (main_width, main_height), interpolation=cv2.INTER_LINEAR)[:, crop_width:-crop_width], (small_width, small_height), interpolation=cv2.INTER_LINEAR) if crop_width > 0 else
            cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ]
        main_frame = cv2.resize(frames[main_idx], (main_width, main_height), interpolation=cv2.INTER_LINEAR)
        if crop_width > 0:
            main_frame = main_frame[:, crop_width:-crop_width]
        # Apply colormap
        if is_depth:
            resized_frames = [
                cv2.applyColorMap((frame * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                for frame in resized_frames
            ]
            main_frame = cv2.applyColorMap((main_frame * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        # Highlight the satellite frame which is the main frame
        resized_frames[main_idx] = cv2.rectangle(resized_frames[main_idx], (0, 0), (small_width - 1, small_height - 1), (144, 238, 144), 1)
        # create frame
        if len(all_cam_dirs) == 12:
            canvas = np.concatenate([
                np.concatenate([dummy_frame1.copy(), *resized_frames[:4]], axis=0),
                np.concatenate([
                    main_frame,
                    np.concatenate(resized_frames[4:8], axis=1),
                ], axis=0),
                np.concatenate([dummy_frame1.copy(), *resized_frames[-4:][::-1]], axis=0),
            ], axis=1)
        elif len(all_cam_dirs) == 8:
            canvas = np.concatenate([
                np.concatenate([dummy_frame3.copy(), *resized_frames[:2]], axis=0),
                np.concatenate([
                    main_frame,
                    np.concatenate(resized_frames[3:7], axis=1),
                ], axis=0),
                np.concatenate([dummy_frame3.copy(), *resized_frames[-2:][::-1]], axis=0),
            ], axis=1)
        elif len(all_cam_dirs) == 6:
            canvas = np.concatenate([
                np.concatenate([dummy_frame4.copy(), resized_frames[0]], axis=0),
                np.concatenate([
                    main_frame,
                    np.concatenate(resized_frames[1:5], axis=1),
                ], axis=0),
                np.concatenate([dummy_frame4.copy(), resized_frames[-1]], axis=0),
            ], axis=1)
        elif len(all_cam_dirs) == 4 or len(all_cam_dirs) == 5:
            canvas = np.concatenate([
                main_frame,
                np.concatenate(resized_frames, axis=1),
            ], axis=0)
        else:
            raise ValueError(f"Unsupported number of cameras: {len(all_cam_dirs)}. Expected 12, 8, 6, or 4 cameras for multicam depth video generation.")

        if video_writer is None:
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,  # fps
                (canvas.shape[1], canvas.shape[0])
            )

        video_writer.write(canvas)

    if video_writer is not None:
        video_writer.release()
        log(f"Multiview video generated for session {capturestudio_cache_path.name} ({subfolder})", 'debug')
    return None
