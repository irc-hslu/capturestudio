import json
from pathlib import Path
from typing import Literal, Optional

from tasks import app, AutoRetryTask
from utils.misc import log, PathUtils


@app.task(name="preprocessing.color.generate_video", base=AutoRetryTask)
def generate_video(cam_color_dir: str, start_offset: int = 0, total_frames: int = -1, fps: int = 30):
    from preprocessing.generate_video import frames_to_video
    frames_to_video(cam_color_dir, start_offset=start_offset, total_frames=total_frames, fps=fps)
    return None


@app.task(name="preprocessing.color.compute_segmentation_mask", base=AutoRetryTask)
def compute_segmentation_mask(cam_color_dir: str, out_dir: str, start_offset: int = 0, total_frames: int = -1, rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None):
    """
    Generate segmentation masks for a camera's color frames. It is assumed that a video has already been generated from the color frames, and a corresponding txt file with the frame paths exist in the color directory.

    Parameters
    ----------
    cam_color_dir : str
        Path to the camera's color directory containing the color frames, the video, and the txt file with frame paths.
    out_dir : str
        Path to the output directory where the segmentation masks will be saved.
    start_offset : int
        The starting index of the color frames to process. Default is 0.
    total_frames : int
        The total number of color frames to process. If -1, all frames from the start offset will be processed.
    rotate : Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']]
        If specified, rotate the input frames by the given angle for detection and segmentation. The outputs will be rotated back to the original orientation.
    """
    # Read frame paths from the txt file
    cam_color_dir = Path(cam_color_dir)
    if total_frames < 0:
        total_frames = len(list(cam_color_dir.glob('*.jpg'))) + total_frames - start_offset + 1
    video_path = cam_color_dir / f'video-{start_offset:06d}-{total_frames:06d}.mp4'
    frame_paths_txt = video_path.with_suffix('.txt')
    assert video_path.exists() and video_path.is_file(), f"Video file {video_path} does not exist"
    if not frame_paths_txt.exists():
        # REGENERATE FRAME PATHS TXT
        log(f"Generating frame paths txt for {cam_color_dir.name} as it does not exist.", 'warning')
        # Get all color files
        color_files = sorted(cam_color_dir.glob('*.jpg'), key=lambda x: int(x.stem))
        if total_frames == -1:
            total_frames = len(color_files) - start_offset
        assert (start_offset + total_frames) <= len(color_files), f"Start offset {start_offset} + total frames {total_frames} exceeds the number of available color files {len(color_files)} in {cam_color_dir}."
        # Write all the frame paths to a text file
        with open(frame_paths_txt, 'w') as f:
            for img_path in color_files[start_offset:start_offset + total_frames]:
                f.write(f"file '{str(img_path.resolve())}'\n")
    assert frame_paths_txt.exists() and frame_paths_txt.is_file(), f"Frame paths file {frame_paths_txt} does not exist"
    with open(frame_paths_txt, 'r') as f:
        color_files = sorted([Path(line.strip().split('file ')[-1].strip('\'\" ')) for line in f.readlines() if line.strip()], key=lambda x: int(x.stem))
    assert all([p.exists() for p in color_files]), '\n'.join([str(p) for p in color_files if not p.exists()])
    # Check if masks already exist
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_file_paths = [out_dir / cf.name for cf in color_files]
    if all([p.exists() and PathUtils.verify_file(p) for p in mask_file_paths]):
        log(f"\tSegmentation masks already exist in {out_dir}. Skipping segmentation.", 'debug')
        return None

    # First run detection on the first frame to get bboxes based on the classes
    first_frame_path = color_files[0]
    detections_path = out_dir / f'detections-{first_frame_path.stem}.json'
    if not detections_path.exists():
        from preprocessing.color import detect
        detections = detect(first_frame_path, rotate=rotate, unrotate_output=True)
        with open(detections_path, 'w') as f:
            json.dump(detections, f, indent=4)
    with open(detections_path, 'r') as f:
        detections = json.load(f)
    # Then segment in video
    from preprocessing.color import segment
    segment(video_path, detections, mask_file_paths, rotate=rotate, unrotate_output=True)
    return None


@app.task(name="preprocessing.color.compute_optical_flow", base=AutoRetryTask)
def compute_optical_flow(
        cam_color_dir: str,
        out_dir_bwd: str,
        out_dir_fwd: Optional[str] = None,
        start_offset: int = 0,
        total_frames: int = -2,
        which: Literal["fwd", "bwd", "fwd+bwd"] = "bwd",
        rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None
):
    """
    Generate OFs for the given color directory, start offset and total frames, and flow mode.

    Parameters
    ----------
    cam_color_dir : str
        Path to the camera's color directory containing the color frames, the video, and the txt file with frame paths.
    out_dir_bwd : str
        Path to the output directory where the bwd optical flows will be saved (i.e. from t-1 --> t).
    out_dir_fwd : Optional[str]
        Path to the output directory where the fwd optical flows will be saved (i.e. from t+1 --> t).
        If None, only bwd optical flows will be generated.
    start_offset : int
        The starting index of the color frames to process. Default is 0.
    total_frames : int
        The total number of color frames to process. If -1, all frames from the start offset will be processed.
    which : Literal['fwd', 'bwd', 'fwd+bwd'] = 'bwd'
        OF mode. Fwd estimates from t+1 --> t, while 'bwd' estimates from t-1 --> t. In both cases the OF is stored in t-th file path.
    rotate : Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']]
        If specified, rotate the input frames by the given angle for OF estimation. The outputs will
        be rotated back to the original orientation.
    """
    if 'fwd' in which:
        assert out_dir_fwd is not None, f'out_dir_fwd must be provided if which={which}.'

    from preprocessing.color import estimate_optical_flow
    estimate_optical_flow(Path(cam_color_dir), Path(out_dir_bwd), Path(out_dir_fwd) if out_dir_fwd is not None else None, start_offset=start_offset, total_frames=total_frames, which=which, rotate=rotate)
    return None