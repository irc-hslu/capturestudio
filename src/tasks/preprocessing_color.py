import json
import os
from pathlib import Path
from typing import Literal, Optional

from tasks import app, AutoRetryTask
from utils.misc import log, PathUtils

os.environ['DECORD_EOF_RETRY_MAX'] = '20480'


@app.task(name="preprocessing.color.generate_video", base=AutoRetryTask)
def generate_video(cam_color_dir: str, start_offset: int = 0, total_frames: int = -1, fps: int = 30):
    from preprocessing.generate_video import frames_to_video
    frames_to_video(cam_color_dir, start_offset=start_offset, total_frames=total_frames, fps=fps)
    return None


@app.task(name="preprocessing.color.interactively_annotate", base=AutoRetryTask)
def interactive_annotate(session_path: str):
    """
    Launch the interactive annotator app for a session and block until annotation is complete.

    Completeness criterion:
    For every camera directory under `<session_dir>/orbbec/cam*`, a JSON file must exist in
    `<cam_dir>/mask/` with name `detections-<first_color_stem>.json`, where `<first_color_stem>`
    is the stem of the first color frame (sorted by numeric stem) in `<cam_dir>/color/*.jpg`.

    Parameters
    ----------
    session_path : str
        Absolute or relative path to a CaptureStudio session directory.

    Returns
    -------
    bool
        True when annotation completeness is reached.
    """
    import os
    import subprocess
    import time

    session_path = str(Path(session_path).resolve())
    orbbec_dir = Path(session_path) / 'orbbec'
    cam_dirs = sorted(
        [_ for _ in orbbec_dir.glob('cam*') if _.is_dir()],
        key=lambda x: int(x.name.replace('cam', '').split()[0])
    )
    annotator_repo_path = PathUtils.project_path() / 'annotator'
    backend_dir = annotator_repo_path / 'python_backend'
    if not backend_dir.exists():
        raise FileNotFoundError(f"Annotator backend directory not found: {backend_dir}")

    # Ensure mask dirs exist (the app will write into them)
    for cam_dir in cam_dirs:
        (cam_dir / 'mask').mkdir(parents=True, exist_ok=True)

    # Prefer running via the annotator's uv-managed venv python
    venv_python = backend_dir / '.venv' / 'bin' / 'python'
    if not venv_python.exists():
        venv_python = backend_dir / '.venv' / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        raise FileNotFoundError(
            f"Annotator venv python not found under: {backend_dir / '.venv'} "
            f"(expected .venv/bin/python or .venv/Scripts/python.exe)."
        )

    # Ports / CORS (frontend dev is typically :3000, backend default per README is :8060)
    env = os.environ.copy()
    env.setdefault("BACKEND_PORT", "8060")
    env.setdefault("FRONTEND_PORT", "3001")
    env.setdefault("ALLOWED_ORIGINS", f'http://localhost:{env["FRONTEND_PORT"]},http://127.0.0.1:{env["FRONTEND_PORT"]}')
    env.setdefault("NEXT_PUBLIC_SESSION_PATH", session_path)  # auto-open the session in the UI
    cmd = [
        str(venv_python),
        "run.py",
        "--session", session_path,
        "--frontend-port", env["FRONTEND_PORT"],
        "--frontend-dir", "..",
        "--frontend-mode", "dev",
    ]

    log(
        f"[interactive_annotate] Starting annotator for session: {session_path}\n"
        f"\tcmd: {' '.join(cmd)}\n"
        f"\tfrontend: http://localhost:{env['FRONTEND_PORT']}\n"
        f"\tbackend:  http://localhost:{env['BACKEND_PORT']}\n"
        f"\t(waiting for detections-<first_stem>.json in each cam*/mask/ ...)",
        "info"
    )

    proc = subprocess.Popen(
        cmd,
        cwd=str(backend_dir),
        env=env,
        stdout=None,
        stderr=None,
    )

    def _first_color_stem(cam_dir: Path) -> str:
        color_dir = cam_dir / 'color'
        if not color_dir.exists():
            raise FileNotFoundError(f"Missing color folder: {color_dir}")
        color_files = sorted(color_dir.glob('*.jpg'), key=lambda p: int(p.stem))
        if not color_files:
            raise FileNotFoundError(f"No color jpgs found in: {color_dir}")
        return color_files[0].stem

    expected_json_per_cam = {}
    for cam_dir in cam_dirs:
        stem0 = _first_color_stem(cam_dir)
        expected_json_per_cam[cam_dir] = cam_dir / 'mask' / f"detections-{stem0}.json"

    poll_s = 2.0
    while True:
        # If the app crashed/exited, fail the task (AutoRetryTask will retry)
        rc = proc.poll()
        if rc is not None:
            missing = [str(p) for p in expected_json_per_cam.values() if not p.exists()]
            raise RuntimeError(
                f"[interactive_annotate] Annotator process exited with code {rc}. "
                f"Missing {len(missing)} detection json files."
            )

        all_done = True
        for cam_dir, json_path in expected_json_per_cam.items():
            if not json_path.exists() or not json_path.is_file():
                all_done = False
                break
            # Basic sanity: file is non-empty
            if json_path.stat().st_size <= 2:
                all_done = False
                break

        if all_done:
            log("[interactive_annotate] Annotation completeness reached for all cameras.", "info")
            break

        time.sleep(poll_s)

    # Terminate annotator process
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    return True


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
    cam_color_dir: Path = Path(cam_color_dir)
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
    out_dir: Path = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_file_paths = [out_dir / cf.name for cf in color_files]
    if all([p.exists() and PathUtils.verify_file(p) for p in mask_file_paths]):
        log(f"\tSegmentation masks already exist in {out_dir}. Skipping segmentation.", 'debug')
        return None

    # First run detection on the first frame to get bboxes based on the classes
    first_frame_path = color_files[0]
    detections_path = out_dir / f'detections-{first_frame_path.stem}.json'

    # Build a global index map from the full color folder
    all_color_files_global = sorted(cam_color_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    stem_to_global_idx = {p.stem: i for i, p in enumerate(all_color_files_global)}

    if not detections_path.exists():
        # If it finds other detections e.g. detections-<another frame stem>.json:
        # Use that instead and compute first_frame_global_idx to be the idx of the stem from the color folder.
        alt_detection_files = sorted(out_dir.glob('detections-*.json'))
        chosen_alt = None
        chosen_alt_idx = None
        for p in alt_detection_files:
            stem = p.stem.split('detections-', 1)[-1]
            if stem in stem_to_global_idx and (cam_color_dir / f"{stem}.jpg").exists():
                chosen_alt = p
                chosen_alt_idx = stem_to_global_idx[stem]
                break

        if chosen_alt is not None:
            detections_path = chosen_alt
            first_frame_global_idx = int(chosen_alt_idx)
        else:
            from preprocessing.color import detect
            detections = detect(first_frame_path, rotate=rotate, unrotate_output=True)
            first_frame_global_idx = int(stem_to_global_idx.get(first_frame_path.stem, 0))
            with open(detections_path, 'w') as f:
                json.dump(detections, f, indent=4)
    else:
        det_stem = detections_path.stem.split('detections-', 1)[-1]
        first_frame_global_idx = int(stem_to_global_idx.get(det_stem, stem_to_global_idx.get(first_frame_path.stem, 0)))

    with open(detections_path, 'r') as f:
        detections = json.load(f)

    # Then segment in video
    from preprocessing.color import segment
    segment(video_path, detections, mask_file_paths, rotate=rotate, unrotate_output=True, first_frame_global_idx=first_frame_global_idx)
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
