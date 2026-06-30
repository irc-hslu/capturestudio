import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from typing import Dict, Tuple, Any, List

from utils.misc import PathUtils, log
from utils.sync import SyncUtils


def synchronize_frames_for_session(session_root: str, excel_sheet: str = 'Apr_May 2025', excel_file_path: str = 'Capture Sessions.xls'):
    """
    Synchronize frames for a session based on the timestamps found in the frame filenames or external flash sync signal.

    Parameters
    ----------
    session_root: str
        Path to the session folder containing camera directories, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".
    excel_sheet: str, optional
        Name of the Excel sheet containing the session data (default: 'Apr_May 2025').
    excel_file_path: str, optional
        Path to the Excel file containing the session data (default: 'Capture Sessions.xls').
    """
    session_root = Path(session_root)
    # os-sync all dirs (push all pending changes to disk)
    PathUtils.sync_dir(session_root / 'orbbec')
    try:
        # try timestamp-based synchronization first
        SyncUtils.synchronize_frames_using_timestamps(str(session_root))
        all_cam_dirs = sorted([d for d in (session_root / 'orbbec').iterdir() if d.is_dir() and d.name.startswith('cam')], key=lambda d: int(d.name.split('cam')[-1].split(' ')[0]))
        all_color_dirs = [cam_dir / 'color' for cam_dir in all_cam_dirs]
        all_color_counts = [len(list(cd.glob('*.jpg'))) for cd in all_color_dirs]
        common_length = all_color_counts[0]
        if common_length == 0:
            raise RuntimeError(f"Common length of synchronized frames for session {session_root.name} is 0. Check the frame timestamps.")
        log(f"Timestamp-based synchronization successful for session {session_root.name}", 'debug')
    except RuntimeError as e:
        log(f"Timestamp-based synchronization failed for session {session_root.name}: {e}", 'error')
        # Synchronize frames based on the flash sync signal
        from utils.excel import ExcelUtils
        row = ExcelUtils.get_session_data(session_root.name, sheet=excel_sheet, excel_file_path=excel_file_path)
        try:
            visual_sync_points = dict(
                orbbec=row['Visual Sync Frame'],
                sony=row['Unnamed: 22'],
                apple=row['Unnamed: 23']
            )
            valid_visual_sync_points = all(
                not pd.isna(sync_point) and sync_point != '-' for sync_point in visual_sync_points.values()
            )
        except (KeyError, ValueError) as e:
            log(f"Error extracting visual sync points for session {session_root.name}: {e}", 'error')
            visual_sync_points = {}
            valid_visual_sync_points = False
        try:
            audio_sync_points = dict(
                orbbec=row['Audio Sync Frame'],
                sony=row['Unnamed: 26'],
                apple=row['Unnamed: 27']
            )
            valid_audio_sync_points = all(
                not pd.isna(sync_point) and sync_point != '-' for sync_point in audio_sync_points.values()
            )
        except (KeyError, ValueError) as e:
            log(f"Error extracting audio sync points for session {session_root.name}: {e}", 'error')
            audio_sync_points = {}
            valid_audio_sync_points = False
        if valid_visual_sync_points or valid_audio_sync_points:
            # synchronize using clap
            log(f"Synchronizing Orbbec cameras for session {session_root.name} using clap", 'debug')
            SyncUtils.synchronize_frames_using_clap(str(session_root), visual_sync_points, audio_sync_points, sync_using_audio=valid_audio_sync_points and not valid_visual_sync_points)
            log(f"Clap-based synchronization successful for session {session_root.name}", 'debug')

    # Post check
    # ask OS to flush all changes to the dirs
    PathUtils.sync_dir(session_root / 'orbbec')

    all_cam_dirs = sorted([d for d in (session_root / 'orbbec').iterdir() if d.is_dir() and d.name.startswith('cam')], key=lambda d: int(d.name.split('cam')[-1].split(' ')[0]))
    all_color_dirs = [cam_dir / 'color' for cam_dir in all_cam_dirs]
    all_color_counts = [len(list(cd.glob('*.jpg'))) for cd in all_color_dirs]
    all_depth_dirs = [cam_dir / 'depth' for cam_dir in all_cam_dirs if (cam_dir / 'depth').exists()]
    all_depth_counts = [max(len(list(dd.glob('*.npy'))), len(list(dd.glob('*.png')))) for dd in all_depth_dirs]
    depth_i = 0
    sync_successful = True
    for cam_dir, color_count in zip(all_cam_dirs, all_color_counts):
        if color_count != all_color_counts[0]:
            sync_successful = False
            log(f"\tSynchronization failed for {cam_dir.name}. len(COLOR)={color_count} != {all_color_counts[0]} (from first folder).", 'error')
        if (cam_dir / 'depth').exists():
            depth_count = all_depth_counts[depth_i]
            if depth_count != all_depth_counts[0] or depth_count != all_color_counts[0]:
                sync_successful = False
                log(f"\tSynchronization failed for {cam_dir.name}. len(DEPTH)={depth_count} != {all_depth_counts[0]} (depth) or != {all_color_counts[0]} (color).", 'error')
            depth_i += 1
    # rename all .<original_ext>.unsync files to .<original_ext> if synchronization was not successful, otherwise remove them
    for cam_dir in all_cam_dirs:
        for subdir in ['color', 'mask', 'depth', 'depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal']:
            subdir_path = cam_dir / subdir
            if subdir_path.exists():
                for unsync_file in subdir_path.glob('*.unsync'):
                    if not sync_successful:
                        new_name = unsync_file.with_suffix(unsync_file.suffix.replace('.unsync', ''))
                        if not new_name.exists():
                            unsync_file.rename(new_name)
                    else:
                        # remove the .unsync files if synchronization was successful
                        unsync_file.unlink()
    if sync_successful:
        common_length = all_color_counts[0]
        assert common_length > 0, f"Common length of synchronized frames for session {session_root.name} is 0."
        log(f"\tSynchronization done. New common length: {common_length}", 'debug')
        return True

    # ask OS to flush all changes to the dirs
    PathUtils.sync_dir(session_root / 'orbbec')
    return False


def _scan_current_counts(cam_dirs: List[Path]) -> Dict[str, Tuple[int, int]]:
    """
    Scan current (color, depth) frame counts per camera.

    Returns
    -------
    Dict[str, Tuple[int,int]]
        { cam_dir_name: (color_count, depth_count) }
        depth_count is -1 if depth folder is absent, else max(#png, #npy).
    """
    counts: Dict[str, Tuple[int, int]] = {}
    for cam_dir in cam_dirs:
        color_dir = cam_dir / 'color'
        color_count = len(list(color_dir.glob('*.jpg'))) if color_dir.exists() else 0

        depth_dir = cam_dir / 'depth'
        if depth_dir.exists():
            depth_png = len(list(depth_dir.glob('*.png')))
            depth_npy = len(list(depth_dir.glob('*.npy')))
            depth_count = max(depth_png, depth_npy)
        else:
            depth_count = -1

        counts[cam_dir.name] = (int(color_count), int(depth_count))
    return counts


def _parse_trim_info_json(trim_info_path: Path) -> Dict[str, Any]:
    """
    Expected schema:
    {
      "trimmed_at": "...",
      "start_frame": int,
      "total_frames": int,
      "window": [start_idx, end_idx],
      "cams": {
        "cam01": {"color_before": int, "color_after": int, "depth_before": int, "depth_after": int},
        ...
      }
    }
    """
    data = json.load(open(trim_info_path, 'r'))
    # Light normalization / defaults
    data.setdefault('cams', {})
    return data


def _write_trim_info_json(
        trim_info_path: Path,
        cam_dirs: List[Path],
        start_frame_offset: int,
        total_frames: int,
        window: Tuple[int, int],
        before_counts: Dict[str, Tuple[int, int]],
        after_counts: Dict[str, Tuple[int, int]],
):
    ts = datetime.now().isoformat()
    cams: Dict[str, Dict[str, int]] = {}
    for cam_dir in cam_dirs:
        cam = cam_dir.name
        cb, db = before_counts.get(cam, (0, -1))
        ca, da = after_counts.get(cam, (0, -1))
        cams[cam] = {
            "color_before": int(cb),
            "color_after": int(ca),
            "depth_before": int(db),
            "depth_after": int(da),
        }

    payload = {
        "trimmed_at": ts,
        "start_frame": int(start_frame_offset),
        "total_frames": int(total_frames),
        "window": [int(window[0]), int(window[1])],
        "cams": cams,
    }
    with open(trim_info_path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _cleanup_recursive_trimmed_folders(session_root: Path):
    """
    Finds folders like 'depth_trimmed_trimmed' and moves their contents back to 'depth_trimmed'.
    """
    orbbec_root = session_root / 'orbbec'
    if not orbbec_root.exists():
        return

    for cam_dir in orbbec_root.glob('cam*'):
        if not cam_dir.is_dir():
            continue

        # Find all directories that contain '_trimmed'
        trimmed_dirs = [d for d in cam_dir.iterdir() if d.is_dir() and '_trimmed' in d.name]

        for td in trimmed_dirs:
            # Extract the base name (e.g., 'depth' from 'depth_trimmed_trimmed')
            base_name = td.name.split('_trimmed')[0]
            target_name = f"{base_name}_trimmed"
            target_dir = cam_dir / target_name

            # If this is a recursively named folder
            if td.name != target_name:
                target_dir.mkdir(parents=True, exist_ok=True)
                log(f"Merging recursive folder {td.name} into {target_name}", 'debug')

                for item in td.iterdir():
                    if not item.is_file():
                        continue

                    dst = target_dir / item.name
                    if dst.exists():
                        # Handle collision to avoid overwriting files
                        k = 1
                        while True:
                            candidate = target_dir / f"{item.stem}__restored{k}{item.suffix}"
                            if not candidate.exists():
                                dst = candidate
                                break
                            k += 1

                    item.replace(dst)

                # Delete the now-empty recursive folder
                try:
                    td.rmdir()
                except OSError:
                    log(f"Could not remove {td.name}, it might not be empty.", 'warning')


def _trim_frames_for_session(session_root: str, start_frame_offset: int, total_frames: int = -1):
    """
    Internal trimming implementation. ALWAYS performs trimming and overwrites trim_info.json.
    It does NOT perform idempotency/count checks (handled by trim_frames_for_session()).
    """
    session_root = Path(session_root)
    orbbec_root = session_root / 'orbbec'
    cam_dirs = sorted(
        [d for d in orbbec_root.iterdir() if d.is_dir() and d.name.startswith('cam')],
        key=lambda d: int(d.name.split('cam')[-1].split(' ')[0])
    )
    if not cam_dirs:
        raise FileNotFoundError(f"No camera folders found under: {orbbec_root}")

    trim_info_path = orbbec_root / 'trim_info.json'

    def _sorted_files(dir_path: Path, ext: str):
        files = list(dir_path.glob(f'*.{ext}'))
        return sorted(files, key=lambda p: int(p.stem))

    def _trimmed_dir_for(dir_path: Path) -> Path:
        if not dir_path.name.endswith('_trimmed'):
            return dir_path.with_name(f"{dir_path.name}_trimmed")
        return dir_path

    def _safe_move_to_trimmed(p: Path, base_dir: Path):
        """
        Move file `p` into sibling `<base_dir>_trimmed/` preserving filename.
        If destination exists, add a numeric suffix to avoid overwriting.
        """
        if not p.exists():
            return
        trimmed_dir = _trimmed_dir_for(base_dir)
        trimmed_dir.mkdir(parents=True, exist_ok=True)

        dst = trimmed_dir / p.name
        if dst.exists():
            stem = p.stem
            suf = p.suffix
            k = 1
            while True:
                candidate = trimmed_dir / f"{stem}__{k}{suf}"
                if not candidate.exists():
                    dst = candidate
                    break
                k += 1

        p.replace(dst)

    # First pass: determine the common trim window based on color timestamps across cams
    per_cam: Dict[Path, Dict[str, Any]] = {}
    color_lengths: List[int] = []

    for cam_dir in cam_dirs:
        color_dir = cam_dir / 'color'
        if not color_dir.exists():
            raise FileNotFoundError(f"Missing color folder: {color_dir}")

        color_files = _sorted_files(color_dir, 'jpg')
        color_stems = [p.stem for p in color_files]
        color_lengths.append(len(color_stems))

        # Depth per camera (optional)
        depth_dir = cam_dir / 'depth'
        depth_ext = None
        depth_files: List[Path] = []
        depth_stems: List[str] = []
        if depth_dir.exists():
            depth_png = list(depth_dir.glob('*.png'))
            depth_npy = list(depth_dir.glob('*.npy'))
            if depth_png:
                depth_ext = 'png'
            elif depth_npy:
                depth_ext = 'npy'
            if depth_ext is not None:
                depth_files = _sorted_files(depth_dir, depth_ext)
                depth_stems = [p.stem for p in depth_files]

        per_cam[cam_dir] = {
            "color_dir": color_dir,
            "color_files": color_files,
            "color_stems": color_stems,
            "depth_dir": depth_dir,
            "depth_ext": depth_ext,
            "depth_files": depth_files,
            "depth_stems": depth_stems,
        }

    # Record "before" counts for trim_info.json
    before_counts: Dict[str, Tuple[int, int]] = {}
    for cam_dir, info in per_cam.items():
        color_before = len(info["color_files"])
        if info["depth_ext"] is not None and info["depth_files"]:
            depth_before = len(info["depth_files"])
        else:
            depth_before = -1 if not (cam_dir / 'depth').exists() else 0
        before_counts[cam_dir.name] = (int(color_before), int(depth_before))

    assert all(_ == color_lengths[0] for _ in color_lengths), f"Color lengths mismatch: {color_lengths}"
    global_n_color = color_lengths[0]
    if global_n_color == 0:
        raise FileNotFoundError(f"No color files found under: {color_lengths}")

    # Compute keep window (by index) on COLOR timeline
    start_idx = max(0, int(start_frame_offset))
    if start_idx > global_n_color:
        start_idx = global_n_color
    if total_frames is None or int(total_frames) == -1:
        end_idx = global_n_color
    else:
        total_frames_i = max(0, int(total_frames))
        end_idx = min(global_n_color, start_idx + total_frames_i)

    # Second pass: trim per-cam
    for cam_dir, info in per_cam.items():
        color_stems_common = info["color_stems"]
        keep_color = set(color_stems_common[start_idx:end_idx])
        removed_color = set(color_stems_common) - keep_color

        # Move color frames outside window
        for p in info["color_files"]:
            if p.stem in removed_color:
                _safe_move_to_trimmed(p, info["color_dir"])

        # Propagate to color modalities (may be missing files)
        for modality in ['mask', 'flow_bwd', 'flow_fwd']:
            mdir = cam_dir / modality
            if not mdir.exists():
                continue
            ext = 'jpg' if modality == 'mask' else 'png'
            for stem in removed_color:
                _safe_move_to_trimmed(mdir / f"{stem}.{ext}", mdir)

        # Depth is optional; if present, trim to match kept color count by index
        if info["depth_ext"] is not None and info["depth_files"]:
            depth_files = info["depth_files"]
            depth_stems = info["depth_stems"]

            common_n = min(len(depth_files), global_n_color)
            depth_stems_common = depth_stems[:common_n]
            keep_depth = set(depth_stems_common[start_idx:min(end_idx, common_n)])
            removed_depth = set(depth_stems_common) - keep_depth

            for p in depth_files[:common_n]:
                if p.stem in removed_depth:
                    _safe_move_to_trimmed(p, info["depth_dir"])
            for p in depth_files[common_n:]:
                _safe_move_to_trimmed(p, info["depth_dir"])

            for d in cam_dir.glob('depth*'):
                if not d.is_dir() or d.name == 'depth' or d.name.endswith('_trimmed'):
                    continue
                for stem in removed_depth:
                    _safe_move_to_trimmed(d / f"{stem}.png", d)

    after_counts = _scan_current_counts(cam_dirs)
    _write_trim_info_json(
        trim_info_path=trim_info_path,
        cam_dirs=cam_dirs,
        start_frame_offset=start_frame_offset,
        total_frames=total_frames,
        window=(start_idx, end_idx),
        before_counts=before_counts,
        after_counts=after_counts,
    )

    return True


def _untrim_frames_for_session(session_root: str):
    """
    Internal untrim implementation. Restores files from `<dir>_trimmed` back to `<dir>`,
    removes `<dir>_trimmed` folders, and deletes `<session_root>/orbbec/trim_info.json` if present.
    """
    session_root = Path(session_root)
    orbbec_root = session_root / 'orbbec'
    cam_dirs = sorted(
        [d for d in orbbec_root.iterdir() if d.is_dir() and d.name.startswith('cam')],
        key=lambda d: int(d.name.split('cam')[-1].split(' ')[0])
    )
    if not cam_dirs:
        raise FileNotFoundError(f"No camera folders found under: {orbbec_root}")

    trim_info_path = orbbec_root / 'trim_info.json'

    def _trimmed_dir_for(dir_path: Path) -> Path:
        return dir_path.with_name(f"{dir_path.name}_trimmed")

    def _rmtree(p: Path):
        if not p.exists():
            return
        for child in p.iterdir():
            if child.is_dir():
                _rmtree(child)
            else:
                child.unlink()
        p.rmdir()

    def _restore_dir(dir_path: Path):
        trimmed_dir = _trimmed_dir_for(dir_path)
        if not trimmed_dir.exists() or not trimmed_dir.is_dir():
            return

        dir_path.mkdir(parents=True, exist_ok=True)

        for src in trimmed_dir.iterdir():
            if not src.is_file():
                continue
            dst = dir_path / src.name
            if dst.exists():
                stem = src.stem
                suf = src.suffix
                k = 1
                while True:
                    candidate = dir_path / f"{stem}__restored{k}{suf}"
                    if not candidate.exists():
                        dst = candidate
                        break
                    k += 1
            src.replace(dst)

        _rmtree(trimmed_dir)

    for cam_dir in cam_dirs:
        for dname in ['color', 'mask', 'flow_bwd', 'flow_fwd']:
            d = cam_dir / dname
            if d.exists() and d.is_dir():
                _restore_dir(d)
            else:
                td = _trimmed_dir_for(d)
                if td.exists() and td.is_dir():
                    _restore_dir(d)

        depth_dir = cam_dir / 'depth'
        if depth_dir.exists() and depth_dir.is_dir():
            _restore_dir(depth_dir)
        else:
            td = _trimmed_dir_for(depth_dir)
            if td.exists() and td.is_dir():
                _restore_dir(depth_dir)
        depth_bases = set()
        for d in cam_dir.glob('depth_*'):
            if not d.is_dir():
                continue
            if d.name.endswith('_trimmed'):
                # Reconstruct the base name by stripping off '_trimmed'
                base_name = d.name[:-8]
                if base_name in depth_bases:
                    continue
                d = cam_dir / base_name
            _restore_dir(d)
            depth_bases.add(d.name)

    if trim_info_path.exists():
        trim_info_path.unlink()

    log(f"[trim_frames_for_session] Untrimmed session {session_root.name}.", 'debug')
    return True


def trim_frames_for_session(session_root: str, start_frame_offset: int, total_frames: int = -1):
    """
    Single public entrypoint.

    Behavior:
    - If trim_info.json exists and args match:
        1) if counts match what's in the file -> return
        2) else -> trim (repair/complete)
    - Else (trim_info.json missing OR args differ):
        1) untrim (best-effort, but without swallowing errors)
        2) trim according to args
    """
    session_root = Path(session_root)
    orbbec_root = session_root / 'orbbec'
    trim_info_path = orbbec_root / 'trim_info.json'

    cam_dirs = sorted(
        [d for d in orbbec_root.iterdir() if d.is_dir() and d.name.startswith('cam')],
        key=lambda d: int(d.name.split('cam')[-1].split(' ')[0])
    )
    if not cam_dirs:
        raise FileNotFoundError(f"No camera folders found under: {orbbec_root}")

    # # FIX: from a bug there were being generated folders with multiple "_trimmed" suffices
    # _cleanup_recursive_trimmed_folders(session_root)

    def _counts_from_trim_info(data: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        cams = data.get('cams', {}) or {}
        out: Dict[str, Tuple[int, int]] = {}
        for cam, row in cams.items():
            try:
                out[cam] = (int(row.get('color_after')), int(row.get('depth_after')))
            except Exception:
                continue
        return out

    if trim_info_path.exists():
        trim_info = _parse_trim_info_json(trim_info_path)

        args_match = (int(trim_info.get('start_frame', -999999)) == int(start_frame_offset)) and (
                int(trim_info.get('total_frames', -999999)) == int(total_frames)
        )

        if args_match:
            from_trim_info = _counts_from_trim_info(trim_info)
            current = _scan_current_counts(cam_dirs)
            if from_trim_info and set(from_trim_info.keys()) == set(current.keys()) and all(current[k] == from_trim_info[k] for k in from_trim_info.keys()):
                log(
                    f"[trim_frames_for_session] Session {session_root.name} already trimmed for "
                    f"start_frame={start_frame_offset}, total_frames={total_frames} (counts match). Skipping.",
                    'debug'
                )
                return True

            return _trim_frames_for_session(str(session_root), start_frame_offset=start_frame_offset, total_frames=total_frames)

        _untrim_frames_for_session(str(session_root))
        return _trim_frames_for_session(str(session_root), start_frame_offset=start_frame_offset, total_frames=total_frames)

    return _trim_frames_for_session(str(session_root), start_frame_offset=start_frame_offset, total_frames=total_frames)


def check_all_cams_have_equal_number_of_frames(session_root: str) -> bool:
    """
    Check if all color folders of all cameras have the same number of frames (jpg images).
    Also check if the depth folders for the cameras that have depth cues, have the same number of frames
    (either npy or png files) and the same number as the color folders.

    Parameters
    ----------
    session_root: str
        Path to the session folder containing camera directories, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".

    Returns
    -------
    bool
        If all ok -> True, else False.
    """
    session_root = Path(session_root)
    orbbec_root = session_root / 'orbbec'
    cam_dirs = sorted(
        [d for d in orbbec_root.iterdir() if d.is_dir() and d.name.startswith('cam')],
        key=lambda d: int(d.name.split('cam')[-1].split(' ')[0])
    )
    if not cam_dirs:
        log(f"[check_all_cams_have_equal_number_of_frames] No camera folders found under {orbbec_root}", 'error')
        return False

    # Color counts (required for all cameras)
    color_counts = []
    color_counts_by_cam = {}
    for cam_dir in cam_dirs:
        color_dir = cam_dir / 'color'
        if not color_dir.exists():
            log(f"[check_all_cams_have_equal_number_of_frames] Missing color folder: {color_dir}", 'error')
            return False
        c = len(list(color_dir.glob('*.jpg')))
        color_counts.append(c)
        color_counts_by_cam[cam_dir.name] = c
    common_color = color_counts[0]
    if not all(_ == common_color for _ in color_counts) or common_color == 0:
        log(f"[check_all_cams_have_equal_number_of_frames] Color frame mismatch: {str(color_counts_by_cam)}", 'error')
        return False

    # Depth counts (optional per camera)
    depth_counts = []
    depth_counts_by_cam = {}
    for cam_dir in cam_dirs:
        depth_dir = cam_dir / 'depth'
        if not depth_dir.exists():
            continue
        n_png = len(list(depth_dir.glob('*.png')))
        n_npy = len(list(depth_dir.glob('*.npy')))
        d = max(n_png, n_npy)
        if d == 0:
            # Depth folder exists but empty -> remove folder
            log(f"[check_all_cams_have_equal_number_of_frames] DEPTH folder exists but empty for {cam_dir.name}: {depth_dir} (REMOVING)", 'warning')
            shutil.rmtree(depth_dir)
            continue
        depth_counts.append(d)
        depth_counts_by_cam[cam_dir.name] = d
    # If any camera has depth, enforce all depth-bearing cameras have equal depth count, and equals common color
    if depth_counts:
        common_depth = depth_counts[0]
        if not all(_ == common_depth for _ in depth_counts) or common_depth != common_color:
            log(f"[check_all_cams_have_equal_number_of_frames] Depth frame mismatch: {str(depth_counts_by_cam)}", 'error')
            return False

    return True
