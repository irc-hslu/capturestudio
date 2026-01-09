from pathlib import Path

import pandas as pd

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