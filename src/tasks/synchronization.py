from pathlib import Path
from typing import Literal

from tasks import app, AutoRetryTask


@app.task(name="synchronization.synchronize_frames", base=AutoRetryTask)
def synchronize_frames(capturestudio_cache_root: str, excel_sheet: str = 'Apr_May 2025', excel_file_path: str = 'Capture Sessions.xls'):
    """
    Synchronize frames for a session based on the timestamps found in the frame filenames or external flash sync signal.

    Parameters
    ----------
    capturestudio_cache_root: str
        Path to the session folder containing camera directories, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".
    excel_sheet: str, optional
        Name of the Excel sheet containing the session data (default: 'Apr_May 2025').
    excel_file_path: str, optional
        Path to the Excel file containing the session data (default: 'Capture Sessions.xls').
    """
    from preprocessing.sync import synchronize_frames_for_session
    sync_successful = synchronize_frames_for_session(capturestudio_cache_root, excel_sheet, excel_file_path)
    if sync_successful:
        return None
    # raise an error to stop chain of tasks
    raise RuntimeError(f"Synchronization failed for session {Path(capturestudio_cache_root).name}.")


@app.task(name="synchronization.generate_multiview_video", base=AutoRetryTask)
def generate_multiview_video(capturestudio_cache_root: str, modality: Literal['color', 'depth'], which_depth: Literal['depth', 'depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal'] = 'depth_aligned', depth_format: Literal['png', 'npy'] = 'png'):
    """
    Generate multi-camera videos for a given capture session. This includes multiview color and depth videos.
    ATTN: This task assumes that the frames have already been synchronized.

    Parameters
    ----------
    capturestudio_cache_root: str
        Path to the session folder containing camera directories, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".
    modality: str, optional
        The modality to use for the multiview video. Options are 'color' or 'depth'. Default is 'color'.
    which_depth: str, optional
        The depth modality to use for the multiview video. Options are 'depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal'. Default is 'depth_aligned'.
    depth_format: str, optional
        The format of the depth images. Options are 'png' or 'npy'. Default is 'png'.
    """
    capturestudio_cache_root = Path(capturestudio_cache_root)
    if modality == 'color':
        subfolder = 'color'
        file_extension = 'jpg'
    else:
        subfolder = which_depth
        file_extension = depth_format
    from preprocessing.generate_video import generate_multiview_video
    generate_multiview_video(str(capturestudio_cache_root), subfolder, file_extension)
    return None
