from pathlib import Path
from typing import Literal, Optional

from tasks import app, AutoRetryTask


@app.task(name='preprocessing.depth.align_depth_to_color', base=AutoRetryTask)
def align_depth_to_color(depth_dir: str, out_dir: str, parameters_dir: str, start_offset: int = 0, total_frames: int = -1, depth_format: Literal['npy', 'png'] = 'png'):
    """
    Generate filtered depth maps from raw depth maps.

    Parameters
    ----------
    depth_dir : Path
        The path to the raw depth maps.
    out_dir : Path
        The output directory for the filtered depth maps.
    parameters_dir : Path
        The path to the parameters directory containing the camera parameters.
    start_offset : int
        The starting frame index.
    total_frames : int, optional
        The total number of frames to process, by default -1 (process all frames).
    depth_format : Literal['npy', 'png'], optional
        The format of the depth maps to save. 'npy' for numpy arrays, 'png' for images. Default is 'png'.
    """
    from preprocessing.depth.d2c import align_depth_to_color_for_cam
    align_depth_to_color_for_cam(depth_dir, out_dir, parameters_dir, start_offset=start_offset, total_frames=total_frames, depth_format=depth_format)
    return True


@app.task(name="preprocessing.depth.filter_depth", base=AutoRetryTask)
def filter_depth(filter_type: Literal['bilateral_spatial', 'bilateral_temporal'], depth_dir: str, color_dir: str, mask_dir: str, out_dir: str, flow_dir: Optional[str] = None, start_offset: int = 0, total_frames: int = -1, depth_format: Literal['npy', 'png'] = 'png'):
    """
    Generate filtered depth maps from raw depth maps.

    Parameters
    ----------
    filter_type : Literal['bilateral_spatial', 'bilateral_temporal']
        The type of depth filtering to apply. 'bilateral_spatial' for spatial filtering,
        'bilateral_temporal' for additional temporal filtering using optical flow.
    depth_dir : Path
        The path to the raw depth maps AFTER alignment.
        Aligned depths (d2c) should be generated BEFOREHAND, as the paths will be updated, so that the aligned ones are used for filtering.
    color_dir : Path
        The path to the color images corresponding to the depth maps.
    mask_dir : Path
        The path to the segmentation masks corresponding to the color images.
    out_dir : Path
        The output directory for the filtered depth maps.
    flow_dir : Optional[str], optional
        The path to the optical flow directory for temporal filtering. Required if filter_type is 'bilateral_temporal'.
    start_offset : int
        The starting frame index.
    total_frames : int, optional
        The total number of frames to process, by default -1 (process all frames).
    depth_format : Literal['npy', 'png'], optional
        The format of the depth maps to save. 'npy' for numpy arrays, 'png' for images. Default is 'png'.
    """
    from preprocessing.depth.depth_filtering import generate_filtered_depth_maps
    generate_filtered_depth_maps(depth_dir, color_dir, mask_dir, out_dir, flow_dir, start_offset=start_offset, total_frames=total_frames, depth_format=depth_format, filter_type=filter_type)
    return True
