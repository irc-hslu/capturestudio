from pathlib import Path
from typing import Optional, Literal, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils.misc import PathUtils


@torch.no_grad()
def align_depth_to_color_for_cam(depth_dir: str, out_dir: str, parameters_dir: str, color_size_hw: Tuple[int, int], start_offset: int = 0, total_frames: int = -1, depth_format: Literal['npy', 'png'] = 'png', force: bool = False, celery_app=None):
    """
    Aligns color frames to depth frames for a given session.

    Parameters
    ----------
    depth_dir : str
        Path to the directory containing raw depth frames.
    out_dir : str
        Path to the directory where aligned depth frames will be saved.
    parameters_dir : str
        Path to the directory containing camera parameters (intrinsics, extrinsics, distortions).
    start_offset : int
        Start offset for the frames to process
    total_frames : int
        Total number of frames to process. If -1, process all frames.
    depth_format : Literal['npy', 'png']
        Format of the depth frames to save. 'npy' for numpy arrays, 'png' for 16-bit PNG images.
    force : bool
        Whether to force re-generation of depth frames when they exist.
    celery_app : Optional[celery.Celery]
        The Celery app instance for task management, by default None. If provided, the task progress will be tracked and tqdm will be disabled.
    """
    depth_dir: Path = Path(depth_dir)
    out_dir: Path = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parameters_dir = Path(parameters_dir)
    # color_dir = depth_dir.parent / 'color'
    # color_size_hw = PathUtils.read_file(Path(str(next(iter(color_dir.glob('*.jpg')))))).shape[:2]

    # align depth frames to color frames
    depth_dir: Path = Path(depth_dir)
    depth_files_npy = sorted(depth_dir.glob('*.npy'), key=lambda x: int(x.stem))  # global indexing
    if len(depth_files_npy) == 0:
        assert depth_format != 'npy', "No .npy depth files found in the directory. Please provide a valid depth directory with .npy files."
        depth_files = sorted(depth_dir.glob(f'*.{depth_format}'), key=lambda x: int(x.stem))  # global indexing
    else:
        depth_files = depth_files_npy
    if total_frames < 0:
        total_frames = len(depth_files) + total_frames - start_offset + 1
    all_depth_files = depth_files[start_offset:start_offset + total_frames]
    dest_depth_files = [out_dir / f.with_suffix(f'.{depth_format}').name for f in all_depth_files]

    # ATTN: some of the existing files were corrupted, so we force checking of each file
    if all(f.exists() and depth_format == 'png' and PathUtils.verify_file(f) for f in dest_depth_files) and not force:
        return None

    # load parameters
    c_intri = np.load(parameters_dir / 'color_intri.npy')
    d_intri = np.load(parameters_dir / 'depth_intri.npy')
    c_dist = np.load(parameters_dir / 'color_dist.npy')
    d_dist = np.load(parameters_dir / 'depth_dist.npy')
    extri = np.load(parameters_dir / 'depth_extri2color.npy')
    first_depth = PathUtils.read_file(all_depth_files[0], png_type='depth')

    from .orbbec_d2c_cuda import (
        AlignImpl,
        OBCameraIntrinsic,
        OBCameraDistortion,
        OBExtrinsic,
        DistortionModel,
    )

    aligner = AlignImpl(device='cuda').initialize(
        depth_intri=OBCameraIntrinsic(
            width=first_depth.shape[-1],
            height=first_depth.shape[-2],
            fx=d_intri[0, 0].item(), fy=d_intri[1, 1].item(),
            cx=d_intri[0, 2].item(), cy=d_intri[1, 2].item()
        ),
        depth_dist=OBCameraDistortion(
            model=DistortionModel.OB_DISTORTION_BROWN_CONRADY_K6,
            k1=d_dist[0].item(), k2=d_dist[1].item(),
            p1=d_dist[2].item(), p2=d_dist[3].item(),
            k3=d_dist[4].item(), k4=d_dist[5].item(), k5=d_dist[6].item(), k6=d_dist[7].item()
        ),
        color_intri=OBCameraIntrinsic(
            width=color_size_hw[1],
            height=color_size_hw[0],
            fx=c_intri[0, 0].item(), fy=c_intri[1, 1].item(),
            cx=c_intri[0, 2].item(), cy=c_intri[1, 2].item()
        ),
        color_dist=OBCameraDistortion(
            model=DistortionModel.OB_DISTORTION_KANNALA_BRANDT4,
            k1=c_dist[0].item(), k2=c_dist[1].item(),
            p1=c_dist[2].item(), p2=c_dist[3].item(),
            k3=c_dist[4].item(), k4=c_dist[5].item(), k5=c_dist[6].item(), k6=c_dist[7].item()
        ),
        depth2color_extri=OBExtrinsic(
            rot=extri[:3, :3],
            trans=extri[:3, 3]
        ),
        depth_unit_mm=1.0,
        add_target_distortion=False
    )
    aligner.prepare("cuda")  # Upload/precompute coefficient LUTs once.
    for i, (depth_path, aligned_depth_path) in tqdm(enumerate(zip(all_depth_files, dest_depth_files)), desc=f'Aligning depth frames (cam: {depth_dir.parent.name})', disable=celery_app is not None):
        if aligned_depth_path.exists() and depth_format == 'png' and PathUtils.verify_file(aligned_depth_path) and not force:
            continue

        # Load depth frame
        depth = PathUtils.read_file(depth_path, png_type='depth')
        # check if already aligned (size matches color size)
        if depth.shape[0] != color_size_hw[0] or depth.shape[1] != color_size_hw[1]:
            # Align depth frame to color frame
            # aligned_depth = aligner(depth, method='quad')
            aligned_depth = aligner.D2C(
                torch.from_numpy(depth).to("cuda", non_blocking=True),
                return_torch=True,
                conservative_raster=True,
            ).cpu().numpy()
        else:
            aligned_depth = depth

        # Save aligned depth frame
        PathUtils.write_file(aligned_depth_path, aligned_depth, png_type='depth')

    del aligner
    torch.cuda.empty_cache()

    return True
