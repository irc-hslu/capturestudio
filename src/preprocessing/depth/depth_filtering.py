from pathlib import Path
from typing import Optional, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from utils.misc import MAX_DEPTH_IN_MASK_MM, log, PathUtils, MIN_DEPTH, MAX_DEPTH, get_depth_filterer


class CannyEdgeErosion:
    CANNY_EDGE_DETECTOR = None

    def __init__(self, k_sobel: int = 3, canary_grad_max_density: float = 0.4, canary_grad_min_density: float = 0.05, canary_grad_kernel_size: int = 41):
        self.canary_grad_max_density = canary_grad_max_density
        self.canary_grad_min_density = canary_grad_min_density
        self.canary_grad_kernel_size = canary_grad_kernel_size
        if self.__class__.CANNY_EDGE_DETECTOR is None:
            with torch.no_grad():
                from preprocessing.depth.canny_edge_detection import CannyFilter
                self.__class__.CANNY_EDGE_DETECTOR = CannyFilter(k_sobel=k_sobel, return_grad_only=True)

    def remove_sparse_clusters(self, edges):
        """Remove sparse clusters of white pixels (perceived as holes)"""
        # Binarize edges
        binary_edges = (edges > 0.2).float()

        # Calculate local density of white pixels
        padding = self.canary_grad_kernel_size // 2
        canary_grad_kernel = torch.ones(1, 1, self.canary_grad_kernel_size, self.canary_grad_kernel_size)
        neighbor_count = F.conv2d(
            binary_edges,
            canary_grad_kernel.to(edges.device),
            padding=padding
        )

        # Normalize density to [0, 1]
        max_count = self.canary_grad_kernel_size ** 2
        density = neighbor_count / max_count

        # Remove sparse pixels (density < min_density)
        mask = (density >= self.canary_grad_min_density).logical_and_(density <= self.canary_grad_max_density).float()
        return binary_edges * mask

    def get_contour(self, inp: torch.Tensor):
        need_squeezing = False
        if inp.dim() == 3:
            inp = inp.unsqueeze(0)
            need_squeezing = True
        edges = self.__class__.CANNY_EDGE_DETECTOR.to(inp.device)(inp)  # [B, 1, H, W], gradient magnitude from Canny edge detector
        return edges.squeeze(0) if need_squeezing else edges

    @torch.no_grad()
    def __call__(self, depths: torch.Tensor, masks: torch.Tensor):
        depths_filtered = depths.clone()
        depths_norm = depths_filtered.div(depths_filtered.max(0, keepdim=True).values) * masks
        depth_contour = self.get_contour(depths_norm)
        depth_contour = self.remove_sparse_clusters(depth_contour)
        return depths * (1 - depth_contour)


class QuantileErosion:
    def __init__(self, quantile: float = 0.9999, contour_dilation_kernel_size: int = 5):
        self.quantile = quantile
        self.contour_dilation_kernel_size = contour_dilation_kernel_size
        self.canary_detector = CannyEdgeErosion(k_sobel=13)

    def __call__(self, depths: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        depths_filtered = depths.clone()
        # filter out mask contours
        mask_contour = (self.canary_detector.get_contour(masks.float()) > 0.2).float()

        # dilation of mask contour TODO UNCOMMENT NEXT TWO LINES
        if self.contour_dilation_kernel_size != 0:
            mask_contour = F.max_pool2d(mask_contour, kernel_size=self.contour_dilation_kernel_size, stride=1, padding=self.contour_dilation_kernel_size // 2)

        # save mask contour for visualization
        # torchvision.utils.save_image(mask_contour, 'mask_contour.png')

        depths_filtered = depths_filtered * (1.0 - mask_contour.float())

        # import torchvision
        # torchvision.utils.save_image(torch.cat([torch.cat((depths * masks).unbind(0), dim=-1), torch.cat((depths_filtered * masks).unbind(0), dim=-1)], dim=-2).cpu() / 5, 'depths_final0.png')

        # filter out depth discontinuities
        max_depth = MAX_DEPTH_IN_MASK_MM # if depths_filtered.max() > 100.0 else MAX_DEPTH_IN_MASK_MM / 1000.0  # if depth is in mm, else in meters; adjust max depth accordingly
        depths_filtered[(depths_filtered > MAX_DEPTH_IN_MASK_MM) & (masks > 0.0)] = 0.0
        depths_norm = depths_filtered * masks / max_depth
        quantile = depths_norm.flatten(1).quantile(self.quantile, dim=-1)
        depths_filtered[depths_norm > quantile[:, None, None, None]] = 0.0

        # torchvision.utils.save_image(torch.cat([torch.cat((depths * masks).unbind(0), dim=-1), torch.cat((depths_filtered * masks).unbind(0), dim=-1)], dim=-2).cpu() / 5, 'depths_final1.png')

        return depths_filtered


class BilateralFiltering:
    EROSION_FILTER = None

    def __init__(self, smoothing_alpha: float = 1.0, filter_radius: int = 7, depth_sigma: float = 15.0, rgb_sigma: float = 1.0, temporal_sigma: float = 5.0, filter_in_time: bool = False, iters: int = 1, erosion_filter: str = 'quantile', **erosion_kwargs):
        self.smoothing_alpha = smoothing_alpha
        self.filter_radius = filter_radius
        self.depth_sigma = depth_sigma
        self.rgb_sigma = rgb_sigma
        self.temporal_sigma = temporal_sigma
        self.filter_in_time = filter_in_time
        self.iters = iters
        self._last_frame = None
        self._prev_depth_frame = None
        self._prev_color_frame = None
        self._prev_mask_frame = None
        self._use_erosion_filter = erosion_filter is not False
        if self.__class__.EROSION_FILTER is None and self._use_erosion_filter:
            self.__class__.EROSION_FILTER = globals()[f'{erosion_filter.title()}Erosion'](**erosion_kwargs)

    def process_spatial(self, depths: torch.Tensor, colors: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        from preprocessing.depth.bilateral_filter_spatial_cuda import bilateral_filter_cuda_torch as bilateral_filter_cuda_torch_spatial
        return bilateral_filter_cuda_torch_spatial(
            depths, colors, masks,
            filter_radius=self.filter_radius,
            depth_sigma=self.depth_sigma,
            rgb_sigma=self.rgb_sigma
        )

    def process_temporal(self, depths: torch.Tensor, colors: torch.Tensor, masks: torch.Tensor, flows: Optional[torch.Tensor] = None, update_prev_depth: bool = True) -> torch.Tensor:
        from preprocessing.depth.bilateral_filter_temporal_cuda import bilateral_filter_cuda_torch as bilateral_filter_cuda_torch_temporal
        depths_temporally_filtered = []
        for i in range(len(depths)):
            if self._prev_depth_frame is None or flows is None:
                self._prev_depth_frame = self.process_spatial(depths[[0]], colors[[0]], masks[[0]])
                self._prev_color_frame = colors[[0]].detach()
                self._prev_mask_frame = masks[[0]].detach()
                depths_temporally_filtered.append(self._prev_depth_frame)
                continue
            depths_temporally_filtered.append(
                bilateral_filter_cuda_torch_temporal(
                    depth_map=depths[[i]],
                    prev_color_image=self._prev_color_frame,
                    color_image=colors[[i]],
                    prev_mask=self._prev_mask_frame,
                    mask=masks[[i]],
                    prev_depth_map=self._prev_depth_frame,
                    optical_flow=(flows[0][[i]], flows[1][[i]]) if isinstance(flows, tuple) else flows[[i]],
                    filter_radius=self.filter_radius,
                    depth_sigma=self.depth_sigma,
                    rgb_sigma=self.rgb_sigma,
                    temporal_sigma=self.temporal_sigma
                )
            )
            if update_prev_depth:
                self._prev_depth_frame = depths_temporally_filtered[-1].detach().clone()
                self._prev_color_frame = colors[[i]].detach()
                self._prev_mask_frame = masks[[i]].detach()
        return torch.concat(depths_temporally_filtered, dim=0)

    # noinspection PyMethodOverriding
    def process(self, depths: torch.Tensor, colors: torch.Tensor, masks: torch.Tensor, flows: Optional[torch.Tensor] = None, intrinsics=None) -> Union[torch.Tensor, np.ndarray]:
        if self._use_erosion_filter:
            depths_eroded = self.__class__.EROSION_FILTER(depths, masks) if self.__class__.EROSION_FILTER is not None else depths
        else:
            depths_eroded = depths
        depths_frame_filtered = depths_eroded
        for i in range(self.iters):
            if self.filter_in_time:
                depths_frame_filtered = self.process_temporal(depths_frame_filtered, colors, masks, flows, update_prev_depth=i == self.iters - 1)
            else:
                depths_frame_filtered = self.process_spatial(depths_frame_filtered, colors, masks)
        for i in range(len(depths_frame_filtered)):
            if self._last_frame is None:
                self._last_frame = depths_frame_filtered[i]
                continue
            depths_frame_filtered[i] = self.smoothing_alpha * depths_frame_filtered[i] + (1.0 - self.smoothing_alpha) * self._last_frame
            self._last_frame = depths_frame_filtered[i]
        return depths_frame_filtered

    def reset(self) -> None:
        self._last_frame = None


@torch.no_grad()
def generate_filtered_depth_maps(depth_dir: str, color_dir: str, mask_dir: str, out_dir: str, flow_dir: Optional[str] = None, start_offset: int = 0, total_frames: int = -1, depth_format: Literal['npy', 'png'] = 'png', filter_type: str = 'bilateral_spatial', force=False, celery_app=None):
    """
    Generate filtered depth maps from raw depth maps.

    Parameters
    ----------
    depth_dir : Path
        The path to the raw depth maps AFTER syncing.
        Aligned depths (d2c) should be generated BEFOREHAND, as the paths will be updated, so that the aligned ones are used for filtering.
    color_dir : Path
        The path to the color images corresponding to the depth maps.
    mask_dir : Path
        The path to the segmentation masks corresponding to the color images.
    out_dir : Path
        The output directory for the filtered depth maps.
    flow_dir : Path, optional
        The path to the optical flow files corresponding to the color images. When filter is 'bilateral_temporal', the backward flow is used to filter the depth maps, and thus the `flow_dir` should not be None.
    start_offset : int
        The starting frame index.
    total_frames : int, optional
        The total number of frames to process, by default -1 (process all frames).
    depth_format : Literal['npy', 'png'], optional
        The format of the depth maps to save. 'npy' for numpy arrays, 'png' for images. Default is 'png'.
    filter_type : str, optional
        The type of filter to apply, by default 'bilateral_spatial'.
    force: bool, optional
        If True, it will force re-generation of filtered depth map in case they already exist.
    celery_app : Optional, optional
        The Celery app instance for task management, by default None. If provided, the task progress will be tracked and tqdm will be disabled.
    """
    depth_dir = Path(depth_dir)
    color_dir = Path(color_dir)
    mask_dir = Path(mask_dir)
    flow_dir = Path(flow_dir) if flow_dir is not None else None
    out_dir = Path(out_dir)

    # Get all depth files
    depth_files_npy = sorted(depth_dir.glob('*.npy'), key=lambda x: int(x.stem))  # global indexing
    if len(depth_files_npy) == 0:
        assert depth_format != 'npy', "No .npy depth files found in the directory. Please provide a valid depth directory with .npy files."
        depth_files = sorted(depth_dir.glob(f'*.{depth_format}'), key=lambda x: int(x.stem))  # global indexing
    else:
        depth_files = depth_files_npy
    color_dir = Path(color_dir)
    color_files = sorted(color_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    if total_frames == -1:
        total_frames = len(depth_files) - start_offset
    depth_files = depth_files[start_offset:start_offset + total_frames]
    color_files = color_files[start_offset:start_offset + total_frames]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # force = False
    out_depth_files = [out_dir / df.with_suffix(f'.{depth_format}').name for df in depth_files]
    if all([out_file.exists() and depth_format == 'png' and PathUtils.verify_file(out_file) for out_file in out_depth_files]) and not force:
        log(f"\tFiltered depth maps already exist for {depth_dir}. Skipping filtering.", 'debug')
        return True

    assert flow_dir is not None or filter_type == 'bilateral_spatial', "When using 'bilateral_temporal' filter, the flow_dir must be provided."

    depth_filterer = get_depth_filterer(filter_type)
    depth_filterer.reset()

    log(f'Filtering depth maps from dir {depth_dir} with type: {filter_type}', 'info')
    for i in tqdm(range(start_offset, start_offset + total_frames), total=total_frames, desc=f'Filtering depth maps', disable=celery_app is not None):
        depth_file = depth_files[i]
        if 'depth_aligned' not in str(depth_file) and Path(str(depth_file).replace('depth', 'depth_aligned')).exists():
            depth_file = Path(str(depth_file).replace('depth', 'depth_aligned'))
        out_file = out_dir / f'{depth_file.stem}.{depth_format}'
        if out_file.exists() and depth_format == 'png' and PathUtils.verify_file(out_file) and not force:
            continue

        color_file = color_files[i]
        try:
            img_torch = torchvision.io.decode_image(str(color_file)).float().div_(255.0).cuda()
        except RuntimeError as e:
            log(f'\t[{color_file.parent.parent.name}/{color_file.parent.name}/{color_file.name}] {str(e)}', 'error')
            img_torch = torchvision.io.decode_image(str(color_files[max(0, i - 1)])).float().div_(255.0).cuda()
        try:
            # Load depth map and guide image/mask
            depth = PathUtils.read_file(depth_file, png_type='depth')
        except (RuntimeError, ValueError) as e:
            raise e

        depth = np.where((depth > MIN_DEPTH) & (depth < MAX_DEPTH), depth, 0)
        depth_torch = torch.from_numpy(depth).unsqueeze(0).float().cuda()
        try:
            segmentation_file = mask_dir / f'{color_file.stem}.jpg'
            if segmentation_file is not None:
                mask_torch = (torchvision.io.decode_image(str(segmentation_file), mode=torchvision.io.ImageReadMode.GRAY).float().div_(255.0) > 0.5).float().cuda()
            else:
                mask_torch = torch.ones_like(depth_torch)
        except RuntimeError as e:
            log(f'\t[{color_file.parent.parent.name}/{color_file.parent.name}/{color_file.name}] {str(e)}', 'error')
            mask_torch = torch.ones_like(depth_torch)
        if 'temporal' in filter_type:
            # load optical flow
            flow_bwd_file = flow_dir / f'{color_file.stem}.png'
            assert i == 0 or flow_bwd_file.exists(), f"Flow file {flow_bwd_file} does not exist. Please generate optical flow first."
            flow = None
            try:
                flow_bwd_torch = PathUtils.read_file(flow_bwd_file, png_type='flow').cuda()[None]
                flow = flow_bwd_torch
            except (RuntimeError, ValueError) as e:
                log(f'\t\t[{flow_bwd_file.parent.parent.name}/{flow_bwd_file.parent.name}/{flow_bwd_file.name}] {str(e)}', 'error')
        else:
            flow = None

        # Apply filtering (bilateral temporal or spatiotemporal)
        depth_filtered_torch = depth_filterer.process(
            depth_torch[None],
            img_torch[None],
            mask_torch[None],
            flows=flow
        )[0]
        depth_filtered = depth_filtered_torch.detach().squeeze().cpu().numpy().astype(np.uint16)
        if depth_filtered.max().item() < 1.0:
            log(f'Depth is empty {out_file}', 'warning')
        # Save the filtered depth map
        PathUtils.write_file(out_file, depth_filtered, png_type='depth')

    return True
