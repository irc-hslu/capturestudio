import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, List, Dict, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.misc import log, PathUtils, get_segmentor, get_of_estimator, get_detector


def detect(image_path: Path, target_classes: Sequence[str] = ('person',), conf_threshold: float = 0.5, rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None, unrotate_output: bool = False) -> List[Dict[str, Union[list, float, str, int]]]:
    """
    Detect objects in an image.

    Parameters
    ----------
    image_path: Path
        Path to the input image
    target_classes: Sequence[str]
        List of target class names to detect (e.g. 'person', 'car', etc
    conf_threshold: float
        Confidence threshold for detections (0-1)
    rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180
        If specified, rotate the input image by the given angle for detection. The outputs will be rotated back to the original orientation.
            - '90_CLOCKWISE': Rotate 90 degrees clockwise
            - '90_COUNTERCLOCKWISE': Rotate 90 degrees counterclockwise
            - '180': Rotate 180 degrees
    unrotate_output: bool
        If True and rotate is specified, the output bounding boxes will be transformed back to the original orientation.

    Returns
    -------
    list
        List of detection dictionaries with keys:
        - 'bbox': [x1, y1, x2, y2] in pixel coordinates
        - 'confidence': Detection confidence (0-1)
        - 'class_name': Detected class name
        - 'class_id': Class ID
    """
    with torch.no_grad():
        detector, detector_class_names = get_detector()
        device = detector.device
        # Convert class names to IDs
        class_ids = []
        for cls in target_classes:
            if cls.lower() in detector_class_names.values():
                class_ids.extend([k for k, v in detector_class_names.items() if v == cls.lower()])
            else:
                print(f"Warning: Class '{cls}' not found in model classes")
        if not class_ids:
            raise ValueError("None of the target classes are valid for this model")
        # Load image and store original shape
        img = cv2.imread(str(image_path))
        if rotate:
            img = cv2.rotate(img, getattr(cv2, f'ROTATE_{rotate}'))
        new_h, new_w = img.shape[:2]
        # Run inference
        results = detector.predict(
            img,
            classes=class_ids,
            conf=conf_threshold,
            device=device
        )
        # Parse results
        detections = []
        for result in results:
            for box in result.boxes:
                bbox_xyxy = box.xyxy.cpu().numpy()[0]
                bbox_xyxy_rotated = bbox_xyxy.copy()
                if rotate and unrotate_output:
                    x1, y1, x2, y2 = bbox_xyxy
                    corners = np.array([
                        [x1, y1],  # A (tl)
                        [x2, y1],  # B (tr)
                        [x2, y2],  # C (br)
                        [x1, y2],  # D (bl)
                    ])
                    if rotate == '90_CLOCKWISE':
                        transformed = np.array([[y, new_w - x] for (x, y) in corners])
                    elif rotate == '90_COUNTERCLOCKWISE':
                        transformed = np.array([[new_h - y, x] for (x, y) in corners])
                    elif rotate == '180':
                        transformed = np.array([[new_w - x, new_h - y] for (x, y) in corners])
                    # Compute new bbox from transformed corners
                    x_coords, y_coords = transformed[:, 0], transformed[:, 1]
                    bbox_xyxy = [float(x_coords.min()), float(y_coords.min()),
                                 float(x_coords.max()), float(y_coords.max())]
                detections.append({
                                      'bbox': [float(c) for c in bbox_xyxy],
                                      'confidence': float(box.conf.item()),
                                      'class_name': detector_class_names[int(box.cls.item())],
                                      'class_id': int(box.cls.item())
                                  } | {f'bbox_rotated_{rotate}': [float(c) for c in bbox_xyxy_rotated]} if rotate else {})
    return detections


def segment(video_path: Path, first_frame_detections: List[Dict[str, Union[list, float, str, int]]], out_paths: List[Path], rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None, unrotate_output: bool = True) -> bool:
    """
    Segment objects in a video based on detections.

    Args:
        video_path: Path to input video
        first_frame_detections: List of detection dictionaries (see detect()). If rotate, then the key 'bbox_rotated_{rotate}' must be present and will be used instead of 'bbox'.
        out_paths: List of output paths for segmentation masks
        rotate: Optional rotation to apply to the video before segmentation
            - '90_CLOCKWISE': Rotate 90 degrees clockwise
            - '90_COUNTERCLOCKWISE': Rotate 90 degrees counterclockwise
            - '180': Rotate 180 degrees
        unrotate_output: If True, the output masks will be transformed back to the original orientation
    """
    # If rotate check if the video is already rotated (stem should end with _rotated_{rotate.lower()})
    if rotate and not video_path.stem.endswith(f'_rotated_{rotate.lower()}'):
        # Rotate video and save to the same dir with _rotated_{rotate.lower()} suffix
        rotated_video_path = video_path.parent / f'{video_path.stem}_rotated_{rotate.lower()}{video_path.suffix}'

        from moviepy.editor import VideoFileClip, vfx
        clip = VideoFileClip(str(video_path))
        if rotate == '90_CLOCKWISE':
            rotated_clip = clip.fx(vfx.rotate, -90)
        elif rotate == '90_COUNTERCLOCKWISE':
            rotated_clip = clip.fx(vfx.rotate, 90)
        elif rotate == '180':
            rotated_clip = clip.fx(vfx.rotate, 180)
        else:
            raise ValueError(f"Unsupported rotation: {rotate}")
        rotated_clip.write_videofile(str(rotated_video_path), codec='libx264', audio=False, verbose=False, logger=None)
        video_path = rotated_video_path

    segmentor = get_segmentor()
    # get embeddings
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        state = segmentor.init_state(str(video_path), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
        # add all bboxes
        single_subject = False
        found_bbox = False
        bbox_thresh = 100
        frame_idx = 0
        tracked_obj_ids = set()
        for obj_id, frame_bbox_data in enumerate(first_frame_detections):
            frame_bbox_i = frame_bbox_data['bbox' if not rotate else f'bbox_rotated_{rotate}']
            conf_i = frame_bbox_data['confidence']
            class_name_i = frame_bbox_data['class_name']
            if conf_i < 0.4:
                log(f'Skipping bbox with low confidence: {frame_bbox_i} | {conf_i}', 'warning')
                continue
            frame_bbox_i = np.array(frame_bbox_i)[:4]
            if ((frame_bbox_i[2] - frame_bbox_i[0]) < bbox_thresh) or ((frame_bbox_i[3] - frame_bbox_i[1]) < bbox_thresh):
                log(f'Skipping bbox with small size: {frame_bbox_i} | {(frame_bbox_i[2] - frame_bbox_i[0], frame_bbox_i[3] - frame_bbox_i[1])}', 'warning')
                continue
            log(f'\t[frame={frame_idx:04d}][obj={obj_id} ({class_name_i})] adding bbox: {frame_bbox_i.astype(int).tolist()}', 'debug')
            _, obj_ids, obj_masks = segmentor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id + 1, box=frame_bbox_i)
            if len(obj_ids) > 1:
                # selected_masks, selected_indices = GenerateSegmentationMaskTask.visualize_and_select_multiple_masks(np.zeros(*obj_masks[0].shape, 3), obj_masks)
                # obj_ids = [obj_ids[i] for i in selected_indices]
                log(f"\t[segment] Found {len(obj_ids)} object IDs.", 'warning')
            tracked_obj_ids.update(obj_ids)
            found_bbox = True
            if single_subject:
                break

        if not found_bbox:
            log(f'No bbox found. Skipping video: \"{video_path}\"', 'error')
            return False

        # Propagate masks and merge per frame
        for frame_idx, object_ids, masks in segmentor.propagate_in_video(state):
            merged_mask = None
            for obj_id, mask in zip(object_ids, masks):
                if obj_id not in tracked_obj_ids:
                    continue
                binary_mask = (mask.detach().cpu().squeeze().numpy() > 0.5)
                if merged_mask is None:
                    merged_mask = binary_mask
                else:
                    merged_mask |= binary_mask  # Logical OR to merge

            if merged_mask is not None:
                merged_mask = (merged_mask * 255).astype(np.uint8)

                if rotate and unrotate_output:
                    if rotate == '90_CLOCKWISE':
                        merged_mask = cv2.rotate(merged_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotate == '90_COUNTERCLOCKWISE':
                        merged_mask = cv2.rotate(merged_mask, cv2.ROTATE_90_CLOCKWISE)
                    elif rotate == '180':
                        merged_mask = cv2.rotate(merged_mask, cv2.ROTATE_180)

                PathUtils.write_file(out_paths[frame_idx], merged_mask)

        del state
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return True


def estimate_optical_flow(cam_color_dir: Path, out_dir_bwd: Path, out_dir_fwd: Optional[Path] = None, start_offset: int = 0, total_frames: int = -1, which: Literal['fwd', 'bwd', 'fwd+bwd'] = 'bwd', rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None):
    """
    Estimate optical flow for a sequence of images in a directory.

    Parameters:
    -----------
    cam_color_dir : Path
        Path to the camera's color directory containing the color frames.
    out_dir_bwd : Path
        Path to the output directory where the backward optical flows will be saved (i.e. from t-1 --> t).
    out_dir_fwd : Optional[Path]
        Path to the output directory where the forward optical flows will be saved (i.e. from t+1 --> t).
        If None, only backward optical flows will be generated.
    start_offset : int
        Starting frame offset (default: 0).
    total_frames : int
        Total number of frames to process. If -1, process all frames from start_offset to the end.
        If negative (e.g. -2), process all frames except the last abs(total_frames) - 1 frames.
    which : Literal['fwd', 'bwd', 'fwd+bwd']
        Which optical flows to compute: 'fwd' for forward, 'bwd' for backward, 'fwd+bwd' for both (default: 'bwd').
    rotate : Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']]
        Optional rotation to apply to the images before processing.
            - '90_CLOCKWISE': Rotate 90 degrees clockwise
            - '90_COUNTERCLOCKWISE': Rotate 90 degrees counterclockwise
            - '180': Rotate 180 degrees
        If None, no rotation is applied.

    Returns:
    --------
    bool
        True if optical flow estimation was successful, False otherwise.
    """
    def rotate_tensor_img(img, rotate):
        img_np = img.permute(1, 2, 0).byte().numpy()
        if rotate == '90_CLOCKWISE':
            img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == '90_COUNTERCLOCKWISE':
            img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == '180':
            img_np = cv2.rotate(img_np, cv2.ROTATE_180)
        return torch.from_numpy(img_np).permute(2, 0, 1).float()

    def unrotate_flow(flow: torch.Tensor, rotate) -> torch.Tensor:
        """
        flow: (2, H, W) tensor, (u, v) = (x, y) in pixels
        rotate: one of {'90_CLOCKWISE','90_COUNTERCLOCKWISE','180'} or None
        """
        if rotate is None:
            return flow

        # spatial rotation helper: rot90 k times CCW across H/W dims (1, 2)
        def rot_ccw(x):
            return torch.rot90(x, 1, (-2, -1))

        def rot_cw(x):
            return torch.rot90(x, -1, (-2, -1))

        def rot_180(x):
            return torch.rot90(x, 2, (-2, -1))

        u, v = flow[0], flow[1]
        if rotate == '90_CLOCKWISE':
            # u = rotate_ccw(v'), v = -rotate_ccw(u')
            u_new = rot_ccw(v)
            v_new = -rot_ccw(u)
        elif rotate == '90_COUNTERCLOCKWISE':
            # u = -rotate_cw(v'), v = rotate_cw(u')
            u_new = -rot_cw(v)
            v_new = rot_cw(u)
        elif rotate == '180':
            # u = -rotate_180(u'), v = -rotate_180(v')
            u_new = -rot_180(u)
            v_new = -rot_180(v)
        else:
            return flow
        return torch.stack((u_new, v_new), dim=0)

    color_files = sorted(cam_color_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    if total_frames == -1:
        total_frames = len(color_files) - start_offset
    elif total_frames < 0:
        total_frames = len(color_files) + total_frames - start_offset + 1

    all_files_exist = True
    if 'fwd' in which:
        out_dir_fwd.mkdir(parents=True, exist_ok=True)
        of_fwd_file_paths = [(out_dir_fwd / cf.name).with_suffix('.png') for cf in color_files[start_offset + 1:start_offset + total_frames]]
        all_files_exist = all_files_exist and all(p.exists() and PathUtils.verify_file(p) for p in of_fwd_file_paths)
    if 'bwd' in which:
        out_dir_bwd.mkdir(parents=True, exist_ok=True)
        of_bwd_file_paths = [(out_dir_bwd / cf.name).with_suffix('.png') for cf in color_files[start_offset + 1:start_offset + total_frames]]
        all_files_exist = all_files_exist and all(p.exists() and PathUtils.verify_file(p) for p in of_bwd_file_paths)
    if all_files_exist:
        log(f'\tOptical flows already exist for "{cam_color_dir.parent.name}/{cam_color_dir.name}". Skipping OF estimation.', 'debug')
        return True

    of_estimator, of_estimator_padder = get_of_estimator()
    input_padder = None
    paths_iterator = zip(of_fwd_file_paths, of_bwd_file_paths) if 'fwd' in which and 'bwd' in which else (of_fwd_file_paths if 'fwd' in which else of_bwd_file_paths)

    for idx, out_path in tqdm(enumerate(paths_iterator), total=total_frames - 1, desc=f'Generating optical flows for "{cam_color_dir.parent.name}/{cam_color_dir.name}"', disable=False):
        if (isinstance(out_path, Path) and out_path.exists() and PathUtils.verify_file(out_path)) or (isinstance(out_path, tuple) and all(p.exists() and PathUtils.verify_file(p) for p in out_path)):
            continue

        prev_img = torch.from_numpy(cv2.cvtColor(cv2.imread(str(color_files[max(0, start_offset + idx)])), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        ref_img = torch.from_numpy(cv2.cvtColor(cv2.imread(str(color_files[start_offset + idx + 1])), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        try:
            next_img = torch.from_numpy(cv2.cvtColor(cv2.imread(str(color_files[min(len(color_files) - 1, start_offset + idx + 2)])), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        except cv2.error:
            dest_path = color_files[start_offset + idx + 2]
            src_path = color_files[start_offset + idx + 1]
            log(f'Failed to read next image: {dest_path.parent.name}/{dest_path.name}. Replacing it with the previous image: {src_path.parent.name}/{src_path.name}.', 'warning')
            shutil.copy(src_path, dest_path)
            next_img = torch.from_numpy(cv2.cvtColor(cv2.imread(str(dest_path)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

        if rotate:
            prev_img = rotate_tensor_img(prev_img, rotate)
            ref_img = rotate_tensor_img(ref_img, rotate)
            next_img = rotate_tensor_img(next_img, rotate)

        input_imgs = torch.stack([prev_img, ref_img, next_img], dim=0)[None].cuda()
        original_size = (ref_img.shape[-2], ref_img.shape[-1])
        input_imgs = F.interpolate(input_imgs.view(-1, 3, original_size[0], original_size[1]), scale_factor=0.25, mode='bilinear', align_corners=False).view(-1, 3, 3, original_size[0] // 4, original_size[1] // 4)

        if input_padder is None:
            input_padder = of_estimator_padder(input_imgs.shape, mode='sintel')
        input_imgs = input_padder.pad(input_imgs)
        with torch.no_grad():
            flow_pred, _ = of_estimator(input_imgs, {})
        flow_pred = input_padder.unpad(flow_pred[0]).cpu()

        H_orig, W_orig = original_size
        H_net, W_net = flow_pred.shape[-2:]
        flow_pred = F.interpolate(
            flow_pred,
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        )
        scale_x = W_orig / W_net
        scale_y = H_orig / H_net
        flow_pred[..., 0, :, :] *= scale_x
        flow_pred[..., 1, :, :] *= scale_y
        flow_fwd = unrotate_flow(flow_pred[0, :, :], rotate)
        flow_bwd = unrotate_flow(flow_pred[1, :, :], rotate)

        if 'fwd' in which:
            PathUtils.write_file(out_path[0] if isinstance(out_path, tuple) else out_path, flow_fwd, png_type='flow')
        if 'bwd' in which:
            PathUtils.write_file(out_path[1] if isinstance(out_path, tuple) else out_path, flow_bwd, png_type='flow')

        del flow_fwd, flow_bwd, flow_pred

    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return True
