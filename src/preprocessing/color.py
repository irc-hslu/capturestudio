import itertools
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, List, Dict, Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm

from utils.flow import FlowUtils
from utils.misc import log, PathUtils, get_segmentor, get_detector, get_of_estimator


def detect(image_path: Path,
           target_classes: Sequence[str] = ('person', 'guitar', 'guitar band', 'drums'),
           conf_threshold: float = 0.5,
           rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None,
           unrotate_output: bool = False,
           which_detector: Literal['yolo', 'sam3'] = 'sam3') -> List[Dict[str, Union[list, float, str, int]]]:
    """
    Detect objects in an image.

    Parameters
    ----------
    image_path: Path
        The path to the input image.
    target_classes: Sequence[str]
        List of target class names to detect (e.g. 'person', 'car', etc.).
    conf_threshold: float
        Confidence threshold for detections (0-1)
    rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180
        If specified, rotate the input image by the given angle for detection. The outputs will be rotated back to the original orientation.
            - '90_CLOCKWISE': Rotate 90 degrees clockwise
            - '90_COUNTERCLOCKWISE': Rotate 90 degrees counterclockwise
            - '180': Rotate 180 degrees
    unrotate_output: bool
        If True and rotate is specified, the output bounding boxes will be transformed back to the original orientation.
    which_detector: Literal['yolo', 'sam3']
        The model to use for object detection. One of `yolo` (YOLO v11), `sam3` (Segment Anything Model v3). Default is `sam3`.

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
        from ultralytics import YOLO
        from sam3.model.sam3_image import Sam3Image
        detector: Union[YOLO, Sam3Image]
        detector, detector_class_names = get_detector(which_detector)
        device = detector.device

        # Convert class names to IDs (works for YOLO and for SAM3 using the detector_class_names catalog)
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
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if rotate:
            img = cv2.rotate(img, getattr(cv2, f'ROTATE_{rotate}'))
        new_h, new_w = img.shape[:2]

        if which_detector == 'yolo':
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
        elif which_detector == 'sam3':
            # --- SAM3 text-prompted detection on a single image ---
            from sam3.model.sam3_image_processor import Sam3Processor
            processor: Sam3Processor = getattr(detector, "_processor", None)
            if processor is None:
                raise AttributeError("SAM3 detector is missing `_processor`. Ensure get_detector('sam3') attaches Sam3Processor to the model.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)

            # Build id->name and name->id maps for requested classes
            requested = [detector_class_names[i] for i in class_ids]
            catalog_map = {name: i for i, name in detector_class_names.items()}
            detections = []
            with torch.inference_mode():
                state = processor.set_image(pil)

                for cls in requested:
                    out = processor.set_text_prompt(state=state, prompt=cls)
                    boxes = out.get("boxes", [])
                    scores = out.get("scores", [])

                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.detach().cpu().numpy()
                    else:
                        boxes = np.asarray(boxes) if len(boxes) else np.zeros((0, 4), dtype=np.float32)
                    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes is not None else np.zeros((0, 4), dtype=np.float32)

                    if isinstance(scores, torch.Tensor):
                        scores = scores.detach().cpu().numpy().astype(np.float32)
                    else:
                        scores = np.asarray(scores, dtype=np.float32) if len(scores) else np.zeros((0,), dtype=np.float32)
                    scores = np.asarray(scores, dtype=np.float32).reshape(-1, ) if scores is not None else np.zeros((0,), dtype=np.float32)

                    if boxes.shape[0] == 0 or scores.shape[0] == 0:
                        continue

                    # Filter by confidence
                    keep_idx = np.where(scores >= float(conf_threshold))[0]
                    if keep_idx.size == 0:
                        continue
                    boxes = boxes[keep_idx]
                    scores = scores[keep_idx]

                    # Clamp and validate
                    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(new_w - 1))
                    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(new_w - 1))
                    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(new_h - 1))
                    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(new_h - 1))
                    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    boxes = boxes[valid]
                    scores = scores[valid]
                    if boxes.shape[0] == 0:
                        continue

                    # Per-class NMS
                    keep = nms(torch.as_tensor(boxes), torch.as_tensor(scores), iou_threshold=0.5).detach().cpu().numpy().tolist()
                    boxes = boxes[keep]
                    scores = scores[keep]
                    for bbox_xyxy, score in zip(boxes, scores):
                        bbox_xyxy = bbox_xyxy.astype(np.float32)
                        bbox_xyxy_rotated = bbox_xyxy.copy()

                        bbox_out = bbox_xyxy.tolist()
                        if rotate and unrotate_output:
                            x1, y1, x2, y2 = bbox_xyxy.tolist()
                            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                            if rotate == '90_CLOCKWISE':
                                transformed = np.array([[y, new_w - x] for (x, y) in corners], dtype=np.float32)
                            elif rotate == '90_COUNTERCLOCKWISE':
                                transformed = np.array([[new_h - y, x] for (x, y) in corners], dtype=np.float32)
                            elif rotate == '180':
                                transformed = np.array([[new_w - x, new_h - y] for (x, y) in corners], dtype=np.float32)
                            else:
                                transformed = corners
                            x_coords, y_coords = transformed[:, 0], transformed[:, 1]
                            bbox_out = [float(x_coords.min()), float(y_coords.min()), float(x_coords.max()), float(y_coords.max())]

                        detections.append({
                                              "bbox": [float(c) for c in bbox_out],
                                              "confidence": float(score),
                                              "class_name": cls,
                                              "class_id": int(catalog_map.get(cls, 0)),
                                          } | ({f"bbox_rotated_{rotate}": [float(c) for c in bbox_xyxy_rotated.tolist()]} if rotate else {}))

                # cleanup prompts for this image
                processor.reset_all_prompts(state)
        else:
            raise ValueError(f"Unsupported detector type: {which_detector}")

    return detections


def segment(video_path: Path,
            first_frame_detections: List[Dict[str, Union[list, float, str, int]]],
            out_paths: List[Path],
            rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None,
            unrotate_output: bool = True,
            which_segmentor: Literal['sam2', 'sam3'] = 'sam2',
            first_frame_global_idx: int = 0) -> bool:
    """
    Segment objects in a video based on detections.
    ATTN: Assumes that detection data (incl. bboxes and pos/neg points) have been generated beforehand.

    Parameters
    ----------
    video_path: Path
        Path to input video.
    first_frame_detections: List[Dict[str, Union[list, float, str, int]]]
        List of detection dictionaries (see detect()). If rotate, then the key 'bbox_rotated_{rotate}' must be present and will be used instead of 'bbox'.
    out_paths: List[Path]
        List of output paths for segmentation masks.
    rotate: Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']], optional
        Optional rotation to apply to the video before segmentation:
        - '90_CLOCKWISE': Rotate 90 degrees clockwise
        - '90_COUNTERCLOCKWISE': Rotate 90 degrees counterclockwise
        - '180': Rotate 180 degrees
    unrotate_output: bool
        If True, the output masks will be transformed back to the original orientation.
    which_segmentor: Literal['sam2', 'sam3']
        Which model to use for segmentation. One of `sam2` (Segment Anything Model v2), `sam3` (Segment Anything Model v3). Default is `sam3`.
    first_frame_global_idx: int, optional
        If != 0, all frame_idx will be recalculated before fed to SAM.
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

    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    segmentor: Union[SAM2VideoPredictor, Sam3VideoPredictor] = get_segmentor(which=which_segmentor)

    # get embeddings
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        if which_segmentor == 'sam2':
            state = segmentor.init_state(str(video_path), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
        else:
            session_id = segmentor.start_session(str(video_path))['session_id']
            # noinspection PyProtectedMember
            state = segmentor._ALL_INFERENCE_STATES[session_id]['state']
        H, W = state.get("orig_height", None), state.get("orig_width", None)
        if H is None or W is None:
            H, W = state['video_height'], state['video_width']
        # add all bboxes
        single_subject = False
        found_bbox = False
        bbox_thresh = 7
        frame_idx = 0
        tracked_obj_ids = set()
        for obj_id, frame_bbox_data in enumerate(first_frame_detections):
            bbox_key = 'bbox' if not rotate else f'bbox_rotated_{rotate}'
            point_key = 'points' if not rotate else f'points_rotated_{rotate}'
            if bbox_key in frame_bbox_data:
                frame_bbox_i = frame_bbox_data[bbox_key]
                conf_i = frame_bbox_data['confidence']
                class_name_i = frame_bbox_data['class_name']
                if conf_i < 0.4:
                    log(f'Skipping bbox with low confidence: {frame_bbox_i} | {conf_i}', 'warning')
                    continue
                frame_bbox_i = np.array(frame_bbox_i)[:4]
                if ((frame_bbox_i[2] - frame_bbox_i[0]) < bbox_thresh) or ((frame_bbox_i[3] - frame_bbox_i[1]) < bbox_thresh):
                    log(f'Skipping bbox with small size: {frame_bbox_i} | {(frame_bbox_i[2] - frame_bbox_i[0], frame_bbox_i[3] - frame_bbox_i[1])}', 'warning')
                    continue
                if 'frame_idx' in frame_bbox_data:
                    frame_idx_this = int(frame_bbox_data['frame_idx'])
                else:
                    frame_idx_this = frame_idx

                if which_segmentor == 'sam2':
                    _, obj_ids, obj_masks = segmentor.add_new_points_or_box(state, frame_idx=frame_idx_this + first_frame_global_idx, obj_id=obj_id + 1, box=frame_bbox_i)
                elif which_segmentor == 'sam3':
                    # add bbox prompts: create labels that are positive
                    x1, y1, x2, y2 = map(float, frame_bbox_i.tolist())
                    # Normalized cx,cy,w,h in [0,1] for geometric prompt
                    cx = ((x1 + x2) * 0.5) / float(W)
                    cy = ((y1 + y2) * 0.5) / float(H)
                    ww = (x2 - x1) / float(W)
                    hh = (y2 - y1) / float(H)
                    box_cxcywh = [float(cx), float(cy), float(ww), float(hh)]
                    _, out = segmentor.model.add_prompt(state, frame_idx=frame_idx_this + first_frame_global_idx, obj_id=obj_id + 1, boxes_xywh=[box_cxcywh], box_labels=[True])
                    obj_ids = out['out_obj_ids']
                    print(obj_ids)
                    obj_masks = out['out_binary_masks']
                    cv2.imwrite(f'mask_{"_".join(list(map(str, obj_ids.tolist())))}.jpg', (out['out_binary_masks'].squeeze()*255).astype(np.byte))
                else:
                    raise NotImplementedError
                log(f'\t[frame={(frame_idx_this + first_frame_global_idx):04d}][obj={obj_id} ({class_name_i})] added bbox: {frame_bbox_i.astype(int).tolist()}', 'debug')

            elif point_key in frame_bbox_data:
                frame_points: List[Tuple[int, int]] = frame_bbox_data[point_key]
                frame_point_labels: List[int] = frame_bbox_data['point_labels']
                if 'frame_idx' in frame_bbox_data:
                    frame_idx_this = int(frame_bbox_data['frame_idx'])
                else:
                    frame_idx_this = frame_idx

                if which_segmentor == 'sam2':
                    segmentor.add_new_points_or_box(state, frame_idx=frame_idx_this + first_frame_global_idx, points=frame_points, obj_id=obj_id + 1, labels=frame_point_labels)
                elif which_segmentor == 'sam3':
                    # compute points and point labels: create a small bbox around each point and input that instead
                    norm_point_boxes: List[Tuple[float, float, float, float]] = []
                    norm_point_labels: List[bool] = []
                    for (px, py), lab in zip(frame_points, frame_point_labels):
                        # Build a 3x3 box centered at (px, py)
                        x1p = max(0.0, px - 1.0)
                        y1p = max(0.0, py - 1.0)
                        x2p = min(W - 1.0, px + 1.0)
                        y2p = min(H - 1.0, py + 1.0)
                        cxp = ((x1p + x2p) * 0.5) / float(W)
                        cyp = ((y1p + y2p) * 0.5) / float(H)
                        wwp = max(1.0, (x2p - x1p)) / float(W)
                        hhp = max(1.0, (y2p - y1p)) / float(H)
                        norm_point_boxes.append((float(cxp), float(cyp), float(wwp), float(hhp)))
                        norm_point_labels.append(bool(1 - lab))
                    segmentor.model.add_prompt(state, frame_idx=frame_idx_this + first_frame_global_idx, points=frame_points, obj_id=obj_id + 1, boxes_xywh=norm_point_boxes, box_labels=norm_point_labels)
                else:
                    raise NotImplementedError
                log(f'\t[frame={(frame_idx_this + first_frame_global_idx):04d}] added points: {frame_bbox_data[point_key]} with labels {frame_bbox_data["point_labels"]}', 'debug')
            else:
                raise KeyError('neither bbox or points found')
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
        if which_segmentor == 'sam2':
            forward_prop = segmentor.propagate_in_video(state, reverse=True)
            if first_frame_global_idx > 0 and False:
                backward_prop = segmentor.propagate_in_video(state, reverse=True)
                propagator = itertools.chain(forward_prop, backward_prop)
            else:
                propagator = forward_prop
            # propagator = segmentor.propagate_in_video(state, reverse=True)
        elif which_segmentor == 'sam3':
            propagator = segmentor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction='both',
                    start_frame_idx=0,
                    # max_frame_num_to_track=30
                )
            )
            # propagator = segmentor.propagate_in_video(session_id, propagation_direction='forward', start_frame_idx=0, max_frame_num_to_track=30)
        else:
            raise NotImplementedError

        masks_per_frame_index = {}
        for _ in propagator:
            if which_segmentor == 'sam2':
                frame_idx, object_ids, masks = _
            elif which_segmentor == 'sam3':
                frame_idx = _['frame_index']
                outputs = _['outputs']
                if outputs is None:
                    log(f'[segment] Got no outputs for frame {frame_idx}', 'warning')
                    continue
                object_ids = outputs['out_obj_ids']
                masks = outputs['out_binary_masks']
            else:
                raise NotImplementedError
            merged_mask = masks_per_frame_index.get(frame_idx, None)
            for obj_id, mask in zip(object_ids, masks):
                if obj_id not in tracked_obj_ids:
                    continue
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().squeeze().numpy()
                binary_mask = mask > 0.5
                if merged_mask is None:
                    merged_mask = binary_mask
                else:
                    merged_mask |= binary_mask  # Logical OR to merge
            if merged_mask is None:
                continue
            masks_per_frame_index[frame_idx] = merged_mask

        for frame_idx, merged_mask in tqdm(masks_per_frame_index.items(), desc='Storing masks to disk'):
            merged_mask = (merged_mask * 255).astype(np.uint8)

            if rotate and unrotate_output:
                if rotate == '90_CLOCKWISE':
                    merged_mask = cv2.rotate(merged_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotate == '90_COUNTERCLOCKWISE':
                    merged_mask = cv2.rotate(merged_mask, cv2.ROTATE_90_CLOCKWISE)
                elif rotate == '180':
                    merged_mask = cv2.rotate(merged_mask, cv2.ROTATE_180)

            PathUtils.write_file(out_paths[int(frame_idx)], merged_mask)

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
        img_np = cv2.rotate(img_np, getattr(cv2, f'ROTATE_{rotate.upper()}'))
        return torch.from_numpy(img_np).permute(2, 0, 1).float()

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
        flow_fwd = FlowUtils.rotate_flow(flow_pred[0, :, :], rotate, inverse=True)
        flow_bwd = FlowUtils.rotate_flow(flow_pred[1, :, :], rotate, inverse=True)

        if 'fwd' in which:
            PathUtils.write_file(out_path[0] if isinstance(out_path, tuple) else out_path, flow_fwd, png_type='flow')
        if 'bwd' in which:
            PathUtils.write_file(out_path[1] if isinstance(out_path, tuple) else out_path, flow_bwd, png_type='flow')

        del flow_fwd, flow_bwd, flow_pred

    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return True


if __name__ == '__main__':
    import json

    with open('/root/capturestudio2/out/reconstructions/Thanos_2_Perf_2/orbbec/cam01/mask/detections-1746110789224.json', 'r') as f:
        first_frame_detections_ = json.load(f)
    color_files_ = sorted(Path('/root/capturestudio2/out/reconstructions/Thanos_2_Perf_2/orbbec/cam01/color').glob('*.jpg'), key=lambda x: int(x.stem))
    out_dir_ = Path('/root/capturestudio2/out/reconstructions/Thanos_2_Perf_2/orbbec/cam01/mask')
    out_paths_ = [out_dir_ / cf.name for cf in color_files_]
    segment(
        video_path='/root/capturestudio2/out/reconstructions/Thanos_2_Perf_2/orbbec/cam01/color/video-000000-000500.mp4',
        first_frame_detections=first_frame_detections_,
        out_paths=out_paths_,
        which_segmentor='sam2'
    )
