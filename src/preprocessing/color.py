import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, List, Dict, Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm

from utils.flow import FlowUtils
from utils.misc import log, PathUtils, get_segmentor, get_detector, get_of_estimator


class SAM3ChunkedSegmentor(object):
    """
    Full chunked SAM3 segmentor.

    Algorithm:
        1. Normalize manual detections into persistent object anchors.
        2. Discover per-object birth/death by chunked backward/forward propagation.
        3. Build singleton annotation intervals.
        4. Merge neighboring intervals by expanding/bridging with SAM3 chunks.
        5. Extend final interval to sequence boundaries.
        6. Write complete union masks.

    Design:
        - External ids are persistent object ids.
        - SAM3 ids are local and never leave _run_one_object().
        - Each persistent object is run in its own SAM3 state, which avoids SAM3
          visual-prompt reset / local-id ambiguity.
        - Prompt boxes passed to SAM3 are normalized [xmin, ymin, width, height].
    """

    def __init__(
            self,
            color_frames: Sequence[Union[str, Path]],
            out_dir: Optional[Union[str, Path]] = None,
            chunk_size: int = 128,
            rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None,
            unrotate_output: bool = True,
            mask_ext: Literal['jpg', 'jpeg', 'png'] = 'jpg',
            min_area: int = 16,
            empty_run: int = 8,
            edge_window: int = 8,
            use_text_prompt: bool = True,
    ):
        self.color_files = [Path(p) for p in color_frames]
        if not self.color_files:
            raise ValueError("color_frames is empty.")

        missing = [p for p in self.color_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing color frame: {missing[0]}")

        self.color_dir = self.color_files[0].parent
        self.out_dir = Path(out_dir) if out_dir is not None else self.color_dir.parent / 'mask'
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.rotate = rotate
        self.unrotate_output = bool(unrotate_output)
        self.mask_ext = mask_ext.lower().lstrip('.')
        self.chunk_size = min(int(chunk_size), len(self.color_files))
        self.min_area = int(min_area)
        self.empty_run = int(empty_run)
        self.edge_window = int(edge_window)
        self.use_text_prompt = bool(use_text_prompt)
        self.total_frames = len(self.color_files)
        self.mask_paths = [(self.out_dir / p.name).with_suffix(f'.{self.mask_ext}') for p in self.color_files]

        assert self.chunk_size >= 2
        assert self.min_area > 0
        assert self.empty_run > 0
        assert self.edge_window > 0

        img0 = cv2.imread(str(self.color_files[0]))
        if img0 is None:
            raise FileNotFoundError(f"Could not read first color frame: {self.color_files[0]}")
        self.frame_h, self.frame_w = img0.shape[:2]
        self.debug = False

    def _dbg(self, msg: str):
        if self.debug:
            log(f"[SAM3ChunkedSegmentor] {msg}", 'debug')

    @staticmethod
    def _merge(dst, *srcs):
        for src in srcs:
            for f, d in src.items():
                dst.setdefault(int(f), {}).update({int(k): v for k, v in d.items()})
        return dst

    def _imgs(self, start: int, end: int) -> List[Image.Image]:
        imgs = []
        for p in self.color_files[start:end + 1]:
            im = Image.open(p).convert('RGB')
            if self.rotate == '90_CLOCKWISE':
                im = im.transpose(Image.Transpose.ROTATE_270)
            elif self.rotate == '90_COUNTERCLOCKWISE':
                im = im.transpose(Image.Transpose.ROTATE_90)
            elif self.rotate == '180':
                im = im.transpose(Image.Transpose.ROTATE_180)
            imgs.append(im)
        return imgs

    def _rot_box(self, xyxy: np.ndarray) -> np.ndarray:
        if self.rotate is None:
            return xyxy.astype(np.float32)

        x1, y1, x2, y2 = xyxy.astype(np.float32).tolist()
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        if self.rotate == '90_CLOCKWISE':
            pts = np.array([[self.frame_h - y, x] for x, y in pts], dtype=np.float32)
        elif self.rotate == '90_COUNTERCLOCKWISE':
            pts = np.array([[y, self.frame_w - x] for x, y in pts], dtype=np.float32)
        elif self.rotate == '180':
            pts = np.array([[self.frame_w - x, self.frame_h - y] for x, y in pts], dtype=np.float32)

        return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)

    def _unrot_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.uint8)
        if not self.rotate or not self.unrotate_output:
            return mask.astype(bool)
        if self.rotate == '90_CLOCKWISE':
            return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool)
        if self.rotate == '90_COUNTERCLOCKWISE':
            return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE).astype(bool)
        if self.rotate == '180':
            return cv2.rotate(mask, cv2.ROTATE_180).astype(bool)
        return mask.astype(bool)

    def _box_xywh(self, prompt, W: int, H: int) -> List[float]:
        box = self._rot_box(np.asarray(prompt['bbox'], dtype=np.float32)[:4])
        x1, y1, x2, y2 = box.tolist()

        x1, x2 = float(np.clip(x1, 0, W - 1)), float(np.clip(x2, 0, W - 1))
        y1, y2 = float(np.clip(y1, 0, H - 1)), float(np.clip(y2, 0, H - 1))

        assert x2 > x1 and y2 > y1, f"Invalid prompt box: {[x1, y1, x2, y2]}"

        return [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]

    def _put(self, dst, frame_idx: int, obj_id: int, mask: np.ndarray):
        mask = self._unrot_mask(np.asarray(mask).squeeze() > 0.5)
        if int(mask.sum()) >= self.min_area:
            dst.setdefault(int(frame_idx), {})[int(obj_id)] = mask

    def _prompt_from_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray):
        mask = np.asarray(mask).astype(bool)
        ys, xs = np.where(mask)

        if len(xs) < self.min_area:
            return None

        meta = self.obj_meta.get(int(obj_id), {})

        return {
            'object_id': int(obj_id),
            'obj_id': int(obj_id),
            'frame_idx': int(frame_idx),
            'bbox': [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            'class_name': meta.get('class_name', ''),
            'class_id': meta.get('class_id', -1),
            'confidence': 1.0,
            'area': int(len(xs)),
            'source': 'propagated',
        }

    def _prompts_at(self, frame_idx: int, masks_for_frame: Dict[int, np.ndarray], keep=None):
        out = {}
        for obj_id, mask in masks_for_frame.items():
            obj_id = int(obj_id)
            if keep is not None and obj_id not in keep:
                continue
            p = self._prompt_from_mask(frame_idx, obj_id, mask)
            if p is not None:
                out[obj_id] = p
        return out

    def _edge(self, masks, obj_ids, ordered_frames):
        """
        Pick a robust handoff edge.

        We scan frames from the desired edge inward. Among the first edge_window
        frames containing any active mask, choose the frame with most active
        objects and largest total area.
        """
        cands = []
        for f in ordered_frames:
            ps = self._prompts_at(f, masks.get(f, {}), keep=obj_ids)
            if ps:
                area = sum(int(p.get('area', 0)) for p in ps.values())
                cands.append((len(ps), area, int(f), ps))
                if len(cands) >= self.edge_window:
                    break

        if not cands:
            return None, {}

        _, _, f, ps = max(cands, key=lambda x: (x[0], x[1]))
        return f, ps

    @staticmethod
    def _live(lives, obj_id: int, start: int, end: int) -> bool:
        life = lives[int(obj_id)]
        return int(life['birth']) <= end and int(life['death']) >= start

    def _active_ids(self, lives, start: int, end: int):
        return {
            int(obj_id)
            for obj_id, life in lives.items()
            if int(life['birth']) <= end and int(life['death']) >= start
        }

    def _select_prompts(self, prompts, lives, start: int, end: int, frame_idx: int):
        out = []
        for obj_id, p in prompts.items():
            if self._live(lives, int(obj_id), start, end):
                q = dict(p)
                q['frame_idx'] = int(frame_idx)
                out.append(q)
        return out

    @torch.inference_mode()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def _run_one_object(
            self,
            inference,
            images,
            start: int,
            end: int,
            prompt: Dict[str, Union[list, float, str, int]],
            direction: Literal['forward', 'backward', 'both'],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        obj_id = int(prompt.get('object_id', prompt.get('obj_id')))
        prompt_frame = int(prompt['frame_idx'])
        local_prompt_frame = prompt_frame - start

        assert start <= prompt_frame <= end

        state = inference.init_state(
            images,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )

        H, W = int(state['orig_height']), int(state['orig_width'])
        box = self._box_xywh(prompt, W, H)

        cls = str(prompt.get('class_name', '')).strip()
        text = cls if self.use_text_prompt and cls else None

        _, out0 = inference.add_prompt(
            state,
            frame_idx=int(local_prompt_frame),
            text_str=text,
            boxes_xywh=[box],
            box_labels=[True],
        )

        sid = int(out0['out_obj_ids'][0])
        out = {}

        for rid, mask in zip(out0['out_obj_ids'], out0['out_binary_masks']):
            if int(rid) == sid:
                self._put(out, prompt_frame, obj_id, mask)

        dirs = ('forward', 'backward') if direction == 'both' else (direction,)

        for d in dirs:
            for local_f, y in inference.propagate_in_video(
                    state,
                    start_frame_idx=int(local_prompt_frame),
                    max_frame_num_to_track=int(end - start + 1),
                    reverse=(d == 'backward'),
            ):
                f = start + int(local_f)
                for rid, mask in zip(y['out_obj_ids'], y['out_binary_masks']):
                    if int(rid) == sid:
                        self._put(out, f, obj_id, mask)

        del state
        return out

    def _run_chunk(
            self,
            start: int,
            end: int,
            prompts: List[Dict[str, Union[list, float, str, int]]],
            direction: Literal['forward', 'backward', 'both'],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        assert 0 <= start <= end < self.total_frames
        assert end - start + 1 <= self.chunk_size

        if not prompts:
            return {}

        images = self._imgs(start, end)
        segmentor = get_segmentor(which='sam3')
        inference = segmentor.model if hasattr(segmentor, 'model') else segmentor

        out = {}
        for p in prompts:
            self._merge(out, self._run_one_object(inference, images, start, end, p, direction))

        n = sum(bool(v) for v in out.values())
        self._dbg(f"chunk [{start}, {end}] dir={direction} prompts={len(prompts)} nonempty_frames={n}")
        return out

    def _normalize(self, detections):
        anchors = {}
        self.obj_meta = {}

        for frame_idx, items in detections.items():
            frame_idx = int(frame_idx)
            assert 0 <= frame_idx < self.total_frames

            for item in items:
                item = dict(item)
                obj_id = int(item.get('object_id', item.get('obj_id')))

                assert 'bbox' in item, f"Expected bbox prompt: {item}"

                item['object_id'] = obj_id
                item['obj_id'] = obj_id
                item['frame_idx'] = frame_idx

                self.obj_meta[obj_id] = {
                    'class_name': item.get('class_name', ''),
                    'class_id': item.get('class_id', -1),
                }

                anchors.setdefault(frame_idx, {})[obj_id] = item

        assert anchors
        return anchors

    def _nonempty(self, masks, obj_id: int, start: int, end: int):
        return [
            f for f in range(start, end + 1)
            if obj_id in masks.get(f, {})
               and int(np.asarray(masks[f][obj_id]).sum()) >= self.min_area
        ]

    def _discover_lifetimes(self, anchors, all_masks):
        by_obj = {}
        for f, ps in anchors.items():
            for obj_id in ps:
                by_obj.setdefault(int(obj_id), []).append(int(f))

        lives = {}

        for obj_id, frames in tqdm(sorted(by_obj.items()), desc='SAM3 birth/death discovery'):
            frames = sorted(set(frames))
            birth = frames[0]
            death = frames[-1]

            p = dict(anchors[birth][obj_id])
            while birth > 0:
                start, end = max(0, birth - self.chunk_size + 1), birth
                p['frame_idx'] = birth

                res = self._run_chunk(start, end, [p], 'backward')
                self._merge(all_masks, res)

                present = [f for f in self._nonempty(res, obj_id, start, end) if f < birth]
                if not present:
                    break

                birth = min(present)
                p = self._prompt_from_mask(birth, obj_id, res[birth][obj_id])

                if birth > start:
                    break

            p = dict(anchors[death][obj_id])
            while death < self.total_frames - 1:
                start, end = death, min(self.total_frames - 1, death + self.chunk_size - 1)
                p['frame_idx'] = death

                res = self._run_chunk(start, end, [p], 'forward')
                self._merge(all_masks, res)

                present = [f for f in self._nonempty(res, obj_id, start, end) if f > death]
                if not present:
                    break

                death = max(present)
                p = self._prompt_from_mask(death, obj_id, res[death][obj_id])

                if death < end:
                    break

            lives[obj_id] = {
                'birth': int(birth),
                'death': int(death),
                'manual_frames': frames,
            }

        self._dbg(f"lifetimes={lives}")
        return lives

    def _singleton_intervals(self, anchors):
        out = []
        for i, f in enumerate(sorted(anchors)):
            ps = {int(k): dict(v) for k, v in anchors[f].items()}
            out.append({
                'id': i,
                'start': int(f),
                'end': int(f),
                'left_edge': {'frame_idx': int(f), 'prompts': ps},
                'right_edge': {'frame_idx': int(f), 'prompts': ps},
                'masks': {},
            })
        return out

    def _merge_pair(self, left, right, lives, all_masks):
        lf = int(left['right_edge']['frame_idx'])
        rf = int(right['left_edge']['frame_idx'])
        active = self._active_ids(lives, lf, rf)

        cur_l, cur_r = lf, rf
        lps = {k: dict(v) for k, v in left['right_edge']['prompts'].items() if k in active}
        rps = {k: dict(v) for k, v in right['left_edge']['prompts'].items() if k in active}
        mid_masks = {}

        while cur_r - cur_l + 1 > self.chunk_size:
            start, end = cur_l, min(cur_r, cur_l + self.chunk_size - 1)
            res = self._run_chunk(start, end, self._select_prompts(lps, lives, start, end, start), 'forward')
            if not res:
                return None, (start, end, 'forward_expansion_failed')

            self._merge(all_masks, res)
            self._merge(mid_masks, res)

            new_l, new_lps = self._edge(res, active, range(end, cur_l, -1))
            if new_l is None or new_l <= cur_l:
                return None, (start, end, 'forward_handoff_failed')

            cur_l, lps = int(new_l), new_lps

            if cur_r - cur_l + 1 <= self.chunk_size:
                break

            start, end = max(cur_l, cur_r - self.chunk_size + 1), cur_r
            res = self._run_chunk(start, end, self._select_prompts(rps, lives, start, end, end), 'backward')
            if not res:
                return None, (start, end, 'backward_expansion_failed')

            self._merge(all_masks, res)
            self._merge(mid_masks, res)

            new_r, new_rps = self._edge(res, active, range(start, cur_r))
            if new_r is None or new_r >= cur_r:
                return None, (start, end, 'backward_handoff_failed')

            cur_r, rps = int(new_r), new_rps

        start, end = cur_l, cur_r
        prompts = (
                self._select_prompts(lps, lives, start, end, start) +
                self._select_prompts(rps, lives, start, end, end)
        )

        res = self._run_chunk(start, end, prompts, 'both')
        if not res:
            return None, (start, end, 'bridge_failed')

        self._merge(all_masks, res)
        self._merge(mid_masks, res)

        merged_masks = {}
        self._merge(merged_masks, left.get('masks', {}), mid_masks, right.get('masks', {}))

        return {
            'id': -1,
            'start': int(left['start']),
            'end': int(right['end']),
            'left_edge': left['left_edge'],
            'right_edge': right['right_edge'],
            'masks': merged_masks,
        }, None

    def _merge_intervals(self, intervals, lives, all_masks):
        failed = set()
        unresolved = []
        next_id = max(i['id'] for i in intervals) + 1 if intervals else 0

        pbar = tqdm(total=max(0, len(intervals) - 1), desc='SAM3 merging annotation intervals')

        while len(intervals) > 1:
            intervals = sorted(intervals, key=lambda x: int(x['start']))

            pairs = [
                (int(intervals[i + 1]['start']) - int(intervals[i]['end']), i)
                for i in range(len(intervals) - 1)
                if (int(intervals[i]['id']), int(intervals[i + 1]['id'])) not in failed
            ]

            if not pairs:
                break

            _, i = min(pairs, key=lambda x: x[0])
            left, right = intervals[i], intervals[i + 1]
            merged, err = self._merge_pair(left, right, lives, all_masks)

            if merged is None:
                failed.add((int(left['id']), int(right['id'])))
                unresolved.append(err)
                continue

            merged['id'] = next_id
            next_id += 1
            intervals = intervals[:i] + [merged] + intervals[i + 2:]
            pbar.update(1)

        pbar.close()
        return intervals, unresolved

    def _extend_left(self, edge, lives, all_masks):
        unresolved = []
        cur = {
            'frame_idx': int(edge['frame_idx']),
            'prompts': {int(k): dict(v) for k, v in edge['prompts'].items()},
        }

        pbar = tqdm(total=cur['frame_idx'], desc='SAM3 extending left')

        while cur['frame_idx'] > 0:
            old = int(cur['frame_idx'])
            start, end = max(0, old - self.chunk_size + 1), old
            active = self._active_ids(lives, start, end)
            prompts = self._select_prompts(
                {k: v for k, v in cur['prompts'].items() if k in active},
                lives,
                start,
                end,
                end,
            )

            res = self._run_chunk(start, end, prompts, 'backward')
            if not res:
                unresolved.append((start, end, 'left_extension_failed'))
                break

            self._merge(all_masks, res)

            new, ps = self._edge(res, active, range(start, end))
            if new is None or new >= old:
                unresolved.append((start, end, 'left_extension_empty'))
                break

            cur = {'frame_idx': int(new), 'prompts': ps}
            pbar.update(old - int(new))

        pbar.close()
        return unresolved

    def _extend_right(self, edge, lives, all_masks):
        unresolved = []
        cur = {
            'frame_idx': int(edge['frame_idx']),
            'prompts': {int(k): dict(v) for k, v in edge['prompts'].items()},
        }

        pbar = tqdm(total=max(0, self.total_frames - 1 - cur['frame_idx']), desc='SAM3 extending right')

        while cur['frame_idx'] < self.total_frames - 1:
            old = int(cur['frame_idx'])
            start, end = old, min(self.total_frames - 1, old + self.chunk_size - 1)
            active = self._active_ids(lives, start, end)
            prompts = self._select_prompts(
                {k: v for k, v in cur['prompts'].items() if k in active},
                lives,
                start,
                end,
                start,
            )

            res = self._run_chunk(start, end, prompts, 'forward')
            if not res:
                unresolved.append((start, end, 'right_extension_failed'))
                break

            self._merge(all_masks, res)

            new, ps = self._edge(res, active, range(end, start, -1))
            if new is None or new <= old:
                unresolved.append((start, end, 'right_extension_empty'))
                break

            cur = {'frame_idx': int(new), 'prompts': ps}
            pbar.update(int(new) - old)

        pbar.close()
        return unresolved

    def _write(self, masks, overwrite: bool, write_blank_for_missing: bool):
        blank = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)

        for f, out_path in enumerate(self.mask_paths):
            if out_path.exists() and PathUtils.verify_file(out_path) and not overwrite:
                continue

            union = None
            for mask in masks.get(f, {}).values():
                mask = np.asarray(mask).astype(bool)
                if int(mask.sum()) >= self.min_area:
                    union = mask if union is None else (union | mask)

            if union is None:
                if write_blank_for_missing:
                    PathUtils.write_file(out_path, blank)
            else:
                PathUtils.write_file(out_path, (union * 255).astype(np.uint8))

    def run(
            self,
            detections: Dict[int, List[Dict[str, Union[list, float, str, int]]]],
            overwrite: bool = True,
            write_blank_for_missing: bool = True,
            fail_if_all_empty: bool = True,
            debug: bool = True,
    ) -> bool:
        self.debug = bool(debug)

        anchors = self._normalize(detections)
        all_masks = {}

        self._dbg(
            f"manual_frames={sorted(anchors)}, "
            f"objects={sorted({obj for ps in anchors.values() for obj in ps})}, "
            f"chunk_size={self.chunk_size}, edge_window={self.edge_window}, "
            f"use_text_prompt={self.use_text_prompt}"
        )

        lives = self._discover_lifetimes(anchors, all_masks)
        intervals = self._singleton_intervals(anchors)
        intervals, unresolved = self._merge_intervals(intervals, lives, all_masks)

        if len(intervals) == 1:
            final = intervals[0]
            unresolved += self._extend_left(final['left_edge'], lives, all_masks)
            unresolved += self._extend_right(final['right_edge'], lives, all_masks)
        else:
            unresolved.append((-1, -1, f'unmerged_intervals={len(intervals)}'))

        non_empty_frames = sum(
            any(int(np.asarray(mask).sum()) >= self.min_area for mask in d.values())
            for d in all_masks.values()
        )

        if non_empty_frames == 0:
            log("[SAM3ChunkedSegmentor] SAM3 produced zero non-empty masks.", 'error')
            return not fail_if_all_empty

        self._write(all_masks, overwrite=overwrite, write_blank_for_missing=write_blank_for_missing)

        log(
            f"[SAM3ChunkedSegmentor] done: "
            f"non_empty_frames={non_empty_frames}/{self.total_frames}, "
            f"remaining_intervals={len(intervals)}, unresolved={unresolved[:20]}",
            'warning' if unresolved else 'debug',
        )

        if fail_if_all_empty and non_empty_frames <= len(anchors):
            log(
                f"[SAM3ChunkedSegmentor] only manual anchors survived; unresolved={unresolved[:20]}",
                'error',
            )
            return False

        return len(intervals) == 1


class ColorProcessor(object):
    """
    Color-frame processor for detection, SAM3 segmentation, and optical-flow estimation.

    Segmentation model:
        - Manual annotations are timeline anchors.
        - SAM3 is only run on bounded windows of <= segment_chunk_size frames.
        - Manual anchors are expanded backward/forward to create propagated anchors.
        - Final masks are produced by bridge windows between adjacent anchors, using endpoint prompts when possible.
        - Output masks are always complete for the selected frame range; missing/unresolved frames become blank masks.
    """

    def __init__(
            self,
            cam_color_dir: str,
            start_offset: int = 0,
            total_frames: int = -1,

            rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None,
            unrotate_output: bool = False,

            detect_classes: Sequence[str] = ('person', 'guitar', 'guitar band', 'drums', 'chair'),
            detect_threshold: float = 0.5,
            segment_chunk_size: int = 128,

            color_ext: Literal['jpg', 'jpeg', 'png', 'heic', 'bmp', 'tiff', 'webp'] = 'jpg',
            mask_folder_name: str = 'mask',
            mask_ext: Literal['jpg', 'jpeg', 'png', 'heic', 'bmp', 'tiff', 'webp'] = 'jpg',
            flow_fwd_folder_name: str = 'flow_fwd',
            flow_bwd_folder_name: str = 'flow_bwd',
            flow_ext: Literal['png', 'tiff', 'flow', 'npy'] = 'png',
    ):
        self.color_dir = Path(cam_color_dir)
        if not self.color_dir.exists() or not self.color_dir.is_dir():
            raise FileNotFoundError(f"Color directory does not exist: {self.color_dir}")

        self.mask_dir = self.color_dir.parent / mask_folder_name
        self.flow_fwd_dir = self.color_dir.parent / flow_fwd_folder_name
        self.flow_bwd_dir = self.color_dir.parent / flow_bwd_folder_name

        self._rotate = rotate
        self._unrotate_output = bool(unrotate_output)

        self._detect_classes = tuple(detect_classes)
        self._detect_threshold = float(detect_threshold)
        self._segment_chunk_size = int(segment_chunk_size)

        self._color_ext = color_ext.lower().lstrip('.')
        self._mask_ext = mask_ext.lower().lstrip('.')
        self._flow_ext = flow_ext.lower().lstrip('.')

        self._all_color_files = sorted(self.color_dir.glob(f'*.{self._color_ext}'), key=lambda x: int(x.stem))
        if not self._all_color_files:
            raise FileNotFoundError(f"No *.{self._color_ext} frames found in: {self.color_dir}")

        if start_offset < 0:
            start_offset = len(self._all_color_files) + start_offset
        if start_offset < 0 or start_offset >= len(self._all_color_files):
            raise ValueError(f"Invalid start_offset={start_offset}; available frames={len(self._all_color_files)}")

        if total_frames == -1:
            total_frames = len(self._all_color_files) - start_offset
        elif total_frames < 0:
            total_frames = len(self._all_color_files) + total_frames - start_offset + 1

        if total_frames <= 0:
            raise ValueError(f"Invalid total_frames={total_frames}")
        if start_offset + total_frames > len(self._all_color_files):
            raise ValueError(
                f"start_offset + total_frames exceeds available frames: "
                f"{start_offset} + {total_frames} > {len(self._all_color_files)}"
            )

        self.start_offset = int(start_offset)
        self.total_frames = int(total_frames)
        self._color_files = self._all_color_files[self.start_offset:self.start_offset + self.total_frames]
        self._mask_paths = [(self.mask_dir / p.name).with_suffix(f'.{self._mask_ext}') for p in self._color_files]

        first_img = cv2.imread(str(self._color_files[0]))
        if first_img is None:
            raise FileNotFoundError(f"Could not read first color frame: {self._color_files[0]}")
        self._frame_h, self._frame_w = first_img.shape[:2]

        self._segment_chunk_size = int(min(self._segment_chunk_size, self.total_frames))

        self.segmentor = SAM3ChunkedSegmentor(
            color_frames=self._color_files,
            out_dir=self.mask_dir,
            chunk_size=self._segment_chunk_size,
            rotate=self._rotate,
            unrotate_output=self._unrotate_output,
            mask_ext=self._mask_ext,
        )

    @torch.inference_mode()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def _segment(
            self,
            detections: Dict[int, List[Dict[str, Union[list, float, str, int]]]],
            **kwargs
    ) -> bool:
        return self.segmentor.run(detections=detections, **kwargs)

    def segment(
            self,
            **kwargs,
    ) -> bool:
        """
        Read ``detections-<color_timestamp>.json`` files, convert them to frame-relative
        detections, and run SAM3 segmentation.

        The filename timestamp gives the base color frame. If an item contains
        ``frame_idx``, it is interpreted as an offset from that base frame. The final
        index passed to SAM3 is relative to the selected ``self._color_files`` list.
        """
        stem_to_global_idx = {p.stem: i for i, p in enumerate(self._all_color_files)}
        detections_rel: Dict[int, List[Dict[str, Union[list, float, str, int]]]] = {}

        self.track_to_obj_id = {}
        self.obj_id_to_track = {}
        next_obj_id = 1

        detection_files = sorted(self.mask_dir.glob('detections-*.json'), key=lambda p: p.stem)
        if not detection_files:
            raise FileNotFoundError(f"No detections-*.json files found in: {self.mask_dir}")

        for detection_path in detection_files:
            color_stem = detection_path.stem[len('detections-'):]
            if color_stem not in stem_to_global_idx:
                log(
                    f"[ColorProcessor] skipping detection file with unknown color timestamp: "
                    f"{detection_path.name}",
                    'warning',
                )
                continue

            base_global_frame_idx = int(stem_to_global_idx[color_stem])
            with open(detection_path, 'r') as f:
                items = json.load(f)
            for item in items:
                item = dict(item)
                if 'object_id' in item:
                    object_id = int(item['object_id'])
                else:
                    object_id = next_obj_id
                    next_obj_id += 1
                    log(
                        f"[ColorProcessor] detection in {detection_path.name} has no object_id/object_id/track_id; "
                        f"assigned object_id={object_id}.",
                        'warning',
                    )
                next_obj_id = max(next_obj_id, object_id + 1)
                frame_offset = int(item.get('frame_idx', 0))
                global_frame_idx = base_global_frame_idx + frame_offset
                local_frame_idx = global_frame_idx - self.start_offset

                if not (0 <= local_frame_idx < self.total_frames):
                    log(
                        f"[ColorProcessor] skipping detection outside selected range: "
                        f"{detection_path.name}, base_global={base_global_frame_idx}, "
                        f"frame_offset={frame_offset}, global_frame_idx={global_frame_idx}, "
                        f"selected_global=[{self.start_offset}, {self.start_offset + self.total_frames})",
                        'warning',
                    )
                    continue

                item['object_id'] = int(object_id)
                item['object_id'] = int(object_id)
                item['frame_idx'] = int(local_frame_idx)
                detections_rel.setdefault(int(local_frame_idx), []).append(item)

        assert detections_rel, f"No valid detections inside selected frame range: {self.mask_dir}"
        for frame_idx, items in sorted(detections_rel.items()):
            for item in items:
                log(
                    f"[ColorProcessor] detection frame={frame_idx}, object_id={item['object_id']}, "
                    f"class={item.get('class_name')}, bbox={item.get('bbox')}",
                    'debug',
                )

        debug_bbox_dir = self.mask_dir / 'debug_bboxes'
        debug_bbox_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, items in sorted(detections_rel.items()):
            color_path = self._color_files[int(frame_idx)]
            out_path = debug_bbox_dir / color_path.name
            torchvision.io.write_jpeg(
                torchvision.utils.draw_bounding_boxes(
                    torchvision.io.read_image(str(color_path)),
                    torch.stack([
                        torch.from_numpy(np.asarray(item['bbox'], dtype=np.float32)[:4].round())
                        for item in items
                    ]),
                    [item['class_name'] for item in items],
                    ['red', 'green', 'blue', 'yellow', 'purple', 'brown', 'magenta', 'cyan'][:len(items)],
                ),
                str(out_path)
            )
            log(f"[ColorProcessor] wrote debug bbox frame: {out_path}", 'debug')

        res = self._segment(detections_rel, **kwargs)

        if res:
            log(f'\t[{self.__class__.__name__}::_segment] completed successfully for "{self.color_dir.parent.name}/{self.color_dir.name}".', 'info')
        else:
            log(f'\t[{self.__class__.__name__}::_segment] FAILED for "{self.color_dir.parent.name}/{self.color_dir.name}".', 'error')

        return res

    def generate_mask_overlay_video(
            self,
            out_name: str = 'mask_overlay.mp4',
            fps: int = 30,
            alpha: float = 0.6,
            mask_color: Tuple[int, int, int] = (0, 255, 0),
            draw_missing_as_blank: bool = True,
            overwrite: bool = True,
    ) -> Path:
        """
        Generate a color video with segmentation masks overlaid and store it in the mask directory.

        Parameters
        ----------
        out_name:
            Output video filename. Saved under self.mask_dir.
        fps:
            Output FPS.
        alpha:
            Overlay opacity in [0, 1].
        mask_color:
            BGR color used for the mask overlay.
        draw_missing_as_blank:
            If True, missing/invalid mask files are treated as empty masks.
            If False, missing/invalid masks raise an error.
        overwrite:
            If False and output exists, skip generation.

        Returns
        -------
        Path
            Path to generated overlay video.
        """
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")

        out_path = self.mask_dir / out_name
        if out_path.exists() and PathUtils.verify_file(out_path) and not overwrite:
            log(f'\tMask overlay video already exists: {out_path.parent.name}/{out_path.name}. Skipping.', 'debug')
            return out_path

        color_files = self._color_files
        mask_paths = self._mask_paths

        if len(color_files) != len(mask_paths):
            raise RuntimeError(f"Color/mask path count mismatch: {len(color_files)} vs {len(mask_paths)}")
        if not color_files:
            raise RuntimeError(f"No color frames selected for overlay video: {self.color_dir}")

        first_frame = cv2.imread(str(color_files[0]))
        if first_frame is None:
            raise FileNotFoundError(f"Could not read first color frame: {color_files[0]}")

        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

        mask_color_arr = np.array(mask_color, dtype=np.uint8).reshape(1, 1, 3)

        try:
            for color_path, mask_path in tqdm(
                    zip(color_files, mask_paths),
                    total=len(color_files),
                    desc=f'Generating mask overlay video for "{self.color_dir.parent.name}/{self.color_dir.name}"',
                    disable=False,
            ):
                frame = cv2.imread(str(color_path))
                if frame is None:
                    raise FileNotFoundError(f"Could not read color frame: {color_path}")

                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    if draw_missing_as_blank:
                        mask = np.zeros((h, w), dtype=np.uint8)
                    else:
                        raise FileNotFoundError(f"Could not read mask frame: {mask_path}")

                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                mask_bool = mask > 0
                if np.any(mask_bool):
                    overlay = np.broadcast_to(mask_color_arr, frame.shape).copy()
                    blended = cv2.addWeighted(frame, 1.0 - float(alpha), overlay, float(alpha), 0.0)
                    frame[mask_bool] = blended[mask_bool]

                writer.write(frame)

        finally:
            writer.release()

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to generate non-empty overlay video: {out_path}")

        log(f'\tGenerated mask overlay video: {out_path.parent.name}/{out_path.name}', 'debug')
        return out_path

    @torch.inference_mode()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def _detect_single_image(self, image_path: Union[str, Path]) -> List[Dict[str, Union[list, float, str, int]]]:
        image_path = Path(image_path)
        detector, detector_class_names = get_detector('sam3')
        # device = detector.device

        class_ids = []
        available_names = {str(v).lower(): k for k, v in detector_class_names.items()}
        for cls in self._detect_classes:
            key = str(cls).lower()
            if key in available_names:
                class_ids.append(available_names[key])
            else:
                log(f"Class '{cls}' not found in model classes", 'warning')

        if not class_ids:
            raise ValueError("None of the target classes are valid for this model")

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        if self._rotate:
            img = cv2.rotate(img, getattr(cv2, f'ROTATE_{self._rotate}'))

        new_h, new_w = img.shape[:2]
        detections: List[Dict[str, Union[list, float, str, int]]] = []

        from sam3.model.sam3_image_processor import Sam3Processor
        processor: Sam3Processor = getattr(detector, "_processor", None)
        if processor is None:
            raise AttributeError(
                "SAM3 detector is missing `_processor`. "
                "Ensure get_detector('sam3') attaches Sam3Processor to the model."
            )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        requested = [detector_class_names[i] for i in class_ids]
        catalog_map = {name: i for i, name in detector_class_names.items()}

        state = processor.set_image(pil)

        for cls in requested:
            out = processor.set_text_prompt(state=state, prompt=cls)
            boxes = out.get("boxes", [])
            scores = out.get("scores", [])

            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            else:
                boxes = np.asarray(boxes) if len(boxes) else np.zeros((0, 4), dtype=np.float32)
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy().astype(np.float32)
            else:
                scores = np.asarray(scores, dtype=np.float32) if len(scores) else np.zeros((0,), dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1, )

            if boxes.shape[0] == 0 or scores.shape[0] == 0:
                continue

            keep_idx = np.where(scores >= self._detect_threshold)[0]
            if keep_idx.size == 0:
                continue

            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

            boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(new_w - 1))
            boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(new_w - 1))
            boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(new_h - 1))
            boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(new_h - 1))

            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            scores = scores[valid]
            if boxes.shape[0] == 0:
                continue

            keep = nms(
                torch.as_tensor(boxes, dtype=torch.float32),
                torch.as_tensor(scores, dtype=torch.float32),
                iou_threshold=0.5,
            ).detach().cpu().numpy().tolist()

            boxes = boxes[keep]
            scores = scores[keep]

            for bbox_xyxy, score in zip(boxes, scores):
                bbox_xyxy = bbox_xyxy.astype(np.float32)
                bbox_xyxy_rotated = bbox_xyxy.copy()
                bbox_out = bbox_xyxy.tolist()

                if self._rotate and self._unrotate_output:
                    x1, y1, x2, y2 = bbox_xyxy.tolist()
                    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                    if self._rotate == '90_CLOCKWISE':
                        transformed = np.array([[y, new_w - x] for x, y in corners], dtype=np.float32)
                    elif self._rotate == '90_COUNTERCLOCKWISE':
                        transformed = np.array([[new_h - y, x] for x, y in corners], dtype=np.float32)
                    elif self._rotate == '180':
                        transformed = np.array([[new_w - x, new_h - y] for x, y in corners], dtype=np.float32)
                    else:
                        transformed = corners
                    xs, ys = transformed[:, 0], transformed[:, 1]
                    bbox_out = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

                d = {
                    "bbox": [float(c) for c in bbox_out],
                    "confidence": float(score),
                    "class_name": cls,
                    "class_id": int(catalog_map.get(cls, 0)),
                }
                if self._rotate:
                    d[f"bbox_rotated_{self._rotate}"] = [float(c) for c in bbox_xyxy_rotated.tolist()]
                detections.append(d)

        processor.reset_all_prompts(state)
        return detections

    def detect(self, frame_idx: int = 0, write_json: bool = True) -> List[Dict[str, Union[list, float, str, int]]]:
        if not (0 <= frame_idx < self.total_frames):
            raise ValueError(f"frame_idx must be in [0, {self.total_frames - 1}], got {frame_idx}")

        frame_path = self._color_files[frame_idx]
        detections = self._detect_single_image(frame_path)
        for d in detections:
            d['frame_idx'] = int(frame_idx)

        if write_json:
            self.mask_dir.mkdir(parents=True, exist_ok=True)
            import json
            out_path = self.mask_dir / f'detections-{frame_path.stem}.json'
            with open(out_path, 'w') as f:
                json.dump(detections, f, indent=4)

        return detections

    @torch.inference_mode()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def _estimate_flow(self, which: Literal['fwd', 'bwd', 'fwd+bwd'] = 'bwd'):
        def rotate_tensor_img(img, rotate):
            img_np = img.permute(1, 2, 0).byte().numpy()
            img_np = cv2.rotate(img_np, getattr(cv2, f'ROTATE_{rotate.upper()}'))
            return torch.from_numpy(img_np).permute(2, 0, 1).float()

        if 'fwd' in which:
            self.flow_fwd_dir.mkdir(parents=True, exist_ok=True)
            of_fwd_file_paths = [
                (self.flow_fwd_dir / cf.name).with_suffix(f'.{self._flow_ext}')
                for cf in self._color_files[1:self.total_frames]
            ]
        else:
            of_fwd_file_paths = []

        if 'bwd' in which:
            self.flow_bwd_dir.mkdir(parents=True, exist_ok=True)
            of_bwd_file_paths = [
                (self.flow_bwd_dir / cf.name).with_suffix(f'.{self._flow_ext}')
                for cf in self._color_files[1:self.total_frames]
            ]
        else:
            of_bwd_file_paths = []

        all_files_exist = True
        if 'fwd' in which:
            all_files_exist = all_files_exist and all(p.exists() and PathUtils.verify_file(p) for p in of_fwd_file_paths)
        if 'bwd' in which:
            all_files_exist = all_files_exist and all(p.exists() and PathUtils.verify_file(p) for p in of_bwd_file_paths)

        if all_files_exist:
            log(f'\tOptical flows already exist for "{self.color_dir.parent.name}/{self.color_dir.name}". Skipping OF estimation.', 'debug')
            return True

        of_estimator, of_estimator_padder = get_of_estimator()
        input_padder = None

        if 'fwd' in which and 'bwd' in which:
            paths_iterator = list(zip(of_fwd_file_paths, of_bwd_file_paths))
        elif 'fwd' in which:
            paths_iterator = of_fwd_file_paths
        else:
            paths_iterator = of_bwd_file_paths

        for idx, out_path in tqdm(
                enumerate(paths_iterator),
                total=max(0, self.total_frames - 1),
                desc=f'Generating optical flows for "{self.color_dir.parent.name}/{self.color_dir.name}"',
                disable=False,
        ):
            if isinstance(out_path, Path):
                if out_path.exists() and PathUtils.verify_file(out_path):
                    continue
            elif all(p.exists() and PathUtils.verify_file(p) for p in out_path):
                continue

            prev_path = self._color_files[max(0, idx)]
            ref_path = self._color_files[idx + 1]
            next_path = self._color_files[min(self.total_frames - 1, idx + 2)]

            prev_img_raw = cv2.imread(str(prev_path))
            ref_img_raw = cv2.imread(str(ref_path))
            next_img_raw = cv2.imread(str(next_path))

            if prev_img_raw is None:
                raise FileNotFoundError(f"Could not read frame: {prev_path}")
            if ref_img_raw is None:
                raise FileNotFoundError(f"Could not read frame: {ref_path}")
            if next_img_raw is None:
                log(f'Failed to read next image: {next_path.parent.name}/{next_path.name}. Replacing it with reference image.', 'warning')
                shutil.copy(ref_path, next_path)
                next_img_raw = cv2.imread(str(next_path))
                if next_img_raw is None:
                    raise FileNotFoundError(f"Could not read replacement frame: {next_path}")

            prev_img = torch.from_numpy(cv2.cvtColor(prev_img_raw, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
            ref_img = torch.from_numpy(cv2.cvtColor(ref_img_raw, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
            next_img = torch.from_numpy(cv2.cvtColor(next_img_raw, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

            if self._rotate:
                prev_img = rotate_tensor_img(prev_img, self._rotate)
                ref_img = rotate_tensor_img(ref_img, self._rotate)
                next_img = rotate_tensor_img(next_img, self._rotate)

            input_imgs = torch.stack([prev_img, ref_img, next_img], dim=0)[None].cuda()
            original_size = (ref_img.shape[-2], ref_img.shape[-1])

            input_imgs = F.interpolate(
                input_imgs.view(-1, 3, original_size[0], original_size[1]),
                scale_factor=0.25,
                mode='bilinear',
                align_corners=False,
            ).view(-1, 3, 3, original_size[0] // 4, original_size[1] // 4)

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
                align_corners=False,
            )

            flow_pred[..., 0, :, :] *= W_orig / W_net
            flow_pred[..., 1, :, :] *= H_orig / H_net

            flow_fwd = FlowUtils.rotate_flow(flow_pred[0, :, :], self._rotate, inverse=True)
            flow_bwd = FlowUtils.rotate_flow(flow_pred[1, :, :], self._rotate, inverse=True)

            if 'fwd' in which:
                PathUtils.write_file(out_path[0] if isinstance(out_path, tuple) else out_path, flow_fwd, png_type='flow')
            if 'bwd' in which:
                PathUtils.write_file(out_path[1] if isinstance(out_path, tuple) else out_path, flow_bwd, png_type='flow')

            del input_imgs, flow_fwd, flow_bwd, flow_pred

        torch.cuda.empty_cache()
        import gc
        gc.collect()
        return True

    def estimate_flow(self, which: Literal['fwd', 'bwd', 'fwd+bwd'] = 'bwd') -> bool:
        return self._estimate_flow(which=which)


if __name__ == '__main__':
    cp_ = ColorProcessor(
        cam_color_dir='/home/charisoudis/CAPTURESTUDIO_CACHE/Captures_Cagliari_Jun_2026/Cagliari_2_5cams_Perf_1/orbbec/cam01/color',
        segment_chunk_size=1000,
        # start_offset=400,
        # total_frames=1000,
    )
    ok_ = cp_.segment()
    if ok_:
        cp_.generate_mask_overlay_video()
    else:
        log('\tSkipping mask overlay video because segmentation failed.', 'warning')
        for p_ in cp_.mask_dir.glob(f'*.{cp_._mask_ext}'):
            p_.unlink()
