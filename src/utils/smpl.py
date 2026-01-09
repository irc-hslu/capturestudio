# COPYRIGHT 2024 by Athanasios Charisoudis <athanasios.charisoudis@ieee.org>
# Licensed under the Apache License, Version 2.0 (the "License");
# Original source: https://github.com/charisoudis/mdmc
from __future__ import annotations

import copy
import enum
import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
import math
import os
import pprint
import re
import warnings
from collections import OrderedDict
from difflib import SequenceMatcher
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Any, Sequence, Optional, Type, Union, Callable, List, MutableMapping, Tuple, Iterable, Dict as DictT, Iterator, TypeVar

import PIL.Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
from torch import Tensor
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import Weights

from utils.misc import log, Str

KernelType = int or Tuple[int, int]


class BboxFormat(enum.Enum):
    XYXY = 'xyxy'
    XYWH = 'xywh'
    CXYWH = 'cxywh'
    CXYWHN = 'cxywhn'  # center and width,height normalized wrt image size

    @property
    def is_normalized(self) -> bool:
        return self == BboxFormat.CXYWHN

    def to_xyxy(self, bbox: Tensor, img_shape: Optional[Union[torch.Size, Tuple[int, int]]] = None) -> Tensor:
        """
        Converts the bounding boxes to the XYXY format.

        Parameters
        ----------
        bbox : Tensor
            The bounding boxes in format (x1, y1, x2, y2) or (x, y, w, h) or (cx, cy, w, h).
        img_shape : torch.Size or Tuple[int, int], optional
            The shape of the image, by default None.

        Returns
        -------
        Tensor
            The converted bounding boxes.
        """
        if self == BboxFormat.XYXY:
            return bbox
        if self == BboxFormat.XYWH:
            return torch.cat([bbox[:, :2], bbox[:, :2] + bbox[:, 2:4]], dim=-1)
        if self == BboxFormat.CXYWH:
            return torch.cat([bbox[:, :2] - bbox[:, 2:4] / 2, bbox[:, :2] + bbox[:, 2:4] / 2], dim=-1)
        if self == BboxFormat.CXYWHN:
            assert img_shape is not None, f'Image shape must be provided for normalized bounding boxes.'
            # unormalize
            bbox[:, 0:4:2] *= img_shape[0]
            bbox[:, 1:4:2] *= img_shape[1]
            # convert coordinates
            return torch.cat([bbox[:, :2] - bbox[:, 2:4] / 2, bbox[:, :2] + bbox[:, 2:4] / 2], dim=-1)


class Bbox:
    """Bbox Class:
    This class implements a collection of useful functions for bounding boxes.
    """

    def __init__(self,
                 bbox: Tensor,
                 fmt: Union[BboxFormat, str] = BboxFormat.XYXY,
                 frame_wh: Optional[Union[Tensor, np.ndarray, Sequence[int]]] = None,
                 masks: Optional[Tensor] = None):
        """Initializes the bounding boxes.

        Parameters
        ----------
        bbox : Tensor
            The bounding boxes in format (x1, y1, x2, y2) or (x, y, w, h) or (cx, cy, w, h), shape (batch_size, 4).
        fmt : str, optional
            The format of the bounding boxes, by default 'xyxy'. The other options are 'top_left_scale', 'center_scale'.
        frame_wh: Union[Tensor, np.ndarray, Sequence[int]], optional
            The image (NOT PATCH) size in width, height format, shape (batch_size, 2).
        masks: Tensor, optional
            The binary mask of the human, shape (batch_size, 1, frame_height, frame_width).
        """
        fmt = BboxFormat(fmt)
        assert frame_wh is not None or not fmt.is_normalized, \
            f'Image shape must be provided for normalized bounding boxes.'
        assert bbox.ndim in [2, 3]
        if bbox.ndim == 3:
            bbox = bbox.squeeze(1)
        self.integerify = torch.round
        if frame_wh is not None:
            if isinstance(frame_wh, Sequence):
                frame_wh = torch.tensor(frame_wh, device=bbox.device).int()[None, :]
            elif isinstance(frame_wh, np.ndarray):
                if frame_wh.ndim == 1:
                    frame_wh = frame_wh[None, :]
                frame_wh = torch.from_numpy(frame_wh).to(device=bbox.device).int()
            if frame_wh.shape[0] == 1:
                frame_wh = frame_wh.repeat(bbox.shape[0] if bbox.ndim == 2 else 1, 1)
            self.frame_wh: Tensor = frame_wh  # (batch_size, 2)
        else:
            self.frame_wh = None
        self.bbox_xyxy = fmt.to_xyxy(bbox, frame_wh).int()
        self.format_xyxy = BboxFormat.XYXY
        self.centers = (self.bbox_xyxy[:, :2] + self.bbox_xyxy[:, 2:4]) / 2  # shape (N,2)
        self.widths: Tensor = self.bbox_xyxy[:, 2] - self.bbox_xyxy[:, 0]  # shape (N,)
        self.heights: Tensor = self.bbox_xyxy[:, 3] - self.bbox_xyxy[:, 1]  # shape (N,)
        if masks is not None:
            self.masks = masks.unsqueeze(0) if masks.ndim == 2 else masks
        else:
            self.masks = None

    def __contains__(self, point_xy: Tensor) -> bool:
        bbox_xyxy = self.bbox_xyxy
        return (bbox_xyxy[..., 0] < point_xy[..., 0] < bbox_xyxy[..., 2] and bbox_xyxy[..., 1] < point_xy[..., 1] < bbox_xyxy[..., 3]).item()

    def __getitem__(self, index: int) -> Bbox:
        return Bbox(
            bbox=self.bbox_xyxy[[index]],
            fmt='xyxy',
            frame_wh=self.frame_wh[index] if isinstance(self.frame_wh, Tensor) and self.frame_wh.ndim == 2 else self.frame_wh,
            masks=self.masks[[index]] if self.masks is not None else None,
        )

    def __iter__(self) -> Iterator[Bbox]:
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __len__(self) -> int:
        return len(self.bbox_xyxy)

    def clone(self) -> Bbox:
        return Bbox(
            bbox=self.bbox_xyxy.clone(),
            fmt='xyxy',
            frame_wh=self.frame_wh,
            masks=self.masks,
        )

    @staticmethod
    def _upcast(t: Tensor) -> Tensor:
        # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
        if t.is_floating_point():
            return t if t.dtype in (torch.float32, torch.float64) else t.float()
        else:
            return t if t.dtype in (torch.int32, torch.int64) else t.int()

    def intersects(self, other: Bbox) -> Tensor:
        lt = torch.max(self.xyxy[:, None, :2], other.xyxy[:, :2])  # [N,M,2]
        rb = torch.min(self.xyxy[:, None, 2:], other.xyxy[:, 2:])  # [N,M,2]
        wh = self._upcast(rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        return (inter > 0).any(dim=-1)  # [N,] - True if any intersection with any other bbox

    def blur(self, inp: Tensor, mode: str = 'bilinear', inplace: bool = False,
             fill_mode: str = 'interpolate') -> Tensor:
        if not inplace:
            inp = inp.clone()
        assert self.xyxy.shape[0] == 1 or self.xyxy.ndim == 1
        for bbox_xyxy in self.xyxy:
            bbox_xyxy = bbox_xyxy.int().tolist()
            y_indices = [
                bbox_xyxy[1],
                bbox_xyxy[1],
                bbox_xyxy[3] - 1,
                bbox_xyxy[3] - 1,
            ]
            x_indices = [
                bbox_xyxy[0],
                bbox_xyxy[2] - 1,
                bbox_xyxy[0],
                bbox_xyxy[2] - 1,
            ]
            if fill_mode == 'interpolate':
                inp[..., bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = \
                    torch.nn.functional.interpolate(
                        inp[..., y_indices, x_indices].reshape(-1, 2, 2).unsqueeze(0),
                        size=(bbox_xyxy[3] - bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0]),
                        mode=mode,
                    ).squeeze(0)
            elif fill_mode == 'zeros':
                inp[..., bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = 0.0
            elif fill_mode == 'ones':
                inp[..., bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = 1.0 if inp.max() <= 1.0 else 255.0
            else:
                raise AttributeError(f'[Bbox::blur] Unrecognized filling mode: {fill_mode}')
        return inp

    def draw(self, frame: Tensor, **drawer_kwargs) -> Tensor:
        """Draw the bounding boxes on a frame.

        Parameters
        ----------
        frame: Tensor
            The frame tensor, with shape [3, height, width].
        drawer_kwargs: Any
            The arguments to pass to the drawer function.

        Returns
        -------
        torch.Tensor
            The frame with the bounding boxes drawn on it.
        """
        needs_norm = frame.is_floating_point() and frame.max() <= 1.0 and frame.min() >= 0
        if needs_norm:
            frame = frame.mul(255).add_(0.5).clamp_(0, 255).byte()
        out = torchvision.utils.draw_bounding_boxes(
            image=frame,
            boxes=self.xyxy.view(-1, 4),
            **drawer_kwargs
        )
        if needs_norm:
            return out.float().div_(255.0)
        return out

    def isolate(self, frame: Tensor, other_bboxes: List[Bbox], restore_bbox: Optional[Bbox] = None, fill_mode: str = 'interpolate') -> Tensor:
        orig_frame = frame.clone()
        for other_bbox in other_bboxes:
            if self.intersects(other_bbox).item() and False:
                # bounding boxes overlap
                #   - find all sub-slices

                #   - blur each of them separately
                pass
            else:
                # no overlap --> just blur inside the other bbox
                other_bbox.blur(frame, inplace=True, fill_mode=fill_mode)
            #   - revert original foreground
            if restore_bbox is not None:
                if restore_bbox.masks is None:
                    # restore the entire bbox
                    restore_xyxy = restore_bbox.bbox_xyxy.flatten().tolist()
                    frame[..., restore_xyxy[1]:restore_xyxy[3], restore_xyxy[0]:restore_xyxy[2]] = \
                        orig_frame[..., restore_xyxy[1]:restore_xyxy[3], restore_xyxy[0]:restore_xyxy[2]]
                else:
                    # restore the segmentation mask
                    mask = restore_bbox.masks[0]
                    frame[..., mask] = orig_frame[..., mask]
        return torch.stack([
            frame[..., bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]
            for bbox_xyxy in self.bbox_xyxy.int()
        ], dim=0)

    @property
    def area(self) -> Tensor:
        """Computes the area of the self bounding boxes.

        Returns
        -------
        Tensor
            The area of the bounding boxes. The shape is (N,).
        """
        return torch.multiply(self.widths, self.heights)

    @property
    def xyxy(self) -> Tensor:
        """Returns the bounding boxes in XYXY format. This corresponds to coordinates of the top left and bottom right
        corners of the bounding boxes.

        Returns
        -------
        Tensor
            The bounding boxes in XYXY format.
        """
        return self.integerify(self.bbox_xyxy)

    @property
    def xywh(self) -> Tensor:
        """Returns the bounding boxes in XYWH format. This corresponds to coordinates of the top left corner and the
        width and height of the bounding boxes.

        Returns
        -------
        Tensor
            The bounding boxes in XYWH format.
        """
        return torch.concat((self.bbox_xyxy[:, :2], self.widths[:, None], self.heights[:, None]), dim=-1)

    @property
    def cxywh(self) -> Tensor:
        """Returns the bounding boxes in CXYWH format. This corresponds to coordinates of the center of the bounding
        boxes and the width and height of the bounding boxes.

        Returns
        -------
        Tensor
            The bounding boxes in CXYWH format.
        """
        return torch.concat((self.centers, self.widths[:, None], self.heights[:, None]), dim=-1)

    @property
    def cxys(self) -> Tensor:
        """Returns the bounding boxes in center-scale format. This corresponds to coordinates of the center of the
        bounding boxes and the maximum between the width and the height of the bounding boxes.

        Returns
        -------
        Tensor
            The bounding boxes in CXYS format, shape (*, 3)
        """
        cxywh = self.cxywh
        return torch.concat([cxywh[..., :2], torch.maximum(cxywh[..., [-2]], cxywh[..., [-1]])], dim=-1)

    @property
    def cxywhn(self) -> Tensor:
        """Returns the bounding boxes in CXYWHN format. This corresponds to coordinates of the center of the bounding
        boxes and the width and height of the bounding boxes normalized to the image size.

        Returns
        -------
        Tensor
            The bounding boxes in CXYWHN format.
        """
        bbox_cxywhn = self.cxywh
        bbox_cxywhn[:, 0:4:2] /= self.frame_wh[..., [0]]
        bbox_cxywhn[:, 1:4:2] /= self.frame_wh[..., [1]]
        return bbox_cxywhn

    @property
    def cxywhn_m1p1(self) -> Tensor:
        """Returns the bounding boxes in CXYWHN format where the centers are normalized to [-1, 1].

        Returns
        -------
        Tensor
            The bounding boxes in CXYWHN format with cxy in the range [-1, 1].
        """
        bbox_cxywhn = self.cxywhn
        scale = torch.maximum(self.frame_wh[..., [0]], self.frame_wh[..., [1]])
        aspect_ratio_xy = torch.hstack([self.frame_wh[..., [0]] / scale, self.frame_wh[..., [1]] / scale])  # (B, 2)
        bbox_cxywhn[..., :2] = 2 * bbox_cxywhn[..., :2] - aspect_ratio_xy
        return bbox_cxywhn

    # noinspection PyUnresolvedReferences
    def clip(self, maintain_aspect_ratio: bool = True) -> Bbox:
        """Clips the bounding boxes to the image size.

        Returns
        -------
        Tensor
            The clipped bounding boxes.
        """
        bbox_out = self.xyxy.clone()
        nan_mask = self.xyxy.isnan()
        inf_mask = self.xyxy.isinf()
        if maintain_aspect_ratio:
            # move the bbox so as is inside the image
            bbox_out[bbox_out[:, 0] < 0, 0:4:2] -= bbox_out[bbox_out[:, 0] < 0, 0][:, None]
            bbox_out[bbox_out[:, 1] < 0, 1:4:2] -= bbox_out[bbox_out[:, 1] < 0, 1][:, None]
            if self.frame_wh is not None:
                bbox_out[bbox_out[:, 2] > self.frame_wh[:, 0], 0:4:2] -= (
                        bbox_out[bbox_out[:, 2] > self.frame_wh[:, 0], 2][:, None] -
                        self.frame_wh[bbox_out[:, 2] > self.frame_wh[:, 0], 0][:, None]
                )
                bbox_out[bbox_out[:, 3] > self.frame_wh[:, 1], 1:4:2] -= (
                        bbox_out[bbox_out[:, 3] > self.frame_wh[:, 1], 3][:, None] -
                        self.frame_wh[bbox_out[:, 3] > self.frame_wh[:, 1], 1][:, None]
                )
        # clip the bbox to the image (hoping that the bbox is inside the image)
        if self.frame_wh is not None:
            bbox_out[..., 0:4:2].clamp_(torch.zeros_like(self.frame_wh[..., [0]]), self.frame_wh[..., [0]])
            bbox_out[..., 1:4:2].clamp_(torch.zeros_like(self.frame_wh[..., [1]]), self.frame_wh[..., [1]])
        bbox_out = bbox_out.float()
        bbox_out[nan_mask] = float('nan')
        bbox_out[inf_mask] = float('inf')
        return Bbox(bbox_out, BboxFormat.XYXY, self.frame_wh, masks=self.masks)

    def pad(self, pad_size: int) -> Bbox:
        """Pads the bounding boxes by the given padding.

        Parameters
        ----------
        pad_size : int
            The padding to apply.

        Returns
        -------
        Tensor
            The padded bounding boxes.
        """
        bbox_out = self.cxywh.clone()
        bbox_out[:, 2:4] += 2 * pad_size
        return Bbox(bbox_out, BboxFormat.CXYWH, self.frame_wh, masks=self.masks).clip()

    def resize(self, scale: Optional[Sequence[float] or float] = None, aspect_ratio: Optional[float] = None) -> Bbox:
        """Resizes the bounding boxes to the given size.

        Parameters
        ----------
        scale : Sequence[float] or float, optional
            The scale to resize the bounding boxes to, by default None (no rescale).
        aspect_ratio : float, optional
            The aspect ratio of the bounding boxes, by default None (no aspect ratio change).

        Returns
        -------
        Tensor
            The resized bounding boxes.
        """
        new_widths = self.widths
        new_heights = self.heights
        if scale is not None:
            if isinstance(scale, float):
                scale = (scale, scale)
            new_widths = self.widths * scale[0]
            new_heights = self.heights * scale[1]
        if aspect_ratio is not None:
            new_widths = torch.where(new_widths > new_heights * aspect_ratio,
                                     new_widths,
                                     new_heights * aspect_ratio)
            # noinspection PyTypeChecker
            new_heights = torch.where(new_widths > new_heights * aspect_ratio,
                                      new_widths / aspect_ratio,
                                      new_heights)
        bbox_out = torch.concat((self.centers, new_widths[:, None], new_heights[:, None]), dim=-1)
        return Bbox(bbox_out, BboxFormat.CXYWH, self.frame_wh, masks=self.masks) \
            .clip(maintain_aspect_ratio=aspect_ratio is not None)

    @classmethod
    def from_mmdet(cls, det) -> Bbox:
        # noinspection PyUnresolvedReferences
        bboxes = det.pred_instances['bboxes'].detach().cpu()
        masks = det.pred_instances['masks'].detach().cpu() if 'masks' in det.pred_instances.keys() else None
        return cls(bboxes, BboxFormat.XYXY, det.metainfo['ori_shape'][::-1], masks=masks)

    @classmethod
    def from_pose2d(cls, keypoints: Tensor, scores: Tensor, score_threshold: float = 0.2, **kwargs) -> Bbox:
        """
        Initialize bbox from keypoints, by first fitting a bounding to the detected keypoints with score higher than
        the given threshold.

        Parameters
        ----------
        keypoints: Tensor
            The input keypoints, shape (batch_size, num_keypoints, 2).
        scores: Tensor
            The corresponding confidence scores, shape (batch_size, num_keypoints).
        score_threshold: float, optional
            The threshold for computing the keypoint mask. Below this, the keypoint is considered hidden and not used
            for the bbox calculation.
        """
        # compute keypoint mask
        mask = (scores > score_threshold).squeeze(-1)
        kps_masked = torch.where(mask.unsqueeze(-1), keypoints, torch.full_like(keypoints, float('nan')))
        valid_x = (~torch.isnan(kps_masked[:, :, 0])).any(dim=1)
        valid_y = (~torch.isnan(kps_masked[:, :, 1])).any(dim=1)
        # compute bounding box from keypoints
        x_min = torch.where(
            valid_x,
            kps_masked[:, :, 0].masked_fill(torch.isnan(kps_masked[:, :, 0]), float('inf')).min(dim=1).values,
            float('nan')
        )
        y_min = torch.where(
            valid_y,
            kps_masked[:, :, 1].masked_fill(torch.isnan(kps_masked[:, :, 1]), float('inf')).min(dim=1).values,
            float('nan')
        )
        x_max = torch.where(
            valid_x,
            kps_masked[:, :, 0].masked_fill(torch.isnan(kps_masked[:, :, 0]), float('-inf')).max(dim=1).values,
            float('nan')
        )
        y_max = torch.where(
            valid_y,
            kps_masked[:, :, 1].masked_fill(torch.isnan(kps_masked[:, :, 1]), float('-inf')).max(dim=1).values,
            float('nan')
        )
        bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        return Bbox(bboxes, fmt=BboxFormat.XYXY, **kwargs).clip(maintain_aspect_ratio=False)


class Dict(dict):
    """Dict Class:
    A subclass of the built-in dict class that provides additional functionality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __add__(self, other: Dict) -> Dict:
        return self.merge(other)

    def __sub__(self, other: Dict) -> Dict:
        return self.besides(*other.keys())

    def besides(self, *keys) -> Dict:
        """Returns a new Dict with the given keys removed. The original Dict will not be modified.

        Parameters
        ----------
        keys: str
            The keys to remove.

        Returns
        -------
        Dict
        """
        return Dict({k: v for k, v in self.items() if k not in keys})

    def flatten(self, delimiter: str = '.') -> Dict:
        def _flatten_dict_gen(d, parent_key, sep):
            for k, v in d.items():
                k = str(k)
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, MutableMapping):
                    yield from _flatten_dict(v, new_key, sep=sep).items()
                else:
                    yield new_key, v

        def _flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
            return dict(_flatten_dict_gen(d, parent_key, sep))

        return Dict(_flatten_dict(self, sep=delimiter))

    def filter(self, filter_fn: Optional[Callable] = None, dtype: Optional[Union[Type, Sequence[Type]]] = None) -> Dict:
        out = self
        if dtype is not None:
            out = {k: v for k, v in out.items() if isinstance(v, dtype)}
        if filter_fn is not None:
            out = {k: v for k, v in out.items() if filter_fn(k, v)}
        return Dict(out)

    def index_select(self, index: int, recursive: bool = True) -> Dict:
        """
        Returns where at each value only the passed index has been retained.

        Parameters
        ----------
        index: int
            The index to select values based upon.
        recursive: bool
            If true, it will recursively select values from indexable data.

        Returns
        -------
        Dict
        """
        out = Dict()
        for k, v in self.items():
            if isinstance(v, dict) and recursive:
                out[k] = Dict(v).index_select(index, recursive=recursive)
            elif isinstance(v, (Tensor, np.ndarray, Sequence)):
                out[k] = v[index]
            else:
                out[k] = v
        return out

    def merge(self, *others) -> Dict:
        """Returns a new Dict with the given Dicts merged. If there are duplicate keys, the last dictionary's value will
        be used. The original Dicts will not be modified.

        Parameters
        ----------
        others: Dict
            The Dicts to merge.

        Returns
        -------
        Dict
        """
        result = self.copy()
        for other in others:
            result.update(other)
        return Dict(result)

    def mix(self, *others: dict or Dict, mix_keys: Optional[Sequence[str]] = None) -> Dict:
        result = dict(**self)
        assert mix_keys is None or len(mix_keys) == len(others) + 1, \
            f'Length of mix_keys ({len(mix_keys)}) must be equal to the number of dicts + 1 ({len(others) + 1})'
        for di, d in enumerate([self] + list(others)):
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                if k not in result.keys():
                    result[k] = v
                else:
                    if mix_keys is None:
                        merge_lists = isinstance(result[k], list) and isinstance(v, list)
                        if len(result[k]) > 0 and len(v) > 0 and type(result[k][0]) is not type(v[0]):
                            merge_lists = False
                        if merge_lists:
                            result[k].extend(v)
                        else:
                            result[k] = (result[k], v)
                    else:
                        if not isinstance(result[k], dict) or mix_keys[0] not in result[k].keys():
                            result[k] = {
                                mix_keys[0]: result[k],
                                mix_keys[di]: v
                            }
                        else:
                            result[k][mix_keys[di]] = v
        return Dict(result)

    def only(self, *keys, remove_prefix: bool = False, prefix_delimiter: str = ':', recursive: bool = False) -> Dict:
        """Returns a new Dict with only the given keys kept. The original Dict will not be modified.

        Parameters
        ----------
        keys: str
            The keys to keep.
        remove_prefix: bool, optional
            If set to True, all the prefixes will be removed first. Prefixes are parts of the keys starting from the
            beginning and ending with the first found `prefix_delimiter`.
        prefix_delimiter: str, optional
            The delimiter with which the prefix ends.
        recursive: bool, optional
            If set to True, it will recursively traverse the dict to keep only keys.

        Returns
        -------
        Dict
        """
        if remove_prefix and all(':' in k for k in self.keys()):
            return self.remove_prefix(delimiter=prefix_delimiter).only(*keys, remove_prefix=False)
        if recursive and not any(str(ok) in self.keys() for ok in keys) \
                and any(isinstance(v, (dict, Dict)) for v in self.values()):
            return Dict({k: Dict(v).only(*keys, recursive=True) if isinstance(v, (dict, Dict)) else v
                         for k, v in self.items()})
        out = Dict()
        for only_key in keys:
            if isinstance(only_key, str) and only_key in self.keys():
                out[only_key] = self[only_key]
            elif isinstance(only_key, Callable):
                out[getattr(only_key, '__name__', repr(only_key))] = only_key(self)
        return out

    def remove_prefix(self, delimiter: str = ':') -> Dict:
        out = Dict()
        for k, v in self.items():
            if delimiter in k:
                _, new_key = k.split(delimiter, maxsplit=1)
            else:
                new_key = k
            out[new_key] = v
        return out

    def replace(self, key: str, value: dict or Any) -> dict:
        # LOOKAT: Add support for nested keys. At the moment, only the first value of the given key is used.
        if type(value) is dict and re.search(list(value.keys())[0], key):
            value = value[list(value.keys())[0]]

        # If key is not in dict, search for it in the values of the dict
        inp = self
        if key not in self.keys():
            for key_, value_ in self.items():
                if isinstance(value_, dict) and key in value_.keys():
                    inp = self[key_]
                    break
            else:
                return Dict(**self)

        for k, v in value.items():
            if k in inp[key].keys():
                if type(v) is dict and type(inp[key]) is dict:
                    inp[key] = Dict(**inp[key]).replace(k, v)
                else:
                    inp[key][k] = v
            elif type(inp[key][list(inp[key].keys())[0]]) is dict and k in inp[key][list(inp[key].keys())[0]].keys():
                inp[key][list(inp[key].keys())[0]][k] = v
            else:
                raise AttributeError(f'\t[Dict::replace] {key}-{value}: Not found in dict. ({inp})')
        return Dict(**inp)

    def sort(self, recursive: bool = True) -> Dict:
        out = {}
        for k, v in sorted(self.items()):
            if isinstance(v, (dict, Dict)) and recursive:
                out[k] = Dict(v).sort(recursive=recursive)
            else:
                out[k] = v
        return Dict(out)

    def split_batch(self, recursive: bool = True) -> Dict:
        out = {}
        for k, v in self.items():
            if isinstance(v, (dict, Dict)) and recursive:
                out[k] = Dict(v).split_batch(recursive=recursive)
            else:
                out[k] = list(map(partial(torch.squeeze, dim=0), torch.split(v, split_size_or_sections=1, dim=0)))
        return Dict(out)

    def tensorify(self, *args, **kwargs) -> Dict:
        return Nutrients.tensorify(self, *args, **kwargs)

    def to_batch_of_dicts(self, return_list: bool = False) -> Union[Dict, List]:
        out = {k: [] for k in self.keys()}
        for k in self.keys():
            d = self[k]
            if isinstance(d, (dict, Dict)):
                total = len(next(iter(d.values())))
                for i in range(total):
                    out[k].append({dk: dv[i] for dk, dv in d.items()})
            elif isinstance(d, Tensor):
                out[k] = [dd.squeeze(0) for dd in torch.split(d, split_size_or_sections=1, dim=0)]
            else:
                out[k] = d
        if not return_list:
            return Dict(out)
        batch_size = len(out[list(out.keys())[0]])
        assert all(len(out[k]) == batch_size for k in out.keys())
        return [
            {k: v[_] for k, v in out.items()}
            for _ in range(batch_size)
        ]

    def to_dict_of_batches(self) -> Dict:
        out = {}
        for k in self.keys():
            d = self[k]
            if isinstance(d, Sequence) and (isinstance(d[0], dict) or hasattr(d[0], 'as_dict')):
                if hasattr(d[0], 'as_dict'):
                    d = [dd.as_dict() for dd in d]
                out[k] = {kk: (torch.stack if isinstance(d[0][kk], Tensor) else list)([dd[kk] for dd in d])
                          for kk in d[0].keys()}
            elif isinstance(d, Sequence) and isinstance(d[0], Tensor):
                # noinspection PyTypeChecker
                out[k] = torch.stack(d)
            else:
                out[k] = d
        return Dict(out)

    def to_list(self) -> list:
        def listify(inp):
            if isinstance(inp, (dict, Dict)):
                out = [None] * len(inp)
                for index, value in zip(map(int, inp.keys()), inp.values()):
                    out[index] = listify(value)
            elif hasattr(inp, 'tolist'):
                out = inp.tolist()
            else:
                out = inp
            return out

        return listify(self)

    def unflatten(self, delimiter: str = '.', base=None) -> Dict:
        """Convert any keys containing dotted paths to nested dicts:
            - no expansion if delimiter is not present in keys
            - recursive path unflattening
            - merging of unflattened dicts
            - insertion-order overwrites
        """
        if base is None:
            base = {}
        for key, value in self.items():
            root = base
            if delimiter in key:
                *parts, key = key.split(delimiter)
                for part in parts:
                    root.setdefault(part, {})
                    root = root[part]
            if isinstance(value, dict):
                value = dict(**Dict(value).unflatten(root.get(key, {})))
            root[key] = value
        return Dict(base)


# noinspection DuplicatedCode
class Nutrients:
    """Nutrients Class:
    This class implements a collection of useful functions for PyTorch.
    """

    LEAF_MODULE_TYPES = []

    @classmethod
    def best_coords(cls,
                    img: Tensor,
                    n_points: int,
                    n_grids: int = 1,
                    gradient_pooling: Optional[int] = None,
                    gradient_processor: Optional[Callable] = None) -> Tensor:
        # Get gradient from images
        gradient = Nutrients.gradient(img, pooling_kernel=gradient_pooling, pooling_stride=gradient_pooling)
        if gradient_processor is not None:
            gradient = gradient_processor(gradient, pooling_kernel=gradient_pooling, pooling_stride=gradient_pooling)
        grids_list = []
        probs = gradient.abs()
        for grid_i in range(n_grids):
            best_flattened_coords = torch.stack([
                torch.multinomial(probs[i].flatten(1), n_points, replacement=False).squeeze()
                for i in range(gradient.shape[0])
            ], dim=0)
            # noinspection PyTypeChecker
            grids_list.append(
                torch.stack(
                    Nutrients.unravel_index(best_flattened_coords, gradient.shape[-2:]),
                    dim=-1
                )
            )
        coords = torch.stack(grids_list, dim=1).long()
        # cls.print_coords(coords.squeeze(), gradient, 'neuron')
        return coords

    @classmethod
    def collect_weights(cls, module: nn.Module, other_state: OrderedDict or Weights) -> Tuple[dict, dict, dict, dict]:
        """Collects the weights for module from `from_state` found_state dictionary.
        `from_state` is the found_state dictionary from a similar module, e.g. a module with the different name and
        parameters, but with same weights. The method first examines both dictionaries sequentially assigning the
        weights from the `from_state` to the corresponding names in the `module` found_state dictionary. If the weights
        in the `from_state` are exhausted, the remaining weights in the `module` state dictionary are left unchanged.

        Parameters
        ----------
        module : nn.Module
            The module to collect the weights for.
        other_state : OrderedDict or Weights
            The found_state dictionary to collect the weights from.

        Returns
        -------
        OrderedDict[str, Tensor], dict[str, List[int]], dict[str, List[int]]
            The collected found_state dictionary, the non-found keys and the unused keys.
        """
        is_file_path = False
        file_path = None
        if hasattr(other_state, "get_state_dict"):
            other_state = other_state.get_state_dict(progress=False)
        elif hasattr(other_state, 'url'):
            other_state = load_state_dict_from_url(other_state.url, progress=False)
        elif isinstance(other_state, (str, Path)) and os.path.exists(other_state):
            if Path(other_state).suffix in ['.npy', '.npz']:
                other_state = np.load(other_state)
                return other_state, {}, {}, {}
            is_file_path, file_path = True, copy.deepcopy(other_state)
            other_state = torch.load(other_state, map_location='cpu')
            if 'state_dict' in other_state.keys():
                other_state = other_state['state_dict']
            elif 'model' in other_state.keys() and ('optimizer' in other_state.keys() or 'epoch' in other_state.keys()):
                other_state = other_state['model']
        else:
            raise ValueError(f"[Nutrients::collect_weights] Invalid other_state type: "
                             f"{type(other_state)}, {other_state}")
        module_state = module.state_dict()

        # Try to remove common prefix
        dict_updated = False
        #   - from other_state
        while True:
            other_prefix = list(other_state.keys())[0].split('.')[0]
            if '.' not in other_prefix:
                break  # it was the last one
            other_prefix += '.'
            if (all(k.startswith(other_prefix) for k in other_state.keys()) and
                    all(not k.startswith(other_prefix) for k in module_state.keys())):
                # remove prefix from other_state
                other_state = {k[len(other_prefix):]: v for k, v in other_state.items()}
                dict_updated = True
            else:
                break
        #   - from module_state
        module_prefix = list(module_state.keys())[0].split('.')[0] + '.'
        if (all(k.startswith(module_prefix) for k in module_state.keys()) and
                not all(k.startswith(module_prefix) for k in other_state.keys())):
            # add prefix to other_state
            other_state = {
                (k if k.startswith(module_prefix) else f'{module_prefix}{k}'): v for k, v in other_state.items()
            }
            dict_updated = True
        if is_file_path and dict_updated:
            torch.save(other_state, file_path)

        # Get name to shape mapping for both states
        other_name_to_shape = {k: v.shape for k, v in other_state.items() if hasattr(v, 'shape')}
        module_name_to_shape = {k: v.shape for k, v in module_state.items() if hasattr(v, 'shape')}

        # Get the names of the parameters in the module that are not in the other_state
        found_state, keys_mapping = dict(), dict()
        found_keys, other_found_keys = set(), set()
        for (name, param), (other_name, other_param) in zip(list(module_state.items()), list(other_state.items())):
            if name == other_name or name in other_state.keys():
                found_state[name] = other_state[name]
                found_keys.add(name)
                other_found_keys.add(name)
                keys_mapping[name] = name
                del module_name_to_shape[name]
                del other_name_to_shape[name]

        for name, shape in list(module_name_to_shape.items()):
            # Find all other shapes that match the shape
            other_names = [k for k, v in other_name_to_shape.items() if v == shape]
            if len(other_names) > 0:
                # Get the closest name to the module name using the SequenceMatcher ratio
                other_name = max(other_names, key=lambda on: SequenceMatcher(None, name, on).ratio())
                # Mark selected name as found
                found_keys.add(name)
                other_found_keys.add(other_name)
                found_state[name] = other_state[other_name]
                keys_mapping[name] = other_name
                del module_name_to_shape[name]
                del other_name_to_shape[other_name]

        return found_state, \
            {k: list(v.shape) for k, v in module_state.items() if k not in found_keys and hasattr(v, 'shape')}, \
            {k: list(v.shape) for k, v in other_state.items() if k not in other_found_keys and hasattr(v, 'shape')}, \
            keys_mapping

    @classmethod
    def coords_grid(cls, batch_size: int, h: int, w: int, device: str = 'cpu', fmt: str = 'ij'):
        """Returns a grid of coordinates tensor with shape (batch_size, 2, h, w) for the given `batch_size` and image
        size (`h` and `w`). The coordinates are in the range [0, h/w] and are in the format (y,x) or (x, y) depending
        on the input argument. The returned tensor is on the given `device`.

        Parameters
        ----------
        batch_size : int
            The batch size.
        h : int
            The height of the image.
        w : int
            The width of the image.
        device: str, optional
            The device to put the tensor on, by default 'cpu'.
        fmt: str, optional
            If "xy" then the returned vector has width-wise indexing first. Otherwise, it is the typical H,W in PyTorch.
        """
        device = torch.device(device)
        coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing=fmt)
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch_size, 1, 1, 1)

    @classmethod
    def correlate(cls, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        b, c, h, w = tensor1.shape
        return torch.bmm(tensor1.flatten(2).transpose(1, 2), tensor2.flatten(2)).reshape(b, h, w, h, w) / math.sqrt(c)

    @classmethod
    def correlate_pack(cls, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        corr = cls.correlate(tensor1, tensor2)
        return corr.flatten(1, -3)

    @classmethod
    def correlate_unpack(cls, corr_packed: Tensor) -> Tensor:
        assert corr_packed.ndim <= 4, f'Packed correlation cannot be more that 4D. Got {corr_packed.ndim}D tensor.'
        scale_factor = int(np.sqrt(corr_packed.shape[1] / np.array(corr_packed.shape[2:]).prod()))
        return corr_packed.view(corr_packed.shape[0],
                                scale_factor * corr_packed.shape[2],
                                scale_factor * corr_packed.shape[3],
                                *corr_packed.shape[2:])

    @classmethod
    def enfold(cls, module: nn.Module, weights: Union[OrderedDict, Weights]) -> int:
        """Load weights to a module, ignoring missing and unused keys during the process.
        This is useful when loading weights from a pre-trained model to a model with a different architecture.

        Parameters
        ----------
        module: nn.Module
            module to load weights to
        weights: OrderedDict or Weights
            weights to be loaded

        Returns
        -------
        int
            number of parameters found in state dict
        """
        state, non_found, unused, _ = cls.collect_weights(module, weights)
        if len(non_found) > 0:
            log(f'[{cls.__name__}::enfold] Weights not found: {non_found}', 'warning')
        if len(unused) > 0:
            log(f'[{cls.__name__}::enfold] Unused weights: {unused}', 'warning')
        params_count = sum(p.numel() for p in state.values())
        module.load_state_dict(state, strict=False)
        return params_count

    @classmethod
    def flatten_module(cls, module: nn.Module) -> OrderedDict[str, nn.Module]:
        return OrderedDict([
            (name, submodule)
            for name, submodule in module.named_modules()
            if len(list(submodule.children())) == 0
        ])

    @classmethod
    def flatten_module_pp(cls, module: nn.Module) -> None:
        pprint.pprint(dict(cls.flatten_module(module)))

    @staticmethod
    def gradient(img: Tensor, pooling_kernel: Optional[int] = None,
                 pooling_stride: Optional[int] = None, normalize: bool = True) -> Tensor:
        if img.ndim == 3 or (img.ndim == 4 and img.shape[1] != 1):
            img = img.unsqueeze(1)
        gray = ((img + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[..., :-1, 1:] - gray[..., :-1, :-1]  # horizontal intensity gradient
        dy = gray[..., 1:, :-1] - gray[..., :-1, :-1]  # vertical intensity gradient
        g = torch.sqrt(dx ** 2 + dy ** 2)
        if pooling_kernel is not None:
            g = F.avg_pool2d(g, pooling_kernel, pooling_stride)  # average pooling to make the same size as fmap
        if normalize:
            g_shape = g.shape
            gf = g.flatten(start_dim=1)
            gn = (gf - gf.min(1, keepdim=True)[0]) / (gf.max(1, keepdim=True)[0] - gf.min(1, keepdim=True)[0])
            return gn.view(*g_shape)
        return g

    @classmethod
    def grid_sample(cls, img: Tensor, absolute_grid: Tensor, **grid_sample_kwargs) -> Tensor:
        """Same as torch's grid_sample, with absolute pixel coordinates instead of normalized coordinates.

        Parameters
        ----------
        img
        absolute_grid

        Returns
        -------
        Tensor
        """
        h, w = img.shape[-2:]
        xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (w - 1) - 1
        # Adding condition if h > 1 to enable this function be reused in raft-stereo
        if h > 1:
            ygrid = 2 * ygrid / (h - 1) - 1
        normalized_grid = torch.cat([xgrid, ygrid], dim=-1)
        if 'align_corners' not in grid_sample_kwargs.keys():
            grid_sample_kwargs['align_corners'] = True
        return F.grid_sample(img, normalized_grid, **grid_sample_kwargs)

    @classmethod
    def stack(cls, tensors: List[Tensor], dim: int = 0) -> Tensor:
        first_shape = list(tensors[0].shape)
        if all(first_shape == list(avi.shape) for avi in tensors):
            joined_tensors = torch.stack(tensors, dim=dim)
        else:
            joined_tensors = torch.cat(tensors, dim=dim).unsqueeze(dim)
        return joined_tensors

    @classmethod
    def match_strings(cls, list1: List[str], list2: List[str], sort: bool = False) -> DictT[int, int]:
        biglist = list(enumerate(list1)) + list(enumerate(list2))
        biglist.sort(key=lambda x: x[1])
        matches = [(biglist[i][0], biglist[i + 1][0]) for i in range(len(biglist) - 1) if
                   biglist[i][1] == biglist[i + 1][1]]
        matches = dict(matches)
        return matches if not sort else dict(sorted(matches.items(), key=lambda p: p[0]))

    @classmethod
    def tensorify(cls, inp: Any, device: Union[str, torch.device] = 'cpu', as_parameter: bool = False,
                  requires_grad: bool = False, clone: bool = False, detach: bool = False) -> Any:
        """
        Maps all input elements to tensor
        """
        if inp is None or isinstance(inp, (int, float, bool, str, enum.Enum)):
            return inp
        if isinstance(inp, nn.Parameter):
            if as_parameter:
                inp.requires_grad = requires_grad
                return inp
            inp = inp.data
        if isinstance(inp, Tensor):
            t = inp
            if detach:
                t = t.detach()
            if clone:
                t = t.clone()
            t = t.to(device)
            if as_parameter:
                t = nn.Parameter(t.float(), requires_grad=requires_grad)
            return t
        if isinstance(inp, np.ndarray):
            return torch.from_numpy(inp).to(device)
        if isinstance(inp, Sequence):
            return type(inp)([
                cls.tensorify(i, device=device, as_parameter=as_parameter, clone=clone, requires_grad=requires_grad,
                              detach=detach)
                for i in inp
            ])
        if hasattr(inp, 'as_dict'):
            inp = inp.as_dict()
        if isinstance(inp, (dict, Dict)):
            # @formatter:off
            return type(inp)(**{
                k: cls.tensorify(v, device=device, as_parameter=as_parameter, clone=clone, requires_grad=requires_grad,
                                 detach=detach)
                for k, v in inp.items()
            })
            # @formatter:on
        raise TypeError(f'[Nutrients::tensorify] Type of input is not supported: {inp}')

    @classmethod
    def tensorize(cls,
                  x: Any,
                  dtype: Optional[torch.dtype] = None,
                  device: torch.types.Device = None,
                  requires_grad: bool = False) -> Tensor or Iterable[Tensor]:
        """Convert input to tensor. If the input is a tensor, then we just return it. If the input is an iterable, then
        we convert each item in the iterable to tensor and return the resulting iterable.

        Parameters
        ----------
        x:  Any
            input value
        dtype:  torch.dtype or None
            data type of the resulting tensor. If None, then the data type of the input is used. (default = None)
        device: torch.types.Device
            device of the resulting tensor. If None, then the device of the input is used. (default = None)
        requires_grad:  bool
            whether the resulting tensor requires gradient. (default = False)

        Returns
        -------
        Tensor or Iterable[Tensor]
        """
        if isinstance(x, (str, bool)):
            return x
        if isinstance(x, dict) or isinstance(x, OrderedDict):
            return {k: cls.tensorize(v) for k, v in x.items()}
        if isinstance(x, tuple):
            return (cls.tensorize(item) for item in x)
        if isinstance(x, Tensor):
            if dtype is not None:
                x = x.type(dtype)
            if device is not None:
                x = x.to(device)
            return x

        x_orig = x
        if 'scipy.sparse' in str(type(x)):
            x = x.toarray()
        elif isinstance(x, list):
            x = np.array(x)
        elif hasattr(x, 'r'):
            x = x.r
        if isinstance(x, np.ndarray):
            # FIX: pytorch cannot convert unsigned integers to tensors
            if x.dtype in [np.uint8, np.uint16]:
                x = x.astype(np.int32)
            elif x.dtype in [np.uint32, np.uint64]:
                x = x.astype(np.int64)
        try:
            return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
        except TypeError:
            return x_orig

    @classmethod
    def tuplenize(cls, x: Any, n: int) -> tuple:
        """Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
        Otherwise, we will make a tuple of length n, all with value of x.

        References:
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

        Parameters
        ----------
        x:  Any
            input value
        n:  int
            length of the resulting tuple

        Returns
        -------
        tuple
        """
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    @classmethod
    def padding_from_kernel(cls, kernel: KernelType, dilation: KernelType = 1) -> KernelType:
        """Get the padding needed to retain spatial dimensions of the input.

        References:
            - torchvision.utils._make_ntuple()

        Parameters
        ----------
        kernel: int or Tuple[int, int]
                kernel size as in nn.Conv2d()
        dilation:   int or Tuple[int, int]
                    dilation size as in nn.Conv2d()

        Returns
        -------
        int or Tuple[int, int]
        """
        if isinstance(kernel, int) and isinstance(dilation, int):
            padding = (kernel - 1) // 2 * dilation
        else:
            n = len(kernel) if isinstance(kernel, Sequence) else len(dilation)
            kernel_size = cls.tuplenize(kernel, n=n)
            dilation = cls.tuplenize(dilation, n=n)
            padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(n))
        return padding

    @classmethod
    def place(cls, target: Tensor, needle: Any, h_min: Optional[int] = None, h_max: Optional[int] = None,
              w_min: Optional[int] = None, w_max: Optional[int] = None) -> Tensor:
        """Place the :arg:`needle` into the :arg:`target` at the specified location.

        Parameters
        ----------
        target: Tensor
            target tensor
        needle: Tensor
            needle tensor
        h_min: int or None
            minimum height index
        h_max: int or None
            maximum height index
        w_min: int or None
            minimum width index
        w_max: int or None
            maximum width index

        Returns
        -------
        Tensor
            The target tensor with needle placed at the specified location.
        """
        target = target.clone().detach()
        if h_min is None:
            h_min = 0
        if h_min < 0:
            h_min = target.shape[1] + h_min
        if h_max is None:
            h_max = target.shape[1]
        if h_max < 0:
            h_max = target.shape[1] + h_max
        if w_min is None:
            w_min = 0
        if w_min < 0:
            w_min = target.shape[2] + w_min
        if w_max is None:
            w_max = target.shape[2]
        if w_max < 0:
            w_max = target.shape[2] + w_max
        mask = torch.zeros(target.shape, device=target.device, dtype=torch.bool)
        mask[:, h_min:h_max, w_min:w_max] = True
        if isinstance(needle, str):
            needle = Str(needle).rgb()
            if target.max() > 150 and min(needle) >= 0.0 and max(needle) <= 1.0:
                needle = [int(_ * 255.0) for _ in needle]
        if not isinstance(needle, Tensor):
            needle = torch.tensor(needle, dtype=target.dtype, device=target.device)
        try:
            return torch.where(mask, target, needle)
        except RuntimeError:
            try:
                return torch.masked_scatter(target, mask, needle)
            except RuntimeError:
                needle = needle.repeat(*target.shape).view(*target.shape, *needle.shape)[0] \
                    .permute(*range(-needle.ndim, 0), *range(target.ndim - needle.ndim)) \
                    .contiguous()
                return torch.where(mask, needle, target)

    @staticmethod
    def print_coords(coords: Tensor, g: Tensor, title: str = 'best') -> None:
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle(title)
        gi = g.cpu().detach().numpy()
        # fig.set_title('Visualization of coordinates used for sampling points and patches')
        for i in range(2):
            for j in range(2):
                gip = PIL.Image.fromarray(np.flipud(gi[i * 2 + j][0] * 255).astype(np.uint8), 'L')
                ax = axs[i, j]
                ax.imshow(gip)
                ax.set_xlim(0, g.shape[-1])
                ax.set_ylim(0, g.shape[-2])
                ax.set_title(f"coords[{i * 2 + j}]")
                coord = coords[i * 2 + j].squeeze(0).detach().cpu().int().numpy()
                ax.plot(coord[:, 0], coord[:, 1], '*g')
        plt.savefig(f'coords_{title}.pdf')

    @classmethod
    def unravel_index(cls, index: Tensor, shape: torch.Size) -> Sequence[Tensor]:
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return out


FunctionT = TypeVar('FunctionT', bound=Callable)


class Reflector:
    KWARGS = '**'

    @classmethod
    def collect_args(cls, function: Type or Any, kwargs: dict,
                     return_unused: bool = False) -> Union[Dict, Tuple[Dict, Dict]]:
        """Returns a dict of the function's arguments, with the keys being the argument names and the values being the
        argument values. If the argument is a keyword argument, the value will be the string '**'. If the argument is
        not in the function's signature, it will be ignored.

        Parameters:
        -----------
        function: Type or Any
            The function to inspect.
        kwargs: dict
            The keyword arguments to the function.
        return_unused: bool, optional
            Set to True to return a dict containing the unused key-value pairs

        Returns:
        --------
        Dict or Tuple[Dict, Dict]
        """
        signature = cls.signature(function)
        if cls.KWARGS in signature.values():
            used = Dict(kwargs)  # return the entire dict since there are kwargs
        else:
            used = Dict({k: v for k, v in kwargs.items() if k in signature.keys()})
        if not return_unused:
            return used
        return used, Dict({k: v for k, v in kwargs.items() if k not in signature.keys()})

    @classmethod
    def partial(cls, function: FunctionT, kwargs: dict) -> FunctionT:
        """Returns a partial function with the given keyword arguments. If the function has keyword arguments, the dict
        will be filtered to only include the keyword arguments. If the function has no keyword arguments, the dict will
        be ignored.

        Parameters:
        -----------
        function: Type or Any
            The function to inspect and initialize.
        kwargs: dict
            The keyword arguments to the function.

        Returns:
        --------
        Type or Any
        """
        return partial(function, **cls.collect_args(function, kwargs))

    @classmethod
    def partial_mp(cls, function: FunctionT, **kwargs) -> FunctionT:
        """Returns a partial function with the given keyword arguments. If the function has keyword arguments, the dict
        will be filtered to only include the keyword arguments. If the function has no keyword arguments, the dict will
        be ignored.

        Parameters:
        -----------
        function: Type or Any
            The function to inspect and initialize.
        kwargs: dict
            The keyword arguments to the function.

        Returns:
        --------
        Type or Any
        """
        return partial(cls.mp, partial_function=partial(function, **cls.collect_args(function, kwargs)))

    @classmethod
    def mp(cls, args, partial_function: partial):
        return partial_function(**dict(zip(
            cls.signature(partial_function.func).besides(*partial_function.keywords.keys()).keys(),
            args)
        ))

    @classmethod
    def signature(cls, function: Type or Any, use_entire_parameter: bool = False) -> Dict:
        """Returns a dict of the function's signature, with the keys being the argument names and the values being the
        argument types. If the argument is a kwargs argument, the value will be the string '**'.

        Parameters:
        -----------
        function: Type or Any
            The function to inspect.

        Returns:
        --------
        Dict
        """
        if inspect.isclass(function):
            function = function.__init__
        sig = {}
        for name, parameter in inspect.signature(function).parameters.items():
            if str(parameter).startswith('**'):
                sig[name] = cls.KWARGS
            elif use_entire_parameter:
                sig[name] = parameter
            else:
                sig[name] = parameter.name if parameter.annotation is not inspect.Parameter.empty else None
        return Dict(sig)


SmplJointNames: DictT[str, Optional[DictT[int, str]]] = dict(
    smpl=None,
    smplh=None,
    smplx=None,
)
SmplSkeleton: DictT[str, Optional[List[Tuple[int, int]]]] = dict(
    smpl=None,
    smplh=None,
    smplx=None,
)

Pose2dInfoDict: Optional[DictT[str, DictT[str, Union[DictT[str, int], List[Tuple[int, int]]]]]] = None


# noinspection DuplicatedCode
def pose2d_info_dict() -> DictT[str, DictT[str, Union[DictT[int, str], List[Tuple[int, int]]]]]:
    """
    Augmented from ViT-Pose's joint dict (it uses COCO-17 format, like COCO but without neck keypoint)
    Original Source: https://github.com/gpastal24/ViTPose-Pytorch/blob/a9f8025ea676b529a1f498bc2c048c6286ca171b/src/vitpose_infer/pose_utils/pose_viz.py#L9
    """
    global Pose2dInfoDict
    if Pose2dInfoDict is None:
        log(f'[pose2d_info_dict] Loading joints dict.', level='debug')
        Pose2dInfoDict = {
            "mpii": {
                "keypoints": {
                    0: "right_ankle",
                    1: "right_knee",
                    2: "right_hip",
                    3: "left_hip",
                    4: "left_knee",
                    5: "left_ankle",
                    6: "pelvis",
                    7: "thorax",
                    8: "neck",
                    9: "head_top",
                    10: "right_wrist",
                    11: "right_elbow",
                    12: "right_shoulder",
                    13: "left_shoulder",
                    14: "left_elbow",
                    15: "left_wrist"
                },
                "skeleton": [
                    # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                    # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                    [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                    [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
                ]
            },
            "mpii14": {
                "keypoints": {
                    0: "right_ankle",
                    1: "right_knee",
                    2: "right_hip",
                    3: "left_hip",
                    4: "left_knee",
                    5: "left_ankle",
                    6: "right_wrist",
                    7: "right_elbow",
                    8: "right_shoulder",
                    9: "left_shoulder",
                    10: "left_elbow",
                    11: "left_wrist",
                    12: "neck",
                    13: "head",
                },
                "skeleton": [

                ]
            },
            "h36m": {
                'keypoints': {
                    0: 'pelvis',
                    1: 'left_hip',
                    2: 'left_knee',
                    3: 'left_ankle',
                    4: 'right_hip',
                    5: 'right_knee',
                    6: 'right_ankle',
                    7: 'torso',
                    8: 'neck',
                    9: 'nose',
                    10: 'head',
                    11: 'left_shoulder',
                    12: 'left_elbow',
                    13: 'left_wrist',
                    14: 'right_shoulder',
                    15: 'right_elbow',
                    16: 'right_wrist',
                },
                'skeleton': [

                ]
            },
            "coco17": {
                "keypoints": {
                    0: "nose",
                    1: "left_eye",
                    2: "right_eye",
                    3: "left_ear",
                    4: "right_ear",
                    5: "left_shoulder",
                    6: "right_shoulder",
                    7: "left_elbow",
                    8: "right_elbow",
                    9: "left_wrist",
                    10: "right_wrist",
                    11: "left_hip",
                    12: "right_hip",
                    13: "left_knee",
                    14: "right_knee",
                    15: "left_ankle",
                    16: "right_ankle",
                },
                "skeleton": [
                    [15, 13], [13, 11], [16, 14], [14, 12],
                    [11, 12], [5, 11], [6, 12], [5, 6],
                    [5, 7], [6, 8], [7, 9], [8, 10],
                    [1, 2], [0, 1], [0, 2], [1, 3],
                    [2, 4], [0, 5], [0, 6]
                ]
            },
            "coco18": {
                "keypoints": {
                    0: "nose",
                    1: "neck",
                    2: "right_shoulder",
                    3: "right_elbow",
                    4: "right_wrist",
                    5: "left_shoulder",
                    6: "left_elbow",
                    7: "left_wrist",
                    8: "right_hip",
                    9: "right_knee",
                    10: "right_ankle",
                    11: "left_hip",
                    12: "left_knee",
                    13: "left_ankle",
                    14: "right_eye",
                    15: "left_eye",
                    16: "right_ear",
                    17: "left_ear",
                },
                "skeleton": [
                    [10, 9], [9, 8], [8, 2], [2, 3],
                    [3, 4], [13, 12], [12, 11], [11, 5],
                    [5, 6], [6, 7], [2, 1], [5, 1],
                    [1, 0], [0, 14], [14, 16], [0, 15],
                    [15, 17]
                ]
            },
            "coco19": {
                "keypoints": {
                    0: "neck",
                    1: "nose",
                    2: "body_center",
                    3: "left_shoulder",
                    4: "left_elbow",
                    5: "left_wrist",
                    6: "left_hip",
                    7: "left_knee",
                    8: "left_ankle",
                    9: "right_shoulder",
                    10: "right_elbow",
                    11: "right_wrist",
                    12: "right_hip",
                    13: "right_knee",
                    14: "right_ankle",
                    15: "right_eye",
                    16: "left_eye",
                    17: "right_ear",
                    18: "left_ear",
                },
                "skeleton": None
            },
            "openpose": {
                "keypoints": {
                    0: "nose",
                    1: "neck",
                    2: "right_shoulder",
                    3: "right_elbow",
                    4: "right_wrist",
                    5: "left_shoulder",
                    6: "left_elbow",
                    7: "left_wrist",
                    8: "pelvis",
                    9: "right_hip",
                    10: "right_knee",
                    11: "right_ankle",
                    12: "left_hip",
                    13: "left_knee",
                    14: "left_ankle",
                    15: "right_eye",
                    16: "left_eye",
                    17: "right_ear",
                    18: "left_ear",
                    19: "left_big_toe",
                    20: "left_small_toe",
                    21: "left_heel",
                    22: "right_big_toe",
                    23: "right_small_toe",
                    24: "right_heel",
                },
                "skeleton": [
                    [1, 8], [1, 2], [1, 5], [2, 3],
                    [3, 4], [5, 6], [6, 7], [8, 9],
                    [9, 10], [10, 11], [8, 12], [12, 13],
                    [13, 14], [1, 0], [0, 15], [15, 17],
                    [0, 16], [16, 18], [14, 19], [19, 20],
                    [14, 21], [11, 22], [22, 23], [11, 24],
                ],
            },
            "body26fk": {
                "keypoints": {
                    0: 'pelvis',
                    1: 'left_hip',
                    2: 'right_hip',
                    3: 'torso',
                    4: 'left_knee',
                    5: 'right_knee',
                    6: 'neck',
                    7: 'left_ankle',
                    8: 'right_ankle',
                    9: 'left_big_toe',
                    10: 'right_big_toe',
                    11: 'left_small_toe',
                    12: 'right_small_toe',
                    13: 'left_heel',
                    14: 'right_heel',
                    15: 'nose',
                    16: 'left_eye',
                    17: 'right_eye',
                    18: 'left_ear',
                    19: 'right_ear',
                    20: 'left_shoulder',
                    21: 'right_shoulder',
                    22: 'left_elbow',
                    23: 'right_elbow',
                    24: 'left_wrist',
                    25: 'right_wrist'
                },
                "skeleton": [
                    [0, 3], [3, 6], [6, 0], [8, 5],
                    [5, 2], [2, 0], [2, 21], [21, 23],
                    [23, 25], [7, 4], [4, 1], [1, 0],
                    [1, 20], [20, 22], [22, 24], [21, 6],
                    [20, 6], [6, 15], [15, 17], [17, 19],
                    [15, 16], [16, 18], [8, 14], [8, 10],
                    [10, 12], [7, 13], [7, 9], [9, 11]
                ]
            },
            "body30": {
                'keypoints': {
                    0: 'pelvis',
                    1: 'left_hip',
                    2: 'right_hip',
                    3: 'torso',
                    4: 'left_knee',
                    5: 'right_knee',
                    6: 'neck',
                    7: 'left_ankle',
                    8: 'right_ankle',
                    9: 'left_big_toe',
                    10: 'right_big_toe',
                    11: 'left_small_toe',
                    12: 'right_small_toe',
                    13: 'left_heel',
                    14: 'right_heel',
                    15: 'nose',
                    16: 'left_eye',
                    17: 'right_eye',
                    18: 'left_ear',
                    19: 'right_ear',
                    20: 'left_shoulder',
                    21: 'right_shoulder',
                    22: 'left_elbow',
                    23: 'right_elbow',
                    24: 'left_wrist',
                    25: 'right_wrist',
                    26: 'left_pinky_knuckle',
                    27: 'right_pinky_knuckle',
                    28: 'left_index_knuckle',
                    29: 'right_index_knuckle'
                },
                "skeleton": [
                    [0, 3], [3, 6], [6, 0], [8, 5],
                    [5, 2], [2, 0], [2, 21], [21, 23],
                    [23, 25], [25, 27], [25, 29], [27, 29],
                    [8, 14], [8, 10], [10, 12], [21, 6],
                    [7, 4], [4, 1], [1, 0], [1, 20],
                    [20, 22], [22, 24], [24, 26], [24, 28],
                    [26, 28], [7, 13], [7, 9], [9, 11],
                    [20, 6], [6, 15], [15, 17], [17, 19],
                    [15, 16], [16, 18]
                ],
            },
        }
        Pose2dInfoDict['coco'] = copy.deepcopy(Pose2dInfoDict.get('coco18'))
        Pose2dInfoDict['coco25'] = copy.deepcopy(Pose2dInfoDict.get('openpose'))
        Pose2dInfoDict['vitpose'] = copy.deepcopy(Pose2dInfoDict.get('coco17'))
    return Pose2dInfoDict


class Pose2dFormat(enum.Enum):
    COCO17 = 'coco17'
    COCO18 = 'coco18'
    COCO19 = 'coco19'
    COCO25 = 'coco25'
    OPENPOSE = 'openpose'
    MPII = 'mpii'
    MPII14 = 'mpii14'
    H36M = 'h36m'
    LSP = 'lsp'

    @property
    def hip_indices(self) -> Tuple[int, int]:
        names = self.names
        return names.index('right_hip'), names.index('left_hip')

    @property
    def kinematic_tree(self) -> List[int]:
        bones = self.skeleton
        parents = [-1] * len(bones)
        for child, parent in bones:
            parents[child] = parent
        return parents

    @property
    def names(self) -> List[str]:
        return list(pose2d_info_dict()[self.value]['keypoints'].values())

    @property
    def skeleton(self) -> List[Tuple[int, int]]:
        return pose2d_info_dict()[self.value]['skeleton']

    @property
    def symmetric_indices(self) -> List[int]:
        names = self.names
        symmetric_names = self.symmetric_names
        return [(names.index(sn) if sn is not None else -1) for sn in symmetric_names]

    @property
    def symmetric_names(self) -> List[str]:
        names = self.names
        symmetric = []
        for name in names:
            if name.startswith('left_') and name.replace('left_', 'right_') in names:
                symmetric.append(name.replace('left_', 'right_'))
            elif name.startswith('right_') and name.replace('right_', 'left_') in names:
                symmetric.append(name.replace('left_', 'left_'))
            else:
                symmetric.append(None)
        return symmetric

    def to(self, new_fmt: Union[Pose2dFormat, str]) -> DictT[int, int]:
        return Nutrients.match_strings(self.names, Pose2dFormat(new_fmt).names, sort=True)  # <cur index>: <new index>

    def to_smpl(self, smpl_fmt: Union[Pose3dFormat, str] = 'smpl') -> DictT[int, int]:
        """
        Get the mapping from the current 2d pose format to the SMPL joint format.
        {
            <2d pose joint index>: <smpl joint index>
        }
        """
        # [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
        smpl_names = Pose3dFormat(smpl_fmt).names.values()
        return Nutrients.match_strings(self.names, list(smpl_names), sort=True)


class Pose3dFormat(enum.Enum):
    SMPL = 'smpl'
    SMPLH = 'smplh'
    SMPLX = 'smplx'

    @property
    def names(self) -> DictT[int, str]:
        global SmplJointNames
        if SmplJointNames[self.value] is None:
            if self.value == 'smpl':
                mapping = [
                              "pelvis",
                              "left_hip",
                              "right_hip",
                              "spine1",
                              "left_knee",
                              "right_knee",
                              "spine2",
                              "left_ankle",
                              "right_ankle",
                              "spine3",
                              "left_foot",
                              "right_foot",
                              "neck",
                              "left_collar",
                              "right_collar",
                              "head",
                              "left_shoulder",
                              "right_shoulder",
                              "left_elbow",
                              "right_elbow",
                              "left_wrist",
                              "right_wrist",
                              "left_hand",
                              "right_hand"
                          ] + \
                          [
                              # Extra joints selected from SMPL vertices
                              # face
                              'nose',
                              'right_eye',
                              'left_eye',
                              'right_ear',
                              'left_ear',
                              # feet
                              'left_big_toe',
                              'left_small_toe',
                              'left_heel',
                              'right_big_toe',
                              'right_small_toe',
                              'right_heel',
                              # hands
                              'left_thumb',
                              'left_index',
                              'left_middle',
                              'left_ring',
                              'left_pinky',
                              'right_thumb',
                              'right_index',
                              'right_middle',
                              'right_ring',
                              'right_pinky',
                          ]
            elif self.value == 'smplh':
                from smplx.joint_names import SMPLH_JOINT_NAMES
                mapping = SMPLH_JOINT_NAMES
            elif self.value == 'smplx':
                from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES
                mapping = SMPLX_JOINT_NAMES
            else:
                raise ValueError(f'Unrecognized mapping: {self.value}')
            SmplJointNames[self.value] = {i: v for i, v in enumerate(mapping)}
        return SmplJointNames[self.value]

    @property
    def skeleton(self) -> List[Tuple[int, int]]:
        global SmplSkeleton
        if SmplSkeleton[self.value] is None:
            from reconstruction.data.synthetic.smpl import Smpl
            kinematic_tree = Smpl(model_path=self.value).parents
            if isinstance(kinematic_tree, Tensor):
                kinematic_tree = kinematic_tree.detach().cpu().flatten().numpy().tolist()
            skeleton = []
            for i, p in enumerate(kinematic_tree):
                if p >= 0 and (p, i) not in skeleton:
                    skeleton.append((p, i))
            SmplSkeleton[self.value] = skeleton
        return SmplSkeleton[self.value]


if __name__ == '__main__':
    pose2d_fmt_ = Pose2dFormat.COCO17
    pose3d_fmt_ = Pose3dFormat.SMPL
    map_to_smpl_ = pose2d_fmt_.to_smpl('smpl')
    print(len(list(map_to_smpl_.keys())))
    print(map_to_smpl_.keys())
    print(map_to_smpl_.values())
    print([n for i, n in enumerate(list(pose2d_fmt_.names)) if i not in list(map_to_smpl_.keys())])
    print([n for i, n in enumerate(list(pose3d_fmt_.names.values())) if i not in list(map_to_smpl_.values())])
    exit(0)


class Pose2d(object):
    # noinspection PyUnusedLocal
    def __init__(self,
                 keypoints: Tensor,
                 scores: Tensor,
                 fmt: Union[Pose2dFormat, str],
                 frame_wh: Tuple[int, int],
                 bbox_xyxy: Optional[Tensor] = None,
                 **kwargs):
        """
        KeyPoints class constructor.

        Parameters
        ----------
        keypoints: torch.Tensor
            The keypoints tensor, with shape [batch_size, num_keypoints, 2].
        scores: torch.Tensor
            The scores tensor, with shape [batch_size, num_keypoints, 1].
        fmt: Union[Pose2dFormat, str]
            The keypoint format. One of "coco17", "coco25", "mpii", "h36m", "lsp". See `Pose2dFormat` for all options.
        frame_wh: Tuple[int, int]
            The original image (NOT PATCH) width and height.
        bbox_xyxy: Tensor, optional
            The bounding box in top-left, bottom-right coordinates format, shape (batch_size, 4)
        """
        self.keypoints = keypoints if keypoints.ndim == 3 else keypoints.unsqueeze(0)
        self.scores = scores if scores.ndim == 3 else scores.unsqueeze(0)
        self.fmt = Pose2dFormat(fmt)
        self.frame_wh = frame_wh
        self.bbox = Bbox(bbox_xyxy, frame_wh=frame_wh) if bbox_xyxy is not None else \
            Bbox.from_pose2d(self.keypoints, self.scores, frame_wh=frame_wh)

    @property
    def cliff_keypoints(self) -> Tensor:
        # Normalize keypoints INSIDE THE BBOX (patch-based normalization)
        bbox_size_wh = torch.stack([self.bbox.widths, self.bbox.heights], dim=-1)
        if bbox_size_wh.ndim == 2 and self.keypoints.ndim == 3:
            bbox_size_wh = bbox_size_wh.unsqueeze(1)
        return 2.0 * (self.keypoints - self.bbox.xyxy[..., :2].expand_as(bbox_size_wh)) / bbox_size_wh - 1.0

    @property
    def cliff(self) -> Tensor:
        # Normalize keypoints INSIDE THE BBOX (patch-based normalization)
        cliff_kp2d = self.cliff_keypoints
        # Normalize bbox INSIDE THE FRAME (frame-based normalization)
        cxywhn_m1p1 = self.bbox.cxywhn_m1p1
        # Concatenate keypoints with normalized bbox center and normalized scale
        return torch.cat([
            cliff_kp2d.flatten(-2),
            cxywhn_m1p1[..., :2],
            torch.maximum(cxywhn_m1p1[..., [-1]], cxywhn_m1p1[..., [-2]])
        ], dim=-1)

    @property
    def skeleton(self) -> List[Tuple[int, int]]:
        return self.fmt.skeleton

    def __iter__(self) -> Iterator[Pose2d]:
        for i in range(len(self.keypoints)):
            yield self.__getitem__(i)

    def __getitem__(self, index: int) -> Pose2d:
        return Pose2d(
            keypoints=self.keypoints[index],
            scores=self.scores[index],
            fmt=self.fmt,
            frame_wh=self.frame_wh[index] if isinstance(self.frame_wh, Tensor) and self.frame_wh.ndim == 2 else self.frame_wh,
            bbox_xyxy=self.bbox.xyxy[index].unsqueeze(0)
        )

    def as_dict(self) -> DictT[str, Tensor]:
        return dict(
            keypoints=self.keypoints,
            fmt=self.fmt.value,
            scores=self.scores,
            img_wh=self.frame_wh,
            frame_wh=self.frame_wh,
            bbox_xyxy=self.bbox.xyxy
        )

    def asdict(self) -> DictT[str, Tensor]:
        return self.as_dict()

    def as_fmt(self, new_fmt: Union[Pose2dFormat, str]) -> Pose2d:
        mapping = Pose2dFormat(new_fmt).to(self.fmt)
        new_names = Pose2dFormat(new_fmt).names
        new_keypoints = torch.zeros(self.keypoints.shape[0], len(new_names), 2, dtype=self.keypoints.dtype,
                                    device=self.keypoints.device)
        new_scores = torch.zeros(self.scores.shape[0], len(new_names), 1, dtype=self.scores.dtype,
                                 device=self.scores.device)
        new_keypoints[..., list(mapping.keys()), :] = self.keypoints[..., list(mapping.values()), :]
        new_scores[..., list(mapping.keys()), :] = self.scores[..., list(mapping.values()), :]
        new_frame_wh = self.frame_wh
        new_bbox_xyxy = self.bbox.xyxy.clone()
        return Pose2d(
            keypoints=new_keypoints,
            scores=new_scores,
            fmt=new_fmt,
            frame_wh=new_frame_wh,
            bbox_xyxy=new_bbox_xyxy,
        )

    def cpu(self) -> Pose2d:
        return self.to('cpu')

    def cuda(self) -> Pose2d:
        return self.to('cuda')

    def detach(self) -> Pose2d:
        self.keypoints = self.keypoints.detach()
        self.scores = self.scores.detach()
        return self

    def draw(self, patch: Tensor, **drawer_kwargs) -> Tensor:
        """Draw the keypoints on a patch.

        Parameters
        ----------
        patch: Tensor
            The patch tensor, with shape [3, height, width].
        drawer_kwargs: Any
            The arguments to pass to the drawer function.

        Returns
        -------
        torch.Tensor
            The patch with the keypoints drawn on it.
        """
        needs_norm = patch.is_floating_point() and patch.max() <= 1.0 and patch.min() >= 0
        if needs_norm:
            patch = patch.mul(255).add_(0.5).clamp_(0, 255).byte()
        tvv_major, tvv_minor = list(map(int, torchvision.__version__.split('.')[:2]))
        if tvv_major == 0 and tvv_minor <= 18:
            # no visibility option
            out = patch.clone()
            skeleton_t = torch.tensor(self.skeleton, device=patch.device, dtype=torch.long)
            for b in range(self.keypoints.shape[0]):
                kp_b = self.keypoints[b]
                invalid = (kp_b == 0).all(dim=-1)
                valid_idx = (~invalid).nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                idx_map = {old_idx.item(): i for i, old_idx in enumerate(valid_idx)}
                skel_filtered = []
                for i, j in skeleton_t:
                    if not invalid[i] and not invalid[j]:
                        skel_filtered.append([idx_map[i.item()], idx_map[j.item()]])
                img_black = torch.zeros_like(patch)
                drawn_kps = torchvision.utils.draw_keypoints(
                    image=img_black,
                    keypoints=kp_b[~invalid].unsqueeze(0),
                    connectivity=skel_filtered,
                    **drawer_kwargs
                )
                mask = (drawn_kps != 0).any(dim=0, keepdim=True)
                out[:, mask[0]] = drawn_kps[:, mask[0]]
        else:
            out = torchvision.utils.draw_keypoints(
                image=patch,
                keypoints=self.keypoints,
                connectivity=self.skeleton,
                visibility=(self.keypoints == 0).all(dim=-1).logical_or_(self.keypoints.isnan().logical_or_(self.keypoints.isinf()).any(dim=-1)).logical_not_(),
                **drawer_kwargs
            )
        if needs_norm:
            return out.float().div_(255.0)
        return out

    def on_image_squarified(self) -> Pose2d:
        assert self.frame_wh is not None
        kp_square = self.keypoints.clone()
        w, h = self.frame_wh
        # pad keypoints
        diff = abs(w - h)
        if h > w:
            kp_square[..., 0:4:2] += diff / 2
        else:
            kp_square[..., 1:4:2] += diff / 2
        # resize
        kp_square *= min(w, h) / max(w, h)
        return Pose2d(kp_square, self.scores.clone(), self.fmt, (min(w, h), min(w, h)))

    @classmethod
    def from_dict(cls, d: dict, fmt: Union[Pose2dFormat, str] = 'coco17', filter_dict: bool = False,
                  filter_prefix: str = 'pose2d', **unflatten_kwargs) -> Pose2d:
        if filter_dict:
            d = Dict(d).unflatten(**unflatten_kwargs).only(filter_prefix)
        return cls(
            keypoints=d['keypoints'],
            scores=d['scores'],
            fmt=Pose2dFormat(fmt),
            frame_wh=d['frame_wh'] if 'frame_wh' in d.keys() else d.get('img_wh', None),
            bbox_xyxy=d.get('bbox_xyxy', None),
        )

    @classmethod
    def from_heatmaps(cls,
                      heatmaps: torch.Tensor,
                      fmt: Union[Pose2dFormat, str],
                      patch_wh: Union[Tuple[int, int], Tensor],
                      frame_wh: Union[Tuple[int, int], Tensor],
                      bbox_xyxy: Optional[Tensor] = None,
                      convert_to_frame: bool = False,
                      **kwargs) -> Pose2d:
        if isinstance(patch_wh, tuple) and isinstance(patch_wh[0], int):
            patch_wh = np.array([[patch_wh[0], patch_wh[1]]], dtype=int)
        elif isinstance(patch_wh, Tensor):
            patch_wh = patch_wh.int().numpy()
        keypoints_np, scores_np = keypoints_from_heatmaps(heatmaps=heatmaps.detach().cpu().numpy(),
                                                          center=patch_wh // 2,
                                                          scale=patch_wh,
                                                          unbiased=True,
                                                          use_udp=True)
        keypoints = torch.from_numpy(keypoints_np)
        if convert_to_frame and bbox_xyxy is not None:
            keypoints += bbox_xyxy[..., :2].view_as(keypoints[..., [0], :])
        return cls(
            keypoints=keypoints,
            scores=torch.from_numpy(scores_np),
            fmt=fmt,
            frame_wh=frame_wh,
            bbox_xyxy=bbox_xyxy,
            **kwargs
        )

    def to(self, *args, **kwargs) -> Pose2d:
        self.keypoints = self.keypoints.to(*args, **kwargs)
        self.scores = self.scores.to(*args, **kwargs)
        return self


class Pose3d(object):
    def __init__(self,
                 joints: torch.Tensor,
                 fmt: Union[Pose3dFormat, str],
                 scores: Optional[torch.Tensor] = None,
                 extra_joint_names: Optional[List[str]] = None):
        """
        Pose3d class constructor.

        Parameters
        ----------
        joints: torch.Tensor
            The keypoints tensor, with shape [B, num_keypoints, 3].
        fmt: Union[Pose3dFormat, str]
            The joints format, one of "smpl", "smplh", "smplx". See Pose3dFormat for all options.
        scores: torch.Tensor, optional
            The scores tensor, with shape [B, num_keypoints].
        extra_joint_names: List[str], optional
            Names of the extra regressed joints (e.g. when extra joints regressor is used in SMPL).
        """
        self.joints = joints if joints.ndim == 3 else joints.unsqueeze(0)
        self.fmt = Pose3dFormat(fmt)
        if scores is None:
            scores = torch.ones(*self.joints.shape[:2], dtype=joints.dtype, device=joints.device)
        self.scores = scores if scores.ndim == 3 else scores.unsqueeze(0)
        self.extra_joint_names = extra_joint_names or []
        assert self.joints.shape[-2] == fmt.names

    @property
    def skeleton(self) -> List[Tuple[int, int]]:
        return self.fmt.skeleton

    def __getitem__(self, index: int) -> Pose3d:
        return Pose3d(
            joints=self.joints[index],
            scores=self.scores[index],
            fmt=self.fmt,
        )

    def __iter__(self) -> Iterator[Pose3d]:
        for joint, score in zip(self.joints, self.scores):
            yield Pose3d(
                joints=joint,
                scores=score,
                fmt=self.fmt,
            )

    def as_dict(self) -> DictT[str, Tensor]:
        return dict(
            joints=self.joints,
            fmt=self.fmt.value,
            scores=self.scores,
            extra_joint_names=self.extra_joint_names
        )

    def as_fmt(self, new_fmt: Union[Pose3dFormat, str]):
        """
        Return a Pose3d object with the specified joint format.
        """
        new_fmt = Pose3dFormat(new_fmt)
        if new_fmt == self.fmt:
            return self

        new_joints = []
        new_scores = []
        self_joint_indices = list(self.fmt.names.keys())
        self_joint_names = list(self.fmt.names.values())
        for new_joint_name in new_fmt.names.values():
            if new_joint_name not in self_joint_names:
                new_joints.append(
                    torch.zeros_like(self.joints[..., [self_joint_indices[self_joint_names.index('spine1')]], :])
                )
                new_scores.append(
                    torch.zeros_like(self.scores[..., [self_joint_indices[self_joint_names.index('spine1')]]])
                )
            else:
                new_joints.append(self.joints[..., [self_joint_indices[self_joint_names.index(new_joint_name)]], :])
                new_scores.append(self.scores[..., [self_joint_indices[self_joint_names.index(new_joint_name)]]])
        return Pose3d(
            joints=torch.stack(new_joints, dim=-2),
            scores=torch.stack(new_scores, dim=-1),
            fmt=new_fmt,
        )

    def as_smpl(self):
        """
        Convert the body to SMPL joints.
        """
        return self.as_fmt(Pose3dFormat.SMPL)

    def as_smplh(self):
        """
        Convert the body to SMPL-H joints.
        """
        return self.as_fmt(Pose3dFormat.SMPLH)

    def as_smplx(self):
        """
        Convert the body to SMPL-X joints.
        """
        return self.as_fmt(Pose3dFormat.SMPLX)

    @classmethod
    def from_smpl(cls, joints: torch.Tensor, scores: Optional[torch.Tensor] = None):
        """
        Create a Body object from SMPL joints.
        """
        return cls(joints, fmt=Pose3dFormat.SMPL, scores=scores)

    @classmethod
    def from_smplh(cls, joints: torch.Tensor, scores: Optional[torch.Tensor] = None):
        """
        Create a Body object from SMPL-H joints.
        """
        return cls(joints, fmt=Pose3dFormat.SMPLH, scores=scores)

    @classmethod
    def from_smplx(cls, joints: torch.Tensor, scores: Optional[torch.Tensor] = None):
        """
        Create a Body object from SMPL-X joints.
        """
        return cls(joints, fmt=Pose3dFormat.SMPLX, scores=scores)


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length * img_size.max() / (256 * bs)
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def convert_yup(xyz):
    """
    converts points in x right y down z forward to x right y up z back
    :param xyz (*, 3)
    """
    x, y, z = torch.split(xyz[..., :3], 1, dim=-1)
    return torch.cat([x, -y, -z], dim=-1)


def full_perspective_projection(
        points,
        cam_intrinsics,
        rotation=None,
        translation=None,
):
    K = cam_intrinsics

    if rotation is not None:
        points = (rotation @ points.transpose(-1, -2)).transpose(-1, -2)
    if translation is not None:
        points = points + translation.unsqueeze(-2)
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = (K @ projected_points.transpose(-1, -2)).transpose(-1, -2)
    return projected_points[..., :-1]


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.

    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Ground-truth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when normalize==0
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _get_max_preds_3d(heatmaps):
    """Get keypoint predictions from 3D score maps.

    Note:
        batch size: N
        num keypoints: K
        heatmap depth size: D
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, D, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 3]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 5, 'heatmaps should be 5-ndim'

    N, K, D, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.zeros((N, K, 3), dtype=np.float32)
    _idx = idx[..., 0]
    preds[..., 2] = _idx // (H * W)
    preds[..., 1] = (_idx // W) % H
    preds[..., 0] = _idx % W

    preds = np.where(maxvals > 0.0, preds, -1)
    return preds, maxvals


def pose_pck_accuracy(output, target, mask, thr=0.05, normalize=None):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Ground-truth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)


def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Ground-truth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)

    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt


def keypoint_auc(pred, gt, mask, normalize, num_step=20):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Ground-truth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (float): Normalization factor.
        num_step (int): 20

    Returns:
        float: Area under curve.
    """
    nor = np.tile(np.array([[normalize, normalize]]), (pred.shape[0], 1))
    x = [1.0 * i / num_step for i in range(num_step)]
    y = []
    for thr in x:
        _, avg_acc, _ = keypoint_pck_accuracy(pred, gt, mask, thr, nor)
        y.append(avg_acc)

    auc = 0
    for i in range(num_step):
        auc += 1.0 / num_step * y[i]
    return auc


def keypoint_nme(pred, gt, mask, normalize_factor):
    """Calculate the normalized mean error (NME).

    Note:
        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Ground-truth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize_factor (np.ndarray[N, 2]): Normalization factor.

    Returns:
        float: normalized mean error
    """
    distances = _calc_distances(pred, gt, mask, normalize_factor)
    distance_valid = distances[distances != -1]
    return distance_valid.sum() / max(1, len(distance_valid))


def keypoint_epe(pred, gt, mask):
    """Calculate the end-point error.

    Note:
        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Ground-truth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        float: Average end-point error.
    """

    distances = _calc_distances(
        pred, gt, mask,
        np.ones((pred.shape[0], pred.shape[2]), dtype=np.float32))
    distance_valid = distances[distances != -1]
    return distance_valid.sum() / max(1, len(distance_valid))


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
                heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
                heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
                heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
                heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
                heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


# noinspection PyTypeChecker
def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-processing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray) and isinstance(batch_heatmaps, torch.Tensor):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    # scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords
