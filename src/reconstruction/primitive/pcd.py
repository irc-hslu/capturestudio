import copy
import functools
import os
from pathlib import Path
from typing import Optional, Tuple, Literal, Union, Dict, List, Any
from warnings import deprecated

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

from utils.calib import CalibrationData
from utils.misc import log, env_get, PathUtils, Str
from utils.vis import VisUtils


# noinspection PyArgumentList,PyTypeHints
class RGBDImage:
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, depth: Union[np.ndarray, None], intrinsic: np.ndarray, extrinsic_w2c: np.ndarray, features: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize an RGBD image.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image of shape (H, W, C).
        mask : np.ndarray
            Valid pixels mask of shape (H, W), where >0 indicates valid pixels.
        depth : Union[np.ndarray, None]
            Depth map of shape (H, W). Depth values should be in meters. Value can be None during initialization, to allow for stereo estimation.
        intrinsic : np.ndarray
            Camera intrinsic matrix of shape (3, 3).
        extrinsic_w2c : np.ndarray
            Camera extrinsic matrix of shape (4, 4).
        features : Optional[Dict[str, np.ndarray]]
            Optional dictionary of additional features, e.g., normals or semantic segmentation masks.
            Each feature should be a numpy array of shape (H, W) or (H, W, C).
            When using pixel-wise splats, those should hold the scale, rotation, and opacity maps.
            Default is None.
        """
        self.rgb = rgb
        self.depth = depth
        self.mask = mask if mask.dtype == bool else (mask > 0)
        self.intrinsic = intrinsic
        self.extrinsic_w2c = extrinsic_w2c
        self.image_size_hw = (rgb.shape[0], rgb.shape[1])
        self.features = features if features is not None else {}

    @functools.cached_property
    def normals(self) -> np.ndarray:
        """
        Get the normals of the RGBD image. If not available, the normals are estimated using KDTree-based nearest neighbors.

        Returns
        -------
        np.ndarray
            The normals of the RGBD image of shape (H, W, 3).
        """
        if 'normals' in self.features:
            normal_full = self.features['normals']
            normal_full[(normal_full == 0).all(-1)] = -1.0
            return normal_full
        return self.normals_from_depth

    @functools.cached_property
    def normals_from_depth(self) -> np.ndarray:
        """
        Get the normals of the RGBD image. If not available, the normals are estimated using KDTree-based nearest neighbors.

        Returns
        -------
        np.ndarray
            The normals of the RGBD image of shape (H, W, 3).
        """
        return self.unproject().normals

    @functools.cached_property
    def open3d(self) -> o3d.geometry.RGBDImage:
        """
        Convert the RGBD image to an Open3D RGBDImage.

        Returns
        -------
        o3d.geometry.RGBDImage
            The Open3D RGBDImage representation of the RGBD image.
        """
        assert self.depth is not None, "Depth map is required for PCD operations."
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(self.rgb))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(self.depth))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,  # 1 unit == 1 m (already metres)
            depth_trunc=self.depth.max(),  # no truncation
            convert_rgb_to_intensity=False  # keep the 3‑channel colour
        )

    @functools.cached_property
    def intrinsic_o3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Convert the camera intrinsic matrix to an Open3D PinholeCameraIntrinsic.

        Returns
        -------
        o3d.camera.PinholeCameraIntrinsic
            The Open3D PinholeCameraIntrinsic representation of the camera intrinsic.
        """
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.width = self.rgb.shape[1]
        intrinsic_o3d.height = self.rgb.shape[0]
        intrinsic_o3d.intrinsic_matrix = self.intrinsic.astype(np.float64)
        return intrinsic_o3d

    @functools.cached_property
    def points_world(self) -> np.ndarray:
        """
        Get the 3D points in world coordinates.

        Returns
        -------
        np.ndarray
            The 3D points of shape (H, W, 3).
        """
        points = self.unproject().points
        return points

    @functools.cached_property
    def points_camera(self) -> np.ndarray:
        """
        Get the 3D points in camera coordinates.

        Returns
        -------
        np.ndarray
            The 3D points of shape (H, W, 3).
        """
        assert self.depth is not None, "Depth map is required for PCD operations."
        H, W = self.image_size_hw
        fx, fy, cx, cy = self.intrinsic[0, 0], self.intrinsic[1, 1], self.intrinsic[0, 2], self.intrinsic[1, 2]
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)
        d = self.depth.reshape(-1)
        return np.stack([(uu - cx) * d / fx, (vv - cy) * d / fy, d], axis=1).reshape([H, W, 3])

    def __repr__(self) -> str:
        """
        String representation of the RGBD image.

        Returns
        -------
        str
            A string representation of the RGBD image.
        """
        return (f"{self.__class__.__name__}(rgb_shape={self.rgb.shape}, "
                f"depth_shape={self.depth.shape if self.depth is not None else None}, "
                f"intrinsic_shape={self.intrinsic.shape}, "
                f"extrinsic_shape={self.extrinsic_w2c.shape}, "
                f"valid_pixels_shape={self.mask.shape})")

    def unproject(self) -> 'PixelPoints':
        """
        Unproject the RGBD image into a 3D point cloud.

        Returns
        -------
        PointCloud
            A PointCloud instance containing the 3D points and their corresponding colors.
        """
        pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(
            self.open3d,
            self.intrinsic_o3d,
            self.extrinsic_w2c.astype(np.float64),
            project_valid_depth_only=False
        )
        pixel_points = np.asarray(pcd_o3d.points).reshape([*self.depth.shape, 3])
        pixel_colors = np.asarray(pcd_o3d.colors).reshape([*self.depth.shape, 3])
        return PixelPoints(
            pixel_points=pixel_points,
            pixel_colors=pixel_colors,
            pixel_valid=(self.depth > 0.0) & np.isfinite(self.depth) & self.mask & ~np.isnan(np.asarray(pcd_o3d.points).reshape([*self.depth.shape, 3])).any(-1),
            pixel_features=self.features,
            extrinsics_c2w=np.linalg.inv(self.extrinsic_w2c)
        )

    def reproject(self, target_intrinsic: np.ndarray, target_extrinsic: np.ndarray, target_image_size_hw: Tuple[int, int], is_c2w: bool = True, use_cache: bool = False, **render_kwargs) -> 'RGBDImage':
        """
        Reproject the RGBD image into a new camera frame using the target intrinsic and extrinsic matrices.

        Parameters
        ----------
        target_intrinsic : np.ndarray
            Target camera intrinsic matrix of shape (3, 3).
        target_extrinsic : np.ndarray
            Target camera extrinsic matrix of shape (4, 4).
        target_image_size_hw : Tuple[int, int]
            The width and height of the target image.
        is_c2w : bool, optional
            If True, the target extrinsic is assumed to be camera-to-world (R,T). Default is True.
        use_cache : bool, optional
            If True, use cached renderer for the target image size. Default is True.

        Returns
        -------
        RGBDImage
            A new RGBDImage instance containing the reprojected RGB and depth images.
        """
        pcd = self.unproject()
        return pcd.project(target_intrinsic, target_extrinsic, target_image_size_hw, is_c2w=is_c2w, use_cache=use_cache, **render_kwargs)

    def reproject_to(self, other: 'RGBDImage', align: bool = False, use_cache: bool = False, **render_kwargs) -> 'RGBDImage':
        """
        Reproject the RGBD image to another RGBD image's camera frame.

        Parameters
        ----------
        other : RGBDImage
            The target RGBD image to reproject to.
        align : bool, optional
            If True, the projected images are aligned to the original by estimating the homography between them. Default is False.
        use_cache : bool, optional
            If True, use cached renderer for the target image size. Default is False.

        Returns
        -------
        RGBDImage
            A new RGBDImage instance containing the reprojected RGB and depth images.
        """
        reprojected = self.reproject(
            target_intrinsic=other.intrinsic,
            target_extrinsic=other.extrinsic_w2c,
            target_image_size_hw=(other.rgb.shape[0], other.rgb.shape[1]),
            is_c2w=False,
            use_cache=use_cache,
            **render_kwargs
        )
        # # align colors
        # ccm = solve_color_correction_matrix(reprojected.rgb, self.rgb, self.mask)
        # reprojected.rgb = color_transfer_reinhard(
        #     apply_color_correction_matrix(reprojected.rgb, ccm),
        #     self.rgb,
        # )
        if align:
            aligned_images, aligned_depths, aligned_valid = PCDUtils.align_projected_to_original(
                projected_images=reprojected.rgb,
                projected_depths=reprojected.depth,
                valid_pixels=reprojected.mask,
                original_images=other.rgb,
            )
            aligned_images = aligned_images.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            aligned_depths = aligned_depths.detach().cpu().numpy().squeeze()
            aligned_valid = aligned_valid.detach().cpu().numpy().squeeze()
            if reprojected.rgb.dtype == np.uint8:
                aligned_images = (aligned_images * 255.0 + 0.5).astype(np.uint8)
            if reprojected.depth.max() > 1000.0:
                aligned_depths = aligned_depths * 1000.0
            reprojected.rgb = aligned_images
            reprojected.depth = aligned_depths
            reprojected.mask = aligned_valid
        return reprojected

    def color_match_to(self, other: 'RGBDImage') -> 'RGBDImage':
        if self.image_size_hw[0] == 2 * other.image_size_hw[0]:
            rgb_l, rgb_r = self.rgb[:self.image_size_hw[0] // 2], self.rgb[self.image_size_hw[0] // 2:]
            ccm_l, ccm_r = self.solve_color_correction_matrix(rgb_l, other.rgb, mask=other.mask), self.solve_color_correction_matrix(rgb_r, other.rgb, mask=other.mask)
            rgb_l_corrected, rgb_r_corrected = self.apply_color_correction_matrix(rgb_l, ccm_l), self.apply_color_correction_matrix(rgb_r, ccm_r)
            rgb_corrected = np.concatenate((rgb_l_corrected, rgb_r_corrected), axis=0)
        else:
            other_rgb = other.rgb[:self.image_size_hw[0]]
            other_mask = other.mask[:self.image_size_hw[0]]
            ccm = self.solve_color_correction_matrix(self.rgb, other_rgb, mask=other_mask)
            rgb_corrected = self.apply_color_correction_matrix(self.rgb, ccm)
        return RGBDImage(
            rgb=rgb_corrected,
            mask=self.mask.copy(),
            depth=self.depth.copy() if self.depth is not None else None,
            intrinsic=self.intrinsic.copy(),
            extrinsic_w2c=self.extrinsic_w2c.copy(),
            features=self.features.copy() if self.features is not None else None,
        )

    @staticmethod
    def solve_color_correction_matrix(source, reference, mask=None):
        """
        Finds a 3x3 color correction matrix (CCM) that maps source colors to reference colors.

        Args:
            source (np.ndarray): The image to be corrected (H, W, 3).
            reference (np.ndarray): The reference image (H, W, 3).
            mask (np.ndarray, optional): A boolean mask (H, W) where True indicates
                                         pixels to use for the calculation.

        Returns:
            np.ndarray: The 3x3 color correction matrix.
        """
        if mask is None:
            # If no mask, use all pixels
            source_pixels = source.reshape(-1, 3)
            reference_pixels = reference.reshape(-1, 3)
        else:
            # Use only the masked pixels
            source_pixels = source[mask]
            reference_pixels = reference[mask]

        # Solve for the matrix M such that: source_pixels @ M = reference_pixels
        # This is a linear least-squares problem.
        ccm, _, _, _ = np.linalg.lstsq(source_pixels, reference_pixels, rcond=None)

        return ccm

    @staticmethod
    def apply_color_correction_matrix(image, ccm):
        """Applies a 3x3 CCM to an image."""
        # Reshape image to be a list of pixels, apply matrix, and reshape back
        pixels = image.reshape(-1, 3)
        corrected_pixels = pixels @ ccm

        # Clip to valid range and convert back to uint8
        corrected_image = corrected_pixels.reshape(image.shape)
        corrected_image = np.clip(corrected_image, 0, 255)

        return corrected_image.astype(np.uint8)

    @staticmethod
    def _resize(x: np.ndarray, new_h: int, new_w: int, mode: Literal['bilinear', 'nearest']) -> np.ndarray:
        return cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_LINEAR if mode == 'bilinear' else cv2.INTER_NEAREST)

    def resize(self, *new_size_hw: int) -> 'RGBDImage':
        """
        Resize the RGBD image to a new size.

        Parameters
        ----------
        new_size_hw : int or Tuple[int, int]
            The new height and width of the image. If a single integer is provided, it will be used for both dimensions.

        Returns
        -------
        RGBDImage
            A new RGBDImage instance with the resized RGB and depth images.
        """
        if len(new_size_hw) == 1:
            new_size_hw = new_size_hw[0], new_size_hw[0]
        elif len(new_size_hw) != 2:
            raise ValueError(f"Invalid new size: {new_size_hw}. Expected a single integer or a tuple of two integers (height, width).")
        if new_size_hw == self.image_size_hw:
            return self

        # resize
        src_h, src_w = self.image_size_hw
        tgt_h, tgt_w = new_size_hw
        scale = max(tgt_h / src_h, tgt_w / src_w)
        new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))

        img_rs = cv2.resize(self.rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize((self.mask.astype(int) * 255 + 0.5).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 128
        if self.depth is not None:
            depth_rs = cv2.resize(self.depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            depth_rs = None

        # center crop
        top = (new_h - tgt_h) // 2
        left = (new_w - tgt_w) // 2
        bot = top + tgt_h
        right = left + tgt_w
        img_rs_cropped = img_rs[top:bot, left:right, :]
        mask_rs_cropped = mask_rs[top:bot, left:right]
        depth_rs_cropped = None if depth_rs is None else depth_rs[top:bot, left:right]

        # adjust intrinsics
        intrinsic_new = self.intrinsic.copy()
        intrinsic_new[0, 0] *= scale
        intrinsic_new[1, 1] *= scale
        intrinsic_new[0, 2] = intrinsic_new[0, 2] * scale - left
        intrinsic_new[1, 2] = intrinsic_new[1, 2] * scale - top

        return RGBDImage(
            rgb=np.ascontiguousarray(img_rs_cropped),
            mask=np.ascontiguousarray(mask_rs_cropped),
            depth=np.ascontiguousarray(depth_rs_cropped) if depth_rs_cropped is not None else None,
            intrinsic=np.ascontiguousarray(intrinsic_new),
            extrinsic_w2c=self.extrinsic_w2c.copy()
        )

    def save_png(self, out_path: Optional[Union[Path, str]] = None, striped: bool = False, white_bg: bool = False, bg_color: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Save the RGBD image as a PNG file.
        The file contains the RGB, mask, and depth images, concatenated along the width:
          ____________________
         | RGB | Mask | Depth |
          --------------------

        Parameters
        ----------
        out_path : Path, optional
            The path where to save the PNG file. If None, it will return the image as a numpy array.
        striped : bool, optional
            If True, the RGB/depth/features images will be concatenated in a striped manner.
        white_bg: bool, optional
            If True, the background of the image will be white. Default is False, which uses a black background.
        bg_color: str, optional
            If provided, the background of the image will have this color. Can provide css colors of hex strings.

        Returns
        -------
        Optional[np.ndarray]
            If out_path is None, returns the concatenated RGBD image as a numpy array.
            Otherwise, saves the image to the specified path and returns None.
        """
        is_stereo = hasattr(self, 'right_intrinsic') and getattr(self, 'right_intrinsic', None) is not None
        rgb_image_bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        mask_image_bgr = cv2.cvtColor((self.mask.astype(np.float32) * 255 + 0.5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out_images = [rgb_image_bgr, mask_image_bgr]
        if self.features is not None and 'disparity' in self.features:
            disparity_vis = self.features['disparity'].copy()
            half_height = disparity_vis.shape[0] // 2
            if is_stereo:
                disparity_images_bgr = []
                for i in range(2):
                    disparity_vis_half = disparity_vis[i * half_height:(i + 1) * half_height, :]
                    disparity_mask = ~np.isinf(disparity_vis_half) & ~np.isnan(disparity_vis_half)
                    disparity_min_val = disparity_vis_half[disparity_mask].min()
                    disparity_max_val = disparity_vis_half[disparity_mask].max()
                    disparity_vis_half = ((disparity_vis_half - disparity_min_val) / (disparity_max_val - disparity_min_val)).clip(0, 1) * 255
                    disparity_image_half_bgr = cv2.applyColorMap(disparity_vis_half.clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # BGR to RGB
                    disparity_image_half_bgr[~disparity_mask] = 0.0
                    disparity_images_bgr.append(disparity_image_half_bgr)
                disparity_image_bgr = np.concatenate(disparity_images_bgr, axis=0)
            else:
                disparity_mask = ~np.isinf(disparity_vis) & ~np.isnan(disparity_vis)
                disparity_min_val = disparity_vis[disparity_mask].min()
                disparity_max_val = disparity_vis[disparity_mask].max()
                disparity_vis = ((disparity_vis - disparity_min_val) / (disparity_max_val - disparity_min_val)).clip(0, 1) * 255
                disparity_image_bgr = cv2.applyColorMap(disparity_vis.clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # BGR to RGB
                disparity_image_bgr[~disparity_mask] = 0.0
            out_images.append(disparity_image_bgr)
        if self.depth is not None:
            depth_vis = ((self.depth.clip(1.0, 3.5) - 1.0) / 2.5 * 255).astype(np.uint8)
            valid_mask = (self.depth > 0) & np.isfinite(self.depth)
            depth_in_mask = depth_vis[valid_mask]
            depth_in_mask = 255 - depth_in_mask
            depth_vis[valid_mask] = depth_in_mask
            depth_image_bgr = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            normal_map_pcd = self.normals_from_depth
            normal_pcd_img = ((normal_map_pcd + 1.0) / 2.0 * 255 + 0.5).astype(np.uint8)
            normal_pcd_img_bgr = cv2.cvtColor(normal_pcd_img, cv2.COLOR_RGB2BGR)
            out_images.extend([depth_image_bgr, normal_pcd_img_bgr])

        if striped:
            png_image = VisUtils.striped_teaser_image(out_images[0] * self.mask[..., None], *out_images[-2:], spacing=0.0, widths=(0.075, 0.075), order=('depth', 'normals'))
            png_image[~self.mask & (png_image[..., 0] < 40)] = 255.0  # white background
        else:
            png_image = np.concatenate(out_images, axis=1)
            if white_bg or bg_color is not None:
                # repeat the mask N times, to fill the width
                if png_image.shape[1] > self.mask.shape[1]:
                    png_mask = np.tile(self.mask, (1, png_image.shape[1] // self.mask.shape[1]))
                else:
                    png_mask = self.mask
                png_image[~png_mask] = np.array(Str(bg_color if bg_color is not None else 'white').rgb(normalize=False), dtype=png_image.dtype)
        if out_path is None:
            return png_image
        cv2.imwrite(str(out_path), png_image)
        log(f'[{self.__class__.__name__}::save_png] RGBDImage saved to {out_path}', 'debug')
        return None

    def visualize_features(self, feat: Literal['normal'] = 'normals', white_bg=False, colormap=None, depth_range=None) -> np.ndarray:
        """
        Visualize the features of the RGBD image.

        Parameters
        ----------
        feat : Literal['normals'], optional
            The feature to visualize. Currently only 'normals' is supported.
        white_bg : bool, optional
            If True, the background of the image will be white. Default is False, which uses
            a black background.

        Returns
        -------
        np.ndarray
            The visualized feature as a numpy array.
        """
        if feat == 'depth':
            dmin, dmax = depth_range if depth_range is not None else (np.nanmin(self.depth[self.mask]), np.nanmax(self.depth[self.mask]))
            depth_vis = ((self.depth.clip(dmin, dmax) - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            valid_mask = (self.depth > 0) & np.isfinite(self.depth)
            depth_in_mask = depth_vis[valid_mask]
            depth_in_mask = 255 - depth_in_mask
            depth_vis[valid_mask] = depth_in_mask
            out_img = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO if colormap is None else colormap)
        elif feat == 'normals':
            normal_map = self.normals_from_depth
            normal_map_img = ((normal_map + 1.0) / 2.0 * 255 + 0.5).astype(np.uint8)
            out_img = cv2.cvtColor(normal_map_img, cv2.COLOR_RGB2BGR)
        elif feat == 'disparity':
            disparity_vis = self.features['disparity'].copy()
            half_height = disparity_vis.shape[0] // 2
            is_stereo = hasattr(self, 'right_intrinsic') and getattr(self, 'right_intrinsic', None) is not None
            if is_stereo:
                disparity_images_bgr = []
                for i in range(2):
                    disparity_vis_half = disparity_vis[i * half_height:(i + 1) * half_height, :]
                    disparity_mask = ~np.isinf(disparity_vis_half) & ~np.isnan(disparity_vis_half)
                    disparity_min_val = disparity_vis_half[disparity_mask].min()
                    disparity_max_val = disparity_vis_half[disparity_mask].max()
                    disparity_vis_half = ((disparity_vis_half - disparity_min_val) / (disparity_max_val - disparity_min_val)).clip(0, 1) * 255
                    disparity_image_half_bgr = cv2.applyColorMap(disparity_vis_half.clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # BGR to RGB
                    disparity_image_half_bgr[~disparity_mask] = 0.0
                    disparity_images_bgr.append(disparity_image_half_bgr)
                disparity_image_bgr = np.concatenate(disparity_images_bgr, axis=0)
            else:
                disparity_mask = ~np.isinf(disparity_vis) & ~np.isnan(disparity_vis)
                disparity_min_val = disparity_vis[disparity_mask].min()
                disparity_max_val = disparity_vis[disparity_mask].max()
                disparity_vis = ((disparity_vis - disparity_min_val) / (disparity_max_val - disparity_min_val)).clip(0, 1) * 255
                disparity_image_bgr = cv2.applyColorMap(disparity_vis.clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # BGR to RGB
                disparity_image_bgr[~disparity_mask] = 0.0
            out_img = disparity_image_bgr
        else:
            raise ValueError(f"Unsupported feature for visualization: {feat}. Supported features are: 'normals'.")
        if white_bg:
            out_img[~self.mask] = 255.0
        return out_img

    def rotate(self, rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None) -> 'RGBDImage':
        """
        Rotate the RGB/depth/mask by 90° steps and update the camera model so that
        unprojection/back-projection yields the *same* world-space point cloud.

        Parameters
        ----------
        rotate : {'90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180'}, optional
            Which way to rotate the *images*. If None, returns self.

        Notes
        -----
        * H_pix uses 0-based pixels with center at integer coords. If your code uses a 0.5-center
          convention, swap (W-1,H-1) with (W-0.5,H-0.5) consistently.
        * Depth is assumed pre-aligned to color. We rotate depth with the same rigid 90° op (no interpolation).
        """
        if rotate is None:
            return self

        assert rotate in ('90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180')

        rgb = self.rgb
        mask = self.mask
        depth = self.depth
        H0, W0 = rgb.shape[:2]

        # rotate images (OpenCV does a transpose+flip, no resampling)
        rot_flag = getattr(cv2, f'ROTATE_{rotate}')
        rgb_r = cv2.rotate(rgb, rot_flag)
        depth_r = None if depth is None else cv2.rotate(depth, rot_flag)
        if mask is not None:
            mask_r = cv2.rotate(mask.astype(np.float32), rot_flag) > 0.5
        else:
            mask_r = None

        def _rz_deg(angle_deg: float) -> np.ndarray:
            a = np.deg2rad(angle_deg)
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s, 0.],
                             [s, c, 0.],
                             [0., 0., 1.]], dtype=np.float64)

        def _Himg(W: int, H: int, rotate: Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180'],
                  half_pixel: bool = False) -> np.ndarray:
            """
            Pixel-space homography mapping original pixel coords p -> rotated coords p' = Himg p.
            Uses 0-based integer centers by default. If your math uses 0.5-centered pixels,
            set half_pixel=True (constants become W-0.5/H-0.5).
            """
            offW = (W - 0.5) if half_pixel else (W - 1.0)
            offH = (H - 0.5) if half_pixel else (H - 1.0)

            if rotate == '90_CLOCKWISE':  # u' = -v + offH v' = u
                return np.array([[0., -1., offH],
                                 [1., 0., 0.],
                                 [0., 0., 1.]], dtype=np.float64)
            elif rotate == '90_COUNTERCLOCKWISE':  # u' = v v' = -u + offW
                return np.array([[0., 1., 0.],
                                 [-1., 0., offW],
                                 [0., 0., 1.]], dtype=np.float64)
            else:  # '180' : u' = -u + offW  v' = -v + offH
                return np.array([[-1., 0., offW],
                                 [0., -1., offH],
                                 [0., 0., 1.]], dtype=np.float64)

        # --- copy camera params ---
        K = np.asarray(self.intrinsic, dtype=np.float64).copy()  # 3x3
        Tw2c = None if self.extrinsic_w2c is None else np.asarray(self.extrinsic_w2c, dtype=np.float64).copy()

        # --- pick camera-frame rotation α that keeps fx,fy positive after update ---
        # For positive focal lengths in K', use:
        #   α = +90° for '90_CLOCKWISE'
        #   α = -90° for '90_COUNTERCLOCKWISE'
        #   α =  180° for '180'
        angle = {'90_CLOCKWISE': +90.0, '90_COUNTERCLOCKWISE': -90.0, '180': 180.0}[rotate]
        Rz = _rz_deg(angle)

        # Pixel-space rotation (original -> rotated pixels)
        Himg = _Himg(W0, H0, rotate, half_pixel=True)

        # --- Core identities for invariance ---
        # We want for every world point X:
        #   Himg * K * (R X + t)  ≍  K' * (R' X + t')
        # with R' = Rz * R, t' = Rz * t  (camera axes rotated)
        # Solve K' from:  K' * Rz  =  Himg * K   =>   K' = Himg * K * Rz.T
        Kp = Himg @ K @ Rz.T

        # --- Update extrinsic_w2c as 3x4 or 4x4 ---
        if Tw2c is not None:
            if Tw2c.shape == (4, 4):
                R = Tw2c[:3, :3]
                t = Tw2c[:3, 3]
                Rn = Rz @ R
                tn = Rz @ t
                Tw2c_p = Tw2c.copy()
                Tw2c_p[:3, :3] = Rn
                Tw2c_p[:3, 3] = tn
            elif Tw2c.shape == (3, 4):
                R = Tw2c[:3, :3]
                t = Tw2c[:3, 3]
                Rn = Rz @ R
                tn = Rz @ t
                Tw2c_p = np.concatenate([Rn, tn[:, None]], axis=1)
            else:
                raise ValueError("extrinsic_w2c must be 3x4 or 4x4 if provided.")
        else:
            Tw2c_p = None

        return RGBDImage(rgb=rgb_r, mask=mask_r, depth=depth_r, intrinsic=Kp, extrinsic_w2c=Tw2c_p)

    @staticmethod
    @deprecated('RGBDImage::estimate_floor is now deprecated in favor of the new more complete and feature-rich class found in vis/teaser/base.py. Please use that instead.')
    def estimate_floor(
            *rgbd_images: 'RGBDImage',
            wall_overshoot_m: float = 2.0,
            wall_height_m: float = 3.0,
            wall_pad_width_m: float = 1.0,
            floor_depth_scale: float = 1.5,  # NEW: scale factor for floor depth (length)
            export: bool = False,
            export_path: Optional[Union[Path, str]] = None
    ) -> Dict[str, Optional[Union[np.ndarray, float, int]]]:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME, dtype=dtype).to(device).eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        floor_ids = [k for k, v in id2label.items()
                     if any(t in v.lower() for t in ("floor", "ground", "road", "sidewalk", "pavement"))] or [0]
        rng = np.random.default_rng(0)

        def seg_floor_prob(rgb: np.ndarray) -> np.ndarray:
            H, W = rgb.shape[:2]
            tile, overlap = 1280, 256
            if max(H, W) <= tile:
                inp = processor(images=rgb, return_tensors="pt")
                inp = {k: v.to(device, dtype=model.dtype) for k, v in inp.items()}
                with torch.inference_mode():
                    logits = model(**inp).logits
                    logits = torch.nn.functional.interpolate(
                        logits, size=(H, W), mode="bilinear", align_corners=False
                    )[0]
                    p = logits.softmax(dim=0)[floor_ids].sum(dim=0).float().cpu().numpy()
                return np.clip(p, 0, 1).astype(np.float32)

            step = max(1, tile - overlap)
            acc = np.zeros((H, W), dtype=np.float32)
            cnt = np.zeros((H, W), dtype=np.float32)
            for y in range(0, H, step):
                for x in range(0, W, step):
                    y1, x1 = min(H, y + tile), min(W, x + tile)
                    patch = rgb[y:y1, x:x1]
                    inp = processor(images=patch, return_tensors="pt")
                    inp = {k: v.to(device, dtype=model.dtype) for k, v in inp.items()}
                    with torch.inference_mode():
                        logits = model(**inp).logits
                        logits = torch.nn.functional.interpolate(
                            logits, size=patch.shape[:2], mode="bilinear", align_corners=False
                        )[0]
                        p = logits.softmax(dim=0)[floor_ids].sum(dim=0).float().cpu().numpy()
                    acc[y:y1, x:x1] += p
                    cnt[y:y1, x:x1] += 1.0
            return np.clip(acc / np.maximum(cnt, 1e-6), 0, 1).astype(np.float32)

        overlays: List[np.ndarray] = [] if export else None
        per_view_planes: List[Optional[Dict[str, Any]]] = []
        per_view_probs: List[np.ndarray] = []
        rgb_cache: List[Optional[np.ndarray]] = []
        K_cache: List[Optional[np.ndarray]] = []
        w2c_cache: List[Optional[np.ndarray]] = []

        all_pts, all_w = [], []

        for img in rgbd_images:
            if img is None or img.rgb is None or img.depth is None:
                if export: overlays.append(np.zeros((1, 1, 3), np.uint8))
                per_view_planes.append(None)
                per_view_probs.append(None)
                rgb_cache.append(None)
                K_cache.append(None)
                w2c_cache.append(None)
                continue

            rgb = img.rgb if img.rgb.dtype == np.uint8 else np.clip(img.rgb, 0, 255).astype(np.uint8)
            rgb_cache.append(rgb.copy())
            K_cache.append(np.asarray(img.intrinsic, dtype=np.float64) if hasattr(img, "intrinsic") else None)
            w2c_cache.append(np.asarray(img.extrinsic_w2c, dtype=np.float64) if hasattr(img, "extrinsic_w2c") else None)

            prob = seg_floor_prob(rgb)
            seg = (prob >= 0.5)
            per_view_probs.append(prob)
            valid_depth = np.isfinite(img.depth) & (img.depth > 0) & (img.depth < 4)
            Pw = img.points_world
            finite_pw = np.isfinite(Pw).all(-1)
            m = seg & valid_depth & finite_pw

            if export:
                import numpy as _np
                base = rgb.astype(_np.float32)
                orange = _np.array([255, 165, 0], _np.float32)
                for c in range(3):
                    ch = base[..., c]
                    ch[seg] = orange[c] * 0.35 + ch[seg] * 0.65
                    base[..., c] = ch
                overlays.append(base.clip(0, 255).astype(np.uint8))

            if np.count_nonzero(m) >= 50:
                P = Pw[m].astype(np.float64)
                wloc = (prob[m].astype(np.float64) + 1e-3)
                wloc /= (wloc.sum() + 1e-12)
                mu = (wloc[:, None] * P).sum(0)
                X0 = P - mu
                Xw = (np.sqrt(wloc)[:, None] * X0)
                _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
                n = Vt[-1]
                d = -n @ mu
                per_view_planes.append({"normal_world": n.astype(np.float64), "offset_world": float(d)})
                all_pts.append(P.astype(np.float32))
                all_w.append((prob[m].astype(np.float32) + 1e-3))
            else:
                per_view_planes.append(None)

        if not all_pts:
            return {
                "floor_corners_world": None,
                "floor_normal": None,
                "floor_offset": None,
                "wall_corners_world": None,
                "wall_normal": None,
                "wall_offset": None,
                "mean_lookat_world": None,
                "arc_end0_world": None,
                "arc_end1_world": None,
                "chord_midpoint_world": None,
                "wall_overshoot_m": wall_overshoot_m,
            }

        X = np.concatenate(all_pts, 0).reshape(-1, 3)
        w = np.concatenate(all_w, 0).reshape(-1)
        X = X[np.isfinite(X).all(-1)]
        w = w[:X.shape[0]]

        if X.shape[0] > 1_000_000:
            psub = w / (w.sum() + 1e-12)
            idx = rng.choice(X.shape[0], 1_000_000, replace=False, p=psub)
            X, w = X[idx], w[idx]

        scale = float(np.median(np.linalg.norm(X, axis=1)))
        tau = max(0.005, 0.01 * scale)
        iters = 3000
        p_full = w / (w.sum() + 1e-12) if np.isfinite(w).all() and w.sum() > 0 else None
        best_inl = None
        best_cnt = -1
        for _ in range(iters):
            idx = rng.choice(X.shape[0], 3, replace=False, p=p_full)
            a, b, c = X[idx]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-9: continue
            n /= nn
            d = -n @ a
            dist = np.abs(X @ n + d)
            inl = dist <= tau
            cnt = int(inl.sum())
            if cnt > best_cnt:
                best_cnt = cnt
                best_inl = inl

        if best_inl is None:
            mu = (w[:, None] * X).sum(0)
            X0 = X - mu
            Xw = (np.sqrt(np.maximum(w, 1e-12))[:, None] * X0)
            _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
            n_final = Vt[-1]
            d_final = -n_final @ mu
            inliers = np.abs(X @ n_final + d_final) <= tau
        else:
            Xr = X[best_inl]
            wr = (w[best_inl] + 1e-6)
            wr = wr / (wr.sum() + 1e-12)
            mu = (wr[:, None] * Xr).sum(0)
            X0 = Xr - mu
            Xw = (np.sqrt(wr)[:, None] * X0)
            _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
            n_final = Vt[-1]
            d_final = -n_final @ mu
            inliers = best_inl

        floor_normal = n_final.astype(np.float64)
        floor_offset = float(d_final)
        Xin = X[inliers]

        # PCA rect (fallback)
        if Xin.shape[0] >= 3:
            mu_f = Xin.mean(0)
            X0f = Xin - mu_f
            Cf = (X0f.T @ X0f) / max(Xin.shape[0] - 1, 1)
            _, evf = np.linalg.eigh(Cf)
            e1f = evf[:, 2]
            e1f = e1f - floor_normal * (e1f @ floor_normal)
            e1f /= (np.linalg.norm(e1f) + 1e-12)
            e2f = np.cross(floor_normal, e1f)
            e2f /= (np.linalg.norm(e2f) + 1e-12)
            U = X0f @ e1f
            V = X0f @ e2f
            umin0, umax0 = float(U.min()), float(U.max())
            vmin0, vmax0 = float(V.min()), float(V.max())
            pca_rect = np.stack([
                mu_f + umin0 * e1f + vmin0 * e2f,
                mu_f + umax0 * e1f + vmin0 * e2f,
                mu_f + umax0 * e1f + vmax0 * e2f,
                mu_f + umin0 * e1f + vmax0 * e2f,
            ], 0).astype(np.float64)
        else:
            pca_rect = None

        # ---- Wall construction ----
        cams = [im for im in rgbd_images if (im is not None and im.extrinsic_w2c is not None)]
        mean_lookat = None
        arc_end0 = None
        arc_end1 = None
        chord_mid = None
        n_wall = None
        d_wall = None
        wall_corners = None
        s_min = None
        s_max = None
        a1w = None
        if len(cams) >= 2:
            c2ws = [np.linalg.inv(im.extrinsic_w2c).astype(np.float64) for im in cams]
            Cw = np.stack([E[:3, 3] for E in c2ws], 0)
            Rw = np.stack([E[:3, :3] for E in c2ws], 0)
            fwd = Rw[:, :, 2]
            fwd /= (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12)

            I = np.eye(3, dtype=np.float64)
            M = np.zeros((3, 3), dtype=np.float64)
            b = np.zeros((3,), dtype=np.float64)
            for i in range(len(cams)):
                fi = fwd[i]
                A = I - np.outer(fi, fi)
                M += A
                b += A @ Cw[i]
            try:
                mean_lookat = np.linalg.solve(M, b)
            except np.linalg.LinAlgError:
                mean_lookat = Cw.mean(0)

            arc_end0 = Cw[0]
            arc_end1 = Cw[-1]
            chord = arc_end1 - arc_end0
            clen = np.linalg.norm(chord)
            if clen < 1e-9:
                t_hat = np.cross(floor_normal, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(t_hat) < 1e-6:
                    t_hat = np.cross(floor_normal, np.array([0.0, 1.0, 0.0]))
                t_hat /= (np.linalg.norm(t_hat) + 1e-12)
            else:
                t_hat = chord / clen

            n_up = floor_normal / (np.linalg.norm(floor_normal) + 1e-12)
            s_look = float(np.dot(n_up, mean_lookat) + floor_offset)
            if s_look < 0.0: n_up = -n_up

            chord_mid = 0.5 * (arc_end0 + arc_end1)
            v_to_look = mean_lookat - chord_mid
            v_norm = np.linalg.norm(v_to_look)
            if v_norm < 1e-9:
                v_dir = np.cross(t_hat, n_up)
                if np.linalg.norm(v_dir) < 1e-9: v_dir = np.array([1.0, 0.0, 0.0])
                v_dir /= (np.linalg.norm(v_dir) + 1e-12)
            else:
                v_dir = v_to_look / v_norm

            wall_back_offset_m = float(wall_overshoot_m)
            anchor_pre = chord_mid - wall_back_offset_m * v_dir
            h_anchor = float(np.dot(floor_normal, anchor_pre) + floor_offset)
            anchor_on_floor = anchor_pre - h_anchor * floor_normal

            a1w = t_hat / (np.linalg.norm(t_hat) + 1e-12)  # width
            a2w = n_up  # vertical
            n_wall_est = np.cross(a2w, a1w)
            n_wall_est /= (np.linalg.norm(n_wall_est) + 1e-12)  # into room (horizontal)
            d_wall_est = -float(np.dot(n_wall_est, anchor_on_floor))

            def proj_to_plane(P):
                return P - (np.dot(n_wall_est, P) + d_wall_est) * n_wall_est

            P0p = proj_to_plane(arc_end0)
            P1p = proj_to_plane(arc_end1)
            s0 = float(np.dot(P0p - anchor_on_floor, a1w))
            s1 = float(np.dot(P1p - anchor_on_floor, a1w))
            s_min = min(s0, s1) - wall_pad_width_m
            s_max = max(s0, s1) + wall_pad_width_m

            wall_corners = np.stack([
                anchor_on_floor + s_min * a1w + 0.0 * a2w,
                anchor_on_floor + s_max * a1w + 0.0 * a2w,
                anchor_on_floor + s_max * a1w + wall_height_m * a2w,
                anchor_on_floor + s_min * a1w + wall_height_m * a2w,
            ], 0).astype(np.float64)

            if wall_corners.shape[0] >= 3:
                nw = np.cross(wall_corners[1] - wall_corners[0], wall_corners[3] - wall_corners[0])
                if np.linalg.norm(nw) > 1e-12:
                    nw = nw / np.linalg.norm(nw)
                    n_wall = nw.astype(np.float64)
                    d_wall = float(-nw @ wall_corners[0])
                else:
                    n_wall = n_wall_est
                    d_wall = d_wall_est
            else:
                n_wall = n_wall_est
                d_wall = d_wall_est

            if mean_lookat is not None:
                v_ref = mean_lookat - anchor_on_floor
                v_ref = v_ref - n_up * (v_ref @ n_up)
                if np.dot(n_wall, v_ref) < 0:
                    n_wall = -n_wall

        # ---- Floor rectangle: share bottom [w0,w1], same width; depth scaled by floor_depth_scale ----
        floor_corners = None
        if Xin.shape[0] >= 3:
            if wall_corners is not None and a1w is not None and s_min is not None and s_max is not None:
                w0 = wall_corners[0]  # left-bottom
                w1 = wall_corners[1]  # right-bottom
                e_t = w1 - w0
                Lw = float(np.linalg.norm(e_t))
                if Lw < 1e-12:
                    floor_corners = pca_rect
                else:
                    e_t /= Lw
                    e_n = (n_wall if n_wall is not None else np.cross(floor_normal, e_t))
                    e_n = e_n / (np.linalg.norm(e_n) + 1e-12)
                    if mean_lookat is not None:
                        v_ref = mean_lookat - w0
                        v_ref = v_ref - floor_normal * (v_ref @ floor_normal)
                        if np.dot(e_n, v_ref) < 0: e_n = -e_n

                    V_all = (Xin - w0) @ e_n
                    V_pos = V_all[V_all > 0]
                    if V_pos.size >= 10:
                        vmax = float(np.percentile(V_pos, 90))
                    elif V_pos.size > 0:
                        vmax = float(np.max(V_pos))
                    else:
                        if pca_rect is not None:
                            v_candidates = (pca_rect - w0) @ e_n
                            vmax = float(np.max(v_candidates) - np.min(v_candidates))
                        else:
                            vmax = 0.3 * max(1.0, scale)

                    vmax = float(max(vmax, 0.1 * max(1.0, scale)))
                    # --- Apply user scale to increase depth ---
                    vmax *= float(max(floor_depth_scale, 0.0))

                    c00 = w0
                    c10 = w1
                    c11 = w1 + vmax * e_n
                    c01 = w0 + vmax * e_n
                    floor_corners = np.stack([c00, c10, c11, c01], 0).astype(np.float64)
            else:
                floor_corners = pca_rect

            if floor_corners is not None and floor_corners.shape[0] >= 3:
                nf = np.cross(floor_corners[1] - floor_corners[0], floor_corners[3] - floor_corners[0])
                if np.linalg.norm(nf) > 1e-12:
                    nf = nf / np.linalg.norm(nf)
                    floor_normal = nf.astype(np.float64)
                    floor_offset = float(-nf @ floor_corners[0])

        if export:
            import cv2, open3d as o3d
            from pathlib import Path as _Path
            base = _Path(str(export_path if export_path is not None else 'floor_wall_debug'))
            base.parent.mkdir(parents=True, exist_ok=True)

            def _project_points(K: np.ndarray, w2c: np.ndarray, Xw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                Xw = np.asarray(Xw, dtype=np.float64)
                Rt = w2c[:3, :3]
                t = w2c[:3, 3]
                Xc = (Rt @ Xw.T).T + t
                z = Xc[:, 2]
                valid = z > 1e-6
                uv = np.empty((Xc.shape[0], 2), dtype=np.float64)
                uv[:, 0] = (K[0, 0] * (Xc[:, 0] / np.maximum(z, 1e-6))) + K[0, 2]
                uv[:, 1] = (K[1, 1] * (Xc[:, 1] / np.maximum(z, 1e-6))) + K[1, 2]
                return uv, valid

            Xin_loc = X[inliers]
            if Xin_loc.shape[0] >= 3:
                mu_f = Xin_loc.mean(0)
                X0 = Xin_loc - mu_f
                C = (X0.T @ X0) / max(Xin_loc.shape[0] - 1, 1)
                _, ev = np.linalg.eigh(C)
                e1 = ev[:, 2]
                e1 = e1 - floor_normal * (e1 @ floor_normal)
                e1 /= (np.linalg.norm(e1) + 1e-12)
                e2 = np.cross(floor_normal, e1)
                e2 /= (np.linalg.norm(e2) + 1e-12)
                U = X0 @ e1
                V = X0 @ e2
                umin, umax = U.min(), U.max()
                vmin, vmax = V.min(), V.max()
                Nu, Nv = 10, 8
                u_line = np.linspace(umin, umax, Nu)
                v_line = np.linspace(vmin, vmax, Nv)
            else:
                u_line = np.linspace(-scale, scale, 10)
                v_line = np.linspace(-scale, scale, 8)
                mu_f = -floor_normal * floor_offset
                e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                e1 = e1 - floor_normal * (e1 @ floor_normal)
                e1 /= (np.linalg.norm(e1) + 1e-12)
                e2 = np.cross(floor_normal, e1)
                e2 /= (np.linalg.norm(e2) + 1e-12)

            for vi, (ov, K, w2c, rgb) in enumerate(zip(overlays, K_cache, w2c_cache, rgb_cache)):
                if ov is None or K is None or w2c is None or rgb is None: continue
                H, W = ov.shape[:2]
                canvas_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)

                def plane_uv_to_xyz(u, v):
                    return mu_f + u * e1 + v * e2

                Nsamp = 64
                v_samp = np.linspace(v_line.min(), v_line.max(), Nsamp)
                u_samp = np.linspace(u_line.min(), u_line.max(), Nsamp)

                for u in u_line:
                    X_line = np.stack([plane_uv_to_xyz(u, vv) for vv in v_samp], axis=0)
                    uv, valid = _project_points(K, w2c, X_line)
                    uv = uv[valid]
                    if uv.shape[0] >= 2:
                        pts = np.round(uv).astype(np.int32)
                        inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
                        pts = pts[inside]
                        if pts.shape[0] >= 2:
                            cv2.polylines(canvas_bgr, [pts.reshape(-1, 1, 2)], False, (255, 255, 0), 1, cv2.LINE_AA)

                for v in v_line:
                    X_line = np.stack([plane_uv_to_xyz(uu, v) for uu in u_samp], axis=0)
                    uv, valid = _project_points(K, w2c, X_line)
                    uv = uv[valid]
                    if uv.shape[0] >= 2:
                        pts = np.round(uv).astype(np.int32)
                        inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
                        pts = pts[inside]
                        if pts.shape[0] >= 2:
                            cv2.polylines(canvas_bgr, [pts.reshape(-1, 1, 2)], False, (255, 255, 0), 1, cv2.LINE_AA)

                png_path = base.with_name(f"{base.stem}_overlay_view{vi:02d}.png")
                cv2.imwrite(str(png_path), canvas_bgr)

            import open3d as o3d
            pts_all, cols_all = [], []
            ref_idx = next((i for i, im in enumerate(rgbd_images) if im is not None and im.depth is not None), None)
            if ref_idx is not None:
                ref_img = rgbd_images[ref_idx]
                ref_pcd = ref_img.unproject().open3d
                pts_ref = np.asarray(ref_pcd.points, dtype=np.float64)
                cols_ref = np.asarray(ref_pcd.colors, dtype=np.float64)
                if pts_ref.size:
                    pts_all.append(pts_ref)
                    cols_all.append(cols_ref)

            def sample_sphere_points(center, radius, n_pts):
                phi = (1.0 + 5 ** 0.5) / 2.0
                i = np.arange(n_pts, dtype=np.float64)
                z = 1.0 - 2.0 * (i + 0.5) / n_pts
                theta = 2.0 * np.pi * i / phi
                rxy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
                unit = np.stack([rxy * np.cos(theta), rxy * np.sin(theta), z], axis=1)
                return center[None, :] + radius * unit

            if floor_corners is not None and floor_corners.shape[0] >= 3:
                r_small = max(0.02 * scale, 0.02)
                n_per = 256
                pink = np.array([[1.0, 0.2, 0.8]], dtype=np.float64)
                for c in floor_corners:
                    sp = sample_sphere_points(c, r_small, n_per)
                    pts_all.append(sp)
                    cols_all.append(np.repeat(pink, sp.shape[0], axis=0))

            if wall_corners is not None:
                r_small = max(0.02 * scale, 0.02)
                n_per = 256
                green = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
                for c in wall_corners:
                    sp = sample_sphere_points(c, r_small, n_per)
                    pts_all.append(sp)
                    cols_all.append(np.repeat(green, sp.shape[0], axis=0))

            def sample_quad(corners: np.ndarray, nu: int, nv: int):
                corners = np.asarray(corners, dtype=np.float64)
                if corners.shape[0] == 3:
                    p0, p1, p2 = corners
                    us = np.linspace(0.0, 1.0, nu)
                    vs = np.linspace(0.0, 1.0, nv)
                    P = []
                    for u in us:
                        vmax = 1.0 - u
                        vv = np.linspace(0.0, vmax, nv)
                        A = (1 - u - vv)[:, None] * p0[None, :] + u * p1[None, :] + vv[:, None] * p2[None, :]
                        P.append(A)
                    return np.vstack(P) if P else np.zeros((0, 3), dtype=np.float64)
                else:
                    p0, p1, p2, p3 = corners[:4]
                    us = np.linspace(0.0, 1.0, nu)
                    vs = np.linspace(0.0, 1.0, nv)
                    uu, vv = np.meshgrid(us, vs)
                    P = ((1 - uu) * (1 - vv))[:, :, None] * p0 + (uu * (1 - vv))[:, :, None] * p1 + \
                        (uu * vv)[:, :, None] * p2 + ((1 - uu) * vv)[:, :, None] * p3
                    return P.reshape(-1, 3)

            if floor_corners is not None and floor_corners.shape[0] >= 3:
                Pf = sample_quad(floor_corners, 200, 200)
                if Pf.size:
                    pts_all.append(Pf)
                    cols_all.append(np.tile(np.array([[1.0, 0.2, 0.8]], dtype=np.float64), (Pf.shape[0], 1)))

            if wall_corners is not None and wall_corners.shape[0] >= 3:
                Pw = sample_quad(wall_corners, 160, 160)
                if Pw.size:
                    pts_all.append(Pw)
                    cols_all.append(np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (Pw.shape[0], 1)))

            pts_all = np.vstack(pts_all) if pts_all else np.zeros((0, 3), dtype=np.float64)
            cols_all = np.vstack(cols_all) if cols_all else np.zeros((0, 3), dtype=np.float64)
            pcd_all = o3d.geometry.PointCloud()
            pcd_all.points = o3d.utility.Vector3dVector(pts_all)
            pcd_all.colors = o3d.utility.Vector3dVector(cols_all)
            ply_path = base.with_suffix('.ply')
            o3d.io.write_point_cloud(ply_path, pcd_all)

        return {
            "floor_corners_world": None if floor_corners is None else floor_corners.astype(np.float64),
            "floor_normal": None if floor_corners is None else floor_normal.astype(np.float64),
            "floor_offset": None if floor_corners is None else float(floor_offset),
            "wall_corners_world": None if wall_corners is None else wall_corners.astype(np.float64),
            "wall_normal": None if wall_corners is None else n_wall.astype(np.float64),
            "wall_offset": None if wall_corners is None else float(d_wall),
            "mean_lookat_world": None if mean_lookat is None else mean_lookat.astype(np.float64),
            "arc_end0_world": None if arc_end0 is None else arc_end0.astype(np.float64),
            "arc_end1_world": None if arc_end1 is None else arc_end1.astype(np.float64),
            "chord_midpoint_world": None if chord_mid is None else chord_mid.astype(np.float64),
            "wall_overshoot_m": wall_overshoot_m,
        }

    @classmethod
    def from_path(
            cls,
            rgb_path: Path,
            depth_path: Union[Path, None],
            intrinsic: Union[Path, np.ndarray],
            extrinsic_w2c: Union[Path, np.ndarray],
            mask_path: Optional[Path] = None,
            is_inverse_depth: bool = False,
            depth_divide: float = 1.0,
    ) -> 'RGBDImage':
        """
        Create an RGBDImage from file paths.

        Parameters
        ----------
        rgb_path : Path
            Path to the RGB image file.
        depth_path : Union[Path, None]
            Path to the depth image file.
        intrinsic : Union[Path, np.ndarray]
            Camera intrinsic matrix of shape (3, 3).
        extrinsic_w2c : Union[Path, np.ndarray]
            Camera extrinsic matrix of shape (4, 4).
        mask_path : Optional[Path]
            Path to the valid pixels mask file. If None, all pixels are considered valid.
        is_inverse_depth : bool, optional
            If True, the depth values are considered as inverse depth (1/depth). Default is False.
        depth_divide : float, optional
            Divide the depth value by this number. Default is 1.

        Returns
        -------
        RGBDImage
            A new RGBDImage instance.
        """
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR_RGB)
        if depth_path is not None:
            if depth_path.suffix == '.png':
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                if depth_divide != 1.0:
                    depth /= depth_divide
            elif depth_path.suffix == '.npy':
                depth = np.load(str(depth_path)).astype(np.float32).clip(min=0, max=10_000)
            else:
                raise ValueError(f"Unsupported depth file format: {depth_path.suffix}. Supported formats are .png and .npy.")

            if is_inverse_depth:
                depth /= 2 ** 15
                depth = np.where(depth > 1e-6, 1.0 / depth, 0.0)
            else:
                if depth.dtype == np.uint16 and depth.max() > 10_000:
                    depth /= 10_000.0
                else:
                    depth /= 1_000.0
            depth = depth.squeeze()
        else:
            depth = None
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.ones_like(depth, dtype=np.float32)
        mask = mask.squeeze()
        intrinsic = np.load(str(intrinsic)) if isinstance(intrinsic, (Path, str)) else intrinsic
        intrinsic = intrinsic.squeeze()
        extrinsic_w2c = np.load(str(extrinsic_w2c)) if isinstance(extrinsic_w2c, (Path, str)) else extrinsic_w2c
        extrinsic_w2c = extrinsic_w2c.squeeze()
        if extrinsic_w2c.shape == (3, 4):
            extrinsic_w2c = np.vstack([extrinsic_w2c, [0, 0, 0, 1]])
        return cls(rgb=rgb, mask=mask > 0.5, depth=depth, intrinsic=intrinsic, extrinsic_w2c=extrinsic_w2c)

    @classmethod
    def from_thuman(cls, thuman_root: Path, model: int, main_cam_idx: int, sub_cam_idx: int = 0, split: Optional[Union[Literal['train'], Literal['val']]] = None) -> 'RGBDImage':
        """
        Create an RGBDImage from a THuman dataset.

        Parameters
        ----------
        thuman_root : Path
            The root directory of the THuman dataset.
        model : int
            The model index (0-9).
        main_cam_idx : int
            The main camera index (0-7).
        sub_cam_idx : int, optional
            The sub camera index. 0: left, 1: right, 2-4: intermediate cameras.
        split : Union[Literal['train'], Literal['val']], optional
            The dataset split to use. If None, it will automatically determine the split based on the existence of the train or val directories.

        Returns
        -------
        RGBDImage
            A new RGBDImage instance containing the RGB and depth images from the THuman dataset.
        """
        if split is None:
            split = 'train' if (thuman_root / 'train' / 'img' / f'{model:04d}_000').exists() else 'val'
        # create data paths
        rgb_path = thuman_root / split / 'img' / f'{model:04d}_{main_cam_idx:03d}' / f'{sub_cam_idx}.jpg'
        depth_path = thuman_root / split / 'depth' / f'{model:04d}_{main_cam_idx:03d}' / f'{sub_cam_idx}.png'
        mask_path = thuman_root / split / 'mask' / f'{model:04d}_{main_cam_idx:03d}' / f'{sub_cam_idx}.png'
        intrinsic_path = thuman_root / split / 'parm' / f'{model:04d}_{main_cam_idx:03d}' / f'{sub_cam_idx}_intrinsic.npy'
        extrinsic_path = thuman_root / split / 'parm' / f'{model:04d}_{main_cam_idx:03d}' / f'{sub_cam_idx}_extrinsic.npy'

        # load data and create RGBDImage
        return cls.from_path(rgb_path, depth_path, intrinsic_path, extrinsic_path, mask_path=mask_path, is_inverse_depth=True)

    @classmethod
    def from_session(
            cls,
            session_root: Path,
            calibration_data: Union[Path, CalibrationData],
            cam_idx: int,
            color_ts: int,
            depth_ts: Union[int, None],
            depth_filter: Literal['aligned', 'bilateral_spatial', 'bilateral_temporal'] = 'aligned'
    ) -> 'RGBDImage':
        """
        Create PixelPoints from a session's camera data.

        Parameters
        ----------
        session_root : Path
            The root directory of the session.
        calibration_data : Union[Path, CalibrationData]
            The root directory of the calibration session.
        cam_idx : int
            The camera index.
        color_ts : int
            The timestamp for the color image.
        depth_ts : Union[int, None]
            The timestamp for the depth image. If None, no depth image will be loaded, allowing for deferred depth estimation.
        depth_filter : Literal['aligned', 'bilateral_spatial', 'bilateral_temporal'], optional
            The depth filtering method to use. Default is 'bilateral_spatial'.

        Returns
        -------
        PixelPoints
            A new PixelPoints instance created from the RGBDImage.
        """
        # load intrinsic and extrinsic matrices
        if isinstance(calibration_data, (Path, str)):
            calibration_data = CalibrationData.from_session(calibration_data)
        intrinsic = calibration_data.intrinsics[calibration_data.cam_names.index(f'orbbec/cam{cam_idx:02d}')].detach().cpu().numpy()
        extrinsic_w2c = calibration_data.extrinsics_w2c[calibration_data.cam_names.index(f'orbbec/cam{cam_idx:02d}')].detach().cpu().numpy()
        # create rgb/depth paths
        rgb_path = session_root / 'orbbec' / f'cam{cam_idx:02d}' / 'color' / f'{color_ts}.jpg'
        mask_path = session_root / 'orbbec' / f'cam{cam_idx:02d}' / 'mask' / f'{color_ts}.jpg'
        if depth_ts is not None:
            depth_path_png = session_root / 'orbbec' / f'cam{cam_idx:02d}' / (f'depth_{depth_filter}' if depth_filter in ['aligned'] else f'depth_filtering_{depth_filter}') / f'{depth_ts}.png'
            depth_path = depth_path_png if depth_path_png.exists() else depth_path_png.with_suffix('.npy')
        else:
            depth_path = None
        # load data and create RGBDImage
        return cls.from_path(rgb_path, depth_path, intrinsic, extrinsic_w2c=extrinsic_w2c, mask_path=mask_path, is_inverse_depth=False, depth_divide=13.0)


class PixelPoints:
    O3D_VISUALIZER_CACHE = {}
    PYRENDER_RENDERER_CACHE = {}

    def __init__(self, pixel_points: np.ndarray, pixel_colors: Optional[np.ndarray] = None, pixel_valid: Optional[np.ndarray] = None, pixel_features: Optional[Dict[str, np.ndarray]] = None, extrinsics_c2w: Optional[np.ndarray] = None):
        """
        Initialize pixel-wise point-clouds.

        Parameters
        ----------
        pixel_points : np.ndarray
            The 3D points in world coordinates of shape (H, W, 3).
        pixel_colors : Optional[np.ndarray]
            The colors corresponding to the points of shape (H, W, 3). If None, random colors will be assigned.
        pixel_valid : Optional[np.ndarray]
            A mask indicating valid pixels of shape (H, W). If None, all pixels are considered valid.
        pixel_features : Optional[Dict[str, np.ndarray]]
            Optional dictionary of additional features, e.g., normals or semantic segmentation masks.
            Each feature should be a numpy array of shape (H, W) or (H, W, C).
            Default is None.
        """
        self.points = pixel_points
        self.colors = pixel_colors if pixel_colors is not None else np.random.rand(*self.points.shape).astype(pixel_points.dtype)
        self.valid = np.asarray(pixel_valid, dtype=bool) if pixel_valid is not None else np.ones(self.points.shape[:-1], dtype=bool)
        self.features = pixel_features if pixel_features is not None else {}
        self.extrinsics_c2w: Optional[np.ndarray] = extrinsics_c2w

    def __repr__(self) -> str:
        return f"PointCloud(points_shape={self.points.shape}, colors_shape={self.colors.shape})"

    @functools.cached_property
    def normals(self) -> np.ndarray:
        """
        Get the normals of the point cloud. If not available, the normals are estimated using KDTree-based nearest neighbors.

        Returns
        -------
        np.ndarray
            The normals of the point cloud of shape (H, W, 3).
        """
        if 'normals' in self.features:
            normal_full = self.features['normals']
            normal_full[(normal_full == 0).all(-1)] = -1.0
        else:
            pcd_o3d = self.open3d
            if pcd_o3d.has_normals():
                normal_valid = np.asarray(pcd_o3d.normals).reshape(self.points.shape[:-1] + (3,))
            else:
                pcd_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=10))
                # normal_valid = pcd_o3d.normalize_normals().normals

                voxel_size = 0.0025  # tune this based on data scale
                pcd_for_mesh = pcd_o3d.voxel_down_sample(voxel_size)
                pcd_for_mesh.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=2 * voxel_size,  # or larger
                        max_nn=20  # larger support for smoother normals
                    )
                )
                pcd_for_mesh.orient_normals_consistent_tangent_plane(k=50)
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd_for_mesh,
                    depth=7
                )
                # Remove low-density garbage
                densities = np.asarray(densities)
                density_thresh = np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(densities < density_thresh)
                # Crop to original bbox
                bbox = pcd_o3d.get_axis_aligned_bounding_box()
                mesh = mesh.crop(bbox)
                # Clean mesh
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                # Simplify (coarser = smoother)
                if len(mesh.triangles) > 200_000:
                    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=200_000)
                # Stronger smoothing (geometry)
                mesh = mesh.filter_smooth_simple(number_of_iterations=5)
                mesh = mesh.filter_smooth_taubin(number_of_iterations=15)
                # Recompute vertex normals after smoothing
                mesh.compute_vertex_normals()
                mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
                mesh_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                tree = cKDTree(mesh_vertices)
                points_np = np.asarray(pcd_o3d.points, dtype=np.float32)
                _, idx = tree.query(points_np, k=1, workers=-1)
                normal_valid = mesh_normals[idx]

                if self.extrinsics_c2w is not None:
                    # camera center in world coordinates from c2w
                    cam_center_w = self.extrinsics_c2w[:3, 3].astype(np.float32)

                    # points in world coordinates, same order as normal_valid
                    points_w = np.asarray(pcd_o3d.points, dtype=np.float32)

                    # vector from point to camera (so normals will point towards camera)
                    view_vec = cam_center_w[None, :] - points_w  # shape (N, 3)

                    # dot product between normal and view direction
                    dot = np.einsum("ij,ij->i", normal_valid, view_vec)

                    # flip normals that are pointing away from the camera
                    flip_mask = dot < 0.0
                    normal_valid[flip_mask] *= -1.0

            normal_full = -np.ones_like(self.points, dtype=np.float32)
            if self.valid is not None:
                normal_full[self.valid] = normal_valid
            else:
                normal_full = normal_valid.reshape(self.points.shape)
        return normal_full

    @functools.cached_property
    def open3d(self) -> o3d.geometry.PointCloud:
        """
        Convert the pixel points to an Open3D PointCloud.

        This version softly removes boundary pixels: we compute a distance transform
        on the foreground mask and invalidate points that are too close to the
        silhouette, which eliminates hard edges / background spill in renders.
        """
        # ---- Soft trim near the FG boundary -----------------
        # If no mask exists yet, assume all pixels valid so we can create one.
        if getattr(self, "valid", None) is None:
            self.valid = np.ones(self.points.shape[:2], dtype=bool)

        # Parameters for boundary trimming (tweak if you want different strength)
        CONTRACT_PX = 1  # optional pre-erosion to reduce spill
        TRIM_PX = 2.0  # remove anything within this many pixels from the boundary

        # Build a binary mask (uint8) and optionally erode to contract the FG a bit
        m = self.valid.astype(np.uint8)
        if CONTRACT_PX > 0:
            k = int(CONTRACT_PX)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            m = cv2.erode(m, kernel, iterations=1)

        # Distance to background, only inside FG (0 at boundary, grows inward)
        dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)

        # Keep only pixels at least TRIM_PX from the boundary
        keep_interior = dist_in >= float(TRIM_PX)

        # Update self.valid in-place (invalidate shallow-edge pixels)
        self.valid &= keep_interior

        # ---- Create the Open3D point cloud from remaining valid pixels -------
        pcd = o3d.geometry.PointCloud()
        if self.valid is not None:
            pts_valid = self.points[self.valid]
            cols_valid = self.colors[self.valid]
            pcd.points = o3d.utility.Vector3dVector(pts_valid)
            pcd.colors = o3d.utility.Vector3dVector(cols_valid)
            valid_idx = np.flatnonzero(self.valid.flatten())
        else:
            # (Fallback: no mask at all — unlikely here because we created one above)
            pts_flat = self.points.reshape(-1, 3)
            cols_flat = self.colors.reshape(-1, 3)
            pcd.points = o3d.utility.Vector3dVector(pts_flat)
            pcd.colors = o3d.utility.Vector3dVector(cols_flat)
            valid_idx = np.arange(self.points.shape[0] * self.points.shape[1])

        # Radius outlier removal (unchanged)
        cl, inlier_idx = pcd.remove_radius_outlier(nb_points=30, radius=0.03)

        # Reflect outlier removal back into self.valid so downstream logic stays consistent
        retained = set(inlier_idx)
        removed_indices = [i for i in range(len(valid_idx)) if i not in retained]
        if self.valid is None:
            self.valid = np.ones((self.points.shape[0], self.points.shape[1]), dtype=bool)
        for i in removed_indices:
            self.valid.flat[valid_idx[i]] = False
        return cl

    # noinspection PyUnusedLocal
    def project_features(self, target_intrinsic: np.ndarray, target_extrinsic: np.ndarray, target_image_size_hw: Tuple[int, int], is_c2w: bool = True) -> Dict[str, np.ndarray]:
        """
        Project the features to the pixel points.

        target_intrinsic : np.ndarray
            Target camera intrinsic matrix of shape (3, 3).
        target_extrinsic : np.ndarray
            Target camera extrinsic matrix of shape (4, 4).
        target_image_size_hw : Tuple[int, int]
            The heigh and width of the target image.
        is_c2w : bool, optional
            If True, the target extrinsic is assumed to be camera-to-world (R,T). Default is True, i.e. the extrinsic is the pose matrix (translation is wrt world frame).

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of projected features with the same keys as the input.
        """
        projected_features = copy.copy(self.features)
        return projected_features

    def project(self, target_intrinsic: np.ndarray, target_extrinsic: np.ndarray, target_image_size_hw: Tuple[int, int], is_c2w: bool, point_size: float = 2.0, use_cache: bool = False, renderer=None, lib: Literal['pyrender', 'open3d'] = 'open3d', **floor_wall_kwargs) -> RGBDImage:
        """
        Project the pixel points into an RGBDImage using the target intrinsic and extrinsic matrices.

        Parameters
        ----------
        target_intrinsic : np.ndarray
            Target camera intrinsic matrix of shape (3, 3).
        target_extrinsic : np.ndarray
            Target camera extrinsic matrix of shape (4, 4).
        target_image_size_hw : Tuple[int, int]
            The heigh and width of the target image.
        is_c2w : bool, optional
            If True, the target extrinsic is assumed to be camera-to-world (R,T). Default is True, i.e. the extrinsic is the pose matrix (translation is wrt world frame).
        point_size : float, optional
            The size of the points in the rendered image. Default is 1.0.
        use_cache : bool, optional
            If True, use cached renderer for the target image size. Default is False.
        renderer : Optional[o3d.visualization.rendering.OffscreenRenderer]
            An optional renderer to use. If None, a new renderer will be created.
        lib: Literal['pyrender', 'open3d'], optional
            Which library to use for rendering. Default is 'pyrender'.

        Returns
        -------
        RGBDImage
            A new RGBDImage instance containing the projected points and colors.
        """
        if lib == 'pyrender':
            return self._project_pyrender(target_intrinsic, target_extrinsic, target_image_size_hw, is_c2w, point_size, use_cache, renderer, **floor_wall_kwargs)
        return self._project_open3d(target_intrinsic, target_extrinsic, target_image_size_hw, is_c2w, point_size, use_cache, renderer, **floor_wall_kwargs)

    def _project_pyrender(
            self,
            target_intrinsic: np.ndarray,
            target_extrinsic: np.ndarray,
            target_image_size_hw: Tuple[int, int],
            is_c2w: bool,
            point_size: float = 2.0,
            use_cache: bool = False,
            renderer: Optional[tuple] = None,
            **floor_wall_kwargs
    ) -> "RGBDImage":
        import pyrender
        if renderer is None:
            cache_key = f'{target_image_size_hw}_{point_size}'
            if use_cache and cache_key in self.__class__.PYRENDER_RENDERER_CACHE:
                renderer, scene = self.__class__.PYRENDER_RENDERER_CACHE[cache_key]
            else:
                h, w = target_image_size_hw
                renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
                renderer.point_size = point_size
                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=[1.0, 1.0, 1.0])
                if use_cache:
                    self.__class__.PYRENDER_RENDERER_CACHE[cache_key] = renderer, scene
        else:
            renderer, scene = renderer

        # scene with the point cloud
        pcd = self.open3d
        mesh = pyrender.Mesh(
            primitives=[pyrender.Primitive(
                positions=np.asarray(pcd.points, dtype=np.float32),
                color_0=np.asarray(pcd.colors, dtype=np.float32),
                material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[1, 1, 1, 1.],
                    metallicFactor=0.0,
                    roughnessFactor=0.0,
                    smooth=False,
                    alphaMode='OPAQUE'
                ),
                mode=pyrender.constants.GLTF.POINTS
            )]
        )
        scene.add(mesh)

        # camera
        fx, fy = float(target_intrinsic[0, 0]), float(target_intrinsic[1, 1])
        cx, cy = float(target_intrinsic[0, 2]), float(target_intrinsic[1, 2])
        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.001, zfar=1000.0)

        c2w = target_extrinsic if is_c2w else np.linalg.inv(target_extrinsic)
        c2w[:3, 1:3] *= -1
        c2w[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
        scene.add(camera, pose=c2w.astype(np.float32))

        render_rgb, render_depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
        render_rgb = render_rgb[:, :, :3]
        render_depth = render_depth.astype(np.float32)
        render_valid = (render_depth > 0.0) & np.isfinite(render_depth)
        render_depth[~render_valid] = 0.0

        scene.clear()

        return RGBDImage(
            render_rgb,
            render_valid,
            render_depth,
            intrinsic=target_intrinsic,
            extrinsic_w2c=(target_extrinsic if not is_c2w else np.linalg.inv(target_extrinsic)),
            features=self.project_features(target_intrinsic, target_extrinsic, target_image_size_hw, is_c2w=is_c2w)
        )

    def _project_open3d(self,
                        target_intrinsic: np.ndarray,
                        target_extrinsic: np.ndarray,
                        target_image_size_hw: Tuple[int, int],
                        is_c2w: bool,
                        point_size: float = 2.0,
                        use_cache: bool = False,
                        renderer=None,
                        floor_normal: Optional[np.ndarray] = None,
                        floor_offset: Optional[np.ndarray] = None,
                        wall_normal: Optional[np.ndarray] = None,
                        wall_offset: Optional[np.ndarray] = None) -> RGBDImage:
        from pathlib import Path

        if renderer is None:
            if use_cache and target_image_size_hw in self.__class__.O3D_VISUALIZER_CACHE:
                renderer = self.__class__.O3D_VISUALIZER_CACHE[target_image_size_hw]
            else:
                renderer = o3d.visualization.rendering.OffscreenRenderer(*target_image_size_hw[::-1])
                renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
                # simple sun light so textured meshes are visible
                renderer.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 100000.0)
                renderer.scene.scene.enable_sun_light(True)
                if use_cache:
                    self.__class__.O3D_VISUALIZER_CACHE[target_image_size_hw] = renderer
        else:
            # ensure clean scene for re-use
            try:
                renderer.scene.clear_geometry()
            except Exception:
                pass

        H, W = target_image_size_hw
        fx, fy = float(target_intrinsic[0, 0]), float(target_intrinsic[1, 1])
        cx, cy = float(target_intrinsic[0, 2]), float(target_intrinsic[1, 2])

        c2w = target_extrinsic if is_c2w else np.linalg.inv(target_extrinsic)
        Cw = c2w[:3, 3].astype(np.float64)
        Rw = c2w[:3, :3].astype(np.float64)

        def make_plane_mesh(n: np.ndarray, d: float, color_rgb=(80, 200, 80), texture_path: Optional[Path] = None):
            n = np.asarray(n, dtype=np.float64)
            n /= (np.linalg.norm(n) + 1e-12)
            d = float(d)
            # in-plane basis
            a = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            u = np.cross(n, a)
            u /= (np.linalg.norm(u) + 1e-12)
            v = np.cross(n, u)
            v /= (np.linalg.norm(v) + 1e-12)
            x0 = -d * n

            # sample rays on image grid and intersect with plane
            grid = 9
            us = np.linspace(0, W - 1, grid, dtype=np.float64)
            vs = np.linspace(0, H - 1, grid, dtype=np.float64)
            U, V = np.meshgrid(us, vs)
            dir_cam = np.stack([(U - cx) / fx, (V - cy) / fy, np.ones_like(U)], axis=-1)
            dir_world = (dir_cam @ Rw.T)  # HxWx3
            denom = dir_world[..., 0] * n[0] + dir_world[..., 1] * n[1] + dir_world[..., 2] * n[2]
            numer = (n @ Cw) + d
            with np.errstate(divide='ignore', invalid='ignore'):
                t = -numer / denom
            ok = np.isfinite(t) & (np.abs(denom) > 1e-9) & (t > 0.05) & (t < 1000.0)
            if np.count_nonzero(ok) < 4:
                s = 3.0
                corners = np.array([[-s, -s], [s, -s], [s, s], [-s, s]], dtype=np.float64)
                verts = x0[None, :] + corners[:, 0:1] * u[None, :] + corners[:, 1:2] * v[None, :]
            else:
                Pw = (Cw[None, None, :] + t[..., None] * dir_world).reshape(-1, 3)[ok.reshape(-1)]
                q1 = (Pw - x0) @ u
                q2 = (Pw - x0) @ v
                q1min, q1max = np.min(q1), np.max(q1)
                q2min, q2max = np.min(q2), np.max(q2)
                pad1 = 0.10 * max(1e-6, q1max - q1min)
                pad2 = 0.10 * max(1e-6, q2max - q2min)
                q1min -= pad1
                q1max += pad1
                q2min -= pad2
                q2max += pad2
                verts = np.stack([
                    x0 + q1min * u + q2min * v,
                    x0 + q1max * u + q2min * v,
                    x0 + q1max * u + q2max * v,
                    x0 + q1min * u + q2max * v
                ], axis=0)

            tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts.astype(np.float64)),
                o3d.utility.Vector3iVector(tris)
            )
            mesh.compute_vertex_normals()

            # material
            if texture_path is not None:
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLit"
                try:
                    tex = o3d.io.read_image(str(texture_path))
                    # map quad to full texture
                    mesh.triangle_uvs = o3d.utility.Vector2dVector(
                        np.array([
                            [0, 0], [1, 0], [1, 1],
                            [0, 0], [1, 1], [0, 1]
                        ], dtype=np.float64)
                    )
                    mesh.textures = [tex]
                    mat.albedo_img = tex
                    mat.base_color = (1.0, 1.0, 1.0, 1.0)
                except Exception as e:
                    log(f'[PixelPoints::_project_open3d::make_plane_mesh] Exception: {e}', 'error')
                    mat.shader = "defaultUnlit"
                    col = (np.array(color_rgb, dtype=np.float32) / 255.0).tolist() + [1.0]
                    mat.base_color = tuple(col)
            else:
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                col = (np.array(color_rgb, dtype=np.float32) / 255.0).tolist() + [1.0]
                mat.base_color = tuple(col)

            return mesh, mat

        # floor+wall meshes
        if floor_normal is not None and floor_offset is not None:
            floor_mesh, floor_mat = make_plane_mesh(
                np.asarray(floor_normal).astype(np.float64),
                float(floor_offset),
                color_rgb=(120, 120, 120),
                texture_path=PathUtils.resources_path() / 'backdrops' / 'floor_dark.jpg'
            )
            renderer.scene.add_geometry('floor', floor_mesh, floor_mat)

        # if wall_normal is not None and wall_offset is not None:
        #     wall_mesh, wall_mat = make_plane_mesh(
        #         np.asarray(wall_normal).astype(np.float64),
        #         float(wall_offset),
        #         color_rgb=(140, 140, 140),
        #         texture_path=PathUtils.resources_path() / 'wall.jpg'
        #     )
        #     renderer.scene.add_geometry('wall', wall_mesh, wall_mat)

        # human(s)
        pcd = self.open3d
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = point_size
        renderer.scene.add_geometry('pcd', pcd, mat)

        # camera
        target_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        target_intrinsic_o3d.intrinsic_matrix = target_intrinsic.astype(np.float64)
        target_intrinsic_o3d.width = target_image_size_hw[1]
        target_intrinsic_o3d.height = target_image_size_hw[0]
        renderer.setup_camera(
            target_intrinsic_o3d,
            (target_extrinsic if not is_c2w else np.linalg.inv(target_extrinsic)).astype(np.float64)
        )
        renderer.scene.view.set_post_processing(False)

        render_rgb = np.asarray(renderer.render_to_image())[:, :, :3]
        render_depth = np.array(renderer.render_to_depth_image(z_in_view_space=True)).astype(np.float32)
        renderer.scene.clear_geometry()

        render_valid = (render_depth > 0.0) & np.isfinite(render_depth)
        render_depth[render_valid == 0] = 0.0
        return RGBDImage(
            render_rgb,
            render_valid,
            render_depth,
            intrinsic=target_intrinsic,
            extrinsic_w2c=(target_extrinsic if not is_c2w else np.linalg.inv(target_extrinsic)),
            features=self.project_features(target_intrinsic, target_extrinsic, target_image_size_hw, is_c2w=is_c2w)
        )

    def save_ply(self, out_path: Union[Path, str]) -> None:
        """
        Save the pixel points to a PLY file.

        Parameters
        ----------
        out_path : Path
            The path where to save the PLY file.
        """
        # noinspection PyTypeChecker
        if o3d.io.write_point_cloud(str(out_path), self.open3d):
            log(f'[{self.__class__.__name__}::save_ply] Point cloud saved to {out_path}', 'debug')

    @classmethod
    def from_partials(cls, *partials: 'PixelPoints') -> 'PixelPoints':
        """
        Create PixelPoints from multiple partial PixelPoints instances. The partials are stitched together.

        Parameters
        ----------
        partials : tuple of PixelPoints
            The partial PixelPoints instances to combine.

        Returns
        -------
        PixelPoints
            A new PixelPoints instance containing the combined points, colors, and valid pixels.
        """
        if len(partials) == 0:
            raise ValueError("At least one partial PixelPoints instance must be provided.")
        if len(partials) == 1:
            return partials[0]
        all_points = np.concatenate([p.points for p in partials], axis=1)
        all_colors = np.concatenate([p.colors for p in partials], axis=1)
        all_valid = np.concatenate([p.valid for p in partials], axis=1)
        all_features = {}
        for key in partials[0].features:
            all_features[key] = np.concatenate([p.features[key] for p in partials], axis=1)
        return cls(pixel_points=all_points, pixel_colors=all_colors, pixel_valid=all_valid, pixel_features=all_features)

    @classmethod
    def from_path(cls, *args, **kwargs) -> 'PixelPoints':
        """
        Create PixelPoints from file paths.

        Parameters
        ----------
        args : tuple
            Positional arguments for RGBDImage.from_path.
        kwargs : dict
            Keyword arguments for RGBDImage.from_path.

        Returns
        -------
        PixelPoints
            A new PixelPoints instance created from the RGBDImage.
        """
        rgbd_image = RGBDImage.from_path(*args, **kwargs)
        return cls.from_rgbd_image(rgbd_image)

    @classmethod
    def from_rgbd_image(cls, rgbd_image: RGBDImage) -> 'PixelPoints':
        """
        Create PixelPoints from an RGBDImage.

        Parameters
        ----------
        rgbd_image : RGBDImage
            The RGBDImage to convert.

        Returns
        -------
        PixelPoints
            A new PixelPoints instance containing the points, colors, and valid pixels from the RGBDImage.
        """
        return rgbd_image.unproject()

    @classmethod
    def from_thuman(cls, *args, **kwargs) -> 'PixelPoints':
        """
        Create PixelPoints from a model/cam combination of THuman dataset.

        Parameters
        ----------
        args : tuple
            Positional arguments for RGBDImage.from_thuman.
        kwargs : dict
            Keyword arguments for RGBDImage.from_thuman.

        Returns
        -------
        PixelPoints
            A new PixelPoints instance created from the RGBDImage.
        """
        rgbd_image = RGBDImage.from_thuman(*args, **kwargs)
        return cls.from_rgbd_image(rgbd_image)

    @classmethod
    def from_session(cls, *args, **kwargs) -> 'PixelPoints':
        """
        Create PixelPoints from a session's camera data.

        Parameters
        ----------
        args : tuple
            Positional arguments for RGBDImage.from_session.
        kwargs : dict
            Keyword arguments for RGBDImage.from_session.

        Returns
        -------
        PixelPoints
            A new PixelPoints instance created from the RGBDImage.
        """
        rgbd_image = RGBDImage.from_session(*args, **kwargs)
        return cls.from_rgbd_image(rgbd_image)


class PCDUtils:
    FEATURE_EXTRACTOR = None
    FEATURE_MATCHER = None
    RENDERER_O3D = None

    # noinspection PyTypeChecker
    @classmethod
    def align_projected_to_original(cls, projected_images: Union[torch.Tensor, np.ndarray], projected_depths: Union[torch.Tensor, np.ndarray], valid_pixels: Union[torch.Tensor, np.ndarray], original_images: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from kornia.geometry.homography import find_homography_dlt
        from kornia.geometry.transform import warp_perspective
        from lightglue import LightGlue, SuperPoint

        # SuperPoint+LightGlue
        if cls.FEATURE_EXTRACTOR is None:
            cls.FEATURE_EXTRACTOR = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
        if cls.FEATURE_MATCHER is None:
            cls.FEATURE_MATCHER = LightGlue(features='superpoint').eval().cuda()

        # np --> torch.Tensor
        if isinstance(original_images, np.ndarray):
            projected_images_dtype = original_images.dtype
            if projected_images_dtype == np.uint8:
                original_images = original_images.astype(np.float32) / 255.0
            if original_images.ndim == 3:
                original_images = original_images.transpose(2, 0, 1)[None]
            original_images = torch.from_numpy(original_images).float().cuda()  # (1, 3, H, W)
        if isinstance(projected_images, np.ndarray):
            projected_images_dtype = projected_images.dtype
            if projected_images_dtype == np.uint8:
                projected_images = projected_images.astype(np.float32) / 255.0
            if projected_images.ndim == 3:
                projected_images = projected_images.transpose(2, 0, 1)[None]
            projected_images = torch.from_numpy(projected_images).float().cuda()  # (1, 3, H, W)
        if isinstance(projected_depths, np.ndarray):
            projected_depths_dtype = projected_depths.dtype
            if projected_depths_dtype == np.uint16 or projected_depths.max() > 1000.0:
                projected_depths = projected_depths.astype(np.float32) / 1000.0  # convert to meters
            if projected_depths.ndim == 2:
                projected_depths = projected_depths[None]
            if projected_depths.ndim == 3:
                projected_depths = projected_depths[None]
            projected_depths = torch.from_numpy(projected_depths).float().cuda()  # (1, 1, H, W)
        if isinstance(valid_pixels, np.ndarray):
            if valid_pixels.ndim == 2:
                valid_pixels = valid_pixels[None]
            if valid_pixels.ndim == 3:
                valid_pixels = valid_pixels[None]
            valid_pixels = torch.from_numpy(valid_pixels).float().cuda()  # (1, 1, H, W)

        # extract local features
        projected_images_aligned, projected_depths_aligned, valid_pixels_aligned = [], [], []
        for i in range(projected_images.shape[0]):
            feats0 = cls.FEATURE_EXTRACTOR.extract(projected_images[[i]])
            feats1 = cls.FEATURE_EXTRACTOR.extract(original_images[[i]])

            # match the features
            matches01 = cls.FEATURE_MATCHER({'image0': feats0, 'image1': feats1})
            matches = matches01['matches']  # indices with shape (K,2)
            points0 = [kpts[m_[:, 0]] for kpts, m_ in zip(feats0['keypoints'], matches)]  # coordinates in image #0, shape (K,2)
            points1 = [kpts[m_[:, 1]] for kpts, m_ in zip(feats1['keypoints'], matches)]  # coordinates in image #1, shape (K,2)
            if any(_.shape[0] < 4 for _ in points0) or any(_.shape[0] < 4 for _ in points1):
                log(f'[{cls.__name__}::align_projected_to_original] Detected less than 4 points in one image pair. Skipping alignment.', 'warning')
                projected_images_aligned.append(projected_images[[i]])
                projected_depths_aligned.append(projected_depths[[i]])
                valid_pixels_aligned.append(valid_pixels[[i]])
                continue

            hs = torch.cat([find_homography_dlt(points0_b[None], points1_b[None]) for points0_b, points1_b in zip(points0, points1)], dim=0)
            projected_images_aligned.append(warp_perspective(projected_images[[i]], hs, projected_images.shape[-2:]))
            projected_depths_aligned.append(warp_perspective(projected_depths[[i]], hs, projected_images.shape[-2:]))
            valid_pixels_aligned.append(warp_perspective(valid_pixels[[i]].float(), hs, projected_images.shape[-2:]).bool())
        return torch.cat(projected_images_aligned, dim=0), torch.cat(projected_depths_aligned, dim=0), torch.cat(valid_pixels_aligned, dim=0)

    @classmethod
    def unproject_depth_maps_torch(cls, depth: torch.Tensor, intrinsics: torch.Tensor, color: Optional[torch.Tensor] = None, dists: Optional[torch.Tensor] = None, rotmats: Optional[torch.Tensor] = None, tvecs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backproject a batch of depth maps into 3D points using their intrinsics.

        Parameters
        ----------
        depth : torch.Tensor
            Depth maps with shape (B, 1, H, W), depth in millimeters.
        color : torch.Tensor
            Color images with shape (B, 3, H, W).
        intrinsics : torch.Tensor
            Camera intrinsic matrices with shape (B, 3, 3).
        dists : torch.Tensor, optional
            Camera dist coefficients with shape (B, 5).
        rotmats: torch.Tensor, optional
            Camera rotation matrices with shape (B, 3, 3). If provided, the points will be transformed to world coordinates.
        tvecs: torch.Tensor, optional
            Camera rotation matrices with shape (B, 3). If provided, the points will be transformed to world coordinates.

        Returns
        -------
        points_3d : torch.Tensor
            The 3D points in camera coordinates with shape (B, H*W, 3).
        colors_3d : torch.Tensor
            The corresponding colors for each 3D point with shape (B, H*W, 3).
        """
        if depth.ndim == 2:
            depth = depth[None]
        if depth.ndim == 3:
            depth = depth[None]
        if depth.max() > 1000.0:
            depth = depth / 1000.0  # Convert to meters
        if intrinsics.ndim == 2:
            intrinsics = intrinsics[None]
        B, _, H, W = depth.shape
        device = depth.device

        # create shared canvas
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        xs = xs.reshape(-1).float()
        ys = ys.reshape(-1).float()

        points_3d_batch = []
        colors_3d_batch = []

        for b in range(B):
            z = depth[b, 0].reshape(-1)
            pixels = torch.stack([xs, ys], dim=-1).cpu().numpy()

            if dists is not None:
                pixels_undistorted = cv2.undistortPoints(
                    pixels.reshape(-1, 1, 2),
                    intrinsics[b].cpu().numpy(),
                    dists[b].cpu().numpy(),
                    P=intrinsics[b].cpu().numpy()
                ).reshape(-1, 2)
                pixels_undistorted = torch.from_numpy(pixels_undistorted).to(device)
            else:
                pixels_undistorted = torch.from_numpy(pixels).to(device)

            fx, fy = intrinsics[b, 0, 0], intrinsics[b, 1, 1]
            cx, cy = intrinsics[b, 0, 2], intrinsics[b, 1, 2]

            x = (pixels_undistorted[:, 0] - cx) * z / fx
            y = (pixels_undistorted[:, 1] - cy) * z / fy
            points_camera = torch.stack([x, y, z], dim=-1)

            # if rotmats is not None and tvecs is not None:
            #     points_camera_hom = torch.cat([points_camera, torch.ones_like(x).unsqueeze(-1)], dim=-1)
            #     extrinsics = torch.eye(4, device=device)
            #     extrinsics[:3, :3] = rotmats[b]
            #     extrinsics[:3, 3] = tvecs[b]
            #
            #     points_world_hom = (extrinsics @ points_camera_hom.T).T
            #     points_world = points_world_hom[:, :3] / points_world_hom[:, 3:]
            #     points_3d_batch.append(points_world)
            # else:
            points_3d_batch.append(points_camera)

            if color is not None:
                c = color[b].permute(1, 2, 0).reshape(-1, 3)
                if c.dtype == torch.uint8:
                    c = c.float() / 255.0
                colors_3d_batch.append(c)
            else:
                colors_3d_batch.append(torch.zeros((H * W, 3), device=device))

        points_3d = torch.stack(points_3d_batch, dim=0)
        colors_3d = torch.stack(colors_3d_batch, dim=0)

        if rotmats is not None:
            assert tvecs is not None
            points_3d = cls.camera_to_world_torch(points_cam=points_3d, rotmats=rotmats, tvecs=tvecs)

        return points_3d, colors_3d

    @classmethod
    def camera_to_world_torch(cls, points_cam: torch.Tensor, extrinsics: Optional[torch.Tensor] = None, rotmats: Optional[torch.Tensor] = None, tvecs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transform points from camera space to world space using extrinsic matrices.

        Parameters
        ----------
        points_cam : torch.Tensor
            The points in camera space, shape (B, N, 3).
        extrinsics : torch.Tensor, optional
            The extrinsic matrices for each batch, shape (B, 4, 4).
            These are assumed to transform from camera to world coordinates.
            If None, rotmats and tvecs are required and are used instead.
        rotmats : torch.Tensor, optional
            The rotation matrices for each batch, shape (B, 3, 3).
        tvecs : torch.Tensor, optional
            The translation vectors for each batch, shape (B, 3).

        Returns
        -------
        points_world : torch.Tensor
            The points transformed into world space, shape (B, N, 3).
        """
        B, N, _ = points_cam.shape

        # Perform batch matrix multiplication:
        # extrinsics: (B,4,4), points_h: (B,N,4) -> (B,N,4)
        # We'll do a batch multiplication by first reshaping points_h appropriately
        if extrinsics is not None:
            # Convert points to homogeneous coordinates: (x, y, z) -> (x, y, z, 1)
            ones = torch.ones((B, N, 1), device=points_cam.device, dtype=points_cam.dtype)
            points_h = torch.cat([points_cam, ones], dim=2)  # (B, N, 4)
            points_h = points_h.transpose(1, 2)  # (B,4,N)
            points_world_h = extrinsics @ points_h  # (B,4,N)
            points_world_h = points_world_h.transpose(1, 2)  # (B,N,4)
            # Convert back from homogeneous coordinates
            points_world = points_world_h[..., :3] / points_world_h[..., 3:].clamp(min=1e-10)
        else:
            # First, transpose points_cam to (B,3,N) so we can perform a batch matrix multiplication
            points_world = rotmats @ points_cam.transpose(1, 2) + tvecs.unsqueeze(-1)  # (B,3,N)
            # Transpose back to (B,N,3)
            points_world = points_world.transpose(1, 2)

        # points_world[points_cam[..., -1] > 3.0] = float('nan')
        return points_world


class SceneAssets:
    def __init__(self, pcds: List[PixelPoints], wall_image_path: Optional[Union[str, Path]] = None, floor_size: float = 20.0, floor_tiles: int = 16, floor_res: int = 400, wall_width: float = 14.0, wall_height: float = 8.0, wall_distance: float = 10.0, wall_repeats_u: float = 6.0, wall_repeats_v: float = 3.0, wall_res_u: int = 800, wall_res_v: int = 400):
        self.pcds = pcds
        self.wall_image_path = str(wall_image_path) if wall_image_path is not None else None
        self.floor_size = float(floor_size)
        self.floor_tiles = int(floor_tiles)
        self.floor_res = int(floor_res)
        self.wall_width = float(wall_width)
        self.wall_height = float(wall_height)
        self.wall_distance = float(wall_distance)
        self.wall_repeats_u = float(wall_repeats_u)
        self.wall_repeats_v = float(wall_repeats_v)
        self.wall_res_u = int(wall_res_u)
        self.wall_res_v = int(wall_res_v)
        self._floor_pcd = None
        self._wall_pcd = None
        self._axes = None
        self._origin = None

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n if n > 1e-8 else 1.0)

    def _fit_smpl(self, pcd: PixelPoints):
        pts = np.asarray(pcd.open3d.points, dtype=np.float64)
        c = pts.mean(axis=0)
        x = pts - c
        cov = (x.T @ x) / max(1, x.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        up = self._normalize(V[:, np.argmax(w)])
        alt = self._normalize(V[:, np.argsort(w)[-2]])
        if abs(np.dot(up, alt)) > 0.95:
            alt = self._normalize(V[:, np.argsort(w)[-3]])
        forward = self._normalize(alt - np.dot(alt, up) * up)
        right = self._normalize(np.cross(forward, up))
        forward = self._normalize(np.cross(up, right))
        proj_up = (pts - c) @ up
        min_h = proj_up.min()
        origin = c + min_h * up
        bbox = pcd.open3d.get_axis_aligned_bounding_box()
        return {"center": c, "right": right, "up": up, "forward": forward, "origin": origin, "bbox": bbox}

    def _process_single_pcd(self, pcd: PixelPoints):
        return self._fit_smpl(pcd)

    def _process_pcds(self):
        return [self._process_single_pcd(p) for p in self.pcds]

    def _compute_scene_cube(self, pcd_data):
        ups = np.stack([d["up"] for d in pcd_data], 0)
        rights = np.stack([d["right"] for d in pcd_data], 0)
        forwards = np.stack([d["forward"] for d in pcd_data], 0)
        origins = np.stack([d["origin"] for d in pcd_data], 0)
        up = self._normalize(ups.mean(0))
        right = self._normalize(rights.mean(0))
        forward = self._normalize(np.cross(up, right))
        right = self._normalize(np.cross(forward, up))
        origin = origins.mean(0)
        self._axes = (right, up, forward)
        self._origin = origin
        return {"right": right, "up": up, "forward": forward, "origin": origin}

    def _create_floor_pcd(self):
        if self._axes is None or self._origin is None:
            raise RuntimeError("Axes not computed")
        right, up, forward = self._axes
        u = np.linspace(-0.5, 0.5, self.floor_res, dtype=np.float32)
        v = np.linspace(-0.5, 0.5, self.floor_res, dtype=np.float32)
        U, V = np.meshgrid(u, v, indexing="xy")
        P = self._origin + (U[..., None] * self.floor_size) * right + (V[..., None] * self.floor_size) * forward
        Uc = (U + 0.5) * self.floor_tiles
        Vc = (V + 0.5) * self.floor_tiles
        mask = ((np.floor(Uc) + np.floor(Vc)) % 2)[..., None]
        c0 = np.array([0.85, 0.85, 0.85], dtype=np.float32)
        c1 = np.array([0.20, 0.20, 0.20], dtype=np.float32)
        C = mask * c0 + (1.0 - mask) * c1
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P.reshape(-1, 3).astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(C.reshape(-1, 3).astype(np.float64))
        self._floor_pcd = pc
        return pc

    def _create_wall_pcd(self):
        if self._axes is None or self._origin is None:
            raise RuntimeError("Axes not computed")
        right, up, forward = self._axes
        if self.wall_image_path is not None:
            img = o3d.io.read_image(str(self.wall_image_path))
            tex = np.asarray(img)
            if tex.ndim == 2:
                tex = np.repeat(tex[..., None], 3, axis=2)
            if tex.shape[2] == 4:
                tex = tex[:, :, :3]
            H, W = tex.shape[:2]
        else:
            H, W = 2, 2
            tex = np.array([[[200, 200, 220], [140, 140, 160]], [[140, 140, 160], [200, 200, 220]]], dtype=np.uint8)
        xs = np.linspace(-0.5, 0.5, self.wall_res_u, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, self.wall_res_v, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        base = self._origin - forward * self.wall_distance
        P = base + (X[..., None] * self.wall_width) * right + (Y[..., None] * self.wall_height) * up
        U = (X + 0.5) * self.wall_repeats_u
        V = Y * self.wall_repeats_v
        xi = np.clip(np.rint((U % 1.0) * (W - 1)).astype(np.int32), 0, W - 1)
        yi = np.clip(np.rint((V % 1.0) * (H - 1)).astype(np.int32), 0, H - 1)
        C = (tex[yi, xi].reshape(-1, 3).astype(np.float32) / 255.0)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P.reshape(-1, 3).astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(C.astype(np.float64))
        self._wall_pcd = pc
        return pc

    def _align_floor_wall_with_scene_cube(self, scene_data):
        return

    def floor_and_wall(self) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, Dict[str, np.ndarray]]:
        pcd_data = self._process_pcds()
        scene_data = self._compute_scene_cube(pcd_data)
        floor = self._create_floor_pcd()
        wall = self._create_wall_pcd()
        return floor, wall, scene_data


if __name__ == '__main__':
    from utils.calib import CalibrationData

    # # read THUMAN data
    # thuman_root_ = Path('/media/charisoudis/nas_transmixr/Simone/Volumetric_Video/Human Datasets/THuman2_1/rendered@2m')
    # # create RGBDImages
    # rgbd_l_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=0)
    # rgbd_l_.save_png('thuman_0000_000_l.png')
    # rgbd_r_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=1)
    # rgbd_r_.save_png('thuman_0000_000_r.png')
    # # reproject RGBDImages to each other
    # rgbd_l_.reproject_to(rgbd_l_, align=False, use_cache=True).save_png('thuman_0000_000_l2l.png')
    # rgbd_l_.reproject_to(rgbd_r_, align=False, use_cache=True).save_png('thuman_0000_000_l2r.png')
    # rgbd_r_.reproject_to(rgbd_r_, align=False, use_cache=True).save_png('thuman_0000_000_r2r.png')
    # rgbd_r_.reproject_to(rgbd_l_, align=False, use_cache=True).save_png('thuman_0000_000_r2l.png')
    # # create point clouds from RGBDImages
    # pcd_l_ = PixelPoints.from_rgbd_image(rgbd_l_)
    # pcd_r_ = PixelPoints.from_rgbd_image(rgbd_r_)
    # pcd_l_.save_ply('thuman_0000_000_l.ply')
    # pcd_r_.save_ply('thuman_0000_000_r.ply')
    # # stitch left and right point clouds
    # pcd_lr_ = PixelPoints.from_partials(pcd_l_, pcd_r_)
    # pcd_lr_.save_ply('thuman_0000_000_lr.ply')
    # pcd_lr_.project(rgbd_l_.intrinsic, rgbd_l_.extrinsic_w2c, rgbd_l_.image_size_hw, use_cache=True, is_c2w=False).save_png('thuman_0000_000_lr2l.png')
    # pcd_lr_.project(rgbd_r_.intrinsic, rgbd_r_.extrinsic_w2c, rgbd_r_.image_size_hw, use_cache=True, is_c2w=False).save_png('thuman_0000_000_lr2r.png')

    # Cagliari test
    session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Cagliari_Nov_2025' / 'Cagliari_1_Perf_7'
    calibration_session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Cagliari_Nov_2025' / 'Cagliari_1_Calib_6'
    calibration_data_ = CalibrationData.from_session(calibration_session_root_)
    rgbd_1_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=1, color_ts=11512358, depth_ts=11512360, depth_filter='bilateral_temporal').resize(1280, 1024)
    rgbd_3_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=7, color_ts=11512338, depth_ts=11512341, depth_filter='bilateral_temporal').resize(1280, 1024)
    pcd_1_ = PixelPoints.from_rgbd_image(rgbd_1_)
    pcd_3_ = PixelPoints.from_rgbd_image(rgbd_3_)
    pcd_all_ = PixelPoints.from_partials(pcd_1_, pcd_3_)
    pcd_all_.save_ply('cagliari_all.ply')
    exit(0)

    # rotate test (Brasov)
    session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025' / 'Thanos_2_Perf_1'
    calibration_session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025' / 'Thanos_2_Calib_1'
    calibration_data_ = CalibrationData.from_session(calibration_session_root_)
    rgbd_1_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=8, color_ts=1746110298900, depth_ts=1746110298901, depth_filter='aligned').rotate('90_CLOCKWISE')  # .resize(1280, 1024)
    rgbd_2_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=5, color_ts=1746110298897, depth_ts=1746110298898, depth_filter='aligned').rotate('90_CLOCKWISE')  # .resize(1280, 1024)
    rgbd_3_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=7, color_ts=1746110298900, depth_ts=1746110298901, depth_filter='aligned').rotate('90_CLOCKWISE')  # .resize(1280, 1024)
    pcd_1_ = PixelPoints.from_rgbd_image(rgbd_1_)
    pcd_2_ = PixelPoints.from_rgbd_image(rgbd_2_)
    pcd_3_ = PixelPoints.from_rgbd_image(rgbd_3_)
    pcd_all_ = PixelPoints.from_partials(pcd_1_, pcd_2_, pcd_3_)
    pcd_all_.save_ply('thanos_all.ply')
    exit(0)

    # rotate test (Brasov)
    session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Brasov_Sep_2025' / 'Brasov_1_Perf_2'
    calibration_session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Brasov_Sep_2025' / 'Brasov_1_Calib_1'
    calibration_data_ = CalibrationData.from_session(calibration_session_root_)
    rgbd_1_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=1, color_ts=1759002758103, depth_ts=1759002758104, depth_filter='aligned')
    rgbd_2_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=2, color_ts=1759002758103, depth_ts=1759002758104, depth_filter='aligned')
    rgbd_3_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=3, color_ts=1759002758103, depth_ts=1759002758105, depth_filter='aligned')
    pcd_1_ = PixelPoints.from_rgbd_image(rgbd_1_)
    pcd_2_ = PixelPoints.from_rgbd_image(rgbd_2_)
    pcd_3_ = PixelPoints.from_rgbd_image(rgbd_3_)
    pcd_all_ = PixelPoints.from_partials(pcd_1_, pcd_2_, pcd_3_)
    pcd_all_.save_ply('brasov_all.ply')
    exit(0)

    # read session data
    # session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025' / 'Thanos_2_Perf_1'
    # calibration_session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025' / 'Thanos_2_Calib_1'
    session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Brasov_Sep_2025' / 'Brasov_1_Perf_2'
    calibration_session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Brasov_Sep_2025' / 'Brasov_1_Calib_1'
    calibration_data_ = CalibrationData.from_session(calibration_session_root_)
    # create RGBDImages
    # rgbd_l_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=9, color_ts=1746110341439, depth_ts=1746110341440, depth_filter='bilateral_temporal').resize(1024, 1024)
    rotate_ = None
    # rotate_ = '90_COUNTERCLOCKWISE'
    rgbd_l_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=1, color_ts=1759002758103, depth_ts=1759002758104, depth_filter='bilateral_temporal').rotate(rotate_).resize(1024, 1024)
    rgbd_l_.save_png('brasov_cam1.png', striped=False)
    # exit(0)
    # rgbd_r_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=8, color_ts=1746110341432, depth_ts=1746110341433, depth_filter='bilateral_temporal').resize(1024, 1024)
    rgbd_r_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=3, color_ts=1759002758103, depth_ts=1759002758105, depth_filter='bilateral_temporal').rotate(rotate_).resize(1024, 1024)
    rgbd_r_.save_png('brasov_cam3.png')

    floor_wall_data = RGBDImage.estimate_floor(
        RGBDImage.from_session(session_root_, calibration_data_, cam_idx=1, color_ts=1759002758103, depth_ts=1759002758104, depth_filter='aligned').rotate(rotate_).resize(1024, 1024),
        RGBDImage.from_session(session_root_, calibration_data_, cam_idx=3, color_ts=1759002758103, depth_ts=1759002758105, depth_filter='aligned').rotate(rotate_).resize(1024, 1024),
        export=True,
        export_path='brasov_floor_test.png',
        # rotate='90_COUNTERCLOCKWISE'
    )
    # print(floor_wall_data)
    # exit(0)

    # reproject RGBDImages to each other
    rgbd_l_.reproject_to(rgbd_l_, align=False, use_cache=True, **floor_wall_data).save_png('session_929.png')
    # rgbd_l_.reproject_to(rgbd_r_, align=False, use_cache=True).save_png('session_928.png')
    rgbd_r_.reproject_to(rgbd_r_, align=False, use_cache=True, **floor_wall_data).save_png('session_828.png')
    # rgbd_r_.reproject_to(rgbd_l_, align=False, use_cache=True).save_png('session_829.png')
    # create point clouds from RGBDImages
    pcd_l_ = PixelPoints.from_rgbd_image(rgbd_l_)
    pcd_r_ = PixelPoints.from_rgbd_image(rgbd_r_)
    # pcd_l_.save_ply('session_9.ply')
    # pcd_r_.save_ply('session_8.ply')
    # stitch left and right point clouds
    pcd_lr_ = PixelPoints.from_partials(pcd_l_, pcd_r_)
    pcd_lr_.save_ply('session_98.ply')
    pcd_lr_.project(rgbd_l_.intrinsic, rgbd_l_.extrinsic_w2c, rgbd_l_.image_size_hw, use_cache=True, is_c2w=False, **floor_wall_data).save_png('session_9829.png')
    pcd_lr_.project(rgbd_r_.intrinsic, rgbd_r_.extrinsic_w2c, rgbd_r_.image_size_hw, use_cache=True, is_c2w=False, **floor_wall_data).save_png('session_9828.png')
