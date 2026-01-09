import copy
import functools
import os
from pathlib import Path
from typing import Optional, Tuple, Literal, Union, Dict, List, Any

import cv2
import numpy as np
import torch

from utils.misc import log, env_get, PathUtils, Str
from utils.vis import VisUtils

os.environ['PYOPENGL_PLATFORM'] = env_get('PYOPENGL_PLATFORM', 'egl')
import open3d as o3d
from utils.calib import CalibrationData


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
            print(dmin, dmax)
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
    def estimate_floor(*rgbd_images: 'RGBDImage', export: bool = False, export_path: Optional[Union[Path, str]] = None) -> Dict[str, np.ndarray]:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME, dtype=dtype).to(device).eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        floor_ids = [k for k, v in id2label.items() if any(t in v.lower() for t in ("floor", "ground", "road", "sidewalk", "pavement"))] or [0]
        rng = np.random.default_rng(0)

        def seg_floor_prob(rgb: np.ndarray) -> np.ndarray:
            H, W = rgb.shape[:2]
            tile, overlap = 1280, 256
            if max(H, W) <= tile:
                inp = processor(images=rgb, return_tensors="pt")
                inp = {k: v.to(device, dtype=model.dtype) for k, v in inp.items()}
                with torch.inference_mode():
                    logits = model(**inp).logits
                    logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)[0]
                    p = logits.softmax(dim=0)[floor_ids].sum(dim=0).float().cpu().numpy()
                return np.clip(p, 0, 1).astype(np.float32)
            step = max(1, tile - overlap)
            acc = np.zeros((H, W), np.float32)
            cnt = np.zeros((H, W), np.float32)
            for y in range(0, H, step):
                for x in range(0, W, step):
                    y1, x1 = min(H, y + tile), min(W, x + tile)
                    patch = rgb[y:y1, x:x1]
                    inp = processor(images=patch, return_tensors="pt")
                    inp = {k: v.to(device, dtype=model.dtype) for k, v in inp.items()}
                    with torch.inference_mode():
                        logits = model(**inp).logits
                        logits = torch.nn.functional.interpolate(logits, size=patch.shape[:2], mode="bilinear", align_corners=False)[0]
                        p = logits.softmax(dim=0)[floor_ids].sum(dim=0).float().cpu().numpy()
                    acc[y:y1, x:x1] += p
                    cnt[y:y1, x:x1] += 1.0
            return np.clip(acc / np.maximum(cnt, 1e-6), 0, 1).astype(np.float32)

        overlays: List[np.ndarray] = [] if export else None
        per_view_planes: List[Dict[str, Any]] = []
        per_view_probs: List[np.ndarray] = []
        all_pts, all_w = [], []

        for img in rgbd_images:
            if img is None or img.rgb is None or img.depth is None:
                if export: overlays.append(np.zeros((1, 1, 3), np.uint8))
                per_view_planes.append(None)
                per_view_probs.append(None)
                continue

            rgb = img.rgb if img.rgb.dtype == np.uint8 else np.clip(img.rgb, 0, 255).astype(np.uint8)
            prob = seg_floor_prob(rgb)
            seg = (prob >= 0.5)
            per_view_probs.append(prob)
            valid_depth = np.isfinite(img.depth) & (img.depth > 0) & (img.depth < 4)
            Pw = img.points_world
            finite_pw = np.isfinite(Pw).all(-1)
            m = seg & valid_depth & finite_pw

            if export:
                import numpy as _np  # avoid shadow
                base = rgb.astype(_np.float32)
                orange = _np.array([255, 165, 0], _np.float32)
                for c in range(3):
                    ch = base[..., c]
                    ch[seg] = orange[c] * 0.30 + ch[seg] * 0.70
                    base[..., c] = ch
                overlays.append(base.clip(0, 255).astype(np.uint8))

            if np.count_nonzero(m) >= 50:
                P = Pw[m].astype(np.float64)
                w = (prob[m].astype(np.float64) + 1e-3)
                w /= (w.sum() + 1e-12)
                mu = (w[:, None] * P).sum(0)
                X0 = P - mu
                Xw = (np.sqrt(w)[:, None] * X0)
                U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
                n = Vt[-1]
                d = -n @ mu
                per_view_planes.append({"normal_world": n.astype(np.float64), "offset_world": float(d)})
                all_pts.append(P.astype(np.float32))
                all_w.append((prob[m].astype(np.float32) + 1e-3))
            else:
                per_view_planes.append(None)

        if not all_pts:
            return {"normal": np.array([0, 0, 1], np.float64), "offset": 0.0, "per_view_planes": per_view_planes, "out_path_ply": None, "wall_normal": None, "wall_offset": None}

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
            U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
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
            U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
            n_final = Vt[-1]
            d_final = -n_final @ mu
            inliers = best_inl

        perp_dot_max = 0.2
        stride_w = 4
        wall_local_pts_world: List[np.ndarray] = []
        wall_local_planes: List[Dict[str, Any]] = []

        for vi, img in enumerate(rgbd_images):
            if img is None or img.depth is None or per_view_planes[vi] is None:
                wall_local_planes.append(None)
                continue

            n_floor_v = per_view_planes[vi]["normal_world"].astype(np.float64)
            Pw = img.points_world
            valid = np.isfinite(img.depth) & (img.depth > 0) & np.isfinite(Pw).all(-1)
            if not np.any(valid):
                wall_local_planes.append(None)
                continue

            sel = valid[::stride_w, ::stride_w]
            Xv = Pw[::stride_w, ::stride_w][sel].reshape(-1, 3).astype(np.float64)
            if Xv.shape[0] < 300:
                wall_local_planes.append(None)
                continue

            scale_v = float(np.median(np.linalg.norm(Xv, axis=1)))
            tau_w = max(0.006, 0.01 * scale_v)

            best_cnt_v, best_n_v, best_d_v, best_inl_v = -1, None, None, None
            it_v = 2000
            for _ in range(it_v):
                idx3 = rng.choice(Xv.shape[0], 3, replace=False)
                a, b, c = Xv[idx3]
                n = np.cross(b - a, c - a)
                nn = np.linalg.norm(n)
                if nn < 1e-9: continue
                n /= nn
                if abs(float(n @ n_floor_v)) > perp_dot_max:
                    continue
                d = -float(n @ a)
                dist = np.abs(Xv @ n + d)
                inl = dist <= tau_w
                cnt = int(inl.sum())
                if cnt > best_cnt_v:
                    best_cnt_v, best_n_v, best_d_v, best_inl_v = cnt, n.copy(), d, inl

            if best_n_v is None or best_cnt_v < 200:
                wall_local_planes.append(None)
                continue

            Xrin = Xv[best_inl_v]
            mu = Xrin.mean(0)
            X0 = Xrin - mu
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            n_ref = Vt[-1]
            if abs(float(n_ref @ n_floor_v)) > perp_dot_max:
                n_ref = n_ref - n_floor_v * (n_ref @ n_floor_v)
                n_ref = n_ref / (np.linalg.norm(n_ref) + 1e-12)
            d_ref = -float(n_ref @ mu)

            wall_local_planes.append({"normal_world": n_ref.astype(np.float64), "offset_world": float(d_ref)})
            wall_local_pts_world.append(Xrin.astype(np.float32))

        if wall_local_pts_world:
            Wglob = np.concatenate(wall_local_pts_world, axis=0)
            keep = np.isfinite(Wglob).all(-1)
            Wglob = Wglob[keep]
        else:
            Wglob = np.zeros((0, 3), np.float64)

        if Wglob.shape[0] >= 200:
            if Wglob.shape[0] > 1_000_000:
                idx = rng.choice(Wglob.shape[0], 1_000_000, replace=False)
                Wglob = Wglob[idx]

            scale_w = float(np.median(np.linalg.norm(Wglob, axis=1)))
            tau_wR = max(0.006, 0.01 * scale_w)
            best_cnt_g, best_nw_g, best_dw_g, best_inl_g = -1, None, None, None
            it_g = 2500
            for _ in range(it_g):
                idx3 = rng.choice(Wglob.shape[0], 3, replace=False)
                a, b, c = Wglob[idx3]
                n = np.cross(b - a, c - a)
                nn = np.linalg.norm(n)
                if nn < 1e-9: continue
                n /= nn
                if abs(float(n @ n_final)) > perp_dot_max:
                    continue
                d = -float(n @ a)
                dist = np.abs(Wglob @ n + d)
                inl = dist <= tau_wR
                cnt = int(inl.sum())
                if cnt > best_cnt_g:
                    best_cnt_g, best_nw_g, best_dw_g, best_inl_g = cnt, n.copy(), d, inl

            if best_nw_g is None:
                muw = Wglob.mean(0)
                Y = Wglob - muw
                U, S, Vt = np.linalg.svd(Y, full_matrices=False)
                n_wall = Vt[-1]
                n_wall = n_wall - n_final * (n_wall @ n_final)
                n_wall = n_wall / (np.linalg.norm(n_wall) + 1e-12)
                d_wall = -float(n_wall @ muw)
                inl_w = np.abs(Wglob @ n_wall + d_wall) <= tau_wR
            else:
                Win = Wglob[best_inl_g]
                muw = Win.mean(0)
                Y = Win - muw
                U, S, Vt = np.linalg.svd(Y, full_matrices=False)
                n_wall = Vt[-1]
                if abs(float(n_wall @ n_final)) > perp_dot_max:
                    n_wall = n_wall - n_final * (n_wall @ n_final)
                    n_wall = n_wall / (np.linalg.norm(n_wall) + 1e-12)
                d_wall = -float(n_wall @ muw)
                inl_w = np.abs(Wglob @ n_wall + d_wall) <= tau_wR
        else:
            n_wall, d_wall, inl_w = None, None, None

        out_path_ply = None
        if export:
            import math, cv2, open3d as o3d

            plane_color = np.array([255, 0, 255], np.float32)
            wall_color = np.array([64, 128, 255], np.float32)
            tiles = []
            for k, img in enumerate(rgbd_images):
                if img is None or img.rgb is None: continue
                H, W = img.rgb.shape[:2]
                base = overlays[k].astype(np.float32)

                fx, fy = float(img.intrinsic[0, 0]), float(img.intrinsic[1, 1])
                cx, cy = float(img.intrinsic[0, 2]), float(img.intrinsic[1, 2])
                uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
                r_cam = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
                c2w = np.linalg.inv(img.extrinsic_w2c)
                R = c2w[:3, :3]
                Cw = c2w[:3, 3]
                d_world = r_cam @ R.T

                denom = d_world[..., 0] * n_final[0] + d_world[..., 1] * n_final[1] + d_world[..., 2] * n_final[2]
                numer = (n_final @ Cw) + d_final
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = -numer / denom
                dep = img.depth.astype(np.float32)
                dep_valid = dep[(dep > 0) & np.isfinite(dep)]
                if dep_valid.size:
                    near = max(0.05, float(np.quantile(dep_valid, 0.02)) * 0.8)
                    far = float(np.quantile(dep_valid, 0.98)) * 1.25
                else:
                    near, far = 0.05, 6.0
                vis_floor = np.isfinite(t) & (np.abs(denom) > 1e-8) & (t > near) & (t < far)
                for c in range(3):
                    ch = base[..., c]
                    ch[vis_floor] = plane_color[c] * 0.45 + ch[vis_floor] * 0.55
                    base[..., c] = ch

                if n_wall is not None:
                    denom_w = d_world[..., 0] * n_wall[0] + d_world[..., 1] * n_wall[1] + d_world[..., 2] * n_wall[2]
                    numer_w = (n_wall @ Cw) + d_wall
                    with np.errstate(divide='ignore', invalid='ignore'):
                        tw = -numer_w / denom_w
                    vis_wall = np.isfinite(tw) & (np.abs(denom_w) > 1e-8) & (tw > near) & (tw < far)
                    for c in range(3):
                        ch = base[..., c]
                        ch[vis_wall] = wall_color[c] * 0.45 + ch[vis_wall] * 0.55
                        base[..., c] = ch

                tiles.append(base.clip(0, 255).astype(np.uint8))

            if tiles:
                cols = int(math.ceil(math.sqrt(len(tiles))))
                rows = int(math.ceil(len(tiles) / cols))
                th = max(im.shape[0] for im in tiles)
                tw = max(im.shape[1] for im in tiles)
                pad = 8
                grid = np.full((rows * th + pad * (rows - 1), cols * tw + pad * (cols - 1), 3), 0, np.uint8)
                for i, im in enumerate(tiles):
                    if im.shape[:2] != (th, tw):
                        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_LINEAR)
                    r, c = divmod(i, cols)
                    y0, x0 = r * (th + pad), c * (tw + pad)
                    grid[y0:y0 + th, x0:x0 + tw] = im
                png_path = Path(export_path if export_path is not None else "floor_seg_and_plane_grid.png")
                cv2.imwrite(str(png_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                log(f'[RGBDImage::estimate_floor] PNG saved to to {png_path.parent.name}/{png_path.name}]')

            ref_idx = next((i for i, im in enumerate(rgbd_images) if im is not None and im.depth is not None), None)
            if ref_idx is not None:
                ref_img = rgbd_images[ref_idx]
                ref_pcd = ref_img.unproject().open3d
                pts_ref = np.asarray(ref_pcd.points, dtype=np.float64)
                cols_ref = np.asarray(ref_pcd.colors, dtype=np.float64)
                if pts_ref.size == 0:
                    pts_ref = np.zeros((0, 3), np.float64)
                    cols_ref = np.zeros((0, 3), np.float64)

                Xin = X[inliers]
                if Xin.shape[0] >= 4:
                    mu = Xin.mean(0)
                    X0 = Xin - mu
                    C2f = (X0[:, :, None] @ X0[:, None, :]).mean(0)
                    _, V2f = np.linalg.eigh(C2f)
                    a1f = V2f[:, 2]
                    a1f = a1f - n_final * (a1f @ n_final)
                    a1f /= (np.linalg.norm(a1f) + 1e-12)
                    a2f = np.cross(n_final, a1f)
                    a2f /= (np.linalg.norm(a2f) + 1e-12)
                    Qf = np.stack([(Xin - mu) @ a1f, (Xin - mu) @ a2f], 1)
                    qmin, qmax = Qf.min(0), Qf.max(0)
                    u = np.linspace(qmin[0], qmax[0], 220)
                    v = np.linspace(qmin[1], qmax[1], 220)
                    Uv, Vv = np.meshgrid(u, v)
                    plane_pts = (mu[None, None, :] + Uv[..., None] * a1f[None, None, :] + Vv[..., None] * a2f[None, None, :]).reshape(-1, 3)
                else:
                    x0 = -d_final * n_final
                    a = np.array([1.0, 0.0, 0.0]) if abs(n_final[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                    e1 = np.cross(n_final, a)
                    e1 /= (np.linalg.norm(e1) + 1e-12)
                    e2 = np.cross(n_final, e1)
                    s = max(scale, 1.0)
                    u = np.linspace(-s, s, 220)
                    v = np.linspace(-s, s, 220)
                    Uv, Vv = np.meshgrid(u, v)
                    plane_pts = (x0[None, None, :] + Uv[..., None] * e1[None, None, :] + Vv[..., None] * e2[None, None, :]).reshape(-1, 3)
                plane_cols = np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float64), (plane_pts.shape[0], 1))

                wall_pts = np.zeros((0, 3), np.float64)
                wall_cols = np.zeros((0, 3), np.float64)
                if n_wall is not None:
                    Win = Wglob[inl_w] if inl_w is not None and np.any(inl_w) else Wglob
                    if Win.shape[0] >= 4:
                        muw = Win.mean(0)
                        Y = Win - muw
                        Cw = (Y[:, :, None] @ Y[:, None, :]).mean(0)
                        _, Vw = np.linalg.eigh(Cw)
                        a1w = Vw[:, 2]
                        a1w = a1w - n_wall * (a1w @ n_wall)
                        a1w /= (np.linalg.norm(a1w) + 1e-12)
                        a2w = np.cross(n_wall, a1w)
                        a2w /= (np.linalg.norm(a2w) + 1e-12)
                        Qw = np.stack([(Win - muw) @ a1w, (Win - muw) @ a2w], 1)
                        qmin, qmax = Qw.min(0), Qw.max(0)
                        u = np.linspace(qmin[0], qmax[0], 180)
                        v = np.linspace(qmin[1], qmax[1], 180)
                        U2, V2 = np.meshgrid(u, v)
                        wall_pts = (muw[None, None, :] + U2[..., None] * a1w[None, None, :] + V2[..., None] * a2w[None, None, :]).reshape(-1, 3)
                    else:
                        x0w = -d_wall * n_wall
                        a0 = np.array([1.0, 0.0, 0.0])
                        a1w = a0 - n_wall * (a0 @ n_wall)
                        if np.linalg.norm(a1w) < 1e-6: a1w = np.array([0.0, 1.0, 0.0])
                        a1w /= (np.linalg.norm(a1w) + 1e-12)
                        a2w = np.cross(n_wall, a1w)
                        a2w /= (np.linalg.norm(a2w) + 1e-12)
                        s = max(scale, 1.0)
                        u = np.linspace(-s, s, 160)
                        v = np.linspace(-s, s, 160)
                        U2, V2 = np.meshgrid(u, v)
                        wall_pts = (x0w[None, None, :] + U2[..., None] * a1w[None, None, :] + V2[..., None] * a2w[None, None, :]).reshape(-1, 3)
                    wall_cols = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (wall_pts.shape[0], 1))

                pts_all = np.vstack([pts_ref, plane_pts, wall_pts])
                cols_all = np.vstack([cols_ref, plane_cols, wall_cols])
                pcd_all = o3d.geometry.PointCloud()
                pcd_all.points = o3d.utility.Vector3dVector(pts_all)
                pcd_all.colors = o3d.utility.Vector3dVector(cols_all)
                ply_path = str(Path(export_path if export_path is not None else 'floor_wall_cloud.png').with_suffix('.ply'))
                if o3d.io.write_point_cloud(ply_path, pcd_all):
                    out_path_ply = Path(ply_path)
                    log(f'[RGBDImage::estimate_floor] PLY saved to to {out_path_ply.parent.name}/{out_path_ply.name}]')

        return {
            "floor_normal": n_final.astype(np.float64),
            "floor_offset": float(d_final),
            "wall_normal": None if 'n_wall' not in locals() or n_wall is None else n_wall.astype(np.float64),
            "wall_offset": None if 'd_wall' not in locals() or d_wall is None else float(d_wall),
        }

    @staticmethod
    def estimate_floor2(
            *rgbd_images: "RGBDImage",
            export: bool = True,
            export_path: Optional[Union[Path, str]] = None,
            rotate: Optional[Literal['90_CLOCKWISE', '90_COUNTERCLOCKWISE', '180']] = None,
    ):
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name, dtype=dtype).to(device).eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        floor_ids = [k for k, v in id2label.items() if any(t in v.lower() for t in ("floor", "ground", "road", "sidewalk", "pavement"))] or [0]

        def rot_cv(img, how):
            if how is None: return img
            code = {'90_CLOCKWISE': cv2.ROTATE_90_CLOCKWISE,
                    '90_COUNTERCLOCKWISE': cv2.ROTATE_90_COUNTERCLOCKWISE,
                    '180': cv2.ROTATE_180}[how]
            return cv2.rotate(img, code)

        def inv_rot(how):
            if how is None: return None
            return {'90_CLOCKWISE': '90_COUNTERCLOCKWISE',
                    '90_COUNTERCLOCKWISE': '90_CLOCKWISE',
                    '180': '180'}[how]

        all_pts, all_w = [], []
        per_view_valid = []
        overlays, originals = [], []
        inv_rotate = inv_rot(rotate)

        for img in rgbd_images:
            if img is None or img.rgb is None or img.depth is None:
                per_view_valid.append(False)
                overlays.append(None)
                originals.append(None)
                continue
            per_view_valid.append(True)

            rgb = img.rgb if img.rgb.dtype == np.uint8 else np.clip(img.rgb, 0, 255).astype(np.uint8)
            rgb_in = rot_cv(rgb, rotate)

            with torch.inference_mode():
                inputs = processor(images=rgb_in, return_tensors="pt")
                inputs = {k: v.to(device, dtype=model.dtype) for k, v in inputs.items()}
                logits = model(**inputs).logits
                logits = torch.nn.functional.interpolate(
                    logits, size=rgb_in.shape[:2], mode="bilinear", align_corners=False
                )[0]
                prob_in = logits.softmax(dim=0)[floor_ids].sum(dim=0).float().cpu().numpy()

            # --- gate to bottom band ---
            H_in, W_in = prob_in.shape
            bottom_band_ratio = 0.50  # bottom %
            yy_in = np.arange(H_in, dtype=np.float32)[:, None]
            bottom_mask_in = (yy_in >= (1.0 - bottom_band_ratio) * (H_in - 1)).astype(np.float32)
            prob_in *= bottom_mask_in

            # rotate probabilities back to original image orientation
            prob = rot_cv(prob_in, inv_rotate).astype(np.float32)
            mask0 = (prob >= 0.5)
            if np.any(mask0):
                num, lab = cv2.connectedComponents(mask0.astype(np.uint8))
                if num > 1:
                    areas = [(lab == i).sum() for i in range(1, num)]
                    keep = 1 + int(np.argmax(areas))
                    mask = (lab == keep)
                else:
                    mask = mask0
            else:
                mask = np.zeros_like(mask0, bool)

            valid = np.isfinite(img.depth) & (img.depth > 0) & np.isfinite(img.points_world).all(-1)
            pick = mask & valid
            if np.count_nonzero(pick) >= 50:
                X = img.points_world[pick].astype(np.float64)
                w = (prob[pick].astype(np.float64) + 1e-3)
                all_pts.append(X.astype(np.float32))
                all_w.append(w.astype(np.float32))

            originals.append(rgb)
            if export:
                base = rgb.astype(np.float32)
                orange = np.array([255, 165, 0], np.float32)
                for c in range(3):
                    ch = base[..., c]
                    ch[mask] = orange[c] * 0.35 + ch[mask] * 0.65
                    base[..., c] = ch
                overlays.append(base.clip(0, 255).astype(np.uint8))
            else:
                overlays.append(None)

        if not all_pts:
            return {"floor_normal": np.array([0, 0, 1], np.float64), "floor_offset": 0.0, "wall_normal": None, "wall_offset": None}

        X = np.concatenate(all_pts, 0).reshape(-1, 3).astype(np.float64)
        w = np.concatenate(all_w, 0).reshape(-1).astype(np.float64)
        w = np.maximum(w, 1e-8)
        w /= (w.sum() + 1e-12)
        mu = (w[:, None] * X).sum(0)
        Xc = X - mu
        C = (w[:, None, None] * (Xc[:, :, None] @ Xc[:, None, :])).sum(0)
        _, V = np.linalg.eigh(C)
        n = V[:, 0]
        d = -float(n @ mu)
        for _ in range(5):
            sd = X @ n + d
            s = 1.4826 * (np.median(np.abs(sd)) + 1e-9)
            rw = 1.0 / (1.0 + (sd / (2.0 * s)) ** 2)
            ww = (w * rw)
            ww /= (ww.sum() + 1e-12)
            mu = (ww[:, None] * X).sum(0)
            Xc = X - mu
            C = (ww[:, None, None] * (Xc[:, :, None] @ Xc[:, None, :])).sum(0)
            _, V = np.linalg.eigh(C)
            n = V[:, 0]
            d = -float(n @ mu)
        sd = X @ n + d
        if np.median(sd) < 0: n, d = -n, -d
        n_final = n.astype(np.float64)
        d_final = float(d)

        mids = [i for i, b in enumerate(per_view_valid) if b]
        mid_idx = mids[len(mids) // 2]
        c2w = np.linalg.inv(rgbd_images[mid_idx].extrinsic_w2c).astype(np.float64)
        Cmid = c2w[:3, 3]
        fwd = c2w[:3, :3][:, 2]
        fwd = fwd - n_final * (fwd @ n_final)
        if np.linalg.norm(fwd) < 1e-9:
            a = np.array([1.0, 0.0, 0.0])
            fwd = a - n_final * (a @ n_final)
        fwd /= (np.linalg.norm(fwd) + 1e-12)
        n_wall = fwd.astype(np.float64)
        d_wall = -float(n_wall @ (Cmid + 5.0 * n_wall))

        if export and export_path is not None:
            # ---- PLY ----
            pts_all, cols_all = [], []
            for img in rgbd_images:
                if img is None or img.depth is None: continue
                pcd = img.unproject().open3d
                if len(pcd.points) > 0:
                    pts_all.append(np.asarray(pcd.points))
                    cols_all.append(np.asarray(pcd.colors))
            if pts_all:
                P = np.concatenate(pts_all, 0)
                Cc = np.concatenate(cols_all, 0)
            else:
                P = np.zeros((0, 3), np.float64)
                Cc = np.zeros((0, 3), np.float64)

            Xproj = X - (X @ n_final + d_final)[:, None] * n_final[None]
            muF = Xproj.mean(0)
            U, Sv, Vt = np.linalg.svd((Xproj - muF)[::max(1, len(Xproj) // 5000)], full_matrices=False)
            e1 = Vt[0]
            e1 -= n_final * (e1 @ n_final)
            e1 /= (np.linalg.norm(e1) + 1e-12)
            e2 = np.cross(n_final, e1)
            e2 /= (np.linalg.norm(e2) + 1e-12)
            Q = np.stack([(Xproj - muF) @ e1, (Xproj - muF) @ e2], 1)
            qmin, qmax = Q.min(0), Q.max(0)
            u = np.linspace(qmin[0], qmax[0], 200)
            v = np.linspace(qmin[1], qmax[1], 200)
            Uv, Vv = np.meshgrid(u, v)
            Fpts = (muF[None, None, :] + Uv[..., None] * e1[None, None, :] + Vv[..., None] * e2[None, None, :]).reshape(-1, 3)
            Fcols = np.tile(np.array([[1.0, 0.0, 1.0]]), (Fpts.shape[0], 1))

            w_e1 = e1 - n_wall * (e1 @ n_wall)
            if np.linalg.norm(w_e1) < 1e-9: w_e1 = e2
            w_e1 /= (np.linalg.norm(w_e1) + 1e-12)
            w_e2 = np.cross(n_wall, w_e1)
            w_e2 /= (np.linalg.norm(w_e2) + 1e-12)
            Wcenter = Cmid + 5.0 * n_wall
            u = np.linspace(qmin[0], qmax[0], 180)
            v = np.linspace(-0.5 * (qmax[1] - qmin[1]), 0.5 * (qmax[1] - qmin[1]), 180)
            U2, V2 = np.meshgrid(u, v)
            Wpts = (Wcenter[None, None, :] + U2[..., None] * w_e1[None, None, :] + V2[..., None] * w_e2[None, None, :]).reshape(-1, 3)
            Wcols = np.tile(np.array([[0.0, 1.0, 0.0]]), (Wpts.shape[0], 1))

            pcd = o3d.geometry.PointCloud()
            if P.size:
                pcd.points = o3d.utility.Vector3dVector(np.vstack([P, Fpts, Wpts]))
                pcd.colors = o3d.utility.Vector3dVector(np.vstack([Cc, Fcols, Wcols]))
            else:
                pcd.points = o3d.utility.Vector3dVector(np.vstack([Fpts, Wpts]))
                pcd.colors = o3d.utility.Vector3dVector(np.vstack([Fcols, Wcols]))

            ply_path = Path(export_path).with_suffix('.ply') if isinstance(export_path, (str, Path)) else Path("floor_wall_debug.ply")
            o3d.io.write_point_cloud(str(ply_path), pcd)
            log(f'[RGBDImage::estimate_floor] PLY saved to to {ply_path.parent.name}/{ply_path.name}]')

            # ---- PNG GRID (row1: originals, row2: overlays) ----
            imgs_orig = [im for im, v in zip(originals, per_view_valid) if v and im is not None]
            imgs_over = [ov for ov, v in zip(overlays, per_view_valid) if v and ov is not None]
            if imgs_orig and imgs_over and len(imgs_orig) == len(imgs_over):
                th = max(im.shape[0] for im in imgs_orig)
                tw = max(im.shape[1] for im in imgs_orig)

                def fit(im):
                    h, w = im.shape[:2]
                    s = min(tw / w, th / h)
                    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
                    out = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    tile = np.zeros((th, tw, 3), np.uint8)
                    y0, x0 = (th - nh) // 2, (tw - nw) // 2
                    tile[y0:y0 + nh, x0:x0 + nw] = out
                    return tile

                row1 = [fit(im) for im in imgs_orig]
                row2 = [fit(im) for im in imgs_over]
                cols = len(row1)
                pad = 8
                grid_h = 2 * th + pad
                grid_w = cols * tw + (cols - 1) * pad
                grid = np.zeros((grid_h, grid_w, 3), np.uint8)
                for j in range(cols):
                    x = j * (tw + pad)
                    grid[0:th, x:x + tw] = row1[j]
                    grid[th + pad:th + pad + th, x:x + tw] = row2[j]
                png_path = ply_path.with_suffix('.png')
                cv2.imwrite(str(png_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                log(f'[RGBDImage::estimate_floor] PNG saved to to {png_path.parent.name}/{png_path.name}]')

        return {
            "floor_normal": n_final.astype(np.float64),
            "floor_offset": float(d_final),
            "wall_normal": n_wall.astype(np.float64),
            "wall_offset": float(d_wall),
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
            depth_filter: Literal['aligned', 'bilateral_spatial', 'bilateral_temporal'] = 'bilateral_spatial'
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
            depth_path = session_root / 'orbbec' / f'cam{cam_idx:02d}' / (f'depth_{depth_filter}' if depth_filter in ['aligned'] else f'depth_filtering_{depth_filter}') / f'{depth_ts}.png'
        else:
            depth_path = None
        # load data and create RGBDImage
        return cls.from_path(rgb_path, depth_path, intrinsic, extrinsic_w2c=extrinsic_w2c, mask_path=mask_path, is_inverse_depth=False, depth_divide=13.0)


class PixelPoints:
    O3D_VISUALIZER_CACHE = {}
    PYRENDER_RENDERER_CACHE = {}

    def __init__(self, pixel_points: np.ndarray, pixel_colors: Optional[np.ndarray] = None, pixel_valid: Optional[np.ndarray] = None, pixel_features: Optional[Dict[str, np.ndarray]] = None):
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
                pcd_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=40))
                normal_valid = pcd_o3d.normalize_normals().normals
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
        # ---- Soft trim near the FG boundary (no API changes) -----------------
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

    def _project_pyrender(  # method of your class
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
            a = np.array([1.0, 0.0, 0.0], np.float64) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], np.float64)
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
                corners = np.array([[-s, -s], [s, -s], [s, s], [-s, s]], np.float64)
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
            if texture_path is not None and Path(texture_path).exists():
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
                except Exception:
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
                texture_path=PathUtils.resources_path() / 'checkerboard.jpg'
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
