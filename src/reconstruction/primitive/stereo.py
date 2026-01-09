from pathlib import Path
from typing import Optional, Tuple, Union, Literal, Dict

import cv2
import numpy as np
import torch
import xxhash
from typing_extensions import Unpack, TypeVarTuple

from reconstruction.primitive.pcd import PixelPoints, RGBDImage
from reconstruction.primitive.splat import PixelGSPoints, GSImage
from utils.misc import log


class StereoImage(RGBDImage):
    """
    This class represents a stereo image containing RGB and depth data.
    It holds the data of both the "Left" and the "Right" RGBD images, stitching the resulting point clouds.
    """

    def __init__(self,
                 left_rgb: np.ndarray,
                 left_mask: np.ndarray,
                 left_intrinsic: np.ndarray,
                 left_extrinsic_w2c: np.ndarray,
                 right_rgb: np.ndarray,
                 right_mask: np.ndarray,
                 right_intrinsic: np.ndarray,
                 right_extrinsic_w2c: np.ndarray,
                 precomputed_lr_disparity: Optional[np.ndarray] = None,
                 precomputed_lr_depth: Optional[np.ndarray] = None,
                 use_cache: bool = True,
                 disparity_estimator_model: Literal['raftstereo', 'foundationstereo'] = 'raftstereo',
                 disparity_estimator_checkpoint: Optional[Union[Path, str]] = None,
                 features: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize a stereo image.
        As a trick to reuse RGBDImage methods, we concatenate the RGB images along the height dimension (axis=0). We supply the parent with the concatenated RGBs and masks, computed depths.
        For intrinsics and extrinsics, we provide the left camera's intrinsics and extrinsics, to maintain the `reproject_to` functionality.
        We store the rectified intrinsics and intrinsics in the class and use them for the unprojection.

        Parameters
        ----------
        left_rgb : np.ndarray
            Left RGB image of shape (H, W, 3).
        left_mask : np.ndarray
            Left mask image of shape (H, W) where True indicates valid pixels.
        left_intrinsic : np.ndarray
            Camera intrinsic matrix of shape (3, 3).
        left_extrinsic_w2c : np.ndarray
            Camera extrinsic matrix of shape (4, 4).
        right_rgb : Optional[np.ndarray]
            Right RGB image of shape (H, W, 3).
        right_mask : np.ndarray
            Right mask image of shape (H, W) where True indicates valid pixels.
        right_intrinsic : np.ndarray
            Camera intrinsic matrix of shape (3, 3).
        right_extrinsic_w2c : np.ndarray
            Camera extrinsic matrix of shape (4, 4).
        precomputed_lr_disparity : Optional[np.ndarray], optional
            Precomputed disparity map of shape (2H, W). If provided, it will be used instead of estimating the disparity.
        precomputed_lr_depth : Optional[np.ndarray], optional
            Precomputed depth map of shape (2H, W). If provided, it will be used instead of estimating the depth.
            This is useful for cases where the depth has already been computed and want to avoid recomputing it.
        use_cache: bool, optional
            If True, the method will use cached rectification data if available. Default is True.
        disparity_estimator_model: Literal['raftstereo', 'foundationstereo'], optional
            The disparity estimation model to use. Default is 'raftstereo'.
        disparity_estimator_checkpoint: Union[Path, str], optional
            The checkpoint for the disparity estimation model. Default is None which will load the default model checkpoint for the specified disparity estimator.
        """
        if precomputed_lr_disparity is not None:
            assert isinstance(precomputed_lr_disparity, np.ndarray) and precomputed_lr_disparity.ndim == 2 and precomputed_lr_disparity.shape == (left_rgb.shape[0] * 2, left_rgb.shape[1]), "Precomputed LR disparity must be a 2D np.ndarray of shape (2H, W)."
            assert isinstance(precomputed_lr_depth, np.ndarray) and precomputed_lr_depth.ndim == 2 and precomputed_lr_depth.shape == (left_rgb.shape[0] * 2, left_rgb.shape[1]), "Precomputed LR depth must be a 2D np.ndarray of shape (2H, W)."
            lr_disparity = precomputed_lr_disparity
            lr_depth = precomputed_lr_depth
        else:
            # compute rectification data
            self.rectification_data = dict(
                left_to_right=StereoUtils.compute_rectification_data(
                    ref_intrinsic=left_intrinsic.copy(),
                    ref_extrinsic_w2c=left_extrinsic_w2c.copy(),
                    other_intrinsic=right_intrinsic.copy(),
                    other_extrinsic_w2c=right_extrinsic_w2c.copy(),
                    image_size_hw=(left_rgb.shape[0], left_rgb.shape[1]),
                    use_cache=use_cache
                ),
                right_to_left=StereoUtils.compute_rectification_data(
                    ref_intrinsic=right_intrinsic.copy(),
                    ref_extrinsic_w2c=right_extrinsic_w2c.copy(),
                    other_intrinsic=left_intrinsic.copy(),
                    other_extrinsic_w2c=left_extrinsic_w2c.copy(),
                    image_size_hw=(right_rgb.shape[0], right_rgb.shape[1]),
                    use_cache=use_cache
                )
            )

            # L -> R disparity
            l2r_rect_left, l2r_rect_right, l2r_rect_left_mask, l2r_rect_right_mask = StereoUtils.rectify(
                ref_rgb=left_rgb,
                other_rgb=right_rgb,
                rectification_data=self.rectification_data['left_to_right'],
                ref_mask=left_mask,
                other_mask=right_mask
            )
            l2r_rect_left_disparity_rc = StereoUtils.estimate_disparity(
                rect_rgb_ref=l2r_rect_left * (1.0 if disparity_estimator_model != 'raftstereo' else l2r_rect_left_mask.astype(np.float32)[..., None]),
                rect_rgb_other=l2r_rect_right * (1.0 if disparity_estimator_model != 'raftstereo' else l2r_rect_right_mask.astype(np.float32)[..., None]),
                model=disparity_estimator_model,
                model_checkpoint=disparity_estimator_checkpoint,
                is_first_left=(np.nonzero(l2r_rect_left_mask)[1].mean() > np.nonzero(l2r_rect_right_mask)[1].mean()).item(),  # FIX: Uni-directional disparity estimator require first image's object righter than second's
            )  # L -> R only
            l2r_rect_left_disparity_rc[~l2r_rect_left_mask] = np.inf
            left_rect_points_rc = StereoUtils.disparity_to_points(
                disparity=l2r_rect_left_disparity_rc,
                rectification_data=self.rectification_data['left_to_right'],
            )
            left_rect_points_rc[np.isnan(left_rect_points_rc)] = 0.0
            left_depth = StereoUtils.points_rc_to_depth_c(
                left_rect_points_rc,
                l2r_rect_left_mask,
                rectification_data=self.rectification_data['left_to_right'],
                target_intrinsic=left_intrinsic,
                target_extrinsic_w2c=left_extrinsic_w2c,
                use_vis_cache=True
            )
            left_disparity = cv2.remap(l2r_rect_left_disparity_rc, self.rectification_data['left_to_right']['ref_rectify_mat_x_inv'], self.rectification_data['left_to_right']['ref_rectify_mat_y_inv'], interpolation=cv2.INTER_LINEAR)  # approximate L -> R disparity by undoing the rectification warping

            # R -> L disparity
            r2l_rect_left, r2l_rect_right, r2l_rect_left_mask, r2l_rect_right_mask = StereoUtils.rectify(
                ref_rgb=right_rgb,
                other_rgb=left_rgb,
                rectification_data=self.rectification_data['right_to_left'],
                ref_mask=right_mask,
                other_mask=left_mask
            )
            r2l_rect_left_disparity_rc = StereoUtils.estimate_disparity(
                rect_rgb_ref=r2l_rect_left * (1.0 if disparity_estimator_model != 'raftstereo' else r2l_rect_left_mask.astype(np.float32)[..., None]),
                rect_rgb_other=r2l_rect_right * (1.0 if disparity_estimator_model != 'raftstereo' else r2l_rect_right_mask.astype(np.float32)[..., None]),
                model=disparity_estimator_model,
                model_checkpoint=disparity_estimator_checkpoint,
                is_first_left=(np.nonzero(r2l_rect_left_mask)[1].mean() > np.nonzero(r2l_rect_right_mask)[1].mean()).item(),  # FIX: Uni-directional disparity estimator require first image's object righter than second's
            )  # L -> R only
            r2l_rect_left_disparity_rc[~r2l_rect_left_mask] = np.inf
            right_rect_points_rc = StereoUtils.disparity_to_points(
                disparity=r2l_rect_left_disparity_rc,
                rectification_data=self.rectification_data['right_to_left'],
            )
            right_rect_points_rc[np.isnan(right_rect_points_rc)] = 0.0
            right_depth = StereoUtils.points_rc_to_depth_c(
                right_rect_points_rc,
                r2l_rect_left_mask,
                rectification_data=self.rectification_data['right_to_left'],
                target_intrinsic=right_intrinsic,
                target_extrinsic_w2c=right_extrinsic_w2c,
                use_vis_cache=True
            )
            right_disparity = cv2.remap(r2l_rect_left_disparity_rc, self.rectification_data['right_to_left']['ref_rectify_mat_x_inv'], self.rectification_data['right_to_left']['ref_rectify_mat_y_inv'], interpolation=cv2.INTER_LINEAR)  # approximate R -> L disparity by undoing the rectification warping
            lr_disparity = np.concatenate([left_disparity, right_disparity], axis=0)  # (2H, W)
            lr_depth = np.concatenate([left_depth, right_depth], axis=0)

        # Create the RGBD image with the quantities from the two source images concatenated along the height dimension
        super().__init__(
            rgb=np.concatenate([left_rgb, right_rgb], axis=0),
            mask=np.concatenate([left_mask, right_mask], axis=0),
            depth=lr_depth,
            intrinsic=left_intrinsic,
            extrinsic_w2c=left_extrinsic_w2c,
            features=dict(
                disparity=lr_disparity,
            ) | (features if features is not None else {})
        )
        self.right_intrinsic = right_intrinsic
        self.right_extrinsic_w2c = right_extrinsic_w2c

    def split_lr(self) -> Tuple[RGBDImage, RGBDImage]:
        """
        Split the stereo image into left and right RGBD images.

        Returns
        -------
        Tuple[RGBDImage, RGBDImage]
            A tuple containing the left and right RGBD images.
        """
        height = self.rgb.shape[0] // 2
        left_rgb = self.rgb[:height]
        left_mask = self.mask[:height]
        left_depth = self.depth[:height]
        left_disparity = self.features['disparity'][:height]
        right_rgb = self.rgb[height:]
        right_mask = self.mask[height:]
        right_depth = self.depth[height:]
        right_disparity = self.features['disparity'][height:]

        left_image = RGBDImage(
            rgb=left_rgb,
            mask=left_mask,
            depth=left_depth,
            intrinsic=self.intrinsic,
            extrinsic_w2c=self.extrinsic_w2c,
            features=dict(disparity=left_disparity)
        )
        right_image = RGBDImage(
            rgb=right_rgb,
            mask=right_mask,
            depth=right_depth,
            intrinsic=self.right_intrinsic,
            extrinsic_w2c=self.right_extrinsic_w2c,
            features=dict(disparity=right_disparity)
        )
        return left_image, right_image

    def unproject(self) -> 'PixelPoints':
        """
        Unproject the stereo image to a point cloud.

        Returns
        -------
        PixelPoints
            A PixelPoints object containing the unprojected point cloud.
        """
        height = self.rgb.shape[0]
        left_pcd = RGBDImage(
            rgb=self.rgb[:height // 2],
            depth=self.depth[:height // 2],
            mask=self.mask[:height // 2],
            intrinsic=self.intrinsic,
            extrinsic_w2c=self.extrinsic_w2c
        ).unproject()
        right_pcd = RGBDImage(
            rgb=self.rgb[height // 2:],
            depth=self.depth[height // 2:],
            mask=self.mask[height // 2:],
            intrinsic=self.right_intrinsic,
            extrinsic_w2c=self.right_extrinsic_w2c
        ).unproject()
        return PixelPoints(
            pixel_points=np.concatenate([left_pcd.points, right_pcd.points], axis=0),
            pixel_colors=np.concatenate([left_pcd.colors, right_pcd.colors], axis=0),
            pixel_valid=np.concatenate([left_pcd.valid, right_pcd.valid], axis=0),
        )

    # noinspection PyTypeChecker
    @classmethod
    def from_rgb_images(cls, *images: RGBDImage, return_last_as_rgbd: bool = False, **stereo_image_kwargs) -> Union['StereoImage', Tuple['StereoImage', ...], Tuple['StereoImage', Unpack[TypeVarTuple("Ts")], RGBDImage]]:
        """
        Create Stereo images instances from successive RGB images (i.e. we ignore the depth information in the RGBDImage instances).

        Parameters
        ----------
        *images : RGBDImage
            A variable number of RGBD images, each of shape (H, W, 3) for RGB and (H, W) for depth.
        return_last_as_rgbd : bool, optional
            If True, the last RGBD image will be included in the output as the last element, otherwise the output will contain only the StereoImage instances, one less than the number of input RGBD images. Default is False.

        Returns
        -------
        Union[Tuple[..., 'StereoImage'], Tuple[..., 'StereoImage', RGBDImage]]
            A tuple containing the StereoImage and optionally the last RGBDImage. If only one StereoImage is created, it will be returned directly.
        """
        if len(images) < 2:
            raise ValueError("At least two RGBD images are required to create a StereoImage.")

        stereo_images = []
        for i in range(len(images) - 1):
            left_image = images[i]
            right_image = images[i + 1]
            stereo_image = StereoImage(
                left_rgb=left_image.rgb,
                left_mask=left_image.mask,
                left_intrinsic=left_image.intrinsic,
                left_extrinsic_w2c=left_image.extrinsic_w2c,
                right_rgb=right_image.rgb,
                right_mask=right_image.mask,
                right_intrinsic=right_image.intrinsic,
                right_extrinsic_w2c=right_image.extrinsic_w2c,
                **stereo_image_kwargs
            )
            stereo_images.append(stereo_image)

        if not return_last_as_rgbd:
            return tuple(stereo_images) if len(stereo_images) > 1 else stereo_images[0]

        return tuple(stereo_images + [images[-1]])


class StereoGSImage(StereoImage):
    def unproject(self, use_cache: bool = True) -> 'PixelGSPoints':
        """
        Unproject the stereo image to a Gaussian splats cloud.

        Parameters
        ----------
        use_cache : bool, optional
            If True, the method will use cached rectification data if available. Default is True.

        Returns
        -------
        PixelGSPoints
            A PixelGSPoints object containing the unprojected and stitched LR splats.
        """
        pcd = super().unproject()
        return PixelGSPoints.from_pcd(pcd, **self.features)

    def split_lr(self) -> Tuple[GSImage, GSImage]:
        """
        Split the stereo GS image into left and right GS images.

        Returns
        -------
        Tuple[GSImage, GSImage]
            A tuple containing the left and right GS images.
        """
        height = self.rgb.shape[0] // 2
        left_gs_image = GSImage(
            rgb=self.rgb[:height],
            mask=self.mask[:height],
            depth=self.depth[:height],
            scale=self.features['scale'][:height],
            rotation=self.features['rotation'][:height],
            opacity=self.features['opacity'][:height],
            intrinsic=self.intrinsic,
            extrinsic_w2c=self.extrinsic_w2c,
            features=dict(disparity=self.features['disparity'][:height])
        )
        right_gs_image = GSImage(
            rgb=self.rgb[height:],
            mask=self.mask[height:],
            depth=self.depth[height:],
            scale=self.features['scale'][height:],
            rotation=self.features['rotation'][height:],
            opacity=self.features['opacity'][height:],
            intrinsic=self.right_intrinsic,
            extrinsic_w2c=self.right_extrinsic_w2c,
            features=dict(disparity=self.features['disparity'][height:])
        )
        return left_gs_image, right_gs_image

    def save_png(self, out_path: Optional[Union[Path, str]] = None, striped: bool = False) -> Optional[np.ndarray]:
        """
        Save the stereo GS image as a PNG file.

        Parameters
        ----------
        out_path : Optional[Union[Path, str]], optional
            The output path to save the PNG file. If None, the method will not save the image. Default is None.
        striped : bool, optional
            If True, the image will be saved with a stripe pattern for visualization. Default is False.

        Returns
        -------
        Optional[np.ndarray]
            The saved RGB image as a numpy array if out_path is provided, otherwise None.
        """
        gs_l, gs_r = self.split_lr()
        gs_img_l = gs_l.save_png(out_path=None, striped=striped)
        gs_img_r = gs_r.save_png(out_path=None, striped=striped)
        png_image = np.concatenate([gs_img_l, gs_img_r], axis=0)
        if out_path is None:
            return png_image
        cv2.imwrite(str(out_path), png_image)
        log(f'[{self.__class__.__name__}::save_png] StereoGSImage saved to {out_path}', 'debug')
        return None


    @classmethod
    def from_gs_images(cls, gs_image_l: GSImage, gs_image_r: GSImage) -> 'StereoGSImage':
        """
        Create a StereoGSImage instance from two GSImage instances.

        Parameters
        ----------
        gs_image_l : GSImage
            The GSImage instance for the left camera.
        gs_image_r : GSImage
            The GSImage instance for the right camera.

        Returns
        -------
        StereoGSImage
            A StereoGSImage instance containing the left and right GSImages.
        """
        stereo_gs_image = cls(
            left_rgb=gs_image_l.rgb,
            left_mask=gs_image_l.mask,
            left_intrinsic=gs_image_l.intrinsic,
            left_extrinsic_w2c=gs_image_l.extrinsic_w2c,
            right_rgb=gs_image_r.rgb,
            right_mask=gs_image_r.mask,
            right_intrinsic=gs_image_r.intrinsic,
            right_extrinsic_w2c=gs_image_r.extrinsic_w2c,
            precomputed_lr_disparity=np.concatenate([gs_image_l.features['disparity'], gs_image_r.features['disparity']], axis=0) if 'disparity' in gs_image_l.features and 'disparity' in gs_image_r.features else None,
            precomputed_lr_depth=np.concatenate([gs_image_l.depth, gs_image_r.depth], axis=0),
            features={k: np.concatenate([gs_image_l.features[k], gs_image_r.features[k]], axis=0) for k in gs_image_l.features if k in gs_image_r.features}
        )
        stereo_gs_image.gs_image_l = gs_image_l
        stereo_gs_image.gs_image_r = gs_image_r
        return stereo_gs_image


class StereoUtils:
    RAFT_STEREO_FEATURE_EXTRACTOR = {}  # <model_checkpoint: UnetExtractor>
    RAFT_STEREO = {}  # <model_checkpoint: GPSRaftStereo>
    FOUNDATION_STEREO = {}  # <model_checkpoint: FoundationStereo>
    RECTIFICATION_CACHE = {}

    @classmethod
    def hash_array(cls, *arr: np.ndarray) -> str:
        """
        Hash one or more numpy arrays to a string for caching purposes.

        Parameters
        ----------
        *arr : np.ndarray
            One or more numpy arrays to hash.

        Returns
        -------
        str
            A SHA-256 hash of the concatenated byte representation of the input arrays.
        """
        return xxhash.xxh64(b''.join(a.tobytes() for a in arr)).hexdigest()

    @classmethod
    def invert_map(cls, map_x: np.ndarray, map_y: np.ndarray, image_size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Approximate inverse of OpenCV warp maps, but without the Python loops.

        Returns
        -------
        inv_map_x, inv_map_y : np.ndarray  (float32, shape = H×W)
            inv_map_y[v, u]  gives the *source‑row* that lands at (v, u)
            inv_map_x[v, u]  gives the *source‑col* that lands at (v, u)
            Unmapped pixels are filled with −1 (change if you prefer NaN).
        """
        H, W = image_size_hw
        inv_map_x = np.full((H, W), -1, np.float32)
        inv_map_y = np.full((H, W), -1, np.float32)

        # 1. integerise the forward map once
        src_x = np.rint(map_x).astype(np.int32)  # round ≈ same as your int(np.round())
        src_y = np.rint(map_y).astype(np.int32)

        # 2. mask-of samples that land *inside* the source image
        valid = (0 <= src_x) & (src_x < W) & (0 <= src_y) & (src_y < H)

        # 3. destination indices (u,v) as 2‑D grids
        v_dst, u_dst = np.indices((H, W), dtype=np.int32)

        # 4. scatter:   (src_y, src_x)  ←‑ (v_dst, u_dst)
        inv_map_x[src_y[valid], src_x[valid]] = u_dst[valid]
        inv_map_y[src_y[valid], src_x[valid]] = v_dst[valid]

        return inv_map_x, inv_map_y

    @classmethod
    def compute_rectification_data(cls, ref_intrinsic: np.ndarray, ref_extrinsic_w2c: np.ndarray, other_intrinsic: np.ndarray, other_extrinsic_w2c: np.ndarray, image_size_hw: Tuple[int, int], use_cache: bool = False, cache_key_prefix: Optional[str] = '') -> Dict[str, np.ndarray]:
        """
        Compute the rectification parameters for a stereo camera pair.

        Details of cv2.stereoRectify (https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6):
        > void cv::stereoRectify	(	InputArray	cameraMatrix1,
        >     InputArray	distCoeffs1,
        >     InputArray	cameraMatrix2,
        >     InputArray	distCoeffs2,
        >     Size	imageSize,
        >     InputArray	R,
        >     InputArray	T,
        >     OutputArray	R1,
        >     OutputArray	R2,
        >     OutputArray	P1,
        >     OutputArray	P2,
        >     OutputArray	Q,
        >     int	flags = CALIB_ZERO_DISPARITY,
        >     double	alpha = -1,
        >     Size	newImageSize = Size(),
        >     Rect *	validPixROI1 = 0,
        >     Rect *	validPixROI2 = 0
        > )
        > Python:
        > cv.stereoRectify(	cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, R1[, R2[, P1[, P2[, Q[, flags[, alpha[, newImageSize]]]]]]]]	) ->	R1, R2, P1, P2, Q, validPixROI1, validPixROI2
        >
        > Computes rectification transforms for each head of a calibrated stereo camera.
        >
        > Parameters:
        > cameraMatrix1	First camera intrinsic matrix.
        > distCoeffs1	First camera distortion parameters.
        > cameraMatrix2	Second camera intrinsic matrix.
        > distCoeffs2	Second camera distortion parameters.
        > imageSize	    Size of the image used for stereo calibration.
        > R	            Rotation matrix from the coordinate system of the first camera to the second camera, see stereoCalibrate.
        > T	            Translation vector from the coordinate system of the first camera to the second camera, see stereoCalibrate.
        > R1	        Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system.
        > R2	        Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix brings points given in the unrectified second camera's coordinate system to points in the rectified second camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified second camera's coordinate system to the rectified second camera's coordinate system.
        > P1            Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified first camera's image.
        > P2            Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified second camera's image.
        > Q	            Output disparity-to-depth mapping matrix (see reprojectImageTo3D).
        > flags	        Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
        > alpha	        Free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an intermediate result between those two extreme cases.
        > newImageSize	New image resolution after rectification. The same size should be passed to initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0) is passed (default), it is set to the original imageSize . Setting it to a larger value can help you preserve details in the original image, especially when there is a big radial distortion.
        > validPixROI1	Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).
        > validPixROI2	Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).

        Parameters
        ----------
        ref_intrinsic : np.ndarray
            Intrinsics of the reference camera (3x3 matrix).
        ref_extrinsic_w2c : np.ndarray
            Extrinsics of the reference camera (4x4 matrix).
        other_intrinsic : np.ndarray
            Intrinsics of the other camera (3x3 matrix).
        other_extrinsic_w2c : np.ndarray
            Extrinsics of the other camera (4x4 matrix).
        image_size_hw : Tuple[int, int]
            The height and width of the images to be rectified, specified as a tuple (H, W).
        use_cache : bool, optional
            If True, the method will use cached rectification data if available. Default is False.
        cache_key_prefix : Optional[str], optional
            A prefix for the cache key to differentiate between different rectification requests. Default is an empty string.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the rectification parameters:
                - 'ref_intrinsic': Rectified intrinsics of the reference camera (3x3 matrix).
                - 'other_intrinsic': Rectified intrinsics of the other camera (3x3 matrix).
                - 'ref_extrinsic_w2c': Rectified extrinsics of the reference camera (4x4 matrix).
                - 'other_extrinsic_w2c': Rectified extrinsics of the other camera (4x4 matrix).
                - 'tf_x': Translation factor in the x direction.
                - 'baseline': Baseline between the two cameras (float).
                - 'ref_rectify_mat_x': Rectification map for the x-coordinates of the reference camera (H, W).
                - 'ref_rectify_mat_y': Rectification map for the y-coordinates of the reference camera (H, W).
                - 'other_rectify_mat_x': Rectification map for the x-coordinates of the other camera (H, W).
                - 'other_rectify_mat_y': Rectification map for the y-coordinates of the other camera (H, W).
        """
        cache_key = str(cache_key_prefix if cache_key_prefix is not None else '') + cls.hash_array(ref_intrinsic, ref_extrinsic_w2c, other_intrinsic, other_extrinsic_w2c)
        if use_cache and cache_key in cls.RECTIFICATION_CACHE:
            return cls.RECTIFICATION_CACHE[cache_key]

        # intrinsics
        K1 = ref_intrinsic.astype(np.float64)
        K2 = other_intrinsic.astype(np.float64)
        # relative pose
        ref_extrinsic_c2w = np.linalg.inv(ref_extrinsic_w2c)
        other_extrinsic_c2w = np.linalg.inv(other_extrinsic_w2c)
        T_ref_to_other = other_extrinsic_w2c @ ref_extrinsic_c2w
        # baseline
        C1_w = ref_extrinsic_c2w[:3, 3].reshape(3, 1)
        C2_w = other_extrinsic_c2w[:3, 3].reshape(3, 1)
        baseline_vec_w = (C2_w - C1_w).ravel()  # from cam‑1 to cam‑2 in world coords
        baseline = float(np.linalg.norm(baseline_vec_w))

        # stereoRectify
        H, W = image_size_hw
        dist_zero = np.zeros((5, 1), dtype=np.float64)
        R1_rect, R2_rect, P1_rect, P2_rect, Q, _, _ = cv2.stereoRectify(
            K1, dist_zero,
            K2, dist_zero,
            (W, H),
            R=T_ref_to_other[:3, :3],
            T=T_ref_to_other[:3, 3],
        )

        # rectification maps
        ref_map_x, ref_map_y = cv2.initUndistortRectifyMap(
            K1, dist_zero, R1_rect, P1_rect, (W, H), cv2.CV_32FC1
        )
        oth_map_x, oth_map_y = cv2.initUndistortRectifyMap(
            K2, dist_zero, R2_rect, P2_rect, (W, H), cv2.CV_32FC1
        )
        ref_map_x_inv, ref_map_y_inv = cls.invert_map(ref_map_x, ref_map_y, image_size_hw=(H, W))
        # oth_map_x_inv, oth_map_y_inv = cls.invert_map(oth_map_x, oth_map_y, image_size_hw=(H, W))

        # rectified extrinsics (world -> rectified‑cam = world -> cam -> rectified‑cam)
        # R1: cam -> rectified-cam
        ref_c2rc = np.eye(4, dtype=ref_extrinsic_w2c.dtype)
        ref_c2rc[:3, :3] = R1_rect
        ref_extrinsic_w2rc = ref_c2rc @ ref_extrinsic_w2c  # T_w2rc = T_c2rc @ T_w2c
        # R2: cam -> rectified-cam
        other_c2rc = np.eye(4, dtype=other_extrinsic_w2c.dtype)
        other_c2rc[:3, :3] = R2_rect
        other_extrinsic_w2rc = other_c2rc @ other_extrinsic_w2c  # T_w2rc = T_c2rc @ T_w2c

        result: Dict[str, np.ndarray] = {
            # intrinsics
            "ref_intrinsic": P1_rect[:3, :3],
            "other_intrinsic": P2_rect[:3, :3],
            # extrinsics (world -> rectified cams)
            "ref_extrinsic_w2rc": ref_extrinsic_w2rc,
            "other_extrinsic_w2rc": other_extrinsic_w2rc,
            # baseline
            "Q": Q,
            "baseline": baseline,
            # remap tables (H,W)
            "ref_rectify_mat_x": ref_map_x,
            "ref_rectify_mat_y": ref_map_y,
            "ref_rectify_mat_x_inv": ref_map_x_inv,
            "ref_rectify_mat_y_inv": ref_map_y_inv,
            "other_rectify_mat_x": oth_map_x,
            "other_rectify_mat_y": oth_map_y,
            # "other_rectify_mat_x_inv": oth_map_x_inv,
            # "other_rectify_mat_y_inv": oth_map_y_inv,
        }
        if use_cache:
            cls.RECTIFICATION_CACHE[cache_key] = result
        return result

    @classmethod
    def rectify(cls, ref_rgb: np.ndarray, other_rgb: np.ndarray, rectification_data: Dict[str, np.ndarray], ref_mask: Optional[np.ndarray] = None, other_mask: Optional[np.ndarray] = None, *other_data) -> Tuple[np.ndarray, ...]:
        """
        Rectify a stereo pair of RGB images using the provided rectification data.

        Parameters
        ----------
        ref_rgb : np.ndarray
            Reference RGB image of shape (H, W, C).
        other_rgb : np.ndarray
            Other RGB image of shape (H, W, C).
        rectification_data : Dict[str, np.ndarray]
            Rectification data containing the rectification maps and intrinsic matrices. See `compute_rectification_data` for details.
        ref_mask : Optional[np.ndarray]
            Mask for the reference image of shape (H, W) where True indicates valid pixels.
        other_mask : Optional[np.ndarray]
            Mask for the other image of shape (H, W) where True indicates valid pixels.

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            Rectified reference and other RGB images and possible masks:
            - ref_rgb_rect, other_rgb_rect[, ref_mask_rect, other_mask_rect]
        """
        ref_map_x = rectification_data['ref_rectify_mat_x']
        ref_map_y = rectification_data['ref_rectify_mat_y']
        other_map_x = rectification_data['other_rectify_mat_x']
        other_map_y = rectification_data['other_rectify_mat_y']

        ref_rgb_rect = cv2.remap(ref_rgb, ref_map_x, ref_map_y, interpolation=cv2.INTER_LINEAR)
        other_rgb_rect = cv2.remap(other_rgb, other_map_x, other_map_y, interpolation=cv2.INTER_LINEAR)
        return_items = [ref_rgb_rect, other_rgb_rect]

        if ref_mask is not None:
            ref_mask_rect = cv2.remap(ref_mask.astype(np.float32), ref_map_x, ref_map_y, interpolation=cv2.INTER_NEAREST).astype(bool)
            other_mask_rect = cv2.remap(other_mask.astype(np.float32), other_map_x, other_map_y, interpolation=cv2.INTER_NEAREST).astype(bool)
            return_items.extend([ref_mask_rect, other_mask_rect])

        for i, od in enumerate(other_data):
            if isinstance(od, np.ndarray) and od.shape[:2] == ref_rgb.shape[:2]:
                # If the other data is an image of the same size as the RGB images, we can remap it
                od_dtype = od.dtype
                if i % 2 == 0:
                    od_rect = cv2.remap(od.astype(np.float32), ref_map_x, ref_map_y, interpolation=cv2.INTER_LINEAR).astype(od_dtype)
                else:
                    od_rect = cv2.remap(od.astype(np.float32), other_map_x, other_map_y, interpolation=cv2.INTER_LINEAR).astype(od_dtype)
                return_items.append(od_rect)

        return tuple(return_items)

    @classmethod
    def disparity_to_points(cls, disparity: np.ndarray, rectification_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert disparity to depth using the rectification data.

        Parameters
        ----------
        disparity : np.ndarray
            Disparity map of shape (H, W). This is from L->R.
        rectification_data : Dict[str, np.ndarray]
            Rectification data containing the Q matrix (from the L camera after rectification). See `compute_rectification_data` for details.

        Returns
        -------
        np.ndarray
            Depth map of shape (H, W).
        """
        Q = rectification_data['Q']
        if Q.shape != (4, 4):
            raise ValueError("Q matrix must be of shape (4, 4).")
        # Convert disparity to depth using the Q matrix
        pcd_rc = np.ascontiguousarray(cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False))
        return pcd_rc  # (H, W, 3), in meters

    @classmethod
    def points_rc_to_depth_c(cls, points_rc: np.ndarray, mask_rc: np.ndarray, rectification_data: Dict[str, np.ndarray], target_intrinsic: np.ndarray, target_extrinsic_w2c: np.ndarray, use_vis_cache: bool = False) -> np.ndarray:
        """
        Convert points in rectified camera coordinates to depth in the target camera's coordinate system.

        Parameters
        ----------
        points_rc : np.ndarray
            Points in rectified camera coordinates of shape (H, W, 3).
        mask_rc : np.ndarray
            Mask indicating valid pixels of shape (H, W).
        rectification_data : Dict[str, np.ndarray]
            Rectification data containing the rectification maps and extrinsic matrices. See `compute_rectification_data` for details.
        target_intrinsic : np.ndarray
            Target camera intrinsic matrix of shape (3, 3).
        target_extrinsic_w2c : np.ndarray
            Target camera extrinsic matrix (world to camera) of shape (4, 4).
        use_vis_cache : bool, optional
            If True, the method will reuse cached OffScreenRendering data for visualization. Default is False.

        Returns
        -------
        np.ndarray
            Depth map in the target camera's coordinate system of shape (H, W).
        """
        image_rc = RGBDImage(
            rgb=np.zeros(points_rc.shape, dtype=np.uint8),
            depth=np.ascontiguousarray(points_rc[..., 2]),
            mask=mask_rc,
            intrinsic=rectification_data['ref_intrinsic'],
            extrinsic_w2c=rectification_data['ref_extrinsic_w2rc'],
        )
        # When converting from rectified camera coordinates back to camera, increasing the point size will close some holes in the reprojection process
        point_size = 2.0
        return image_rc.reproject(target_intrinsic, target_extrinsic_w2c, target_image_size_hw=(points_rc.shape[0], points_rc.shape[1]), point_size=point_size, is_c2w=False, use_cache=use_vis_cache).depth

    @classmethod
    def batch_lr_to_llrr(cls, lr: torch.Tensor) -> torch.Tensor:
        return lr.swapaxes(0, 1).flatten(0, 1)

    @classmethod
    def llrr_to_batch_lr(cls, lr: torch.Tensor, n_cams_per_batch_sample: int) -> torch.Tensor:
        return lr.unflatten(0, (n_cams_per_batch_sample, -1)).swapaxes(0, 1)

    # noinspection PyUnusedLocal
    @classmethod
    def load_foundationstereo(cls, model_checkpoint: Literal['vits', 'vitl'] = 'vits', **model_kwargs) -> torch.nn.Module:
        """
        Load the Foundation Stereo model. This is a placeholder for the actual implementation.

        Parameters
        ----------
        model_checkpoint : Literal['vits', 'vitl'], optional
            The model checkpoint to load. Default is 'vits'.

        Returns
        -------
        torch.nn.Module
            FoundationStereo model as a torch nn.Module sub-instance.
        """
        if model_checkpoint not in cls.FOUNDATION_STEREO or cls.FOUNDATION_STEREO[model_checkpoint] is None:
            from reconstruction.arch.gpsgaussian.wrapper import GPSFoundationStereo
            cls.FOUNDATION_STEREO[model_checkpoint] = GPSFoundationStereo(model_checkpoint).eval().cuda()
        return cls.FOUNDATION_STEREO[model_checkpoint]

    @classmethod
    def load_raftstereo(cls, model_checkpoint: Union[Path, str] = 'gpsgaussian/gps_21_stage2_final.pth', **model_kwargs) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load the RAFT Stereo model. We use the RaftStereo model from the GPS Gaussian architecture.

        Parameters
        ----------
        model_checkpoint : Optional[Path]
            Path to the model checkpoint. If None, the default model will be used.
        **model_kwargs : dict
            Additional keyword arguments for the model.

        Returns
        -------
        Tuple[torch.nn.Module, torch.nn.Module]
            A tuple containing the RAFT Stereo model and the feature extractor.
        """
        cache_key = str(model_checkpoint if model_checkpoint is not None else 'default')
        if cache_key not in cls.RAFT_STEREO or cls.RAFT_STEREO[cache_key] is None:
            from reconstruction.primitive.splat import GSUtils
            gps_full = GSUtils.load_gps(model_checkpoint, load_raft=True, **model_kwargs)
            cls.RAFT_STEREO[cache_key] = gps_full.raft_stereo
            cls.RAFT_STEREO_FEATURE_EXTRACTOR[cache_key] = gps_full.img_encoder
        return cls.RAFT_STEREO[cache_key], cls.RAFT_STEREO_FEATURE_EXTRACTOR[cache_key]

    @classmethod
    def visualize_disparity(cls, disparity: torch.Tensor) -> torch.Tensor:
        """
        Visualize the disparity map and save it to a file.

        Parameters
        ----------
        disparity : torch.Tensor
            Disparity map of shape (B, 1, H, W).
        """
        # Convert to numpy and normalize for visualization
        disparity_vis = disparity.squeeze().cpu().numpy().copy()
        disparity_mask = ~np.isinf(disparity_vis) & ~np.isnan(disparity_vis)
        disparity_min_val = disparity_vis[disparity_mask].min()
        disparity_max_val = disparity_vis[disparity_mask].max()
        disparity_vis = ((disparity_vis - disparity_min_val) / (disparity_max_val - disparity_min_val)).clip(0, 1) * 255
        disparity_image = cv2.applyColorMap(disparity_vis.clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]  # BGR to RGB
        disparity_image[~disparity_mask] = 0.0
        return torch.from_numpy(np.ascontiguousarray(disparity_image)).float().div(255.0).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    @classmethod
    def estimate_disparity_foundationstereo(cls, rect_rgb_ref: torch.Tensor, rect_rgb_other: torch.Tensor, **model_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate depth from disparity, which was estimated using FoundationStereo.

        Parameters
        ----------
        rect_rgb_ref : torch.Tensor
            Rectified reference RGB image of shape (B, C, H, W).
        rect_rgb_other : torch.Tensor
            Rectified other RGB image of shape (B, C, H, W).
        model_kwargs : dict
            Additional keyword arguments for the model.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the disparity maps from reference to other and from other to reference, each of shape (B, 1, H, W).
        """
        from foundationstereo.core.utils.utils import InputPadder

        # get model
        foundation_stereo = cls.load_foundationstereo(**model_kwargs)

        img_l = rect_rgb_ref
        img_r = rect_rgb_other

        # resize images to 1/4 of the original size
        img_l = torch.nn.functional.interpolate(img_l, scale_factor=0.25, align_corners=True, mode='bilinear')
        img_r = torch.nn.functional.interpolate(img_r, scale_factor=0.25, align_corners=True, mode='bilinear')
        padder = InputPadder(img_l.shape, divis_by=32, force_square=False)
        img_l, img_r = padder.pad(img_l, img_r)
        # forward pass through the model
        with torch.cuda.amp.autocast(True):
            disparity_l2r = foundation_stereo(img_l, img_r, iters=32, test_mode=True)  # (B, 1, H/4, W/4)
        # upscale the disparity maps to the original size
        disparity_l2r = padder.unpad(disparity_l2r.float())
        disparity_l2r = torch.nn.functional.interpolate(disparity_l2r, scale_factor=4.0, align_corners=True, mode='bilinear') * 4.0  # (B, 1, H, W)

        return disparity_l2r  # (B, 1, H, W)

    @classmethod
    def estimate_disparity_raftstereo(cls, rect_rgb_ref: torch.Tensor, rect_rgb_other: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        Estimate depth from disparity, which was estimated using RAFT-Stereo.

        Parameters
        ----------
        rect_rgb_ref : torch.Tensor
            Rectified reference RGB image of shape (B, C, H, W).
        rect_rgb_other : torch.Tensor
            Rectified other RGB image of shape (B, C, H, W).
        model_kwargs : dict
            Additional keyword arguments for the model.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the disparity maps from reference to other and from other to reference, each of shape (B, 1, H, W).
        """
        # get model and feature extractor
        raft_stereo, img_encoder = cls.load_raftstereo(**model_kwargs)

        # prepare inputs
        rect_rgb_llrr = cls.batch_lr_to_llrr(torch.stack([rect_rgb_ref, rect_rgb_other], dim=1))  # (B, 2, C, H, W) -> (B*2, C, H, W)

        # get image features
        with torch.autocast(enabled=True, device_type='cuda'):
            img_feat_llrr = img_encoder(rect_rgb_llrr)  # List[torch.Tensor] each of shape (B*2, D_d, H_d, W_d)

        # Use RAFT stereo to compute disparity
        disparity_llrr = raft_stereo(img_feat_llrr[-1])
        if isinstance(disparity_llrr, (tuple, list)):
            disparity_llrr = torch.cat(disparity_llrr, dim=0)  # Concatenate left and right flows

        # FIX: RAFT-Stereo in GPSGaussian return -disparity
        disparity_llrr = -1.0 * disparity_llrr

        rect_disparity = cls.llrr_to_batch_lr(disparity_llrr, 2)  # (B*2, 1, H, W) -> (B, 2, 1, H, W)
        rect_disparity_l2r = rect_disparity.unbind(dim=1)[0]  # (B, 2, 1, H, W) -> (B, 1, H, W) only l2r
        return rect_disparity_l2r

    @classmethod
    def estimate_disparity(cls, rect_rgb_ref: Union[torch.Tensor, np.ndarray], rect_rgb_other: Union[torch.Tensor, np.ndarray], model: Literal['raftstereo', 'foundationstereo'] = 'raftstereo', model_checkpoint: Optional[Union[Path, str]] = None, is_first_left: bool = True) -> Union[torch.Tensor, np.ndarray]:
        if model_checkpoint is None:
            model_checkpoint = 'neptune://85/best' if model == 'raftstereo' else 'vitl'
        numpy_mode = False
        if isinstance(rect_rgb_ref, np.ndarray):
            numpy_mode = True
            if rect_rgb_ref.ndim == 3:
                rect_rgb_ref = rect_rgb_ref[None]
            rect_rgb_ref = (torch.from_numpy(rect_rgb_ref).permute(0, 3, 1, 2).float() / 255.0).cuda()
            if rect_rgb_other.ndim == 3:
                rect_rgb_other = rect_rgb_other[None]
            rect_rgb_other = (torch.from_numpy(rect_rgb_other).permute(0, 3, 1, 2).float() / 255.0).cuda()
        if not is_first_left:
            rect_rgb_ref = torch.flip(rect_rgb_ref, dims=[-1])
            rect_rgb_other = torch.flip(rect_rgb_other, dims=[-1])
        rect_disparity_l2r = getattr(cls, f'estimate_disparity_{model}')(rect_rgb_ref, rect_rgb_other, model_checkpoint=model_checkpoint)
        if not is_first_left:
            rect_disparity_l2r = -torch.flip(rect_disparity_l2r, dims=[-1])
        if numpy_mode:
            rect_disparity_l2r = rect_disparity_l2r[0].detach().cpu().squeeze().numpy()
        return rect_disparity_l2r


if __name__ == "__main__":
    from utils.calib import CalibrationData
    from reconstruction.primitive.splat import GSImage

    disparity_estimator_model_ = 'foundationstereo'  # 'raftstereo' or 'foundationstereo'
    gs_regressor_checkpoint_ = 'neptune://154/best'  # 'gpsgaussian/gps_21_stage2_final.pth' or 'neptune://85/best'
    data_src_ = 'session'  # 'session' or 'thuman'

    if data_src_ == 'session':
        # read session data
        session_root_ = Path('/home/charisoudis/PyCharmMiscProject/irc-hslu/capturestudio/out/Captures_Apr_May_2025/Aggregated/Thanos_2_Perf_1')
        calibration_session_root_ = Path('/home/charisoudis/PyCharmMiscProject/irc-hslu/capturestudio/out/Captures_Apr_May_2025/Aggregated/Thanos_2_Calib_1')
        calibration_data_ = CalibrationData.from_session(calibration_session_root_)
        #   - read 2 RGBDImages
        rgbd_0_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=8, color_ts=1746110341432, depth_ts=None).resize(1024, 1280 if data_src_ == 'session' else 1024)
        rgbd_1_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=7, color_ts=1746110341432, depth_ts=None).resize(1024, 1280 if data_src_ == 'session' else 1024)
    else:
        # read THUMAN data
        thuman_root_ = Path('/media/charisoudis/nas_transmixr/Simone/Volumetric_Video/Human Datasets/THuman2_1/rendered@2m')
        #   - read 2 RGBDImages
        rgbd_0_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=2)
        rgbd_1_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=3)

    #   - create stereo image
    stereo_image_: StereoImage = StereoImage.from_rgb_images(rgbd_0_, rgbd_1_, disparity_estimator_model=disparity_estimator_model_)
    stereo_image_.save_png(f'{data_src_}_stereo_{disparity_estimator_model_}.png')
    stereo_image_.unproject().save_ply(f'{data_src_}_stereo_{disparity_estimator_model_}.ply')
    #   - reproject to src cameras
    stereo_image_.reproject_to(rgbd_0_).save_png(f'{data_src_}_stereo_0120_{disparity_estimator_model_}.png')
    stereo_image_.reproject_to(rgbd_1_).save_png(f'{data_src_}_stereo_0121_{disparity_estimator_model_}.png')
    #   - create stereo GSImage
    rgbd_s_0_, rgbd_s_1_ = stereo_image_.split_lr()
    rgbd_s_0_.save_png(f'{data_src_}_stereo_0_{disparity_estimator_model_}.png')
    rgbd_s_1_.save_png(f'{data_src_}_stereo_1_{disparity_estimator_model_}.png')
    gs_image_0_ = GSImage.from_rgbd_image(rgbd_s_0_, gs_regressor_model='gps', gs_regressor_checkpoint=gs_regressor_checkpoint_)
    gs_image_0_.save_png(f'{data_src_}_stereo_gs_0_{disparity_estimator_model_}.png')
    gs_image_0_.reproject_to(gs_image_0_).save_png(f'{data_src_}_stereo_gs_020_{disparity_estimator_model_}.png')
    gs_image_1_ = GSImage.from_rgbd_image(rgbd_s_1_, gs_regressor_model='gps', gs_regressor_checkpoint=gs_regressor_checkpoint_)
    gs_image_1_.save_png(f'{data_src_}_stereo_gs_1_{disparity_estimator_model_}.png')
    gs_image_1_.reproject_to(gs_image_1_).save_png(f'{data_src_}_stereo_gs_121_{disparity_estimator_model_}.png')
    stereo_gs_image_ = StereoGSImage.from_gs_images(gs_image_0_, gs_image_1_)
    stereo_gs_image_.unproject().save_ply(f'{data_src_}_stereo_gs_{disparity_estimator_model_}.ply')
    #   - save stereo GSImage
    stereo_gs_image_.save_png(f'{data_src_}_stereo_gs_{disparity_estimator_model_}.png')
    stereo_gs_image_.reproject_to(gs_image_0_).save_png(f'{data_src_}_stereo_gs_0120_{disparity_estimator_model_}.png')
    stereo_gs_image_.reproject_to(gs_image_1_).save_png(f'{data_src_}_stereo_gs_0121_{disparity_estimator_model_}.png')
