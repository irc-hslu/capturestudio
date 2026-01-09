import functools
import math
from pathlib import Path
from typing import Union, Literal, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
from diff_gauss import GaussianRasterizer, GaussianRasterizationSettings
from plyfile import PlyElement, PlyData
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_axis_angle

from utils.misc import log, PathUtils
from reconstruction.primitive.pcd import RGBDImage, PixelPoints
from utils.vis import VisUtils


class GSImage(RGBDImage):
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray, scale: np.ndarray, rotation: np.ndarray, opacity: np.ndarray, intrinsic: np.ndarray, extrinsic_w2c: np.ndarray, features: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize an RGBD image.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image of shape (H, W, C).
        mask : np.ndarray
            Valid pixels mask of shape (H, W), where >0 indicates valid pixels.
        depth : np.ndarray
            Depth map of shape (H, W). Depth values should be in meters.
        scale : np.ndarray
            Gaussian splat scale map of shape (H, W, 3).
        rotation : np.ndarray
            Gaussian splat rotation map of shape (H, W, 4) in quaternion format (WXYZ).
        opacity : np.ndarray
            Gaussian splat opacity map of shape (H, W).
        intrinsic : np.ndarray
            Camera intrinsic matrix of shape (3, 3).
        extrinsic_w2c : np.ndarray
            Camera extrinsic matrix of shape (4, 4).
        """
        super().__init__(rgb, mask, depth, intrinsic, extrinsic_w2c, features=(features if features is not None else {}) | dict(scale=scale, rotation=rotation, opacity=opacity))

    def __repr__(self) -> str:
        """
        String representation of the RGBD image.

        Returns
        -------
        str
            A string representation of the RGBD image.
        """
        return (f"{self.__class__.__name__}(\n"
                f"\trgb_shape={self.rgb.shape}, \n"
                f"\tdepth_shape={self.depth.shape}, \n"
                f"\tgs_scale_shape={self.features['scale'].shape}, \n"
                f"\tgs_rot_shape={self.features['rotation'].shape}, \n"
                f"\tgs_opacity_shape={self.features['opacity'].shape}, \n"
                f"\tintrinsic_shape={self.intrinsic.shape}, \n"
                f"\textrinsic_shape={self.extrinsic_w2c.shape}, \n"
                f"\tvalid_pixels_shape={self.mask.shape}\n"
                f")")

    @functools.cached_property
    def normals(self) -> np.ndarray:
        """
        Get the normals of the GS image. If not available, the normals are estimated using diff_gauss rasterizer by reprojecting the image to itself.

        Returns
        -------
        np.ndarray
            The normals of the RGBD image of shape (H, W, 3).
        """
        if 'normals' in self.features:
            normal_full = self.features['normals']
        else:
            normal_full = self.reproject_to(self, align=False, use_cache=True).features['normals'].copy()
            self.features['normals'] = normal_full
        normal_full[(normal_full == 0).all(-1)] = -1.0
        return normal_full

    @functools.cached_property
    def rotation_matrix(self) -> np.ndarray:
        """
        Get the rotation matrix of the Gaussian splats.

        Returns
        -------
        np.ndarray
            The rotation matrix of shape (H, W, 3, 3).
        """
        rot = quaternion_to_matrix(torch.from_numpy(self.features['rotation'])).numpy()
        return rot

    @functools.cached_property
    def rotation_matrix_camera(self) -> np.ndarray:
        """
        Get the rotation of the Gaussian splats in matrix format, in camera space.

        Returns
        -------
        np.ndarray
            The rotation matrix of shape (H, W, 3).
        """
        rot = quaternion_to_matrix(torch.from_numpy(self.features['rotation'])).numpy() @ self.extrinsic_w2c[:3, :3].T
        return rot

    @functools.cached_property
    def rotation_axis_angle_camera(self) -> np.ndarray:
        """
        Get the rotation of the Gaussian splats in matrix format, in camera space.

        Returns
        -------
        np.ndarray
            The rotation matrix of shape (H, W, 3).
        """
        rot = matrix_to_axis_angle(torch.from_numpy(quaternion_to_matrix(torch.from_numpy(self.features['rotation'])).numpy() @ self.extrinsic_w2c[:3, :3].T)).numpy()
        return rot

    @functools.cached_property
    def gs_cov3d(self) -> np.ndarray:
        """
        Get the 3D covariance matrix of the Gaussian splats.

        Returns
        -------
        np.ndarray
            The 3D covariance matrix of shape (H, W, 3, 3).
        """
        # quaternion -> rotation matrix
        rot = self.rotation_matrix  # (H, W, 3, 3)
        # 3D covariance: C=(RS)*(RS)^T
        s_sq = np.zeros((*self.features['scale'].shape, 3), dtype=self.features['scale'].dtype)
        for _ in range(3):
            s_sq[:, :, _, _] = self.features['scale'][:, :, _] ** 2
        cov3d = rot @ s_sq @ np.swapaxes(rot, -1, -2)
        return cov3d

    @functools.cached_property
    def cov_2d(self) -> np.ndarray:
        """
        Get the 2D covariance matrix of the Gaussian splats.

        Returns
        -------
        np.ndarray
            The 2D covariance matrix of shape (H, W, 2, 2).
        """
        H, W = self.image_size_hw

        # view parameters
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        fov_x = 2 * np.arctan(W / (2 * fx))
        fov_y = 2 * np.arctan(H / (2 * fy))
        tan_fovx, tan_fovy = np.tan(fov_x * 0.5), np.tan(fov_y * 0.5)

        # camera transform
        points_cam = self.points_camera.reshape(-1, 3)
        tx = (points_cam[:, 0] / points_cam[:, 2]).clip(-tan_fovx * 1.3, tan_fovx * 1.3) * points_cam[:, 2]
        ty = (points_cam[:, 1] / points_cam[:, 2]).clip(-tan_fovy * 1.3, tan_fovy * 1.3) * points_cam[:, 2]
        tz = points_cam[:, 2]

        # Jacobian
        J = np.zeros((H * W, 3, 3), dtype=points_cam.dtype)
        J[:, 0, 0] = fx / tz
        J[:, 0, 2] = -tx * fx / (tz * tz)
        J[:, 1, 1] = fy / tz
        J[:, 1, 2] = -ty * fy / (tz * tz)

        cov3d = self.gs_cov3d
        cov3d = cov3d.reshape(-1, 3, 3)
        r_w2c = self.extrinsic_w2c[:3, :3]
        w = r_w2c.T
        cov_2d = (J @ w @ cov3d @ w.T @ np.swapaxes(J, 1, 2))[:, :2, :2] + np.eye(2) * 0.3  # antialiasing (equivalent to 3x3 smoothing)
        return cov_2d.reshape(H, W, 2, 2)

    def unproject(self) -> 'PixelGSPoints':
        """
        Unproject the GS image to a PixelGSPoints object.

        Returns
        -------
        PixelGSPoints
            A new PixelGSPoints instance containing the points, colors, and valid pixels from the RGBDImage.
        """
        pcd = super().unproject()
        return PixelGSPoints.from_pcd(pcd, **self.features)

    def save_png(self, out_path: Optional[Union[Path, str]] = None, striped:bool=False, white_bg: bool = False) -> Optional[np.ndarray]:
        """
        Save the GS image as a PNG file.
        The file contains the RGB, mask, depth, and gs attributes, concatenated along the width:
          ____________________________________________________________
         | RGB | Mask | Depth | Normal | GS_normal | GS_splats | GS_α |
          ____________________________________________________________

        Parameters
        ----------
        out_path : Union[Path, str], optional
            The path where to save the PNG file. If None, the image will be returned as a numpy array.
        striped : bool, optional
            If True, the image will be saved with a striped pattern for better visibility of the features.
            Default is False, i.e. no stripes are added.
        white_bg : bool, optional
            If True, the background of the image will be set to white. Default is False, i.e. the background will be black.

        Returns
        -------
        Optional[np.ndarray]
            If out_path is None, the concatenated image will be returned as a numpy array.
            Otherwise, None is returned after saving the image to the specified path.
        """
        # RGB, mask, depth, normals
        rgbd_image = super().save_png(out_path=None, striped=False)  # [:, :-self.image_size_hw[1] ,:]
        # GS normals
        gs_normal_img = self.visualize_features(feat='normals')
        gs_normal_img_bgr = cv2.cvtColor(gs_normal_img, cv2.COLOR_RGB2BGR)
        # GS splats and opacity
        gs_cov_op_img = self.visualize_features(feat='cov+opacity')
        gs_cov_op_img_bgr = cv2.cvtColor(gs_cov_op_img, cv2.COLOR_RGB2BGR)
        # create the final image
        if striped:
            png_image = VisUtils.striped_teaser_image(rgbd_image[:, :self.image_size_hw[1]] * self.mask[..., None], gs_cov_op_img_bgr[:, :self.image_size_hw[1]], gs_cov_op_img_bgr[:, self.image_size_hw[1]:], spacing=0.0, widths=(0.075, 0.075), order=('normals', 'depth'))
            png_image[~self.mask & (png_image[..., 0] == 0.0)] = 255.0  # white background
        else:
            png_image = np.concatenate([rgbd_image, gs_normal_img_bgr, gs_cov_op_img_bgr], axis=1)
            if white_bg:
                # repeat the mask N times, to fill the width
                if png_image.shape[1] > self.mask.shape[1]:
                    png_mask = np.tile(self.mask, (1, png_image.shape[1] // self.mask.shape[1]))
                else:
                    png_mask = self.mask
                png_image[~png_mask] = 255.0  # white background
        if out_path is None:
            return png_image
        cv2.imwrite(str(out_path), png_image)
        log(f'[{self.__class__.__name__}::save_png] GSImage saved to {out_path}', 'debug')
        return None

    def visualize_features(self, feat: Literal['normals', 'cov', 'opacity', 'cov+opacity'] = 'normals') -> np.ndarray:
        """
        Visualize the features of the RGBD image.

        Parameters
        ----------
        feat : Literal['cov', 'opacity', 'cov+opacity']
            The feature to visualize. Supported features are:
            - 'normals': Visualize the normals of the Gaussian splats.
            - 'cov': Visualize the 2D covariance (i.e. the projected splats covariance, in terms of size and orientation) of the Gaussian splats.
            - 'opacity': Visualize the opacity of the Gaussian splats.
            - 'cov+opacity': Visualize both the 2D covariance and the opacity of the Gaussian splats, overlaying the covariance visualization and mixing based on the opacity.

        Returns
        -------
        np.ndarray
            The visualized feature as a numpy array.
        """
        # GS normals
        if feat == 'normals':
            normal_map_gs = self.normals
            normal_gs_img = ((normal_map_gs + 1.0) / 2.0 * 255 + 0.5).astype(np.uint8)
            return normal_gs_img

        # GS splats and opacity
        gs_cov_2d = self.cov_2d  # (H, W, 4)
        gs_opacity = self.features['opacity']  # (H, W)
        H, W = self.image_size_hw
        # 2‑D covariance -> (n,2,2)
        cov = gs_cov_2d.reshape(-1, 4)
        cov = np.stack([cov[:, 0], cov[:, 1], cov[:, 2], cov[:, 3]], 1).reshape(-1, 2, 2)
        # eigen‑decomposition
        eigval, eigvec = np.linalg.eigh(cov)
        axes_len = 6 * np.sqrt(np.clip(eigval, 0, None))  # 3 stds each side
        ang = np.degrees(np.arctan2(eigvec[:, 1, 0], eigvec[:, 0, 0]))
        # plot the splats
        color = self.rgb.reshape(-1, 3)
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        u, v = uu.reshape(-1), vv.reshape(-1)

        # --- build a flat valid index list -----------------------------------
        axes_len_flat = axes_len.reshape(-1, 2)
        valid_flat = (
                self.mask.flatten()
                & (gs_opacity.flatten() > 0)
                & np.all(np.isfinite(axes_len_flat), axis=-1)
                & (axes_len_flat[:, 0] > 0)
                & (axes_len_flat[:, 1] > 0)
        )
        valid_idx = np.flatnonzero(valid_flat)

        # --- random subsample (≈ 1/20 density), reproducible -----------------
        keep_ratio = 1.0 / 50.0
        rng = np.random.default_rng(42)
        # Option A: Bernoulli thinning (fast, approximate count)
        sel = valid_idx[rng.random(valid_idx.size) < keep_ratio]
        # Option B: exact count (uncomment to use)
        # K = max(1, int(np.ceil(keep_ratio * valid_idx.size)))
        # sel = rng.choice(valid_idx, size=K, replace=False)

        gs_cov_img = np.zeros((*self.image_size_hw, 3), dtype=np.uint8)  # black canvas
        gs_op_img = np.zeros((*self.image_size_hw, 3), dtype=np.uint8)  # black canvas
        col_max_opacity = np.array([255, 255, 255], dtype=np.uint8)
        col_min_opacity = np.array([1, 1, 1], dtype=np.uint8)
        valid = (self.mask & (gs_opacity > 0) & np.all(np.isfinite(axes_len.reshape((H, W, 2))), axis=-1))
        op_max = gs_opacity[valid].max()
        op_min = gs_opacity[valid].min()
        opacities = np.ascontiguousarray(gs_opacity.reshape(-1))
        for i in sel:
            w_px, h_px = axes_len[i]
            c = (int(u[i]), int(v[i]))
            axes_len_i = (int(w_px * 1.0), int(h_px * 1.0))
            cv2.ellipse(gs_cov_img, c, axes_len_i, ang[i], 0, 360, tuple(color[i].tolist()), -1, cv2.LINE_AA)
            op_i = opacities[i]
            op_alpha = np.clip((op_i - op_min) / max(op_max - op_min, 1e-8), 0.0, 1.0)
            col_i = (col_min_opacity * (1.0 - op_alpha) + col_max_opacity * op_alpha).astype(np.uint8)
            cv2.ellipse(gs_op_img, c, axes_len_i, ang[i], 0, 360, tuple(col_i.tolist()), -1, cv2.LINE_AA)
        if feat == 'cov':
            return gs_cov_img
        if feat == 'opacity':
            return gs_op_img
        if feat == 'cov+opacity':
            return np.concatenate((gs_cov_img, gs_op_img), axis=1)
        raise ValueError(f"Unsupported feature for visualization: {feat}. Supported features are: 'normals', 'cov', 'opacity', 'cov+opacity'.")

    @classmethod
    def from_rgbd_image(cls, rgbd_image: RGBDImage, gs_regressor_model: Literal['gps'] = 'gps', gs_regressor_checkpoint: Union[Path, str] = 'neptune://85/best') -> 'GSImage':
        """
        Create GSImage from an RGBDImage.

        Parameters
        ----------
        rgbd_image : RGBDImage
            The RGBDImage to convert.
        gs_regressor_model : Literal['gps']
            The Gaussian splat regressor model to use. Currently only 'gps' is supported, which uses the GSRegressor network from GPSGaussian model.
        gs_regressor_checkpoint : Union[Path, str]
            Path to the checkpoint file for the Gaussian splat regressor model.
            If it starts with neptune://, it will be treated as a Neptune run ID and the checkpoint will be downloaded from there. E.g. "neptune://<project_name>/<run_id:int or str>/<neptune_path.ckpt>", or "neptune://<run_idx:int>/best".
            See `utils.npt.NeptuneUtils` for more details on what is supported.

        Returns
        -------
        PixelPoints
            A new PixelPoints instance containing the points, colors, and valid pixels from the RGBDImage.
        """
        # initialize gaussian splat regressor
        if gs_regressor_model == 'gps':
            gs_regressor = GSUtils.load_gps(gs_regressor_checkpoint, load_raft=False)
        else:
            raise ValueError(f"Unsupported GS regressor model: {gs_regressor_model}. Currently only 'gps' is supported.")
        # estimate gaussian params
        device = next(iter(gs_regressor.parameters())).device
        gs_rot, gs_scale, gs_opacity = gs_regressor.forward(
            image=torch.tensor(rgbd_image.rgb.astype(np.float32) / 255.0, device=device).permute(2, 0, 1).unsqueeze(0).unsqueeze(0),  # (1,1,3,H,W)
            mask=torch.tensor(rgbd_image.mask.astype(np.float32), device=device).unsqueeze(0).unsqueeze(1).unsqueeze(0),  # (1,1,1,H,W)
            intrinsic=torch.tensor(rgbd_image.intrinsic.astype(np.float32), device=device).unsqueeze(0).unsqueeze(0),  # (1,1,3,3)
            extrinsic=torch.tensor(np.linalg.inv(rgbd_image.extrinsic_w2c.astype(np.float32)), device=device).unsqueeze(0).unsqueeze(0),  # (1,1,4,4), in c2w format
            depth=torch.tensor(rgbd_image.depth.astype(np.float32), device=device).unsqueeze(0).unsqueeze(1).unsqueeze(0),  # (1,1,1,H,W)
            camera_idx=torch.tensor([0], device=device).long().unsqueeze(0),  # (1,1)
        )[-3:]
        # create GSImage
        return cls(
            rgb=rgbd_image.rgb,
            mask=rgbd_image.mask,
            depth=rgbd_image.depth,
            scale=gs_scale.detach().cpu().squeeze().permute(1, 2, 0).numpy(),  # (H,W,3)
            rotation=gs_rot.detach().cpu().squeeze().permute(1, 2, 0).numpy(),  # (H,W,4)
            opacity=gs_opacity.detach().cpu().squeeze().numpy(),  # (H,W)
            intrinsic=rgbd_image.intrinsic,
            extrinsic_w2c=rgbd_image.extrinsic_w2c,
            features=rgbd_image.features,  # copy existing features
        )


class PixelGSPoints(PixelPoints):
    def __init__(self, pixel_points: np.ndarray, pixel_colors: np.ndarray, pixel_valid: np.ndarray, pixel_features: Dict[str, np.ndarray], **extra_features):
        """
        Initialize a PixelGSPoints instance.

        Parameters
        ----------
        pixel_points : np.ndarray
            Points in the image of shape (H, W, 3).
        pixel_colors : np.ndarray
            Colors of the points in the image of shape (H, W, 3).
        pixel_valid : np.ndarray
            Valid pixels mask of shape (H, W), where >0 indicates valid pixels.
        pixel_features : Dict[str, np.ndarray]
            Additional features for Gaussian splats, including 'scale', 'rotation', and 'opacity'.
            - 'scale': Gaussian splat scale map of shape (H, W, 3).
            - 'rotation': Gaussian splat rotation map of shape (H, W, 4) in quaternion format (WXYZ).
            - 'opacity': Gaussian splat opacity map of shape (H, W).
        """
        assert all([_ in pixel_features and isinstance(pixel_features[_], np.ndarray) for _ in ['scale', 'rotation', 'opacity']]), \
            "PixelGSPoints requires 'scale', 'rotation', and 'opacity' features in pixel_features."
        super().__init__(pixel_points, pixel_colors, pixel_valid, pixel_features=pixel_features | extra_features)

    @classmethod
    def from_pcd(cls, pcd: PixelPoints, scale: np.ndarray, rotation: np.ndarray, opacity: np.ndarray, **extra_features) -> 'PixelGSPoints':
        """
        Create PixelGSPoints from a PixelPoints instance and additional Gaussian splat parameters.

        Parameters
        ----------
        pcd : PixelPoints
            The PixelPoints instance to convert.
        scale : np.ndarray
            Gaussian splat scale map of shape (H, W, 3).
        rotation : np.ndarray
            Gaussian splat rotation map of shape (H, W, 4) in quaternion format (WXYZ).
        opacity : np.ndarray
            Gaussian splat opacity map of shape (H, W).
        extra_features : Dict[str, np.ndarray]
            Additional pixel-wise features to include in the PixelGSPoints instance.

        Returns
        -------
        PixelGSPoints
            A new PixelGSPoints instance containing the points, colors, valid pixels, and Gaussian splat parameters.
        """
        return cls(
            pixel_points=pcd.points,
            pixel_colors=pcd.colors,
            pixel_valid=pcd.valid,
            pixel_features=dict(
                scale=scale,
                rotation=rotation,
                opacity=opacity
            ) | extra_features,
        )

    @functools.cached_property
    def as_3dgs(self) -> Dict[str, torch.Tensor]:
        device = 'cuda'
        return dict(
            means3D=torch.from_numpy(self.points[self.valid]).to(device=device, dtype=torch.float32),
            colors_precomp=torch.from_numpy(self.colors[self.valid]).to(device=device, dtype=torch.float32),
            rotations=torch.from_numpy(self.features['rotation'][self.valid]).to(device=device, dtype=torch.float32),
            scales=torch.from_numpy(self.features['scale'][self.valid]).to(device=device, dtype=torch.float32),
            opacities=torch.from_numpy(self.features['opacity'][self.valid]).to(device=device, dtype=torch.float32)
        )

    # noinspection PyMethodOverriding
    def project(self, target_intrinsic: np.ndarray, target_extrinsic: np.ndarray, target_image_size_hw: Tuple[int, int], is_c2w: bool, point_size: float = 1.0, use_cache: bool = False, rasterizer=None) -> 'RGBDImage':
        """
        Project the pixel points into an GSImage using the target intrinsic and extrinsic matrices.

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
            This is not used in the 3DGS-based rasterization.
        use_cache : bool, optional
            If True, use cached renderer for the target intrinsics and extrinsics. Default is False.
        rasterizer : Optional[GaussianRasterizer]
            An optional renderer to use. If None, a new renderer will be created.

        Returns
        -------
        RGBDImage
            A new RGBDImage instance containing the projected points and colors.
        """
        # get renderer
        if rasterizer is None:
            if not use_cache:
                rasterizer = GSUtils.create_rasterizer(target_intrinsic, target_extrinsic, target_image_size_hw, is_c2w=is_c2w)
            else:
                cache_key = (target_intrinsic.tobytes(), target_extrinsic.tobytes(), target_image_size_hw, is_c2w)
                if cache_key in self.__class__.O3D_VISUALIZER_CACHE:
                    rasterizer = self.__class__.O3D_VISUALIZER_CACHE[cache_key]
                else:
                    rasterizer = GSUtils.create_rasterizer(target_intrinsic, target_extrinsic, target_image_size_hw, is_c2w=is_c2w)
                    self.__class__.O3D_VISUALIZER_CACHE[cache_key] = rasterizer

        # render the gaussian pointcloud
        pcd_3dgs = self.as_3dgs
        render_rgb, render_depth, render_norm, render_opacity = rasterizer(means2D=torch.zeros_like(pcd_3dgs['means3D'], requires_grad=True), **pcd_3dgs)[:4]

        # create GSImage
        render_valid = (render_opacity > 0.3).detach().cpu().squeeze().numpy()
        render_depth = render_depth.detach().cpu().squeeze().numpy()
        render_depth[~render_valid] = float('nan')
        return RGBDImage(
            np.ascontiguousarray(render_rgb.detach().cpu().squeeze_().mul_(255).add_(0.5).byte().permute(1, 2, 0).numpy()),
            np.ascontiguousarray(render_valid),
            np.ascontiguousarray(render_depth),
            features=dict(
                normals=render_norm.detach().cpu().squeeze().permute(1, 2, 0).numpy(),  # (H, W, 3)
                opacity=render_opacity.detach().cpu().squeeze().numpy(),  # (H, W)
            ),
            intrinsic=target_intrinsic,
            extrinsic_w2c=(target_extrinsic if not is_c2w else np.linalg.inv(target_extrinsic)),
        )

    def save_ply(self, out_path: Union[Path, str]) -> None:
        """
        Save the gaussian pixel points to a PLY file (compatible with SuperSplat).

        Parameters
        ----------
        out_path : Path
            The path where to save the PLY file.
        """
        xyz = self.points[self.valid]
        normals = np.zeros_like(xyz)
        rgb = self.colors[self.valid]
        f_dc = (rgb - 0.5) / 0.28209479177387814
        f_rest = np.zeros((rgb.shape[0], 3 * 15))
        rotation = self.features['rotation'][self.valid]
        scale = self.features['scale'][self.valid]
        opacity = self.features['opacity'][self.valid].reshape(-1, 1)
        sh_degree = 3

        # construct_list_of_attributes
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(np.prod(f_dc.shape[1:])):
            l.append('f_dc_{}'.format(i))
        for i in range(int((sh_degree + 1) ** 2 - 1) * 3):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        elements = np.empty(xyz.shape[0], dtype=[(attribute, 'f4') for attribute in l])
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacity, np.log(scale), rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        # noinspection PyTypeChecker
        el = PlyElement.describe(elements, 'vertex')
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        PlyData([el]).write(str(out_path))
        log(f'[{self.__class__.__name__}::save_ply] Gaussian point cloud saved to {out_path}', 'debug')


class GSUtils:
    GPS_MODEL_CACHE = {}

    @classmethod
    def focal2fov(cls, focal: float, pixels: int) -> float:
        return 2 * math.atan(pixels / (2 * focal))

    @classmethod
    def get_projection_matrix_3dgs(cls, intrinsic: np.ndarray, image_size_hw: Tuple[int, int], znear: float = 0.01, zfar: float = 100.0) -> np.ndarray:
        """
        OpenGL‐style perspective matrix that honours principal‐point offsets.
        Pixel origin is (0,0)=top‐left; NDC is x∈[‑1,1] left→right, y∈[‑1,1] top→bottom, z∈[‑1,1] near→far.

        Parameters
        ----------
        intrinsic : np.ndarray
            Camera intrinsic matrix of shape (3, 3).
        image_size_hw : Tuple[int, int]
            The height and width of the image.
        znear : float, optional
            The near clipping plane distance. Default is 0.01.
        zfar : float, optional
            The far clipping plane distance. Default is 1000.0.

        Returns
        -------
        np.ndarray
            The projection matrix of shape (4, 4).
        """
        H, W = image_size_hw
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, :3] = 2 * intrinsic[0, :] / W
        proj[1, :3] = 2 * intrinsic[1, :] / H
        proj[:2, 2] -= 1
        proj[2, 2] = zfar / (zfar - znear)
        proj[2, 3] = -zfar * znear / (zfar - znear)
        proj[3, 2] = 1.0
        return proj

    @classmethod
    def create_rasterizer(cls, intrinsic: np.ndarray, extrinsic: np.ndarray, image_size_hw: Tuple[int, int], is_c2w: bool = False, device: Union[torch.device, str] = 'cuda') -> 'GaussianRasterizer':
        """
        Create a Gaussian rasterizer for the given intrinsic and extrinsic matrices.

        Parameters
        ----------
        intrinsic : np.ndarray
            Camera intrinsic matrix of shape (3, 3).
        extrinsic : np.ndarray
            Camera extrinsic matrix of shape (4, 4).
            If is_c2w is True, this is the camera-to-world (R,T) matrix.
        image_size_hw : Tuple[int, int]
            The height and width of the image.
        is_c2w : bool, optional
            If True, the extrinsic is assumed to be camera-to-world (R,T). Default is True.
        device : Union[str, torch.device], optional
            Device to create the rasterizer on. Default is 'cuda', which will use the default CUDA GPU.

        Returns
        -------
        GaussianRasterizer
            A new Gaussian rasterizer instance.
        """
        w2c_matrix = torch.from_numpy(extrinsic if not is_c2w else np.linalg.inv(extrinsic)).to(device=device, dtype=torch.float32)
        proj_matrix = torch.from_numpy(cls.get_projection_matrix_3dgs(intrinsic, image_size_hw)).to(device=device, dtype=torch.float32)
        return GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
            image_height=image_size_hw[0],
            image_width=image_size_hw[1],
            tanfovx=math.tan(cls.focal2fov(intrinsic[0, 0].item(), image_size_hw[1]) * 0.5),
            tanfovy=math.tan(cls.focal2fov(intrinsic[1, 1].item(), image_size_hw[0]) * 0.5),
            bg=torch.tensor((0.0, 0.0, 0.0), device=device, dtype=torch.float32),
            scale_modifier=1.0,
            viewmatrix=w2c_matrix.T,
            projmatrix=(proj_matrix @ w2c_matrix).T,
            sh_degree=0,
            campos=w2c_matrix.inverse()[:3, 3],
            prefiltered=False,
            debug=False
        ))

    # noinspection PyUnresolvedReferences
    @classmethod
    def load_gps(cls, checkpoint_path: Union[Path, str], load_raft: bool = False, device: Union[str, torch.device] = 'cuda') -> 'GPS':
        """
        Load the GPS Gaussian model from a checkpoint.

        Parameters
        ----------
        checkpoint_path : Union[Path, str]
            Path to the checkpoint file for the Gaussian splat regressor model.
            If it starts with neptune://, it will be treated as a Neptune run ID and the checkpoint will be downloaded from there. E.g. "neptune://<project_name>/<run_id:int or str>/<neptune_path.ckpt>", or "neptune://<run_idx:int>/best".
            See `utils.npt.NeptuneUtils` for more details on what is supported.
        load_raft : bool, optional
            Whether to load the RAFT stereo model for depth estimation. Default is False, which means that sensor depth will be used.
        device : Union[str, torch.device], optional
            Device to load the model on. Default is 'cuda', which will use the GPU.

        Returns
        -------
        PixelPoints
            A new PixelPoints instance containing the points, colors, and valid pixels from the RGBDImage.
        """
        cache_key = Path(checkpoint_path).stem + f"{'-raft' if load_raft else ''}"
        if cache_key in cls.GPS_MODEL_CACHE:
            return cls.GPS_MODEL_CACHE[cache_key]

        # get model checkpoint
        if isinstance(checkpoint_path, str) and checkpoint_path.startswith('neptune://'):
            from utils.npt import NeptuneUtils
            checkpoint_path = NeptuneUtils.decode_checkpoint_path(checkpoint_path, overwrite=False)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            # try inside the checkpoints directory
            checkpoint_path = PathUtils.checkpoints_path() / str(checkpoint_path)
        assert checkpoint_path.exists(), f"Checkpoint file {checkpoint_path} does not exist."
        if ('best' in checkpoint_path.stem and 'RGBD2GS'.lower() not in checkpoint_path.stem.lower()) and int(checkpoint_path.stem.split('-')[-2]) > 21:
            # sft runs
            run_id = int(checkpoint_path.stem.split('-')[-2])
            scale_head_tanh = run_id in [151, 153]
            scale_head_norm = run_id >= 154
            if run_id == 123:
                rot_bug_fix_version = 'simone'
            elif 121 <= run_id <= 154:
                rot_bug_fix_version = 'thanos'
            elif run_id >= 155:
                rot_bug_fix_version = 'correct'
            else:
                rot_bug_fix_version = 'none'
        else:
            # trained with original intrinsics, so fix is needed
            scale_head_tanh = False
            scale_head_norm = False
            if 'RGBD2GS'.lower() in checkpoint_path.stem.lower():
                # Simone's checkpoints
                # intrinsics_fix = True
                rot_bug_fix_version = 'simone'
            else:
                rot_bug_fix_version = 'none'
        if 'RGBD2GS'.lower() in checkpoint_path.stem.lower():
            from sft.gps_impl import GPSLightningModule
            sd = GPSLightningModule.convert_ckpt(checkpoint_path, checkpoint_path, overwrite=True, return_sd=True)
        else:
            sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # load model
        from sft.gps_impl import GPS
        gps_model = GPS(use_raft_stereo=load_raft, do_render=False, scale_head_tanh=scale_head_tanh, scale_head_norm=scale_head_norm, rot_bug_fix_version=rot_bug_fix_version)
        gps_model.load_state_dict(sd)
        log(f'[{cls.__name__}::load_gps] Loaded GPS model from {checkpoint_path} with parameters: scale_head_tanh={scale_head_tanh}, scale_head_norm={scale_head_norm}, rot_bug_fix_version={rot_bug_fix_version}', 'info')
        gps_model = gps_model.eval().to(device=device)
        cls.GPS_MODEL_CACHE[cache_key] = gps_model
        return gps_model


if __name__ == '__main__':
    from utils.calib import CalibrationData

    data_src_ = 'session'  # 'session' or 'thuman'

    if data_src_ == 'session':
        # read session data
        session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025' / 'Thanos_2_Perf_1'
        calibration_session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025' / 'Thanos_2_Calib_1'
        calibration_data_ = CalibrationData.from_session(calibration_session_root_)
        #   - read 2 RGBDImages
        rgbd_l_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=8, color_ts=1746110341432, depth_ts=1746110341433).resize(1024, 1280 if data_src_ == 'session' else 1024)
        rgbd_r_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=7, color_ts=1746110341432, depth_ts=1746110341433).resize(1024, 1280 if data_src_ == 'session' else 1024)
    else:
        # read THUMAN data
        thuman_root_ = Path('/media/charisoudis/nas_transmixr/Simone/Volumetric_Video/Human Datasets/THuman2_1/rendered@2m')
        #   - read 2 RGBDImages
        rgbd_l_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=2)
        rgbd_r_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=3)

    # create GSImages
    gs_l_ = GSImage.from_rgbd_image(rgbd_l_, gs_regressor_model='gps', gs_regressor_checkpoint='neptune://154/best')
    gs_l_.save_png(f'{data_src_}_gs_l.png', striped=True)
    exit(0)
    gs_r_ = GSImage.from_rgbd_image(rgbd_r_, gs_regressor_model='gps', gs_regressor_checkpoint='neptune://154/best')
    gs_r_.save_png(f'{data_src_}_gs_r.png')
    # reproject GSImages to each other
    with torch.no_grad():
        gs_l_.reproject_to(gs_l_, align=False, use_cache=True).save_png(f'{data_src_}_gs_l2l.png')
        gs_l_.reproject_to(gs_r_, align=False, use_cache=True).save_png(f'{data_src_}_gs_l2r.png')
        gs_r_.reproject_to(gs_l_, align=False, use_cache=True).save_png(f'{data_src_}_gs_r2l.png')
        gs_r_.reproject_to(gs_r_, align=False, use_cache=True).save_png(f'{data_src_}_gs_r2r.png')
    # create point clouds from GSImages
    gs_pcd_l_ = gs_l_.unproject()
    gs_pcd_r_ = gs_r_.unproject()
    gs_pcd_l_.save_ply(f'{data_src_}_l_gs.ply')
    gs_pcd_r_.save_ply(f'{data_src_}_r_gs.ply')
    # stitch left and right point clouds
    gs_pcd_lr_ = PixelGSPoints.from_partials(gs_pcd_l_, gs_pcd_r_)
    gs_pcd_lr_.save_ply(f'{data_src_}_lr_gs.ply')
    gs_pcd_lr_.project(gs_l_.intrinsic, gs_l_.extrinsic_w2c, gs_l_.image_size_hw, use_cache=True, is_c2w=False, point_size=2.0).save_png(f'{data_src_}_lr2l_gs.png')
    gs_pcd_lr_.project(gs_r_.intrinsic, gs_r_.extrinsic_w2c, gs_r_.image_size_hw, use_cache=True, is_c2w=False, point_size=2.0).save_png(f'{data_src_}_lr2r_gs.png')
