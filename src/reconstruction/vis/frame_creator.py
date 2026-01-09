import abc
from pathlib import Path
from typing import Literal, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reconstruction.primitive.pcd import RGBDImage, PixelPoints
from reconstruction.primitive.splat import GSImage
from reconstruction.primitive.stereo import StereoImage
from utils.misc import PathUtils


class FrameCreator(metaclass=abc.ABCMeta):
    """
    Base class for creating frames in a grid layout.
    This class is intended to be subclassed for specific frame layouts.

    In all frame creator subclasses, the focus camera is always the main camera, and the satellite cameras are the projections to the ground truth cameras.
    The input to the forward pass is:
        - rgbd images of all gt cameras
        - camera intrinsics and extrinsics for all gt cameras
        - pcd or gs reconstruction
        - camera intrinsics and extrinsics for the main (virtual) camera

    The output is a grid of images, as a torch.tensor of shape (C, H, W), where W is the width of the main camera, H is the height of the main camera + height of satellite cameras, and C is the number of channels.
    The supported modes are:
        - 'rgb_gt|rgb_gt': main camera shows selected gt cam image, satellite cameras show gt rgb images.
        - 'rgb_recon|rgb_gt': main camera shows projected reconstruction on the virtual camera, satellite cameras show gt rgb images.
        - 'rgb_recon|rgb_recon': main camera shows projected reconstruction on the virtual camera, satellite cameras show projections on the ground truth cameras.
        - 'depth_gt|depth_gt': main camera shows projected reconstruction's depth on the virtual camera, satellite cameras show gt depth images.
        - 'depth_recon|depth_recon': main camera shows projected reconstruction's depth on the virtual camera, satellite cameras show projections' depths on the ground truth cameras.
        - 'feat_recon|feat_recon': main camera shows projected reconstruction's primitives on the virtual camera, satellite cameras show projections' primitives on the ground truth cameras. For GS, this corresponds to visualizing the Gaussians as 3D balls, for PCD it corresponds to visualizing the point cloud as spheres.
    """

    def __init__(self, mode: str = 'rgb_recon|rgb_gt', evaluator: Optional = None):
        """
        Initialize the FrameCreator with a specific mode.

        Parameters
        ----------
        mode : str, optional
            The mode of the frame creation. Defaults to 'rgb_recon|rgb_gt'.
        """
        from reconstruction.eval.metrics import MetricsAggregator
        self.evaluator: Optional[MetricsAggregator] = evaluator
        self.mode = mode
        self.main_mode, self.satellite_mode = mode.split('|', 1)
        self.main_mode_modality, self.main_mode_src = self.main_mode.split('_', 1)
        self.satellite_mode_modality, self.satellite_mode_src = self.satellite_mode.split('_', 1)
        if self.satellite_mode_src == 'recon':
            assert self.main_mode_src == 'recon', "Satellite mode can only be '*_recon' if the main mode is also '*_recon'."

    @abc.abstractmethod
    def create_frame(self, main_image: np.ndarray, satellite_images: List[np.ndarray], satellite_text: List[Union[str, None]], highlight_idx: Optional[List[int]] = None, is_stereo: bool = False) -> np.ndarray:
        """
        Create a frame grid from the provided images.

        Parameters
        ----------
        main_image : np.ndarray
            The main camera image.
        satellite_images : List[np.ndarray]
            List of satellite camera images.
        satellite_text : List[Union[str, None]]
            List of text labels for the satellite images. If None, no text will be displayed.
        highlight_idx : Optional[List[int]], optional
            Indices of the satellite cameras to highlight. If provided, a lightgreen border will be drawn around the corresponding satellite images.
        is_stereo : bool, optional
            If True, the images are stereo images and the left camera's data have be used. In that case we need to make sure that both source images (i, i+1) are highlighted. Defaults to False.

        Returns
        -------
        np.ndarray
            The created frame grid as a numpy array.
        """
        raise NotImplementedError

    @staticmethod
    def _visualize_modality(image: RGBDImage, modality: Literal['rgb', 'depth', 'feat']) -> np.ndarray:
        """
        Visualize the image based on the specified modality.

        Parameters
        ----------
        image : RGBDImage
            The image to visualize.
        modality : str
            The modality to visualize. This can be 'rgb', 'depth', or 'feat'.

        Returns
        -------
        RGBDImage
            The visualized image, which is a numpy array of shape (H, W, 3).
        """
        if modality == 'rgb':
            out = image.rgb
        elif modality == 'depth':
            depth_vis = ((image.depth.clip(1.0, 3.5) - 1.0) / 2.5 * 255).astype(np.uint8)
            depth_in_mask = depth_vis[(depth_vis > 0)]
            depth_in_mask = 255 - depth_in_mask
            depth_vis[(depth_vis > 0)] = depth_in_mask
            depth_image = cv2.cvtColor(cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
            out = depth_image
        elif modality.startswith('feat'):
            out = cv2.cvtColor(image.visualize_features(modality.split(':', 1)[-1]) if ':' in modality else image.visualize_features(), cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported modality: {modality}. Supported modalities are 'rgb', 'depth', and 'feat'.")

        # if the image is a stereo image, we only return the data corresponding to the left camera
        if isinstance(image, StereoImage):
            out = out[:out.shape[0] // 2, :, :]
        return out

    @staticmethod
    def _draw_text_top_right(image: np.ndarray, text: str, font: Union[Path, str] = 'JetBrainsMono-Regular.ttf', font_size: int = 16, color=(0, 255, 0), padding: int = 10) -> None:
        """
        Draws text near the top-right corner of a NumPy image using a custom TrueType font.

        Parameters
        ----------
        image : np.ndarray
            The image on which to draw the text.
        text : str
            The text to draw.
        font : str
            Path to the TrueType font file or a font name. Defaults to 'JetBrainsMono-Regular.ttf' which is found inside resources/fonts dir.
        font_size : int, optional
            Size of the font. Defaults to 20.
        color : tuple, optional
            Color of the text in RGB format. Defaults to white (255, 255, 255).
        padding : int, optional
            Padding from the edges of the image. Defaults to 10.

        Returns
        -------
        None
        """
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(str(PathUtils.font(font)), font_size)
        tl_x, _, br_x, _ = draw.textbbox((0,0), text, font=font)
        x = image_pil.width - (br_x - tl_x) - padding
        y = padding
        draw.text((x, y), text, font=font, fill=color)
        image[:] = np.asarray(image_pil)

    def forward(self, gt_images: List[RGBDImage], virtual_intrinsic: np.ndarray, virtual_extrinsic: np.ndarray, assignment: List[int], is_c2w: bool, virtual_image_size_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Forward pass to create a frame grid from the provided images.

        Parameters
        ----------
        gt_images : RGBDImage
            The ground truth images containing RGB and depth data.
        virtual_intrinsic : np.ndarray
            The intrinsic parameters of the virtual camera (3,3).
        virtual_extrinsic : np.ndarray
            The extrinsic parameters of the virtual camera (4,4).
        is_c2w : bool, optional
            If True, the provided extrinsics are camera-to-world (R,T). Defaults to False, meaning the extrinsics are world-to-camera.
        assignment : List[int]
            A list of indices of the ground truth cameras that should be used to create the main camera's frame.
        virtual_image_size_hw : Optional[Tuple[int, int]], optional
            The height and width of the virtual camera's image. If not provided, it defaults to the size of the ground truth images.

        Returns
        -------
        np.ndarray
            The created frame grid as a numpy array of shape (H, W, 3).
        """
        is_stereo = len(gt_images) >= 2 and isinstance(gt_images[0], StereoImage)

        # create main camera frame
        if self.main_mode_src == 'gt':
            assert len(assignment) == 1, "For rgb_gt mode, only one ground truth camera can be assigned to the main camera."
            main_image = gt_images[assignment[0]]
        elif self.main_mode_src == 'recon':
            # create virtual image by stitching the ground truth PCDs and projecting them to the virtual camera
            gt_partials = []
            for gt_idx in assignment:
                gt_image = gt_images[gt_idx]
                gt_partial = gt_image.unproject()
                gt_partials.append(gt_partial)
            main_pcd = gt_partials[0].__class__.from_partials(*gt_partials)
            main_image = main_pcd.project(virtual_intrinsic, virtual_extrinsic, virtual_image_size_hw if virtual_image_size_hw is not None else gt_images[0].image_size_hw, is_c2w=is_c2w, point_size=2.0, use_cache=True)
        else:
            raise ValueError(f"Unsupported main mode source: {self.main_mode_src}. Supported sources are '*_gt' and '*_recon'.")
        main_frame = self._visualize_modality(main_image, self.main_mode_modality)

        # create satellite camera frames
        satellite_frames = []
        satellite_texts = []
        for i, gt_image in enumerate(gt_images):
            if self.satellite_mode_src == 'gt':
                satellite_image = gt_image
            elif self.satellite_mode_src == 'recon':
                # project the reconstruction to the ground truth cameras
                # noinspection PyUnboundLocalVariable
                if isinstance(gt_image, StereoImage):
                    gt_image = gt_image.split_lr()[0]
                satellite_image = main_pcd.project(gt_image.intrinsic, gt_image.extrinsic_w2c, gt_image.image_size_hw, use_cache=True, is_c2w=False, point_size=2.0)
            else:
                raise ValueError(f"Unsupported satellite mode source: {self.satellite_mode_src}. Supported sources are '*_gt' and '*_recon'.")
            satellite_frame = self._visualize_modality(satellite_image, self.satellite_mode_modality)
            satellite_frames.append(satellite_frame)
            # metrics
            if self.main_mode_modality == self.satellite_mode_modality == 'rgb' and assignment is not None and self.evaluator is not None and (i in assignment or (is_stereo and (i - 1) in assignment)):
                if isinstance(gt_image, StereoImage):
                    gt_image = gt_image.split_lr()[0]
                projected_image = satellite_image if self.satellite_mode_src == 'recon' else main_pcd.project(gt_image.intrinsic, gt_image.extrinsic_w2c, gt_image.image_size_hw, use_cache=True, is_c2w=False, point_size=2.0)
                psnr = self.evaluator(gt_image, projected_image, group=i)['PSNR']
                satellite_texts.append(f"{int(psnr):02d}dB")
            else:
                satellite_texts.append(None)

        return self.create_frame(main_frame, satellite_frames, satellite_texts, highlight_idx=assignment, is_stereo=is_stereo)

    @staticmethod
    def from_num_cameras(num_cameras: int, **kwargs) -> 'FrameCreator':
        """
        Factory method to create a FrameCreator instance based on the number of cameras.

        Parameters
        ----------
        num_cameras : int
            The number of cameras in the frame.

        Returns
        -------
        FrameCreator
            An instance of a FrameCreator subclass based on the number of cameras.
        """
        if num_cameras == 0:
            return N0FrameCreator(**kwargs)
        elif num_cameras == 4:
            return N4FrameCreator(**kwargs)
        elif num_cameras == 5:
            return N5FrameCreator(**kwargs)
        else:
            raise ValueError(f"Unsupported number of cameras: {num_cameras}. Currently only 4 cameras are supported.")


class N0FrameCreator(FrameCreator):
    """
    This creates a frame grid composed of only the main (focus) camera, arranged as follows:
     _______________
    |               |
    |     MAIN      |
    |               |
    |_______________|

    where MAIN is the camera of focus.
    """

    def __init__(self, main_size_hw: Tuple[int, int] = (1024, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_size_hw = main_size_hw
        self.satellite_size_hw = (0, 0)
        self.grid_size_hw = main_size_hw

    def create_frame(self, main_image: np.ndarray, satellite_images: List[np.ndarray], satellite_texts: List[Union[str, None]], highlight_idx: Optional[List[int]] = None, is_stereo: bool = False) -> np.ndarray:
        """
        Create a frame grid from the provided images.

        Parameters
        ----------
        main_image : np.ndarray
            The main camera image.
        satellite_images : List[np.ndarray]
            List of satellite camera images.
        satellite_texts : List[Union[str, None]]
            List of text labels for the satellite images. If None, no text will be displayed.
        highlight_idx : Optional[List[int]], optional
            Indices of the satellite cameras to highlight. If provided, a lightgreen border will be drawn around the corresponding satellite images.
        is_stereo : bool, optional
            If True, the images are stereo images and the left camera's data have be used. In that case we need to make sure that both source images (i, i+1) are highlighted. Defaults to False.

        Returns
        -------
        np.ndarray
            The created frame grid as a numpy array.
        """
        # resize main image to focus size
        main_image = cv2.resize(main_image, (self.main_size_hw[1], self.main_size_hw[0]), interpolation=cv2.INTER_LINEAR)
        return main_image


class N4FrameCreator(FrameCreator):
    """
    This creates a frame grid composed of 4 satellite cameras + 1 main (focus) camera, arranged as follows:
     _______________
    |               |
    |     MAIN      |
    |               |
    |_______________|
    | 1 | 2 | 3 | 4 |
    |_______________|

    where MAIN is the camera of focus and 1-4 are the satellite cameras or the projections on other cameras.
    """

    def __init__(self, main_size_hw: Tuple[int, int] = (1024, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_size_hw = main_size_hw
        self.satellite_size_hw = int(main_size_hw[0] * (main_size_hw[1] // 4) / main_size_hw[1]), main_size_hw[1] // 4
        self.grid_size_hw = (main_size_hw[0] + self.satellite_size_hw[0], main_size_hw[1])

    def create_frame(self, main_image: np.ndarray, satellite_images: List[np.ndarray], satellite_texts: List[Union[str, None]], highlight_idx: Optional[List[int]] = None, is_stereo: bool = False) -> np.ndarray:
        """
        Create a frame grid from the provided images.

        Parameters
        ----------
        main_image : np.ndarray
            The main camera image.
        satellite_images : List[np.ndarray]
            List of satellite camera images.
        satellite_texts : List[Union[str, None]]
            List of text labels for the satellite images. If None, no text will be displayed.
        highlight_idx : Optional[List[int]], optional
            Indices of the satellite cameras to highlight. If provided, a lightgreen border will be drawn around the corresponding satellite images.
        is_stereo : bool, optional
            If True, the images are stereo images and the left camera's data have be used. In that case we need to make sure that both source images (i, i+1) are highlighted. Defaults to False.

        Returns
        -------
        np.ndarray
            The created frame grid as a numpy array.
        """
        assert len(satellite_images) == 4, "For N4FrameCreator, exactly 4 satellite images are required."
        # resize main image to focus size
        main_image = cv2.resize(main_image, (self.main_size_hw[1], self.main_size_hw[0]), interpolation=cv2.INTER_LINEAR)
        satellite_images = [cv2.resize(img, (self.satellite_size_hw[1], self.satellite_size_hw[0]), interpolation=cv2.INTER_LINEAR) for img in satellite_images]
        # create the grid
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)
        grid[:self.main_size_hw[0], :self.main_size_hw[1]] = main_image
        for i, (satellite_image, satellite_text) in enumerate(zip(satellite_images, satellite_texts)):
            if satellite_text is not None:
                self._draw_text_top_right(satellite_image, satellite_text)
            row = self.main_size_hw[0]
            col = i * self.satellite_size_hw[1]
            grid[row:row + self.satellite_size_hw[0], col:col + self.satellite_size_hw[1], :] = satellite_image
            # draw borders around the satellite images
        if highlight_idx is not None:
            for i in range(len(satellite_images)):
                if i in highlight_idx or (is_stereo and (i - 1) in highlight_idx):
                    row = self.main_size_hw[0]
                    col = i * self.satellite_size_hw[1]
                    cv2.rectangle(grid, (col, row), (col + self.satellite_size_hw[1] - 1, row + self.satellite_size_hw[0] - 1), (144, 238, 144), 1)
        return grid


class N5FrameCreator(FrameCreator):
    """
    This creates a frame grid composed of 5 satellite cameras + 1 main (focus) camera, arranged as follows:
     ____________________
    |                   |
    |        MAIN       |
    |                   |
    |___________________|
    | 1 | 2 | 3 | 4 | 5 |
    |___________________|

    where MAIN is the camera of focus and 1-4 are the satellite cameras or the projections on other cameras.
    """

    def __init__(self, main_size_hw: Tuple[int, int] = (1024, 1280), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_size_hw = main_size_hw
        self.satellite_size_hw = int(main_size_hw[0] * (main_size_hw[1] // 5) / main_size_hw[1]), main_size_hw[1] // 5
        self.grid_size_hw = (main_size_hw[0] + self.satellite_size_hw[0], main_size_hw[1])

    def create_frame(self, main_image: np.ndarray, satellite_images: List[np.ndarray], satellite_texts: List[Union[str, None]], highlight_idx: Optional[List[int]] = None, is_stereo: bool = False) -> np.ndarray:
        """
        Create a frame grid from the provided images.

        Parameters
        ----------
        main_image : np.ndarray
            The main camera image.
        satellite_images : List[np.ndarray]
            List of satellite camera images.
        satellite_texts : List[Union[str, None]]
            List of text labels for the satellite images. If None, no text will be displayed.
        highlight_idx : Optional[List[int]], optional
            Indices of the satellite cameras to highlight. If provided, a lightgreen border will be drawn around the corresponding satellite images.
        is_stereo : bool, optional
            If True, the images are stereo images and the left camera's data have be used. In that case we need to make sure that both source images (i, i+1) are highlighted. Defaults to False.

        Returns
        -------
        np.ndarray
            The created frame grid as a numpy array.
        """
        assert len(satellite_images) == 5, "For N5FrameCreator, exactly 4 satellite images are required."
        # resize main image to focus size
        main_image = cv2.resize(main_image, (self.main_size_hw[1], self.main_size_hw[0]), interpolation=cv2.INTER_LINEAR)
        satellite_images = [cv2.resize(img, (self.satellite_size_hw[1], self.satellite_size_hw[0]), interpolation=cv2.INTER_LINEAR) for img in satellite_images]
        # create the grid
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)
        grid[:self.main_size_hw[0], :self.main_size_hw[1]] = main_image
        for i, (satellite_image, satellite_text) in enumerate(zip(satellite_images, satellite_texts)):
            if satellite_text is not None:
                self._draw_text_top_right(satellite_image, satellite_text)
            row = self.main_size_hw[0]
            col = i * self.satellite_size_hw[1]
            grid[row:row + self.satellite_size_hw[0], col:col + self.satellite_size_hw[1], :] = satellite_image
            # draw borders around the satellite images
        if highlight_idx is not None:
            for i in range(len(satellite_images)):
                if i in highlight_idx or (is_stereo and (i - 1) in highlight_idx):
                    row = self.main_size_hw[0]
                    col = i * self.satellite_size_hw[1]
                    cv2.rectangle(grid, (col, row), (col + self.satellite_size_hw[1] - 1, row + self.satellite_size_hw[0] - 1), (144, 238, 144), 1)
        return grid


if __name__ == '__main__':
    # # read THUMAN data
    # thuman_root_ = Path('/media/charisoudis/nas_transmixr/Simone/Volumetric_Video/Human Datasets/THuman2_1/rendered@2m')
    # # read RGBDImages
    # rgbd_0_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=0)
    # gs_0_ = GSImage.from_rgbd_image(rgbd_0_, gs_regressor_model='gps', gs_regressor_checkpoint='neptune://85/best')
    # rgbd_1_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=2)
    # gs_1_ = GSImage.from_rgbd_image(rgbd_1_, gs_regressor_model='gps', gs_regressor_checkpoint='neptune://85/best')
    # rgbd_2_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=3)
    # gs_2_ = GSImage.from_rgbd_image(rgbd_2_, gs_regressor_model='gps', gs_regressor_checkpoint='neptune://85/best')
    # rgbd_3_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=1)
    # gs_3_ = GSImage.from_rgbd_image(rgbd_3_, gs_regressor_model='gps', gs_regressor_checkpoint='neptune://85/best')
    # virtual_rgbd_ = RGBDImage.from_thuman(thuman_root_, model=0, main_cam_idx=0, sub_cam_idx=1)
    # # create a N4 frame, with the stitched reconstruction is projected to the virtual camera, shown as the main, and the projection to the ground truth cameras shown in satellite cameras
    # frame_creator_ = N4FrameCreator(mode='depth_recon|depth_gt')
    # rgbd_frame_ = frame_creator_.forward(
    #     gt_images=[rgbd_0_, rgbd_1_, rgbd_2_, rgbd_3_],
    #     virtual_intrinsic=virtual_rgbd_.intrinsic,
    #     virtual_extrinsic=virtual_rgbd_.extrinsic_w2c,
    #     assignment=[1, 2],
    #     virtual_image_size_hw=virtual_rgbd_.image_size_hw,
    #     is_c2w=False,
    # )
    # cv2.imwrite('thuman_n4frame_test.png', cv2.cvtColor(rgbd_frame_, cv2.COLOR_RGB2BGR))
    # gs_frame_ = frame_creator_.forward(
    #     gt_images=[gs_0_, gs_1_, gs_2_, gs_3_],
    #     virtual_intrinsic=virtual_rgbd_.intrinsic,
    #     virtual_extrinsic=virtual_rgbd_.extrinsic_w2c,
    #     assignment=[1, 2],
    #     virtual_image_size_hw=virtual_rgbd_.image_size_hw,
    #     is_c2w=False,
    # )
    # cv2.imwrite('thuman_n4frame_test_gs.png', cv2.cvtColor(gs_frame_, cv2.COLOR_RGB2BGR))
    # stereo_frame_ = frame_creator_.forward(
    #     gt_images=StereoImage.from_rgb_images(rgbd_0_, rgbd_1_, rgbd_2_, rgbd_3_, return_last_as_rgbd=True, disparity_estimator_model='raftstereo'),
    #     virtual_intrinsic=virtual_rgbd_.intrinsic,
    #     virtual_extrinsic=virtual_rgbd_.extrinsic_w2c,
    #     assignment=[1],
    #     virtual_image_size_hw=virtual_rgbd_.image_size_hw,
    #     is_c2w=False,
    # )
    # cv2.imwrite('thuman_n4frame_test_stereo.png', cv2.cvtColor(stereo_frame_, cv2.COLOR_RGB2BGR))

    # read session data
    session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025/Thanos_2_Perf_1'
    calibration_session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Apr_May_2025/Thanos_2_Calib_1'
    from utils.calib import CalibrationData

    calibration_data_ = CalibrationData.from_session(calibration_session_root_)
    # create RGBDImages
    rgbd_0_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=4, color_ts=1746110341442, depth_ts=1746110341443).resize(1024, 1280)
    rgbd_1_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=5, color_ts=1746110341429, depth_ts=1746110341430).resize(1024, 1280)
    rgbd_2_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=7, color_ts=1746110341432, depth_ts=1746110341433).resize(1024, 1280)
    rgbd_3_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=8, color_ts=1746110341432, depth_ts=1746110341433).resize(1024, 1280)
    rgbd_4_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=9, color_ts=1746110341439, depth_ts=1746110341440).resize(1024, 1280)
    all_rgbds_ = [rgbd_0_, rgbd_1_, rgbd_2_, rgbd_3_, rgbd_4_]
    virtual_rgbd_ = rgbd_1_
    frame_creator_ = FrameCreator.from_num_cameras(len(all_rgbds_), mode='rgb_recon|rgb_gt')
    rgbd_frame_ = frame_creator_.forward(
        gt_images=all_rgbds_,
        virtual_intrinsic=virtual_rgbd_.intrinsic,
        virtual_extrinsic=virtual_rgbd_.extrinsic_w2c,
        assignment=[1],
        virtual_image_size_hw=virtual_rgbd_.image_size_hw,
        is_c2w=False,
    )
    cv2.imwrite('session_n5frame_test.png', cv2.cvtColor(rgbd_frame_, cv2.COLOR_RGB2BGR))
    gs_frame_ = frame_creator_.forward(
        gt_images=[
            GSImage.from_rgbd_image(rgbd_i_, gs_regressor_model='gps', gs_regressor_checkpoint='neptune://85/best')
            for rgbd_i_ in all_rgbds_
        ],
        virtual_intrinsic=virtual_rgbd_.intrinsic,
        virtual_extrinsic=virtual_rgbd_.extrinsic_w2c,
        assignment=[1],
        virtual_image_size_hw=virtual_rgbd_.image_size_hw,
        is_c2w=False,
    )
    cv2.imwrite('session_n5frame_test_gs.png', cv2.cvtColor(gs_frame_, cv2.COLOR_RGB2BGR))
    stereo_frame_ = frame_creator_.forward(
        gt_images=StereoImage.from_rgb_images(*all_rgbds_, return_last_as_rgbd=True, disparity_estimator_model='raftstereo'),
        virtual_intrinsic=virtual_rgbd_.intrinsic,
        virtual_extrinsic=virtual_rgbd_.extrinsic_w2c,
        assignment=[1],
        virtual_image_size_hw=virtual_rgbd_.image_size_hw,
        is_c2w=False,
    )
    cv2.imwrite('session_n5frame_test_stereo.png', cv2.cvtColor(stereo_frame_, cv2.COLOR_RGB2BGR))

    PixelPoints.O3D_VISUALIZER_CACHE.clear()
    import gc

    gc.collect()
