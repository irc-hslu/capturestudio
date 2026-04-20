import abc
from functools import partial
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
        tl_x, _, br_x, _ = draw.textbbox((0, 0), text, font=font)
        x = image_pil.width - (br_x - tl_x) - padding
        y = padding
        draw.text((x, y), text, font=font, fill=color)
        image[:] = np.asarray(image_pil)

    def forward(self, gt_images: List[RGBDImage], virtual_intrinsic: np.ndarray, virtual_extrinsic: np.ndarray, assignment: List[int], is_c2w: bool, virtual_image_size_hw: Optional[Tuple[int, int]] = None, **floor_wall_kwargs) -> np.ndarray:
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
            for gt_idx in assignment:  # ORIGINAL LINE
                gt_image = gt_images[gt_idx]
                gt_partial = gt_image.unproject()
                gt_partials.append(gt_partial)
            main_pcd = gt_partials[0].__class__.from_partials(*gt_partials)
            main_image = main_pcd.project(virtual_intrinsic, virtual_extrinsic, virtual_image_size_hw if virtual_image_size_hw is not None else gt_images[0].image_size_hw, is_c2w=is_c2w, point_size=2.0, use_cache=True, **floor_wall_kwargs)
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
    def from_num_cameras(num_cameras: int, rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None, **kwargs) -> 'FrameCreator':
        """
        Factory method to create a FrameCreator instance based on the number of cameras.

        Parameters
        ----------
        num_cameras : int
            The number of cameras in the frame.
        rotate: Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180'], optional
            If not None, indicates the rotation of the cameras in the physical space.

        Returns
        -------
        FrameCreator
            An instance of a FrameCreator subclass based on the number of cameras.
        """
        if num_cameras == 0:
            return N0FrameCreator(**kwargs)  # if (rotate is None or rotate == '180') else N0FrameCreator90(**kwargs)
        elif num_cameras == 4:
            return N4FrameCreator(**kwargs)  # if (rotate is None or rotate == '180') else N4FrameCreator90(**kwargs)
        elif num_cameras == 5:
            return N5FrameCreator(**kwargs) if (rotate is None or rotate == '180') else N5FrameCreator90(rotate=rotate, **kwargs)
        elif num_cameras == 6:
            return N6FrameCreator(**kwargs) if (rotate is None or rotate == '180') else N6FrameCreator90(rotate=rotate, **kwargs)
        elif num_cameras == 8:
            return N8FrameCreator(**kwargs) if (rotate is None or rotate == '180') else N8FrameCreator90(rotate=rotate, **kwargs)
        elif num_cameras == 12 and (rotate is None or rotate == '180'):
            return N12FrameCreator(**kwargs)
        else:
            raise ValueError(f"Unsupported number of cameras: {num_cameras}. Currently up to 8 cameras are supported.")


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
     _____________________________
    |                             |
    |             MAIN            |
    |                             |
    |_____________________________|
    |  1  |  2  |  3  |  4  |  5  |
    +------------------------------

    where MAIN is the camera of focus and 1-4 are the satellite cameras or the projections on other cameras.
    """

    def __init__(self, main_size_hw: Tuple[int, int] = (1280, 1024), *args, **kwargs):
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


class N5FrameCreator90(FrameCreator):
    """
    This creates a frame grid composed of 5 satellite cameras + 1 main (focus) camera, arranged as follows:
     +------------------------+-----+
     |                        |     |
     |                        |  1  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |                        |  2  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |          MAIN          |  3  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |                        |  4  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |                        |  5  |
     |                        |     |
     +------------------------+-----+

    where MAIN is the camera of focus and 1-5 are the satellite cameras or the projections on other cameras.

    Note:
    The constructor keeps the same signature as N5FrameCreator and expects (H, W). For this portrait layout,
    the main image target size is interpreted as (W, H) so calling code does not need to change.
    """

    def __init__(self, rotate: Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE'], main_size_hw: Tuple[int, int] = (1280, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        main_size_hw = main_size_hw[1], main_size_hw[0]
        self.main_size_hw = main_size_hw
        sat_h = self.main_size_hw[0] // 5
        sat_w = int(self.main_size_hw[1] * sat_h / self.main_size_hw[0])
        self.satellite_size_hw = (sat_h, sat_w)
        self.grid_size_hw = (self.main_size_hw[0], self.main_size_hw[1] + self.satellite_size_hw[1])
        self.rotate = rotate

    def create_frame(
            self,
            main_image: np.ndarray,
            satellite_images: List[np.ndarray],
            satellite_texts: List[Union[str, None]],
            highlight_idx: Optional[List[int]] = None,
            is_stereo: bool = False
    ) -> np.ndarray:
        """
        Create a frame grid from the provided images.

        Parameters
        ----------
        main_image : np.ndarray
            The main camera image (already portrait).
        satellite_images : List[np.ndarray]
            List of satellite camera images (already portrait).
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
        assert len(satellite_images) == 5, "For N5FrameCreator90, exactly 5 satellite images are required."

        # rotate
        rotator = partial(cv2.rotate, rotateCode=getattr(cv2, f'ROTATE_{self.rotate}'))

        # resize main image to portrait focus size
        main_image = cv2.resize(rotator(main_image), (self.main_size_hw[1], self.main_size_hw[0]), interpolation=cv2.INTER_LINEAR)
        # resize satellites to portrait tiles
        satellite_images = [
            cv2.resize(rotator(img), (self.satellite_size_hw[1], self.satellite_size_hw[0]), interpolation=cv2.INTER_LINEAR)
            for img in satellite_images
        ]

        # create the grid and place main image on the left
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)
        grid[:self.main_size_hw[0], :self.main_size_hw[1]] = main_image

        # place satellites in a vertical column on the right
        for i, (satellite_image, satellite_text) in enumerate(zip(satellite_images, satellite_texts)):
            if satellite_text is not None:
                self._draw_text_top_right(satellite_image, satellite_text)
            row = i * self.satellite_size_hw[0]
            col = self.main_size_hw[1]
            grid[row:row + self.satellite_size_hw[0], col:col + self.satellite_size_hw[1], :] = satellite_image

        # draw borders around the satellite images
        if highlight_idx is not None:
            for i in range(len(satellite_images)):
                if i in highlight_idx or (is_stereo and (i - 1) in highlight_idx):
                    row = i * self.satellite_size_hw[0]
                    col = self.main_size_hw[1]
                    cv2.rectangle(
                        grid,
                        (col, row),
                        (col + self.satellite_size_hw[1] - 1, row + self.satellite_size_hw[0] - 1),
                        (144, 238, 144),
                        1
                    )
        return grid


class N6FrameCreator(FrameCreator):
    """
    This creates a frame grid composed of 6 satellite cameras + 1 main (focus) camera, arranged as follows:
     ____________________________________
    |                                   |
    |               MAIN                |
    |                                   |
    |___________________________________|
    |  1  |  2  |  3  |  4  |  5  |  6  |
    +-----------------------------------

    where MAIN is the camera of focus and 1-6 are the satellite cameras (or projections on other cameras).
    """

    def __init__(self, main_size_hw: Tuple[int, int] = (1280, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_size_hw = main_size_hw

        sat_w = self.main_size_hw[1] // 6
        sat_h = int(self.main_size_hw[0] * sat_w / self.main_size_hw[1])
        self.satellite_size_hw = (sat_h, sat_w)

        # same overall width as MAIN; satellites go as a single row below
        self.grid_size_hw = (self.main_size_hw[0] + self.satellite_size_hw[0], self.main_size_hw[1])

    def create_frame(
            self,
            main_image: np.ndarray,
            satellite_images: List[np.ndarray],
            satellite_texts: List[Union[str, None]],
            highlight_idx: Optional[List[int]] = None,
            is_stereo: bool = False
    ) -> np.ndarray:
        """
        Create a frame grid from the provided images.
        """
        assert len(satellite_images) == 6, "For N6FrameCreator, exactly 6 satellite images are required."

        # resize main image to focus size
        main_image = cv2.resize(
            main_image,
            (self.main_size_hw[1], self.main_size_hw[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # resize satellite images
        satellite_images = [
            cv2.resize(img, (self.satellite_size_hw[1], self.satellite_size_hw[0]), interpolation=cv2.INTER_LINEAR)
            for img in satellite_images
        ]

        # create the grid
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)

        # place main
        grid[:self.main_size_hw[0], :self.main_size_hw[1]] = main_image

        # place satellites in bottom row
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
                    cv2.rectangle(
                        grid,
                        (col, row),
                        (col + self.satellite_size_hw[1] - 1, row + self.satellite_size_hw[0] - 1),
                        (144, 238, 144),
                        1
                    )

        return grid


class N6FrameCreator90(FrameCreator):
    """
    This creates a frame grid composed of 6 satellite cameras + 1 main (focus) camera, arranged as follows:
     +------------------------+-----+
     |                        |     |
     |                        |  1  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |                        |  2  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |                        |  3  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |          MAIN          |  4  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |                        |  5  |
     |                        |     |
     |                        +-----+
     |                        |     |
     |                        |  6  |
     |                        |     |
     +------------------------+-----+

    where MAIN is the camera of focus and 1-6 are the satellite cameras (or projections on other cameras).

    Note:
    The constructor keeps the same signature as N5FrameCreator and expects (H, W). For this portrait layout,
    the main image target size is interpreted as (W, H) so calling code does not need to change.
    """

    def __init__(self, rotate: Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE'], main_size_hw: Tuple[int, int] = (1280, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_size_hw = main_size_hw
        sat_h = self.main_size_hw[0] // 6
        sat_w = int(self.main_size_hw[1] * sat_h / self.main_size_hw[0])
        self.satellite_size_hw = (sat_h, sat_w)
        # main on the left, satellites in a vertical column on the right
        self.grid_size_hw = (self.main_size_hw[0], self.main_size_hw[1] + self.satellite_size_hw[1])
        self.rotate = rotate

    def create_frame(
            self,
            main_image: np.ndarray,
            satellite_images: List[np.ndarray],
            satellite_texts: List[Union[str, None]],
            highlight_idx: Optional[List[int]] = None,
            is_stereo: bool = False
    ) -> np.ndarray:
        """
        Create a frame grid from the provided images.
        """
        assert len(satellite_images) == 6, "For N6FrameCreator90, exactly 6 satellite images are required."

        # # rotate
        # rotator = partial(cv2.rotate, rotateCode=getattr(cv2, f'ROTATE_{self.rotate}'))

        # resize main image to portrait focus size (after rotation)
        main_image = cv2.resize(main_image, (self.main_size_hw[1], self.main_size_hw[0]), interpolation=cv2.INTER_LINEAR)

        # resize satellites to portrait tiles (after rotation)
        satellite_images = [
            cv2.resize(img, (self.satellite_size_hw[1], self.satellite_size_hw[0]), interpolation=cv2.INTER_LINEAR)
            for img in satellite_images
        ]

        # create the grid and place main image on the left
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)
        grid[:self.main_size_hw[0], :self.main_size_hw[1]] = main_image

        # place satellites in a vertical column on the right
        for i, (satellite_image, satellite_text) in enumerate(zip(satellite_images, satellite_texts)):
            if satellite_text is not None:
                self._draw_text_top_right(satellite_image, satellite_text)

            row = i * self.satellite_size_hw[0]
            col = self.main_size_hw[1]
            grid[row:row + self.satellite_size_hw[0], col:col + self.satellite_size_hw[1], :] = satellite_image

        # draw borders around the satellite images
        if highlight_idx is not None:
            for i in range(len(satellite_images)):
                if i in highlight_idx or (is_stereo and (i - 1) in highlight_idx):
                    row = i * self.satellite_size_hw[0]
                    col = self.main_size_hw[1]
                    cv2.rectangle(grid, (col, row), (col + self.satellite_size_hw[1] - 1, row + self.satellite_size_hw[0] - 1), (144, 238, 144), 1)

        return grid


class N8FrameCreator(FrameCreator):
    """
    This creates a frame grid composed of 8 satellite cameras + 1 main (focus) camera, arranged as follows:

     +-------------+------------------------------+-------------+
     |             |                              |             |
     |    BLACK    |                              |    BLACK    |
     |             |             MAIN             |             |
     +-------------+                              +-------------+
     |      1      |                              |      8      |
     +-------------+------------------------------+-------------+
     |      2      |   3   |   4   |   5   |   6  |      7      |
     +-------------+------------------------------+-------------+

    Index mapping (0-based in satellite_images):
      0 -> tile "1" (row 1, left)
      1 -> tile "2" (row 2, left)
      2 -> tile "3" (row 2, under main)
      3 -> tile "4" (row 2, under main)
      4 -> tile "5" (row 2, under main)
      5 -> tile "6" (row 2, under main)
      6 -> tile "7" (row 2, right)
      7 -> tile "8" (row 1, right)

    Notes
    -----
    - The black areas are simply left as zeros in the output grid.
    - Satellite tiles are sized to preserve the main aspect ratio given their target width.
    """

    def __init__(self, main_size_hw: Tuple[int, int] = (1280, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_size_hw = main_size_hw

        # Middle row under MAIN has 4 satellites across => each satellite width = main_w / 4
        sat_w = self.main_size_hw[1] // 4
        sat_h = int(self.main_size_hw[0] * sat_w / self.main_size_hw[1])
        self.satellite_size_hw = (sat_h, sat_w)

        # Grid:
        #   width  = left col (sat_w) + main_w + right col (sat_w)
        #   height = main_h + 2 satellite rows
        self.grid_size_hw = (
            self.main_size_hw[0] + 2 * self.satellite_size_hw[0],
            self.main_size_hw[1] + 2 * self.satellite_size_hw[1],
        )

    def create_frame(
            self,
            main_image: np.ndarray,
            satellite_images: List[np.ndarray],
            satellite_texts: List[Union[str, None]],
            highlight_idx: Optional[List[int]] = None,
            is_stereo: bool = False
    ) -> np.ndarray:
        """
        Create a frame grid from the provided images.
        """
        assert len(satellite_images) == 8, "For N8FrameCreator, exactly 8 satellite images are required."

        # resize main image to focus size
        main_image = cv2.resize(main_image, (self.main_size_hw[1], self.main_size_hw[0]), interpolation=cv2.INTER_LINEAR)

        # resize satellite images
        satellite_images = [
            cv2.resize(img, (self.satellite_size_hw[1], self.satellite_size_hw[0]), interpolation=cv2.INTER_LINEAR)
            for img in satellite_images
        ]

        # create the grid (initialized black)
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)

        # place MAIN at top-middle
        main_row0 = 0
        main_col0 = self.satellite_size_hw[1]
        grid[main_row0:main_row0 + self.main_size_hw[0], main_col0:main_col0 + self.main_size_hw[1], :] = main_image

        def _sat_pos(i: int) -> Tuple[int, int]:
            """
            Return (row, col) top-left for satellite index i (0-based) in this layout.
            """
            sat_h, sat_w = self.satellite_size_hw
            main_h, main_w = self.main_size_hw
            right_col = sat_w + main_w

            # Row 1 under MAIN: sat 1 (idx 0) on left, sat 8 (idx 7) on right
            if i == 0:  # "1"
                return main_h, 0
            if i == 7:  # "8"
                return main_h, right_col

            # Row 2: sat 2 (idx 1) on left, sats 3-6 (idx 2..5) under main, sat 7 (idx 6) on right
            row = main_h + sat_h
            if i == 1:  # "2"
                return row, 0
            if 2 <= i <= 5:  # "3".."6"
                # under MAIN: 4 tiles across
                col = sat_w + (i - 2) * sat_w
                return row, col
            if i == 6:  # "7"
                return row, right_col

            raise ValueError(f"Invalid satellite index {i}; expected 0..7.")

        # place satellites
        for i, (satellite_image, satellite_text) in enumerate(zip(satellite_images, satellite_texts)):
            if satellite_text is not None:
                self._draw_text_top_right(satellite_image, satellite_text)
            row, col = _sat_pos(i)
            grid[row:row + self.satellite_size_hw[0], col:col + self.satellite_size_hw[1], :] = satellite_image

        # draw borders around the satellite images
        if highlight_idx is not None:
            for i in range(len(satellite_images)):
                if i in highlight_idx or (is_stereo and (i - 1) in highlight_idx):
                    row, col = _sat_pos(i)
                    cv2.rectangle(grid, (col, row), (col + self.satellite_size_hw[1] - 1, row + self.satellite_size_hw[0] - 1), (144, 238, 144), 1)

        return grid


class N8FrameCreator90(FrameCreator):
    """
    Portrait layout for 8 satellites + 1 main (focus) camera:

     +-----+------------------------+-----+
     |  1  |                        |  5  |
     +-----+                        +-----+
     |  2  |                        |  6  |
     +-----+          MAIN          +-----+
     |  3  |                        |  7  |
     +-----+                        +-----+
     |  4  |                        |  8  |
     +-----+------------------------+-----+

    - 4 satellites on the left column (top-to-bottom)
    - 4 satellites on the right column (top-to-bottom)

    Note:
    The constructor keeps the same signature as N5FrameCreator and expects (H, W). For this portrait layout,
    the main image target size is interpreted as (W, H) so calling code does not need to change.

    Index mapping (0-based):
      0..3 -> left column (top-to-bottom)
      4..7 -> right column (top-to-bottom)
    """

    def __init__(self, rotate: Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE'], main_size_hw: Tuple[int, int] = (1280, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        # interpret (H, W) as (W, H) for portrait layout
        main_size_hw = (main_size_hw[1], main_size_hw[0])
        self.main_size_hw = main_size_hw
        sat_h = self.main_size_hw[0] // 4
        sat_w = int(self.main_size_hw[1] * sat_h / self.main_size_hw[0])
        self.satellite_size_hw = (sat_h, sat_w)
        # satellites on both sides of the main
        self.grid_size_hw = (self.main_size_hw[0], self.main_size_hw[1] + 2 * self.satellite_size_hw[1])
        self.rotate = rotate

    def create_frame(self, main_image: np.ndarray, satellite_images: List[np.ndarray], satellite_texts: List[Union[str, None]], highlight_idx: Optional[List[int]] = None, is_stereo: bool = False) -> np.ndarray:
        """
        Create a frame grid from the provided images.
        """
        assert len(satellite_images) == 8, "For N8FrameCreator90, exactly 8 satellite images are required."

        # resize main image to portrait focus size (after rotation)
        main_image = cv2.resize(main_image, (self.main_size_hw[1], self.main_size_hw[0]), interpolation=cv2.INTER_LINEAR)

        # resize satellites to portrait tiles (after rotation)
        satellite_images = [
            cv2.resize(img, (self.satellite_size_hw[1], self.satellite_size_hw[0]), interpolation=cv2.INTER_LINEAR)
            for img in satellite_images
        ]
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)

        # place main in the middle
        main_col0 = self.satellite_size_hw[1]
        grid[:self.main_size_hw[0], main_col0:main_col0 + self.main_size_hw[1], :] = main_image

        def _sat_pos(i: int) -> Tuple[int, int]:
            """Return (row, col) top-left for satellite index i (0-based) in this layout."""
            if 0 <= i <= 3:
                # left column
                row = i * self.satellite_size_hw[0]
                col = 0
                return row, col
            # right column
            row = (i - 4) * self.satellite_size_hw[0]
            col = self.satellite_size_hw[1] + self.main_size_hw[1]
            return row, col

        # place satellites
        for i, (satellite_image, satellite_text) in enumerate(zip(satellite_images, satellite_texts)):
            if satellite_text is not None:
                self._draw_text_top_right(satellite_image, satellite_text)
            row, col = _sat_pos(i)
            grid[row:row + self.satellite_size_hw[0], col:col + self.satellite_size_hw[1], :] = satellite_image

        # highlight borders
        if highlight_idx is not None:
            for i in range(len(satellite_images)):
                if i in highlight_idx or (is_stereo and (i - 1) in highlight_idx):
                    row, col = _sat_pos(i)
                    cv2.rectangle(grid, (col, row), (col + self.satellite_size_hw[1] - 1, row + self.satellite_size_hw[0] - 1), (144, 238, 144), 1)

        return grid


class N12FrameCreator(FrameCreator):
    """
    12 satellite cameras + 1 main (focus) camera arranged as:

     +-----+-----+-----+-----+-----+-----+
     | BLK |                       | BLK |
     +-----+                       +-----+
     |  1  |                       | 12  |
     +-----+          MAIN         +-----+
     |  2  |                       | 11  |
     +-----+                       +-----+
     |  3  |                       | 10  |
     +-----+-----+-----+-----+-----+-----+
     |  4  |  5  |  6  |  7  |  8  |  9  |
     +-----+-----+-----+-----+-----+-----+

    Logical grid: 5 rows x 6 cols.

    Round order mapping (1-based satellites shown above):
      1..4   : left column, top -> bottom  (rows 1..4, col 0)
      5..8   : bottom row, left -> right  (row 4, cols 1..4)
      9..12  : right column, bottom -> top (rows 4..1, col 5)

    satellite_images index mapping (0-based):
      0..3   -> 1..4
      4..7   -> 5..8
      8..11  -> 9..12
    """

    def __init__(self, main_size_hw: Tuple[int, int] = (1280, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_size_hw = main_size_hw

        # MAIN occupies 4x4 logical cells (rows 0..3, cols 1..4)
        cell_h = self.main_size_hw[0] // 4
        cell_w = self.main_size_hw[1] // 4
        self.cell_size_hw = (cell_h, cell_w)

        # Satellites are 1 cell each
        self.satellite_size_hw = self.cell_size_hw

        # Full grid is 5 rows x 6 cols (includes left and right columns + 4 main columns)
        self.grid_size_hw = (5 * cell_h, 6 * cell_w)

    def create_frame(
            self,
            main_image: np.ndarray,
            satellite_images: List[np.ndarray],
            satellite_texts: List[Union[str, None]],
            highlight_idx: Optional[List[int]] = None,
            is_stereo: bool = False
    ) -> np.ndarray:
        assert len(satellite_images) == 12, "For N12FrameCreator, exactly 12 satellite images are required."

        cell_h, cell_w = self.cell_size_hw

        # Resize main to 4x4 cells
        main_image = cv2.resize(
            main_image,
            (4 * cell_w, 4 * cell_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Resize satellites to 1 cell
        satellite_images = [
            cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
            for img in satellite_images
        ]

        # Create grid (initialized black)
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)

        # Place MAIN at rows 0..3, cols 1..4
        grid[0:4 * cell_h, 1 * cell_w:5 * cell_w, :] = main_image

        def _sat_pos(i: int) -> Tuple[int, int]:
            """
            Return (row_px, col_px) top-left for satellite index i (0-based),
            following the requested round order placement.
            """
            # 1..4 (idx 0..3): left column, rows 1..4, col 0
            if 0 <= i <= 3:
                row = (i + 1) * cell_h
                col = 0
                return row, col

            # 5..8 (idx 4..7): bottom row (row 4), cols 1..4
            if 4 <= i <= 7:
                row = 4 * cell_h
                col = (i - 3) * cell_w  # i=4->col1, i=7->col4
                return row, col

            # 9..12 (idx 8..11): right column, bottom->top, col 5, rows 4..1
            if 8 <= i <= 11:
                row = (4 - (i - 8)) * cell_h  # i=8->row4, i=11->row1
                col = 5 * cell_w
                return row, col

            raise ValueError(f"Invalid satellite index {i}; expected 0..11.")

        # Place satellites
        for i, (sat_img, sat_text) in enumerate(zip(satellite_images, satellite_texts)):
            if sat_text is not None:
                self._draw_text_top_right(sat_img, sat_text)

            r, c = _sat_pos(i)
            grid[r:r + cell_h, c:c + cell_w, :] = sat_img

        # Draw highlights
        if highlight_idx is not None:
            for i in range(len(satellite_images)):
                if i in highlight_idx or (is_stereo and (i - 1) in highlight_idx):
                    r, c = _sat_pos(i)
                    cv2.rectangle(
                        grid,
                        (c, r),
                        (c + cell_w - 1, r + cell_h - 1),
                        (144, 238, 144),
                        1
                    )

        return grid


class N12FrameCreator90(FrameCreator):
    """
    Rotated (90° CCW) layout for 12 satellite cameras + 1 main (focus) camera.

    ASCII layout (portrait-ish cells):

    +-----+-----+-----+-----+-----+
    |     |     |     |     |     |
    | BLK |  1  |  2  |  3  |  4  |
    |     |     |     |     |     |
    +-----+-----+-----+-----+-----+
    |                       |     |
    |                       |  5  |
    |                       |     |
    +                       +-----+
    |                       |     |
    |                       |  6  |
    |                       |     |
    +                       +-----+
    |                       |     |
    |                       |  7  |
    |                       |     |
    +                       +-----+
    |                       |     |
    |                       |  8  |
    |                       |     |
    +-----+-----+-----+-----+-----+
    |     |     |     |     |     |
    | BLK | 12  | 11  | 10  |  9  |
    |     |     |     |     |     |
    +-----+-----+-----+-----+-----+

    - Grid is 6 rows x 5 cols of *cells*.
    - MAIN is one merged block occupying rows 1..4 and cols 0..3 (4x4 cells).
    - Right column (col 4), rows 1..4: satellites 5,6,7,8 (top->bottom).
    - Top row (row 0): [BLK, 1, 2, 3, 4] across cols 0..4.
    - Bottom row (row 5): [BLK, 12, 11, 10, 9] across cols 0..4.

    Rotation:
    - Both MAIN and satellite images are rotated 90° CCW before placement/resizing.
    """

    def __init__(self, main_size_hw: Tuple[int, int] = (1280, 1024), *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Target size of the MAIN merged block (H, W) after rotation/resizing.
        # Keep signature consistent with other creators (H, W).
        self.main_size_hw = main_size_hw

        # MAIN is 4x4 cells -> define one cell size from main target size.
        self.cell_h = self.main_size_hw[0] // 4
        self.cell_w = self.main_size_hw[1] // 4
        self.satellite_size_hw = (self.cell_h, self.cell_w)

        # Whole grid is 6 rows x 5 cols of cells.
        self.grid_size_hw = (6 * self.cell_h, 5 * self.cell_w)

    def create_frame(
            self,
            main_image: np.ndarray,
            satellite_images: List[np.ndarray],
            satellite_texts: List[Union[str, None]],
            highlight_idx: Optional[List[int]] = None,
            is_stereo: bool = False
    ) -> np.ndarray:
        assert len(satellite_images) == 12, "For N12FrameCreatpo90, exactly 12 satellite images are required."

        # Rotate everything 90° CCW
        rotator = partial(cv2.rotate, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Resize MAIN to merged 4x4 block
        main_image = cv2.resize(
            rotator(main_image),
            (4 * self.cell_w, 4 * self.cell_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Rotate + resize satellites to one cell
        satellite_images = [
            cv2.resize(rotator(img), (self.cell_w, self.cell_h), interpolation=cv2.INTER_LINEAR)
            for img in satellite_images
        ]

        # Create output grid (black)
        grid = np.zeros((self.grid_size_hw[0], self.grid_size_hw[1], 3), dtype=np.uint8)

        # Place MAIN (rows 1..4, cols 0..3)
        main_row0 = 1 * self.cell_h
        main_col0 = 0 * self.cell_w
        grid[main_row0:main_row0 + 4 * self.cell_h, main_col0:main_col0 + 4 * self.cell_w, :] = main_image

        def _sat_pos(i: int) -> Tuple[int, int]:
            """
            Satellite index mapping (0-based in satellite_images) -> cell position.

            We interpret satellite_images[0] as "1", ..., satellite_images[11] as "12",
            matching your ASCII labels.

            Positions:
              1..4  -> top row, cols 1..4
              5..8  -> right col (col 4), rows 1..4
              9..12 -> bottom row, cols 4..1 (left-to-right shows 12,11,10,9)
            """
            # 1..4 (idx 0..3): top row, cols 1..4
            if 0 <= i <= 3:
                row = 0
                col = i + 1
                return row * self.cell_h, col * self.cell_w

            # 5..8 (idx 4..7): right col, rows 1..4
            if 4 <= i <= 7:
                row = (i - 4) + 1  # 1..4
                col = 4
                return row * self.cell_h, col * self.cell_w

            # 9..12 (idx 8..11): bottom row, cols 4..1 such that [12,11,10,9] appear left-to-right at cols 1..4
            # ASCII bottom row: BLK, 12, 11, 10, 9
            # So: idx 11 ("12") -> col 1, idx 10 ("11") -> col 2, idx 9 ("10") -> col 3, idx 8 ("9") -> col 4
            if 8 <= i <= 11:
                row = 5
                col = 4 - (i - 8)  # i=8->4, i=9->3, i=10->2, i=11->1
                return row * self.cell_h, col * self.cell_w

            raise ValueError(f"Invalid satellite index {i}; expected 0..11.")

        # Place satellites + labels
        for i, (sat_img, sat_text) in enumerate(zip(satellite_images, satellite_texts)):
            if sat_text is not None:
                self._draw_text_top_right(sat_img, sat_text)
            r0, c0 = _sat_pos(i)
            grid[r0:r0 + self.cell_h, c0:c0 + self.cell_w, :] = sat_img

        # Highlight
        if highlight_idx is not None:
            for i in range(len(satellite_images)):
                if i in highlight_idx or (is_stereo and (i - 1) in highlight_idx):
                    r0, c0 = _sat_pos(i)
                    cv2.rectangle(
                        grid,
                        (c0, r0),
                        (c0 + self.cell_w - 1, r0 + self.cell_h - 1),
                        (144, 238, 144),
                        1
                    )

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
    session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Brasov_Sep_2025/Brasov_1_Perf_2'
    calibration_session_root_ = PathUtils.capturestudio_cache_path() / 'Captures_Brasov_Sep_2025/Brasov_1_Calib_1'
    from utils.calib import CalibrationData

    calibration_data_ = CalibrationData.from_session(calibration_session_root_)
    # create RGBDImages
    rgbd_0_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=1, color_ts=1759002758103, depth_ts=1759002758104).rotate('90_COUNTERCLOCKWISE').resize(1280, 1024)
    rgbd_1_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=2, color_ts=1759002758103, depth_ts=1759002758104).rotate('90_COUNTERCLOCKWISE').resize(1280, 1024)
    rgbd_2_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=3, color_ts=1759002758103, depth_ts=1759002758105).rotate('90_COUNTERCLOCKWISE').resize(1280, 1024)
    rgbd_3_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=4, color_ts=1759002758103, depth_ts=1759002758105).rotate('90_COUNTERCLOCKWISE').resize(1280, 1024)
    rgbd_4_ = RGBDImage.from_session(session_root_, calibration_data_, cam_idx=5, color_ts=1759002758102, depth_ts=1759002758103).rotate('90_COUNTERCLOCKWISE').resize(1280, 1024)
    all_rgbds_ = [rgbd_0_, rgbd_1_, rgbd_2_, rgbd_3_, rgbd_4_]
    virtual_rgbd_ = rgbd_1_
    frame_creator_ = FrameCreator.from_num_cameras(len(all_rgbds_), mode='rgb_recon|rgb_gt', rotate='90_COUNTERCLOCKWISE')
    rgbd_frame_ = frame_creator_.forward(
        gt_images=all_rgbds_,
        virtual_intrinsic=virtual_rgbd_.intrinsic,
        virtual_extrinsic=virtual_rgbd_.extrinsic_w2c,
        assignment=[1],
        virtual_image_size_hw=virtual_rgbd_.image_size_hw,
        is_c2w=False,
    )
    cv2.imwrite('session_n5frame_test.png', cv2.cvtColor(rgbd_frame_, cv2.COLOR_RGB2BGR))
    exit(0)

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
