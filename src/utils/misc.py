from __future__ import annotations

import enum
import hashlib
import importlib
import inspect
import io
import logging
import os
import random
import re
import string
import subprocess
import sys
import threading
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple, Union, Optional, Literal, Type, Dict, Any

import cv2
import humanize
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

try:
    import pycolmap

    pycolmap.logging.minloglevel = pycolmap.logging.ERROR
except ImportError:
    pass

from colorlog import ColoredFormatter
from dotenv import dotenv_values, load_dotenv

warnings.filterwarnings("ignore")

GLOBAL_LOGGER = None
GLOBAL_ENV = None
DETECTORS = None
DETECTOR_CLASS_NAMES = None
SEGMENTORS = None
OF_ESTIMATOR = None
OF_ESTIMATOR_PADDER = None
DEPTH_FILTERERS: Dict[str, Optional[Any]] = {
    'bilateral_spatial': None,
    'bilateral_temporal': None,
}
MIN_DEPTH = 200  # mm
MAX_DEPTH = 5_000  # mm
MAX_DEPTH_IN_MASK_MM = 4_500  # mm, used for depth filtering in masks
DETECTOR_LOCK = threading.Lock()
SEGMENTOR_LOCK = threading.Lock()
OF_ESTIMATOR_LOCK = threading.Lock()
DEPTH_FILTERER_LOCK = threading.Lock()


class Env(dict):
    """Env Class:
    A class that represents the environment variables of the project. It is a dictionary of the environment variables
    defined in the `.env` file in the project root directory.
    """
    LOADED_ENV = False

    def __init__(self, env_path: Optional[Path] = None):
        if env_path is None:
            env_path = PathUtils.project_path() / '.env'
        super().__init__(**dotenv_values(env_path))
        for key, value in os.environ.items():
            if key not in self.keys():
                self.__setitem__(key, value)
        if env_path.exists() and not self.__class__.LOADED_ENV:
            load_dotenv(str(env_path))
            self.__class__.LOADED_ENV = True

    def __getitem__(self, item: str) -> Union[str, dict]:
        item_parts = map(lambda x: x.strip(), item.split(','))
        env_key = '_'.join(item_parts).upper()
        if env_key not in self.keys():
            # Return a sub-dictionary with key-value pairs where key start with env_key
            return {k[len(env_key) + 1:].lower(): v for k, v in self.items() if k.startswith(env_key)}
        return super().__getitem__(env_key)

    def __getattr__(self, item):
        if not hasattr(super(dict, self), item):
            return self.__getitem__(item)

    def __setattr__(self, key, value):
        os.environ[key] = value
        self.__setitem__(key, value)


class Gender(enum.Enum):
    MALE = 'm'  # 0
    FEMALE = 'f'  # 1
    NEUTRAL = 'n'  # -1

    @property
    def full(self) -> str:
        if self == Gender.NEUTRAL:
            return 'neutral'
        if self == Gender.MALE:
            return 'male'
        return 'female'

    def int(self):
        if self == Gender.NEUTRAL:
            return -1
        if self == Gender.MALE:
            return 0
        return 1

    def __int__(self):
        return self.int()

    def __str__(self):
        return self.name


class Logger(logging.Logger):
    """ Logger Class:
    The main logger of the project. It is a utility class to log colorfully messages to console.
    """

    LOG_FORMAT_DEFAULT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

    def __init__(self, log_level: str = 'debug', log_format: str = LOG_FORMAT_DEFAULT, name: Optional[str] = None):
        """ Logger class constructor.

        Parameters
        ----------
        log_level: str
            debug Level (one of 'info', 'debug', 'warning', 'error', 'critical')
        log_format: str, optional
            log format or empty to use the default one
        name: str
            logger name (enables logs grouping/isolation)
        """
        self._stream = logging.StreamHandler()
        if name is None:
            name = Str.random(length=10).__str__()
        super().__init__(
            name=name,
            level=log_level.upper()
        )
        self._formatter = ColoredFormatter(log_format)

        self._log_level = None
        self._log_format = None

        self.log_level = log_level
        self.log_format = log_format

        self.addHandler(self._stream)

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, log_level: str):
        log_level = log_level.upper()
        if self._log_level == log_level:
            return

        # Update internal param
        self._log_level = log_level
        # Set log level
        logging.root.setLevel(log_level)
        self._stream.setLevel(log_level)
        self.setLevel(log_level)

    @property
    def log_format(self):
        return self._log_format

    @log_format.setter
    def log_format(self, log_format: str):
        if self._log_format == log_format:
            return
        # Update private param
        self._log_format = log_format
        # Set log format
        new_formatter = ColoredFormatter(log_format)
        self._formatter = new_formatter
        self._stream.setFormatter(new_formatter)


class PathUtils:
    @classmethod
    def ants_path(cls) -> Path:
        """Returns the path to the `ants` directory.

        Returns
        -------
        Path
        """
        return cls.src_path() / "ants"

    @classmethod
    def aria_path(cls) -> Path:
        """Returns the path to the `aria` directory.

        Returns
        -------
        Path
        """
        return cls.src_path() / "aria"

    @classmethod
    def artisan_path(cls) -> Path:
        """Returns the path to the `artisan` directory.

        Returns
        -------
        Path
        """
        return cls.src_path() / "artisan"

    @classmethod
    def assets_path(cls) -> Path:
        """Returns the path to the `resources/assets` directory.

        Returns
        -------
        Path
        """
        return cls.resources_path() / "assets"

    @classmethod
    def capturestudio_cache_path(cls) -> Path:
        """Returns the path to the `capturestudio_cache` directory.

        Returns
        -------
        Path
        """
        return Path(env_get('CAPTURESTUDIO_CACHE', str(cls.out_path() / 'CAPTURESTUDIO_CACHE')))

    @classmethod
    def nas_path(cls) -> Path:
        """Returns the path to the local NAS mount point.

        Returns
        -------
        Path
        """
        return Path(env_get('NAS', str(cls.out_path() / 'NAS')))

    @classmethod
    def nas_cache_path(cls) -> Path:
        """Returns the path to the `nas_cache` directory.

        Returns
        -------
        Path
        """
        return Path(env_get('NAS_CACHE', str(cls.out_path() / 'NAS_CACHE')))

    @classmethod
    def checkpoints_path(cls) -> Path:
        """Returns the path to the `out/checkpoints` directory.

        Returns
        -------
        Path
        """
        return Path(env_get('TORCH_HOME', str(cls.out_path()))) / 'checkpoints'

    @classmethod
    def classes_in(cls, path: Path, cls_type: Optional[Type] = None, exhaustive: bool = False) -> Dict[str, Type]:
        """Returns a dictionary of class names to class types found in the directory `path` and its subdirectories.

        Parameters
        ----------
        path: Path
            The path to the directory.
        cls_type: Optional[Type]
            The type of the classes to return. If None, all classes are returned.
        exhaustive: bool
            If True, all classes in the directory `path` and its subdirectories are returned. If False, only the first
            class of the given type found (in the first module that it is found) is returned.

        Returns
        -------
        Dict[str, Type]
            The dictionary of class names to class types.
        """
        # Get src path
        src_path = cls.src_path()
        # Recursively visit the directory `path` and find all the classes in it and its subdirectories
        classes = {}
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):
                    file_path_abs = os.path.join(root, file[:-3])
                    module_name = os.path.relpath(file_path_abs, src_path).replace(os.sep, ".")
                    module_name = module_name.replace('__init__', '').rstrip('.')
                    try:
                        module = importlib.import_module(module_name)
                    except (ModuleNotFoundError, OSError):
                        continue
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) \
                                and (cls_type is None or issubclass(obj, cls_type)) \
                                and not name.startswith('_'):
                            classes[name] = obj
                            if not exhaustive and cls_type is not None:
                                return classes
        return classes

    @classmethod
    def config_path(cls) -> Path:
        """Returns the path to the `config` directory.

        Returns
        -------
        Path
        """
        return cls.project_path() / "config"

    @classmethod
    def data_path(cls) -> Path:
        """Returns the path to the `data` directory.

        Returns
        -------
        Path
        """
        return cls.project_path() / 'data'

    @classmethod
    def dataset_path(cls, dataset_name: str) -> Path:
        """Returns the path to the directory of the dataset with name `dataset_name`.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset.

        Returns
        -------
        Path
        """
        return Path(env_get('DATA_HOME', str(cls.data_path() / 'Datasets'))) / dataset_name

    @classmethod
    def dependencies_path(cls) -> Path:
        """Returns the path to the `dependencies` directory.

        Returns
        -------
        Path
        """
        return cls.project_path() / "dependencies"

    @classmethod
    def font(cls, font: str) -> Optional[Path]:
        """Returns the path to the font file with name `font`. The function calls the `locate` command of the
        operating system if the font file is not found in the `fonts` directory.

        Parameters
        ----------
        font: str
            The name of the font.

        Returns
        -------
        Optional[Path]
        """
        if os.path.exists(font):
            return Path(font)
        if os.path.exists(cls.fonts_path() / font):
            return cls.fonts_path() / font
        if not font.endswith('.ttf'):
            font = f'{font}.ttf'
            if os.path.exists(cls.fonts_path() / font):
                return cls.fonts_path() / font
        return cls.locate(font, root=str(cls.fonts_path()))

    @classmethod
    def fonts_path(cls, user: bool = True) -> Path:
        """Returns the path to the `fonts` directory.

        Parameters
        ----------
        user: bool
            If True, it will return the path to the user's fonts directory. Otherwise, it will return the
            default fonts directory of Ubuntu.

        Returns
        -------
        Path
        """
        if (cls.resources_path() / 'fonts').exists():
            return cls.resources_path() / 'fonts'
        if user and (Path.home() / '.local' / 'share' / 'fonts').exists():
            return Path.home() / '.local' / 'share' / 'fonts'
        return Path('/usr/share/fonts')

    @classmethod
    def locate(cls, arg: str, root: str = '') -> Optional[Path]:
        """Returns the path to the file whose name is given by `arg`. The function calls the `locate` command of the
        operating system.

        Parameters
        ----------
        arg: str
            The name of the directory.
        root: str, optional
            The location of the root folder to search for `arg` in.

        Returns
        -------
        Path
        """
        # Call the locate command of the operating system
        if sys.platform in ['win32', 'win64', 'darwin']:
            result = os.popen(f'locate {arg}').readlines()
        else:
            result = os.popen(f'find {root} -name {arg}').readlines()
        # If the result is empty, return None
        if len(result) == 0:
            return None
        # Otherwise, return the last result
        return Path(result[-1].strip())

    @classmethod
    def out_path(cls) -> Path:
        """Returns the path to the `out` directory.

        Returns
        -------
        Path
        """
        return cls.project_path() / 'out'

    @classmethod
    def project_path(cls) -> Path:
        """Returns the path to the project (root) directory, which is the parent of the `src` directory.

        Returns
        -------
        Path
        """
        return cls.src_path().parent

    @classmethod
    def read_file(cls, file_path: Path, png_type: Literal['depth', 'flow', 'generic'] = 'generic') -> Union[np.ndarray, torch.Tensor]:
        file_path = Path(file_path)
        tries = 0
        while True:
            try:
                if file_path.suffix == '.npy':
                    return np.load(str(file_path))
                if file_path.suffix == '.png' and png_type == 'depth':
                    data = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                    if data is None:
                        raise ValueError(f"Failed to read depth image from {file_path}. It may be corrupted or not a valid PNG.")
                    return (data.astype(np.float32) / 13.0).astype(np.uint16)  # convert from 16-bit PNG to mm
                if file_path.suffix == '.png' and png_type == 'flow':
                    from utils.flow import FlowUtils
                    return FlowUtils.read_flow_from_png_custom(file_path)
                return cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE if 'mask' in str(file_path) else cv2.IMREAD_COLOR)
            except (ValueError, Exception) as e:
                tries += 1
                if tries > 3:
                    raise RuntimeError(f"Failed to read file {file_path} after 3 attempts: {e}")
                else:
                    # log(f"\t[{cls.__name__}::read_file] Error reading file {file_path}: {e}. Retrying ({tries}/5)...", 'warning')
                    # sleep for a random number of milliseconds to avoid busy waiting
                    time.sleep(random.random() * 0.2 + 0.1)  # between 100ms and 300ms

    @classmethod
    def verify_mp4_decord(cls, file_path: Path, required_frames: int = -1) -> bool:
        """
        Verify the integrity of an MP4 file using Decord.
        Returns True if the file is valid and has >= require_frames frames, False otherwise.
        """
        try:
            from decord import VideoReader
            vr = VideoReader(str(file_path), num_threads=2)
            if required_frames > 0 and len(vr) < required_frames:
                return False
            # Force actual decoding (random access triggers full video decode)
            _ = vr[0]  # first frame
            if len(vr) > 1:
                _ = vr[len(vr) - 1]  # last frame
            return True
        except Exception as e:
            logging.error(f"Decord verification failed for {file_path}: {e}")
            return False

    @classmethod
    def verify_mp4_ffmpeg(cls, file_path: Path, required_frames: int = -1) -> bool:
        """
        MP4 verification using FFmpeg.

        Checks:
        - file exists,
        - metadata can be read,
        - first video stream can decode at least one frame,
        - optional frame estimate is >= required_frames.

        This is faster but less strict than full-file decoding.
        """
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            if not file_path.exists() or not file_path.is_file():
                logging.error("MP4 verification failed; file does not exist: %s", file_path)
                return False

            ffmpeg = get_ffmpeg_exe()

            if required_frames > 0:
                estimated_frames = cls._estimate_frame_count_ffmpeg(file_path)
                if estimated_frames > 0 and estimated_frames < required_frames:
                    return False

            cmd = [
                ffmpeg,
                "-hide_banner",
                "-v", "error",
                "-xerror",
                "-i", str(file_path),
                "-map", "0:v:0",
                "-an",
                "-frames:v", "1",
                "-f", "null",
                "-",
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logging.error(
                    "FFmpeg fast verification failed for %s: %s",
                    file_path,
                    result.stderr.strip(),
                )
                return False

            return True

        except Exception as e:
            logging.error("FFmpeg fast verification failed for %s: %s", file_path, e)
            return False

    @staticmethod
    def _estimate_frame_count_ffmpeg(file_path: Path) -> int:
        """
        Estimate frame count from FFmpeg metadata.

        Returns 0 if unavailable.
        """
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-i", str(file_path),
            "-map", "0:v:0",
            "-f", "null",
            "-",
            "-frames:v", "0",
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        # FFmpeg does not reliably expose frame count without ffprobe.
        # Keep this intentionally conservative.
        return 0

    @classmethod
    def verify_file(cls, file_path: Path, **kwargs) -> bool:
        """
        Quick integrity check:
          • PNG  → Pillow's Image.verify()
          • JPEG → jpeginfo -c  (requires `sudo apt install jpeginfo`)
        Returns True if the file passes, False otherwise.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".png":
            try:
                with Image.open(file_path) as im:
                    im.verify()  # parses header & validates chunk CRCs
                return True
            except (UnidentifiedImageError, OSError, SyntaxError):
                return False

        elif ext in [".jpg", ".jpeg"]:
            proc = subprocess.run(
                ["jpeginfo", "-c", str(file_path)],
                capture_output=True,
                text=True,
            )
            return proc.returncode == 0

        elif ext in ['.mp4', '.mov', '.avi']:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # If we get a duration, the file is likely valid
            return proc.returncode == 0 and proc.stdout.strip() != "" and cls.verify_mp4_ffmpeg(file_path, **kwargs)

        else:
            raise ValueError(f"Unsupported file type for verification: {ext}")

    @classmethod
    def sync_dir(cls, dir_path: Path) -> None:
        for camera_dir in Path(dir_path).iterdir():
            if not camera_dir.is_dir() or not camera_dir.exists():
                continue
            dir_fd = os.open(camera_dir, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            for subdir in camera_dir.iterdir():
                if not subdir.is_dir() or not subdir.exists():
                    continue
                dir_fd = os.open(subdir, os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)

    @classmethod
    def write_file(cls, file_path: Path, data: Union[np.ndarray, torch.Tensor], png_type: Literal["depth", "flow", "generic"] = "generic") -> None:
        """
        Store `data` at `file_path` in one atomic write and force it to disk.

        Supported extensions: .jpg / .jpeg, .png, .npy

        * PNG
            • depth   – clip to [0 mm, 5000 mm], multiply by 13, save as 16-bit PNG
            • flow    – delegated to FlowUtils.write_flow_to_png_custom(data)
            • generic – regular 8-bit or 16-bit PNG via cv2
        * JPEG        – saved with quality = 95
        * NumPy .npy  – round-tripped through an in-memory buffer

        Raises
        ------
        ValueError on unsupported extension or encoding error.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        # ------------------------------------------------------------------
        # 1. Encode completely in memory
        # ------------------------------------------------------------------
        if ext == ".npy":
            bio = io.BytesIO()
            np.save(bio, data)
            payload = bio.getvalue()

        elif ext == ".png":
            if png_type == "depth":
                processed = (np.clip(data, 0, 5_000) * 13.0).astype(np.uint16)
                ok, buf = cv2.imencode(".png", processed)
                if not ok:
                    raise ValueError("cv2.imencode failed for depth PNG")
                payload = buf.tobytes()

            elif png_type == "flow":
                from utils.flow import FlowUtils  # local import to avoid hard dep

                payload = FlowUtils.write_flow_to_png_custom(data, return_bytes=True)

            else:  # generic PNG
                ok, buf = cv2.imencode(".png", data)
                if not ok:
                    raise ValueError("cv2.imencode failed for PNG")
                payload = buf.tobytes()

        elif ext in [".jpg", ".jpeg"]:
            ok, buf = cv2.imencode(".jpg", data, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise ValueError("cv2.imencode failed for JPEG")
            payload = buf.tobytes()

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # single write
        with open(file_path, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

    @classmethod
    def resources_path(cls) -> Path:
        """Returns the path to the `resources` directory.

        Returns
        -------
        Path
        """
        return cls.project_path() / "resources"

    @classmethod
    def torch_extensions_path(cls) -> Path:
        """Returns the path to the `torch_extensions` directory.

        Returns
        -------
        Path
        """
        return cls.resources_path() / 'torch_ext'

    @classmethod
    def torch_extension_path(cls, extension_name: str) -> Path:
        """Returns the path to the specific torch extension directory.

        Parameters
        ----------
        extension_name: str
            The name of the torch extension.

        Returns
        -------
        Path
        """
        return cls.torch_extensions_path() / extension_name

    # noinspection PyIncorrectDocstring
    @classmethod
    def smpl_dir(cls, model: str, model_version: Optional[str] = None, gender: Gender = Gender.NEUTRAL) -> Tuple[Path, str, str]:
        """ Returns the path to the directory containing the smpl model file.

        Parameters
        ----------
        model: str
            The model variant. One of "smpl", "smplh", "smplx", "star", "supr"
        model_version: str
            The model version or None if there's no version suffix in model's directory name. (default = None)

        Returns
        -------
        Path
        """
        if model_version is None:
            model_version = ''
        return cls.checkpoints_path() / 'smpl' / f'{model.lower()}_{str(model_version).replace(".", "_").rstrip("_")}', model, model_version

    @classmethod
    def smpl_path(cls, gender: Gender = Gender.NEUTRAL, *smpl_dir_args, **smpl_dir_kwargs) -> Path:
        """ Returns the path to the smpl model file.

        Parameters
        ----------
        gender: str
            Gender to select model parameters. One of "male", "female", "neutral". (default = "neutral")
        smpl_dir_args: Any
            Positional arguments for the cls.smpl_dir method.
        smpl_dir_args: Any
            Keyword arguments for the cls.smpl_dir method.

        Returns
        -------
        Path
        """
        model_dir, model, model_version = cls.smpl_dir(*smpl_dir_args, **smpl_dir_kwargs)
        for extension in ['pkl', 'npy', 'npz']:
            if (model_dir / f'{gender.full.lower()}.{extension}').exists():
                return model_dir / f'{gender.full.lower()}.{extension}'
        raise FileNotFoundError(f'Cannot find smpl model file for model {model} with version {model_version} and gender {gender}')

    @classmethod
    def src_path(cls) -> Path:
        """ Returns the path to the `src` directory.

        Returns
        -------
        Path
        """
        path = Path(__file__).parent
        while path.name != "src":
            path = path.parent
        return path

    @classmethod
    def results_path(cls) -> Path:
        """Returns the path to the `recon_vis` webapp root directory.

        Returns
        -------
        Path
        """
        return Path(env_get('RESULTS_ROOT', str(cls.out_path())))

    @classmethod
    def webapp_path(cls) -> Path:
        """Returns the path to the `recon_vis` webapp root directory.

        Returns
        -------
        Path
        """
        return Path(env_get('WEBAPP_ROOT', str(cls.out_path() / 'WEBAPP_ROOT')))


def get_depth_filterer(filter_type: str) -> Optional[Any]:
    """Returns the depth filterer of the given type.

    Parameters
    ----------
    filter_type: str
        The type of the depth filterer. One of 'bilateral_spatial', 'bilateral_temporal'.

    Returns
    -------
    Optional[Any]
        The depth filterer of the given type or None if the type is not supported.
    """
    global DEPTH_FILTERERS, DEPTH_FILTERER_LOCK
    with DEPTH_FILTERER_LOCK:
        if filter_type not in DEPTH_FILTERERS:
            raise ValueError(f"Unsupported depth filterer type: {filter_type}")
        if DEPTH_FILTERERS[filter_type] is None:
            from preprocessing.depth.depth_filtering import BilateralFiltering
            DEPTH_FILTERERS[filter_type] = BilateralFiltering(
                smoothing_alpha=1.0,  # linear smoothing weight: 1.0 = no smoothing, > 0.5 = over-smoothing
                filter_radius=9,
                depth_sigma=2.0,
                rgb_sigma=4.0,
                temporal_sigma=3.0,
                iters=3,
                filter_in_time=filter_type == 'bilateral_temporal',
                quantile=0.9999,  # outlier removal quantile threshold
            )
        return DEPTH_FILTERERS[filter_type]


def get_detector(which: Literal['yolo', 'sam3'] = 'sam3', device: Union[str, Literal['cpu', 'cuda']] = 'cuda'):
    global DETECTORS, DETECTOR_CLASS_NAMES
    with DETECTOR_LOCK:
        if DETECTORS is None or which not in DETECTORS:
            if DETECTORS is None:
                DETECTORS = {}

            if which == 'yolo':
                detector_path = PathUtils.checkpoints_path() / 'yolo' / 'yolo11x.pt'
                from ultralytics import YOLO
                DETECTORS[which] = YOLO(detector_path).to(device)
                DETECTORS[which]._class_names = DETECTORS[which].model.names
            elif which == 'sam3':
                from sam3 import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                model = build_sam3_image_model(
                    checkpoint_path=str(PathUtils.checkpoints_path() / "sam" / "sam3.pt"),
                    eval_mode=True,
                    device=device,
                )
                processor = Sam3Processor(model)
                model._processor = processor
                if not hasattr(model, "device"):
                    model.device = device
                # Minimal class catalog for compatibility with existing class-id resolution logic
                sam3_class_list = [
                    "person",
                    "guitar",
                    "guitar band",
                    "drums",
                ]
                model._class_names = {i: n for i, n in enumerate(sam3_class_list)}
                DETECTORS[which] = model
            else:
                raise ValueError(f"Unsupported detector type: {which}")

        # Always return the correct class-name mapping for the selected detector
        detector = DETECTORS[which]
        DETECTOR_CLASS_NAMES = getattr(detector, "_class_names", DETECTOR_CLASS_NAMES)

    return DETECTORS[which], DETECTOR_CLASS_NAMES


def get_segmentor(which: Literal['sam2', 'sam3'] = 'sam3', **kwargs):
    global SEGMENTORS
    with SEGMENTOR_LOCK:
        if SEGMENTORS is None or which not in SEGMENTORS:
            if SEGMENTORS is None:
                SEGMENTORS = {}
            if which == 'sam2':
                segmentor_paths = dict(
                    config_file='/' + str(PathUtils.checkpoints_path() / 'sam' / 'sam2.1_hiera_base_plus.yaml'),
                    ckpt_path=str(PathUtils.checkpoints_path() / 'sam' / 'sam2.1_hiera_base_plus.pt'),
                )
                from sam2.build_sam import build_sam2_video_predictor
                SEGMENTORS[which] = build_sam2_video_predictor(**segmentor_paths, vos_optimized=True, **kwargs)
            elif which == 'sam3':
                ckpt = Path(env_get("SAM3_CKPT", str(PathUtils.checkpoints_path() / "sam" / "sam3.pt"))).resolve()
                from sam3.model_builder import build_sam3_video_predictor
                SEGMENTORS[which] = build_sam3_video_predictor(
                    checkpoint_path=str(ckpt),
                    **kwargs
                )
    return SEGMENTORS[which]


def get_of_estimator():
    global OF_ESTIMATOR, OF_ESTIMATOR_PADDER
    with OF_ESTIMATOR_LOCK:
        if OF_ESTIMATOR is None:
            of_estimator_ckpt = str(PathUtils.checkpoints_path() / 'videoflow' / 'BOF_sintel.pth')
            from videoflow.core.Networks import build_network
            from videoflow.configs.sintel import get_cfg
            sintel_cfg = get_cfg()
            bofnet = build_network(sintel_cfg)
            bofnet.load_state_dict({k[7:]: v for k, v in torch.load(of_estimator_ckpt).items() if k.startswith('module.')})
            bofnet.eval().cuda()
            OF_ESTIMATOR = bofnet
            from videoflow.core.utils.utils import InputPadder
            OF_ESTIMATOR_PADDER = InputPadder
        return OF_ESTIMATOR, OF_ESTIMATOR_PADDER


def get_global_env() -> Env:
    global GLOBAL_ENV
    if GLOBAL_ENV is None:
        GLOBAL_ENV = Env()
    return GLOBAL_ENV


def env_get(key: str, default: Optional[str] = None) -> Union[str, dict]:
    global_env = get_global_env()
    env_value = None
    try:
        for part in key.split('.'):
            if env_value is None:
                env_value = global_env.get(part, default)
            else:
                env_value = env_value.get(part, default)
        return env_value
    except KeyError:
        return default


def env_set(key: str, value: str) -> None:
    global_env = get_global_env()
    return global_env.__setattr__(key, value)


def get_global_logger() -> Logger:
    global GLOBAL_LOGGER
    if GLOBAL_LOGGER is None:
        global_log_level = env_get('LOG_LEVEL', 'debug')
        GLOBAL_LOGGER = Logger(name='global', log_level=global_log_level)
    return GLOBAL_LOGGER


class Str(str):
    def append(self, other: str) -> Str:
        return Str(str(self) + other)

    def append_if(self, other: str) -> Str:
        if not self.endswith(other):
            return self.append(other)

    def camel(self) -> Str:
        """Converts a string to camelCase."""
        pascal = self.pascal().__str__()
        return Str(pascal[0].lower() + pascal[1:])

    def human_readable(self, size_format: str = '%.1f', append_original: bool = False) -> Str:
        """ Convert input number to a human-readable string (e.g. 15120 --> 15K).

        Parameters
        ----------
        size_format: str
            format argument of humanize.naturalsize()
        append_original: bool
            set to True to return input number after human-readable and inside parentheses

        Returns
        -------
        Str
            human-readable formatted string
        """
        num_string = humanize.naturalsize(int(self), format=size_format)
        num_string = num_string.replace('.0', '').replace('Byte', '').replace('kB', 'K').rstrip('Bs').replace(' ', '')
        num_string = num_string.replace('G', 'B')  # billions
        return Str(num_string + (f' ({self})' if append_original else ''))

    def lizard(self) -> Str:
        """Converts a string to lizardcase."""
        return self.snake().replace('_', '')

    def lower(self) -> Str:
        """Converts a string to lowercase."""
        return Str(super().lower())

    def pascal(self) -> Str:
        """Converts a string to PascalCase."""
        return Str(''.join(i.capitalize() for i in self.lower().split("_")))

    def replace(self, haystack: str, needle: str = '', count: int = -1) -> Str:
        return Str(re.sub(haystack, needle, self))

    def rgb(self, normalize: bool = True) -> Tuple[Union[int, float], Union[int, float], Union[int, float]]:
        """Converts a hex color string or CSS color name to an RGB tuple.

        Parameters
        ----------
        normalize: bool
            set to True to normalize the RGB values to the range [0, 1]

        Returns
        -------
        Tuple[int or float, ...]
            RGB tuple, with values in the range [0, 255] if normalize is False, or [0, 1] if normalize is True
        """
        if self.startswith('#'):
            value = self.lstrip('#')
            lv = len(value)
            # noinspection PyTypeChecker
            out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        else:
            os.environ['COLOUR_SCIENCE__COLOUR__IMPORT_VAAB_COLOUR'] = 'True'
            # noinspection PyUnresolvedReferences
            from colour import Color
            out = Color(self.replace(" ", "")).rgb
        return tuple(map(lambda x: x * 255 if not normalize else x, out))

    def scanf(self, regex: str) -> list:
        return list(re.match(regex, self).groups())

    def snake(self) -> Str:
        return Str(re.sub(r'(?<!^)(?=[A-Z])', '_', self).lower())

    def split(self, regex: Optional[str] = None, maxsplit: int = -1) -> list:
        if regex is None:
            regex = ' '
        return list(re.split(regex, self))

    def trim(self) -> Str:
        return Str(self.strip())

    def upper(self) -> Str:
        return Str(super().upper())

    @staticmethod
    def callable(transforms: str) -> Callable:
        transforms_list = list(map(str.strip, transforms.split('>')))
        if len(transforms_list) == 1:
            return lambda item: getattr(Str(item), transforms)()
        return partial(Str.chain_callables, transforms=transforms_list)

    @staticmethod
    def chain_callables(item: str, transforms: List[str]):
        item = Str(item)
        for transform in transforms:
            arguments = []
            if ':' in transform:
                transform, arguments = list(map(str.strip, transform.split(':')))
                arguments = arguments.split(',')
            item = getattr(item, transform)(*arguments)
        return item

    @staticmethod
    def group_by_prefix(strings: Union[List[str], List[dict]], separator: str = '_',
                        dict_key: Optional[str] = None) -> dict:
        """
        Groups a list of strings by the common prefixes found in the strings.

        Parameters
        ----------
        strings: List[str] or List[dict]
            list of strings that will be grouped by their prefixes
        separator: str
            prefix separator, splits string in two parts: before and after the 1st appearance of the separator
            (defaults to "_")
        dict_key: str, optional
            if str_list contains dictionaries, this will be used to extract the key to separate strings on

        Returns
        -------
        dict
            dictionary in the form {'prefix1': [suffix1, suffix2, ...], 'prefix2': [suffix1, suffix2, ...]}
        """
        strings_by_prefix = {}
        for s in strings:
            _key_to_split = s[dict_key] if isinstance(s, dict) else s
            prefix, suffix = map(str.strip, _key_to_split.split(sep=separator, maxsplit=1))
            group = strings_by_prefix.setdefault(prefix, [])
            group.append(suffix if isinstance(s, str) else s)
        return strings_by_prefix

    @classmethod
    def hash(cls, inp: Union[bytes, str, io.IOBase], algorithm: str = 'sha256') -> Str:
        if isinstance(inp, io.IOBase):
            if hasattr(inp, 'closed') and inp.closed is True:
                raise RuntimeError('File was closed')
            inp.seek(0, 0)
            if isinstance(inp, io.RawIOBase):
                inp_bytes = inp.readall()
            elif isinstance(inp, io.TextIOBase):
                inp_bytes = '\n'.join(inp.readlines()).encode()
            else:
                inp_bytes = '\n'.encode().join(inp.readlines())
            inp.close()
        elif isinstance(inp, str):
            inp_bytes = inp.encode()
        else:
            inp_bytes = inp
        return Str(getattr(hashlib, algorithm)(inp_bytes).hexdigest())

    @staticmethod
    def random(length: int) -> Str:
        """ Get a random string containing ASCII alphanumerical characters.

        Parameters
        ----------
        length: int
            length of the generated string

        Returns
        -------
        Str
            random string with length equal to :attr:`length` containing random characters.
        """
        return Str(''.join(random.choices(string.ascii_letters + string.digits, k=length)))


def log(message: str, level: str = 'debug', logger: Optional[Logger] = None, **kwargs) -> None:
    if logger is None:
        logger = get_global_logger()
    if env_get('LOG', 'true').lower() == 'false':
        return
    return getattr(logger, level)(message, **kwargs)


if __name__ == '__main__':
    print(get_global_env())
    print(env_get('neptune'))
    print(env_get('neptune.api_key'))
    print(env_get('neptune.project'))

    print(PathUtils.project_path(), os.path.exists(PathUtils.project_path()))
    print(PathUtils.src_path(), os.path.exists(PathUtils.src_path()))
    print(PathUtils.data_path(), os.path.exists(PathUtils.data_path()))
    print(PathUtils.dataset_path('Athlone'), os.path.exists(PathUtils.data_path()))

    # logger_ = Logger()
    # logger_.debug('This is a debug message')
    # logger_.info('This is an info message')
    # logger_.warning('This is an warning message')
    # logger_.error('This is an error message')
    # logger_.critical('This is a critical message')

    log('This is a debug message', level='debug')
    log('This is an info message', level='info')
    log('This is an warning message', level='warning')
    log('This is an error message', level='error')
    log('This is a critical message', level='critical')
