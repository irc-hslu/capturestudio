import multiprocessing
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from kombu import Queue

src_root = Path(__file__).resolve().parent.parent
sys.path.append(str(src_root))
sys.path.append(str(src_root.parent / 'resources' / 'submodules'))
load_dotenv(str(src_root.parent / '.env'))

if os.name != "nt":
    # Force the spawn method for multiprocessing to avoid issues with fork on Linux
    # This is necessary for Celery to work correctly with multiprocessing
    from billiard import context

    context._force_start_method("spawn")
    multiprocessing.set_start_method('spawn', force=True)
else:
    # TODO: Check if this is needed for Windows
    pass

from celery import Celery, Task


class AutoRetryTask(Task):
    autoretry_for = (TypeError,)
    max_retries = 3
    retry_backoff = True
    retry_backoff_max = 100
    retry_jitter = False
    default_retry_delay = 60 * 60  # 1 hour


app = Celery(
    "orb_pipeline",
    broker='amqp://guest:guest@localhost:5672//',
    backend='redis://localhost:6379/0',
    task_cls=AutoRetryTask,
)
app.conf.update(
    result_backend="redis://localhost:6379/0",
    task_ignore_result=False,  # for chords
    result_expires=86400,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=False,  # FIX: set to False if there are long-running tasks
    worker_prefetch_multiplier=1,  # important for long tasks
    worker_enable_remote_control=False,
    task_track_started=True,
    task_soft_time_limit=5400,  # 90 min
    task_time_limit=21600,  # 6 hours
    imports=(
        "tasks.download",
        "tasks.synchronization",
        "tasks.calibration",
        "tasks.preprocessing_color",
        "tasks.preprocessing_depth",
        "tasks.reconstruction",
        "tasks.upload",
    ),
    task_queues=(
        Queue("cpu"),
        Queue("gpu"),
    ),
    task_routes={
        "download.download_from_nas": {"queue": "cpu"},
        "preprocessing.color.generate_video": {"queue": "cpu"},
        "preprocessing.color.interactively_annotate": {"queue": "cpu"},
        "preprocessing.color.compute_segmentation_mask": {"queue": "gpu"},
        "preprocessing.color.compute_optical_flow": {"queue": "gpu"},
        "preprocessing.depth.align_depth_to_color": {"queue": "cpu"},
        "preprocessing.depth.filter_depth": {"queue": "cpu"},
        "synchronization.generate_multiview_video": {"queue": "cpu"},
        "synchronization.synchronize_frames": {"queue": "cpu"},
        "synchronization.trim_frames": {"queue": "cpu"},
        "calibration.generate_caliscope_config": {"queue": "cpu"},
        "calibration.generate_caliscope_videos": {"queue": "cpu"},
        "calibration.detect_corners_2d": {"queue": "cpu"},
        "calibration.lift_corners_3d": {"queue": "cpu"},
        "calibration.calibrate_hslu": {"queue": "cpu"},
        "reconstruction.pcd_reconstruction": {"queue": "cpu"},
        "reconstruction.gs_reconstruction": {"queue": "gpu"},
        "reconstruction.generate_teaser_video": {"queue": "cpu"},
        "reconstruction.generate_teaser_grid_video": {"queue": "cpu"},
        "reconstruction.link_to_webapp": {"queue": "cpu"},
        "upload.upload_to_nas": {"queue": "cpu"},
    }
)

import logging, os
from celery.signals import after_setup_logger, after_setup_task_logger, worker_process_init

NOISY_LOGGERS = [
    # PyOpenGL
    "OpenGL", "OpenGL.arrays", "OpenGL.platform", "OpenGL.GL",
    # OpenAPI/jsonschema-ish
    "jsonschema", "prance", "connexion", "openapi_spec_validator",
    # Common chatty libs (choose what you use)
    "urllib3", "botocore", "boto3", "s3transfer",
    "PIL", "matplotlib", "numba", "asyncio", "kombu", "amqp",
    # Neptune
    "neptune", "neptune.new",
]


def _silence_loggers():
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)  # or logging.ERROR


@after_setup_logger.connect
def _silence_root_logger(logger=None, **kwargs):
    # Runs once in the worker main process
    _silence_loggers()


@after_setup_task_logger.connect
def _silence_task_logger(logger=None, **kwargs):
    # Runs once per task logger creation
    _silence_loggers()


@worker_process_init.connect
def _silence_in_children(**kwargs):
    # Runs in every forked/spawned worker process (this is the critical one)
    _silence_loggers()

    # --- OpenCV: silence C++/Python logs ---
    # Env var works for both C++ and Python bindings
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")  # DEBUG/INFO/WARN/ERROR/FATAL/SILENT
    try:
        import cv2
        # OpenCV 4.x preferred API
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        # Fallback (older OpenCV)
        elif hasattr(cv2, "setLogLevel"):
            # 0 is SILENT in older builds
            cv2.setLogLevel(0)
    except Exception:
        pass

    # --- PyOpenGL: keep only errors ---
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logging.getLogger("OpenGL.arrays").setLevel(logging.ERROR)
