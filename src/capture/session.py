import copy
import struct
import sys
import threading
import time
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Union, Literal, Optional, List, Dict, Tuple

import click
import h5py
import numpy as np
from dotenv import load_dotenv

from capture.setup import CaptureSystem
from utils.filesystem import LocalFilesystem

src_root = Path(__file__).resolve().parent.parent
sys.path.append(str(src_root))
sys.path.append(str(src_root.parent / 'resources' / 'submodules' / 'videoflow'))
load_dotenv(str(src_root.parent / '.env'))

from celery.canvas import chain
from celery.result import AsyncResult

from tasks.pipeline import TaskSpec, GroupSpec, ChainSpec, ChordSpec, Stage, Pipeline
from utils.excel import ExcelUtils
from utils.misc import log, PathUtils, env_get

load_dotenv(PathUtils.project_path() / '.env')


class CaptureSession:
    def __init__(self,
                 system: CaptureSystem,
                 live_monitor: bool = False,
                 is_calibration: bool = False,
                 calibration_pattern: Optional[str] = None,
                 **session_metadata):
        self.system = system
        self.use_nas = system.use_nas
        self.live_monitor = live_monitor
        self.is_calibration = is_calibration
        self.calibration_pattern = calibration_pattern
        self.monitor_thread: Optional[threading.Thread] = None
        if session_metadata is not None:
            self.session_metadata = session_metadata
        if 'start_time' not in session_metadata:
            self.session_metadata['start_time'] = None
        if 'end_time' not in session_metadata:
            self.session_metadata['end_time'] = None
        if 'duration' not in session_metadata:
            self.session_metadata['duration'] = None
        if 'frame_counts' not in session_metadata:
            self.session_metadata['frame_counts'] = []

    def state(self):
        # Capture all constructor args
        return dict(
            system=self.system.state(),
            live_monitor=self.live_monitor,
            is_calibration=self.is_calibration,
            calibration_pattern=self.calibration_pattern,
            session_metadata=self.session_metadata
        )

    @classmethod
    def from_state(cls, state):
        state = copy.copy(state)
        state['system'] = CaptureSystem.from_state(state['system'])
        session = cls(**state)
        if session.use_nas:
            from capture.setup import NASUploader
            session.system.nas_uploader = NASUploader(
                session.system.nas_buffers,
                nas_root=session.system.nas_root,
                local_root=str(session.system.fs.get_root()),
                experiment_name=session.system.experiment_name,
                cam_names=[_.name for _ in sorted([d for d in session.system.fs.get_root().iterdir() if d.is_dir() and d.name != 'caliscope'], key=lambda p: int(p.name.split(' ')[0]))],
                store_using_h5=session.system.store_using_h5,
                termination_event=session.system.termination_event,
            )
        return session

    def start(self):
        """Start the capture system with optional monitoring"""
        if not self.system.is_recording.value:
            self.session_metadata['start_time'] = datetime.now()
            self.system.start()
            if self.live_monitor:
                self.system.start_monitor()
            if self.use_nas:
                self.system.start_nas_uploader()

    # noinspection PyUnresolvedReferences
    def stop(self):
        """Stop the capture system and clean up resources"""
        if self.system.is_recording.value:
            # noinspection PyTypedDict
            self.session_metadata['end_time'] = datetime.now()
            self.session_metadata['duration'] = (self.session_metadata['end_time'] - self.session_metadata['start_time']
                                                 ).total_seconds()
            # Get all profile files' contents and remove files
            self.session_metadata['cam_profiles'] = {}
            for p in self.system.producers:
                with open(p.profile_path, 'r') as f:
                    self.session_metadata['cam_profiles'][p.camera_sn] = json.load(f)
                p.profile_path.unlink()
            shutil.rmtree(self.system.fs.get_root() / 'profiles')

            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)

            # Save session metadata
            self._save_session_metadata()

            self.system.stop()

    def _save_session_metadata(self):
        """Save session metadata to JSON file"""
        metadata_path = self.system.fs.get_root() / 'session_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.state(), f, indent=2, default=str)
        log(f'Saved session metadata to {metadata_path}', 'debug')
        if self.system.nas_uploader is not None and self.system.nas_uploader.nas_fs is not None:
            if isinstance(self.system.nas_uploader.nas_fs, LocalFilesystem):
                self.system.nas_uploader.nas_fs.mkdir('caliscope')
                self.system.nas_uploader.nas_fs.store(str(self.system.fs.get_root() / 'session_metadata.json'),
                                                      str(self.system.fs.get_root() / 'session_metadata.json').replace(str(self.system.nas_uploader.local_root),
                                                                                                                       str(self.system.nas_uploader.nas_root)))
            else:
                self.system.nas_uploader.nas_fss.upload_file(self.system.nas_uploader.nas_root, str(self.system.fs.get_root() / 'session_metadata.json'),
                                                             overwrite=False, create_parents=True, progress_bar=False)
            log(f'Uploaded session metadata to NAS', 'debug')

    @classmethod
    def capture_path(cls, session_name: str) -> Path:
        """Generate standardized capture path with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return PathUtils.out_path() / 'Captures' / f"{session_name}_{timestamp}"

    @classmethod
    def build_multisync_info(cls, ds_root: Path, cam_names: Union[List[str], Literal['all']] = 'all', max_clusters=100_000):
        is_raw = (ds_root / 'raw_color').exists()
        cam_root = (ds_root / 'raw_color') if is_raw else ds_root
        if cam_names == 'all':
            cam_names = sorted([d.name for d in cam_root.iterdir() if d.is_dir() and d.name != 'caliscope'], key=lambda name: int(name.split(' ')[0]))
        # collect timestamps and paths for each camera
        per_cam_ts = []
        total_frames = []
        for cam in cam_names:
            cam_dir = cam_root / cam
            ts_list = []
            paths = sorted(cam_dir.glob('*.jpg' if is_raw else '*.h5'), key=lambda path: int(path.stem.split('-')[0]))
            frame_count = 0
            for p in paths:
                if is_raw:
                    ts_list.append((int(p.stem.split('-')[0]), (p, p.stem, frame_count)))
                    frame_count += 1
                else:
                    with h5py.File(p, 'r') as f:
                        keys = sorted(f['color'].keys(), key=lambda x: float(x))
                    for k in keys:
                        ts_list.append((int(float(k)), (p, k, frame_count)))
                        frame_count += 1
            per_cam_ts.append(ts_list)
            total_frames.append(frame_count)
        # perform multi-camera synchronization
        clusters = []
        unflushed = []
        for cam_name, ts_list in zip(cam_names, per_cam_ts):
            cam_idx = int(cam_name.split(' ')[0].replace('orbbec/cam', '').replace('sony', '13').replace('apple', '14'))
            for ts, info in ts_list:
                if not clusters:
                    clusters.append({'min_ts': ts, 'max_ts': ts, 'members': {cam_idx: info}, 'flushed': False})
                    unflushed.append(0)
                    continue
                last_idx = len(clusters) - 1
                last = clusters[last_idx]
                if ts > last['max_ts'] + 10:
                    cutoff = ts - 10
                    to_flush = [i for i in unflushed if clusters[i]['max_ts'] < cutoff]
                    for i in sorted(to_flush):
                        unflushed.remove(i)
                        clusters[i]['flushed'] = True
                    clusters.append({'min_ts': ts, 'max_ts': ts, 'members': {cam_idx: info}, 'flushed': False})
                    unflushed.append(len(clusters) - 1)
                else:
                    if (last['min_ts'] - 10) <= ts <= (last['max_ts'] + 10):
                        last['members'][cam_idx] = info
                        if ts < last['min_ts']:
                            last['min_ts'] = ts
                        if ts > last['max_ts']:
                            last['max_ts'] = ts
                    else:
                        placed = False
                        for i in range(last_idx - 1, -1, -1):
                            cl = clusters[i]
                            if cam_idx in cl['members']:
                                continue
                            if (cl['min_ts'] - 10) <= ts <= (cl['max_ts'] + 10):
                                cl['members'][cam_idx] = info
                                if ts < cl['min_ts']:
                                    cl['min_ts'] = ts
                                if ts > cl['max_ts']:
                                    cl['max_ts'] = ts
                                placed = True
                                break
                        if not placed:
                            insert_i = bisect_right([c['min_ts'] for c in clusters], ts)
                            clusters.insert(insert_i, {'min_ts': ts, 'max_ts': ts, 'members': {cam_idx: info}, 'flushed': False})
                            unflushed = [u + 1 if u >= insert_i else u for u in unflushed] + [insert_i]
        with open(ds_root / 'multi_sync.info', 'wb') as f:
            cnt = 0
            for cl in clusters:
                f.write(struct.pack('QQI', int(cl['min_ts']), int(cl['max_ts']), len(cl['members'])))
                for cam in sorted(cl['members'].keys()):
                    p, ts_val, write_idx = cl['members'][cam]
                    # print('write_idx', write_idx, 'ts_val', ts_val, 'cam', cam, 'p', p)
                    f.write(struct.pack('IQQ', int(cam), int(ts_val), int(write_idx)))
                cnt += 1
                if cnt >= max_clusters:
                    break

    @classmethod
    def load_multisync_info(cls, ds_root: Path, cam_names: Union[List[str], Literal['all']] = 'all') -> List[Dict[str, Union[int, Dict[int, Tuple[Path, str, int]]]]]:
        """Load multi-sync info from file"""
        if not (ds_root / 'multi_sync.info').exists():
            raise FileNotFoundError(f"Multi-sync info not found at {ds_root}. Please build it first using `build_multisync_info`.")

        is_raw = (ds_root / 'raw_color').exists()
        cam_root = (ds_root / 'raw_color') if is_raw else ds_root
        all_cam_names_int = sorted([int(d.name.split(' ')[0].replace('orbbec/cam', '').replace('sony', '13').replace('apple', '14'))
                                    for d in cam_root.iterdir() if d.is_dir() and d.name != 'caliscope'])
        cam_names_int = [cam_name.split(' ')[0].replace('orbbec/cam', '').replace('sony', '13').replace('apple', '14')
                         for cam_name in cam_names] if cam_names != 'all' else all_cam_names_int
        if cam_names == 'all':
            cam_names = sorted([d.name for d in cam_root.iterdir() if d.is_dir() and d.name != 'caliscope'], key=lambda name: int(name.split(' ')[0]))
        metadata_size = struct.calcsize('QQI')  # min_ts, max_ts, num_cams
        per_cam_data_size = struct.calcsize('IQQ')  # cam_idx, ts, write_idx
        color_paths = {
            cam_name_int: sorted(list((cam_root / cam_name).glob('*.jpg')), key=lambda p: int(p.stem))
            for cam_name, cam_name_int in zip(cam_names, cam_names_int)
        }
        depth_paths = {
            cam_name_int: sorted(list((ds_root / 'raw_depth' / cam_name).glob('*.npy')), key=lambda p: int(p.stem))
            for cam_name, cam_name_int in zip(cam_names, cam_names_int)
        }
        cluster_records = []
        with open(ds_root / 'multi_sync.info', 'rb') as f:
            while True:
                metadata_packed = f.read(metadata_size)
                if not metadata_packed:
                    break
                min_ts, max_ts, num_cams = struct.unpack('QQI', metadata_packed)
                if num_cams == 0:
                    continue
                members = {
                    cam_idx: (None, None)
                    for cam_idx in all_cam_names_int
                }
                found_cam_names_int = []
                for i in range(num_cams):
                    cam_idx, ts_val, write_idx = struct.unpack('IQQ', f.read(per_cam_data_size))
                    if cam_idx in color_paths:
                        if is_raw:
                            members[cam_idx] = color_paths[cam_idx][write_idx], depth_paths[cam_idx][write_idx] if len(depth_paths[cam_idx]) > write_idx else None
                        else:
                            members[cam_idx] = ts_val, write_idx
                        found_cam_names_int.append(cam_idx)
                if sorted(found_cam_names_int) == sorted(cam_names_int):
                    cluster_records.append(dict(sorted(members.items(), key=lambda x: int(x[0]))))
        return cluster_records  # {<cam_idx_int>: (<color_path>, <depth_path>)}

    # noinspection PyUnresolvedReferences
    @staticmethod
    def generate_session_videos(session_path: Path, start_frame_from_end=-1, total_frames: int = -1):
        log(f"Processing capture session: {session_path}", 'info')
        with open(session_path / 'session_metadata.json', 'r') as fp:
            session_state = json.load(fp)
        session_restored = CaptureSession.from_state(session_state)
        capture_system_restored = session_restored.system
        datasets = session_restored.load_session(capture_system_restored.fs.get_root(), mode='hdf5' if capture_system_restored.store_using_h5 else 'raw', fps_synced=capture_system_restored.sync_color_fps_to_depth)
        log(f"Loaded {sum(len(_) for _ in datasets)} frames from capture session", 'debug')

        # =-----------------------------------------------------------------------------------------------------------------#
        # = 1. Generate multi-view videos from captured data
        # =---------------------------------------------------------------------------------------------------------------=#
        if not (capture_system_restored.fs.get_root() / 'vis_color.mp4').exists() or not (capture_system_restored.fs.get_root() / 'vis_depth.mp4').exists():
            color_resizer_ = partial(cv2.resize, dsize=(960, 540), interpolation=cv2.INTER_LINEAR)
            depth_resizer_ = partial(cv2.resize, dsize=tuple([_ // 2 for _ in capture_system_restored.depth_resolution]), interpolation=cv2.INTER_LINEAR)
            multi_dataset_ = MultiCamDataset(datasets, common_point='at_end')
            depth_colormap = cm.jet
            color_video_writer = None
            depth_video_writer = None
            for i, (colors_t_, _, depths_t_, _) in tqdm(enumerate(multi_dataset_), total=len(multi_dataset_), desc='Generating video'):
                color_frame = CaptureSession.make_grid_numpy([color_resizer_(color_t_) for color_t_ in colors_t_], nrow=3, padding=2)[2:-2, 2:-2, ...]
                if color_video_writer is None:
                    color_video_writer = cv2.VideoWriter(str(capture_system_restored.fs.get_root() / 'vis_color.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), capture_system_restored.color_fps, (color_frame.shape[1], color_frame.shape[0]))
                color_video_writer.write(color_frame)
                if capture_system_restored.depth_fps == 15 and capture_system_restored.color_fps == 30 and i % 2 == 1:
                    continue
                depth_frame = (CaptureSession.make_grid_numpy([depth_resizer_(np.clip(depth_ti_, 0, 5_000)) for depth_ti_ in depths_t_], nrow=3, padding=2)[2:-2, 2:-2, ...].astype(np.float32) / 5_000)[..., 0]
                if depth_video_writer is None:
                    depth_video_writer = cv2.VideoWriter(str(capture_system_restored.fs.get_root() / 'vis_depth.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), capture_system_restored.depth_fps, (depth_frame.shape[1], depth_frame.shape[0]))
                depth_video_writer.write((depth_colormap((depth_frame * 255).astype(np.uint8)) * 255)[..., :3].astype(np.uint8))
                if 0 < total_frames <= i:
                    break
            color_video_writer.release()
            log(f"Saved multi-view color video to {capture_system_restored.fs.get_root() / 'vis_color.mp4'}", 'debug')
            depth_video_writer.release()
            log(f"Saved multi-view depth video to {capture_system_restored.fs.get_root() / 'vis_depth.mp4'}", 'debug')
            if capture_system_restored.nas_uploader is not None and capture_system_restored.nas_uploader.nas_fs is not None:
                if isinstance(capture_system_restored.nas_uploader.nas_fs, LocalFilesystem):
                    capture_system_restored.nas_uploader.nas_fs.store(str(capture_system_restored.fs.get_root() / 'vis_color.mp4'),
                                                                      str(capture_system_restored.fs.get_root() / 'vis_color.mp4').replace(str(capture_system_restored.nas_uploader.local_root),
                                                                                                                                           str(capture_system_restored.nas_uploader.nas_root)))
                    capture_system_restored.nas_uploader.nas_fs.store(str(capture_system_restored.fs.get_root() / 'vis_depth.mp4'),
                                                                      str(capture_system_restored.fs.get_root() / 'vis_depth.mp4').replace(str(capture_system_restored.nas_uploader.local_root),
                                                                                                                                           str(capture_system_restored.nas_uploader.nas_root)))
                else:
                    capture_system_restored.nas_uploader.nas_fss.upload_file(capture_system_restored.nas_uploader.nas_root, str(capture_system_restored.fs.get_root() / 'vis_color.mp4'), overwrite=False, create_parents=False, progress_bar=False)
                    capture_system_restored.nas_uploader.nas_fss.upload_file(capture_system_restored.nas_uploader.nas_root, str(capture_system_restored.fs.get_root() / 'vis_depth.mp4'), overwrite=False, create_parents=False, progress_bar=False)
                log(f"Uploaded capture videos to NAS", 'debug')
            log(f"Multi-view videos were generated and stored", 'info')

        # =-----------------------------------------------------------------------------------------------------------------#
        # = 2. Calibrate cameras using Caliscope
        # =---------------------------------------------------------------------------------------------------------------=#
        if session_restored.is_calibration:
            # Load camera profiles
            session_metadata = session_restored.session_metadata
            sn2phy = {capture_system_restored.config[k]['serial_number']: capture_system_restored.config[k]['physical_index'] for k in session_metadata['cam_profiles'].keys()}
            phy2sn = {v: k for k, v in sn2phy.items()}
            cam_profiles = {capture_system_restored.config[k]['physical_index']: v['color'] for k, v in session_metadata['cam_profiles'].items()}
            cam_idx_s1 = {physical_idx: (i + 1) for i, physical_idx in enumerate(sorted([k for k in cam_profiles.keys()], key=lambda x: int(x)))}
            # Generate caliscope configuration file
            if not (capture_system_restored.fs.get_root() / 'caliscope' / 'config.toml').exists():
                import toml

                # Load charuco profile
                with open(PathUtils.resources_path() / 'calibration_patterns' / session_restored.calibration_pattern / 'charuco_info.json', 'r') as fp:
                    charuco_profile = json.load(fp)
                # Generate config file using intrinsic and distortion parameters
                cd = {
                    f'cam_{cam_idx_s1[cam_idx]}': dict(
                        port=cam_idx_s1[cam_idx],
                        physical_index=cam_idx,
                        serial_number=phy2sn[cam_idx],
                        rotation_count=0,
                        error=0.01,
                        translation="null",
                        rotation="null",
                        exposure="null",
                        grid_count=20,
                        size=(cam_profile['width'], cam_profile['height']),
                        matrix=[[cam_profile['intrinsic']['fx'], 0.0, cam_profile['intrinsic']['cx']], [0.0, cam_profile['intrinsic']['fy'], cam_profile['intrinsic']['cy']], [0.0, 0.0, 1.0]],
                        distortions=[cam_profile['distortion']['k1'], cam_profile['distortion']['k2'], cam_profile['distortion']['p1'], cam_profile['distortion']['p2'], cam_profile['distortion']['k3']],
                    )
                    for cam_idx, cam_profile in cam_profiles.items()
                }
                capture_system_restored.fs.mkdir('caliscope')
                with open(capture_system_restored.fs.get_root() / 'caliscope' / 'config.toml', 'w') as fp:
                    toml.dump(
                        dict(
                            camera_count=len(cam_profiles),
                            creation_date=datetime.now().isoformat(),
                            save_tracked_points_video=True,
                            **OrderedDict(sorted(cd.items(), key=lambda x: int(x[0].split('_')[1]))),
                            charuco=charuco_profile,
                        ),
                        fp
                    )
                log(f"Caliscope configuration file generated", 'debug')
                if capture_system_restored.nas_uploader is not None and capture_system_restored.nas_uploader.nas_fs is not None:
                    if isinstance(capture_system_restored.nas_uploader.nas_fs, LocalFilesystem):
                        capture_system_restored.nas_uploader.nas_fs.mkdir('caliscope')
                        capture_system_restored.nas_uploader.nas_fs.store(str(capture_system_restored.fs.get_root() / 'caliscope' / 'config.toml'),
                                                                          str(capture_system_restored.fs.get_root() / 'caliscope' / 'config.toml').replace(str(capture_system_restored.nas_uploader.local_root),
                                                                                                                                                           str(capture_system_restored.nas_uploader.nas_root)))
                    else:
                        capture_system_restored.nas_uploader.nas_fss.upload_file(capture_system_restored.nas_uploader.nas_root, str(capture_system_restored.fs.get_root() / 'caliscope' / 'config.toml'),
                                                                                 overwrite=False, create_parents=True, progress_bar=False)
                    log(f'Uploaded caliscope configuration to NAS', 'debug')
            # Generate extrinsic videos
            if not all((capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'extrinsic' / f'port_{cam_idx_s1[cam_idx]}.mp4').exists() for cam_idx in cam_profiles.keys()):
                extrinsic_video_root = capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'extrinsic'
                extrinsic_video_root.mkdir(parents=True, exist_ok=True)
                intrinsic_video_root = capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'intrinsic'
                intrinsic_video_root.mkdir(parents=True, exist_ok=True)
                multi_dataset_ = MultiCamDataset(datasets, common_point='at_end')
                for dataset_i, (dataset_, start_idx_i) in enumerate(zip(datasets, multi_dataset_.start_idx_per_dataset)):
                    sn_i = dataset_.cam_name.split(' ')[-1].replace('(', '').replace(')', '')
                    phy_idx = sn2phy[sn_i]
                    port_idx = cam_idx_s1[phy_idx]
                    video_writer = cv2.VideoWriter(str(extrinsic_video_root / f'port_{port_idx}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), cam_profiles[phy_idx]['fps'], (cam_profiles[phy_idx]['width'], cam_profiles[phy_idx]['height']))
                    # for color_t_ in tqdm(range(start_idx_i, len(dataset_)), desc=f'Generating calibration videos for CAM {phy_idx:03d} (port={port_idx})'):
                    for color_t_ in tqdm(range(start_idx_i + (len(dataset_) - 1) - start_frame_from_end, start_idx_i + (len(dataset_) - 1) - start_frame_from_end + total_frames), desc=f'Generating calibration videos for CAM {phy_idx:03d} (port={port_idx})'):
                        color_torch = dataset_[color_t_][0]  # (3, H, W)
                        video_writer.write(color_torch.permute(1, 2, 0).numpy() if isinstance(color_torch, torch.Tensor) else color_torch)
                    video_writer.release()
                    shutil.copy(extrinsic_video_root / f'port_{port_idx}.mp4', intrinsic_video_root / f'port_{port_idx}.mp4')
                    log(f'Generated calibration videos for CAM {phy_idx:03d}', 'debug')
                    if capture_system_restored.nas_uploader is not None and capture_system_restored.nas_uploader.nas_fs is not None:
                        if isinstance(capture_system_restored.nas_uploader.nas_fs, LocalFilesystem):
                            capture_system_restored.nas_uploader.nas_fs.mkdir('caliscope/calibration')
                            capture_system_restored.nas_uploader.nas_fs.mkdir('caliscope/calibration/extrinsic')
                            capture_system_restored.nas_uploader.nas_fs.store(str(capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'extrinsic' / f'port_{port_idx}.mp4'),
                                                                              str(capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'extrinsic' / f'port_{port_idx}.mp4').replace(
                                                                                  str(capture_system_restored.nas_uploader.local_root),
                                                                                  str(capture_system_restored.nas_uploader.nas_root)))
                            capture_system_restored.nas_uploader.nas_fs.mkdir('caliscope/calibration/intrinsic')
                            capture_system_restored.nas_uploader.nas_fs.store(str(capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'intrinsic' / f'port_{port_idx}.mp4'),
                                                                              str(capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'intrinsic' / f'port_{port_idx}.mp4').replace(
                                                                                  str(capture_system_restored.nas_uploader.local_root),
                                                                                  str(capture_system_restored.nas_uploader.nas_root)))
                        else:
                            capture_system_restored.nas_uploader.nas_fss.upload_file(capture_system_restored.nas_uploader.nas_root,
                                                                                     str(capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'extrinsic' / f'port_{port_idx}.mp4'),
                                                                                     overwrite=False, create_parents=True, progress_bar=False)
                            capture_system_restored.nas_uploader.nas_fss.upload_file(capture_system_restored.nas_uploader.nas_root,
                                                                                     str(capture_system_restored.fs.get_root() / 'caliscope' / 'calibration' / 'intrinsic' / f'port_{port_idx}.mp4'),
                                                                                     overwrite=False, create_parents=True, progress_bar=False)
                        log(f'Uploaded caliscope extrinsic videos to NAS', 'debug')
        log(f"Capture session completed successfully", 'info')


class SyncedSession:
    def __init__(self, session_name: str, excel_sheet: str = 'Apr_May 2025', excel_file_path: Union[str, Path] = 'Capture Sessions.xls'):
        self.session_name = session_name
        # Open excel to find the session metadata
        self.excel_sheet = excel_sheet
        self.excel_file_path = Path(excel_file_path)
        self.excel_data = ExcelUtils.get_session_data(session_name, sheet=self.excel_sheet, excel_file_path=self.excel_file_path)
        self.orbbec_id = self.excel_data['Session ID']

        self.nas_root = PathUtils.nas_path() / f'Captures_{excel_sheet.replace(" ", "_")}'
        self.nas_path = self.nas_root / 'Orbbec_Fempto_Bolt' / self.orbbec_id
        assert self.nas_path.exists(), f"NAS path {self.nas_path} does not exist. Please check the session name and NAS configuration."
        self.nas_storage_type = 'raw' if (self.nas_path / 'raw_color').exists() else 'h5'

        self.nas_cache_root = PathUtils.nas_cache_path() / f'Captures_{excel_sheet.replace(" ", "_")}'
        self.nas_cache_storage_type = env_get('NAS_CACHE_STORE_AS', 'h5')
        assert self.nas_cache_storage_type == 'h5', f"Only 'h5' storage type is supported for NAS_CACHE. Selected type: {self.nas_cache_storage_type}."

        self.capturestudio_cache_root = PathUtils.capturestudio_cache_path() / f'Captures_{excel_sheet.replace(" ", "_")}'
        self.capturestudio_cache_path = self.capturestudio_cache_root / self.session_name
        self.capturestudio_cache_storage_type = env_get('CAPTURESTUDIO_CACHE_STORE_AS', 'raw')
        assert self.capturestudio_cache_storage_type == 'raw', f"Only 'raw' storage type is supported for CAPTURESTUDIO_CACHE. Selected type: {self.capturestudio_cache_storage_type}."

        # create empty pipeline
        self._pipeline = Pipeline(stages=[])
        self._celery_pipeline: Optional[chain] = None

    def download_from_nas(self, force: bool = False) -> 'SyncedSession':
        """
        Downloads the session data from NAS to the local NAS cache path, on a camera-by-camera basis.
        This method assumes that the NAS path and cache path are correctly set.
        This method does not submit any tasks to the Celery queue, it only appends the tasks to the Celery pipeline.

        Parameters
        ----------
        force : bool
            If True, forces the download even if the data already exists in the cache.
        """
        if not self.nas_cache_root.exists():
            self.nas_cache_root.mkdir(parents=True, exist_ok=True)

        # Get all cameras
        nas_cam_folders = sorted([item for item in (self.nas_path if self.nas_storage_type == 'h5' else (self.nas_path / 'raw_color')).iterdir() if item.is_dir() and not item.name.startswith('_') and item.name not in ['caliscope']],
                                 key=lambda x: int(x.name.split(' ', 1)[0]))

        multi_sync_info_path = self.capturestudio_cache_path / 'orbbec' / 'multi_sync.info'
        download_tasks = []
        for nas_cam_folder in nas_cam_folders:
            nas_cam_name = nas_cam_folder.name
            cc_cam_name = f'cam{int(nas_cam_name.split(" ", 1)[0]):02d}'

            cc_path = self.capturestudio_cache_path / 'orbbec' / cc_cam_name
            if cc_path.exists() and (cc_path / 'color').exists() and multi_sync_info_path.exists() and not force:
                log(f"[{self.session_name}::download_from_nas] Camera {cc_cam_name} already exists in {self.capturestudio_cache_path}. Skipping download.", 'debug')
                continue

            download_tasks.append(
                TaskSpec(
                    name='download.download_from_nas',
                    kwargs=dict(
                        nas_root=str(self.nas_root),
                        orbbec_id=self.orbbec_id,
                        cam_name=nas_cam_name,
                        capturestudio_cache_root=str(self.capturestudio_cache_path),
                        nas_cache_root=str(self.nas_cache_root),
                    )
                )
            )

        download_stage = Stage(
            name='download_from_nas',
            parts=download_tasks
        )
        if download_tasks:
            self._pipeline.stages.append(download_stage)
        return self

    def synchronize(self, force: bool = False) -> 'SyncedSession':
        should_sync = True
        should_generate_multiview_videos = True
        if (self.capturestudio_cache_path / 'orbbec').exists():
            color_counts = []
            depth_counts = []
            cam_dir_names = []
            for cam_dir in [d for d in (self.capturestudio_cache_path / 'orbbec').iterdir() if d.is_dir() and d.name.startswith('cam')]:
                cam_dir_names.append(cam_dir.name)
                color_counts.append(
                    len(list((cam_dir / 'color').glob('*.jpg')))
                )
                if (cam_dir / 'depth').exists():
                    depth_count = max(len(list((cam_dir / 'depth').glob('*.npy'))), len(list((cam_dir / 'depth').glob('*.png'))))
                    if depth_count > 0:
                        depth_counts.append(
                            depth_count
                        )
            local_should_sync = False
            if any(c != color_counts[0] for c in color_counts):
                log(f"[{self.session_name}::synchronize] Color frames count mismatch across cameras: {dict(zip(cam_dir_names, color_counts))}. Re-syncing...", 'warning')
                local_should_sync = True
            if depth_counts and any(d != depth_counts[0] for d in depth_counts):
                log(f"[{self.session_name}::synchronize] Depth frames count mismatch across cameras: {dict(zip(cam_dir_names, depth_counts))}. Re-syncing...", 'warning')
                local_should_sync = True
            if depth_counts and any(d != c for d, c in zip(depth_counts, color_counts)):
                log(f"[{self.session_name}::synchronize] Depth frames count mismatch with color frames: {dict(zip(cam_dir_names, zip(depth_counts, color_counts)))}. Re-syncing...", 'warning')
                local_should_sync = True
            should_sync = local_should_sync

            local_should_generate_multiview_videos = False
            if not (self.capturestudio_cache_root / 'orbbec' / 'multiview_color.mp4').exists() or not (self.capturestudio_cache_root / 'orbbec' / 'multiview_depth.mp4').exists():
                log(f"[{self.session_name}::synchronize] Multiview videos do not exist. Will generate them.", 'debug')
                local_should_generate_multiview_videos = True
            should_generate_multiview_videos = local_should_generate_multiview_videos

            if not should_sync and not should_generate_multiview_videos and not force:
                log(f"All cameras are synchronized and videos generated. Skipping synchronization.", 'info')
                return self

        multiview_video_generation_tasks = GroupSpec(
            parts=[
                TaskSpec(
                    name='synchronization.generate_multiview_video',
                    kwargs=dict(
                        capturestudio_cache_root=str(self.capturestudio_cache_path),
                        modality='color',
                    ),
                ),
                TaskSpec(
                    name='synchronization.generate_multiview_video',
                    kwargs=dict(
                        capturestudio_cache_root=str(self.capturestudio_cache_path),
                        modality='depth',
                        which_depth='depth',
                        depth_format='npy' if self.nas_storage_type == 'raw' else 'png'
                    ),
                )
            ]
        )

        if should_sync:
            sync_tasks = ChainSpec(
                parts=[
                    TaskSpec(
                        name='synchronization.synchronize_frames',
                        kwargs=dict(
                            capturestudio_cache_root=str(self.capturestudio_cache_path),
                            excel_sheet=self.excel_sheet,
                            excel_file_path=str(self.excel_file_path),
                        )
                    ),
                    multiview_video_generation_tasks
                ]
            )
        else:
            sync_tasks = multiview_video_generation_tasks

        sync_stage = Stage(
            name='synchronize',
            parts=[sync_tasks]
        )
        self._pipeline.stages.append(sync_stage)
        return self

    def calibrate(self, calibration_method: Literal['Caliscope', 'MultiCamCalib'], start_offset: int = 0, total_frames: int = -1) -> 'SyncedSession':
        nas_cam_folders = sorted([item for item in (self.nas_path if self.nas_storage_type == 'h5' else (self.nas_path / 'raw_color')).iterdir() if item.is_dir() and not item.name.startswith('_') and item.name not in ['caliscope']],
                                 key=lambda x: int(x.name.split(' ', 1)[0]))
        if calibration_method == 'Caliscope':
            video_generation_tasks = [
                TaskSpec(
                    name='calibration.generate_caliscope_videos',
                    kwargs=dict(
                        capturestudio_cache_root=str(self.capturestudio_cache_path),
                        cam_name=f'cam{int(nas_cam_folder.name.split(" ", 1)[0]):02d}',
                        start_offset=start_offset,
                        total_frames=total_frames,
                        fps=30,
                    )
                )
                for nas_cam_folder in nas_cam_folders
            ]
            config_generation_task = TaskSpec(
                name='calibration.generate_caliscope_config',
                kwargs=dict(
                    capturestudio_cache_root=str(self.capturestudio_cache_path),
                )
            )
            calibration_tasks = GroupSpec(
                parts=[config_generation_task, *video_generation_tasks]
            )
        else:
            raise NotImplementedError(f"Calibration method {calibration_method} not implemented")

        calibration_stage = Stage(
            name='calibrate',
            parts=[calibration_tasks]
        )

        self._pipeline.stages.append(calibration_stage)
        return self

    def preprocess(self, rotate: Optional[Literal['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180']] = None) -> 'SyncedSession':
        if rotate is not None:
            assert rotate in ['90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180'], f"Invalid rotate value: {rotate}. Must be one of '90_COUNTERCLOCKWISE', '90_CLOCKWISE', '180'."

        nas_cam_folders = sorted([item for item in (self.nas_path if self.nas_storage_type == 'h5' else (self.nas_path / 'raw_color')).iterdir() if item.is_dir() and not item.name.startswith('_') and item.name not in ['caliscope']],
                                 key=lambda x: int(x.name.split(' ', 1)[0]))
        all_cam_tasks = []
        for nas_cam_folder in nas_cam_folders:
            nas_cam_name = nas_cam_folder.name
            cc_cam_name = f'cam{int(nas_cam_name.split(" ", 1)[0]):02d}'
            cam_dir = self.capturestudio_cache_path / 'orbbec' / cc_cam_name
            mask_generation_tasks = ChainSpec(
                parts=[
                    TaskSpec(
                        name='preprocessing.color.generate_video',
                        kwargs=dict(
                            cam_color_dir=str(cam_dir / 'color'),
                            start_offset=0,
                            total_frames=-1,
                            fps=30,
                        )
                    ),
                    TaskSpec(
                        name='preprocessing.color.compute_segmentation_mask',
                        kwargs=dict(
                            cam_color_dir=str(cam_dir / 'color'),
                            out_dir=str(cam_dir / 'mask'),
                            start_offset=0,
                            total_frames=-1,
                            rotate=rotate,
                        )
                    )
                ]
            )

            has_depth = (cam_dir / 'depth').exists() and (len(list((cam_dir / 'depth').glob('*.npy'))) > 0 or len(list((cam_dir / 'depth').glob('*.png'))) > 0)
            if has_depth:
                header = GroupSpec(
                    parts=[
                        mask_generation_tasks,
                        TaskSpec(
                            name='preprocessing.depth.align_depth_to_color',
                            kwargs=dict(
                                depth_dir=str(cam_dir / 'depth'),
                                out_dir=str(cam_dir / 'depth_aligned'),
                                parameters_dir=str(cam_dir / 'parameters'),
                                start_offset=0,
                                total_frames=-1,
                                depth_format='png'
                            )
                        )
                    ]
                )
                body = GroupSpec(
                    parts=[
                        TaskSpec(
                            name='preprocessing.depth.filter_depth',
                            kwargs=dict(
                                filter_type='bilateral_spatial',
                                depth_dir=str(cam_dir / 'depth_aligned'),
                                color_dir=str(cam_dir / 'color'),
                                mask_dir=str(cam_dir / 'mask'),
                                out_dir=str(cam_dir / 'depth_filtering_bilateral_spatial'),
                                start_offset=0,
                                total_frames=-1,
                                depth_format='png'
                            )
                        ),
                        ChainSpec(
                            parts=[
                                TaskSpec(
                                    name='preprocessing.color.compute_optical_flow',
                                    kwargs=dict(
                                        cam_color_dir=str(cam_dir / 'color'),
                                        out_dir_bwd=str(cam_dir / 'flow_bwd'),
                                        start_offset=0,
                                        total_frames=-1,
                                        which='bwd',
                                        rotate=rotate,
                                    )
                                ),
                                TaskSpec(
                                    name='preprocessing.depth.filter_depth',
                                    kwargs=dict(
                                        filter_type='bilateral_temporal',
                                        depth_dir=str(cam_dir / 'depth_aligned'),
                                        color_dir=str(cam_dir / 'color'),
                                        mask_dir=str(cam_dir / 'mask'),
                                        out_dir=str(cam_dir / 'depth_filtering_bilateral_temporal'),
                                        flow_dir=str(cam_dir / 'flow_bwd'),
                                        start_offset=0,
                                        total_frames=-1,
                                        depth_format='png'
                                    )
                                ),
                            ]
                        )
                    ]
                )
                cam_tasks = ChordSpec(
                    header=header,
                    body=body
                )
            else:
                cam_tasks = mask_generation_tasks

            all_cam_tasks.append(cam_tasks)

        preprocessing_stage = Stage(
            name='preprocess',
            parts=all_cam_tasks
        )
        self._pipeline.stages.append(preprocessing_stage)
        return self

    def reconstruct(self, calibration_session_name: str, start_frame: int, total_frames: int, cam_idx: List[int], force: bool = False) -> 'SyncedSession':
        if total_frames == 0:
            log(f"[{self.session_name}::reconstruct] Total frames is 0. Skipping reconstruction.", 'warning')
            return self

        all_tasks = []
        for recon_type in ['pcd', 'gs']:
            for depth_source in ['bilateral_temporal', 'stereo']:
                reconstruction_task = TaskSpec(
                    name=f'reconstruction.{recon_type}_reconstruction',
                    kwargs=dict(
                        capturestudio_cache_root=str(self.capturestudio_cache_path),
                        calibration_session_name=calibration_session_name,
                        depth_source=depth_source,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        cam_idx=cam_idx,
                        excel_data=self.excel_data.to_dict(),
                        force=force
                    )
                )
                teaser_task = TaskSpec(
                    name='reconstruction.generate_teaser_video',
                    kwargs=dict(
                        capturestudio_cache_root=str(self.capturestudio_cache_path),
                        recon_type=recon_type,
                        depth_source=depth_source,
                        disparity_estimator_model='raftstereo',
                        start_frame=start_frame,
                        total_frames=total_frames,
                        force=force
                    )
                )
                branch_tasks = ChainSpec(parts=[reconstruction_task, teaser_task])
                all_tasks.append(branch_tasks)

        teaser_grid_task = TaskSpec(
            name='reconstruction.generate_teaser_grid_video',
            kwargs=dict(
                capturestudio_cache_root=str(self.capturestudio_cache_path),
                recon_types=['pcd', 'gs'],
                depth_sources=['bilateral_temporal', 'stereo'],
                disparity_estimator_model='raftstereo',
                start_frame=start_frame,
                total_frames=total_frames,
                force=force,
            )
        )
        link_to_webapp_task = TaskSpec(
            name='reconstruction.link_to_webapp',
            kwargs=dict(
                capturestudio_cache_root=str(self.capturestudio_cache_path),
                webapp_public_root=str(PathUtils.results_path())
            )
        )
        reconstruction_stage = Stage(
            name='reconstruct',
            parts=[
                ChordSpec(
                    header=GroupSpec(parts=all_tasks),
                    body=GroupSpec(parts=[
                        teaser_grid_task,
                        link_to_webapp_task,
                    ])
                )
            ]
        )
        self._pipeline.stages.append(reconstruction_stage)
        return self

    def export_tasks_graph(self, export_path: Union[Path, str, Literal['auto']] = 'auto', export_format: Literal['svg', 'pdf', 'png'] = 'pdf') -> 'SyncedSession':
        if export_path == 'auto':
            export_path = self.capturestudio_cache_path / f'pipeline_{datetime.now().strftime("%d_%m_%Y__%H_%S")}.{export_format}'
        self._pipeline.to_svg(svg_path=export_path)
        return self

    def to_celery(self, export_graph: bool = False, export_format: Literal['svg', 'pdf', 'png'] = 'pdf') -> 'SyncedSession':
        if not self._pipeline.stages:
            raise ValueError("Pipeline has no stages. Please add tasks before submitting.")

        if export_graph:
            self.export_tasks_graph(export_path='auto', export_format=export_format)

        from tasks import app
        app.loader.import_default_modules()
        self._celery_pipeline = self._pipeline.to_celery(app=app)
        return self

    def submit(self) -> AsyncResult:
        """
        Submits the pipeline to the Celery queue and returns the AsyncResult.
        This method will compile the pipeline to a Celery signature and submit it.
        """
        if self._celery_pipeline is None:
            raise ValueError("Pipeline has not been converted to Celery signature. Please call `to_celery()` before submitting.")

        result = self._celery_pipeline.apply_async()
        log(f"[{self.session_name}] Submitted pipeline with ID {result.id}.", 'info')
        return result

    def __repr__(self):
        return (f"SyncedSession(\n"
                f"\tsession_name={self.session_name},\n"
                f"\texcel_data={self.excel_data.to_dict()},\n"
                f"\tnas_root={self.nas_root},\n"
                f"\tnas_path={self.nas_path},\n"
                f"\tnas_storage_type={self.nas_storage_type},\n"
                f"\tnas_cache_root={self.nas_cache_root},\n"
                f"\tnas_cache_storage_type={self.nas_cache_storage_type},\n"
                f"\tcapturestudio_cache_root={self.capturestudio_cache_root},\n"
                f"\tcapturestudio_cache_path={self.capturestudio_cache_path},\n"
                f"\tcapturestudio_cache_storage_type={self.capturestudio_cache_storage_type}\n"
                f")")


@click.command()
@click.option(
    "--performer-name",
    prompt="Enter the name of the participant",
    help="The name of the participant.",
)
@click.option(
    "--performance-index",
    type=int,
    default=1,
    prompt="Enter the index of the performance",
    show_default=True,
    help="The index of the performance.",
)
@click.option(
    "--n-cameras",
    type=int,
    default=12,
    prompt="Total number of cameras",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--n-depth-cameras",
    type=click.Choice(["6", "8", "10", "12"]),
    default="6",
    prompt="Number of depth cameras",
    show_default=True,
    show_choices=True,
    help="Number of depth cameras used in the session.",
)
@click.option(
    "--primary-camera",
    type=click.Choice(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "freerun"]),
    default="2",
    prompt="Primary camera (or freerun)",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--is-calibration",
    type=click.Choice(["Yes", "No"], case_sensitive=False),
    default="No",
    prompt="Is this a calibration run?",
    show_default=True,
    show_choices=True,
    help="Whether this is a calibration run or not.",
)
@click.option(
    "--calibration-pattern",
    type=click.Choice(["Charuco", "Checkerboard"], case_sensitive=False),
    default="Charuco",
    prompt="Calibration pattern (for calibration runs only)",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--storage-batch-size",
    type=int,
    default=30,
    prompt="Storage batch size",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--use-h5",
    type=click.Choice(["Yes", "No"], case_sensitive=False),
    default="Yes",
    prompt="Store batches using HDF5?",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--color-resolution",
    type=click.Choice(["2160p", "1080p"]),
    default="2160p",
    prompt="Resolution of the color cameras",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--color-fps",
    type=click.Choice(["30", "15"]),
    default="30",
    prompt="FPS of the color cameras",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--depth-resolution",
    type=click.Choice(["576p", "1024p"]),
    default="576p",
    prompt="Resolution of the depth cameras",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--depth-fps",
    type=click.Choice(["30", "15"]),
    default="30",
    prompt="FPS of the depth cameras",
    show_default=True,
    show_choices=True,
)
@click.option(
    "--sony-id",
    type=click.types.STRING,
    default="None",
    prompt="Sony ID",
    show_default=True,
)
def ask_session_info(performer_name, performance_index, n_cameras, n_depth_cameras, primary_camera, is_calibration, calibration_pattern, storage_batch_size, use_h5, color_resolution, color_fps, depth_resolution, depth_fps, sony_id):
    click.echo("------------------------------------------------------------------------------")
    click.echo(f"Session Information:")
    click.echo(f"Performer's Name: {performer_name}")
    click.echo(f"Performance Index: {performance_index}")
    click.echo(f"Number of Cameras: {n_cameras}")
    click.echo(f"Number of Depth Cameras: {n_depth_cameras}")
    click.echo(f"Primary Camera: {primary_camera}" if primary_camera != "freerun" else "Using FREERUN mode for synchronization")
    click.echo(f"Calibration Run: {is_calibration}")
    if is_calibration.lower() == "yes":
        click.echo(f"Calibration Pattern: {calibration_pattern}")
    click.echo(f"Storage Batch Size: {storage_batch_size}")
    click.echo(f"Store using HDF5: {use_h5}")
    click.echo(f"Color Resolution: {color_resolution}")
    click.echo(f"Color FPS: {color_fps}")
    click.echo(f"Depth Resolution: {depth_resolution}")
    click.echo(f"Depth FPS: {depth_fps}")
    click.echo(f"Sony ID: {sony_id}")
    assert not (depth_fps == "30" and depth_resolution == "1024p"), "30 FPS is not supported for 1024p depth resolution. Please select 15 FPS instead."
    return dict(
        performer_name=performer_name,
        performance_index=performance_index,
        num_cameras=n_cameras,
        n_depth_cameras=int(n_depth_cameras),
        primary_camera=primary_camera if primary_camera.lower() != "freerun" else None,
        is_calibration=is_calibration.lower() == "yes",
        calibration_pattern_type=calibration_pattern.lower(),
        storage_batch_size=int(storage_batch_size),
        store_using_h5=use_h5.lower() == "yes",
        use_4k_color=color_resolution == "2160p",
        use_1k_depth=depth_resolution == "1024p",
        color_fps=int(color_fps),
        depth_fps=int(depth_fps),
        sync_color_fps_to_depth=int(color_fps) == int(depth_fps),
        sony_id=sony_id if sony_id.lower() != "none" else None,
    )


if __name__ == "__main__":
    # Get session information from user
    print('')
    print('------------------------------------------------------------------------------')
    user_info_ = ask_session_info(standalone_mode=False)
    session_kwargs_ = {k: v for k, v in user_info_.items() if k in ['is_calibration', 'calibration_pattern_type', 'sony_id', 'performer_name', 'performance_index']}
    system_kwargs_ = {k: v for k, v in user_info_.items() if k not in ['is_calibration', 'calibration_pattern_type']}
    print('------------------------------------------------------------------------------')
    # print(f"Session kwargs: {session_kwargs_}")
    # print(f"System kwargs: {system_kwargs_}")
    # exit(0)

    # Initialize system
    capture_system_ = CaptureSystem(
        # nas_root='/1-VoViCa/Captures_March_2025',  # use Synology API
        # nas_root=r'Z:\Captures_March_2025',  # use local filesystem to access NAS
        **system_kwargs_,
    )

    # Create session with live monitoring
    session_ = CaptureSession(
        capture_system_,
        live_monitor=True,
        **session_kwargs_,
    )

    # Start capture
    try:
        session_.start()
        log('Capture started - press Ctrl+C to stop...', 'info')
        # playsound(PathUtils.resources_path() / "audio" / "start.mp3", block=False)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log('Stopping capture...', 'info')
    finally:
        session_.stop()
        log('Capture stopped successfully', 'info')

    # Load session for post-capture processing
    log(f"Loading capture session", 'debug')
    try:
        CaptureSession.generate_session_videos(capture_system_.fs.get_root(), 100)
    except Exception as e:
        log(f"Failed to generate session videos: {e}", 'error', exc_info=e)
        exit(1)
    exit(0)

    # #------------------------------------------------------------------------------------------------------------#
    # # Generate videos from all captures
    # # -----------------------------------------------------------------------------------------------------------#

    # sp = Path(r'C:\Users\Thanos\PycharmProjects\capturestudio\out\Captures\l12_d6_s2x2_c12_synced_225_@02_05_2025_14_39_38')
    # CaptureSession.generate_session_videos(sp, start_frame_from_end=1939, total_frames=1537)

    sp = Path(r'C:\Users\Thanos\PycharmProjects\capturestudio\out\Captures\l12_d6_s2x2_c12_synced_225_@02_05_2025_15_33_51')
    CaptureSession.generate_session_videos(sp, start_frame_from_end=2468, total_frames=2102)

    # sp = Path(r'C:\Users\Thanos\PycharmProjects\capturestudio\out\Captures\l12_d6_s2x2_c12_synced_225_@01_05_2025_16_50_51')
    # CaptureSession.generate_session_videos(sp, start_frame_from_end=4598, total_frames=2099)

    # for sp in (PathUtils.out_path() / 'Captures').glob('l12_d6_s2x2_c12_synced_225_@05_05_2025_14_38_2*'):
    #     if not sp.is_dir() or not (sp / 'session_metadata.json').exists():
    #         continue
    #     CaptureSession.generate_session_videos(sp, total_frames=300)

    # ------------------------------------------------------------------------------------------------------------#
    # Access captured data
    # -----------------------------------------------------------------------------------------------------------#
    """
    Here is a sample code snippet to access the captured data:

    ```
    capture_path = Path('<path_to_capture_session>')
    log(f"Processing capture session: {capture_path}", 'info')
    with open(capture_path / 'session_metadata.json', 'r') as fp:
        session_state = json.load(fp)
    session_restored = CaptureSession.from_state(session_state)
    capture_system_restored = session_restored.system
    datasets = session_restored.load_session(capture_system_restored.fs.get_root(), mode='hdf5' if capture_system_restored.store_using_h5 else 'raw', fps_synced=capture_system_restored.color_fps == capture_system_restored.depth_fps)
    multi_dataset = MultiCamDataset(datasets, common_point='at_end')
    for colors_t, depths_t, color_tss, depth_tss in multi_dataset:
        # colors_t: (N, C, H, W) 
        # depths_t: (N, Hd, Wd) 
        # color_tss: (N,)
        # depth_tss: (N,)
        pass
    ```
    """
