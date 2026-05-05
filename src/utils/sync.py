import os
import shutil
import struct
from bisect import bisect_right, bisect_left
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple

from utils.misc import log


class SyncUtils:
    @classmethod
    def generate_multisync_info(cls, capture_path: str, max_clusters: int = 100_000):
        """
        Build multi-camera synchronization info file from color frames.

        Parameters
        ----------
        capture_path: str
            Path to the capture session root directory, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".
        max_clusters: int, optional
            Maximum number of clusters to write to the info file (default: 100K). A cluster holds the frame timestamps for all frames corresponding to a single point in wall clock time.
        """
        capture_path = Path(capture_path)
        if (capture_path / 'multi_sync.info').exists():
            return None

        # collect timestamps and paths for each camera
        per_cam_ts, per_cam_frame_count = [], []
        all_cam_dirs = sorted([d for d in (capture_path / 'orbbec').iterdir() if d.is_dir() and d.name.startswith('cam')], key=lambda d: int(d.name.split('cam')[1].split(' ')[0]))
        for cam_dir in all_cam_dirs:
            ts_list = []
            paths = sorted((cam_dir / 'color').glob('*.jpg'), key=lambda path: int(path.stem))
            frame_count = 0
            for p in paths:
                ts_list.append((int(p.stem), (p, p.stem, frame_count)))
                frame_count += 1
            per_cam_ts.append(ts_list)
            per_cam_frame_count.append(frame_count)
        # perform multi-camera synchronization
        clusters = []
        unflushed = []
        for cam_name, ts_list in zip([_.name for _ in all_cam_dirs], per_cam_ts):
            cam_idx = int(cam_name.split(' ')[0].replace('cam', ''))
            for ts, info in ts_list:
                if not clusters:
                    clusters.append({'min_ts': ts, 'max_ts': ts, 'members': {cam_idx: info}, 'flushed': False})
                    unflushed.append(0)
                    continue
                last_idx = len(clusters) - 1
                last = clusters[last_idx]
                if ts > last['max_ts'] + 15:
                    cutoff = ts - 15
                    to_flush = [i for i in unflushed if clusters[i]['max_ts'] < cutoff]
                    for i in sorted(to_flush):
                        unflushed.remove(i)
                        clusters[i]['flushed'] = True
                    clusters.append({'min_ts': ts, 'max_ts': ts, 'members': {cam_idx: info}, 'flushed': False})
                    unflushed.append(len(clusters) - 1)
                else:
                    if (last['min_ts'] - 15) <= ts <= (last['max_ts'] + 15):
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
                            if (cl['min_ts'] - 15) <= ts <= (cl['max_ts'] + 15):
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

        # create file bytes
        payload = bytearray()
        cnt = 0
        for cl in clusters:
            payload += struct.pack('QQI', int(cl['min_ts']), int(cl['max_ts']), len(cl['members']))
            for cam in sorted(cl['members'].keys()):
                p, ts_val, write_idx = cl['members'][cam]
                # print('write_idx', write_idx, 'ts_val', ts_val, 'cam', cam, 'p', p)
                payload += struct.pack('IQQ', int(cam), int(ts_val), int(write_idx))
            cnt += 1
            if cnt >= max_clusters:
                break
        # write the payload to file
        with open(capture_path / 'orbbec' / 'multi_sync.info', 'wb') as f:
            f.write(payload)  # single write
            f.flush()  # empty kernel buffers
            os.fsync(f.fileno())  # write to disk
        return None

    @classmethod
    def load_multisync_info(cls, capture_path: str) -> List[Dict[str, Union[int, Dict[int, Tuple[Path, str, int]]]]]:
        """
        Load multi-sync info from file.

        Parameters
        ----------
        capture_path: str
            Path to the capture session root directory, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".

        Returns
        -------
        List[Dict[str, Union[int, Dict[int, Tuple[Path, str, int]]]]]:
            List of clusters, each containing:
            - 'min_ts': Minimum timestamp in the cluster.
            - 'max_ts': Maximum timestamp in the cluster.
            - 'members': Dictionary mapping camera indices to tuples of (color_path, depth_path)
                or (timestamp, write_index) if not raw.
        """
        capture_path = Path(capture_path)
        if not (capture_path / 'multi_sync.info').exists():
            cls.generate_multisync_info(str(capture_path))

        metadata_size = struct.calcsize('QQI')  # min_ts, max_ts, num_cams
        per_cam_data_size = struct.calcsize('IQQ')  # cam_idx, ts, write_idx

        all_cam_dirs = sorted([d for d in (capture_path / 'orbbec').iterdir() if d.is_dir() and d.name.startswith('cam')], key=lambda d: int(d.name.split('cam')[1].split(' ')[0]))
        all_cam_names_int = [int(d.name.split('cam')[1].split(' ')[0]) for d in all_cam_dirs]
        color_paths = {
            cam_name_int: sorted(list((cam_dir / 'color').glob('*.jpg')), key=lambda p: int(p.stem))
            for cam_dir, cam_name_int in zip(all_cam_dirs, all_cam_names_int)
        }
        depth_paths = {
            cam_name_int: sorted(list((cam_dir / 'depth').glob('*.npy' if len(list((cam_dir / 'depth').glob('*.npy'))) > 0 else '.png')), key=lambda p: int(p.stem)) if (cam_dir / 'depth').exists() else []
            for cam_dir, cam_name_int in zip(all_cam_dirs, all_cam_names_int)
        }
        cluster_records = []
        with open(capture_path / 'orbbec' / 'multi_sync.info', 'rb') as f:
            while True:
                metadata_packed = f.read(metadata_size)
                if not metadata_packed:
                    break
                min_ts, max_ts, num_cams = struct.unpack('QQI', metadata_packed)
                if num_cams == 0:
                    continue
                members: Dict[int, Tuple[Optional[Path], Optional[Path]]] = {
                    cam_idx: (None, None)
                    for cam_idx in all_cam_names_int
                }
                found_cam_names_int = []
                for i in range(num_cams):
                    cam_idx, ts_val, write_idx = struct.unpack('IQQ', f.read(per_cam_data_size))
                    if cam_idx in color_paths:
                        members[cam_idx] = color_paths[cam_idx][write_idx], depth_paths[cam_idx][write_idx] if len(depth_paths[cam_idx]) > write_idx else None
                        found_cam_names_int.append(cam_idx)
                if sorted(found_cam_names_int) == sorted(all_cam_names_int):
                    cluster_records.append(dict(sorted(members.items(), key=lambda x: int(x[0]))))
        return cluster_records  # {<cam_idx_int>: (<color_path>, <depth_path>)}

    @classmethod
    def move_unsynced_frames_by_first_and_last_index(cls, cam_dir: Path, first_index: int, last_index: int, color_extension: str = 'jpg', depth_extension: str = 'png', rename_instead_of_moving: bool = False):
        """
        Move unsynchronized frames to a new directory.

        Convert the below args to numpy format:
        Parameters
        ----------
        cam_dir: Path
            Path to the camera directory containing frames (e.g., "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1/orbbec/cam05").
        first_index: int
            The first synced frame index (inclusive).
        last_index: int
            The last synced frame index (exclusive).
        color_extension: str, optional
            The extension of color frames (default: 'jpg').
        depth_extension: str, optional
            The extension of depth frames (default: 'png').
        rename_instead_of_moving: bool, optional
            Whether to rename unsynced frames instead of moving them (default: False). If true, the unsynced frames will be renamed to "<original_name>.unsync".
        """
        color_extension = color_extension.strip('.')
        depth_extension = depth_extension.strip('.')
        # Get all frame files sorted numerically
        color_frames = sorted((cam_dir / 'color').glob(f"*.{color_extension}"), key=lambda x: int(x.stem))
        if first_index < 0:
            first_index = len(color_frames) + first_index
        if last_index <= 0:
            last_index = len(color_frames) + last_index
        # Move unsynchronized frames
        for color_frame_i, color_frame in enumerate(color_frames):
            if not (first_index <= color_frame_i < last_index):
                if not (cam_dir / "color_unsynced").exists():
                    (cam_dir / "color_unsynced").mkdir(parents=True, exist_ok=True)
                if rename_instead_of_moving:
                    new_name = color_frame.with_suffix(f'.{color_extension}.unsync')
                    if not new_name.exists():
                        color_frame.rename(new_name)
                else:
                    shutil.move(str(color_frame), str(cam_dir / "color_unsynced" / color_frame.name))
                for folder in ['mask|jpg', 'flow_fwd|png', 'flow_bwd|png']:
                    folder, ext = folder.split('|', 1)
                    folder_file_path = cam_dir / folder / color_frame.with_suffix(f'.{ext}').name
                    if folder_file_path.exists():
                        if rename_instead_of_moving:
                            new_name = folder_file_path.with_suffix(f'.{ext}.unsync')
                            if not new_name.exists():
                                folder_file_path.rename(new_name)
                        else:
                            folder_file_path.unlink()  # remove processed files corresponding to the unsynced color frame
        if (cam_dir / 'depth').exists():
            depth_extension_raw = 'npy'
            depth_frames = sorted((cam_dir / 'depth').glob(f"*.{depth_extension_raw}"), key=lambda x: int(x.stem))
            if len(depth_frames) == 0:
                depth_extension_raw = 'png'
                depth_frames = sorted((cam_dir / 'depth').glob(f"*.{depth_extension_raw}"), key=lambda x: int(x.stem))
                if len(depth_frames) == 0:
                    log(f"No depth frames found in {cam_dir / 'depth'} with extensions {depth_extension_raw} or {depth_extension}. Removing depth directory.", 'warning')
                    # rename directory to .empty
                    os.rename(cam_dir / 'depth', cam_dir / 'depth.empty')
                    return None
                assert depth_extension == 'png', f"Expected depth extension to be 'png', but got '{depth_extension_raw}'. Please check the directory."
            for depth_frame_i, depth_frame in enumerate(depth_frames):
                if not (first_index <= depth_frame_i < last_index):
                    if not (cam_dir / "depth_unsynced").exists():
                        (cam_dir / "depth_unsynced").mkdir(parents=True, exist_ok=True)
                    if rename_instead_of_moving:
                        new_name = depth_frame.with_suffix(f'.{depth_extension}.unsync')
                        if not new_name.exists():
                            depth_frame.rename(new_name)
                    else:
                        shutil.move(str(depth_frame), str(cam_dir / "depth_unsynced" / depth_frame.name))
                    for folder in ['depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal']:
                        folder_file_path = cam_dir / folder / depth_frame.with_suffix(f'.{depth_extension}').name
                        if folder_file_path.exists():
                            if rename_instead_of_moving:
                                new_name = folder_file_path.with_suffix(f'.{depth_extension}.unsync')
                                if not new_name.exists():
                                    folder_file_path.rename(new_name)
                            else:
                                folder_file_path.unlink()  # remove processed files corresponding to the unsynced depth frame
        return None

    @classmethod
    def move_unsynced_frames_by_idx(cls, cam_dir: Path, keep_idx: List[int], color_extension: str = 'jpg', depth_extension: str = 'png', rename_instead_of_moving: bool = False):
        """
        Move unsynchronized frames to a new directory.

        Convert the below args to numpy format:
        Parameters
        ----------
        cam_dir: Path
            Path to the camera directory containing frames (e.g., "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1/orbbec/cam05").
        keep_idx: List[int]
            The list of indices to keep.
        color_extension: str, optional
            The extension of color frames (default: 'jpg').
        depth_extension: str, optional
            The extension of depth frames (default: 'png').
        rename_instead_of_moving: bool, optional
            Whether to rename unsynced frames instead of moving them (default: False). If true, the unsynced frames will be renamed to "<original_name>.unsync".
        """
        color_extension = color_extension.strip('.')
        depth_extension = depth_extension.strip('.')
        # Get all frame files sorted numerically
        color_frames = sorted((cam_dir / 'color').glob(f"*.{color_extension}"), key=lambda x: int(x.stem))
        # Move unsynchronized frames
        for color_frame_i, color_frame in enumerate(color_frames):
            if color_frame_i not in keep_idx:
                if rename_instead_of_moving:
                    new_name = color_frame.with_suffix(f'.{color_extension}.unsync')
                    if not new_name.exists():
                        color_frame.rename(new_name)
                else:
                    if not (cam_dir / "color_unsynced").exists():
                        (cam_dir / "color_unsynced").mkdir(parents=True, exist_ok=True)
                    shutil.move(str(color_frame), str(cam_dir / "color_unsynced" / color_frame.name))
                for folder in ['mask|jpg', 'flow_fwd|png', 'flow_bwd|png']:
                    folder, ext = folder.split('|', 1)
                    folder_file_path = cam_dir / folder / color_frame.with_suffix(f'.{ext}').name
                    if folder_file_path.exists():
                        if rename_instead_of_moving:
                            new_name = folder_file_path.with_suffix(f'.{ext}.unsync')
                            if not new_name.exists():
                                folder_file_path.rename(new_name)
                        else:
                            folder_file_path.unlink()  # remove processed files corresponding to the unsynced color frame
        if (cam_dir / 'depth').exists():
            depth_extension_raw = 'npy'
            depth_frames = sorted((cam_dir / 'depth').glob(f"*.{depth_extension_raw}"), key=lambda x: int(x.stem))
            if len(depth_frames) == 0:
                depth_extension_raw = 'png'
                depth_frames = sorted((cam_dir / 'depth').glob(f"*.{depth_extension_raw}"), key=lambda x: int(x.stem))
                if len(depth_frames) == 0:
                    log(f"No depth frames found in {cam_dir / 'depth'} with extensions {depth_extension_raw} or {depth_extension}. Removing depth directory.", 'warning')
                    # rename directory to .empty
                    os.rename(cam_dir / 'depth', cam_dir / 'depth.empty')
                    return None
                assert depth_extension == 'png', f"Expected depth extension to be 'png', but got '{depth_extension_raw}'. Please check the directory."
            for depth_frame_i, depth_frame in enumerate(depth_frames):
                if depth_frame_i not in keep_idx:
                    if not (cam_dir / "depth_unsynced").exists():
                        (cam_dir / "depth_unsynced").mkdir(parents=True, exist_ok=True)
                    if rename_instead_of_moving:
                        new_name = depth_frame.with_suffix(f'.{depth_extension}.unsync')
                        if not new_name.exists():
                            depth_frame.rename(new_name)
                    else:
                        shutil.move(str(depth_frame), str(cam_dir / "depth_unsynced" / depth_frame.name))
                    for folder in ['depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal']:
                        folder_file_path = cam_dir / folder / depth_frame.with_suffix(f'.{depth_extension}').name
                        if folder_file_path.exists():
                            if rename_instead_of_moving:
                                new_name = folder_file_path.with_suffix(f'.{depth_extension}.unsync')
                                if not new_name.exists():
                                    folder_file_path.rename(new_name)
                            else:
                                folder_file_path.unlink()  # remove processed files corresponding to the unsynced depth frame
        return None


    @classmethod
    def synchronize_frames_using_timestamps(cls, capture_path: str):
        clusters = cls.load_multisync_info(capture_path)
        if not clusters:
            raise RuntimeError(f"No clusters found in {capture_path}. Cannot synchronize frames.")
        capture_path = Path(capture_path)

        all_cam_dirs = sorted([d for d in (capture_path / 'orbbec').iterdir() if d.is_dir() and d.name.startswith('cam')], key=lambda d: int(d.name.split('cam')[1].split(' ')[0]))
        for cam_dir in all_cam_dirs:
            cam_idx = int(cam_dir.name.split('cam')[-1])
            all_color_ts = sorted(int(p.stem) for p in (cam_dir / 'color').glob('*.jpg'))
            keep_ts_cam = [int(Path(c[cam_idx][0]).stem) for c in clusters if cam_idx in c and c[cam_idx][0] is not None]
            keep_idx_cam = [all_color_ts.index(ts) for ts in keep_ts_cam if ts in all_color_ts]
            cls.move_unsynced_frames_by_idx(cam_dir, keep_idx_cam, rename_instead_of_moving=False)
        return None

    @classmethod
    def synchronize_frames_using_clap(cls, session_path: str, visual_sync_points: Dict[str, str], audio_sync_points: Optional[Dict[str, str]] = None, orbbec_ref_cam: str = "cam05", sync_using_audio: bool = False):
        """
        Synchronize frames across all cameras based on common frame points from Excel.
        Moves unsynchronized frames to 'color_unsynced' folders.

        Parameters
        ----------
        session_path: str
            Path to the session folder containing camera directories, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".
        visual_sync_points: Dict[str, str]
            Dictionary of flash-based sync points for each camera type (e.g., {"orbbec": <color_ts>, "sony": <color_t>, "apple": <color_t>}).
            The values are frame file stems, corresponding to the common sync frames.
        audio_sync_points: Dict[str, str], optional
            Dictionary of clap-based sync points for each camera type (e.g., {"orbbec": <color_ts>, "sony": <color_t>, "apple": <color_t> or "-"}).
            The values are frame file stems, corresponding to the common sync frames.
        orbbec_ref_cam: str, optional
            Reference Orbbec camera for sync (default: "cam05").
        sync_using_audio: bool, optional
            Whether to use audio sync points instead of visual sync points (default: False).
        """
        session_path = Path(session_path)
        # Read sync points from Excel
        sync_points = visual_sync_points if not sync_using_audio else audio_sync_points
        # Process all Orbbec cameras
        orbbec_cam_dirs = [d for d in (session_path / "orbbec").iterdir() if d.is_dir() and d.name.startswith("cam")]
        # the new common length is the number of frames from the sync frame to the end
        # we need to find the file that corresponds to the sync frame in the reference camera, and use its index to compute the common length
        #   - orbbecs common length is the number of frames from the sync frame to the end
        ref_cam_dir = next(cam_dir for cam_dir in orbbec_cam_dirs if cam_dir.name == orbbec_ref_cam)
        ref_cam_files = sorted(ref_cam_dir.glob("color/*.jpg"), key=lambda x: int(x.stem))
        ref_sync_file = next(f for f in ref_cam_files if int(f.stem) == int(sync_points['orbbec']))
        common_length = len(ref_cam_files) - ref_cam_files.index(ref_sync_file)
        # update the sync points for all cameras, by removing the frames that are not in the common length (from the common point at the end)
        first_camera_length = len(list((orbbec_cam_dirs[0] / "color").glob("*.jpg")))
        if not all(len(list((cam_dir / "color").glob("*.jpg"))) == first_camera_length for cam_dir in orbbec_cam_dirs):
            for cam_dir in orbbec_cam_dirs:
                cls.move_unsynced_frames_by_first_and_last_index(cam_dir, first_index=-common_length, last_index=0, color_extension='jpg', rename_instead_of_moving=True)
        return None
