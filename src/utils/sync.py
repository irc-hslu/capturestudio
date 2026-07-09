import os
import shutil
import struct
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple

from utils.misc import log


class SyncUtils:
    @classmethod
    def generate_multisync_info(
            cls,
            capture_path: str,
            hosts: Optional[Dict[int, Dict[str, Union[List[int], int, float, bool]]]] = None,
            tolerance_ms: int = 15,
            max_clusters: int = 100_000,
            info_name: str = 'multi_sync.info',
            force: bool = False,
    ):
        """
        Generate Orbbec multi-sync info using host-corrected (virtual) timestamps.

        Host correction:

            virtual_ts = raw_ts + offset_ms + drift_ms

        where:

            drift_ms = round((raw_ts - drift_anchor_ts) * drift_ppm / 1_000_000)

        hosts example:

            {
                0: {"cams": [1, 2, 4], "offset_ms": 0, "is_master": True},
                1: {"cams": [3, 5], "offset_ms": -1149},
            }

        The binary record stores raw_ts and write_idx, to be loaded by load_multisync_info().
        Clustering uses virtual_ts.
        """
        capture_path = Path(capture_path)
        orbbec_path = capture_path / 'orbbec'
        info_path = orbbec_path / info_name

        assert orbbec_path.exists(), f"Missing Orbbec root: {orbbec_path}"
        assert tolerance_ms >= 0, f"Invalid tolerance_ms={tolerance_ms}"
        assert max_clusters > 0, f"Invalid max_clusters={max_clusters}"

        cam_dirs = sorted(
            [d for d in orbbec_path.iterdir() if d.is_dir() and d.name.startswith('cam')],
            key=lambda d: int(d.name.split('cam')[1].split()[0]),
        )
        assert cam_dirs, f"No camXX folders found in {orbbec_path}"

        cam_indices = [
            int(d.name.split('cam')[1].split()[0])
            for d in cam_dirs
        ]

        if info_path.exists():
            if not force:
                return None
            info_path.unlink()

        cam_to_host = {}
        cam_to_offset = {cam_idx: 0 for cam_idx in cam_indices}
        cam_to_drift_ppm = {cam_idx: 0.0 for cam_idx in cam_indices}
        cam_to_drift_anchor_ts = {cam_idx: None for cam_idx in cam_indices}

        if hosts is not None:
            seen = set()

            for host_idx, spec in hosts.items():
                assert 'cams' in spec, f"hosts[{host_idx}] missing 'cams'"

                host_idx = int(host_idx)
                offset_ms = int(spec.get('offset_ms', 0))
                drift_ppm = float(spec.get('drift_ppm', 0.0))
                drift_anchor_ts = spec.get('drift_anchor_ts', None)

                if drift_ppm != 0.0:
                    assert drift_anchor_ts is not None, (
                        f"hosts[{host_idx}] has drift_ppm={drift_ppm} but no drift_anchor_ts."
                    )
                    drift_anchor_ts = int(drift_anchor_ts)

                for cam_idx in spec['cams']:
                    cam_idx = int(cam_idx)

                    assert cam_idx in cam_indices, f"cam{cam_idx} from hosts[{host_idx}] not found."
                    assert cam_idx not in seen, f"cam{cam_idx} appears in multiple hosts."

                    seen.add(cam_idx)
                    cam_to_host[cam_idx] = host_idx
                    cam_to_offset[cam_idx] = offset_ms
                    cam_to_drift_ppm[cam_idx] = drift_ppm
                    cam_to_drift_anchor_ts[cam_idx] = drift_anchor_ts

            missing = sorted(set(cam_indices) - seen)
            assert not missing, f"These cameras are missing from hosts: {missing}"

        events = []

        for cam_dir in cam_dirs:
            cam_idx = int(cam_dir.name.split('cam')[1].split()[0])
            offset_ms = int(cam_to_offset[cam_idx])
            drift_ppm = float(cam_to_drift_ppm[cam_idx])
            drift_anchor_ts = cam_to_drift_anchor_ts[cam_idx]

            color_paths = sorted((cam_dir / 'color').glob('*.jpg'), key=lambda p: int(p.stem))

            for write_idx, p in enumerate(color_paths):
                raw_ts = int(p.stem)

                drift_ms = 0
                if drift_ppm != 0.0:
                    drift_ms = int(round((raw_ts - int(drift_anchor_ts)) * drift_ppm / 1_000_000.0))

                virtual_ts = int(raw_ts + offset_ms + drift_ms)
                assert virtual_ts >= 0, (
                    f"Negative virtual timestamp for cam{cam_idx}: "
                    f"raw_ts={raw_ts}, offset_ms={offset_ms}, drift_ms={drift_ms}, "
                    f"virtual_ts={virtual_ts}"
                )

                events.append({
                    'cam_idx': int(cam_idx),
                    'raw_ts': int(raw_ts),
                    'virtual_ts': int(virtual_ts),
                    'write_idx': int(write_idx),
                    'path': p,
                })

            log(
                f"[SyncUtils::generate_multisync_info] cam{cam_idx:02d}: "
                f"frames={len(color_paths)}, "
                f"host={cam_to_host.get(cam_idx, None)}, "
                f"offset_ms={offset_ms}, "
                f"drift_ppm={drift_ppm}, "
                f"drift_anchor_ts={drift_anchor_ts}",
                'debug',
            )

        events = sorted(events, key=lambda e: (e['virtual_ts'], e['cam_idx'], e['raw_ts']))

        clusters = []
        active = []

        for e in events:
            ts = int(e['virtual_ts'])
            cam_idx = int(e['cam_idx'])

            active = [
                ci for ci in active
                if clusters[ci]['max_ts'] >= ts - tolerance_ms
            ]

            best_ci = None
            best_key = None

            for ci in active:
                cl = clusters[ci]

                if cam_idx in cl['members']:
                    continue

                new_min = min(int(cl['min_ts']), ts)
                new_max = max(int(cl['max_ts']), ts)
                spread = new_max - new_min

                if spread > tolerance_ms:
                    continue

                key = (
                    spread,
                    -len(cl['members']),
                    abs(ts - ((int(cl['min_ts']) + int(cl['max_ts'])) * 0.5)),
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best_ci = ci

            if best_ci is None:
                clusters.append({
                    'min_ts': ts,
                    'max_ts': ts,
                    'members': {
                        cam_idx: e,
                    },
                })
                active.append(len(clusters) - 1)
            else:
                cl = clusters[best_ci]
                cl['members'][cam_idx] = e
                cl['min_ts'] = min(int(cl['min_ts']), ts)
                cl['max_ts'] = max(int(cl['max_ts']), ts)

        for ci, cl in enumerate(clusters[:max_clusters]):
            vals = [int(e['virtual_ts']) for e in cl['members'].values()]
            spread = max(vals) - min(vals)

            assert spread <= tolerance_ms, (
                f"Invalid virtual cluster {ci}: spread={spread}, tolerance={tolerance_ms}, "
                f"members={{{', '.join([str(k) + ': raw=' + str(v['raw_ts']) + ', virtual=' + str(v['virtual_ts']) for k, v in sorted(cl['members'].items())])}}}"
            )

        payload = bytearray()
        written_clusters = 0

        for cl in clusters:
            if written_clusters >= max_clusters:
                break

            payload += struct.pack(
                'QQI',
                int(cl['min_ts']),
                int(cl['max_ts']),
                len(cl['members']),
            )

            for cam_idx in sorted(cl['members'].keys()):
                e = cl['members'][cam_idx]

                payload += struct.pack(
                    'IQQ',
                    int(cam_idx),
                    int(e['raw_ts']),
                    int(e['write_idx']),
                )

            written_clusters += 1

        info_path.parent.mkdir(parents=True, exist_ok=True)

        with open(info_path, 'wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        full_clusters = sum(
            1
            for cl in clusters[:written_clusters]
            if sorted(cl['members'].keys()) == sorted(cam_indices)
        )

        log(
            f"[SyncUtils::generate_multisync_info] wrote {info_path}: "
            f"clusters={written_clusters}/{len(clusters)}, "
            f"full_clusters={full_clusters}, "
            f"cams={cam_indices}, "
            f"hosts={'yes' if hosts is not None else 'no'}, "
            f"tolerance_ms={tolerance_ms}",
            'info',
        )

        return None

    @classmethod
    def load_multisync_info(cls, capture_path: str, file_stem: str = 'multi_sync') -> List[Dict[str, Union[int, Dict[int, Tuple[Path, str, int]]]]]:
        """
        Load multi-sync info from file.

        Parameters
        ----------
        capture_path: str
            Path to the capture session root directory, e.g. "/mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1".
        file_stem: str
            Stem of the file to be loaded. Default: "multi_sync" (so the file to be loaded is "multi_sync.info").

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
        if not (capture_path / f'{file_stem}.info').exists():
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
            cam_name_int: sorted(list((cam_dir / 'depth').glob('*.npy' if len(list((cam_dir / 'depth').glob('*.npy'))) > 0 else '*.png')), key=lambda p: int(p.stem)) if (cam_dir / 'depth').exists() else []
            for cam_dir, cam_name_int in zip(all_cam_dirs, all_cam_names_int)
        }
        cluster_records = []
        with open(capture_path / 'orbbec' / f'{file_stem}.info', 'rb') as f:
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


if __name__ == '__main__':
    for calib_, offset_ in zip([1, 2], [-1149, -1184]):
        p_ = f'/home/charisoudis/capturestudio/data/Cagliari_2_5cams_Calib_{calib_}'
        SyncUtils.generate_multisync_info(
            p_,
            hosts={
                0: {"cams": [1, 2, 4], "offset_ms": 0, "is_master": True},
                1: {"cams": [3, 5], "offset_ms": offset_},
            },
            force=True
        )
        SyncUtils.synchronize_frames_using_timestamps(
            p_
        )
        from preprocessing.generate_video import generate_multiview_video

        generate_multiview_video(
            p_,
            'color', 'jpg'
        )
        generate_multiview_video(
            p_,
            'depth', 'png'
        )
