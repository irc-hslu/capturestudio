import json
import logging
import multiprocessing as mp
import multiprocessing.shared_memory
import os
import signal
import struct
import subprocess
import time
from bisect import bisect_right
from collections import OrderedDict
from datetime import datetime
from multiprocessing import Process, Event
from multiprocessing.shared_memory import ShareableList
from pathlib import Path
from typing import Union, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pytz

try:
    from synology_api.filestation import FileStation
except ImportError:
    FileStation = None
from tqdm import tqdm

from utils.filesystem import LocalFilesystem, IFilesystem
from utils.misc import PathUtils, log, env_get

try:
    import pyorbbecsdk as porb
    from misc.orbbec import frame_to_bgr_image
except ImportError:
    log('PyORBBEC SDK not installed. Required for camera operations.', 'error')
    porb = None
    frame_to_bgr_image = None

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
logging.getLogger('urllib3').setLevel(logging.WARNING)


class SharedRingBuffer:
    """Lock-free ring buffer using shared memory with atomic indexes"""

    def __init__(self, name: str, num_slots: int, slot_size: int, idx2ts_list: Optional[ShareableList] = None, frame_count: Optional[mp.Value] = None, meta_format: str = '<QQI', meta_size_bytes: int = 20):
        self.name = name
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.buffer_size = num_slots * (slot_size + meta_size_bytes)  # 8 bytes for metadata
        self.frame_count = frame_count
        self.meta_format = meta_format
        self.meta_size_bytes = meta_size_bytes
        try:
            self.shm = mp.shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=self.buffer_size
            )
        except FileExistsError:
            self.shm = mp.shared_memory.SharedMemory(name=self.name)

        if idx2ts_list is not None:
            self.idx2ts_list = idx2ts_list
            self.idx2ts_list_len = len(self.idx2ts_list)
        else:
            self.idx2ts_list = None
            self.idx2ts_list_len = 0

        self.write_idx = mp.Value('Q', 0, lock=False)
        self.read_idx = mp.Value('Q', 0, lock=False)

    def write_frame(self, data: bytes, color_timestamp: int = 0, depth_timestamp: int = 0) -> bool:
        if (self.write_idx.value - self.read_idx.value) >= self.num_slots:
            logging.warning("Buffer overflow - dropping frame")
            return False

        slot = self.write_idx.value % self.num_slots
        offset = slot * (self.slot_size + self.meta_size_bytes)

        # Write timestamp for syncing
        if self.idx2ts_list is not None:
            self.idx2ts_list[self.write_idx.value] = color_timestamp

        # Write data
        metadata = struct.pack(self.meta_format, color_timestamp, depth_timestamp, len(data))
        self.shm.buf[offset:offset + self.meta_size_bytes] = metadata
        self.shm.buf[offset + self.meta_size_bytes:offset + self.meta_size_bytes + len(data)] = data
        self.write_idx.value += 1
        if self.frame_count is not None:
            self.frame_count.value += 1
        return True

    def read_frame(self):
        if self.read_idx.value >= self.write_idx.value:
            return None, None, None

        slot = self.read_idx.value % self.num_slots
        offset = slot * (self.slot_size + self.meta_size_bytes)

        metadata = self.shm.buf[offset:offset + self.meta_size_bytes]
        color_timestamp, depth_timestamp, data_len = struct.unpack(self.meta_format, metadata)
        data = bytes(self.shm.buf[offset + self.meta_size_bytes:offset + self.meta_size_bytes + data_len])
        self.read_idx.value += 1
        return data, color_timestamp, depth_timestamp

    def read_frame_at_index(self, idx: int):
        slot = idx % self.num_slots
        offset = slot * (self.slot_size + self.meta_size_bytes)
        metadata = self.shm.buf[offset:offset + self.meta_size_bytes]
        color_timestamp, depth_timestamp, data_len = struct.unpack(self.meta_format, metadata)
        data = bytes(self.shm.buf[offset + self.meta_size_bytes:offset + self.meta_size_bytes + data_len])
        return data, color_timestamp, depth_timestamp

    def close(self):
        self.shm.close()


class SharedRingBufferNas(SharedRingBuffer):
    def __init__(self, name: str, num_slots: int = 100_000, slot_size: int = 1_000, meta_format: str = '<I', meta_size_bytes: int = 4, delimiter: str = ';'):
        super().__init__(name, num_slots, slot_size, None, meta_format, meta_size_bytes)
        self.delimiter = delimiter

    def write_frame(self, *paths: str) -> bool:
        if (self.write_idx.value - self.read_idx.value) >= self.num_slots:
            log(f"[{self.__class__.__name__}::write_frame] Buffer overflow - dropping frame", 'error')
            return False

        slot = self.write_idx.value % self.num_slots
        offset = slot * (self.slot_size + self.meta_size_bytes)

        # Metadata: timestamp (unsigned int) and data length (unsigned int)
        data = self.delimiter.join([str(p) for p in paths]).encode('utf-8')
        metadata = struct.pack(self.meta_format, len(data))
        self.shm.buf[offset:offset + self.meta_size_bytes] = metadata
        self.shm.buf[offset + self.meta_size_bytes:offset + self.meta_size_bytes + len(data)] = data
        self.write_idx.value += 1
        return True

    def read_frame(self):
        if self.read_idx.value >= self.write_idx.value:
            return None

        slot = self.read_idx.value % self.num_slots
        offset = slot * (self.slot_size + self.meta_size_bytes)

        metadata = self.shm.buf[offset:offset + self.meta_size_bytes]
        data_len, = struct.unpack(self.meta_format, metadata)
        data = tuple(bytes(self.shm.buf[offset + self.meta_size_bytes:offset + self.meta_size_bytes + data_len]).decode('utf-8').split(self.delimiter))
        self.read_idx.value += 1
        return data

    def close(self):
        self.shm.close()


class NASUploader(Process):
    def __init__(self, buffers: List[SharedRingBufferNas], local_root: str, nas_root: str, experiment_name: str, cam_names: List[str], store_using_h5: bool, termination_event: Event):
        super().__init__(daemon=True)
        self.buffers = buffers
        self.local_root = local_root
        self.experiment_name = experiment_name
        self.cam_names = cam_names
        self.store_using_h5 = store_using_h5
        self.nas_root = str(nas_root)
        self.nas_fs = None
        if Path(nas_root).exists():
            self.nas_root = str(os.path.join(nas_root, experiment_name))
            self.nas_fs = LocalFilesystem(Path(self.nas_root))
        self.termination_event = termination_event

    def _poll(self) -> int:
        n_processed = 0
        for buffer in self.buffers:
            file_paths = buffer.read_frame()
            if file_paths is None or self.nas_fs is None:
                continue
            if isinstance(self.nas_fs, LocalFilesystem):
                # Copy file to NAS
                for file_path in file_paths:
                    self.nas_fs.store(file_path, file_path.replace(self.local_root, self.nas_root))
            elif isinstance(self.nas_fs, FileStation):
                for file_path in file_paths:
                    # Upload file to NAS using Synology API
                    nas_folder = file_path.replace(self.local_root, self.nas_root).replace('\\', '/').replace(file_path.split('/')[-1], '')
                    # print(f"Uploading {file_path} to NAS {nas_folder}")
                    self.nas_fs.upload_file(nas_folder, file_path, overwrite=False, create_parents=False, progress_bar=False)
                    # # Upload file to NAS using Synology API
                    # self.nas_fs.upload_file(file_path.replace(self.local_root, self.nas_root), file_path, overwrite=False, create_parents=False, progress_bar=False)
            n_processed += 1
        return n_processed

    # noinspection PyUnboundLocalVariable
    def run(self):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Create connection to NAS
        try:
            if isinstance(self.nas_fs, LocalFilesystem):
                log(f"[{self.__class__.__name__}::run] Connected to NAS attached locally at: \"{self.nas_fs.get_root()}\"", 'info')
                if not self.store_using_h5:
                    self.nas_fs.mkdir(self.nas_fs.get_root() / 'raw_color', parents=True)
                    self.nas_fs.mkdir(self.nas_fs.get_root() / 'raw_depth', parents=True)
                    self.nas_fs.mkdir(self.nas_fs.get_root() / 'raw_ir', parents=True)
                for cam_name in self.cam_names:
                    if self.store_using_h5:
                        self.nas_fs.mkdir(self.nas_fs.get_root() / cam_name)
                    else:
                        self.nas_fs.mkdir(self.nas_fs.get_root() / 'raw_color' / cam_name)
                        self.nas_fs.mkdir(self.nas_fs.get_root() / 'raw_depth' / cam_name)
                        self.nas_fs.mkdir(self.nas_fs.get_root() / 'raw_ir' / cam_name)
            else:
                # noinspection PyCallingNonCallable
                self.nas_fs = FileStation(
                    env_get('SYNO_IP', '192.168.1.200'),
                    port=env_get('SYNO_PORT', '5000'),
                    username=env_get('SYNO_USERNAME', 'username'),
                    password=env_get('SYNO_PASSWORD', 'password'),
                    secure=False,
                    cert_verify=False,
                    dsm_version=7,
                    debug=False,
                    otp_code=None,
                    interactive_output=False,
                )
                self.nas_fs.create_folder(self.nas_root, self.experiment_name)
                self.nas_root = self.nas_root + '/' + self.experiment_name
                self.nas_fs.create_folder(self.nas_root, 'raw_color')
                self.nas_fs.create_folder(self.nas_root, 'raw_depth')
                self.nas_fs.create_folder(self.nas_root, 'raw_ir')
                for cam_name in self.cam_names:
                    if self.store_using_h5:
                        self.nas_fs.create_folder(self.nas_root, cam_name)
                    else:
                        self.nas_fs.create_folder(f'{self.nas_root}/raw_color', cam_name)
                        self.nas_fs.create_folder(f'{self.nas_root}/raw_depth', cam_name)
                        self.nas_fs.create_folder(f'{self.nas_root}/raw_ir', cam_name)
                log(f"[{self.__class__.__name__}::run] Connected to NAS at: \"{self.nas_root}\"", 'info')
        except Exception as e:
            log(f"[{self.__class__.__name__}::run] Failed to connect to NAS: {e}", 'error')
            os._exit(-1)

        t = time.time()
        while True:
            n_polled = self._poll()
            if self.termination_event.is_set():
                remaining_frames = sum([buf.write_idx.value - buf.read_idx.value for buf in self.buffers])
                if time.time() - t > 30:
                    log(f"[{self.__class__.__name__}::run] Remaining files in queue: {remaining_frames}", 'warning')
                    t = time.time()
                if remaining_frames == 0:
                    break
                if n_polled == 0:
                    time.sleep(0.1)
                    n_polled = self._poll()
                    if n_polled == 0:
                        log(f"[{self.__class__.__name__}::run] No files processed - terminating", 'warning')
                        break
            # try:
            # except Exception as e:
            #     # Handle errors (e.g., log the error, retry the transfer)
            #     log(f"[{self.__class__.__name__}::run] Failed uploading to NAS: {e}", 'error')


class SyncProcess(Process):
    def __init__(self, buffers: List[SharedRingBuffer], file_path: Path, termination_event: Event):
        assert all(hasattr(buf, 'idx2ts_list') and isinstance(buf.idx2ts_list, ShareableList) for buf in buffers), "All buffers must have an initialized idx2ts_list."
        super().__init__(daemon=True)

        self.idx2ts_lists = [buf.idx2ts_list for buf in buffers]
        self.cam_ids = [int(buf.name.split('_')[0].replace('cam', '')) for buf in buffers] # TODO: check me
        self.write_pos_list = [buf.write_idx for buf in buffers]
        self.termination_event = termination_event
        self.file_fpath = file_path
        self._file = None

        self.num_cameras = len(self.cam_ids)
        # Record layout: min_ts (Q), max_ts (Q), num_cams (I),
        # then for each camera: cam_idx (I), ts (Q), write_idx (Q)
        self.record_size = 8 + 8 + 4 + self.num_cameras * (4 + 8 + 8)
        self.max_clusters = len(self.idx2ts_lists[0])

        # Pre-allocate a ShareableList of empty bytes for cluster records
        initial = [b'\x00' * self.record_size] * self.max_clusters
        self.cluster_sl = ShareableList(initial, name="sync_clusters")
        self.cluster_write_pos = mp.Value('I', 0, lock=False)

        # In-memory clusters
        # Each cluster: {
        #   "min_ts": int,
        #   "max_ts": int,
        #   "members": {cam_id: (ts, write_idx), ...},
        #   "flushed": bool
        # }
        self.clusters = []
        # Keep sorted list of indices of clusters that haven't been flushed yet
        self.unflushed_indices = []

    def sig_handler(self, signum, frame):
        if signum in (signal.SIGTERM, signal.SIGINT):
            self.terminate()
            os._exit(0)

    def _flush_cluster(self, idx):
        """
        Flush cluster at self.clusters[idx], but first flush any unflushed with index < idx.
        """
        # Flush earlier unflushed clusters
        to_flush = []
        for u in self.unflushed_indices:
            if u < idx:
                to_flush.append(u)
        for u in to_flush:
            self._write_cluster(u)

        # Now flush idx itself
        self._write_cluster(idx)

    # noinspection PyTypeChecker
    def _write_cluster(self, idx):
        """
        Pack cluster at index idx to binary, write to file and ShareableList,
        mark flushed, and remove from unflushed_indices.
        """
        cl = self.clusters[idx]
        fmt = '<QQI' + 'IQQ' * self.num_cameras
        members = cl["members"]
        count = len(members)
        tup = [cl["min_ts"], cl["max_ts"], count]
        for cam in self.cam_ids:
            if cam in members:
                ts_i, idx_i = members[cam]
                tup += [cam, ts_i, idx_i]
            else:
                tup += [cam, 0, 0]
        packed = struct.pack(fmt, *tup)

        # write to binary file
        self._file.write(packed)
        # write to ShareableList
        pos = self.cluster_write_pos.value
        if len(cl['member']) == self.num_cameras and pos < self.max_clusters:  # only write complete clusters to the shared list
            self.cluster_sl[pos] = packed
            self.cluster_write_pos.value = pos + 1

        cl["flushed"] = True
        # remove idx from unflushed_indices
        if idx in self.unflushed_indices:
            self.unflushed_indices.remove(idx)

    def _insert_unflushed(self, idx):
        """
        Insert cluster index into unflushed_indices in sorted order by index.
        """
        pos = bisect_right(self.unflushed_indices, idx)
        self.unflushed_indices.insert(pos, idx)

    def run(self):
        signal.signal(signal.SIGTERM, self.sig_handler)
        signal.signal(signal.SIGINT, self.sig_handler)

        processed = [0] * self.num_cameras
        self._file = open(self.file_fpath, "wb")

        try:
            while True:
                any_new = False
                # 1) Read new timestamps and assign to clusters
                for cam_id, sl in enumerate(self.idx2ts_lists):
                    wp = self.write_pos_list[cam_id].value
                    while processed[cam_id] < wp:
                        ts = sl[processed[cam_id]]
                        idx = processed[cam_id]
                        processed[cam_id] += 1
                        any_new = True

                        if not self.clusters:
                            # first cluster
                            new_cl = {
                                "min_ts": ts,
                                "max_ts": ts,
                                "members": {cam_id: (ts, idx)},
                                "flushed": False
                            }
                            self.clusters.append(new_cl)
                            self._insert_unflushed(len(self.clusters) - 1)
                            continue

                        last_idx = len(self.clusters) - 1
                        last = self.clusters[last_idx]

                        # If ts is far to the right → create new cluster
                        if ts > last["max_ts"] + 16:
                            # create new cluster
                            new_cl = {
                                "min_ts": ts,
                                "max_ts": ts,
                                "members": {cam_id: (ts, idx)},
                                "flushed": False
                            }
                            self.clusters.append(new_cl)
                            self._insert_unflushed(len(self.clusters) - 1)
                        else:
                            # Check if ts fits into last by <=10 from max or <=10 from min
                            if (last["min_ts"] - 16) <= ts <= (last["max_ts"] + 16):
                                last["members"][cam_id] = (ts, idx)
                                if ts < last["min_ts"]:
                                    last["min_ts"] = ts
                                if ts > last["max_ts"]:
                                    last["max_ts"] = ts
                                # after updating last, flush if complete? no, we wait until explicit flush call
                            else:
                                # ts is left of last.min_ts -10 → search backwards
                                placed = False
                                for back_idx in range(last_idx - 1, -1, -1):
                                    cl = self.clusters[back_idx]
                                    if cam_id in cl["members"]:
                                        continue
                                    if (cl["min_ts"] - 16) <= ts <= (cl["max_ts"] + 16):
                                        cl["members"][cam_id] = (ts, idx)
                                        if ts < cl["min_ts"]:
                                            cl["min_ts"] = ts
                                        if ts > cl["max_ts"]:
                                            cl["max_ts"] = ts
                                        placed = True
                                        break
                                if not placed:
                                    # create a left‐sorted new cluster
                                    new_cl = {
                                        "min_ts": ts,
                                        "max_ts": ts,
                                        "members": {cam_id: (ts, idx)},
                                        "flushed": False
                                    }
                                    # find insertion index by min_ts
                                    insert_i = 0
                                    while (insert_i < len(self.clusters) and
                                           self.clusters[insert_i]["min_ts"] <= ts):
                                        insert_i += 1
                                    self.clusters.insert(insert_i, new_cl)
                                    self._insert_unflushed(insert_i)
                                    # adjust unflushed indices > insert_i by +1
                                    for i, u in enumerate(self.unflushed_indices):
                                        if u > insert_i:
                                            self.unflushed_indices[i] = u + 1

                        # Attempt to flush clusters if needed:
                        # If the last cluster in sequence is now complete (i.e., has all cameras),
                        # flush it and all prior unflushed clusters.
                        # we only need to check clusters that might have become complete:
                        # last cluster or any we just inserted/updated
                        # so gather indices of all clusters updated above
                        # for simplicity, check last and all unflushed
                        to_check = self.unflushed_indices.copy()
                        for cl_idx in to_check:
                            cl = self.clusters[cl_idx]
                            if len(cl["members"]) == self.num_cameras:
                                self._flush_cluster(cl_idx)

                # 2) Termination logic: if event set and no new data left, break
                if self.termination_event.is_set():
                    caught_up = True
                    for cam_id in range(self.num_cameras):
                        if processed[cam_id] < self.write_pos_list[cam_id].value:
                            caught_up = False
                            break
                    if caught_up:
                        break

                if not any_new:
                    time.sleep(0.001)
        except Exception as e:
            try:
                log(f"Sync process failed: {e}", "critical")
            except NameError:
                pass
        finally:
            # after termination, flush any remaining unflushed clusters
            for cl_idx in list(self.unflushed_indices):
                self._flush_cluster(cl_idx)

            # close resources
            self._file.close()
            self.cluster_sl.shm.close()


class MonitorProcess(Process):
    def __init__(self, buffers: List[SharedRingBuffer], sync_list: ShareableList, sync_list_head: mp.Value, video_fpath: Path, is_recording: mp.Value, start_time: mp.Value, frame_counts: List[mp.Value], termination_event: Event, window_name: str = 'CaptureStudio', fullscreen: bool = False):
        super().__init__(daemon=True)
        self.buffers = buffers
        self.sync_list = sync_list
        self.sync_list_head = sync_list_head
        self.video_fpath = video_fpath
        self.is_recording = is_recording
        self.start_time = start_time
        self.frame_counts = frame_counts
        self.termination_event = termination_event
        self.window_name = window_name
        self.fullscreen = fullscreen

    def sig_handler(self, signum, frame):
        # stop pipeline
        if signum in [signal.SIGTERM, signal.SIGINT]:
            self.terminate()
            # Exit the process immediately
            # noinspection PyProtectedMember
            os._exit(0)

    def terminate(self):
        if not self.termination_event.is_set():
            self.termination_event.set()
            time.sleep(0.1)
        cv2.destroyAllWindows()

    def _create_cv_window(self) -> Tuple[int, int]:
        from screeninfo import get_monitors
        # 1. Find all monitors and pick the rightmost one
        monitors = get_monitors()
        # Each monitor has .x, .y, .width, .height
        # Compute (x + width) to find the right‐edge; pick the monitor whose right‐edge is largest
        rightmost = max(monitors, key=lambda m: m.x + m.width)
        # 2. Extract its geometry
        x_offset = rightmost.x
        y_offset = rightmost.y
        w = rightmost.width
        h = rightmost.height
        # 3. Create a window and move it to that monitor’s top‐left
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, x_offset, y_offset)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window_name, w, h)
            # cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        # Return canvas size
        return w, h

    # noinspection PyUnboundLocalVariable
    def run(self):
        signal.signal(signal.SIGTERM, self.sig_handler)
        signal.signal(signal.SIGINT, self.sig_handler)
        try:
            # Calculate grid dimensions (e.g., 3x4 for 12 cameras)
            grid_rows = 4
            grid_cols = 3

            # Create a single large window
            window_width, window_height = self._create_cv_window()

            # Frame size (adjust as needed)
            color_frame_width = window_width // grid_cols
            color_frame_height = window_height // grid_rows
            depth_frame_width = color_frame_width // 2
            depth_frame_height = color_frame_height // 2

            # # Font settings for overlays
            # font = cv2.FONT_HERSHEY_PLAIN
            # font_scale = 0.45
            # font_color_green = (0, 255, 0)  # Green
            # font_color_red = (255, 0, 0)  # Red
            # font_thickness = 1

            # Create a grid layout
            grid_image = np.zeros((grid_rows * color_frame_height, grid_cols * color_frame_width, 3), dtype=np.uint8)
            sync_data_fmt = '<QQI' + 'IQQ' * len(self.buffers)
            # initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(self.video_fpath), fourcc, 10.0, (window_width, window_height))
            while not self.termination_event.is_set():
                # get last available cluster of frames
                if self.sync_list_head.value <= 0:
                    continue

                # Get last cluster of frames
                last_cluster_data_packed = self.sync_list[self.sync_list_head.value - 1]
                sync_data = struct.unpack(sync_data_fmt, last_cluster_data_packed)
                min_ts, max_ts, num_cams = sync_data[:3]
                sync_data = sync_data[3:]
                # Check if all cameras are present
                if num_cams != len(self.buffers):
                    continue

                # Get actual frames
                for i in range(len(self.buffers)):
                    cam_idx, ts, buffer_idx = sync_data[i * 3:i * 3 + 3]
                    data = self.buffers[cam_idx].read_frame_at_index(buffer_idx)

                    # Unpack data
                    meta_size = struct.calcsize('<fIIII')
                    depth_scale, depth_width, depth_height, color_data_len, depth_data_len = struct.unpack('<fIIII', data[:meta_size])

                    # get color frame
                    color_data_bytes = data[meta_size:meta_size + color_data_len]
                    color_frame = cv2.imdecode(np.frombuffer(color_data_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    color_frame = cv2.resize(color_frame, (color_frame_width, color_frame_height))

                    # get depth frame
                    if depth_data_len > 0:
                        depth_data_bytes = data[meta_size + color_data_len:meta_size + color_data_len + depth_data_len]
                        depth_data = ((np.frombuffer(depth_data_bytes, dtype=np.uint16).reshape((depth_height, depth_width)).astype(np.float32).clip(200, 5_000) - 200) / (5_000 - 200) * 255).astype(np.uint8)
                        depth_frame = cv2.applyColorMap(depth_data, cv2.COLORMAP_JET)
                        depth_frame = cv2.resize(depth_frame, (depth_frame_width, depth_frame_height))
                        color_frame[-depth_height:, -depth_width:, :] = depth_frame[:, :, :3]

                    # Calculate grid position
                    row = cam_idx // grid_cols
                    col = cam_idx % grid_cols

                    # Place the frame in the grid
                    grid_image[row * color_frame_height:(row + 1) * color_frame_height, col * color_frame_width:(col + 1) * color_frame_width, :] = color_frame[:, :, :3]

                    # # Add overlays (frame rate, buffer status, etc.)
                    # font_color = font_color_green if self.is_recording.value else font_color_red
                    # #   - camera ID
                    # cam_id_text = f"Cam {i + 1:03d}"
                    # cv2.putText(grid_image, cam_id_text, (col * color_width + 10, row * color_height + 15),
                    #             font, font_scale, font_color, font_thickness)
                    # #   - fps
                    # fps = self.frame_counts[i].value / max(1, time.time() - self.start_time.value)
                    # fps_text = f"FPS: {fps:.1f}"
                    # cv2.putText(grid_image, fps_text, (col * color_width + 10, row * color_height + 30),
                    #             font, font_scale, font_color, font_thickness)
                    # #   - buffer usage
                    # frames_behind = buffer.write_idx.value - buffer.read_idx.value
                    # buffer_usage = min(100, int(frames_behind / buffer.num_slots * 100))
                    # buffer_text = f"Buffer: {buffer_usage}%"
                    # cv2.putText(grid_image, buffer_text, (col * color_width + 10, row * color_height + 45),
                    #             font, font_scale, font_color, font_thickness)
                    # #   - frames captured
                    # frames_captured = int(self.frame_counts[i].value)
                    # frames_captured_text = f"T={frames_captured:05d}"
                    # cv2.putText(grid_image, frames_captured_text, (col * color_width + 10, row * color_height + 60),
                    #             font, font_scale, font_color, font_thickness)

                # Display the grid
                cv2.imshow(self.window_name, grid_image)
                video_writer.write(grid_image)

                # Sleep to simulate lower fps
                time.sleep(0.01)

                # Check for key presses (e.g., to quit)
                if cv2.waitKey(1) == ord('q'):
                    break
        except Exception as e:
            log(f"Monitor process failed: {e}", 'critical')
        finally:
            video_writer.release()
            cv2.destroyAllWindows()


# noinspection PyUnresolvedReferences
class CameraProducer(Process):
    """Handles camera acquisition and frame writing to shared memory"""
    PROFILE_META = None

    def __init__(self,
                 camera_id: int,
                 camera_sn: str,
                 config: dict,
                 ring_buffer: SharedRingBuffer,
                 imu_ring_buffer: SharedRingBuffer,
                 fs: LocalFilesystem,
                 termination_event: Event,
                 start_barrier: mp.Barrier,
                 color_quality: int = 90,
                 color_resolution: tuple = (1920, 1080),
                 color_fps: int = 30,
                 depth_resolution: tuple = (640, 576),
                 depth_fps: int = 30,
                 use_depth: bool = True, ):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.camera_sn = camera_sn
        self.config = config
        self.ring_buffer = ring_buffer
        self.imu_ring_buffer = imu_ring_buffer
        self.termination_event = termination_event
        self.color_quality = color_quality
        self.fs = fs
        self.start_barrier = start_barrier
        self.rotation = config.get('rotate', 0)
        self.frame_count = 0
        self.degraded_mode = False
        self.pipeline: Optional[porb.Pipeline] = None
        self.is_stopped = True
        self.profile_path = self.fs.get_root() / 'profiles' / f'{self.camera_sn}.json'
        self._fallback_ts = 0
        self.color_resolution = color_resolution
        self.color_fps = color_fps
        self.depth_resolution = depth_resolution
        self.depth_fps = depth_fps
        self.fps_synced = self.color_fps == self.depth_fps
        self.use_depth = use_depth

    def terminate(self):
        # stop pipeline
        if not self.termination_event.is_set():
            self.termination_event.set()
        if not self.is_stopped:
            self.pipeline.stop()
            self.is_stopped = True
            log(f"[{self.__class__.__name__}::terminate] Camera {self.camera_id:03d} ({self.camera_sn}) stopped "
                f"(written {self.ring_buffer.read_idx.value} out of {self.ring_buffer.read_idx.value} frames).",
                'debug')

    def sig_handler(self, signum, frame):
        # stop pipeline
        if signum in [signal.SIGTERM, signal.SIGINT]:
            self.terminate()
            # Exit the process immediately
            # noinspection PyProtectedMember
            os._exit(0)

    # noinspection PyUnboundLocalVariable
    def run(self):
        signal.signal(signal.SIGTERM, self.sig_handler)
        signal.signal(signal.SIGINT, self.sig_handler)
        try:
            # Initialize camera
            ctx = porb.Context()
            device: porb.Device = ctx.query_devices().get_device_by_serial_number(self.camera_sn)
            # sync device's timestamp with host
            # device.timestamp_reset()
            # create pipeline
            pipeline = porb.Pipeline(device)
            self.pipeline = pipeline
            # Apply camera configuration
            orb_config, orb_profile_meta = self._configure_camera(device, pipeline)
            # align_filter = porb.AlignFilter(align_to_stream=porb.OBStreamType.COLOR_STREAM)
            self.profile_path.parent.mkdir(exist_ok=True)
            with open(self.profile_path, 'w') as f:
                json.dump(orb_profile_meta, f, indent=2)
            log(f"[{self.__class__.__name__}::run] Camera {self.camera_sn} profile metadata saved at: {self.profile_path}", 'debug')

            def frame_handler(frame):
                if self.termination_event.is_set():
                    return

                try:
                    # accel_frame = frame.get_frame(porb.OBFrameType.ACCEL_FRAME)
                    # gyro_frame = frame.get_frame(porb.OBFrameType.GYRO_FRAME)
                    # if accel_frame is not None or gyro_frame is not None:
                    #     self._process_imu_frame(accel_frame.as_accel_frame() if accel_frame is not None else None, gyro_frame.as_gyro_frame() if gyro_frame is not None else None)

                    color_frame = frame.get_color_frame()
                    depth_frame = frame.get_depth_frame()

                    # if (self.fps_synced and color_frame and depth_frame) or (not self.fps_synced and color_frame):
                    # d2c first
                    # frame = align_filter.process(frame).as_frame_set()
                    # color_frame = frame.get_color_frame()
                    # depth_frame = frame.get_depth_frame()
                    # process the aligned frames
                    self._process_frames(color_frame, depth_frame)
                except Exception as _e:
                    logging.error(f"Frame processing error: {_e}")
                    if not self.degraded_mode:
                        self.degraded_mode = True
                        logging.warning(f"Entering degraded mode for camera {self.camera_id}")

            # wait for other producers to arrive here
            self.start_barrier.wait()
            pipeline.enable_frame_sync()
            pipeline.start(orb_config, frame_handler)
            self.is_stopped = False
            log(f"[{self.__class__.__name__}::run] Camera {self.camera_sn} started", 'debug')

            while not self.termination_event.is_set():
                time.sleep(0.001)

        except Exception as e:
            logging.critical(f"[{self.__class__.__name__}::run] Camera {self.camera_sn} failed: {e}", exc_info=e)
        finally:
            self.terminate()

    def _configure_camera(self, device, pipeline):
        # Reset state
        # if device.is_property_supported(porb.OBPropertyID.OB_PROP_RESTORE_FACTORY_SETTINGS_BOOL, porb.OBPermissionType.PERMISSION_READ_WRITE):
        #     device.set_bool_property(porb.OBPropertyID.OB_PROP_RESTORE_FACTORY_SETTINGS_BOOL, True)
        # # if device.is_property_supported(porb.OBPropertyID.OB_PROP_TIMESTAMP_OFFSET_INT, porb.OBPermissionType.PERMISSION_READ_WRITE):
        # #     device.set_int_property(porb.OBPropertyID.OB_PROP_TIMESTAMP_OFFSET_INT, 0)
        # if device.is_property_supported(porb.OBPropertyID.OB_PROP_INDICATOR_LIGHT_BOOL, porb.OBPermissionType.PERMISSION_READ_WRITE):
        #     device.set_bool_property(porb.OBPropertyID.OB_PROP_INDICATOR_LIGHT_BOOL, False)
        # if device.is_property_supported(porb.OBPropertyID.OB_PROP_HEARTBEAT_BOOL, porb.OBPermissionType.PERMISSION_READ_WRITE):
        #     device.set_bool_property(porb.OBPropertyID.OB_PROP_HEARTBEAT_BOOL, True)
        # if device.is_property_supported(porb.OBPropertyID.OB_PROP_USB_POWER_STATE_INT, porb.OBPermissionType.PERMISSION_READ_WRITE):
        #     device.set_int_property(porb.OBPropertyID.OB_PROP_USB_POWER_STATE_INT, 0)
        # if device.is_property_supported(porb.OBPropertyID.OB_PROP_DC_POWER_STATE_INT, porb.OBPermissionType.PERMISSION_READ_WRITE):
        #     device.set_int_property(porb.OBPropertyID.OB_PROP_DC_POWER_STATE_INT, 1)
        # # if device.is_property_supported(porb.OBPropertyID.OB_PROP_TIMER_RESET_ENABLE_BOOL, porb.OBPermissionType.PERMISSION_READ_WRITE):
        # #     device.set_bool_property(porb.OBPropertyID.OB_PROP_TIMER_RESET_ENABLE_BOOL, True)
        # if device.is_property_supported(porb.OBPropertyID.OB_PROP_SWITCH_IR_MODE_INT, porb.OBPermissionType.PERMISSION_READ_WRITE):
        #     device.set_int_property(porb.OBPropertyID.OB_PROP_SWITCH_IR_MODE_INT, 0)
        # device.timestamp_reset()
        # Apply synchronization config from JSON
        sync_config = device.get_multi_device_sync_config()
        for key, value in self.config['config'].items():
            if key == 'mode':
                value = getattr(porb.OBMultiDeviceSyncMode, value.upper())
            setattr(sync_config, key, value)

        device.set_multi_device_sync_config(sync_config)
        # sync timestamps
        # device.timestamp_reset()
        device.timer_sync_with_host()
        # set depth sensor properties
        if device.is_property_supported(porb.OBPropertyID.OB_PROP_MIN_DEPTH_INT, porb.OBPermissionType.PERMISSION_READ_WRITE) and \
                (device.get_int_property(porb.OBPropertyID.OB_PROP_MIN_DEPTH_INT) != 200 or device.get_int_property(porb.OBPropertyID.OB_PROP_MAX_DEPTH_INT) != 5_000):
            # Set the min Depth value, and the Depth less than the modified value will be set to 0,Unit: mm
            device.set_int_property(porb.OBPropertyID.OB_PROP_MIN_DEPTH_INT, 200)
            # Set the max Depth value, the Depth greater than the modified value will be set to 0, unit mm
            device.set_int_property(porb.OBPropertyID.OB_PROP_MAX_DEPTH_INT, 5_000)
        if device.is_property_supported(porb.OBPropertyID.OB_PROP_DEPTH_PRECISION_LEVEL_INT, porb.OBPermissionType.PERMISSION_READ_WRITE) and \
                device.get_int_property(porb.OBPropertyID.OB_PROP_DEPTH_PRECISION_LEVEL_INT) != porb.OBDepthPrecisionLevel.ONE_MM:
            device.set_int_property(porb.OBPropertyID.OB_PROP_DEPTH_PRECISION_LEVEL_INT, porb.OBDepthPrecisionLevel.ONE_MM)
        # set powerline
        if device.is_property_supported(porb.OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, porb.OBPermissionType.PERMISSION_READ_WRITE) and \
                device.get_int_property(porb.OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT) != porb.OBPowerLineFreqMode.FREQUENCY_50HZ:
            device.set_int_property(porb.OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, porb.OBPowerLineFreqMode.FREQUENCY_50HZ)
        # set color sensor properties
        if device.is_property_supported(porb.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, porb.OBPermissionType.PERMISSION_READ_WRITE) and \
                device.get_bool_property(porb.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL):
            device.set_bool_property(porb.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
            device.set_int_property(porb.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 120)
            device.set_int_property(porb.OBPropertyID.OB_PROP_COLOR_GAIN_INT, 20)
        if device.is_property_supported(porb.OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, porb.OBPermissionType.PERMISSION_READ_WRITE) and \
                device.get_bool_property(porb.OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL):
            device.set_bool_property(porb.OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, False)
            device.set_int_property(porb.OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, 3700)
        if device.is_property_supported(porb.OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT, porb.OBPermissionType.PERMISSION_READ_WRITE):
            if device.get_int_property(porb.OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT) != 30:
                device.set_int_property(porb.OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT, 30)
            if device.get_int_property(porb.OBPropertyID.OB_PROP_COLOR_SATURATION_INT) != 64:
                device.set_int_property(porb.OBPropertyID.OB_PROP_COLOR_SATURATION_INT, 64)
            if device.get_int_property(porb.OBPropertyID.OB_PROP_COLOR_CONTRAST_INT) != 35:
                device.set_int_property(porb.OBPropertyID.OB_PROP_COLOR_CONTRAST_INT, 35)
        # set up camera profiles
        config = porb.Config()
        c_profile = pipeline.get_stream_profile_list(porb.OBSensorType.COLOR_SENSOR).get_video_stream_profile(*self.color_resolution, porb.OBFormat.MJPG, self.color_fps)
        config.enable_stream(c_profile)
        log(f'[{self.__class__.__name__}::_configure_camera] Using color profile: {c_profile.get_width()}x{c_profile.get_height()}@{c_profile.get_fps()}_{c_profile.get_format()}')
        if self.use_depth:
            d_profile = pipeline.get_stream_profile_list(porb.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(*self.depth_resolution, porb.OBFormat.Y16, self.depth_fps)
            config.enable_stream(d_profile)
            log(f'[{self.__class__.__name__}::_configure_camera] Using depth profile: {d_profile.get_width()}x{d_profile.get_height()}@{d_profile.get_fps()}_{d_profile.get_format()}')
            config.set_align_mode(porb.OBAlignMode.SW_MODE)
            config.set_depth_scale_require(True)
        # config.enable_gyro_stream(porb.OBGyroFullScaleRange.FS_16dps, porb.OBGyroSampleRate.SAMPLE_RATE_1_KHZ)
        # log(f'[{self.__class__.__name__}::_configure_camera] Using GYRO profile: {porb.OBGyroFullScaleRange.FS_16dps.name}@{porb.OBGyroSampleRate.SAMPLE_RATE_1_KHZ.name.replace("SAMPLE_RATE_", "")}')
        # config.enable_accel_stream(porb.OBAccelFullScaleRange.ACCEL_FS_2g, porb.OBGyroSampleRate.SAMPLE_RATE_1_KHZ)
        # log(f'[{self.__class__.__name__}::_configure_camera] Using ACCEL profile: {porb.OBAccelFullScaleRange.ACCEL_FS_2g.name}@{porb.OBGyroSampleRate.SAMPLE_RATE_1_KHZ.name.replace("SAMPLE_RATE_", "")}')
        config.set_frame_aggregate_output_mode(porb.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        # get profile metadata
        c_intri = c_profile.get_intrinsic()
        c_dist = c_profile.get_distortion()
        profile_meta = dict(
            color=dict(
                fps=c_profile.get_fps(),
                height=c_profile.get_height(),
                width=c_profile.get_width(),
                intrinsic=dict(
                    fx=c_intri.fx,
                    fy=c_intri.fy,
                    cx=c_intri.cx,
                    cy=c_intri.cy,
                    width=c_intri.width,
                    height=c_intri.height,
                ),
                distortion=dict(
                    k1=c_dist.k1,
                    k2=c_dist.k2,
                    k3=c_dist.k3,
                    k4=c_dist.k4,
                    k5=c_dist.k5,
                    k6=c_dist.k6,
                    p1=c_dist.p1,
                    p2=c_dist.p2,
                ),
            )
        )
        if self.use_depth:
            d_intri = d_profile.get_intrinsic()
            d_dist = d_profile.get_distortion()
            d2c_transform = d_profile.get_extrinsic_to(c_profile)
            profile_meta |= dict(
                depth=dict(
                    fps=d_profile.get_fps(),
                    height=d_profile.get_height(),
                    width=d_profile.get_width(),
                    intrinsic=dict(
                        fx=d_intri.fx,
                        fy=d_intri.fy,
                        cx=d_intri.cx,
                        cy=d_intri.cy,
                        width=d_intri.width,
                        height=d_intri.height,
                    ),
                    distortion=dict(
                        k1=d_dist.k1,
                        k2=d_dist.k2,
                        k3=d_dist.k3,
                        k4=d_dist.k4,
                        k5=d_dist.k5,
                        k6=d_dist.k6,
                        p1=d_dist.p1,
                        p2=d_dist.p2,
                    ),
                    extrinsic_to_color=dict(
                        rot=d2c_transform.rot.tolist(),
                        trans=d2c_transform.transform.tolist(),
                    )
                )
            )
        return config, profile_meta

    def _process_imu_frame(self, accel_frame, gyro_frame):
        try:
            if accel_frame is not None:
                accel_index = int(accel_frame.get_index())
                accel_ts = int(accel_frame.get_timestamp())
                accel_x, accel_y, accel_z = accel_frame.get_x(), accel_frame.get_y(), accel_frame.get_z()
            else:
                accel_index, accel_ts = 0, 0
                accel_x, accel_y, accel_z = 0.0, 0.0, 0.0
            if gyro_frame is not None:
                gyro_index = gyro_frame.get_index()
                gyro_ts = int(gyro_frame.get_timestamp())
                gyro_x, gyro_y, gyro_z = gyro_frame.get_x(), gyro_frame.get_y(), gyro_frame.get_z()
            else:
                gyro_index, gyro_ts = 0, 0
                gyro_x, gyro_y, gyro_z = 0.0, 0.0, 0.0
            imu_data = struct.pack('<QfffQfff', accel_index, accel_x, accel_y, accel_z, gyro_index, gyro_x, gyro_y, gyro_z)
            if not self.imu_ring_buffer.write_frame(imu_data, accel_ts, gyro_ts):
                log(f"[{self.__class__.__name__}::_process_imu_frame] Camera {self.camera_sn} IMU buffer full - frame dropped", 'warning')
        except Exception as e:
            log(f"[{self.__class__.__name__}::_process_imu_frame] Camera {self.camera_sn} IMU frame processing error: {e}", 'error', exc_info=e)

    def _process_frames(self, color_frame, depth_frame=None):
        try:
            # Process color frame
            color_data_bytes = color_frame.get_data().tobytes()
            color_ts = int(color_frame.get_timestamp())

            # Process depth frame
            if not self.use_depth or depth_frame is None:
                depth_data_bytes = bytes('\x00'.encode())  # null byte
                depth_scale = float(0)
                depth_width, depth_height = 0, 0
                depth_ts = 0
            else:
                depth_data_bytes = depth_frame.get_data().tobytes()
                depth_scale = depth_frame.get_depth_scale()
                depth_width, depth_height = depth_frame.get_width(), depth_frame.get_height()
                depth_ts = int(depth_frame.get_timestamp())

            # Pack data with zero-copy
            frame_meta = struct.pack('<fIIII', depth_scale, depth_width, depth_height, len(color_data_bytes), len(depth_data_bytes))
            packed = frame_meta + color_data_bytes + depth_data_bytes
            if color_ts == depth_ts == 0:
                color_ts = depth_ts = self._fallback_ts
                if color_ts == 0:
                    log(f'[{self.__class__.__name__}::_process_frames] Fallback timestamp activated for camera {self.camera_id} ({self.camera_sn})', 'warning')
                self._fallback_ts += 1

            if not self.ring_buffer.write_frame(packed, color_ts, depth_ts):
                log(f"[{self.__class__.__name__}::_process_frames] Camera {self.camera_sn} buffer full - frame dropped", 'warning')
            self.frame_count += 1

        except Exception as e:
            log(f"[{self.__class__.__name__}::_process_frames] Camera {self.camera_sn} frame processing error: {e}", 'error', exc_info=e)


class StorageConsumer(Process):
    """Handles batch storage to disk using memory-mapped files"""

    def __init__(self,
                 ring_buffer: SharedRingBuffer,
                 fs: IFilesystem,
                 cam_name: str,
                 store_frame_format: str,
                 termination_event: Event,
                 batch_size: int = 100,
                 save_raw_frames: bool = False,
                 nas_buffer: Optional[SharedRingBufferNas] = None,
                 use_depth: bool = True):
        super().__init__(daemon=True)
        self.ring_buffer = ring_buffer
        self.fs = fs
        self.cam_name = cam_name
        self.store_frame_format = store_frame_format
        self.termination_event = termination_event
        self.batch_size = batch_size
        self.save_raw_frames = save_raw_frames
        self.batch = []
        self.last_flush = time.time()
        self.nas_buffer = nas_buffer
        self.use_depth = use_depth

    def terminate(self):
        if not self.termination_event.is_set():
            self.termination_event.set()
        time.sleep(0.1)
        # print(f"[{self.__class__.__name__}::sig_handler] Terminating {self.cam_name}. Remaining frames: "
        #       f"{self.ring_buffer.write_idx.value, self.ring_buffer.read_idx.value}")
        # Flush remaining frames
        while self.ring_buffer.write_idx.value > self.ring_buffer.read_idx.value:
            log(f"[{self.__class__.__name__}::terminate] Flushing remaining frames for {self.cam_name}: "
                f"{self.ring_buffer.write_idx.value - self.ring_buffer.read_idx.value}", 'debug')
            data, color_timestamp, depth_timestamp = self.ring_buffer.read_frame()
            if data:
                self.batch.append((data, color_timestamp, depth_timestamp))
            # Flush based on size or time
            if len(self.batch) >= self.batch_size or (time.time() - self.last_flush) > 1.0:
                self._flush_batch()
            time.sleep(0.001)
        self._flush_batch()

    def sig_handler(self, signum, frame):
        if signum in [signal.SIGTERM, signal.SIGINT]:
            self.terminate()
            # Exit the process immediately
            # noinspection PyProtectedMember
            os._exit(0)

    def run(self):
        signal.signal(signal.SIGTERM, self.sig_handler)
        signal.signal(signal.SIGINT, self.sig_handler)
        try:
            if self.save_raw_frames:
                (self.fs.get_root() / 'raw_color' / self.cam_name).mkdir(parents=True, exist_ok=True)
                if self.use_depth:
                    (self.fs.get_root() / 'raw_depth' / self.cam_name).mkdir(parents=True, exist_ok=True)
                (self.fs.get_root() / 'raw_ir' / self.cam_name).mkdir(parents=True, exist_ok=True)
            else:
                (self.fs.get_root() / self.cam_name).mkdir(parents=True, exist_ok=True)

            while not self.termination_event.is_set() or self.ring_buffer.write_idx.value > self.ring_buffer.read_idx.value:
                data, color_timestamp, depth_timestamp = self.ring_buffer.read_frame()
                if data:
                    self.batch.append((data, color_timestamp, depth_timestamp))

                # Flush based on size or time
                if len(self.batch) >= self.batch_size:
                    self._flush_batch()
                time.sleep(0.001)
            # Final flush
            self._flush_batch()

        except Exception as e:
            log(f"Storage consumer failed: {e}", 'critical', exc_info=e)

    def _flush_batch(self):
        if not self.batch:
            return

        try:
            if self.save_raw_frames:
                self._save_raw_frames()
            else:
                self._save_hdf5_batch()

            self.batch.clear()
            self.last_flush = time.time()
        except Exception as e:
            log(f'[{self.__class__.__name__}::_flush_batch] Batch storage failed: {e}', 'error', exc_info=e)

    def _save_hdf5_batch(self):
        first_color_ts, last_color_ts = self.batch[0][1], self.batch[-1][1]
        first_depth_ts, last_depth_ts = self.batch[0][2], self.batch[-1][2]
        first_ts, last_ts = min(first_color_ts, first_depth_ts), max(last_color_ts, last_depth_ts)
        filename = f'{first_ts:08d}-{last_ts:08d}.h5'
        path = self.fs.get_root() / self.cam_name / filename
        depth_group_created = False
        depth_group = None
        with h5py.File(path, 'w') as hf:
            color_group = hf.create_group('color')
            for i, (data, color_timestamp, depth_timestamp) in enumerate(self.batch):
                # Unpack data
                meta_size = struct.calcsize('<fIIII')
                depth_scale, depth_width, depth_height, color_data_len, depth_data_len = struct.unpack('<fIIII', data[:meta_size])
                # Store color data
                color_data_bytes = data[meta_size:meta_size + color_data_len]
                color_data = np.frombuffer(color_data_bytes, dtype=np.uint8)
                color_group.create_dataset(str(color_timestamp), data=color_data)
                # Store depth data
                if self.use_depth and depth_height > 1 and depth_width > 1:
                    if not depth_group_created:
                        depth_group = hf.create_group('depth')
                        depth_group_created = True
                    depth_data_bytes = data[meta_size + color_data_len:meta_size + color_data_len + depth_data_len]
                    depth_data = np.frombuffer(depth_data_bytes, dtype=np.uint16).reshape((depth_height, depth_width))
                    depth_group.create_dataset(str(depth_timestamp), data=depth_data)
                    depth_group[str(depth_timestamp)].attrs['scale'] = depth_scale
                    depth_group[str(depth_timestamp)].attrs['width'] = depth_width
                    depth_group[str(depth_timestamp)].attrs['height'] = depth_height
            hf.close()

        # inform NAS uploader
        if self.nas_buffer is not None:
            self.nas_buffer.write_frame(str(path))

    def _save_raw_frames(self):
        for data, color_timestamp, depth_timestamp in self.batch:
            # Unpack data
            meta_size = struct.calcsize('<fIIII')
            depth_scale, depth_width, depth_height, color_data_len, depth_data_len = struct.unpack('<fIIII', data[:meta_size])

            # Extract color data
            color_data_bytes = data[meta_size:meta_size + color_data_len]
            # color_img = cv2.imdecode(np.frombuffer(color_data_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            # Save color frame
            color_path = self.fs.get_root() / 'raw_color' / self.cam_name / self.store_frame_format.format(frame=int(color_timestamp))
            with open(color_path, 'wb') as f:
                f.write(color_data_bytes)
            # cv2.imwrite(str(color_path), color_img)

            # Extract depth data
            if self.use_depth and depth_width > 1 and depth_height > 1:
                depth_data_bytes = data[meta_size + color_data_len:meta_size + color_data_len + depth_data_len]
                depth_data = np.frombuffer(depth_data_bytes, dtype=np.uint16).reshape((depth_height, depth_width))
                # Save depth frame
                depth_path = (self.fs.get_root() / 'raw_depth' / self.cam_name / self.store_frame_format.format(frame=int(depth_timestamp))).with_suffix('.npy')
                np.save(str(depth_path), depth_data)

            # inform NAS uploader
            if self.nas_buffer is not None:
                if depth_width > 1 and depth_height > 1:
                    self.nas_buffer.write_frame(str(color_path), str(depth_path))  # , str(ir_path))
                else:
                    self.nas_buffer.write_frame(str(color_path))


class IMUStorageConsumer(StorageConsumer):
    def _write_batch_to_csv(self, store_folder: Path) -> None:
        first_accel_ts, last_accel_ts = self.batch[0][1], self.batch[-1][1]
        first_gyro_ts, last_gyro_ts = self.batch[0][2], self.batch[-1][2]
        first_ts, last_ts = min(first_accel_ts, first_gyro_ts), max(last_accel_ts, last_gyro_ts)

        imu_path = store_folder / f'{first_ts:08d}-{last_ts:08d}.csv'
        with open(imu_path, 'w') as f:
            # Write CSV header
            f.write("accel_timestamp,accel_x,accel_y,accel_z,gyro_timestamp,gyro_x,gyro_y,gyro_z\n")
            for data, accel_timestamp, gyro_timestamp in self.batch:
                meta_size = struct.calcsize('<QfffQfff')
                accel_index, accel_x, accel_y, accel_z, gyro_index, gyro_x, gyro_y, gyro_z = struct.unpack('<QfffQfff', data[:meta_size])
                # Write one line per IMU sample
                f.write(f"{accel_index},{accel_x:.6f},{accel_y:.6f},{accel_z:.6f},{gyro_index},{gyro_x:.6f},{gyro_y:.6f},{gyro_z:.6f}\n")

        # inform NAS uploader
        if self.nas_buffer is not None:
            self.nas_buffer.write_frame(str(imu_path))

    def _save_hdf5_batch(self):
        self._write_batch_to_csv(self.fs.get_root() / self.cam_name)

    def _save_raw_frames(self):
        self._write_batch_to_csv(self.fs.get_root() / 'raw_ir' / self.cam_name)


class CaptureSystem:
    """Main capture system management class"""

    def __init__(self,
                 num_cameras: int = 12,
                 num_depth_cameras: int = 12,
                 freerun_mode: bool = False,
                 date_fmt: str = "%d_%m_%Y_%H_%M_%S",
                 cam_names: Union[List[str], str] = '__auto__',
                 store_frame_format: str = '{frame:06d}.jpg',
                 use_4k_color: bool = False,
                 use_1k_depth: bool = False,
                 sync_color_fps_to_depth: bool = False,
                 num_mem_slots: int = 1_000,  # 1000 slots per camera
                 slot_size_mb: int = 8,  # 6 MB per slot
                 storage_batch_size: int = 100,
                 store_using_h5: bool = False,
                 nas_root: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 color_fps: Optional[int] = None,
                 depth_fps: Optional[int] = None,
                 create_shm: bool = True,
                 **metadata):
        self._constructor_args = {k: v for k, v in locals().items() if k != 'self'}
        self.metadata = metadata
        self.config_fpath = PathUtils.config_path() / 'orbbec' / f'c{num_cameras}.json'
        self.config_name = self.config_fpath.stem
        self.freerun_mode = freerun_mode
        if freerun_mode:
            self.config_name += '_freerun'
        else:
            self.config_name += f'_synced'
        self.experiment_name = experiment_name if experiment_name is not None else \
            f"l{num_cameras}_d{num_depth_cameras}_{self.config_name.lower().strip()}_@{datetime.now(pytz.timezone('Europe/Zurich')).strftime(date_fmt)}"
        self.cam_names = cam_names
        self.num_cameras = num_cameras
        self.store_frame_format = store_frame_format
        self.store_using_h5 = store_using_h5
        self.storage_batch_size = storage_batch_size
        self.use_4k_color = use_4k_color
        self.use_1k_depth = use_1k_depth
        self.sync_color_fps_to_depth = sync_color_fps_to_depth
        self.depth_resolution = (1024, 1024) if use_1k_depth else (640, 576)
        self.depth_fps = depth_fps if depth_fps is not None else (30 if not use_1k_depth else 15)
        self.color_resolution = (3840, 2160) if use_4k_color else (1920, 1080)
        self.color_fps = color_fps if color_fps is not None else (30 if not sync_color_fps_to_depth else self.depth_fps)

        self.xml_config = None
        self.config = self.load_config()
        self.context = self.load_context() if porb else None
        self.termination_event = Event()

        # Initialize status variables
        self.is_recording = mp.Value('b', False)
        self.start_time = mp.Value('d', 0.0)
        self.frame_counts = [mp.Value('i', 0) for _ in range(num_cameras)]

        # Initialize shared memory buffers
        self.num_mem_slots = num_mem_slots
        self.slot_size_mb = slot_size_mb

        self.idx2ts_lists = [
            ShareableList(
                [0] * 100_000,
                name=f"cam_{i}_idx2ts_list",
            ) if create_shm else
            [0] * 100_000
            for i in range(num_cameras)
        ]

        self.video_buffers = [
            SharedRingBuffer(
                name=f"cam_{i}_buffer",
                num_slots=num_mem_slots,
                idx2ts_list=self.idx2ts_lists[i],
                slot_size=slot_size_mb * 1024 * 1024,
                frame_count=self.frame_counts[i],
            ) if create_shm else
            None
            for i in range(num_cameras)
        ]
        self.imu_buffers = [
            SharedRingBuffer(
                name=f"cam_{i}_imu_buffer",
                num_slots=5_000,  # more slots because 100Hz is fast
                slot_size=64,  # ~few bytes per IMU frame
            ) if create_shm else
            None
            for i in range(num_cameras)
        ]
        self.consumers = []
        self.imu_consumers = []
        self.producers = []
        self.sync_process: Optional[SyncProcess] = None
        self.monitor_process = None
        self.nas_uploader = None

        # Filesystem setup
        self.fs = LocalFilesystem(root=PathUtils.out_path() / 'Captures' / self.experiment_name)
        self.use_nas = nas_root is not None
        self.nas_root = nas_root
        self.nas_buffers = None
        if self.use_nas:
            self.nas_buffers = [
                SharedRingBufferNas(
                    name=f"nas_buffer_{i}",
                    delimiter=';',
                ) if create_shm else
                None
                for i in range(num_cameras)
            ]
        self.num_depth_cameras = num_depth_cameras

    def state(self):
        # return all constructor args
        return self._constructor_args

    @classmethod
    def from_state(cls, state):
        if 'config_name' in state:
            num_cameras = 12
            config_name = state.pop('config_name')
            depth_delay = config_name.split('_')[-1]
            if depth_delay == '225':
                num_depth_cameras = 6
            elif depth_delay == '165':
                num_depth_cameras = 8
            else:
                num_depth_cameras = 12
            state['freerun_mode'] = 'sync' not in config_name
            state['num_cameras'] = num_cameras
            state['num_depth_cameras'] = num_depth_cameras
        if 'color_fps' not in state:
            state['color_fps'] = 30
        if 'depth_fps' not in state:
            state['depth_fps'] = 15 if not state.get('use_1k_depth', False) else (30 if state.get('sync_color_fps_to_depth', False) else 15)
        if 'freerun_mode' not in state:
            state['freerun_mode'] = False
        state['create_shm'] = False  # do not create shared memory when loading from state
        return cls(**state)

    def start_nas_uploader(self):
        if self.nas_uploader is None:
            self.nas_uploader = NASUploader(
                self.nas_buffers,
                nas_root=self.nas_root,
                local_root=str(self.fs.get_root()),
                experiment_name=self.experiment_name,
                cam_names=[c.cam_name for c in self.consumers],
                store_using_h5=self.store_using_h5,
                termination_event=self.termination_event,
            )
        self.nas_uploader.start()

    def start_monitor(self):
        self.monitor_process = MonitorProcess(
            buffers=self.video_buffers,
            sync_list=self.sync_process.cluster_sl,
            sync_list_head=self.sync_process.cluster_write_pos,
            video_fpath=self.fs.get_root() / 'monitor.mp4',
            is_recording=self.is_recording,
            start_time=self.start_time,
            frame_counts=self.frame_counts,
            termination_event=self.termination_event
        )
        self.monitor_process.start()

    def start(self):
        """Start all capture and storage processes"""
        if not self.is_recording.value:
            self.is_recording.value = True
            self.start_time.value = time.time()
            try:
                ctx = self.context
                device_list = ctx.query_devices()
                curr_device_cnt = device_list.get_count()
                if curr_device_cnt == 0:
                    log(f'[{self.__class__.__name__}::__setup__] No device connected', 'critical')
                    exit(-1)
                if curr_device_cnt > self.num_cameras:
                    log(f'[{self.__class__.__name__}::__setup__] Too many devices connected: {curr_device_cnt}',
                        'critical')
                    exit(-1)

                log(f'[{self.__class__.__name__}::start] Setting up {device_list.get_count()} devices...', 'info')
                total_cameras = device_list.get_count()
                start_barrier = mp.Barrier(total_cameras, timeout=10)
                pbar = tqdm(range(total_cameras), desc='Connecting devices')
                self.producers.clear()
                self.consumers.clear()
                self.imu_consumers.clear()
                vs = sorted([v for v in self.config.values()], key=lambda x: int(x['physical_index']))
                all_serial_numbers_by_phy_index = [v['serial_number'] for v in vs]
                num_diff = self.num_cameras - self.num_depth_cameras
                if num_diff > 0:
                    depth_serial_numbers = all_serial_numbers_by_phy_index[num_diff // 2:-num_diff // 2]
                else:
                    depth_serial_numbers = all_serial_numbers_by_phy_index
                log(f'[{self.__class__.__name__}::start] Depth cameras: {depth_serial_numbers}', 'info')
                # Start camera producers and storage consumers
                for i in pbar:
                    device = device_list.get_device_by_index(i)
                    serial_number = device.get_device_info().get_serial_number()
                    pbar.set_postfix_str(f'{serial_number}')
                    self.video_buffers[i].name = f'cam{self.config[serial_number]["physical_index"]}_video_buffer'
                    self.imu_buffers[i].name = f'cam{self.config[serial_number]["physical_index"]}_imu_buffer'
                    self.producers.append(
                        CameraProducer(
                            camera_id=self.config[serial_number]['physical_index'],
                            camera_sn=serial_number,
                            config=self.config[serial_number],
                            ring_buffer=self.video_buffers[i],
                            imu_ring_buffer=self.imu_buffers[i],
                            termination_event=self.termination_event,
                            fs=self.fs,
                            start_barrier=start_barrier,
                            color_resolution=self.color_resolution,
                            color_fps=self.color_fps,
                            depth_resolution=self.depth_resolution,
                            depth_fps=self.depth_fps,
                            use_depth=serial_number in depth_serial_numbers,
                        )
                    )
                    self.consumers.append(
                        StorageConsumer(
                            ring_buffer=self.video_buffers[i],
                            fs=self.fs,
                            cam_name=f'{(self.config[serial_number]["physical_index"]):03d} ({serial_number})',
                            termination_event=self.termination_event,
                            batch_size=self.storage_batch_size,
                            store_frame_format=self.store_frame_format,
                            save_raw_frames=not self.store_using_h5,
                            nas_buffer=self.nas_buffers[i] if self.use_nas else None,
                            use_depth=serial_number in depth_serial_numbers,
                        )
                    )
                    self.imu_consumers.append(
                        IMUStorageConsumer(
                            ring_buffer=self.imu_buffers[i],
                            fs=self.fs,
                            cam_name=f'{(self.config[serial_number]["physical_index"]):03d} ({serial_number})',
                            termination_event=self.termination_event,
                            batch_size=1_0000,
                            store_frame_format=self.store_frame_format,
                            save_raw_frames=not self.store_using_h5,
                            nas_buffer=self.nas_buffers[i] if self.use_nas else None,
                        )
                    )
                for p in self.producers + self.consumers + self.imu_consumers:
                    p.start()
                self.sync_process = SyncProcess(
                    buffers=self.video_buffers,
                    file_path=self.fs.get_root() / 'multi_sync.info',
                    termination_event=self.termination_event,
                )
                self.sync_process.start()
                log(f'[{self.__class__.__name__}::start] Capture system started', 'info')
            except Exception as e:
                logging.critical(f"System startup failed: {e}", exc_info=e)
                self.stop()

    def stop(self):
        """Stop all processes and clean up resources"""
        if self.is_recording.value:
            self.is_recording.value = False
            self.termination_event.set()

            # Stop the producers
            time.sleep(0.1)
            for p in self.producers:
                if p.is_alive():
                    p.join(timeout=5)
            log(f"[{self.__class__.__name__}::stop] Camera producers stopped", 'debug')

            # Stop the consumers (allowing them to flush)
            time.sleep(1.0)
            for c in self.consumers:
                if c.is_alive() and c.ring_buffer.write_idx.value > c.ring_buffer.read_idx.value:
                    log(f"[{self.__class__.__name__}::stop] Flushing remaining frames for {c.cam_name}", 'info')
                    time.sleep(1)
                elif c.is_alive():
                    c.join(timeout=5)
            for c in self.imu_consumers:
                if c.is_alive() and c.ring_buffer.write_idx.value > c.ring_buffer.read_idx.value:
                    log(f"[{self.__class__.__name__}::stop] Flushing remaining IMU frames for {c.cam_name}", 'info')
                    time.sleep(1)
                elif c.is_alive():
                    c.join(timeout=5)
            log(f"[{self.__class__.__name__}::stop] Camera consumers stopped", 'debug')

            # Stop the sync process
            if self.sync_process is not None and self.sync_process.is_alive():
                self.sync_process.join(timeout=10)
                log(f"[{self.__class__.__name__}::stop] Sync process stopped", 'debug')

            # Close shared memory buffers
            time.sleep(0.1)
            for buf in self.video_buffers:
                buf.close()
            log(f"[{self.__class__.__name__}::stop] Shared memory buffers closed", 'debug')

            # Stop the NAS uploader
            if self.use_nas:
                self.nas_uploader.join(60 * 10)  # 10 minutes
                log(f"[{self.__class__.__name__}::stop] NAS uploader stopped", 'info')

                # Close NAS buffers
                time.sleep(0.1)
                for buf in self.nas_buffers:
                    buf.close()
                log(f"[{self.__class__.__name__}::stop] NAS buffers closed", 'debug')

            time.sleep(0.1)
            log(f"[{self.__class__.__name__}::stop] Capture system stopped", 'info')

    def load_config(self) -> OrderedDict:
        """Load camera configuration from JSON file"""
        log(f'[{self.__class__.__name__}::load_config] Loading config from {self.config_fpath}')
        with open(self.config_fpath) as f:
            config = json.load(f)
        primary_camera = int(config.get('primary', 2))
        config_out = []
        m = 1
        for device in config['devices']:
            sync_mode = 'FREE_RUN' if self.freerun_mode else ('PRIMARY' if int(device['physical_index']) == primary_camera else 'SECONDARY')
            # set depth delay
            if sync_mode == 'SECONDARY':
                # multiples of 160us
                depth_delay = (m % 8) * 160
                m += 1
            else:
                depth_delay = 0
            # setup sync config
            device['config'] = dict(
                mode=sync_mode,
                color_delay_us=0,
                depth_delay_us=depth_delay,
                # trigger_to_image_delay_us=0,
                # trigger_out_enable=True,
                # trigger_out_delay_us=0,
                # frames_per_trigger=1,
            )
            config_out.append((device['serial_number'], device))
        config_out = OrderedDict(config_out)
        return config_out

    def load_context(self):
        # noinspection PyArgumentList
        porb.Context.set_logger_level(porb.OBLogLevel.ERROR)
        ctx = porb.Context(str(PathUtils.config_path() / 'orbbec' / 'orbbec_sdk.xml'))
        ctx.enable_multi_device_sync(60_000)
        return ctx

    def generate_caliscope(self):
        """Generate calibration files using FFmpaeg"""
        caliscope_root = self.fs.get_root() / 'caliscope'
        caliscope_root.mkdir(exist_ok=True)

        intrinsic_root = caliscope_root / 'calibration' / 'intrinsic'
        intrinsic_root.mkdir(parents=True, exist_ok=True)

        for i in range(self.num_cameras):
            cam_name = f"cam_{i}"
            frames_root = self.fs.get_root() / 'raw_color' / cam_name
            if not frames_root.exists():
                continue

            with open(frames_root / 'ffmpeg_input.txt', 'wb') as outfile:
                for filename in sorted(frames_root.glob('*.jpg')):
                    outfile.write(f"file '{str(filename)}'\n".encode())
                    outfile.write(f"duration {1 / 30:.6f}\n".encode())

            output_file = intrinsic_root / f'port{i + 1}.mp4'
            cmd = f"ffmpeg -r 30 -f concat -safe 0 -i \"{str(frames_root / 'ffmpeg_input.txt')}\" " \
                  f"-c:v libx265 -pix_fmt yuv420p \"{str(output_file)}\""

            try:
                subprocess.run(cmd, shell=True, check=True)
                logging.info(f"Created calibration video for {cam_name} at {output_file}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to create calibration video for {cam_name}: {e}")


if __name__ == "__main__":
    system = CaptureSystem(
        config_name='s2x2_c10_freerun',
        num_cameras=2
    )

    try:
        system.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        system.stop()
