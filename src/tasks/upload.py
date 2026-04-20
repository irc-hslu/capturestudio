import shutil
from pathlib import Path
from typing import Union, Literal

import h5py
import numpy as np

from tasks import app, AutoRetryTask


def _from_capturestudio_cache_raw_to_nas_cache_h5_from_h5(capturestudio_cache_root: str, nas_cache_root: str, orbbec_id: str, cam_name: str, delete_local: bool = False):
    """
    Repack processed files from CAPTURESTUDIO_CACHE to h5 and move to NAS_CACHE, for one camera.
    The NAS_CACHE already contained h5 files; existing entries are kept and any processed data
    from CAPTURESTUDIO_CACHE is added or overwritten (no deletions on NAS side).

    Parameters
    ----------
    capturestudio_cache_root : Union[str, Path]
        Destination root, up until and incl. the session name , e.g. /mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1.
    nas_cache_root : Union[str, Path]
        The root of the nas, before 'Orbbec_Fempto_Bolt', e.g. /mnt/d/NAS_CACHE/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".
    cam_name : str
        Full camera name as appearing on the NAS, e.g. "001 (CL8K14100HK)".
    delete_local: bool
        If True, local files inside CAPTURE_CACHE will be removed on successful repacking to h5 and transferring to NAS_CACHE.
    """
    nas_cache_root = Path(nas_cache_root)
    nas_cache_cam_dir = nas_cache_root / 'Orbbec_Fempto_Bolt' / orbbec_id / cam_name
    capturestudio_cache_root = Path(capturestudio_cache_root)
    capturestudio_cache_cam_name = f'cam{int(cam_name.split(" ")[0]):02d}'
    capturestudio_cache_cam_dir = capturestudio_cache_root / 'orbbec' / capturestudio_cache_cam_name

    # detect available depth and depth-modality folders in CAPTURESTUDIO cache
    has_depth_cache = any(
        (capturestudio_cache_cam_dir / depth_key).exists()
        and max(
            len(list((capturestudio_cache_cam_dir / depth_key).glob('*.npy'))),
            len(list((capturestudio_cache_cam_dir / depth_key).glob('*.png')))
        ) > 0
        for depth_key in ['depth', 'depth_aligned']
    )
    depth_folder_names = [
        d.name for d in capturestudio_cache_cam_dir.glob('depth*')
        if d.is_dir() and d.name != 'depth'
    ]

    # iterate over existing h5 files; do not delete files or any existing entries,
    # only add or overwrite processed data from capturestudio cache
    for file in sorted(nas_cache_cam_dir.glob('*.h5'), key=lambda x: int(x.stem.split('-')[0])):
        with h5py.File(file, 'r+') as f:
            file_groups = list(f.keys())

            # if there is no color group, we cannot attach color-based modalities; keep file as is
            if 'color' not in file_groups:
                continue

            # use all existing color timestamps in this h5 file
            color_timestamps_in_h5 = sorted(f['color'].keys(), key=lambda x: int(x))

            # depth-related additions only if both cache depth and h5 depth exist
            if has_depth_cache and 'depth' in file_groups:
                depth_timestamps_in_h5 = sorted(f['depth'].keys(), key=lambda x: int(x))

                # add/overwrite depth modalities (e.g. depth_aligned, depth_filtering_*)
                for depth_folder_name in depth_folder_names:
                    if depth_folder_name not in file_groups:
                        f.create_group(depth_folder_name)
                        file_groups.append(depth_folder_name)
                    depth_group = f[depth_folder_name]

                    for depth_timestamp in depth_timestamps_in_h5:
                        cc_depth_path = capturestudio_cache_cam_dir / depth_folder_name / f'{depth_timestamp}.png'
                        if not cc_depth_path.exists():
                            continue
                        with open(cc_depth_path, 'rb') as df:
                            depth_bytes = df.read()
                        depth_arr = np.frombuffer(depth_bytes, dtype=np.uint8)
                        if depth_timestamp in depth_group:
                            depth_group[depth_timestamp][...] = depth_arr
                        else:
                            depth_group.create_dataset(depth_timestamp, data=depth_arr, dtype='uint8')

            # add/overwrite optical flows
            for flow in ['flow_bwd', 'flow_fwd']:
                if (capturestudio_cache_cam_dir / flow).exists():
                    if flow not in file_groups:
                        f.create_group(flow)
                        file_groups.append(flow)
                    flow_group = f[flow]

                    for color_timestamp in color_timestamps_in_h5:
                        cc_flow_path = capturestudio_cache_cam_dir / flow / f'{color_timestamp}.png'
                        if not cc_flow_path.exists():
                            continue
                        with open(cc_flow_path, 'rb') as ff:
                            flow_bytes = ff.read()
                        flow_arr = np.frombuffer(flow_bytes, dtype=np.uint8)
                        if color_timestamp in flow_group:
                            flow_group[color_timestamp][...] = flow_arr
                        else:
                            flow_group.create_dataset(color_timestamp, data=flow_arr, dtype='uint8')

            # add/overwrite masks
            if (capturestudio_cache_cam_dir / 'mask').exists():
                if 'mask' not in file_groups:
                    f.create_group('mask')
                    file_groups.append('mask')
                mask_group = f['mask']

                for color_timestamp in color_timestamps_in_h5:
                    cc_mask_path = capturestudio_cache_cam_dir / 'mask' / f'{color_timestamp}.jpg'
                    if not cc_mask_path.exists():
                        continue
                    with open(cc_mask_path, 'rb') as mf:
                        mask_bytes = mf.read()
                    mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)

                    if color_timestamp in mask_group:
                        mask_group[color_timestamp][...] = mask_arr
                    else:
                        mask_group.create_dataset(color_timestamp, data=mask_arr, dtype='uint8')

    # delete capturestudio cache files if requested (NAS contents are not touched)
    if delete_local:
        shutil.rmtree(capturestudio_cache_cam_dir)
        # if the parent directory (orbbec) becomes empty, remove it as well
        parent = capturestudio_cache_cam_dir.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            # parent not empty or other os issue
            pass

    return True


def _from_capturestudio_cache_raw_to_nas_cache_h5_from_raw(capturestudio_cache_root: str, nas_cache_root: str, orbbec_id: str, cam_name: str, h5_batch_size: int = 30, delete_local: bool = False):
    """
    Pack processed files from CAPTURESTUDIO_CACHE to h5 and move to NAS_CACHE, for one camera.
    BUT: the NAS_CACHE folder contained raw files, which need to be removed, and data need to be packed h5 files in batches.

    Parameters
    ----------
    capturestudio_cache_root : Union[str, Path]
        Destination root, up until and incl. the session name , e.g. /mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1.
    nas_cache_root : Union[str, Path]
        The root of the nas, before 'Orbbec_Fempto_Bolt', e.g. /mnt/d/NAS_CACHE/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".
    cam_name : str
        Full camera name as appearing on the NAS, e.g. "001 (CL8K14100HK)".
    h5_batch_size: int
        Batch size for storing multiple raw files in one h5 file.
    delete_local: bool
        If True, local files inside CAPTURE_CACHE will be removed on successful repacking to h5 and transferring to NAS_CACHE.
    """
    from bisect import bisect_left

    nas_cache_root = Path(nas_cache_root)
    base_dir = nas_cache_root / 'Orbbec_Fempto_Bolt' / orbbec_id
    nas_cache_cam_dir = base_dir / cam_name
    nas_cache_cam_dir.mkdir(parents=True, exist_ok=True)

    # NAS raw dirs (original location)
    nas_raw_color_dir = base_dir / 'raw_color' / cam_name
    nas_raw_depth_dir = base_dir / 'raw_depth' / cam_name
    # nas_raw_ir_dir = base_dir / 'raw_ir' / cam_name

    capturestudio_cache_root = Path(capturestudio_cache_root)
    capturestudio_cache_cam_name = f'cam{int(cam_name.split(" ")[0]):02d}'
    capturestudio_cache_cam_dir = capturestudio_cache_root / 'orbbec' / capturestudio_cache_cam_name

    # ---- Discover NAS raw frames (ground truth for packing) ----
    nas_color_files = sorted(nas_raw_color_dir.glob('*.jpg'), key=lambda p: int(p.stem)) if nas_raw_color_dir.exists() else []
    nas_color_ts = [p.stem for p in nas_color_files]
    nas_color_ts_int = [int(x) for x in nas_color_ts]

    # Depth (optional, from NAS raw_depth)
    nas_depth_type = None
    nas_depth_files = []
    if nas_raw_depth_dir.exists():
        depth_png_files = sorted(nas_raw_depth_dir.glob('*.png'), key=lambda p: int(p.stem))
        depth_npy_files = sorted(nas_raw_depth_dir.glob('*.npy'), key=lambda p: int(p.stem))
        if depth_png_files:
            nas_depth_type = 'png'
            nas_depth_files = depth_png_files
        elif depth_npy_files:
            nas_depth_type = 'npy'
            nas_depth_files = depth_npy_files
    nas_depth_ts = [p.stem for p in nas_depth_files]
    nas_depth_ts_int = [int(x) for x in nas_depth_ts]
    nas_depth_ts_set = set(nas_depth_ts)

    # # ---- Discover computed quantities in CS cache (overlay only) ----
    # cs_color_dir = capturestudio_cache_cam_dir / 'color'
    # cs_color_ts_set = set()
    # if cs_color_dir.exists():
    #     cs_color_ts_set = {p.stem for p in cs_color_dir.glob('*.jpg')}
    cs_depth_dir = (capturestudio_cache_cam_dir / 'depth') if (capturestudio_cache_cam_dir / 'depth').exists() else (capturestudio_cache_cam_dir / 'depth_aligned')
    cs_depth_type = None
    cs_depth_ts_set = set()
    if cs_depth_dir.exists():
        cs_depth_png_files = list(cs_depth_dir.glob('*.png'))
        cs_depth_npy_files = list(cs_depth_dir.glob('*.npy'))
        if cs_depth_png_files:
            cs_depth_type = 'png'
            cs_depth_ts_set = {p.stem for p in cs_depth_png_files}
        elif cs_depth_npy_files:
            cs_depth_type = 'npy'
            cs_depth_ts_set = {p.stem for p in cs_depth_npy_files}

    cs_depth_modality_dirs = {
        d.name: d for d in capturestudio_cache_cam_dir.glob('depth*')
        if d.is_dir() and d.name != cs_depth_dir.name
    }

    cs_color_modality_dirs = {}
    for modality in ['mask', 'flow_fwd', 'flow_bwd']:
        mdir = capturestudio_cache_cam_dir / modality
        if mdir.exists():
            cs_color_modality_dirs[modality] = mdir

    # ---- Helper: upsert dataset (create or overwrite). Handles shape changes by delete+recreate within the file. ----
    def _upsert_u8_dataset(group: h5py.Group, name: str, data_u8: np.ndarray):
        if name in group:
            ds = group[name]
            if ds.dtype == np.uint8 and ds.shape == data_u8.shape:
                ds[...] = data_u8
                return
            del group[name]
        group.create_dataset(name, data=data_u8, dtype='uint8')

    def _upsert_u16_dataset(group: h5py.Group, name: str, data_u16: np.ndarray):
        if name in group:
            ds = group[name]
            if ds.dtype == np.uint16 and ds.shape == data_u16.shape:
                ds[...] = data_u16
                return
            del group[name]
        group.create_dataset(name, data=data_u16, dtype='uint16')

    # ---- Simple color->depth matching: depth_ts >= color_ts and (depth_ts - color_ts) < 5 ----
    # Map color timestamp string -> matched depth timestamp string or None
    color_to_depth_ts = {}
    if nas_depth_ts_int:
        for ts_str, t_int in zip(nas_color_ts, nas_color_ts_int):
            i = bisect_left(nas_depth_ts_int, t_int)  # first depth_ts >= color_ts
            if i < len(nas_depth_ts_int) and (nas_depth_ts_int[i] - t_int) < 5:
                color_to_depth_ts[ts_str] = str(nas_depth_ts_int[i])
            else:
                color_to_depth_ts[ts_str] = None
    else:
        for ts_str in nas_color_ts:
            color_to_depth_ts[ts_str] = None

    # ---- Pack NAS raw frames into h5 in batches, then overlay CS computed modalities where available ----
    num_frames = len(nas_color_ts)
    for start_idx in range(0, num_frames, h5_batch_size):
        batch_color_ts = nas_color_ts[start_idx:start_idx + h5_batch_size]
        if not batch_color_ts:
            continue

        h5_name = f"{batch_color_ts[0]}-{batch_color_ts[-1]}.h5"
        h5_path = nas_cache_cam_dir / h5_name

        with h5py.File(h5_path, 'a') as hf:
            # Ensure base groups
            color_group = hf['color'] if 'color' in hf else hf.create_group('color')
            depth_group = None
            if nas_depth_type is not None:
                depth_group = hf['depth'] if 'depth' in hf else hf.create_group('depth')

            # Ensure modality groups
            depth_modality_groups = {}
            for name in cs_depth_modality_dirs.keys():
                if name not in hf:
                    hf.create_group(name)
                depth_modality_groups[name] = hf[name]

            mask_group = None
            if 'mask' in cs_color_modality_dirs:
                mask_group = hf['mask'] if 'mask' in hf else hf.create_group('mask')

            flow_groups = {}
            for flow_name in ['flow_fwd', 'flow_bwd']:
                if flow_name in cs_color_modality_dirs:
                    flow_groups[flow_name] = hf[flow_name] if flow_name in hf else hf.create_group(flow_name)

            # Store per-file sync mapping (color_ts -> depth_ts or -1) for robust unpacking
            sync_group = hf['sync'] if 'sync' in hf else hf.create_group('sync')
            c2d_group = sync_group['color_to_depth'] if 'color_to_depth' in sync_group else sync_group.create_group('color_to_depth')

            for ts in batch_color_ts:
                # 1) NAS raw color -> h5/color
                nas_color_path = nas_raw_color_dir / f"{ts}.jpg"
                if nas_color_path.exists():
                    with open(nas_color_path, 'rb') as f:
                        b = f.read()
                    _upsert_u8_dataset(color_group, ts, np.frombuffer(b, dtype=np.uint8))

                # Mapping for this color ts (depth ts or -1)
                depth_ts = color_to_depth_ts.get(ts, None)
                depth_ts_val = np.int64(int(depth_ts)) if depth_ts is not None else np.int64(-1)
                if ts in c2d_group:
                    ds = c2d_group[ts]
                    if ds.dtype == np.int64 and ds.shape == ():
                        ds[()] = depth_ts_val
                    else:
                        del c2d_group[ts]
                        c2d_group.create_dataset(ts, data=depth_ts_val, dtype=np.int64)
                else:
                    c2d_group.create_dataset(ts, data=depth_ts_val, dtype=np.int64)

                # 2) NAS raw depth -> h5/depth (keyed by DEPTH timestamp)
                if depth_group is not None and depth_ts is not None and depth_ts in nas_depth_ts_set:
                    if nas_depth_type == 'png':
                        nas_depth_path = nas_raw_depth_dir / f"{depth_ts}.png"
                        if nas_depth_path.exists():
                            with open(nas_depth_path, 'rb') as f:
                                b = f.read()
                            _upsert_u8_dataset(depth_group, depth_ts, np.frombuffer(b, dtype=np.uint8))
                    elif nas_depth_type == 'npy':
                        nas_depth_path = nas_raw_depth_dir / f"{depth_ts}.npy"
                        if nas_depth_path.exists():
                            _upsert_u16_dataset(depth_group, depth_ts, np.load(nas_depth_path).astype(np.uint16))

                # 3) Overlay CS depth (for matched depth_ts)
                if depth_group is not None and depth_ts is not None and depth_ts in cs_depth_ts_set:
                    if cs_depth_type == 'png':
                        cs_depth_path = cs_depth_dir / f"{depth_ts}.png"
                        if cs_depth_path.exists():
                            with open(cs_depth_path, 'rb') as f:
                                b = f.read()
                            _upsert_u8_dataset(depth_group, depth_ts, np.frombuffer(b, dtype=np.uint8))
                    elif cs_depth_type == 'npy':
                        cs_depth_path = cs_depth_dir / f"{depth_ts}.npy"
                        if cs_depth_path.exists():
                            _upsert_u16_dataset(depth_group, depth_ts, np.load(cs_depth_path).astype(np.uint16))

                # 4) Overlay CS depth modalities (for matched depth_ts)
                if depth_ts is not None:
                    for name, ddir in cs_depth_modality_dirs.items():
                        cs_mod_path = ddir / f"{depth_ts}.png"
                        if cs_mod_path.exists():
                            with open(cs_mod_path, 'rb') as f:
                                b = f.read()
                            _upsert_u8_dataset(depth_modality_groups[name], depth_ts, np.frombuffer(b, dtype=np.uint8))

                # 5) Overlay CS mask (keyed by COLOR timestamp)
                if mask_group is not None:
                    cs_mask_path = cs_color_modality_dirs['mask'] / f"{ts}.jpg"
                    if cs_mask_path.exists():
                        with open(cs_mask_path, 'rb') as f:
                            b = f.read()
                        _upsert_u8_dataset(mask_group, ts, np.frombuffer(b, dtype=np.uint8))

                # 6) Overlay CS optical flow (keyed by COLOR timestamp)
                for flow_name in ['flow_fwd', 'flow_bwd']:
                    if flow_name in flow_groups:
                        cs_flow_path = cs_color_modality_dirs[flow_name] / f"{ts}.png"
                        if cs_flow_path.exists():
                            with open(cs_flow_path, 'rb') as f:
                                b = f.read()
                            _upsert_u8_dataset(flow_groups[flow_name], ts, np.frombuffer(b, dtype=np.uint8))

    # ---- Rename NAS raw cam dirs to "_<cam_name>" first; only if ALL cam dirs are prefixed, prefix parent raw_* dirs ----
    for raw_name in ['raw_color', 'raw_depth', 'raw_ir']:
        raw_parent = base_dir / raw_name
        if not raw_parent.exists():
            continue

        # Prefix this camera dir (directory rename only; contents untouched)
        cam_dir = raw_parent / cam_name
        cam_dir_prefixed = raw_parent / f"_{cam_name}"
        if cam_dir.exists() and not cam_dir_prefixed.exists():
            cam_dir.rename(cam_dir_prefixed)

        # If every remaining camera dir under raw_parent is already prefixed, then prefix the parent
        try:
            cam_dirs = [p for p in raw_parent.iterdir() if p.is_dir() and (p.name.startswith('cam') or p.name.startswith('_cam'))]
        except OSError:
            continue
        if cam_dirs and all(p.name.startswith('_') for p in cam_dirs):
            parent_prefixed = base_dir / f"_{raw_name}"
            if not parent_prefixed.exists():
                raw_parent.rename(parent_prefixed)

    # ---- Optionally delete local capturestudio cache (NAS untouched) ----
    if delete_local and capturestudio_cache_cam_dir.exists():
        shutil.rmtree(capturestudio_cache_cam_dir)
        parent = capturestudio_cache_cam_dir.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass

    return True


def _from_nas_cache_h5_to_nas_h5(nas_cache_root: str, nas_root: str, orbbec_id: str, cam_name: str, current_nas_format: str, delete_local: bool = False):
    """
    Transfer packed h5 files from NAS_CACHE to NAS for th given camera.

    Parameters
    ----------
    nas_cache_root: str
        The root of NAS_CACHE, before 'Orbbec_Fempto_Bolt', e.g. /media/charisoudis/nas_captures/Captures_Apr_May_2025.
    nas_root : str
        The root of the NAS, before 'Orbbec_Fempto_Bolt', e.g. /media/charisoudis/nas_captures/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".
    cam_name: str
        Full camera name as appearing on the NAS, e.g. "001 (CL8K14100HK)".
    current_nas_format: Literal['raw', 'h5']
        The stored format in the NAS atm. One of 'raw', 'h5'.
    delete_local: bool, optional
        If True, empty the NAS_CACHE for this camera's data.
    """
    nas_cache_root = Path(nas_cache_root)
    nas_cache_cam_dir = nas_cache_root / 'Orbbec_Fempto_Bolt' / orbbec_id / cam_name
    nas_root = Path(nas_root)
    nas_session_root = nas_root / 'Orbbec_Fempto_Bolt' / orbbec_id
    nas_session_root.mkdir(parents=True, exist_ok=True)

    # Destination camera dir for h5 format on NAS
    nas_cam_h5_dir = nas_session_root / cam_name
    nas_cam_h5_dir.mkdir(parents=True, exist_ok=True)

    # --- COPY: from NAS_CACHE/cam_name/*.h5 -> NAS/cam_name/* (overwrite same filenames) ---
    if nas_cache_cam_dir.exists():
        for src in sorted(nas_cache_cam_dir.glob('*.h5')):
            dst = nas_cam_h5_dir / src.name
            if dst.exists() and not dst.is_dir():
                if dst.with_suffix(f'{dst.suffix}.old').exists():
                    dst.with_suffix(f'{dst.suffix}.old').unlink()
                dst.rename(dst.with_suffix(f'{dst.suffix}.old'))
            shutil.copy2(src, dst)

    # --- If NAS was in raw format, rename raw_*/<cam_name> -> raw_*/_<cam_name> for this cam (no deletion) ---
    if current_nas_format == 'raw':
        for raw_name in ['raw_color', 'raw_depth', 'raw_ir']:
            raw_parent = nas_session_root / raw_name
            if not raw_parent.exists():
                continue
            src_cam_dir = raw_parent / cam_name
            dst_cam_dir = raw_parent / f"_{cam_name}"
            if src_cam_dir.exists() and not dst_cam_dir.exists():
                src_cam_dir.rename(dst_cam_dir)
            try:
                cam_dirs = [p for p in raw_parent.iterdir() if p.is_dir() and (p.name.startswith('cam') or p.name.startswith('_cam'))]
            except OSError:
                continue
            if cam_dirs and all(p.name.startswith('_') for p in cam_dirs):
                parent_prefixed = nas_session_root / f"_{raw_name}"
                if not parent_prefixed.exists():
                    raw_parent.rename(parent_prefixed)

    # --- EMPTY: delete local copy (NAS_CACHE) ---
    if delete_local and nas_cache_cam_dir.exists():
        shutil.rmtree(nas_cache_cam_dir)
        p = nas_cache_cam_dir.parent
        if p.exists() and not any(p.iterdir()):
            p.rmdir()


def _detect_nas_format(nas_root: str, orbbec_id: str) -> Literal['raw', 'h5']:
    """
    Detect the format of the Orbbec recording on the NAS.

    Parameters
    ----------
    nas_root : Union[str, Path]
        The root of the nas, before 'Orbbec_Fempto_Bolt', e.g. /media/charisoudis/nas_captures/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".

    Returns
    -------
    str
        'raw' if the recording is in raw format, 'h5' if it is in h5 format.
    """
    nas_path = Path(nas_root) / 'Orbbec_Fempto_Bolt' / orbbec_id
    if (nas_path / 'raw_color').exists():
        return 'raw'
    # # find the first camer folder (i.e. a folder that starts with numbers3 digits, followed by a space and a camera SN in parentheses)
    # first_camera_folder = next((d for d in nas_path.iterdir() if d.is_dir() and d.name[0:3].isdigit() and ' (' in d.name), None)
    # if first_camera_folder is None:
    #     raise ValueError(f"Could not find any camera folder in {nas_path}. Expected a folder starting with 3 digits followed by a space and a camera SN in parentheses.")
    return 'h5'


@app.task(name="upload.upload_to_nas", base=AutoRetryTask)
def upload_to_nas(capturestudio_cache_root: str, nas_cache_root: str, nas_root: str, orbbec_id: str, cam_name: str, delete_capturestudio_cache: bool = False, delete_nas_cache: bool = False):
    """
    Run the download task based on the provided parameters.

    Parameters
    ----------
    nas_root : str
        The root of the nas, before 'Orbbec_Fempto_Bolt', e.g. /media/charisoudis/nas_captures/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".
    cam_name : str
        Full camera name as appearing on the NAS, e.g. "001 (CL8K14100HK)".
    capturestudio_cache_root : str
        Destination root, up until and incl. the session name , e.g. /root/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1.
    nas_cache_root : str
        The root of the nas cache, before 'Orbbec_Fempto_Bolt', e.g. /root/NAS_CACHE/Captures_Apr_May_2025.
        If not provided, it will be set to the same value as `capturestudio_cache_root`.
    delete_capturestudio_cache: bool, optional
        If True, capturestudio cache will be emptied after packing/transfer to NAS_CACHE.
    delete_nas_cache: bool, optional
        If True, NAS_CACHE will be emptied after packing/transfer to NAS.
    """
    (Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id).mkdir(parents=True, exist_ok=True)

    nas_format = _detect_nas_format(nas_root, orbbec_id)
    nas_cache_format = _detect_nas_format(nas_cache_root, orbbec_id)
    all_cam_dirs = sorted([int(x.name.replace('cam', '')) for x in (Path(capturestudio_cache_root) / 'orbbec').glob('cam*')])
    is_first_camera = all_cam_dirs[0] == int(cam_name.split(' ')[0])

    # CAPTURESTUDIO_CACHE (raw) --> NAS_CACHE (h5)
    if nas_cache_format == 'raw':
        _from_capturestudio_cache_raw_to_nas_cache_h5_from_raw(capturestudio_cache_root, nas_cache_root, orbbec_id, cam_name, delete_local=delete_capturestudio_cache)
    else:
        _from_capturestudio_cache_raw_to_nas_cache_h5_from_h5(capturestudio_cache_root, nas_cache_root, orbbec_id, cam_name, delete_local=delete_capturestudio_cache)
    if is_first_camera:
        # transfer other files
        for fn in ['multi_sync.info', 'multiview_color.mp4', 'multiview_depth.mp4', 'session_metadata.json']:
            if (Path(capturestudio_cache_root) / 'orbbec' / fn).exists():
                shutil.copy(Path(capturestudio_cache_root) / 'orbbec' / fn, Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / fn)
                if delete_capturestudio_cache:
                    (Path(capturestudio_cache_root) / 'orbbec' / fn).unlink(missing_ok=True)

    # NAS_CACHE (h5) --> NAS (h5)
    _from_nas_cache_h5_to_nas_h5(nas_cache_root, nas_root, orbbec_id, cam_name, current_nas_format=nas_format, delete_local=delete_nas_cache)
    if is_first_camera:
        # transfer other files
        for fn in ['multi_sync.info', 'multiview_color.mp4', 'multiview_depth.mp4', 'session_metadata.json']:
            if (Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / fn).exists():
                shutil.copy(Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / fn, Path(nas_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / fn)
                if delete_nas_cache:
                    (Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / fn).unlink(missing_ok=True)

    return None
