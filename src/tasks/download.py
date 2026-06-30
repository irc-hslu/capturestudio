import json
import os
from pathlib import Path
from typing import Union, Literal

import cv2
import h5py
import numpy as np

from tasks import app, AutoRetryTask
from utils.misc import log, PathUtils


def sendfile_copy(src: Path, dst: Path):
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        offset = 0
        size = src.stat().st_size
        while offset < size:
            sent = os.sendfile(fdst.fileno(), fsrc.fileno(), offset, size - offset)
            offset += sent


def _extract_orbbec_parameters_for_cam(captures_path_cam: Union[str, Path], cam_data: dict) -> None:
    """
    Extract camera parameters from JSON and store as numpy arrays.

    Parameters
    ----------
    captures_path_cam : Union[str, Path]
        Camera path where parameter folder will be created.
    cam_data : dict
        JSON data containing camera profile parameters.

    Returns
    -------
    None

    """
    captures_path_cam = Path(captures_path_cam)
    param_dir = captures_path_cam / 'parameters'
    param_dir.mkdir(parents=True, exist_ok=True)

    # Color intrinsics (3x3)
    if not (param_dir / 'color_intri.npy').exists():
        c_intri = np.array([
            [cam_data['color']['intrinsic']['fx'], 0, cam_data['color']['intrinsic']['cx']],
            [0, cam_data['color']['intrinsic']['fy'], cam_data['color']['intrinsic']['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        np.save(str(param_dir / 'color_intri.npy'), c_intri)

    # Color distortion (8,)
    if not (param_dir / 'color_dist.npy').exists():
        c_dist = np.array([
            cam_data['color']['distortion']['k1'],
            cam_data['color']['distortion']['k2'],
            cam_data['color']['distortion']['p1'],
            cam_data['color']['distortion']['p2'],
            cam_data['color']['distortion']['k3'],
            cam_data['color']['distortion'].get('k4', 0),
            cam_data['color']['distortion'].get('k5', 0),
            cam_data['color']['distortion'].get('k6', 0)
        ], dtype=np.float32)
        np.save(str(param_dir / 'color_dist.npy'), c_dist)

    if 'depth' in cam_data:
        # Depth intrinsics (3x3)
        if not (param_dir / 'depth_intri.npy').exists():
            d_intri = np.array([
                [cam_data['depth']['intrinsic']['fx'], 0, cam_data['depth']['intrinsic']['cx']],
                [0, cam_data['depth']['intrinsic']['fy'], cam_data['depth']['intrinsic']['cy']],
                [0, 0, 1]
            ], dtype=np.float32)
            np.save(str(param_dir / 'depth_intri.npy'), d_intri)

        # Depth distortion (8,)
        if not (param_dir / 'depth_dist.npy').exists():
            d_dist = np.array([
                cam_data['depth']['distortion']['k1'],
                cam_data['depth']['distortion']['k2'],
                cam_data['depth']['distortion']['p1'],
                cam_data['depth']['distortion']['p2'],
                cam_data['depth']['distortion']['k3'],
                cam_data['depth']['distortion'].get('k4', 0),
                cam_data['depth']['distortion'].get('k5', 0),
                cam_data['depth']['distortion'].get('k6', 0)
            ], dtype=np.float32)
            np.save(str(param_dir / 'depth_dist.npy'), d_dist)

        # Depth to color extrinsic (4x4 homogeneous)
        if not (param_dir / 'depth_extri2color.npy').exists():
            extri2color = cam_data['depth'].get('extrinsic_to_color', {
                "rot": [
                    [
                        0.9944801926612854,
                        0.0014649699442088604,
                        0.0010425772052258253
                    ],
                    [
                        -0.0015662702498957515,
                        0.9944796562194824,
                        0.10491778701543808
                    ],
                    [
                        -0.0008831204031594098,
                        -0.10491925477981567,
                        0.994480311870575
                    ]
                ],
                "trans": [
                    -32.3697509765625,
                    -1.0761730670928955,
                    1.671700358390808
                ]
            })
            rot = np.array(extri2color['rot'], dtype=np.float32)
            trans = np.array(extri2color['trans'], dtype=np.float32)
            extri = np.eye(4, dtype=np.float32)
            extri[:3, :3] = rot
            extri[:3, 3] = trans
            np.save(str(param_dir / 'depth_extri2color.npy'), extri)
    return True


def _from_nas_raw_to_capturestudio_cache_raw(nas_root: str, orbbec_id: str, cam_name: str, capturestudio_cache_root: str):
    """
    Transfer raw files from NAS to cache, and pack them as .h5 files.

    Parameters
    ----------
    nas_root : Union[str, Path]
        The root of the nas, before 'Orbbec_Fempto_Bolt', e.g. /media/charisoudis/nas_captures/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".
    cam_name : str
        Full camera name as appearing on the NAS, e.g. "001 (CL8K14100HK)".
    capturestudio_cache_root : Union[str, Path]
        Destination root, up until and incl. the session name , e.g. /mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1.
    """
    nas_root = Path(nas_root)
    capturestudio_cache_root = Path(capturestudio_cache_root)

    # extract d2c parameters from session_metadata.json
    with open(Path(nas_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / 'session_metadata.json', 'r') as f:
        json_data = json.load(f)['session_metadata']['cam_profiles']
    serial = cam_name.split(' ', 1)[1].strip('() ')
    _extract_orbbec_parameters_for_cam(Path(capturestudio_cache_root) / 'orbbec' / f'cam{int(cam_name.split(" ", 1)[0]):02d}', json_data[serial])

    # create copy files list
    color_files_to_be_copied = []
    depth_files_to_be_copied = []
    for sub_folder_idx, subfolder in enumerate(['raw_color', 'mask', 'flow_fwd', 'flow_bwd', 'raw_depth', 'depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal']):
        nas_path = nas_root / 'Orbbec_Fempto_Bolt' / orbbec_id / subfolder / cam_name
        if nas_path.exists():
            cache_path = capturestudio_cache_root / 'orbbec' / f'cam{int(cam_name.split(" ", 1)[0]):02d}' / subfolder.replace('raw_', '')
            if not cache_path.exists():
                cache_path.mkdir(parents=True, exist_ok=True)
                for item in nas_path.iterdir():
                    if item.is_file() and not (cache_path / item.name).exists():
                        (color_files_to_be_copied if 'depth' not in subfolder else depth_files_to_be_copied).append((item, cache_path / item.name))

    # copy files and show progress
    #   - color files
    for i, (src, dst) in enumerate(color_files_to_be_copied):
        sendfile_copy(src, dst)
    #   - depth files
    if len(depth_files_to_be_copied) > 0:
        for i, (src, dst) in enumerate(depth_files_to_be_copied):
            sendfile_copy(src, dst)
    return True


def _from_nas_h5_to_nas_cache_h5(nas_root: str, orbbec_id: str, cam_name: str, nas_cache_root: str):
    """
    Downloads a camera folder from NAS to the NAS Cache path.

    Parameters
    ----------
    nas_root : Union[str, Path]
        The root of the nas, before 'Orbbec_Fempto_Bolt', e.g. /media/charisoudis/nas_captures/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".
    cam_name : str
        Full camera name as appearing on the NAS, e.g. "001 (CL8K14100HK)".
    nas_cache_root : Union[str, Path]
        Destination root, before 'Orbbec_Fempto_Bolt' , e.g. /mnt/sdata/NAS_CACHE/.
    """
    nas_path = Path(nas_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / cam_name
    nas_cache_path = Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / cam_name
    if not nas_cache_path.exists():
        nas_cache_path.mkdir(parents=True, exist_ok=True)
    h5_files_to_be_copied = []
    for item in nas_path.glob('*.h5'):
        if item.is_file():
            if (nas_cache_path / item.name).exists():
                # check file size
                if (nas_cache_path / item.name).stat().st_size != item.stat().st_size:
                    # delete the existing file and copy the new one
                    (nas_cache_path / item.name).unlink()
                else:
                    # file already exists and is the same size, skip copying
                    continue
            # add to copy list
            h5_files_to_be_copied.append((item, nas_cache_path / item.name))

    # Download session_metadata.json (only for first camera)
    if int(cam_name.split(' ', 1)[0]) == 1:
        nas_metadata_path = Path(nas_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / 'session_metadata.json'
        nas_cache_metadata_path = Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / 'session_metadata.json'
        if nas_metadata_path.exists() and not nas_cache_metadata_path.exists():
            h5_files_to_be_copied.append((nas_metadata_path, nas_cache_metadata_path))

    # start copying files and show progress
    for i, (src, dst) in enumerate(h5_files_to_be_copied):
        sendfile_copy(src, dst)
    return True


def _from_nas_cache_h5_to_capturestudio_cache_raw(nas_cache_root: str, orbbec_id: str, cam_name: str, capturestudio_cache_root: str):
    """
    Downloads a camera folder from NAS Cache to CaptureStudio Cache path.

    Parameters
    ----------
    nas_cache_root : Union[str, Path]
        The root of the nas cache, before 'Orbbec_Fempto_Bolt', e.g. /mnt/sdata/NAS_CACHE/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".
    cam_name : str
        Full camera name as appearing on the NAS and the NAS_CACHE, e.g. "001 (CL8K14100HK)".
    capturestudio_cache_root : Union[str, Path]
        Destination root, up until and incl. the session name , e.g. /mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1.
    """
    nas_cache_path = Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / cam_name
    capturestudio_cache_path = Path(capturestudio_cache_root) / 'orbbec' / f'cam{int(cam_name.split(" ", 1)[0]):02d}'
    if not capturestudio_cache_path.exists():
        capturestudio_cache_path.mkdir(parents=True, exist_ok=True)

    # extract d2c parameters from session_metadata.json
    with open(Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / 'session_metadata.json', 'r') as f:
        json_data = json.load(f)['session_metadata']['cam_profiles']
    serial = cam_name.split(' ', 1)[1].strip('() ')
    _extract_orbbec_parameters_for_cam(Path(capturestudio_cache_root) / 'orbbec' / f'cam{int(cam_name.split(" ", 1)[0]):02d}', json_data[serial])
    # copy session_metadata.json to capturestudio_cache_path]
    if not (capturestudio_cache_path.parent / 'session_metadata.json').exists():
        sendfile_copy(Path(nas_cache_root) / 'Orbbec_Fempto_Bolt' / orbbec_id / 'session_metadata.json', capturestudio_cache_path.parent / 'session_metadata.json')

    all_h5_files = sorted(nas_cache_path.glob('*.h5'), key=lambda x: int(x.stem.split('-')[0]))
    for h5_i, h5_path in enumerate(all_h5_files):
        # Unpack the h5 file to raw files
        with h5py.File(h5_path, 'r') as hf:
            color_keys = [str(k) for k in hf['color'].keys()]
            depth_keys = list(hf['depth'].keys()) if 'depth' in hf else []
            for i_global, color_ts in enumerate(color_keys):
                i = i_global
                try:
                    color = cv2.imdecode(hf['color'][color_ts][:], cv2.IMREAD_COLOR)
                except cv2.error:
                    log(f'[from_nas_cache_h5_to_capturestudio_cache_raw] Error decoding color image for timestamp {color_ts} in file {h5_path.name}. Duplicating previous image.', 'warning')
                    i = i_global - 1
                    color_ts = color_keys[i]
                    color = cv2.imdecode(hf['color'][color_ts][:], cv2.IMREAD_COLOR)
                # store color
                if not (capturestudio_cache_path / 'color').exists():
                    (capturestudio_cache_path / 'color').mkdir()
                cv2.imwrite(str(capturestudio_cache_path / 'color' / f'{color_ts}.jpg'), color, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # check for other color modalities
                for modality in ['mask', 'flow_fwd', 'flow_bwd']:
                    if modality in hf and color_ts in hf[modality]:
                        if not (capturestudio_cache_path / modality).exists():
                            (capturestudio_cache_path / modality).mkdir()
                        modality_data = hf[modality][color_ts]
                        if modality == 'mask':
                            with open(capturestudio_cache_path / modality / f'{color_ts}.jpg', "wb") as f:
                                f.write(modality_data[...].tobytes())
                        elif modality in ['flow_fwd', 'flow_bwd']:
                            with open(capturestudio_cache_path / modality / f'{color_ts}.png', "wb") as f:
                                f.write(modality_data[...].tobytes())

                if 'depth' in hf and i < len(depth_keys):  # Prevent IndexError if there are fewer depth frames than color frames
                    depth_ts = depth_keys[i]
                    depth_data = hf['depth'][depth_ts]
                    if not (capturestudio_cache_path / 'depth').exists():
                        (capturestudio_cache_path / 'depth').mkdir()
                    depth_png_path = capturestudio_cache_path / 'depth' / f'{depth_ts}.png'
                    # Depth datasets are stored either as:
                    #  - uint8 1D array of PNG bytes (preferred for PNG storage), or
                    #  - uint16 2D array of depth values (numeric storage)
                    depth_data_is_png = (depth_data.dtype == np.uint8 and depth_data.ndim == 1)
                    if depth_data_is_png:
                        with open(depth_png_path, "wb") as f:
                            f.write(depth_data[...].tobytes())
                    else:
                        depth = np.asarray(depth_data[:], dtype=np.uint16)
                        if not (capturestudio_cache_path / 'depth').exists():
                            (capturestudio_cache_path / 'depth').mkdir()
                        PathUtils.write_file(depth_png_path, depth, png_type='depth')

                    # other depth modalities
                    for modality in ['depth_aligned', 'depth_filtering_bilateral_spatial', 'depth_filtering_bilateral_temporal']:
                        if modality in hf and depth_ts in hf[modality]:
                            if not (capturestudio_cache_path / modality).exists():
                                (capturestudio_cache_path / modality).mkdir()
                            # Assume that all depth modalities are encoded as 16-bit PNGs in the h5 file
                            modality_data_png_bytes = hf[modality][depth_ts]
                            modality_png_path = capturestudio_cache_path / modality / f'{depth_ts}.png'
                            with open(modality_png_path, "wb") as f:
                                f.write(modality_data_png_bytes[...].tobytes())
    return True


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

    # find the first camer folder (i.e. a folder that starts with numbers3 digits, followed by a space and a camera SN in parentheses)
    first_camera_folder = next((d for d in nas_path.iterdir() if d.is_dir() and d.name[0:3].isdigit() and ' (' in d.name), None)
    if first_camera_folder is None:
        raise ValueError(f"Could not find any camera folder in {nas_path}. Expected a folder starting with 3 digits followed by a space and a camera SN in parentheses.")
    return 'h5'


@app.task(name="download.download_from_nas", base=AutoRetryTask)
def download_from_nas(nas_root: str, orbbec_id: str, cam_name: str, capturestudio_cache_root: str, nas_cache_root: str = None):
    """
    Run the download task based on the provided parameters.

    Parameters
    ----------
    nas_root : Union[str, Path]
        The root of the nas, before 'Orbbec_Fempto_Bolt', e.g. /media/charisoudis/nas_captures/Captures_Apr_May_2025.
    orbbec_id : str
        Orbbec recording ID, e.g. "l12_d6_s2x2_c12_synced_225_@01_05_2025_16_38_02".
    cam_name : str
        Full camera name as appearing on the NAS, e.g. "001 (CL8K14100HK)".
    capturestudio_cache_root : Union[str, Path]
        Destination root, up until and incl. the session name , e.g. /mnt/fdata/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1 or /root/CAPTURESTUDIO_CACHE/Thanos_2_Perf_1.
    nas_cache_root : Union[str, Path], optional
        The root of the nas cache, before 'Orbbec_Fempto_Bolt', e.g. /mnt/sdata/NAS_CACHE/Captures_Apr_May_2025 or /root/NAS_CACHE/Captures_Apr_May_2025.
        If not provided, it will be set to the same value as `capturestudio_cache_root`.
    """
    nas_format = _detect_nas_format(nas_root, orbbec_id)
    if nas_format == 'raw':
        # NAS (raw) --> CAPTURESTUDIO_CACHE (raw)
        _from_nas_raw_to_capturestudio_cache_raw(nas_root, orbbec_id, cam_name, capturestudio_cache_root)
    else:
        # NAS (h5) --> NAS_CACHE (h5) --> CAPTURESTUDIO_CACHE (raw)
        _from_nas_h5_to_nas_cache_h5(nas_root, orbbec_id, cam_name, nas_cache_root)
        _from_nas_cache_h5_to_capturestudio_cache_raw(nas_cache_root, orbbec_id, cam_name, capturestudio_cache_root)
    return None
