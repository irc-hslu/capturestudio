import os

import torch
from torch.utils.cpp_extension import load

from utils.misc import PathUtils, env_get

# Compile and load the CUDA extension
if env_get('CC', None) is not None:
    os.environ["CC"] = env_get('CC')
if env_get('CXX', None) is not None:
    os.environ["CXX"] = env_get('CXX')
if env_get('CUDA_INCLUDE_PATH', None) is not None:
    extra_include_paths = [env_get('CUDA_INCLUDE_PATH')]
else:
    extra_include_paths = []
# Compile and load the CUDA extension with the new filenames
bilateral_filter_spatial_cuda = load(
    name="bilateral_filter_spatial_cuda",
    sources=[
        str(PathUtils.torch_extension_path('bilateral_filter') / "bilateral_filter_spatial_cuda.cpp"),
        str(PathUtils.torch_extension_path('bilateral_filter') / "bilateral_filter_spatial_cuda_kernel.cu"),
    ],
    extra_cflags=['-O2', '-std=c++17'],
    extra_cuda_cflags=['-O2'],
    extra_include_paths=extra_include_paths
)


# Wrapper function to call the CUDA kernel
def bilateral_filter_cuda_torch(depth_map, color_image, mask, filter_radius=3, depth_sigma=5.0, rgb_sigma=0.1):
    # depth_map: [B, 1, H, W]
    # color_image: [B, 3, H, W]
    # mask: [B, 1, H, W]

    B, _, H, W = depth_map.size()
    device = depth_map.device

    # Ensure input tensors are of the same size
    assert depth_map.size() == (B, 1, H, W), "Depth map must be of shape [B, 1, H, W]"
    assert color_image.size() == (B, 3, H, W), "Color image must be of shape [B, 3, H, W]"
    assert mask.size() == (B, 1, H, W), "Mask must be of shape [B, 1, H, W]"

    # Ensure data is contiguous and on the correct device
    depth_map = depth_map.view(B, H, W).contiguous().to(device)  # Shape: [B, H, W]
    color_image = color_image.view(B, 3, H, W).permute(0, 2, 3, 1).contiguous().to(device)  # Shape: [B, H, W, 3]
    mask = mask.view(B, H, W).contiguous().to(device)  # Shape: [B, H, W]

    # TODO: Erode depth map so that it is shrunk by 1-3 pixels according to segmentation mask's value (we want the depth map to be inside the mask)
    # TODO: Decide on the erosion window size

    # Create Gaussian depth kernel
    window_size = 2 * filter_radius + 1
    y = torch.arange(-filter_radius, filter_radius + 1, device=device, dtype=torch.float32)
    x = torch.arange(-filter_radius, filter_radius + 1, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(y, x, indexing='ij')
    # Use window_size in reshaping the depth_filter_kernel
    depth_filter_kernel = torch.exp(-0.5 * (X ** 2 + Y ** 2) / (depth_sigma ** 2)).reshape(window_size * window_size).contiguous().to(device)

    # Call the CUDA extension function
    filt_depth = bilateral_filter_spatial_cuda.bilateral_filter(
        depth_map, color_image, mask, depth_filter_kernel, filter_radius, rgb_sigma
    )

    return filt_depth.view(B, 1, H, W)
