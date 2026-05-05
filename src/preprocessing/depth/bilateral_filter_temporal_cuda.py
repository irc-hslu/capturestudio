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
bilateral_filter_temporal_cuda = load(
    name="bilateral_filter_temporal_cuda",
    sources=[
        str(PathUtils.torch_extension_path('bilateral_filter') / "bilateral_filter_temporal_cuda.cpp"),
        str(PathUtils.torch_extension_path('bilateral_filter') / "bilateral_filter_temporal_cuda_kernel.cu"),
    ],
    extra_cflags=['-O2', '-std=c++17'],
    extra_cuda_cflags=['-O2'],
    extra_include_paths=extra_include_paths
)


# Wrapper function to call the CUDA kernel
def bilateral_filter_cuda_torch(depth_map, prev_color_image, color_image, prev_mask, mask, prev_depth_map, optical_flow, filter_radius=3, depth_sigma=5.0, rgb_sigma=0.1, temporal_sigma=5.0, temporal_lambda=0.3):
    # depth_map: [B, 1, H, W]
    # prev_depth_map: [B, 1, H, W]
    # prev_color_image: [B, 3, H, W]
    # color_image: [B, 3, H, W]
    # prev_mask: [B, 1, H, W]
    # mask: [B, 1, H, W]
    # optical_flow: [B, 2, H, W]

    B, _, H, W = depth_map.size()
    device = depth_map.device

    # Ensure input tensors are of the same size
    assert depth_map.size() == (B, 1, H, W), "Current depth map must be of shape [B, 1, H, W]"
    assert prev_depth_map.size() == (B, 1, H, W), "Previous depth map must be of shape [B, 1, H, W]"
    assert color_image.size() == (B, 3, H, W), "Color image must be of shape [B, 3, H, W], got {}".format(color_image.size())
    assert mask.size() == (B, 1, H, W), "Mask must be of shape [B, 1, H, W]"
    if isinstance(optical_flow, tuple):
        assert len(optical_flow) == 2, "Optical flow must be a tuple of (flow_fwd, flow_bwd)"
        assert optical_flow[0].size() == (B, 2, H, W), "Forward optical flow must be of shape [B, 2, H, W]"
        if optical_flow[1] is not None:
            assert optical_flow[1].size() == (B, 2, H, W), "Backward optical flow must be of shape [B, 2, H, W]"
    else:
        assert optical_flow.size() == (B, 2, H, W), "Optical flow must be of shape [B, 2, H, W]"

    # Ensure data is contiguous and on the correct device
    depth_map = depth_map.view(B, H, W).contiguous().to(device)  # [B, H, W]
    prev_depth_map = prev_depth_map.view(B, H, W).contiguous().to(device)  # [B, H, W]
    color_image = color_image.view(B, 3, H, W).permute(0, 2, 3, 1).contiguous().to(device)  # [B, H, W, 3]
    mask = mask.view(B, H, W).contiguous().to(device)  # [B, H, W]
    if isinstance(optical_flow, tuple):
        flow_fwd, flow_bwd = optical_flow
    else:
        flow_fwd, flow_bwd = None, optical_flow  # need backward flow (i.e. from t-1 --> t)
    flow_bwd = flow_bwd.view(B, 2, H, W).permute(0, 2, 3, 1).contiguous().to(device)  # [B, H, W, 2]
    # if flow_fwd is not None:
    #     flow_fwd = flow_fwd.view(B, 2, H, W).permute(0, 2, 3, 1).contiguous().to(device)

    # TODO: Errode depth map so that it is shrinked by 1-3 pixels according to segmentation mask's value (we want the depth map to be inside the mask)
    # TODO: Decide on the errosion window size

    # Create Gaussian depth kernel
    window_size = 2 * filter_radius + 1
    y = torch.arange(-filter_radius, filter_radius + 1, device=device, dtype=torch.float32)
    x = torch.arange(-filter_radius, filter_radius + 1, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(y, x, indexing='ij')
    depth_filter_kernel = torch.exp(-0.5 * (X ** 2 + Y ** 2) / (depth_sigma ** 2)).reshape(window_size * window_size).contiguous().to(device)
    # Call the CUDA extension function
    filt_depth = bilateral_filter_temporal_cuda.bilateral_filter_temporal(
        prev_depth_map,
        depth_map,
        prev_color_image,
        color_image,
        prev_mask,
        mask,
        flow_bwd,
        depth_filter_kernel,
        filter_radius,
        rgb_sigma,
        temporal_sigma,
        temporal_lambda
    )
    return filt_depth.view(B, 1, H, W)


if __name__ == '__main__':
    # Dummy data for testing
    B = 4  # Batch size
    C = 3
    H = 256
    W = 256
    device = 'cuda'

    # Random current and previous depth maps, color images, masks, and optical flow
    depth_maps = torch.rand(B, 1, H, W, device=device, dtype=torch.float32)
    prev_depth_maps = torch.rand(B, 1, H, W, device=device, dtype=torch.float32)
    color_images = torch.rand(B, 3, H, W, device=device, dtype=torch.float32)
    masks = torch.rand(B, 1, H, W, device=device, dtype=torch.float32)

    # Random optical flow (from prev to current)
    optical_flow_ = torch.rand(B, 2, H, W, device=device, dtype=torch.float32)  # BWD

    # Run the CUDA-based bilateral temporal filter
    filtered_depths = bilateral_filter_cuda_torch(
        depth_maps,
        color_images,
        color_images,
        masks,
        masks,
        prev_depth_maps,
        optical_flow_,
        filter_radius=3,
        depth_sigma=5.0,
        rgb_sigma=0.1,
        temporal_sigma=5.0
    )

    print(filtered_depths.shape)  # Should be [B, 1, H, W]
