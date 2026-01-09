#include <torch/extension.h>
#include <cuda_runtime.h>

// Declare the CUDA kernel launcher
void bilateral_filter_temporal_kernel_launcher(
    float* filt_depth,
    const float* prev_depth,
    const float* depth,
    const float* prev_rgb,
    const float* rgb,
    const float* prev_mask,
    const float* mask,
    const float* optical_flow,
    int batch_size,
    int height,
    int width,
    const float* depth_filter_kernel,
    int filter_radius,
    float rgb_sigma,
    float temporal_sigma,
    float temporal_lambda
);

// PyTorch wrapper function that calls the CUDA kernel
torch::Tensor bilateral_filter_temporal(
    torch::Tensor prev_depth,
    torch::Tensor depth,
    torch::Tensor prev_rgb,
    torch::Tensor rgb,
    torch::Tensor prev_mask,
    torch::Tensor mask,
    torch::Tensor optical_flow,
    torch::Tensor depth_filter_kernel,
    int filter_radius,
    float rgb_sigma,
    float temporal_sigma,
    float temporal_lambda
) {
    // Ensure tensors are contiguous
    depth = depth.contiguous();
    prev_depth = prev_depth.contiguous();
    rgb = rgb.contiguous();
    prev_rgb = rgb.contiguous();
    mask = mask.contiguous();
    prev_mask = mask.contiguous();
    optical_flow = optical_flow.contiguous();
    depth_filter_kernel = depth_filter_kernel.contiguous();

    // Prepare output tensor for filtered depth
    auto filt_depth = torch::zeros_like(depth);

    int batch_size = depth.size(0);
    int height = depth.size(1);
    int width = depth.size(2);

    // Get raw pointers to the data
    float* filt_depth_ptr = filt_depth.data_ptr<float>();
    const float* prev_depth_ptr = prev_depth.data_ptr<float>();
    const float* depth_ptr = depth.data_ptr<float>();
    const float* prev_rgb_ptr = prev_rgb.data_ptr<float>();
    const float* rgb_ptr = rgb.data_ptr<float>();
    const float* prev_mask_ptr = prev_mask.data_ptr<float>();
    const float* mask_ptr = mask.data_ptr<float>();
    const float* optical_flow_ptr = optical_flow.data_ptr<float>();

    const float* depth_filter_kernel_ptr = depth_filter_kernel.data_ptr<float>();

    // Launch the CUDA kernel via a launcher function
    bilateral_filter_temporal_kernel_launcher(
        filt_depth_ptr,
        prev_depth_ptr,
        depth_ptr,
        prev_rgb_ptr,
        rgb_ptr,
        prev_mask_ptr,
        mask_ptr,
        optical_flow_ptr,
        batch_size,
        height,
        width,
        depth_filter_kernel_ptr,
        filter_radius,
        rgb_sigma,
        temporal_sigma,
        temporal_lambda
    );

    return filt_depth;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilateral_filter_temporal", &bilateral_filter_temporal, "Bilateral Filter Temporal (CUDA)");
}
