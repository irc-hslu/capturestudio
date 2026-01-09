#include <torch/extension.h>
#include <cuda_runtime.h>

// Declare the CUDA kernel launcher
void bilateral_filter_kernel_launcher(
    float* filt_depth,
    const float* depth,
    const float* rgb,
    const float* mask,
    int batch_size,
    int height,
    int width,
    const float* depth_filter_kernel,
    int filter_radius,
    float rgb_sigma
);

// PyTorch wrapper function that calls the CUDA kernel
torch::Tensor bilateral_filter(
    torch::Tensor depth,
    torch::Tensor rgb,
    torch::Tensor mask,
    torch::Tensor depth_filter_kernel,
    int filter_radius,
    float rgb_sigma
) {
    // Ensure tensors are contiguous
    depth = depth.contiguous();
    rgb = rgb.contiguous();
    mask = mask.contiguous();
    depth_filter_kernel = depth_filter_kernel.contiguous();

    // Prepare output tensor for filtered depth
    auto filt_depth = torch::zeros_like(depth);

    int batch_size = depth.size(0);
    int height = depth.size(1);
    int width = depth.size(2);

    // Get raw pointers to the data
    float* filt_depth_ptr = filt_depth.data_ptr<float>();
    const float* depth_ptr = depth.data_ptr<float>();
    const float* rgb_ptr = rgb.data_ptr<float>();
    const float* mask_ptr = mask.data_ptr<float>();

    const float* depth_filter_kernel_ptr = depth_filter_kernel.data_ptr<float>();

    // Launch the CUDA kernel via a launcher function
    bilateral_filter_kernel_launcher(
        filt_depth_ptr,
        depth_ptr,
        rgb_ptr,
        mask_ptr,
        batch_size,
        height,
        width,
        depth_filter_kernel_ptr,
        filter_radius,
        rgb_sigma
    );

    return filt_depth;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilateral_filter", &bilateral_filter, "Bilateral Filter (CUDA)");
}
