#include <torch/extension.h>

#include <cstdint>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_UINT16(x) TORCH_CHECK((x).scalar_type() == torch::kUInt16, #x " must be uint16")

void d2c_launch_cuda(
    const torch::Tensor& depth,
    const torch::Tensor& coeffs,
    const torch::Tensor& trans,
    torch::Tensor& out,
    int64_t rgb_h,
    int64_t rgb_w,
    float fx,
    float fy,
    float cx,
    float cy,
    int64_t distortion_model,
    bool add_target_distortion,
    const torch::Tensor& dist_coeffs,
    float k6_r2_limit,
    int64_t max_depth_value,
    int64_t max_footprint_px,
    bool conservative_raster,
    bool center_fallback,
    bool fill_holes,
    int64_t hole_radius,
    int64_t hole_max_depth_delta,
    int64_t hole_min_valid_neighbors,
    int64_t hole_fill_iterations);

torch::Tensor d2c_forward_cuda(
    torch::Tensor depth,
    torch::Tensor coeffs,
    torch::Tensor trans,
    int64_t rgb_h,
    int64_t rgb_w,
    double fx,
    double fy,
    double cx,
    double cy,
    int64_t distortion_model,
    bool add_target_distortion,
    torch::Tensor dist_coeffs,
    double k6_r2_limit,
    int64_t max_depth_value,
    int64_t max_footprint_px,
    bool conservative_raster,
    bool center_fallback,
    bool fill_holes,
    int64_t hole_radius,
    int64_t hole_max_depth_delta,
    int64_t hole_min_valid_neighbors,
    int64_t hole_fill_iterations) {

    CHECK_CUDA(depth);
    CHECK_CUDA(coeffs);
    CHECK_CUDA(trans);
    CHECK_CUDA(dist_coeffs);
    CHECK_CONTIGUOUS(depth);
    CHECK_CONTIGUOUS(coeffs);
    CHECK_CONTIGUOUS(trans);
    CHECK_CONTIGUOUS(dist_coeffs);
    CHECK_UINT16(depth);
    CHECK_FLOAT32(coeffs);
    CHECK_FLOAT32(trans);
    CHECK_FLOAT32(dist_coeffs);

    TORCH_CHECK(depth.dim() == 2 || depth.dim() == 3, "depth must have shape (H,W) or (B,H,W)");
    TORCH_CHECK(coeffs.dim() == 3 && coeffs.size(0) == 5 && coeffs.size(2) == 3,
                "coeffs must have shape (5, H*W, 3)");
    TORCH_CHECK(trans.numel() == 3, "trans must have 3 elements");
    TORCH_CHECK(dist_coeffs.numel() == 8, "dist_coeffs must have 8 elements");
    TORCH_CHECK(rgb_h > 0 && rgb_w > 0, "rgb_h and rgb_w must be positive");
    TORCH_CHECK(max_depth_value >= 1 && max_depth_value <= 65535, "max_depth_value must be in [1,65535]");
    TORCH_CHECK(max_footprint_px >= 1, "max_footprint_px must be positive");

    const bool batched = depth.dim() == 3;
    const int64_t depth_h = depth.size(depth.dim() - 2);
    const int64_t depth_w = depth.size(depth.dim() - 1);
    const int64_t n = depth_h * depth_w;
    TORCH_CHECK(coeffs.size(1) == n, "coeffs second dimension must be H*W");

    std::vector<int64_t> out_shape;
    if (batched) {
        out_shape = {depth.size(0), rgb_h, rgb_w};
    } else {
        out_shape = {rgb_h, rgb_w};
    }

    auto out = torch::empty(out_shape, depth.options().dtype(torch::kUInt16));

    d2c_launch_cuda(
        depth,
        coeffs,
        trans,
        out,
        rgb_h,
        rgb_w,
        static_cast<float>(fx),
        static_cast<float>(fy),
        static_cast<float>(cx),
        static_cast<float>(cy),
        distortion_model,
        add_target_distortion,
        dist_coeffs,
        static_cast<float>(k6_r2_limit),
        max_depth_value,
        max_footprint_px,
        conservative_raster,
        center_fallback,
        fill_holes,
        hole_radius,
        hole_max_depth_delta,
        hole_min_valid_neighbors,
        hole_fill_iterations);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("d2c_forward_cuda", &d2c_forward_cuda, "Orbbec D2C forward rasterization (CUDA)");
}
