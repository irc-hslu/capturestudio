#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

__device__ float rgb_dist(const float* rgb1, const float* rgb2) {
    return sqrtf(
        powf(rgb1[0] - rgb2[0], 2) +
        powf(rgb1[1] - rgb2[1], 2) +
        powf(rgb1[2] - rgb2[2], 2)
    );
}

__device__ float get_gauss(const float x, const float sigma) {
    return expf(-0.5f * (x * x) / (sigma * sigma));
}

__global__ void bilateral_filter_kernel(
    float* filt_depth,
    const float* depth,
    const float* rgb,
    const float* mask,
    const int batch_size,
    const int height,
    const int width,
    const float* depth_filter_kernel,
    const int filter_radius,
    const float rgb_sigma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = batch_size * height * width;

    if (idx >= total_pixels) return;

    int b = idx / (height * width);
    int idx_in_batch = idx % (height * width);
    int y = idx_in_batch / width;
    int x = idx_in_batch % width;

    // Offsets for batch
    int offset = b * height * width;
    int rgb_offset = b * height * width * 3;

    float mask_value = mask[offset + y * width + x];
    if (mask_value == 0.0f) {
        filt_depth[offset + y * width + x] = 0.0f;
        return;
    }

    float dsum = 0.0f;
    float fsum = 0.0f;

    float cur_rgb[3] = {
        rgb[rgb_offset + (y * width + x) * 3 + 0],
        rgb[rgb_offset + (y * width + x) * 3 + 1],
        rgb[rgb_offset + (y * width + x) * 3 + 2]
    };

    int window_size = 2 * filter_radius + 1;

    for (int dy = -filter_radius; dy <= filter_radius; dy++) {
        for (int dx = -filter_radius; dx <= filter_radius; dx++) {
            int x2 = x + dx;
            int y2 = y + dy;

            if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height) continue;

            float neighbor_mask = mask[offset + y2 * width + x2];
            if (neighbor_mask == 0.0f) continue;

            float cur_depth = depth[offset + y2 * width + x2];
            if (cur_depth == 0.0f) continue;

            float cur_rgb2[3] = {
                rgb[rgb_offset + (y2 * width + x2) * 3 + 0],
                rgb[rgb_offset + (y2 * width + x2) * 3 + 1],
                rgb[rgb_offset + (y2 * width + x2) * 3 + 2]
            };

            int kernel_index = (dy + filter_radius) * window_size + (dx + filter_radius);
            float depth_weight = depth_filter_kernel[kernel_index];
            float rgb_weight = (rgb_sigma != 0.0f) ? get_gauss(rgb_dist(cur_rgb, cur_rgb2), rgb_sigma) : 1.0f;

            float combined_weight = depth_weight * rgb_weight * neighbor_mask;

            dsum += cur_depth * combined_weight;
            fsum += combined_weight;
        }
    }

    filt_depth[offset + y * width + x] = (fsum != 0.0f) ? dsum / fsum : 0.0f;
}

// Launcher function
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
) {
    int total_pixels = batch_size * height * width;
    int threads_per_block = 256;
    int num_blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    bilateral_filter_kernel<<<num_blocks, threads_per_block>>>(
        filt_depth,
        depth,
        rgb,
        mask,
        batch_size,
        height,
        width,
        depth_filter_kernel,
        filter_radius,
        rgb_sigma
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in bilateral_filter_kernel: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
    }
}
