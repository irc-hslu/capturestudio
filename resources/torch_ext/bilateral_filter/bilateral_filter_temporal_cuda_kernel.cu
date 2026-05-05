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


__global__ void bilateral_filter_temporal_kernel(
    float* filt_depth,
    const float* prev_depth,
    const float* depth,
    const float* prev_rgb, // sim: new
    const float* rgb,
    const float* prev_mask, // sim: new
    const float* mask,
    const float* optical_flow,
    const int batch_size,
    const int height,
    const int width,
    const float* depth_filter_kernel,
    const int filter_radius,
    const float rgb_sigma,
    const float temporal_sigma,
    const float temporal_lambda // sim: new
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
    int flow_offset = b * height * width * 2;

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

    // Spatial Filtering (Current Frame)
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

    // Temporal Filtering (Previous Frame)
    // Retrieve optical flow vectors
    float flow_x = optical_flow[flow_offset + (y * width + x) * 2 + 0];
    float flow_y = optical_flow[flow_offset + (y * width + x) * 2 + 1];

    // Compute warped coordinates
    float x_prev = x - flow_x;
    float y_prev = y - flow_y;

    // Bilinear interpolation weights
    int x0 = floorf(x_prev);
    int y0 = floorf(y_prev);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx = x_prev - x0;
    float wy = y_prev - y0;

    // Check bounds and interpolate mask value from previous frame
    if (x0 >= 0 && x1 < width && y0 >= 0 && y1 < height) {
        float prev_mask_values[4];
        prev_mask_values[0] = prev_mask[offset + y0 * width + x0];  // x0, y0
        prev_mask_values[1] = prev_mask[offset + y0 * width + x1];  // x1, y0
        prev_mask_values[2] = prev_mask[offset + y1 * width + x0];  // x0, y1
        prev_mask_values[3] = prev_mask[offset + y1 * width + x1];  // x1, y1

        float prev_depth_values[4];
        prev_depth_values[0] = prev_depth[offset + y0 * width + x0];  // x0, y0
        prev_depth_values[1] = prev_depth[offset + y0 * width + x1];  // x1, y0
        prev_depth_values[2] = prev_depth[offset + y1 * width + x0];  // x0, y1
        prev_depth_values[3] = prev_depth[offset + y1 * width + x1];  // x1, y1

        bool prev_mask_check = prev_mask_values[0] > 0.0 && prev_mask_values[1] > 0.0 && prev_mask_values[2] > 0.0 && prev_mask_values[3] > 0.0;
        bool prev_depth_check = prev_depth_values[0] > 0.0 && prev_depth_values[1] > 0.0 && prev_depth_values[2] > 0.0 && prev_depth_values[3] > 0.0;

        // Check if all pixels are in previous mask and whether the previous depth values are valid
        if (prev_mask_check && prev_depth_check) {
            float prev_rgb_0[3] = { // x0, y0
                prev_rgb[rgb_offset + (y0 * width + x0) * 3 + 0],
                prev_rgb[rgb_offset + (y0 * width + x0) * 3 + 1],
                prev_rgb[rgb_offset + (y0 * width + x0) * 3 + 2]
            };

            float prev_rgb_1[3] = { // x1, y0
                prev_rgb[rgb_offset + (y0 * width + x1) * 3 + 0],
                prev_rgb[rgb_offset + (y0 * width + x1) * 3 + 1],
                prev_rgb[rgb_offset + (y0 * width + x1) * 3 + 2]
            };

            float prev_rgb_2[3] = { // x0, y1
                prev_rgb[rgb_offset + (y1 * width + x0) * 3 + 0],
                prev_rgb[rgb_offset + (y1 * width + x0) * 3 + 1],
                prev_rgb[rgb_offset + (y1 * width + x0) * 3 + 2]
            };

            float prev_rgb_3[3] = { // x1, y1
                prev_rgb[rgb_offset + (y1 * width + x1) * 3 + 0],
                prev_rgb[rgb_offset + (y1 * width + x1) * 3 + 1],
                prev_rgb[rgb_offset + (y1 * width + x1) * 3 + 2]
            };

            // Bilinear interpolation
            float prev_rgb_interp[3] = {
                (1 - wx) * (1 - wy) * prev_rgb_0[0] + wx * (1 - wy) * prev_rgb_1[0] + (1 - wx) * wy * prev_rgb_2[0] + wx * wy * prev_rgb_3[0],
                (1 - wx) * (1 - wy) * prev_rgb_0[1] + wx * (1 - wy) * prev_rgb_1[1] + (1 - wx) * wy * prev_rgb_2[1] + wx * wy * prev_rgb_3[1],
                (1 - wx) * (1 - wy) * prev_rgb_0[2] + wx * (1 - wy) * prev_rgb_1[2] + (1 - wx) * wy * prev_rgb_2[2] + wx * wy * prev_rgb_3[2]
            };

            float prev_depth_interp =
                (1 - wx) * (1 - wy) * prev_depth_values[0] +
                wx * (1 - wy) * prev_depth_values[1] +
                (1 - wx) * wy * prev_depth_values[2] +
                wx * wy * prev_depth_values[3];

            // Temporal weight based on temporal_sigma
            float temporal_weight = get_gauss(rgb_dist(cur_rgb, prev_rgb_interp), temporal_sigma);

            dsum += prev_depth_interp * temporal_weight * temporal_lambda;
            fsum += temporal_weight * temporal_lambda;
        }
    }

    filt_depth[offset + y * width + x] = (fsum != 0.0f) ? dsum / fsum : 0.0f;
}


// Launcher function
void bilateral_filter_temporal_kernel_launcher(
    float* filt_depth,
    const float* prev_depth, // sim: new
    const float* depth,
    const float* prev_rgb, // sim: new
    const float* rgb,
    const float* prev_mask, // sim: new
    const float* mask,
    const float* optical_flow,
    int batch_size,
    int height,
    int width,
    const float* depth_filter_kernel,
    int filter_radius,
    float rgb_sigma,
    float temporal_sigma,
    float temporal_lambda   // sim: new
) {
    int total_pixels = batch_size * height * width;
    int threads_per_block = 256;
    int num_blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    bilateral_filter_temporal_kernel<<<num_blocks, threads_per_block>>>(
        filt_depth,
        prev_depth,
        depth,
        prev_rgb,   // sim: new
        rgb,
        prev_mask,  // sim: new
        mask,
        optical_flow,
        batch_size,
        height,
        width,
        depth_filter_kernel,
        filter_radius,
        rgb_sigma,
        temporal_sigma,
        temporal_lambda // sim: new
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in bilateral_filter_temporal_kernel: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error in cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
    }
}