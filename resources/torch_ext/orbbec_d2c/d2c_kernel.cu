#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <climits>

namespace {

constexpr float EPS = 1.0e-6f;
constexpr int Z_SENTINEL = INT_MAX;

__device__ __forceinline__ bool finite2(float x, float y) {
    return isfinite(x) && isfinite(y);
}

__device__ __forceinline__ bool apply_distortion(
    float& x,
    float& y,
    int model,
    const float* d,
    float k6_r2_limit) {

    if (model == 0) {
        return true;
    }

    const float k1 = d[0];
    const float k2 = d[1];
    const float k3 = d[2];
    const float k4 = d[3];
    const float k5 = d[4];
    const float k6 = d[5];
    const float p1 = d[6];
    const float p2 = d[7];

    const float r2 = x * x + y * y;

    if (model == 1 || model == 2) {
        if (model == 2 && k6_r2_limit > 0.0f && r2 >= k6_r2_limit) {
            return false;
        }

        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        float radial;

        if (model == 1) {
            radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
        } else {
            const float numerator = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
            const float denominator = 1.0f + k4 * r2 + k5 * r4 + k6 * r6;
            if (fabsf(denominator) <= EPS) {
                return false;
            }
            radial = numerator / denominator;
        }

        const float two_xy = 2.0f * x * y;
        const float x_tangential = p2 * (r2 + 2.0f * x * x) + p1 * two_xy;
        const float y_tangential = p1 * (r2 + 2.0f * y * y) + p2 * two_xy;
        x = x * radial + x_tangential;
        y = y * radial + y_tangential;
        return finite2(x, y);
    }

    if (model == 3) {
        const float r = sqrtf(r2);
        if (r <= EPS) {
            return true;
        }
        const float theta = atanf(r);
        const float theta2 = theta * theta;
        const float theta4 = theta2 * theta2;
        const float theta6 = theta4 * theta2;
        const float theta8 = theta4 * theta4;
        const float theta_d = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        const float scale = theta_d / r;
        x *= scale;
        y *= scale;
        return finite2(x, y);
    }

    return false;
}

__device__ __forceinline__ bool project_coeff(
    float depth_value,
    const float* coeff,
    const float* trans,
    float fx,
    float fy,
    float cx,
    float cy,
    int distortion_model,
    bool add_target_distortion,
    const float* dist_coeffs,
    float k6_r2_limit,
    float* out_x,
    float* out_y,
    float* out_z) {

    const float xc = depth_value * coeff[0] + trans[0];
    const float yc = depth_value * coeff[1] + trans[1];
    const float zc = depth_value * coeff[2] + trans[2];

    if (!(zc > EPS) || !isfinite(zc)) {
        return false;
    }

    float tx = xc / zc;
    float ty = yc / zc;

    if (!finite2(tx, ty)) {
        return false;
    }

    if (add_target_distortion) {
        if (!apply_distortion(tx, ty, distortion_model, dist_coeffs, k6_r2_limit)) {
            return false;
        }
    }

    const float px = tx * fx + cx;
    const float py = ty * fy + cy;
    if (!finite2(px, py)) {
        return false;
    }

    *out_x = px;
    *out_y = py;
    *out_z = zc;
    return true;
}

__device__ __forceinline__ bool point_in_tri(
    float px,
    float py,
    float ax,
    float ay,
    float bx,
    float by,
    float cx,
    float cy) {
    const float denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (fabsf(denom) < 1.0e-12f) {
        return false;
    }
    const float a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom;
    const float b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom;
    const float c = 1.0f - a - b;
    const float eps = -1.0e-5f;
    return a >= eps && b >= eps && c >= eps;
}

__device__ __forceinline__ bool point_in_quad(
    float px,
    float py,
    float ax,
    float ay,
    float bx,
    float by,
    float cx,
    float cy,
    float dx,
    float dy) {
    return point_in_tri(px, py, ax, ay, bx, by, cx, cy) ||
           point_in_tri(px, py, ax, ay, cx, cy, dx, dy);
}

__device__ __forceinline__ bool conservative_pixel_hits_quad(
    int ix,
    int iy,
    float ax,
    float ay,
    float bx,
    float by,
    float cx,
    float cy,
    float dx,
    float dy,
    bool conservative) {

    const float x = static_cast<float>(ix);
    const float y = static_cast<float>(iy);

    if (point_in_quad(x, y, ax, ay, bx, by, cx, cy, dx, dy)) {
        return true;
    }

    if (!conservative) {
        return false;
    }

    // Conservative rasterization approximation: also test target pixel corners.
    if (point_in_quad(x - 0.5f, y - 0.5f, ax, ay, bx, by, cx, cy, dx, dy)) return true;
    if (point_in_quad(x + 0.5f, y - 0.5f, ax, ay, bx, by, cx, cy, dx, dy)) return true;
    if (point_in_quad(x - 0.5f, y + 0.5f, ax, ay, bx, by, cx, cy, dx, dy)) return true;
    if (point_in_quad(x + 0.5f, y + 0.5f, ax, ay, bx, by, cx, cy, dx, dy)) return true;

    // Also handle the case where a quad vertex lies inside the target pixel square.
    const float xmin = x - 0.5f;
    const float xmax = x + 0.5f;
    const float ymin = y - 0.5f;
    const float ymax = y + 0.5f;
    if (ax >= xmin && ax <= xmax && ay >= ymin && ay <= ymax) return true;
    if (bx >= xmin && bx <= xmax && by >= ymin && by <= ymax) return true;
    if (cx >= xmin && cx <= xmax && cy >= ymin && cy <= ymax) return true;
    if (dx >= xmin && dx <= xmax && dy >= ymin && dy <= ymax) return true;

    return false;
}

__global__ void d2c_quad_kernel(
    const uint16_t* __restrict__ depth,
    const float* __restrict__ coeffs,
    const float* __restrict__ trans,
    int* __restrict__ zbuf,
    int batch,
    int n_depth,
    int rgb_h,
    int rgb_w,
    float fx,
    float fy,
    float cx,
    float cy,
    int distortion_model,
    bool add_target_distortion,
    const float* __restrict__ dist_coeffs,
    float k6_r2_limit,
    int max_depth_value,
    int max_footprint_px,
    bool conservative_raster,
    bool center_fallback) {

    const int64_t global = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(n_depth);
    if (global >= total) {
        return;
    }

    const int b = static_cast<int>(global / n_depth);
    const int p = static_cast<int>(global - static_cast<int64_t>(b) * n_depth);
    const uint16_t d_raw = depth[global];
    if (d_raw == 0) {
        return;
    }
    const float d = static_cast<float>(d_raw);

    const float* center_coeff = coeffs + (4 * n_depth + p) * 3;
    float center_x = 0.0f;
    float center_y = 0.0f;
    float center_z = 0.0f;
    if (!project_coeff(
            d,
            center_coeff,
            trans,
            fx,
            fy,
            cx,
            cy,
            distortion_model,
            add_target_distortion,
            dist_coeffs,
            k6_r2_limit,
            &center_x,
            &center_y,
            &center_z)) {
        return;
    }

    int z = __float2int_rn(center_z);
    if (z < 1) z = 1;
    if (z > max_depth_value) z = max_depth_value;

    float qx[4];
    float qy[4];
    float qz_unused = 0.0f;
    #pragma unroll
    for (int cidx = 0; cidx < 4; ++cidx) {
        const float* coeff = coeffs + (cidx * n_depth + p) * 3;
        if (!project_coeff(
                d,
                coeff,
                trans,
                fx,
                fy,
                cx,
                cy,
                distortion_model,
                add_target_distortion,
                dist_coeffs,
                k6_r2_limit,
                &qx[cidx],
                &qy[cidx],
                &qz_unused)) {
            return;
        }
    }

    float min_x_f = fminf(fminf(qx[0], qx[1]), fminf(qx[2], qx[3]));
    float max_x_f = fmaxf(fmaxf(qx[0], qx[1]), fmaxf(qx[2], qx[3]));
    float min_y_f = fminf(fminf(qy[0], qy[1]), fminf(qy[2], qy[3]));
    float max_y_f = fmaxf(fmaxf(qy[0], qy[1]), fmaxf(qy[2], qy[3]));

    int min_x = static_cast<int>(floorf(min_x_f));
    int max_x = static_cast<int>(ceilf(max_x_f));
    int min_y = static_cast<int>(floorf(min_y_f));
    int max_y = static_cast<int>(ceilf(max_y_f));
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > rgb_w - 1) max_x = rgb_w - 1;
    if (max_y > rgb_h - 1) max_y = rgb_h - 1;

    bool wrote = false;
    int* zbase = zbuf + static_cast<int64_t>(b) * rgb_h * rgb_w;

    if (min_x <= max_x && min_y <= max_y &&
        (max_x - min_x + 1) <= max_footprint_px &&
        (max_y - min_y + 1) <= max_footprint_px) {

        for (int yy = min_y; yy <= max_y; ++yy) {
            for (int xx = min_x; xx <= max_x; ++xx) {
                if (conservative_pixel_hits_quad(
                        xx,
                        yy,
                        qx[0],
                        qy[0],
                        qx[1],
                        qy[1],
                        qx[2],
                        qy[2],
                        qx[3],
                        qy[3],
                        conservative_raster)) {
                    atomicMin(zbase + yy * rgb_w + xx, z);
                    wrote = true;
                }
            }
        }
    }

    // Fallback for tiny or numerically awkward projected footprints. This is a small
    // safety net for isolated black speckles; the z-buffer prevents it from punching
    // through closer surfaces.
    if (center_fallback && !wrote) {
        const int u = __float2int_rn(center_x);
        const int v = __float2int_rn(center_y);
        if (u >= 0 && u < rgb_w && v >= 0 && v < rgb_h) {
            atomicMin(zbase + v * rgb_w + u, z);
        }
    }
}

__global__ void zbuf_to_u16_kernel(
    const int* __restrict__ zbuf,
    uint16_t* __restrict__ out,
    int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int z = zbuf[idx];
    out[idx] = (z == Z_SENTINEL) ? static_cast<uint16_t>(0) : static_cast<uint16_t>(z);
}

__device__ __forceinline__ void sort_u16_small(uint16_t* vals, int count) {
    for (int i = 1; i < count; ++i) {
        uint16_t key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) {
            vals[j + 1] = vals[j];
            --j;
        }
        vals[j + 1] = key;
    }
}

__global__ void hole_fill_kernel(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int batch,
    int h,
    int w,
    int radius,
    int max_depth_delta,
    int min_valid_neighbors) {

    const int64_t global = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t pixels_per_frame = static_cast<int64_t>(h) * static_cast<int64_t>(w);
    const int64_t total = static_cast<int64_t>(batch) * pixels_per_frame;
    if (global >= total) return;

    const uint16_t current = src[global];
    if (current != 0) {
        dst[global] = current;
        return;
    }

    if (radius < 1) radius = 1;
    if (radius > 3) radius = 3;
    if (min_valid_neighbors < 1) min_valid_neighbors = 1;

    const int b = static_cast<int>(global / pixels_per_frame);
    const int local = static_cast<int>(global - static_cast<int64_t>(b) * pixels_per_frame);
    const int y = local / w;
    const int x = local - y * w;
    const uint16_t* base = src + static_cast<int64_t>(b) * pixels_per_frame;

    uint16_t vals[49];
    int count = 0;
    uint16_t minv = 65535;
    uint16_t maxv = 0;

    for (int yy = y - radius; yy <= y + radius; ++yy) {
        if (yy < 0 || yy >= h) continue;
        for (int xx = x - radius; xx <= x + radius; ++xx) {
            if (xx < 0 || xx >= w) continue;
            if (xx == x && yy == y) continue;
            const uint16_t v = base[yy * w + xx];
            if (v == 0) continue;
            if (count < 49) {
                vals[count++] = v;
                if (v < minv) minv = v;
                if (v > maxv) maxv = v;
            }
        }
    }

    if (count < min_valid_neighbors) {
        dst[global] = 0;
        return;
    }

    if (static_cast<int>(maxv) - static_cast<int>(minv) > max_depth_delta) {
        dst[global] = 0;
        return;
    }

    sort_u16_small(vals, count);
    dst[global] = vals[count / 2];
}

}  // namespace

void d2c_launch_cuda(
    const torch::Tensor& depth,
    const torch::Tensor& coeffs,
    const torch::Tensor& trans,
    torch::Tensor& out,
    int64_t rgb_h64,
    int64_t rgb_w64,
    float fx,
    float fy,
    float cx,
    float cy,
    int64_t distortion_model64,
    bool add_target_distortion,
    const torch::Tensor& dist_coeffs,
    float k6_r2_limit,
    int64_t max_depth_value64,
    int64_t max_footprint_px64,
    bool conservative_raster,
    bool center_fallback,
    bool fill_holes,
    int64_t hole_radius64,
    int64_t hole_max_depth_delta64,
    int64_t hole_min_valid_neighbors64,
    int64_t hole_fill_iterations64) {

    const int rgb_h = static_cast<int>(rgb_h64);
    const int rgb_w = static_cast<int>(rgb_w64);
    const int depth_h = static_cast<int>(depth.size(depth.dim() - 2));
    const int depth_w = static_cast<int>(depth.size(depth.dim() - 1));
    const int n_depth = depth_h * depth_w;
    const int batch = depth.dim() == 3 ? static_cast<int>(depth.size(0)) : 1;
    const int distortion_model = static_cast<int>(distortion_model64);
    const int max_depth_value = static_cast<int>(max_depth_value64);
    const int max_footprint_px = static_cast<int>(max_footprint_px64);
    const int hole_radius = static_cast<int>(hole_radius64);
    const int hole_max_depth_delta = static_cast<int>(hole_max_depth_delta64);
    const int hole_min_valid_neighbors = static_cast<int>(hole_min_valid_neighbors64);
    const int hole_fill_iterations = static_cast<int>(hole_fill_iterations64);

    auto zbuf = torch::empty({batch, rgb_h, rgb_w}, depth.options().dtype(torch::kInt32));
    zbuf.fill_(Z_SENTINEL);

    const int threads = 256;
    const int64_t total_depth_threads = static_cast<int64_t>(batch) * static_cast<int64_t>(n_depth);
    const dim3 blocks_depth(static_cast<unsigned int>((total_depth_threads + threads - 1) / threads));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    d2c_quad_kernel<<<blocks_depth, threads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(depth.data_ptr<uint16_t>()),
        coeffs.data_ptr<float>(),
        trans.data_ptr<float>(),
        zbuf.data_ptr<int>(),
        batch,
        n_depth,
        rgb_h,
        rgb_w,
        fx,
        fy,
        cx,
        cy,
        distortion_model,
        add_target_distortion,
        dist_coeffs.data_ptr<float>(),
        k6_r2_limit,
        max_depth_value,
        max_footprint_px,
        conservative_raster,
        center_fallback);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t total_out = static_cast<int64_t>(batch) * static_cast<int64_t>(rgb_h) * static_cast<int64_t>(rgb_w);
    const dim3 blocks_out(static_cast<unsigned int>((total_out + threads - 1) / threads));
    zbuf_to_u16_kernel<<<blocks_out, threads, 0, stream>>>(
        zbuf.data_ptr<int>(),
        reinterpret_cast<uint16_t*>(out.data_ptr<uint16_t>()),
        total_out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (fill_holes && hole_fill_iterations > 0) {
        torch::Tensor current = out;
        torch::Tensor tmp = torch::empty_like(out);
        for (int i = 0; i < hole_fill_iterations; ++i) {
            hole_fill_kernel<<<blocks_out, threads, 0, stream>>>(
                reinterpret_cast<const uint16_t*>(current.data_ptr<uint16_t>()),
                reinterpret_cast<uint16_t*>(tmp.data_ptr<uint16_t>()),
                batch,
                rgb_h,
                rgb_w,
                hole_radius,
                hole_max_depth_delta,
                hole_min_valid_neighbors);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            if (i + 1 < hole_fill_iterations) {
                current = tmp;
                tmp = torch::empty_like(out);
            } else {
                out.copy_(tmp);
            }
        }
    }
}
