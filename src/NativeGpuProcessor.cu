#include "NativeGpuProcessor.cuh"
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

NativeGpuProcessor::NativeGpuProcessor() {
    for (int i = 0; i < NUM_PIPES; i++) {
        cudaError_t err = cudaStreamCreate(&pipes[i].stream);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: Failed to create stream " << i << ": " << cudaGetErrorString(err) << std::endl;
        }
    }
}

NativeGpuProcessor::~NativeGpuProcessor() {
    for (int i = 0; i < NUM_PIPES; i++) {
        auto& p = pipes[i];
        if (p.stream) cudaStreamSynchronize(p.stream);
        if (p.d_src) cudaFree(p.d_src);
        if (p.d_tex_full) cudaFree(p.d_tex_full);
        if (p.d_gauss_tmp) cudaFree(p.d_gauss_tmp);
        if (p.d_dst_img) cudaFree(p.d_dst_img);
        if (p.d_dst_tex) cudaFree(p.d_dst_tex);
        if (p.h_src_pinned) cudaFreeHost(p.h_src_pinned);
        if (p.h_img_pinned) cudaFreeHost(p.h_img_pinned);
        if (p.h_tex_pinned) cudaFreeHost(p.h_tex_pinned);
        if (p.stream) cudaStreamDestroy(p.stream);
    }
    if (d_xofs) cudaFree(d_xofs);
    if (d_yofs) cudaFree(d_yofs);
    if (d_xalpha) cudaFree(d_xalpha);
    if (d_yalpha) cudaFree(d_yalpha);
}

// ============================================================================
// Kernel 1: 全分辨率纹理特征提取 (dilate -> erode -> subtract 融合)
// ============================================================================
__global__ void texture_full_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ tex,
    int w, int h, int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uint8_t min_v = 255, max_v = 0;
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int sy = min(max(y + ky, 0), h - 1);
            int sx = min(max(x + kx, 0), w - 1);
            uint8_t val = src[sy * w + sx];
            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
        }
    }
    tex[y * w + x] = max_v - min_v;
}

// ============================================================================
// Kernel 2: GaussianBlur 行滤波 (CV_8U → CV_32F)
// 精确模拟 OpenCV SymmRowSmallVec_8u32f (ksize=3, sigma=0):
//   - 对称对 s[x-1]+s[x+1] 先做 16-bit 整数加法, 再转 float
//   - 系数固定 [0.25, 0.5, 0.25] (OpenCV small_gaussian_tab)
//   - 编译须 --fmad=false, 使 FMUL+FMUL+FADD 匹配 SSE2 MULPS+MULPS+ADDPS
// ============================================================================
__global__ void gauss_row_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba
    int xl = x > 0 ? x - 1 : 1;
    int xr = x < w - 1 ? x + 1 : w - 2;

    // 与 OpenCV SSE2 一致: 对称对先做 integer 加法, 再 cvt→float
    int sym = (int)src[y * w + xl] + (int)src[y * w + xr];
    float f_sym = (float)sym;
    float f_center = (float)src[y * w + x];

    // --fmad=false → FMUL(sym*0.25) + FMUL(center*0.5) + FADD
    dst[y * w + x] = f_sym * 0.25f + f_center * 0.5f;
}

// ============================================================================
// Kernel 3: GaussianBlur 列滤波 (CV_32F → CV_8U)
// 精确模拟 OpenCV SymmColumnSmallVec_32f8u:
//   - 对称对 float 加法, 乘系数, 最后 cvRound (round-to-nearest-even)
//   - __float2int_rn 与 SSE2 CVTPS2DQ 行为完全等价
// ============================================================================
__global__ void gauss_col_kernel(
    const float* __restrict__ src,
    uint8_t* __restrict__ dst,
    int w, int h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // BORDER_REFLECT_101
    int yt = y > 0 ? y - 1 : 1;
    int yb = y < h - 1 ? y + 1 : h - 2;

    float f_top = src[yt * w + x];
    float f_bot = src[yb * w + x];
    float f_center = src[y * w + x];

    // 与 SSE2 一致: ADDPS(top,bot) → MULPS(sym,k0) → MULPS(center,k1) → ADDPS
    float sym = f_top + f_bot;
    float result = sym * 0.25f + f_center * 0.5f;

    // 关键: OpenCV SSE2 列滤波器用 _mm_cvttps_epi32(result + 0.5f) (截断+预加0.5)
    // 这是 round-half-up, 不是 __float2int_rn 的 round-to-nearest-even
    int ival = (int)(result + 0.5f);
    dst[y * w + x] = (uint8_t)(ival < 0 ? 0 : (ival > 255 ? 255 : ival));
}

// ============================================================================
// Kernel 4: 双线性缩放 (CV_8U → CV_8U)
// 使用 CPU 预计算的系数表, 消除 CUDA vs x86 浮点差异
// 垂直插值: 匹配 OpenCV VResizeLinear<uchar,int,short> 特化版的分步截断公式
// ============================================================================
__global__ void resize_linear_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int sw, int sh, int dw, int dh,
    const int* __restrict__ xofs,
    const short* __restrict__ xalpha,
    const int* __restrict__ yofs,
    const short* __restrict__ yalpha,
    int xmin, int xmax
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dw || dy >= dh) return;

    int sy = yofs[dy];
    int sy1 = min(sy + 1, sh - 1);
    short b0 = yalpha[dy * 2];
    short b1 = yalpha[dy * 2 + 1];

    const uint8_t* S0 = src + sy * sw;
    const uint8_t* S1 = src + sy1 * sw;

    // 水平插值 (匹配 OpenCV hresize, 区分 xmin/xmax 边界)
    int h0, h1;
    int sx = xofs[dx];
    if (dx < xmin || dx >= xmax) {
        // 边界区域: D = S[sx] * 2048
        h0 = S0[sx] * 2048;
        h1 = S1[sx] * 2048;
    } else {
        short a0 = xalpha[dx * 2];
        short a1 = xalpha[dx * 2 + 1];
        h0 = S0[sx] * a0 + S0[sx + 1] * a1;
        h1 = S1[sx] * a0 + S1[sx + 1] * a1;
    }

    // 垂直插值: 匹配 OpenCV VResizeLinear<uchar,int,short> 特化版
    int val = (((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2;

    dst[dy * dw + dx] = (uint8_t)(val < 0 ? 0 : (val > 255 ? 255 : val));
}

// ============================================================================
// 缓冲区管理: 按需分配/扩容每条流水线的 device + pinned host 缓冲区
// ============================================================================
void NativeGpuProcessor::ensure_buffers(size_t src_size, size_t dst_size) {
    bool need_src = (src_size > alloc_src_size);
    bool need_dst = (dst_size > alloc_dst_size);
    if (!need_src && !need_dst) return;

    // 扩容前先等待所有流完成
    for (int i = 0; i < NUM_PIPES; i++)
        cudaStreamSynchronize(pipes[i].stream);

    for (int i = 0; i < NUM_PIPES; i++) {
        auto& p = pipes[i];
        if (need_src) {
            if (p.d_src) cudaFree(p.d_src);
            if (p.d_tex_full) cudaFree(p.d_tex_full);
            if (p.d_gauss_tmp) cudaFree(p.d_gauss_tmp);
            if (p.h_src_pinned) cudaFreeHost(p.h_src_pinned);
            cudaMalloc(&p.d_src, src_size);
            cudaMalloc(&p.d_tex_full, src_size);
            cudaMalloc(&p.d_gauss_tmp, src_size * sizeof(float));
            cudaMallocHost(&p.h_src_pinned, src_size);
        }
        if (need_dst) {
            if (p.d_dst_img) cudaFree(p.d_dst_img);
            if (p.d_dst_tex) cudaFree(p.d_dst_tex);
            if (p.h_img_pinned) cudaFreeHost(p.h_img_pinned);
            if (p.h_tex_pinned) cudaFreeHost(p.h_tex_pinned);
            cudaMalloc(&p.d_dst_img, dst_size);
            cudaMalloc(&p.d_dst_tex, dst_size);
            cudaMallocHost(&p.h_img_pinned, dst_size);
            cudaMallocHost(&p.h_tex_pinned, dst_size);
        }
    }
    if (need_src) alloc_src_size = src_size;
    if (need_dst) alloc_dst_size = dst_size;
}

// ============================================================================
// Resize 系数预计算 (CPU 侧, 所有 pipeline 共享)
// ============================================================================
void NativeGpuProcessor::ensure_coefficients(int sw, int sh, int dw, int dh) {
    if (sw == last_coeff_sw && sh == last_coeff_sh && dw == last_coeff_dw && dh == last_coeff_dh)
        return;
    last_coeff_sw = sw; last_coeff_sh = sh;
    last_coeff_dw = dw; last_coeff_dh = dh;

    double inv_sx = (double)dw / (double)sw;
    double inv_sy = (double)dh / (double)sh;
    double sc_x = 1.0 / inv_sx;
    double sc_y = 1.0 / inv_sy;

    std::vector<int> h_xofs(dw);
    std::vector<short> h_xalpha(dw * 2);
    int xmin = 0, xmax = dw;
    for (int dx = 0; dx < dw; dx++) {
        float ffx = (float)((dx + 0.5) * sc_x - 0.5);
        int ssx = (int)floorf(ffx);
        ffx -= ssx;
        if (ssx < 0) { ffx = 0; ssx = 0; xmin = dx + 1; }
        if (ssx >= sw - 1) { ffx = 0; ssx = sw - 1; xmax = std::min(xmax, dx); }
        h_xofs[dx] = ssx;
        h_xalpha[dx * 2] = (short)cvRound((1.0f - ffx) * 2048.0f);
        h_xalpha[dx * 2 + 1] = (short)cvRound(ffx * 2048.0f);
    }

    std::vector<int> h_yofs(dh);
    std::vector<short> h_yalpha(dh * 2);
    for (int dy = 0; dy < dh; dy++) {
        float ffy = (float)((dy + 0.5) * sc_y - 0.5);
        int ssy = (int)floorf(ffy);
        ffy -= ssy;
        if (ssy < 0) { ffy = 0; ssy = 0; }
        if (ssy >= sh - 1) { ffy = 0; ssy = sh - 1; }
        h_yofs[dy] = ssy;
        h_yalpha[dy * 2] = (short)cvRound((1.0f - ffy) * 2048.0f);
        h_yalpha[dy * 2 + 1] = (short)cvRound(ffy * 2048.0f);
    }

    if (d_xofs) cudaFree(d_xofs);
    if (d_yofs) cudaFree(d_yofs);
    if (d_xalpha) cudaFree(d_xalpha);
    if (d_yalpha) cudaFree(d_yalpha);
    cudaMalloc(&d_xofs, dw * sizeof(int));
    cudaMalloc(&d_yofs, dh * sizeof(int));
    cudaMalloc(&d_xalpha, dw * 2 * sizeof(short));
    cudaMalloc(&d_yalpha, dh * 2 * sizeof(short));
    cudaMemcpy(d_xofs, h_xofs.data(), dw * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yofs, h_yofs.data(), dh * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xalpha, h_xalpha.data(), dw * 2 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yalpha, h_yalpha.data(), dh * 2 * sizeof(short), cudaMemcpyHostToDevice);
    resize_xmin = xmin;
    resize_xmax = xmax;
}

// ============================================================================
// 异步提交: memcpy 到 pinned → 入队 H2D + kernels + D2H, 立即返回
// ============================================================================
int NativeGpuProcessor::submit(const uint8_t* h_src, int sw, int sh,
                                int dw, int dh, int radius) {
    size_t src_size = (size_t)sw * sh;
    size_t dst_size = (size_t)dw * dh;

    ensure_buffers(src_size, dst_size);
    ensure_coefficients(sw, sh, dw, dh);

    int p = next_pipe;
    next_pipe = (next_pipe + 1) % NUM_PIPES;
    auto& pipe = pipes[p];

    // CPU→pinned (同步, ~0.5ms for 1958×1958)
    memcpy(pipe.h_src_pinned, h_src, src_size);

    // 异步 H2D (从 pinned memory 才是真正异步)
    cudaMemcpyAsync(pipe.d_src, pipe.h_src_pinned, src_size, cudaMemcpyHostToDevice, pipe.stream);

    dim3 block(16, 16);
    dim3 grid_src((sw + 15) / 16, (sh + 15) / 16);
    dim3 grid_dst((dw + 15) / 16, (dh + 15) / 16);

    // Step 1: 全分辨率纹理特征 (dilate-erode-subtract 融合)
    texture_full_kernel<<<grid_src, block, 0, pipe.stream>>>(pipe.d_src, pipe.d_tex_full, sw, sh, radius);

    // Step 2: 高斯模糊行滤波 (uint8 → float)
    gauss_row_kernel<<<grid_src, block, 0, pipe.stream>>>(pipe.d_tex_full, pipe.d_gauss_tmp, sw, sh);

    // Step 3: 高斯模糊列滤波 (float → uint8, 结果覆写 d_tex_full)
    gauss_col_kernel<<<grid_src, block, 0, pipe.stream>>>(pipe.d_gauss_tmp, pipe.d_tex_full, sw, sh);

    // Step 4 & 5: 双线性缩放 (使用预计算系数, 所有 pipeline 共享)
    resize_linear_kernel<<<grid_dst, block, 0, pipe.stream>>>(
        pipe.d_src, pipe.d_dst_img, sw, sh, dw, dh,
        d_xofs, d_xalpha, d_yofs, d_yalpha, resize_xmin, resize_xmax);
    resize_linear_kernel<<<grid_dst, block, 0, pipe.stream>>>(
        pipe.d_tex_full, pipe.d_dst_tex, sw, sh, dw, dh,
        d_xofs, d_xalpha, d_yofs, d_yalpha, resize_xmin, resize_xmax);

    // 异步 D2H → pinned memory
    cudaMemcpyAsync(pipe.h_img_pinned, pipe.d_dst_img, dst_size, cudaMemcpyDeviceToHost, pipe.stream);
    cudaMemcpyAsync(pipe.h_tex_pinned, pipe.d_dst_tex, dst_size, cudaMemcpyDeviceToHost, pipe.stream);

    return p;
}

// ============================================================================
// 等待指定 pipeline 完成, 将 pinned memory 中的结果拷贝给调用方
// ============================================================================
void NativeGpuProcessor::wait(int pipe_idx, uint8_t* h_img_s, uint8_t* h_tex_s,
                               int dw, int dh) {
    auto& pipe = pipes[pipe_idx];
    cudaStreamSynchronize(pipe.stream);

    size_t dst_size = (size_t)dw * dh;
    memcpy(h_img_s, pipe.h_img_pinned, dst_size);
    memcpy(h_tex_s, pipe.h_tex_pinned, dst_size);
}

// ============================================================================
// 同步便捷接口 (submit + wait)
// ============================================================================
void NativeGpuProcessor::process(const uint8_t* h_src, int sw, int sh,
                                  uint8_t* h_img_s, uint8_t* h_tex_s,
                                  int dw, int dh, int radius, int scale) {
    int p = submit(h_src, sw, sh, dw, dh, radius);
    wait(p, h_img_s, h_tex_s, dw, dh);
}
