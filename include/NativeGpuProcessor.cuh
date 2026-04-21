#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdint.h>

class NativeGpuProcessor {
public:
    static constexpr int NUM_PIPES = 3;

    NativeGpuProcessor();
    ~NativeGpuProcessor();

    // 异步提交: 拷贝到 pinned memory, 启动 GPU 流水线, 返回 pipeline 索引
    int submit(const uint8_t* h_src, int sw, int sh, int dw, int dh, int radius);

    // 等待指定 pipeline 完成, 将结果拷贝到 h_img_s / h_tex_s
    void wait(int pipe_idx, uint8_t* h_img_s, uint8_t* h_tex_s, int dw, int dh);

    // 同步接口 (submit + wait)
    void process(const uint8_t* h_src, int sw, int sh, uint8_t* h_img_s, uint8_t* h_tex_s, int dw, int dh, int radius, int scale);

private:
    struct Pipeline {
        cudaStream_t stream = nullptr;
        uint8_t *d_src = nullptr;
        uint8_t *d_tex_full = nullptr;
        float *d_gauss_tmp = nullptr;
        uint8_t *d_dst_img = nullptr;
        uint8_t *d_dst_tex = nullptr;
        uint8_t *h_src_pinned = nullptr;  // pinned host for async H2D
        uint8_t *h_img_pinned = nullptr;  // pinned host for async D2H
        uint8_t *h_tex_pinned = nullptr;  // pinned host for async D2H
    };

    Pipeline pipes[NUM_PIPES];
    int next_pipe = 0;
    size_t alloc_src_size = 0;
    size_t alloc_dst_size = 0;

    // 共享 resize 系数 (所有 pipeline 共用, 因为图像尺寸相同)
    int *d_xofs = nullptr, *d_yofs = nullptr;
    short *d_xalpha = nullptr, *d_yalpha = nullptr;
    int last_coeff_dw = 0, last_coeff_dh = 0, last_coeff_sw = 0, last_coeff_sh = 0;
    int resize_xmin = 0, resize_xmax = 0;

    void ensure_buffers(size_t src_size, size_t dst_size);
    void ensure_coefficients(int sw, int sh, int dw, int dh);
};
