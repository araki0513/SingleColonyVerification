#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdint.h>

class NativeGpuProcessor {
public:
    NativeGpuProcessor();
    ~NativeGpuProcessor();

    void process(const uint8_t* h_src, int sw, int sh, uint8_t* h_img_s, uint8_t* h_tex_s, int dw, int dh, int radius, int scale);

private:
    cudaStream_t stream = nullptr;
    uint8_t *d_src = nullptr;
    uint8_t *d_tex_full = nullptr;
    float *d_gauss_tmp = nullptr;
    uint8_t *d_dst_img = nullptr;
    uint8_t *d_dst_tex = nullptr;
    size_t last_src_size = 0;
    size_t last_dst_size = 0;
    // resize 系数缓冲区 (CPU 预计算, 上传 GPU)
    int *d_xofs = nullptr, *d_yofs = nullptr;
    short *d_xalpha = nullptr, *d_yalpha = nullptr;
    int last_coeff_dw = 0, last_coeff_dh = 0, last_coeff_sw = 0, last_coeff_sh = 0;
    int resize_xmin = 0, resize_xmax = 0;
};
