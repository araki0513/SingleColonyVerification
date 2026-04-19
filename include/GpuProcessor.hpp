#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

/**
 * @brief CUDA 处理上下文，管理 Stream 和算子，避免重复分配资源
 */
struct CudaPipelineContext {
    cv::cuda::Stream stream;
    cv::Ptr<cv::cuda::Filter> dilate_filter;
    cv::Ptr<cv::cuda::Filter> erode_filter;
    cv::Ptr<cv::cuda::Filter> gauss_filter;

    CudaPipelineContext(int radius) {
        auto rect_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(radius, radius));
        dilate_filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8U, rect_element);
        erode_filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8U, rect_element);
        gauss_filter = cv::cuda::createGaussianFilter(CV_8U, CV_8U, cv::Size(3, 3), 0);
    }
};

/**
 * @brief 简单的异步 CUDA 处理器，用于封装生产/消费逻辑
 */
class GpuAsyncProcessor {
public:
    GpuAsyncProcessor(int win_size) : context(win_size) {}

    // 图像上传并计算纹理特征 (Async)
    void process_texture_async(const cv::Mat& h_src, cv::cuda::GpuMat& d_img_s, cv::cuda::GpuMat& d_tex_s, int scale) {
        d_raw.upload(h_src, context.stream);
        
        // 1. 计算纹理 (dilate - erode)
        context.dilate_filter->apply(d_raw, d_max, context.stream);
        context.erode_filter->apply(d_raw, d_min, context.stream);
        cv::cuda::subtract(d_max, d_min, d_tex, cv::noArray(), -1, context.stream);
        
        // 2. 高斯平滑
        context.gauss_filter->apply(d_tex, d_tex, context.stream);

        // 3. 降采样
        cv::cuda::resize(d_raw, d_img_s, cv::Size(d_raw.cols / scale, d_raw.rows / scale), 0, 0, cv::INTER_LINEAR, context.stream);
        cv::cuda::resize(d_tex, d_tex_s, cv::Size(d_tex.cols / scale, d_tex.rows / scale), 0, 0, cv::INTER_LINEAR, context.stream);
    }

    void wait_completion() {
        context.stream.waitForCompletion();
    }

private:
    CudaPipelineContext context;
    cv::cuda::GpuMat d_raw, d_max, d_min, d_tex;
};
