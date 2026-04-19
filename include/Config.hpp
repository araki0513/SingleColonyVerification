#include <string>
#include <filesystem>

namespace fs = std::filesystem;
struct Config1
{
    /* data */
    // 常量参数
    fs::path IMAGE_FOLDER = R"(.\\Input)";
    fs::path OUTPUT_FOLDER = R"(.\\Output)";
    int NUM_THREADS = 8;
    // 物理分辨率：降采样后 1 pixel = 3 µm
    double PIXEL_SIZE_UM = 3.0;
    // 纹理分析的窗口直径 (µm)
    double TEXTURE_WINDOW_UM = 8.0;
    // 强度阈值
    double INTENSITY_THRESHOLD = 8;
    // 形态学闭运算核大小（像素）
    double MORPH_CLOSE_KERNEL = 5;

    // 克隆过滤条件
    double MIN_COLONY_AREA_UM2 = 15000.0;  // 最小面积 (µm²)
    // MAX_COLONY_AREA_UM2 = 500000.0;  // 最大面积 (µm²)
    double MIN_COLONY_ASPECT_RATIO = 0.09;  // 最小长宽比
    double MIN_FORM_FACTOR = 0.35;  // 最小形状因子（过滤细长条噪声）
    double MIN_INTENSITY = 5;  // 最小强度
    double MAX_INTENSITY = 255;  // 最大强度

    // 孔板有效半径（µm），用于过滤边缘区域
    double WELL_RADIUS_UM = 15600 * 0.5 * 0.95;  // Corning 3526 24孔板孔直径约 15600 µm，使用 95% Well Mask 过滤孔壁

    // 实验标识
    std::string EXPERIMENT_ID = "4";
    std::string PLATE_NAME = "Plate1";
};

extern Config1 config;
