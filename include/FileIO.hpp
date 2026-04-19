#include <string>
#include <tuple>
#include <map>
#include <opencv2/opencv.hpp>


struct CellInfo {
    std::string Well, Row, Column;
    bool Total;
    double X, Y, DistNN, DistCenter, Area, FormFactor, Smoothness, AspectRatio, MeanIntensity, IntegratedIntensity;
};

// 结构体用于JSON输出（如需写JSON可扩展）
struct CellJson {
    int cell_id;
    int cx, cy;
    std::vector<std::pair<int, int>> contour;
    std::string cell_class;
};
// 1. 从文件名提取坐标信息
std::tuple<std::string, std::string, std::string, double, double> extract_coordinates(const std::string& filename);

std::tuple<double, double, double> calculate_geometry(const std::vector<cv::Point>& contour, double exact_area);

std::pair<cv::Mat, cv::Mat> load_img(const std::string& path);

std::pair<std::map<std::string, std::vector<CellJson>>, std::vector<CellInfo>>
process_well(const std::string& well_name, const std::vector<std::string>& fov_list, std::pair<double, double> well_origin, const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& texs);

void write_well_plate_csv(const std::vector<CellInfo>& all_cells_data, const std::string& output_path);
void write_object_csv(const std::vector<CellInfo> &records, const std::string& output_path);

void batch_process(const std::map<std::string, std::vector<std::string>>& well_groups,
                  const std::map<std::string, std::pair<double, double>>& well_origin_map,
                  const std::string& output_folder);