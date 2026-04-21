// Utils2.cpp: C++ 实现 single_colony_verification.py 中除 run_pipeline 外的所有方法
#include <regex>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <numbers>
#include <opencv2/opencv.hpp>
#include "FileIO.hpp"
#include "Config.hpp"

extern Config1 config;  // 引用全局配置对象
// #define DOWN_SAMPLE_FIRST true  // 是否先降采样再计算纹理特征，还是直接在原图上计算纹理特征（后者更慢但可能更准确）
// 1. extract_coordinates
std::tuple<std::string, std::string, std::string, double, double> extract_coordinates(const std::string& filename) {
    std::regex pattern(R"(Well_([A-Z])(\d+)_Xmm([\-\d\.]+)_Ymm([\-\d\.]+))", std::regex::icase);
    std::smatch match;
    if (std::regex_search(filename, match, pattern)) {
        std::string row = match[1].str();
        std::string col = match[2].str();
        double x_mm = std::stod(match[3].str());
        double y_mm = std::stod(match[4].str());
        return {row + col, row, col, x_mm, y_mm};
    }
    std::regex fallback_pattern(R"(([A-Z])(\d+))", std::regex::icase);
    if (std::regex_search(filename, match, fallback_pattern)) {
        std::string row = match[1].str();
        std::string col = match[2].str();
        std::transform(row.begin(), row.end(), row.begin(), ::toupper);
        return {row + col, row, col, 0.0, 0.0};
    }
    return {"A1", "A", "1", 0.0, 0.0};
}

// 2. calculate_geometry
std::tuple<double, double, double> calculate_geometry(const std::vector<cv::Point>& contour, double exact_area) {
    if (contour.empty()) return {0.0, 0.0, 0.0};
    double perimeter = cv::arcLength(contour, true);
    if (perimeter == 0 || exact_area == 0) return {0.0, 0.0, 0.0};
    double form_factor = (4 * M_PI * exact_area) / (perimeter * perimeter);
    double long_axis = 0, short_axis = 0;
    if (contour.size() >= 5) {
        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        long_axis = std::max(ellipse.size.width, ellipse.size.height);
        short_axis = std::min(ellipse.size.width, ellipse.size.height);
    } else {
        cv::RotatedRect rect = cv::minAreaRect(contour);
        long_axis = std::max(rect.size.width, rect.size.height);
        short_axis = std::min(rect.size.width, rect.size.height);
    }
    double aspect_ratio = (long_axis > 0) ? (short_axis / long_axis) : 0.0;
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double hull_perimeter = cv::arcLength(hull, true);
    double smoothness = (perimeter > 0) ? (hull_perimeter / perimeter) : 0.0;
    return {form_factor, smoothness, aspect_ratio};
}

cv::Mat load_img(const std::string& path){
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(img.empty()){
        std::cerr << "Failed to load image: " << path << std::endl;
        return cv::Mat();
    }
    return img; // 仅返回原图，纹理计算移至 GPU
}

// 3. 简化版 process_well（仅结构，需根据实际需求补充）
// 参数说明：
// well_name: 孔名
// fov_list: FOV 文件名列表
// well_origin: pair<double, double>，孔中心坐标
// 返回：fov_json_map, csv_rows


std::pair<std::map<std::string, std::vector<CellJson>>, std::vector<CellInfo>>
process_well(const std::string& well_name, const std::vector<std::string>& fov_list, std::pair<double, double> well_origin, const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& texs) {
    using Clock = std::chrono::steady_clock;
    std::chrono::steady_clock::time_point DSt0, DSt1;
    const auto t0 = Clock::now();
    int DSms = 0;
    int w_orig = 1958, h_orig = 1958;
    double origin_x_mm = well_origin.first, origin_y_mm = well_origin.second;
    // 1. 计算所有FOV的全局坐标范围
    double min_x = 1e12, min_y = 1e12, max_x = -1e12, max_y = -1e12;
    struct FovInfo {
        std::string file, well, row, col;
        double tl_x, tl_y, x_mm, y_mm;
    };
    std::vector<FovInfo> fov_infos;
    fov_infos.reserve(fov_list.size());
    for (const auto& f : fov_list) {
        auto [well, row, col, x_mm, y_mm] = extract_coordinates(f);
        double tl_x = (x_mm - origin_x_mm) * 1000.0 - w_orig / 2.0;
        double tl_y = (y_mm - origin_y_mm) * 1000.0 - h_orig / 2.0;
        min_x = std::min(min_x, tl_x);
        min_y = std::min(min_y, tl_y);
        max_x = std::max(max_x, tl_x + w_orig);
        max_y = std::max(max_y, tl_y + h_orig);
        fov_infos.push_back({f, well, row, col, tl_x, tl_y, x_mm, y_mm});
    }
    int width = static_cast<int>(std::ceil(max_x - min_x));
    int height = static_cast<int>(std::ceil(max_y - min_y));
    int scale = static_cast<int>(config.PIXEL_SIZE_UM);
    int width_small = width / scale + 1;
    int height_small = height / scale + 1;
    cv::Mat full_img_small = cv::Mat::zeros(height_small, width_small, CV_8U);
    cv::Mat full_texture_small = cv::Mat::zeros(height_small, width_small, CV_8U);
    const auto t1 = Clock::now();
    // 2. 拼接纹理
    int win_size_px_orig = std::max(3, static_cast<int>(std::round(config.TEXTURE_WINDOW_UM / 1.0)));
    if (win_size_px_orig % 2 == 0) win_size_px_orig += 1;
    cv::Mat kernel_orig = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(win_size_px_orig, win_size_px_orig));
    // int win_size_px_small = std::max(3, static_cast<int>(std::round(config.TEXTURE_WINDOW_UM / config.PIXEL_SIZE_UM)));
    // if (win_size_px_small % 2 == 0) win_size_px_small += 1;
    // cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(win_size_px_small, win_size_px_small));
    for (const auto& fov : fov_infos) {
        /*const cv::Mat img = imgs[&fov - &fov_infos[0]]; // 根据fov在fov_infos中的位置获取对应的图像
        //cv::Mat img = cv::imread(fov.file, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        int h_i = img.rows, w_i = img.cols;
        cv::Mat img_s, tex_s, texture;
        #ifndef DOWN_SAMPLE_FIRST
        DSt0 = Clock::now();
        cv::Mat local_max, local_min;
        cv::dilate(img, local_max, kernel_orig);
        cv::erode(img, local_min, kernel_orig);
        cv::subtract(local_max, local_min, texture);
        cv::GaussianBlur(texture, texture, cv::Size(3, 3), 0);
        DSt1 = Clock::now();
        cv::resize(img, img_s, cv::Size(w_i / scale, h_i / scale));
        cv::resize(texture, tex_s, cv::Size(w_i / scale, h_i / scale));
        
        DSms += std::chrono::duration_cast<std::chrono::milliseconds>(DSt1 - DSt0).count();
        #else
        cv::resize(img, img_s, cv::Size(w_i / scale, h_i / scale));
        //cv::resize(texture, tex_s, cv::Size(w_i / scale, h_i / scale));
        cv::Mat local_max, local_min;
        cv::dilate(img_s, local_max, kernel_orig);
        cv::erode(img_s, local_min, kernel_orig);
        cv::subtract(local_max, local_min, tex_s);
        cv::GaussianBlur(tex_s, tex_s, cv::Size(3, 3), 0);
        #endif*/
        const cv::Mat img_s = imgs[&fov - &fov_infos[0]];
        const cv::Mat tex_s = texs[&fov - &fov_infos[0]];
        int x_start = static_cast<int>(std::round((fov.tl_x - min_x) / scale));
        int y_start = static_cast<int>(std::round((fov.tl_y - min_y) / scale));
        int h_s = img_s.rows, w_s = img_s.cols;
        int y_end = std::min(y_start + h_s, height_small);
        int x_end = std::min(x_start + w_s, width_small);
        int h_s_crop = y_end - y_start;
        int w_s_crop = x_end - x_start;
        cv::Mat roi_tex = full_texture_small(cv::Rect(x_start, y_start, w_s_crop, h_s_crop));
        cv::max(roi_tex, tex_s(cv::Rect(0, 0, w_s_crop, h_s_crop)), roi_tex);
        cv::Mat roi_img = full_img_small(cv::Rect(x_start, y_start, w_s_crop, h_s_crop));
        img_s(cv::Rect(0, 0, w_s_crop, h_s_crop)).copyTo(roi_img);
    }
    cv::Mat img_small = full_img_small;
    const auto t2 = Clock::now();
    // 3. 分割
    cv::Mat binary;
    cv::threshold(full_texture_small, binary, config.INTENSITY_THRESHOLD, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(config.MORPH_CLOSE_KERNEL, config.MORPH_CLOSE_KERNEL));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    const auto t3 = Clock::now();
    // 4. 连通域分析
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    std::map<std::string, std::vector<CellJson>> fov_json_map;
    for (const auto& fov : fov_infos) {
        fov_json_map.emplace(fov.file, std::vector<CellJson>{});
    }
    std::vector<CellInfo> csv_rows;
    csv_rows.reserve(num_labels > 1 ? static_cast<size_t>(num_labels - 1) : 0);
    int cell_id = 1;
    for (int i = 1; i < num_labels; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w_b = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h_b = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area_px = stats.at<int>(i, cv::CC_STAT_AREA);
        double area_um2 = area_px * (config.PIXEL_SIZE_UM * config.PIXEL_SIZE_UM);
        if (area_um2 < config.MIN_COLONY_AREA_UM2) continue;
        cv::Mat roi_mask = (labels(cv::Rect(x, y, w_b, h_b)) == i);
        roi_mask.convertTo(roi_mask, CV_8U, 255);
        cv::Mat roi_img = img_small(cv::Rect(x, y, w_b, h_b));
        double mean_int = cv::mean(roi_img, roi_mask)[0];
        if (mean_int < config.MIN_INTENSITY || mean_int > config.MAX_INTENSITY) continue;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(roi_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (contours.empty()) continue;
        auto& cnt = *std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
        std::vector<cv::Point> cnt_small;
        for (const auto& pt : cnt) cnt_small.push_back(cv::Point(pt.x + x, pt.y + y));
        auto [c_ff, c_sm, c_ar] = calculate_geometry(cnt_small, area_px);
        if (c_ar < config.MIN_COLONY_ASPECT_RATIO || c_ff < config.MIN_FORM_FACTOR) continue;
        std::vector<cv::Point> cnt_orig;
        for (const auto& pt : cnt_small) cnt_orig.push_back(cv::Point(pt.x * scale, pt.y * scale));
        double cx_px_orig = centroids.at<double>(i, 0) * scale;
        double cy_px_orig = centroids.at<double>(i, 1) * scale;
        double global_x_um = cx_px_orig + min_x;
        double global_y_um = cy_px_orig + min_y;
        double dist_to_center_um = std::sqrt(global_x_um * global_x_um + global_y_um * global_y_um);
        if (dist_to_center_um > config.WELL_RADIUS_UM) continue;
        double integrated_int = mean_int * area_px;
        // 寻找最近FOV
        const FovInfo* best_fov = nullptr;
        double min_dist = 1e12;
        for (const auto& fov : fov_infos) {
            double fov_cx = fov.tl_x + w_orig / 2.0;
            double fov_cy = fov.tl_y + h_orig / 2.0;
            double dist = (global_x_um - fov_cx) * (global_x_um - fov_cx) + (global_y_um - fov_cy) * (global_y_um - fov_cy);
            if (dist < min_dist) { min_dist = dist; best_fov = &fov; }
        }
        if (!best_fov) continue;
        // 填充csv_rows
        csv_rows.push_back(CellInfo{
            well_name, best_fov->row, best_fov->col, true,
            global_x_um, global_y_um, 0.0, dist_to_center_um, area_um2,
            c_ff, c_sm, c_ar, mean_int, integrated_int
        });
        // 填充fov_json_map（仅结构，未写JSON文件）
        if (cnt_orig.empty()) continue;
        cv::Rect cnt_rect = cv::boundingRect(cnt_orig);
        for (const auto& fov : fov_infos) {
            double fov_tl_x = fov.tl_x - min_x;
            double fov_tl_y = fov.tl_y - min_y;
            if (cnt_rect.x < fov_tl_x + w_orig && cnt_rect.x + cnt_rect.width > fov_tl_x &&
                cnt_rect.y < fov_tl_y + h_orig && cnt_rect.y + cnt_rect.height > fov_tl_y) {
                std::vector<std::pair<int, int>> contour_pts;
                for (const auto& pt : cnt_orig) {
                    int px = static_cast<int>(pt.x - fov_tl_x);
                    int py = static_cast<int>(pt.y - fov_tl_y);
                    contour_pts.emplace_back(px, py);
                }
                int cx_fov = static_cast<int>(cx_px_orig - fov_tl_x);
                int cy_fov = static_cast<int>(cy_px_orig - fov_tl_y);
                fov_json_map[fov.file].push_back(CellJson{cell_id, cx_fov, cy_fov, contour_pts, "Colony"});
            }
        }
        ++cell_id;
    }
    const auto t4 = Clock::now();
    const auto ms_layout = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const auto ms_stitch = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    const auto ms_segment = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    const auto ms_analyze = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    const auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t0).count();
    std::cout << "[Perf] well=" << well_name
              << " layout=" << ms_layout << "ms"
              << " stitch=" << ms_stitch << "ms"
              << " segment=" << ms_segment << "ms"
              << " analyze=" << ms_analyze << "ms"
              << " total=" << ms_total << "ms"
              << " colonies=" << csv_rows.size() << '\n';
    return {fov_json_map, csv_rows};
}
namespace {

std::string current_report_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now);
#else
    local_tm = *std::localtime(&now);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y/%m/%d %H:%M:%S");
    return oss.str();
}

std::string make_file_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now);
#else
    local_tm = *std::localtime(&now);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y-%m-%d-%H-%M-%S");
    return oss.str();
}

void write_utf8_bom(std::ofstream& ofs) {
    ofs << "\xEF\xBB\xBF";
}

std::string escape_json(const std::string& input) {
    std::ostringstream oss;
    for (unsigned char ch : input) {
        switch (ch) {
        case '"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            if (ch < 0x20) {
                oss << "\\u"
                    << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch)
                    << std::dec << std::setfill(' ');
            } else {
                oss << static_cast<char>(ch);
            }
        }
    }
    return oss.str();
}

int parse_column_number(const std::string& column) {
    try {
        return std::stoi(column);
    } catch (...) {
        return 0;
    }
}

std::string format_general(double value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    std::ostringstream oss;
    oss << std::setprecision(15) << value;
    return oss.str();
}

std::string format_percent(double value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(5) << value << '%';
    return oss.str();
}

double sample_stddev(const std::vector<double>& values) {
    if (values.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double squared_sum = 0.0;
    for (double value : values) {
        const double diff = value - mean;
        squared_sum += diff * diff;
    }
    return std::sqrt(squared_sum / static_cast<double>(values.size() - 1));
}

std::string metric_value_or_default(
    const std::map<std::pair<std::string, int>, double>& values,
    const std::string& row,
    int col,
    bool integer_metric,
    bool fill_missing_with_zero,
    bool percent_metric) {
    const auto it = values.find({row, col});
    if (it == values.end()) {
        if (percent_metric) {
            return "NaN";
        }
        return fill_missing_with_zero ? "0" : "NaN";
    }
    if (integer_metric) {
        return std::to_string(static_cast<int>(std::llround(it->second)));
    }
    if (percent_metric) {
        return format_percent(it->second);
    }
    return format_general(it->second);
}

}  // namespace

// #define DOWN_SAMPLE_FIRST true  // 是否先降采样再计算纹理特征，还是直接在原图上计算纹理特征（后者更慢但可能更准确）
// 1. extract_coordinates
double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    }
    return values[mid];
}

// 4. CSV 写入实现，增加头部信息
void write_object_csv(const std::vector<CellInfo>& all_cells_data, const std::string& output_path) {
    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs)
    {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }
    write_utf8_bom(ofs);
    const std::string report_timestamp = current_report_timestamp();
    // 头部信息
    ofs << "Plate ID,\"Colony Counting - Single Colony Verification - HeLa-RFP - 24-well\"\n";
    ofs << "Plate Name,\"\"\n";
    ofs << "Plate Description,\"24-well Corning 3526 plate, RFP-Hela Gradient 125, 25, 5, 1 cell/well\"\n";
    ofs << "Scan ID,\"2012/1/30 15:56:07\"\n";
    ofs << "Scan Description,\"Day 6\"\n";
    ofs << "Scan Result ID,\"" << report_timestamp << "\"\n";
    ofs << "Scan Result Description,\"\"\n";
    ofs << "Software Version,\"5.5.1.0\"\n";
    ofs << "Experiment Name,\"\"\n";
    ofs << "Application Name,\"Single Colony Verification\"\n";
    ofs << "Plate Type,\"24-Well Corning™ 3526 Plate Well Wall\"\n";
    ofs << "Acquisition Start/End Times,\"2012/1/30 15:56:07 - 2012/1/30 16:20:01\"\n";
    ofs << "Analysis Start Time,\"" << report_timestamp << "\"\n";
    ofs << "User ID,\"Local Administrator\"\n\n";
    ofs << "Scan Object-Level Data CSV Report\n\n";
    ofs << "Well,Row,Column,Total,X Position (µm),Y Position (µm),Distance to Nearest Neighbor (µm),Distance to Well Center (µm),Area (µm²),Form Factor,Smoothness,Aspect Ratio,Mean Intensity,Integrated Intensity\n";
    // 计算最近邻距离
    std::vector<CellInfo> sorted_cells = all_cells_data;
    std::sort(sorted_cells.begin(), sorted_cells.end(), [](const CellInfo& a, const CellInfo& b) {
        if (a.Row != b.Row) return a.Row < b.Row;
        if (a.Column != b.Column) return parse_column_number(a.Column) < parse_column_number(b.Column);
        if (a.Y != b.Y) return a.Y < b.Y;
        return a.X < b.X;
    });
    // 按Well分组计算最近邻
    std::map<std::string, std::vector<size_t>> well_indices;
    for (size_t i = 0; i < sorted_cells.size(); ++i) well_indices[sorted_cells[i].Well].push_back(i);
    for (const auto& [well, idxs] : well_indices) {
        for (size_t i : idxs) {
            double min_dist = 1e12;
            for (size_t j : idxs) {
                if (i == j) continue;
                double dx = sorted_cells[i].X - sorted_cells[j].X;
                double dy = sorted_cells[i].Y - sorted_cells[j].Y;
                double dist = std::sqrt(dx * dx + dy * dy);
                if (dist < min_dist) min_dist = dist;
            }
            sorted_cells[i].DistNN = (idxs.size() > 1) ? min_dist : 0.0;
        }
    }
    for (const auto& c : sorted_cells) {
        ofs << c.Well << ',' << c.Row << ',' << c.Column << ',' << (c.Total ? "True" : "False") << ','
            << format_general(c.X) << ',' << format_general(c.Y) << ',' << format_general(c.DistNN) << ','
            << format_general(c.DistCenter) << ',' << format_general(c.Area) << ','
            << format_general(c.FormFactor) << ',' << format_general(c.Smoothness) << ','
            << format_general(c.AspectRatio) << ',' << format_general(c.MeanIntensity) << ','
            << format_general(c.IntegratedIntensity) << '\n';
    }
}

// 5. well plate CSV 写入实现，增加头部和分组统计
void write_well_plate_csv(const std::vector<CellInfo>& all_cells_data, const std::string& output_path) {
    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs)
    {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }
    write_utf8_bom(ofs);
    const std::string report_timestamp = current_report_timestamp();

    std::map<std::pair<std::string, int>, std::vector<const CellInfo*>> grouped_cells;
    std::vector<std::string> all_rows;
    std::vector<int> all_cols;
    for (const auto& cell : all_cells_data) {
        const int col_num = parse_column_number(cell.Column);
        grouped_cells[{cell.Row, col_num}].push_back(&cell);
        all_rows.push_back(cell.Row);
        all_cols.push_back(col_num);
    }

    std::sort(all_rows.begin(), all_rows.end());
    all_rows.erase(std::unique(all_rows.begin(), all_rows.end()), all_rows.end());
    std::sort(all_cols.begin(), all_cols.end());
    all_cols.erase(std::unique(all_cols.begin(), all_cols.end()), all_cols.end());
    if (all_rows.empty()) {
        all_rows = {"A", "B", "C", "D", "E", "F", "G", "H"};
    }
    if (all_cols.empty()) {
        all_cols = {1, 2, 3, 4, 5, 6};
    }

    std::map<std::pair<std::string, int>, double> colony_count;
    std::map<std::pair<std::string, int>, double> avg_area;
    std::map<std::pair<std::string, int>, double> sd_area;
    std::map<std::pair<std::string, int>, double> total_area;
    std::map<std::pair<std::string, int>, double> avg_intensity;
    std::map<std::pair<std::string, int>, double> sd_intensity;
    std::map<std::pair<std::string, int>, double> avg_integrated;
    std::map<std::pair<std::string, int>, double> sd_integrated;
    std::map<std::pair<std::string, int>, double> cv_area;
    std::map<std::pair<std::string, int>, double> cv_intensity;
    std::map<std::pair<std::string, int>, double> cv_integrated;
    std::map<std::pair<std::string, int>, double> confluency;

    for (const auto& [key, cells] : grouped_cells) {
        std::vector<double> areas;
        std::vector<double> intensities;
        std::vector<double> integrateds;
        areas.reserve(cells.size());
        intensities.reserve(cells.size());
        integrateds.reserve(cells.size());
        for (const CellInfo* cell : cells) {
            areas.push_back(cell->Area);
            intensities.push_back(cell->MeanIntensity);
            integrateds.push_back(cell->IntegratedIntensity);
        }

        const double area_sum = std::accumulate(areas.begin(), areas.end(), 0.0);
        const double intensity_sum = std::accumulate(intensities.begin(), intensities.end(), 0.0);
        const double integrated_sum = std::accumulate(integrateds.begin(), integrateds.end(), 0.0);
        const double count = static_cast<double>(cells.size());
        const double area_mean = area_sum / count;
        const double intensity_mean = intensity_sum / count;
        const double integrated_mean = integrated_sum / count;
        const double area_std = sample_stddev(areas);
        const double intensity_std = sample_stddev(intensities);
        const double integrated_std = sample_stddev(integrateds);

        colony_count[key] = count;
        avg_area[key] = area_mean;
        sd_area[key] = area_std;
        total_area[key] = area_sum;
        avg_intensity[key] = intensity_mean;
        sd_intensity[key] = intensity_std;
        avg_integrated[key] = integrated_mean;
        sd_integrated[key] = integrated_std;
        cv_area[key] = std::isnan(area_std) ? std::numeric_limits<double>::quiet_NaN() : (area_std / area_mean) * 100.0;
        cv_intensity[key] = std::isnan(intensity_std) ? std::numeric_limits<double>::quiet_NaN() : (intensity_std / intensity_mean) * 100.0;
        cv_integrated[key] = std::isnan(integrated_std) ? std::numeric_limits<double>::quiet_NaN() : (integrated_std / integrated_mean) * 100.0;
        const double pi = std::acos(-1.0);
        confluency[key] = (area_sum / (pi * config.WELL_RADIUS_UM * config.WELL_RADIUS_UM)) * 100.0;
    }

    auto write_plate_map = [&](const std::string& metric_name,
                               const std::map<std::pair<std::string, int>, double>& values,
                               bool integer_metric,
                               bool percent_metric,
                               bool fill_missing_with_zero) {
        ofs << metric_name << ',';
        for (size_t i = 0; i < all_cols.size(); ++i) {
            ofs << all_cols[i];
            if (i + 1 != all_cols.size()) {
                ofs << ',';
            }
        }
        ofs << '\n';
        for (const auto& row : all_rows) {
            ofs << row << ',';
            for (size_t i = 0; i < all_cols.size(); ++i) {
                ofs << metric_value_or_default(values, row, all_cols[i], integer_metric, fill_missing_with_zero, percent_metric);
                ofs << ',';
            }
            ofs << '\n';
        }
        ofs << '\n';
    };

    ofs << "Plate ID,\"Colony Counting - Single Colony Verification - HeLa-RFP - 24-well\"\n";
    ofs << "Plate Name,\"\"\n";
    ofs << "Plate Description,\"24-well Corning 3526 plate, RFP-Hela Gradient 125, 25, 5, 1 cell/well\"\n";
    ofs << "Scan ID,\"2012/1/30 15:56:07\"\n";
    ofs << "Scan Description,\"Day 6\"\n";
    ofs << "Scan Result ID,\"" << report_timestamp << "\"\n";
    ofs << "Scan Result Description,\"\"\n";
    ofs << "Software Version,\"5.5.1.0\"\n";
    ofs << "Experiment Name,\"\"\n";
    ofs << "Application Name,\"Single Colony Verification\"\n";
    ofs << "Plate Type,\"24-Well Corning™ 3526 Plate Well Wall\"\n";
    ofs << "Acquisition Start/End Times,\"2012/1/30 15:56:07 - 2012/1/30 16:20:01\"\n";
    ofs << "Analysis Start Time,\"" << report_timestamp << "\"\n";
    ofs << "User ID,\"Local Administrator\"\n\n";
    ofs << "Measurement Plate Maps\n";

    write_plate_map("Confluency (%)", confluency, false, true, false);
    write_plate_map("Colony Count", colony_count, true, false, true);
    write_plate_map("Colony AVG Area (µm²)", avg_area, false, false, false);
    write_plate_map("Colony SD Area (µm²)", sd_area, false, false, false);
    write_plate_map("Colony Total Area (µm²)", total_area, false, false, true);
    write_plate_map("Colony %CV Area", cv_area, false, true, false);
    write_plate_map("AVG Intensity", avg_intensity, false, false, false);
    write_plate_map("SD Intensity", sd_intensity, false, false, false);
    write_plate_map("%CV Intensity", cv_intensity, false, true, false);
    write_plate_map("AVG Integrated Intensity", avg_integrated, false, false, false);
    write_plate_map("SD Integrated Intensity", sd_integrated, false, false, false);
    write_plate_map("%CV Integrated Intensity", cv_integrated, false, true, false);

    ofs << "Well Sampled (%),";
    for (size_t i = 0; i < all_cols.size(); ++i) {
        ofs << all_cols[i];
        if (i + 1 != all_cols.size()) {
            ofs << ',';
        }
    }
    ofs << '\n';
    for (const auto& row : all_rows) {
        ofs << row << ',';
        for (size_t i = 0; i < all_cols.size(); ++i) {
            ofs << "100.00000%,";
        }
        ofs << '\n';
    }
    ofs << '\n';

    ofs << "Analysis Settings Plate Maps,";
    for (size_t i = 0; i < all_cols.size(); ++i) {
        ofs << all_cols[i];
        if (i + 1 != all_cols.size()) {
            ofs << ',';
        }
    }
    ofs << '\n';
    for (const auto& row : all_rows) {
        ofs << row << ',';
        for (size_t i = 0; i < all_cols.size(); ++i) {
            ofs << "1,";
        }
        ofs << '\n';
    }
    ofs << '\n';

    const std::map<std::string, std::vector<double>> focus_map_rows = {
        {"A", {3.11785626411438, 3.14482641220093, 3.16523957252502, 3.16630959510803, 3.15267777442932, 3.13433504104614}},
        {"B", {3.1365954875946, 3.1433253288269, 3.17933750152588, 3.17716312408447, 3.15200471878052, 3.14075422286987}},
        {"C", {3.15345406532288, 3.16203022003174, 3.18679165840149, 3.1864812374115, 3.1582338809967, 3.14781165122986}},
        {"D", {3.16901874542236, 3.19069147109985, 3.20327067375183, 3.20356392860413, 3.18556666374207, 3.16289281845093}}
    };
    ofs << "Focus Position Map,";
    for (size_t i = 0; i < all_cols.size(); ++i) {
        ofs << all_cols[i];
        if (i + 1 != all_cols.size()) {
            ofs << ',';
        }
    }
    ofs << '\n';
    for (const auto& row : all_rows) {
        ofs << row << ',';
        auto it = focus_map_rows.find(row);
        std::vector<double> values = (it != focus_map_rows.end()) ? it->second : std::vector<double>(all_cols.size(), 0.0);
        if (values.size() < all_cols.size() && !values.empty()) {
            values.resize(all_cols.size(), values.back());
        }
        if (values.empty()) {
            values.resize(all_cols.size(), 0.0);
        }
        for (size_t i = 0; i < all_cols.size(); ++i) {
            ofs << format_general(values[i]) << ',';
        }
        ofs << '\n';
    }
    ofs << '\n';

    ofs << '\n' << '\n' << '\n';
    ofs << "Analysis Settings,Name,Version,Instance\n";
    ofs << ",Single Colony Verification - HeLa-RFP 24-well,1000,1\n";
    ofs << "General Settings,Item,Value\n";
    ofs << ",Analysis Resolution (µm/pixel),3\n";
    ofs << ",Well Mask,True\n";
    ofs << ",Well Mask Usage Mode,Automatic\n";
    ofs << ",% Well Mask,95\n";
    ofs << ",Well Mask Shape,Default\n";
    ofs << "Identification Settings,Colony Frame Settings\n";
    ofs << ",Item,Value\n";
    ofs << ",Algorithm,Texture\n";
    ofs << ",Intensity Threshold,10\n";
    ofs << ",Precision,High\n";
    ofs << ",Sharpen,None\n";
    ofs << ",Diameter (µm),8\n";
    ofs << ",Background Correction,True\n";
    ofs << ",Separate Touching Colonies,True\n";
    ofs << ",Minimum Thickness (µm),15\n";
    ofs << ",Saturated Intensity,0\n";
    ofs << "Pre-Filtering Settings,Colony Feature Settings\n";
    ofs << ",Item,Value\n";
    ofs << ",Min Colony Size (µm²),20000\n";
    ofs << ",Min Colony Aspect Ratio,0.09\n";
    ofs << ",Colony Intensity Range,20 to 255\n\n";
    ofs << "Classification Settings,Name,Version\n";
    ofs << ",Untitled Classification Settings,1000\n";
    ofs << "Classes,Class Name,Source Population Name\n";
    ofs << ",Total,ALL\n";
    ofs << "Gating,-none-\n\n\n";
    ofs << "Image Channel Settings,Gain,Exposure,SetupMode,Illumination Source\n";
    ofs << "41afc28e-223a-40cb-b9b5-1e752ee0c7f2,0,8561 (ums),Always,Brightfield\n\n\n";
}

void write_well_json(const std::string& well_name,
                     const std::vector<FovData>& fovs_data,
                     const std::filesystem::path& output_folder) {
    std::ofstream json_ofs(output_folder / (well_name + ".json"), std::ios::binary);
    if (!json_ofs) {
        std::cerr << "Failed to open json output for well " << well_name << std::endl;
        return;
    }

    json_ofs << '{'
             << "\"experiment_id\":\"" << escape_json(config.EXPERIMENT_ID) << "\"," 
             << "\"plate_name\":\"" << escape_json(config.PLATE_NAME) << "\"," 
             << "\"well_name\":\"" << escape_json(well_name) << "\"," 
             << "\"elapsed_time_seconds\":0,"
             << "\"fovs_data\":[";
    for (size_t fov_idx = 0; fov_idx < fovs_data.size(); ++fov_idx) {
        const auto& [fov_file, total_cells, cells] = fovs_data[fov_idx];
        json_ofs << '{'
                 << "\"pic_name\":\"" << escape_json(fov_file) << "\"," 
                 << "\"total_cells\":" << total_cells << ','
                 << "\"cells\":[";
        for (size_t cell_idx = 0; cell_idx < cells.size(); ++cell_idx) {
            const auto& cell = cells[cell_idx];
            json_ofs << '{'
                     << "\"cell_id\":" << cell.cell_id << ','
                     << "\"center\":{\"x\":" << cell.cx << ",\"y\":" << cell.cy << "},"
                     << "\"contour\":[";
            for (size_t pt_idx = 0; pt_idx < cell.contour.size(); ++pt_idx) {
                const auto& [px, py] = cell.contour[pt_idx];
                json_ofs << '[' << px << ',' << py << ']';
                if (pt_idx + 1 != cell.contour.size()) {
                    json_ofs << ',';
                }
            }
            json_ofs << "],\"class\":\"" << escape_json(cell.cell_class) << "\"}";
            if (cell_idx + 1 != cells.size()) {
                json_ofs << ',';
            }
        }
        json_ofs << "]}";
        if (fov_idx + 1 != fovs_data.size()) {
            json_ofs << ',';
        }
    }
    json_ofs << "]}";
}

void write_all_outputs(const std::map<std::string, std::vector<FovData>>& well_fov_results,
                       const std::vector<CellInfo>& all_colonies_data,
                       const std::filesystem::path& output_folder) {
    for (const auto& [well_name, fovs_data] : well_fov_results) {
        write_well_json(well_name, fovs_data, output_folder);
    }

    if (!all_colonies_data.empty()) {
        const std::string timestamp = make_file_timestamp();
        write_object_csv(all_colonies_data, (output_folder / (timestamp + "_object.csv")).string());
        write_well_plate_csv(all_colonies_data, (output_folder / (timestamp + "_well_plate.csv")).string());
    }
}


// 6. 批量处理主流程
// void batch_process(const std::map<std::string, std::vector<std::string>>& well_groups,
//                   const std::map<std::string, std::pair<double, double>>& well_origin_map,
//                   const std::string& output_folder) {
//     std::vector<CellInfo> all_colonies_data;
//     for (const auto& [well_name, fovs] : well_groups) {
//         auto it = well_origin_map.find(well_name);
//         if (it == well_origin_map.end()) continue;
//         auto [fov_json_map, csv_rows] = process_well(well_name, fovs, it->second, );
//         all_colonies_data.insert(all_colonies_data.end(), csv_rows.begin(), csv_rows.end());
//         // 可在此处写JSON文件，如需
//     }
//     if (!all_colonies_data.empty()) {
//         std::string csv_path = output_folder + "/object.csv";
//         std::string well_plate_path = output_folder + "/well_plate.csv";
//         write_object_csv(all_colonies_data, csv_path);
//         write_well_plate_csv(all_colonies_data, well_plate_path);
//     }
// }