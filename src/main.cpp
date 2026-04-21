#include <iostream>
#include <chrono>
#include <filesystem>
#include "ThreadPool.hpp"
#include "Config.hpp"
#include "FileIO.hpp"
#include "NativeGpuProcessor.cuh"

#define IGNORE_CODE 0
#include <mutex>
#include <filesystem>

Config1 config;  // 定义全局配置对象
namespace fs = std::filesystem;
int example_task(int id){
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Task " << id << " done\n";
    return id;
}
std::mutex wirte_mtx; // 用于保护写文件的互斥锁
#ifndef IGNORE_CODE
    int main(){
        auto start = std::chrono::high_resolution_clock::now();
        ThreadPool pool(config.NUM_THREADS);
        std::vector<std::future<int>> results;
        for(int i = 0; i < 20; ++i){
            results.push_back(pool.enqueue(example_task, i));
        }
        for(auto &res : results){
            std::cout << "Result: " << res.get() << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Enqueued 20 tasks in " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms\n";
        return 0;
    }
#endif

int main(){
    try {
        auto start = std::chrono::high_resolution_clock::now();
        cv::setUseOptimized(true);
        cv::setNumThreads(1);
        // 创建输出目录，防止写文件失败
        std::filesystem::create_directories(config.OUTPUT_FOLDER);
        // 载入文件列表
        std::vector<std::string> file_lists;
        for(auto &fp: fs::directory_iterator(config.IMAGE_FOLDER)){
            fs::path f = fp.path();
            if(f.extension() == ".jpg" && f.string().find("Xmm") != std::string::npos)
                file_lists.push_back(f.filename().string());
        }
        // 按孔位分组FOV文件
        std::map<std::string, std::vector<std::string>> well_groups;
        std::map<std::string, std::tuple<std::string, std::string, std::string, double, double>> parsed_info;
        for(auto &f: file_lists){
            std::string well, row, col;
            double x_mm, y_mm;
            std::tie(well, row, col, x_mm, y_mm) = extract_coordinates(f);
            well_groups[well].push_back(f);
            parsed_info[f] = {well, row, col, x_mm, y_mm};
        }
        // 计算每个孔的中心
        std::map<std::string, std::pair<double, double>> well_origin_map;
        for(const auto& [well, files]: well_groups){
            double sum_x = 0, sum_y = 0;
            for(const auto& f: files){
                auto [w, r, c, x_mm, y_mm] = parsed_info[f];
                sum_x += x_mm;
                sum_y += y_mm;
            }
            int n = files.size();
            well_origin_map[well] = {sum_x / n, sum_y / n};
        }
        // 开始处理
        typedef std::tuple<std::string, int, std::vector<CellJson>> FovData;
        std::map<std::string, std::vector<FovData>> well_fov_results;
        std::vector<CellInfo> all_colonies_data;
        std::cout << "开始处理" << well_groups.size() << "个孔位(共 " <<file_lists.size() << " 个FOV文件)\n";

        ThreadPool process_pool(config.NUM_THREADS);
        ThreadPool load_pool(config.NUM_THREADS);
        std::vector<std::future<std::pair<std::map<std::string, std::vector<CellJson>>, std::vector<CellInfo>>>> future_results;
        
        std::vector<std::string> well_names;
        future_results.reserve(well_groups.size());
        well_names.reserve(well_groups.size());
        /*{   
            ThreadPool pool(config.NUM_THREADS);
            for(const auto& [well_name, fovs]: well_groups){
                well_names.push_back(well_name);
                auto fut= pool.enqueue([well_name, fovs, &well_fov_results, &all_colonies_data](std::pair<double, double> well_origin){
                    try{
                        std::cout << "[Thread] Start processing well: " << well_name << std::endl;
                        auto result = process_well(well_name, fovs, well_origin);
                        std::cout << "[Thread] Done processing well: " << well_name << std::endl;
                        {
                            std::lock_guard<std::mutex> lock(wirte_mtx);
                            auto [cell_json_map, cell_info_list] = result;
                            std::vector<FovData> fovs_data;
                            for(const auto& [fov_file, cells]: cell_json_map){
                                fovs_data.emplace_back(std::make_tuple(fov_file, cells.size(), cells));
                            }
                            well_fov_results[well_name] = fovs_data;
                            all_colonies_data.insert(all_colonies_data.end(), cell_info_list.begin(), cell_info_list.end());
                        }
                        return result;
                    }catch(const std::exception& e){
                        std::cerr << "Error processing well " << well_name << ": " << e.what() << std::endl;
                        return std::make_pair(std::map<std::string, std::vector<CellJson>>(), std::vector<CellInfo>());
                    }catch(...){
                        std::cerr << "Unknown error in well " << well_name << std::endl;
                        return std::make_pair(std::map<std::string, std::vector<CellJson>>(), std::vector<CellInfo>());
                    }
                    
                }, well_origin_map[well_name]);
                //auto [cell_json_map, cell_info_list] = fut.get();
                // 整理该孔位下所有 FOV 的 JSON 数据
            }
        }*/
        using Clock = std::chrono::steady_clock;
        int win_size_px_orig = std::max(3, static_cast<int>(std::round(config.TEXTURE_WINDOW_UM / 1.0)));
        if (win_size_px_orig % 2 == 0) win_size_px_orig += 1;
        int radius = win_size_px_orig / 2;
        int scale = static_cast<int>(config.PIXEL_SIZE_UM);
        NativeGpuProcessor gpu_proc;
        const auto t_global = Clock::now();

        // ===== 全局流水线: 所有 well 的 FOV 一次性提交到 load_pool =====
        struct FovMeta { int well_idx, fov_idx; };
        struct WellBucket {
            std::string name;
            std::vector<std::string> fov_paths;
            std::vector<cv::Mat> imgs, texs;
            int remaining;
        };
        std::vector<WellBucket> wells;
        wells.reserve(well_groups.size());
        std::vector<std::future<cv::Mat>> all_futures;
        std::vector<FovMeta> all_meta;
        all_futures.reserve(file_lists.size());
        all_meta.reserve(file_lists.size());

        for (const auto& [wn, fovs] : well_groups) {
            int wi = (int)wells.size();
            wells.push_back({wn, fovs,
                std::vector<cv::Mat>(fovs.size()),
                std::vector<cv::Mat>(fovs.size()),
                (int)fovs.size()});
            for (int fi = 0; fi < (int)fovs.size(); fi++) {
                all_futures.push_back(
                    load_pool.enqueue([p = fovs[fi]]() { return load_img(p); }));
                all_meta.push_back({wi, fi});
            }
        }

        // 当某 well 全部图像 GPU 处理完毕, 自动提交 process_well
        auto submit_well = [&](WellBucket& wb) {
            well_names.push_back(wb.name);
            future_results.push_back(
                process_pool.enqueue(
                    [name = wb.name, fovs = std::move(wb.fov_paths),
                     imgs = std::move(wb.imgs), texs = std::move(wb.texs)]
                    (std::pair<double, double> origin) {
                        try {
                            std::cout << "[Thread] Start processing well: " << name << std::endl;
                            auto result = process_well(name, fovs, origin, imgs, texs);
                            std::cout << "[Thread] Done processing well: " << name << std::endl;
                            return result;
                        } catch (const std::exception& e) {
                            std::cerr << "Error processing well " << name << ": " << e.what() << std::endl;
                            return std::make_pair(std::map<std::string, std::vector<CellJson>>(), std::vector<CellInfo>());
                        } catch (...) {
                            std::cerr << "Unknown error in well " << name << std::endl;
                            return std::make_pair(std::map<std::string, std::vector<CellJson>>(), std::vector<CellInfo>());
                        }
                    },
                    well_origin_map[wb.name])
            );
        };

        // GPU 结果收集 + 按 well 聚合
        struct PendingGpu { int pipe_idx, dw, dh, well_idx, fov_idx; };
        std::vector<PendingGpu> pending;
        pending.reserve(all_futures.size());
        size_t gpu_done = 0;

        auto collect_gpu = [&]() {
            auto& pg = pending[gpu_done++];
            cv::Mat h_img(pg.dh, pg.dw, CV_8U);
            cv::Mat h_tex(pg.dh, pg.dw, CV_8U);
            gpu_proc.wait(pg.pipe_idx, h_img.data, h_tex.data, pg.dw, pg.dh);
            auto& wb = wells[pg.well_idx];
            wb.imgs[pg.fov_idx] = h_img;
            wb.texs[pg.fov_idx] = h_tex;
            if (--wb.remaining == 0) submit_well(wb);
        };

        // 主循环: 按顺序取加载结果 → 多流水线 GPU → 自动触发 well 后处理
        for (size_t i = 0; i < all_futures.size(); i++) {
            auto img_raw = all_futures[i].get();
            auto [wi, fi] = all_meta[i];

            if (img_raw.empty()) {
                if (--wells[wi].remaining == 0) submit_well(wells[wi]);
                continue;
            }

            int sw = img_raw.cols, sh = img_raw.rows;
            int dw = sw / scale, dh = sh / scale;

            if ((int)(pending.size() - gpu_done) >= NativeGpuProcessor::NUM_PIPES)
                collect_gpu();

            int pipe = gpu_proc.submit(img_raw.data, sw, sh, dw, dh, radius);
            pending.push_back({pipe, dw, dh, wi, fi});
        }
        while (gpu_done < pending.size()) collect_gpu();

        const auto t_global_end = Clock::now();
        std::cout << "[Main] Global pipeline: " << file_lists.size() << " images in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_global_end - t_global).count() << " ms\n";
        for(size_t i = 0; i < future_results.size(); ++i){
            auto [cell_json_map, cell_info_list] = future_results[i].get();
            std::string well_name = well_names[i];
            // 整理该孔位下所有 FOV 的 JSON 数据
            {
                std::lock_guard<std::mutex> lock(wirte_mtx);
                std::vector<FovData> fovs_data;
                fovs_data.reserve(cell_json_map.size());
                for(auto& [fov_file, cells]: cell_json_map){
                    fovs_data.emplace_back(std::make_tuple(fov_file, static_cast<int>(cells.size()), std::move(cells)));
                }
                well_fov_results[well_name] = std::move(fovs_data);
                all_colonies_data.insert(
                    all_colonies_data.end(),
                    std::make_move_iterator(cell_info_list.begin()),
                    std::make_move_iterator(cell_info_list.end())
                );
            }
        }
        //write_all_outputs(well_fov_results, all_colonies_data, config.OUTPUT_FOLDER);
        // write_well_plate_csv(all_colonies_data, config.OUTPUT_FOLDER.string() + "/well_plate_data.csv");
        // write_object_csv(all_colonies_data, config.OUTPUT_FOLDER.string() + "/object_data.csv");
        write_all_outputs(well_fov_results, all_colonies_data, config.OUTPUT_FOLDER);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Complete tasks in " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "[Main] Fatal exception: " << e.what() << std::endl;
        return 1;
    } catch(...) {
        std::cerr << "[Main] Unknown fatal exception." << std::endl;
        return 2;
    }
}
