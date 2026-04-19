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
                file_lists.push_back(f.string());
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
        NativeGpuProcessor gpu_proc;

        for(const auto& [well_name, fovs]: well_groups){
            well_names.push_back(well_name);
            std::vector<std::future<std::pair<cv::Mat, cv::Mat>>> load_futures;
            const auto t0 = Clock::now();
            for(const auto& path: fovs){
                const auto p = path; 
                load_futures.emplace_back(
                    load_pool.enqueue([p](){
                        return load_img(p);
                    })
                );
            }
            
            std::vector<cv::Mat> imgs_small, texs_small;
            int scale = static_cast<int>(config.PIXEL_SIZE_UM);

            for(auto& fut: load_futures){
                auto [img_raw, _unused] = fut.get();
                if(img_raw.empty()) continue;

                int sw = img_raw.cols;
                int sh = img_raw.rows;
                int dw = sw / scale;
                int dh = sh / scale;

                cv::Mat h_img_s(dh, dw, CV_8U);
                cv::Mat h_tex_s(dh, dw, CV_8U);

                // 使用原生 CUDA Kernel 处理
                gpu_proc.process(img_raw.data, sw, sh, h_img_s.data, h_tex_s.data, dw, dh, radius, scale);

                imgs_small.emplace_back(h_img_s);
                texs_small.emplace_back(h_tex_s);
            }
            const auto t1 = Clock::now();
            std::cout << "[Main] Loaded and Native GPU preprocessed for well " << well_name << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
            future_results.push_back(
                process_pool.enqueue([well_name, fovs, imgs = std::move(imgs_small), texs = std::move(texs_small)](std::pair<double, double> well_origin){
                    try{
                        std::cout << "[Thread] Start processing well: " << well_name << std::endl;
                        auto result = process_well(well_name, fovs, well_origin, imgs, texs);
                        std::cout << "[Thread] Done processing well: " << well_name << std::endl;
                        return result;
                    }catch(const std::exception& e){
                        std::cerr << "Error processing well " << well_name << ": " << e.what() << std::endl;
                        return std::make_pair(std::map<std::string, std::vector<CellJson>>(), std::vector<CellInfo>());
                    }catch(...){
                        std::cerr << "Unknown error in well " << well_name << std::endl;
                        return std::make_pair(std::map<std::string, std::vector<CellJson>>(), std::vector<CellInfo>());
                    }
                }, well_origin_map[well_name])
            );
            
        }
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
        write_object_csv(all_colonies_data, config.OUTPUT_FOLDER.string() + "\\object.csv");
        write_well_plate_csv(all_colonies_data, config.OUTPUT_FOLDER.string() + "\\well_plate.csv");
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
