#include "FileIO.hpp"
#include <regex>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <iomanip>

static const std::vector<std::string> SUPPORTED_EXT = {".jpg",".jpeg"};//,".tif",".tiff",".png"

std::pair<std::string,std::string> parse_filename_fov(const std::string &filename){
    // pattern similar to Python: (Well_..._Xmm..._Ymm...)_Ch(\d)_...
    std::regex re(R"((Well_A\d+_Xmm.*_Ymm.*)_Ch(\d)_.*\.(?:jpg|jpeg)$)", std::regex::icase);
    std::smatch m;
    if(std::regex_search(filename, m, re) && m.size()>2){
        return {m[1].str(), m[2].str()};
    }
    return {"",""};
}

std::tuple<std::string,std::string,int,double,double> extract_coordinates(const std::string &filename){
    std::regex re(R"(Well_([A-Z])(\d+)_Xmm([-\d\.]+)_Ymm([-\d\.]+)_)", std::regex::icase);
    std::smatch m;
    if(std::regex_search(filename, m, re) && m.size()>4){
        std::string row = m[1].str();
        int col = std::stoi(m[2].str());
        double x_mm = std::stod(m[3].str());
        double y_mm = std::stod(m[4].str());
        return {row + std::to_string(col), row, col, x_mm, y_mm};
    }
    return {"","",0,0.0,0.0};
}

bool ends_with_supported(const std::string &name){
    std::string lower = name;
    for(auto &c: lower) c = tolower(c);
    for(auto &ext: SUPPORTED_EXT) if(lower.size()>=ext.size() && lower.substr(lower.size()-ext.size())==ext) return true;
    return false;
}

void write_csv(const std::string &path, const std::vector<CellRecord> &rows){
    std::ofstream ofs(path);
    ofs << "Well,Row,Column,X Position (um),Y Position (um),Total Area (um2),Total Mean Intensity,Live Mean,Dead Mean,Class_Label\n";
    for(auto &r: rows){
        ofs << r.well << "," << r.row << "," << r.column << "," << r.x_um << "," << r.y_um << ","
            << r.total_area_um2 << "," << r.total_mean_intensity << "," << r.live_mean_intensity << ","
            << r.dead_mean_intensity << "," << r.class_label << "\n";
    }
    std::cout << "Wrote " << rows.size() << " rows to " << path << std::endl;
}

void write_plate_csv(const std::string &path, const std::vector<CellRecord> &rows){
    // group by Row and Column
    struct Agg{int total=0; int both=0; int live=0; int dead=0;};
    std::map<std::pair<std::string,int>, Agg> agg;
    for(auto &r: rows){
        int both = (r.class_label=="Both")?1:0;
        int live = (r.class_label=="Live"||r.class_label=="Both")?1:0;
        int dead = (r.class_label=="Dead"||r.class_label=="Both")?1:0;
        auto key = std::make_pair(r.row, r.column);
        auto &a = agg[key];
        a.total += 1; a.both += both; a.live += live; a.dead += dead;
    }
    std::ofstream ofs(path);
    ofs << "Well,Row,Column,Total Count,Live Count,Dead Count,Live+Dead Count,% Live,% Dead,% Live (corrected)\n";
    for(auto &kv: agg){
        const auto &rk = kv.first; const auto &a = kv.second;
        std::string well = rk.first + std::to_string(rk.second);
        double pct_live = a.total>0 ? (double)a.live*100.0/a.total : 0.0;
        double pct_dead = a.total>0 ? (double)a.dead*100.0/a.total : 0.0;
        double pct_live_corrected = a.total>0 ? (double)(a.live - a.both)*100.0/a.total : 0.0;
        ofs<<well<<","<<rk.first<<","<<rk.second<<","<<a.total<<","<<a.live<<","<<a.dead<<","<<a.both<<","<<pct_live<<","<<pct_dead<<","<<pct_live_corrected<<"\n";
    }
}

void write_well_json(const std::string &path, const std::string &well_name, const std::vector<FovData> &fovs, int experiment_id=3, const std::string &plate_name="Plate1", double elapsed_seconds=0.0){
    std::ofstream ofs(path);
    ofs << "{";
    ofs << "\"experiment_id\":" << experiment_id << ",";
    ofs << "\"plate_name\":\"" << plate_name << "\",";
    ofs << "\"well_name\":\"" << well_name << "\",";
    ofs << "\"elapsed_time_seconds\":" << std::fixed << std::setprecision(3) << elapsed_seconds << ",";
    ofs << "\"fovs_data\":[";
    for(size_t i=0;i<fovs.size();++i){
        const FovData &fd = fovs[i];
        ofs << "{";
        ofs << "\"pic_name\":\"" << fd.pic_name << "\",";
        ofs << "\"total_cells\":" << fd.total_cells << ",";
        ofs << "\"cells\": [";
        for(size_t j=0;j<fd.cells.size();++j){
            const JsonCell &jc = fd.cells[j];
            ofs << "{";
            ofs << "\"cell_id\":" << jc.cell_id << ",";
            ofs << "\"center\":{\"x\":"<<jc.center_x<<",\"y\":"<<jc.center_y<<"},";
            // contour
            ofs << "\"contour\":[";
            for(size_t k=0;k<jc.contour.size();++k){ ofs << "["<<jc.contour[k].first<<","<<jc.contour[k].second<<"]"; if(k+1<jc.contour.size()) ofs<<","; }
            ofs << "],";
            // live_contour
            ofs << "\"live_contour\":[";
            for(size_t k=0;k<jc.live_contour.size();++k){ ofs << "["<<jc.live_contour[k].first<<","<<jc.live_contour[k].second<<"]"; if(k+1<jc.live_contour.size()) ofs<<","; }
            ofs << "],";
            // dead_contour
            ofs << "\"dead_contour\":[";
            for(size_t k=0;k<jc.dead_contour.size();++k){ ofs << "["<<jc.dead_contour[k].first<<","<<jc.dead_contour[k].second<<"]"; if(k+1<jc.dead_contour.size()) ofs<<","; }
            ofs << "],";
            ofs << "\"class\":\""<<jc.class_label<<"\"";
            ofs << "}";
            if(j+1<fd.cells.size()) ofs<<",";
        }
        ofs << "]}";
        if(i+1<fovs.size()) ofs<<",";
    }
    ofs << "]}";
}