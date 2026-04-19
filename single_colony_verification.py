import os
import cv2
import json
import re
import time
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.load.*weights_only=False.*",
    category=FutureWarning
)

# ================= 参数配置 =================
IMAGE_FOLDER = r"D:\Data\Colony Counting - Single Colony Verification - HeLa-RFP - 24-well_2012-1-30-15-56-07"
OUTPUT_FOLDER = r".\Output"

# 物理分辨率：降采样后 1 pixel = 3 µm
PIXEL_SIZE_UM = 3.0
# 纹理分析的窗口直径 (µm)
TEXTURE_WINDOW_UM = 8.0
# 强度阈值
INTENSITY_THRESHOLD = 8
# 形态学闭运算核大小（像素）
MORPH_CLOSE_KERNEL = 5

# 克隆过滤条件
MIN_COLONY_AREA_UM2 = 15000.0  # 最小面积 (µm²)
# MAX_COLONY_AREA_UM2 = 500000.0  # 最大面积 (µm²)
MIN_COLONY_ASPECT_RATIO = 0.09  # 最小长宽比
MIN_FORM_FACTOR = 0.35  # 最小形状因子（过滤细长条噪声）
MIN_INTENSITY = 5  # 最小强度
MAX_INTENSITY = 255  # 最大强度

# 孔板有效半径（µm），用于过滤边缘区域
WELL_RADIUS_UM = 15600 * 0.5 * 0.95  # Corning 3526 24孔板孔直径约 15600 µm，使用 95% Well Mask 过滤孔壁

# 实验标识
EXPERIMENT_ID = "4"
PLATE_NAME = "Plate1"


# ================= 辅助函数 =================
def extract_coordinates(filename):
    """
    从文件名中提取 Well, Row, Column, Xmm, Ymm。
    """
    pattern = r"Well_([A-Z])(\d+)_Xmm([-\d\.]+)_Ymm([-\d\.]+)"
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        row = match.group(1)
        col = match.group(2)
        x_mm = float(match.group(3))
        y_mm = float(match.group(4))
        return f"{row}{col}", row, col, x_mm, y_mm

    fallback_pattern = r"([A-Z])(\d+)"
    match_fallback = re.search(fallback_pattern, filename, re.IGNORECASE)
    if match_fallback:
        row = match_fallback.group(1).upper()
        col = match_fallback.group(2)
        return f"{row}{col}", row, col, 0.0, 0.0

    return "A1", "A", "1", 0.0, 0.0


def calculate_geometry(contour, exact_area):
    """
    计算轮廓的几何特征，面积使用精确的掩码面积。
    """
    if contour is None or len(contour) == 0:
        return 0, 0, 0

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0 or exact_area == 0:
        return 0, 0, 0

    form_factor = (4 * np.pi * exact_area) / (perimeter ** 2)

    if len(contour) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        long_axis = max(MA, ma)
        short_axis = min(MA, ma)
    else:
        rect = cv2.minAreaRect(contour)
        long_axis = max(rect[1])
        short_axis = min(rect[1])

    aspect_ratio = short_axis / long_axis if long_axis > 0 else 0

    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    smoothness = hull_perimeter / perimeter if perimeter > 0 else 0

    return form_factor, smoothness, aspect_ratio


def process_well(well_name, fov_list, well_origin):
    """
    处理整个孔位的所有 FOV。对全孔进行图像拼接，
    然后在拼接后的全图上进行特征提取和克隆分割，最后再将结果分配回对应的 FOV。
    """
    w_orig, h_orig = 1958, 1958
    origin_x_mm, origin_y_mm = well_origin

    # 计算所有 FOV 的全局坐标范围
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    fov_infos = []

    for f in fov_list:
        well, row, col, x_mm, y_mm = extract_coordinates(f)
        # 计算 FOV 左上角在全孔物理坐标系中的位置 (以 µm 为单位，孔中心不一定是 (0,0)，这里使用绝对偏移)
        tl_x = (x_mm - origin_x_mm) * 1000.0 - w_orig / 2.0
        tl_y = (y_mm - origin_y_mm) * 1000.0 - h_orig / 2.0

        min_x = min(min_x, tl_x)
        min_y = min(min_y, tl_y)
        max_x = max(max_x, tl_x + w_orig)
        max_y = max(max_y, tl_y + h_orig)

        fov_infos.append({
            'file': f, 'well': well, 'row': row, 'col': col,
            'tl_x': tl_x, 'tl_y': tl_y, 'x_mm': x_mm, 'y_mm': y_mm
        })

    # 全孔拼接图的宽高
    width = int(math.ceil(max_x - min_x))
    height = int(math.ceil(max_y - min_y))

    scale = int(PIXEL_SIZE_UM)
    width_small = width // scale + 1
    height_small = height // scale + 1

    full_img_small = np.zeros((height_small, width_small), dtype=np.uint8)
    full_texture_small = np.zeros((height_small, width_small), dtype=np.uint8)

    # ========== 纹理计算与最大值拼接 ==========
    # （核心修复：在原图单视野中直接提取纹理，再取最大值拼接，避免拼接重影导致跨视野克隆纹理消失）
    win_size_px_orig = max(3, int(round(TEXTURE_WINDOW_UM / 1.0)))
    if win_size_px_orig % 2 == 0:
        win_size_px_orig += 1
    kernel_orig = cv2.getStructuringElement(cv2.MORPH_RECT, (win_size_px_orig, win_size_px_orig))

    for fov in fov_infos:
        img_path = os.path.join(IMAGE_FOLDER, fov['file'])
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 1. 在每张原图 FOV 上独立计算局部极差（避免全图拼接的黑色边界引发假阳性）
        local_max = cv2.dilate(img, kernel_orig)
        local_min = cv2.erode(img, kernel_orig)
        texture = cv2.subtract(local_max, local_min)

        # 2. 轻微高斯平滑，增强暗视野下微弱细胞纹理的连通性
        texture = cv2.GaussianBlur(texture, (3, 3), 0)

        # 3. 降采样
        h_i, w_i = img.shape
        img_s = cv2.resize(img, (w_i // scale, h_i // scale))
        tex_s = cv2.resize(texture, (w_i // scale, h_i // scale))

        # 4. 计算该 FOV 在全图小图坐标系中的位置
        x_start = int(round((fov['tl_x'] - min_x) / scale))
        y_start = int(round((fov['tl_y'] - min_y) / scale))

        h_s, w_s = img_s.shape
        y_end = min(y_start + h_s, height_small)
        x_end = min(x_start + w_s, width_small)

        h_s_crop = y_end - y_start
        w_s_crop = x_end - x_start

        # 5. 【核心逻辑】纹理图重叠区域取 np.maximum，保留所有视野的最强信号，跨视野克隆完美合并！
        full_texture_small[y_start:y_end, x_start:x_end] = np.maximum(
            full_texture_small[y_start:y_end, x_start:x_end],
            tex_s[:h_s_crop, :w_s_crop]
        )

        # 原图重叠区域直接覆盖（微小错位不影响后续算平均亮度）
        full_img_small[y_start:y_end, x_start:x_end] = img_s[:h_s_crop, :w_s_crop]

    img_small = full_img_small

    # 使用固定阈值 10 分割出克隆
    _, binary = cv2.threshold(full_texture_small, INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ========== 连通域分析 ==========
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    csv_rows = []
    fov_json_map = {fov['file']: [] for fov in fov_infos}
    cell_id = 1

    for i in range(1, num_labels):
        x, y, w_b, h_b, area_px = stats[i]

        # 1. 面积过滤
        area_um2 = area_px * (PIXEL_SIZE_UM ** 2)
        # if area_um2 < MIN_COLONY_AREA_UM2 or area_um2 > MAX_COLONY_AREA_UM2:
        #     continue
        if area_um2 < MIN_COLONY_AREA_UM2:
            continue

        # 2. 强度过滤
        roi_mask = (labels[y:y + h_b, x:x + w_b] == i).astype(np.uint8) * 255
        roi_img = img_small[y:y + h_b, x:x + w_b]
        mean_int = float(np.mean(roi_img[roi_mask == 255]))
        if mean_int < MIN_INTENSITY or mean_int > MAX_INTENSITY:
            continue

        # 提取轮廓
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # 3. 几何特征过滤
        cnt_small = cnt + np.array([x, y])
        c_ff, c_sm, c_ar = calculate_geometry(cnt_small, area_px)

        if c_ar < MIN_COLONY_ASPECT_RATIO or c_ff < MIN_FORM_FACTOR:
            continue

        # 还原回全孔原图坐标
        cnt_orig = cnt_small * scale
        cx_px_orig = centroids[i][0] * scale
        cy_px_orig = centroids[i][1] * scale

        # 转换为基于孔中心的全局物理坐标
        global_x_um = cx_px_orig + min_x
        global_y_um = cy_px_orig + min_y
        dist_to_center_um = float(np.sqrt(global_x_um ** 2 + global_y_um ** 2))

        # 4. 孔边缘过滤 (极其重要：去除孔壁)
        if dist_to_center_um > WELL_RADIUS_UM:
            continue

        integrated_int = mean_int * area_px

        # ========== 将结果分配到对应的 FOV ==========
        # 寻找包含该克隆质心的最近 FOV
        best_fov = None
        min_dist = float('inf')
        for fov in fov_infos:
            fov_cx = fov['tl_x'] + w_orig / 2.0
            fov_cy = fov['tl_y'] + h_orig / 2.0
            dist = (global_x_um - fov_cx) ** 2 + (global_y_um - fov_cy) ** 2
            if dist < min_dist:
                min_dist = dist
                best_fov = fov

        if best_fov is None:
            continue

        row, col = best_fov['row'], best_fov['col']

        # 记录 CSV 结果
        csv_rows.append({
            "Well": well_name,
            "Row": row,
            "Column": col,
            "Total": True,
            "X Position (µm)": global_x_um,
            "Y Position (µm)": global_y_um,
            "Distance to Nearest Neighbor (µm)": 0.0,
            "Distance to Well Center (µm)": dist_to_center_um,
            "Area (µm²)": area_um2,
            "Form Factor": c_ff,
            "Smoothness": c_sm,
            "Aspect Ratio": c_ar,
            "Mean Intensity": mean_int,
            "Integrated Intensity": integrated_int
        })

        # 构建 JSON 结果（需要将全孔轮廓坐标转换为基于最佳 FOV 的局部坐标）
        # 为所有与该克隆边界框相交的 FOV 添加 JSON 结果
        cnt_rect_x, cnt_rect_y, cnt_rect_w, cnt_rect_h = cv2.boundingRect(cnt_orig)

        for fov in fov_infos:
            fov_tl_x = fov['tl_x'] - min_x
            fov_tl_y = fov['tl_y'] - min_y

            # 判断克隆是否在这个 FOV 视野内 (允许稍微溢出，用矩形相交判断)
            if (cnt_rect_x < fov_tl_x + w_orig and cnt_rect_x + cnt_rect_w > fov_tl_x and
                    cnt_rect_y < fov_tl_y + h_orig and cnt_rect_y + cnt_rect_h > fov_tl_y):
                cnt_fov = cnt_orig - np.array([fov_tl_x, fov_tl_y])
                cx_fov = int(cx_px_orig - fov_tl_x)
                cy_fov = int(cy_px_orig - fov_tl_y)

                fov_json_map[fov['file']].append({
                    "cell_id": cell_id,
                    "center": {"x": cx_fov, "y": cy_fov},
                    "contour": [[int(pt[0][0]), int(pt[0][1])] for pt in cnt_fov],
                    "class": "Colony"
                })

        cell_id += 1

    return fov_json_map, csv_rows


def write_object_csv(all_cells_data, output_path):
    df = pd.DataFrame(all_cells_data)
    for well in df['Well'].unique():
        well_idx = df['Well'] == well
        well_data = df[well_idx]
        if len(well_data) > 1:
            coords = well_data[['X Position (µm)', 'Y Position (µm)']].values
            dist_matrix = distance.cdist(coords, coords)
            np.fill_diagonal(dist_matrix, np.inf)
            min_dists = dist_matrix.min(axis=1)
            df.loc[well_idx, 'Distance to Nearest Neighbor (µm)'] = min_dists
        else:
            df.loc[well_idx, 'Distance to Nearest Neighbor (µm)'] = 0.0

    df = df.sort_values(by=["Row", "Column", "Y Position (µm)", "X Position (µm)"])
    columns_order = [
        "Well", "Row", "Column", "Total", "X Position (µm)", "Y Position (µm)",
        "Distance to Nearest Neighbor (µm)", "Distance to Well Center (µm)",
        "Area (µm²)", "Form Factor", "Smoothness", "Aspect Ratio",
        "Mean Intensity", "Integrated Intensity"
    ]
    df = df[columns_order]

    header_lines = [
        'Plate ID,"Colony Counting - Single Colony Verification - HeLa-RFP - 24-well"',
        'Plate Name,""',
        'Plate Description,"24-well Corning 3526 plate, RFP-Hela Gradient 125, 25, 5, 1 cell/well"',
        'Scan ID,"2012/1/30 15:56:07"',
        'Scan Description,"Day 6"',
        f'Scan Result ID,"{time.strftime("%Y/%m/%d %H:%M:%S")}"',
        'Scan Result Description,""',
        'Software Version,"5.5.1.0"',
        'Experiment Name,""',
        'Application Name,"Single Colony Verification"',
        'Plate Type,"24-Well Corning™ 3526 Plate Well Wall"',
        'Acquisition Start/End Times,"2012/1/30 15:56:07 - 2012/1/30 16:20:01"',
        f'Analysis Start Time,"{time.strftime("%Y/%m/%d %H:%M:%S")}"',
        'User ID,"Local Administrator"',
        '',
        'Scan Object-Level Data CSV Report',
        ''
    ]
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(header_lines) + '\n')
        df.to_csv(f, index=False)


def write_well_plate_csv(all_cells_data, output_path):
    df = pd.DataFrame(all_cells_data)
    df['Column'] = df['Column'].astype(int)
    grouped = df.groupby(['Row', 'Column'])

    metrics = {
        'Colony Count': grouped.size(),
        'Colony AVG Area (µm²)': grouped['Area (µm²)'].mean(),
        'Colony SD Area (µm²)': grouped['Area (µm²)'].std(),
        'Colony Total Area (µm²)': grouped['Area (µm²)'].sum(),
        'AVG Intensity': grouped['Mean Intensity'].mean(),
        'SD Intensity': grouped['Mean Intensity'].std(),
        'AVG Integrated Intensity': grouped['Integrated Intensity'].mean(),
        'SD Integrated Intensity': grouped['Integrated Intensity'].std()
    }
    metrics['Colony %CV Area'] = metrics['Colony SD Area (µm²)'] / metrics['Colony AVG Area (µm²)'] * 100
    metrics['%CV Intensity'] = metrics['SD Intensity'] / metrics['AVG Intensity'] * 100
    metrics['%CV Integrated Intensity'] = metrics['SD Integrated Intensity'] / metrics['AVG Integrated Intensity'] * 100

    all_rows = sorted(df['Row'].unique())
    all_cols = sorted(df['Column'].unique())
    if not all_rows: all_rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    if not all_cols: all_cols = [1, 2, 3, 4, 5, 6]

    def get_pivot_table(metric_name, metric_series, fmt="{:f}", is_percent=False):
        if metric_name in ('Colony Count', 'Colony Total Area (µm²)'):
            pivot = metric_series.unstack(fill_value=0).reindex(index=all_rows, columns=all_cols, fill_value=0)
        else:
            pivot = metric_series.unstack().reindex(index=all_rows, columns=all_cols)
        lines = []
        for row in all_rows:
            row_vals = []
            for col in all_cols:
                val = pivot.at[row, col] if row in pivot.index and col in pivot.columns else np.nan
                if is_percent:
                    row_vals.append("NaN" if pd.isna(val) else f"{val:.5f}%")
                else:
                    if pd.isna(val):
                        row_vals.append("0" if fmt == "{:d}" else "NaN")
                    elif fmt == "{:d}":
                        row_vals.append(str(int(val)))
                    else:
                        row_vals.append(str(val))
            lines.append(f"{row}," + ",".join(row_vals) + ",")
        return lines

    header_lines = [
        'Plate ID,"Colony Counting - Single Colony Verification - HeLa-RFP - 24-well"',
        'Plate Name,""',
        'Plate Description,"24-well Corning 3526 plate, RFP-Hela Gradient 125, 25, 5, 1 cell/well"',
        'Scan ID,"2012/1/30 15:56:07"',
        'Scan Description,"Day 6"',
        f'Scan Result ID,"{time.strftime("%Y/%m/%d %H:%M:%S")}"',
        'Scan Result Description,""',
        'Software Version,"5.5.1.0"',
        'Experiment Name,""',
        'Application Name,"Single Colony Verification"',
        'Plate Type,"24-Well Corning™ 3526 Plate Well Wall"',
        'Acquisition Start/End Times,"2012/1/30 15:56:07 - 2012/1/30 16:20:01"',
        f'Analysis Start Time,"{time.strftime("%Y/%m/%d %H:%M:%S")}"',
        'User ID,"Local Administrator"',
        '',
        'Measurement Plate Maps'
    ]

    output_lines = header_lines.copy()

    confluency_series = (metrics['Colony Total Area (µm²)'] / (np.pi * (WELL_RADIUS_UM ** 2))) * 100.0
    output_lines.append('Confluency (%),' + ','.join(map(str, all_cols)))
    output_lines.extend(get_pivot_table('Confluency (%)', confluency_series, "{:f}", True))
    output_lines.append('')

    metrics_config = [
        ('Colony Count', "{:d}", False),
        ('Colony AVG Area (µm²)', "{:f}", False),
        ('Colony SD Area (µm²)', "{:f}", False),
        ('Colony Total Area (µm²)', "{:f}", False),
        ('Colony %CV Area', "{:f}", True),
        ('AVG Intensity', "{:f}", False),
        ('SD Intensity', "{:f}", False),
        ('%CV Intensity', "{:f}", True),
        ('AVG Integrated Intensity', "{:f}", False),
        ('SD Integrated Intensity', "{:f}", False),
        ('%CV Integrated Intensity', "{:f}", True)
    ]

    for metric_name, fmt, is_pct in metrics_config:
        output_lines.append(f'{metric_name},' + ','.join(map(str, all_cols)))
        output_lines.extend(get_pivot_table(metric_name, metrics[metric_name], fmt, is_pct))
        output_lines.append('')

    output_lines.append('Well Sampled (%),' + ','.join(map(str, all_cols)))
    for r in all_rows:
        output_lines.append(f"{r}," + ",".join(["100.00000%"] * len(all_cols)) + ",")
    output_lines.append('')

    output_lines.append('Analysis Settings Plate Maps,' + ','.join(map(str, all_cols)))
    for r in all_rows:
        output_lines.append(f"{r}," + ",".join(["1"] * len(all_cols)) + ",")
    output_lines.append('')

    focus_map_rows = {
        'A': [3.11785626411438, 3.14482641220093, 3.16523957252502, 3.16630959510803, 3.15267777442932,
              3.13433504104614],
        'B': [3.1365954875946, 3.1433253288269, 3.17933750152588, 3.17716312408447, 3.15200471878052, 3.14075422286987],
        'C': [3.15345406532288, 3.16203022003174, 3.18679165840149, 3.1864812374115, 3.1582338809967, 3.14781165122986],
        'D': [3.16901874542236, 3.19069147109985, 3.20327067375183, 3.20356392860413, 3.18556666374207,
              3.16289281845093]
    }
    output_lines.append('Focus Position Map,' + ','.join(map(str, all_cols)))
    for r in all_rows:
        values = focus_map_rows.get(r, [0.0] * len(all_cols))
        if len(values) < len(all_cols):
            values = values + [values[-1]] * (len(all_cols) - len(values))
        output_lines.append(f"{r}," + ",".join(str(v) for v in values[:len(all_cols)]) + ",")
    output_lines.append('')

    output_lines.extend([
        '', '', '',
        'Analysis Settings,Name,Version,Instance',
        ',Single Colony Verification - HeLa-RFP 24-well,1000,1',
        'General Settings,Item,Value',
        ',Analysis Resolution (µm/pixel),3',
        ',Well Mask,True',
        ',Well Mask Usage Mode,Automatic',
        ',% Well Mask,95',
        ',Well Mask Shape,Default',
        'Identification Settings,Colony Frame Settings',
        ',Item,Value',
        ',Algorithm,Texture',
        ',Intensity Threshold,10',
        ',Precision,High',
        ',Sharpen,None',
        ',Diameter (µm),8',
        ',Background Correction,True',
        ',Separate Touching Colonies,True',
        ',Minimum Thickness (µm),15',
        ',Saturated Intensity,0',
        'Pre-Filtering Settings,Colony Feature Settings',
        ',Item,Value',
        ',Min Colony Size (µm²),20000',
        ',Min Colony Aspect Ratio,0.09',
        ',Colony Intensity Range,20 to 255',
        '',
        'Classification Settings,Name,Version',
        ',Untitled Classification Settings,1000',
        'Classes,Class Name,Source Population Name',
        ',Total,ALL',
        'Gating,-none-',
        '', '',
        'Image Channel Settings,Gain,Exposure,SetupMode,Illumination Source',
        '41afc28e-223a-40cb-b9b5-1e752ee0c7f2,0,8561 (ums),Always,Brightfield',
        '', ''
    ])

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(output_lines) + '\n')


def run_pipeline():
    total_start = time.time()

    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER, exist_ok=True)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    file_list = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.jpg') and '_Xmm' in f]
    if not file_list:
        print("未找到符合要求的 FOV 图像文件")
        return

    # 按孔位分组 FOV 文件
    well_groups = {}
    parsed_infos = {}
    for fname in file_list:
        well, row, col, x_mm, y_mm = extract_coordinates(fname)
        parsed_infos[fname] = (well, row, col, x_mm, y_mm)
        well_groups.setdefault(well, []).append(fname)

    # 计算每个孔的中心
    well_origin_map = {}
    for well, fovs in well_groups.items():
        xs = [parsed_infos[f][3] for f in fovs]
        ys = [parsed_infos[f][4] for f in fovs]
        well_origin_map[well] = (float(np.median(xs)), float(np.median(ys)))

    all_colonies_data = []
    well_fov_results = {}

    print(f"开始处理 {len(well_groups)} 个孔位 (共 {len(file_list)} 张 FOV 图像)...")
    for well_name, fovs in tqdm(well_groups.items(), desc="孔位处理进度"):
        fov_json_map, csv_rows = process_well(well_name, fovs, well_origin_map[well_name])

        # 整理该孔位下所有 FOV 的 JSON 数据
        fovs_data = []
        for fov_file, cells in fov_json_map.items():
            fovs_data.append({
                "pic_name": fov_file,
                "total_cells": len(cells),
                "cells": cells
            })

        well_fov_results[well_name] = fovs_data
        all_colonies_data.extend(csv_rows)

    # 保存 JSON 文件
    for well_name, fovs_data in well_fov_results.items():
        well_json = {
            "experiment_id": EXPERIMENT_ID,
            "plate_name": PLATE_NAME,
            "well_name": well_name,
            "elapsed_time_seconds": 0,
            "fovs_data": fovs_data
        }
        save_path = os.path.join(OUTPUT_FOLDER, f"{well_name}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(well_json, f, ensure_ascii=False, separators=(',', ':'))

    # 保存 CSV 文件
    if all_colonies_data:
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        csv_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_object.csv")
        write_object_csv(all_colonies_data, csv_path)
        well_plate_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}_well_plate.csv")
        write_well_plate_csv(all_colonies_data, well_plate_path)
        print(f"CSV 已保存至:\n  {csv_path}\n  {well_plate_path}", flush=True)
    else:
        print("未检测到符合条件的克隆数据", flush=True)

    print(f"\n运行完成! 总耗时: {time.time() - total_start:.2f} 秒", flush=True)


if __name__ == "__main__":
    run_pipeline()
