# -*- coding: utf-8 -*-
"""
nnUNet 数据桥接脚本：.nii.gz → .mat + 量化特征提取

功能：
1. 读取 nnUNet 预测 mask (.nii.gz) 和原始 OCT 图像 (.nii.gz)
2. 输出 .mat 文件（Data_fill, Data_label, data_scatt），兼容现有量化流程
3. 直接提取形态学特征（体积、表面积、囊腔、长短轴、壁厚、球形度、散射统计）到 Excel

输入结构：
    nnUNet_Data/
    ├── FXN_0701/                    (原始图: organoid_001_0000.nii.gz)
    ├── FXN_0701_prediction/         (预测mask: organoid_001.nii.gz)
    ├── FXN_0703/
    └── FXN_0703_prediction/

输出结构：
    nnUNet_Data/
    ├── FXN_0701_mat/               (转换后的 .mat 文件)
    │   ├── seg_fill/               (Data_fill)
    │   ├── seg_label/              (Data_label)
    │   └── scatt_mat/              (data_scatt)
    └── FXN_0701_measure/           (量化 Excel)
        └── measure_excel/

作者: Claude
日期: 2026-06-22
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import savemat, loadmat
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_fill_holes, ball, erosion, dilation, remove_small_objects
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ======================== 配置区域 ========================
# 输入根目录
INPUT_ROOT = "nnUNet_Data"

# 需要处理的时间点文件夹（原始图 + 预测成对出现）
DATE_PAIRS = [
    ("FXN_0701", "FXN_0701_prediction"),
    ("FXN_0703", "FXN_0703_prediction"),
]

# 输出根目录（会建在 INPUT_ROOT 同级或内部）
OUTPUT_ROOT = os.path.join(INPUT_ROOT, "nnUNet_Converted")

# 最小 organoid 体积阈值（像素数），过滤噪声
MIN_SIZE = 50

# 并行进程数
N_WORKERS = 8

# 是否对 mask 做 fill_holes（填充空洞，计算体积时建议 True）
FILL_HOLES = True

# ======================== I/O 工具 ========================

def load_nii(path):
    """加载 .nii.gz 并返回 numpy array（float32）"""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    return data


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_sample_id(pred_name):
    """
    从预测文件名解析 sample ID
    例: organoid_001.nii.gz -> organoid_001
    """
    return pred_name.replace(".nii.gz", "")


# ======================== 核心量化函数 ========================

def compute_cavity_features(mask_binary, spacing=(1.0, 1.0, 1.0)):
    """
    计算囊腔相关特征
    输入: 二值掩膜 (Z, Y, X)
    返回: cavity_volume, cavity_num, wall_thickness_proxy
    """
    # 简单方法：用形态学操作估计"内部空腔"
    # 对 mask 做 erosion，然后原mask - eroded = 边界
    # 更精确的方法：若 organoid 是空心球，可通过距离变换判断
    # 这里采用一个实用的 proxy：
    #   cavity ≈ mask 中没有被 fill_holes 填补的区域（如果输入是 fill 过的，此方法不适用）
    # 建议：如果 nnUNet 分割的是 organoid 实体，cavity 需通过原始 OCT 强度阈值分割得到

    # 简化版：假设 mask 是 organoid 外轮廓，内部用 fill_holes 后的差值估计空腔
    filled = binary_fill_holes(mask_binary)
    cavity = filled & (~mask_binary)

    # 去除微小噪声
    cavity = remove_small_objects(cavity, min_size=MIN_SIZE)

    cavity_labels = label(cavity)
    cavity_num = cavity_labels.max()
    cavity_volume = int(np.sum(cavity))

    # 壁厚 proxy：体积 / 表面积 的近似
    # 这里返回 0，后续在 regionprops 中统一计算
    return cavity_volume, cavity_num, 0


def extract_organoid_features(mask_instance, intensity_vol, sample_id, spacing=(1.0, 1.0, 1.0)):
    """
    从实例标签图和强度图中提取所有类器官的特征
    输入:
        mask_instance: (Z, Y, X) int, 0=背景, 1,2,3...=各organoid
        intensity_vol: (Z, Y, X) float32, OCT 原始强度/散射
        sample_id: 样本名，如 "organoid_001"
    返回:
        DataFrame: 每行一个 organoid
    """
    props = regionprops(mask_instance, intensity_image=intensity_vol, spacing=spacing)

    records = []
    for i, p in enumerate(props):
        # 基础形态
        volume = int(p.area)  # 像素数（若 spacing≠1，需乘 voxel_size）
        volume_fill = volume  # 若已 fill_holes，二者相同
        surface_area = float(p.area_bbox)  # proxy，3D surface area 需要更精确计算
        # 更精确的表面积：使用 marching cubes 或凸包，regionprops 的 area 在 3D 就是体积
        # 这里用等效直径推算表面积 proxy
        equiv_diameter = p.equivalent_diameter
        surface_proxy = 4.0 * np.pi * (equiv_diameter / 2.0) ** 2

        # 轴长
        long_axis = float(p.axis_major_length) if hasattr(p, 'axis_major_length') else 0
        short_axis = float(p.axis_minor_length) if hasattr(p, 'axis_minor_length') else 0

        # 球形度 proxy
        if long_axis > 0:
            sphericity = short_axis / long_axis  # 简化为长短轴比
        else:
            sphericity = 0

        # 壁厚 proxy（基于体积和表面积）
        if surface_proxy > 0:
            wall_thickness = volume / surface_proxy
        else:
            wall_thickness = 0

        # 囊腔（简化：假设空心，用 bounding box 体积 - 实体体积 proxy）
        # 更精确需要原始 OCT 强度阈值分割内部空腔
        bbox_vol = np.prod([b - a for a, b in zip(p.bbox[:3], p.bbox[3:])])
        cavity_volume = max(0, bbox_vol - volume)
        cavity_num = 1 if cavity_volume > MIN_SIZE else 0

        # 散射统计（从 intensity_vol 中该 organoid 区域统计）
        coords = p.coords  # (N, 3)
        if len(coords) > 0 and intensity_vol is not None:
            vals = intensity_vol[coords[:, 0], coords[:, 1], coords[:, 2]]
            scatt_mean = float(np.mean(vals))
            scatt_std = float(np.std(vals))
        else:
            scatt_mean = 0.0
            scatt_std = 0.0

        records.append({
            'Index': f"{sample_id}_{i + 1}",
            'Organoids_Volume': volume,
            'Organoids_Volume_Fill': volume_fill,
            'Organoids_Surface': round(surface_proxy, 2),
            'Cavity_Volume': cavity_volume,
            'CavityNum': cavity_num,
            'LongAxis': round(long_axis, 2),
            'ShortAxis': round(short_axis, 2),
            'Wall_Thickness': round(wall_thickness, 4),
            'Sphericity': round(sphericity, 4),
            'Scatt_Mean': round(scatt_mean, 2),
            'Scatt_STD': round(scatt_std, 2),
        })

    return pd.DataFrame(records)


# ======================== 单样本处理 ========================

def process_one_sample(args):
    """
    处理单个样本（一对 .nii.gz）
    """
    sample_id, pred_path, img_path, out_dirs = args
    try:
        # 1. 加载数据
        mask_pred = load_nii(pred_path)  # nnUNet 预测 (0=bg, 1=organoid)
        if img_path and os.path.exists(img_path):
            intensity_vol = load_nii(img_path)
        else:
            intensity_vol = None

        # 确保 mask 是二值
        mask_binary = (mask_pred > 0).astype(np.uint8)

        # 2. 可选：填充空洞（得到 Data_fill）
        if FILL_HOLES:
            # 逐层或整体 fill holes
            mask_fill = binary_fill_holes(mask_binary).astype(np.uint8)
        else:
            mask_fill = mask_binary

        # 3. 连通域标记（得到 Data_label）
        # 如果 nnUNet 已经是单个 organoid，label 后也是 1
        mask_label = label(mask_fill)
        # 过滤小物体
        mask_label = remove_small_objects(mask_label, min_size=MIN_SIZE)
        # 重新编号
        mask_label = label(mask_label > 0)

        # 4. 保存 .mat 文件（兼容现有流程）
        # Data_fill: 二值填充掩膜
        savemat(
            os.path.join(out_dirs['seg_fill'], f"{sample_id}_fill.mat"),
            {'Data_fill': mask_fill.astype(np.uint8)},
            do_compression=True
        )

        # Data_label: 实例标签
        savemat(
            os.path.join(out_dirs['seg_label'], f"{sample_id}_label.mat"),
            {'Data_label': mask_label.astype(np.uint16)},
            do_compression=True
        )

        # data_scatt: OCT 强度作为散射 proxy
        if intensity_vol is not None:
            savemat(
                os.path.join(out_dirs['scatt_mat'], f"{sample_id}_scatt.mat"),
                {'data_scatt': intensity_vol.astype(np.float32)},
                do_compression=True
            )

        # 5. 提取量化特征
        df_features = extract_organoid_features(
            mask_label, intensity_vol, sample_id
        )

        # 保存单文件 Excel
        excel_path = os.path.join(out_dirs['measure_excel'], f"{sample_id}.xlsx")
        df_features.to_excel(excel_path, index=False)

        return f"✅ {sample_id} | organoids: {len(df_features)}"

    except Exception as e:
        return f"❌ {sample_id} | Error: {e}"


# ======================== 主程序 ========================

def main():
    print("=" * 60)
    print("nnUNet 数据桥接与量化脚本")
    print("=" * 60)

    all_tasks = []

    for img_folder, pred_folder in DATE_PAIRS:
        img_dir = os.path.join(INPUT_ROOT, img_folder)
        pred_dir = os.path.join(INPUT_ROOT, pred_folder)

        if not os.path.exists(pred_dir):
            print(f"⚠️ 跳过: 预测目录不存在 {pred_dir}")
            continue

        # 创建输出目录结构
        out_base = os.path.join(OUTPUT_ROOT, img_folder + "_converted")
        out_dirs = {
            'seg_fill': os.path.join(out_base, "seg_fill"),
            'seg_label': os.path.join(out_base, "seg_label"),
            'scatt_mat': os.path.join(out_base, "scatt_mat"),
            'measure_excel': os.path.join(out_base, "measure_excel"),
        }
        for d in out_dirs.values():
            ensure_dir(d)

        # 收集预测文件
        pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
        print(f"\n📂 {img_folder}: 发现 {len(pred_files)} 个预测文件")

        for pred_path in pred_files:
            pred_name = os.path.basename(pred_path)
            sample_id = parse_sample_id(pred_name)

            # 寻找对应的原始图像
            # 命名规则: organoid_001.nii.gz -> organoid_001_0000.nii.gz
            img_path = os.path.join(img_dir, f"{sample_id}_0000.nii.gz")
            if not os.path.exists(img_path):
                img_path = None  # 允许缺失原始图，此时只做 mask 转换

            all_tasks.append((sample_id, pred_path, img_path, out_dirs))

    print(f"\n🚀 共收集 {len(all_tasks)} 个任务，开始并行处理...")

    # 并行处理
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_one_sample, task): task[0] for task in all_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            result = future.result()
            if "❌" in result:
                print(result)

    print("\n" + "=" * 60)
    print("✅ 全部完成！输出目录:")
    print(f"   {OUTPUT_ROOT}")
    print("\n输出结构:")
    print("   ├── seg_fill/      -> Data_fill (.mat)")
    print("   ├── seg_label/     -> Data_label (.mat)")
    print("   ├── scatt_mat/     -> data_scatt (.mat)")
    print("   └── measure_excel/ -> 单样本量化表 (.xlsx)")
    print("=" * 60)

    # 可选：汇总所有 measure_excel 为一张大表
    print("\n📊 正在汇总所有量化表...")
    all_dfs = []
    for img_folder, pred_folder in DATE_PAIRS:
        out_base = os.path.join(OUTPUT_ROOT, img_folder + "_converted")
        measure_dir = os.path.join(out_base, "measure_excel")
        if not os.path.exists(measure_dir):
            continue
        excel_files = glob.glob(os.path.join(measure_dir, "*.xlsx"))
        for f in excel_files:
            try:
                df = pd.read_excel(f)
                all_dfs.append(df)
            except Exception as e:
                print(f"⚠️ 读取失败 {f}: {e}")

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        summary_path = os.path.join(OUTPUT_ROOT, "nnUNet_All_Measure.xlsx")
        df_all.to_excel(summary_path, index=False)
        print(f"✅ 汇总表已保存: {summary_path} (共 {len(df_all)} 个 organoids)")
    else:
        print("⚠️ 未找到可汇总的量化表")


if __name__ == "__main__":
    main()
