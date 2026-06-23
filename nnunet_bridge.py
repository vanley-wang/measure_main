# -*- coding: utf-8 -*-
"""
nnUNet 数据桥接脚本：.nii.gz → .mat（兼容现有量化流程）

功能：
1. 读取 nnUNet 整孔预测 mask (.nii.gz) 和原始 OCT 图像 (.nii.gz)
2. 对整孔 mask 做连通域标记，得到每个类器官的实例标签
3. 输出 .mat 文件（Data_fill, Data_label, data_scatt），命名映射回孔板编号
4. 输出目录结构完全匹配现有代码的输入路径

整孔映射规则（0701 & 0703 相同）：
    organoid_001 → B2, 002 → B3, ..., 010 → B11,
    011 → C2, ..., 020 → C11,
    021 → D11, 022 → E11,
    023 → F2, ..., 032 → F11

输出结构：
    Data/nnUNet_FXN/FXN_20230701/
        ├── seg_fill/       (Data_fill: 二值填充掩膜)
        ├── seg_label/      (Data_label: 实例标签，每个organoid一个ID)
        ├── scatt_mat/      (data_scatt: OCT原始强度/散射)
        └── image/          (可选: PNG切片，供seg_amira.py使用)

复用方式：
    1. 运行本脚本完成转换
    2. 修改 scatt.py / roughness.py / cluster.py 等的路径指向上述目录
    3. 直接执行现有脚本完成量化
"""

import os
import glob
import re
import numpy as np
import nibabel as nib
from scipy.io import savemat
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ======================== 配置区域 ========================

# 输入根目录
INPUT_ROOT = "nnUNet_Data"

# 输出根目录（会和现有代码结构一致）
# 输出示例: Data/nnUNet_FXN/FXN_20230701/seg_fill/...
OUTPUT_ROOT = "Data/nnUNet_FXN"

# 需要处理的时间点（原始图文件夹, 预测文件夹, 日期后缀）
DATE_CONFIGS = [
    ("FXN_0701", "FXN_0701_prediction", "0701"),
    ("FXN_0703", "FXN_0703_prediction", "0703"),
]

# organoid 序号 → Well ID 映射（0701 & 0703 共用同一映射）
ID_MAP = {
    "001": "B2", "002": "B3", "003": "B4", "004": "B5", "005": "B6",
    "006": "B7", "007": "B8", "008": "B9", "009": "B10", "010": "B11",
    "011": "C2", "012": "C3", "013": "C4", "014": "C5", "015": "C6",
    "016": "C7", "017": "C8", "018": "C9", "019": "C10", "020": "C11",
    "021": "D11", "022": "E11",
    "023": "F2", "024": "F3", "025": "F4", "026": "F5", "027": "F6",
    "028": "F7", "029": "F8", "030": "F9", "031": "F10", "032": "F11",
}

# 最小 organoid 体积（像素数），过滤噪声连通域
MIN_SIZE = 50

# 并行进程数
N_WORKERS = 2

# 是否同时导出 PNG 切片（供 seg_amira.py / tr2png 等使用）
EXPORT_PNG = False

# PNG 导出时的并行线程数（仅当 EXPORT_PNG=True 时生效）
PNG_WORKERS = 4

# ======================== 工具函数 ========================


def load_nii(path):
    """加载 .nii.gz 并返回 numpy array（float32）"""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    return data


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def crop_to_intersection(vol_a, vol_b):
    """将两个 3D 矩阵裁剪到公共最小尺寸（左上对齐）"""
    if vol_a.shape == vol_b.shape:
        return vol_a, vol_b
    min_z = min(vol_a.shape[0], vol_b.shape[0])
    min_y = min(vol_a.shape[1], vol_b.shape[1])
    min_x = min(vol_a.shape[2], vol_b.shape[2])
    return vol_a[:min_z, :min_y, :min_x], vol_b[:min_z, :min_y, :min_x]


def extract_numeric_id(filename):
    """从 organoid_001.nii.gz 中提取 '001'"""
    m = re.search(r"organoid_(\d+)", filename)
    return m.group(1) if m else None


def map_to_well_id(num_id, date_suffix):
    """映射为 B2_0701 格式"""
    well = ID_MAP.get(num_id)
    if well is None:
        return None
    return f"{well}_{date_suffix}"


def save_png_stack(volume, out_dir, prefix):
    """将 3D volume 保存为 PNG 切片序列"""
    ensure_dir(out_dir)
    depth = volume.shape[0]
    for z in range(depth):
        slice_img = volume[z, :, :]
        # 归一化到 0-255
        mn, mx = slice_img.min(), slice_img.max()
        if mx > mn:
            img_uint8 = ((slice_img - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img_uint8 = np.zeros_like(slice_img, dtype=np.uint8)
        fname = os.path.join(out_dir, f"{prefix}_{z + 1:03d}.png")
        Image.fromarray(img_uint8).save(fname)


# ======================== 单样本处理 ========================

def process_one_sample(args):
    """
    处理单个样本（一对 .nii.gz）
    返回: (status_msg, well_id)
    """
    pred_path, img_path, out_dirs, date_suffix = args
    pred_name = os.path.basename(pred_path)
    num_id = extract_numeric_id(pred_name)
    well_id = map_to_well_id(num_id, date_suffix)

    if well_id is None:
        return f"❌ 跳过 {pred_name}: 无 Well ID 映射", None

    try:
        # 1. 加载 mask（nnUNet 预测）
        mask_pred = load_nii(pred_path)
        mask_binary = (mask_pred > 0).astype(np.uint8)

        # 2. 加载原始 OCT 图像（作为散射数据）
        if img_path and os.path.exists(img_path):
            intensity_vol = load_nii(img_path)
            # 裁剪到与 mask 一致（防止 nnUNet 预处理导致尺寸不同）
            mask_binary, intensity_vol = crop_to_intersection(mask_binary, intensity_vol)
        else:
            intensity_vol = None

        # 3. 填充空洞 → Data_fill
        # 逐 slice 做 fill holes（3D binary_fill_holes 有时对薄壁不友好，这里用整体）
        mask_fill = binary_fill_holes(mask_binary).astype(np.uint8)

        # 4. 连通域标记 → Data_label
        mask_label = label(mask_fill)
        # 过滤小物体
        mask_label = remove_small_objects(mask_label, min_size=MIN_SIZE)
        # 重新编号（从1开始）
        mask_label = label(mask_label > 0)
        n_objects = mask_label.max()

        # 5. 保存 .mat（兼容现有代码变量名）
        savemat(
            os.path.join(out_dirs["seg_fill"], f"{well_id}_fill.mat"),
            {"Data_fill": mask_fill},
            do_compression=True,
        )
        savemat(
            os.path.join(out_dirs["seg_label"], f"{well_id}_label.mat"),
            {"Data_label": mask_label.astype(np.uint16)},
            do_compression=True,
        )
        if intensity_vol is not None:
            savemat(
                os.path.join(out_dirs["scatt_mat"], f"{well_id}_scatt.mat"),
                {"data_scatt": intensity_vol.astype(np.float32)},
                do_compression=True,
            )

        # 6. 可选：导出 PNG 切片
        if EXPORT_PNG and intensity_vol is not None:
            png_subdir = os.path.join(out_dirs["image"], well_id)
            save_png_stack(intensity_vol, png_subdir, well_id)

        return f"✅ {well_id} ({pred_name}) | organoids: {n_objects}", well_id

    except Exception as e:
        return f"❌ {well_id} ({pred_name}) | Error: {e}", well_id


# ======================== 主程序 ========================


def main():
    print("=" * 60)
    print("nnUNet → .mat 桥接脚本（复用现有量化流程）")
    print("=" * 60)

    all_tasks = []

    for img_folder, pred_folder, date_suffix in DATE_CONFIGS:
        img_dir = os.path.join(INPUT_ROOT, img_folder)
        pred_dir = os.path.join(INPUT_ROOT, pred_folder)

        if not os.path.isdir(pred_dir):
            print(f"⚠️ 跳过: 预测目录不存在 {pred_dir}")
            continue

        # 创建输出目录（与现有代码路径风格一致）
        out_base = os.path.join(OUTPUT_ROOT, f"FXN_{date_suffix}")
        out_dirs = {
            "seg_fill": os.path.join(out_base, "seg_fill"),
            "seg_label": os.path.join(out_base, "seg_label"),
            "scatt_mat": os.path.join(out_base, "scatt_mat"),
        }
        if EXPORT_PNG:
            out_dirs["image"] = os.path.join(out_base, "image")

        for d in out_dirs.values():
            ensure_dir(d)

        # 收集预测文件
        pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
        print(f"\n📂 {img_folder} ({date_suffix}): 发现 {len(pred_files)} 个预测文件")

        for pred_path in pred_files:
            pred_name = os.path.basename(pred_path)
            num_id = extract_numeric_id(pred_name)
            # 寻找对应原始图: organoid_001.nii.gz → organoid_001_0000.nii.gz
            img_name = f"organoid_{num_id}_0000.nii.gz"
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                img_path = None

            all_tasks.append((pred_path, img_path, out_dirs, date_suffix))

    print(f"\n🚀 共收集 {len(all_tasks)} 个任务，串行处理中...")

    success = 0
    failed = 0
    for task in tqdm(all_tasks, desc="转换进度"):
        msg, well_id = process_one_sample(task)
        if "❌" in msg:
            print(msg)
            failed += 1
        else:
            success += 1

    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print(f"   成功: {success} | 失败: {failed}")
    print(f"\n输出根目录: {os.path.abspath(OUTPUT_ROOT)}")
    print("\n目录结构示例:")
    print(f"   {OUTPUT_ROOT}/")
    print("   ├── FXN_0701/")
    print("   │   ├── seg_fill/     → B2_0701_fill.mat ...")
    print("   │   ├── seg_label/    → B2_0701_label.mat ...")
    print("   │   └── scatt_mat/    → B2_0701_scatt.mat ...")
    print("   └── FXN_0703/")
    print("       └── ...")
    print("\n下一步：修改现有脚本路径，直接运行量化。")
    print("=" * 60)


if __name__ == "__main__":
    main()