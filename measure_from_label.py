# -*- coding: utf-8 -*-
"""
从 seg_label (.mat) 提取基础形态特征 → measure_excel (.xlsx)
为后续 cluster.py / scatt.py / roughness.py 提供输入

输入：Data/nnUNet_FXN/FXN_20230701/seg_label/*_label.mat
输出：Data/nnUNet_FXN/FXN_20230701/measure_excel/*.xlsx

每行一个 organoid，包含：
    Index, Organoids_Volume, Organoids_Volume_Fill, Organoids_Surface,
    Cavity_Volume, CavityNum, LongAxis, ShortAxis, Wall_Thickness, Sphericity
（Scatt_Mean / Scatt_STD 占位，后续由 scatt.py 填充）
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage.measure import regionprops_table, label
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ======================== 配置 ========================

LOOPS = [
    r"Data\nnUNet_FXN\FXN_0701",
    r"Data\nnUNet_FXN\FXN_0703",
]

MIN_SIZE = 50  # 过滤小连通域（与 nnunet_bridge.py 一致）

# ======================== 主程序 ========================


def extract_features(label_path, well_id):
    """从单个 _label.mat 提取所有 organoid 的形态特征"""
    try:
        mat = loadmat(label_path)
        data_label = mat["Data_label"]
    except Exception as e:
        return None, f"加载失败 {well_id}: {e}"

    # 保险：过滤小物体后重新编号
    cleaned = remove_small_objects(data_label > 0, min_size=MIN_SIZE)
    data_label = label(cleaned)

    if data_label.max() == 0:
        return pd.DataFrame(), f"{well_id}: 无有效 organoid"

    # 提取 regionprops（3D）
    # 注意：skimage regionprops_table 会把多维属性展开为 bbox-0, bbox-1 ...
    props = regionprops_table(
        data_label,
        properties=[
            "label",
            "area",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
        ],
    )

    # bbox 需要单独提取，因为它返回多维值
    bbox_props = regionprops_table(
        data_label,
        properties=["bbox"],
    )
    # bbox 展开为 bbox-0 ~ bbox-5
    bbox_arr = np.column_stack([bbox_props[f"bbox-{i}"] for i in range(6)])

    n = len(props["label"])
    records = []
    for i in range(n):
        area = int(props["area"][i])
        equiv_d = props["equivalent_diameter"][i]
        major = props["major_axis_length"][i]
        minor = props["minor_axis_length"][i]

        # 表面积 proxy：等效球面积
        surface = 4.0 * np.pi * (equiv_d / 2.0) ** 2 if equiv_d > 0 else 0

        # 壁厚 proxy
        wall_thick = area / surface if surface > 0 else 0

        # 球形度 proxy
        sphericity = minor / (major + 1e-6) if major > 0 else 0

        # 囊腔（简化：用 bbox 体积 - 实体体积）
        bbox = bbox_arr[i]  # [min_z, min_y, min_x, max_z, max_y, max_x]
        bbox_vol = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2])
        cavity_vol = max(0, bbox_vol - area)
        cavity_num = 1 if cavity_vol > MIN_SIZE else 0

        records.append({
            "Index": f"{well_id}_{i + 1}",
            "Organoids_Volume": area,
            "Organoids_Volume_Fill": area,
            "Organoids_Surface": round(surface, 2),
            "Cavity_Volume": cavity_vol,
            "CavityNum": cavity_num,
            "LongAxis": round(major, 2),
            "ShortAxis": round(minor, 2),
            "Wall_Thickness": round(wall_thick, 4),
            "Sphericity": round(sphericity, 4),
            "Scatt_Mean": np.nan,   # 占位，scatt.py 后续填充
            "Scatt_STD": np.nan,    # 占位
        })

    return pd.DataFrame(records), None


def main():
    print("=" * 60)
    print("从 seg_label 提取形态特征 → measure_excel")
    print("=" * 60)

    total_organoids = 0

    for root in LOOPS:
        label_dir = os.path.join(root, "seg_label")
        save_dir = os.path.join(root, "measure_excel")
        os.makedirs(save_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(label_dir, "*_label.mat")))
        print(f"\n📂 {os.path.basename(root)}: {len(files)} 个 label 文件")

        for fpath in tqdm(files, desc=f"处理 {os.path.basename(root)}"):
            fname = os.path.basename(fpath)
            well_id = fname.replace("_label.mat", "")

            df, err = extract_features(fpath, well_id)
            if err:
                print(f"  ⚠️ {err}")
                continue

            if df is not None and len(df) > 0:
                df.to_excel(os.path.join(save_dir, f"{well_id}.xlsx"), index=False)
                total_organoids += len(df)

    print(f"\n✅ 全部完成！共生成 {total_organoids} 个 organoid 记录")
    print("=" * 60)


if __name__ == "__main__":
    main()
