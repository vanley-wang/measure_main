import os
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def extract_scatt_stats_fast(label_map, scatt_map):
    """提取每个 label 区域的散射系数均值和标准差（内存优化版）"""
    assert label_map.shape == scatt_map.shape
    # 用 ravel() 代替 flatten()，避免复制
    lbl = label_map.ravel().astype(np.int64)
    sc = scatt_map.ravel().astype(np.float64)

    max_lbl = int(lbl.max())
    if max_lbl == 0:
        return np.array([]), [], []

    # 用 bincount 一次性统计所有 label，避免布尔 mask 和循环
    counts = np.bincount(lbl, minlength=max_lbl + 1)[1:]
    sums = np.bincount(lbl, weights=sc, minlength=max_lbl + 1)[1:]
    sq_sums = np.bincount(lbl, weights=sc ** 2, minlength=max_lbl + 1)[1:]

    valid = counts > 0
    means = np.zeros(max_lbl, dtype=np.float64)
    stds = np.zeros(max_lbl, dtype=np.float64)
    means[valid] = sums[valid] / counts[valid]
    stds[valid] = np.sqrt(np.maximum(0, sq_sums[valid] / counts[valid] - means[valid] ** 2))

    labels = np.arange(1, max_lbl + 1)[valid]
    means = np.round(means[valid]).astype(int)
    stds = np.round(stds[valid]).astype(int)

    return labels, means.tolist(), stds.tolist()

def process_one_sample(seg_label_dir, scatt_mat_dir, output_dir, measure_dir, fname):
    try:
        basename = fname.replace('_label.mat', '')
        label_path = os.path.join(seg_label_dir, fname)
        scatt_path = os.path.join(scatt_mat_dir, f"{basename}_scatt.mat")
        output_path = os.path.join(output_dir, f"{basename}_scatt.xlsx")
        measure_path = os.path.join(measure_dir, f"{basename}.xlsx")

        # 加载标签图和散射系数图
        label_data = loadmat(label_path)['Data_label']
        # scatt_data = loadmat(scatt_path)['Data_scatt']
        scatt_mat = loadmat(scatt_path)
        
        # 兼容不同文件里的 key 大小写差异
        scatt_key = 'Data_scatt' if 'Data_scatt' in scatt_mat else 'data_scatt'
        scatt_data = scatt_mat[scatt_key]

        # 提取统计信息
        labels, means, stds = extract_scatt_stats_fast(label_data, scatt_data)
        index = [f"{basename}_{i+1}" for i in range(len(labels))]

        # 保存单独的散射表格
        df_scatt = pd.DataFrame({
            "Index": index,
            "Scatt_Mean": means,
            "Scatt_STD": stds
        })
        df_scatt.to_excel(output_path, index=False)

        # 合并到原有的量化表格（如果存在）
        if os.path.isfile(measure_path):
            df_measure = pd.read_excel(measure_path)

            # 删除旧的 Scatt_Mean 和 Scatt_STD（若存在）
            for col in ["Scatt_Mean", "Scatt_STD"]:
                if col in df_measure.columns:
                    df_measure.drop(columns=[col], inplace=True)

            df_merged = pd.merge(df_measure, df_scatt, on="Index", how="left")

            # 保存覆盖原表
            df_merged.to_excel(measure_path, index=False)

        return f"✅ 合并完成: {basename}"

    except Exception as e:
        return f"❌ 错误处理 {fname}: {e}"

def process_one_root_folder(root_dir, max_workers=8):
    seg_label_dir = os.path.join(root_dir, "seg_label")
    scatt_mat_dir = os.path.join(root_dir, "scatt_mat")
    output_dir = os.path.join(root_dir, "scatt")
    measure_dir = os.path.join(root_dir, "measure_excel")

    os.makedirs(output_dir, exist_ok=True)

    mat_files = sorted([f for f in os.listdir(seg_label_dir) if f.endswith('_label.mat')])
    folder_suffix = os.path.basename(root_dir)[-4:]

    print(f"\n📁 正在处理大文件夹: {root_dir}, 共 {len(mat_files)} 个孔位文件夹")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_sample, seg_label_dir, scatt_mat_dir, output_dir, measure_dir, fname): fname
            for fname in mat_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{folder_suffix} 孔位文件夹", unit="folder"):
            result = future.result()
            print(result)

if __name__ == "__main__":
    roots = [
        r"Data\nnUNet_FXN\FXN_0701",
        r"Data\nnUNet_FXN\FXN_0703"
    ]
    print(f"总共 {len(roots)} 个大文件夹；")
    for i, root in enumerate(roots):
        print(f"\n[{i+1}/{len(roots)}] 开始处理大文件夹: {root}")
        start_time = time.time()
        process_one_root_folder(root, max_workers=1)
        print(f"✅ 处理完成: {root}, 耗时 {time.time() - start_time:.2f}s\n")
