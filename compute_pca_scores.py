# -*- coding: utf-8 -*-
"""
计算 nnUNet 每个孔的 PCA Score
输入：nnUNet_Analysis.xlsx
输出：nnUNet_PCA_Scores.xlsx（含 Name, PC1~PC4, Result）
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re

# ================= 配置 =================
input_path = 'Data/nnUNet_FXN/nnUNet_Analysis.xlsx'
output_path = 'Data/nnUNet_FXN/nnUNet_PCA_Scores.xlsx'

# 与 PCA.py 完全一致的特征列表
features_list = [
    'Cavity_Volume_All_1', 'Cyst_Thick_Avg_3', 'Long_Axis_Avg_4', 'Number_2', 'Number_3',
    'Roughness_All',
    'Scatt_Mean_Avg_3',
    'Scatt_Mean_Avg_4',
    'Short_Axis_Avg_2',
    'Short_Axis_Avg_3',
    'Short_Axis_Avg_4',
    'Surface_Avg_1', 'Surface_Avg_2', 'Surface_Avg_3', 'Volume_Fill_Avg_2'
]

def get_feature_sort_key(feature_name):
    if 'Roughness' in feature_name:
        return (1, 999, feature_name)
    match = re.search(r'_(\d+)$', feature_name)
    if match:
        return (int(match.group(1)), 0, feature_name)
    return (99, 0, feature_name)

features_list.sort(key=get_feature_sort_key)

# ================= 读取数据 =================
print(f"读取: {input_path}")
Data_All = pd.read_excel(input_path)
existing_cols = [c for c in features_list if c in Data_All.columns]
Data = Data_All[existing_cols].fillna(0)

# ================= PCA 计算 =================
scaler = StandardScaler()
Data_std = scaler.fit_transform(Data)

n_components = 4
pca = PCA(n_components=n_components, svd_solver='full', random_state=42)
Data_Pca = pca.fit_transform(Data_std)

# 权重（与 PCA.py 一致）
variance_ratio = pca.explained_variance_ratio_
weights = variance_ratio / np.sum(variance_ratio)

# Result = weighted sum of PC scores
Result = np.dot(Data_Pca, weights)

# ================= 保存 =================
df_scores = pd.DataFrame({
    'Name': Data_All['Name'],
    'PC1': Data_Pca[:, 0],
    'PC2': Data_Pca[:, 1],
    'PC3': Data_Pca[:, 2],
    'PC4': Data_Pca[:, 3],
    'Result': Result
})

df_scores.to_excel(output_path, index=False)
print(f"[OK] 已保存: {output_path} ({len(df_scores)} 个孔)")
print(f"\n方差解释率: PC1={variance_ratio[0]:.2%}, PC2={variance_ratio[1]:.2%}, "
      f"PC3={variance_ratio[2]:.2%}, PC4={variance_ratio[3]:.2%}")
print(f"累计: {np.sum(variance_ratio[:2]):.2%} (PC1+PC2), {np.sum(variance_ratio):.2%} (全部)")
