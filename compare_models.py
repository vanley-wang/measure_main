# -*- coding: utf-8 -*-
"""
nnUNet vs SAM2UNet 对比分析脚本
输出：
  - Excel: Data/nnUNet_FXN/nnUNet_vs_SAM2UNet_Comparison.xlsx
  - 控制台打印：各维度对比统计
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ================= 配置 =================

SAM_PATH = r'Data/FXN_2023_new（闭10新聚类）/FXN_2023_Analysis.xlsx'
SAM_PCA_PATH = r'Data/FXN_2023_new（闭10新聚类）/FXN_2023_PCA.xlsx'
NN_PATH = 'Data/nnUNet_FXN/nnUNet_Analysis.xlsx'
NN_PCA_PATH = 'Data/nnUNet_FXN/nnUNet_PCA_Scores.xlsx'

OUT_DIR = 'Data/nnUNet_FXN'
OUT_EXCEL = os.path.join(OUT_DIR, 'nnUNet_vs_SAM2UNet_Comparison.xlsx')

# Well ID -> 分组
GROUP_MAPPING = {
    'E11': 'Control', 'F2': 'Control', 'F6': 'Control', 'F8': 'Control', 'F9': 'Control', 'F11': 'Control',
    'B2': '20', 'B3': '20', 'B4': '20', 'C2': '20', 'C3': '20', 'C4': '20',
    'B5': '40', 'B6': '40', 'B7': '40', 'C5': '40', 'C6': '40', 'C7': '40',
    'B8': '80', 'B9': '80', 'B10': '80', 'C8': '80', 'C9': '80', 'C10': '80',
}

OAC_RATES = {'Control': -14.3, '20': 1.9, '40': 9.3, '80': 18.5}


def parse_well(name):
    """从 B2_0701 提取 Well ID"""
    return name.split('_')[0]


def add_group(df):
    """添加 Group 列"""
    df = df.copy()
    df['Well_ID'] = df['Name'].apply(parse_well)
    df['Group'] = df['Well_ID'].map(GROUP_MAPPING)
    df['OAC'] = df['Group'].map(OAC_RATES)
    return df


# ================= 1. 读取数据 =================
print("=" * 60)
print("nnUNet vs SAM2UNet 对比分析")
print("=" * 60)

df_sam = pd.read_excel(SAM_PATH)
df_nn = pd.read_excel(NN_PATH)
df_sam_pca = pd.read_excel(SAM_PCA_PATH)

# 检查 nnUNet PCA scores 是否存在
if os.path.exists(NN_PCA_PATH):
    df_nn_pca = pd.read_excel(NN_PCA_PATH)
else:
    print(f"\n[WARN] {NN_PCA_PATH} 不存在，请先运行: python compute_pca_scores.py")
    exit()

# 添加分组
df_sam = add_group(df_sam)
df_nn = add_group(df_nn)
df_sam_pca = add_group(df_sam_pca)
df_nn_pca = add_group(df_nn_pca)

# 只保留有分组的行
df_sam = df_sam[df_sam['Group'].notna()]
df_nn = df_nn[df_nn['Group'].notna()]
df_sam_pca = df_sam_pca[df_sam_pca['Group'].notna()]
df_nn_pca = df_nn_pca[df_nn_pca['Group'].notna()]

print(f"\n[STATS] SAM2UNet: {len(df_sam)} 个孔 | nnUNet: {len(df_nn)} 个孔")

# ================= 2. 表型分布对比 =================
print("\n" + "=" * 60)
print("[1]表型数量分布对比（按浓度分组平均）")
print("=" * 60)

# 计算每个孔的 total organoids
df_sam['Total'] = df_sam['Number_1'] + df_sam['Number_2'] + df_sam['Number_3'] + df_sam['Number_4']
df_nn['Total'] = df_nn['Number_1'] + df_nn['Number_2'] + df_nn['Number_3'] + df_nn['Number_4']

# 计算比例
for i in range(1, 5):
    df_sam[f'Pct_{i}'] = df_sam[f'Number_{i}'] / df_sam['Total'] * 100
    df_nn[f'Pct_{i}'] = df_nn[f'Number_{i}'] / df_nn['Total'] * 100

# 按浓度分组平均
groups_order = ['Control', '20', '40', '80']
dist_sam = df_sam.groupby('Group')[[f'Pct_{i}' for i in range(1, 5)]].mean().reindex(groups_order)
dist_nn = df_nn.groupby('Group')[[f'Pct_{i}' for i in range(1, 5)]].mean().reindex(groups_order)

print("\nSAM2UNet 各表型比例 (%):")
print(dist_sam.round(2).to_string())
print("\nnnUNet 各表型比例 (%):")
print(dist_nn.round(2).to_string())

# ================= 3. 体积响应曲线对比 =================
print("\n" + "=" * 60)
print("[2]体积响应曲线对比")
print("=" * 60)

# 计算每个孔的总体积（加权平均）
for df, label in [(df_sam, 'SAM'), (df_nn, 'NN')]:
    vol_cols = [f'Volume_Fill_Avg_{i}' for i in range(1, 5)]
    num_cols = [f'Number_{i}' for i in range(1, 5)]
    df['Total_Volume'] = sum(df[vol_cols[i]] * df[num_cols[i]] for i in range(4))

# 0701 = Day3, 0703 = Day5
df_sam['Date'] = df_sam['Name'].str[-4:]
df_nn['Date'] = df_nn['Name'].str[-4:]

vol_sam_d3 = df_sam[df_sam['Date'] == '0701'].groupby('Group')['Total_Volume'].mean().reindex(groups_order)
vol_sam_d5 = df_sam[df_sam['Date'] == '0703'].groupby('Group')['Total_Volume'].mean().reindex(groups_order)
vol_nn_d3 = df_nn[df_nn['Date'] == '0701'].groupby('Group')['Total_Volume'].mean().reindex(groups_order)
vol_nn_d5 = df_nn[df_nn['Date'] == '0703'].groupby('Group')['Total_Volume'].mean().reindex(groups_order)

# 计算变化率
vol_rate_sam = ((vol_sam_d5 - vol_sam_d3) / vol_sam_d3 * 100).round(2)
vol_rate_nn = ((vol_nn_d5 - vol_nn_d3) / vol_nn_d3 * 100).round(2)

print("\nSAM2UNet 体积增长率 (%):")
print(vol_rate_sam.to_string())
print("\nnnUNet 体积增长率 (%):")
print(vol_rate_nn.to_string())

# ================= 4. PCA Score 对比 =================
print("\n" + "=" * 60)
print("[3]PCA Score 对比")
print("=" * 60)

# 合并两个模型的 PCA 结果
df_compare = pd.merge(
    df_sam_pca[['Name', 'Result', 'Group']],
    df_nn_pca[['Name', 'Result']],
    on='Name', suffixes=('_SAM', '_NN')
)

r, p = stats.pearsonr(df_compare['Result_SAM'], df_compare['Result_NN'])
print(f"\nPearson r = {r:.4f}, p = {p:.4e}")
if abs(r) > 0.8:
    print("   -> [OK] 两个模型的 PCA Score 高度一致")
elif abs(r) > 0.5:
    print("   -> 中等一致")
else:
    print("   -> [WARN] 一致性较低，建议检查分割差异")

# 按浓度分组看平均 Score
score_sam = df_compare.groupby('Group')['Result_SAM'].mean().reindex(groups_order)
score_nn = df_compare.groupby('Group')['Result_NN'].mean().reindex(groups_order)
print("\nSAM2UNet 平均 PCA Score:")
print(score_sam.round(4).to_string())
print("\nnnUNet 平均 PCA Score:")
print(score_nn.round(4).to_string())

# ================= 5. 保存 Excel =================
print("\n" + "=" * 60)
print("[4]保存对比结果")
print("=" * 60)

os.makedirs(OUT_DIR, exist_ok=True)
with pd.ExcelWriter(OUT_EXCEL, engine='openpyxl') as writer:
    # Sheet 1: 表型分布
    dist_compare = pd.concat([dist_sam.add_suffix('_SAM'), dist_nn.add_suffix('_NN')], axis=1)
    dist_compare.to_excel(writer, sheet_name='1_Phenotype_Distribution')

    # Sheet 2: 体积响应
    vol_compare = pd.DataFrame({
        'Group': groups_order,
        'VolRate_SAM': vol_rate_sam.values,
        'VolRate_NN': vol_rate_nn.values
    })
    vol_compare.to_excel(writer, sheet_name='2_Volume_Response', index=False)

    # Sheet 3: PCA Score 逐孔对比
    df_compare.to_excel(writer, sheet_name='3_PCA_Score_Comparison', index=False)

    # Sheet 4: 分组平均 PCA Score
    score_compare = pd.DataFrame({
        'Group': groups_order,
        'Score_SAM': score_sam.values,
        'Score_NN': score_nn.values
    })
    score_compare.to_excel(writer, sheet_name='4_PCA_Score_by_Group', index=False)

print(f"[OK] 已保存: {OUT_EXCEL}")
print("\n包含 4 个 Sheet:")
print("  1. Phenotype_Distribution — 各表型比例对比")
print("  2. Volume_Response — 体积增长率对比")
print("  3. PCA_Score_Comparison — 逐孔 PCA Score 对比")
print("  4. PCA_Score_by_Group — 分组平均 PCA Score")
