# 类器官分割量化项目 — Claude 上下文

## 项目概述
基于 nnUNet / SAM2UNet 的 OCT 三维类器官分割 + 多维度量化 + 聚类分型 + PCA 综合评分 + 药效关联分析。

## 数据流（必须遵循的顺序）

```
nnUNet .nii.gz → nnunet_bridge.py → .mat (seg_fill, seg_label, scatt_mat)
    ↓
measure_from_label.py → measure_excel/*.xlsx (形态特征)
    ↓
scatt.py → 合并 Scatt_Mean/Scatt_STD 到 measure_excel
    ↓
cluster-merge.py → cluster_merge/*.xlsx (4类表型: 巨大囊泡型/中等过渡型/小体积基准型/高散射实心型)
    ↓
cluster_analysis_1.py → 各孔板 Sheet2 汇总
    ↓
merge_analysis.py → nnUNet_Analysis.xlsx (主表)
    ↓
roughness.py → roughness/*.xlsx (表面粗糙度)
    ↓
roughness_analysis.py → 汇总到主表
    ↓
PCA.py → PCA_Model_Details.xlsx (综合评分)
    ↓
vol_rate_oac_efficacy_correlation.py → 药效关联 (vol_rate vs OAC)
```

## 关键路径（nnUNet 数据）

- 输入：`nnUNet_Data/FXN_0701`, `nnUNet_Data/FXN_0703`
- 输出根目录：`Data/nnUNet_FXN/FXN_0701`, `Data/nnUNet_FXN/FXN_0703`
- 主表：`Data/nnUNet_FXN/nnUNet_Analysis.xlsx`
- PCA 详情：`Data/nnUNet_FXN/PCA_Model_Details.xlsx`

## Well ID → 分组映射

```python
Control: E11, F2, F6, F8, F9, F11
20 μM:  B2, B3, B4, C2, C3, C4
40 μM:  B5, B6, B7, C5, C6, C7
80 μM:  B8, B9, B10, C8, C9, C10
```

## OAC 药效参考值

```python
OAC_RATES = {'Control': -14.3, '20': 1.9, '40': 9.3, '80': 18.5}
```

## 聚类 4 类表型映射

```python
0: '巨大囊泡型'      (原始 6 类的 3,5 合并)
1: '中等过渡型'      (原始 6 类的 1)
2: '小体积基准型'    (原始 6 类的 0,4 合并)
3: '高散射实心型'    (原始 6 类的 2)
```

## 模型维度限制（重要！）

- `scaler-scatt.pickle`: 11 维（含 Scatt_Mean, Scatt_STD）
- `Kmeans-scatt.pickle`: 9 维（仅形态学，不含 Scatt）
- 使用方式：先 scaler 全部 11 维 → 取前 9 维给 KMeans
- 必须用 `.values` 传 numpy array，不能用 DataFrame（否则 sklearn 报列名验证错误）

## 已知问题

1. **B2_0701 原始图损坏**：`nnUNet_Data/FXN_0701/organoid_001_0000.nii.gz` gzip 解压失败，该孔无 Scatt 数据
2. **Windows 多进程限制**：`.nii.gz` 读取避免用 `ProcessPoolExecutor`，推荐串行或 `max_workers=1`
3. **内存限制**：32GB RAM，单个 `.mat` 约 300MB，多进程并行容易 OOM

## 常用操作

### 重新生成完整链路（从头开始）
```bash
python nnunet_bridge.py          # .nii.gz → .mat
python measure_from_label.py     # seg_label → measure_excel
python scatt.py                  # 合并散射统计
python cluster-merge.py          # 聚类合并为4类
python cluster_analysis_1.py     # 孔板级汇总
python merge_analysis.py         # 汇总为主表
python roughness.py              # 粗糙度计算
python roughness_analysis.py     # 粗糙度汇总到主表
python PCA.py                    # 综合评分
python vol_rate_oac_efficacy_correlation.py  # 药效关联
```

### 快速检查
```bash
# 检查 Scatt 是否已填充
python -c "import pandas as pd; df=pd.read_excel('Data/nnUNet_FXN/FXN_0701/cluster_merge/B3_0701_merge.xlsx'); print(df['Scatt_Mean'].head())"

# 检查主表行数
python -c "import pandas as pd; df=pd.read_excel('Data/nnUNet_FXN/nnUNet_Analysis.xlsx'); print(len(df), 'rows'); print(list(df.columns))"
```

## 汇报关注指标

1. **PCA 方差解释率**：`1_Eigenvalues` 中 PC1+PC2 累计 > 70% 为好
2. **药效剂量响应**：`vol_rate_mean` 随 OAC 浓度递减（Control 284% → 80μM 0.7%）
3. **体积响应比与 OAC 抑制率相关性**：Pearson r > 0.9 说明量化指标有效
4. **关键特征排序**：`3_Final_Coefficients` 中绝对值最大的特征对药效区分贡献最大

## 代码规范

- 路径统一使用正斜杠 `/` 或 `os.path.join`
- 多进程在 Windows 下优先设为 `max_workers=1`
- `.mat` 变量名：
  - 填充掩膜：`Data_fill`
  - 实例标签：`Data_label`
  - 散射数据：`data_scatt` / `Data_scatt`
