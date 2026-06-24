# -*- coding: utf-8 -*-
"""
汇总 cluster_analysis_1.py 的 Sheet2 到 nnUNet_Analysis.xlsx
"""
import os
import glob
import pandas as pd

paths = [
    'Data/nnUNet_FXN/FXN_0701/cluster_excel',
    'Data/nnUNet_FXN/FXN_0703/cluster_excel'
]

dfs = []
for p in paths:
    files = glob.glob(os.path.join(p, '*.xlsx'))
    print(f"📂 {p}: {len(files)} 个文件")
    for f in files:
        try:
            df = pd.read_excel(f, sheet_name='Sheet2')
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ 跳过 {os.path.basename(f)}: {e}")

if dfs:
    df_all = pd.concat(dfs, ignore_index=True)
    out_path = 'Data/nnUNet_FXN/nnUNet_Analysis.xlsx'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_all.to_excel(out_path, index=False)
    print(f"\n✅ 汇总完成: {out_path}")
    print(f"   共 {len(df_all)} 行数据")
else:
    print("❌ 未找到有效数据")
