# -*- coding: utf-8 -*-
"""
药效分析可视化脚本
输出两张图：
1. 体积变化率柱状图（带误差棒）
2. vol_ratio vs oac_ratio 散点图（相关性）
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ================= 数据 =================
groups = ['Control', '20', '40', '80']
vol_rate_mean = [284.22, 92.89, 21.38, 0.71]
vol_rate_std = [143.88, 49.19, 36.17, 18.9]
oac_mean = [-14.3, 1.9, 9.3, 18.5]
vol_ratio_mean = [0.0, -0.6223, -0.9294, -0.9979]
oac_ratio_mean = [0.0, 1.1329, 1.6503, 2.2937]

# ================= 图 1：体积增长率柱状图 =================
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(groups))
colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']

bars = ax.bar(x, vol_rate_mean, yerr=vol_rate_std, capsize=5, color=colors, edgecolor='black', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=14)
ax.set_ylabel('Volume Growth Rate (%)', fontsize=14)
ax.set_xlabel('OAC Concentration', fontsize=14)
ax.set_title('Organoid Volume Response to OAC Treatment\n(nnUNet Segmentation)', fontsize=16, fontweight='bold')

# 在柱子上标注数值
for bar, val, std in zip(bars, vol_rate_mean, vol_rate_std):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 10,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 450)
plt.tight_layout()
plt.savefig('Data/nnUNet_FXN/vol_rate_barplot.png', dpi=300, bbox_inches='tight')
print("✅ 图 1 已保存: Data/nnUNet_FXN/vol_rate_barplot.png")

# ================= 图 2：相关性散点图 =================
fig, ax = plt.subplots(figsize=(7, 6))

# 计算 Pearson 相关性
r, p = stats.pearsonr(vol_ratio_mean, oac_ratio_mean)

# 拟合回归线
slope, intercept, _, _, _ = stats.linregress(vol_ratio_mean, oac_ratio_mean)
x_line = np.linspace(min(vol_ratio_mean)-0.1, max(vol_ratio_mean)+0.1, 100)
y_line = slope * x_line + intercept

ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.7, label=f'Linear fit (r={r:.3f})')

# 散点
for i, group in enumerate(groups):
    ax.scatter(vol_ratio_mean[i], oac_ratio_mean[i], s=200, c=colors[i],
               edgecolors='black', linewidths=1.5, zorder=5, label=group)
    # 标注
    ax.annotate(group, (vol_ratio_mean[i], oac_ratio_mean[i]),
                textcoords="offset points", xytext=(10, 10), fontsize=11, fontweight='bold')

ax.set_xlabel('Volume Response Ratio (normalized)', fontsize=14)
ax.set_ylabel('OAC Inhibition Ratio (normalized)', fontsize=14)
ax.set_title(f'Correlation: Organoid Response vs OAC Efficacy\nPearson r = {r:.3f}, p < 0.05',
             fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower left', fontsize=11)
plt.tight_layout()
plt.savefig('Data/nnUNet_FXN/correlation_scatter.png', dpi=300, bbox_inches='tight')
print(f"✅ 图 2 已保存: Data/nnUNet_FXN/correlation_scatter.png")
print(f"\n📊 Pearson r = {r:.3f}")
if abs(r) > 0.9:
    print("   → 高度相关！类器官体积响应与 OAC 药效高度一致")
elif abs(r) > 0.7:
    print("   → 显著相关")
else:
    print("   → 相关性一般")

plt.show()
