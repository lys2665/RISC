import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 学术论文风格设置
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

# 手动输入数据（请仔细核对）
data = {
    "Dataset": [
        "Activity", "Asblu", "Context", "Epitope", "Gene", "News",
        "Pioneer", "Question", "Reuters", "Robot", "Skating", "Unix", "Webkb"
    ],
    "RISC": [0.004, 0.025, 0.146, 0.081, 1.214, 2.839, 0.056, 0.035, 0.449, 0.454, 0.108, 0.727, 2.938],
    "SigISC": [0.168, 0.024, 0.177, 0.097, 0.136, 0.876, 0.026, 0.033, 0.156, 0.511, 0.075, 0.517, 0.745],
    "ISCT": [1.870, 14.490, 27.749, 11.352, 279.628, 799.166, 5.773, 5.982, 73.082, 50.308, 32.419, 333.306, 317.148],
    "kkmeans": [0.030, 0.460, 0.950, 10.110, 87.760, 1129.820, 0.240, 4.150, 37.990, 57.800, 2.110, 62.640, 676.750],
    "RSC": [0.130, 2.520, 1.570, 29.370, 33.660, 273.350, 2.450, 14.920, 34.800, 87.480, 6.79, 95.01, 185.5],
    "RFSC": [0.288, 28.497, 6.278, 65.299, 92.409, 7086.468, 8.197, 114.000, 494.419, 174.644, 11.371, 255.546, 1516.368]
}

df = pd.DataFrame(data).set_index("Dataset")

# 创建图形
plt.figure(figsize=(7, 4))

# 颜色映射（ColorBrewer Set1）
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

# 绘制折线
for idx, method in enumerate(df.columns):
    plt.semilogy(df.index, df[method],
                marker='o',
                markersize=6,
                linewidth=1.5,
                color=colors[idx],
                linestyle='--' if method in ["RSC", "RFSC"] else '-',
                label=method)

# 坐标轴设置
plt.xticks(rotation=45, ha="right")
plt.ylabel("Running Time (seconds)")
plt.ylim(1e-3, 1e5)  # 根据数据范围调整

# 网格线设置
# plt.grid(True, which="both", linestyle='--', alpha=0.6)

# 图例设置
plt.legend(ncol=3, loc="upper right", frameon=True, framealpha=0.9, edgecolor="black", fontsize=10, markerscale=0.8)

plt.tight_layout()

# 保存文件（可选）
plt.savefig("runtime_comparison.pdf", bbox_inches="tight")
plt.show()