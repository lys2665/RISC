import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 学术论文样式设置
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (15, 10)
})


def load_and_process():
    """加载并处理所有结果文件"""
    all_files = glob.glob("min_support_parameter/*_sensitivity.csv")
    datasets_order = ['activity', 'aslbu', 'context',
                      'epitope', 'gene', 'news', 'pioneer',
                      'question', 'reuters', 'robot', 'skating',
                      'unix', 'webkb']

    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dataset = f.split('/')[-1].split('_')[0]
        df['Dataset'] = dataset
        dfs.append(df)

    combined = pd.concat(dfs)
    combined['Dataset'] = pd.Categorical(combined['Dataset'], categories=datasets_order, ordered=True)
    return combined.sort_values(['Dataset', 'min_support'])


def plot_corrected_heatmaps():
    """生成坐标轴正确的四热力图布局"""
    df = load_and_process()
    metrics = ['Purity', 'NMI', 'F1', 'Time']
    titles = ['(a) Purity', '(b) NMI',
              '(c) F1-score', '(d) Running Time (seconds)']

    fig, axs = plt.subplots(1, 4, figsize=(10, 2.5))

    cmap = sns.color_palette("viridis", as_cmap=True)

    # 获取数据集顺序和支持度阈值范围
    datasets = df['Dataset'].cat.categories.tolist()
    min_supports = sorted(df['min_support'].unique())

    for i, (ax, metric, title) in enumerate(zip(axs.flat, metrics, titles)):
        # 生成正确方向的矩阵（行：支持度阈值，列：数据集）
        matrix = df.pivot_table(index='min_support',
                                columns='Dataset',
                                values=metric,
                                observed=True)

        # 确保矩阵行列顺序正确
        matrix = matrix.reindex(index=min_supports, columns=datasets)

        # 绘制热力图（不再转置矩阵）
        im = ax.imshow(matrix,
                       cmap=cmap,
                       aspect='auto',
                       vmin=matrix.min().min(),
                       vmax=matrix.max().max())

        # 设置坐标轴标签
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels(datasets, rotation=90)
        ax.set_yticks(np.arange(len(min_supports)))
        ax.set_yticklabels([f"{ms:.2f}" for ms in min_supports])

        # 添加颜色条
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.outline.set_linewidth(0.5)

        # 添加标题
        ax.set_title(title, pad=12)

        # 添加网格线
        ax.grid(visible=False)
        # ax.set_xlabel('Dataset', fontweight='bold', labelpad=10)
        # ax.set_ylabel('Minimum Support', fontweight='bold', labelpad=10)

    plt.tight_layout()
    plt.savefig("fig_min.pdf", bbox_inches='tight')
    plt.show()


# 执行绘图
plot_corrected_heatmaps()