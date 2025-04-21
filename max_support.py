import time  # 新增时间模块
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from RandomCluster.RISC import run_single_experiment
from RandomCluster.datainput import datainput

# 实验配置
PARAM_RANGE = np.round(np.arange(0.5, 1.0, 0.1), 2)
DATASETS = ['activity', 'aslbu', 'context',
            'epitope', 'gene', 'news', 'pioneer',
            'question', 'reuters', 'robot', 'skating',
            'unix', 'webkb']
OUTPUT_DIR = "max_support_parameter"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_experiment_pipeline():
    """主实验流程（添加运行时间统计）"""
    for dataset in tqdm(DATASETS, desc="总体进度"):
        try:
            db, data_label, _, _ = datainput(f'dataset/{dataset}.txt')
            true_cluster_num = len(set(data_label))
        except Exception as e:
            print(f"数据集 {dataset} 加载失败: {str(e)}")
            continue

        results = []
        for max_support in tqdm(PARAM_RANGE, desc=f"{dataset} 参数扫描", leave=False):
            start_time = time.perf_counter()  # 实验开始计时
            try:
                res = run_single_experiment(
                    db, data_label, true_cluster_num,
                    min_support=0.25 * len(db),
                    max_support=max_support * len(db)
                )
                elapsed_time = time.perf_counter() - start_time  # 计算耗时

                results.append({
                    'max_support': max_support,
                    'Purity': res['purity'],
                    'NMI': res['nmi'],
                    'F1': res['f1'],
                    'Time': elapsed_time  # 新增时间字段
                })
            except Exception as e:
                print(f"数据集 {dataset} 参数 {max_support} 失败: {str(e)}")
                results.append({
                    'min_support': max_support,
                    'Purity': np.nan,
                    'NMI': np.nan,
                    'F1': np.nan,
                    'Time': np.nan  # 异常情况记录NaN
                })

        df = pd.DataFrame(results)
        df.to_csv(f"{OUTPUT_DIR}/{dataset}_sensitivity.csv", index=False)
        print(f"数据集 {dataset} 结果已保存")


if __name__ == "__main__":
    run_experiment_pipeline()
    print(f"所有实验完成，结果保存在 {OUTPUT_DIR} 目录")