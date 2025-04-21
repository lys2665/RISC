import time
import numpy as np
import metis
from sklearn import metrics
from datainput import datainput
from collections import Counter
from metis import idx_t  # 从METIS库导入数据类型

def is_subsequence(seq, pattern):
    """检查seq是否包含pattern"""
    n = len(seq)
    m = len(pattern)
    j = 0  # 用于遍历subsequence的指针
    for i in range(n):
        if seq[i] == pattern[j]:
            j += 1
            if j == m:
                return True
        elif j > 0:
            i -= 1  # 回溯到上一个与subsequence匹配的元素位置
    return False


def extract_kmers(seq, length):
    """从一个序列中提取长度为 `length` 的 k-mer"""
    return [tuple(seq[i: i + length]) for i in range(len(seq) - length + 1)]


def get_all_kmers(db, length):
    """从整个数据库中提取所有 k-mer"""
    all_kmers = []
    for seq in db:
        all_kmers.extend(extract_kmers(seq, length))
    return all_kmers

def filter_kmers_by_support(kmer_counts, min_support, max_support):
    """根据支持度筛选 k-mer，并按计数从高到低排序"""
    sorted_kmers = sorted(kmer_counts.items(), key=lambda x: x[1])
    filtered = [(kmer, count) for kmer, count in sorted_kmers if min_support <= count <= max_support]
    filtered = filtered[::-1]
    return [list(kmer) for kmer, count in filtered]

def build_bipartite_graph(db, patterns):
    """构建模式-序列二分图"""
    pattern_dict = {tuple(p): idx for idx, p in enumerate(patterns)}
    edges = []

    # 为每个序列创建节点索引（模式节点在前，序列节点在后）
    pattern_count = len(patterns)
    for seq_idx, seq in enumerate(db):
        seq_node = pattern_count + seq_idx  # 序列节点索引

        # 寻找匹配的所有模式
        for pattern in patterns:
            if is_subsequence(seq, pattern):
                pattern_node = pattern_dict[tuple(pattern)]
                edges.append((pattern_node, seq_node))

    return edges, pattern_count


def partition_and_analyze(edges, pattern_count, db, k):
    total_nodes = pattern_count + len(db)
    adj_list = [[] for _ in range(total_nodes)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # 构建邻接表数据结构
    xadj = [0]
    adjncy = []
    ptr = 0
    for node_edges in adj_list:
        xadj.append(ptr + len(node_edges))
        adjncy.extend(node_edges)
        ptr += len(node_edges)

    # 转换为METIS需要的ctypes数组类型（关键修复点）
    xadj_array = (idx_t * len(xadj))(*xadj)
    adjncy_array = (idx_t * len(adjncy))(*adjncy)
    vwgt_array = (idx_t * total_nodes)(*([int(len(db) / k)] * pattern_count + [1] * len(db)))

    # 构建图对象（添加类型验证）
    import ctypes
    assert isinstance(xadj_array, ctypes.Array), "xadj必须转换为ctypes数组"
    assert isinstance(adjncy_array, ctypes.Array), "adjncy必须转换为ctypes数组"

    graph = metis.METIS_Graph(
        nvtxs=idx_t(total_nodes),
        ncon=idx_t(1),
        xadj=xadj_array,
        adjncy=adjncy_array,
        vwgt=vwgt_array,
        vsize=None,
        adjwgt=None
    )

    # 执行图划分（添加异常捕获）
    try:
        _, parts = metis.part_graph(graph, nparts=k)
    except Exception as e:
        print(f"划分失败，参数详情：")
        print(f"顶点数：{total_nodes}, 分区数：{k}")
        print(f"邻接表指针长度：{len(xadj)}, 正确值应为{total_nodes + 1}")
        print(f"邻接节点数量：{len(adjncy)}")
        raise

    # 解析结果
    clusters = {}
    seq_to_cluster = [-1] * len(db)

    for node_idx, cluster_id in enumerate(parts):
        if node_idx < pattern_count:
            clusters.setdefault(cluster_id, {'patterns': [], 'sequences': []})['patterns'].append(node_idx)
        else:
            seq_idx = node_idx - pattern_count
            clusters.setdefault(cluster_id, {'patterns': [], 'sequences': []})['sequences'].append(seq_idx)
            seq_to_cluster[seq_idx] = cluster_id

    return clusters, seq_to_cluster

def optimize_clusters(clusters, db, patterns):
    optimized = {}
    for cid, data in clusters.items():
        # 获取所有相关簇信息
        cluster_info = {
            'cluster_id': cid,
            'sequences': data['sequences'],
            'patterns': data['patterns']
        }
        # 执行改进的贪心选择
        selected = discriminative_greedy(cluster_info, clusters, db, patterns)
        optimized[cid] = {
            'patterns': selected,
            'sequences': data['sequences']
        }
    return optimized

def discriminative_greedy(cluster_data, all_clusters, db, patterns):
    """基于F1分数优化的贪心算法：选择能提升集合F1的模式"""
    current_cluster_id = cluster_data['cluster_id']
    candidate_patterns = cluster_data['patterns'].copy()
    selected = []

    # 预计算每个模式在各簇中的覆盖集合（序列索引）
    cluster_coverage = {}
    for cid in all_clusters:
        cluster_seq_indices = all_clusters[cid]['sequences']
        for p_idx in candidate_patterns:
            covered = set()
            for seq_idx in cluster_seq_indices:
                if is_subsequence(db[seq_idx], patterns[p_idx]):
                    covered.add(seq_idx)
            cluster_coverage[(p_idx, cid)] = covered

    # 计算每个模式的初始RR
    rr_values = {}
    current_size = len(cluster_data['sequences'])
    other_cids = [cid for cid in all_clusters if cid != current_cluster_id]

    for p_idx in candidate_patterns:
        tp = len(cluster_coverage[(p_idx, current_cluster_id)])
        pos_support = tp / current_size if current_size > 0 else 0

        other_supports = []
        for cid in other_cids:
            cid_size = len(all_clusters[cid]['sequences'])
            if cid_size == 0:
                continue
            fp = len(cluster_coverage[(p_idx, cid)])
            other_supports.append(fp / cid_size)

        avg_other = np.mean(other_supports) if other_supports else 0
        rr = pos_support / (avg_other + 1e-6)
        rr_values[p_idx] = rr

    # 筛选并排序候选模式（RR > 1）
    filtered = [p for p in candidate_patterns if rr_values[p] > 1]
    if not filtered:
        print(f"簇{current_cluster_id}无有效候选模式")
        return []
    filtered.sort(key=lambda x: rr_values[x], reverse=True)

    # 动态维护覆盖状态以计算F1
    current_TP = set()  # 当前簇被覆盖的序列
    current_FP = {cid: set() for cid in other_cids}  # 其他簇被覆盖的序列
    best_f1 = 0.0

    for p_idx in filtered:
        # 计算合并后的覆盖情况
        new_TP = current_TP.union(cluster_coverage[(p_idx, current_cluster_id)])
        new_FP = {cid: current_FP[cid].union(cluster_coverage[(p_idx, cid)])
                  for cid in other_cids}

        # 计算F1分数
        tp = len(new_TP)
        fp_total = sum(len(new_FP[cid]) for cid in other_cids)
        fn = current_size - tp

        precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # 仅保留提升F1的模式
        if f1 > best_f1:
            selected.append(p_idx)
            current_TP = new_TP
            current_FP = new_FP
            best_f1 = f1

    return selected


def measure_performance(data_label, y_pred):
    # 1. Purity
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np

    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix_ = contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix_, axis=0)) / np.sum(contingency_matrix_)

    purity = purity_score(data_label, y_pred)
    # 2 . NMI
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(data_label, y_pred)

    def f_measure(labels_true, labels_pred, beta=1.):
        (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(
            labels_true, labels_pred)
        p, r = tp / (tp + fp), tp / (tp + fn)
        f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
        return f_beta

    f1 = f_measure(data_label, y_pred)
    acc = [purity, nmi, f1]
    # print(acc)
    return acc


def validate_pattern_discriminability(db, optimized_clusters, patterns):
    """计算每个簇模式的目标簇正确覆盖率（覆盖自身序列）和非目标簇错误覆盖率（覆盖其他簇序列）"""
    # 构建簇到模式的映射 {cluster_id: [patterns]}
    cluster_patterns = {}
    for cluster_id, data in optimized_clusters.items():
        cluster_patterns[cluster_id] = [patterns[p_idx] for p_idx in data['patterns']]

    # 构建序列到所属簇的映射 {seq_idx: cluster_id}
    seq_to_cluster = {}
    cluster_sequences = {}  # {cluster_id: [seq_indices]}
    for cluster_id, data in optimized_clusters.items():
        cluster_sequences[cluster_id] = data['sequences']
        for seq_idx in data['sequences']:
            seq_to_cluster[seq_idx] = cluster_id

    results = {}
    for cluster_id in optimized_clusters:
        # 当前簇的模式和序列信息
        current_patterns = cluster_patterns[cluster_id]
        current_seqs = cluster_sequences[cluster_id]
        current_size = len(current_seqs)

        # 其他簇的序列信息
        other_seqs = [seq_idx for seq_idx in seq_to_cluster if seq_to_cluster[seq_idx] != cluster_id]
        other_size = len(other_seqs)

        # 统计覆盖率
        correct_covered = sum(
            1 for seq_idx in current_seqs
            if any(is_subsequence(db[seq_idx], p) for p in current_patterns)
        )
        wrong_covered = sum(
            1 for seq_idx in other_seqs
            if any(is_subsequence(db[seq_idx], p) for p in current_patterns)
        )

        # 计算比例
        correct_coverage = correct_covered / current_size if current_size > 0 else 0.0
        wrong_coverage = wrong_covered / other_size if other_size > 0 else 0.0

        results[cluster_id] = {
            'correct_coverage': correct_coverage,
            'wrong_coverage': wrong_coverage,
            'correct_count': correct_covered,
            'wrong_count': wrong_covered,
            'self_size': current_size,
            'other_size': other_size
        }

    return results


def run_single_experiment(db, data_label, k, min_support = None, max_support = None):
    """单次实验运行"""
    patterns = []
    current_k = 8
    if min_support is None:
        min_support = 0.25 * len(db)  # 原固定值
    if max_support is None:
        max_support = 0.7 * len(db)   # 原固定值
    start_time = time.time()
    def check_coverage(patterns, db):
        if not patterns:
            return False
        return all(any(is_subsequence(seq, pattern) for pattern in patterns) for seq in db)

    while not check_coverage(patterns, db) or patterns is []:
        current_kmers = get_all_kmers(db, current_k)
        kmer_counter = Counter(current_kmers)
        candidate_kmers = filter_kmers_by_support(kmer_counter, min_support, max_support)
        patterns.extend(candidate_kmers)
        # patterns.extend(current_kmers)

        current_k -= 1
        if current_k == 0:
            break
    edges, pattern_count = build_bipartite_graph(db, patterns)
    clusters, pred_label = partition_and_analyze(edges, pattern_count, db, k)
    #
    # for cid, cluster_data in clusters.items():
    #     print(f"簇 {cid} 的模式: {cluster_data['patterns']}")

    optimized_clusters = optimize_clusters(clusters, db, patterns)
    end_time = time.time()
    eval_time = end_time - start_time
    # 统计优化后模式
    all_optimized = set()
    for cluster_data in optimized_clusters.values():
        all_optimized.update(cluster_data['patterns'])

    optimized_patterns = [patterns[idx] for idx in all_optimized]
    total_patterns = len(optimized_patterns)
    avg_length = sum(len(p) for p in optimized_patterns) / total_patterns if total_patterns > 0 else 0

    # 计算评估指标
    acc = measure_performance(data_label, pred_label)
    coverage_stats = validate_pattern_discriminability(db, optimized_clusters, patterns)

    return {
        'purity': acc[0],
        'nmi': acc[1],
        'f1': acc[2],
        'num_patterns': total_patterns,
        'avg_length': avg_length,
        'coverage_stats': coverage_stats,  # 直接存储覆盖率统计结果
        'eval_time': eval_time
    }


if __name__ == '__main__':
    dataset = ['activity', 'aslbu', 'context', 'epitope', 'gene', 'news',
               'pioneer', 'question', 'reuters', 'robot', 'skating', 'unix', 'webkb']
    # dataset = ['auslan2']

    for dataset_name in dataset:
        db, data_label, _, _ = datainput(f'RandomCluster/dataset/{dataset_name}.txt')
        true_cluster_num = len(set(data_label))
        result = run_single_experiment(db, data_label, true_cluster_num)

        print(f"[{dataset_name}]:")
        print(f"Purity: {result['purity']:.4f}")
        print(f"NMI: {result['nmi']:.4f}")
        print(f"F1-score: {result['f1']:.4f}")
        print(f"模式数: {result['num_patterns']}")
        print(f"平均模式长度: {result['avg_length']:.2f}")
        print(f"运行时间: {result['eval_time']: .4f}")

        # 处理新的覆盖率统计
        coverage_stats = result['coverage_stats']
        if coverage_stats:
            print("\n各簇模式覆盖效果:")
            for cid, stats in coverage_stats.items():
                print(
                    f"  簇 {cid}: "
                    f"自身覆盖 {stats['correct_count']}/{stats['self_size']} ({stats['correct_coverage']:.2%}) | "
                    f"错误覆盖其他簇 {stats['wrong_count']}/{stats['other_size']} ({stats['wrong_coverage']:.2%})"
                )

            # 计算整体指标
            avg_correct = sum(s['correct_coverage'] for s in coverage_stats.values()) / len(coverage_stats)
            avg_wrong = sum(s['wrong_coverage'] for s in coverage_stats.values()) / len(coverage_stats)
            print(f"\n整体平均:")
            print(f"自身覆盖率: {avg_correct:.2%}  |  错误覆盖其他簇率: {avg_wrong:.2%}")
        else:
            print("无有效的覆盖率统计")

        print("=" * 80)