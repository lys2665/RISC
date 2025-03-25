import numpy as np
import metis
from sklearn import metrics
from datainput import datainput
import Levenshtein
from collections import Counter


def is_subsequence(seq, pattern):
    i, j = 0, 0
    while i < len(seq) and j < len(pattern):
        if seq[i] == pattern[j]:
            j += 1
        i += 1
    return j == len(pattern)


def extract_kmers(seq, length):
    """从一个序列中提取长度为 `length` 的 k-mer"""
    return [tuple(seq[i: i + length]) for i in range(len(seq) - length + 1)]

def get_all_kmers(db, length):
    """从整个数据库中提取所有 k-mer"""
    all_kmers = []
    for seq in db:
        all_kmers.extend(extract_kmers(seq, length))
    return all_kmers

def calculate_support_thresholds(kmer_counts, db, p):
    """计算支持度的最小值和最大值"""
    db_size = len(db)
    sorted_counts = sorted(kmer_counts.items(), key=lambda x: x[1])
    value_distribution = Counter(kmer_counts.values())
    sorted_dist = sorted(value_distribution.items())

    # Start and end indices for calculating thresholds
    start_idx = int(len(sorted_dist) * (1 - p))
    start_idx = max(start_idx, 0)  # Ensure start_idx is within bounds
    end_idx = len(sorted_dist)

    # Calculate min_support and max_support
    min_support = sorted_dist[start_idx][0] if sorted_dist[start_idx][0] > 1 else 2
    max_support = sorted_dist[end_idx - 1][0]

    # Adjust threshold values
    while max_support > db_size * 0.5 and start_idx > 0:
        start_idx -= 1
        end_idx -= 1
        min_support = sorted_dist[start_idx][0] if sorted_dist[start_idx][0] > 1 else 2
        max_support = sorted_dist[end_idx - 1][0]

    return min_support, max_support


def filter_kmers_by_support(kmer_counts, min_support, max_support):
    """根据支持度筛选 k-mer，并按计数从高到低排序"""
    sorted_kmers = sorted(kmer_counts.items(), key=lambda x: x[1])
    filtered = [(kmer, count) for kmer, count in sorted_kmers if min_support <= count <= max_support]
    filtered = filtered[::-1]
    return [list(kmer) for kmer, count in filtered]

def filter_by_edit_distance(candidates, min_threshold, max_threshold):
    # 第一次过滤：确保所有距离在[min, max]之间
    filtered_sequences = [c for c in candidates]
    i = 0
    while i < len(filtered_sequences):
        # 检查当前序列与所有其他序列的编辑距离
        distances = [Levenshtein.distance(filtered_sequences[i], other) for other in filtered_sequences if
                     filtered_sequences[i] != other]
        if not all(min_threshold <= dist <= max_threshold for dist in distances):
            filtered_sequences.remove(filtered_sequences[i])
        else:
            i += 1

    # 第二次过滤：移除所有其他距离都等于max_threshold的kmer
    i = 0
    while i < len(filtered_sequences):
        # 检查当前序列与所有其他序列的编辑距离
        distances = [Levenshtein.distance(filtered_sequences[i], other) for other in filtered_sequences if
                     filtered_sequences[i] != other]
        if all(dist == max_threshold for dist in distances):
            filtered_sequences.remove(filtered_sequences[i])
        else:
            i += 1

    return filtered_sequences


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


def partition_and_analyze(edges, pattern_count, db, k, patterns):
    total_nodes = pattern_count + len(db)
    adj_list = [[] for _ in range(total_nodes)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # METIS图划分
    _, parts = metis.part_graph(adj_list, k)

    # 解析划分结果
    clusters = {}
    seq_to_cluster = [-1] * len(db)  # 初始化一个列表，用于记录每个序列所属的簇编号

    for node_idx, cluster_id in enumerate(parts):
        if node_idx < pattern_count:  # 模式节点
            clusters.setdefault(cluster_id, {'patterns': [], 'sequences': []})
            clusters[cluster_id]['patterns'].append(node_idx)  # 记录模式索引
        else:  # 序列节点
            seq_idx = node_idx - pattern_count
            clusters.setdefault(cluster_id, {'patterns': [], 'sequences': []})
            clusters[cluster_id]['sequences'].append(seq_idx)  # 记录序列索引

            # 记录每个序列所属的簇编号
            seq_to_cluster[seq_idx] = cluster_id

    return clusters, seq_to_cluster

def optimize_clusters(clusters, db, patterns):
    optimized_clusters = {}
    for cluster_id, data in clusters.items():
        orig_patterns = data['patterns']
        sequences = data['sequences']

        selected_patterns = greedy_set_cover(orig_patterns, sequences, db, patterns)

        if not  selected_patterns and orig_patterns:
            selected_patterns = [orig_patterns[0]]

        optimized_clusters[cluster_id] = {
            'patterns': selected_patterns,
            'sequences': sequences
        }

    return optimized_clusters

def greedy_set_cover(cluster_patterns, cluster_sequences, db, patterns):
    pattern_coverage = {}
    for pattern_index in cluster_patterns:
        pattern = patterns[pattern_index]
        covered = set()
        for seq_index in cluster_sequences:
            if is_subsequence(db[seq_index], pattern):
                covered.add(seq_index)
        if covered:
            pattern_coverage[pattern_index] = covered

    selected_patterns = []
    remaining_sequences = set(cluster_sequences)

    while remaining_sequences and pattern_coverage:
        best_pattern = max(pattern_coverage, key=lambda p: len(pattern_coverage[p] & remaining_sequences))
        selected_patterns.append(best_pattern)
        remaining_sequences -= pattern_coverage[best_pattern]
        del pattern_coverage[best_pattern]

    return selected_patterns

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
    """验证模式在不同簇中的分布情况"""
    # 构建簇序列映射
    cluster_sequences = {}
    for cluster_id, data in optimized_clusters.items():
        cluster_sequences[cluster_id] = [db[i] for i in data['sequences']]

    # 构建模式到簇的映射
    pattern_cluster_map = {}
    for cluster_id, data in optimized_clusters.items():
        for pattern_idx in data['patterns']:
            pattern = tuple(patterns[pattern_idx])
            pattern_cluster_map[pattern] = cluster_id

    # 统计每个模式在各簇中的出现次数
    pattern_distribution = {}
    for pattern in pattern_cluster_map:
        pattern_distribution[pattern] = {}
        total_count = 0

        # 检查每个簇
        for cluster_id, seq_list in cluster_sequences.items():
            count = 0
            for seq in seq_list:
                if is_subsequence(seq, pattern):
                    count += 1
            pattern_distribution[pattern][cluster_id] = {
                'count': count,
                'support': count / len(seq_list) if len(seq_list) > 0 else 0
            }
            total_count += count

        # 计算主要簇判别性指标
        home_cluster = pattern_cluster_map[pattern]
        home_support = pattern_distribution[pattern][home_cluster]['support']
        other_support = sum(v['support'] for cid, v in pattern_distribution[pattern].items()
                            if cid != home_cluster) / (len(cluster_sequences) - 1) if len(cluster_sequences) > 1 else 0
        discriminability = home_support - other_support

        pattern_distribution[pattern]['metrics'] = {
            'home_cluster': home_cluster,
            'discriminability': discriminability,
            'total_occurrence': total_count
        }

    return pattern_distribution


def run_single_experiment(db, data_label, k):
    """单次实验运行"""
    all_kmers = get_all_kmers(db, 3)

    kmer_counter = Counter(all_kmers)

    min_support, max_support = calculate_support_thresholds(kmer_counter, db, 0.3)
    # print(min_support, max_support)
    candidate_kmers = filter_kmers_by_support(kmer_counter, min_support, max_support)
    # print(len(candidate_kmers))
    patterns = filter_by_edit_distance(candidate_kmers, 2, 3)
    print(len(patterns))


    edges, pattern_count = build_bipartite_graph(db, patterns)
    clusters, pred_label = partition_and_analyze(edges, pattern_count, db, k, patterns)
    optimized_clusters = optimize_clusters(clusters, db, patterns)

    # 统计优化后模式
    all_optimized = set()
    for cluster_data in optimized_clusters.values():
        all_optimized.update(cluster_data['patterns'])

    optimized_patterns = [patterns[idx] for idx in all_optimized]
    total_patterns = len(optimized_patterns)
    avg_length = sum(len(p) for p in optimized_patterns) / total_patterns if total_patterns > 0 else 0

    # 计算评估指标
    acc = measure_performance(data_label, pred_label)
    pattern_dist = validate_pattern_discriminability(db, optimized_clusters, patterns)

    return {
        'purity': acc[0],
        'nmi': acc[1],
        'f1': acc[2],
        'num_patterns': total_patterns,
        'avg_length': avg_length,
        'pattern_analysis': {
            'distributions': pattern_dist,
            'avg_discriminability': np.mean([v['metrics']['discriminability'] for v in pattern_dist.values()]),
            'min_discriminability': np.min([v['metrics']['discriminability'] for v in pattern_dist.values()]),
            'max_discriminability': np.max([v['metrics']['discriminability'] for v in pattern_dist.values()])
        }
    }

if __name__ == '__main__':
    dataset = ['activity', 'aslbu', 'auslan2', 'context', 'epitope', 'gene', 'news',
               'pioneer', 'question', 'reuters', 'robot', 'skating', 'unix', 'webkb']
    dataset = ['webkb']

    for dataset_name in dataset:
        db, data_label, _, _ = datainput(f'RandomCluster/dataset/{dataset_name}.txt')
        true_cluster_num = len(set(data_label))
        result = run_single_experiment(db, data_label, true_cluster_num)

        # 直接输出单次实验结果
        print(f"[{dataset_name}]:")
        print(f"Purity: {result['purity']:.4f}")
        print(f"NMI: {result['nmi']:.4f}")
        print(f"F1-score: {result['f1']:.4f}")
        print(f"模式数: {result['num_patterns']:.1f}")
        print(f"平均模式长度: {result['avg_length']:.2f}")
        print("=" * 80)

        # 输出新增的判别性指标
        print("\n模式判别性分析:")
        print(f"平均判别性: {result['pattern_analysis']['avg_discriminability']:.2f}")
        print(f"最小判别性: {result['pattern_analysis']['min_discriminability']:.2f}")
        print(f"最大判别性: {result['pattern_analysis']['max_discriminability']:.2f}")

        # 详细分布
        print("模式分布:")
        for i, (pattern, dist) in enumerate(result['pattern_analysis']['distributions'].items()):
            print(f"模式 {i + 1}: {pattern}")
            print(f"所属簇: {dist['metrics']['home_cluster']}")
            print(f"总出现次数: {dist['metrics']['total_occurrence']}")
            print("各簇支持度:")
            for cid, vals in dist.items():
                if cid == 'metrics':
                    continue
                print(f"  簇 {cid}: {vals['support']:.2f}")
            print("-" * 40)

        print("=" * 80)





