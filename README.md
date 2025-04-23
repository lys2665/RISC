# Rule-based Interpretable Discrete Sequence Clustering 

This repository implements a rule-based interpretable discrete sequence clustering algorithm, written in Python.

## Main Steps

- **k-mer Extraction**: Extract k-mers of a specified length from the sequences.
- **Pattern Selection**: Select frequent patterns based on support threshold.
- **Bipartite Graph Construction**: Construct a bipartite graph between patterns and sequences for graph partitioning.
- **Graph Partitioning**: Use the METIS library to partition the bipartite graph and obtain the sequence clustering results.
- **Pattern Optimization**: Apply a greedy algorithm to optimize the pattern selection for better interpretability and clustering performance.
- **Performance Evaluation**: Evaluate the clustering results using various metrics such as purity, NMI, and F1 score.

## View the Results of Our Method

To view the final performance results presented in the paper, simply run `RISC.py` in Python.

## Installation

This project requires the following Python libraries:

- `metis`: For graph partitioning.
- `numpy`: For numerical computation.
- `sklearn`: For evaluation metrics.
- `collections`: For counting operations.

You can install the required dependencies using the following command:

```bash
pip install metis numpy scikit-learn
