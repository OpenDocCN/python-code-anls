# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_agglomerative_dendrogram.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
"""
=========================================
Plot Hierarchical Clustering Dendrogram
=========================================
This example plots the corresponding dendrogram of a hierarchical clustering
using AgglomerativeClustering and the dendrogram method available in scipy.

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    
    # 计算每个节点下的样本数
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # 叶子节点
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # 构建链接矩阵，包括子节点索引、距离和样本计数
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # 绘制对应的树状图
    dendrogram(linkage_matrix, **kwargs)


iris = load_iris()
X = iris.data

# 设置 distance_threshold=0 确保计算完整的树
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# 绘制树状图的前三级
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
```