# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_dbscan.py`

```
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds core
samples in regions of high density and expands clusters from them. This
algorithm is good for data which contains clusters of similar density.

See the :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py` example
for a demo of different clustering algorithms on 2D datasets.

"""

# %%
# Data generation
# ---------------
#
# We use :class:`~sklearn.datasets.make_blobs` to create 3 synthetic clusters.

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 定义三个中心点，用于生成合成数据集
centers = [[1, 1], [-1, -1], [1, -1]]

# 使用 make_blobs 生成合成数据集，包括750个样本，以 centers 为中心，标准差为0.4
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

# 对数据进行标准化处理
X = StandardScaler().fit_transform(X)

# %%
# We can visualize the resulting data:

import matplotlib.pyplot as plt

# 绘制散点图展示生成的数据集
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# %%
# Compute DBSCAN
# --------------
#
# One can access the labels assigned by :class:`~sklearn.cluster.DBSCAN` using
# the `labels_` attribute. Noisy samples are given the label math:`-1`.

import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

# 使用 DBSCAN 算法拟合数据
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# 计算得到的簇数，忽略噪声点
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# %%
# Clustering algorithms are fundamentally unsupervised learning methods.
# However, since :class:`~sklearn.datasets.make_blobs` gives access to the true
# labels of the synthetic clusters, it is possible to use evaluation metrics
# that leverage this "supervised" ground truth information to quantify the
# quality of the resulting clusters. Examples of such metrics are the
# homogeneity, completeness, V-measure, Rand-Index, Adjusted Rand-Index and
# Adjusted Mutual Information (AMI).
#
# If the ground truth labels are not known, evaluation can only be performed
# using the model results itself. In that case, the Silhouette Coefficient comes
# in handy.
#
# For more information, see the
# :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`
# example or the :ref:`clustering_evaluation` module.

# 输出聚类结果的多个评估指标
print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
print(
    "Adjusted Mutual Information:"
    f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
)
print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

# %%
# Plot results
# ------------
#
# 将标签中的唯一值存储在集合中
unique_labels = set(labels)
# 创建一个与标签相同形状的布尔数组，用于标记核心样本
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# 生成一组颜色，用于表示不同的簇
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# 遍历唯一标签和颜色，为每个簇绘制样本点
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 用黑色表示噪声点
        col = [0, 0, 0, 1]

    # 标记属于当前簇的样本
    class_member_mask = labels == k

    # 绘制核心样本的大点
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    # 绘制非核心样本的小点
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

# 设置图表标题
plt.title(f"Estimated number of clusters: {n_clusters_}")
# 显示图表
plt.show()
```