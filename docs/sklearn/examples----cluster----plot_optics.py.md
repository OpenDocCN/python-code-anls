# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_optics.py`

```
"""
===================================
OPTICS聚类算法演示
===================================

.. currentmodule:: sklearn

找到高密度核心样本，并从它们扩展聚类。
此示例使用生成的数据，使得聚类具有不同的密度。

首先使用 :class:`~cluster.OPTICS` 以其 Xi 聚类检测方法，
然后在可达性上设置特定阈值，对应于 :class:`~cluster.DBSCAN`。
我们可以看到，通过在 DBSCAN 中选择不同的阈值，可以恢复 OPTICS 的 Xi 方法的不同聚类。

"""

# 作者：scikit-learn 开发人员
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# 导入 OPTICS 相关模块
from sklearn.cluster import OPTICS, cluster_optics_dbscan

# 生成样本数据

np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + 0.8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + 0.1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + 0.3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

# 初始化 OPTICS 聚类器
clust = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05)

# 运行拟合
clust.fit(X)

# 使用不同的阈值运行 cluster_optics_dbscan 函数
labels_050 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=0.5,
)
labels_200 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=2,
)

space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

# 创建绘图窗口
plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# 绘制可达性图
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in enumerate(colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
ax1.set_ylabel("Reachability (epsilon distance)")
ax1.set_title("Reachability Plot")

# 绘制 OPTICS 图
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in enumerate(colors):
    Xk = X[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.1)
ax2.set_title("Automatic Clustering\nOPTICS")

# 绘制 DBSCAN 在 0.5 时的图
colors = ["g.", "r.", "b.", "c."]
for klass, color in enumerate(colors):
    Xk = X[labels_050 == klass]
    # 在 ax3 对象上绘制散点图，使用 Xk 的第一列作为 x 坐标，第二列作为 y 坐标，指定颜色和透明度为 0.3
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# 在ax3中绘制DBSCAN聚类结果中被标记为-1的样本点，使用黑色"+"标记，透明度为0.1
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], "k+", alpha=0.1)
# 设置子图ax3的标题为"Clustering at 0.5 epsilon cut\nDBSCAN"

# 定义颜色列表，用于不同类别的可视化表示
colors = ["g.", "m.", "y.", "c."]
# 遍历每个类别和对应的颜色，绘制在ax4中
for klass, color in enumerate(colors):
    # 提取属于当前类别klass的数据点
    Xk = X[labels_200 == klass]
    # 在ax4中绘制当前类别的数据点，使用指定的颜色和透明度0.3
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# 在ax4中绘制DBSCAN聚类结果中被标记为-1的样本点，使用黑色"+"标记，透明度为0.1
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], "k+", alpha=0.1)
# 设置子图ax4的标题为"Clustering at 2.0 epsilon cut\nDBSCAN"

# 调整子图布局，确保图形紧凑显示
plt.tight_layout()
# 显示绘制的图形
plt.show()
```