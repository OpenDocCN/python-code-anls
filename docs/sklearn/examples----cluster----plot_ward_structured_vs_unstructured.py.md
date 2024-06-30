# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_ward_structured_vs_unstructured.py`

```
"""
===========================================================
Hierarchical clustering: structured vs unstructured ward
===========================================================

Example builds a swiss roll dataset and runs
hierarchical clustering on their position.

For more information, see :ref:`hierarchical_clustering`.

In a first step, the hierarchical clustering is performed without connectivity
constraints on the structure and is solely based on distance, whereas in
a second step the clustering is restricted to the k-Nearest Neighbors
graph: it's a hierarchical clustering with structure prior.

Some of the clusters learned without connectivity constraints do not
respect the structure of the swiss roll and extend across different folds of
the manifolds. On the opposite, when opposing connectivity constraints,
the clusters form a nice parcellation of the swiss roll.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time as time  # 导入time模块，用作时间测量

# The following import is required
# for 3D projection to work with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

import numpy as np  # 导入NumPy库，用于数组操作

# %%
# Generate data
# -------------
#
# We start by generating the Swiss Roll dataset.
from sklearn.datasets import make_swiss_roll

n_samples = 1500
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise=noise)
# Make it thinner
X[:, 1] *= 0.5  # 将数据集中第二列的值减半，使其变窄

# %%
# Compute clustering
# ------------------
#
# We perform AgglomerativeClustering which comes under Hierarchical Clustering
# without any connectivity constraints.

from sklearn.cluster import AgglomerativeClustering

print("Compute unstructured hierarchical clustering...")
st = time.time()  # 记录开始时间
ward = AgglomerativeClustering(n_clusters=6, linkage="ward").fit(X)
elapsed_time = time.time() - st  # 计算时间差，即执行时间
label = ward.labels_  # 获取聚类结果的标签
print(f"Elapsed time: {elapsed_time:.2f}s")
print(f"Number of points: {label.size}")

# %%
# Plot result
# -----------
# Plotting the unstructured hierarchical clusters.

import matplotlib.pyplot as plt

fig1 = plt.figure()  # 创建图形对象
ax1 = fig1.add_subplot(111, projection="3d", elev=7, azim=-80)  # 添加3D子图
ax1.set_position([0, 0, 0.95, 1])  # 设置子图位置和大小
for l in np.unique(label):
    ax1.scatter(
        X[label == l, 0],
        X[label == l, 1],
        X[label == l, 2],
        color=plt.cm.jet(float(l) / np.max(label + 1)),
        s=20,
        edgecolor="k",
    )  # 绘制散点图
_ = fig1.suptitle(f"Without connectivity constraints (time {elapsed_time:.2f}s)")  # 设置图标题

# %%
# We are defining k-Nearest Neighbors with 10 neighbors
# -----------------------------------------------------

from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)  # 构建k-Nearest Neighbors图

# %%
# Compute clustering
# ------------------
#
# We perform AgglomerativeClustering again with connectivity constraints.

print("Compute structured hierarchical clustering...")
st = time.time()  # 记录开始时间
ward = AgglomerativeClustering(
    n_clusters=6, connectivity=connectivity, linkage="ward"
).fit(X)  # 使用连接约束进行层次聚类
elapsed_time = time.time() - st  # 计算执行时间
# 定义变量 label 为聚类结果的标签
label = ward.labels_

# 打印经过的时间，保留两位小数
print(f"Elapsed time: {elapsed_time:.2f}s")

# 打印标签的数量
print(f"Number of points: {label.size}")

# %%
# 绘图结果
# -----------
#
# 绘制结构化的层次聚类结果。

# 创建一个新的图形对象
fig2 = plt.figure()

# 在图形对象上添加一个子图，使用3D投影
ax2 = fig2.add_subplot(121, projection="3d", elev=7, azim=-80)

# 设置子图的位置和大小
ax2.set_position([0, 0, 0.95, 1])

# 对每个唯一的聚类标签进行循环
for l in np.unique(label):
    # 绘制散点图，根据聚类标签选择数据点和颜色
    ax2.scatter(
        X[label == l, 0],
        X[label == l, 1],
        X[label == l, 2],
        color=plt.cm.jet(float(l) / np.max(label + 1)),
        s=20,
        edgecolor="k",
    )

# 设置图形的总标题，显示带有连接约束的层次聚类结果的时间
fig2.suptitle(f"With connectivity constraints (time {elapsed_time:.2f}s)")

# 展示图形
plt.show()
```