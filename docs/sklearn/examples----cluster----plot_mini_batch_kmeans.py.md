# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_mini_batch_kmeans.py`

```
# %%
# Generate the data
# -----------------
#
# We start by generating the blobs of data to be clustered.
import numpy as np
from sklearn.datasets import make_blobs

np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)

# Generate synthetic data points arranged in blobs centered around 'centers'
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

# %%
# Compute clustering with KMeans
# ------------------------------

import time
from sklearn.cluster import KMeans

# Initialize KMeans with k-means++ initialization and attempt 10 times
k_means = KMeans(init="k-means++", n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0  # Measure time taken for KMeans clustering

# %%
# Compute clustering with MiniBatchKMeans
# ---------------------------------------

from sklearn.cluster import MiniBatchKMeans

# Initialize MiniBatchKMeans with k-means++ initialization and other parameters
mbk = MiniBatchKMeans(
    init="k-means++",
    n_clusters=3,
    batch_size=batch_size,
    n_init=10,
    max_no_improvement=10,
    verbose=0,
)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0  # Measure time taken for MiniBatchKMeans clustering

# %%
# Establishing parity between clusters
# ------------------------------------
#
# Match the cluster centers of KMeans with those of MiniBatchKMeans
# based on closest distance.
from sklearn.metrics.pairwise import pairwise_distances_argmin

k_means_cluster_centers = k_means.cluster_centers_
order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
mbk_means_cluster_centers = mbk.cluster_centers_[order]

# Assign labels to data points based on closest cluster center
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

# %%
# Plotting the results
# --------------------

import matplotlib.pyplot as plt

# Create a figure for plotting
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

# Plot for KMeans clustering results
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_batch, k_means.inertia_))

# MiniBatchKMeans subplot setup is continued in the next part of the exercise.
# 循环遍历每个簇的索引和颜色
for k, col in zip(range(n_clusters), colors):
    # 找出属于当前簇的数据点的布尔索引
    my_members = mbk_means_labels == k
    # 获取当前簇的中心点坐标
    cluster_center = mbk_means_cluster_centers[k]
    # 在图上绘制属于当前簇的数据点，用白色标记，填充色为给定颜色
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    # 在图上绘制当前簇的中心点，用圆圈标记，填充色为给定颜色，边缘色为黑色，大小为6
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
    
# 设置图的标题为 "MiniBatchKMeans"
ax.set_title("MiniBatchKMeans")
# 设置图的 x 轴刻度为空
ax.set_xticks(())
# 设置图的 y 轴刻度为空
ax.set_yticks(())

# 在图上添加文本，显示训练时间和惯性值
plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_mini_batch, mbk.inertia_))

# 初始化布尔数组 different，全部初始化为 False
different = mbk_means_labels == 4
# 将图的当前子图设置为第三个子图
ax = fig.add_subplot(1, 3, 3)

# 循环遍历每个簇的索引
for k in range(n_clusters):
    # 计算与 KMeans 和 MiniBatchKMeans 簇分配不同的布尔数组
    different += (k_means_labels == k) != (mbk_means_labels == k)

# 计算相同的数据点的布尔数组
identical = np.logical_not(different)
# 在图上绘制相同的数据点，用白色标记，填充色为浅灰色
ax.plot(X[identical, 0], X[identical, 1], "w", markerfacecolor="#bbbbbb", marker=".")
# 在图上绘制不同的数据点，用白色标记，填充色为洋红色
ax.plot(X[different, 0], X[different, 1], "w", markerfacecolor="m", marker=".")
# 设置图的标题为 "Difference"
ax.set_title("Difference")
# 设置图的 x 轴刻度为空
ax.set_xticks(())
# 设置图的 y 轴刻度为空

# 展示图形
plt.show()
```