# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_birch_vs_minibatchkmeans.py`

```
"""
=================================
Compare BIRCH and MiniBatchKMeans
=================================

This example compares the timing of BIRCH (with and without the global
clustering step) and MiniBatchKMeans on a synthetic dataset having
25,000 samples and 2 features generated using make_blobs.

Both ``MiniBatchKMeans`` and ``BIRCH`` are very scalable algorithms and could
run efficiently on hundreds of thousands or even millions of datapoints. We
chose to limit the dataset size of this example in the interest of keeping
our Continuous Integration resource usage reasonable but the interested
reader might enjoy editing this script to rerun it with a larger value for
`n_samples`.

If ``n_clusters`` is set to None, the data is reduced from 25,000
samples to a set of 158 clusters. This can be viewed as a preprocessing
step before the final (global) clustering step that further reduces these
158 clusters to 100 clusters.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import cycle  # 导入循环迭代工具函数
from time import time  # 导入时间计算函数

import matplotlib.colors as colors  # 导入 matplotlib 的颜色模块
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图模块
import numpy as np  # 导入数值计算模块 numpy
from joblib import cpu_count  # 导入并行计算模块中的 CPU 数量函数

from sklearn.cluster import Birch, MiniBatchKMeans  # 导入 BIRCH 和 MiniBatchKMeans 聚类算法
from sklearn.datasets import make_blobs  # 导入生成聚类数据的函数 make_blobs

# Generate centers for the blobs so that it forms a 10 X 10 grid.
xx = np.linspace(-22, 22, 10)  # 在 x 方向生成等间距的数列
yy = np.linspace(-22, 22, 10)  # 在 y 方向生成等间距的数列
xx, yy = np.meshgrid(xx, yy)  # 生成网格坐标矩阵
n_centers = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis]))  # 将坐标矩阵展平并堆叠成中心点坐标

# Generate blobs to do a comparison between MiniBatchKMeans and BIRCH.
X, y = make_blobs(n_samples=25000, centers=n_centers, random_state=0)  # 生成具有中心点的聚类数据集

# Use all colors that matplotlib provides by default.
colors_ = cycle(colors.cnames.keys())  # 获取 matplotlib 提供的所有颜色，并创建颜色迭代器

fig = plt.figure(figsize=(12, 4))  # 创建绘图窗口
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)  # 调整子图之间的间距和边界

# Compute clustering with BIRCH with and without the final clustering step
# and plot.
birch_models = [
    Birch(threshold=1.7, n_clusters=None),  # 创建没有全局聚类步骤的 BIRCH 对象
    Birch(threshold=1.7, n_clusters=100),  # 创建包含全局聚类步骤的 BIRCH 对象
]
final_step = ["without global clustering", "with global clustering"]  # 对应两种不同的聚类步骤

for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
    t = time()  # 记录开始时间
    birch_model.fit(X)  # 使用 BIRCH 对数据 X 进行聚类
    print("BIRCH %s as the final step took %0.2f seconds" % (info, (time() - t)))  # 打印聚类时间信息

    # Plot result
    labels = birch_model.labels_  # 获取聚类标签
    centroids = birch_model.subcluster_centers_  # 获取子簇中心
    n_clusters = np.unique(labels).size  # 计算聚类的簇数
    print("n_clusters : %d" % n_clusters)  # 打印簇的数量信息

    ax = fig.add_subplot(1, 3, ind + 1)  # 添加子图
    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k  # 创建标签掩码
        ax.scatter(X[mask, 0], X[mask, 1], c="w", edgecolor=col, marker=".", alpha=0.5)  # 绘制数据点
        if birch_model.n_clusters is None:
            ax.scatter(this_centroid[0], this_centroid[1], marker="+", c="k", s=25)  # 绘制子簇中心点
    ax.set_ylim([-25, 25])  # 设置 y 轴的范围
    ax.set_xlim([-25, 25])  # 设置 x 轴的范围
    ax.set_autoscaley_on(False)  # 禁止 y 轴自动缩放
    ax.set_title("BIRCH %s" % info)  # 设置子图标题
# 使用 MiniBatchKMeans 进行聚类计算。
mbk = MiniBatchKMeans(
    init="k-means++",  # 使用 k-means++ 初始化方法
    n_clusters=100,  # 聚类数量设定为 100
    batch_size=256 * cpu_count(),  # 批处理大小为 256 乘以 CPU 核心数
    n_init=10,  # 进行初始化聚类中心的次数
    max_no_improvement=10,  # 允许连续多少个迭代次数内没有改善的情况
    verbose=0,  # 不输出过程信息
    random_state=0,  # 随机数种子设定为 0，保证可重复性
)

t0 = time()  # 记录开始时间
mbk.fit(X)  # 对数据 X 进行 MiniBatchKMeans 聚类
t_mini_batch = time() - t0  # 计算 MiniBatchKMeans 运行所需时间
print("Time taken to run MiniBatchKMeans %0.2f seconds" % t_mini_batch)

# 提取 MiniBatchKMeans 所有独特的标签
mbk_means_labels_unique = np.unique(mbk.labels_)

# 在图形中添加子图，位置为 1 行 3 列中的第 3 列
ax = fig.add_subplot(1, 3, 3)

# 对于每个质心，k 值和颜色 col，绘制相应的散点图
for this_centroid, k, col in zip(mbk.cluster_centers_, range(n_clusters), colors_):
    mask = mbk.labels_ == k  # 创建一个布尔掩码来选择属于当前质心的点
    ax.scatter(X[mask, 0], X[mask, 1], marker=".", c="w", edgecolor=col, alpha=0.5)  # 绘制数据点
    ax.scatter(this_centroid[0], this_centroid[1], marker="+", c="k", s=25)  # 绘制质心点

# 设置 x 和 y 轴的范围
ax.set_xlim([-25, 25])
ax.set_ylim([-25, 25])

# 设置子图的标题为 "MiniBatchKMeans"
ax.set_title("MiniBatchKMeans")

# 关闭 y 轴的自动缩放
ax.set_autoscaley_on(False)

# 展示图形
plt.show()
```