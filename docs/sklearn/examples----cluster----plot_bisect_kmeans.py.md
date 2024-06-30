# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_bisect_kmeans.py`

```
"""
=============================================================
Bisecting K-Means and Regular K-Means Performance Comparison
=============================================================

This example shows differences between Regular K-Means algorithm and Bisecting K-Means.

While K-Means clusterings are different when increasing n_clusters,
Bisecting K-Means clustering builds on top of the previous ones. As a result, it
tends to create clusters that have a more regular large-scale structure. This
difference can be visually observed: for all numbers of clusters, there is a
dividing line cutting the overall data cloud in two for BisectingKMeans, which is not
present for regular K-Means.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图

from sklearn.cluster import BisectingKMeans, KMeans  # 导入 BisectingKMeans 和 KMeans 算法类
from sklearn.datasets import make_blobs  # 导入 make_blobs 数据生成器函数

print(__doc__)  # 打印本示例的文档字符串


# Generate sample data
n_samples = 10000  # 生成数据的样本数
random_state = 0  # 随机数种子，用于复现随机数据的生成过程

X, _ = make_blobs(n_samples=n_samples, centers=2, random_state=random_state)
# 生成样本数据，centers=2 表示生成两个簇的数据，_ 为生成的类别标签，这里不使用

# Number of cluster centers for KMeans and BisectingKMeans
n_clusters_list = [4, 8, 16]  # 分别尝试的簇数列表

# Algorithms to compare
clustering_algorithms = {
    "Bisecting K-Means": BisectingKMeans,  # 用于比较的算法之一：Bisecting K-Means
    "K-Means": KMeans,  # 用于比较的算法之一：K-Means
}

# Make subplots for each variant
fig, axs = plt.subplots(
    len(clustering_algorithms), len(n_clusters_list), figsize=(12, 5)
)
# 创建子图布局，每个算法的不同簇数变体都有自己的子图，图像大小为 12x5

axs = axs.T  # 转置 axs 数组，使得列对应不同的算法，行对应不同的簇数

for i, (algorithm_name, Algorithm) in enumerate(clustering_algorithms.items()):
    for j, n_clusters in enumerate(n_clusters_list):
        algo = Algorithm(n_clusters=n_clusters, random_state=random_state, n_init=3)
        # 使用指定的算法和参数初始化算法对象
        algo.fit(X)  # 对数据 X 进行聚类

        centers = algo.cluster_centers_  # 获取聚类中心坐标

        axs[j, i].scatter(X[:, 0], X[:, 1], s=10, c=algo.labels_)
        # 绘制散点图，显示数据点，颜色按照聚类结果着色
        axs[j, i].scatter(centers[:, 0], centers[:, 1], c="r", s=20)
        # 绘制聚类中心点，颜色为红色，大小为 20

        axs[j, i].set_title(f"{algorithm_name} : {n_clusters} clusters")
        # 设置子图标题，显示算法名称和簇数目


# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()  # 隐藏子图的标签（刻度标签）
    ax.set_xticks([])  # 设置 x 轴刻度为空列表，即不显示刻度
    ax.set_yticks([])  # 设置 y 轴刻度为空列表，即不显示刻度

plt.show()  # 显示所有子图
```