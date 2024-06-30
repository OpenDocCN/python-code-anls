# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_digits_linkage.py`

```
"""
=============================================================================
Various Agglomerative Clustering on a 2D embedding of digits
=============================================================================

An illustration of various linkage options for agglomerative clustering on
a 2D embedding of the digits dataset.

The goal of this example is to show intuitively how the metrics behave, and
not to find good clusters for the digits. This is why the example works on a
2D embedding.

What this example shows us is the "rich getting richer" behavior of
agglomerative clustering that tends to create uneven cluster sizes.

This behavior is pronounced for the average linkage strategy,
which ends up with a couple of clusters having few data points.

The case of single linkage is even more extreme with a very
large cluster covering most digits, an intermediate size (clean)
cluster with mostly zero digits, and all other clusters being drawn
from noise points around the fringes.

The other linkage strategies lead to more evenly distributed
clusters that are therefore likely to be less sensitive to a
random resampling of the dataset.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from time import time  # 导入时间计算模块

import numpy as np  # 导入数值计算库numpy
from matplotlib import pyplot as plt  # 导入绘图库matplotlib

from sklearn import datasets, manifold  # 导入sklearn中的数据集和流形学习模块

digits = datasets.load_digits()  # 载入手写数字数据集
X, y = digits.data, digits.target  # 获取特征数据和标签数据
n_samples, n_features = X.shape  # 获取样本数和特征数

np.random.seed(0)  # 设定随机数种子以保证可重复性


# ----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)  # 计算每个特征的最小值和最大值
    X_red = (X_red - x_min) / (x_max - x_min)  # 归一化数据

    plt.figure(figsize=(6, 4))  # 创建一个新的图形窗口
    for digit in digits.target_names:
        plt.scatter(
            *X_red[y == digit].T,  # 根据标签绘制散点图
            marker=f"${digit}$",  # 设定标记为数字本身
            s=50,  # 设定散点的大小
            c=plt.cm.nipy_spectral(labels[y == digit] / 10),  # 设定颜色映射
            alpha=0.5,  # 设定透明度
        )

    plt.xticks([])  # 不显示x轴刻度
    plt.yticks([])  # 不显示y轴刻度
    if title is not None:
        plt.title(title, size=17)  # 如果有标题，则设定标题
    plt.axis("off")  # 关闭坐标轴
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应显示


# ----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")  # 打印计算嵌入的信息
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)  # 使用谱嵌入降维到二维空间
print("Done.")  # 打印完成信息

from sklearn.cluster import AgglomerativeClustering  # 导入凝聚聚类模块

for linkage in ("ward", "average", "complete", "single"):  # 遍历不同的连接方式
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)  # 初始化聚类器
    t0 = time()  # 记录开始时间
    clustering.fit(X_red)  # 对降维后的数据进行聚类
    print("%s :\t%.2fs" % (linkage, time() - t0))  # 打印每种连接方式的耗时信息

    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)  # 绘制聚类结果的可视化图像


plt.show()  # 显示所有图形
```