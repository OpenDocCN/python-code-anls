# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_cluster_iris.py`

```
"""
=========================================================
K-means Clustering
=========================================================

The plot shows:

- top left: What a K-means algorithm would yield using 8 clusters.

- top right: What using three clusters would deliver.

- bottom left: What the effect of a bad initialization is
  on the classification process: By setting n_init to only 1
  (default is 10), the amount of times that the algorithm will
  be run with different centroid seeds is reduced.

- bottom right: The ground truth.

"""

# 代码来源: Gaël Varoquaux
# 由 Jaques Grobler 修改用于文档
# SPDX-License-Identifier: BSD-3-Clause

# 导入 matplotlib 库用于绘图
import matplotlib.pyplot as plt

# 尽管下面的导入没有直接使用，但对于 matplotlib < 3.2，这是必需的
import mpl_toolkits.mplot3d  # noqa: F401

# 导入 numpy 库
import numpy as np

# 导入 sklearn 库中的 datasets 和 KMeans 类
from sklearn import datasets
from sklearn.cluster import KMeans

# 设定随机种子，以便结果可重复
np.random.seed(5)

# 载入鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 提取特征数据
y = iris.target  # 提取标签数据

# 定义 KMeans 聚类器的列表，每个聚类器有不同的设置
estimators = [
    ("k_means_iris_8", KMeans(n_clusters=8)),  # 使用 8 个簇的 KMeans
    ("k_means_iris_3", KMeans(n_clusters=3)),  # 使用 3 个簇的 KMeans
    ("k_means_iris_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),  # 使用随机初始化的 KMeans
]

# 创建一个图形窗口，设置大小为 10x8 英寸
fig = plt.figure(figsize=(10, 8))

# 定义子图的标题
titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]

# 遍历每个聚类器和标题，生成对应的子图
for idx, ((name, est), title) in enumerate(zip(estimators, titles)):
    ax = fig.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134)  # 添加 3D 投影子图
    est.fit(X)  # 对数据进行聚类
    labels = est.labels_  # 获取聚类标签

    # 绘制散点图，以特定特征作为坐标轴，不同类别用不同颜色表示
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor="k")

    # 设置坐标轴的刻度标签为空
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    # 设置坐标轴的标签名称
    ax.set_xlabel("Petal width")
    ax.set_ylabel("Sepal length")
    ax.set_zlabel("Petal length")

    # 设置子图标题
    ax.set_title(title)

# 绘制真实分类的子图
ax = fig.add_subplot(2, 2, 4, projection="3d", elev=48, azim=134)

# 在图中添加文本标签，表示不同类别的均值点
for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 3].mean(),
        X[y == label, 0].mean(),
        X[y == label, 2].mean() + 2,
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
    )

# 绘制散点图，表示真实的分类情况
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor="k")

# 设置坐标轴的刻度标签为空
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# 设置坐标轴的标签名称
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")

# 设置子图标题
ax.set_title("Ground Truth")

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.25, hspace=0.25)

# 显示绘制的所有子图
plt.show()
```