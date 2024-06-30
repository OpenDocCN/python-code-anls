# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_kmeans_plusplus.py`

```
"""
===========================================================
An example of K-Means++ initialization
===========================================================

An example to show the output of the :func:`sklearn.cluster.kmeans_plusplus`
function for generating initial seeds for clustering.

K-Means++ is used as the default initialization for :ref:`k_means`.

"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 导入 kmeans_plusplus 函数用于 K-Means++ 初始化，make_blobs 用于生成样本数据集
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs

# 生成样本数据
n_samples = 4000  # 样本数
n_components = 4   # 簇的数量

# 使用 make_blobs 生成具有指定中心和标准差的样本数据集
X, y_true = make_blobs(
    n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0
)

# 将数据集 X 的每个样本的特征颠倒顺序
X = X[:, ::-1]

# 使用 kmeans_plusplus 函数计算 K-Means++ 初始化的种子中心点
centers_init, indices = kmeans_plusplus(X, n_clusters=4, random_state=0)

# 绘制初始化种子点和样本数据的散点图
plt.figure(1)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]

# 根据不同的簇标签将样本数据分组并绘制不同颜色的散点图
for k, col in enumerate(colors):
    cluster_data = y_true == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)

# 绘制初始化种子点的散点图，并使用蓝色标识
plt.scatter(centers_init[:, 0], centers_init[:, 1], c="b", s=50)

# 设置图表标题和坐标轴刻度
plt.title("K-Means++ Initialization")
plt.xticks([])
plt.yticks([])

# 显示图表
plt.show()
```