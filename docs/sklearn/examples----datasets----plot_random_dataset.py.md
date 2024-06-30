# `D:\src\scipysrc\scikit-learn\examples\datasets\plot_random_dataset.py`

```
"""
==============================================
Plot randomly generated classification dataset
==============================================

This example plots several randomly generated classification datasets.
For easy visualization, all datasets have 2 features, plotted on the x and y
axis. The color of each point represents its class label.

The first 4 plots use the :func:`~sklearn.datasets.make_classification` with
different numbers of informative features, clusters per class and classes.
The final 2 plots use :func:`~sklearn.datasets.make_blobs` and
:func:`~sklearn.datasets.make_gaussian_quantiles`.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入生成数据集的函数库
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles

# 设置整体图的大小和子图之间的间距
plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)

# 第一个子图
plt.subplot(321)
plt.title("One informative feature, one cluster per class", fontsize="small")
# 使用 make_classification 生成数据集，指定特征数、无关特征数、信息特征数、每类簇数
X1, Y1 = make_classification(
    n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1
)
# 绘制散点图
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

# 第二个子图
plt.subplot(322)
plt.title("Two informative features, one cluster per class", fontsize="small")
# 使用 make_classification 生成数据集，指定特征数、无关特征数、信息特征数、每类簇数
X1, Y1 = make_classification(
    n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1
)
# 绘制散点图
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

# 第三个子图
plt.subplot(323)
plt.title("Two informative features, two clusters per class", fontsize="small")
# 使用 make_classification 生成数据集，指定特征数、无关特征数、信息特征数
X2, Y2 = make_classification(n_features=2, n_redundant=0, n_informative=2)
# 绘制散点图
plt.scatter(X2[:, 0], X2[:, 1], marker="o", c=Y2, s=25, edgecolor="k")

# 第四个子图
plt.subplot(324)
plt.title("Multi-class, two informative features, one cluster", fontsize="small")
# 使用 make_classification 生成数据集，指定特征数、无关特征数、信息特征数、每类簇数、类别数
X1, Y1 = make_classification(
    n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3
)
# 绘制散点图
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

# 第五个子图
plt.subplot(325)
plt.title("Three blobs", fontsize="small")
# 使用 make_blobs 生成数据集，指定特征数、中心数
X1, Y1 = make_blobs(n_features=2, centers=3)
# 绘制散点图
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

# 第六个子图
plt.subplot(326)
plt.title("Gaussian divided into three quantiles", fontsize="small")
# 使用 make_gaussian_quantiles 生成数据集，指定特征数、类别数
X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3)
# 绘制散点图
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")

# 显示图形
plt.show()
```