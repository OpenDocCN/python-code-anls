# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_pca_iris.py`

```
"""
=========================================================
PCA example with Iris Data-set
=========================================================

Principal Component Analysis applied to the Iris dataset.

See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.

"""

# 代码来源：Gaël Varoquaux
# 使用 BSD-3-Clause 许可证

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图

# 用于在 matplotlib < 3.2 版本中进行 3D 投影的未使用但需要的导入
import mpl_toolkits.mplot3d  # noqa: F401

import numpy as np  # 导入 numpy 库，用于数值计算

from sklearn import datasets, decomposition  # 导入 sklearn 库中的 datasets 和 decomposition 模块

np.random.seed(5)  # 设置随机种子以保证结果的可复现性

iris = datasets.load_iris()  # 载入鸢尾花数据集
X = iris.data  # 将数据集中的数据部分赋给变量 X
y = iris.target  # 将数据集中的标签部分赋给变量 y

fig = plt.figure(1, figsize=(4, 3))  # 创建一个新的图形窗口，指定尺寸为 (4, 3)
plt.clf()  # 清空当前图形

ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)  # 在图形窗口中添加一个 3D 子图
ax.set_position([0, 0, 0.95, 1])  # 设置子图的位置和大小

plt.cla()  # 清除当前图形的坐标轴及内容

pca = decomposition.PCA(n_components=3)  # 创建一个 PCA 对象，设定要保留的主成分数量为 3
pca.fit(X)  # 对数据集 X 进行 PCA 拟合
X = pca.transform(X)  # 将原始数据 X 转换到 PCA 空间中

# 对每个类别的数据点进行标注
for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 0].mean(),  # 根据标签选择对应类别的数据，计算其在第一个主成分上的均值
        X[y == label, 1].mean() + 1.5,  # 计算其在第二个主成分上的均值，并稍微提升以避免重叠
        X[y == label, 2].mean(),  # 计算其在第三个主成分上的均值
        name,  # 类别名称
        horizontalalignment="center",  # 水平对齐方式为中心
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),  # 设置文本框的样式
    )

# 重新排列标签以匹配聚类结果的颜色
y = np.choose(y, [1, 2, 0]).astype(float)  # 重新排列标签顺序，以便颜色与聚类结果匹配
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")  # 绘制散点图，使用不同颜色表示不同类别

ax.xaxis.set_ticklabels([])  # 设置 x 轴刻度标签为空
ax.yaxis.set_ticklabels([])  # 设置 y 轴刻度标签为空
ax.zaxis.set_ticklabels([])  # 设置 z 轴刻度标签为空

plt.show()  # 显示图形
```