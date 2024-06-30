# `D:\src\scipysrc\scikit-learn\examples\datasets\plot_iris_dataset.py`

```
"""
================
The Iris Dataset
================
This data sets consists of 3 different types of irises'
(Setosa, Versicolour, and Virginica) petal and sepal
length, stored in a 150x4 numpy.ndarray

The rows being the samples and the columns being:
Sepal Length, Sepal Width, Petal Length and Petal Width.

The below plot uses the first two features.
See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.

"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Loading the iris dataset
# ------------------------
# 导入 sklearn 库中的 datasets 模块，用于加载标准的鸢尾花数据集
from sklearn import datasets

# 加载鸢尾花数据集并将其存储在变量 iris 中
iris = datasets.load_iris()

# %%
# Scatter Plot of the Iris dataset
# --------------------------------
# 导入 matplotlib 库中的 pyplot 模块，用于绘制散点图
import matplotlib.pyplot as plt

# 创建一个新的图形和子图
_, ax = plt.subplots()

# 绘制散点图，以鸢尾花数据集中的第一列（萼片长度）和第二列（萼片宽度）作为 x 和 y 坐标，
# 根据花的类型（iris.target）着色
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)

# 设置 x 轴和 y 轴的标签为数据集中各个特征的名称
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])

# 添加图例，显示不同类型的鸢尾花（Setosa, Versicolour, Virginica），放置在图的右下角
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

# %%
# Each point in the scatter plot refers to one of the 150 iris flowers
# in the dataset, with the color indicating their respective type
# (Setosa, Versicolour, and Virginica).
# You can already see a pattern regarding the Setosa type, which is
# easily identifiable based on its short and wide sepal. Only
# considering these 2 dimensions, sepal width and length, there's still
# overlap between the Versicolor and Virginica types.

# %%
# Plot a PCA representation
# -------------------------
# Let's apply a Principal Component Analysis (PCA) to the iris dataset
# and then plot the irises across the first three PCA dimensions.
# This will allow us to better differentiate between the three types!

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

# 导入 sklearn 库中的 PCA 模块，用于执行主成分分析
from sklearn.decomposition import PCA

# 创建一个新的图形对象，设置大小为 8x6
fig = plt.figure(1, figsize=(8, 6))

# 在图形上添加一个 3D 子图
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

# 对原始数据应用 PCA 转换为 3 个主成分，并获取转换后的数据 X_reduced
X_reduced = PCA(n_components=3).fit_transform(iris.data)

# 绘制散点图，使用前三个主成分作为 x, y, z 坐标，根据花的类型（iris.target）着色
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

# 设置子图的标题
ax.set_title("First three PCA dimensions")

# 设置 x 轴、y 轴和 z 轴的标签
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])

# 显示图形
plt.show()

# %%
# PCA will create 3 new features that are a linear combination of the
# 4 original features. In addition, this transform maximizes the variance.
# With this transformation, we see that we can identify each species using
# only the first feature (i.e. first eigenvalues).
```