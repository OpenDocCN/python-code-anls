# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_random_forest_embedding.py`

```
"""
=========================================================
Hashing feature transformation using Totally Random Trees
=========================================================

RandomTreesEmbedding provides a way to map data to a
very high-dimensional, sparse representation, which might
be beneficial for classification.
The mapping is completely unsupervised and very efficient.

This example visualizes the partitions given by several
trees and shows how the transformation can also be used for
non-linear dimensionality reduction or non-linear classification.

Points that are neighboring often share the same leaf of a tree and therefore
share large parts of their hashed representation. This allows to
separate two concentric circles simply based on the principal components
of the transformed data with truncated SVD.

In high-dimensional spaces, linear classifiers often achieve
excellent accuracy. For sparse binary data, BernoulliNB
is particularly well-suited. The bottom row compares the
decision boundary obtained by BernoulliNB in the transformed
space with an ExtraTreesClassifier forests learned on the
original data.
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_circles
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomTreesEmbedding
from sklearn.naive_bayes import BernoulliNB

# make a synthetic dataset
X, y = make_circles(factor=0.5, random_state=0, noise=0.05)

# use RandomTreesEmbedding to transform data
hasher = RandomTreesEmbedding(n_estimators=10, random_state=0, max_depth=3)
X_transformed = hasher.fit_transform(X)

# Visualize result after dimensionality reduction using truncated SVD
svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X_transformed)

# Learn a Naive Bayes classifier on the transformed data
nb = BernoulliNB()
nb.fit(X_transformed, y)

# Learn an ExtraTreesClassifier for comparison
trees = ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=0)
trees.fit(X, y)

# scatter plot of original and reduced data
fig = plt.figure(figsize=(9, 8))

ax = plt.subplot(221)
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k")
ax.set_title("Original Data (2d)")
ax.set_xticks(())
ax.set_yticks(())

ax = plt.subplot(222)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=50, edgecolor="k")
ax.set_title(
    "Truncated SVD reduction (2d) of transformed data (%dd)" % X_transformed.shape[1]
)
ax.set_xticks(())
ax.set_yticks(())

# Plot the decision in original space. For that, we will assign a color
# to each point in the mesh [x_min, x_max]x[y_min, y_max].
h = 0.01
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# transform grid using RandomTreesEmbedding
transformed_grid = hasher.transform(np.c_[xx.ravel(), yy.ravel()])
y_grid_pred = nb.predict_proba(transformed_grid)[:, 1]
# 创建一个子图 ax，位于 2x2 的图中的第三个位置 (从左上角开始数)
ax = plt.subplot(223)

# 设置子图标题为 "Naive Bayes on Transformed data"
ax.set_title("Naive Bayes on Transformed data")

# 绘制填充颜色的网格，使用预测结果 y_grid_pred 重塑为二维网格形状 (xx, yy)
ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))

# 绘制散点图，显示数据集 X 的点，颜色由 y 标签决定，点的大小为 50，边缘颜色为黑色
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k")

# 设置 y 轴范围为 -1.4 到 1.4
ax.set_ylim(-1.4, 1.4)

# 设置 x 轴范围为 -1.4 到 1.4
ax.set_xlim(-1.4, 1.4)

# 设置 x 轴不显示刻度
ax.set_xticks(())

# 设置 y 轴不显示刻度
ax.set_yticks(())

# 使用 ExtraTreesClassifier 对网格进行转换，生成预测概率 y_grid_pred
y_grid_pred = trees.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# 创建一个子图 ax，位于 2x2 的图中的第四个位置
ax = plt.subplot(224)

# 设置子图标题为 "ExtraTrees predictions"
ax.set_title("ExtraTrees predictions")

# 绘制填充颜色的网格，使用预测结果 y_grid_pred 重塑为二维网格形状 (xx, yy)
ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))

# 绘制散点图，显示数据集 X 的点，颜色由 y 标签决定，点的大小为 50，边缘颜色为黑色
ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k")

# 设置 y 轴范围为 -1.4 到 1.4
ax.set_ylim(-1.4, 1.4)

# 设置 x 轴范围为 -1.4 到 1.4
ax.set_xlim(-1.4, 1.4)

# 设置 x 轴不显示刻度
ax.set_xticks(())

# 设置 y 轴不显示刻度
ax.set_yticks(())

# 调整子图布局，确保子图之间的间距合适
plt.tight_layout()

# 显示图形
plt.show()
```