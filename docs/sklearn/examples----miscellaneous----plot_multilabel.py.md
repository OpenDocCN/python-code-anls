# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_multilabel.py`

```
"""
=========================
Multilabel classification
=========================

This example simulates a multi-label document classification problem. The
dataset is generated randomly based on the following process:

    - pick the number of labels: n ~ Poisson(n_labels)
    - n times, choose a class c: c ~ Multinomial(theta)
    - pick the document length: k ~ Poisson(length)
    - k times, choose a word: w ~ Multinomial(theta_c)

In the above process, rejection sampling is used to make sure that n is more
than 2, and that the document length is never zero. Likewise, we reject classes
which have already been chosen.  The documents that are assigned to both
classes are plotted surrounded by two colored circles.

The classification is performed by projecting to the first two principal
components found by PCA and CCA for visualisation purposes, followed by using
the :class:`~sklearn.multiclass.OneVsRestClassifier` metaclassifier using two
SVCs with linear kernels to learn a discriminative model for each class.
Note that PCA is used to perform an unsupervised dimensionality reduction,
while CCA is used to perform a supervised one.

Note: in the plot, "unlabeled samples" does not mean that we don't know the
labels (as in semi-supervised learning) but that the samples simply do *not*
have a label.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 数学计算库

from sklearn.cross_decomposition import CCA  # 导入 CCA 模块
from sklearn.datasets import make_multilabel_classification  # 导入生成多标签分类数据集的函数
from sklearn.decomposition import PCA  # 导入 PCA 模块
from sklearn.multiclass import OneVsRestClassifier  # 导入 OneVsRest 分类器
from sklearn.svm import SVC  # 导入支持向量分类器（SVC）


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # 获取分类器的分隔超平面
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # 确保直线足够长
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)  # 使用 PCA 进行无监督降维
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)  # 使用 CCA 进行有监督降维
    else:
        raise ValueError("Unsupported transform type")  # 抛出异常，不支持的转换类型

    min_x = np.min(X[:, 0])  # 获取 X 第一列的最小值
    max_x = np.max(X[:, 0])  # 获取 X 第一列的最大值

    min_y = np.min(X[:, 1])  # 获取 X 第二列的最小值
    max_y = np.max(X[:, 1])  # 获取 X 第二列的最大值

    classif = OneVsRestClassifier(SVC(kernel="linear"))  # 使用线性核的 OneVsRest 分类器
    classif.fit(X, Y)  # 训练分类器

    plt.subplot(2, 2, subplot)  # 创建子图
    plt.title(title)  # 设置子图标题

    zero_class = np.where(Y[:, 0])  # 获取第一个类别的索引
    one_class = np.where(Y[:, 1])  # 获取第二个类别的索引
    plt.scatter(X[:, 0], X[:, 1], s=40, c="gray", edgecolors=(0, 0, 0))  # 绘制灰色点
    plt.scatter(
        X[zero_class, 0],
        X[zero_class, 1],
        s=160,
        edgecolors="b",
        facecolors="none",
        linewidths=2,
        label="Class 1",
    )  # 绘制第一个类别的点
    plt.scatter(
        X[one_class, 0],
        X[one_class, 1],
        s=80,
        edgecolors="orange",
        facecolors="none",
        linewidths=2,
        label="Class 2",
    )  # 绘制第二个类别的点
    )

    # 绘制第一个分类器的超平面
    plot_hyperplane(
        classif.estimators_[0], min_x, max_x, "k--", "Boundary\nfor class 1"
    )
    # 绘制第二个分类器的超平面
    plot_hyperplane(
        classif.estimators_[1], min_x, max_x, "k-.", "Boundary\nfor class 2"
    )
    # 设置 x 轴和 y 轴的刻度为空
    plt.xticks(())
    plt.yticks(())

    # 设置 x 和 y 轴的范围
    plt.xlim(min_x - 0.5 * max_x, max_x + 0.5 * max_x)
    plt.ylim(min_y - 0.5 * max_y, max_y + 0.5 * max_y)
    # 如果是第二个子图，设置 x 轴和 y 轴的标签，以及图例位置
    if subplot == 2:
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.legend(loc="upper left")
# 创建一个新的图形窗口，设置其尺寸为 8x6 英寸
plt.figure(figsize=(8, 6))

# 使用 make_multilabel_classification 函数生成多标签分类数据集
# 参数设置如下：
#   - n_classes=2: 生成的数据集中包含的类别数量为 2
#   - n_labels=1: 每个样本有一个标签
#   - allow_unlabeled=True: 允许生成未标记的样本
#   - random_state=1: 设置随机数种子为 1，确保结果的可重复性
X, Y = make_multilabel_classification(
    n_classes=2, n_labels=1, allow_unlabeled=True, random_state=1
)

# 调用 plot_subfigure 函数，绘制子图
# 参数解释：
#   - X, Y: 分别为输入特征和标签
#   - 1: 子图的位置编号
#   - "With unlabeled samples + CCA": 子图标题
#   - "cca": 子图使用的算法类型
plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")

# 同上，绘制另一个子图，使用不同的算法类型 "pca"
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

# 重新生成数据集，但这次禁用未标记的样本生成
X, Y = make_multilabel_classification(
    n_classes=2, n_labels=1, allow_unlabeled=False, random_state=1
)

# 绘制第三个子图，标题指示没有未标记样本，并使用 CCA 算法
plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")

# 绘制第四个子图，标题指示没有未标记样本，并使用 PCA 算法
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

# 调整子图的布局参数，以确保它们之间的正确分隔和显示
plt.subplots_adjust(0.04, 0.02, 0.97, 0.94, 0.09, 0.2)

# 显示所有绘制的子图
plt.show()
```