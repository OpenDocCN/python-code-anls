# `D:\src\scipysrc\scikit-learn\examples\svm\plot_separating_hyperplane_unbalanced.py`

```
"""
=================================================
SVM: Separating hyperplane for unbalanced classes
=================================================

Find the optimal separating hyperplane using an SVC for classes that
are unbalanced.

We first find the separating plane with a plain SVC and then plot
(dashed) the separating hyperplane with automatic correction for
unbalanced classes.

.. currentmodule:: sklearn.linear_model

.. note::

    This example will also work by replacing ``SVC(kernel="linear")``
    with ``SGDClassifier(loss="hinge")``. Setting the ``loss`` parameter
    of the :class:`SGDClassifier` equal to ``hinge`` will yield behaviour
    such as that of a SVC with a linear kernel.

    For example try instead of the ``SVC``::

        clf = SGDClassifier(n_iter=100, alpha=0.01)

"""

import matplotlib.lines as mlines  # 导入用于绘制线条的模块
import matplotlib.pyplot as plt  # 导入绘图库

from sklearn import svm  # 导入支持向量机模型
from sklearn.datasets import make_blobs  # 导入生成聚类数据的函数
from sklearn.inspection import DecisionBoundaryDisplay  # 导入决策边界显示函数

# 创建两个随机点的聚类数据集
n_samples_1 = 1000  # 第一个聚类的样本数
n_samples_2 = 100  # 第二个聚类的样本数
centers = [[0.0, 0.0], [2.0, 2.0]]  # 聚类的中心点坐标
clusters_std = [1.5, 0.5]  # 聚类的标准差
X, y = make_blobs(
    n_samples=[n_samples_1, n_samples_2],  # 每个聚类的样本数
    centers=centers,  # 聚类的中心点
    cluster_std=clusters_std,  # 聚类的标准差
    random_state=0,  # 随机种子，确保结果的可重复性
    shuffle=False,  # 不打乱样本的顺序
)

# 拟合模型并获取分隔超平面
clf = svm.SVC(kernel="linear", C=1.0)  # 创建线性核支持向量分类器
clf.fit(X, y)  # 拟合模型

# 拟合带有加权类别的模型并获取分隔超平面
wclf = svm.SVC(kernel="linear", class_weight={1: 10})  # 创建带有加权类别的线性核支持向量分类器
wclf.fit(X, y)  # 拟合模型

# 绘制样本点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

# 绘制两个分类器的决策函数
ax = plt.gca()  # 获取当前的坐标轴
disp = DecisionBoundaryDisplay.from_estimator(
    clf,  # 使用普通SVC模型
    X,  # 特征数据
    plot_method="contour",  # 使用等高线绘制
    colors="k",  # 等高线的颜色为黑色
    levels=[0],  # 绘制决策边界的值为0
    alpha=0.5,  # 绘制透明度为0.5
    linestyles=["-"],  # 使用实线绘制
    ax=ax,  # 绘制在指定的坐标轴上
)

# 绘制加权类别的决策边界和边界
wdisp = DecisionBoundaryDisplay.from_estimator(
    wclf,  # 使用加权类别的SVC模型
    X,  # 特征数据
    plot_method="contour",  # 使用等高线绘制
    colors="r",  # 等高线的颜色为红色
    levels=[0],  # 绘制决策边界的值为0
    alpha=0.5,  # 绘制透明度为0.5
    linestyles=["-"],  # 使用实线绘制
    ax=ax,  # 绘制在指定的坐标轴上
)

# 添加图例，显示非加权和加权分类器的区别
plt.legend(
    [
        mlines.Line2D([], [], color="k", label="non weighted"),  # 非加权分类器的图例
        mlines.Line2D([], [], color="r", label="weighted"),  # 加权分类器的图例
    ],
    ["non weighted", "weighted"],  # 图例的标签
    loc="upper right",  # 图例的位置
)
plt.show()  # 显示图形
```