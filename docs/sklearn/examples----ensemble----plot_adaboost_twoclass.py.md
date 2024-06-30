# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_adaboost_twoclass.py`

```
"""
==================
Two-class AdaBoost
==================

This example fits an AdaBoosted decision stump on a non-linearly separable
classification dataset composed of two "Gaussian quantiles" clusters
(see :func:`sklearn.datasets.make_gaussian_quantiles`) and plots the decision
boundary and decision scores. The distributions of decision scores are shown
separately for samples of class A and B. The predicted class label for each
sample is determined by the sign of the decision score. Samples with decision
scores greater than zero are classified as B, and are otherwise classified
as A. The magnitude of a decision score determines the degree of likeness with
the predicted class label. Additionally, a new dataset could be constructed
containing a desired purity of class B, for example, by only selecting samples
with a decision score above some value.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并用 plt 别名表示
import numpy as np  # 导入 numpy 库，并用 np 别名表示

from sklearn.datasets import make_gaussian_quantiles  # 导入生成高斯量化数据集的函数
from sklearn.ensemble import AdaBoostClassifier  # 导入 AdaBoost 分类器
from sklearn.inspection import DecisionBoundaryDisplay  # 导入用于绘制决策边界的类
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器

# Construct dataset
X1, y1 = make_gaussian_quantiles(  # 生成第一个高斯量化数据集
    cov=2.0, n_samples=200, n_features=2, n_classes=2, random_state=1
)
X2, y2 = make_gaussian_quantiles(  # 生成第二个高斯量化数据集
    mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1
)
X = np.concatenate((X1, X2))  # 将两个数据集合并
y = np.concatenate((y1, -y2 + 1))  # 将标签合并，确保类别是二元的

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(  # 创建 AdaBoost 分类器，基础分类器为深度为1的决策树
    DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200
)

bdt.fit(X, y)  # 使用数据 X, y 进行模型训练

plot_colors = "br"  # 绘图颜色序列
plot_step = 0.02  # 绘图步长
class_names = "AB"  # 类别名称

plt.figure(figsize=(10, 5))  # 创建绘图窗口大小为 10x5

# Plot the decision boundaries
ax = plt.subplot(121)  # 创建子图，位置为 1x2 中的第一个
disp = DecisionBoundaryDisplay.from_estimator(  # 从分类器创建决策边界显示
    bdt,
    X,
    cmap=plt.cm.Paired,
    response_method="predict",
    ax=ax,
    xlabel="x",
    ylabel="y",
)
x_min, x_max = disp.xx0.min(), disp.xx0.max()  # 获取 x 范围
y_min, y_max = disp.xx1.min(), disp.xx1.max()  # 获取 y 范围
plt.axis("tight")  # 调整坐标轴范围

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):  # 遍历类别，绘制训练点
    idx = np.where(y == i)  # 找到当前类别的索引
    plt.scatter(
        X[idx, 0],
        X[idx, 1],
        c=c,
        s=20,
        edgecolor="k",
        label="Class %s" % n,
    )  # 绘制散点图
plt.xlim(x_min, x_max)  # 设置 x 轴范围
plt.ylim(y_min, y_max)  # 设置 y 轴范围
plt.legend(loc="upper right")  # 添加图例，位置在右上角
plt.title("Decision Boundary")  # 设置子图标题

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)  # 获取两类决策分数
plot_range = (twoclass_output.min(), twoclass_output.max())  # 获取绘图范围
plt.subplot(122)  # 创建子图，位置为 1x2 中的第二个
for i, n, c in zip(range(2), class_names, plot_colors):  # 遍历类别，绘制直方图
    plt.hist(
        twoclass_output[y == i],
        bins=10,
        range=plot_range,
        facecolor=c,
        label="Class %s" % n,
        alpha=0.5,
        edgecolor="k",
    )
x1, x2, y1, y2 = plt.axis()  # 获取当前坐标轴范围
plt.axis((x1, x2, y1, y2 * 1.2))  # 设置坐标轴范围
plt.legend(loc="upper right")  # 添加图例，位置在右上角
plt.ylabel("Samples")  # 设置 y 轴标签
plt.xlabel("Score")  # 设置 x 轴标签
# 设置图表的标题为 "Decision Scores"
plt.title("Decision Scores")

# 调整子图之间的布局，使它们更紧凑
plt.tight_layout()

# 调整子图之间的水平空白间距为 0.35
plt.subplots_adjust(wspace=0.35)

# 显示绘制的图表
plt.show()
```