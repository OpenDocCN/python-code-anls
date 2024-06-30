# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_logistic_multinomial.py`

```
"""
====================================================
Plot multinomial and One-vs-Rest Logistic Regression
====================================================

Plot decision surface of multinomial and One-vs-Rest Logistic Regression.
The hyperplanes corresponding to the three One-vs-Rest (OVR) classifiers
are represented by the dashed lines.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

from sklearn.datasets import make_blobs  # 导入 make_blobs 函数，用于生成模拟数据集
from sklearn.inspection import DecisionBoundaryDisplay  # 导入 DecisionBoundaryDisplay 类，用于显示决策边界
from sklearn.linear_model import LogisticRegression  # 导入 LogisticRegression 类，用于逻辑回归模型
from sklearn.multiclass import OneVsRestClassifier  # 导入 OneVsRestClassifier 类，用于一对多分类策略

# make 3-class dataset for classification
centers = [[-5, 0], [0, 1.5], [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)  # 生成包含三个类别的模拟数据集
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X = np.dot(X, transformation)  # 对数据集进行线性变换

for multi_class in ("multinomial", "ovr"):
    clf = LogisticRegression(solver="sag", max_iter=100, random_state=42)  # 创建逻辑回归模型对象
    if multi_class == "ovr":
        clf = OneVsRestClassifier(clf)  # 如果是一对多分类策略，则使用 OneVsRestClassifier 封装模型
    clf.fit(X, y)  # 拟合模型

    # print the training scores
    print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))  # 打印训练分数

    _, ax = plt.subplots()  # 创建图形和子图对象
    DecisionBoundaryDisplay.from_estimator(
        clf, X, response_method="predict", cmap=plt.cm.Paired, ax=ax
    )  # 显示决策边界
    plt.title("Decision surface of LogisticRegression (%s)" % multi_class)  # 设置图表标题
    plt.axis("tight")  # 设置坐标轴范围

    # Plot also the training points
    colors = "bry"  # 设置类别颜色
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, edgecolor="black", s=20)  # 绘制训练数据点

    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()  # 获取 x 轴范围
    ymin, ymax = plt.ylim()  # 获取 y 轴范围
    if multi_class == "ovr":
        coef = np.concatenate([est.coef_ for est in clf.estimators_])  # 获取每个分类器的系数
        intercept = np.concatenate([est.intercept_ for est in clf.estimators_])  # 获取每个分类器的截距
    else:
        coef = clf.coef_
        intercept = clf.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]  # 计算超平面的直线方程

        plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)  # 绘制超平面

    for i, color in zip(clf.classes_, colors):
        plot_hyperplane(i, color)  # 绘制每个类别的超平面

plt.show()  # 显示图形
```