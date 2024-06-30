# `D:\src\scipysrc\scikit-learn\examples\svm\plot_svm_margin.py`

```
"""
=========================================================
SVM Margins Example
=========================================================
The plots below illustrate the effect the parameter `C` has
on the separation line. A large value of `C` basically tells
our model that we do not have that much faith in our data's
distribution, and will only consider points close to the line
of separation.

A small value of `C` includes more/all the observations, allowing
the margins to be calculated using all the data in the area.
"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库
import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy

from sklearn import svm  # 导入支持向量机模块

# 设置随机种子，保证结果可重复
np.random.seed(0)

# 生成40个线性可分的数据点
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# 设置图表编号
fignum = 1

# 拟合模型
for name, penalty in (("unreg", 1), ("reg", 0.05)):
    clf = svm.SVC(kernel="linear", C=penalty)  # 使用线性核的支持向量分类器，设置惩罚参数 C
    clf.fit(X, Y)  # 使用数据拟合模型

    # 获取分离超平面
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)  # 在指定区间生成均匀分布的数据点
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # 绘制通过支持向量的分离超平面平行线
    margin = 1 / np.sqrt(np.sum(clf.coef_**2))  # 计算间隔
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin

    # 绘制超平面、数据点及其支持向量
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, "k-")  # 绘制分离超平面
    plt.plot(xx, yy_down, "k--")  # 绘制支持向量平行线
    plt.plot(xx, yy_up, "k--")  # 绘制支持向量平行线

    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
    )  # 绘制支持向量
    plt.scatter(
        X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.get_cmap("RdBu"), edgecolors="k"
    )  # 绘制数据点

    plt.axis("tight")  # 自动调整坐标轴范围
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # 绘制决策函数的等高线图
    plt.contourf(XX, YY, Z, cmap=plt.get_cmap("RdBu"), alpha=0.5, linestyles=["-"])

    plt.xlim(x_min, x_max)  # 设置 x 轴显示范围
    plt.ylim(y_min, y_max)  # 设置 y 轴显示范围

    plt.xticks(())  # 隐藏 x 轴刻度
    plt.yticks(())  # 隐藏 y 轴刻度
    fignum = fignum + 1

plt.show()  # 显示所有绘图结果
```