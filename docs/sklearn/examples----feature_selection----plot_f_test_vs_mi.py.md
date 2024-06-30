# `D:\src\scipysrc\scikit-learn\examples\feature_selection\plot_f_test_vs_mi.py`

```
"""
===========================================
Comparison of F-test and mutual information
===========================================

This example illustrates the differences between univariate F-test statistics
and mutual information.

We consider 3 features x_1, x_2, x_3 distributed uniformly over [0, 1], the
target depends on them as follows:

y = x_1 + sin(6 * pi * x_2) + 0.1 * N(0, 1), that is the third feature is
completely irrelevant.

The code below plots the dependency of y against individual x_i and normalized
values of univariate F-tests statistics and mutual information.

As F-test captures only linear dependency, it rates x_1 as the most
discriminative feature. On the other hand, mutual information can capture any
kind of dependency between variables and it rates x_2 as the most
discriminative feature, which probably agrees better with our intuitive
perception for this example. Both methods correctly mark x_3 as irrelevant.
"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from sklearn.feature_selection import f_regression, mutual_info_regression  # 从sklearn.feature_selection模块导入f_regression和mutual_info_regression函数

np.random.seed(0)  # 设定随机种子，以便结果可重复
X = np.random.rand(1000, 3)  # 创建一个1000x3的随机数矩阵X，元素取值在[0, 1]之间
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)  # 根据特定公式生成目标变量y

f_test, _ = f_regression(X, y)  # 计算特征X对目标变量y的F-test统计量
f_test /= np.max(f_test)  # 将F-test统计量标准化到[0, 1]之间

mi = mutual_info_regression(X, y)  # 计算特征X对目标变量y的互信息
mi /= np.max(mi)  # 将互信息标准化到[0, 1]之间

plt.figure(figsize=(15, 5))  # 创建一个图形窗口，大小为15x5
for i in range(3):
    plt.subplot(1, 3, i + 1)  # 创建1x3的子图区域，当前是第i+1个子图
    plt.scatter(X[:, i], y, edgecolor="black", s=20)  # 绘制散点图，X轴为第i个特征，Y轴为目标变量y
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)  # 设置X轴标签，显示特征的编号
    if i == 0:
        plt.ylabel("$y$", fontsize=14)  # 设置Y轴标签，显示目标变量y
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16)  # 设置子图标题，显示F-test和互信息的值

plt.show()  # 显示绘制的图形
```