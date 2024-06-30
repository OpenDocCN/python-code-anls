# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_ols_ridge_variance.py`

```
"""
=========================================================
Ordinary Least Squares and Ridge Regression Variance
=========================================================
Due to the few points in each dimension and the straight
line that linear regression uses to follow these points
as well as it can, noise on the observations will cause
great variance as shown in the first plot. Every line's slope
can vary quite a bit for each prediction due to the noise
induced in the observations.

Ridge regression is basically minimizing a penalised version
of the least-squared function. The penalising `shrinks` the
value of the regression coefficients.
Despite the few data points in each dimension, the slope
of the prediction is much more stable and the variance
in the line itself is greatly reduced, in comparison to that
of the standard linear regression

"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# SPDX-License-Identifier: BSD-3-Clause

# 导入绘图和数据处理库
import matplotlib.pyplot as plt
import numpy as np

# 导入线性模型库
from sklearn import linear_model

# 训练数据
X_train = np.c_[0.5, 1].T
y_train = [0.5, 1]
X_test = np.c_[0, 2].T

# 设置随机种子
np.random.seed(0)

# 定义线性模型字典，包括OLS和岭回归
classifiers = dict(
    ols=linear_model.LinearRegression(), ridge=linear_model.Ridge(alpha=0.1)
)

# 遍历每个线性模型
for name, clf in classifiers.items():
    # 创建绘图窗口和子图
    fig, ax = plt.subplots(figsize=(4, 3))

    # 生成带有噪声的训练数据
    for _ in range(6):
        this_X = 0.1 * np.random.normal(size=(2, 1)) + X_train
        clf.fit(this_X, y_train)

        # 绘制预测线
        ax.plot(X_test, clf.predict(X_test), color="gray")
        # 绘制训练数据点
        ax.scatter(this_X, y_train, s=3, c="gray", marker="o", zorder=10)

    # 使用原始训练数据重新拟合模型
    clf.fit(X_train, y_train)
    # 绘制最终的预测线
    ax.plot(X_test, clf.predict(X_test), linewidth=2, color="blue")
    # 绘制原始训练数据点
    ax.scatter(X_train, y_train, s=30, c="red", marker="+", zorder=10)

    # 设置图标题和坐标轴范围
    ax.set_title(name)
    ax.set_xlim(0, 2)
    ax.set_ylim((0, 1.6))
    ax.set_xlabel("X")
    ax.set_ylabel("y")

    # 调整布局以适应子图
    fig.tight_layout()

# 显示绘图
plt.show()
```