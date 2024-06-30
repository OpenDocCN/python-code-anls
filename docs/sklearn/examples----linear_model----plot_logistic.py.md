# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_logistic.py`

```
"""
=========================================================
Logistic function
=========================================================

Shown in the plot is how the logistic regression would, in this
synthetic dataset, classify values as either 0 or 1,
i.e. class one or two, using the logistic curve.

"""

# Code source: Gael Varoquaux
# SPDX-License-Identifier: BSD-3-Clause

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算
from scipy.special import expit  # 从scipy.special中导入expit函数，用于逻辑函数的计算

from sklearn.linear_model import LinearRegression, LogisticRegression  # 导入线性回归和逻辑回归模型

# 生成一个简单的数据集，这是一个带有高斯噪声的直线：
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(float)
X[X > 0] *= 4
X += 0.3 * np.random.normal(size=n_samples)

X = X[:, np.newaxis]

# 拟合分类器
clf = LogisticRegression(C=1e5)  # 创建逻辑回归模型对象
clf.fit(X, y)  # 在数据集上拟合逻辑回归模型

# 绘制结果
plt.figure(1, figsize=(4, 3))  # 创建一个图形窗口，指定大小为4x3英寸
plt.clf()  # 清除当前图形窗口中的内容
plt.scatter(X.ravel(), y, label="example data", color="black", zorder=20)  # 绘制散点图，展示样本数据
X_test = np.linspace(-5, 10, 300)  # 在指定范围内生成300个均匀分布的测试数据点

# 计算逻辑回归模型的预测结果
loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)  # 绘制逻辑回归模型的曲线

ols = LinearRegression()
ols.fit(X, y)
plt.plot(
    X_test,
    ols.coef_ * X_test + ols.intercept_,
    label="Linear Regression Model",
    linewidth=1,
)  # 绘制线性回归模型的直线

plt.axhline(0.5, color=".5")  # 添加水平线，y轴坐标为0.5，颜色为灰色

plt.ylabel("y")  # 设置y轴标签
plt.xlabel("X")  # 设置x轴标签
plt.xticks(range(-5, 10))  # 设置x轴刻度范围
plt.yticks([0, 0.5, 1])  # 设置y轴刻度范围
plt.ylim(-0.25, 1.25)  # 设置y轴显示范围
plt.xlim(-4, 10)  # 设置x轴显示范围
plt.legend(
    loc="lower right",
    fontsize="small",
)  # 添加图例，位置为右下角，字体大小为小号

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()  # 显示图形
```