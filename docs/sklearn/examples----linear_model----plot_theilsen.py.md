# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_theilsen.py`

```
"""
====================
Theil-Sen Regression
====================

Computes a Theil-Sen Regression on a synthetic dataset.

See :ref:`theil_sen_regression` for more information on the regressor.

Compared to the OLS (ordinary least squares) estimator, the Theil-Sen
estimator is robust against outliers. It has a breakdown point of about 29.3%
in case of a simple linear regression which means that it can tolerate
arbitrary corrupted data (outliers) of up to 29.3% in the two-dimensional
case.

The estimation of the model is done by calculating the slopes and intercepts
of a subpopulation of all possible combinations of p subsample points. If an
intercept is fitted, p must be greater than or equal to n_features + 1. The
final slope and intercept is then defined as the spatial median of these
slopes and intercepts.

In certain cases Theil-Sen performs better than :ref:`RANSAC
<ransac_regression>` which is also a robust method. This is illustrated in the
second example below where outliers with respect to the x-axis perturb RANSAC.
Tuning the ``residual_threshold`` parameter of RANSAC remedies this but in
general a priori knowledge about the data and the nature of the outliers is
needed.
Due to the computational complexity of Theil-Sen it is recommended to use it
only for small problems in terms of number of samples and features. For larger
problems the ``max_subpopulation`` parameter restricts the magnitude of all
possible combinations of p subsample points to a randomly chosen subset and
therefore also limits the runtime. Therefore, Theil-Sen is applicable to larger
problems with the drawback of losing some of its mathematical properties since
it then works on a random subset.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time  # 导入时间模块，用于计算执行时间

import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor  # 导入线性回归和稳健回归算法

estimators = [
    ("OLS", LinearRegression()),  # 使用普通最小二乘法线性回归
    ("Theil-Sen", TheilSenRegressor(random_state=42)),  # 使用Theil-Sen回归
    ("RANSAC", RANSACRegressor(random_state=42)),  # 使用RANSAC稳健回归
]
colors = {"OLS": "turquoise", "Theil-Sen": "gold", "RANSAC": "lightgreen"}  # 设置不同算法的绘图颜色
lw = 2  # 绘图线条宽度

# %%
# Outliers only in the y direction
# --------------------------------

np.random.seed(0)
n_samples = 200
# Linear model y = 3*x + N(2, 0.1**2)
x = np.random.randn(n_samples)  # 生成随机正态分布的x样本
w = 3.0
c = 2.0
noise = 0.1 * np.random.randn(n_samples)  # 生成随机正态分布的噪声
y = w * x + c + noise  # 根据线性模型生成y样本数据
# 10% outliers
y[-20:] += -20 * x[-20:]  # 引入10%的y方向上的异常值
X = x[:, np.newaxis]  # 将x转换为二维数组形式

plt.scatter(x, y, color="indigo", marker="x", s=40)  # 绘制散点图，显示数据分布
line_x = np.array([-3, 3])  # 定义绘制线性的x坐标范围
for name, estimator in estimators:
    t0 = time.time()  # 记录开始时间
    estimator.fit(X, y)  # 使用当前算法拟合数据
    elapsed_time = time.time() - t0  # 计算拟合所花费的时间
    y_pred = estimator.predict(line_x.reshape(2, 1))  # 预测给定x值的y值
    plt.plot(
        line_x,
        y_pred,
        color=colors[name],
        linewidth=lw,
        label="%s (fit time: %.2fs)" % (name, elapsed_time),
    )  # 绘制拟合直线，并添加标签显示拟合时间

plt.axis("tight")  # 调整坐标轴范围以适应数据点
plt.legend(loc="upper left")  # 添加图例并设置位置
_ = plt.title("Corrupt y")  # 设置图表标题

# %%
# Outliers in the X direction
# ---------------------------

np.random.seed(0)
# 设定随机数种子，确保结果可复现性

# Generate random data points following a normal distribution for x
# 生成符合正态分布的随机数据点作为 x
x = np.random.randn(n_samples)

# Generate noise following a normal distribution with mean 0 and variance (0.1**2)
# 生成符合均值为0，方差为(0.1**2)的正态分布噪声
noise = 0.1 * np.random.randn(n_samples)

# Create y values based on the linear model y = 3*x + 2 + noise
# 根据线性模型 y = 3*x + 2 + noise 创建 y 值
y = 3 * x + 2 + noise

# Introduce outliers by modifying the last 20 elements of x and y
# 引入异常值，修改 x 和 y 的最后20个元素
x[-20:] = 9.9
y[-20:] += 22

# Reshape x to be a column vector
# 将 x 重塑为列向量
X = x[:, np.newaxis]

# Create a new figure for plotting
# 创建一个新的图形用于绘图
plt.figure()

# Scatter plot of x versus y with specific styling
# 绘制散点图，展示 x 和 y 的关系，设置颜色为靛青色，标记为 'x'，大小为 40
plt.scatter(x, y, color="indigo", marker="x", s=40)

# Define a line for plotting purposes
# 定义一个用于绘图的线段 x 范围
line_x = np.array([-3, 10])

# Iterate over the list of estimators and plot their fitted lines
# 遍历估计器列表，拟合数据并绘制拟合的直线
for name, estimator in estimators:
    # Record the starting time for fitting the estimator
    # 记录开始拟合的时间
    t0 = time.time()
    # Fit the estimator on the data X and y
    # 在数据 X 和 y 上拟合估计器
    estimator.fit(X, y)
    # Calculate the elapsed time for fitting
    # 计算拟合所用的时间
    elapsed_time = time.time() - t0
    # Predict y values for the line_x data points
    # 预测线段上各点的 y 值
    y_pred = estimator.predict(line_x.reshape(2, 1))
    # Plot the predicted line with specific color and label
    # 绘制预测的直线，设置颜色为 colors[name]，线宽为 lw，标签包含估计器名称和拟合时间
    plt.plot(
        line_x,
        y_pred,
        color=colors[name],
        linewidth=lw,
        label="%s (fit time: %.2fs)" % (name, elapsed_time),
    )

# Adjust plot limits to tightly fit the data
# 调整图的界限以紧密包含数据
plt.axis("tight")

# Add a legend to the upper left corner of the plot
# 在图的左上角添加图例
plt.legend(loc="upper left")

# Set the title of the plot
# 设置图的标题
plt.title("Corrupt x")

# Display the plot
# 显示绘制的图
plt.show()
```