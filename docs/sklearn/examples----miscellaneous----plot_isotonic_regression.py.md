# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_isotonic_regression.py`

```
"""
===================
Isotonic Regression
===================

An illustration of the isotonic regression on generated data (non-linear
monotonic trend with homoscedastic uniform noise).

The isotonic regression algorithm finds a non-decreasing approximation of a
function while minimizing the mean squared error on the training data. The
benefit of such a non-parametric model is that it does not assume any shape for
the target function besides monotonicity. For comparison a linear regression is
also presented.

The plot on the right-hand side shows the model prediction function that
results from the linear interpolation of thresholds points. The thresholds
points are a subset of the training input observations and their matching
target values are computed by the isotonic non-parametric fit.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算
from matplotlib.collections import LineCollection  # 导入 LineCollection 类，用于绘制线段集合

from sklearn.isotonic import IsotonicRegression  # 导入 IsotonicRegression 类，用于等渗回归
from sklearn.linear_model import LinearRegression  # 导入 LinearRegression 类，用于线性回归
from sklearn.utils import check_random_state  # 导入 check_random_state 函数，用于随机数控制

n = 100  # 设置数据点数量为 100
x = np.arange(n)  # 生成从 0 到 n-1 的整数数组作为 x 值
rs = check_random_state(0)  # 初始化随机数生成器
y = rs.randint(-50, 50, size=(n,)) + 50.0 * np.log1p(np.arange(n))  # 生成具有噪声的数据 y

# %%
# Fit IsotonicRegression and LinearRegression models:

ir = IsotonicRegression(out_of_bounds="clip")  # 创建 IsotonicRegression 对象，设置越界处理方式为“clip”
y_ = ir.fit_transform(x, y)  # 对数据进行等渗回归拟合

lr = LinearRegression()  # 创建 LinearRegression 对象
lr.fit(x[:, np.newaxis], y)  # 对数据进行线性回归拟合，需要将 x 转换为二维数组形式

# %%
# Plot results:

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]  # 创建线段列表，用于绘制 IsotonicRegression 拟合效果
lc = LineCollection(segments, zorder=0)  # 创建线段集合对象
lc.set_array(np.ones(len(y)))  # 设置线段集合的数组

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))  # 创建包含两个子图的图形对象

ax0.plot(x, y, "C0.", markersize=12)  # 绘制原始数据散点图
ax0.plot(x, y_, "C1.-", markersize=12)  # 绘制 IsotonicRegression 拟合的曲线
ax0.plot(x, lr.predict(x[:, np.newaxis]), "C2-")  # 绘制线性回归拟合的曲线
ax0.add_collection(lc)  # 将线段集合添加到图中
ax0.legend(("Training data", "Isotonic fit", "Linear fit"), loc="lower right")  # 设置图例
ax0.set_title("Isotonic regression fit on noisy data (n=%d)" % n)  # 设置子图标题

x_test = np.linspace(-10, 110, 1000)  # 生成测试用的 x 数据
ax1.plot(x_test, ir.predict(x_test), "C1-")  # 绘制预测函数曲线
ax1.plot(ir.X_thresholds_, ir.y_thresholds_, "C1.", markersize=12)  # 绘制用于拟合的阈值点
ax1.set_title("Prediction function (%d thresholds)" % len(ir.X_thresholds_))  # 设置子图标题

plt.show()  # 显示图形

# %%
# Note that we explicitly passed `out_of_bounds="clip"` to the constructor of
# `IsotonicRegression` to control the way the model extrapolates outside of the
# range of data observed in the training set. This "clipping" extrapolation can
# be seen on the plot of the decision function on the right-hand.
```