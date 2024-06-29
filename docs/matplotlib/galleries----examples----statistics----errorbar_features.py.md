# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\errorbar_features.py`

```py
"""
=======================================
Different ways of specifying error bars
=======================================

Errors can be specified as a constant value (as shown in
:doc:`/gallery/statistics/errorbar`). However, this example demonstrates
how they vary by specifying arrays of error values.

If the raw ``x`` and ``y`` data have length N, there are two options:

Array of shape (N,):
    Error varies for each point, but the error values are
    symmetric (i.e. the lower and upper values are equal).

Array of shape (2, N):
    Error varies for each point, and the lower and upper limits
    (in that order) are different (asymmetric case)

In addition, this example demonstrates how to use log
scale with error bars.
"""

import matplotlib.pyplot as plt
import numpy as np

# example data
x = np.arange(0.1, 4, 0.5)  # 创建一个包含0.1到4（不含）的等差数组作为x轴数据
y = np.exp(-x)  # 计算x轴数据的指数函数作为y轴数据

# example error bar values that vary with x-position
error = 0.1 + 0.2 * x  # 创建一个数组，其元素随着x位置变化而变化

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)  # 创建包含两个子图的图形，它们共享相同的x轴

# 在第一个子图上绘制y误差条，对应的x误差为默认值
ax0.errorbar(x, y, yerr=error, fmt='-o')
ax0.set_title('variable, symmetric error')  # 设置第一个子图的标题

# error bar values w/ different -/+ errors that
# also vary with the x-position
lower_error = 0.4 * error  # 创建一个数组，其元素为error数组的元素乘以0.4
upper_error = error  # 将error数组赋值给upper_error数组，表示上限误差等于error数组
asymmetric_error = [lower_error, upper_error]  # 创建一个列表，包含lower_error和upper_error数组

# 在第二个子图上绘制x误差条，对应的y误差为asymmetric_error数组
ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
ax1.set_title('variable, asymmetric error')  # 设置第二个子图的标题
ax1.set_yscale('log')  # 将第二个子图的y轴设置为对数尺度
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`
```