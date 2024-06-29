# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\errorbar.py`

```py
"""
=================
Errorbar function
=================

This exhibits the most basic use of the error bar method.
In this case, constant values are provided for the error
in both the x- and y-directions.
"""

# 导入 matplotlib 的 pyplot 模块，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np

# example data
# 创建一个从 0.1 到 4（不包括4），步长为0.5的数组作为 x 值
x = np.arange(0.1, 4, 0.5)
# 计算 y 值，e 的负 x 次幂
y = np.exp(-x)

# 创建图形和轴对象，fig是整个图形，ax是坐标轴
fig, ax = plt.subplots()
# 在坐标轴上画出带有误差线的数据图，xerr表示 x 方向的误差，yerr表示 y 方向的误差
ax.errorbar(x, y, xerr=0.2, yerr=0.4)
# 显示图形
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