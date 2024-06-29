# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\axhspan_demo.py`

```
"""
============
axhspan Demo
============

Create lines or rectangles that span the Axes in either the horizontal or
vertical direction, and lines than span the Axes with an arbitrary orientation.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 数学计算模块

t = np.arange(-1, 2, .01)  # 创建一个从 -1 到 2，步长为 0.01 的数组 t
s = np.sin(2 * np.pi * t)  # 计算数组 t 中每个元素的正弦值，并赋值给数组 s

fig, ax = plt.subplots()  # 创建一个新的图形窗口和一个 subplot，将 subplot 对象保存在 ax 中

ax.plot(t, s)  # 在 subplot 上绘制 t 对应的正弦函数值 s

# 在 y=0 处绘制粗红色水平线，横跨 x 范围
ax.axhline(linewidth=8, color='#d62728')

# 在 y=1 处绘制水平线，横跨 x 范围
ax.axhline(y=1)

# 在 x=1 处绘制垂直线，纵跨 y 范围
ax.axvline(x=1)

# 在 x=0 处绘制粗蓝色垂直线，从 y 范围的下四分之三开始到上四分之一结束
ax.axvline(x=0, ymin=0.75, linewidth=8, color='#1f77b4')

# 在 y=0.5 处绘制默认水平线，横跨 Axes 中间一半的宽度
ax.axhline(y=.5, xmin=0.25, xmax=0.75)

# 绘制穿过 (0, 0) 到 (1, 1) 的无限黑色直线
ax.axline((0, 0), (1, 1), color='k')

# 绘制从 y=0.25 到 y=0.75 的灰色矩形，横跨 Axes 的宽度
ax.axhspan(0.25, 0.75, facecolor='0.5')

# 绘制从 x=1.25 到 x=1.55 的绿色矩形，纵跨 Axes 的高度
ax.axvspan(1.25, 1.55, facecolor='#2ca02c')

plt.show()  # 显示图形
```