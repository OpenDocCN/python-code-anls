# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\markevery_demo.py`

```
"""
==============
Markevery Demo
==============

The ``markevery`` property of `.Line2D` allows drawing markers at a subset of
data points.

The list of possible parameters is specified at `.Line2D.set_markevery`.
In short:

- A single integer N draws every N-th marker.
- A tuple of integers (start, N) draws every N-th marker, starting at data
  index *start*.
- A list of integers draws the markers at the specified indices.
- A slice draws the markers at the sliced indices.
- A float specifies the distance between markers as a fraction of the Axes
  diagonal in screen space. This will lead to a visually uniform distribution
  of the points along the line, irrespective of scales and zooming.
"""

import matplotlib.pyplot as plt
import numpy as np

# define a list of markevery cases to plot
# 定义要绘制的各种 markevery 情况的列表
cases = [
    None,             # 情况1：无标记点
    8,                # 情况2：每隔8个点绘制一个标记点
    (30, 8),          # 情况3：从第30个点开始，每隔8个点绘制一个标记点
    [16, 24, 32],     # 情况4：在第16、24和32个点处绘制标记点
    [0, -1],          # 情况5：在第0个和最后一个点处绘制标记点
    slice(100, 200, 3),  # 情况6：在从第100到199的索引中每隔3个点绘制一个标记点
    0.1,              # 情况7：在屏幕空间中以Axes对角线的10%间隔绘制标记点
    0.4,              # 情况8：在屏幕空间中以Axes对角线的40%间隔绘制标记点
    (0.2, 0.4)        # 情况9：在屏幕空间中以Axes对角线的20%开始，40%间隔绘制标记点
]

# data points
# 数据点
delta = 0.11
x = np.linspace(0, 10 - 2 * delta, 200) + delta
y = np.sin(x) + 1.0 + delta

# %%
# markevery with linear scales
# ----------------------------
# 线性比例尺下的 markevery 演示

fig, axs = plt.subplots(3, 3, figsize=(10, 6), layout='constrained')
for ax, markevery in zip(axs.flat, cases):
    ax.set_title(f'markevery={markevery}')
    ax.plot(x, y, 'o', ls='-', ms=4, markevery=markevery)

# %%
# markevery with log scales
# -------------------------
# 对数比例尺下的 markevery 演示
#
# 注意，在整数基础上的 markevery 在对数比例尺下会导致标记点距离的视觉不对称性。
# 相比之下，基于图形大小的浮点数设置会根据 Axes 对角线的分数均匀分布点，而不受比例尺和缩放的影响。

fig, axs = plt.subplots(3, 3, figsize=(10, 6), layout='constrained')
for ax, markevery in zip(axs.flat, cases):
    ax.set_title(f'markevery={markevery}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(x, y, 'o', ls='-', ms=4, markevery=markevery)

# %%
# markevery on zoomed plots
# -------------------------
# 缩放后的 markevery 演示
#
# 基于整数的 markevery 规范会选择基础数据的点，独立于视图。相比之下，基于浮点数的规范与 Axes 对角线相关。
# 缩放不会改变 Axes 对角线，但会改变显示的数据范围，缩放时将显示更多的点。

fig, axs = plt.subplots(3, 3, figsize=(10, 6), layout='constrained')
for ax, markevery in zip(axs.flat, cases):
    ax.set_title(f'markevery={markevery}')
    ax.plot(x, y, 'o', ls='-', ms=4, markevery=markevery)
    ax.set_xlim((6, 6.7))
    ax.set_ylim((1.1, 1.7))

# %%
# markevery on polar plots
# ------------------------
# 极坐标下的 markevery 演示

r = np.linspace(0, 3.0, 200)
theta = 2 * np.pi * r

fig, axs = plt.subplots(3, 3, figsize=(10, 6), layout='constrained',
                        subplot_kw={'projection': 'polar'})
for ax, markevery in zip(axs.flat, cases):
    ax.set_title(f'markevery={markevery}')
    # 在极坐标图上绘制散点图
    ax.plot(theta, r, 'o', ls='-', ms=4, markevery=markevery)
    # 使用给定的数据 theta 和 r，在极坐标上绘制散点图
    # 'o' 表示使用圆圈作为散点的标记
    # ls='-' 表示使用实线作为连接线的样式
    # ms=4 表示设置散点的大小为 4
    # markevery=markevery 表示设置每隔一定间隔显示一个标记点
# 显示 matplotlib 中当前的图形
plt.show()
```