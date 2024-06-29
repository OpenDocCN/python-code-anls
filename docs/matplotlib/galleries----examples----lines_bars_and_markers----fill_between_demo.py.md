# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\fill_between_demo.py`

```py
"""
==============================
Filling the area between lines
==============================

This example shows how to use `~.axes.Axes.fill_between` to color the area
between two lines.
"""

import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

# %%
#
# Basic usage
# -----------
# The parameters *y1* and *y2* can be scalars, indicating a horizontal
# boundary at the given y-values. If only *y1* is given, *y2* defaults to 0.

x = np.arange(0.0, 2, 0.01)  # 生成从0到2（不包括2），步长为0.01的数组作为x坐标
y1 = np.sin(2 * np.pi * x)  # 计算sin(2πx)作为y1
y2 = 0.8 * np.sin(4 * np.pi * x)  # 计算0.8*sin(4πx)作为y2

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))  # 创建包含3个子图的图像对象

ax1.fill_between(x, y1)  # 在x和y1之间填充区域
ax1.set_title('fill between y1 and 0')  # 设置子图1的标题

ax2.fill_between(x, y1, 1)  # 在x和y1与y=1之间填充区域
ax2.set_title('fill between y1 and 1')  # 设置子图2的标题

ax3.fill_between(x, y1, y2)  # 在x和y1与y2之间填充区域
ax3.set_title('fill between y1 and y2')  # 设置子图3的标题
ax3.set_xlabel('x')  # 设置x轴标签
fig.tight_layout()  # 调整子图布局，使之紧凑显示

# %%
#
# Example: Confidence bands
# -------------------------
# A common application for `~.axes.Axes.fill_between` is the indication of
# confidence bands.
#
# `~.axes.Axes.fill_between` uses the colors of the color cycle as the fill
# color. These may be a bit strong when applied to fill areas. It is
# therefore often a good practice to lighten the color by making the area
# semi-transparent using *alpha*.

# sphinx_gallery_thumbnail_number = 2

N = 21
x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

# fit a linear curve and estimate its y-values and their error.
a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

fig, ax = plt.subplots()  # 创建一个新的图像对象
ax.plot(x, y_est, '-')  # 绘制拟合直线
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)  # 使用透明度为0.2填充拟合曲线的误差范围
ax.plot(x, y, 'o', color='tab:brown')  # 绘制原始数据点

# %%
#
# Selectively filling horizontal regions
# --------------------------------------
# The parameter *where* allows to specify the x-ranges to fill. It's a boolean
# array with the same size as *x*.
#
# Only x-ranges of contiguous *True* sequences are filled. As a result the
# range between neighboring *True* and *False* values is never filled. This
# often undesired when the data points should represent a contiguous quantity.
# It is therefore recommended to set ``interpolate=True`` unless the
# x-distance of the data points is fine enough so that the above effect is not
# noticeable. Interpolation approximates the actual x position at which the
# *where* condition will change and extends the filling up to there.

x = np.array([0, 1, 2, 3])  # 创建一个包含4个元素的数组作为x坐标
y1 = np.array([0.8, 0.8, 0.2, 0.2])  # 创建一个包含4个元素的数组作为y1坐标
y2 = np.array([0, 0, 1, 1])  # 创建一个包含4个元素的数组作为y2坐标

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 创建包含2个子图的图像对象

ax1.set_title('interpolation=False')  # 设置子图1的标题
ax1.plot(x, y1, 'o--')  # 绘制y1的数据点
ax1.plot(x, y2, 'o--')  # 绘制y2的数据点
ax1.fill_between(x, y1, y2, where=(y1 > y2), color='C0', alpha=0.3)  # 在y1>y2的区域填充，使用C0色彩，透明度0.3
ax1.fill_between(x, y1, y2, where=(y1 < y2), color='C1', alpha=0.3)  # 在y1<y2的区域填充，使用C1色彩，透明度0.3

ax2.set_title('interpolation=True')  # 设置子图2的标题
ax2.plot(x, y1, 'o--')  # 绘制y1的数据点
ax2.plot(x, y2, 'o--')  # 绘制y2的数据点
# 填充图中两条曲线之间的区域，当 y1 > y2 时使用颜色 'C0' 和透明度 0.3，进行插值处理
ax2.fill_between(x, y1, y2, where=(y1 > y2), color='C0', alpha=0.3,
                 interpolate=True)

# 填充图中两条曲线之间的区域，当 y1 <= y2 时使用颜色 'C1' 和透明度 0.3，进行插值处理
ax2.fill_between(x, y1, y2, where=(y1 <= y2), color='C1', alpha=0.3,
                 interpolate=True)

# 调整图的布局，使其紧凑显示
fig.tight_layout()

# %%
#
# .. note::
#
#    如果 *y1* 或 *y2* 是掩码数组（masked arrays），将会出现类似的间隙。由于缺失值无法近似，
#    在这种情况下 *interpolate* 参数无效。掩码值周围的间隙只能通过在接近掩码值的位置添加更多数据点来减少。

# %%
#
# 选择性地标记整个 Axes 中的水平区域
# ------------------------------------------------------------
# 同样的选择机制可以应用于填充整个 Axes 的完整垂直高度。为了独立于 y 轴限制，我们添加了一个变换，
# 将 x 值解释为数据坐标，y 值解释为 Axes 坐标。
#
# 以下示例标记了 y 数据超过给定阈值的区域。

fig, ax = plt.subplots()
x = np.arange(0, 4 * np.pi, 0.01)
y = np.sin(x)
ax.plot(x, y, color='black')

threshold = 0.75
# 在图中添加一条水平线表示阈值
ax.axhline(threshold, color='green', lw=2, alpha=0.7)
# 填充 y > threshold 的区域，使用颜色 'green' 和透明度 0.5，使用 x 轴数据坐标变换
ax.fill_between(x, 0, 1, where=y > threshold,
                color='green', alpha=0.5, transform=ax.get_xaxis_transform())

# %%
#
# .. admonition:: References
#
#    本示例中展示了以下函数、方法、类和模块的使用:
#
#    - `matplotlib.axes.Axes.fill_between` / `matplotlib.pyplot.fill_between`
#    - `matplotlib.axes.Axes.get_xaxis_transform`
```