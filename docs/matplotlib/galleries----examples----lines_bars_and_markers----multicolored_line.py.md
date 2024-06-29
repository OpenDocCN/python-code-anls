# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\multicolored_line.py`

```py
"""
==================
Multicolored lines
==================

The example shows two ways to plot a line with a varying color defined by
a third value. The first example defines the color at each (x, y) point.
The second example defines the color between pairs of points, so the length
of the color value list is one less than the length of the x and y lists.

Color values at points
----------------------

"""

import warnings  # 导入警告模块

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

from matplotlib.collections import LineCollection  # 导入线集合类


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}  # 设置线段末端样式为平直
    default_kwargs.update(lc_kwargs)  # 更新传入的关键字参数

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)  # 将x坐标转换为numpy数组
    y = np.asarray(y)  # 将y坐标转换为numpy数组
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))  # 计算线段的中点x坐标
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))  # 计算线段的中点y坐标

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]  # 起始点坐标
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]  # 中间点坐标
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]  # 终点坐标
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)  # 合并起始、中间和终止坐标形成线段集合

    lc = LineCollection(segments, **default_kwargs)  # 创建线段集合对象
    lc.set_array(c)  # 设置每个线段的颜色，参数 c 是颜色数组

    return ax.add_collection(lc)
# -------------- Create and show plot --------------
# 创建并展示绘图

# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt

# 创建一些 x, y 和 color 值，这些值用于绘制彩色线条
t = np.linspace(-7.4, -0.5, 200)
x = 0.9 * np.sin(t)
y = 0.9 * np.cos(1.6 * t)
color = np.linspace(0, 2, t.size)

# 创建一个图形和轴，并在其上绘制彩色线条
fig1, ax1 = plt.subplots()
lines = colored_line(x, y, color, ax1, linewidth=10, cmap="plasma")
fig1.colorbar(lines)  # 添加一个颜色图例

# 设置轴的限制和刻度位置
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_xticks((-1, 0, 1))
ax1.set_yticks((-1, 0, 1))
ax1.set_title("Color at each point")  # 设置图表标题

plt.show()

####################################################################
# This method is designed to give a smooth impression when distances and color
# differences between adjacent points are not too large. The following example
# does not meet this criteria and by that serves to illustrate the segmentation
# and coloring mechanism.
# 该方法旨在在相邻点之间的距离和颜色差异不太大时提供平滑的印象。以下示例不符合这些条件，
# 因此用来说明分段和着色机制。
x = [0, 1, 2, 3, 4]
y = [0, 1, 2, 1, 1]
c = [1, 2, 3, 4, 5]
fig, ax = plt.subplots()
ax.scatter(x, y, c=c, cmap='rainbow')
colored_line(x, y, c=c, ax=ax, cmap='rainbow')

plt.show()

####################################################################
# Color values between points
# ---------------------------
# 点之间的颜色值

# 定义一个函数，用于在点之间绘制带有颜色的线条
def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Check color array size (LineCollection still works, but values are unused)
    # 检查颜色数组的大小（LineCollection仍然可以工作，但值不会被使用）
    if len(c) != len(x) - 1:
        warnings.warn(
            "The c argument should have a length one less than the length of x and y. "
            "If it has the same length, use the colored_line function instead."
        )

    # Create a set of line segments so that we can color them individually
    # 创建一组线段，以便我们可以对它们进行单独着色
    # 创建一个 N x 1 x 2 的数组，用来存储点的坐标，以便轻松地堆叠点来得到线段。
    # 对于线集合（Line Collection），segments 数组的形状应为 (numlines) x (每条线的点数) x 2（用于存储 x 和 y 坐标）。
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    
    # 将相邻的点堆叠起来，形成线段数组
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 使用给定的 lc_kwargs 参数创建 LineCollection 对象
    lc = LineCollection(segments, **lc_kwargs)

    # 设置用于颜色映射的数值
    lc.set_array(c)

    # 将 LineCollection 对象添加到坐标轴 ax 中
    return ax.add_collection(lc)
# -------------- 创建并展示图表 --------------

# 使用 numpy.linspace 创建一个包含 500 个点的等间距数列，范围从 0 到 3π
x = np.linspace(0, 3 * np.pi, 500)

# 计算 x 对应的正弦值
y = np.sin(x)

# 计算 x 点的中点的余弦值，作为 x 点的导数的近似值
dydx = np.cos(0.5 * (x[:-1] + x[1:]))

# 创建一个包含图表和轴对象的图形窗口
fig2, ax2 = plt.subplots()

# 调用 colored_line_between_pts 函数，绘制曲线并着色
line = colored_line_between_pts(x, y, dydx, ax2, linewidth=2, cmap="viridis")

# 在图表 ax2 上添加颜色条，并设置标签为 "dy/dx"
fig2.colorbar(line, ax=ax2, label="dy/dx")

# 设置 x 轴的显示范围
ax2.set_xlim(x.min(), x.max())

# 设置 y 轴的显示范围
ax2.set_ylim(-1.1, 1.1)

# 设置图表的标题
ax2.set_title("Color between points")

# 展示图表
plt.show()
```