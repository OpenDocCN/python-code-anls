# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\fill.py`

```
"""
==============
Filled polygon
==============

`~.Axes.fill()` draws a filled polygon based on lists of point
coordinates *x*, *y*.

This example uses the `Koch snowflake`_ as an example polygon.

.. _Koch snowflake: https://en.wikipedia.org/wiki/Koch_snowflake

"""

import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入NumPy库，用于数值计算


def koch_snowflake(order, scale=10):
    """
    Return two lists x, y of point coordinates of the Koch snowflake.

    Parameters
    ----------
    order : int
        The recursion depth.
    scale : float
        The extent of the snowflake (edge length of the base triangle).
    """
    def _koch_snowflake_complex(order):
        if order == 0:
            # initial triangle
            angles = np.array([0, 120, 240]) + 90
            return scale / np.sqrt(3) * np.exp(np.deg2rad(angles) * 1j)
        else:
            ZR = 0.5 - 0.5j * np.sqrt(3) / 3

            p1 = _koch_snowflake_complex(order - 1)  # 递归调用，获取起始点
            p2 = np.roll(p1, shift=-1)  # 循环移位，获取结束点
            dp = p2 - p1  # 连接向量

            new_points = np.empty(len(p1) * 4, dtype=np.complex128)
            new_points[::4] = p1
            new_points[1::4] = p1 + dp / 3
            new_points[2::4] = p1 + dp * ZR
            new_points[3::4] = p1 + dp / 3 * 2
            return new_points

    points = _koch_snowflake_complex(order)  # 调用内部函数生成Koch雪花的复数点
    x, y = points.real, points.imag  # 获取实部和虚部作为x和y坐标
    return x, y


# %%
# Basic usage:

x, y = koch_snowflake(order=5)  # 生成阶数为5的Koch雪花坐标

plt.figure(figsize=(8, 8))  # 创建8x8英寸大小的图像
plt.axis('equal')  # 设置坐标轴比例为相等
plt.fill(x, y)  # 填充多边形，使用生成的x和y坐标
plt.show()  # 显示图像

# %%
# Use keyword arguments *facecolor* and *edgecolor* to modify the colors
# of the polygon. Since the *linewidth* of the edge is 0 in the default
# Matplotlib style, we have to set it as well for the edge to become visible.

x, y = koch_snowflake(order=2)  # 生成阶数为2的Koch雪花坐标

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3),
                                    subplot_kw={'aspect': 'equal'})  # 创建1行3列的子图，设置纵横比为相等
ax1.fill(x, y)  # 在第一个子图中填充多边形，使用默认颜色
ax2.fill(x, y, facecolor='lightsalmon', edgecolor='orangered', linewidth=3)  # 在第二个子图中填充多边形，设置填充颜色和边框颜色，并增加边框宽度
ax3.fill(x, y, facecolor='none', edgecolor='purple', linewidth=3)  # 在第三个子图中填充多边形，设置透明填充和紫色边框，并增加边框宽度

plt.show()  # 显示图像

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.fill` / `matplotlib.pyplot.fill`
#    - `matplotlib.axes.Axes.axis` / `matplotlib.pyplot.axis`
```