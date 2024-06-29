# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\axes_margins.py`

```py
"""
======================================================
Controlling view limits using margins and sticky_edges
======================================================

The first figure in this example shows how to zoom in and out of a
plot using `~.Axes.margins` instead of `~.Axes.set_xlim` and
`~.Axes.set_ylim`. The second figure demonstrates the concept of
edge "stickiness" introduced by certain methods and artists and how
to effectively work around that.

"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy模块，用于数值计算

from matplotlib.patches import Polygon  # 从matplotlib.patches模块导入Polygon类


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)  # 定义一个函数f(t)，返回指数衰减乘以余弦波形的结果


t1 = np.arange(0.0, 3.0, 0.01)  # 创建一个从0到3（不含）的间隔为0.01的数组

ax1 = plt.subplot(212)  # 创建一个2行1列的子图中的第2个子图
ax1.margins(0.05)           # 设置边缘为0.05，0表示自适应
ax1.plot(t1, f(t1))  # 绘制图像

ax2 = plt.subplot(221)  # 创建一个2行2列的子图中的第1个子图
ax2.margins(2, 2)           # 设置边缘为2，大于0的值会放大图像
ax2.plot(t1, f(t1))  # 绘制图像
ax2.set_title('Zoomed out')  # 设置子图标题为'Zoomed out'

ax3 = plt.subplot(222)  # 创建一个2行2列的子图中的第2个子图
ax3.margins(x=0, y=-0.25)   # 设置x轴边缘为0，y轴边缘为-0.25，缩小中心部分的图像
ax3.plot(t1, f(t1))  # 绘制图像
ax3.set_title('Zoomed in')  # 设置子图标题为'Zoomed in'

plt.show()  # 显示图像


# %%
#
# On the "stickiness" of certain plotting methods
# """""""""""""""""""""""""""""""""""""""""""""""
#
# Some plotting functions make the axis limits "sticky" or immune to the will
# of the `~.Axes.margins` methods. For instance, `~.Axes.imshow` and
# `~.Axes.pcolor` expect the user to want the limits to be tight around the
# pixels shown in the plot. If this behavior is not desired, you need to set
# `~.Axes.use_sticky_edges` to `False`. Consider the following example:

y, x = np.mgrid[:5, 1:6]  # 创建一个5x5的y和一个5x5的x网格矩阵
poly_coords = [
    (0.25, 2.75), (3.25, 2.75),
    (2.25, 0.75), (0.25, 0.75)
]
fig, (ax1, ax2) = plt.subplots(ncols=2)  # 创建一个包含2列子图的图形

# Here we set the stickiness of the Axes object...
# ax1 we'll leave as the default, which uses sticky edges
# and we'll turn off stickiness for ax2
ax2.use_sticky_edges = False  # 关闭ax2的粘性边缘特性

for ax, status in zip((ax1, ax2), ('Is', 'Is Not')):
    cells = ax.pcolor(x, y, x+y, cmap='inferno', shading='auto')  # 创建一个根据x+y值着色的二维图像，这种方法会使用粘性边缘
    ax.add_patch(
        Polygon(poly_coords, color='forestgreen', alpha=0.5)
    )  # 向子图中添加一个半透明的绿色多边形，这个操作不受粘性边缘的影响
    ax.margins(x=0.1, y=0.05)  # 设置子图的边缘，使得中心区域放大
    ax.set_aspect('equal')  # 设置子图的纵横比一致
    ax.set_title(f'{status} Sticky')  # 设置子图的标题，包含状态信息

plt.show()  # 显示图像


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.margins` / `matplotlib.pyplot.margins`
#    - `matplotlib.axes.Axes.use_sticky_edges`
#    - `matplotlib.axes.Axes.pcolor` / `matplotlib.pyplot.pcolor`
#    - `matplotlib.patches.Polygon`
```