# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\tight_layout_guide.py`

```py
"""
.. redirect-from:: /tutorial/intermediate/tight_layout_guide

.. _tight_layout_guide:

==================
Tight layout guide
==================

How to use tight-layout to fit plots within your figure cleanly.

*tight_layout* automatically adjusts subplot params so that the
subplot(s) fits in to the figure area. This is an experimental
feature and may not work for some cases. It only checks the extents
of ticklabels, axis labels, and titles.

An alternative to *tight_layout* is :ref:`constrained_layout
<constrainedlayout_guide>`.


Simple example
==============

With the default Axes positioning, the axes title, axis labels, or tick labels
can sometimes go outside the figure area, and thus get clipped.
"""

# sphinx_gallery_thumbnail_number = 7

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['savefig.facecolor'] = "0.8"


def example_plot(ax, fontsize=12):
    # 绘制简单的折线图
    ax.plot([1, 2])

    # 设置刻度标签数量
    ax.locator_params(nbins=3)
    # 设置 x 轴标签
    ax.set_xlabel('x-label', fontsize=fontsize)
    # 设置 y 轴标签
    ax.set_ylabel('y-label', fontsize=fontsize)
    # 设置图表标题
    ax.set_title('Title', fontsize=fontsize)

plt.close('all')
# 创建一个包含单个子图的图表
fig, ax = plt.subplots()
# 调用示例函数来绘制图表
example_plot(ax, fontsize=24)

# %%
# 为了避免标签超出图表区域被裁剪，需要调整 Axes 的位置。对于子图，
# 可以通过手动调整 subplot 参数来完成，也可以使用 `.Figure.tight_layout` 自动完成调整。
fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.tight_layout()

# %%
# 注意，:func:`matplotlib.pyplot.tight_layout` 只有在被调用时才会调整 subplot 参数。
# 如果想要每次图表重新绘制时都进行调整，可以调用 ``fig.set_tight_layout(True)``，或者等效地设置 :rc:`figure.autolayout` 为 ``True``。
#
# 当有多个子图时，经常会看到不同 Axes 的标签重叠在一起。
plt.close('all')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)

# %%
# :func:`~matplotlib.pyplot.tight_layout` 也会调整子图之间的间距，以尽量减少重叠。
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()

# %%
# :func:`~matplotlib.pyplot.tight_layout` 还可以接受 *pad*, *w_pad* 和 *h_pad* 等关键字参数。
# 这些参数控制图表边界周围和子图之间的额外填充。填充以 fontsize 的比例指定。
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# %%
# 即使子图的大小不同，只要它们的网格规格兼容，:func:`~matplotlib.pyplot.tight_layout` 也能正常工作。
# 在下面的例子中，*ax1* 和 *ax2* 是一个 2x2 网格中的子图，而 *ax3* 是一个 1x2 网格中的子图。
plt.close('all')
fig = plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(223)
ax3 = plt.subplot(122)


# 创建一个包含3个子图的画布，并将它们分配给不同的Axes对象
ax1 = plt.subplot(221)
ax2 = plt.subplot(223)
ax3 = plt.subplot(122)



example_plot(ax1)
example_plot(ax2)
example_plot(ax3)


# 在每个子图上调用示例绘图函数example_plot，用于填充每个Axes对象的
# 创建一个新的图形对象和一个子图对象，指定子图的尺寸为宽4高3
fig, ax = plt.subplots(figsize=(4, 3))

# 在子图上绘制一条简单的折线，指定标签为'A simple plot'
lines = ax.plot(range(10), label='A simple plot')

# 在子图上创建图例，设置其位置为相对坐标(0.7, 0.5)，位于左上角
ax.legend(bbox_to_anchor=(0.7, 0.5), loc='center left',)

# 调整图形布局，使图形内容紧凑显示
fig.tight_layout()

# 显示图形
plt.show()

# %%
# 然而，有时不希望图例参与边界框计算（比如在保存图像时使用``fig.savefig('outname.png', bbox_inches='tight')``）。
# 为了排除图例对边界框计算的影响，我们简单地设置图例的``leg.set_in_layout(False)``，这样图例将被忽略。

# 创建一个新的图形对象和一个子图对象，指定子图的尺寸为宽4高3
fig, ax = plt.subplots(figsize=(4, 3))

# 在子图上绘制一条简单的折线，指定标签为'B simple plot'
lines = ax.plot(range(10), label='B simple plot')

# 在子图上创建图例，设置其位置为相对坐标(0.7, 0.5)，位于左上角
leg = ax.legend(bbox_to_anchor=(0.7, 0.5), loc='center left',)

# 设置图例不参与布局计算
leg.set_in_layout(False)

# 调整图形布局，使图形内容紧凑显示
fig.tight_layout()

# 显示图形
plt.show()

# %%
# 使用AxesGrid1
# =============
#
# 提供对 :mod:`mpl_toolkits.axes_grid1` 的有限支持。

# 导入AxesGrid1工具箱中的Grid
from mpl_toolkits.axes_grid1 import Grid

# 关闭所有已存在的图形
plt.close('all')

# 创建一个新的图形对象
fig = plt.figure()

# 在图形上创建Grid对象，设置位置为111，行列数为2行2列，轴间距为0.25，标签模式为L
grid = Grid(fig, rect=111, nrows_ncols=(2, 2),
            axes_pad=0.25, label_mode='L')

# 在每个网格中绘制示例图
for ax in grid:
    example_plot(ax)

# 设置标题不可见
ax.title.set_visible(False)

# 调整图形布局，使图形内容紧凑显示
plt.tight_layout()

# %%
# Colorbar
# ========
#
# 如果使用 `.Figure.colorbar` 创建颜色条，创建的颜色条将绘制在子图上，只要父轴也是一个子图，因此 `.Figure.tight_layout` 将起作用。

# 关闭所有已存在的图形
plt.close('all')

# 创建一个10x10的数组
arr = np.arange(100).reshape((10, 10))

# 创建一个新的图形对象，设置尺寸为宽4高4
fig = plt.figure(figsize=(4, 4))

# 在图形上显示数组的热图，设置插值方式为'none'
im = plt.imshow(arr, interpolation="none")

# 创建颜色条
plt.colorbar(im)

# 调整图形布局，使图形内容紧凑显示
plt.tight_layout()

# %%
# 另一种选择是使用AxesGrid1工具箱显式为颜色条创建一个轴。

# 导入AxesGrid1工具箱中的make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 关闭所有已存在的图形
plt.close('all')

# 创建一个10x10的数组
arr = np.arange(100).reshape((10, 10))

# 创建一个新的图形对象，设置尺寸为宽4高4
fig = plt.figure(figsize=(4, 4))

# 在图形上显示数组的热图，设置插值方式为'none'
im = plt.imshow(arr, interpolation="none")

# 使用make_axes_locatable创建一个分隔器，将轴分隔为右侧5%，底部3%的位置，轴间距为3%
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")

# 创建颜色条，指定轴为cax
plt.colorbar(im, cax=cax)

# 调整图形布局，使图形内容紧凑显示
plt.tight_layout()
```