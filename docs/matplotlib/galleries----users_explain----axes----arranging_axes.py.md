# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\arranging_axes.py`

```
"""
.. redirect-from:: /tutorials/intermediate/gridspec
.. redirect-from:: /tutorials/intermediate/arranging_axes

.. _arranging_axes:

===================================
Arranging multiple Axes in a Figure
===================================

Often more than one Axes is wanted on a figure at a time, usually
organized into a regular grid.  Matplotlib has a variety of tools for
working with grids of Axes that have evolved over the history of the library.
Here we will discuss the tools we think users should use most often, the tools
that underpin how Axes are organized, and mention some of the older tools.

.. note::

    Matplotlib uses *Axes* to refer to the drawing area that contains
    data, x- and y-axis, ticks, labels, title, etc. See :ref:`figure_parts`
    for more details.  Another term that is often used is "subplot", which
    refers to an Axes that is in a grid with other Axes objects.

Overview
========

Create grid-shaped combinations of Axes
---------------------------------------

`~matplotlib.pyplot.subplots`
    主要用于创建图形和Axes的网格。一次性创建并放置所有的Axes，并返回一个Axes数组对象。
    参见 `.Figure.subplots`。

or

`~matplotlib.pyplot.subplot_mosaic`
    简单地创建图形和Axes的网格，具有更灵活的功能，允许Axes跨越多行或多列。
    返回一个带标签的字典，而不是一个数组。参见 `.Figure.subplot_mosaic` 和 :ref:`mosaic`。

Sometimes it is natural to have more than one distinct group of Axes grids,
in which case Matplotlib has the concept of `.SubFigure`:

`~matplotlib.figure.SubFigure`
    在一个图中的虚拟子图。

Underlying tools
----------------

Underlying these are the concept of a `~.gridspec.GridSpec` and
a `~.SubplotSpec`:

`~matplotlib.gridspec.GridSpec`
    指定将subplot放置在其中的网格的几何形状。需要设置网格的行数和列数。
    可以选择调整subplot的布局参数（例如左、右等）。

`~matplotlib.gridspec.SubplotSpec`
    指定给定 `.GridSpec` 中subplot的位置。

.. _fixed_size_axes:

Adding single Axes at a time
----------------------------

The above functions create all Axes in a single function call.  It is also
possible to add Axes one at a time, and this was originally how Matplotlib
used to work.  Doing so is generally less elegant and flexible, though
sometimes useful for interactive work or to place an Axes in a custom
location:

`~matplotlib.figure.Figure.add_axes`
    在指定的 `[left, bottom, width, height]` 位置上，以图形宽度或高度的分数添加单个Axes。

`~matplotlib.pyplot.subplot` or `.Figure.add_subplot`
    在图形上添加单个subplot，使用基于1的索引（继承自...
    Matlab).  Columns and rows can be spanned by specifying a range of grid
    cells.



    # 在 MATLAB 风格的注释中，通过指定网格单元格的范围，可以跨列和行进行跨越
# %%
# As a simple example of manually adding an Axes *ax*, lets add a 3 inch x 2 inch
# Axes to a 4 inch x 3 inch figure.  Note that the location of the subplot is
# defined as [left, bottom, width, height] in figure-normalized units:

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库
import numpy as np  # 导入 numpy 库

w, h = 4, 3  # 定义 figure 的宽度和高度
margin = 0.5  # 定义边距
fig = plt.figure(figsize=(w, h), facecolor='lightblue')  # 创建一个指定大小和背景颜色的 figure 对象
ax = fig.add_axes([margin / w, margin / h, (w - 2 * margin) / w,
                      (h - 2 * margin) / h])  # 在 figure 上添加一个 Axes 对象，并指定其位置和大小

# %%
# High-level methods for making grids
# ===================================
#
# Basic 2x2 grid
# --------------
#
# We can create a basic 2-by-2 grid of Axes using
# `~matplotlib.pyplot.subplots`.  It returns a `~matplotlib.figure.Figure`
# instance and an array of `~matplotlib.axes.Axes` objects.  The Axes
# objects can be used to access methods to place artists on the Axes; here
# we use `~.Axes.annotate`, but other examples could be `~.Axes.plot`,
# `~.Axes.pcolormesh`, etc.

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5),
                        layout="constrained")
# 创建一个包含 2 行 2 列的子图网格，使用 constrained layout 来自动调整子图的布局
# 返回一个 figure 对象和一个二维数组 axs，其中包含四个 Axes 对象
# 可以使用 Axes 对象的方法在子图上添加图形元素，如 annotate、plot、pcolormesh 等

# add an artist, in this case a nice label in the middle...
for row in range(2):
    for col in range(2):
        axs[row, col].annotate(f'axs[{row}, {col}]', (0.5, 0.5),
                               transform=axs[row, col].transAxes,
                               ha='center', va='center', fontsize=18,
                               color='darkgrey')
fig.suptitle('plt.subplots()')  # 设置主标题

# %%
# We will annotate a lot of Axes, so let's encapsulate the annotation, rather
# than having that large piece of annotation code every time we need it:

def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")
# 定义一个函数用于在 Axes 上添加注释，减少重复代码

# %%
# The same effect can be achieved with `~.pyplot.subplot_mosaic`,
# but the return type is a dictionary instead of an array, where the user
# can give the keys useful meanings.  Here we provide two lists, each list
# representing a row, and each element in the list a key representing the
# column.

fig, axd = plt.subplot_mosaic([['upper left', 'upper right'],
                               ['lower left', 'lower right']],
                              figsize=(5.5, 3.5), layout="constrained")
# 使用 subplot_mosaic 函数创建类似效果的子图网格，返回一个字典而不是数组
# 用户可以为每个子图指定有意义的键值，这里提供了两个列表，每个列表代表一行，列表中的元素代表列

for k, ax in axd.items():
    annotate_axes(ax, f'axd[{k!r}]', fontsize=14)
fig.suptitle('plt.subplot_mosaic()')  # 设置主标题

# %%
#
# Grids of fixed-aspect ratio Axes
# --------------------------------
#
# Fixed-aspect ratio Axes are common for images or maps.  However, they
# present a challenge to layout because two sets of constraints are being
# imposed on the size of the Axes - that they fit in the figure and that they
# have a set aspect ratio.  This leads to large gaps between Axes by default:
#
fig, axs = plt.subplots(2, 2, layout="constrained",
                        figsize=(5.5, 3.5), facecolor='lightblue')
# 创建一个包含 2x2 子图的 Figure 对象，使用 constrained 布局
# 设置子图的大小为 (5.5, 3.5) 英寸，背景色为浅蓝色

for ax in axs.flat:
    ax.set_aspect(1)
# 对每个子图设置相同的纵横比，使其看起来是正方形

fig.suptitle('Fixed aspect Axes')
# 设置整个 Figure 的标题为 'Fixed aspect Axes'

# %%
# One way to address this is to change the aspect of the figure to be close
# to the aspect ratio of the Axes, however that requires trial and error.
# Matplotlib also supplies ``layout="compressed"``, which will work with
# simple grids to reduce the gaps between Axes.  (The ``mpl_toolkits`` also
# provides `~.mpl_toolkits.axes_grid1.axes_grid.ImageGrid` to accomplish
# a similar effect, but with a non-standard Axes class).

fig, axs = plt.subplots(2, 2, layout="compressed", figsize=(5.5, 3.5),
                        facecolor='lightblue')
# 创建一个包含 2x2 子图的新 Figure 对象，使用 compressed 布局
# 设置子图的大小为 (5.5, 3.5) 英寸，背景色为浅蓝色

for ax in axs.flat:
    ax.set_aspect(1)
# 对每个子图设置相同的纵横比，使其看起来是正方形

fig.suptitle('Fixed aspect Axes: compressed')
# 设置整个 Figure 的标题为 'Fixed aspect Axes: compressed'

# %%
# Axes spanning rows or columns in a grid
# ---------------------------------------
#
# Sometimes we want Axes to span rows or columns of the grid.
# There are actually multiple ways to accomplish this, but the most
# convenient is probably to use `~.pyplot.subplot_mosaic` by repeating one
# of the keys:

fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              figsize=(5.5, 3.5), layout="constrained")
# 创建一个包含指定布局的子图网格，并返回每个 Axes 的字典 axd
# 每个子图网格的大小为 (5.5, 3.5) 英寸，使用 constrained 布局

for k, ax in axd.items():
    annotate_axes(ax, f'axd[{k!r}]', fontsize=14)
# 对每个 Axes 进行标注，显示其在 axd 字典中的键名和字体大小

fig.suptitle('plt.subplot_mosaic()')
# 设置整个 Figure 的标题为 'plt.subplot_mosaic()'

# %%
# See below for the description of how to do the same thing using
# `~matplotlib.gridspec.GridSpec` or `~matplotlib.pyplot.subplot2grid`.
#
# Variable widths or heights in a grid
# ------------------------------------
#
# Both `~.pyplot.subplots` and `~.pyplot.subplot_mosaic` allow the rows
# in the grid to be different heights, and the columns to be different
# widths using the *gridspec_kw* keyword argument.
# Spacing parameters accepted by `~matplotlib.gridspec.GridSpec`
# can be passed to `~matplotlib.pyplot.subplots` and
# `~matplotlib.pyplot.subplot_mosaic`:

gs_kw = dict(width_ratios=[1.4, 1], height_ratios=[1, 2])
fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              gridspec_kw=gs_kw, figsize=(5.5, 3.5),
                              layout="constrained")
# 创建一个包含指定布局和网格参数的子图网格
# 指定每行的宽度比例为 [1.4, 1]，每列的高度比例为 [1, 2]
# 网格的大小为 (5.5, 3.5) 英寸，使用 constrained 布局

for k, ax in axd.items():
    annotate_axes(ax, f'axd[{k!r}]', fontsize=14)
# 对每个 Axes 进行标注，显示其在 axd 字典中的键名和字体大小

fig.suptitle('plt.subplot_mosaic()')
# 设置整个 Figure 的标题为 'plt.subplot_mosaic()'

# %%
# .. _nested_axes_layouts:
#
# Nested Axes layouts
# -------------------
#
# Sometimes it is helpful to have two or more grids of Axes that
# may not need to be related to one another.  The most simple way to
# accomplish this is to use `.Figure.subfigures`.  Note that the subfigure
# layouts are independent, so the Axes spines in each subfigure are not
# necessarily aligned.  See below for a more verbose way to achieve the same
# effect with `~.gridspec.GridSpecFromSubplotSpec`.

fig = plt.figure(layout="constrained")
# 创建一个新的 Figure 对象，使用 constrained 布局
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1.5, 1.])
# 创建包含1行2列的子图集合subfigs，设置水平间距为0.07，指定宽度比例为[1.5, 1.]
axs0 = subfigs[0].subplots(2, 2)
# 在subfigs的第一个子图中创建一个2行2列的Axes网格，存储在axs0中
subfigs[0].set_facecolor('lightblue')
# 设置subfigs的第一个子图的背景色为浅蓝色
subfigs[0].suptitle('subfigs[0]\nLeft side')
# 给subfigs的第一个子图添加总标题，文本为'subfigs[0]\nLeft side'
subfigs[0].supxlabel('xlabel for subfigs[0]')
# 给subfigs的第一个子图添加x轴标签，文本为'xlabel for subfigs[0]'

axs1 = subfigs[1].subplots(3, 1)
# 在subfigs的第二个子图中创建一个3行1列的Axes网格，存储在axs1中
subfigs[1].suptitle('subfigs[1]')
# 给subfigs的第二个子图添加总标题，文本为'subfigs[1]'
subfigs[1].supylabel('ylabel for subfigs[1]')
# 给subfigs的第二个子图添加y轴标签，文本为'ylabel for subfigs[1]'
# 创建一个新的图形对象 `fig`，设置布局为 `None`，背景色为 'lightblue'
fig = plt.figure(layout=None, facecolor='lightblue')

# 添加一个 3x3 的网格布局 `gs` 到图形 `fig` 中，
# 设置左边界为 0.05，右边界为 0.75，水平间距为 0.1，垂直间距为 0.05
gs = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.75,
                      hspace=0.1, wspace=0.05)

# 在网格布局 `gs` 中添加一个子图 `ax0`，占据前两行所有列
ax0 = fig.add_subplot(gs[:-1, :])
annotate_axes(ax0, 'ax0')  # 标注子图 `ax0`

# 在网格布局 `gs` 中添加一个子图 `ax1`，占据最后一行的前两列
ax1 = fig.add_subplot(gs[-1, :-1])
annotate_axes(ax1, 'ax1')  # 标注子图 `ax1`

# 在网格布局 `gs` 中添加一个子图 `ax2`，占据最后一行最后一列
ax2 = fig.add_subplot(gs[-1, -1])
annotate_axes(ax2, 'ax2')  # 标注子图 `ax2`

# 设置整个图形的标题为 'Manual gridspec with right=0.75'
fig.suptitle('Manual gridspec with right=0.75')

# %%
# 使用 SubplotSpec 创建嵌套布局
# -------------------------------
#
# 可以使用 `~.gridspec.SubplotSpec.subgridspec` 创建类似 `~.Figure.subfigures` 的嵌套布局。
# 在这里，子图的坐标轴脊柱是对齐的。
#
# 注意，这也可以通过更详细的 `.gridspec.GridSpecFromSubplotSpec` 实现。

# 创建一个新的图形对象 `fig`，设置布局为 'constrained'
fig = plt.figure(layout="constrained")

# 在图形 `fig` 上添加一个 1x2 的网格布局 `gs0`
gs0 = fig.add_gridspec(1, 2)

# 在 `gs0` 的第一个格子上创建一个 2x2 的子网格布局 `gs00`
gs00 = gs0[0].subgridspec(2, 2)

# 在 `gs0` 的第二个格子上创建一个 3x1 的子网格布局 `gs01`
gs01 = gs0[1].subgridspec(3, 1)

# 循环创建 `gs00` 中的子图 `ax`，并标注其名称
for a in range(2):
    for b in range(2):
        ax = fig.add_subplot(gs00[a, b])
        annotate_axes(ax, f'axLeft[{a}, {b}]', fontsize=10)
        if a == 1 and b == 1:
            ax.set_xlabel('xlabel')

# 循环创建 `gs01` 中的子图 `ax`，并标注其名称
for a in range(3):
    ax = fig.add_subplot(gs01[a])
    annotate_axes(ax, f'axRight[{a}, {b}]')
    if a == 2:
        ax.set_ylabel('ylabel')

# 设置整个图形的标题为 'nested gridspecs'
fig.suptitle('nested gridspecs')

# %%
# 下面是更复杂的嵌套 *GridSpec* 的例子：我们创建一个外部的 4x4 网格，每个单元格包含一个内部的 3x3 网格的坐标轴。
# 我们通过在每个内部 3x3 网格中隐藏适当的脊柱来勾勒外部的 4x4 网格。

# 定义一个函数 `squiggle_xy` 来生成坐标点
def squiggle_xy(a, b, c, d, i=np.arange(0.0, 2*np.pi, 0.05)):
    return np.sin(i*a)*np.cos(i*b), np.sin(i*c)*np.cos(i*d)

# 创建一个新的图形对象 `fig`，设置大小为 (8, 8)，布局为 'constrained'
fig = plt.figure(figsize=(8, 8), layout='constrained')

# 添加一个外部的 4x4 网格布局 `outer_grid` 到图形 `fig` 中
outer_grid = fig.add_gridspec(4, 4, wspace=0, hspace=0)

# 循环创建外部网格 `outer_grid` 中的每个单元格中的内部 3x3 网格 `inner_grid`
for a in range(4):
    for b in range(4):
        inner_grid = outer_grid[a, b].subgridspec(3, 3, wspace=0, hspace=0)
        axs = inner_grid.subplots()  # 在内部网格中创建所有的子图
        for (c, d), ax in np.ndenumerate(axs):
            ax.plot(*squiggle_xy(a + 1, b + 1, c + 1, d + 1))
            ax.set(xticks=[], yticks=[])  # 设置坐标轴的刻度为空

# 仅显示外部脊柱
for ax in fig.get_axes():
    ss = ax.get_subplotspec()
    ax.spines.top.set_visible(ss.is_first_row())
    ax.spines.bottom.set_visible(ss.is_last_row())
    ax.spines.left.set_visible(ss.is_first_col())
    ax.spines.right.set_visible(ss.is_last_col())

# 显示图形
plt.show()

# %%
#
# 进一步阅读
# ============
#
# - 关于 :ref:`subplot mosaic <mosaic>` 的更多细节。
# - 关于 :ref:`constrained layout <constrainedlayout_guide>` 的更多细节，用于在大多数示例中对齐间距。
#
# .. admonition:: References
#
#    This section documents the usage of various functions, methods, classes, and modules in this example:
#
#    - `matplotlib.pyplot.subplots`: Create a figure and a set of subplots.
#    - `matplotlib.pyplot.subplot_mosaic`: Create a subplot layout defined by a mosaic of axes.
#    - `matplotlib.figure.Figure.add_gridspec`: Add a GridSpec with a given number of rows and columns to a Figure.
#    - `matplotlib.figure.Figure.add_subplot`: Add an Axes to the Figure instance.
#    - `matplotlib.gridspec.GridSpec`: Define a grid layout to place subplots within a figure.
#    - `matplotlib.gridspec.SubplotSpec.subgridspec`: Create a GridSpec within a SubplotSpec for nested grids.
#    - `matplotlib.gridspec.GridSpecFromSubplotSpec`: Create a GridSpec from a SubplotSpec for customizing subplot layouts.
```