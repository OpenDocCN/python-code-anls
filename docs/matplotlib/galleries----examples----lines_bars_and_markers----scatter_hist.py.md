# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\scatter_hist.py`

```
"""
============================
Scatter plot with histograms
============================

Show the marginal distributions of a scatter plot as histograms at the sides of
the plot.

For a nice alignment of the main Axes with the marginals, two options are shown
below:

.. contents::
   :local:

While `.Axes.inset_axes` may be a bit more complex, it allows correct handling
of main Axes with a fixed aspect ratio.

An alternative method to produce a similar figure using the ``axes_grid1``
toolkit is shown in the :doc:`/gallery/axes_grid1/scatter_hist_locatable_axes`
example.  Finally, it is also possible to position all Axes in absolute
coordinates using `.Figure.add_axes` (not shown here).

Let us first define a function that takes x and y data as input, as well
as three Axes, the main Axes for the scatter, and two marginal Axes. It will
then create the scatter and histograms inside the provided Axes.
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# some random data
x = np.random.randn(1000)
y = np.random.randn(1000)


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    # 设置不显示X轴标签
    ax_histx.tick_params(axis="x", labelbottom=False)
    # 设置不显示Y轴标签
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    # 绘制散点图
    ax.scatter(x, y)

    # now determine nice limits by hand:
    # 设置直方图的bin宽度
    binwidth = 0.25
    # 计算数据的最大绝对值
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    # 计算限制范围
    lim = (int(xymax/binwidth) + 1) * binwidth

    # 设置bins范围
    bins = np.arange(-lim, lim + binwidth, binwidth)
    # 绘制X轴方向的直方图
    ax_histx.hist(x, bins=bins)
    # 绘制Y轴方向的直方图
    ax_histy.hist(y, bins=bins, orientation='horizontal')


# %%
#
# Defining the Axes positions using a gridspec
# --------------------------------------------
#
# We define a gridspec with unequal width- and height-ratios to achieve desired
# layout.  Also see the :ref:`arranging_axes` tutorial.

# Start with a square Figure.
# 创建一个正方形的Figure
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal Axes and the main Axes in both directions.
# Also adjust the subplot parameters for a square plot.
# 创建一个gridspec，定义两行两列，宽高比例分别为(4:1)和(1:4)，调整子图参数以得到正方形布局
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
# 创建主要的Axes和两个边缘的Axes
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
# 绘制散点图和边缘直方图
scatter_hist(x, y, ax, ax_histx, ax_histy)


# %%
#
# Defining the Axes positions using inset_axes
# --------------------------------------------
#
# `~.Axes.inset_axes` can be used to position marginals *outside* the main
# Axes.  The advantage of doing so is that the aspect ratio of the main Axes
# can be fixed, and the marginals will always be drawn relative to the position
# of the Axes.

# Create a Figure, which doesn't have to be square.
# 创建一个不需要正方形布局的Figure
fig = plt.figure(layout='constrained')
# 创建主要的图形坐标系，将图形空间的顶部和右侧各留出25%的空间用于放置边缘图。
ax = fig.add_gridspec(top=0.75, right=0.75).subplots()

# 设置主要坐标系的纵横比为1。
ax.set(aspect=1)

# 创建边缘坐标系，其大小为主要坐标系的25%。注意，插入的坐标系位于主坐标系的外部（右侧和顶部），
# 通过指定大于1的坐标系坐标。坐标系坐标小于0同样可以指定在主坐标系的左侧和底部的位置。
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

# 绘制散点图和边缘图。
scatter_hist(x, y, ax, ax_histx, ax_histy)

# 显示图形
plt.show()
```