# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\scatter_hist_locatable_axes.py`

```py
"""
==================================
Scatter Histogram (Locatable Axes)
==================================

Show the marginal distributions of a scatter plot as histograms at the sides of
the plot.

For a nice alignment of the main Axes with the marginals, the Axes positions
are defined by a ``Divider``, produced via `.make_axes_locatable`.  Note that
the ``Divider`` API allows setting Axes sizes and pads in inches, which is its
main feature.

If one wants to set Axes sizes and pads relative to the main Figure, see the
:doc:`/gallery/lines_bars_and_markers/scatter_hist` example.
"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from mpl_toolkits.axes_grid1 import make_axes_locatable  # 导入make_axes_locatable函数，用于创建可定位的Axes

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设置随机数种子，以便结果可重现

# the random data
x = np.random.randn(1000)  # 生成1000个服从标准正态分布的随机数作为x轴数据
y = np.random.randn(1000)  # 生成1000个服从标准正态分布的随机数作为y轴数据

fig, ax = plt.subplots(figsize=(5.5, 5.5))  # 创建图形和主要的Axes，指定图形大小为5.5x5.5英寸

# the scatter plot:
ax.scatter(x, y)  # 绘制散点图，以x和y为坐标

# Set aspect of the main Axes.
ax.set_aspect(1.)  # 设置主要Axes的纵横比为1（即使得x和y轴的单位长度相等）

# create new Axes on the right and on the top of the current Axes
divider = make_axes_locatable(ax)
# below height and pad are in inches
ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

# make some labels invisible
ax_histx.xaxis.set_tick_params(labelbottom=False)  # 设置顶部直方图x轴标签不可见
ax_histy.yaxis.set_tick_params(labelleft=False)  # 设置右侧直方图y轴标签不可见

# now determine nice limits by hand:
binwidth = 0.25  # 设置直方图的柱宽度为0.25
xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))  # 计算x和y的绝对值的最大值
lim = (int(xymax/binwidth) + 1)*binwidth  # 根据柱宽度计算合适的极限值

bins = np.arange(-lim, lim + binwidth, binwidth)  # 计算直方图的柱边界
ax_histx.hist(x, bins=bins)  # 绘制顶部直方图
ax_histy.hist(y, bins=bins, orientation='horizontal')  # 绘制右侧直方图，水平方向

# the xaxis of ax_histx and yaxis of ax_histy are shared with ax,
# thus there is no need to manually adjust the xlim and ylim of these
# axis.

ax_histx.set_yticks([0, 50, 100])  # 设置顶部直方图y轴刻度位置
ax_histy.set_xticks([0, 50, 100])  # 设置右侧直方图x轴刻度位置

plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable`
#    - `matplotlib.axes.Axes.set_aspect`
#    - `matplotlib.axes.Axes.scatter`
#    - `matplotlib.axes.Axes.hist`
```