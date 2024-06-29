# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\subplots_demo.py`

```
"""
=================================================
Creating multiple subplots using ``plt.subplots``
=================================================

`.pyplot.subplots` creates a figure and a grid of subplots with a single call,
while providing reasonable control over how the individual plots are created.
For more advanced use cases you can use `.GridSpec` for a more general subplot
layout or `.Figure.add_subplot` for adding subplots at arbitrary locations
within the figure.
"""

# 引入 matplotlib.pyplot 库并使用 plt 别名
import matplotlib.pyplot as plt
# 引入 numpy 库并使用 np 别名
import numpy as np

# Some example data to display
# 创建一个包含 400 个点的等间距数组
x = np.linspace(0, 2 * np.pi, 400)
# 计算 x 的平方再求正弦值，作为 y 的值
y = np.sin(x ** 2)

# %%
# A figure with just one subplot
# """"""""""""""""""""""""""""""
#
# ``subplots()`` without arguments returns a `.Figure` and a single
# `~.axes.Axes`.
#
# This is actually the simplest and recommended way of creating a single
# Figure and Axes.

# 创建一个仅包含一个 subplot 的 Figure 和 Axes
fig, ax = plt.subplots()
# 在 Axes 上绘制 x 和 y 的曲线
ax.plot(x, y)
# 设置 Axes 的标题
ax.set_title('A single plot')

# %%
# Stacking subplots in one direction
# """"""""""""""""""""""""""""""""""
#
# The first two optional arguments of `.pyplot.subplots` define the number of
# rows and columns of the subplot grid.
#
# When stacking in one direction only, the returned ``axs`` is a 1D numpy array
# containing the list of created Axes.

# 创建一个包含 2 个 subplot 的 Figure，沿垂直方向堆叠
fig, axs = plt.subplots(2)
# 设置 Figure 的标题
fig.suptitle('Vertically stacked subplots')
# 在第一个 Axes 上绘制 x 和 y 的曲线
axs[0].plot(x, y)
# 在第二个 Axes 上绘制 x 和 -y 的曲线
axs[1].plot(x, -y)

# %%
# If you are creating just a few Axes, it's handy to unpack them immediately to
# dedicated variables for each Axes. That way, we can use ``ax1`` instead of
# the more verbose ``axs[0]``.

# 创建一个包含 2 个 subplot 的 Figure，沿垂直方向堆叠，并将它们分别解包给 ax1 和 ax2
fig, (ax1, ax2) = plt.subplots(2)
# 设置 Figure 的标题
fig.suptitle('Vertically stacked subplots')
# 在 ax1 上绘制 x 和 y 的曲线
ax1.plot(x, y)
# 在 ax2 上绘制 x 和 -y 的曲线
ax2.plot(x, -y)

# %%
# To obtain side-by-side subplots, pass parameters ``1, 2`` for one row and two
# columns.

# 创建一个包含 2 个 subplot 的 Figure，沿水平方向堆叠
fig, (ax1, ax2) = plt.subplots(1, 2)
# 设置 Figure 的标题
fig.suptitle('Horizontally stacked subplots')
# 在 ax1 上绘制 x 和 y 的曲线
ax1.plot(x, y)
# 在 ax2 上绘制 x 和 -y 的曲线
ax2.plot(x, -y)

# %%
# Stacking subplots in two directions
# """""""""""""""""""""""""""""""""""
#
# When stacking in two directions, the returned ``axs`` is a 2D NumPy array.
#
# If you have to set parameters for each subplot it's handy to iterate over
# all subplots in a 2D grid using ``for ax in axs.flat:``.

# 创建一个包含 2x2 的 subplot 网格的 Figure
fig, axs = plt.subplots(2, 2)
# 在左上角的 Axes 上绘制 x 和 y 的曲线，并设置标题
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0, 0]')
# 在右上角的 Axes 上绘制 x 和 y 的橙色曲线，并设置标题
axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Axis [0, 1]')
# 在左下角的 Axes 上绘制 x 和 -y 的绿色曲线，并设置标题
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1, 0]')
# 在右下角的 Axes 上绘制 x 和 -y 的红色曲线，并设置标题
axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Axis [1, 1]')

# 设置所有 Axes 的 x 和 y 标签
for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# 隐藏顶部 subplot 的 x 标签和刻度，以及右侧 subplot 的 y 刻度
for ax in axs.flat:
    ax.label_outer()

# %%
# You can use tuple-unpacking also in 2D to assign all subplots to dedicated
# variables:

# 创建一个包含 2x2 的 subplot 网格的 Figure，并将每个 subplot 解包到各自的变量中
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# 设置 Figure 的标题
fig.suptitle('Sharing x per column, y per row')
# 在 ax1 上绘制 x 和 y 的曲线
ax1.plot(x, y)
# 在 ax2 上绘制 x 和 y 的平方的橙色曲线
ax2.plot(x, y**2, 'tab:orange')
# 在 ax3 上绘制 x 和 -y 的绿色曲线
ax3.plot(x, -y, 'tab:green')
# 使用 'tab:red' 颜色绘制 x 和 -y**2 的折线图，并将图表添加到 ax4 上
ax4.plot(x, -y**2, 'tab:red')

# 遍历图表 fig 中的每个 Axes 对象，并调用 ax.label_outer() 方法隐藏非边缘的轴标签和刻度
for ax in fig.get_axes():
    ax.label_outer()

# %%
# 共享坐标轴
# """"""""""""
#
# 默认情况下，每个 Axes 对象都会单独缩放。因此，如果范围不同，子图的刻度值将不会对齐。

# 创建包含两个子图的图表 fig，并设置标题为 'Axes values are scaled individually by default'
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x, y)          # 在 ax1 中绘制 x 和 y 的折线图
ax2.plot(x + 1, -y)     # 在 ax2 中绘制 x+1 和 -y 的折线图

# %%
# 使用 *sharex* 或 *sharey* 可以对齐水平或垂直轴。

# 创建包含两个子图的图表 fig，并设置 sharex=True，以共享 x 轴
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(x, y)          # 在 ax1 中绘制 x 和 y 的折线图
ax2.plot(x + 1, -y)     # 在 ax2 中绘制 x+1 和 -y 的折线图

# %%
# 将 *sharex* 或 *sharey* 设置为 ``True`` 可以全局共享整个网格，例如使用 ``sharey=True`` 时，
# 垂直堆叠的子图的 y 轴也具有相同的刻度。

# 创建包含三个子图的图表 fig，并设置 sharex=True 和 sharey=True 以共享 x 和 y 轴
fig, axs = plt.subplots(3, sharex=True, sharey=True)
axs[0].plot(x, y ** 2)   # 在 axs[0] 中绘制 x 和 y 的平方的折线图
axs[1].plot(x, 0.3 * y, 'o')  # 在 axs[1] 中绘制 x 和 0.3*y 的散点图
axs[2].plot(x, y, '+')   # 在 axs[2] 中绘制 x 和 y 的加号图

# %%
# 对于共享轴的子图，只需要一组刻度标签。*sharex* 和 *sharey* 会自动移除内部 Axes 的刻度标签。
# 然而，子图之间仍会保留未使用的空白空间。
#
# 若要精确控制子图的位置，可以显式创建 `.GridSpec`，并使用 `.Figure.add_gridspec`，然后调用
# 其 `~.GridSpecBase.subplots` 方法。例如，可以使用 ``add_gridspec(hspace=0)`` 减少垂直子图之间的高度间隔。
#
# `.label_outer` 是一个方便的方法，用于从不在网格边缘的子图中移除标签和刻度。

# 创建一个新的图表 fig
fig = plt.figure()
# 添加一个 3x1 的网格布局，并设置垂直方向的间距为 0
gs = fig.add_gridspec(3, hspace=0)
# 在网格布局上创建子图 axs，并共享 x 和 y 轴
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Sharing both axes')  # 设置图表的主标题
axs[0].plot(x, y ** 2)   # 在 axs[0] 中绘制 x 和 y 的平方的折线图
axs[1].plot(x, 0.3 * y, 'o')  # 在 axs[1] 中绘制 x 和 0.3*y 的散点图
axs[2].plot(x, y, '+')   # 在 axs[2] 中绘制 x 和 y 的加号图

# 隐藏所有子图的 x 轴标签和刻度标签，保留底部子图的标签
for ax in axs:
    ax.label_outer()

# %%
# 除了 ``True`` 和 ``False``，*sharex* 和 *sharey* 还接受 'row' 和 'col' 来仅在行或列之间共享值。

# 创建一个新的图表 fig
fig = plt.figure()
# 添加一个 2x2 的网格布局，并设置水平和垂直方向的间距为 0
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
# 在网格布局上创建子图，并根据 'col' 和 'row' 共享 x 和 y 轴
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Sharing x per column, y per row')  # 设置图表的主标题
ax1.plot(x, y)          # 在 ax1 中绘制 x 和 y 的折线图
ax2.plot(x, y**2, 'tab:orange')  # 在 ax2 中绘制 x 和 y^2 的橙色折线图
ax3.plot(x + 1, -y, 'tab:green')  # 在 ax3 中绘制 x+1 和 -y 的绿色折线图
ax4.plot(x + 2, -y**2, 'tab:red')  # 在 ax4 中绘制 x+2 和 -y^2 的红色折线图

# 隐藏所有子图的轴标签和刻度标签
for ax in fig.get_axes():
    ax.label_outer()

# %%
# 如果需要更复杂的共享结构，可以首先创建不共享的 Axes 网格，然后后期使用 `.axes.Axes.sharex` 或 `.axes.Axes.sharey`
# 来添加共享信息。

# 创建一个包含 2x2 的子图的图表 fig
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)            # 在 axs[0, 0] 中绘制 x 和 y 的折线图
axs[0, 0].set_title("main")     # 设置 axs[0, 0] 的标题为 "main"
axs[1, 0].plot(x, y**2)         # 在 axs[1, 0] 中绘制 x 和 y^2 的折线图
axs[1, 0].set_title("shares x with main")  # 设置 axs[1, 0] 的标题为 "shares x with main"
axs[1, 0].sharex(axs[0, 0])     # 将 axs[1, 0] 的 x 轴与 axs[0, 0] 共享
axs[0, 1].plot(x + 1, y + 1)    # 在 axs[0, 1] 中绘制 x+1 和 y+1 的折线图
axs[0, 1].set_title("unrelated")    # 设置 axs[0, 1] 的标题为 "unrelated"
axs[1, 1].plot(x + 2, y + 2)    # 在 axs[1, 1] 中绘制 x+2 和 y+2 的折线图
axs[1, 1].set_title("also unrelated")  # 设置 axs[1, 1] 的标题为 "also unrelated"
fig.tight_layout()              # 调整子图布局，使其紧凑

# %%
# 极坐标轴
# """"""""""
#
# 使用 `plt.subplots` 函数创建一个包含两个子图的图形对象 `fig` 和子图对象 `ax1`、`ax2`。
# 这里的参数 `(1, 2)` 指定了子图的排列方式，表示一行两列。
# `subplot_kw=dict(projection='polar')` 用于指定子图的投影类型为极坐标。
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))

# 在第一个子图 `ax1` 上绘制一个折线图，数据为 `x` 和 `y`。
ax1.plot(x, y)

# 在第二个子图 `ax2` 上绘制一个折线图，数据为 `x` 和 `y` 的平方。
ax2.plot(x, y ** 2)

# 显示图形，展示创建的子图和它们的内容。
plt.show()
```