# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\constrainedlayout_guide.py`

```py
# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库中的 pyplot 模块
import numpy as np  # 导入 numpy 库，并使用 np 别名

import matplotlib.colors as mcolors  # 导入 matplotlib 库中的 colors 模块
import matplotlib.gridspec as gridspec  # 导入 matplotlib 库中的 gridspec 模块

# 设置全局绘图参数
plt.rcParams['savefig.facecolor'] = "0.8"  # 设置保存图像时的背景色
plt.rcParams['figure.figsize'] = 4.5, 4.  # 设置图像的默认尺寸为 4.5x4 英寸
plt.rcParams['figure.max_open_warning'] = 50  # 设置打开的图像文件数量的最大警告阈值为 50

# 定义一个绘图函数 example_plot，用于在给定的 Axes 对象上绘制一条线
def example_plot(ax, fontsize=12, hide_labels=False):
    ax.plot([1, 2])  # 在指定的 Axes 对象上绘制直线

    ax.locator_params(nbins=3)  # 设置坐标轴的定位器参数，nbins 表示刻度的数量
    if hide_labels:
        ax.set_xticklabels([])  # 如果 hide_labels 为 True，则隐藏 x 轴刻度标签
        ax.set_yticklabels([])  # 如果 hide_labels 为 True，则隐藏 y 轴刻度标签
    else:
        ax.set_xlabel('x-label', fontsize=fontsize)  # 设置 x 轴标签及其字体大小
        ax.set_ylabel('y-label', fontsize=fontsize)  # 设置 y 轴标签及其字体大小
        ax.set_title('Title', fontsize=fontsize)  # 设置图表标题及其字体大小

# 创建一个新的图像和 Axes 对象，使用 constrained layout 自动调整子图的位置
fig, ax = plt.subplots(layout=None)
example_plot(ax, fontsize=24)  # 调用 example_plot 函数，在当前 Axes 对象上绘制图形

# %%
# 要避免坐标轴标签超出图像区域被裁剪，需要调整 Axes 的位置。
# 对于子图，可以手动调整子图参数，使用 .Figure.subplots_adjust 方法。
# 然而，通过指定 ``layout="constrained"`` 关键字参数创建图像会自动进行调整。

# 创建一个新的图像和 Axes 对象，使用 constrained layout 自动调整子图的位置
fig, ax = plt.subplots(layout="constrained")
example_plot(ax, fontsize=24)  # 调用 example_plot 函数，在当前 Axes 对象上绘制图形

# %%
# 当存在多个子图时，通常会看到不同 Axes 的标签彼此重叠。
# 创建一个包含 2x2 子图的图形对象 `fig` 和子图对象数组 `axs`
fig, axs = plt.subplots(2, 2, layout=None)
# 遍历所有子图对象，并在每个子图上调用 `example_plot` 函数
for ax in axs.flat:
    example_plot(ax)

# %%
# 在调用 `plt.subplots` 函数时，通过指定 `layout="constrained"` 参数，
# 可以确保子图布局受到限制，使得布局更为合理。

# 创建一个包含 2x2 子图的图形对象 `fig` 和子图对象数组 `axs`，
# 并通过 `layout="constrained"` 参数确保布局受到约束
fig, axs = plt.subplots(2, 2, layout="constrained")
# 遍历所有子图对象，并在每个子图上调用 `example_plot` 函数
for ax in axs.flat:
    example_plot(ax)

# %%
#
# Colorbars
# =========
#
# 如果使用 `.Figure.colorbar` 创建一个颜色条，需要为其腾出空间。
# *Constrained layout* 会自动处理这一点。注意，如果指定了 `use_gridspec=True`，
# 这个选项会被忽略，因为它是为了通过 `tight_layout` 改善布局而设立的。
#
# .. note::
#
#   对于 `~.axes.Axes.pcolormesh` 的关键字参数（`pc_kwargs`），我们使用一个
#   字典来保持整个文档中的调用一致性。

# 创建一个 10x10 的数组 `arr`
arr = np.arange(100).reshape((10, 10))
# 创建一个归一化对象 `norm`，范围从 0 到 100
norm = mcolors.Normalize(vmin=0., vmax=100.)
# 根据前面的注释，使用一个包含归一化等参数的字典 `pc_kwargs`，
# 保持所有 `pcolormesh` 调用的一致性
pc_kwargs = {'rasterized': True, 'cmap': 'viridis', 'norm': norm}
# 创建一个 4x4 大小的图形对象 `fig` 和一个子图对象 `ax`，
# 并通过 `layout="constrained"` 确保布局受到约束
fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
# 在子图 `ax` 上绘制 `arr` 的彩色网格，使用 `pc_kwargs` 中的参数
im = ax.pcolormesh(arr, **pc_kwargs)
# 在图形对象 `fig` 上创建一个颜色条，并将其放置在子图 `ax` 的一侧，缩小其大小为原来的 60%
fig.colorbar(im, ax=ax, shrink=0.6)

# %%
# 如果将一个包含 Axes 对象的列表（或其他可迭代容器）指定给 `ax` 参数，
# *constrained layout* 将会从指定的 Axes 对象中腾出空间。

# 创建一个包含 2x2 子图的图形对象 `fig` 和子图对象数组 `axs`，
# 并通过 `layout="constrained"` 确保布局受到约束
fig, axs = plt.subplots(2, 2, figsize=(4, 4), layout="constrained")
# 遍历所有子图对象，并在每个子图上绘制 `arr` 的彩色网格，使用 `pc_kwargs` 中的参数
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
# 在图形对象 `fig` 上创建一个颜色条，并将其放置在子图 `axs` 的一侧，缩小其大小为原来的 60%
fig.colorbar(im, ax=axs, shrink=0.6)

# %%
# 如果在 Axes 网格内部指定了一个 Axes 对象列表，颜色条会适当地占用空间，并留下间隙，
# 但所有子图仍然保持相同的大小。

# 创建一个包含 3x3 子图的图形对象 `fig` 和子图对象数组 `axs`，
# 并通过 `layout="constrained"` 确保布局受到约束
fig, axs = plt.subplots(3, 3, figsize=(4, 4), layout="constrained")
# 遍历所有子图对象，并在每个子图上绘制 `arr` 的彩色网格，使用 `pc_kwargs` 中的参数
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
# 在图形对象 `fig` 上创建一个颜色条，并将其放置在子图 `axs[1:, 1]` 的一侧，缩小其大小为原来的 80%
fig.colorbar(im, ax=axs[1:, 1], shrink=0.8)
# 在图形对象 `fig` 上创建一个颜色条，并将其放置在子图 `axs[:, -1]` 的一侧，缩小其大小为原来的 60%
fig.colorbar(im, ax=axs[:, -1], shrink=0.6)

# %%
# Suptitle
# =========
#
# *Constrained layout* 还可以为 `~.Figure.suptitle` 腾出空间。

# 创建一个包含 2x2 子图的图形对象 `fig` 和子图对象数组 `axs`，
# 并通过 `layout="constrained"` 确保布局受到约束
fig, axs = plt.subplots(2, 2, figsize=(4, 4), layout="constrained")
# 遍历所有子图对象，并在每个子图上绘制 `arr` 的彩色网格，使用 `pc_kwargs` 中的参数
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
# 在图形对象 `fig` 上创建一个颜色条，并将其放置在子图 `axs` 的一侧，缩小其大小为原来的 60%
fig.colorbar(im, ax=axs, shrink=0.6)
# 在图形对象 `fig` 上添加一个大标题 `Big Suptitle`
fig.suptitle('Big Suptitle')

# %%
# Legends
# =======
#
# 图例可以放置在其父坐标轴之外。
# *Constrained layout* 设计用于处理 :meth:`.Axes.legend` 的这种情况。
# 但是，*constrained layout* 尚不能处理通过 :meth:`.Figure.legend` 创建的图例。

# 创建一个图形对象 `fig`，并通过 `layout="constrained"` 确保布局受到约束
fig, ax = plt.subplots(layout="constrained")
# 在坐标轴 `ax` 上绘制一个包含 10 个点的折线图，并添加图例 `This is a plot`，
# 将图例放置在左侧中间，指定位置的偏移量为 (0.8, 0.5)
ax.plot(np.arange(10), label='This is a plot')
ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))

# %%
# 但是，这将从子图布局中占用空间：

# 创建一个包含 1x2 子图的图形对象 `fig` 和子图对象数组 `axs`，
# 并通过 `layout="constrained"` 确保布局受到约束
fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout="constrained")
# 在子图 `axs[0]` 上绘制一个包含 10 个点的折线图
axs[0].plot(np.arange(10))
# 在子图 `axs[1]` 上绘制一个包含 10 个点的折线图，并添加图例 `This is a plot`，
# 将图例放置在左侧中间，指定位置的偏移量为 (0.8, 0.5)
axs[1].plot(np.arange(10), label='This is a plot')
axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))

# %%
# 为了使图例或其他艺术元素不从子图布局中占用空间，
# 我们可以使用 `leg.set_in_layout(False)`。
# 创建包含两个子图的图形对象，设置图形大小为 (4, 2)，并使用约束布局
fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout="constrained")

# 在第一个子图 axs[0] 上绘制折线图，横轴为从 0 到 9 的整数序列
axs[0].plot(np.arange(10))

# 在第二个子图 axs[1] 上绘制折线图，同时为折线添加图例 'This is a plot'
axs[1].plot(np.arange(10), label='This is a plot')

# 在第二个子图 axs[1] 上添加图例，位置设定为居中左侧，边界框锚点在 (0.8, 0.5)
leg = axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))

# 设置图例在布局中的状态为不包括（False），这样在保存文件时不会影响布局
leg.set_in_layout(False)

# 手动触发一次绘制，以便在打印之前执行约束布局调整子图尺寸
fig.canvas.draw()

# 设置图例在布局中的状态为包括（True），以便在保存文件时将图例包括在内
leg.set_in_layout(True)

# 设置图形的布局引擎为 'none'，即在此处不希望改变布局
fig.set_layout_engine('none')

# 尝试保存图形为文件，设置边界框紧密包围图形内容，分辨率为 100 dpi
try:
    fig.savefig('../../../doc/_static/constrained_layout_1b.png',
                bbox_inches='tight', dpi=100)
except FileNotFoundError:
    # 如果文件路径不存在，则捕获 FileNotFoundError 异常，脚本可以继续运行
    pass


# 创建包含两个子图的图形对象，设置图形大小为 (4, 2)，并使用约束布局
fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout="constrained")

# 在第一个子图 axs[0] 上绘制折线图，横轴为从 0 到 9 的整数序列
axs[0].plot(np.arange(10))

# 在第二个子图 axs[1] 上绘制折线图，并为折线添加图例 'This is a plot'
lines = axs[1].plot(np.arange(10), label='This is a plot')

# 获取第二个子图 axs[1] 的所有图例标签
labels = [l.get_label() for l in lines]

# 在整个图形上添加图例，使用先前获取的线条和标签，位置设定为居中左侧，边界框锚点在 (0.8, 0.5)，转换坐标系为 axs[1] 的相对坐标系
leg = fig.legend(lines, labels, loc='center left',
                 bbox_to_anchor=(0.8, 0.5), bbox_transform=axs[1].transAxes)

# 尝试保存图形为文件，设置边界框紧密包围图形内容，分辨率为 100 dpi
try:
    fig.savefig('../../../doc/_static/constrained_layout_2b.png',
                bbox_inches='tight', dpi=100)
except FileNotFoundError:
    # 如果文件路径不存在，则捕获 FileNotFoundError 异常，脚本可以继续运行
    pass


# 创建包含四个子图的图形对象，并使用约束布局
fig, axs = plt.subplots(2, 2, layout="constrained")

# 对每个子图应用一个示例绘图函数，隐藏标签
for ax in axs.flat:
    example_plot(ax, hide_labels=True)

# 获取图形的布局引擎，并设置水平和垂直间距，水平间距为 4/72 英寸，垂直间距为 4/72 英寸，水平间隔和垂直间隔为 0
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0,
                            wspace=0)
# from the above, but the space between subplots does.

fig, axs = plt.subplots(2, 2, layout="constrained")
# 创建一个包含 2x2 子图的 Figure 对象，使用约束布局
for ax in axs.flat:
    # 针对每个子图，调用 example_plot 函数并隐藏标签
    example_plot(ax, hide_labels=True)
# 设置布局引擎的子图之间的水平和垂直间距，以及子图组之间的水平和垂直间距
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                            wspace=0.2)

# %%

# If there are more than two columns, the *wspace* is shared between them,
# so here the wspace is divided in two, with a *wspace* of 0.1 between each
# column:

fig, axs = plt.subplots(2, 3, layout="constrained")
# 创建一个包含 2x3 子图的 Figure 对象，使用约束布局
for ax in axs.flat:
    # 针对每个子图，调用 example_plot 函数并隐藏标签
    example_plot(ax, hide_labels=True)
# 设置布局引擎的子图之间的水平和垂直间距，以及子图组之间的水平和垂直间距
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
                            wspace=0.2)

# %%

# GridSpecs also have optional *hspace* and *wspace* keyword arguments,
# that will be used instead of the pads set by *constrained layout*:

fig, axs = plt.subplots(2, 2, layout="constrained",
                        gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
# 创建一个包含 2x2 子图的 Figure 对象，使用约束布局和自定义的网格参数
for ax in axs.flat:
    # 针对每个子图，调用 example_plot 函数并隐藏标签
    example_plot(ax, hide_labels=True)
# 这里设置的间距设置不生效，因为网格参数中设置的空间值覆盖了约束布局设置的值
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.0,
                            wspace=0.0)

# %%

# Spacing with colorbars
# -----------------------
#
# Colorbars are placed a distance *pad* from their parent, where *pad*
# is a fraction of the width of the parent(s).  The spacing to the
# next subplot is then given by *w/hspace*.

fig, axs = plt.subplots(2, 2, layout="constrained")
# 创建一个包含 2x2 子图的 Figure 对象，使用约束布局
pads = [0, 0.05, 0.1, 0.2]
for pad, ax in zip(pads, axs.flat):
    # 在每个子图上绘制颜色网格，并添加颜色条，设置 shrink 和 pad 参数
    pc = ax.pcolormesh(arr, **pc_kwargs)
    fig.colorbar(pc, ax=ax, shrink=0.6, pad=pad)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'pad: {pad}')
# 设置布局引擎的子图之间的水平和垂直间距，以及子图组之间的水平和垂直间距
fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2,
                            wspace=0.2)

# %%

# rcParams
# ========
#
# There are five :ref:`rcParams<customizing-with-dynamic-rc-settings>`
# that can be set, either in a script or in the :file:`matplotlibrc`
# file. They all have the prefix ``figure.constrained_layout``:
#
# - *use*: Whether to use *constrained layout*. Default is False
# - *w_pad*, *h_pad*:    Padding around Axes objects.
#   Float representing inches.  Default is 3./72. inches (3 pts)
# - *wspace*, *hspace*:  Space between subplot groups.
#   Float representing a fraction of the subplot widths being separated.
#   Default is 0.02.

plt.rcParams['figure.constrained_layout.use'] = True
# 设置全局参数，启用约束布局
fig, axs = plt.subplots(2, 2, figsize=(3, 3))
for ax in axs.flat:
    # 针对每个子图，调用 example_plot 函数
    example_plot(ax)

# %%

# Use with GridSpec
# =================
#
# *Constrained layout* is meant to be used
# with :func:`~matplotlib.figure.Figure.subplots`,
# :func:`~matplotlib.figure.Figure.subplot_mosaic`, or
# :func:`~matplotlib.gridspec.GridSpec` with
# :func:`~matplotlib.figure.Figure.add_subplot`.
#
# Note that in what follows ``layout="constrained"``

plt.rcParams['figure.constrained_layout.use'] = False
# 设置全局参数，禁用约束布局
fig = plt.figure(layout="constrained")
# 使用 gridspec 创建一个包含两行一列的网格布局，将其应用于给定的图形对象 fig
gs1 = gridspec.GridSpec(2, 1, figure=fig)

# 在图形 fig 上添加一个子图 ax1，位于 gs1 的第一行
ax1 = fig.add_subplot(gs1[0])

# 在图形 fig 上添加一个子图 ax2，位于 gs1 的第二行
ax2 = fig.add_subplot(gs1[1])

# 在 ax1 上绘制示例图
example_plot(ax1)

# 在 ax2 上绘制示例图
example_plot(ax2)

# %%
# 更复杂的 gridspec 布局也是可能的。这里我们使用 `~.Figure.add_gridspec` 和
# `~.SubplotSpec.subgridspec` 方便函数来实现。

# 创建一个具有约束布局的新图形对象 fig
fig = plt.figure(layout="constrained")

# 添加一个 1x2 的网格布局 gs0 到 fig
gs0 = fig.add_gridspec(1, 2)

# 在 gs0 的第一列上创建一个包含两行一列的子网格布局 gs1
gs1 = gs0[0].subgridspec(2, 1)

# 在 gs1 的第一行上添加一个子图 ax1 到 fig
ax1 = fig.add_subplot(gs1[0])

# 在 gs1 的第二行上添加一个子图 ax2 到 fig
ax2 = fig.add_subplot(gs1[1])

# 在 ax1 上绘制示例图
example_plot(ax1)

# 在 ax2 上绘制示例图
example_plot(ax2)

# 在 gs0 的第二列上创建一个包含三行一列的子网格布局 gs2
gs2 = gs0[1].subgridspec(3, 1)

# 遍历 gs2 中的每一个子网格 ss
for ss in gs2:
    # 在当前子网格 ss 上创建一个子图 ax
    ax = fig.add_subplot(ss)

    # 在 ax 上绘制示例图
    example_plot(ax)

    # 设置子图 ax 的标题为空字符串
    ax.set_title("")

    # 设置子图 ax 的 x 轴标签为空字符串
    ax.set_xlabel("")

# 在最后一个子图 ax 上设置 x 轴标签为 "x-label"，字体大小为 12
ax.set_xlabel("x-label", fontsize=12)

# %%
# 注意，在上面的布局中，左列和右列的垂直扩展不同。
# 如果我们希望两个网格的顶部和底部对齐，则它们需要位于同一个 gridspec 中。
# 同时，为了避免轴折叠到零高度，我们还需要增大图形的尺寸：

# 创建一个具有特定尺寸（4x6）和约束布局的新图形对象 fig
fig = plt.figure(figsize=(4, 6), layout="constrained")

# 添加一个 6x2 的网格布局 gs0 到 fig
gs0 = fig.add_gridspec(6, 2)

# 在 gs0 的第一列上创建一个包含前三行的子网格布局 ax1
ax1 = fig.add_subplot(gs0[:3, 0])

# 在 gs0 的第二列上创建一个包含后三行的子网格布局 ax2
ax2 = fig.add_subplot(gs0[3:, 0])

# 在 ax1 上绘制示例图
example_plot(ax1)

# 在 ax2 上绘制示例图
example_plot(ax2)

# 在 gs0 的第一行第二列上创建一个包含前两行的子网格布局 ax
ax = fig.add_subplot(gs0[0:2, 1])

# 在 ax 上绘制示例图，同时隐藏标签
example_plot(ax, hide_labels=True)

# 在 gs0 的第三行第二列上创建一个包含中间两行的子网格布局 ax
ax = fig.add_subplot(gs0[2:4, 1])

# 在 ax 上绘制示例图，同时隐藏标签
example_plot(ax, hide_labels=True)

# 在 gs0 的第五行第二列上创建一个包含后两行的子网格布局 ax
ax = fig.add_subplot(gs0[4:, 1])

# 在 ax 上绘制示例图，同时隐藏标签
example_plot(ax, hide_labels=True)

# 在图形 fig 上设置主标题为 'Overlapping Gridspecs'
fig.suptitle('Overlapping Gridspecs')

# %%
# 这个示例使用两个 gridspecs，使得颜色条仅适用于一组 pcolors。
# 注意左列由于此原因比右侧两列宽。当然，如果希望子图大小相同，只需一个 gridspec 即可。
# 同样的效果也可以通过 `~.Figure.subfigures` 实现。

# 创建一个具有约束布局的新图形对象 fig
fig = plt.figure(layout="constrained")

# 添加一个 1x2 的网格布局 gs0 到 fig，同时指定左列宽度比为 [1, 2]
gs0 = fig.add_gridspec(1, 2, figure=fig, width_ratios=[1, 2])

# 在 gs0 的第一列上创建一个包含两行一列的子网格布局 gs_left
gs_left = gs0[0].subgridspec(2, 1)

# 遍历 gs_left 中的每一个子网格 gs
for gs in gs_left:
    # 在当前子网格 gs 上创建一个子图 ax
    ax = fig.add_subplot(gs)

    # 在 ax 上绘制示例图
    example_plot(ax)

# 创建一个空列表 axs，用于存储 gs_right 中的子图
axs = []

# 在 gs0 的第二列上创建一个包含两行两列的子网格布局 gs_right
for gs in gs_right:
    # 在当前子网格 gs 上创建一个子图 ax
    ax = fig.add_subplot(gs)

    # 使用 pcolormesh 方法在 ax 上绘制颜色图，并获取返回的 QuadMesh 对象 pcm
    pcm = ax.pcolormesh(arr, **pc_kwargs)

    # 设置子图 ax 的 x 轴标签为 'x-label'
    ax.set_xlabel('x-label')

    # 设置子图 ax 的 y 轴标签为 'y-label'
    ax.set_ylabel('y-label')

    # 设置子图 ax 的标题为 'title'
    ax.set_title('title')

    # 将子图 ax 添加到 axs 列表中
    axs += [ax]

# 在图形 fig 上添加颜色条，关联到 axs 中的子图
fig.colorbar(pcm, ax=axs)

# 在图形 fig 上设置主标题为 'Nested plots using subgridspec'
fig.suptitle('Nested plots using subgridspec')

# %%
# 而不是使用 subgridspecs，Matplotlib 现在提供了 `~.Figure.subfigures`，
# 也可以与 *constrained layout* 一起使用：

# 创建一个具有约束布局的新图形对象 fig
fig = plt.figure(layout="constrained")

# 在 fig 上创建一个包含 1x2 的子图组 sfigs，同时指定左列宽度比为 [1, 2]
sfigs = fig.subfigures(1, 2, width_ratios=[1, 2])

# 在 sfigs[0] 上创建一个 2x1 的子图数组 axs_left
axs_left = sfigs[0].subplots(2, 1)

# 遍历 axs_left 中的每一个子图 ax
for ax in axs_left.flat:
    # 在 ax 上绘制示例图
    example_plot(ax)

# 在 sfigs[1] 上创建一个 2x2 的子图数组 axs_right
axs_right = sfigs[1].subplots(2, 2)

# 遍历 axs_right 中的每一个子图 ax
for ax in axs_right.flat:
    # 使用 pcolormesh 方法在 ax 上绘制颜色图，并获取返回的 QuadMesh 对象 pcm
    pcm = ax.pcolormesh(arr, **pc_kwargs)

    # 设置子图 ax 的 x 轴标签为 'x-label'
    ax.set_xlabel('x-label')

    # 设置子图 ax 的 y 轴标签为 'y-label'
    ax.set_ylabel('y-label')

    # 设置子图 ax 的标题为 'title'
    ax.set_title('title')

# 在图形 fig 上添加颜
# There can be good reasons to manually set an Axes position.  A manual call
# to `~.axes.Axes.set_position` will set the Axes so *constrained layout* has
# no effect on it anymore. (Note that *constrained layout* still leaves the
# space for the Axes that is moved).

fig, axs = plt.subplots(1, 2, layout="constrained")
example_plot(axs[0], fontsize=12)
axs[1].set_position([0.2, 0.2, 0.4, 0.4])



# %%
# .. _compressed_layout:
#
# Grids of fixed aspect-ratio Axes: "compressed" layout
# =====================================================
#
# *Constrained layout* operates on the grid of "original" positions for
# Axes. However, when Axes have fixed aspect ratios, one side is usually made
# shorter, and leaves large gaps in the shortened direction. In the following,
# the Axes are square, but the figure quite wide so there is a horizontal gap:

fig, axs = plt.subplots(2, 2, figsize=(5, 3),
                        sharex=True, sharey=True, layout="constrained")
for ax in axs.flat:
    ax.imshow(arr)
fig.suptitle("fixed-aspect plots, layout='constrained'")



# %%
# One obvious way of fixing this is to make the figure size more square,
# however, closing the gaps exactly requires trial and error.  For simple grids
# of Axes we can use ``layout="compressed"`` to do the job for us:

fig, axs = plt.subplots(2, 2, figsize=(5, 3),
                        sharex=True, sharey=True, layout='compressed')
for ax in axs.flat:
    ax.imshow(arr)
fig.suptitle("fixed-aspect plots, layout='compressed'")



# %%
# Manually turning off *constrained layout*
# ===========================================
#
# *Constrained layout* usually adjusts the Axes positions on each draw
# of the figure.  If you want to get the spacing provided by
# *constrained layout* but not have it update, then do the initial
# draw and then call ``fig.set_layout_engine('none')``.
# This is potentially useful for animations where the tick labels may
# change length.
#
# Note that *constrained layout* is turned off for ``ZOOM`` and ``PAN``
# GUI events for the backends that use the toolbar.  This prevents the
# Axes from changing position during zooming and panning.
#
#
# Limitations
# ===========
#
# Incompatible functions
# ----------------------
#
# *Constrained layout* will work with `.pyplot.subplot`, but only if the
# number of rows and columns is the same for each call.
# The reason is that each call to `.pyplot.subplot` will create a new
# `.GridSpec` instance if the geometry is not the same, and
# *constrained layout*.  So the following works fine:

fig = plt.figure(layout="constrained")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
# third Axes that spans both rows in second column:
ax3 = plt.subplot(2, 2, (2, 4))

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
plt.suptitle('Homogenous nrows, ncols')



# %%
# but the following leads to a poor layout:

fig = plt.figure(layout="constrained")

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
ax3 = plt.subplot(1, 2, 2)
# %%
# 类似地，
# `~matplotlib.pyplot.subplot2grid` 也存在相同的限制，
# 就是 nrows 和 ncols 不能变化，以确保布局看起来良好。

# 创建一个具有约束布局的新图形对象
fig = plt.figure(layout="constrained")

# 创建四个子图，使用 subplot2grid 方法进行定位和布局
ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

# 在每个子图上调用示例绘图函数
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)

# 设置整个图形的标题
fig.suptitle('subplot2grid')

# %%
# 其它注意事项
# -------------
#
# * *约束布局* 仅考虑刻度标签、坐标轴标签、标题和图例。因此，其它的艺术元素可能会被裁剪或重叠。
#
# * 它假设刻度标签、坐标轴标签和标题所需的额外空间与 Axes 的原始位置无关。这通常是正确的，但在极少数情况下可能不是。
#
# * 各后端在渲染字体时存在细微差异，因此结果可能不是像素完全相同的。
#
# * 如果一个艺术家使用的 Axes 坐标超出了 Axes 的边界，将其添加到 Axes 中会导致不寻常的布局。可以通过直接将艺术家添加到 :class:`~matplotlib.figure.Figure` 中来避免这种情况，例如使用 :meth:`~matplotlib.figure.Figure.add_artist` 方法。有关示例，请参阅 :class:`~matplotlib.patches.ConnectionPatch`。

# %%
# 调试
# =========
#
# *约束布局* 可能以一些意想不到的方式失败。由于它使用约束求解器，求解器可能找到数学上正确但用户不想要的解决方案。通常的失败模式是所有尺寸都收缩到其允许的最小值。如果发生这种情况，原因可能是：
#
# 1. 没有足够的空间来绘制您请求的元素。
# 2. 存在错误 - 如果是这种情况，请在 https://github.com/matplotlib/matplotlib/issues 报告问题。
#
# 如果存在错误，请使用不需要外部数据或依赖项（除了 numpy 之外）的自包含示例进行报告。

# %%
# .. _cl_notes_on_algorithm:
#
# 算法注意事项
# ======================
#
# 约束布局的算法相对直观，但由于图形布局方式的复杂性，具有一定的复杂性。
#
# 在 Matplotlib 中，布局通过 gridspecs 实现，通过 `.GridSpec` 类进行逻辑上的行和列划分，行和列的相对宽度由 *width_ratios* 和 *height_ratios* 设置。
#
# 在 *约束布局* 中，每个 gridspec 都有一个关联的 *layoutgrid*。*layoutgrid* 每列有一系列的 ``left`` 和 ``right`` 变量，每行有 ``bottom`` 和 ``top`` 变量，还有左、右、底部和顶部的边距。在每个
# 导入绘图库中的 plot_children 函数
from matplotlib._layoutgrid import plot_children

# 创建一个包含单个 Axes 的图形对象，使用约束布局
fig, ax = plt.subplots(layout="constrained")
# 在 Axes 上绘制示例图，设置字体大小为 24
example_plot(ax, fontsize=24)
# 在图形中绘制所有子元素（如 Axes）
plot_children(fig)

# %%
# 简单情况：两个 Axes
# ---------------------
# 当存在多个 Axes 时，它们的布局会以简单的方式相互绑定。在这个示例中，
# 左侧的 Axes 拥有比右侧更大的装饰，但它们共享一个底部边距，足以容纳更大的 xlabel。
# 顶部边距也是一样。左右边距不共享，因此可以有不同的大小。

fig, ax = plt.subplots(1, 2, layout="constrained")
# 在第一个 Axes 上绘制示例图，设置字体大小为 32
example_plot(ax[0], fontsize=32)
# 在第二个 Axes 上绘制示例图，设置字体大小为 8
example_plot(ax[1], fontsize=8)
# 在图形中绘制所有子元素（如 Axes）
plot_children(fig)

# %%
# 两个 Axes 和颜色条
# ---------------------
#
# 颜色条只是父 layoutgrid 单元格中另一个扩展边距的项目：

fig, ax = plt.subplots(1, 2, layout="constrained")
im = ax[0].pcolormesh(arr, **pc_kwargs)
# 在第一个 Axes 上添加颜色条，缩小为原图的 60%
fig.colorbar(im, ax=ax[0], shrink=0.6)
im = ax[1].pcolormesh(arr, **pc_kwargs)
# 在图形中绘制所有子元素（如 Axes）
plot_children(fig)

# %%
# 与 Gridspec 相关的颜色条
# -----------------------------------
#
# 如果颜色条属于网格的多个单元格，则为每个单元格增加更大的边距：

fig, axs = plt.subplots(2, 2, layout="constrained")
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
# 在多个 Axes 上添加颜色条，缩小为原图的 60%
fig.colorbar(im, ax=axs, shrink=0.6)
# 在图形中绘制所有子元素（如 Axes）
plot_children(fig)

# %%
# 不均匀大小的 Axes
# -----------------
#
# 在 Gridspec 布局中，使 Axes 变得不均匀大小有两种方法，一种是指定它们跨越 Gridspecs 的行或列，
# 另一种是指定宽度和高度比例。
#
# 这里使用了第一种方法。请注意，中间的“top”和“bottom”边距不受左侧列的影响。
# 这是算法的有意决定，导致右侧的两个 Axes 具有相同的高度，但不是左侧 Axes 高度的一半。
# 这与没有“constrained layout”时 gridspec 的工作方式一致。

fig = plt.figure(layout="constrained")
gs = gridspec.GridSpec(2, 2, figure=fig)
ax = fig.add_subplot(gs[:, 0])
# 在当前轴上绘制数组的伪彩图，并返回生成的图像对象
im = ax.pcolormesh(arr, **pc_kwargs)
# 将一个新的子图添加到图形中，放置在第一行第二列的位置
ax = fig.add_subplot(gs[0, 1])
# 在当前轴上绘制数组的伪彩图，并返回生成的图像对象
im = ax.pcolormesh(arr, **pc_kwargs)
# 将一个新的子图添加到图形中，放置在第二行第二列的位置
ax = fig.add_subplot(gs[1, 1])
# 在当前轴上绘制数组的伪彩图，并返回生成的图像对象
im = ax.pcolormesh(arr, **pc_kwargs)
# 在图形中绘制所有子对象
plot_children(fig)

# %%
# 如果边距没有任何约束其宽度的艺术家，则需要优化的一个案例。
# 在下面的情况中，列0的右边距和列3的左边距没有边距艺术家来设置它们的宽度，
# 因此我们采用具有艺术家的边距宽度的最大宽度。
# 这样可以确保所有轴具有相同的大小：

# 创建一个具有受约束布局的新图形对象
fig = plt.figure(layout="constrained")
# 添加一个2x4网格布局到图形中
gs = fig.add_gridspec(2, 4)
# 添加子图到指定网格位置：第一行的第一列到第二列
ax00 = fig.add_subplot(gs[0, 0:2])
# 添加子图到指定网格位置：第一行的第三列到第四列
ax01 = fig.add_subplot(gs[0, 2:])
# 添加子图到指定网格位置：第二行的第二列到第三列
ax10 = fig.add_subplot(gs[1, 1:3])
# 在指定子图上绘制示例图，并设置字体大小为14
example_plot(ax10, fontsize=14)
# 在图形中绘制所有子对象
plot_children(fig)
# 显示图形
plt.show()
```