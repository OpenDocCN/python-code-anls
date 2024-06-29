# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\subfigures.py`

```py
"""
=================
Figure subfigures
=================

Sometimes it is desirable to have a figure with two different layouts in it.
This can be achieved with
:doc:`nested gridspecs</gallery/subplots_axes_and_figures/gridspec_nested>`,
but having a virtual figure with its own artists is helpful, so
Matplotlib also has "subfigures", accessed by calling
`matplotlib.figure.Figure.add_subfigure` in a way that is analogous to
`matplotlib.figure.Figure.add_subplot`, or
`matplotlib.figure.Figure.subfigures` to make an array of subfigures.  Note
that subfigures can also have their own child subfigures.

.. note::
    The *subfigure* concept is new in v3.4, and the API is still provisional.

"""
import matplotlib.pyplot as plt
import numpy as np


def example_plot(ax, fontsize=12, hide_labels=False):
    # 创建一个色彩网格图在给定的Axes对象上，用随机数据填充，设定值范围为-2.5到2.5
    pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
    if not hide_labels:
        # 如果不隐藏标签，设置X轴标签为'x-label'，字体大小为给定的fontsize
        ax.set_xlabel('x-label', fontsize=fontsize)
        # 设置Y轴标签为'y-label'，字体大小为给定的fontsize
        ax.set_ylabel('y-label', fontsize=fontsize)
        # 设置标题为'Title'，字体大小为给定的fontsize
        ax.set_title('Title', fontsize=fontsize)
    return pc

np.random.seed(19680808)

# 创建一个约束布局（constrained layout），大小为(10, 4)的图形对象
fig = plt.figure(layout='constrained', figsize=(10, 4))
# 在图形对象中创建1行2列的子图对象数组，水平间距为0.07
subfigs = fig.subfigures(1, 2, wspace=0.07)

# 在第一个子图中创建1行2列的子图对象数组，共享Y轴
axsLeft = subfigs[0].subplots(1, 2, sharey=True)
# 设置第一个子图的背景色为浅灰色
subfigs[0].set_facecolor('0.75')
# 在每个左侧子图对象上调用example_plot函数，绘制随机色彩网格图
for ax in axsLeft:
    pc = example_plot(ax)
# 设置第一个子图的总标题为'Left plots'，字体大小为'x-large'
subfigs[0].suptitle('Left plots', fontsize='x-large')
# 在左侧子图上添加颜色条，位置在底部，缩小系数为0.6
subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

# 在第二个子图中创建3行1列的子图对象数组，共享X轴
axsRight = subfigs[1].subplots(3, 1, sharex=True)
# 在每个右侧子图对象上调用example_plot函数，隐藏标签
for nn, ax in enumerate(axsRight):
    pc = example_plot(ax, hide_labels=True)
    # 根据索引号设置X轴和Y轴标签
    if nn == 2:
        ax.set_xlabel('xlabel')
    if nn == 1:
        ax.set_ylabel('ylabel')

# 设置第二个子图的背景色为浅灰色
subfigs[1].set_facecolor('0.85')
# 在右侧子图上添加颜色条，缩小系数为0.6
subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)
# 设置第二个子图的总标题为'Right plots'，字体大小为'x-large'
subfigs[1].suptitle('Right plots', fontsize='x-large')

# 设置整个图形对象的总标题为'Figure suptitle'，字体大小为'xx-large'
fig.suptitle('Figure suptitle', fontsize='xx-large')

# 显示图形
plt.show()

# %%
# It is possible to mix subplots and subfigures using
# `matplotlib.figure.Figure.add_subfigure`.  This requires getting
# the gridspec that the subplots are laid out on.

# 创建一个2行3列的子图对象数组，约束布局，大小为(10, 4)
fig, axs = plt.subplots(2, 3, layout='constrained', figsize=(10, 4))
# 获取第一个子图对象的子图规范对象，再获取其所在的总体网格规范对象
gridspec = axs[0, 0].get_subplotspec().get_gridspec()

# 清除第一列的子图对象
for a in axs[:, 0]:
    a.remove()

# 在剩余的Axes对象中绘制数据
for a in axs[:, 1:].flat:
    a.plot(np.arange(10))

# 在空的网格规范位置添加一个子图对象
subfig = fig.add_subfigure(gridspec[:, 0])

# 在添加的子图对象中创建1行2列的子图对象数组，共享Y轴
axsLeft = subfig.subplots(1, 2, sharey=True)
# 设置子图对象的背景色为浅灰色
subfig.set_facecolor('0.75')
# 在左侧子图对象上调用example_plot函数，绘制随机色彩网格图
for ax in axsLeft:
    pc = example_plot(ax)
# 设置子图对象的总标题为'Left plots'，字体大小为'x-large'
subfig.suptitle('Left plots', fontsize='x-large')
# 在左侧子图上添加颜色条，位置在底部，缩小系数为0.6
subfig.colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

# 设置整个图形对象的总标题为'Figure suptitle'，字体大小为'xx-large'
fig.suptitle('Figure suptitle', fontsize='xx-large')
# 显示图形
plt.show()

# %%
# Subfigures can have different widths and heights.  This is exactly the
# same example as the first example, but *width_ratios* has been changed:

# 创建一个约束布局，大小为(10, 4)的图形对象
fig = plt.figure(layout='constrained', figsize=(10, 4))
# 创建包含 1 行 2 列子图的 subfigs 对象，设置子图之间的水平间距为 0.07，左右两列子图的宽度比例分别为 2:1
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[2, 1])

# 在 subfigs[0] 中创建包含 1 行 2 列的子图 axsLeft，这两个子图共享 y 轴
axsLeft = subfigs[0].subplots(1, 2, sharey=True)

# 设置 subfigs[0] 的背景颜色为浅灰色（RGB 值为 '0.75'）
subfigs[0].set_facecolor('0.75')

# 对 axsLeft 中的每个子图调用 example_plot 函数，并将返回的图像对象保存到 pc 变量中
for ax in axsLeft:
    pc = example_plot(ax)

# 设置 subfigs[0] 的总标题为 'Left plots'，字体大小为 'x-large'
subfigs[0].suptitle('Left plots', fontsize='x-large')

# 在 subfigs[0] 中为 axsLeft 中的图像添加颜色条，颜色条的尺寸缩小至原来的 60%，位置在底部
subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

# 在 subfigs[1] 中创建包含 3 行 1 列的子图 axsRight，这三个子图共享 x 轴
axsRight = subfigs[1].subplots(3, 1, sharex=True)

# 对 axsRight 中的每个子图调用 example_plot 函数，并将返回的图像对象保存到 pc 变量中，同时隐藏标签
for nn, ax in enumerate(axsRight):
    pc = example_plot(ax, hide_labels=True)
    if nn == 2:
        ax.set_xlabel('xlabel')  # 如果是第三个子图，设置 x 轴标签为 'xlabel'
    if nn == 1:
        ax.set_ylabel('ylabel')  # 如果是第二个子图，设置 y 轴标签为 'ylabel'

# 设置 subfigs[1] 的背景颜色为浅灰色（RGB 值为 '0.85'）
subfigs[1].set_facecolor('0.85')

# 在 subfigs[1] 中为 axsRight 中的图像添加颜色条，颜色条的尺寸缩小至原来的 60%
subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)

# 设置 subfigs[1] 的总标题为 'Right plots'，字体大小为 'x-large'
subfigs[1].suptitle('Right plots', fontsize='x-large')

# 设置整个图形 fig 的总标题为 'Figure suptitle'，字体大小为 'xx-large'
fig.suptitle('Figure suptitle', fontsize='xx-large')

# 显示绘制的图形
plt.show()
```