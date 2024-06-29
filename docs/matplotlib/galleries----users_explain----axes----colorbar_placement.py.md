# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\colorbar_placement.py`

```
"""
.. _colorbar_placement:

.. redirect-from:: /gallery/subplots_axes_and_figures/colorbar_placement

=================
Placing colorbars
=================

Colorbars indicate the quantitative extent of image data.  Placing in
a figure is non-trivial because room needs to be made for them.

Automatic placement of colorbars
================================

The simplest case is just attaching a colorbar to each Axes.  Note in this
example that the colorbars steal some space from the parent Axes.
"""
# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# Fixing random state for reproducibility
# 设定随机种子以保证可重复性
np.random.seed(19680801)

# 创建一个 2x2 的子图
fig, axs = plt.subplots(2, 2)
# 定义两种 colormap
cmaps = ['RdBu_r', 'viridis']
# 遍历每个子图的列和行
for col in range(2):
    for row in range(2):
        # 获取当前子图的 Axes 对象
        ax = axs[row, col]
        # 生成一个随机的 20x20 的数组，并根据当前列数调整其范围，应用不同的 colormap
        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
                            cmap=cmaps[col])
        # 在当前 Axes 上添加颜色条
        fig.colorbar(pcm, ax=ax)

# %%
# 第一列的数据类型相同，因此可能希望只有一个颜色条。可以通过向 `.Figure.colorbar` 传递包含 *ax* 关键字参数的 Axes 列表来实现。

# 创建一个新的 2x2 子图
fig, axs = plt.subplots(2, 2)
# 定义两种 colormap
cmaps = ['RdBu_r', 'viridis']
# 遍历每个子图的列和行
for col in range(2):
    for row in range(2):
        # 获取当前子图的 Axes 对象
        ax = axs[row, col]
        # 生成一个随机的 20x20 的数组，并根据当前列数调整其范围，应用不同的 colormap
        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
                            cmap=cmaps[col])
# 在整列的 Axes 上添加一个颜色条，缩小其尺寸
fig.colorbar(pcm, ax=axs[:, col], shrink=0.6)

# %%
# 颜色条占用的空间会导致同一子图布局中的 Axes 大小不同，这通常是不希望的，特别是当各个图的 x 轴应该是可比较的时候：

# 创建一个新的包含两个子图的 2x1 布局，指定其大小为 (4, 5)，共享 x 轴
fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True)
# 生成一个随机的 20x20 的数组 X
X = np.random.randn(20, 20)
# 在第一个子图上绘制 X 按列求和的结果
axs[0].plot(np.sum(X, axis=0))
# 在第二个子图上绘制 X 的伪彩色图
pcm = axs[1].pcolormesh(X)
# 在第二个子图上添加颜色条，并缩小其尺寸
fig.colorbar(pcm, ax=axs[1], shrink=0.6)

# %%
# 这通常是不希望的，并且可以通过多种方法解决，例如在其他 Axes 上添加颜色条然后将其删除。然而，最直接的方法是使用 :ref:`constrained layout <constrainedlayout_guide>`：

# 创建一个新的包含两个子图的 2x1 布局，指定其大小为 (4, 5)，共享 x 轴，使用 constrained layout
fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True, constrained_layout=True)
# 在第一个子图上绘制 X 按列求和的结果
axs[0].plot(np.sum(X, axis=0))
# 在第二个子图上绘制 X 的伪彩色图
pcm = axs[1].pcolormesh(X)
# 在第二个子图上添加颜色条，并缩小其尺寸
fig.colorbar(pcm, ax=axs[1], shrink=0.6)

# %%
# 使用这种方法可以实现相对复杂的颜色条布局。请注意，这个例子在 ``layout='constrained'`` 下效果更好。

# 创建一个包含九个子图的 3x3 布局，使用 constrained layout
fig, axs = plt.subplots(3, 3, constrained_layout=True)
# 遍历每个子图，并在其上绘制一个随机的 20x20 的数组的伪彩色图
for ax in axs.flat:
    pcm = ax.pcolormesh(np.random.random((20, 20)))

# 在第一行的前两列上添加一个在底部位置的颜色条，并缩小其尺寸
fig.colorbar(pcm, ax=axs[0, :2], shrink=0.6, location='bottom')
# 在第一行的第三列上添加一个在底部位置的颜色条
fig.colorbar(pcm, ax=[axs[0, 2]], location='bottom')
# 在第二行的所有列上添加一个在右侧位置的颜色条，并缩小其尺寸
fig.colorbar(pcm, ax=axs[1:, :], location='right', shrink=0.6)
# 在第三行的第二列上添加一个在左侧位置的颜色条
fig.colorbar(pcm, ax=[axs[2, 1]], location='left')

# %%
# 调整颜色条与父 Axes 之间的间距
# =======================================================
#
# 可以使用 *pad* 关键字参数调整颜色条与父 Axes 之间的距离。这个距离单位是父 Axes 的比例。
# 创建一个包含3个子图的图形对象，每个子图垂直排列，使用布局管理器'constrained'，设置图形大小为5x5英寸
fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(5, 5))

# 对于每个子图ax和对应的pad值，在每个子图上绘制一个20x20的随机数据的伪彩图，并生成颜色条，设置pad参数，颜色条标题为pad值
for ax, pad in zip(axs, [0.025, 0.05, 0.1]):
    pcm = ax.pcolormesh(np.random.randn(20, 20), cmap='viridis')
    fig.colorbar(pcm, ax=ax, pad=pad, label=f'pad: {pad}')

# 设置整个图形的标题
fig.suptitle("layout='constrained'")

# %%
# 如果不使用constrained layout，pad命令会使父级Axes缩小：
fig, axs = plt.subplots(3, 1, figsize=(5, 5))

# 对于每个子图ax和对应的pad值，在每个子图上绘制一个20x20的随机数据的伪彩图，并生成颜色条，设置pad参数，颜色条标题为pad值
for ax, pad in zip(axs, [0.025, 0.05, 0.1]):
    pcm = ax.pcolormesh(np.random.randn(20, 20), cmap='viridis')
    fig.colorbar(pcm, ax=ax, pad=pad, label=f'pad: {pad}')

# 设置整个图形的标题
fig.suptitle("No layout manager")

# %%
# 手动放置颜色条
# =============================
#
# 有时由``colorbar``自动放置的位置效果不佳。我们可以手动创建一个Axes，并通过将该Axes传递给*cax*关键字参数告诉``colorbar``使用它。
#
# 使用``inset_axes``
# --------------------
#
# 我们可以手动创建任何类型的Axes供颜色条使用，但`.Axes.inset_axes`很有用，因为它是父Axes的子级，并且可以相对于父级定位。在这里，我们添加一个位于父级Axes底部附近的居中颜色条。

fig, ax = plt.subplots(layout='constrained', figsize=(4, 4))
pcm = ax.pcolormesh(np.random.randn(20, 20), cmap='viridis')
ax.set_ylim([-4, 20])

# 在父级Axes内创建一个插入的Axes，设置其位置和大小
cax = ax.inset_axes([0.3, 0.07, 0.4, 0.04])
fig.colorbar(pcm, cax=cax, orientation='horizontal')

# %%
# `.Axes.inset_axes`还可以使用*transform*关键字参数指定其在数据坐标系中的位置，如果希望Axes位于图形上的某个特定数据位置：

fig, ax = plt.subplots(layout='constrained', figsize=(4, 4))
pcm = ax.pcolormesh(np.random.randn(20, 20), cmap='viridis')
ax.set_ylim([-4, 20])

# 在父级Axes内创建一个插入的Axes，设置其在数据坐标系中的位置和大小
cax = ax.inset_axes([7.5, -1.7, 5, 1.2], transform=ax.transData)
fig.colorbar(pcm, cax=cax, orientation='horizontal')

# %%
# 为具有固定纵横比的Axes放置颜色条
# ---------------------------------------------
#
# 为具有固定纵横比的Axes放置颜色条是一项特殊挑战，因为父Axes的大小会根据数据视图而变化。

fig, axs = plt.subplots(2, 2, layout='constrained')
cmaps = ['RdBu_r', 'viridis']

# 遍历2x2的子图网格
for col in range(2):
    for row in range(2):
        ax = axs[row, col]
        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1), cmap=cmaps[col])
        
        # 如果是第一列的子图，设置纵横比为2，否则为1/2
        if col == 0:
            ax.set_aspect(2)
        else:
            ax.set_aspect(1/2)
        
        # 如果是第二行的子图，为其添加一个缩小为0.6的颜色条
        if row == 1:
            fig.colorbar(pcm, ax=ax, shrink=0.6)

# %%
# 我们使用`.Axes.inset_axes`解决此问题，将Axes定位在“Axes坐标”中（参见：ref:`transforms_tutorial`）。注意，如果放大父级Axes，从而改变其形状，颜色条也会改变位置。
fig, axs = plt.subplots(2, 2, layout='constrained')
# 创建一个包含两行两列子图的 Figure 对象，并返回子图数组 axs

cmaps = ['RdBu_r', 'viridis']
# 定义颜色映射列表 cmaps 包含两种颜色映射

for col in range(2):
    # 对于每一列（col = 0, 1）
    for row in range(2):
        # 对于每一行（row = 0, 1）
        ax = axs[row, col]
        # 获取当前行列位置对应的子图对象 ax

        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
                            cmap=cmaps[col])
        # 在当前子图上绘制一个伪彩色网格，使用随机生成的数据数组，并根据 col 使用对应的颜色映射

        if col == 0:
            ax.set_aspect(2)
            # 如果当前列是第一列，设置子图的纵横比为 2:1
        else:
            ax.set_aspect(1/2)
            # 如果当前列不是第一列，设置子图的纵横比为 1:2

        if row == 1:
            cax = ax.inset_axes([1.04, 0.2, 0.05, 0.6])
            # 如果当前行是第二行，创建一个嵌入式的轴 cax 在当前子图上，
            # 位置和尺寸分别为 [1.04, 0.2, 0.05, 0.6]
            fig.colorbar(pcm, cax=cax)
            # 在嵌入轴 cax 上创建一个颜色条，使用之前创建的 pcm 对象

# %%
# 可以参考以下链接查看有关手动创建颜色条轴的更多信息：
#
#  :ref:`axes_grid` 提供了手动创建颜色条轴的方法：
#
#  - :ref:`demo-colorbar-with-inset-locator`
#  - :ref:`demo-colorbar-with-axes-divider`
```