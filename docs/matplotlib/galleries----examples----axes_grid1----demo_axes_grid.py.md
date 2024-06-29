# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_axes_grid.py`

```py
"""
==============
Demo Axes Grid
==============

Grid of 2x2 images with a single colorbar or with one colorbar per Axes.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

from matplotlib import cbook  # 导入 matplotlib 的 cbook 模块
from mpl_toolkits.axes_grid1 import ImageGrid  # 导入 axes_grid1 中的 ImageGrid 类

fig = plt.figure(figsize=(10.5, 2.5))  # 创建一个大小为 10.5x2.5 的新图形对象
Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")  # 获取示例数据 "axes_grid/bivariate_normal.npy"，返回一个 15x15 的数组
extent = (-3, 4, -4, 3)  # 定义数据的显示范围


# A grid of 2x2 images with 0.05 inch pad between images and only the
# lower-left Axes is labeled.
grid = ImageGrid(
    fig, 141,  # 在 fig 上创建一个子图位置，类似于 fig.add_subplot(141)。
    nrows_ncols=(2, 2), axes_pad=0.05, label_mode="1")  # 创建一个2x2的图像网格，图像之间的间距为0.05英寸，只有左下角的 Axes 带有标签
for ax in grid:
    ax.imshow(Z, extent=extent)  # 在每个 Axes 上显示 Z 数组的内容
# This only affects Axes in first column and second row as share_all=False.
grid.axes_llc.set(xticks=[-2, 0, 2], yticks=[-2, 0, 2])  # 设置左下角的 Axes 的 x 和 y 轴刻度


# A grid of 2x2 images with a single colorbar.
grid = ImageGrid(
    fig, 142,  # 在 fig 上创建一个子图位置，类似于 fig.add_subplot(142)。
    nrows_ncols=(2, 2), axes_pad=0.0, label_mode="L", share_all=True,
    cbar_location="top", cbar_mode="single")  # 创建一个2x2的图像网格，所有图像共用一个 colorbar，colorbar 位于顶部
for ax in grid:
    im = ax.imshow(Z, extent=extent)  # 在每个 Axes 上显示 Z 数组的内容
grid.cbar_axes[0].colorbar(im)  # 在第一个 colorbar Axes 上添加 colorbar
for cax in grid.cbar_axes:
    cax.tick_params(labeltop=False)  # 设置 colorbar 的顶部标签不可见
# This affects all Axes as share_all = True.
grid.axes_llc.set(xticks=[-2, 0, 2], yticks=[-2, 0, 2])  # 设置左下角的 Axes 的 x 和 y 轴刻度


# A grid of 2x2 images. Each image has its own colorbar.
grid = ImageGrid(
    fig, 143,  # 在 fig 上创建一个子图位置，类似于 fig.add_subplot(143)。
    nrows_ncols=(2, 2), axes_pad=0.1, label_mode="1", share_all=True,
    cbar_location="top", cbar_mode="each", cbar_size="7%", cbar_pad="2%")  # 创建一个2x2的图像网格，每个图像都有自己的 colorbar，colorbar 位于顶部
for ax, cax in zip(grid, grid.cbar_axes):
    im = ax.imshow(Z, extent=extent)  # 在每个 Axes 上显示 Z 数组的内容
    cax.colorbar(im)  # 在当前 colorbar Axes 上添加 colorbar
    cax.tick_params(labeltop=False)  # 设置 colorbar 的顶部标签不可见
# This affects all Axes as share_all = True.
grid.axes_llc.set(xticks=[-2, 0, 2], yticks=[-2, 0, 2])  # 设置左下角的 Axes 的 x 和 y 轴刻度


# A grid of 2x2 images. Each image has its own colorbar.
grid = ImageGrid(
    fig, 144,  # 在 fig 上创建一个子图位置，类似于 fig.add_subplot(144)。
    nrows_ncols=(2, 2), axes_pad=(0.45, 0.15), label_mode="1", share_all=True,
    cbar_location="right", cbar_mode="each", cbar_size="7%", cbar_pad="2%")  # 创建一个2x2的图像网格，每个图像都有自己的 colorbar，colorbar 位于右侧
# Use a different colorbar range every time
limits = ((0, 1), (-2, 2), (-1.7, 1.4), (-1.5, 1))
for ax, cax, vlim in zip(grid, grid.cbar_axes, limits):
    im = ax.imshow(Z, extent=extent, vmin=vlim[0], vmax=vlim[1])  # 在每个 Axes 上显示 Z 数组的内容，并设置不同的颜色范围
    cb = cax.colorbar(im)  # 在当前 colorbar Axes 上添加 colorbar
    cb.set_ticks((vlim[0], vlim[1]))  # 设置 colorbar 的刻度位置
# This affects all Axes as share_all = True.
grid.axes_llc.set(xticks=[-2, 0, 2], yticks=[-2, 0, 2])  # 设置左下角的 Axes 的 x 和 y 轴刻度

plt.show()  # 显示图形
```