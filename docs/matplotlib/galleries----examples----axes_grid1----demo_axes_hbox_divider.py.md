# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_axes_hbox_divider.py`

```py
"""
================================
HBoxDivider and VBoxDivider demo
================================

Using an `.HBoxDivider` to arrange subplots.

Note that both Axes' location are adjusted so that they have
equal heights while maintaining their aspect ratios.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块
import numpy as np  # 导入 numpy 模块

from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider, VBoxDivider  # 从 axes_grid1.axes_divider 中导入 HBoxDivider 和 VBoxDivider
import mpl_toolkits.axes_grid1.axes_size as Size  # 导入 axes_grid1.axes_size 模块

arr1 = np.arange(20).reshape((4, 5))  # 创建一个 4x5 的数组 arr1
arr2 = np.arange(20).reshape((5, 4))  # 创建一个 5x4 的数组 arr2

fig, (ax1, ax2) = plt.subplots(1, 2)  # 创建一个包含两个子图的图形对象 fig，ax1 和 ax2 分别表示两个子图
ax1.imshow(arr1)  # 在 ax1 子图上显示 arr1 数组的图像
ax2.imshow(arr2)  # 在 ax2 子图上显示 arr2 数组的图像

pad = 0.5  # 设置间距为 0.5 英寸

# 创建一个水平方向的 HBoxDivider 对象 divider，用于控制两个子图的排列
divider = HBoxDivider(
    fig, 111,
    horizontal=[Size.AxesX(ax1), Size.Fixed(pad), Size.AxesX(ax2)],
    vertical=[Size.AxesY(ax1), Size.Scaled(1), Size.AxesY(ax2)])
ax1.set_axes_locator(divider.new_locator(0))  # 设置 ax1 子图的位置定位器
ax2.set_axes_locator(divider.new_locator(2))  # 设置 ax2 子图的位置定位器

plt.show()  # 显示图形

# %%
# Using a `.VBoxDivider` to arrange subplots.
#
# Note that both Axes' location are adjusted so that they have
# equal widths while maintaining their aspect ratios.

fig, (ax1, ax2) = plt.subplots(2, 1)  # 创建一个包含两个垂直排列的子图的图形对象 fig，ax1 和 ax2 分别表示两个子图
ax1.imshow(arr1)  # 在 ax1 子图上显示 arr1 数组的图像
ax2.imshow(arr2)  # 在 ax2 子图上显示 arr2 数组的图像

# 创建一个垂直方向的 VBoxDivider 对象 divider，用于控制两个子图的排列
divider = VBoxDivider(
    fig, 111,
    horizontal=[Size.AxesX(ax1), Size.Scaled(1), Size.AxesX(ax2)],
    vertical=[Size.AxesY(ax1), Size.Fixed(pad), Size.AxesY(ax2)])

ax1.set_axes_locator(divider.new_locator(0))  # 设置 ax1 子图的位置定位器
ax2.set_axes_locator(divider.new_locator(2))  # 设置 ax2 子图的位置定位器

plt.show()  # 显示图形
```