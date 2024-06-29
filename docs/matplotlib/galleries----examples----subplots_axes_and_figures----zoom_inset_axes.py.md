# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\zoom_inset_axes.py`

```
"""
======================
Zoom region inset Axes
======================

Example of an inset Axes and a rectangle showing where the zoom is located.
"""

import numpy as np  # 导入 NumPy 库

from matplotlib import cbook  # 导入 matplotlib 的 cbook 模块
from matplotlib import pyplot as plt  # 导入 matplotlib 的 pyplot 模块

fig, ax = plt.subplots()  # 创建一个新的 Figure 和 Axes 对象

# make data
Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")  # 获取示例数据文件的路径并加载数据，得到一个 15x15 的数组
Z2 = np.zeros((150, 150))  # 创建一个全零的 150x150 数组
ny, nx = Z.shape  # 获取数组 Z 的形状信息
Z2[30:30+ny, 30:30+nx] = Z  # 将数组 Z 的内容复制到 Z2 的特定位置
extent = (-3, 4, -4, 3)  # 设置数据的显示范围

ax.imshow(Z2, extent=extent, origin="lower")  # 在 Axes 上显示 Z2 数据，设置数据范围和原点位置

# inset Axes....
x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # 原始图像的子区域的坐标范围
axins = ax.inset_axes(
    [0.5, 0.5, 0.47, 0.47],  # 在主图上创建插入的子图，设置位置和大小
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])  # 设置子图的坐标轴范围和标签
axins.imshow(Z2, extent=extent, origin="lower")  # 在子图上显示 Z2 数据，设置数据范围和原点位置

ax.indicate_inset_zoom(axins, edgecolor="black")  # 在主图上指示插入子图的缩放区域

plt.show()  # 显示绘制的图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.inset_axes`
#    - `matplotlib.axes.Axes.indicate_inset_zoom`
#    - `matplotlib.axes.Axes.imshow`
```