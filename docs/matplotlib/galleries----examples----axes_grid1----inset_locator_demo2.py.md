# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\inset_locator_demo2.py`

```py
"""
====================
Inset locator demo 2
====================

This demo shows how to create a zoomed inset via `.zoomed_inset_axes`.
In the first subplot an `.AnchoredSizeBar` shows the zoom effect.
In the second subplot a connection to the region of interest is
created via `.mark_inset`.

A version of the second subplot, not using the toolkit, is available in
:doc:`/gallery/subplots_axes_and_figures/zoom_inset_axes`.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入numpy数值计算库

from matplotlib import cbook  # 导入matplotlib的cbook模块
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # 导入AnchoredSizeBar类
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes  # 导入mark_inset和zoomed_inset_axes函数

# 创建包含两个子图的图像对象
fig, (ax, ax2) = plt.subplots(ncols=2, figsize=[6, 3])

# 第一个子图，显示带有大小标尺的插图
ax.set_aspect(1)  # 设置子图的纵横比为1

# 创建一个放大的插图区域，位于右上角
axins = zoomed_inset_axes(ax, zoom=0.5, loc='upper right')
# 设置插图Axes上的主要刻度数量
axins.yaxis.get_major_locator().set_params(nbins=7)
axins.xaxis.get_major_locator().set_params(nbins=7)
axins.tick_params(labelleft=False, labelbottom=False)

# 定义添加大小标尺的函数
def add_sizebar(ax, size):
    asb = AnchoredSizeBar(ax.transData,
                          size,
                          str(size),
                          loc=8,
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)

add_sizebar(ax, 0.5)  # 在主图上添加大小标尺
add_sizebar(axins, 0.5)  # 在插图上添加大小标尺


# 第二个子图，显示带有插图缩放和标记插图的图像
Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")  # 获取示例数据
extent = (-3, 4, -4, 3)
Z2 = np.zeros((150, 150))
ny, nx = Z.shape
Z2[30:30+ny, 30:30+nx] = Z

ax2.imshow(Z2, extent=extent, origin="lower")  # 在第二个子图上显示图像

axins2 = zoomed_inset_axes(ax2, zoom=6, loc=1)  # 创建第二个子图的插图区域
axins2.imshow(Z2, extent=extent, origin="lower")  # 在插图上显示图像

# 设置插图区域的x和y轴限制
x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
# 设置插图Axes上的主要刻度数量
axins2.yaxis.get_major_locator().set_params(nbins=7)
axins2.xaxis.get_major_locator().set_params(nbins=7)
axins2.tick_params(labelleft=False, labelbottom=False)

# 在父图上绘制插图Axes区域的bbox和连接线
mark_inset(ax2, axins2, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()  # 显示图形
```