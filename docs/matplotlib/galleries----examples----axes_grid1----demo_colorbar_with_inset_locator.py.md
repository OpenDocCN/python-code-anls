# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_colorbar_with_inset_locator.py`

```
"""
.. _demo-colorbar-with-inset-locator:

==============================================================
Controlling the position and size of colorbars with Inset Axes
==============================================================

This example shows how to control the position, height, and width of colorbars
using `~mpl_toolkits.axes_grid1.inset_locator.inset_axes`.

Inset Axes placement is controlled as for legends: either by providing a *loc*
option ("upper right", "best", ...), or by providing a locator with respect to
the parent bbox.  Parameters such as *bbox_to_anchor* and *borderpad* likewise
work in the same way, and are also demonstrated here.

Users should consider using `.Axes.inset_axes` instead (see
:ref:`colorbar_placement`).

.. redirect-from:: /gallery/axes_grid1/demo_colorbar_of_inset_axes
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 从 mpl_toolkits.axes_grid1.inset_locator 导入 inset_axes 函数
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 创建一个包含两个子图的 Figure 对象，并指定子图的大小为 [6, 3]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6, 3])

# 在 ax1 上绘制图像，并存储返回的图像对象
im1 = ax1.imshow([[1, 2], [2, 3]])

# 创建一个插入坐标轴 axins1，位于 ax1 的右上角，宽度为父坐标框宽度的 50%，高度为 5%
axins1 = inset_axes(
    ax1,
    width="50%",  # 宽度：父坐标框宽度的 50%
    height="5%",  # 高度：5%
    loc="upper right",  # 插入位置：右上角
)

# 设置插入坐标轴 axins1 的 x 轴刻度位置为底部
axins1.xaxis.set_ticks_position("bottom")

# 在 axins1 上创建水平方向的颜色条，与图像 im1 关联，设置刻度为 [1, 2, 3]
fig.colorbar(im1, cax=axins1, orientation="horizontal", ticks=[1, 2, 3])

# 在 ax2 上绘制图像，并存储返回的图像对象
im = ax2.imshow([[1, 2], [2, 3]])

# 创建一个插入坐标轴 axins，位于 ax2 的左下角，宽度为父坐标框宽度的 5%，高度为 50%
# bbox_to_anchor 设置为 (1.05, 0., 1, 1)，使用 ax2 的坐标系进行转换，borderpad 设置为 0
axins = inset_axes(
    ax2,
    width="5%",  # 宽度：父坐标框宽度的 5%
    height="50%",  # 高度：50%
    loc="lower left",  # 插入位置：左下角
    bbox_to_anchor=(1.05, 0., 1, 1),  # 相对于父坐标框的位置和大小
    bbox_transform=ax2.transAxes,  # 使用 ax2 的坐标系进行转换
    borderpad=0,  # 边距设置为 0
)

# 在 axins 上创建颜色条，与图像 im 关联，设置刻度为 [1, 2, 3]
fig.colorbar(im, cax=axins, ticks=[1, 2, 3])

# 显示绘图结果
plt.show()
```