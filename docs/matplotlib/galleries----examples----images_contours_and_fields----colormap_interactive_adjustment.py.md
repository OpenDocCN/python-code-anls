# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\colormap_interactive_adjustment.py`

```py
"""
========================================
Interactive Adjustment of Colormap Range
========================================

Demonstration of how a colorbar can be used to interactively adjust the
range of colormapping on an image. To use the interactive feature, you must
be in either zoom mode (magnifying glass toolbar button) or
pan mode (4-way arrow toolbar button) and click inside the colorbar.

When zooming, the bounding box of the zoom region defines the new vmin and
vmax of the norm. Zooming using the right mouse button will expand the
vmin and vmax proportionally to the selected region, in the same manner that
one can zoom out on an axis. When panning, the vmin and vmax of the norm are
both shifted according to the direction of movement. The
Home/Back/Forward buttons can also be used to get back to a previous state.

.. redirect-from:: /gallery/userdemo/colormap_interactive_adjustment
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

# 生成数据
t = np.linspace(0, 2 * np.pi, 1024)  # 在0到2π之间生成1024个等间距的点
data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]  # 生成二维数据，sin(t) * cos(t)

# 创建图形和轴
fig, ax = plt.subplots()  # 创建一个新的图形和一个轴
im = ax.imshow(data2d)  # 在轴上绘制二维数据的图像
ax.set_title('Pan on the colorbar to shift the color mapping\n'
             'Zoom on the colorbar to scale the color mapping')  # 设置图像的标题

# 添加交互式颜色条
fig.colorbar(im, ax=ax, label='Interactive colorbar')  # 在图形上添加一个交互式颜色条，关联到ax轴

# 显示图形
plt.show()  # 显示整个图形，包括颜色条和图像
```