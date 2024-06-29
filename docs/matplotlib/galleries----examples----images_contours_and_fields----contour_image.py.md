# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\contour_image.py`

```
"""
=============
Contour Image
=============

Test combinations of contouring, filled contouring, and image plotting.
For contour labelling, see also the :doc:`contour demo example
</gallery/images_contours_and_fields/contour_demo>`.

The emphasis in this demo is on showing how to make contours register
correctly on images, and on how to get both of them oriented as desired.
In particular, note the usage of the :ref:`"origin" and "extent"
<imshow_extent>` keyword arguments to imshow and
contour.
"""
# 导入需要的库
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

# 设置默认的步长（delta），较大以提高性能并展示图像和等高线的正确对齐
delta = 0.5

# 定义图像的范围
extent = (-3, 4, -4, 3)

# 生成 X 和 Y 的网格
x = np.arange(-3.0, 4.001, delta)
y = np.arange(-4.0, 3.001, delta)
X, Y = np.meshgrid(x, y)

# 生成两个高斯分布的图像数据
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# 设置等高线的级别
levels = np.arange(-2.0, 1.601, 0.4)

# 创建归一化对象和颜色映射
norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cmap = cm.PRGn

# 创建包含四个子图的图像对象
fig, _axs = plt.subplots(nrows=2, ncols=2)
fig.subplots_adjust(hspace=0.3)
axs = _axs.flatten()

# 在第一个子图中绘制填充的等高线图
cset1 = axs[0].contourf(X, Y, Z, levels, norm=norm,
                        cmap=cmap.resampled(len(levels) - 1))

# 添加等高线线条到第一个子图，使用与填充图相同的级别
cset2 = axs[0].contour(X, Y, Z, cset1.levels, colors='k')

# 设置实线样式以替代默认的虚线样式
cset2.set_linestyle('solid')

# 绘制一条厚度为2的绿色等值线作为零值等高线
cset3 = axs[0].contour(X, Y, Z, (0,), colors='g', linewidths=2)
axs[0].set_title('Filled contours')

# 添加第一个子图的颜色条
fig.colorbar(cset1, ax=axs[0])

# 在第二个子图中绘制图像，使用'upper'作为原点
axs[1].imshow(Z, extent=extent, cmap=cmap, norm=norm)
axs[1].contour(Z, levels, colors='k', origin='upper', extent=extent)
axs[1].set_title("Image, origin 'upper'")

# 在第三个子图中绘制图像，使用'lower'作为原点
axs[2].imshow(Z, origin='lower', extent=extent, cmap=cmap, norm=norm)
axs[2].contour(Z, levels, colors='k', origin='lower', extent=extent)
axs[2].set_title("Image, origin 'lower'")

# 在最后一个子图中使用'nearest'插值显示实际图像像素
# 注意，等高线不会延伸到边框的边缘，这是故意的，因为 Z 值是在每个图像像素的中心定义的
# 在第四个子图(axs[3])上显示二维数组Z的热图，使用最近邻插值(interpolation='nearest')，
# 并设置其坐标范围为extent，使用指定的颜色映射cmap和归一化对象norm。
im = axs[3].imshow(Z, interpolation='nearest', extent=extent,
                   cmap=cmap, norm=norm)

# 在第四个子图(axs[3])上绘制二维数组Z的等高线，指定等高线的高度levels，
# 颜色为黑色('k')，坐标原点为'image'，并设置其坐标范围为extent。
axs[3].contour(Z, levels, colors='k', origin='image', extent=extent)

# 获取当前第四个子图(axs[3])的y轴坐标范围
ylim = axs[3].get_ylim()

# 将第四个子图(axs[3])的y轴坐标范围反转
axs[3].set_ylim(ylim[::-1])

# 设置第四个子图(axs[3])的标题为"Origin from rc, reversed y-axis"
axs[3].set_title("Origin from rc, reversed y-axis")

# 在整个图形(fig)上添加颜色条，关联到第四个子图(axs[3])上的imshow对象im
fig.colorbar(im, ax=axs[3])

# 调整整个图形(fig)的布局，以确保所有子图适当排列
fig.tight_layout()

# 显示图形
plt.show()



# %%
#
# .. admonition:: References
#
#    This example demonstrates the usage of the following functions, methods, classes, and modules:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.Normalize`
```