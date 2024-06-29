# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\pcolor_demo.py`

```
"""
=============
pcolor images
=============

`~.Axes.pcolor` generates 2D image-style plots, as illustrated below.
"""

# 导入 matplotlib.pyplot 和 numpy 库
import matplotlib.pyplot as plt
import numpy as np

# 从 matplotlib.colors 模块导入 LogNorm 类
from matplotlib.colors import LogNorm

# 设置随机数种子以保证可复现性
np.random.seed(19680801)

# %%
# A simple pcolor demo
# --------------------

# 生成一个 6x10 的随机数组
Z = np.random.rand(6, 10)

# 创建一个包含两个子图的图形对象
fig, (ax0, ax1) = plt.subplots(2, 1)

# 在第一个子图中绘制 pcolor 图，不显示边缘线
c = ax0.pcolor(Z)
ax0.set_title('default: no edges')

# 在第二个子图中绘制 pcolor 图，设置边缘线的颜色为黑色，线宽为4
c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
ax1.set_title('thick edges')

# 调整子图之间的布局
fig.tight_layout()
plt.show()

# %%
# Comparing pcolor with similar functions
# ---------------------------------------

# Demonstrates similarities between `~.axes.Axes.pcolor`,
# `~.axes.Axes.pcolormesh`, `~.axes.Axes.imshow` and
# `~.axes.Axes.pcolorfast` for drawing quadrilateral grids.
# Note that we call ``imshow`` with ``aspect="auto"`` so that it doesn't force
# the data pixels to be square (the default is ``aspect="equal"``).

# 减小步长以增加分辨率
dx, dy = 0.15, 0.05

# 生成用于 x 和 y 边界的网格
y, x = np.mgrid[-3:3+dy:dy, -3:3+dx:dx]
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)

# x 和 y 定义边界，因此 z 应该是这些边界内的值，因此删除 z 数组的最后一个值
z = z[:-1, :-1]
z_min, z_max = -abs(z).max(), abs(z).max()

# 创建包含四个子图的图形对象
fig, axs = plt.subplots(2, 2)

# 在第一个子图中绘制 pcolor 图，使用 'RdBu' 颜色映射，指定最小和最大值，并添加颜色条
ax = axs[0, 0]
c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolor')
fig.colorbar(c, ax=ax)

# 在第二个子图中绘制 pcolormesh 图，使用 'RdBu' 颜色映射，指定最小和最大值，并添加颜色条
ax = axs[0, 1]
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
fig.colorbar(c, ax=ax)

# 在第三个子图中绘制 imshow 图，使用 'RdBu' 颜色映射，指定最小和最大值，并添加颜色条，
# 设置插值方式为 'nearest'，原点为 'lower'，长宽比为 'auto'
ax = axs[1, 0]
c = ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
              extent=[x.min(), x.max(), y.min(), y.max()],
              interpolation='nearest', origin='lower', aspect='auto')
ax.set_title('image (nearest, aspect="auto")')
fig.colorbar(c, ax=ax)

# 在第四个子图中绘制 pcolorfast 图，使用 'RdBu' 颜色映射，指定最小和最大值，并添加颜色条
ax = axs[1, 1]
c = ax.pcolorfast(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolorfast')
fig.colorbar(c, ax=ax)

# 调整子图之间的布局
fig.tight_layout()
plt.show()

# %%
# Pcolor with a log scale
# -----------------------

# The following shows pcolor plots with a log scale.

N = 100
X, Y = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-2, 2, N))

# A low hump with a spike coming out.
# Needs to have z/colour axis on a log scale, so we see both hump and spike.
# A linear scale only shows the spike.
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
Z = Z1 + 50 * Z2

# 创建包含两个子图的图形对象
fig, (ax0, ax1) = plt.subplots(2, 1)

# 在第一个子图中绘制 pcolor 图，使用 'PuBu_r' 颜色映射，并添加颜色条，使用对数标准化
c = ax0.pcolor(X, Y, Z, shading='auto',
               norm=LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='PuBu_r')
fig.colorbar(c, ax=ax0)

# 在第二个子图中绘制 pcolor 图，使用 'PuBu_r' 颜色映射，并添加颜色条
c = ax1.pcolor(X, Y, Z, cmap='PuBu_r', shading='auto')
fig.colorbar(c, ax=ax1)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.pcolor` / `matplotlib.pyplot.pcolor`
# 创建一个二维的颜色网格，使用指定的网格值来填充每个单元格
def pcolormesh(self, X, Y, C, **kwargs):
    pass

# 快速绘制一个二维颜色网格，使用指定的颜色值来填充每个单元格
def pcolorfast(self, *args, **kwargs):
    pass

# 显示图像数据，可以是二维数组，也可以是三维的RGB数据
def imshow(self, X, **kwargs):
    pass

# 为图形添加一个颜色条，关联到给定的图像或颜色网格对象
def colorbar(self, mappable=None, cax=None, ax=None, **kw):
    pass

# 提供一个对数尺度的归一化对象，用于在绘图时进行对数尺度的数据转换
class LogNorm(Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False):
        pass
```