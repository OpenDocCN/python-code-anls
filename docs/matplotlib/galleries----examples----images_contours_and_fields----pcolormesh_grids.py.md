# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\pcolormesh_grids.py`

```
"""
============================
pcolormesh grids and shading
============================

`.axes.Axes.pcolormesh` and `~.axes.Axes.pcolor` have a few options for
how grids are laid out and the shading between the grid points.

Generally, if *Z* has shape *(M, N)* then the grid *X* and *Y* can be
specified with either shape *(M+1, N+1)* or *(M, N)*, depending on the
argument for the ``shading`` keyword argument.  Note that below we specify
vectors *x* as either length N or N+1 and *y* as length M or M+1, and
`~.axes.Axes.pcolormesh` internally makes the mesh matrices *X* and *Y* from
the input vectors.

"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# Flat Shading
# ------------
#
# The grid specification with the least assumptions is ``shading='flat'``
# and if the grid is one larger than the data in each dimension, i.e. has shape
# *(M+1, N+1)*.  In that case *X* and *Y* specify the corners of quadrilaterals
# that are colored with the values in *Z*. Here we specify the edges of the
# *(3, 5)* quadrilaterals with *X* and *Y* that are *(4, 6)*.

nrows = 3
ncols = 5
Z = np.arange(nrows * ncols).reshape(nrows, ncols)
x = np.arange(ncols + 1)
y = np.arange(nrows + 1)

fig, ax = plt.subplots()
# 使用 `pcolormesh` 函数创建一个平面阴影图，指定 X 和 Y 网格边界，数据为 Z，采用 'flat' 阴影模式，
# vmin 和 vmax 分别设置颜色映射的最小和最大值。
ax.pcolormesh(x, y, Z, shading='flat', vmin=Z.min(), vmax=Z.max())


def _annotate(ax, x, y, title):
    # this all gets repeated below:
    # 使用 np.meshgrid 函数从 x 和 y 向量创建网格矩阵 X 和 Y
    X, Y = np.meshgrid(x, y)
    # 在图上绘制所有点的散点图，用 'o' 表示，颜色为 'm' (magenta)
    ax.plot(X.flat, Y.flat, 'o', color='m')
    # 设置 x 和 y 轴的显示范围
    ax.set_xlim(-0.7, 5.2)
    ax.set_ylim(-0.7, 3.2)
    # 设置图的标题
    ax.set_title(title)

# 在创建的图上添加 'flat' 阴影模式的注释
_annotate(ax, x, y, "shading='flat'")


# %%
# Flat Shading, same shape grid
# -----------------------------
#
# Often, however, data is provided where *X* and *Y* match the shape of *Z*.
# While this makes sense for other ``shading`` types, it is not permitted
# when ``shading='flat'``. Historically, Matplotlib silently dropped the last
# row and column of *Z* in this case, to match Matlab's behavior. If this
# behavior is still desired, simply drop the last row and column manually:

x = np.arange(ncols)  # note *not* ncols + 1 as before
y = np.arange(nrows)
fig, ax = plt.subplots()
# 创建一个平面阴影图，X 和 Y 网格与 Z 的形状相同，使用 'flat' 阴影模式，
# 手动去除 Z 的最后一行和一列以匹配 'flat' 模式的行为
ax.pcolormesh(x, y, Z[:-1, :-1], shading='flat', vmin=Z.min(), vmax=Z.max())
# 在图上添加 'flat' 阴影模式的注释
_annotate(ax, x, y, "shading='flat': X, Y, C same shape")

# %%
# Nearest Shading, same shape grid
# --------------------------------
#
# Usually, dropping a row and column of data is not what the user means when
# they make *X*, *Y* and *Z* all the same shape.  For this case, Matplotlib
# allows ``shading='nearest'`` and centers the colored quadrilaterals on the
# grid points.
#
# If a grid that is not the correct shape is passed with ``shading='nearest'``
# an error is raised.

fig, ax = plt.subplots()
# 创建一个平面阴影图，X 和 Y 网格与 Z 的形状相同，使用 'nearest' 阴影模式
ax.pcolormesh(x, y, Z, shading='nearest', vmin=Z.min(), vmax=Z.max())
# 在图上添加 'nearest' 阴影模式的注释
_annotate(ax, x, y, "shading='nearest'")

# %%
# Auto Shading
# ------------
#
# It's possible that the user would like the code to automatically choose which
# 创建一个包含两个子图的图形对象，布局为约束布局
fig, axs = plt.subplots(2, 1, layout='constrained')

# 获取第一个子图对象
ax = axs[0]

# 创建一维数组 x 和 y，长度分别为 ncols 和 nrows
x = np.arange(ncols)
y = np.arange(nrows)

# 在第一个子图上绘制伪彩色图，使用 'auto' 参数选择 'flat' 或 'nearest' 着色方式，
# vmin 和 vmax 分别设置为 Z 数组的最小值和最大值
ax.pcolormesh(x, y, Z, shading='auto', vmin=Z.min(), vmax=Z.max())

# 调用 _annotate 函数，在第一个子图上添加注释，说明使用了 'shading='auto''，并且 X、Y、Z 的形状相同（使用最近邻插值）
_annotate(ax, x, y, "shading='auto'; X, Y, Z: same shape (nearest)")

# 获取第二个子图对象
ax = axs[1]

# 创建一维数组 x 和 y，长度分别为 ncols+1 和 nrows+1
x = np.arange(ncols + 1)
y = np.arange(nrows + 1)

# 在第二个子图上绘制伪彩色图，使用 'auto' 参数选择 'flat' 或 'nearest' 着色方式，
# vmin 和 vmax 分别设置为 Z 数组的最小值和最大值
ax.pcolormesh(x, y, Z, shading='auto', vmin=Z.min(), vmax=Z.max())

# 调用 _annotate 函数，在第二个子图上添加注释，说明使用了 'shading='auto''，并且 X 比 Y 和 Z 的形状多一个单位（使用平面着色）
_annotate(ax, x, y, "shading='auto'; X, Y one larger than Z (flat)")

# %%
# Gouraud Shading
# ---------------
#
# `Gouraud shading <https://en.wikipedia.org/wiki/Gouraud_shading>`_ can also
# be specified, where the color in the quadrilaterals is linearly interpolated
# between the grid points.  The shapes of *X*, *Y*, *Z* must be the same.

# 创建一个包含一个子图的图形对象，布局为约束布局
fig, ax = plt.subplots(layout='constrained')

# 创建一维数组 x 和 y，长度分别为 ncols 和 nrows
x = np.arange(ncols)
y = np.arange(nrows)

# 在子图上绘制伪彩色图，使用 'gouraud' 参数选择 Gouraud 着色方式，
# vmin 和 vmax 分别设置为 Z 数组的最小值和最大值
ax.pcolormesh(x, y, Z, shading='gouraud', vmin=Z.min(), vmax=Z.max())

# 调用 _annotate 函数，在子图上添加注释，说明使用了 'shading='gouraud''，并且 X、Y 和 Z 的形状相同
_annotate(ax, x, y, "shading='gouraud'; X, Y same shape as Z")

# 显示图形
plt.show()
# %%

# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
```