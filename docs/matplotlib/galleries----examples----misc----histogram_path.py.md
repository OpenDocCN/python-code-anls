# `D:\src\scipysrc\matplotlib\galleries\examples\misc\histogram_path.py`

```
"""
========================================================
Building histograms using Rectangles and PolyCollections
========================================================

Using a path patch to draw rectangles.

The technique of using lots of `.Rectangle` instances, or the faster method of
using `.PolyCollection`, were implemented before we had proper paths with
moveto, lineto, closepoly, etc. in Matplotlib.  Now that we have them, we can
draw collections of regularly shaped objects with homogeneous properties more
efficiently with a PathCollection. This example makes a histogram -- it's more
work to set up the vertex arrays at the outset, but it should be much faster
for large numbers of objects.
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy

import matplotlib.patches as patches  # 导入绘图图形库中的 patches 模块
import matplotlib.path as path  # 导入绘图路径库中的 path 模块

np.random.seed(19680801)  # 设置随机种子，以便结果可复现

# 用 numpy 绘制数据的直方图
data = np.random.randn(1000)  # 生成随机数据
n, bins = np.histogram(data, 50)  # 计算直方图的数据

# 获取直方图矩形的角点
left = bins[:-1]  # 左边界
right = bins[1:]  # 右边界
bottom = np.zeros(len(left))  # 底边为0
top = bottom + n  # 上边界为直方图频数

# 我们需要一个 (numrects x numsides x 2) 的 numpy 数组，用于构建复合路径的辅助函数
XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# 获取 Path 对象
barpath = path.Path.make_compound_path_from_polys(XY)

# 创建一个 PathPatch 对象，不在 y=0 处添加边距
patch = patches.PathPatch(barpath)
patch.sticky_edges.y[:] = [0]

fig, ax = plt.subplots()  # 创建一个图形窗口和坐标轴
ax.add_patch(patch)  # 将 PathPatch 对象添加到坐标轴上
ax.autoscale_view()  # 自动调整坐标轴范围
plt.show()  # 显示图形

# %%
# 替代使用三维数组并使用 `~.path.Path.make_compound_path_from_polys`，
# 我们也可以直接使用顶点和代码创建复合路径，如下所示

nrects = len(left)  # 矩形数量
nverts = nrects*(1+3+1)  # 顶点数量
verts = np.zeros((nverts, 2))  # 初始化顶点数组
codes = np.ones(nverts, int) * path.Path.LINETO  # 初始化代码数组，默认为 LINETO
codes[0::5] = path.Path.MOVETO  # 每5个点设置一个 MOVETO
codes[4::5] = path.Path.CLOSEPOLY  # 每5个点设置一个 CLOSEPOLY
verts[0::5, 0] = left  # 设置左下角顶点的 x 坐标
verts[0::5, 1] = bottom  # 设置左下角顶点的 y 坐标
verts[1::5, 0] = left  # 设置左上角顶点的 x 坐标
verts[1::5, 1] = top  # 设置左上角顶点的 y 坐标
verts[2::5, 0] = right  # 设置右上角顶点的 x 坐标
verts[2::5, 1] = top  # 设置右上角顶点的 y 坐标
verts[3::5, 0] = right  # 设置右下角顶点的 x 坐标
verts[3::5, 1] = bottom  # 设置右下角顶点的 y 坐标

barpath = path.Path(verts, codes)  # 创建路径对象

# 创建一个 PathPatch 对象，不在 y=0 处添加边距
patch = patches.PathPatch(barpath)
patch.sticky_edges.y[:] = [0]

fig, ax = plt.subplots()  # 创建一个图形窗口和坐标轴
ax.add_patch(patch)  # 将 PathPatch 对象添加到坐标轴上
ax.autoscale_view()  # 自动调整坐标轴范围
plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.PathPatch`
#    - `matplotlib.path`
#    - `matplotlib.path.Path`
#    - `matplotlib.path.Path.make_compound_path_from_polys`
#    - `matplotlib.axes.Axes.add_patch`
#    - `matplotlib.collections.PathCollection`
#
#    This example shows an alternative to
#
#    - `matplotlib.collections.PolyCollection`
#    - `matplotlib.axes.Axes.hist`
```