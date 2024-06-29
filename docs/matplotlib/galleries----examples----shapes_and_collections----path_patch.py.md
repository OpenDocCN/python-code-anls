# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\path_patch.py`

```
"""
================
PathPatch object
================

This example shows how to create `~.path.Path` and `~.patches.PathPatch`
objects through Matplotlib's API.
"""

# 导入 matplotlib 的 pyplot 和 patches 模块
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# 创建一个图形窗口和轴对象
fig, ax = plt.subplots()

# 定义路径对象的常量
Path = mpath.Path

# 定义路径数据，包括移动到、曲线和闭合多边形等命令和顶点坐标
path_data = [
    (Path.MOVETO, (1.58, -2.57)),
    (Path.CURVE4, (0.35, -1.1)),
    (Path.CURVE4, (-1.75, 2.0)),
    (Path.CURVE4, (0.375, 2.0)),
    (Path.LINETO, (0.85, 1.15)),
    (Path.CURVE4, (2.2, 3.2)),
    (Path.CURVE4, (3, 0.05)),
    (Path.CURVE4, (2.0, -0.5)),
    (Path.CLOSEPOLY, (1.58, -2.57)),
]
# 将路径数据解压缩成命令列表和顶点列表
codes, verts = zip(*path_data)

# 使用顶点和命令创建路径对象
path = mpath.Path(verts, codes)

# 使用路径对象创建路径补丁对象，设置颜色为红色，透明度为0.5
patch = mpatches.PathPatch(path, facecolor='r', alpha=0.5)

# 将路径补丁对象添加到轴上
ax.add_patch(patch)

# 绘制控制点和连接线
x, y = zip(*path.vertices)
line, = ax.plot(x, y, 'go-')

# 设置网格和坐标轴比例相等
ax.grid()
ax.axis('equal')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.path`
#    - `matplotlib.path.Path`
#    - `matplotlib.patches`
#    - `matplotlib.patches.PathPatch`
#    - `matplotlib.axes.Axes.add_patch`
```