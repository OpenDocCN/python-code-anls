# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\compound_path.py`

```
"""
=============
Compound path
=============

Make a compound path -- in this case two simple polygons, a rectangle
and a triangle.  Use ``CLOSEPOLY`` and ``MOVETO`` for the different parts of
the compound path
"""

# 导入 matplotlib 的 pyplot 模块，简称为 plt
import matplotlib.pyplot as plt

# 从 matplotlib 的 patches 模块导入 PathPatch 类和 path 模块
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# 定义顶点列表和代码列表
vertices = []
codes = []

# 定义第一个多边形的路径代码和顶点
codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
vertices = [(1, 1), (1, 2), (2, 2), (2, 1), (0, 0)]

# 追加第二个多边形的路径代码和顶点
codes += [Path.MOVETO] + [Path.LINETO]*2 + [Path.CLOSEPOLY]
vertices += [(4, 4), (5, 5), (5, 4), (0, 0)]

# 创建 Path 对象，使用顶点和代码
path = Path(vertices, codes)

# 创建 PathPatch 对象，设置边框为绿色
pathpatch = PathPatch(path, facecolor='none', edgecolor='green')

# 创建图形和坐标轴对象
fig, ax = plt.subplots()

# 将 PathPatch 对象添加到坐标轴中
ax.add_patch(pathpatch)

# 设置图表标题
ax.set_title('A compound path')

# 自动缩放视图以适应所有图形
ax.autoscale_view()

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
#    - `matplotlib.axes.Axes.autoscale_view`
```