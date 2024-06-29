# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\quad_bezier.py`

```
"""
============
Bezier Curve
============

This example showcases the `~.patches.PathPatch` object to create a Bezier
polycurve path patch.
"""

# 导入 matplotlib 的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt

# 导入 matplotlib 的 patches 模块，用于创建图形路径补丁
import matplotlib.patches as mpatches

# 导入 matplotlib 的 path 模块，用于定义路径
import matplotlib.path as mpath

# 定义别名 Path 来简化 mpath.Path 的使用
Path = mpath.Path

# 创建一个图形窗口和一个子图
fig, ax = plt.subplots()

# 创建一个 PathPatch 对象，表示一个 Bezier 曲线路径补丁
pp1 = mpatches.PathPatch(
    # 使用 Path 对象定义路径，包括起始点、两个三次贝塞尔曲线点和闭合点
    Path([(0, 0), (1, 0), (1, 1), (0, 0)],
         [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
    # 设置路径补丁的填充颜色为无（透明）
    fc="none", transform=ax.transData)

# 将路径补丁添加到子图中
ax.add_patch(pp1)

# 在图上绘制一个红色圆点
ax.plot([0.75], [0.25], "ro")

# 设置图的标题
ax.set_title('The red point should be on the path')

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