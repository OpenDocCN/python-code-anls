# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\contours_in_optimization_demo.py`

```
"""
==============================================
Contouring the solution space of optimizations
==============================================

Contour plotting is particularly handy when illustrating the solution
space of optimization problems.  Not only can `.axes.Axes.contour` be
used to represent the topography of the objective function, it can be
used to generate boundary curves of the constraint functions.  The
constraint lines can be drawn with
`~matplotlib.patheffects.TickedStroke` to distinguish the valid and
invalid sides of the constraint boundaries.

`.axes.Axes.contour` generates curves with larger values to the left
of the contour.  The angle parameter is measured zero ahead with
increasing values to the left.  Consequently, when using
`~matplotlib.patheffects.TickedStroke` to illustrate a constraint in
a typical optimization problem, the angle should be set between
zero and 180 degrees.
"""

# 导入需要的库
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patheffects

# 创建一个图形和坐标轴对象
fig, ax = plt.subplots(figsize=(6, 6))

# 设置网格点数
nx = 101
ny = 105

# 设置调查向量
xvec = np.linspace(0.001, 4.0, nx)
yvec = np.linspace(0.001, 4.0, ny)

# 设置调查矩阵。设计盘载和齿轮比例。
x1, x2 = np.meshgrid(xvec, yvec)

# 计算要绘制的一些数据
obj = x1**2 + x2**2 - 2*x1 - 2*x2 + 2
g1 = -(3*x1 + x2 - 5.5)
g2 = -(x1 + 2*x2 - 4.5)
g3 = 0.8 + x1**-3 - x2

# 绘制等高线，表示目标函数的拓扑图
cntr = ax.contour(x1, x2, obj, [0.01, 0.1, 0.5, 1, 2, 4, 8, 16],
                  colors='black')
# 标记等高线的数值
ax.clabel(cntr, fmt="%2.1f", use_clabeltext=True)

# 绘制约束条件的等高线，并设置路径效果以区分有效和无效约束边界
cg1 = ax.contour(x1, x2, g1, [0], colors='sandybrown')
cg1.set(path_effects=[patheffects.withTickedStroke(angle=135)])

cg2 = ax.contour(x1, x2, g2, [0], colors='orangered')
cg2.set(path_effects=[patheffects.withTickedStroke(angle=60, length=2)])

cg3 = ax.contour(x1, x2, g3, [0], colors='mediumblue')
cg3.set(path_effects=[patheffects.withTickedStroke(spacing=7)])

# 设置坐标轴的范围
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

# 显示图形
plt.show()
```