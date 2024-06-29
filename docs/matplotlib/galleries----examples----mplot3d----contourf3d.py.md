# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\contourf3d.py`

```py
"""
===============
Filled contours
===============

`.Axes3D.contourf` differs from `.Axes3D.contour` in that it creates filled
contours, i.e. a discrete number of colours are used to shade the domain.

This is like a `.Axes.contourf` plot in 2D except that the shaded region
corresponding to the level c is graphed on the plane ``z=c``.
"""

# 导入需要的绘图库
import matplotlib.pyplot as plt

# 导入颜色映射和三维绘图工具
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

# 创建一个带有3D投影的图形，并获取测试数据 X, Y, Z
ax = plt.figure().add_subplot(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

# 使用 contourf 方法在3D图上绘制填充的等高线，使用 coolwarm 颜色映射
ax.contourf(X, Y, Z, cmap=cm.coolwarm)

# 显示绘制的图形
plt.show()
```