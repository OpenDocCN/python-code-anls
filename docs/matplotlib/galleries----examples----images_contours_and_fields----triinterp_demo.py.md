# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\triinterp_demo.py`

```
"""
==============
Triinterp Demo
==============

Interpolation from triangular grid to quad grid.
"""

# 导入需要使用的库
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入numpy数学计算库

import matplotlib.tri as mtri  # 导入matplotlib的三角网格处理模块

# 创建三角网格
x = np.asarray([0, 1, 2, 3, 0.5, 1.5, 2.5, 1, 2, 1.5])  # x坐标数组
y = np.asarray([0, 0, 0, 0, 1.0, 1.0, 1.0, 2, 2, 3.0])  # y坐标数组
triangles = [[0, 1, 4], [1, 2, 5], [2, 3, 6], [1, 5, 4], [2, 6, 5], [4, 5, 7],
             [5, 6, 8], [5, 8, 7], [7, 8, 9]]  # 三角形顶点索引数组
triang = mtri.Triangulation(x, y, triangles)  # 创建三角网格对象

# 在规则间隔的四边形网格上进行插值
z = np.cos(1.5 * x) * np.cos(1.5 * y)  # 生成z坐标数组
xi, yi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 3, 20))  # 创建目标四边形网格坐标

# 使用线性插值器进行插值
interp_lin = mtri.LinearTriInterpolator(triang, z)
zi_lin = interp_lin(xi, yi)

# 使用几何类型的三次插值器进行插值
interp_cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
zi_cubic_geom = interp_cubic_geom(xi, yi)

# 使用最小能量类型的三次插值器进行插值
interp_cubic_min_E = mtri.CubicTriInterpolator(triang, z, kind='min_E')
zi_cubic_min_E = interp_cubic_min_E(xi, yi)

# 设置图形和子图
fig, axs = plt.subplots(nrows=2, ncols=2)
axs = axs.flatten()

# 绘制三角网格
axs[0].tricontourf(triang, z)
axs[0].triplot(triang, 'ko-')
axs[0].set_title('Triangular grid')

# 绘制线性插值到四边形网格
axs[1].contourf(xi, yi, zi_lin)
axs[1].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
axs[1].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
axs[1].set_title("Linear interpolation")

# 绘制几何类型的三次插值到四边形网格
axs[2].contourf(xi, yi, zi_cubic_geom)
axs[2].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
axs[2].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
axs[2].set_title("Cubic interpolation,\nkind='geom'")

# 绘制最小能量类型的三次插值到四边形网格
axs[3].contourf(xi, yi, zi_cubic_min_E)
axs[3].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
axs[3].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
axs[3].set_title("Cubic interpolation,\nkind='min_E'")

fig.tight_layout()  # 调整布局，使子图紧凑显示
plt.show()  # 显示图形
```