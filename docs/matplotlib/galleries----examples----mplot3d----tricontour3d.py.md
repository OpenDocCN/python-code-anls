# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\tricontour3d.py`

```
"""
==========================
Triangular 3D contour plot
==========================

Contour plots of unstructured triangular grids.

The data used is the same as in the second plot of :doc:`trisurf3d_2`.
:doc:`tricontourf3d` shows the filled version of this example.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 库

import matplotlib.tri as tri  # 导入 matplotlib 的三角剖分模块

# 定义网格的参数
n_angles = 48  # 角度数
n_radii = 8  # 半径数
min_radius = 0.25  # 最小半径

# 在极坐标下创建网格，并计算 x, y, z 坐标
radii = np.linspace(min_radius, 0.95, n_radii)  # 在最小半径到0.95之间生成等间距的半径值
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)  # 在0到2π之间生成等间距的角度值，不包括终点
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)  # 扩展角度数组以匹配半径数组的维度
angles[:, 1::2] += np.pi/n_angles  # 调整奇数列的角度值，增加角度的间距

x = (radii*np.cos(angles)).flatten()  # 计算 x 坐标
y = (radii*np.sin(angles)).flatten()  # 计算 y 坐标
z = (np.cos(radii)*np.cos(3*angles)).flatten()  # 计算 z 坐标

# 创建自定义的三角剖分
triang = tri.Triangulation(x, y)

# 屏蔽掉不需要的三角形
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)

# 创建一个 3D subplot
ax = plt.figure().add_subplot(projection='3d')

# 绘制三角形的等高线
ax.tricontour(triang, z, cmap=plt.cm.CMRmap)

# 自定义视角，使图形更容易理解
ax.view_init(elev=45.)

# 显示图形
plt.show()
```