# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\tricontourf3d.py`

```
"""
=================================
Triangular 3D filled contour plot
=================================

Filled contour plots of unstructured triangular grids.

The data used is the same as in the second plot of :doc:`trisurf3d_2`.
:doc:`tricontour3d` shows the unfilled version of this example.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy

import matplotlib.tri as tri  # 导入三角网格处理模块

# 定义网格参数
n_angles = 48  # 角度方向的点数
n_radii = 8  # 半径方向的点数
min_radius = 0.25  # 最小半径

# 在极坐标系中创建网格，并计算 x, y, z 坐标
radii = np.linspace(min_radius, 0.95, n_radii)  # 在最小半径到0.95之间均匀分布的半径
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)  # 在0到2*pi之间均匀分布的角度
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)  # 扩展角度数组以匹配半径维度
angles[:, 1::2] += np.pi/n_angles  # 调整奇数列的角度，以创建非均匀的角度分布

# 计算三维坐标
x = (radii*np.cos(angles)).flatten()  # x 坐标
y = (radii*np.sin(angles)).flatten()  # y 坐标
z = (np.cos(radii)*np.cos(3*angles)).flatten()  # z 坐标

# 创建自定义三角网格
triang = tri.Triangulation(x, y)

# 掩盖不需要的三角形
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)

# 创建 3D subplot
ax = plt.figure().add_subplot(projection='3d')

# 绘制三角形填充等高线图
ax.tricontourf(triang, z, cmap=plt.cm.CMRmap)

# 自定义视角以便更好地理解图形
ax.view_init(elev=45.)

# 显示图形
plt.show()
```