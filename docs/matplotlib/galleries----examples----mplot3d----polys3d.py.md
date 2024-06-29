# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\polys3d.py`

```py
"""
====================
Generate 3D polygons
====================

Demonstrate how to create polygons in 3D. Here we stack 3 hexagons.
"""

import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入numpy数学库

from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 从matplotlib的3D绘图工具包中导入多边形3D集合类

# Coordinates of a hexagon
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)  # 计算六边形顶点的角度，从0到2π，不包括终点
x = np.cos(angles)  # 计算六边形顶点的x坐标
y = np.sin(angles)  # 计算六边形顶点的y坐标
zs = [-3, -2, -1]  # 定义三个不同高度的z坐标

# Close the hexagon by repeating the first vertex
x = np.append(x, x[0])  # 为了闭合六边形，将第一个顶点再添加一次到末尾
y = np.append(y, y[0])  # 同样，将第一个顶点的y坐标再添加一次到末尾

verts = []  # 初始化一个空列表用于存储所有多边形的顶点信息
for z in zs:
    verts.append(list(zip(x*z, y*z, np.full_like(x, z))))  # 将每个高度对应的顶点信息添加到verts中，形成多边形的顶点列表
verts = np.array(verts)  # 将verts转换为numpy数组，便于后续处理

ax = plt.figure().add_subplot(projection='3d')  # 创建一个新的3D图形对象，并添加一个3D坐标轴子图

poly = Poly3DCollection(verts, alpha=.7)  # 创建一个3D多边形集合对象，使用前面生成的顶点信息verts，设置透明度为0.7
ax.add_collection3d(poly)  # 将多边形集合添加到3D坐标轴中显示

ax.set_aspect('equalxy')  # 设置坐标轴的纵横比为相等，保证显示的图形不会出现拉伸变形

plt.show()  # 显示绘制的3D图形
```