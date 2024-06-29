# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\quiver3d.py`

```
"""
==============
3D quiver plot
==============

Demonstrates plotting directional arrows at points on a 3D meshgrid.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 库，并使用 np 别名

ax = plt.figure().add_subplot(projection='3d')  # 创建一个 3D 图形对象，并获取其子图对象

# 创建一个三维网格
x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),  # 创建 x 坐标轴的网格点数组
                      np.arange(-0.8, 1, 0.2),  # 创建 y 坐标轴的网格点数组
                      np.arange(-0.8, 1, 0.8))  # 创建 z 坐标轴的网格点数组

# 创建箭头的方向数据
u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)  # 箭头 x 方向的数据
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)  # 箭头 y 方向的数据
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))  # 箭头 z 方向的数据

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)  # 在图形上绘制箭头

plt.show()  # 显示绘制的图形
```