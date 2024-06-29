# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\surface3d.py`

```
"""
=====================
3D surface (colormap)
=====================

Demonstrates plotting a 3D surface colored with the coolwarm colormap.
The surface is made opaque by using ``antialiased=False``.

Also demonstrates using the `.LinearLocator` and custom formatting for the
z axis tick labels.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 模块

from matplotlib import cm  # 从 matplotlib 中导入 colormap 模块
from matplotlib.ticker import LinearLocator  # 从 matplotlib.ticker 中导入 LinearLocator 类

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})  # 创建一个 3D 图形对象和坐标系对象

# Make data.
X = np.arange(-5, 5, 0.25)  # 创建 X 轴数据，范围从 -5 到 5，步长为 0.25
Y = np.arange(-5, 5, 0.25)  # 创建 Y 轴数据，范围从 -5 到 5，步长为 0.25
X, Y = np.meshgrid(X, Y)  # 根据 X 和 Y 创建网格坐标
R = np.sqrt(X**2 + Y**2)  # 计算每个网格点到原点的距离
Z = np.sin(R)  # 计算每个网格点的 Z 值，使用正弦函数

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,  # 绘制三维曲面，使用 coolwarm 颜色映射
                       linewidth=0, antialiased=False)  # 设置曲面线宽为 0，关闭反锯齿

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)  # 设置 Z 轴的显示范围
ax.zaxis.set_major_locator(LinearLocator(10))  # 设置 Z 轴主刻度的定位器为 LinearLocator，分为 10 个刻度
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')  # 设置 Z 轴主刻度标签的格式为保留两位小数的字符串格式

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)  # 添加颜色条，映射数值到颜色，收缩因子为 0.5，长宽比为 5

plt.show()  # 显示绘制的图形
```