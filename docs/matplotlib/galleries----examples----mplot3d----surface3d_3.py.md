# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\surface3d_3.py`

```
"""
=========================
3D surface (checkerboard)
=========================

Demonstrates plotting a 3D surface colored in a checkerboard pattern.
"""

import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from matplotlib.ticker import LinearLocator  # 从matplotlib中导入LinearLocator类

ax = plt.figure().add_subplot(projection='3d')  # 创建一个3D图形的子图对象

# Make data.
X = np.arange(-5, 5, 0.25)  # 生成X轴数据，范围从-5到5，步长为0.25
xlen = len(X)  # 获取X轴数据长度
Y = np.arange(-5, 5, 0.25)  # 生成Y轴数据，范围从-5到5，步长为0.25
ylen = len(Y)  # 获取Y轴数据长度
X, Y = np.meshgrid(X, Y)  # 生成网格数据
R = np.sqrt(X**2 + Y**2)  # 计算每个点到原点的距离
Z = np.sin(R)  # 计算Z轴数据，使用sin函数生成波浪状数据

# Create an empty array of strings with the same shape as the meshgrid, and
# populate it with two colors in a checkerboard pattern.
colortuple = ('y', 'b')  # 定义颜色元组，黄色和蓝色
colors = np.empty(X.shape, dtype=str)  # 创建一个与网格形状相同的空字符串数组
for y in range(ylen):  # 遍历Y轴上的每个元素
    for x in range(xlen):  # 遍历X轴上的每个元素
        colors[y, x] = colortuple[(x + y) % len(colortuple)]  # 按照棋盘格模式填充颜色数组

# Plot the surface with face colors taken from the array we made.
surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)  # 绘制3D曲面图，使用颜色数组作为面颜色，线宽为0

# Customize the z axis.
ax.set_zlim(-1, 1)  # 设置Z轴的显示范围为-1到1
ax.zaxis.set_major_locator(LinearLocator(6))  # 设置Z轴主刻度的定位器为LinearLocator，分为6个主刻度

plt.show()  # 显示图形
```