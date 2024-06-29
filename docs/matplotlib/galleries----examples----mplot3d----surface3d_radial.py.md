# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\surface3d_radial.py`

```py
"""
=================================
3D surface with polar coordinates
=================================

Demonstrates plotting a surface defined in polar coordinates.
Uses the reversed version of the YlGnBu colormap.
Also demonstrates writing axis labels with latex math mode.

Example contributed by Armin Moser.
"""

# 导入需要使用的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库并简写为 plt
import numpy as np  # 导入 numpy 库并简写为 np

# 创建一个新的图形窗口
fig = plt.figure()

# 在图形窗口中添加一个三维子图
ax = fig.add_subplot(projection='3d')

# 在极坐标系中创建网格并计算对应的 Z 值
r = np.linspace(0, 1.25, 50)  # 创建半径的线性空间
p = np.linspace(0, 2*np.pi, 50)  # 创建角度的线性空间
R, P = np.meshgrid(r, p)  # 创建 R 和 P 的网格
Z = ((R**2 - 1)**2)  # 计算 Z 值，这里是一个特定的数学表达式

# 将极坐标系转换为笛卡尔坐标系
X, Y = R*np.cos(P), R*np.sin(P)  # 根据极坐标转换公式计算 X 和 Y 坐标

# 绘制三维曲面图
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)  # 使用指定的颜色映射绘制曲面

# 调整 Z 轴的限制和添加 LaTeX 数学标签
ax.set_zlim(0, 1)  # 设置 Z 轴的显示范围
ax.set_xlabel(r'$\phi_\mathrm{real}$')  # 设置 X 轴的标签，使用 LaTeX 数学模式
ax.set_ylabel(r'$\phi_\mathrm{im}$')  # 设置 Y 轴的标签，使用 LaTeX 数学模式
ax.set_zlabel(r'$V(\phi)$')  # 设置 Z 轴的标签，使用 LaTeX 数学模式

# 显示图形
plt.show()
```