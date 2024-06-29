# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\surface3d_2.py`

```
"""
========================
3D surface (solid color)
========================

Demonstrates a very basic plot of a 3D surface using a solid color.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

# 创建一个新的图形
fig = plt.figure()

# 添加一个3D子图
ax = fig.add_subplot(projection='3d')

# 生成数据
u = np.linspace(0, 2 * np.pi, 100)  # 在0到2*pi之间生成100个均匀分布的数作为参数u
v = np.linspace(0, np.pi, 100)  # 在0到pi之间生成100个均匀分布的数作为参数v
x = 10 * np.outer(np.cos(u), np.sin(v))  # 计算x坐标值
y = 10 * np.outer(np.sin(u), np.sin(v))  # 计算y坐标值
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))  # 计算z坐标值

# 绘制三维表面图
ax.plot_surface(x, y, z)

# 设置等比例缩放
ax.set_aspect('equal')

# 显示图形
plt.show()
```