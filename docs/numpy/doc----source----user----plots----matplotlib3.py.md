# `.\numpy\doc\source\user\plots\matplotlib3.py`

```py
# 导入 NumPy 库，通常用于处理数组和数值计算
import numpy as np
# 导入 Matplotlib 的 pyplot 模块，用于绘图操作
import matplotlib.pyplot as plt

# 创建一个新的图形对象
fig = plt.figure()
# 在图形对象上添加一个三维坐标系的子图
ax = fig.add_subplot(projection='3d')

# 生成一个 X 轴上的坐标数组，范围从 -5 到 5，步长为 0.15
X = np.arange(-5, 5, 0.15)
# 生成一个 Y 轴上的坐标数组，范围从 -5 到 5，步长为 0.15
Y = np.arange(-5, 5, 0.15)
# 将 X 和 Y 坐标数组转换成网格形式
X, Y = np.meshgrid(X, Y)
# 计算每个网格点的极径 R
R = np.sqrt(X**2 + Y**2)
# 计算每个网格点的 Z 值，使用 sin 函数作为高度值
Z = np.sin(R)

# 在三维坐标系上绘制一个表面图，使用 viridis 颜色映射
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')

# 显示绘制的图形
plt.show()
```