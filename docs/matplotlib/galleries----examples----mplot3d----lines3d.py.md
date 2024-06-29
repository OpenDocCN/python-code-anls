# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\lines3d.py`

```py
"""
================
Parametric curve
================

This example demonstrates plotting a parametric curve in 3D.
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 库，用于数值计算

# 创建一个带有 3D 投影的子图
ax = plt.figure().add_subplot(projection='3d')

# 准备参数数组 x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)  # 在 -4π 到 4π 之间生成 100 个等间隔的值作为角度参数
z = np.linspace(-2, 2, 100)  # 在 -2 到 2 之间生成 100 个等间隔的值作为 z 坐标
r = z**2 + 1  # 计算 r 的值，这里是一个依赖于 z 的函数
x = r * np.sin(theta)  # 根据参数 theta 和 r 计算 x 坐标
y = r * np.cos(theta)  # 根据参数 theta 和 r 计算 y 坐标

# 绘制 3D 参数曲线，并添加标签
ax.plot(x, y, z, label='parametric curve')
ax.legend()

# 显示图形
plt.show()
```