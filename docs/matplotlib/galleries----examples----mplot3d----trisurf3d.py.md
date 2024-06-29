# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\trisurf3d.py`

```
"""
======================
Triangular 3D surfaces
======================

Plot a 3D surface with a triangular mesh.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于科学计算
import numpy as np

# 定义放射线和角度数量
n_radii = 8
n_angles = 36

# 创建放射线和角度的数组（去除半径为 r=0 的重复项）
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]

# 将极坐标 (放射线, 角度) 转换为笛卡尔坐标 (x, y)
# 在这个阶段手动添加 (0, 0)，以确保在 (x, y) 平面上没有重复点
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# 计算 z 值，以形成弯曲的曲面
z = np.sin(-x*y)

# 创建一个具有 3D 投影的子图
ax = plt.figure().add_subplot(projection='3d')

# 绘制三角面片的三维图形，使用 x, y, z 数组作为坐标，设置线宽和抗锯齿
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

# 显示绘图
plt.show()
```