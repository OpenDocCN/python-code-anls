# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\2dcollections3d.py`

```py
"""
=======================
Plot 2D data on 3D plot
=======================

Demonstrates using ax.plot's *zdir* keyword to plot 2D data on
selective axes of a 3D plot.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入numpy库

# 创建一个带有3D投影的子图
ax = plt.figure().add_subplot(projection='3d')

# 使用x和y轴绘制正弦曲线
x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')

# 在x和z轴上绘制散点图数据（每种颜色20个2D点）
colors = ('r', 'g', 'b', 'k')

# 设置随机种子以确保结果可重现性
np.random.seed(19680801)

x = np.random.sample(20 * len(colors))
y = np.random.sample(20 * len(colors))
c_list = []
for c in colors:
    c_list.extend([c] * 20)
# 使用zdir='y'，这些点的y值固定为zs值0，并且(x, y)点在x和z轴上绘制
ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x, z)')

# 创建图例，设置坐标轴限制和标签
ax.legend()  # 创建图例
ax.set_xlim(0, 1)  # 设置x轴范围
ax.set_ylim(0, 1)  # 设置y轴范围
ax.set_zlim(0, 1)  # 设置z轴范围
ax.set_xlabel('X')  # 设置x轴标签
ax.set_ylabel('Y')  # 设置y轴标签
ax.set_zlabel('Z')  # 设置z轴标签

# 自定义视角，使散点图点位于y=0平面上更易于观察
ax.view_init(elev=20., azim=-35, roll=0)

plt.show()  # 显示图形
```