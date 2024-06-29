# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\scatter3d.py`

```py
"""
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设定随机数种子，以便结果可重复

def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin  # 返回一个包含 n 个在 [vmin, vmax] 范围内均匀分布的随机数数组

fig = plt.figure()  # 创建一个新的图形对象
ax = fig.add_subplot(projection='3d')  # 在图形对象中添加一个 3D 子图

n = 100  # 设置数据点的数量

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# 对于每一组样式和范围设置，绘制一个包含 n 个随机点的散点图，
# 其中 x 范围在 [23, 32]，y 范围在 [0, 100]，z 范围在 [zlow, zhigh]。
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)  # 生成 x 坐标数组
    ys = randrange(n, 0, 100)  # 生成 y 坐标数组
    zs = randrange(n, zlow, zhigh)  # 生成 z 坐标数组
    ax.scatter(xs, ys, zs, marker=m)  # 绘制散点图

ax.set_xlabel('X Label')  # 设置 x 轴标签
ax.set_ylabel('Y Label')  # 设置 y 轴标签
ax.set_zlabel('Z Label')  # 设置 z 轴标签

plt.show()  # 显示图形
```