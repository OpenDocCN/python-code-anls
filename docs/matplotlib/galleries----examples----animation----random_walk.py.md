# `D:\src\scipysrc\matplotlib\galleries\examples\animation\random_walk.py`

```
"""
=======================
Animated 3D random walk
=======================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 库用于数值计算

import matplotlib.animation as animation  # 导入 matplotlib 的动画模块

# 设置随机种子以便结果可重现
np.random.seed(19680801)


def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    # 随机生成起始位置
    start_pos = np.random.random(3)
    # 生成步长在 [-max_step, max_step] 之间的随机步长
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    # 计算累积随机步长得到随机漫步路径
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk


def update_lines(num, walks, lines):
    # 更新每条线的数据到当前步数为止的位置
    for line, walk in zip(lines, walks):
        line.set_data_3d(walk[:num, :].T)
    return lines


# Data: 40 random walks as (num_steps, 3) arrays
num_steps = 30
# 生成 40 条长度为 num_steps 的随机漫步路径
walks = [random_walk(num_steps) for index in range(40)]

# Attaching 3D axis to the figure
fig = plt.figure()  # 创建一个新的 matplotlib 图形对象
ax = fig.add_subplot(projection="3d")  # 在图形对象上添加一个 3D 坐标系

# Create lines initially without data
# 创建与随机漫步路径数量相同的空线条对象列表
lines = [ax.plot([], [], [])[0] for _ in walks]

# Setting the Axes properties
# 设置坐标轴的范围和标签
ax.set(xlim3d=(0, 1), xlabel='X')
ax.set(ylim3d=(0, 1), ylabel='Y')
ax.set(zlim3d=(0, 1), zlabel='Z')

# Creating the Animation object
# 创建动画对象，将 update_lines 函数作为更新函数
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines), interval=100)

plt.show()  # 显示动画
```