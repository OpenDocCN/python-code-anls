# `D:\src\scipysrc\matplotlib\galleries\examples\animation\rain.py`

```py
"""
===============
Rain simulation
===============

Simulates rain drops on a surface by animating the scale and opacity
of 50 scatter points.

Author: Nicolas P. Rougier

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 库

from matplotlib.animation import FuncAnimation  # 从 matplotlib.animation 模块导入 FuncAnimation 类

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设置随机种子以便结果可重现


# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))  # 创建一个新的图形，并设置尺寸为 7x7 英寸
ax = fig.add_axes([0, 0, 1, 1], frameon=False)  # 在图形上添加一个填充整个图形的坐标轴，并关闭边框
ax.set_xlim(0, 1), ax.set_xticks([])  # 设置 x 轴的范围为 0 到 1，并移除刻度
ax.set_ylim(0, 1), ax.set_yticks([])  # 设置 y 轴的范围为 0 到 1，并移除刻度

# Create rain data
n_drops = 50  # 定义雨滴数量为 50
rain_drops = np.zeros(n_drops, dtype=[('position', float, (2,)),  # 创建一个 numpy 数组来存储雨滴的位置、大小、增长率和颜色信息
                                      ('size',     float),
                                      ('growth',   float),
                                      ('color',    float, (4,))])

# Initialize the raindrops in random positions and with
# random growth rates.
rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))  # 在 [0, 1) 范围内随机初始化雨滴的位置
rain_drops['growth'] = np.random.uniform(50, 200, n_drops)  # 随机初始化雨滴的增长率

# Construct the scatter which we will update during animation
# as the raindrops develop.
scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],  # 创建散点图对象，用于动态更新雨滴的位置、大小和颜色
                  s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
                  facecolors='none')


def update(frame_number):
    # Get an index which we can use to re-spawn the oldest raindrop.
    current_index = frame_number % n_drops  # 计算当前帧数对应的雨滴索引

    # Make all colors more transparent as time progresses.
    rain_drops['color'][:, 3] -= 1.0 / len(rain_drops)  # 随着时间推移使所有雨滴颜色更加透明
    rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)  # 将透明度限制在 [0, 1] 范围内

    # Make all circles bigger.
    rain_drops['size'] += rain_drops['growth']  # 增大所有雨滴的大小

    # Pick a new position for oldest rain drop, resetting its size,
    # color and growth factor.
    rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)  # 为最老的雨滴选择一个新的位置
    rain_drops['size'][current_index] = 5  # 重置最老雨滴的大小
    rain_drops['color'][current_index] = (0, 0, 0, 1)  # 重置最老雨滴的颜色为黑色不透明
    rain_drops['growth'][current_index] = np.random.uniform(50, 200)  # 重置最老雨滴的增长率

    # Update the scatter collection, with the new colors, sizes and positions.
    scat.set_edgecolors(rain_drops['color'])  # 更新散点图对象的边缘颜色
    scat.set_sizes(rain_drops['size'])  # 更新散点图对象的大小
    scat.set_offsets(rain_drops['position'])  # 更新散点图对象的位置


# Construct the animation, using the update function as the animation director.
animation = FuncAnimation(fig, update, interval=10, save_count=100)  # 创建动画对象，使用 update 函数作为动画更新函数，每 10 毫秒更新一次，共保存 100 帧
plt.show()  # 显示动画
```