# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\plot_interp_grid.py`

```
import math  # 导入数学库，用于数学计算

import numpy as np  # 导入numpy库，用于数组操作
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
from matplotlib import ticker  # 从matplotlib库中导入ticker模块，用于设置刻度

orders = [2, 3]  # 定义插值阶数列表

fig, axes = plt.subplots(1, len(orders), figsize=(11, 5))  # 创建子图，根据插值阶数设定子图数量和大小
n_cells = 7  # 网格大小为 (n_cells, n_cells)

# 插值的目标坐标 (xi, yi)
xi, yi = 3.3, 3.7

def get_start(cc, order):
    if order % 1 == 0:
        start = math.floor(cc) - order // 2
    else:
        start = math.floor(cc + 0.5) - order // 2
    return start

for ax, order in zip(axes, orders):
    # 在像素中心位置绘制空心圆圈
    for n in range(n_cells):
        ax.plot(np.arange(n_cells), -np.full(n_cells, n), 'ko', fillstyle='none')

    # 绘制像素边界
    for n in range(n_cells + 1):
        ax.plot([n - 0.5, n - 0.5], [0.5, -n_cells + .5], 'k-')
        ax.plot([-0.5, n_cells - .5], [-n + 0.5, -n + 0.5], 'k-')

    # 绘制示例插值坐标位置
    ax.plot([xi], [-yi], 'rx')

    # 绘制参与插值的点的填充圆圈
    startx = get_start(xi, order)
    starty = get_start(yi, order)
    xc = np.tile(np.arange(startx, startx + order + 1)[:, np.newaxis], (1, order + 1)).ravel()
    yc = np.tile(np.arange(starty, starty + order + 1)[np.newaxis, :], (order + 1, 1)).ravel()
    ax.plot(xc, -yc, 'ko')

    # 设置子图标题，显示插值阶数
    ax.set_title(f"Interpolation (order = {order})", fontdict=dict(size=16, weight='bold'))

    # 设置坐标轴属性
    ax.axis('square')  # 设置坐标轴为正方形
    ax.set_xticks(np.arange(n_cells + 1))  # 设置x轴刻度
    ax.xaxis.tick_top()  # 将x轴刻度放在顶部
    ax.xaxis.set_label_position('top')  # 设置x轴标签位置为顶部
    yticks = ticker.FixedLocator(-np.arange(n_cells, -1, -1))  # 设置y轴刻度位置
    ax.yaxis.set_major_locator(yticks)
    yticklabels = ticker.FixedFormatter(np.arange(n_cells, -1, -1))  # 设置y轴刻度标签
    ax.yaxis.set_major_formatter(yticklabels)
    ax.set_ylim([-n_cells + 0.5, 0.5])  # 设置y轴显示范围
    ax.set_xlim([-0.5, n_cells - 0.5])  # 设置x轴显示范围

plt.tight_layout()  # 调整子图布局，使其紧凑显示
plt.plot()  # 绘图
```