# `D:\src\scipysrc\matplotlib\galleries\examples\spines\spines_dropped.py`

```py
"""
==============
Dropped spines
==============

Demo of spines offset from the axes (a.k.a. "dropped spines").
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np


# 定义函数 adjust_spines，调整图轴的边框样式
def adjust_spines(ax, visible_spines):
    # 移除内部刻度标签
    ax.label_outer(remove_inner_ticks=True)
    # 设置网格线颜色为浅灰色
    ax.grid(color='0.9')

    # 遍历轴的边框
    for loc, spine in ax.spines.items():
        # 如果当前边框位置在可见边框列表中
        if loc in visible_spines:
            # 设置边框向外偏移 10 点（points）
            spine.set_position(('outward', 10))
        else:
            # 否则隐藏当前边框
            spine.set_visible(False)


# 生成从 0 到 2π 的等间距的 100 个数值作为 x 值
x = np.linspace(0, 2 * np.pi, 100)

# 创建一个 2x2 的图像布局
fig, axs = plt.subplots(2, 2)

# 在各个子图中绘制不同的函数
axs[0, 0].plot(x, np.sin(x))
axs[0, 1].plot(x, np.cos(x))
axs[1, 0].plot(x, -np.cos(x))
axs[1, 1].plot(x, -np.sin(x))

# 调用 adjust_spines 函数，分别调整各个子图的边框样式
adjust_spines(axs[0, 0], ['left'])
adjust_spines(axs[0, 1], [])
adjust_spines(axs[1, 0], ['left', 'bottom'])
adjust_spines(axs[1, 1], ['bottom'])

# 显示绘制的图像
plt.show()
```