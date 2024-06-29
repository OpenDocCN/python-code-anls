# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\demo_gridspec06.py`

```
r"""
================
Nested GridSpecs
================

This example demonstrates the use of nested `.GridSpec`\s.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值计算


def squiggle_xy(a, b, c, d):
    # 生成在 [0, 2π) 范围内以 0.05 为步长的数组
    i = np.arange(0.0, 2*np.pi, 0.05)
    # 返回正弦和余弦函数的乘积，生成图形的曲线数据
    return np.sin(i*a)*np.cos(i*b), np.sin(i*c)*np.cos(i*d)


fig = plt.figure(figsize=(8, 8))  # 创建一个大小为 8x8 的图形窗口
outer_grid = fig.add_gridspec(4, 4, wspace=0, hspace=0)  # 在图形中添加一个4x4的网格布局，设置水平和垂直间距为0

for a in range(4):
    for b in range(4):
        # 在外部网格位置 (a, b) 添加一个内部的子网格布局，大小为3x3，间距设置为0
        inner_grid = outer_grid[a, b].subgridspec(3, 3, wspace=0, hspace=0)
        # 在内部子网格布局上创建所有的子图
        axs = inner_grid.subplots()
        for (c, d), ax in np.ndenumerate(axs):
            # 在每个子图上绘制由 squiggle_xy 函数生成的曲线
            ax.plot(*squiggle_xy(a + 1, b + 1, c + 1, d + 1))
            # 设置子图的坐标轴刻度为空列表，不显示刻度
            ax.set(xticks=[], yticks=[])

# 仅显示外部子图的边框
for ax in fig.get_axes():
    ss = ax.get_subplotspec()
    # 根据子图的位置决定是否显示上下左右边框
    ax.spines.top.set_visible(ss.is_first_row())
    ax.spines.bottom.set_visible(ss.is_last_row())
    ax.spines.left.set_visible(ss.is_first_col())
    ax.spines.right.set_visible(ss.is_last_col())

plt.show()  # 显示图形
```