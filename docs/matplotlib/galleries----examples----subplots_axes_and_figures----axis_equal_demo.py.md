# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\axis_equal_demo.py`

```
"""
=======================
Equal axis aspect ratio
=======================

How to set and adjust plots with equal axis aspect ratios.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

# 绘制半径为3的圆形

an = np.linspace(0, 2 * np.pi, 100)  # 在0到2π之间生成100个等间距的角度
fig, axs = plt.subplots(2, 2)  # 创建一个2x2的子图布局

axs[0, 0].plot(3 * np.cos(an), 3 * np.sin(an))  # 在第一个子图中绘制圆形，横坐标为3*cos(an)，纵坐标为3*sin(an)
axs[0, 0].set_title('not equal, looks like ellipse', fontsize=10)  # 设置子图标题为'not equal, looks like ellipse'

axs[0, 1].plot(3 * np.cos(an), 3 * np.sin(an))  # 在第二个子图中绘制圆形，横坐标为3*cos(an)，纵坐标为3*sin(an)
axs[0, 1].axis('equal')  # 设置第二个子图的坐标轴比例为相等，使得图形显示为圆形
axs[0, 1].set_title('equal, looks like circle', fontsize=10)  # 设置子图标题为'equal, looks like circle'

axs[1, 0].plot(3 * np.cos(an), 3 * np.sin(an))  # 在第三个子图中绘制圆形，横坐标为3*cos(an)，纵坐标为3*sin(an)
axs[1, 0].axis('equal')  # 设置第三个子图的坐标轴比例为相等，使得图形显示为圆形
axs[1, 0].set(xlim=(-3, 3), ylim=(-3, 3))  # 设置第三个子图的坐标轴范围为(-3, 3)，以确保显示为圆形
axs[1, 0].set_title('still a circle, even after changing limits', fontsize=10)  # 设置子图标题为'still a circle, even after changing limits'

axs[1, 1].plot(3 * np.cos(an), 3 * np.sin(an))  # 在第四个子图中绘制圆形，横坐标为3*cos(an)，纵坐标为3*sin(an)
axs[1, 1].set_aspect('equal', 'box')  # 设置第四个子图的数据坐标与显示坐标比例相等，自动调整数据范围以显示为圆形
axs[1, 1].set_title('still a circle, auto-adjusted data limits', fontsize=10)  # 设置子图标题为'still a circle, auto-adjusted data limits'

fig.tight_layout()  # 调整子图布局，使其更为紧凑

plt.show()  # 显示绘制的图形
```