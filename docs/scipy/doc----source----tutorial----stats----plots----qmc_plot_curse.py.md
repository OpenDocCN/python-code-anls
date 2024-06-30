# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\qmc_plot_curse.py`

```
"""Visualize the curse-of-dimensionality.

It presents a saturated design in 1, 2 and 3 dimensions for a
given discretization.
"""
# 导入需要的绘图库和数学计算库
import matplotlib.pyplot as plt
import numpy as np

# 设定离散化的数量
disc = 10

# 在 [0, 1] 区间均匀分布地生成离散化的数据点
x = np.linspace(0, 1, disc)
y = np.linspace(0, 1, disc)
z = np.linspace(0, 1, disc)

# 创建三维网格，用于三维可视化
xx, yy, zz = np.meshgrid(x, y, z)

# 创建一个图形窗口，并设定其大小
fig = plt.figure(figsize=(12, 4))

# 添加第一个子图，展示一维情况
ax = fig.add_subplot(131)
ax.set_aspect('equal')  # 设置坐标轴纵横比相等
ax.scatter(xx, yy * 0)   # 绘制散点图，y轴乘以0仅展示在x轴上的分布
ax.set_xlabel(r'$x_1$')  # 设置x轴标签
ax.get_yaxis().set_visible(False)  # 隐藏y轴

# 添加第二个子图，展示二维情况
ax = fig.add_subplot(132)
ax.set_aspect('equal')  # 设置坐标轴纵横比相等
ax.scatter(xx, yy)       # 绘制二维散点图
ax.set_xlabel(r'$x_1$')  # 设置x轴标签
ax.set_ylabel(r'$x_2$')  # 设置y轴标签

# 添加第三个子图，展示三维情况，使用3D投影
ax = fig.add_subplot(133, projection='3d')
ax.scatter(xx, yy, zz)   # 绘制三维散点图
ax.set_xlabel(r'$x_1$')  # 设置x轴标签
ax.set_ylabel(r'$x_2$')  # 设置y轴标签
ax.set_zlabel(r'$x_3$')  # 设置z轴标签

# 调整子图的布局，使它们之间的间距更合适
plt.tight_layout(pad=2)

# 显示图形窗口
plt.show()
```