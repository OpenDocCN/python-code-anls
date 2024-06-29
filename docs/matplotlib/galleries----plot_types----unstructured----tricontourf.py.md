# `D:\src\scipysrc\matplotlib\galleries\plot_types\unstructured\tricontourf.py`

```py
"""
====================
tricontourf(x, y, z)
====================
Draw contour regions on an unstructured triangular grid.

See `~matplotlib.axes.Axes.tricontourf`.
"""
# 导入 matplotlib 的 pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 使用指定的样式表 '_mpl-gallery-nogrid'
plt.style.use('_mpl-gallery-nogrid')

# 生成数据:
np.random.seed(1)
# 生成 256 个在区间 [-3, 3] 中均匀分布的随机数作为 x 坐标
x = np.random.uniform(-3, 3, 256)
# 生成 256 个在区间 [-3, 3] 中均匀分布的随机数作为 y 坐标
y = np.random.uniform(-3, 3, 256)
# 根据 x, y 计算对应的 z 值，形成数据的高度
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
# 生成用于绘制等高线的级别，从 z 的最小值到最大值均匀分布 7 个级别
levels = np.linspace(z.min(), z.max(), 7)

# 创建图形和坐标轴对象
fig, ax = plt.subplots()

# 绘制散点图，显示数据点
ax.plot(x, y, 'o', markersize=2, color='grey')
# 根据 x, y, z 数据绘制三角形网格的填充等高线图
ax.tricontourf(x, y, z, levels=levels)

# 设置 x 和 y 轴的显示范围
ax.set(xlim=(-3, 3), ylim=(-3, 3))

# 显示图形
plt.show()
```