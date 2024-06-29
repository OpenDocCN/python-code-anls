# `D:\src\scipysrc\matplotlib\galleries\plot_types\unstructured\tricontour.py`

```py
python
"""
===================
tricontour(x, y, z)
===================
在非结构化三角形网格上绘制等高线。

参见 `~matplotlib.axes.Axes.tricontour`。
"""
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库
import numpy as np  # 导入 numpy 库

plt.style.use('_mpl-gallery-nogrid')  # 使用指定的绘图风格

# 生成数据:
np.random.seed(1)  # 设置随机数种子以便复现
x = np.random.uniform(-3, 3, 256)  # 在区间[-3, 3]均匀分布的256个随机数作为 x 坐标
y = np.random.uniform(-3, 3, 256)  # 在区间[-3, 3]均匀分布的256个随机数作为 y 坐标
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)  # 根据 x, y 计算 z 值
levels = np.linspace(z.min(), z.max(), 7)  # 生成等分的 z 值范围，用于绘制等高线

# 绘图:
fig, ax = plt.subplots()  # 创建图形和子图对象

ax.plot(x, y, 'o', markersize=2, color='lightgrey')  # 在图中绘制散点图，表示数据点
ax.tricontour(x, y, z, levels=levels)  # 在三角形网格上绘制等高线图

ax.set(xlim=(-3, 3), ylim=(-3, 3))  # 设置 x 和 y 轴的显示范围

plt.show()  # 显示图形
```