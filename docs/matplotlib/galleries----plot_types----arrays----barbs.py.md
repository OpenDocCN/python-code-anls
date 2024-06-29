# `D:\src\scipysrc\matplotlib\galleries\plot_types\arrays\barbs.py`

```py
"""
=================
barbs(X, Y, U, V)
=================
Plot a 2D field of wind barbs.

See `~matplotlib.axes.Axes.barbs`.
"""
# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt
# 导入numpy库，用于科学计算
import numpy as np

# 使用指定的绘图风格
plt.style.use('_mpl-gallery-nogrid')

# 创建数据：
# 创建网格数据 X 和 Y
X, Y = np.meshgrid([1, 2, 3, 4], [1, 2, 3, 4])
# 风向角度数据
angle = np.pi / 180 * np.array([[15., 30, 35, 45],
                                [25., 40, 55, 60],
                                [35., 50, 65, 75],
                                [45., 60, 75, 90]])
# 风速振幅数据
amplitude = np.array([[5, 10, 25, 50],
                      [10, 15, 30, 60],
                      [15, 26, 50, 70],
                      [20, 45, 80, 100]])
# 根据角度和振幅计算 U 和 V 分量
U = amplitude * np.sin(angle)
V = amplitude * np.cos(angle)

# 绘图：
# 创建一个新的图形和坐标轴对象
fig, ax = plt.subplots()

# 绘制风羽图
ax.barbs(X, Y, U, V, barbcolor='C0', flagcolor='C0', length=7, linewidth=1.5)

# 设置坐标轴范围
ax.set(xlim=(0, 4.5), ylim=(0, 4.5))

# 显示图形
plt.show()
```