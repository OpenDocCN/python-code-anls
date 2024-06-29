# `D:\src\scipysrc\matplotlib\galleries\plot_types\arrays\quiver.py`

```
"""
==================
quiver(X, Y, U, V)
==================
Plot a 2D field of arrows.

See `~matplotlib.axes.Axes.quiver`.
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 使用指定的样式风格 '_mpl-gallery-nogrid'
plt.style.use('_mpl-gallery-nogrid')

# 创建数据
# 生成 x 和 y 的均匀间隔的数组
x = np.linspace(-4, 4, 6)
y = np.linspace(-4, 4, 6)
# 创建网格点坐标矩阵 X 和 Y
X, Y = np.meshgrid(x, y)
# 计算箭头的水平和垂直方向的分量
U = X + Y
V = Y - X

# 绘图
# 创建图形对象和轴对象
fig, ax = plt.subplots()

# 在轴对象 ax 上绘制箭头图
ax.quiver(X, Y, U, V, color="C0", angles='xy',
          scale_units='xy', scale=5, width=.015)

# 设置 x 和 y 轴的显示范围
ax.set(xlim=(-5, 5), ylim=(-5, 5))

# 显示图形
plt.show()
```