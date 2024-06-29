# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\fill_betweenx_demo.py`

```
"""
==================
Fill Betweenx Demo
==================

Using `~.Axes.fill_betweenx` to color along the horizontal direction between
two curves.
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，并使用 np 别名
import numpy as np

# 创建 y 值数组，从 0 到 2，步长为 0.01
y = np.arange(0.0, 2, 0.01)
# 根据 y 值计算 x1 数组，使用正弦函数
x1 = np.sin(2 * np.pi * y)
# 根据 y 值计算 x2 数组，使用更高频率的正弦函数，并乘以 1.2
x2 = 1.2 * np.sin(4 * np.pi * y)

# 创建包含三个子图的 Figure 对象，并获取子图对象的数组 ax1, ax2, ax3
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize=(6, 6))

# 在 ax1 子图中填充 x1 和 0 之间的区域
ax1.fill_betweenx(y, 0, x1)
ax1.set_title('between (x1, 0)')  # 设置子图标题

# 在 ax2 子图中填充 x1 和 1 之间的区域
ax2.fill_betweenx(y, x1, 1)
ax2.set_title('between (x1, 1)')  # 设置子图标题
ax2.set_xlabel('x')  # 设置 x 轴标签

# 在 ax3 子图中填充 x1 和 x2 之间的区域
ax3.fill_betweenx(y, x1, x2)
ax3.set_title('between (x1, x2)')  # 设置子图标题

# %%
# 现在填充 x1 和 x2 满足逻辑条件的区域。注意，这与调用 fill_between(y[where], x1[where], x2[where]) 不同，
# 因为它考虑了多个连续区域之间的边缘效应。

# 创建包含两个子图的 Figure 对象，并获取子图对象的数组 ax 和 ax1
fig, [ax, ax1] = plt.subplots(1, 2, sharey=True, figsize=(6, 6))

# 在 ax 子图中绘制 x1 和 x2 的曲线，并填充 x1 和 x2 之间满足条件 x2 >= x1 的区域为绿色
ax.plot(x1, y, x2, y, color='black')
ax.fill_betweenx(y, x1, x2, where=x2 >= x1, facecolor='green')
# 填充 x1 和 x2 之间满足条件 x2 <= x1 的区域为红色
ax.fill_betweenx(y, x1, x2, where=x2 <= x1, facecolor='red')
ax.set_title('fill_betweenx where')  # 设置子图标题

# 测试对掩码数组的支持
x2 = np.ma.masked_greater(x2, 1.0)
ax1.plot(x1, y, x2, y, color='black')
ax1.fill_betweenx(y, x1, x2, where=x2 >= x1, facecolor='green')
ax1.fill_betweenx(y, x1, x2, where=x2 <= x1, facecolor='red')
ax1.set_title('regions with x2 > 1 are masked')  # 设置子图标题

# %%
# 此示例说明了一个问题；由于数据格点化，交叉点处存在不希望出现的未填充三角形。一种解决方法是在绘图前对所有数组进行插值，以获得非常细的网格。

# 显示图形
plt.show()
```