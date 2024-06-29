# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\auto_ticks.py`

```
"""
====================================
Automatically setting tick positions
====================================

Setting the behavior of tick auto-placement.

By default, Matplotlib will choose the number of ticks and tick positions so
that there is a reasonable number of ticks on the axis and they are located
at "round" numbers.

As a result, there may be no ticks on the edges of the plot.
"""

# 导入 Matplotlib 库
import matplotlib.pyplot as plt
import numpy as np

# 设置随机数种子，以便结果可重复
np.random.seed(19680801)

# 创建一个图形窗口和坐标系
fig, ax = plt.subplots()

# 生成一组坐标点
dots = np.linspace(0.3, 1.2, 10)
X, Y = np.meshgrid(dots, dots)
x, y = X.ravel(), Y.ravel()

# 在坐标系上绘制散点图，颜色基于 x 和 y 的值
ax.scatter(x, y, c=x+y)

# 显示图形
plt.show()

# %%
# 如果想要保持刻度在整数位置，并且在边缘也有刻度，可以将 :rc:`axes.autolimit_mode` 设置为 'round_numbers'。
# 这会将坐标轴限制扩展到下一个整数位置。

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

# 注意：限制是在绘制时计算的。因此，在使用 :rc:`axes.autolimit_mode` 的上下文管理器中，
# 确保 ``show()`` 命令位于上下文之内是很重要的。

fig, ax = plt.subplots()
ax.scatter(x, y, c=x+y)
plt.show()

# %%
# 即使使用 `.Axes.set_xmargin` / `.Axes.set_ymargin` 设置数据周围的额外边距，
# 也仍然会尊重整数位置的 autolimit_mode：

fig, ax = plt.subplots()
ax.scatter(x, y, c=x+y)
ax.set_xmargin(0.8)
plt.show()
```