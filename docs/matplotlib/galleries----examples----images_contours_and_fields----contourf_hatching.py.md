# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\contourf_hatching.py`

```py
"""
=================
Contourf Hatching
=================

Demo filled contour plots with hatched patterns.
"""
# 导入 matplotlib 的 pyplot 模块，并使用 plt 作为别名
import matplotlib.pyplot as plt
# 导入 numpy 模块，并使用 np 作为别名
import numpy as np

# invent some numbers, turning the x and y arrays into simple
# 2d arrays, which make combining them together easier.
# 创建 x 和 y 数组，分别包含等间隔的数值，然后将它们变形为二维数组
x = np.linspace(-3, 5, 150).reshape(1, -1)
y = np.linspace(-3, 5, 120).reshape(-1, 1)
# 计算 z 数组，表示 x 和 y 的余弦和正弦的和
z = np.cos(x) + np.sin(y)

# we no longer need x and y to be 2 dimensional, so flatten them.
# 将 x 和 y 数组展平为一维数组
x, y = x.flatten(), y.flatten()

# %%
# Plot 1: the simplest hatched plot with a colorbar

# 创建图表和轴对象，返回 fig1 和 ax1
fig1, ax1 = plt.subplots()
# 绘制填充的等高线图，使用指定的图案填充
cs = ax1.contourf(x, y, z, hatches=['-', '/', '\\', '//'],
                  cmap='gray', extend='both', alpha=0.5)
# 添加颜色条到图表
fig1.colorbar(cs)

# %%
# Plot 2: a plot of hatches without color with a legend

# 创建图表和轴对象，返回 fig2 和 ax2
fig2, ax2 = plt.subplots()
# 绘制等高线，指定线条的样式和颜色
n_levels = 6
ax2.contour(x, y, z, n_levels, colors='black', linestyles='-')
# 绘制填充的等高线图，指定不同的图案填充
cs = ax2.contourf(x, y, z, n_levels, colors='none',
                  hatches=['.', '/', '\\', None, '\\\\', '*'],
                  extend='lower')

# create a legend for the contour set
# 为等高线集合创建图例
artists, labels = cs.legend_elements(str_format='{:2.1f}'.format)
ax2.legend(artists, labels, handleheight=2, framealpha=1)
# 展示图表
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
#    - `matplotlib.contour.ContourSet`
#    - `matplotlib.contour.ContourSet.legend_elements`
```