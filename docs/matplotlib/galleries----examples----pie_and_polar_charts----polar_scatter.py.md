# `D:\src\scipysrc\matplotlib\galleries\examples\pie_and_polar_charts\polar_scatter.py`

```
"""
==========================
Scatter plot on polar axis
==========================

Size increases radially in this example and color increases with angle
(just to verify the symbols are being scattered correctly).
"""
# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# Fixing random state for reproducibility
# 设置随机数种子，以便结果可复现
np.random.seed(19680801)

# Compute areas and colors
# 计算散点的面积和颜色
N = 150
r = 2 * np.random.rand(N)  # 随机生成 N 个 [0, 1) 之间的数，并放大到 [0, 2)
theta = 2 * np.pi * np.random.rand(N)  # 随机生成 N 个 [0, 1) 之间的数，并映射到 [0, 2π)
area = 200 * r**2  # 根据半径 r 计算面积，放大到 [0, 200)
colors = theta  # 根据角度 theta 确定颜色

# 创建一个新的图形窗口
fig = plt.figure()
# 在图形窗口中添加一个极坐标子图
ax = fig.add_subplot(projection='polar')
# 绘制散点图，位置由 theta 和 r 确定，颜色由 colors 确定，大小由 area 确定，使用 'hsv' 颜色映射，透明度为 0.75
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

# %%
# Scatter plot on polar axis, with offset origin
# ----------------------------------------------
#
# The main difference with the previous plot is the configuration of the origin
# radius, producing an annulus. Additionally, the theta zero location is set to
# rotate the plot.

# 创建一个新的图形窗口
fig = plt.figure()
# 在图形窗口中添加一个极坐标子图
ax = fig.add_subplot(projection='polar')
# 绘制散点图，位置由 theta 和 r 确定，颜色由 colors 确定，大小由 area 确定，使用 'hsv' 颜色映射，透明度为 0.75
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

# 设置极坐标的原点半径为 -2.5
ax.set_rorigin(-2.5)
# 设置极坐标的起始角度为 'W' 方向，偏移量为 10 度
ax.set_theta_zero_location('W', offset=10)

# %%
# Scatter plot on polar axis confined to a sector
# -----------------------------------------------
#
# The main difference with the previous plots is the configuration of the
# theta start and end limits, producing a sector instead of a full circle.

# 创建一个新的图形窗口
fig = plt.figure()
# 在图形窗口中添加一个极坐标子图
ax = fig.add_subplot(projection='polar')
# 绘制散点图，位置由 theta 和 r 确定，颜色由 colors 确定，大小由 area 确定，使用 'hsv' 颜色映射，透明度为 0.75
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

# 设置极坐标的起始角度限制为 45 度
ax.set_thetamin(45)
# 设置极坐标的结束角度限制为 135 度
ax.set_thetamax(135)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`
#    - `matplotlib.projections.polar`
#    - `matplotlib.projections.polar.PolarAxes.set_rorigin`
#    - `matplotlib.projections.polar.PolarAxes.set_theta_zero_location`
#    - `matplotlib.projections.polar.PolarAxes.set_thetamin`
#    - `matplotlib.projections.polar.PolarAxes.set_thetamax`
```