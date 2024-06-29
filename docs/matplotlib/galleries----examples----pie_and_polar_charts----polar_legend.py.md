# `D:\src\scipysrc\matplotlib\galleries\examples\pie_and_polar_charts\polar_legend.py`

```
"""
============
Polar legend
============

Using a legend on a polar-axis plot.
"""

# 导入 matplotlib 库中的 pyplot 模块，命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，命名为 np
import numpy as np

# 创建一个新的图形对象
fig = plt.figure()
# 在图形对象上添加一个极坐标子图，背景色设置为 'lightgoldenrodyellow'
ax = fig.add_subplot(projection="polar", facecolor="lightgoldenrodyellow")

# 生成极坐标系下的 r 值数组，从 0 到 3，共 301 个点
r = np.linspace(0, 3, 301)
# 根据 r 计算对应的角度值 theta
theta = 2 * np.pi * r
# 绘制极坐标系上的线条，使用橙色，线宽 3，标签为 'a line'
ax.plot(theta, r, color="tab:orange", lw=3, label="a line")
# 再绘制一条极坐标系上的线条，使用蓝色虚线，线宽 3，标签为 'another line'
ax.plot(0.5 * theta, r, color="tab:blue", ls="--", lw=3, label="another line")
# 设置极坐标系的刻度参数，网格线颜色设置为 'palegoldenrod'
ax.tick_params(grid_color="palegoldenrod")

# 对于极坐标系，将图例稍微移动到远离坐标系中心的位置，避免与坐标系重叠
# 下面的代码将图例的左下角放置在极坐标系外，位置偏移 67.5 度
angle = np.deg2rad(67.5)
ax.legend(loc="lower left",
          bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))

# 显示绘制的图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
#    - `matplotlib.projections.polar`
#    - `matplotlib.projections.polar.PolarAxes`
```