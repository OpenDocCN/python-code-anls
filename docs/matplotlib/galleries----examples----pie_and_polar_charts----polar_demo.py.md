# `D:\src\scipysrc\matplotlib\galleries\examples\pie_and_polar_charts\polar_demo.py`

```py
"""
==========
Polar plot
==========

Demo of a line plot on a polar axis.
"""
# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt
# 导入numpy库，用于生成数值数据
import numpy as np

# 生成半开区间[0, 2)内，步长为0.01的等差数组，作为极径数据
r = np.arange(0, 2, 0.01)
# 根据极径数据计算极角数据
theta = 2 * np.pi * r

# 创建一个极坐标子图
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# 在极坐标子图上绘制极线图
ax.plot(theta, r)
# 设置极径轴的最大值
ax.set_rmax(2)
# 设置极径轴的刻度位置
ax.set_rticks([0.5, 1, 1.5, 2])  # 更少的极径刻度
# 调整极径标签的位置，使其远离绘制的线
ax.set_rlabel_position(-22.5)
# 在图上显示网格线
ax.grid(True)

# 设置图的标题
ax.set_title("A line plot on a polar axis", va='bottom')
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.projections.polar`
#    - `matplotlib.projections.polar.PolarAxes`
#    - `matplotlib.projections.polar.PolarAxes.set_rticks`
#    - `matplotlib.projections.polar.PolarAxes.set_rmax`
#    - `matplotlib.projections.polar.PolarAxes.set_rlabel_position`
```