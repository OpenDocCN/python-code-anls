# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\parasite_simple.py`

```py
"""
===============
Parasite Simple
===============
"""

# 导入 matplotlib 的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt

# 从 axes_grid1 中导入 host_subplot 函数，用于创建带主次坐标轴的子图
from mpl_toolkits.axes_grid1 import host_subplot

# 创建主坐标轴对象
host = host_subplot(111)

# 创建次坐标轴对象
par = host.twinx()

# 设置主坐标轴的 x 轴标签
host.set_xlabel("Distance")

# 设置主坐标轴的 y 轴标签
host.set_ylabel("Density")

# 设置次坐标轴的 y 轴标签
par.set_ylabel("Temperature")

# 绘制主坐标轴的线条，返回线条对象 p1
p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")

# 绘制次坐标轴的线条，返回线条对象 p2
p2, = par.plot([0, 1, 2], [0, 3, 2], label="Temperature")

# 设置主坐标轴的图例颜色为线条的颜色，但这里的 labelcolor 参数不正确，应为 labelcolors
host.legend(labelcolor="linecolor")

# 设置主坐标轴 y 轴标签的颜色为 p1 线条的颜色
host.yaxis.get_label().set_color(p1.get_color())

# 设置次坐标轴 y 轴标签的颜色为 p2 线条的颜色
par.yaxis.get_label().set_color(p2.get_color())

# 显示绘制的图形
plt.show()
```