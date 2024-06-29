# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_parasite_axes2.py`

```py
"""
==================
Parasite axis demo
==================

This example demonstrates the use of parasite axis to plot multiple datasets
onto one single plot.

Notice how in this example, *par1* and *par2* are both obtained by calling
``twinx()``, which ties their x-limits with the host's x-axis. From there, each
of those two axis behave separately from each other: different datasets can be
plotted, and the y-limits are adjusted separately.

This approach uses `mpl_toolkits.axes_grid1.parasite_axes.host_subplot` and
`mpl_toolkits.axisartist.axislines.Axes`.

The standard and recommended approach is to use instead standard Matplotlib
axes, as shown in the :doc:`/gallery/spines/multiple_yaxis_with_spines`
example.

An alternative approach using `mpl_toolkits.axes_grid1.parasite_axes.HostAxes`
and `mpl_toolkits.axes_grid1.parasite_axes.ParasiteAxes` is shown in the
:doc:`/gallery/axisartist/demo_parasite_axes` example.
"""

import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块

from mpl_toolkits import axisartist  # 导入 Matplotlib 的 axisartist 模块
from mpl_toolkits.axes_grid1 import host_subplot  # 导入 Matplotlib 的 host_subplot 函数

host = host_subplot(111, axes_class=axisartist.Axes)  # 创建主图对象
plt.subplots_adjust(right=0.75)  # 调整子图布局，右侧留出空间给双轴

par1 = host.twinx()  # 创建第一个双轴对象 par1
par2 = host.twinx()  # 创建第二个双轴对象 par2

par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))  # 设置 par2 的右侧轴

par1.axis["right"].toggle(all=True)  # 显示 par1 的右侧轴
par2.axis["right"].toggle(all=True)  # 显示 par2 的右侧轴

p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")  # 绘制主轴上的数据点 p1
p2, = par1.plot([0, 1, 2], [0, 3, 2], label="Temperature")  # 绘制 par1 上的数据点 p2
p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")  # 绘制 par2 上的数据点 p3

host.set(xlim=(0, 2), ylim=(0, 2), xlabel="Distance", ylabel="Density")  # 设置主轴的 x、y 轴限制和标签
par1.set(ylim=(0, 4), ylabel="Temperature")  # 设置 par1 的 y 轴限制和标签
par2.set(ylim=(1, 65), ylabel="Velocity")  # 设置 par2 的 y 轴限制和标签

host.legend()  # 在主轴上添加图例

host.axis["left"].label.set_color(p1.get_color())  # 设置主轴左侧标签颜色
par1.axis["right"].label.set_color(p2.get_color())  # 设置 par1 右侧标签颜色
par2.axis["right"].label.set_color(p3.get_color())  # 设置 par2 右侧标签颜色

plt.show()  # 显示绘制的图形
```