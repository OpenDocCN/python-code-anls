# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_parasite_axes.py`

```
"""
==================
Parasite Axes demo
==================

Create a parasite Axes. Such Axes would share the x scale with a host Axes,
but show a different scale in y direction.

This approach uses `mpl_toolkits.axes_grid1.parasite_axes.HostAxes` and
`mpl_toolkits.axes_grid1.parasite_axes.ParasiteAxes`.

The standard and recommended approach is to use instead standard Matplotlib
axes, as shown in the :doc:`/gallery/spines/multiple_yaxis_with_spines`
example.

An alternative approach using `mpl_toolkits.axes_grid1` and
`mpl_toolkits.axisartist` is shown in the
:doc:`/gallery/axisartist/demo_parasite_axes2` example.
"""

import matplotlib.pyplot as plt

from mpl_toolkits.axisartist.parasite_axes import HostAxes  # 导入主要的宿主轴

fig = plt.figure()  # 创建一个新的图形对象

host = fig.add_axes([0.15, 0.1, 0.65, 0.8], axes_class=HostAxes)  # 在图形上添加主轴
par1 = host.get_aux_axes(viewlim_mode=None, sharex=host)  # 获取第一个辅助轴，共享主轴的 x 刻度
par2 = host.get_aux_axes(viewlim_mode=None, sharex=host)  # 获取第二个辅助轴，共享主轴的 x 刻度

host.axis["right"].set_visible(False)  # 主轴右侧不可见

par1.axis["right"].set_visible(True)  # 第一个辅助轴右侧可见
par1.axis["right"].major_ticklabels.set_visible(True)  # 第一个辅助轴右侧主刻度标签可见
par1.axis["right"].label.set_visible(True)  # 第一个辅助轴右侧轴标签可见

par2.axis["right2"] = par2.new_fixed_axis(loc="right", offset=(60, 0))  # 在第二个辅助轴上创建一个新的固定轴

p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")  # 在主轴上绘制密度图
p2, = par1.plot([0, 1, 2], [0, 3, 2], label="Temperature")  # 在第一个辅助轴上绘制温度图
p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")  # 在第二个辅助轴上绘制速度图

host.set(xlim=(0, 2), ylim=(0, 2), xlabel="Distance", ylabel="Density")  # 设置主轴的 x 和 y 轴范围，以及标签
par1.set(ylim=(0, 4), ylabel="Temperature")  # 设置第一个辅助轴的 y 轴范围和标签
par2.set(ylim=(1, 65), ylabel="Velocity")  # 设置第二个辅助轴的 y 轴范围和标签

host.legend()  # 添加图例到主轴

host.axis["left"].label.set_color(p1.get_color())  # 设置主轴左侧轴标签颜色与密度图线条颜色一致
par1.axis["right"].label.set_color(p2.get_color())  # 设置第一个辅助轴右侧轴标签颜色与温度图线条颜色一致
par2.axis["right2"].label.set_color(p3.get_color())  # 设置第二个辅助轴右侧轴标签颜色与速度图线条颜色一致

plt.show()  # 显示图形
```