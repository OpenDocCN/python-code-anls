# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\shared_axis_demo.py`

```
"""
===========
Shared axis
===========

You can share the x- or y-axis limits for one axis with another by
passing an `~.axes.Axes` instance as a *sharex* or *sharey* keyword argument.

Changing the axis limits on one Axes will be reflected automatically
in the other, and vice-versa, so when you navigate with the toolbar
the Axes will follow each other on their shared axis.  Ditto for
changes in the axis scaling (e.g., log vs. linear).  However, it is
possible to have differences in tick labeling, e.g., you can selectively
turn off the tick labels on one Axes.

The example below shows how to customize the tick labels on the
various axes.  Shared axes share the tick locator, tick formatter,
view limits, and transformation (e.g., log, linear).  But the ticklabels
themselves do not share properties.  This is a feature and not a bug,
because you may want to make the tick labels smaller on the upper
axes, e.g., in the example below.

If you want to turn off the ticklabels for a given Axes (e.g., on
subplot(211) or subplot(212)), you cannot do the standard trick::

   setp(ax2, xticklabels=[])

because this changes the tick Formatter, which is shared among all
Axes.  But you can alter the visibility of the labels, which is a
property::

  setp(ax2.get_xticklabels(), visible=False)

"""
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入numpy数值计算模块

t = np.arange(0.01, 5.0, 0.01)  # 创建时间序列数组
s1 = np.sin(2 * np.pi * t)  # 计算正弦波信号
s2 = np.exp(-t)  # 计算指数衰减信号
s3 = np.sin(4 * np.pi * t)  # 计算高频正弦波信号

ax1 = plt.subplot(311)  # 创建第一个子图，3行1列中的第1个子图
plt.plot(t, s1)  # 绘制第一个子图的曲线
plt.tick_params('x', labelsize=6)  # 设置x轴刻度标签的大小为6

# share x only
ax2 = plt.subplot(312, sharex=ax1)  # 创建第二个子图，3行1列中的第2个子图，并共享x轴
plt.plot(t, s2)  # 绘制第二个子图的曲线
plt.tick_params('x', labelbottom=False)  # 不显示第二个子图的x轴刻度标签

# share x and y
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)  # 创建第三个子图，3行1列中的第3个子图，并共享x和y轴
plt.plot(t, s3)  # 绘制第三个子图的曲线
plt.xlim(0.01, 5.0)  # 设置x轴的显示范围
plt.show()  # 显示所有子图
```