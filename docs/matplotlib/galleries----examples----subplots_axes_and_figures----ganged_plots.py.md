# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\ganged_plots.py`

```
"""
==========================
Creating adjacent subplots
==========================

To create plots that share a common axis (visually) you can set the hspace
between the subplots to zero. Passing sharex=True when creating the subplots
will automatically turn off all x ticks and labels except those on the bottom
axis.

In this example the plots share a common x-axis, but you can follow the same
logic to supply a common y-axis.
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于生成数据
import numpy as np

# 生成时间序列数据
t = np.arange(0.0, 2.0, 0.01)

# 计算三个不同的数据序列
s1 = np.sin(2 * np.pi * t)
s2 = np.exp(-t)
s3 = s1 * s2

# 创建包含三个子图的图形对象 fig 和子图对象 axs
fig, axs = plt.subplots(3, 1, sharex=True)
# 设置子图之间的垂直间距为零
fig.subplots_adjust(hspace=0)

# 在第一个子图 axs[0] 上绘制 s1
axs[0].plot(t, s1)
# 手动设置第一个子图的 y 轴刻度值
axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
# 设置第一个子图的 y 轴范围为 -1 到 1
axs[0].set_ylim(-1, 1)

# 在第二个子图 axs[1] 上绘制 s2
axs[1].plot(t, s2)
# 手动设置第二个子图的 y 轴刻度值
axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
# 设置第二个子图的 y 轴范围为 0 到 1
axs[1].set_ylim(0, 1)

# 在第三个子图 axs[2] 上绘制 s3
axs[2].plot(t, s3)
# 手动设置第三个子图的 y 轴刻度值
axs[2].set_yticks(np.arange(-0.9, 1.0, 0.4))
# 设置第三个子图的 y 轴范围为 -1 到 1
axs[2].set_ylim(-1, 1)

# 显示绘制的图形
plt.show()
```