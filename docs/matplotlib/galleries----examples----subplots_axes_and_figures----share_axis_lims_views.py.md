# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\share_axis_lims_views.py`

```py
"""
Sharing axis limits and views
=============================

It's common to make two or more plots which share an axis, e.g., two subplots
with time as a common axis.  When you pan and zoom around on one, you want the
other to move around with you.  To facilitate this, matplotlib Axes support a
``sharex`` and ``sharey`` attribute.  When you create a `~.pyplot.subplot` or
`~.pyplot.axes`, you can pass in a keyword indicating what Axes you want to
share with.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 生成一个包含时间序列的数组
t = np.arange(0, 10, 0.01)

# 创建第一个子图，2行1列的布局，第一个子图
ax1 = plt.subplot(211)
# 在第一个子图上绘制正弦波，时间为 x 轴
ax1.plot(t, np.sin(2*np.pi*t))

# 创建第二个子图，2行1列的布局，第二个子图，并共享 x 轴（与第一个子图共享）
ax2 = plt.subplot(212, sharex=ax1)
# 在第二个子图上绘制正弦波，时间为 x 轴（与第一个子图共享）
ax2.plot(t, np.sin(4*np.pi*t))

# 显示绘制的图形
plt.show()
```