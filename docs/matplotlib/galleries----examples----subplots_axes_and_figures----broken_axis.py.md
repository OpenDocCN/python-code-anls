# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\broken_axis.py`

```
"""
===========
Broken Axis
===========

Broken axis example, where the y-axis will have a portion cut out.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于生成随机数据

np.random.seed(19680801)  # 设置随机数种子，确保随机数据可重复

pts = np.random.rand(30)*.2  # 生成长度为30的随机数组，每个元素乘以0.2
# 现在生成两个远离其他点的异常值点
pts[[3, 14]] += .8

# 如果仅简单绘制pts，由于异常值，会丢失大部分有趣的细节。因此，我们要在y轴上进行‘断裂’或‘切割’，
# 使用顶部(ax1)显示异常值，底部(ax2)显示大部分数据的细节
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 创建包含两个子图的图形对象
fig.subplots_adjust(hspace=0.05)  # 调整子图之间的垂直间距

# 在两个子图上绘制相同的数据
ax1.plot(pts)  # 在ax1上绘制pts数据
ax2.plot(pts)  # 在ax2上绘制pts数据

# 缩放视图，限制显示数据的不同部分
ax1.set_ylim(.78, 1.)  # 仅显示异常值
ax2.set_ylim(0, .22)  # 显示大部分数据的细节

# 隐藏ax1和ax2之间的轴线
ax1.spines.bottom.set_visible(False)  # 隐藏ax1的底部轴线
ax2.spines.top.set_visible(False)  # 隐藏ax2的顶部轴线
ax1.xaxis.tick_top()  # 将ax1的x轴刻度放置在顶部
ax1.tick_params(labeltop=False)  # 不在顶部放置x轴刻度标签
ax2.xaxis.tick_bottom()  # 将ax2的x轴刻度放置在底部

# 现在，让我们转向切割的斜线部分。
# 我们在轴坐标系中创建线对象，其中(0,0)、(0,1)、(1,0)和(1,1)是轴的四个角落。
# 斜线本身是在这些位置上的标记，使得线保持其角度和位置，不受轴大小或比例的影响
# 最后，我们需要禁用裁剪。
d = .5  # 斜线垂直到水平范围的比例
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)  # 在ax1上绘制斜线
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)  # 在ax2上绘制斜线

# 显示图形
plt.show()
```