# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\eventcollection_demo.py`

```
"""
====================
EventCollection Demo
====================

Plot two curves, then use `.EventCollection`\s to mark the locations of the x
and y data points on the respective Axes for each curve.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值计算

from matplotlib.collections import EventCollection  # 从 matplotlib 的 collections 模块中导入 EventCollection 类

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设置随机数种子，保证结果的可重现性

# create random data
xdata = np.random.random([2, 10])  # 生成一个 2x10 的随机数矩阵，表示 x 轴的数据

# split the data into two parts
xdata1 = xdata[0, :]  # 取第一行作为第一组 x 数据
xdata2 = xdata[1, :]  # 取第二行作为第二组 x 数据

# sort the data so it makes clean curves
xdata1.sort()  # 对第一组 x 数据进行排序，确保绘制的曲线平滑
xdata2.sort()  # 对第二组 x 数据进行排序，确保绘制的曲线平滑

# create some y data points
ydata1 = xdata1 ** 2  # 根据第一组 x 数据生成对应的 y 数据
ydata2 = 1 - xdata2 ** 3  # 根据第二组 x 数据生成对应的 y 数据

# plot the data
fig = plt.figure()  # 创建一个新的图形窗口
ax = fig.add_subplot(1, 1, 1)  # 添加一个子图，1行1列的第一个位置
ax.plot(xdata1, ydata1, color='tab:blue')  # 绘制第一条曲线，使用蓝色
ax.plot(xdata2, ydata2, color='tab:orange')  # 绘制第二条曲线，使用橙色

# create the events marking the x data points
xevents1 = EventCollection(xdata1, color='tab:blue', linelength=0.05)  # 创建标记第一组 x 数据点的 EventCollection 对象，使用蓝色
xevents2 = EventCollection(xdata2, color='tab:orange', linelength=0.05)  # 创建标记第二组 x 数据点的 EventCollection 对象，使用橙色

# create the events marking the y data points
yevents1 = EventCollection(ydata1, color='tab:blue', linelength=0.05,
                           orientation='vertical')  # 创建标记第一组 y 数据点的 EventCollection 对象，使用蓝色，垂直方向
yevents2 = EventCollection(ydata2, color='tab:orange', linelength=0.05,
                           orientation='vertical')  # 创建标记第二组 y 数据点的 EventCollection 对象，使用橙色，垂直方向

# add the events to the axis
ax.add_collection(xevents1)  # 将第一组 x 数据点的 EventCollection 对象添加到子图中
ax.add_collection(xevents2)  # 将第二组 x 数据点的 EventCollection 对象添加到子图中
ax.add_collection(yevents1)  # 将第一组 y 数据点的 EventCollection 对象添加到子图中
ax.add_collection(yevents2)  # 将第二组 y 数据点的 EventCollection 对象添加到子图中

# set the limits
ax.set_xlim([0, 1])  # 设置 x 轴显示范围为 [0, 1]
ax.set_ylim([0, 1])  # 设置 y 轴显示范围为 [0, 1]

ax.set_title('line plot with data points')  # 设置子图标题

# display the plot
plt.show()  # 显示绘制的图形
```