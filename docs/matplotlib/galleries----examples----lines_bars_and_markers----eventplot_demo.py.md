# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\eventplot_demo.py`

```py
"""
==============
Eventplot demo
==============

An `~.axes.Axes.eventplot` showing sequences of events with various line
properties. The plot is shown in both horizontal and vertical orientations.
"""

# 导入 matplotlib.pyplot 库作为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库作为 np
import numpy as np

# 设置全局字体大小为 8.0
import matplotlib
matplotlib.rcParams['font.size'] = 8.0

# 设定随机数种子以便结果可重现
np.random.seed(19680801)

# 创建随机数据
data1 = np.random.random([6, 50])

# 为每组位置设置不同的颜色
colors1 = [f'C{i}' for i in range(6)]

# 为每组位置设置不同的线条偏移量，注意有些重叠
lineoffsets1 = [-15, -3, 1, 1.5, 6, 10]
# 为每组位置设置不同的线条长度
linelengths1 = [5, 2, 1, 1, 3, 1.5]

# 创建 2x2 的图形布局
fig, axs = plt.subplots(2, 2)

# 在第一个子图中创建水平事件图
axs[0, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                    linelengths=linelengths1)

# 在第二个子图中创建垂直事件图
axs[1, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                    linelengths=linelengths1, orientation='vertical')

# 创建另一组随机数据
# Gamma 分布仅用于美学目的
data2 = np.random.gamma(4, size=[60, 50])

# 这次使用单独的值设定参数
# 这些值将用于所有数据集（除了 lineoffsets2，它设置每个数据集之间的增量）
colors2 = 'black'
lineoffsets2 = 1
linelengths2 = 1

# 在第三个子图中创建水平事件图
axs[0, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                    linelengths=linelengths2)

# 在第四个子图中创建垂直事件图
axs[1, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                    linelengths=linelengths2, orientation='vertical')

# 显示图形
plt.show()
```