# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\simple_axisline.py`

```
"""
===============
Simple Axisline
===============

"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库

from mpl_toolkits.axisartist.axislines import AxesZero  # 从 axisartist 模块导入 AxesZero 类

fig = plt.figure()  # 创建一个新的图形对象
fig.subplots_adjust(right=0.85)  # 调整子图布局，右边留出空间

ax = fig.add_subplot(axes_class=AxesZero)  # 在图形对象中添加一个子图，使用 AxesZero 类作为轴

# make right and top axis invisible
ax.axis["right"].set_visible(False)  # 设置右边轴线不可见
ax.axis["top"].set_visible(False)    # 设置顶部轴线不可见

# make xzero axis (horizontal axis line through y=0) visible.
ax.axis["xzero"].set_visible(True)  # 设置 x=0 的水平轴线可见
ax.axis["xzero"].label.set_text("Axis Zero")  # 设置 x=0 轴线的标签文本为 "Axis Zero"

ax.set_ylim(-2, 4)  # 设置 y 轴的数值范围为 -2 到 4
ax.set_xlabel("Label X")  # 设置 x 轴的标签文本为 "Label X"
ax.set_ylabel("Label Y")  # 设置 y 轴的标签文本为 "Label Y"
# Or:
# ax.axis["bottom"].label.set_text("Label X")
# ax.axis["left"].label.set_text("Label Y")

# make new (right-side) yaxis, but with some offset
ax.axis["right2"] = ax.new_fixed_axis(loc="right", offset=(20, 0))  # 创建一个新的右边轴线，带有一定的偏移量
ax.axis["right2"].label.set_text("Label Y2")  # 设置右边轴线的标签文本为 "Label Y2"

ax.plot([-2, 3, 2])  # 绘制简单的折线图

plt.show()  # 显示图形
```