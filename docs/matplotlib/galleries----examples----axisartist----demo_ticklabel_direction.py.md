# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\demo_ticklabel_direction.py`

```py
"""
===================
Ticklabel direction
===================

"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库

import mpl_toolkits.axisartist.axislines as axislines  # 导入 axislines 模块


def setup_axes(fig, pos):
    ax = fig.add_subplot(pos, axes_class=axislines.Axes)  # 在指定位置添加带有 axislines 的子图
    ax.set_yticks([0.2, 0.8])  # 设置 y 轴刻度位置
    ax.set_xticks([0.2, 0.8])  # 设置 x 轴刻度位置
    return ax  # 返回设置好的子图对象


fig = plt.figure(figsize=(6, 3))  # 创建一个大小为 6x3 的图形对象
fig.subplots_adjust(bottom=0.2)  # 调整子图的底部边距

ax = setup_axes(fig, 131)  # 设置第一个子图的轴线
for axis in ax.axis.values():
    axis.major_ticks.set_tick_out(True)  # 将主刻度线设置为向外显示
# or you can simply do "ax.axis[:].major_ticks.set_tick_out(True)"

ax = setup_axes(fig, 132)  # 设置第二个子图的轴线
ax.axis["left"].set_axis_direction("right")  # 设置左侧轴线方向为右侧
ax.axis["bottom"].set_axis_direction("top")  # 设置底部轴线方向为顶部
ax.axis["right"].set_axis_direction("left")  # 设置右侧轴线方向为左侧
ax.axis["top"].set_axis_direction("bottom")  # 设置顶部轴线方向为底部

ax = setup_axes(fig, 133)  # 设置第三个子图的轴线
ax.axis["left"].set_axis_direction("right")  # 设置左侧轴线方向为右侧
ax.axis[:].major_ticks.set_tick_out(True)  # 所有轴线的主刻度线向外显示

ax.axis["left"].label.set_text("Long Label Left")  # 设置左侧轴线的标签文本
ax.axis["bottom"].label.set_text("Label Bottom")  # 设置底部轴线的标签文本
ax.axis["right"].label.set_text("Long Label Right")  # 设置右侧轴线的标签文本
ax.axis["right"].label.set_visible(True)  # 显示右侧轴线的标签
ax.axis["left"].label.set_pad(0)  # 设置左侧轴线标签的间距为 0
ax.axis["bottom"].label.set_pad(10)  # 设置底部轴线标签的间距为 10

plt.show()  # 显示图形
```