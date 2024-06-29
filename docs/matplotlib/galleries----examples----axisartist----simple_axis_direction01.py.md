# `D:\src\scipysrc\matplotlib\galleries\examples\axisartist\simple_axis_direction01.py`

```py
"""
=====================
Simple axis direction
=====================

"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 导入 mpl_toolkits.axisartist 库，用于自定义坐标轴
import mpl_toolkits.axisartist as axisartist

# 创建一个新的图形对象，并设置大小为 4x2.5 英寸
fig = plt.figure(figsize=(4, 2.5))

# 在图形对象中添加一个子图，使用 axisartist.Axes 类来自定义坐标轴
ax1 = fig.add_subplot(axes_class=axisartist.Axes)

# 调整图形的右边距为 0.8，以便留出空间给右边的标签
fig.subplots_adjust(right=0.8)

# 设置左侧坐标轴主刻度标签的方向为顶部
ax1.axis["left"].major_ticklabels.set_axis_direction("top")

# 设置左侧坐标轴的标签文本为 "Left label"
ax1.axis["left"].label.set_text("Left label")

# 设置右侧坐标轴的标签可见
ax1.axis["right"].label.set_visible(True)

# 设置右侧坐标轴的标签文本为 "Right label"
ax1.axis["right"].label.set_text("Right label")

# 设置右侧坐标轴的标签方向为左侧
ax1.axis["right"].label.set_axis_direction("left")

# 显示绘制的图形
plt.show()
```