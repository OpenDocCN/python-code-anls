# `D:\src\scipysrc\matplotlib\galleries\examples\spines\centered_spines_with_arrows.py`

```
"""
===========================
Centered spines with arrows
===========================

This example shows a way to draw a "math textbook" style plot, where the
spines ("axes lines") are drawn at ``x = 0`` and ``y = 0``, and have arrows at
their ends.
"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于数学运算和数据处理

fig, ax = plt.subplots()  # 创建一个新的Figure和Axes对象

# Move the left and bottom spines to x = 0 and y = 0, respectively.
ax.spines[["left", "bottom"]].set_position(("data", 0))
# 将左侧和底部的坐标轴线移动到x=0和y=0的位置

# Hide the top and right spines.
ax.spines[["top", "right"]].set_visible(False)
# 隐藏顶部和右侧的坐标轴线

# Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
# case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
# respectively) and the other one (1) is an axes coordinate (i.e., at the very
# right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
# actually spills out of the Axes.
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
# 在y轴末端绘制一个黑色的大于号箭头，箭头的位置是y=0处，transform=ax.get_yaxis_transform()表示坐标系转换为y轴坐标系，clip_on=False表示不裁剪超出Axes范围的部分
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
# 在x轴末端绘制一个黑色的上箭头，箭头的位置是x=0处，transform=ax.get_xaxis_transform()表示坐标系转换为x轴坐标系，clip_on=False表示不裁剪超出Axes范围的部分

# Some sample data.
x = np.linspace(-0.5, 1., 100)  # 生成一个从-0.5到1之间100个均匀间隔的数值序列作为x轴数据
ax.plot(x, np.sin(x*np.pi))  # 在Axes上绘制sin函数曲线，使用numpy的sin函数和pi常数

plt.show()  # 显示绘制的图形
```