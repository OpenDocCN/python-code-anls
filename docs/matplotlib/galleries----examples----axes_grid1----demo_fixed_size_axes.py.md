# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_fixed_size_axes.py`

```py
"""
===============================
Axes with a fixed physical size
===============================

Note that this can be accomplished with the main library for
Axes on Figures that do not change size: :ref:`fixed_size_axes`
"""

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import Divider, Size

# 创建一个新的画布对象，设置大小为 6x6 英寸
fig = plt.figure(figsize=(6, 6))

# 设置水平和垂直尺寸数组，单位为英寸
# 第一个元素是填充，第二个元素是 Axes 的大小
h = [Size.Fixed(1.0), Size.Fixed(4.5)]
v = [Size.Fixed(0.7), Size.Fixed(5.)]

# 创建一个 Divider 对象来划分画布区域
divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# 矩形的宽度和高度被忽略

# 在画布上添加一个 Axes 对象，使用刚刚创建的 Divider 对象定义位置
ax = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))

# 在 Axes 上绘制一个简单的线图
ax.plot([1, 2, 3])

# 创建第二个画布对象，设置大小为 6x6 英寸
fig = plt.figure(figsize=(6, 6))

# 设置水平和垂直尺寸数组，单位为英寸
# 第一个和第三个元素是填充，第二个元素是 Axes 的大小
h = [Size.Fixed(1.0), Size.Scaled(1.), Size.Fixed(.2)]
v = [Size.Fixed(0.7), Size.Scaled(1.), Size.Fixed(.5)]

# 创建一个新的 Divider 对象来划分画布区域
divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# 矩形的宽度和高度被忽略

# 在画布上添加一个 Axes 对象，使用刚刚创建的 Divider 对象定义位置
ax = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))

# 在 Axes 上绘制一个简单的线图
ax.plot([1, 2, 3])

# 显示图形
plt.show()
```