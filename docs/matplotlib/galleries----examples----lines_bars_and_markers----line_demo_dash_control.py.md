# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\line_demo_dash_control.py`

```
"""
==============================
Customizing dashed line styles
==============================

The dashing of a line is controlled via a dash sequence. It can be modified
using `.Line2D.set_dashes`.

The dash sequence is a series of on/off lengths in points, e.g.
``[3, 1]`` would be 3pt long lines separated by 1pt spaces.

Some functions like `.Axes.plot` support passing Line properties as keyword
arguments. In such a case, you can already set the dashing when creating the
line.

*Note*: The dash style can also be configured via a
:ref:`property_cycle <color_cycle>`
by passing a list of dash sequences using the keyword *dashes* to the
cycler. This is not shown within this example.

Other attributes of the dash may also be set either with the relevant method
(`~.Line2D.set_dash_capstyle`, `~.Line2D.set_dash_joinstyle`,
`~.Line2D.set_gapcolor`) or by passing the property through a plotting
function.
"""
# 引入 matplotlib 和 numpy 库
import matplotlib.pyplot as plt
import numpy as np

# 生成 x 值数组
x = np.linspace(0, 10, 500)
# 生成对应的 sin(x) 值数组
y = np.sin(x)

# 设置全局线条宽度为 2.5
plt.rc('lines', linewidth=2.5)
# 创建一个新的图形和轴对象
fig, ax = plt.subplots()

# 使用 set_dashes() 和 set_dash_capstyle() 修改现有线条的虚线样式
line1, = ax.plot(x, y, label='Using set_dashes() and set_dash_capstyle()')
line1.set_dashes([2, 2, 10, 2])  # 设置虚线样式为：2pt 实线，2pt 空白，10pt 实线，2pt 空白
line1.set_dash_capstyle('round')  # 设置虚线端点为圆形

# 使用 plot(..., dashes=...) 在创建线条时设置虚线样式
line2, = ax.plot(x, y - 0.2, dashes=[6, 2], label='Using the dashes parameter')

# 使用 plot(..., dashes=..., gapcolor=...) 在创建线条时设置虚线和间隔颜色
line3, = ax.plot(x, y - 0.4, dashes=[4, 4], gapcolor='tab:pink',
                 label='Using the dashes and gapcolor parameters')

# 添加图例，并设置图例句柄长度为 4
ax.legend(handlelength=4)

# 显示图形
plt.show()
```