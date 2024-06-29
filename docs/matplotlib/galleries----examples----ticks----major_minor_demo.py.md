# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\major_minor_demo.py`

```py
r"""
=====================
Major and minor ticks
=====================

Demonstrate how to use major and minor tickers.

The two relevant classes are `.Locator`\s and `.Formatter`\s.  Locators
determine where the ticks are, and formatters control the formatting of tick
labels.

Minor ticks are off by default (using `.NullLocator` and `.NullFormatter`).
Minor ticks can be turned on without labels by setting the minor locator.
Minor tick labels can be turned on by setting the minor formatter.

`.MultipleLocator` places ticks on multiples of some base.
`.StrMethodFormatter` uses a format string (e.g., ``'{x:d}'`` or ``'{x:1.2f}'``
or ``'{x:1.1f} cm'``) to format the tick labels (the variable in the format
string must be ``'x'``).  For a `.StrMethodFormatter`, the string can be passed
directly to `.Axis.set_major_formatter` or
`.Axis.set_minor_formatter`.  An appropriate `.StrMethodFormatter` will
be created and used automatically.

`.pyplot.grid` changes the grid settings of the major ticks of the x- and
y-axis together.  If you want to control the grid of the minor ticks for a
given axis, use for example ::

  ax.xaxis.grid(True, which='minor')

Note that a given locator or formatter instance can only be used on a single
axis (because the locator stores references to the axis data and view limits).
"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块
import numpy as np  # 导入numpy模块

from matplotlib.ticker import AutoMinorLocator, MultipleLocator  # 从matplotlib.ticker模块导入AutoMinorLocator和MultipleLocator类

t = np.arange(0.0, 100.0, 0.1)  # 创建一个从0到100的数组t，步长为0.1
s = np.sin(0.1 * np.pi * t) * np.exp(-t * 0.01)  # 计算sin函数乘以指数函数的值并赋给数组s

fig, ax = plt.subplots()  # 创建一个图形窗口和一个子图对象ax
ax.plot(t, s)  # 在ax上绘制t和s的图像

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '.0f' formatting but don't label
# minor ticks.  The string is used directly, the `StrMethodFormatter` is
# created automatically.
ax.xaxis.set_major_locator(MultipleLocator(20))  # 设置x轴的主要刻度定位器为20的倍数
ax.xaxis.set_major_formatter('{x:.0f}')  # 设置x轴的主要刻度格式为不带小数点的整数格式

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(5))  # 设置x轴的次要刻度定位器为5的倍数

plt.show()  # 显示图形

# %%
# Automatic tick selection for major and minor ticks.
#
# Use interactive pan and zoom to see how the tick intervals change. There will
# be either 4 or 5 minor tick intervals per major interval, depending on the
# major interval.
#
# One can supply an argument to `.AutoMinorLocator` to specify a fixed number
# of minor intervals per major interval, e.g. ``AutoMinorLocator(2)`` would
# lead to a single minor tick between major ticks.

t = np.arange(0.0, 100.0, 0.01)  # 创建一个从0到100的数组t，步长为0.01
s = np.sin(2 * np.pi * t) * np.exp(-t * 0.01)  # 计算sin函数乘以指数函数的值并赋给数组s

fig, ax = plt.subplots()  # 创建一个图形窗口和一个子图对象ax
ax.plot(t, s)  # 在ax上绘制t和s的图像

ax.xaxis.set_minor_locator(AutoMinorLocator())  # 设置x轴的次要刻度定位器为自动调整的次要定位器

ax.tick_params(which='both', width=2)  # 设置所有刻度的宽度为2
ax.tick_params(which='major', length=7)  # 设置主要刻度的长度为7
ax.tick_params(which='minor', length=4, color='r')  # 设置次要刻度的长度为4，颜色为红色

plt.show()  # 显示图形
# 设置 X 轴主要刻度的格式化器
# 参数 `formatter` 是用于格式化主要刻度的对象或函数
ax.xaxis.set_major_formatter()

# 设置 X 轴主要刻度的定位器
# 参数 `locator` 是确定主要刻度位置的对象
ax.xaxis.set_major_locator()

# 设置 X 轴次要刻度的定位器
# 参数 `locator` 是确定次要刻度位置的对象
ax.xaxis.set_minor_locator()

# 自动次要刻度定位器，用于自动计算并设置次要刻度的位置
# 参数 `nbins` 控制次要刻度的数量
matplotlib.ticker.AutoMinorLocator()

# 多重刻度定位器，用于设置多个刻度位置
# 参数 `base` 是刻度的基数，`offset` 是刻度的偏移量
matplotlib.ticker.MultipleLocator()

# 字符串方法格式化器，用于按照指定的格式化字符串格式化刻度值
# 参数 `fmt` 是格式化的字符串模板
matplotlib.ticker.StrMethodFormatter()
```