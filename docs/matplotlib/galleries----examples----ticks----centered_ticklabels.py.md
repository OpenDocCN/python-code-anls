# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\centered_ticklabels.py`

```py
"""
==============================
Centering labels between ticks
==============================

Ticklabels are aligned relative to their associated tick. The alignment
'center', 'left', or 'right' can be controlled using the horizontal alignment
property::

    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

However, there is no direct way to center the labels between ticks. To fake
this behavior, one can place a label on the minor ticks in between the major
ticks, and hide the major tick labels and minor ticks.

Here is an example that labels the months, centered between the ticks.
"""

import matplotlib.pyplot as plt

import matplotlib.cbook as cbook  # 导入matplotlib的cbook模块，提供杂项实用功能
import matplotlib.dates as dates  # 导入matplotlib的dates模块，用于日期处理
import matplotlib.ticker as ticker  # 导入matplotlib的ticker模块，用于设置刻度标签格式

# Load some financial data; Google's stock price
r = cbook.get_sample_data('goog.npz')['price_data']
r = r[-250:]  # get the last 250 days

fig, ax = plt.subplots()  # 创建一个新的图形和一个子图，ax是Axes对象的实例

ax.plot(r["date"], r["adj_close"])  # 在子图上绘制Google股票收盘价随时间变化的折线图

ax.xaxis.set_major_locator(dates.MonthLocator())  # 设置x轴主刻度为月份
# 16 is a slight approximation since months differ in number of days.
ax.xaxis.set_minor_locator(dates.MonthLocator(bymonthday=16))  # 设置x轴次刻度为每个月的16号

ax.xaxis.set_major_formatter(ticker.NullFormatter())  # 主刻度不显示标签
ax.xaxis.set_minor_formatter(dates.DateFormatter('%b'))  # 次刻度显示月份的缩写形式作为标签

# Remove the tick lines
ax.tick_params(axis='x', which='minor', tick1On=False, tick2On=False)  # 隐藏x轴次刻度线和标签

# Align the minor tick label
for label in ax.get_xticklabels(minor=True):  # 遍历所有x轴次刻度的标签
    label.set_horizontalalignment('center')  # 设置次刻度标签的水平对齐方式为居中

imid = len(r) // 2  # 计算数据长度的一半作为中间位置的索引
ax.set_xlabel(str(r["date"][imid].item().year))  # 设置x轴的标签为数据中间日期的年份

plt.show()  # 显示图形
```