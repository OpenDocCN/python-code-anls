# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\date_concise_formatter.py`

```py
"""
.. _date_concise_formatter:

================================================
Formatting date ticks using ConciseDateFormatter
================================================

Finding good tick values and formatting the ticks for an axis that
has date data is often a challenge.  `~.dates.ConciseDateFormatter` is
meant to improve the strings chosen for the ticklabels, and to minimize
the strings used in those tick labels as much as possible.

.. note::

    This formatter is a candidate to become the default date tick formatter
    in future versions of Matplotlib.  Please report any issues or
    suggestions for improvement to the GitHub repository or mailing list.

"""
import datetime  # 导入 datetime 模块

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 模块

import matplotlib.dates as mdates  # 导入 matplotlib 的 dates 模块

# %%
# First, the default formatter.

base = datetime.datetime(2005, 2, 1)  # 创建一个基准时间
dates = [base + datetime.timedelta(hours=(2 * i)) for i in range(732)]  # 创建包含732个日期的列表
N = len(dates)  # 获取日期列表的长度
np.random.seed(19680801)  # 设置随机数种子
y = np.cumsum(np.random.randn(N))  # 生成累积随机数的数组

fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(6, 6))  # 创建包含3个子图的图形对象
lims = [(np.datetime64('2005-02'), np.datetime64('2005-04')),  # 设定子图的 x 轴范围
        (np.datetime64('2005-02-03'), np.datetime64('2005-02-15')),
        (np.datetime64('2005-02-03 11:00'), np.datetime64('2005-02-04 13:20'))]
for nn, ax in enumerate(axs):  # 遍历子图列表
    ax.plot(dates, y)  # 在当前子图中绘制日期与随机数的折线图
    ax.set_xlim(lims[nn])  # 设置当前子图的 x 轴限制
    # rotate_labels...
    for label in ax.get_xticklabels():  # 遍历当前子图的 x 轴刻度标签
        label.set_rotation(40)  # 设置刻度标签文字旋转角度为40度
        label.set_horizontalalignment('right')  # 设置刻度标签水平对齐方式为右对齐
axs[0].set_title('Default Date Formatter')  # 设置第一个子图的标题
plt.show()

# %%
# The default date formatter is quite verbose, so we have the option of
# using `~.dates.ConciseDateFormatter`, as shown below.  Note that
# for this example the labels do not need to be rotated as they do for the
# default formatter because the labels are as small as possible.

fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(6, 6))  # 创建包含3个子图的图形对象
for nn, ax in enumerate(axs):  # 遍历子图列表
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)  # 创建自动日期定位器
    formatter = mdates.ConciseDateFormatter(locator)  # 创建简洁日期格式化器
    ax.xaxis.set_major_locator(locator)  # 设置 x 轴主刻度定位器
    ax.xaxis.set_major_formatter(formatter)  # 设置 x 轴主刻度格式化器

    ax.plot(dates, y)  # 在当前子图中绘制日期与随机数的折线图
    ax.set_xlim(lims[nn])  # 设置当前子图的 x 轴限制
axs[0].set_title('Concise Date Formatter')  # 设置第一个子图的标题

plt.show()

# %%
# If all calls to axes that have dates are to be made using this converter,
# it is probably most convenient to use the units registry where you do
# imports:

import matplotlib.units as munits  # 导入 matplotlib 的 units 模块

converter = mdates.ConciseDateConverter()  # 创建简洁日期转换器对象
munits.registry[np.datetime64] = converter  # 将 numpy datetime64 类型注册到简洁日期转换器
munits.registry[datetime.date] = converter  # 将 datetime.date 类型注册到简洁日期转换器
munits.registry[datetime.datetime] = converter  # 将 datetime.datetime 类型注册到简洁日期转换器

fig, axs = plt.subplots(3, 1, figsize=(6, 6), layout='constrained')  # 创建包含3个子图的图形对象
for nn, ax in enumerate(axs):  # 遍历子图列表
    ax.plot(dates, y)  # 在当前子图中绘制日期与随机数的折线图
    ax.set_xlim(lims[nn])  # 设置当前子图的 x 轴限制
axs[0].set_title('Concise Date Formatter')  # 设置第一个子图的标题

plt.show()

# %%
# Localization of date formats
# ============================
#
# Dates formats can be localized if the default formats are not desirable by
# manipulating one of three lists of strings.
#
# The ``formatter.formats`` list of formats is for the normal tick labels,
# There are six levels: years, months, days, hours, minutes, seconds.
# The ``formatter.offset_formats`` is how the "offset" string on the right
# of the axis is formatted.  This is usually much more verbose than the tick
# labels. Finally, the ``formatter.zero_formats`` are the formats of the
# ticks that are "zeros".  These are tick values that are either the first of
# the year, month, or day of month, or the zeroth hour, minute, or second.
# These are usually the same as the format of
# the ticks a level above.  For example if the axis limits mean the ticks are
# mostly days, then we label 1 Mar 2005 simply with a "Mar".  If the axis
# limits are mostly hours, we label Feb 4 00:00 as simply "Feb-4".
#
# Note that these format lists can also be passed to `.ConciseDateFormatter`
# as optional keyword arguments.

# Here we modify the labels to be "day month year", instead of the ISO
# "year month day":

fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(6, 6))

for nn, ax in enumerate(axs):
    # Automatically select the locator for dates
    locator = mdates.AutoDateLocator()
    # Create a concise date formatter using the locator
    formatter = mdates.ConciseDateFormatter(locator)
    # Set formats for different levels of ticks
    formatter.formats = ['%y',    # ticks are mostly years
                         '%b',    # ticks are mostly months
                         '%d',    # ticks are mostly days
                         '%H:%M', # ticks are mostly hours
                         '%H:%M', # ticks are mostly minutes
                         '%S.%f',] # ticks are mostly seconds
    # Set zero formats for "zero" ticks
    formatter.zero_formats = [''] + formatter.formats[:-1]
    # Adjust format for ticks that are mostly hours
    formatter.zero_formats[3] = '%d-%b'

    # Set offset formats for the offset string on the right of the axis
    formatter.offset_formats = ['',
                                '%Y',
                                '%b %Y',
                                '%d %b %Y',
                                '%d %b %Y',
                                '%d %b %Y %H:%M',]
    # Set major locator and formatter for the x-axis
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Plot the data on each subplot
    ax.plot(dates, y)
    # Set x-axis limits based on the current subplot
    ax.set_xlim(lims[nn])

# Set title for the first subplot
axs[0].set_title('Concise Date Formatter')

# Display the plot
plt.show()

# %%
# Registering a converter with localization
# =========================================
#
# `.ConciseDateFormatter` doesn't have rcParams entries, but localization can
# be accomplished by passing keyword arguments to `.ConciseDateConverter` and
# registering the datatypes you will use with the units registry:

import datetime

# Define formats for different levels of ticks
formats = ['%y',    # ticks are mostly years
           '%b',    # ticks are mostly months
           '%d',    # ticks are mostly days
           '%H:%M', # ticks are mostly hours
           '%H:%M', # ticks are mostly minutes
           '%S.%f',] # ticks are mostly seconds
# Define zero formats for "zero" ticks
zero_formats = [''] + formats[:-1]
# Adjust format for ticks that are mostly hours
zero_formats[3] = '%d-%b'
# 定义了一组日期偏移格式，用于不同级别的日期显示
offset_formats = ['',
                  '%Y',               # 年份格式
                  '%b %Y',           # 月份和年份格式
                  '%d %b %Y',        # 日、月、年格式
                  '%d %b %Y',        # 日、月、年格式（重复的格式，可能是误留或需求不同）
                  '%d %b %Y %H:%M', ]  # 日、月、年、小时、分钟格式

# 创建 ConciseDateConverter 对象，用于将日期转换为紧凑格式
converter = mdates.ConciseDateConverter(
    formats=formats, zero_formats=zero_formats, offset_formats=offset_formats)

# 将 np.datetime64 类型注册到 converter 对象，以便使用其日期转换功能
munits.registry[np.datetime64] = converter

# 将 datetime.date 类型注册到 converter 对象，以便使用其日期转换功能
munits.registry[datetime.date] = converter

# 将 datetime.datetime 类型注册到 converter 对象，以便使用其日期转换功能
munits.registry[datetime.datetime] = converter

# 创建一个包含三个子图的图形对象，布局方式为 constrained，尺寸为 6x6 英寸
fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(6, 6))

# 在每个子图上绘制日期和对应的数据 y
for nn, ax in enumerate(axs):
    ax.plot(dates, y)
    ax.set_xlim(lims[nn])  # 设置当前子图的 x 轴显示范围

# 设置第一个子图的标题
axs[0].set_title('Concise Date Formatter registered non-default')

# 显示图形
plt.show()
```