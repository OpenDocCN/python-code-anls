# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\date_formatters_locators.py`

```
"""
.. _date_formatters_locators:

=================================
Date tick locators and formatters
=================================

This example illustrates the usage and effect of the various date locators and
formatters.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from matplotlib.dates import (FR, MO, MONTHLY, SA, SU, TH, TU, WE,  # 导入日期相关的类和函数
                              AutoDateFormatter, AutoDateLocator,
                              ConciseDateFormatter, DateFormatter, DayLocator,
                              HourLocator, MicrosecondLocator, MinuteLocator,
                              MonthLocator, RRuleLocator, SecondLocator,
                              WeekdayLocator, YearLocator, rrulewrapper)
import matplotlib.ticker as ticker  # 导入 matplotlib 的 ticker 模块，用于设置刻度


def plot_axis(ax, locator=None, xmax='2002-02-01', fmt=None, formatter=None):
    """Set up common parameters for the Axes in the example."""
    ax.spines[['left', 'right', 'top']].set_visible(False)  # 设置图的左侧、右侧、顶部边框不可见
    ax.yaxis.set_major_locator(ticker.NullLocator())  # 设置 y 轴主刻度定位器为空定位器，即无主刻度
    ax.tick_params(which='major', width=1.00, length=5)  # 设置主刻度的参数：线宽为1.00，长度为5
    ax.tick_params(which='minor', width=0.75, length=2.5)  # 设置次刻度的参数：线宽为0.75，长度为2.5
    ax.set_xlim(np.datetime64('2000-02-01'), np.datetime64(xmax))  # 设置 x 轴的显示范围从 '2000-02-01' 到指定的 xmax
    if locator:
        ax.xaxis.set_major_locator(eval(locator))  # 如果提供了定位器字符串，将其转换并设置为 x 轴的主定位器
        ax.xaxis.set_major_formatter(DateFormatter(fmt))  # 使用指定的日期格式化器 fmt 格式化 x 轴的主刻度
    else:
        ax.xaxis.set_major_formatter(eval(formatter))  # 否则，使用提供的格式化器字符串来格式化 x 轴的主刻度
    ax.text(0.0, 0.2, locator or formatter, transform=ax.transAxes,  # 在图上添加文本，显示当前使用的定位器或格式化器
            fontsize=14, fontname='Monospace', color='tab:blue')

# %%
# :ref:`date-locators`
# --------------------

locators = [
    # locator as str, xmax, fmt
    ('AutoDateLocator(maxticks=8)', '2003-02-01', '%Y-%m'),  # 自动日期定位器，最大刻度数为 8
    ('YearLocator(month=4)', '2003-02-01', '%Y-%m'),  # 年定位器，每年的四月为一个刻度
    ('MonthLocator(bymonth=[4, 8, 12])', '2003-02-01', '%Y-%m'),  # 按指定月份划分的月定位器
    ('DayLocator(interval=180)', '2003-02-01', '%Y-%m-%d'),  # 每隔 180 天一个刻度的日定位器
    ('WeekdayLocator(byweekday=SU, interval=4)', '2000-07-01', '%a %Y-%m-%d'),  # 每隔 4 周日一个刻度的工作日定位器
    ('HourLocator(byhour=range(0, 24, 6))', '2000-02-04', '%H h'),  # 每隔 6 小时一个刻度的小时定位器
    ('MinuteLocator(interval=15)', '2000-02-01 02:00', '%H:%M'),  # 每隔 15 分钟一个刻度的分钟定位器
    ('SecondLocator(bysecond=(0, 30))', '2000-02-01 00:02', '%H:%M:%S'),  # 每隔 30 秒一个刻度的秒定位器
    ('MicrosecondLocator(interval=1000)', '2000-02-01 00:00:00.005', '%S.%f'),  # 每隔 1000 微秒一个刻度的微秒定位器
    ('RRuleLocator(rrulewrapper(freq=MONTHLY, \nbyweekday=(MO, TU, WE, TH, FR), '
     'bysetpos=-1))', '2000-07-01', '%Y-%m-%d'),  # 使用指定规则生成刻度的规则定位器
]

fig, axs = plt.subplots(len(locators), 1, figsize=(8, len(locators) * .8),
                        layout='constrained')  # 创建多个子图，每个定位器对应一个子图
fig.suptitle('Date Locators')  # 设置图的总标题为 'Date Locators'
for ax, (locator, xmax, fmt) in zip(axs, locators):
    plot_axis(ax, locator, xmax, fmt)  # 在每个子图上绘制对应的定位器

# %%
# :ref:`date-formatters`
# ----------------------

formatters = [
    'AutoDateFormatter(ax.xaxis.get_major_locator())',  # 使用自动日期格式化器
    'ConciseDateFormatter(ax.xaxis.get_major_locator())',  # 使用简洁日期格式化器
    'DateFormatter("%b %Y")',  # 使用自定义日期格式化器，显示月份和年份
]

fig, axs = plt.subplots(len(formatters), 1, figsize=(8, len(formatters) * .8),
                        layout='constrained')  # 创建多个子图，每个格式化器对应一个子图
fig.suptitle('Date Formatters')  # 设置图的总标题为 'Date Formatters'
for ax, fmt in zip(axs, formatters):
    plot_axis(ax, formatter=fmt)  # 在每个子图上绘制对应的格式化器
    # 调用 plot_axis 函数，并传入 ax 参数作为参数，同时指定了 formatter=fmt 作为关键字参数
    plot_axis(ax, formatter=fmt)
# %%
#
# .. admonition:: References
#
#    This section lists various functions, methods, classes, and modules utilized in this example for handling date and time in matplotlib plotting:
#
#    - `matplotlib.dates.AutoDateLocator`: Automatically selects major and minor date tick locations.
#    - `matplotlib.dates.YearLocator`: Locates ticks at the beginning of each year.
#    - `matplotlib.dates.MonthLocator`: Locates ticks at the beginning of each month.
#    - `matplotlib.dates.DayLocator`: Locates ticks at the beginning of each day.
#    - `matplotlib.dates.WeekdayLocator`: Locates ticks at the beginning of each weekday.
#    - `matplotlib.dates.HourLocator`: Locates ticks at the beginning of each hour.
#    - `matplotlib.dates.MinuteLocator`: Locates ticks at the beginning of each minute.
#    - `matplotlib.dates.SecondLocator`: Locates ticks at the beginning of each second.
#    - `matplotlib.dates.MicrosecondLocator`: Locates ticks at the beginning of each microsecond.
#    - `matplotlib.dates.RRuleLocator`: Locates ticks according to a recurrence rule.
#    - `matplotlib.dates.rrulewrapper`: Provides a wrapper for working with recurrence rules.
#    - `matplotlib.dates.DateFormatter`: Formats tick labels as dates.
#    - `matplotlib.dates.AutoDateFormatter`: Automatically selects date formatting.
#    - `matplotlib.dates.ConciseDateFormatter`: Formats dates concisely for tick labels.
```