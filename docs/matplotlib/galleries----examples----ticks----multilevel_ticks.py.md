# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\multilevel_ticks.py`

```
"""
=========================
Multilevel (nested) ticks
=========================

Sometimes we want another level of tick labels on an axis, perhaps to indicate
a grouping of the ticks.

Matplotlib does not provide an automated way to do this, but it is relatively
straightforward to annotate below the main axis.

These examples use `.Axes.secondary_xaxis`, which is one approach. It has the
advantage that we can use Matplotlib Locators and Formatters on the axis that
does the grouping if we want.

This first example creates a secondary xaxis and manually adds the ticks and
labels using `.Axes.set_xticks`.  Note that the tick labels have a newline
(e.g. ``"\nOughts"``) at the beginning of them to put the second-level tick
labels below the main tick labels.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates

rng = np.random.default_rng(19680801)

fig, ax = plt.subplots(layout='constrained', figsize=(4, 4))

ax.plot(np.arange(30))

# 创建一个次级 x 轴，并使用 `.Axes.set_xticks` 手动添加刻度和标签
sec = ax.secondary_xaxis(location=0)
sec.set_xticks([5, 15, 25], labels=['\nOughts', '\nTeens', '\nTwenties'])

# %%
# This second example adds a second level of annotation to a categorical axis.
# Here we need to note that each animal (category) is assigned an integer, so
# ``cats`` is at x=0, ``dogs`` at x=1 etc.  Then we place the ticks on the
# second level on an x that is at the middle of the animal class we are trying
# to delineate.
#
# This example also adds tick marks between the classes by adding a second
# secondary xaxis, and placing long, wide ticks at the boundaries between the
# animal classes.

fig, ax = plt.subplots(layout='constrained', figsize=(7, 4))

ax.plot(['cats', 'dogs', 'pigs', 'snakes', 'lizards', 'chickens',
         'eagles', 'herons', 'buzzards'],
        rng.normal(size=9), 'o')

# 标记各类动物：
sec = ax.secondary_xaxis(location=0)
sec.set_xticks([1, 3.5, 6.5], labels=['\n\nMammals', '\n\nReptiles', '\n\nBirds'])
sec.tick_params('x', length=0)

# 在类别之间添加刻度线：
sec2 = ax.secondary_xaxis(location=0)
sec2.set_xticks([-0.5, 2.5, 4.5, 8.5], labels=[])
sec2.tick_params('x', length=40, width=1.5)
ax.set_xlim(-0.6, 8.6)

# %%
# Dates are another common place where we may want to have a second level of
# tick labels.  In this last example, we take advantage of the ability to add
# an automatic locator and formatter to the secondary xaxis, which means we do
# not need to set the ticks manually.
#
# This example also differs from the above, in that we placed it at a location
# below the main axes ``location=-0.075`` and then we hide the spine by setting
# the line width to zero.  That means that our formatter no longer needs the
# carriage returns of the previous two examples.

fig, ax = plt.subplots(layout='constrained', figsize=(7, 4))

time = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-03-31'),
                 np.timedelta64(1, 'D'))

ax.plot(time, rng.random(size=len(time)))

# 仅格式化日期：
# 设置主要 x 轴的日期格式为 '%d'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

# 创建次要 x 轴，位置设置为左侧偏移 -0.075
sec = ax.secondary_xaxis(location=-0.075)

# 设置次要 x 轴的主要定位器为每月第一天，并标记月份
sec.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))

# 注意标签中额外的空格，用于将月份标签对齐到月份内部。
# 可以通过修改上面的 ``bymonthday`` 参数来实现同样的效果。
sec.xaxis.set_major_formatter(mdates.DateFormatter('  %b'))

# 设置次要 x 轴的刻度长度为0，使得刻度不可见
sec.tick_params('x', length=0)

# 设置次要 x 轴底部的边框线宽度为0，隐藏底部的边框线
sec.spines['bottom'].set_linewidth(0)

# 设置次要 x 轴的标签
sec.set_xlabel('Dates (2020)')

# 显示绘图
plt.show()
```