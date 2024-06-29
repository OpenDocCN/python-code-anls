# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\date_demo_rrule.py`

```
"""
=========================================
Placing date ticks using recurrence rules
=========================================

The `iCalender RFC`_ specifies *recurrence rules* (rrules), that define
date sequences. You can use rrules in Matplotlib to place date ticks.

This example sets custom date ticks on every 5th easter.

See https://dateutil.readthedocs.io/en/stable/rrule.html for help with rrules.

.. _iCalender RFC: https://tools.ietf.org/html/rfc5545
"""
# 导入所需的库
import datetime  # 导入处理日期和时间的标准库
import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

# 导入用于处理日期的 Matplotlib 模块
from matplotlib.dates import (YEARLY, DateFormatter, RRuleLocator, drange,
                              rrulewrapper)

# 设置随机种子，确保随机数据可重复生成
np.random.seed(19680801)

# 每5个复活节为间隔的日期标记
rule = rrulewrapper(YEARLY, byeaster=1, interval=5)  # 使用复活节作为标记的间隔规则
loc = RRuleLocator(rule)  # 创建一个基于规则的日期定位器
formatter = DateFormatter('%m/%d/%y')  # 创建日期格式化器，以月/日/年格式显示日期

date1 = datetime.date(1952, 1, 1)  # 开始日期
date2 = datetime.date(2004, 4, 12)  # 结束日期
delta = datetime.timedelta(days=100)  # 时间间隔设置为100天

dates = drange(date1, date2, delta)  # 生成日期范围数组
s = np.random.rand(len(dates))  # 生成与日期数组长度相同的随机数作为 y 值

# 创建图形和轴对象
fig, ax = plt.subplots()
plt.plot(dates, s, 'o')  # 在图中绘制日期和随机数关系的散点图
ax.xaxis.set_major_locator(loc)  # 设置主要刻度的定位器为自定义的日期定位器
ax.xaxis.set_major_formatter(formatter)  # 设置主要刻度的格式化器为自定义的日期格式化器
ax.xaxis.set_tick_params(rotation=30, labelsize=10)  # 设置 x 轴刻度标签的旋转角度和字体大小

plt.show()  # 显示图形
```