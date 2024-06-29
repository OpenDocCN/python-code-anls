# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\date_demo_convert.py`

```
"""
=================
Date Demo Convert
=================

"""
# 导入必要的模块
import datetime  # 导入日期时间处理模块

import matplotlib.pyplot as plt  # 导入绘图模块
import numpy as np  # 导入数值计算模块

from matplotlib.dates import DateFormatter, DayLocator, HourLocator, drange  # 从matplotlib.dates模块导入日期格式化、日期定位器和日期范围函数

# 创建起始日期和结束日期
date1 = datetime.datetime(2000, 3, 2)  # 起始日期时间对象
date2 = datetime.datetime(2000, 3, 6)  # 结束日期时间对象

delta = datetime.timedelta(hours=6)  # 定义时间增量为6小时
dates = drange(date1, date2, delta)  # 使用drange生成日期范围

y = np.arange(len(dates))  # 生成与日期数量相同的一维数组作为y轴数据

fig, ax = plt.subplots()  # 创建图形和子图对象
ax.plot(dates, y**2, 'o')  # 在子图中绘制日期与y轴数据的平方关系散点图

# 自动缩放应该能正确完成，但是可以使用date2num和num2date在日期与浮点数之间进行转换，
# 如果需要的话；date2num和num2date可以将日期实例或日期序列进行转换
ax.set_xlim(dates[0], dates[-1])  # 设置x轴的范围为起始日期和结束日期

# HourLocator接受小时或小时序列作为刻度，而不是基础倍数
ax.xaxis.set_major_locator(DayLocator())  # 设置主刻度为每天
ax.xaxis.set_minor_locator(HourLocator(range(0, 25, 6)))  # 设置次刻度为每6小时
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  # 设置主刻度的日期格式为年-月-日

ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')  # 设置数据点悬停格式为年-月-日 时:分:秒
fig.autofmt_xdate()  # 自动调整日期标签的显示以避免重叠

plt.show()  # 显示绘制的图形
```