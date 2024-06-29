# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\date.py`

```
"""
================
Date tick labels
================

Matplotlib date plotting is done by converting date instances into
days since an epoch (by default 1970-01-01T00:00:00). The
:mod:`matplotlib.dates` module provides the converter functions `.date2num`
and `.num2date` that convert `datetime.datetime` and `numpy.datetime64`
objects to and from Matplotlib's internal representation.  These data
types are registered with the unit conversion mechanism described in
:mod:`matplotlib.units`, so the conversion happens automatically for the user.
The registration process also sets the default tick ``locator`` and
``formatter`` for the axis to be `~.matplotlib.dates.AutoDateLocator` and
`~.matplotlib.dates.AutoDateFormatter`.

An alternative formatter is the `~.dates.ConciseDateFormatter`,
used in the second ``Axes`` below (see
:doc:`/gallery/ticks/date_concise_formatter`), which often removes the need to
rotate the tick labels. The last ``Axes`` formats the dates manually, using
`~.dates.DateFormatter` to format the dates using the format strings documented
at `datetime.date.strftime`.
"""

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块

import matplotlib.cbook as cbook  # 导入matplotlib.cbook模块
import matplotlib.dates as mdates  # 导入matplotlib.dates模块

# 从mpl-data/sample_data目录中的yahoo csv数据加载一个numpy记录数组，包含字段date, open, high,
# low, close, volume, adj_close。记录数组将日期存储为np.datetime64，单位为天（'D'）。
data = cbook.get_sample_data('goog.npz')['price_data']

fig, axs = plt.subplots(3, 1, figsize=(6.4, 7), layout='constrained')  # 创建一个包含3个子图的图形对象
# 共同的设置适用于所有三个子图:
for ax in axs:
    ax.plot('date', 'adj_close', data=data)  # 在每个子图上绘制日期和调整后的收盘价
    # 主要刻度每半年显示一次，次要刻度每个月显示一次
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True)  # 打开网格
    ax.set_ylabel(r'Price [\$]')  # 设置y轴标签为价格（美元）

# 不同的格式设置:
ax = axs[0]
ax.set_title('DefaultFormatter', loc='left', y=0.85, x=0.02, fontsize='medium')  # 设置标题和位置

ax = axs[1]
ax.set_title('ConciseFormatter', loc='left', y=0.85, x=0.02, fontsize='medium')  # 设置标题和位置
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))  # 使用ConciseDateFormatter格式化主要刻度

ax = axs[2]
ax.set_title('Manual DateFormatter', loc='left', y=0.85, x=0.02,
             fontsize='medium')  # 设置标题和位置
# x轴上的文本将以'YYYY-mm'格式显示
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
# 旋转并右对齐x标签，以防它们互相挤在一起
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')

plt.show()  # 显示图形
```