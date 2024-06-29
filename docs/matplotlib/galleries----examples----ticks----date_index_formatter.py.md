# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\date_index_formatter.py`

```
"""
=====================================
Custom tick formatter for time series
=====================================

.. redirect-from:: /gallery/text_labels_and_annotations/date_index_formatter
.. redirect-from:: /gallery/ticks/date_index_formatter2

When plotting daily data, e.g., financial time series, one often wants
to leave out days on which there is no data, for instance weekends, so that
the data are plotted at regular intervals without extra spaces for the days
with no data.
The example shows how to use an 'index formatter' to achieve the desired plot.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cbook as cbook
from matplotlib.dates import DateFormatter, DayLocator
import matplotlib.lines as ml
from matplotlib.ticker import Formatter

# 从 mpl-data/sample_data 目录加载一个包含日期、开盘价、最高价、最低价、收盘价、交易量和调整后收盘价字段的结构化 numpy 数组。
# 数据记录中，日期列以 np.datetime64 类型存储，单位为天 ('D')。
r = cbook.get_sample_data('goog.npz')['price_data']
r = r[:9]  # 获取前 9 天的数据

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 6), layout='constrained')
fig.get_layout_engine().set(hspace=0.15)

# 默认方式绘制，周末会留出间隙
ax1.plot(r["date"], r["adj_close"], 'o-')

# 标出每日数据的间隙
gaps = np.flatnonzero(np.diff(r["date"]) > np.timedelta64(1, 'D'))
for gap in r[['date', 'adj_close']][np.stack((gaps, gaps + 1)).T]:
    ax1.plot(gap['date'], gap['adj_close'], 'w--', lw=2)
ax1.legend(handles=[ml.Line2D([], [], ls='--', label='Gaps in daily data')])

ax1.set_title("Plot y at x Coordinates")
ax1.xaxis.set_major_locator(DayLocator())
ax1.xaxis.set_major_formatter(DateFormatter('%a'))


# 编写自定义索引格式化函数。在下面的绘图中，我们将数据绘制在从 0 开始的索引上，而不是日期。
def format_date(x, _):
    try:
        # 将 datetime64 转换为 datetime，并使用 datetime 的 strftime 方法格式化日期
        return r["date"][round(x)].item().strftime('%a')
    except IndexError:
        pass

# 创建一个索引坐标绘图（如果省略 x，默认为 range(len(y))）
ax2.plot(r["adj_close"], 'o-')

ax2.set_title("Plot y at Index Coordinates Using Custom Formatter")
ax2.xaxis.set_major_formatter(format_date)  # 内部创建 FuncFormatter

# %%
# 不仅可以传递函数给 `.Axis.set_major_formatter`，还可以使用任何可调用对象，例如实现了 __call__ 方法的类实例：

class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%a'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        """返回位置 pos 处时间 x 的标签。"""
        try:
            return self.dates[round(x)].item().strftime(self.fmt)
        except IndexError:
            pass
# 设置 x 轴主要刻度的格式化器，使用自定义的日期格式 '%a'
ax2.xaxis.set_major_formatter(MyFormatter(r["date"], '%a'))

# 显示绘图结果
plt.show()
```