# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\custom_ticker1.py`

```
"""
=============
Custom Ticker
=============

The :mod:`matplotlib.ticker` module defines many preset tickers, but was
primarily designed for extensibility, i.e., to support user customized ticking.

In this example, a user defined function is used to format the ticks in
millions of dollars on the y-axis.
"""

import matplotlib.pyplot as plt

# 定义一个函数，用于格式化 y 轴刻度显示为以百万美元计的金额
def millions(x, pos):
    """The two arguments are the value and tick position."""
    return f'${x*1e-6:1.1f}M'

# 创建一个包含图形和轴对象的图表
fig, ax = plt.subplots()

# 设置 y 轴的主刻度格式化器为 millions 函数，将刻度转换为百万美元显示
ax.yaxis.set_major_formatter(millions)

# 定义一些数据
money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]

# 创建一个柱状图，展示 'Bill', 'Fred', 'Mary', 'Sue' 这四个人的金额数据
ax.bar(['Bill', 'Fred', 'Mary', 'Sue'], money)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axis.Axis.set_major_formatter`
```