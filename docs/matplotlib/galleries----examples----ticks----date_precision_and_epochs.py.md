# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\date_precision_and_epochs.py`

```py
"""
=========================
Date Precision and Epochs
=========================

Matplotlib can handle `.datetime` objects and `numpy.datetime64` objects using
a unit converter that recognizes these dates and converts them to floating
point numbers.

Before Matplotlib 3.3, the default for this conversion returns a float that was
days since "0000-12-31T00:00:00".  As of Matplotlib 3.3, the default is
days from "1970-01-01T00:00:00".  This allows more resolution for modern
dates.  "2020-01-01" with the old epoch converted to 730120, and a 64-bit
floating point number has a resolution of 2^{-52}, or approximately
14 microseconds, so microsecond precision was lost.  With the new default
epoch "2020-01-01" is 10957.0, so the achievable resolution is 0.21
microseconds.

"""
import datetime  # 导入 datetime 模块

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块
import numpy as np  # 导入 numpy 模块

import matplotlib.dates as mdates  # 导入 matplotlib.dates 模块


def _reset_epoch_for_tutorial():
    """
    Users (and downstream libraries) should not use the private method of
    resetting the epoch.
    """
    mdates._reset_epoch_test_example()  # 调用私有方法 _reset_epoch_test_example


# %%
# Datetime
# --------
#
# Python `.datetime` objects have microsecond resolution, so with the
# old default matplotlib dates could not round-trip full-resolution datetime
# objects.

old_epoch = '0000-12-31T00:00:00'  # 定义旧的 epoch 时间为 "0000-12-31T00:00:00"
new_epoch = '1970-01-01T00:00:00'  # 定义新的 epoch 时间为 "1970-01-01T00:00:00"

_reset_epoch_for_tutorial()  # 调用函数 _reset_epoch_for_tutorial，用于本教程，不建议实际使用
mdates.set_epoch(old_epoch)  # 设置 epoch 为旧的 epoch（MPL 3.3 之前）

date1 = datetime.datetime(2000, 1, 1, 0, 10, 0, 12,
                          tzinfo=datetime.timezone.utc)  # 创建一个 datetime 对象 date1
mdate1 = mdates.date2num(date1)  # 将 datetime 对象转换为 Matplotlib 的日期格式
print('Before Roundtrip: ', date1, 'Matplotlib date:', mdate1)
date2 = mdates.num2date(mdate1)  # 将 Matplotlib 的日期格式转换回 datetime 对象
print('After Roundtrip:  ', date2)

# %%
# Note this is only a round-off error, and there is no problem for
# dates closer to the old epoch:

date1 = datetime.datetime(10, 1, 1, 0, 10, 0, 12,
                          tzinfo=datetime.timezone.utc)  # 创建另一个 datetime 对象 date1
mdate1 = mdates.date2num(date1)  # 将 datetime 对象转换为 Matplotlib 的日期格式
print('Before Roundtrip: ', date1, 'Matplotlib date:', mdate1)
date2 = mdates.num2date(mdate1)  # 将 Matplotlib 的日期格式转换回 datetime 对象
print('After Roundtrip:  ', date2)

# %%
# If a user wants to use modern dates at microsecond precision, they
# can change the epoch using `.set_epoch`.  However, the epoch has to be
# set before any date operations to prevent confusion between different
# epochs. Trying to change the epoch later will raise a `RuntimeError`.

try:
    mdates.set_epoch(new_epoch)  # 设置 epoch 为新的 epoch（MPL 3.3 默认）
except RuntimeError as e:
    print('RuntimeError:', str(e))

# %%
# For this tutorial, we reset the sentinel using a private method, but users
# should just set the epoch once, if at all.

_reset_epoch_for_tutorial()  # 仅供本教程使用，重置 epoch 的私有方法
mdates.set_epoch(new_epoch)  # 设置 epoch 为新的 epoch

date1 = datetime.datetime(2020, 1, 1, 0, 10, 0, 12,
                          tzinfo=datetime.timezone.utc)  # 创建另一个 datetime 对象 date1
mdate1 = mdates.date2num(date1)  # 将 datetime 对象转换为 Matplotlib 的日期格式
print('Before Roundtrip: ', date1, 'Matplotlib date:', mdate1)
date2 = mdates.num2date(mdate1)  # 将 Matplotlib 的日期格式转换回 datetime 对象
print('After Roundtrip:  ', date2)
# 打印出经过转换后的日期 date2

# %%
# datetime64
# ----------
#
# `numpy.datetime64` 对象具有微秒级精度，时间跨度比 `.datetime` 对象大得多。
# 然而，目前 Matplotlib 将时间仅转换为具有微秒分辨率的 datetime 对象，
# 并且年份仅覆盖从 0000 到 9999 年。

_reset_epoch_for_tutorial()  # 不要这样做。仅用于本教程。
mdates.set_epoch(new_epoch)

date1 = np.datetime64('2000-01-01T00:10:00.000012')
mdate1 = mdates.date2num(date1)
print('Before Roundtrip: ', date1, 'Matplotlib date:', mdate1)
# 将 `date1` 转换为 Matplotlib 的日期数字表示 `mdate1`
date2 = mdates.num2date(mdate1)
print('After Roundtrip:  ', date2)
# 将 `mdate1` 转换回日期对象 `date2`

# %%
# Plotting
# --------
#
# 这当然会影响绘图结果。在旧的默认时代下，
# 在内部的 ``date2num`` 转换期间，时间会被舍入，导致数据跳跃:

_reset_epoch_for_tutorial()  # 不要这样做。仅用于本教程。
mdates.set_epoch(old_epoch)

x = np.arange('2000-01-01T00:00:00.0', '2000-01-01T00:00:00.000100',
              dtype='datetime64[us]')
# 模拟使用旧时代进行绘图的情况
xold = np.array([mdates.num2date(mdates.date2num(d)) for d in x])
y = np.arange(0, len(x))

# 重置时代以便进行比较
_reset_epoch_for_tutorial()  # 不要这样做。仅用于本教程。
mdates.set_epoch(new_epoch)

fig, ax = plt.subplots(layout='constrained')
ax.plot(xold, y)
ax.set_title('Epoch: ' + mdates.get_epoch())
ax.xaxis.set_tick_params(rotation=40)
plt.show()

# %%
# 对于使用较新时代绘制的日期，绘图结果是平滑的:

fig, ax = plt.subplots(layout='constrained')
ax.plot(x, y)
ax.set_title('Epoch: ' + mdates.get_epoch())
ax.xaxis.set_tick_params(rotation=40)
plt.show()

_reset_epoch_for_tutorial()  # 不要这样做。仅用于本教程。

# %%
#
# .. admonition:: References
#
#    此示例中展示了以下函数、方法、类和模块的使用:
#
#    - `matplotlib.dates.num2date`
#    - `matplotlib.dates.date2num`
#    - `matplotlib.dates.set_epoch`
```