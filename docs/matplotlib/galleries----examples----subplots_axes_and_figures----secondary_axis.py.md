# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\secondary_axis.py`

```
"""
==============
Secondary Axis
==============

Sometimes we want a secondary axis on a plot, for instance to convert
radians to degrees on the same plot.  We can do this by making a child
axes with only one axis visible via `.axes.Axes.secondary_xaxis` and
`.axes.Axes.secondary_yaxis`.  This secondary axis can have a different scale
than the main axis by providing both a forward and an inverse conversion
function in a tuple to the *functions* keyword argument:
"""

# 引入需要的库
import datetime  # 导入日期时间模块

import matplotlib.pyplot as plt  # 导入绘图模块
import numpy as np  # 导入数值计算模块

import matplotlib.dates as mdates  # 导入日期格式化模块
from matplotlib.ticker import AutoMinorLocator  # 导入刻度自动设定模块

# 创建一个包含受限布局的图形和主要坐标轴
fig, ax = plt.subplots(layout='constrained')
x = np.arange(0, 360, 1)  # 创建一个角度数组
y = np.sin(2 * x * np.pi / 180)  # 计算对应的正弦值
ax.plot(x, y)  # 绘制正弦波形
ax.set_xlabel('angle [degrees]')  # 设置x轴标签为角度[度]
ax.set_ylabel('signal')  # 设置y轴标签为信号
ax.set_title('Sine wave')  # 设置图表标题为正弦波形


def deg2rad(x):
    return x * np.pi / 180  # 定义角度转弧度的函数


def rad2deg(x):
    return x * 180 / np.pi  # 定义弧度转角度的函数


secax = ax.secondary_xaxis('top', functions=(deg2rad, rad2deg))  # 创建顶部的次坐标轴，并指定转换函数
secax.set_xlabel('angle [rad]')  # 设置次坐标轴x轴标签为角度[弧度]
plt.show()

# %%
# By default, the secondary axis is drawn in the Axes coordinate space.
# We can also provide a custom transform to place it in a different
# coordinate space. Here we put the axis at Y = 0 in data coordinates.

fig, ax = plt.subplots(layout='constrained')
x = np.arange(0, 10)  # 创建一个从0到9的数组
np.random.seed(19680801)  # 设置随机数种子
y = np.random.randn(len(x))  # 生成随机数，并计算其正态分布
ax.plot(x, y)  # 绘制随机数据曲线
ax.set_xlabel('X')  # 设置x轴标签为X
ax.set_ylabel('Y')  # 设置y轴标签为Y
ax.set_title('Random data')  # 设置图表标题为随机数据

# 使用ax.transData作为转换，将次坐标轴放置在数据坐标中Y = 0的位置
secax = ax.secondary_xaxis(0, transform=ax.transData)
secax.set_xlabel('Axis at Y = 0')  # 设置次坐标轴x轴标签为Y = 0
plt.show()

# %%
# Here is the case of converting from wavenumber to wavelength in a
# log-log scale.
#
# .. note::
#
#   In this case, the xscale of the parent is logarithmic, so the child is
#   made logarithmic as well.

fig, ax = plt.subplots(layout='constrained')
x = np.arange(0.02, 1, 0.02)  # 创建一个从0.02到0.98，步长为0.02的数组
np.random.seed(19680801)  # 设置随机数种子
y = np.random.randn(len(x)) ** 2  # 生成随机数并计算其平方
ax.loglog(x, y)  # 绘制对数-对数坐标系下的数据曲线
ax.set_xlabel('f [Hz]')  # 设置x轴标签为频率[Hz]
ax.set_ylabel('PSD')  # 设置y轴标签为PSD
ax.set_title('Random spectrum')  # 设置图表标题为随机频谱


def one_over(x):
    """向量化的1/x函数，手动处理x==0的情况"""
    x = np.array(x, float)  # 将输入转换为浮点数数组
    near_zero = np.isclose(x, 0)  # 找出接近零的元素
    x[near_zero] = np.inf  # 将接近零的元素设置为无穷大
    x[~near_zero] = 1 / x[~near_zero]  # 对非零元素取倒数
    return x


# 函数"1/x"的反函数是其自身
inverse = one_over

# 创建顶部的次坐标轴，并指定转换函数
secax = ax.secondary_xaxis('top', functions=(one_over, inverse))
secax.set_xlabel('period [s]')  # 设置次坐标轴x轴标签为周期[s]
plt.show()

# %%
# Sometime we want to relate the axes in a transform that is ad-hoc from
# the data, and is derived empirically.  In that case we can set the
# forward and inverse transforms functions to be linear interpolations from the
# one data set to the other.
#
# .. note::
#
#   In order to properly handle the data margins, the mapping functions
#   (``forward`` and ``inverse`` in this example) need to be defined beyond the
#   nominal plot limits.
#
#   In the specific case of the numpy linear interpolation, `numpy.interp`,
# 创建一个包含有限制条件的子图对象
fig, ax = plt.subplots(layout='constrained')

# 生成一组 x 值和对应的随机 y 值
xdata = np.arange(1, 11, 0.4)
ydata = np.random.randn(len(xdata))

# 在子图中绘制 xdata 对应的 ydata 数据，并添加标签
ax.plot(xdata, ydata, label='Plotted data')

# 创建一个人工的 xold 数据集，它与另一个基于数据推导的坐标相关联。
# xnew 必须是单调递增的，因此我们对其进行排序…
xold = np.arange(0, 11, 0.2)
xnew = np.sort(10 * np.exp(-xold / 4) + np.random.randn(len(xold)) / 3)

# 在子图中绘制 xold[3:] 对应的 xnew[3:] 数据，并添加标签
ax.plot(xold[3:], xnew[3:], label='Transform data')

# 设置 x 轴标签
ax.set_xlabel('X [m]')

# 添加图例
ax.legend()

# 定义一个函数，用于将 xold 数据映射到 xnew 数据
def forward(x):
    return np.interp(x, xold, xnew)

# 定义一个函数，用于将 xnew 数据映射回 xold 数据
def inverse(x):
    return np.interp(x, xnew, xold)

# 在子图上创建一个辅助 x 轴，使用 forward 和 inverse 函数映射
secax = ax.secondary_xaxis('top', functions=(forward, inverse))

# 设置辅助 x 轴的次要刻度定位器
secax.xaxis.set_minor_locator(AutoMinorLocator())

# 设置辅助 x 轴的标签
secax.set_xlabel('$X_{other}$')

# 显示图形
plt.show()

# %%
# 最后的例子将 np.datetime64 转换为年日并显示在 x 轴上，
# 将温度从摄氏度转换为华氏度并显示在 y 轴上。注意添加了第三个 y 轴，
# 可以使用浮点数指定其位置参数。

# 生成一组日期数据和对应的随机温度数据
dates = [datetime.datetime(2018, 1, 1) + datetime.timedelta(hours=k * 6)
         for k in range(240)]
temperature = np.random.randn(len(dates)) * 4 + 6.7

# 创建一个包含有限制条件的子图对象
fig, ax = plt.subplots(layout='constrained')

# 在子图中绘制日期和温度数据
ax.plot(dates, temperature)

# 设置 y 轴标签
ax.set_ylabel(r'$T\ [^oC]$')

# 设置 x 轴刻度的旋转角度
ax.xaxis.set_tick_params(rotation=70)

# 定义一个函数，将 matplotlib 的日期数值转换为自 2018-01-01 以来的天数
def date2yday(x):
    """Convert matplotlib datenum to days since 2018-01-01."""
    y = x - mdates.date2num(datetime.datetime(2018, 1, 1))
    return y

# 定义一个函数，将自 2018-01-01 以来的天数转换为 matplotlib 的日期数值
def yday2date(x):
    """Return a matplotlib datenum for *x* days after 2018-01-01."""
    y = x + mdates.date2num(datetime.datetime(2018, 1, 1))
    return y

# 在子图上创建一个辅助 x 轴，使用 date2yday 和 yday2date 函数映射
secax_x = ax.secondary_xaxis('top', functions=(date2yday, yday2date))

# 设置辅助 x 轴的标签
secax_x.set_xlabel('yday [2018]')

# 定义一个函数，将摄氏度转换为华氏度
def celsius_to_fahrenheit(x):
    return x * 1.8 + 32

# 定义一个函数，将华氏度转换为摄氏度
def fahrenheit_to_celsius(x):
    return (x - 32) / 1.8

# 在子图上创建一个辅助 y 轴，使用 celsius_to_fahrenheit 和 fahrenheit_to_celsius 函数映射
secax_y = ax.secondary_yaxis('right', functions=(celsius_to_fahrenheit, fahrenheit_to_celsius))

# 设置辅助 y 轴的标签
secax_y.set_ylabel(r'$T\ [^oF]$')

# 定义一个函数，将摄氏度转换为温度异常值（与平均温度的偏差）
def celsius_to_anomaly(x):
    return (x - np.mean(temperature))

# 定义一个函数，将温度异常值转换为摄氏度
def anomaly_to_celsius(x):
    return (x + np.mean(temperature))

# 在子图上创建一个辅助 y 轴，使用 celsius_to_anomaly 和 anomaly_to_celsius 函数映射，
# 并指定其位置参数为 1.2
secax_y2 = ax.secondary_yaxis(1.2, functions=(celsius_to_anomaly, anomaly_to_celsius))

# 设置辅助 y 轴的标签
secax_y2.set_ylabel(r'$T - \overline{T}\ [^oC]$')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.secondary_xaxis`
#    - `matplotlib.axes.Axes.secondary_yaxis`
```