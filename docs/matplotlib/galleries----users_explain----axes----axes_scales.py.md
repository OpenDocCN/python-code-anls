# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\axes_scales.py`

```
"""
.. _user_axes_scales:

===========
Axis scales
===========

By default Matplotlib displays data on the axis using a linear scale.
Matplotlib also supports `logarithmic scales
<https://en.wikipedia.org/wiki/Logarithmic_scale>`_, and other less common
scales as well. Usually this can be done directly by using the
`~.axes.Axes.set_xscale` or `~.axes.Axes.set_yscale` methods.

"""
# 导入 matplotlib.pyplot 库并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并简称为 np
import numpy as np

# 导入 matplotlib 的 scale 模块，并命名为 mscale
import matplotlib.scale as mscale
# 从 matplotlib.ticker 中导入 FixedLocator 和 NullFormatter 类
from matplotlib.ticker import FixedLocator, NullFormatter

# 创建一个包含子图的图形对象，采用指定的子图布局
fig, axs = plt.subplot_mosaic([['linear', 'linear-log'],
                               ['log-linear', 'log-log']], layout='constrained')

# 生成一组数据
x = np.arange(0, 3*np.pi, 0.1)
y = 2 * np.sin(x) + 3

# 在 'linear' 子图上绘制 x 和 y 的图像
ax = axs['linear']
ax.plot(x, y)
ax.set_xlabel('linear')  # 设置 x 轴标签
ax.set_ylabel('linear')  # 设置 y 轴标签

# 在 'linear-log' 子图上绘制 x 和 y 的图像，并设置 y 轴为对数刻度
ax = axs['linear-log']
ax.plot(x, y)
ax.set_yscale('log')     # 设置 y 轴为对数刻度
ax.set_xlabel('linear')  # 设置 x 轴标签
ax.set_ylabel('log')     # 设置 y 轴标签

# 在 'log-linear' 子图上绘制 x 和 y 的图像，并设置 x 轴为对数刻度
ax = axs['log-linear']
ax.plot(x, y)
ax.set_xscale('log')     # 设置 x 轴为对数刻度
ax.set_xlabel('log')     # 设置 x 轴标签
ax.set_ylabel('linear')  # 设置 y 轴标签

# 在 'log-log' 子图上绘制 x 和 y 的图像，并设置 x、y 轴为对数刻度
ax = axs['log-log']
ax.plot(x, y)
ax.set_xscale('log')     # 设置 x 轴为对数刻度
ax.set_yscale('log')     # 设置 y 轴为对数刻度
ax.set_xlabel('log')     # 设置 x 轴标签
ax.set_ylabel('log')     # 设置 y 轴标签

# %%
# loglog and semilogx/y
# =====================
#
# The logarithmic axis is used so often that there are a set
# helper functions, that do the same thing: `~.axes.Axes.semilogy`,
# `~.axes.Axes.semilogx`, and `~.axes.Axes.loglog`.

# 创建新的图形对象和子图布局
fig, axs = plt.subplot_mosaic([['linear', 'linear-log'],
                               ['log-linear', 'log-log']], layout='constrained')

# 生成一组新的数据
x = np.arange(0, 3*np.pi, 0.1)
y = 2 * np.sin(x) + 3

# 在 'linear' 子图上绘制 x 和 y 的图像，并设置标题
ax = axs['linear']
ax.plot(x, y)
ax.set_xlabel('linear')  # 设置 x 轴标签
ax.set_ylabel('linear')  # 设置 y 轴标签
ax.set_title('plot(x, y)')  # 设置子图标题

# 在 'linear-log' 子图上绘制 x 和 y 的图像，y 轴使用对数刻度，并设置标题
ax = axs['linear-log']
ax.semilogy(x, y)
ax.set_xlabel('linear')  # 设置 x 轴标签
ax.set_ylabel('log')     # 设置 y 轴标签
ax.set_title('semilogy(x, y)')  # 设置子图标题

# 在 'log-linear' 子图上绘制 x 和 y 的图像，x 轴使用对数刻度，并设置标题
ax = axs['log-linear']
ax.semilogx(x, y)
ax.set_xlabel('log')     # 设置 x 轴标签
ax.set_ylabel('linear')  # 设置 y 轴标签
ax.set_title('semilogx(x, y)')  # 设置子图标题

# 在 'log-log' 子图上绘制 x 和 y 的图像，x、y 轴使用对数刻度，并设置标题
ax = axs['log-log']
ax.loglog(x, y)
ax.set_xlabel('log')     # 设置 x 轴标签
ax.set_ylabel('log')     # 设置 y 轴标签
ax.set_title('loglog(x, y)')  # 设置子图标题

# %%
# Other built-in scales
# =====================
#
# There are other scales that can be used.  The list of registered
# scales can be returned from `.scale.get_scale_names`:

# 创建新的图形对象和子图布局
fig, axs = plt.subplot_mosaic([['asinh', 'symlog'],
                               ['log', 'logit']], layout='constrained')

# 生成一组新的数据
x = np.arange(0, 1000)

# 遍历每个子图及其名称
for name, ax in axs.items():
    if name in ['asinh', 'symlog']:
        yy = x - np.mean(x)
    elif name in ['logit']:
        yy = (x-np.min(x))
        yy = yy / np.max(np.abs(yy))
    else:
        yy = x

    ax.plot(yy, yy)
    ax.set_yscale(name)   # 设置 y 轴的比例尺为当前名称
    ax.set_title(name)    # 设置子图标题

# %%
# Optional arguments for scales
# =============================
#
# Some of the default scales have optional arguments.  These are
# documented in the API reference for the respective scales at
# `~.matplotlib.scale`.  One can change the base of the logarithm
# being plotted (eg 2 below) or the linear threshold range
# 在 'symlog' 位置创建一个图形对象
fig, axs = plt.subplot_mosaic([['log', 'symlog']], layout='constrained',
                              figsize=(6.4, 3))

# 遍历 axs 中的每个子图对象
for name, ax in axs.items():
    # 如果子图名称是 'log'
    if name in ['log']:
        # 在当前子图 ax 上绘制 x vs x 的图像
        ax.plot(x, x)
        # 设置 y 轴的缩放为对数缩放，底数为2
        ax.set_yscale('log', base=2)
        # 设置子图标题为 'log base=2'
        ax.set_title('log base=2')
    else:
        # 如果子图名称不是 'log'
        # 在当前子图 ax 上绘制 x - np.mean(x) vs x - np.mean(x) 的图像
        ax.plot(x - np.mean(x), x - np.mean(x))
        # 设置 y 轴的缩放为对称对数刻度，线性阈值为100
        ax.set_yscale('symlog', linthresh=100)
        # 设置子图标题为 'symlog linthresh=100'
        ax.set_title('symlog linthresh=100')


# %%
#
# Arbitrary function scales
# ============================
#
# Users can define a full scale class and pass that to `~.axes.Axes.set_xscale`
# and `~.axes.Axes.set_yscale` (see :ref:`custom_scale`).  A short cut for this
# is to use the 'function' scale, and pass as extra arguments a ``forward`` and
# an ``inverse`` function.  The following performs a `Mercator transform
# <https://en.wikipedia.org/wiki/Mercator_projection>`_ to the y-axis.

# Function Mercator transform
def forward(a):
    # 将角度转换为弧度
    a = np.deg2rad(a)
    # 执行 Mercator 变换
    return np.rad2deg(np.log(np.abs(np.tan(a) + 1.0 / np.cos(a))))


def inverse(a):
    # 将角度转换为弧度
    a = np.deg2rad(a)
    # 执行 Mercator 反变换
    return np.rad2deg(np.arctan(np.sinh(a)))


t = np.arange(0, 170.0, 0.1)
s = t / 2.

# 创建一个图形对象，使用 'constrained' 布局
fig, ax = plt.subplots(layout='constrained')
# 在当前子图 ax 上绘制 t vs s 的图像
ax.plot(t, s, '-', lw=2)

# 设置 y 轴的缩放为自定义函数缩放 'function'，并传入 forward 和 inverse 函数
ax.set_yscale('function', functions=(forward, inverse))
# 设置子图标题为 'function: Mercator'
ax.set_title('function: Mercator')
# 启用网格线
ax.grid(True)
# 设置 x 轴的限制为 [0, 180]
ax.set_xlim([0, 180])
# 设置 y 轴的次要刻度格式化器为空
ax.yaxis.set_minor_formatter(NullFormatter())
# 设置 y 轴的主刻度定位器为固定的位置
ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 90, 10)))


# %%
#
# What is a "scale"?
# ==================
#
# A scale is an object that gets attached to an axis.  The class documentation
# is at `~matplotlib.scale`. `~.axes.Axes.set_xscale` and `~.axes.Axes.set_yscale`
# set the scale on the respective Axis objects.  You can determine the scale
# on an axis with `~.axis.Axis.get_scale`:

# 创建一个图形对象，使用 'constrained' 布局和指定大小
fig, ax = plt.subplots(layout='constrained', figsize=(3.2, 3))
# 在当前子图 ax 上绘制 x vs x 的半对数图像
ax.semilogy(x, x)

# 打印 x 轴的缩放类型
print(ax.xaxis.get_scale())
# 打印 y 轴的缩放类型
print(ax.yaxis.get_scale())

# %%
#
# Setting a scale does three things.  First it defines a transform on the axis
# that maps between data values to position along the axis.  This transform can
# be accessed via ``get_transform``:

# 打印 y 轴的变换
print(ax.yaxis.get_transform())

# %%
#
# Transforms on the axis are a relatively low-level concept, but is one of the
# important roles played by ``set_scale``.
#
# Setting the scale also sets default tick locators (`~.ticker`) and tick
# formatters appropriate for the scale.   An axis with a 'log' scale has a
# `~.ticker.LogLocator` to pick ticks at decade intervals, and a
# `~.ticker.LogFormatter` to use scientific notation on the decades.

# 打印 'X axis'
print('X axis')
# 打印 x 轴的主刻度定位器
print(ax.xaxis.get_major_locator())
# 打印 x 轴的主刻度格式化器
print(ax.xaxis.get_major_formatter())

# 打印 'Y axis'
print('Y axis')
# 打印 y 轴的主刻度定位器
print(ax.yaxis.get_major_locator())
# 打印 y 轴的主刻度格式化器
print(ax.yaxis.get_major_formatter())
```