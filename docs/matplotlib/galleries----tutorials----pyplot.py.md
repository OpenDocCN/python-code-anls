# `D:\src\scipysrc\matplotlib\galleries\tutorials\pyplot.py`

```
"""
.. redirect-from:: /tutorials/introductory/pyplot

.. _pyplot_tutorial:

===============
Pyplot tutorial
===============

An introduction to the pyplot interface.  Please also see
:ref:`quick_start` for an overview of how Matplotlib
works and :ref:`api_interfaces` for an explanation of the trade-offs between the
supported user APIs.

"""

# %%
# Introduction to pyplot
# ======================
#
# :mod:`matplotlib.pyplot` is a collection of functions that make matplotlib
# work like MATLAB.  Each ``pyplot`` function makes some change to a figure:
# e.g., creates a figure, creates a plotting area in a figure, plots some lines
# in a plotting area, decorates the plot with labels, etc.
#
# In :mod:`matplotlib.pyplot` various states are preserved
# across function calls, so that it keeps track of things like
# the current figure and plotting area, and the plotting
# functions are directed to the current Axes (please note that we use uppercase
# Axes to refer to the `~.axes.Axes` concept, which is a central
# :ref:`part of a figure <figure_parts>`
# and not only the plural of *axis*).
#
# .. note::
#
#    The implicit pyplot API is generally less verbose but also not as flexible as the
#    explicit API.  Most of the function calls you see here can also be called
#    as methods from an ``Axes`` object. We recommend browsing the tutorials
#    and examples to see how this works. See :ref:`api_interfaces` for an
#    explanation of the trade-off of the supported user APIs.
#
# Generating visualizations with pyplot is very quick:

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

plt.plot([1, 2, 3, 4])  # 绘制简单的折线图，y 值为 [1, 2, 3, 4]，x 值默认为 [0, 1, 2, 3]
plt.ylabel('some numbers')  # 设置 y 轴标签为 'some numbers'
plt.show()  # 显示图形

# %%
# You may be wondering why the x-axis ranges from 0-3 and the y-axis
# from 1-4.  If you provide a single list or array to
# `~.pyplot.plot`, matplotlib assumes it is a
# sequence of y values, and automatically generates the x values for
# you.  Since python ranges start with 0, the default x vector has the
# same length as y but starts with 0; therefore, the x data are
# ``[0, 1, 2, 3]``.
#
# `~.pyplot.plot` is a versatile function, and will take an arbitrary number of
# arguments.  For example, to plot x versus y, you can write:

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])  # 绘制 x versus y 的折线图

# %%
# Formatting the style of your plot
# ---------------------------------
#
# For every x, y pair of arguments, there is an optional third argument
# which is the format string that indicates the color and line type of
# the plot.  The letters and symbols of the format string are from
# MATLAB, and you concatenate a color string with a line style string.
# The default format string is 'b-', which is a solid blue line.  For
# example, to plot the above with red circles, you would issue

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')  # 绘制红色圆圈的折线图
plt.axis((0, 6, 0, 20))  # 设置坐标轴范围为 (0, 6, 0, 20)
plt.show()  # 显示图形

# %%
# See the `~.pyplot.plot` documentation for a complete
# list of line styles and format strings.  The
# `~.pyplot.axis` function in the example above takes a
# list of ``[xmin, xmax, ymin, ymax]`` and specifies the viewport of the
# Axes.
#
# If matplotlib were limited to working with lists, it would be fairly
# useless for numeric processing.  Generally, you will use `numpy
# <https://numpy.org/>`_ arrays.  In fact, all sequences are
# converted to numpy arrays internally.  The example below illustrates
# plotting several lines with different format styles in one function call
# using arrays.

import numpy as np

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

# %%
# .. _plotting-with-keywords:
#
# Plotting with keyword strings
# =============================
#
# There are some instances where you have data in a format that lets you
# access particular variables with strings. For example, with `structured arrays`_
# or `pandas.DataFrame`.
#
# .. _structured arrays: https://numpy.org/doc/stable/user/basics.rec.html#structured-arrays
#
# Matplotlib allows you to provide such an object with
# the ``data`` keyword argument. If provided, then you may generate plots with
# the strings corresponding to these variables.

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()

# %%
# .. _plotting-with-categorical-vars:
#
# Plotting with categorical variables
# ===================================
#
# It is also possible to create a plot using categorical variables.
# Matplotlib allows you to pass categorical variables directly to
# many plotting functions. For example:

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()

# %%
# .. _controlling-line-properties:
#
# Controlling line properties
# ===========================
#
# Lines have many attributes that you can set: linewidth, dash style,
# antialiased, etc; see `matplotlib.lines.Line2D`.  There are
# several ways to set line properties
#
# * Use keyword arguments::
#
#       plt.plot(x, y, linewidth=2.0)
#
#
# * Use the setter methods of a ``Line2D`` instance.  ``plot`` returns a list
#   of ``Line2D`` objects; e.g., ``line1, line2 = plot(x1, y1, x2, y2)``.  In the code
#   below we will suppose that we have only
#   one line so that the list returned is of length 1.  We use tuple unpacking with
#   ``line,`` to get the first element of that list::
#
#       line, = plt.plot(x, y, '-')
#       line.set_antialiased(False) # turn off antialiasing
#
# * Use `~.pyplot.setp`.  The example below
#   uses a MATLAB-style function to set multiple properties
#   on a list of lines.  ``setp`` works transparently with a list of objects
#   or a single object.  You can either use python keyword arguments or
#   MATLAB-style string/value pairs::

# 被应用在一组线条上。`setp` 函数可以透明地处理一个对象列表或单个对象。
# 可以使用 Python 的关键字参数或 MATLAB 风格的字符串/值对：

#       lines = plt.plot(x1, y1, x2, y2)
#       # use keyword arguments
#       plt.setp(lines, color='r', linewidth=2.0)
#       # or MATLAB style string value pairs
#       plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
#
#
# Here are the available `~.lines.Line2D` properties.
#
# ======================  ==================================================
# Property                Value Type
# ======================  ==================================================
# alpha                   float
# animated                [True | False]
# antialiased or aa       [True | False]
# clip_box                a matplotlib.transform.Bbox instance
# clip_on                 [True | False]
# clip_path               a Path instance and a Transform instance, a Patch
# color or c              any matplotlib color
# contains                the hit testing function
# dash_capstyle           [`'butt'` | `'round'` | `'projecting'`]
# dash_joinstyle          [`'miter'` | `'round'` | `'bevel'`]
# dashes                  sequence of on/off ink in points
# data                    (np.array xdata, np.array ydata)
# figure                  a matplotlib.figure.Figure instance
# label                   any string
# linestyle or ls         [`'-'` | `'--'` | `'-.'` | `':'` | `'steps'` | ...]
# linewidth or lw         float value in points
# marker                  [`'+'` | `','` | `'.'` | `'1'` | `'2'` | `'3'` | `'4'`]
# markeredgecolor or mec  any matplotlib color
# markeredgewidth or mew  float value in points
# markerfacecolor or mfc  any matplotlib color
# markersize or ms        float
# markevery               [None | integer | (startind, stride)]
# picker                  used in interactive line selection
# pickradius              the line pick selection radius
# solid_capstyle          [`'butt'` | `'round'` | `'projecting'`]
# solid_joinstyle         [`'miter'` | `'round'` | `'bevel'`]
# transform               a matplotlib.transforms.Transform instance
# visible                 [True | False]
# xdata                   np.array
# ydata                   np.array
# zorder                  any number
# ======================  ==================================================
#
# To get a list of settable line properties, call the
# `~.pyplot.setp` function with a line or lines as argument
#
# .. sourcecode:: ipython
#
#     In [69]: lines = plt.plot([1, 2, 3])
#
#     In [70]: plt.setp(lines)
#       alpha: float
#       animated: [True | False]
#       antialiased or aa: [True | False]
#       ...snip
#
# .. _multiple-figs-axes:
#
#
# Working with multiple figures and Axes
# ======================================
#
# MATLAB, and :mod:`.pyplot`, have the concept of the current figure

# 这些是可用的 `~.lines.Line2D` 属性。
#
# ======================  ==================================================
# 属性名                  类型
# ======================  ==================================================
# alpha                   浮点数
# animated                [True | False]
# antialiased 或 aa       [True | False]
# clip_box                一个 matplotlib.transform.Bbox 实例
# clip_on                 [True | False]
# clip_path               一个 Path 实例和一个 Transform 实例，一个 Patch
# color 或 c              任意 matplotlib 颜色
# contains                点击测试函数
# dash_capstyle           [`'butt'` | `'round'` | `'projecting'`]
# dash_joinstyle          [`'miter'` | `'round'` | `'bevel'`]
# dashes                  以点为单位的 on/off 墨水序列
# data                    (np.array xdata, np.array ydata)
# figure                  一个 matplotlib.figure.Figure 实例
# label                   任意字符串
# linestyle 或 ls         [`'-'` | `'--'` | `'-.'` | `':'` | `'steps'` | ...]
# linewidth 或 lw         浮点数，单位为点
# marker                  [`'+'` | `','` | `'.'` | `'1'` | `'2'` | `'3'` | `'4'`]
# markeredgecolor 或 mec  任意 matplotlib 颜色
# markeredgewidth 或 mew  浮点数，单位为点
# markerfacecolor 或 mfc  任意 matplotlib 颜色
# markersize 或 ms        浮点数
# markevery               [None | 整数 | (startind, stride)]
# picker                  用于交互式线选择
# pickradius              线选择半径
# solid_capstyle          [`'butt'` | `'round'` | `'projecting'`]
# solid_joinstyle         [`'miter'` | `'round'` | `'bevel'`]
# transform               一个 matplotlib.transforms.Transform 实例
# visible                 [True | False]
# xdata                   np.array
# ydata                   np.array
# zorder                  任意数值
# ======================  ==================================================
#
# 要获取可设置的线条属性列表，请使用 `~.pyplot.setp` 函数并将线条作为参数传递进去
# and the current Axes.  All plotting functions apply to the current
# Axes.  The function `~.pyplot.gca` returns the current Axes (a
# `matplotlib.axes.Axes` instance), and `~.pyplot.gcf` returns the current
# figure (a `matplotlib.figure.Figure` instance). Normally, you don't have to
# worry about this, because it is all taken care of behind the scenes.  Below
# is a script to create two subplots.

# 定义一个函数 f(t)，返回指数衰减乘以余弦函数的结果
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

# 生成两组时间序列数据
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

# 创建一个新的图形窗口
plt.figure()
# 创建第一个子图，subplot(211) 表示总共2行1列的布局中的第1个子图
plt.subplot(211)
# 绘制 t1 对应的函数 f(t1) 的散点图（蓝色圆点），以及 t2 对应的函数 f(t2) 的线图（黑色实线）
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

# 创建第二个子图，subplot(212) 表示总共2行1列的布局中的第2个子图
plt.subplot(212)
# 绘制 t2 对应的余弦函数的线图（红色虚线）
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# 显示图形
plt.show()

# %%
# The `~.pyplot.figure` call here is optional because a figure will be created
# if none exists, just as an Axes will be created (equivalent to an explicit
# ``subplot()`` call) if none exists.
# The `~.pyplot.subplot` call specifies ``numrows,
# numcols, plot_number`` where ``plot_number`` ranges from 1 to
# ``numrows*numcols``.  The commas in the ``subplot`` call are
# optional if ``numrows*numcols<10``.  So ``subplot(211)`` is identical
# to ``subplot(2, 1, 1)``.
#
# You can create an arbitrary number of subplots
# and Axes.  If you want to place an Axes manually, i.e., not on a
# rectangular grid, use `~.pyplot.axes`,
# which allows you to specify the location as ``axes([left, bottom,
# width, height])`` where all values are in fractional (0 to 1)
# coordinates.  See :doc:`/gallery/subplots_axes_and_figures/axes_demo` for an example of
# placing Axes manually and :doc:`/gallery/subplots_axes_and_figures/subplot` for an
# example with lots of subplots.
#
# You can create multiple figures by using multiple
# `~.pyplot.figure` calls with an increasing figure
# number.  Of course, each figure can contain as many Axes and subplots
# as your heart desires::
#
#     import matplotlib.pyplot as plt
#     plt.figure(1)                # the first figure
#     plt.subplot(211)             # the first subplot in the first figure
#     plt.plot([1, 2, 3])
#     plt.subplot(212)             # the second subplot in the first figure
#     plt.plot([4, 5, 6])
#
#
#     plt.figure(2)                # a second figure
#     plt.plot([4, 5, 6])          # creates a subplot() by default
#
#     plt.figure(1)                # first figure current;
#                                  # subplot(212) still current
#     plt.subplot(211)             # make subplot(211) in the first figure
#                                  # current
#     plt.title('Easy as 1, 2, 3') # subplot 211 title
#
# You can clear the current figure with `~.pyplot.clf`
# and the current Axes with `~.pyplot.cla`.  If you find
# it annoying that states (specifically the current image, figure and Axes)
# are being maintained for you behind the scenes, don't despair: this is just a thin
# stateful wrapper around an object-oriented API, which you can use
# instead (see :ref:`artists_tutorial`)
#
# 创建一个子图并将其分配给变量ax，这个子图默认是一个1x1的网格中的第一个子图
ax = plt.subplot()

# 生成一个包含0到4.99之间，步长为0.01的等差数列，并赋值给变量t
t = np.arange(0.0, 5.0, 0.01)

# 计算t的余弦值，并赋值给变量s
s = np.cos(2*np.pi*t)

# 在子图ax上绘制余弦函数曲线，线宽为2个单位，并将返回的线条对象赋值给变量line
line, = plt.plot(t, s, lw=2)
# %%
# In this section, an annotation is added to the plot using `plt.annotate()`.
# It marks a 'local max' at the coordinates (2, 1) with a text location set at (3, 1.5).
# The annotation includes an arrow pointing to the 'local max' with specified properties,
# such as arrow color and shrinkage.
plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

# %%
# Set the y-axis limits of the current plot to be between -2 and 2.
plt.ylim(-2, 2)

# Display the plot with all modifications applied.
plt.show()

# %%
# In this section, various examples of different axis scales are demonstrated.
#
# The first example shows a linear scale for the y-axis.
#
# The second example demonstrates a logarithmic scale for the y-axis.
#
# The third example displays a symmetric logarithmic scale (`symlog`) for the y-axis
# with a linear threshold (`linthresh`) set to 0.01.
#
# The fourth example exhibits a logit scale for the y-axis, which is useful for
# representing data that spans from 0 to 1 exclusively.
#
# Each subplot includes a grid and a title indicating the type of scale used.
# The layout of subplots is adjusted to accommodate different scales and label sizes.
#
# More details on annotations and different scales can be found in the Matplotlib
# documentation.
plt.figure()

# %%
# Fixing the random seed for reproducibility of the generated data.
np.random.seed(19680801)

# Generate some random data (`y`) with a normal distribution.
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
# Select data points within the open interval (0, 1).
y = y[(y > 0) & (y < 1)]
# Sort the data points.
y.sort()
# Create an array (`x`) representing the index of sorted `y` values.
x = np.arange(len(y))

# Plotting with various axis scales in separate subplots.
plt.figure()

# %%
# Subplot 1: Linear scale for the y-axis.
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# %%
# Subplot 2: Logarithmic scale (base 10) for the y-axis.
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# %%
# Subplot 3: Symmetric logarithmic scale for the y-axis (`symlog`).
# `linthresh` parameter is set to 0.01 to define the linear range around zero.
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)

# %%
# Subplot 4: Logit scale for the y-axis, which compresses values between 0 and 1.
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)

# Adjusting the layout of subplots to ensure proper spacing and alignment.
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

# Display all subplots.
plt.show()

# %%
# Additional note: Matplotlib supports the creation of custom scales through
# `matplotlib.scale` module. Refer to the documentation for more details.
```