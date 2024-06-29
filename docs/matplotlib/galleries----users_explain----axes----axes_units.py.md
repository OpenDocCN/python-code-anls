# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\axes_units.py`

```py
"""
.. _user_axes_units:

==========================
Plotting dates and strings
==========================

The most basic way to use Matplotlib plotting methods is to pass coordinates in
as numerical numpy arrays.  For example, ``plot(x, y)`` will work if ``x`` and
``y`` are numpy arrays of floats (or integers).  Plotting methods will also
work if `numpy.asarray` will convert ``x`` and ``y`` to an array of floating
point numbers; e.g. ``x`` could be a python list.

Matplotlib also has the ability to convert other data types if a "unit
converter" exists for the data type.  Matplotlib has two built-in converters,
one for dates and the other for lists of strings.  Other downstream libraries
have their own converters to handle their data types.

The method to add converters to Matplotlib is described in `matplotlib.units`.
Here we briefly overview the built-in date and string converters.

Date conversion
===============

If ``x`` and/or ``y`` are a list of `datetime` or an array of
`numpy.datetime64`, Matplotlib has a built-in converter that will convert the
datetime to a float, and add tick locators and formatters to the axis that are
appropriate for dates.  See `matplotlib.dates`.

In the following example, the x-axis gains a converter that converts from
`numpy.datetime64` to float, and a locator that put ticks at the beginning of
the month, and a formatter that label the ticks appropriately:
"""

import numpy as np

import matplotlib.dates as mdates  # 导入 Matplotlib 的日期处理模块
import matplotlib.units as munits  # 导入 Matplotlib 的单位处理模块

import matplotlib.pyplot as plt  # 导入 Matplotlib 的绘图模块

fig, ax = plt.subplots(figsize=(5.4, 2), layout='constrained')  # 创建一个子图，并设置大小和布局约束
time = np.arange('1980-01-01', '1980-06-25', dtype='datetime64[D]')  # 生成一个日期范围的 numpy 数组
x = np.arange(len(time))  # 生成与日期范围长度相同的整数 numpy 数组
ax.plot(time, x)  # 绘制时间和 x 数组的折线图

# %%
#
# Note that if we try to plot a float on the x-axis, it will be plotted in
# units of days since the "epoch" for the converter, in this case 1970-01-01
# (see :ref:`date-format`).  So when we plot the value 0, the ticks start at
# 1970-01-01.  (The locator also now chooses every two years for a tick instead
# of every month):

fig, ax = plt.subplots(figsize=(5.4, 2), layout='constrained')  # 创建一个子图，并设置大小和布局约束
time = np.arange('1980-01-01', '1980-06-25', dtype='datetime64[D]')  # 生成一个日期范围的 numpy 数组
x = np.arange(len(time))  # 生成与日期范围长度相同的整数 numpy 数组
ax.plot(time, x)  # 绘制时间和 x 数组的折线图
# 0 gets labeled as 1970-01-01
ax.plot(0, 0, 'd')  # 在坐标 (0, 0) 处绘制一个菱形点
ax.text(0, 0, ' Float x=0', rotation=45)  # 在 (0, 0) 处添加文本 ' Float x=0'，并旋转 45 度

# %%
#
# We can customize the locator and the formatter; see :ref:`date-locators` and
# :ref:`date-formatters` for a complete list, and
# :ref:`date_formatters_locators` for examples of them in use.  Here we locate
# by every second month, and format just with the month's 3-letter name using
# ``"%b"`` (see `~datetime.datetime.strftime` for format codes):

fig, ax = plt.subplots(figsize=(5.4, 2), layout='constrained')  # 创建一个子图，并设置大小和布局约束
time = np.arange('1980-01-01', '1980-06-25', dtype='datetime64[D]')  # 生成一个日期范围的 numpy 数组
x = np.arange(len(time))  # 生成与日期范围长度相同的整数 numpy 数组
ax.plot(time, x)  # 绘制时间和 x 数组的折线图
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=np.arange(1, 13, 2)))  # 设置主刻度定位器为每隔两个月
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # 设置主刻度格式化器为月份的缩写 '%b'
# %%
#
# Set the x-axis label of the current axes to '1980'.
ax.set_xlabel('1980')

# %%
#
# The default locator is the `~.dates.AutoDateLocator`, and the default
# Formatter `~.dates.AutoDateFormatter`.  There are also  "concise" formatter
# and locators that give a more compact labelling, and can be set via rcParams.
# Note how instead of the redundant "Jan" label at the start of the year,
# "1980" is used instead.  See :ref:`date_concise_formatter` for more examples.

# Set the date converter to use a "concise" formatter for date labels.
plt.rcParams['date.converter'] = 'concise'

# Create a figure and an axes with a constrained layout, size 5.4 x 2 inches.
fig, ax = plt.subplots(figsize=(5.4, 2), layout='constrained')

# Generate time data from January 1, 1980 to June 25, 1980 in datetime64[D] dtype.
time = np.arange('1980-01-01', '1980-06-25', dtype='datetime64[D]')
x = np.arange(len(time))

# Plot x against time on the axes.
ax.plot(time, x)

# %%
#
# We can set the limits on the axis either by passing the appropriate dates as
# limits, or by passing a floating-point value in the proper units of days
# since the epoch.  If we need it, we can get this value from
# `~.dates.date2num`.

# Create a figure with 2 subplots vertically, each with constrained layout and size 5.4 x 3 inches.
fig, axs = plt.subplots(2, 1, figsize=(5.4, 3), layout='constrained')

# Iterate over each subplot in axs.
for ax in axs.flat:
    # Generate time data from January 1, 1980 to June 25, 1980 in datetime64[D] dtype.
    time = np.arange('1980-01-01', '1980-06-25', dtype='datetime64[D]')
    x = np.arange(len(time))
    
    # Plot x against time on the current axis.
    ax.plot(time, x)

# Set x-axis limits for the first subplot using datetime64.
axs[0].set_xlim(np.datetime64('1980-02-01'), np.datetime64('1980-04-01'))

# Set x-axis limits for the second subplot using floating-point values.
# Note: The float values correspond to days since the epoch.
axs[1].set_xlim(3683, 3683+60)

# %%
#
# String conversion: categorical plots
# ====================================
#
# Sometimes we want to label categories on an axis rather than numbers.
# Matplotlib allows this using a "categorical" converter (see
# `~.matplotlib.category`).

# Data for categorical plotting.
data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}
names = list(data.keys())
values = list(data.values())

# Create a figure with 3 subplots horizontally, sharing y-axis, with constrained layout.
fig, axs = plt.subplots(1, 3, figsize=(7, 3), sharey=True, layout='constrained')

# Plot a bar chart of names against values on the first subplot.
axs[0].bar(names, values)

# Plot a scatter plot of names against values on the second subplot.
axs[1].scatter(names, values)

# Plot a line plot of names against values on the third subplot.
axs[2].plot(names, values)

# Set the figure title to 'Categorical Plotting'.
fig.suptitle('Categorical Plotting')

# %%
#
# Note that the "categories" are plotted in the order that they are first
# specified and that subsequent plotting in a different order will not affect
# the original order.  Further, new additions will be added on the end (see
# "pear" below):

# Create a figure with a constrained layout, size 5 x 3 inches.
fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')

# Plot a bar chart of names against values on the axis.
ax.bar(names, values)

# Plot a scatter plot of selected categories ('lemon' and 'apple') against new values.
ax.scatter(['lemon', 'apple'], [7, 12])

# Add a new category 'pear' and plot all categories in a different order.
ax.plot(['pear', 'orange', 'apple', 'lemon'], [13, 10, 7, 12], color='C1')

# %%
#
# Note that when using ``plot`` like in the above, the order of the plotting is
# mapped onto the original order of the data, so the new line goes in the order
# specified.
#
# The category converter maps from categories to integers, starting at zero. So
# data can also be manually added to the axis using a float.  Note that if a
# float is passed in that does not have a "category" associated with it, the
# data point can still be plotted, but a tick will not be created.  In the
# following, we plot data at 4.0 and 2.5, but no tick is added there because
# those are not categories.

fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
ax.bar(names, values)
# arguments for styling the labels below:
args = {'rotation': 70, 'color': 'C1',
        'bbox': {'color': 'white', 'alpha': .7, 'boxstyle': 'round'}}

# 0 gets labeled as "apple"
ax.plot(0, 2, 'd', color='C1')
ax.text(0, 3, 'Float x=0', **args)

# 2 gets labeled as "lemon"
ax.plot(2, 2, 'd', color='C1')
ax.text(2, 3, 'Float x=2', **args)

# 4 doesn't get a label
ax.plot(4, 2, 'd', color='C1')
ax.text(4, 3, 'Float x=4', **args)

# 2.5 doesn't get a label
ax.plot(2.5, 2, 'd', color='C1')
ax.text(2.5, 3, 'Float x=2.5', **args)

# %%
#
# Setting the limits for a category axis can be done by specifying the
# categories, or by specifying floating point numbers:

fig, axs = plt.subplots(2, 1, figsize=(5, 5), layout='constrained')
ax = axs[0]
ax.bar(names, values)
ax.set_xlim('orange', 'lemon')
ax.set_xlabel('limits set with categories')
ax = axs[1]
ax.bar(names, values)
ax.set_xlim(0.5, 2.5)
ax.set_xlabel('limits set with floats')

# %%
#
# The category axes are helpful for some plot types, but can lead to confusion
# if data is read in as a list of strings, even if it is meant to be a list of
# floats or dates.  This sometimes happens when reading comma-separated value
# (CSV) files. The categorical locator and formatter will put a tick at every
# string value and label each one as well:

fig, ax = plt.subplots(figsize=(5.4, 2.5), layout='constrained')
x = [str(xx) for xx in np.arange(100)]  # list of strings
ax.plot(x, np.arange(100))
ax.set_xlabel('x is list of strings')

# %%
#
# If this is not desired, then simply convert the data to floats before plotting:

fig, ax = plt.subplots(figsize=(5.4, 2.5), layout='constrained')
x = np.asarray(x, dtype='float')  # array of float.
ax.plot(x, np.arange(100))
ax.set_xlabel('x is array of floats')

# %%
#
# Determine converter, formatter, and locator on an axis
# ======================================================
#
# Sometimes it is helpful to be able to debug what Matplotlib is using to
# convert the incoming data.  We can do that by querying the ``converter``
# property on the axis.  We can also query the formatters and locators using
# `~.axis.Axis.get_major_locator` and `~.axis.Axis.get_major_formatter`.
#
# Note that by default the converter is *None*.

fig, axs = plt.subplots(3, 1, figsize=(6.4, 7), layout='constrained')
x = np.arange(100)
ax = axs[0]
ax.plot(x, x)
label = f'Converter: {ax.xaxis.converter}\n '
label += f'Locator: {ax.xaxis.get_major_locator()}\n'
label += f'Formatter: {ax.xaxis.get_major_formatter()}\n'
ax.set_xlabel(label)

ax = axs[1]
time = np.arange('1980-01-01', '1980-06-25', dtype='datetime64[D]')
x = np.arange(len(time))
ax.plot(time, x)
label = f'Converter: {ax.xaxis.converter}\n '
label += f'Locator: {ax.xaxis.get_major_locator()}\n'
# 将文本标签更新为包含主 X 轴格式化器的信息
label += f'Formatter: {ax.xaxis.get_major_formatter()}\n'
# 将更新后的文本标签设置为 X 轴的标签
ax.set_xlabel(label)

# 获取第三个子图对象
ax = axs[2]
# 准备要绘制的数据和名称
data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}
names = list(data.keys())
values = list(data.values())
# 在第三个子图上绘制数据
ax.plot(names, values)
# 构建包含轴转换器和主要刻度定位器、格式化器信息的标签
label = f'Converter: {ax.xaxis.converter}\n '
label += f'Locator: {ax.xaxis.get_major_locator()}\n'
label += f'Formatter: {ax.xaxis.get_major_formatter()}\n'
# 将更新后的标签设置为 X 轴的标签
ax.set_xlabel(label)

# %%
#
# 关于“单位”支持的更多信息
# =========================
#
# 日期和类别的支持是 Matplotlib 内置的“单位”支持的一部分。
# 这在 `.matplotlib.units` 和 :ref:`basic_units` 示例中有描述。
#
# 单位支持通过查询传递给绘图函数的数据类型，并分派给接受该类型数据的第一个转换器来工作。
# 因此，如果 ``x`` 包含日期时间对象，则转换器将是 ``_SwitchableDateConverter``；
# 如果包含字符串，则将发送到 ``StrCategoryConverter``。

for k, v in munits.registry.items():
    # 打印单位注册表中的类型和相应的转换器类型
    print(f"type: {k};\n    converter: {type(v)}")

# %%
#
# 有许多下游库提供了自己的具有定位器和格式化器的转换器。
# 物理单位支持由 `astropy <https://www.astropy.org>`_、`pint <https://pint.readthedocs.io>`_ 和
# `unyt <https://unyt.readthedocs.io>`_ 等提供。
#
# 像 `pandas <https://pandas.pydata.org>`_ 和 `nc-time-axis <https://nc-time-axis.readthedocs.io>`_
# （因此 `xarray <https://docs.xarray.dev>`_）这样的高级库提供了自己的日期时间支持。
# 如果使用这些库，使用 Matplotlib 的定位器和格式化器时需要注意它们可能不兼容 Matplotlib 的原生日期时间支持。
```