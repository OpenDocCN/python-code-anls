# `D:\src\scipysrc\matplotlib\galleries\users_explain\quick_start.py`

```
"""
.. redirect-from:: /tutorials/introductory/usage
.. redirect-from:: /tutorials/introductory/quick_start

.. _quick_start:

*****************
Quick start guide
*****************

This tutorial covers some basic usage patterns and best practices to
help you get started with Matplotlib.

"""

import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

# sphinx_gallery_thumbnail_number = 3

# %%
#
# A simple example
# ================
#
# Matplotlib graphs your data on `.Figure`\s (e.g., windows, Jupyter
# widgets, etc.), each of which can contain one or more `~.axes.Axes`, an
# area where points can be specified in terms of x-y coordinates (or theta-r
# in a polar plot, x-y-z in a 3D plot, etc.).  The simplest way of
# creating a Figure with an Axes is using `.pyplot.subplots`. We can then use
# `.Axes.plot` to draw some data on the Axes, and `~.pyplot.show` to display
# the figure:

fig, ax = plt.subplots()             # 创建一个包含单个 Axes 的图形对象。
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # 在 Axes 上绘制一些数据。
plt.show()                           # 显示图形。

# %%
#
# Depending on the environment you are working in, ``plt.show()`` can be left
# out. This is for example the case with Jupyter notebooks, which
# automatically show all figures created in a code cell.
#
# .. _figure_parts:
#
# Parts of a Figure
# =================
#
# Here are the components of a Matplotlib Figure.
#
# .. image:: ../../_static/anatomy.png
#
# :class:`~matplotlib.figure.Figure`
# ----------------------------------
#
# The **whole** figure.  The Figure keeps
# track of all the child :class:`~matplotlib.axes.Axes`, a group of
# 'special' Artists (titles, figure legends, colorbars, etc.), and
# even nested subfigures.
#
# Typically, you'll create a new Figure through one of the following
# functions::
#
#    fig = plt.figure()             # 创建一个空的图形对象，没有 Axes。
#    fig, ax = plt.subplots()       # 创建一个包含单个 Axes 的图形对象。
#    fig, axs = plt.subplots(2, 2)  # 创建一个包含 2x2 网格的图形对象。
#    # 创建一个左边有一个 Axes，右边有两个 Axes 的图形对象:
#    fig, axs = plt.subplot_mosaic([['left', 'right_top'],
#                                   ['left', 'right_bottom']])
#
# `~.pyplot.subplots()` 和 `~.pyplot.subplot_mosaic` 是方便的函数，
# 它们在图形对象内部额外创建 Axes 对象，但您也可以稍后手动添加 Axes。
#
# For more on Figures, including panning and zooming, see :ref:`figure-intro`.
#
# :class:`~matplotlib.axes.Axes`
# ------------------------------
#
# An Axes is an Artist attached to a Figure that contains a region for
# plotting data, and usually includes two (or three in the case of 3D)
# :class:`~matplotlib.axis.Axis` objects (be aware of the difference
# between **Axes** and **Axis**) that provide ticks and tick labels to
# provide scales for the data in the Axes. Each :class:`~.axes.Axes` also
# has a title
np.random.seed(19680801)  # 设置随机数生成器的种子。

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# 创建一个新的图形和一个包含单个子图的新Axes对象，并设置其尺寸为(5, 2.7)。
ax.scatter('a', 'b', c='c', s='d', data=data)
# 在Axes对象上绘制散点图，x轴数据为'a'列，y轴数据为'b'列，颜色由'c'列指定，大小由'd'列指定。

ax.set_xlabel('entry a')
# 设置x轴标签为'entry a'。
ax.set_ylabel('entry b')
# 设置y轴标签为'entry b'
# - Explicitly create Figures and Axes, and call methods on them (the
#   "object-oriented (OO) style").
# - Rely on pyplot to implicitly create and manage the Figures and Axes, and
#   use pyplot functions for plotting.
#
# See :ref:`api_interfaces` for an explanation of the tradeoffs between the
# implicit and explicit interfaces.
#
# So one can use the OO-style

# 生成样本数据
x = np.linspace(0, 2, 100)  # Sample data.

# 注意，在面向对象（OO）风格中，仍然使用 `.pyplot.figure` 创建 Figure。
# 创建包含约束布局的 Figure 和 Axes 对象
fig, ax = plt.subplots(figsize=(5, 2.7), constrained_layout=True)
# 在 Axes 上绘制数据
ax.plot(x, x, label='linear')  # Plot some data on the Axes.
ax.plot(x, x**2, label='quadratic')  # Plot more data on the Axes...
ax.plot(x, x**3, label='cubic')  # ... and some more.
# 添加 x 轴标签
ax.set_xlabel('x label')  # Add an x-label to the Axes.
# 添加 y 轴标签
ax.set_ylabel('y label')  # Add a y-label to the Axes.
# 添加标题
ax.set_title("Simple Plot")  # Add a title to the Axes.
# 添加图例
ax.legend()  # Add a legend.

# %%
# or the pyplot-style:

# 重新生成样本数据
x = np.linspace(0, 2, 100)  # Sample data.

# 使用 pyplot 风格创建 Figure
plt.figure(figsize=(5, 2.7), constrained_layout=True)
# 在 (隐式) Axes 上绘制数据
plt.plot(x, x, label='linear')  # Plot some data on the (implicit) Axes.
plt.plot(x, x**2, label='quadratic')  # etc.
plt.plot(x, x**3, label='cubic')
# 添加 x 轴标签
plt.xlabel('x label')
# 添加 y 轴标签
plt.ylabel('y label')
# 添加标题
plt.title("Simple Plot")
# 添加图例
plt.legend()

# %%
# (In addition, there is a third approach, for the case when embedding
# Matplotlib in a GUI application, which completely drops pyplot, even for
# figure creation. See the corresponding section in the gallery for more info:
# :ref:`user_interfaces`.)
#
# Matplotlib's documentation and examples use both the OO and the pyplot
# styles. In general, we suggest using the OO style, particularly for
# complicated plots, and functions and scripts that are intended to be reused
# as part of a larger project. However, the pyplot style can be very convenient
# for quick interactive work.
#
# .. note::
#
#    You may find older examples that use the ``pylab`` interface,
#    via ``from pylab import *``. This approach is strongly deprecated.
#
# Making a helper functions
# -------------------------
#
# If you need to make the same plots over and over again with different data
# sets, or want to easily wrap Matplotlib methods, use the recommended
# signature function below.

def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph.
    """
    out = ax.plot(data1, data2, **param_dict)
    return out

# %%
# which you would then use twice to populate two subplots:

# 创建四组随机数据集
data1, data2, data3, data4 = np.random.randn(4, 100)
# 创建包含两个子图的 Figure 和 Axes 对象
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
# 使用自定义的 helper function 在两个子图上绘制数据
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})

# %%
# Note that if you want to install these as a python package, or any other
# customizations you could use one of the many templates on the web;
# Matplotlib has one at `mpl-cookiecutter
# <https://github.com/matplotlib/matplotlib-extension-cookiecutter>`_
#
#
# Styling Artists
# ===============
#
# Most plotting methods have styling options for the Artists, accessible either
# when a plotting method is called, or from a "setter" on the Artist.  In the
# plot below we manually set the *color*, *linewidth*, and *linestyle* of the
# Artists created by `~.Axes.plot`, and we set the linestyle of the second line
# after the fact with `~.Line2D.set_linestyle`.

fig, ax = plt.subplots(figsize=(5, 2.7))  # 创建一个大小为5x2.7英寸的图形窗口和Axes对象
x = np.arange(len(data1))  # 创建一个与data1长度相同的数组
ax.plot(x, np.cumsum(data1), color='blue', linewidth=3, linestyle='--')  # 绘制累积数据1的折线图，设置颜色为蓝色，线宽为3，线型为虚线
l, = ax.plot(x, np.cumsum(data2), color='orange', linewidth=2)  # 绘制累积数据2的折线图，设置颜色为橙色，线宽为2
l.set_linestyle(':')  # 设置第二条线的线型为点线

# %%
# Colors
# ------
#
# Matplotlib has a very flexible array of colors that are accepted for most
# Artists; see :ref:`allowable color definitions <colors_def>` for a
# list of specifications. Some Artists will take multiple colors.  i.e. for
# a `~.Axes.scatter` plot, the edge of the markers can be different colors
# from the interior:

fig, ax = plt.subplots(figsize=(5, 2.7))  # 创建一个大小为5x2.7英寸的图形窗口和Axes对象
ax.scatter(data1, data2, s=50, facecolor='C0', edgecolor='k')  # 绘制散点图，设置点的大小为50，填充颜色为'C0'，边缘颜色为黑色

# %%
# Linewidths, linestyles, and markersizes
# ---------------------------------------
#
# Line widths are typically in typographic points (1 pt = 1/72 inch) and
# available for Artists that have stroked lines.  Similarly, stroked lines
# can have a linestyle.  See the :doc:`linestyles example
# </gallery/lines_bars_and_markers/linestyles>`.
#
# Marker size depends on the method being used.  `~.Axes.plot` specifies
# markersize in points, and is generally the "diameter" or width of the
# marker.  `~.Axes.scatter` specifies markersize as approximately
# proportional to the visual area of the marker.  There is an array of
# markerstyles available as string codes (see :mod:`~.matplotlib.markers`), or
# users can define their own `~.MarkerStyle` (see
# :doc:`/gallery/lines_bars_and_markers/marker_reference`):

fig, ax = plt.subplots(figsize=(5, 2.7))  # 创建一个大小为5x2.7英寸的图形窗口和Axes对象
ax.plot(data1, 'o', label='data1')  # 绘制数据1的折线图，点的样式为圆圈'o'，设置标签为'data1'
ax.plot(data2, 'd', label='data2')  # 绘制数据2的折线图，点的样式为菱形'd'，设置标签为'data2'
ax.plot(data3, 'v', label='data3')  # 绘制数据3的折线图，点的样式为倒三角'v'，设置标签为'data3'
ax.plot(data4, 's', label='data4')  # 绘制数据4的折线图，点的样式为正方形's'，设置标签为'data4'
ax.legend()  # 显示图例

# %%
#
# Labelling plots
# ===============
#
# Axes labels and text
# --------------------
#
# `~.Axes.set_xlabel`, `~.Axes.set_ylabel`, and `~.Axes.set_title` are used to
# add text in the indicated locations (see :ref:`text_intro`
# for more discussion).  Text can also be directly added to plots using
# `~.Axes.text`:

mu, sigma = 115, 15
x = mu + sigma * np.random.randn(10000)
fig, ax = plt.subplots(figsize=(5, 2.7))  # 创建一个大小为5x2.7英寸的图形窗口和Axes对象
# the histogram of the data
n, bins, patches = ax.hist(x, 50, density=True, facecolor='C0', alpha=0.75)  # 绘制直方图，设置50个柱子，密度为True，填充颜色为'C0'，透明度为0.75

ax.set_xlabel('Length [cm]')  # 设置x轴标签为'Length [cm]'
ax.set_ylabel('Probability')  # 设置y轴标签为'Probability'
ax.set_title('Aardvark lengths\n (not really)')  # 设置图表标题为'Aardvark lengths\n (not really)'
ax.text(75, .025, r'$\mu=115,\ \sigma=15$')  # 在坐标(75, .025)处添加文本，显示公式'μ=115, σ=15'
ax.axis([55, 175, 0, 0.03])  # 设置坐标轴范围，x轴范围为55到175，y轴范围为0到0.03
ax.grid(True)  # 显示网格线

# %%
# All of the `~.Axes.text` functions return a `matplotlib.text.Text`
# instance.  Just as with lines above, you can customize the properties by
# passing keyword arguments into the text functions::
#
#   t = ax.set_xlabel('my data', fontsize=14, color='red')
#
# 这些属性在 :ref:`text_props` 中有更详细的介绍。
#
# Using mathematical expressions in text
# --------------------------------------
#
# Matplotlib 在任何文本表达式中接受 TeX 方程表达式。
# 例如，要在标题中写入表达式 :math:`\sigma_i=15`，可以使用 TeX 表达式，用美元符号包围::
#
#     ax.set_title(r'$\sigma_i=15$')
#
# 其中标题字符串前面的 ``r`` 表示该字符串是一个“原始”字符串，不会将反斜杠视为 Python 转义符号。
# Matplotlib 有一个内置的 TeX 表达式解析器和布局引擎，并内置数学字体 - 详细信息请参见 :ref:`mathtext`。
# 您还可以直接使用 LaTeX 格式化您的文本，并直接将输出并入到您的显示图形或保存的后置文件中 - 请参见 :ref:`usetex`。
#
# Annotations
# -----------
#
# 我们还可以在绘图中注释点，通常是通过连接指向 *xy* 的箭头，并标注 *xytext* 上的文本来实现的:
fig, ax = plt.subplots(figsize=(5, 2.7))

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
line, = ax.plot(t, s, lw=2)

ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.set_ylim(-2, 2)

# %%
# 在这个基本示例中，*xy* 和 *xytext* 都是数据坐标系中的点。
# 还可以选择多种其他坐标系 - 详细信息请参见 :ref:`annotations-tutorial` 和 :ref:`plotting-guide-annotation`。
# 在 :doc:`/gallery/text_labels_and_annotations/annotation_demo` 中还可以找到更多示例。
#
# Legends
# -------
#
# 经常需要用 `.Axes.legend` 标识线条或标记:
fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(np.arange(len(data1)), data1, label='data1')
ax.plot(np.arange(len(data2)), data2, label='data2')
ax.plot(np.arange(len(data3)), data3, 'd', label='data3')
ax.legend()

# %%
# Matplotlib 中的图例在布局、放置和表示的艺术品方面非常灵活。详细讨论请参见 :ref:`legend_guide`。
#
# Axis scales and ticks
# =====================
#
# 每个 Axes 都有两个（或三个）`~.axis.Axis` 对象，代表 x 和 y 轴。这些控制轴的*刻度*、*定位器*和*格式化程序*。还可以附加额外的 Axes 来显示更多的 Axis 对象。
#
# Scales
# ------
#
# 除了线性比例尺之外，Matplotlib 还提供非线性比例尺，如对数比例尺。由于经常使用对数比例尺，因此还有像 `~.Axes.loglog`、`~.Axes.semilogx` 和 `~.Axes.semilogy` 这样的直接方法。还有许多其他比例尺类型（参见 :doc:`/gallery/scales/scales`）。在这里，我们手动设置比例尺:
fig, axs = plt.subplots(1, 2, figsize=(5, 2.7), constrained_layout=True)
xdata = np.arange(len(data1))  # 为此创建一个序数
data = 10**data1
# %%
# 在第一个子图中绘制 xdata 对应的数据，使用默认的线性比例尺
axs[0].plot(xdata, data)

# 在第二个子图中设置对数比例尺
axs[1].set_yscale('log')
# 在第二个子图中绘制 xdata 对应的数据
axs[1].plot(xdata, data)

# %%
# 比例尺设置了数据值与坐标轴上间距的映射关系。这种映射是双向的，并且会结合成一个“变换”，
# 这是 Matplotlib 用来将数据坐标映射到坐标轴、图形或屏幕坐标的方式。详见 :ref:`transforms_tutorial`。
#
# 刻度定位器和格式化器
# ----------------------------
#
# 每个坐标轴都有一个刻度定位器（locator）和格式化器（formatter），用于决定在坐标轴上放置刻度线的位置。
# `~.Axes.set_xticks` 提供了一个简单的接口：

fig, axs = plt.subplots(2, 1, layout='constrained')
# 在第一个子图中绘制 xdata 对应的 data1 数据
axs[0].plot(xdata, data1)
# 设置第一个子图的标题为 'Automatic ticks'
axs[0].set_title('Automatic ticks')

# 在第二个子图中绘制 xdata 对应的 data1 数据
axs[1].plot(xdata, data1)
# 设置第二个子图的 x 轴刻度为 0 到 100，步长为 30，并设置对应的标签
axs[1].set_xticks(np.arange(0, 100, 30), ['zero', '30', 'sixty', '90'])
# 设置第二个子图的 y 轴刻度为固定值 [-1.5, 0, 1.5]，注意这里不需要指定标签
axs[1].set_yticks([-1.5, 0, 1.5])
# 设置第二个子图的标题为 'Manual ticks'
axs[1].set_title('Manual ticks')

# %%
# 不同的比例尺可以有不同的定位器和格式化器；例如上面的对数比例尺使用了 `~.LogLocator` 和 `~.LogFormatter`。
# 参见 :doc:`/gallery/ticks/tick-locators` 和 :doc:`/gallery/ticks/tick-formatters` 查看其他格式化器和定位器，
# 以及如何编写自定义的格式化器和定位器。
#
# 绘制日期和字符串
# --------------------------
#
# Matplotlib 可以处理日期数组、字符串数组，以及浮点数。这些会根据情况获得特定的定位器和格式化器。例如日期：

from matplotlib.dates import ConciseDateFormatter

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# 生成一个日期数组，从 '2021-11-15' 到 '2021-12-25'，步长为 1 小时
dates = np.arange(np.datetime64('2021-11-15'), np.datetime64('2021-12-25'),
                  np.timedelta64(1, 'h'))
# 生成对应日期数组的随机累积数据
data = np.cumsum(np.random.randn(len(dates)))
# 在坐标轴上绘制日期与数据
ax.plot(dates, data)
# 设置 x 轴主要刻度的格式化器为 ConciseDateFormatter，使用与当前主要定位器相匹配的格式
ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))

# %%
# 更多信息请参见日期示例（例如 :doc:`/gallery/text_labels_and_annotations/date`）
#
# 对于字符串，我们得到分类变量的绘制（参见 :doc:`/gallery/lines_bars_and_markers/categorical_variables`）。

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# 定义分类变量列表
categories = ['turnips', 'rutabaga', 'cucumber', 'pumpkins']
# 绘制柱状图，横坐标为分类变量，纵坐标为随机生成的数据
ax.bar(categories, np.random.rand(len(categories)))

# %%
# 关于分类变量绘图的一个注意事项是，某些文本文件解析方法返回的是字符串列表，即使这些字符串都表示数字或日期。
# 如果传递了 1000 个字符串，Matplotlib 会认为你想要 1000 个类别，并在图表上添加 1000 个刻度线！
#
#
# 添加额外的坐标轴对象
# ------------------------
#
# 在同一个图表中绘制不同数量级的数据可能需要额外的 y 轴。可以使用 `~.Axes.twinx` 添加一个新的 Axes，
# 其中 x 轴不可见，y 轴位于右侧（`~.Axes.twiny` 类似）。参见 :doc:`/gallery/subplots_axes_and_figures/two_scales` 的另一个示例。
#
# 类似地，可以通过 `~.Axes.secondary_xaxis` 或
# 创建一个包含两个子图的图形对象，水平排列，大小为7x2.7，布局为'constrained'
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(7, 2.7), layout='constrained')

# 在第一个子图(ax1)上绘制曲线，返回曲线对象l1
l1, = ax1.plot(t, s)

# 在第一个子图(ax1)上创建一个与主坐标轴共享X轴的次坐标轴(ax2)
ax2 = ax1.twinx()

# 在次坐标轴(ax2)上绘制另一条曲线，返回曲线对象l2，使用颜色'C1'
l2, = ax2.plot(t, range(len(t)), 'C1')

# 在第一个子图(ax1)上创建图例，分别显示l1和l2的标签
ax2.legend([l1, l2], ['Sine (left)', 'Straight (right)'])

# 在第二个子图(ax3)上绘制曲线
ax3.plot(t, s)

# 设置第二个子图(ax3)的X轴标签为'Angle [rad]'
ax3.set_xlabel('Angle [rad]')

# 在第二个子图(ax3)上创建一个与原X轴共享位置但显示不同单位的次坐标轴(ax4)
ax4 = ax3.secondary_xaxis('top', (np.rad2deg, np.deg2rad))

# 设置第二个子图(ax3)的次X轴标签为'Angle [°]'
ax4.set_xlabel('Angle [°]')



# 创建一个包含四个子图的图形对象，布局为'constrained'
fig, axs = plt.subplots(2, 2, layout='constrained')

# 在第一个子图(axs[0, 0])上绘制伪彩色图，使用X, Y, Z数据，颜色映射为'RdBu_r'，值范围为-1到1
pc = axs[0, 0].pcolormesh(X, Y, Z, vmin=-1, vmax=1, cmap='RdBu_r')

# 在第一个子图(axs[0, 0])上创建颜色条
fig.colorbar(pc, ax=axs[0, 0])

# 设置第一个子图(axs[0, 0])的标题为'pcolormesh()'
axs[0, 0].set_title('pcolormesh()')

# 在第二个子图(axs[0, 1])上绘制等高线填充图，使用X, Y, Z数据，等高线间隔为-1.25到1.25之间的11个水平线
co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))

# 在第二个子图(axs[0, 1])上创建颜色条
fig.colorbar(co, ax=axs[0, 1])

# 设置第二个子图(axs[0, 1])的标题为'contourf()'
axs[0, 1].set_title('contourf()')

# 在第三个子图(axs[1, 0])上绘制图像，使用Z的平方乘以100，颜色映射为'plasma'，使用对数标准化，值范围为0.01到100
pc = axs[1, 0].imshow(Z**2 * 100, cmap='plasma', norm=LogNorm(vmin=0.01, vmax=100))

# 在第三个子图(axs[1, 0])上创建颜色条，显示标尺的延伸
fig.colorbar(pc, ax=axs[1, 0], extend='both')

# 设置第三个子图(axs[1, 0])的标题为'imshow() with LogNorm()'
axs[1, 0].set_title('imshow() with LogNorm()')

# 在第四个子图(axs[1, 1])上绘制散点图，使用data1, data2作为坐标，data3作为颜色，颜色映射为'RdBu_r'
pc = axs[1, 1].scatter(data1, data2, c=data3, cmap='RdBu_r')

# 在第四个子图(axs[1, 1])上创建颜色条，显示标尺的延伸
fig.colorbar(pc, ax=axs[1, 1], extend='both')

# 设置第四个子图(axs[1, 1])的标题为'scatter()'
axs[1, 1].set_title('scatter()')
# Working with multiple Figures and Axes
# ======================================
#
# You can open multiple Figures with multiple calls to
# ``fig = plt.figure()`` or ``fig2, ax = plt.subplots()``.  By keeping the
# object references you can add Artists to either Figure.
#
# Multiple Axes can be added a number of ways, but the most basic is
# ``plt.subplots()`` as used above.  One can achieve more complex layouts,
# with Axes objects spanning columns or rows, using `~.pyplot.subplot_mosaic`.

# 使用 plt.subplot_mosaic() 方法创建具有多个子图的图形布局，通过指定的布局参数来组织子图的位置关系
fig, axd = plt.subplot_mosaic([['upleft', 'right'],
                               ['lowleft', 'right']], layout='constrained')

# 为左上角的子图设置标题
axd['upleft'].set_title('upleft')
# 为左下角的子图设置标题
axd['lowleft'].set_title('lowleft')
# 为右侧的子图设置标题
axd['right'].set_title('right')

# %%
# Matplotlib has quite sophisticated tools for arranging Axes: See
# :ref:`arranging_axes` and :ref:`mosaic`.
#
#
# More reading
# ============
#
# For more plot types see :doc:`Plot types </plot_types/index>` and the
# :doc:`API reference </api/index>`, in particular the
# :doc:`Axes API </api/axes_api>`.

# Matplotlib 提供了丰富的工具用于布局和管理坐标轴：请参考 :ref:`arranging_axes` 和 :ref:`mosaic`。
#
#
# 更多阅读
# ============
#
# 若要了解更多绘图类型，请参阅 :doc:`Plot types </plot_types/index>` 和
# :doc:`API reference </api/index>`，尤其是 :doc:`Axes API </api/axes_api>`。
```