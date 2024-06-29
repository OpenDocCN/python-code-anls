# `D:\src\scipysrc\matplotlib\galleries\users_explain\text\text_intro.py`

```py
"""
==================
Text in Matplotlib
==================

Introduction to plotting and working with text in Matplotlib.

Matplotlib has extensive text support, including support for
mathematical expressions, truetype support for raster and
vector outputs, newline separated text with arbitrary
rotations, and Unicode support.

Because it embeds fonts directly in output documents, e.g., for postscript
or PDF, what you see on the screen is what you get in the hardcopy.
`FreeType <https://www.freetype.org/>`_ support
produces very nice, antialiased fonts, that look good even at small
raster sizes.  Matplotlib includes its own
:mod:`matplotlib.font_manager` (thanks to Paul Barrett), which
implements a cross platform, `W3C <https://www.w3.org/>`_
compliant font finding algorithm.

The user has a great deal of control over text properties (font size, font
weight, text location and color, etc.) with sensible defaults set in
the :ref:`rc file <customizing>`.
And significantly, for those interested in mathematical
or scientific figures, Matplotlib implements a large number of TeX
math symbols and commands, supporting :ref:`mathematical expressions
<mathtext>` anywhere in your figure.


Basic text commands
===================

The following commands are used to create text in the implicit and explicit
interfaces (see :ref:`api_interfaces` for an explanation of the tradeoffs):

=================== =================== ======================================
implicit API        explicit API        description
=================== =================== ======================================
`~.pyplot.text`     `~.Axes.text`       Add text at an arbitrary location of
                                        the `~matplotlib.axes.Axes`.

`~.pyplot.annotate` `~.Axes.annotate`   Add an annotation, with an optional
                                        arrow, at an arbitrary location of the
                                        `~matplotlib.axes.Axes`.

`~.pyplot.xlabel`   `~.Axes.set_xlabel` Add a label to the
                                        `~matplotlib.axes.Axes`\\'s x-axis.

`~.pyplot.ylabel`   `~.Axes.set_ylabel` Add a label to the
                                        `~matplotlib.axes.Axes`\\'s y-axis.

`~.pyplot.title`    `~.Axes.set_title`  Add a title to the
                                        `~matplotlib.axes.Axes`.

`~.pyplot.figtext`  `~.Figure.text`     Add text at an arbitrary location of
                                        the `.Figure`.

`~.pyplot.suptitle` `~.Figure.suptitle` Add a title to the `.Figure`.
=================== =================== ======================================

All of these functions create and return a `.Text` instance, which can be
configured with a variety of font and other properties.  The example below
shows all of these commands in action, and more detail is provided in the
sections that follow.

"""

import matplotlib.pyplot as plt
import matplotlib  # 导入matplotlib库

fig = plt.figure()  # 创建一个新的Figure对象
ax = fig.add_subplot()  # 在Figure对象上添加一个子图
fig.subplots_adjust(top=0.85)  # 调整子图在Figure中的位置，使顶部留出空间

# 设置Figure的总标题和子图的标题
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
ax.set_title('axes title')  # 设置子图的标题

ax.set_xlabel('xlabel')  # 设置子图的x轴标签
ax.set_ylabel('ylabel')  # 设置子图的y轴标签

# 设置x和y轴的显示范围为[0, 10]，而不是默认的[0, 1]
ax.axis([0, 10, 0, 10])

# 在数据坐标系中添加带有样式和边框的斜体文本
ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

# 在子图中添加带有数学公式的文本
ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

# 在子图中添加包含Unicode字符的文本
ax.text(3, 2, 'Unicode: Institut für Festkörperphysik')

# 在轴坐标系中指定位置添加带颜色的文本
ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)

# 在子图中绘制一个圆点
ax.plot([2], [1], 'o')

# 在子图中添加注释，包括箭头的样式设置
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()  # 显示绘制的图形



# Labels for x- and y-axis
# ========================
#
# 通过 `~matplotlib.axes.Axes.set_xlabel` 和 `~matplotlib.axes.Axes.set_ylabel` 方法，
# 可以简单地指定x轴和y轴的标签。



# The x- and y-labels are automatically placed so that they clear the x- and
# y-ticklabels.  Compare the plot below with that above, and note the y-label
# is to the left of the one above.



# If you want to move the labels, you can specify the *labelpad* keyword
# argument, where the value is points (1/72", the same unit used to specify
# fontsizes).



# Or, the labels accept all the `.Text` keyword arguments, including
# *position*, via which we can manually specify the label positions.  Here we
# put the xlabel to the far left of the axis.  Note, that the y-coordinate of
# this position has no effect - to adjust the y-position we need to use the
# *labelpad* keyword argument.



# All the labelling in this tutorial can be changed by manipulating the
# `matplotlib.font_manager.FontProperties` method, or by named keyword
# arguments to `~matplotlib.axes.Axes.set_xlabel`
# 导入 FontProperties 类从 matplotlib.font_manager 模块
from matplotlib.font_manager import FontProperties

# 创建 FontProperties 对象
font = FontProperties()

# 设置字体族为 serif
font.set_family('serif')

# 设置字体名称为 'Times New Roman'
font.set_name('Times New Roman')

# 设置字体样式为斜体
font.set_style('italic')

# 创建一个新的图形对象和一个轴对象
fig, ax = plt.subplots(figsize=(5, 3))

# 调整子图的底部边距和左侧边距
fig.subplots_adjust(bottom=0.15, left=0.2)

# 在轴上绘制 x1 对应的数据 y1
ax.plot(x1, y1)

# 设置 x 轴标签文本和字体大小、粗细
ax.set_xlabel('Time [s]', fontsize='large', fontweight='bold')

# 设置 y 轴标签文本和使用自定义的字体属性
ax.set_ylabel('Damped oscillation [V]', fontproperties=font)

# 显示图形
plt.show()

# %%
# 最后，我们可以在所有文本对象中使用本地 TeX 渲染，并且可以有多行文本：

# 创建一个新的图形对象和一个轴对象
fig, ax = plt.subplots(figsize=(5, 3))

# 调整子图的底部边距和左侧边距
fig.subplots_adjust(bottom=0.2, left=0.2)

# 在轴上绘制 x1 对应的数据的累积和
ax.plot(x1, np.cumsum(y1**2))

# 设置 x 轴标签文本，包含多行文本
ax.set_xlabel('Time [s] \n This was a long experiment')

# 设置 y 轴标签文本，使用 LaTeX 公式
ax.set_ylabel(r'$\int\ Y^2\ dt\ \ [V^2 s]$')

# 显示图形
plt.show()

# %%
# 标题
# ======
#
# 子图标题的设置方式与标签类似，但有一个 *loc* 关键字参数可以改变位置和对齐方式，默认为 ``loc=center``。

# 创建包含 3 个子图的图形对象和轴对象数组
fig, axs = plt.subplots(3, 1, figsize=(5, 6), tight_layout=True)

# 不同的子图标题位置
locs = ['center', 'left', 'right']
for ax, loc in zip(axs, locs):
    # 在每个子图上绘制 x1 对应的数据 y1
    ax.plot(x1, y1)
    # 设置子图标题，并指定位置和对齐方式
    ax.set_title('Title with loc at '+loc, loc=loc)

# 显示图形
plt.show()

# %%
# 标题的垂直间距由 :rc:`axes.titlepad` 控制。
# 将其设置为不同的值可以移动标题的位置。

# 创建一个新的图形对象和一个轴对象
fig, ax = plt.subplots(figsize=(5, 3))

# 调整子图的顶部边距
fig.subplots_adjust(top=0.8)

# 在轴上绘制 x1 对应的数据 y1
ax.plot(x1, y1)

# 设置标题文本和标题偏移量
ax.set_title('Vertically offset title', pad=30)

# 显示图形
plt.show()

# %%
# 刻度和刻度标签
# ====================
#
# 放置刻度和刻度标签是制作图形的一个非常棘手的部分。
# Matplotlib 尽力自动完成任务，但也提供了一个非常灵活的框架来确定刻度位置和标签。

# 创建包含 2 个子图的图形对象和轴对象数组
fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)

# 在第一个子图上绘制 x1 对应的数据 y1
axs[0].plot(x1, y1)

# 在第二个子图上绘制 x1 对应的数据 y1
axs[1].plot(x1, y1)

# 设置第二个子图 x 轴的刻度位置
axs[1].xaxis.set_ticks(np.arange(0., 8.1, 2.))

# 显示图形
plt.show()

# %%
# 当然，我们可以事后修复这个问题，但这突显了一个
# weakness of hard-coding the ticks.  This example also changes the format
# of the ticks:
# 创建一个包含两个子图的画布，每个子图为1行2列，尺寸为5x3，紧凑布局
fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
# 在第一个子图上绘制 x1 vs y1 的图像
axs[0].plot(x1, y1)
# 在第二个子图上绘制 x1 vs y1 的图像
axs[1].plot(x1, y1)
# 使用 np.arange 创建一个从 0 到 8.1（不包括）的数组，步长为2
ticks = np.arange(0., 8.1, 2.)
# 使用列表推导式生成所有刻度标签的列表，保留两位小数
tickla = [f'{tick:1.2f}' for tick in ticks]
# 设置第二个子图的 x 轴刻度位置
axs[1].xaxis.set_ticks(ticks)
# 设置第二个子图的 x 轴刻度标签
axs[1].xaxis.set_ticklabels(tickla)
# 设置第二个子图的 x 轴限制与第一个子图相同
axs[1].set_xlim(axs[0].get_xlim())
# 显示图形
plt.show()

# %%
# Tick Locators and Formatters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Instead of making a list of all the ticklabels, we could have
# used `matplotlib.ticker.StrMethodFormatter` (new-style ``str.format()``
# format string) or `matplotlib.ticker.FormatStrFormatter` (old-style '%'
# format string) and passed it to the ``ax.xaxis``.  A
# `matplotlib.ticker.StrMethodFormatter` can also be created by passing a
# ``str`` without having to explicitly create the formatter.

# 创建一个包含两个子图的画布，每个子图为1行2列，尺寸为5x3，紧凑布局
fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
# 在第一个子图上绘制 x1 vs y1 的图像
axs[0].plot(x1, y1)
# 在第二个子图上绘制 x1 vs y1 的图像
axs[1].plot(x1, y1)
# 使用 np.arange 创建一个从 0 到 8.1（不包括）的数组，步长为2
ticks = np.arange(0., 8.1, 2.)
# 设置第二个子图的 x 轴刻度位置
axs[1].xaxis.set_ticks(ticks)
# 设置第二个子图的 x 轴主要格式化字符串为 '{x:1.1f}'
axs[1].xaxis.set_major_formatter('{x:1.1f}')
# 设置第二个子图的 x 轴限制与第一个子图相同
axs[1].set_xlim(axs[0].get_xlim())
# 显示图形
plt.show()

# %%
# And of course we could have used a non-default locator to set the
# tick locations.  Note we still pass in the tick values, but the
# x-limit fix used above is *not* needed.

# 创建一个包含两个子图的画布，每个子图为1行2列，尺寸为5x3，紧凑布局
fig, axs = plt.subplots(2, 1, figsize=(5, 3), tight_layout=True)
# 在第一个子图上绘制 x1 vs y1 的图像
axs[0].plot(x1, y1)
# 在第二个子图上绘制 x1 vs y1 的图像
axs[1].plot(x1, y1)
# 使用 matplotlib.ticker.FixedLocator 创建一个固定刻度位置的定位器对象
locator = matplotlib.ticker.FixedLocator(ticks)
# 设置第二个子图的 x 轴主要定位器
axs[1].xaxis.set_major_locator(locator)
# 设置第二个子图的 x 轴主要格式化字符串为 '±{x}°'
axs[1].xaxis.set_major_formatter('±{x}°')
# 显示图形
plt.show()

# %%
# The default formatter is the `matplotlib.ticker.MaxNLocator` called as
# ``ticker.MaxNLocator(self, nbins='auto', steps=[1, 2, 2.5, 5, 10])``
# The *steps* keyword contains a list of multiples that can be used for
# tick values.  i.e. in this case, 2, 4, 6 would be acceptable ticks,
# as would 20, 40, 60 or 0.2, 0.4, 0.6. However, 3, 6, 9 would not be
# acceptable because 3 doesn't appear in the list of steps.
#
# ``nbins=auto`` uses an algorithm to determine how many ticks will
# be acceptable based on how long the axis is.  The fontsize of the
# ticklabel is taken into account, but the length of the tick string
# is not (because it's not yet known.)  In the bottom row, the
# ticklabels are quite large, so we set ``nbins=4`` to make the
# labels fit in the right-hand plot.

# 创建一个包含四个子图的画布，每个子图为2行2列，尺寸为8x5，紧凑布局
fig, axs = plt.subplots(2, 2, figsize=(8, 5), tight_layout=True)
# 对所有子图循环，每个子图绘制 x1*10 vs y1 的图像
for n, ax in enumerate(axs.flat):
    ax.plot(x1*10., y1)
# 使用 matplotlib.ticker.FormatStrFormatter 创建一个格式化字符串格式化器
formatter = matplotlib.ticker.FormatStrFormatter('%1.1f')
# 使用 matplotlib.ticker.MaxNLocator 创建一个最大数量定位器，nbins 设置为 'auto'，步长设置为 [1, 4, 10]
locator = matplotlib.ticker.MaxNLocator(nbins='auto', steps=[1, 4, 10])
# 设置第一个子图右边的 x 轴主要定位器和主要格式化器
axs[0, 1].xaxis.set_major_locator(locator)
axs[0, 1].xaxis.set_major_formatter(formatter)

# 使用 matplotlib.ticker.AutoLocator 创建一个自动定位器
locator = matplotlib.ticker.AutoLocator()
# 设置第二行第一列子图的 x 轴主要格式化字符串为 '%1.5f'
axs[1, 0].xaxis.set_major_formatter('%1.5f')
# 设置第二行第一列子图的 x 轴主要定位器
axs[1, 0].xaxis.set_major_locator(locator)

# 使用 matplotlib.ticker.FormatStrFormatter 创建一个格式化字符串格式化器
formatter = matplotlib.ticker.FormatStrFormatter('%1.5f')
locator = matplotlib.ticker.MaxNLocator(nbins=4)
# 创建一个最大刻度定位器对象，指定刻度的数量为4个

axs[1, 1].xaxis.set_major_formatter(formatter)
# 设置子图 axs[1, 1] 的 x 轴主要刻度的格式化方式为预先定义的 formatter

axs[1, 1].xaxis.set_major_locator(locator)
# 设置子图 axs[1, 1] 的 x 轴主要刻度的定位器为之前创建的 locator

plt.show()
# 显示图形

# %%
# 最后，我们可以使用 `matplotlib.ticker.FuncFormatter` 来指定格式化函数。
# 此外，类似于 `matplotlib.ticker.StrMethodFormatter`，传递一个函数将自动创建一个 `matplotlib.ticker.FuncFormatter`。

def formatoddticks(x, pos):
    """格式化奇数位置的刻度。"""
    if x % 2:
        return f'{x:1.2f}'  # 返回保留两位小数的格式化字符串
    else:
        return ''  # 返回空字符串，即偶数位置的刻度不显示

fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
ax.plot(x1, y1)
locator = matplotlib.ticker.MaxNLocator(nbins=6)
ax.xaxis.set_major_formatter(formatoddticks)
ax.xaxis.set_major_locator(locator)
# 创建新的图形和子图，绘制数据，并设置 x 轴的主要定位器和格式化器

plt.show()

# %%
# 日期刻度
# ^^^^^^^^^
#
# Matplotlib 可以接受 `datetime.datetime` 和 `numpy.datetime64` 对象作为绘图参数。
# 日期和时间需要特殊的格式化，通常需要手动干预。为了帮助这一点，日期有特殊的定位器和格式化器，
# 在 `matplotlib.dates` 模块中定义。
#
# 以下是一个简单的例子。注意如何旋转刻度标签，以防止它们互相重叠。

import datetime

fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
base = datetime.datetime(2017, 1, 1, 0, 0, 1)
time = [base + datetime.timedelta(days=x) for x in range(len(x1))]

ax.plot(time, y1)
ax.tick_params(axis='x', rotation=70)
# 创建新的图形和子图，绘制时间数据，并旋转 x 轴的刻度标签

plt.show()

# %%
# 我们可以向 `matplotlib.dates.DateFormatter` 传递格式。还要注意，29日和下个月非常接近。
# 我们可以使用 `.dates.DayLocator` 类修复这个问题，它允许我们指定要使用的月份中的日期列表。
# 类似的格式化器在 `matplotlib.dates` 模块中列出。

import matplotlib.dates as mdates

locator = mdates.DayLocator(bymonthday=[1, 15])
formatter = mdates.DateFormatter('%b %d')

fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.plot(time, y1)
ax.tick_params(axis='x', rotation=70)
# 创建新的图形和子图，设置 x 轴的主要定位器和格式化器，并绘制时间数据

plt.show()

# %%
# 图例和注释
# =======================
#
# - 图例: :ref:`legend_guide`
# - 注释: :ref:`annotations`
#
```