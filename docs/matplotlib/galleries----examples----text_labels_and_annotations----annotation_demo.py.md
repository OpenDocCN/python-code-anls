# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\annotation_demo.py`

```
"""
================
Annotating Plots
================

The following examples show ways to annotate plots in Matplotlib.
This includes highlighting specific points of interest and using various
visual tools to call attention to this point. For a more complete and in-depth
description of the annotation and text tools in Matplotlib, see the
:ref:`tutorial on annotation <annotations>`.
"""

# 导入 Matplotlib 的 pyplot 模块，并将其重命名为 plt
import matplotlib.pyplot as plt
# 导入 NumPy 模块，并将其重命名为 np
import numpy as np
# 从 Matplotlib 的 patches 模块中导入 Ellipse 类
from matplotlib.patches import Ellipse
# 从 Matplotlib 的 text 模块中导入 OffsetFrom 类
from matplotlib.text import OffsetFrom

# %%
# Specifying text points and annotation points
# --------------------------------------------
#
# You must specify an annotation point ``xy=(x, y)`` to annotate this point.
# Additionally, you may specify a text point ``xytext=(x, y)`` for the location
# of the text for this annotation.  Optionally, you can specify the coordinate
# system of *xy* and *xytext* with one of the following strings for *xycoords*
# and *textcoords* (default is 'data')::
#
#  'figure points'   : points from the lower left corner of the figure
#  'figure pixels'   : pixels from the lower left corner of the figure
#  'figure fraction' : (0, 0) is lower left of figure and (1, 1) is upper right
#  'axes points'     : points from lower left corner of the Axes
#  'axes pixels'     : pixels from lower left corner of the Axes
#  'axes fraction'   : (0, 0) is lower left of Axes and (1, 1) is upper right
#  'offset points'   : Specify an offset (in points) from the xy value
#  'offset pixels'   : Specify an offset (in pixels) from the xy value
#  'data'            : use the Axes data coordinate system
#
# Note: for physical coordinate systems (points or pixels) the origin is the
# (bottom, left) of the figure or Axes.
#
# Optionally, you can specify arrow properties which draws and arrow
# from the text to the annotated point by giving a dictionary of arrow
# properties
#
# Valid keys are::
#
#   width : the width of the arrow in points
#   frac  : the fraction of the arrow length occupied by the head
#   headwidth : the width of the base of the arrow head in points
#   shrink : move the tip and base some percent away from the
#            annotated point and text
#   any key for matplotlib.patches.polygon  (e.g., facecolor)

# Create our figure and data we'll use for plotting
# 创建一个图形和我们用于绘图的数据
fig, ax = plt.subplots(figsize=(4, 4))

# Generate some data points for plotting
# 生成用于绘图的数据点
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)

# Plot a line and add some simple annotations
# 绘制一条线，并添加一些简单的注释
line, = ax.plot(t, s)

# Annotate at a specific point in figure pixels
# 在图像像素的特定点进行注释
ax.annotate('figure pixels',
            xy=(10, 10), xycoords='figure pixels')

# Annotate at a specific point in figure points with specified font size
# 在图像点的特定点进行注释，并指定字体大小
ax.annotate('figure points',
            xy=(107, 110), xycoords='figure points',
            fontsize=12)

# Annotate at a specific fraction of the figure coordinates
# 在图像坐标的特定分数处进行注释
ax.annotate('figure fraction',
            xy=(.025, .975), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=20)

# The following examples show off how these arrows are drawn.
# 下面的示例展示了箭头是如何绘制的。
# 在图形中添加注释，指定文本和箭头的位置以及样式
ax.annotate('point offset from data',  # 注释文本内容
            xy=(3, 1),                # 箭头指向的数据点坐标
            xycoords='data',          # 数据点的坐标系
            xytext=(-10, 90),         # 文本的偏移量
            textcoords='offset points',  # 文本坐标的类型为偏移点
            arrowprops=dict(facecolor='black', shrink=0.05),  # 箭头的属性
            horizontalalignment='center', verticalalignment='bottom')  # 文本的水平和垂直对齐方式

ax.annotate('axes fraction',
            xy=(2, 1),                # 箭头指向的数据点坐标
            xycoords='data',          # 数据点的坐标系
            xytext=(0.36, 0.68),      # 文本的坐标位置
            textcoords='axes fraction',  # 文本坐标的类型为轴分数
            arrowprops=dict(facecolor='black', shrink=0.05),  # 箭头的属性
            horizontalalignment='right', verticalalignment='top')  # 文本的水平和垂直对齐方式

# 使用负的点数或像素指定从 (右侧，顶部) 的位置偏移量。
# 例如，(-10, 10) 指的是距离 Axes 右侧 10 点，并且距离底部 10 点。
ax.annotate('pixel offset from axes fraction',
            xy=(1, 0),                # 箭头指向的数据点坐标
            xycoords='axes fraction',  # 数据点的坐标系
            xytext=(-20, 20),         # 文本的偏移量
            textcoords='offset pixels',  # 文本坐标的类型为偏移像素
            horizontalalignment='right',  # 文本的水平对齐方式
            verticalalignment='bottom')  # 文本的垂直对齐方式

ax.set(xlim=(-1, 5), ylim=(-3, 5))


# %%
# 使用多个坐标系统和轴类型
# ------------------------------------------------
#
# 您可以在不同位置和坐标系统中指定 *xy* 点和 *xytext*，并可选择打开连接线并用标记标记点。
# 注释也适用于极坐标轴。
#
# 在下面的示例中，*xy* 点在本地坐标中（*xycoords* 默认为 'data'）。对于极坐标轴，这是 (theta, radius) 空间中的坐标。
# 文本放置在图形坐标系中。文本关键字参数如水平和垂直对齐方式会被尊重。
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(3, 3))
r = np.arange(0, 1, 0.001)
theta = 2*2*np.pi*r
line, = ax.plot(theta, r)

ind = 800
thisr, thistheta = r[ind], theta[ind]
ax.plot([thistheta], [thisr], 'o')
ax.annotate('a polar annotation',
            xy=(thistheta, thisr),    # theta, radius
            xytext=(0.05, 0.05),      # 分数，分数
            textcoords='figure fraction',  # 文本坐标的类型为图形分数
            arrowprops=dict(facecolor='black', shrink=0.05),  # 箭头的属性
            horizontalalignment='left',  # 文本的水平对齐方式
            verticalalignment='bottom')  # 文本的垂直对齐方式

# %%
# 您还可以在笛卡尔坐标轴上使用极坐标表示法。在这里，本地坐标系统（'data'）是笛卡尔坐标，因此如果要使用 (theta, radius)，
# 您需要将 xycoords 和 textcoords 指定为 'polar'。
el = Ellipse((0, 0), 10, 20, facecolor='r', alpha=0.5)

fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'))
ax.add_artist(el)
el.set_clip_box(ax.bbox)
ax.annotate('the top',
            xy=(np.pi/2., 10.),      # theta, radius
            xytext=(np.pi/3, 20.),   # theta, radius
            xycoords='polar',        # 数据点的坐标系为极坐标
            textcoords='polar',      # 文本坐标的类型为极坐标
            arrowprops=dict(facecolor='black', shrink=0.05),  # 箭头的属性
            horizontalalignment='left',  # 文本的水平对齐方式
            verticalalignment='bottom',  # 文本的垂直对齐方式
            clip_on=True)  # 限制到坐标轴边界框内

ax.set(xlim=[-20, 20], ylim=[-20, 20])


# %%
# Customizing arrow and bubble styles
# -----------------------------------
#
# The arrow between *xytext* and the annotation point, as well as the bubble
# that covers the annotation text, are highly customizable. Below are a few
# parameter options as well as their resulting output.

fig, ax = plt.subplots(figsize=(8, 5))

t = np.arange(0.0, 5.0, 0.01)  # 创建一个从0到5，步长为0.01的数组
s = np.cos(2*np.pi*t)  # 计算t数组中每个元素的余弦值
line, = ax.plot(t, s, lw=3)  # 在坐标轴ax上绘制(t, s)的线条，设置线宽为3

ax.annotate(
    'straight',  # 注释文本内容为'straight'
    xy=(0, 1), xycoords='data',  # 注释位置在数据坐标系中的(0, 1)点
    xytext=(-50, 30), textcoords='offset points',  # 文本偏移量为(-50, 30)个点
    arrowprops=dict(arrowstyle="->"))  # 箭头样式为'->'

ax.annotate(
    'arc3,\nrad 0.2',  # 注释文本内容为'arc3,\nrad 0.2'
    xy=(0.5, -1), xycoords='data',  # 注释位置在数据坐标系中的(0.5, -1)点
    xytext=(-80, -60), textcoords='offset points',  # 文本偏移量为(-80, -60)个点
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3,rad=.2"))  # 箭头样式为'->'，连接样式为arc3,rad=0.2

ax.annotate(
    'arc,\nangle 50',  # 注释文本内容为'arc,\nangle 50'
    xy=(1., 1), xycoords='data',  # 注释位置在数据坐标系中的(1.0, 1.0)点
    xytext=(-90, 50), textcoords='offset points',  # 文本偏移量为(-90, 50)个点
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc,angleA=0,armA=50,rad=10"))  # 箭头样式为'->'，连接样式为arc,angleA=0,armA=50,rad=10

ax.annotate(
    'arc,\narms',  # 注释文本内容为'arc,\narms'
    xy=(1.5, -1), xycoords='data',  # 注释位置在数据坐标系中的(1.5, -1)点
    xytext=(-80, -60), textcoords='offset points',  # 文本偏移量为(-80, -60)个点
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc,angleA=0,armA=40,angleB=-90,armB=30,rad=7"))  # 箭头样式为'->'，连接样式为arc,angleA=0,armA=40,angleB=-90,armB=30,rad=7

ax.annotate(
    'angle,\nangle 90',  # 注释文本内容为'angle,\nangle 90'
    xy=(2., 1), xycoords='data',  # 注释位置在数据坐标系中的(2.0, 1.0)点
    xytext=(-70, 30), textcoords='offset points',  # 文本偏移量为(-70, 30)个点
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle,angleA=0,angleB=90,rad=10"))  # 箭头样式为'->'，连接样式为angle,angleA=0,angleB=90,rad=10

ax.annotate(
    'angle3,\nangle -90',  # 注释文本内容为'angle3,\nangle -90'
    xy=(2.5, -1), xycoords='data',  # 注释位置在数据坐标系中的(2.5, -1)点
    xytext=(-80, -60), textcoords='offset points',  # 文本偏移量为(-80, -60)个点
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle3,angleA=0,angleB=-90"))  # 箭头样式为'->'，连接样式为angle3,angleA=0,angleB=-90

ax.annotate(
    'angle,\nround',  # 注释文本内容为'angle,\nround'
    xy=(3., 1), xycoords='data',  # 注释位置在数据坐标系中的(3.0, 1.0)点
    xytext=(-60, 30), textcoords='offset points',  # 文本偏移量为(-60, 30)个点
    bbox=dict(boxstyle="round", fc="0.8"),  # 注释框样式为圆角，填充颜色为0.8
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle,angleA=0,angleB=90,rad=10"))  # 箭头样式为'->'，连接样式为angle,angleA=0,angleB=90,rad=10

ax.annotate(
    'angle,\nround4',  # 注释文本内容为'angle,\nround4'
    xy=(3.5, -1), xycoords='data',  # 注释位置在数据坐标系中的(3.5, -1)点
    xytext=(-70, -80), textcoords='offset points',  # 文本偏移量为(-70, -80)个点
    size=20,  # 文本大小为20
    bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),  # 注释框样式为圆角4，填充颜色为0.8
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="angle,angleA=0,angleB=-90,rad=10"))  # 箭头样式为'->'，连接样式为angle,angleA=0,angleB=-90,rad=10

ax.annotate(
    'angle,\nshrink',  # 注释文本内容为'angle,\nshrink'
    xy=(4., 1), xycoords='data',  # 注释位置在数据坐标系中的(4.0, 1.0)点
    xytext=(-60, 30), textcoords='offset points',  # 文本偏移量为(-60, 30)个点
    bbox=dict(boxstyle="round", fc="0.8"),  # 注释框样式为圆角，填充颜色为0.8
    arrowprops=dict(arrowstyle="->",
                    shrinkA=0, shrinkB=10,  # 箭头起始和结束处不收缩
                    connectionstyle="angle,angleA=0,angleB=90,rad=10"))  # 箭头样式为'->'，连接样式为angle,angleA=0,angleB=90,rad=10

# You can pass an empty string to get only annotation arrows rendered
ax.annotate('', xy=(4., 1.), xycoords='data',  # 注释位置在数据坐标系中的(4.0, 1.0)点
            xytext=(4.5, -1), textcoords='data',  # 文本偏移量为(4.5, -1)个点
            arrowprops=dict(arrowstyle="<->",  # 箭头样式为'<->'
                            connectionstyle="bar",  # 连接样式为bar
                            ec="k",  # 箭头边缘颜色为黑色
                            shrinkA=5, shrinkB=5))  # 箭头起始和结束处收缩5个点

ax.set(xlim=(-1, 5), ylim=(-4, 3))  # 设置坐标轴范围

# %%
# 创建一个新的图形和轴对象
fig, ax = plt.subplots()

# 创建一个椭圆对象，并将其添加到轴对象上
el = Ellipse((2, -1), 0.5, 0.5)
ax.add_patch(el)

# 添加第一个注释，带箭头的文本标记
ax.annotate('$->$',
            xy=(2., -1), xycoords='data',  # 标记位置在数据坐标系下的 (2, -1)
            xytext=(-150, -140), textcoords='offset points',  # 文本位置相对于标记位置的偏移量
            bbox=dict(boxstyle="round", fc="0.8"),  # 文本框样式为圆角，填充颜色为灰度 0.8
            arrowprops=dict(arrowstyle="->",  # 箭头样式为朝右
                            patchB=el,  # 连接箭头的对象为前面创建的椭圆对象 el
                            connectionstyle="angle,angleA=90,angleB=0,rad=10"))  # 连接样式为角度连接

# 添加第二个注释，带箭头的文本标记，箭头样式为 fancy
ax.annotate('arrow\nfancy',
            xy=(2., -1), xycoords='data',
            xytext=(-100, 60), textcoords='offset points',
            size=20,
            arrowprops=dict(arrowstyle="fancy",
                            fc="0.6", ec="none",
                            patchB=el,
                            connectionstyle="angle3,angleA=0,angleB=-90"))

# 添加第三个注释，带箭头的文本标记，箭头样式为 simple
ax.annotate('arrow\nsimple',
            xy=(2., -1), xycoords='data',
            xytext=(100, 60), textcoords='offset points',
            size=20,
            arrowprops=dict(arrowstyle="simple",
                            fc="0.6", ec="none",
                            patchB=el,
                            connectionstyle="arc3,rad=0.3"))

# 添加第四个注释，带箭头的文本标记，箭头样式为 wedge
ax.annotate('wedge',
            xy=(2., -1), xycoords='data',
            xytext=(-100, -100), textcoords='offset points',
            size=20,
            arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                            fc="0.6", ec="none",
                            patchB=el,
                            connectionstyle="arc3,rad=-0.3"))

# 添加第五个注释，带箭头的文本标记，箭头样式为 wedge，带有 bubble 效果
ax.annotate('bubble,\ncontours',
            xy=(2., -1), xycoords='data',
            xytext=(0, -70), textcoords='offset points',
            size=20,
            bbox=dict(boxstyle="round",  # 文本框样式为圆角矩形
                      fc=(1.0, 0.7, 0.7),  # 填充颜色为 RGB (1.0, 0.7, 0.7)
                      ec=(1., .5, .5)),  # 边框颜色为 RGB (1.0, 0.5, 0.5)
            arrowprops=dict(arrowstyle="wedge,tail_width=1.",  # 箭头样式为楔形，尾部宽度为1
                            fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),  # 箭头填充和边框颜色
                            patchA=None,  # 不连接其他对象
                            patchB=el,  # 连接箭头的对象为 el
                            relpos=(0.2, 0.8),  # 相对位置
                            connectionstyle="arc3,rad=-0.1"))  # 连接样式为圆弧

# 添加第六个注释，带箭头的文本标记，箭头样式为 wedge
ax.annotate('bubble',
            xy=(2., -1), xycoords='data',
            xytext=(55, 0), textcoords='offset points',
            size=20, va="center",  # 垂直对齐方式为中心
            bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),  # 文本框样式为圆角矩形
            arrowprops=dict(arrowstyle="wedge,tail_width=1.",  # 箭头样式为楔形，尾部宽度为1
                            fc=(1.0, 0.7, 0.7), ec="none",  # 箭头填充颜色和无边框
                            patchA=None,  # 不连接其他对象
                            patchB=el,  # 连接箭头的对象为 el
                            relpos=(0.2, 0.5)))  # 相对位置

# 设置轴的 x 和 y 范围
ax.set(xlim=(-1, 5), ylim=(-5, 3))
# 在这里我们将展示坐标系的范围及如何放置注释文本。

ax1.annotate('figure fraction : 0, 0', xy=(0, 0), xycoords='figure fraction',
             xytext=(20, 20), textcoords='offset points',
             ha="left", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args)

ax1.annotate('figure fraction : 1, 1', xy=(1, 1), xycoords='figure fraction',
             xytext=(-20, -20), textcoords='offset points',
             ha="right", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args)

ax1.annotate('axes fraction : 0, 0', xy=(0, 0), xycoords='axes fraction',
             xytext=(20, 20), textcoords='offset points',
             ha="left", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args)

ax1.annotate('axes fraction : 1, 1', xy=(1, 1), xycoords='axes fraction',
             xytext=(-20, -20), textcoords='offset points',
             ha="right", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args)

# 这里演示了可生成可拖动注释的功能

an1 = ax1.annotate('Drag me 1', xy=(.5, .7), xycoords='data',
                   ha="center", va="center",
                   bbox=bbox_args)

an2 = ax1.annotate('Drag me 2', xy=(.5, .5), xycoords=an1,
                   xytext=(.5, .3), textcoords='axes fraction',
                   ha="center", va="center",
                   bbox=bbox_args,
                   arrowprops=dict(patchB=an1.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args))
an1.draggable()
an2.draggable()

an3 = ax1.annotate('', xy=(.5, .5), xycoords=an2,
                   xytext=(.5, .5), textcoords=an1,
                   ha="center", va="center",
                   bbox=bbox_args,
                   arrowprops=dict(patchA=an1.get_bbox_patch(),
                                   patchB=an2.get_bbox_patch(),
                                   connectionstyle="arc3,rad=0.2",
                                   **arrow_args))

# 最后我们展示一些更复杂的注释和放置方式

text = ax2.annotate('xy=(0, 1)\nxycoords=("data", "axes fraction")',
                    xy=(0, 1), xycoords=("data", 'axes fraction'),
                    xytext=(0, -20), textcoords='offset points',
                    ha="center", va="top",
                    bbox=bbox_args,
                    arrowprops=arrow_args)

ax2.annotate('xy=(0.5, 0)\nxycoords=artist',
             xy=(0.5, 0.), xycoords=text,
             xytext=(0, -20), textcoords='offset points',
             ha="center", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args)
# 在 ax2 上添加注释文本，标注在 ax1 的数据坐标系中位置为 (0.8, 0.5)
ax2.annotate('xy=(0.8, 0.5)\nxycoords=ax1.transData',
             xy=(0.8, 0.5), xycoords=ax1.transData,
             xytext=(10, 10),  # 注释文本的偏移量设置为 (10, 10)
             textcoords=OffsetFrom(ax2.bbox, (0, 0), "points"),  # 文本的坐标系设置为相对于 ax2 的边界框的偏移
             ha="left", va="bottom",  # 水平对齐方式为左对齐，垂直对齐方式为底部对齐
             bbox=bbox_args,  # 注释框的属性参数
             arrowprops=arrow_args)  # 箭头的属性参数设置

# 设置 ax2 的 x 和 y 轴限制范围为 [-2, 2]，用于显示范围设置
ax2.set(xlim=[-2, 2], ylim=[-2, 2])

# 显示绘图结果
plt.show()
```