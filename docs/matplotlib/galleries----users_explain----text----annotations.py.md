# `D:\src\scipysrc\matplotlib\galleries\users_explain\text\annotations.py`

```py
r"""
.. redirect-from:: /gallery/userdemo/annotate_simple01
.. redirect-from:: /gallery/userdemo/annotate_simple02
.. redirect-from:: /gallery/userdemo/annotate_simple03
.. redirect-from:: /gallery/userdemo/annotate_simple04
.. redirect-from:: /gallery/userdemo/anchored_box04
.. redirect-from:: /gallery/userdemo/annotate_simple_coord01
.. redirect-from:: /gallery/userdemo/annotate_simple_coord02
.. redirect-from:: /gallery/userdemo/annotate_simple_coord03
.. redirect-from:: /gallery/userdemo/connect_simple01
.. redirect-from:: /tutorials/text/annotations

.. _annotations:

Annotations
===========

Annotations are graphical elements, often pieces of text, that explain, add
context to, or otherwise highlight some portion of the visualized data.
`~.Axes.annotate` supports a number of coordinate systems for flexibly
positioning data and annotations relative to each other and a variety of
options of for styling the text. Axes.annotate also provides an optional arrow
from the text to the data and this arrow can be styled in various ways.
`~.Axes.text` can also be used for simple text annotation, but does not
provide as much flexibility in positioning and styling as `~.Axes.annotate`.

.. contents:: Table of Contents
   :depth: 3
"""
# %%
# .. _annotations-tutorial:
#
# Basic annotation
# ----------------
#
# In an annotation, there are two points to consider: the location of the data
# being annotated *xy* and the location of the annotation text *xytext*.  Both
# of these arguments are ``(x, y)`` tuples:

# 导入matplotlib.pyplot和numpy库
import matplotlib.pyplot as plt
import numpy as np

# 创建一个3x3尺寸的图形对象和对应的轴对象
fig, ax = plt.subplots(figsize=(3, 3))

# 创建数据
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)

# 绘制数据曲线
line, = ax.plot(t, s, lw=2)

# 在图上添加注释，'local max'是注释文本，xy=(2, 1)是箭头的位置，xytext=(3, 1.5)是文本的位置
ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05))

# 设置y轴的范围
ax.set_ylim(-2, 2)

# %%
# In this example, both the *xy* (arrow tip) and *xytext* locations
# (text location) are in data coordinates.  There are a variety of other
# coordinate systems one can choose -- you can specify the coordinate
# system of *xy* and *xytext* with one of the following strings for
# *xycoords* and *textcoords* (default is 'data')
#
# ==================  ========================================================
# argument            coordinate system
# ==================  ========================================================
# 'figure points'     points from the lower left corner of the figure
# 'figure pixels'     pixels from the lower left corner of the figure
# 'figure fraction'   (0, 0) is lower left of figure and (1, 1) is upper right
# 'axes points'       points from lower left corner of the Axes
# 'axes pixels'       pixels from lower left corner of the Axes
# 'axes fraction'     (0, 0) is lower left of Axes and (1, 1) is upper right
# 'data'              use the axes data coordinate system
# ==================  ========================================================
#
# The following strings are also valid arguments for *textcoords*
#
# ==================  ========================================================
# argument            coordinate system
# ==================  ========================================================
# 'offset points'     以点为单位的偏移量，相对于xy值
# 'offset pixels'     以像素为单位的偏移量，相对于xy值
# ==================  ========================================================
#
# 对于物理坐标系统（点或像素），原点位于图形或Axes的左下角。点是
# `排版点 <https://en.wikipedia.org/wiki/Point_(typography)>`_
# 这意味着它们是一个物理单位，等于1/72英寸。点和像素在 :ref:`transforms-fig-scale-dpi` 中有更详细的讨论。
#
# .. _annotation-data:
#
# 标注数据
# ^^^^^^^^^^^^^^^
#
# 本示例将文本坐标放置在分数轴坐标中：

fig, ax = plt.subplots(figsize=(3, 3))

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = ax.plot(t, s, lw=2)

ax.annotate('local max', xy=(2, 1), xycoords='data',
            xytext=(0.01, .99), textcoords='axes fraction',
            va='top', ha='left',
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.set_ylim(-2, 2)

# %%
#
# 标注一个Artist
# ^^^^^^^^^^^^^^^^^^^^
#
# 通过将Artist实例作为*xycoords*传入，可以相对于该Artist实例定位标注。
# 这时，*xy*被解释为Artist边界框的分数。

import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(3, 3))
arr = mpatches.FancyArrowPatch((1.25, 1.5), (1.75, 1.5),
                               arrowstyle='->,head_width=.15', mutation_scale=20)
ax.add_patch(arr)
ax.annotate("label", (.5, .5), xycoords=arr, ha='center', va='bottom')
ax.set(xlim=(1, 2), ylim=(1, 2))

# %%
# 这里的标注位置是相对于箭头左下角的(.5, .5)，并且在垂直和水平方向上都基于该位置。
# 垂直方向上，底部对齐到参考点，使标签位于线的上方。有关链式标注Artist的示例，请参见
# :ref:`Artist section <artist_annotation_coord>` 的
# :ref:`annotating_coordinate_systems`。
#
#
# .. _annotation-with-arrow:
#
# 使用箭头标注
# ^^^^^^^^^^^^^^^^^^^^^^
#
# 可以通过在可选关键字参数*arrowprops*中提供箭头属性字典来启用从文本到标注点的箭头绘制。
#
# ==================== =====================================================
# *arrowprops* 键      描述
# ==================== =====================================================
# width                箭头的宽度，以点为单位
# frac                 箭头长度中箭头头部所占的比例
# headwidth            箭头头部在点中的基宽度
# shrink               将箭头的尖端和基部移动到标注点和文本的一定百分比处
#
# **kwargs           any key for :class:`matplotlib.patches.Polygon`,
#                      e.g., ``facecolor``
# ==================== =====================================================
#
# In the example below, the *xy* point is in the data coordinate system
# since *xycoords* defaults to 'data'. For a polar Axes, this is in
# (theta, radius) space. The text in this example is placed in the
# fractional figure coordinate system. :class:`matplotlib.text.Text`
# keyword arguments like *horizontalalignment*, *verticalalignment* and
# *fontsize* are passed from `~matplotlib.axes.Axes.annotate` to the
# ``Text`` instance.

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
r = np.arange(0, 1, 0.001)
theta = 2 * 2*np.pi * r
line, = ax.plot(theta, r, color='#ee8d18', lw=3)

ind = 800
thisr, thistheta = r[ind], theta[ind]
ax.plot([thistheta], [thisr], 'o')
ax.annotate('a polar annotation',
            xy=(thistheta, thisr),  # theta, radius
            xytext=(0.05, 0.05),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='bottom')

# %%
# For more on plotting with arrows, see :ref:`annotation_with_custom_arrow`
#
# .. _annotations-offset-text:
#
# Placing text annotations relative to data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Annotations can be positioned at a relative offset to the *xy* input to
# annotation by setting the *textcoords* keyword argument to ``'offset points'``
# or ``'offset pixels'``.

fig, ax = plt.subplots(figsize=(3, 3))
x = [1, 3, 5, 7, 9]
y = [2, 4, 6, 8, 10]
annotations = ["A", "B", "C", "D", "E"]
ax.scatter(x, y, s=20)

for xi, yi, text in zip(x, y, annotations):
    ax.annotate(text,
                xy=(xi, yi), xycoords='data',  # position of the point to annotate
                xytext=(1.5, 1.5), textcoords='offset points')  # offset of the annotation text

# %%
# The annotations are offset 1.5 points (1.5*1/72 inches) from the *xy* values.
#
# .. _plotting-guide-annotation:
#
# Advanced annotation
# -------------------
#
# We recommend reading :ref:`annotations-tutorial`, :func:`~matplotlib.pyplot.text`
# and :func:`~matplotlib.pyplot.annotate` before reading this section.
#
# Annotating with boxed text
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# `~.Axes.text` takes a *bbox* keyword argument, which draws a box around the
# text:

fig, ax = plt.subplots(figsize=(5, 5))
t = ax.text(0.5, 0.5, "Direction",
            ha="center", va="center", rotation=45, size=15,
            bbox=dict(boxstyle="rarrow,pad=0.3",
                      fc="lightblue", ec="steelblue", lw=2))

# %%
# The arguments are the name of the box style with its attributes as
# keyword arguments. Currently, following box styles are implemented:
#
# ==========   ==============   ==========================
# Class        Name             Attrs
# ==========   ==============   ==========================
# Circle       ``circle``       pad=0.3
# 创建一个新的图形对象和一个包含单个子图的轴对象
fig, ax = plt.subplots(figsize=(3, 3))

# 在轴上添加文本，位于坐标 (0.5, 0.5)，文本内容为 "Test"，字体大小为 30，垂直居中、水平居中对齐，
# 旋转角度为 30 度，文本周围有一个自定义风格的框（boxstyle），透明度为 0.2
ax.text(0.5, 0.5, "Test", size=30, va="center", ha="center", rotation=30,
        bbox=dict(boxstyle=custom_box_style, alpha=0.2))
# %%
# The arrow is drawn as follows:
#
# 1. A path connecting the two points is created, as specified by the
#    *connectionstyle* parameter.
# 2. The path is clipped to avoid patches *patchA* and *patchB*, if these are
#    set.
# 3. The path is further shrunk by *shrinkA* and *shrinkB* (in pixels).
# 4. The path is transmuted to an arrow patch, as specified by the *arrowstyle*
#    parameter.
#
# .. figure:: /gallery/userdemo/images/sphx_glr_annotate_explain_001.png
#    :target: /gallery/userdemo/annotate_explain.html
#    :align: center
#
# The creation of the connecting path between two points is controlled by
# ``connectionstyle`` key and the following styles are available:
#
# ==========   =============================================
# Name         Attrs
# ==========   =============================================
# ``angle``    angleA=90,angleB=0,rad=0.0
# ``angle3``   angleA=90,angleB=0
# ``arc``      angleA=0,angleB=0,armA=None,armB=None,rad=0.0
# ``arc3``     rad=0.0
# ``bar``      armA=0.0,armB=0.0,fraction=0.3,angle=None
# ==========   =============================================
#
# Note that "3" in ``angle3`` and ``arc3`` is meant to indicate that the
# resulting path is a quadratic spline segment (three control
# points). As will be discussed below, some arrow style options can only
# be used when the connecting path is a quadratic spline.
#
# The behavior of each connection style is (limitedly) demonstrated in the
# example below. (Warning: The behavior of the ``bar`` style is currently not
# well-defined and may be changed in the future).
#
# .. figure:: /gallery/userdemo/images/sphx_glr_connectionstyle_demo_001.png
#    :target: /gallery/userdemo/connectionstyle_demo.html
#    :align: center
#
# The connecting path (after clipping and shrinking) is then mutated to
# an arrow patch, according to the given ``arrowstyle``:
#
# ==========   =============================================
# Name         Attrs
# ==========   =============================================
# ``-``        None
# ``->``       head_length=0.4,head_width=0.2
# ``-[``       widthB=1.0,lengthB=0.2,angleB=None
# ``|-|``      widthA=1.0,widthB=1.0
# ``-|>``      head_length=0.4,head_width=0.2
# ``<-``       head_length=0.4,head_width=0.2
# ``<->``      head_length=0.4,head_width=0.2
# ``<|-``      head_length=0.4,head_width=0.2
# ``<|-|>``    head_length=0.4,head_width=0.2
# ``fancy``    head_length=0.4,head_width=0.4,tail_width=0.4
# ``simple``   head_length=0.5,head_width=0.5,tail_width=0.2
# ``wedge``    tail_width=0.3,shrink_factor=0.5
# ==========   =============================================
#
# .. figure:: /gallery/text_labels_and_annotations/images/sphx_glr_fancyarrow_demo_001.png
#    :target: /gallery/text_labels_and_annotations/fancyarrow_demo.html
#    :align: center
# 创建一个新的 Figure 对象，并返回包含的 Axes 对象
fig, ax = plt.subplots(figsize=(3, 3))

# 在 Axes 对象上添加注释文本 "Test"，指定文本的数据坐标为 (0.2, 0.2)
# 文本的位置坐标为数据坐标系，文本显示位置为 (0.8, 0.8) 数据坐标系
# 设置文本大小为 20，垂直对齐方式为居中，水平对齐方式为居中
# 设置箭头属性，箭头风格为 "simple"，连接风格为 "arc3,rad=-0.2"
ax.annotate("Test",
            xy=(0.2, 0.2), xycoords='data',
            xytext=(0.8, 0.8), textcoords='data',
            size=20, va="center", ha="center",
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.2"))

# %%
# 与 `~.Axes.text` 类似，可以使用 *bbox* 参数绘制文本周围的框
fig, ax = plt.subplots(figsize=(3, 3))

# 在 Axes 对象上添加注释文本 "Test"
# 设置文本的数据坐标为 (0.2, 0.2)，文本位置坐标为 (0.8, 0.8) 数据坐标系
# 设置文本大小为 20，垂直对齐方式为居中，水平对齐方式为居中
# 设置文本框的风格为 "round4"，填充颜色为白色
# 设置箭头属性，箭头风格为 "-|>"，连接风格为 "arc3,rad=-0.2"，箭头填充颜色为白色
ann = ax.annotate("Test",
                  xy=(0.2, 0.2), xycoords='data',
                  xytext=(0.8, 0.8), textcoords='data',
                  size=20, va="center", ha="center",
                  bbox=dict(boxstyle="round4", fc="w"),
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=-0.2",
                                  fc="w"))

# %%
# 默认情况下，起始点设置为文本范围的中心。可以使用 ``relpos`` 键值调整起始点位置
# ``relpos`` 的值被归一化到文本的范围。例如，(0, 0) 表示左下角，(1, 1) 表示右上角
fig, ax = plt.subplots(figsize=(3, 3))

# 在 Axes 对象上添加注释文本 "Test"
# 设置文本的数据坐标为 (0.2, 0.2)，文本位置坐标为 (0.8, 0.8) 数据坐标系
# 设置文本大小为 20，垂直对齐方式为居中，水平对齐方式为居中
# 设置文本框的风格为 "round4"，填充颜色为白色
# 设置箭头属性，箭头风格为 "-|>"，连接风格为 "arc3,rad=0.2"，箭头填充颜色为白色
# 设置相对位置参数为 (0., 0.)
ann = ax.annotate("Test",
                  xy=(0.2, 0.2), xycoords='data',
                  xytext=(0.8, 0.8), textcoords='data',
                  size=20, va="center", ha="center",
                  bbox=dict(boxstyle="round4", fc="w"),
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=0.2",
                                  relpos=(0., 0.),
                                  fc="w"))

# 在 Axes 对象上添加另一个注释文本 "Test"
# 设置文本的数据坐标为 (0.2, 0.2)，文本位置坐标为 (0.8, 0.8) 数据坐标系
# 设置文本大小为 20，垂直对齐方式为居中，水平对齐方式为居中
# 设置文本框的风格为 "round4"，填充颜色为白色
# 设置箭头属性，箭头风格为 "-|>"，连接风格为 "arc3,rad=-0.2"，箭头填充颜色为白色
# 设置相对位置参数为 (1., 0.)
ann = ax.annotate("Test",
                  xy=(0.2, 0.2), xycoords='data',
                  xytext=(0.8, 0.8), textcoords='data',
                  size=20, va="center", ha="center",
                  bbox=dict(boxstyle="round4", fc="w"),
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=-0.2",
                                  relpos=(1., 0.),
                                  fc="w"))

# %%
# 在 Axes 的锚定位置放置 Artist
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 有些类型的 Artist 可以放置在 Axes 的锚定位置上。一个常见的例子是图例（legend）。
# 可以通过使用 `.OffsetBox` 类创建这种类型的 Artist。在 :mod:`matplotlib.offsetbox` 和
# :mod:`mpl_toolkits.axes_grid1.anchored_artists` 中有几个预定义的类可用。

# 导入 AnchoredText 类
from matplotlib.offsetbox import AnchoredText

# 创建一个新的 Figure 对象，并返回包含的 Axes 对象
fig, ax = plt.subplots(figsize=(3, 3))

# 创建一个锚定文本的对象，文本内容为 "Figure 1a"
# 设置字体大小为 15，显示边框，锚定位置为左上角
at = AnchoredText("Figure 1a",
                  prop=dict(size=15), frameon=True, loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)


# 设置 AnnotationBbox 的边框样式为圆形，并指定 padding 和 rounding 大小
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
# 将 AnnotationBbox 添加到当前 Axes 中
ax.add_artist(at)



# %%
# *loc* 关键字的含义与 legend 命令中的含义相同。
#
# 当艺术家（或艺术家集合）的大小在创建时以像素大小为已知时，可以简单地应用。
# 例如，如果要绘制一个固定大小为 20 像素 x 20 像素的圆（半径为 10 像素），
# 可以利用 `~mpl_toolkits.axes_grid1.anchored_artists.AnchoredDrawingArea`。
# 实例是根据绘图区域的大小（以像素为单位）创建的，可以向绘图区域添加任意艺术家。
# 注意，添加到绘图区域的艺术家的范围与绘图区域本身的放置无关，只有初始大小很重要。
#
# 添加到绘图区域的艺术家不应设置转换（它将被覆盖），这些艺术家的尺寸被解释为像素坐标，
# 即上面示例中圆的半径分别为 10 像素和 5 像素。
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea

fig, ax = plt.subplots(figsize=(3, 3))
# 创建一个 AnchoredDrawingArea 实例，大小为 40 像素 x 20 像素，位于坐标 (0, 0)
# loc='upper right' 表示将其放置在 Axes 的右上角
# pad=0. 表示不添加额外的内边距，frameon=False 表示不显示边框
ada = AnchoredDrawingArea(40, 20, 0, 0,
                          loc='upper right', pad=0., frameon=False)
# 在 ada 中心绘制一个半径为 10 像素的圆，位置为 (10, 10)
p1 = Circle((10, 10), 10)
ada.drawing_area.add_artist(p1)
# 在 ada 中心绘制一个半径为 5 像素、填充颜色为红色的圆，位置为 (30, 10)
p2 = Circle((30, 10), 5, fc="r")
ada.drawing_area.add_artist(p2)
# 将 ada 添加到当前 Axes 中
ax.add_artist(ada)



# %%
# 有时，您希望您的艺术家与数据坐标（或画布像素以外的坐标）一起缩放。
# 您可以使用 `~mpl_toolkits.axes_grid1.anchored_artists.AnchoredAuxTransformBox` 类。
# 这与 `~mpl_toolkits.axes_grid1.anchored_artists.AnchoredDrawingArea` 类似，
# 但艺术家的范围是在绘制时根据指定的转换确定的。
#
# 下面示例中的椭圆将在数据坐标中具有宽度和高度分别为 0.1 和 0.4，并在 Axes 视图限制更改时自动缩放。
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox

fig, ax = plt.subplots(figsize=(3, 3))
# 创建一个 AnchoredAuxTransformBox 实例，其转换为 ax.transData，位于 'upper left'
box = AnchoredAuxTransformBox(ax.transData, loc='upper left')
# 在 box 中心绘制一个在数据坐标中宽度为 0.1，高度为 0.4，角度为 30 度的椭圆
el = Ellipse((0, 0), width=0.1, height=0.4, angle=30)  # 在数据坐标中！
box.drawing_area.add_artist(el)
# 将 box 添加到当前 Axes 中
ax.add_artist(box)



# %%
# 相对于父 Axes 或锚点定位艺术家的另一种方法是通过 `.AnchoredOffsetbox` 的 *bbox_to_anchor* 参数。
# 此艺术家随后可以使用 `.HPacker` 和 `.VPacker` 自动相对于另一个艺术家定位。
from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
                                  TextArea)

fig, ax = plt.subplots(figsize=(3, 3))

# 创建一个 TextArea 实例，内容为 " Test: "，文本属性设置为黑色
box1 = TextArea(" Test: ", textprops=dict(color="k"))
# 创建一个 DrawingArea 实例，大小为 60 像素 x 20 像素，位于坐标 (0, 0)
box2 = DrawingArea(60, 20, 0, 0)

# 在坐标 (10, 10) 处绘制一个宽度为 16，高度为 5，角度为 30 度，填充颜色为红色的椭圆
el1 = Ellipse((10, 10), width=16, height=5, angle=30, fc="r")
# 创建一个椭圆对象 `el2`，位于坐标 (30, 10)，宽度为 16，高度为 5，角度为 170 度，填充颜色为绿色 ("g")
el2 = Ellipse((30, 10), width=16, height=5, angle=170, fc="g")
# 创建一个椭圆对象 `el3`，位于坐标 (50, 10)，宽度为 16，高度为 5，角度为 230 度，填充颜色为蓝色 ("b")
el3 = Ellipse((50, 10), width=16, height=5, angle=230, fc="b")
# 将 `el1` 添加到 `box2` 容器中作为一个艺术元素
box2.add_artist(el1)
# 将 `el2` 添加到 `box2` 容器中作为一个艺术元素
box2.add_artist(el2)
# 将 `el3` 添加到 `box2` 容器中作为一个艺术元素
box2.add_artist(el3)

# 创建一个水平排列器 `box`，包含 `box1` 和 `box2` 两个子元素，居中对齐，内部填充为 0，子元素间距为 5
box = HPacker(children=[box1, box2],
              align="center",
              pad=0, sep=5)

# 创建一个带有偏移的容器 `anchored_box`，位于左下角，包含 `box` 作为子元素，无内边距，有边框，
# 边界框锚点在 Axes 坐标系中的 (0., 1.02) 处，使用当前坐标变换 `ax.transAxes`
anchored_box = AnchoredOffsetbox(loc='lower left',
                                 child=box, pad=0.,
                                 frameon=True,
                                 bbox_to_anchor=(0., 1.02),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.,)

# 将 `anchored_box` 添加到图形 `ax` 上作为一个艺术元素
ax.add_artist(anchored_box)
# 调整子图的顶部边界，使其位置为 0.8
fig.subplots_adjust(top=0.8)

# %%
# 注意，与 `.Legend` 不同，这里的 ``bbox_transform`` 默认设置为 `.IdentityTransform`
#
# .. _annotating_coordinate_systems:
#
# 注释的坐标系统
# ----------------------------------
#
# Matplotlib 注释支持几种类型的坐标系统。:ref:`annotations-tutorial` 中的示例使用了 `data` 坐标系统；
# 其他一些高级选项包括：
#
# `.Transform` 实例
# ^^^^^^^^^^^^^^^^^^^^^
#
# 转换将坐标映射到不同的坐标系统，通常是显示坐标系统。详细解释请参见 :ref:`transforms_tutorial`。
# 这里使用 Transform 对象来识别相应点的坐标系统。例如，`Axes.transAxes` 转换将注释相对于 Axes 坐标定位；
# 因此使用它等同于将坐标系统设置为 "axes fraction":

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
# 在 `ax1` 上注释文字 "Test"，坐标为 (0.2, 0.2)，坐标系统为 `ax1.transAxes`
ax1.annotate("Test", xy=(0.2, 0.2), xycoords=ax1.transAxes)
# 在 `ax2` 上注释文字 "Test"，坐标为 (0.2, 0.2)，坐标系统为 "axes fraction"
ax2.annotate("Test", xy=(0.2, 0.2), xycoords="axes fraction")

# %%
# 另一个常用的 `.Transform` 实例是 `Axes.transData`。该转换是 Axes 中绘制数据的坐标系统。在这个例子中，
# 它用于在两个 Axes 中的相关数据点之间绘制箭头。我们传递了一个空文本，因为在这种情况下，注释连接数据点。

x = np.linspace(-1, 1)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
ax1.plot(x, -x**3)
ax2.plot(x, -3*x**2)
# 在 `ax2` 上使用 `ax1.transData` 和 `ax2.transData` 绘制箭头
ax2.annotate("",
             xy=(0, 0), xycoords=ax1.transData,
             xytext=(0, 0), textcoords=ax2.transData,
             arrowprops=dict(arrowstyle="<->"))

# %%
# .. _artist_annotation_coord:
#
# `.Artist` 实例
# ^^^^^^^^^^^^^^^^^^
#
# *xy* 值（或 *xytext*）被解释为艺术元素边界框 (bbox) 的分数坐标：

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
# 在 `ax` 上注释文字 "Test 1"，坐标为 (0.5, 0.5)，坐标系统为 "data"，垂直对齐方式为中心，水平对齐方式为中心，
# 使用圆形风格的边界框，填充颜色为白色 ("w")
an1 = ax.annotate("Test 1",
                  xy=(0.5, 0.5), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))
an2 = ax.annotate("Test 2",
                  xy=(1, 0.5), xycoords=an1,  # (1, 0.5) of an1's bbox
                  xytext=(30, 0), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))

# %%
# 确保在绘制 *an2* 之前确定坐标艺术家（例如此例中的 *an1*）的范围。
# 通常意味着 *an2* 需要在 *an1* 之后绘制。所有边界框的基类是 `.BboxBase`。

# Callable that returns `.Transform` of `.BboxBase`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 返回 `.Transform` 或 `.BboxBase` 的可调用对象。例如，`.Artist.get_window_extent`
# 的返回值是一个边界框（bbox），因此这个方法等同于将艺术家作为参数传递：

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
an1 = ax.annotate("Test 1",
                  xy=(0.5, 0.5), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))

an2 = ax.annotate("Test 2",
                  xy=(1, 0.5), xycoords=an1.get_window_extent,
                  xytext=(30, 0), textcoords="offset points",
                  va="center", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))

# %%
# `.Artist.get_window_extent` 是 Axes 对象的边界框，因此与设置坐标系为轴分数相同：

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

an1 = ax1.annotate("Test1", xy=(0.5, 0.5), xycoords="axes fraction")
an2 = ax2.annotate("Test 2", xy=(0.5, 0.5), xycoords=ax2.get_window_extent)

# %%
# Blended coordinate specification
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 混合的坐标规范对 -- 第一个为 x 坐标，第二个为 y 坐标。例如，x=0.5 是数据坐标，
# y=1 是归一化轴坐标：

fig, ax = plt.subplots(figsize=(3, 3))
ax.annotate("Test", xy=(0.5, 1), xycoords=("data", "axes fraction"))
ax.axvline(x=.5, color='lightgray')
ax.set(xlim=(0, 2), ylim=(1, 2))

# %%
# 任何支持的坐标系统都可以在混合规范中使用。例如，文本 "Anchored to 1 & 2"
# 是相对于两个 `.Text` 艺术家定位的：

fig, ax = plt.subplots(figsize=(3, 3))

t1 = ax.text(0.05, .05, "Text 1", va='bottom', ha='left')
t2 = ax.text(0.90, .90, "Text 2", ha='right')
t3 = ax.annotate("Anchored to 1 & 2", xy=(0, 0), xycoords=(t1, t2),
                 va='bottom', color='tab:orange',)

# %%
# `.text.OffsetFrom`
# ^^^^^^^^^^^^^^^^^^
#
# 有时，您希望注释相对于某个点或艺术家的某些 "偏移点"，而不是从注释点。`.text.OffsetFrom`
# 是这种情况的帮助器。
# 导入 matplotlib 中的 OffsetFrom 类
from matplotlib.text import OffsetFrom

# 创建一个新的图形和一个子图，并指定子图的大小为 3x3 英寸
fig, ax = plt.subplots(figsize=(3, 3))

# 在子图 ax 上创建一个注释对象 an1，注释内容为 "Test 1"，位置为 (0.5, 0.5)，使用数据坐标系
an1 = ax.annotate("Test 1", xy=(0.5, 0.5), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))

# 创建一个 OffsetFrom 对象 offset_from，用于指定 an2 的文本位置的偏移量
offset_from = OffsetFrom(an1, (0.5, 0))

# 在子图 ax 上创建另一个注释对象 an2，注释内容为 "Test 2"，位置为 (0.1, 0.1)，使用数据坐标系
# xytext 指定文本的偏移量为 (0, -10) 点，相对于 an1 的位置
# textcoords 使用 offset_from 对象作为文本的坐标系，即基于 an1 的位置偏移
# va 垂直对齐方式为顶部，ha 水平对齐方式为居中
# bbox 指定注释框的样式为圆角矩形，填充颜色为白色
# arrowprops 指定箭头样式为 "->"
an2 = ax.annotate("Test 2", xy=(0.1, 0.1), xycoords="data",
                  xytext=(0, -10), textcoords=offset_from,
                  va="top", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"))



# 导入 matplotlib 中的 ConnectionPatch 类
from matplotlib.patches import ConnectionPatch

# 创建一个包含两个子图的新图形 fig，子图分别为 ax1 和 ax2，大小为 6x3 英寸
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

# 定义连接点的坐标 xy，用于连接两个子图的数据坐标
xy = (0.3, 0.2)

# 创建一个 ConnectionPatch 对象 con，连接点 A 在 ax1 的数据坐标系中，连接点 B 也在 ax2 的数据坐标系中
con = ConnectionPatch(xyA=xy, coordsA=ax1.transData,
                      xyB=xy, coordsB=ax2.transData)

# 将 ConnectionPatch 对象添加到图形 fig 上
fig.add_artist(con)
```