# `D:\src\scipysrc\matplotlib\galleries\tutorials\artists.py`

```
# 文档字符串，可能是模块或函数的描述，包含有关如何使用Artist对象在画布上进行渲染的信息

# 引用重定向，指示从/tutorials/intermediate/artists页面重定向到当前页面

# 标题，指定了本文档的标题为"Artist tutorial"

# 第一个子标题，介绍了Matplotlib API的三个层次结构

# 列表项，解释了FigureCanvas是绘制图形的区域

# 列表项，解释了Renderer是知道如何在FigureCanvas上绘制的对象

# 列表项，解释了Artist是知道如何使用渲染器在画布上绘制的对象

# 段落，解释了FigureCanvas和Renderer处理与用户界面工具包和绘图语言的交互的细节

# 段落，解释了Artist处理高级构造，如表示和布局图、文本和线条

# 段落，解释了典型用户在使用Artists时大约占据95%的时间

# 段落，解释了Artists有两种类型：primitives（基本元素）和containers（容器）

# 段落，解释了primitives表示要绘制到画布上的标准图形对象

# 段落，解释了containers是放置primitives的位置

# 段落，解释了典型用法是创建Figure实例，使用Figure创建一个或多个Axes实例，然后使用Axes的帮助方法创建primitives

# 段落，解释了在下面的示例中，使用matplotlib.pyplot.figure创建了一个Figure实例

# 代码块的起始，引入了matplotlib.pyplot模块并命名为plt

# 创建Figure实例并赋值给变量fig

# 使用Figure的add_subplot方法创建一个Axes实例，并赋值给变量ax

# 注释结束，代码块结束，示例结束
# 导入 matplotlib.pyplot 库，通常用 plt 作为别名
import matplotlib.pyplot as plt
# 导入 numpy 库，通常用 np 作为别名
import numpy as np

# 创建一个新的图形对象
fig = plt.figure()
# 调整子图的顶部边界，使其留出空间给标题
fig.subplots_adjust(top=0.8)
# 在图形中添加一个子图，参数 211 表示在两行一列的布局中选择第一个子图
ax1 = fig.add_subplot(211)
# 设置第一个子图的 y 轴标签
ax1.set_ylabel('Voltage [V]')
# 设置第一个子图的标题
ax1.set_title('A sine wave')

# 生成一组时间数据 t，范围从 0 到 1（不含），步长为 0.01
t = np.arange(0.0, 1.0, 0.01)
# 使用正弦函数生成一组数据 s，频率为 2*pi
s = np.sin(2*np.pi*t)
# 在第一个子图 ax1 上绘制蓝色线条，线宽为 2
line, = ax1.plot(t, s, color='blue', lw=2)

# 设置随机数生成器的种子，以便结果可重复
np.random.seed(19680801)

# 在图形中添加一个新的坐标轴，位置在相对于图形左下角的 [0.15, 0.1] 处，宽度为 0.7，高度为 0.3
ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
# 使用正态分布生成一组随机数据，绘制直方图，分成 50 个箱子，填充色为黄色，边缘色也为黄色
n, bins, patches = ax2.hist(np.random.randn(1000), 50,
                            facecolor='yellow', edgecolor='yellow')
# 设置第二个子图 ax2 的 x 轴标签
ax2.set_xlabel('Time [s]')

# 显示图形
plt.show()
# :class:`~matplotlib.artist.Artist`, and each has an extensive list of
# properties to configure its appearance.  The figure itself contains a
# :class:`~matplotlib.patches.Rectangle` exactly the size of the figure,
# which you can use to set the background color and transparency of the
# figures.  Likewise, each :class:`~matplotlib.axes.Axes` bounding box
# (the standard white box with black edges in the typical Matplotlib
# plot, has a ``Rectangle`` instance that determines the color,
# transparency, and other properties of the Axes.  These instances are
# stored as member variables :attr:`Figure.patch
# <matplotlib.figure.Figure.patch>` and :attr:`Axes.patch
# <matplotlib.axes.Axes.patch>` ("Patch" is a name inherited from
# MATLAB, and is a 2D "patch" of color on the figure, e.g., rectangles,
# circles and polygons).  Every Matplotlib ``Artist`` has the following
# properties
#
# ==========  =================================================================
# Property    Description
# ==========  =================================================================
# alpha       The transparency - a scalar from 0-1
# animated    A boolean that is used to facilitate animated drawing
# axes        The Axes that the Artist lives in, possibly None
# clip_box    The bounding box that clips the Artist
# clip_on     Whether clipping is enabled
# clip_path   The path the artist is clipped to
# contains    A picking function to test whether the artist contains the pick
#             point
# figure      The figure instance the artist lives in, possibly None
# label       A text label (e.g., for auto-labeling)
# picker      A python object that controls object picking
# transform   The transformation
# visible     A boolean whether the artist should be drawn
# zorder      A number which determines the drawing order
# rasterized  Boolean; Turns vectors into raster graphics (for compression &
#             EPS transparency)
# ==========  =================================================================
#
# Each of the properties is accessed with an old-fashioned setter or
# getter (yes we know this irritates Pythonistas and we plan to support
# direct access via properties or traits but it hasn't been done yet).
# For example, to multiply the current alpha by a half::
#
#     a = o.get_alpha()
#     o.set_alpha(0.5*a)
#
# If you want to set a number of properties at once, you can also use
# the ``set`` method with keyword arguments.  For example::
#
#     o.set(alpha=0.5, zorder=2)
#
# If you are working interactively at the python shell, a handy way to
# inspect the ``Artist`` properties is to use the
# :func:`matplotlib.artist.getp` function (simply
# :func:`~matplotlib.pyplot.getp` in pyplot), which lists the properties
# and their values.  This works for classes derived from ``Artist`` as
# well, e.g., ``Figure`` and ``Rectangle``.  Here are the ``Figure`` rectangle
# properties mentioned above:
#
# .. sourcecode:: ipython
#
# In [149]: matplotlib.artist.getp(fig.patch)
# 获取图形对象 fig 的背景图形的属性

# agg_filter = None
# 聚合滤波器属性，默认为 None

# alpha = None
# 透明度属性，默认为 None

# animated = False
# 是否动画属性，默认为 False

# antialiased or aa = False
# 是否抗锯齿属性，默认为 False

# bbox = Bbox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
# 边界框属性，描述了图形对象的位置和大小，默认为整个图形的范围

# capstyle = butt
# 线帽风格属性，默认为 butt

# children = []
# 子元素列表属性，当前图形对象下的子元素列表为空列表

# clip_box = None
# 剪切框属性，默认为 None

# clip_on = True
# 是否开启剪切属性，默认为 True

# clip_path = None
# 剪切路径属性，默认为 None

# contains = None
# 包含对象属性，默认为 None

# data_transform = BboxTransformTo(TransformedBbox(Bbox...
# 数据变换属性，描述了数据空间到图形空间的变换关系

# edgecolor or ec = (1.0, 1.0, 1.0, 1.0)
# 边界颜色属性，默认为白色 (1.0, 1.0, 1.0, 1.0)

# extents = Bbox(x0=0.0, y0=0.0, x1=640.0, y1=480.0)
# 扩展属性，描述了图形对象的范围，默认为图形大小 (640, 480)

# facecolor or fc = (1.0, 1.0, 1.0, 1.0)
# 填充颜色属性，默认为白色 (1.0, 1.0, 1.0, 1.0)

# figure = Figure(640x480)
# 所属图形属性，描述了该图形对象所属的 Figure 对象，尺寸为 640x480

# fill = True
# 是否填充属性，默认为 True

# gid = None
# 组 ID 属性，默认为 None

# hatch = None
# 草图样式属性，默认为 None

# height = 1
# 高度属性，默认为 1

# in_layout = False
# 是否在布局中属性，默认为 False

# joinstyle = miter
# 连接风格属性，默认为 miter

# label =
# 标签属性，默认为空

# linestyle or ls = solid
# 线条样式属性，默认为实线 solid

# linewidth or lw = 0.0
# 线宽属性，默认为 0.0

# patch_transform = CompositeGenericTransform(BboxTransformTo(...
# 图形变换属性，描述了图形对象的变换关系

# path = Path(array([[0., 0.], [1., 0.], [1., 480.], [0., 480....
# 路径属性，描述了图形对象的路径信息

# path_effects = []
# 路径效果属性，默认为空列表

# picker = None
# 选取器属性，默认为 None

# rasterized = None
# 栅格化属性，默认为 None

# sketch_params = None
# 素描参数属性，默认为 None

# snap = None
# 对齐属性，默认为 None

# transform = CompositeGenericTransform(CompositeGenericTra...
# 变换属性，描述了图形对象的整体变换关系

# transformed_clip_path_and_affine = (None, None)
# 转换后剪切路径和仿射变换属性，默认为 (None, None)

# url = None
# URL 属性，默认为 None

# verts = [[0., 0.], [640., 0.], [640., 480.], [0., 480....
# 顶点属性，描述了图形对象的顶点坐标信息

# visible = True
# 可见性属性，默认为 True

# width = 1
# 宽度属性，默认为 1

# window_extent = Bbox(x0=0.0, y0=0.0, x1=640.0, y1=480.0)
# 窗口范围属性，描述了图形对象在窗口中的范围，默认为图形大小 (640, 480)

# x = 0
# X 坐标属性，默认为 0

# xy = (0, 0)
# XY 坐标属性，默认为 (0, 0)

# y = 0
# Y 坐标属性，默认为 0

# zorder = 1
# Z 轴顺序属性，默认为 1

# 图形对象所有类的文档字符串也包含 ``Artist`` 属性，因此可以参考交互式帮助或 :ref:`artist-api` 获取特定对象的属性列表。
#
# :ref:`object-containers` 链接到对象容器部分，详细介绍了如何检查和设置特定对象的属性。
#
# 图形容器
# =========
#
# 最顶层的容器 ``Artist`` 是 :class:`matplotlib.figure.Figure`，它包含了图形中的所有内容。
# 图形的背景是一个 :class:`~matplotlib.patches.Rectangle`，存储在 :attr:`Figure.patch <matplotlib.figure.Figure.patch>` 属性中。
# 如同介绍中提到的，有两种对象类型：基本元素和容器。基本元素通常是您想要配置的对象（例如 :class:`~matplotlib.text.Text` 的字体，:class:`~matplotlib.lines.Line2D` 的宽度），
# 虽然容器也有一些属性，例如 :class:`~matplotlib.axes.Axes` 是一个包含了绘图中许多基本元素的容器，但它也有像 ``xscale`` 这样的属性，用于控制 x 轴是 'linear' 还是 'log'。
# 本节将回顾各种容器对象存储的 ``Artists`` 的位置。
#
# :ref:`figure-container` 链接到图形容器部分。
# 创建一个新的 matplotlib 图形对象
import matplotlib.lines as lines
fig = plt.figure()

# 创建两条直线对象，使用 figure 的 transFigure 变换将它们放置在图形的坐标系中
l1 = lines.Line2D([0, 1], [0, 1], transform=fig.transFigure, figure=fig)
l2 = lines.Line2D([0, 1], [1, 0], transform=fig.transFigure, figure=fig)

# 将这两条直线对象添加到图形对象的 lines 属性中
fig.lines.extend([l1, l2])

# 显示图形
plt.show()
# lines            A list of Figure `.Line2D` instances
#                  (rarely used, see ``Axes.lines``)
# patches          A list of Figure `.Patch`\s
#                  (rarely used, see ``Axes.patches``)
# texts            A list Figure `.Text` instances
# ================ ============================================================
#
# .. _axes-container:
#
# Axes container
# --------------
#
# The :class:`matplotlib.axes.Axes` is the center of the Matplotlib
# universe -- it contains the vast majority of all the ``Artists`` used
# in a figure with many helper methods to create and add these
# ``Artists`` to itself, as well as helper methods to access and
# customize the ``Artists`` it contains.  Like the
# :class:`~matplotlib.figure.Figure`, it contains a
# :class:`~matplotlib.patches.Patch`
# :attr:`~matplotlib.axes.Axes.patch` which is a
# :class:`~matplotlib.patches.Rectangle` for Cartesian coordinates and a
# :class:`~matplotlib.patches.Circle` for polar coordinates; this patch
# determines the shape, background and border of the plotting region::
#
#     ax = fig.add_subplot()
#     rect = ax.patch  # a Rectangle instance
#     rect.set_facecolor('green')
#
# When you call a plotting method, e.g., the canonical
# `~matplotlib.axes.Axes.plot` and pass in arrays or lists of values, the
# method will create a `matplotlib.lines.Line2D` instance, update the line with
# all the ``Line2D`` properties passed as keyword arguments, add the line to
# the ``Axes``, and return it to you:
#
# .. sourcecode:: ipython
#
#     In [213]: x, y = np.random.rand(2, 100)
#
#     In [214]: line, = ax.plot(x, y, '-', color='blue', linewidth=2)
#
# ``plot`` returns a list of lines because you can pass in multiple x, y
# pairs to plot, and we are unpacking the first element of the length
# one list into the line variable.  The line has been added to the
# ``Axes.lines`` list:
#
# .. sourcecode:: ipython
#
#     In [229]: print(ax.lines)
#     [<matplotlib.lines.Line2D at 0xd378b0c>]
#
# Similarly, methods that create patches, like
# :meth:`~matplotlib.axes.Axes.bar` creates a list of rectangles, will
# add the patches to the :attr:`Axes.patches
# <matplotlib.axes.Axes.patches>` list:
#
# .. sourcecode:: ipython
#
#     In [233]: n, bins, rectangles = ax.hist(np.random.randn(1000), 50)
#
#     In [234]: rectangles
#     Out[234]: <BarContainer object of 50 artists>
#
#     In [235]: print(len(ax.patches))
#     Out[235]: 50
#
# You should not add objects directly to the ``Axes.lines`` or ``Axes.patches``
# lists, because the ``Axes`` needs to do a few things when it creates and adds
# an object:
#
# - It sets the ``figure`` and ``axes`` property of the ``Artist``;
# - It sets the default ``Axes`` transformation (unless one is already set);
# - It inspects the data contained in the ``Artist`` to update the data
#   structures controlling auto-scaling, so that the view limits can be
#   adjusted to contain the plotted data.
#
# You can, nonetheless, create objects yourself and add them directly to the
# ``Axes`` using helper methods like `~matplotlib.axes.Axes.add_line` and
# `~matplotlib.axes.Axes.add_patch`.  Here is an annotated interactive session
# illustrating what is going on:
#
# .. sourcecode:: ipython
#
#     In [262]: fig, ax = plt.subplots()
#
#     # create a rectangle instance
#     In [263]: rect = matplotlib.patches.Rectangle((1, 1), width=5, height=12)
#
#     # by default the Axes instance is None
#     In [264]: print(rect.axes)
#     None
#
#     # and the transformation instance is set to the "identity transform"
#     In [265]: print(rect.get_data_transform())
#     IdentityTransform()
#
#     # now we add the Rectangle to the Axes
#     In [266]: ax.add_patch(rect)
#
#     # and notice that the ax.add_patch method has set the Axes
#     # instance
#     In [267]: print(rect.axes)
#     Axes(0.125,0.1;0.775x0.8)
#
#     # and the transformation has been set too
#     In [268]: print(rect.get_data_transform())
#     CompositeGenericTransform(
#         TransformWrapper(
#             BlendedAffine2D(
#                 IdentityTransform(),
#                 IdentityTransform())),
#         CompositeGenericTransform(
#             BboxTransformFrom(
#                 TransformedBbox(
#                     Bbox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
#                     TransformWrapper(
#                         BlendedAffine2D(
#                             IdentityTransform(),
#                             IdentityTransform())))),
#             BboxTransformTo(
#                 TransformedBbox(
#                     Bbox(x0=0.125, y0=0.10999999999999999, x1=0.9, y1=0.88),
#                     BboxTransformTo(
#                         TransformedBbox(
#                             Bbox(x0=0.0, y0=0.0, x1=6.4, y1=4.8),
#                             Affine2D(
#                                 [[100.   0.   0.]
#                                  [  0. 100.   0.]
#                                  [  0.   0.   1.]])))))))
#
#     # the default Axes transformation is ax.transData
#     In [269]: print(ax.transData)
#     CompositeGenericTransform(
#         TransformWrapper(
#             BlendedAffine2D(
#                 IdentityTransform(),
#                 IdentityTransform())),
#         CompositeGenericTransform(
#             BboxTransformFrom(
#                 TransformedBbox(
#                     Bbox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
#                     TransformWrapper(
#                         BlendedAffine2D(
#                             IdentityTransform(),
#                             IdentityTransform())))),
#             BboxTransformTo(
#                 TransformedBbox(
#                     Bbox(x0=0.125, y0=0.10999999999999999, x1=0.9, y1=0.88),
#                     BboxTransformTo(
#                         TransformedBbox(
#                             Bbox(x0=0.0, y0=0.0, x1=6.4, y1=4.8),
#                             Affine2D(
#                                 [[100.   0.   0.]
#                                  [  0. 100.   0.]
#                                  [  0.   0.   1.]])))))))
#
#     # the default Axes transformation is ax.transData
#     In [270]: print(ax.transData)
#     CompositeGenericTransform(
#         TransformWrapper(
#             BlendedAffine2D(
#                 IdentityTransform(),
#                 IdentityTransform())),
#         CompositeGenericTransform(
#             BboxTransformFrom(
#                 TransformedBbox(
#                     Bbox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
#                     TransformWrapper(
#                         BlendedAffine2D(
#                             IdentityTransform(),
#                             IdentityTransform())))),
#             BboxTransformTo(
#                 TransformedBbox(
#                     Bbox(x0=0.125, y0=0.10999999999999999, x1=0.9, y1=0.88),
#                     BboxTransformTo(
#                         TransformedBbox(
#                             Bbox(x0=0.0, y0=0.0, x1=6.4, y1=4.8),
#                             Affine2D(
#                                 [[100.   0.   0.]
#                                  [  0. 100.   0.]
#                                  [  0.   0.   1.]])))))))
#                             Affine2D(
#                                 [[100.   0.   0.]
#                                  [  0. 100.   0.]
#                                  [  0.   0.   1.]])))))))
#
#     # notice that the xlimits of the Axes have not been changed
#     In [270]: print(ax.get_xlim())
#     (0.0, 1.0)
#
#     # but the data limits have been updated to encompass the rectangle
#     In [271]: print(ax.dataLim.bounds)
#     (1.0, 1.0, 5.0, 12.0)
#
#     # we can manually invoke the auto-scaling machinery
#     In [272]: ax.autoscale_view()
#
#     # and now the xlim are updated to encompass the rectangle, plus margins
#     In [273]: print(ax.get_xlim())
#     (0.75, 6.25)
#
#     # we have to manually force a figure draw
#     In [274]: fig.canvas.draw()
#
#
# There are many, many ``Axes`` helper methods for creating primitive
# ``Artists`` and adding them to their respective containers.  The table
# below summarizes a small sampling of them, the kinds of ``Artist`` they
# create, and where they store them
#
# =========================================  =================  ===============
# Axes helper method                         Artist             Container
# =========================================  =================  ===============
# `~.axes.Axes.annotate` - text annotations  `.Annotation`      ax.texts
# `~.axes.Axes.bar` - bar charts             `.Rectangle`       ax.patches
# `~.axes.Axes.errorbar` - error bar plots   `.Line2D` and      ax.lines and
#                                            `.Rectangle`       ax.patches
# `~.axes.Axes.fill` - shared area           `.Polygon`         ax.patches
# `~.axes.Axes.hist` - histograms            `.Rectangle`       ax.patches
# `~.axes.Axes.imshow` - image data          `.AxesImage`       ax.images
# `~.axes.Axes.legend` - Axes legend         `.Legend`          ax.get_legend()
# `~.axes.Axes.plot` - xy plots              `.Line2D`          ax.lines
# `~.axes.Axes.scatter` - scatter charts     `.PolyCollection`  ax.collections
# `~.axes.Axes.text` - text                  `.Text`            ax.texts
# =========================================  =================  ===============
#
#
# In addition to all of these ``Artists``, the ``Axes`` contains two
# important ``Artist`` containers: the :class:`~matplotlib.axis.XAxis`
# and :class:`~matplotlib.axis.YAxis`, which handle the drawing of the
# ticks and labels.  These are stored as instance variables
# :attr:`~matplotlib.axes.Axes.xaxis` and
# :attr:`~matplotlib.axes.Axes.yaxis`.  The ``XAxis`` and ``YAxis``
# containers will be detailed below, but note that the ``Axes`` contains
# many helper methods which forward calls on to the
# :class:`~matplotlib.axis.Axis` instances, so you often do not need to
# work with them directly unless you want to.  For example, you can set
# the font color of the ``XAxis`` ticklabels using the ``Axes`` helper
# method::
#
#     ax.tick_params(axis='x', labelcolor='orange')
#
# 以下是关于 `~.axes.Axes` 包含的艺术家（Artists）的摘要

# ==============    =========================================
# Axes 属性         描述
# ==============    =========================================
# artists           一个包含 `.Artist` 实例的 `.ArtistList`
# patch             用于 Axes 背景的 `.Rectangle` 实例
# collections       一个包含 `.Collection` 实例的 `.ArtistList`
# images            一个包含 `.AxesImage` 实例的 `.ArtistList`
# lines             一个包含 `.Line2D` 实例的 `.ArtistList`
# patches           一个包含 `.Patch` 实例的 `.ArtistList`
# texts             一个包含 `.Text` 实例的 `.ArtistList`
# xaxis             一个 `matplotlib.axis.XAxis` 实例
# yaxis             一个 `matplotlib.axis.YAxis` 实例
# ==============    =========================================

# 可以通过 `~.axes.Axes.get_legend` 访问图例

# .. _axis-container:

# 轴容器
# ---------------
#
# :class:`matplotlib.axis.Axis` 实例负责绘制刻度线、网格线、刻度标签和轴标签。
# 您可以分别为 y 轴配置左右刻度，以及为 x 轴配置上下刻度。
# `Axis` 还存储用于自动缩放、平移和缩放的数据和视图间隔，
# 以及控制刻度位置和刻度字符串表示的 :class:`~matplotlib.ticker.Locator` 和
# :class:`~matplotlib.ticker.Formatter` 实例。
#
# 每个 `Axis` 对象都包含一个 :attr:`~matplotlib.axis.Axis.label` 属性
# （这是 `~.pyplot.xlabel` 和 `~.pyplot.ylabel` 调用中 :mod:`.pyplot` 修改的内容），
# 以及主要和次要刻度的列表。刻度是 `.axis.XTick` 和 `.axis.YTick` 实例，
# 包含渲染刻度和刻度标签的实际线条和文本原语。
# 因为刻度在需要时动态创建（例如在平移和缩放时），应通过其访问器方法
# `.axis.Axis.get_major_ticks` 和 `.axis.Axis.get_minor_ticks` 访问主要和次要刻度列表。
# 虽然刻度包含所有的基本元素并将在下面详细讨论，但 `Axis` 实例具有返回刻度线、刻度标签、
# 刻度位置等的访问器方法：

fig, ax = plt.subplots()
axis = ax.xaxis

axis.get_ticklocs()

# %%

axis.get_ticklabels()

# %%
# 注意，ticklines 的数量是标签数量的两倍，因为默认情况下顶部和底部都有刻度线，
# 但只有 x 轴下方有刻度标签；然而，这可以自定义。

axis.get_ticklines()

# %%
# 通过上述方法，默认情况下只返回主要刻度的列表，但也可以请求次要刻度：

axis.get_ticklabels(minor=True)
axis.get_ticklines(minor=True)

# %%
# 这里是 `Axis` 的一些有用访问器方法的摘要
# （其中有对应的设置器在有用时使用，比如 :meth:`~matplotlib.axis.Axis.set_major_formatter`。）
#
# =============================  ==============================================
# Axis accessor method           Description
# =============================  ==============================================
# `~.Axis.get_scale`             返回轴的比例尺类型，例如 'log' 或 'linear'
# `~.Axis.get_view_interval`     返回轴视图限制的间隔实例
# `~.Axis.get_data_interval`     返回轴数据限制的间隔实例
# `~.Axis.get_gridlines`         返回轴的网格线列表
# `~.Axis.get_label`             返回轴的标签 - 一个 `.Text` 实例
# `~.Axis.get_offset_text`       返回轴的偏移文本 - 一个 `.Text` 实例
# `~.Axis.get_ticklabels`        返回主要或次要刻度标签的 `.Text` 实例列表 -
#                                关键字 minor=True|False
# `~.Axis.get_ticklines`         返回主要或次要刻度线的 `.Line2D` 实例列表 -
#                                关键字 minor=True|False
# `~.Axis.get_ticklocs`          返回刻度位置的列表 -
#                                关键字 minor=True|False
# `~.Axis.get_major_locator`     返回主刻度的 `.ticker.Locator` 实例
# `~.Axis.get_major_formatter`   返回主刻度的 `.ticker.Formatter` 实例
# `~.Axis.get_minor_locator`     返回次刻度的 `.ticker.Locator` 实例
# `~.Axis.get_minor_formatter`   返回次刻度的 `.ticker.Formatter` 实例
# `~.axis.Axis.get_major_ticks`  返回主刻度的 `.Tick` 实例列表
# `~.axis.Axis.get_minor_ticks`  返回次刻度的 `.Tick` 实例列表
# `~.Axis.grid`                  控制主要或次要刻度的网格线的显示与否
# =============================  ==============================================
#
# 这里有一个示例，虽然不以其美丽而著称，但它自定义了Axes和Tick的属性。
#
# plt.figure 创建一个 matplotlib.figure.Figure 实例
fig = plt.figure()
rect = fig.patch  # 一个矩形实例
rect.set_facecolor('lightgoldenrodyellow')

ax1 = fig.add_axes([0.1, 0.3, 0.4, 0.4])
rect = ax1.patch
rect.set_facecolor('lightslategray')

# 针对 ax1 的 x 轴刻度标签进行遍历处理
for label in ax1.xaxis.get_ticklabels():
    # label 是一个 Text 实例
    label.set_color('red')  # 设置标签颜色为红色
    label.set_rotation(45)  # 设置标签旋转角度为45度
    label.set_fontsize(16)   # 设置标签字体大小为16号字体

# 针对 ax1 的 y 轴刻度线进行遍历处理
for line in ax1.yaxis.get_ticklines():
    # line 是一个 Line2D 实例
    line.set_color('green')           # 设置刻度线颜色为绿色
    line.set_markersize(25)           # 设置刻度线标记大小为25
    line.set_markeredgewidth(3)       # 设置刻度线标记边缘宽度为3

plt.show()

# %%
# .. _tick-container:
#
# 刻度容器
# ---------------
#
# :class:`matplotlib.axis.Tick` 是从 :class:`~matplotlib.figure.Figure`
# 到 :class:`~matplotlib.axes.Axes` 到 :class:`~matplotlib.axis.Axis`
# 到 :class:`~matplotlib.axis.Tick` 的最终容器对象。`Tick` 包含刻度和网格线实例，
# 以及上下刻度的标签实例。每个实例可以直接作为 `Tick` 的属性访问。
#
# ==============  ==========================================================
# Tick attribute  Description
# ==============  ==========================================================
# tick1line       用于绘制刻度线的 `.Line2D` 实例
# tick2line       用于绘制刻度线的 `.Line2D` 实例
# gridline        用于绘制网格线的 `.Line2D` 实例
# label1          用于标注刻度标签的 `.Text` 实例
# label2          用于标注刻度标签的 `.Text` 实例
# ==============  ==========================================================
#
# 这里是一个示例，展示了如何在 y 轴右侧设置带有美元符号的刻度标签，并将其颜色设为绿色。
#
#
# .. include:: ../gallery/ticks/dollar_ticks.rst
#    :start-after: .. redirect-from:: /gallery/pyplots/dollar_ticks
#    :end-before: .. admonition:: References


这段注释解释了一系列与 matplotlib 中刻度标签相关的属性，以及如何在图表中设置特定格式的刻度标签样式的示例。
```