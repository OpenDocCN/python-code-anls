# `D:\src\scipysrc\matplotlib\galleries\users_explain\artists\transforms_tutorial.py`

```py
"""
.. redirect-from:: /tutorials/advanced/transforms_tutorial

.. _transforms_tutorial:

========================
Transformations Tutorial
========================

Like any graphics packages, Matplotlib is built on top of a transformation
framework to easily move between coordinate systems, the userland *data*
coordinate system, the *axes* coordinate system, the *figure* coordinate
system, and the *display* coordinate system.  In 95% of your plotting, you
won't need to think about this, as it happens under the hood, but as you push
the limits of custom figure generation, it helps to have an understanding of
these objects, so you can reuse the existing transformations Matplotlib makes
available to you, or create your own (see :mod:`matplotlib.transforms`).  The
table below summarizes some useful coordinate systems, a description of each
system, and the transformation object for going from each coordinate system to
the *display* coordinates.  In the "Transformation Object" column, ``ax`` is a
:class:`~matplotlib.axes.Axes` instance, ``fig`` is a
:class:`~matplotlib.figure.Figure` instance, and ``subfigure`` is a
:class:`~matplotlib.figure.SubFigure` instance.


+----------------+-----------------------------------+-----------------------------+
|Coordinate      |Description                        |Transformation object        |
|system          |                                   |from system to display       |
+================+===================================+=============================+
|"data"          |The coordinate system of the data  |``ax.transData``             |
|                |in the Axes.                       |                             |
+----------------+-----------------------------------+-----------------------------+
|"axes"          |The coordinate system of the       |``ax.transAxes``             |
|                |`~matplotlib.axes.Axes`; (0, 0)    |                             |
|                |is bottom left of the Axes, and    |                             |
|                |(1, 1) is top right of the Axes.   |                             |
+----------------+-----------------------------------+-----------------------------+
|"subfigure"     |The coordinate system of the       |``subfigure.transSubfigure`` |
|                |`.SubFigure`; (0, 0) is bottom left|                             |
|                |of the subfigure, and (1, 1) is top|                             |
|                |right of the subfigure.  If a      |                             |
|                |figure has no subfigures, this is  |                             |
|                |the same as ``transFigure``.       |                             |
+----------------+-----------------------------------+-----------------------------+
|"figure"        |The coordinate system of the       |``fig.transFigure``          |
|                |`.Figure`; (0, 0) is bottom left   |                             |
"""

# 以上是一个包含注释和表格的文档字符串，描述了Matplotlib中的坐标系统和转换对象
# The coordinate system of the `.Figure` in inches; (0, 0) is bottom left of the figure, and (width, height) is the top right of the figure in inches.
"figure-inches"    |``fig.dpi_scale_trans``      

# Blended coordinate systems, using data coordinates on one direction and axes coordinates on the other.
"xaxis", "yaxis"   |``ax.get_xaxis_transform()``, ``ax.get_yaxis_transform()``

# The native coordinate system of the output; (0, 0) is the bottom left of the window, and (width, height) is top right of the output in "display units".
"display"          |`None`, or `.IdentityTransform()`
# %%
# 导入 matplotlib 库和 numpy 库
import matplotlib.pyplot as plt
import numpy as np

# 导入 matplotlib 的 patches 模块
import matplotlib.patches as mpatches

# 创建一个数组 x 包含从 0 到 10 的数据，步长为 0.005
x = np.arange(0, 10, 0.005)

# 创建一个函数 y，使用指数衰减和正弦函数生成数据
y = np.exp(-x/2.) * np.sin(2*np.pi*x)

# 创建一个图形窗口和一个坐标系
fig, ax = plt.subplots()

# 在坐标系上绘制 x 和 y 的数据点
ax.plot(x, y)

# 设置 x 轴的显示范围为 0 到 10
ax.set_xlim(0, 10)

# 设置 y 轴的显示范围为 -1 到 1
ax.set_ylim(-1, 1)

# 显示图形
plt.show()

# %%
# 您可以使用 ax.transData 实例将 *data* 坐标系中的点或点序列转换到 *display* 坐标系中，如下所示:
#
# .. sourcecode:: ipython
#
#     In [14]: type(ax.transData)
#     Out[14]: <class 'matplotlib.transforms.CompositeGenericTransform'>
#
#     In [15]: ax.transData.transform((5, 0))
#     Out[15]: array([ 335.175,  247.   ])
#
#     In [16]: ax.transData.transform([(5, 0), (1, 2)])
#     Out[16]:
#     array([[ 335.175,  247.   ],
#            [ 132.435,  642.2  ]])
#
# 您可以使用 :meth:`~matplotlib.transforms.Transform.inverted` 方法创建一个转换，
# 该转换可以将 *display* 坐标系中的点转换回 *data* 坐标系:
#
# .. sourcecode:: ipython
#
#     In [41]: inv = ax.transData.inverted()
#
#     In [42]: type(inv)
#     Out[42]: <class 'matplotlib.transforms.CompositeGenericTransform'>
#
#     In [43]: inv.transform((335.175,  247.))
#     Out[43]: array([ 5.,  0.])
#
# 如果您正在按照本教程输入代码，*display* 坐标的确切值可能会有所不同，
# 如果窗口大小或 dpi 设置不同。同样，在下面的图中，显示标记的点可能与 ipython 会话中的不同，
# 因为文档图形大小默认值不同。

# 创建一个新的数组 x，包含从 0 到 10 的数据，步长为 0.005
x = np.arange(0, 10, 0.005)
y = np.exp(-x/2.) * np.sin(2*np.pi*x)

# 计算 y 值，使用指数函数和正弦函数对 x 进行变换


fig, ax = plt.subplots()

# 创建一个新的图形对象和一个包含单个子图的 Axes 对象，并将它们分配给 fig 和 ax 变量


ax.plot(x, y)

# 在 ax 对象上绘制曲线，x 是横坐标，y 是纵坐标


ax.set_xlim(0, 10)

# 设置 x 轴的显示范围，从 0 到 10


ax.set_ylim(-1, 1)

# 设置 y 轴的显示范围，从 -1 到 1


xdata, ydata = 5, 0

# 定义 xdata 和 ydata 变量，并分别赋值为 5 和 0


xdisplay, ydisplay = ax.transData.transform((xdata, ydata))

# 将数据坐标 (xdata, ydata) 转换为显示坐标系中的坐标 (xdisplay, ydisplay)，存储在 xdisplay 和 ydisplay 变量中


bbox = dict(boxstyle="round", fc="0.8")

# 创建一个文本框的样式定义字典，圆角矩形，填充颜色为灰色


arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle,angleA=0,angleB=90,rad=10")

# 创建一个箭头样式定义字典，设置箭头形状为 "->"，连接样式为角度连接，角度设置为 10 度


offset = 72

# 设置偏移量为 72，用于调整文本注释和箭头的位置


ax.annotate(f'data = ({xdata:.1f}, {ydata:.1f})',
            (xdata, ydata), xytext=(-2*offset, offset), textcoords='offset points',
            bbox=bbox, arrowprops=arrowprops)

# 在 ax 对象上添加注释，显示为 "data = (5.0, 0.0)"，注释位置为 (xdata, ydata)，文本位置根据偏移量 (-144, 72)，使用偏移坐标点，文本框样式为 bbox，箭头样式为 arrowprops


disp = ax.annotate(f'display = ({xdisplay:.1f}, {ydisplay:.1f})',
                   (xdisplay, ydisplay), xytext=(0.5*offset, -offset),
                   xycoords='figure pixels',
                   textcoords='offset points',
                   bbox=bbox, arrowprops=arrowprops)

# 在 ax 对象上添加注释，显示为 "display = (335.2, 247.0)"，注释位置为 (xdisplay, ydisplay)，文本位置根据偏移量 (36, -72)，使用像素坐标系，文本框样式为 bbox，箭头样式为 arrowprops


plt.show()

# 显示图形

# %%
# .. warning::
#
#   如果在 GUI 后端中运行上面的示例源代码，可能会发现 *data* 和 *display* 两个注释的箭头并不完全指向同一点。这是因为显示点是在显示图形之前计算的，GUI 后端在创建图形时可能会稍微调整图形大小。如果手动调整图形大小，效果会更加明显。这是为什么很少使用 *display* 空间工作的一个很好的理由，但可以连接到 ``'on_draw'`` 事件来更新 *figure* 坐标，以在图形绘制时更新；参见 :ref:`event-handling`。
#
# 在更改坐标轴的 x 或 y 范围时，数据限制会更新，因此变换会得到新的显示点。注意，当我们仅更改 ylim 时，只有 y-display 坐标会改变，而当我们同时更改 xlim 时，两者都会改变。稍后我们将详细讨论 :class:`~matplotlib.transforms.Bbox`。
#
# .. sourcecode:: ipython
#
#     In [54]: ax.transData.transform((5, 0))
#     Out[54]: array([ 335.175,  247.   ])
#
#     In [55]: ax.set_ylim(-1, 2)
#     Out[55]: (-1, 2)
#
#     In [56]: ax.transData.transform((5, 0))
#     Out[56]: array([ 335.175     ,  181.13333333])
#
#     In [57]: ax.set_xlim(10, 20)
#     Out[57]: (10, 20)
#
#     In [58]: ax.transData.transform((5, 0))
#     Out[58]: array([-171.675     ,  181.13333333])
#
#
# .. _axes-coords:
#
# Axes coordinates
# ================
#
# 在 *data* 坐标系之后，*axes* 坐标系可能是第二个最有用的坐标系。这里的点 (0, 0) 是您的 Axes 或 subplot 的左下角，(0.5, 0.5) 是中心，(1.0, 1.0) 是右上角。您还可以引用范围外的点，因此 (-0.1, 1.1) 是在 Axes 的左侧和上方。在放置文本到 Axes 中时，这个坐标系非常有用，因为通常希望文本气泡
# 创建一个新的图形对象
fig, ax = plt.subplots()

# 生成一个包含1000个随机数的数据集 x
x = np.random.randn(1000)

# 在坐标系中绘制 x 的直方图
ax.hist(x, 30)

# 设置图表的标题，使用 LaTeX 渲染带有数学符号的文本
ax.set_title(r'$\sigma=1 \/ \dots \/ \sigma=2$', fontsize=16)

# 创建一个混合转换对象，将数据坐标系 (transData) 和坐标轴坐标系 (transAxes) 混合在一起
trans = transforms.blended_transform_factory(
    ax.transData, ax.transAxes)

# 准备用于绘制的混合坐标系转换对象
# 注意：接下来的代码行需要根据上下文补充，这里只是开始了 blended_transform_factory 的使用，但没有完成绘制部分。
# 创建一个矩形对象，左下角顶点在数据坐标 (1, 0)，宽度为 1，高度为 1，使用指定的变换 transform
rect = mpatches.Rectangle((1, 0), width=1, height=1, transform=trans,
                          color='yellow', alpha=0.5)
# 将矩形对象添加到 Axes 对象 ax 中
ax.add_patch(rect)

# 显示绘图结果
plt.show()

# %%
# .. note::
#
#   混合的变换，其中 x 是数据坐标，y 是坐标轴坐标，非常有用，因此我们有助手方法用于返回
#   Matplotlib 内部用于绘制刻度、刻度标签等的变换。这些方法是 :meth:`matplotlib.axes.Axes.get_xaxis_transform` 和
#   :meth:`matplotlib.axes.Axes.get_yaxis_transform`。因此，在上面的示例中，
#   调用 :meth:`~matplotlib.transforms.blended_transform_factory` 可以被替换为 ``get_xaxis_transform``::
#
#     trans = ax.get_xaxis_transform()
#
# .. _transforms-fig-scale-dpi:
#
# 在物理坐标中绘图
# ================================
#
# 有时我们希望对象在绘图中具有特定的物理大小。
# 在这里，我们画出了与上面相同的圆，但使用物理坐标。如果是交互式绘图，可以看到更改图形大小
# 不会改变圆与左下角的偏移量，也不会改变其大小，圆保持圆形，不受坐标轴纵横比的影响。

fig, ax = plt.subplots(figsize=(5, 4))
x, y = 10*np.random.rand(2, 1000)
# 在数据坐标中绘制一些数据点
ax.plot(x, y*10., 'go', alpha=0.2)
# 添加一个圆在固定坐标系中
circ = mpatches.Circle((2.5, 2), 1.0, transform=fig.dpi_scale_trans,
                       facecolor='blue', alpha=0.75)
ax.add_patch(circ)
plt.show()

# %%
# 如果更改图形大小，圆圈不会改变其绝对位置，并且可能被裁剪。

fig, ax = plt.subplots(figsize=(7, 2))
x, y = 10*np.random.rand(2, 1000)
# 在数据坐标中绘制一些数据点
ax.plot(x, y*10., 'go', alpha=0.2)
# 添加一个圆在固定坐标系中
circ = mpatches.Circle((2.5, 2), 1.0, transform=fig.dpi_scale_trans,
                       facecolor='blue', alpha=0.75)
ax.add_patch(circ)
plt.show()

# %%
# 另一个用途是在坐标轴上的数据点周围放置具有固定物理尺寸的补丁。这里我们将两个变换相加。
# 第一个设置椭圆的缩放大小，第二个设置其位置。然后椭圆被放置在原点，
# 然后我们使用助手变换 :class:`~matplotlib.transforms.ScaledTranslation`
# 将其移动到正确的位置，位于 ``ax.transData`` 坐标系中。
# 这个助手被实例化为::
#
#   trans = ScaledTranslation(xt, yt, scale_trans)
#
# 其中 *xt* 和 *yt* 是平移偏移量，*scale_trans* 是一个在转换时缩放 *xt* 和 *yt* 的变换，
# 然后应用偏移量。
#
# 注意下面变换的加法操作。
# 这段代码的含义是：首先应用缩放变换 ``fig.dpi_scale_trans``，使椭圆具有正确的大小，
# 但仍然以 (0, 0) 为中心，
# 创建一个新的图形和轴对象
fig, ax = plt.subplots()

# 用给定的数据绘制散点图
xdata, ydata = (0.2, 0.7), (0.5, 0.5)
ax.plot(xdata, ydata, "o")

# 设置 x 轴的显示范围
ax.set_xlim((0, 1))

# 创建一个偏移变换对象，将数据点 (xdata[0], ydata[0]) 转换到显示空间
trans = (fig.dpi_scale_trans +
         transforms.ScaledTranslation(xdata[0], ydata[0], ax.transData))

# 创建一个椭圆对象，围绕指定点绘制，其大小为150x130点，旋转角度为40度
circle = mpatches.Ellipse((0, 0), 150/72, 130/72, angle=40,
                          fill=None, transform=trans)

# 将椭圆对象添加到轴上
ax.add_patch(circle)

# 显示图形
plt.show()
# 在坐标轴上绘制线条，设置线宽为3，颜色为灰色，
# 使用阴影变换进行坐标变换，设置绘制顺序为当前线条 zorder 属性的一半
ax.plot(x, y, lw=3, color='gray',
        transform=shadow_transform,
        zorder=0.5*line.get_zorder())

# 设置图表的标题
ax.set_title('creating a shadow effect with an offset transform')
# 显示图表
plt.show()
# The final piece is the ``self.transScale`` attribute, which is
# responsible for the optional non-linear scaling of the data, e.g., for
# logarithmic axes.  When an Axes is initially setup, this is just set to
# the identity transform, since the basic Matplotlib axes has linear
# scale, but when you call a logarithmic scaling function like
# :meth:`~matplotlib.axes.Axes.semilogx` or explicitly set the scale to
# logarithmic with :meth:`~matplotlib.axes.Axes.set_xscale`, then the
# ``ax.transScale`` attribute is set to handle the nonlinear projection.
# The scales transforms are properties of the respective ``xaxis`` and
# ``yaxis`` :class:`~matplotlib.axis.Axis` instances.  For example, when
# you call ``ax.set_xscale('log')``, the xaxis updates its scale to a
# :class:`matplotlib.scale.LogScale` instance.

# For non-separable axes the PolarAxes, there is one more piece to
# consider, the projection transformation.  The ``transData``
# :class:`matplotlib.projections.polar.PolarAxes` is similar to that for
# the typical separable matplotlib Axes, with one additional piece
# ``transProjection``::

#        self.transData = (
#            self.transScale + self.transShift + self.transProjection +
#            (self.transProjectionAffine + self.transWedge + self.transAxes))

# ``transProjection`` handles the projection from the space,
# e.g., latitude and longitude for map data, or radius and theta for polar
# data, to a separable Cartesian coordinate system.  There are several
# projection examples in the :mod:`matplotlib.projections` package, and the
# best way to learn more is to open the source for those packages and
# see how to make your own, since Matplotlib supports extensible axes
# and projections.  Michael Droettboom has provided a nice tutorial
# example of creating a Hammer projection axes; see
# :doc:`/gallery/misc/custom_projection`.
```