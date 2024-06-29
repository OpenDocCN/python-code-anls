# `D:\src\scipysrc\matplotlib\lib\matplotlib\quiver.py`

```py
"""
支持绘制向量场的功能。

目前包括 Quiver 和 Barb。Quiver 在向量方向上绘制箭头，箭头的大小与向量的大小相关。

Barb 类似于 Quiver，指示向量的方向，但向量的大小通过 barbs 或 flags 的图示表示。

这也将成为标准偏差椭圆等内容的容器，这些内容可以很容易地从 Quiver 代码中衍生出来。
"""

import math  # 导入数学函数库

import numpy as np  # 导入 numpy 库
from numpy import ma  # 导入 numpy 的 ma 模块

from matplotlib import _api, cbook, _docstring  # 导入 matplotlib 内部模块和文档字符串模块
import matplotlib.artist as martist  # 导入 matplotlib 的 artist 模块
import matplotlib.collections as mcollections  # 导入 matplotlib 的 collections 模块
from matplotlib.patches import CirclePolygon  # 从 matplotlib 的 patches 模块导入 CirclePolygon 类
import matplotlib.text as mtext  # 导入 matplotlib 的 text 模块
import matplotlib.transforms as transforms  # 导入 matplotlib 的 transforms 模块


_quiver_doc = """
绘制二维箭头场。

调用签名：

  quiver([X, Y], U, V, [C], /, **kwargs)

*X*, *Y* 定义箭头的位置，*U*, *V* 定义箭头的方向，*C* 可选设置箭头的颜色。参数 *X*, *Y*, *U*, *V*, *C* 是位置参数。

**箭头长度**

默认设置会自动调整箭头的长度至合理大小。若要更改此行为，请参见 *scale* 和 *scale_units* 参数。

**箭头形状**

箭头形状由 *width*, *headwidth*, *headlength* 和 *headaxislength* 决定。详见下文的注释。

**箭头样式**

每个箭头在内部由填充的多边形表示，边缘默认的线宽为 0。因此，箭头更像是填充区域，而不是具有头部的线条，`.PolyCollection` 的属性如 *linewidth*, *edgecolor*,
*facecolor* 等相应地生效。


参数
----------
X, Y : 1D 或 2D 数组样式，可选
    箭头位置的 x 和 y 坐标。

    如果未给出，将根据 *U* 和 *V* 的维度生成均匀整数网格。

    如果 *X* 和 *Y* 是 1D，但 *U*, *V* 是 2D，则使用 ``X, Y = np.meshgrid(X, Y)`` 将其扩展为 2D。
    在此情况下，“len(X)” 和 “len(Y)” 必须与 *U* 和 *V* 的列和行维度匹配。

U, V : 1D 或 2D 数组样式
    箭头向量的 x 和 y 方向分量。这些分量的解释（数据空间或屏幕空间）取决于 *angles*。

    *U* 和 *V* 必须具有相同数量的元素，与 *X*, *Y* 中箭头位置的数量匹配。*U* 和 *V* 可能是掩码的。在 *U*, *V* 和 *C* 中的任何位置掩码将不会被绘制。

C : 1D 或 2D 数组样式，可选
    数值数据，通过 *norm* 和 *cmap* 进行颜色映射来定义箭头的颜色。

    不支持显式颜色设置。如果想直接设置颜色，请使用 *color* 代替。*C* 的大小必须与箭头位置的数量匹配。

angles : {'uv', 'xy'} 或数组样式，默认值为 'uv'
    确定箭头角度的方法。

"""
    # 'uv': 屏幕坐标系中的箭头方向。如果箭头表示的是不基于 X、Y 数据坐标的量，请使用此选项。
    
    # 如果 U == V，箭头在绘图中的方向是相对于水平轴逆时针 45 度（向右为正方向）。

    # 'xy': 数据坐标系中的箭头方向，即箭头从 (x, y) 指向 (x+u, y+v)。例如用于绘制梯度场。

    # 可以通过明确指定角度数组来设置任意角度，角度是相对于水平轴逆时针方向的度数。

    # 在这种情况下，U、V 仅用于确定箭头的长度。

    # 注意：如果反转了数据轴，那么只有在 angles='xy' 时箭头才会相应地反转。
pivot : {'tail', 'mid', 'middle', 'tip'}, default: 'tail'
    # 箭头锚定在网格上的部分。箭头围绕此点旋转。
    # 可选值包括 'tail', 'mid', 'middle', 'tip'，默认为 'tail'。
    # 'mid' 是 'middle' 的同义词。

scale : float, optional
    # 反比例缩放箭头的长度。
    # 数据单位每箭头长度单位的数量，例如每个绘图宽度的米/秒；较小的 scale 参数使箭头变长。默认为 *None*。
    # 如果为 *None*，将使用简单的自动缩放算法，基于平均向量长度和向量数量。箭头长度单位由 scale_units 参数给出。

scale_units : {'width', 'height', 'dots', 'inches', 'x', 'y', 'xy'}, optional
    # 如果 *scale* 参数为 *None*，则为箭头长度单位。默认为 *None*。
    # 例如，如果 *scale_units* 为 'inches'，*scale* 为 2.0，并且 ``(u, v) = (1, 0)``，
    # 那么向量长度将为 0.5 英寸。
    # 如果 *scale_units* 为 'width' 或 'height'，则向量将为轴宽度/高度的一半。
    # 如果 *scale_units* 为 'x'，则向量将为 x 轴的 0.5 单位。
    # 若要在 x-y 平面上绘制向量，并且 u 和 v 具有与 x 和 y 相同的单位，请使用
    # ``angles='xy', scale_units='xy', scale=1``。

units : {'width', 'height', 'dots', 'inches', 'x', 'y', 'xy'}, default: 'width'
    # 影响箭头大小（除长度外）。特别是，轴的宽度以此单位的倍数测量。
    # 支持的值包括：
    # - 'width', 'height'：轴的宽度或高度。
    # - 'dots', 'inches'：基于图像 dpi 的像素或英寸。
    # - 'x', 'y', 'xy'：*X*、*Y* 或 :math:`\\sqrt{X^2 + Y^2}` 中的单位。

    # 以下表格总结了这些值在缩放和图形大小变化下如何影响可见的箭头大小：
    # =================  =================   ==================
    # units              zoom                figure size change
    # =================  =================   ==================
    # 'x', 'y', 'xy'     箭头大小随缩放而变化   —
    # 'width', 'height'  —                   箭头大小随图形大小变化
    # 'dots', 'inches'   —                   —
    # =================  =================   ==================

width : float, optional
    # 箭轴宽度，以箭头单位为准。所有头部参数都是相对于 *width* 的。

    # 默认值取决于上述的 *units* 选择和向量数量；
    # 一个典型的起始值大约为绘图宽度的 0.005 倍。

headwidth : float, default: 3
    # 头部宽度，作为轴宽度的倍数。详见下面的注释。

headlength : float, default: 5
    # 头部长度，作为轴宽度的倍数。详见下面的注释。

headaxislength : float, default: 4.5
    # 在轴交点处的头部长度，作为轴宽度的倍数。详见下面的注释。

minshaft : float, default: 1
    # 箭头缩放到以下长度以下的长度，以头部长度单位计。不要将其设置为小于 1，否则小箭头看起来会很糟糕！

minlength : float, default: 1
    # 箭头的最小长度。
    Minimum length as a multiple of shaft width; if an arrow length
    is less than this, plot a dot (hexagon) of this diameter instead.
color : :mpltype:`color` or list :mpltype:`color`, optional
    # 箭头的显示颜色，可以是单一颜色或颜色列表。如果已设置 *C* 参数，则此参数无效。
    # 这是 `.PolyCollection` 的 *facecolor* 参数的同义词。

Other Parameters
----------------
data : indexable object, optional
    # 数据参数的占位符

**kwargs : `~matplotlib.collections.PolyCollection` properties, optional
    # 所有其他关键字参数都传递给 `.PolyCollection` 对象：
    # %(PolyCollection:kwdoc)s

Returns
-------
`~matplotlib.quiver.Quiver`
    # 返回一个 `~matplotlib.quiver.Quiver` 对象

See Also
--------
.Axes.quiverkey : Add a key to a quiver plot.

Notes
-----
**Arrow shape**

The arrow is drawn as a polygon using the nodes as shown below. The values
*headwidth*, *headlength*, and *headaxislength* are in units of *width*.

.. image:: /_static/quiver_sizes.svg
   :width: 500px

   # 箭头形状的说明图像，显示了节点的使用方式，其中 *headwidth*、*headlength* 和 *headaxislength* 的单位是 *width*。

   # 默认情况下会得到稍微倾斜的箭头。以下是一些指南，如何得到其他的箭头头部形状：

   - 要使箭头头部成为三角形，将 *headaxislength* 设置为与 *headlength* 相同。
   - 要使箭头更尖锐，减小 *headwidth* 或增加 *headlength* 和 *headaxislength*。
   - 要使箭头头部相对于箭身变小，按比例缩小所有头部参数。
   - 要完全移除箭头头部，将所有 *head* 参数设为 0。
   - 要得到菱形头部，使 *headaxislength* 大于 *headlength*。
   - 警告：对于 *headaxislength* < (*headlength* / *headwidth*)，"headaxis" 节点（即连接头部和箭身的节点）将向前方突出，使箭头头部看起来断裂。

""" % _docstring.interpd.params

_docstring.interpd.update(quiver_doc=_quiver_doc)


class QuiverKey(martist.Artist):
    """Labelled arrow for use as a quiver plot scale key."""
    halign = {'N': 'center', 'S': 'center', 'E': 'left', 'W': 'right'}
    valign = {'N': 'bottom', 'S': 'top', 'E': 'center', 'W': 'center'}
    pivot = {'N': 'middle', 'S': 'middle', 'E': 'tip', 'W': 'tail'}

    @property
    def labelsep(self):
        return self._labelsep_inches * self.Q.axes.figure.dpi
    # 初始化方法，检查是否需要重新初始化对象
    def _init(self):
        # 如果条件为真，重新初始化对象
        if True:  # self._dpi_at_last_init != self.axes.figure.dpi
            # 检查是否需要重新初始化 Q 对象
            if self.Q._dpi_at_last_init != self.Q.axes.figure.dpi:
                self.Q._init()
            # 设置对象的变换
            self._set_transform()
            # 在设置属性的上下文中，设置属性 pivot 和 Umask
            with cbook._setattr_cm(self.Q, pivot=self.pivot[self.labelpos],
                                   Umask=ma.nomask):
                # 计算向量的 u 和 v 坐标
                u = self.U * np.cos(np.radians(self.angle))
                v = self.U * np.sin(np.radians(self.angle))
                # 根据坐标值创建顶点
                self.verts = self.Q._make_verts([[0., 0.]],
                                                np.array([u]), np.array([v]), 'uv')
            # 更新关键字参数
            kwargs = self.Q.polykw
            kwargs.update(self.kw)
            # 创建多边形集合对象 vector
            self.vector = mcollections.PolyCollection(
                self.verts,
                offsets=[(self.X, self.Y)],
                offset_transform=self.get_transform(),
                **kwargs)
            # 如果颜色不为 None，设置对象的颜色
            if self.color is not None:
                self.vector.set_color(self.color)
            # 设置对象的变换
            self.vector.set_transform(self.Q.get_transform())
            # 记录最后一次初始化时的 DPI 值
            self._dpi_at_last_init = self.Q.axes.figure.dpi

    # 返回文本偏移量的字典，根据 labelpos 不同而有所不同
    def _text_shift(self):
        return {
            "N": (0, +self.labelsep),
            "S": (0, -self.labelsep),
            "E": (+self.labelsep, 0),
            "W": (-self.labelsep, 0),
        }[self.labelpos]

    # 绘制方法，用于绘制对象及其文本
    @martist.allow_rasterization
    def draw(self, renderer):
        # 初始化对象
        self._init()
        # 绘制向量对象
        self.vector.draw(renderer)
        # 计算文本的位置并绘制文本
        pos = self.get_transform().transform((self.X, self.Y))
        self.text.set_position(pos + self._text_shift())
        self.text.draw(renderer)
        # 更新对象状态为非过时
        self.stale = False

    # 设置对象的变换方法
    def _set_transform(self):
        # 设置对象的变换，根据坐标系统设置不同的变换
        self.set_transform(_api.check_getitem({
            "data": self.Q.axes.transData,
            "axes": self.Q.axes.transAxes,
            "figure": self.Q.axes.figure.transFigure,
            "inches": self.Q.axes.figure.dpi_scale_trans,
        }, coordinates=self.coord))

    # 设置对象所属的 Figure
    def set_figure(self, fig):
        # 调用父类方法设置 Figure
        super().set_figure(fig)
        # 设置对象的文本所属的 Figure
        self.text.set_figure(fig)

    # 检查鼠标事件是否包含在对象内
    def contains(self, mouseevent):
        # 如果鼠标事件发生在不同的画布上，返回 False
        if self._different_canvas(mouseevent):
            return False, {}
        # 检查鼠标事件是否发生在文本或向量对象内
        if (self.text.contains(mouseevent)[0] or
                self.vector.contains(mouseevent)[0]):
            return True, {}
        # 如果事件不在对象内，返回 False
        return False, {}
# 定义一个辅助函数，用于解析用于彩色矢量图的位置参数

def _parse_args(*args, caller_name='function'):
    """
    Helper function to parse positional parameters for colored vector plots.

    This is currently used for Quiver and Barbs.

    Parameters
    ----------
    *args : list
        list of 2-5 arguments. Depending on their number they are parsed to::

            U, V
            U, V, C
            X, Y, U, V
            X, Y, U, V, C

    caller_name : str
        Name of the calling method (used in error messages).
    """
    # 初始化变量 X, Y, C 为 None
    X = Y = C = None

    # 获取参数个数
    nargs = len(args)

    # 根据参数个数不同进行解析
    if nargs == 2:
        # 使用 np.atleast_1d 处理参数，支持处理标量参数和掩码数组
        U, V = np.atleast_1d(*args)
    elif nargs == 3:
        U, V, C = np.atleast_1d(*args)
    elif nargs == 4:
        X, Y, U, V = np.atleast_1d(*args)
    elif nargs == 5:
        X, Y, U, V, C = np.atleast_1d(*args)
    else:
        # 如果参数个数不在2到5之间，则抛出异常
        raise _api.nargs_error(caller_name, takes="from 2 to 5", given=nargs)

    # 确定 U 的行数和列数
    nr, nc = (1, U.shape[0]) if U.ndim == 1 else U.shape

    # 如果 X 不为 None，则处理 X, Y 的形状匹配
    if X is not None:
        X = X.ravel()
        Y = Y.ravel()
        # 检查 X 和 Y 的长度是否与 nr 和 nc 匹配
        if len(X) == nc and len(Y) == nr:
            X, Y = [a.ravel() for a in np.meshgrid(X, Y)]
        elif len(X) != len(Y):
            # 如果 X 和 Y 的长度不一致，则抛出异常
            raise ValueError('X and Y must be the same size, but '
                             f'X.size is {X.size} and Y.size is {Y.size}.')
    else:
        # 如果 X 为 None，则生成索引网格
        indexgrid = np.meshgrid(np.arange(nc), np.arange(nr))
        X, Y = [np.ravel(a) for a in indexgrid]

    # 返回解析后的结果 X, Y, U, V, C
    # 对于 U, V, C 的大小验证交给 set_UVC 方法处理
    return X, Y, U, V, C

# _parse_args 函数到此结束

def _check_consistent_shapes(*arrays):
    # 收集所有数组的形状
    all_shapes = {a.shape for a in arrays}
    # 如果数组形状不一致，则抛出异常
    if len(all_shapes) != 1:
        raise ValueError('The shapes of the passed in arrays do not match')

# _check_consistent_shapes 函数到此结束

class Quiver(mcollections.PolyCollection):
    """
    Specialized PolyCollection for arrows.

    The only API method is set_UVC(), which can be used
    to change the size, orientation, and color of the
    arrows; their locations are fixed when the class is
    instantiated.  Possibly this method will be useful
    in animations.

    Much of the work in this class is done in the draw()
    method so that as much information as possible is available
    about the plot.  In subsequent draw() calls, recalculation
    is limited to things that might have changed, so there
    should be no performance penalty from putting the calculations
    in the draw() method.
    """

    _PIVOT_VALS = ('tail', 'middle', 'tip')

    # 使用 _quiver_doc 替换 _docstring.Substitution
    @_docstring.Substitution(_quiver_doc)
    def __init__(self, ax, *args,
                 scale=None, headwidth=3, headlength=5, headaxislength=4.5,
                 minshaft=1, minlength=1, units='width', scale_units=None,
                 angles='uv', width=None, color='k', pivot='tail', **kwargs):
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %s
        """
        self._axes = ax  # The attr actually set by the Artist.axes property.
        X, Y, U, V, C = _parse_args(*args, caller_name='quiver')
        self.X = X  # Store X coordinates for plotting arrows
        self.Y = Y  # Store Y coordinates for plotting arrows
        self.XY = np.column_stack((X, Y))  # Combine X and Y into a 2D array
        self.N = len(X)  # Number of arrows to plot
        self.scale = scale  # Scaling factor for arrow lengths
        self.headwidth = headwidth  # Width of arrow heads
        self.headlength = float(headlength)  # Length of arrow heads
        self.headaxislength = headaxislength  # Length of arrow head axis
        self.minshaft = minshaft  # Minimum length of arrow shafts
        self.minlength = minlength  # Minimum length of arrows
        self.units = units  # Units for scaling arrows
        self.scale_units = scale_units  # Units for scale
        self.angles = angles  # Specifies arrow direction ('uv' for unit vector)
        self.width = width  # Width of the arrow lines

        if pivot.lower() == 'mid':
            pivot = 'middle'
        self.pivot = pivot.lower()  # Set pivot point for arrows
        _api.check_in_list(self._PIVOT_VALS, pivot=self.pivot)  # Validate pivot point

        self.transform = kwargs.pop('transform', ax.transData)  # Transformation for arrow placement
        kwargs.setdefault('facecolors', color)  # Set default face color for arrows
        kwargs.setdefault('linewidths', (0,))  # Set default linewidths
        super().__init__([], offsets=self.XY, offset_transform=self.transform,
                         closed=False, **kwargs)  # Initialize the parent class with necessary parameters
        self.polykw = kwargs  # Store additional keyword arguments for polygon properties
        self.set_UVC(U, V, C)  # Set the magnitude (U, V) and color (C) of arrows
        self._dpi_at_last_init = None  # Initialize DPI attribute

    def _init(self):
        """
        Initialization delayed until first draw;
        allow time for axes setup.
        """
        # It seems that there are not enough event notifications
        # available to have this work on an as-needed basis at present.
        if True:  # Check if DPI has changed since last initialization
            trans = self._set_transform()  # Set transformation for the arrow object
            self.span = trans.inverted().transform_bbox(self.axes.bbox).width  # Calculate span of transformed axes
            if self.width is None:
                sn = np.clip(math.sqrt(self.N), 8, 25)  # Calculate suggested number of arrows
                self.width = 0.06 * self.span / sn  # Set default width based on span and number of arrows

            # _make_verts sets self.scale if not already specified
            if (self._dpi_at_last_init != self.axes.figure.dpi
                    and self.scale is None):
                self._make_verts(self.XY, self.U, self.V, self.angles)  # Generate vertices for arrows

            self._dpi_at_last_init = self.axes.figure.dpi  # Update DPI attribute

    def get_datalim(self, transData):
        trans = self.get_transform()  # Get current transformation
        offset_trf = self.get_offset_transform()  # Get offset transformation
        full_transform = (trans - transData) + (offset_trf - transData)  # Combine transformations
        XY = full_transform.transform(self.XY)  # Transform XY data points
        bbox = transforms.Bbox.null()  # Initialize bounding box
        bbox.update_from_data_xy(XY, ignore=True)  # Update bounding box from transformed data
        return bbox  # Return the calculated bounding box

    @martist.allow_rasterization
    def draw(self, renderer):
        # 初始化绘图设置
        self._init()
        # 根据给定的XY坐标、U和V数组以及角度生成顶点信息
        verts = self._make_verts(self.XY, self.U, self.V, self.angles)
        # 设置顶点信息并绘制多边形集合
        self.set_verts(verts, closed=False)
        # 调用父类方法进行绘制
        super().draw(renderer)
        # 标记绘图状态为非stale，表示绘图已更新
        self.stale = False

    def set_UVC(self, U, V, C=None):
        # 确保我们拥有一个副本而不是一个可能在draw()之前更改的数组引用
        U = ma.masked_invalid(U, copy=True).ravel()
        V = ma.masked_invalid(V, copy=True).ravel()
        if C is not None:
            C = ma.masked_invalid(C, copy=True).ravel()
        for name, var in zip(('U', 'V', 'C'), (U, V, C)):
            # 检查每个变量的大小是否与箭头位置数量self.N匹配
            if not (var is None or var.size == self.N or var.size == 1):
                raise ValueError(f'Argument {name} has a size {var.size}'
                                 f' which does not match {self.N},'
                                 ' the number of arrow positions')

        # 创建掩码，包含U和V的掩码
        mask = ma.mask_or(U.mask, V.mask, copy=False, shrink=True)
        if C is not None:
            # 如果C不为空，继续添加C的掩码
            mask = ma.mask_or(mask, C.mask, copy=False, shrink=True)
            if mask is ma.nomask:
                C = C.filled()
            else:
                C = ma.array(C, mask=mask, copy=False)
        # 用填充值1填充U和V中的掩码
        self.U = U.filled(1)
        self.V = V.filled(1)
        self.Umask = mask
        if C is not None:
            # 如果C不为空，设置数组
            self.set_array(C)
        # 标记状态为stale，表示数据已更新
        self.stale = True

    def _dots_per_unit(self, units):
        """返回一个比例因子，用于从单位转换为像素。"""
        bb = self.axes.bbox
        vl = self.axes.viewLim
        # 检查并获取单位对应的比例因子
        return _api.check_getitem({
            'x': bb.width / vl.width,
            'y': bb.height / vl.height,
            'xy': np.hypot(*bb.size) / np.hypot(*vl.size),
            'width': bb.width,
            'height': bb.height,
            'dots': 1.,
            'inches': self.axes.figure.dpi,
        }, units=units)

    def _set_transform(self):
        """
        将PolyCollection的变换设置为从箭头宽度单位到像素的转换。
        """
        dx = self._dots_per_unit(self.units)
        self._trans_scale = dx  # 每个箭头宽度单位的像素数
        trans = transforms.Affine2D().scale(dx)
        self.set_transform(trans)
        return trans

    # 计算从(X, Y)到(X+U, Y+V)段的角度和长度
    def _angles_lengths(self, XY, U, V, eps=1):
        xy = self.axes.transData.transform(XY)
        uv = np.column_stack((U, V))
        xyp = self.axes.transData.transform(XY + eps * uv)
        dxy = xyp - xy
        angles = np.arctan2(dxy[:, 1], dxy[:, 0])
        lengths = np.hypot(*dxy.T) / eps
        return angles, lengths

    # XY被堆叠为[X, Y]。
    # 详见quiver()文档，了解X、Y、U、V和angles的含义。
    def _make_verts(self, XY, U, V, angles):
        # 将 U 和 V 合成复数表示的向量 uv
        uv = (U + V * 1j)
        # 如果 angles 是字符串 'xy' 并且 scale_units 也是 'xy'
        str_angles = angles if isinstance(angles, str) else ''
        if str_angles == 'xy' and self.scale_units == 'xy':
            # 这里 eps 设为 1，以确保通过对 X、Y 数组进行差分得到 U、V 后，
            # 向量能够连接点，不受轴缩放（包括对数轴）影响。
            angles, lengths = self._angles_lengths(XY, U, V, eps=1)
        elif str_angles == 'xy' or self.scale_units == 'xy':
            # 根据绘图区域的范围计算 eps，避免由于在一个较大的数上加上一个较小的数
            # 而产生的舍入误差。
            eps = np.abs(self.axes.dataLim.extents).max() * 0.001
            angles, lengths = self._angles_lengths(XY, U, V, eps=eps)

        # 如果 str_angles 是非空字符串且 scale_units 是 'xy'
        if str_angles and self.scale_units == 'xy':
            a = lengths
        else:
            a = np.abs(uv)

        # 如果 scale 为 None
        if self.scale is None:
            sn = max(10, math.sqrt(self.N))
            # 如果 Umask 不是 ma.nomask
            if self.Umask is not ma.nomask:
                amean = a[~self.Umask].mean()
            else:
                amean = a.mean()
            # 粗略的自动缩放
            # scale 是典型箭头长度与箭头宽度的倍数
            scale = 1.8 * amean * sn / self.span

        # 如果 scale_units 为 None
        if self.scale_units is None:
            # 如果 scale 为 None
            if self.scale is None:
                self.scale = scale
            widthu_per_lenu = 1.0
        else:
            # 如果 scale_units 是 'xy'
            if self.scale_units == 'xy':
                dx = 1
            else:
                dx = self._dots_per_unit(self.scale_units)
            widthu_per_lenu = dx / self._trans_scale
            # 如果 scale 为 None
            if self.scale is None:
                self.scale = scale * widthu_per_lenu

        # 计算长度
        length = a * (widthu_per_lenu / (self.scale * self.width))
        # 计算箭头的端点坐标 X, Y
        X, Y = self._h_arrows(length)

        # 根据 str_angles 设置箭头的角度 theta
        if str_angles == 'xy':
            theta = angles
        elif str_angles == 'uv':
            theta = np.angle(uv)
        else:
            theta = ma.masked_invalid(np.deg2rad(angles)).filled(0)
        theta = theta.reshape((-1, 1))  # 用于广播操作

        # 计算箭头的位置 XY
        xy = (X + Y * 1j) * np.exp(1j * theta) * self.width
        XY = np.stack((xy.real, xy.imag), axis=2)

        # 如果 Umask 不是 ma.nomask，则将对应位置的 XY 设为 masked
        if self.Umask is not ma.nomask:
            XY = ma.array(XY)
            XY[self.Umask] = ma.masked
            # 由于最终路径中会出现 nan，因此可能更高效地处理它们。

        return XY
    def _h_arrows(self, length):
        """Length is in arrow width units."""
        # 定义最小箭头的长度
        minsh = self.minshaft * self.headlength
        # 获取长度数组的大小
        N = len(length)
        # 将长度数组重塑为列向量
        length = length.reshape(N, 1)
        # 限制长度数组的取值范围，避免在渲染时出现像素值溢出的渲染错误
        np.clip(length, 0, 2 ** 16, out=length)
        
        # x, y: 定义普通水平箭头的形状
        x = np.array([0, -self.headaxislength,
                      -self.headlength, 0],
                     np.float64)
        x = x + np.array([0, 1, 1, 1]) * length
        y = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        y = np.repeat(y[np.newaxis, :], N, axis=0)
        
        # x0, y0: 定义没有箭杆的箭头，用于较短的向量
        x0 = np.array([0, minsh - self.headaxislength,
                       minsh - self.headlength, minsh], np.float64)
        y0 = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        ii = [0, 1, 2, 3, 2, 1, 0, 0]
        X = x[:, ii]
        Y = y[:, ii]
        Y[:, 3:-1] *= -1
        X0 = x0[ii]
        Y0 = y0[ii]
        Y0[3:-1] *= -1
        
        # 根据箭头长度与最小箭杆长度比例来缩放箭头
        shrink = length / minsh if minsh != 0. else 0.
        X0 = shrink * X0[np.newaxis, :]
        Y0 = shrink * Y0[np.newaxis, :]
        
        # 判断哪些箭头长度太短，选择相应的箭头形状
        short = np.repeat(length < minsh, 8, axis=1)
        np.copyto(X, X0, where=short)
        np.copyto(Y, Y0, where=short)
        
        # 根据旋转中心设置箭头的位置
        if self.pivot == 'middle':
            X -= 0.5 * X[:, 3, np.newaxis]
        elif self.pivot == 'tip':
            X = X - X[:, 3, np.newaxis]
        elif self.pivot != 'tail':
            _api.check_in_list(["middle", "tip", "tail"], pivot=self.pivot)

        # 如果箭头长度过短，则使用七边形点表示箭头
        tooshort = length < self.minlength
        if tooshort.any():
            th = np.arange(0, 8, 1, np.float64) * (np.pi / 3.0)
            x1 = np.cos(th) * self.minlength * 0.5
            y1 = np.sin(th) * self.minlength * 0.5
            X1 = np.repeat(x1[np.newaxis, :], N, axis=0)
            Y1 = np.repeat(y1[np.newaxis, :], N, axis=0)
            tooshort = np.repeat(tooshort, 8, 1)
            np.copyto(X, X1, where=tooshort)
            np.copyto(Y, Y1, where=tooshort)
        
        # 将处理结果返回，由调用者函数 _make_verts 进行进一步处理
        # 处理遮罩的操作在 _make_verts 中完成
        return X, Y
# 多行字符串，包含有关绘制风羽图的文档字符串
_barbs_doc = r"""
Plot a 2D field of wind barbs.

Call signature::

  barbs([X, Y], U, V, [C], /, **kwargs)

Where *X*, *Y* define the barb locations, *U*, *V* define the barb
directions, and *C* optionally sets the color.

The arguments *X*, *Y*, *U*, *V*, *C* are positional-only and may be
1D or 2D. *U*, *V*, *C* may be masked arrays, but masked *X*, *Y*
are not supported at present.

Barbs are traditionally used in meteorology as a way to plot the speed
and direction of wind observations, but can technically be used to
plot any two dimensional vector quantity.  As opposed to arrows, which
give vector magnitude by the length of the arrow, the barbs give more
quantitative information about the vector magnitude by putting slanted
lines or a triangle for various increments in magnitude, as show
schematically below::

  :                   /\    \
  :                  /  \    \
  :                 /    \    \    \
  :                /      \    \    \
  :               ------------------------------

The largest increment is given by a triangle (or "flag"). After those
come full lines (barbs). The smallest increment is a half line.  There
is only, of course, ever at most 1 half line.  If the magnitude is
small and only needs a single half-line and no full lines or
triangles, the half-line is offset from the end of the barb so that it
can be easily distinguished from barbs with a single full line.  The
magnitude for the barb shown above would nominally be 65, using the
standard increments of 50, 10, and 5.

See also https://en.wikipedia.org/wiki/Wind_barb.

Parameters
----------
X, Y : 1D or 2D array-like, optional
    The x and y coordinates of the barb locations. See *pivot* for how the
    barbs are drawn to the x, y positions.

    If not given, they will be generated as a uniform integer meshgrid based
    on the dimensions of *U* and *V*.

    If *X* and *Y* are 1D but *U*, *V* are 2D, *X*, *Y* are expanded to 2D
    using ``X, Y = np.meshgrid(X, Y)``. In this case ``len(X)`` and ``len(Y)``
    must match the column and row dimensions of *U* and *V*.

U, V : 1D or 2D array-like
    The x and y components of the barb shaft.

C : 1D or 2D array-like, optional
    Numeric data that defines the barb colors by colormapping via *norm* and
    *cmap*.

    This does not support explicit colors. If you want to set colors directly,
    use *barbcolor* instead.

length : float, default: 7
    Length of the barb in points; the other parts of the barb
    are scaled against this.

pivot : {'tip', 'middle'} or float, default: 'tip'
    The part of the arrow that is anchored to the *X*, *Y* grid. The barb
    rotates about this point. This can also be a number, which shifts the
    start of the barb that many points away from grid point.

barbcolor : :mpltype:`color` or color sequence
    The color of all parts of the barb except for the flags.  This parameter
    is analogous to the *edgecolor* parameter for polygons, which can be used
"""
    # 导入必要的库：matplotlib.pyplot作为plt，numpy作为np
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成一个包含x轴数据的数组，范围是从0到5，步长是0.1
    x = np.arange(0., 5., 0.1)

    # 使用x数组作为输入，生成对应的sin(x)和cos(x)的值，并保存在y1和y2中
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 创建一个新的图形
    fig = plt.figure()

    # 绘制第一个子图，设置图形的大小和位置，颜色为蓝色，线型为实线，标记为圆圈
    ax1 = fig.add_subplot(211)
    ax1.plot(x, y1, 'b-')

    # 绘制第二个子图，设置图形的大小和位置，颜色为红色，线型为虚线，标记为方形
    ax2 = fig.add_subplot(212)
    ax2.plot(x, y2, 'r--')

    # 显示图形
    plt.show()
"""
flagcolor : :mpltype:`color` or color sequence
    标志的颜色。可以是单一颜色或颜色序列。类似于多边形的 *facecolor* 参数，可以代替使用。
    如果设置了此参数，将覆盖 *facecolor*。如果未设置此参数且未设置 *C* 参数，则 *flagcolor* 将与 *barbcolor* 相同。
    如果设置了 *C* 参数，则 *flagcolor* 将不起作用。

sizes : dict, optional
    系数字典，指定了各个特征与风羽长度的比率。只包括希望覆盖的值。

    - 'spacing' - 特征之间的间距（标志、全/半风羽）
    - 'height' - 标志或全风羽的高度（从轴到顶部的距离）
    - 'width' - 标志的宽度，全风羽宽度的两倍
    - 'emptybarb' - 用于低速度的圆的半径

fill_empty : bool, default: False
    是否填充绘制的空风羽（圆）以使用标志颜色。如果不填充，中心将是透明的。

rounding : bool, default: True
    是否在分配风羽组件时对向量大小进行四舍五入。如果为 True，则大小将四舍五入为半风羽增量的最近倍数。
    如果为 False，则大小将简单地截断为最接近的较低倍数。

barb_increments : dict, optional
    增量字典，指定与风羽不同部分相关联的值。只包括希望覆盖的值。

    - 'half' - 半风羽（默认为 5）
    - 'full' - 全风羽（默认为 10）
    - 'flag' - 标志（默认为 50）

flip_barb : bool or array-like of bool, default: False
    是否使风羽和标志指向相反方向。正常行为是风羽和标志指向右侧（北半球风羽指向低压）。
    单个值适用于所有风羽。通过传递与 *U* 和 *V* 相同大小的布尔数组，可以翻转单个风羽。

Returns
-------
barbs : `~matplotlib.quiver.Barbs`
    返回一个 `~matplotlib.quiver.Barbs` 对象。

Other Parameters
----------------
data : indexable object, optional
    DATA_PARAMETER_PLACEHOLDER

**kwargs
    可以使用 `.PolyCollection` 的关键字参数进一步自定义风羽：

    %(PolyCollection:kwdoc)s
""" % _docstring.interpd.params

_docstring.interpd.update(barbs_doc=_barbs_doc)


class Barbs(mcollections.PolyCollection):
    """
    专门用于风羽的 `PolyCollection`。

    唯一的 API 方法是 :meth:`set_UVC`，可以用于改变箭头的大小、方向和颜色。可以使用 :meth:`set_offsets`
    改变位置。这个方法在动画中可能很有用。

    有一个内部函数 :meth:`_find_tails`，根据向量大小确定应该放置在风羽上的内容。
    """
    # 定义一个类，用于绘制风羽（barbs），基于给定的Axes实例和其他参数
    @_docstring.interpd
    def __init__(self, ax, *args,
                 pivot='tip', length=7, barbcolor=None, flagcolor=None,
                 sizes=None, fill_empty=False, barb_increments=None,
                 rounding=True, flip_barb=False, **kwargs):
        """
        构造函数接受一个必需的参数，一个Axes实例，后面跟着下面描述的args和kwargs，
        这些参数在下面的pyplot接口文档中有描述:
        %(barbs_doc)s
        """
        
        self.sizes = sizes or dict()  # 初始化风羽的尺寸字典
        self.fill_empty = fill_empty  # 是否填充空白
        self.barb_increments = barb_increments or dict()  # 风羽增量字典
        self.rounding = rounding  # 是否进行四舍五入
        self.flip = np.atleast_1d(flip_barb)  # 确保flip_barb至少是一维的numpy数组
        transform = kwargs.pop('transform', ax.transData)  # 获取或设置transform参数，默认为ax.transData
        self._pivot = pivot  # 风羽的旋转点
        self._length = length  # 风羽的长度

        # 方便地设置风羽多边形的填充色和边线色
        if None in (barbcolor, flagcolor):
            kwargs['edgecolors'] = 'face'
            if flagcolor:
                kwargs['facecolors'] = flagcolor
            elif barbcolor:
                kwargs['facecolors'] = barbcolor
            else:
                # 如果未指定，使用传入的facecolor或默认为黑色
                kwargs.setdefault('facecolors', 'k')
        else:
            kwargs['edgecolors'] = barbcolor
            kwargs['facecolors'] = flagcolor

        # 如果未指定linewidth或lw参数，则显式设置线宽，否则多边形将没有轮廓线，不会显示风羽
        if 'linewidth' not in kwargs and 'lw' not in kwargs:
            kwargs['linewidth'] = 1

        # 从不同的配置中解析出数据数组
        x, y, u, v, c = _parse_args(*args, caller_name='barbs')
        self.x = x  # x坐标数组
        self.y = y  # y坐标数组
        xy = np.column_stack((x, y))  # 将x和y坐标堆叠成二维数组

        # 创建一个集合（Collection）
        barb_size = self._length ** 2 / 4  # 经验确定的风羽大小
        super().__init__(
            [], (barb_size,), offsets=xy, offset_transform=transform, **kwargs)
        self.set_transform(transforms.IdentityTransform())  # 设置变换为IdentityTransform

        # 设置风速和风向数据
        self.set_UVC(u, v, c)
    def _find_tails(self, mag, rounding=True, half=5, full=10, flag=50):
        """
        Find how many of each of the tail pieces is necessary.

        Parameters
        ----------
        mag : `~numpy.ndarray`
            Vector magnitudes; must be non-negative (and an actual ndarray).
        rounding : bool, default: True
            Whether to round or to truncate to the nearest half-barb.
        half, full, flag : float, defaults: 5, 10, 50
            Increments for a half-barb, a barb, and a flag.

        Returns
        -------
        n_flags, n_barbs : int array
            For each entry in *mag*, the number of flags and barbs.
        half_flag : bool array
            For each entry in *mag*, whether a half-barb is needed.
        empty_flag : bool array
            For each entry in *mag*, whether nothing is drawn.
        """
        # 如果 rounding 为 True，则对 mag 进行最接近的半条刻度的四舍五入
        if rounding:
            mag = half * np.around(mag / half)
        # 计算 mag 中每个元素被 flag 整除的商和余数
        n_flags, mag = divmod(mag, flag)
        # 计算余数被 full 整除的商和余数
        n_barb, mag = divmod(mag, full)
        # 判断余数是否大于等于 half，得出是否需要半条刻度的布尔数组
        half_flag = mag >= half
        # 判断是否既不需要半条刻度，也没有绘制任何东西，得出空标志的布尔数组
        empty_flag = ~(half_flag | (n_flags > 0) | (n_barb > 0))
        # 返回结果，将计算得到的整数数组和布尔数组作为结果返回
        return n_flags.astype(int), n_barb.astype(int), half_flag, empty_flag
    def set_UVC(self, U, V, C=None):
        # 确保我们拥有的是副本而不是可能在 draw() 调用前会改变的数组的引用。
        self.u = ma.masked_invalid(U, copy=True).ravel()
        self.v = ma.masked_invalid(V, copy=True).ravel()

        # 如果 flip 只有一个元素，使用 broadcast_to 避免创建一个充满相同值的庞大数组。
        # （不能依赖于实际的广播）
        if len(self.flip) == 1:
            flip = np.broadcast_to(self.flip, self.u.shape)
        else:
            flip = self.flip

        if C is not None:
            # 将 C 数组进行遮罩处理，确保不包含无效数据，然后展平数组
            c = ma.masked_invalid(C, copy=True).ravel()
            # 删除遮罩点并返回有效的点的坐标和数据
            x, y, u, v, c, flip = cbook.delete_masked_points(
                self.x.ravel(), self.y.ravel(), self.u, self.v, c,
                flip.ravel())
            # 检查所有数组的形状是否一致
            _check_consistent_shapes(x, y, u, v, c, flip)
        else:
            # 删除遮罩点并返回有效的点的坐标和数据（不包含颜色数据）
            x, y, u, v, flip = cbook.delete_masked_points(
                self.x.ravel(), self.y.ravel(), self.u, self.v, flip.ravel())
            # 检查所有数组的形状是否一致
            _check_consistent_shapes(x, y, u, v, flip)

        # 计算向量的大小
        magnitude = np.hypot(u, v)
        # 根据向量大小计算标志、barb、halves 和 empty
        flags, barbs, halves, empty = self._find_tails(
            magnitude, self.rounding, **self.barb_increments)

        # 根据计算的标志和 barb 数据创建 barbs 的顶点数组
        plot_barbs = self._make_barbs(u, v, flags, barbs, halves, empty,
                                      self._length, self._pivot, self.sizes,
                                      self.fill_empty, flip)
        # 设置顶点数据
        self.set_verts(plot_barbs)

        # 如果存在颜色数据 C，则设置颜色数组
        if C is not None:
            self.set_array(c)

        # 更新偏移量以反映遮罩数据的变化
        xy = np.column_stack((x, y))
        self._offsets = xy
        self.stale = True

    def set_offsets(self, xy):
        """
        Set the offsets for the barb polygons.  This saves the offsets passed
        in and masks them as appropriate for the existing U/V data.

        Parameters
        ----------
        xy : sequence of pairs of floats
        """
        # 设置 barb 多边形的偏移量。保存传入的偏移量，并根据现有的 U/V 数据进行遮罩处理。
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        # 删除遮罩点并返回有效的点的坐标和数据
        x, y, u, v = cbook.delete_masked_points(
            self.x.ravel(), self.y.ravel(), self.u, self.v)
        # 检查所有数组的形状是否一致
        _check_consistent_shapes(x, y, u, v)
        xy = np.column_stack((x, y))
        # 调用父类的方法设置偏移量
        super().set_offsets(xy)
        # 标记状态为需要更新
        self.stale = True
```