# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\inset_locator.py`

```
"""
A collection of functions and objects for creating or placing inset axes.
"""

from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox

from . import axes_size as Size  # 导入本地模块 axes_size，并命名为 Size
from .parasite_axes import HostAxes  # 导入本地模块 parasite_axes 中的 HostAxes 类


@_api.deprecated("3.8", alternative="Axes.inset_axes")
class InsetPosition:
    @_docstring.dedent_interpd
    def __init__(self, parent, lbwh):
        """
        An object for positioning an inset axes.

        This is created by specifying the normalized coordinates in the axes,
        instead of the figure.

        Parameters
        ----------
        parent : `~matplotlib.axes.Axes`
            Axes to use for normalizing coordinates.

        lbwh : iterable of four floats
            The left edge, bottom edge, width, and height of the inset axes, in
            units of the normalized coordinate of the *parent* axes.

        See Also
        --------
        :meth:`matplotlib.axes.Axes.set_axes_locator`

        Examples
        --------
        The following bounds the inset axes to a box with 20%% of the parent
        axes height and 40%% of the width. The size of the axes specified
        ([0, 0, 1, 1]) ensures that the axes completely fills the bounding box:

        >>> parent_axes = plt.gca()
        >>> ax_ins = plt.axes([0, 0, 1, 1])
        >>> ip = InsetPosition(parent_axes, [0.5, 0.1, 0.4, 0.2])
        >>> ax_ins.set_axes_locator(ip)
        """
        self.parent = parent  # 设置父坐标系对象
        self.lbwh = lbwh  # 设置相对于父坐标系的位置和大小的四个浮点数值

    def __call__(self, ax, renderer):
        bbox_parent = self.parent.get_position(original=False)  # 获取父坐标系的位置信息
        trans = BboxTransformTo(bbox_parent)  # 创建一个转换，将当前坐标系转换为父坐标系
        bbox_inset = Bbox.from_bounds(*self.lbwh)  # 创建一个边界框对象，根据给定的 lbwh 参数
        bb = TransformedBbox(bbox_inset, trans)  # 使用转换对象将边界框转换为新的边界框
        return bb


class AnchoredLocatorBase(AnchoredOffsetbox):
    def __init__(self, bbox_to_anchor, offsetbox, loc,
                 borderpad=0.5, bbox_transform=None):
        super().__init__(
            loc, pad=0., child=None, borderpad=borderpad,
            bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform
        )

    def draw(self, renderer):
        raise RuntimeError("No draw method should be called")  # 抛出运行时错误，不应调用 draw 方法

    def __call__(self, ax, renderer):
        if renderer is None:
            renderer = ax.figure._get_renderer()
        self.axes = ax
        bbox = self.get_window_extent(renderer)  # 获取偏移框的窗口范围
        px, py = self.get_offset(bbox.width, bbox.height, 0, 0, renderer)  # 获取偏移量
        bbox_canvas = Bbox.from_bounds(px, py, bbox.width, bbox.height)  # 创建画布边界框对象
        tr = ax.figure.transSubfigure.inverted()  # 获取子图转换的反向转换
        return TransformedBbox(bbox_canvas, tr)  # 返回转换后的边界框对象


class AnchoredSizeLocator(AnchoredLocatorBase):
    # 初始化方法，接收参数 bbox_to_anchor, x_size, y_size, loc, borderpad, bbox_transform
    def __init__(self, bbox_to_anchor, x_size, y_size, loc,
                 borderpad=0.5, bbox_transform=None):
        # 调用父类的初始化方法，传入 bbox_to_anchor, None, loc, borderpad, bbox_transform 参数
        super().__init__(
            bbox_to_anchor, None, loc,
            borderpad=borderpad, bbox_transform=bbox_transform
        )

        # 使用 Size.from_any 方法处理 x_size 和 y_size，并保存到当前对象的属性中
        self.x_size = Size.from_any(x_size)
        self.y_size = Size.from_any(y_size)

    # 根据渲染器 renderer 获取并返回当前对象的边界框
    def get_bbox(self, renderer):
        # 获取当前对象的基准边界框
        bbox = self.get_bbox_to_anchor()
        # 获取渲染器的 DPI
        dpi = renderer.points_to_pixels(72.)

        # 计算水平尺寸的比例因子 r 和附加值 a
        r, a = self.x_size.get_size(renderer)
        # 计算最终宽度，考虑基准边界框的宽度和附加值 a 乘以 DPI
        width = bbox.width * r + a * dpi

        # 计算垂直尺寸的比例因子 r 和附加值 a
        r, a = self.y_size.get_size(renderer)
        # 计算最终高度，考虑基准边界框的高度和附加值 a 乘以 DPI
        height = bbox.height * r + a * dpi

        # 获取当前属性对象的字体大小并转换为像素
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        # 计算填充值，为当前对象的 pad 属性乘以字体大小
        pad = self.pad * fontsize

        # 根据给定的边界框坐标创建 Bbox 对象，并添加填充
        return Bbox.from_bounds(0, 0, width, height).padded(pad)
class AnchoredZoomLocator(AnchoredLocatorBase):
    # 定义 AnchoredZoomLocator 类，继承自 AnchoredLocatorBase 类
    def __init__(self, parent_axes, zoom, loc,
                 borderpad=0.5,
                 bbox_to_anchor=None,
                 bbox_transform=None):
        # 初始化函数，接受父轴对象 parent_axes、缩放比例 zoom、位置 loc
        self.parent_axes = parent_axes
        self.zoom = zoom
        if bbox_to_anchor is None:
            bbox_to_anchor = parent_axes.bbox
        # 调用父类构造函数初始化
        super().__init__(
            bbox_to_anchor, None, loc, borderpad=borderpad,
            bbox_transform=bbox_transform)

    def get_bbox(self, renderer):
        # 获取包围框的方法，接受渲染器对象 renderer
        bb = self.parent_axes.transData.transform_bbox(self.axes.viewLim)
        # 将父轴数据坐标变换后的包围框，传递给 bb 变量
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        # 获取文本属性的字号大小，并将其转换为像素
        pad = self.pad * fontsize
        # 计算填充值 pad，为边界填充大小乘以字号像素大小
        return (
            Bbox.from_bounds(
                0, 0, abs(bb.width * self.zoom), abs(bb.height * self.zoom))
            .padded(pad))


class BboxPatch(Patch):
    # 定义 BboxPatch 类，继承自 Patch 类
    @_docstring.dedent_interpd
    def __init__(self, bbox, **kwargs):
        """
        Patch showing the shape bounded by a Bbox.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.Bbox`
            Bbox to use for the extents of this patch.

        **kwargs
            Patch properties. Valid arguments include:

            %(Patch:kwdoc)s
        """
        # 构造函数，接受 bbox 参数作为 BboxPatch 的边界框
        if "transform" in kwargs:
            raise ValueError("transform should not be set")

        kwargs["transform"] = IdentityTransform()
        # 设置默认的 transform 为 IdentityTransform
        super().__init__(**kwargs)
        self.bbox = bbox
        # 初始化 Patch，并存储边界框参数到 self.bbox 中

    def get_path(self):
        # 获取路径的方法，继承自 Patch 类
        # 继承 Patch 类的文档注释
        x0, y0, x1, y1 = self.bbox.extents
        # 获取边界框的四个角的坐标
        return Path._create_closed([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


class BboxConnector(Patch):
    # 定义 BboxConnector 类，继承自 Patch 类
    @staticmethod
    def get_bbox_edge_pos(bbox, loc):
        """
        Return the ``(x, y)`` coordinates of corner *loc* of *bbox*; parameters
        behave as documented for the `.BboxConnector` constructor.
        """
        # 静态方法，返回边界框 bbox 的 loc 角的坐标 (x, y)
        x0, y0, x1, y1 = bbox.extents
        if loc == 1:
            return x1, y1
        elif loc == 2:
            return x0, y1
        elif loc == 3:
            return x0, y0
        elif loc == 4:
            return x1, y0

    @staticmethod
    def connect_bbox(bbox1, bbox2, loc1, loc2=None):
        """
        Construct a `.Path` connecting corner *loc1* of *bbox1* to corner
        *loc2* of *bbox2*, where parameters behave as documented as for the
        `.BboxConnector` constructor.
        """
        # 静态方法，连接两个边界框 bbox1 和 bbox2 的 loc1 和 loc2 角，生成连接路径
        if isinstance(bbox1, Rectangle):
            bbox1 = TransformedBbox(Bbox.unit(), bbox1.get_transform())
        if isinstance(bbox2, Rectangle):
            bbox2 = TransformedBbox(Bbox.unit(), bbox2.get_transform())
        if loc2 is None:
            loc2 = loc1
        # 如果 loc2 未指定，默认与 loc1 相同
        x1, y1 = BboxConnector.get_bbox_edge_pos(bbox1, loc1)
        x2, y2 = BboxConnector.get_bbox_edge_pos(bbox2, loc2)
        # 获取边界框角落的坐标 (x1, y1) 和 (x2, y2)
        return Path([[x1, y1], [x2, y2]])

    @_docstring.dedent_interpd
    def __init__(self, bbox1, bbox2, loc1, loc2=None, **kwargs):
        """
        Connect two bboxes with a straight line.

        Parameters
        ----------
        bbox1, bbox2 : `~matplotlib.transforms.Bbox`
            Bounding boxes to connect.

        loc1, loc2 : {1, 2, 3, 4}
            Corner of *bbox1* and *bbox2* to draw the line. Valid values are::

                'upper right'  : 1,
                'upper left'   : 2,
                'lower left'   : 3,
                'lower right'  : 4

            *loc2* is optional and defaults to *loc1*.

        **kwargs
            Patch properties for the line drawn. Valid arguments include:

            %(Patch:kwdoc)s
        """
        # 检查是否在 kwargs 中设置了 "transform"，如果设置了则抛出 ValueError
        if "transform" in kwargs:
            raise ValueError("transform should not be set")

        # 强制将 "transform" 设置为 IdentityTransform()
        kwargs["transform"] = IdentityTransform()
        
        # 设置默认的填充属性，根据 kwargs 中的关键字判断是否填充
        kwargs.setdefault(
            "fill", bool({'fc', 'facecolor', 'color'}.intersection(kwargs)))
        
        # 调用父类的构造函数，传入所有的 kwargs
        super().__init__(**kwargs)
        
        # 初始化对象的属性：bbox1, bbox2, loc1, loc2
        self.bbox1 = bbox1
        self.bbox2 = bbox2
        self.loc1 = loc1
        self.loc2 = loc2

    def get_path(self):
        # docstring inherited
        # 返回连接两个 bbox 的路径
        return self.connect_bbox(self.bbox1, self.bbox2,
                                 self.loc1, self.loc2)
class BboxConnectorPatch(BboxConnector):
    @_docstring.dedent_interpd
    def __init__(self, bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, **kwargs):
        """
        Connect two bboxes with a quadrilateral.

        The quadrilateral is specified by two lines that start and end at
        corners of the bboxes. The four sides of the quadrilateral are defined
        by the two lines given, the line between the two corners specified in
        *bbox1* and the line between the two corners specified in *bbox2*.

        Parameters
        ----------
        bbox1, bbox2 : `~matplotlib.transforms.Bbox`
            Bounding boxes to connect.

        loc1a, loc2a, loc1b, loc2b : {1, 2, 3, 4}
            The first line connects corners *loc1a* of *bbox1* and *loc2a* of
            *bbox2*; the second line connects corners *loc1b* of *bbox1* and
            *loc2b* of *bbox2*.  Valid values are::

                'upper right'  : 1,
                'upper left'   : 2,
                'lower left'   : 3,
                'lower right'  : 4

        **kwargs
            Patch properties for the line drawn:

            %(Patch:kwdoc)s
        """
        # 检查是否设置了不应设置的转换参数
        if "transform" in kwargs:
            raise ValueError("transform should not be set")
        # 调用父类的构造函数初始化
        super().__init__(bbox1, bbox2, loc1a, loc2a, **kwargs)
        # 初始化第二条连接线的位置
        self.loc1b = loc1b
        self.loc2b = loc2b

    def get_path(self):
        # docstring inherited
        # 获取第一条连接线的路径
        path1 = self.connect_bbox(self.bbox1, self.bbox2, self.loc1, self.loc2)
        # 获取第二条连接线的路径，注意交换起点和终点的顺序
        path2 = self.connect_bbox(self.bbox2, self.bbox1,
                                  self.loc2b, self.loc1b)
        # 合并两条路径的顶点，并闭合路径
        path_merged = [*path1.vertices, *path2.vertices, path1.vertices[0]]
        return Path(path_merged)


def _add_inset_axes(parent_axes, axes_class, axes_kwargs, axes_locator):
    """Helper function to add an inset axes and disable navigation in it."""
    # 如果未指定axes_class，则默认为HostAxes
    if axes_class is None:
        axes_class = HostAxes
    # 如果未指定axes_kwargs，则设为空字典
    if axes_kwargs is None:
        axes_kwargs = {}
    # 创建inset_axes对象，禁止其导航，根据传入的参数进行设置
    inset_axes = axes_class(
        parent_axes.figure, parent_axes.get_position(),
        **{"navigate": False, **axes_kwargs, "axes_locator": axes_locator})
    # 将创建的inset_axes对象添加到parent_axes的图形中并返回
    return parent_axes.figure.add_axes(inset_axes)


@_docstring.dedent_interpd
def inset_axes(parent_axes, width, height, loc='upper right',
               bbox_to_anchor=None, bbox_transform=None,
               axes_class=None, axes_kwargs=None,
               borderpad=0.5):
    """
    Create an inset axes with a given width and height.

    Both sizes used can be specified either in inches or percentage.
    For example,::

        inset_axes(parent_axes, width='40%%', height='30%%', loc='lower left')

    creates in inset axes in the lower left corner of *parent_axes* which spans
    over 30%% in height and 40%% in width of the *parent_axes*. Since the usage
    of `.inset_axes` may become slightly tricky when exceeding such standard
    cases, it is recommended to read :doc:`the examples
    """
    # 创建指定大小和位置的inset_axes对象
    # 注意：具体实现逻辑可能与这里的注释略有不同，但不影响理解
    pass
    Parameters
    ----------
    parent_axes : `matplotlib.axes.Axes`
        要放置插图轴的主轴对象。

    width, height : float or str
        创建插图轴的尺寸。如果是浮点数，则单位是英寸，例如 *width=1.3*。如果是字符串，则是相对单位，例如 *width='40%%'*。如果未指定 *bbox_to_anchor* 或 *bbox_transform*，尺寸相对于主轴。否则，相对于通过 *bbox_to_anchor* 提供的边界框。

    loc : str, default: 'upper right'
        插图轴的位置。有效位置包括 'upper left', 'upper center', 'upper right',
        'center left', 'center', 'center right',
        'lower left', 'lower center', 'lower right'。也可以使用数字值（用于向后兼容性），详细信息请参见 `.Legend` 的 *loc* 参数。

    bbox_to_anchor : tuple or `~matplotlib.transforms.BboxBase`, optional
        插图轴将锚定到的边界框。如果为 None，并且 *bbox_transform* 设置为 *parent_axes.transAxes* 或 *parent_axes.figure.transFigure*，则使用元组 (0, 0, 1, 1)。否则使用 *parent_axes.bbox*。如果是元组，则可以是 [left, bottom, width, height] 或 [left, bottom]。如果使用相对单位指定了 *width* 和/或 *height*，则不能使用二元组 [left, bottom]。注意，除非设置了 *bbox_transform*，否则边界框的单位将解释为像素坐标。使用元组的 *bbox_to_anchor* 时，几乎总是有意义的同时指定 *bbox_transform*。这通常是轴变换 *parent_axes.transAxes*。
    bbox_transform : `~matplotlib.transforms.Transform`, optional
        # bbox_transform参数：用于包含插图坐标轴的边界框的变换。
        # 如果为None，则使用`.transforms.IdentityTransform`。
        # *bbox_to_anchor*的值（或其get_points方法的返回值）将通过*bbox_transform*进行变换，
        # 然后解释为像素坐标中的点（这取决于dpi）。
        # 您可以在某些标准化坐标中提供*bbox_to_anchor*，并提供适当的变换（例如*parent_axes.transAxes*）。

    axes_class : `~matplotlib.axes.Axes` type, default: `.HostAxes`
        # axes_class参数：新创建的插图坐标轴的类型，默认为`.HostAxes`。
        
    axes_kwargs : dict, optional
        # axes_kwargs参数：传递给插图坐标轴构造函数的关键字参数。
        # 有效的参数包括：
        # %(Axes:kwdoc)s

    borderpad : float, default: 0.5
        # borderpad参数：插图坐标轴与bbox_to_anchor之间的填充。
        # 单位是轴字体大小，即对于默认字体大小为10点，
        # *borderpad = 0.5* 相当于填充5点。

    Returns
    -------
    inset_axes : *axes_class*
        # 返回插图坐标轴对象。
        插图坐标轴对象已创建。
    """

    if (bbox_transform in [parent_axes.transAxes, parent_axes.figure.transFigure]
            and bbox_to_anchor is None):
        # 检查bbox_transform是否为parent_axes.transAxes或parent_axes.figure.transFigure，
        # 并且bbox_to_anchor是否为None。
        _api.warn_external("Using the axes or figure transform requires a "
                           "bounding box in the respective coordinates. "
                           "Using bbox_to_anchor=(0, 0, 1, 1) now.")
        # 发出外部警告，提示在使用轴或图形变换时需要相应坐标系中的边界框，
        # 并设置bbox_to_anchor=(0, 0, 1, 1)作为默认值。
        bbox_to_anchor = (0, 0, 1, 1)
    if bbox_to_anchor is None:
        # 如果bbox_to_anchor仍然为None，则将其设置为parent_axes的边界框。
        bbox_to_anchor = parent_axes.bbox
    if (isinstance(bbox_to_anchor, tuple) and
            (isinstance(width, str) or isinstance(height, str))):
        # 如果bbox_to_anchor是tuple类型，并且width或height是str类型之一，
        # 则确保bbox_to_anchor是4元组或Bbox实例。
        if len(bbox_to_anchor) != 4:
            # 如果bbox_to_anchor的长度不等于4，则引发ValueError。
            raise ValueError("Using relative units for width or height "
                             "requires to provide a 4-tuple or a "
                             "`Bbox` instance to `bbox_to_anchor.")
    return _add_inset_axes(
        parent_axes, axes_class, axes_kwargs,
        AnchoredSizeLocator(
            bbox_to_anchor, width, height, loc=loc,
            bbox_transform=bbox_transform, borderpad=borderpad))
# 导入 _docstring.dedent_interpd 函数，用于处理文档字符串的缩进
@_docstring.dedent_interpd
# 定义 zoomed_inset_axes 函数，创建一个缩放的插图坐标系，基于一个父级坐标系进行缩放
def zoomed_inset_axes(parent_axes, zoom, loc='upper right',
                      bbox_to_anchor=None, bbox_transform=None,
                      axes_class=None, axes_kwargs=None,
                      borderpad=0.5):
    """
    Create an anchored inset axes by scaling a parent axes. For usage, also see
    :doc:`the examples </gallery/axes_grid1/inset_locator_demo2>`.

    Parameters
    ----------
    parent_axes : `~matplotlib.axes.Axes`
        Axes to place the inset axes.

    zoom : float
        Scaling factor of the data axes. *zoom* > 1 will enlarge the
        coordinates (i.e., "zoomed in"), while *zoom* < 1 will shrink the
        coordinates (i.e., "zoomed out").

    loc : str, default: 'upper right'
        Location to place the inset axes.  Valid locations are
        'upper left', 'upper center', 'upper right',
        'center left', 'center', 'center right',
        'lower left', 'lower center', 'lower right'.
        For backward compatibility, numeric values are accepted as well.
        See the parameter *loc* of `.Legend` for details.

    bbox_to_anchor : tuple or `~matplotlib.transforms.BboxBase`, optional
        Bbox that the inset axes will be anchored to. If None,
        *parent_axes.bbox* is used. If a tuple, can be either
        [left, bottom, width, height], or [left, bottom].
        If the kwargs *width* and/or *height* are specified in relative units,
        the 2-tuple [left, bottom] cannot be used. Note that
        the units of the bounding box are determined through the transform
        in use. When using *bbox_to_anchor* it almost always makes sense to
        also specify a *bbox_transform*. This might often be the axes transform
        *parent_axes.transAxes*.

    bbox_transform : `~matplotlib.transforms.Transform`, optional
        Transformation for the bbox that contains the inset axes.
        If None, a `.transforms.IdentityTransform` is used (i.e. pixel
        coordinates). This is useful when not providing any argument to
        *bbox_to_anchor*. When using *bbox_to_anchor* it almost always makes
        sense to also specify a *bbox_transform*. This might often be the
        axes transform *parent_axes.transAxes*. Inversely, when specifying
        the axes- or figure-transform here, be aware that not specifying
        *bbox_to_anchor* will use *parent_axes.bbox*, the units of which are
        in display (pixel) coordinates.

    axes_class : `~matplotlib.axes.Axes` type, default: `.HostAxes`
        The type of the newly created inset axes.

    axes_kwargs : dict, optional
        Keyword arguments to pass to the constructor of the inset axes.
        Valid arguments include:

        %(Axes:kwdoc)s

    borderpad : float, default: 0.5
        Padding between inset axes and the bbox_to_anchor.
        The units are axes font size, i.e. for a default font size of 10 points
        *borderpad = 0.5* is equivalent to a padding of 5 points.

    Returns
    -------
    ```
    # 返回新创建的插图坐标系对象，可以用来进一步绘图或操作
    ```
    -------
    inset_axes : *axes_class*
        Inset axes object created.
    """
    
    # 返回一个创建的插入轴对象，并指定了其类型为 axes_class
    return _add_inset_axes(
        parent_axes, axes_class, axes_kwargs,
        AnchoredZoomLocator(
            parent_axes, zoom=zoom, loc=loc,
            bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform,
            borderpad=borderpad))
class _TransformedBboxWithCallback(TransformedBbox):
    """
    Variant of `.TransformBbox` which calls *callback* before returning points.

    Used by `.mark_inset` to unstale the parent axes' viewlim as needed.
    """

    def __init__(self, *args, callback, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback  # 设置回调函数，用于在返回点之前调用

    def get_points(self):
        self._callback()  # 调用设置的回调函数
        return super().get_points()  # 返回通过父类获取的点


@_docstring.dedent_interpd
def mark_inset(parent_axes, inset_axes, loc1, loc2, **kwargs):
    """
    Draw a box to mark the location of an area represented by an inset axes.

    This function draws a box in *parent_axes* at the bounding box of
    *inset_axes*, and shows a connection with the inset axes by drawing lines
    at the corners, giving a "zoomed in" effect.

    Parameters
    ----------
    parent_axes : `~matplotlib.axes.Axes`
        Axes which contains the area of the inset axes.

    inset_axes : `~matplotlib.axes.Axes`
        The inset axes.

    loc1, loc2 : {1, 2, 3, 4}
        Corners to use for connecting the inset axes and the area in the
        parent axes.

    **kwargs
        Patch properties for the lines and box drawn:

        %(Patch:kwdoc)s

    Returns
    -------
    pp : `~matplotlib.patches.Patch`
        The patch drawn to represent the area of the inset axes.

    p1, p2 : `~matplotlib.patches.Patch`
        The patches connecting two corners of the inset axes and its area.
    """
    rect = _TransformedBboxWithCallback(
        inset_axes.viewLim, parent_axes.transData,
        callback=parent_axes._unstale_viewLim)  # 创建一个带有回调函数的转换后的边界框

    kwargs.setdefault("fill", bool({'fc', 'facecolor', 'color'}.intersection(kwargs)))
    pp = BboxPatch(rect, **kwargs)  # 在父轴上绘制表示插入轴区域的框

    parent_axes.add_patch(pp)  # 将框添加到父轴中

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1, **kwargs)  # 在插入轴和其区域之间的两个角落之间绘制连接线
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)

    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2, **kwargs)  # 在插入轴和其区域之间的另外两个角落之间绘制连接线
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2  # 返回绘制的补丁对象
```