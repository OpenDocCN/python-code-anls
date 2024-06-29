# `D:\src\scipysrc\matplotlib\lib\matplotlib\offsetbox.py`

```
# 包含 `.Artist` 的容器类。

# 所有容器艺术家的基类。
class OffsetBox:
    pass

# `AnchoredOffsetbox` 和 `AnchoredText`。
# 相对于父轴或特定锚点锚定和对齐任意 `.Artist` 或文本。
class AnchoredOffsetbox:
    pass

class AnchoredText:
    pass

# `DrawingArea`。
# 固定宽度和高度的容器。子元素在容器内有固定位置并可能被裁剪。
class DrawingArea:
    pass

# `HPacker` 和 `VPacker`。
# 用于垂直或水平布局其子元素的容器。
class HPacker:
    pass

class VPacker:
    pass

# `PaddedBox`。
# 在 `.Artist` 周围添加填充的容器。
class PaddedBox:
    pass

# `TextArea`。
# 包含单个 `.Text` 实例的容器。
class TextArea:
    pass
    """
    - 'expand': 如果设置了 *total*，则将总长度等分成N个区间，每个盒子在其子空间内左对齐。
      否则（*total* 为 *None*），必须提供 *sep*，每个盒子在其宽度为 ``(max(widths) + sep)`` 的子空间内左对齐。
      然后计算总宽度为 ``N * (max(widths) + sep)``。

    - 'equal': 如果设置了 *total*，则将总空间等分为N个相等的区间，每个盒子在其子空间内左对齐。
      否则（*total* 为 *None*），必须提供 *sep*，每个盒子在其宽度为 ``(max(widths) + sep)`` 的子空间内左对齐。
      然后计算总宽度为 ``N * (max(widths) + sep)``。

    Parameters
    ----------
    widths : list of float
        要排列的盒子的宽度列表。
    total : float or None
        预期的总长度。如果不使用，则为 *None*。
    sep : float or None
        盒子之间的间距。
    mode : {'fixed', 'expand', 'equal'}
        排列模式。

    Returns
    -------
    total : float
        用于容纳排列好的盒子所需的总宽度。
    offsets : array of float
        盒子的左侧偏移量。
    """
    _api.check_in_list(["fixed", "expand", "equal"], mode=mode)

    if mode == "fixed":
        # 计算固定模式下的偏移量
        offsets_ = np.cumsum([0] + [w + sep for w in widths])
        offsets = offsets_[:-1]
        if total is None:
            total = offsets_[-1] - sep
        return total, offsets

    elif mode == "expand":
        # 避免在 *total* 为 *None* 且与紧凑布局共同使用时引发 TypeError 的小技巧
        if total is None:
            total = 1
        # 计算扩展模式下的偏移量
        if len(widths) > 1:
            sep = (total - sum(widths)) / (len(widths) - 1)
        else:
            sep = 0
        offsets_ = np.cumsum([0] + [w + sep for w in widths])
        offsets = offsets_[:-1]
        return total, offsets

    elif mode == "equal":
        maxh = max(widths)
        if total is None:
            if sep is None:
                raise ValueError("total 和 sep 在使用布局模式 'equal' 时不能同时为 None")
            total = (maxh + sep) * len(widths)
        else:
            sep = total / len(widths) - maxh
        # 计算等分模式下的偏移量
        offsets = (maxh + sep) * np.arange(len(widths))
        return total, offsets
# 定义一个函数 `_get_aligned_offsets`，用于计算对齐后的偏移量和相关信息
def _get_aligned_offsets(yspans, height, align="baseline"):
    """
    Align boxes each specified by their ``(y0, y1)`` spans.

    For simplicity of the description, the terminology used here assumes a
    horizontal layout (i.e., vertical alignment), but the function works
    equally for a vertical layout.

    Parameters
    ----------
    yspans
        List of (y0, y1) spans of boxes to be aligned.
    height : float or None
        Intended total height. If None, the maximum of the heights
        (``y1 - y0``) in *yspans* is used.
    align : {'baseline', 'left', 'top', 'right', 'bottom', 'center'}
        The alignment anchor of the boxes.

    Returns
    -------
    (y0, y1)
        y range spanned by the packing.  If a *height* was originally passed
        in, then for all alignments other than "baseline", a span of ``(0,
        height)`` is used without checking that it is actually large enough).
    descent
        The descent of the packing.
    offsets
        The bottom offsets of the boxes.
    """
    
    # 检查对齐方式是否在允许的列表中
    _api.check_in_list(
        ["baseline", "left", "top", "right", "bottom", "center"], align=align)
    
    # 如果未指定总高度，则使用 yspans 中 (y1 - y0) 的最大值作为高度
    if height is None:
        height = max(y1 - y0 for y0, y1 in yspans)

    # 根据不同的对齐方式计算偏移量和 y 范围
    if align == "baseline":
        # 对齐到基线，计算出包含所有 boxes 的 y 范围
        yspan = (min(y0 for y0, y1 in yspans), max(y1 for y0, y1 in yspans))
        # 所有 boxes 的偏移量初始化为 0
        offsets = [0] * len(yspans)
    elif align in ["left", "bottom"]:
        # 左对齐或底部对齐，y 范围为 (0, height)，偏移量为 -y0
        yspan = (0, height)
        offsets = [-y0 for y0, y1 in yspans]
    elif align in ["right", "top"]:
        # 右对齐或顶部对齐，y 范围为 (0, height)，偏移量为 height - y1
        yspan = (0, height)
        offsets = [height - y1 for y0, y1 in yspans]
    elif align == "center":
        # 居中对齐，y 范围为 (0, height)，偏移量计算为 (height - (y1 - y0)) * 0.5 - y0
        offsets = [(height - (y1 - y0)) * .5 - y0 for y0, y1 in yspans]

    # 返回计算得到的 y 范围、偏移量和其它相关信息
    return yspan, offsets


class OffsetBox(martist.Artist):
    """
    The OffsetBox is a simple container artist.

    The child artists are meant to be drawn at a relative position to its
    parent.

    Being an artist itself, all parameters are passed on to `.Artist`.
    """

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，并传递所有参数
        super().__init__(*args)
        # 根据 kwargs 更新内部状态
        self._internal_update(kwargs)
        # 禁用裁剪以保持一致性，因为 OffsetBox 系列尚未实现裁剪功能
        self.set_clip_on(False)
        # 初始化子元素列表为空
        self._children = []
        # 初始化偏移量为 (0, 0)
        self._offset = (0, 0)

    def set_figure(self, fig):
        """
        Set the `.Figure` for the `.OffsetBox` and all its children.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
        """
        # 调用父类的 set_figure 方法设置图形对象
        super().set_figure(fig)
        # 设置所有子元素的图形对象为当前图形对象
        for c in self.get_children():
            c.set_figure(fig)

    @martist.Artist.axes.setter
    def axes(self, ax):
        # TODO deal with this better
        # 设置 axes 属性，并将其传递给所有子元素
        martist.Artist.axes.fset(self, ax)
        for c in self.get_children():
            if c is not None:
                c.axes = ax
    def contains(self, mouseevent):
        """
        Delegate the mouse event contains-check to the children.

        As a container, the `.OffsetBox` does not respond itself to
        mouseevents.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`
            The mouse event to check against.

        Returns
        -------
        contains : bool
            Whether any values are within the radius.
        details : dict
            An artist-specific dictionary of details of the event context,
            such as which points are contained in the pick radius. See the
            individual Artist subclasses for details.

        See Also
        --------
        .Artist.contains
            Parent method for checking if a point is within an artist.

        """
        # 检查鼠标事件是否在不同的画布上，如果是则返回 False 和空字典
        if self._different_canvas(mouseevent):
            return False, {}
        
        # 遍历所有子元素
        for c in self.get_children():
            # 对每个子元素调用其 contains 方法，获取结果 a, b
            a, b = c.contains(mouseevent)
            # 如果子元素包含鼠标事件，立即返回子元素的结果
            if a:
                return a, b
        
        # 如果没有子元素包含鼠标事件，则返回 False 和空字典
        return False, {}

    def set_offset(self, xy):
        """
        Set the offset.

        Parameters
        ----------
        xy : (float, float) or callable
            The (x, y) coordinates of the offset in display units. These can
            either be given explicitly as a tuple (x, y), or by providing a
            function that converts the extent into the offset. This function
            must have the signature::

                def offset(width, height, xdescent, ydescent, renderer) \

        """
        # 设置偏移量的方法，参数 xy 可以是坐标元组或者可调用对象
        pass
# 定义一个方法签名，表明该方法返回一个包含两个 float 类型值的元组
-> (float, float)
        """
        设置偏移量为参数 xy 所表示的值，并标记为过时（stale=True）
        
        Parameters
        ----------
        xy : tuple
            包含 x 和 y 偏移量的元组
        """
        self._offset = xy
        self.stale = True

    @_compat_get_offset
    def get_offset(self, bbox, renderer):
        """
        返回偏移量的元组 (x, y)。

        当偏移量由可调用对象动态确定时（参见 `~.OffsetBox.set_offset`），必须提供 extent 参数。

        Parameters
        ----------
        bbox : `.Bbox`
            包围框对象，描述绘制区域的边界框
        renderer : `.RendererBase` subclass
            渲染器对象，用于绘制操作的引擎
        """
        return (
            self._offset(bbox.width, bbox.height, -bbox.x0, -bbox.y0, renderer)
            if callable(self._offset)
            else self._offset)

    def set_width(self, width):
        """
        设置偏移框的宽度。

        Parameters
        ----------
        width : float
            偏移框的宽度值
        """
        self.width = width
        self.stale = True

    def set_height(self, height):
        """
        设置偏移框的高度。

        Parameters
        ----------
        height : float
            偏移框的高度值
        """
        self.height = height
        self.stale = True

    def get_visible_children(self):
        r"""返回可见子 `.Artist` 的列表。"""
        return [c for c in self._children if c.get_visible()]

    def get_children(self):
        r"""返回子 `.Artist` 的列表。"""
        return self._children

    def _get_bbox_and_child_offsets(self, renderer):
        """
        返回偏移框及其子元素偏移量的边界框。

        边界框应满足 ``x0 <= x1 and y0 <= y1``。

        Parameters
        ----------
        renderer : `.RendererBase` subclass
            渲染器对象，用于绘制操作的引擎

        Returns
        -------
        bbox : Bbox
            偏移框的边界框对象
        list of (xoffset, yoffset) pairs
            子元素的偏移量列表，每个元素是一个二元组 (xoffset, yoffset)
        """
        raise NotImplementedError(
            "get_bbox_and_offsets must be overridden in derived classes")

    def get_bbox(self, renderer):
        """返回偏移框的边界框，忽略父级偏移。"""
        bbox, offsets = self._get_bbox_and_child_offsets(renderer)
        return bbox

    def get_window_extent(self, renderer=None):
        # 继承的文档字符串
        if renderer is None:
            renderer = self.figure._get_renderer()
        bbox = self.get_bbox(renderer)
        try:  # Some subclasses redefine get_offset to take no args.
            px, py = self.get_offset(bbox, renderer)
        except TypeError:
            px, py = self.get_offset()
        return bbox.translated(px, py)

    def draw(self, renderer):
        """
        如果需要，更新子元素的位置并将它们绘制到给定的 *renderer* 中。
        """
        bbox, offsets = self._get_bbox_and_child_offsets(renderer)
        px, py = self.get_offset(bbox, renderer)
        for c, (ox, oy) in zip(self.get_visible_children(), offsets):
            c.set_offset((px + ox, py + oy))
            c.draw(renderer)
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.))
        self.stale = False
    # 初始化函数，用于创建一个容器对象，可以设置各种属性来控制布局和显示

    def __init__(self, pad=0., sep=0., width=None, height=None,
                 align="baseline", mode="fixed", children=None):
        """
        Parameters
        ----------
        pad : float, default: 0.0
            The boundary padding in points.
            边界填充的大小，单位为点（points）。

        sep : float, default: 0.0
            The spacing between items in points.
            各个项目之间的间距大小，单位为点（points）。

        width, height : float, optional
            Width and height of the container box in pixels, calculated if
            *None*.
            容器框的宽度和高度，单位为像素（pixels），如果为 *None* 则会自动计算。

        align : {'top', 'bottom', 'left', 'right', 'center', 'baseline'}, \
                default: 'baseline'
            Alignment of the content within the container.
            内容在容器中的对齐方式，可选值有：'top', 'bottom', 'left', 'right', 'center', 'baseline'。

        mode : str, default: 'fixed'
            Mode of the container layout.
            容器布局的模式，默认为固定布局 ('fixed')。

        children : list or None, optional
            List of child elements within the container.
            容器中的子元素列表，如果没有则为 None。
        """
        default: 'baseline'
            Alignment of boxes.

        mode : {'fixed', 'expand', 'equal'}, default: 'fixed'
            The packing mode.

            - 'fixed' packs the given `.Artist`\\s tight with *sep* spacing.
            - 'expand' uses the maximal available space to distribute the
              artists with equal spacing in between.
            - 'equal': Each artist an equal fraction of the available space
              and is left-aligned (or top-aligned) therein.

        children : list of `.Artist`
            The artists to pack.

        Notes
        -----
        *pad* and *sep* are in points and will be scaled with the renderer
        dpi, while *width* and *height* are in pixels.
        """
        # 继承自基类，初始化高度、宽度、间距、填充、模式和对齐方式，并设置子元素列表
        super().__init__()
        self.height = height
        self.width = width
        self.sep = sep  # 设置间距，单位为点，将根据渲染器 DPI 进行缩放
        self.pad = pad  # 设置填充，单位为点，将根据渲染器 DPI 进行缩放
        self.mode = mode  # 设置排列模式，默认为 'fixed'
        self.align = align  # 设置对齐方式，默认为 'baseline'
        self._children = children  # 设置子元素列表

class VPacker(PackerBase):
    """
    VPacker packs its children vertically, automatically adjusting their
    relative positions at draw time.
    """

    def _get_bbox_and_child_offsets(self, renderer):
        # docstring inherited
        dpicor = renderer.points_to_pixels(1.)
        pad = self.pad * dpicor  # 根据渲染器 DPI 缩放填充值
        sep = self.sep * dpicor  # 根据渲染器 DPI 缩放间距值

        if self.width is not None:
            for c in self.get_visible_children():
                if isinstance(c, PackerBase) and c.mode == "expand":
                    c.set_width(self.width)

        bboxes = [c.get_bbox(renderer) for c in self.get_visible_children()]  # 获取可见子元素的边界框
        (x0, x1), xoffsets = _get_aligned_offsets(
            [bbox.intervalx for bbox in bboxes], self.width, self.align)  # 获取对齐偏移量
        height, yoffsets = _get_packed_offsets(
            [bbox.height for bbox in bboxes], self.height, sep, self.mode)  # 获取纵向排列的偏移量和总高度

        yoffsets = height - (yoffsets + [bbox.y1 for bbox in bboxes])  # 计算纵向偏移量
        ydescent = yoffsets[0]  # 获取第一个元素的纵向偏移量
        yoffsets = yoffsets - ydescent  # 调整纵向偏移量

        return (
            Bbox.from_bounds(x0, -ydescent, x1 - x0, height).padded(pad),  # 返回调整后的边界框
            [*zip(xoffsets, yoffsets)])  # 返回子元素的偏移量列表

class HPacker(PackerBase):
    """
    HPacker packs its children horizontally, automatically adjusting their
    relative positions at draw time.
    """
    # 获取渲染器对应的像素尺寸，作为1个数据点的像素大小
    dpicor = renderer.points_to_pixels(1.)
    # 根据对象的填充比例计算填充量
    pad = self.pad * dpicor
    # 根据对象的间隔比例计算间隔量
    sep = self.sep * dpicor

    # 获取所有可见子对象的边界框列表
    bboxes = [c.get_bbox(renderer) for c in self.get_visible_children()]
    # 如果边界框列表为空，则返回一个填充后的空边界框和空的偏移量列表
    if not bboxes:
        return Bbox.from_bounds(0, 0, 0, 0).padded(pad), []

    # 获取对齐后的偏移量和高度范围
    (y0, y1), yoffsets = _get_aligned_offsets(
        [bbox.intervaly for bbox in bboxes], self.height, self.align)
    # 获取打包后的偏移量和宽度
    width, xoffsets = _get_packed_offsets(
        [bbox.width for bbox in bboxes], self.width, sep, self.mode)

    # 计算第一个边界框的左侧位置 x0
    x0 = bboxes[0].x0
    # 调整 xoffsets，使得所有偏移量相对于第一个边界框的左侧位置 x0 对齐
    xoffsets -= ([bbox.x0 for bbox in bboxes] - x0)

    # 返回根据计算得到的边界框和偏移量的元组列表
    return (Bbox.from_bounds(x0, y0, width, y1 - y0).padded(pad),
            [*zip(xoffsets, yoffsets)])
class PaddedBox(OffsetBox):
    """
    A container to add a padding around an `.Artist`.

    The `.PaddedBox` contains a `.FancyBboxPatch` that is used to visualize
    it when rendering.
    """

    def __init__(self, child, pad=0., *, draw_frame=False, patch_attrs=None):
        """
        Parameters
        ----------
        child : `~matplotlib.artist.Artist`
            The contained `.Artist`.
        pad : float, default: 0.0
            The padding in points. This will be scaled with the renderer dpi.
            In contrast, *width* and *height* are in *pixels* and thus not
            scaled.
        draw_frame : bool
            Whether to draw the contained `.FancyBboxPatch`.
        patch_attrs : dict or None
            Additional parameters passed to the contained `.FancyBboxPatch`.
        """
        super().__init__()
        self.pad = pad  # 设置填充的大小，以点为单位，会随渲染器 DPI 缩放
        self._children = [child]  # 存储子元素列表，初始只有一个子元素
        self.patch = FancyBboxPatch(
            xy=(0.0, 0.0), width=1., height=1.,
            facecolor='w', edgecolor='k',
            mutation_scale=1,  # 突变尺度，用于设置框的大小
            snap=True,
            visible=draw_frame,  # 是否绘制 FancyBboxPatch
            boxstyle="square,pad=0",  # 方框样式，这里使用方形并设置填充为 0
        )
        if patch_attrs is not None:
            self.patch.update(patch_attrs)  # 更新方框的属性

    def _get_bbox_and_child_offsets(self, renderer):
        # docstring inherited.
        pad = self.pad * renderer.points_to_pixels(1.)  # 计算填充大小，并转换为像素
        return (self._children[0].get_bbox(renderer).padded(pad), [(0, 0)])

    def draw(self, renderer):
        # docstring inherited
        bbox, offsets = self._get_bbox_and_child_offsets(renderer)
        px, py = self.get_offset(bbox, renderer)  # 获取偏移量
        for c, (ox, oy) in zip(self.get_visible_children(), offsets):
            c.set_offset((px + ox, py + oy))  # 设置每个子元素的偏移位置

        self.draw_frame(renderer)  # 绘制框架

        for c in self.get_visible_children():
            c.draw(renderer)  # 绘制每个可见子元素

        self.stale = False  # 标记为未过时

    def update_frame(self, bbox, fontsize=None):
        self.patch.set_bounds(bbox.bounds)  # 设置方框的边界
        if fontsize:
            self.patch.set_mutation_scale(fontsize)  # 设置字体大小
        self.stale = True  # 标记为过时

    def draw_frame(self, renderer):
        # update the location and size of the legend
        self.update_frame(self.get_window_extent(renderer))  # 更新框架的位置和大小
        self.patch.draw(renderer)  # 绘制方框


class DrawingArea(OffsetBox):
    """
    The DrawingArea can contain any Artist as a child. The DrawingArea
    has a fixed width and height. The position of children relative to
    the parent is fixed. The children can be clipped at the
    boundaries of the parent.
    """
    def __init__(self, width, height, xdescent=0., ydescent=0., clip=False):
        """
        Parameters
        ----------
        width, height : float
            Width and height of the container box.
        xdescent, ydescent : float
            Descent of the box in x- and y-direction.
        clip : bool
            Whether to clip the children to the box.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置容器框的宽度和高度
        self.width = width
        self.height = height
        # 设置容器框在x和y方向的下降量
        self.xdescent = xdescent
        self.ydescent = ydescent
        # 是否裁剪容器内的子元素
        self._clip_children = clip
        # 偏移变换对象，用于控制容器的偏移
        self.offset_transform = mtransforms.Affine2D()
        # DPI变换对象，用于处理容器的DPI相关变换
        self.dpi_transform = mtransforms.Affine2D()

    @property
    def clip_children(self):
        """
        If the children of this DrawingArea should be clipped
        by DrawingArea bounding box.
        """
        # 返回是否裁剪子元素的布尔值
        return self._clip_children

    @clip_children.setter
    def clip_children(self, val):
        # 设置是否裁剪子元素的布尔值，并标记为过时以便更新
        self._clip_children = bool(val)
        self.stale = True

    def get_transform(self):
        """
        Return the `~matplotlib.transforms.Transform` applied to the children.
        """
        # 返回应用于子元素的变换对象
        return self.dpi_transform + self.offset_transform

    def set_transform(self, t):
        """
        set_transform is ignored.
        """
        # 忽略设置变换的操作

    def set_offset(self, xy):
        """
        Set the offset of the container.

        Parameters
        ----------
        xy : (float, float)
            The (x, y) coordinates of the offset in display units.
        """
        # 设置容器的偏移量
        self._offset = xy
        # 清除当前的偏移变换并重新设置为给定的偏移
        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])
        # 标记为过时，需要更新
        self.stale = True

    def get_offset(self):
        """Return offset of the container."""
        # 返回容器的偏移量
        return self._offset

    def get_bbox(self, renderer):
        # docstring inherited
        # 返回容器的边界框，根据渲染器的DPI转换比例计算
        dpi_cor = renderer.points_to_pixels(1.)
        return Bbox.from_bounds(
            -self.xdescent * dpi_cor, -self.ydescent * dpi_cor,
            self.width * dpi_cor, self.height * dpi_cor)

    def add_artist(self, a):
        """Add an `.Artist` to the container box."""
        # 将一个艺术家对象添加到容器框中
        self._children.append(a)
        # 如果艺术家对象未设置变换，则设置为容器的变换
        if not a.is_transform_set():
            a.set_transform(self.get_transform())
        # 如果容器有轴对象，则将其赋给艺术家对象
        if self.axes is not None:
            a.axes = self.axes
        # 如果容器所属的图形对象不为空，则将其设置为艺术家对象的图形对象
        fig = self.figure
        if fig is not None:
            a.set_figure(fig)
    def draw(self, renderer):
        # 绘制方法，用于绘制图形到指定的渲染器上

        # 获取每个点对应的像素数，用于 DPI 转换
        dpi_cor = renderer.points_to_pixels(1.)
        # 清空 DPI 转换的当前状态
        self.dpi_transform.clear()
        # 缩放 DPI 转换
        self.dpi_transform.scale(dpi_cor)

        # 此时 DrawingArea 已经有一个到显示空间的变换，
        # 因此创建的路径适合用于裁剪子对象
        tpath = mtransforms.TransformedPath(
            mpath.Path([[0, 0], [0, self.height],
                        [self.width, self.height],
                        [self.width, 0]]),
            self.get_transform())
        
        # 遍历子对象列表，根据需要设置裁剪路径，并绘制每个子对象
        for c in self._children:
            if self._clip_children and not (c.clipbox or c._clippath):
                c.set_clip_path(tpath)
            c.draw(renderer)

        # 使用 _bbox_artist 绘制边界框，不填充，设置填充属性为 False
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.))
        # 状态标记为非过期
        self.stale = False
class TextArea(OffsetBox):
    """
    The TextArea is a container artist for a single Text instance.

    The text is placed at (0, 0) with baseline+left alignment, by default. The
    width and height of the TextArea instance is the width and height of its
    child text.
    """

    def __init__(self, s,
                 *,
                 textprops=None,
                 multilinebaseline=False,
                 ):
        """
        Parameters
        ----------
        s : str
            The text to be displayed.
        textprops : dict, default: {}
            Dictionary of keyword parameters to be passed to the `.Text`
            instance in the TextArea.
        multilinebaseline : bool, default: False
            Whether the baseline for multiline text is adjusted so that it
            is (approximately) center-aligned with single-line text.
        """
        if textprops is None:
            textprops = {}
        # 创建一个文本对象，基于传入的文本s和可选的textprops参数
        self._text = mtext.Text(0, 0, s, **textprops)
        super().__init__()
        # 将文本对象作为子元素添加到TextArea中
        self._children = [self._text]
        # 初始化偏移变换和基线变换
        self.offset_transform = mtransforms.Affine2D()
        self._baseline_transform = mtransforms.Affine2D()
        # 将文本对象的变换设置为偏移变换和基线变换的组合
        self._text.set_transform(self.offset_transform +
                                 self._baseline_transform)
        # 设置是否多行文本基线对齐的标志
        self._multilinebaseline = multilinebaseline

    def set_text(self, s):
        """Set the text of this area as a string."""
        # 设置TextArea显示的文本内容
        self._text.set_text(s)
        self.stale = True

    def get_text(self):
        """Return the string representation of this area's text."""
        # 返回当前TextArea显示的文本内容
        return self._text.get_text()

    def set_multilinebaseline(self, t):
        """
        Set multilinebaseline.

        If True, the baseline for multiline text is adjusted so that it is
        (approximately) center-aligned with single-line text.  This is used
        e.g. by the legend implementation so that single-line labels are
        baseline-aligned, but multiline labels are "center"-aligned with them.
        """
        # 设置是否多行文本基线对齐的属性，并标记为需要更新
        self._multilinebaseline = t
        self.stale = True

    def get_multilinebaseline(self):
        """
        Get multilinebaseline.
        """
        # 返回当前是否多行文本基线对齐的属性
        return self._multilinebaseline

    def set_transform(self, t):
        """
        set_transform is ignored.
        """
        # 忽略设置变换的操作

    def set_offset(self, xy):
        """
        Set the offset of the container.

        Parameters
        ----------
        xy : (float, float)
            The (x, y) coordinates of the offset in display units.
        """
        # 设置TextArea容器的偏移量，并更新偏移变换
        self._offset = xy
        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])
        self.stale = True

    def get_offset(self):
        """Return offset of the container."""
        # 返回TextArea容器当前的偏移量
        return self._offset
    def get_bbox(self, renderer):
        # 调用 renderer 的方法获取文本宽度、高度和下降值
        _, h_, d_ = renderer.get_text_width_height_descent(
            "lp", self._text._fontproperties,
            ismath="TeX" if self._text.get_usetex() else False)

        # 调用 _text 对象的方法获取文本布局信息
        bbox, info, yd = self._text._get_layout(renderer)
        # 获取文本框的宽度和高度
        w, h = bbox.size

        # 清空 _baseline_transform 变换对象
        self._baseline_transform.clear()

        # 如果文本行数大于1并且允许多行基线
        if len(info) > 1 and self._multilinebaseline:
            # 计算新的 yd 值，使得文本居中对齐
            yd_new = 0.5 * h - 0.5 * (h_ - d_)
            self._baseline_transform.translate(0, yd - yd_new)
            yd = yd_new
        else:  # 单行文本
            # 计算单行文本的高度
            h_d = max(h_ - d_, h - yd)
            h = h_d + yd

        # 获取文本的水平对齐方式
        ha = self._text.get_horizontalalignment()
        # 根据水平对齐方式计算起始位置 x0
        x0 = {"left": 0, "center": -w / 2, "right": -w}[ha]

        # 返回一个 Bbox 对象，表示文本的边界框
        return Bbox.from_bounds(x0, -yd, w, h)

    def draw(self, renderer):
        # 绘制文本内容到 renderer
        self._text.draw(renderer)
        # 绘制当前对象的边界框，使用 _bbox_artist 函数
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.))
        # 更新对象状态为非过期
        self.stale = False
class AuxTransformBox(OffsetBox):
    """
    Offset Box with the aux_transform. Its children will be
    transformed with the aux_transform first then will be
    offsetted. The absolute coordinate of the aux_transform is meaning
    as it will be automatically adjust so that the left-lower corner
    of the bounding box of children will be set to (0, 0) before the
    offset transform.

    It is similar to drawing area, except that the extent of the box
    is not predetermined but calculated from the window extent of its
    children. Furthermore, the extent of the children will be
    calculated in the transformed coordinate.
    """

    def __init__(self, aux_transform):
        """
        Initialize AuxTransformBox with an auxiliary transform.

        Parameters
        ----------
        aux_transform : :class:`~matplotlib.transforms.Transform`
            The auxiliary transform applied to children before offsetting.
        """
        self.aux_transform = aux_transform
        super().__init__()
        self.offset_transform = mtransforms.Affine2D()
        # ref_offset_transform makes offset_transform always relative to the
        # lower-left corner of the bbox of its children.
        self.ref_offset_transform = mtransforms.Affine2D()

    def add_artist(self, a):
        """
        Add an `.Artist` to the container box.

        Parameters
        ----------
        a : :class:`~matplotlib.artist.Artist`
            The artist (such as a plot element) to be added to the box.
        """
        self._children.append(a)
        a.set_transform(self.get_transform())
        self.stale = True

    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform` applied
        to the children.
        """
        return (self.aux_transform
                + self.ref_offset_transform
                + self.offset_transform)

    def set_transform(self, t):
        """
        set_transform is ignored.
        
        Parameters
        ----------
        t : :class:`~matplotlib.transforms.Transform`
            The transform to set (ignored in this implementation).
        """
        pass

    def set_offset(self, xy):
        """
        Set the offset of the container.

        Parameters
        ----------
        xy : (float, float)
            The (x, y) coordinates of the offset in display units.
        """
        self._offset = xy
        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])
        self.stale = True

    def get_offset(self):
        """
        Return offset of the container.
        
        Returns
        -------
        (float, float)
            The current offset coordinates of the container.
        """
        return self._offset

    def get_bbox(self, renderer):
        """
        Return the bounding box of the container.

        Parameters
        ----------
        renderer : :class:`~matplotlib.backend_bases.RendererBase`
            The renderer that will be used to draw the container.

        Returns
        -------
        :class:`~matplotlib.transforms.Bbox`
            The bounding box (`Bbox`) of the container.
        """
        # clear the offset transforms
        _off = self.offset_transform.get_matrix()  # to be restored later
        self.ref_offset_transform.clear()
        self.offset_transform.clear()
        
        # calculate the extent
        bboxes = [c.get_window_extent(renderer) for c in self._children]
        ub = Bbox.union(bboxes)
        
        # adjust ref_offset_transform
        self.ref_offset_transform.translate(-ub.x0, -ub.y0)
        
        # restore offset transform
        self.offset_transform.set_matrix(_off)
        
        return Bbox.from_bounds(0, 0, ub.width, ub.height)

    def draw(self, renderer):
        """
        Draw the container and its children.

        Parameters
        ----------
        renderer : :class:`~matplotlib.backend_bases.RendererBase`
            The renderer that will be used to draw the container.
        """
        # docstring inherited
        for c in self._children:
            c.draw(renderer)
        
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.))
        self.stale = False
    # AnchoredOffsetbox类有一个子元素。当需要多个子元素时，使用额外的OffsetBox来包裹它们。
    # 默认情况下，偏移框相对于其父Axes进行定位。您可以显式指定*bbox_to_anchor*。
    """
    zorder = 5  # 图例的zorder，控制图层顺序
    
    # 位置代码
    codes = {'upper right': 1,    # 右上角
             'upper left': 2,     # 左上角
             'lower left': 3,     # 左下角
             'lower right': 4,    # 右下角
             'right': 5,          # 右侧中央
             'center left': 6,    # 左侧中央
             'center right': 7,   # 右侧中央
             'lower center': 8,   # 下方中央
             'upper center': 9,   # 上方中央
             'center': 10,        # 正中央
             }
    def __init__(self, loc, *,
                 pad=0.4, borderpad=0.5,
                 child=None, prop=None, frameon=True,
                 bbox_to_anchor=None,
                 bbox_transform=None,
                 **kwargs):
        """
        Parameters
        ----------
        loc : str
            The box location.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.4
            Padding around the child as fraction of the fontsize.
        borderpad : float, default: 0.5
            Padding between the offsetbox frame and the *bbox_to_anchor*.
        child : `.OffsetBox`
            The box that will be anchored.
        prop : `.FontProperties`
            This is only used as a reference for paddings. If not given,
            :rc:`legend.fontsize` is used.
        frameon : bool
            Whether to draw a frame around the box.
        bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats
            Box that is used to position the legend in conjunction with *loc*.
        bbox_transform : None or :class:`matplotlib.transforms.Transform`
            The transform for the bounding box (*bbox_to_anchor*).
        **kwargs
            All other parameters are passed on to `.OffsetBox`.

        Notes
        -----
        See `.Legend` for a detailed description of the anchoring mechanism.
        """
        # 调用父类构造函数，传入所有额外的关键字参数
        super().__init__(**kwargs)

        # 设置边界框锚点和变换
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)
        # 设置子对象
        self.set_child(child)

        # 如果loc是字符串，则根据字符串获取相应的位置码
        if isinstance(loc, str):
            loc = _api.check_getitem(self.codes, loc=loc)

        # 设置位置码、边界填充和填充
        self.loc = loc
        self.borderpad = borderpad
        self.pad = pad

        # 如果prop为None，则使用默认的字体属性大小
        if prop is None:
            self.prop = FontProperties(size=mpl.rcParams["legend.fontsize"])
        else:
            # 否则从prop创建字体属性对象
            self.prop = FontProperties._from_any(prop)
            # 如果prop是字典且没有指定大小，则设置字体属性大小为默认大小
            if isinstance(prop, dict) and "size" not in prop:
                self.prop.set_size(mpl.rcParams["legend.fontsize"])

        # 创建FancyBboxPatch对象作为边界框
        self.patch = FancyBboxPatch(
            xy=(0.0, 0.0), width=1., height=1.,
            facecolor='w', edgecolor='k',
            mutation_scale=self.prop.get_size_in_points(),
            snap=True,
            visible=frameon,
            boxstyle="square,pad=0",
        )

    def set_child(self, child):
        """Set the child to be anchored."""
        # 设置将被锚定的子对象，并标记为需要更新
        self._child = child
        if child is not None:
            child.axes = self.axes
        self.stale = True

    def get_child(self):
        """Return the child."""
        # 返回当前被锚定的子对象
        return self._child

    def get_children(self):
        """Return the list of children."""
        # 返回包含当前被锚定子对象的列表
        return [self._child]
    def get_bbox(self, renderer):
        # 继承文档字符串
        # 获取字体大小（像素），根据当前属性的大小转换
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        # 计算填充值，将填充参数乘以字体大小
        pad = self.pad * fontsize
        # 调用子对象的 get_bbox 方法，返回填充后的边界框
        return self.get_child().get_bbox(renderer).padded(pad)

    def get_bbox_to_anchor(self):
        """Return the bbox that the box is anchored to."""
        # 如果 _bbox_to_anchor 为 None，则返回 axes 的边界框
        if self._bbox_to_anchor is None:
            return self.axes.bbox
        else:
            # 否则，返回经过 _bbox_to_anchor_transform 变换后的 _bbox_to_anchor
            transform = self._bbox_to_anchor_transform
            if transform is None:
                return self._bbox_to_anchor
            else:
                return TransformedBbox(self._bbox_to_anchor, transform)

    def set_bbox_to_anchor(self, bbox, transform=None):
        """
        Set the bbox that the box is anchored to.

        *bbox* can be a Bbox instance, a list of [left, bottom, width,
        height], or a list of [left, bottom] where the width and
        height will be assumed to be zero. The bbox will be
        transformed to display coordinate by the given transform.
        """
        # 如果 bbox 是 None 或 BboxBase 的实例，直接赋值给 _bbox_to_anchor
        if bbox is None or isinstance(bbox, BboxBase):
            self._bbox_to_anchor = bbox
        else:
            # 否则，尝试获取 bbox 的长度，如果出错则抛出 ValueError 异常
            try:
                l = len(bbox)
            except TypeError as err:
                raise ValueError(f"Invalid bbox: {bbox}") from err

            # 如果长度为 2，则将 bbox 转换为 [left, bottom, 0, 0] 形式
            if l == 2:
                bbox = [bbox[0], bbox[1], 0, 0]

            # 使用 Bbox.from_bounds 方法创建 Bbox 对象并赋值给 _bbox_to_anchor
            self._bbox_to_anchor = Bbox.from_bounds(*bbox)

        # 将 transform 赋值给 _bbox_to_anchor_transform
        self._bbox_to_anchor_transform = transform
        # 将 stale 标记为 True，表示需要更新
        self.stale = True

    @_compat_get_offset
    def get_offset(self, bbox, renderer):
        # 继承文档字符串
        # 计算填充值，将边框填充参数乘以字体大小
        pad = (self.borderpad
               * renderer.points_to_pixels(self.prop.get_size_in_points()))
        # 获取与锚点边界框相关的 bbox_to_anchor
        bbox_to_anchor = self.get_bbox_to_anchor()
        # 获取锚点的偏移坐标
        x0, y0 = _get_anchored_bbox(
            self.loc, Bbox.from_bounds(0, 0, bbox.width, bbox.height),
            bbox_to_anchor, pad)
        # 返回偏移坐标
        return x0 - bbox.x0, y0 - bbox.y0

    def update_frame(self, bbox, fontsize=None):
        # 设置 patch 对象的边界
        self.patch.set_bounds(bbox.bounds)
        # 如果提供了 fontsize，则设置变异比例
        if fontsize:
            self.patch.set_mutation_scale(fontsize)

    def draw(self, renderer):
        # 继承文档字符串
        # 如果不可见，则直接返回
        if not self.get_visible():
            return

        # 更新图例的位置和大小
        bbox = self.get_window_extent(renderer)
        # 获取字体大小（像素），根据当前属性的大小转换
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        # 更新框架（边界框）并绘制 patch
        self.update_frame(bbox, fontsize)
        self.patch.draw(renderer)

        # 获取偏移坐标
        px, py = self.get_offset(self.get_bbox(renderer), renderer)
        # 设置子对象的偏移
        self.get_child().set_offset((px, py))
        # 绘制子对象
        self.get_child().draw(renderer)
        # 将 stale 标记为 False，表示不需要更新
        self.stale = False
def _get_anchored_bbox(loc, bbox, parentbbox, borderpad):
    """
    Return the (x, y) position of the *bbox* anchored at the *parentbbox* with
    the *loc* code with the *borderpad*.
    """
    # 确定辅助框的定位，基于给定的位置代码和父辅助框的边框填充
    c = [None, "NE", "NW", "SW", "SE", "E", "W", "E", "S", "N", "C"][loc]
    # 基于父辅助框的边框填充，创建一个包含边框填充的容器
    container = parentbbox.padded(-borderpad)
    # 返回相对于容器的锚定边框的左上角位置
    return bbox.anchored(c, container=container).p0


class AnchoredText(AnchoredOffsetbox):
    """
    AnchoredOffsetbox with Text.
    """

    def __init__(self, s, loc, *, pad=0.4, borderpad=0.5, prop=None, **kwargs):
        """
        Parameters
        ----------
        s : str
            Text.

        loc : str
            Location code. See `AnchoredOffsetbox`.

        pad : float, default: 0.4
            Padding around the text as fraction of the fontsize.

        borderpad : float, default: 0.5
            Spacing between the offsetbox frame and the *bbox_to_anchor*.

        prop : dict, optional
            Dictionary of keyword parameters to be passed to the
            `~matplotlib.text.Text` instance contained inside AnchoredText.

        **kwargs
            All other parameters are passed to `AnchoredOffsetbox`.
        """

        if prop is None:
            prop = {}
        # 禁止在AnchoredText中混合使用垂直对齐参数
        badkwargs = {'va', 'verticalalignment'}
        if badkwargs & set(prop):
            raise ValueError(
                'Mixing verticalalignment with AnchoredText is not supported.')

        # 创建一个包含文本的TextArea对象
        self.txt = TextArea(s, textprops=prop)
        # 获取文本字体属性
        fp = self.txt._text.get_fontproperties()
        # 调用AnchoredOffsetbox的构造函数，初始化带有文本的偏移框
        super().__init__(
            loc, pad=pad, borderpad=borderpad, child=self.txt, prop=fp,
            **kwargs)


class OffsetImage(OffsetBox):

    def __init__(self, arr, *,
                 zoom=1,
                 cmap=None,
                 norm=None,
                 interpolation=None,
                 origin=None,
                 filternorm=True,
                 filterrad=4.0,
                 resample=False,
                 dpi_cor=True,
                 **kwargs
                 ):

        super().__init__()
        # 是否进行DPI校正
        self._dpi_cor = dpi_cor

        # 创建一个BboxImage对象，用于显示偏移图像
        self.image = BboxImage(bbox=self.get_window_extent,
                               cmap=cmap,
                               norm=norm,
                               interpolation=interpolation,
                               origin=origin,
                               filternorm=filternorm,
                               filterrad=filterrad,
                               resample=resample,
                               **kwargs
                               )

        # 将图像对象作为子元素添加到偏移框的子元素列表中
        self._children = [self.image]

        # 设置图像的缩放比例
        self.set_zoom(zoom)
        # 设置图像的数据
        self.set_data(arr)

    def set_data(self, arr):
        # 将输入数组转换为NumPy数组
        self._data = np.asarray(arr)
        # 设置图像对象的数据
        self.image.set_data(self._data)
        # 标记对象为过时的（需要重新绘制）
        self.stale = True

    def get_data(self):
        # 返回当前图像数据
        return self._data
    # 设置缩放级别
    def set_zoom(self, zoom):
        # 将缩放级别存储在实例变量中
        self._zoom = zoom
        # 设置对象状态为过时，需要重新绘制
        self.stale = True

    # 获取缩放级别
    def get_zoom(self):
        # 返回存储在实例变量中的缩放级别
        return self._zoom

    # 获取容器的偏移量
    def get_offset(self):
        """Return offset of the container."""
        # 返回存储在实例变量中的偏移量
        return self._offset

    # 获取子元素列表
    def get_children(self):
        # 返回包含唯一子元素（图像）的列表
        return [self.image]

    # 获取边界框
    def get_bbox(self, renderer):
        # 根据渲染器将点转换为像素，并考虑 DPI 校正因素
        dpi_cor = renderer.points_to_pixels(1.) if self._dpi_cor else 1.
        # 获取当前缩放级别
        zoom = self.get_zoom()
        # 获取数据（假定是二维数组），获取其形状
        data = self.get_data()
        ny, nx = data.shape[:2]
        # 计算边界框的宽度和高度，考虑 DPI 校正和缩放级别
        w, h = dpi_cor * nx * zoom, dpi_cor * ny * zoom
        # 创建并返回边界框对象，基于给定的边界参数
        return Bbox.from_bounds(0, 0, w, h)

    # 绘制方法
    def draw(self, renderer):
        # 继承的文档字符串（假设是从父类继承的）
        # 调用子元素（图像）的绘制方法
        self.image.draw(renderer)
        # 标记对象状态为非过时，不需要重新绘制
        self.stale = False
class AnnotationBbox(martist.Artist, mtext._AnnotationBase):
    """
    Container for an `OffsetBox` referring to a specific position *xy*.

    Optionally an arrow pointing from the offsetbox to *xy* can be drawn.

    This is like `.Annotation`, but with `OffsetBox` instead of `.Text`.
    """

    zorder = 3  # 设置对象的绘制顺序，用于控制图层叠加顺序

    def __str__(self):
        return f"AnnotationBbox({self.xy[0]:g},{self.xy[1]:g})"
        # 返回对象的字符串表示，包含 xy 坐标信息

    @_docstring.dedent_interpd
    def __init__(self, offsetbox, xy, xybox=None, xycoords='data', boxcoords=None, *,
                 frameon=True, pad=0.4,  # FancyBboxPatch boxstyle.
                 annotation_clip=None,
                 box_alignment=(0.5, 0.5),
                 bboxprops=None,
                 arrowprops=None,
                 fontsize=None,
                 **kwargs):
        """
        Parameters
        ----------
        offsetbox : `OffsetBox`
            用于注释的偏移框对象

        xy : (float, float)
            要注释的点 *(x, y)* 的坐标。其坐标系统由 *xycoords* 决定。

        xybox : (float, float), default: *xy*
            文本放置位置的坐标 *(x, y)*。其坐标系统由 *boxcoords* 决定。

        xycoords : single or two-tuple of str or `.Artist` or `.Transform` or \
callable, default: 'data'
            *xy* 使用的坐标系统。详见 `.Annotation` 中的 *xycoords* 参数的详细描述。

        boxcoords : single or two-tuple of str or `.Artist` or `.Transform` \
@property
    def xyann(self):
        return self.xybox
        # 获取 xyann 属性的值，返回 xybox 的坐标

    @xyann.setter
    def xyann(self, xyann):
        self.xybox = xyann
        self.stale = True
        # 设置 xyann 属性，更新 xybox 的坐标，并标记为不可用状态

    @property
    def anncoords(self):
        return self.boxcoords
        # 获取 anncoords 属性的值，返回 boxcoords 的坐标系统

    @anncoords.setter
    def anncoords(self, coords):
        self.boxcoords = coords
        self.stale = True
        # 设置 anncoords 属性，更新 boxcoords 的坐标系统，并标记为不可用状态

    def contains(self, mouseevent):
        if self._different_canvas(mouseevent):
            return False, {}
            # 检查鼠标事件是否在不同的画布上，若是则返回 False
        if not self._check_xy(None):
            return False, {}
            # 检查 xy 坐标是否有效，若无效则返回 False
        return self.offsetbox.contains(mouseevent)
        # 检查 offsetbox 是否包含鼠标事件，返回检查结果

    def get_children(self):
        children = [self.offsetbox, self.patch]
        if self.arrow_patch:
            children.append(self.arrow_patch)
            # 返回对象的子元素列表，包括 offsetbox 和 patch，以及可能的 arrow_patch
        return children

    def set_figure(self, fig):
        if self.arrow_patch is not None:
            self.arrow_patch.set_figure(fig)
        self.offsetbox.set_figure(fig)
        martist.Artist.set_figure(self, fig)
        # 设置对象所属的图形 figure，更新相关的子元素的图形属性

    def set_fontsize(self, s=None):
        """
        Set the fontsize in points.

        If *s* is not given, reset to :rc:`legend.fontsize`.
        """
        if s is None:
            s = mpl.rcParams["legend.fontsize"]
            # 设置字体大小（以点为单位）

        self.prop = FontProperties(size=s)
        self.stale = True
        # 更新字体属性并标记为不可用状态

    def get_fontsize(self):
        """Return the fontsize in points."""
        return self.prop.get_size_in_points()
        # 获取当前字体大小（以点为单位）
    def get_window_extent(self, renderer=None):
        # docstring inherited
        # 如果没有指定渲染器，则使用图形对象的渲染器
        if renderer is None:
            renderer = self.figure._get_renderer()
        # 更新子对象的位置信息
        self.update_positions(renderer)
        # 返回所有子对象的窗口边界框的并集
        return Bbox.union([child.get_window_extent(renderer)
                           for child in self.get_children()])

    def get_tightbbox(self, renderer=None):
        # docstring inherited
        # 如果没有指定渲染器，则使用图形对象的渲染器
        if renderer is None:
            renderer = self.figure._get_renderer()
        # 更新子对象的位置信息
        self.update_positions(renderer)
        # 返回所有子对象的紧凑边界框的并集
        return Bbox.union([child.get_tightbbox(renderer)
                           for child in self.get_children()])

    def update_positions(self, renderer):
        """Update pixel positions for the annotated point, the text, and the arrow."""

        # 获取注释点、文本和箭头的像素位置更新

        # 根据指定的坐标系和偏移框获取起始点的偏移量
        ox0, oy0 = self._get_xy(renderer, self.xybox, self.boxcoords)
        # 获取偏移框的边界框
        bbox = self.offsetbox.get_bbox(renderer)
        # 获取框的对齐方式
        fw, fh = self._box_alignment
        # 设置偏移框的偏移量
        self.offsetbox.set_offset(
            (ox0 - fw*bbox.width - bbox.x0, oy0 - fh*bbox.height - bbox.y0))

        # 获取偏移框的窗口边界框
        bbox = self.offsetbox.get_window_extent(renderer)
        # 设置图形对象的边界
        self.patch.set_bounds(bbox.bounds)

        # 根据字体大小设置突变尺度
        mutation_scale = renderer.points_to_pixels(self.get_fontsize())
        self.patch.set_mutation_scale(mutation_scale)

        if self.arrowprops:
            # 如果箭头属性存在，则使用FancyArrowPatch

            # 调整箭头的起始点相对于文本框的位置
            # TODO: 需要考虑旋转
            arrow_begin = bbox.p0 + bbox.size * self._arrow_relpos
            # 获取箭头的结束点位置
            arrow_end = self._get_position_xy(renderer)
            # 设置箭头的位置
            self.arrow_patch.set_positions(arrow_begin, arrow_end)

            # 如果箭头属性中定义了"mutation_scale"，则使用其定义的尺度
            if "mutation_scale" in self.arrowprops:
                mutation_scale = renderer.points_to_pixels(
                    self.arrowprops["mutation_scale"])
            # 否则，使用基于字体大小的突变尺度

            self.arrow_patch.set_mutation_scale(mutation_scale)

            # 获取箭头属性中的patchA，默认使用self.patch
            patchA = self.arrowprops.get("patchA", self.patch)
            self.arrow_patch.set_patchA(patchA)

    def draw(self, renderer):
        # docstring inherited
        # 如果对象不可见或者检查坐标不通过，则直接返回
        if not self.get_visible() or not self._check_xy(renderer):
            return
        # 打开渲染器的分组
        renderer.open_group(self.__class__.__name__, gid=self.get_gid())
        # 更新对象的位置信息
        self.update_positions(renderer)
        # 如果箭头对象存在且尚未与图形对象关联，则关联它们
        if self.arrow_patch is not None:
            if self.arrow_patch.figure is None and self.figure is not None:
                self.arrow_patch.figure = self.figure
            # 绘制箭头对象
            self.arrow_patch.draw(renderer)
        # 绘制主要图形对象的边界
        self.patch.draw(renderer)
        # 绘制偏移框对象
        self.offsetbox.draw(renderer)
        # 关闭渲染器的分组
        renderer.close_group(self.__class__.__name__)
        # 设置对象为非过时状态
        self.stale = False
    """
    Helper base class for a draggable artist (legend, offsetbox).

    Derived classes must override the following methods::

        def save_offset(self):
            '''
            Called when the object is picked for dragging; should save the
            reference position of the artist.
            '''

        def update_offset(self, dx, dy):
            '''
            Called during the dragging; (*dx*, *dy*) is the pixel offset from
            the point where the mouse drag started.
            '''

    Optionally, you may override the following method::

        def finalize_offset(self):
            '''Called when the mouse is released.'''

    In the current implementation of `.DraggableLegend` and
    `DraggableAnnotation`, `update_offset` places the artists in display
    coordinates, and `finalize_offset` recalculates their position in axes
    coordinate and set a relevant attribute.
    """

    def __init__(self, ref_artist, use_blit=False):
        # Initialize the DraggableBase with a reference artist and blitting option
        self.ref_artist = ref_artist
        # Ensure the reference artist is pickable
        if not ref_artist.pickable():
            ref_artist.set_picker(True)
        self.got_artist = False
        # Determine if blitting is supported by the canvas and requested
        self._use_blit = use_blit and self.canvas.supports_blit
        # Access canvas callbacks and setup disconnectors for pick events
        callbacks = self.canvas.callbacks
        self._disconnectors = [
            functools.partial(
                callbacks.disconnect, callbacks._connect_picklable(name, func))
            for name, func in [
                ("pick_event", self.on_pick),  # Connects pick event to on_pick method
                ("button_release_event", self.on_release),  # Connects release event to on_release method
                ("motion_notify_event", self.on_motion),  # Connects motion event to on_motion method
            ]
        ]

    # A property, not an attribute, to maintain picklability.
    canvas = property(lambda self: self.ref_artist.figure.canvas)
    # Retrieves the event identifiers for pick_event and button_release_event
    cids = property(lambda self: [
        disconnect.args[0] for disconnect in self._disconnectors[:2]])

    def on_motion(self, evt):
        # Handles mouse motion events when an artist is being dragged
        if self._check_still_parented() and self.got_artist:
            # Calculate pixel offsets
            dx = evt.x - self.mouse_x
            dy = evt.y - self.mouse_y
            # Update the offset of the artist
            self.update_offset(dx, dy)
            if self._use_blit:
                # If blitting, restore the canvas region and redraw the artist
                self.canvas.restore_region(self.background)
                self.ref_artist.draw(
                    self.ref_artist.figure._get_renderer())
                self.canvas.blit()
            else:
                # If not blitting, redraw the entire canvas
                self.canvas.draw()

    def on_pick(self, evt):
        # Handles pick events when an artist is selected
        if self._check_still_parented() and evt.artist == self.ref_artist:
            # Record mouse coordinates
            self.mouse_x = evt.mouseevent.x
            self.mouse_y = evt.mouseevent.y
            self.got_artist = True
            if self._use_blit:
                # Enable animated mode and initiate blitting
                self.ref_artist.set_animated(True)
                self.canvas.draw()
                self.background = \
                    self.canvas.copy_from_bbox(self.ref_artist.figure.bbox)
                self.ref_artist.draw(
                    self.ref_artist.figure._get_renderer())
                self.canvas.blit()
            # Save the initial offset
            self.save_offset()
    # 在鼠标释放事件发生时调用的方法
    def on_release(self, event):
        # 检查当前对象是否仍然附属于父对象，并且已经获取到艺术对象
        if self._check_still_parented() and self.got_artist:
            # 完成最终的偏移处理
            self.finalize_offset()
            # 将 got_artist 标记设为 False，表示不再持有艺术对象
            self.got_artist = False
            # 如果使用了 blit 技术，取消参考艺术对象的动画状态
            if self._use_blit:
                self.ref_artist.set_animated(False)

    # 检查当前参考艺术对象是否仍然属于其图形对象的方法
    def _check_still_parented(self):
        if self.ref_artist.figure is None:
            # 如果参考艺术对象的图形对象为空，断开当前对象的连接并返回 False
            self.disconnect()
            return False
        else:
            # 如果参考艺术对象仍然有图形对象，返回 True
            return True

    # 断开当前对象的回调连接的方法
    def disconnect(self):
        """Disconnect the callbacks."""
        # 遍历所有的断开连接器，并执行它们，用于断开当前对象的所有回调连接
        for disconnector in self._disconnectors:
            disconnector()

    # 保存偏移量的方法（占位方法，暂时未实现具体功能）
    def save_offset(self):
        pass

    # 更新偏移量的方法（占位方法，暂时未实现具体功能）
    def update_offset(self, dx, dy):
        pass

    # 完成偏移处理的方法（占位方法，暂时未实现具体功能）
    def finalize_offset(self):
        pass
# 定义一个可拖动的偏移框类，继承自DraggableBase
class DraggableOffsetBox(DraggableBase):
    def __init__(self, ref_artist, offsetbox, use_blit=False):
        # 调用父类的初始化方法
        super().__init__(ref_artist, use_blit=use_blit)
        # 设置偏移框属性
        self.offsetbox = offsetbox

    # 保存偏移量的方法
    def save_offset(self):
        # 获取偏移框对象
        offsetbox = self.offsetbox
        # 获取偏移框所在图形的渲染器
        renderer = offsetbox.figure._get_renderer()
        # 获取偏移框的边界框，并计算偏移量
        offset = offsetbox.get_offset(offsetbox.get_bbox(renderer), renderer)
        # 将偏移量分别保存到对象的属性中
        self.offsetbox_x, self.offsetbox_y = offset
        # 设置偏移框的偏移量
        self.offsetbox.set_offset(offset)

    # 更新偏移量的方法
    def update_offset(self, dx, dy):
        # 获取偏移框对象
        offsetbox = self.offsetbox
        # 计算新的在画布中的位置
        loc_in_canvas = self.offsetbox_x + dx, self.offsetbox_y + dy
        # 设置偏移框的新偏移量
        self.offsetbox.set_offset(loc_in_canvas)

    # 获取偏移框在画布中的位置的方法
    def get_loc_in_canvas(self):
        # 获取偏移框对象
        offsetbox = self.offsetbox
        # 获取偏移框所在图形的渲染器
        renderer = offsetbox.figure._get_renderer()
        # 获取偏移框的边界框
        bbox = offsetbox.get_bbox(renderer)
        # 获取偏移框当前的偏移量
        ox, oy = offsetbox._offset
        # 计算偏移框在画布中的具体位置
        loc_in_canvas = (ox + bbox.x0, oy + bbox.y0)
        # 返回计算出的位置
        return loc_in_canvas


# 定义一个可拖动的注释类，继承自DraggableBase
class DraggableAnnotation(DraggableBase):
    def __init__(self, annotation, use_blit=False):
        # 调用父类的初始化方法
        super().__init__(annotation, use_blit=use_blit)
        # 设置注释对象属性
        self.annotation = annotation

    # 保存偏移量的方法
    def save_offset(self):
        # 获取注释对象
        ann = self.annotation
        # 获取当前注释的偏移量，并保存到对象的属性中
        self.ox, self.oy = ann.get_transform().transform(ann.xyann)

    # 更新偏移量的方法
    def update_offset(self, dx, dy):
        # 获取注释对象
        ann = self.annotation
        # 计算新的偏移位置，并设置给注释对象
        ann.xyann = ann.get_transform().inverted().transform(
            (self.ox + dx, self.oy + dy))
```