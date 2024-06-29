# `D:\src\scipysrc\matplotlib\lib\matplotlib\legend.py`

```py
"""
The legend module defines the Legend class, which is responsible for
drawing legends associated with Axes and/or figures.

.. important::

    It is unlikely that you would ever create a Legend instance manually.
    Most users would normally create a legend via the `~.Axes.legend`
    function. For more details on legends there is also a :ref:`legend guide
    <legend_guide>`.

The `Legend` class is a container of legend handles and legend texts.

The legend handler map specifies how to create legend handles from artists
(lines, patches, etc.) in the Axes or figures. Default legend handlers are
defined in the :mod:`~matplotlib.legend_handler` module. While not all artist
types are covered by the default legend handlers, custom legend handlers can be
defined to support arbitrary objects.

See the :ref`<legend_guide>` for more
information.
"""

import itertools                     # 导入 itertools 模块，用于创建迭代器的函数
import logging                       # 导入 logging 模块，用于记录日志
import numbers                       # 导入 numbers 模块，用于数值相关的抽象基类
import time                          # 导入 time 模块，用于时间相关操作

import numpy as np                   # 导入 NumPy 库并重命名为 np

import matplotlib as mpl             # 导入 matplotlib 库并重命名为 mpl
from matplotlib import _api, _docstring, cbook, colors, offsetbox  # 导入 matplotlib 中的各类模块和函数
from matplotlib.artist import Artist, allow_rasterization          # 导入 Artist 类和 allow_rasterization 函数
from matplotlib.cbook import silent_list                           # 导入 silent_list 函数
from matplotlib.font_manager import FontProperties                 # 导入 FontProperties 类
from matplotlib.lines import Line2D                               # 导入 Line2D 类
from matplotlib.patches import (                                   # 导入各种图形类，如 Patch, Rectangle, 等
    Patch, Rectangle, Shadow, FancyBboxPatch,
    StepPatch
)
from matplotlib.collections import (                               # 导入集合类，如 Collection, CircleCollection, 等
    Collection, CircleCollection, LineCollection, PathCollection,
    PolyCollection, RegularPolyCollection
)
from matplotlib.text import Text                                   # 导入 Text 类
from matplotlib.transforms import (                                # 导入变换相关类和函数，如 Bbox, BboxBase, 等
    Bbox, BboxBase, TransformedBbox,
    BboxTransformTo, BboxTransformFrom
)
from matplotlib.offsetbox import (                                 # 导入偏移框相关类，如 AnchoredOffsetbox, 等
    AnchoredOffsetbox, DraggableOffsetBox,
    HPacker, VPacker,
    DrawingArea, TextArea
)
from matplotlib.container import (                                 # 导入容器类，如 ErrorbarContainer, BarContainer, 等
    ErrorbarContainer, BarContainer, StemContainer
)
from . import legend_handler                                       # 导入当前包中的 legend_handler 模块

class DraggableLegend(DraggableOffsetBox):
    def __init__(self, legend, use_blit=False, update="loc"):
        """
        Wrapper around a `.Legend` to support mouse dragging.

        Parameters
        ----------
        legend : `.Legend`
            The `.Legend` instance to wrap.
        use_blit : bool, optional
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.
        update : {'loc', 'bbox'}, optional
            If "loc", update the *loc* parameter of the legend upon finalizing.
            If "bbox", update the *bbox_to_anchor* parameter.
        """
        self.legend = legend                                # 初始化属性 legend，表示要包装的 Legend 实例

        _api.check_in_list(["loc", "bbox"], update=update)   # 检查 update 参数是否在 ["loc", "bbox"] 中
        self._update = update                               # 初始化属性 _update，表示更新参数类型

        super().__init__(legend, legend._legend_box, use_blit=use_blit)  # 调用父类 DraggableOffsetBox 的初始化方法

    def finalize_offset(self):
        if self._update == "loc":                           # 如果 _update 属性为 "loc"
            self._update_loc(self.get_loc_in_canvas())      # 调用 _update_loc 方法，更新 legend 的位置参数
        elif self._update == "bbox":                        # 如果 _update 属性为 "bbox"
            self._update_bbox_to_anchor(self.get_loc_in_canvas())  # 调用 _update_bbox_to_anchor 方法，更新 legend 的 bbox_to_anchor 参数
    # 更新图例对象的位置信息
    def _update_loc(self, loc_in_canvas):
        # 获取图例对象相对于锚点的边界框
        bbox = self.legend.get_bbox_to_anchor()
        # 如果边界框的宽度或高度为零，转换操作将无法定义。回退到默认的bbox_to_anchor。
        if bbox.width == 0 or bbox.height == 0:
            self.legend.set_bbox_to_anchor(None)  # 设置图例对象的边界框锚点为None
            bbox = self.legend.get_bbox_to_anchor()  # 重新获取更新后的边界框
        # 使用边界框创建一个从边界框到画布空间的变换对象
        _bbox_transform = BboxTransformFrom(bbox)
        # 更新图例对象的位置为画布空间中指定位置的坐标
        self.legend._loc = tuple(_bbox_transform.transform(loc_in_canvas))

    # 更新图例对象的边界框锚点位置
    def _update_bbox_to_anchor(self, loc_in_canvas):
        # 将画布空间中的位置坐标转换为相对于轴的坐标系空间中的位置
        loc_in_bbox = self.legend.axes.transAxes.transform(loc_in_canvas)
        # 设置图例对象的边界框锚点位置为新计算的相对位置
        self.legend.set_bbox_to_anchor(loc_in_bbox)
# _legend_kw_doc_base 是一个字符串，包含了 legend 函数的关键字参数的文档基础信息
_legend_kw_doc_base = """
bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats
    用于指定图例相对于 *loc* 的位置的框。默认为 `axes.bbox`（如果调用 `.Axes.legend`）或 `figure.bbox`（如果调用 `.Figure.legend`）。
    此参数允许任意放置图例。

    Bbox 的坐标由 *bbox_transform* 指定的坐标系解释，默认为 Axes 或 Figure 坐标，具体取决于调用哪个 `legend` 方法。

    如果给出了 4-tuple 或 `.BboxBase`，则指定了图例放置的 bbox ``(x, y, width, height)``。
    例如，将图例放置在 Axes（或 figure）右下象限的最佳位置上::

        loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5)

    2-tuple ``(x, y)`` 将由 *loc* 指定的图例角放置在 x, y 处。
    例如，将图例的右上角放在 Axes（或 figure）中心的以下关键字可用于指定::

        loc='upper right', bbox_to_anchor=(0.5, 0.5)

ncols : int, default: 1
    图例的列数。

    为了向后兼容，也支持 *ncol* 的拼写，但不鼓励使用。如果两者都给出，则 *ncols* 优先。

prop : None or `~matplotlib.font_manager.FontProperties` or dict
    图例的字体属性。如果为 None（默认），将使用当前的 :data:`matplotlib.rcParams`。

fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', \
'x-large', 'xx-large'}
    图例的字体大小。如果值是数字，则大小将是绝对的点大小。字符串值相对于当前默认字体大小。只有在未指定 *prop* 时才使用此参数。

labelcolor : str or list, default: :rc:`legend.labelcolor`
    图例文本的颜色。可以是有效的颜色字符串（例如 'red'），或颜色字符串列表。labelcolor 也可以使用 'linecolor'、'markerfacecolor'（或 'mfc'）、'markeredgecolor'（或 'mec'）来匹配线或标记的颜色。

    可以使用 :rc:`legend.labelcolor` 全局设置 Labelcolor。如果为 None，则使用 :rc:`text.color`。

numpoints : int, default: :rc:`legend.numpoints`
    创建 `.Line2D`（线）的图例条目时图例中的标记点数。

scatterpoints : int, default: :rc:`legend.scatterpoints`
    创建 `.PathCollection`（散点图）的图例条目时图例中的标记点数。

scatteryoffsets : iterable of floats, default: ``[0.375, 0.5, 0.3125]``
    用于散点图图例条目中标记的垂直偏移（相对于字体大小）。0.0 在图例文本的基底，1.0 在顶部。为了在相同高度绘制所有标记，请设置为 ``[0.5]``。

markerscale : float, default: :rc:`legend.markerscale`
"""
    # 定义相对于原始绘制的图例标记的大小比例
    The relative size of legend markers compared to the originally drawn ones.
markerfirst : bool, default: True
    # 控制图例标记的位置，默认为 True，表示标记位于图例标签的左侧
    If *True*, legend marker is placed to the left of the legend label.
    # 如果为 True，则图例标记位于图例标签的左侧
    If *False*, legend marker is placed to the right of the legend label.
    # 如果为 False，则图例标记位于图例标签的右侧

reverse : bool, default: False
    # 控制是否以相反的顺序显示图例标签，默认为 False
    If *True*, the legend labels are displayed in reverse order from the input.
    # 如果为 True，则以输入的相反顺序显示图例标签
    If *False*, the legend labels are displayed in the same order as the input.
    # 如果为 False，则以与输入相同的顺序显示图例标签

    .. versionadded:: 3.7
    # 从版本 3.7 开始添加的功能

frameon : bool, default: :rc:`legend.frameon`
    # 控制是否在图例周围绘制边框，默认使用 :rc:`legend.frameon` 的设置
    Whether the legend should be drawn on a patch (frame).

fancybox : bool, default: :rc:`legend.fancybox`
    # 控制图例背景是否启用圆角，默认使用 :rc:`legend.fancybox` 的设置
    Whether round edges should be enabled around the `.FancyBboxPatch` which
    makes up the legend's background.

shadow : None, bool or dict, default: :rc:`legend.shadow`
    # 控制是否在图例后面绘制阴影，默认使用 :rc:`legend.shadow` 的设置
    Whether to draw a shadow behind the legend.
    # 是否在图例后面绘制阴影
    The shadow can be configured using `.Patch` keywords.
    # 可以使用 `.Patch` 的关键词配置阴影
    Customization via :rc:`legend.shadow` is currently not supported.
    # 当前不支持通过 :rc:`legend.shadow` 进行定制化配置

framealpha : float, default: :rc:`legend.framealpha`
    # 控制图例背景的透明度，默认使用 :rc:`legend.framealpha` 的设置
    The alpha transparency of the legend's background.
    # 图例背景的透明度
    If *shadow* is activated and *framealpha* is ``None``, the default value is
    ignored.
    # 如果启用了阴影，并且 *framealpha* 是 ``None``，则忽略默认值

facecolor : "inherit" or color, default: :rc:`legend.facecolor`
    # 控制图例背景的颜色，默认使用 :rc:`legend.facecolor` 的设置
    The legend's background color.
    # 图例背景的颜色
    If ``"inherit"``, use :rc:`axes.facecolor`.
    # 如果是 ``"inherit"``，则使用 :rc:`axes.facecolor`

edgecolor : "inherit" or color, default: :rc:`legend.edgecolor`
    # 控制图例背景边缘的颜色，默认使用 :rc:`legend.edgecolor` 的设置
    The legend's background patch edge color.
    # 图例背景边缘的颜色
    If ``"inherit"``, use :rc:`axes.edgecolor`.
    # 如果是 ``"inherit"``，则使用 :rc:`axes.edgecolor`

mode : {"expand", None}
    # 控制图例的显示模式，默认为 None
    If *mode* is set to ``"expand"`` the legend will be horizontally
    expanded to fill the Axes area (or *bbox_to_anchor* if defines
    the legend's size).
    # 如果设置为 ``"expand"``，则图例将水平扩展以填充 Axes 区域（或 *bbox_to_anchor* 如果定义了图例的大小）

bbox_transform : None or `~matplotlib.transforms.Transform`
    # 定义边界框 (*bbox_to_anchor*) 的变换方式，默认为 ``None``
    The transform for the bounding box (*bbox_to_anchor*). For a value
    of ``None`` (default) the Axes'
    :data:`~matplotlib.axes.Axes.transAxes` transform will be used.

title : str or None
    # 图例的标题，默认为无标题（``None``）
    The legend's title. Default is no title (``None``).

title_fontproperties : None or `~matplotlib.font_manager.FontProperties` or dict
    # 图例标题的字体属性，默认为 ``None``
    The font properties of the legend's title. If None (default), the
    *title_fontsize* argument will be used if present; if *title_fontsize* is
    also None, the current :rc:`legend.title_fontsize` will be used.

title_fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', \
'x-large', 'xx-large'}, default: :rc:`legend.title_fontsize`
    # 图例标题的字体大小，默认使用 :rc:`legend.title_fontsize` 的设置
    The font size of the legend's title.
    # 图例标题的字体大小
    Note: This cannot be combined with *title_fontproperties*. If you want
    to set the fontsize alongside other font properties, use the *size*
    parameter in *title_fontproperties*.
    # 注意：不能与 *title_fontproperties* 结合使用。如果要与其他字体属性一起设置字体大小，请在 *title_fontproperties* 中使用 *size* 参数。

alignment : {'center', 'left', 'right'}, default: 'center'
    # 图例标题和条目框的对齐方式，默认为 'center'
    The alignment of the legend title and the box of entries. The entries
    are aligned as a single block, so that markers always lined up.
    # 图例标题和条目框的对齐方式。条目将作为单个块对齐，使标记始终对齐。

borderpad : float, default: :rc:`legend.borderpad`
    # 图例边框内部的空白百分比，默认使用 :rc:`legend.borderpad` 的设置
    The fractional whitespace inside the legend border, in font-size units.

labelspacing : float, default: :rc:`legend.labelspacing`
    # 图例条目之间的垂直间距，默认使用 :rc:`legend.labelspacing` 的设置
    # 设置图例条目之间的垂直间距，单位是字体大小的倍数。
# legend.handlelength: 图例句柄的长度，以字体大小单位表示，默认使用rc配置中的legend.handlelength值
handlelength : float, default: :rc:`legend.handlelength`
    The length of the legend handles, in font-size units.

# legend.handleheight: 图例句柄的高度，以字体大小单位表示，默认使用rc配置中的legend.handleheight值
handleheight : float, default: :rc:`legend.handleheight`
    The height of the legend handles, in font-size units.

# legend.handletextpad: 图例句柄和文本之间的间距，以字体大小单位表示，默认使用rc配置中的legend.handletextpad值
handletextpad : float, default: :rc:`legend.handletextpad`
    The pad between the legend handle and text, in font-size units.

# legend.borderaxespad: 图例和坐标轴边框之间的间距，以字体大小单位表示，默认使用rc配置中的legend.borderaxespad值
borderaxespad : float, default: :rc:`legend.borderaxespad`
    The pad between the Axes and legend border, in font-size units.

# legend.columnspacing: 列之间的间距，以字体大小单位表示，默认使用rc配置中的legend.columnspacing值
columnspacing : float, default: :rc:`legend.columnspacing`
    The spacing between columns, in font-size units.

# handler_map: 自定义字典，将实例或类型映射到图例处理程序。此handler_map更新matplotlib.legend.Legend.get_legend_handler_map中找到的默认处理程序映射。
handler_map : dict or None
    The custom dictionary mapping instances or types to a legend
    handler. This *handler_map* updates the default handler map
    found at `matplotlib.legend.Legend.get_legend_handler_map`.

# draggable: 是否允许使用鼠标拖动图例，默认为False
draggable : bool, default: False
    Whether the legend can be dragged with the mouse.
"""

_loc_doc_base = """
# loc: 图例的位置，可以是字符串或包含两个浮点数的元组，默认为{default}，表示rc配置中的legend.loc
loc : str or pair of floats, default: {default}
    The location of the legend.

    The strings ``'upper left'``, ``'upper right'``, ``'lower left'``,
    ``'lower right'`` place the legend at the corresponding corner of the
    {parent}.

    The strings ``'upper center'``, ``'lower center'``, ``'center left'``,
    ``'center right'`` place the legend at the center of the corresponding edge
    of the {parent}.

    The string ``'center'`` places the legend at the center of the {parent}.
{best}
    The location can also be a 2-tuple giving the coordinates of the lower-left
    corner of the legend in {parent} coordinates (in which case *bbox_to_anchor*
    will be ignored).

    For back-compatibility, ``'center right'`` (but no other location) can also
    be spelled ``'right'``, and each "string" location can also be given as a
    numeric value:

    ==================   =============
    Location String      Location Code
    ==================   =============
    'best' (Axes only)   0
    'upper right'        1
    'upper left'         2
    'lower left'         3
    'lower right'        4
    'right'              5
    'center left'        6
    'center right'       7
    'lower center'       8
    'upper center'       9
    'center'             10
    ==================   =============
    {outside}"""

_loc_doc_best = """
    The string ``'best'`` places the legend at the location, among the nine
    locations defined so far, with the minimum overlap with other drawn
    artists.  This option can be quite slow for plots with large amounts of
    data; your plotting speed may benefit from providing a specific location.
"""

# 将_loc_doc_base的内容添加到_legend_kw_doc_base中
_legend_kw_axes_st = (
    _loc_doc_base.format(parent='axes', default=':rc:`legend.loc`',
                         best=_loc_doc_best, outside='') +
    _legend_kw_doc_base)
# 更新文档插值的字典，用于将_loc_doc_base注入_legend_kw_axes_st
_docstring.interpd.update(_legend_kw_axes=_legend_kw_axes_st)

# _outside_doc的内容表示了在有限空间布局管理器中，使用字符串代码的可能性
_outside_doc = """
    If a figure is using the constrained layout manager, the string codes
    # The *loc* keyword argument specifies the location of the legend within the plot.
    # By using the prefix 'outside', the legend's layout behavior improves.
    # When using 'outside upper right', it reserves space for the legend above all other axes in the layout,
    # and 'outside right upper' reserves space on the right side of the layout.
    # Additionally, other combinations include 'outside right lower', 'outside left upper', and 'outside left lower'.
    # For more detailed information, refer to the legend guide in the documentation.
"""

_legend_kw_figure_st = (
    _loc_doc_base.format(parent='figure', default="'upper right'",
                         best='', outside=_outside_doc) +
    _legend_kw_doc_base)
# 更新文档字符串的插值字典，用于图例关键字参数说明
_docstring.interpd.update(_legend_kw_figure=_legend_kw_figure_st)

_legend_kw_both_st = (
    _loc_doc_base.format(parent='axes/figure',
                         default=":rc:`legend.loc` for Axes, 'upper right' for Figure",
                         best=_loc_doc_best, outside=_outside_doc) +
    _legend_kw_doc_base)
# 更新文档字符串的插值字典，用于同时适用于 Axes 和 Figure 的图例关键字参数说明
_docstring.interpd.update(_legend_kw_doc=_legend_kw_both_st)

_legend_kw_set_loc_st = (
    _loc_doc_base.format(parent='axes/figure',
                         default=":rc:`legend.loc` for Axes, 'upper right' for Figure",
                         best=_loc_doc_best, outside=_outside_doc))
# 更新文档字符串的插值字典，用于设置图例位置的关键字参数说明
_docstring.interpd.update(_legend_kw_set_loc_doc=_legend_kw_set_loc_st)


class Legend(Artist):
    """
    Place a legend on the figure/axes.
    """

    # 'best' is only implemented for Axes legends
    # 定义代码映射字典，包括 'best' 作为 Axes 图例的特定代码
    codes = {'best': 0, **AnchoredOffsetbox.codes}
    # 设置图例的层叠顺序
    zorder = 5

    def __str__(self):
        # 返回类的字符串表示为 "Legend"
        return "Legend"

    @_docstring.dedent_interpd
    # 初始化函数，用于创建图例对象
    def __init__(
        self, parent, handles, labels,
        *,
        loc=None,  # 图例位置，默认为 None
        numpoints=None,  # 图例线上的点数
        markerscale=None,  # 图例标记相对于原始大小的比例
        markerfirst=True,  # 图例标记和标签的左/右排序
        reverse=False,  # 图例标记和标签的顺序是否反向
        scatterpoints=None,  # 散点图中散点的数量
        scatteryoffsets=None,
        prop=None,  # 图例文本的属性
        fontsize=None,  # 直接设置字体大小的关键字
        labelcolor=None,  # 设置文本颜色的关键字

        # 空白和填充定义为字体大小的一部分
        borderpad=None,  # 图例边框内部的空白
        labelspacing=None,  # 图例条目之间的垂直间距
        handlelength=None,  # 图例标记的长度
        handleheight=None,  # 图例标记的高度
        handletextpad=None,  # 图例标记和文本之间的填充
        borderaxespad=None,  # 图例边框和 Axes 之间的填充
        columnspacing=None,  # 列之间的间距

        ncols=1,  # 列数
        mode=None,  # 列的水平分布方式: None 或 "expand"

        fancybox=None,  # True: 使用华丽的框，False: 使用圆角框，None: 使用 rcParam
        shadow=None,  # 是否显示阴影
        title=None,  # 图例标题
        title_fontsize=None,  # 图例标题的字体大小
        framealpha=None,  # 设置框架的透明度
        edgecolor=None,  # 框架补丁的边缘颜色
        facecolor=None,  # 框架补丁的填充颜色

        bbox_to_anchor=None,  # 图例锚定的边界框
        bbox_transform=None,  # 边界框的变换
        frameon=None,  # 是否绘制框架
        handler_map=None,  # 处理程序映射
        title_fontproperties=None,  # 图例标题的属性
        alignment="center",  # 控制图例框内的文本对齐方式
        ncol=1,  # 列数的同义词（向后兼容）
        draggable=False  # 是否可以使用鼠标拖动图例
    ):
        """
        Set the boilerplate props for artists added to Axes.
        """
        # 设置艺术家添加到 Axes 的基本属性
        a.set_figure(self.figure)
        if self.isaxes:
            # 如果是 Axes，设置其属性
            a.axes = self.axes

        # 设置艺术家的变换
        a.set_transform(self.get_transform())

    # 从文档字符串中去除缩进和空格
    @_docstring.dedent_interpd
    def set_loc(self, loc=None):
        """
        Set the location of the legend.

        .. versionadded:: 3.8

        Parameters
        ----------
        %(_legend_kw_set_loc_doc)s
        """
        loc0 = loc  # 将 loc 参数保存到 loc0 变量中
        self._loc_used_default = loc is None  # 检查 loc 是否为 None，并将结果保存在 _loc_used_default 属性中
        if loc is None:
            loc = mpl.rcParams["legend.loc"]  # 如果 loc 为 None，则使用默认的图例位置配置
            if not self.isaxes and loc in [0, 'best']:
                loc = 'upper right'

        type_err_message = ("loc must be string, coordinate tuple, or"
                            f" an integer 0-10, not {loc!r}")

        # 处理外部图例：
        self._outside_loc = None  # 初始化外部图例位置为 None
        if isinstance(loc, str):
            if loc.split()[0] == 'outside':
                # 去除字符串开头的 "outside" 关键字
                loc = loc.split('outside ')[1]
                # 去除开头的 "center"
                self._outside_loc = loc.replace('center ', '')
                # 再次去除字符串开头的其他描述
                self._outside_loc = self._outside_loc.split()[0]
                locs = loc.split()
                if len(locs) > 1 and locs[0] in ('right', 'left'):
                    # locs 不接受 "left upper" 等，所以交换位置
                    if locs[0] != 'center':
                        locs = locs[::-1]
                    loc = locs[0] + ' ' + locs[1]
            # 检查 loc 是否为可接受的字符串
            loc = _api.check_getitem(self.codes, loc=loc)
        elif np.iterable(loc):
            # 将可迭代对象强制转换为元组
            loc = tuple(loc)
            # 验证元组是否表示实数坐标
            if len(loc) != 2 or not all(isinstance(e, numbers.Real) for e in loc):
                raise ValueError(type_err_message)
        elif isinstance(loc, int):
            # 验证整数是否表示为字符串数值
            if loc < 0 or loc > 10:
                raise ValueError(type_err_message)
        else:
            # 所有其他情况均为 loc 的无效值
            raise ValueError(type_err_message)

        if self.isaxes and self._outside_loc:
            raise ValueError(
                f"'outside' option for loc='{loc0}' keyword argument only "
                "works for figure legends")

        if not self.isaxes and loc == 0:
            raise ValueError(
                "Automatic legend placement (loc='best') not implemented for "
                "figure legend")

        tmp = self._loc_used_default
        self._set_loc(loc)
        self._loc_used_default = tmp  # 忽略 _set_loc 所做的更改

    def _set_loc(self, loc):
        # find_offset 函数将被提供给 _legend_box，并且 _legend_box 将在返回值的位置绘制自身
        self._loc_used_default = False  # 将 _loc_used_default 属性设置为 False
        self._loc_real = loc  # 将 loc 参数保存在 _loc_real 属性中
        self.stale = True  # 设置 stale 属性为 True，标记需要更新
        self._legend_box.set_offset(self._findoffset)  # 设置 _legend_box 的偏移量为 _findoffset 的返回值

    def set_ncols(self, ncols):
        """Set the number of columns."""
        self._ncols = ncols  # 设置 _ncols 属性为给定的 ncols 参数值
    def _get_loc(self):
        return self._loc_real


# 返回当前对象的位置属性 _loc_real
def _get_loc(self):
    return self._loc_real



    _loc = property(_get_loc, _set_loc)


# 定义 _loc 属性，使用 _get_loc 方法获取值，_set_loc 方法设置值
_loc = property(_get_loc, _set_loc)



    def _findoffset(self, width, height, xdescent, ydescent, renderer):
        """Helper function to locate the legend."""

        if self._loc == 0:  # "best".
            x, y = self._find_best_position(width, height, renderer)
        elif self._loc in Legend.codes.values():  # Fixed location.
            bbox = Bbox.from_bounds(0, 0, width, height)
            x, y = self._get_anchored_bbox(self._loc, bbox,
                                           self.get_bbox_to_anchor(),
                                           renderer)
        else:  # Axes or figure coordinates.
            fx, fy = self._loc
            bbox = self.get_bbox_to_anchor()
            x, y = bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy

        return x + xdescent, y + ydescent


# 帮助函数，用于定位图例的位置
def _findoffset(self, width, height, xdescent, ydescent, renderer):
    """Helper function to locate the legend."""

    if self._loc == 0:  # "best".
        # 如果位置为 "best"，调用 _find_best_position 方法寻找最佳位置
        x, y = self._find_best_position(width, height, renderer)
    elif self._loc in Legend.codes.values():  # Fixed location.
        # 如果位置在 Legend.codes.values() 中固定位置的列表中，计算锚定框的位置
        bbox = Bbox.from_bounds(0, 0, width, height)
        x, y = self._get_anchored_bbox(self._loc, bbox,
                                       self.get_bbox_to_anchor(),
                                       renderer)
    else:  # Axes or figure coordinates.
        # 否则，假定是以坐标轴或图形坐标系为基础的位置
        fx, fy = self._loc
        bbox = self.get_bbox_to_anchor()
        x, y = bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy

    # 返回偏移后的坐标
    return x + xdescent, y + ydescent



    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return

        renderer.open_group('legend', gid=self.get_gid())

        fontsize = renderer.points_to_pixels(self._fontsize)

        # if mode == fill, set the width of the legend_box to the
        # width of the parent (minus pads)
        if self._mode in ["expand"]:
            pad = 2 * (self.borderaxespad + self.borderpad) * fontsize
            self._legend_box.set_width(self.get_bbox_to_anchor().width - pad)

        # update the location and size of the legend. This needs to
        # be done in any case to clip the figure right.
        bbox = self._legend_box.get_window_extent(renderer)
        self.legendPatch.set_bounds(bbox.bounds)
        self.legendPatch.set_mutation_scale(fontsize)

        # self.shadow is validated in __init__
        # So by here it is a bool and self._shadow_props contains any configs

        if self.shadow:
            # 如果开启阴影，使用 _shadow_props 配置绘制阴影
            Shadow(self.legendPatch, **self._shadow_props).draw(renderer)

        # 绘制 legendPatch
        self.legendPatch.draw(renderer)
        # 绘制 _legend_box
        self._legend_box.draw(renderer)

        renderer.close_group('legend')
        self.stale = False


@allow_rasterization
def draw(self, renderer):
    # docstring inherited
    if not self.get_visible():
        return

    # 打开名为 'legend' 的绘图组
    renderer.open_group('legend', gid=self.get_gid())

    # 计算字体大小
    fontsize = renderer.points_to_pixels(self._fontsize)

    # 如果模式为 'expand'，设置 legend_box 的宽度为父元素的宽度减去 padding
    if self._mode in ["expand"]:
        pad = 2 * (self.borderaxespad + self.borderpad) * fontsize
        self._legend_box.set_width(self.get_bbox_to_anchor().width - pad)

    # 更新图例的位置和大小，确保正确裁剪图形
    bbox = self._legend_box.get_window_extent(renderer)
    self.legendPatch.set_bounds(bbox.bounds)
    self.legendPatch.set_mutation_scale(fontsize)

    # 检查阴影是否开启
    if self.shadow:
        # 如果开启阴影，使用 _shadow_props 配置绘制阴影
        Shadow(self.legendPatch, **self._shadow_props).draw(renderer)

    # 绘制 legendPatch
    self.legendPatch.draw(renderer)
    # 绘制 _legend_box
    self._legend_box.draw(renderer)

    # 关闭 'legend' 绘图组
    renderer.close_group('legend')
    self.stale = False



    # _default_handler_map defines the default mapping between plot
    # elements and the legend handlers.


# _default_handler_map 定义了绘图元素与图例处理程序之间的默认映射关系
_default_handler_map defines the default mapping between plot
elements and the legend handlers.
    # 定义默认的处理器映射表，将每种图形对象映射到对应的处理器类实例
    _default_handler_map = {
        StemContainer: legend_handler.HandlerStem(),
        ErrorbarContainer: legend_handler.HandlerErrorbar(),
        Line2D: legend_handler.HandlerLine2D(),
        Patch: legend_handler.HandlerPatch(),
        StepPatch: legend_handler.HandlerStepPatch(),
        LineCollection: legend_handler.HandlerLineCollection(),
        RegularPolyCollection: legend_handler.HandlerRegularPolyCollection(),
        CircleCollection: legend_handler.HandlerCircleCollection(),
        BarContainer: legend_handler.HandlerPatch(
            update_func=legend_handler.update_from_first_child),
        tuple: legend_handler.HandlerTuple(),
        PathCollection: legend_handler.HandlerPathCollection(),
        PolyCollection: legend_handler.HandlerPolyCollection()
        }

    # (get|set|update)_default_handler_maps 是公共接口，用于修改默认的处理器映射表。

    @classmethod
    def get_default_handler_map(cls):
        """返回全局默认的处理器映射表，所有图例共享该映射表。"""
        return cls._default_handler_map

    @classmethod
    def set_default_handler_map(cls, handler_map):
        """设置全局默认的处理器映射表，所有图例共享该映射表。"""
        cls._default_handler_map = handler_map

    @classmethod
    def update_default_handler_map(cls, handler_map):
        """更新全局默认的处理器映射表，所有图例共享该映射表。"""
        cls._default_handler_map.update(handler_map)

    def get_legend_handler_map(self):
        """返回此图例实例的处理器映射表。"""
        default_handler_map = self.get_default_handler_map()
        return ({**default_handler_map, **self._custom_handler_map}
                if self._custom_handler_map else default_handler_map)

    @staticmethod
    def get_legend_handler(legend_handler_map, orig_handle):
        """
        从 *legend_handler_map* 中返回与 *orig_handler* 对应的图例处理器。

        *legend_handler_map* 应为由 get_legend_handler_map 方法返回的字典对象。

        首先检查 *orig_handle* 是否是 *legend_handler_map* 中的键，并返回关联的值。
        否则，检查其方法解析顺序中的每个类。如果找不到匹配的键，则返回 ``None``。
        """
        try:
            return legend_handler_map[orig_handle]
        except (TypeError, KeyError):  # 如果无法哈希则抛出 TypeError
            pass
        for handle_type in type(orig_handle).mro():
            try:
                return legend_handler_map[handle_type]
            except KeyError:
                pass
        return None
    def _auto_legend_data(self):
        """
        Return display coordinates for hit testing for "best" positioning.

        Returns
        -------
        bboxes
            List of bounding boxes of all patches.
        lines
            List of `.Path` corresponding to each line.
        offsets
            List of (x, y) offsets of all collection.
        """
        assert self.isaxes  # always holds, as this is only called internally
        # 初始化空列表，用于存储不同类型艺术元素的边界框、路径和偏移量
        bboxes = []
        lines = []
        offsets = []
        # 遍历父对象的所有子对象
        for artist in self.parent._children:
            if isinstance(artist, Line2D):
                # 如果是线条对象，则将其路径进行坐标变换并加入列表
                lines.append(
                    artist.get_transform().transform_path(artist.get_path()))
            elif isinstance(artist, Rectangle):
                # 如果是矩形对象，则获取其边界框并进行数据变换后加入列表
                bboxes.append(
                    artist.get_bbox().transformed(artist.get_data_transform()))
            elif isinstance(artist, Patch):
                # 如果是补丁对象，则将其路径进行坐标变换并加入列表
                lines.append(
                    artist.get_transform().transform_path(artist.get_path()))
            elif isinstance(artist, PolyCollection):
                # 如果是多边形集合对象，则将每个路径进行坐标变换后加入列表
                lines.extend(artist.get_transform().transform_path(path)
                             for path in artist.get_paths())
            elif isinstance(artist, Collection):
                # 如果是集合对象，则准备其点的变换和偏移量，将偏移量进行坐标变换后加入列表
                transform, transOffset, hoffsets, _ = artist._prepare_points()
                if len(hoffsets):
                    offsets.extend(transOffset.transform(hoffsets))
            elif isinstance(artist, Text):
                # 如果是文本对象，则获取其窗口范围并加入边界框列表
                bboxes.append(artist.get_window_extent())

        # 返回收集到的边界框、路径和偏移量列表
        return bboxes, lines, offsets

    def get_children(self):
        # docstring inherited
        # 返回父对象的两个子对象列表作为子对象
        return [self._legend_box, self.get_frame()]

    def get_frame(self):
        """Return the `~.patches.Rectangle` used to frame the legend."""
        # 返回用于框架图例的矩形对象
        return self.legendPatch

    def get_lines(self):
        r"""Return the list of `~.lines.Line2D`\s in the legend."""
        # 返回图例中所有 `Line2D` 类型的线条对象列表
        return [h for h in self.legend_handles if isinstance(h, Line2D)]

    def get_patches(self):
        r"""Return the list of `~.patches.Patch`\s in the legend."""
        # 返回图例中所有 `Patch` 类型的补丁对象列表
        return silent_list('Patch',
                           [h for h in self.legend_handles
                            if isinstance(h, Patch)])

    def get_texts(self):
        r"""Return the list of `~.text.Text`\s in the legend."""
        # 返回图例中所有 `Text` 类型的文本对象列表
        return silent_list('Text', self.texts)

    def set_alignment(self, alignment):
        """
        Set the alignment of the legend title and the box of entries.

        The entries are aligned as a single block, so that markers always
        lined up.

        Parameters
        ----------
        alignment : {'center', 'left', 'right'}.

        """
        # 检查对齐参数是否合法
        _api.check_in_list(["center", "left", "right"], alignment=alignment)
        # 设置图例标题和条目框的对齐方式
        self._alignment = alignment
        self._legend_box.align = alignment

    def get_alignment(self):
        """Get the alignment value of the legend box"""
        # 返回图例框的对齐方式值
        return self._legend_box.align
    def set_title(self, title, prop=None):
        """
        Set legend title and title style.

        Parameters
        ----------
        title : str
            The legend title.

        prop : `.font_manager.FontProperties` or `str` or `pathlib.Path`
            The font properties of the legend title.
            If a `str`, it is interpreted as a fontconfig pattern parsed by
            `.FontProperties`.  If a `pathlib.Path`, it is interpreted as the
            absolute path to a font file.

        """
        # 设置图例标题文本内容为给定的 title
        self._legend_title_box._text.set_text(title)
        # 如果 title 不为空字符串
        if title:
            # 设置图例标题文本可见
            self._legend_title_box._text.set_visible(True)
            # 设置图例标题框可见
            self._legend_title_box.set_visible(True)
        else:
            # 设置图例标题文本不可见
            self._legend_title_box._text.set_visible(False)
            # 设置图例标题框不可见
            self._legend_title_box.set_visible(False)

        # 如果 prop 参数不为 None
        if prop is not None:
            # 设置图例标题文本的字体属性
            self._legend_title_box._text.set_fontproperties(prop)

        # 将对象标记为过时，需要重新绘制
        self.stale = True

    def get_title(self):
        """Return the `.Text` instance for the legend title."""
        # 返回图例标题文本对象
        return self._legend_title_box._text

    def get_window_extent(self, renderer=None):
        # docstring inherited
        # 如果 renderer 参数为 None，则使用 figure 的 renderer
        if renderer is None:
            renderer = self.figure._get_renderer()
        # 返回图例框架的窗口区域边界
        return self._legend_box.get_window_extent(renderer=renderer)

    def get_tightbbox(self, renderer=None):
        # docstring inherited
        # 返回图例框架的紧凑边界框
        return self._legend_box.get_window_extent(renderer)

    def get_frame_on(self):
        """Get whether the legend box patch is drawn."""
        # 返回图例框是否绘制边框补丁
        return self.legendPatch.get_visible()

    def set_frame_on(self, b):
        """
        Set whether the legend box patch is drawn.

        Parameters
        ----------
        b : bool
            Whether to draw the legend box patch.
        """
        # 设置图例框是否绘制边框补丁
        self.legendPatch.set_visible(b)
        # 将对象标记为过时，需要重新绘制
        self.stale = True

    draw_frame = set_frame_on  # Backcompat alias.

    def get_bbox_to_anchor(self):
        """Return the bbox that the legend will be anchored to."""
        # 如果 _bbox_to_anchor 属性为 None，则返回父对象的 bbox
        if self._bbox_to_anchor is None:
            return self.parent.bbox
        else:
            # 否则返回 _bbox_to_anchor 属性指定的边界框
            return self._bbox_to_anchor
    def set_bbox_to_anchor(self, bbox, transform=None):
        """
        Set the bbox that the legend will be anchored to.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.BboxBase` or tuple
            The bounding box can be specified in the following ways:

            - A `.BboxBase` instance
            - A tuple of ``(left, bottom, width, height)`` in the given
              transform (normalized axes coordinate if None)
            - A tuple of ``(left, bottom)`` where the width and height will be
              assumed to be zero.
            - *None*, to remove the bbox anchoring, and use the parent bbox.

        transform : `~matplotlib.transforms.Transform`, optional
            A transform to apply to the bounding box. If not specified, this
            will use a transform to the bounding box of the parent.
        """
        # 如果 bbox 为 None，则将 _bbox_to_anchor 设置为 None，并返回
        if bbox is None:
            self._bbox_to_anchor = None
            return
        # 如果 bbox 是 BboxBase 的实例，则直接将 _bbox_to_anchor 设置为 bbox
        elif isinstance(bbox, BboxBase):
            self._bbox_to_anchor = bbox
        else:
            # 尝试获取 bbox 的长度，如果出错则抛出 ValueError
            try:
                l = len(bbox)
            except TypeError as err:
                raise ValueError(f"Invalid bbox: {bbox}") from err

            # 如果 bbox 的长度为 2，则将其转换为包含宽度和高度的列表
            if l == 2:
                bbox = [bbox[0], bbox[1], 0, 0]

            # 使用 from_bounds 方法创建 Bbox 对象
            self._bbox_to_anchor = Bbox.from_bounds(*bbox)

        # 如果 transform 为 None，则使用 BboxTransformTo(self.parent.bbox)
        if transform is None:
            transform = BboxTransformTo(self.parent.bbox)

        # 使用 TransformedBbox 将 _bbox_to_anchor 转换为指定 transform 后的 Bbox
        self._bbox_to_anchor = TransformedBbox(self._bbox_to_anchor,
                                               transform)
        # 设置 stale 属性为 True，表示需要重新绘制
        self.stale = True

    def _get_anchored_bbox(self, loc, bbox, parentbbox, renderer):
        """
        Place the *bbox* inside the *parentbbox* according to a given
        location code. Return the (x, y) coordinate of the bbox.

        Parameters
        ----------
        loc : int
            A location code in range(1, 11). This corresponds to the possible
            values for ``self._loc``, excluding "best".
        bbox : `~matplotlib.transforms.Bbox`
            bbox to be placed, in display coordinates.
        parentbbox : `~matplotlib.transforms.Bbox`
            A parent box which will contain the bbox, in display coordinates.
        """
        # 调用 offsetbox._get_anchored_bbox 方法，将 loc、bbox、parentbbox 和计算后的偏移值传递进去
        return offsetbox._get_anchored_bbox(
            loc, bbox, parentbbox,
            self.borderaxespad * renderer.points_to_pixels(self._fontsize))
    def _find_best_position(self, width, height, renderer):
        """Determine the best location to place the legend."""
        assert self.isaxes  # 始终成立，因为这仅在内部调用时使用

        start_time = time.perf_counter()  # 记录开始时间以计算函数执行时间

        bboxes, lines, offsets = self._auto_legend_data()  # 获取自动计算的图例数据

        bbox = Bbox.from_bounds(0, 0, width, height)  # 创建一个表示给定宽度和高度的边界框对象

        candidates = []
        for idx in range(1, len(self.codes)):
            l, b = self._get_anchored_bbox(idx, bbox,
                                           self.get_bbox_to_anchor(),
                                           renderer)
            legendBox = Bbox.from_bounds(l, b, width, height)  # 根据位置和大小创建一个图例框对象
            # XXX TODO: If markers are present, it would be good to take them
            # into account when checking vertex overlaps in the next line.
            # 计算图例框的不良度，考虑顶点重叠和边界框相交
            badness = (sum(legendBox.count_contains(line.vertices)
                           for line in lines)
                       + legendBox.count_contains(offsets)
                       + legendBox.count_overlaps(bboxes)
                       + sum(line.intersects_bbox(legendBox, filled=False)
                             for line in lines))
            # 将索引包含在内，以便在平局情况下偏好较低的代码
            candidates.append((badness, idx, (l, b)))
            if badness == 0:
                break

        _, _, (l, b) = min(candidates)  # 选择不良度最低的候选图例框

        if self._loc_used_default and time.perf_counter() - start_time > 1:
            _api.warn_external(
                'Creating legend with loc="best" can be slow with large '
                'amounts of data.')  # 如果使用默认位置并且执行时间超过1秒，发出警告

        return l, b  # 返回选择的最佳位置的左上角坐标

    @_api.rename_parameter("3.8", "event", "mouseevent")
    def contains(self, mouseevent):
        return self.legendPatch.contains(mouseevent)  # 检查图例区域是否包含鼠标事件
    def set_draggable(self, state, use_blit=False, update='loc'):
        """
        Enable or disable mouse dragging support of the legend.

        Parameters
        ----------
        state : bool
            Whether mouse dragging is enabled.
        use_blit : bool, optional
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.
        update : {'loc', 'bbox'}, optional
            The legend parameter to be changed when dragged:

            - 'loc': update the *loc* parameter of the legend
            - 'bbox': update the *bbox_to_anchor* parameter of the legend

        Returns
        -------
        `.DraggableLegend` or *None*
            If *state* is ``True`` this returns the `.DraggableLegend` helper
            instance. Otherwise this returns *None*.
        """
        # 如果状态为真（即要启用拖动功能）
        if state:
            # 如果当前没有创建拖动实例
            if self._draggable is None:
                # 创建一个 DraggableLegend 实例，并赋值给 self._draggable
                self._draggable = DraggableLegend(self,
                                                  use_blit,
                                                  update=update)
        else:
            # 如果状态为假（即要禁用拖动功能）
            if self._draggable is not None:
                # 断开当前拖动实例的连接
                self._draggable.disconnect()
            # 将 self._draggable 置为 None
            self._draggable = None
        # 返回当前 self._draggable 的状态，可能是 DraggableLegend 实例或 None
        return self._draggable

    def get_draggable(self):
        """Return ``True`` if the legend is draggable, ``False`` otherwise."""
        # 返回判断 self._draggable 是否为 None 的结果，表明是否可拖动
        return self._draggable is not None
# Helper functions to parse legend arguments for both `figure.legend` and
# `axes.legend`:
def _get_legend_handles(axs, legend_handler_map=None):
    """Yield artists that can be used as handles in a legend."""
    # 初始化空列表，用于存储所有图例的句柄
    handles_original = []
    # 遍历每个 Axes 对象
    for ax in axs:
        # 将该 Axes 中的 Line2D、Patch、Collection 和 Text 类型的子对象添加到句柄列表中
        handles_original += [
            *(a for a in ax._children
              if isinstance(a, (Line2D, Patch, Collection, Text))),
            *ax.containers]
        # 支持寄生 Axes：
        if hasattr(ax, 'parasites'):
            # 遍历寄生 Axes 中的每个子 Axes
            for axx in ax.parasites:
                # 将每个寄生 Axes 中的 Line2D、Patch、Collection 和 Text 类型的子对象添加到句柄列表中
                handles_original += [
                    *(a for a in axx._children
                      if isinstance(a, (Line2D, Patch, Collection, Text))),
                    *axx.containers]

    # 获取默认的图例处理映射
    handler_map = {**Legend.get_default_handler_map(),
                   **(legend_handler_map or {})}
    # 获取是否有自定义的图例处理函数
    has_handler = Legend.get_legend_handler
    # 遍历所有原始句柄
    for handle in handles_original:
        # 获取句柄的标签
        label = handle.get_label()
        # 如果标签不是 '_nolegend_' 并且有对应的处理函数，则生成该句柄
        if label != '_nolegend_' and has_handler(handler_map, handle):
            yield handle
        # 如果标签存在且不以下划线开头，并且没有对应的处理函数，则发出警告
        elif (label and not label.startswith('_') and
                not has_handler(handler_map, handle)):
            _api.warn_external(
                             "Legend does not support handles for "
                             f"{type(handle).__name__} "
                             "instances.\nSee: https://matplotlib.org/stable/"
                             "tutorials/intermediate/legend_guide.html"
                             "#implementing-a-custom-legend-handler")
            continue


def _get_legend_handles_labels(axs, legend_handler_map=None):
    """Return handles and labels for legend."""
    # 初始化空列表，用于存储句柄和标签
    handles = []
    labels = []
    # 遍历通过 _get_legend_handles 函数获取的句柄和标签
    for handle in _get_legend_handles(axs, legend_handler_map):
        # 获取句柄的标签
        label = handle.get_label()
        # 如果标签存在且不以下划线开头，则添加到句柄和标签列表中
        if label and not label.startswith('_'):
            handles.append(handle)
            labels.append(label)
    return handles, labels


def _parse_legend_args(axs, *args, handles=None, labels=None, **kwargs):
    """
    Get the handles and labels from the calls to either ``figure.legend``
    or ``axes.legend``.

    The parser is a bit involved because we support::

        legend()
        legend(labels)
        legend(handles, labels)
        legend(labels=labels)
        legend(handles=handles)
        legend(handles=handles, labels=labels)

    The behavior for a mixture of positional and keyword handles and labels
    is undefined and issues a warning; it will be an error in the future.

    Parameters
    ----------
    axs : list of `.Axes`
        If handles are not given explicitly, the artists in these Axes are
        used as handles.
    *args : tuple
        Positional parameters passed to ``legend()``.
    handles
        The value of the keyword argument ``legend(handles=...)``, or *None*
        if that keyword argument was not used.
    """
    def legend(*args, labels=None, **kwargs):
        """
        Parameters
        ----------
        handles
            The value of the keyword argument ``legend(handles=...)``, or *None*
            if that keyword argument was not used.
        labels
            The value of the keyword argument ``legend(labels=...)``, or *None*
            if that keyword argument was not used.
        **kwargs
            All other keyword arguments passed to ``legend()``.
    
        Returns
        -------
        handles : list of (`.Artist` or tuple of `.Artist`)
            The legend handles.
        labels : list of str
            The legend labels.
        kwargs : dict
            *kwargs* with keywords handles and labels removed.
        """
        # 获取全局日志记录器
        log = logging.getLogger(__name__)
    
        # 获取关键字参数中的 handler_map 值
        handlers = kwargs.get('handler_map')
    
        # 检查是否同时传入了位置参数和关键字参数 handles 或 labels
        if (handles is not None or labels is not None) and args:
            _api.warn_deprecated("3.9", message=(
                "You have mixed positional and keyword arguments, some input may "
                "be discarded.  This is deprecated since %(since)s and will "
                "become an error %(removal)s."))
    
        # 检查 handles 和 labels 是否都有长度，并且长度不相等时发出警告
        if (hasattr(handles, "__len__") and
                hasattr(labels, "__len__") and
                len(handles) != len(labels)):
            _api.warn_external(f"Mismatched number of handles and labels: "
                               f"len(handles) = {len(handles)} "
                               f"len(labels) = {len(labels)}")
    
        # 如果同时存在 handles 和 labels，则将它们组合成对
        if handles and labels:
            handles, labels = zip(*zip(handles, labels))
    
        # 如果仅有 handles 而没有 labels，则根据 handles 获取各自的标签
        elif handles is not None and labels is None:
            labels = [handle.get_label() for handle in handles]
    
        # 如果仅有 labels 而没有 handles，则根据 labels 获取对应的 handles
        elif labels is not None and handles is None:
            # 获取与 labels 数量相匹配的 handles
            handles = [handle for handle, label
                       in zip(_get_legend_handles(axs, handlers), labels)]
    
        # 当未传入任何位置参数时，自动检测并获取 labels 和 handles
        elif len(args) == 0:
            handles, labels = _get_legend_handles_labels(axs, handlers)
            if not handles:
                _api.warn_external(
                    "No artists with labels found to put in legend.  Note that "
                    "artists whose label start with an underscore are ignored "
                    "when legend() is called with no argument.")
    
        # 当传入一个位置参数时，用户自定义 labels，自动检测 handles
        elif len(args) == 1:
            labels, = args
            if any(isinstance(l, Artist) for l in labels):
                raise TypeError("A single argument passed to legend() must be a "
                                "list of labels, but found an Artist in there.")
            # 获取与 labels 数量相匹配的 handles
            handles = [handle for handle, label
                       in zip(_get_legend_handles(axs, handlers), labels)]
    
        # 当传入两个位置参数时，用户自定义 handles 和 labels
        elif len(args) == 2:
            handles, labels = args[:2]
    
        # 如果参数数量不在 0 到 2 之间，则引发异常
        else:
            raise _api.nargs_error('legend', '0-2', len(args))
    
        return handles, labels, kwargs
```