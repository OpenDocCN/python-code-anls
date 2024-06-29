# `D:\src\scipysrc\matplotlib\lib\matplotlib\lines.py`

```py
"""
2D lines with support for a variety of line styles, markers, colors, etc.
"""

# 引入必要的模块和库
import copy  # 用于深拷贝对象

from numbers import Integral, Number, Real  # 引入数值类型相关的模块
import logging  # 引入日志记录模块

import numpy as np  # 引入 NumPy 库

import matplotlib as mpl  # 引入 Matplotlib 主库
from . import _api, cbook, colors as mcolors, _docstring  # 导入自定义的模块和函数
from .artist import Artist, allow_rasterization  # 引入 Artist 类和允许栅格化的功能
from .cbook import (  # 引入自定义的 cbook 模块中的函数和常量
    _to_unmasked_float_array, ls_mapper, ls_mapper_r, STEP_LOOKUP_MAP)
from .markers import MarkerStyle  # 引入 MarkerStyle 类
from .path import Path  # 引入 Path 类
from .transforms import Bbox, BboxTransformTo, TransformedPath  # 引入变换相关的类
from ._enums import JoinStyle, CapStyle  # 引入枚举类型 JoinStyle 和 CapStyle

# 为了向后兼容性而导入，尽管它们不是真正属于这里的模块
from . import _path  # 导入 _path 模块
from .markers import (  # 导入标记相关的常量
    CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
    CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE,
    TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN)

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def _get_dash_pattern(style):
    """Convert linestyle to dash pattern."""
    # 将简写的线型样式转换为完整的虚线模式
    if isinstance(style, str):
        style = ls_mapper.get(style, style)
    # 处理非虚线样式
    if style in ['solid', 'None']:
        offset = 0
        dashes = None
    # 处理虚线样式
    elif style in ['dashed', 'dashdot', 'dotted']:
        offset = 0
        dashes = tuple(mpl.rcParams[f'lines.{style}_pattern'])
    #
    elif isinstance(style, tuple):
        offset, dashes = style
        if offset is None:
            raise ValueError(f'Unrecognized linestyle: {style!r}')
    else:
        raise ValueError(f'Unrecognized linestyle: {style!r}')

    # 将偏移量规范化为正数，并确保它比虚线周期更短
    if dashes is not None:
        dsum = sum(dashes)
        if dsum:
            offset %= dsum

    return offset, dashes


def _get_inverse_dash_pattern(offset, dashes):
    """Return the inverse of the given dash pattern, for filling the gaps."""
    # 根据给定的虚线模式返回其反转模式，以填补间隙
    # 将最后一个间隙移至序列的开头
    gaps = dashes[-1:] + dashes[:-1]
    # 设置偏移量，使得新的第一个片段被跳过
    # （参见 backend_bases.GraphicsContextBase.set_dashes 中对偏移量的定义）
    offset_gaps = offset + dashes[-1]

    return offset_gaps, gaps


def _scale_dashes(offset, dashes, lw):
    """Scale dashes by the line width."""
    # 如果配置允许，按线宽缩放虚线
    if not mpl.rcParams['lines.scale_dashes']:
        return offset, dashes
    scaled_offset = offset * lw
    scaled_dashes = ([x * lw if x is not None else None for x in dashes]
                     if dashes is not None else None)
    return scaled_offset, scaled_dashes


def segment_hits(cx, cy, x, y, radius):
    """
    Return the indices of the segments in the polyline with coordinates (*cx*,
    *cy*) that are within a distance *radius* of the point (*x*, *y*).
    """
    # 返回多边形线段中与点 (*x*, *y*) 距离不超过 *radius* 的线段的索引
    # 对于单个点进行特殊处理
    if len(x) <= 1:
        res, = np.nonzero((cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2)
        return res

    # 我们需要经常截断最后一个元素。
    xr, yr = x[:-1], y[:-1]
    # 仅考虑最接近点C的线段，其最近点在该线段内部。
    dx, dy = x[1:] - xr, y[1:] - yr
    Lnorm_sq = dx ** 2 + dy ** 2  # 可能希望排除 Lnorm==0 的情况
    u = ((cx - xr) * dx + (cy - yr) * dy) / Lnorm_sq
    candidates = (u >= 0) & (u <= 1)
    
    # 注意，每个点周围都有一个小区域
    # 在两个线段附近，具体取决于线段的角度。
    # 下面的半径测试消除了这些模糊性。
    point_hits = (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2
    candidates = candidates & ~(point_hits[:-1] | point_hits[1:])
    
    # 对于剩余的候选点，确定它们离直线的距离。
    px, py = xr + u * dx, yr + u * dy
    line_hits = (cx - px) ** 2 + (cy - py) ** 2 <= radius ** 2
    line_hits = line_hits & candidates
    points, = point_hits.ravel().nonzero()
    lines, = line_hits.ravel().nonzero()
    return np.concatenate((points, lines))
def _mark_every_path(markevery, tpath, affine, ax):
    """
    Helper function that sorts out how to deal the input
    `markevery` and returns the points where markers should be drawn.

    Takes in the `markevery` value and the line path and returns the
    sub-sampled path.
    """
    # 从路径对象 `tpath` 中分离出我们需要的两部分数据：`codes` 和 `verts`
    codes, verts = tpath.codes, tpath.vertices

    def _slice_or_none(in_v, slc):
        """Helper function to cope with `codes` being an ndarray or `None`."""
        # 如果输入 `in_v` 是 `None`，直接返回 `None`，否则返回 `in_v` 的切片 `slc`
        if in_v is None:
            return None
        return in_v[slc]

    # 如果 `markevery` 是整数类型，假定从 0 开始，转换为起始点为 0 的元组
    if isinstance(markevery, Integral):
        markevery = (0, markevery)
    # 如果 `markevery` 是浮点数类型，假定从 0.0 开始，转换为起始点为 0.0 的元组
    elif isinstance(markevery, Real):
        markevery = (0.0, markevery)
    # 如果 `markevery` 是一个元组
    if isinstance(markevery, tuple):
        # 检查元组长度是否为2，否则抛出异常
        if len(markevery) != 2:
            raise ValueError('`markevery` is a tuple but its len is not 2; '
                             f'markevery={markevery}')
        # 解构元组，获取 start 和 step
        start, step = markevery

        # 如果 step 是整数，保持旧的行为
        if isinstance(step, Integral):
            # 如果 start 不是整数，抛出异常
            if not isinstance(start, Integral):
                raise ValueError(
                    '`markevery` is a tuple with len 2 and second element is '
                    'an int, but the first element is not an int; '
                    f'markevery={markevery}')
            # 直接返回结果，结束函数
            return Path(verts[slice(start, None, step)],
                        _slice_or_none(codes, slice(start, None, step)))

        # 如果 step 是实数
        elif isinstance(step, Real):
            # 如果 start 不是实数，抛出异常
            if not isinstance(start, Real):
                raise ValueError(
                    '`markevery` is a tuple with len 2 and second element is '
                    'a float, but the first element is not a float or an int; '
                    f'markevery={markevery}')
            # 如果 ax 为空，抛出异常
            if ax is None:
                raise ValueError(
                    "markevery is specified relative to the Axes size, but "
                    "the line does not have a Axes as parent")

            # 计算路径的累积距离（在显示坐标系中）
            fin = np.isfinite(verts).all(axis=1)
            fverts = verts[fin]
            disp_coords = affine.transform(fverts)

            # 计算路径上的距离变化
            delta = np.empty((len(disp_coords), 2))
            delta[0, :] = 0
            delta[1:, :] = disp_coords[1:, :] - disp_coords[:-1, :]
            delta = np.hypot(*delta.T).cumsum()

            # 根据 Axes 的边界框对角线的距离单位，计算标记点之间的理论距离
            (x0, y0), (x1, y1) = ax.transAxes.transform([[0, 0], [1, 1]])
            scale = np.hypot(x1 - x0, y1 - y0)
            marker_delta = np.arange(start * scale, delta[-1], step * scale)

            # 找到最接近理论路径距离的实际数据点
            inds = np.abs(delta[np.newaxis, :] - marker_delta[:, np.newaxis])
            inds = inds.argmin(axis=1)
            inds = np.unique(inds)

            # 直接返回结果，结束函数
            return Path(fverts[inds], _slice_or_none(codes, inds))

        # 如果 step 不是整数也不是实数，抛出异常
        else:
            raise ValueError(
                f"markevery={markevery!r} is a tuple with len 2, but its "
                f"second element is not an int or a float")

    # 如果 `markevery` 是一个切片
    elif isinstance(markevery, slice):
        # 直接返回结果，结束函数
        return Path(verts[markevery], _slice_or_none(codes, markevery))
    # 如果 markevery 是可迭代对象，则进行 fancy indexing（高级索引）
    elif np.iterable(markevery):
        try:
            # 使用 fancy indexing 获取指定索引的路径和对应的代码
            return Path(verts[markevery], _slice_or_none(codes, markevery))
        except (ValueError, IndexError) as err:
            # 如果出现值错误或索引错误，抛出新的值错误异常
            raise ValueError(
                f"markevery={markevery!r} is iterable but not a valid numpy "
                f"fancy index") from err
    else:
        # 如果 markevery 既不是整数也不是可迭代对象，抛出值错误异常
        raise ValueError(f"markevery={markevery!r} is not a recognized value")
# 使用文档字符串插值（interpd）装饰器来处理文档字符串的格式化
# 定义别名，将一些参数名映射到其缩写形式，用于简化参数传递
@_docstring.interpd
@_api.define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):
    """
    A line - the line can have both a solid linestyle connecting all
    the vertices, and a marker at each vertex.  Additionally, the
    drawing of the solid line is influenced by the drawstyle, e.g., one
    can create "stepped" lines in various styles.
    """

    # 隐藏的名称，已弃用，用于定义不同线条样式的函数名映射
    lineStyles = _lineStyles = {
        '-':    '_draw_solid',
        '--':   '_draw_dashed',
        '-.':   '_draw_dash_dot',
        ':':    '_draw_dotted',
        'None': '_draw_nothing',
        ' ':    '_draw_nothing',
        '':     '_draw_nothing',
    }

    # 长线条样式的绘制函数名映射
    _drawStyles_l = {
        'default':    '_draw_lines',
        'steps-mid':  '_draw_steps_mid',
        'steps-pre':  '_draw_steps_pre',
        'steps-post': '_draw_steps_post',
    }

    # 短线条样式的绘制函数名映射
    _drawStyles_s = {
        'steps': '_draw_steps_pre',
    }

    # 用于绘制线条的样式映射，包括长线条和短线条的定义
    # 此处提示 drawStyles 应该已经弃用
    drawStyles = {**_drawStyles_l, **_drawStyles_s}
    # 需要按照长名称优先的顺序列出的样式键列表
    drawStyleKeys = [*_drawStyles_l, *_drawStyles_s]

    # 引用这些以保持 API 兼容性，这些定义在 MarkerStyle 中
    # 定义标记、填充标记和填充样式
    markers = MarkerStyle.markers
    filled_markers = MarkerStyle.filled_markers
    fillStyles = MarkerStyle.fillstyles

    # 在绘制层次中的顺序
    zorder = 2

    # 最小子切片优化的最小尺寸
    _subslice_optim_min_size = 1000

    def __str__(self):
        # 如果有标签则返回带标签的描述，否则根据点的数量生成描述字符串
        if self._label != "":
            return f"Line2D({self._label})"
        elif self._x is None:
            return "Line2D()"
        elif len(self._x) > 3:
            return "Line2D(({:g},{:g}),({:g},{:g}),...,({:g},{:g}))".format(
                self._x[0], self._y[0],
                self._x[1], self._y[1],
                self._x[-1], self._y[-1])
        else:
            return "Line2D(%s)" % ",".join(
                map("({:g},{:g})".format, self._x, self._y))
    def contains(self, mouseevent):
        """
        Test whether *mouseevent* occurred on the line.

        An event is deemed to have occurred "on" the line if it is less
        than ``self.pickradius`` (default: 5 points) away from it.  Use
        `~.Line2D.get_pickradius` or `~.Line2D.set_pickradius` to get or set
        the pick radius.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`

        Returns
        -------
        contains : bool
            Whether any values are within the radius.
        details : dict
            A dictionary ``{'ind': pointlist}``, where *pointlist* is a
            list of points of the line that are within the pickradius around
            the event position.

            TODO: sort returned indices by distance
        """
        # 检查事件是否发生在相同的画布上
        if self._different_canvas(mouseevent):
            return False, {}

        # 确保有数据可绘制
        if self._invalidy or self._invalidx:
            # 重新缓存数据
            self.recache()
        if len(self._xy) == 0:
            return False, {}

        # 将点转换为像素
        transformed_path = self._get_transformed_path()
        path, affine = transformed_path.get_transformed_path_and_affine()
        path = affine.transform_path(path)
        xy = path.vertices
        xt = xy[:, 0]
        yt = xy[:, 1]

        # 将选取半径从点转换为像素
        if self.figure is None:
            _log.warning('no figure set when check if mouse is on line')
            pixels = self._pickradius
        else:
            pixels = self.figure.dpi / 72. * self._pickradius

        # 在检查包含性时所涉及的数学运算（此处和segment_hits内部）假定允许溢出，
        # 因此临时设置错误标志。
        with np.errstate(all='ignore'):
            # 检查碰撞
            if self._linestyle in ['None', None]:
                # 如果没有线条，返回附近的点
                ind, = np.nonzero(
                    (xt - mouseevent.x) ** 2 + (yt - mouseevent.y) ** 2
                    <= pixels ** 2)
            else:
                # 如果有线条，返回附近的线段
                ind = segment_hits(mouseevent.x, mouseevent.y, xt, yt, pixels)
                if self._drawstyle.startswith("steps"):
                    ind //= 2

        # 调整索引偏移量
        ind += self.ind_offset

        # 返回在半径内的点
        return len(ind) > 0, dict(ind=ind)

    def get_pickradius(self):
        """
        Return the pick radius used for containment tests.

        See `.contains` for more details.
        """
        # 返回用于包含性测试的选取半径
        return self._pickradius
    def set_pickradius(self, pickradius):
        """
        设置用于包含测试的拾取半径。

        查看 `.contains` 获取更多详情。

        Parameters
        ----------
        pickradius : float
            拾取半径，单位为点（points）。
        """
        # 如果 pickradius 不是实数类型或者小于 0，则抛出数值错误异常
        if not isinstance(pickradius, Real) or pickradius < 0:
            raise ValueError("pick radius should be a distance")
        self._pickradius = pickradius

    pickradius = property(get_pickradius, set_pickradius)

    def get_fillstyle(self):
        """
        返回标记点的填充样式。

        参见 `~.Line2D.set_fillstyle`。
        """
        return self._marker.get_fillstyle()

    def set_fillstyle(self, fs):
        """
        设置标记点的填充样式。

        Parameters
        ----------
        fs : {'full', 'left', 'right', 'bottom', 'top', 'none'}
            可选值:

            - 'full': 使用 *markerfacecolor* 全部填充标记点。
            - 'left', 'right', 'bottom', 'top': 在给定的一侧使用 *markerfacecolor* 填充标记点的一半。
              标记点的另一半使用 *markerfacecoloralt* 填充。
            - 'none': 不填充。

            有关示例，请参阅 :ref:`marker_fill_styles`。
        """
        # 设置标记点的样式为给定的填充样式 fs
        self.set_marker(MarkerStyle(self._marker.get_marker(), fs))
        self.stale = True

    def set_markevery(self, every):
        """
        设置 markevery 属性，以便在使用标记点时对绘图进行子采样。

        例如，如果 ``every=5``，则每隔 5 个标记点将绘制一个。

        Parameters
        ----------
        every : None or int or (int, int) or slice or list[int] or float or \
# 设置用于绘制标记的条件，可以是元组、列表或浮点数
(float, float) or list[bool]
Which markers to plot.

# 每种不同的`every`值对应的解释如下：
- ``every=None``: 每个点都会被绘制。
- ``every=N``: 每隔N个标记中的一个将被绘制，起始标记为0。
- ``every=(start, N)``: 从索引为start开始，每隔N个标记中的一个将被绘制。
- ``every=slice(start, end, N)``: 从索引为start开始，到索引为end之前，每隔N个标记中的一个将被绘制。
- ``every=[i, j, m, ...]``: 只有指定索引处的标记将被绘制。
- ``every=[True, False, True, ...]``: 只有为True的位置将被绘制，列表长度需与数据点相同。
- ``every=0.1``, （即浮点数）: 标记将以大致相等的视觉距离沿线分布；标记间的距离由乘以Axes边界框对角线的显示坐标距离得出。
- ``every=(0.5, 0.1)``（即包含两个浮点数的元组）: 类似于``every=0.1``，但第一个标记将沿线偏移0.5乘以显示坐标对角线距离沿线。

# 查看示例，请参见 /gallery/lines_bars_and_markers/markevery_demo

Notes
-----
# 设置*markevery*仍然只会在实际数据点上绘制标记。
# 虽然浮点数参数形式旨在实现均匀的视觉间距，但必须将其从理想间距强制转换为最接近的可用数据点。
# 根据数据点的数量和分布，结果可能仍然看起来不均匀。

# 使用起始偏移量来指定第一个标记时，偏移量将从第一个数据点开始，这可能与可见数据点不同，如果图表进行了缩放。

# 当使用浮点数参数进行缩放时，实际具有标记的数据点会改变，因为标记之间的距离始终是根据显示坐标轴边界框对角线来确定的，而不考虑实际的轴数据限制。
Setting *markevery* will still only draw markers at actual data points.
While the float argument form aims for uniform visual spacing, it has
to coerce from the ideal spacing to the nearest available data point.
Depending on the number and distribution of data points, the result
may still not look evenly spaced.

When using a start offset to specify the first marker, the offset will
be from the first data point which may be different from the first
the visible data point if the plot is zoomed in.

If zooming in on a plot when using float arguments then the actual
data points that have markers will change because the distance between
markers is always determined from the display-coordinates
axes-bounding-box-diagonal regardless of the actual axes data limits.

"""
self._markevery = every
self.stale = True
    # 设置线条的事件拾取器详情。

    def set_picker(self, p):
        """
        Set the event picker details for the line.

        Parameters
        ----------
        p : float or callable[[Artist, Event], tuple[bool, dict]]
            If a float, it is used as the pick radius in points.
        """
        # 如果 p 不是可调用对象，则将其作为拾取半径设置
        if not callable(p):
            self.set_pickradius(p)
        # 将 p 设置为当前对象的拾取器
        self._picker = p

    # 获取线条的包围框（边界框）。

    def get_bbox(self):
        """Get the bounding box of this line."""
        # 创建一个初始的边界框对象
        bbox = Bbox([[0, 0], [0, 0]])
        # 更新边界框，根据线条的数据点坐标
        bbox.update_from_data_xy(self.get_xydata())
        return bbox

    # 获取线条在渲染器中的窗口范围。

    def get_window_extent(self, renderer=None):
        # 创建一个初始的边界框对象
        bbox = Bbox([[0, 0], [0, 0]])
        # 获取数据到绘图坐标的变换函数
        trans_data_to_xy = self.get_transform().transform
        # 根据线条的数据点坐标，更新边界框
        bbox.update_from_data_xy(trans_data_to_xy(self.get_xydata()),
                                 ignore=True)
        # 如果存在标记点，校正边界框以考虑标记点的大小
        if self._marker:
            ms = (self._markersize / 72.0 * self.figure.dpi) * 0.5
            bbox = bbox.padded(ms)
        return bbox

    # 设置线条的数据（x 和 y 轴数据）。

    def set_data(self, *args):
        """
        Set the x and y data.

        Parameters
        ----------
        *args : (2, N) array or two 1D arrays

        See Also
        --------
        set_xdata
        set_ydata
        """
        # 根据参数个数判断传入的数据形式
        if len(args) == 1:
            (x, y), = args
        else:
            x, y = args

        # 调用相应的方法设置 x 和 y 轴数据
        self.set_xdata(x)
        self.set_ydata(y)

    # 总是重新计算缓存数据。

    def recache_always(self):
        # 调用 recache 方法，传入 always=True
        self.recache(always=True)
    def recache(self, always=False):
        # 如果参数 always 为 True 或者标记 self._invalidx 为 True，则重新计算 x 值
        if always or self._invalidx:
            # 将 self._xorig 转换为数据可处理的 xconv，并将其展平为一维数组 x
            xconv = self.convert_xunits(self._xorig)
            x = _to_unmasked_float_array(xconv).ravel()
        else:
            # 否则直接使用已有的 self._x
            x = self._x

        # 如果参数 always 为 True 或者标记 self._invalidy 为 True，则重新计算 y 值
        if always or self._invalidy:
            # 将 self._yorig 转换为数据可处理的 yconv，并将其展平为一维数组 y
            yconv = self.convert_yunits(self._yorig)
            y = _to_unmasked_float_array(yconv).ravel()
        else:
            # 否则直接使用已有的 self._y
            y = self._y

        # 将 x 和 y 组合成二维数组，并转换为 float 类型
        self._xy = np.column_stack(np.broadcast_arrays(x, y)).astype(float)
        # 将二维数组分别赋值给 self._x 和 self._y，此处是视图操作
        self._x, self._y = self._xy.T  # views

        # 初始化子切片标志为 False
        self._subslice = False
        # 如果满足一系列条件，设置子切片标志为 True
        if (self.axes
                and len(x) > self._subslice_optim_min_size
                and _path.is_sorted_and_has_non_nan(x)
                and self.axes.name == 'rectilinear'
                and self.axes.get_xscale() == 'linear'
                and self._markevery is None
                and self.get_clip_on()
                and self.get_transform() == self.axes.transData):
            self._subslice = True
            # 如果 x 中存在 NaN 值，使用插值法填充 NaN 值
            nanmask = np.isnan(x)
            if nanmask.any():
                self._x_filled = self._x.copy()
                indices = np.arange(len(x))
                self._x_filled[nanmask] = np.interp(
                    indices[nanmask], indices[~nanmask], self._x[~nanmask])
            else:
                # 否则直接复制 self._x 到 self._x_filled
                self._x_filled = self._x

        # 如果存在 self._path，则使用其 _interpolation_steps，否则设为 1
        if self._path is not None:
            interpolation_steps = self._path._interpolation_steps
        else:
            interpolation_steps = 1
        # 根据绘制风格从 STEP_LOOKUP_MAP 中获取对应的 xy 数据
        xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy.T)
        # 根据 xy 数据创建 Path 对象，并存储在 self._path 中
        self._path = Path(np.asarray(xy).T,
                          _interpolation_steps=interpolation_steps)
        # 清空 transformed_path 属性
        self._transformed_path = None
        # 将 self._invalidx 和 self._invalidy 设为 False，表示已有效
        self._invalidx = False
        self._invalidy = False

    def _transform_path(self, subslice=None):
        """
        Put a TransformedPath instance at self._transformed_path;
        all invalidation of the transform is then handled by the
        TransformedPath instance.
        """
        # 处理 subslice 参数，若非 None，则对 self._xy 进行子切片
        if subslice is not None:
            xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy[subslice, :].T)
            # 创建 Path 对象 _path，使用 subslice 切片的 xy 数据
            _path = Path(np.asarray(xy).T,
                         _interpolation_steps=self._path._interpolation_steps)
        else:
            # 否则直接使用 self._path
            _path = self._path
        # 创建 TransformedPath 实例，并存储在 self._transformed_path 中
        self._transformed_path = TransformedPath(_path, self.get_transform())

    def _get_transformed_path(self):
        """Return this line's `~matplotlib.transforms.TransformedPath`."""
        # 如果 self._transformed_path 为 None，则调用 _transform_path 方法进行转换
        if self._transformed_path is None:
            self._transform_path()
        return self._transformed_path

    def set_transform(self, t):
        # docstring inherited
        # 设置 self._invalidx 和 self._invalidy 为 True，表示需要重新计算 x 和 y
        self._invalidx = True
        self._invalidy = True
        super().set_transform(t)

    @allow_rasterization
    def get_antialiased(self):
        """Return whether antialiased rendering is used."""
        # 返回当前线条是否使用抗锯齿渲染的状态
        return self._antialiased
    def get_color(self):
        """
        Return the line color.

        See also `~.Line2D.set_color`.
        """
        return self._color
    def get_drawstyle(self):
        """
        Return the drawstyle.

        See also `~.Line2D.set_drawstyle`.
        """
        return self._drawstyle
    def get_gapcolor(self):
        """
        Return the line gapcolor.

        See also `~.Line2D.set_gapcolor`.
        """
        return self._gapcolor
    def get_linestyle(self):
        """
        Return the linestyle.

        See also `~.Line2D.set_linestyle`.
        """
        return self._linestyle
    def get_linewidth(self):
        """
        Return the linewidth in points.

        See also `~.Line2D.set_linewidth`.
        """
        return self._linewidth
    def get_marker(self):
        """
        Return the line marker.

        See also `~.Line2D.set_marker`.
        """
        return self._marker.get_marker()
    def get_markeredgecolor(self):
        """
        Return the marker edge color.

        See also `~.Line2D.set_markeredgecolor`.
        """
        mec = self._markeredgecolor
        if cbook._str_equal(mec, 'auto'):
            if mpl.rcParams['_internal.classic_mode']:
                if self._marker.get_marker() in ('.', ','):
                    return self._color
                if (self._marker.is_filled()
                        and self._marker.get_fillstyle() != 'none'):
                    return 'k'  # Bad hard-wired default...
            return self._color
        else:
            return mec
    def get_markeredgewidth(self):
        """
        Return the marker edge width in points.

        See also `~.Line2D.set_markeredgewidth`.
        """
        return self._markeredgewidth
    def _get_markerfacecolor(self, alt=False):
        if self._marker.get_fillstyle() == 'none':
            return 'none'
        fc = self._markerfacecoloralt if alt else self._markerfacecolor
        if cbook._str_lower_equal(fc, 'auto'):
            return self._color
        else:
            return fc
    def get_markerfacecolor(self):
        """
        Return the marker face color.

        See also `~.Line2D.set_markerfacecolor`.
        """
        return self._get_markerfacecolor(alt=False)
    def get_markerfacecoloralt(self):
        """
        Return the alternate marker face color.

        See also `~.Line2D.set_markerfacecoloralt`.
        """
        return self._get_markerfacecolor(alt=True)
    def get_markersize(self):
        """
        Return the marker size in points.

        See also `~.Line2D.set_markersize`.
        """
        return self._markersize
    def get_data(self, orig=True):
        """
        Return the line data as an ``(xdata, ydata)`` pair.

        If *orig* is *True*, return the original data.
        """
        return self.get_xdata(orig=orig), self.get_ydata(orig=orig)
    def get_xdata(self, orig=True):
        """
        Return the xdata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
        # 如果 orig 参数为 True，则返回原始的 x 数据
        if orig:
            return self._xorig
        # 如果 _invalidx 标记为 True，则重新计算并返回处理后的 x 数据
        if self._invalidx:
            self.recache()
        return self._x

    def get_ydata(self, orig=True):
        """
        Return the ydata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
        # 如果 orig 参数为 True，则返回原始的 y 数据
        if orig:
            return self._yorig
        # 如果 _invalidy 标记为 True，则重新计算并返回处理后的 y 数据
        if self._invalidy:
            self.recache()
        return self._y

    def get_path(self):
        """Return the `~matplotlib.path.Path` associated with this line."""
        # 如果 _invalidy 或 _invalidx 标记为 True，则重新计算相关路径数据并返回
        if self._invalidy or self._invalidx:
            self.recache()
        return self._path

    def get_xydata(self):
        """Return the *xy* data as a (N, 2) array."""
        # 如果 _invalidy 或 _invalidx 标记为 True，则重新计算并返回处理后的 xy 数据
        if self._invalidy or self._invalidx:
            self.recache()
        return self._xy

    def set_antialiased(self, b):
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        b : bool
            Boolean value indicating whether to use antialiased rendering.
        """
        # 如果当前的抗锯齿状态与新设置的状态不同，则将 stale 置为 True
        if self._antialiased != b:
            self.stale = True
        self._antialiased = b

    def set_color(self, color):
        """
        Set the color of the line.

        Parameters
        ----------
        color : :mpltype:`color`
            Color to set for the line.
        """
        # 检查颜色格式是否正确，设置新的颜色并将 stale 置为 True
        mcolors._check_color_like(color=color)
        self._color = color
        self.stale = True

    def set_drawstyle(self, drawstyle):
        """
        Set the drawstyle of the plot.

        The drawstyle determines how the points are connected.

        Parameters
        ----------
        drawstyle : {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}
            Drawstyle to set for the plot.
        """
        # 设置新的绘制风格并将 stale 置为 True
        self._drawstyle = drawstyle
        self.stale = True
    def set_drawstyle(self, drawstyle):
        """
        Set the drawing style for the line.

        Parameters
        ----------
        drawstyle : {'default', 'steps-pre', 'steps-mid', 'steps-post', 'steps'}
            Specifies how to connect the points in the line.

            For 'default', the points are connected with straight lines.

            The steps variants connect the points with step-like lines,
            i.e. horizontal lines with vertical steps. They differ in the
            location of the step:

            - 'steps-pre': The step is at the beginning of the line segment,
              i.e. the line will be at the y-value of point to the right.
            - 'steps-mid': The step is halfway between the points.
            - 'steps-post': The step is at the end of the line segment,
              i.e. the line will be at the y-value of the point to the left.
            - 'steps' is equal to 'steps-pre' and is maintained for
              backward-compatibility.

            For examples see :doc:`/gallery/lines_bars_and_markers/step_demo`.
        """
        # If drawstyle is not provided, set it to 'default'
        if drawstyle is None:
            drawstyle = 'default'
        # Validate that drawstyle is in the allowed list of draw styles
        _api.check_in_list(self.drawStyles, drawstyle=drawstyle)
        # Mark the plot as stale if the drawstyle has changed
        if self._drawstyle != drawstyle:
            self.stale = True
            # invalidate to trigger a recache of the path
            self._invalidx = True
        # Set the internal drawstyle attribute to the provided drawstyle
        self._drawstyle = drawstyle

    def set_gapcolor(self, gapcolor):
        """
        Set a color to fill the gaps in the dashed line style.

        .. note::

            Striped lines are created by drawing two interleaved dashed lines.
            There can be overlaps between those two, which may result in
            artifacts when using transparency.

            This functionality is experimental and may change.

        Parameters
        ----------
        gapcolor : :mpltype:`color` or None
            The color with which to fill the gaps. If None, the gaps are
            unfilled.
        """
        # Check if gapcolor is not None, then validate it as a color-like object
        if gapcolor is not None:
            mcolors._check_color_like(color=gapcolor)
        # Set the internal gapcolor attribute to the provided gapcolor
        self._gapcolor = gapcolor
        # Mark the plot as stale to trigger a refresh
        self.stale = True

    def set_linewidth(self, w):
        """
        Set the line width in points.

        Parameters
        ----------
        w : float
            Line width, in points.
        """
        # Ensure w is converted to float
        w = float(w)
        # Mark the plot as stale if the linewidth has changed
        if self._linewidth != w:
            self.stale = True
        # Set the internal linewidth attribute to the provided width
        self._linewidth = w
        # Scale the dash pattern according to the new linewidth
        self._dash_pattern = _scale_dashes(*self._unscaled_dash_pattern, w)
    def set_linestyle(self, ls):
        """
        设置线条的线型。

        Parameters
        ----------
        ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            可选的值:

            - 字符串:

              ==========================================  =================
              linestyle                                   描述
              ==========================================  =================
              ``'-'`` 或 ``'solid'``                      实线
              ``'--'`` 或  ``'dashed'``                   虚线
              ``'-.'`` 或  ``'dashdot'``                  点划线
              ``':'`` 或 ``'dotted'``                     点线
              ``'none'``, ``'None'``, ``' '``, 或 ``''``  不绘制任何东西
              ==========================================  =================

            - 或者可以提供以下形式的虚线元组::

                  (offset, onoffseq)

              其中 ``onoffseq`` 是一对偶数长度的元组，表示点线和空白之间的点数。参见 :meth:`set_dashes`。

            示例请参见 :doc:`/gallery/lines_bars_and_markers/linestyles`。
        """
        if isinstance(ls, str):
            if ls in [' ', '', 'none']:
                ls = 'None'
            _api.check_in_list([*self._lineStyles, *ls_mapper_r], ls=ls)
            if ls not in self._lineStyles:
                ls = ls_mapper_r[ls]
            self._linestyle = ls
        else:
            self._linestyle = '--'
        self._unscaled_dash_pattern = _get_dash_pattern(ls)
        self._dash_pattern = _scale_dashes(
            *self._unscaled_dash_pattern, self._linewidth)
        self.stale = True

    @_docstring.interpd
    def set_marker(self, marker):
        """
        设置线条的标记样式。

        Parameters
        ----------
        marker : 标记样式字符串, `~.path.Path` 或 `~.markers.MarkerStyle`
            查看 `~matplotlib.markers` 获取所有可能的参数的详细描述。
        """
        self._marker = MarkerStyle(marker, self._marker.get_fillstyle())
        self.stale = True

    def _set_markercolor(self, name, has_rcdefault, val):
        if val is None:
            val = mpl.rcParams[f"lines.{name}"] if has_rcdefault else "auto"
        attr = f"_{name}"
        current = getattr(self, attr)
        if current is None:
            self.stale = True
        else:
            neq = current != val
            # 如果不使用数组，比 `np.any(current != val)` 快得多。
            if neq.any() if isinstance(neq, np.ndarray) else neq:
                self.stale = True
        setattr(self, attr, val)

    def set_markeredgecolor(self, ec):
        """
        设置标记边缘颜色。

        Parameters
        ----------
        ec : :mpltype:`color`
        """
        self._set_markercolor("markeredgecolor", True, ec)
    # 设置标记点的填充颜色
    def set_markerfacecolor(self, fc):
        """
        Set the marker face color.

        Parameters
        ----------
        fc : :mpltype:`color`
             Marker face color.
        """
        # 调用内部方法设置标记点的颜色属性
        self._set_markercolor("markerfacecolor", True, fc)

    # 设置备用标记点的填充颜色
    def set_markerfacecoloralt(self, fc):
        """
        Set the alternate marker face color.

        Parameters
        ----------
        fc : :mpltype:`color`
             Alternate marker face color.
        """
        # 调用内部方法设置备用标记点的颜色属性
        self._set_markercolor("markerfacecoloralt", False, fc)

    # 设置标记点边缘的宽度（单位：点）
    def set_markeredgewidth(self, ew):
        """
        Set the marker edge width in points.

        Parameters
        ----------
        ew : float
             Marker edge width, in points.
        """
        # 如果未指定边缘宽度，则使用默认设置
        if ew is None:
            ew = mpl.rcParams['lines.markeredgewidth']
        # 如果当前边缘宽度与新值不同，则设置为脏数据状态
        if self._markeredgewidth != ew:
            self.stale = True
        # 更新边缘宽度属性
        self._markeredgewidth = ew

    # 设置标记点的大小（单位：点）
    def set_markersize(self, sz):
        """
        Set the marker size in points.

        Parameters
        ----------
        sz : float
             Marker size, in points.
        """
        # 将大小转换为浮点数
        sz = float(sz)
        # 如果当前大小与新值不同，则设置为脏数据状态
        if self._markersize != sz:
            self.stale = True
        # 更新标记点大小属性
        self._markersize = sz

    # 设置 x 轴数据数组
    def set_xdata(self, x):
        """
        Set the data array for x.

        Parameters
        ----------
        x : 1D array
            Data array for x coordinates.

        See Also
        --------
        set_data
        set_ydata
        """
        # 检查 x 是否可迭代，否则引发运行时错误
        if not np.iterable(x):
            raise RuntimeError('x must be a sequence')
        # 复制并更新原始数据，标记 x 数据为无效并设置脏数据状态
        self._xorig = copy.copy(x)
        self._invalidx = True
        self.stale = True

    # 设置 y 轴数据数组
    def set_ydata(self, y):
        """
        Set the data array for y.

        Parameters
        ----------
        y : 1D array
            Data array for y coordinates.

        See Also
        --------
        set_data
        set_xdata
        """
        # 检查 y 是否可迭代，否则引发运行时错误
        if not np.iterable(y):
            raise RuntimeError('y must be a sequence')
        # 复制并更新原始数据，标记 y 数据为无效并设置脏数据状态
        self._yorig = copy.copy(y)
        self._invalidy = True
        self.stale = True

    # 设置虚线样式
    def set_dashes(self, seq):
        """
        Set the dash sequence.

        The dash sequence is a sequence of floats of even length describing
        the length of dashes and spaces in points.

        For example, (5, 2, 1, 2) describes a sequence of 5 point and 1 point
        dashes separated by 2 point spaces.

        See also `~.Line2D.set_gapcolor`, which allows those spaces to be
        filled with a color.

        Parameters
        ----------
        seq : sequence of floats (on/off ink in points) or (None, None)
            If *seq* is empty or ``(None, None)``, the linestyle will be set
            to solid.
        """
        # 如果 seq 为空或者为 (None, None)，则设置实线样式
        if seq == (None, None) or len(seq) == 0:
            self.set_linestyle('-')
        else:
            # 否则设置指定的虚线样式
            self.set_linestyle((0, seq))
    def update_from(self, other):
        """Copy properties from *other* to self."""
        # 调用父类方法，将属性从 *other* 复制到 self
        super().update_from(other)
        # 复制线条样式属性
        self._linestyle = other._linestyle
        # 复制线宽属性
        self._linewidth = other._linewidth
        # 复制颜色属性
        self._color = other._color
        # 复制间隙颜色属性
        self._gapcolor = other._gapcolor
        # 复制标记大小属性
        self._markersize = other._markersize
        # 复制标记填充颜色属性
        self._markerfacecolor = other._markerfacecolor
        # 复制备用标记填充颜色属性
        self._markerfacecoloralt = other._markerfacecoloralt
        # 复制标记边缘颜色属性
        self._markeredgecolor = other._markeredgecolor
        # 复制标记边缘宽度属性
        self._markeredgewidth = other._markeredgewidth
        # 复制未缩放的虚线模式属性
        self._unscaled_dash_pattern = other._unscaled_dash_pattern
        # 复制虚线模式属性
        self._dash_pattern = other._dash_pattern
        # 复制虚线端点样式属性
        self._dashcapstyle = other._dashcapstyle
        # 复制虚线连接样式属性
        self._dashjoinstyle = other._dashjoinstyle
        # 复制实线端点样式属性
        self._solidcapstyle = other._solidcapstyle
        # 复制实线连接样式属性
        self._solidjoinstyle = other._solidjoinstyle

        # 再次设置线条样式属性
        self._linestyle = other._linestyle
        # 创建标记样式对象，并设置其属性为 *other* 的标记样式
        self._marker = MarkerStyle(marker=other._marker)
        # 复制绘制样式属性
        self._drawstyle = other._drawstyle



    @_docstring.interpd
    def set_dash_joinstyle(self, s):
        """
        How to join segments of the line if it `~Line2D.is_dashed`.

        The default joinstyle is :rc:`lines.dash_joinstyle`.

        Parameters
        ----------
        s : `.JoinStyle` or %(JoinStyle)s
        """
        # 将输入参数 s 转换为 JoinStyle 对象
        js = JoinStyle(s)
        # 如果当前对象的虚线连接样式不等于 js
        if self._dashjoinstyle != js:
            # 标记对象已过时
            self.stale = True
        # 设置当前对象的虚线连接样式为 js
        self._dashjoinstyle = js



    @_docstring.interpd
    def set_solid_joinstyle(self, s):
        """
        How to join segments if the line is solid (not `~Line2D.is_dashed`).

        The default joinstyle is :rc:`lines.solid_joinstyle`.

        Parameters
        ----------
        s : `.JoinStyle` or %(JoinStyle)s
        """
        # 将输入参数 s 转换为 JoinStyle 对象
        js = JoinStyle(s)
        # 如果当前对象的实线连接样式不等于 js
        if self._solidjoinstyle != js:
            # 标记对象已过时
            self.stale = True
        # 设置当前对象的实线连接样式为 js
        self._solidjoinstyle = js



    def get_dash_joinstyle(self):
        """
        Return the `.JoinStyle` for dashed lines.

        See also `~.Line2D.set_dash_joinstyle`.
        """
        # 返回当前对象的虚线连接样式的名称
        return self._dashjoinstyle.name



    def get_solid_joinstyle(self):
        """
        Return the `.JoinStyle` for solid lines.

        See also `~.Line2D.set_solid_joinstyle`.
        """
        # 返回当前对象的实线连接样式的名称
        return self._solidjoinstyle.name



    @_docstring.interpd
    def set_dash_capstyle(self, s):
        """
        How to draw the end caps if the line is `~Line2D.is_dashed`.

        The default capstyle is :rc:`lines.dash_capstyle`.

        Parameters
        ----------
        s : `.CapStyle` or %(CapStyle)s
        """
        # 将输入参数 s 转换为 CapStyle 对象
        cs = CapStyle(s)
        # 如果当前对象的虚线端点样式不等于 cs
        if self._dashcapstyle != cs:
            # 标记对象已过时
            self.stale = True
        # 设置当前对象的虚线端点样式为 cs
        self._dashcapstyle = cs



    @_docstring.interpd
    def set_solid_capstyle(self, s):
        """
        How to draw the end caps if the line is solid (not `~Line2D.is_dashed`).

        The default capstyle is :rc:`lines.solid_capstyle`.

        Parameters
        ----------
        s : `.CapStyle` or %(CapStyle)s
        """
        # 将输入参数 s 转换为 CapStyle 对象
        cs = CapStyle(s)
        # 如果当前对象的实线端点样式不等于 cs
        if self._solidcapstyle != cs:
            # 标记对象已过时
            self.stale = True
        # 设置当前对象的实线端点样式为 cs
        self._solidcapstyle = cs
    # 设置线条的实线端点样式
    def set_solid_capstyle(self, s):
        """
        How to draw the end caps if the line is solid (not `~Line2D.is_dashed`)

        The default capstyle is :rc:`lines.solid_capstyle`.

        Parameters
        ----------
        s : `.CapStyle` or %(CapStyle)s
        """
        # 创建一个 CapStyle 对象，用于表示线条端点的样式
        cs = CapStyle(s)
        # 如果当前的实线端点样式不等于新设定的样式，标记为需要更新
        if self._solidcapstyle != cs:
            self.stale = True
        # 更新实线端点样式为新设定的样式
        self._solidcapstyle = cs

    # 获取虚线的端点样式
    def get_dash_capstyle(self):
        """
        Return the `.CapStyle` for dashed lines.

        See also `~.Line2D.set_dash_capstyle`.
        """
        # 返回当前虚线端点样式的名称
        return self._dashcapstyle.name

    # 获取实线的端点样式
    def get_solid_capstyle(self):
        """
        Return the `.CapStyle` for solid lines.

        See also `~.Line2D.set_solid_capstyle`.
        """
        # 返回当前实线端点样式的名称
        return self._solidcapstyle.name

    # 判断线条是否为虚线
    def is_dashed(self):
        """
        Return whether line has a dashed linestyle.

        A custom linestyle is assumed to be dashed, we do not inspect the
        ``onoffseq`` directly.

        See also `~.Line2D.set_linestyle`.
        """
        # 判断当前线条的样式是否属于虚线风格
        return self._linestyle in ('--', '-.', ':')
class AxLine(Line2D):
    """
    A helper class that implements `~.Axes.axline`, by recomputing the artist
    transform at draw time.
    """

    def __init__(self, xy1, xy2, slope, **kwargs):
        """
        Parameters
        ----------
        xy1 : (float, float)
            The first set of (x, y) coordinates for the line to pass through.
        xy2 : (float, float) or None
            The second set of (x, y) coordinates for the line to pass through.
            Both *xy2* and *slope* must be passed, but one of them must be None.
        slope : float or None
            The slope of the line. Both *xy2* and *slope* must be passed, but one of
            them must be None.
        """
        super().__init__([0, 1], [0, 1], **kwargs)

        # Check that exactly one of 'xy2' and 'slope' is given
        if (xy2 is None and slope is None or
                xy2 is not None and slope is not None):
            raise TypeError(
                "Exactly one of 'xy2' and 'slope' must be given")

        self._slope = slope
        self._xy1 = xy1
        self._xy2 = xy2

    def get_transform(self):
        ax = self.axes
        points_transform = self._transform - ax.transData + ax.transScale

        if self._xy2 is not None:
            # Calculate transformation for two given points
            (x1, y1), (x2, y2) = \
                points_transform.transform([self._xy1, self._xy2])
            dx = x2 - x1
            dy = y2 - y1
            # Calculate slope based on the transformed coordinates
            if np.allclose(x1, x2):
                if np.allclose(y1, y2):
                    raise ValueError(
                        f"Cannot draw a line through two identical points "
                        f"(x={(x1, x2)}, y={(y1, y2)})")
                slope = np.inf
            else:
                slope = dy / dx
        else:
            # Calculate transformation for one point and a given slope
            x1, y1 = points_transform.transform(self._xy1)
            slope = self._slope
        # Transform view limits and axes limits to draw the line within the plot area
        (vxlo, vylo), (vxhi, vyhi) = ax.transScale.transform(ax.viewLim)
        if np.isclose(slope, 0):
            start = vxlo, y1
            stop = vxhi, y1
        elif np.isinf(slope):
            start = x1, vylo
            stop = x1, vyhi
        else:
            # Find intersections with view limits and draw between the middle two points
            _, start, stop, _ = sorted([
                (vxlo, y1 + (vxlo - x1) * slope),
                (vxhi, y1 + (vxhi - x1) * slope),
                (x1 + (vylo - y1) / slope, vylo),
                (x1 + (vyhi - y1) / slope, vyhi),
            ])
        # Return transformation to draw the line within the axis limits
        return (BboxTransformTo(Bbox([start, stop]))
                + ax.transLimits + ax.transAxes)

    def draw(self, renderer):
        # Force regeneration of transformed path to ensure updated drawing
        self._transformed_path = None
        super().draw(renderer)

    def get_xy1(self):
        """
        Return the *xy1* value of the line.
        """
        return self._xy1

    def get_xy2(self):
        """
        Return the *xy2* value of the line.
        """
        return self._xy2
    # 返回线的斜率值
    def get_slope(self):
        """
        Return the *slope* value of the line.
        """
        return self._slope

    # 设置线段通过的第一个点坐标
    def set_xy1(self, x, y):
        """
        Set the *xy1* value of the line.

        Parameters
        ----------
        x, y : float
            Points for the line to pass through.
        """
        self._xy1 = x, y

    # 设置线段通过的第二个点坐标，前提是斜率未设置
    def set_xy2(self, x, y):
        """
        Set the *xy2* value of the line.

        Parameters
        ----------
        x, y : float
            Points for the line to pass through.
        """
        if self._slope is None:
            self._xy2 = x, y
        else:
            raise ValueError("Cannot set an 'xy2' value while 'slope' is set;"
                             " they differ but their functionalities overlap")

    # 设置线的斜率，前提是第二个点坐标未设置
    def set_slope(self, slope):
        """
        Set the *slope* value of the line.

        Parameters
        ----------
        slope : float
            The slope of the line.
        """
        if self._xy2 is None:
            self._slope = slope
        else:
            raise ValueError("Cannot set a 'slope' value while 'xy2' is set;"
                             " they differ but their functionalities overlap")
class VertexSelector:
    """
    Manage the callbacks to maintain a list of selected vertices for `.Line2D`.
    Derived classes should override the `process_selected` method to do
    something with the picks.

    Here is an example which highlights the selected verts with red circles::

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.lines as lines

        class HighlightSelected(lines.VertexSelector):
            def __init__(self, line, fmt='ro', **kwargs):
                super().__init__(line)
                self.markers, = self.axes.plot([], [], fmt, **kwargs)

            def process_selected(self, ind, xs, ys):
                self.markers.set_data(xs, ys)
                self.canvas.draw()

        fig, ax = plt.subplots()
        x, y = np.random.rand(2, 30)
        line, = ax.plot(x, y, 'bs-', picker=5)

        selector = HighlightSelected(line)
        plt.show()
    """

    def __init__(self, line):
        """
        Parameters
        ----------
        line : `~matplotlib.lines.Line2D`
            The line must already have been added to an `~.axes.Axes` and must
            have its picker property set.
        """
        # 检查线条是否已经添加到 Axes 对象中
        if line.axes is None:
            raise RuntimeError('You must first add the line to the Axes')
        # 检查线条的 picker 属性是否已设置
        if line.get_picker() is None:
            raise RuntimeError('You must first set the picker property '
                               'of the line')
        # 获取线条所在的 Axes 对象
        self.axes = line.axes
        # 存储线条对象
        self.line = line
        # 连接 pick_event 事件到 onpick 方法
        self.cid = self.canvas.callbacks._connect_picklable(
            'pick_event', self.onpick)
        # 初始化选中的索引集合
        self.ind = set()

    canvas = property(lambda self: self.axes.figure.canvas)

    def process_selected(self, ind, xs, ys):
        """
        Default "do nothing" implementation of the `process_selected` method.

        Parameters
        ----------
        ind : list of int
            The indices of the selected vertices.
        xs, ys : array-like
            The coordinates of the selected vertices.
        """
        # 默认的 process_selected 方法实现为空，可以在派生类中覆盖此方法
        pass

    def onpick(self, event):
        """When the line is picked, update the set of selected indices."""
        # 确保事件源是当前处理的线条对象
        if event.artist is not self.line:
            return
        # 对选中的索引集合进行更新
        self.ind ^= set(event.ind)
        # 对索引进行排序
        ind = sorted(self.ind)
        # 获取线条数据
        xdata, ydata = self.line.get_data()
        # 调用 process_selected 方法处理选中的数据
        self.process_selected(ind, xdata[ind], ydata[ind])


lineStyles = Line2D._lineStyles
lineMarkers = MarkerStyle.markers
drawStyles = Line2D.drawStyles
fillStyles = MarkerStyle.fillstyles
```