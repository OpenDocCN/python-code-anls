# `D:\src\scipysrc\matplotlib\lib\matplotlib\colorbar.py`

```
"""
Colorbars are a visualization of the mapping from scalar values to colors.
In Matplotlib they are drawn into a dedicated `~.axes.Axes`.

.. note::
   Colorbars are typically created through `.Figure.colorbar` or its pyplot
   wrapper `.pyplot.colorbar`, which internally use `.Colorbar` together with
   `.make_axes_gridspec` (for `.GridSpec`-positioned Axes) or `.make_axes` (for
   non-`.GridSpec`-positioned Axes).

   End-users most likely won't need to directly use this module's API.
"""

import logging  # 导入日志模块

import numpy as np  # 导入 NumPy 数组操作库

import matplotlib as mpl  # 导入 Matplotlib 库
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker  # 导入 Matplotlib 内部模块
import matplotlib.artist as martist  # 导入 Matplotlib 艺术家模块
import matplotlib.patches as mpatches  # 导入 Matplotlib 图形模块
import matplotlib.path as mpath  # 导入 Matplotlib 路径模块
import matplotlib.spines as mspines  # 导入 Matplotlib 边框模块
import matplotlib.transforms as mtransforms  # 导入 Matplotlib 变换模块
from matplotlib import _docstring  # 导入 Matplotlib 文档字符串模块

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_docstring.interpd.update(
    _make_axes_kw_doc="""
location : None or {'left', 'right', 'top', 'bottom'}
    The location, relative to the parent Axes, where the colorbar Axes
    is created.  It also determines the *orientation* of the colorbar
    (colorbars on the left and right are vertical, colorbars at the top
    and bottom are horizontal).  If None, the location will come from the
    *orientation* if it is set (vertical colorbars on the right, horizontal
    ones at the bottom), or default to 'right' if *orientation* is unset.

orientation : None or {'vertical', 'horizontal'}
    The orientation of the colorbar.  It is preferable to set the *location*
    of the colorbar, as that also determines the *orientation*; passing
    incompatible values for *location* and *orientation* raises an exception.

fraction : float, default: 0.15
    Fraction of original Axes to use for colorbar.

shrink : float, default: 1.0
    Fraction by which to multiply the size of the colorbar.

aspect : float, default: 20
    Ratio of long to short dimensions.

pad : float, default: 0.05 if vertical, 0.15 if horizontal
    Fraction of original Axes between colorbar and new image Axes.

anchor : (float, float), optional
    The anchor point of the colorbar Axes.
    Defaults to (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal.

panchor : (float, float), or *False*, optional
    The anchor point of the colorbar parent Axes. If *False*, the parent
    axes' anchor will be unchanged.
    Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.""",
    _colormap_kw_doc="""
extend : {'neither', 'both', 'min', 'max'}
    Make pointed end(s) for out-of-range values (unless 'neither').  These are
    set for a given colormap using the colormap set_under and set_over methods.

extendfrac : {*None*, 'auto', length, lengths}
    If set to *None*, both the minimum and maximum triangular colorbar
    extensions will have a length of 5% of the interior colorbar length (this
    is the default setting).

    If set to 'auto', makes the triangular colorbar extensions the same lengths
"""
    # 如果 *spacing* 设置为 'uniform'，则该参数指定内部色条扩展的长度，可以是一个标量或者一个两元素序列。
    # 当标量时，表示最小和最大三角形色条扩展的长度均为内部色条长度的一部分。
    # 当为两元素序列时，分别表示最小和最大色条扩展的长度，作为内部色条长度的一部分。
    # 如果 *spacing* 设置为 'proportional'，则该参数指定内部色条扩展的长度与相邻内部色条的长度相同。
extendrect : bool
    # 如果为 False，则颜色条的最小和最大扩展将是三角形（默认行为）。
    # 如果为 True，则扩展将是矩形的。
    If *False* the minimum and maximum colorbar extensions will be triangular
    (the default).  If *True* the extensions will be rectangular.

spacing : {'uniform', 'proportional'}
    # 对于离散的颜色条（`.BoundaryNorm` 或等高线），'uniform' 表示每种颜色的间距相同；
    # 'proportional' 则使间距与数据间隔成比例。
    For discrete colorbars (`.BoundaryNorm` or contours), 'uniform' gives each
    color the same space; 'proportional' makes the space proportional to the
    data interval.

ticks : None or list of ticks or Locator
    # 如果为 None，则从输入自动确定刻度。
    If None, ticks are determined automatically from the input.

format : None or str or Formatter
    # 如果为 None，则使用 `~.ticker.ScalarFormatter`。
    # 支持格式字符串，例如 ``"%4.2e"`` 或 ``"{x:.2e}"``。
    # 也可以指定另一种 `~.ticker.Formatter`。
    If None, `~.ticker.ScalarFormatter` is used.
    Format strings, e.g., ``"%4.2e"`` or ``"{x:.2e}"``, are supported.
    An alternative `~.ticker.Formatter` may be given instead.

drawedges : bool
    # 是否在颜色边界处绘制线条。
    Whether to draw lines at color boundaries.

label : str
    # 颜色条长轴上的标签。
    The label on the colorbar's long axis.

boundaries, values : None or a sequence
    # 如果未设置，则颜色映射将在 0-1 范围内显示。
    # 如果是序列，则 *values* 的长度必须比 *boundaries* 少 1。
    # 对于由 *boundaries* 相邻条目限定的每个区域，将使用 *values* 中相应值的颜色。
    If unset, the colormap will be displayed on a 0-1 scale.
    If sequences, *values* must have a length 1 less than *boundaries*.  For
    each region delimited by adjacent entries in *boundaries*, the color mapped
    to the corresponding value in values will be used.
    Normally only useful for indexed colors (i.e. ``norm=NoNorm()``) or other
    unusual circumstances.""")


def _set_ticks_on_axis_warn(*args, **kwargs):
    # 一个顶层函数，由 Colorbar.__init__ 放置在轴的 set_xticks 和 set_yticks 中。
    # 提示使用 colorbar 的 set_ticks() 方法代替。
    # a top level function which gets put in at the axes'
    # set_xticks and set_yticks by Colorbar.__init__.
    _api.warn_external("Use the colorbar set_ticks() method instead.")


class _ColorbarSpine(mspines.Spine):
    def __init__(self, axes):
        # 初始化函数，用于创建颜色条的脊柱对象。
        # 参数 axes 表示所属的轴对象。
        self._ax = axes
        super().__init__(axes, 'colorbar', mpath.Path(np.empty((0, 2))))
        mpatches.Patch.set_transform(self, axes.transAxes)

    def get_window_extent(self, renderer=None):
        # 此脊柱对象不与任何轴相关联，也不需要调整其位置，
        # 因此可以直接从超级父类中获取窗口范围。
        # This Spine has no Axis associated with it, and doesn't need to adjust
        # its location, so we can directly get the window extent from the
        # super-super-class.
        return mpatches.Patch.get_window_extent(self, renderer=renderer)

    def set_xy(self, xy):
        # 设置脊柱对象的顶点坐标。
        self._path = mpath.Path(xy, closed=True)
        self._xy = xy
        self.stale = True

    def draw(self, renderer):
        # 绘制脊柱对象。
        ret = mpatches.Patch.draw(self, renderer)
        self.stale = False
        return ret


class _ColorbarAxesLocator:
    """
    Shrink the Axes if there are triangular or rectangular extends.
    """
    def __init__(self, cbar):
        # 初始化函数，用于缩小轴对象，如果有三角形或矩形的扩展。
        # 参数 cbar 表示所属的颜色条对象。
        self._cbar = cbar
        self._orig_locator = cbar.ax._axes_locator
    def __call__(self, ax, renderer):
        # 如果存在原始定位器，则使用原始定位器确定位置
        if self._orig_locator is not None:
            pos = self._orig_locator(ax, renderer)
        else:
            # 否则，获取轴的原始位置
            pos = ax.get_position(original=True)
        
        # 如果颜色条不延伸，则直接返回位置
        if self._cbar.extend == 'neither':
            return pos

        # 获取比例缩放后的 y 值和延伸长度
        y, extendlen = self._cbar._proportional_y()
        
        # 如果不延伸到下限，则将下限的延伸长度设为 0
        if not self._cbar._extend_lower():
            extendlen[0] = 0
        
        # 如果不延伸到上限，则将上限的延伸长度设为 0
        if not self._cbar._extend_upper():
            extendlen[1] = 0
        
        # 计算延伸的总长度
        len = sum(extendlen) + 1
        
        # 计算收缩比例
        shrink = 1 / len
        
        # 计算偏移量
        offset = extendlen[0] / len
        
        # 如果轴具有 '_colorbar_info' 属性，则获取其比例
        if hasattr(ax, '_colorbar_info'):
            aspect = ax._colorbar_info['aspect']
        else:
            aspect = False
        
        # 根据颜色条的方向调整轴的长宽比和位置
        if self._cbar.orientation == 'vertical':
            if aspect:
                # 设置颜色条轴的长宽比
                self._cbar.ax.set_box_aspect(aspect * shrink)
            
            # 垂直方向上收缩轴的高度，并根据偏移量调整位置
            pos = pos.shrunk(1, shrink).translated(0, offset * pos.height)
        else:
            if aspect:
                # 设置颜色条轴的长宽比
                self._cbar.ax.set_box_aspect(1 / (aspect * shrink))
            
            # 水平方向上收缩轴的宽度，并根据偏移量调整位置
            pos = pos.shrunk(shrink, 1).translated(offset * pos.width, 0)
        
        # 返回调整后的位置
        return pos

    def get_subplotspec(self):
        # 使 tight_layout 布局满意，返回颜色条轴的子图规范或者原始定位器的子图规范（如果有的话）
        return (
            self._cbar.ax.get_subplotspec()
            or getattr(self._orig_locator, "get_subplotspec", lambda: None)())
@_docstring.interpd
class Colorbar:
    r"""
    Draw a colorbar in an existing Axes.

    Typically, colorbars are created using `.Figure.colorbar` or
    `.pyplot.colorbar` and associated with `.ScalarMappable`\s (such as an
    `.AxesImage` generated via `~.axes.Axes.imshow`).

    In order to draw a colorbar not associated with other elements in the
    figure, e.g. when showing a colormap by itself, one can create an empty
    `.ScalarMappable`, or directly pass *cmap* and *norm* instead of *mappable*
    to `Colorbar`.

    Useful public methods are :meth:`set_label` and :meth:`add_lines`.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    lines : list
        A list of `.LineCollection` (empty if no lines were drawn).
    dividers : `.LineCollection`
        A LineCollection (empty if *drawedges* is ``False``).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.

    mappable : `.ScalarMappable`
        The mappable whose colormap and norm will be used.

        To show the under- and over- value colors, the mappable's norm should
        be specified as ::

            norm = colors.Normalize(clip=False)

        To show the colors versus index instead of on a 0-1 scale, use::

            norm=colors.NoNorm()

    cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The colormap to use.  This parameter is ignored, unless *mappable* is
        None.

    norm : `~matplotlib.colors.Normalize`
        The normalization to use.  This parameter is ignored, unless *mappable*
        is None.

    alpha : float
        The colorbar transparency between 0 (transparent) and 1 (opaque).

    orientation : None or {'vertical', 'horizontal'}
        If None, use the value determined by *location*. If both
        *orientation* and *location* are None then defaults to 'vertical'.

    ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}
        The location of the colorbar ticks. The *ticklocation* must match
        *orientation*. For example, a horizontal colorbar can only have ticks
        at the top or the bottom. If 'auto', the ticks will be the same as
        *location*, so a colorbar to the left will have ticks to the left. If
        *location* is None, the ticks will be at the bottom for a horizontal
        colorbar and at the right for a vertical.

    %(_colormap_kw_doc)s
    """
    """
    location : None or {'left', 'right', 'top', 'bottom'}
        设置颜色条的方向和刻度位置。使用单个参数设置。左侧和右侧的颜色条是垂直的，
        顶部和底部的颜色条是水平的。刻度位置与方向相同，因此如果位置为'top'，则刻度位于顶部。
        也可以单独提供*orientation*和/或*ticklocation*，将覆盖*location*设置的值，
        但对于不兼容的组合将会出错。

        .. versionadded:: 3.7
    """

    n_rasterize = 50  # 当颜色数目 >= n_rasterize 时，将光栅化实体

    @property
    def locator(self):
        """获取颜色条的主刻度`.Locator`。"""
        return self._long_axis().get_major_locator()

    @locator.setter
    def locator(self, loc):
        """设置颜色条的主刻度`.Locator`。"""
        self._long_axis().set_major_locator(loc)
        self._locator = loc

    @property
    def minorlocator(self):
        """获取颜色条的次刻度`.Locator`。"""
        return self._long_axis().get_minor_locator()

    @minorlocator.setter
    def minorlocator(self, loc):
        """设置颜色条的次刻度`.Locator`。"""
        self._long_axis().set_minor_locator(loc)
        self._minorlocator = loc

    @property
    def formatter(self):
        """获取颜色条的主刻度标签`.Formatter`。"""
        return self._long_axis().get_major_formatter()

    @formatter.setter
    def formatter(self, fmt):
        """设置颜色条的主刻度标签`.Formatter`。"""
        self._long_axis().set_major_formatter(fmt)
        self._formatter = fmt

    @property
    def minorformatter(self):
        """获取颜色条的次刻度标签`.Formatter`。"""
        return self._long_axis().get_minor_formatter()

    @minorformatter.setter
    def minorformatter(self, fmt):
        """设置颜色条的次刻度标签`.Formatter`。"""
        self._long_axis().set_minor_formatter(fmt)
        self._minorformatter = fmt

    def _cbar_cla(self):
        """清除交互式颜色条状态的函数。"""
        for x in self._interactive_funcs:
            delattr(self.ax, x)
        # 现在恢复旧的 cla() 并可以直接调用它
        del self.ax.cla
        self.ax.cla()
    def update_normal(self, mappable):
        """
        Update solid patches, lines, etc.

        This is meant to be called when the norm of the image or contour plot
        to which this colorbar belongs changes.

        If the norm on the mappable is different than before, this resets the
        locator and formatter for the axis, so if these have been customized,
        they will need to be customized again.  However, if the norm only
        changes values of *vmin*, *vmax* or *cmap* then the old formatter
        and locator will be preserved.
        """
        # 打印调试信息，记录 mappable 的 norm 和 colorbar 自身的 norm
        _log.debug('colorbar update normal %r %r', mappable.norm, self.norm)
        
        # 将 mappable 赋值给 colorbar 的 mappable 属性
        self.mappable = mappable
        
        # 设置 colorbar 的透明度与 mappable 相同
        self.set_alpha(mappable.get_alpha())
        
        # 将 mappable 的 cmap 赋值给 colorbar 的 cmap
        self.cmap = mappable.cmap
        
        # 如果 mappable 的 norm 与 colorbar 的 norm 不同
        if mappable.norm != self.norm:
            # 更新 colorbar 的 norm
            self.norm = mappable.norm
            # 重置 colorbar 的 locator 和 formatter
            self._reset_locator_formatter_scale()

        # 绘制所有元素
        self._draw_all()
        
        # 如果 mappable 是 contour.ContourSet 类型
        if isinstance(self.mappable, contour.ContourSet):
            CS = self.mappable
            # 如果 ContourSet 不是填充类型，则为 colorbar 添加线条
            if not CS.filled:
                self.add_lines(CS)
        
        # 将 colorbar 标记为需要更新
        self.stale = True
    def _draw_all(self):
        """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """
        # 如果方向为垂直方向
        if self.orientation == 'vertical':
            # 检查是否配置了显示次要Y轴刻度，如果是则打开
            if mpl.rcParams['ytick.minor.visible']:
                self.minorticks_on()
        else:
            # 检查是否配置了显示次要X轴刻度，如果是则打开
            if mpl.rcParams['xtick.minor.visible']:
                self.minorticks_on()
        
        # 获取长轴并设置标签位置和刻度位置为ticklocation
        self._long_axis().set(label_position=self.ticklocation,
                              ticks_position=self.ticklocation)
        
        # 设置短轴刻度为空列表，包括次要刻度
        self._short_axis().set_ticks([])
        self._short_axis().set_ticks([], minor=True)

        # 设置self._boundaries和self._values，包括扩展部分
        # self._boundaries是每个颜色方块的边界
        # self._values是映射到norm的值以获取颜色的值
        self._process_values()
        
        # 将self.vmin和self.vmax设置为边界的第一个和最后一个值，不包括扩展部分
        self.vmin, self.vmax = self._boundaries[self._inside][[0, -1]]
        
        # 计算X和Y的网格
        X, Y = self._mesh()
        
        # 绘制扩展三角形，并缩小内部Axes以适应
        # 同时将轮廓路径添加到self.outline spine
        self._do_extends()
        
        # 设置lower和upper为vmin和vmax
        lower, upper = self.vmin, self.vmax
        
        # 如果长轴被反转，需要交换vmin和vmax
        if self._long_axis().get_inverted():
            lower, upper = upper, lower
        
        # 根据方向设置轴的xlim和ylim
        if self.orientation == 'vertical':
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(lower, upper)
        else:
            self.ax.set_ylim(0, 1)
            self.ax.set_xlim(lower, upper)
        
        # 设置刻度定位器和格式化程序
        # 由于边界norms和均匀间距要求手动定位器，所以有些复杂
        self.update_ticks()
        
        # 如果是填充的情况
        if self._filled:
            ind = np.arange(len(self._values))
            # 如果存在下扩展
            if self._extend_lower():
                ind = ind[1:]
            # 如果存在上扩展
            if self._extend_upper():
                ind = ind[:-1]
            # 添加实心图形
            self._add_solids(X, Y, self._values[ind, np.newaxis])
    def _add_solids(self, X, Y, C):
        """Draw the colors; optionally add separators."""
        # Cleanup previously set artists.
        如果已存在 self.solids，则删除之
        if self.solids is not None:
            self.solids.remove()
        # 清理先前设置的所有 solid patches
        for solid in self.solids_patches:
            solid.remove()
        
        # 根据 mappable 类型决定添加新的 artist(s)，如果需要 hatch，则使用 individual patches，否则使用 pcolormesh
        mappable = getattr(self, 'mappable', None)
        if (isinstance(mappable, contour.ContourSet)
                and any(hatch is not None for hatch in mappable.hatches)):
            self._add_solids_patches(X, Y, C, mappable)
        else:
            # 使用 pcolormesh 创建 solids，绘制颜色图
            self.solids = self.ax.pcolormesh(
                X, Y, C, cmap=self.cmap, norm=self.norm, alpha=self.alpha,
                edgecolors='none', shading='flat')
            # 如果不需要绘制边缘线，并且 _y 的长度达到了 n_rasterize，则将 solids 设置为矢量化
            if not self.drawedges:
                if len(self._y) >= self.n_rasterize:
                    self.solids.set_rasterized(True)
        
        # 更新 dividers
        self._update_dividers()

    def _update_dividers(self):
        # 如果不需要绘制边缘线，则将 dividers 的 segments 设置为空并返回
        if not self.drawedges:
            self.dividers.set_segments([])
            return
        
        # 确定所有内部 dividers 的位置
        if self.orientation == 'vertical':
            lims = self.ax.get_ylim()
            bounds = (lims[0] < self._y) & (self._y < lims[1])
        else:
            lims = self.ax.get_xlim()
            bounds = (lims[0] < self._y) & (self._y < lims[1])
        y = self._y[bounds]
        
        # 如果扩展 lower bound，则添加相应的外部 dividers
        if self._extend_lower():
            y = np.insert(y, 0, lims[0])
        # 如果扩展 upper bound，则添加相应的外部 dividers
        if self._extend_upper():
            y = np.append(y, lims[1])
        
        # 创建 X, Y 的网格
        X, Y = np.meshgrid([0, 1], y)
        
        # 根据 orientation 创建 segments
        if self.orientation == 'vertical':
            segments = np.dstack([X, Y])
        else:
            segments = np.dstack([Y, X])
        
        # 设置 dividers 的 segments
        self.dividers.set_segments(segments)

    def _add_solids_patches(self, X, Y, C, mappable):
        # 创建足够数量的 hatches
        hatches = mappable.hatches * (len(C) + 1)
        
        # 如果扩展 lower bound，则移除第一个 hatch
        if self._extend_lower():
            hatches = hatches[1:]
        
        patches = []
        # 遍历每一个 X 的行
        for i in range(len(X) - 1):
            # 创建 xy 数组
            xy = np.array([[X[i, 0], Y[i, 1]],
                           [X[i, 1], Y[i, 0]],
                           [X[i + 1, 1], Y[i + 1, 0]],
                           [X[i + 1, 0], Y[i + 1, 1]]])
            
            # 创建 PathPatch 对象，并添加到 Axes 中
            patch = mpatches.PathPatch(mpath.Path(xy),
                                       facecolor=self.cmap(self.norm(C[i][0])),
                                       hatch=hatches[i], linewidth=0,
                                       antialiased=False, alpha=self.alpha)
            self.ax.add_patch(patch)
            patches.append(patch)
        
        # 将创建的 patches 赋给 solids_patches
        self.solids_patches = patches
    def update_ticks(self):
        """
        Set up the ticks and ticklabels. This should not be needed by users.
        """
        # 调用 _get_ticker_locator_formatter 方法获取刻度定位器和格式化器
        self._get_ticker_locator_formatter()
        # 设置主刻度定位器为 _locator 所指定的定位器
        self._long_axis().set_major_locator(self._locator)
        # 设置次刻度定位器为 _minorlocator 所指定的定位器
        self._long_axis().set_minor_locator(self._minorlocator)
        # 设置主刻度格式化器为 _formatter 所指定的格式化器
        self._long_axis().set_major_formatter(self._formatter)

    def _get_ticker_locator_formatter(self):
        """
        Return the ``locator`` and ``formatter`` of the colorbar.

        If they have not been defined (i.e. are *None*), the formatter and
        locator are retrieved from the axis, or from the value of the
        boundaries for a boundary norm.

        Called by update_ticks...
        """
        # 获取当前实例的定位器和格式化器
        locator = self._locator
        formatter = self._formatter
        minorlocator = self._minorlocator
        # 如果色标的规范是 BoundaryNorm 类型
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
            # 如果定位器为 None，则使用固定定位器 FixedLocator，并设置刻度数量为 10
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
            # 如果次定位器为 None，则使用固定定位器 FixedLocator
            if minorlocator is None:
                minorlocator = ticker.FixedLocator(b)
        # 如果色标的规范是 NoNorm 类型
        elif isinstance(self.norm, colors.NoNorm):
            if locator is None:
                # 将刻度放置在 NoNorm 的边界之间的整数上
                nv = len(self._values)
                base = 1 + int(nv / 10)
                locator = ticker.IndexLocator(base=base, offset=.5)
        # 如果有指定 boundaries
        elif self.boundaries is not None:
            b = self._boundaries[self._inside]
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
        else:  # 大多数情况
            if locator is None:
                # 如果未显式设置定位器，则使用该轴的默认主定位器
                locator = self._long_axis().get_major_locator()
            # 如果次定位器为 None，则使用该轴的默认次定位器
            if minorlocator is None:
                minorlocator = self._long_axis().get_minor_locator()

        # 如果次定位器为 None，则使用 NullLocator（无刻度的定位器）
        if minorlocator is None:
            minorlocator = ticker.NullLocator()

        # 如果格式化器为 None，则使用该轴的默认主格式化器
        if formatter is None:
            formatter = self._long_axis().get_major_formatter()

        # 更新实例的定位器、格式化器和次定位器
        self._locator = locator
        self._formatter = formatter
        self._minorlocator = minorlocator
        # 记录调试信息
        _log.debug('locator: %r', locator)
    def set_ticks(self, ticks, *, labels=None, minor=False, **kwargs):
        """
        Set tick locations.

        Parameters
        ----------
        ticks : 1D array-like
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        minor : bool, default: False
            If ``False``, set the major ticks; if ``True``, the minor ticks.
        **kwargs
            `.Text` properties for the labels. These take effect only if you
            pass *labels*. In other cases, please use `~.Axes.tick_params`.
        """
        # 如果 ticks 是可迭代对象（如数组），设置主要或次要 ticks 和对应的标签
        if np.iterable(ticks):
            self._long_axis().set_ticks(ticks, labels=labels, minor=minor,
                                        **kwargs)
            # 获取设置后的主要定位器
            self._locator = self._long_axis().get_major_locator()
        else:
            # 如果 ticks 是一个单独的定位器对象，直接设置为主要定位器
            self._locator = ticks
            self._long_axis().set_major_locator(self._locator)
        # 设置标志以指示需要重新绘制
        self.stale = True

    def get_ticks(self, minor=False):
        """
        Return the ticks as a list of locations.

        Parameters
        ----------
        minor : boolean, default: False
            if True return the minor ticks.
        """
        # 根据参数 minor 返回主要或次要 ticks 的位置列表
        if minor:
            return self._long_axis().get_minorticklocs()
        else:
            return self._long_axis().get_majorticklocs()

    def set_ticklabels(self, ticklabels, *, minor=False, **kwargs):
        """
        [*Discouraged*] Set tick labels.

        .. admonition:: Discouraged

            The use of this method is discouraged, because of the dependency
            on tick positions. In most cases, you'll want to use
            ``set_ticks(positions, labels=labels)`` instead.

            If you are using this method, you should always fix the tick
            positions before, e.g. by using `.Colorbar.set_ticks` or by
            explicitly setting a `~.ticker.FixedLocator` on the long axis
            of the colorbar. Otherwise, ticks are free to move and the
            labels may end up in unexpected positions.

        Parameters
        ----------
        ticklabels : sequence of str or of `.Text`
            Texts for labeling each tick location in the sequence set by
            `.Colorbar.set_ticks`; the number of labels must match the number
            of locations.

        update_ticks : bool, default: True
            This keyword argument is ignored and will be removed.
            Deprecated

        minor : bool
            If True, set minor ticks instead of major ticks.

        **kwargs
            `.Text` properties for the labels.
        """
        # 使用长轴对象设置主要或次要 ticks 的标签
        self._long_axis().set_ticklabels(ticklabels, minor=minor, **kwargs)

    def minorticks_on(self):
        """
        Turn on colorbar minor ticks.
        """
        # 打开次要 ticks 的显示
        self.ax.minorticks_on()
        # 设置短轴的次要定位器为空定位器
        self._short_axis().set_minor_locator(ticker.NullLocator())
    # 将色标的次要刻度关闭
    def minorticks_off(self):
        # 使用 NullLocator 创建一个空的次要刻度定位器
        self._minorlocator = ticker.NullLocator()
        # 获取色标的长轴并设置其次要刻度定位器为 NullLocator
        self._long_axis().set_minor_locator(self._minorlocator)

    # 为色标的长轴添加标签
    def set_label(self, label, *, loc=None, **kwargs):
        """
        添加标签到色标的长轴。

        Parameters
        ----------
        label : str
            标签文本。
        loc : str, optional
            标签位置。

            - 对于水平方向，可选值为 {'left', 'center', 'right'}
            - 对于垂直方向，可选值为 {'bottom', 'center', 'top'}

            默认值为 :rc:`xaxis.labellocation` 或 :rc:`yaxis.labellocation`，取决于方向。
        **kwargs
            关键字参数传递给 `~.Axes.set_xlabel` / `~.Axes.set_ylabel`。
            支持的关键字包括 *labelpad* 和 `.Text` 属性。
        """
        # 如果色标为垂直方向，则使用 set_ylabel 方法设置标签
        if self.orientation == "vertical":
            self.ax.set_ylabel(label, loc=loc, **kwargs)
        else:
            # 如果色标为水平方向，则使用 set_xlabel 方法设置标签
            self.ax.set_xlabel(label, loc=loc, **kwargs)
        # 设置色标为需要更新状态
        self.stale = True

    # 设置色标的透明度
    def set_alpha(self, alpha):
        """
        设置透明度，范围在 0（透明）到 1（不透明）之间。

        如果提供了一个数组，则将 alpha 设置为 None，以使用与颜色映射相关联的透明度值。
        """
        # 如果 alpha 是 numpy 数组，则将 self.alpha 设置为 None，否则设置为 alpha
        self.alpha = None if isinstance(alpha, np.ndarray) else alpha

    # 设置色标的长轴比例尺
    def _set_scale(self, scale, **kwargs):
        """
        设置色标的长轴比例尺。

        Parameters
        ----------
        scale : {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
            要应用的轴比例尺类型。

        **kwargs
            根据比例尺不同，接受不同的关键字参数。
            参见各类关键字参数：

            - `matplotlib.scale.LinearScale`
            - `matplotlib.scale.LogScale`
            - `matplotlib.scale.SymmetricalLogScale`
            - `matplotlib.scale.LogitScale`
            - `matplotlib.scale.FuncScale`
            - `matplotlib.scale.AsinhScale`

        Notes
        -----
        默认情况下，Matplotlib 支持上述提到的比例尺。
        此外，可以使用 `matplotlib.scale.register_scale` 注册自定义比例尺。
        这些比例尺也可以在此处使用。
        """
        # 获取色标的长轴并调用其 _set_axes_scale 方法来设置比例尺
        self._long_axis()._set_axes_scale(scale, **kwargs)
    def remove(self):
        """
        Remove this colorbar from the figure.

        If the colorbar was created with ``use_gridspec=True`` the previous
        gridspec is restored.
        """
        # 检查是否在 Axes 对象上有 _colorbar_info 属性
        if hasattr(self.ax, '_colorbar_info'):
            # 获取所有父级对象
            parents = self.ax._colorbar_info['parents']
            # 遍历每个父级对象
            for a in parents:
                # 如果当前 Axes 对象在父级对象的 _colorbars 中，则移除
                if self.ax in a._colorbars:
                    a._colorbars.remove(self.ax)

        # 从 Axes 对象中移除自身
        self.ax.remove()

        # 断开与 mappable 对象的 colorbar_cid 回调连接
        self.mappable.callbacks.disconnect(self.mappable.colorbar_cid)
        # 将 mappable 对象的 colorbar 相关属性重置为 None
        self.mappable.colorbar = None
        self.mappable.colorbar_cid = None
        # 移除与扩展功能相关的回调连接
        self.ax.callbacks.disconnect(self._extend_cid1)
        self.ax.callbacks.disconnect(self._extend_cid2)

        try:
            # 尝试获取 mappable 对象的 axes 属性
            ax = self.mappable.axes
        except AttributeError:
            # 如果没有 axes 属性，则返回
            return
        try:
            # 尝试获取当前 Axes 对象的 subplotspec，并恢复原始 gridspec
            subplotspec = self.ax.get_subplotspec().get_gridspec()._subplot_spec
        except AttributeError:  # 如果 use_gridspec 是 False
            # 获取当前 Axes 对象的位置，并设置回去
            pos = ax.get_position(original=True)
            ax._set_position(pos)
        else:  # 如果 use_gridspec 是 True
            # 将当前 Axes 对象重新设置到之前保存的 subplotspec
            ax.set_subplotspec(subplotspec)
    def _process_values(self):
        """
        根据 self.boundaries 和 self.values 设置 self._boundaries 和 self._values，
        如果它们不为 None，否则根据色彩映射的大小以及 norm 的 vmin/vmax 来设置。
        """
        if self.values is not None:
            # 如果 self.values 不为 None，则使用其值设置 self._boundaries...
            self._values = np.array(self.values)
            if self.boundaries is None:
                # 将值按照 1/2 dv 进行边界设置：
                b = np.zeros(len(self.values) + 1)
                b[1:-1] = 0.5 * (self._values[:-1] + self._values[1:])
                b[0] = 2.0 * b[1] - b[2]
                b[-1] = 2.0 * b[-2] - b[-3]
                self._boundaries = b
                return
            self._boundaries = np.array(self.boundaries)
            return

        # 否则根据 boundaries 设置 values
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
        elif isinstance(self.norm, colors.NoNorm):
            # NoNorm 有 N 个块，因此有 N+1 个边界，以整数为中心：
            b = np.arange(self.cmap.N + 1) - .5
        elif self.boundaries is not None:
            b = self.boundaries
        else:
            # 否则根据色彩映射的大小生成边界：
            N = self.cmap.N + 1
            b, _ = self._uniform_y(N)
        # 如果需要，添加额外的边界：
        if self._extend_lower():
            b = np.hstack((b[0] - 1, b))
        if self._extend_upper():
            b = np.hstack((b, b[-1] + 1))

        # 将边界从 0-1 转换为 vmin-vmax：
        if self.mappable.get_array() is not None:
            self.mappable.autoscale_None()
        if not self.norm.scaled():
            # 如果在自动缩放后仍未缩放，则使用默认值 0 和 1
            self.norm.vmin = 0
            self.norm.vmax = 1
        self.norm.vmin, self.norm.vmax = mtransforms.nonsingular(
            self.norm.vmin, self.norm.vmax, expander=0.1)
        if (not isinstance(self.norm, colors.BoundaryNorm) and
                (self.boundaries is None)):
            b = self.norm.inverse(b)

        self._boundaries = np.asarray(b, dtype=float)
        self._values = 0.5 * (self._boundaries[:-1] + self._boundaries[1:])
        if isinstance(self.norm, colors.NoNorm):
            self._values = (self._values + 0.00001).astype(np.int16)
    def _mesh(self):
        """
        Return the coordinate arrays for the colorbar pcolormesh/patches.

        These are scaled between vmin and vmax, and already handle colorbar
        orientation.
        """
        # 获取比例化后的 y 坐标数组和一个未使用的值
        y, _ = self._proportional_y()
        
        # 使用 colorbar 的 vmin 和 vmax，这些值可能不同于 norm。当色彩映射的范围
        # 比 colorbar 的范围窄时，我们希望容纳额外的等高线。
        if (isinstance(self.norm, (colors.BoundaryNorm, colors.NoNorm))
                or self.boundaries is not None):
            # 不使用 norm。
            y = y * (self.vmax - self.vmin) + self.vmin
        else:
            # 在上下文管理器中更新 norm 的值，因为这只是一个临时更改，我们不希望
            # 传播到与 norm 相关的任何信号（如 callbacks.blocked）。
            with self.norm.callbacks.blocked(), \
                    cbook._setattr_cm(self.norm,
                                      vmin=self.vmin,
                                      vmax=self.vmax):
                y = self.norm.inverse(y)
        
        # 将处理后的 y 值赋给实例变量 _y
        self._y = y
        
        # 创建一个网格，X 和 Y 是网格化后的坐标数组
        X, Y = np.meshgrid([0., 1.], y)
        
        # 如果 colorbar 的方向是垂直的，则返回 (X, Y)，否则返回 (Y, X)
        if self.orientation == 'vertical':
            return (X, Y)
        else:
            return (Y, X)

    def _forward_boundaries(self, x):
        # 将边界等均匀映射到 0 到 1 之间...
        b = self._boundaries
        y = np.interp(x, b, np.linspace(0, 1, len(b)))
        
        # 下面的操作避免了位于边界区域的刻度线：
        eps = (b[-1] - b[0]) * 1e-6
        
        # 将超出边界的值映射到 -1 和 2，以避免这些刻度线出现在 extends 区域内...
        y[x < b[0]-eps] = -1
        y[x > b[-1]+eps] = 2
        
        return y

    def _inverse_boundaries(self, x):
        # 反转上述操作...
        b = self._boundaries
        return np.interp(x, np.linspace(0, 1, len(b)), b)
    # 重置定位器（locator）、格式化器（formatter）及其比例尺为默认值。
    # 如果调用了此方法（在初始化或颜色条的 mappable 标准变化时，如 Colorbar.update_normal），任何用户硬编码的更改都需要重新输入。
    def _reset_locator_formatter_scale(self):
        # 处理值，确保在重置前进入了正确的状态
        self._process_values()
        # 将定位器设置为 None，重置为默认值
        self._locator = None
        # 将次要定位器设置为 None，重置为默认值
        self._minorlocator = None
        # 将格式化器设置为 None，重置为默认值
        self._formatter = None
        # 将次要格式化器设置为 None，重置为默认值
        self._minorformatter = None
        # 如果 mappable 是 contour.ContourSet 类型并且 norm 是 colors.LogNorm 类型
        if (isinstance(self.mappable, contour.ContourSet) and
                isinstance(self.norm, colors.LogNorm)):
            # 如果轮廓具有对数标准化，则给它们设置对数比例尺
            self._set_scale('log')
        # 如果 boundaries 不是 None 或者 norm 是 colors.BoundaryNorm 类型
        elif (self.boundaries is not None or
                isinstance(self.norm, colors.BoundaryNorm)):
            # 如果 spacing 是 'uniform'，则使用 _forward_boundaries 和 _inverse_boundaries 函数设置比例尺
            if self.spacing == 'uniform':
                funcs = (self._forward_boundaries, self._inverse_boundaries)
                self._set_scale('function', functions=funcs)
            # 如果 spacing 是 'proportional'，则使用线性比例尺
            elif self.spacing == 'proportional':
                self._set_scale('linear')
        # 如果 norm 具有 _scale 属性（如果它存在且不为 None）
        elif getattr(self.norm, '_scale', None):
            # 使用 norm 的 _scale 属性（如果它存在且不为 None）设置比例尺
            self._set_scale(self.norm._scale)
        # 如果 norm 是 colors.Normalize 类型
        elif type(self.norm) is colors.Normalize:
            # 使用线性比例尺
            self._set_scale('linear')
        else:
            # 否则，从 Norm 推导比例尺
            funcs = (self.norm, self.norm.inverse)
            self._set_scale('function', functions=funcs)

    # 根据颜色数据值集合，返回对应的颜色条数据坐标
    def _locate(self, x):
        # 如果 norm 是 colors.NoNorm 或者 colors.BoundaryNorm 类型
        if isinstance(self.norm, (colors.NoNorm, colors.BoundaryNorm)):
            # 使用当前 _boundaries 和 x
            b = self._boundaries
            xn = x
        else:
            # 使用归一化坐标进行计算，以使插值更精确
            b = self.norm(self._boundaries, clip=False).filled()
            xn = self.norm(x, clip=False).filled()

        # 取出当前内部的边界值
        bunique = b[self._inside]
        # 取出当前的 y 值
        yunique = self._y

        # 使用 np.interp 进行插值计算，得到颜色条数据坐标
        z = np.interp(xn, bunique, yunique)
        return z

    # 简单的辅助函数：生成均匀分布的 colorbar 数据坐标
    def _uniform_y(self, N):
        # 返回 N 个均匀分布的 colorbar 数据坐标，以及如果需要的话的扩展长度
        automin = automax = 1. / (N - 1.)
        extendlength = self._get_extension_lengths(self.extendfrac,
                                                   automin, automax,
                                                   default=0.05)
        y = np.linspace(0, 1, N)
        return y, extendlength
    def _proportional_y(self):
        """
        Return colorbar data coordinates for the boundaries of
        a proportional colorbar, plus extension lengths if required:
        """
        # Check if normalization is based on BoundaryNorm or explicit boundaries
        if (isinstance(self.norm, colors.BoundaryNorm) or
                self.boundaries is not None):
            # Calculate normalized y coordinates within the range [0, 1]
            y = (self._boundaries - self._boundaries[self._inside][0])
            y = y / (self._boundaries[self._inside][-1] -
                     self._boundaries[self._inside][0])
            
            # Calculate yscaled based on the forward boundaries if spacing is uniform
            if self.spacing == 'uniform':
                yscaled = self._forward_boundaries(self._boundaries)
            else:
                yscaled = y
        else:
            # Calculate normalized y using the provided norm function
            y = self.norm(self._boundaries.copy())
            y = np.ma.filled(y, np.nan)
            yscaled = y
        
        # Extract values within the valid range defined by _inside
        y = y[self._inside]
        yscaled = yscaled[self._inside]
        
        # Normalize y and yscaled to the range [0, 1] using Normalize
        norm = colors.Normalize(y[0], y[-1])
        y = np.ma.filled(norm(y), np.nan)
        norm = colors.Normalize(yscaled[0], yscaled[-1])
        yscaled = np.ma.filled(norm(yscaled), np.nan)
        
        # Calculate extension lengths based on the first and last boundary spacings
        automin = yscaled[1] - yscaled[0]
        automax = yscaled[-1] - yscaled[-2]
        extendlength = [0, 0]
        
        # Determine extension lengths if _extend_lower or _extend_upper are True
        if self._extend_lower() or self._extend_upper():
            extendlength = self._get_extension_lengths(
                    self.extendfrac, automin, automax, default=0.05)
        
        return y, extendlength

    def _get_extension_lengths(self, frac, automin, automax, default=0.05):
        """
        Return the lengths of colorbar extensions.

        This is a helper method for _uniform_y and _proportional_y.
        """
        # Initialize default extension lengths
        extendlength = np.array([default, default])
        
        # Handle different cases for extendfrac parameter
        if isinstance(frac, str):
            _api.check_in_list(['auto'], extendfrac=frac.lower())
            # Set extendlength based on automin and automax when 'auto' is specified
            extendlength[:] = [automin, automax]
        elif frac is not None:
            try:
                # Attempt to directly set min and max extension fractions
                extendlength[:] = frac
                # Check for NaN values in extendlength which would be invalid
                if np.isnan(extendlength).any():
                    raise ValueError()
            except (TypeError, ValueError) as err:
                # Raise an error for invalid extendfrac values
                raise ValueError('invalid value for extendfrac') from err
        
        return extendlength
    def _extend_lower(self):
        """
        Return whether the lower limit is open ended.
        """
        # 根据轴的倒置状态确定要使用的极值（最大值或最小值）
        minmax = "max" if self._long_axis().get_inverted() else "min"
        # 检查是否在两侧都开放或者根据长轴的倒置状态开放下限
        return self.extend in ('both', minmax)

    def _extend_upper(self):
        """
        Return whether the upper limit is open ended.
        """
        # 根据轴的倒置状态确定要使用的极值（最小值或最大值）
        minmax = "min" if self._long_axis().get_inverted() else "max"
        # 检查是否在两侧都开放或者根据长轴的倒置状态开放上限
        return self.extend in ('both', minmax)

    def _long_axis(self):
        """
        Return the long axis based on orientation.
        """
        # 如果图表是垂直方向，则返回 y 轴；否则返回 x 轴
        if self.orientation == 'vertical':
            return self.ax.yaxis
        return self.ax.xaxis

    def _short_axis(self):
        """
        Return the short axis based on orientation.
        """
        # 如果图表是垂直方向，则返回 x 轴；否则返回 y 轴
        if self.orientation == 'vertical':
            return self.ax.xaxis
        return self.ax.yaxis

    def _get_view(self):
        """
        Return the current view range (vmin, vmax) of the colorbar.
        """
        # 从 norm 对象获取当前视图范围的最小值和最大值
        return self.norm.vmin, self.norm.vmax

    def _set_view(self, view):
        """
        Set the view range (vmin, vmax) of the colorbar.
        """
        # 设置 colorbar 的视图范围（vmin, vmax），从给定的视图参数中获取
        self.norm.vmin, self.norm.vmax = view

    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        """
        Set the view range (vmin, vmax) of the colorbar using a bounding box.
        """
        # 使用给定的缩放边界框设置 colorbar 的视图范围（vmin, vmax）
        new_xbound, new_ybound = self.ax._prepare_view_from_bbox(
            bbox, direction=direction, mode=mode, twinx=twinx, twiny=twiny)
        # 根据 colorbar 的方向设置 norm 对象的视图范围（vmin, vmax）
        if self.orientation == 'horizontal':
            self.norm.vmin, self.norm.vmax = new_xbound
        elif self.orientation == 'vertical':
            self.norm.vmin, self.norm.vmax = new_ybound

    def drag_pan(self, button, key, x, y):
        """
        Perform drag pan operation on the colorbar.
        """
        # 获取用于拖动平移的点坐标
        points = self.ax._get_pan_points(button, key, x, y)
        # 如果存在拖动点，则根据 colorbar 的方向设置 norm 对象的视图范围（vmin, vmax）
        if points is not None:
            if self.orientation == 'horizontal':
                self.norm.vmin, self.norm.vmax = points[:, 0]
            elif self.orientation == 'vertical':
                self.norm.vmin, self.norm.vmax = points[:, 1]
# Backcompat API：ColorbarBase现在指向Colorbar，用于向后兼容。

def _normalize_location_orientation(location, orientation):
    # 如果未指定location，则从orientation获取适当的tick位置
    if location is None:
        location = _get_ticklocation_from_orientation(orientation)
    # 使用_api.check_getitem检查和获取特定location的设置信息
    loc_settings = _api.check_getitem({
        "left":   {"location": "left", "anchor": (1.0, 0.5),
                   "panchor": (0.0, 0.5), "pad": 0.10},
        "right":  {"location": "right", "anchor": (0.0, 0.5),
                   "panchor": (1.0, 0.5), "pad": 0.05},
        "top":    {"location": "top", "anchor": (0.5, 0.0),
                   "panchor": (0.5, 1.0), "pad": 0.05},
        "bottom": {"location": "bottom", "anchor": (0.5, 1.0),
                   "panchor": (0.5, 0.0), "pad": 0.15},
    }, location=location)
    # 将location设置的orientation添加到loc_settings中
    loc_settings["orientation"] = _get_orientation_from_location(location)
    # 如果指定了orientation且与loc_settings中的不一致，则引发TypeError
    if orientation is not None and orientation != loc_settings["orientation"]:
        raise TypeError("location and orientation are mutually exclusive")
    return loc_settings


def _get_orientation_from_location(location):
    # 使用_api.check_getitem根据location返回相应的orientation
    return _api.check_getitem(
        {None: None, "left": "vertical", "right": "vertical",
         "top": "horizontal", "bottom": "horizontal"}, location=location)


def _get_ticklocation_from_orientation(orientation):
    # 使用_api.check_getitem根据orientation返回相应的tick位置
    return _api.check_getitem(
        {None: "right", "vertical": "right", "horizontal": "bottom"},
        orientation=orientation)


@_docstring.interpd
def make_axes(parents, location=None, orientation=None, fraction=0.15,
              shrink=1.0, aspect=20, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The Axes is placed in the figure of the *parents* Axes, by resizing and
    repositioning *parents*.

    Parameters
    ----------
    parents : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of `~.axes.Axes`
        The Axes to use as parents for placing the colorbar.
    %(_make_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        The child Axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    """
    # 标准化location和orientation，获取相关设置信息
    loc_settings = _normalize_location_orientation(location, orientation)
    # 将loc_settings中的orientation添加到kwargs字典中，并设定ticklocation为location
    kwargs['orientation'] = loc_settings['orientation']
    location = kwargs['ticklocation'] = loc_settings['location']

    # 将anchor和panchor从kwargs中弹出，使用loc_settings中的默认值或传入值
    anchor = kwargs.pop('anchor', loc_settings['anchor'])
    panchor = kwargs.pop('panchor', loc_settings['panchor'])
    aspect0 = aspect

    # 如果parents是numpy数组，则转换为列表；如果是可迭代对象，则转换为列表；否则，转换为包含一个元素的列表
    if isinstance(parents, np.ndarray):
        parents = list(parents.flat)
    elif np.iterable(parents):
        parents = list(parents)
    else:
        parents = [parents]

    # 获取第一个parent的figure对象
    fig = parents[0].get_figure()
    # 根据图形的约束布局确定填充值，否则使用给定的位置设置中的填充值
    pad0 = 0.05 if fig.get_constrained_layout() else loc_settings['pad']
    pad = kwargs.pop('pad', pad0)

    # 检查所有父级 Axes 是否都属于同一图形，否则抛出数值错误异常
    if not all(fig is ax.get_figure() for ax in parents):
        raise ValueError('Unable to create a colorbar Axes as not all '
                         'parents share the same figure.')

    # 计算包围给定 Axes 的边界框
    parents_bbox = mtransforms.Bbox.union(
        [ax.get_position(original=True).frozen() for ax in parents])

    pb = parents_bbox

    # 根据位置调整边界框的大小和位置
    if location in ('left', 'right'):
        if location == 'left':
            pbcb, _, pb1 = pb.splitx(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splitx(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(1.0, shrink).anchored(anchor, pbcb)
    else:
        if location == 'bottom':
            pbcb, _, pb1 = pb.splity(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splity(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(shrink, 1.0).anchored(anchor, pbcb)

        # 根据 y 轴相对于 x 轴的比例定义纵横比
        aspect = 1.0 / aspect

    # 定义一个转换，将旧的 Axes 坐标转换为新的 Axes 坐标
    shrinking_trans = mtransforms.BboxTransform(parents_bbox, pb1)

    # 使用新的转换，调整每个父级 Axes 的位置
    for ax in parents:
        new_posn = shrinking_trans.transform(ax.get_position(original=True))
        new_posn = mtransforms.Bbox(new_posn)
        ax._set_position(new_posn)
        if panchor is not False:
            ax.set_anchor(panchor)

    # 在图形上添加一个新的 Axes 用作 colorbar
    cax = fig.add_axes(pbcb, label="<colorbar>")
    for a in parents:
        # 告知父级 Axes 它有一个 colorbar
        a._colorbars += [cax]
    # 存储 colorbar 相关信息
    cax._colorbar_info = dict(
        parents=parents,
        location=location,
        shrink=shrink,
        anchor=anchor,
        panchor=panchor,
        fraction=fraction,
        aspect=aspect0,
        pad=pad)
    # 设置 colorbar 的锚点和纵横比
    cax.set_anchor(anchor)
    cax.set_box_aspect(aspect)
    cax.set_aspect('auto')

    # 返回创建的 colorbar Axes 和剩余的 kwargs
    return cax, kwargs
# 用于文档字符串格式化的修饰器，用于函数文档的插值
@_docstring.interpd
# 创建一个适用于颜色条的 `~.axes.Axes` 对象
def make_axes_gridspec(parent, *, location=None, orientation=None,
                       fraction=0.15, shrink=1.0, aspect=20, **kwargs):
    """
    在 `parent` Axes 中创建一个适合颜色条的 `~.axes.Axes` 对象。

    通过调整和重新定位 `parent` 来放置这个 Axes。

    此函数类似于 `.make_axes`，并且与之大部分兼容。
    主要的区别在于：

    - `.make_axes_gridspec` 需要 `parent` 具有一个子图规格。
    - `.make_axes` 将 Axes 放置在图形坐标中；
      `.make_axes_gridspec` 使用子图规格来定位它。
    - `.make_axes` 更新父对象的位置。`.make_axes_gridspec`
      则用新的子图规格替换父对象的子图规格。

    Parameters
    ----------
    parent : `~matplotlib.axes.Axes`
        用于放置颜色条的父 Axes。
    %(_make_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        子 Axes。
    kwargs : dict
        用于创建颜色条实例时传递的减少的关键字字典。
    """

    # 规范化位置和方向设置
    loc_settings = _normalize_location_orientation(location, orientation)
    # 设置颜色条的方向
    kwargs['orientation'] = loc_settings['orientation']
    # 设置刻度位置
    location = kwargs['ticklocation'] = loc_settings['location']

    # 初始化一些变量
    aspect0 = aspect
    # 弹出一些关键字参数
    anchor = kwargs.pop('anchor', loc_settings['anchor'])
    panchor = kwargs.pop('panchor', loc_settings['panchor'])
    pad = kwargs.pop('pad', loc_settings["pad"])
    # 计算宽度或高度空间
    wh_space = 2 * pad / (1 - pad)

    # 根据位置不同选择不同的子图规格
    if location in ('left', 'right'):
        gs = parent.get_subplotspec().subgridspec(
            3, 2, wspace=wh_space, hspace=0,
            height_ratios=[(1-anchor[1])*(1-shrink), shrink, anchor[1]*(1-shrink)])
        if location == 'left':
            gs.set_width_ratios([fraction, 1 - fraction - pad])
            ss_main = gs[:, 1]
            ss_cb = gs[1, 0]
        else:
            gs.set_width_ratios([1 - fraction - pad, fraction])
            ss_main = gs[:, 0]
            ss_cb = gs[1, 1]
    else:
        gs = parent.get_subplotspec().subgridspec(
            2, 3, hspace=wh_space, wspace=0,
            width_ratios=[anchor[0]*(1-shrink), shrink, (1-anchor[0])*(1-shrink)])
        if location == 'top':
            gs.set_height_ratios([fraction, 1 - fraction - pad])
            ss_main = gs[1, :]
            ss_cb = gs[0, 1]
        else:
            gs.set_height_ratios([1 - fraction - pad, fraction])
            ss_main = gs[0, :]
            ss_cb = gs[1, 1]
        aspect = 1 / aspect

    # 设置父对象的子图规格
    parent.set_subplotspec(ss_main)
    # 设置锚点
    if panchor is not False:
        parent.set_anchor(panchor)

    # 获取图形对象
    fig = parent.get_figure()
    # 添加一个子图到图形中，作为颜色条
    cax = fig.add_subplot(ss_cb, label="<colorbar>")
    # 设置子图的锚点
    cax.set_anchor(anchor)
    # 设置子图的盒子比例
    cax.set_box_aspect(aspect)
    # 设置子图的纵横比为自动调整
    cax.set_aspect('auto')
    # 将颜色条信息存储为一个字典，并赋给 cax._colorbar_info
    cax._colorbar_info = dict(
        location=location,   # 设置颜色条的位置
        parents=[parent],    # 指定颜色条的父对象
        shrink=shrink,       # 缩放比例
        anchor=anchor,       # 锚点位置
        panchor=panchor,     # 父锚点位置
        fraction=fraction,   # 颜色条长度占整体的比例
        aspect=aspect0,      # 长宽比
        pad=pad              # 颜色条与其周围内容的间距
    )

    # 返回颜色条对象 cax 和其它参数 kwargs
    return cax, kwargs
```