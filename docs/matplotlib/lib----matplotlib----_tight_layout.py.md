# `D:\src\scipysrc\matplotlib\lib\matplotlib\_tight_layout.py`

```
"""
Routines to adjust subplot params so that subplots are
nicely fit in the figure. In doing so, only axis labels, tick labels, Axes
titles and offsetboxes that are anchored to Axes are currently considered.

Internally, this module assumes that the margins (left margin, etc.) which are
differences between ``Axes.get_tightbbox`` and ``Axes.bbox`` are independent of
Axes position. This may fail if ``Axes.adjustable`` is ``datalim`` as well as
such cases as when left or right margin are affected by xlabel.
"""

import numpy as np

import matplotlib as mpl
from matplotlib import _api, artist as martist
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox


def _auto_adjust_subplotpars(
        fig, renderer, shape, span_pairs, subplot_list,
        ax_bbox_list=None, pad=1.08, h_pad=None, w_pad=None, rect=None):
    """
    Return a dict of subplot parameters to adjust spacing between subplots
    or ``None`` if resulting Axes would have zero height or width.

    Note that this function ignores geometry information of subplot itself, but
    uses what is given by the *shape* and *subplot_list* parameters.  Also, the
    results could be incorrect if some subplots have ``adjustable=datalim``.

    Parameters
    ----------
    shape : tuple[int, int]
        Number of rows and columns of the grid.
    span_pairs : list[tuple[slice, slice]]
        List of rowspans and colspans occupied by each subplot.
    subplot_list : list of subplots
        List of subplots that will be used to calculate optimal subplot_params.
    pad : float
        Padding between the figure edge and the edges of subplots, as a
        fraction of the font size.
    h_pad, w_pad : float
        Padding (height/width) between edges of adjacent subplots, as a
        fraction of the font size.  Defaults to *pad*.
    rect : tuple
        (left, bottom, right, top), default: None.
    """
    # Determine the number of rows and columns from the given shape
    rows, cols = shape

    # Calculate the font size in inches based on the current font size setting
    font_size_inch = (FontProperties(
        size=mpl.rcParams["font.size"]).get_size_in_points() / 72)
    
    # Convert padding from font size fraction to inches
    pad_inch = pad * font_size_inch
    vpad_inch = h_pad * font_size_inch if h_pad is not None else pad_inch
    hpad_inch = w_pad * font_size_inch if w_pad is not None else pad_inch

    # Validate the length of span_pairs and subplot_list
    if len(span_pairs) != len(subplot_list) or len(subplot_list) == 0:
        raise ValueError

    # Initialize margins to None if rect is not provided
    if rect is None:
        margin_left = margin_bottom = margin_right = margin_top = None
    else:
        # Extract margins from the rect tuple
        margin_left, margin_bottom, _right, _top = rect
        margin_right = 1 - _right if _right else None
        margin_top = 1 - _top if _top else None

    # Initialize vertical and horizontal spaces arrays
    vspaces = np.zeros((rows + 1, cols))
    hspaces = np.zeros((rows, cols + 1))

    # If ax_bbox_list is not provided, compute it based on the subplot_list
    if ax_bbox_list is None:
        ax_bbox_list = [
            Bbox.union([ax.get_position(original=True) for ax in subplots])
            for subplots in subplot_list]
    # 遍历subplot_list、ax_bbox_list和span_pairs的元素，同时迭代
    for subplots, ax_bbox, (rowspan, colspan) in zip(
            subplot_list, ax_bbox_list, span_pairs):
        # 如果所有子图中的坐标轴都不可见，则跳过本次迭代
        if all(not ax.get_visible() for ax in subplots):
            continue

        # 初始化空列表bb，用于存储可见坐标轴的紧凑边界框
        bb = []
        # 遍历subplots中的每个坐标轴
        for ax in subplots:
            # 如果当前坐标轴可见，则获取其布局边界框并添加到bb列表中
            if ax.get_visible():
                bb += [martist._get_tightbbox_for_layout_only(ax, renderer)]

        # 计算bb列表中所有边界框的并集，得到初始的紧凑边界框tight_bbox_raw
        tight_bbox_raw = Bbox.union(bb)
        # 将紧凑边界框tight_bbox_raw转换为相对于fig的Figure坐标系的边界框tight_bbox
        tight_bbox = fig.transFigure.inverted().transform_bbox(tight_bbox_raw)

        # 更新水平空间数组hspaces和垂直空间数组vspaces中的值，根据坐标轴边界框和布局的相对位置
        hspaces[rowspan, colspan.start] += ax_bbox.xmin - tight_bbox.xmin  # l
        hspaces[rowspan, colspan.stop] += tight_bbox.xmax - ax_bbox.xmax  # r
        vspaces[rowspan.start, colspan] += tight_bbox.ymax - ax_bbox.ymax  # t
        vspaces[rowspan.stop, colspan] += ax_bbox.ymin - tight_bbox.ymin  # b

    # 获取图形的尺寸（单位为英寸）
    fig_width_inch, fig_height_inch = fig.get_size_inches()

    # 如果未指定左边缘的边距，则计算左边缘的边距
    if not margin_left:
        margin_left = max(hspaces[:, 0].max(), 0) + pad_inch/fig_width_inch
        # 检查是否存在超级标签并在布局中，若存在，则计算相对宽度
        suplabel = fig._supylabel
        if suplabel and suplabel.get_in_layout():
            rel_width = fig.transFigure.inverted().transform_bbox(
                suplabel.get_window_extent(renderer)).width
            margin_left += rel_width + pad_inch/fig_width_inch

    # 如果未指定右边缘的边距，则计算右边缘的边距
    if not margin_right:
        margin_right = max(hspaces[:, -1].max(), 0) + pad_inch/fig_width_inch

    # 如果未指定顶部边缘的边距，则计算顶部边缘的边距
    if not margin_top:
        margin_top = max(vspaces[0, :].max(), 0) + pad_inch/fig_height_inch
        # 检查是否存在超级标题并在布局中，若存在，则计算相对高度
        if fig._suptitle and fig._suptitle.get_in_layout():
            rel_height = fig.transFigure.inverted().transform_bbox(
                fig._suptitle.get_window_extent(renderer)).height
            margin_top += rel_height + pad_inch/fig_height_inch

    # 如果未指定底部边缘的边距，则计算底部边缘的边距
    if not margin_bottom:
        margin_bottom = max(vspaces[-1, :].max(), 0) + pad_inch/fig_height_inch
        # 检查是否存在超级X标签并在布局中，若存在，则计算相对高度
        suplabel = fig._supxlabel
        if suplabel and suplabel.get_in_layout():
            rel_height = fig.transFigure.inverted().transform_bbox(
                suplabel.get_window_extent(renderer)).height
            margin_bottom += rel_height + pad_inch/fig_height_inch

    # 如果左右边距之和大于或等于1，提示布局紧凑度无法应用，并返回None
    if margin_left + margin_right >= 1:
        _api.warn_external('Tight layout not applied. The left and right '
                           'margins cannot be made large enough to '
                           'accommodate all Axes decorations.')
        return None

    # 如果上下边距之和大于或等于1，提示布局紧凑度无法应用，并返回None
    if margin_bottom + margin_top >= 1:
        _api.warn_external('Tight layout not applied. The bottom and top '
                           'margins cannot be made large enough to '
                           'accommodate all Axes decorations.')
        return None

    # 使用计算得到的边距值构造关键字参数kwargs，表示图形的边界范围
    kwargs = dict(left=margin_left,
                  right=1 - margin_right,
                  bottom=margin_bottom,
                  top=1 - margin_top)
    # 如果图表有多列
    if cols > 1:
        # 计算水平间距，将最大间距与水平填充量和图表宽度比例化
        hspace = hspaces[:, 1:-1].max() + hpad_inch / fig_width_inch
        # 计算每个子图的宽度
        h_axes = (1 - margin_right - margin_left - hspace * (cols - 1)) / cols
        # 如果计算出的子图宽度小于0，显示警告信息并返回空值
        if h_axes < 0:
            _api.warn_external('Tight layout not applied. tight_layout '
                               'cannot make Axes width small enough to '
                               'accommodate all Axes decorations')
            return None
        else:
            # 计算调整后的子图水平间距，并存入参数字典中
            kwargs["wspace"] = hspace / h_axes

    # 如果图表有多行
    if rows > 1:
        # 计算垂直间距，将最大间距与垂直填充量和图表高度比例化
        vspace = vspaces[1:-1, :].max() + vpad_inch / fig_height_inch
        # 计算每个子图的高度
        v_axes = (1 - margin_top - margin_bottom - vspace * (rows - 1)) / rows
        # 如果计算出的子图高度小于0，显示警告信息并返回空值
        if v_axes < 0:
            _api.warn_external('Tight layout not applied. tight_layout '
                               'cannot make Axes height small enough to '
                               'accommodate all Axes decorations.')
            return None
        else:
            # 计算调整后的子图垂直间距，并存入参数字典中
            kwargs["hspace"] = vspace / v_axes

    # 返回参数字典
    return kwargs
# 返回一个由给定的 Axes 列表生成的子图规范列表
def get_subplotspec_list(axes_list, grid_spec=None):
    # 初始化一个空列表，用于存储子图规范
    subplotspec_list = []
    # 遍历每个 Axes 对象
    for ax in axes_list:
        # 获取 Axes 对象的定位器，如果没有定位器则使用该 Axes 对象本身
        axes_or_locator = ax.get_axes_locator()
        if axes_or_locator is None:
            axes_or_locator = ax
        
        # 检查对象是否具有 get_subplotspec 方法
        if hasattr(axes_or_locator, "get_subplotspec"):
            # 获取子图规范对象
            subplotspec = axes_or_locator.get_subplotspec()
            if subplotspec is not None:
                # 获取最顶层的子图规范对象
                subplotspec = subplotspec.get_topmost_subplotspec()
                # 获取子图规范对象所属的 GridSpec 对象
                gs = subplotspec.get_gridspec()
                # 如果指定了 grid_spec 参数，检查子图规范对象是否属于指定的 GridSpec
                if grid_spec is not None:
                    if gs != grid_spec:
                        subplotspec = None
                # 如果子图规范对象的 subplot 参数被本地修改，则置为 None
                elif gs.locally_modified_subplot_params():
                    subplotspec = None
        else:
            subplotspec = None
        
        # 将处理后的子图规范对象添加到列表中
        subplotspec_list.append(subplotspec)

    # 返回子图规范列表
    return subplotspec_list


# 返回具有指定填充的紧凑布局图形的子图参数
def get_tight_layout_figure(fig, axes_list, subplotspec_list, renderer,
                            pad=1.08, h_pad=None, w_pad=None, rect=None):
    # 多个 Axes 可能共享相同的子图规范（例如使用 axes_grid1），需要将它们分组在一起
    ss_to_subplots = {ss: [] for ss in subplotspec_list}
    # 将每个 Axes 对象与其子图规范对应起来
    for ax, ss in zip(axes_list, subplotspec_list):
        ss_to_subplots[ss].append(ax)
    # 弹出不兼容 tight_layout 的 Axes 的警告
    if ss_to_subplots.pop(None, None):
        _api.warn_external(
            "This figure includes Axes that are not compatible with "
            "tight_layout, so results might be incorrect.")
    # 如果没有有效的子图规范对象，则返回空字典
    if not ss_to_subplots:
        return {}
    # 提取分组后的子图列表
    subplot_list = list(ss_to_subplots.values())
    # 获取每个子图规范对象的位置边界框列表
    ax_bbox_list = [ss.get_position(fig) for ss in ss_to_subplots]

    # 计算所有子图规范对象中最大的行数和列数
    max_nrows = max(ss.get_gridspec().nrows for ss in ss_to_subplots)
    max_ncols = max(ss.get_gridspec().ncols for ss in ss_to_subplots)

    # 初始化一个空的跨度对列表
    span_pairs = []
    # 遍历 ss_to_subplots 中的每个 subplot 对象
    for ss in ss_to_subplots:
        # 获取当前 subplot 的网格规格（行数和列数）
        rows, cols = ss.get_gridspec().get_geometry()
        # 计算最大行数和列数与当前 subplot 行数和列数的商和余数
        div_row, mod_row = divmod(max_nrows, rows)
        div_col, mod_col = divmod(max_ncols, cols)
        
        # 检查行数的余数是否为零，如果不是则警告布局调整未应用
        if mod_row != 0:
            _api.warn_external('tight_layout not applied: number of rows '
                               'in subplot specifications must be '
                               'multiples of one another.')
            return {}  # 返回空字典表示布局调整失败
        
        # 检查列数的余数是否为零，如果不是则警告布局调整未应用
        if mod_col != 0:
            _api.warn_external('tight_layout not applied: number of '
                               'columns in subplot specifications must be '
                               'multiples of one another.')
            return {}  # 返回空字典表示布局调整失败
        
        # 计算当前 subplot 在整体布局中的切片范围，并添加到 span_pairs 列表中
        span_pairs.append((
            slice(ss.rowspan.start * div_row, ss.rowspan.stop * div_row),
            slice(ss.colspan.start * div_col, ss.colspan.stop * div_col)))

    # 调用 _auto_adjust_subplotpars 函数自动调整子图参数
    kwargs = _auto_adjust_subplotpars(fig, renderer,
                                      shape=(max_nrows, max_ncols),
                                      span_pairs=span_pairs,
                                      subplot_list=subplot_list,
                                      ax_bbox_list=ax_bbox_list,
                                      pad=pad, h_pad=h_pad, w_pad=w_pad)

    # 如果 rect 参数不为空且 kwargs 不为空，则进一步调整布局参数
    if rect is not None and kwargs is not None:
        # 如果指定了 rect，子图区域（包括标签）将适合于 rect 指定的矩形区域
        # 注意，rect 参数指定了整体 axes.bbox 覆盖的区域
        left, bottom, right, top = rect
        if left is not None:
            left += kwargs["left"]  # 调整左侧边界
        if bottom is not None:
            bottom += kwargs["bottom"]  # 调整底部边界
        if right is not None:
            right -= (1 - kwargs["right"])  # 调整右侧边界
        if top is not None:
            top -= (1 - kwargs["top"])  # 调整顶部边界
        
        # 重新调用 _auto_adjust_subplotpars 函数，使用调整后的 rect 参数
        kwargs = _auto_adjust_subplotpars(fig, renderer,
                                          shape=(max_nrows, max_ncols),
                                          span_pairs=span_pairs,
                                          subplot_list=subplot_list,
                                          ax_bbox_list=ax_bbox_list,
                                          pad=pad, h_pad=h_pad, w_pad=w_pad,
                                          rect=(left, bottom, right, top))

    # 返回调整后的布局参数 kwargs
    return kwargs
```