# `D:\src\scipysrc\matplotlib\lib\matplotlib\_constrained_layout.py`

```py
"""
Adjust subplot layouts so that there are no overlapping Axes or Axes
decorations.  All Axes decorations are dealt with (labels, ticks, titles,
ticklabels) and some dependent artists are also dealt with (colorbar,
suptitle).

Layout is done via `~matplotlib.gridspec`, with one constraint per gridspec,
so it is possible to have overlapping Axes if the gridspecs overlap (i.e.
using `~matplotlib.gridspec.GridSpecFromSubplotSpec`).  Axes placed using
``figure.subplots()`` or ``figure.add_subplots()`` will participate in the
layout.  Axes manually placed via ``figure.add_axes()`` will not.

See Tutorial: :ref:`constrainedlayout_guide`

General idea:
-------------

First, a figure has a gridspec that divides the figure into nrows and ncols,
with heights and widths set by ``height_ratios`` and ``width_ratios``,
often just set to 1 for an equal grid.

Subplotspecs that are derived from this gridspec can contain either a
``SubPanel``, a ``GridSpecFromSubplotSpec``, or an ``Axes``.  The ``SubPanel``
and ``GridSpecFromSubplotSpec`` are dealt with recursively and each contain an
analogous layout.

Each ``GridSpec`` has a ``_layoutgrid`` attached to it.  The ``_layoutgrid``
has the same logical layout as the ``GridSpec``.   Each row of the grid spec
has a top and bottom "margin" and each column has a left and right "margin".
The "inner" height of each row is constrained to be the same (or as modified
by ``height_ratio``), and the "inner" width of each column is
constrained to be the same (as modified by ``width_ratio``), where "inner"
is the width or height of each column/row minus the size of the margins.

Then the size of the margins for each row and column are determined as the
max width of the decorators on each Axes that has decorators in that margin.
For instance, a normal Axes would have a left margin that includes the
left ticklabels, and the ylabel if it exists.  The right margin may include a
colorbar, the bottom margin the xaxis decorations, and the top margin the
title.

With these constraints, the solver then finds appropriate bounds for the
columns and rows.  It's possible that the margins take up the whole figure,
in which case the algorithm is not applied and a warning is raised.

See the tutorial :ref:`constrainedlayout_guide`
for more discussion of the algorithm with examples.
"""

import logging  # 导入日志记录模块

import numpy as np  # 导入数值计算模块

from matplotlib import _api, artist as martist  # 导入 Matplotlib 内部 API 和 artist 模块
import matplotlib.transforms as mtransforms  # 导入转换模块
import matplotlib._layoutgrid as mlayoutgrid  # 导入布局网格模块


_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


######################################################
def do_constrained_layout(fig, h_pad, w_pad,
                          hspace=None, wspace=None, rect=(0, 0, 1, 1),
                          compress=False):
    """
    Do the constrained_layout.  Called at draw time in
     ``figure.constrained_layout()``

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        `.Figure` instance to do the layout in.
    h_pad : float
        Horizontal padding between subplot grids, expressed as a fraction of the subplot width.
    w_pad : float
        Vertical padding between subplot grids, expressed as a fraction of the subplot height.
    hspace : float or None, optional
        Horizontal space between subplots, expressed as a fraction of the average subplot width.
    wspace : float or None, optional
        Vertical space between subplots, expressed as a fraction of the average subplot height.
    rect : tuple (l, b, w, h), optional
        A rectangle (left, bottom, width, height) in normalized figure coordinates that the whole subplots area (including labels) will fit into.
    compress : bool, optional
        Whether to compress any empty space around the subplots.

    Notes
    -----
    This function adjusts subplot layouts so that there are no overlapping Axes or Axes decorations. All Axes decorations are managed, including labels, ticks, titles, and ticklabels. Artists dependent on Axes are also adjusted, such as colorbars and suptitle.

    The layout uses `~matplotlib.gridspec`, with each gridspec having a constraint. Overlapping Axes are possible if gridspecs overlap, such as with `~matplotlib.gridspec.GridSpecFromSubplotSpec`. Axes created with `figure.subplots()` or `figure.add_subplots()` are included in the layout, but manually placed Axes with `figure.add_axes()` are not.

    See the Tutorial: :ref:`constrainedlayout_guide` for more details.
    """
    pass  # Placeholder function, actual implementation would follow
    h_pad, w_pad : float
      Padding around the Axes elements in figure-normalized units.
    hspace, wspace : float
       Fraction of the figure to dedicate to space between the
       Axes.  These are evenly spread between the gaps between the Axes.
       A value of 0.2 for a three-column layout would have a space
       of 0.1 of the figure width between each column.
       If h/wspace < h/w_pad, then the pads are used instead.
    rect : tuple of 4 floats
        Rectangle in figure coordinates to perform constrained layout in
        [left, bottom, width, height], each from 0-1.
    compress : bool
        Whether to shift Axes so that white space in between them is
        removed. This is useful for simple grids of fixed-aspect Axes (e.g.
        a grid of images).
    Returns
    -------
    layoutgrid : private debugging structure
    """
    # 获取图形的渲染器对象
    renderer = fig._get_renderer()
    # 调用 make_layoutgrids 函数生成布局网格结构
    layoutgrids = make_layoutgrids(fig, None, rect=rect)
    # 如果没有生成任何布局网格
    if not layoutgrids['hasgrids']:
        # 输出警告信息，说明没有使用 "figure" 关键字调用父级 GridSpec
        _api.warn_external('There are no gridspecs with layoutgrids. '
                           'Possibly did not call parent GridSpec with the'
                           ' "figure" keyword')
        # 函数返回，不返回任何结果
        return
    # 迭代两次算法。必须进行两次迭代，因为修饰物在第一次重新定位后大小会改变（例如 x/y 轴标签会变大/变小）。
    # 第二次重新定位通常较为温和，这样做可以使事情正常运行。
    for _ in range(2):
        # 为图中的所有 Axes 和子图添加边距。为 colorbar 添加边距...
        make_layout_margins(layoutgrids, fig, renderer, h_pad=h_pad,
                            w_pad=w_pad, hspace=hspace, wspace=wspace)
        # 为图中的超级标题添加边距。
        make_margin_suptitles(layoutgrids, fig, renderer, h_pad=h_pad,
                              w_pad=w_pad)

        # 如果布局使得某些列（或行）的边距没有约束，我们需要确保网格中所有这样的实例的边距大小匹配。
        match_submerged_margins(layoutgrids, fig)

        # 更新布局中的所有变量。
        layoutgrids[fig].update_variables()

        # 如果检测到没有折叠的轴，则执行以下操作。
        warn_collapsed = ('constrained_layout not applied because '
                          'axes sizes collapsed to zero.  Try making '
                          'figure larger or Axes decorations smaller.')
        if check_no_collapsed_axes(layoutgrids, fig):
            # 重新定位所有轴。
            reposition_axes(layoutgrids, fig, renderer, h_pad=h_pad,
                            w_pad=w_pad, hspace=hspace, wspace=wspace)
            # 如果开启了压缩选项，则压缩具有固定纵横比的布局。
            if compress:
                layoutgrids = compress_fixed_aspect(layoutgrids, fig)
                # 更新布局中的所有变量。
                layoutgrids[fig].update_variables()
                # 如果检测到没有折叠的轴，则再次重新定位所有轴。
                if check_no_collapsed_axes(layoutgrids, fig):
                    reposition_axes(layoutgrids, fig, renderer, h_pad=h_pad,
                                    w_pad=w_pad, hspace=hspace, wspace=wspace)
                else:
                    # 如果轴折叠，则发出警告。
                    _api.warn_external(warn_collapsed)
        else:
            # 如果轴折叠，则发出警告。
            _api.warn_external(warn_collapsed)
        
        # 重置布局中的边距。
        reset_margins(layoutgrids, fig)

    # 返回更新后的布局网格。
    return layoutgrids
# 创建布局网格树结构的函数，用于为图形（fig）创建布局网格以设置边距

def make_layoutgrids(fig, layoutgrids, rect=(0, 0, 1, 1)):
    """
    Make the layoutgrid tree.

    (Sub)Figures get a layoutgrid so we can have figure margins.

    Gridspecs that are attached to Axes get a layoutgrid so Axes
    can have margins.
    """

    # 如果未提供 layoutgrids 参数，则初始化为空字典，表示没有布局网格
    if layoutgrids is None:
        layoutgrids = dict()
        layoutgrids['hasgrids'] = False
    
    # 如果 fig 没有 '_parent' 属性，表示为顶层图形，使用 rect 参数作为父节点
    if not hasattr(fig, '_parent'):
        # 为顶层图形创建一个布局网格，允许用户指定边距
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(parent=rect, name='figlb')
    else:
        # 如果是子图
        gs = fig._subplotspec.get_gridspec()
        # 可能包含此子图的 gridspec 还未添加到树中，递归调用以确保添加
        layoutgrids = make_layoutgrids_gs(layoutgrids, gs)
        # 为子图添加布局网格
        parentlb = layoutgrids[gs]
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(
            parent=parentlb,
            name='panellb',
            parent_inner=True,
            nrows=1, ncols=1,
            parent_pos=(fig._subplotspec.rowspan,
                        fig._subplotspec.colspan))
    
    # 递归处理该图形中的所有子图
    for sfig in fig.subfigs:
        layoutgrids = make_layoutgrids(sfig, layoutgrids)

    # 遍历该图形中的每个局部轴（Axes），添加其 gridspec
    for ax in fig._localaxes:
        gs = ax.get_gridspec()
        if gs is not None:
            layoutgrids = make_layoutgrids_gs(layoutgrids, gs)

    return layoutgrids


def make_layoutgrids_gs(layoutgrids, gs):
    """
    Make the layoutgrid for a gridspec (and anything nested in the gridspec)
    """

    # 如果 gridspec 已经在 layoutgrids 中存在或者其所属的 figure 为 None，则直接返回 layoutgrids
    if gs in layoutgrids or gs.figure is None:
        return layoutgrids
    
    # 如果需要进行 constrained_layout，则至少需要在树中有一个 gridspec
    layoutgrids['hasgrids'] = True
    
    # 如果 gridspec 没有 '_subplot_spec' 属性，表示为普通的 gridspec
    if not hasattr(gs, '_subplot_spec'):
        parent = layoutgrids[gs.figure]
        # 为普通的 gridspec 创建布局网格
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(
                parent=parent,
                parent_inner=True,
                name='gridspec',
                ncols=gs._ncols, nrows=gs._nrows,
                width_ratios=gs.get_width_ratios(),
                height_ratios=gs.get_height_ratios())
    else:
        # 如果是 gridspecfromsubplotspec 类型：
        # 获取 subplot_spec 对象
        subplot_spec = gs._subplot_spec
        # 获取父 gridspec
        parentgs = subplot_spec.get_gridspec()
        
        # 如果父 gridspec 不在 layoutgrids 中，需要创建它
        if parentgs not in layoutgrids:
            layoutgrids = make_layoutgrids_gs(layoutgrids, parentgs)
        
        # 获取子 gridspec 的位置信息
        subspeclb = layoutgrids[parentgs]
        
        # gridspecfromsubplotspec 需要一个外部容器：
        # 获取唯一表示：
        rep = (gs, 'top')
        
        # 如果 rep 不在 layoutgrids 中，创建一个新的 LayoutGrid 对象
        if rep not in layoutgrids:
            layoutgrids[rep] = mlayoutgrid.LayoutGrid(
                parent=subspeclb,
                name='top',
                nrows=1, ncols=1,
                parent_pos=(subplot_spec.rowspan, subplot_spec.colspan))
        
        # 创建当前 gridspec 的 LayoutGrid 对象
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(
                parent=layoutgrids[rep],
                name='gridspec',
                nrows=gs._nrows, ncols=gs._ncols,
                width_ratios=gs.get_width_ratios(),
                height_ratios=gs.get_height_ratios())
    
    # 返回 layoutgrids 字典作为结果
    return layoutgrids
def check_no_collapsed_axes(layoutgrids, fig):
    """
    Check that no Axes have collapsed to zero size.

    Args:
        layoutgrids (dict): A dictionary mapping gridspecs to layout grids.
        fig (Figure): The Figure object containing subfigures and axes.

    Returns:
        bool: True if no axes have collapsed, False otherwise.
    """
    # Iterate over subfigures recursively
    for sfig in fig.subfigs:
        # Check recursively for collapsed axes in subfigures
        ok = check_no_collapsed_axes(layoutgrids, sfig)
        if not ok:
            return False
    
    # Iterate over each axis in the current figure
    for ax in fig.axes:
        # Get the gridspec associated with the current axis
        gs = ax.get_gridspec()
        if gs in layoutgrids:  # also implies gs is not None.
            lg = layoutgrids[gs]
            # Iterate over each cell in the gridspec
            for i in range(gs.nrows):
                for j in range(gs.ncols):
                    # Get the inner bounding box for the cell
                    bb = lg.get_inner_bbox(i, j)
                    # Check if the width or height of the bounding box is zero or negative
                    if bb.width <= 0 or bb.height <= 0:
                        return False
    
    # If all axes are valid (not collapsed), return True
    return True


def compress_fixed_aspect(layoutgrids, fig):
    """
    Compresses layout to ensure fixed aspect ratio.

    Args:
        layoutgrids (dict): A dictionary mapping gridspecs to layout grids.
        fig (Figure): The Figure object to be compressed.

    Returns:
        dict: The updated layoutgrids dictionary after compression.
    
    Raises:
        ValueError: If Axes are not all from the same gridspec or no Axes
                    are part of a gridspec.
    """
    gs = None
    # Iterate over each axis in the figure
    for ax in fig.axes:
        if ax.get_subplotspec() is None:
            continue
        ax.apply_aspect()
        sub = ax.get_subplotspec()
        _gs = sub.get_gridspec()
        # Initialize gridspec if not already set
        if gs is None:
            gs = _gs
            extraw = np.zeros(gs.ncols)
            extrah = np.zeros(gs.nrows)
        # Check if all axes are from the same gridspec
        elif _gs != gs:
            raise ValueError('Cannot do compressed layout if Axes are not'
                             'all from the same gridspec')
        
        # Calculate differences in width and height
        orig = ax.get_position(original=True)
        actual = ax.get_position(original=False)
        dw = orig.width - actual.width
        if dw > 0:
            extraw[sub.colspan] = np.maximum(extraw[sub.colspan], dw)
        dh = orig.height - actual.height
        if dh > 0:
            extrah[sub.rowspan] = np.maximum(extrah[sub.rowspan], dh)

    # Ensure at least one axes is part of a gridspec
    if gs is None:
        raise ValueError('Cannot do compressed layout if no Axes '
                         'are part of a gridspec.')
    
    # Compute margins based on extraw and extrah
    w = np.sum(extraw) / 2
    layoutgrids[fig].edit_margin_min('left', w)
    layoutgrids[fig].edit_margin_min('right', w)

    h = np.sum(extrah) / 2
    layoutgrids[fig].edit_margin_min('top', h)
    layoutgrids[fig].edit_margin_min('bottom', h)
    
    return layoutgrids


def get_margin_from_padding(obj, *, w_pad=0, h_pad=0,
                            hspace=0, wspace=0):
    """
    Computes margin values based on padding and spacing parameters.

    Args:
        obj: An object containing subplot specifications.
        w_pad (float): Width padding.
        h_pad (float): Height padding.
        hspace (float): Vertical space between subplots.
        wspace (float): Horizontal space between subplots.

    Returns:
        dict: A dictionary containing computed margin values.
    """
    ss = obj._subplotspec
    gs = ss.get_gridspec()

    # Determine the horizontal and vertical space
    if hasattr(gs, 'hspace'):
        _hspace = (gs.hspace if gs.hspace is not None else hspace)
        _wspace = (gs.wspace if gs.wspace is not None else wspace)
    else:
        _hspace = (gs._hspace if gs._hspace is not None else hspace)
        _wspace = (gs._wspace if gs._wspace is not None else wspace)

    _wspace = _wspace / 2
    _hspace = _hspace / 2

    nrows, ncols = gs.get_geometry()
    # Define margins for pads and colorbars, and for Axes decorations
    margin = {'leftcb': w_pad, 'rightcb': w_pad,
              'bottomcb': h_pad, 'topcb': h_pad,
              'left': 0, 'right': 0,
              'top': 0, 'bottom': 0}
    # 如果列之间的空白区域大于水平填充宽度
    if _wspace / ncols > w_pad:
        # 如果单元格起始列号大于0，则左侧添加左侧边距
        if ss.colspan.start > 0:
            margin['leftcb'] = _wspace / ncols
        # 如果单元格结束列号小于总列数，则右侧添加右侧边距
        if ss.colspan.stop < ncols:
            margin['rightcb'] = _wspace / ncols
    # 如果行之间的空白区域大于垂直填充高度
    if _hspace / nrows > h_pad:
        # 如果单元格结束行号小于总行数，则底部添加底部边距
        if ss.rowspan.stop < nrows:
            margin['bottomcb'] = _hspace / nrows
        # 如果单元格起始行号大于0，则顶部添加顶部边距
        if ss.rowspan.start > 0:
            margin['topcb'] = _hspace / nrows

    # 返回计算后的边距字典
    return margin
def make_layout_margins(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0,
                        hspace=0, wspace=0):
    """
    对每个 Axes，使得 *pos* 布局框与 *axes* 布局框之间的边距至少能容纳轴上的装饰物。

    然后为色条留出空间。

    Parameters
    ----------
    layoutgrids : dict
        存储网格规格的字典。
    fig : `~matplotlib.figure.Figure`
        进行布局的 `.Figure` 实例。
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        要使用的渲染器。
    w_pad, h_pad : float, default: 0
        宽度和高度填充（作为图形的一部分的比例）。
    hspace, wspace : float, default: 0
        宽度和高度填充，作为图形大小的一部分，由列或行的数量除以。

    """
    for sfig in fig.subfigs:  # 递归地设置子面板的边距
        ss = sfig._subplotspec
        gs = ss.get_gridspec()

        make_layout_margins(layoutgrids, sfig, renderer,
                            w_pad=w_pad, h_pad=h_pad,
                            hspace=hspace, wspace=wspace)

        # 获取子图的边距，基于填充参数
        margins = get_margin_from_padding(sfig, w_pad=0, h_pad=0,
                                          hspace=hspace, wspace=wspace)
        # 编辑网格规格以设置外边距的最小值
        layoutgrids[gs].edit_outer_margin_mins(margins, ss)

    # 为图例设置图形级别的边距：
    for leg in fig.legends:
        inv_trans_fig = None
        if leg._outside_loc and leg._bbox_to_anchor is None:
            if inv_trans_fig is None:
                inv_trans_fig = fig.transFigure.inverted().transform_bbox
            bbox = inv_trans_fig(leg.get_tightbbox(renderer))
            w = bbox.width + 2 * w_pad
            h = bbox.height + 2 * h_pad
            legendloc = leg._outside_loc
            if legendloc == 'lower':
                layoutgrids[fig].edit_margin_min('bottom', h)
            elif legendloc == 'upper':
                layoutgrids[fig].edit_margin_min('top', h)
            if legendloc == 'right':
                layoutgrids[fig].edit_margin_min('right', w)
            elif legendloc == 'left':
                layoutgrids[fig].edit_margin_min('left', w)


def make_margin_suptitles(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0):
    """
    计算超级标题的大小，并使顶层图形的边距更大。

    Parameters
    ----------
    layoutgrids : dict
        存储网格规格的字典。
    fig : `~matplotlib.figure.Figure`
        进行布局的 `.Figure` 实例。
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        要使用的渲染器。
    w_pad, h_pad : float, default: 0
        宽度和高度填充（作为图形的一部分的比例）。

    """
    # 将图形坐标系反转，并将填充转换为局部子图坐标系中的距离
    inv_trans_fig = fig.transFigure.inverted().transform_bbox
    padbox = mtransforms.Bbox([[0, 0], [w_pad, h_pad]])
    padbox = (fig.transFigure -
              fig.transSubfigure).transform_bbox(padbox)
    h_pad_local = padbox.height
    w_pad_local = padbox.width

    for sfig in fig.subfigs:
        make_margin_suptitles(layoutgrids, sfig, renderer,
                              w_pad=w_pad, h_pad=h_pad)
    # 检查是否存在超标题，并且超标题设置为布局中显示
    if fig._suptitle is not None and fig._suptitle.get_in_layout():
        # 获取当前超标题的位置
        p = fig._suptitle.get_position()
        # 如果超标题被自动定位标记所控制
        if getattr(fig._suptitle, '_autopos', False):
            # 设置超标题的位置，使其位于顶部，考虑局部垂直间距
            fig._suptitle.set_position((p[0], 1 - h_pad_local))
            # 获取超标题边界框的反向转换
            bbox = inv_trans_fig(fig._suptitle.get_tightbbox(renderer))
            # 编辑图表网格布局中顶部边距的最小值，考虑边界框高度和局部垂直间距
            layoutgrids[fig].edit_margin_min('top', bbox.height + 2 * h_pad)

    # 检查是否存在超X标签，并且超X标签设置为布局中显示
    if fig._supxlabel is not None and fig._supxlabel.get_in_layout():
        # 获取当前超X标签的位置
        p = fig._supxlabel.get_position()
        # 如果超X标签被自动定位标记所控制
        if getattr(fig._supxlabel, '_autopos', False):
            # 设置超X标签的位置，使其位于底部，考虑局部垂直间距
            fig._supxlabel.set_position((p[0], h_pad_local))
            # 获取超X标签边界框的反向转换
            bbox = inv_trans_fig(fig._supxlabel.get_tightbbox(renderer))
            # 编辑图表网格布局中底部边距的最小值，考虑边界框高度和局部垂直间距
            layoutgrids[fig].edit_margin_min('bottom',
                                             bbox.height + 2 * h_pad)

    # 检查是否存在超Y标签，并且超Y标签设置为布局中显示
    if fig._supylabel is not None and fig._supylabel.get_in_layout():
        # 获取当前超Y标签的位置
        p = fig._supylabel.get_position()
        # 如果超Y标签被自动定位标记所控制
        if getattr(fig._supylabel, '_autopos', False):
            # 设置超Y标签的位置，使其位于左侧，考虑局部水平间距
            fig._supylabel.set_position((w_pad_local, p[1]))
            # 获取超Y标签边界框的反向转换
            bbox = inv_trans_fig(fig._supylabel.get_tightbbox(renderer))
            # 编辑图表网格布局中左侧边距的最小值，考虑边界框宽度和局部水平间距
            layoutgrids[fig].edit_margin_min('left', bbox.width + 2 * w_pad)
def match_submerged_margins(layoutgrids, fig):
    """
    Make the margins that are submerged inside an Axes the same size.

    This allows Axes that span two columns (or rows) that are offset
    from one another to have the same size.

    This gives the proper layout for something like::
        fig = plt.figure(constrained_layout=True)
        axs = fig.subplot_mosaic("AAAB\nCCDD")

    Without this routine, the Axes D will be wider than C, because the
    margin width between the two columns in C has no width by default,
    whereas the margins between the two columns of D are set by the
    width of the margin between A and B. However, obviously the user would
    like C and D to be the same size, so we need to add constraints to these
    "submerged" margins.

    This routine makes all the interior margins the same, and the spacing
    between the three columns in A and the two column in C are all set to the
    margins between the two columns of D.

    See test_constrained_layout::test_constrained_layout12 for an example.
    """

    # 对于图形中的每个子图（subfig），递归调用匹配内嵌边距的函数
    for sfig in fig.subfigs:
        match_submerged_margins(layoutgrids, sfig)

    # 获取图形中所有具有子图规格并且在布局中的坐标轴（Axes）对象
    axs = [a for a in fig.get_axes()
           if a.get_subplotspec() is not None and a.get_in_layout()]
    # 遍历 axs 列表中的每个子图对象 ax1
    for ax1 in axs:
        # 获取当前子图对象 ax1 的子图规格
        ss1 = ax1.get_subplotspec()
        # 如果当前子图的子图网格不在 layoutgrids 字典中，则移除该子图对象并继续下一个循环
        if ss1.get_gridspec() not in layoutgrids:
            axs.remove(ax1)
            continue
        # 获取当前子图 ax1 的布局网格对象 lg1
        lg1 = layoutgrids[ss1.get_gridspec()]

        # 处理内部列边距：
        if len(ss1.colspan) > 1:
            # 计算内部列左边距的最大值
            maxsubl = np.max(
                lg1.margin_vals['left'][ss1.colspan[1:]] +
                lg1.margin_vals['leftcb'][ss1.colspan[1:]]
            )
            # 计算内部列右边距的最大值
            maxsubr = np.max(
                lg1.margin_vals['right'][ss1.colspan[:-1]] +
                lg1.margin_vals['rightcb'][ss1.colspan[:-1]]
            )
            # 再次遍历 axs 列表中的每个子图对象 ax2
            for ax2 in axs:
                # 获取当前子图对象 ax2 的子图规格
                ss2 = ax2.get_subplotspec()
                # 获取当前子图 ax2 的布局网格对象 lg2
                lg2 = layoutgrids[ss2.get_gridspec()]
                # 如果 lg2 不为空且当前子图对象 ss2 的列跨度大于 1
                if lg2 is not None and len(ss2.colspan) > 1:
                    # 计算内部列左边距的最大值
                    maxsubl2 = np.max(
                        lg2.margin_vals['left'][ss2.colspan[1:]] +
                        lg2.margin_vals['leftcb'][ss2.colspan[1:]]
                    )
                    # 更新 maxsubl 变量为较大的值
                    if maxsubl2 > maxsubl:
                        maxsubl = maxsubl2
                    # 计算内部列右边距的最大值
                    maxsubr2 = np.max(
                        lg2.margin_vals['right'][ss2.colspan[:-1]] +
                        lg2.margin_vals['rightcb'][ss2.colspan[:-1]]
                    )
                    # 更新 maxsubr 变量为较大的值
                    if maxsubr2 > maxsubr:
                        maxsubr = maxsubr2
            # 对 ss1 的列跨度中除了第一个和最后一个的每个列应用最大边距值
            for i in ss1.colspan[1:]:
                lg1.edit_margin_min('left', maxsubl, cell=i)
            for i in ss1.colspan[:-1]:
                lg1.edit_margin_min('right', maxsubr, cell=i)

        # 处理内部行边距：
        if len(ss1.rowspan) > 1:
            # 计算内部行顶边距的最大值
            maxsubt = np.max(
                lg1.margin_vals['top'][ss1.rowspan[1:]] +
                lg1.margin_vals['topcb'][ss1.rowspan[1:]]
            )
            # 计算内部行底边距的最大值
            maxsubb = np.max(
                lg1.margin_vals['bottom'][ss1.rowspan[:-1]] +
                lg1.margin_vals['bottomcb'][ss1.rowspan[:-1]]
            )
            # 再次遍历 axs 列表中的每个子图对象 ax2
            for ax2 in axs:
                # 获取当前子图对象 ax2 的子图规格
                ss2 = ax2.get_subplotspec()
                # 获取当前子图 ax2 的布局网格对象 lg2
                lg2 = layoutgrids[ss2.get_gridspec()]
                # 如果 lg2 不为空且当前子图对象 ss2 的行跨度大于 1
                if lg2 is not None:
                    if len(ss2.rowspan) > 1:
                        # 更新 maxsubt 变量为包含 lg2 边距的更大值
                        maxsubt = np.max([np.max(
                            lg2.margin_vals['top'][ss2.rowspan[1:]] +
                            lg2.margin_vals['topcb'][ss2.rowspan[1:]]
                        ), maxsubt])
                        # 更新 maxsubb 变量为包含 lg2 边距的更大值
                        maxsubb = np.max([np.max(
                            lg2.margin_vals['bottom'][ss2.rowspan[:-1]] +
                            lg2.margin_vals['bottomcb'][ss2.rowspan[:-1]]
                        ), maxsubb])
            # 对 ss1 的行跨度中除了第一个和最后一个的每个行应用最大边距值
            for i in ss1.rowspan[1:]:
                lg1.edit_margin_min('top', maxsubt, cell=i)
            for i in ss1.rowspan[:-1]:
                lg1.edit_margin_min('bottom', maxsubb, cell=i)
# 根据传入的 colorbar Axes 对象确定其所属的子图规格

def get_cb_parent_spans(cbax):
    """
    Figure out which subplotspecs this colorbar belongs to.

    Parameters
    ----------
    cbax : `~matplotlib.axes.Axes`
        颜色条所在的 Axes 对象.
    """
    # 初始化行和列的起始和终止索引
    rowstart = np.inf
    rowstop = -np.inf
    colstart = np.inf
    colstop = -np.inf
    
    # 遍历每一个父级对象，获取其子图规格
    for parent in cbax._colorbar_info['parents']:
        ss = parent.get_subplotspec()
        # 更新行和列的起始和终止索引
        rowstart = min(ss.rowspan.start, rowstart)
        rowstop = max(ss.rowspan.stop, rowstop)
        colstart = min(ss.colspan.start, colstart)
        colstop = max(ss.colspan.stop, colstop)

    # 根据计算得到的行和列索引范围，创建范围对象
    rowspan = range(rowstart, rowstop)
    colspan = range(colstart, colstop)
    return rowspan, colspan


# 获取 Axes 的位置和边界框信息

def get_pos_and_bbox(ax, renderer):
    """
    Get the position and the bbox for the Axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        要获取位置和边界框信息的 Axes 对象.
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        渲染器对象.

    Returns
    -------
    pos : `~matplotlib.transforms.Bbox`
        位置信息，以图形坐标表示.
    bbox : `~matplotlib.transforms.Bbox`
        边界框信息，以图形坐标表示.
    """
    # 获取图形对象
    fig = ax.figure
    # 获取原始的位置信息
    pos = ax.get_position(original=True)
    # 将位置信息从面板坐标转换到图形坐标
    pos = pos.transformed(fig.transSubfigure - fig.transFigure)
    # 获取仅用于布局的紧凑边界框
    tightbbox = martist._get_tightbbox_for_layout_only(ax, renderer)
    # 如果紧凑边界框为 None，则使用位置信息作为边界框
    if tightbbox is None:
        bbox = pos
    else:
        # 否则，将紧凑边界框转换为图形坐标
        bbox = tightbbox.transformed(fig.transFigure.inverted())
    return pos, bbox


# 基于新的内部边界框重新定位所有的 Axes 对象

def reposition_axes(layoutgrids, fig, renderer, *,
                    w_pad=0, h_pad=0, hspace=0, wspace=0):
    """
    Reposition all the Axes based on the new inner bounding box.
    根据新的内部边界框重新定位所有的 Axes 对象.
    """
    # 计算图形到子图的变换矩阵
    trans_fig_to_subfig = fig.transFigure - fig.transSubfigure
    # 遍历每个子图对象
    for sfig in fig.subfigs:
        # 获取外部边界框
        bbox = layoutgrids[sfig].get_outer_bbox()
        # 使用新的变换矩阵重新计算相对图形的转换
        sfig._redo_transform_rel_fig(
            bbox=bbox.transformed(trans_fig_to_subfig))
        # 递归调用，重新定位子图对象内部的所有 Axes 对象
        reposition_axes(layoutgrids, sfig, renderer,
                        w_pad=w_pad, h_pad=h_pad,
                        wspace=wspace, hspace=hspace)
    # 遍历图形对象的本地轴列表
    for ax in fig._localaxes:
        # 如果轴没有子图规范或者不在布局中，则继续下一轴
        if ax.get_subplotspec() is None or not ax.get_in_layout():
            continue

        # 获取轴的子图规范
        ss = ax.get_subplotspec()
        # 获取子图规范的网格规范
        gs = ss.get_gridspec()
        
        # 如果网格规范不在布局网格字典中，则返回
        if gs not in layoutgrids:
            return

        # 获取轴在布局中的内部边界框
        bbox = layoutgrids[gs].get_inner_bbox(rows=ss.rowspan,
                                              cols=ss.colspan)

        # 将边界框从图形坐标系转换到子图坐标系，为了设置位置
        newbbox = trans_fig_to_subfig.transform_bbox(bbox)
        ax._set_position(newbbox)

        # 移动颜色条：
        # 如果存在多个颜色条，需要跟踪旧的宽度和高度
        offset = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
        for nn, cbax in enumerate(ax._colorbars[::-1]):
            # 如果轴是颜色条信息中的父轴
            if ax == cbax._colorbar_info['parents'][0]:
                # 重新定位颜色条
                reposition_colorbar(layoutgrids, cbax, renderer,
                                    offset=offset)
def reposition_colorbar(layoutgrids, cbax, renderer, *, offset=None):
    """
    Place the colorbar in its new place.

    Parameters
    ----------
    layoutgrids : dict
        包含子图网格布局信息的字典。
    cbax : `~matplotlib.axes.Axes`
        颜色条的坐标轴对象。
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        使用的渲染器。
    offset : array-like
        需要调整颜色条的偏移量，以考虑多个颜色条。

    """

    parents = cbax._colorbar_info['parents']
    gs = parents[0].get_gridspec()
    fig = cbax.figure
    trans_fig_to_subfig = fig.transFigure - fig.transSubfigure

    cb_rspans, cb_cspans = get_cb_parent_spans(cbax)
    bboxparent = layoutgrids[gs].get_bbox_for_cb(rows=cb_rspans,
                                                 cols=cb_cspans)
    pb = layoutgrids[gs].get_inner_bbox(rows=cb_rspans, cols=cb_cspans)

    location = cbax._colorbar_info['location']
    anchor = cbax._colorbar_info['anchor']
    fraction = cbax._colorbar_info['fraction']
    aspect = cbax._colorbar_info['aspect']
    shrink = cbax._colorbar_info['shrink']

    cbpos, cbbbox = get_pos_and_bbox(cbax, renderer)

    # Colorbar gets put at extreme edge of outer bbox of the subplotspec
    # It needs to be moved in by: 1) a pad 2) its "margin" 3) by
    # any colorbars already added at this location:
    cbpad = colorbar_get_pad(layoutgrids, cbax)
    if location in ('left', 'right'):
        # fraction and shrink are fractions of parent
        pbcb = pb.shrunk(fraction, shrink).anchored(anchor, pb)
        # The colorbar is at the left side of the parent.  Need
        # to translate to right (or left)
        if location == 'right':
            lmargin = cbpos.x0 - cbbbox.x0
            dx = bboxparent.x1 - pbcb.x0 + offset['right']
            dx += cbpad + lmargin
            offset['right'] += cbbbox.width + cbpad
            pbcb = pbcb.translated(dx, 0)
        else:
            lmargin = cbpos.x0 - cbbbox.x0
            dx = bboxparent.x0 - pbcb.x0  # edge of parent
            dx += -cbbbox.width - cbpad + lmargin - offset['left']
            offset['left'] += cbbbox.width + cbpad
            pbcb = pbcb.translated(dx, 0)
    else:  # horizontal axes:
        pbcb = pb.shrunk(shrink, fraction).anchored(anchor, pb)
        if location == 'top':
            bmargin = cbpos.y0 - cbbbox.y0
            dy = bboxparent.y1 - pbcb.y0 + offset['top']
            dy += cbpad + bmargin
            offset['top'] += cbbbox.height + cbpad
            pbcb = pbcb.translated(0, dy)
        else:
            bmargin = cbpos.y0 - cbbbox.y0
            dy = bboxparent.y0 - pbcb.y0
            dy += -cbbbox.height - cbpad + bmargin - offset['bottom']
            offset['bottom'] += cbbbox.height + cbpad
            pbcb = pbcb.translated(0, dy)

    pbcb = trans_fig_to_subfig.transform_bbox(pbcb)
    cbax.set_transform(fig.transSubfigure)
    cbax._set_position(pbcb)
    cbax.set_anchor(anchor)
    # 如果位置是在底部或顶部，调整纵横比以适应设定值
    if location in ['bottom', 'top']:
        aspect = 1 / aspect
    
    # 设置颜色条轴的盒子纵横比
    cbax.set_box_aspect(aspect)
    
    # 将颜色条轴的纵横比设置为自动调整
    cbax.set_aspect('auto')
    
    # 返回偏移量
    return offset
# 重置布局网格中所有子图的边距，确保它们可以根据需要重新增长
def reset_margins(layoutgrids, fig):
    """
    Reset the margins in the layoutboxes of *fig*.

    Margins are usually set as a minimum, so if the figure gets smaller
    the minimum needs to be zero in order for it to grow again.
    """
    # 遍历图形对象的所有子图
    for sfig in fig.subfigs:
        reset_margins(layoutgrids, sfig)
    
    # 遍历图形对象的所有坐标轴
    for ax in fig.axes:
        # 检查坐标轴是否在布局中
        if ax.get_in_layout():
            # 获取坐标轴所在的网格规范
            gs = ax.get_gridspec()
            # 如果该网格规范在布局网格字典中，说明需要重置边距
            if gs in layoutgrids:  # also implies gs is not None.
                layoutgrids[gs].reset_margins()
    
    # 对整个图形对象的布局网格执行边距重置
    layoutgrids[fig].reset_margins()


# 获取颜色条对象的填充值，基于其父网格规范
def colorbar_get_pad(layoutgrids, cax):
    parents = cax._colorbar_info['parents']
    # 获取颜色条的父网格规范
    gs = parents[0].get_gridspec()

    # 获取颜色条的行和列跨度
    cb_rspans, cb_cspans = get_cb_parent_spans(cax)
    # 获取颜色条外边界框的内部边界框
    bboxouter = layoutgrids[gs].get_inner_bbox(rows=cb_rspans, cols=cb_cspans)

    # 如果颜色条的位置在左侧或右侧，则使用宽度作为尺寸
    if cax._colorbar_info['location'] in ['right', 'left']:
        size = bboxouter.width
    else:
        size = bboxouter.height

    # 返回颜色条填充值乘以尺寸
    return cax._colorbar_info['pad'] * size
```