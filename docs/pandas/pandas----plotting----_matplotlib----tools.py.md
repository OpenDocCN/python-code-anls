# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\tools.py`

```
# 导入未来版本的 annotations，支持类型提示中的 Forward references
from __future__ import annotations

# 导入 ceil 函数，用于向上取整
from math import ceil

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入警告模块
import warnings

# 导入 matplotlib 库
import matplotlib as mpl

# 导入 numpy 库
import numpy as np

# 导入 pandas 中的异常处理函数
from pandas.util._exceptions import find_stack_level

# 导入 pandas 中的数据类型相关函数
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

# 如果是类型检查模式，则导入额外的类型
if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterable,
    )

    # 导入 matplotlib 中的图表元素类型
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.table import Table

    # 导入 pandas 中的 DataFrame 和 Series 类型
    from pandas import (
        DataFrame,
        Series,
    )


# 检查是否启用了 constrained_layout，返回是否启用的布尔值
def do_adjust_figure(fig: Figure) -> bool:
    """Whether fig has constrained_layout enabled."""
    if not hasattr(fig, "get_constrained_layout"):
        return False
    return not fig.get_constrained_layout()


# 如果未启用 constrained_layout，调整图表布局
def maybe_adjust_figure(fig: Figure, *args, **kwargs) -> None:
    """Call fig.subplots_adjust unless fig has constrained_layout enabled."""
    if do_adjust_figure(fig):
        fig.subplots_adjust(*args, **kwargs)


# 格式化日期标签，调整标签的水平对齐和旋转角度
def format_date_labels(ax: Axes, rot) -> None:
    """mini version of autofmt_xdate"""
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation(rot)
    fig = ax.get_figure()
    if fig is not None:
        maybe_adjust_figure(fig, bottom=0.2)


# 创建表格，根据输入数据类型选择处理方式，并返回 Table 对象
def table(
    ax, data: DataFrame | Series, rowLabels=None, colLabels=None, **kwargs
) -> Table:
    if isinstance(data, ABCSeries):
        data = data.to_frame()
    elif isinstance(data, ABCDataFrame):
        pass
    else:
        raise ValueError("Input data must be DataFrame or Series")

    if rowLabels is None:
        rowLabels = data.index

    if colLabels is None:
        colLabels = data.columns

    cellText = data.values

    # 使用 matplotlib 的 table 函数创建表格对象
    return mpl.table.table(
        ax,
        cellText=cellText,  # type: ignore[arg-type]
        rowLabels=rowLabels,
        colLabels=colLabels,
        **kwargs,
    )


# 获取布局参数，返回绘制子图的行列数
def _get_layout(
    nplots: int,
    layout: tuple[int, int] | None = None,
    layout_type: str = "box",
) -> tuple[int, int]:
    # 如果给定了布局参数，则进行以下逻辑
    if layout is not None:
        # 检查布局参数是否为元组或列表，并且长度为2
        if not isinstance(layout, (tuple, list)) or len(layout) != 2:
            raise ValueError("Layout must be a tuple of (rows, columns)")

        # 解包布局参数为行数和列数
        nrows, ncols = layout

        # 根据给定的规则调整布局参数
        if nrows == -1 and ncols > 0:
            layout = (ceil(nplots / ncols), ncols)
        elif ncols == -1 and nrows > 0:
            layout = (nrows, ceil(nplots / nrows))
        elif ncols <= 0 and nrows <= 0:
            msg = "At least one dimension of layout must be positive"
            raise ValueError(msg)

        # 更新行数和列数为调整后的值
        nrows, ncols = layout

        # 检查布局是否足够容纳指定数量的子图
        if nrows * ncols < nplots:
            raise ValueError(
                f"Layout of {nrows}x{ncols} must be larger than required size {nplots}"
            )

        # 返回调整后的布局参数
        return layout

    # 根据布局类型返回默认布局
    if layout_type == "single":
        return (1, 1)
    elif layout_type == "horizontal":
        return (1, nplots)
    elif layout_type == "vertical":
        return (nplots, 1)

    # 预定义的特定数量子图的布局
    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}

    # 尝试从预定义的布局中获取对应数量的布局
    try:
        return layouts[nplots]
    except KeyError:
        # 如果预定义布局中不存在，则根据平方根规则计算布局
        k = 1
        while k**2 < nplots:
            k += 1

        # 检查特定条件以确定最终布局
        if (k - 1) * k >= nplots:
            return k, (k - 1)
        else:
            return k, k
# 导入 matplotlib.pyplot 模块，用于创建图形和子图
import matplotlib.pyplot as plt

# 创建一个带有预先创建的子图集的图形。
# 这个实用程序包装函数方便地在单个调用中创建子图的常见布局，包括包含的图形对象。
# 参数 naxes 指定所需的子图数量，超出的子图将被设为不可见。默认为行数乘以列数。
# 参数 sharex 如果为 True，则所有子图共享 X 轴。
# 参数 sharey 如果为 True，则所有子图共享 Y 轴。
# 参数 squeeze 如果为 True，则从返回的轴对象中挤出额外的维度：
#   - 如果只构建一个子图 (nrows=ncols=1)，则返回单个 Axis 对象作为标量。
#   - 对于 Nx1 或 1xN 子图，返回一个 1 维 numpy 对象数组的 Axis 对象。
#   - 对于 NxM 子图（其中 N>1 且 M>1），作为 2 维数组返回。
# 参数 subplot_kw 是传递给 add_subplot() 调用的关键字字典，用于创建每个子图。
# 参数 ax 是 Matplotlib 的轴对象，可选。
# 参数 layout 是子图网格的行数和列数的元组。如果未指定，将从 naxes 和 layout_type 计算得出。
# 参数 layout_type 指定如何布局子图网格，默认为 'box'。
# 参数 fig_kw 是传递给 figure() 调用的其他关键字参数。
def create_subplots(
    naxes: int,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    subplot_kw=None,
    ax=None,
    layout=None,
    layout_type: str = "box",
    **fig_kw,
):
    """
    Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    naxes : int
      Number of required axes. Exceeded axes are set invisible. Default is
      nrows * ncols.

    sharex : bool
      If True, the X axis will be shared amongst all subplots.

    sharey : bool
      If True, the Y axis will be shared amongst all subplots.

    squeeze : bool

      If True, extra dimensions are squeezed out from the returned axis object:
        - if only one subplot is constructed (nrows=ncols=1), the resulting
        single Axis object is returned as a scalar.
        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy object
        array of Axis objects are returned as numpy 1-d arrays.
        - for NxM subplots with N>1 and M>1 are returned as a 2d array.

      If False, no squeezing is done: the returned axis object is always
      a 2-d array containing Axis instances, even if it ends up being 1x1.

    subplot_kw : dict
      Dict with keywords passed to the add_subplot() call used to create each
      subplots.

    ax : Matplotlib axis object, optional

    layout : tuple
      Number of rows and columns of the subplot grid.
      If not specified, calculated from naxes and layout_type

    layout_type : {'box', 'horizontal', 'vertical'}, default 'box'
      Specify how to layout the subplot grid.

    fig_kw : Other keyword arguments to be passed to the figure() call.
        Note that all keywords not recognized above will be
        automatically included here.

    Returns
    -------
    fig, ax : tuple
      - fig is the Matplotlib Figure object
      - ax can be either a single axis object or an array of axis objects if
      more than one subplot was created.  The dimensions of the resulting array
      can be controlled with the squeeze keyword, see above.

    Examples
    --------
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # Just a figure and one subplot
    f, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Two subplots, unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Four polar axes
    plt.subplots(2, 2, subplot_kw=dict(polar=True))
    """
    # 如果 subplot_kw 未指定，则设为一个空字典
    if subplot_kw is None:
        subplot_kw = {}

    # 如果未提供轴对象 ax，则创建一个新的图形对象 fig，使用 fig_kw 中的关键字参数
    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        # 如果传入的 ax 参数不是单个 Axes 对象，而是类似列表的对象
        if is_list_like(ax):
            # 如果需要压缩轴，将多个轴展平为一维数组，并转换为 NumPy 对象
            if squeeze:
                ax = np.fromiter(flatten_axes(ax), dtype=object)
            # 如果指定了 layout 参数，给出警告，因为在传入多个轴时，layout 参数将被忽略
            if layout is not None:
                warnings.warn(
                    "When passing multiple axes, layout keyword is ignored.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
            # 如果指定了 sharex 或 sharey 参数，给出警告，因为在传入多个轴时，这些设置将被忽略
            if sharex or sharey:
                warnings.warn(
                    "When passing multiple axes, sharex and sharey "
                    "are ignored. These settings must be specified when creating axes.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
            # 如果传入的轴数量与所需的数量相同，则返回所属的图形对象和轴数组
            if ax.size == naxes:
                fig = ax.flat[0].get_figure()
                return fig, ax
            else:
                # 否则，引发 ValueError，说明传入的轴数量与输出的绘图数量不匹配
                raise ValueError(
                    f"The number of passed axes must be {naxes}, the "
                    "same as the output plot"
                )

        # 如果传入的是单个 Axes 对象，则获取所属的图形对象
        fig = ax.get_figure()
        # 如果传入的轴数量为 1，并且需要压缩轴，则直接返回图形对象和轴对象
        if naxes == 1:
            if squeeze:
                return fig, ax
            else:
                # 否则，返回图形对象和展平后的轴数组对象
                return fig, np.fromiter(flatten_axes(ax), dtype=object)
        else:
            # 给出警告，因为需要输出多个子图，清除包含传入轴的图形对象
            warnings.warn(
                "To output multiple subplots, the figure containing "
                "the passed axes is being cleared.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            # 清除图形对象
            fig.clear()

    # 计算子图布局的行数和列数
    nrows, ncols = _get_layout(naxes, layout=layout, layout_type=layout_type)
    nplots = nrows * ncols

    # 创建一个空的对象数组，用于存储所有的轴对象
    axarr = np.empty(nplots, dtype=object)

    # 单独创建第一个子图，如果请求共享轴，则设置共享属性
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)

    if sharex:
        subplot_kw["sharex"] = ax0
    if sharey:
        subplot_kw["sharey"] = ax0
    axarr[0] = ax0

    # 注意 MATLAB 的基于 1 的计数约定，因此从 1 开始创建子图
    for i in range(1, nplots):
        kwds = subplot_kw.copy()
        # 对于空白/虚拟轴，设置 sharex 和 sharey 为 None，避免干扰可见轴的正确轴限制
        if i >= naxes:
            kwds["sharex"] = None
            kwds["sharey"] = None
        # 创建子图，并将其存储到轴数组中
        ax = fig.add_subplot(nrows, ncols, i + 1, **kwds)
        axarr[i] = ax

    # 如果传入的轴数量不等于所需的绘图数量，则将超出数量的轴设为不可见
    if naxes != nplots:
        for ax in axarr[naxes:]:
            ax.set_visible(False)

    # 处理共享轴的情况
    handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey)
    # 如果需要压缩（squeeze）数组：
    if squeeze:
        # 将数组重塑为最终期望的维度（nrows, ncols），
        # 丢弃那些等于1的不必要的维度。如果只有一个子图，直接返回它，而不是一个包含一个元素的数组。
        if nplots == 1:
            axes = axarr[0]  # 如果只有一个子图，直接使用第一个元素作为axes
        else:
            axes = axarr.reshape(nrows, ncols).squeeze()  # 重塑数组并压缩到二维
    else:
        # 返回的轴数组始终是二维的，即使 nrows=ncols=1 也是如此
        axes = axarr.reshape(nrows, ncols)  # 将数组重塑为 nrows x ncols 的二维数组

    # 返回图形对象和轴数组
    return fig, axes
# 从给定的 axis 对象中移除所有主刻度标签的可见性
def _remove_labels_from_axis(axis: Axis) -> None:
    # 遍历主刻度标签，并将它们设为不可见
    for t in axis.get_majorticklabels():
        t.set_visible(False)

    # 如果次坐标轴使用了 NullLocator 和 NullFormatter（默认情况下），set_visible 将无效
    # 检查次坐标轴是否使用了 NullLocator，若是，则将其改为 AutoLocator
    if isinstance(axis.get_minor_locator(), mpl.ticker.NullLocator):
        axis.set_minor_locator(mpl.ticker.AutoLocator())
    # 检查次坐标轴是否使用了 NullFormatter，若是，则将其改为空字符串格式化
    if isinstance(axis.get_minor_formatter(), mpl.ticker.NullFormatter):
        axis.set_minor_formatter(mpl.ticker.FormatStrFormatter(""))
    # 将所有次刻度标签设为不可见
    for t in axis.get_minorticklabels():
        t.set_visible(False)

    # 设置坐标轴的标签不可见
    axis.get_label().set_visible(False)


# 判断给定的 ax1 轴是否在比较轴（X轴或Y轴）上被外部共享
def _has_externally_shared_axis(ax1: Axes, compare_axis: str) -> bool:
    """
    返回一个轴是否被外部共享的布尔值。

    参数
    ----------
    ax1 : matplotlib.axes.Axes
        要查询的轴。
    compare_axis : str
        `"x"` 或 `"y"`，根据是否比较X轴或Y轴。

    返回
    -------
    bool
        如果轴被外部共享返回 `True`。否则返回 `False`。

    注意
    -----
    如果两个位置不同的轴共享一个轴，则可以称为*外部*共享公共轴。

    如果共享一个轴的两个轴还具有相同的位置，则可以称为*内部*共享公共轴（也称为twining）。

    _handle_shared_axes() 只关心外部共享公共轴的轴，而不管这些轴是否还与第三个轴内部共享。
    """
    if compare_axis == "x":
        axes = ax1.get_shared_x_axes()
    elif compare_axis == "y":
        axes = ax1.get_shared_y_axes()
    else:
        raise ValueError(
            "_has_externally_shared_axis() 需要 'x' 或 'y' 作为第二个参数"
        )

    # 获取与 ax1 共享轴的所有同级轴
    axes_siblings = axes.get_siblings(ax1)

    # 保留 ax1 和那些与其位置不同的同级轴
    ax1_points = ax1.get_position().get_points()

    for ax2 in axes_siblings:
        # 如果 ax2 与 ax1 的位置不同，则返回 True，表示轴被外部共享
        if not np.array_equal(ax1_points, ax2.get_position().get_points()):
            return True

    # 如果所有同级轴与 ax1 的位置相同，则返回 False，表示轴未被外部共享
    return False


def handle_shared_axes(
    axarr: Iterable[Axes],
    nplots: int,
    naxes: int,
    nrows: int,
    ncols: int,
    sharex: bool,
    sharey: bool,
) -> None:
    # 如果有多个子图（即 nplots > 1），则定义以下三个 lambda 函数来获取子图的行数、列数以及是否为第一列
    row_num = lambda x: x.get_subplotspec().rowspan.start
    col_num = lambda x: x.get_subplotspec().colspan.start
    is_first_col = lambda x: x.get_subplotspec().is_first_col()

    # 如果行数大于 1，则进行以下操作
    if nrows > 1:
        try:
            # 创建一个布尔类型的二维数组 layout 用来表示子图的布局情况（是否可见）
            layout = np.zeros((nrows + 1, ncols + 1), dtype=np.bool_)
            # 遍历所有子图 axarr 中的子图 ax，根据其位置信息更新布局数组 layout
            for ax in axarr:
                layout[row_num(ax), col_num(ax)] = ax.get_visible()

            # 再次遍历所有子图 axarr 中的子图 ax
            for ax in axarr:
                # 只有最后一行的子图应该有 x 轴标签 -> 其它情况通过 off 处理
                if not layout[row_num(ax) + 1, col_num(ax)]:
                    continue
                # 如果 sharex 为真或者子图的 x 轴被外部共享，则移除 x 轴的标签
                if sharex or _has_externally_shared_axis(ax, "x"):
                    _remove_labels_from_axis(ax.xaxis)

        # 处理可能出现的索引错误异常
        except IndexError:
            # 如果使用了 gridspec，ax.rowNum 和 ax.colNum 可能与布局的形状不同
            # 在这种情况下，使用 is_last_row 函数来确定是否为最后一行的子图
            is_last_row = lambda x: x.get_subplotspec().is_last_row()
            # 再次遍历所有子图 axarr 中的子图 ax
            for ax in axarr:
                # 如果是最后一行的子图，则跳过不处理
                if is_last_row(ax):
                    continue
                # 如果 sharex 为真或者子图的 x 轴被外部共享，则移除 x 轴的标签
                if sharex or _has_externally_shared_axis(ax, "x"):
                    _remove_labels_from_axis(ax.xaxis)

    # 如果列数大于 1，则遍历所有子图 axarr 中的子图 ax
    if ncols > 1:
        for ax in axarr:
            # 只有第一列的子图应该有 y 轴标签 -> 其它情况通过 off 处理
            if is_first_col(ax):
                continue
            # 如果 sharey 为真或者子图的 y 轴被外部共享，则移除 y 轴的标签
            if sharey or _has_externally_shared_axis(ax, "y"):
                _remove_labels_from_axis(ax.yaxis)
# 生成器函数，用于将传入的坐标轴对象（单个或可迭代集合）展平为生成器
def flatten_axes(axes: Axes | Iterable[Axes]) -> Generator[Axes, None, None]:
    # 如果传入的 axes 不是可迭代对象，直接生成该 axes
    if not is_list_like(axes):
        yield axes  # type: ignore[misc]
    # 如果 axes 是 ndarray 或者 ABCIndex 类型，将其展平为一维数组后逐个生成
    elif isinstance(axes, (np.ndarray, ABCIndex)):
        yield from np.asarray(axes).reshape(-1)
    # 否则，逐个生成 axes 中的元素
    else:
        yield from axes  # type: ignore[misc]


# 设置坐标轴标签的属性，包括字体大小和旋转角度
def set_ticks_props(
    axes: Axes | Iterable[Axes],
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
):
    # 遍历展平后的所有坐标轴对象
    for ax in flatten_axes(axes):
        # 如果指定了 xlabelsize，设置 x 轴刻度标签的字体大小
        if xlabelsize is not None:
            mpl.artist.setp(ax.get_xticklabels(), fontsize=xlabelsize)  # type: ignore[arg-type]
        # 如果指定了 xrot，设置 x 轴刻度标签的旋转角度
        if xrot is not None:
            mpl.artist.setp(ax.get_xticklabels(), rotation=xrot)  # type: ignore[arg-type]
        # 如果指定了 ylabelsize，设置 y 轴刻度标签的字体大小
        if ylabelsize is not None:
            mpl.artist.setp(ax.get_yticklabels(), fontsize=ylabelsize)  # type: ignore[arg-type]
        # 如果指定了 yrot，设置 y 轴刻度标签的旋转角度
        if yrot is not None:
            mpl.artist.setp(ax.get_yticklabels(), rotation=yrot)  # type: ignore[arg-type]
    # 返回设置后的坐标轴对象
    return axes


# 获取指定坐标轴对象中的所有线对象
def get_all_lines(ax: Axes) -> list[Line2D]:
    # 获取主要坐标轴对象中的所有线对象
    lines = ax.get_lines()

    # 如果坐标轴对象有右边附加坐标轴，获取右边附加坐标轴中的所有线对象并添加到列表中
    if hasattr(ax, "right_ax"):
        lines += ax.right_ax.get_lines()

    # 如果坐标轴对象有左边附加坐标轴，获取左边附加坐标轴中的所有线对象并添加到列表中
    if hasattr(ax, "left_ax"):
        lines += ax.left_ax.get_lines()

    # 返回包含所有线对象的列表
    return lines


# 获取所有线对象的 x 轴数据的范围（最小值和最大值）
def get_xlim(lines: Iterable[Line2D]) -> tuple[float, float]:
    # 初始化最小值和最大值为无穷大和负无穷大
    left, right = np.inf, -np.inf
    # 遍历所有线对象
    for line in lines:
        # 获取当前线对象的 x 轴数据，orig=False 表示获取非原始数据
        x = line.get_xdata(orig=False)
        # 更新最小值和最大值，考虑 NaN 值的情况
        left = min(np.nanmin(x), left)
        right = max(np.nanmax(x), right)
    # 返回 x 轴数据的最小值和最大值
    return left, right
```