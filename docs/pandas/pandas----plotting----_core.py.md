# `D:\src\scipysrc\pandas\pandas\plotting\_core.py`

```
from __future__ import annotations
# 引入了 __future__ 模块的 annotations 特性，支持在函数参数和返回值中使用类型提示

import importlib
# 导入 importlib 模块，用于动态加载模块和重载模块等操作

from typing import (
    TYPE_CHECKING,
    Literal,
)
# 从 typing 模块导入了多个类型相关的工具，包括 TYPE_CHECKING 和 Literal

from pandas._config import get_option
# 从 pandas._config 模块中导入 get_option 函数，用于获取 pandas 的配置选项

from pandas.util._decorators import (
    Appender,
    Substitution,
)
# 从 pandas.util._decorators 模块导入 Appender 和 Substitution 装饰器

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
# 从 pandas.core.dtypes.common 模块导入 is_integer 和 is_list_like 函数

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
# 从 pandas.core.dtypes.generic 模块导入 ABCDataFrame 和 ABCSeries 类

from pandas.core.base import PandasObject
# 从 pandas.core.base 模块导入 PandasObject 基类

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )
    import types
    # 如果在类型检查环境中，导入 collections.abc 模块的 Callable、Hashable 和 Sequence 类型
    from matplotlib.axes import Axes
    import numpy as np
    # 导入 matplotlib.axes 模块的 Axes 类和 numpy 模块的 np 对象
    from pandas._typing import IndexLabel
    # 导入 pandas._typing 模块的 IndexLabel 类型
    from pandas import (
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.groupby.generic import DataFrameGroupBy
    # 导入 pandas 模块的 DataFrame、Index、Series 类和 pandas.core.groupby.generic 模块的 DataFrameGroupBy 类

def holds_integer(column: Index) -> bool:
    # 函数用于判断 Index 类型的列是否包含整数类型数据
    return column.inferred_type in {"integer", "mixed-integer"}

def hist_series(
    self: Series,
    by=None,
    ax=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot: float | None = None,
    ylabelsize: int | None = None,
    yrot: float | None = None,
    figsize: tuple[int, int] | None = None,
    bins: int | Sequence[int] = 10,
    backend: str | None = None,
    legend: bool = False,
    **kwargs,
):
    """
    Draw histogram of the input series using matplotlib.

    Parameters
    ----------
    by : object, optional
        If passed, then used to form histograms for separate groups.
    ax : matplotlib axis object
        If not passed, uses gca().
    grid : bool, default True
        Whether to show axis grid lines.
    xlabelsize : int, default None
        If specified changes the x-axis label size.
    xrot : float, default None
        Rotation of x axis labels.
    ylabelsize : int, default None
        If specified changes the y-axis label size.
    yrot : float, default None
        Rotation of y axis labels.
    figsize : tuple, default None
        Figure size in inches by default.
    bins : int or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    legend : bool, default False
        Whether to show the legend.

    **kwargs
        To be passed to the actual plotting function.

    Returns
    -------
    matplotlib.axes.Axes
        A histogram plot.

    See Also
    --------
    matplotlib.axes.Axes.hist : Plot a histogram using matplotlib.

    Examples
    --------
    For Series:
    """
    # 绘制输入 Series 的直方图，使用 matplotlib
    # 获取当前使用的绘图后端
    plot_backend = _get_plot_backend(backend)
    # 调用绘图后端的 hist_series 方法来绘制系列数据的直方图
    return plot_backend.hist_series(
        self,                 # 调用方法的对象本身，即当前的 Series 对象
        by=by,                # 分组依据，用于分组数据并绘制多个直方图
        ax=ax,                # 指定绘图的坐标轴对象
        grid=grid,            # 是否显示网格线
        xlabelsize=xlabelsize,# X 轴标签的字体大小
        xrot=xrot,            # X 轴标签的旋转角度
        ylabelsize=ylabelsize,# Y 轴标签的字体大小
        yrot=yrot,            # Y 轴标签的旋转角度
        figsize=figsize,      # 绘图的尺寸
        bins=bins,            # 直方图的柱子数量
        legend=legend,        # 是否显示图例
        **kwargs,             # 其他可选参数，传递给底层的绘图方法
    )
# 创建一个函数 hist_frame，用于绘制数据框中每一列的直方图

def hist_frame(
    data: DataFrame,                           # 接收一个 DataFrame 对象作为数据源
    column: IndexLabel | None = None,          # 可选参数，指定要绘制直方图的列名或列名列表
    by=None,                                   # 可选参数，用于按指定对象分组绘制直方图
    grid: bool = True,                         # 是否显示坐标轴网格线，默认为 True
    xlabelsize: int | None = None,             # 可选参数，指定 x 轴标签的字体大小
    xrot: float | None = None,                 # 可选参数，指定 x 轴标签的旋转角度
    ylabelsize: int | None = None,             # 可选参数，指定 y 轴标签的字体大小
    yrot: float | None = None,                 # 可选参数，指定 y 轴标签的旋转角度
    ax=None,                                   # Matplotlib 的 axes 对象，用于绘制直方图
    sharex: bool = False,                      # 是否共享 x 轴，用于子图设置，默认为 False
    sharey: bool = False,                      # 是否共享 y 轴，用于子图设置，默认为 False
    figsize: tuple[int, int] | None = None,    # 可选参数，指定绘图的尺寸大小
    layout: tuple[int, int] | None = None,     # 可选参数，指定直方图在图中的布局（行数，列数）
    bins: int | Sequence[int] = 10,            # 直方图的柱数或柱边界序列，默认为 10
    backend: str | None = None,                # 可选参数，指定绘图的后端
    legend: bool = False,                      # 是否显示图例，默认为 False
    **kwargs,                                  # 其他关键字参数，传递给 Matplotlib 的 hist 函数
):
    """
    Make a histogram of the DataFrame's columns.

    A `histogram`_ is a representation of the distribution of data.
    This function calls :meth:`matplotlib.pyplot.hist`, on each series in
    the DataFrame, resulting in one histogram per column.

    .. _histogram: https://en.wikipedia.org/wiki/Histogram

    Parameters
    ----------
    data : DataFrame
        The pandas object holding the data.
    column : str or sequence, optional
        If passed, will be used to limit data to a subset of columns.
    by : object, optional
        If passed, then used to form histograms for separate groups.
    grid : bool, default True
        Whether to show axis grid lines.
    xlabelsize : int, default None
        If specified changes the x-axis label size.
    xrot : float, default None
        Rotation of x axis labels. For example, a value of 90 displays the
        x labels rotated 90 degrees clockwise.
    ylabelsize : int, default None
        If specified changes the y-axis label size.
    yrot : float, default None
        Rotation of y axis labels. For example, a value of 90 displays the
        y labels rotated 90 degrees clockwise.
    ax : Matplotlib axes object, default None
        The axes to plot the histogram on.
    sharex : bool, default True if ax is None else False
        In case subplots=True, share x axis and set some x axis labels to
        invisible; defaults to True if ax is None otherwise False if an ax
        is passed in.
        Note that passing in both an ax and sharex=True will alter all x axis
        labels for all subplots in a figure.
    sharey : bool, default False
        In case subplots=True, share y axis and set some y axis labels to
        invisible.
    figsize : tuple, optional
        The size in inches of the figure to create. Uses the value in
        `matplotlib.rcParams` by default.
    layout : tuple, optional
        Tuple of (rows, columns) for the layout of the histograms.
    bins : int or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.
    backend : str, optional
        The backend to use for rendering the plot. If None, defaults to
        matplotlib's default backend.
    legend : bool, default False
        Whether to display a legend on the plot.

    Returns
    -------
    None
    """
    # 指定要使用的绘图后端，默认为 None
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    
    # 是否显示图例，默认为 False
    legend : bool, default False
        Whether to show the legend.
    
    # **kwargs
    # 传递给 matplotlib.pyplot.hist 方法的所有其他绘图关键字参数
    
    Returns
    -------
    # 返回 matplotlib.Axes 或其 numpy.ndarray
    # 如果参数 column 或 by 是列表，则返回 AxesSubplot 对象的 numpy 数组
    matplotlib.Axes or numpy.ndarray of them
        Returns a AxesSubplot object a numpy array of AxesSubplot objects.
    
    See Also
    --------
    # 查看 matplotlib.pyplot.hist 文档以绘制直方图
    matplotlib.pyplot.hist : Plot a histogram using matplotlib.
    
    Examples
    --------
    # 此示例基于动物的长度和宽度绘制直方图，显示三个 bin
    
    .. plot::
        :context: close-figs
    
        >>> data = {"length": [1.5, 0.5, 1.2, 0.9, 3], "width": [0.7, 0.2, 0.15, 0.2, 1.1]}
        >>> index = ["pig", "rabbit", "duck", "chicken", "horse"]
        >>> df = pd.DataFrame(data, index=index)
        >>> hist = df.hist(bins=3)
    """  # noqa: E501
    
    # 根据指定的 backend 获取绘图后端
    plot_backend = _get_plot_backend(backend)
    # 调用绘图后端的 hist_frame 方法进行绘图
    return plot_backend.hist_frame(
        data,
        column=column,
        by=by,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        ax=ax,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
        legend=legend,
        bins=bins,
        **kwargs,
    )
# 定义一个多行字符串，描述了如何从 DataFrame 列生成箱线图的功能和参数说明
_boxplot_doc = """
Make a box plot from DataFrame columns.

Make a box-and-whisker plot from DataFrame columns, optionally grouped
by some other columns. A box plot is a method for graphically depicting
groups of numerical data through their quartiles.
The box extends from the Q1 to Q3 quartile values of the data,
with a line at the median (Q2). The whiskers extend from the edges
of box to show the range of the data. By default, they extend no more than
`1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest
data point within that interval. Outliers are plotted as separate dots.

For further details see
Wikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.

Parameters
----------
%(data)s\
column : str or list of str, optional
    Column name or list of names, or vector.
    Can be any valid input to :meth:`pandas.DataFrame.groupby`.
by : str or array-like, optional
    Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.
    One box-plot will be done per value of columns in `by`.
ax : object of class matplotlib.axes.Axes, optional
    The matplotlib axes to be used by boxplot.
fontsize : float or str
    Tick label font size in points or as a string (e.g., `large`).
rot : float, default 0
    The rotation angle of labels (in degrees)
    with respect to the screen coordinate system.
grid : bool, default True
    Setting this to True will show the grid.
figsize : A tuple (width, height) in inches
    The size of the figure to create in matplotlib.
layout : tuple (rows, columns), optional
    For example, (3, 5) will display the subplots
    using 3 rows and 5 columns, starting from the top-left.
return_type : {'axes', 'dict', 'both'} or None, default 'axes'
    The kind of object to return. The default is ``axes``.

    * 'axes' returns the matplotlib axes the boxplot is drawn on.
    * 'dict' returns a dictionary whose values are the matplotlib
      Lines of the boxplot.
    * 'both' returns a namedtuple with the axes and dict.
    * when grouping with ``by``, a Series mapping columns to
      ``return_type`` is returned.

      If ``return_type`` is `None`, a NumPy array
      of axes with the same shape as ``layout`` is returned.
%(backend)s\

**kwargs
    All other plotting keyword arguments to be passed to
    :func:`matplotlib.pyplot.boxplot`.

Returns
-------
result
    See Notes.

See Also
--------
Series.plot.hist: Make a histogram.
matplotlib.pyplot.boxplot : Matplotlib equivalent plot.

Notes
-----
The return type depends on the `return_type` parameter:

* 'axes' : object of class matplotlib.axes.Axes
* 'dict' : dict of matplotlib.lines.Line2D objects
* 'both' : a namedtuple with structure (ax, lines)

For data grouped with ``by``, return a Series of the above or a numpy
array:

* :class:`~pandas.Series`
* :class:`~numpy.array` (for ``return_type = None``)

Use ``return_type='dict'`` when you want to tweak the appearance
"""
# 设置图形绘制的后端，例如使用哪种绘图引擎（matplotlib的后端），默认为None

_backend_doc = """\
backend : str, default None
    # 替代 `plotting.backend` 选项中指定的后端使用的后端
    # 例如，'matplotlib'。或者，要为整个会话指定 `plotting.backend`，请设置 `pd.options.plotting.backend`。
    Backend to use instead of the backend specified in the option
    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
    specify the ``plotting.backend`` for the whole session, set
    ``pd.options.plotting.backend``.
"""
_box_or_line_doc = """
        Parameters
        ----------
        x : label or position, optional
            Allows plotting of one column versus another. If not specified,
            the index of the DataFrame is used.
        y : label or position, optional
            Allows plotting of one column versus another. If not specified,
            all numerical columns are used.
        color : str, array-like, or dict, optional
            The color for each of the DataFrame's columns. Possible values are:

            - A single color string referred to by name, RGB or RGBA code,
                for instance 'red' or '#a98d19'.

            - A sequence of color strings referred to by name, RGB or RGBA
                code, which will be used for each column recursively. For
                instance ['green','yellow'] each column's %(kind)s will be filled in
                green or yellow, alternatively. If there is only a single column to
                be plotted, then only the first color from the color list will be
                used.

            - A dict of the form {column name : color}, so that each column will be
                colored accordingly. For example, if your columns are called `a` and
                `b`, then passing {'a': 'green', 'b': 'red'} will color %(kind)ss for
                column `a` in green and %(kind)ss for column `b` in red.

        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            An ndarray is returned with one :class:`matplotlib.axes.Axes`
            per column when ``subplots=True``.
"""


@Substitution(data="data : DataFrame\n    The data to visualize.\n", backend="")
@Appender(_boxplot_doc)
def boxplot(
    data: DataFrame,
    column: str | list[str] | None = None,
    by: str | list[str] | None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: str | None = None,
    **kwargs,
):
    """
    Generate a box plot for the given DataFrame `data`.

    Parameters
    ----------
    data : DataFrame
        The data to visualize.

    column : str or list of str, optional
        Column name or list of column names to plot. If not specified, all
        numerical columns are plotted.

    by : str or list of str, optional
        Column name or list of column names to group by for generating
        separate plots. If not specified, no grouping is performed.

    ax : matplotlib.axes.Axes or None, optional
        The axis to plot on. If None, a new figure and axis is created.

    fontsize : float or str or None, optional
        Font size for plot labels and titles.

    rot : int, optional, default 0
        Rotation angle of labels on x-axis.

    grid : bool, optional, default True
        Whether to show grid lines on the plot.

    figsize : tuple of (float, float) or None, optional
        Size of the figure (width, height) in inches. If None, defaults to
        matplotlib's default figure size.

    layout : tuple of (int, int) or None, optional
        Number of rows and columns of subplots. If None, subplots are
        arranged automatically.

    return_type : str or None, optional
        The type of object to return. If 'axes', returns a single matplotlib
        axes object. If 'dict', returns a dictionary mapping column names to
        axes objects. If None, returns an ndarray of axes objects when
        `subplots=True`, otherwise a single axes object.

    **kwargs
        Additional keyword arguments passed to the plotting backend.

    Returns
    -------
    matplotlib.axes.Axes or np.ndarray of them
        An ndarray is returned with one :class:`matplotlib.axes.Axes`
        per column when ``subplots=True``.
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.boxplot(
        data,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


@Substitution(data="", backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self: DataFrame,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    backend=None,
    **kwargs,
):
    """
    Generate a box plot for the DataFrame object.

    Parameters
    ----------
    self : DataFrame
        The DataFrame object itself.

    column : str or list of str, optional
        Column name or list of column names to plot. If not specified, all
        numerical columns are plotted.

    by : str or list of str, optional
        Column name or list of column names to group by for generating
        separate plots. If not specified, no grouping is performed.

    ax : matplotlib.axes.Axes or None, optional
        The axis to plot on. If None, a new figure and axis is created.

    fontsize : int or None, optional
        Font size for plot labels and titles.

    rot : int, optional, default 0
        Rotation angle of labels on x-axis.

    grid : bool, optional, default True
        Whether to show grid lines on the plot.

    figsize : tuple of (float, float) or None, optional
        Size of the figure (width, height) in inches. If None, defaults to
        matplotlib's default figure size.

    layout : tuple of (int, int) or None, optional
        Number of rows and columns of subplots. If None, subplots are
        arranged automatically.

    return_type : str or None, optional
        The type of object to return. If 'axes', returns a single matplotlib
        axes object. If 'dict', returns a dictionary mapping column names to
        axes objects. If None, returns an ndarray of axes objects when
        `subplots=True`, otherwise a single axes object.

    backend : str or None, optional
        The plotting backend to use. If None, defaults to the current
        backend.

    **kwargs
        Additional keyword arguments passed to the plotting backend.

    Returns
    -------
    matplotlib.axes.Axes or np.ndarray of them
        An ndarray is returned with one :class:`matplotlib.axes.Axes`
        per column when ``subplots=True``.
    """
    plot_backend = _get_plot_backend(backend)
    # 调用绘图后端的箱线图绘制函数，并返回结果

    return plot_backend.boxplot_frame(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )
# 定义一个函数用于从 DataFrameGroupBy 数据创建箱线图

def boxplot_frame_groupby(
    grouped: DataFrameGroupBy,  # 接收一个分组后的 DataFrameGroupBy 对象
    subplots: bool = True,  # 是否创建子图，默认为 True
    column=None,  # 分组依据的列名或列名列表，可用于进一步分组
    fontsize: int | None = None,  # 字体大小设定，默认为 None
    rot: int = 0,  # 标签旋转角度，默认为 0 度
    grid: bool = True,  # 是否显示网格，默认为 True
    ax=None,  # Matplotlib 的坐标轴对象，默认为 None
    figsize: tuple[float, float] | None = None,  # 图的尺寸，单位为英寸，默认为 None
    layout=None,  # 子图的布局，以元组形式表示 (行数, 列数)，可选
    sharex: bool = False,  # 是否共享 x 轴，默认为 False
    sharey: bool = True,  # 是否共享 y 轴，默认为 True
    backend=None,  # 绘图的后端，默认为 None
    **kwargs,  # 其他传递给 Matplotlib 箱线图函数的绘图关键字参数
):
    """
    从 DataFrameGroupBy 数据创建箱线图。

    Parameters
    ----------
    grouped : Grouped DataFrame
        分组后的 DataFrameGroupBy 对象
    subplots : bool
        * ``False`` - 不使用子图
        * ``True`` - 为每个组创建一个子图
    column : 列名或列名列表，或向量
        可以是任何有效的 groupby 输入
    fontsize : 整数或字符串
        字体大小设定
    rot : 标签旋转角度
    grid : 设置为 True 将显示网格
    ax : Matplotlib 坐标轴对象，默认为 None
    figsize : 元组 (宽度, 高度)，单位为英寸
    layout : 元组 (可选)
        绘图的布局：(行数, 列数)
    sharex : bool，默认 False
        是否在子图之间共享 x 轴
    sharey : bool，默认 True
        是否在子图之间共享 y 轴
    backend : 字符串，默认 None
        要使用的后端，而不是选项中指定的后端
    **kwargs
        传递给 Matplotlib 箱线图函数的所有其他绘图关键字参数

    Returns
    -------
    dict 或 DataFrame.boxplot 返回值
        如果 subplots=False，则返回组键/DataFrame.boxplot 返回值

    Examples
    --------
    你可以为分组数据创建箱线图并将它们显示为单独的子图：

    .. plot::
        :context: close-figs

        >>> import itertools
        >>> tuples = [t for t in itertools.product(range(1000), range(4))]
        >>> index = pd.MultiIndex.from_tuples(tuples, names=["lvl0", "lvl1"])
        >>> data = np.random.randn(len(index), 4)
        >>> df = pd.DataFrame(data, columns=list("ABCD"), index=index)
        >>> grouped = df.groupby(level="lvl1")
        >>> grouped.boxplot(rot=45, fontsize=12, figsize=(8, 10))  # doctest: +SKIP

    使用 ``subplots=False`` 选项将箱线图显示在单个图中。

    .. plot::
        :context: close-figs

        >>> grouped.boxplot(subplots=False, rot=45, fontsize=12)  # doctest: +SKIP
    """
    # 获取绘图后端
    plot_backend = _get_plot_backend(backend)
    # 调用绘图后端的箱线图生成函数，并返回结果
    return plot_backend.boxplot_frame_groupby(
        grouped,
        subplots=subplots,
        column=column,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        ax=ax,
        figsize=figsize,
        layout=layout,
        sharex=sharex,
        sharey=sharey,
        **kwargs,
    )
    # 定义一个绘图函数，根据参数选择合适的方式绘制图形，默认使用 matplotlib 库
    
    Parameters
    ----------
    data : Series or DataFrame
        要绘制图形的数据对象。
    x : label or position, default None
        如果 data 是 DataFrame，则指定 x 轴的数据标签或位置。
    y : label, position or list of label, positions, default None
        允许绘制一列数据相对于另一列的图形。仅当 data 是 DataFrame 时使用。
    kind : str
        要生成的图形类型：
    
        - 'line' : 折线图（默认）
        - 'bar' : 垂直条形图
        - 'barh' : 水平条形图
        - 'hist' : 直方图
        - 'box' : 箱线图
        - 'kde' : 核密度估计图
        - 'density' : 与 'kde' 相同
        - 'area' : 面积图
        - 'pie' : 饼图
        - 'scatter' : 散点图（仅限 DataFrame）
        - 'hexbin' : 六边形区域图（仅限 DataFrame）
    ax : matplotlib axes 对象, default None
        当前图形的坐标轴对象。
    subplots : bool 或者可迭代序列, default False
        是否将列分组到子图中：
    
        - ``False`` : 不使用子图
        - ``True`` : 为每列创建单独的子图。
        - 列标签的可迭代序列：为每组列创建一个子图。例如 `[('a', 'c'), ('b', 'd')]` 将
          创建 2 个子图：一个包含列 'a' 和 'c'，另一个包含列 'b' 和 'd'。未指定的其余列
          将在额外的子图中绘制（每列一个子图）。
    
          .. versionadded:: 1.5.0
    
    sharex : bool, default True 如果 ax 为 None 则为 True，否则为 False
        如果 ``subplots=True``，共享 x 轴，并将某些 x 轴标签设为不可见；如果 ax 为 None，则默认为 True，否则如果传入了 ax，则为 False；
        请注意，同时传入 ax 和 ``sharex=True`` 将会更改图中所有轴的 x 轴标签。
    sharey : bool, default False
        如果 ``subplots=True``，共享 y 轴，并将某些 y 轴标签设为不可见。
    layout : tuple, optional
        子图的布局 (行数, 列数)。
    figsize : a tuple (width, height) in inches
        图形对象的大小，单位为英寸。
    use_index : bool, default True
        是否使用索引作为 x 轴的刻度。
    title : str 或者列表
        要用于图形的标题。如果传入字符串，则将字符串打印在图形顶部。如果传入列表且 `subplots` 为 True，则在每个子图上方打印列表中的每个项。
    grid : bool, default None (matlab style default)
        是否显示坐标轴网格线。
    legend : bool 或者 {'reverse'}
        是否在坐标轴子图上放置图例。
    style : 列表或者字典
        每列的 matplotlib 线条样式。
    logx : bool 或者 'sym', default False
        是否在 x 轴上使用对数缩放或对称对数缩放。
    logy : bool 或者 'sym' default False
        是否在 y 轴上使用对数缩放或对称对数缩放。
    # 是否使用对数或对称对数标度来缩放x和y轴
    loglog : bool or 'sym', default False

    # 设置x轴刻度的数值
    xticks : sequence

    # 设置y轴刻度的数值
    yticks : sequence

    # 设置当前轴的x轴限制范围
    xlim : 2-tuple/list

    # 设置当前轴的y轴限制范围
    ylim : 2-tuple/list

    # 可选参数，用于指定x轴标签的名称
    xlabel : label, optional

        # 默认使用索引名称作为x轴标签，或者在平面图中使用x列名称作为标签。

        .. versionchanged:: 2.0.0

            现在适用于直方图。

    # 可选参数，用于指定y轴标签的名称
    ylabel : label, optional

        # 默认不显示y轴标签，或者在平面图中使用y列名称作为标签。

        .. versionchanged:: 2.0.0

            现在适用于直方图。

    # 刻度旋转角度（垂直方向的xticks或水平方向的yticks）
    rot : float, default None

    # x轴和y轴刻度的字体大小
    fontsize : float, default None

    # 字符串或matplotlib颜色映射对象，用于选择颜色的颜色映射
    colormap : str or matplotlib colormap object, default None

    # 如果为True，则绘制颜色条（仅对'scatter'和'hexbin'图有效）
    colorbar : bool, optional

    # 柱状图布局的相对对齐位置
    position : float

        # 从0（左侧/底部端）到1（右侧/顶部端）的相对位置。默认为0.5（中心）。

    # 如果为True，则使用DataFrame中的数据绘制表格，并将数据转置以符合matplotlib的默认布局。
    table : bool, Series or DataFrame, default False

    # 查看详细信息，请参阅:ref:`Plotting with Error Bars <visualization.errorbars>`
    yerr : DataFrame, Series, array-like, dict and str

    # 等效于yerr。
    xerr : DataFrame, Series, array-like, dict and str

    # 在线图和柱状图中默认为False，在面积图中默认为True。如果为True，则创建堆叠图。
    stacked : bool, default False

    # 是否在辅助y轴上绘制（如果是列表/元组，则指定在辅助y轴上绘制哪些列）
    secondary_y : bool or sequence, default False

    # 使用辅助y轴时，是否自动在图例中标记列标签为“(right)”
    mark_right : bool, default True

    # 如果为True，则可以绘制布尔值。
    include_bool : bool, default is False

    # 替代默认的绘图后端，而不是使用选项中指定的后端``plotting.backend``。例如，'matplotlib'。
    # 或者，要为整个会话指定``plotting.backend``，请设置``pd.options.plotting.backend``。
    backend : str, default None

    # 传递给matplotlib绘图方法的其他选项。
    **kwargs
    """
    :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        If the backend is not the default matplotlib one, the return value
        will be the object returned by the backend.

    See Also
    --------
    matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.
    DataFrame.hist : Make a histogram.
    DataFrame.boxplot : Make a box plot.
    DataFrame.plot.scatter : Make a scatter plot with varying marker
        point size and color.
    DataFrame.plot.hexbin : Make a hexagonal binning plot of
        two variables.
    DataFrame.plot.kde : Make Kernel Density Estimate plot using
        Gaussian kernels.
    DataFrame.plot.area : Make a stacked area plot.
    DataFrame.plot.bar : Make a bar plot.
    DataFrame.plot.barh : Make a horizontal bar plot.

    Notes
    -----
    - See matplotlib documentation online for more on this subject
    - If `kind` = 'bar' or 'barh', you can specify relative alignments
      for bar plot layout by `position` keyword.
      From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5
      (center)

    Examples
    --------
    For Series:

    .. plot::
        :context: close-figs

        >>> ser = pd.Series([1, 2, 3, 3])
        >>> plot = ser.plot(kind="hist", title="My plot")

    For DataFrame:

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame(
        ...     {"length": [1.5, 0.5, 1.2, 0.9, 3], "width": [0.7, 0.2, 0.15, 0.2, 1.1]},
        ...     index=["pig", "rabbit", "duck", "chicken", "horse"],
        ... )
        >>> plot = df.plot(title="DataFrame Plot")

    For SeriesGroupBy:

    .. plot::
        :context: close-figs

        >>> lst = [-1, -2, -3, 1, 2, 3]
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> plot = ser.groupby(lambda x: x > 0).plot(title="SeriesGroupBy Plot")

    For DataFrameGroupBy:

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": ["A", "B", "A", "B"]})
        >>> plot = df.groupby("col2").plot(kind="bar", title="DataFrameGroupBy Plot")
    """  # noqa: E501

    _common_kinds = ("line", "bar", "barh", "kde", "density", "area", "hist", "box")
    _series_kinds = ("pie",)
    _dataframe_kinds = ("scatter", "hexbin")
    _kind_aliases = {"density": "kde"}
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data: Series | DataFrame) -> None:
        """
        Initialize the plot object for given Series or DataFrame.

        Parameters
        ----------
        data : Series or DataFrame
            The data to be plotted.

        Returns
        -------
        None
        """
        self._parent = data

    @staticmethod
    def __call__(self, kind: str = None, **kwargs) -> None:
        """
        Call method to delegate plotting based on kind.

        Parameters
        ----------
        kind : str, optional
            The kind of plot to produce (e.g., 'line', 'bar').
        **kwargs
            Other keyword arguments passed to the plotting function.

        Returns
        -------
        None
        """
        pass
    @Appender(
        """
        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.

        Examples
        --------

        .. plot::
            :context: close-figs

            >>> s = pd.Series([1, 3, 2])
            >>> s.plot.line()  # doctest: +SKIP

        .. plot::
            :context: close-figs

            The following example shows the populations for some animals
            over the years.

            >>> df = pd.DataFrame({
            ...     'pig': [20, 18, 489, 675, 1776],
            ...     'horse': [4, 25, 281, 600, 1900]
            ... }, index=[1990, 1997, 2003, 2009, 2014])
            >>> lines = df.plot.line()

        .. plot::
           :context: close-figs

           An example with subplots, so an array of axes is returned.

           >>> axes = df.plot.line(subplots=True)
           >>> type(axes)
           <class 'numpy.ndarray'>

        .. plot::
           :context: close-figs

           Let's repeat the same example, but specifying colors for
           each column (in this case, for each animal).

           >>> axes = df.plot.line(
           ...     subplots=True, color={"pig": "pink", "horse": "#742802"}
           ... )

        .. plot::
            :context: close-figs

            The following example shows the relationship between both
            populations.

            >>> lines = df.plot.line(x='pig', y='horse')
        """
    )
    # 应用注释到方法，添加相关文档内容，提供示例和用法说明
    @Substitution(kind="line")
    # 使用替换参数 "kind" 设置为 "line"
    @Appender(_bar_or_line_doc)
    # 在文档末尾附加 _bar_or_line_doc 的内容
    def line(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        color: str | Sequence[str] | dict | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        Plot Series or DataFrame as lines.

        This function is useful to plot lines using DataFrame's values
        as coordinates.
        """
        # 如果提供了颜色参数，则将其设置到 kwargs 中
        if color is not None:
            kwargs["color"] = color
        # 调用 self(kind="line", x=x, y=y, **kwargs) 并返回结果
        return self(kind="line", x=x, y=y, **kwargs)
    @Appender(
        """
        See Also
        --------
        DataFrame.plot.barh : Horizontal bar plot.
        DataFrame.plot : Make plots of a DataFrame.
        matplotlib.pyplot.bar : Make a bar plot with matplotlib.

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.bar(x='lab', y='val', rot=0)

        Plot a whole dataframe to a bar plot. Each column is assigned a
        distinct color, and each row is nested in a group along the
        horizontal axis.

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.bar(rot=0)

        Plot stacked bar charts for the DataFrame

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(stacked=True)

        Instead of nesting, the figure can be split by column with
        ``subplots=True``. In this case, a :class:`numpy.ndarray` of
        :class:`matplotlib.axes.Axes` are returned.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(rot=0, subplots=True)
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        If you don't like the default colours, you can specify how you'd
        like each column to be colored.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(
            ...     rot=0, subplots=True, color={"speed": "red", "lifespan": "green"}
            ... )
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        Plot a single column.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(y='speed', rot=0)

        Plot only selected categories for the DataFrame.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(x='lifespan', rot=0)
        """
    )
    @Substitution(kind="bar")
    @Appender(_bar_or_line_doc)
    # 定义 bar 方法，用于绘制条形图
    def bar(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        color: str | Sequence[str] | dict | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        返回一个PlotAccessor对象，用于绘制垂直条形图。

        条形图用于展示分类数据，使用长度与其代表的数值成比例的矩形条。条形图显示离散类别之间的比较。
        图的一条轴显示具体的比较类别，另一条轴表示测量数值。
        """
        if color is not None:
            kwargs["color"] = color
        return self(kind="bar", x=x, y=y, **kwargs)

    @Appender(
        """
        参见
        --------
        DataFrame.plot.bar : 垂直条形图。
        DataFrame.plot : 使用matplotlib绘制DataFrame的图表。
        matplotlib.axes.Axes.bar : 使用matplotlib绘制垂直条形图。

        示例
        --------
        基本示例

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.barh(x='lab', y='val')

        将整个DataFrame绘制为水平条形图

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh()

        绘制堆叠的水平条形图

        .. plot::
            :context: close-figs

            >>> ax = df.plot.barh(stacked=True)

        我们可以为每一列指定颜色

        .. plot::
            :context: close-figs

            >>> ax = df.plot.barh(color={"speed": "red", "lifespan": "green"})

        将DataFrame的一列绘制为水平条形图

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(y='speed')

        将DataFrame与所需列进行比较绘制

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(x='lifespan')
        """
    )
    @Substitution(kind="bar")
    @Appender(_bar_or_line_doc)
    def barh(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        color: str | Sequence[str] | dict | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        Make a horizontal bar plot.

        A horizontal bar plot is a plot that presents quantitative data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.
        """
        # 如果 color 参数不为 None，则将其加入 kwargs 字典中
        if color is not None:
            kwargs["color"] = color
        # 调用 PlotAccessor 对象的 barh 方法，制作一个水平条形图，传入 x, y 和所有的 kwargs 参数
        return self(kind="barh", x=x, y=y, **kwargs)
    # 定义一个方法用于绘制DataFrame列的箱线图
    def box(self, by: IndexLabel | None = None, **kwargs) -> PlotAccessor:
        """
        绘制DataFrame列的箱线图。

        箱线图是一种通过四分位数来图形化显示数值数据组的方法。
        箱子从数据的Q1到Q3四分位数值延伸，中间有一条中位数线（Q2）。
        指向数据范围的线条称为“须”，默认情况下须的位置设置为距离箱子边缘1.5倍的IQR（IQR = Q3 - Q1）。
        超出须末端的点被认为是异常值。

        进一步细节请参考维基百科关于 `箱线图 <https://en.wikipedia.org/wiki/Box_plot>`__ 的条目。

        当使用此图表时需要考虑箱子和须可能重叠，这在绘制小数据集时非常常见。

        Parameters
        ----------
        by : str or sequence
            DataFrame中用于分组的列名。

            .. versionchanged:: 1.4.0
               之前，`by` 参数被静默忽略且不进行分组。

        **kwargs
            其它可用参数详见 :meth:`DataFrame.plot`。

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them
            包含箱线图的 matplotlib axes 对象。

        See Also
        --------
        DataFrame.boxplot: 另一种绘制箱线图的方法。
        Series.plot.box: 从 Series 对象绘制箱线图。
        matplotlib.pyplot.boxplot: 在 matplotlib 中绘制箱线图。

        Examples
        --------
        从包含四列随机生成数据的DataFrame绘制箱线图。

        .. plot::
            :context: close-figs

            >>> data = np.random.randn(25, 4)
            >>> df = pd.DataFrame(data, columns=list("ABCD"))
            >>> ax = df.plot.box()

        如果指定 `by` 参数，可以生成分组：

        .. versionchanged:: 1.4.0

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.box(column="age", by="gender", figsize=(10, 8))
        """
        return self(kind="box", by=by, **kwargs)
    ) -> PlotAccessor:
        """
        Draw one histogram of the DataFrame's columns.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.
        This is useful when the DataFrame's Series are in a similar scale.

        Parameters
        ----------
        by : str or sequence, optional
            Column in the DataFrame to group by.

            .. versionchanged:: 1.4.0

               Previously, `by` is silently ignore and makes no groupings

        bins : int, default 10
            Number of histogram bins to be used.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Return a histogram plot.

        See Also
        --------
        DataFrame.hist : Draw histograms per DataFrame's Series.
        Series.hist : Draw a histogram with Series' data.

        Examples
        --------
        When we roll a die 6000 times, we expect to get each value around 1000
        times. But when we roll two dice and sum the result, the distribution
        is going to be quite different. A histogram illustrates those
        distributions.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(np.random.randint(1, 7, 6000), columns=["one"])
            >>> df["two"] = df["one"] + np.random.randint(1, 7, 6000)
            >>> ax = df.plot.hist(bins=12, alpha=0.5)

        A grouped histogram can be generated by providing the parameter `by` (which
        can be a column name, or a list of column names):

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.hist(column=["age"], by="gender", figsize=(10, 8))
        """
        return self(kind="hist", by=by, bins=bins, **kwargs)

    def kde(
        self,
        bw_method: Literal["scott", "silverman"] | float | Callable | None = None,
        ind: np.ndarray | int | None = None,
        **kwargs,
    ):
        """
        Kernel Density Estimate (KDE) plot estimation.

        This function estimates and plots the Kernel Density Estimate for the data
        in the DataFrame. KDE is a non-parametric way to estimate the probability
        density function of a random variable.

        Parameters
        ----------
        bw_method : {"scott", "silverman"} or float or callable, default None
            Method to calculate bandwidth. If None, "scott" is used.
            For more details, refer to :func:`scipy.stats.gaussian_kde`.
        ind : np.ndarray or int or None, optional
            Points to evaluate the KDE on. If None, it uses the data points.

        **kwargs
            Additional keyword arguments are passed to :meth:`DataFrame.plot`.

        Returns
        -------
        density : :class:`matplotlib.axes.Axes`
            Returns a KDE plot.

        See Also
        --------
        DataFrame.plot.kde : Draw KDE plots per DataFrame's Series.
        Series.plot.kde : Draw a KDE plot with Series' data.

        Examples
        --------
        A KDE plot can be drawn directly from DataFrame:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(np.random.randn(100, 2), columns=["A", "B"])
            >>> ax = df.plot.kde()

        A grouped KDE plot by a column (e.g., gender):

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.kde(by="gender", figsize=(10, 8))
        """
        return self(kind="kde", bw_method=bw_method, ind=ind, **kwargs)

    def area(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        stacked: bool = True,
        **kwargs,
    ):
        """
        Draw a stacked area plot.

        This function draws a stacked area plot using the data in the DataFrame.
        Stacked area plots are useful to show how different variables contribute
        to the whole over time or any other category.

        Parameters
        ----------
        x : Hashable or None, optional
            Label or position of the x-axis.
        y : Hashable or None, optional
            Label or position of the y-axis.
        stacked : bool, default True
            If True, the areas are stacked on top of each other.

        **kwargs
            Additional keyword arguments are documented in :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes`
            Returns an area plot.

        See Also
        --------
        DataFrame.plot.area : Draw area plots per DataFrame's Series.
        Series.plot.area : Draw an area plot with Series' data.

        Examples
        --------
        A stacked area plot can be drawn directly from DataFrame:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
            >>> ax = df.plot.area()

        A grouped stacked area plot by a column (e.g., gender):

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.area(x="gender", stacked=False, figsize=(10, 8))
        """
    ) -> PlotAccessor:
        """
        Draw a stacked area plot.

        An area plot displays quantitative data visually.
        This function wraps the matplotlib area function.

        Parameters
        ----------
        x : label or position, optional
            Coordinates for the X axis. By default uses the index.
        y : label or position, optional
            Column to plot. By default uses all columns.
        stacked : bool, default True
            Area plots are stacked by default. Set to False to create a
            unstacked plot.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Area plot, or array of area plots if subplots is True.

        See Also
        --------
        DataFrame.plot : Make plots of DataFrame using matplotlib.

        Examples
        --------
        Draw an area plot based on basic business metrics:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     {
            ...         "sales": [3, 2, 3, 9, 10, 6],
            ...         "signups": [5, 5, 6, 12, 14, 13],
            ...         "visits": [20, 42, 28, 62, 81, 50],
            ...     },
            ...     index=pd.date_range(start="2018/01/01", end="2018/07/01", freq="ME"),
            ... )
            >>> ax = df.plot.area()

        Area plots are stacked by default. To produce an unstacked plot,
        pass ``stacked=False``:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.area(stacked=False)

        Draw an area plot for a single column:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.area(y="sales")

        Draw with a different `x`:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     {
            ...         "sales": [3, 2, 3],
            ...         "visits": [20, 42, 28],
            ...         "day": [1, 2, 3],
            ...     }
            ... )
            >>> ax = df.plot.area(x="day")
        """  # noqa: E501
        # 调用 DataFrame.plot 的 area 方法，用于绘制堆叠区域图
        return self(kind="area", x=x, y=y, stacked=stacked, **kwargs)
    def pie(self, y: IndexLabel | None = None, **kwargs) -> PlotAccessor:
        """
        Generate a pie plot.

        A pie plot is a proportional representation of the numerical data in a
        column. This function wraps :meth:`matplotlib.pyplot.pie` for the
        specified column. If no column reference is passed and
        ``subplots=True`` a pie plot is drawn for each numerical column
        independently.

        Parameters
        ----------
        y : int or label, optional
            Label or position of the column to plot.
            If not provided, ``subplots=True`` argument must be passed.
        **kwargs
            Keyword arguments to pass on to :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            A NumPy array is returned when `subplots` is True.

        See Also
        --------
        Series.plot.pie : Generate a pie plot for a Series.
        DataFrame.plot : Make plots of a DataFrame.

        Examples
        --------
        In the example below we have a DataFrame with the information about
        planet's mass and radius. We pass the 'mass' column to the
        pie function to get a pie plot.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     {"mass": [0.330, 4.87, 5.97], "radius": [2439.7, 6051.8, 6378.1]},
            ...     index=["Mercury", "Venus", "Earth"],
            ... )
            >>> plot = df.plot.pie(y="mass", figsize=(5, 5))

        .. plot::
            :context: close-figs

            >>> plot = df.plot.pie(subplots=True, figsize=(11, 6))
        """
        # 如果提供了 y 参数，则将其加入 kwargs
        if y is not None:
            kwargs["y"] = y
        # 如果 self._parent 是 ABCDataFrame 的实例，并且未提供 y 参数，并且未设置 subplots=True，抛出异常
        if (
            isinstance(self._parent, ABCDataFrame)
            and kwargs.get("y", None) is None
            and not kwargs.get("subplots", False)
        ):
            raise ValueError("pie requires either y column or 'subplots=True'")
        # 调用 DataFrame.plot(kind="pie")，并传入 kwargs
        return self(kind="pie", **kwargs)

    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[Hashable] | None = None,
        c: Hashable | Sequence[Hashable] | None = None,
        **kwargs,
    ):
        """
        Generate a scatter plot.

        Parameters
        ----------
        x, y : label or position, optional
            Coordinates of the points.
        s : scalar or array_like, optional
            Size of the markers.
        c : color, sequence, or sequence of color, optional
            Color of the markers.
        **kwargs
            Additional keyword arguments passed to :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Returns the Axes object with the plot drawn onto it.

        See Also
        --------
        DataFrame.plot : Make plots of a DataFrame.

        Examples
        --------
        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     {"mass": [0.330, 4.87, 5.97], "radius": [2439.7, 6051.8, 6378.1]},
            ...     index=["Mercury", "Venus", "Earth"],
            ... )
            >>> plot = df.plot.scatter(x="mass", y="radius")

        """
        pass  # scatter 方法暂未实现，留待以后开发

    def hexbin(
        self,
        x: Hashable,
        y: Hashable,
        C: Hashable | None = None,
        reduce_C_function: Callable | None = None,
        gridsize: int | tuple[int, int] | None = None,
        **kwargs,
    ):
        """
        Generate a hexagonal binning plot.

        Parameters
        ----------
        x, y : label or position, optional
            Coordinates of the points.
        C : label or position, optional
            Numeric values to be aggregated.
        reduce_C_function : function, optional
            Function to reduce C values within each hexagon.
        gridsize : int or tuple, optional
            Number of hexagons in the x-direction. If gridsize is a tuple,
            it specifies the number of hexagons in the x- and y-directions.

        **kwargs
            Additional keyword arguments passed to :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Returns the Axes object with the plot drawn onto it.

        See Also
        --------
        DataFrame.plot : Make plots of a DataFrame.

        Examples
        --------
        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     {"mass": [0.330, 4.87, 5.97], "radius": [2439.7, 6051.8, 6378.1]},
            ...     index=["Mercury", "Venus", "Earth"],
            ... )
            >>> plot = df.plot.hexbin(x="mass", y="radius", gridsize=20)

        """
        pass  # hexbin 方法暂未实现，留待以后开发
# 存储已加载的不同绘图后端模块的字典
_backends: dict[str, types.ModuleType] = {}


def _load_backend(backend: str) -> types.ModuleType:
    """
    加载 pandas 绘图后端模块。

    Parameters
    ----------
    backend : str
        后端的标识符。可以是使用 importlib.metadata 注册的入口点项，"matplotlib" 或者是一个模块名。

    Returns
    -------
    types.ModuleType
        已导入的后端模块。
    """
    from importlib.metadata import entry_points

    if backend == "matplotlib":
        # 因为 matplotlib 是一个可选依赖和一级后端，这里需要尝试导入，以便在需要时引发 ImportError。
        try:
            module = importlib.import_module("pandas.plotting._matplotlib")
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting when the "
                'default backend "matplotlib" is selected.'
            ) from None
        return module

    found_backend = False

    eps = entry_points()
    key = "pandas_plotting_backends"
    # entry_points 在 PY 3.10 以下版本中丢失 dict API 的支持
    if hasattr(eps, "select"):
        entry = eps.select(group=key)
    else:
        # Argument 2 to "get" of "dict" has incompatible type "Tuple[]";
        # expected "EntryPoints"  [arg-type]
        entry = eps.get(key, ())  # type: ignore[arg-type]
    for entry_point in entry:
        found_backend = entry_point.name == backend
        if found_backend:
            module = entry_point.load()
            break

    if not found_backend:
        # 回退到未注册的模块名方法。
        try:
            module = importlib.import_module(backend)
            found_backend = True
        except ImportError:
            # 后面会重新引发错误。
            pass

    if found_backend:
        if hasattr(module, "plot"):
            # 在设置选项时验证接口是否实现，而不是在绘图时验证。
            return module

    raise ValueError(
        f"Could not find plotting backend '{backend}'. Ensure that you've "
        f"installed the package providing the '{backend}' entrypoint, or that "
        "the package has a top-level `.plot` method."
    )


def _get_plot_backend(backend: str | None = None):
    """
    返回要使用的绘图后端（例如 `pandas.plotting._matplotlib`）。

    pandas 的绘图系统默认使用 matplotlib，但这里的想法是它也可以与其他第三方后端一起工作。
    此函数返回一个模块，该模块提供一个顶级的 `.plot` 方法，用于实际绘图。
    后端由字符串指定，可以来自关键字参数 `backend`，或者如果未指定，则来自选项 `pandas.options.plotting.backend`。
    文件中的其余代码都使用此处指定的后端进行绘图。
    """
    The backend is imported lazily, as matplotlib is a soft dependency, and
    pandas can be used without it being installed.

    Notes
    -----
    Modifies `_backends` with imported backend as a side effect.
    """
    # 从全局变量或配置中获取绘图后端的字符串表示，如果未指定则使用默认配置
    backend_str: str = backend or get_option("plotting.backend")

    # 如果已经加载过指定的后端，则直接返回该后端模块
    if backend_str in _backends:
        return _backends[backend_str]

    # 否则，根据后端字符串动态加载对应的后端模块
    module = _load_backend(backend_str)
    # 将加载的后端模块存储在 _backends 字典中，以备下次直接使用
    _backends[backend_str] = module
    # 返回加载的后端模块
    return module
```