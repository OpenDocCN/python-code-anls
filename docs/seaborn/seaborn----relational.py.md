# `D:\src\scipysrc\seaborn\seaborn\relational.py`

```
    """
    导入必要的模块和函数
    """
from functools import partial
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs

"""
从内部模块中导入必要的函数和类
"""
from ._base import (
    VectorPlotter,
)
from .utils import (
    adjust_legend_subtitles,
    _default_color,
    _deprecate_ci,
    _get_transform_functions,
    _scatter_legend_artist,
)
from ._compat import groupby_apply_include_groups
from ._statistics import EstimateAggregator, WeightedAggregator

"""
从axisgrid模块导入类和文档片段
"""
from .axisgrid import FacetGrid, _facet_docs

"""
导入文档字符串和核心文档片段
"""
from ._docstrings import DocstringComponents, _core_docs

"""
定义导出的模块成员
"""
__all__ = ["relplot", "scatterplot", "lineplot"]

"""
定义关于关系绘图的叙述性文段
"""
_relational_narrative = DocstringComponents(dict(

    # ---  Introductory prose
    main_api="""
`x` 和 `y` 之间的关系可以使用 `hue`、`size` 和 `style` 参数显示不同的数据子集。这些参数控制用于识别不同子集的视觉语义。可以通过使用这三种语义类型独立地显示最多三个维度，但这种绘图风格可能难以解释且通常效果不佳。对于增加可访问性，使用冗余语义（例如为同一变量同时使用 `hue` 和 `style`）可能会有所帮助。

详细信息请参见:ref:`tutorial <relational_tutorial>`。
    """,

    relational_semantic="""
`hue`（和较小程度上的 `size`）语义的默认处理方式取决于该变量是否被推断为表示“数值型”或“分类型”数据。特别是，数值变量默认使用顺序色彩图表示，并且图例条目显示常规的“刻度”，其值可能存在也可能不存在于数据中。可以通过各种参数控制此行为，具体描述和示例见下文。
    """,
))

"""
定义关于关系绘图的文档字典
"""
_relational_docs = dict(

    # --- Shared function parameters
    data_vars="""
x, y : `data` 中的变量名或向量数据
    输入数据变量；必须为数值。可以直接传递数据或引用 `data` 中的列。
    """,
    data="""
data : DataFrame、数组或数组列表
    输入数据结构。如果将 `x` 和 `y` 指定为变量名，则应为包含这些列的“长格式”DataFrame。否则，它被视为“宽格式”数据，且组合变量将被忽略。
    请参见示例，了解可以指定此参数的不同方式及其各自的不同效果。
    """,
    palette="""
palette : 字符串、列表、字典或matplotlib色彩映射
    用于在使用 `hue` 时确定颜色选择方式的对象。可以是 seaborn 调色板或 matplotlib 色彩映射的名称、颜色列表（任何 matplotlib 理解的内容）、将 `hue` 变量水平映射到颜色的字典，或者是一个 matplotlib 色彩映射对象。
    """,
    hue_order="""
hue_order : 列表
    `hue` 变量水平显示顺序的指定顺序，
"""
    otherwise they are determined from the data. Not relevant when the
    `hue` variable is numeric.
    """
    # 如果未指定，色调（hue）的规范化方式会根据数据自动确定。当 `hue` 变量为数值型时，这个参数不相关。
    hue_norm="""
hue_norm : tuple or :class:`matplotlib.colors.Normalize` object
    # `hue_norm`参数，可以是元组或者`matplotlib.colors.Normalize`对象
    Normalization in data units for colormap applied to the `hue`
    # 当`hue`变量为数值型时，用于对颜色映射进行数据单位的标准化。如果`hue`是分类变量则不相关。
    variable when it is numeric. Not relevant if `hue` is categorical.
    # 如果`hue`是分类变量，则此参数无关紧要。

sizes : list, dict, or tuple
    # `sizes`参数，可以是列表、字典或元组
    An object that determines how sizes are chosen when `size` is used.
    # 决定在使用`size`时如何选择大小的对象。
    List or dict arguments should provide a size for each unique data value,
    # 列表或字典参数应为每个唯一数据值提供一个大小，
    which forces a categorical interpretation. The argument may also be a
    # 这将强制进行分类解释。参数也可以是一个
    min, max tuple.
    # 最小值和最大值的元组。

size_order : list
    # `size_order`参数，应为列表类型
    Specified order for appearance of the `size` variable levels,
    # 指定`size`变量级别的出现顺序，
    otherwise they are determined from the data. Not relevant when the
    # 否则它们将从数据中确定。当`size`变量是数值型时不相关。
    `size` variable is numeric.

size_norm : tuple or Normalize object
    # `size_norm`参数，可以是元组或`Normalize`对象
    Normalization in data units for scaling plot objects when the
    # 当`size`变量是数值型时，用于在绘图对象上进行数据单位的标准化。
    `size` variable is numeric.

dashes : boolean, list, or dictionary
    # `dashes`参数，可以是布尔值、列表或字典
    Object determining how to draw the lines for different levels of the
    # 决定如何绘制不同`style`变量级别的线条的对象。
    `style` variable. Setting to `True` will use default dash codes, or
    # 设置为`True`将使用默认的虚线代码，
    you can pass a list of dash codes or a dictionary mapping levels of the
    # 或者您可以传递一个虚线代码列表或将`style`变量级别映射到虚线代码的字典。
    `style` variable to dash codes. Setting to `False` will use solid
    # 设置为`False`将对所有子集使用实线。
    lines for all subsets. Dashes are specified as in matplotlib: a tuple
    # 虚线的指定方式与matplotlib相同：为`(段长, 间隔长)`的元组，或空字符串以绘制实线。

markers : boolean, list, or dictionary
    # `markers`参数，可以是布尔值、列表或字典
    Object determining how to draw the markers for different levels of the
    # 决定如何为不同`style`变量级别绘制标记的对象。
    `style` variable. Setting to `True` will use default markers, or
    # 设置为`True`将使用默认的标记，
    you can pass a list of markers or a dictionary mapping levels of the
    # 或者您可以传递一个标记列表或将`style`变量级别映射到标记的字典。
    `style` variable to markers. Setting to `False` will draw
    # 设置为`False`将绘制无标记的线条。
    marker-less lines. Markers are specified as in matplotlib.

style_order : list
    # `style_order`参数，应为列表类型
    Specified order for appearance of the `style` variable levels
    # 指定`style`变量级别的出现顺序，
    otherwise they are determined from the data. Not relevant when the
    # 否则它们将从数据中确定。当`style`变量是数值型时不相关。
    `style` variable is numeric.

units : vector or key in `data`
    # `units`参数，可以是数据中的向量或键
    Grouping variable identifying sampling units. When used, a separate
    # 识别采样单元的分组变量。当使用时，
    line will be drawn for each unit with appropriate semantics, but no
    # 将为每个单元绘制一条具有适当语义的线条，但不添加图例条目。
    legend entry will be added. Useful for showing distribution of
    # 用于显示实验复制的分布，当不需要确切的身份时。
    experimental replicates when exact identities are not needed.

estimator : name of pandas method or callable or None
    # `estimator`参数，可以是pandas方法的名称、可调用对象或`None`
    Method for aggregating across multiple observations of the `y`
    # 跨`y`变量的多个观察聚合的方法。
    variable at the same `x` level. If `None`, all observations will
    # 在相同的`x`级别上聚合`y`变量的所有观察。
    be drawn.

ci : int or "sd" or None
    # `ci`参数，可以是整数、"sd"或`None`
    Size of the confidence interval to draw when aggregating.
    # 聚合时绘制置信区间的大小。
    
    .. deprecated:: 0.12.0
        # 标记此功能为已弃用，从版本0.12.0开始
        Use the new `errorbar` parameter for more flexibility.
        # 使用新的`errorbar`参数获得更多灵活性。

n_boot : int
    # `n_boot`参数，应为整数类型
    Number of bootstraps to use for computing the confidence interval.
    # 用于计算置信区间的自助法重抽样次数。

seed : int, numpy.random.Generator, or numpy.random.RandomState
    # `seed`参数，可以是整数、numpy.random.Generator对象或numpy.random.RandomState对象
    # Seed or random number generator for reproducible bootstrapping.
    """
    这是用于可重复引导过程的种子或随机数生成器。
    """
    legend="""
    """
    """
    这是一个空字符串，可能是为了在图表或文档中显示注释或图例而预留的位置。
    """
legend : "auto", "brief", "full", or False
    # 图例的绘制方式。如果为"brief"，则数值型的`hue`和`size`变量将以均匀间隔的值示例表示。
    # 如果为"full"，每个分组将在图例中显示条目。如果为"auto"，根据级别数量选择brief或full表示。
    # 如果为`False`，则不添加图例数据，也不绘制图例。

ax_in="""
ax : matplotlib Axes
    # 要绘制图表的 matplotlib Axes 对象，否则使用当前的 Axes。
    """,

ax_out="""
ax : matplotlib Axes
    # 返回绘制了图表的 Axes 对象。
    """,

)


_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    facets=DocstringComponents(_facet_docs),
    rel=DocstringComponents(_relational_docs),
    stat=DocstringComponents.from_function_params(EstimateAggregator.__init__),
)


class _RelationalPlotter(VectorPlotter):

    wide_structure = {
        "x": "@index", "y": "@values", "hue": "@columns", "style": "@columns",
    }

    # TODO where best to define default parameters?
    # 默认情况下进行排序
    sort = True


class _LinePlotter(_RelationalPlotter):

    _legend_attributes = ["color", "linewidth", "marker", "dashes"]

    def __init__(
        self, *,
        data=None, variables={},
        estimator=None, n_boot=None, seed=None, errorbar=None,
        sort=True, orient="x", err_style=None, err_kws=None, legend=None
    ):

        # TODO this is messy, we want the mapping to be agnostic about
        # the kind of plot to draw, but for the time being we need to set
        # this information so the SizeMapping can use it
        # 设置默认的大小范围，以便 SizeMapping 使用
        self._default_size_range = (
            np.r_[.5, 2] * mpl.rcParams["lines.linewidth"]
        )

        super().__init__(data=data, variables=variables)

        self.estimator = estimator
        self.errorbar = errorbar
        self.n_boot = n_boot
        self.seed = seed
        self.sort = sort
        self.orient = orient
        self.err_style = err_style
        self.err_kws = {} if err_kws is None else err_kws

        self.legend = legend

class _ScatterPlotter(_RelationalPlotter):

    _legend_attributes = ["color", "s", "marker"]

    def __init__(self, *, data=None, variables={}, legend=None):

        # TODO this is messy, we want the mapping to be agnostic about
        # the kind of plot to draw, but for the time being we need to set
        # this information so the SizeMapping can use it
        # 设置默认的大小范围，以便 SizeMapping 使用
        self._default_size_range = (
            np.r_[.5, 2] * np.square(mpl.rcParams["lines.markersize"])
        )

        super().__init__(data=data, variables=variables)

        self.legend = legend
    # --- Determine the visual attributes of the plot

    # Drop rows with NaN values from comp_data and assign the resulting DataFrame to data
    data = self.comp_data.dropna()
    # If data is empty, return from the function
    if data.empty:
        return

    # Normalize keyword arguments using mpl.collections.PathCollection as the baseline
    kws = normalize_kwargs(kws, mpl.collections.PathCollection)

    # Define vectors x and y from data['x'] and data['y'], respectively, handling NaN values
    empty = np.full(len(data), np.nan)
    x = data.get("x", empty)
    y = data.get("y", empty)

    # Apply inverse scaling to x and y coordinates based on the plot's axes
    _, inv_x = _get_transform_functions(ax, "x")
    _, inv_y = _get_transform_functions(ax, "y")
    x, y = inv_x(x), inv_y(y)

    # Check if 'style' is a variable in self.variables
    if "style" in self.variables:
        # Use the marker associated with the first level of _style_map for scatter plots
        example_level = self._style_map.levels[0]
        example_marker = self._style_map(example_level, "marker")
        kws.setdefault("marker", example_marker)

    # Conditionally set edgecolor of markers based on whether they are "filled"
    m = kws.get("marker", mpl.rcParams.get("marker", "o"))
    if not isinstance(m, mpl.markers.MarkerStyle):
        m = mpl.markers.MarkerStyle(m)
    if m.is_filled():
        kws.setdefault("edgecolor", "w")

    # Draw a scatter plot using x and y coordinates and additional keyword arguments
    points = ax.scatter(x=x, y=y, **kws)

    # Apply color mapping to the facecolors of points based on the 'hue' variable
    if "hue" in self.variables:
        points.set_facecolors(self._hue_map(data["hue"]))

    # Apply size mapping to the sizes of points based on the 'size' variable
    if "size" in self.variables:
        points.set_sizes(self._size_map(data["size"]))

    # Apply path mapping to the paths of points based on the 'style' variable
    if "style" in self.variables:
        p = [self._style_map(val, "path") for val in data["style"]]
        points.set_paths(p)

    # Apply default attributes dependent on the plot
    if "linewidth" not in kws:
        sizes = points.get_sizes()
        # Calculate linewidth based on point sizes
        linewidth = .08 * np.sqrt(np.percentile(sizes, 10))
        points.set_linewidths(linewidth)
        kws["linewidth"] = linewidth

    # Finalize the axes details
    # Add axis labels to the plot
    self._add_axis_labels(ax)
    # If legend is enabled, add legend data and handle adjustments
    if self.legend:
        attrs = {"hue": "color", "size": "s", "style": None}
        self.add_legend_data(ax, _scatter_legend_artist, kws, attrs)
        handles, _ = ax.get_legend_handles_labels()
        # If there are handles, create a legend with a specified title
        if handles:
            legend = ax.legend(title=self.legend_title)
            # Adjust subtitles in the legend
            adjust_legend_subtitles(legend)
def lineplot(
    data=None, *,
    x=None, y=None, hue=None, size=None, style=None, units=None, weights=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    dashes=True, markers=None, style_order=None,
    estimator="mean", errorbar=("ci", 95), n_boot=1000, seed=None,
    orient="x", sort=True, err_style="band", err_kws=None,
    legend="auto", ci="deprecated", ax=None, **kwargs
):
    # 处理 ci 参数的弃用
    errorbar = _deprecate_ci(errorbar, ci)

    # 创建 _LinePlotter 对象
    p = _LinePlotter(
        data=data,
        variables=dict(
            x=x, y=y, hue=hue, size=size, style=style, units=units, weight=weights
        ),
        estimator=estimator, n_boot=n_boot, seed=seed, errorbar=errorbar,
        sort=sort, orient=orient, err_style=err_style, err_kws=err_kws,
        legend=legend,
    )

    # 根据 hue 参数映射调色板颜色
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    
    # 根据 size 参数映射线条宽度
    p.map_size(sizes=sizes, order=size_order, norm=size_norm)
    
    # 根据 style 参数映射线条样式（标记和虚线）
    p.map_style(markers=markers, dashes=dashes, order=style_order)

    # 如果没有指定 ax 参数，则使用当前的绘图区域
    if ax is None:
        ax = plt.gca()

    # 如果没有设置样式变量并且没有设置虚线样式（ls 或 linestyle），则根据 dashes 参数设置虚线样式
    if "style" not in p.variables and not {"ls", "linestyle"} & set(kwargs):  # XXX
        kwargs["dashes"] = "" if dashes is None or isinstance(dashes, bool) else dashes

    # 如果数据不包含 x 和 y 的信息，则直接返回当前绘图区域
    if not p.has_xy_data:
        return ax

    # 将图形对象 p 附加到 ax 上
    p._attach(ax)

    # 从 kwargs 中获取颜色参数，并使用 _default_color 函数设置默认颜色
    color = kwargs.pop("color", kwargs.pop("c", None))
    kwargs["color"] = _default_color(ax.plot, hue, color, kwargs)

    # 绘制图形
    p.plot(ax, kwargs)
    return ax


# lineplot 函数的文档字符串，说明了其功能和参数详细信息
lineplot.__doc__ = """\
Draw a line plot with possibility of several semantic groupings.

{narrative.main_api}

{narrative.relational_semantic}

By default, the plot aggregates over multiple `y` values at each value of
`x` and shows an estimate of the central tendency and a confidence
interval for that estimate.

Parameters
----------
{params.core.data}
{params.core.xy}
hue : vector or key in `data`
    Grouping variable that will produce lines with different colors.
    Can be either categorical or numeric, although color mapping will
    behave differently in latter case.
size : vector or key in `data`
    Grouping variable that will produce lines with different widths.
    Can be either categorical or numeric, although size mapping will
    behave differently in latter case.
style : vector or key in `data`
    Grouping variable that will produce lines with different dashes
    and/or markers. Can have a numeric dtype but will always be treated
    as categorical.
{params.rel.units}
weights : vector or key in `data`
    Data values or column used to compute weighted estimation.
    Note that use of weights currently limits the choice of statistics
    to a 'mean' estimator and 'ci' errorbar.
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}
{params.rel.sizes}
{params.rel.size_order}
{params.rel.size_norm}
{params.rel.dashes}
{params.rel.markers}
"""
{params.rel.style_order}
{params.rel.estimator}
{params.stat.errorbar}
{params.rel.n_boot}
{params.rel.seed}
orient : "x" or "y"
    数据按照其在 x 或 y 轴上的方向进行排序或聚合。也可以理解为结果函数的“自变量”。
sort : boolean
    如果为 True，则按照 x 和 y 变量对数据进行排序；否则，连接点的顺序将按照数据集中出现的顺序。
err_style : "band" or "bars"
    是否使用半透明的误差带或离散的误差条来绘制置信区间。
err_kws : dict of keyword arguments
    用于控制误差条美观性的额外参数字典。这些关键字参数会传递给 :meth:`matplotlib.axes.Axes.fill_between`
    或 :meth:`matplotlib.axes.Axes.errorbar`，取决于 `err_style` 的设置。
{params.rel.legend}
{params.rel.ci}
{params.core.ax}
kwargs : key, value mappings
    其他关键字参数会传递给 :meth:`matplotlib.axes.Axes.plot`。

Returns
-------
{returns.ax}

See Also
--------
{seealso.scatterplot}
{seealso.pointplot}

Examples
--------

.. include:: ../docstrings/lineplot.rst

""".format(
    narrative=_relational_narrative,
    params=_param_docs,
    returns=_core_docs["returns"],
    seealso=_core_docs["seealso"],
)
{params.rel.size_order}
{params.rel.size_norm}
{params.rel.markers}
{params.rel.style_order}
{params.rel.legend}
{params.core.ax}
kwargs : key, value mappings
    Other keyword arguments are passed down to
    :meth:`matplotlib.axes.Axes.scatter`.

Returns
-------
{returns.ax}

See Also
--------
{seealso.lineplot}
{seealso.stripplot}
{seealso.swarmplot}

Examples
--------

.. include:: ../docstrings/scatterplot.rst

""".format(
    narrative=_relational_narrative,
    params=_param_docs,
    returns=_core_docs["returns"],
    seealso=_core_docs["seealso"],
)


def relplot(
    data=None, *,
    x=None, y=None, hue=None, size=None, style=None, units=None, weights=None,
    row=None, col=None, col_wrap=None, row_order=None, col_order=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    markers=None, dashes=None, style_order=None,
    legend="auto", kind="scatter", height=5, aspect=1, facet_kws=None,
    **kwargs
):

    if kind == "scatter":

        # Determine the plotter class and plotting function for scatter plots
        Plotter = _ScatterPlotter
        func = scatterplot
        markers = True if markers is None else markers  # Use default markers if not provided

    elif kind == "line":

        # Determine the plotter class and plotting function for line plots
        Plotter = _LinePlotter
        func = lineplot
        dashes = True if dashes is None else dashes  # Use default dashes if not provided

    else:
        # Raise an error if the plot kind is not recognized
        err = f"Plot kind {kind} not recognized"
        raise ValueError(err)

    # Check for attempt to specify axes parameter, which is not supported
    if "ax" in kwargs:
        msg = (
            "relplot is a figure-level function and does not accept "
            "the `ax` parameter. You may wish to try {}".format(kind + "plot")
        )
        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    # Map semantic variables to the plotter object
    variables = dict(x=x, y=y, hue=hue, size=size, style=style)
    if kind == "line":
        variables["units"] = units
        variables["weight"] = weights
    else:
        # Warn if irrelevant parameters (units, weights) are provided for scatter plot
        if units is not None:
            msg = "The `units` parameter has no effect with kind='scatter'."
            warnings.warn(msg, stacklevel=2)
        if weights is not None:
            msg = "The `weights` parameter has no effect with kind='scatter'."
            warnings.warn(msg, stacklevel=2)

    # Create the plotter object
    p = Plotter(
        data=data,
        variables=variables,
        legend=legend,
    )

    # Map hue-related attributes using plotter's methods
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    # Map size-related attributes using plotter's methods
    p.map_size(sizes=sizes, order=size_order, norm=size_norm)
    # Map style-related attributes using plotter's methods
    p.map_style(markers=markers, dashes=dashes, order=style_order)

    # Extract the semantic mappings for hue and size
    if "hue" in p.variables:
        palette = p._hue_map.lookup_table
        hue_order = p._hue_map.levels
        hue_norm = p._hue_map.norm
    else:
        palette = hue_order = hue_norm = None

    if "size" in p.variables:
        sizes = p._size_map.lookup_table
        size_order = p._size_map.levels
        size_norm = p._size_map.norm
    # 检查是否在参数对象 p 的 variables 属性中存在 "style" 键
    if "style" in p.variables:
        # 如果存在，获取样式顺序列表
        style_order = p._style_map.levels
        # 如果 markers 参数不为空，创建标记符号字典，键为样式顺序中的元素，值为对应样式的标记符号
        if markers:
            markers = {k: p._style_map(k, "marker") for k in style_order}
        else:
            markers = None
        # 如果 dashes 参数不为空，创建虚线样式字典，键为样式顺序中的元素，值为对应样式的虚线样式
        if dashes:
            dashes = {k: p._style_map(k, "dashes") for k in style_order}
        else:
            dashes = None
    else:
        # 如果不存在 "style" 键，将 markers、dashes 和 style_order 都设为 None
        markers = dashes = style_order = None

    # 获取参数对象 p 的 variables 属性中的变量列表
    variables = p.variables
    # 获取参数对象 p 的 plot_data 属性，该属性包含用于绘制单个图的数据
    plot_data = p.plot_data

    # 定义通用绘图参数
    plot_kws = dict(
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes, size_order=size_order, size_norm=size_norm,
        markers=markers, dashes=dashes, style_order=style_order,
        legend=False,
    )
    # 更新绘图参数字典 plot_kws，添加额外的关键字参数 kwargs
    plot_kws.update(kwargs)
    # 如果绘图类型是 "scatter"，则从 plot_kws 中移除 "dashes" 键
    if kind == "scatter":
        plot_kws.pop("dashes")

    # 将网格语义添加到绘图器中
    grid_variables = dict(
        x=x, y=y, row=row, col=col, hue=hue, size=size, style=style,
    )
    # 如果绘图类型是 "line"，则更新 grid_variables，添加额外的 units 和 weights 变量
    if kind == "line":
        grid_variables.update(units=units, weights=weights)
    # 使用 p.assign_variables 方法，将数据和网格变量分配给绘图器
    p.assign_variables(data, grid_variables)

    # 定义用于绘制每个面的命名变量
    # 将变量名重命名为带有下划线前缀的形式，以避免与分面变量名冲突
    plot_variables = {v: f"_{v}" for v in variables}
    # 如果 plot_variables 中包含 "weight" 键，则将其改为 "weights"
    if "weight" in plot_variables:
        plot_variables["weights"] = plot_variables.pop("weight")
    # 更新 plot_kws，将 plot_variables 中的内容添加到 plot_kws 中
    plot_kws.update(plot_variables)

    # 将行和列变量传递给 FacetGrid，保持它们的原始名称以正确渲染轴标题
    for var in ["row", "col"]:
        # 处理缺少名称信息的分面变量
        if var in p.variables and p.variables[var] is None:
            p.variables[var] = f"_{var}_"
    # 创建 grid_kws 字典，包含 "row" 和 "col" 变量的值
    grid_kws = {v: p.variables.get(v) for v in ["row", "col"]}

    # 根据 plot_variables 和 grid_kws 重命名 plot_data 结构的列
    new_cols = plot_variables.copy()
    new_cols.update(grid_kws)
    # 使用 plot_data 的 rename 方法，将列名重命名为新的列名
    full_data = p.plot_data.rename(columns=new_cols)

    # 设置 FacetGrid 对象
    facet_kws = {} if facet_kws is None else facet_kws.copy()
    # 创建 FacetGrid 对象 g，使用 full_data 数据和其他关键字参数
    g = FacetGrid(
        data=full_data.dropna(axis=1, how="all"),
        **grid_kws,
        col_wrap=col_wrap, row_order=row_order, col_order=col_order,
        height=height, aspect=aspect, dropna=False,
        **facet_kws
    )

    # 使用 g.map_dataframe 方法，将 func 应用于数据框中的每一行，使用 plot_kws 作为参数
    g.map_dataframe(func, **plot_kws)

    # 标记轴标签，使用原始变量名
    # 当变量名为 None 时，传递 ""，以覆盖内部变量
    g.set_axis_labels(variables.get("x") or "", variables.get("y") or "")
    # 如果存在图例信息
    if legend:
        # 替换原始的绘图数据，以便图例使用正确类型的数值数据，
        # 因为我们在上面强制进行了分类映射。
        p.plot_data = plot_data

        # 处理这里的额外非语义关键字参数。
        # 我们是有选择性地添加它们，因为有些 kwargs 可能是 seaborn 特定的，
        # 而不适用于进入图例的 matplotlib 艺术家。
        # 理想情况下，我们将有一个更好的解决方案，不需要在这里重新创建图例，
        # 并且将与轴级函数具有一致性。
        keys = ["c", "color", "alpha", "m", "marker"]

        # 如果是散点图
        if kind == "scatter":
            legend_artist = _scatter_legend_artist
            keys += ["s", "facecolor", "fc", "edgecolor", "ec", "linewidth", "lw"]
        else:
            # 如果不是散点图，则使用 Line2D 创建图例艺术家的偏函数
            legend_artist = partial(mpl.lines.Line2D, xdata=[], ydata=[])
            keys += [
                "markersize", "ms",
                "markeredgewidth", "mew",
                "markeredgecolor", "mec",
                "linestyle", "ls",
                "linewidth", "lw",
            ]

        # 从 kwargs 中提取公共关键字参数
        common_kws = {k: v for k, v in kwargs.items() if k in keys}

        # 属性映射
        attrs = {"hue": "color", "style": None}
        if kind == "scatter":
            attrs["size"] = "s"
        elif kind == "line":
            attrs["size"] = "linewidth"

        # 将图例数据添加到指定的轴上
        p.add_legend_data(g.axes.flat[0], legend_artist, common_kws, attrs)

        # 如果存在图例数据，则将图例添加到图形对象 g 中
        if p.legend_data:
            g.add_legend(legend_data=p.legend_data,
                         label_order=p.legend_order,
                         title=p.legend_title,
                         adjust_subtitles=True)

    # 将 FacetGrid 的 `data` 属性的列重命名为原始的列名
    orig_cols = {
        f"_{k}": f"_{k}_" if v is None else v for k, v in variables.items()
    }

    # 将 grid_data 重命名为 g.data，并合并数据（如果提供了数据）
    grid_data = g.data.rename(columns=orig_cols)

    # 如果提供了数据，并且 x 或 y 不为 None
    if data is not None and (x is not None or y is not None):
        # 如果 data 不是 DataFrame，则转换为 DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # 将 data 与 grid_data 合并，基于索引
        g.data = pd.merge(
            data,
            grid_data[grid_data.columns.difference(data.columns)],
            left_index=True,
            right_index=True,
        )
    else:
        # 否则，直接将 grid_data 赋值给 g.data
        g.data = grid_data

    # 返回 FacetGrid 对象 g
    return g
# 将 relplot 的文档字符串定义为一个长字符串，用于描述关系绘图的高级界面
relplot.__doc__ = """\
Figure-level interface for drawing relational plots onto a FacetGrid.

This function provides access to several different axes-level functions
that show the relationship between two variables with semantic mappings
of subsets. The `kind` parameter selects the underlying axes-level
function to use:

- :func:`scatterplot` (with `kind="scatter"`; the default)
- :func:`lineplot` (with `kind="line"`)

Extra keyword arguments are passed to the underlying function, so you
should refer to the documentation for each to see kind-specific options.

{narrative.main_api}

{narrative.relational_semantic}

After plotting, the :class:`FacetGrid` with the plot is returned and can
be used directly to tweak supporting plot details or add other layers.

Parameters
----------
{params.core.data}
{params.core.xy}
hue : vector or key in `data`
    Grouping variable that will produce elements with different colors.
    Can be either categorical or numeric, although color mapping will
    behave differently in latter case.
size : vector or key in `data`
    Grouping variable that will produce elements with different sizes.
    Can be either categorical or numeric, although size mapping will
    behave differently in latter case.
style : vector or key in `data`
    Grouping variable that will produce elements with different styles.
    Can have a numeric dtype but will always be treated as categorical.
{params.rel.units}
weights : vector or key in `data`
    Data values or column used to compute weighted estimation.
    Note that use of weights currently limits the choice of statistics
    to a 'mean' estimator and 'ci' errorbar.
{params.facets.rowcol}
{params.facets.col_wrap}
row_order, col_order : lists of strings
    Order to organize the rows and/or columns of the grid in, otherwise the
    orders are inferred from the data objects.
{params.core.palette}
{params.core.hue_order}
{params.core.hue_norm}
{params.rel.sizes}
{params.rel.size_order}
{params.rel.size_norm}
{params.rel.style_order}
{params.rel.dashes}
{params.rel.markers}
{params.rel.legend}
kind : string
    Kind of plot to draw, corresponding to a seaborn relational plot.
    Options are `"scatter"` or `"line"`.
{params.facets.height}
{params.facets.aspect}
facet_kws : dict
    Dictionary of other keyword arguments to pass to :class:`FacetGrid`.
kwargs : key, value pairings
    Other keyword arguments are passed through to the underlying plotting
    function.

Returns
-------
{returns.facetgrid}

Examples
--------

.. include:: ../docstrings/relplot.rst

""".format(
    # 插入关系绘图的叙述和语义
    narrative=_relational_narrative,
    # 插入参数文档
    params=_param_docs,
    # 插入返回值文档
    returns=_core_docs["returns"],
)
```