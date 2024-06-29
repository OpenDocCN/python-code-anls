# `D:\src\scipysrc\pandas\pandas\plotting\_misc.py`

```
# 导入未来版本的注解支持
from __future__ import annotations

# 导入上下文管理器和类型提示
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
)

# 从 pandas.plotting._core 模块中导入 _get_plot_backend 函数
from pandas.plotting._core import _get_plot_backend

# 如果是类型检查阶段，则导入必要的类型
if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Mapping,
    )

    from matplotlib.axes import Axes  # 导入 Axes 类型
    from matplotlib.colors import Colormap  # 导入 Colormap 类型
    from matplotlib.figure import Figure  # 导入 Figure 类型
    from matplotlib.table import Table  # 导入 Table 类型
    import numpy as np  # 导入 numpy

    from pandas import (  # 导入 DataFrame 和 Series 类型
        DataFrame,
        Series,
    )

# 定义函数 table，接受 Matplotlib 的 Axes 对象和 DataFrame 或 Series 类型的 data 数据
def table(ax: Axes, data: DataFrame | Series, **kwargs) -> Table:
    """
    辅助函数，将 DataFrame 和 Series 转换为 matplotlib.table 对象。

    Parameters
    ----------
    ax : Matplotlib axes 对象
        绘制表格的坐标轴。
    data : DataFrame 或 Series
        表格内容的数据。
    **kwargs
        传递给 matplotlib.table.table 的关键字参数。
        如果未指定 `rowLabels` 或 `colLabels`，则将使用数据的索引或列名。

    Returns
    -------
    matplotlib.table 对象
        创建的 matplotlib Table 对象。

    See Also
    --------
    DataFrame.plot : 使用 matplotlib 绘制 DataFrame 图表。
    matplotlib.pyplot.table : 在 Matplotlib 图中创建表格。

    Examples
    --------

    .. plot::
            :context: close-figs

            >>> import matplotlib.pyplot as plt
            >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
            >>> fix, ax = plt.subplots()
            >>> ax.axis("off")
            (0.0, 1.0, 0.0, 1.0)
            >>> table = pd.plotting.table(
            ...     ax, df, loc="center", cellLoc="center", colWidths=list([0.2, 0.2])
            ... )
    """
    # 获取绘图后端
    plot_backend = _get_plot_backend("matplotlib")
    # 调用绘图后端的 table 函数来创建表格
    return plot_backend.table(
        ax=ax, data=data, rowLabels=None, colLabels=None, **kwargs
    )


# 定义函数 register，没有参数和返回值
def register() -> None:
    """
    将 pandas 的格式化器和转换器注册到 matplotlib。

    此函数修改全局的 ``matplotlib.units.registry`` 字典。pandas 添加了以下自定义转换器：

    * pd.Timestamp
    * pd.Period
    * np.datetime64
    * datetime.datetime
    * datetime.date
    * datetime.time

    See Also
    --------
    deregister_matplotlib_converters : 移除 pandas 的格式化器和转换器。

    Examples
    --------
    .. plot::
       :context: close-figs

        下面的行是由 pandas 自动完成的，以便能够渲染图表：

        >>> pd.plotting.register_matplotlib_converters()

        >>> df = pd.DataFrame(
        ...     {"ts": pd.period_range("2020", periods=2, freq="M"), "y": [1, 2]}
        ... )
        >>> plot = df.plot.line(x="ts", y="y")

    如果手动取消注册，将会引发错误：

    >>> pd.set_option(
    ...     "plotting.matplotlib.register_converters", False
    ... )  # doctest: +SKIP
    >>> df.plot.line(x="ts", y="y")  # doctest: +SKIP

    """
    pass  # 函数体为空，什么也不做
    Traceback (most recent call last):
    TypeError: float() argument must be a string or a real number, not 'Period'
    
    
    
    # 导入的模块中可能会引发异常，TypeError 指出了具体的错误类型和错误信息
    plot_backend = _get_plot_backend("matplotlib")
    plot_backend.register()
    
    
    这段代码的注释解释了在调用 `_get_plot_backend()` 函数和 `register()` 方法时可能会发生的异常，提示了可能的错误类型和错误信息。
# 定义函数，用于取消注册 pandas 的格式化程序和转换器
def deregister() -> None:
    """
    Remove pandas formatters and converters.

    Removes the custom converters added by :func:`register`. This
    attempts to set the state of the registry back to the state before
    pandas registered its own units. Converters for pandas' own types like
    Timestamp and Period are removed completely. Converters for types
    pandas overwrites, like ``datetime.datetime``, are restored to their
    original value.

    See Also
    --------
    register_matplotlib_converters : Register pandas formatters and converters
        with matplotlib.

    Examples
    --------
    .. plot::
       :context: close-figs

        The following line is done automatically by pandas so
        the plot can be rendered:

        >>> pd.plotting.register_matplotlib_converters()

        >>> df = pd.DataFrame(
        ...     {"ts": pd.period_range("2020", periods=2, freq="M"), "y": [1, 2]}
        ... )
        >>> plot = df.plot.line(x="ts", y="y")

    Unsetting the register manually an error will be raised:

    >>> pd.set_option(
    ...     "plotting.matplotlib.register_converters", False
    ... )  # doctest: +SKIP
    >>> df.plot.line(x="ts", y="y")  # doctest: +SKIP
    Traceback (most recent call last):
    TypeError: float() argument must be a string or a real number, not 'Period'
    """
    # 获取绘图后端为 matplotlib 的对象
    plot_backend = _get_plot_backend("matplotlib")
    # 调用绘图后端对象的取消注册方法
    plot_backend.deregister()


# 函数用于绘制散点矩阵图
def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds: Mapping[str, Any] | None = None,
    hist_kwds: Mapping[str, Any] | None = None,
    range_padding: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """
    Draw a matrix of scatter plots.

    Parameters
    ----------
    frame : DataFrame
        输入的数据帧
    alpha : float, optional
        应用的透明度。
    figsize : (float,float), optional
        以英寸为单位的元组（宽度，高度）。
    ax : Matplotlib axis object, optional
        Matplotlib 的轴对象，可选。
    grid : bool, optional
        设置为 True 将显示网格。
    diagonal : {'hist', 'kde'}
        在对角线上选择 'kde' 或 'hist'，表示核密度估计或直方图。
    marker : str, optional
        Matplotlib 的标记类型，默认为 '.'。
    density_kwds : keywords
        传递给核密度估计绘图的关键字参数。
    hist_kwds : keywords
        传递给直方图函数的关键字参数。
    range_padding : float, default 0.05
        相对于轴范围的扩展，在 x 和 y 上分别为 (x_max - x_min) 或 (y_max - y_min)。
    **kwargs
        传递给散点图函数的关键字参数。

    Returns
    -------
    numpy.ndarray
        一个散点图矩阵。

    Examples
    --------

    """
    # 函数实现主体略
    """
    # 根据指定的 DataFrame 创建散点矩阵图
    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame(np.random.randn(1000, 4), columns=["A", "B", "C", "D"])
        >>> pd.plotting.scatter_matrix(df, alpha=0.2)
        创建一个散点矩阵，显示了 DataFrame 中每对列之间的关系
        array([[<Axes: xlabel='A', ylabel='A'>, <Axes: xlabel='B', ylabel='A'>,
                <Axes: xlabel='C', ylabel='A'>, <Axes: xlabel='D', ylabel='A'>],
               [<Axes: xlabel='A', ylabel='B'>, <Axes: xlabel='B', ylabel='B'>,
                <Axes: xlabel='C', ylabel='B'>, <Axes: xlabel='D', ylabel='B'>],
               [<Axes: xlabel='A', ylabel='C'>, <Axes: xlabel='B', ylabel='C'>,
                <Axes: xlabel='C', ylabel='C'>, <Axes: xlabel='D', ylabel='C'>],
               [<Axes: xlabel='A', ylabel='D'>, <Axes: xlabel='B', ylabel='D'>,
                <Axes: xlabel='C', ylabel='D'>, <Axes: xlabel='D', ylabel='D'>]],
              dtype=object)
    """
    # 获取 matplotlib 作为绘图后端
    plot_backend = _get_plot_backend("matplotlib")
    # 调用后端的散点矩阵绘制函数，传入参数以及其他关键字参数
    return plot_backend.scatter_matrix(
        frame=frame,
        alpha=alpha,
        figsize=figsize,
        ax=ax,
        grid=grid,
        diagonal=diagonal,
        marker=marker,
        density_kwds=density_kwds,
        hist_kwds=hist_kwds,
        range_padding=range_padding,
        **kwargs,
    )
# 定义函数 radviz，用于绘制多维数据集在二维空间上的 RadViz 图形
def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = None,
    color: list[str] | tuple[str, ...] | None = None,
    colormap: Colormap | str | None = None,
    **kwds,
) -> Axes:
    """
    Plot a multidimensional dataset in 2D.

    Each Series in the DataFrame is represented as a evenly distributed
    slice on a circle. Each data point is rendered in the circle according to
    the value on each Series. Highly correlated `Series` in the `DataFrame`
    are placed closer on the unit circle.

    RadViz allow to project a N-dimensional data set into a 2D space where the
    influence of each dimension can be interpreted as a balance between the
    influence of all dimensions.

    More info available at the `original article
    <https://doi.org/10.1145/331770.331775>`_
    describing RadViz.

    Parameters
    ----------
    frame : `DataFrame`
        Object holding the data.
    class_column : str
        Column name containing the name of the data point category.
    ax : :class:`matplotlib.axes.Axes`, optional
        A plot instance to which to add the information.
    color : list[str] or tuple[str], optional
        Assign a color to each category. Example: ['blue', 'green'].
    colormap : str or :class:`matplotlib.colors.Colormap`, default None
        Colormap to select colors from. If string, load colormap with that
        name from matplotlib.
    **kwds
        Options to pass to matplotlib scatter plotting method.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        The Axes object from Matplotlib.

    See Also
    --------
    plotting.andrews_curves : Plot clustering visualization.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame(
        ...     {
        ...         "SepalLength": [6.5, 7.7, 5.1, 5.8, 7.6, 5.0, 5.4, 4.6, 6.7, 4.6],
        ...         "SepalWidth": [3.0, 3.8, 3.8, 2.7, 3.0, 2.3, 3.0, 3.2, 3.3, 3.6],
        ...         "PetalLength": [5.5, 6.7, 1.9, 5.1, 6.6, 3.3, 4.5, 1.4, 5.7, 1.0],
        ...         "PetalWidth": [1.8, 2.2, 0.4, 1.9, 2.1, 1.0, 1.5, 0.2, 2.1, 0.2],
        ...         "Category": [
        ...             "virginica",
        ...             "virginica",
        ...             "setosa",
        ...             "virginica",
        ...             "virginica",
        ...             "versicolor",
        ...             "versicolor",
        ...             "setosa",
        ...             "virginica",
        ...             "setosa",
        ...         ],
        ...     }
        ... )
        >>> pd.plotting.radviz(df, "Category")  # doctest: +SKIP
    """
    # 获取绘图后端（这里是 matplotlib）的实例
    plot_backend = _get_plot_backend("matplotlib")
    # 调用绘制 RadViz 图形的函数，传入相应参数
    return plot_backend.radviz(
        frame=frame,
        class_column=class_column,
        ax=ax,
        color=color,
        colormap=colormap,
        **kwds,
    )
    samples: int = 200,
    color: list[str] | tuple[str, ...] | None = None,
    colormap: Colormap | str | None = None,
    **kwargs,



# 定义一个函数参数列表，包含以下参数：
# - samples: 表示采样数，默认为 200，类型为整数
# - color: 表示颜色信息，可以是字符串列表、字符串元组或者 None，默认为 None
# - colormap: 表示颜色映射，可以是 Colormap 对象、字符串或者 None，默认为 None
# - **kwargs: 表示其它可能的关键字参数，未指定类型限制
def bootstrap_plot(
    series: Series,
    fig: Figure | None = None,
    size: int = 50,
    samples: int = 500,
    **kwds,
) -> Figure:
    """
    Bootstrap plot on mean, median and mid-range statistics.

    The bootstrap plot is used to estimate the uncertainty of a statistic
    by relying on random sampling with replacement [1]_. This function will
    generate bootstrapping plots for mean, median and mid-range statistics
    for the given number of samples of the given size.

    .. [1] "Bootstrapping (statistics)" in \
    https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    Parameters
    ----------
    series : pandas.Series
        Series from where to get the samplings for the bootstrapping.
    fig : matplotlib.figure.Figure, default None
        If given, it will use the `fig` reference for plotting instead of
        creating a new one with default parameters.
    size : int, default 50
        Number of data points to consider during each sampling. It must be
        less than or equal to the length of the `series`.
    samples : int, default 500
        Number of bootstrap samples to generate.
    **kwds
        Additional keyword arguments passed to the plotting function.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the bootstrap plot.

    Examples
    --------
    Example of usage:

    >>> series = pd.Series([1, 2, 3, 4, 5])
    >>> pd.plotting.bootstrap_plot(series)  # Visualize bootstrap plot

    Notes
    -----
    The bootstrap plot helps in visualizing the spread of statistical
    estimators like mean, median, and mid-range across multiple samples,
    providing insights into their uncertainties.
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.bootstrap_plot(
        series=series,
        fig=fig,
        size=size,
        samples=samples,
        **kwds,
    )
    samples : int, default 500
        Number of times the bootstrap procedure is performed.
    **kwds
        Options to pass to matplotlib plotting method.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure.

    See Also
    --------
    DataFrame.plot : Basic plotting for DataFrame objects.
    Series.plot : Basic plotting for Series objects.

    Examples
    --------
    This example draws a basic bootstrap plot for a Series.

    .. plot::
        :context: close-figs

        >>> s = pd.Series(np.random.uniform(size=100))
        >>> pd.plotting.bootstrap_plot(s)  # doctest: +SKIP
        <Figure size 640x480 with 6 Axes>
    """
    # 获取适当的绘图后端（默认为 matplotlib）
    plot_backend = _get_plot_backend("matplotlib")
    # 调用绘制 bootstrap plot 的方法，并传入参数
    return plot_backend.bootstrap_plot(
        series=series, fig=fig, size=size, samples=samples, **kwds
    )
# 定义函数 parallel_coordinates，用于绘制平行坐标图
def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: list[str] | None = None,
    ax: Axes | None = None,
    color: list[str] | tuple[str, ...] | None = None,
    use_columns: bool = False,
    xticks: list | tuple | None = None,
    colormap: Colormap | str | None = None,
    axvlines: bool = True,
    axvlines_kwds: Mapping[str, Any] | None = None,
    sort_labels: bool = False,
    **kwargs,
) -> Axes:
    """
    Parallel coordinates plotting.

    Parameters
    ----------
    frame : DataFrame
        要绘制的 DataFrame 数据。
    class_column : str
        包含类名的列名。
    cols : list, optional
        要使用的列名列表。
    ax : matplotlib.axis, optional
        Matplotlib 的 axis 对象。
    color : list or tuple, optional
        不同类别使用的颜色。
    use_columns : bool, optional
        如果为 True，则使用列作为 x 轴刻度。
    xticks : list or tuple, optional
        用于 x 轴刻度的值列表。
    colormap : str or matplotlib colormap, default None
        用于线条颜色的 colormap。
    axvlines : bool, optional
        如果为 True，则在每个 x 轴刻度处添加垂直线。
    axvlines_kwds : keywords, optional
        传递给 axvline 方法的选项，用于垂直线。
    sort_labels : bool, default False
        是否对 class_column 标签进行排序，对颜色分配有用。
    **kwargs
        传递给 matplotlib 绘图方法的选项。

    Returns
    -------
    matplotlib.axes.Axes
        包含平行坐标图的 matplotlib axes 对象。

    See Also
    --------
    plotting.andrews_curves : 生成用于可视化多变量数据聚类的 matplotlib 图。
    plotting.radviz : 在2D中绘制多维数据集。

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> df = pd.read_csv(
        ...     "https://raw.githubusercontent.com/pandas-dev/"
        ...     "pandas/main/pandas/tests/io/data/csv/iris.csv"
        ... )
        >>> pd.plotting.parallel_coordinates(
        ...     df, "Name", color=("#556270", "#4ECDC4", "#C7F464")
        ... )  # doctest: +SKIP
    """
    # 获取绘图后端，这里使用 matplotlib
    plot_backend = _get_plot_backend("matplotlib")
    # 调用绘制平行坐标图的函数，传递参数并返回结果
    return plot_backend.parallel_coordinates(
        frame=frame,
        class_column=class_column,
        cols=cols,
        ax=ax,
        color=color,
        use_columns=use_columns,
        xticks=xticks,
        colormap=colormap,
        axvlines=axvlines,
        axvlines_kwds=axvlines_kwds,
        sort_labels=sort_labels,
        **kwargs,
    )


# 定义函数 lag_plot，用于绘制时间序列的滞后散点图
def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
    """
    Lag plot for time series.

    Parameters
    ----------
    series : Series
        要可视化的时间序列数据。
    lag : int, default 1
        散点图的滞后长度。
    ax : Matplotlib axis object, optional
        要使用的 matplotlib axis 对象。
    **kwds
        其他传递给函数的选项。
    """
    # 获取用于绘制 lag plot 的后端引擎，默认为 Matplotlib
    plot_backend = _get_plot_backend("matplotlib")
    # 调用后端引擎的 lag_plot 方法，传入时间序列、滞后值、轴对象以及其他关键字参数
    return plot_backend.lag_plot(series=series, lag=lag, ax=ax, **kwds)
def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwargs) -> Axes:
    """
    Autocorrelation plot for time series.

    Parameters
    ----------
    series : Series
        The time series to visualize.
    ax : Matplotlib axis object, optional
        The matplotlib axis object to use.
    **kwargs
        Options to pass to matplotlib plotting method.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes containing the autocorrelation plot.

    See Also
    --------
    Series.autocorr : Compute the lag-N autocorrelation for a Series.
    plotting.lag_plot : Lag plot for time series.

    Examples
    --------
    The horizontal lines in the plot correspond to 95% and 99% confidence bands.

    The dashed line is 99% confidence band.

    .. plot::
        :context: close-figs

        >>> spacing = np.linspace(-9 * np.pi, 9 * np.pi, num=1000)
        >>> s = pd.Series(0.7 * np.random.rand(1000) + 0.3 * np.sin(spacing))
        >>> pd.plotting.autocorrelation_plot(s)  # doctest: +SKIP
    """
    # 获取绘图后端，这里使用matplotlib
    plot_backend = _get_plot_backend("matplotlib")
    # 调用matplotlib后端的autocorrelation_plot方法进行自相关图绘制
    return plot_backend.autocorrelation_plot(series=series, ax=ax, **kwargs)


class _Options(dict):
    """
    Stores pandas plotting options.

    Allows for parameter aliasing so you can just use parameter names that are
    the same as the plot function parameters, but is stored in a canonical
    format that makes it easy to breakdown into groups later.

    See Also
    --------
    plotting.register_matplotlib_converters : Register pandas formatters and
        converters with matplotlib.
    plotting.bootstrap_plot : Bootstrap plot on mean, median and mid-range statistics.
    plotting.autocorrelation_plot : Autocorrelation plot for time series.
    plotting.lag_plot : Lag plot for time series.

    Examples
    --------

    .. plot::
            :context: close-figs

             >>> np.random.seed(42)
             >>> df = pd.DataFrame(
             ...     {"A": np.random.randn(10), "B": np.random.randn(10)},
             ...     index=pd.date_range("1/1/2000", freq="4MS", periods=10),
             ... )
             >>> with pd.plotting.plot_params.use("x_compat", True):
             ...     _ = df["A"].plot(color="r")
             ...     _ = df["B"].plot(color="g")
    """

    # 别名字典，使参数名与绘图方法参数名相同
    _ALIASES = {"x_compat": "xaxis.compat"}
    # 默认键列表
    _DEFAULT_KEYS = ["xaxis.compat"]

    def __init__(self) -> None:
        # 设置默认参数
        super().__setitem__("xaxis.compat", False)

    def __getitem__(self, key):
        # 获取参数值，通过规范的键名
        key = self._get_canonical_key(key)
        if key not in self:
            raise ValueError(f"{key} is not a valid pandas plotting option")
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        # 设置参数值，通过规范的键名
        key = self._get_canonical_key(key)
        super().__setitem__(key, value)

    def _get_canonical_key(self, key):
        # 返回规范的键名
        return self._ALIASES.get(key, key)
    # 删除指定键对应的项，确保键是规范化后的形式
    def __delitem__(self, key) -> None:
        key = self._get_canonical_key(key)
        # 如果键是默认键之一，抛出值错误异常
        if key in self._DEFAULT_KEYS:
            raise ValueError(f"Cannot remove default parameter {key}")
        # 调用父类的 __delitem__ 方法删除键对应的项
        super().__delitem__(key)

    # 检查指定键是否存在于选项存储中，确保键是规范化后的形式
    def __contains__(self, key) -> bool:
        key = self._get_canonical_key(key)
        # 调用父类的 __contains__ 方法检查键是否存在
        return super().__contains__(key)

    # 将选项存储重置为初始状态
    def reset(self) -> None:
        """
        Reset the option store to its initial state

        Returns
        -------
        None
        """
        # 错误：无法直接访问 "__init__"
        # 重新调用对象的 __init__ 方法，类型忽略杂项错误
        self.__init__()  # type: ignore[misc]

    # 根据别名获取键的规范化形式
    def _get_canonical_key(self, key: str) -> str:
        return self._ALIASES.get(key, key)

    # 使用上下文管理器临时设置参数值
    @contextmanager
    def use(self, key, value) -> Generator[_Options, None, None]:
        """
        Temporarily set a parameter value using the with statement.
        Aliasing allowed.
        """
        # 获取当前键对应的旧值
        old_value = self[key]
        try:
            # 设置指定键的新值
            self[key] = value
            # 使用 yield 将当前对象返回给 with 语句的上下文
            yield self
        finally:
            # 恢复键的原始值
            self[key] = old_value
# 创建一个新的 _Options 对象并将其赋值给 plot_params 变量
plot_params = _Options()
```