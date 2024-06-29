# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\hist.py`

```
    # 导入必要的模块和类
    from __future__ import annotations

    from typing import (
        TYPE_CHECKING,
        Any,
        Literal,
        final,
    )

    import numpy as np

    from pandas.core.dtypes.common import (
        is_integer,
        is_list_like,
    )
    from pandas.core.dtypes.generic import (
        ABCDataFrame,
        ABCIndex,
    )
    from pandas.core.dtypes.missing import (
        isna,
        remove_na_arraylike,
    )

    from pandas.io.formats.printing import pprint_thing
    from pandas.plotting._matplotlib.core import (
        LinePlot,
        MPLPlot,
    )
    from pandas.plotting._matplotlib.groupby import (
        create_iter_data_given_by,
        reformat_hist_y_given_by,
    )
    from pandas.plotting._matplotlib.misc import unpack_single_str_list
    from pandas.plotting._matplotlib.tools import (
        create_subplots,
        flatten_axes,
        maybe_adjust_figure,
        set_ticks_props,
    )

    # 如果是类型检查，则导入类型相关的模块
    if TYPE_CHECKING:
        from matplotlib.axes import Axes
        from matplotlib.container import BarContainer
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon

        from pandas._typing import PlottingOrientation

        from pandas import (
            DataFrame,
            Series,
        )


    class HistPlot(LinePlot):
        @property
        # 指定属性 _kind 的类型为 Literal 类型 "hist" 或 "kde"
        def _kind(self) -> Literal["hist", "kde"]:
            return "hist"

        def __init__(
            self,
            data,
            bins: int | np.ndarray | list[np.ndarray] = 10,
            bottom: int | np.ndarray = 0,
            *,
            range=None,
            weights=None,
            **kwargs,
        ) -> None:
            # 如果 bottom 是类似列表的对象，则转换为 numpy 数组
            if is_list_like(bottom):
                bottom = np.array(bottom)
            self.bottom = bottom

            # 设置 _bin_range 和 weights 属性
            self._bin_range = range
            self.weights = weights

            # 从 kwargs 中获取 xlabel 和 ylabel，作为属性
            self.xlabel = kwargs.get("xlabel")
            self.ylabel = kwargs.get("ylabel")
            # 调用 MPLPlot 的构造函数，而不是 LinePlot.__init__，以避免填充 NaN
            MPLPlot.__init__(self, data, **kwargs)

            # 调用 _adjust_bins 方法来调整 bins
            self.bins = self._adjust_bins(bins)

        def _adjust_bins(self, bins: int | np.ndarray | list[np.ndarray]):
            # 如果 bins 是整数，则根据数据计算 bins
            if is_integer(bins):
                if self.by is not None:
                    by_modified = unpack_single_str_list(self.by)
                    grouped = self.data.groupby(by_modified)[self.columns]
                    bins = [self._calculate_bins(group, bins) for key, group in grouped]
                else:
                    bins = self._calculate_bins(self.data, bins)
            return bins

        def _calculate_bins(self, data: Series | DataFrame, bins) -> np.ndarray:
            """根据数据计算 bins"""
            # 从数据中提取数值类型的数据
            nd_values = data.infer_objects()._get_numeric_data()
            values = nd_values.values
            if nd_values.ndim == 2:
                values = values.reshape(-1)
            values = values[~isna(values)]

            # 使用 numpy 的直方图计算函数计算 bins 边界
            return np.histogram_bin_edges(values, bins=bins, range=self._bin_range)

        # error: Signature of "_plot" incompatible with supertype "LinePlot"
        @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        y: np.ndarray,
        style=None,
        bottom: int | np.ndarray = 0,
        column_num: int = 0,
        stacking_id=None,
        *,
        bins,
        **kwds,
        # might return a subset from the possible return types of Axes.hist(...)[2]?
    ) -> BarContainer | Polygon | list[BarContainer | Polygon]:
        # 如果 column_num 为 0，则初始化堆叠图的基础值
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)

        base = np.zeros(len(bins) - 1)
        # 获取堆叠图的底部值，并加上当前数据的堆叠值
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds["label"])
        # 忽略 style 参数

        # 绘制直方图，并返回生成的图形对象 patches
        n, bins, patches = ax.hist(y, bins=bins, bottom=bottom, **kwds)
        
        # 更新堆叠图的值
        cls._update_stacker(ax, stacking_id, n)
        
        # 返回绘制的图形对象列表 patches
        return patches

    def _make_plot(self, fig: Figure) -> None:
        # 获取颜色设置
        colors = self._get_colors()
        # 获取堆叠图的标识符
        stacking_id = self._get_stacking_id()

        # 如果存在按组绘制的数据，重新创建迭代数据
        data = (
            create_iter_data_given_by(self.data, self._kind)
            if self.by is not None
            else self.data
        )

        # 遍历数据，i 为索引，label 为标签名，y 为数据
        for i, (label, y) in enumerate(self._iter_data(data=data)):  # type: ignore[arg-type]
            # 获取对应的子图 ax
            ax = self._get_ax(i)

            # 复制关键字参数 kwds
            kwds = self.kwds.copy()
            # 如果指定了颜色，则设置图形的颜色
            if self.color is not None:
                kwds["color"] = self.color

            # 对标签进行格式化输出
            label = pprint_thing(label)
            # 标记正确的标签位置
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label

            # 应用风格和颜色设置
            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds["style"] = style

            # 应用绘图关键字参数设置
            self._make_plot_keywords(kwds, y)

            # 当应用分组时，设置 bins 为当前组的数据，标签为列名
            if self.by is not None:
                kwds["bins"] = kwds["bins"][i]
                kwds["label"] = self.columns
                kwds.pop("color")

            # 如果存在权重设置，则应用权重到当前数据
            if self.weights is not None:
                kwds["weights"] = type(self)._get_column_weights(self.weights, i, y)

            # 根据分组情况重新格式化 y 数据
            y = reformat_hist_y_given_by(y, self.by)

            # 绘制图形并获取图形对象
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)

            # 当应用分组时，设置子图标题以显示分组信息
            if self.by is not None:
                ax.set_title(pprint_thing(label))

            # 将图例句柄和标签添加到图例中
            # 注意：artists 是图形对象的列表，取第一个元素作为图例处理
            self._append_legend_handles_labels(artists[0], label)  # type: ignore[index,arg-type]
    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        """
        将 BoxPlot/KdePlot 的属性合并到传入的 kwds 中
        """
        # y is required for KdePlot
        # 设置 kwds 中的 "bottom" 键为当前对象的 self.bottom 属性值
        kwds["bottom"] = self.bottom
        # 设置 kwds 中的 "bins" 键为当前对象的 self.bins 属性值
        kwds["bins"] = self.bins

    @final
    @staticmethod
    def _get_column_weights(weights, i: int, y):
        """
        获取列的权重数组，用于数据的加权计算

        允许 weights 是多维数组，例如 (10, 2) 的数组，每次迭代调用一个子数组 (10,)。
        如果用户提供的是 1 维数组，假定权重在所有迭代中都相同。
        """
        # 如果 weights 不为 None
        if weights is not None:
            # 如果 weights 的维度不为 1 且最后一个维度不为 1
            if np.ndim(weights) != 1 and np.shape(weights)[-1] != 1:
                try:
                    # 取出 weights 的第 i 列
                    weights = weights[:, i]
                except IndexError as err:
                    raise ValueError(
                        "weights must have the same shape as data, "
                        "or be a single column"
                    ) from err
            # 将 weights 中非空数据对应的 y 的索引位置的数据提取出来
            weights = weights[~isna(y)]
        return weights

    def _post_plot_logic(self, ax: Axes, data) -> None:
        """
        根据对象的 orientation 属性设置图表的坐标轴标签
        """
        # 如果 orientation 属性为 "horizontal"
        if self.orientation == "horizontal":
            # 设置 X 轴标签为 "Frequency" 或者 self.xlabel 的值（如果不为 None）
            ax.set_xlabel(
                "Frequency" if self.xlabel is None else self.xlabel  # type: ignore[arg-type]
            )
            # 设置 Y 轴标签为 self.ylabel 的值
            ax.set_ylabel(self.ylabel)  # type: ignore[arg-type]
        else:
            # 设置 X 轴标签为 self.xlabel 的值
            ax.set_xlabel(self.xlabel)  # type: ignore[arg-type]
            # 设置 Y 轴标签为 "Frequency" 或者 self.ylabel 的值（如果不为 None）
            ax.set_ylabel(
                "Frequency" if self.ylabel is None else self.ylabel  # type: ignore[arg-type]
            )

    @property
    def orientation(self) -> PlottingOrientation:
        """
        返回对象的图表方向属性

        如果 kwds 中的 "orientation" 键为 "horizontal"，则返回 "horizontal"，
        否则返回 "vertical"。
        """
        if self.kwds.get("orientation", None) == "horizontal":
            return "horizontal"
        else:
            return "vertical"
class KdePlot(HistPlot):
    @property
    def _kind(self) -> Literal["kde"]:
        return "kde"

    @property
    def orientation(self) -> Literal["vertical"]:
        return "vertical"

    def __init__(
        self, data, bw_method=None, ind=None, *, weights=None, **kwargs
    ) -> None:
        # Do not call LinePlot.__init__ which may fill nan
        # 继承自父类 MPLPlot 的初始化方法，不调用 LinePlot.__init__ 是为了避免填充 NaN 值
        MPLPlot.__init__(self, data, **kwargs)
        self.bw_method = bw_method  # 设置核密度估计方法的参数
        self.ind = ind  # 设置用于估计密度的指数
        self.weights = weights  # 设置权重参数

    @staticmethod
    def _get_ind(y: np.ndarray, ind):
        if ind is None:
            # np.nanmax() and np.nanmin() ignores the missing values
            # 如果未提供指数值，使用样本数据范围自动生成指数数组
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                1000,
            )
        elif is_integer(ind):
            # 如果提供的指数是整数，根据指数数量生成指数数组
            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(
                np.nanmin(y) - 0.5 * sample_range,
                np.nanmax(y) + 0.5 * sample_range,
                ind,
            )
        return ind  # 返回生成的指数数组

    @classmethod
    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        y: np.ndarray,
        style=None,
        bw_method=None,
        ind=None,
        column_num=None,
        stacking_id: int | None = None,
        **kwds,
    ):
        from scipy.stats import gaussian_kde

        y = remove_na_arraylike(y)  # 移除 y 中的 NaN 值
        gkde = gaussian_kde(y, bw_method=bw_method)  # 创建高斯核密度估计对象

        y = gkde.evaluate(ind)  # 计算估计的密度值
        lines = MPLPlot._plot(ax, ind, y, style=style, **kwds)  # 调用父类的绘图方法
        return lines  # 返回绘制的线条对象

    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        kwds["bw_method"] = self.bw_method  # 设置绘图参数中的核密度估计方法
        kwds["ind"] = type(self)._get_ind(y, ind=self.ind)  # 设置绘图参数中的指数数组

    def _post_plot_logic(self, ax: Axes, data) -> None:
        ax.set_ylabel("Density")  # 设置 y 轴标签为 "Density"


def _grouped_plot(
    plotf,
    data: Series | DataFrame,
    column=None,
    by=None,
    numeric_only: bool = True,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = True,
    layout=None,
    rot: float = 0,
    ax=None,
    **kwargs,
):
    # error: Non-overlapping equality check (left operand type: "Optional[Tuple[float,
    # float]]", right operand type: "Literal['default']")
    if figsize == "default":  # type: ignore[comparison-overlap]
        # allowed to specify mpl default with 'default'
        raise ValueError(
            "figsize='default' is no longer supported. "
            "Specify figure size by tuple instead"
        )

    grouped = data.groupby(by)  # 按照指定的分组键进行数据分组
    if column is not None:
        grouped = grouped[column]  # 如果指定了列名，则从分组结果中选择该列数据

    naxes = len(grouped)  # 计算分组后的数量
    fig, axes = create_subplots(
        naxes=naxes, figsize=figsize, sharex=sharex, sharey=sharey, ax=ax, layout=layout
    )  # 创建子图布局
    # 使用 flatten_axes 函数展开 axes，然后使用 grouped 进行迭代
    for ax, (key, group) in zip(flatten_axes(axes), grouped):
        # 如果 numeric_only 为 True 并且 group 是 ABCDataFrame 的实例，则获取其数值数据
        if numeric_only and isinstance(group, ABCDataFrame):
            group = group._get_numeric_data()
        # 调用 plotf 函数绘制 group 数据到 ax 上，使用 kwargs 作为额外参数
        plotf(group, ax, **kwargs)
        # 为当前 ax 设置标题，使用 pprint_thing 函数格式化 key
        ax.set_title(pprint_thing(key))

    # 返回绘制的图形 fig 和 axes 对象
    return fig, axes
def _grouped_hist(
    data: Series | DataFrame,
    column=None,
    by=None,
    ax=None,
    bins: int = 50,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = False,
    rot: float = 90,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    legend: bool = False,
    **kwargs,
):
    """
    Grouped histogram

    Parameters
    ----------
    data : Series or DataFrame
        The data to be plotted, can be a Series or DataFrame.
    column : object, optional
        If specified and data is a DataFrame, the column to use for plotting.
    by : object, optional
        Grouping variable that will produce elements of the same type as data.
    ax : axes, optional
        Matplotlib axes object to draw the plot onto, otherwise uses the current axes.
    bins : int, default 50
        Number of bins for the histogram.
    figsize : tuple, optional
        Figure size in inches (width, height).
    layout : optional
        Unused parameter.
    sharex : bool, default False
        Whether to share x-axis among subplots.
    sharey : bool, default False
        Whether to share y-axis among subplots.
    rot : float, default 90
        Rotation angle of x-axis labels.
    grid : bool, default True
        Whether to show grid lines.
    xlabelsize : int or None, optional
        Font size of the x-axis labels.
    xrot : float or None, optional
        Rotation angle of x-axis tick labels.
    ylabelsize : int or None, optional
        Font size of the y-axis labels.
    yrot : float or None, optional
        Rotation angle of y-axis tick labels.
    legend : bool, default False
        Whether to display legend.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib.Axes.hist.

    Returns
    -------
    collection of Matplotlib Axes
        Returns the Matplotlib Axes containing the plotted histograms.
    """

    if legend:
        assert "label" not in kwargs
        if data.ndim == 1:
            kwargs["label"] = data.name
        elif column is None:
            kwargs["label"] = data.columns
        else:
            kwargs["label"] = column

    def plot_group(group, ax) -> None:
        """
        Plot histogram for a grouped data.

        Parameters
        ----------
        group : Series or DataFrame
            Grouped data to plot as a histogram.
        ax : axes
            Matplotlib axes object to draw the plot onto.

        Returns
        -------
        None
        """
        ax.hist(group.dropna().values, bins=bins, **kwargs)
        if legend:
            ax.legend()

    if xrot is None:
        xrot = rot

    fig, axes = _grouped_plot(
        plot_group,
        data,
        column=column,
        by=by,
        sharex=sharex,
        sharey=sharey,
        ax=ax,
        figsize=figsize,
        layout=layout,
        rot=rot,
    )

    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )

    maybe_adjust_figure(
        fig, bottom=0.15, top=0.9, left=0.1, right=0.9, hspace=0.5, wspace=0.3
    )
    return axes
    # 如果 by 参数为 None
    if by is None:
        # 如果关键字参数中包含 'layout'，则抛出值错误异常
        if kwds.get("layout", None) is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")
        
        # 当绘图接口还不够统一时的临时处理方法
        # 从关键字参数中弹出 'figure'，如果不存在则创建一个新的图形对象或使用当前图形对象
        fig = kwds.pop(
            "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
        )
        
        # 如果指定了 figsize 并且与当前图形大小不同，则设置图形大小
        if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
            fig.set_size_inches(*figsize, forward=True)
        
        # 如果未指定轴对象，则获取当前图形的主轴对象
        if ax is None:
            ax = fig.gca()
        # 否则，如果传入的轴对象不属于当前图形，则抛出断言错误
        elif ax.get_figure() != fig:
            raise AssertionError("passed axis not bound to passed figure")
        
        # 从数据中删除缺失值并获取其数值
        values = self.dropna().values
        
        # 如果设置了图例选项，则在图上标注数据集名称
        if legend:
            kwds["label"] = self.name
        
        # 绘制直方图，使用指定的 bins 参数和其他关键字参数
        ax.hist(values, bins=bins, **kwds)
        
        # 如果设置了图例选项，则在轴上显示图例
        if legend:
            ax.legend()
        
        # 设置轴上的网格显示与否
        ax.grid(grid)
        
        # 将单个轴对象放入 numpy 数组中
        axes = np.array([ax])
        
        # 设置刻度的属性
        set_ticks_props(
            axes,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
        )
    
    else:
        # 如果关键字参数中包含 'figure'，则不支持使用 'by' 参数，因为会创建新的 'Figure' 实例
        if "figure" in kwds:
            raise ValueError(
                "Cannot pass 'figure' when using the "
                "'by' argument, since a new 'Figure' instance will be created"
            )
        
        # 调用 _grouped_hist 函数，根据 'by' 参数对数据进行分组并绘制分组直方图
        axes = _grouped_hist(
            self,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )
    
    # 如果轴对象具有属性 'ndim'
    if hasattr(axes, "ndim"):
        # 如果轴对象是一维且长度为1，则返回其第一个元素
        if axes.ndim == 1 and len(axes) == 1:
            return axes[0]
    
    # 返回轴对象
    return axes
def hist_frame(
    data: DataFrame,
    column=None,
    by=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
    ax=None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    layout=None,
    bins: int = 10,
    legend: bool = False,
    **kwds,
):
    # 如果同时设置了 legend 和 label，抛出异常
    if legend and "label" in kwds:
        raise ValueError("Cannot use both legend and label")
    
    # 如果指定了 by 参数，则调用 _grouped_hist 函数处理并返回 axes 对象
    if by is not None:
        axes = _grouped_hist(
            data,
            column=column,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            layout=layout,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            legend=legend,
            **kwds,
        )
        return axes

    # 如果指定了 column 参数，则仅保留指定的列
    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndex)):
            column = [column]
        data = data[column]
    
    # 选择数据框中的数值型或日期时间型列，排除时间增量列
    data = data.select_dtypes(
        include=(np.number, "datetime64", "datetimetz"), exclude="timedelta"
    )
    
    # 获取数据框中列的数量
    naxes = len(data.columns)

    # 如果数据框中没有可用于绘图的列，抛出异常
    if naxes == 0:
        raise ValueError(
            "hist method requires numerical or datetime columns, nothing to plot."
        )

    # 创建子图，并返回图形对象 fig 和轴对象 axes
    fig, axes = create_subplots(
        naxes=naxes,
        ax=ax,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
    )
    
    # 判断是否可以设置标签
    can_set_label = "label" not in kwds

    # 对每个子图对象 axes 和对应的列 col 进行迭代
    for ax, col in zip(flatten_axes(axes), data.columns):
        # 如果设置了 legend 并且可以设置标签，则为每个子图设置标签
        if legend and can_set_label:
            kwds["label"] = col
        
        # 绘制直方图，并设置相关参数
        ax.hist(data[col].dropna().values, bins=bins, **kwds)
        ax.set_title(col)  # 设置子图标题
        ax.grid(grid)  # 设置是否显示网格
        if legend:
            ax.legend()  # 如果需要图例，则显示图例

    # 设置轴标签的属性
    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )
    
    # 可能调整图形的布局，设置子图之间的间距
    maybe_adjust_figure(fig, wspace=0.3, hspace=0.3)

    return axes  # 返回绘制的轴对象
```