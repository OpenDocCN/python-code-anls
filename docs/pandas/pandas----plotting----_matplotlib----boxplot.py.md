# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\boxplot.py`

```
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
)
import warnings

import matplotlib as mpl
import numpy as np

from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike

import pandas as pd
import pandas.core.common as com

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    maybe_adjust_figure,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from pandas._typing import MatplotlibColor


def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs) -> None:
    """Set the tick labels of a given axis.

    Due to https://github.com/matplotlib/matplotlib/pull/17266, we need to handle the
    case of repeated ticks (due to `FixedLocator`) and thus we duplicate the number of
    labels.
    """
    # 获取横轴或纵轴的刻度位置
    ticks = ax.get_xticks() if is_vertical else ax.get_yticks()
    # 如果刻度数与标签数不一致，则复制标签以匹配刻度数
    if len(ticks) != len(labels):
        i, remainder = divmod(len(ticks), len(labels))
        assert remainder == 0, remainder
        labels *= i
    # 根据是否是纵向，设置刻度标签
    if is_vertical:
        ax.set_xticklabels(labels, **kwargs)
    else:
        ax.set_yticklabels(labels, **kwargs)


class BoxPlot(LinePlot):
    @property
    def _kind(self) -> Literal["box"]:
        return "box"

    _layout_type = "horizontal"

    _valid_return_types = (None, "axes", "dict", "both")

    class BP(NamedTuple):
        # namedtuple to hold results
        ax: Axes
        lines: dict[str, list[Line2D]]

    def __init__(self, data, return_type: str = "axes", **kwargs) -> None:
        # 检查返回类型是否合法
        if return_type not in self._valid_return_types:
            raise ValueError("return_type must be {None, 'axes', 'dict', 'both'}")

        self.return_type = return_type
        # 调用父类的构造函数，初始化绘图设置
        MPLPlot.__init__(self, data, **kwargs)

        if self.subplots:
            # 如果是多子图模式，禁止标签轴共享，避免所有子图显示最后一列标签
            if self.orientation == "vertical":
                self.sharex = False
            else:
                self.sharey = False

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls, ax: Axes, y: np.ndarray, column_num=None, return_type: str = "axes", **kwds
    ) -> None:
        """
        绘制箱线图。

        Parameters:
        - ax: 绘制图形的轴对象
        - y: 绘制箱线图的数据
        - column_num: 列数（可选）
        - return_type: 返回类型，支持 'axes', 'dict', 'both' 中的一个
        - kwds: 其他关键字参数
        """
        # 在这里实现绘制箱线图的具体逻辑
        pass
    ):
        ys: np.ndarray | list[np.ndarray]
        # 如果 y 的维度为 2，则对每个元素调用 remove_na_arraylike 函数
        if y.ndim == 2:
            ys = [remove_na_arraylike(v) for v in y]
            # 箱线图在遇到空数组时会失败，因此需要添加一个 NaN
            #   如果有任何列是空的
            # GH 8181
            ys = [v if v.size > 0 else np.array([np.nan]) for v in ys]
        else:
            # 否则直接调用 remove_na_arraylike 函数
            ys = remove_na_arraylike(y)
        # 在轴上绘制箱线图，使用传入的关键字参数
        bp = ax.boxplot(ys, **kwds)

        # 根据 return_type 返回不同的结果
        if return_type == "dict":
            return bp, bp
        elif return_type == "both":
            return cls.BP(ax=ax, lines=bp), bp
        else:
            return ax, bp

    def _validate_color_args(self, color, colormap):
        # 如果 color 是 lib.no_default，则返回 None
        if color is lib.no_default:
            return None

        # 如果 colormap 不为 None，则发出警告信息
        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used "
                "simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )

        # 如果 color 是字典，则验证其键是否合法
        if isinstance(color, dict):
            valid_keys = ["boxes", "whiskers", "medians", "caps"]
            for key in color:
                if key not in valid_keys:
                    raise ValueError(
                        f"color dict contains invalid key '{key}'. "
                        f"The key must be either {valid_keys}"
                    )
        # 返回验证后的 color
        return color

    @cache_readonly
    def _color_attrs(self):
        # 获取默认情况下的标准颜色
        # 默认使用两种颜色，用于箱线图和中位线
        # 这里不需要 flier 的颜色，因为可以通过 ``sym`` 关键字参数指定
        return get_standard_colors(num_colors=3, colormap=self.colormap, color=None)

    @cache_readonly
    def _boxes_c(self):
        # 返回箱线图颜色属性的第一个元素
        return self._color_attrs[0]

    @cache_readonly
    def _whiskers_c(self):
        # 返回边缘线颜色属性的第一个元素
        return self._color_attrs[0]

    @cache_readonly
    def _medians_c(self):
        # 返回中位线颜色属性的第三个元素
        return self._color_attrs[2]

    @cache_readonly
    def _caps_c(self):
        # 返回边缘颜色属性的第一个元素
        return self._color_attrs[0]

    def _get_colors(
        self,
        num_colors=None,
        color_kwds: dict[str, MatplotlibColor]
        | MatplotlibColor
        | Collection[MatplotlibColor]
        | None = "color",
    ) -> None:
        # 这个方法暂时没有实现任何功能，只是用于占位

    def maybe_color_bp(self, bp) -> None:
        # 如果 self.color 是字典，则从中获取各个部分的颜色；否则使用默认颜色
        if isinstance(self.color, dict):
            boxes = self.color.get("boxes", self._boxes_c)
            whiskers = self.color.get("whiskers", self._whiskers_c)
            medians = self.color.get("medians", self._medians_c)
            caps = self.color.get("caps", self._caps_c)
        else:
            # 其他类型将被转发给 matplotlib
            # 如果为 None，则使用默认颜色
            boxes = self.color or self._boxes_c
            whiskers = self.color or self._whiskers_c
            medians = self.color or self._medians_c
            caps = self.color or self._caps_c

        # 构建颜色元组
        color_tup = (boxes, whiskers, medians, caps)
        # 调用 maybe_color_bp 函数，传入颜色元组和其他关键字参数
        maybe_color_bp(bp, color_tup=color_tup, **self.kwds)
    def _make_plot(self, fig: Figure) -> None:
        # 如果有子图存在，初始化返回对象为一个空的 pandas Series
        if self.subplots:
            self._return_obj = pd.Series(dtype=object)

            # 如果用户指定了 `by` 参数，重新创建迭代数据
            # 否则直接使用现有数据
            data = (
                create_iter_data_given_by(self.data, self._kind)
                if self.by is not None
                else self.data
            )

            # 遍历迭代数据，获取标签和数据
            # 错误："_iter_data" 方法的 "data" 参数类型与 "MPLPlot" 的定义不兼容
            for i, (label, y) in enumerate(self._iter_data(data=data)):  # type: ignore[arg-type]
                ax = self._get_ax(i)
                kwds = self.kwds.copy()

                # 当使用 `by` 参数时，设置子图标题以显示组别信息
                # 类似于 df.boxplot，并且需要对 y 应用 T 转置以提供正确的输入
                if self.by is not None:
                    y = y.T
                    ax.set_title(pprint_thing(label))

                    # 当 `by` 被指定时，ticklabels 将变为唯一的分组值
                    # 而不是标签，标签在这种情况下被用作子标题
                    # 错误："Index" 没有 "levels" 属性；也许应该使用 "nlevels"？
                    levels = self.data.columns.levels  # type: ignore[attr-defined]
                    ticklabels = [pprint_thing(col) for col in levels[0]]
                else:
                    ticklabels = [pprint_thing(label)]

                # 执行绘图操作，并返回结果和绘制的图形对象
                ret, bp = self._plot(
                    ax, y, column_num=i, return_type=self.return_type, **kwds
                )
                self.maybe_color_bp(bp)
                self._return_obj[label] = ret

                # 设置刻度标签，根据图的方向是否为垂直来确定
                _set_ticklabels(
                    ax=ax, labels=ticklabels, is_vertical=self.orientation == "vertical"
                )
        else:
            # 如果没有子图存在，将数据进行转置并获取第一个轴
            y = self.data.values.T
            ax = self._get_ax(0)
            kwds = self.kwds.copy()

            # 执行绘图操作，并返回结果和绘制的图形对象
            ret, bp = self._plot(
                ax, y, column_num=0, return_type=self.return_type, **kwds
            )
            self.maybe_color_bp(bp)
            self._return_obj = ret

            # 设置刻度标签，根据图的方向是否为垂直来确定
            labels = [pprint_thing(left) for left in self.data.columns]
            if not self.use_index:
                labels = [pprint_thing(key) for key in range(len(labels))]
            _set_ticklabels(
                ax=ax, labels=labels, is_vertical=self.orientation == "vertical"
            )

    def _make_legend(self) -> None:
        # 此方法暂时为空，用于在之后添加图例逻辑
        pass

    def _post_plot_logic(self, ax: Axes, data) -> None:
        # GH 45465: 确保箱线图不忽略 xlabel 和 ylabel
        # 如果存在 xlabel，则设置 x 轴标签
        if self.xlabel:
            ax.set_xlabel(pprint_thing(self.xlabel))
        # 如果存在 ylabel，则设置 y 轴标签
        if self.ylabel:
            ax.set_ylabel(pprint_thing(self.ylabel))
    # 定义一个方法orientation，返回类型为"horizontal"或"vertical"
    def orientation(self) -> Literal["horizontal", "vertical"]:
        # 如果self.kwds中的"vert"键对应的值为True，则返回"vertical"
        if self.kwds.get("vert", True):
            return "vertical"
        # 否则返回"horizontal"
        else:
            return "horizontal"

    # 定义一个属性方法result
    @property
    def result(self):
        # 如果self.return_type为None，则调用父类的result方法
        if self.return_type is None:
            return super().result
        # 否则返回self._return_obj
        else:
            return self._return_obj
# 当用户显式指定这些参数时，覆盖我们的默认设置；否则使用 Pandas 的设置
def maybe_color_bp(bp, color_tup, **kwds) -> None:
    if not kwds.get("boxprops"):
        # 设置箱体的颜色和透明度
        mpl.artist.setp(bp["boxes"], color=color_tup[0], alpha=1)
    if not kwds.get("whiskerprops"):
        # 设置须的颜色和透明度
        mpl.artist.setp(bp["whiskers"], color=color_tup[1], alpha=1)
    if not kwds.get("medianprops"):
        # 设置中位数线的颜色和透明度
        mpl.artist.setp(bp["medians"], color=color_tup[2], alpha=1)
    if not kwds.get("capprops"):
        # 设置箱线图顶部和底部线段（盖帽）的颜色和透明度
        mpl.artist.setp(bp["caps"], color=color_tup[3], alpha=1)


def _grouped_plot_by_column(
    plotf,
    data,
    columns=None,
    by=None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: tuple[float, float] | None = None,
    ax=None,
    layout=None,
    return_type=None,
    **kwargs,
):
    # 按照指定的 "by" 列分组数据
    grouped = data.groupby(by, observed=False)
    # 如果未指定列名，则根据除去 "by" 列的数值型数据列进行分组
    if columns is None:
        if not isinstance(by, (list, tuple)):
            by = [by]
        columns = data._get_numeric_data().columns.difference(by)
    # 计算需要创建的子图数量
    naxes = len(columns)
    # 创建子图
    fig, axes = create_subplots(
        naxes=naxes,
        sharex=kwargs.pop("sharex", True),
        sharey=kwargs.pop("sharey", True),
        figsize=figsize,
        ax=ax,
        layout=layout,
    )

    # 根据参数 "vert" 决定横轴或纵轴的标签位置
    xlabel, ylabel = kwargs.pop("xlabel", None), kwargs.pop("ylabel", None)
    if kwargs.get("vert", True):
        xlabel = xlabel or by
    else:
        ylabel = ylabel or by

    # 存储每个子图的返回值
    ax_values = []

    # 对每个子图和对应的列进行迭代
    for ax, col in zip(flatten_axes(axes), columns):
        # 获取当前列的分组数据
        gp_col = grouped[col]
        keys, values = zip(*gp_col)
        # 调用指定的绘图函数进行绘图，并返回结果
        re_plotf = plotf(keys, values, ax, xlabel=xlabel, ylabel=ylabel, **kwargs)
        # 设置子图的标题为列名
        ax.set_title(col)
        # 根据参数设置是否显示网格线
        ax.grid(grid)
        # 将返回结果添加到列表中
        ax_values.append(re_plotf)

    # 创建结果的 Pandas Series，索引为列名，值为绘图函数返回的结果
    result = pd.Series(ax_values, index=columns, copy=False)

    # 如果未指定返回类型，则返回子图对象
    if return_type is None:
        result = axes

    # 设置整个图的标题，指明按照哪个列名进行的分组
    byline = by[0] if len(by) == 1 else by
    fig.suptitle(f"Boxplot grouped by {byline}")

    # 可选地调整图的边界
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)

    # 返回结果对象
    return result


def boxplot(
    data,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    **kwds,
):
    import matplotlib.pyplot as plt

    # 验证返回类型是否有效
    if return_type not in BoxPlot._valid_return_types:
        raise ValueError("return_type must be {'axes', 'dict', 'both'}")

    # 如果输入数据是 Pandas Series，则转换为 DataFrame，并指定列名为 "x"
    if isinstance(data, ABCSeries):
        data = data.to_frame("x")
        column = "x"
    def _get_colors():
        # 定义一个内部函数来获取绘图颜色的规范列表，要求必须有3个颜色，因为方法maybe_color_bp会使用颜色在位置0和2。
        # 如果未提供颜色，则使用与DataFrame.plot.box相同的默认值。
        result_list = get_standard_colors(num_colors=3)
        # 从result_list中取出索引为0和2的元素组成新的结果列表
        result = np.take(result_list, [0, 0, 2])
        # 在结果列表末尾添加一个黑色元素
        result = np.append(result, "k")

        # 从kwds参数中弹出color键的值，如果有的话
        colors = kwds.pop("color", None)
        if colors:
            if is_dict_like(colors):
                # 如果colors参数是类字典结构，用用户指定的颜色替换结果数组中的颜色
                # 根据colors字典的键值，将"boxes"的颜色放在位置0，"whiskers"放在1，依此类推
                valid_keys = ["boxes", "whiskers", "medians", "caps"]
                key_to_index = dict(zip(valid_keys, range(4)))
                for key, value in colors.items():
                    if key in valid_keys:
                        result[key_to_index[key]] = value
                    else:
                        raise ValueError(
                            f"color dict contains invalid key '{key}'. "
                            f"The key must be either {valid_keys}"
                        )
            else:
                # 否则将result数组填充为同一种颜色
                result.fill(colors)

        # 返回最终的颜色列表result
        return result

    def plot_group(keys, values, ax: Axes, **kwds):
        # GH 45465: 在绘图之前需要弹出xlabel和ylabel
        xlabel, ylabel = kwds.pop("xlabel", None), kwds.pop("ylabel", None)
        if xlabel:
            # 如果有xlabel，设置坐标轴的x标签
            ax.set_xlabel(pprint_thing(xlabel))
        if ylabel:
            # 如果有ylabel，设置坐标轴的y标签
            ax.set_ylabel(pprint_thing(ylabel))

        # 将keys中的每个元素都格式化为字符串
        keys = [pprint_thing(x) for x in keys]
        # 将values中的每个元素转换为包含对象类型的NumPy数组，移除NaN值
        values = [np.asarray(remove_na_arraylike(v), dtype=object) for v in values]
        # 调用Axes的boxplot方法绘制箱线图，传入参数kwds
        bp = ax.boxplot(values, **kwds)
        if fontsize is not None:
            # 如果fontsize不为None，设置坐标轴标签的字体大小
            ax.tick_params(axis="both", labelsize=fontsize)

        # GH 45465: 当"vert"参数改变时，x/y轴标签也会改变
        _set_ticklabels(
            ax=ax, labels=keys, is_vertical=kwds.get("vert", True), rotation=rot
        )
        # 可能会对箱线图的颜色进行设置，调用maybe_color_bp函数
        maybe_color_bp(bp, color_tup=colors, **kwds)

        # 在多图情况下返回axes对象，也许以后会重新考虑 # 985
        if return_type == "dict":
            # 如果返回类型是字典，返回箱线图对象bp
            return bp
        elif return_type == "both":
            # 如果返回类型是both，返回BoxPlot.BP的实例对象，包含ax和bp
            return BoxPlot.BP(ax=ax, lines=bp)
        else:
            # 否则返回axes对象
            return ax

    # 调用_get_colors函数获取绘图颜色的规范列表
    colors = _get_colors()
    if column is None:
        # 如果column为None，将columns设为None
        columns = None
    elif isinstance(column, (list, tuple)):
        # 如果column是列表或元组类型，直接将columns设为column
        columns = column
    else:
        # 否则将column作为单元素列表赋值给columns
        columns = [column]

    if by is not None:
        # 如果by不为None，根据by列进行分组绘图
        # 更喜欢返回数组类型，以匹配子图布局的2D绘图
        # https://github.com/pandas-dev/pandas/pull/12216#issuecomment-241175580
        result = _grouped_plot_by_column(
            plot_group,
            data,
            columns=columns,
            by=by,
            grid=grid,
            figsize=figsize,
            ax=ax,
            layout=layout,
            return_type=return_type,
            **kwds,
        )
    # 如果没有指定 'by' 参数，则进入以下分支
    else:
        # 如果没有指定返回类型，则设定默认返回类型为 "axes"
        if return_type is None:
            return_type = "axes"
        # 如果指定了布局参数而 'by' 参数为 None，则抛出数值错误异常
        if layout is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")

        # 如果未指定绘图轴对象，则根据 figsize 创建绘图参数
        if ax is None:
            rc = {"figure.figsize": figsize} if figsize is not None else {}
            # 使用绘图参数创建 matplotlib 上下文，并获取当前绘图轴对象
            with mpl.rc_context(rc):
                ax = plt.gca()
        
        # 从数据中获取数值列数据
        data = data._get_numeric_data()
        
        # 获取数据列数
        naxes = len(data.columns)
        
        # 如果数据列数为 0，则抛出数值错误异常，因为没有数据可绘制箱线图
        if naxes == 0:
            raise ValueError(
                "boxplot method requires numerical columns, nothing to plot."
            )
        
        # 如果未指定列名，则使用数据的全部列名
        if columns is None:
            columns = data.columns
        else:
            # 否则，根据指定的列名选择数据
            data = data[columns]

        # 调用 plot_group 函数绘制分组箱线图，结果存储在 result 中
        result = plot_group(columns, data.values.T, ax, **kwds)
        
        # 设置是否显示网格
        ax.grid(grid)

    # 返回绘制的结果对象
    return result
# 定义一个方法用于绘制数据框的箱线图，支持多种参数和子图显示
def boxplot_frame(
    self,
    column=None,               # 指定绘制箱线图的列名或列的列表
    by=None,                   # 按照此列的值分组绘制箱线图
    ax=None,                   # 可选的 matplotlib Axes 对象，用于绘制箱线图
    fontsize: int | None = None,  # 字体大小设置，如果未指定则为 None
    rot: int = 0,               # x 轴刻度旋转角度，默认为 0
    grid: bool = True,          # 是否显示网格线，默认显示
    figsize: tuple[float, float] | None = None,  # 图像尺寸，可以指定为 None
    layout=None,               # 子图布局设置
    return_type=None,          # 返回对象类型设置
    **kwds,                    # 其他关键字参数，传递给底层的箱线图绘制函数
):
    import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块

    # 调用 pandas 的箱线图绘制函数 boxplot，返回绘图的 Axes 对象
    ax = boxplot(
        self,                   # 数据源，数据框
        column=column,          # 列名或列表
        by=by,                  # 分组依据
        ax=ax,                  # matplotlib Axes 对象
        fontsize=fontsize,      # 字体大小
        grid=grid,              # 是否显示网格线
        rot=rot,                # x 轴刻度旋转角度
        figsize=figsize,        # 图像尺寸
        layout=layout,          # 子图布局设置
        return_type=return_type,  # 返回对象类型设置
        **kwds,                 # 其他关键字参数
    )

    # 如果是交互模式，重新绘制图形
    plt.draw_if_interactive()

    # 返回绘制的 Axes 对象
    return ax


# 定义一个方法用于按分组对象绘制数据框的箱线图
def boxplot_frame_groupby(
    grouped,                   # 分组后的数据框或序列
    subplots: bool = True,     # 是否创建子图，默认为 True
    column=None,               # 列名或列的列表
    fontsize: int | None = None,  # 字体大小设置，如果未指定则为 None
    rot: int = 0,               # x 轴刻度旋转角度，默认为 0
    grid: bool = True,          # 是否显示网格线，默认显示
    ax=None,                   # 可选的 matplotlib Axes 对象，用于绘制箱线图
    figsize: tuple[float, float] | None = None,  # 图像尺寸，可以指定为 None
    layout=None,               # 子图布局设置
    sharex: bool = False,      # 是否共享 x 轴刻度
    sharey: bool = True,       # 是否共享 y 轴刻度
    **kwds,                    # 其他关键字参数，传递给底层的箱线图绘制函数
):
    # 如果设置为创建子图
    if subplots is True:
        # 计算分组的数量
        naxes = len(grouped)
        # 创建子图，返回 matplotlib 的 Figure 和 Axes 对象
        fig, axes = create_subplots(
            naxes=naxes,         # 子图数量
            squeeze=False,       # 是否压缩
            ax=ax,               # matplotlib Axes 对象
            sharex=sharex,       # 是否共享 x 轴
            sharey=sharey,       # 是否共享 y 轴
            figsize=figsize,     # 图像尺寸
            layout=layout,       # 子图布局设置
        )
        data = {}
        # 遍历分组数据和对应的 Axes 对象
        for (key, group), ax in zip(grouped, flatten_axes(axes)):
            # 在当前 Axes 上绘制分组数据的箱线图
            d = group.boxplot(
                ax=ax,           # matplotlib Axes 对象
                column=column,   # 列名或列的列表
                fontsize=fontsize,  # 字体大小设置
                rot=rot,         # x 轴刻度旋转角度
                grid=grid,       # 是否显示网格线
                **kwds,          # 其他关键字参数
            )
            # 设置子图的标题为分组键的字符串表示形式
            ax.set_title(pprint_thing(key))
            # 将每个分组数据的绘图结果存储在字典中
            data[key] = d
        # 将数据字典封装成 pandas 的 Series 对象
        ret = pd.Series(data)
        # 调整图形的布局和间距
        maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    else:
        # 将分组的键和数据框的列表解压缩为两个分别存储键和数据框的元组
        keys, frames = zip(*grouped)
        # 使用 pd.concat 方法将多个数据框按列合并为一个新的数据框
        df = pd.concat(frames, keys=keys, axis=1)

        # 如果指定了列名，将其转换为列表形式
        if column is not None:
            column = com.convert_to_list_like(column)
            # 创建包含组合键和列名的多级索引
            multi_key = pd.MultiIndex.from_product([keys, column])
            column = list(multi_key.values)
        
        # 在合并后的数据框上绘制箱线图，返回绘图的对象
        ret = df.boxplot(
            column=column,       # 列名或列的列表
            fontsize=fontsize,  # 字体大小设置
            rot=rot,             # x 轴刻度旋转角度
            grid=grid,           # 是否显示网格线
            ax=ax,               # matplotlib Axes 对象
            figsize=figsize,     # 图像尺寸
            layout=layout,       # 子图布局设置
            **kwds,              # 其他关键字参数
        )
    
    # 返回绘制的对象
    return ret
```