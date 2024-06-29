# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\core.py`

```
from __future__ import annotations

# 引入未来的语法特性，允许在类型注解中使用字符串形式的类型名称

from abc import (
    ABC,
    abstractmethod,
)
# 导入抽象基类（ABC）和抽象方法装饰器（abstractmethod）

from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
# 导入集合类抽象基类中的哈希可变、可迭代、迭代器和序列抽象类

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    final,
)
# 导入类型提示中的类型检查标记、任意类型、字面值类型、类型转换和最终方法装饰器

import warnings
# 导入警告模块

import matplotlib as mpl
# 导入 matplotlib 库并将其命名为 mpl

import numpy as np
# 导入 numpy 库并将其命名为 np

from pandas._libs import lib
# 导入 pandas 私有库中的 lib 模块

from pandas.errors import AbstractMethodError
# 导入 pandas 错误模块中的抽象方法错误类

from pandas.util._decorators import cache_readonly
# 导入 pandas 工具模块中的只读缓存装饰器

from pandas.util._exceptions import find_stack_level
# 导入 pandas 工具模块中的查找堆栈级别异常

from pandas.core.dtypes.common import (
    is_any_real_numeric_dtype,
    is_bool,
    is_float,
    is_float_dtype,
    is_hashable,
    is_integer,
    is_integer_dtype,
    is_iterator,
    is_list_like,
    is_number,
    is_numeric_dtype,
)
# 从 pandas 核心数据类型模块中导入常见数据类型判断函数

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
# 从 pandas 核心数据类型模块中导入分类数据类型和扩展数据类型

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
    ABCSeries,
)
# 从 pandas 核心数据类型模块中导入数据框、日期时间索引、索引、多级索引、周期索引和系列的抽象基类

from pandas.core.dtypes.missing import isna
# 从 pandas 核心数据类型模块中导入缺失值判断函数

import pandas.core.common as com
# 导入 pandas 核心公共模块中的 com 别名

from pandas.core.frame import DataFrame
# 从 pandas 核心数据框模块中导入数据框类

from pandas.util.version import Version
# 导入 pandas 工具模块中的版本类

from pandas.io.formats.printing import pprint_thing
# 从 pandas 输入输出格式打印模块中导入格式化输出函数

from pandas.plotting._matplotlib import tools
# 从 pandas 绘图 matplotlib 子模块中导入工具模块

from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters
# 从 pandas 绘图 matplotlib 子模块中导入注册 pandas 到 matplotlib 转换器函数

from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by
# 从 pandas 绘图 matplotlib 子模块中导入使用分组数据重构数据函数

from pandas.plotting._matplotlib.misc import unpack_single_str_list
# 从 pandas 绘图 matplotlib 子模块中导入解包单个字符串列表函数

from pandas.plotting._matplotlib.style import get_standard_colors
# 从 pandas 绘图 matplotlib 子模块中导入获取标准颜色函数

from pandas.plotting._matplotlib.timeseries import (
    decorate_axes,
    format_dateaxis,
    maybe_convert_index,
    maybe_resample,
    use_dynamic_x,
)
# 从 pandas 绘图 matplotlib 时间序列子模块中导入装饰坐标轴、格式化日期轴、可能转换索引、可能重采样和使用动态 x 轴函数

from pandas.plotting._matplotlib.tools import (
    create_subplots,
    flatten_axes,
    format_date_labels,
    get_all_lines,
    get_xlim,
    handle_shared_axes,
)
# 从 pandas 绘图 matplotlib 工具子模块中导入创建子图、展平坐标轴、格式化日期标签、获取所有线条、获取 x 轴限制和处理共享坐标轴函数

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure

    from pandas._typing import (
        IndexLabel,
        NDFrameT,
        PlottingOrientation,
        npt,
    )
    from pandas import (
        Index,
        Series,
    )
# 如果是类型检查阶段，则导入类型检查相关的类和类型定义
    # 初始化方法，用于创建一个新的图形/图表对象
    def __init__(
        self,
        data,  # 数据输入，可以是 Series 或 DataFrame
        kind=None,  # 图表类型，例如 'line'、'bar' 等
        by: IndexLabel | None = None,  # 指定分组的依据列标签
        subplots: bool | Sequence[Sequence[str]] = False,  # 是否创建多个子图，可以是布尔值或指定子图结构的序列
        sharex: bool | None = None,  # 是否共享 x 轴
        sharey: bool = False,  # 是否共享 y 轴，默认为 False
        use_index: bool = True,  # 是否使用数据的索引作为轴刻度
        figsize: tuple[float, float] | None = None,  # 图形的尺寸，可以指定为 (宽度, 高度)
        grid=None,  # 是否显示网格线
        legend: bool | str = True,  # 是否显示图例，或者指定图例的位置
        rot=None,  # x 轴刻度标签的旋转角度
        ax=None,  # 绘图时使用的轴对象
        fig=None,  # 绘图时使用的图形对象
        title=None,  # 图表的标题
        xlim=None,  # x 轴的显示范围
        ylim=None,  # y 轴的显示范围
        xticks=None,  # x 轴的刻度位置
        yticks=None,  # y 轴的刻度位置
        xlabel: Hashable | None = None,  # x 轴的标签
        ylabel: Hashable | None = None,  # y 轴的标签
        fontsize: int | None = None,  # 字体大小
        secondary_y: bool | tuple | list | np.ndarray = False,  # 是否显示第二个 y 轴
        colormap=None,  # 颜色映射
        table: bool = False,  # 是否绘制表格
        layout=None,  # 子图布局
        include_bool: bool = False,  # 是否包含布尔值列
        column: IndexLabel | None = None,  # 指定操作的列标签
        *,
        logx: bool | None | Literal["sym"] = False,  # 是否对 x 轴进行对数缩放
        logy: bool | None | Literal["sym"] = False,  # 是否对 y 轴进行对数缩放
        loglog: bool | None | Literal["sym"] = False,  # 是否同时对 x 和 y 轴进行对数缩放
        mark_right: bool = True,  # 是否在右侧 y 轴上标记
        stacked: bool = False,  # 是否堆叠柱状图
        label: Hashable | None = None,  # 数据标签
        style=None,  # 绘图风格
        **kwds,  # 其他关键字参数，用于接收用户自定义的参数
    ):
        pass  # 初始化方法不执行具体操作，只定义了参数和默认值，待实例化时根据参数创建图表对象

    @final
    @staticmethod
    def _validate_sharex(sharex: bool | None, ax, by) -> bool:
        # 验证 sharex 参数，确保其为布尔值或 None
        if sharex is None:
            # 如果定义了 by 参数且未指定轴对象 ax，则使用子图模式，sharex 应为 False
            if ax is None and by is None:
                sharex = True  # 未定义轴对象且没有指定 by 参数时，sharex 应为 True
            else:
                sharex = False  # 否则，sharex 应为 False
        elif not is_bool(sharex):  # 如果 sharex 不是布尔值，则抛出类型错误异常
            raise TypeError("sharex must be a bool or None")
        return bool(sharex)  # 返回经验证后的 sharex 值

    @classmethod
    def _validate_log_kwd(
        cls,
        kwd: str,
        value: bool | None | Literal["sym"],
    ) -> bool | None | Literal["sym"]:
        # 验证对数相关的关键字参数，确保其为布尔值、None 或 "sym"
        if (
            value is None
            or isinstance(value, bool)
            or (isinstance(value, str) and value == "sym")
        ):
            return value  # 如果符合要求，则返回原值
        raise ValueError(
            f"keyword '{kwd}' should be bool, None, or 'sym', not '{value}'"
        )  # 否则，抛出值错误异常，提示参数应为布尔值、None 或 "sym"

    @final
    @staticmethod
    def _validate_subplots_kwarg(
        subplots: bool | Sequence[Sequence[str]], data: Series | DataFrame, kind: str
    ):
        pass  # 验证子图参数的方法，目前只定义了参数类型，未具体实现验证逻辑
    def _validate_color_args(self, color, colormap):
        if color is lib.no_default:
            # 如果未由用户提供颜色参数
            if "colors" in self.kwds and colormap is not None:
                # 如果关键字参数中包含'colors'且同时指定了colormap，发出警告
                warnings.warn(
                    "'color' and 'colormap' cannot be used simultaneously. "
                    "Using 'color'",
                    stacklevel=find_stack_level(),
                )
            return None
        if self.nseries == 1 and color is not None and not is_list_like(color):
            # 支持单系列数据的绘图，使用单一颜色
            color = [color]

        if isinstance(color, tuple) and self.nseries == 1 and len(color) in (3, 4):
            # 支持系列绘图中使用 RGB 和 RGBA 元组
            color = [color]

        if colormap is not None:
            # 如果同时指定了colormap，发出警告
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )

        if self.style is not None:
            if is_list_like(self.style):
                styles = self.style
            else:
                styles = [self.style]
            # 只需匹配一个样式
            for s in styles:
                if _color_in_style(s):
                    # 如果样式字符串中包含颜色符号，并且同时使用了'color'关键字参数，抛出错误
                    raise ValueError(
                        "Cannot pass 'style' string with a color symbol and "
                        "'color' keyword argument. Please use one or the "
                        "other or pass 'style' without a color symbol"
                    )
        return color

    @final
    @staticmethod
    def _iter_data(
        data: DataFrame | dict[Hashable, Series | DataFrame],
    ) -> Iterator[tuple[Hashable, np.ndarray]]:
        for col, values in data.items():
            # 原本使用values.values，现在添加np.asarray(...)以保持类型一致性
            yield col, np.asarray(values.values)

    def _get_nseries(self, data: Series | DataFrame) -> int:
        # 当显式指定了'by'时，分组数据的大小将决定子图数量（self.nseries）
        if data.ndim == 1:
            return 1
        elif self.by is not None and self._kind == "hist":
            return len(self._grouped)
        elif self.by is not None and self._kind == "box":
            return len(self.columns)
        else:
            return data.shape[1]

    @final
    @property
    def nseries(self) -> int:
        # 返回数据的系列数目
        return self._get_nseries(self.data)

    @final
    def generate(self) -> None:
        # 生成绘图
        self._compute_plot_data()
        fig = self.fig
        self._make_plot(fig)
        self._add_table()
        self._make_legend()
        self._adorn_subplots(fig)

        for ax in self.axes:
            self._post_plot_logic_common(ax)
            self._post_plot_logic(ax, self.data)
    def _has_plotted_object(ax: Axes) -> bool:
        """检查 axes 对象是否包含绘图数据"""
        # 检查是否有线条、艺术家或容器对象
        return len(ax.lines) != 0 or len(ax.artists) != 0 or len(ax.containers) != 0

    @final
    def _maybe_right_yaxis(self, ax: Axes, axes_num: int) -> Axes:
        """根据条件可能返回右侧的 y 轴 Axes 对象"""
        if not self.on_right(axes_num):
            # 如果不需要右侧轴，则返回当前轴
            return self._get_ax_layer(ax)

        if hasattr(ax, "right_ax"):
            # 如果 ax 有 right_ax 属性，则 ax 必须是左侧轴，返回其右侧轴
            return ax.right_ax
        elif hasattr(ax, "left_ax"):
            # 如果 ax 有 left_ax 属性，则 ax 必须是右侧轴，直接返回 ax
            return ax
        else:
            # 否则，创建一个新的双轴对象
            orig_ax, new_ax = ax, ax.twinx()
            # TODO: 在 Matplotlib 的公共 API 可用时使用
            new_ax._get_lines = orig_ax._get_lines  # type: ignore[attr-defined]
            # TODO #54485
            new_ax._get_patches_for_fill = (  # type: ignore[attr-defined]
                orig_ax._get_patches_for_fill  # type: ignore[attr-defined]
            )
            # TODO #54485
            orig_ax.right_ax, new_ax.left_ax = (  # type: ignore[attr-defined]
                new_ax,
                orig_ax,
            )

            if not self._has_plotted_object(orig_ax):  # 如果左侧轴没有数据
                orig_ax.get_yaxis().set_visible(False)

            if self.logy is True or self.loglog is True:
                new_ax.set_yscale("log")
            elif self.logy == "sym" or self.loglog == "sym":
                new_ax.set_yscale("symlog")
            return new_ax  # type: ignore[return-value]

    @final
    @cache_readonly
    def fig(self) -> Figure:
        """返回缓存的图形对象"""
        return self._axes_and_fig[1]

    @final
    @cache_readonly
    # TODO: 能否将此注解同时标记为 Sequence[Axes] 和 ndarray[object]？
    def axes(self) -> Sequence[Axes]:
        """返回缓存的 Axes 序列"""
        return self._axes_and_fig[0]

    @final
    @cache_readonly
    def _axes_and_fig(self) -> tuple[Sequence[Axes], Figure]:
        import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图操作

        if self.subplots:
            naxes = (
                self.nseries if isinstance(self.subplots, bool) else len(self.subplots)
            )
            # 根据self.subplots的情况确定要创建的子图数量
            fig, axes = create_subplots(
                naxes=naxes,
                sharex=self.sharex,
                sharey=self.sharey,
                figsize=self.figsize,
                ax=self.ax,
                layout=self.layout,
                layout_type=self._layout_type,
            )
        elif self.ax is None:
            # 如果未提供轴对象，则创建一个新的图形对象，并添加一个子图
            fig = plt.figure(figsize=self.figsize)
            axes = fig.add_subplot(111)
        else:
            # 如果已提供轴对象，则使用其所属的图形对象，并根据需要设置图形大小
            fig = self.ax.get_figure()
            if self.figsize is not None:
                fig.set_size_inches(self.figsize)
            axes = self.ax

        axes = np.fromiter(flatten_axes(axes), dtype=object)  # 将axes展平为一维数组

        if self.logx is True or self.loglog is True:
            # 如果logx或loglog为True，则设置所有轴对象的x轴为对数刻度
            [a.set_xscale("log") for a in axes]
        elif self.logx == "sym" or self.loglog == "sym":
            # 如果logx或loglog为"sym"，则设置所有轴对象的x轴为对称对数刻度
            [a.set_xscale("symlog") for a in axes]

        if self.logy is True or self.loglog is True:
            # 如果logy或loglog为True，则设置所有轴对象的y轴为对数刻度
            [a.set_yscale("log") for a in axes]
        elif self.logy == "sym" or self.loglog == "sym":
            # 如果logy或loglog为"sym"，则设置所有轴对象的y轴为对称对数刻度
            [a.set_yscale("symlog") for a in axes]

        axes_seq = cast(Sequence["Axes"], axes)  # 将axes转换为Axes类型的序列
        return axes_seq, fig  # 返回轴对象序列和图形对象

    @property
    def result(self):
        """
        Return result axes
        """
        if self.subplots:
            if self.layout is not None and not is_list_like(self.ax):
                # 如果self.subplots为True且layout不为空且self.ax不是序列类型，返回根据layout调整后的轴对象
                return self.axes.reshape(*self.layout)  # type: ignore[attr-defined]
            else:
                # 否则直接返回轴对象
                return self.axes
        else:
            sec_true = isinstance(self.secondary_y, bool) and self.secondary_y
            # 判断是否所有的secondary_y数据都在次要轴上绘制
            all_sec = (
                is_list_like(self.secondary_y) and len(self.secondary_y) == self.nseries  # type: ignore[arg-type]
            )
            if sec_true or all_sec:
                # 如果所有数据都在次要轴上绘制，则返回右侧轴对象
                return self._get_ax_layer(self.axes[0], primary=False)
            else:
                # 否则返回主轴对象
                return self.axes[0]

    @final
    @staticmethod
    def _convert_to_ndarray(data):
        # GH31357: categorical columns are processed separately
        # 如果数据类型是 CategoricalDtype，则直接返回数据，不需要转换
        if isinstance(data.dtype, CategoricalDtype):
            return data

        # GH32073: cast to float if values contain nulled integers
        # 如果数据类型是 ExtensionDtype，并且包含空值整数，则转换为 float 类型的 numpy 数组
        if (is_integer_dtype(data.dtype) or is_float_dtype(data.dtype)) and isinstance(
            data.dtype, ExtensionDtype
        ):
            return data.to_numpy(dtype="float", na_value=np.nan)

        # GH25587: cast ExtensionArray of pandas (IntegerArray, etc.) to
        # np.ndarray before plot.
        # 将 pandas 的 ExtensionArray（如 IntegerArray 等）转换为 np.ndarray 以便绘图
        if len(data) > 0:
            return np.asarray(data)

        return data

    @final
    def _ensure_frame(self, data) -> DataFrame:
        # 如果数据是 ABCSeries 类型
        if isinstance(data, ABCSeries):
            label = self.label
            # 如果 label 为空并且数据的名称也为空，则将数据转换为 DataFrame
            if label is None and data.name is None:
                label = ""
            # 如果 label 为空，则使用数据的名称创建 DataFrame
            if label is None:
                data = data.to_frame()
            else:
                # 否则，使用 label 创建 DataFrame
                data = data.to_frame(name=label)
        elif self._kind in ("hist", "box"):
            # 如果图表类型是直方图或箱线图，则根据是否有分组标准选择列
            cols = self.columns if self.by is None else self.columns + self.by
            data = data.loc[:, cols]
        return data

    @final
    def _compute_plot_data(self) -> None:
        data = self.data

        # GH15079 reconstruct data if by is defined
        # 如果定义了分组标准 self.by，则重构数据
        if self.by is not None:
            self.subplots = True
            data = reconstruct_data_with_by(self.data, by=self.by, cols=self.columns)

        # GH16953, infer_objects is needed as fallback, for ``Series``
        # with ``dtype == object``
        # 使用 infer_objects 处理 dtype 是 object 的 Series，作为回退操作
        data = data.infer_objects()
        include_type = [np.number, "datetime", "datetimetz", "timedelta"]

        # GH23719, allow plotting boolean
        # 如果允许绘制布尔类型数据，则加入 np.bool_ 到包含类型列表
        if self.include_bool is True:
            include_type.append(np.bool_)

        # GH22799, exclude datetime-like type for boxplot
        # 对于箱线图，排除 datetime 类型的数据
        exclude_type = None
        if self._kind == "box":
            # TODO: change after solving issue 27881
            include_type = [np.number]
            exclude_type = ["timedelta"]

        # GH 18755, include object and category type for scatter plot
        # 对于散点图，包括 object 和 category 类型的数据
        if self._kind == "scatter":
            include_type.extend(["object", "category", "string"])

        # 根据包含和排除的数据类型选择数值类型的数据
        numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)

        is_empty = numeric_data.shape[-1] == 0
        # 检查是否没有数值型数据可绘制，如果是则引发 TypeError
        if is_empty:
            raise TypeError("no numeric data to plot")

        self.data = numeric_data.apply(type(self)._convert_to_ndarray)

    def _make_plot(self, fig: Figure) -> None:
        raise AbstractMethodError(self)

    @final
    def _add_table(self) -> None:
        # 如果禁用表格，则直接返回
        if self.table is False:
            return
        elif self.table is True:
            # 如果表格为 True，则转置数据
            data = self.data.transpose()
        else:
            # 否则，使用指定的表格数据
            data = self.table
        ax = self._get_ax(0)
        tools.table(ax, data)

    @final
    def _post_plot_logic_common(self, ax: Axes) -> None:
        """Common post process for each axes"""

        # 如果图表方向为垂直或未指定
        if self.orientation == "vertical" or self.orientation is None:
            # 对 x 轴应用特定的属性，如旋转和字体大小
            type(self)._apply_axis_properties(
                ax.xaxis, rot=self.rot, fontsize=self.fontsize
            )
            # 对 y 轴应用字体大小属性
            type(self)._apply_axis_properties(ax.yaxis, fontsize=self.fontsize)

            # 如果存在右侧轴，则对其 y 轴也应用字体大小属性
            if hasattr(ax, "right_ax"):
                type(self)._apply_axis_properties(
                    ax.right_ax.yaxis, fontsize=self.fontsize
                )

        # 如果图表方向为水平
        elif self.orientation == "horizontal":
            # 对 y 轴应用特定的属性，如旋转和字体大小
            type(self)._apply_axis_properties(
                ax.yaxis, rot=self.rot, fontsize=self.fontsize
            )
            # 对 x 轴应用字体大小属性
            type(self)._apply_axis_properties(ax.xaxis, fontsize=self.fontsize)

            # 如果存在右侧轴，则对其 y 轴也应用字体大小属性
            if hasattr(ax, "right_ax"):
                type(self)._apply_axis_properties(
                    ax.right_ax.yaxis, fontsize=self.fontsize
                )

        else:  # pragma no cover
            # 如果出现未知的图表方向，抛出值错误
            raise ValueError

    @abstractmethod
    def _post_plot_logic(self, ax: Axes, data) -> None:
        """Post process for each axes. Overridden in child classes"""

    @final
    @final
    @staticmethod
    def _apply_axis_properties(
        axis: Axis, rot=None, fontsize: int | None = None
    ):
        """
        Apply axis properties such as rotation and font size to the given axis.

        Parameters:
        - axis: Axis object to which properties are applied.
        - rot: Rotation angle for axis labels.
        - fontsize: Font size for axis labels.

        This method applies rotation and font size settings to the axis labels
        based on the provided parameters. It is a static method and does not
        depend on the instance state.
        """
    ) -> None:
        """
        Tick creation within matplotlib is reasonably expensive and is
        internally deferred until accessed as Ticks are created/destroyed
        multiple times per draw. It's therefore beneficial for us to avoid
        accessing unless we will act on the Tick.
        """
        # 如果旋转角度或字体大小不为空，则需要操作 Tick 标签
        if rot is not None or fontsize is not None:
            # 获取主刻度和次刻度标签
            labels = axis.get_majorticklabels() + axis.get_minorticklabels()
            # 遍历所有标签
            for label in labels:
                # 如果指定了旋转角度，则设置标签的旋转角度
                if rot is not None:
                    label.set_rotation(rot)
                # 如果指定了字体大小，则设置标签的字体大小
                if fontsize is not None:
                    label.set_fontsize(fontsize)

    @final
    @property
    def legend_title(self) -> str | None:
        # 如果数据列不是多级索引，则获取列名作为图例标题
        if not isinstance(self.data.columns, ABCMultiIndex):
            name = self.data.columns.name
            # 如果列名不为空，则美化输出并返回
            if name is not None:
                name = pprint_thing(name)
            return name
        else:
            # 如果是多级索引，则将每级索引名美化后用逗号连接并返回
            stringified = map(pprint_thing, self.data.columns.names)
            return ",".join(stringified)

    @final
    def _mark_right_label(self, label: str, index: int) -> str:
        """
        Append ``(right)`` to the label of a line if it's plotted on the right axis.

        Note that ``(right)`` is only appended when ``subplots=False``.
        """
        # 如果不是子图，并且标志为右轴显示，并且当前索引在右侧轴上，则添加“(right)”到标签末尾
        if not self.subplots and self.mark_right and self.on_right(index):
            label += " (right)"
        return label

    @final
    def _append_legend_handles_labels(self, handle: Artist, label: str) -> None:
        """
        Append current handle and label to ``legend_handles`` and ``legend_labels``.

        These will be used to make the legend.
        """
        # 将当前的图例句柄和标签添加到图例句柄列表和图例标签列表中
        self.legend_handles.append(handle)
        self.legend_labels.append(label)
    def _make_legend(self) -> None:
        # 获取第一个子图的坐标轴和图例对象
        ax, leg = self._get_ax_legend(self.axes[0])

        # 初始化图例的句柄和标签
        handles = []
        labels = []
        title = ""

        # 如果不是子图模式
        if not self.subplots:
            # 如果存在图例对象，获取其标题和句柄
            if leg is not None:
                title = leg.get_title().get_text()
                # 根据 Matplotlib 版本选择合适的方法获取图例句柄
                if Version(mpl.__version__) < Version("3.7"):
                    handles = leg.legendHandles
                else:
                    handles = leg.legend_handles
                # 获取图例中的文本标签
                labels = [x.get_text() for x in leg.get_texts()]

            # 如果需要添加用户定义的图例
            if self.legend:
                if self.legend == "reverse":
                    # 反转图例句柄和标签
                    handles += reversed(self.legend_handles)
                    labels += reversed(self.legend_labels)
                else:
                    # 添加默认顺序的图例句柄和标签
                    handles += self.legend_handles
                    labels += self.legend_labels

                # 如果设置了图例的标题，使用用户定义的标题
                if self.legend_title is not None:
                    title = self.legend_title

                # 如果存在有效的图例句柄，添加图例到坐标轴
                if len(handles) > 0:
                    ax.legend(handles, labels, loc="best", title=title)

        # 如果是子图模式且需要图例
        elif self.subplots and self.legend:
            # 遍历所有子图的坐标轴
            for ax in self.axes:
                # 如果子图可见，添加图例
                if ax.get_visible():
                    ax.legend(loc="best")

    @final
    @staticmethod
    def _get_ax_legend(ax: Axes):
        """
        Take in axes and return ax and legend under different scenarios
        """
        # 获取当前坐标轴的图例对象
        leg = ax.get_legend()

        # 获取关联的其他坐标轴（左或右）的图例对象
        other_ax = getattr(ax, "left_ax", None) or getattr(ax, "right_ax", None)
        other_leg = None
        if other_ax is not None:
            other_leg = other_ax.get_legend()
        # 如果当前坐标轴没有图例但关联的其他坐标轴有图例，则使用关联的图例
        if leg is None and other_leg is not None:
            leg = other_leg
            ax = other_ax
        return ax, leg

    _need_to_set_index = False

    @final
    def _get_xticks(self):
        # 获取数据的索引
        index = self.data.index
        # 检查索引类型是否为日期时间类型
        is_datetype = index.inferred_type in ("datetime", "date", "datetime64", "time")

        # TODO: be stricter about x?
        # 声明变量 x，可以是整数列表或 NumPy 数组
        x: list[int] | np.ndarray
        if self.use_index:
            if isinstance(index, ABCPeriodIndex):
                # 如果索引是周期性索引，转换为时间戳后获取其 Matplotlib 表示形式
                x = index.to_timestamp()._mpl_repr()
                # TODO: why do we need to do to_timestamp() here but not other
                #  places where we call mpl_repr?
                # 为什么这里需要调用 to_timestamp() 而其他调用 mpl_repr() 的地方不需要？
            elif is_any_real_numeric_dtype(index.dtype):
                # 如果索引是数值类型，则直接获取其 Matplotlib 表示形式
                # Matplotlib 支持数值或日期时间对象作为 x 轴值
                x = index._mpl_repr()
            elif isinstance(index, ABCDatetimeIndex) or is_datetype:
                # 如果索引是日期时间索引或推断为日期时间类型，则获取其 Matplotlib 表示形式
                x = index._mpl_repr()
            else:
                # 否则，需要设置索引，使用整数列表作为 x 轴值
                self._need_to_set_index = True
                x = list(range(len(index)))
        else:
            # 如果不使用索引，则使用整数列表作为 x 轴值
            x = list(range(len(index)))

        return x

    @classmethod
    @register_pandas_matplotlib_converters
    def _plot(
        cls, ax: Axes, x, y: np.ndarray, style=None, is_errorbar: bool = False, **kwds
    ):
        # 检查 y 是否有缺失值
        mask = isna(y)
        if mask.any():
            # 将 y 转换为带掩码的 NumPy 数组，遮盖掉缺失值
            y = np.ma.array(y)
            y = np.ma.masked_where(mask, y)

        if isinstance(x, ABCIndex):
            # 如果 x 是索引类型，则获取其 Matplotlib 表示形式
            x = x._mpl_repr()

        if is_errorbar:
            # 如果是误差条图，则处理 xerr 和 yerr
            if "xerr" in kwds:
                kwds["xerr"] = np.array(kwds.get("xerr"))
            if "yerr" in kwds:
                kwds["yerr"] = np.array(kwds.get("yerr"))
            return ax.errorbar(x, y, **kwds)
        else:
            # 如果不是误差条图，防止 style 参数传递给不支持的 errorbar 函数
            args = (x, y, style) if style is not None else (x, y)
            return ax.plot(*args, **kwds)

    def _get_custom_index_name(self):
        """Specify whether xlabel/ylabel should be used to override index name"""
        # 返回 xlabel 是否应该用于覆盖索引名称
        return self.xlabel

    @final
    def _get_index_name(self) -> str | None:
        # 获取索引名称，如果是多重索引则处理为逗号分隔的字符串
        if isinstance(self.data.index, ABCMultiIndex):
            name = self.data.index.names
            if com.any_not_none(*name):
                name = ",".join([pprint_thing(x) for x in name])
            else:
                name = None
        else:
            name = self.data.index.name
            if name is not None:
                name = pprint_thing(name)

        # GH 45145, override the default axis label if one is provided.
        # 如果提供了自定义的轴标签，则覆盖默认的轴标签
        index_name = self._get_custom_index_name()
        if index_name is not None:
            name = pprint_thing(index_name)

        return name

    @final
    @classmethod
    def _get_ax_layer(cls, ax, primary: bool = True):
        """获取左侧（主要）或右侧（次要）轴对象"""
        if primary:
            return getattr(ax, "left_ax", ax)
        else:
            return getattr(ax, "right_ax", ax)

    @final
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """返回列索引所对应的轴索引"""
        if isinstance(self.subplots, list):
            # 如果 subplots 是列表：多个列会在同一个轴上分组显示
            return next(
                group_idx
                for (group_idx, group) in enumerate(self.subplots)
                if col_idx in group
            )
        else:
            # 如果 subplots 是 True：每个列使用单独的轴
            return col_idx

    @final
    def _get_ax(self, i: int) -> Axes:
        # 如果需要，获取 twinx 轴对象
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            ax = self.axes[i]
            ax = self._maybe_right_yaxis(ax, i)
            # 错误：不支持对索引分配目标（"Sequence[Any]"）
            self.axes[i] = ax  # type: ignore[index]
        else:
            ax = self.axes[0]
            ax = self._maybe_right_yaxis(ax, i)

        ax.get_yaxis().set_visible(True)
        return ax

    @final
    def on_right(self, i: int) -> bool:
        if isinstance(self.secondary_y, bool):
            return self.secondary_y

        if isinstance(self.secondary_y, (tuple, list, np.ndarray, ABCIndex)):
            return self.data.columns[i] in self.secondary_y

    @final
    def _apply_style_colors(
        self, colors, kwds: dict[str, Any], col_num: int, label: str
    ):
        """
        根据列数和标签管理样式和颜色。
        返回适当的样式和关键字，其中可能会添加"color"。
        """
        style = None
        if self.style is not None:
            if isinstance(self.style, list):
                try:
                    style = self.style[col_num]
                except IndexError:
                    pass
            elif isinstance(self.style, dict):
                style = self.style.get(label, style)
            else:
                style = self.style

        has_color = "color" in kwds or self.colormap is not None
        nocolor_style = style is None or not _color_in_style(style)
        if (has_color or self.subplots) and nocolor_style:
            if isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[col_num % len(colors)]
        return style, kwds

    def _get_colors(
        self,
        num_colors: int | None = None,
        color_kwds: str = "color",
    ):
        """获取颜色"""
    # 如果未指定颜色数量，则使用数据集的系列数量作为默认值
    if num_colors is None:
        num_colors = self.nseries
    # 根据颜色关键字确定使用哪种颜色方案
    if color_kwds == "color":
        color = self.color
    else:
        color = self.kwds.get(color_kwds)
    # 调用 get_standard_colors 函数，获取标准颜色配置
    return get_standard_colors(
        num_colors=num_colors,
        colormap=self.colormap,
        color=color,
    )

# TODO: tighter typing for first return?
@final
@staticmethod
# 解析错误条（error bars）相关信息
def _parse_errorbars(
    label: str, err, data: NDFrameT, nseries: int
):
    errors = {}

    for kw, flag in zip(["xerr", "yerr"], [xerr, yerr]):
        if flag:
            err = self.errors[kw]
            # 如果用户提供了与标签匹配的错误数据框（DataFrame）或字典
            if isinstance(err, (ABCDataFrame, dict)):
                if label is not None and label in err.keys():
                    err = err[label]
                else:
                    err = None
            # 如果指定了索引，并且存在相应的错误条目
            elif index is not None and err is not None:
                err = err[index]

            if err is not None:
                errors[kw] = err
    return errors

@final
# 获取错误条（error bars）的相关信息
def _get_errorbars(
    self, label=None, index=None, xerr: bool = True, yerr: bool = True
) -> dict[str, Any]:
    errors = {}

    for kw, flag in zip(["xerr", "yerr"], [xerr, yerr]):
        if flag:
            err = self.errors[kw]
            # 如果用户提供了与标签匹配的错误数据框（DataFrame）或字典
            if isinstance(err, (ABCDataFrame, dict)):
                if label is not None and label in err.keys():
                    err = err[label]
                else:
                    err = None
            # 如果指定了索引，并且存在相应的错误条目
            elif index is not None and err is not None:
                err = err[index]

            if err is not None:
                errors[kw] = err
    return errors

@final
# 获取图表的子图列表
def _get_subplots(self, fig: Figure) -> list[Axes]:
    # 根据 Matplotlib 版本选择合适的子图类型类
    if Version(mpl.__version__) < Version("3.8"):
        Klass = mpl.axes.Subplot
    else:
        Klass = mpl.axes.Axes

    # 返回图表中所有符合条件的子图对象列表
    return [
        ax
        for ax in fig.get_axes()
        if (isinstance(ax, Klass) and ax.get_subplotspec() is not None)
    ]

@final
# 获取图表的布局信息，返回行数和列数
def _get_axes_layout(self, fig: Figure) -> tuple[int, int]:
    # 获取图表中所有子图对象
    axes = self._get_subplots(fig)
    x_set = set()
    y_set = set()
    for ax in axes:
        # 检查子图的坐标位置以估算布局
        points = ax.get_position().get_points()
        x_set.add(points[0][0])
        y_set.add(points[0][1])
    # 返回行数和列数的元组
    return (len(y_set), len(x_set))
class PlanePlot(MPLPlot, ABC):
    """
    Abstract class for plotting on plane, currently scatter and hexbin.
    """

    _layout_type = "single"

    def __init__(self, data, x, y, **kwargs) -> None:
        MPLPlot.__init__(self, data, **kwargs)
        # 检查是否提供了必要的 x 和 y 列
        if x is None or y is None:
            raise ValueError(self._kind + " requires an x and y column")
        # 如果 x 或 y 是整数索引，转换为对应的列名
        if is_integer(x) and not holds_integer(self.data.columns):
            x = self.data.columns[x]
        if is_integer(y) and not holds_integer(self.data.columns):
            y = self.data.columns[y]

        self.x = x
        self.y = y

    @final
    def _get_nseries(self, data: Series | DataFrame) -> int:
        # 返回数据中的系列数量（通常为1，单一系列）
        return 1

    @final
    def _post_plot_logic(self, ax: Axes, data) -> None:
        x, y = self.x, self.y
        # 如果未指定 xlabel 或 ylabel，则使用默认的列名或对象表示
        xlabel = self.xlabel if self.xlabel is not None else pprint_thing(x)
        ylabel = self.ylabel if self.ylabel is not None else pprint_thing(y)
        # 设置 x 轴和 y 轴的标签
        ax.set_xlabel(xlabel)  # type: ignore[arg-type]
        ax.set_ylabel(ylabel)  # type: ignore[arg-type]

    @final
    def _plot_colorbar(self, ax: Axes, *, fig: Figure, **kwds):
        # 解决问题 #10611 和 #10678：
        # 在 IPython 内联后端绘制散点图和六边形图时，
        # 颜色条轴的高度通常不完全匹配父轴的高度。
        # 这种差异是由浮点数表示中类似表示的小数部分差异造成的。
        # 为了处理这个问题，该方法强制颜色条的高度与父轴的高度相匹配。
        # 更详细的问题描述可参见以下链接：
        # https://github.com/ipython/ipython/issues/11215

        # GH33389，如果 ax 被多次使用，应始终使用包含最新信息的最后一个
        img = ax.collections[-1]
        # 返回添加到图中的颜色条对象
        return fig.colorbar(img, ax=ax, **kwds)


class ScatterPlot(PlanePlot):
    @property
    def _kind(self) -> Literal["scatter"]:
        # 返回绘图类型，此处为散点图
        return "scatter"

    def __init__(
        self,
        data,
        x,
        y,
        s=None,
        c=None,
        *,
        colorbar: bool | lib.NoDefault = lib.no_default,
        norm=None,
        **kwargs,
    ) -> None:
        if s is None:
            # 隐藏 matplotlib 默认的大小设置，以备将来可能更改参数处理方式
            s = 20
        elif is_hashable(s) and s in data.columns:
            s = data[s]
        self.s = s

        self.colorbar = colorbar
        self.norm = norm

        # 调用父类构造函数初始化基本参数
        super().__init__(data, x, y, **kwargs)
        # 如果 c 是整数索引，转换为对应的列名
        if is_integer(c) and not holds_integer(self.data.columns):
            c = self.data.columns[c]
        self.c = c
    # 定义一个方法 "_make_plot"，用于生成散点图
    def _make_plot(self, fig: Figure) -> None:
        # 从对象属性中获取 x, y, c, data
        x, y, c, data = self.x, self.y, self.c, self.data
        # 获取第一个坐标轴
        ax = self.axes[0]

        # 检查 c 是否为可哈希且存在于数据列中，确定是否按类别颜色分类
        c_is_column = is_hashable(c) and c in self.data.columns

        # 检查是否按照分类数据类型的类别颜色来设置颜色
        color_by_categorical = c_is_column and isinstance(
            self.data[c].dtype, CategoricalDtype
        )

        # 从对象属性中获取颜色设置
        color = self.color
        # 获取 c 对应的值
        c_values = self._get_c_values(color, color_by_categorical, c_is_column)
        # 获取归一化和颜色映射对象
        norm, cmap = self._get_norm_and_cmap(c_values, color_by_categorical)
        # 获取颜色条对象
        cb = self._get_colorbar(c_values, c_is_column)

        # 如果需要添加图例，则获取标签
        if self.legend:
            label = self.label
        else:
            label = None
        # 绘制散点图
        scatter = ax.scatter(
            data[x].values,
            data[y].values,
            c=c_values,
            label=label,
            cmap=cmap,
            norm=norm,
            s=self.s,
            **self.kwds,
        )
        
        # 如果需要绘制颜色条，则调用 _plot_colorbar 方法
        if cb:
            cbar_label = c if c_is_column else ""
            cbar = self._plot_colorbar(ax, fig=fig, label=cbar_label)
            # 如果按分类颜色，则设置颜色条刻度和标签
            if color_by_categorical:
                n_cats = len(self.data[c].cat.categories)
                cbar.set_ticks(np.linspace(0.5, n_cats - 0.5, n_cats))
                cbar.ax.set_yticklabels(self.data[c].cat.categories)

        # 如果需要添加图例项，则调用 _append_legend_handles_labels 方法
        if label is not None:
            self._append_legend_handles_labels(
                scatter,
                label,  # type: ignore[arg-type]  # 忽略类型检查的注释
            )

        # 获取 x 和 y 方向的误差条设置
        errors_x = self._get_errorbars(label=x, index=0, yerr=False)
        errors_y = self._get_errorbars(label=y, index=0, xerr=False)
        # 如果存在误差条设置，则绘制误差条
        if len(errors_x) > 0 or len(errors_y) > 0:
            err_kwds = dict(errors_x, **errors_y)
            err_kwds["ecolor"] = scatter.get_facecolor()[0]
            ax.errorbar(data[x].values, data[y].values, linestyle="none", **err_kwds)

    # 获取 c 对应的值，根据给定的 color, color_by_categorical 和 c_is_column 参数
    def _get_c_values(self, color, color_by_categorical: bool, c_is_column: bool):
        c = self.c
        # 如果 c 和 color 同时存在，则抛出类型错误
        if c is not None and color is not None:
            raise TypeError("Specify exactly one of `c` and `color`")
        # 如果 c 和 color 都为 None，则使用默认的面颜色
        if c is None and color is None:
            c_values = mpl.rcParams["patch.facecolor"]
        # 如果指定了 color 参数，则使用该参数作为颜色值
        elif color is not None:
            c_values = color
        # 如果按照分类数据类型的类别颜色设置，则获取对应的类别编码
        elif color_by_categorical:
            c_values = self.data[c].cat.codes
        # 如果 c 是一个列名，则直接使用该列的数值作为颜色值
        elif c_is_column:
            c_values = self.data[c].values
        # 否则，使用 c 自身作为颜色值
        else:
            c_values = c
        return c_values
    # 获取规范化对象和颜色映射对象，根据传入的参数确定颜色是否按类别分类
    def _get_norm_and_cmap(self, c_values, color_by_categorical: bool):
        # 获取当前实例的颜色属性
        c = self.c
        # 如果已指定颜色映射，则使用matplotlib获取对应的颜色映射对象
        if self.colormap is not None:
            cmap = mpl.colormaps.get_cmap(self.colormap)
        # 如果c_values不是字符串且是整数数据类型，则使用"Greys"颜色映射
        # 注意：需要额外调用isinstance()，因为is_integer_dtype在某些情况下会误判
        elif not isinstance(c_values, str) and is_integer_dtype(c_values):
            # pandas使用colormap，matplotlib使用cmap，这里做对应调整
            cmap = mpl.colormaps["Greys"]
        else:
            # 否则，不使用颜色映射
            cmap = None

        # 如果按类别分类且有有效的颜色映射对象，则创建基于类别的ListedColormap和BoundaryNorm
        if color_by_categorical and cmap is not None:
            # 获取数据集中c列的唯一类别数
            n_cats = len(self.data[c].cat.categories)
            # 创建基于颜色映射对象的ListedColormap
            cmap = mpl.colors.ListedColormap([cmap(i) for i in range(cmap.N)])
            # 生成分界值数组，用于规范化颜色映射
            bounds = np.linspace(0, n_cats, n_cats + 1)
            # 创建BoundaryNorm对象，用于规范化颜色映射
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            # TODO: 发出警告，如果用户指定了self.norm，我们目前正在忽略它
            #  这种情况在任何测试中并未发生 2023-11-09
        else:
            # 否则，使用已有的规范化对象self.norm
            norm = self.norm
        
        # 返回规范化对象和颜色映射对象
        return norm, cmap

    # 获取颜色条对象，根据传入的参数确定是否需要绘制颜色条
    def _get_colorbar(self, c_values, c_is_column: bool) -> bool:
        # 需要绘制颜色条的条件：
        # 1. 已分配颜色映射对象（self.colormap不为None），或者
        # 2. c是包含仅数值的列（c_is_column为True）
        plot_colorbar = self.colormap or c_is_column
        # 获取当前实例的颜色条属性
        cb = self.colorbar
        # 如果颜色条未指定（即使用默认设置），则检查c_values是否包含数值类型，以确定是否需要绘制颜色条
        if cb is lib.no_default:
            return is_numeric_dtype(c_values) and plot_colorbar
        # 否则，返回已指定的颜色条属性
        return cb
class`
class HexBinPlot(PlanePlot):
    # 定义 HexBinPlot 类，继承自 PlanePlot 类
    @property
    def _kind(self) -> Literal["hexbin"]:
        # 定义一个只读属性，返回字符串 "hexbin"
        return "hexbin"

    def __init__(self, data, x, y, C=None, *, colorbar: bool = True, **kwargs) -> None:
        # 初始化 HexBinPlot 对象，传入数据、x轴、y轴、颜色数据列、颜色条标志以及其他参数
        super().__init__(data, x, y, **kwargs)
        # 如果 C 是整数并且数据列不是整数类型，将 C 转换为数据列的列名
        if is_integer(C) and not holds_integer(self.data.columns):
            C = self.data.columns[C]
        # 设置颜色数据列
        self.C = C

        # 设置颜色条标志
        self.colorbar = colorbar

        # 检查 x 轴数据是否为数值型，散点图要求 x 轴数据为数值型
        if len(self.data[self.x]._get_numeric_data()) == 0:
            raise ValueError(self._kind + " requires x column to be numeric")
        # 检查 y 轴数据是否为数值型，散点图要求 y 轴数据为数值型
        if len(self.data[self.y]._get_numeric_data()) == 0:
            raise ValueError(self._kind + " requires y column to be numeric")

    def _make_plot(self, fig: Figure) -> None:
        # 创建图形绘制函数，传入图形对象 fig
        x, y, data, C = self.x, self.y, self.data, self.C
        ax = self.axes[0]
        # pandas 使用 colormap，matplotlib 使用 cmap
        cmap = self.colormap or "BuGn"
        cmap = mpl.colormaps.get_cmap(cmap)
        cb = self.colorbar

        # 如果没有设置颜色列，则颜色值为 None
        if C is None:
            c_values = None
        else:
            c_values = data[C].values

        # 使用 hexbin 方法在坐标轴 ax 上绘制数据
        ax.hexbin(data[x].values, data[y].values, C=c_values, cmap=cmap, **self.kwds)
        # 如果设置了颜色条，则绘制颜色条
        if cb:
            self._plot_colorbar(ax, fig=fig)

    def _make_legend(self) -> None:
        # 定义空的 _make_legend 方法，暂时不实现
        pass


class LinePlot(MPLPlot):
    _default_rot = 0

    @property
    def orientation(self) -> PlottingOrientation:
        # 返回图形的方向，默认为 "vertical"
        return "vertical"

    @property
    def _kind(self) -> Literal["line", "area", "hist", "kde", "box"]:
        # 定义一个只读属性，返回字符串 "line"
        return "line"

    def __init__(self, data, **kwargs) -> None:
        # 初始化 LinePlot 对象，传入数据和其他参数
        from pandas.plotting import plot_params

        MPLPlot.__init__(self, data, **kwargs)
        # 如果设置了堆叠，填充缺失值为 0
        if self.stacked:
            self.data = self.data.fillna(value=0)
        # 获取 x 兼容参数的默认值
        self.x_compat = plot_params["x_compat"]
        # 如果关键字参数中包含 "x_compat"，更新 x_compat 的值
        if "x_compat" in self.kwds:
            self.x_compat = bool(self.kwds.pop("x_compat"))

    @final
    def _is_ts_plot(self) -> bool:
        # 判断是否为时间序列图，返回布尔值
        # 这个方法略显误导，返回条件是 x_兼容参数未启用，使用索引，并且使用动态 x 轴
        return not self.x_compat and self.use_index and self._use_dynamic_x()

    @final
    def _use_dynamic_x(self) -> bool:
        # 判断是否使用动态 x 轴，返回布尔值
        # 调用 use_dynamic_x 函数，传入轴对象和数据
        return use_dynamic_x(self._get_ax(0), self.data)
    # 定义一个方法 _make_plot，接受一个 Figure 对象作为参数，并不返回任何结果
    def _make_plot(self, fig: Figure) -> None:
        # 如果当前绘制的图表是时间序列图
        if self._is_ts_plot():
            # 对数据进行索引转换
            data = maybe_convert_index(self._get_ax(0), self.data)

            # 获取 x 轴数据，这里只是一个虚拟变量，未被使用
            x = data.index  # dummy, not used

            # 选择时间序列绘图方法
            plotf = self._ts_plot

            # 迭代数据项
            it = data.items()
        else:
            # 获取 x 轴刻度
            x = self._get_xticks()

            # 错误：赋值类型不兼容（expression 类型为 "Callable[[Any, Any, Any, Any, Any, Any, KwArg(Any)], Any]"，
            # 变量类型为 "Callable[[Any, Any, Any, Any, KwArg(Any)], Any]"）
            plotf = self._plot  # type: ignore[assignment]

            # 错误：赋值类型不兼容（expression 类型为 "Iterator[tuple[Hashable, ndarray[Any, Any]]]"，
            # 变量类型为 "Iterable[tuple[Hashable, Series]]"）
            it = self._iter_data(data=self.data)  # type: ignore[assignment]

        # 获取堆叠标识符
        stacking_id = self._get_stacking_id()

        # 检查是否存在误差条
        is_errorbar = com.any_not_none(*self.errors.values())

        # 获取颜色配置
        colors = self._get_colors()

        # 遍历数据项
        for i, (label, y) in enumerate(it):
            # 获取当前子图对象
            ax = self._get_ax(i)

            # 复制关键字参数
            kwds = self.kwds.copy()

            # 如果指定了颜色，则设置颜色属性
            if self.color is not None:
                kwds["color"] = self.color

            # 应用样式和颜色
            style, kwds = self._apply_style_colors(
                colors,
                kwds,
                i,
                # 错误："_apply_style_colors" 的第四个参数类型不兼容 ("Hashable" 预期，"str" 实际)
                label,  # type: ignore[arg-type]
            )

            # 获取误差条
            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)

            # 格式化标签信息
            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)
            kwds["label"] = label

            # 调用绘图方法绘制图表
            newlines = plotf(
                ax,
                x,
                y,
                style=style,
                column_num=i,
                stacking_id=stacking_id,
                is_errorbar=is_errorbar,
                **kwds,
            )

            # 添加图例句柄和标签
            self._append_legend_handles_labels(newlines[0], label)

            # 如果是时间序列图，重设 x 轴限制
            if self._is_ts_plot():
                # 重设 x 轴的限制应该用于时间序列数据
                # TODO: GH28021, 应该找到一种改变 x 轴视图限制的方法
                lines = get_all_lines(ax)
                left, right = get_xlim(lines)
                ax.set_xlim(left, right)

    # 错误："_plot" 的签名与超类 "MPLPlot" 不兼容
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        style=None,
        column_num=None,
        stacking_id=None,
        **kwds,
        # column_num is used to get the target column from plotf in line and
        # area plots
        # 如果 column_num 为 0，则调用 _initialize_stacker 方法初始化堆叠信息，传入 ax、stacking_id 和 y 的长度
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        # 调用 _get_stacked_values 方法获取堆叠后的 y 值
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])
        # 调用 MPLPlot 类的 _plot 方法进行绘图，传入 ax、x、y_values 和 style 等参数
        lines = MPLPlot._plot(ax, x, y_values, style=style, **kwds)
        # 更新堆叠信息，传入 ax、stacking_id 和 y
        cls._update_stacker(ax, stacking_id, y)
        # 返回绘制的线条对象
        return lines

    @final
    def _ts_plot(self, ax: Axes, x, data: Series, style=None, **kwds):
        # accept x to be consistent with normal plot func,
        # x is not passed to tsplot as it uses data.index as x coordinate
        # column_num must be in kwds for stacking purpose
        # 对数据进行可能的重采样，返回重采样后的频率 freq 和数据 data
        freq, data = maybe_resample(data, ax, kwds)

        # 根据 freq 给 ax 添加装饰
        decorate_axes(ax, freq)
        # 如果 ax 有 left_ax 属性，则根据 freq 给 left_ax 添加装饰
        if hasattr(ax, "left_ax"):
            decorate_axes(ax.left_ax, freq)
        # 如果 ax 有 right_ax 属性，则根据 freq 给 right_ax 添加装饰
        if hasattr(ax, "right_ax"):
            decorate_axes(ax.right_ax, freq)
        # TODO #54485
        # 将数据、图表类型和 kwds 存入 ax 的 _plot_data 属性中
        ax._plot_data.append((data, self._kind, kwds))  # type: ignore[attr-defined]

        # 调用 _plot 方法绘制图表，传入 ax、data.index、data.values 和 style 等参数
        lines = self._plot(ax, data.index, np.asarray(data.values), style=style, **kwds)
        # 设置日期格式、定位器和重新调整限制
        # TODO #54485
        # 根据 ax 的 freq 和 data.index 设置日期轴格式
        format_dateaxis(ax, ax.freq, data.index)  # type: ignore[arg-type, attr-defined]
        # 返回绘制的线条对象
        return lines

    @final
    def _get_stacking_id(self) -> int | None:
        # 如果 stacked 为真，则返回数据对象 self.data 的 id 作为 stacking_id
        if self.stacked:
            return id(self.data)
        else:
            # 否则返回 None
            return None

    @final
    @classmethod
    def _initialize_stacker(cls, ax: Axes, stacking_id, n: int) -> None:
        # 如果 stacking_id 为 None，则直接返回
        if stacking_id is None:
            return
        # 如果 ax 没有 _stacker_pos_prior 属性，则为 ax 添加 _stacker_pos_prior 属性
        if not hasattr(ax, "_stacker_pos_prior"):
            # TODO #54485
            # 给 ax 添加 _stacker_pos_prior 属性，初始化为空字典
            ax._stacker_pos_prior = {}  # type: ignore[attr-defined]
        # 如果 ax 没有 _stacker_neg_prior 属性，则为 ax 添加 _stacker_neg_prior 属性
        if not hasattr(ax, "_stacker_neg_prior"):
            # TODO #54485
            # 给 ax 添加 _stacker_neg_prior 属性，初始化为空字典
            ax._stacker_neg_prior = {}  # type: ignore[attr-defined]
        # TODO #54485
        # 给 ax 的 _stacker_pos_prior[stacking_id] 赋值为长度为 n 的全零数组
        ax._stacker_pos_prior[stacking_id] = np.zeros(n)  # type: ignore[attr-defined]
        # TODO #54485
        # 给 ax 的 _stacker_neg_prior[stacking_id] 赋值为长度为 n 的全零数组
        ax._stacker_neg_prior[stacking_id] = np.zeros(n)  # type: ignore[attr-defined]

    @final
    @classmethod
    def _get_stacked_values(
        cls, ax: Axes, stacking_id: int | None, values: np.ndarray, label
    ):
        # 如果 stacking_id 为 None，则直接返回 values
        if stacking_id is None:
            return values
        # 否则根据 stacking_id 从 ax 的 _stacker_pos_prior 中获取正向堆叠的先前值
        pos_prior = ax._stacker_pos_prior.get(stacking_id, np.zeros(len(values)))  # type: ignore[attr-defined]
        # 根据 stacking_id 从 ax 的 _stacker_neg_prior 中获取负向堆叠的先前值
        neg_prior = ax._stacker_neg_prior.get(stacking_id, np.zeros(len(values)))  # type: ignore[attr-defined]
        # 将 values 加上正向和负向的堆叠先前值，得到堆叠后的值
        stacked_values = values + pos_prior - neg_prior
        # 如果 label 不为 None，则将 label 添加到正向堆叠的先前值的描述中
        if label is not None:
            ax._stacker_pos_prior[stacking_id] += values  # type: ignore[attr-defined]
        # 返回堆叠后的值
        return stacked_values
    ) -> np.ndarray:
        # 如果 stacking_id 为 None，则直接返回 values
        if stacking_id is None:
            return values
        # 如果 ax 没有属性 "_stacker_pos_prior"，则说明 stacker 可能未初始化用于子图
        if not hasattr(ax, "_stacker_pos_prior"):
            # 初始化 stacker 对象
            cls._initialize_stacker(ax, stacking_id, len(values))

        # 如果 values 中所有值都大于等于 0
        if (values >= 0).all():
            # TODO #54485
            # 返回 ax._stacker_pos_prior[stacking_id] 与 values 的和
            return (
                ax._stacker_pos_prior[stacking_id]  # type: ignore[attr-defined]
                + values
            )
        # 如果 values 中所有值都小于等于 0
        elif (values <= 0).all():
            # TODO #54485
            # 返回 ax._stacker_neg_prior[stacking_id] 与 values 的和
            return (
                ax._stacker_neg_prior[stacking_id]  # type: ignore[attr-defined]
                + values
            )

        # 如果 values 同时包含正数和负数，则抛出异常
        raise ValueError(
            "When stacked is True, each column must be either "
            "all positive or all negative. "
            f"Column '{label}' contains both positive and negative values"
        )

    @final
    @classmethod
    def _update_stacker(cls, ax: Axes, stacking_id: int | None, values) -> None:
        # 如果 stacking_id 为 None，则直接返回
        if stacking_id is None:
            return
        # 如果 values 中所有值都大于等于 0
        if (values >= 0).all():
            # TODO #54485
            # 将 values 加到 ax._stacker_pos_prior[stacking_id] 上
            ax._stacker_pos_prior[stacking_id] += values  # type: ignore[attr-defined]
        # 如果 values 中所有值都小于等于 0
        elif (values <= 0).all():
            # TODO #54485
            # 将 values 加到 ax._stacker_neg_prior[stacking_id] 上
            ax._stacker_neg_prior[stacking_id] += values  # type: ignore[attr-defined]

    def _post_plot_logic(self, ax: Axes, data) -> None:
        def get_label(i):
            # 如果 i 是浮点数且是整数，则转换为整数类型
            if is_float(i) and i.is_integer():
                i = int(i)
            try:
                # 返回 data.index[i] 的格式化字符串表示
                return pprint_thing(data.index[i])
            except Exception:
                # 若出错则返回空字符串
                return ""

        # 如果需要设置索引
        if self._need_to_set_index:
            # 获取当前 x 轴刻度位置和对应的标签
            xticks = ax.get_xticks()
            xticklabels = [get_label(x) for x in xticks]
            # 错误：FixedLocator 的第一个参数类型为 "ndarray[Any, Any]"，预期是 "Sequence[float]"
            # 设置 x 轴主要刻度为给定的位置
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks))  # type: ignore[arg-type]
            # 设置 x 轴刻度的标签
            ax.set_xticklabels(xticklabels)

        # 如果索引是不规则时间序列，并且需要使用索引
        condition = (
            not self._use_dynamic_x()
            and (data.index._is_all_dates and self.use_index)
            and (not self.subplots or (self.subplots and self.sharex))
        )

        # 获取索引名称
        index_name = self._get_index_name()

        # 如果满足条件
        if condition:
            # 如果旋转角度未设置，则设置为默认的 30 度
            if not self._rot_set:
                self.rot = 30
            # 格式化日期标签，并设置旋转角度为 self.rot
            format_date_labels(ax, rot=self.rot)

        # 如果存在索引名称且需要使用索引，则设置 x 轴的标签为索引名称
        if index_name is not None and self.use_index:
            ax.set_xlabel(index_name)
class AreaPlot(LinePlot):
    # 定义属性 _kind，表示图类型为面积图
    @property
    def _kind(self) -> Literal["area"]:
        return "area"

    # 初始化函数，接受数据和关键字参数
    def __init__(self, data, **kwargs) -> None:
        # 设置默认堆叠参数为 True
        kwargs.setdefault("stacked", True)
        # 将数据中的 NaN 值填充为 0
        data = data.fillna(value=0)
        # 调用父类 LinePlot 的初始化方法
        LinePlot.__init__(self, data, **kwargs)

        # 如果不是堆叠状态，设置透明度较小以区分重叠部分
        if not self.stacked:
            self.kwds.setdefault("alpha", 0.5)

        # 如果是对数坐标或对数-对数坐标，则抛出异常
        if self.logy or self.loglog:
            raise ValueError("Log-y scales are not supported in area plot")

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        style=None,
        column_num=None,
        stacking_id=None,
        is_errorbar: bool = False,
        **kwds,
    ):
        # 如果是第一列数据，初始化堆叠器
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        # 获取堆叠后的数值
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])

        # 需要移除标签，因为子图使用 mpl 的图例
        line_kwds = kwds.copy()
        line_kwds.pop("label")
        # 调用父类 MPLPlot 的 _plot 方法绘制线条
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)

        # 从线条中获取数据以获取 fill_between 的坐标
        xdata, y_values = lines[0].get_data(orig=False)

        # 无法在此处使用 _get_stacked_values 获取起始点
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            # TODO #54485
            start = ax._stacker_pos_prior[stacking_id]  # type: ignore[attr-defined]
        elif (y <= 0).all():
            # TODO #54485
            start = ax._stacker_neg_prior[stacking_id]  # type: ignore[attr-defined]
        else:
            start = np.zeros(len(y))

        # 如果关键字参数中没有颜色，则使用线条的颜色
        if "color" not in kwds:
            kwds["color"] = lines[0].get_color()

        # 使用 fill_between 方法填充区域
        rect = ax.fill_between(xdata, start, y_values, **kwds)
        # 更新堆叠器
        cls._update_stacker(ax, stacking_id, y)

        # LinePlot 预期返回艺术家对象列表
        res = [rect]
        return res

    # 执行绘图后的逻辑处理
    def _post_plot_logic(self, ax: Axes, data) -> None:
        # 调用父类 LinePlot 的 _post_plot_logic 方法
        LinePlot._post_plot_logic(self, ax, data)

        # 判断是否存在共享的 y 轴
        is_shared_y = len(list(ax.get_shared_y_axes())) > 0
        # 如果 ylim 未设置且没有共享 y 轴，则根据数据范围设置 y 轴范围
        if self.ylim is None and not is_shared_y:
            if (data >= 0).all().all():
                ax.set_ylim(0, None)
            elif (data <= 0).all().all():
                ax.set_ylim(None, 0)
    ) -> None:
        # 处理数据为 Series 或 DataFrame 的情况，以便正确处理颜色
        self._is_series = isinstance(data, ABCSeries)
        self.bar_width = width  # 设置条形图的宽度
        self._align = align  # 设置条形图的对齐方式
        self._position = position  # 设置条形图的位置
        self.tick_pos = np.arange(len(data))  # 生成刻度位置数组

        if is_list_like(bottom):
            bottom = np.array(bottom)  # 将底部位置列表转换为 NumPy 数组
        if is_list_like(left):
            left = np.array(left)  # 将左侧位置列表转换为 NumPy 数组
        self.bottom = bottom  # 设置条形图的底部位置
        self.left = left  # 设置条形图的左侧位置

        self.log = log  # 设置是否对数刻度

        MPLPlot.__init__(self, data, **kwargs)  # 调用父类 MPLPlot 的初始化方法

    @cache_readonly
    def ax_pos(self) -> np.ndarray:
        return self.tick_pos - self.tickoffset  # 返回刻度位置调整后的数组

    @cache_readonly
    def tickoffset(self):
        if self.stacked or self.subplots:
            return self.bar_width * self._position  # 处理堆叠或子图时的刻度偏移
        elif self._align == "edge":
            w = self.bar_width / self.nseries
            return self.bar_width * (self._position - 0.5) + w * 0.5  # 处理边缘对齐时的刻度偏移
        else:
            return self.bar_width * self._position  # 默认情况下的刻度偏移

    @cache_readonly
    def lim_offset(self):
        if self.stacked or self.subplots:
            if self._align == "edge":
                return self.bar_width / 2  # 处理堆叠或子图时的极限偏移
            else:
                return 0
        elif self._align == "edge":
            w = self.bar_width / self.nseries
            return w * 0.5  # 处理边缘对齐时的极限偏移
        else:
            return 0  # 默认情况下的极限偏移

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        w,
        start: int | npt.NDArray[np.intp] = 0,
        log: bool = False,
        **kwds,
    ):
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)  # 使用 Matplotlib 绘制条形图

    @property
    def _start_base(self):
        return self.bottom  # 返回条形图的底部位置数组
    def _make_plot(self, fig: Figure) -> None:
        # 获取颜色方案
        colors = self._get_colors()
        # 计算颜色数量
        ncolors = len(colors)

        # 初始化正向和负向优先级数组
        pos_prior = neg_prior = np.zeros(len(self.data))
        # 获取数据并填充缺失值
        K = self.nseries
        data = self.data.fillna(0)

        # 遍历数据并生成图表
        for i, (label, y) in enumerate(self._iter_data(data=data)):
            # 获取子图对象
            ax = self._get_ax(i)
            # 复制关键字参数
            kwds = self.kwds.copy()

            # 根据数据类型设置颜色
            if self._is_series:
                kwds["color"] = colors
            elif isinstance(colors, dict):
                kwds["color"] = colors[label]
            else:
                kwds["color"] = colors[i % ncolors]

            # 获取误差条
            errors = self._get_errorbars(label=label, index=i)
            kwds = dict(kwds, **errors)

            # 格式化标签
            label = pprint_thing(label)
            label = self._mark_right_label(label, index=i)

            # 如果存在误差条且未指定颜色，设置默认颜色
            if (("yerr" in kwds) or ("xerr" in kwds)) and (kwds.get("ecolor") is None):
                kwds["ecolor"] = mpl.rcParams["xtick.color"]

            # 初始化起始位置
            start = 0
            # 如果使用对数坐标且所有数据大于等于1，则起始位置设为1
            if self.log and (y >= 1).all():
                start = 1
            start = start + self._start_base

            # 设置对齐方式
            kwds["align"] = self._align

            # 如果使用子图，调整条形图宽度，并绘制
            if self.subplots:
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                ax.set_title(label)
            # 如果堆叠条形图，计算起始位置并绘制
            elif self.stacked:
                mask = y > 0
                start = np.where(mask, pos_prior, neg_prior) + self._start_base
                w = self.bar_width / 2
                rect = self._plot(
                    ax,
                    self.ax_pos + w,
                    y,
                    self.bar_width,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )
                pos_prior = pos_prior + np.where(mask, y, 0)
                neg_prior = neg_prior + np.where(mask, 0, y)
            # 否则，均匀分布条形图并绘制
            else:
                w = self.bar_width / K
                rect = self._plot(
                    ax,
                    self.ax_pos + (i + 0.5) * w,
                    y,
                    w,
                    start=start,
                    label=label,
                    log=self.log,
                    **kwds,
                )

            # 添加图例句柄和标签
            self._append_legend_handles_labels(rect, label)
    # 对绘图逻辑进行后处理，根据参数设置不同的索引字符串列表
    def _post_plot_logic(self, ax: Axes, data) -> None:
        # 如果使用索引，将索引键值转换成可打印的字符串形式
        if self.use_index:
            str_index = [pprint_thing(key) for key in data.index]
        else:
            # 如果不使用索引，生成从 0 到数据行数的索引字符串列表
            str_index = [pprint_thing(key) for key in range(data.shape[0])]

        # 计算起始和结束边缘的位置，考虑限制的偏移量和条形图宽度
        s_edge = self.ax_pos[0] - 0.25 + self.lim_offset
        e_edge = self.ax_pos[-1] + 0.25 + self.bar_width + self.lim_offset

        # 调用内部方法装饰坐标轴刻度
        self._decorate_ticks(ax, self._get_index_name(), str_index, s_edge, e_edge)

    # 内部方法：装饰坐标轴刻度
    def _decorate_ticks(
        self,
        ax: Axes,
        name: str | None,
        ticklabels: list[str],
        start_edge: float,
        end_edge: float,
    ) -> None:
        # 设置坐标轴的 x 范围
        ax.set_xlim((start_edge, end_edge))

        # 如果给定了自定义的 x 刻度位置，则使用它们
        if self.xticks is not None:
            ax.set_xticks(np.array(self.xticks))
        else:
            # 否则使用默认的刻度位置，并设置刻度标签为给定的 ticklabels
            ax.set_xticks(self.tick_pos)
            ax.set_xticklabels(ticklabels)

        # 如果指定了名称且使用索引，则设置 x 轴标签
        if name is not None and self.use_index:
            ax.set_xlabel(name)
class BarhPlot(BarPlot):
    @property
    def _kind(self) -> Literal["barh"]:
        # 返回图表类型为水平条形图
        return "barh"

    _default_rot = 0  # 默认旋转角度为0

    @property
    def orientation(self) -> Literal["horizontal"]:
        # 返回图表方向为水平
        return "horizontal"

    @property
    def _start_base(self):
        # 返回起始基线位置，从左侧开始
        return self.left

    # error: Signature of "_plot" incompatible with supertype "MPLPlot"
    @classmethod
    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        w,
        start: int | npt.NDArray[np.intp] = 0,
        log: bool = False,
        **kwds,
    ):
        # 绘制水平条形图
        return ax.barh(x, y, w, left=start, log=log, **kwds)

    def _get_custom_index_name(self):
        # 返回自定义的索引名称作为y轴标签
        return self.ylabel

    def _decorate_ticks(
        self,
        ax: Axes,
        name: str | None,
        ticklabels: list[str],
        start_edge: float,
        end_edge: float,
    ) -> None:
        # 设置y轴范围和刻度标签
        ax.set_ylim((start_edge, end_edge))
        ax.set_yticks(self.tick_pos)
        ax.set_yticklabels(ticklabels)
        if name is not None and self.use_index:
            ax.set_ylabel(name)
        # error: Argument 1 to "set_xlabel" of "_AxesBase" has incompatible type
        # "Hashable | None"; expected "str"
        ax.set_xlabel(self.xlabel)  # type: ignore[arg-type]


class PiePlot(MPLPlot):
    @property
    def _kind(self) -> Literal["pie"]:
        # 返回图表类型为饼图
        return "pie"

    _layout_type = "horizontal"  # 布局类型为水平

    def __init__(self, data, kind=None, **kwargs) -> None:
        # 确保数据没有空值，如果有负值则引发异常
        data = data.fillna(value=0)
        if (data < 0).any().any():
            raise ValueError(f"{self._kind} plot doesn't allow negative values")
        MPLPlot.__init__(self, data, kind=kind, **kwargs)

    @classmethod
    def _validate_log_kwd(
        cls,
        kwd: str,
        value: bool | None | Literal["sym"],
    ) -> bool | None | Literal["sym"]:
        # 验证日志关键字参数，忽略非False的值，并发出警告
        super()._validate_log_kwd(kwd=kwd, value=value)
        if value is not False:
            warnings.warn(
                f"PiePlot ignores the '{kwd}' keyword",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        return False

    def _validate_color_args(self, color, colormap) -> None:
        # TODO: 如果颜色参数被传递但被忽略，则发出警告
        return None
    # 根据传入的 Figure 对象生成图表
    def _make_plot(self, fig: Figure) -> None:
        # 获取用于绘图的颜色，如果未指定则使用默认颜色方案
        colors = self._get_colors(num_colors=len(self.data), color_kwds="colors")
        self.kwds.setdefault("colors", colors)

        # 遍历数据集中的每组数据
        for i, (label, y) in enumerate(self._iter_data(data=self.data)):
            # 获取第 i 个子图（Axes 对象）
            ax = self._get_ax(i)

            # 复制绘图参数以免修改原始参数
            kwds = self.kwds.copy()

            # 定义一个函数，用于处理标签，将值为 0 的标签替换为空字符串
            def blank_labeler(label, value):
                if value == 0:
                    return ""
                else:
                    return label

            # 获取数据索引，用作标签（labels）的备选值
            idx = [pprint_thing(v) for v in self.data.index]
            labels = kwds.pop("labels", idx)

            # 对每个楔形图的标签进行处理，将值为 0 的标签替换为空字符串，以防止重叠
            if labels is not None:
                blabels = [blank_labeler(left, value) for left, value in zip(labels, y)]
            else:
                blabels = None

            # 使用给定的数据 y 和处理后的标签 blabels 绘制饼图
            results = ax.pie(y, labels=blabels, **kwds)

            # 如果设置了 autopct 参数，解包结果（patches, texts, autotexts）
            if kwds.get("autopct", None) is not None:
                patches, texts, autotexts = results  # type: ignore[misc]
            else:
                # 否则，解包结果（patches, texts），autotexts 设为空列表
                patches, texts = results  # type: ignore[misc]
                autotexts = []

            # 如果设置了字体大小，将 texts 和 autotexts 中的文本都设置为相同的字体大小
            if self.fontsize is not None:
                for t in texts + autotexts:
                    t.set_fontsize(self.fontsize)

            # 设置图例标签（leglabels）使用的标签，若未指定则使用数据索引
            leglabels = labels if labels is not None else idx

            # 将 patches 和 leglabels 分别添加到图例中
            for _patch, _leglabel in zip(patches, leglabels):
                self._append_legend_handles_labels(_patch, _leglabel)

    # 该方法用于处理生成图表后的逻辑，目前没有实际操作，故 pass
    def _post_plot_logic(self, ax: Axes, data) -> None:
        pass
```