# `D:\src\scipysrc\seaborn\seaborn\_core\plot.py`

```
"""The classes for specifying and compiling a declarative visualization."""
# 引入未来版本支持的注解功能
from __future__ import annotations

# 导入需要的标准库和第三方库
import io
import os
import re
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator, Mapping
from typing import Any, List, Literal, Optional, cast
from xml.etree import ElementTree

# 导入 cycler 库，用于循环配色
from cycler import cycler

# 导入 pandas 库，并且从中导入 DataFrame, Series, Index 等类
import pandas as pd
from pandas import DataFrame, Series, Index

# 导入 matplotlib 库，并且从中导入 Axes, Artist, Figure 等类
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure

# 导入 numpy 库
import numpy as np

# 导入 PIL 库中的 Image 类
from PIL import Image

# 导入 seaborn 内部模块
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import (
    DataSource,
    VariableSpec,
    VariableSpecList,
    OrderSpec,
    Default,
)
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.rules import categorical_order
from seaborn._compat import get_layout_engine, set_layout_engine
from seaborn.utils import _version_predates
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette

# 导入类型检查相关的库和模块
from typing import TYPE_CHECKING, TypedDict

# 如果是类型检查环境，则从 matplotlib.figure 导入 SubFigure 类
if TYPE_CHECKING:
    from matplotlib.figure import SubFigure


# 定义默认值对象
default = Default()


# ---- Definitions for internal specs ---------------------------------------------- #


# 定义 Layer 类型，继承自 TypedDict，用于描述图层相关的规格
class Layer(TypedDict, total=False):

    mark: Mark  # TODO allow list?
    stat: Stat | None  # TODO allow list?
    move: Move | list[Move] | None
    data: PlotData
    source: DataSource
    vars: dict[str, VariableSpec]
    orient: str
    legend: bool
    label: str | None


# 定义 FacetSpec 类型，继承自 TypedDict，用于描述分面相关的规格
class FacetSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    wrap: int | None


# 定义 PairSpec 类型，继承自 TypedDict，用于描述成对图相关的规格
class PairSpec(TypedDict, total=False):

    variables: dict[str, VariableSpec]
    structure: dict[str, list[str]]
    cross: bool
    wrap: int | None


# --- Local helpers ---------------------------------------------------------------- #


# 定义 theme_context 上下文管理器，用于临时修改 matplotlib 的 rcParams
@contextmanager
def theme_context(params: dict[str, Any]) -> Generator:
    """Temporarily modify specific matplotlib rcParams."""
    # 备份原始的 rcParams 设置
    orig_params = {k: mpl.rcParams[k] for k in params}
    # 定义默认的颜色代码序列和美化后的颜色序列
    color_codes = "bgrmyck"
    nice_colors = [*color_palette("deep6"), (.15, .15, .15)]
    # 备份原始的颜色设置
    orig_colors = [mpl.colors.colorConverter.colors[x] for x in color_codes]
    # TODO how to allow this to reflect the color cycle when relevant?
    try:
        # 更新 matplotlib 的 rcParams
        mpl.rcParams.update(params)
        # 更新颜色转换器的颜色代码映射
        for (code, color) in zip(color_codes, nice_colors):
            mpl.colors.colorConverter.colors[code] = color
        # 使用 yield 返回上下文
        yield
    finally:
        # 恢复原始的 matplotlib 参数设置
        mpl.rcParams.update(orig_params)
        # 将颜色代码和原始颜色一一对应，更新颜色映射表
        for (code, color) in zip(color_codes, orig_colors):
            mpl.colors.colorConverter.colors[code] = color
def build_plot_signature(cls):
    """
    Decorator function for giving Plot a useful signature.

    Currently this mostly saves us some duplicated typing, but we would
    like eventually to have a way of registering new semantic properties,
    at which point dynamic signature generation would become more important.

    """
    # 获取传入类的签名信息
    sig = inspect.signature(cls)
    # 定义装饰器函数要添加的参数列表
    params = [
        inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
        inspect.Parameter("data", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    # 添加额外的关键字参数，这些参数来自于全局变量 PROPERTIES 中定义的名称列表
    params.extend([
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
        for name in PROPERTIES
    ])
    # 用新的参数列表替换原始类的签名
    new_sig = sig.replace(parameters=params)
    cls.__signature__ = new_sig

    # 生成已知属性的描述信息，用于格式化类的文档字符串
    known_properties = textwrap.fill(
        ", ".join([f"|{p}|" for p in PROPERTIES]),
        width=78, subsequent_indent=" " * 8,
    )

    if cls.__doc__ is not None:  # 支持 python -OO 模式
        # 格式化类的文档字符串，插入已知属性的描述信息
        cls.__doc__ = cls.__doc__.format(known_properties=known_properties)

    return cls


# ---- Plot configuration ---------------------------------------------------------- #


class ThemeConfig(mpl.RcParams):
    """
    Configuration object for the Plot.theme, using matplotlib rc parameters.
    """
    # 定义主题配置对象，继承自 mpl.RcParams
    THEME_GROUPS = [
        "axes", "figure", "font", "grid", "hatch", "legend", "lines",
        "mathtext", "markers", "patch", "savefig", "scatter",
        "xaxis", "xtick", "yaxis", "ytick",
    ]

    def __init__(self):
        super().__init__()
        # 初始化方法，调用父类的初始化方法，并重置主题配置
        self.reset()

    @property
    def _default(self) -> dict[str, Any]:
        # 返回默认的主题配置字典，包括默认的 matplotlib 参数、axes 样式、plotting 上下文和颜色循环设置
        return {
            **self._filter_params(mpl.rcParamsDefault),
            **axes_style("darkgrid"),
            **plotting_context("notebook"),
            "axes.prop_cycle": cycler("color", color_palette("deep")),
        }

    def reset(self) -> None:
        """Update the theme dictionary with seaborn's default values."""
        # 更新主题配置为 seaborn 的默认值
        self.update(self._default)

    def update(self, other: dict[str, Any] | None = None, /, **kwds):
        """Update the theme with a dictionary or keyword arguments of rc parameters."""
        # 更新主题配置，接受字典或关键字参数
        if other is not None:
            theme = self._filter_params(other)
        else:
            theme = {}
        theme.update(kwds)
        super().update(theme)

    def _filter_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Restruct to thematic rc params."""
        # 过滤主题相关的 rc 参数，保留符合主题组的参数
        return {
            k: v for k, v in params.items()
            if any(k.startswith(p) for p in self.THEME_GROUPS)
        }

    def _html_table(self, params: dict[str, Any]) -> list[str]:
        """Generate an HTML table representation of the parameters."""
        # 生成参数字典的 HTML 表格表示
        lines = ["<table>"]
        for k, v in params.items():
            row = f"<tr><td>{k}:</td><td style='text-align:left'>{v!r}</td></tr>"
            lines.append(row)
        lines.append("</table>")
        return lines
    def _repr_html_(self) -> str:
        """
        # 将对象转换为 HTML 表示形式的方法，返回一个字符串
        repr = [
            # 创建一个 div 容器，设置高度为 300px
            "<div style='height: 300px'>",
            # 在 div 内部创建一个带有边框的 div 容器
            "<div style='border-style: inset; border-width: 2px'>",
            # 调用对象的 _html_table 方法生成 HTML 表格，并将结果展开添加到列表中
            *self._html_table(self),
            # 关闭内部 div 容器
            "</div>",
            # 关闭外部 div 容器
            "</div>",
        ]
        # 将列表中的所有元素连接成一个字符串并返回
        return "\n".join(repr)
        """
class DisplayConfig(TypedDict):
    """Configuration for IPython's rich display hooks."""
    # 定义 DisplayConfig 类，用于配置 IPython 中富显示的钩子
    format: Literal["png", "svg"]
    # 图像格式，可选值为 "png" 或 "svg"
    scaling: float
    # 图像缩放比例
    hidpi: bool
    # 是否启用高 DPI


class PlotConfig:
    """Configuration for default behavior / appearance of class:`Plot` instances."""
    # 定义 PlotConfig 类，用于配置 Plot 实例的默认行为和外观

    def __init__(self):
        # 初始化方法，设置默认主题和显示配置
        self._theme = ThemeConfig()  # 创建 ThemeConfig 实例并赋给 _theme
        self._display = {"format": "png", "scaling": .85, "hidpi": True}
        # 设置默认的显示配置，包括图像格式、缩放比例和高 DPI 设置

    @property
    def theme(self) -> dict[str, Any]:
        """
        Dictionary of base theme parameters for :class:`Plot`.

        Keys and values correspond to matplotlib rc params, as documented here:
        https://matplotlib.org/stable/tutorials/introductory/customizing.html

        """
        # 返回基础主题参数的字典，用于 Plot 类

        return self._theme

    @property
    def display(self) -> DisplayConfig:
        """
        Dictionary of parameters for rich display in Jupyter notebook.

        Valid parameters:

        - format ("png" or "svg"): Image format to produce
        - scaling (float): Relative scaling of embedded image
        - hidpi (bool): When True, double the DPI while preserving the size

        """
        # 返回富显示在 Jupyter 笔记本中的参数字典，格式、缩放和高 DPI 设置

        return self._display


# ---- The main interface for declarative plotting --------------------------------- #


@build_plot_signature
# 使用装饰器 build_plot_signature 注册 Plot 类
class Plot:
    """
    An interface for declaratively specifying statistical graphics.

    Plots are constructed by initializing this class and adding one or more
    layers, comprising a `Mark` and optional `Stat` or `Move`.  Additionally,
    faceting variables or variable pairings may be defined to divide the space
    into multiple subplots. The mappings from data values to visual properties
    can be parametrized using scales, although the plot will try to infer good
    defaults when scales are not explicitly defined.

    The constructor accepts a data source (a :class:`pandas.DataFrame` or
    dictionary with columnar values) and variable assignments. Variables can be
    passed as keys to the data source or directly as data vectors.  If multiple
    data-containing objects are provided, they will be index-aligned.

    The data source and variables defined in the constructor will be used for
    all layers in the plot, unless overridden or disabled when adding a layer.

    The following variables can be defined in the constructor:
        {known_properties}

    The `data`, `x`, and `y` variables can be passed as positional arguments or
    using keywords. Whether the first positional argument is interpreted as a
    data source or `x` variable depends on its type.

    The methods of this class return a copy of the instance; use chaining to
    build up a plot through multiple calls. Methods can be called in any order.

    Most methods only add information to the plot spec; no actual processing
    happens until the plot is shown or saved. It is also possible to compile
    the plot without rendering it to access the lower-level representation.

    """
    # Plot 类，用于声明性地指定统计图形的接口

    config = PlotConfig()
    # 设置 Plot 类的配置属性为 PlotConfig 实例

    _data: PlotData
    # 定义 Plot 类的私有属性 _data，类型为 PlotData
    # 表示图形的图层列表
    _layers: list[Layer]

    # 存储比例尺的字典，键为名称，值为对应的比例尺对象
    _scales: dict[str, Scale]
    
    # 存储是否共享的状态或名称的字典，值可以是布尔值或字符串
    _shares: dict[str, bool | str]
    
    # 存储各种属性的取值范围的字典，值为元组，包含最小值和最大值
    _limits: dict[str, tuple[Any, Any]]
    
    # 存储标签信息的字典，值可以是字符串或一个将字符串映射为字符串的函数
    _labels: dict[str, str | Callable[[str], str]]
    
    # 存储主题相关的属性的字典，可以包含任意类型的值
    _theme: dict[str, Any]

    # 存储子图规格的对象
    _facet_spec: FacetSpec
    
    # 存储成对规格的对象
    _pair_spec: PairSpec

    # 存储整体图形规格的字典，包含各种图形的设置属性
    _figure_spec: dict[str, Any]
    
    # 存储子图设置的字典，包含子图的各种属性设置
    _subplot_spec: dict[str, Any]
    
    # 存储布局设置的字典，包含整体布局的各种属性设置
    _layout_spec: dict[str, Any]

    def __init__(
        self,
        *args: DataSource | VariableSpec,
        data: DataSource = None,
        **variables: VariableSpec,
    ):
        """
        初始化方法，接受位置参数和关键字参数作为数据源和变量规格。

        如果有位置参数，根据参数解析数据和变量。
        检查是否有未知的关键字参数，如果有则抛出类型错误。
        初始化 _data 属性为 PlotData 类的实例，用给定的数据和变量字典。
        初始化各个属性为初始值。
        """

        if args:
            data, variables = self._resolve_positionals(args, data, variables)

        unknown = [x for x in variables if x not in PROPERTIES]
        if unknown:
            err = f"Plot() got unexpected keyword argument(s): {', '.join(unknown)}"
            raise TypeError(err)

        self._data = PlotData(data, variables)

        self._layers = []

        self._scales = {}
        self._shares = {}
        self._limits = {}
        self._labels = {}
        self._theme = {}

        self._facet_spec = {}
        self._pair_spec = {}

        self._figure_spec = {}
        self._subplot_spec = {}
        self._layout_spec = {}

        self._target = None

    def _resolve_positionals(
        self,
        args: tuple[DataSource | VariableSpec, ...],
        data: DataSource,
        variables: dict[str, VariableSpec],
    ) -> tuple[DataSource, dict[str, VariableSpec]]:
        """
        处理位置参数，可能包含数据、x、y等信息。

        检查位置参数的数量是否符合要求，不超过三个。
        根据参数类型检查数据源是否有效，如果数据源被重复指定则抛出类型错误。
        根据位置参数的数量和类型设置 x 和 y 变量。
        将变量与对应的坐标保持一致，将其前置在变量字典中。
        返回处理后的数据源和变量字典。
        """

        if len(args) > 3:
            err = "Plot() accepts no more than 3 positional arguments (data, x, y)."
            raise TypeError(err)

        if (
            isinstance(args[0], (abc.Mapping, pd.DataFrame))
            or hasattr(args[0], "__dataframe__")
        ):
            if data is not None:
                raise TypeError("`data` given by both name and position.")
            data, args = args[0], args[1:]

        if len(args) == 2:
            x, y = args
        elif len(args) == 1:
            x, y = *args, None
        else:
            x = y = None

        for name, var in zip("yx", (y, x)):
            if var is not None:
                if name in variables:
                    raise TypeError(f"`{name}` given by both name and position.")
                # Keep coordinates at the front of the variables dict
                # Cast type because we know this isn't a DataSource at this point
                variables = {name: cast(VariableSpec, var), **variables}

        return data, variables

    def __add__(self, other):
        """
        重载加法运算符，处理 Plot 对象和其他类型的对象相加的情况。

        如果其他对象是 Mark 或 Stat 类型，则抛出类型错误。
        否则抛出类型错误，指明不支持 Plot 对象与其他类型的对象相加。
        """

        if isinstance(other, Mark) or isinstance(other, Stat):
            raise TypeError("Sorry, this isn't ggplot! Perhaps try Plot.add?")

        other_type = other.__class__.__name__
        raise TypeError(f"Unsupported operand type(s) for +: 'Plot' and '{other_type}")
    # 返回表示对象为 PNG 图片的元组 (bytes, dict[str, float]) 或 None
    def _repr_png_(self) -> tuple[bytes, dict[str, float]] | None:
        # 如果配置中的显示格式不是 PNG，则返回 None
        if Plot.config.display["format"] != "png":
            return None
        # 调用 plot() 方法并返回其 _repr_png_() 方法的结果
        return self.plot()._repr_png_()

    # 返回表示对象为 SVG 图片的字符串或 None
    def _repr_svg_(self) -> str | None:
        # 如果配置中的显示格式不是 SVG，则返回 None
        if Plot.config.display["format"] != "svg":
            return None
        # 调用 plot() 方法并返回其 _repr_svg_() 方法的结果
        return self.plot()._repr_svg_()

    # 生成一个新的 Plot 对象，包含与当前对象相同的信息
    def _clone(self) -> Plot:
        new = Plot()

        # 将当前对象的数据复制到新对象中，不允许数据被修改的方式封装
        new._data = self._data

        # 将当前对象的图层列表扩展到新对象的图层列表中
        new._layers.extend(self._layers)

        # 更新新对象的比例尺、共享配置、限制、标签和主题
        new._scales.update(self._scales)
        new._shares.update(self._shares)
        new._limits.update(self._limits)
        new._labels.update(self._labels)
        new._theme.update(self._theme)

        # 更新新对象的分面规格和配对规格
        new._facet_spec.update(self._facet_spec)
        new._pair_spec.update(self._pair_spec)

        # 更新新对象的图形规格、子图规格和布局规格
        new._figure_spec.update(self._figure_spec)
        new._subplot_spec.update(self._subplot_spec)
        new._layout_spec.update(self._layout_spec)

        # 复制当前对象的目标属性到新对象
        new._target = self._target

        return new

    # 返回当前对象的主题与默认主题合并后的字典
    def _theme_with_defaults(self) -> dict[str, Any]:
        theme = self.config.theme.copy()
        theme.update(self._theme)
        return theme

    # 返回当前对象涉及的所有变量的列表
    @property
    def _variables(self) -> list[str]:
        # 从数据框、配对规格和分面规格中获取所有变量的列表
        variables = (
            list(self._data.frame)
            + list(self._pair_spec.get("variables", []))
            + list(self._facet_spec.get("variables", []))
        )
        # 遍历图层列表，将不在变量列表中的变量添加进去
        for layer in self._layers:
            variables.extend(v for v in layer["vars"] if v not in variables)

        # 强制转换为字符串列表，以满足类型检查要求
        # 这些变量只会是字符串，但是目前的类型系统不支持严格类型化的 DataFrame
        return [str(v) for v in variables]
    def on(self, target: Axes | SubFigure | Figure) -> Plot:
        """
        Provide existing Matplotlib figure or axes for drawing the plot.

        When using this method, you will also need to explicitly call a method that
        triggers compilation, such as :meth:`Plot.show` or :meth:`Plot.save`. If you
        want to postprocess using matplotlib, you'd need to call :meth:`Plot.plot`
        first to compile the plot without rendering it.

        Parameters
        ----------
        target : Axes, SubFigure, or Figure
            Matplotlib object to use. Passing :class:`matplotlib.axes.Axes` will add
            artists without otherwise modifying the figure. Otherwise, subplots will be
            created within the space of the given :class:`matplotlib.figure.Figure` or
            :class:`matplotlib.figure.SubFigure`.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.on.rst

        """
        accepted_types: tuple  # Allow tuple of various length
        accepted_types = (
            mpl.axes.Axes, mpl.figure.SubFigure, mpl.figure.Figure
        )
        accepted_types_str = (
            f"{mpl.axes.Axes}, {mpl.figure.SubFigure}, or {mpl.figure.Figure}"
        )

        if not isinstance(target, accepted_types):
            err = (
                f"The `Plot.on` target must be an instance of {accepted_types_str}. "
                f"You passed an instance of {target.__class__} instead."
            )
            raise TypeError(err)

        # Clone the current Plot instance
        new = self._clone()
        # Set the target matplotlib object for the plot
        new._target = target

        # Return the new Plot instance with the updated target
        return new

    def add(
        self,
        mark: Mark,
        *transforms: Stat | Move,
        orient: str | None = None,
        legend: bool = True,
        label: str | None = None,
        data: DataSource = None,
        **variables: VariableSpec,
    ):
        """
        Add a mark to the plot.

        Parameters
        ----------
        mark : Mark
            The graphical mark to add to the plot.
        *transforms : Stat or Move
            Optional statistical transforms or positional adjustments to apply to the mark.
        orient : str or None, optional
            Orientation of the mark (e.g., 'horizontal' or 'vertical').
        legend : bool, optional
            Whether to include the mark in the plot legend.
        label : str or None, optional
            Label for the mark in the legend.
        data : DataSource, optional
            Data source for the mark.
        **variables : VariableSpec
            Additional variables to map to aesthetics of the mark.

        """

    def pair(
        self,
        x: VariableSpecList = None,
        y: VariableSpecList = None,
        wrap: int | None = None,
        cross: bool = True,
    ) -> Plot:
        """
        Produce subplots by pairing multiple `x` and/or `y` variables.

        Parameters
        ----------
        x, y : sequence(s) of data vectors or identifiers
            Variables that will define the grid of subplots.
        wrap : int
            When using only `x` or `y`, "wrap" subplots across a two-dimensional grid
            with this many columns (when using `x`) or rows (when using `y`).
        cross : bool
            When False, zip the `x` and `y` lists such that the first subplot gets the
            first pair, the second gets the second pair, etc. Otherwise, create a
            two-dimensional grid from the cartesian product of the lists.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.pair.rst

        """
        # TODO Add transpose= arg, which would then draw pair(y=[...]) across rows
        # This may also be possible by setting `wrap=1`, but is that too unobvious?
        # TODO PairGrid features not currently implemented: diagonals, corner

        # Initialize an empty dictionary to store pairing specifications
        pair_spec: PairSpec = {}

        # Initialize dictionaries for x and y axes
        axes = {"x": [] if x is None else x, "y": [] if y is None else y}

        # Validate that x and y are sequences of data vectors or identifiers
        for axis, arg in axes.items():
            if isinstance(arg, (str, int)):
                err = f"You must pass a sequence of variable keys to `{axis}`"
                raise TypeError(err)

        # Initialize nested dictionaries in pair_spec for variables and structure
        pair_spec["variables"] = {}
        pair_spec["structure"] = {}

        # Populate pair_spec with variables and their corresponding keys
        for axis in "xy":
            keys = []
            for i, col in enumerate(axes[axis]):
                key = f"{axis}{i}"
                keys.append(key)
                pair_spec["variables"][key] = col

            if keys:
                pair_spec["structure"][axis] = keys

        # Ensure that x and y lists have matching lengths if cross=False
        if not cross and len(axes["x"]) != len(axes["y"]):
            err = "Lengths of the `x` and `y` lists must match with cross=False"
            raise ValueError(err)

        # Store cross and wrap settings in pair_spec
        pair_spec["cross"] = cross
        pair_spec["wrap"] = wrap

        # Create a new instance of the Plot object with updated pair_spec
        new = self._clone()
        new._pair_spec.update(pair_spec)
        return new
    ) -> Plot:
        """
        Produce subplots with conditional subsets of the data.

        Parameters
        ----------
        col, row : data vectors or identifiers
            Variables used to define subsets along the columns and/or rows of the grid.
            Can be references to the global data source passed in the constructor.
        order : list of strings, or dict with dimensional keys
            Define the order of the faceting variables.
        wrap : int
            When using only `col` or `row`, wrap subplots across a two-dimensional
            grid with this many subplots on the faceting dimension.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.facet.rst

        """
        # 定义一个方法，用于生成带有条件数据子集的子图。

        variables: dict[str, VariableSpec] = {}
        # 初始化一个空的字典，用于存储变量及其规格

        if col is not None:
            variables["col"] = col
            # 如果 col 不为空，则将其添加到变量字典中的 "col" 键下

        if row is not None:
            variables["row"] = row
            # 如果 row 不为空，则将其添加到变量字典中的 "row" 键下

        structure = {}
        # 初始化一个空的结构字典，用于存储 faceting 结构信息

        if isinstance(order, dict):
            # 如果 order 是字典类型
            for dim in ["col", "row"]:
                dim_order = order.get(dim)
                # 获取 order 中维度为 dim 的排序信息
                if dim_order is not None:
                    structure[dim] = list(dim_order)
                    # 如果排序信息不为空，则将其作为列表存储到结构字典中的 dim 键下
        elif order is not None:
            # 如果 order 不是字典且不为空
            if col is not None and row is not None:
                err = " ".join([
                    "When faceting on both col= and row=, passing `order` as a list"
                    "is ambiguous. Use a dict with 'col' and/or 'row' keys instead."
                ])
                raise RuntimeError(err)
                # 如果同时在 col 和 row 上进行 faceting，将 order 作为列表传递是不明确的，应该使用包含 'col' 和/或 'row' 键的字典
            elif col is not None:
                structure["col"] = list(order)
                # 如果只在 col 上进行 faceting，则将 order 转换为列表存储到结构字典中的 "col" 键下
            elif row is not None:
                structure["row"] = list(order)
                # 如果只在 row 上进行 faceting，则将 order 转换为列表存储到结构字典中的 "row" 键下

        spec: FacetSpec = {
            "variables": variables,
            "structure": structure,
            "wrap": wrap,
        }
        # 创建一个 FacetSpec 类型的规格字典，包含变量、结构和 wrap 信息

        new = self._clone()
        # 克隆当前对象，创建一个新的对象实例

        new._facet_spec.update(spec)
        # 更新新对象的 faceting 规格信息

        return new
        # 返回更新后的新对象
    def scale(self, **scales: Scale) -> Plot:
        """
        Specify mappings from data units to visual properties.

        Keywords correspond to variables defined in the plot, including coordinate
        variables (`x`, `y`) and semantic variables (`color`, `pointsize`, etc.).

        A number of "magic" arguments are accepted, including:
            - The name of a transform (e.g., `"log"`, `"sqrt"`)
            - The name of a palette (e.g., `"viridis"`, `"muted"`)
            - A tuple of values, defining the output range (e.g. `(1, 5)`)
            - A dict, implying a :class:`Nominal` scale (e.g. `{"a": .2, "b": .5}`)
            - A list of values, implying a :class:`Nominal` scale (e.g. `["b", "r"]`)

        For more explicit control, pass a scale spec object such as :class:`Continuous`
        or :class:`Nominal`. Or pass `None` to use an "identity" scale, which treats
        data values as literally encoding visual properties.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.scale.rst

        """
        # 创建当前对象的副本
        new = self._clone()
        # 更新副本对象的比例尺信息
        new._scales.update(scales)
        # 返回更新后的副本对象
        return new

    def share(self, **shares: bool | str) -> Plot:
        """
        Control sharing of axis limits and ticks across subplots.

        Keywords correspond to variables defined in the plot, and values can be
        boolean (to share across all subplots), or one of "row" or "col" (to share
        more selectively across one dimension of a grid).

        Behavior for non-coordinate variables is currently undefined.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.share.rst

        """
        # 创建当前对象的副本
        new = self._clone()
        # 更新副本对象的共享设置
        new._shares.update(shares)
        # 返回更新后的副本对象
        return new

    def limit(self, **limits: tuple[Any, Any]) -> Plot:
        """
        Control the range of visible data.

        Keywords correspond to variables defined in the plot, and values are a
        `(min, max)` tuple (where either can be `None` to leave unset).

        Limits apply only to the axis; data outside the visible range are
        still used for any stat transforms and added to the plot.

        Behavior for non-coordinate variables is currently undefined.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.limit.rst

        """
        # 创建当前对象的副本
        new = self._clone()
        # 更新副本对象的限制范围
        new._limits.update(limits)
        # 返回更新后的副本对象
        return new

    def label(
        self, *,
        title: str | None = None,
        legend: str | None = None,
        **variables: str | Callable[[str], str]
    ) -> Plot:
        """
        Specify titles and legends for the plot.

        Parameters
        ----------
        title : str or None, optional
            The title of the plot.
        legend : str or None, optional
            The legend title.

        **variables : str or callable
            Additional variables to set labels dynamically.

        Returns
        -------
        Plot
            A new plot object with updated labels.

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.label.rst

        """
        # 创建当前对象的副本
        new = self._clone()
        # 如果给定了标题，则更新副本对象的标题
        if title is not None:
            new._title = title
        # 如果给定了图例标题，则更新副本对象的图例标题
        if legend is not None:
            new._legend = legend
        # 更新副本对象的其他动态标签变量
        new._variables.update(variables)
        # 返回更新后的副本对象
        return new
    ) -> Plot:
        """
        Control the labels and titles for axes, legends, and subplots.

        Additional keywords correspond to variables defined in the plot.
        Values can be one of the following types:

        - string (used literally; pass "" to clear the default label)
        - function (called on the default label)

        For coordinate variables, the value sets the axis label.
        For semantic variables, the value sets the legend title.
        For faceting variables, `title=` modifies the subplot-specific label,
        while `col=` and/or `row=` add a label for the faceting variable.

        When using a single subplot, `title=` sets its title.

        The `legend=` parameter sets the title for the "layer" legend
        (i.e., when using `label` in :meth:`Plot.add`).

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.label.rst


        """
        # 创建当前对象的副本，用于修改标签和标题，而不影响原始对象
        new = self._clone()
        # 如果传入了 `title` 参数，则更新副本的标题标签
        if title is not None:
            new._labels["title"] = title
        # 如果传入了 `legend` 参数，则更新副本的图例标题标签
        if legend is not None:
            new._labels["legend"] = legend
        # 更新副本的其他变量标签，根据传入的 `variables` 字典
        new._labels.update(variables)
        # 返回更新后的副本对象
        return new

    def layout(
        self,
        *,
        size: tuple[float, float] | Default = default,
        engine: str | None | Default = default,
        extent: tuple[float, float, float, float] | Default = default,
    ) -> Plot:
        """
        控制图形的大小和布局。

        .. note::

            默认的图形大小和指定图形大小的API可能在未来的“实验性”版本中发生变化。
            默认的布局引擎也可能会改变。

        Parameters
        ----------
        size : (width, height)
            结果图形的大小，单位为英寸。包括 pyplot 使用时的图例，但其他情况不包括。
        engine : {"tight", "constrained", "none"}
            用于自动调整布局以消除重叠的方法名称。默认取决于是否使用了 :meth:`Plot.on`。
        extent : (left, bottom, right, top)
            绘图布局的边界，以图形大小的分数表示。通过布局引擎生效；不同的引擎会有不同的确切结果。
            注意：当使用布局引擎时，extent 包括轴装饰，但在 `engine="none"` 时不包括。

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.layout.rst

        """
        # TODO add an "auto" mode for figsize that roughly scales with the rcParams
        # figsize (so that works), but expands to prevent subplots from being squished
        # Also should we have height=, aspect=, exclusive with figsize? Or working
        # with figsize when only one is defined?

        new = self._clone()

        if size is not default:
            # 如果指定了 size 参数，则更新新对象的图形规格
            new._figure_spec["figsize"] = size
        if engine is not default:
            # 如果指定了 engine 参数，则更新新对象的布局规格中的引擎类型
            new._layout_spec["engine"] = engine
        if extent is not default:
            # 如果指定了 extent 参数，则更新新对象的布局规格中的范围
            new._layout_spec["extent"] = extent

        return new

    # TODO def legend (ugh)

    def theme(self, config: Mapping[str, Any], /) -> Plot:
        """
        控制绘图元素的外观。

        .. note::

            自定义绘图外观的API尚未最终确定。
            目前，唯一有效的参数是 matplotlib rc 参数的字典。
            （这个字典必须作为位置参数传递。）

            未来版本可能会增强此方法。

        Matplotlib rc 参数在以下页面有详细文档：
        https://matplotlib.org/stable/tutorials/introductory/customizing.html

        Examples
        --------
        .. include:: ../docstrings/objects.Plot.theme.rst

        """
        new = self._clone()

        rc = mpl.RcParams(config)
        # 更新新对象的主题设置，使用给定的 rc 参数
        new._theme.update(rc)

        return new
    # 将绘图编译并保存到缓冲区或磁盘文件中

    def save(self, loc, **kwargs) -> Plot:
        """
        Compile the plot and write it to a buffer or file on disk.

        Parameters
        ----------
        loc : str, path, or buffer
            Location on disk to save the figure, or a buffer to write into.
        kwargs
            Other keyword arguments are passed through to
            :meth:`matplotlib.figure.Figure.savefig`.

        """
        # TODO expose important keyword arguments in our signature?
        # 在主题上下文中使用带有默认设置的主题
        with theme_context(self._theme_with_defaults()):
            # 编译绘图并保存到指定位置，传递其他关键字参数
            self._plot().save(loc, **kwargs)
        # 返回当前 Plot 对象实例
        return self

    # 显示绘图，利用 pyplot 显示

    def show(self, **kwargs) -> None:
        """
        Compile the plot and display it by hooking into pyplot.

        Calling this method is not necessary to render a plot in notebook context,
        but it may be in other environments (e.g., in a terminal). After compiling the
        plot, it calls :func:`matplotlib.pyplot.show` (passing any keyword parameters).

        Unlike other :class:`Plot` methods, there is no return value. This should be
        the last method you call when specifying a plot.

        """
        # TODO make pyplot configurable at the class level, and when not using,
        # import IPython.display and call on self to populate cell output?

        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024

        # 获取绘图对象并调用其 show 方法显示绘图
        self.plot(pyplot=True).show(**kwargs)

    # 编译绘图规格并返回 Plotter 对象

    def plot(self, pyplot: bool = False) -> Plotter:
        """
        Compile the plot spec and return the Plotter object.
        """
        # 在主题上下文中使用带有默认设置的主题
        with theme_context(self._theme_with_defaults()):
            # 返回编译后的绘图对象
            return self._plot(pyplot)
    # 定义一个方法 `_plot`，用于生成图表对象 `Plotter`
    def _plot(self, pyplot: bool = False) -> Plotter:

        # TODO 如果存在 `_target` 对象，如何检查它是否连接到 pyplot 状态机来确定 `pyplot` 参数的值？
        
        # 创建一个 Plotter 对象，根据传入的 pyplot 参数和当前主题创建
        plotter = Plotter(pyplot=pyplot, theme=self._theme_with_defaults())

        # 处理变量赋值并初始化图表
        # 从当前对象中提取共同数据和图层数据
        common, layers = plotter._extract_data(self)
        plotter._setup_figure(self, common, layers)

        # 处理坐标变量的比例规格，并转换它们的数据
        coord_vars = [v for v in self._variables if re.match(r"^x|y", v)]
        plotter._setup_scales(self, common, layers, coord_vars)

        # 应用统计变换
        plotter._compute_stats(self, layers)

        # 处理语义变量和由统计计算得到的坐标的比例规格
        plotter._setup_scales(self, common, layers)

        # TODO 更新其他方法后移除这些代码
        # ---- 可能有一个 debug= 参数，当设置为 True 时，附加这些代码？
        # 将 common 数据和 layers 数据分配给 plotter 对象的属性 _data 和 _layers
        plotter._data = common
        plotter._layers = layers

        # 处理每个图层的数据并添加 matplotlib 的图形元素
        for layer in layers:
            plotter._plot_layer(self, layer)

        # 添加各种图表装饰
        plotter._make_legend(self)
        plotter._finalize_figure(self)

        # 返回生成的 Plotter 对象
        return plotter
# ---- The plot compilation engine ---------------------------------------------- #

# 定义 Plotter 类，用于将 Plot 规范编译成 Matplotlib 图形的引擎
class Plotter:
    """
    Engine for compiling a :class:`Plot` spec into a Matplotlib figure.

    This class is not intended to be instantiated directly by users.

    """
    # TODO decide if we ever want these (Plot.plot(debug=True))?
    
    # 类属性定义
    _data: PlotData  # 存储绘图数据的对象，类型为 PlotData
    _layers: list[Layer]  # 存储图层的列表，每个图层类型为 Layer
    _figure: Figure  # 存储 Matplotlib 图形对象的引用，类型为 Figure

    # 初始化方法，接受 pyplot 参数和主题参数
    def __init__(self, pyplot: bool, theme: dict[str, Any]):
        # 标志是否使用 pyplot
        self._pyplot = pyplot
        # 存储主题参数的字典
        self._theme = theme
        # 存储图例内容的列表，每个元素是一个元组，包含 (图例标题, 艺术家对象列表, 标签列表)
        self._legend_contents: list[tuple[
            tuple[str, str | int], list[Artist], list[str],
        ]] = []
        # 存储比例尺对象的字典，键为比例尺名称，值为 Scale 对象
        self._scales: dict[str, Scale] = {}

    # save 方法，用于保存绘图到指定位置
    def save(self, loc, **kwargs) -> Plotter:  # TODO type args
        # 设置默认参数 dpi 为 96
        kwargs.setdefault("dpi", 96)
        try:
            # 尝试扩展用户目录中的位置
            loc = os.path.expanduser(loc)
        except TypeError:
            # 如果 loc 是一个缓冲区，则忽略这个错误
            pass
        # 调用 Figure 对象的 savefig 方法保存图形到 loc 地址
        self._figure.savefig(loc, **kwargs)
        return self

    # show 方法，用于显示绘图结果
    def show(self, **kwargs) -> None:
        """
        Display the plot by hooking into pyplot.

        This method calls :func:`matplotlib.pyplot.show` with any keyword parameters.

        """
        # TODO if we did not create the Plotter with pyplot, is it possible to do this?
        # If not we should clearly raise.
        
        # 导入 matplotlib.pyplot 模块，并在指定主题下显示图形
        import matplotlib.pyplot as plt
        with theme_context(self._theme):
            plt.show(**kwargs)

    # TODO API for accessing the underlying matplotlib objects
    # TODO what else is useful in the public API for this class?

    # _repr_png_ 方法，用于返回图形的 PNG 数据和元数据
    def _repr_png_(self) -> tuple[bytes, dict[str, float]] | None:

        # TODO use matplotlib backend directly instead of going through savefig?

        # TODO perhaps have self.show() flip a switch to disable this, so that
        # user does not end up with two versions of the figure in the output

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        # Better solution would be to default (with option to change)
        # to using constrained/tight layout.

        # 如果配置中的显示格式不是 PNG，则返回 None
        if Plot.config.display["format"] != "png":
            return None

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()

        # 根据配置决定 DPI 和缩放因子
        factor = 2 if Plot.config.display["hidpi"] else 1
        scaling = Plot.config.display["scaling"] / factor
        dpi = 96 * factor  # TODO put dpi in Plot.config?

        # 在指定主题下保存图形到缓冲区，使用 PNG 格式和紧凑边界框
        with theme_context(self._theme):  # TODO _theme_with_defaults?
            self._figure.savefig(buffer, dpi=dpi, format="png", bbox_inches="tight")
        data = buffer.getvalue()

        # 获取图像的宽度和高度，并根据缩放因子调整元数据
        w, h = Image.open(buffer).size
        metadata = {"width": w * scaling, "height": h * scaling}
        return data, metadata
    def _repr_svg_(self) -> str | None:
        # 检查配置中的显示格式是否为 SVG，如果不是则返回 None
        if Plot.config.display["format"] != "svg":
            return None

        # 获取显示比例配置
        scaling = Plot.config.display["scaling"]

        # 创建一个字符串缓冲区
        buffer = io.StringIO()
        # 使用当前主题上下文保存的主题进行绘图
        with theme_context(self._theme):  # TODO _theme_with_defaults?
            # 将图形保存到缓冲区中，格式为 SVG，边界框调整为紧凑模式
            self._figure.savefig(buffer, format="svg", bbox_inches="tight")

        # 从缓冲区获取 XML 根元素
        root = ElementTree.fromstring(buffer.getvalue())
        # 计算缩放后的宽度和高度，并更新 XML 根元素的相关属性
        w = scaling * float(root.attrib["width"][:-2])
        h = scaling * float(root.attrib["height"][:-2])
        root.attrib.update(width=f"{w}pt", height=f"{h}pt", viewbox=f"0 0 {w} {h}")
        # 将更新后的 XML 根元素写入到字节流中
        ElementTree.ElementTree(root).write(out := io.BytesIO())

        # 返回字节流内容的解码结果作为 SVG 字符串
        return out.getvalue().decode()

    def _extract_data(self, p: Plot) -> tuple[PlotData, list[Layer]]:
        # 合并所有数据、面板和配对规格的共同数据
        common_data = (
            p._data
            .join(None, p._facet_spec.get("variables"))
            .join(None, p._pair_spec.get("variables"))
        )

        # 初始化图层列表
        layers: list[Layer] = []
        # 遍历所有图层
        for layer in p._layers:
            # 复制图层规格
            spec = layer.copy()
            # 将共同数据与图层的源和变量连接起来，更新到图层规格中
            spec["data"] = common_data.join(layer.get("source"), layer.get("vars"))
            # 将更新后的图层规格添加到图层列表中
            layers.append(spec)

        # 返回共同数据和图层列表作为元组
        return common_data, layers

    def _resolve_label(self, p: Plot, var: str, auto_label: str | None) -> str:
        # 如果变量符合格式 '[xy]\d+'，则使用变量本身或其首字母作为键
        if re.match(r"[xy]\d+", var):
            key = var if var in p._labels else var[0]
        else:
            key = var

        # 初始化标签为字符串类型
        label: str
        # 如果键存在于标签映射中
        if key in p._labels:
            # 获取手动设置的标签
            manual_label = p._labels[key]
            # 如果手动标签是可调用对象且自动标签不为 None，则使用自动标签生成标签
            if callable(manual_label) and auto_label is not None:
                label = manual_label(auto_label)
            else:
                # 否则直接使用手动设置的标签
                label = cast(str, manual_label)
        elif auto_label is None:
            # 如果自动标签为 None，则标签为空字符串
            label = ""
        else:
            # 否则使用自动标签作为标签
            label = auto_label

        # 返回解析后的标签
        return label
    # 计算统计信息并更新图表规范对象
    def _compute_stats(self, spec: Plot, layers: list[Layer]) -> None:
        # 获取所有除了 "xy" 以外的属性名称作为分组变量
        grouping_vars = [v for v in PROPERTIES if v not in "xy"]
        # 添加额外的分组变量到列表中
        grouping_vars += ["col", "row", "group"]

        # 从图表规范对象中获取配对变量的结构信息
        pair_vars = spec._pair_spec.get("structure", {})

        # 遍历图层列表
        for layer in layers:
            # 提取图层的数据、标记和统计方法
            data = layer["data"]
            mark = layer["mark"]
            stat = layer["stat"]

            # 如果统计方法为空，则跳过当前图层
            if stat is None:
                continue

            # 根据配对变量生成坐标轴的迭代器
            iter_axes = itertools.product(*[
                pair_vars.get(axis, [axis]) for axis in "xy"
            ])

            # 保存原始数据帧的引用
            old = data.frame

            # 如果存在配对变量，则清空数据帧的数据
            if pair_vars:
                data.frames = {}
                data.frame = data.frame.iloc[:0]  # TODO to simplify typing

            # 遍历坐标轴的组合
            for coord_vars in iter_axes:
                # 构建变量的配对组合
                pairings = "xy", coord_vars

                # 复制原始数据帧
                df = old.copy()
                # 复制比例尺
                scales = self._scales.copy()

                # 根据配对组合重命名列名，并删除多余的列
                for axis, var in zip(*pairings):
                    if axis != var:
                        df = df.rename(columns={var: axis})
                        drop_cols = [x for x in df if re.match(rf"{axis}\d+", str(x))]
                        df = df.drop(drop_cols, axis=1)
                        scales[axis] = scales[var]

                # 推断图层的方向性
                orient = layer["orient"] or mark._infer_orient(scales)

                # 如果需要按方向分组，则将方向性添加到分组器中
                if stat.group_by_orient:
                    grouper = [orient, *grouping_vars]
                else:
                    grouper = grouping_vars
                # 创建分组对象
                groupby = GroupBy(grouper)
                # 应用统计方法并得到结果
                res = stat(df, groupby, orient, scales)

                # 如果存在配对变量，则将结果存储到数据帧中
                if pair_vars:
                    data.frames[coord_vars] = res
                else:
                    data.frame = res

    # 获取比例尺对象
    def _get_scale(
        self, p: Plot, var: str, prop: Property, values: Series
    ) -> Scale:
        # 根据变量名称匹配比例尺对象的键
        if re.match(r"[xy]\d+", var):
            key = var if var in p._scales else var[0]
        else:
            key = var

        # 如果比例尺对象中存在对应键，则返回其值，否则返回默认比例尺对象
        if key in p._scales:
            arg = p._scales[key]
            if arg is None or isinstance(arg, Scale):
                scale = arg
            else:
                scale = prop.infer_scale(arg, values)
        else:
            scale = prop.default_scale(values)

        # 返回比例尺对象
        return scale
    # 根据参数设置来获取子图的数据
    def _get_subplot_data(self, df, var, view, share_state):

        if share_state in [True, "all"]:
            # 全部共享的情况下，每个子图可以看到所有数据
            seed_values = df[var]
        else:
            # 否则，需要为不同的子图设置独立的坐标轴
            if share_state in [False, "none"]:
                # 完全独立的轴，每个子图使用自己的数据
                idx = self._get_subplot_index(df, view)
            elif share_state in df:
                # 在行或列内部共享较为复杂
                use_rows = df[share_state] == view[share_state]
                idx = df.index[use_rows]
            else:
                # 这种配置可能不太合理，但也可以处理
                idx = df.index

            seed_values = df.loc[idx, var]

        return seed_values

    # 设置绘图的比例尺
    def _setup_scales(
        self,
        p: Plot,
        common: PlotData,
        layers: list[Layer],
        variables: list[str] | None = None,
    ):
        # TODO do we still have numbers in the variable name at this point?
        # 查找数据框中包含坐标的列，用于反向缩放
        coord_cols = [c for c in df if re.match(r"^[xy]\D*$", str(c))]
        # 创建一个新的数据框，删除坐标相关的列，并保持其它列的位置不变
        out_df = (
            df
            .drop(coord_cols, axis=1)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
            .copy(deep=False)
        )

        for view in subplots:
            # 过滤出当前子图的数据
            view_df = self._filter_subplot_data(df, view)
            # 提取子图中的坐标列数据
            axes_df = view_df[coord_cols]
            for var, values in axes_df.items():
                # 获取当前坐标轴对象，进行反向变换
                axis = getattr(view["ax"], f"{str(var)[0]}axis")
                # TODO see https://github.com/matplotlib/matplotlib/issues/22713
                transform = axis.get_transform().inverted().transform
                inverted = transform(values)
                out_df.loc[values.index, str(var)] = inverted

        return out_df

    # 生成数据配对
    def _generate_pairings(
        self, data: PlotData, pair_variables: dict,
    ) -> Generator[
        tuple[list[dict], DataFrame, dict[str, Scale]], None, None
    ]:
    # TODO retype return with subplot_spec or similar
    iter_axes = itertools.product(*[
        pair_variables.get(axis, [axis]) for axis in "xy"
    ])

    # 遍历每个可能的坐标轴组合，根据给定的变量对列表生成迭代器
    for x, y in iter_axes:
        # 筛选与当前坐标轴组合匹配的子图
        subplots = []
        for view in self._subplots:
            if (view["x"] == x) and (view["y"] == y):
                subplots.append(view)

        # 根据条件选择数据框中的子集，并进行拷贝
        if data.frame.empty and data.frames:
            out_df = data.frames[(x, y)].copy()
        elif not pair_variables:
            out_df = data.frame.copy()
        else:
            if data.frame.empty and data.frames:
                out_df = data.frames[(x, y)].copy()
            else:
                out_df = data.frame.copy()

        # 复制尺度设置
        scales = self._scales.copy()
        # 如果 x 在数据框中存在，则更新 x 轴的尺度
        if x in out_df:
            scales["x"] = self._scales[x]
        # 如果 y 在数据框中存在，则更新 y 轴的尺度
        if y in out_df:
            scales["y"] = self._scales[y]

        # 根据坐标轴和变量重命名数据框列名，并删除不需要的列
        for axis, var in zip("xy", (x, y)):
            if axis != var:
                out_df = out_df.rename(columns={var: axis})
                cols = [col for col in out_df if re.match(rf"{axis}\d+", str(col))]
                out_df = out_df.drop(cols, axis=1)

        # 返回当前坐标轴组合对应的子图列表、处理后的数据框及尺度设置
        yield subplots, out_df, scales


def _get_subplot_index(self, df: DataFrame, subplot: dict) -> Index:
    # 获取子图在数据框中的索引，仅限包含特定列名的数据框
    dims = df.columns.intersection(["col", "row"])
    if dims.empty:
        return df.index

    # 筛选出与给定子图匹配的数据行索引
    keep_rows = pd.Series(True, df.index, dtype=bool)
    for dim in dims:
        keep_rows &= df[dim] == subplot[dim]
    return df.index[keep_rows]


def _filter_subplot_data(self, df: DataFrame, subplot: dict) -> DataFrame:
    # TODO note redundancies with preceding function ... needs refactoring
    # 筛选出与给定子图匹配的数据行，并返回筛选后的数据框
    dims = df.columns.intersection(["col", "row"])
    if dims.empty:
        return df

    keep_rows = pd.Series(True, df.index, dtype=bool)
    for dim in dims:
        keep_rows &= df[dim] == subplot[dim]
    return df[keep_rows]


def _setup_split_generator(
    self, grouping_vars: list[str], df: DataFrame, subplots: list[dict[str, Any]],
):
    # 设置用于生成拆分数据的生成器，根据分组变量、数据框和子图列表
    ...


def _update_legend_contents(
    self,
    p: Plot,
    mark: Mark,
    data: PlotData,
    scales: dict[str, Scale],
    layer_label: str | None,
):
    # 更新图例内容，接受绘图对象、标记、绘图数据、尺度和图层标签作为参数
    ...
    ) -> None:
        """为绘图中的每一层添加图例艺术家或标签。"""
        # 如果 data.frame 是空的而 data.frames 不为空，则初始化一个空列表 legend_vars
        if data.frame.empty and data.frames:
            legend_vars: list[str] = []
            # 遍历 data.frames 中的每一个 frame
            for frame in data.frames.values():
                # 获取 frame 中与 scales 中列表的交集，并将结果扩展到 legend_vars 中
                frame_vars = frame.columns.intersection(list(scales))
                legend_vars.extend(v for v in frame_vars if v not in legend_vars)
        else:
            # 否则，将 data.frame 中与 scales 中列表的交集赋给 legend_vars
            legend_vars = list(data.frame.columns.intersection(list(scales)))

        # 处理层次图例，它们在 legend_contents 中占据单个条目。
        if layer_label is not None:
            # 获取图层的标签
            legend_title = str(p._labels.get("legend", ""))
            # 构建图层键
            layer_key = (legend_title, -1)
            # 创建图例艺术家
            artist = mark._legend_artist([], None, {})
            # 如果 artist 不为空
            if artist is not None:
                # 遍历 self._legend_contents 中的内容
                for content in self._legend_contents:
                    # 如果找到匹配的图层键
                    if content[0] == layer_key:
                        # 添加艺术家和标签到内容中
                        content[1].append(artist)
                        content[2].append(layer_label)
                        break
                else:
                    # 如果未找到匹配的图层键，则添加新的内容条目
                    self._legend_contents.append((layer_key, [artist], [layer_label]))

        # 处理比例尺图例
        # 第一步：识别每个变量将显示的值
        schema: list[tuple[
            tuple[str, str | int], list[str], tuple[list[Any], list[str]]
        ]] = []
        schema = []
        for var in legend_vars:
            # 获取变量对应的图例信息
            var_legend = scales[var]._legend
            if var_legend is not None:
                values, labels = var_legend
                for (_, part_id), part_vars, _ in schema:
                    # 如果 data.ids[var] 与 part_id 匹配，则将变量添加到 part_vars 中
                    if data.ids[var] == part_id:
                        part_vars.append(var)
                        break
                else:
                    # 否则创建新的条目，并添加到 schema 中
                    title = self._resolve_label(p, var, data.names[var])
                    entry = (title, data.ids[var]), [var], (values, labels)
                    schema.append(entry)

        # 第二步：为每个值生成相应的艺术家
        contents: list[tuple[tuple[str, str | int], Any, list[str]]] = []
        for key, variables, (values, labels) in schema:
            artists = []
            for val in values:
                # 根据变量和值创建图例艺术家
                artist = mark._legend_artist(variables, val, scales)
                if artist is not None:
                    artists.append(artist)
            if artists:
                # 添加内容到 legend_contents 中
                contents.append((key, artists, labels))

        self._legend_contents.extend(contents)
    def _make_legend(self, p: Plot) -> None:
        """Create the legend artist(s) and add onto the figure."""
        # Combine artists representing same information across layers
        # Input list has an entry for each distinct variable in each layer
        # Output dict has an entry for each distinct variable
        merged_contents: dict[
            tuple[str, str | int], tuple[list[tuple[Artist, ...]], list[str]],
        ] = {}
        for key, new_artists, labels in self._legend_contents:
            # Key is (name, id); we need the id to resolve variable uniqueness,
            # but will need the name in the next step to title the legend
            if key not in merged_contents:
                # Matplotlib accepts a tuple of artists and will overlay them
                new_artist_tuples = [tuple([a]) for a in new_artists]
                merged_contents[key] = new_artist_tuples, labels
            else:
                existing_artists = merged_contents[key][0]
                for i, new_artist in enumerate(new_artists):
                    existing_artists[i] += tuple([new_artist])

        # When using pyplot, an "external" legend won't be shown, so this
        # keeps it inside the axes (though still attached to the figure)
        # This is necessary because matplotlib layout engines currently don't
        # support figure legends — ideally this will change.
        loc = "center right" if self._pyplot else "center left"

        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():

            legend = mpl.legend.Legend(
                self._figure,
                handles,  # type: ignore  # matplotlib/issues/26639
                labels,
                title=name,  # Title the legend with the name
                loc=loc,  # Set the location of the legend
                bbox_to_anchor=(.98, .55),  # Adjust the legend box position
            )

            if base_legend:
                # Matplotlib has no public API for this so it is a bit of a hack.
                # Ideally we'd define our own legend class with more flexibility,
                # but that is a lot of work!
                base_legend_box = base_legend.get_children()[0]
                this_legend_box = legend.get_children()[0]
                base_legend_box.get_children().extend(this_legend_box.get_children())
            else:
                base_legend = legend
                self._figure.legends.append(legend)
    # 对每个子图进行最终处理
    def _finalize_figure(self, p: Plot) -> None:
        # 遍历所有子图
        for sub in self._subplots:
            ax = sub["ax"]
            # 遍历处理x轴和y轴
            for axis in "xy":
                axis_key = sub[axis]
                axis_obj = getattr(ax, f"{axis}axis")

                # 处理坐标轴的限制
                if axis_key in p._limits or axis in p._limits:
                    convert_units = getattr(ax, f"{axis}axis").convert_units
                    a, b = p._limits.get(axis_key) or p._limits[axis]
                    lo = a if a is None else convert_units(a)
                    hi = b if b is None else convert_units(b)
                    if isinstance(a, str):
                        lo = cast(float, lo) - 0.5
                    if isinstance(b, str):
                        hi = cast(float, hi) + 0.5
                    ax.set(**{f"{axis}lim": (lo, hi)})

                # 如果存在轴的缩放设置
                if axis_key in self._scales:  # TODO when would it not be?
                    self._scales[axis_key]._finalize(p, axis_obj)

        # 获取绘图引擎名称，如果没有指定则使用默认引擎
        if (engine_name := p._layout_spec.get("engine", default)) is not default:
            # None是Figure.set_layout_engine的有效参数，因此使用'default'
            set_layout_engine(self._figure, engine_name)
        elif p._target is None:
            # 如果用户提供了自己的matplotlib图形并且没有通过Plot指定引擎，则不修改布局引擎
            # TODO 是否切换默认值为"constrained"？
            set_layout_engine(self._figure, "tight")

        # 处理布局的extent设置
        if (extent := p._layout_spec.get("extent")) is not None:
            engine = get_layout_engine(self._figure)
            if engine is None:
                self._figure.subplots_adjust(*extent)
            else:
                # 不同的布局引擎矩形参数化方式略有不同
                left, bottom, right, top = extent
                width, height = right - left, top - bottom
                try:
                    # 基础的LayoutEngine.set方法可能不支持rect参数，捕获TypeError异常
                    engine.set(rect=[left, bottom, width, height])  # type: ignore
                except TypeError:
                    # 在正常情况下不应该到达此处
                    pass
```