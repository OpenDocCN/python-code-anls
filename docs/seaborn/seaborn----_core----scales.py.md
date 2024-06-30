# `D:\src\scipysrc\seaborn\seaborn\_core\scales.py`

```
from __future__ import annotations
# 导入未来版本的类型注解支持，使得可以在类中引用自身类型

import re
# 导入正则表达式模块，用于处理字符串匹配和替换操作

from copy import copy
# 导入 copy 函数，用于创建对象的浅拷贝

from collections.abc import Sequence
# 导入 Sequence 抽象基类，用于定义序列类型的通用接口

from dataclasses import dataclass
# 导入 dataclass 装饰器，用于自动创建数据类

from functools import partial
# 导入 partial 函数，用于部分应用一个函数，固定其中的一些参数

from typing import Any, Callable, Tuple, Optional, ClassVar
# 导入类型注解相关的类型，用于函数参数和返回值的类型提示

import numpy as np
# 导入 NumPy 库，用于支持大量的数值计算操作

import matplotlib as mpl
# 导入 Matplotlib 库的主模块

from matplotlib.ticker import (
    Locator,
    Formatter,
    AutoLocator,
    AutoMinorLocator,
    FixedLocator,
    LinearLocator,
    LogLocator,
    SymmetricalLogLocator,
    MaxNLocator,
    MultipleLocator,
    EngFormatter,
    FuncFormatter,
    LogFormatterSciNotation,
    ScalarFormatter,
    StrMethodFormatter,
)
# 从 Matplotlib 的 ticker 模块中导入多种刻度定位器和格式化器类

from matplotlib.dates import (
    AutoDateLocator,
    AutoDateFormatter,
    ConciseDateFormatter,
)
# 从 Matplotlib 的 dates 模块中导入日期相关的定位器和格式化器类

from matplotlib.axis import Axis
# 导入 Matplotlib 的 axis 模块中的 Axis 类

from matplotlib.scale import ScaleBase
# 导入 Matplotlib 的 scale 模块中的 ScaleBase 类

from pandas import Series
# 导入 Pandas 库中的 Series 类型

from seaborn._core.rules import categorical_order
# 从 seaborn 库的 _core.rules 模块中导入 categorical_order 函数

from seaborn._core.typing import Default, default
# 从 seaborn 库的 _core.typing 模块中导入 Default 和 default 类型别名

from typing import TYPE_CHECKING
# 导入 TYPE_CHECKING 常量，用于类型检查时避免循环导入问题

if TYPE_CHECKING:
    from seaborn._core.plot import Plot
    from seaborn._core.properties import Property
    from numpy.typing import ArrayLike, NDArray
    # 在类型检查模式下，导入与绘图相关的 Plot 类和属性相关的 Property 类
    TransFuncs = Tuple[
        Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]
    ]
    # 声明 TransFuncs 类型别名，表示两个数组操作函数的元组

    # 由于类型推断过于复杂，暂时将 Pipeline 类型别名恢复为 Any，待后续重新审视！
    Pipeline = Sequence[Optional[Callable[[Any], Any]]]
    # 声明 Pipeline 类型别名，表示一系列可选的函数对象或 None

class Scale:
    """Base class for objects that map data values to visual properties."""

    values: tuple | str | list | dict | None
    # 声明 values 实例变量，可以是元组、字符串、列表、字典或 None

    _priority: ClassVar[int]
    _pipeline: Pipeline
    _matplotlib_scale: ScaleBase
    _spacer: staticmethod
    _legend: tuple[list[Any], list[str]] | None
    # 声明类变量 _priority、_pipeline、_matplotlib_scale、_spacer 和 _legend

    def __post_init__(self):
        # 初始化方法，在对象创建后自动调用
        self._tick_params = None
        self._label_params = None
        self._legend = None
        # 初始化实例变量 _tick_params、_label_params 和 _legend

    def tick(self):
        # 抽象方法，子类需实现具体逻辑，用于处理刻度相关操作
        raise NotImplementedError()

    def label(self):
        # 抽象方法，子类需实现具体逻辑，用于处理标签相关操作
        raise NotImplementedError()

    def _get_locators(self):
        # 抽象方法，子类需实现具体逻辑，用于获取定位器对象
        raise NotImplementedError()

    def _get_formatter(self, locator: Locator | None = None):
        # 抽象方法，子类需实现具体逻辑，用于获取格式化器对象
        raise NotImplementedError()

    def _get_scale(self, name: str, forward: Callable, inverse: Callable):
        # 方法用于创建和配置刻度对象的内部类 InternalScale
        major_locator, minor_locator = self._get_locators(**self._tick_params)
        # 获取主定位器和次要定位器
        major_formatter = self._get_formatter(major_locator, **self._label_params)
        # 获取主格式化器

        class InternalScale(mpl.scale.FuncScale):
            # 内部类 InternalScale，继承自 mpl.scale.FuncScale

            def set_default_locators_and_formatters(self, axis):
                # 方法用于设置默认的定位器和格式化器
                axis.set_major_locator(major_locator)
                if minor_locator is not None:
                    axis.set_minor_locator(minor_locator)
                axis.set_major_formatter(major_formatter)

        return InternalScale(name, (forward, inverse))
        # 返回 InternalScale 类的实例对象，用于映射数据值到视觉属性
    # 定义一个方法 `_spacing`，接收一个 pandas Series 对象 x 作为参数，返回一个浮点数
    def _spacing(self, x: Series) -> float:
        # 调用内部方法 `_spacer` 处理 x，返回一个数值作为间距
        space = self._spacer(x)
        # 如果间距是 NaN，通常表示坐标数据没有方差
        if np.isnan(space):
            # 当坐标数据的方差不存在时，返回默认值 1
            # 实际上应该根据具体情况确定最佳的默认值，但返回 1 似乎是合理的选择
            return 1
        # 返回计算得到的间距
        return space

    # 定义一个方法 `_setup`，用于设置比例尺，接收 data (Series 类型)、prop (Property 类型) 和可选的 axis (Axis 类型或 None) 作为参数，返回一个 Scale 对象
    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:
        # 抛出未实现错误，子类需要实现该方法以适应具体的比例尺设置
        raise NotImplementedError()

    # 定义一个方法 `_finalize`，在添加艺术家后执行特定于比例尺的轴调整，接收 p (Plot 对象) 和 axis (Axis 对象) 作为参数，无返回值
    def _finalize(self, p: Plot, axis: Axis) -> None:
        """Perform scale-specific axis tweaks after adding artists."""
        # 此处仅注释方法作用，实际没有执行任何操作

    # 定义一个方法 `__call__`，允许对象被调用，接收 data (Series 类型) 作为参数，返回 ArrayLike 对象
    def __call__(self, data: Series) -> ArrayLike:

        # 声明变量 trans_data，类型为 Series 或 NDArray 或 list
        trans_data: Series | NDArray | list

        # TODO 有时需要处理标量数据（例如 Line），但处理的最佳方式是什么？
        # 如果 data 是标量（scalar），返回 True；否则返回 False
        scalar_data = np.isscalar(data)
        # 如果 data 是标量
        if scalar_data:
            # 将 data 转换为包含单个元素的 numpy 数组
            trans_data = np.array([data])
        else:
            # 否则直接使用 data
            trans_data = data

        # 对于管道中的每个函数，依次应用于 trans_data
        for func in self._pipeline:
            # 如果 func 不为 None，将 func 应用于 trans_data
            if func is not None:
                trans_data = func(trans_data)

        # 如果 data 是标量，返回处理后的第一个元素
        if scalar_data:
            return trans_data[0]
        else:
            # 否则返回处理后的 trans_data
            return trans_data

    # 定义一个静态方法 `_identity`
    @staticmethod
    def _identity():

        # 定义一个内部类 Identity，继承自 Scale 类
        class Identity(Scale):
            # 管道为空列表
            _pipeline = []
            # _spacer 为 None
            _spacer = None
            # _legend 为 None
            _legend = None
            # _matplotlib_scale 为 None
            _matplotlib_scale = None

        # 返回 Identity 类的实例
        return Identity()
@dataclass
class Boolean(Scale):
    """
    表示一个离散域为True和False值的标尺。

    其行为类似于:class:`Nominal`标尺，但是属性映射和图例将使用[True, False]的顺序，而不是使用数值规则进行排序。
    坐标变量通过反转轴限制来实现这一点，以保持底层数值定位。
    输入数据将被转换为布尔值，尊重缺失数据。

    """
    values: tuple | list | dict | None = None  # 定义values属性，可以是元组、列表、字典或None

    _priority: ClassVar[int] = 3  # 类变量_priority，优先级为3

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:
        """
        设置标尺对象的初始化状态。

        Args:
            data: 数据系列对象，用于初始化标尺。
            prop: 属性对象，包含标尺的配置信息。
            axis: 可选的坐标轴对象，用于标尺的设置。

        Returns:
            Scale: 更新后的标尺对象。

        """
        new = copy(self)  # 复制当前对象
        if new._tick_params is None:
            new = new.tick()  # 如果tick参数为空，则调用tick方法设置
        if new._label_params is None:
            new = new.label()  # 如果label参数为空，则调用label方法设置

        def na_safe_cast(x):
            # 安全的缺失数据处理函数，将输入数据转换为布尔型数据。

            # 如果输入x是标量
            if np.isscalar(x):
                return float(bool(x))
            else:
                # 如果x具有属性"notna"，处理pd.NA；否则使用np<>pd的NA处理
                if hasattr(x, "notna"):
                    use = x.notna().to_numpy()
                else:
                    use = np.isfinite(x)
                out = np.full(len(x), np.nan, dtype=float)  # 创建一个全为NaN的数组
                out[use] = x[use].astype(bool).astype(float)  # 将有效数据转换为布尔型并赋值给out数组
                return out

        new._pipeline = [na_safe_cast, prop.get_mapping(new, data)]  # 设置数据处理管道
        new._spacer = _default_spacer  # 设置默认间隔函数为_default_spacer
        if prop.legend:
            new._legend = [True, False], ["True", "False"]  # 如果有图例属性，设置图例内容为[True, False], ["True", "False"]

        forward, inverse = _make_identity_transforms()  # 调用函数创建身份转换
        mpl_scale = new._get_scale(str(data.name), forward, inverse)  # 获取matplotlib标尺对象

        axis = PseudoAxis(mpl_scale) if axis is None else axis  # 创建伪坐标轴对象
        mpl_scale.set_default_locators_and_formatters(axis)  # 设置默认定位器和格式化器
        new._matplotlib_scale = mpl_scale  # 设置matplotlib标尺对象

        return new  # 返回更新后的标尺对象

    def _finalize(self, p: Plot, axis: Axis) -> None:
        """
        最终处理函数，用于完成标尺的最后设置。

        Args:
            p: Plot对象，表示绘图对象。
            axis: Axis对象，表示坐标轴对象。

        """
        ax = axis.axes  # 获取坐标轴对象
        name = axis.axis_name  # 获取坐标轴名称
        axis.grid(False, which="both")  # 关闭网格线显示
        if name not in p._limits:
            nticks = len(axis.get_major_ticks())  # 获取主要刻度的数量
            lo, hi = -.5, nticks - .5
            if name == "x":
                lo, hi = hi, lo
            set_lim = getattr(ax, f"set_{name}lim")
            set_lim(lo, hi, auto=None)

    def tick(self, locator: Locator | None = None):
        """
        设置刻度参数并返回新的标尺对象。

        Args:
            locator: 刻度定位器对象，用于设置刻度。

        Returns:
            Boolean: 更新后的标尺对象。

        """
        new = copy(self)  # 复制当前对象
        new._tick_params = {"locator": locator}  # 设置tick参数为给定的定位器对象
        return new  # 返回更新后的标尺对象

    def label(self, formatter: Formatter | None = None):
        """
        设置标签参数并返回新的标尺对象。

        Args:
            formatter: 标签格式化器对象，用于设置标签格式。

        Returns:
            Boolean: 更新后的标尺对象。

        """
        new = copy(self)  # 复制当前对象
        new._label_params = {"formatter": formatter}  # 设置label参数为给定的格式化器对象
        return new  # 返回更新后的标尺对象
    # 返回给定的定位器，如果给定定位器为None，则返回一个固定的定位器和None
    def _get_locators(self, locator):
        # 检查传入的定位器是否为None，如果不是则直接返回该定位器
        if locator is not None:
            return locator
        # 如果定位器为None，则创建一个固定定位器，返回固定定位器和None
        return FixedLocator([0, 1]), None
    
    # 返回给定的格式化器，如果给定格式化器为None，则返回一个lambda函数
    # 该lambda函数将布尔值转换为字符串
    def _get_formatter(self, locator, formatter):
        # 检查传入的格式化器是否为None，如果不是则直接返回该格式化器
        if formatter is not None:
            return formatter
        # 如果格式化器为None，则创建一个lambda函数，用于将布尔值转换为字符串
        return FuncFormatter(lambda x, _: str(bool(x)))
@dataclass
class Nominal(Scale):
    """
    A categorical scale without relative importance / magnitude.
    """
    # Categorical (convert to strings), un-sortable

    values: tuple | str | list | dict | None = None  # 可接受多种类型的值，用于定义标度的取值
    order: list | None = None  # 可选的值的排序顺序

    _priority: ClassVar[int] = 4  # 类变量，优先级设为4

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    """
    Prepare the scale setup for plotting.

    Parameters
    ----------
    data : pandas Series
        The data to be plotted.
    prop : Property
        Property object for configuring the plot.
    axis : Axis or None, optional
        The axis to be plotted on, if specified.

    """

    def _finalize(self, p: Plot, axis: Axis) -> None:
        """
        Finalize the plot settings for the scale.

        Parameters
        ----------
        p : Plot
            The plot object containing all plot settings.
        axis : Axis
            The axis object to be finalized.

        """
        ax = axis.axes
        name = axis.axis_name
        axis.grid(False, which="both")  # 关闭网格线
        if name not in p._limits:  # 如果轴名称不在限制范围内
            nticks = len(axis.get_major_ticks())  # 获取主要刻度的数量
            lo, hi = -.5, nticks - .5  # 设置轴的最小值和最大值
            if name == "y":
                lo, hi = hi, lo
            set_lim = getattr(ax, f"set_{name}lim")  # 根据轴的名称动态获取设置轴限制的方法
            set_lim(lo, hi, auto=None)

    def tick(self, locator: Locator | None = None) -> Nominal:
        """
        Configure the selection of ticks for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.
            At the moment, it is probably not very useful.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass
            Pre-configured matplotlib locator; other parameters will not be used.

        Returns
        -------
        Copy of self with new tick configuration.

        """
        new = copy(self)  # 复制当前对象
        new._tick_params = {"locator": locator}  # 设置新的刻度参数
        return new

    def label(self, formatter: Formatter | None = None) -> Nominal:
        """
        Configure the selection of labels for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.
            At the moment, it is probably not very useful.

        Parameters
        ----------
        formatter : :class:`matplotlib.ticker.Formatter` subclass
            Pre-configured matplotlib formatter; other parameters will not be used.

        Returns
        -------
        Copy of self with new tick configuration.

        """
        new = copy(self)  # 复制当前对象
        new._label_params = {"formatter": formatter}  # 设置新的标签参数
        return new

    def _get_locators(self, locator):
        """
        Get the locators for the scale.

        Parameters
        ----------
        locator : Locator or None
            Matplotlib locator or None.

        Returns
        -------
        Tuple of (locator, None).

        """
        if locator is not None:
            return locator, None

        locator = mpl.category.StrCategoryLocator({})  # 使用字符串类别定位器

        return locator, None

    def _get_formatter(self, locator, formatter):
        """
        Get the formatter for the scale.

        Parameters
        ----------
        locator : Locator
            Matplotlib locator.
        formatter : Formatter or None
            Matplotlib formatter or None.

        Returns
        -------
        formatter : Formatter
            Matplotlib formatter.

        """
        if formatter is not None:
            return formatter

        formatter = mpl.category.StrCategoryFormatter({})  # 使用字符串类别格式化器

        return formatter


@dataclass
class Ordinal(Scale):
    # Categorical (convert to strings), sortable, can skip ticklabels
    ...


@dataclass
class Discrete(Scale):
    # Numeric, integral, can skip ticks/ticklabels
    ...


@dataclass
class ContinuousBase(Scale):

    values: tuple | str | None = None
    norm: tuple | None = None

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
        ) -> Scale:
        # 返回类型声明为 Scale
        new = copy(self)
        # 复制当前对象以进行修改，确保不修改原始对象
        if new._tick_params is None:
            # 如果 tick 参数为空，调用 tick 方法获取默认参数
            new = new.tick()
        if new._label_params is None:
            # 如果 label 参数为空，调用 label 方法获取默认参数
            new = new.label()

        forward, inverse = new._get_transform()
        # 获取前向和反向转换函数

        mpl_scale = new._get_scale(str(data.name), forward, inverse)
        # 使用数据名称、前向和反向转换函数获取 matplotlib 的缩放对象

        if axis is None:
            # 如果未指定 axis 参数，创建一个伪轴对象并更新单位
            axis = PseudoAxis(mpl_scale)
            axis.update_units(data)

        mpl_scale.set_default_locators_and_formatters(axis)
        # 设置 matplotlib 缩放对象的默认定位器和格式化器
        new._matplotlib_scale = mpl_scale
        # 将 matplotlib 缩放对象保存到 new 对象中

        normalize: Optional[Callable[[ArrayLike], ArrayLike]]
        if prop.normed:
            # 如果 prop.normed 为真，执行以下代码块
            if new.norm is None:
                # 如果未设置 new.norm，使用数据的最小值和最大值作为规范化的范围
                vmin, vmax = data.min(), data.max()
            else:
                # 否则使用 new.norm 中指定的范围作为规范化的范围
                vmin, vmax = new.norm
            # 转换 vmin 和 vmax 到与轴相关的单位，并将其转换为浮点数
            vmin, vmax = map(float, axis.convert_units((vmin, vmax)))
            a = forward(vmin)
            b = forward(vmax) - forward(vmin)

            def normalize(x):
                # 定义一个规范化函数，将数据 x 规范化到 [0, 1] 区间
                return (x - a) / b

        else:
            # 如果 prop.normed 不为真，则 normalize、vmin 和 vmax 都设为 None
            normalize = vmin = vmax = None

        new._pipeline = [
            axis.convert_units,  # 单位转换函数
            forward,             # 前向转换函数
            normalize,           # 规范化函数（可能为 None）
            prop.get_mapping(new, data)  # 获取映射函数
        ]

        def spacer(x):
            # 定义一个间隔计算函数，计算唯一值之间的最小差值
            x = x.dropna().unique()
            if len(x) < 2:
                return np.nan
            return np.min(np.diff(np.sort(x)))
        new._spacer = spacer
        # 将间隔计算函数保存到 new 对象中

        # TODO 如何禁用所有属性使用的图例？
        # 可能添加一个 Scale 参数或 Scale.suppress() 方法来实现
        # Scale.legend() 方法可能除了允许 Scale.legend(False) 外，还有其他有用的参数
        # 如何在图例中避免偏移/科学计数法显示问题？
        if prop.legend:
            # 如果 prop.legend 为真，执行以下代码块
            axis.set_view_interval(vmin, vmax)
            locs = axis.major.locator()
            locs = locs[(vmin <= locs) & (locs <= vmax)]
            # 避免图例中的偏移或科学计数法显示问题
            if hasattr(axis.major.formatter, "set_useOffset"):
                axis.major.formatter.set_useOffset(False)
            if hasattr(axis.major.formatter, "set_scientific"):
                axis.major.formatter.set_scientific(False)
            labels = axis.major.formatter.format_ticks(locs)
            # 格式化刻度位置，并保存到 new 对象中的 _legend 属性
            new._legend = list(locs), list(labels)

        return new
        # 返回修改后的 new 对象
    # 定义一个私有方法 `_get_transform`，用于获取变换参数
    def _get_transform(self):

        # 将 self.trans 赋给局部变量 arg
        arg = self.trans

        # 定义内部函数 get_param，用于从参数中获取方法及其默认值
        def get_param(method, default):
            # 如果 arg 等于 method，则返回 default
            if arg == method:
                return default
            # 否则从 arg 中提取以 method 开头的部分作为浮点数返回
            return float(arg[len(method):])

        # 如果 arg 是 None，则返回单位变换
        if arg is None:
            return _make_identity_transforms()
        # 如果 arg 是元组，则直接返回该元组
        elif isinstance(arg, tuple):
            return arg
        # 如果 arg 是字符串
        elif isinstance(arg, str):
            # 如果 arg 是 "ln"，则返回对数变换
            if arg == "ln":
                return _make_log_transforms()
            # 如果 arg 是 "logit"，则根据后续数字创建对数逻辑变换
            elif arg == "logit":
                base = get_param("logit", 10)
                return _make_logit_transforms(base)
            # 如果 arg 以 "log" 开头，则根据后续数字创建对数变换
            elif arg.startswith("log"):
                base = get_param("log", 10)
                return _make_log_transforms(base)
            # 如果 arg 以 "symlog" 开头，则根据后续数字创建对称对数变换
            elif arg.startswith("symlog"):
                c = get_param("symlog", 1)
                return _make_symlog_transforms(c)
            # 如果 arg 以 "pow" 开头，则根据后续数字创建幂次变换
            elif arg.startswith("pow"):
                exp = get_param("pow", 2)
                return _make_power_transforms(exp)
            # 如果 arg 是 "sqrt"，则返回平方根变换
            elif arg == "sqrt":
                return _make_sqrt_transforms()
            # 否则抛出异常，显示未知的 trans 参数值
            else:
                raise ValueError(f"Unknown value provided for trans: {arg!r}")
@dataclass
class Continuous(ContinuousBase):
    """
    A numeric scale supporting norms and functional transforms.
    """
    values: tuple | str | None = None
    trans: str | TransFuncs | None = None

    # TODO Add this to deal with outliers?
    # outside: Literal["keep", "drop", "clip"] = "keep"

    _priority: ClassVar[int] = 1

    def tick(
        self,
        locator: Locator | None = None, *,
        at: Sequence[float] | None = None,
        upto: int | None = None,
        count: int | None = None,
        every: float | None = None,
        between: tuple[float, float] | None = None,
        minor: int | None = None,
    ) -> Continuous:
        """
        Configure the selection of ticks for the scale's axis or legend.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass
            Pre-configured matplotlib locator; other parameters will not be used.
        at : sequence of floats
            Place ticks at these specific locations (in data units).
        upto : int
            Choose "nice" locations for ticks, but do not exceed this number.
        count : int
            Choose exactly this number of ticks, bounded by `between` or axis limits.
        every : float
            Choose locations at this interval of separation (in data units).
        between : pair of floats
            Bound upper / lower ticks when using `every` or `count`.
        minor : int
            Number of unlabeled ticks to draw between labeled "major" ticks.

        Returns
        -------
        scale
            Copy of self with new tick configuration.

        """
        # Input checks

        # 如果 locator 不为空且不是 Locator 的实例，则引发类型错误异常
        if locator is not None and not isinstance(locator, Locator):
            raise TypeError(
                f"Tick locator must be an instance of {Locator!r}, "
                f"not {type(locator)!r}."
            )

        # 解析用于对数参数的 log_base 和 symlog_thresh
        log_base, symlog_thresh = self._parse_for_log_params(self.trans)

        # 如果 log_base 或 symlog_thresh 不为零，则进行以下检查
        if log_base or symlog_thresh:
            # 如果 count 不为 None 并且 between 为 None，则引发运行时错误异常
            if count is not None and between is None:
                raise RuntimeError("`count` requires `between` with log transform.")
            # 如果 every 不为 None，则引发运行时错误异常
            if every is not None:
                raise RuntimeError("`every` not supported with log transform.")

        # 复制当前对象
        new = copy(self)

        # 更新 _tick_params 字典
        new._tick_params = {
            "locator": locator,
            "at": at,
            "upto": upto,
            "count": count,
            "every": every,
            "between": between,
            "minor": minor,
        }

        # 返回更新后的对象副本
        return new

    def label(
        self,
        formatter: Formatter | None = None, *,
        like: str | Callable | None = None,
        base: int | None | Default = default,
        unit: str | None = None,
    def configure_ticks(
        self, formatter: Formatter | None, like: str | callable | None, base: Optional[number] = None, unit: str | tuple[str, str] | None = None
    ) -> Continuous:
        """
        Configure the appearance of tick labels for the scale's axis or legend.

        Parameters
        ----------
        formatter : :class:`matplotlib.ticker.Formatter` subclass
            Pre-configured formatter to use; other parameters will be ignored.
        like : str or callable
            Either a format pattern (e.g., `".2f"`), a format string with fields named
            `x` and/or `pos` (e.g., `"${x:.2f}"`), or a callable with a signature like
            `f(x: float, pos: int) -> str`. In the latter variants, `x` is passed as the
            tick value and `pos` is passed as the tick index.
        base : number, optional
            Use log formatter (with scientific notation) having this value as the base.
            Set to `None` to override the default formatter with a log transform.
        unit : str or (str, str) tuple, optional
            Use SI prefixes with these units (e.g., with `unit="g"`, a tick value
            of 5000 will appear as `5 kg`). When a tuple, the first element gives the
            separator between the number and unit.

        Returns
        -------
        scale
            Copy of self with new label configuration.

        """
        # Input checks
        if formatter is not None and not isinstance(formatter, Formatter):
            raise TypeError(
                f"Label formatter must be an instance of {Formatter!r}, "
                f"not {type(formatter)!r}"
            )
        if like is not None and not (isinstance(like, str) or callable(like)):
            msg = f"`like` must be a string or callable, not {type(like).__name__}."
            raise TypeError(msg)

        # Create a copy of the current object
        new = copy(self)
        # Update label configuration parameters
        new._label_params = {
            "formatter": formatter,
            "like": like,
            "base": base,
            "unit": unit,
        }
        return new

    def _parse_for_log_params(
        self, trans: str | TransFuncs | None
    ) -> tuple[float | None, float | None]:
        """
        Parse transformation parameters for log scale.

        Parameters
        ----------
        trans : str or TransFuncs or None
            Transformation type to parse, can be a string like 'log10' or 'symlog1',
            a TransFuncs object, or None.

        Returns
        -------
        tuple
            Tuple containing log_base (float or None) and symlog_thresh (float or None).

        """
        log_base = symlog_thresh = None
        # Parse for logarithmic base
        if isinstance(trans, str):
            m = re.match(r"^log(\d*)", trans)
            if m is not None:
                log_base = float(m[1] or 10)
            m = re.match(r"symlog(\d*)", trans)
            if m is not None:
                symlog_thresh = float(m[1] or 1)
        return log_base, symlog_thresh
    # 获取坐标轴刻度定位器（locator），根据给定的参数和条件
    def _get_locators(self, locator, at, upto, count, every, between, minor):

        # 解析并获取对数刻度相关参数
        log_base, symlog_thresh = self._parse_for_log_params(self.trans)

        # 如果指定了locator参数，则使用该参数作为主刻度定位器
        if locator is not None:
            major_locator = locator

        # 如果指定了upto参数，则根据对数基数或者普通情况创建LogLocator或MaxNLocator
        elif upto is not None:
            if log_base:
                major_locator = LogLocator(base=log_base, numticks=upto)
            else:
                major_locator = MaxNLocator(upto, steps=[1, 1.5, 2, 2.5, 3, 5, 10])

        # 如果指定了count参数，则根据between参数创建LinearLocator或FixedLocator
        elif count is not None:
            if between is None:
                # 这种情况很少见（除非正在设置限制）
                major_locator = LinearLocator(count)
            else:
                if log_base or symlog_thresh:
                    forward, inverse = self._get_transform()
                    lo, hi = forward(between)
                    ticks = inverse(np.linspace(lo, hi, num=count))
                else:
                    ticks = np.linspace(*between, num=count)
                major_locator = FixedLocator(ticks)

        # 如果指定了every参数，则根据between参数创建MultipleLocator或FixedLocator
        elif every is not None:
            if between is None:
                major_locator = MultipleLocator(every)
            else:
                lo, hi = between
                ticks = np.arange(lo, hi + every, every)
                major_locator = FixedLocator(ticks)

        # 如果指定了at参数，则使用FixedLocator创建主刻度定位器
        elif at is not None:
            major_locator = FixedLocator(at)

        # 如果没有指定任何参数，则根据对数基数或者对称对数阈值创建LogLocator、SymmetricalLogLocator或AutoLocator
        else:
            if log_base:
                major_locator = LogLocator(log_base)
            elif symlog_thresh:
                major_locator = SymmetricalLogLocator(linthresh=symlog_thresh, base=10)
            else:
                major_locator = AutoLocator()

        # 如果未指定minor参数，则根据对数基数创建LogLocator或者为None
        if minor is None:
            minor_locator = LogLocator(log_base, subs=None) if log_base else None
        else:
            # 如果指定了minor参数，则根据对数基数或者创建AutoMinorLocator
            if log_base:
                subs = np.linspace(0, log_base, minor + 2)[1:-1]
                minor_locator = LogLocator(log_base, subs=subs)
            else:
                minor_locator = AutoMinorLocator(minor + 1)

        # 返回主刻度定位器和次刻度定位器
        return major_locator, minor_locator
    # 获取格式化器函数，根据给定的定位器、格式化器、参考样式、基数和单位进行处理

    # 解析用于对数参数的日志基数和对数阈值
    log_base, symlog_thresh = self._parse_for_log_params(self.trans)

    # 如果未指定基数，则根据对数阈值自动确定基数为10
    if base is default:
        if symlog_thresh:
            log_base = 10
        base = log_base

    # 如果已经指定了格式化器，则直接返回
    if formatter is not None:
        return formatter

    # 如果指定了参考样式，根据字符串类型或函数类型进行不同的处理
    if like is not None:
        if isinstance(like, str):
            # 如果参考样式包含 "{x" 或 "{pos"，直接使用该样式作为格式化字符串
            if "{x" in like or "{pos" in like:
                fmt = like
            else:
                # 否则，将样式插入到 "{x:样式}" 的格式化字符串中
                fmt = f"{{x:{like}}}"
            formatter = StrMethodFormatter(fmt)
        else:
            # 如果参考样式是函数类型，则使用 FuncFormatter 进行处理
            formatter = FuncFormatter(like)

    # 如果未指定参考样式，但指定了基数，则使用科学计数法格式化器
    elif base is not None:
        # 这里可以根据需要添加其他对数选项
        formatter = LogFormatterSciNotation(base)

    # 如果未指定基数，但指定了单位
    elif unit is not None:
        if isinstance(unit, tuple):
            sep, unit = unit
        elif not unit:
            sep = ""
        else:
            sep = " "
        # 使用 EngFormatter 进行格式化，设置单位之间的分隔符
        formatter = EngFormatter(unit, sep=sep)

    # 如果未指定任何特定格式化选项，则使用 ScalarFormatter 作为默认格式化器
    else:
        formatter = ScalarFormatter()

    # 返回确定的格式化器
    return formatter
@dataclass
class Temporal(ContinuousBase):
    """
    A scale for date/time data.
    """
    # TODO date: bool?
    # For when we only care about the time component, would affect
    # default formatter and norm conversion. Should also happen in
    # Property.default_scale. The alternative was having distinct
    # Calendric / Temporal scales, but that feels a bit fussy, and it
    # would get in the way of using first-letter shorthands because
    # Calendric and Continuous would collide. Still, we haven't implemented
    # those yet, and having a clear distinction betewen date(time) / time
    # may be more useful.

    trans = None  # 初始化属性 trans 为 None

    _priority: ClassVar[int] = 2  # 类变量 _priority 被设定为 2

    def tick(
        self, locator: Locator | None = None, *,
        upto: int | None = None,
    ) -> Temporal:
        """
        Configure the selection of ticks for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass
            Pre-configured matplotlib locator; other parameters will not be used.
        upto : int
            Choose "nice" locations for ticks, but do not exceed this number.

        Returns
        -------
        scale
            Copy of self with new tick configuration.

        """
        if locator is not None and not isinstance(locator, Locator):
            err = (
                f"Tick locator must be an instance of {Locator!r}, "
                f"not {type(locator)!r}."
            )
            raise TypeError(err)

        new = copy(self)  # 复制当前对象 self
        new._tick_params = {"locator": locator, "upto": upto}  # 设置新的 tick 参数
        return new  # 返回配置后的新对象

    def label(
        self,
        formatter: Formatter | None = None, *,
        concise: bool = False,
    ) -> Temporal:
        """
        Configure the appearance of tick labels for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.

        Parameters
        ----------
        formatter : :class:`matplotlib.ticker.Formatter` subclass
            Pre-configured formatter to use; other parameters will be ignored.
        concise : bool
            If True, use :class:`matplotlib.dates.ConciseDateFormatter` to make
            the tick labels as compact as possible.

        Returns
        -------
        scale
            Copy of self with new label configuration.

        """
        new = copy(self)  # 复制当前对象 self
        new._label_params = {"formatter": formatter, "concise": concise}  # 设置新的 label 参数
        return new  # 返回配置后的新对象

    def _get_locators(self, locator, upto):
        """
        Determine major and minor locators for tick marks.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass or None
            Pre-configured locator for major ticks; if None, one will be auto-generated.
        upto : int or None
            Maximum number of tick marks desired.

        Returns
        -------
        Tuple
            A tuple containing major and minor locators.

        """
        if locator is not None:
            major_locator = locator  # 如果有指定 locator，则使用该 locator
        elif upto is not None:
            major_locator = AutoDateLocator(minticks=2, maxticks=upto)  # 如果 upto 参数不为 None，则生成对应的 AutoDateLocator

        else:
            major_locator = AutoDateLocator(minticks=2, maxticks=6)  # 否则生成默认参数的 AutoDateLocator
        minor_locator = None  # 次要定位器设为 None

        return major_locator, minor_locator  # 返回主要和次要定位器
    # 定义一个方法用于获取格式化器，接受定位器、格式化器和是否简洁作为参数
    def _get_formatter(self, locator, formatter, concise):
        # 如果 formatter 参数已经提供，则直接返回该格式化器
        if formatter is not None:
            return formatter

        # 如果 concise 参数为 True，则创建一个 ConciseDateFormatter 格式化器
        if concise:
            # TODO 理想情况下我们会有简洁的坐标刻度，但是完整的语义刻度。这是否可能？
            formatter = ConciseDateFormatter(locator)
        else:
            # 否则创建一个 AutoDateFormatter 格式化器
            formatter = AutoDateFormatter(locator)

        # 返回选定的格式化器
        return formatter
# ----------------------------------------------------------------------------------- #

# TODO Have this separate from Temporal or have Temporal(date=True) or similar?
# class Calendric(Scale):
# 
# TODO Needed? Or handle this at layer (in stat or as param, eg binning=)
# class Binned(Scale):
# 
# TODO any need for color-specific scales?
# class Sequential(Continuous):
# class Diverging(Continuous):
# class Qualitative(Nominal):

# ----------------------------------------------------------------------------------- #

class PseudoAxis:
    """
    Internal class implementing minimal interface equivalent to matplotlib Axis.

    Coordinate variables are typically scaled by attaching the Axis object from
    the figure where the plot will end up. Matplotlib has no similar concept of
    and axis for the other mappable variables (color, etc.), but to simplify the
    code, this object acts like an Axis and can be used to scale other variables.

    """
    axis_name = ""  # Matplotlib requirement but not actually used

    def __init__(self, scale):
        # 初始化 PseudoAxis 实例
        self.converter = None
        self.units = None
        self.scale = scale  # 设置比例尺对象
        self.major = mpl.axis.Ticker()  # 创建主要刻度对象
        self.minor = mpl.axis.Ticker()  # 创建次要刻度对象

        # It appears that this needs to be initialized this way on matplotlib 3.1,
        # but not later versions. It is unclear whether there are any issues with it.
        self._data_interval = None, None  # 初始化数据区间

        # 设置默认的定位器和格式化器
        scale.set_default_locators_and_formatters(self)
        # self.set_default_intervals()  Is this ever needed? 是否需要设置默认间隔?

    def set_view_interval(self, vmin, vmax):
        # 设置视图区间
        self._view_interval = vmin, vmax

    def get_view_interval(self):
        # 获取视图区间
        return self._view_interval

    # TODO do we want to distinguish view/data intervals? e.g. for a legend
    # we probably want to represent the full range of the data values, but
    # still norm the colormap. If so, we'll need to track data range separately
    # from the norm, which we currently don't do.

    def set_data_interval(self, vmin, vmax):
        # 设置数据区间
        self._data_interval = vmin, vmax

    def get_data_interval(self):
        # 获取数据区间
        return self._data_interval

    def get_tick_space(self):
        # 获取刻度空间
        # TODO how to do this in a configurable / auto way?
        # Would be cool to have legend density adapt to figure size, etc.
        return 5

    def set_major_locator(self, locator):
        # 设置主要刻度定位器
        self.major.locator = locator
        locator.set_axis(self)

    def set_major_formatter(self, formatter):
        # 设置主要刻度格式化器
        self.major.formatter = formatter
        formatter.set_axis(self)

    def set_minor_locator(self, locator):
        # 设置次要刻度定位器
        self.minor.locator = locator
        locator.set_axis(self)

    def set_minor_formatter(self, formatter):
        # 设置次要刻度格式化器
        self.minor.formatter = formatter
        formatter.set_axis(self)

    def set_units(self, units):
        # 设置单位
        self.units = units
    def update_units(self, x):
        """
        Pass units to the internal converter, potentially updating its mapping.
        将单位传递给内部转换器，可能更新其映射。

        """
        self.converter = mpl.units.registry.get_converter(x)
        """
        获取由matplotlib单位注册表管理的转换器对象。
        """

        if self.converter is not None:
            self.converter.default_units(x, self)
            """
            如果转换器对象不为None，则设置默认单位。

            """

            info = self.converter.axisinfo(self.units, self)
            """
            使用转换器获取关于当前单位和轴的信息。

            """

            if info is None:
                return
            """
            如果info为None，则退出函数。

            """

            if info.majloc is not None:
                self.set_major_locator(info.majloc)
                """
                如果主刻度位置不为None，则设置主刻度定位器。

                """

            if info.majfmt is not None:
                self.set_major_formatter(info.majfmt)
                """
                如果主刻度格式化器不为None，则设置主刻度格式化器。

                """

            # This is in matplotlib method; do we need this?
            # self.set_default_intervals()

    def convert_units(self, x):
        """
        Return a numeric representation of the input data.
        返回输入数据的数值表示。

        """
        if np.issubdtype(np.asarray(x).dtype, np.number):
            return x
        elif self.converter is None:
            return x
        """
        如果输入数据的数据类型是数值，则直接返回。
        如果转换器对象为None，则直接返回输入数据。

        """

        return self.converter.convert(x, self.units, self)
        """
        否则，使用转换器对象将输入数据转换为当前单位的数值表示。

        """

    def get_scale(self):
        """
        Note that matplotlib actually returns a string here!
        (e.g., with a log scale, axis.get_scale() returns "log")
        Currently we just hit it with minor ticks where it checks for
        scale == "log". I'm not sure how you'd actually use log-scale
        minor "ticks" in a legend context, so this is fine....
        返回当前轴的比例尺。
        注意，matplotlib实际上在这里返回一个字符串！
        （例如，使用对数比例尺，axis.get_scale()返回"log"）
        目前我们只是在检查比例尺是否为"log"时使用了次要刻度标记。
        我不确定你实际上如何在图例上下文中使用对数比例尺的次要“刻度”，所以这样做没问题....

        """
        return self.scale

    def get_majorticklocs(self):
        """
        Return the locations of the major ticks.
        返回主刻度的位置。

        """
        return self.major.locator()
        """
        返回主刻度定位器的位置。

        """
# ------------------------------------------------------------------------------------ #
# Transform function creation

# 创建返回恒等函数的工厂函数
def _make_identity_transforms() -> TransFuncs:

    # 定义一个恒等函数，返回输入值
    def identity(x):
        return x

    return identity, identity  # 返回恒等函数两次，作为一对变换函数


# 创建返回对数变换函数的工厂函数，基于给定的基数
def _make_logit_transforms(base: float | None = None) -> TransFuncs:

    # 调用_make_log_transforms函数获取对数和指数函数
    log, exp = _make_log_transforms(base)

    # 定义logit函数，实现对数变换
    def logit(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return log(x) - log(1 - x)

    # 定义expit函数，实现逆对数变换
    def expit(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return exp(x) / (1 + exp(x))

    return logit, expit  # 返回对数变换函数和逆对数变换函数


# 创建返回对数变换函数的工厂函数，基于给定的基数
def _make_log_transforms(base: float | None = None) -> TransFuncs:

    # 定义一个元组变量fs用于存储对数和指数函数
    fs: TransFuncs

    # 根据给定的基数选择对数和指数函数
    if base is None:
        fs = np.log, np.exp
    elif base == 2:
        fs = np.log2, partial(np.power, 2)
    elif base == 10:
        fs = np.log10, partial(np.power, 10)
    else:
        # 如果基数不是None、2或10，定义一个自定义的对数函数
        def forward(x):
            return np.log(x) / np.log(base)
        fs = forward, partial(np.power, base)

    # 定义log函数，实现对数变换
    def log(x: ArrayLike) -> ArrayLike:
        with np.errstate(invalid="ignore", divide="ignore"):
            return fs[0](x)

    # 定义exp函数，实现逆对数变换
    def exp(x: ArrayLike) -> ArrayLike:
        with np.errstate(invalid="ignore", divide="ignore"):
            return fs[1](x)

    return log, exp  # 返回对数变换函数和逆对数变换函数


# 创建返回对称对数变换函数的工厂函数，基于给定的参数c和基数
def _make_symlog_transforms(c: float = 1, base: float = 10) -> TransFuncs:

    # 使用_make_log_transforms函数获取对数和指数函数
    log, exp = _make_log_transforms(base)

    # 定义symlog函数，实现对称对数变换
    def symlog(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sign(x) * log(1 + np.abs(np.divide(x, c)))

    # 定义symexp函数，实现逆对称对数变换
    def symexp(x):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sign(x) * c * (exp(np.abs(x)) - 1)

    return symlog, symexp  # 返回对称对数变换函数和逆对称对数变换函数


# 创建返回平方根变换函数的工厂函数
def _make_sqrt_transforms() -> TransFuncs:

    # 定义sqrt函数，实现平方根变换
    def sqrt(x):
        return np.sign(x) * np.sqrt(np.abs(x))

    # 定义square函数，实现平方变换
    def square(x):
        return np.sign(x) * np.square(x)

    return sqrt, square  # 返回平方根变换函数和平方变换函数


# 创建返回幂次变换函数的工厂函数，基于给定的指数exp
def _make_power_transforms(exp: float) -> TransFuncs:

    # 定义forward函数，实现幂次正向变换
    def forward(x):
        return np.sign(x) * np.power(np.abs(x), exp)

    # 定义inverse函数，实现幂次反向变换
    def inverse(x):
        return np.sign(x) * np.power(np.abs(x), 1 / exp)

    return forward, inverse  # 返回幂次正向变换函数和反向变换函数


# 定义默认的空间函数，返回1作为默认值
def _default_spacer(x: Series) -> float:
    return 1
```