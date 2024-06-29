# `D:\src\scipysrc\matplotlib\lib\matplotlib\ticker.pyi`

```
from collections.abc import Callable, Sequence  # 导入标准库中的 Callable 和 Sequence 抽象基类
from typing import Any, Literal  # 导入用于类型提示的 Any 和 Literal

from matplotlib.axis import Axis  # 导入 matplotlib 库中的 Axis 类
from matplotlib.transforms import Transform  # 导入 matplotlib 库中的 Transform 类
from matplotlib.projections.polar import _AxisWrapper  # 导入 matplotlib 库中的 _AxisWrapper 类

import numpy as np  # 导入 numpy 库，并将其重命名为 np

class _DummyAxis:
    __name__: str  # 类型提示，表示类有一个名为 __name__ 的字符串属性

    def __init__(self, minpos: float = ...) -> None: ...
    # 构造函数，接受一个可选的浮点数参数 minpos，并不返回任何内容

    def get_view_interval(self) -> tuple[float, float]: ...
    # 返回视图间隔的方法，返回一个包含两个浮点数的元组

    def set_view_interval(self, vmin: float, vmax: float) -> None: ...
    # 设置视图间隔的方法，接受两个浮点数参数 vmin 和 vmax，不返回任何内容

    def get_minpos(self) -> float: ...
    # 获取最小位置的方法，返回一个浮点数

    def get_data_interval(self) -> tuple[float, float]: ...
    # 返回数据间隔的方法，返回一个包含两个浮点数的元组

    def set_data_interval(self, vmin: float, vmax: float) -> None: ...
    # 设置数据间隔的方法，接受两个浮点数参数 vmin 和 vmax，不返回任何内容

    def get_tick_space(self) -> int: ...
    # 获取刻度空间的方法，返回一个整数

class TickHelper:
    axis: None | Axis | _DummyAxis | _AxisWrapper  # axis 属性可以是 None 或 Axis、_DummyAxis 或 _AxisWrapper 的实例，或者为空
    def set_axis(self, axis: Axis | _DummyAxis | _AxisWrapper | None) -> None: ...
    # 设置 axis 属性的方法，接受一个参数 axis，类型可以是 Axis、_DummyAxis 或 _AxisWrapper 的实例，或者为空，不返回任何内容

    def create_dummy_axis(self, **kwargs) -> None: ...
    # 创建虚拟坐标轴的方法，接受任意关键字参数，不返回任何内容

class Formatter(TickHelper):
    locs: list[float]  # locs 属性是一个浮点数列表

    def __call__(self, x: float, pos: int | None = ...) -> str: ...
    # 调用对象时的方法，接受一个浮点数参数 x 和一个可选的整数参数 pos，返回一个字符串

    def format_ticks(self, values: list[float]) -> list[str]: ...
    # 格式化刻度的方法，接受一个浮点数列表参数 values，返回一个字符串列表

    def format_data(self, value: float) -> str: ...
    # 格式化数据的方法，接受一个浮点数参数 value，返回一个字符串

    def format_data_short(self, value: float) -> str: ...
    # 简短格式化数据的方法，接受一个浮点数参数 value，返回一个字符串

    def get_offset(self) -> str: ...
    # 获取偏移量的方法，返回一个字符串

    def set_locs(self, locs: list[float]) -> None: ...
    # 设置 locs 属性的方法，接受一个浮点数列表参数 locs，不返回任何内容

    @staticmethod
    def fix_minus(s: str) -> str: ...
    # 静态方法，接受一个字符串参数 s，返回一个字符串

class NullFormatter(Formatter): ...
# NullFormatter 类继承自 Formatter 类，未提供进一步的具体说明

class FixedFormatter(Formatter):
    seq: Sequence[str]  # seq 属性是一个字符串序列
    offset_string: str  # offset_string 属性是一个字符串

    def __init__(self, seq: Sequence[str]) -> None: ...
    # 构造函数，接受一个字符串序列参数 seq，并不返回任何内容

    def set_offset_string(self, ofs: str) -> None: ...
    # 设置 offset_string 属性的方法，接受一个字符串参数 ofs，不返回任何内容

class FuncFormatter(Formatter):
    func: Callable[[float, int | None], str]  # func 属性是一个接受浮点数和可选整数参数的可调用对象，返回一个字符串
    offset_string: str  # offset_string 属性是一个字符串

    def __init__(self, func: Callable[..., str]) -> None: ...
    # 构造函数，接受一个接受任意参数的可调用对象 func，返回一个字符串，并不返回任何内容

    def set_offset_string(self, ofs: str) -> None: ...
    # 设置 offset_string 属性的方法，接受一个字符串参数 ofs，不返回任何内容

class FormatStrFormatter(Formatter):
    fmt: str  # fmt 属性是一个字符串

    def __init__(self, fmt: str) -> None: ...
    # 构造函数，接受一个字符串参数 fmt，并不返回任何内容

class StrMethodFormatter(Formatter):
    fmt: str  # fmt 属性是一个字符串

    def __init__(self, fmt: str) -> None: ...
    # 构造函数，接受一个字符串参数 fmt，并不返回任何内容

class ScalarFormatter(Formatter):
    orderOfMagnitude: int  # orderOfMagnitude 属性是一个整数
    format: str  # format 属性是一个字符串

    def __init__(
        self,
        useOffset: bool | float | None = ...,
        useMathText: bool | None = ...,
        useLocale: bool | None = ...,
    ) -> None: ...
    # 构造函数，接受三个可选参数 useOffset、useMathText 和 useLocale，不返回任何内容

    offset: float  # offset 属性是一个浮点数

    def get_useOffset(self) -> bool: ...
    # 获取 useOffset 属性的方法，返回一个布尔值

    def set_useOffset(self, val: bool | float) -> None: ...
    # 设置 useOffset 属性的方法，接受一个布尔值或浮点数参数 val，不返回任何内容

    @property
    def useOffset(self) -> bool: ...
    # useOffset 属性的 getter 方法，返回一个布尔值

    @useOffset.setter
    def useOffset(self, val: bool | float) -> None: ...
    # useOffset 属性的 setter 方法，接受一个布尔值或浮点数参数 val，不返回任何内容

    def get_useLocale(self) -> bool: ...
    # 获取 useLocale 属性的方法，返回一个布尔值

    def set_useLocale(self, val: bool | None) -> None: ...
    # 设置 useLocale 属性的方法，接受一个布尔值或 None 参数 val，不返回任何内容

    @property
    def useLocale(self) -> bool: ...
    # useLocale 属性的 getter 方法，返回一个布尔值

    @useLocale.setter
    def useLocale(self, val: bool | None) -> None: ...
    # useLocale 属性的 setter 方法，接受一个布尔值或 None 参数 val，不返回任何内容

    def get_useMathText(self) -> bool: ...
    # 获取 useMathText 属性的方法，返回一个布尔值

    def set_useMathText(self, val: bool | None) -> None: ...
    # 设置 useMathText 属性的方法，接受一个布尔值或 None 参数 val，不返回任何内容

    @property
    def useMathText(self) -> bool: ...
    # useMathText 属性的 getter 方法，返回一个布尔值

    @useMathText.setter
    def useMathText(self, val: bool | None) -> None: ...
    # useMathText 属性的 setter 方法，接受一个布尔值或 None 参数 val，不返回任何内容
    # 设置科学计数法显示的格式，接受一个布尔值作为参数
    def set_scientific(self, b: bool) -> None: ...
    
    # 设置数值的显示范围限制，接受一个包含两个整数的元组作为参数
    def set_powerlimits(self, lims: tuple[int, int]) -> None: ...
    
    # 格式化浮点数或者可能包含掩码的 MaskedArray 数据为字符串，返回格式化后的短字符串
    def format_data_short(self, value: float | np.ma.MaskedArray) -> str: ...
    
    # 格式化浮点数为字符串，返回完整的格式化后的字符串
    def format_data(self, value: float) -> str: ...
class LogFormatter(Formatter):
    # LogFormatter类继承自Formatter类，用于格式化对数轴的标签显示

    minor_thresholds: tuple[float, float]
    # minor_thresholds属性，用于存储次要阈值的元组，指定标签显示的次要刻度范围

    def __init__(
        self,
        base: float = ...,
        labelOnlyBase: bool = ...,
        minor_thresholds: tuple[float, float] | None = ...,
        linthresh: float | None = ...,
    ) -> None:
        # 初始化方法，设定基数base，是否仅显示基数labelOnlyBase，次要阈值minor_thresholds和linthresh
        ...

    def set_base(self, base: float) -> None:
        # 设置基数base的方法
        ...

    def set_label_minor(self, labelOnlyBase: bool) -> None:
        # 设置是否仅显示基数的方法
        ...

    def set_locs(self, locs: Any | None = ...) -> None:
        # 设置刻度位置的方法
        ...

    def format_data(self, value: float) -> str:
        # 格式化数据值为字符串的方法
        ...

    def format_data_short(self, value: float) -> str:
        # 格式化数据值为简短字符串的方法
        ...


class LogFormatterExponent(LogFormatter):
    # LogFormatterExponent类继承自LogFormatter类，用于指数格式的对数轴标签显示
    ...


class LogFormatterMathtext(LogFormatter):
    # LogFormatterMathtext类继承自LogFormatter类，用于数学文本格式的对数轴标签显示
    ...


class LogFormatterSciNotation(LogFormatterMathtext):
    # LogFormatterSciNotation类继承自LogFormatterMathtext类，用于科学计数法格式的对数轴标签显示
    ...


class LogitFormatter(Formatter):
    # LogitFormatter类继承自Formatter类，用于逻辑刻度标签的格式化显示

    def __init__(
        self,
        *,
        use_overline: bool = ...,
        one_half: str = ...,
        minor: bool = ...,
        minor_threshold: int = ...,
        minor_number: int = ...
    ) -> None:
        # 初始化方法，设定是否使用overline(use_overline)，one_half标记(one_half)，是否使用次要刻度(minor)，次要阈值(minor_threshold)，次要刻度数(minor_number)
        ...

    def use_overline(self, use_overline: bool) -> None:
        # 设置是否使用overline的方法
        ...

    def set_one_half(self, one_half: str) -> None:
        # 设置one_half标记的方法
        ...

    def set_minor_threshold(self, minor_threshold: int) -> None:
        # 设置次要阈值的方法
        ...

    def set_minor_number(self, minor_number: int) -> None:
        # 设置次要刻度数的方法
        ...

    def format_data_short(self, value: float) -> str:
        # 格式化数据值为简短字符串的方法
        ...


class EngFormatter(Formatter):
    # EngFormatter类继承自Formatter类，用于工程符号前缀的格式化显示

    ENG_PREFIXES: dict[int, str]
    # ENG_PREFIXES属性，存储工程符号前缀的字典

    unit: str
    places: int | None
    sep: str

    def __init__(
        self,
        unit: str = ...,
        places: int | None = ...,
        sep: str = ...,
        *,
        usetex: bool | None = ...,
        useMathText: bool | None = ...
    ) -> None:
        # 初始化方法，设定单位unit，小数位数places，分隔符sep，以及是否使用LaTeX(usetex)和数学文本(useMathText)
        ...

    def get_usetex(self) -> bool:
        # 获取是否使用LaTeX的方法
        ...

    def set_usetex(self, val: bool | None) -> None:
        # 设置是否使用LaTeX的方法
        ...

    @property
    def usetex(self) -> bool:
        # 获取是否使用LaTeX的属性
        ...

    @usetex.setter
    def usetex(self, val: bool | None) -> None:
        # 设置是否使用LaTeX的属性
        ...

    def get_useMathText(self) -> bool:
        # 获取是否使用数学文本的方法
        ...

    def set_useMathText(self, val: bool | None) -> None:
        # 设置是否使用数学文本的方法
        ...

    @property
    def useMathText(self) -> bool:
        # 获取是否使用数学文本的属性
        ...

    @useMathText.setter
    def useMathText(self, val: bool | None) -> None:
        # 设置是否使用数学文本的属性
        ...

    def format_eng(self, num: float) -> str:
        # 格式化工程符号前缀的方法
        ...


class PercentFormatter(Formatter):
    # PercentFormatter类继承自Formatter类，用于百分比格式的数据显示

    xmax: float
    decimals: int | None

    def __init__(
        self,
        xmax: float = ...,
        decimals: int | None = ...,
        symbol: str | None = ...,
        is_latex: bool = ...,
    ) -> None:
        # 初始化方法，设定最大值xmax，小数位数decimals，符号symbol，是否使用LaTeX(is_latex)
        ...

    def format_pct(self, x: float, display_range: float) -> str:
        # 格式化百分比的方法
        ...

    def convert_to_pct(self, x: float) -> float:
        # 转换为百分比的方法
        ...

    @property
    def symbol(self) -> str:
        # 获取符号的属性
        ...

    @symbol.setter
    def symbol(self, symbol: str) -> None:
        # 设置符号的属性
        ...


class Locator(TickHelper):
    # Locator类继承自TickHelper类，用于刻度定位辅助功能

    MAXTICKS: int
    # MAXTICKS常量，指定最大刻度数

    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
        # 计算刻度值的方法，根据给定的最小值vmin和最大值vmax返回刻度值的序列
        ...

    def set_params(self) -> None:
        # 设置参数的方法，实际上是一个空操作，只发出警告
        # 声明为**kwargs，但除了警告之外没有任何作用
        ...

    def __call__(self) -> Sequence[float]:
        # 调用实例时返回刻度值的方法
        ...
    # 定义一个类方法用于检查位置序列中的数值是否超过某个阈值，如果超过则引发异常
    def raise_if_exceeds(self, locs: Sequence[float]) -> Sequence[float]: ...
    
    # 定义一个类方法用于检查两个浮点数是否都非零，返回一个元组包含两个浮点数
    def nonsingular(self, v0: float, v1: float) -> tuple[float, float]: ...
    
    # 定义一个类方法用于确定视图的上下限，接受两个浮点数参数，返回一个包含两个浮点数的元组
    def view_limits(self, vmin: float, vmax: float) -> tuple[float, float]: ...
class IndexLocator(Locator):
    offset: float  # 偏移量，表示索引定位器的偏移值

    def __init__(self, base: float, offset: float) -> None: ...
        # 初始化方法，base为基础值，offset为偏移值

    def set_params(
        self, base: float | None = ..., offset: float | None = ...
    ) -> None: ...
        # 设置参数方法，可以设置基础值和偏移值

class FixedLocator(Locator):
    nbins: int | None  # 分段数，或者为None表示分段数未知

    def __init__(self, locs: Sequence[float], nbins: int | None = ...) -> None: ...
        # 初始化方法，locs为浮点数序列，nbins为分段数或者未知

    def set_params(self, nbins: int | None = ...) -> None: ...
        # 设置参数方法，可以设置分段数或者未知

class NullLocator(Locator): ...
    # 空定位器类，未实现具体功能

class LinearLocator(Locator):
    presets: dict[tuple[float, float], Sequence[float]]
    # 预设值，字典结构，键为(float, float)元组，值为浮点数序列

    def __init__(
        self,
        numticks: int | None = ...,
        presets: dict[tuple[float, float], Sequence[float]] | None = ...,
    ) -> None: ...
        # 初始化方法，numticks为刻度数或者未知，presets为预设值或者未知

    @property
    def numticks(self) -> int: ...
        # 属性方法，返回刻度数

    @numticks.setter
    def numticks(self, numticks: int | None) -> None: ...
        # 属性方法，设置刻度数

    def set_params(
        self,
        numticks: int | None = ...,
        presets: dict[tuple[float, float], Sequence[float]] | None = ...,
    ) -> None: ...
        # 设置参数方法，可以设置刻度数和预设值

class MultipleLocator(Locator):
    def __init__(self, base: float = ..., offset: float = ...) -> None: ...
        # 初始化方法，base为基础值，默认为...，offset为偏移值，默认为...

    def set_params(self, base: float | None = ..., offset: float | None = ...) -> None: ...
        # 设置参数方法，可以设置基础值或偏移值或未知

    def view_limits(self, dmin: float, dmax: float) -> tuple[float, float]: ...
        # 查看限制方法，给定最小值和最大值，返回元组表示限制范围

class _Edge_integer:
    step: float  # 步长，表示整数边界的步长值

    def __init__(self, step: float, offset: float) -> None: ...
        # 初始化方法，step为步长，offset为偏移值

    def closeto(self, ms: float, edge: float) -> bool: ...
        # 靠近方法，判断ms是否接近edge，返回布尔值

    def le(self, x: float) -> float: ...
        # 小于等于方法，返回小于等于x的值

    def ge(self, x: float) -> float: ...
        # 大于等于方法，返回大于等于x的值

class MaxNLocator(Locator):
    default_params: dict[str, Any]
    # 默认参数，字典结构，键为字符串，值为任意类型

    def __init__(self, nbins: int | Literal["auto"] | None = ..., **kwargs) -> None: ...
        # 初始化方法，nbins为分段数、自动或未知，**kwargs为额外关键字参数

    def set_params(self, **kwargs) -> None: ...
        # 设置参数方法，接受任意数量的关键字参数

    def view_limits(self, dmin: float, dmax: float) -> tuple[float, float]: ...
        # 查看限制方法，给定最小值和最大值，返回元组表示限制范围

class LogLocator(Locator):
    numdecs: float  # 数量，表示对数定位器的数量
    numticks: int | None  # 刻度数，或者为None表示刻度数未知

    def __init__(
        self,
        base: float = ...,
        subs: None | Literal["auto", "all"] | Sequence[float] = ...,
        numdecs: float = ...,
        numticks: int | None = ...,
    ) -> None: ...
        # 初始化方法，base为基数，默认为...，subs为替换或全部或序列，numdecs为数量，numticks为刻度数或未知

    def set_params(
        self,
        base: float | None = ...,
        subs: Literal["auto", "all"] | Sequence[float] | None = ...,
        numdecs: float | None = ...,
        numticks: int | None = ...,
    ) -> None: ...
        # 设置参数方法，可以设置基数、替换或全部或序列、数量、刻度数或未知

class SymmetricalLogLocator(Locator):
    numticks: int  # 刻度数，表示对称对数定位器的刻度数

    def __init__(
        self,
        transform: Transform | None = ...,
        subs: Sequence[float] | None = ...,
        linthresh: float | None = ...,
        base: float | None = ...,
    ) -> None: ...
        # 初始化方法，transform为转换或未知，subs为序列或未知，linthresh为线性阈值或未知，base为基数或未知

    def set_params(
        self, subs: Sequence[float] | None = ..., numticks: int | None = ...
    ) -> None: ...
        # 设置参数方法，可以设置替换序列或未知、刻度数或未知

class AsinhLocator(Locator):
    linear_width: float  # 线性宽度，表示反正弦定位器的线性宽度
    numticks: int  # 刻度数，表示反正弦定位器的刻度数
    symthresh: float  # 对称阈值，表示反正弦定位器的对称阈值
    base: int  # 基数，表示反正弦定位器的基数
    subs: Sequence[float] | None  # 替换或未知，表示反正弦定位器的替换序列或未知
    # 初始化方法，用于设置对象的初始属性
    def __init__(
        self,
        linear_width: float,
        numticks: int = ...,
        symthresh: float = ...,
        base: int = ...,
        subs: Sequence[float] | None = ...,
    ) -> None:
        # 初始化方法的参数包括：
        # linear_width: 浮点数，线性宽度
        # numticks: 整数，刻度数目，默认为...
        # symthresh: 浮点数，符号阈值，默认为...
        # base: 整数，基数，默认为...
        # subs: 可选的浮点数序列或None，子集，默认为...
        ...

    # 设置参数的方法，用于修改对象的属性
    def set_params(
        self,
        numticks: int | None = ...,
        symthresh: float | None = ...,
        base: int | None = ...,
        subs: Sequence[float] | None = ...,
    ) -> None:
        # 设置参数的方法包括：
        # numticks: 可选的整数或None，刻度数目，默认为...
        # symthresh: 可选的浮点数或None，符号阈值，默认为...
        # base: 可选的整数或None，基数，默认为...
        # subs: 可选的浮点数序列或None，子集，默认为...
        ...
# 定义 LogitLocator 类，继承自 MaxNLocator 类
class LogitLocator(MaxNLocator):
    
    # 初始化方法
    def __init__(
        self, minor: bool = ..., *, nbins: Literal["auto"] | int = ...
    ) -> None:
        ...

    # 设置参数的方法
    def set_params(self, minor: bool | None = ..., **kwargs) -> None:
        ...

    # minor 属性的 getter 方法
    @property
    def minor(self) -> bool:
        ...

    # minor 属性的 setter 方法
    @minor.setter
    def minor(self, value: bool) -> None:
        ...

# 定义 AutoLocator 类，继承自 MaxNLocator 类
class AutoLocator(MaxNLocator):
    
    # 初始化方法
    def __init__(self) -> None:
        ...

# 定义 AutoMinorLocator 类，继承自 Locator 类
class AutoMinorLocator(Locator):
    
    # 类属性，表示划分的次数
    ndivs: int
    
    # 初始化方法
    def __init__(self, n: int | None = ...) -> None:
        ...
```