# `D:\src\scipysrc\matplotlib\lib\matplotlib\scale.pyi`

```
# 从 matplotlib.axis 模块导入 Axis 类
# 从 matplotlib.transforms 模块导入 Transform 类
from matplotlib.axis import Axis
from matplotlib.transforms import Transform

# 从 collections.abc 模块导入 Callable 和 Iterable 类型
from collections.abc import Callable, Iterable
# 从 typing 模块导入 Literal 类型
from typing import Literal
# 从 numpy.typing 模块导入 ArrayLike 类型
from numpy.typing import ArrayLike

# 定义一个 ScaleBase 类，表示比例尺基类
class ScaleBase:
    # 初始化方法，接受一个 Axis 对象或者 None
    def __init__(self, axis: Axis | None) -> None: ...
    # 返回一个 Transform 对象的方法
    def get_transform(self) -> Transform: ...
    # 设置默认的定位器和格式化器的方法，接受一个 Axis 对象
    def set_default_locators_and_formatters(self, axis: Axis) -> None: ...
    # 限制比例尺范围的方法，接受最小值、最大值和最小正数值，并返回两个浮点数元组
    def limit_range_for_scale(
        self, vmin: float, vmax: float, minpos: float
    ) -> tuple[float, float]: ...

# 定义一个 LinearScale 类，表示线性比例尺，继承自 ScaleBase 类
class LinearScale(ScaleBase):
    # 名称属性
    name: str

# 定义一个 FuncTransform 类，表示函数变换，继承自 Transform 类
class FuncTransform(Transform):
    # 输入维度和输出维度属性
    input_dims: int
    output_dims: int
    # 初始化方法，接受一个正向和反向的 Callable 数组
    def __init__(
        self,
        forward: Callable[[ArrayLike], ArrayLike],
        inverse: Callable[[ArrayLike], ArrayLike],
    ) -> None: ...
    # 返回一个反转的 FuncTransform 对象的方法
    def inverted(self) -> FuncTransform: ...

# 定义一个 FuncScale 类，表示函数比例尺，继承自 ScaleBase 类
class FuncScale(ScaleBase):
    # 名称属性
    name: str
    # 初始化方法，接受一个 Axis 对象或者 None 和一对正向和反向的 Callable 数组
    def __init__(
        self,
        axis: Axis | None,
        functions: tuple[
            Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]
        ],
    ) -> None: ...

# 定义一个 LogTransform 类，表示对数变换，继承自 Transform 类
class LogTransform(Transform):
    # 输入维度和输出维度属性
    input_dims: int
    output_dims: int
    # 初始化方法，接受一个基数和非正常数的处理方式（裁剪或遮罩）
    def __init__(
        self, base: float, nonpositive: Literal["clip", "mask"] = ...
    ) -> None: ...
    # 返回一个反转的 InvertedLogTransform 对象的方法
    def inverted(self) -> InvertedLogTransform: ...

# 定义一个 InvertedLogTransform 类，表示反转的对数变换，继承自 Transform 类
class InvertedLogTransform(Transform):
    # 输入维度和输出维度属性
    input_dims: int
    output_dims: int
    # 初始化方法，接受一个基数
    def __init__(self, base: float) -> None: ...
    # 返回一个 LogTransform 对象的方法
    def inverted(self) -> LogTransform: ...

# 定义一个 LogScale 类，表示对数比例尺，继承自 ScaleBase 类
class LogScale(ScaleBase):
    # 名称属性
    name: str
    # 可选的替代列表
    subs: Iterable[int] | None
    # 初始化方法，接受一个 Axis 对象或者 None，以及基数、替代列表和非正常数的处理方式
    def __init__(
        self,
        axis: Axis | None,
        *,
        base: float = ...,
        subs: Iterable[int] | None = ...,
        nonpositive: Literal["clip", "mask"] = ...
    ) -> None: ...
    # 返回基数属性的方法
    @property
    def base(self) -> float: ...
    # 返回一个 Transform 对象的方法
    def get_transform(self) -> Transform: ...

# 定义一个 FuncScaleLog 类，表示函数对数比例尺，继承自 LogScale 类
class FuncScaleLog(LogScale):
    # 初始化方法，接受一个 Axis 对象或者 None，一对正向和反向的 Callable 数组，以及一个基数
    def __init__(
        self,
        axis: Axis | None,
        functions: tuple[
            Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]
        ],
        base: float = ...,
    ) -> None: ...
    # 返回基数属性的方法
    @property
    def base(self) -> float: ...
    # 返回一个 Transform 对象的方法
    def get_transform(self) -> Transform: ...

# 定义一个 SymmetricalLogTransform 类，表示对称对数变换，继承自 Transform 类
class SymmetricalLogTransform(Transform):
    # 输入维度和输出维度属性
    input_dims: int
    output_dims: int
    # 初始化方法，接受一个基数、线性阈值和线性缩放
    def __init__(self, base: float, linthresh: float, linscale: float) -> None: ...
    # 返回一个反转的 InvertedSymmetricalLogTransform 对象的方法
    def inverted(self) -> InvertedSymmetricalLogTransform: ...

# 定义一个 InvertedSymmetricalLogTransform 类，表示反转的对称对数变换，继承自 Transform 类
class InvertedSymmetricalLogTransform(Transform):
    # 输入维度和输出维度属性
    input_dims: int
    output_dims: int
    # 初始化方法，接受一个基数、线性阈值和线性缩放
    def __init__(self, base: float, linthresh: float, linscale: float) -> None: ...
    # 返回一个 SymmetricalLogTransform 对象的方法
    def inverted(self) -> SymmetricalLogTransform: ...

# 定义一个 SymmetricalLogScale 类，表示对称对数比例尺，继承自 ScaleBase 类
class SymmetricalLogScale(ScaleBase):
    # 名称属性
    name: str
    # 可选的替代列表
    subs: Iterable[int] | None
    # 初始化方法，构造 SymmetricalLogScale 对象
    def __init__(
        self,
        axis: Axis | None,
        *,
        base: float = ...,
        linthresh: float = ...,
        subs: Iterable[int] | None = ...,
        linscale: float = ...
    ) -> None:
        # 父类的初始化方法，初始化坐标轴属性
        super().__init__(axis)
        # 设置对数转换的基数
        self._base = base
        # 设置对数转换的线性阈值
        self._linthresh = linthresh
        # 设置对数转换的缩放因子
        self._linscale = linscale
        # 设置对数刻度的整数子刻度数组
        self._subs = subs

    # 返回对数转换的基数
    @property
    def base(self) -> float:
        return self._base

    # 返回对数转换的线性阈值
    @property
    def linthresh(self) -> float:
        return self._linthresh

    # 返回对数转换的缩放因子
    @property
    def linscale(self) -> float:
        return self._linscale

    # 返回对数刻度的转换方法 SymmetricalLogTransform 对象
    def get_transform(self) -> SymmetricalLogTransform:
        # 创建并返回对数转换对象，传递基数、线性阈值和缩放因子作为参数
        return SymmetricalLogTransform(self._base, self._linthresh, self._linscale)
# 定义一个基于 asinh 函数的变换类，继承自 Transform 类
class AsinhTransform(Transform):
    # 输入维度
    input_dims: int
    # 输出维度
    output_dims: int
    # 线性宽度
    linear_width: float
    
    # 构造函数，初始化线性宽度
    def __init__(self, linear_width: float) -> None: ...

    # 返回反转后的逆 asinh 变换对象
    def inverted(self) -> InvertedAsinhTransform: ...

# 定义 asinh 变换的逆变换类，继承自 Transform 类
class InvertedAsinhTransform(Transform):
    # 输入维度
    input_dims: int
    # 输出维度
    output_dims: int
    # 线性宽度
    linear_width: float
    
    # 构造函数，初始化线性宽度
    def __init__(self, linear_width: float) -> None: ...

    # 返回反转后的 asinh 变换对象
    def inverted(self) -> AsinhTransform: ...

# 定义一个基于 asinh 缩放的比例尺类，继承自 ScaleBase 类
class AsinhScale(ScaleBase):
    # 名称
    name: str
    # 自动刻度倍数的字典
    auto_tick_multipliers: dict[int, tuple[int, ...]]
    
    # 构造函数，可以设置轴、线性宽度、基数和替代列表等参数
    def __init__(
        self,
        axis: Axis | None,
        *,
        linear_width: float = ...,
        base: float = ...,
        subs: Iterable[int] | Literal["auto"] | None = ...,
        **kwargs
    ) -> None: ...

    # 返回当前比例尺的线性宽度
    @property
    def linear_width(self) -> float: ...

    # 返回基于 asinh 变换的转换对象
    def get_transform(self) -> AsinhTransform: ...

# 定义一个基于 logit 函数的变换类，继承自 Transform 类
class LogitTransform(Transform):
    # 输入维度
    input_dims: int
    # 输出维度
    output_dims: int
    
    # 构造函数，根据需要选择非正值的处理方式（掩码或截断）
    def __init__(self, nonpositive: Literal["mask", "clip"] = ...) -> None: ...

    # 返回反转后的逆 logit 变换对象
    def inverted(self) -> LogisticTransform: ...

# 定义 logit 变换的逆变换类，继承自 Transform 类
class LogisticTransform(Transform):
    # 输入维度
    input_dims: int
    # 输出维度
    output_dims: int
    
    # 构造函数，根据需要选择非正值的处理方式（掩码或截断）
    def __init__(self, nonpositive: Literal["mask", "clip"] = ...) -> None: ...

    # 返回反转后的 logit 变换对象
    def inverted(self) -> LogitTransform: ...

# 定义一个基于 logit 缩放的比例尺类，继承自 ScaleBase 类
class LogitScale(ScaleBase):
    # 名称
    name: str
    
    # 构造函数，可以设置轴、非正值的处理方式、以及其他参数
    def __init__(
        self,
        axis: Axis | None,
        nonpositive: Literal["mask", "clip"] = ...,
        *,
        one_half: str = ...,
        use_overline: bool = ...
    ) -> None: ...

    # 返回基于 logit 变换的转换对象
    def get_transform(self) -> LogitTransform: ...

# 返回注册的所有比例尺名称的列表
def get_scale_names() -> list[str]: ...

# 根据比例尺名称和轴返回相应的比例尺对象实例
def scale_factory(scale: str, axis: Axis, **kwargs) -> ScaleBase: ...

# 注册指定的比例尺类到系统中
def register_scale(scale_class: type[ScaleBase]) -> None: ...
```