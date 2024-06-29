# `D:\src\scipysrc\matplotlib\lib\matplotlib\colors.pyi`

```
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from matplotlib import cbook, scale
import re

from typing import Any, Literal, overload
from .typing import ColorType

import numpy as np
from numpy.typing import ArrayLike

# 明确导出颜色字典，这些字典在实现中被导入
BASE_COLORS: dict[str, ColorType]
CSS4_COLORS: dict[str, ColorType]
TABLEAU_COLORS: dict[str, ColorType]
XKCD_COLORS: dict[str, ColorType]

class _ColorMapping(dict[str, ColorType]):
    # 缓存颜色映射到 RGBA 值的结果，使用元组作为键，元组包含颜色和可选的透明度
    cache: dict[tuple[ColorType, float | None], tuple[float, float, float, float]]
    def __init__(self, mapping) -> None: ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...

def get_named_colors_mapping() -> _ColorMapping:
    # 返回一个特定的颜色映射对象
    ...

class ColorSequenceRegistry(Mapping):
    # 注册和管理颜色序列的注册表
    def __init__(self) -> None: ...
    def __getitem__(self, item: str) -> list[ColorType]: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def register(self, name: str, color_list: Iterable[ColorType]) -> None: ...
    def unregister(self, name: str) -> None: ...

_color_sequences: ColorSequenceRegistry = ...

def is_color_like(c: Any) -> bool:
    # 判断输入是否类似颜色
    ...

def same_color(c1: ColorType, c2: ColorType) -> bool:
    # 判断两个颜色是否相同
    ...

def to_rgba(
    c: ColorType, alpha: float | None = ...
) -> tuple[float, float, float, float]:
    # 将颜色转换为 RGBA 四元组，可以选择指定透明度
    ...

def to_rgba_array(
    c: ColorType | ArrayLike, alpha: float | ArrayLike | None = ...
) -> np.ndarray:
    # 将颜色或颜色数组转换为 RGBA 数组，可以选择指定透明度
    ...

def to_rgb(c: ColorType) -> tuple[float, float, float]:
    # 将颜色转换为 RGB 三元组
    ...

def to_hex(c: ColorType, keep_alpha: bool = ...) -> str:
    # 将颜色转换为十六进制表示，可以选择是否保留透明度信息
    ...

cnames: dict[str, ColorType]
hexColorPattern: re.Pattern
rgb2hex = to_hex
hex2color = to_rgb

class ColorConverter:
    # 颜色转换器，提供颜色转换的静态方法
    colors: _ColorMapping
    cache: dict[tuple[ColorType, float | None], tuple[float, float, float, float]]
    @staticmethod
    def to_rgb(c: ColorType) -> tuple[float, float, float]:
        # 将颜色转换为 RGB 三元组
        ...
    @staticmethod
    def to_rgba(
        c: ColorType, alpha: float | None = ...
    ) -> tuple[float, float, float, float]:
        # 将颜色转换为 RGBA 四元组，可以选择指定透明度
        ...
    @staticmethod
    def to_rgba_array(
        c: ColorType | ArrayLike, alpha: float | ArrayLike | None = ...
    ) -> np.ndarray:
        # 将颜色或颜色数组转换为 RGBA 数组，可以选择指定透明度
        ...

colorConverter: ColorConverter

class Colormap:
    # 色彩映射对象，用于将数据映射为颜色
    name: str
    N: int
    colorbar_extend: bool
    def __init__(self, name: str, N: int = ...) -> None:
        # 初始化色彩映射对象，指定名称和可选的数据点数目
        ...
    @overload
    def __call__(
        self, X: Sequence[float] | np.ndarray, alpha: ArrayLike | None = ..., bytes: bool = ...
    ) -> np.ndarray:
        # 根据数据 X 返回颜色数组，可以选择指定透明度和输出格式
        ...
    @overload
    def __call__(
        self, X: float, alpha: float | None = ..., bytes: bool = ...
    ) -> tuple[float, float, float, float]:
        # 根据数据 X 返回单个颜色，可以选择指定透明度和输出格式
        ...
    @overload
    def __call__(
        self, X: ArrayLike, alpha: ArrayLike | None = ..., bytes: bool = ...
    ) -> tuple[float, float, float, float] | np.ndarray:
        # 根据数据 X 返回颜色数组或单个颜色，可以选择指定透明度和输出格式
        ...
    def __copy__(self) -> Colormap:
        # 复制当前色彩映射对象
        ...
    def __eq__(self, other: object) -> bool:
        # 判断当前色彩映射对象是否等于另一个对象
        ...
    def get_bad(self) -> np.ndarray:
        # 返回定义的坏数据颜色数组
        ...
    # 设置颜色映射的“bad”颜色和透明度属性
    def set_bad(self, color: ColorType = ..., alpha: float | None = ...) -> None: ...
    
    # 返回颜色映射中的“under”颜色数组
    def get_under(self) -> np.ndarray: ...
    
    # 设置颜色映射的“under”颜色和透明度属性
    def set_under(self, color: ColorType = ..., alpha: float | None = ...) -> None: ...
    
    # 返回颜色映射中的“over”颜色数组
    def get_over(self) -> np.ndarray: ...
    
    # 设置颜色映射的“over”颜色和透明度属性
    def set_over(self, color: ColorType = ..., alpha: float | None = ...) -> None: ...
    
    # 设置颜色映射的“bad”、“under”和“over”颜色属性
    def set_extremes(
        self,
        *,
        bad: ColorType | None = ...,
        under: ColorType | None = ...,
        over: ColorType | None = ...
    ) -> None: ...
    
    # 创建一个带有新的“bad”、“under”和“over”颜色属性的颜色映射对象
    def with_extremes(
        self,
        *,
        bad: ColorType | None = ...,
        under: ColorType | None = ...,
        over: ColorType | None = ...
    ) -> Colormap: ...
    
    # 检查颜色映射是否为灰度颜色映射
    def is_gray(self) -> bool: ...
    
    # 返回具有给定LUT大小的重采样后的颜色映射对象
    def resampled(self, lutsize: int) -> Colormap: ...
    
    # 返回颠倒了颜色映射的名称（如果有）的新颜色映射对象
    def reversed(self, name: str | None = ...) -> Colormap: ...
    
    # 返回颜色映射对象的HTML表示形式
    def _repr_html_(self) -> str: ...
    
    # 返回颜色映射对象的PNG表示形式的字节数据
    def _repr_png_(self) -> bytes: ...
    
    # 返回颜色映射对象的浅拷贝
    def copy(self) -> Colormap: ...
class LinearSegmentedColormap(Colormap):
    monochrome: bool
    # 定义线性分段颜色映射对象
    def __init__(
        self,
        name: str,
        segmentdata: dict[
            Literal["red", "green", "blue", "alpha"], Sequence[tuple[float, ...]]
        ],
        N: int = ...,
        gamma: float = ...,
    ) -> None: ...
    # 设置 gamma 值的方法
    def set_gamma(self, gamma: float) -> None: ...
    # 从颜色列表创建线性分段颜色映射对象的静态方法
    @staticmethod
    def from_list(
        name: str, colors: ArrayLike | Sequence[tuple[float, ColorType]], N: int = ..., gamma: float = ...
    ) -> LinearSegmentedColormap: ...
    # 对颜色映射对象进行重新采样的方法
    def resampled(self, lutsize: int) -> LinearSegmentedColormap: ...
    # 返回颜色映射对象的反转版本的方法
    def reversed(self, name: str | None = ...) -> LinearSegmentedColormap: ...


class ListedColormap(Colormap):
    monochrome: bool
    colors: ArrayLike | ColorType
    # 定义列表颜色映射对象
    def __init__(
        self, colors: ArrayLike | ColorType, name: str = ..., N: int | None = ...
    ) -> None: ...
    # 对列表颜色映射对象进行重新采样的方法
    def resampled(self, lutsize: int) -> ListedColormap: ...
    # 返回颜色映射对象的反转版本的方法
    def reversed(self, name: str | None = ...) -> ListedColormap: ...


class Normalize:
    callbacks: cbook.CallbackRegistry
    # 标准化对象初始化方法
    def __init__(
        self, vmin: float | None = ..., vmax: float | None = ..., clip: bool = ...
    ) -> None: ...
    # 最小值的属性访问方法
    @property
    def vmin(self) -> float | None: ...
    # 最小值的属性设置方法
    @vmin.setter
    def vmin(self, value: float | None) -> None: ...
    # 最大值的属性访问方法
    @property
    def vmax(self) -> float | None: ...
    # 最大值的属性设置方法
    @vmax.setter
    def vmax(self, value: float | None) -> None: ...
    # 是否剪切的属性访问方法
    @property
    def clip(self) -> bool: ...
    # 是否剪切的属性设置方法
    @clip.setter
    def clip(self, value: bool) -> None: ...
    # 处理值的静态方法
    @staticmethod
    def process_value(value: ArrayLike) -> tuple[np.ma.MaskedArray, bool]: ...
    # 根据值进行标准化的调用方法的重载
    @overload
    def __call__(self, value: float, clip: bool | None = ...) -> float: ...
    @overload
    def __call__(self, value: np.ndarray, clip: bool | None = ...) -> np.ma.MaskedArray: ...
    @overload
    def __call__(self, value: ArrayLike, clip: bool | None = ...) -> ArrayLike: ...
    # 根据值进行反标准化的方法的重载
    @overload
    def inverse(self, value: float) -> float: ...
    @overload
    def inverse(self, value: np.ndarray) -> np.ma.MaskedArray: ...
    @overload
    def inverse(self, value: ArrayLike) -> ArrayLike: ...
    # 自动缩放到给定数组的方法
    def autoscale(self, A: ArrayLike) -> None: ...
    # 自动缩放空的方法
    def autoscale_None(self, A: ArrayLike) -> None: ...
    # 是否已经标准化的属性访问方法
    def scaled(self) -> bool: ...


class TwoSlopeNorm(Normalize):
    # 双斜坡标准化对象初始化方法
    def __init__(
        self, vcenter: float, vmin: float | None = ..., vmax: float | None = ...
    ) -> None: ...
    # 中心值的属性访问方法
    @property
    def vcenter(self) -> float: ...
    # 中心值的属性设置方法
    @vcenter.setter
    def vcenter(self, value: float) -> None: ...
    # 自动缩放空的方法
    def autoscale_None(self, A: ArrayLike) -> None: ...


class CenteredNorm(Normalize):
    # 居中标准化对象初始化方法
    def __init__(
        self, vcenter: float = ..., halfrange: float | None = ..., clip: bool = ...
    ) -> None: ...
    # 中心值的属性访问方法
    @property
    def vcenter(self) -> float: ...
    # 中心值的属性设置方法
    @vcenter.setter
    def vcenter(self, vcenter: float) -> None: ...
    # 范围的一半的属性访问方法
    @property
    def halfrange(self) -> float: ...
    # 范围的一半的属性设置方法
    @halfrange.setter
    def halfrange(self, halfrange: float) -> None: ...
    # 定义一个方法 halfrange，该方法属于某个类的成员方法，self 表示该类的实例本身
    def halfrange(self, halfrange: float) -> None:
        # 这是一个占位符方法，暂时没有实现具体功能，参数 halfrange 是一个浮点数类型，方法返回 None
        ...
@overload
def make_norm_from_scale(
    scale_cls: type[scale.ScaleBase],
    base_norm_cls: type[Normalize],
    *,
    init: Callable | None = ...
) -> type[Normalize]: ...

# 声明函数重载，用于根据给定的比例尺类和基础规范类创建一个新的规范类。返回类型为规范类的类型。


@overload
def make_norm_from_scale(
    scale_cls: type[scale.ScaleBase],
    base_norm_cls: None = ...,
    *,
    init: Callable | None = ...
) -> Callable[[type[Normalize]], type[Normalize]]: ...

# 另一个函数重载声明，用于根据比例尺类和可选的基础规范类或初始化函数创建一个规范类。返回类型为接受规范类作为参数并返回规范类的函数类型。


class FuncNorm(Normalize):
    def __init__(
            self,
            functions: tuple[Callable, Callable],
            vmin: float | None = ...,
            vmax: float | None = ...,
            clip: bool = ...,
    ) -> None: ...

# 定义一个名为FuncNorm的类，继承自Normalize，用于根据给定的函数对数据进行规范化。初始化函数接受一对函数、可选的最小值和最大值限制以及剪裁标志。


class LogNorm(Normalize): ...

# 定义一个名为LogNorm的类，继承自Normalize，用于对数据进行对数规范化。


class SymLogNorm(Normalize):
    def __init__(
            self,
            linthresh: float,
            linscale: float = ...,
            vmin: float | None = ...,
            vmax: float | None = ...,
            clip: bool = ...,
            *,
            base: float = ...,
    ) -> None: ...
    @property
    def linthresh(self) -> float: ...
    @linthresh.setter
    def linthresh(self, value: float) -> None: ...

# 定义一个名为SymLogNorm的类，继承自Normalize，用于对数据进行对称对数规范化。初始化函数接受线性阈值、线性比例、可选的最小值和最大值限制、剪裁标志以及基数。linthresh属性和setter方法用于获取和设置线性阈值。


class AsinhNorm(Normalize):
    def __init__(
        self,
        linear_width: float = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        clip: bool = ...,
    ) -> None: ...
    @property
    def linear_width(self) -> float: ...
    @linear_width.setter
    def linear_width(self, value: float) -> None: ...

# 定义一个名为AsinhNorm的类，继承自Normalize，用于对数据进行反正弦规范化。初始化函数接受线性宽度、可选的最小值和最大值限制以及剪裁标志。linear_width属性和setter方法用于获取和设置线性宽度。


class PowerNorm(Normalize):
    gamma: float
    def __init__(
        self,
        gamma: float,
        vmin: float | None = ...,
        vmax: float | None = ...,
        clip: bool = ...,
    ) -> None: ...

# 定义一个名为PowerNorm的类，继承自Normalize，用于对数据进行幂规范化。初始化函数接受幂值gamma、可选的最小值和最大值限制以及剪裁标志。


class BoundaryNorm(Normalize):
    boundaries: np.ndarray
    N: int
    Ncmap: int
    extend: Literal["neither", "both", "min", "max"]
    def __init__(
        self,
        boundaries: ArrayLike,
        ncolors: int,
        clip: bool = ...,
        *,
        extend: Literal["neither", "both", "min", "max"] = ...
    ) -> None: ...

# 定义一个名为BoundaryNorm的类，继承自Normalize，用于根据边界值和颜色数创建数据规范化对象。初始化函数接受边界数组、颜色数、剪裁标志以及扩展类型（枚举值为"neither", "both", "min", "max"）。


class NoNorm(Normalize): ...

# 定义一个名为NoNorm的类，继承自Normalize，表示无需对数据进行规范化。


def rgb_to_hsv(arr: ArrayLike) -> np.ndarray: ...

# 定义函数rgb_to_hsv，接受一个数组（类似数组的类型）作为输入，返回一个NumPy数组，表示RGB到HSV颜色空间的转换结果。


def hsv_to_rgb(hsv: ArrayLike) -> np.ndarray: ...

# 定义函数hsv_to_rgb，接受一个数组（类似数组的类型）作为输入，返回一个NumPy数组，表示HSV到RGB颜色空间的转换结果。


class LightSource:
    azdeg: float
    altdeg: float
    hsv_min_val: float
    hsv_max_val: float
    hsv_min_sat: float
    hsv_max_sat: float
    def __init__(
        self,
        azdeg: float = ...,
        altdeg: float = ...,
        hsv_min_val: float = ...,
        hsv_max_val: float = ...,
        hsv_min_sat: float = ...,
        hsv_max_sat: float = ...,
    ) -> None: ...
    @property
    def direction(self) -> np.ndarray: ...
    def hillshade(
        self,
        elevation: ArrayLike,
        vert_exag: float = ...,
        dx: float = ...,
        dy: float = ...,
        fraction: float = ...,
    ) -> np.ndarray: ...
    def shade_normals(
        self, normals: np.ndarray, fraction: float = ...
    ) -> np.ndarray: ...

# 定义名为LightSource的类，表示光源对象，用于地形遮蔽和法线着色。初始化函数接受方位角、高度角、HSV颜色空间的最小值和最大值、饱和度的最小值和最大值。direction属性返回光源的方向向量，hillshade方法根据高程数据生成地形遮蔽效果，shade_normals方法根据法线数组生成阴影效果。
    # 对象方法：在给定数据上应用阴影效果，并返回结果数组
    def shade(
        self,
        data: ArrayLike,  # 输入数据，可以是类数组的结构
        cmap: Colormap,   # 颜色映射对象，用于将数据映射到颜色
        norm: Normalize | None = ...,  # 数据归一化对象，可选参数，默认为None
        blend_mode: Literal["hsv", "overlay", "soft"] | Callable = ...,  # 混合模式，可以是预设字符串或自定义函数
        vmin: float | None = ...,  # 数据的最小值，可选参数，默认为None
        vmax: float | None = ...,  # 数据的最大值，可选参数，默认为None
        vert_exag: float = ...,  # 垂直方向的放大系数，默认值为...
        dx: float = ...,   # X轴方向的步进值，默认值为...
        dy: float = ...,   # Y轴方向的步进值，默认值为...
        fraction: float = ...,  # 混合的分数或程度，默认值为...
        **kwargs  # 其他可选参数，用于接收额外的关键字参数
    ) -> np.ndarray:  # 返回一个NumPy数组作为阴影处理后的结果
        ...
    
    # 对象方法：对RGB数据应用阴影效果，并返回结果数组
    def shade_rgb(
        self,
        rgb: ArrayLike,  # RGB颜色数据，类数组结构
        elevation: ArrayLike,  # 高程或强度数据，类数组结构
        fraction: float = ...,  # 混合的分数或程度，默认值为...
        blend_mode: Literal["hsv", "overlay", "soft"] | Callable = ...,  # 混合模式，可以是预设字符串或自定义函数
        vert_exag: float = ...,  # 垂直方向的放大系数，默认值为...
        dx: float = ...,   # X轴方向的步进值，默认值为...
        dy: float = ...,   # Y轴方向的步进值，默认值为...
        **kwargs  # 其他可选参数，用于接收额外的关键字参数
    ) -> np.ndarray:  # 返回一个NumPy数组作为阴影处理后的结果
        ...
    
    # 对象方法：使用HSV颜色空间进行混合处理
    def blend_hsv(
        self,
        rgb: ArrayLike,  # RGB颜色数据，类数组结构
        intensity: ArrayLike,  # 强度或值数据，类数组结构
        hsv_max_sat: float | None = ...,  # HSV颜色空间的最大饱和度，可选参数，默认为None
        hsv_max_val: float | None = ...,  # HSV颜色空间的最大值，可选参数，默认为None
        hsv_min_val: float | None = ...,  # HSV颜色空间的最小值，可选参数，默认为None
        hsv_min_sat: float | None = ...,  # HSV颜色空间的最小饱和度，可选参数，默认为None
    ) -> ArrayLike:  # 返回一个数组，代表HSV混合后的结果
        ...
    
    # 对象方法：使用软光混合模式对RGB数据进行处理
    def blend_soft_light(
        self, rgb: np.ndarray, intensity: np.ndarray
    ) -> np.ndarray:  # 返回一个NumPy数组，代表软光混合后的结果
        ...
    
    # 对象方法：使用叠加混合模式对RGB数据进行处理
    def blend_overlay(self, rgb: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        ...  # 返回一个NumPy数组，代表叠加混合后的结果
# 定义一个函数，接受以下参数：
# levels: 一个浮点数序列，用于定义数据的分级阈值
# colors: 一个颜色类型的序列，定义与每个分级阈值相关联的颜色
# extend: 可选参数，字面值类型，指定颜色映射的扩展方式，可以是"neither"、"min"、"max"或"both"
# 返回一个元组，包含两个对象：
#   - ListedColormap: 表示颜色映射的列出的颜色映射对象
#   - BoundaryNorm: 表示归一化分级值到颜色映射索引的边界归一化对象
def from_levels_and_colors(
    levels: Sequence[float],
    colors: Sequence[ColorType],
    extend: Literal["neither", "min", "max", "both"] = ...,
) -> tuple[ListedColormap, BoundaryNorm]: ...
```