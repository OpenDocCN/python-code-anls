# `D:\src\scipysrc\scipy\scipy\stats\_unuran\unuran_wrapper.pyi`

```
from __future__ import annotations
# 引入用于支持类型提示中类型自身的注解，对未来版本的向后兼容性
import numpy as np
# 引入 NumPy 库
from typing import (overload, Callable, NamedTuple, Protocol)
# 引入类型提示相关的模块
import numpy.typing as npt
# 引入 NumPy 类型提示支持
from scipy._lib._util import SeedType
# 从 SciPy 库中引入随机种子类型
import scipy.stats as stats
# 引入 SciPy 统计模块

ArrayLike0D = bool | int | float | complex | str | bytes | np.generic
# 定义用于表示零维数组的类型别名

__all__: list[str]
# 声明一个字符串列表，用于模块中的公开接口

class UNURANError(RuntimeError):
    ...
    # UNURANError 类，继承自 RuntimeError，未定义额外行为

class Method:
    @overload
    def rvs(self, size: None = ...) -> float | int: ...
    # 方法重载：接受参数为 None，返回 float 或 int
    @overload
    def rvs(self, size: int | tuple[int, ...] = ...) -> np.ndarray: ...
    # 方法重载：接受参数为整数或整数元组，返回 NumPy 数组
    def set_random_state(self, random_state: SeedType) -> None: ...
    # 设置随机状态的方法

class TDRDist(Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...
    # 定义属性 pdf，返回一个可调用对象，接受任意参数，返回 float
    @property
    def dpdf(self) -> Callable[..., float]: ...
    # 定义属性 dpdf，返回一个可调用对象，接受任意参数，返回 float
    @property
    def support(self) -> tuple[float, float]: ...
    # 定义属性 support，返回支持范围的元组，包含两个 float 值

class TransformedDensityRejection(Method):
    def __init__(self,
                 dist: TDRDist,
                 *,
                 mode: None | float = ...,
                 center: None | float = ...,
                 domain: None | tuple[float, float] = ...,
                 c: float = ...,
                 construction_points: int | npt.ArrayLike = ...,
                 use_dars: bool = ...,
                 max_squeeze_hat_ratio: float = ...,
                 random_state: SeedType = ...) -> None: ...
    # 初始化方法，接受多个参数，包括 TDRDist 类型的 dist 参数

    @property
    def squeeze_hat_ratio(self) -> float: ...
    # 定义属性 squeeze_hat_ratio，返回一个 float 值
    @property
    def squeeze_area(self) -> float: ...
    # 定义属性 squeeze_area，返回一个 float 值

    @overload
    def ppf_hat(self, u: ArrayLike0D) -> float: ...
    # 方法重载：接受参数为 ArrayLike0D，返回 float
    @overload
    def ppf_hat(self, u: npt.ArrayLike) -> np.ndarray: ...
    # 方法重载：接受参数为 NumPy 数组，返回 NumPy 数组

class SROUDist(Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...
    # 定义属性 pdf，返回一个可调用对象，接受任意参数，返回 float
    @property
    def support(self) -> tuple[float, float]: ...
    # 定义属性 support，返回支持范围的元组，包含两个 float 值

class SimpleRatioUniforms(Method):
    def __init__(self,
                 dist: SROUDist,
                 *,
                 mode: None | float = ...,
                 pdf_area: float = ...,
                 domain: None | tuple[float, float] = ...,
                 cdf_at_mode: float = ...,
                 random_state: SeedType = ...) -> None: ...
    # 初始化方法，接受多个参数，包括 SROUDist 类型的 dist 参数

class UError(NamedTuple):
    max_error: float
    mean_absolute_error: float
    # 命名元组 UError，包含 max_error 和 mean_absolute_error 两个 float 属性

class PINVDist(Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...
    # 定义属性 pdf，返回一个可调用对象，接受任意参数，返回 float
    @property
    def cdf(self) -> Callable[..., float]: ...
    # 定义属性 cdf，返回一个可调用对象，接受任意参数，返回 float
    @property
    def logpdf(self) -> Callable[..., float]: ...
    # 定义属性 logpdf，返回一个可调用对象，接受任意参数，返回 float

class NumericalInversePolynomial(Method):
    def __init__(self,
                 dist: PINVDist,
                 *,
                 mode: None | float = ...,
                 center: None | float = ...,
                 domain: None | tuple[float, float] = ...,
                 order: int = ...,
                 u_resolution: float = ...,
                 random_state: SeedType = ...) -> None: ...
    # 初始化方法，接受多个参数，包括 PINVDist 类型的 dist 参数

    @property
    def intervals(self) -> int: ...
    # 定义属性 intervals，返回一个整数值

    @overload
    def ppf(self, u: ArrayLike0D) -> float: ...
    # 方法重载：接受参数为 ArrayLike0D，返回 float
    # 定义函数ppf的类型重载，接受npt.ArrayLike类型参数并返回np.ndarray类型结果
    @overload
    def ppf(self, u: npt.ArrayLike) -> np.ndarray: ...

    # 定义函数cdf的类型重载，接受ArrayLike0D类型参数并返回float类型结果，
    # 并且声明忽略重载重叠错误
    @overload
    def cdf(self, x: ArrayLike0D) -> float: ...  # type: ignore[overload-overlap]

    # 定义函数cdf的类型重载，接受npt.ArrayLike类型参数并返回np.ndarray类型结果
    @overload
    def cdf(self, x: npt.ArrayLike) -> np.ndarray: ...

    # 定义函数u_error，接受一个整数类型的可选参数sample_size，
    # 并返回类型为UError的结果对象
    def u_error(self, sample_size: int = ...) -> UError: ...

    # 定义函数qrvs，接受多种形式的参数：None、整数、元组或指定类型的qmc_engine，
    # 返回类型为npt.ArrayLike的结果
    def qrvs(self,
             size: None | int | tuple[int, ...] = ...,
             d: None | int = ...,
             qmc_engine: None | stats.qmc.QMCEngine = ...) -> npt.ArrayLike: ...
# 定义一个协议 HINVDist，包含用于概率密度函数、累积分布函数和支持区间的属性
class HINVDist(Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...
    @property
    def cdf(self) -> Callable[..., float]: ...
    @property
    def support(self) -> tuple[float, float]: ...


# 数值逆 Hermite 方法的类，继承自 Method 类
class NumericalInverseHermite(Method):
    def __init__(self,
                 dist: HINVDist,
                 *,
                 domain: None | tuple[float, float] = ...,
                 order: int= ...,
                 u_resolution: float = ...,
                 construction_points: None | npt.ArrayLike = ...,
                 max_intervals: int = ...,
                 random_state: SeedType = ...) -> None: ...
    @property
    def intervals(self) -> int: ...
    # 重载方法，计算给定值 u 的分位点函数，返回单个浮点数
    @overload
    def ppf(self, u: ArrayLike0D) -> float: ...  # type: ignore[overload-overlap]
    # 重载方法，计算给定值 u 的分位点函数，返回 NumPy 数组
    @overload
    def ppf(self, u: npt.ArrayLike) -> np.ndarray: ...
    # 生成指定尺寸的随机变量，返回 NumPy 类型数组
    def qrvs(self,
             size: None | int | tuple[int, ...] = ...,
             d: None | int = ...,
             qmc_engine: None | stats.qmc.QMCEngine = ...) -> npt.ArrayLike: ...
    # 计算给定样本大小的 u 误差，返回 UError 对象
    def u_error(self, sample_size: int = ...) -> UError: ...


# 定义一个协议 DAUDist，包含用于概率质量函数和支持区间的属性
class DAUDist(Protocol):
    @property
    def pmf(self) -> Callable[..., float]: ...
    @property
    def support(self) -> tuple[float, float]: ...


# 离散 Alias Urn 方法的类，继承自 Method 类
class DiscreteAliasUrn(Method):
    def __init__(self,
                 dist: npt.ArrayLike | DAUDist,
                 *,
                 domain: None | tuple[float, float] = ...,
                 urn_factor: float = ...,
                 random_state: SeedType = ...) -> None: ...


# 定义一个协议 DGTDist，包含用于概率质量函数和支持区间的属性
class DGTDist(Protocol):
    @property
    def pmf(self) -> Callable[..., float]: ...
    @property
    def support(self) -> tuple[float, float]: ...


# 离散 Guide Table 方法的类，继承自 Method 类
class DiscreteGuideTable(Method):
    def __init__(self,
                 dist: npt.ArrayLike | DGTDist,
                 *,
                 domain: None | tuple[float, float] = ...,
                 guide_factor: float = ...,
                 random_state: SeedType = ...) -> None: ...
    # 重载方法，计算给定值 u 的分位点函数，返回单个浮点数
    @overload
    def ppf(self, u: ArrayLike0D) -> float: ...  # type: ignore[overload-overlap]
    # 重载方法，计算给定值 u 的分位点函数，返回 NumPy 数组
    @overload
    def ppf(self, u: npt.ArrayLike) -> np.ndarray: ...
```