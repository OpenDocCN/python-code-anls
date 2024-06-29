# `D:\src\scipysrc\numpy\numpy\random\_generator.pyi`

```
# 导入必要的模块和类
from collections.abc import Callable
from typing import Any, overload, TypeVar, Literal

# 导入 NumPy 库及其子模块
import numpy as np
from numpy import (
    dtype,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    int_,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
)
from numpy.random import BitGenerator, SeedSequence
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _DoubleCodes,
    _DTypeLikeBool,
    _DTypeLikeInt,
    _DTypeLikeUInt,
    _Float32Codes,
    _Float64Codes,
    _FloatLike_co,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _IntCodes,
    _ShapeLike,
    _SingleCodes,
    _SupportsDType,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UIntCodes,
)

# 定义一个类型变量用于 NumPy 数组
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

# 定义支持 float32 类型的多种类型
_DTypeLikeFloat32 = (
    dtype[float32]
    | _SupportsDType[dtype[float32]]
    | type[float32]
    | _Float32Codes
    | _SingleCodes
)

# 定义支持 float64 类型的多种类型
_DTypeLikeFloat64 = (
    dtype[float64]
    | _SupportsDType[dtype[float64]]
    | type[float]
    | type[float64]
    | _Float64Codes
    | _DoubleCodes
)

# 定义 Generator 类，用于生成随机数
class Generator:
    # 初始化方法，接受一个比特生成器对象作为参数
    def __init__(self, bit_generator: BitGenerator) -> None: ...
    
    # 返回对象的字符串表示形式
    def __repr__(self) -> str: ...
    
    # 返回对象的字符串表示形式
    def __str__(self) -> str: ...
    
    # 获取对象的序列化状态
    def __getstate__(self) -> None: ...
    
    # 设置对象的序列化状态
    def __setstate__(self, state: dict[str, Any] | None) -> None: ...
    
    # 用于 pickle 模块序列化和反序列化的特殊方法
    def __reduce__(self) -> tuple[
        Callable[[BitGenerator], Generator],
        tuple[BitGenerator],
        None]: ...
    
    # 返回比特生成器对象
    @property
    def bit_generator(self) -> BitGenerator: ...
    
    # 生成指定数量的 Generator 对象列表
    def spawn(self, n_children: int) -> list[Generator]: ...
    
    # 生成指定长度的随机字节串
    def bytes(self, length: int) -> bytes: ...
    
    # 生成标准正态分布的随机数，支持多种重载形式
    @overload
    def standard_normal(
        self,
        size: None = ...,
        dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64 = ...,
        out: None = ...,
    ) -> float: ...
    @overload
    def standard_normal(
        self,
        size: _ShapeLike = ...,
    ) -> NDArray[float64]: ...
    @overload
    def standard_normal(
        self,
        *,
        out: NDArray[float64] = ...,
    ) -> NDArray[float64]: ...
    @overload
    def standard_normal(
        self,
        size: _ShapeLike = ...,
        dtype: _DTypeLikeFloat32 = ...,
        out: None | NDArray[float32] = ...,
    ) -> NDArray[float32]: ...
    @overload
    def standard_normal(
        self,
        size: _ShapeLike = ...,
        dtype: _DTypeLikeFloat64 = ...,
        out: None | NDArray[float64] = ...,
    ) -> NDArray[float64]: ...
    
    # 返回数组 x 的随机排列，支持多种重载形式
    @overload
    def permutation(self, x: int, axis: int = ...) -> NDArray[int64]: ...
    @overload
    def permutation(self, x: ArrayLike, axis: int = ...) -> NDArray[Any]: ...
    
    # 更多的 permutation 方法重载定义省略
    def standard_exponential(  # type: ignore[misc]
        self,
        size: None = ...,
        dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64 = ...,
        method: Literal["zig", "inv"] = ...,
        out: None = ...,
    ) -> float:
    ...
    @overload
    def standard_exponential(
        self,
        size: _ShapeLike = ...,
    ) -> NDArray[float64]:
    ...
    @overload
    def standard_exponential(
        self,
        *,
        out: NDArray[float64] = ...,
    ) -> NDArray[float64]:
    ...
    @overload
    def standard_exponential(
        self,
        size: _ShapeLike = ...,
        *,
        method: Literal["zig", "inv"] = ...,
        out: None | NDArray[float64] = ...,
    ) -> NDArray[float64]:
    ...
    @overload
    def standard_exponential(
        self,
        size: _ShapeLike = ...,
        dtype: _DTypeLikeFloat32 = ...,
        method: Literal["zig", "inv"] = ...,
        out: None | NDArray[float32] = ...,
    ) -> NDArray[float32]:
    ...
    @overload
    def standard_exponential(
        self,
        size: _ShapeLike = ...,
        dtype: _DTypeLikeFloat64 = ...,
        method: Literal["zig", "inv"] = ...,
        out: None | NDArray[float64] = ...,
    ) -> NDArray[float64]:
    ...
    @overload
    def random(  # type: ignore[misc]
        self,
        size: None = ...,
        dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64 = ...,
        out: None = ...,
    ) -> float:
    ...
    @overload
    def random(
        self,
        *,
        out: NDArray[float64] = ...,
    ) -> NDArray[float64]:
    ...
    @overload
    def random(
        self,
        size: _ShapeLike = ...,
        *,
        out: None | NDArray[float64] = ...,
    ) -> NDArray[float64]:
    ...
    @overload
    def random(
        self,
        size: _ShapeLike = ...,
        dtype: _DTypeLikeFloat32 = ...,
        out: None | NDArray[float32] = ...,
    ) -> NDArray[float32]:
    ...
    @overload
    def random(
        self,
        size: _ShapeLike = ...,
        dtype: _DTypeLikeFloat64 = ...,
        out: None | NDArray[float64] = ...,
    ) -> NDArray[float64]:
    ...
    @overload
    def beta(
        self,
        a: _FloatLike_co,
        b: _FloatLike_co,
        size: None = ...,
    ) -> float:  # type: ignore[misc]
    ...
    @overload
    def beta(
        self, a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]:
    ...
    @overload
    def exponential(self, scale: _FloatLike_co = ..., size: None = ...) -> float:  # type: ignore[misc]
    ...
    @overload
    def exponential(
        self, scale: _ArrayLikeFloat_co = ..., size: None | _ShapeLike = ...
    ) -> NDArray[float64]:
    ...
    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
    ) -> int:
    ...
    @overload



# 注释：
定义了一系列重载函数 `standard_exponential`, `random`, `beta`, `exponential`, `integers`，用于生成不同分布（指数分布、随机分布、贝塔分布、指数分布和整数）的随机数或随机数组。每个函数支持多个参数组合，返回不同的类型和形状的随机数或数组。
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: type[bool] = ...,
        endpoint: bool = ...,
    ) -> bool: ...

# 定义 `integers` 方法，用于生成整数序列
# - `low`：最低值，生成的整数序列的下限
# - `high`：最高值，生成的整数序列的上限（可选）
# - `size`：生成整数序列的大小（可选）
# - `dtype`：生成整数序列的数据类型，默认为布尔类型（可选）
# - `endpoint`：如果为 True，生成的整数序列包括 `high` 值；如果为 False，则不包括 `high` 值（可选）
# - 返回类型为布尔类型的整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: type[np.bool] = ...,
        endpoint: bool = ...,
    ) -> np.bool: ...

# 定义 `integers` 方法的重载，用于生成布尔类型的整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `np.bool`
# - 返回类型为 NumPy 中的布尔类型的整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: type[int] = ...,
        endpoint: bool = ...,
    ) -> int: ...

# 定义 `integers` 方法的重载，用于生成普通整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为整数类型
# - 返回类型为普通整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint8] | type[uint8] | _UInt8Codes | _SupportsDType[dtype[uint8]] = ...,
        endpoint: bool = ...,
    ) -> uint8: ...

# 定义 `integers` 方法的重载，用于生成无符号8位整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `uint8`
# - 返回类型为无符号8位整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint16] | type[uint16] | _UInt16Codes | _SupportsDType[dtype[uint16]] = ...,
        endpoint: bool = ...,
    ) -> uint16: ...

# 定义 `integers` 方法的重载，用于生成无符号16位整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `uint16`
# - 返回类型为无符号16位整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint32] | type[uint32] | _UInt32Codes | _SupportsDType[dtype[uint32]] = ...,
        endpoint: bool = ...,
    ) -> uint32: ...

# 定义 `integers` 方法的重载，用于生成无符号32位整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `uint32`
# - 返回类型为无符号32位整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint] | type[uint] | _UIntCodes | _SupportsDType[dtype[uint]] = ...,
        endpoint: bool = ...,
    ) -> uint: ...

# 定义 `integers` 方法的重载，用于生成无符号整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `uint`
# - 返回类型为无符号整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint64] | type[uint64] | _UInt64Codes | _SupportsDType[dtype[uint64]] = ...,
        endpoint: bool = ...,
    ) -> uint64: ...

# 定义 `integers` 方法的重载，用于生成无符号64位整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `uint64`
# - 返回类型为无符号64位整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int8] | type[int8] | _Int8Codes | _SupportsDType[dtype[int8]] = ...,
        endpoint: bool = ...,
    ) -> int8: ...

# 定义 `integers` 方法的重载，用于生成有符号8位整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `int8`
# - 返回类型为有符号8位整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int16] | type[int16] | _Int16Codes | _SupportsDType[dtype[int16]] = ...,
        endpoint: bool = ...,
    ) -> int16: ...

# 定义 `integers` 方法的重载，用于生成有符号16位整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `int16`
# - 返回类型为有符号16位整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int32] | type[int32] | _Int32Codes | _SupportsDType[dtype[int32]] = ...,
        endpoint: bool = ...,
    ) -> int32: ...

# 定义 `integers` 方法的重载，用于生成有符号32位整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `int32`
# - 返回类型为有符号32位整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int] | type[int] | _IntCodes | _SupportsDType[dtype[int]] = ...,
        endpoint: bool = ...,
    ) -> int: ...

# 定义 `integers` 方法的重载，用于生成普通有符号整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为普通整数类型
# - 返回类型为普通有符号整数序列


    @overload
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int64] | type[int64] | _Int64Codes | _SupportsDType[dtype[int64]] = ...,
        endpoint: bool = ...,
    ) -> int64: ...

# 定义 `integers` 方法的重载，用于生成有符号64位整数序列
# - 参数与前述方法相同，只是 `dtype` 指定为 `int64`
# - 返回类型为有符
    def integers(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int_] | type[int] | type[int_] | _IntCodes | _SupportsDType[dtype[int_]] = ...,
        endpoint: bool = ...,
    ) -> int_: ...
    # 生成一个方法 integers 的重载声明，返回一个 uint16 类型的 NumPy 数组
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[uint16] | type[uint16] | _UInt16Codes | _SupportsDType[dtype[uint16]] = ...,
        endpoint: bool = ...,
    ) -> NDArray[uint16]: ...

    # 生成一个方法 integers 的重载声明，返回一个 uint32 类型的 NumPy 数组
    @overload
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[uint32] | type[uint32] | _UInt32Codes | _SupportsDType[dtype[uint32]] = ...,
        endpoint: bool = ...,
    ) -> NDArray[uint32]: ...

    # 生成一个方法 integers 的重载声明，返回一个 uint64 类型的 NumPy 数组
    @overload
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[uint64] | type[uint64] | _UInt64Codes | _SupportsDType[dtype[uint64]] = ...,
        endpoint: bool = ...,
    ) -> NDArray[uint64]: ...

    # 生成一个方法 integers 的重载声明，返回一个 int 类型的 NumPy 数组
    @overload
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[int_] | type[int] | type[int_] | _IntCodes | _SupportsDType[dtype[int_]] = ...,
        endpoint: bool = ...,
    ) -> NDArray[int_]: ...

    # 生成一个方法 integers 的重载声明，返回一个 uint 类型的 NumPy 数组
    @overload
    def integers(
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[uint] | type[uint] | _UIntCodes | _SupportsDType[dtype[uint]] = ...,
        endpoint: bool = ...,
    ) -> NDArray[uint]: ...

    # TODO: 使用一个 TypeVar _T 来避免返回任何类型的结果？应该是 int -> NDArray[int64]，ArrayLike[_T] -> _T | NDArray[Any]
    # 生成一个方法 choice 的重载声明，返回一个 int 类型的值
    @overload
    def choice(
        self,
        a: int,
        size: None = ...,
        replace: bool = ...,
        p: None | _ArrayLikeFloat_co = ...,
        axis: int = ...,
        shuffle: bool = ...,
    ) -> int: ...

    # 生成一个方法 choice 的重载声明，返回一个 int64 类型的 NumPy 数组
    @overload
    def choice(
        self,
        a: int,
        size: _ShapeLike = ...,
        replace: bool = ...,
        p: None | _ArrayLikeFloat_co = ...,
        axis: int = ...,
        shuffle: bool = ...,
    ) -> NDArray[int64]: ...

    # 生成一个方法 choice 的重载声明，返回一个任意类型的结果
    @overload
    def choice(
        self,
        a: ArrayLike,
        size: None = ...,
        replace: bool = ...,
        p: None | _ArrayLikeFloat_co = ...,
        axis: int = ...,
        shuffle: bool = ...,
    ) -> Any: ...

    # 生成一个方法 choice 的重载声明，返回一个任意类型的 NumPy 数组
    @overload
    def choice(
        self,
        a: ArrayLike,
        size: _ShapeLike = ...,
        replace: bool = ...,
        p: None | _ArrayLikeFloat_co = ...,
        axis: int = ...,
        shuffle: bool = ...,
    ) -> NDArray[Any]: ...

    # 生成一个方法 uniform 的重载声明，返回一个浮点数类型的值
    @overload
    def uniform(
        self,
        low: _FloatLike_co = ...,
        high: _FloatLike_co = ...,
        size: None = ...,
    ) -> float: ...
    @overload
    def uniform(
        self,
        low: _ArrayLikeFloat_co = ...,
        high: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]:
    """定义一个方法重载，用于生成均匀分布的随机数。

    Parameters:
    - low: 最小值，可以是单个浮点数或浮点数数组
    - high: 最大值，可以是单个浮点数或浮点数数组
    - size: 数组形状，可以是 None 或者形状描述

    Returns:
    - NDArray[float64]: 生成的均匀分布的随机数数组
    """



    @overload
    def normal(
        self,
        loc: _FloatLike_co = ...,
        scale: _FloatLike_co = ...,
        size: None = ...,
    ) -> float:  # type: ignore[misc]
    """定义一个方法重载，用于生成正态分布的随机数（返回单个浮点数）。

    Parameters:
    - loc: 正态分布的均值，可以是单个浮点数
    - scale: 正态分布的标准差，可以是单个浮点数
    - size: 数组形状，可以是 None

    Returns:
    - float: 生成的正态分布的随机数
    """



    @overload
    def normal(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]:
    """定义一个方法重载，用于生成正态分布的随机数（返回浮点数数组）。

    Parameters:
    - loc: 正态分布的均值，可以是单个浮点数或浮点数数组
    - scale: 正态分布的标准差，可以是单个浮点数或浮点数数组
    - size: 数组形状，可以是 None 或者形状描述

    Returns:
    - NDArray[float64]: 生成的正态分布的随机数数组
    """


（以下类似地注释其他函数重载的定义，保留格式一致性和详细解释）
    @overload
    def noncentral_chisquare(
        self, df: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...


# 方法重载定义：noncentral_chisquare 方法的类型签名
# 接受 df 和 nonc 参数作为数组或类似数组的浮点数，size 参数可以是 None 或者形状对象，返回一个浮点数数组



    @overload
    def standard_t(self, df: _FloatLike_co, size: None = ...) -> float: ...  # type: ignore[misc]


# 方法重载定义：standard_t 方法的类型签名
# 接受 df 参数作为浮点数，size 参数可以是 None，返回一个浮点数
# 标注类型忽略某些杂项警告



    @overload
    def standard_t(
        self, df: _ArrayLikeFloat_co, size: None = ...
    ) -> NDArray[float64]: ...


# 方法重载定义：standard_t 方法的类型签名
# 接受 df 参数作为数组或类似数组的浮点数，size 参数可以是 None，返回一个浮点数数组



    @overload
    def standard_t(
        self, df: _ArrayLikeFloat_co, size: _ShapeLike = ...
    ) -> NDArray[float64]: ...


# 方法重载定义：standard_t 方法的类型签名
# 接受 df 参数作为数组或类似数组的浮点数，size 参数可以是形状对象或 None，返回一个浮点数数组



    @overload
    def vonmises(self, mu: _FloatLike_co, kappa: _FloatLike_co, size: None = ...) -> float: ...  # type: ignore[misc]


# 方法重载定义：vonmises 方法的类型签名
# 接受 mu 和 kappa 参数作为浮点数，size 参数可以是 None，返回一个浮点数
# 标注类型忽略某些杂项警告



    @overload
    def vonmises(
        self, mu: _ArrayLikeFloat_co, kappa: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...


# 方法重载定义：vonmises 方法的类型签名
# 接受 mu 和 kappa 参数作为数组或类似数组的浮点数，size 参数可以是 None 或者形状对象，返回一个浮点数数组



    @overload
    def pareto(self, a: _FloatLike_co, size: None = ...) -> float: ...  # type: ignore[misc]


# 方法重载定义：pareto 方法的类型签名
# 接受 a 参数作为浮点数，size 参数可以是 None，返回一个浮点数
# 标注类型忽略某些杂项警告



    @overload
    def pareto(
        self, a: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...


# 方法重载定义：pareto 方法的类型签名
# 接受 a 参数作为数组或类似数组的浮点数，size 参数可以是 None 或者形状对象，返回一个浮点数数组



    @overload
    def weibull(self, a: _FloatLike_co, size: None = ...) -> float: ...  # type: ignore[misc]


# 方法重载定义：weibull 方法的类型签名
# 接受 a 参数作为浮点数，size 参数可以是 None，返回一个浮点数
# 标注类型忽略某些杂项警告



    @overload
    def weibull(
        self, a: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...


# 方法重载定义：weibull 方法的类型签名
# 接受 a 参数作为数组或类似数组的浮点数，size 参数可以是 None 或者形状对象，返回一个浮点数数组



    @overload
    def power(self, a: _FloatLike_co, size: None = ...) -> float: ...  # type: ignore[misc]


# 方法重载定义：power 方法的类型签名
# 接受 a 参数作为浮点数，size 参数可以是 None，返回一个浮点数
# 标注类型忽略某些杂项警告



    @overload
    def power(
        self, a: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...


# 方法重载定义：power 方法的类型签名
# 接受 a 参数作为数组或类似数组的浮点数，size 参数可以是 None 或者形状对象，返回一个浮点数数组



    @overload
    def standard_cauchy(self, size: None = ...) -> float: ...  # type: ignore[misc]


# 方法重载定义：standard_cauchy 方法的类型签名
# 接受 size 参数可以是 None，返回一个浮点数
# 标注类型忽略某些杂项警告



    @overload
    def standard_cauchy(self, size: _ShapeLike = ...) -> NDArray[float64]: ...


# 方法重载定义：standard_cauchy 方法的类型签名
# 接受 size 参数可以是形状对象或 None，返回一个浮点数数组



    @overload
    def laplace(
        self,
        loc: _FloatLike_co = ...,
        scale: _FloatLike_co = ...,
        size: None = ...,
    ) -> float: ...  # type: ignore[misc]


# 方法重载定义：laplace 方法的类型签名
# 接受 loc 和 scale 参数作为浮点数，size 参数可以是 None，返回一个浮点数
# 标注类型忽略某些杂项警告



    @overload
    def laplace(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...


# 方法重载定义：laplace 方法的类型签名
# 接受 loc 和 scale 参数作为数组或类似数组的浮点数，size 参数可以是 None 或者形状对象，返回一个浮点数数组



    @overload
    def gumbel(
        self,
        loc: _FloatLike_co = ...,
        scale: _FloatLike_co = ...,
        size: None = ...,
    ) -> float: ...  # type: ignore[misc]


# 方法重载定义：gumbel 方法的类型签名
# 接受 loc 和 scale 参数作为浮点数，size 参数可以是 None，返回一个浮点数
# 标注类型忽略某些杂项警告



    @overload
    def gumbel(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...


# 方法重载定义：gumbel 方法的类型签名
# 接受 loc 和 scale 参数作为数组或类似数组的浮点数，size 参数可以是 None 或者形状对象，返回一个浮点数数组



    @overload
    def logistic(
        self,
        loc: _FloatLike_co = ...,
        scale: _FloatLike_co = ...,
        size: None = ...,
    ) -> float: ...  # type: ignore[misc]


# 方法重载定义：logistic 方法的类型签名
# 接受 loc 和 scale 参数作为浮点数，size 参数可以是 None，返回一个浮点数
# 标注类型忽略某
    ) -> float: ...  # type: ignore[misc]
    # 声明一个方法签名，该方法返回一个浮点数，忽略类型检查警告
    @overload
    def lognormal(
        self,
        mean: _ArrayLikeFloat_co = ...,
        sigma: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...
    # 方法重载注解，定义了 lognormal 方法的不同参数组合及其返回类型
    @overload
    def rayleigh(self, scale: _FloatLike_co = ..., size: None = ...) -> float: ...  # type: ignore[misc]
    # 方法重载注解，定义了 rayleigh 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def rayleigh(
        self, scale: _ArrayLikeFloat_co = ..., size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...
    # 方法重载注解，定义了 rayleigh 方法的不同参数组合及其返回类型
    @overload
    def wald(self, mean: _FloatLike_co, scale: _FloatLike_co, size: None = ...) -> float: ...  # type: ignore[misc]
    # 方法重载注解，定义了 wald 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def wald(
        self, mean: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...
    # 方法重载注解，定义了 wald 方法的不同参数组合及其返回类型
    @overload
    def triangular(
        self,
        left: _FloatLike_co,
        mode: _FloatLike_co,
        right: _FloatLike_co,
        size: None = ...,
    ) -> float: ...
    # 方法重载注解，定义了 triangular 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def triangular(
        self,
        left: _ArrayLikeFloat_co,
        mode: _ArrayLikeFloat_co,
        right: _ArrayLikeFloat_co,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...
    # 方法重载注解，定义了 triangular 方法的不同参数组合及其返回类型
    @overload
    def binomial(self, n: int, p: _FloatLike_co, size: None = ...) -> int: ...  # type: ignore[misc]
    # 方法重载注解，定义了 binomial 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def binomial(
        self, n: _ArrayLikeInt_co, p: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[int64]: ...
    # 方法重载注解，定义了 binomial 方法的不同参数组合及其返回类型
    @overload
    def negative_binomial(self, n: _FloatLike_co, p: _FloatLike_co, size: None = ...) -> int: ...  # type: ignore[misc]
    # 方法重载注解，定义了 negative_binomial 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def negative_binomial(
        self, n: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[int64]: ...
    # 方法重载注解，定义了 negative_binomial 方法的不同参数组合及其返回类型
    @overload
    def poisson(self, lam: _FloatLike_co = ..., size: None = ...) -> int: ...  # type: ignore[misc]
    # 方法重载注解，定义了 poisson 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def poisson(
        self, lam: _ArrayLikeFloat_co = ..., size: None | _ShapeLike = ...
    ) -> NDArray[int64]: ...
    # 方法重载注解，定义了 poisson 方法的不同参数组合及其返回类型
    @overload
    def zipf(self, a: _FloatLike_co, size: None = ...) -> int: ...  # type: ignore[misc]
    # 方法重载注解，定义了 zipf 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def zipf(
        self, a: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[int64]: ...
    # 方法重载注解，定义了 zipf 方法的不同参数组合及其返回类型
    @overload
    def geometric(self, p: _FloatLike_co, size: None = ...) -> int: ...  # type: ignore[misc]
    # 方法重载注解，定义了 geometric 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def geometric(
        self, p: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[int64]: ...
    # 方法重载注解，定义了 geometric 方法的不同参数组合及其返回类型
    @overload
    def hypergeometric(self, ngood: int, nbad: int, nsample: int, size: None = ...) -> int: ...  # type: ignore[misc]
    # 方法重载注解，定义了 hypergeometric 方法的不同参数组合及其返回类型，忽略类型检查警告
    @overload
    def hypergeometric(
        self,
        ngood: _ArrayLikeInt_co,
        nbad: _ArrayLikeInt_co,
        nsample: _ArrayLikeInt_co,
        size: None | _ShapeLike = ...,
    ) -> NDArray[int64]: ...
    # 方法重载注解，定义了 hypergeometric 方法的不同参数组合及其返回类型
    @overload
    def logseries(self, p: _FloatLike_co, size: None = ...) -> int: ...  # type: ignore[misc]
    # 方法重载注解，定义了 logseries 方法的不同参数组合及其返回类型，忽略类型检查警告
    # 定义 logseries 方法，生成服从对数级数分布的随机整数数组
    def logseries(
        self, p: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[int64]: ...

    # 定义 multivariate_normal 方法，生成多变量正态分布的随机数数组
    def multivariate_normal(
        self,
        mean: _ArrayLikeFloat_co,
        cov: _ArrayLikeFloat_co,
        size: None | _ShapeLike = ...,
        check_valid: Literal["warn", "raise", "ignore"] = ...,
        tol: float = ...,
        *,
        method: Literal["svd", "eigh", "cholesky"] = ...,
    ) -> NDArray[float64]: ...

    # 定义 multinomial 方法，生成多项分布的随机整数数组
    def multinomial(
        self, n: _ArrayLikeInt_co,
            pvals: _ArrayLikeFloat_co,
            size: None | _ShapeLike = ...
    ) -> NDArray[int64]: ...

    # 定义 multivariate_hypergeometric 方法，生成多元超几何分布的随机整数数组
    def multivariate_hypergeometric(
        self,
        colors: _ArrayLikeInt_co,
        nsample: int,
        size: None | _ShapeLike = ...,
        method: Literal["marginals", "count"] = ...,
    ) -> NDArray[int64]: ...

    # 定义 dirichlet 方法，生成 Dirichlet 分布的随机数数组
    def dirichlet(
        self, alpha: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...

    # 定义 permuted 方法，对数组进行轴向排列，并返回排列后的新数组
    def permuted(
        self, x: ArrayLike, *, axis: None | int = ..., out: None | NDArray[Any] = ...
    ) -> NDArray[Any]: ...

    # 定义 shuffle 方法，对数组进行原地乱序操作
    def shuffle(self, x: ArrayLike, axis: int = ...) -> None: ...
# 定义一个函数 default_rng，用于创建一个随机数生成器对象
def default_rng(
    # 函数参数 seed 可以接受以下类型的值：
    seed: None | _ArrayLikeInt_co | SeedSequence | BitGenerator | Generator = ...
) -> Generator:
    # 函数返回一个 Generator 对象，用于生成随机数
    ...
```