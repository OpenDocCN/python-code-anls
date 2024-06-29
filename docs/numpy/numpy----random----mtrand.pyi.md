# `D:\src\scipysrc\numpy\numpy\random\mtrand.pyi`

```
# 导入内置模块 builtins，提供内置函数和异常
import builtins
# 从 collections.abc 模块导入 Callable 抽象基类，用于支持函数调用的类型提示
from collections.abc import Callable
# 从 typing 模块导入 Any、overload、Literal 类型提示
from typing import Any, overload, Literal

# 导入 NumPy 库，并从中导入所需的数据类型和函数
import numpy as np
from numpy import (
    dtype,        # 导入 dtype 类型，用于定义数组的数据类型
    float32,      # 32 位浮点数数据类型
    float64,      # 64 位浮点数数据类型
    int8,         # 8 位整数数据类型
    int16,        # 16 位整数数据类型
    int32,        # 32 位整数数据类型
    int64,        # 64 位整数数据类型
    int_,         # int 的别名，用于通用整数类型
    long,         # Python 2 中的长整数类型，在 Python 3 中为 int 的别名
    uint8,        # 无符号 8 位整数数据类型
    uint16,       # 无符号 16 位整数数据类型
    uint32,       # 无符号 32 位整数数据类型
    uint64,       # 无符号 64 位整数数据类型
    uint,         # 无符号整数类型
    ulong,        # 无符号长整数类型
)
# 从 numpy.random.bit_generator 模块导入 BitGenerator 类，用于生成随机数的位生成器
from numpy.random.bit_generator import BitGenerator
# 从 numpy._typing 模块导入各种类型提示，用于定义数组及其元素的类型
from numpy._typing import (
    ArrayLike,           # 数组或类数组类型提示
    NDArray,             # NumPy 数组类型提示
    _ArrayLikeFloat_co,  # 兼容浮点数的数组或类数组类型提示
    _ArrayLikeInt_co,    # 兼容整数的数组或类数组类型提示
    _DoubleCodes,        # 双精度浮点数代码提示
    _DTypeLikeBool,      # 布尔类型数据类型提示
    _DTypeLikeInt,       # 整数类型数据类型提示
    _DTypeLikeUInt,      # 无符号整数类型数据类型提示
    _Float32Codes,       # 单精度浮点数代码提示
    _Float64Codes,       # 双精度浮点数代码提示
    _Int8Codes,          # 8 位整数代码提示
    _Int16Codes,         # 16 位整数代码提示
    _Int32Codes,         # 32 位整数代码提示
    _Int64Codes,         # 64 位整数代码提示
    _IntCodes,           # 整数代码提示
    _LongCodes,          # 长整数代码提示
    _ShapeLike,          # 数组形状的类型提示
    _SingleCodes,        # 单精度浮点数代码提示
    _SupportsDType,      # 支持数据类型的通用类型提示
    _UInt8Codes,         # 无符号 8 位整数代码提示
    _UInt16Codes,        # 无符号 16 位整数代码提示
    _UInt32Codes,        # 无符号 32 位整数代码提示
    _UInt64Codes,        # 无符号 64 位整数代码提示
    _UIntCodes,          # 无符号整数代码提示
    _ULongCodes,         # 无符号长整数代码提示
)

# 定义 _DTypeLikeFloat32 类型别名，包括 float32 数据类型及其支持的相关代码提示
_DTypeLikeFloat32 = (
    dtype[float32]
    | _SupportsDType[dtype[float32]]
    | type[float32]
    | _Float32Codes
    | _SingleCodes
)

# 定义 _DTypeLikeFloat64 类型别名，包括 float64 数据类型及其支持的相关代码提示
_DTypeLikeFloat64 = (
    dtype[float64]
    | _SupportsDType[dtype[float64]]
    | type[float]
    | type[float64]
    | _Float64Codes
    | _DoubleCodes
)

# 定义 RandomState 类，用于生成随机数的状态管理器
class RandomState:
    _bit_generator: BitGenerator  # 随机数位生成器

    # 初始化方法，接受种子参数，初始化随机数生成器状态
    def __init__(self, seed: None | _ArrayLikeInt_co | BitGenerator = ...) -> None: ...

    # 返回对象的字符串表示形式
    def __repr__(self) -> str: ...

    # 返回对象的字符串表示形式
    def __str__(self) -> str: ...

    # 返回对象的序列化状态，以字典形式
    def __getstate__(self) -> dict[str, Any]: ...

    # 设置对象的序列化状态
    def __setstate__(self, state: dict[str, Any]) -> None: ...

    # 返回用于重建对象的可调用函数和参数
    def __reduce__(self) -> tuple[Callable[[BitGenerator], RandomState], tuple[BitGenerator], dict[str, Any]]: ...

    # 设置随机数生成器的种子
    def seed(self, seed: None | _ArrayLikeFloat_co = ...) -> None: ...

    # 获取当前随机数生成器的状态，支持不同的返回格式
    @overload
    def get_state(self, legacy: Literal[False] = ...) -> dict[str, Any]: ...

    # 获取当前随机数生成器的状态，支持不同的返回格式
    @overload
    def get_state(
        self, legacy: Literal[True] = ...
    ) -> dict[str, Any] | tuple[str, NDArray[uint32], int, int, float]: ...

    # 设置随机数生成器的状态
    def set_state(
        self, state: dict[str, Any] | tuple[str, NDArray[uint32], int, int, float]
    ) -> None: ...

    # 生成指定形状的随机样本，支持不同的返回类型
    @overload
    def random_sample(self, size: None = ...) -> float: ...  # type: ignore[misc]

    # 生成指定形状的随机样本，支持不同的返回类型
    @overload
    def random_sample(self, size: _ShapeLike) -> NDArray[float64]: ...

    # 生成指定形状的随机数，支持不同的返回类型
    @overload
    def random(self, size: None = ...) -> float: ...  # type: ignore[misc]

    # 生成指定形状的随机数，支持不同的返回类型
    @overload
    def random(self, size: _ShapeLike) -> NDArray[float64]: ...

    # 生成 Beta 分布的随机样本，支持不同的返回类型
    @overload
    def beta(self, a: float, b: float, size: None = ...) -> float: ...  # type: ignore[misc]

    # 生成 Beta 分布的随机样本，支持不同的返回类型
    @overload
    def beta(
        self, a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...

    # 生成指数分布的随机样本，支持不同的返回类型
    @overload
    def exponential(self, scale: float = ..., size: None = ...) -> float: ...  # type: ignore[misc]

    # 生成指数分布的随机样本，支持不同的返回类型
    @overload
    def exponential(
        self, scale: _ArrayLikeFloat_co = ..., size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...

    # 生成标准指数分布的随机样本，支持不同的返回类型
    @overload
    def standard_exponential(self, size: None = ...) -> float: ...  # type: ignore[misc]
    # 定义一个方法，生成一个标准指数分布的随机数数组
    def standard_exponential(self, size: _ShapeLike) -> NDArray[float64]: ...
    
    # 重载方法签名，将生成的随机数数组转换为整数类型的最大整数值
    def tomaxint(self, size: None = ...) -> int: ...  # type: ignore[misc]
    
    # 重载方法签名，生成一个指定大小的整数数组，存储在64位整数中
    # 生成的随机数范围为low到high之间（包括low但不包括high）
    def tomaxint(self, size: _ShapeLike) -> NDArray[int64]: ...
    
    # 重载方法签名，生成一个指定范围的随机整数
    # 如果未提供size，默认生成一个整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
    ) -> int: ...
    
    # 重载方法签名，生成一个指定范围的随机布尔值
    # 如果未提供size，默认生成一个布尔值
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: type[bool] = ...,
    ) -> bool: ...
    
    # 重载方法签名，生成一个指定范围的随机NumPy布尔值
    # 如果未提供size，默认生成一个NumPy布尔值
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: type[np.bool] = ...,
    ) -> np.bool: ...
    
    # 重载方法签名，生成一个指定范围的随机整数
    # 如果未提供size，默认生成一个整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: type[int] = ...,
    ) -> int: ...
    
    # 重载方法签名，生成一个指定范围的随机8位无符号整数
    # 如果未提供size，默认生成一个8位无符号整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint8] | type[uint8] | _UInt8Codes | _SupportsDType[dtype[uint8]] = ...,
    ) -> uint8: ...
    
    # 重载方法签名，生成一个指定范围的随机16位无符号整数
    # 如果未提供size，默认生成一个16位无符号整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint16] | type[uint16] | _UInt16Codes | _SupportsDType[dtype[uint16]] = ...,
    ) -> uint16: ...
    
    # 重载方法签名，生成一个指定范围的随机32位无符号整数
    # 如果未提供size，默认生成一个32位无符号整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint32] | type[uint32] | _UInt32Codes | _SupportsDType[dtype[uint32]] = ...,
    ) -> uint32: ...
    
    # 重载方法签名，生成一个指定范围的随机无符号整数
    # 如果未提供size，默认生成一个无符号整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint] | type[uint] | _UIntCodes | _SupportsDType[dtype[uint]] = ...,
    ) -> uint: ...
    
    # 重载方法签名，生成一个指定范围的随机无符号长整数
    # 如果未提供size，默认生成一个无符号长整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[ulong] | type[ulong] | _ULongCodes | _SupportsDType[dtype[ulong]] = ...,
    ) -> ulong: ...
    
    # 重载方法签名，生成一个指定范围的随机64位无符号整数
    # 如果未提供size，默认生成一个64位无符号整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[uint64] | type[uint64] | _UInt64Codes | _SupportsDType[dtype[uint64]] = ...,
    ) -> uint64: ...
    
    # 重载方法签名，生成一个指定范围的随机8位有符号整数
    # 如果未提供size，默认生成一个8位有符号整数
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int8] | type[int8] | _Int8Codes | _SupportsDType[dtype[int8]] = ...,
    ) -> int8: ...
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int16] | type[int16] | _Int16Codes | _SupportsDType[dtype[int16]] = ...,
    ) -> int16: ...
    ```
    # 定义名为 `randint` 的方法，用于生成随机整数
    # `self`: 方法的隐含参数，指代对象实例本身
    # `low`: 最小的可能返回值
    # `high`: 可选参数，最大的可能返回值
    # `size`: 可选参数，控制返回值的形状或大小
    # `dtype`: 可选参数，指定返回值的数据类型
    # `-> int16`: 方法返回一个 `int16` 类型的整数

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int32] | type[int32] | _Int32Codes | _SupportsDType[dtype[int32]] = ...,
    ) -> int32: ...
    ```
    # 方法的重载定义，接受不同的参数组合来支持返回不同数据类型的整数

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int_] | type[int_] | _IntCodes | _SupportsDType[dtype[int_]] = ...,
    ) -> int_: ...
    ```
    # 另一种重载定义，支持返回 `int_` 类型的整数

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[long] | type[long] | _LongCodes | _SupportsDType[dtype[long]] = ...,
    ) -> long: ...
    ```
    # 又一种重载定义，支持返回 `long` 类型的整数

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: int,
        high: None | int = ...,
        size: None = ...,
        dtype: dtype[int64] | type[int64] | _Int64Codes | _SupportsDType[dtype[int64]] = ...,
    ) -> int64: ...
    ```
    # 再一种重载定义，支持返回 `int64` 类型的整数

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[long]: ...
    ```
    # 支持接受数组形式的 `low` 参数，并返回一个 `NDArray[long]` 类型的数组

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: _DTypeLikeBool = ...,
    ) -> NDArray[np.bool]: ...
    ```
    # 支持接受数组形式的 `low` 参数，并返回一个 `NDArray[np.bool]` 类型的布尔数组

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[int8] | type[int8] | _Int8Codes | _SupportsDType[dtype[int8]] = ...,
    ) -> NDArray[int8]: ...
    ```
    # 支持接受数组形式的 `low` 参数，并返回一个 `NDArray[int8]` 类型的整数数组

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[int16] | type[int16] | _Int16Codes | _SupportsDType[dtype[int16]] = ...,
    ) -> NDArray[int16]: ...
    ```
    # 支持接受数组形式的 `low` 参数，并返回一个 `NDArray[int16]` 类型的整数数组

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[int32] | type[int32] | _Int32Codes | _SupportsDType[dtype[int32]] = ...,
    ) -> NDArray[int32]: ...
    ```
    # 支持接受数组形式的 `low` 参数，并返回一个 `NDArray[int32]` 类型的整数数组

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: None | dtype[int64] | type[int64] | _Int64Codes | _SupportsDType[dtype[int64]] = ...,
    ) -> NDArray[int64]: ...
    ```
    # 支持接受数组形式的 `low` 参数，并返回一个 `NDArray[int64]` 类型的整数数组

    @overload
    ```
    # 最后一种重载定义，支持不同的参数组合，以适应不同的调用需求
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[uint8] | type[uint8] | _UInt8Codes | _SupportsDType[dtype[uint8]] = ...,
    ) -> NDArray[uint8]: ...

# 定义了一个名为 `randint` 的方法，用于生成随机整数数组。此处的注释用于指示类型检查器忽略与杂项相关的类型错误。

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[uint16] | type[uint16] | _UInt16Codes | _SupportsDType[dtype[uint16]] = ...,
    ) -> NDArray[uint16]: ...

# `randint` 方法的重载，用于生成 `uint16` 类型的随机整数数组。同样使用了类型检查器忽略与杂项相关的类型错误。

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[uint32] | type[uint32] | _UInt32Codes | _SupportsDType[dtype[uint32]] = ...,
    ) -> NDArray[uint32]: ...

# `randint` 方法的重载，用于生成 `uint32` 类型的随机整数数组。同样使用了类型检查器忽略与杂项相关的类型错误。

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[uint64] | type[uint64] | _UInt64Codes | _SupportsDType[dtype[uint64]] = ...,
    ) -> NDArray[uint64]: ...

# `randint` 方法的重载，用于生成 `uint64` 类型的随机整数数组。同样使用了类型检查器忽略与杂项相关的类型错误。

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[long] | type[int] | type[long] | _LongCodes | _SupportsDType[dtype[long]] = ...,
    ) -> NDArray[long]: ...

# `randint` 方法的重载，用于生成 `long` 类型的随机整数数组。同样使用了类型检查器忽略与杂项相关的类型错误。

    @overload
    def randint(  # type: ignore[misc]
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
        dtype: dtype[ulong] | type[ulong] | _ULongCodes | _SupportsDType[dtype[ulong]] = ...,
    ) -> NDArray[ulong]: ...

# `randint` 方法的重载，用于生成 `ulong` 类型的随机整数数组。同样使用了类型检查器忽略与杂项相关的类型错误。

    def bytes(self, length: int) -> builtins.bytes: ...

# 定义了一个名为 `bytes` 的方法，用于生成指定长度的字节数组。

    @overload
    def choice(
        self,
        a: int,
        size: None = ...,
        replace: bool = ...,
        p: None | _ArrayLikeFloat_co = ...,
    ) -> int: ...

# `choice` 方法的重载，用于从整数 `a` 中做出选择，并返回一个整数。可选的参数包括选择结果的尺寸 `size`、是否替换 `replace`，以及概率分布 `p`。

    @overload
    def choice(
        self,
        a: int,
        size: _ShapeLike = ...,
        replace: bool = ...,
        p: None | _ArrayLikeFloat_co = ...,
    ) -> NDArray[long]: ...

# `choice` 方法的重载，用于从整数 `a` 中做出选择，并返回一个 `long` 类型的数组。可选的参数包括选择结果的尺寸 `size`、是否替换 `replace`，以及概率分布 `p`。

    @overload
    def choice(
        self,
        a: ArrayLike,
        size: None = ...,
        replace: bool = ...,
        p: None | _ArrayLikeFloat_co = ...,
    ) -> Any: ...

# `choice` 方法的重载，用于从数组 `a` 中做出选择，并返回任意类型的结果。可选的参数包括选择结果的尺寸 `size`、是否替换 `replace`，以及概率分布 `p`。

    @overload
    def choice(
        self,
        a: ArrayLike,
        size: _ShapeLike = ...,
        replace: bool = ...,
        p: None | _ArrayLikeFloat_co = ...,
    ) -> NDArray[Any]: ...

# `choice` 方法的重载，用于从数组 `a` 中做出选择，并返回一个 `Any` 类型的数组。可选的参数包括选择结果的尺寸 `size`、是否替换 `replace`，以及概率分布 `p`。

    @overload
    def uniform(self, low: float = ..., high: float = ..., size: None = ...) -> float: ...  # type: ignore[misc]

# 定义了一个名为 `uniform` 的方法，用于生成指定范围内的均匀分布的随机浮点数。此处的注释用于指示类型检查器忽略与杂项相关的类型错误。

    @overload
    def uniform(
        self,
        low: _ArrayLikeFloat_co = ...,
        high: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...

# `uniform` 方法的重载，用于生成指定范围内的均匀分布的随机浮点数数组。同样使用了类型检查器忽略与杂项相关的类型错误。
    # 定义一个 rand 方法，返回一个随机浮点数
    def rand(self) -> float: ...

    # 使用装饰器声明一个 rand 方法的重载，接受任意个整数参数，返回一个浮点数的 NumPy 数组
    @overload
    def rand(self, *args: int) -> NDArray[float64]: ...

    # 定义一个 randn 方法，返回一个标准正态分布的随机浮点数
    @overload
    def randn(self) -> float: ...

    # 使用装饰器声明一个 randn 方法的重载，接受任意个整数参数，返回一个浮点数的 NumPy 数组，表示标准正态分布
    @overload
    def randn(self, *args: int) -> NDArray[float64]: ...

    # 定义一个 random_integers 方法，返回一个指定范围内的随机整数
    @overload
    def random_integers(self, low: int, high: None | int = ..., size: None = ...) -> int: ...  # type: ignore[misc]

    # 使用装饰器声明一个 random_integers 方法的重载，接受一个整数或整数数组作为 low 参数，返回一个长整型的 NumPy 数组
    @overload
    def random_integers(
        self,
        low: _ArrayLikeInt_co,
        high: None | _ArrayLikeInt_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[long]: ...

    # 定义一个 standard_normal 方法，返回一个标准正态分布的随机浮点数
    @overload
    def standard_normal(self, size: None = ...) -> float: ...  # type: ignore[misc]

    # 使用装饰器声明一个 standard_normal 方法的重载，接受一个形状参数 size，返回一个浮点数的 NumPy 数组，表示标准正态分布
    @overload
    def standard_normal(
        self, size: _ShapeLike = ...
    ) -> NDArray[float64]: ...

    # 定义一个 normal 方法，返回一个指定均值和标准差的随机浮点数
    @overload
    def normal(self, loc: float = ..., scale: float = ..., size: None = ...) -> float: ...  # type: ignore[misc]

    # 使用装饰器声明一个 normal 方法的重载，接受均值和标准差参数可以是单个数或数组，返回一个浮点数的 NumPy 数组，表示正态分布
    @overload
    def normal(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...

    # 定义一个 standard_gamma 方法，返回一个形状参数指定的标准 Gamma 分布的随机浮点数
    @overload
    def standard_gamma(  # type: ignore[misc]
        self,
        shape: float,
        size: None = ...,
    ) -> float: ...

    # 使用装饰器声明一个 standard_gamma 方法的重载，接受形状参数可以是单个数或数组，返回一个浮点数的 NumPy 数组，表示标准 Gamma 分布
    @overload
    def standard_gamma(
        self,
        shape: _ArrayLikeFloat_co,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...

    # 定义一个 gamma 方法，返回一个形状和尺度参数指定的 Gamma 分布的随机浮点数
    @overload
    def gamma(self, shape: float, scale: float = ..., size: None = ...) -> float: ...  # type: ignore[misc]

    # 使用装饰器声明一个 gamma 方法的重载，接受形状和尺度参数可以是单个数或数组，返回一个浮点数的 NumPy 数组，表示 Gamma 分布
    @overload
    def gamma(
        self,
        shape: _ArrayLikeFloat_co,
        scale: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...

    # 定义一个 f 方法，返回自由度参数指定的 F 分布的随机浮点数
    @overload
    def f(self, dfnum: float, dfden: float, size: None = ...) -> float: ...  # type: ignore[misc]

    # 使用装饰器声明一个 f 方法的重载，接受自由度参数可以是单个数或数组，返回一个浮点数的 NumPy 数组，表示 F 分布
    @overload
    def f(
        self, dfnum: _ArrayLikeFloat_co, dfden: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...

    # 定义一个 noncentral_f 方法，返回自由度和非心参数指定的非心 F 分布的随机浮点数
    @overload
    def noncentral_f(self, dfnum: float, dfden: float, nonc: float, size: None = ...) -> float: ...  # type: ignore[misc]

    # 使用装饰器声明一个 noncentral_f 方法的重载，接受自由度和非心参数可以是单个数或数组，返回一个浮点数的 NumPy 数组，表示非心 F 分布
    @overload
    def noncentral_f(
        self,
        dfnum: _ArrayLikeFloat_co,
        dfden: _ArrayLikeFloat_co,
        nonc: _ArrayLikeFloat_co,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...

    # 定义一个 chisquare 方法，返回自由度参数指定的卡方分布的随机浮点数
    @overload
    def chisquare(self, df: float, size: None = ...) -> float: ...  # type: ignore[misc]

    # 使用装饰器声明一个 chisquare 方法的重载，接受自由度参数可以是单个数或数组，返回一个浮点数的 NumPy 数组，表示卡方分布
    @overload
    def chisquare(
        self, df: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...

    # 定义一个 noncentral_chisquare 方法，返回自由度和非心参数指定的非心卡方分布的随机浮点数
    @overload
    def noncentral_chisquare(self, df: float, nonc: float, size: None = ...) -> float: ...  # type: ignore[misc]

    # 使用装饰器声明一个 noncentral_chisquare 方法的重载，接受自由度和非心参数可以是单个数或数组，返回一个浮点数的 NumPy 数组，表示非心卡方分布
    @overload
    def noncentral_chisquare(
        self, df: _ArrayLikeFloat_co, nonc: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...

    # 定义一个 standard_t 方法，返回自由度参数指定的 t 分布的随机浮点数
    @overload
    def standard_t(self, df: float, size: None = ...) -> float: ...  # type: ignore[misc]
    def standard_t(
        self, df: _ArrayLikeFloat_co, size: None = ...
    ) -> NDArray[float64]:
    # 标准 t 分布函数的声明，接受一个自由度参数 df 和一个可选的尺寸参数 size
    ...

    @overload
    def standard_t(
        self, df: _ArrayLikeFloat_co, size: _ShapeLike = ...
    ) -> NDArray[float64]:
    # 标准 t 分布函数的重载声明，接受一个自由度参数 df 和一个形状参数 size
    ...

    @overload
    def vonmises(self, mu: float, kappa: float, size: None = ...) -> float:
    # 冯·米塞斯分布函数的声明，接受一个均值参数 mu，一个集中度参数 kappa 和一个可选的尺寸参数 size
    ...

    @overload
    def vonmises(
        self, mu: _ArrayLikeFloat_co, kappa: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]:
    # 冯·米塞斯分布函数的重载声明，接受一个均值参数 mu，一个集中度参数 kappa 和一个可选的形状参数 size
    ...

    @overload
    def pareto(self, a: float, size: None = ...) -> float:
    # 帕累托分布函数的声明，接受一个形状参数 a 和一个可选的尺寸参数 size
    ...

    @overload
    def pareto(
        self, a: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]:
    # 帕累托分布函数的重载声明，接受一个形状参数 a 和一个可选的形状参数 size
    ...

    @overload
    def weibull(self, a: float, size: None = ...) -> float:
    # 威布尔分布函数的声明，接受一个形状参数 a 和一个可选的尺寸参数 size
    ...

    @overload
    def weibull(
        self, a: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]:
    # 威布尔分布函数的重载声明，接受一个形状参数 a 和一个可选的形状参数 size
    ...

    @overload
    def power(self, a: float, size: None = ...) -> float:
    # 功率分布函数的声明，接受一个形状参数 a 和一个可选的尺寸参数 size
    ...

    @overload
    def power(
        self, a: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]:
    # 功率分布函数的重载声明，接受一个形状参数 a 和一个可选的形状参数 size
    ...

    @overload
    def standard_cauchy(self, size: None = ...) -> float:
    # 标准柯西分布函数的声明，接受一个可选的尺寸参数 size
    ...

    @overload
    def standard_cauchy(self, size: _ShapeLike = ...) -> NDArray[float64]:
    # 标准柯西分布函数的重载声明，接受一个形状参数 size
    ...

    @overload
    def laplace(self, loc: float = ..., scale: float = ..., size: None = ...) -> float:
    # 拉普拉斯分布函数的声明，接受一个位置参数 loc，一个尺度参数 scale 和一个可选的尺寸参数 size
    ...

    @overload
    def laplace(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]:
    # 拉普拉斯分布函数的重载声明，接受一个位置参数 loc，一个尺度参数 scale 和一个可选的形状参数 size
    ...

    @overload
    def gumbel(self, loc: float = ..., scale: float = ..., size: None = ...) -> float:
    # 甘贝尔分布函数的声明，接受一个位置参数 loc，一个尺度参数 scale 和一个可选的尺寸参数 size
    ...

    @overload
    def gumbel(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]:
    # 甘贝尔分布函数的重载声明，接受一个位置参数 loc，一个尺度参数 scale 和一个可选的形状参数 size
    ...

    @overload
    def logistic(self, loc: float = ..., scale: float = ..., size: None = ...) -> float:
    # 逻辑斯蒂分布函数的声明，接受一个位置参数 loc，一个尺度参数 scale 和一个可选的尺寸参数 size
    ...

    @overload
    def logistic(
        self,
        loc: _ArrayLikeFloat_co = ...,
        scale: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]:
    # 逻辑斯蒂分布函数的重载声明，接受一个位置参数 loc，一个尺度参数 scale 和一个可选的形状参数 size
    ...

    @overload
    def lognormal(self, mean: float = ..., sigma: float = ..., size: None = ...) -> float:
    # 对数正态分布函数的声明，接受一个均值参数 mean，一个标准差参数 sigma 和一个可选的尺寸参数 size
    ...

    @overload
    def lognormal(
        self,
        mean: _ArrayLikeFloat_co = ...,
        sigma: _ArrayLikeFloat_co = ...,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]:
    # 对数正态分布函数的重载声明，接受一个均值参数 mean，一个标准差参数 sigma 和一个可选的形状参数 size
    ...

    @overload
    def rayleigh(self, scale: float = ..., size: None = ...) -> float:
    # 瑞利分布函数的声明，接受一个尺度参数 scale 和一个可选的尺寸参数 size
    ...

    @overload
    def rayleigh(
        self, scale: _ArrayLikeFloat_co = ..., size: None | _ShapeLike = ...
    ) -> NDArray[float64]:
    # 瑞利分布函数的重载声明，接受一个尺度参数 scale 和一个可选的形状参数 size
    ...

    @overload
    ```
    def wald(self, mean: float, scale: float, size: None = ...) -> float: ...  # type: ignore[misc]
    # 定义 Wald 分布的概率密度函数，接受平均值和尺度参数
    @overload
    def wald(
        self, mean: _ArrayLikeFloat_co, scale: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[float64]
    
    @overload
    def triangular(self, left: float, mode: float, right: float, size: None = ...) -> float: ...  # type: ignore[misc]
    # 定义三角分布的概率密度函数，接受左侧、众数和右侧参数
    @overload
    def triangular(
        self,
        left: _ArrayLikeFloat_co,
        mode: _ArrayLikeFloat_co,
        right: _ArrayLikeFloat_co,
        size: None | _ShapeLike = ...,
    ) -> NDArray[float64]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[float64]
    
    @overload
    def binomial(self, n: int, p: float, size: None = ...) -> int: ...  # type: ignore[misc]
    # 定义二项分布的概率质量函数，接受试验次数和成功概率参数
    @overload
    def binomial(
        self, n: _ArrayLikeInt_co, p: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[long]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[long]，表示每个试验的成功次数
    
    @overload
    def negative_binomial(self, n: float, p: float, size: None = ...) -> int: ...  # type: ignore[misc]
    # 定义负二项分布的概率质量函数，接受成功次数和成功概率参数
    @overload
    def negative_binomial(
        self, n: _ArrayLikeFloat_co, p: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[long]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[long]，表示达到指定成功次数前所需的失败次数
    
    @overload
    def poisson(self, lam: float = ..., size: None = ...) -> int: ...  # type: ignore[misc]
    # 定义泊松分布的概率质量函数，接受泊松率参数
    @overload
    def poisson(
        self, lam: _ArrayLikeFloat_co = ..., size: None | _ShapeLike = ...
    ) -> NDArray[long]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[long]，表示指定区间内的事件发生次数
    
    @overload
    def zipf(self, a: float, size: None = ...) -> int: ...  # type: ignore[misc]
    # 定义 Zipf 分布的概率质量函数，接受参数 a
    @overload
    def zipf(
        self, a: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[long]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[long]
    
    @overload
    def geometric(self, p: float, size: None = ...) -> int: ...  # type: ignore[misc]
    # 定义几何分布的概率质量函数，接受成功概率参数
    @overload
    def geometric(
        self, p: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[long]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[long]，表示首次成功所需的试验次数
    
    @overload
    def hypergeometric(self, ngood: int, nbad: int, nsample: int, size: None = ...) -> int: ...  # type: ignore[misc]
    # 定义超几何分布的概率质量函数，接受总体好样本数、总体坏样本数和样本量参数
    @overload
    def hypergeometric(
        self,
        ngood: _ArrayLikeInt_co,
        nbad: _ArrayLikeInt_co,
        nsample: _ArrayLikeInt_co,
        size: None | _ShapeLike = ...,
    ) -> NDArray[long]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[long]，表示样本中好样本的数量
    
    @overload
    def logseries(self, p: float, size: None = ...) -> int: ...  # type: ignore[misc]
    # 定义对数系列分布的概率质量函数，接受参数 p
    @overload
    def logseries(
        self, p: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[long]: ...
    # 重载函数：处理数组输入的情况，返回 NumPy 数组 NDArray[long]，表示首次成功所需的试验次数
    
    def multivariate_normal(
        self,
        mean: _ArrayLikeFloat_co,
        cov: _ArrayLikeFloat_co,
        size: None | _ShapeLike = ...,
        check_valid: Literal["warn", "raise", "ignore"] = ...,
        tol: float = ...,
    ) -> NDArray[float64]: ...
    # 定义多元正态分布的概率密度函数，接受均值向量和协方差矩阵参数
    
    def multinomial(
        self, n: _ArrayLikeInt_co, pvals: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[long]: ...
    # 定义多项式分布的概率质量函数，接受试验次数和每个类别的概率参数
    
    def dirichlet(
        self, alpha: _ArrayLikeFloat_co, size: None | _ShapeLike = ...
    ) -> NDArray[float64]: ...
    # 定义狄利克雷分布的概率密度函数，接受参数 alpha
    
    def shuffle(self, x: ArrayLike) -> None: ...
    # 对数组 x 进行随机重排
    # 标注函数的重载方法，用于不同类型参数的排列组合生成
    @overload
    def permutation(self, x: int) -> NDArray[long]: ...
    # 标注函数的重载方法，用于不同类型参数的排列组合生成，支持任意数组类型
    @overload
    def permutation(self, x: ArrayLike) -> NDArray[Any]: ...
# 初始化一个 RandomState 对象，用于生成随机数
_rand: RandomState

# 以下是 RandomState 对象提供的一些随机数生成方法
beta = _rand.beta  # 生成 Beta 分布的随机数
binomial = _rand.binomial  # 生成二项分布的随机数
bytes = _rand.bytes  # 生成指定长度的随机字节串
chisquare = _rand.chisquare  # 生成卡方分布的随机数
choice = _rand.choice  # 从给定的样本中随机抽取元素
dirichlet = _rand.dirichlet  # 生成 Dirichlet 分布的随机数
exponential = _rand.exponential  # 生成指数分布的随机数
f = _rand.f  # 生成 F 分布的随机数
gamma = _rand.gamma  # 生成 Gamma 分布的随机数
get_state = _rand.get_state  # 获取当前 RandomState 对象的状态
geometric = _rand.geometric  # 生成几何分布的随机数
gumbel = _rand.gumbel  # 生成 Gumbel 分布的随机数
hypergeometric = _rand.hypergeometric  # 生成超几何分布的随机数
laplace = _rand.laplace  # 生成拉普拉斯分布的随机数
logistic = _rand.logistic  # 生成 Logistic 分布的随机数
lognormal = _rand.lognormal  # 生成对数正态分布的随机数
logseries = _rand.logseries  # 生成对数级数分布的随机数
multinomial = _rand.multinomial  # 生成多项分布的随机数
multivariate_normal = _rand.multivariate_normal  # 生成多变量正态分布的随机数
negative_binomial = _rand.negative_binomial  # 生成负二项分布的随机数
noncentral_chisquare = _rand.noncentral_chisquare  # 生成非中心卡方分布的随机数
noncentral_f = _rand.noncentral_f  # 生成非中心 F 分布的随机数
normal = _rand.normal  # 生成正态分布的随机数
pareto = _rand.pareto  # 生成帕累托分布的随机数
permutation = _rand.permutation  # 对给定序列进行随机排列
poisson = _rand.poisson  # 生成泊松分布的随机数
power = _rand.power  # 生成 Power 分布的随机数
rand = _rand.rand  # 生成均匀分布的随机数
randint = _rand.randint  # 生成指定范围内的随机整数
randn = _rand.randn  # 生成标准正态分布的随机数
random = _rand.random  # 生成 [0.0, 1.0) 范围内的随机浮点数
random_integers = _rand.random_integers  # 生成指定范围内的随机整数（包括上下界）
random_sample = _rand.random_sample  # 生成 [0.0, 1.0) 范围内的随机浮点数
rayleigh = _rand.rayleigh  # 生成 Rayleigh 分布的随机数
seed = _rand.seed  # 设置随机数生成器的种子
set_state = _rand.set_state  # 设置 RandomState 对象的状态
shuffle = _rand.shuffle  # 将序列随机打乱
standard_cauchy = _rand.standard_cauchy  # 生成标准 Cauchy 分布的随机数
standard_exponential = _rand.standard_exponential  # 生成标准指数分布的随机数
standard_gamma = _rand.standard_gamma  # 生成标准 Gamma 分布的随机数
standard_normal = _rand.standard_normal  # 生成标准正态分布的随机数
standard_t = _rand.standard_t  # 生成标准 t 分布的随机数
triangular = _rand.triangular  # 生成三角分布的随机数
uniform = _rand.uniform  # 生成均匀分布的随机数
vonmises = _rand.vonmises  # 生成 Von Mises 分布的随机数
wald = _rand.wald  # 生成 Wald 分布的随机数
weibull = _rand.weibull  # 生成 Weibull 分布的随机数
zipf = _rand.zipf  # 生成 Zipf 分布的随机数

# 以下两个是 random_sample 方法的别名，保留为向后兼容
sample = _rand.random_sample  # 生成 [0.0, 1.0) 范围内的随机浮点数
ranf = _rand.random_sample  # 生成 [0.0, 1.0) 范围内的随机浮点数

# 设置位生成器为指定的 BitGenerator 对象
def set_bit_generator(bitgen: BitGenerator) -> None:
    ...

# 获取当前的位生成器对象
def get_bit_generator() -> BitGenerator:
    ...
```