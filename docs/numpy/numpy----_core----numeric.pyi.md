# `D:\src\scipysrc\numpy\numpy\_core\numeric.pyi`

```py
# 从 collections.abc 导入 Callable 和 Sequence 类型
from collections.abc import Callable, Sequence
# 导入 typing 模块的多个类型和装饰器
from typing import (
    Any,                # 任意类型
    overload,           # 函数重载装饰器
    TypeVar,            # 泛型类型变量
    Literal as L,       # 字面值类型别名
    SupportsAbs,        # 支持绝对值的类型
    SupportsIndex,      # 支持索引的类型
    NoReturn,           # 函数无返回值
)

# 根据系统版本导入不同的类型
if sys.version_info >= (3, 10):
    from typing import TypeGuard   # 类型守卫（Python 3.10 及以上版本）
else:
    from typing_extensions import TypeGuard  # 类型守卫（Python 3.10 以下版本的扩展）

import numpy as np                 # 导入 numpy 库
from numpy import (                # 从 numpy 导入多个子模块和类
    ComplexWarning as ComplexWarning,   # 复数警告类别
    generic,                        # 泛型类型
    unsignedinteger,                # 无符号整数类型
    signedinteger,                  # 有符号整数类型
    floating,                       # 浮点数类型
    complexfloating,                # 复数类型
    int_,                           # 整数类型
    intp,                           # 整数指针类型
    float64,                        # 双精度浮点数类型
    timedelta64,                    # 时间增量类型
    object_,                        # 对象类型
    _OrderKACF,                     # 数组排序类型（知识缺失）
    _OrderCF,                       # 数组排序类型（知识缺失）
)

from numpy._typing import (         # 导入 numpy 的类型注解
    ArrayLike,                      # 类数组类型
    NDArray,                        # 多维数组类型
    DTypeLike,                      # 数据类型或类似类型
    _ShapeLike,                     # 形状或类似类型
    _DTypeLike,                     # 数据类型或类似类型
    _ArrayLike,                     # 类数组类型
    _SupportsArrayFunc,             # 支持数组函数的类型
    _ScalarLike_co,                 # 协变标量类型
    _ArrayLikeBool_co,              # 协变布尔数组类型
    _ArrayLikeUInt_co,              # 协变无符号整数数组类型
    _ArrayLikeInt_co,               # 协变有符号整数数组类型
    _ArrayLikeFloat_co,             # 协变浮点数数组类型
    _ArrayLikeComplex_co,           # 协变复数数组类型
    _ArrayLikeTD64_co,              # 协变时间增量数组类型
    _ArrayLikeObject_co,            # 协变对象数组类型
    _ArrayLikeUnknown,              # 未知协变数组类型
)

_T = TypeVar("_T")                  # 定义泛型类型变量 _T
_SCT = TypeVar("_SCT", bound=generic)   # 定义泛型类型变量 _SCT，限制为泛型类型
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])   # 定义泛型类型变量 _ArrayType，限制为任意类型的 NDArray

_CorrelateMode = L["valid", "same", "full"]    # 定义字面值类型别名 _CorrelateMode，包含三个字符串元素

__all__: list[str] = []             # 定义模块的公开接口列表，初始为空列表

@overload
def zeros_like(
    a: _ArrayType,                  # 参数 a 的类型为 _ArrayType
    dtype: None = ...,              # dtype 参数，默认为 None
    order: _OrderKACF = ...,        # order 参数，默认为 _OrderKACF 类型
    subok: L[True] = ...,           # subok 参数，默认为字面值类型 True
    shape: None = ...,              # shape 参数，默认为 None
    *,
    device: None | L["cpu"] = ...,  # device 参数，默认为 None 或 "cpu" 字面值类型
) -> _ArrayType: ...                # 返回类型为 _ArrayType 的函数重载

@overload
def zeros_like(
    a: _ArrayLike[_SCT],            # 参数 a 的类型为 _ArrayLike[_SCT]
    dtype: None = ...,              # dtype 参数，默认为 None
    order: _OrderKACF = ...,        # order 参数，默认为 _OrderKACF 类型
    subok: bool = ...,              # subok 参数，默认为布尔类型
    shape: None | _ShapeLike = ..., # shape 参数，默认为 None 或 _ShapeLike 类型
    *,
    device: None | L["cpu"] = ...,  # device 参数，默认为 None 或 "cpu" 字面值类型
) -> NDArray[_SCT]: ...             # 返回类型为 NDArray[_SCT] 的函数重载

@overload
def zeros_like(
    a: object,                      # 参数 a 的类型为 object
    dtype: None = ...,              # dtype 参数，默认为 None
    order: _OrderKACF = ...,        # order 参数，默认为 _OrderKACF 类型
    subok: bool = ...,              # subok 参数，默认为布尔类型
    shape: None | _ShapeLike = ..., # shape 参数，默认为 None 或 _ShapeLike 类型
    *,
    device: None | L["cpu"] = ...,  # device 参数，默认为 None 或 "cpu" 字面值类型
) -> NDArray[Any]: ...              # 返回类型为 NDArray[Any] 的函数重载

@overload
def zeros_like(
    a: Any,                         # 参数 a 的类型为 Any
    dtype: _DTypeLike[_SCT],        # dtype 参数的类型为 _DTypeLike[_SCT]
    order: _OrderKACF = ...,        # order 参数，默认为 _OrderKACF 类型
    subok: bool = ...,              # subok 参数，默认为布尔类型
    shape: None | _ShapeLike = ..., # shape 参数，默认为 None 或 _ShapeLike 类型
    *,
    device: None | L["cpu"] = ...,  # device 参数，默认为 None 或 "cpu" 字面值类型
) -> NDArray[_SCT]: ...             # 返回类型为 NDArray[_SCT] 的函数重载

@overload
def zeros_like(
    a: Any,                         # 参数 a 的类型为 Any
    dtype: DTypeLike,               # dtype 参数的类型为 DTypeLike
    order: _OrderKACF = ...,        # order 参数，默认为 _OrderKACF 类型
    subok: bool = ...,              # subok 参数，默认为布尔类型
    shape: None | _ShapeLike = ..., # shape 参数，默认为 None 或 _ShapeLike 类型
    *,
    device: None | L["cpu"] = ...,  # device 参数，默认为 None 或 "cpu" 字面值类型
) -> NDArray[Any]: ...              # 返回类型为 NDArray[Any] 的函数重载

@overload
def ones(
    shape: _ShapeLike,              # 参数 shape 的类型为 _ShapeLike
    dtype: None = ...,              # dtype 参数，默认为 None
    order: _OrderCF = ...,          # order 参数，默认为 _OrderCF 类型
    *,
    device: None | L["cpu"] = ...,  # device 参数，默认为 None 或 "cpu" 字面值类型
    like: _SupportsArrayFunc = ..., # like 参数，默认为 _SupportsArrayFunc 类型
) -> NDArray[float64]: ...          # 返回类型为 NDArray[float64] 的函数重载

@overload
def ones(
    shape: _ShapeLike,              # 参数 shape 的类型为 _ShapeLike
    dtype: _DTypeLike[_SCT],        # dtype 参数的类型为 _DTypeLike[_SCT]
    order: _OrderCF = ...,          # order 参数，默认为 _OrderCF 类型
    *,
    device: None | L["cpu"] = ...,  # device 参数，默认为 None 或 "cpu" 字面值类型
    like: _SupportsArrayFunc = ..., # like 参数，默认为 _SupportsArrayFunc 类型
) -> NDArray[_SCT]: ...             # 返回类型为 NDArray[_SCT] 的函数重载

@overload
def ones(
    shape: _ShapeLike,              # 参数 shape 的类型为 _ShapeLike
    dtype: DTypeLike,               # dtype 参数的类型为 DTypeLike
    order: _OrderCF = ...,          # order 参数，默认为 _OrderCF 类型
    *,
    device: None | L["cpu"] = ...,  # device 参数，默认为 None 或 "cpu" 字面值类型
    like: _SupportsArray
    a: _ArrayLike[_SCT],  # 定义变量a，其类型为_ArrayLike[_SCT]，表示可能是某种数组类型
    dtype: None = ...,    # 定义变量dtype，默认为None，可能表示数据类型
    order: _OrderKACF = ...,  # 定义变量order，默认为_OrdeKACF类型，可能表示数组的存储顺序
    subok: bool = ...,    # 定义变量subok，默认为bool类型，可能表示是否允许返回子类对象
    shape: None | _ShapeLike = ...,  # 定义变量shape，默认为None或者_ShapeLike类型，可能表示数组的形状
    *,                    # 分隔位置参数和关键字参数的分隔符
    device: None | L["cpu"] = ...,  # 定义变量device，默认为None或者L["cpu"]类型，可能表示数据所在的设备
# 定义一个函数签名，声明返回类型为 NDArray[_SCT]
def ones_like(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike= ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...

# 函数重载：根据输入参数 a 的类型和 dtype 创建形状相同的全 1 数组
@overload
def ones_like(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike= ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[_SCT]: ...

# 函数重载：根据输入参数 a 的类型和 dtype 创建形状相同的全 1 数组
@overload
def ones_like(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike= ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...

# 函数签名：根据形状、填充值和数据类型创建全填充数组
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 函数重载：根据形状、填充值、数据类型和类似对象创建全填充数组
@overload
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...

# 函数重载：根据形状、填充值、数据类型创建全填充数组
@overload
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: DTypeLike,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 函数签名：根据输入数组 a 的形状创建全填充数组，填充值为指定值
@overload
def full_like(
    a: _ArrayType,
    fill_value: Any,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: L[True] = ...,
    shape: None = ...,
    *,
    device: None | L["cpu"] = ...,
) -> _ArrayType: ...

# 函数重载：根据输入数组 a 的形状创建全填充数组，填充值为指定值，数据类型为指定类型
@overload
def full_like(
    a: _ArrayLike[_SCT],
    fill_value: Any,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[_SCT]: ...

# 函数重载：根据输入对象 a 的形状创建全填充数组，填充值为指定值
@overload
def full_like(
    a: object,
    fill_value: Any,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike= ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...

# 函数重载：根据输入对象 a 的形状创建全填充数组，填充值为指定值，数据类型为指定类型
@overload
def full_like(
    a: Any,
    fill_value: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike= ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[_SCT]: ...

# 函数重载：根据输入对象 a 的形状创建全填充数组，填充值为指定值，数据类型为指定类型
@overload
def full_like(
    a: Any,
    fill_value: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike= ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...

# 函数签名：计算数组中非零元素的数量
@overload
def count_nonzero(
    a: ArrayLike,
    axis: None = ...,
    *,
    keepdims: L[False] = ...,
) -> int: ...

# 函数签名：计算数组中非零元素的数量，可以指定计算的轴
@overload
def count_nonzero(
    a: ArrayLike,
    axis: _ShapeLike = ...,
    *,
    keepdims: bool = ...,
) -> Any: ...  # TODO: np.intp or ndarray[np.intp]

# 函数签名：判断数组是否按 Fortran 顺序存储
def isfortran(a: NDArray[Any] | generic) -> bool: ...

# 函数签名：返回数组中非零元素的索引数组
def argwhere(a: ArrayLike) -> NDArray[intp]: ...

# 函数签名：返回数组中所有非零元素的扁平索引数组
def flatnonzero(a: ArrayLike) -> NDArray[intp]: ...

# 函数签名：计算两个数组的相关性
@overload
def correlate(
    a: _ArrayLikeUnknown,
    v: _ArrayLikeUnknown,
    mode: _CorrelateMode = ...,


    # v: _ArrayLikeUnknown 是一个变量，其类型可能是数组或类似数组的结构，具体类型未知
    # mode: _CorrelateMode 是一个变量，其类型为 _CorrelateMode 类型，默认值为 ...
# 定义 correlate 函数的多个重载，计算两个数组的相关性
@overload
def correlate(
    a: _ArrayLikeBool_co,
    v: _ArrayLikeBool_co,
    mode: _CorrelateMode = ...,
) -> NDArray[np.bool]: ...
@overload
def correlate(
    a: _ArrayLikeUInt_co,
    v: _ArrayLikeUInt_co,
    mode: _CorrelateMode = ...,
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def correlate(
    a: _ArrayLikeInt_co,
    v: _ArrayLikeInt_co,
    mode: _CorrelateMode = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def correlate(
    a: _ArrayLikeFloat_co,
    v: _ArrayLikeFloat_co,
    mode: _CorrelateMode = ...,
) -> NDArray[floating[Any]]: ...
@overload
def correlate(
    a: _ArrayLikeComplex_co,
    v: _ArrayLikeComplex_co,
    mode: _CorrelateMode = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def correlate(
    a: _ArrayLikeTD64_co,
    v: _ArrayLikeTD64_co,
    mode: _CorrelateMode = ...,
) -> NDArray[timedelta64]: ...
@overload
def correlate(
    a: _ArrayLikeObject_co,
    v: _ArrayLikeObject_co,
    mode: _CorrelateMode = ...,
) -> NDArray[object_]: ...

# 定义 convolve 函数的多个重载，计算两个数组的卷积
@overload
def convolve(
    a: _ArrayLikeUnknown,
    v: _ArrayLikeUnknown,
    mode: _CorrelateMode = ...,
) -> NDArray[Any]: ...
@overload
def convolve(
    a: _ArrayLikeBool_co,
    v: _ArrayLikeBool_co,
    mode: _CorrelateMode = ...,
) -> NDArray[np.bool]: ...
@overload
def convolve(
    a: _ArrayLikeUInt_co,
    v: _ArrayLikeUInt_co,
    mode: _CorrelateMode = ...,
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def convolve(
    a: _ArrayLikeInt_co,
    v: _ArrayLikeInt_co,
    mode: _CorrelateMode = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def convolve(
    a: _ArrayLikeFloat_co,
    v: _ArrayLikeFloat_co,
    mode: _CorrelateMode = ...,
) -> NDArray[floating[Any]]: ...
@overload
def convolve(
    a: _ArrayLikeComplex_co,
    v: _ArrayLikeComplex_co,
    mode: _CorrelateMode = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def convolve(
    a: _ArrayLikeTD64_co,
    v: _ArrayLikeTD64_co,
    mode: _CorrelateMode = ...,
) -> NDArray[timedelta64]: ...
@overload
def convolve(
    a: _ArrayLikeObject_co,
    v: _ArrayLikeObject_co,
    mode: _CorrelateMode = ...,
) -> NDArray[object_]: ...

# 定义 outer 函数的多个重载，计算两个数组的外积
@overload
def outer(
    a: _ArrayLikeUnknown,
    b: _ArrayLikeUnknown,
    out: None = ...,
) -> NDArray[Any]: ...
@overload
def outer(
    a: _ArrayLikeBool_co,
    b: _ArrayLikeBool_co,
    out: None = ...,
) -> NDArray[np.bool]: ...
@overload
def outer(
    a: _ArrayLikeUInt_co,
    b: _ArrayLikeUInt_co,
    out: None = ...,
) -> NDArray[unsignedinteger[Any]]: ...
@overload
def outer(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    out: None = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def outer(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[floating[Any]]: ...
@overload
def outer(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    out: None = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def outer(
    a: _ArrayLikeTD64_co,
    b: _ArrayLikeTD64_co,
    out: None = ...,
) -> NDArray[timedelta64]: ...
@overload
def outer(
    a: _ArrayLikeObject_co,
    b: _ArrayLikeObject_co,
    out: None = ...,
) -> NDArray[object_]: ...
    b: _ArrayLikeTD64_co,
    out: None = ...,


# 声明变量b，其类型为_ArrayLikeTD64_co
b: _ArrayLikeTD64_co,
# 声明变量out，默认值为None
out: None = ...,
# 定义了一个函数签名，声明其返回类型为 NDArray[timedelta64]
) -> NDArray[timedelta64]: ...

# 函数重载：定义了一个函数 outer，接受两个 _ArrayLikeObject_co 类型的参数 a 和 b，并且 out 参数默认为 None
@overload
def outer(
    a: _ArrayLikeObject_co,
    b: _ArrayLikeObject_co,
    out: None = ...,
) -> NDArray[object_]: ...

# 函数重载：定义了一个函数 outer，接受三个可能类型的参数 a、b 和 out，其中 out 参数类型为 _ArrayType
@overload
def outer(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    b: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    out: _ArrayType,
) -> _ArrayType: ...

# 函数重载：定义了一个函数 tensordot，接受两个 _ArrayLikeUnknown 类型的参数 a 和 b，还有一个可选的 axes 参数，其默认类型为 int 或元组 _ShapeLike
@overload
def tensordot(
    a: _ArrayLikeUnknown,
    b: _ArrayLikeUnknown,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[Any]: ...

# 以下是 tensordot 函数的多个重载版本，分别适用于不同类型的数组 a 和 b，返回的数组类型分别是 bool、unsigned integer、signed integer、float、complex 和 timedelta64
@overload
def tensordot(
    a: _ArrayLikeBool_co,
    b: _ArrayLikeBool_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[np.bool]: ...

@overload
def tensordot(
    a: _ArrayLikeUInt_co,
    b: _ArrayLikeUInt_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[unsignedinteger[Any]]: ...

@overload
def tensordot(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[signedinteger[Any]]: ...

@overload
def tensordot(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[floating[Any]]: ...

@overload
def tensordot(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

@overload
def tensordot(
    a: _ArrayLikeTD64_co,
    b: _ArrayLikeTD64_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[timedelta64]: ...

@overload
def tensordot(
    a: _ArrayLikeObject_co,
    b: _ArrayLikeObject_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = ...,
) -> NDArray[object_]: ...

# 函数重载：定义了一个函数 roll，接受一个 _ArrayLike[_SCT] 类型的参数 a，一个 _ShapeLike 类型的参数 shift，以及一个可选的 axis 参数，其类型为 None 或 _ShapeLike
@overload
def roll(
    a: _ArrayLike[_SCT],
    shift: _ShapeLike,
    axis: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...

# 函数重载：定义了一个函数 roll，接受一个 ArrayLike 类型的参数 a，一个 _ShapeLike 类型的参数 shift，以及一个可选的 axis 参数，其类型为 None 或 _ShapeLike
@overload
def roll(
    a: ArrayLike,
    shift: _ShapeLike,
    axis: None | _ShapeLike = ...,
) -> NDArray[Any]: ...

# 定义了一个函数 rollaxis，接受一个 NDArray[_SCT] 类型的参数 a，一个整数类型的参数 axis，以及一个可选的 start 参数，默认为 ...
def rollaxis(
    a: NDArray[_SCT],
    axis: int,
    start: int = ...,
) -> NDArray[_SCT]: ...

# 定义了一个函数 moveaxis，接受一个 NDArray[_SCT] 类型的参数 a，两个 _ShapeLike 类型的参数 source 和 destination
def moveaxis(
    a: NDArray[_SCT],
    source: _ShapeLike,
    destination: _ShapeLike,
) -> NDArray[_SCT]: ...

# 函数重载：定义了一个函数 cross，接受两个 _ArrayLikeUnknown 类型的参数 x1 和 x2，还有四个可选的整数类型参数 axisa、axisb、axisc 和 axis
@overload
def cross(
    x1: _ArrayLikeUnknown,
    x2: _ArrayLikeUnknown,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[Any]: ...

# 函数重载：定义了一个函数 cross，接受两个 _ArrayLikeBool_co 类型的参数 x1 和 x2，还有四个可选的整数类型参数 axisa、axisb、axisc 和 axis，但是此函数不返回值
@overload
def cross(
    x1: _ArrayLikeBool_co,
    x2: _ArrayLikeBool_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NoReturn: ...

# 函数重载：定义了一个函数 cross，接受两个 _ArrayLikeUInt_co 类型的参数 x1 和 x2，还有四个可选的整数类型参数 axisa、axisb、axisc 和 axis，返回的数组类型为 unsigned integer
@overload
def cross(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[unsignedinteger[Any]]: ...

# 函数重载：定义了一个函数 cross，接受两个 _ArrayLikeInt_co 类型的参数 x1 和 x2，还有四个可选的整数类型参数 axisa、axisb、axisc 和 axis，返回的数组类型为 signed integer
@overload
def cross(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[signedinteger[Any]]: ...

# 此处省略了其余 cross 函数重载的注释，因为示例已经包含了详细的描述
    x2: _ArrayLikeFloat_co,
    # x2 是一个类型注解，表示 x2 是一个类似数组的浮点数类型的协变类型
    axisa: int = ...,
    # axisa 是一个整数类型的参数，默认值为省略号（...）
    axisb: int = ...,
    # axisb 是一个整数类型的参数，默认值为省略号（...）
    axisc: int = ...,
    # axisc 是一个整数类型的参数，默认值为省略号（...）
    axis: None | int = ...,
    # axis 是一个可以是 None 或整数类型的参数，默认值为省略号（...）
# 定义一个函数，计算两个数组的叉积
def cross(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: None | int = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# 定义一个函数，生成指定维度的索引数组
def indices(
    dimensions: Sequence[int],
    dtype: type[int] = ...,
    sparse: L[False] = ...,
) -> NDArray[int_]: ...

# 定义一个函数，根据给定函数和形状生成数组
def fromfunction(
    function: Callable[..., _T],
    shape: Sequence[int],
    *,
    dtype: DTypeLike = ...,
    like: _SupportsArrayFunc = ...,
    **kwargs: Any,
) -> _T: ...

# 定义一个函数，判断一个元素是否为标量类型
def isscalar(element: object) -> TypeGuard[
    generic | bool | int | float | complex | str | bytes | memoryview
]: ...

# 定义一个函数，将一个数字转换为指定进制的字符串表示
def binary_repr(num: SupportsIndex, width: None | int = ...) -> str: ...

# 定义一个函数，将一个数字转换为指定进制的字符串表示
def base_repr(
    number: SupportsAbs[float],
    base: float = ...,
    padding: SupportsIndex = ...,
) -> str: ...

# 定义一个函数，生成指定维度的单位矩阵
def identity(
    n: int,
    dtype: None = ...,
    *,
    like: _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...

# 定义一个函数，判断两个数组是否在误差范围内相等
def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = ...,
    atol: ArrayLike = ...,
    equal_nan: bool = ...,
) -> bool: ...

# 定义一个函数，判断两个数或数组是否在误差范围内相等
def isclose(
    a: _ScalarLike_co,
    b: _ScalarLike_co,
    rtol: ArrayLike = ...,
    atol: ArrayLike = ...,
    equal_nan: bool = ...,
) -> np.bool: ...

# 定义一个函数，判断两个数组是否完全相等
def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: bool = ...) -> bool: ...

# 定义一个函数，判断两个数组是否等价
def array_equiv(a1: ArrayLike, a2: ArrayLike) -> bool: ...

# 定义一个函数，将数组转换为指定数据类型
def astype(
    x: NDArray[Any],
    dtype: _DTypeLike[_SCT],
    copy: bool = ...,
) -> NDArray[_SCT]: ...
# 定义一个函数，返回类型为 `NDArray[Any]`
) -> NDArray[Any]: ...
```