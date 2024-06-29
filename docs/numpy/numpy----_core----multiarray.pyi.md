# `D:\src\scipysrc\numpy\numpy\_core\multiarray.pyi`

```
# 导入内置模块和库
import builtins
import os
import datetime as dt
from collections.abc import Sequence, Callable, Iterable
from typing import (
    Literal as L,
    Any,
    overload,
    TypeVar,
    SupportsIndex,
    final,
    Final,
    Protocol,
    ClassVar,
)

# 导入 NumPy 库及其子模块
import numpy as np
from numpy import (
    # 重新导出的部分
    busdaycalendar as busdaycalendar,
    broadcast as broadcast,
    dtype as dtype,
    ndarray as ndarray,
    nditer as nditer,

    # 其余部分
    ufunc,
    str_,
    uint8,
    intp,
    int_,
    float64,
    timedelta64,
    datetime64,
    generic,
    unsignedinteger,
    signedinteger,
    floating,
    complexfloating,
    _OrderKACF,
    _OrderCF,
    _CastingKind,
    _ModeKind,
    _SupportsBuffer,
    _IOProtocol,
    _CopyMode,
    _NDIterFlagsKind,
    _NDIterOpFlagsKind,
)

from numpy._typing import (
    # 形状相关
    _ShapeLike,

    # 数据类型相关
    DTypeLike,
    _DTypeLike,

    # 数组相关
    NDArray,
    ArrayLike,
    _ArrayLike,
    _SupportsArrayFunc,
    _NestedSequence,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,
    _ArrayLikeStr_co,
    _ArrayLikeBytes_co,
    _ScalarLike_co,
    _IntLike_co,
    _FloatLike_co,
    _TD64Like_co,
)

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)
_SCT = TypeVar("_SCT", bound=generic)
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

# 时间单位的类型定义
_UnitKind = L[
    "Y",
    "M",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us", "μs",
    "ns",
    "ps",
    "fs",
    "as",
]
# 滚动类型的定义，排除了 `raise`
_RollKind = L[
    "nat",
    "forward",
    "following",
    "backward",
    "preceding",
    "modifiedfollowing",
    "modifiedpreceding",
]

# 定义支持长度和索引获取操作的协议
class _SupportsLenAndGetItem(Protocol[_T_contra, _T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, key: _T_contra, /) -> _T_co: ...

# 导出的符号列表
__all__: list[str]

# 允许线程标识常量
ALLOW_THREADS: Final[int]  # 0 或 1（依赖系统）
# 缓冲区大小常量
BUFSIZE: L[8192]
# 剪切模式常量
CLIP: L[0]
# 包裹模式常量
WRAP: L[1]
# 抛出模式常量
RAISE: L[2]
# 最大维度常量
MAXDIMS: L[32]
# 共享边界标志常量
MAY_SHARE_BOUNDS: L[0]
# 共享精确标志常量
MAY_SHARE_EXACT: L[-1]
# 跟踪内存分配域标识常量
tracemalloc_domain: L[389047]

@overload
def empty_like(
    prototype: _ArrayType,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> _ArrayType: ...
@overload
def empty_like(
    prototype: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[_SCT]: ...
@overload
def empty_like(
    prototype: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...
@overload
def empty_like(
    # 返回与原型形状和数据类型相同的空数组
    prototype: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...
    prototype: Any,
    # 参数：prototype，可以是任意类型的对象，作为函数的原型或模板参数使用

    dtype: _DTypeLike[_SCT],
    # 参数：dtype，类型为 _DTypeLike[_SCT]，表示数据类型，通常用于指定数据的存储类型或格式

    order: _OrderKACF = ...,
    # 参数：order，类型为 _OrderKACF，表示数据在内存中的存储顺序，通常是 C、F 等

    subok: bool = ...,
    # 参数：subok，布尔类型，表示是否允许返回与输入类型不同的子类对象

    shape: None | _ShapeLike = ...,
    # 参数：shape，可以是 None 或 _ShapeLike 类型，表示数据的形状或维度

    *,
    # '*' 表示以下的参数只能以关键字参数的形式传递

    device: None | L["cpu"] = ...,
    # 参数：device，可以是 None 或 L["cpu"] 类型，表示数据所在的设备，如 CPU
# 定义一个类型注解，指明函数返回的类型为 NDArray[_SCT]
) -> NDArray[_SCT]: ...

# 定义 empty_like 函数的函数签名，并指定各个参数的类型和默认值
@overload
def empty_like(
    prototype: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...

# 定义 array 函数的函数签名，并指定各个参数的类型和默认值
@overload
def array(
    object: _ArrayType,
    dtype: None = ...,
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: L[True],
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> _ArrayType: ...

# 定义 array 函数的函数签名，并指定各个参数的类型和默认值
@overload
def array(
    object: _ArrayLike[_SCT],
    dtype: None = ...,
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...

# 定义 array 函数的函数签名，并指定各个参数的类型和默认值
@overload
def array(
    object: object,
    dtype: None = ...,
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 定义 array 函数的函数签名，并指定各个参数的类型和默认值
@overload
def array(
    object: Any,
    dtype: _DTypeLike[_SCT],
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...

# 定义 array 函数的函数签名，并指定各个参数的类型和默认值
@overload
def array(
    object: Any,
    dtype: DTypeLike,
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 定义 zeros 函数的函数签名，并指定各个参数的类型和默认值
@overload
def zeros(
    shape: _ShapeLike,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...

# 定义 zeros 函数的函数签名，并指定各个参数的类型和默认值
@overload
def zeros(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...

# 定义 zeros 函数的函数签名，并指定各个参数的类型和默认值
@overload
def zeros(
    shape: _ShapeLike,
    dtype: DTypeLike,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 定义 empty 函数的函数签名，并指定各个参数的类型和默认值
@overload
def empty(
    shape: _ShapeLike,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...

# 定义 empty 函数的函数签名，并指定各个参数的类型和默认值
@overload
def empty(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...

# 定义 empty 函数的函数签名，并指定各个参数的类型和默认值
@overload
def empty(
    shape: _ShapeLike,
    dtype: DTypeLike,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 定义 unravel_index 函数的函数签名，并指定各个参数的类型和默认值
@overload
def unravel_index(
    indices: _IntLike_co,
    shape: _ShapeLike,
    order: _OrderCF = ...,
) -> tuple[intp, ...]: ...

# 定义 unravel_index 函数的函数签名，并指定各个参数的类型和默认值
@overload
def unravel_index(
    indices: _ArrayLikeInt_co,
    shape: _ShapeLike,
    order: _OrderCF = ...,


# 定义一个变量order，类型为_OrderCF，初始值为...
# 定义了一个函数签名，接受任意数量的数组和数据类型，并返回它们的结果数据类型
def result_type(
    *arrays_and_dtypes: ArrayLike | DTypeLike,
) -> dtype[Any]: ...

# 利用类型重载，实现了矩阵点积运算，返回值类型与输入类型相同
@overload
def dot(a: ArrayLike, b: ArrayLike, out: None = ...) -> Any: ...
# 利用类型重载，实现了矩阵点积运算，可以将结果输出到指定数组中
@overload
def dot(a: ArrayLike, b: ArrayLike, out: _ArrayType) -> _ArrayType: ...

# 求解条件数组中满足条件的索引，并返回索引数组组成的元组
@overload
def where(
    condition: ArrayLike,
    /,
) -> tuple[NDArray[intp], ...]: ...
# 根据条件数组返回两个数组元素（根据条件选择的 x 或 y）中符合条件的元素组成的数组
@overload
def where(
    condition: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    /,
) -> NDArray[Any]: ...

# 对多个排序键进行词法排序，返回排序后的索引数组
def lexsort(
    keys: ArrayLike,
    axis: None | SupportsIndex = ...,
) -> Any: ...

# 判断是否可以将给定的数据类型或数组强制转换为目标数据类型
def can_cast(
    from_: ArrayLike | DTypeLike,
    to: DTypeLike,
    casting: None | _CastingKind = ...,
) -> bool: ...

# 返回可以容纳给定数组最小标量类型的数据类型
def min_scalar_type(
    a: ArrayLike, /,
) -> dtype[Any]: ...

# 根据输入的数组或数据类型，返回它们的结果数据类型
def result_type(
    *arrays_and_dtypes: ArrayLike | DTypeLike,
) -> dtype[Any]: ...

# 实现了对两个数组的向量点积运算，根据数组类型返回相应的数据类型
@overload
def vdot(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co, /) -> np.bool: ...  # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co, /) -> unsignedinteger[Any]: ...  # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, /) -> signedinteger[Any]: ...  # type: ignore[misc]
# 这里应该还有其他的重载定义，但未提供完整内容
# 定义了一个函数 vdot，计算两个数组的点积，返回值类型依赖于参数的类型
@overload
def vdot(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, /) -> floating[Any]: ...  # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, /) -> complexfloating[Any, Any]: ...  # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeTD64_co, b: _ArrayLikeTD64_co, /) -> timedelta64: ...
@overload
def vdot(a: _ArrayLikeObject_co, b: Any, /) -> Any: ...
@overload
def vdot(a: Any, b: _ArrayLikeObject_co, /) -> Any: ...

# 定义了一个函数 bincount，统计数组中每个整数出现的次数，并返回一个数组
def bincount(
    x: ArrayLike,
    /,
    weights: None | ArrayLike = ...,
    minlength: SupportsIndex = ...,
) -> NDArray[intp]: ...

# 定义了一个函数 copyto，将一个数组的内容复制到另一个数组
def copyto(
    dst: NDArray[Any],
    src: ArrayLike,
    casting: None | _CastingKind = ...,
    where: None | _ArrayLikeBool_co = ...,
) -> None: ...

# 定义了一个函数 putmask，根据掩码数组更新目标数组的值
def putmask(
    a: NDArray[Any],
    /,
    mask: _ArrayLikeBool_co,
    values: ArrayLike,
) -> None: ...

# 定义了一个函数 packbits，将整数数组中的每个元素转换为二进制位，并返回数组
def packbits(
    a: _ArrayLikeInt_co,
    /,
    axis: None | SupportsIndex = ...,
    bitorder: L["big", "little"] = ...,
) -> NDArray[uint8]: ...

# 定义了一个函数 unpackbits，将二进制位数组转换为整数数组，并返回数组
def unpackbits(
    a: _ArrayLike[uint8],
    /,
    axis: None | SupportsIndex = ...,
    count: None | SupportsIndex = ...,
    bitorder: L["big", "little"] = ...,
) -> NDArray[uint8]: ...

# 定义了一个函数 shares_memory，检查两个对象是否共享内存
def shares_memory(
    a: object,
    b: object,
    /,
    max_work: None | int = ...,
) -> bool: ...

# 定义了一个函数 may_share_memory，检查两个对象是否可能共享内存
def may_share_memory(
    a: object,
    b: object,
    /,
    max_work: None | int = ...,
) -> bool: ...

# 定义了一个函数 asarray，将输入转换为数组，支持多种参数和选项
@overload
def asarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asarray(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
@overload
def asarray(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asarray(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 定义了一个函数 asanyarray，将输入转换为任意类型的数组，支持多种参数和选项
@overload
def asanyarray(
    a: _ArrayType,  # 保留子类信息
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> _ArrayType: ...
@overload
def asanyarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asanyarray(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
    like: None | _SupportsArrayFunc = ...,


    # 声明一个变量 like，其类型可以是 None 或者 _SupportsArrayFunc 类型
# 返回类型标注为 `NDArray`，表示函数返回一个 NumPy 数组
@overload
def asanyarray(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
# 返回类型标注为 `NDArray[_SCT]`，表示函数返回一个指定类型的 NumPy 数组
@overload
def asanyarray(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 返回类型标注为 `NDArray[_SCT]`，表示函数返回一个指定类型的 NumPy 数组
@overload
def ascontiguousarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
# 返回类型标注为 `NDArray[Any]`，表示函数返回一个任意类型的 NumPy 数组
@overload
def ascontiguousarray(
    a: object,
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
# 返回类型标注为 `NDArray[_SCT]`，表示函数返回一个指定类型的 NumPy 数组
@overload
def ascontiguousarray(
    a: Any,
    dtype: _DTypeLike[_SCT],
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
# 返回类型标注为 `NDArray[Any]`，表示函数返回一个任意类型的 NumPy 数组
@overload
def ascontiguousarray(
    a: Any,
    dtype: DTypeLike,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 返回类型标注为 `NDArray[_SCT]`，表示函数返回一个指定类型的 NumPy 数组
@overload
def asfortranarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
# 返回类型标注为 `NDArray[Any]`，表示函数返回一个任意类型的 NumPy 数组
@overload
def asfortranarray(
    a: object,
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
# 返回类型标注为 `NDArray[_SCT]`，表示函数返回一个指定类型的 NumPy 数组
@overload
def asfortranarray(
    a: Any,
    dtype: _DTypeLike[_SCT],
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
# 返回类型标注为 `NDArray[Any]`，表示函数返回一个任意类型的 NumPy 数组
@overload
def asfortranarray(
    a: Any,
    dtype: DTypeLike,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 函数返回类型标注为 `dtype[Any]`，表示函数返回一个数据类型对象
def promote_types(__type1: DTypeLike, __type2: DTypeLike) -> dtype[Any]: ...

# 函数 `fromstring` 的注释解释 `sep` 参数是一个必选的参数，因为其默认值已经过时
@overload
def fromstring(
    string: str | bytes,
    dtype: None = ...,
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...
# 返回类型标注为 `NDArray[float64]`，表示函数返回一个 float64 类型的 NumPy 数组
@overload
def fromstring(
    string: str | bytes,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
# 返回类型标注为 `NDArray[Any]`，表示函数返回一个任意类型的 NumPy 数组
@overload
def fromstring(
    string: str | bytes,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 函数 `frompyfunc` 返回类型标注为 `ufunc`，表示函数返回一个通用函数对象
def frompyfunc(
    func: Callable[..., Any], /,
    nin: SupportsIndex,
    nout: SupportsIndex,
    *,
    identity: Any = ...,
) -> ufunc: ...

# 返回类型标注为 `NDArray[float64]`，表示函数返回一个 float64 类型的 NumPy 数组
@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: None = ...,
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...
# 返回类型标注为 `NDArray[_SCT]`，表示函数返回一个指定类型的 NumPy 数组
@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    # 定义一个命名关键字参数 `like`，类型可以是 None 或者 _SupportsArrayFunc 类型
    *,
    # 其他参数
    like: None | _SupportsArrayFunc = ...,
# 返回一个 NumPy 数组，从文件中读取数据，根据文件名或字节流、路径、或文件对象，以及数据类型和其他参数确定数据的格式和数量
@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 从迭代器中创建一个 NumPy 数组，根据给定的数据类型和可选的数据数量，以及其他参数确定数组的格式
@overload
def fromiter(
    iter: Iterable[Any],
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def fromiter(
    iter: Iterable[Any],
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 从缓冲区创建一个 NumPy 数组，根据给定的数据类型、缓冲区、数据数量和偏移量，以及其他参数确定数组的格式
@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: None = ...,
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...
@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 创建一个包含等差数列的 NumPy 数组，根据给定的起始点、终止点、步长和其他参数确定数组的格式和设备位置
@overload
def arange(  # type: ignore[misc]
    stop: _IntLike_co,
    /, *,
    dtype: None = ...,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def arange(  # type: ignore[misc]
    start: _IntLike_co,
    stop: _IntLike_co,
    step: _IntLike_co = ...,
    dtype: None = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def arange(  # type: ignore[misc]
    stop: _FloatLike_co,
    /, *,
    dtype: None = ...,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[floating[Any]]: ...
@overload
def arange(  # type: ignore[misc]
    start: _FloatLike_co,
    stop: _FloatLike_co,
    step: _FloatLike_co = ...,
    dtype: None = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[floating[Any]]: ...
@overload
def arange(
    stop: _TD64Like_co,
    /, *,
    dtype: None = ...,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[timedelta64]: ...
@overload
def arange(
    start: _TD64Like_co,
    stop: _TD64Like_co,
    step: _TD64Like_co = ...,
    dtype: None = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[timedelta64]: ...
@overload
def arange(  # both start and stop must always be specified for datetime64
    start: datetime64,
    stop: datetime64,
    step: datetime64 = ...,
    dtype: None = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[datetime64]: ...
# 定义了多个重载版本的 arange 函数，用于创建一维数组范围
@overload
def arange(
    stop: Any,
    /, *,
    dtype: _DTypeLike[_SCT],
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...

# 定义了多个重载版本的 arange 函数，用于创建一维数组范围
@overload
def arange(
    start: Any,
    stop: Any,
    step: Any = ...,
    dtype: _DTypeLike[_SCT] = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...

# 定义了多个重载版本的 arange 函数，用于创建一维数组范围
@overload
def arange(
    stop: Any, /,
    *,
    dtype: DTypeLike,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 定义了多个重载版本的 arange 函数，用于创建一维数组范围
@overload
def arange(
    start: Any,
    stop: Any,
    step: Any = ...,
    dtype: DTypeLike = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 定义了 datetime_data 函数，返回一个包含数据类型和整数的元组
def datetime_data(
    dtype: str | _DTypeLike[datetime64] | _DTypeLike[timedelta64], /,
) -> tuple[str, int]: ...

# 定义了多个重载版本的 busday_count 函数，用于计算工作日天数
@overload
def busday_count(  # type: ignore[misc]
    begindates: _ScalarLike_co | dt.date,
    enddates: _ScalarLike_co | dt.date,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> int_: ...

# 定义了多个重载版本的 busday_count 函数，用于计算工作日天数
@overload
def busday_count(  # type: ignore[misc]
    begindates: ArrayLike | dt.date | _NestedSequence[dt.date],
    enddates: ArrayLike | dt.date | _NestedSequence[dt.date],
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[int_]: ...

# 定义了多个重载版本的 busday_count 函数，用于计算工作日天数
@overload
def busday_count(
    begindates: ArrayLike | dt.date | _NestedSequence[dt.date],
    enddates: ArrayLike | dt.date | _NestedSequence[dt.date],
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...

# 定义了多个重载版本的 busday_offset 函数，用于计算工作日的偏移日期
@overload
def busday_offset(  # type: ignore[misc]
    dates: datetime64 | dt.date,
    offsets: _TD64Like_co | dt.timedelta,
    roll: L["raise"] = ...,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> datetime64: ...

# 定义了多个重载版本的 busday_offset 函数，用于计算工作日的偏移日期
@overload
def busday_offset(  # type: ignore[misc]
    dates: _ArrayLike[datetime64] | dt.date | _NestedSequence[dt.date],
    offsets: _ArrayLikeTD64_co | dt.timedelta | _NestedSequence[dt.timedelta],
    roll: L["raise"] = ...,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[datetime64]: ...

# 定义了多个重载版本的 busday_offset 函数，用于计算工作日的偏移日期
@overload
def busday_offset(  # type: ignore[misc]
    dates: _ArrayLike[datetime64] | dt.date | _NestedSequence[dt.date],
    # dates参数接受的类型注解，可以是datetime64数组，也可以是单个日期或日期嵌套序列

    offsets: _ArrayLikeTD64_co | dt.timedelta | _NestedSequence[dt.timedelta],
    # offsets参数接受的类型注解，可以是timedelta数组，也可以是单个时间间隔或时间间隔嵌套序列

    roll: L["raise"] = ...,
    # roll参数，默认值为"raise"，表示在日期超出范围时如何处理的选项

    weekmask: ArrayLike = ...,
    # weekmask参数，接受类似数组的对象，用于定义一周中哪些天是有效的

    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    # holidays参数，可选值为None或者日期数组，或单个日期，或日期嵌套序列，表示假日日期

    busdaycal: None | busdaycalendar = ...,
    # busdaycal参数，可选值为None或busdaycalendar对象，用于定义工作日的规则

    out: _ArrayType = ...,
    # out参数的类型注解，表示返回结果的数组类型
# 定义一个函数签名，用于计算工作日偏移量，返回一个数组
@overload
def busday_offset(
    dates: _ScalarLike_co | dt.date,  # 接受标量或日期对象作为输入的日期参数
    offsets: _ScalarLike_co | dt.timedelta,  # 接受标量或时间间隔对象作为输入的偏移量参数
    roll: _RollKind,  # 指定工作日偏移的滚动方式
    weekmask: ArrayLike = ...,  # 工作日掩码数组，默认为省略值
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,  # 节假日日期或数组，默认为省略值
    busdaycal: None | busdaycalendar = ...,  # 工作日历对象或省略值
    out: None = ...,  # 输出数组的类型，默认为省略值
) -> datetime64:  # 返回值为日期时间数组
    ...

@overload
def busday_offset(
    dates: ArrayLike | dt.date | _NestedSequence[dt.date],  # 接受数组、日期或日期嵌套序列作为输入的日期参数
    offsets: ArrayLike | dt.timedelta | _NestedSequence[dt.timedelta],  # 接受数组、时间间隔或时间间隔嵌套序列作为输入的偏移量参数
    roll: _RollKind,  # 指定工作日偏移的滚动方式
    weekmask: ArrayLike = ...,  # 工作日掩码数组，默认为省略值
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,  # 节假日日期或数组，默认为省略值
    busdaycal: None | busdaycalendar = ...,  # 工作日历对象或省略值
    out: None = ...,  # 输出数组的类型，默认为省略值
) -> NDArray[datetime64]:  # 返回值为日期时间数组的 NumPy 数组
    ...

@overload
def busday_offset(
    dates: ArrayLike | dt.date | _NestedSequence[dt.date],  # 接受数组、日期或日期嵌套序列作为输入的日期参数
    offsets: ArrayLike | dt.timedelta | _NestedSequence[dt.timedelta],  # 接受数组、时间间隔或时间间隔嵌套序列作为输入的偏移量参数
    roll: _RollKind,  # 指定工作日偏移的滚动方式
    weekmask: ArrayLike = ...,  # 工作日掩码数组，默认为省略值
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,  # 节假日日期或数组，默认为省略值
    busdaycal: None | busdaycalendar = ...,  # 工作日历对象或省略值
    out: _ArrayType = ...,  # 输出数组的类型，默认为省略值
) -> _ArrayType:  # 返回值为自定义数组类型
    ...

# 定义一个函数签名，用于检查日期是否为工作日，返回布尔值或数组
@overload
def is_busday(
    dates: _ScalarLike_co | dt.date,  # 接受标量或日期对象作为输入的日期参数
    weekmask: ArrayLike = ...,  # 工作日掩码数组，默认为省略值
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,  # 节假日日期或数组，默认为省略值
    busdaycal: None | busdaycalendar = ...,  # 工作日历对象或省略值
    out: None = ...,  # 输出数组的类型，默认为省略值
) -> np.bool:  # 返回值为布尔值
    ...

@overload
def is_busday(
    dates: ArrayLike | _NestedSequence[dt.date],  # 接受数组或日期嵌套序列作为输入的日期参数
    weekmask: ArrayLike = ...,  # 工作日掩码数组，默认为省略值
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,  # 节假日日期或数组，默认为省略值
    busdaycal: None | busdaycalendar = ...,  # 工作日历对象或省略值
    out: None = ...,  # 输出数组的类型，默认为省略值
) -> NDArray[np.bool]:  # 返回值为布尔值数组的 NumPy 数组
    ...

@overload
def is_busday(
    dates: ArrayLike | _NestedSequence[dt.date],  # 接受数组或日期嵌套序列作为输入的日期参数
    weekmask: ArrayLike = ...,  # 工作日掩码数组，默认为省略值
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,  # 节假日日期或数组，默认为省略值
    busdaycal: None | busdaycalendar = ...,  # 工作日历对象或省略值
    out: _ArrayType = ...,  # 输出数组的类型，默认为省略值
) -> _ArrayType:  # 返回值为自定义数组类型
    ...

# 定义一个函数签名，用于将日期时间对象转换为字符串
@overload
def datetime_as_string(
    arr: datetime64 | dt.date,  # 接受日期时间或日期对象作为输入的数组
    unit: None | L["auto"] | _UnitKind = ...,  # 时间单位或自动选择，或者省略值
    timezone: L["naive", "UTC", "local"] | dt.tzinfo = ...,  # 时区信息，包括本地时间或者 UTC 时间，或者省略值
    casting: _CastingKind = ...,  # 转换方式，包括精确或粗略，或者省略值
) -> str_:  # 返回值为字符串
    ...

@overload
def datetime_as_string(
    arr: _ArrayLikeDT64_co | _NestedSequence[dt.date],  # 接受日期时间数组或日期嵌套序列作为输入的数组
    unit: None | L["auto"] | _UnitKind = ...,  # 时间单位或自动选择，或者省略值
    timezone: L["naive", "UTC", "local"] | dt.tzinfo = ...,  # 时区信息，包括本地时间或者 UTC 时间，或者省略值
    casting: _CastingKind = ...,  # 转换方式，包括精确或粗略，或者省略值
) -> NDArray[str_]:  # 返回值为字符串数组的 NumPy 数组
    ...

# 定义一个函数签名，用于比较字符数组，返回布尔值数组
@overload
def compare_chararrays(
    a1: _ArrayLikeStr_co,  # 接受字符串数组或字符串嵌套序列作为输入的第一个数组
    a2: _ArrayLikeStr_co,  # 接受字符串数组或字符串嵌套序列作为输入的第二个数组
    cmp: L["<", "<=", "==", ">=", ">", "!="],  # 比较运算符
    rstrip: bool,  # 是否在比较前去除字符串末尾的空白字符
) -> NDArray[np.bool]:  # 返回值为布尔值数组的 NumPy 数组
    ...

@overload
def compare_chararrays(
    a1: _ArrayLikeBytes_co,  # 接受字节串数组或字节串嵌套序列作为输入的第一个数组
    a2: _ArrayLikeBytes_co,  # 接受字节串数组或字节串嵌套序列作为输入的第二个数组
    cmp: L["<", "<=", "==", ">=", ">", "!="],  # 比较运算符
    rstrip: bool,  # 是否在比较前去除字符串末尾的空白字符
) -> NDArray[np.bool]:  # 返回值为布尔值数组的 NumPy 数组
    ...

# 定义一个函数，用于向对象添加文档字符串
def add_docstring(obj: Callable[..., Any], docstring: str, /) ->
    # "F", "FORTRAN"：数组的存储顺序被视为 Fortran 风格
    # "F_CONTIGUOUS"：数组在内存中是以 Fortran 连续（列优先）的方式存储
    # "W", "WRITEABLE"：数组可以被写入
    # "B", "BEHAVED"：指定数组的行为符合某些规范
    # "O", "OWNDATA"：数组拥有它自己的数据并负责管理它
    # "A", "ALIGNED"：数据以正确的边界存储
    # "X", "WRITEBACKIFCOPY"：在副本写回到原始数组时允许写回
    # "CA", "CARRAY"：数组可被解释为 C 风格数组
    # "FA", "FARRAY"：数组可被解释为 Fortran 风格数组
    # "FNC"：表示按照 Fortran 的命名约定
    # "FORC"：表示按照 C 的命名约定
# 定义一个列表，包含用于设置项的键值对，每个键代表一个特定的标志，对应的值是人类可读的描述
_SetItemKeys = [
    "A", "ALIGNED",               # "A"代表"ALIGNED"，表示数据存储是否对齐
    "W", "WRITEABLE",             # "W"代表"WRITEABLE"，表示数据是否可写
    "X", "WRITEBACKIFCOPY",       # "X"代表"WRITEBACKIFCOPY"，表示是否在复制时写回
]

# 定义一个装饰器，使得类 flagsobj 变为不可继承的最终类
@final
class flagsobj:
    __hash__: ClassVar[None]  # type: ignore[assignment]  # 声明 __hash__ 为不可变类型，并禁止类型检查

    aligned: bool               # 表示是否对齐数据
    # NOTE: deprecated          # 注意：已弃用的注释
    # updateifcopy: bool        # 更新时是否复制数据（已弃用）
    writeable: bool             # 表示数据是否可写
    writebackifcopy: bool       # 表示是否在复制时写回数据

    @property
    def behaved(self) -> bool: ...  # 未指定行为的占位符属性
    @property
    def c_contiguous(self) -> bool: ...  # 判断是否 C 连续的占位符属性
    @property
    def carray(self) -> bool: ...  # 判断是否为 C 数组的占位符属性
    @property
    def contiguous(self) -> bool: ...  # 判断是否为连续数组的占位符属性
    @property
    def f_contiguous(self) -> bool: ...  # 判断是否 F 连续的占位符属性
    @property
    def farray(self) -> bool: ...  # 判断是否为 F 数组的占位符属性
    @property
    def fnc(self) -> bool: ...  # 未指定 FNC 的占位符属性
    @property
    def forc(self) -> bool: ...  # 未指定 FORC 的占位符属性
    @property
    def fortran(self) -> bool: ...  # 判断是否为 Fortran 的占位符属性
    @property
    def num(self) -> int: ...  # 返回一个整数的占位符属性
    @property
    def owndata(self) -> bool: ...  # 判断是否为自有数据的占位符属性

    def __getitem__(self, key: _GetItemKeys) -> bool: ...  # 用于按键获取值的占位符方法
    def __setitem__(self, key: _SetItemKeys, value: bool) -> None: ...  # 用于按键设置值的占位符方法

# 定义一个函数，用于生成多个迭代器对象的元组，支持多种参数配置
def nested_iters(
    op: ArrayLike | Sequence[ArrayLike],  # 输入操作数或其序列的类型提示
    axes: Sequence[Sequence[SupportsIndex]],  # 操作的轴序列，每个轴可能支持的索引类型
    flags: None | Sequence[_NDIterFlagsKind] = ...,  # 迭代器的标志序列或 None
    op_flags: None | Sequence[Sequence[_NDIterOpFlagsKind]] = ...,  # 操作标志的序列或 None
    op_dtypes: DTypeLike | Sequence[DTypeLike] = ...,  # 操作数据类型或其序列
    order: _OrderKACF = ...,  # 数据的存储顺序
    casting: _CastingKind = ...,  # 类型转换的规则
    buffersize: SupportsIndex = ...,  # 缓冲区大小，支持索引类型
) -> tuple[nditer, ...]: ...  # 返回一个迭代器对象的元组
```