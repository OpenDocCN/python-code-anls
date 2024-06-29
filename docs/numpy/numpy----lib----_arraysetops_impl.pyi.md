# `D:\src\scipysrc\numpy\numpy\lib\_arraysetops_impl.pyi`

```
# 导入必要的类型和函数声明
from typing import (
    Any,
    Generic,
    Literal as L,
    NamedTuple,
    overload,
    SupportsIndex,
    TypeVar,
)

# 导入 numpy 库，并选择性地导入其中的类型和函数
import numpy as np
from numpy import (
    generic,
    number,
    ushort,
    ubyte,
    uintc,
    uint,
    ulonglong,
    short,
    int8,
    byte,
    intc,
    int_,
    intp,
    longlong,
    half,
    single,
    double,
    longdouble,
    csingle,
    cdouble,
    clongdouble,
    timedelta64,
    datetime64,
    object_,
    str_,
    bytes_,
    void,
)

# 导入 numpy 内部类型声明
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeDT64_co,
    _ArrayLikeTD64_co,
    _ArrayLikeObject_co,
    _ArrayLikeNumber_co,
)

# 设置类型变量 _SCT，表示支持的 numpy 泛型类型
_SCT = TypeVar("_SCT", bound=generic)
# 设置类型变量 _NumberType，表示支持的数字类型
_NumberType = TypeVar("_NumberType", bound=number[Any])

# 设置类型变量 _SCTNoCast，用于避免意外转换为抽象数据类型
# 包含了所有明确允许的数据类型，防止错误转换
_SCTNoCast = TypeVar(
    "_SCTNoCast",
    np.bool,
    ushort,
    ubyte,
    uintc,
    uint,
    ulonglong,
    short,
    byte,
    intc,
    int_,
    longlong,
    half,
    single,
    double,
    longdouble,
    csingle,
    cdouble,
    clongdouble,
    timedelta64,
    datetime64,
    object_,
    str_,
    bytes_,
    void,
)

# 定义命名元组 UniqueAllResult，用于存储 unique 函数的返回结果
class UniqueAllResult(NamedTuple, Generic[_SCT]):
    values: NDArray[_SCT]           # 存储唯一值数组
    indices: NDArray[intp]          # 存储唯一值在原始数组中的索引数组
    inverse_indices: NDArray[intp]  # 存储原始数组在唯一值数组中的索引数组
    counts: NDArray[intp]           # 存储每个唯一值在原始数组中出现的次数数组

# 定义命名元组 UniqueCountsResult，用于存储 unique 函数的返回结果
class UniqueCountsResult(NamedTuple, Generic[_SCT]):
    values: NDArray[_SCT]           # 存储唯一值数组
    counts: NDArray[intp]           # 存储每个唯一值在原始数组中出现的次数数组

# 定义命名元组 UniqueInverseResult，用于存储 unique 函数的返回结果
class UniqueInverseResult(NamedTuple, Generic[_SCT]):
    values: NDArray[_SCT]           # 存储唯一值数组
    inverse_indices: NDArray[intp]  # 存储原始数组在唯一值数组中的索引数组

# 设置 __all__ 变量，指定导入模块时包含的公开名称列表
__all__: list[str]

# 定义 ediff1d 函数的多态重载，计算数组 ary 中相邻元素的差异
@overload
def ediff1d(
    ary: _ArrayLikeBool_co,
    to_end: None | ArrayLike = ...,
    to_begin: None | ArrayLike = ...,
) -> NDArray[int8]: ...

@overload
def ediff1d(
    ary: _ArrayLike[_NumberType],
    to_end: None | ArrayLike = ...,
    to_begin: None | ArrayLike = ...,
) -> NDArray[_NumberType]: ...

@overload
def ediff1d(
    ary: _ArrayLikeNumber_co,
    to_end: None | ArrayLike = ...,
    to_begin: None | ArrayLike = ...,
) -> NDArray[Any]: ...

@overload
def ediff1d(
    ary: _ArrayLikeDT64_co | _ArrayLikeTD64_co,
    to_end: None | ArrayLike = ...,
    to_begin: None | ArrayLike = ...,
) -> NDArray[timedelta64]: ...

@overload
def ediff1d(
    ary: _ArrayLikeObject_co,
    to_end: None | ArrayLike = ...,
    to_begin: None | ArrayLike = ...,
) -> NDArray[object_]: ...

# 定义 unique 函数的多态重载，返回数组 ar 中的唯一值及相关信息
@overload
def unique(
    ar: _ArrayLike[_SCT],
    return_index: L[False] = ...,
    return_inverse: L[False] = ...,
    return_counts: L[False] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> NDArray[_SCT]: ...

@overload
def unique(
    ar: ArrayLike,
    return_index: L[False] = ...,
    return_inverse: L[False] = ...,
    return_counts: L[False] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> NDArray[Any]: ...
    # 定义一个名为 `return_counts` 的参数，其类型是 `L[False]`，初始值未指定
    return_counts: L[False] = ...,
    # 定义一个名为 `axis` 的参数，其类型可以是 `None` 或者支持索引的类型，初始值未指定
    axis: None | SupportsIndex = ...,
    # `equal_nan` 是一个命名关键字参数，其类型为布尔值，默认值未指定，表示是否比较 NaN（Not a Number）
    *,
    equal_nan: bool = ...,
@overload
# 定义函数重载，处理返回唯一元素的情况，并返回元素及其索引或反向索引
def unique(
    ar: _ArrayLike[_SCT],
    return_index: L[True] = ...,
    return_inverse: L[False] = ...,
    return_counts: L[False] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[_SCT], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素的情况，并返回元素及其索引或反向索引
def unique(
    ar: ArrayLike,
    return_index: L[True] = ...,
    return_inverse: L[False] = ...,
    return_counts: L[False] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[Any], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素的情况，并返回元素及其索引或反向索引
def unique(
    ar: _ArrayLike[_SCT],
    return_index: L[False] = ...,
    return_inverse: L[True] = ...,
    return_counts: L[False] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[_SCT], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素的情况，并返回元素及其索引或反向索引
def unique(
    ar: ArrayLike,
    return_index: L[False] = ...,
    return_inverse: L[True] = ...,
    return_counts: L[False] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[Any], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素的情况，并返回元素及其计数
def unique(
    ar: _ArrayLike[_SCT],
    return_index: L[False] = ...,
    return_inverse: L[False] = ...,
    return_counts: L[True] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[_SCT], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素的情况，并返回元素及其计数
def unique(
    ar: ArrayLike,
    return_index: L[False] = ...,
    return_inverse: L[False] = ...,
    return_counts: L[True] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[Any], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素及其索引及反向索引的情况
def unique(
    ar: _ArrayLike[_SCT],
    return_index: L[True] = ...,
    return_inverse: L[True] = ...,
    return_counts: L[False] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[_SCT], NDArray[intp], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素及其索引及反向索引的情况
def unique(
    ar: ArrayLike,
    return_index: L[True] = ...,
    return_inverse: L[True] = ...,
    return_counts: L[False] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[Any], NDArray[intp], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素及其索引及反向索引的情况
def unique(
    ar: _ArrayLike[_SCT],
    return_index: L[True] = ...,
    return_inverse: L[False] = ...,
    return_counts: L[True] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[_SCT], NDArray[intp], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素及其索引及反向索引的情况
def unique(
    ar: ArrayLike,
    return_index: L[True] = ...,
    return_inverse: L[False] = ...,
    return_counts: L[True] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[Any], NDArray[intp], NDArray[intp]]: ...

@overload
# 定义函数重载，处理返回唯一元素及其索引及反向索引的情况
def unique(
    ar: _ArrayLike[_SCT],
    return_index: L[False] = ...,
    return_inverse: L[True] = ...,
    return_counts: L[True] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[_SCT], NDArray[intp], NDArray[intp]]: ...

# 最后一个函数重载定义，处理返回唯一元素及其索引及反向索引的情况
# 定义 unique 函数的多个函数签名的类型提示，每个函数签名返回一个元组，包含不同的 NDArray 数组
@overload
def unique(
    ar: ArrayLike,
    return_index: L[False] = ...,
    return_inverse: L[True] = ...,
    return_counts: L[True] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[Any], NDArray[intp], NDArray[intp]]: ...

@overload
def unique(
    ar: _ArrayLike[_SCT],
    return_index: L[True] = ...,
    return_inverse: L[True] = ...,
    return_counts: L[True] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[_SCT], NDArray[intp], NDArray[intp], NDArray[intp]]: ...

@overload
def unique(
    ar: ArrayLike,
    return_index: L[True] = ...,
    return_inverse: L[True] = ...,
    return_counts: L[True] = ...,
    axis: None | SupportsIndex = ...,
    *,
    equal_nan: bool = ...,
) -> tuple[NDArray[Any], NDArray[intp], NDArray[intp], NDArray[intp]]: ...

# 定义 unique_all 函数的多个函数签名的类型提示，每个函数签名返回 UniqueAllResult 对象
@overload
def unique_all(
    x: _ArrayLike[_SCT], /
) -> UniqueAllResult[_SCT]: ...

@overload
def unique_all(
    x: ArrayLike, /
) -> UniqueAllResult[Any]: ...

# 定义 unique_counts 函数的多个函数签名的类型提示，每个函数签名返回 UniqueCountsResult 对象
@overload
def unique_counts(
    x: _ArrayLike[_SCT], /
) -> UniqueCountsResult[_SCT]: ...

@overload
def unique_counts(
    x: ArrayLike, /
) -> UniqueCountsResult[Any]: ...

# 定义 unique_inverse 函数的多个函数签名的类型提示，每个函数签名返回 UniqueInverseResult 对象
@overload
def unique_inverse(x: _ArrayLike[_SCT], /) -> UniqueInverseResult[_SCT]: ...

@overload
def unique_inverse(x: ArrayLike, /) -> UniqueInverseResult[Any]: ...

# 定义 unique_values 函数的多个函数签名的类型提示，每个函数签名返回 NDArray 数组
@overload
def unique_values(x: _ArrayLike[_SCT], /) -> NDArray[_SCT]: ...

@overload
def unique_values(x: ArrayLike, /) -> NDArray[Any]: ...

# 定义 intersect1d 函数的多个函数签名的类型提示，每个函数签名返回 NDArray 数组或者包含 NDArray 数组的元组
@overload
def intersect1d(
    ar1: _ArrayLike[_SCTNoCast],
    ar2: _ArrayLike[_SCTNoCast],
    assume_unique: bool = ...,
    return_indices: L[False] = ...,
) -> NDArray[_SCTNoCast]: ...

@overload
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = ...,
    return_indices: L[False] = ...,
) -> NDArray[Any]: ...

@overload
def intersect1d(
    ar1: _ArrayLike[_SCTNoCast],
    ar2: _ArrayLike[_SCTNoCast],
    assume_unique: bool = ...,
    return_indices: L[True] = ...,
) -> tuple[NDArray[_SCTNoCast], NDArray[intp], NDArray[intp]]: ...

@overload
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = ...,
    return_indices: L[True] = ...,
) -> tuple[NDArray[Any], NDArray[intp], NDArray[intp]]: ...

# 定义 setxor1d 函数的多个函数签名的类型提示，每个函数签名返回 NDArray 数组
@overload
def setxor1d(
    ar1: _ArrayLike[_SCTNoCast],
    ar2: _ArrayLike[_SCTNoCast],
    assume_unique: bool = ...,
) -> NDArray[_SCTNoCast]: ...

@overload
def setxor1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = ...,
) -> NDArray[Any]: ...

# 定义 isin 函数的类型提示，返回一个布尔类型的 NDArray 数组
def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: bool = ...,
    invert: bool = ...,
    *,
    kind: None | str = ...,
) -> NDArray[np.bool]: ...

# 定义 union1d 函数的多个函数签名的类型提示，每个函数签名返回 NDArray 数组
@overload
def union1d(
    ar1: _ArrayLike[_SCTNoCast],
    ar2: _ArrayLike[_SCTNoCast],
) -> NDArray[_SCTNoCast]: ...

@overload
def union1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
) -> NDArray[Any]: ...
# 定义了函数重载，用于计算两个数组的差集
@overload
def setdiff1d(
    ar1: _ArrayLike[_SCTNoCast],  # 第一个数组参数，接受类型为 _ArrayLike[_SCTNoCast] 的输入
    ar2: _ArrayLike[_SCTNoCast],  # 第二个数组参数，接受类型为 _ArrayLike[_SCTNoCast] 的输入
    assume_unique: bool = ...,    # 布尔类型的可选参数，用于指定输入数组是否唯一
) -> NDArray[_SCTNoCast]:         # 返回类型为 _SCTNoCast 类型的 NumPy 数组

@overload
def setdiff1d(
    ar1: ArrayLike,               # 第一个数组参数，接受任意类型的输入
    ar2: ArrayLike,               # 第二个数组参数，接受任意类型的输入
    assume_unique: bool = ...,    # 布尔类型的可选参数，用于指定输入数组是否唯一
) -> NDArray[Any]:                # 返回类型为任意类型的 NumPy 数组
```