# `.\numpy\numpy\_core\fromnumeric.pyi`

```py
from collections.abc import Sequence  # 导入 Sequence 抽象基类
from typing import Any, overload, TypeVar, Literal, SupportsIndex  # 导入类型相关模块

import numpy as np  # 导入 NumPy 库
from numpy import (  # 导入 NumPy 的部分子模块和类型
    number,
    uint64,
    int_,
    int64,
    intp,
    float16,
    floating,
    complexfloating,
    object_,
    generic,
    _OrderKACF,
    _OrderACF,
    _ModeKind,
    _PartitionKind,
    _SortKind,
    _SortSide,
    _CastingKind,
)
from numpy._typing import (  # 导入 NumPy 的类型注解
    DTypeLike,
    _DTypeLike,
    ArrayLike,
    _ArrayLike,
    NDArray,
    _ShapeLike,
    _Shape,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
    _IntLike_co,
    _BoolLike_co,
    _ComplexLike_co,
    _NumberLike_co,
    _ScalarLike_co,
)

_SCT = TypeVar("_SCT", bound=generic)  # 定义类型变量 _SCT
_SCT_uifcO = TypeVar("_SCT_uifcO", bound=number[Any] | object_)  # 定义类型变量 _SCT_uifcO
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])  # 定义类型变量 _ArrayType

__all__: list[str]  # 定义 __all__ 变量为字符串列表

@overload
def take(
    a: _ArrayLike[_SCT],
    indices: _IntLike_co,
    axis: None = ...,
    out: None = ...,
    mode: _ModeKind = ...,
) -> _SCT: ...
@overload
def take(
    a: ArrayLike,
    indices: _IntLike_co,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    mode: _ModeKind = ...,
) -> Any: ...
@overload
def take(
    a: _ArrayLike[_SCT],
    indices: _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    mode: _ModeKind = ...,
) -> NDArray[_SCT]: ...
@overload
def take(
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    mode: _ModeKind = ...,
) -> NDArray[Any]: ...
@overload
def take(
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
    out: _ArrayType = ...,
    mode: _ModeKind = ...,
) -> _ArrayType: ...
# take 函数的重载定义，用于从数组中获取指定索引的元素或子数组

@overload
def reshape(
    a: _ArrayLike[_SCT],
    newshape: _ShapeLike,
    order: _OrderACF = ...,
    copy: None | bool = ...,
) -> NDArray[_SCT]: ...
@overload
def reshape(
    a: ArrayLike,
    newshape: _ShapeLike,
    order: _OrderACF = ...,
    copy: None | bool = ...,
) -> NDArray[Any]: ...
# reshape 函数的重载定义，用于改变数组的形状

@overload
def choose(
    a: _IntLike_co,
    choices: ArrayLike,
    out: None = ...,
    mode: _ModeKind = ...,
) -> Any: ...
@overload
def choose(
    a: _ArrayLikeInt_co,
    choices: _ArrayLike[_SCT],
    out: None = ...,
    mode: _ModeKind = ...,
) -> NDArray[_SCT]: ...
@overload
def choose(
    a: _ArrayLikeInt_co,
    choices: ArrayLike,
    out: None = ...,
    mode: _ModeKind = ...,
) -> NDArray[Any]: ...
@overload
def choose(
    a: _ArrayLikeInt_co,
    choices: ArrayLike,
    out: _ArrayType = ...,
    mode: _ModeKind = ...,
) -> _ArrayType: ...
# choose 函数的重载定义，用于根据索引数组从选择列表中获取元素或子数组

@overload
def repeat(
    a: _ArrayLike[_SCT],
    repeats: _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
) -> NDArray[_SCT]: ...
@overload
def repeat(
    a: ArrayLike,
    repeats: _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
) -> NDArray[Any]: ...
# repeat 函数的重载定义，用于在指定轴上重复数组元素

def put(
    a: NDArray[Any],


以上是对给定代码段的注释，每个函数和类型变量都有详细的解释说明其作用和参数用途。
    ind: _ArrayLikeInt_co,  # 定义一个参数 ind，类型为 _ArrayLikeInt_co，表示这是一个约束为整数数组的类型
    v: ArrayLike,  # 定义一个参数 v，类型为 ArrayLike，表示这是一个类似数组的对象
    mode: _ModeKind = ...,  # 定义一个参数 mode，类型为 _ModeKind，初始化为省略号（待后续具体赋值）
# 定义 swapaxes 函数的类型签名，用于交换数组的两个轴
@overload
def swapaxes(
    a: _ArrayLike[_SCT],
    axis1: SupportsIndex,
    axis2: SupportsIndex,
) -> NDArray[_SCT]: ...

# 另一种 swapaxes 函数的类型签名，支持交换任意类型的数组的两个轴
@overload
def swapaxes(
    a: ArrayLike,
    axis1: SupportsIndex,
    axis2: SupportsIndex,
) -> NDArray[Any]: ...

# 定义 transpose 函数的类型签名，用于数组转置操作
@overload
def transpose(
    a: _ArrayLike[_SCT],
    axes: None | _ShapeLike = ...
) -> NDArray[_SCT]: ...

# 另一种 transpose 函数的类型签名，支持对任意类型的数组进行转置
@overload
def transpose(
    a: ArrayLike,
    axes: None | _ShapeLike = ...
) -> NDArray[Any]: ...

# 定义 matrix_transpose 函数的类型签名，用于矩阵转置
@overload
def matrix_transpose(x: _ArrayLike[_SCT]) -> NDArray[_SCT]: ...

# 另一种 matrix_transpose 函数的类型签名，支持对任意类型的矩阵进行转置
@overload
def matrix_transpose(x: ArrayLike) -> NDArray[Any]: ...

# 定义 partition 函数的类型签名，用于对数组进行分区操作
@overload
def partition(
    a: _ArrayLike[_SCT],
    kth: _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
    kind: _PartitionKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[_SCT]: ...

# 另一种 partition 函数的类型签名，支持对任意类型的数组进行分区
@overload
def partition(
    a: ArrayLike,
    kth: _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
    kind: _PartitionKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[Any]: ...

# 定义 argpartition 函数的类型签名，用于对数组进行分区并返回索引
def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
    kind: _PartitionKind = ...,
    order: None | str | Sequence[str] = ...,
) -> NDArray[intp]: ...

# 定义 sort 函数的类型签名，用于对数组进行排序操作
@overload
def sort(
    a: _ArrayLike[_SCT],
    axis: None | SupportsIndex = ...,
    kind: None | _SortKind = ...,
    order: None | str | Sequence[str] = ...,
    *,
    stable: None | bool = ...,
) -> NDArray[_SCT]: ...

# 另一种 sort 函数的类型签名，支持对任意类型的数组进行排序
@overload
def sort(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    kind: None | _SortKind = ...,
    order: None | str | Sequence[str] = ...,
    *,
    stable: None | bool = ...,
) -> NDArray[Any]: ...

# 定义 argsort 函数的类型签名，用于对数组进行排序并返回索引
def argsort(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    kind: None | _SortKind = ...,
    order: None | str | Sequence[str] = ...,
    *,
    stable: None | bool = ...,
) -> NDArray[intp]: ...

# 定义 argmax 函数的类型签名，用于找出数组中最大值的索引
@overload
def argmax(
    a: ArrayLike,
    axis: None = ...,
    out: None = ...,
    *,
    keepdims: Literal[False] = ...,
) -> intp: ...

# 另一种 argmax 函数的类型签名，支持对任意类型的数组找出最大值的索引
@overload
def argmax(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    *,
    keepdims: bool = ...,
) -> Any: ...

# 另一种 argmax 函数的类型签名，支持对任意类型的数组找出最大值的索引并保持维度
@overload
def argmax(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    out: _ArrayType = ...,
    *,
    keepdims: bool = ...,
) -> _ArrayType: ...

# 定义 argmin 函数的类型签名，用于找出数组中最小值的索引
@overload
def argmin(
    a: ArrayLike,
    axis: None = ...,
    out: None = ...,
    *,
    keepdims: Literal[False] = ...,
) -> intp: ...

# 另一种 argmin 函数的类型签名，支持对任意类型的数组找出最小值的索引
@overload
def argmin(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    *,
    keepdims: bool = ...,
) -> Any: ...

# 另一种 argmin 函数的类型签名，支持对任意类型的数组找出最小值的索引并保持维度
@overload
def argmin(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    out: _ArrayType = ...,
    *,
    keepdims: bool = ...,
) -> _ArrayType: ...

# 定义 searchsorted 函数的类型签名，用于在有序数组中寻找插入值的位置
@overload
def searchsorted(
    a: ArrayLike,
    v: _ScalarLike_co,
    side: _SortSide = ...,
    sorter: None | _ArrayLikeInt_co = ...,  # 1D int array
) -> intp: ...

# 另一种 searchsorted 函数的类型签名，支持在任意类型的有序数组中寻找插入值的位置
@overload
def searchsorted(
    a: ArrayLike,
    v: _ScalarLike_co,
    side: _SortSide = ...,
    sorter: None | _ArrayLikeInt_co = ...,
) -> intp: ...
    v: ArrayLike,
    # v 是一个类似数组的对象，可能是数组或类似数组的结构

    side: _SortSide = ...,
    # side 是一个 _SortSide 类型的变量，默认为未指定值

    sorter: None | _ArrayLikeInt_co = ...,
    # sorter 是一个可选的 _ArrayLikeInt_co 类型的变量，表示可以是空值或整数数组
    # 该数组应为一维整数数组
# 定义一个函数签名，用于接收一个 NDArray[intp] 类型的参数，并返回 NDArray[intp] 类型的结果
) -> NDArray[intp]: ...

# 重载函数签名，接收一个 _ArrayLike[_SCT] 类型的参数 a 和 _ShapeLike 类型的参数 new_shape，返回 NDArray[_SCT] 类型的结果
def resize(
    a: _ArrayLike[_SCT],
    new_shape: _ShapeLike,
) -> NDArray[_SCT]: ...

# 重载函数签名，接收一个 ArrayLike 类型的参数 a 和 _ShapeLike 类型的参数 new_shape，返回 NDArray[Any] 类型的结果
def resize(
    a: ArrayLike,
    new_shape: _ShapeLike,
) -> NDArray[Any]: ...

# 重载函数签名，接收一个 _SCT 类型的参数 a 和可选的 None 或 _ShapeLike 类型的参数 axis，返回 _SCT 类型的结果
def squeeze(
    a: _SCT,
    axis: None | _ShapeLike = ...,
) -> _SCT: ...

# 重载函数签名，接收一个 _ArrayLike[_SCT] 类型的参数 a 和可选的 None 或 _ShapeLike 类型的参数 axis，返回 NDArray[_SCT] 类型的结果
def squeeze(
    a: _ArrayLike[_SCT],
    axis: None | _ShapeLike = ...,
) -> NDArray[_SCT]: ...

# 重载函数签名，接收一个 ArrayLike 类型的参数 a 和可选的 None 或 _ShapeLike 类型的参数 axis，返回 NDArray[Any] 类型的结果
def squeeze(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
) -> NDArray[Any]: ...

# 重载函数签名，接收一个 _ArrayLike[_SCT] 类型的参数 a，以及可选的 offset、axis1 和 axis2 参数（用于 >= 2D 数组），返回 NDArray[_SCT] 类型的结果
def diagonal(
    a: _ArrayLike[_SCT],
    offset: SupportsIndex = ...,
    axis1: SupportsIndex = ...,
    axis2: SupportsIndex = ...,  # >= 2D array
) -> NDArray[_SCT]: ...

# 重载函数签名，接收一个 ArrayLike 类型的参数 a，以及可选的 offset、axis1 和 axis2 参数（用于 >= 2D 数组），返回 NDArray[Any] 类型的结果
def diagonal(
    a: ArrayLike,
    offset: SupportsIndex = ...,
    axis1: SupportsIndex = ...,
    axis2: SupportsIndex = ...,  # >= 2D array
) -> NDArray[Any]: ...

# 重载函数签名，接收一个 ArrayLike 类型的参数 a，以及可选的 offset、axis1、axis2、dtype 和 out 参数（用于 >= 2D 数组），返回 Any 类型的结果
def trace(
    a: ArrayLike,  # >= 2D array
    offset: SupportsIndex = ...,
    axis1: SupportsIndex = ...,
    axis2: SupportsIndex = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
) -> Any: ...

# 重载函数签名，接收一个 ArrayLike 类型的参数 a，以及可选的 offset、axis1、axis2、dtype 和 out 参数（用于 >= 2D 数组），返回 _ArrayType 类型的结果
def trace(
    a: ArrayLike,  # >= 2D array
    offset: SupportsIndex = ...,
    axis1: SupportsIndex = ...,
    axis2: SupportsIndex = ...,
    dtype: DTypeLike = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...

# 重载函数签名，接收一个 _ArrayLike[_SCT] 类型的参数 a 和可选的 _OrderKACF 类型的参数 order，返回 NDArray[_SCT] 类型的结果
def ravel(a: _ArrayLike[_SCT], order: _OrderKACF = ...) -> NDArray[_SCT]: ...

# 重载函数签名，接收一个 ArrayLike 类型的参数 a 和可选的 _OrderKACF 类型的参数 order，返回 NDArray[Any] 类型的结果
def ravel(a: ArrayLike, order: _OrderKACF = ...) -> NDArray[Any]: ...

# 定义一个函数非零，接收一个 ArrayLike 类型的参数 a，返回一个元组，包含 NDArray[intp] 类型的结果
def nonzero(a: ArrayLike) -> tuple[NDArray[intp], ...]: ...

# 定义一个函数 shape，接收一个 ArrayLike 类型的参数 a，返回 _Shape 类型的结果
def shape(a: ArrayLike) -> _Shape: ...

# 重载函数签名，接收一个 _ArrayLikeBool_co 类型的参数 condition（用于 1D 布尔数组）、一个 _ArrayLike[_SCT] 类型的参数 a 和可选的 None 或 SupportsIndex 类型的参数 axis，返回 NDArray[_SCT] 类型的结果
def compress(
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: _ArrayLike[_SCT],
    axis: None | SupportsIndex = ...,
    out: None = ...,
) -> NDArray[_SCT]: ...

# 重载函数签名，接收一个 _ArrayLikeBool_co 类型的参数 condition（用于 1D 布尔数组）、一个 ArrayLike 类型的参数 a 和可选的 None 或 SupportsIndex 类型的参数 axis，返回 NDArray[Any] 类型的结果
def compress(
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    out: None = ...,
) -> NDArray[Any]: ...

# 重载函数签名，接收一个 _ArrayLikeBool_co 类型的参数 condition（用于 1D 布尔数组）、一个 ArrayLike 类型的参数 a 和可选的 None 或 SupportsIndex 类型的参数 axis，返回 _ArrayType 类型的结果
def compress(
    condition: _ArrayLikeBool_co,  # 1D bool array
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...

# 重载函数签名，接收多个参数，包括 _SCT 类型的参数 a、可选的 None 或 ArrayLike 类型的参数 a_min 和 a_max 等，返回 _SCT 类型的结果
def clip(
    a: _SCT,
    a_min: None | ArrayLike,
    a_max: None | ArrayLike,
    out: None = ...,
    *,
    dtype: None = ...,
    where: None | _ArrayLikeBool_co = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    signature: str | tuple[None | str, ...] = ...,
    casting: _CastingKind = ...,
) -> _SCT: ...

# 重载函数签名，接收多个参数，包括 _ScalarLike_co 类型的参数 a、可选的 None 或 ArrayLike 类型的参数 a_min 和 a_max 等，返回 Any 类型的结果
def clip(
    a: _ScalarLike_co,
    a_min: None | ArrayLike,
    a_max: None | ArrayLike,
    out: None = ...,
    *,
    dtype: None = ...,
    where: None | _ArrayLikeBool_co = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    signature: str | tuple[None | str, ...] = ...,
    casting: _CastingKind = ...,
) -> Any: ...

# 重载函数签名，接收多个参数，包括 _ArrayLike[_SCT] 类型的参数 a、可选的 None 或 ArrayLike 类型的参数 a_min 和 a_max 等，返回 _ArrayType 类型的结果
def clip(
    a: _ArrayLike[_SCT],
    a_min: None | ArrayLike,
    a_max: None | ArrayLike,
    out: None = ...,
    *,
    dtype: None = ...,
    where: None | _ArrayLikeBool_co = ...,
    order: _OrderKACF = ...,  # 定义一个名为 `order` 的变量，类型为 `_OrderKACF`，并初始化为 `...`
    subok: bool = ...,  # 定义一个名为 `subok` 的变量，类型为 `bool`，并初始化为 `...`
    signature: str | tuple[None | str, ...] = ...,  # 定义一个名为 `signature` 的变量，类型为 `str` 或包含 `None` 或多个 `str` 的元组，并初始化为 `...`
    casting: _CastingKind = ...,  # 定义一个名为 `casting` 的变量，类型为 `_CastingKind`，并初始化为 `...`
# 函数签名，指定函数返回类型为 NDArray[_SCT]
def clip(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 最小值限制，可以为 None 或者类数组类型
    a_min: None | ArrayLike,
    # 最大值限制，可以为 None 或者类数组类型
    a_max: None | ArrayLike,
    # 输出数组，如果为 None 则表示不输出
    out: None = ...,
    # 数据类型，如果为 None 则表示不指定
    *,
    dtype: None = ...,
    # where 参数，用于指定条件
    where: None | _ArrayLikeBool_co = ...,
    # 数组顺序，如 C 或者 F
    order: _OrderKACF = ...,
    # 是否允许子类数组
    subok: bool = ...,
    # 函数签名，可以是字符串或者元组
    signature: str | tuple[None | str, ...] = ...,
    # 强制转换类型
    casting: _CastingKind = ...,
) -> NDArray[Any]: ...
@overload
def clip(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 最小值限制，可以为 None 或者类数组类型
    a_min: None | ArrayLike,
    # 最大值限制，可以为 None 或者类数组类型
    a_max: None | ArrayLike,
    # 输出数组，指定为 _ArrayType 类型
    out: _ArrayType = ...,
    # 数据类型，指定为 DTypeLike 类型
    *,
    dtype: DTypeLike,
    # where 参数，用于指定条件
    where: None | _ArrayLikeBool_co = ...,
    # 数组顺序，如 C 或者 F
    order: _OrderKACF = ...,
    # 是否允许子类数组
    subok: bool = ...,
    # 函数签名，可以是字符串或者元组
    signature: str | tuple[None | str, ...] = ...,
    # 强制转换类型
    casting: _CastingKind = ...,
) -> Any: ...
@overload
def clip(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 最小值限制，可以为 None 或者类数组类型
    a_min: None | ArrayLike,
    # 最大值限制，可以为 None 或者类数组类型
    a_max: None | ArrayLike,
    # 输出数组，指定为 _ArrayType 类型
    out: _ArrayType,
    # 数据类型，可以为 DTypeLike 类型，默认为 None
    *,
    dtype: DTypeLike = ...,
    # where 参数，用于指定条件
    where: None | _ArrayLikeBool_co = ...,
    # 数组顺序，如 C 或者 F
    order: _OrderKACF = ...,
    # 是否允许子类数组
    subok: bool = ...,
    # 函数签名，可以是字符串或者元组
    signature: str | tuple[None | str, ...] = ...,
    # 强制转换类型
    casting: _CastingKind = ...,
) -> _ArrayType: ...

@overload
def sum(
    # 第一个参数 a，可以是类数组类型
    a: _ArrayLike[_SCT],
    # 沿着哪个轴求和，如果为 None 则表示全部求和
    axis: None = ...,
    # 指定输出数据类型，如果为 None 则表示不指定
    dtype: None = ...,
    # 输出数组，如果为 None 则表示不输出
    out: None  = ...,
    # 是否保持维度，对应结果是否保持原数组的维度
    keepdims: bool = ...,
    # 初始值，对应求和的初始值
    initial: _NumberLike_co = ...,
    # where 参数，用于指定条件
    where: _ArrayLikeBool_co = ...,
) -> _SCT: ...
@overload
def sum(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 沿着哪个轴求和，如果为 None 则表示全部求和
    axis: None | _ShapeLike = ...,
    # 指定输出数据类型，可以是 DTypeLike 类型
    dtype: DTypeLike = ...,
    # 输出数组，如果为 None 则表示不输出
    out: None  = ...,
    # 是否保持维度，对应结果是否保持原数组的维度
    keepdims: bool = ...,
    # 初始值，对应求和的初始值
    initial: _NumberLike_co = ...,
    # where 参数，用于指定条件
    where: _ArrayLikeBool_co = ...,
) -> Any: ...
@overload
def sum(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 沿着哪个轴求和，如果为 None 则表示全部求和
    axis: None | _ShapeLike = ...,
    # 指定输出数据类型，可以是 DTypeLike 类型
    dtype: DTypeLike = ...,
    # 输出数组，指定为 _ArrayType 类型
    out: _ArrayType  = ...,
    # 是否保持维度，对应结果是否保持原数组的维度
    keepdims: bool = ...,
    # 初始值，对应求和的初始值
    initial: _NumberLike_co = ...,
    # where 参数，用于指定条件
    where: _ArrayLikeBool_co = ...,
) -> _ArrayType: ...

@overload
def all(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 沿着哪个轴进行 all 操作，如果为 None 则表示全部维度
    axis: None = ...,
    # 输出数组，如果为 None 则表示不输出
    out: None = ...,
    # 是否保持维度，对应结果是否保持原数组的维度
    keepdims: Literal[False] = ...,
    # where 参数，用于指定条件
    *,
    where: _ArrayLikeBool_co = ...,
) -> np.bool: ...
@overload
def all(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 沿着哪个轴进行 all 操作，如果为 None 则表示全部维度
    axis: None | _ShapeLike = ...,
    # 输出数组，如果为 None 则表示不输出
    out: None = ...,
    # 是否保持维度，对应结果是否保持原数组的维度
    keepdims: bool = ...,
    # where 参数，用于指定条件
    *,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...
@overload
def all(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 沿着哪个轴进行 all 操作，如果为 None 则表示全部维度
    axis: None | _ShapeLike = ...,
    # 输出数组，指定为 _ArrayType 类型
    out: _ArrayType = ...,
    # 是否保持维度，对应结果是否保持原数组的维度
    keepdims: bool = ...,
    # where 参数，用于指定条件
    *,
    where: _ArrayLikeBool_co = ...,
) -> _ArrayType: ...

@overload
def any(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 沿着哪个轴进行 any 操作，如果为 None 则表示全部维度
    axis: None = ...,
    # 输出数组，如果为 None 则表示不输出
    out: None = ...,
    # 是否保持维度，对应结果是否保持原数组的维度
    keepdims: Literal[False] = ...,
    # where 参数，用于指定条件
    *,
    where: _ArrayLikeBool_co = ...,
) -> np.bool: ...
@overload
def any(
    # 第一个参数 a，可以是类数组类型
    a: ArrayLike,
    # 沿着哪个轴进行 any 操作，如果为 None 则表示全部维度
    axis: None | _ShapeLike = ...,
    # 输出数组，如果为 None 则表示不输出
    out: None = ...,
    # 是否保持维度，对应结果是否保持原数组的维度
    keepdims: bool = ...,
    # where 参数，用于指定条件
# 返回值类型注解，表示此函数返回一个 NDArray 对象，其元素类型为 `_SCT`
) -> NDArray[_SCT]: ...

# `cumsum` 函数的第一个重载：计算数组元素的累积和
@overload
def cumsum(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    dtype: None = ...,
    out: None = ...,
) -> NDArray[Any]: ...

# `cumsum` 函数的第二个重载：计算数组元素的累积和，指定元素类型为 `_SCT`
@overload
def cumsum(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    dtype: _DTypeLike[_SCT] = ...,
    out: None = ...,
) -> NDArray[_SCT]: ...

# `cumsum` 函数的第三个重载：计算数组元素的累积和，指定元素类型为 `DTypeLike`
@overload
def cumsum(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
) -> NDArray[Any]: ...

# `cumsum` 函数的第四个重载：计算数组元素的累积和，指定元素类型为 `DTypeLike`，并将结果存入指定数组类型 `out`
@overload
def cumsum(
    a: ArrayLike,
    axis: None | SupportsIndex = ...,
    dtype: DTypeLike = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...

# `ptp` 函数的第一个重载：计算数组沿指定轴的最大值与最小值之差
@overload
def ptp(
    a: _ArrayLike[_SCT],
    axis: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
) -> _SCT: ...

# `ptp` 函数的第二个重载：计算数组沿指定轴的最大值与最小值之差，返回任意类型的值
@overload
def ptp(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
) -> Any: ...

# `ptp` 函数的第三个重载：计算数组沿指定轴的最大值与最小值之差，结果存入指定数组类型 `out`
@overload
def ptp(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: _ArrayType = ...,
    keepdims: bool = ...,
) -> _ArrayType: ...

# `amax` 函数的第一个重载：计算数组沿指定轴的最大值
@overload
def amax(
    a: _ArrayLike[_SCT],
    axis: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> _SCT: ...

# `amax` 函数的第二个重载：计算数组沿指定轴的最大值，返回任意类型的值
@overload
def amax(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

# `amax` 函数的第三个重载：计算数组沿指定轴的最大值，结果存入指定数组类型 `out`
@overload
def amax(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: _ArrayType = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> _ArrayType: ...

# `amin` 函数的第一个重载：计算数组沿指定轴的最小值
@overload
def amin(
    a: _ArrayLike[_SCT],
    axis: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> _SCT: ...

# `amin` 函数的第二个重载：计算数组沿指定轴的最小值，返回任意类型的值
@overload
def amin(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

# `amin` 函数的第三个重载：计算数组沿指定轴的最小值，结果存入指定数组类型 `out`
@overload
def amin(
    a: ArrayLike,
    axis: None | _ShapeLike = ...,
    out: _ArrayType = ...,
    keepdims: bool = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> _ArrayType: ...

# `prod` 函数的第一个重载：计算数组元素沿指定轴的乘积
# 注意：对于对象数组，参数 `initial` 不必是数值标量。
# 唯一的要求是它与传递给数组元素的 `.__mul__()` 方法兼容。
@overload
def prod(
    a: _ArrayLikeBool_co,
    axis: None = ...,
    dtype: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike_co = ...,
    where: _ArrayLikeBool_co = ...,
) -> int_: ...
    keepdims: Literal[False] = ...,  # 参数 keepdims，默认为 False，用于指定是否保留维度信息
    initial: _NumberLike_co = ...,   # 参数 initial，默认为任何数字类型，用于指定初始值
    where: _ArrayLikeBool_co = ...,  # 参数 where，默认为任何布尔数组类型，用于条件选择
# 定义函数签名，指定返回类型为 uint64 的 prod 函数重载
@overload
def prod(
    # 第一个参数 a：接受 _ArrayLikeInt_co 类型的数组或可迭代对象
    a: _ArrayLikeInt_co,
    # axis 参数：指定沿着哪个轴进行计算，这里为 None 表示沿所有维度进行计算
    axis: None = ...,
    # dtype 参数：指定输出的数据类型，默认为 None，即保持输入的数据类型
    dtype: None = ...,
    # out 参数：指定输出结果的存储位置，默认为 None，表示新创建数组来存储结果
    out: None = ...,
    # keepdims 参数：指定是否保持维度，这里设为 False，即不保持
    keepdims: Literal[False] = ...,
    # initial 参数：指定初始值，可以是数字或者支持的数据类型
    initial: _NumberLike_co = ...,
    # where 参数：指定条件，用于选择参与计算的元素，默认为全部参与
    where: _ArrayLikeBool_co = ...,
) -> int64: ...
# 其他数据类型的 prod 函数重载类似，具体参数含义相同

# cumprod 函数的定义，用于计算累积乘积
@overload
def cumprod(
    # a 参数：接受 _ArrayLikeBool_co 类型的数组或可迭代对象
    a: _ArrayLikeBool_co,
    # axis 参数：指定沿着哪个轴进行计算，支持 None 或者整数类型，表示全部或指定轴
    axis: None | SupportsIndex = ...,
    # dtype 参数：指定输出的数据类型，默认为 None，即保持输入的数据类型
    dtype: None = ...,
    # out 参数：指定输出结果的存储位置，默认为 None，表示新创建数组来存储结果
    out: None = ...,
) -> NDArray[int_]: ...
# 其他数据类型的 cumprod 函数重载类似，具体参数含义相同
# 函数签名声明，定义 cumprod 函数，返回类型为 NDArray[_SCT]
@overload
def cumprod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | SupportsIndex = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
) -> NDArray[Any]: ...

# 函数签名声明，定义 cumprod 函数，返回类型为 _ArrayType
@overload
def cumprod(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | SupportsIndex = ...,
    dtype: DTypeLike = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...

# 函数签名声明，定义 ndim 函数，接收 ArrayLike 类型参数，返回 int 类型
def ndim(a: ArrayLike) -> int: ...

# 函数签名声明，定义 size 函数，接收 ArrayLike 类型参数和 axis 参数（可选的 int 或 None），返回 int 类型
def size(a: ArrayLike, axis: None | int = ...) -> int: ...

# 函数签名声明，定义 around 函数，对 _BoolLike_co 类型的参数 a 进行处理，返回 float16 类型
@overload
def around(
    a: _BoolLike_co,
    decimals: SupportsIndex = ...,
    out: None = ...,
) -> float16: ...

# 函数签名声明，定义 around 函数，对 _SCT_uifcO 类型的参数 a 进行处理，返回 _SCT_uifcO 类型
@overload
def around(
    a: _SCT_uifcO,
    decimals: SupportsIndex = ...,
    out: None = ...,
) -> _SCT_uifcO: ...

# 函数签名声明，定义 around 函数，对 _ComplexLike_co 或 object_ 类型的参数 a 进行处理，返回 Any 类型
@overload
def around(
    a: _ComplexLike_co | object_,
    decimals: SupportsIndex = ...,
    out: None = ...,
) -> Any: ...

# 函数签名声明，定义 around 函数，对 _ArrayLikeBool_co 类型的参数 a 进行处理，返回 NDArray[float16] 类型
@overload
def around(
    a: _ArrayLikeBool_co,
    decimals: SupportsIndex = ...,
    out: None = ...,
) -> NDArray[float16]: ...

# 函数签名声明，定义 around 函数，对 _ArrayLike[_SCT_uifcO] 类型的参数 a 进行处理，返回 NDArray[_SCT_uifcO] 类型
@overload
def around(
    a: _ArrayLike[_SCT_uifcO],
    decimals: SupportsIndex = ...,
    out: None = ...,
) -> NDArray[_SCT_uifcO]: ...

# 函数签名声明，定义 around 函数，对 _ArrayLikeComplex_co 或 _ArrayLikeObject_co 类型的参数 a 进行处理，返回 NDArray[Any] 类型
@overload
def around(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    decimals: SupportsIndex = ...,
    out: None = ...,
) -> NDArray[Any]: ...

# 函数签名声明，定义 around 函数，对 _ArrayLikeComplex_co 或 _ArrayLikeObject_co 类型的参数 a 进行处理，返回 _ArrayType 类型
@overload
def around(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    decimals: SupportsIndex = ...,
    out: _ArrayType = ...,
) -> _ArrayType: ...

# 函数签名声明，定义 mean 函数，对 _ArrayLikeFloat_co 类型的参数 a 进行处理，返回 floating[Any] 类型
@overload
def mean(
    a: _ArrayLikeFloat_co,
    axis: None = ...,
    dtype: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> floating[Any]: ...

# 函数签名声明，定义 mean 函数，对 _ArrayLikeComplex_co 类型的参数 a 进行处理，返回 complexfloating[Any, Any] 类型
@overload
def mean(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    dtype: None = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> complexfloating[Any, Any]: ...

# 函数签名声明，定义 mean 函数，对 _ArrayLikeComplex_co 或 _ArrayLikeObject_co 类型的参数 a 进行处理，返回 Any 类型
@overload
def mean(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    dtype: None = ...,
    out: None = ...,
    keepdims: bool = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

# 函数签名声明，定义 mean 函数，对 _ArrayLikeComplex_co 或 _ArrayLikeObject_co 类型的参数 a 进行处理，返回 _SCT 类型
@overload
def mean(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None = ...,
    dtype: _DTypeLike[_SCT] = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> _SCT: ...

# 函数签名声明，定义 mean 函数，对 _ArrayLikeComplex_co 或 _ArrayLikeObject_co 类型的参数 a 进行处理，返回 Any 类型
@overload
def mean(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> Any: ...

# 函数签名声明，定义 mean 函数，对 _ArrayLikeComplex_co 或 _ArrayLikeObject_co 类型的参数 a 进行处理，返回 _ArrayType 类型
@overload
def mean(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    dtype: DTypeLike = ...,
    out: _ArrayType = ...,
    keepdims: bool = ...,
    *,
    where: _ArrayLikeBool_co = ...,
) -> _ArrayType: ...

# 函数签名声明，定义 std 函数，对 _ArrayLikeComplex_co 类型的参数 a 进行处理，返回 Any 类型
@overload
def std(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    dtype: None = ...,
    out: None = ...,
    ddof: int | float = ...,
    keepdims: Literal[False] = ...,
    # keepdims 参数，指定是否保留每个维度的长度信息，这里默认为 False
    *,
    # 星号 * 表示这之后的参数只能通过关键字指定
    where: _ArrayLikeBool_co = ...,
    # where 参数，用于指定条件，必须是类数组类型，含有布尔值元素
    mean: _ArrayLikeComplex_co = ...,
    # mean 参数，用于指定均值，必须是类数组类型，含有复数元素
    correction: int | float = ...,
    # correction 参数，用于指定修正值，可以是整数或浮点数类型
# 函数签名，定义了 std 函数的重载情况，返回类型可能是任意类型
@overload
def std(
    # 参数 a 可以是复杂数组或对象数组
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    # 没有指定轴，或者轴的形状是 None 或可形状化的对象
    axis: None | _ShapeLike = ...,
    # 数据类型默认为 None
    dtype: None = ...,
    # 输出对象默认为 None
    out: None = ...,
    # 自由度修正参数，默认为整数或浮点数
    ddof: int | float = ...,
    # 是否保持维度的布尔值，默认为 False
    keepdims: bool = ...,
    # 以下是关键字参数，使用 * 标记强制指定后续参数
    *,
    # where 参数作为布尔数组
    where: _ArrayLikeBool_co = ...,
    # 均值作为复杂数组或对象数组
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    # 修正值，默认为整数或浮点数
    correction: int | float = ...,
) -> Any: ...
# 下一重载情况的 std 函数，返回类型为 _SCT 类型
@overload
def std(
    # 参数 a 可以是复杂数组或对象数组
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    # 没有指定轴
    axis: None = ...,
    # 数据类型是 _DTypeLike[_SCT] 类型
    dtype: _DTypeLike[_SCT] = ...,
    # 输出对象默认为 None
    out: None = ...,
    # 自由度修正参数，默认为整数或浮点数
    ddof: int | float = ...,
    # 是否保持维度的布尔值，默认为 False
    keepdims: Literal[False] = ...,
    # 以下是关键字参数，使用 * 标记强制指定后续参数
    *,
    # where 参数作为布尔数组
    where: _ArrayLikeBool_co = ...,
    # 均值作为复杂数组或对象数组
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    # 修正值，默认为整数或浮点数
    correction: int | float = ...,
) -> _SCT: ...
# 下一重载情况的 std 函数，返回类型可能是任意类型
@overload
def std(
    # 参数 a 可以是复杂数组或对象数组
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    # 没有指定轴，或者轴的形状是 None 或可形状化的对象
    axis: None | _ShapeLike = ...,
    # 数据类型可以是 DTypeLike 类型
    dtype: DTypeLike = ...,
    # 输出对象默认为 None
    out: None = ...,
    # 自由度修正参数，默认为整数或浮点数
    ddof: int | float = ...,
    # 是否保持维度的布尔值，默认为 False
    keepdims: bool = ...,
    # 以下是关键字参数，使用 * 标记强制指定后续参数
    *,
    # where 参数作为布尔数组
    where: _ArrayLikeBool_co = ...,
    # 均值作为复杂数组或对象数组
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    # 修正值，默认为整数或浮点数
    correction: int | float = ...,
) -> Any: ...
# 最后一重载情况的 std 函数，返回类型是 _ArrayType 类型
@overload
def std(
    # 参数 a 可以是复杂数组或对象数组
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    # 没有指定轴，或者轴的形状是 None 或可形状化的对象
    axis: None | _ShapeLike = ...,
    # 数据类型可以是 DTypeLike 类型
    dtype: DTypeLike = ...,
    # 输出对象可以是 _ArrayType 类型
    out: _ArrayType = ...,
    # 自由度修正参数，默认为整数或浮点数
    ddof: int | float = ...,
    # 是否保持维度的布尔值，默认为 False
    keepdims: bool = ...,
    # 以下是关键字参数，使用 * 标记强制指定后续参数
    *,
    # where 参数作为布尔数组
    where: _ArrayLikeBool_co = ...,
    # 均值作为复杂数组或对象数组
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    # 修正值，默认为整数或浮点数
    correction: int | float = ...,
) -> _ArrayType: ...

# 函数签名，定义了 var 函数的重载情况，返回类型可能是任意类型
@overload
def var(
    # 参数 a 可以是复杂数组
    a: _ArrayLikeComplex_co,
    # 没有指定轴
    axis: None = ...,
    # 数据类型默认为 None
    dtype: None = ...,
    # 输出对象默认为 None
    out: None = ...,
    # 自由度修正参数，默认为整数或浮点数
    ddof: int | float = ...,
    # 是否保持维度的布尔值，默认为 False
    keepdims: Literal[False] = ...,
    # 以下是关键字参数，使用 * 标记强制指定后续参数
    *,
    # where 参数作为布尔数组
    where: _ArrayLikeBool_co = ...,
    # 均值作为复杂数组
    mean: _ArrayLikeComplex_co = ...,
    # 修正值，默认为整数或浮点数
    correction: int | float = ...,
) -> floating[Any]: ...
# 下一重载情况的 var 函数，返回类型可能是任意类型
@overload
def var(
    # 参数 a 可以是复杂数组或对象数组
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    # 没有指定轴，或者轴的形状是 None 或可形状化的对象
    axis: None | _ShapeLike = ...,
    # 数据类型默认为 None
    dtype: None = ...,
    # 输出对象默认为 None
    out: None = ...,
    # 自由度修正参数，默认为整数或浮点数
    ddof: int | float = ...,
    # 是否保持维度的布尔值，默认为 False
    keepdims: bool = ...,
    # 以下是关键字参数，使用 * 标记强制指定后续参数
    *,
    # where 参数作为布尔数组
    where: _ArrayLikeBool_co = ...,
    # 均值作为复杂数组或对象数组
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    # 修正值，默认为整数或浮点数
    correction: int | float = ...,
) -> Any: ...
# 下一重载情况的 var 函数，返回类型是 _SCT 类型
@overload
def var(
    # 参数 a 可以是复杂数组或对象数组
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    # 没有指定轴
    axis: None = ...,
    # 数据类型是 _DTypeLike[_SCT] 类型
    dtype: _DTypeLike[_SCT] = ...,
    # 输出对象默认为 None
    out: None = ...,
    # 自由度修正参数，默认为整数或浮点数
    ddof: int | float = ...,
    # 是否保持维度的布尔值，默认为 False
    keepdims: Literal[False] = ...,
    # 以下是关键字参数，使用 * 标记强制指定后续参数
    *,
    # where 参数作为布尔数组
    where: _ArrayLikeBool_co = ...,
    # 均值作为复杂数组或对象数组
    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    # 修正值，默认为整数或浮点数
    correction: int | float = ...,
) -> _SCT: ...
# 下一重载情
    dtype: DTypeLike = ...,  
    # dtype 参数指定返回数组的数据类型，可以是 DTypeLike 类型的任何值，通常是一个数据类型或与数组兼容的对象

    out: _ArrayType = ...,  
    # out 参数指定结果存放的数组，类型为 _ArrayType，通常用于指定结果的存储位置以节省内存

    ddof: int | float = ...,  
    # ddof 参数表示自由度的校正值，可以是整数或浮点数，用于调整标准差计算中的自由度

    keepdims: bool = ...,  
    # keepdims 参数指定是否保持减少维度后的维度数，为布尔值，控制是否保留维度

    *,  
    # 星号 * 后的参数表示接下来的参数必须使用关键字传递，而非位置传递

    where: _ArrayLikeBool_co = ...,  
    # where 参数是一个布尔类型的数组或类似数组，用于指定元素操作的条件

    mean: _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,  
    # mean 参数是一个复杂数或对象数组，用于计算平均值时的输入数组

    correction: int | float = ...,  
    # correction 参数是一个整数或浮点数，用于在计算过程中进行修正
# 定义一个函数签名，该函数接受参数并返回一个 _ArrayType 类型的值
) -> _ArrayType: ...
# 将 amax 赋值给 max，可能是为了简化变量命名或避免命名冲突
max = amax
# 将 amin 赋值给 min，可能是为了简化变量命名或避免命名冲突
min = amin
# 将 around 赋值给 round，可能是为了简化变量命名或避免命名冲突
round = around
```