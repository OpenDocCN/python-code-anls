# `D:\src\scipysrc\numpy\numpy\_core\shape_base.pyi`

```py
# 导入必要的模块和类型定义
from collections.abc import Sequence
from typing import TypeVar, overload, Any, SupportsIndex

# 导入必要的numpy模块和类型定义
from numpy import generic, _CastingKind
from numpy._typing import (
    NDArray,
    ArrayLike,
    DTypeLike,
    _ArrayLike,
    _DTypeLike,
)

# __all__列表用于指定在使用 from 模块 import * 语句时需要导入的名称
__all__: list[str]

# 函数重载定义，用于确保输入参数至少是1维数组
@overload
def atleast_1d(arys: _ArrayLike[_SCT], /) -> NDArray[_SCT]: ...
@overload
def atleast_1d(arys: ArrayLike, /) -> NDArray[Any]: ...
@overload
def atleast_1d(*arys: ArrayLike) -> tuple[NDArray[Any], ...]: ...

# 函数重载定义，用于确保输入参数至少是2维数组
@overload
def atleast_2d(arys: _ArrayLike[_SCT], /) -> NDArray[_SCT]: ...
@overload
def atleast_2d(arys: ArrayLike, /) -> NDArray[Any]: ...
@overload
def atleast_2d(*arys: ArrayLike) -> tuple[NDArray[Any], ...]: ...

# 函数重载定义，用于确保输入参数至少是3维数组
@overload
def atleast_3d(arys: _ArrayLike[_SCT], /) -> NDArray[_SCT]: ...
@overload
def atleast_3d(arys: ArrayLike, /) -> NDArray[Any]: ...
@overload
def atleast_3d(*arys: ArrayLike) -> tuple[NDArray[Any], ...]: ...

# 垂直堆叠多个数组
@overload
def vstack(
    tup: Sequence[_ArrayLike[_SCT]],
    *,
    dtype: None = ...,
    casting: _CastingKind = ...
) -> NDArray[_SCT]: ...
@overload
def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: _DTypeLike[_SCT],
    casting: _CastingKind = ...
) -> NDArray[_SCT]: ...
@overload
def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike = ...,
    casting: _CastingKind = ...
) -> NDArray[Any]: ...

# 水平堆叠多个数组
@overload
def hstack(
    tup: Sequence[_ArrayLike[_SCT]],
    *,
    dtype: None = ...,
    casting: _CastingKind = ...
) -> NDArray[_SCT]: ...
@overload
def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: _DTypeLike[_SCT],
    casting: _CastingKind = ...
) -> NDArray[_SCT]: ...
@overload
def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike = ...,
    casting: _CastingKind = ...
) -> NDArray[Any]: ...

# 沿指定轴堆叠多个数组
@overload
def stack(
    arrays: Sequence[_ArrayLike[_SCT]],
    axis: SupportsIndex = ...,
    out: None = ...,
    *,
    dtype: None = ...,
    casting: _CastingKind = ...
) -> NDArray[_SCT]: ...
@overload
def stack(
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = ...,
    out: None = ...,
    *,
    dtype: _DTypeLike[_SCT],
    casting: _CastingKind = ...
) -> NDArray[_SCT]: ...
@overload
def stack(
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = ...,
    out: None = ...,
    *,
    dtype: DTypeLike = ...,
    casting: _CastingKind = ...
) -> NDArray[Any]: ...
@overload
def stack(
    arrays: Sequence[ArrayLike],
    axis: SupportsIndex = ...,
    out: _ArrayType = ...,
    *,
    dtype: DTypeLike = ...,
    casting: _CastingKind = ...
) -> _ArrayType: ...

# 将多个数组块连接成一个数组
@overload
def block(arrays: _ArrayLike[_SCT]) -> NDArray[_SCT]: ...
@overload
def block(arrays: ArrayLike) -> NDArray[Any]: ...
```