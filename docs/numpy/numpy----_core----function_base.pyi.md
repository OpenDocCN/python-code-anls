# `D:\src\scipysrc\numpy\numpy\_core\function_base.pyi`

```py
# 导入必要的类型和函数，包括字面量类型、重载函数、任意类型、支持索引的类型变量
from typing import (
    Literal as L,
    overload,
    Any,
    SupportsIndex,
    TypeVar,
)

# 从 numpy 库中导入特定的数据类型：浮点数、复数浮点数、泛型
from numpy import floating, complexfloating, generic
# 从 numpy._typing 模块导入特定的类型别名
from numpy._typing import (
    NDArray,
    DTypeLike,
    _DTypeLike,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
)

# 类型变量，用于支持泛型
_SCT = TypeVar("_SCT", bound=generic)

# __all__ 是模块中公开的对象列表，这里初始化为空列表
__all__: list[str]

# 以下是对 linspace 函数的多个重载定义，用于创建均匀间隔的数组
@overload
def linspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[False] = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[floating[Any]]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[False] = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[False] = ...,
    dtype: _DTypeLike[_SCT] = ...,
    axis: SupportsIndex = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[_SCT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[False] = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...
@overload
def linspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[True] = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
    *,
    device: None | L["cpu"] = ...,
) -> tuple[NDArray[floating[Any]], floating[Any]]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[True] = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
    *,
    device: None | L["cpu"] = ...,
) -> tuple[NDArray[complexfloating[Any, Any]], complexfloating[Any, Any]]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[True] = ...,
    dtype: _DTypeLike[_SCT] = ...,
    axis: SupportsIndex = ...,
    *,
    device: None | L["cpu"] = ...,
) -> tuple[NDArray[_SCT], _SCT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    retstep: L[True] = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
    *,
    device: None | L["cpu"] = ...,
) -> tuple[NDArray[Any], Any]: ...



# logspace 函数的重载定义，用于创建对数间隔的数组
@overload
def logspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeFloat_co = ...,  # base 是一个类型为 _ArrayLikeFloat_co 的变量，通常用于表示基础数组或数据结构
    dtype: None = ...,  # dtype 是一个类型为 None 的变量，通常用于指定数据类型
    axis: SupportsIndex = ...,  # axis 是一个类型为 SupportsIndex 的变量，通常用于表示支持索引的轴或维度
# 定义一个函数签名，该函数返回浮点数类型的 NumPy 数组
) -> NDArray[floating[Any]]: ...

# logspace 函数的重载定义，用于生成等比数列
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeComplex_co = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# logspace 函数的重载定义，用于生成等比数列
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeComplex_co = ...,
    dtype: _DTypeLike[_SCT] = ...,
    axis: SupportsIndex = ...,
) -> NDArray[_SCT]: ...

# logspace 函数的重载定义，用于生成等比数列
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    base: _ArrayLikeComplex_co = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> NDArray[Any]: ...

# geomspace 函数的重载定义，用于生成等比几何数列
@overload
def geomspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[floating[Any]]: ...

# geomspace 函数的重载定义，用于生成等比几何数列
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: None = ...,
    axis: SupportsIndex = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# geomspace 函数的重载定义，用于生成等比几何数列
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: _DTypeLike[_SCT] = ...,
    axis: SupportsIndex = ...,
) -> NDArray[_SCT]: ...

# geomspace 函数的重载定义，用于生成等比几何数列
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = ...,
    endpoint: bool = ...,
    dtype: DTypeLike = ...,
    axis: SupportsIndex = ...,
) -> NDArray[Any]: ...

# add_newdoc 函数用于添加文档字符串到指定的对象（模块、类、函数等）
def add_newdoc(
    place: str,
    obj: str,
    doc: str | tuple[str, str] | list[tuple[str, str]],
    warn_on_python: bool = ...,
) -> None: ...
```