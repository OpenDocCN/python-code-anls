# `D:\src\scipysrc\numpy\numpy\lib\_twodim_base_impl.pyi`

```py
# 导入内建模块和函数
import builtins
# 导入抽象基类中的可调用对象和序列
from collections.abc import Callable, Sequence
# 导入类型提示
from typing import (
    Any,
    overload,
    TypeVar,
    Literal as L,
)

# 导入 NumPy 库，并导入一系列特定的数据类型和函数
import numpy as np
from numpy import (
    generic,
    number,
    timedelta64,
    datetime64,
    int_,
    intp,
    float64,
    signedinteger,
    floating,
    complexfloating,
    object_,
    _OrderCF,
)

# 导入 NumPy 内部的类型定义
from numpy._typing import (
    DTypeLike,
    _DTypeLike,
    ArrayLike,
    _ArrayLike,
    NDArray,
    _SupportsArrayFunc,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
)

# 定义类型变量 _T 和 _SCT
_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=generic)

# 定义回调函数类型 _MaskFunc，接受一个整数类型的 NumPy 数组和一个泛型参数，返回一个与 np.equal 兼容的数据类型数组
_MaskFunc = Callable[
    [NDArray[int_], _T],
    NDArray[number[Any] | np.bool | timedelta64 | datetime64 | object_],
]

# 定义模块的导出列表
__all__: list[str]

# 函数装饰器 @overload，用于指定函数的重载形式

# 函数 fliplr 的第一个重载形式，接受 _ArrayLike 类型参数 m，返回 _SCT 类型的 NumPy 数组
@overload
def fliplr(m: _ArrayLike[_SCT]) -> NDArray[_SCT]: ...

# 函数 fliplr 的第二个重载形式，接受 ArrayLike 类型参数 m，返回任意类型的 NumPy 数组
@overload
def fliplr(m: ArrayLike) -> NDArray[Any]: ...

# 函数 flipud 的第一个重载形式，接受 _ArrayLike 类型参数 m，返回 _SCT 类型的 NumPy 数组
@overload
def flipud(m: _ArrayLike[_SCT]) -> NDArray[_SCT]: ...

# 函数 flipud 的第二个重载形式，接受 ArrayLike 类型参数 m，返回任意类型的 NumPy 数组
@overload
def flipud(m: ArrayLike) -> NDArray[Any]: ...

# 函数 eye 的第一个重载形式，返回 float64 类型的 NumPy 数组
@overload
def eye(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...

# 函数 eye 的第二个重载形式，返回 _SCT 类型的 NumPy 数组
@overload
def eye(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: _DTypeLike[_SCT] = ...,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...

# 函数 eye 的第三个重载形式，返回任意类型的 NumPy 数组
@overload
def eye(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

# 函数 diag 的第一个重载形式，接受 _ArrayLike 类型参数 v，返回 _SCT 类型的 NumPy 数组
@overload
def diag(v: _ArrayLike[_SCT], k: int = ...) -> NDArray[_SCT]: ...

# 函数 diag 的第二个重载形式，接受 ArrayLike 类型参数 v，返回任意类型的 NumPy 数组
@overload
def diag(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...

# 函数 diagflat 的第一个重载形式，接受 _ArrayLike 类型参数 v，返回 _SCT 类型的 NumPy 数组
@overload
def diagflat(v: _ArrayLike[_SCT], k: int = ...) -> NDArray[_SCT]: ...

# 函数 diagflat 的第二个重载形式，接受 ArrayLike 类型参数 v，返回任意类型的 NumPy 数组
@overload
def diagflat(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...

# 函数 tri 的第一个重载形式，返回 float64 类型的 NumPy 数组
@overload
def tri(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[float64]: ...

# 函数 tri 的第二个重载形式，返回 _SCT 类型的 NumPy 数组
@overload
def tri(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: _DTypeLike[_SCT] = ...,
    *,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[_SCT]: ...

# 函数 tri 的第三个重载形式，返回任意类型的 NumPy 数组
@overload
def tri(
    N: int,
    M: None | int = ...,
    k: int = ...,
    dtype: DTypeLike = ...,
    *,
    like: None | _SupportsArrayFunc = ...
) -> NDArray[Any]: ...

# 函数 tril 的第一个重载形式，接受 _ArrayLike 类型参数 v，返回 _SCT 类型的 NumPy 数组
@overload
def tril(v: _ArrayLike[_SCT], k: int = ...) -> NDArray[_SCT]: ...

# 函数 tril 的第二个重载形式，接受 ArrayLike 类型参数 v，返回任意类型的 NumPy 数组
@overload
def tril(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...

# 函数 triu 的第一个重载形式，接受 _ArrayLike 类型参数 v，返回 _SCT 类型的 NumPy 数组
@overload
def triu(v: _ArrayLike[_SCT], k: int = ...) -> NDArray[_SCT]: ...

# 函数 triu 的第二个重载形式，接受 ArrayLike 类型参数 v，返回任意类型的 NumPy 数组
@overload
def triu(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...

# 函数 vander 的声明，类型检查被忽略（type: ignore[misc]），接受 _ArrayLikeInt_co 类型参数 x
# 返回类型未指定，应根据具体实现确定
def vander(
    x: _ArrayLikeInt_co,
    N: None | int = ...,
    increasing: bool = ...,



# 定义两个变量 N 和 increasing，分别类型为可空的整数或 None，和布尔值
N: None | int = ...,
increasing: bool = ...,


这段代码定义了两个变量 `N` 和 `increasing`。在类型注释中，`N` 的类型是可空的整数或者 `None`，而 `increasing` 是布尔类型。这些变量的具体值在这里用 `...` 表示，实际应用中应该有具体的赋值。
@overload
def vander(
    x: _ArrayLikeFloat_co,
    N: None | int = ...,
    increasing: bool = ...,
) -> NDArray[floating[Any]]: ...
@overload
def vander(
    x: _ArrayLikeComplex_co,
    N: None | int = ...,
    increasing: bool = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def vander(
    x: _ArrayLikeObject_co,
    N: None | int = ...,
    increasing: bool = ...,
) -> NDArray[object_]: ...


# 生成范德蒙矩阵（Vandermonde matrix）。
# 根据不同的输入类型和参数生成不同的 V 阵（Vandermonde matrix）。
# - x: 输入数组或类数组（_ArrayLike），可以是浮点数、复数或对象类型。
# - N: V 阵的列数，如果为 None 则取 x 的长度。
# - increasing: 控制 V 阵的列的递增或递减排列。
# 返回生成的 V 阵作为浮点数、复数或对象数组。

@overload
def histogram2d(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    bins: int | Sequence[int] = ...,
    range: None | _ArrayLikeFloat_co = ...,
    density: None | bool = ...,
    weights: None | _ArrayLikeFloat_co = ...,
) -> tuple[
    NDArray[float64],
    NDArray[floating[Any]],
    NDArray[floating[Any]],
]: ...
@overload
def histogram2d(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    bins: int | Sequence[int] = ...,
    range: None | _ArrayLikeFloat_co = ...,
    density: None | bool = ...,
    weights: None | _ArrayLikeFloat_co = ...,
) -> tuple[
    NDArray[float64],
    NDArray[complexfloating[Any, Any]],
    NDArray[complexfloating[Any, Any]],
]: ...
@overload  # TODO: Sort out `bins`
def histogram2d(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    bins: Sequence[_ArrayLikeInt_co],
    range: None | _ArrayLikeFloat_co = ...,
    density: None | bool = ...,
    weights: None | _ArrayLikeFloat_co = ...,
) -> tuple[
    NDArray[float64],
    NDArray[Any],
    NDArray[Any],
]: ...


# 生成二维直方图。
# 根据输入的数组 x 和 y，以及可选的 bins、range、density 和 weights 参数生成二维直方图。
# - x: 输入数组或类数组，可以是浮点数或复数。
# - y: 输入数组或类数组，可以是浮点数或复数，与 x 的形状必须一致。
# - bins: 定义直方图箱子的数量或边界。
# - range: 定义 x 和 y 的范围。
# - density: 如果为 True，则返回的直方图是归一化的概率密度。
# - weights: 可选的权重数组，用于每个点的加权统计。
# 返回一个包含三个数组的元组，分别是直方图统计值、x 方向上的直方图箱子边界、y 方向上的直方图箱子边界。

@overload
def mask_indices(
    n: int,
    mask_func: _MaskFunc[int],
    k: int = ...,
) -> tuple[NDArray[intp], NDArray[intp]]: ...
@overload
def mask_indices(
    n: int,
    mask_func: _MaskFunc[_T],
    k: _T,
) -> tuple[NDArray[intp], NDArray[intp]]: ...


# 根据掩码函数生成索引数组的元组。
# 根据给定的参数生成用于掩码的索引数组。
# - n: 生成的索引数组的维度。
# - mask_func: 一个函数，根据索引位置返回一个布尔值数组。
# - k: 控制生成的索引与对角线的关系。
# 返回一个元组，包含两个整数数组，分别是行索引和列索引。

def tril_indices(
    n: int,
    k: int = ...,
    m: None | int = ...,
) -> tuple[NDArray[int_], NDArray[int_]]: ...


# 返回一个下三角矩阵的索引元组。
# 返回一个下三角矩阵的所有行列索引。
# - n: 矩阵的行数和列数。
# - k: 控制生成的索引与主对角线的关系。
# - m: 可选参数，控制生成的索引的上限。

def tril_indices_from(
    arr: NDArray[Any],
    k: int = ...,
) -> tuple[NDArray[int_], NDArray[int_]]: ...


# 返回给定数组的下三角矩阵的索引元组。
# 返回给定数组的下三角矩阵的所有行列索引。
# - arr: 输入的数组，返回其下三角矩阵的索引。
# - k: 控制生成的索引与主对角线的关系。

def triu_indices(
    n: int,
    k: int = ...,
    m: None | int = ...,
) -> tuple[NDArray[int_], NDArray[int_]]: ...


# 返回一个上三角矩阵的索引元组。
# 返回一个上三角矩阵的所有行列索引。
# - n: 矩阵的行数和列数。
# - k: 控制生成的索引与主对角线的关系。
# - m: 可选参数，控制生成的索引的上限。

def triu_indices_from(
    arr: NDArray[Any],
    k: int = ...,
) -> tuple[NDArray[int_], NDArray[int_]]: ...


# 返回给定数组的上三角矩阵的索引元组。
# 返回给定数组的上三角矩阵的所有行列索引。
# - arr: 输入的数组，返回其上三角矩阵的索引。
# - k: 控制生成的索引与主对角线的关系。
```