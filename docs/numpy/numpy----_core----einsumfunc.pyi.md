# `D:\src\scipysrc\numpy\numpy\_core\einsumfunc.pyi`

```py
from collections.abc import Sequence
from typing import TypeVar, Any, overload, Literal

import numpy as np
from numpy import number, _OrderKACF
from numpy._typing import (
    NDArray,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
    _DTypeLikeBool,
    _DTypeLikeUInt,
    _DTypeLikeInt,
    _DTypeLikeFloat,
    _DTypeLikeComplex,
    _DTypeLikeComplex_co,
    _DTypeLikeObject,
)

_ArrayType = TypeVar(
    "_ArrayType",
    bound=NDArray[np.bool | number[Any]],
)

_OptimizeKind = None | bool | Literal["greedy", "optimal"] | Sequence[Any]
_CastingSafe = Literal["no", "equiv", "safe", "same_kind"]
_CastingUnsafe = Literal["unsafe"]

__all__: list[str]

# TODO: Properly handle the `casting`-based combinatorics
# TODO: We need to evaluate the content `__subscripts` in order
# to identify whether or an array or scalar is returned. At a cursory
# glance this seems like something that can quite easily be done with
# a mypy plugin.
# Something like `is_scalar = bool(__subscripts.partition("->")[-1])`

@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeBool_co,
    out: None = ...,
    dtype: None | _DTypeLikeBool = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeUInt_co,
    out: None = ...,
    dtype: None | _DTypeLikeUInt = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeInt_co,
    out: None = ...,
    dtype: None | _DTypeLikeInt = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeFloat_co,
    out: None = ...,
    dtype: None | _DTypeLikeFloat = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeComplex_co,
    out: None = ...,
    dtype: None | _DTypeLikeComplex = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    casting: _CastingUnsafe,
    dtype: None | _DTypeLikeComplex_co = ...,
    out: None = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeComplex_co,
    out: _ArrayType,
    dtype: None | _DTypeLikeComplex_co = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...


注释：

# 导入必要的模块和类型声明
from collections.abc import Sequence
from typing import TypeVar, Any, overload, Literal

import numpy as np
from numpy import number, _OrderKACF
from numpy._typing import (
    NDArray,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
    _DTypeLikeBool,
    _DTypeLikeUInt,
    _DTypeLikeInt,
    _DTypeLikeFloat,
    _DTypeLikeComplex,
    _DTypeLikeComplex_co,
    _DTypeLikeObject,
)

# 定义类型变量
_ArrayType = TypeVar(
    "_ArrayType",
    bound=NDArray[np.bool | number[Any]],
)

# 定义类型别名
_OptimizeKind = None | bool | Literal["greedy", "optimal"] | Sequence[Any]
_CastingSafe = Literal["no", "equiv", "safe", "same_kind"]
_CastingUnsafe = Literal["unsafe"]

# 声明 __all__ 列表
__all__: list[str]

# TODO: Properly handle the `casting`-based combinatorics
# TODO: We need to evaluate the content `__subscripts` in order
# to identify whether or an array or scalar is returned. At a cursory
# glance this seems like something that can quite easily be done with
# a mypy plugin.
# Something like `is_scalar = bool(__subscripts.partition("->")[-1])`

# 函数重载定义，根据不同的输入类型签名处理 `einsum` 函数
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeBool_co,
    out: None = ...,
    dtype: None | _DTypeLikeBool = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeUInt_co,
    out: None = ...,
    dtype: None | _DTypeLikeUInt = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeInt_co,
    out: None = ...,
    dtype: None | _DTypeLikeInt = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeFloat_co,
    out: None = ...,
    dtype: None | _DTypeLikeFloat = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeComplex_co,
    out: None = ...,
    dtype: None | _DTypeLikeComplex = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    casting: _CastingUnsafe,
    dtype: None | _DTypeLikeComplex_co = ...,
    out: None = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeComplex_co,
    out: _ArrayType,
    dtype: None | _DTypeLikeComplex_co = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any:
    ...
    optimize: _OptimizeKind = ...,


    # 定义一个名为 optimize 的变量，类型为 _OptimizeKind，初始值为省略号（未指定具体值）
# NOTE: `einsum` function with an overload taking specific arguments and returning `_ArrayType`.
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    out: _ArrayType,
    casting: _CastingUnsafe,
    dtype: None | _DTypeLikeComplex_co = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = ...,
) -> _ArrayType: ...

# NOTE: `einsum` function with an overload taking `_ArrayLikeObject_co` operands and returning `Any`.
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeObject_co,
    out: None = ...,
    dtype: None | _DTypeLikeObject = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> Any: ...

# NOTE: `einsum` function with an overload taking specific arguments and returning `Any`.
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    casting: _CastingUnsafe,
    dtype: None | _DTypeLikeObject = ...,
    out: None = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = ...,
) -> Any: ...

# NOTE: `einsum` function with an overload taking `_ArrayLikeObject_co` operands and returning `_ArrayType`.
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeObject_co,
    out: _ArrayType,
    dtype: None | _DTypeLikeObject = ...,
    order: _OrderKACF = ...,
    casting: _CastingSafe = ...,
    optimize: _OptimizeKind = ...,
) -> _ArrayType: ...

# NOTE: `einsum` function with an overload taking specific arguments and returning `_ArrayType`.
@overload
def einsum(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: Any,
    out: _ArrayType,
    casting: _CastingUnsafe,
    dtype: None | _DTypeLikeObject = ...,
    order: _OrderKACF = ...,
    optimize: _OptimizeKind = ...,
) -> _ArrayType: ...

# NOTE: `einsum_path` function calculates and returns the optimal path for `einsum`.
# The function takes subscripts and operands, where operands are either complex arrays or object types.
# The `optimize` parameter controls optimization behavior.
def einsum_path(
    subscripts: str | _ArrayLikeInt_co,
    /,
    *operands: _ArrayLikeComplex_co | _DTypeLikeObject,
    optimize: _OptimizeKind = ...,
) -> tuple[list[Any], str]: ...
```