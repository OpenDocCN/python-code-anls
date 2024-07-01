# `.\numpy\numpy\_core\_asarray.pyi`

```py
# 导入所需模块和类型定义
from collections.abc import Iterable
from typing import Any, TypeVar, overload, Literal

from numpy._typing import NDArray, DTypeLike, _SupportsArrayFunc

# 定义类型变量
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

# 定义需求标记的字面量类型
_Requirements = Literal[
    "C", "C_CONTIGUOUS", "CONTIGUOUS",
    "F", "F_CONTIGUOUS", "FORTRAN",
    "A", "ALIGNED",
    "W", "WRITEABLE",
    "O", "OWNDATA"
]
# 定义带有'E'需求标记的字面量类型
_E = Literal["E", "ENSUREARRAY"]
_RequirementsWithE = _Requirements | _E

# 函数重载，用于对输入数组或对象施加特定要求
@overload
def require(
    a: _ArrayType,
    dtype: None = ...,
    requirements: None | _Requirements | Iterable[_Requirements] = ...,
    *,
    like: _SupportsArrayFunc = ...
) -> _ArrayType: ...
@overload
def require(
    a: object,
    dtype: DTypeLike = ...,
    requirements: _E | Iterable[_RequirementsWithE] = ...,
    *,
    like: _SupportsArrayFunc = ...
) -> NDArray[Any]: ...
@overload
def require(
    a: object,
    dtype: DTypeLike = ...,
    requirements: None | _Requirements | Iterable[_Requirements] = ...,
    *,
    like: _SupportsArrayFunc = ...
) -> NDArray[Any]: ...
```