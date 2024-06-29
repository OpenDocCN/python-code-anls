# `D:\src\scipysrc\numpy\numpy\lib\_ufunclike_impl.pyi`

```
# 引入必要的类型和模块
from typing import Any, overload, TypeVar

import numpy as np
from numpy import floating, object_
from numpy._typing import (
    NDArray,
    _FloatLike_co,
    _ArrayLikeFloat_co,
    _ArrayLikeObject_co,
)

# 定义一个类型变量 _ArrayType，它是一个绑定了 NDArray[Any] 的类型变量
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

# __all__ 列表用于声明在 from module import * 时需要导出的符号
__all__: list[str]

# 以下是函数 fix 的多个重载定义，用于修正不同类型输入的数据类型
@overload
def fix(
    x: _FloatLike_co,
    out: None = ...,
) -> floating[Any]: ...

@overload
def fix(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[floating[Any]]: ...

@overload
def fix(
    x: _ArrayLikeObject_co,
    out: None = ...,
) -> NDArray[object_]: ...

@overload
def fix(
    x: _ArrayLikeFloat_co | _ArrayLikeObject_co,
    out: _ArrayType,
) -> _ArrayType: ...

# 以下是函数 isposinf 的多个重载定义，用于检查输入是否为正无穷
@overload
def isposinf(
    x: _FloatLike_co,
    out: None = ...,
) -> np.bool: ...

@overload
def isposinf(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[np.bool]: ...

@overload
def isposinf(
    x: _ArrayLikeFloat_co,
    out: _ArrayType,
) -> _ArrayType: ...

# 以下是函数 isneginf 的多个重载定义，用于检查输入是否为负无穷
@overload
def isneginf(
    x: _FloatLike_co,
    out: None = ...,
) -> np.bool: ...

@overload
def isneginf(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[np.bool]: ...

@overload
def isneginf(
    x: _ArrayLikeFloat_co,
    out: _ArrayType,
) -> _ArrayType: ...
```