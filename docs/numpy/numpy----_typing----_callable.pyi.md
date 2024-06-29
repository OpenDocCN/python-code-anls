# `D:\src\scipysrc\numpy\numpy\_typing\_callable.pyi`

```py
"""
A module with various ``typing.Protocol`` subclasses that implement
the ``__call__`` magic method.

See the `Mypy documentation`_ on protocols for more details.

.. _`Mypy documentation`: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols

"""

# 导入必要的模块和类
from __future__ import annotations

from typing import (
    TypeVar,
    overload,
    Any,
    NoReturn,
    Protocol,
)

import numpy as np
from numpy import (
    generic,
    timedelta64,
    number,
    integer,
    unsignedinteger,
    signedinteger,
    int8,
    int_,
    floating,
    float64,
    complexfloating,
    complex128,
)
from ._nbit import _NBitInt, _NBitDouble
from ._scalars import (
    _BoolLike_co,
    _IntLike_co,
    _FloatLike_co,
    _NumberLike_co,
)
from . import NBitBase
from ._array_like import NDArray
from ._nested_sequence import _NestedSequence

# 定义类型变量
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T1_contra = TypeVar("_T1_contra", contravariant=True)
_T2_contra = TypeVar("_T2_contra", contravariant=True)
_2Tuple = tuple[_T1, _T1]

_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

_IntType = TypeVar("_IntType", bound=integer)
_FloatType = TypeVar("_FloatType", bound=floating)
_NumberType = TypeVar("_NumberType", bound=number)
_NumberType_co = TypeVar("_NumberType_co", covariant=True, bound=number)
_GenericType_co = TypeVar("_GenericType_co", covariant=True, bound=generic)

# 定义 `_BoolOp` 协议，包含多个 `__call__` 方法的重载
class _BoolOp(Protocol[_GenericType_co]):
    @overload
    def __call__(self, other: _BoolLike_co, /) -> _GenericType_co: ...
    @overload  # 平台相关
    def __call__(self, other: int, /) -> int_: ...
    @overload
    def __call__(self, other: float, /) -> float64: ...
    @overload
    def __call__(self, other: complex, /) -> complex128: ...
    @overload
    def __call__(self, other: _NumberType, /) -> _NumberType: ...

# 定义 `_BoolBitOp` 协议，包含多个 `__call__` 方法的重载
class _BoolBitOp(Protocol[_GenericType_co]):
    @overload
    def __call__(self, other: _BoolLike_co, /) -> _GenericType_co: ...
    @overload  # 平台相关
    def __call__(self, other: int, /) -> int_: ...
    @overload
    def __call__(self, other: _IntType, /) -> _IntType: ...

# 定义 `_BoolSub` 协议，包含多个 `__call__` 方法的重载
class _BoolSub(Protocol):
    # 注意：这里没有 `other: bool`
    @overload
    def __call__(self, other: bool, /) -> NoReturn: ...
    @overload  # 平台相关
    def __call__(self, other: int, /) -> int_: ...
    @overload
    def __call__(self, other: float, /) -> float64: ...
    @overload
    def __call__(self, other: complex, /) -> complex128: ...
    @overload
    def __call__(self, other: _NumberType, /) -> _NumberType: ...

# 定义 `_BoolTrueDiv` 协议，包含多个 `__call__` 方法的重载
class _BoolTrueDiv(Protocol):
    @overload
    def __call__(self, other: float | _IntLike_co, /) -> float64: ...
    @overload
    def __call__(self, other: complex, /) -> complex128: ...
    @overload
    def __call__(self, other: _NumberType, /) -> _NumberType: ...

# 定义 `_BoolMod` 协议，包含多个 `__call__` 方法的重载
class _BoolMod(Protocol):
    @overload
    def __call__(self, other: _BoolLike_co, /) -> int8: ...
    # 使用 @overload 装饰器指定多个重载函数，具体实现因平台而异
    @overload  # platform dependent
    # 定义一个接受整数参数并返回整数类型结果的重载函数
    def __call__(self, other: int, /) -> int_: ...
    # 定义一个接受浮点数参数并返回浮点数类型结果的重载函数
    @overload
    def __call__(self, other: float, /) -> float64: ...
    # 定义一个接受 _IntType 类型参数并返回相同类型结果的重载函数
    @overload
    def __call__(self, other: _IntType, /) -> _IntType: ...
    # 定义一个接受 _FloatType 类型参数并返回相同类型结果的重载函数
    @overload
    def __call__(self, other: _FloatType, /) -> _FloatType: ...
class _BoolDivMod(Protocol):
    # 定义一个协议 _BoolDivMod，用于处理不同类型参数的除法和取模操作
    @overload
    def __call__(self, other: _BoolLike_co, /) -> _2Tuple[int8]: ...
    @overload  # 平台相关
    def __call__(self, other: int, /) -> _2Tuple[int_]: ...
    @overload
    def __call__(self, other: float, /) -> _2Tuple[floating[_NBit1 | _NBitDouble]]: ...
    @overload
    def __call__(self, other: _IntType, /) -> _2Tuple[_IntType]: ...
    @overload
    def __call__(self, other: _FloatType, /) -> _2Tuple[_FloatType]: ...


class _TD64Div(Protocol[_NumberType_co]):
    # 定义一个协议 _TD64Div，用于处理 timedelta64 类型的除法操作
    @overload
    def __call__(self, other: timedelta64, /) -> _NumberType_co: ...
    @overload
    def __call__(self, other: _BoolLike_co, /) -> NoReturn: ...
    @overload
    def __call__(self, other: _FloatLike_co, /) -> timedelta64: ...


class _IntTrueDiv(Protocol[_NBit1]):
    # 定义一个协议 _IntTrueDiv，用于处理整数类型的真除操作
    @overload
    def __call__(self, other: bool, /) -> floating[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> floating[_NBit1 | _NBitInt]: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    @overload
    def __call__(self, other: integer[_NBit2], /) -> floating[_NBit1 | _NBit2]: ...


class _UnsignedIntOp(Protocol[_NBit1]):
    # 定义一个协议 _UnsignedIntOp，用于处理无符号整数类型的操作
    # 注意：`uint64 + signedinteger -> float64`
    @overload
    def __call__(self, other: bool, /) -> unsignedinteger[_NBit1]: ...
    @overload
    def __call__(
        self, other: int | signedinteger[Any], /
    ) -> Any: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: unsignedinteger[_NBit2], /
    ) -> unsignedinteger[_NBit1 | _NBit2]: ...


class _UnsignedIntBitOp(Protocol[_NBit1]):
    # 定义一个协议 _UnsignedIntBitOp，用于处理无符号整数类型的位操作
    @overload
    def __call__(self, other: bool, /) -> unsignedinteger[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> signedinteger[Any]: ...
    @overload
    def __call__(self, other: signedinteger[Any], /) -> signedinteger[Any]: ...
    @overload
    def __call__(
        self, other: unsignedinteger[_NBit2], /
    ) -> unsignedinteger[_NBit1 | _NBit2]: ...


class _UnsignedIntMod(Protocol[_NBit1]):
    # 定义一个协议 _UnsignedIntMod，用于处理无符号整数类型的取模操作
    @overload
    def __call__(self, other: bool, /) -> unsignedinteger[_NBit1]: ...
    @overload
    def __call__(
        self, other: int | signedinteger[Any], /
    ) -> Any: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: unsignedinteger[_NBit2], /
    ) -> unsignedinteger[_NBit1 | _NBit2]: ...


class _UnsignedIntDivMod(Protocol[_NBit1]):
    # 定义一个协议 _UnsignedIntDivMod，用于处理无符号整数类型的除法和取模操作
    @overload
    def __call__(self, other: bool, /) -> _2Tuple[signedinteger[_NBit1]]: ...
    @overload
    # 定义 __call__ 方法，使该类的实例可以被调用
    def __call__(
        self, other: int | signedinteger[Any], /
    ) -> _2Tuple[Any]: ...

    # __call__ 方法的重载，用于处理参数为 float 类型的情况
    @overload
    def __call__(self, other: float, /) -> _2Tuple[floating[_NBit1 | _NBitDouble]]: ...

    # __call__ 方法的重载，用于处理参数为 unsignedinteger 类型的情况
    @overload
    def __call__(
        self, other: unsignedinteger[_NBit2], /
    ) -> _2Tuple[unsignedinteger[_NBit1 | _NBit2]]: ...
class _SignedIntOp(Protocol[_NBit1]):
    # 定义 _SignedIntOp 协议，指定其支持的多态函数签名

    @overload
    def __call__(self, other: bool, /) -> signedinteger[_NBit1]: ...
    # 定义协议方法的重载：接受布尔类型参数，返回指定位数的有符号整数

    @overload
    def __call__(self, other: int, /) -> signedinteger[_NBit1 | _NBitInt]: ...
    # 定义协议方法的重载：接受整数类型参数，返回指定位数的有符号整数或整数类型

    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    # 定义协议方法的重载：接受浮点数类型参数，返回指定位数的浮点数

    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    # 定义协议方法的重载：接受复数类型参数，返回指定位数的复数

    @overload
    def __call__(
        self, other: signedinteger[_NBit2], /,
    ) -> signedinteger[_NBit1 | _NBit2]: ...
    # 定义协议方法的重载：接受指定位数的有符号整数参数，返回指定位数的有符号整数

class _SignedIntBitOp(Protocol[_NBit1]):
    # 定义 _SignedIntBitOp 协议，指定其支持的多态函数签名

    @overload
    def __call__(self, other: bool, /) -> signedinteger[_NBit1]: ...
    # 定义协议方法的重载：接受布尔类型参数，返回指定位数的有符号整数

    @overload
    def __call__(self, other: int, /) -> signedinteger[_NBit1 | _NBitInt]: ...
    # 定义协议方法的重载：接受整数类型参数，返回指定位数的有符号整数或整数类型

    @overload
    def __call__(
        self, other: signedinteger[_NBit2], /,
    ) -> signedinteger[_NBit1 | _NBit2]: ...
    # 定义协议方法的重载：接受指定位数的有符号整数参数，返回指定位数的有符号整数

class _SignedIntMod(Protocol[_NBit1]):
    # 定义 _SignedIntMod 协议，指定其支持的多态函数签名

    @overload
    def __call__(self, other: bool, /) -> signedinteger[_NBit1]: ...
    # 定义协议方法的重载：接受布尔类型参数，返回指定位数的有符号整数

    @overload
    def __call__(self, other: int, /) -> signedinteger[_NBit1 | _NBitInt]: ...
    # 定义协议方法的重载：接受整数类型参数，返回指定位数的有符号整数或整数类型

    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    # 定义协议方法的重载：接受浮点数类型参数，返回指定位数的浮点数

    @overload
    def __call__(
        self, other: signedinteger[_NBit2], /,
    ) -> signedinteger[_NBit1 | _NBit2]: ...
    # 定义协议方法的重载：接受指定位数的有符号整数参数，返回指定位数的有符号整数

class _SignedIntDivMod(Protocol[_NBit1]):
    # 定义 _SignedIntDivMod 协议，指定其支持的多态函数签名

    @overload
    def __call__(self, other: bool, /) -> _2Tuple[signedinteger[_NBit1]]: ...
    # 定义协议方法的重载：接受布尔类型参数，返回一个元组，包含两个指定位数的有符号整数

    @overload
    def __call__(self, other: int, /) -> _2Tuple[signedinteger[_NBit1 | _NBitInt]]: ...
    # 定义协议方法的重载：接受整数类型参数，返回一个元组，包含两个指定位数的有符号整数或整数类型

    @overload
    def __call__(self, other: float, /) -> _2Tuple[floating[_NBit1 | _NBitDouble]]: ...
    # 定义协议方法的重载：接受浮点数类型参数，返回一个元组，包含两个指定位数的浮点数

    @overload
    def __call__(
        self, other: signedinteger[_NBit2], /,
    ) -> _2Tuple[signedinteger[_NBit1 | _NBit2]]: ...
    # 定义协议方法的重载：接受指定位数的有符号整数参数，返回一个元组，包含两个指定位数的有符号整数

class _FloatOp(Protocol[_NBit1]):
    # 定义 _FloatOp 协议，指定其支持的多态函数签名

    @overload
    def __call__(self, other: bool, /) -> floating[_NBit1]: ...
    # 定义协议方法的重载：接受布尔类型参数，返回指定位数的浮点数

    @overload
    def __call__(self, other: int, /) -> floating[_NBit1 | _NBitInt]: ...
    # 定义协议方法的重载：接受整数类型参数，返回指定位数的浮点数或整数类型

    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    # 定义协议方法的重载：接受浮点数类型参数，返回指定位数的浮点数

    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    # 定义协议方法的重载：接受复数类型参数，返回指定位数的复数

    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> floating[_NBit1 | _NBit2]: ...
    # 定义协议方法的重载：接受指定位数的整数或浮点数类型参数，返回指定位数的浮点数

class _FloatMod(Protocol[_NBit1]):
    # 定义 _FloatMod 协议，指定其支持的多态函数签名

    @overload
    def __call__(self, other: bool, /) -> floating[_NBit1]: ...
    # 定义协议方法的重载：接受布尔类型参数，返回指定位数的浮点数

    @overload
    def __call__(self, other: int, /) -> floating[_NBit1 | _NBitInt]: ...
    # 定义协议方法的重载：接受整数类型参数，返回指定位数的浮点数或整数类型

    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    # 定义协议方法的重载：接受浮点数类型参数，返回指定位数的浮点数

    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> floating[_NBit1 | _NBit2]: ...
    # 定义协议方法的重载：接受指定位数的整数或浮点数类型参数，返回指定位数的浮点数

class _FloatDivMod(Protocol[_NBit1]):
    # 定义 _FloatDivMod 协议，指定其支持的多态函数签
    # 定义 __call__ 方法，使对象可以被调用
    def __call__(self, other: int, /) -> _2Tuple[floating[_NBit1 | _NBitInt]]: ...
    # 重载 __call__ 方法，支持参数类型为 float 的调用
    @overload
    def __call__(self, other: float, /) -> _2Tuple[floating[_NBit1 | _NBitDouble]]: ...
    # 重载 __call__ 方法，支持参数类型为 integer 或 floating 的调用，参数类型为 _NBit2
    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> _2Tuple[floating[_NBit1 | _NBit2]]: ...
# 定义一个复杂运算的协议 `_ComplexOp`
class _ComplexOp(Protocol[_NBit1]):
    # 重载：接受布尔类型参数，返回复数浮点数
    @overload
    def __call__(self, other: bool, /) -> complexfloating[_NBit1, _NBit1]: ...
    
    # 重载：接受整数参数，返回复数浮点数
    @overload
    def __call__(self, other: int, /) -> complexfloating[_NBit1 | _NBitInt, _NBit1 | _NBitInt]: ...
    
    # 重载：接受复数参数，返回复数浮点数
    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    
    # 重载：接受整数、浮点数或复数参数，返回复数浮点数
    @overload
    def __call__(
        self,
        other: (
            integer[_NBit2]
            | floating[_NBit2]
            | complexfloating[_NBit2, _NBit2]
        ), /,
    ) -> complexfloating[_NBit1 | _NBit2, _NBit1 | _NBit2]: ...

# 定义一个数字操作的协议 `_NumberOp`
class _NumberOp(Protocol):
    # 重载：接受 `_NumberLike_co` 类型的参数，返回任意类型
    def __call__(self, other: _NumberLike_co, /) -> Any: ...

# 定义支持小于比较的协议 `_SupportsLT`
class _SupportsLT(Protocol):
    # 方法：小于比较，接受任意类型参数，返回对象
    def __lt__(self, other: Any, /) -> object: ...

# 定义支持大于比较的协议 `_SupportsGT`
class _SupportsGT(Protocol):
    # 方法：大于比较，接受任意类型参数，返回对象
    def __gt__(self, other: Any, /) -> object: ...

# 定义比较操作的协议 `_ComparisonOp`，参数类型为 `_T1_contra` 和 `_T2_contra`
class _ComparisonOp(Protocol[_T1_contra, _T2_contra]):
    # 重载：接受 `_T1_contra` 类型参数，返回 NumPy 布尔类型
    @overload
    def __call__(self, other: _T1_contra, /) -> np.bool: ...
    
    # 重载：接受 `_T2_contra` 类型参数，返回 NumPy 布尔数组
    @overload
    def __call__(self, other: _T2_contra, /) -> NDArray[np.bool]: ...
    
    # 重载：接受 `_SupportsLT` 或 `_SupportsGT` 或 `_NestedSequence[_SupportsLT | _SupportsGT]` 参数，
    # 返回任意类型
    @overload
    def __call__(
        self,
        other: _SupportsLT | _SupportsGT | _NestedSequence[_SupportsLT | _SupportsGT],
        /,
    ) -> Any: ...
```