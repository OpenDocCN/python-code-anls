# `.\numpy\numpy\_typing\_array_like.py`

```py
from __future__ import annotations
# 导入未来版本兼容性的模块

import sys
# 导入系统相关的模块
from collections.abc import Collection, Callable, Sequence
# 导入集合、可调用对象和序列相关的抽象基类
from typing import Any, Protocol, Union, TypeVar, runtime_checkable
# 导入类型提示相关的模块

import numpy as np
# 导入NumPy库并使用别名np
from numpy import (
    ndarray,
    dtype,
    generic,
    unsignedinteger,
    integer,
    floating,
    complexfloating,
    number,
    timedelta64,
    datetime64,
    object_,
    void,
    str_,
    bytes_,
)
# 从NumPy库中导入多个特定的类和函数

from ._nested_sequence import _NestedSequence
# 从嵌套序列模块导入_NestedSequence类

_T = TypeVar("_T")
# 创建类型变量_T
_ScalarType = TypeVar("_ScalarType", bound=generic)
# 创建类型变量_ScalarType，限定为泛型类型
_ScalarType_co = TypeVar("_ScalarType_co", bound=generic, covariant=True)
# 创建协变的泛型类型变量_ScalarType_co
_DType = TypeVar("_DType", bound=dtype[Any])
# 创建类型变量_DType，限定为dtype类型
_DType_co = TypeVar("_DType_co", covariant=True, bound=dtype[Any])
# 创建协变的类型变量_DType_co

NDArray = ndarray[Any, dtype[_ScalarType_co]]
# 创建NDArray类型别名，表示任意形状的NumPy数组

# The `_SupportsArray` protocol only cares about the default dtype
# (i.e. `dtype=None` or no `dtype` parameter at all) of the to-be returned
# array.
# Concrete implementations of the protocol are responsible for adding
# any and all remaining overloads
@runtime_checkable
class _SupportsArray(Protocol[_DType_co]):
    def __array__(self) -> ndarray[Any, _DType_co]: ...
# 创建SupportsArray协议类，用于支持数组操作，协变于dtype类型_DType_co

@runtime_checkable
class _SupportsArrayFunc(Protocol):
    """A protocol class representing `~class.__array_function__`."""
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Collection[type[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object: ...
# 创建SupportsArrayFunc协议类，用于支持__array_function__方法的协议定义

# TODO: Wait until mypy supports recursive objects in combination with typevars
_FiniteNestedSequence = Union[
    _T,
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]
# 创建有限的嵌套序列类型别名，支持多级嵌套的序列类型

# A subset of `npt.ArrayLike` that can be parametrized w.r.t. `np.generic`
_ArrayLike = Union[
    _SupportsArray[dtype[_ScalarType]],
    _NestedSequence[_SupportsArray[dtype[_ScalarType]]],
]
# 创建ArrayLike类型别名，表示可以参数化为np.generic类型的npt.ArrayLike子集

# A union representing array-like objects; consists of two typevars:
# One representing types that can be parametrized w.r.t. `np.dtype`
# and another one for the rest
_DualArrayLike = Union[
    _SupportsArray[_DType],
    _NestedSequence[_SupportsArray[_DType]],
    _T,
    _NestedSequence[_T],
]
# 创建DualArrayLike类型别名，表示数组样式对象的联合类型，包含可参数化为np.dtype的类型和其他类型

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
    # 如果Python版本高于等于3.12，则导入Buffer抽象基类

    ArrayLike = Buffer | _DualArrayLike[
        dtype[Any],
        Union[bool, int, float, complex, str, bytes],
    ]
else:
    ArrayLike = _DualArrayLike[
        dtype[Any],
        Union[bool, int, float, complex, str, bytes],
    ]
# 根据Python版本选择性定义ArrayLike类型别名，支持缓冲区或其他类型

# `ArrayLike<X>_co`: array-like objects that can be coerced into `X`
# given the casting rules `same_kind`
_ArrayLikeBool_co = _DualArrayLike[
    dtype[np.bool],
    bool,
]
# 创建ArrayLikeBool_co类型别名，表示可以强制转换为布尔类型的数组样式对象

_ArrayLikeUInt_co = _DualArrayLike[
    dtype[Union[np.bool, unsignedinteger[Any]]],
    bool,
]
# 创建ArrayLikeUInt_co类型别名，表示可以强制转换为无符号整数类型的数组样式对象

_ArrayLikeInt_co = _DualArrayLike[
    dtype[Union[np.bool, integer[Any]]],
    Union[bool, int],
]
# 创建ArrayLikeInt_co类型别名，表示可以强制转换为整数类型的数组样式对象

_ArrayLikeFloat_co = _DualArrayLike[
    dtype[Union[np.bool, integer[Any], floating[Any]]],
    Union[bool, int, float],


# 定义一个类型注解，表示此处的变量或参数可以是 bool、int 或 float 类型之一
# 定义一个类型别名 _ArrayLikeComplex_co，表示复杂类型数组，可以包含布尔值、任意整数、任意浮点数、任意复数
_ArrayLikeComplex_co = _DualArrayLike[
    dtype[Union[
        np.bool,
        integer[Any],
        floating[Any],
        complexfloating[Any, Any],
    ]],
    Union[bool, int, float, complex],
]

# 定义一个类型别名 _ArrayLikeNumber_co，表示数字类型数组，可以包含布尔值或者任意数值类型
_ArrayLikeNumber_co = _DualArrayLike[
    dtype[Union[np.bool, number[Any]]],
    Union[bool, int, float, complex],
]

# 定义一个类型别名 _ArrayLikeTD64_co，表示时间差类型数组，可以包含布尔值、任意整数或者 timedelta64 类型
_ArrayLikeTD64_co = _DualArrayLike[
    dtype[Union[np.bool, integer[Any], timedelta64]],
    Union[bool, int],
]

# 定义一个类型别名 _ArrayLikeDT64_co，表示日期时间类型数组，可以是支持日期时间 dtype 的数组或者嵌套数组
_ArrayLikeDT64_co = Union[
    _SupportsArray[dtype[datetime64]],
    _NestedSequence[_SupportsArray[dtype[datetime64]]],
]

# 定义一个类型别名 _ArrayLikeObject_co，表示对象类型数组，可以是支持 object_ dtype 的数组或者嵌套数组
_ArrayLikeObject_co = Union[
    _SupportsArray[dtype[object_]],
    _NestedSequence[_SupportsArray[dtype[object_]]],
]

# 定义一个类型别名 _ArrayLikeVoid_co，表示空类型数组，可以是支持 void dtype 的数组或者嵌套数组
_ArrayLikeVoid_co = Union[
    _SupportsArray[dtype[void]],
    _NestedSequence[_SupportsArray[dtype[void]]],
]

# 定义一个类型别名 _ArrayLikeStr_co，表示字符串类型数组，可以包含 str_ dtype 或者普通字符串类型
_ArrayLikeStr_co = _DualArrayLike[
    dtype[str_],
    str,
]

# 定义一个类型别名 _ArrayLikeBytes_co，表示字节类型数组，可以包含 bytes_ dtype 或者普通字节串类型
_ArrayLikeBytes_co = _DualArrayLike[
    dtype[bytes_],
    bytes,
]

# 定义一个类型别名 _ArrayLikeInt，表示整数类型数组，可以包含任意整数类型
_ArrayLikeInt = _DualArrayLike[
    dtype[integer[Any]],
    int,
]

# 定义一个特殊类型 _UnknownType，用于处理 NDArray[Any]，但不匹配任何具体类型
# 用作第一个重载，只匹配 NDArray[Any]，而不匹配任何实际类型
# 参考：https://github.com/numpy/numpy/pull/22193
class _UnknownType:
    ...

# 定义一个类型别名 _ArrayLikeUnknown，表示未知类型数组，可以包含 _UnknownType 或者其 dtype 类型
_ArrayLikeUnknown = _DualArrayLike[
    dtype[_UnknownType],
    _UnknownType,
]
```