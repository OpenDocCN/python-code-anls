# `.\numpy\numpy\ctypeslib.pyi`

```py
# NOTE: Numpy's mypy plugin is used for importing the correct
# platform-specific `ctypes._SimpleCData[int]` sub-type
# 导入正确的平台特定的 `ctypes._SimpleCData[int]` 子类型，使用了 Numpy 的 mypy 插件

from ctypes import c_int64 as _c_intp

import os
import ctypes
from collections.abc import Iterable, Sequence
from typing import (
    Literal as L,
    Any,
    TypeVar,
    Generic,
    overload,
    ClassVar,
)

import numpy as np
from numpy import (
    ndarray,
    dtype,
    generic,
    byte,
    short,
    intc,
    long,
    longlong,
    intp,
    ubyte,
    ushort,
    uintc,
    ulong,
    ulonglong,
    uintp,
    single,
    double,
    longdouble,
    void,
)
from numpy._core._internal import _ctypes
from numpy._core.multiarray import flagsobj
from numpy._typing import (
    # Arrays
    NDArray,
    _ArrayLike,

    # Shapes
    _ShapeLike,

    # DTypes
    DTypeLike,
    _DTypeLike,
    _VoidDTypeLike,
    _BoolCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _ULongCodes,
    _ULongLongCodes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _LongCodes,
    _LongLongCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
)

# TODO: Add a proper `_Shape` bound once we've got variadic typevars
# TODO: 添加适当的 `_Shape` 约束一旦我们有了变长类型变量（PEP 646）

_DType = TypeVar("_DType", bound=dtype[Any])
_DTypeOptional = TypeVar("_DTypeOptional", bound=None | dtype[Any])
_SCT = TypeVar("_SCT", bound=generic)

_FlagsKind = L[
    'C_CONTIGUOUS', 'CONTIGUOUS', 'C',
    'F_CONTIGUOUS', 'FORTRAN', 'F',
    'ALIGNED', 'A',
    'WRITEABLE', 'W',
    'OWNDATA', 'O',
    'WRITEBACKIFCOPY', 'X',
]

# TODO: Add a shape typevar once we have variadic typevars (PEP 646)
# TODO: 一旦有了变长类型变量，添加一个形状类型变量（PEP 646）

class _ndptr(ctypes.c_void_p, Generic[_DTypeOptional]):
    # In practice these 4 classvars are defined in the dynamic class
    # returned by `ndpointer`
    # 实际上这四个类变量在由 `ndpointer` 返回的动态类中定义
    _dtype_: ClassVar[_DTypeOptional]
    _shape_: ClassVar[None]
    _ndim_: ClassVar[None | int]
    _flags_: ClassVar[None | list[_FlagsKind]]

    @overload
    @classmethod
    def from_param(cls: type[_ndptr[None]], obj: NDArray[Any]) -> _ctypes[Any]: ...
    @overload
    @classmethod
    def from_param(cls: type[_ndptr[_DType]], obj: ndarray[Any, _DType]) -> _ctypes[Any]: ...

class _concrete_ndptr(_ndptr[_DType]):
    _dtype_: ClassVar[_DType]
    _shape_: ClassVar[tuple[int, ...]]
    @property
    def contents(self) -> ndarray[Any, _DType]: ...
    # 属性方法，返回 ndarray，其元素类型为 `_DType`

def load_library(
    libname: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    loader_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
) -> ctypes.CDLL:
    # 加载动态链接库，返回 ctypes.CDLL 对象
    pass

__all__: list[str]

c_intp = _c_intp

@overload
def ndpointer(
    dtype: None = ...,
    ndim: int = ...,
    shape: None | _ShapeLike = ...,
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,
) -> type[_ndptr[None]]: ...
@overload
def ndpointer(
    dtype: _DTypeLike[_SCT],
    ndim: int = ...,
    *,
    shape: _ShapeLike,
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,
) -> type[_concrete_ndptr[dtype[_SCT]]]: ...
@overload
def ndpointer(
    dtype: DTypeLike,
    ndim: int = ...,
    *,
    shape: _ShapeLike = ...,
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,
) -> type[_ndptr[dtype]]: ...
    # 根据指定的参数生成 ndarray 指针类型，支持多种重载形式
    pass
    ndim: int = ...,  # 定义一个类型为整数的变量 ndim，并初始化为占位符 ...
    *,  # 分隔位置参数和关键字参数的标记
    shape: _ShapeLike,  # 定义一个形状变量 shape，其类型为 _ShapeLike
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,  # 定义一个 flags 变量，其类型可以是 None、_FlagsKind 类型、_FlagsKind 的可迭代对象、整数或 flagsobj 类型，并初始化为占位符 ...
# 定义一个函数签名，用于创建特定类型的指针对象
def ndpointer(
    dtype: _DTypeLike[_SCT],    # 数据类型参数，可以是具体类型或类型的别名
    ndim: int = ...,            # 数组的维度，默认为省略值，表示维度不固定
    shape: None = ...,          # 数组的形状，默认为None，表示形状不固定
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,  # 标志参数，可以是None、单个标志、标志集合或整数
) -> type[_ndptr[dtype[_SCT]]]:  # 返回值类型为特定数据类型的指针类型

# 函数重载，支持更多的数据类型作为输入
def ndpointer(
    dtype: DTypeLike,           # 数据类型参数，可以是具体类型或类型的别名
    ndim: int = ...,            # 数组的维度，默认为省略值，表示维度不固定
    shape: None = ...,          # 数组的形状，默认为None，表示形状不固定
    flags: None | _FlagsKind | Iterable[_FlagsKind] | int | flagsobj = ...,  # 标志参数，可以是None、单个标志、标志集合或整数
) -> type[_ndptr[dtype[Any]]]:  # 返回值类型为任意数据类型的指针类型

# 函数重载，将特定的 NumPy 数据类型转换为对应的 ctypes 类型
def as_ctypes_type(dtype: _BoolCodes | _DTypeLike[np.bool] | type[ctypes.c_bool]) -> type[ctypes.c_bool]:  # 返回值类型为 ctypes.c_bool 类型

def as_ctypes_type(dtype: _ByteCodes | _DTypeLike[byte] | type[ctypes.c_byte]) -> type[ctypes.c_byte]:  # 返回值类型为 ctypes.c_byte 类型

def as_ctypes_type(dtype: _ShortCodes | _DTypeLike[short] | type[ctypes.c_short]) -> type[ctypes.c_short]:  # 返回值类型为 ctypes.c_short 类型

def as_ctypes_type(dtype: _IntCCodes | _DTypeLike[intc] | type[ctypes.c_int]) -> type[ctypes.c_int]:  # 返回值类型为 ctypes.c_int 类型

def as_ctypes_type(dtype: _LongCodes | _DTypeLike[long] | type[ctypes.c_long]) -> type[ctypes.c_long]:  # 返回值类型为 ctypes.c_long 类型

def as_ctypes_type(dtype: type[int]) -> type[c_intp]:  # 返回值类型为 c_intp 类型

def as_ctypes_type(dtype: _LongLongCodes | _DTypeLike[longlong] | type[ctypes.c_longlong]) -> type[ctypes.c_longlong]:  # 返回值类型为 ctypes.c_longlong 类型

def as_ctypes_type(dtype: _UByteCodes | _DTypeLike[ubyte] | type[ctypes.c_ubyte]) -> type[ctypes.c_ubyte]:  # 返回值类型为 ctypes.c_ubyte 类型

def as_ctypes_type(dtype: _UShortCodes | _DTypeLike[ushort] | type[ctypes.c_ushort]) -> type[ctypes.c_ushort]:  # 返回值类型为 ctypes.c_ushort 类型

def as_ctypes_type(dtype: _UIntCCodes | _DTypeLike[uintc] | type[ctypes.c_uint]) -> type[ctypes.c_uint]:  # 返回值类型为 ctypes.c_uint 类型

def as_ctypes_type(dtype: _ULongCodes | _DTypeLike[ulong] | type[ctypes.c_ulong]) -> type[ctypes.c_ulong]:  # 返回值类型为 ctypes.c_ulong 类型

def as_ctypes_type(dtype: _ULongLongCodes | _DTypeLike[ulonglong] | type[ctypes.c_ulonglong]) -> type[ctypes.c_ulonglong]:  # 返回值类型为 ctypes.c_ulonglong 类型

def as_ctypes_type(dtype: _SingleCodes | _DTypeLike[single] | type[ctypes.c_float]) -> type[ctypes.c_float]:  # 返回值类型为 ctypes.c_float 类型

def as_ctypes_type(dtype: _DoubleCodes | _DTypeLike[double] | type[float | ctypes.c_double]) -> type[ctypes.c_double]:  # 返回值类型为 ctypes.c_double 类型

def as_ctypes_type(dtype: _LongDoubleCodes | _DTypeLike[longdouble] | type[ctypes.c_longdouble]) -> type[ctypes.c_longdouble]:  # 返回值类型为 ctypes.c_longdouble 类型

def as_ctypes_type(dtype: _VoidDTypeLike) -> type[Any]:  # 返回值类型为任意类型，通常用于 ctypes.Union 或 ctypes.Structure

def as_ctypes_type(dtype: str) -> type[Any]:  # 返回值类型为任意类型，接受字符串参数

# 函数重载，将 ctypes 类型的对象转换为 NumPy 数组
def as_array(obj: ctypes._PointerLike, shape: Sequence[int]) -> NDArray[Any]:  # 接受 ctypes 指针对象和形状参数，返回 NumPy 数组

def as_array(obj: _ArrayLike[_SCT], shape: None | _ShapeLike = ...) -> NDArray[_SCT]:  # 接受数组对象和形状参数，返回 NumPy 数组

def as_array(obj: object, shape: None | _ShapeLike = ...) -> NDArray[Any]:  # 接受任意对象和形状参数，返回 NumPy 数组

# 函数重载，将 NumPy 对象转换为对应的 ctypes 类型
def as_ctypes(obj: np.bool) -> ctypes.c_bool:  # 将 NumPy 布尔值转换为 ctypes.c_bool 类型

def as_ctypes(obj: byte) -> ctypes.c_byte:  # 将 NumPy 字节值转换为 ctypes.c_byte 类型

def as_ctypes(obj: short) -> ctypes.c_short:  # 将 NumPy 短整数值转换为 ctypes.c_short 类型

def as_ctypes(obj: intc) -> ctypes.c_int:  # 将 NumPy 整数值转换为 ctypes.c_int 类型
# 将 long 类型对象转换为 ctypes.c_long 类型
@overload
def as_ctypes(obj: long) -> ctypes.c_long: ...

# 将 longlong 类型对象转换为 ctypes.c_longlong 类型
@overload
def as_ctypes(obj: longlong) -> ctypes.c_longlong: ...

# 将 ubyte 类型对象转换为 ctypes.c_ubyte 类型
@overload
def as_ctypes(obj: ubyte) -> ctypes.c_ubyte: ...

# 将 ushort 类型对象转换为 ctypes.c_ushort 类型
@overload
def as_ctypes(obj: ushort) -> ctypes.c_ushort: ...

# 将 uintc 类型对象转换为 ctypes.c_uint 类型
@overload
def as_ctypes(obj: uintc) -> ctypes.c_uint: ...

# 将 ulong 类型对象转换为 ctypes.c_ulong 类型
@overload
def as_ctypes(obj: ulong) -> ctypes.c_ulong: ...

# 将 ulonglong 类型对象转换为 ctypes.c_ulonglong 类型
@overload
def as_ctypes(obj: ulonglong) -> ctypes.c_ulonglong: ...

# 将 single 类型对象转换为 ctypes.c_float 类型
@overload
def as_ctypes(obj: single) -> ctypes.c_float: ...

# 将 double 类型对象转换为 ctypes.c_double 类型
@overload
def as_ctypes(obj: double) -> ctypes.c_double: ...

# 将 longdouble 类型对象转换为 ctypes.c_longdouble 类型
@overload
def as_ctypes(obj: longdouble) -> ctypes.c_longdouble: ...

# 将 void 类型对象转换为 ctypes.Union 或 ctypes.Structure 类型
@overload
def as_ctypes(obj: void) -> Any:  # `ctypes.Union` or `ctypes.Structure`
    ...

# 将 NDArray[np.bool] 类型对象转换为 ctypes.Array[ctypes.c_bool] 类型
@overload
def as_ctypes(obj: NDArray[np.bool]) -> ctypes.Array[ctypes.c_bool]: ...

# 将 NDArray[byte] 类型对象转换为 ctypes.Array[ctypes.c_byte] 类型
@overload
def as_ctypes(obj: NDArray[byte]) -> ctypes.Array[ctypes.c_byte]: ...

# 将 NDArray[short] 类型对象转换为 ctypes.Array[ctypes.c_short] 类型
@overload
def as_ctypes(obj: NDArray[short]) -> ctypes.Array[ctypes.c_short]: ...

# 将 NDArray[intc] 类型对象转换为 ctypes.Array[ctypes.c_int] 类型
@overload
def as_ctypes(obj: NDArray[intc]) -> ctypes.Array[ctypes.c_int]: ...

# 将 NDArray[long] 类型对象转换为 ctypes.Array[ctypes.c_long] 类型
@overload
def as_ctypes(obj: NDArray[long]) -> ctypes.Array[ctypes.c_long]: ...

# 将 NDArray[longlong] 类型对象转换为 ctypes.Array[ctypes.c_longlong] 类型
@overload
def as_ctypes(obj: NDArray[longlong]) -> ctypes.Array[ctypes.c_longlong]: ...

# 将 NDArray[ubyte] 类型对象转换为 ctypes.Array[ctypes.c_ubyte] 类型
@overload
def as_ctypes(obj: NDArray[ubyte]) -> ctypes.Array[ctypes.c_ubyte]: ...

# 将 NDArray[ushort] 类型对象转换为 ctypes.Array[ctypes.c_ushort] 类型
@overload
def as_ctypes(obj: NDArray[ushort]) -> ctypes.Array[ctypes.c_ushort]: ...

# 将 NDArray[uintc] 类型对象转换为 ctypes.Array[ctypes.c_uint] 类型
@overload
def as_ctypes(obj: NDArray[uintc]) -> ctypes.Array[ctypes.c_uint]: ...

# 将 NDArray[ulong] 类型对象转换为 ctypes.Array[ctypes.c_ulong] 类型
@overload
def as_ctypes(obj: NDArray[ulong]) -> ctypes.Array[ctypes.c_ulong]: ...

# 将 NDArray[ulonglong] 类型对象转换为 ctypes.Array[ctypes.c_ulonglong] 类型
@overload
def as_ctypes(obj: NDArray[ulonglong]) -> ctypes.Array[ctypes.c_ulonglong]: ...

# 将 NDArray[single] 类型对象转换为 ctypes.Array[ctypes.c_float] 类型
@overload
def as_ctypes(obj: NDArray[single]) -> ctypes.Array[ctypes.c_float]: ...

# 将 NDArray[double] 类型对象转换为 ctypes.Array[ctypes.c_double] 类型
@overload
def as_ctypes(obj: NDArray[double]) -> ctypes.Array[ctypes.c_double]: ...

# 将 NDArray[longdouble] 类型对象转换为 ctypes.Array[ctypes.c_longdouble] 类型
@overload
def as_ctypes(obj: NDArray[longdouble]) -> ctypes.Array[ctypes.c_longdouble]: ...

# 将 NDArray[void] 类型对象转换为 ctypes.Array[Any] 类型，可能是 `ctypes.Union` 或 `ctypes.Structure`
@overload
def as_ctypes(obj: NDArray[void]) -> ctypes.Array[Any]: ...
```