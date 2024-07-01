# `.\numpy\numpy\_core\numerictypes.pyi`

```py
```python`
# 导入必要的类型和函数

from typing import (
    Literal as L,  # 导入类型别名 L 作为 Literal
    Any,  # 导入 Any 类型
    TypeVar,  # 导入泛型类型变量 TypeVar
    TypedDict,  # 导入 TypedDict 类型
)

import numpy as np  # 导入 NumPy 库并使用 np 别名
from numpy import (  # 导入 NumPy 中的多个数据类型
    dtype,  # 导入 dtype 函数
    generic,  # 导入 generic 类型
    ubyte, ushort, uintc, ulong, ulonglong,  # 导入无符号整型别名
    byte, short, intc, long, longlong,  # 导入有符号整型别名
    half, single, double, longdouble,  # 导入浮点型别名
    csingle, cdouble, clongdouble,  # 导入复数浮点型别名
    datetime64, timedelta64,  # 导入日期时间和时间间隔类型
    object_, str_, bytes_, void,  # 导入对象、字符串、字节和空类型
)

from numpy._core._type_aliases import (  # 导入 NumPy 内部的类型别名
    sctypeDict as sctypeDict,  # 导入 sctypeDict 别名
)

from numpy._typing import DTypeLike  # 导入 NumPy 的类型别名 DTypeLike

_T = TypeVar("_T")  # 定义泛型类型变量 _T
_SCT = TypeVar("_SCT", bound=generic)  # 定义受限泛型类型变量 _SCT

class _TypeCodes(TypedDict):  # 定义 TypedDict 类型 _TypeCodes
    Character: L['c']  # 字符类型别名
    Integer: L['bhilqp']  # 整数类型别名
    UnsignedInteger: L['BHILQP']  # 无符号整数类型别名
    Float: L['efdg']  # 浮点数类型别名
    Complex: L['FDG']  # 复数类型别名
    AllInteger: L['bBhHiIlLqQpP']  # 所有整数类型别名
    AllFloat: L['efdgFDG']  # 所有浮点数类型别名
    Datetime: L['Mm']  # 日期时间类型别名
    All: L['?bhilqpBHILQPefdgFDGSUVOMm']  # 所有类型别名

__all__: list[str]  # 导出的所有符号列表，类型为 str

def isdtype(  # 定义函数 isdtype
    dtype: dtype[Any] | type[Any],  # 参数 dtype，可以是 dtype 或 type 类型
    kind: DTypeLike | tuple[DTypeLike, ...]  # 参数 kind，可以是 DTypeLike 或其元组
) -> bool:  # 返回布尔值

def issubdtype(arg1: DTypeLike, arg2: DTypeLike) -> bool:  # 定义函数 issubdtype，参数和返回类型都是布尔值

typecodes: _TypeCodes  # 定义类型别名 typecodes，类型为 _TypeCodes

ScalarType: tuple[  # 定义标量类型元组 ScalarType，包含多种数据类型
    type[int], type[float], type[complex],  # 整数、浮点数和复数类型
    type[bool], type[bytes], type[str],  # 布尔、字节和字符串类型
    type[memoryview], type[np.bool],  # 内存视图和 NumPy 布尔类型
    type[csingle], type[cdouble], type[clongdouble],  # 复数浮点数类型
    type[half], type[single], type[double], type[longdouble],  # 浮点数类型
    type[byte], type[short], type[intc], type[long],  # 整数类型
    type[longlong], type[timedelta64], type[datetime64],  # 时间间隔和日期时间类型
    type[object_], type[bytes_], type[str_],  # 对象、字节和字符串类型
    type[ubyte], type[ushort], type[uintc], type[ulong], type[ulonglong], type[void],  # 无符号整数类型和空类型
]
```