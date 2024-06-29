# `.\numpy\numpy\_typing\_scalars.py`

```py
# 引入必要的类型声明模块，Union 用于指定多个类型中的一个作为类型注解
from typing import Union, Any

# 引入 NumPy 库，简称为 np
import numpy as np

# 定义 _CharLike_co 类型别名，表示可以是 str 或 bytes 类型的对象
_CharLike_co = Union[str, bytes]

# 下面的六个 `<X>Like_co` 类型别名分别表示可以被强制转换为对应类型 `<X>` 的所有标量类型
# 使用 `same_kind` 规则进行强制转换
_BoolLike_co = Union[bool, np.bool]         # 可以是 bool 或者 np.bool 类型的对象
_UIntLike_co = Union[_BoolLike_co, np.unsignedinteger[Any]]  # 可以是 _BoolLike_co 或 np.unsignedinteger[Any] 类型的对象
_IntLike_co = Union[_BoolLike_co, int, np.integer[Any]]  # 可以是 _BoolLike_co, int 或 np.integer[Any] 类型的对象
_FloatLike_co = Union[_IntLike_co, float, np.floating[Any]]  # 可以是 _IntLike_co, float 或 np.floating[Any] 类型的对象
_ComplexLike_co = Union[_FloatLike_co, complex, np.complexfloating[Any, Any]]  # 可以是 _FloatLike_co, complex 或 np.complexfloating[Any, Any] 类型的对象
_TD64Like_co = Union[_IntLike_co, np.timedelta64]  # 可以是 _IntLike_co 或 np.timedelta64 类型的对象

# 定义 _NumberLike_co 类型别名，表示可以是 int, float, complex, np.number[Any], np.bool 中的一个
_NumberLike_co = Union[int, float, complex, np.number[Any], np.bool]

# 定义 _ScalarLike_co 类型别名，表示可以是 int, float, complex, str, bytes, np.generic 中的一个
_ScalarLike_co = Union[
    int,
    float,
    complex,
    str,
    bytes,
    np.generic,
]

# 定义 _VoidLike_co 类型别名，表示可以是 tuple[Any, ...], np.void 中的一个
# 虽然 _VoidLike_co 不是标量类型，但其用途与标量类型相似
_VoidLike_co = Union[tuple[Any, ...], np.void]
```