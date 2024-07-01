# `.\numpy\numpy\lib\_type_check_impl.pyi`

```py
# 从 collections.abc 导入 Container 和 Iterable 接口
from collections.abc import Container, Iterable
# 从 typing 模块导入 Literal 别名 L，Any 类型，overload 装饰器，TypeVar 类型变量和 Protocol 协议
from typing import (
    Literal as L,
    Any,
    overload,
    TypeVar,
    Protocol,
)

# 导入 numpy 库，并将其别名为 np
import numpy as np
# 从 numpy 中导入 dtype、generic、floating、float64、complexfloating 和 integer
from numpy import (
    dtype,
    generic,
    floating,
    float64,
    complexfloating,
    integer,
)

# 从 numpy._typing 模块导入各种类型变量
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,
    _64Bit,
    _SupportsDType,
    _ScalarLike_co,
    _ArrayLike,
    _DTypeLikeComplex,
)

# 定义类型变量 _T 和 _T_co
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
# 定义类型变量 _SCT，限定为 generic 的子类
_SCT = TypeVar("_SCT", bound=generic)
# 定义类型变量 _NBit1 和 _NBit2，限定为 NBitBase 的子类

# 定义 _SupportsReal 协议，泛型为 _T_co
class _SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co: ...

# 定义 _SupportsImag 协议，泛型为 _T_co
class _SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co: ...

# 定义 __all__ 列表，用于模块导出

# 定义函数 mintypecode，参数为 typechars、typeset 和 default，返回值为 str 类型
def mintypecode(
    typechars: Iterable[str | ArrayLike],
    typeset: Container[str] = ...,
    default: str = ...,
) -> str: ...

# 函数 real 的重载定义，参数为 _SupportsReal 或 ArrayLike 类型，返回值为 _T 或 NDArray[Any]
@overload
def real(val: _SupportsReal[_T]) -> _T: ...
@overload
def real(val: ArrayLike) -> NDArray[Any]: ...

# 函数 imag 的重载定义，参数为 _SupportsImag 或 ArrayLike 类型，返回值为 _T 或 NDArray[Any]
@overload
def imag(val: _SupportsImag[_T]) -> _T: ...
@overload
def imag(val: ArrayLike) -> NDArray[Any]: ...

# 函数 iscomplex 的重载定义，参数为 _ScalarLike_co 或 ArrayLike 类型，返回值为 np.bool 类型
@overload
def iscomplex(x: _ScalarLike_co) -> np.bool: ...  # type: ignore[misc]
@overload
def iscomplex(x: ArrayLike) -> NDArray[np.bool]: ...

# 函数 isreal 的重载定义，参数为 _ScalarLike_co 或 ArrayLike 类型，返回值为 np.bool 类型
@overload
def isreal(x: _ScalarLike_co) -> np.bool: ...  # type: ignore[misc]
@overload
def isreal(x: ArrayLike) -> NDArray[np.bool]: ...

# 函数 iscomplexobj，参数为 _SupportsDType[dtype[Any]] 或 ArrayLike 类型，返回值为 bool 类型
def iscomplexobj(x: _SupportsDType[dtype[Any]] | ArrayLike) -> bool: ...

# 函数 isrealobj，参数为 _SupportsDType[dtype[Any]] 或 ArrayLike 类型，返回值为 bool 类型
def isrealobj(x: _SupportsDType[dtype[Any]] | ArrayLike) -> bool: ...

# 函数 nan_to_num 的重载定义，参数为不同类型的 x，返回值类型根据参数类型不同而定
@overload
def nan_to_num(  # type: ignore[misc]
    x: _SCT,
    copy: bool = ...,
    nan: float = ...,
    posinf: None | float = ...,
    neginf: None | float = ...,
) -> _SCT: ...
@overload
def nan_to_num(
    x: _ScalarLike_co,
    copy: bool = ...,
    nan: float = ...,
    posinf: None | float = ...,
    neginf: None | float = ...,
) -> Any: ...
@overload
def nan_to_num(
    x: _ArrayLike[_SCT],
    copy: bool = ...,
    nan: float = ...,
    posinf: None | float = ...,
    neginf: None | float = ...,
) -> NDArray[_SCT]: ...
@overload
def nan_to_num(
    x: ArrayLike,
    copy: bool = ...,
    nan: float = ...,
    posinf: None | float = ...,
    neginf: None | float = ...,
) -> NDArray[Any]: ...

# 函数 real_if_close 的重载定义，参数为不同类型的 a 和 tol，返回值类型根据参数类型不同而定
@overload
def real_if_close(  # type: ignore[misc]
    a: _ArrayLike[complexfloating[_NBit1, _NBit1]],
    tol: float = ...,
) -> NDArray[floating[_NBit1]] | NDArray[complexfloating[_NBit1, _NBit1]]: ...
@overload
def real_if_close(
    a: _ArrayLike[_SCT],
    tol: float = ...,
) -> NDArray[_SCT]: ...
@overload
def real_if_close(
    a: ArrayLike,
    tol: float = ...,
) -> NDArray[Any]: ...

# 函数 typename 的重载定义，参数为字符类型的 char，返回值类型根据参数类型不同而定
@overload
def typename(char: L['S1']) -> L['character']: ...
@overload
def typename(char: L['?']) -> L['bool']: ...
@overload
def typename(char: L['b']) -> L['signed char']: ...
# 函数重载定义，接受字符类型 'b'，返回类型为 'signed char'

@overload
def typename(char: L['B']) -> L['unsigned char']: ...
# 函数重载定义，接受字符类型 'B'，返回类型为 'unsigned char'

@overload
def typename(char: L['h']) -> L['short']: ...
# 函数重载定义，接受字符类型 'h'，返回类型为 'short'

@overload
def typename(char: L['H']) -> L['unsigned short']: ...
# 函数重载定义，接受字符类型 'H'，返回类型为 'unsigned short'

@overload
def typename(char: L['i']) -> L['integer']: ...
# 函数重载定义，接受字符类型 'i'，返回类型为 'integer'

@overload
def typename(char: L['I']) -> L['unsigned integer']: ...
# 函数重载定义，接受字符类型 'I'，返回类型为 'unsigned integer'

@overload
def typename(char: L['l']) -> L['long integer']: ...
# 函数重载定义，接受字符类型 'l'，返回类型为 'long integer'

@overload
def typename(char: L['L']) -> L['unsigned long integer']: ...
# 函数重载定义，接受字符类型 'L'，返回类型为 'unsigned long integer'

@overload
def typename(char: L['q']) -> L['long long integer']: ...
# 函数重载定义，接受字符类型 'q'，返回类型为 'long long integer'

@overload
def typename(char: L['Q']) -> L['unsigned long long integer']: ...
# 函数重载定义，接受字符类型 'Q'，返回类型为 'unsigned long long integer'

@overload
def typename(char: L['f']) -> L['single precision']: ...
# 函数重载定义，接受字符类型 'f'，返回类型为 'single precision'

@overload
def typename(char: L['d']) -> L['double precision']: ...
# 函数重载定义，接受字符类型 'd'，返回类型为 'double precision'

@overload
def typename(char: L['g']) -> L['long precision']: ...
# 函数重载定义，接受字符类型 'g'，返回类型为 'long precision'

@overload
def typename(char: L['F']) -> L['complex single precision']: ...
# 函数重载定义，接受字符类型 'F'，返回类型为 'complex single precision'

@overload
def typename(char: L['D']) -> L['complex double precision']: ...
# 函数重载定义，接受字符类型 'D'，返回类型为 'complex double precision'

@overload
def typename(char: L['G']) -> L['complex long double precision']: ...
# 函数重载定义，接受字符类型 'G'，返回类型为 'complex long double precision'

@overload
def typename(char: L['S']) -> L['string']: ...
# 函数重载定义，接受字符类型 'S'，返回类型为 'string'

@overload
def typename(char: L['U']) -> L['unicode']: ...
# 函数重载定义，接受字符类型 'U'，返回类型为 'unicode'

@overload
def typename(char: L['V']) -> L['void']: ...
# 函数重载定义，接受字符类型 'V'，返回类型为 'void'

@overload
def typename(char: L['O']) -> L['object']: ...
# 函数重载定义，接受字符类型 'O'，返回类型为 'object'

@overload
def common_type(  # type: ignore[misc]
    *arrays: _SupportsDType[dtype[
        integer[Any]
    ]]
) -> type[floating[_64Bit]]: ...
# 函数重载定义，接受任意数量的整数类型数组，返回类型为 64 位浮点数的类型

@overload
def common_type(  # type: ignore[misc]
    *arrays: _SupportsDType[dtype[
        floating[_NBit1]
    ]]
) -> type[floating[_NBit1]]: ...
# 函数重载定义，接受任意数量的指定位数浮点数类型数组，返回类型为指定位数浮点数的类型

@overload
def common_type(  # type: ignore[misc]
    *arrays: _SupportsDType[dtype[
        integer[Any] | floating[_NBit1]
    ]]
) -> type[floating[_NBit1 | _64Bit]]: ...
# 函数重载定义，接受任意数量的整数或指定位数浮点数类型数组，返回类型为指定位数浮点数或64位浮点数的类型

@overload
def common_type(  # type: ignore[misc]
    *arrays: _SupportsDType[dtype[
        floating[_NBit1] | complexfloating[_NBit2, _NBit2]
    ]]
) -> type[complexfloating[_NBit1 | _NBit2, _NBit1 | _NBit2]]: ...
# 函数重载定义，接受任意数量的指定位数浮点数或复数类型数组，返回类型为复数类型的类型

@overload
def common_type(
    *arrays: _SupportsDType[dtype[
        integer[Any] | floating[_NBit1] | complexfloating[_NBit2, _NBit2]
    ]]
) -> type[complexfloating[_64Bit | _NBit1 | _NBit2, _64Bit | _NBit1 | _NBit2]]: ...
# 函数重载定义，接受任意数量的整数、指定位数浮点数或复数类型数组，返回类型为复数类型、指定位数浮点数或64位浮点数的类型
```