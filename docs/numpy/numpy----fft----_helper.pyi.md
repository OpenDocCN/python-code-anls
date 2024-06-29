# `D:\src\scipysrc\numpy\numpy\fft\_helper.pyi`

```py
# 导入必要的类型和函数
from typing import Any, TypeVar, overload, Literal as L

# 从 numpy 库中导入特定类型
from numpy import generic, integer, floating, complexfloating

# 导入 numpy 中定义的类型别名
from numpy._typing import (
    NDArray,
    ArrayLike,
    _ShapeLike,
    _ArrayLike,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
)

# 定义一个类型变量 _SCT，它是 generic 类型的子类型
_SCT = TypeVar("_SCT", bound=generic)

# 定义模块公开的函数和类型列表
__all__: list[str]

# fftshift 函数的类型重载，用于将数组按指定轴移位后进行 FFT
@overload
def fftshift(x: _ArrayLike[_SCT], axes: None | _ShapeLike = ...) -> NDArray[_SCT]: ...

# fftshift 函数的类型重载，用于将类数组按指定轴移位后进行 FFT
@overload
def fftshift(x: ArrayLike, axes: None | _ShapeLike = ...) -> NDArray[Any]: ...

# ifftshift 函数的类型重载，用于将数组按指定轴逆移位后进行逆 FFT
@overload
def ifftshift(x: _ArrayLike[_SCT], axes: None | _ShapeLike = ...) -> NDArray[_SCT]: ...

# ifftshift 函数的类型重载，用于将类数组按指定轴逆移位后进行逆 FFT
@overload
def ifftshift(x: ArrayLike, axes: None | _ShapeLike = ...) -> NDArray[Any]: ...

# fftfreq 函数的类型重载，生成 FFT 中使用的频率数组
@overload
def fftfreq(
    n: int | integer[Any],
    d: _ArrayLikeFloat_co = ...,
    device: None | L["cpu"] = ...,
) -> NDArray[floating[Any]]: ...

# fftfreq 函数的类型重载，生成 FFT 中使用的复数频率数组
@overload
def fftfreq(
    n: int | integer[Any],
    d: _ArrayLikeComplex_co = ...,
    device: None | L["cpu"] = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# rfftfreq 函数的类型重载，生成实部 FFT 中使用的频率数组
@overload
def rfftfreq(
    n: int | integer[Any],
    d: _ArrayLikeFloat_co = ...,
    device: None | L["cpu"] = ...,
) -> NDArray[floating[Any]]: ...

# rfftfreq 函数的类型重载，生成实部 FFT 中使用的复数频率数组
@overload
def rfftfreq(
    n: int | integer[Any],
    d: _ArrayLikeComplex_co = ...,
    device: None | L["cpu"] = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
```