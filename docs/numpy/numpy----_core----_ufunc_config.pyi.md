# `D:\src\scipysrc\numpy\numpy\_core\_ufunc_config.pyi`

```
from collections.abc import Callable
from typing import Any, Literal, TypedDict
from numpy import _SupportsWrite

# 定义错误类型的文字字面量类型
_ErrKind = Literal["ignore", "warn", "raise", "call", "print", "log"]

# 定义错误设置字典类型
class _ErrDict(TypedDict):
    divide: _ErrKind    # 错误种类：除法错误
    over: _ErrKind      # 错误种类：溢出错误
    under: _ErrKind     # 错误种类：下溢错误
    invalid: _ErrKind   # 错误种类：无效输入错误

# 定义可选的错误设置字典类型，允许所有错误或特定错误设置为 None
class _ErrDictOptional(TypedDict, total=False):
    all: None | _ErrKind
    divide: None | _ErrKind
    over: None | _ErrKind
    under: None | _ErrKind
    invalid: None | _ErrKind

# 设置错误处理函数的签名及返回类型，返回一个错误设置字典
def seterr(
    all: None | _ErrKind = ...,
    divide: None | _ErrKind = ...,
    over: None | _ErrKind = ...,
    under: None | _ErrKind = ...,
    invalid: None | _ErrKind = ...,
) -> _ErrDict: ...

# 获取当前的错误设置，返回一个错误设置字典
def geterr() -> _ErrDict: ...

# 设置缓冲区大小，返回设置后的大小
def setbufsize(size: int) -> int: ...

# 获取当前缓冲区大小，返回当前设置的大小
def getbufsize() -> int: ...

# 设置错误回调函数的签名及返回类型，可以是 None、函数或支持写操作的对象
def seterrcall(
    func: None | _ErrFunc | _SupportsWrite[str]
) -> None | _ErrFunc | _SupportsWrite[str]: ...

# 获取当前错误回调函数，返回值可以是 None、函数或支持写操作的对象
def geterrcall() -> None | _ErrFunc | _SupportsWrite[str]: ...

# 查阅 `numpy/__init__.pyi` 中的 `errstate` 类和 `no_nep5_warnings` 的相关说明
```