# `D:\src\scipysrc\matplotlib\lib\matplotlib\_api\deprecation.pyi`

```
# 导入模块 `Callable` 从 `collections.abc` 中
# 导入模块 `contextlib` 整个模块
# 从 `typing` 模块导入 `Any`, `TypedDict`, `TypeVar`, `overload`
# 从 `typing_extensions` 模块导入 `ParamSpec`, `Unpack`
from collections.abc import Callable
import contextlib
from typing import Any, TypedDict, TypeVar, overload
from typing_extensions import (
    ParamSpec,  # < Py 3.10
    Unpack,  # < Py 3.11
)

# 定义 `_P`, `_R` 作为类型变量
_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")

# 定义一个空类 `MatplotlibDeprecationWarning` 用于模拟 Matplotlib 的过时警告
class MatplotlibDeprecationWarning(DeprecationWarning): ...

# 定义 `DeprecationKwargs` 类型字典，包含部分可选键
class DeprecationKwargs(TypedDict, total=False):
    message: str
    alternative: str
    pending: bool
    obj_type: str
    addendum: str
    removal: str

# 定义 `NamedDeprecationKwargs` 类型字典，继承自 `DeprecationKwargs`，包含更多可选键
class NamedDeprecationKwargs(DeprecationKwargs, total=False):
    name: str

# 定义函数 `warn_deprecated`，给定自 Python 3.10 的参数解包语法
# 函数用于发出过时警告，返回空值
def warn_deprecated(since: str, **kwargs: Unpack[NamedDeprecationKwargs]) -> None: ...

# 定义装饰器函数 `deprecated`，给定自 Python 3.10 的参数解包语法
# 用于标记函数或方法已过时，返回相同类型的函数或方法
def deprecated(
    since: str, **kwargs: Unpack[NamedDeprecationKwargs]
) -> Callable[[_T], _T]: ...

# 定义类 `deprecate_privatize_attribute`，继承自 `Any` 类
# 用于将属性标记为过时且私有化
class deprecate_privatize_attribute(Any):
    # 初始化方法，接受自 Python 3.10 的参数解包语法
    def __init__(self, since: str, **kwargs: Unpack[NamedDeprecationKwargs]): ...
    # 设置属性名方法，返回空值
    def __set_name__(self, owner: type[object], name: str) -> None: ...

# 定义常量 `DECORATORS`，其类型为从 `Callable` 到 `Callable` 的字典
DECORATORS: dict[Callable, Callable] = ...

# 定义装饰器函数 `rename_parameter`，支持 Python 3.10 的参数解包语法
# 用于重命名函数或方法的参数，返回相同类型的函数或方法
@overload
def rename_parameter(
    since: str, old: str, new: str, func: None = ...
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
@overload
def rename_parameter(
    since: str, old: str, new: str, func: Callable[_P, _R]
) -> Callable[_P, _R]: ...

# 定义类 `_deprecated_parameter_class`，用于描述一个已过时的参数类
class _deprecated_parameter_class: ...

# 定义装饰器函数 `delete_parameter`，支持 Python 3.10 的参数解包语法
# 用于删除函数或方法的参数，返回相同类型的函数或方法
@overload
def delete_parameter(
    since: str, name: str, func: None = ..., **kwargs: Unpack[DeprecationKwargs]
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
@overload
def delete_parameter(
    since: str, name: str, func: Callable[_P, _R], **kwargs: Unpack[DeprecationKwargs]
) -> Callable[_P, _R]: ...

# 定义装饰器函数 `make_keyword_only`，支持 Python 3.10 的参数解包语法
# 用于将函数或方法的参数转换为仅限关键字参数，返回相同类型的函数或方法
@overload
def make_keyword_only(
    since: str, name: str, func: None = ...
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
@overload
def make_keyword_only(
    since: str, name: str, func: Callable[_P, _R]
) -> Callable[_P, _R]: ...

# 定义函数 `deprecate_method_override`，支持 Python 3.10 的参数解包语法
# 用于标记方法的重写为过时，返回相同类型的方法
def deprecate_method_override(
    method: Callable[_P, _R],
    obj: object | type,
    *,
    allow_empty: bool = ...,
    since: str,
    **kwargs: Unpack[NamedDeprecationKwargs]
) -> Callable[_P, _R]: ...

# 定义函数 `suppress_matplotlib_deprecation_warning`，返回一个上下文管理器
# 用于抑制 Matplotlib 的过时警告
def suppress_matplotlib_deprecation_warning() -> (
    contextlib.AbstractContextManager[None]
): ...
```