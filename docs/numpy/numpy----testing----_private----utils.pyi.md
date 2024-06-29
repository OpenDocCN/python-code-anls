# `D:\src\scipysrc\numpy\numpy\testing\_private\utils.pyi`

```py
import os
import sys
import ast
import types
import warnings
import unittest
import contextlib
from re import Pattern
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Literal as L,
    Any,
    AnyStr,
    ClassVar,
    NoReturn,
    overload,
    type_check_only,
    TypeVar,
    Final,
    SupportsIndex,
)
if sys.version_info >= (3, 10):
    from typing import ParamSpec  # 导入 ParamSpec 类型，用于 Python 3.10 及以上版本
else:
    from typing_extensions import ParamSpec  # 兼容导入 ParamSpec 类型，用于 Python 3.9 及以下版本

import numpy as np
from numpy import number, object_, _FloatValue
from numpy._typing import (
    NDArray,
    ArrayLike,
    DTypeLike,
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
)

from unittest.case import (
    SkipTest as SkipTest,  # 导入 SkipTest 类并起别名为 SkipTest
)

_P = ParamSpec("_P")  # 定义 ParamSpec 的类型变量 _P
_T = TypeVar("_T")  # 定义类型变量 _T
_ET = TypeVar("_ET", bound=BaseException)  # 定义绑定到 BaseException 类的类型变量 _ET
_FT = TypeVar("_FT", bound=Callable[..., Any])  # 定义绑定到 Callable 类型的类型变量 _FT

# Must return a bool or an ndarray/generic type
# that is supported by `np.logical_and.reduce`
_ComparisonFunc = Callable[
    [NDArray[Any], NDArray[Any]],
    (
        bool
        | np.bool
        | number[Any]
        | NDArray[np.bool | number[Any] | object_]
    )
]  # 定义 _ComparisonFunc 类型，表示可调用对象，接受两个 NDArray 参数并返回特定类型

__all__: list[str]  # 定义 __all__ 变量为字符串列表

class KnownFailureException(Exception): ...  # 定义 KnownFailureException 类，继承自 Exception 类
class IgnoreException(Exception): ...  # 定义 IgnoreException 类，继承自 Exception 类

class clear_and_catch_warnings(warnings.catch_warnings):
    class_modules: ClassVar[tuple[types.ModuleType, ...]]  # 类型变量 class_modules，类范围为 types.ModuleType 的元组
    modules: set[types.ModuleType]  # 类型变量 modules，集合类型，元素为 types.ModuleType
    @overload
    def __new__(
        cls,
        record: L[False] = ...,
        modules: Iterable[types.ModuleType] = ...,
    ) -> _clear_and_catch_warnings_without_records: ...  # __new__ 方法的重载，返回 _clear_and_catch_warnings_without_records 类型
    @overload
    def __new__(
        cls,
        record: L[True],
        modules: Iterable[types.ModuleType] = ...,
    ) -> _clear_and_catch_warnings_with_records: ...  # __new__ 方法的重载，返回 _clear_and_catch_warnings_with_records 类型
    @overload
    def __new__(
        cls,
        record: bool,
        modules: Iterable[types.ModuleType] = ...,
    ) -> clear_and_catch_warnings: ...  # __new__ 方法的重载，返回 clear_and_catch_warnings 类型
    def __enter__(self) -> None | list[warnings.WarningMessage]: ...  # __enter__ 方法，返回 None 或 warnings.WarningMessage 列表
    def __exit__(
        self,
        __exc_type: None | type[BaseException] = ...,
        __exc_val: None | BaseException = ...,
        __exc_tb: None | types.TracebackType = ...,
    ) -> None: ...  # __exit__ 方法，无返回值

# Type-check only `clear_and_catch_warnings` subclasses for both values of the
# `record` parameter. Copied from the stdlib `warnings` stubs.

@type_check_only
class _clear_and_catch_warnings_with_records(clear_and_catch_warnings):
    def __enter__(self) -> list[warnings.WarningMessage]: ...  # __enter__ 方法，返回 warnings.WarningMessage 列表

@type_check_only
class _clear_and_catch_warnings_without_records(clear_and_catch_warnings):
    def __enter__(self) -> None: ...  # __enter__ 方法，无返回值

class suppress_warnings:
    log: list[warnings.WarningMessage]  # 类型变量 log，列表类型，元素为 warnings.WarningMessage
    def __init__(
        self,
        forwarding_rule: L["always", "module", "once", "location"] = ...,
    ) -> None: ...  # 初始化方法 __init__，参数 forwarding_rule 可选值为 "always", "module", "once", "location"
    def filter(
        self,
        category: type[Warning] = ...,
        message: str = ...,
        module: None | types.ModuleType = ...,
    ) -> None: ...  # filter 方法，用于过滤警告信息，无返回值
    # 定义一个方法record，用于记录警告信息，返回一个警告消息列表
    def record(
        self,
        category: type[Warning] = ...,
        message: str = ...,
        module: None | types.ModuleType = ...,
    ) -> list[warnings.WarningMessage]: ...
    
    # 实现上下文管理器方法__enter__，返回自身实例_T
    def __enter__(self: _T) -> _T: ...
    
    # 实现上下文管理器方法__exit__，接收异常信息，无返回值
    def __exit__(
        self,
        __exc_type: None | type[BaseException] = ...,
        __exc_val: None | BaseException = ...,
        __exc_tb: None | types.TracebackType = ...,
    ) -> None: ...
    
    # 定义一个方法__call__，接收一个函数func作为参数，返回函数_func
    def __call__(self, func: _FT) -> _FT: ...
verbose: int
IS_PYPY: Final[bool]
IS_PYSTON: Final[bool]
HAS_REFCOUNT: Final[bool]
HAS_LAPACK64: Final[bool]

# 定义一个名为 assert_ 的函数，用于执行断言检查，没有具体实现
def assert_(val: object, msg: str | Callable[[], str] = ...) -> None: ...

# 根据操作系统平台不同定义不同的 memusage 函数，用于获取内存使用情况
if sys.platform == "win32" or sys.platform == "cygwin":
    def memusage(processName: str = ..., instance: int = ...) -> int: ...
elif sys.platform == "linux":
    def memusage(_proc_pid_stat: str | bytes | os.PathLike[Any] = ...) -> None | int: ...
else:
    def memusage() -> NoReturn: ...

# 根据操作系统平台不同定义不同的 jiffies 函数，用于获取处理器时钟周期数
if sys.platform == "linux":
    def jiffies(
        _proc_pid_stat: str | bytes | os.PathLike[Any] = ...,
        _load_time: list[float] = ...,
    ) -> int: ...
else:
    def jiffies(_load_time: list[float] = ...) -> int: ...

# 构建错误信息字符串，用于测试失败时的错误消息
def build_err_msg(
    arrays: Iterable[object],
    err_msg: str,
    header: str = ...,
    verbose: bool = ...,
    names: Sequence[str] = ...,
    precision: None | SupportsIndex = ...,
) -> str: ...

# 断言两个对象相等，用于单元测试中判断预期结果与实际结果是否一致
def assert_equal(
    actual: object,
    desired: object,
    err_msg: object = ...,
    verbose: bool = ...,
    *,
    strict: bool = ...
) -> None: ...

# 打印测试用例的实际结果和期望结果
def print_assert_equal(
    test_string: str,
    actual: object,
    desired: object,
) -> None: ...

# 断言两个数值或对象在给定精度内近似相等
def assert_almost_equal(
    actual: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    desired: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    decimal: int = ...,
    err_msg: object = ...,
    verbose: bool = ...,
) -> None: ...

# 断言两个浮点数在给定的有效数字内近似相等
def assert_approx_equal(
    actual: _FloatValue,
    desired: _FloatValue,
    significant: int = ...,
    err_msg: object = ...,
    verbose: bool = ...,
) -> None: ...

# 断言两个数组或对象在给定条件下进行比较
def assert_array_compare(
    comparison: _ComparisonFunc,
    x: ArrayLike,
    y: ArrayLike,
    err_msg: object = ...,
    verbose: bool = ...,
    header: str = ...,
    precision: SupportsIndex = ...,
    equal_nan: bool = ...,
    equal_inf: bool = ...,
    *,
    strict: bool = ...
) -> None: ...

# 断言两个数组或对象完全相等
def assert_array_equal(
    x: ArrayLike,
    y: ArrayLike,
    /,
    err_msg: object = ...,
    verbose: bool = ...,
    *,
    strict: bool = ...
) -> None: ...

# 断言两个数组或对象在给定精度下近似相等
def assert_array_almost_equal(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    /,
    decimal: float = ...,
    err_msg: object = ...,
    verbose: bool = ...,
) -> None: ...

# 对 assert_array_less 函数进行重载，用于比较不同类型的数组或对象
@overload
def assert_array_less(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    err_msg: object = ...,
    verbose: bool = ...,
    *,
    strict: bool = ...
) -> None: ...
@overload
def assert_array_less(
    x: _ArrayLikeTD64_co,
    y: _ArrayLikeTD64_co,
    err_msg: object = ...,
    verbose: bool = ...,
    *,
    strict: bool = ...
) -> None: ...
@overload
def assert_array_less(
    x: _ArrayLikeDT64_co,
    y: _ArrayLikeDT64_co,
    err_msg: object = ...,
    verbose: bool = ...,
    *,
    # 声明一个变量 strict，类型为 bool，但未指定初始值
    strict: bool = ...
# 定义一个空函数，不返回任何内容
) -> None: ...

# 运行给定的代码字符串、字节串或者代码对象，可以在指定的字典环境中执行
def runstring(
    astr: str | bytes | types.CodeType,
    dict: None | dict[str, Any],
) -> Any: ...

# 断言两个字符串相等，如果不相等则抛出 AssertionError
def assert_string_equal(actual: str, desired: str) -> None: ...

# 运行文档测试，可以指定文件名，如果出错可以选择是否抛出异常
def rundocs(
    filename: None | str | os.PathLike[str] = ...,
    raise_on_error: bool = ...,
) -> None: ...

# 返回一个装饰器，用于测试中的方法，可以指定要匹配的测试名称模式
def decorate_methods(
    cls: type[Any],
    decorator: Callable[[Callable[..., Any]], Any],
    testmatch: None | str | bytes | Pattern[Any] = ...,
) -> None: ...

# 测量给定代码的执行时间，可以指定执行次数和标签
def measure(
    code_str: str | bytes | ast.mod | ast.AST,
    times: int = ...,
    label: None | str = ...,
) -> float: ...

# 断言两个数组或类数组接近（元素值相近），可设置容忍的相对和绝对误差
def assert_allclose(
    actual: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    desired: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
    err_msg: object = ...,
    verbose: bool = ...,
    *,
    strict: bool = ...
) -> None: ...

# 断言两个数组的最大单位误差小于给定值，可以指定数据类型
def assert_array_max_ulp(
    a: _ArrayLikeNumber_co,
    b: _ArrayLikeNumber_co,
    maxulp: float = ...,
    dtype: DTypeLike = ...,
) -> NDArray[Any]: ...

# 断言两个数组的近似零误差单位小数点
def assert_array_almost_equal_nulp(
    x: _ArrayLikeNumber_co,
    y: _ArrayLikeNumber_co,
    nulp: float = ...,
) -> None: ...

# 断言在调用函数时会产生指定类型的警告
def assert_warns(
    warning_class: type[Warning],
    func: Callable[_P, _T],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T: ...

# 断言在调用函数时不会产生任何警告
def assert_no_warnings(
    func: Callable[_P, _T],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T: ...

# 返回一个装饰器，用于测试中的方法，可以指定要匹配的测试名称模式
def raises(*args: type[BaseException]) -> Callable[[_FT], _FT]: ...

# 断言在调用函数时会产生指定类型和正则表达式匹配的异常
def assert_raises_regex(
    expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    expected_regex: str | bytes | Pattern[Any],
    callable: Callable[_P, Any],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None: ...

# 断言在调用函数时会产生指定类型和正则表达式匹配的异常
def assert_raises(
    expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    callable: Callable[_P, Any],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None: ...
# 返回一个上下文管理器，用于创建一个临时目录路径字符串
@overload
def tempdir(
    suffix: None | AnyStr = ...,
    prefix: None | AnyStr = ...,
    dir: None | AnyStr | os.PathLike[AnyStr] = ...,
) -> contextlib._GeneratorContextManager[AnyStr]: ...

# 返回一个上下文管理器，用于创建一个临时文件路径字符串
@overload
def temppath(
    suffix: None = ...,
    prefix: None = ...,
    dir: None = ...,
    text: bool = ...,
) -> contextlib._GeneratorContextManager[str]: ...

# 返回一个上下文管理器，用于创建一个临时文件路径字符串
@overload
def temppath(
    suffix: None | AnyStr = ...,
    prefix: None | AnyStr = ...,
    dir: None | AnyStr | os.PathLike[AnyStr] = ...,
    text: bool = ...,
) -> contextlib._GeneratorContextManager[AnyStr]: ...

# 返回一个上下文管理器，用于确保在函数执行期间没有垃圾回收循环
@overload
def assert_no_gc_cycles() -> contextlib._GeneratorContextManager[None]: ...

# 在函数执行期间确保没有垃圾回收循环，接受一个函数和其参数
@overload
def assert_no_gc_cycles(
    func: Callable[_P, Any],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None: ...

# 中断并尝试清除所有可能的垃圾回收循环
def break_cycles() -> None: ...
```