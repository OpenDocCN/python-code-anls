# `D:\src\scipysrc\pandas\pandas\core\util\numba_.py`

```
# 为 Numba 操作提供的通用工具函数模块

from __future__ import annotations
# 使用未来版本的语法特性，支持类型注解中的字符串类型

import inspect
import types
from typing import TYPE_CHECKING

import numpy as np

# 导入可选依赖项的函数
from pandas.compat._optional import import_optional_dependency
# 导入 NumbaUtilError 异常类
from pandas.errors import NumbaUtilError

if TYPE_CHECKING:
    from collections.abc import Callable

# 全局变量，指示是否使用 Numba
GLOBAL_USE_NUMBA: bool = False


def maybe_use_numba(engine: str | None) -> bool:
    """判断是否使用 Numba 的相关例程。"""
    return engine == "numba" or (engine is None and GLOBAL_USE_NUMBA)


def set_use_numba(enable: bool = False) -> None:
    """设置是否使用 Numba。"""
    global GLOBAL_USE_NUMBA
    if enable:
        import_optional_dependency("numba")
    GLOBAL_USE_NUMBA = enable


def get_jit_arguments(
    engine_kwargs: dict[str, bool] | None = None, kwargs: dict | None = None
) -> dict[str, bool]:
    """
    返回传递给 numba.JIT 的参数，如果没有则使用 pandas 的默认 JIT 设置。

    Parameters
    ----------
    engine_kwargs : dict, default None
        用户传递给 numba.JIT 的关键字参数
    kwargs : dict, default None
        用户传递给 JIT 函数的关键字参数

    Returns
    -------
    dict[str, bool]
        nopython, nogil, parallel

    Raises
    ------
    NumbaUtilError
        如果 numba 不支持关键字参数
    """
    if engine_kwargs is None:
        engine_kwargs = {}

    # 获取 nopython 参数，默认为 True
    nopython = engine_kwargs.get("nopython", True)
    if kwargs:
        # 注意：如果未来版本的 numba 支持关键字参数，应删除此检查。但这似乎不太可能很快发生。
        raise NumbaUtilError(
            "numba does not support keyword-only arguments"
            "https://github.com/numba/numba/issues/2916, "
            "https://github.com/numba/numba/issues/6846"
        )
    # 获取 nogil 参数，默认为 False
    nogil = engine_kwargs.get("nogil", False)
    # 获取 parallel 参数，默认为 False
    parallel = engine_kwargs.get("parallel", False)
    return {"nopython": nopython, "nogil": nogil, "parallel": parallel}


def jit_user_function(func: Callable) -> Callable:
    """
    如果用户函数尚未被 JIT 编译，将用户函数标记为可 JIT 的。

    Parameters
    ----------
    func : function
        用户定义的函数

    Returns
    -------
    function
        经过 Numba JIT 编译的函数，或者被标记为 JIT 可用的函数
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    if numba.extending.is_jitted(func):
        # 如果用户传递的函数已经被 JIT 编译过，则直接使用该函数
        numba_func = func
    elif getattr(np, func.__name__, False) is func or isinstance(
        func, types.BuiltinFunctionType
    ):
        # 不需要对内置函数或 np 函数进行 JIT 编译
        # 这会影响 register_jitable 函数
        numba_func = func
    else:
        # 否则，将函数注册为可 JIT 的函数
        numba_func = numba.extending.register_jitable(func)

    return numba_func


_sentinel = object()


def prepare_function_arguments(
    func: Callable, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    """
    准备函数调用所需的参数。
    # 如果没有关键字参数 kwargs，则直接返回位置参数 args 和空的 kwargs
    if not kwargs:
        return args, kwargs

    # 获取用户定义函数 func 的参数签名
    signature = inspect.signature(func)
    # 使用用户传入的 args 和 kwargs 绑定函数签名中的参数
    arguments = signature.bind(_sentinel, *args, **kwargs)
    # 应用函数参数的默认值（如果定义了）
    arguments.apply_defaults()
    # 根据 PEP 362，*args 或 **kwargs 中的参数将只包含在 BoundArguments.args 属性中
    args = arguments.args
    kwargs = arguments.kwargs

    # 断言第一个参数必须是 _sentinel（特殊标记），移除它
    assert args[0] is _sentinel
    args = args[1:]

    # 返回处理后的位置参数 args 和关键字参数 kwargs
    return args, kwargs
```