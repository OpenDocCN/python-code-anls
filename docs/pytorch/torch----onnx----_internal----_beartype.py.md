# `.\pytorch\torch\onnx\_internal\_beartype.py`

```py
# mypy: allow-untyped-defs
"""An internal wrapper for the beartype library.

The module returns a no-op decorator when the beartype library is not installed.
"""
import enum  # 导入枚举类型模块
import functools  # 导入 functools 模块
import os  # 导入操作系统相关功能的模块
import traceback  # 导入追踪异常信息的模块
import typing  # 导入类型提示相关的模块
import warnings  # 导入警告模块
from types import ModuleType  # 从 types 模块导入 ModuleType 类型

try:
    import beartype as _beartype_lib  # type: ignore[import]
    from beartype import roar as _roar  # type: ignore[import]

    # Beartype warns when we import from typing because the types are deprecated
    # in Python 3.9. But there will be a long time until we can move to using
    # the native container types for type annotations (when 3.9 is the lowest
    # supported version). So we silence the warning.
    warnings.filterwarnings(
        "ignore",
        category=_roar.BeartypeDecorHintPep585DeprecationWarning,
    )

    if _beartype_lib.__version__ == "0.16.0":
        # beartype 0.16.0 has a bug that causes it to crash when used with
        # PyTorch. See https://github.com/beartype/beartype/issues/282
        warnings.warn("beartype 0.16.0 is not supported. Please upgrade to 0.16.1+.")
        _beartype_lib = None  # type: ignore[assignment]
except ImportError:
    _beartype_lib = None  # type: ignore[assignment]
except Exception as e:
    # Warn errors that are not import errors (unexpected).
    warnings.warn(f"{e}")
    _beartype_lib = None  # type: ignore[assignment]


@enum.unique
class RuntimeTypeCheckState(enum.Enum):
    """Runtime type check state."""

    # Runtime type checking is disabled.
    DISABLED = enum.auto()
    # Runtime type checking is enabled but warnings are shown only.
    WARNINGS = enum.auto()
    # Runtime type checking is enabled.
    ERRORS = enum.auto()


class CallHintViolationWarning(UserWarning):
    """Warning raised when a type hint is violated during a function call."""

    pass


def _no_op_decorator(func):
    return func  # 返回传入的函数，不进行任何装饰


def _create_beartype_decorator(
    runtime_check_state: RuntimeTypeCheckState,
):
    # beartype needs to be imported outside of the function and aliased because
    # this module overwrites the name "beartype".

    if runtime_check_state == RuntimeTypeCheckState.DISABLED:
        return _no_op_decorator  # 如果运行时类型检查被禁用，则返回一个不做任何操作的装饰器
    if _beartype_lib is None:
        # If the beartype library is not installed, return a no-op decorator
        return _no_op_decorator  # 如果未安装 beartype 库，则返回一个不做任何操作的装饰器

    assert isinstance(_beartype_lib, ModuleType)

    if runtime_check_state == RuntimeTypeCheckState.ERRORS:
        # Enable runtime type checking which errors on any type hint violation.
        return _beartype_lib.beartype  # 如果运行时类型检查设置为 ERRORS，则返回 beartype 库的装饰器函数

    # Warnings only
    def beartype(func):
        """Decorator that applies type checking with beartype."""
    
        # Check if there's a 'return' type annotation in the function
        if "return" in func.__annotations__:
            # Store the original return type
            return_type = func.__annotations__["return"]
            # Remove the 'return' type annotation temporarily
            del func.__annotations__["return"]
            # Apply beartype decorator to the function without 'return' type annotation
            beartyped = _beartype_lib.beartype(func)
            # Restore the 'return' type annotation
            func.__annotations__["return"] = return_type
        else:
            # Apply beartype decorator directly if no 'return' type annotation exists
            beartyped = _beartype_lib.beartype(func)
    
        # Wrapper function that handles exceptions and warnings
        @functools.wraps(func)
        def _coerce_beartype_exceptions_to_warnings(*args, **kwargs):
            try:
                # Attempt to call the decorated function
                return beartyped(*args, **kwargs)
            except _roar.BeartypeCallHintParamViolation:
                # Catch specific exception and issue a warning
                warnings.warn(
                    traceback.format_exc(),
                    category=CallHintViolationWarning,
                    stacklevel=2,
                )
    
            # Return the result of the original function call
            return func(*args, **kwargs)  # noqa: B012
    
        # Return the wrapped function that handles type checking and warnings
        return _coerce_beartype_exceptions_to_warnings
    
    # Return the beartype decorator function for use
    return beartype
if typing.TYPE_CHECKING:
    # 如果正在进行类型检查
    # 这是一种方法，用来让 mypy 与 beartype 装饰器协同工作。
    def beartype(func):
        return func

else:
    # 如果不是在进行类型检查
    # 获取环境变量 TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK 的值
    _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK = os.getenv(
        "TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK"
    )
    # 根据环境变量的不同设置运行时类型检查的状态
    if _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK == "ERRORS":
        _runtime_type_check_state = RuntimeTypeCheckState.ERRORS
    elif _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK == "DISABLED":
        _runtime_type_check_state = RuntimeTypeCheckState.DISABLED
    else:
        _runtime_type_check_state = RuntimeTypeCheckState.WARNINGS
    # 使用确定的运行时类型检查状态创建 beartype 装饰器
    beartype = _create_beartype_decorator(_runtime_type_check_state)
    # 确保无论哪个路径被选择，beartype 装饰器都已启用
    assert beartype is not None
```