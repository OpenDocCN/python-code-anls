# `.\pytorch\torch\backends\opt_einsum\__init__.py`

```py
# mypy: allow-untyped-defs
# 导入系统相关的模块
import sys
# 导入警告模块
import warnings
# 导入上下文管理器模块
from contextlib import contextmanager
# 导入 functools 模块中的 lru_cache 函数，并起别名为 _lru_cache
from functools import lru_cache as _lru_cache
# 导入 Any 类型提示
from typing import Any

# 从 torch.backends 中导入三个标识符
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule

try:
    # 尝试导入 opt_einsum 模块，如果导入失败则将 _opt_einsum 置为 None
    import opt_einsum as _opt_einsum  # type: ignore[import]
except ImportError:
    _opt_einsum = None


@_lru_cache
# 定义函数 is_available，返回 opt_einsum 是否可用的布尔值
def is_available() -> bool:
    r"""Return a bool indicating if opt_einsum is currently available."""
    return _opt_einsum is not None


# 定义函数 get_opt_einsum，返回 opt_einsum 模块对象或者 None
def get_opt_einsum() -> Any:
    r"""Return the opt_einsum package if opt_einsum is currently available, else None."""
    return _opt_einsum


# 定义函数 _set_enabled，设置 opt_einsum 是否可用
def _set_enabled(_enabled: bool) -> None:
    # 如果 opt_einsum 不可用且试图设置为 True，则引发 ValueError 异常
    if not is_available() and _enabled:
        raise ValueError(
            f"opt_einsum is not available, so setting `enabled` to {_enabled} will not reap "
            "the benefits of calculating an optimal path for einsum. torch.einsum will "
            "fall back to contracting from left to right. To enable this optimal path "
            "calculation, please install opt-einsum."
        )
    # 设置全局变量 enabled 的值
    global enabled
    enabled = _enabled


# 定义函数 _get_enabled，获取当前 opt_einsum 是否可用的状态
def _get_enabled() -> bool:
    return enabled


# 定义函数 _set_strategy，设置 opt_einsum 的计算策略
def _set_strategy(_strategy: str) -> None:
    # 如果 opt_einsum 不可用，则引发 ValueError 异常
    if not is_available():
        raise ValueError(
            f"opt_einsum is not available, so setting `strategy` to {_strategy} will not be meaningful. "
            "torch.einsum will bypass path calculation and simply contract from left to right. "
            "Please install opt_einsum or unset `strategy`."
        )
    # 如果 opt_einsum 不启用，则引发 ValueError 异常
    if not enabled:
        raise ValueError(
            f"opt_einsum is not enabled, so setting a `strategy` to {_strategy} will not be meaningful. "
            "torch.einsum will bypass path calculation and simply contract from left to right. "
            "Please set `enabled` to `True` as well or unset `strategy`."
        )
    # 如果策略不是 ["auto", "greedy", "optimal"] 中的一个，则引发 ValueError 异常
    if _strategy not in ["auto", "greedy", "optimal"]:
        raise ValueError(
            f"`strategy` must be one of the following: [auto, greedy, optimal] but is {_strategy}"
        )
    # 设置全局变量 strategy 的值
    global strategy
    strategy = _strategy


# 定义函数 _get_strategy，获取当前 opt_einsum 的计算策略
def _get_strategy() -> str:
    return strategy


# 定义函数 set_flags，用于设置 enabled 和 strategy 的值，并返回原始值的元组
def set_flags(_enabled=None, _strategy=None):
    orig_flags = (enabled, None if not is_available() else strategy)
    if _enabled is not None:
        _set_enabled(_enabled)
    if _strategy is not None:
        _set_strategy(_strategy)
    return orig_flags


# 定义上下文管理器 flags，用于在上下文中设置 enabled 和 strategy 的值
@contextmanager
def flags(enabled=None, strategy=None):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, strategy)
    try:
        yield
    finally:
        # 恢复之前的值
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# OptEinsumModule 类继承自 PropModule，用于处理与 opt_einsum 相关的属性设置
class OptEinsumModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    # 此处可能是为了在模块中设置 opt_einsum 的 enabled 属性，但是代码块不完整，无法确定
    global enabled
    # 创建一个 ContextProp 对象并将其赋值给 enabled，使用 _get_enabled 和 _set_enabled 作为获取和设置函数
    enabled = ContextProp(_get_enabled, _set_enabled)
    # 设置全局变量 strategy，并初始化为 None
    global strategy
    strategy = None
    # 检查当前环境是否可用
    if is_available():
        # 如果可用，则创建一个 ContextProp 对象并将其赋值给 strategy，使用 _get_strategy 和 _set_strategy 作为获取和设置函数
        strategy = ContextProp(_get_strategy, _set_strategy)
# 将当前模块在 sys.modules 中替换为 OptEinsumModule 的实例，实现模块的替换技巧。
# 参考链接：https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = OptEinsumModule(sys.modules[__name__], __name__)

# 根据 is_available() 函数的返回值设置 enabled 变量，如果可用则为 True，否则为 False。
enabled = True if is_available() else False

# 根据 is_available() 函数的返回值设置 strategy 变量，如果可用则为 "auto"，否则为 None。
strategy = "auto" if is_available() else None
```