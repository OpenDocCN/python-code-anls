# `.\pytorch\torch\distributed\_shard\op_registry_utils.py`

```py
# 设置类型检查允许未类型化的定义
mypy: allow-untyped-defs

# 导入 functools 模块，用于装饰器功能
import functools
# 从 inspect 模块中导入 signature 函数，用于获取函数签名信息
from inspect import signature

# 从当前包中导入 _basic_validation 函数
from .common_op_utils import _basic_validation

"""
ShardedTensor 和 PartialTensor 上注册操作的常用工具函数。
"""

def _register_op(op, func, op_table):
    """
    执行基本验证并将提供的操作注册到给定的操作表中。
    """
    # 检查函数签名参数数量是否为4
    if len(signature(func).parameters) != 4:
        # 如果参数数量不符合预期，抛出类型错误异常
        raise TypeError(
            f"Custom sharded op function expects signature: "
            f"(types, args, kwargs, process_group), but received "
            f"signature: {signature(func)}"
        )

    # 将操作函数注册到操作表中
    op_table[op] = func


def _decorator_func(wrapped_func, op, op_table):
    """
    装饰器函数，用于将给定的操作注册到提供的操作表中。
    """

    # 定义装饰后的函数
    @functools.wraps(wrapped_func)
    def wrapper(types, args, kwargs, process_group):
        # 执行基本验证操作
        _basic_validation(op, args, kwargs)
        # 调用原始函数并返回结果
        return wrapped_func(types, args, kwargs, process_group)

    # 将装饰后的函数注册到操作表中
    _register_op(op, wrapper, op_table)
    # 返回装饰后的函数
    return wrapper
```