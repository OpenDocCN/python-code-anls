# `.\pytorch\torch\_functorch\utils.py`

```py
# 引入标准库 contextlib 用于支持上下文管理器
import contextlib
# 引入 typing 库中的 Tuple 和 Union 类型，用于类型标注
from typing import Tuple, Union

# 引入 torch 库
import torch
# 从 torch._C._functorch 模块中导入指定函数
from torch._C._functorch import (
    get_single_level_autograd_function_allowed,
    set_single_level_autograd_function_allowed,
    unwrap_if_dead,
)
# 从 torch.utils._exposed_in 模块中导入 exposed_in 函数
from torch.utils._exposed_in import exposed_in

# 定义公开的全局变量列表 __all__
__all__ = [
    "exposed_in",
    "argnums_t",
    "enable_single_level_autograd_function",
    "unwrap_dead_wrappers",
]

# 定义上下文管理器 enable_single_level_autograd_function
@contextlib.contextmanager
def enable_single_level_autograd_function():
    try:
        # 获取当前是否允许单层自动微分函数的状态，并保存为 prev_state
        prev_state = get_single_level_autograd_function_allowed()
        # 设置允许单层自动微分函数
        set_single_level_autograd_function_allowed(True)
        # yield 语句之前的代码作为上下文管理器的进入部分
        yield
    finally:
        # 在上下文管理器结束后，恢复之前的单层自动微分函数状态
        set_single_level_autograd_function_allowed(prev_state)

# 定义函数 unwrap_dead_wrappers，用于解包已死亡的包装器
def unwrap_dead_wrappers(args):
    # 使用生成器表达式遍历参数 args，如果是 torch.Tensor 类型，则解包死亡的包装器，否则保持原样
    result = tuple(
        unwrap_if_dead(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    )
    # 返回解包后的结果作为元组
    return result

# 定义类型别名 argnums_t，表示参数索引或索引元组
argnums_t = Union[int, Tuple[int, ...]]
```