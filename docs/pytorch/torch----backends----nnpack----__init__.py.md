# `.\pytorch\torch\backends\nnpack\__init__.py`

```py
# mypy: allow-untyped-defs
# 引入上下文管理器模块
from contextlib import contextmanager

# 导入 torch 库
import torch
# 从 torch.backends 中导入特定内容
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule

# 定义公开的模块成员
__all__ = ["is_available", "flags", "set_flags"]

# 检查 PyTorch 是否构建了 NNPACK 支持，返回布尔值
def is_available():
    r"""Return whether PyTorch is built with NNPACK support."""
    return torch._nnpack_available()

# 设置全局 nnpack 是否启用的标志，并返回之前的标志状态
def set_flags(_enabled):
    r"""Set if nnpack is enabled globally"""
    # 获取当前 nnpack 的启用状态，并保存到 orig_flags 中
    orig_flags = (torch._C._get_nnpack_enabled(),)
    # 设置 nnpack 的全局启用状态为 _enabled
    torch._C._set_nnpack_enabled(_enabled)
    # 返回设置前的 nnpack 状态
    return orig_flags

# 上下文管理器，用于设置全局 nnpack 是否启用的标志
@contextmanager
def flags(enabled=False):
    r"""Context manager for setting if nnpack is enabled globally"""
    # 使用 __allow_nonbracketed_mutation 上下文管理器允许非括号化的变异操作
    with __allow_nonbracketed_mutation():
        # 设置 nnpack 的启用状态，并保存当前状态到 orig_flags 中
        orig_flags = set_flags(enabled)
    try:
        # 在上下文中执行代码块
        yield
    finally:
        # 在 finally 块中恢复 nnpack 的启用状态到原始状态
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])
```