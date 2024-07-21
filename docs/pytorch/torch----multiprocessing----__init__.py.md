# `.\pytorch\torch\multiprocessing\__init__.py`

```py
# mypy: allow-untyped-defs
"""torch.multiprocessing is a wrapper around the native :mod:`multiprocessing` module.

It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""
# 导入 multiprocessing 模块
import multiprocessing
# 导入 sys 模块
import sys

# 导入 torch 库
import torch
# 导入 reductions 模块中的 init_reductions 函数
from .reductions import init_reductions

# 定义 __all__ 列表，包含对外公开的符号
__all__ = ["set_sharing_strategy", "get_sharing_strategy", "get_all_sharing_strategies"]

# 从 multiprocessing 模块导入所有符号
from multiprocessing import *  # noqa: F403

# 将 multiprocessing 模块的所有符号添加到 __all__ 列表中，忽略类型检查错误
__all__ += multiprocessing.__all__  # noqa: PLE0605 type: ignore[attr-defined]

# This call adds a Linux specific prctl(2) wrapper function to this module.
# See https://github.com/pytorch/pytorch/pull/14391 for more information.
# 调用此函数向该模块添加了一个特定于 Linux 的 prctl(2) 包装函数。
# 更多信息请参考 https://github.com/pytorch/pytorch/pull/14391
torch._C._multiprocessing_init()

# Add helper function to spawn N processes and wait for completion of any of
# them. This depends `mp.get_context` which was added in Python 3.4.
"""添加一个辅助函数以生成 N 个进程并等待任何一个完成。这依赖于 Python 3.4 中添加的 `mp.get_context`。"""
from .spawn import (
    ProcessContext,
    ProcessExitedException,
    ProcessRaisedException,
    spawn,
    SpawnContext,
    start_processes,
)

# 根据不同的系统平台设置共享策略
if sys.platform == "darwin" or sys.platform == "win32":
    _sharing_strategy = "file_system"
    _all_sharing_strategies = {"file_system"}
else:
    _sharing_strategy = "file_descriptor"
    _all_sharing_strategies = {"file_descriptor", "file_system"}

# 设置共享策略的函数，允许选择不同的共享策略
def set_sharing_strategy(new_strategy):
    """Set the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    assert new_strategy in _all_sharing_strategies
    _sharing_strategy = new_strategy

# 获取当前共享策略的函数
def get_sharing_strategy():
    """Return the current strategy for sharing CPU tensors."""
    return _sharing_strategy

# 获取所有支持的共享策略的函数
def get_all_sharing_strategies():
    """Return a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies

# 设置当前线程的名称的函数
def _set_thread_name(name: str) -> None:
    """Set the name of the current thread.

    Args:
        name (str): Name of the current thread.
    """
    torch._C._set_thread_name(name)

# 获取当前线程名称的函数
def _get_thread_name() -> str:
    """Get the name of the current thread.

    Returns:
        str: Name of the current thread.
    """
    return torch._C._get_thread_name()

# 初始化 reductions 模块
init_reductions()
```