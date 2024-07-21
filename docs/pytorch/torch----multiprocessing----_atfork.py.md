# `.\pytorch\torch\multiprocessing\_atfork.py`

```py
# mypy: allow-untyped-defs
# 导入 sys 模块，用于系统相关操作
import sys

# 定义 __all__ 列表，指定模块导出的公共接口
__all__ = ["register_after_fork"]

# 根据系统平台判断导入不同的模块
if sys.platform == "win32":
    # 在 Windows 平台下，导入 multiprocessing.util 模块作为 _util
    import multiprocessing.util as _util

    # 定义 _register 函数，接受一个函数作为参数，并注册一个 wrapper 函数
    def _register(func):
        # 定义 wrapper 函数，接受一个参数，并调用传入的 func 函数
        def wrapper(arg):
            func()

        # 使用 multiprocessing.util 模块的 register_after_fork 方法注册 _register 函数
        _util.register_after_fork(_register, wrapper)

else:
    # 在非 Windows 平台下，导入 os 模块
    import os

    # 定义 _register 函数，接受一个函数作为参数，并在子进程中调用传入的 func 函数
    def _register(func):
        os.register_at_fork(after_in_child=func)


# 定义 register_after_fork 函数，用于注册一个在子进程 fork 后执行的回调函数
def register_after_fork(func):
    """Register a callable to be executed in the child process after a fork.

    Note:
        In python < 3.7 this will only work with processes created using the
        ``multiprocessing`` module. In python >= 3.7 it also works with
        ``os.fork()``.

    Args:
        func (function): Function taking no arguments to be called in the child after fork

    """
    # 调用 _register 函数，将传入的 func 函数注册为子进程 fork 后执行的回调函数
    _register(func)
```