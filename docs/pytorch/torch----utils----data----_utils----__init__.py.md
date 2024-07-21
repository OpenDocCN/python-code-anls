# `.\pytorch\torch\utils\data\_utils\__init__.py`

```py
# mypy: allow-untyped-defs
r"""Utility classes & functions for data loading. Code in this folder is mostly used by ../dataloder.py.

A lot of multiprocessing is used in data loading, which only supports running
functions defined in global environment (py2 can't serialize static methods).
Therefore, for code tidiness we put these functions into different files in this
folder.
"""

import atexit  # 导入 atexit 模块，用于注册退出时的清理函数
import sys  # 导入 sys 模块，用于访问系统相关的变量和函数

# old private location of the ExceptionWrapper that some users rely on:
from torch._utils import ExceptionWrapper  # 导入 ExceptionWrapper 类，可能是为了兼容旧版 torch 中的功能

IS_WINDOWS = sys.platform == "win32"  # 判断当前系统是否为 Windows，返回布尔值


MP_STATUS_CHECK_INTERVAL = 5.0
r"""Interval (in seconds) to check status of processes to avoid hanging in
    multiprocessing data loading. This is mainly used in getting data from
    another process, in which case we need to periodically check whether the
    sender is alive to prevent hanging."""
# 多进程数据加载中检查进程状态的时间间隔，防止阻塞。主要用于从另一个进程获取数据时，需要定期检查发送方是否活跃以避免阻塞。


python_exit_status = False
r"""Whether Python is shutting down. This flag is guaranteed to be set before
the Python core library resources are freed, but Python may already be exiting
for some time when this is set.

Hook to set this flag is `_set_python_exit_flag`, and is inspired by a similar
hook in Python 3.7 multiprocessing library:
https://github.com/python/cpython/blob/d4d60134b29290049e28df54f23493de4f1824b6/Lib/multiprocessing/util.py#L277-L327
"""
# Python 是否正在关闭的标志。这个标志在释放 Python 核心库资源之前必定被设置，但当设置时，Python 可能已经退出了一段时间。

try:
    import numpy  # 尝试导入 numpy 库

    HAS_NUMPY = True  # 设置是否成功导入 numpy 的标志
except ModuleNotFoundError:
    HAS_NUMPY = False  # 如果导入失败，则设置标志为 False


def _set_python_exit_flag():
    global python_exit_status
    python_exit_status = True  # 设置 Python 正在退出的标志


atexit.register(_set_python_exit_flag)  # 在程序退出时注册 _set_python_exit_flag 函数以设置退出标志


from . import collate, fetch, pin_memory, signal_handling, worker
# 导入本地模块 collate, fetch, pin_memory, signal_handling, worker
# 这些模块可能包含了数据加载过程中使用的协调、抓取、内存固定、信号处理和工作进程管理的功能。
```