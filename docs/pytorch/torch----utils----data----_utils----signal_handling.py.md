# `.\pytorch\torch\utils\data\_utils\signal_handling.py`

```py
# mypy: allow-untyped-defs
r"""Signal handling for multiprocessing data loading.

NOTE [ Signal handling in multiprocessing data loading ]

In cases like DataLoader, if a worker process dies due to bus error/segfault
or just hang, the main process will hang waiting for data. This is difficult
to avoid on PyTorch side as it can be caused by limited shm, or other
libraries users call in the workers. In this file and `DataLoader.cpp`, we make
our best effort to provide some error message to users when such unfortunate
events happen.

When a _BaseDataLoaderIter starts worker processes, their pids are registered in a
defined in `DataLoader.cpp`: id(_BaseDataLoaderIter) => Collection[ Worker pids ]
via `_set_worker_pids`.

When an error happens in a worker process, the main process received a SIGCHLD,
and Python will eventually call the handler registered below
(in `_set_SIGCHLD_handler`). In the handler, the `_error_if_any_worker_fails`
call checks all registered worker pids and raise proper error message to
prevent main process from hanging waiting for data from worker.

Additionally, at the beginning of each worker's `_utils.worker._worker_loop`,
`_set_worker_signal_handlers` is called to register critical signal handlers
(e.g., for SIGSEGV, SIGBUS, SIGFPE, SIGTERM) in C, which just prints an error
message to stderr before triggering the default handler. So a message will also
be printed from the worker process when it is killed by such signals.

See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for the reasoning of
this signal handling design and other mechanism we implement to make our
multiprocessing data loading robust to errors.
"""

import signal
import threading

# Some of the following imported functions are not used in this file, but are to
# be used `_utils.signal_handling.XXXXX`.
from torch._C import (  # noqa: F401
    _error_if_any_worker_fails,
    _remove_worker_pids,
    _set_worker_pids,
    _set_worker_signal_handlers,
)

from . import IS_WINDOWS


_SIGCHLD_handler_set = False
r"""Whether SIGCHLD handler is set for DataLoader worker failures. Only one
handler needs to be set for all DataLoaders in a process."""


def _set_SIGCHLD_handler():
    """
    Set SIGCHLD signal handler to monitor child process state changes.

    On Windows, SIGCHLD is not supported. In non-main threads, signals cannot
    be set. If the SIGCHLD handler is already set, do not set it again.
    """
    if IS_WINDOWS:
        return  # Windows does not support SIGCHLD handler
    if not isinstance(threading.current_thread(), threading._MainThread):  # type: ignore[attr-defined]
        return  # Can't set signals in non-main threads
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return  # If SIGCHLD handler is already set, do not set again
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        # SIGCHLD default handler is a no-op, so set previous_handler to None
        previous_handler = None
    # 定义信号处理函数 `handler`，处理信号 `signum` 和信号处理框架 `frame`
    def handler(signum, frame):
        # 以下调用使用 `waitid` 与 WNOHANG 从 C 侧进行。因此，
        # Python 仍然可以成功获取和更新进程状态。
        _error_if_any_worker_fails()
        # 如果存在先前的信号处理程序，确保它是可调用的，并调用它
        if previous_handler is not None:
            assert callable(previous_handler)
            previous_handler(signum, frame)

    # 设置信号 `SIGCHLD` 的处理程序为 `handler` 函数
    signal.signal(signal.SIGCHLD, handler)
    # 标记 `_SIGCHLD_handler_set` 为已设置
    _SIGCHLD_handler_set = True
```