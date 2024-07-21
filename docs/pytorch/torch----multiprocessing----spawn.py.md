# `.\pytorch\torch\multiprocessing\spawn.py`

```py
# mypy: allow-untyped-defs
import logging  # 导入日志模块
import multiprocessing  # 导入多进程模块
import multiprocessing.connection  # 导入多进程连接模块
import os  # 导入操作系统相关功能模块
import pickle  # 导入 pickle 序列化模块
import signal  # 导入信号处理模块
import sys  # 导入系统相关模块
import tempfile  # 导入临时文件模块
import time  # 导入时间模块
import warnings  # 导入警告模块
from typing import Optional  # 导入类型提示相关功能

from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]  # 导入 Linux 特有的进程控制函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class ProcessException(Exception):
    __slots__ = ["error_index", "error_pid"]

    def __init__(self, msg: str, error_index: int, pid: int):
        super().__init__(msg)
        self.msg = msg
        self.error_index = error_index
        self.pid = pid

    def __reduce__(self):
        return type(self), (self.msg, self.error_index, self.pid)


class ProcessRaisedException(ProcessException):
    """Exception raised when a process failed due to an exception raised by the code."""

    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
    ):
        super().__init__(msg, error_index, error_pid)


class ProcessExitedException(ProcessException):
    """Exception raised when a process failed due to signal or exited with a specific code."""

    __slots__ = ["exit_code"]

    def __init__(
        self,
        msg: str,
        error_index: int,
        error_pid: int,
        exit_code: int,
        signal_name: Optional[str] = None,
    ):
        super().__init__(msg, error_index, error_pid)
        self.exit_code = exit_code
        self.signal_name = signal_name

    def __reduce__(self):
        return (
            type(self),
            (self.msg, self.error_index, self.pid, self.exit_code, self.signal_name),
        )


def _wrap(fn, i, args, error_file):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)  # 设置父进程退出时发送 SIGINT 信号给子进程

    try:
        fn(i, *args)  # 调用传入的函数 fn，传入参数 i 和 args
    except KeyboardInterrupt:
        pass  # 捕获键盘中断信号 SIGINT，父进程终止，子进程不做任何处理
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback

        with open(error_file, "wb") as fh:
            pickle.dump(traceback.format_exc(), fh)  # 将异常信息以序列化形式保存到指定文件中
        sys.exit(1)  # 异常处理后退出子进程


class ProcessContext:
    def __init__(self, processes, error_files):
        self.error_files = error_files  # 初始化错误文件列表
        self.processes = processes  # 初始化进程列表
        self.sentinels = {
            process.sentinel: index for index, process in enumerate(processes)
        }  # 创建进程 sentinels 的索引字典

    def pids(self):
        return [int(process.pid) for process in self.processes]  # 返回所有进程的进程号列表


class SpawnContext(ProcessContext):
    def __init__(self, processes, error_files):
        warnings.warn("SpawnContext is renamed to ProcessContext since 1.4 release.")  # 发出警告，自 1.4 版本起 SpawnContext 更名为 ProcessContext
        super().__init__(processes, error_files)  # 调用父类初始化方法


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# 更通用的 API，比 mp.spawn 更一般化。目前我们仅将 mp.spawn 文档化，因为它是与 CUDA 兼容的 start_method。
# 然而，在像 IPython 笔记本这样的环境中，'fork' 比 'spawn' 更有效。我们为 mp.spawn 创建的每个辅助函数确实足够通用，
# 并且像 XLA 这样的后端也可以在 Colab 笔记本中重用它们。
# 目前我们先添加这个 API，以后可以考虑根据需要将其添加到文档中。
def start_processes(
    fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"
):
    # 使用指定的 start_method 获取 multiprocessing 上下文
    mp = multiprocessing.get_context(start_method)
    # 存储错误文件名的列表
    error_files = []
    # 存储进程对象的列表
    processes = []
    for i in range(nprocs):
        # 每个进程分配一个文件用于写入回溯信息。我们使用文件是否非空来指示异常发生（而不是预期的关闭）。
        # 注意：之前使用过 multiprocessing.Queue，但那可能会导致死锁，因此我们选择了更简单的一次性消息解决方案。
        tf = tempfile.NamedTemporaryFile(
            prefix="pytorch-errorfile-", suffix=".pickle", delete=False
        )
        tf.close()
        os.unlink(tf.name)
        # 创建进程对象，指定目标函数为 _wrap，参数包括 fn 函数、进程编号 i、传入参数 args 和错误文件名 tf.name
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, tf.name),
            daemon=daemon,
        )
        # 启动进程
        process.start()
        # 将错误文件名加入到列表中
        error_files.append(tf.name)
        # 将进程对象加入到列表中
        processes.append(process)

    # 创建 ProcessContext 对象，包含进程列表和错误文件名列表
    context = ProcessContext(processes, error_files)
    # 如果不需要等待进程结束，则直接返回 ProcessContext 对象
    if not join:
        return context

    # 循环等待所有进程结束
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.
    """
    # 如果启动方法不是 "spawn"，则发出警告并将来自 `start_processes` 函数的信息存储在 `msg` 变量中
    if start_method != "spawn":
        msg = (
            f"This method only supports start_method=spawn (got: {start_method}).\n"
            "To use a different start_method use:\n\t\t"
            " torch.multiprocessing.start_processes(...)"
        )
        # 发出未来警告，指定警告类型为 FutureWarning，警告等级为函数调用的下两层
        warnings.warn(msg, FutureWarning, stacklevel=2)
    
    # 调用 `start_processes` 函数，传递给它的参数为 `fn, args, nprocs, join, daemon, start_method="spawn"`
    # 如果 `join` 参数为 `True`，则返回 `None`；如果 `join` 参数为 `False`，则返回 `ProcessContext` 类的实例
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
```