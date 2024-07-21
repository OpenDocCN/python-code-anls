# `.\pytorch\torch\_inductor\compile_worker\watchdog.py`

```
# mypy: allow-untyped-defs
# 引入操作系统相关的功能
import os
# 信号处理模块，用于处理信号
import signal
# 多线程模块中的线程类
from threading import Thread
# 时间模块中的睡眠函数
from time import sleep
# 类型提示模块中的可选类型
from typing import Optional

# 如果进程异常终止（如段错误），工作线程将不会正常关闭。
# 相反，工作线程的父进程将重新分配到init进程。
# 这段代码启动一个独立的线程来监视工作线程是否被重新分配，
# 并在这种情况下进行清理。
#
# 这个函数不能是内部函数，否则ProcessPoolExecutor的mp_context="spawn"模式无法工作，
# 因为内部函数无法被pickle序列化。
def _async_compile_initializer(orig_ppid) -> None:
    # 定义内部函数run，用于监视原始父进程ID是否发生变化
    def run() -> None:
        while True:
            # 每隔1秒钟检查一次
            sleep(1)
            # 如果原始父进程ID发生变化，则杀死当前进程
            if orig_ppid != os.getppid():
                os.kill(os.getpid(), signal.SIGKILL)

    # 声明全局变量_watchdog_thread和_original_parent
    global _watchdog_thread, _original_parent
    # 记录原始父进程ID
    _original_parent = orig_ppid
    # 创建线程_watchdog_thread，使其运行内部函数run，作为守护线程
    _watchdog_thread = Thread(target=run, daemon=True)
    # 启动守护线程_watchdog_thread
    _watchdog_thread.start()
    # 忽略对池工作进程发送的Ctrl-C（即SIGINT），以避免无意义的日志信息。
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# 声明全局变量_watchdog_thread，类型为Thread的可选类型
_watchdog_thread: Optional[Thread] = None
# 声明全局变量_original_parent，类型为整数的可选类型
_original_parent: Optional[int] = None


# 检查原始父进程ID是否发生变化的函数
def has_parent_changed() -> bool:
    # 返回原始父进程ID与当前父进程ID是否不相等的布尔值结果
    return _original_parent != os.getppid()
```