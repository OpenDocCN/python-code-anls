# `.\pytorch\torch\mps\profiler.py`

```py
# mypy: allow-untyped-defs
# 引入上下文管理模块contextlib
import contextlib

# 引入torch模块
import torch

# 定义__all__变量，包含公开的函数名称列表
__all__ = ["start", "stop", "profile"]


# 定义函数start，启动 OS Signpost 跟踪功能
def start(mode: str = "interval", wait_until_completed: bool = False) -> None:
    r"""Start OS Signpost tracing from MPS backend.

    The generated OS Signposts could be recorded and viewed in
    XCode Instruments Logging tool.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    # 标准化跟踪模式字符串，去除空格并转为小写
    mode_normalized = mode.lower().replace(" ", "")
    # 调用torch内部函数，启动 MPS Profiler 的跟踪
    torch._C._mps_profilerStartTrace(mode_normalized, wait_until_completed)


# 定义函数stop，停止 OS Signpost 跟踪功能
def stop():
    r"""Stops generating OS Signpost tracing from MPS backend."""
    # 调用torch内部函数，停止 MPS Profiler 的跟踪
    torch._C._mps_profilerStopTrace()


# 定义上下文管理器profile，用于启动和停止 OS Signpost 跟踪功能
@contextlib.contextmanager
def profile(mode: str = "interval", wait_until_completed: bool = False):
    r"""Context Manager to enabling generating OS Signpost tracing from MPS backend.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    try:
        # 调用start函数，开始 MPS Profiler 的跟踪
        start(mode, wait_until_completed)
        # 使用yield暂时离开上下文管理器，执行嵌套在with语句中的代码块
        yield
    finally:
        # 不论代码块中是否发生异常，最终都调用stop函数，停止 MPS Profiler 的跟踪
        stop()
```