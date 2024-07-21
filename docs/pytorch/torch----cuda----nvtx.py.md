# `.\pytorch\torch\cuda\nvtx.py`

```
# mypy: allow-untyped-defs
r"""This package adds support for NVIDIA Tools Extension (NVTX) used in profiling."""

from contextlib import contextmanager  # 导入上下文管理器模块

try:
    from torch._C import _nvtx  # 尝试导入 _nvtx 模块
except ImportError:

    class _NVTXStub:
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError(
                "NVTX functions not installed. Are you sure you have a CUDA build?"
            )

        rangePushA = _fail  # 定义 rangePushA 方法为 _fail 方法
        rangePop = _fail  # 定义 rangePop 方法为 _fail 方法
        markA = _fail  # 定义 markA 方法为 _fail 方法

    _nvtx = _NVTXStub()  # type: ignore[assignment] 创建 _nvtx 实例，用于兼容性处理

__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]  # 导出的函数列表


def range_push(msg):
    """
    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
    """
    return _nvtx.rangePushA(msg)  # 调用 _nvtx 实例的 rangePushA 方法，并返回结果


def range_pop():
    """Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended."""
    return _nvtx.rangePop()  # 调用 _nvtx 实例的 rangePop 方法，并返回结果


def range_start(msg) -> int:
    """
    Mark the start of a range with string message. It returns an unique handle
    for this range to pass to the corresponding call to rangeEnd().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns: A range handle (uint64_t) that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
    """
    return _nvtx.rangeStartA(msg)  # 调用 _nvtx 实例的 rangeStartA 方法，并返回结果


def range_end(range_id) -> None:
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): an unique handle for the start range.
    """
    _nvtx.rangeEnd(range_id)  # 调用 _nvtx 实例的 rangeEnd 方法，结束给定 range_id 的范围


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
    return _nvtx.markA(msg)  # 调用 _nvtx 实例的 markA 方法，并返回结果


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))  # 在范围的开始推入一个 NVTX 范围
    try:
        yield  # 执行作用域内的代码
    finally:
        range_pop()  # 在范围的结束弹出 NVTX 范围
```