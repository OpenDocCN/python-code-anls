# `.\pytorch\torch\mps\event.py`

```py
# mypy: allow-untyped-defs
# 导入 torch 库，用于与 PyTorch 相关的操作
import torch

# 定义一个 Event 类，用于包装 MPS（Metal Performance Shaders）事件
class Event:
    r"""Wrapper around an MPS event.

    MPS events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MPS streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    # 初始化方法，创建一个 MPS 事件
    def __init__(self, enable_timing=False):
        # 调用 torch._C._mps_acquireEvent 方法获取一个 MPS 事件 ID
        self.__eventId = torch._C._mps_acquireEvent(enable_timing)

    # 析构方法，在对象被销毁时调用，用于释放 MPS 事件
    def __del__(self):
        # 检查 torch._C 是否已销毁，并且事件 ID 大于 0
        if hasattr(torch._C, "_mps_releaseEvent") and self.__eventId > 0:
            # 调用 torch._C._mps_releaseEvent 方法释放 MPS 事件
            torch._C._mps_releaseEvent(self.__eventId)

    # 记录事件到默认流中的方法
    def record(self):
        r"""Records the event in the default stream."""
        # 调用 torch._C._mps_recordEvent 方法记录当前事件
        torch._C._mps_recordEvent(self.__eventId)

    # 使默认流中所有未来提交的工作等待此事件完成的方法
    def wait(self):
        r"""Makes all future work submitted to the default stream wait for this event."""
        # 调用 torch._C._mps_waitForEvent 方法使所有未来工作等待当前事件完成
        torch._C._mps_waitForEvent(self.__eventId)

    # 查询当前事件是否已完成方法
    def query(self):
        r"""Returns True if all work currently captured by event has completed."""
        # 调用 torch._C._mps_queryEvent 方法查询当前事件是否已完成
        return torch._C._mps_queryEvent(self.__eventId)

    # 同步方法，等待当前事件所捕获的所有工作完成
    def synchronize(self):
        r"""Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        # 调用 torch._C._mps_synchronizeEvent 方法等待当前事件完成
        torch._C._mps_synchronizeEvent(self.__eventId)

    # 计算从当前事件记录到结束事件记录之间经过的时间（单位：毫秒）方法
    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        # 调用 torch._C._mps_elapsedTimeOfEvents 方法计算两个事件之间的时间差
        return torch._C._mps_elapsedTimeOfEvents(self.__eventId, end_event.__eventId)
```