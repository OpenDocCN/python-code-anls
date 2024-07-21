# `.\pytorch\torch\xpu\streams.py`

```py
# mypy: allow-untyped-defs
# 导入 ctypes 模块，用于处理 C 数据类型
import ctypes

# 导入 torch 库
import torch
# 导入 _EventBase 和 _StreamBase 类
from torch._streambase import _EventBase, _StreamBase

# 导入 _dummy_type 函数，用于创建虚拟类型
from .._utils import _dummy_type

# 如果 torch._C 模块没有定义 "_XpuStreamBase" 属性
if not hasattr(torch._C, "_XpuStreamBase"):
    # 定义虚拟基类 "_XpuStreamBase"
    torch._C.__dict__["_XpuStreamBase"] = _dummy_type("_XpuStreamBase")
    # 定义虚拟基类 "_XpuEventBase"
    torch._C.__dict__["_XpuEventBase"] = _dummy_type("_XpuEventBase")

# 定义 Stream 类，继承自 torch._C._XpuStreamBase 和 _StreamBase
class Stream(torch._C._XpuStreamBase, _StreamBase):
    r"""Wrapper around a XPU stream.

    A XPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, should be 0 or
            negative, where negative numbers indicate higher priority. By default,
            streams have priority 0.
    """

    # 构造函数，根据参数创建 Stream 对象
    def __new__(cls, device=None, priority=0, **kwargs):
        # 设置设备管理器是昂贵的，除非有必要，否则避免使用
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            # 返回父类构造方法的结果
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            # 使用 torch.xpu.device 上下文管理器设置指定设备
            with torch.xpu.device(device):
                # 返回父类构造方法的结果
                return super().__new__(cls, priority=priority, **kwargs)

    # 等待事件的方法，使流中提交的所有未来工作等待事件完成
    def wait_event(self, event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (torch.xpu.Event): an event to wait for.
        """
        event.wait(self)

    # 等待另一个流的方法，同步当前流和给定流中的工作
    def wait_stream(self, stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.
        """
        self.wait_event(stream.record_event())

    # 记录事件的方法
    def record_event(self, event=None):
        r"""Record an event.

        Args:
            event (torch.xpu.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    # 查询方法，检查提交的所有工作是否已完成
    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    # 同步方法，等待当前流中的所有核心完成
    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    # 属性方法，返回当前流的参数表示
    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.sycl_queue)

    # 等式比较方法，判断两个 Stream 对象是否相等
    def __eq__(self, o):
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False
    # 定义对象的哈希方法，用于计算对象的哈希值
    def __hash__(self):
        return hash((self.sycl_queue, self.device))

    # 定义对象的字符串表示方法，返回一个描述对象的字符串
    def __repr__(self):
        return f"torch.xpu.Stream(device={self.device} sycl_queue={self.sycl_queue:#x})"
class Event(torch._C._XpuEventBase, _EventBase):
    r"""Wrapper around a XPU event.

    XPU events are synchronization markers that can be used to monitor the
    device's progress, and to synchronize XPU streams.

    The underlying XPU events are lazily initialized when the event is first
    recorded. After creation, only streams on the same device may record the
    event. However, streams on any device can wait on the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """

    def __new__(cls, enable_timing=False):
        # 创建一个新的实例，继承自基类，可以选择是否启用计时功能
        return super().__new__(cls, enable_timing=enable_timing)

    def record(self, stream=None) -> None:
        r"""Record the event in a given stream.

        Uses ``torch.xpu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        # 如果未指定流，则使用当前的 XPU 流
        if stream is None:
            stream = torch.xpu.current_stream()
        # 调用基类方法，在给定的流上记录事件
        super().record(stream)

    def wait(self, stream=None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Use ``torch.xpu.current_stream()`` if no stream is specified.
        """
        # 如果未指定流，则使用当前的 XPU 流
        if stream is None:
            stream = torch.xpu.current_stream()
        # 调用基类方法，使给定流上的所有未来工作等待此事件
        super().wait(stream)

    def query(self) -> bool:
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        # 调用基类方法，检查当前事件捕获的所有工作是否已完成
        return super().query()

    def elapsed_time(self, end_event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.
        """
        # 返回从记录事件到记录 end_event 之间经过的时间（毫秒）
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        # 调用基类方法，等待事件完成，直到当前事件捕获的所有工作完成
        super().synchronize()

    @property
    def _as_parameter_(self):
        # 返回作为参数的值，此处为内部 SYCL 事件的 void 指针
        return ctypes.c_void_p(self.sycl_event)

    def __repr__(self):
        # 返回事件的字符串表示形式，如果有 SYCL 事件则显示其地址，否则显示未初始化
        if self.sycl_event:
            return f"torch.xpu.Event(sycl_event={self.sycl_event:#x})"
        else:
            return "torch.xpu.Event(uninitialized)"
```