# `.\pytorch\torch\cuda\streams.py`

```
# mypy: allow-untyped-defs
# 引入 ctypes 库，用于处理 C 语言类型和函数
import ctypes

# 引入 torch 库
import torch
# 引入内部模块 _EventBase 和 _StreamBase
from torch._streambase import _EventBase, _StreamBase
# 引入内部工具模块 _dummy_type
from .._utils import _dummy_type

# 如果 torch._C 模块中没有定义 "_CudaStreamBase" 属性
if not hasattr(torch._C, "_CudaStreamBase"):
    # 定义虚拟的基础类 _CudaStreamBase 和 _CudaEventBase
    torch._C.__dict__["_CudaStreamBase"] = _dummy_type("_CudaStreamBase")
    torch._C.__dict__["_CudaEventBase"] = _dummy_type("_CudaEventBase")

# 定义 Stream 类，继承自 torch._C._CudaStreamBase 和 _StreamBase
class Stream(torch._C._CudaStreamBase, _StreamBase):
    r"""Wrapper around a CUDA stream.

    A CUDA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`cuda-semantics` for
    details.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, should be 0 or
            negative, where negative numbers indicate higher priority. By default,
            streams have priority 0.

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        # 如果 device 是 None 或者 kwargs 中包含 "stream_id" 和 "device_index"
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            # 调用父类的 __new__ 方法，创建一个新的实例
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            # 否则，使用指定的设备
            with torch.cuda.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (torch.cuda.Event): an event to wait for.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
           `CUDA Stream documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        .. _CUDA Stream documentation:
           https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
        # 让提交到流中的所有未来工作等待事件的发生
        event.wait(self)

    def wait_stream(self, stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        # 与另一个流同步，等待所有当前提交给给定流的内核完成
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Record an event.

        Args:
            event (torch.cuda.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        # 记录一个事件
        if event is None:
            event = Event()
        event.record(self)
        return event
    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        # 调用父类方法，检查所有提交的工作是否已完成，并返回结果
        return super().query()

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``cudaStreamSynchronize()``: see
           `CUDA Stream documentation`_ for more info.
        """
        # 调用父类方法，等待此流中的所有内核完成
        super().synchronize()

    @property
    def _as_parameter_(self):
        # 返回当前 CUDA 流的 ctypes.c_void_p 对象表示形式
        return ctypes.c_void_p(self.cuda_stream)

    def __eq__(self, o) -> bool:
        if isinstance(o, Stream):
            # 检查另一个对象是否是 Stream 类型，然后调用父类方法判断是否相等
            return super().__eq__(o)
        # 如果不是 Stream 类型，则返回 False
        return False

    def __hash__(self):
        # 返回此对象的哈希值，基于 CUDA 流和设备的元组
        return hash((self.cuda_stream, self.device))

    def __repr__(self):
        # 返回对象的字符串表示形式，包含设备和 CUDA 流的详细信息
        return f"<torch.cuda.Stream device={self.device} cuda_stream={self.cuda_stream:#x}>"
# 定义了一个外部 CUDA 流的包装器类
class ExternalStream(Stream):
    r"""Wrapper around an externally allocated CUDA stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `cudaStream_t` value.
            allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """

    def __new__(cls, stream_ptr, device=None, **kwargs):
        # 在指定的设备上创建 CUDA 流对象
        with torch.cuda.device(device):
            # 调用父类的构造方法来创建实例
            return super().__new__(cls, stream_ptr=stream_ptr, **kwargs)


# 定义了一个 CUDA 事件的包装器类
class Event(torch._C._CudaEventBase, _EventBase):
    r"""Wrapper around a CUDA event.

    CUDA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize CUDA
    streams.

    The underlying CUDA events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

    .. _CUDA Event Documentation:
       https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        # 调用父类构造方法创建 CUDA 事件实例
        return super().__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
        )

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        # 从 IPC 句柄和设备重建事件对象
        return super().from_ipc_handle(device, handle)

    def record(self, stream=None):
        r"""Record the event in a given stream.

        Uses ``torch.cuda.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        # 如果未指定流，则使用当前 CUDA 流
        if stream is None:
            stream = torch.cuda.current_stream()
        # 调用父类方法，在指定的流中记录事件
        super().record(stream)
    def wait(self, stream=None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Use ``torch.cuda.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
            `CUDA Event documentation`_ for more info.
        """
        # 如果未指定流，则使用当前的 CUDA 流
        if stream is None:
            stream = torch.cuda.current_stream()
        # 调用父类方法，使给定流上提交的所有未来工作等待此事件完成
        super().wait(stream)

    def query(self):
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        # 调用父类方法，查询当前事件所捕获的所有工作是否已经完成
        return super().query()

    def elapsed_time(self, end_event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.
        """
        # 调用父类方法，返回事件记录后到 end_event 记录前经过的时间，以毫秒为单位
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``cudaEventSynchronize()``: see
            `CUDA Event documentation`_ for more info.
        """
        # 调用父类方法，等待事件完成，直到当前事件捕获的所有工作完成
        super().synchronize()

    def ipc_handle(self):
        r"""Return an IPC handle of this event.

        If not recorded yet, the event will use the current device.
        """
        # 调用父类方法，返回此事件的 IPC 句柄
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        # 返回此事件的 CUDA 事件句柄作为 void 指针
        return ctypes.c_void_p(self.cuda_event)

    def __repr__(self) -> str:
        if self.cuda_event:
            # 如果存在 CUDA 事件句柄，返回 CUDA 事件的字符串表示形式
            return f"<torch.cuda.Event {self._as_parameter_.value:#x}>"
        else:
            # 如果不存在 CUDA 事件句柄，返回未初始化的 CUDA 事件的字符串表示形式
            return "<torch.cuda.Event uninitialized>"
```