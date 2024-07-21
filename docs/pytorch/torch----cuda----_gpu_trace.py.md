# `.\pytorch\torch\cuda\_gpu_trace.py`

```
# 引入类型提示中的 Callable 类型
from typing import Callable

# 引入 Torch 库中的 CallbackRegistry 类
from torch._utils import CallbackRegistry

# 定义一个回调注册表，用于 CUDA 事件的创建，存储 int 类型的回调函数
EventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event creation"
)

# 定义一个回调注册表，用于 CUDA 事件的删除，存储 int 类型的回调函数
EventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event deletion"
)

# 定义一个回调注册表，用于 CUDA 事件的记录，存储两个 int 类型参数的回调函数
EventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "CUDA event record"
)

# 定义一个回调注册表，用于 CUDA 事件的等待，存储两个 int 类型参数的回调函数
EventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "CUDA event wait"
)

# 定义一个回调注册表，用于 CUDA 内存的分配，存储 int 类型的回调函数
MemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory allocation"
)

# 定义一个回调注册表，用于 CUDA 内存的释放，存储 int 类型的回调函数
MemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory deallocation"
)

# 定义一个回调注册表，用于 CUDA 流的创建，存储 int 类型的回调函数
StreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream creation"
)

# 定义一个回调注册表，用于 CUDA 设备的同步，存储无参数的回调函数
DeviceSynchronizationCallbacks: "CallbackRegistry[[]]" = CallbackRegistry(
    "CUDA device synchronization"
)

# 定义一个回调注册表，用于 CUDA 流的同步，存储 int 类型的回调函数
StreamSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream synchronization"
)

# 定义一个回调注册表，用于 CUDA 事件的同步，存储 int 类型的回调函数
EventSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event synchronization"
)

# 注册一个回调函数，用于 CUDA 事件的创建，接受一个 int 参数和返回值为 None
def register_callback_for_event_creation(cb: Callable[[int], None]) -> None:
    EventCreationCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 事件的删除，接受一个 int 参数和返回值为 None
def register_callback_for_event_deletion(cb: Callable[[int], None]) -> None:
    EventDeletionCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 事件的记录，接受两个 int 参数和返回值为 None
def register_callback_for_event_record(cb: Callable[[int, int], None]) -> None:
    EventRecordCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 事件的等待，接受两个 int 参数和返回值为 None
def register_callback_for_event_wait(cb: Callable[[int, int], None]) -> None:
    EventWaitCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 内存的分配，接受一个 int 参数和返回值为 None
def register_callback_for_memory_allocation(cb: Callable[[int], None]) -> None:
    MemoryAllocationCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 内存的释放，接受一个 int 参数和返回值为 None
def register_callback_for_memory_deallocation(cb: Callable[[int], None]) -> None:
    MemoryDeallocationCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 流的创建，接受一个 int 参数和返回值为 None
def register_callback_for_stream_creation(cb: Callable[[int], None]) -> None:
    StreamCreationCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 设备的同步，接受无参数和返回值为 None
def register_callback_for_device_synchronization(cb: Callable[[], None]) -> None:
    DeviceSynchronizationCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 流的同步，接受一个 int 参数和返回值为 None
def register_callback_for_stream_synchronization(cb: Callable[[int], None]) -> None:
    StreamSynchronizationCallbacks.add_callback(cb)

# 注册一个回调函数，用于 CUDA 事件的同步，接受一个 int 参数和返回值为 None
def register_callback_for_event_synchronization(cb: Callable[[int], None]) -> None:
    EventSynchronizationCallbacks.add_callback(cb)
```