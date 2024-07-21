# `.\pytorch\torch\xpu\_gpu_trace.py`

```py
# 导入必要的模块和类型定义
from typing import Callable
from torch._utils import CallbackRegistry

# 创建用于不同事件的回调注册表实例，并提供相关描述
EventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU event creation"
)
EventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU event deletion"
)
EventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "XPU event record"
)
EventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "XPU event wait"
)
MemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU memory allocation"
)
MemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU memory deallocation"
)
StreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU stream creation"
)
DeviceSynchronizationCallbacks: "CallbackRegistry[[]]" = CallbackRegistry(
    "XPU device synchronization"
)
StreamSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU stream synchronization"
)
EventSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "XPU event synchronization"
)

# 定义用于注册回调函数的函数，并将其添加到相应的回调注册表中
def register_callback_for_event_creation(cb: Callable[[int], None]) -> None:
    EventCreationCallbacks.add_callback(cb)

def register_callback_for_event_deletion(cb: Callable[[int], None]) -> None:
    EventDeletionCallbacks.add_callback(cb)

def register_callback_for_event_record(cb: Callable[[int, int], None]) -> None:
    EventRecordCallbacks.add_callback(cb)

def register_callback_for_event_wait(cb: Callable[[int, int], None]) -> None:
    EventWaitCallbacks.add_callback(cb)

def register_callback_for_memory_allocation(cb: Callable[[int], None]) -> None:
    MemoryAllocationCallbacks.add_callback(cb)

def register_callback_for_memory_deallocation(cb: Callable[[int], None]) -> None:
    MemoryDeallocationCallbacks.add_callback(cb)

def register_callback_for_stream_creation(cb: Callable[[int], None]) -> None:
    StreamCreationCallbacks.add_callback(cb)

def register_callback_for_device_synchronization(cb: Callable[[], None]) -> None:
    DeviceSynchronizationCallbacks.add_callback(cb)

def register_callback_for_stream_synchronization(cb: Callable[[int], None]) -> None:
    StreamSynchronizationCallbacks.add_callback(cb)

def register_callback_for_event_synchronization(cb: Callable[[int], None]) -> None:
    EventSynchronizationCallbacks.add_callback(cb)
```