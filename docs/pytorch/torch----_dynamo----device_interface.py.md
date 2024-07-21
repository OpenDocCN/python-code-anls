# `.\pytorch\torch\_dynamo\device_interface.py`

```
# mypy: allow-untyped-defs
# 导入 inspect 模块，用于获取对象信息
import inspect
# 导入类型相关模块
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

# 导入 PyTorch 模块
import torch
# 导入 PyTorch 内部流相关模块
from torch._streambase import _EventBase, _StreamBase

# 声明一个可选的函数 get_cuda_stream，用于获取当前 CUDA 流
get_cuda_stream: Optional[Callable[[int], int]]
# 如果 PyTorch 的 CUDA 模块已编译
if torch.cuda._is_compiled():
    # 导入获取当前原始 CUDA 流的函数
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
else:
    # 否则将 get_cuda_stream 设置为 None
    get_cuda_stream = None

# _device_t 可以是 torch.device, str, int, None 中的一种
_device_t = Union[torch.device, str, int, None]

# 缓存主进程中的设备属性，但在工作进程中使用
caching_worker_device_properties: Dict[str, Any] = {}
# 缓存当前设备的信息，键为设备名，值为设备编号
caching_worker_current_devices: Dict[str, int] = {}


class DeviceInterfaceMeta(type):
    def __new__(metacls, *args, **kwargs):
        # 获取类成员字典
        class_member = args[2]
        # 如果类成员中包含 "Event"
        if "Event" in class_member:
            # 断言 Event 是 _EventBase 的子类
            assert inspect.isclass(class_member["Event"]) and issubclass(
                class_member["Event"], _EventBase
            ), "DeviceInterface member Event should be inherit from _EventBase"
        # 如果类成员中包含 "Stream"
        if "Stream" in class_member:
            # 断言 Stream 是 _StreamBase 的子类
            assert inspect.isclass(class_member["Stream"]) and issubclass(
                class_member["Stream"], _StreamBase
            ), "DeviceInterface member Stream should be inherit from _StreamBase"
        # 返回创建的类
        return super().__new__(metacls, *args, **kwargs)


class DeviceInterface(metaclass=DeviceInterfaceMeta):
    """
    This is a simple device runtime interface for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """

    class device:
        def __new__(cls, device: _device_t):
            raise NotImplementedError

    class Worker:
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """

        @staticmethod
        # 设置设备编号
        def set_device(device: int):
            raise NotImplementedError

        @staticmethod
        # 获取当前设备编号
        def current_device() -> int:
            raise NotImplementedError

        @staticmethod
        # 获取设备属性信息
        def get_device_properties(device: _device_t = None):
            raise NotImplementedError

    @staticmethod
    # 获取当前设备编号
    def current_device():
        raise NotImplementedError

    @staticmethod
    # 设置设备
    def set_device(device: _device_t):
        raise NotImplementedError

    @staticmethod
    # 可能交换设备
    def maybe_exchange_device(device: int) -> int:
        raise NotImplementedError

    @staticmethod
    # 交换设备
    def exchange_device(device: int) -> int:
        raise NotImplementedError

    @staticmethod
    # 获取设备数量
    def device_count():
        raise NotImplementedError

    @staticmethod
    # 检查设备是否可用
    def is_available() -> bool:
        raise NotImplementedError

    @staticmethod
    # 设置流
    def stream(stream: torch.Stream):
        raise NotImplementedError

    @staticmethod
    # 获取当前流
    def current_stream():
        raise NotImplementedError

    @staticmethod
    # 设置流对象，接受一个 torch.Stream 类型的参数
    def set_stream(stream: torch.Stream):
        # 抛出未实现错误，提示该方法尚未被具体实现
        raise NotImplementedError
    
    # 根据流的ID、设备索引和设备类型设置流对象
    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int):
        # 抛出未实现错误，提示该方法尚未被具体实现
        raise NotImplementedError
    
    # 获取原始流对象
    @staticmethod
    def get_raw_stream():
        # 抛出未实现错误，提示该方法尚未被具体实现
        raise NotImplementedError
    
    # 同步设备上的操作，接受一个 _device_t 类型的设备参数，如果未提供则默认为 None
    @staticmethod
    def synchronize(device: _device_t = None):
        # 抛出未实现错误，提示该方法尚未被具体实现
        raise NotImplementedError
    
    # 获取设备的属性信息，接受一个 _device_t 类型的设备参数，如果未提供则默认为 None
    @staticmethod
    def get_device_properties(device: _device_t = None):
        # 抛出未实现错误，提示该方法尚未被具体实现
        raise NotImplementedError
    
    # 获取设备的计算能力信息，接受一个 _device_t 类型的设备参数，如果未提供则默认为 None
    @staticmethod
    def get_compute_capability(device: _device_t = None):
        # 抛出未实现错误，提示该方法尚未被具体实现
        raise NotImplementedError
# 设备保护上下文管理器类，用于设备切换
class DeviceGuard:
    """
    This class provides a context manager for device switching. This is a stripped
    down version of torch.{device_name}.device.

    The context manager changes the current device to the given device index
    on entering the context and restores the original device on exiting.
    The device is switched using the provided device interface.
    """

    def __init__(self, device_interface: Type[DeviceInterface], index: Optional[int]):
        # 初始化函数，接收设备接口类型和设备索引作为参数
        self.device_interface = device_interface
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        # 进入上下文管理器时执行的操作
        if self.idx is not None:
            # 使用设备接口切换到指定设备，并记录当前设备索引
            self.prev_idx = self.device_interface.exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 离开上下文管理器时执行的操作
        if self.idx is not None:
            # 恢复原先的设备索引
            self.idx = self.device_interface.maybe_exchange_device(self.prev_idx)
        return False


class CudaInterface(DeviceInterface):
    # CUDA 设备接口，继承自 DeviceInterface

    device = torch.cuda.device

    # 将 Event 和 Stream 类注册到后端接口中
    # 确保 Event 和 Stream 类实现并继承自 _EventBase 和 _StreamBase
    Event = torch.cuda.Event
    Stream = torch.cuda.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            # 设置当前工作线程的 CUDA 设备
            caching_worker_current_devices["cuda"] = device

        @staticmethod
        def current_device() -> int:
            # 获取当前工作线程的 CUDA 设备索引
            if "cuda" in caching_worker_current_devices:
                return caching_worker_current_devices["cuda"]
            return torch.cuda.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            # 获取指定设备的属性信息
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "cuda"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = CudaInterface.Worker.current_device()

            # 如果缓存中没有 CUDA 设备属性信息，则获取并缓存
            if "cuda" not in caching_worker_device_properties:
                device_prop = [
                    torch.cuda.get_device_properties(i)
                    for i in range(torch.cuda.device_count())
                ]
                caching_worker_device_properties["cuda"] = device_prop

            return caching_worker_device_properties["cuda"][device]

    current_device = staticmethod(torch.cuda.current_device)
    set_device = staticmethod(torch.cuda.set_device)
    device_count = staticmethod(torch.cuda.device_count)
    stream = staticmethod(torch.cuda.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.cuda.current_stream)
    set_stream = staticmethod(torch.cuda.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.cuda._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.cuda.synchronize)
    # 将 torch.cuda.get_device_properties 方法作为静态方法 get_device_properties 赋给当前类
    get_device_properties = staticmethod(torch.cuda.get_device_properties)  # type: ignore[assignment]
    # 将 get_cuda_stream 方法作为静态方法 get_raw_stream 赋给当前类
    get_raw_stream = staticmethod(get_cuda_stream)  # type: ignore[arg-type]
    # 将 torch.cuda._exchange_device 方法作为静态方法 exchange_device 赋给当前类
    exchange_device = staticmethod(torch.cuda._exchange_device)  # type: ignore[arg-type]
    # 将 torch.cuda._maybe_exchange_device 方法作为静态方法 maybe_exchange_device 赋给当前类
    maybe_exchange_device = staticmethod(torch.cuda._maybe_exchange_device)  # type: ignore[arg-type]

    # 可以被 @patch 装饰器模拟替换。
    @staticmethod
    # 返回当前系统是否支持 CUDA 加速
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    # 获取指定 CUDA 设备的计算能力版本
    def get_compute_capability(device: _device_t = None):
        # 如果当前系统不是 HIP 架构
        if torch.version.hip is None:
            # 获取指定设备的主要和次要计算能力版本号
            major, min = torch.cuda.get_device_capability(device)
            # 返回计算能力版本号的乘积
            return major * 10 + min
        else:
            # 获取指定 CUDA 设备的属性，并提取 GCN 架构名称的主要版本号
            return torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
# 定义一个可选的回调函数类型，用于获取 XPU 设备的流
get_xpu_stream: Optional[Callable[[int], int]]

# 如果 torch.xpu._is_compiled() 返回 True，则导入相关的 XPU 流处理函数
if torch.xpu._is_compiled():
    from torch._C import _xpu_getCurrentRawStream as get_xpu_stream
else:
    # 否则将 get_xpu_stream 设置为 None
    get_xpu_stream = None

# 定义 XpuInterface 类，继承自 DeviceInterface
class XpuInterface(DeviceInterface):
    # 设备类型为 torch.xpu.device
    device = torch.xpu.device
    # 定义 Event 类型为 torch.xpu.Event
    Event = torch.xpu.Event
    # 定义 Stream 类型为 torch.xpu.Stream
    Stream = torch.xpu.Stream

    # 定义内部 Worker 类
    class Worker:
        # 静态方法：设置当前设备
        @staticmethod
        def set_device(device: int):
            # 将 "xpu" 设备的当前设备设置为指定的 device
            caching_worker_current_devices["xpu"] = device

        # 静态方法：获取当前设备
        @staticmethod
        def current_device() -> int:
            # 如果 "xpu" 在 caching_worker_current_devices 中，返回其值
            if "xpu" in caching_worker_current_devices:
                return caching_worker_current_devices["xpu"]
            # 否则返回当前 XPU 设备的索引
            return torch.xpu.current_device()

        # 静态方法：获取设备的属性信息
        @staticmethod
        def get_device_properties(device: _device_t = None):
            # 如果指定了 device 参数
            if device is not None:
                # 如果 device 是字符串类型，转换为 torch.device，并断言设备类型为 "xpu"
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "xpu"
                # 如果 device 是 torch.device 类型，获取其索引
                if isinstance(device, torch.device):
                    device = device.index
            # 如果未指定 device，获取当前 XPU 设备的索引
            if device is None:
                device = XpuInterface.Worker.current_device()

            # 如果 "xpu" 不在缓存中，获取所有 XPU 设备的属性信息
            if "xpu" not in caching_worker_device_properties:
                device_prop = [
                    torch.xpu.get_device_properties(i)
                    for i in range(torch.xpu.device_count())
                ]
                caching_worker_device_properties["xpu"] = device_prop

            # 返回指定设备的属性信息
            return caching_worker_device_properties["xpu"][device]

    # 静态方法：获取当前 XPU 设备索引
    current_device = staticmethod(torch.xpu.current_device)
    # 静态方法：设置当前设备
    set_device = staticmethod(torch.xpu.set_device)
    # 静态方法：获取 XPU 设备数量
    device_count = staticmethod(torch.xpu.device_count)
    # 静态方法：获取 XPU 流
    stream = staticmethod(torch.xpu.stream)  # type: ignore[assignment]
    # 静态方法：获取当前 XPU 流
    current_stream = staticmethod(torch.xpu.current_stream)
    # 静态方法：设置 XPU 流
    set_stream = staticmethod(torch.xpu.set_stream)  # type: ignore[assignment]
    # 静态方法：通过 ID 设置 XPU 流
    _set_stream_by_id = staticmethod(torch.xpu._set_stream_by_id)  # type: ignore[assignment]
    # 静态方法：同步 XPU 设备
    synchronize = staticmethod(torch.xpu.synchronize)
    # 静态方法：获取 XPU 设备的属性信息
    get_device_properties = staticmethod(torch.xpu.get_device_properties)  # type: ignore[assignment]
    # 静态方法：获取原始 XPU 流
    get_raw_stream = staticmethod(get_xpu_stream)  # type: ignore[arg-type]
    # 静态方法：交换 XPU 设备
    exchange_device = staticmethod(torch.xpu._exchange_device)  # type: ignore[arg-type]
    # 静态方法：可能交换 XPU 设备
    maybe_exchange_device = staticmethod(torch.xpu._maybe_exchange_device)  # type: ignore[arg-type]

    # 静态方法：检查当前是否有可用的 XPU 设备
    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    # 静态方法：获取指定 XPU 设备的计算能力
    @staticmethod
    def get_compute_capability(device: _device_t = None):
        # 获取指定 XPU 设备的计算能力
        cc = torch.xpu.get_device_capability(device)
        return cc


# 定义一个空字典，用于存储设备接口类型
device_interfaces: Dict[str, Type[DeviceInterface]] = {}
# 设备是否已初始化的标志
_device_initialized = False


# 注册设备接口类型到指定设备
def register_interface_for_device(
    device: Union[str, torch.device], device_interface: Type[DeviceInterface]
):
    # 如果 device 是 torch.device 类型，转换为字符串类型
    if isinstance(device, torch.device):
        device = str(device)
    # 将指定设备的接口类型注册到 device_interfaces 字典中
    device_interfaces[device] = device_interface
# 根据输入的设备名称或torch.device对象返回相应的设备接口类型
def get_interface_for_device(device: Union[str, torch.device]) -> Type[DeviceInterface]:
    # 如果device是torch.device对象，则转换为对应的字符串表示
    if isinstance(device, torch.device):
        device = str(device)
    
    # 如果设备注册未初始化，则初始化设备注册表
    if not _device_initialized:
        init_device_reg()
    
    # 如果设备在注册的设备接口字典中，则返回对应的设备接口类型
    if device in device_interfaces:
        return device_interfaces[device]
    
    # 如果设备未注册，则抛出NotImplementedError异常
    raise NotImplementedError(f"No interface for device {device}")


# 返回已注册设备接口的迭代器，每个元素为设备名称和对应的设备接口类型
def get_registered_device_interfaces() -> Iterable[Tuple[str, Type[DeviceInterface]]]:
    # 如果设备注册未初始化，则初始化设备注册表
    if not _device_initialized:
        init_device_reg()
    
    # 返回设备接口字典的items视图，每个元素是设备名称和对应的设备接口类型
    return device_interfaces.items()


# 初始化设备注册表，注册cuda和xpu设备及其接口类型
def init_device_reg():
    global _device_initialized
    
    # 注册cuda设备和对应的CudaInterface接口类型
    register_interface_for_device("cuda", CudaInterface)
    # 遍历所有cuda设备并注册对应的CudaInterface接口类型
    for i in range(torch.cuda.device_count()):
        register_interface_for_device(f"cuda:{i}", CudaInterface)
    
    # 注册xpu设备和对应的XpuInterface接口类型
    register_interface_for_device("xpu", XpuInterface)
    # 遍历所有xpu设备并注册对应的XpuInterface接口类型
    for i in range(torch.xpu.device_count()):
        register_interface_for_device(f"xpu:{i}", XpuInterface)
    
    # 设备注册初始化完成
    _device_initialized = True
```