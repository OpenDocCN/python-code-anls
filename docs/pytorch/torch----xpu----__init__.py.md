# `.\pytorch\torch\xpu\__init__.py`

```py
# mypy: allow-untyped-defs
"""
This package introduces support for the XPU backend, specifically tailored for
Intel GPU optimization.

This package is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports XPU.
"""
import threading  # 导入多线程支持模块
import traceback  # 导入异常跟踪模块
from functools import lru_cache  # 导入 functools 模块中的 lru_cache 装饰器
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入 PyTorch 模块
import torch._C  # 导入 PyTorch C++ 扩展模块
from .. import device as _device  # 导入相对路径的 device 模块，并将其命名为 _device
from .._utils import _dummy_type, _LazySeedTracker  # 导入相对路径的 _utils 模块中的 _dummy_type 和 _LazySeedTracker
from ._utils import _get_device_index  # 导入当前目录的 _utils 模块中的 _get_device_index 函数
from .streams import Event, Stream  # 导入当前目录的 streams 模块中的 Event 和 Stream 类

_initialized = False  # 初始化状态标志，默认为 False
_tls = threading.local()  # 创建线程局部存储对象
_initialization_lock = threading.Lock()  # 创建初始化锁对象，用于线程同步
_queued_calls: List[Tuple[Callable[[], None], List[str]]] = []  # 延迟调用列表，元素为 (函数, 调用栈信息列表)
_is_in_bad_fork = getattr(torch._C, "_xpu_isInBadFork", lambda: False)  # 获取 torch._C._xpu_isInBadFork 函数，如果不存在则定义为返回 False
_device_t = Union[_device, str, int, None]  # 设备类型的联合类型，可以是 _device、str、int 或 None
_lazy_seed_tracker = _LazySeedTracker()  # 创建 _LazySeedTracker 实例对象
default_generators: Tuple[torch._C.Generator] = ()  # 默认生成器元组，类型为 torch._C.Generator

def _is_compiled() -> bool:
    """
    Return true if compile with XPU support.
    """
    return torch._C._has_xpu  # 检查是否编译了 XPU 支持

if _is_compiled():
    _XpuDeviceProperties = torch._C._XpuDeviceProperties  # XPU 设备属性类
    _exchange_device = torch._C._xpu_exchangeDevice  # XPU 设备交换函数
    _maybe_exchange_device = torch._C._xpu_maybeExchangeDevice  # 可能的 XPU 设备交换函数
else:
    # Define dummy if PyTorch was compiled without XPU
    _XpuDeviceProperties = _dummy_type("_XpuDeviceProperties")  # 如果 PyTorch 未编译 XPU 支持，定义一个 _XpuDeviceProperties 的虚拟类型
    def _exchange_device(device: int) -> int:
        raise NotImplementedError("PyTorch was compiled without XPU support")  # 如果 PyTorch 未编译 XPU 支持，定义一个 _exchange_device 函数抛出未实现异常
    def _maybe_exchange_device(device: int) -> int:
        raise NotImplementedError("PyTorch was compiled without XPU support")  # 如果 PyTorch 未编译 XPU 支持，定义一个 _maybe_exchange_device 函数抛出未实现异常

@lru_cache(maxsize=1)
def device_count() -> int:
    """
    Return the number of XPU device available.
    """
    if not _is_compiled():
        return 0  # 如果未编译 XPU 支持，返回 0
    return torch._C._xpu_getDeviceCount()  # 返回 XPU 设备的数量

def is_available() -> bool:
    """
    Return a bool indicating if XPU is currently available.
    """
    return device_count() > 0  # 返回 XPU 是否可用的布尔值

def is_bf16_supported():
    """
    Return a bool indicating if the current XPU device supports dtype bfloat16.
    """
    return True  # 假设当前 XPU 设备支持 bfloat16，始终返回 True

def is_initialized():
    """
    Return whether PyTorch's XPU state has been initialized.
    """
    return _initialized and not _is_in_bad_fork()  # 返回 PyTorch 的 XPU 状态是否已初始化，并且不处于坏的分叉状态

def _lazy_call(callable, **kwargs):
    """
    Lazy function call depending on PyTorch's XPU initialization state.
    """
    if is_initialized():
        callable()  # 如果已初始化，直接调用传入的 callable 函数
    else:
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())  # 如果设置了 seed_all 参数，将调用信息和 callable 加入到延迟队列中
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())  # 如果设置了 seed 参数，将调用信息和 callable 加入到延迟队列中
        else:
            _queued_calls.append((callable, traceback.format_stack()))  # 否则，将调用信息和 callable 加入到延迟队列中

def init():
    """
    Initialize PyTorch's XPU state.
    """
    global _initialized, _tls, _queued_calls
    with _initialization_lock:
        if _initialized:
            return  # 如果已经初始化，直接返回
        _initialized = True  # 标记为已初始化
        for lazy_call, stack in _queued_calls:
            lazy_call()  # 执行所有延迟调用
        _queued_calls = []  # 清空延迟调用队列
    This is a Python API about lazy initialization that avoids initializing
    XPU until the first time it is accessed. Does nothing if the XPU state is
    already initialized.
    """
    # 调用 lazy_init 函数，实现延迟初始化，确保只在第一次访问时才初始化 XPU 状态
    _lazy_init()
# 定义一个函数用于延迟初始化操作
def _lazy_init():
    # 声明全局变量
    global _initialized, _queued_calls
    # 如果已经初始化或者线程本地存储中已经存在"is_initializing"属性，则直接返回
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    # 使用初始化锁进行线程安全的初始化操作
    with _initialization_lock:
        # 通过获取全局解释器锁（GIL）保护此测试，再次检查是否已经初始化
        if is_initialized():
            return
        # 如果处于不良进程分叉状态，则立即停止初始化并抛出运行时错误
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize XPU in forked subprocess. To use XPU with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        # 如果 Torch 没有启用 XPU 编译，则抛出断言错误
        if not _is_compiled():
            raise AssertionError("Torch not compiled with XPU enabled")
        # 调用 Torch C++ 接口进行 XPU 后端初始化并检测不良分叉处理
        torch._C._xpu_init()
        # 一些延迟调用可能会重新调用 _lazy_init(); 在这种情况下，我们只需返回而不进行初始化
        _tls.is_initializing = True

        # 从 _lazy_seed_tracker 获取所有调用列表并将其加入 _queued_calls
        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            # 对于 _queued_calls 中的每个调用和原始回溯信息
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    # 捕获异常并抛出带有详细信息的新异常
                    msg = (
                        f"XPU call failed lazily at initialization with error: {str(e)}\n\n"
                        f"XPU call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise Exception(msg) from e  # noqa: TRY002
        finally:
            # 最终清除 is_initializing 属性
            delattr(_tls, "is_initializing")
        # 设置 _initialized 标志为 True，表示初始化完成
        _initialized = True


# 定义一个内部类 _DeviceGuard，用作设备切换的上下文管理器
class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    # 进入上下文时调用的方法
    def __enter__(self):
        # 切换设备至指定索引并记录之前的设备索引
        self.prev_idx = torch.xpu._exchange_device(self.idx)

    # 退出上下文时调用的方法
    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 恢复之前的设备索引状态并返回 False 表示不处理任何异常
        self.idx = torch.xpu._maybe_exchange_device(self.prev_idx)
        return False


# 定义一个 device 类，用作设备切换的上下文管理器
class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int or str): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        # 获取设备索引并初始化上一个设备索引
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    # 进入上下文时调用的方法
    def __enter__(self):
        # 切换设备至指定索引并记录之前的设备索引
        self.prev_idx = torch.xpu._exchange_device(self.idx)

    # 退出上下文时调用的方法
    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 恢复之前的设备索引状态并返回 False 表示不处理任何异常
        self.idx = torch.xpu._maybe_exchange_device(self.prev_idx)
        return False


# 定义一个 device_of 类，继承自 device，用作将当前设备切换为给定对象的设备上下文管理器
class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a XPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """
    # 定义一个初始化方法，接受一个参数 obj
    def __init__(self, obj):
        # 如果 obj 是在 XPU 上，则获取其设备编号，否则设为 -1
        idx = obj.get_device() if obj.is_xpu else -1
        # 调用父类的初始化方法，传入设备编号作为参数
        super().__init__(idx)
def set_device(device: _device_t) -> None:
    r"""Set the current device.

    Args:
        device (torch.device or int or str): selected device. This function is a
            no-op if this argument is negative.
    """
    # 确保 torch 初始化完毕
    _lazy_init()
    # 获取设备的索引
    device = _get_device_index(device)
    # 如果设备索引有效，则设置当前设备
    if device >= 0:
        torch._C._xpu_setDevice(device)


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to
            return the name. This function is a no-op if this argument is a
            negative integer. It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    # 获取设备的属性并返回设备名称
    return get_device_properties(device).name


@lru_cache(None)
def get_device_capability(device: Optional[_device_t] = None) -> Dict[str, Any]:
    r"""Get the xpu capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to
            return the device capability. This function is a no-op if this
            argument is a negative integer. It uses the current device, given by
            :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        Dict[str, Any]: the xpu capability dictionary of the device
    """
    # 获取设备的属性，并以字典形式返回除私有属性外的所有属性
    props = get_device_properties(device)
    return {
        prop: getattr(props, prop) for prop in dir(props) if not prop.startswith("__")
    }


def get_device_properties(device: Optional[_device_t] = None) -> _XpuDeviceProperties:
    r"""Get the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _XpuDeviceProperties: the properties of the device
    """
    # 确保 torch 初始化完毕
    _lazy_init()
    # 获取设备的索引
    device = _get_device_index(device, optional=True)
    # 检查设备索引是否有效
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device index")
    # 返回设备的属性
    return _get_device_properties(device)  # type: ignore[name-defined]  # noqa: F821


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    # 确保 torch 初始化完毕
    _lazy_init()
    # 返回当前选中设备的索引
    return torch._C._xpu_getDevice()


def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int or str): selected device.
    """
    # 如果设备是字符串类型，则转换为 torch.device 对象
    if isinstance(device, str):
        device = torch.device(device)
    # 如果设备是整数类型，则创建 xpu 设备对象
    elif isinstance(device, int):
        device = torch.device("xpu", device)
    # 返回设备对象
    return device


class StreamContext:
    r"""Context-manager that selects a given stream.

    All XPU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    """
    .. note:: Streams are per-device.
    """
    cur_stream: Optional["torch.xpu.Stream"]
    
    def __init__(self, stream: Optional["torch.xpu.Stream"]):
        # 初始化函数，接受一个流对象作为参数
        self.stream = stream
        # 获取当前设备的索引
        self.idx = _get_device_index(None, True)
        if self.idx is None:
            self.idx = -1
    
    def __enter__(self):
        # 进入上下文管理器时执行的方法
        cur_stream = self.stream
        # 如果当前流为空或者设备索引为-1，则直接返回
        if cur_stream is None or self.idx == -1:
            return
        # 保存当前设备上的流对象
        self.src_prev_stream = torch.xpu.current_stream(None)
    
        # 如果当前流不在当前设备上，则设置当前设备上的流
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                # 设置目标设备上的当前流
                self.dst_prev_stream = torch.xpu.current_stream(cur_stream.device)
        # 将当前流设置为传入的流对象
        torch.xpu.set_stream(cur_stream)
    
    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 离开上下文管理器时执行的方法
        cur_stream = self.stream
        # 如果当前流为空或者设备索引为-1，则直接返回
        if cur_stream is None or self.idx == -1:
            return
    
        # 如果之前保存的流对象不在当前设备上，则恢复该流对象
        if self.src_prev_stream.device != cur_stream.device:
            torch.xpu.set_stream(self.dst_prev_stream)
        # 恢复当前设备上的流对象
        torch.xpu.set_stream(self.src_prev_stream)
def stream(stream: Optional["torch.xpu.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """
    return StreamContext(stream)

def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and device type

    Args: stream_id (int): not visible to the user, used to assigned to the specific stream.
          device_index (int): selected device index.
          device_type (int): selected device type.
    """
    torch._C._xpu_setStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )

def set_stream(stream: Stream):
    r"""Set the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _lazy_init()  # 懒加载初始化 Torch XPU 模块
    _set_stream_by_id(
        stream_id=stream.stream_id,  # 设置流的 ID
        device_index=stream.device_index,  # 设置设备索引
        device_type=stream.device_type,  # 设置设备类型
    )

def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()  # 懒加载初始化 Torch XPU 模块
    streamdata = torch._C._xpu_getCurrentStream(
        _get_device_index(device, optional=True)  # 获取设备的索引
    )
    return Stream(
        stream_id=streamdata[0],  # 当前流的 ID
        device_index=streamdata[1],  # 当前设备索引
        device_type=streamdata[2]  # 当前设备类型
    )

def synchronize(device: _device_t = None) -> None:
    r"""Wait for all kernels in all streams on a XPU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()  # 懒加载初始化 Torch XPU 模块
    device = _get_device_index(device, optional=True)  # 获取设备的索引
    return torch._C._xpu_synchronize(device)  # 同步设备上所有流中的所有内核

def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other XPU application.

    .. note::
        :func:`~torch.xpu.empty_cache` doesn't increase the amount of XPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of XPU memory in certain cases.
    """
    if is_initialized():  # 如果 Torch XPU 模块已经初始化
        torch._C._xpu_emptyCache()  # 释放当前未使用的缓存内存

def _get_generator(device: torch.device) -> torch._C.Generator:
    # 获取给定设备的 XPU 生成器对象

    # 获取设备的索引
    idx = device.index
    
    # 如果设备索引为 None，则使用当前设备的索引
    if idx is None:
        idx = current_device()
    
    # 返回指定设备索引对应的 XPU 默认生成器对象
    return torch.xpu.default_generators[idx]
def _set_rng_state_offset(
    offset: int, device: Union[int, str, torch.device] = "xpu"
) -> None:
    r"""Set the random number generator state offset of the specified GPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).
    """
    # 确定最终使用的设备
    final_device = _get_device(device)

    # 定义回调函数，用于设置指定设备的随机数生成器状态偏移量
    def cb():
        # 获取指定设备的默认随机数生成器
        default_generator = _get_generator(final_device)
        # 设置随机数生成器的偏移量
        default_generator.set_offset(offset)

    # 调用延迟执行函数来执行回调函数
    _lazy_call(cb)


def _get_rng_state_offset(device: Union[int, str, torch.device] = "xpu") -> int:
    r"""Return the random number generator state offset of the specified GPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    # 确保 XPU 已经初始化
    _lazy_init()
    # 确定最终使用的设备
    final_device = _get_device(device)
    # 获取指定设备的默认随机数生成器，并返回其偏移量
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()
```