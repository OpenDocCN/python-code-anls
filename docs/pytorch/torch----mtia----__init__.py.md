# `.\pytorch\torch\mtia\__init__.py`

```
# mypy: allow-untyped-defs
r"""
This package enables an interface for accessing MTIA backend in python
"""

import threading  # 导入线程模块，用于多线程支持
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入PyTorch模块

from torch.types import Device  # 从torch.types中导入Device类型

from .. import device as _device  # 导入相对路径下的device模块，并重命名为_device
from .._utils import _dummy_type, _LazySeedTracker, classproperty  # 从相对路径下的_utils模块导入几个函数和类
from ._utils import _get_device_index  # 从当前路径下的_utils模块导入_get_device_index函数

_device_t = Union[_device, str, int, None]  # 定义_device_t类型，可以是_device、str、int或None类型

# torch.mtia.Event/Stream is alias of torch.Event/Stream
Event = torch.Event  # 定义Event为torch.Event的别名
Stream = torch.Stream  # 定义Stream为torch.Stream的别名

_initialized = False  # 初始化_initialized标志为False，表示未初始化
_queued_calls: List[
    Tuple[Callable[[], None], List[str]]
] = []  # 定义_queued_calls为一个空的列表，用于存储待调用的函数和参数列表，格式为[(函数, 参数列表)]
_tls = threading.local()  # 创建一个线程本地存储对象_tls，用于存储线程本地数据
_initialization_lock = threading.Lock()  # 创建一个线程锁_initialization_lock，用于多线程初始化时的同步
_lazy_seed_tracker = _LazySeedTracker()  # 创建一个_LazySeedTracker对象_lazy_seed_tracker，用于惰性种子跟踪


def init():
    _lazy_init()  # 调用_lazy_init函数，进行初始化


def is_initialized():
    r"""Return whether PyTorch's MTIA state has been initialized."""
    return _initialized and not _is_in_bad_fork()  # 返回_initialized状态和不处于坏分支中的状态


def _is_in_bad_fork() -> bool:
    return torch._C._mtia_isInBadFork()  # 调用torch._C._mtia_isInBadFork()函数，检查是否处于坏分支


def _lazy_init() -> None:
    global _initialized, _queued_calls  # 声明使用全局变量_initialized和_queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):  # 如果已经初始化或者_tls对象中有"is_initializing"属性
        return  # 直接返回，不执行初始化操作
    with _initialization_lock:
        # 使用双重检查锁定，确保只有一个线程进行初始化操作。由于上面的测试已经受GIL保护，所以这是安全的。
        # 内部的测试是为了处理一个线程在另一个正在进行初始化的线程上被阻塞的情况；当它们获取锁时，
        # 它们会发现没有什么需要做的了。
        
        if is_initialized():
            return
        # 如果已经初始化完成，则直接返回，避免重复初始化
        
        # 在我们仍然拥有GIL保证的时候，阻止其他线程立即进入_lazy_init非常重要，
        # 因为下面的某些C调用会释放GIL。
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize MTIA in forked subprocess. To use MTIA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        # 如果在错误的子进程中，禁止重新初始化MTIA
        
        if not _is_compiled():
            raise AssertionError("Torch not compiled with MTIA enabled")
        # 如果Torch没有启用MTIA编译，则抛出断言错误
        
        torch._C._mtia_init()
        # 初始化MTIA
        
        # 一些排队的调用可能会重新调用_lazy_init(); 
        # 在这种情况下我们需要仅返回而不进行初始化。
        # 但是，我们不能让任何*其他*线程进入！
        _tls.is_initializing = True
        # 设置TLS标志表示正在初始化
        
        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)
        # 获取懒惰种子追踪器中的调用，并将其添加到队列中
        
        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"MTIA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"MTIA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredMtiaCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        # 在最终处理中，删除TLS的is_initializing属性
        
        _initialized = True
class DeferredMtiaCallError(Exception):
    pass



def _is_compiled() -> bool:
    r"""Return true if compiled with MTIA support."""
    # 检查是否使用 MTIA 支持进行编译
    return torch._C._mtia_isBuilt()



def is_available() -> bool:
    r"""Return true if MTIA device is available"""
    if not _is_compiled():
        return False
    # MTIA 需要先初始化设备以确定是否有可用设备
    return device_count() > 0



def synchronize(device: Optional[_device_t] = None) -> None:
    r"""Waits for all jobs in all streams on a MTIA device to complete."""
    # 使用上下文管理器设置指定设备的同步操作
    with torch.mtia.device(device):
        return torch._C._mtia_deviceSynchronize()



def device_count() -> int:
    r"""Return the number of MTIA devices available."""
    # 返回可用的 MTIA 设备数量
    return torch._C._accelerator_hooks_device_count()



def current_device() -> int:
    r"""Return the index of a currently selected device."""
    # 返回当前选择设备的索引
    return torch._C._accelerator_hooks_get_current_device()



def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """
    # 返回给定设备的当前选择的 Stream
    return torch._C._mtia_getCurrentStream(_get_device_index(device, optional=True))



def default_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """
    # 返回给定设备的默认 Stream
    return torch._C._mtia_getDefaultStream(_get_device_index(device, optional=True))



def set_stream(stream: Stream):
    r"""Set the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    # 设置当前的 Stream，建议优先使用上下文管理器中的 stream
    if stream is None:
        return
    torch._C._mtia_setCurrentStream(stream)



def set_device(device: _device_t) -> None:
    r"""Set the current device.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    # 设置当前的设备
    device = _get_device_index(device)
    if device >= 0:
        torch._C._accelerator_hooks_set_current_device(device)



class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """
    # 上下文管理器，用于更改选择的设备
    pass
    # 初始化方法，接受一个设备参数并获取其索引，允许设备参数可选
    def __init__(self, device: Any):
        # 使用_get_device_index函数获取设备的索引，如果设备参数未提供则默认为None
        self.idx = _get_device_index(device, optional=True)
        # 初始化上一个设备索引为-1
        self.prev_idx = -1

    # 进入上下文管理器时调用的方法
    def __enter__(self):
        # 将当前设备索引传递给torch._C._accelerator_hooks_maybe_exchange_device函数，并更新self.prev_idx
        self.prev_idx = torch._C._accelerator_hooks_maybe_exchange_device(self.idx)

    # 退出上下文管理器时调用的方法
    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 将上一个设备索引传递给torch._C._accelerator_hooks_maybe_exchange_device函数，并更新self.idx
        self.idx = torch._C._accelerator_hooks_maybe_exchange_device(self.prev_idx)
        # 始终返回False，以便不会压制任何异常
        return False
# 定义一个上下文管理器，用于选择给定的流

class StreamContext:
    r"""Context-manager that selects a given stream.

    All MTIA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["torch.mtia.Stream"]

    def __init__(self, stream: Optional["torch.mtia.Stream"]):
        # 初始化函数，接收一个流作为参数
        self.stream = stream
        # 获取当前设备的索引
        self.idx = _get_device_index(None, True)
        # 如果不是脚本模式，检查当前设备索引是否为None
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        # 设置源前一个流和目标前一个流为默认流，如果不是脚本模式
        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.mtia.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.mtia.default_stream(None)
        )

    def __enter__(self):
        # 进入上下文管理器时执行的操作

        # 本地cur_stream变量用于类型细化
        cur_stream = self.stream
        # 如果流为None或MTIA设备不可用，则直接返回
        if cur_stream is None or self.idx == -1:
            return
        # 保存当前源流到src_prev_stream
        self.src_prev_stream = torch.mtia.current_stream(None)

        # 如果源流不在当前设备上，则将当前流设置为指定设备上的流
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                # 设置目标前一个流为当前设备上的流
                self.dst_prev_stream = torch.mtia.current_stream(cur_stream.device)
        # 设置当前MTIA流为指定的流
        torch.mtia.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 退出上下文管理器时执行的操作

        # 本地cur_stream变量用于类型细化
        cur_stream = self.stream
        # 如果流为None或MTIA设备不可用，则直接返回
        if cur_stream is None or self.idx == -1:
            return

        # 如果源流不在当前设备上，则设置目标前一个流为当前设备上的流
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.mtia.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        # 恢复源流为当前MTIA流
        torch.mtia.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch.mtia.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    ..Note:: In eager mode stream is of type Stream class while in JIT it doesn't support torch.mtia.stream
    """
    # 返回一个StreamContext对象，用于管理指定的流
    return StreamContext(stream)


__all__ = [
    "init",
    "is_available",
    "is_initialized",
    "synchronize",
    "device_count",
    "current_device",
    "current_stream",
    "default_stream",
    "set_device",
    "set_stream",
    "stream",
    "device",
]
```