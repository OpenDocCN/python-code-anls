# `.\pytorch\torch\cuda\__init__.py`

```
# 设置 mypy: 允许未类型化的定义，用于允许不对定义类型进行类型化的情况
r"""
This package adds support for CUDA tensor types.

It implements the same function as CPU tensors, but they utilize
GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""

# 导入需要的模块
import contextlib  # 上下文管理工具模块
import importlib  # 模块导入工具模块
import os  # 系统操作模块
import sys  # 系统相关模块
import threading  # 多线程支持模块
import traceback  # 异常跟踪模块
import warnings  # 警告模块
from functools import lru_cache  # 缓存装饰器模块
from typing import Any, Callable, cast, List, Optional, Tuple, Union  # 类型提示相关模块

import torch  # PyTorch 主模块
import torch._C  # PyTorch C++ 扩展模块
from torch.types import Device  # 设备类型定义
from .. import device as _device  # 设备相关模块
from .._utils import _dummy_type, _LazySeedTracker, classproperty  # 辅助工具模块
from ._utils import _get_device_index  # 私有工具函数，获取设备索引
from .graphs import (  # 图形处理模块导入
    CUDAGraph,  # CUDA 图模块
    graph,  # 图模块
    graph_pool_handle,  # 图池句柄模块
    is_current_stream_capturing,  # 是否当前流捕获模块
    make_graphed_callables,  # 创建图调用模块
)
from .streams import Event, ExternalStream, Stream  # 流事件模块导入

try:
    from torch._C import _cudart  # type: ignore[attr-defined]
except ImportError:
    _cudart = None  # 如果导入失败，则设为 None

_initialized = False  # 初始化状态标志
_tls = threading.local()  # 线程局部存储对象
_initialization_lock = threading.Lock()  # 初始化锁，用于多线程安全
_queued_calls: List[Tuple[Callable[[], None], List[str]]] = []  # 待调用函数队列，元素为函数及其参数列表
_is_in_bad_fork = getattr(torch._C, "_cuda_isInBadFork", lambda: False)  # 判断是否处于不良分支状态的函数
_device_t = Union[_device, str, int, None]  # 设备类型定义，包括设备对象、字符串、整数或 None

_HAS_PYNVML = False  # 是否存在 pynvml 标志
_PYNVML_ERR = None  # pynvml 错误变量

try:
    try:
        import pynvml  # type: ignore[import]
        _HAS_PYNVML = True  # 导入成功，设置标志为 True
    except ModuleNotFoundError:
        pass  # 如果 pynvml 模块未找到，则忽略异常

    try:
        import amdsmi  # type: ignore[import]
        _HAS_PYNVML = True  # 导入成功，设置标志为 True
    except ModuleNotFoundError:
        pass  # 如果 amdsmi 模块未找到，则忽略异常

except ImportError as err:
    _PYNVML_ERR = err  # 如果导入出错，记录错误信息

_lazy_seed_tracker = _LazySeedTracker()  # 懒加载种子跟踪器对象

# 如果 PyTorch 编译时没有 CUDA 支持，则定义虚拟的 _CudaDeviceProperties 类型
if hasattr(torch._C, "_CudaDeviceProperties"):
    _CudaDeviceProperties = torch._C._CudaDeviceProperties  # 如果有 _CudaDeviceProperties 类型，则使用之
else:
    _CudaDeviceProperties = _dummy_type("_CudaDeviceProperties")  # 否则使用虚拟类型

# 如果 PyTorch 编译时有 _cuda_exchangeDevice 函数，则使用之
if hasattr(torch._C, "_cuda_exchangeDevice"):
    _exchange_device = torch._C._cuda_exchangeDevice  # 获取 _cuda_exchangeDevice 函数
else:

    def _exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without CUDA support")  # 若无 _cuda_exchangeDevice 函数，抛出运行时错误

# 如果 PyTorch 编译时有 _cuda_maybeExchangeDevice 函数，则使用之
if hasattr(torch._C, "_cuda_maybeExchangeDevice"):
    _maybe_exchange_device = torch._C._cuda_maybeExchangeDevice  # 获取 _cuda_maybeExchangeDevice 函数
else:

    def _maybe_exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without CUDA support")  # 若无 _cuda_maybeExchangeDevice 函数，抛出运行时错误

has_half: bool = True  # 是否支持半精度浮点数标志，默认为 True
has_magma: bool = torch._C._has_magma  # 是否支持 magma 库标志

default_generators: Tuple[torch._C.Generator] = ()  # 默认生成器，类型为元组，初始为空

def _is_compiled() -> bool:
    r"""Return true if compile with CUDA support."""
    # 检查 torch._C 模块是否具有 _cuda_getDeviceCount 属性，并返回结果
    return hasattr(torch._C, "_cuda_getDeviceCount")
# 检查环境变量 'PYTORCH_NVML_BASED_CUDA_CHECK' 是否设置为 "1"，返回相应的布尔值
def _nvml_based_avail() -> bool:
    return os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"


# 返回一个布尔值，指示当前 CUDA 是否可用
def is_available() -> bool:
    # 如果未编译完成，则返回 False
    if not _is_compiled():
        return False
    # 如果用户设置了环境变量请求 NVML 基础的 CUDA 可用性检查
    if _nvml_based_avail():
        # 当 NVML 发现/初始化失败时，这个检查会回退到默认的 CUDA Runtime API 检查 (`cudaGetDeviceCount`)
        return device_count() > 0
    else:
        # 默认的可用性检查不会抛出异常，如果驱动缺失或无法初始化则返回 0
        return torch._C._cuda_getDeviceCount() > 0


# 返回一个布尔值，指示当前 CUDA/ROCm 设备是否支持 dtype bfloat16
def is_bf16_supported(including_emulation: bool = True):
    # 如果是 ROCm 环境，直接返回 True，不需要检查 ROCM_VERSION，因为支持 AMD GPU 架构
    if torch.version.hip:
        return True

    device = torch.cuda.current_device()

    # 检查 CUDA 版本和设备的 compute capability，这是一个快速检查方式
    cuda_version = torch.version.cuda
    if (
        cuda_version is not None
        and int(cuda_version.split(".")[0]) >= 11
        and torch.cuda.get_device_properties(device).major >= 8
    ):
        return True

    # 如果不包括仿真模式，直接返回 False
    if not including_emulation:
        return False

    # 最后尝试创建一个 bfloat16 设备
    return _check_bf16_tensor_supported(device)


# 使用 lru_cache 缓存，检查给定设备是否支持 bfloat16 张量
@lru_cache(maxsize=16)
def _check_bf16_tensor_supported(device: _device_t):
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return True
    except Exception:
        return False


# 使用 CUDA API 使当前线程休眠指定周期数
def _sleep(cycles):
    torch._C._cuda_sleep(cycles)


# 从 CUDA 版本字符串中提取架构信息
def _extract_arch_version(arch_string: str):
    base = arch_string.split("_")[1]
    if base.endswith("a"):
        base = base[:-1]
    return int(base)


# 错误消息模板，用于指出找到的 GPU 不受支持的情况和建议
incorrect_binary_warn = """
Found GPU%d %s which requires CUDA_VERSION >= %d to
 work properly, but your PyTorch was compiled
 with CUDA_VERSION %d. Please install the correct PyTorch binary
 using instructions from https://pytorch.org
"""

# 警告消息模板，指出找到的 GPU 太旧而不再受 PyTorch 支持
old_gpu_warn = """
Found GPU%d %s which is of cuda capability %d.%d.
PyTorch no longer supports this GPU because it is too old.
The minimum cuda capability supported by this library is %d.%d.
"""
    # 如果当前环境是 CUDA，执行以下检查；在 ROCm 上不执行这个检查
    if torch.version.cuda is not None:
        # 获取当前 CUDA 编译版本
        CUDA_VERSION = torch._C._cuda_getCompiledVersion()
        
        # 遍历所有设备，获取各设备的计算能力
        for d in range(device_count()):
            capability = get_device_capability(d)
            major = capability[0]
            minor = capability[1]
            name = get_device_name(d)
            
            # 计算当前设备的架构号，形如 "major.minor"
            current_arch = major * 10 + minor
            
            # 获取当前 CUDA 架构的最低要求
            min_arch = min(
                (_extract_arch_version(arch) for arch in torch.cuda.get_arch_list()),
                default=35,
            )
            
            # 如果当前设备架构低于最低要求，发出警告
            if current_arch < min_arch:
                warnings.warn(
                    old_gpu_warn
                    % (d, name, major, minor, min_arch // 10, min_arch % 10)
                )
# 定义一个多行字符串，用于在不兼容的设备上显示警告消息模板
incompatible_device_warn = """
{} with CUDA capability sm_{} is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities {}.
If you want to use the {} GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
"""

# 检查当前是否使用 CUDA，如果不是在 ROCm 上，则不进行检查
if torch.version.cuda is None:
    return

# 获取当前环境支持的 GPU 架构列表
arch_list = get_arch_list()

# 如果 GPU 架构列表为空，则直接返回
if len(arch_list) == 0:
    return

# 从支持的架构列表中提取并解析出主要版本号
supported_sm = [_extract_arch_version(arch) for arch in arch_list if "sm_" in arch]

# 遍历所有的设备索引号
for idx in range(device_count()):
    # 获取当前设备的主要和次要计算能力版本号
    cap_major, cap_minor = get_device_capability(idx)

    # 检查当前设备是否在支持的 GPU 架构列表中
    # NVIDIA GPU 计算架构在同一个主版本内向后兼容
    supported = any(sm // 10 == cap_major for sm in supported_sm)

    # 如果当前设备不被支持，则发出警告
    if not supported:
        device_name = get_device_name(idx)
        capability = cap_major * 10 + cap_minor
        warnings.warn(
            incompatible_device_warn.format(
                device_name, capability, " ".join(arch_list), device_name
            )
        )


def is_initialized():
    r"""Return whether PyTorch's CUDA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    # 如果 CUDA 状态已初始化，则直接调用传入的可调用对象
    if is_initialized():
        callable()
    else:
        # 否则，将需要延迟调用的任务信息加入队列
        # 如果需要种子初始化，则记录相应的种子调用任务和调用堆栈信息
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
        else:
            # 否则，只记录任务和调用堆栈信息，不存储实际的 traceback，以避免内存循环引用
            _queued_calls.append((callable, traceback.format_stack()))


# 将 _check_capability 和 _check_cubins 添加到延迟调用任务中
_lazy_call(_check_capability)
_lazy_call(_check_cubins)


class DeferredCudaCallError(Exception):
    # 定义一个异常类，用于表示延迟的 CUDA 调用错误
    pass


# 引入 torch._C 中的 OutOfMemoryError 到当前命名空间中
OutOfMemoryError = torch._C.OutOfMemoryError


def init():
    r"""Initialize PyTorch's CUDA state.

    You may need to call this explicitly if you are interacting with
    PyTorch via its C API, as Python bindings for CUDA functionality
    will not be available until this initialization takes place.
    Ordinary users should not need this, as all of PyTorch's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    """
    # 执行延迟初始化操作
    _lazy_init()


def _lazy_init():
    # 全局延迟初始化函数，用于初始化 PyTorch 的 CUDA 状态
    global _initialized, _queued_calls

    # 如果 CUDA 状态已初始化或正在初始化过程中，则直接返回
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    # 使用 _initialization_lock 来保证初始化代码的线程安全性
    with _initialization_lock:
        # 双重检查锁定模式，检查是否已经初始化，避免重复初始化
        if is_initialized():
            return
        # 在获取 GIL 保护下，防止其他线程在此期间进入 _lazy_init 函数
        if _is_in_bad_fork():
            # 如果在 forked 子进程中尝试重新初始化 CUDA，抛出运行时错误
            raise RuntimeError(
                "Cannot re-initialize CUDA in forked subprocess. To use CUDA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        # 检查 torch._C 模块是否包含 _cuda_getDeviceCount 属性，确保 Torch 编译时启用了 CUDA
        if not hasattr(torch._C, "_cuda_getDeviceCount"):
            raise AssertionError("Torch not compiled with CUDA enabled")
        # 检查 _cudart 变量是否为 None，确认 libcudart 函数是否可用
        if _cudart is None:
            raise AssertionError(
                "libcudart functions unavailable. It looks like you have a broken build?"
            )
        # 设置环境变量 CUDA_MODULE_LOADING 为 "LAZY"，用于延迟加载 CUDA 模块
        if "CUDA_MODULE_LOADING" not in os.environ:
            os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        # 初始化 CUDA，调用底层 torch._C._cuda_init() 函数
        torch._C._cuda_init()
        # 将当前线程标记为正在初始化状态
        _tls.is_initializing = True

        # 从 _lazy_seed_tracker 获取所有待处理的 CUDA 调用，并添加到 _queued_calls 中
        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            # 遍历 _queued_calls 中的所有 CUDA 调用并执行
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    # 捕获 CUDA 调用过程中的异常，并生成详细的错误消息
                    msg = (
                        f"CUDA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"CUDA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    # 抛出 DeferredCudaCallError 异常
                    raise DeferredCudaCallError(msg) from e
        finally:
            # 清除 _tls 中的 is_initializing 属性
            delattr(_tls, "is_initializing")
        # 设置 _initialized 为 True，标记 CUDA 初始化已完成
        _initialized = True
def cudart():
    r"""Retrieves the CUDA runtime API module.


    This function initializes the CUDA runtime environment if it is not already
    initialized and returns the CUDA runtime API module (_cudart). The CUDA
    runtime API module provides access to various CUDA runtime functions.

    Args:
        ``None``

    Returns:
        module: The CUDA runtime API module (_cudart).

    Raises:
        RuntimeError: If CUDA cannot be re-initialized in a forked subprocess.
        AssertionError: If PyTorch is not compiled with CUDA support or if libcudart functions are unavailable.

    Example of CUDA operations with profiling:
        >>> import torch
        >>> from torch.cuda import cudart, check_error
        >>> import os
        >>>
        >>> os.environ['CUDA_PROFILE'] = '1'
        >>>
        >>> def perform_cuda_operations_with_streams():
        >>>     stream = torch.cuda.Stream()
        >>>     with torch.cuda.stream(stream):
        >>>         x = torch.randn(100, 100, device='cuda')
        >>>         y = torch.randn(100, 100, device='cuda')
        >>>         z = torch.mul(x, y)
        >>>     return z
        >>>
        >>> torch.cuda.synchronize()
        >>> print("====== Start nsys profiling ======")
        >>> check_error(cudart().cudaProfilerStart())
        >>> with torch.autograd.profiler.emit_nvtx():
        >>>     result = perform_cuda_operations_with_streams()
        >>>     print("CUDA operations completed.")
        >>> check_error(torch.cuda.cudart().cudaProfilerStop())
        >>> print("====== End nsys profiling ======")

    To run this example and save the profiling information, execute:
        >>> $ nvprof --profile-from-start off --csv --print-summary -o trace_name.prof -f -- python cudart_test.py

    This command profiles the CUDA operations in the provided script and saves
    the profiling information to a file named `trace_name.prof`.
    The `--profile-from-start off` option ensures that profiling starts only
    after the `cudaProfilerStart` call in the script.
    The `--csv` and `--print-summary` options format the profiling output as a
    CSV file and print a summary, respectively.
    The `-o` option specifies the output file name, and the `-f` option forces the
    overwrite of the output file if it already exists.
    """
    # 调用 _lazy_init() 函数以懒加载初始化 CUDA 运行时环境
    _lazy_init()
    # 返回 CUDA 运行时 API 模块 _cudart
    return _cudart


class cudaStatus:
    SUCCESS: int = 0
    ERROR_NOT_READY: int = 34


class CudaError(RuntimeError):
    def __init__(self, code: int) -> None:
        # 使用 _cudart.cudaGetErrorString() 获取 CUDA 错误码的描述信息
        msg = _cudart.cudaGetErrorString(_cudart.cudaError(code))
        # 调用父类构造函数，传入错误信息和错误码
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    # 如果返回结果不等于 _cudart.cudaError.success，则抛出 CudaError 异常
    if res != _cudart.cudaError.success:
        raise CudaError(res)


class _DeviceGuard:
    def __init__(self, index: int):
        # 初始化 _DeviceGuard 类，保存当前设备索引和之前的设备索引
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        # 在进入 with 代码块时，使用 torch.cuda._exchange_device() 切换设备索引，并保存当前设备索引
        self.prev_idx = torch.cuda._exchange_device(self.idx)
    # 定义特殊方法 __exit__，用于退出上下文管理器时调用
    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 恢复之前的 CUDA 设备索引，通过调用 _maybe_exchange_device 方法
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        # 返回 False，表示未处理异常，使得异常可以继续传播
        return False
class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        # 获取设备索引，如果设备参数为负数或None，则不进行任何操作
        self.idx = _get_device_index(device, optional=True)
        # 记录先前的设备索引，默认为-1
        self.prev_idx = -1

    def __enter__(self):
        # 切换当前设备到新设备，并记录先前的设备索引
        self.prev_idx = torch.cuda._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 恢复先前的设备索引，并返回False表示不处理任何异常
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        # 获取对象所在的设备索引，如果对象未在GPU上分配，则设为-1
        idx = obj.get_device() if obj.is_cuda else -1
        super().__init__(idx)


def set_device(device: _device_t) -> None:
    r"""Set the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    # 获取设备索引，如果为负数则不进行任何操作
    device = _get_device_index(device)
    if device >= 0:
        # 设置当前CUDA设备
        torch._C._cuda_setDevice(device)


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    # 获取设备名称
    return get_device_properties(device).name


def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    r"""Get the cuda capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    # 获取设备的CUDA能力主版本号和次版本号
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_properties(device: _device_t) -> _CudaDeviceProperties:
    r"""Get the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _CudaDeviceProperties: the properties of the device
    """
    # 调用函数 _lazy_init() 进行初始化操作，此操作将定义 _get_device_properties 函数
    _lazy_init()  # will define _get_device_properties
    
    # 调用 _get_device_index 函数获取设备索引，允许设备索引可选
    device = _get_device_index(device, optional=True)
    
    # 检查设备索引是否在有效范围内，如果不在有效范围内则引发断言错误
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    
    # 返回通过 _get_device_properties 函数获取的设备属性，类型标注为 ignore[name-defined]
    return _get_device_properties(device)  # type: ignore[name-defined]
def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    r"""Check if peer access between two devices is possible."""
    # 初始化相关资源（如果需要的话）
    _lazy_init()
    # 获取设备对应的索引，允许设备参数为空
    device = _get_device_index(device, optional=True)
    # 获取对等设备的索引
    peer_device = _get_device_index(peer_device)
    # 检查设备索引是否在有效范围内
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    # 检查对等设备索引是否在有效范围内
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError("Invalid peer device id")
    # 调用底层函数检查设备之间是否可以互相访问
    return torch._C._cuda_canDeviceAccessPeer(device, peer_device)


class StreamContext:
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: Optional["torch.cuda.Stream"]

    def __init__(self, stream: Optional["torch.cuda.Stream"]):
        # 初始化上下文管理器，设置选定的流
        self.stream = stream
        # 获取当前设备索引，允许设备参数为空
        self.idx = _get_device_index(None, True)
        # 如果不是在 Torch 脚本模式下
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        # 设置源流为当前默认流（如果不是 Torch 脚本模式下）
        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        )
        # 设置目标流为当前默认流（如果不是 Torch 脚本模式下）
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        )

    def __enter__(self):
        # 本地化 cur_stream 变量以便进行类型细化
        cur_stream = self.stream
        # 如果流为空或者 CUDA 设备不可用，则直接返回
        if cur_stream is None or self.idx == -1:
            return
        # 将源流设为当前 CUDA 设备的当前流
        self.src_prev_stream = torch.cuda.current_stream(None)

        # 如果流不在当前设备上，则设置当前设备的当前流
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.cuda.current_stream(cur_stream.device)
        # 设置当前流为给定的流
        torch.cuda.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # 本地化 cur_stream 变量以便进行类型细化
        cur_stream = self.stream
        # 如果流为空或者 CUDA 设备不可用，则直接返回
        if cur_stream is None or self.idx == -1:
            return

        # 如果源流不在当前设备上，则设置目标设备的当前流
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.cuda.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        # 设置当前设备的当前流为源流
        torch.cuda.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch.cuda.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    # 返回一个 StreamContext 上下文管理器，选择指定的流
    return StreamContext(stream)
    """
    创建一个新的 StreamContext 对象，并使用给定的 stream 参数进行初始化。
    """
    return StreamContext(stream)
def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    # 调用底层函数设置指定流的属性，由流ID、设备索引和设备类型决定
    torch._C._cuda_setStream(
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
    # 如果未提供流对象，则不执行任何操作
    if stream is None:
        return
    # 使用给定流对象的属性设置当前流
    _set_stream_by_id(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def _parse_visible_devices() -> Union[List[int], List[str]]:
    r"""Parse CUDA_VISIBLE_DEVICES environment variable."""
    # 获取环境变量 CUDA_VISIBLE_DEVICES 或 HIP_VISIBLE_DEVICES
    var = os.getenv(
        "CUDA_VISIBLE_DEVICES" if not torch.version.hip else "HIP_VISIBLE_DEVICES"
    )
    # 如果环境变量未设置，则默认返回包含数字 0 到 63 的列表
    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with."""
        # 将字符串转换为整数，支持以正整数或以 +、- 开头的字符串
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not (c.isdigit() or (idx == 0 and c in "+-")):
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    def parse_list_with_prefix(lst: str, prefix: str) -> List[str]:
        rcs: List[str] = []
        for elem in lst.split(","):
            # 如果出现重复的ID，则返回空列表
            if elem in rcs:
                return cast(List[str], [])
            # 如果元素不以指定前缀开头，则忽略后续元素
            if not elem.startswith(prefix):
                break
            rcs.append(elem)
        return rcs

    # 如果变量以 "GPU-" 开头，则使用 "GPU-" 前缀解析列表
    if var.startswith("GPU-"):
        return parse_list_with_prefix(var, "GPU-")
    # 如果变量以 "MIG-" 开头，则使用 "MIG-" 前缀解析列表
    if var.startswith("MIG-"):
        return parse_list_with_prefix(var, "MIG-")
    # 如果是 CUDA_VISIBLE_DEVICES 类似格式的变量，则使用 strtoul 解析
    rc: List[int] = []
    for elem in var.split(","):
        x = _strtoul(elem.strip())
        # 如果序号重复，则返回空列表
        if x in rc:
            return cast(List[int], [])
        # 如果序号为负数，则终止序列解析
        if x < 0:
            break
        rc.append(x)
    return rc


def _raw_device_count_amdsmi() -> int:
    # 如果未安装 pynvml 库，则返回 -1
    if not _HAS_PYNVML:  # If amdsmi is not available
        return -1
    try:
        # 尝试初始化 amdsmi 库
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        # 如果初始化失败，则记录警告并返回 -1
        warnings.warn(f"Can't initialize amdsmi - Error code: {e.err_code}")
        return -1
    # 获取处理器句柄列表的长度，即设备数量
    socket_handles = amdsmi.amdsmi_get_processor_handles()
    return len(socket_handles)
# 返回由 NVML 报告的设备数量，如果 NVML 发现/初始化失败则返回负值
def _raw_device_count_nvml() -> int:
    from ctypes import byref, c_int, CDLL

    # 加载 NVML 库
    nvml_h = CDLL("libnvidia-ml.so.1")
    # 初始化 NVML，若返回值不为 0 则警告初始化失败
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML")
        return -1
    # 创建一个 c_int 类型的对象，用于存储设备数量
    dev_count = c_int(-1)
    # 获取设备数量，若返回值不为 0 则警告获取失败
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count")
        return -1
    # 删除 NVML 句柄，释放资源
    del nvml_h
    # 返回设备数量的值
    return dev_count.value


# 返回由 amdsmi 报告的设备 UUID 列表，如果 amdsmi 不可用则返回 None
def _raw_device_uuid_amdsmi() -> Optional[List[str]]:
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    # 如果 _HAS_PYNVML 为 False，则 amdsmi 不可用，返回 None
    if not _HAS_PYNVML:  # If amdsmi is not available
        return None
    try:
        # 初始化 amdsmi，若发生异常则警告初始化失败
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException:
        warnings.warn("Can't initialize amdsmi")
        return None
    try:
        # 获取处理器句柄列表，若发生异常则警告获取失败
        socket_handles = amdsmi.amdsmi_get_processor_handles()
        dev_count = len(socket_handles)
    except amdsmi.AmdSmiException:
        warnings.warn("Can't get amdsmi device count")
        return None
    # 创建一个空列表，用于存储 UUID
    uuids: List[str] = []
    # 遍历处理器句柄列表，获取每个设备的 UUID
    for idx in range(dev_count):
        try:
            handler = amdsmi.amdsmi_get_processor_handles()[idx]
        except amdsmi.AmdSmiException:
            warnings.warn("Cannot get amd device handler")
            return None
        try:
            # 获取指定处理器的 UUID，若发生异常则警告获取失败
            uuid = amdsmi.amdsmi_get_gpu_device_uuid(handler)
        except amdsmi.AmdSmiException:
            warnings.warn("Cannot get uuid for amd device")
            return None
        # 将获取到的 UUID 添加到 uuids 列表中
        uuids.append(str(uuid))
    # 返回 UUID 列表
    return uuids


# 返回由 NVML 报告的设备 UUID 列表，如果 NVML 发现/初始化失败则返回 None
def _raw_device_uuid_nvml() -> Optional[List[str]]:
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    # 加载 NVML 库
    nvml_h = CDLL("libnvidia-ml.so.1")
    # 初始化 NVML，若返回值不为 0 则警告初始化失败
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML")
        return None
    # 创建一个 c_int 类型的对象，用于存储设备数量
    dev_count = c_int(-1)
    # 获取设备数量，若返回值不为 0 则警告获取失败
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count")
        return None
    # 创建一个空列表，用于存储 UUID
    uuids: List[str] = []
    # 遍历设备数量，获取每个设备的 UUID
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        # 获取指定索引的设备句柄，若返回值不为 0 则警告获取失败
        rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
            warnings.warn("Can't get device handle")
            return None
        buf_len = 96
        buf = create_string_buffer(buf_len)
        # 获取设备的 UUID 到缓冲区 buf 中，若返回值不为 0 则警告获取失败
        rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn("Can't get device UUID")
            return None
        # 将获取到的 UUID 添加到 uuids 列表中，去除末尾的空字符
        uuids.append(buf.raw.decode("ascii").strip("\0"))
    # 删除 NVML 句柄，释放资源
    del nvml_h
    # 返回 UUID 列表
    return uuids


# 给定部分 UUID 和已知 UUID 列表，构建一组排除模糊不清的部分 ID 的顺序数列表
def _transform_uuid_to_ordinals(candidates: List[str], uuids: List[str]) -> List[int]:
    pass  # 此函数的实现尚未给出
    # 定义函数 uuid_to_orinal，将给定的候选字符串与 UUID 列表进行比较，返回最佳匹配的索引或特定的错误码
    def uuid_to_orinal(candidate: str, uuids: List[str]) -> int:
        # 初始化最佳匹配索引为 -1
        best_match = -1
        # 遍历 UUID 列表，获取索引和对应的 UUID
        for idx, uuid in enumerate(uuids):
            # 如果当前 UUID 不以候选字符串开头，则跳过
            if not uuid.startswith(candidate):
                continue
            # 如果已经有一个最佳匹配索引，表示存在多个匹配，返回错误码 -1
            if best_match != -1:
                return -1
            # 更新最佳匹配索引为当前索引
            best_match = idx
        # 返回最佳匹配索引或者错误码
        return best_match

    # 初始化结果列表 rc
    rc: List[int] = []
    # 遍历候选字符串列表 candidates
    for candidate in candidates:
        # 调用 uuid_to_orinal 函数，查找候选字符串在 UUID 列表中的匹配索引
        idx = uuid_to_orinal(candidate, uuids)
        # 如果返回的索引小于 0，表示找不到有效的匹配，停止解析
        if idx < 0:
            break
        # 如果已经在结果列表中存在相同的索引，返回空列表作为结果（表示重复）
        if idx in rc:
            return cast(List[int], [])
        # 将找到的索引添加到结果列表中
        rc.append(idx)
    # 返回最终的结果列表 rc
    return rc
# 返回通过 AMD SMI 或 NVML 报告的设备数量
def _device_count_amdsmi() -> int:
    # 解析可见设备列表
    visible_devices = _parse_visible_devices()
    # 如果没有可见设备，则返回 0
    if not visible_devices:
        return 0
    
    try:
        # 如果第一个可见设备是字符串类型，则返回 -1（暂不支持）
        if type(visible_devices[0]) is str:
            return -1
        else:
            # 获取原始设备数量
            raw_cnt = _raw_device_count_amdsmi()
            # 如果原始设备数量小于等于 0，则返回该值
            if raw_cnt <= 0:
                return raw_cnt
            
            # 将设备列表修剪到最大可用设备数
            for idx, val in enumerate(visible_devices):
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        # 捕获 OSError 异常，返回 -1 表示初始化失败
        return -1
    except AttributeError:
        # 捕获 AttributeError 异常，返回 -1 表示初始化失败
        return -1
    
    # 返回可见设备列表的长度
    return len(visible_devices)


# 返回通过 NVML 报告的设备数量
def _device_count_nvml() -> int:
    r"""Return number of devices as reported by NVML taking CUDA_VISIBLE_DEVICES into account.

    Negative value is returned if NVML discovery or initialization has failed.
    """
    # 解析可见设备列表
    visible_devices = _parse_visible_devices()
    # 如果没有可见设备，则返回 0
    if not visible_devices:
        return 0
    
    try:
        # 如果第一个可见设备是字符串类型
        if type(visible_devices[0]) is str:
            # 跳过 MIG 解析
            if visible_devices[0].startswith("MIG-"):
                return -1
            # 获取原始设备的 UUID
            uuids = _raw_device_uuid_nvml()
            # 如果 UUID 为空，则返回 -1 表示初始化失败
            if uuids is None:
                return -1
            # 将 UUID 转换为序数列表
            visible_devices = _transform_uuid_to_ordinals(
                cast(List[str], visible_devices), uuids
            )
        else:
            # 获取原始设备数量
            raw_cnt = _raw_device_count_nvml()
            # 如果原始设备数量小于等于 0，则返回该值
            if raw_cnt <= 0:
                return raw_cnt
            
            # 将设备列表修剪到最大可用设备数
            for idx, val in enumerate(visible_devices):
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        # 捕获 OSError 异常，返回 -1 表示初始化失败
        return -1
    except AttributeError:
        # 捕获 AttributeError 异常，返回 -1 表示初始化失败
        return -1
    
    # 返回可见设备列表的长度
    return len(visible_devices)


# 返回设备在 NVML 中的索引，考虑 CUDA_VISIBLE_DEVICES 的影响
def _get_nvml_device_index(device: Optional[Union[int, Device]]) -> int:
    r"""Return the NVML index of the device, taking CUDA_VISIBLE_DEVICES into account."""
    # 获取设备索引，允许为空
    idx = _get_device_index(device, optional=True)
    # 解析可见设备列表
    visible_devices = _parse_visible_devices()
    # 如果第一个可见设备是字符串类型
    if type(visible_devices[0]) is str:
        # 获取原始设备的 UUID
        uuids = _raw_device_uuid_nvml()
        # 如果 UUID 为空，则抛出 RuntimeError
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        # 将 UUID 转换为序数列表
        visible_devices = _transform_uuid_to_ordinals(
            cast(List[str], visible_devices), uuids
        )
    
    # 强制类型转换为整数列表
    visible_devices = cast(List[int], visible_devices)
    # 如果索引小于 0 或大于等于可见设备列表长度，则抛出 RuntimeError
    if idx < 0 or idx >= len(visible_devices):
        raise RuntimeError(
            f"device {idx} is not visible (CUDA_VISIBLE_DEVICES={visible_devices})"
        )
    
    # 返回设备索引对应的可见设备列表中的位置
    return visible_devices[idx]


# 可缓存的设备数量，默认为 None
_cached_device_count: Optional[int] = None


# 返回可用的 GPU 设备数量
def device_count() -> int:
    r"""Return the number of GPUs available."""
    # 全局变量 _cached_device_count，如果未编译则返回 0
    if not _is_compiled():
        return 0
    # 如果 _cached_device_count 不为 None，则直接返回其值
    if _cached_device_count is not None:
        return _cached_device_count
    # 如果是 rocm 平台，则绕过 _device_count_nvml()，因为不支持
    nvml_count = _device_count_amdsmi() if torch.version.hip else _device_count_nvml()
    # 获取 CUDA 设备数量，如果 nvml_count 小于 0 则使用 torch 库函数获取
    r = torch._C._cuda_getDeviceCount() if nvml_count < 0 else nvml_count
    
    # 注意：在 CUDA 初始化之前不要缓存设备数量，因为 CUDA_VISIBLE_DEVICES 的设置
    # 在 CUDA 初始化之前可能会更改设备数量。
    
    # 如果已经初始化过 CUDA，则将获取的设备数量缓存起来
    if _initialized:
        _cached_device_count = r
    
    # 返回获取的设备数量
    return r
# 返回此库编译的CUDA架构列表
def get_arch_list() -> List[str]:
    # 如果CUDA不可用，返回空列表
    if not is_available():
        return []
    # 获取当前CUDA架构标志
    arch_flags = torch._C._cuda_getArchFlags()
    # 如果没有获取到架构标志，返回空列表
    if arch_flags is None:
        return []
    # 返回由空格分隔的CUDA架构列表
    return arch_flags.split()


# 返回此库使用的NVCC gencode标志
def get_gencode_flags() -> str:
    # 获取CUDA架构列表
    arch_list = get_arch_list()
    # 如果架构列表为空，返回空字符串
    if len(arch_list) == 0:
        return ""
    # 将架构列表中的每个架构名称分割成列表
    arch_list_ = [arch.split("_") for arch in arch_list]
    # 返回gencode标志的字符串表示形式
    return " ".join(
        [
            f"-gencode compute=compute_{arch},code={kind}_{arch}"
            for (kind, arch) in arch_list_
        ]
    )


# 返回当前选定设备的索引
def current_device() -> int:
    # 确保CUDA环境已初始化
    _lazy_init()
    # 返回当前CUDA设备的索引
    return torch._C._cuda_getDevice()


# 等待CUDA设备上所有流中的所有内核完成
def synchronize(device: _device_t = None) -> None:
    # 确保CUDA环境已初始化
    _lazy_init()
    # 使用指定设备进行同步
    with torch.cuda.device(device):
        return torch._C._cuda_synchronize()


# 在CUDA IPC释放后强制收集GPU内存
def ipc_collect():
    # 确保CUDA环境已初始化
    _lazy_init()
    # 强制收集CUDA IPC释放的GPU内存
    return torch._C._cuda_ipc_collect()


# 返回给定设备的当前选定流
def current_stream(device: Optional[_device_t] = None) -> Stream:
    # 确保CUDA环境已初始化
    _lazy_init()
    # 获取当前选定设备的当前流数据
    streamdata = torch._C._cuda_getCurrentStream(
        _get_device_index(device, optional=True)
    )
    # 返回对应的Stream对象
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


# 返回给定设备的默认流
def default_stream(device: Optional[_device_t] = None) -> Stream:
    # 确保CUDA环境已初始化
    _lazy_init()
    # 获取指定设备的默认流数据
    streamdata = torch._C._cuda_getDefaultStream(
        _get_device_index(device, optional=True)
    )
    # 返回一个 Stream 对象，使用给定的 streamdata 参数进行初始化
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )
# 返回当前 cuBLAS 句柄的 cublasHandle_t 指针
def current_blas_handle():
    # 执行懒初始化
    _lazy_init()
    # 调用 torch 库函数获取当前 cuBLAS 句柄
    return torch._C._cuda_getCurrentBlasHandle()


# 设置 CUDA 同步操作的调试模式
def set_sync_debug_mode(debug_mode: Union[int, str]) -> None:
    # 执行懒初始化
    _lazy_init()
    # 如果 debug_mode 是字符串类型，则进行相应的映射转换
    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            # 抛出运行时异常，提示 debug_mode 值无效
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )
    # 调用 torch 库函数设置 CUDA 同步操作的调试模式
    torch._C._cuda_set_sync_debug_mode(debug_mode)


# 获取当前 CUDA 同步操作调试模式的值
def get_sync_debug_mode() -> int:
    # 执行懒初始化
    _lazy_init()
    # 调用 torch 库函数获取当前 CUDA 同步操作调试模式的值
    return torch._C._cuda_get_sync_debug_mode()


# 获取 pynvml 句柄处理器
def _get_pynvml_handler(device: Optional[Union[Device, int]] = None):
    # 如果 pynvml 模块未安装或无法导入，抛出模块未找到异常
    if not _HAS_PYNVML:
        raise ModuleNotFoundError(
            "pynvml does not seem to be installed or it can't be imported."
        ) from _PYNVML_ERR
    # 导入 pynvml 异常类 NVMLError_DriverNotLoaded
    from pynvml import NVMLError_DriverNotLoaded

    try:
        # 初始化 pynvml
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded as e:
        # 如果 CUDA 驱动加载失败，抛出运行时异常
        raise RuntimeError("cuda driver can't be loaded, is cuda enabled?") from e

    # 获取指定设备的 pynvml 句柄
    device = _get_nvml_device_index(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    return handle


# 获取 amdsmi 句柄处理器
def _get_amdsmi_handler(device: Optional[Union[Device, int]] = None):
    # 如果 amdsmi 模块未安装或无法导入，抛出模块未找到异常
    if not _HAS_PYNVML:
        raise ModuleNotFoundError(
            "amdsmi does not seem to be installed or it can't be imported."
        ) from _PYNVML_ERR
    try:
        # 初始化 amdsmi
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        # 如果 amdsmi 驱动加载失败，抛出运行时异常
        raise RuntimeError(
            "amdsmi driver can't be loaded, requires >=ROCm5.6 installation"
        ) from e
    # 获取指定设备的 amdsmi 句柄
    device = _get_amdsmi_device_index(device)
    handle = amdsmi.amdsmi_get_processor_handles()[device]
    return handle


# 获取 amdsmi 设备索引
def _get_amdsmi_device_index(device: Optional[Union[int, Device]]) -> int:
    # 获取设备索引，考虑 HIP_VISIBLE_DEVICES 环境变量的影响
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    if type(visible_devices[0]) is str:
        # 如果 HIP_VISIBLE_DEVICES 包含字符串而不是索引，抛出运行时异常
        raise RuntimeError("HIP_VISIBLE_DEVICES should be indices and not strings")
    # 构建索引映射字典
    idx_map = dict(enumerate(cast(List[int], visible_devices)))
    # 检查 idx 是否在 idx_map 中，如果不在则抛出 RuntimeError 异常
    if idx not in idx_map:
        raise RuntimeError(
            f"device {idx} is not visible (HIP_VISIBLE_DEVICES={visible_devices})"
        )
    # 如果 idx 在 idx_map 中，则返回对应的 idx_map[idx]
    return idx_map[idx]
def _get_amdsmi_memory_usage(device: Optional[Union[Device, int]] = None) -> int:
    # 获取 AMD SMI 句柄
    handle = _get_amdsmi_handler()
    # 获取 AMD SMI 设备索引
    device = _get_amdsmi_device_index(device)
    # 调用 AMD SMI API 获取 GPU 内存使用情况
    return amdsmi.amdsmi_get_gpu_vram_usage(handle)["vram_used"]


def _get_amdsmi_utilization(device: Optional[Union[Device, int]] = None) -> int:
    # 获取 AMD SMI 句柄
    handle = _get_amdsmi_handler()
    # 获取 AMD SMI 设备索引
    device = _get_amdsmi_device_index(device)
    # 获取处理器句柄并获取 GPU 活动信息
    handle = amdsmi.amdsmi_get_processor_handles()[device]
    return amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"]


def _get_amdsmi_temperature(device: Optional[Union[Device, int]] = None) -> int:
    # 获取 AMD SMI 句柄和设备温度信息
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_temp_metric(
        handle,
        amdsmi.AmdSmiTemperatureType.JUNCTION,
        amdsmi.AmdSmiTemperatureMetric.CURRENT,
    )


def _get_amdsmi_power_draw(device: Optional[Union[Device, int]] = None) -> int:
    # 获取 AMD SMI 句柄和设备功耗信息
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_power_info(handle)["current_socket_power"]


def _get_amdsmi_clock_rate(device: Optional[Union[Device, int]] = None) -> int:
    # 获取 AMD SMI 句柄和设备当前时钟信息
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)["cur_clk"]


def memory_usage(device: Optional[Union[Device, int]] = None) -> int:
    r"""Return the percent of time over the past sample period during which global (device)
    memory was being read or written as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        # 获取 PyNVML 句柄和设备索引
        handle = _get_pynvml_handler()
        device = _get_nvml_device_index(device)
        # 获取设备句柄并返回内存利用率
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).memory
    else:
        # 如果是 HIP 环境，则调用 AMD SMI 获取内存使用情况
        return _get_amdsmi_memory_usage(device)


def utilization(device: Optional[Union[Device, int]] = None) -> int:
    r"""Return the percent of time over the past sample period during which one or
    more kernels was executing on the GPU as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        # 获取 PyNVML 句柄和设备索引
        handle = _get_pynvml_handler(device)
        device = _get_nvml_device_index(device)
        # 获取设备句柄并返回 GPU 利用率
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    else:
        # 如果是 HIP 环境，则调用 AMD SMI 获取 GPU 利用率
        return _get_amdsmi_utilization(device)
def temperature(device: Optional[Union[Device, int]] = None) -> int:
    r"""Return the average temperature of the GPU sensor in Degrees C (Centigrades).

    The average temperature is computed based on past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    # 如果不是使用 HIP 版本（AMD GPU），则获取对应设备的 pynvml 句柄
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        # 返回 GPU 芯片的温度，0 表示 GPU 芯片的温度传感器
        return pynvml.nvmlDeviceGetTemperature(handle, 0)
    else:
        # 如果是 HIP 版本（AMD GPU），则调用 AMD GPU 温度获取函数
        return _get_amdsmi_temperature(device)


def power_draw(device: Optional[Union[Device, int]] = None) -> int:
    r"""Return the average power draw of the GPU sensor in mW (MilliWatts)
        over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    # 如果不是使用 HIP 版本（AMD GPU），则获取对应设备的 pynvml 句柄
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        # 返回 GPU 的功耗使用情况
        return pynvml.nvmlDeviceGetPowerUsage(handle)
    else:
        # 如果是 HIP 版本（AMD GPU），则调用 AMD GPU 功耗获取函数
        return _get_amdsmi_power_draw(device)


def clock_rate(device: Optional[Union[Device, int]] = None) -> int:
    r"""Return the clock speed of the GPU SM in Hz Hertz over the past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    # 如果不是使用 HIP 版本（AMD GPU），则获取对应设备的 pynvml 句柄
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        # 返回 GPU SM 的时钟速率，1 表示 SM 的时钟速率
        return pynvml.nvmlDeviceGetClockInfo(handle, 1)
    else:
        # 如果是 HIP 版本（AMD GPU），则调用 AMD GPU 时钟速率获取函数
        return _get_amdsmi_clock_rate(device)


def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    # 如果 device 是字符串类型，则转换成 torch.device 对象
    if isinstance(device, str):
        device = torch.device(device)
    # 如果 device 是整数类型，则使用 "cuda" + device 作为参数创建 torch.device 对象
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    return device


def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the CUDA Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """
    # 获取 device 对应的索引
    idx = device.index
    # 如果 idx 参数为 None，则获取当前设备的索引
    if idx is None:
        idx = current_device()
    # 返回在 CUDA 中使用的默认随机数生成器，使用给定的设备索引
    return torch.cuda.default_generators[idx]
def _set_rng_state_offset(
    offset: int, device: Union[int, str, torch.device] = "cuda"
) -> None:
    r"""Set the random number generator state offset of the specified GPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
    # 获取最终设备对象
    final_device = _get_device(device)

    def cb():
        # 获取指定设备的默认随机数生成器对象
        default_generator = _get_generator(final_device)
        # 设置随机数生成器的偏移量
        default_generator.set_offset(offset)

    # 执行延迟调用
    _lazy_call(cb)


def _get_rng_state_offset(device: Union[int, str, torch.device] = "cuda") -> int:
    r"""Return the random number generator state offset of the specified GPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    # 确保 CUDA 已经初始化
    _lazy_init()
    # 获取最终设备对象
    final_device = _get_device(device)
    # 获取指定设备的默认随机数生成器对象
    default_generator = _get_generator(final_device)
    # 返回随机数生成器的偏移量
    return default_generator.get_offset()


from .memory import *  # noqa: F403


from .random import *  # noqa: F403

################################################################################
# Define Storage and Tensor classes
################################################################################


@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    # 惰性初始化
    _lazy_init()
    # 如果是分支子进程，可能需要再次调用惰性初始化
    # 删除 _CudaBase.__new__ 可能是为了重新定义 __new__ 方法？
    return super(_CudaBase, cls).__new__(cls, *args, **kwargs)


class _CudaBase:
    is_cuda = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # 这里可以使用 Protocol 来告诉 mypy，self 有 `get_device` 方法，
        # 但是只有在 Python >= 3.8 时才在 typing 模块中可用，
        # 或者在 Python >= 3.6 时在 typing_extensions 模块中可用。
        with device(self.get_device()):  # type: ignore[attr-defined]
            return super().type(*args, **kwargs)  # type: ignore[misc]

    __new__ = _lazy_new


from torch.storage import _LegacyStorage, _warn_typed_storage_removal


class _CudaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        # 警告类型化存储的移除
        _warn_typed_storage_removal()
        # 抛出运行时异常，CUDA 存储不支持 from_buffer 方法
        raise RuntimeError("from_buffer: Not available for CUDA storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        # 抛出运行时异常，CUDA 存储不支持 _new_with_weak_ptr 方法
        raise RuntimeError("_new_with_weak_ptr: Not available for CUDA storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        # 抛出运行时异常，CUDA 存储不支持 _new_shared_filename 方法
        raise RuntimeError("_new_shared_filename: Not available for CUDA storage")


class ByteStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        # 警告类型化存储的移除
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    # 定义一个方法 _dtype，返回 torch 库中的 uint8 数据类型
    def _dtype(self):
        return torch.uint8
# 定义 DoubleStorage 类，继承自 _CudaLegacyStorage 类
class DoubleStorage(_CudaLegacyStorage):
    # 类属性装饰器，获取 dtype 属性，会触发 _warn_typed_storage_removal 函数警告移除类型化存储
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        # 返回该类的 _dtype 属性
        return self._dtype

    # 返回 torch.double 类型
    @classproperty
    def _dtype(self):
        return torch.double


# 定义 FloatStorage 类，继承自 _CudaLegacyStorage 类，以下类似
class FloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat


# 删除 _LegacyStorage 和 _CudaLegacyStorage 类
del _LegacyStorage
del _CudaLegacyStorage

# 将各种数据类型的 Storage 类加入 torch._storage_classes 集合中
torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)  # 未在提供的代码中定义，可能是其他地方定义的类
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(ComplexDoubleStorage)
torch._storage_classes.add(ComplexFloatStorage)

# 定义 _WrappedTritonKernel 类
class _WrappedTritonKernel:
    """Just a simple wrapper to store some metadata for testing purposes."""
    
    # 定义一个简单的包装器类，用于存储测试目的的元数据
    class KernelWrapper:
        # 初始化方法，接收一个 kernel 参数并存储在实例中，同时初始化 kernel_invoked 为 False
        def __init__(self, kernel):
            self.kernel = kernel  # 存储传入的 kernel 函数或对象
            self.kernel_invoked = False  # 初始化 kernel_invoked 为 False，表示 kernel 尚未被调用过
    
        # 实现 __call__ 方法，使实例对象可以像函数一样被调用
        def __call__(self, *args, **kwargs):
            # 调用存储的 kernel 对象，并传递参数 args 和 kwargs
            res = self.kernel(*args, **kwargs)
            self.kernel_invoked = True  # 设置 kernel_invoked 为 True，表示 kernel 已被调用过
            return res  # 返回 kernel 执行的结果
# 如果当前环境是 Torch 部署环境，则直接返回，不注册 Triton 内核
def _register_triton_kernels():
    if torch._running_with_deploy():
        return

    # 定义装饰的 Triton 内核函数 kernel_impl
    @_WrappedTritonKernel
    def kernel_impl(*args, **kwargs):
        # 导入 Triton 稀疏操作的函数 bsr_dense_mm
        from torch.sparse._triton_ops import bsr_dense_mm

        # 调用 bsr_dense_mm 函数进行稀疏矩阵乘法
        return bsr_dense_mm(*args, skip_checks=True, **kwargs)

    # 定义装饰的 Triton 内核函数 addmm_kernel_impl
    @_WrappedTritonKernel
    def addmm_kernel_impl(*args, **kwargs):
        # 导入 Triton 稀疏操作的函数 bsr_dense_addmm
        from torch.sparse._triton_ops import bsr_dense_addmm

        # 调用 bsr_dense_addmm 函数进行稀疏矩阵乘加操作
        return bsr_dense_addmm(*args, skip_checks=True, **kwargs)

    # 检查是否导入了 Triton 库
    has_triton = importlib.util.find_spec("triton") is not None
    if has_triton:
        # 如果导入了 Triton 库，则注册 Triton 稀疏操作到 Torch TritonLibrary 中
        torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_mm_out",
            "_triton_bsr_dense_mm_out(Tensor bsr, Tensor dense, *, Tensor(a!) out) -> Tensor(a!)",
            kernel_impl,
            "SparseCsrCUDA",
        )

        torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_addmm_out",
            (
                "_triton_bsr_dense_addmm_out(Tensor input, Tensor bsr, Tensor dense,"
                " *, Scalar beta, Scalar alpha, Tensor(a!) out) -> Tensor(a!)"
            ),
            addmm_kernel_impl,
            "SparseCsrCUDA",
        )

# 使用 _register_triton_kernels 函数注册 Triton 内核
_lazy_call(_register_triton_kernels)
    "memory_cached",  
    "memory_reserved",  
    "memory_snapshot",  
    "memory_stats",  
    "memory_stats_as_nested_dict",  
    "memory_summary",  
    "memory_usage",  
    "temperature",  
    "power_draw",  
    "clock_rate",  
    "nccl",  
    "nvtx",  
    "profiler",  
    "random",  
    "reset_accumulated_memory_stats",  
    "reset_max_memory_allocated",  
    "reset_max_memory_cached",  
    "reset_peak_memory_stats",  
    "seed",  
    "seed_all",  
    "set_device",  
    "set_per_process_memory_fraction",  
    "set_rng_state",  
    "set_rng_state_all",  
    "set_stream",  
    "set_sync_debug_mode",  
    "sparse",  
    "stream",  
    "streams",  
    "synchronize",  
    "tunable",  
    "utilization",

# 下面是一系列可能由某个库或框架提供的功能或方法名称的字符串列表
]


注释：


# 这行代码似乎是一个独立的方括号，但是缺少上下文无法准确解释其作用。
# 在正常情况下，方括号可能用于列表、索引或者切片操作中，但这里单独存在，可能是一个错误或者截断的代码片段。
# 需要更多的上下文来理解它的具体含义和用法。
```