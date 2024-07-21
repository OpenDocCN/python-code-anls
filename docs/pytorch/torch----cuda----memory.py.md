# `.\pytorch\torch\cuda\memory.py`

```py
# mypy: allow-untyped-defs
r"""This package adds support for device memory management implemented in CUDA."""

import collections  # 导入 Python 标准库 collections，用于高效的数据结构
import contextlib  # 导入 Python 标准库 contextlib，用于创建上下文管理器
import ctypes  # 导入 Python 标准库 ctypes，用于与 C 语言兼容的数据类型
import pickle  # 导入 Python 标准库 pickle，用于对象的序列化和反序列化
import sys  # 导入 Python 标准库 sys，用于访问系统相关的参数和功能
import warnings  # 导入 Python 标准库 warnings，用于处理警告
from inspect import signature  # 从 inspect 模块中导入 signature 函数，用于获取函数的签名信息

from typing import Any, Dict, Optional, Tuple, Union  # 导入 typing 模块，用于类型提示
from typing_extensions import deprecated  # 导入 typing_extensions 模块的 deprecated 类型

import torch  # 导入 PyTorch 库
from torch import _C  # 从 torch 模块中导入 _C 对象

from torch.types import Device  # 从 torch.types 模块中导入 Device 类型
from .._utils import _dummy_type  # 从当前包的 _utils 模块中导入 _dummy_type 函数
from . import (
    _get_amdsmi_device_index,  # 从当前包中的 _get_amdsmi_device_index 模块导入函数
    _get_device_index,  # 从当前包中的 _get_device_index 模块导入函数
    _get_nvml_device_index,  # 从当前包中的 _get_nvml_device_index 模块导入函数
    _lazy_init,  # 从当前包中的 _lazy_init 模块导入函数
    is_initialized,  # 从当前包中的 is_initialized 模块导入函数
)

from ._memory_viz import memory as _memory, segments as _segments  # 从当前包中的 _memory_viz 模块导入 memory 和 segments

__all__ = [  # 定义导出的所有名称列表
    "caching_allocator_alloc",  # 分配内存的函数
    "caching_allocator_delete",  # 释放内存的函数
    "set_per_process_memory_fraction",  # 设置每个进程的内存分配比例
    "empty_cache",  # 清空缓存的函数
    "memory_stats",  # 内存统计信息
    "memory_stats_as_nested_dict",  # 以嵌套字典形式返回内存统计信息
    "reset_accumulated_memory_stats",  # 重置累计的内存统计信息
    "reset_peak_memory_stats",  # 重置峰值内存统计信息
    "reset_max_memory_allocated",  # 重置最大内存分配统计信息
    "reset_max_memory_cached",  # 重置最大缓存统计信息
    "memory_allocated",  # 当前分配的内存量
    "max_memory_allocated",  # 峰值分配的内存量
    "memory_reserved",  # 当前保留的内存量
    "max_memory_reserved",  # 峰值保留的内存量
    "memory_cached",  # 当前缓存的内存量
    "max_memory_cached",  # 峰值缓存的内存量
    "memory_snapshot",  # 内存快照
    "memory_summary",  # 内存摘要
    "list_gpu_processes",  # 列出 GPU 进程
    "mem_get_info",  # 获取内存信息
    "get_allocator_backend",  # 获取分配器后端
    "CUDAPluggableAllocator",  # CUDA 可插拔分配器
    "change_current_allocator",  # 更改当前分配器
]

if not hasattr(torch._C, "_cuda_CUDAAllocator"):  # 如果 _cuda_CUDAAllocator 不在 torch._C 中
    # 定义一个虚拟的基类 _cuda_CUDAAllocator
    torch._C.__dict__["_cuda_CUDAAllocator"] = _dummy_type("_cuda_CUDAAllocator")


def _host_allocator():  # 定义主机内存分配器函数
    _lazy_init()  # 惰性初始化
    return torch._C._cuda_cudaHostAllocator()  # 返回 CUDA 主机内存分配器


@contextlib.contextmanager
def _free_mutex():  # 定义释放互斥锁的上下文管理器函数
    torch._C._cuda_lock_mutex()  # 加锁
    try:
        yield  # 执行上下文管理器的主体部分
    finally:
        torch._C._cuda_unlock_mutex()  # 解锁


def caching_allocator_alloc(size, device: Union[Device, int] = None, stream=None):
    r"""Perform a memory allocation using the CUDA memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch.cuda.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
        stream (torch.cuda.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    if device is None:  # 如果设备未指定
        device = torch.cuda.current_device()  # 获取当前 CUDA 设备
    device = _get_device_index(device)  # 获取设备索引
    if stream is None:  # 如果流未指定
        stream = torch.cuda.current_stream(device)  # 获取当前设备的默认流
    if isinstance(stream, torch.cuda.streams.Stream):  # 如果流是 Torch CUDA 流对象
        stream = stream.cuda_stream  # 获取 CUDA 流的底层流
    # 如果流对象不是整数类型，抛出类型错误异常
    if not isinstance(stream, int):
        raise TypeError(
            "Invalid type for stream argument, must be "
            "`torch.cuda.Stream` or `int` representing a pointer "
            "to an existing stream"
        )
    # 设置当前 CUDA 设备为指定的设备
    with torch.cuda.device(device):
        # 使用 CUDA 缓存分配器进行原始分配操作，分配指定大小的内存块
        return torch._C._cuda_cudaCachingAllocator_raw_alloc(size, stream)
def caching_allocator_delete(mem_ptr):
    r"""Delete memory allocated using the CUDA memory allocator.

    Memory allocated with :func:`~torch.cuda.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    # 调用底层函数释放由CUDA内存分配器分配的内存
    torch._C._cuda_cudaCachingAllocator_raw_delete(mem_ptr)


def set_per_process_memory_fraction(
    fraction, device: Union[Device, int] = None
) -> None:
    r"""Set memory fraction for a process.

    The fraction is used to limit an caching allocator to allocated memory on a CUDA device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    # 惰性初始化
    _lazy_init()
    # 如果未指定设备，则使用当前CUDA设备
    if device is None:
        device = torch.cuda.current_device()
    # 获取设备索引
    device = _get_device_index(device)
    # 检查分数是否为浮点数类型
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    # 检查分数是否在0到1之间
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~1")

    # 调用底层函数设置CUDA内存分配器的内存分配比例
    torch._C._cuda_setMemoryFraction(fraction, device)


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch.cuda.empty_cache` doesn't increase the amount of GPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of GPU memory in certain cases. See :ref:`cuda-memory-management` for
        more details about GPU memory management.
    """
    # 如果已经初始化，则释放所有未使用的缓存内存
    if is_initialized():
        torch._C._cuda_emptyCache()


def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Return a dictionary of CUDA memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``cudaMalloc()``.

    """
    # 返回给定设备的CUDA内存分配器统计信息的字典
    return torch._C._cuda_memoryStats(device)
    # "reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
    # 表示预留内存的统计信息。
    # "active.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
    # 表示活跃内存块的数量。
    # "active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
    # 表示活跃内存的量。
    # "inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
    # 表示非活跃、不可释放的内存块的数量。
    # "inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}":
    # 表示非活跃、不可释放的内存的量。
    #
    # 对于这些核心统计数据，具体分解如下：
    #
    # Pool type:
    # - "all": 所有内存池的综合统计。
    # - "large_pool": 大型分配池的统计（从2019年10月起，用于大小>=1MB的分配）。
    # - "small_pool": 小型分配池的统计（从2019年10月起，用于大小<1MB的分配）。
    #
    # Metric type:
    # - "current": 当前指标值。
    # - "peak": 指标的最大值。
    # - "allocated": 此指标的历史总增加量。
    # - "freed": 此指标的历史总减少量。
    #
    # 除了核心统计数据外，还提供一些简单的事件计数器：
    # - "num_alloc_retries": 失败的cudaMalloc调用数，导致缓存刷新和重试。
    # - "num_ooms": 抛出的内存不足错误数。
    # - "num_sync_all_streams": synchronize_and_free_events调用数。
    # - "num_device_alloc": CUDA分配调用数，包括cuMemMap和cudaMalloc。
    # - "num_device_free": CUDA释放调用数，包括cuMemUnmap和cudaFree。
    #
    # 可以通过环境变量配置缓存分配器，以避免分割大于指定大小的块（请参阅CUDA语义文档的内存管理部分）。
    # 这有助于避免内存碎片化，但可能会影响性能。
    # 提供额外的输出来帮助调整和评估影响：
    # - "max_split_size": 不会分割超过此大小的块。
    # - "oversize_allocations.{current,peak,allocated,freed}":
    #   内存分配器接收的超大尺寸分配请求数量。
    # - "oversize_segments.{current,peak,allocated,freed}":
    #   来自cudaMalloc()的超大尺寸保留段数量。
    #
    # 可以通过环境变量配置缓存分配器，以舍入内存分配以减少碎片化。
    # 有时舍入带来的开销可能高于其有助于减少的碎片化。
    # 可以使用以下统计数据来检查舍入是否增加了过多的开销：
    """
    # 返回 GPU 内存统计信息的嵌套字典，按照指定的前缀和对象递归添加到结果列表中
    result = []
    
    def _recurse_add_to_result(prefix, obj):
        # 如果对象是字典类型，则递归处理其键值对
        if isinstance(obj, dict):
            # 如果前缀非空，则在其末尾添加点号作为连接符
            if len(prefix) > 0:
                prefix += "."
            # 遍历字典中的键值对
            for k, v in obj.items():
                # 递归调用自身，处理子对象，更新前缀
                _recurse_add_to_result(prefix + k, v)
        else:
            # 如果对象不是字典类型，则将前缀和对象添加到结果列表中
            result.append((prefix, obj))
    
    # 调用函数获取 GPU 内存的嵌套字典统计信息
    stats = memory_stats_as_nested_dict(device=device)
    # 递归处理获取的统计信息，将结果添加到结果列表中
    _recurse_add_to_result("", stats)
    # 对结果列表按照键排序
    result.sort()
    
    # 返回排序后的有序字典作为最终结果
    return collections.OrderedDict(result)
    """
# 返回包含 GPU 内存统计结果的嵌套字典
def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    # 如果未初始化 CUDA 环境，则返回空字典
    if not is_initialized():
        return {}
    # 获取设备索引
    device = _get_device_index(device, optional=True)
    # 调用底层 C 函数获取 CUDA 内存统计信息并返回
    return torch._C._cuda_memoryStats(device)


# 重置 CUDA 内存分配器跟踪的累积统计数据
def reset_accumulated_memory_stats(device: Union[Device, int] = None) -> None:
    # 获取设备索引
    device = _get_device_index(device, optional=True)
    # 调用底层 C 函数重置 CUDA 内存累积统计数据
    return torch._C._cuda_resetAccumulatedMemoryStats(device)


# 重置 CUDA 内存分配器跟踪的峰值统计数据
def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    # 获取设备索引
    device = _get_device_index(device, optional=True)
    # 调用底层 C 函数重置 CUDA 内存峰值统计数据
    return torch._C._cuda_resetPeakMemoryStats(device)


# 重置给定设备上张量占用的最大 GPU 内存的起始点
def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    # 发出警告，说明此函数现在调用 reset_peak_memory_stats 来重置所有峰值内存统计数据
    warnings.warn(
        "torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    # 调用 reset_peak_memory_stats 来重置最大内存占用统计
    return reset_peak_memory_stats(device=device)


# 重置给定设备上张量缓存的最大 GPU 内存的起始点
def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.

    See :func:`~torch.cuda.max_memory_cached` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    # 发出警告，说明 reset_max_memory_cached 现在调用 reset_peak_memory_stats，将重置所有峰值内存统计信息
    warnings.warn(
        "torch.cuda.reset_max_memory_cached now calls torch.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    # 调用 reset_peak_memory_stats 函数，传入设备参数，重置最大内存缓存的峰值统计信息
    return reset_peak_memory_stats(device=device)
def memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
    # 使用 memory_stats 函数获取当前设备上的已分配内存字节数
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    # 使用 memory_stats 函数获取当前设备上的峰值已分配内存字节数
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    # 使用 memory_stats 函数获取当前设备上的已预留内存字节数
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    ```
    # 使用 memory_stats 函数获取当前设备上的峰值已预留内存字节数
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)
    """
    返回给定设备的GPU内存峰值保留字节数。

    .. note::
        查看 :ref:`cuda-memory-management` 以获取有关GPU内存管理的更多详细信息。
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)
# 标记函数已被废弃，并提供替代函数名和相关警告信息
@deprecated(
    "`torch.cuda.memory_cached` has been renamed to `torch.cuda.memory_reserved`",
    category=FutureWarning,
)
# 定义函数 `memory_cached`，返回 CUDA 设备上已缓存的内存大小
def memory_cached(device: Union[Device, int] = None) -> int:
    # 函数文档字符串，指出函数已被废弃，建议使用 `memory_reserved` 函数代替
    r"""Deprecated; see :func:`~torch.cuda.memory_reserved`."""
    # 调用 `memory_reserved` 函数，并返回其结果
    return memory_reserved(device=device)


# 标记函数已被废弃，并提供替代函数名和相关警告信息
@deprecated(
    "`torch.cuda.max_memory_cached` has been renamed to `torch.cuda.max_memory_reserved`",
    category=FutureWarning,
)
# 定义函数 `max_memory_cached`，返回 CUDA 设备上已缓存的最大内存大小
def max_memory_cached(device: Union[Device, int] = None) -> int:
    # 函数文档字符串，指出函数已被废弃，建议使用 `max_memory_reserved` 函数代替
    r"""Deprecated; see :func:`~torch.cuda.max_memory_reserved`."""
    # 调用 `max_memory_reserved` 函数，并返回其结果
    return max_memory_reserved(device=device)


# 定义函数 `memory_snapshot`，返回跨所有设备的 CUDA 内存分配器状态快照
def memory_snapshot():
    # 函数文档字符串，描述返回的是 CUDA 内存分配器状态快照
    r"""Return a snapshot of the CUDA memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    # 调用 torch._C._cuda_memorySnapshot() 函数并返回其 "segments" 键对应的值
    return torch._C._cuda_memorySnapshot()["segments"]


# 定义函数 `memory_summary`，返回指定设备的当前内存分配器统计信息的人类可读打印输出
def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    # 函数文档字符串，描述返回当前设备的内存分配器统计信息的打印输出
    r"""Return a human-readable printout of the current memory allocator statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    # 获取设备的索引
    device = _get_device_index(device, optional=True)
    # 调用 memory_stats 函数获取设备的内存统计信息
    stats = memory_stats(device=device)

    # 定义内部函数 `_format_size`，用于格式化字节数为可读的字符串
    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    # 定义内部函数 `_format_count`，用于格式化计数值为可读的字符串
    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return f"{cnt:7d} {prefix} "
    # 定义要展示的指标及其对应的描述和格式化函数
    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("requested_bytes", "Requested memory", _format_size),
        ("reserved_bytes", "GPU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "GPU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]
    
    # 初始化空列表 lines 用于存储最终生成的输出行
    lines = []
    
    # 添加一条分隔线作为标题上方的装饰
    lines.append("=" * 75)
    
    # 添加标题行，包括设备 ID 的占位符
    lines.append(" {_:16} PyTorch CUDA memory summary, device ID {device:<17d} ")
    
    # 添加分隔线作为标题下方的装饰
    lines.append("-" * 75)
    
    # 添加另一条分隔线作为表头的装饰
    lines.append(
        "  {_:9} CUDA OOMs: {num_ooms:<12d} | {_:6} cudaMalloc retries: {num_alloc_retries:<8d}  "
    )
    
    # 添加最终的分隔线作为表格顶部的装饰
    lines.append("=" * 75)
    
    # 添加表格的列标题行
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )
    
    # 遍历要展示的指标和其相关信息
    for metric_key, metric_name, formatter in metrics_to_display:
        # 添加分隔线用于分隔不同的指标数据行
        lines.append("-" * 75)
        
        # 如果不是简略模式，则添加额外的子指标信息
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))
    
        # 初始化当前、峰值、已分配和已释放值
        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )
    
        # 遍历每个子指标，构建数据行
        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."
    
            # 获取当前、峰值、已分配和已释放的具体数值
            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]
    
            # 如果是第一次迭代，初始化基准值
            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed
    
            # 格式化并添加当前行的数据
            lines.append(
                f" {submetric_name:<21} | {formatter(current, current_prefval)} | {formatter(peak, peak_prefval)} | "
                f"{formatter(allocated, allocated_prefval)} | {formatter(freed, freed_prefval)} ",
            )
    
    # 更新要展示的指标，添加额外的指标信息
    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize GPU segments", _format_count),
    ]
    
    # 遍历要展示的指标和其相关信息
    for metric_key, metric_name, formatter in metrics_to_display:
        # 添加分隔线用于分隔不同的指标数据行
        lines.append("-" * 75)
    
        prefix = metric_key + "."
    
        # 获取当前、峰值、已分配和已释放的具体数值
        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]
    
        # 添加当前行的数据
        lines.append(
            f" {metric_name:<21} | {formatter(current, current)} | {formatter(peak, peak)} | "
            f"{formatter(allocated, allocated)} | {formatter(freed, freed)} ",
        )
    
    # 添加最终的分隔线作为表格底部的装饰
    lines.append("=" * 75)
    
    # 初始化格式化字典，用于替换标题行中的占位符
    fmt_dict = {"_": "", "device": device}
    # 遍历 stats 字典中的键值对
    for k, v in stats.items():
        # 将键中的点替换为破折号，存入 fmt_dict 字典中
        fmt_dict[k.replace(".", "-")] = v
    # 将 lines 列表中的元素以竖线分隔并格式化，使用 fmt_dict 中的值替换占位符，最终返回格式化后的字符串
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"
# 返回给定设备上运行进程的人类可读输出，包括它们的 GPU 内存使用情况

def list_gpu_processes(device: Union[Device, int] = None) -> str:
    r"""Return a human-readable printout of the running processes and their GPU memory use for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """

    # 检查是否为 HIP 环境，若不是则执行以下代码块
    if not torch.version.hip:
        try:
            import pynvml  # type: ignore[import]
        except ModuleNotFoundError:
            return "pynvml module not found, please install pynvml"
        from pynvml import NVMLError_DriverNotLoaded

        # 初始化 NVML 库，用于获取 GPU 运行进程信息
        try:
            pynvml.nvmlInit()
        except NVMLError_DriverNotLoaded:
            return "cuda driver can't be loaded, is cuda enabled?"

        # 获取指定设备的 NVML 设备索引
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

    # 如果是 HIP 环境，则执行以下代码块
    else:
        try:
            import amdsmi  # type: ignore[import]
        except ModuleNotFoundError:
            return "amdsmi module not found, please install amdsmi"
        try:
            amdsmi.amdsmi_init()  # type: ignore[attr-defined]
        except amdsmi.AmdSmiException:  # type: ignore[attr-defined]
            return "amdsmi driver can't be loaded, is ROCm installed?"

        # 获取指定设备的 AMD SMI 设备索引
        device = _get_amdsmi_device_index(device)
        handle = amdsmi.amdsmi_get_processor_handles()[device]  # type: ignore[attr-defined]
        procs = amdsmi.amdsmi_get_gpu_process_list(handle)  # type: ignore[attr-defined]

    # 创建输出行列表，用于存储 GPU 进程信息的人类可读格式
    lines = []
    lines.append(f"GPU:{device}")

    # 如果没有运行中的进程，添加相应提示到输出行列表
    if len(procs) == 0:
        lines.append("no processes are running")

    # 遍历每个进程，生成其 GPU 内存使用信息的人类可读格式
    for p in procs:
        if not torch.version.hip:
            mem = p.usedGpuMemory / (1024 * 1024)  # 转换为 MB
            pid = p.pid
        else:
            try:
                proc_info = amdsmi.amdsmi_get_gpu_process_info(handle, p)  # type: ignore[possibly-undefined]
            except AttributeError:
                # 处理 ROCm 最新版本可能删除 amdsmi_get_gpu_process_info API 的情况
                proc_info = p
            mem = proc_info["memory_usage"]["vram_mem"] / (1024 * 1024)  # 转换为 MB
            pid = proc_info["pid"]

        lines.append(f"process {pid:>10d} uses {mem:>12.3f} MB GPU memory")

    # 将所有行连接成一个字符串，并返回
    return "\n".join(lines)


def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    r"""Return the global free and total GPU memory for a given device using cudaMemGetInfo.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    # 如果未指定设备，则使用当前 CUDA 设备
    if device is None:
        device = torch.cuda.current_device()
    # 获取设备的索引，确保设备索引的正确性
    device = _get_device_index(device)
    # 调用 CUDA 运行时的 cudaMemGetInfo 函数，获取特定设备的内存信息
    return torch.cuda.cudart().cudaMemGetInfo(device)
def _record_memory_history_legacy(
    enabled: bool,
    record_context=True,
    trace_alloc_max_entries=1,
    trace_alloc_record_context=False,
    device: Union[Device, int] = None,
    record_context_cpp=False,
):
    # 调用 C++ 函数来记录内存历史，根据参数来确定记录的详细内容和条件
    _C._cuda_record_memory_history_legacy(
        enabled,
        record_context,
        trace_alloc_max_entries,
        trace_alloc_record_context,
        record_context_cpp,
    )


def _record_memory_history(enabled="all", *args, **kwargs):
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch.cuda.memory._snapshot()`.

    In addition too keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch.cuda.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currently allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
    """
    # 如果 enabled 参数是布尔类型，则调用 _record_memory_history_legacy 函数
    if isinstance(enabled, bool):
        return _record_memory_history_legacy(enabled, *args, **kwargs)
    else:
        # 否则调用 _record_memory_history_impl 函数
        return _record_memory_history_impl(enabled, *args, **kwargs)


def _record_memory_history_impl(
    enabled: Optional[str] = "all",
    context: Optional[str] = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: Union[Device, int] = None,
):
    # 调用 C++ 函数来记录 CUDA 内存历史，根据参数确定记录的详细内容和条件
    _C._cuda_record_memory_history(enabled, context, stacks, max_entries)


_record_memory_history.__signature__ = signature(_record_memory_history_impl)  # type: ignore[attr-defined]


def _snapshot(device: Union[Device, int] = None):
    """Save a snapshot of CUDA memory state at the time it was called.
    # 在调用时保存 CUDA 内存状态的快照
    """
    # 调用 CUDA 模块的函数 `_cuda_memorySnapshot()` 来获取内存快照信息
    # 此函数返回一个表示状态的字典对象，其结构如下所示：
    # {
    #   'active_bytes': int,        # 活跃内存的字节数
    #   'inactive_bytes': int,      # 非活跃内存的字节数
    #   'reserved_bytes': int,      # 保留内存的字节数
    #   'active_allocations': int,  # 活跃分配的内存块数
    #   'inactive_allocations': int,# 非活跃分配的内存块数
    #   'reserved_allocations': int # 保留分配的内存块数
    # }
    return _C._cuda_memorySnapshot()
# 将 torch.memory._snapshot() 的字典版本序列化为 pickle 格式，保存到文件中
def _dump_snapshot(filename="dump_snapshot.pickle"):
    """
    Save a pickled version of the `torch.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
    """
    # 获取当前的内存快照
    s = _snapshot()
    # 将内存快照对象序列化并保存到指定的 pickle 文件中
    with open(filename, "wb") as f:
        pickle.dump(s, f)


# 将 _snapshot() 返回的快照数据保存为 SVG 格式的文件，用于展示内存使用情况
def _save_segment_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        # 如果快照为空，则获取当前的内存快照
        snapshot = _snapshot()
    # 将内存分段的使用情况写入到指定的 SVG 文件中
    with open(filename, "w") as f:
        f.write(_segments(snapshot))


# 将 _snapshot() 返回的快照数据保存为 SVG 格式的文件，用于展示内存详细信息
def _save_memory_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        # 如果快照为空，则获取当前的内存快照
        snapshot = _snapshot()
    # 将内存使用详细信息写入到指定的 SVG 文件中
    with open(filename, "w") as f:
        f.write(_memory(snapshot))


# 设置 CUDA 内存分配器的环境设置
def _set_allocator_settings(env: str):
    return torch._C._cuda_cudaCachingAllocator_set_allocator_settings(env)


# 获取当前 CUDA 内存分配器的后端类型描述
def get_allocator_backend() -> str:
    r"""Return a string describing the active allocator backend as set by
    ``PYTORCH_CUDA_ALLOC_CONF``. Currently available backends are
    ``native`` (PyTorch's native caching allocator) and `cudaMallocAsync``
    (CUDA's built-in asynchronous allocator).

    .. note::
        See :ref:`cuda-memory-management` for details on choosing the allocator backend.
    """
    return torch._C._cuda_getAllocatorBackend()


class _CUDAAllocator:
    r"""Wrapper over internal CUDA memory allocators."""

    def __init__(self, allocator: torch._C._cuda_CUDAAllocator):
        self._allocator = allocator

    # 返回当前 CUDA 内存分配器对象
    def allocator(self):
        return self._allocator


class CUDAPluggableAllocator(_CUDAAllocator):
    r"""CUDA memory allocator loaded from a so file."""
    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        r"""Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`torch.memory.cuda.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, cudaStream_t stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, cudaStream_t stream);

        .. warning::
            This is currently supported only in unix OSs

        .. note::
            See :ref:`cuda-memory-management` for details on creating and using a custom allocator
        """
        # 使用ctypes加载指定路径下的共享库文件
        allocator = ctypes.CDLL(path_to_so_file)
        # 从加载的共享库中获取分配内存函数的地址，并转换为函数指针
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        # 从加载的共享库中获取释放内存函数的地址，并转换为函数指针
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        # 断言确保获取的函数指针有效
        assert alloc_fn is not None
        assert free_fn is not None
        # 使用获取的分配和释放函数创建一个自定义的CUDA分配器对象
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)
def change_current_allocator(allocator: _CUDAAllocator) -> None:
    r"""Change the currently used memory allocator to be the one provided.

    If the current allocator has already been used/initialized, this function will error.

    Args:
        allocator (torch.cuda.memory._CUDAAllocator): allocator to be set as the active one.

    .. note::
        See :ref:`cuda-memory-management` for details on creating and using a custom allocator
    """
    # 调用 C++ 扩展函数，传入 allocator 对象的底层句柄
    torch._C._cuda_changeCurrentAllocator(allocator.allocator())


def _get_current_allocator() -> _CUDAAllocator:
    r"""Return the allocator being currently used.

    .. note::
        See :ref:`cuda-memory-management` for details on creating and using a custom allocator
    """
    # 调用 C++ 扩展函数获取当前 CUDA 分配器的句柄，并使用 _CUDAAllocator 包装返回
    return _CUDAAllocator(torch._C._cuda_getAllocator())
```