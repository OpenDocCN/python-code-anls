# `.\pytorch\torch\mps\__init__.py`

```py
"""
This package enables an interface for accessing MPS (Metal Performance Shaders) backend in Python.
Metal is Apple's API for programming metal GPU (graphics processor unit). Using MPS means that increased
performance can be achieved, by running work on the metal GPU(s).
See https://developer.apple.com/documentation/metalperformanceshaders for more details.
"""
# 导入必要的库和模块
from typing import Union

import torch  # 导入 PyTorch 库
from .. import Tensor  # 导入当前包中的 Tensor 类型


# 是否在错误的分支中运行标记
_is_in_bad_fork = getattr(torch._C, "_mps_is_in_bad_fork", lambda: False)
# 默认的 MPS 生成器，类型为 torch._C.Generator，初始值为 None
_default_mps_generator: torch._C.Generator = None  # type: ignore[assignment]


# 本地辅助函数（非公共或导出函数）
def _get_default_mps_generator() -> torch._C.Generator:
    global _default_mps_generator
    # 如果 _default_mps_generator 为空，则从 C++ 扩展中获取默认生成器
    if _default_mps_generator is None:
        _default_mps_generator = torch._C._mps_get_default_generator()
    return _default_mps_generator


# 返回可用的 MPS 设备数量
def device_count() -> int:
    r"""Returns the number of available MPS devices."""
    return int(torch._C._has_mps and torch._C._mps_is_available())


# 等待所有 MPS 设备上所有流中的内核完成
def synchronize() -> None:
    r"""Waits for all kernels in all streams on a MPS device to complete."""
    return torch._C._mps_deviceSynchronize()


# 返回指定设备（默认为当前 MPS 设备）的随机数生成器状态
def get_rng_state(device: Union[int, str, torch.device] = "mps") -> Tensor:
    r"""Returns the random number generator state as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    return _get_default_mps_generator().get_state()


# 设置指定设备（默认为当前 MPS 设备）的随机数生成器状态
def set_rng_state(
    new_state: Tensor, device: Union[int, str, torch.device] = "mps"
) -> None:
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    _get_default_mps_generator().set_state(new_state_copy)


# 设置生成随机数的种子
def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    # 如果没有 MPS 支持，直接返回
    if not torch._C._has_mps:
        return
    seed = int(seed)
    _get_default_mps_generator().manual_seed(seed)


# 设置生成随机数的种子为随机数
def seed() -> None:
    r"""Sets the seed for generating random numbers to a random number."""
    _get_default_mps_generator().seed()


# 释放由缓存分配器持有的所有未使用的缓存内存，以供其他 GPU 应用程序使用
def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications.
    """
    torch._C._mps_emptyCache()


# 设置每个进程的内存分配比例
def set_per_process_memory_fraction(fraction) -> None:
    r"""Set memory fraction for limiting process's memory allocation on MPS device.
    The allowed value equals the fraction multiplied by recommended maximum device memory
    (obtained from Metal API device.recommendedMaxWorkingSetSize).
    If trying to allocate more than the allowed value in a process, it will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~2. Allowed memory equals total_memory * fraction.

    .. note::
       Passing 0 to fraction means unlimited allocations
       (may cause system failure if out of memory).
       Passing fraction greater than 1.0 allows limits beyond the value
       returned from device.recommendedMaxWorkingSetSize.
    """
    # 检查 fraction 是否为 float 类型，否则抛出类型错误异常
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    # 检查 fraction 是否在合法范围内（0 到 2 之间），否则抛出数值错误异常
    if fraction < 0 or fraction > 2:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~2")

    # 调用 Torch 库中的 C 函数 _mps_setMemoryFraction 设置 MPS 设备的内存分配比例
    torch._C._mps_setMemoryFraction(fraction)
# 返回当前由张量占用的 GPU 内存大小（以字节为单位）
def current_allocated_memory() -> int:
    r"""Returns the current GPU memory occupied by tensors in bytes.

    .. note::
       The returned size does not include cached allocations in
       memory pools of MPSAllocator.
    """
    return torch._C._mps_currentAllocatedMemory()


# 返回 Metal 驱动程序为进程分配的总 GPU 内存大小（以字节为单位）
def driver_allocated_memory() -> int:
    r"""Returns total GPU memory allocated by Metal driver for the process in bytes.

    .. note::
       The returned size includes cached allocations in MPSAllocator pools
       as well as allocations from MPS/MPSGraph frameworks.
    """
    return torch._C._mps_driverAllocatedMemory()


# 返回推荐的 GPU 内存最大工作集大小（以字节为单位）
def recommended_max_memory() -> int:
    r"""Returns recommended max Working set size for GPU memory in bytes.

    .. note::
       Recommended max working set size for Metal.
       returned from device.recommendedMaxWorkingSetSize.
    """
    return torch._C._mps_recommendedMaxMemory()


# 导入模块和类以供外部访问
from . import profiler
from .event import Event

# 暴露给外部的所有接口、类和函数
__all__ = [
    "device_count",
    "get_rng_state",
    "manual_seed",
    "seed",
    "set_rng_state",
    "synchronize",
    "empty_cache",
    "set_per_process_memory_fraction",
    "current_allocated_memory",
    "driver_allocated_memory",
    "Event",
    "profiler",
    "recommended_max_memory",
]
```