# `.\pytorch\torch\xpu\random.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import Iterable, List, Union

# 导入 PyTorch 库
import torch
from .. import Tensor
from . import _lazy_call, _lazy_init, current_device, device_count

# 返回指定 GPU 的随机数生成器状态作为 ByteTensor
def get_rng_state(device: Union[int, str, torch.device] = "xpu") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    # 惰性初始化
    _lazy_init()
    
    # 根据参数类型转换设备表示
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)
        
    # 获取设备索引
    idx = device.index
    if idx is None:
        idx = current_device()
        
    # 获取默认生成器并返回其状态
    default_generator = torch.xpu.default_generators[idx]
    return default_generator.get_state()


# 返回所有设备的随机数生成器状态列表
def get_rng_state_all() -> List[Tensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))
    return results


# 设置指定 GPU 的随机数生成器状态
def set_rng_state(
    new_state: Tensor, device: Union[int, str, torch.device] = "xpu"
) -> None:
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).
    """
    # 禁用 Torch 函数以确保状态设置的连续性
    with torch._C._DisableFuncTorch():
        new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    
    # 根据参数类型转换设备表示
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)

    # 回调函数，设置设备的生成器状态
    def cb():
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.set_state(new_state_copy)

    _lazy_call(cb)


# 设置所有设备的随机数生成器状态
def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    # 遍历所有设备，逐个设置生成器状态
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


# 为当前 GPU 设置种子以生成随机数
def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    # 将输入种子转换为整数类型
    seed = int(seed)
    # 定义一个函数 cb()
    def cb():
        # 调用 current_device() 函数获取当前设备的索引
        idx = current_device()
        # 根据设备索引从 torch.xpu.default_generators 中获取默认生成器
        default_generator = torch.xpu.default_generators[idx]
        # 使用指定的种子设置生成器的随机种子
        default_generator.manual_seed(seed)
    
    # 调用 _lazy_call 函数，传入 cb() 函数作为回调函数，并设置 seed=True
    _lazy_call(cb, seed=True)
# 设置所有 GPU 上生成随机数的种子
def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    # 将输入的种子值转换为整数
    seed = int(seed)

    def cb():
        # 遍历所有设备的数量
        for i in range(device_count()):
            # 获取第 i 个设备的默认生成器
            default_generator = torch.xpu.default_generators[i]
            # 手动设置该设备的随机数种子为给定的种子值
            default_generator.manual_seed(seed)

    # 调用内部函数 `_lazy_call`，传递回调函数 `cb` 并设置 `seed_all=True`
    _lazy_call(cb, seed_all=True)


# 为当前 GPU 设置随机数种子
def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb():
        # 获取当前设备的索引
        idx = current_device()
        # 获取当前设备的默认生成器
        default_generator = torch.xpu.default_generators[idx]
        # 生成一个随机数种子
        default_generator.seed()

    # 调用内部函数 `_lazy_call`，传递回调函数 `cb`
    _lazy_call(cb)


# 为所有 GPU 设置随机数种子
def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.
    """

    def cb():
        # 初始随机种子为 0
        random_seed = 0
        # 是否已经为第一个设备设置了种子
        seeded = False
        # 遍历所有设备
        for i in range(device_count()):
            # 获取第 i 个设备的默认生成器
            default_generator = torch.xpu.default_generators[i]
            # 如果尚未为设备设置种子，则设置种子并记录初始种子
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                # 否则，为该设备手动设置种子为记录的初始种子
                default_generator.manual_seed(random_seed)

    # 调用内部函数 `_lazy_call`，传递回调函数 `cb`
    _lazy_call(cb)


# 返回当前 GPU 的当前随机种子
def initial_seed() -> int:
    r"""Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes XPU.
    """
    # 懒初始化 XPU
    _lazy_init()
    # 获取当前设备的索引
    idx = current_device()
    # 获取当前设备的默认生成器
    default_generator = torch.xpu.default_generators[idx]
    # 返回当前设备的初始随机种子
    return default_generator.initial_seed()


# 导出所有的函数名，使它们可以被模块导入
__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]
```