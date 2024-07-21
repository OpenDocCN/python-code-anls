# `.\pytorch\torch\cuda\random.py`

```
# mypy: allow-untyped-defs
# 引入需要的类型声明和模块
from typing import Iterable, List, Union

import torch
from .. import Tensor  # 从上级目录导入 Tensor 类型
from . import _lazy_call, _lazy_init, current_device, device_count  # 从当前目录导入函数和变量

__all__ = [  # 将下面定义的函数添加到模块的导出列表中
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

# 返回指定 GPU 的随机数生成器状态作为 ByteTensor
def get_rng_state(device: Union[int, str, torch.device] = "cuda") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()  # 惰性初始化函数
    if isinstance(device, str):
        device = torch.device(device)  # 将字符串设备名称转换为 torch.device 对象
    elif isinstance(device, int):
        device = torch.device("cuda", device)  # 将整数设备索引转换为 CUDA 设备对象
    idx = device.index  # 获取设备索引
    if idx is None:
        idx = current_device()  # 获取当前设备索引
    default_generator = torch.cuda.default_generators[idx]  # 获取默认的 CUDA 随机数生成器
    return default_generator.get_state()  # 返回随机数生成器的当前状态


# 返回所有设备的随机数生成器状态列表
def get_rng_state_all() -> List[Tensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))  # 获取每个设备的随机数生成器状态并添加到结果列表中
    return results  # 返回结果列表


# 设置指定 GPU 的随机数生成器状态
def set_rng_state(
    new_state: Tensor, device: Union[int, str, torch.device] = "cuda"
) -> None:
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
    with torch._C._DisableFuncTorch():
        new_state_copy = new_state.clone(memory_format=torch.contiguous_format)  # 克隆新状态并使用连续的内存格式
    if isinstance(device, str):
        device = torch.device(device)  # 将字符串设备名称转换为 torch.device 对象
    elif isinstance(device, int):
        device = torch.device("cuda", device)  # 将整数设备索引转换为 CUDA 设备对象

    def cb():
        idx = device.index  # 获取设备索引
        if idx is None:
            idx = current_device()  # 获取当前设备索引
        default_generator = torch.cuda.default_generators[idx]  # 获取默认的 CUDA 随机数生成器
        default_generator.set_state(new_state_copy)  # 设置随机数生成器的状态为新状态

    _lazy_call(cb)  # 惰性调用回调函数


# 设置所有设备的随机数生成器状态
def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)  # 为每个设备设置指定的随机数生成器状态


# 为当前 GPU 设置随机数种子
def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    pass  # 空操作，如果 CUDA 不可用，则安全地忽略该函数
    # 将输入的种子值转换为整数类型
    seed = int(seed)
    
    # 定义一个回调函数cb，用于设置当前设备的随机数生成器种子
    def cb():
        # 获取当前设备的索引
        idx = current_device()
        # 获取当前设备的默认随机数生成器
        default_generator = torch.cuda.default_generators[idx]
        # 使用给定的种子值手动设置该随机数生成器的种子
        default_generator.manual_seed(seed)
    
    # 调用_lazy_call函数，传递回调函数cb以及seed=True的标志参数
    _lazy_call(cb, seed=True)
def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    # 将传入的种子转换为整数
    seed = int(seed)

    # 定义回调函数cb，用于设置所有GPU上的随机数生成种子
    def cb():
        # 遍历所有设备的数量
        for i in range(device_count()):
            # 获取当前设备的默认生成器对象
            default_generator = torch.cuda.default_generators[i]
            # 设置该设备的生成器的随机数种子为传入的seed
            default_generator.manual_seed(seed)

    # 调用_lazy_call函数，将回调函数cb传入，seed_all参数设置为True
    _lazy_call(cb, seed_all=True)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    # 定义回调函数cb，用于设置当前GPU的随机数生成种子
    def cb():
        # 获取当前设备的索引
        idx = current_device()
        # 获取当前设备的默认生成器对象
        default_generator = torch.cuda.default_generators[idx]
        # 设置该设备的生成器的随机数种子为随机数
        default_generator.seed()

    # 调用_lazy_call函数，将回调函数cb传入
    _lazy_call(cb)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    """

    # 定义回调函数cb，用于设置所有GPU上的随机数生成种子
    def cb():
        # 初始随机数种子为0
        random_seed = 0
        # 是否已经设置过种子的标志
        seeded = False
        # 遍历所有设备的数量
        for i in range(device_count()):
            # 获取当前设备的默认生成器对象
            default_generator = torch.cuda.default_generators[i]
            # 如果还没有设置过种子，则设置当前设备的生成器的随机数种子为随机数，并记录初始种子
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                # 如果已经设置过种子，则将当前设备的生成器的随机数种子设置为记录的初始种子
                default_generator.manual_seed(random_seed)

    # 调用_lazy_call函数，将回调函数cb传入
    _lazy_call(cb)


def initial_seed() -> int:
    r"""Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    """
    # 确保CUDA已经初始化
    _lazy_init()
    # 获取当前设备的索引
    idx = current_device()
    # 获取当前设备的默认生成器对象
    default_generator = torch.cuda.default_generators[idx]
    # 返回当前设备的生成器的初始种子
    return default_generator.initial_seed()
```