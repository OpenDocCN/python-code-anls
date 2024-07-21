# `.\pytorch\torch\utils\_device.py`

```
# mypy: allow-untyped-defs
# 引入类型提示模块 Optional
from typing import Optional
# 引入 PyTorch 库
import torch
# 引入 TorchFunctionMode 类
from torch.overrides import TorchFunctionMode
# 引入上下文管理器装饰器
from torch.utils._contextlib import context_decorator
# 引入 functools 模块
import functools

# 当前设备，默认为空
CURRENT_DEVICE: Optional[torch.device] = None

# 使用 functools 的 lru_cache 装饰器，缓存 _device_constructors 函数的结果
@functools.lru_cache(1)
def _device_constructors():
    # 返回包含各种 Torch 张量构造函数的字典
    return {
        # 标准的张量构造函数
        torch.empty,
        torch.empty_permuted,
        torch.empty_strided,
        torch.empty_quantized,
        torch.ones,
        torch.arange,
        torch.bartlett_window,
        torch.blackman_window,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.full,
        torch.fill,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.nested.nested_tensor,  # torch.nested 模块的嵌套张量构造函数
        # 该函数实际上不接受设备参数
        # torch.normal,
        torch.ones,
        torch.rand,
        torch.randn,
        torch.randint,
        torch.randperm,
        torch.range,
        torch.sparse_coo_tensor,
        torch.sparse_compressed_tensor,
        torch.sparse_csr_tensor,
        torch.sparse_csc_tensor,
        torch.sparse_bsr_tensor,
        torch.sparse_bsc_tensor,
        torch.tril_indices,
        torch.triu_indices,
        torch.vander,
        torch.zeros,
        torch.asarray,
        # 奇怪的张量构造函数
        torch.tensor,
        torch.as_tensor,
        torch.scalar_tensor,
        torch.asarray,
    }

# NB: 此类直接从 torch/csrc/Device.cpp 中的 C++ 调用
# 设备上下文管理器，继承自 TorchFunctionMode 类
class DeviceContext(TorchFunctionMode):
    def __init__(self, device):
        self.device = torch.device(device)

    # 进入上下文时调用
    def __enter__(self):
        global CURRENT_DEVICE
        # 保存旧的当前设备
        self.old_device = CURRENT_DEVICE
        # 将全局当前设备设置为新的设备
        CURRENT_DEVICE = self.device
        return super().__enter__()

    # 退出上下文时调用
    def __exit__(self, exc_type, exc_val, exc_tb):
        global CURRENT_DEVICE
        # 恢复旧的当前设备
        CURRENT_DEVICE = self.old_device
        return super().__exit__(exc_type, exc_val, exc_tb)

    # 实现 __torch_function__ 方法，用于 Torch 函数的自定义行为
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # 如果函数在 _device_constructors 中，并且未指定设备参数
        if func in _device_constructors() and kwargs.get('device') is None:
            # 将设备参数设为当前设备
            kwargs['device'] = self.device
        return func(*args, **kwargs)

# NB: 此函数直接从 torch/csrc/Device.cpp 中的 C++ 调用
# 设备装饰器函数，接受设备和函数作为参数，返回上下文装饰器
def device_decorator(device, func):
    return context_decorator(lambda: device, func)

# 设置默认设备的函数装饰器
def set_device(device):
    """
    在装饰的函数内部设置默认设备。

    如果要将其作为上下文管理器使用，请直接使用 device 作为上下文管理器，例如 ``with torch.device(device)``。
    """
    return lambda func: device_decorator(torch.device(device), func)
```