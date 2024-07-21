# `.\pytorch\torch\cuda\_utils.py`

```py
# 从 typing 模块导入 Any 类型
from typing import Any

# 导入 PyTorch 库
import torch

# 从 torch._utils 模块导入 _get_device_index，别名为 _torch_get_device_index
# _get_device_index 已移动到 torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


# 定义函数 _get_device_index，接受 device 参数（任意类型），可选参数 optional 和 allow_cpu
# 返回一个整数表示设备索引
def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.
    
    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.
    
    If :attr:`device` is a Python integer, it is returned as is.
    
    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    # 如果 device 是整数类型，直接返回该整数
    if isinstance(device, int):
        return device
    # 如果 device 是字符串类型，将其转换为 torch.device 对象
    if isinstance(device, str):
        device = torch.device(device)
    # 如果 device 是 torch.device 对象
    if isinstance(device, torch.device):
        # 如果 allow_cpu 为 True，则允许 CPU 设备
        if allow_cpu:
            # 如果 device 类型不是 "cuda" 或 "cpu"，抛出 ValueError 异常
            if device.type not in ["cuda", "cpu"]:
                raise ValueError(f"Expected a cuda or cpu device, but got: {device}")
        # 如果不允许 CPU 设备，并且 device 类型不是 "cuda"，抛出 ValueError 异常
        elif device.type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
    # 如果当前不是 Torch 脚本模式（即未进行 Torch JIT 编译）
    if not torch.jit.is_scripting():
        # 如果 device 是 torch.cuda.device 对象，返回其索引
        if isinstance(device, torch.cuda.device):
            return device.idx
    # 调用 _torch_get_device_index 函数处理剩余情况，返回设备索引
    return _torch_get_device_index(device, optional, allow_cpu)
```