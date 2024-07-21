# `.\pytorch\torch\xpu\_utils.py`

```py
# 引入了用于类型提示的 Any 类型
from typing import Any

# 引入了 PyTorch 库
import torch

# 从 torch._utils 模块中导入 _get_device_index，并重命名为 _torch_get_device_index
# 该函数已经从 torch._utils 中移到了 torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


# 定义一个函数 _get_device_index，接受一个 device 参数，可选的 optional 参数，默认为 False，
# 以及 allow_cpu 参数，默认为 False，返回一个整数
def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a XPU device. Note that for a XPU device without a specified index,
    i.e., ``torch.device('xpu')``, this will return the current default XPU
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default XPU
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
        # 如果 allow_cpu 为 True，接受 CPU 设备，返回 -1
        if allow_cpu:
            if device.type not in ["xpu", "cpu"]:
                raise ValueError(f"Expected a xpu or cpu device, but got: {device}")
        # 否则，要求必须是 XPU 设备
        elif device.type != "xpu":
            raise ValueError(f"Expected a xpu device, but got: {device}")
    
    # 如果当前不在 Torch 的脚本执行环境下
    if not torch.jit.is_scripting():
        # 如果 device 是 torch.xpu.device 类型的对象，返回其索引号 device.idx
        if isinstance(device, torch.xpu.device):
            return device.idx
    
    # 调用 _torch_get_device_index 函数，传入 device, optional, allow_cpu 参数
    return _torch_get_device_index(device, optional, allow_cpu)
```