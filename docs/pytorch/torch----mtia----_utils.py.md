# `.\pytorch\torch\mtia\_utils.py`

```
# 引入 Any 类型用于参数类型提示
from typing import Any

# 引入 Torch 库
import torch

# 导入 _get_device_index 函数，已从 torch._utils 中移动到 torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


# 定义函数 _get_device_index，用于获取设备索引
def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a MTIA device. Note that for a MTIA device without a specified index,
    i.e., ``torch.device('mtia')``, this will return the current default MTIA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default MTIA
    device if :attr:`optional` is ``True``.
    """
    # 如果 device 是整数类型，直接返回
    if isinstance(device, int):
        return device
    # 如果 device 是字符串类型，转换为 torch.device 对象
    if isinstance(device, str):
        device = torch.device(device)
    # 如果 device 是 torch.device 对象
    if isinstance(device, torch.device):
        # 如果允许 CPU 并且设备类型不是 "mtia" 或 "cpu"，则抛出异常
        if allow_cpu:
            if device.type not in ["mtia", "cpu"]:
                raise ValueError(f"Expected a mtia or cpu device, but got: {device}")
        # 如果不允许 CPU 并且设备类型不是 "mtia"，则抛出异常
        elif device.type != "mtia":
            raise ValueError(f"Expected a mtia device, but got: {device}")
    # 如果不处于 Torch 脚本模式下
    if not torch.jit.is_scripting():
        # 如果 device 是 torch.mtia.device 类型，返回其索引
        if isinstance(device, torch.mtia.device):
            return device.idx
    # 调用 _torch_get_device_index 函数，返回其结果
    return _torch_get_device_index(device, optional, allow_cpu)
```