# `.\pytorch\torch\nn\utils\convert_parameters.py`

```
# 导入必要的库，从 typing 模块导入 Iterable 和 Optional 类型
from typing import Iterable, Optional

# 导入 PyTorch 库
import torch


# 将参数的可迭代集合展平成单个向量
def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Flatten an iterable of parameters into a single vector.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # 用于标记参数所在的设备
    param_device = None

    # 用于存储展平后的参数向量
    vec = []
    for param in parameters:
        # 确保所有参数位于相同的设备上
        param_device = _check_param_device(param, param_device)

        # 将参数展平并加入到向量中
        vec.append(param.view(-1))
    return torch.cat(vec)


# 将向量的切片复制到参数的可迭代集合中
def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Copy slices of a vector into an iterable of parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
    # 确保 vec 是 Tensor 类型
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")

    # 用于标记参数所在的设备
    param_device = None

    # 指针，用于迭代向量中的切片，每个参数对应一个切片
    pointer = 0
    for param in parameters:
        # 确保所有参数位于相同的设备上
        param_device = _check_param_device(param, param_device)

        # 参数的长度
        num_param = param.numel()

        # 切片向量，重塑并替换参数的旧数据
        param.data = vec[pointer : pointer + num_param].view_as(param).data

        # 更新指针位置
        pointer += num_param


# 检查参数是否位于同一设备上
def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""Check if the parameters are located on the same device.

    Currently, the conversion between model parameters and single vector form is not supported
    for multiple allocations, e.g. parameters in different GPUs/PrivateUse1s, or mixture of CPU/GPU/PrivateUse1.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """
    # 支持的设备类型列表
    support_device_types = ["cuda", torch._C._get_privateuse1_backend_name()]

    # 如果第一个参数，确定设备
    if old_param_device is None:
        old_param_device = (
            param.get_device() if param.device.type in support_device_types else -1
        )

    return old_param_device
    else:
        # 初始化警告标志为 False
        warn = False
        # 检查参数的设备类型是否在支持的设备类型列表中
        if (
            param.device.type in support_device_types
        ):  # 如果在同一GPU或PrivateUse1设备上
            # 若参数的设备与旧参数的设备不同，则设置警告标志为 True
            warn = param.get_device() != old_param_device
        else:  # 如果在CPU上
            # 若旧参数的设备不是 -1，则设置警告标志为 True
            warn = old_param_device != -1
        # 如果警告标志为 True，则抛出类型错误异常
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    # 返回旧参数的设备
    return old_param_device
```