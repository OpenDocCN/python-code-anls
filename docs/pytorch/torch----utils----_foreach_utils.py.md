# `.\pytorch\torch\utils\_foreach_utils.py`

```py
# 导入必要的模块和类型别名
from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad
from typing_extensions import TypeAlias

# 定义类型别名
TensorListList: TypeAlias = List[List[Optional[Tensor]]]
Indices: TypeAlias = List[int]

# foreach kernels 支持的设备类型列表
def _get_foreach_kernels_supported_devices() -> List[str]:
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]

# fused kernels 支持的设备类型列表
def _get_fused_kernels_supported_devices() -> List[str]:
    r"""Return the device type list that supports fused kernels in optimizer."""
    return ["mps", "cuda", "xpu", "cpu", torch._C._get_privateuse1_backend_name()]

# 这个实用函数将张量按设备和数据类型分组，这在将张量发送到 foreach 实现之前非常有用，foreach 要求张量在同一设备和数据类型上。
# 如果 tensorlistlist 包含多个张量列表，则假设但不验证以下情况：
#   - tensorlists 可能为 None
#   - 第一个指定列表中的所有张量不能为 None
#   - 对于给定的索引 i，指定的所有 tensorlist[i] 在数据类型和设备上匹配
# with_indices (bool, optional): 是否在字典条目的最后一个列表中跟踪先前的索引。
#   如果 tensorlists 中有 None 或字面量正在散布出去，则这很方便。
#   虽然在结果拆分后的 tensorlists 中对张量进行突变会将更改传播回原始输入 tensorlists，但更改 None/字面量不会传播，
#   可能需要手动传播。查看 torch/optim/sgd.py 以获取示例。
@no_grad()
def _group_tensors_by_device_and_dtype(
    tensorlistlist: TensorListList,
    with_indices: bool = False,
) -> Dict[Tuple[torch.device, torch.dtype], Tuple[TensorListList, Indices]]:
    return torch._C._group_tensors_by_device_and_dtype(tensorlistlist, with_indices)

# 检查设备是否支持 foreach 操作
def _device_has_foreach_support(device: torch.device) -> bool:
    return device.type in (_get_foreach_kernels_supported_devices() + ["cpu"]) and not torch.jit.is_scripting()

# 检查设备和张量列表是否支持 foreach 操作
def _has_foreach_support(tensors: List[Tensor], device: torch.device) -> bool:
    return _device_has_foreach_support(device) and all(t is None or type(t) in _foreach_supported_types for t in tensors)
```