# `.\pytorch\torch\distributed\tensor\parallel\_data_parallel_utils.py`

```
# 导入 functools 模块中的 partial 函数
from functools import partial
# 导入 typing 模块中的 no_type_check, Optional, Tuple
from typing import no_type_check, Optional, Tuple

# 导入 torch 模块
import torch
# 导入 torch.distributed._functional_collectives 模块中的 AsyncCollectiveTensor 类
from torch.distributed._functional_collectives import AsyncCollectiveTensor
# 导入 torch.distributed._tensor 模块中的 DTensor 类
from torch.distributed._tensor import DTensor
# 导入 torch.distributed._tensor.placement_types 模块中的 DTensorSpec 类
from torch.distributed._tensor.placement_types import DTensorSpec


@no_type_check
# 定义 sync_grad_hook 函数，接收 grad 参数，可选的 device_handle 和 compute_stream 参数
def sync_grad_hook(grad, *, device_handle=None, compute_stream=None):
    # 如果 grad 是 AsyncCollectiveTensor 类型的对象
    if isinstance(grad, AsyncCollectiveTensor):
        # 如果 compute_stream 不为 None，则在指定 compute_stream 上执行等待操作
        if compute_stream is not None:
            with device_handle.stream(compute_stream):
                grad = grad.wait()
        else:
            # 否则直接执行等待操作
            grad = grad.wait()

    return grad


# 定义 _flatten_tensor 函数，接收 torch.Tensor 类型的 tensor 参数，返回一个元组
def _flatten_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[DTensorSpec]]:
    # 如果 tensor 是 DTensor 类型的对象
    if isinstance(tensor, DTensor):
        # 将 tensor 的本地张量设置为需要梯度跟踪
        tensor._local_tensor.requires_grad_()
        # 返回本地张量和 tensor 的 spec 属性
        return tensor._local_tensor, tensor._spec
    # 否则返回原 tensor 和 None
    return tensor, None


@no_type_check
# 定义 _unflatten_tensor 函数，接收 tensor, spec 参数，可选的 device_handle 和 compute_stream 参数
def _unflatten_tensor(tensor, spec, *, device_handle=None, compute_stream=None):
    # unflatten 主要用于每次 FSDP 所有聚合参数时调用。
    # 从本地 tensor 创建 DTensor 对象，使用给定的 spec 属性
    result = DTensor.from_local(
        tensor,
        spec.mesh,
        spec.placements,
        run_check=False,
        shape=spec.shape,
        stride=spec.stride,
    )
    # 如果 tensor 需要梯度跟踪
    if tensor.requires_grad:
        # 注册一个钩子函数，用于同步梯度
        tensor.register_hook(
            partial(
                sync_grad_hook,
                device_handle=device_handle,
                compute_stream=compute_stream,
            )
        )
    # 返回创建的 DTensor 对象
    return result
```