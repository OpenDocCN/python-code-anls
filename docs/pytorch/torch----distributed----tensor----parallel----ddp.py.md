# `.\pytorch\torch\distributed\tensor\parallel\ddp.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型声明
from typing import Any, List, Tuple

import torch.nn as nn
# 从 torch.distributed.tensor.parallel._data_parallel_utils 模块中引入必要的函数
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _flatten_tensor,
    _unflatten_tensor,
)

__all__ = []  # type: ignore[var-annotated]
# 初始化 __all__，用于声明此模块导出的公共接口，此处不做类型注解


def _get_submodule_n_params(module: nn.Module, path: str):
    """
    Get submodule and the direct path of parameter from the module
    从模块中获取子模块和参数的直接路径
    """
    if "." in path:
        path_list = path.split(".")
        parent_module_path = ".".join(path_list[:-1])
        module = module.get_submodule(parent_module_path)
        path = path_list[-1]
    return module, path


def _update_module_param(param_list: List[Tuple[nn.Module, str, nn.Parameter]]):
    """
    Update parameters within the module
    在模块内更新参数
    """
    for item in param_list:
        parent_module, module_path, t = item
        assert hasattr(parent_module, module_path)
        # 删除现有属性并设置新的参数值
        delattr(parent_module, module_path)
        setattr(parent_module, module_path, t)


def _reconstruct_dtensor(module: nn.Module, _input: Any):
    """
    Recontruct DTensor parameters from local tensors
    从本地张量中重建 DTensor 参数
    """
    param_list = []
    # TODO: To add perf optimizations to this iterations
    # 遍历模块中的命名参数
    for name, t in module.named_parameters():
        if hasattr(t, "_st_info"):
            # 使用 _unflatten_tensor 函数重建 DTensor
            dtensor = _unflatten_tensor(t, t._st_info)
            param_list.append((*_get_submodule_n_params(module, name), dtensor))
    _update_module_param(param_list)  # type: ignore[arg-type]


def _localize_dtensor(module: nn.Module, *_: Any):
    """
    Convert DTensor parameters to local tensors
    将 DTensor 参数转换为本地张量
    """
    param_list = []
    # 遍历模块中的命名参数
    for name, param in module.named_parameters():
        # 使用 _flatten_tensor 函数将参数扁平化
        t, sharding_info = _flatten_tensor(param)
        if sharding_info is not None:
            # 将扁平化后的张量包装成 nn.Parameter，并添加 _st_info 属性
            t = nn.Parameter(t)
            t._st_info = sharding_info  # type: ignore[attr-defined]
            param_list.append((*_get_submodule_n_params(module, name), t))
    _update_module_param(param_list)  # type: ignore[arg-type]


def _pre_dp_module_transform(module: nn.Module):
    """
    Enable the composability between Tensor Parallelism (TP) and Data
    Parallelism(DP) in PyTorch when using DDP. We need to convert Parameters which
    are DTensors to local tensors before wrapping with data parallelism API.
    We then register two hooks, one for converting local tensors back to DTensor
    preforward and one to convert DTensors back to tensors after Forward. By
    integrating this way, we avoid any special handling of DTensor parameters by DDP
    and get DTensor's gradients propagated back to DP, e.g. gradient buckets of DDP.

    For now, this API only works with ``DistributedDataParallel``. It will later support
    other DP methods such as FSDP.

    Args:
        module (:class:`nn.Module`):
            Module which has been applied TP on.
    启用 Tensor Parallelism (TP) 和 Data Parallelism (DP) 在 PyTorch 中的组合使用，
    当使用 DDP 时需要将 DTensor 参数转换为本地张量，然后再使用数据并行 API 进行封装。
    我们注册两个钩子函数，一个用于在前向传播前将本地张量转换回 DTensor，另一个用于在
    Forward 后将 DTensor 转换回张量。通过这种集成方式，我们避免了 DDP 对 DTensor
    参数的特殊处理，并确保 DTensor 的梯度能够传播回 DP，例如 DDP 的梯度桶。

    目前，此 API 仅支持 ``DistributedDataParallel``，将来将支持其他 DP 方法，如 FSDP。

    Args:
        module (:class:`nn.Module`):
            已应用 TP 的模块。
    """
    # 调用 _localize_dtensor 函数，将 module 本地化，不进行任何数据并行操作
    _localize_dtensor(module, None, None)
    # TODO: 添加测试用例，确保它对嵌套模块也能正常工作
    # 在 module 上注册一个 forward pre hook，用于在前向传播之前重构数据张量
    module.register_forward_pre_hook(_reconstruct_dtensor)
    # 在 module 上注册一个 forward hook，用于在前向传播过程中本地化数据张量
    module.register_forward_hook(_localize_dtensor)
```