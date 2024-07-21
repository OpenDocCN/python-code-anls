# `.\pytorch\torch\distributed\tensor\parallel\input_reshard.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
from functools import partial  # 导入 functools 模块的 partial 函数，用于创建函数的偏函数
from typing import Any, Optional, Tuple  # 导入类型提示模块，用于类型注解

import torch  # 导入 PyTorch 库
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard  # 导入 PyTorch 的分布式张量相关类


__all__ = [
    "input_reshard",
]


def input_reshard(
    module: torch.nn.Module,
    tp_device_mesh: DeviceMesh,
    input_reshard_dim: Optional[int] = None,
) -> torch.nn.Module:
    """
    Register hooks to an nn.Module for input resharding, enabling sharding and restoration during backward computation.

    Register hooks to an nn.Module with input resharding so that we can shard
    per the given `tp_device_mesh` and `input_reshard_dim` and restore the
    input back when recomputing the activations in the backward. The reason
    why we can do this is that for Tensor Parallel(TP), the input are same
    across all TP ranks.

    Args:
        module (:class:`nn.Module`):
            Module to be registered with input resharding.
        tp_device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for Tensor Parallel.
        input_reshard_dim (Optional[int]):
            The dimension of where we perform the sharding
            of input. If set None, there is no sharding of input.
            Default: None

    Return:
        A :class:`nn.Module` object registered with TP input resharding.
    """
    cx: Optional[torch.autograd.graph.saved_tensors_hooks] = None  # 声明一个可选类型的 saved_tensors_hooks 对象

    def input_reshard_forward_pre_hook(_: torch.nn.Module, _i: Tuple[Any, ...]) -> None:
        # 前向预处理钩子函数，用于注册输入重分片
        saved_tensor_hooks = torch.autograd.graph.saved_tensors_hooks(
            partial(_pack_hook_tp, tp_device_mesh, input_reshard_dim),
            partial(_unpack_hook_tp, tp_device_mesh, input_reshard_dim),
        )
        saved_tensor_hooks.__enter__()  # 进入上下文
        nonlocal cx
        cx = saved_tensor_hooks  # 记录 saved_tensor_hooks 对象

    def input_reshard_backward_hook(
        _: torch.nn.Module, _i: Tuple[Any, ...], _o: Any
    ) -> Any:
        # 反向钩子函数，用于在反向传播时恢复输入
        nonlocal cx
        cx.__exit__()  # 退出上下文

    if input_reshard_dim is None:
        return module
    module.register_forward_pre_hook(input_reshard_forward_pre_hook)  # 注册前向预处理钩子
    module.register_forward_hook(input_reshard_backward_hook)  # 注册前向钩子
    return module


def _pack_hook_tp(
    mesh: DeviceMesh, input_reshard_dim: int, x: torch.Tensor
) -> Any:  # noqa: D401
    """Hook function called after FWD to shard input."""
    # 输入重分片的钩子函数，用于在前向传播后执行分片
    if isinstance(x, DTensor) and all(p.is_replicate() for p in x._spec.placements):
        return x.redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(x, device_mesh=mesh)
            .redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
            .to_local()
        )
    else:
        return x


def _unpack_hook_tp(
    mesh: DeviceMesh, input_reshard_dim: int, x: Any
) -> Any:  # noqa: D401
    """Hook function called after BWD to restore input."""
    # 输入恢复的钩子函数，用于在反向传播后恢复输入
    if isinstance(x, DTensor) and x.is_replicated():
        return x.to_local()
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return x.to('cpu')
    else:
        return x
    # 定义函数的参数：
    # - mesh: 表示设备上的网格数据结构，类型为 DeviceMesh
    # - input_reshard_dim: 表示输入数据的重分片维度，是一个整数类型
    # - x: 表示任意类型的输入数据，可以是任何数据类型
    mesh: DeviceMesh, input_reshard_dim: int, x: Any
def restore_input_before_activation_recompute(x: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Hook function called before activation recomputing in BWD to restore input."""
    # 如果输入是 DTensor 类型，并且只有一个位置，且该位置是分片的
    if (
        isinstance(x, DTensor)
        and len(x._spec.placements) == 1
        and x._spec.placements[0].is_shard()
    ):
        # 返回重新分布后的 DTensor 对象，设备网格为 mesh，使用 Replicate() 策略
        return x.redistribute(device_mesh=mesh, placements=[Replicate()])
    # 如果输入不是 DTensor 类型，并且是 torch.Tensor 类型，并且元素数量大于等于 mesh 的大小
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        # 创建一个新的 DTensor 对象，从本地 tensor x 创建，设备网格为 mesh，使用 Shard(input_reshard_dim) 策略
        # 然后重新分布到设备网格为 mesh，使用 Replicate() 策略，并转换为本地 tensor
        return (
            DTensor.from_local(
                x, device_mesh=mesh, placements=[Shard(input_reshard_dim)]
            )
            .redistribute(device_mesh=mesh, placements=[Replicate()])
            .to_local()
        )
    else:
        # 其他情况下直接返回输入 x
        return x
```