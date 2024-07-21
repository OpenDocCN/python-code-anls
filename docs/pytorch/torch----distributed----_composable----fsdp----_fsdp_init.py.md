# `.\pytorch\torch\distributed\_composable\fsdp\_fsdp_init.py`

```py
import itertools
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, init_device_mesh
from torch.distributed.device_mesh import _get_device_handle
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from ._fsdp_common import _is_composable_with_fsdp, FSDPMeshInfo, HSDPMeshInfo
from ._fsdp_state import _get_module_fsdp_state


def _get_post_forward_mesh_info(
    reshard_after_forward: Union[bool, int], mesh_info: FSDPMeshInfo
) -> Optional[FSDPMeshInfo]:
    shard_mesh_size = mesh_info.shard_mesh_size
    if not isinstance(reshard_after_forward, (bool, int)):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )
    # NOTE: `isinstance(False, int)` returns `True`.
    if not isinstance(reshard_after_forward, bool) and isinstance(
        reshard_after_forward, int
    ):
        if (
            reshard_after_forward < 1
            or reshard_after_forward > shard_mesh_size
            or shard_mesh_size % reshard_after_forward != 0
        ):
            raise ValueError(
                "If passing reshard_after_forward as an int, it should be a "
                f"factor of {shard_mesh_size}, not {reshard_after_forward}"
            )
        elif reshard_after_forward == 1:
            reshard_after_forward = False
        elif reshard_after_forward == shard_mesh_size:
            reshard_after_forward = True
    post_forward_mesh_info = None
    if reshard_after_forward is True:
        post_forward_mesh_info = mesh_info
    elif reshard_after_forward is not False:  # int case
        # For HSDP, we can flatten the two replicate dims into the 0th dim
        post_forward_mesh_tensor = mesh_info.mesh.mesh.view(-1, reshard_after_forward)
        post_forward_mesh = DeviceMesh(
            mesh_info.mesh.device_type, post_forward_mesh_tensor
        )
        post_forward_mesh_info = HSDPMeshInfo(
            post_forward_mesh, shard_mesh_dim=1, replicate_mesh_dim=0
        )
    return post_forward_mesh_info


def _init_default_fully_shard_mesh() -> DeviceMesh:
    """Default to global CUDA mesh if possible else global CPU mesh."""
    if not dist.distributed_c10d.is_initialized():
        # 初始化分布式处理组，如果尚未初始化
        dist.distributed_c10d.init_process_group()
    # 获取默认的分布式处理组
    default_pg = dist.distributed_c10d._get_default_group()
    # 确定设备类型为CUDA如果可用，否则为CPU
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用默认进程组的大小初始化设备网格
    mesh = init_device_mesh(device_type, mesh_shape=(default_pg.size(),))
    return mesh


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    if mesh.device_type == "cpu":
        # 如果设备类型是CPU，则返回CPU设备
        return torch.device("cpu")
    # 否则，获取设备句柄并返回相应设备
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())
def _get_managed_modules(root_module: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    # Track visisted modules to avoid visiting shared modules multiple times
    visited_modules: Set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        """
        Runs a DFS to collect managed modules, not recursing into modules with
        a non-composable API or ``fully_shard`` already applied.
        """
        # 检查模块是否可以与FSDP协同工作，如果不能则返回
        if not _is_composable_with_fsdp(module):
            return
        # 如果模块不是根模块，并且已经应用了`fully_shard`，则返回，不进行进一步递归
        elif module is not root_module and _get_module_fsdp_state(module) is not None:
            return  # nested `fully_shard` module
        # 将当前模块标记为已访问
        visited_modules.add(module)
        # 遍历当前模块的子模块
        for submodule in module.children():
            if submodule not in visited_modules:
                dfs(submodule)
        # 将当前模块添加到结果列表中
        modules.append(module)

    # 调用深度优先搜索函数开始收集管理的模块
    dfs(root_module)
    return modules


def _get_managed_states(
    modules: List[nn.Module],
) -> Tuple[List[nn.Parameter], List[torch.Tensor]]:
    params: List[nn.Parameter] = []
    buffers: List[torch.Tensor] = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params: Set[nn.Parameter] = set()
    visited_buffers: Set[torch.Tensor] = set()

    # 遍历所有管理的模块
    for module in modules:
        # 遍历当前模块的参数，不递归
        for param in module.parameters(recurse=False):
            if param not in visited_params:
                # 将参数添加到参数列表，并标记为已访问
                params.append(param)
                visited_params.add(param)
        # 遍历当前模块的缓冲区，不递归
        for buffer in module.buffers(recurse=False):
            if buffer not in visited_buffers:
                # 将缓冲区添加到缓冲区列表，并标记为已访问
                buffers.append(buffer)
                visited_buffers.add(buffer)

    return params, buffers


def _move_states_to_device(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    device: torch.device,
) -> None:
    """
    We have FSDP move states to device for simpler and faster initialization
    since FSDP almost always uses CUDA for training. We move parameters/buffers
    rather than modules since modules to support ignoring parameters/buffers in
    the future.
    """
    # Follow the logic in `nn.Module._apply`
    # 遵循`nn.Module._apply`中的逻辑，将状态（参数和缓冲区）移动到指定设备上
    # 遍历 params 和 buffers 中的张量
    for tensor in itertools.chain(params, buffers):
        # 检查张量是否在目标设备上或者是 meta 设备上的张量
        if tensor.device == device or tensor.device.type == "meta":
            # 如果是 meta 设备上的张量，延迟初始化时保持在 meta 设备上
            continue
        
        # 检查张量是否是 DTensor 类型
        if isinstance(tensor, DTensor):
            # 获取 DTensor 的设备网格类型
            if (dtensor_mesh_type := tensor.device_mesh.device_type) != device.type:
                # 抛出数值错误，要求 DTensor 的网格类型与 FSDP 网格类型相同
                raise ValueError(
                    "Requires DTensor to have mesh of the same type as the FSDP mesh "
                    f"but got {dtensor_mesh_type} for DTensor and {device.type} for FSDP"
                )
            # 如果设备网格类型匹配不上，则引发断言错误
            raise AssertionError(
                f"Expects DTensor to be moved to {dtensor_mesh_type} but got {tensor.device}"
            )
        
        # 检查张量是否是可追踪包装器的子类
        if is_traceable_wrapper_subclass(tensor):
            # 对于可追踪包装器的子类，使用 torch.no_grad() 来避免 autograd 增加 C++ 引用计数 1 次
            with torch.no_grad():
                # 将张量移动到目标设备上，并封装为 nn.Parameter
                tensor_on_device = nn.Parameter(tensor.to(device))
            # 使用 torch.utils.swap_tensors 将原张量和新张量交换
            torch.utils.swap_tensors(tensor, tensor_on_device)
        else:
            # 将张量的数据移到目标设备上
            tensor.data = tensor.to(device)
```