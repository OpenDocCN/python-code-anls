# `.\pytorch\torch\distributed\fsdp\_fsdp_extensions.py`

```py
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法的装饰器
from typing import Any, List, Optional, Tuple  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
from torch.distributed._shard.sharded_tensor.api import ShardedTensor  # 导入ShardedTensor类
from torch.distributed._shard.sharded_tensor.shard import Shard  # 导入Shard类
from torch.distributed._tensor import DeviceMesh, DTensor  # 导入DeviceMesh和DTensor类
from torch.distributed.fsdp._shard_utils import (  # 导入FSDP扩展相关的工具函数
    _all_gather_dtensor,
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
)

class FSDPExtensions(ABC):
    """
    This enables some customizable hooks to enable composability with tensor
    parallelism. To activate these hooks, use :func:`_set_fsdp_extensions` to
    set a custom :class:`FSDPExtensions` that implements the hooks.
    """

    @abstractmethod
    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """E.g. converting ``DistributedTensor`` to local tensor."""
        ...

    @abstractmethod
    def post_unflatten_transform(
        self,
        tensor: torch.Tensor,
        param_extension: Any,
    ) -> torch.Tensor:
        """E.g. converting local tensor to ``DistributedTensor``."""
        ...

    @abstractmethod
    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Shards a tensor to chunks and returns the local chunk."""
        ...

    @abstractmethod
    def chunk_dtensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Shards a tensor/DTensor to DTensor and returns the local DTensor."""
        ...

    @abstractmethod
    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Shard]]:
        """
        This is to be called before loading a *sharded* model state dict and
        should return the tensor and list of shards from which to load data.
        """
        ...

    @abstractmethod
    def all_gather_dtensor(
        self,
        tensor: DTensor,
        parent_mesh: Optional[DeviceMesh],
    ) -> torch.Tensor:
        """
        This is to be called before loading a *sharded* DTensor state dict.
        This gathers tensor in FSDP dimension and returns local tensor of
        TP DTensor.
        """
        ...

_extensions: Optional[FSDPExtensions] = None  # 定义一个可选的FSDPExtensions对象，初始为None


def _set_fsdp_extensions(flattener: FSDPExtensions) -> None:
    global _extensions  # 声明全局变量_extensions
    _extensions = flattener  # 将传入的flattener对象赋值给_extensions


def _ext_pre_flatten_transform(
    tensor: torch.Tensor,
    fsdp_extension: Optional[FSDPExtensions] = None,
) -> Tuple[torch.Tensor, Optional[Any]]:
    # 此函数用于执行预扁平化转换操作，接收一个torch.Tensor对象和可选的FSDPExtensions对象作为参数
    # 如果存在 fsdp_extension，则执行以下操作
    if fsdp_extension is not None:
        # 调用 fsdp_extension 的 pre_flatten_transform 方法，对 tensor 进行预扁平化转换
        new_tensor, param_extension = fsdp_extension.pre_flatten_transform(tensor)
        # 如果 pre_flatten_transform 返回的 param_extension 不为 None，则返回新的张量和参数扩展
        if param_extension is not None:
            return new_tensor, param_extension
    
    # 如果 fsdp_extension 为 None 或 pre_flatten_transform 没有返回参数扩展，直接返回原始的 tensor 和 None
    return tensor, None
# 根据给定的参数对张量进行后处理的扩展函数
def _ext_post_unflatten_transform(
    tensor: torch.Tensor,
    param_extension: Any,
    fsdp_extension: Optional[FSDPExtensions] = None,
) -> torch.Tensor:
    # 如果存在 FSDP 扩展和参数扩展，则调用 FSDP 扩展对象的后处理函数
    if fsdp_extension is not None and param_extension is not None:
        return fsdp_extension.post_unflatten_transform(tensor, param_extension)
    # 否则直接返回原张量
    return tensor


# 根据给定的参数对张量进行分块的扩展函数
def _ext_chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
    fsdp_extension: Optional[FSDPExtensions] = None,
) -> torch.Tensor:
    # 根据 FSDP 扩展的情况选择相应的张量分块函数
    chunk_tensor_fn = (
        fsdp_extension.chunk_tensor
        if fsdp_extension is not None
        else _create_chunk_sharded_tensor
    )
    # 调用选定的张量分块函数来进行张量分块操作
    return chunk_tensor_fn(
        tensor,
        rank,
        world_size,
        num_devices_per_node,
        pg,
    )


# 根据给定的参数对分布式张量进行分块的扩展函数
def _ext_chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
    fsdp_extension: Optional[FSDPExtensions] = None,
) -> torch.Tensor:
    # 根据 FSDP 扩展的情况选择相应的分布式张量分块函数
    chunk_dtensor_fn = (
        fsdp_extension.chunk_dtensor
        if fsdp_extension is not None
        else _create_chunk_dtensor
    )
    # 调用选定的分布式张量分块函数来进行张量分块操作
    return chunk_dtensor_fn(
        tensor,
        rank,
        device_mesh,
    )


# 根据给定的参数对加载状态字典前进行预处理的扩展函数
def _ext_pre_load_state_dict_transform(
    tensor: torch.Tensor,
    fsdp_extension: Optional[FSDPExtensions] = None,
) -> Tuple[torch.Tensor, List[Shard]]:
    # 如果存在 FSDP 扩展，则调用 FSDP 扩展对象的状态字典预处理函数
    if fsdp_extension is not None:
        return fsdp_extension.pre_load_state_dict_transform(tensor)

    # 如果不存在 FSDP 扩展，则断言张量类型为 ShardedTensor，并返回本地分片
    assert type(tensor) is ShardedTensor
    shards = tensor.local_shards()
    return (tensor, shards)


# 根据给定的参数对分布式张量进行全局收集的扩展函数
def _ext_all_gather_dtensor(
    tensor: DTensor,
    parent_mesh: Optional[DeviceMesh],
    fsdp_extension: Optional[FSDPExtensions] = None,
) -> torch.Tensor:
    # 根据 FSDP 扩展的情况选择相应的分布式张量全局收集函数
    all_gather_dtensor_fn = (
        fsdp_extension.all_gather_dtensor
        if fsdp_extension is not None
        else _all_gather_dtensor
    )
    # 调用选定的分布式张量全局收集函数来进行全局收集操作
    return all_gather_dtensor_fn(tensor, parent_mesh)
```