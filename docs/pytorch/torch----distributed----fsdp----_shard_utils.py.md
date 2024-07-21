# `.\pytorch\torch\distributed\fsdp\_shard_utils.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import copy  # 导入copy模块，用于复制对象
import itertools  # 导入itertools模块，用于生成迭代器
import math  # 导入math模块，用于数学运算
from typing import Optional  # 从typing模块导入Optional类型

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
from torch.distributed import distributed_c10d  # 导入PyTorch分布式c10d模块
from torch.distributed._shard.sharded_tensor import (  # 导入ShardedTensor相关模块
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import ShardMetadata  # 导入ShardMetadata模块
from torch.distributed._tensor import (  # 导入分布式tensor相关模块
    DeviceMesh,
    DTensor,
    Replicate,
    Shard as DShard,
)


def _get_remote_device_str(rank, device_type, num_devices_per_node):
    # 根据设备类型和排名生成远程设备字符串表示
    if device_type.lower() == "cpu":
        return f"rank:{rank}/{device_type}"
    else:
        return f"rank:{rank}/{device_type}:{rank % num_devices_per_node}"


def _create_chunk_sharded_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
    device: Optional[torch.device] = None,
) -> ShardedTensor:
    """
    将张量按第一维度切分为多个块。本地进程将得到对应的块作为本地分片，以创建一个ShardedTensor。
    """
    # 将张量按指定维度切分成块
    chunks = tensor.chunk(world_size, dim=0)
    if len(chunks) > rank:
        # 复制本地进程对应的块
        local_shard = chunks[rank].clone()
        # 计算偏移量，以便定位本地分片
        offsets = [0 for _ in tensor.size()]
        offsets[0] = math.ceil(tensor.size()[0] / world_size) * rank
        local_shards = [Shard.from_tensor_and_offsets(local_shard, offsets, rank)]
    else:
        local_shards = []

    # 创建ShardedTensor，但不触发通信
    chunk_sizes = [list(chunk.size()) for chunk in chunks]
    dim0_offsets = [0] + list(
        itertools.accumulate([chunk_size[0] for chunk_size in chunk_sizes])
    )[:-1]
    offsets = [0] * (len(chunk_sizes[0]) - 1)
    chunk_offsets = [[d0] + offsets for d0 in dim0_offsets]
    device_type = (
        distributed_c10d._get_pg_default_device(pg).type
        if device is None
        else device.type
    )
    # 生成每个块的远程设备字符串表示
    placements = [
        _get_remote_device_str(
            dist.get_global_rank(pg, r),
            device_type,
            num_devices_per_node,
        )
        for r in range(len(chunk_sizes))
    ]
    # 断言确保各列表长度一致
    assert len(chunk_sizes) == len(chunk_offsets) == len(placements)
    # 创建ShardMetadata列表
    shard_metadata = [
        ShardMetadata(offset, size, placement)
        for offset, size, placement in zip(chunk_offsets, chunk_sizes, placements)
    ]
    # 创建ShardedTensorMetadata对象
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shard_metadata,
        size=tensor.size(),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        ),
    )
    # 从本地分片和全局元数据初始化ShardedTensor对象
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards, sharded_tensor_metadata=sharded_tensor_metadata, process_group=pg
    )


def _create_chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    # 继续实现这个函数...
    device_mesh: DeviceMesh,


    # 声明一个变量 device_mesh，其类型为 DeviceMesh
# 将张量沿着第一个维度分片。每个本地进程将获取其对应的分片作为本地张量，创建成为一个 DTensor。
def ) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension. The local rank will gets its
    corresponding chunk as the local tensor to create a DTensor.
    """
    # 我们需要显式调用 .detach() 来返回一个从当前计算图分离的新张量。
    tensor = tensor.clone().detach()

    # FSDP 部署：[Shard(0)]
    # HSDP 部署：[Replicate(), Shard(0)]
    replicate_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements[-1] = DShard(0)  # type: ignore[call-overload]

    # 使用本地张量创建一个 DTensor，并按指定的部署方式重新分布
    return DTensor.from_local(
        tensor, device_mesh, replicate_placements, run_check=False
    ).redistribute(
        placements=shard_placements,
    )


def _all_gather_dtensor(
    tensor: DTensor,
    parent_mesh: Optional[DeviceMesh],
) -> torch.Tensor:
    """
    All gather a DTensor in its sharded dimension and return the local tensor.
    """
    # 断言父设备网格为空
    assert parent_mesh is None

    # 复制张量的部署方式列表
    placements = list(copy.deepcopy(tensor.placements))
    # FSDP 部署：[Shard(0)] -> [Replicate()]
    # HSDP 部署：[Replicate(), Shard(0)] -> [Replicate(), Replicate()]
    placements[-1] = Replicate()
    
    # 根据新的部署方式重新分布张量
    tensor = tensor.redistribute(
        device_mesh=tensor.device_mesh,
        placements=placements,
    )

    # 将张量转换为本地张量并返回
    return tensor.to_local()
```