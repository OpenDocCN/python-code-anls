# `.\pytorch\torch\distributed\_shard\sharded_tensor\reshard.py`

```
# mypy: allow-untyped-defs
# 引入必要的库和模块
import copy  # 导入 copy 模块，用于复制对象
from typing import List, Tuple  # 导入类型提示相关的 List 和 Tuple

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入 PyTorch 分布式模块
import torch.distributed._shard.sharding_spec as shard_spec  # 导入分片规范模块
from torch._C._distributed_c10d import ProcessGroup  # 导入分布式进程组类
from torch.distributed._shard.metadata import ShardMetadata  # 导入分片元数据类
from torch.distributed._shard.sharding_spec._internals import (
    get_chunked_dim_size,  # 导入获取分块维度大小的函数
    get_split_size,  # 导入获取分割大小的函数
)
from torch.distributed.nn.functional import all_to_all, all_to_all_single  # 导入分布式操作函数

from .shard import Shard  # 导入本地的 shard 模块中的 Shard 类


def get_idx_from_placements(placements, current_rank) -> int:
    """
    返回当前 rank 在给定 placements 列表中的位置索引。

    Args:
        placements(List[Union[_remote_device, str]]):
            指定每个 Tensor 分片的放置位置。列表的大小表示要创建的分片数。
            可以是由 torch.distributed._remote_device 组成的列表。
            列表也可以包含作为 torch.distributed._remote_device 接受的远程设备的字符串。
        current_rank (int): 当前设备的编号。

    Returns:
        int: 当前设备在放置列表中的位置索引。
    """
    for idx, placement in enumerate(placements):  # 遍历 placements 列表
        if current_rank == placement.rank():  # 检查当前 rank 是否等于 placement 的 rank
            return idx  # 如果相等，返回当前索引
    raise RuntimeError("current_rank not in the placement.")  # 如果未找到匹配项，抛出运行时错误


def build_reshard_metadata(
    st_size: torch.Size,
    sharding_spec: shard_spec.ShardingSpec,
    world_size: int,
) -> Tuple[List[ShardMetadata], List[int]]:
    """
    根据给定的分片规范，计算偏移量和本地分片大小。
    然后基于计算结果构建 ShardMetadata。

    Args:
        st_size (torch.Size): Tensor 的分片大小。
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): 描述张量如何分片的规范。
        world_size (int): rank 的总数。

    Returns:
        Tuple[List[ShardMetadata], List[int]]: 包含分片元数据的列表和按放置顺序排列的 rank 列表。
    """
    shard_dim = int(sharding_spec.dim)  # 获取分片维度
    shards_metadata = [None] * world_size  # 创建一个长度为 world_size 的 None 列表
    ranks = []  # 初始化空列表用于存放 rank
    offsets = [0] * len(st_size)  # 根据 st_size 初始化偏移量列表
    split_size = get_split_size(st_size[shard_dim], world_size)  # 获取分割大小
    # 遍历 sharding_spec 中的 placements 列表，同时获取索引和每个 placement 对象
    for idx, placement in enumerate(sharding_spec.placements):  # type: ignore[attr-defined]
        # 将当前 placement 的 rank 添加到 ranks 列表中
        ranks.append(placement.rank())
        
        # 计算分片后的维度大小，根据索引 idx 使用 get_chunked_dim_size 函数
        sharded_dim_size = get_chunked_dim_size(st_size[shard_dim], split_size, idx)
        
        # 复制原始的 st_size 列表，更新 shard_dim 维度为 sharded_dim_size
        local_tensor_size = list(st_size)
        local_tensor_size[shard_dim] = sharded_dim_size
        
        # 使用 placement.rank() 作为 key，创建 ShardMetadata 对象并存储到 shards_metadata 中
        shards_metadata[placement.rank()] = ShardMetadata(  # type: ignore[call-overload]
            shard_offsets=copy.deepcopy(offsets),  # 深拷贝 offsets 字典作为 shard_offsets
            shard_sizes=local_tensor_size,  # 使用更新后的 local_tensor_size 作为 shard_sizes
            placement=placement,  # 将当前 placement 对象作为 placement
        )
        
        # 更新 offsets 字典中的 shard_dim 维度值，增加 sharded_dim_size
        offsets[shard_dim] += sharded_dim_size
    
    # 返回创建的 shards_metadata 字典和 ranks 列表作为结果
    return shards_metadata, ranks  # type: ignore[return-value]
def reshuffle_local_shard(
    local_shard: torch.Tensor,
    st_size: torch.Size,
    sharding_spec: shard_spec.ShardingSpec,
    resharding_spec: shard_spec.ShardingSpec,
    pg: ProcessGroup,
) -> Tuple[List[Shard], List[ShardMetadata]]:
    """
    Reshuffle the local shard directly when the reshard dim is same as the original
    sharding dim. Logically we do this in two step:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending the local tensor to
    the new shard directly based on the resharding spec.

    Args:
        local_shard (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the followings:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
    current_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    
    # Build shards_metadata first.
    shards_metadata, ranks = build_reshard_metadata(
        st_size, resharding_spec, world_size
    )
    
    # Get input split size for all2all.
    reshard_dim = int(resharding_spec.dim)  # type: ignore[attr-defined]
    split_size = get_split_size(st_size[reshard_dim], world_size)
    input_split_sizes = [0] * world_size
    idx = get_idx_from_placements(sharding_spec.placements, current_rank)  # type: ignore[attr-defined]
    new_rank = resharding_spec.placements[idx].rank()  # type: ignore[union-attr, attr-defined]
    input_split_sizes[new_rank] = local_shard.size(reshard_dim)
    
    # Get output split size for all2all.
    output_split_sizes = [0] * world_size
    new_idx = ranks.index(current_rank)
    sharded_dim_size = get_chunked_dim_size(st_size[reshard_dim], split_size, new_idx)
    output_split_sizes[new_rank] = sharded_dim_size
    
    # Get gathered_input for all2all.
    local_shard = local_shard.transpose(0, reshard_dim).contiguous()
    gathered_input_size = list(local_shard.size())
    gathered_input_size[0] = sharded_dim_size
    gathered_input = torch.empty(
        gathered_input_size, device=local_shard.device, dtype=local_shard.dtype
    )
    
    # all2all.
    local_shard = all_to_all_single(
        gathered_input,
        local_shard,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=pg,
    )
    # 对本地张量进行维度转置，使得指定的维度（reshard_dim）成为第一个维度，并保证内存中数据的连续性
    local_tensor = local_shard.transpose(0, reshard_dim).contiguous()
    
    # 创建一个包含单个本地分片的列表，每个分片使用指定的元数据（shards_metadata[current_rank]）
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    
    # 返回本地分片列表及其相关的元数据
    return local_shards, shards_metadata
# 定义一个函数，用于重新分片本地张量
def reshard_local_shard(
    local_tensor: torch.Tensor,
    st_size: torch.Size,
    sharding_spec: shard_spec.ShardingSpec,
    resharding_spec: shard_spec.ShardingSpec,
    pg: ProcessGroup,
) -> Tuple[List[Shard], List[ShardMetadata]]:
    """
    Reshard a sharded tensor given the ``resharding_spec``. When the reshard dim is
    different from the original sharding dim, we need to do two steps logically:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending each rank the new
    shard based on the resharding spec.

    Args:
        local_tensor (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the followings:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
    # 获取当前进程在进程组中的排名
    current_rank = dist.get_rank(pg)
    # 获取进程组中的进程总数
    world_size = dist.get_world_size(pg)
    # 获取原始分片维度并转换为整数类型
    current_sharding_dim = int(sharding_spec.dim)  # type: ignore[attr-defined]
    # 获取重新分片维度并转换为整数类型
    reshard_dim = int(resharding_spec.dim)  # type: ignore[attr-defined]

    # 首先构建分片元数据和排名信息
    shards_metadata, ranks = build_reshard_metadata(
        st_size, resharding_spec, world_size
    )

    # 计算预期大小
    input_split_sizes = []
    for metadata in shards_metadata:
        input_split_sizes.append(metadata.shard_sizes[reshard_dim])
    # 检查是否需要重新排列输入
    rearrange_input = any(ranks[i] > ranks[i + 1] for i in range(len(ranks) - 1))

    if rearrange_input:
        # 在进行全局通信前，需要重新排列本地张量在重新分片维度上的数据
        indices: List[int] = []
        for metadata in shards_metadata:
            offset_start_idx = metadata.shard_offsets[reshard_dim]
            split_size = metadata.shard_sizes[reshard_dim]
            indices += range(offset_start_idx, offset_start_idx + split_size)
        # 使用索引选择方法对本地张量进行重新排列
        local_tensor = local_tensor.index_select(
            reshard_dim, torch.tensor(indices, device=local_tensor.device)
        )

    # 因为重新分片维度与原始分片维度不同，我们需要计算每个进程中张量的大小
    output_tensor_list = [torch.tensor(1)] * world_size
    # 计算每个分片的大小
    split_size = get_split_size(st_size[current_sharding_dim], world_size)
    # 检查是否需要重新排列输出列表
    rearrange_output_list = False
    # 初始化索引列表
    indices = []
    for idx, placement in enumerate(sharding_spec.placements):  # type: ignore[attr-defined]
        # 使用enumerate()遍历sharding_spec.placements列表，并获取索引idx和每个placement对象
        sharded_dim_size = get_chunked_dim_size(
            st_size[current_sharding_dim], split_size, idx
        )
        # 计算分片后的维度大小，调用get_chunked_dim_size函数来确定
        output_tensor_size = list(st_size)
        # 创建输出张量的大小列表，复制st_size的内容
        output_tensor_size[current_sharding_dim] = sharded_dim_size
        # 更新输出张量大小列表中的当前分片维度的大小
        output_tensor_size[reshard_dim] = input_split_sizes[current_rank]
        # 更新输出张量大小列表中的重新分片维度的大小
        output_tensor_list[
            placement.rank()
        ] = torch.empty(  # type: ignore[union-attr, index]
            output_tensor_size, device=local_tensor.device, dtype=local_tensor.dtype
        )
        # 在output_tensor_list中为当前placement的rank创建一个空的torch张量
        indices.append(placement.rank())  # type: ignore[union-attr, index, arg-type]
        # 将当前placement的rank添加到索引列表indices中
        if idx != placement.rank():  # type: ignore[union-attr]
            rearrange_output_list = True
        # 如果索引idx不等于当前placement的rank，则设置rearrange_output_list标志为True

    # 执行启用自动微分的all2all操作。
    input_tensor_tuple = torch.split(local_tensor, input_split_sizes, dim=reshard_dim)
    # 将本地张量按指定尺寸input_split_sizes和维度reshard_dim进行分割，并返回元组
    input_tensor_list = [tensor.contiguous() for tensor in input_tensor_tuple]
    # 创建一个连续的张量列表，将分割后的张量转换为连续存储
    output_tensor_list = all_to_all(
        output_tensor_list,
        input_tensor_list,
        group=pg,
    )
    # 调用all_to_all函数执行all-to-all通信，将output_tensor_list和input_tensor_list传递给pg组

    if rearrange_output_list:
        # 需要重新排列output_tensor_list的原始shard_dim。
        output_tensor_list = [output_tensor_list[idx] for idx in indices]  # type: ignore[call-overload]
        # 如果需要重新排列，则根据indices重新排列output_tensor_list中的张量

    local_tensor = torch.cat(output_tensor_list, dim=current_sharding_dim)
    # 将重新排列后的output_tensor_list按current_sharding_dim维度连接为本地张量
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    # 创建本地shard列表，每个shard由local_tensor和对应的shards_metadata[current_rank]组成
    return local_shards, shards_metadata
    # 返回本地shards列表和shards_metadata
```