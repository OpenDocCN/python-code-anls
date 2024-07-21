# `.\pytorch\torch\distributed\_shard\sharded_tensor\utils.py`

```py
# 设置类型检查允许未声明类型的函数和变量
# import 语句导入必要的模块和类
import collections.abc
import copy
from typing import List, Optional, Sequence, TYPE_CHECKING

# 导入 torch 库及其分布式功能
import torch
from torch.distributed import distributed_c10d as c10d, rpc
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,
    validate_non_overlapping_shards_metadata,
)

# 导入本地的模块和类
from .metadata import ShardedTensorMetadata, TensorProperties
from .shard import Shard

# 如果是类型检查阶段，导入类型检查所需的额外模块
if TYPE_CHECKING:
    from torch.distributed._shard.metadata import ShardMetadata


# 定义一个函数，用于解析和验证远程设备
def _parse_and_validate_remote_device(pg, remote_device):
    # 如果远程设备为 None，则抛出值错误异常
    if remote_device is None:
        raise ValueError("remote device is None")

    # 获取远程设备的工作节点名、秩和设备信息
    worker_name = remote_device.worker_name()
    rank = remote_device.rank()
    device = remote_device.device()

    # 如果秩不为 None 并且不是不在进程组中的秩，则进行秩的有效性验证
    if rank is not None and not c10d._rank_not_in_group(pg):
        pg_global_ranks = c10d.get_process_group_ranks(pg)
        # 如果秩不在全局秩列表中，则抛出值错误异常
        if rank not in pg_global_ranks:
            raise ValueError(
                f"Global rank {rank} does not exist in input process group: {pg_global_ranks}"
            )

    # 如果存在工作节点名，则验证当前 RPC 框架是否已初始化
    if worker_name is not None:
        if not rpc._is_current_rpc_agent_set():
            raise RuntimeError(
                f"RPC framework needs to be initialized for using worker names: {worker_name}"
            )

        # 获取当前 RPC 代理的所有工作节点信息，并逐个检查是否匹配给定的工作节点名
        workers = rpc._get_current_rpc_agent().get_worker_infos()
        for worker in workers:
            if worker.name == worker_name:
                return worker.id, device

        # 如果未找到匹配的工作节点名，则抛出值错误异常
        raise ValueError(f"Invalid worker name: {worker_name}")

    # 返回秩和设备信息
    return rank, device


# 定义一个函数，用于验证聚合操作的输出张量是否符合预期
def _validate_output_tensor_for_gather(
    my_rank: int,
    dst_rank: int,
    size: torch.Size,
    dst_tensor: Optional[torch.Tensor],
) -> None:
    # 如果目标秩与当前秩相同
    if dst_rank == my_rank:
        # 如果目标张量为 None，则抛出值错误异常
        if dst_tensor is None:
            raise ValueError(
                f"Argument ``dst_tensor`` must be specified on destination rank {dst_rank}"
            )
        # 如果目标张量的大小与给定大小不匹配，则抛出值错误异常
        if tuple(size) != tuple(dst_tensor.size()):
            raise ValueError(
                f"Argument ``dst_tensor`` have size {tuple(dst_tensor.size())},"
                f"but should be {tuple(size)}"
            )
    # 如果目标秩与当前秩不同，并且目标张量不为 None，则抛出值错误异常
    elif dst_tensor:
        raise ValueError(
            "Argument ``dst_tensor`` must NOT be specified " "on non-destination ranks."
        )


# 定义一个函数，用于展平张量的大小并返回 torch.Size 对象
def _flatten_tensor_size(size) -> torch.Size:
    """
    检查张量大小是否有效，然后展平并返回 torch.Size 对象。
    """
    # 如果大小为单个元素且为序列类型，则将其展平为列表
    if len(size) == 1 and isinstance(size[0], collections.abc.Sequence):
        dims = list(*size)
    else:
        dims = list(size)

    # 检查每个维度是否为整数类型，如果不是则抛出类型错误异常
    for dim in dims:
        if not isinstance(dim, int):
            raise TypeError(f"size has to be a sequence of ints, found: {dims}")

    # 返回展平后的大小作为 torch.Size 对象
    return torch.Size(dims)


# 定义一个函数，用于在不匹配时引发异常
def _raise_if_mismatch(expected, actual, prop_name, ranks, is_local=True):
    # 如果是本地模式
    if is_local:
        # 断言 ranks 是整数类型
        assert isinstance(ranks, int)
        # 如果期望值和实际值不相等，抛出数值错误异常
        if expected != actual:
            raise ValueError(
                # 组合异常消息，说明在指定 rank 上本地片段的 tensor 属性需要相同
                f"Local shards' tensor {prop_name} property need to be the same on rank:{ranks}! "
                f"Found one local shard tensor {prop_name}={expected}, "
                f"the other local shard tensor {prop_name}={actual}."
            )
    else:
        # 非本地模式下，比较失败检查跨多个 ranks，ranks 应该有两个 rank
        assert len(ranks) == 2
        # 如果期望值和实际值不相等，抛出数值错误异常
        if expected != actual:
            raise ValueError(
                # 组合异常消息，说明在不同 ranks 上 ShardedTensor 的属性不匹配
                f"ShardedTensor {prop_name} property does not match from different ranks! "
                f"Found {prop_name}={expected} on rank:{ranks[0]}, "
                f"and {prop_name}={actual} on rank:{ranks[1]}."
            )
# 根据本地分片构建元数据，返回一个ShardedTensorMetadata对象
def build_metadata_from_local_shards(
    local_shards: List[Shard],  # 本地分片列表，每个元素为一个Shard对象
    global_size: torch.Size,    # 全局尺寸信息，类型为torch.Size
    current_rank: int,          # 当前进程的排名
    pg: c10d.ProcessGroup,      # 进程组对象，用于分布式通信
) -> ShardedTensorMetadata:
    assert len(local_shards) > 0, "must have local shards!"  # 断言确保本地分片列表非空

    local_shard_metadatas: List[ShardMetadata] = []  # 用于存储本地分片的元数据列表

    first_shard_dtype = local_shards[0].tensor.dtype  # 获取第一个分片的数据类型
    first_shard_layout = local_shards[0].tensor.layout  # 获取第一个分片的布局信息
    first_shard_requires_grad = local_shards[0].tensor.requires_grad  # 获取第一个分片的梯度需求信息
    first_shard_is_pinned = local_shards[0].tensor.is_pinned()  # 获取第一个分片是否被固定在内存中

    # 1). 验证本地张量及其关联的元数据
    for local_shard in local_shards:
        local_shard_tensor = local_shard.tensor  # 获取当前分片的张量
        local_shard_meta = local_shard.metadata  # 获取当前分片的元数据
        local_shard_metadatas.append(local_shard_meta)  # 将当前分片的元数据添加到列表中

        # 解析和验证远程设备信息，获取当前排名及本地设备
        rank, local_device = _parse_and_validate_remote_device(
            pg, local_shard_meta.placement
        )

        # 检查张量布局是否为torch.strided，如果不是则引发错误
        if (
            local_shard_tensor.layout != torch.strided
            or local_shard_tensor.layout != first_shard_layout
        ):
            raise ValueError(
                f"Only torch.strided layout is currently supported, but found "
                f"{local_shard_tensor.layout} on rank:{current_rank}!"
            )

        # 检查张量是否是连续的，如果不是则引发错误
        if not local_shard_tensor.is_contiguous():
            raise ValueError(
                "Only torch.contiguous_format memory_format is currently supported!"
            )

        # 检查排名是否与当前进程的排名匹配，如果不匹配则引发错误
        if rank != current_rank:
            raise ValueError(
                f"Local shard metadata's rank does not match with the rank in its process group! "
                f"Found current rank in the process group: {current_rank}, "
                f"local ShardMetadata placement's rank: {rank}"
            )

        # 检查张量设备是否与元数据中的设备匹配，如果不匹配则引发错误
        if local_shard_tensor.device != local_device:
            raise ValueError(
                f"Local shard tensor device does not match with local Shard's placement! "
                f"Found local shard tensor device: {local_shard_tensor.device}, "
                f"local shard metadata placement device: {local_device}"
            )

        # 检查张量大小是否与元数据中的大小匹配，如果不匹配则引发错误
        _raise_if_mismatch(
            local_shard_meta.shard_sizes,
            list(local_shard_tensor.size()),
            "size",
            current_rank,
        )

        # 检查张量是否固定在内存中与第一个分片的固定状态是否匹配，如果不匹配则引发错误
        _raise_if_mismatch(
            local_shard_tensor.is_pinned(),
            first_shard_is_pinned,
            "pin_memory",
            current_rank,
        )

        # 检查张量数据类型是否与第一个分片的数据类型匹配，如果不匹配则引发错误
        _raise_if_mismatch(
            local_shard_tensor.dtype, first_shard_dtype, "dtype", current_rank
        )

        # 检查张量是否需要梯度与第一个分片的梯度需求是否匹配，如果不匹配则引发错误
        _raise_if_mismatch(
            local_shard_tensor.requires_grad,
            first_shard_requires_grad,
            "requires_grad",
            current_rank,
        )

    # 2). 构建一个“本地”ShardedTensorMetadata对象，包含当前排名上所有本地分片，
    #    然后执行all_gather来收集来自所有排名的local_sharded_tensor_metadata
    # 创建一个包含本地张量属性的对象，用于描述张量的特性
    local_tensor_properties = TensorProperties(
        dtype=first_shard_dtype,                # 张量的数据类型，使用第一个分片的数据类型
        layout=first_shard_layout,              # 张量的布局，使用第一个分片的布局
        requires_grad=first_shard_requires_grad, # 张量是否需要梯度，使用第一个分片的设置
        memory_format=torch.contiguous_format,  # 张量的内存格式，使用连续格式
        pin_memory=first_shard_is_pinned,       # 是否将张量存储在固定内存中，使用第一个分片的设置
    )
    
    # 创建本地分片张量的元数据对象，用于描述分布式张量的元信息
    local_sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=local_shard_metadatas,  # 分片元数据列表，描述每个分片的信息
        size=global_size,                       # 全局张量的大小
        tensor_properties=local_tensor_properties,  # 本地张量的属性
    )
    
    # 返回描述本地分片张量的元数据对象
    return local_sharded_tensor_metadata
# 构建全局元数据，从收集的元数据序列中生成
def build_global_metadata(
    gathered_metadatas: Sequence[Optional[ShardedTensorMetadata]],
):
    # 初始化全局分片张量元数据为 None
    global_sharded_tensor_metadata = None
    # 初始化全局元数据的排名为 0
    global_metadata_rank = 0

    # 遍历收集到的元数据序列
    for rank, rank_metadata in enumerate(gathered_metadatas):
        # 如果当前排名的元数据为 None，则跳过
        if rank_metadata is None:
            continue

        # 如果全局分片张量元数据为 None，则使用当前排名的元数据进行深拷贝，并更新全局元数据排名
        if global_sharded_tensor_metadata is None:
            global_sharded_tensor_metadata = copy.deepcopy(rank_metadata)
            global_metadata_rank = rank
        else:
            # 检查全局大小是否匹配，如果不匹配则抛出异常
            _raise_if_mismatch(
                global_sharded_tensor_metadata.size,
                rank_metadata.size,
                "global_size",
                [global_metadata_rank, rank],
                is_local=False,
            )

            # 不需要检查布局和内存格式，因为在本地分片验证阶段已经检查过
            # 检查数据类型是否匹配，如果不匹配则抛出异常
            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.dtype,
                rank_metadata.tensor_properties.dtype,
                "dtype",
                [global_metadata_rank, rank],
                is_local=False,
            )

            # 检查是否需要梯度是否匹配，如果不匹配则抛出异常
            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.requires_grad,
                rank_metadata.tensor_properties.requires_grad,
                "requires_grad",
                [global_metadata_rank, rank],
                is_local=False,
            )

            # 检查是否需要 pin_memory 是否匹配，如果不匹配则抛出异常
            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.pin_memory,
                rank_metadata.tensor_properties.pin_memory,
                "pin_memory",
                [global_metadata_rank, rank],
                is_local=False,
            )
            # 通过所有验证，将分片元数据扩展到全局分片张量元数据中
            global_sharded_tensor_metadata.shards_metadata.extend(
                rank_metadata.shards_metadata
            )

    # 如果全局分片张量元数据不为 None，则进行进一步验证

    if global_sharded_tensor_metadata is not None:
        # 检查分片元数据是否存在重叠分片
        validate_non_overlapping_shards_metadata(
            global_sharded_tensor_metadata.shards_metadata
        )

        # 检查分片元数据是否与全局分片张量的大小兼容
        check_tensor(
            global_sharded_tensor_metadata.shards_metadata,
            global_sharded_tensor_metadata.size,
        )
    else:
        # 如果全局分片张量元数据为 None，则抛出 ValueError
        raise ValueError("ShardedTensor have no local shards on all ranks!")

    # 返回全局分片张量元数据
    return global_sharded_tensor_metadata
```