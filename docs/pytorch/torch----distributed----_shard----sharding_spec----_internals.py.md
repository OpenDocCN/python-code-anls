# `.\pytorch\torch\distributed\_shard\sharding_spec\_internals.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型和模块
from typing import List, Optional, Tuple

from torch.distributed._shard.metadata import ShardMetadata


def _check_shard_metadata_pair_overlap(shard1: ShardMetadata, shard2: ShardMetadata):
    """
    检查两个分片是否重叠。
    """

    # 针对每个分片的每个维度，检查一个分片是否在另一个分片的另一端
    # 例如，对于一个二维分片，我们检查一个分片是否在另一个分片的上方或左侧。
    ndims = len(shard1.shard_offsets)
    for i in range(ndims):
        if shard1.shard_offsets[i] >= shard2.shard_offsets[i] + shard2.shard_sizes[i]:
            return False
        if shard2.shard_offsets[i] >= shard1.shard_offsets[i] + shard1.shard_sizes[i]:
            return False

    return True


def _find_nd_overlapping_shards(
    shards: List[ShardMetadata], sharded_dims: List[int]
) -> Optional[Tuple[int, int]]:
    # 每个分片在每个维度上有 len(sharded_dims) 个元组。每个元组表示该维度的 [begin, end]（包括）对。
    shard_intervals = [
        [
            (s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1)
            for dim in sharded_dims
        ]
        for s in shards
    ]

    for i in range(len(shards)):
        shard_i = shard_intervals[i]
        for j in range(i + 1, len(shards)):
            shard_j = shard_intervals[j]
            # 针对每个分片的每个维度，检查一个分片是否在另一个分片的另一端
            # 例如，对于一个二维分片，我们检查一个分片是否在另一个分片的上方或左侧。
            overlap = True
            for interval_i, interval_j in zip(shard_i, shard_j):
                if interval_i[0] > interval_j[1] or interval_j[0] > interval_i[1]:
                    overlap = False
                    break
            if overlap:
                return (i, j)
    return None


def _find_1d_overlapping_shards(
    shards: List[ShardMetadata], dim: int
) -> Optional[Tuple[int, int]]:
    # (begin, end, index_in_shards). Begin and end are inclusive.
    # 每个分片在给定维度上的区间列表，同时记录其在 shards 中的索引。
    intervals = [
        (s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1, i)
        for i, s in enumerate(shards)
    ]
    intervals.sort()
    for i in range(len(shards) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            return (intervals[i][2], intervals[i + 1][2])
    return None


def validate_non_overlapping_shards_metadata(shards: List[ShardMetadata]):
    """
    确保没有任何两个分片重叠。

    Args:
        shards(List[ShardMetadata]): 包含每个分片的 :class:`ShardMetadata` 对象的列表。
    Raises:
        ``ValueError``: 如果任何两个分片存在重叠。
    """
    if not shards or len(shards) == 1:
        return

    # 存储分片在哪些维度上进行分片
    sharded_dims: List[int] = []
    # 遍历所有分片的第一个分片的分片偏移量列表的长度范围
    for dim in range(len(shards[0].shard_offsets)):
        # 遍历除第一个分片外的所有分片
        for i in range(1, len(shards)):
            # 检查当前维度的分片偏移量和分片大小是否与第一个分片相同
            if (
                shards[i].shard_offsets[dim] != shards[0].shard_offsets[dim]
                or shards[i].shard_sizes[dim] != shards[0].shard_sizes[dim]
            ):
                # 如果不同，则将该维度添加到已分片的维度列表中，并结束内循环
                sharded_dims.append(dim)
                break

    # 声明一个可选的元组，用于存储重叠的分片索引对
    pair: Optional[Tuple[int, int]] = None
    # 如果没有已分片的维度
    if len(sharded_dims) == 0:
        # 所有分片都相同，所有维度均未分片。选择任意两个分片。
        pair = (0, 1)
    # 如果只有一个已分片的维度
    elif len(sharded_dims) == 1:
        # 分片仅在一个维度上分片。可以使用O(nlogn)的重叠区间算法找到重叠。
        pair = _find_1d_overlapping_shards(shards, sharded_dims[0])
    # 如果有多个已分片的维度
    else:
        # 分片在多个维度上分片。回退到逐对检查。尽管2D重叠的O(nlogn)算法（线性扫描）存在，
        # 但其实现并不简单，在大多数情况下可能无法证明时间节省。
        pair = _find_nd_overlapping_shards(shards, sharded_dims)

    # 如果存在重叠的分片索引对
    if pair:
        # 抛出值错误，指出重叠的分片
        raise ValueError(f"Shards {shards[pair[0]]} and {shards[pair[1]]} overlap")
def check_tensor(shards_metadata, tensor_dims) -> None:
    """
    Checks if the shards_metadata is compatible with the provided tensor dims.

    Args:
        shards_metadata(List[ShardMetadata]): List of :class:`ShardMetadata`
            objects representing each shard of the tensor.
        tensor_dims(Sequence of int): Dimensions of tensor to verify

    Raises:
        ``ValueError`` if not compatible.
    """

    # 计算张量和分片总体积
    tensor_rank = len(tensor_dims)
    shards_rank = len(shards_metadata[0].shard_offsets)
    if tensor_rank != shards_rank:
        raise ValueError(
            f"Rank of tensor is {tensor_rank}, but shards rank is {shards_rank}"
        )

    total_shard_volume = 0
    # 遍历每个分片的元数据
    for shard in shards_metadata:
        shard_volume = 1
        # 计算每个分片的体积，并检查分片边界是否在张量维度范围内
        for i, shard_length in enumerate(shard.shard_sizes):
            shard_volume *= shard_length
            if shard.shard_offsets[i] + shard.shard_sizes[i] > tensor_dims[i]:
                raise ValueError(
                    f"Shard offset {shard.shard_offsets[i]} and length "
                    f"{shard.shard_sizes[i]} exceeds tensor dim: {tensor_dims[i]} for shard {shard}"
                )
        total_shard_volume += shard_volume

    # 计算张量的总体积
    tensor_volume = 1
    for size in tensor_dims:
        tensor_volume *= size

    # 检查总体积是否匹配，若不匹配则抛出异常
    if total_shard_volume != tensor_volume:
        # TODO: Can we improve this error message to point out the gaps?
        raise ValueError(
            f"Total volume of shards: {total_shard_volume} "
            f"does not match tensor volume: {tensor_volume}, in other words "
            f"all the individual shards do not cover the entire tensor"
        )


def get_split_size(dim_size, chunks):
    """
    Computes the split size inline with ``torch.chunk``

    Args:
        dim_size(int): Size of the dimension being chunked.
        chunks(int): Number of chunks to create for ``dim_size``.

    Returns:
        An int indicating the split size to use.
    """
    # 计算每个分片的大小，保证分片能够覆盖整个维度
    return (dim_size + chunks - 1) // chunks


def get_chunked_dim_size(dim_size, split_size, idx):
    """
    Computes the dim size of the chunk for provided ``idx`` given ``dim_size``
    and ``split_size``.

    Args:
        dim_size(int): Size of the dimension being chunked.
        split_size(int): The chunk size for each chunk of ``dim_size``.
        idx(int): The index of chunk whose dim size is being requested.

    Returns:
        An int indicating the dim size of the chunk.
    """
    # 计算给定索引处的分片维度大小
    return max(min(dim_size, split_size * (idx + 1)) - split_size * idx, 0)


def get_chunk_sharding_params(sharding_dim_size, world_size, spec, rank):
    """
    Generate the start pos and offset length for the current rank for
    chunk sharding.
    """
    # 此处还需要补充代码和注释
    Args:
        sharding_dim_size(int): 进行分片的维度长度。
        world_size(int): 进程数。
        spec (:class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec`):
            分片规格对象。
        rank(int): 当前 CUDA 进程的排名。

    Returns:
        start_pos(int): 给定排名上分片张量的起始位置。
        chunk_size(int): 给定排名上分片张量的块大小。
    """
    # 根据维度长度和进程数计算每个分片的大小
    split_size = get_split_size(sharding_dim_size, world_size)
    current_offsets = 0
    start_pos = current_offsets
    # 遍历分片规格中的每个分片配置
    for idx, placement in enumerate(spec.placements):
        # 根据分片维度大小、切片大小和索引计算当前分片的大小
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        # 如果当前进程的排名与分片配置的排名匹配，则记录起始位置并结束循环
        if rank == placement.rank():
            start_pos = current_offsets
            break
        # 更新当前偏移量，准备处理下一个分片
        current_offsets += chunk_size
    # 返回分片张量的起始位置和块大小
    return start_pos, chunk_size  # type: ignore[possibly-undefined]
```