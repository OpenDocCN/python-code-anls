# `.\pytorch\torch\distributed\checkpoint\resharding.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型 List 和 Tuple
from typing import List, Tuple

# 从 torch.distributed.checkpoint.metadata 模块中导入 ChunkStorageMetadata 类
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata

# 初始化空列表，用于存储公开的模块成员名称
__all__: List[str] = []


def _check_shard_metadata_pair_overlap(
    shard1: ChunkStorageMetadata, shard2: ChunkStorageMetadata
):
    """Check if two shards overlap."""
    # 对每个维度的每个分片，检查一个分片是否在另一个分片的另一端
    # 例如，对于一个二维分片，我们会检查一个分片是否在另一个分片的上方或左侧
    ndims = len(shard1.offsets)
    for i in range(ndims):
        if shard1.offsets[i] >= shard2.offsets[i] + shard2.sizes[i]:
            return False
        if shard2.offsets[i] >= shard1.offsets[i] + shard1.sizes[i]:
            return False

    return True


def _shards_get_overlap_region_wrt_saved_tensor(
    saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata
) -> List[Tuple[int, int, int, int]]:
    """
    Return the overlapping region between saved_shard and current_shard.

    There returned list has the same number of elements as the tensor's dimension.
    For each element, we produce a tuple with the following contents:
        (dimension, `saved_shard` offset, `current_shard` offset, length)

    Offsets are relative to each shard.
    """
    # 初始化空列表，用于存储重叠区域的元组
    narrows = []
    # 遍历保存的分片和当前分片的偏移和大小
    for dim, (
        saved_shard_offset,
        current_shard_offset,
        saved_shard_size,
        current_shard_size,
    ) in enumerate(
        zip(
            saved_shard.offsets,
            current_shard.offsets,
            saved_shard.sizes,
            current_shard.sizes,
        )
    ):
        # 计算两个分片在当前维度上的最小范围末端
        min_range_end = min(
            saved_shard_offset + saved_shard_size,
            current_shard_offset + current_shard_size,
        )

        # 计算重叠区域的长度
        length = min_range_end - max(current_shard_offset, saved_shard_offset)

        # 根据偏移计算每个分片在张量中的相对偏移
        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0

        # 将结果添加到 narrows 列表中
        narrows.append(
            (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
        )

    return narrows
```