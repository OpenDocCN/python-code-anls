# `.\pytorch\torch\distributed\_shard\metadata.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Union

# 导入 torch 分布式模块中的远程设备类
from torch.distributed.remote_device import _remote_device


@dataclass
class ShardMetadata:
    """
    Represents a shard of the overall Tensor including its
    offsets, lengths and device placement.

    Args:
        shard_offsets(List[int]): Offsets in the original tensor indicating
            the start offsets for this shard. Should have the same rank as
            the original tensor.
        shard_sizes(List[int]): Integers indicating the size of each
            dimension for this shard. Should have the same rank as the
            original tensor.
        placement(:class:`torch.distributed._remote_device`):
            Specifies the placement of this shard.
    """

    # 使用 __slots__ 优化内存占用
    __slots__ = ["shard_offsets", "shard_sizes", "placement"]

    # 定义数据字段
    shard_offsets: List[int]
    shard_sizes: List[int]
    placement: Optional[_remote_device]

    def __init__(
        self,
        shard_offsets: List[int],
        shard_sizes: List[int],
        placement: Optional[Union[str, _remote_device]] = None,
    ):
        # 初始化对象，设置 shard_offsets、shard_sizes 和 placement
        self.shard_offsets = shard_offsets
        self.shard_sizes = shard_sizes
        # 如果 placement 是字符串，创建 _remote_device 对象
        if isinstance(placement, str):
            self.placement = _remote_device(placement)
        else:
            self.placement = placement
        
        # 检查 shard_offsets 和 shard_sizes 的长度是否一致
        if len(self.shard_offsets) != len(self.shard_sizes):
            raise ValueError(
                f"shard_offsets and shard_sizes should have "
                f"the same number of elements, found {len(self.shard_offsets)} "
                f"and {self.shard_sizes} respectively"
            )
        
        # 检查 shard_offsets 和 shard_sizes 的元素是否合法
        for i in range(len(self.shard_offsets)):
            if self.shard_offsets[i] < 0:
                raise ValueError("shard_offsets should be >=0")
            if self.shard_sizes[i] < 0:
                raise ValueError("shard_sizes should be >= 0")

    def __hash__(self):
        # 定义哈希函数，使用 reduce 和 hash 运算
        def _hash_reduce(a, b):
            return (a << 8) + hash(b)

        # 计算哈希值，从 shard_offsets 开始
        res = reduce(_hash_reduce, self.shard_offsets, 37)
        # 加入 shard_sizes 计算哈希值
        res = reduce(_hash_reduce, self.shard_sizes, res)
        # 加入 placement 计算哈希值
        res = _hash_reduce(res, self.placement)
        return res
```