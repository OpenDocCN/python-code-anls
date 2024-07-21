# `.\pytorch\torch\distributed\_shard\sharded_tensor\shard.py`

```py
# mypy: allow-untyped-defs
# 引入需要的模块和类
from dataclasses import dataclass
from typing import List

import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed.remote_device import _remote_device

# 使用 @dataclass 装饰器，定义一个数据类 Shard
@dataclass
class Shard:
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.

    Args:
        tensor(torch.Tensor): Local tensor for the shard.
        metadata(:class `torch.distributed._shard.sharded_tensor.ShardMetadata`):
            The metadata for the shard, including offsets, lengths and device placement.
    """

    # 定义数据类的__slots__，以优化内存使用
    __slots__ = ["tensor", "metadata"]
    # 定义数据类的两个字段：tensor 和 metadata
    tensor: torch.Tensor
    metadata: ShardMetadata

    # 构造函数后处理方法
    def __post_init__(self):
        # 验证本地张量和元数据的匹配性
        if list(self.tensor.size()) != self.metadata.shard_sizes:
            raise ValueError(
                "Shard tensor size does not match with metadata.shard_lengths! "
                f"Found shard tensor size: {list(self.tensor.size())}, "
                f"metadata.shard_lengths: {self.metadata.shard_sizes}, "
            )
        # 检查张量的设备与元数据的设备位置是否匹配
        placement_device = self.metadata.placement
        if (
            placement_device is not None
            and placement_device.device() != self.tensor.device
        ):
            raise ValueError(
                f"Local shard tensor device does not match with local Shard's placement! "
                f"Found local shard tensor device: {self.tensor.device}, "
                f"local shard metadata placement device: {placement_device.device()}"
            )

    # 类方法：从张量、偏移和排名创建 Shard 实例
    @classmethod
    def from_tensor_and_offsets(
        cls, tensor: torch.Tensor, shard_offsets: List[int], rank: int
    ):
        """
        Creates a Shard of a ShardedTensor from a local torch.Tensor, shard_offsets and rank.

        Args:
            tensor(torch.Tensor): Local tensor for the shard.
            shard_offsets(List[int]): List of integers specify the offset
                of the shard on each dimension.
            rank(int): Specify the rank for the shard.
        """
        # 获取张量的大小
        shard_sizes = list(tensor.size())
        # 根据排名和张量的设备信息获取放置设备
        placement = _remote_device(f"rank:{rank}/{str(tensor.device)}")
        # 创建 ShardMetadata 实例，包括偏移、大小和放置设备信息
        shard_meta = ShardMetadata(
            shard_offsets=shard_offsets, shard_sizes=shard_sizes, placement=placement
        )
        # 返回创建的 Shard 实例
        return Shard(tensor, shard_meta)
```