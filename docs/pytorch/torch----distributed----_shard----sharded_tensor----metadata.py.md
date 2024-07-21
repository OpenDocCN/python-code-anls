# `.\pytorch\torch\distributed\_shard\sharded_tensor\metadata.py`

```py
# 使用 mypy: allow-untyped-defs 以允许未类型化的定义
from dataclasses import dataclass, field  # 导入 dataclass 和 field 装饰器
from enum import Enum  # 导入枚举类型 Enum
from typing import List  # 导入 List 类型

import torch  # 导入 PyTorch 库
from torch.distributed._shard.metadata import ShardMetadata  # 导入 ShardMetadata 类


class MEM_FORMAT_ENCODING(Enum):
    # 定义枚举类型 MEM_FORMAT_ENCODING，表示 torch.memory_format 的编码方式
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: torch.dtype = field(default=torch.get_default_dtype())  # 数据类型，默认为当前默认数据类型
    layout: torch.layout = field(default=torch.strided)  # 张量布局，默认为 strided
    requires_grad: bool = False  # 是否需要梯度，默认为 False
    memory_format: torch.memory_format = field(default=torch.contiguous_format)  # 存储格式，默认为 contiguous_format
    pin_memory: bool = False  # 是否固定内存，默认为 False

    def __getstate__(self):
        # 序列化对象时调用，返回对象的状态
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f"Invalid torch.memory_format: {memory_format}")

        return (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        ) = state

        if mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(
                f"Invalid torch.memory_format encoding: {mem_format_encoding}"
            )

        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> "TensorProperties":
        # 从给定的 tensor 创建 TensorProperties 对象
        return TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(device=tensor.device),
        )


@dataclass
class ShardedTensorMetadata:
    """
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: List[ShardMetadata] = field(default_factory=list)  # 每个分片的元数据列表

    # Size of each dim of the overall Tensor.
    size: torch.Size = field(default=torch.Size([]))  # 整体张量的各维度大小
    tensor_properties: TensorProperties = field(default_factory=TensorProperties)


# 声明一个名为 tensor_properties 的变量，其类型为 TensorProperties，使用默认工厂函数初始化
```