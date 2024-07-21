# `.\pytorch\torch\distributed\checkpoint\metadata.py`

```
# mypy: allow-untyped-defs
# 导入所需的模块
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.distributed.checkpoint.stateful import StatefulT

# 暴露给外部的符号列表
__all__ = [
    "ChunkStorageMetadata",
    "TensorStorageMetadata",
    "BytesStorageMetadata",
    "Metadata",
    "MetadataIndex",
    "TensorProperties",
    "StorageMeta",
]

# 用于存储块的元数据
@dataclass
class ChunkStorageMetadata:
    """
    Each chunk is expected to have the same properties of the TensorStorageMetadata
    that includes it.
    """
    offsets: torch.Size  # 块的偏移量信息
    sizes: torch.Size    # 块的大小信息


# 描述张量的内存格式编码
class _MEM_FORMAT_ENCODING(Enum):
    """Describe the memory format of a tensor."""
    TORCH_CONTIGUOUS_FORMAT = 0    # Torch连续格式
    TORCH_CHANNELS_LAST = 1        # Torch通道最后格式
    TORCH_PRESERVE_FORMAT = 2      # Torch保持格式

# 用于创建Tensor的属性
@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""
    
    dtype: torch.dtype = field(default_factory=torch.get_default_dtype)  # 张量的数据类型，默认为默认数据类型
    layout: torch.layout = field(default=torch.strided)                  # 张量的布局，已弃用，默认为分布式
    requires_grad: bool = False                                          # 是否需要梯度，默认为False，已弃用
    memory_format: torch.memory_format = field(default=torch.contiguous_format)  # 张量的内存格式，默认为连续格式，已弃用
    pin_memory: bool = False                                             # 是否使用固定内存，默认为False，已弃用

    def __getstate__(self):
        # 由于torch.memory_format不能被pickle化！
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT  # 内存格式编码：Torch连续格式
        elif memory_format == torch.channels_last:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST      # 内存格式编码：Torch通道最后格式
        elif memory_format == torch.preserve_format:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT    # 内存格式编码：Torch保持格式
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

        if mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format     # 设置为Torch连续格式
        elif mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last         # 设置为Torch通道最后格式
        elif mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format       # 设置为Torch保持格式
        else:
            raise RuntimeError(
                f"Invalid torch.memory_format encoding: {mem_format_encoding}"
            )

        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> "TensorProperties":
        # 根据给定的张量创建一个TensorProperties对象，并返回该对象
        return TensorProperties(
            # 设置TensorProperties对象的dtype属性为给定张量的数据类型
            dtype=tensor.dtype,
            # 设置TensorProperties对象的layout属性为给定张量的布局
            layout=tensor.layout,
            # 设置TensorProperties对象的requires_grad属性为给定张量的梯度是否需要计算
            requires_grad=tensor.requires_grad,
            # 设置TensorProperties对象的memory_format属性为torch的连续内存格式
            memory_format=torch.contiguous_format,
            # 设置TensorProperties对象的pin_memory属性为给定张量是否被固定在内存中
            pin_memory=tensor.is_pinned(device=tensor.device),
        )
@dataclass
class TensorStorageMetadata:
    properties: TensorProperties  # 存储张量的属性信息
    size: torch.Size  # 张量的大小信息
    chunks: List[ChunkStorageMetadata]  # 存储张量块的元数据列表


@dataclass
class BytesStorageMetadata:
    pass  # 字节数据的存储元数据，目前为空


STORAGE_TYPES = Union[TensorStorageMetadata, BytesStorageMetadata]
STATE_DICT_TYPE = Dict[str, Union[StatefulT, Any]]


@dataclass
class StorageMeta:
    checkpoint_id: Union[str, os.PathLike, None] = None  # 存储元数据的检查点 ID，可以是字符串或路径
    save_id: Optional[str] = None  # 存储元数据的保存 ID，可选的字符串
    load_id: Optional[str] = None  # 存储元数据的加载 ID，可选的字符串


@dataclass
class Metadata:
    """This class represents the metadata of the checkpoint."""

    state_dict_metadata: Dict[str, STORAGE_TYPES]
    """
    检查点的元数据字典，键为状态字典的键，值为相应的存储类型（张量或字节数据的元数据）
    
    It is the responsibility of the planner and storage plugins to ensure
    backward compatibility of the planner_data and storage_data. DCP will
    also ensure the backward compatibility of the metadata in this file and
    the metadata of the built-in planner and storage plugins.
    """
    planner_data: Any = None  # 计划器数据，由计划器和存储插件负责确保向后兼容性
    storage_data: Any = None  # 存储数据，由计划器和存储插件负责确保向后兼容性
    storage_meta: Optional[StorageMeta] = None  # 存储元数据，可选的存储元信息


@dataclass(frozen=True)
class MetadataIndex:
    """This class represents a lookup key for items in a state dict or Metadata."""

    fqn: str  # 对象的完全限定名称
    offset: Optional[torch.Size] = None  # 对象在张量中的偏移量（如果是张量的话）
    index: Optional[int] = field(hash=False, compare=False, default=None)
    """
    当搜索张量块以加快查找时的索引提示（可选）
    
    A common representation of a sharded tensor is as a list of chunks so to
    find the index in such a list you need to linear search it.
    
    When constructing an instance of MetadataIndex that points to that list,
    one can provide the index as a hint and it will be probed first before
    the linear search and thus making it significantly faster.
    """

    def __init__(
        self,
        fqn: str,
        offset: Optional[Sequence[int]] = None,
        index: Optional[int] = None,
    ):
        # We must use object.__setattr__ due to frozen=True
        object.__setattr__(self, "fqn", fqn)
        object.__setattr__(self, "index", index)
        if offset is not None:
            object.__setattr__(self, "offset", torch.Size(offset))
```