# `.\pytorch\torch\distributed\_checkpointable.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 引入必要的类型相关模块
from typing import Any, Protocol, runtime_checkable

# 引入 PyTorch 模块
import torch

# 定义一个运行时可检查的协议
@runtime_checkable
class _Checkpointable(Protocol):  # noqa: PYI046
    """
    Interface for checkpointable objects.
    Implemented as a protocol, implicit subtyping is supported so subclasses do not need to inherit this explicitly.
    This is to allow arbitrary objects/tensor subclasses to hook into DCP seamlessly through implementing the interface.
    """

    def __create_write_items__(self, fqn: str, object: Any):
        """
        根据对象内容返回 WriteItems 列表。
        """
        raise NotImplementedError(
            "_Checkpointable._create_write_items is not implemented"
        )

    def __create_chunk_list__(self):
        """
        根据对象内容返回 ChunkStorageMetadata 列表。
        """
        raise NotImplementedError(
            "_Checkpointable._create_chunk_list is not implemented"
        )

    def __get_tensor_shard__(self, index) -> torch.Tensor:
        """
        根据 MetadataIndex 返回一个 'torch.Tensor' 分片。
        """
        raise NotImplementedError(
            "_Checkpointable._get_tensor_shard is not implemented"
        )
```