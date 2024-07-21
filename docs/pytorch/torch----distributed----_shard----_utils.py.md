# `.\pytorch\torch\distributed\_shard\_utils.py`

```py
# 导入必要的库和模块
from typing import Sequence  # 引入Sequence类型用于类型提示

import torch  # 导入PyTorch库
from torch.distributed._shard.metadata import ShardMetadata  # 从torch.distributed._shard.metadata模块导入ShardMetadata类

# 弃用警告信息
DEPRECATE_MSG = "Please use DTensor instead and we are deprecating ShardedTensor."

# 根据索引偏移和大小缩小张量的函数
def narrow_tensor_by_index(
    tensor: torch.Tensor,
    offsets: Sequence[int],
    sizes: Sequence[int],
) -> torch.Tensor:
    """
    根据 ``offsets`` 和 ``sizes`` 缩小张量。
    """
    narrowed_tensor = tensor  # 初始化缩小后的张量为输入张量本身
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            # 通过narrow方法在指定维度idx上缩小张量，此处不希望在narrow操作中记录autograd信息，
            # 'local_shard' 应在autograd图中作为叶子变量。
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor  # 返回缩小后的张量


def narrow_tensor(tensor: torch.Tensor, metadata: ShardMetadata) -> torch.Tensor:
    """
    根据元数据缩小张量。
    """
    return narrow_tensor_by_index(tensor, metadata.shard_offsets, metadata.shard_sizes)
```