# `.\pytorch\torch\distributed\_shard\checkpoint\__init__.py`

```
# 引入系统模块和警告模块
import sys
import warnings

# 引入 torch 库，并从中导入分布式检查点相关的所有内容
import torch
from torch.distributed.checkpoint import *  # noqa: F403

# 使用警告模块捕获所有警告
with warnings.catch_warnings():
    # 设置警告的处理方式为始终显示
    warnings.simplefilter("always")
    # 发出警告，提示用户 `torch.distributed._shard.checkpoint` 将被弃用，建议使用 `torch.distributed.checkpoint`
    warnings.warn(
        "`torch.distributed._shard.checkpoint` will be deprecated, "
        "use `torch.distributed.checkpoint` instead",
        DeprecationWarning,
        stacklevel=2,
    )

# 将旧模块路径映射到新的分布式检查点模块，以确保向后兼容性
sys.modules["torch.distributed._shard.checkpoint"] = torch.distributed.checkpoint
```