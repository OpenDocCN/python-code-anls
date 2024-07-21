# `.\pytorch\torch\distributed\_sharded_tensor\__init__.py`

```py
# 为了向后兼容而保留旧包，一旦所有内容都迁移到 `torch.distributed._shard` 包中，应删除此文件。

# 导入系统模块
import sys
# 导入警告模块，用于显示警告信息
import warnings

# 导入 Torch 库
import torch
# 从 `torch.distributed._shard.sharded_tensor` 中导入所有内容，禁止 Flake8 校验 (F403)
from torch.distributed._shard.sharded_tensor import *

# 使用 `warnings.catch_warnings()` 上下文管理器捕获警告
with warnings.catch_warnings():
    # 始终显示警告
    warnings.simplefilter("always")
    # 发出警告，提示使用 `torch.distributed._shard.sharded_tensor` 取代 `torch.distributed._sharded_tensor`
    warnings.warn(
        "`torch.distributed._sharded_tensor` will be deprecated, "
        "use `torch.distributed._shard.sharded_tensor` instead",
        DeprecationWarning,
        stacklevel=2,
    )

# 将模块字典中的键 "torch.distributed._sharded_tensor" 指向 `torch.distributed._shard.sharded_tensor`
sys.modules[
    "torch.distributed._sharded_tensor"
] = torch.distributed._shard.sharded_tensor
```