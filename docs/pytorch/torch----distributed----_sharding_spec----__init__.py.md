# `.\pytorch\torch\distributed\_sharding_spec\__init__.py`

```
# 为了向后兼容，保留旧的包引入，一旦所有内容迁移到 `torch.distributed._shard` 包中，应该删除此文件。
import sys  # 导入系统模块，用于操作系统相关功能
import warnings  # 导入警告模块，用于发出和控制警告消息

import torch  # 导入 PyTorch 库
from torch.distributed._shard.sharding_spec import *  # 导入 `_sharding_spec` 包的所有内容，禁止 Flake8 检查

# 使用警告模块捕获所有警告
with warnings.catch_warnings():
    warnings.simplefilter("always")  # 设置警告过滤器，始终显示警告
    # 发出警告，提示用户 `torch.distributed._sharding_spec` 将被弃用，建议使用 `torch.distributed._shard.sharding_spec` 替代
    warnings.warn(
        "`torch.distributed._sharding_spec` will be deprecated, "
        "use `torch.distributed._shard.sharding_spec` instead",
        DeprecationWarning,
        stacklevel=2,  # 警告的堆栈级别为 2
    )

import torch.distributed._shard.sharding_spec as _sharding_spec  # 导入 `_sharding_spec` 模块，并重命名为 `_sharding_spec`

# 将 `torch.distributed._sharding_spec` 模块注册到 `sys.modules` 中，使得旧的引用指向新的 `_sharding_spec` 模块
sys.modules["torch.distributed._sharding_spec"] = _sharding_spec
```