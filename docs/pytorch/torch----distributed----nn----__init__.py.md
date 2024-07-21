# `.\pytorch\torch\distributed\nn\__init__.py`

```py
# 导入 PyTorch 库
import torch

# 导入当前目录下的所有功能模块，禁止检查 F403
from .functional import *  # noqa: F403

# 检查是否支持分布式 RPC（远程过程调用）
if torch.distributed.rpc.is_available():
    # 如果支持分布式 RPC，导入远程模块 API
    from .api.remote_module import RemoteModule
```