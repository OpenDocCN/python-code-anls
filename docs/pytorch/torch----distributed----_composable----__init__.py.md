# `.\pytorch\torch\distributed\_composable\__init__.py`

```py
# 从当前目录导入 checkpoint_activation 模块中的 checkpoint 函数
from .checkpoint_activation import checkpoint
# 从当前目录导入 contract 模块中的 _get_registry 和 contract 函数
from .contract import _get_registry, contract
# 从当前目录导入 fully_shard 模块中的 fully_shard 函数
from .fully_shard import fully_shard
# 从当前目录导入 replicate 模块中的 replicate 函数
from .replicate import replicate
```