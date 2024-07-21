# `.\pytorch\torch\distributed\_shard\sharded_tensor\_ops\__init__.py`

```py
# 导入需要的模块或函数

# 导入 torch 分布式 sharded_tensor 模块下的 misc_ops 子模块
import torch.distributed._shard.sharded_tensor._ops.misc_ops

# 导入 torch 分布式 sharded_tensor 模块下的 tensor_ops 子模块
import torch.distributed._shard.sharded_tensor._ops.tensor_ops

# 从 torch 分布式 sharding_spec.chunk_sharding_spec_ops.embedding 模块导入 sharded_embedding 函数
# 这是用于处理分片嵌入的函数
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding import (
    sharded_embedding,
)

# 从 torch 分布式 sharding_spec.chunk_sharding_spec_ops.embedding_bag 模块导入 sharded_embedding_bag 函数
# 这是用于处理分片嵌入袋的函数
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding_bag import (
    sharded_embedding_bag,
)

# 从当前目录下的 binary_cmp 模块导入 allclose 和 equal 函数
from .binary_cmp import allclose, equal

# 从当前目录下的 init 模块导入 constant_, kaiming_uniform_, normal_, uniform_ 函数
from .init import constant_, kaiming_uniform_, normal_, uniform_
```