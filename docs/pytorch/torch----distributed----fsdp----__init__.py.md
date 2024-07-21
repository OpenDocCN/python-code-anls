# `.\pytorch\torch\distributed\fsdp\__init__.py`

```py
# 导入模块 _flat_param 中的 FlatParameter 类别名为 FlatParameter
# 导入 fully_sharded_data_parallel 模块中的多个类和函数
from ._flat_param import FlatParameter as FlatParameter
from .fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    OptimStateKeyType,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)

# 模块中需要暴露给外部的变量名列表
__all__ = [
    "BackwardPrefetch",                    # 引入 BackwardPrefetch 类
    "CPUOffload",                          # 引入 CPUOffload 类
    "FullOptimStateDictConfig",            # 引入 FullOptimStateDictConfig 类
    "FullStateDictConfig",                 # 引入 FullStateDictConfig 类
    "FullyShardedDataParallel",            # 引入 FullyShardedDataParallel 类
    "LocalOptimStateDictConfig",           # 引入 LocalOptimStateDictConfig 类
    "LocalStateDictConfig",                # 引入 LocalStateDictConfig 类
    "MixedPrecision",                      # 引入 MixedPrecision 类
    "OptimStateDictConfig",                # 引入 OptimStateDictConfig 类
    "OptimStateKeyType",                   # 引入 OptimStateKeyType 类
    "ShardedOptimStateDictConfig",         # 引入 ShardedOptimStateDictConfig 类
    "ShardedStateDictConfig",              # 引入 ShardedStateDictConfig 类
    "ShardingStrategy",                    # 引入 ShardingStrategy 类
    "StateDictConfig",                     # 引入 StateDictConfig 类
    "StateDictSettings",                   # 引入 StateDictSettings 类
    "StateDictType",                       # 引入 StateDictType 类
]
```