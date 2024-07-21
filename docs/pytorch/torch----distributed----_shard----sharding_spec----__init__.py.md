# `.\pytorch\torch\distributed\_shard\sharding_spec\__init__.py`

```
# 从torch.distributed._shard.metadata模块中导入ShardMetadata类
from torch.distributed._shard.metadata import ShardMetadata

# 从当前目录下的api模块中导入以下内容：
# _infer_sharding_spec_from_shards_metadata函数
# DevicePlacementSpec类
# EnumerableShardingSpec类
# PlacementSpec类
# ShardingSpec类
from .api import (
    _infer_sharding_spec_from_shards_metadata,
    DevicePlacementSpec,
    EnumerableShardingSpec,
    PlacementSpec,
    ShardingSpec,
)

# 从当前目录下的chunk_sharding_spec模块中导入ChunkShardingSpec类，并起一个别名ChunkShardingSpec
from .chunk_sharding_spec import ChunkShardingSpec as ChunkShardingSpec
```