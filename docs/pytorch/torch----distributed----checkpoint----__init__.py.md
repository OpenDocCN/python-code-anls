# `.\pytorch\torch\distributed\checkpoint\__init__.py`

```
# 导入自定义模块中的异常类
from .api import CheckpointException
# 导入默认的加载和保存规划器类
from .default_planner import DefaultLoadPlanner, DefaultSavePlanner
# 导入文件系统读写相关的类
from .filesystem import FileSystemReader, FileSystemWriter
# 导入存储元数据相关的类
from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
# 导入加载分片优化器状态字典的函数
from .optimizer import load_sharded_optimizer_state_dict
# 导入加载和保存计划相关的类
from .planner import LoadPlan, LoadPlanner, ReadItem, SavePlan, SavePlanner, WriteItem
# 导入状态字典加载相关的函数和类
from .state_dict_loader import load, load_state_dict
# 导入状态字典保存相关的异步保存函数和普通保存函数
from .state_dict_saver import async_save, save, save_state_dict
# 导入存储读写相关的类
from .storage import StorageReader, StorageWriter
```