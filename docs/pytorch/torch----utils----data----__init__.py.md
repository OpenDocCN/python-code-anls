# `.\pytorch\torch\utils\data\__init__.py`

```
# 从 torch.utils.data.dataloader 模块中导入以下类和函数
from torch.utils.data.dataloader import (
    _DatasetKind,       # 数据集种类的内部表示
    DataLoader,         # 数据加载器，用于加载数据集
    default_collate,     # 默认的数据集合并函数
    default_convert,     # 默认的数据类型转换函数
    get_worker_info,     # 获取当前工作进程的信息
)

# 从 torch.utils.data.datapipes._decorator 模块中导入以下装饰器函数
from torch.utils.data.datapipes._decorator import (
    argument_validation,                        # 参数验证装饰器
    functional_datapipe,                        # 功能型数据管道装饰器
    guaranteed_datapipes_determinism,           # 保证数据管道确定性装饰器
    non_deterministic,                          # 非确定性装饰器
    runtime_validation,                         # 运行时验证装饰器
    runtime_validation_disabled,                # 禁用运行时验证装饰器
)

# 从 torch.utils.data.datapipes.datapipe 模块中导入以下类
from torch.utils.data.datapipes.datapipe import (
    DataChunk,          # 数据块类
    DFIterDataPipe,     # 数据流迭代器类
    IterDataPipe,       # 迭代数据管道类
    MapDataPipe,        # 映射数据管道类
)

# 从 torch.utils.data.dataset 模块中导入以下类和函数
from torch.utils.data.dataset import (
    ChainDataset,       # 链式数据集类
    ConcatDataset,      # 连接数据集类
    Dataset,            # 基础数据集类
    IterableDataset,    # 可迭代数据集类
    random_split,       # 随机分割数据集函数
    StackDataset,       # 堆叠数据集类
    Subset,             # 子集类
    TensorDataset,      # 张量数据集类
)

# 从 torch.utils.data.distributed 模块中导入以下类
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器类

# 从 torch.utils.data.sampler 模块中导入以下类
from torch.utils.data.sampler import (
    BatchSampler,               # 批量采样器类
    RandomSampler,              # 随机采样器类
    Sampler,                    # 采样器基类
    SequentialSampler,          # 顺序采样器类
    SubsetRandomSampler,        # 子集随机采样器类
    WeightedRandomSampler,      # 加权随机采样器类
)

# __all__ 列表包含了所有导出的类和函数的名称，用于模块导入时的自动导入机制
__all__ = [
    "BatchSampler",
    "ChainDataset",
    "ConcatDataset",
    "DFIterDataPipe",
    "DataChunk",
    "DataLoader",
    "Dataset",
    "DistributedSampler",
    "IterDataPipe",
    "IterableDataset",
    "MapDataPipe",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "StackDataset",
    "Subset",
    "SubsetRandomSampler",
    "TensorDataset",
    "WeightedRandomSampler",
    "_DatasetKind",
    "argument_validation",
    "default_collate",
    "default_convert",
    "functional_datapipe",
    "get_worker_info",
    "guaranteed_datapipes_determinism",
    "non_deterministic",
    "random_split",
    "runtime_validation",
    "runtime_validation_disabled",
]

# 确保 __all__ 列表中的名称按字母顺序排列，用于模块导入时的一致性检查
assert __all__ == sorted(__all__)
```