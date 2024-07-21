# `.\pytorch\torch\nn\parallel\__init__.py`

```py
# 设置类型检查工具 mypy 允许未定义类型的函数
# 导入被弃用的功能模块，来自 typing_extensions 库
from typing_extensions import deprecated

# 从内部模块中导入以下功能和类
from .data_parallel import data_parallel, DataParallel
from .distributed import DistributedDataParallel
from .parallel_apply import parallel_apply
from .replicate import replicate
from .scatter_gather import gather, scatter

# 将以下内容添加到模块的公共接口中
__all__ = [
    "replicate",
    "scatter",
    "parallel_apply",
    "gather",
    "data_parallel",
    "DataParallel",
    "DistributedDataParallel",
]

# 使用 @deprecated 装饰器标记此类，提醒用户该类即将弃用
@deprecated(
    "`torch.nn.parallel.DistributedDataParallelCPU` is deprecated, "
    "please use `torch.nn.parallel.DistributedDataParallel` instead.",
    category=FutureWarning,
)
# 继承自 DistributedDataParallel 类，用于处理分布式数据并行
class DistributedDataParallelCPU(DistributedDataParallel):
    pass
```