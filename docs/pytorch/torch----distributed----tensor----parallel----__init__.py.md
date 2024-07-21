# `.\pytorch\torch\distributed\tensor\parallel\__init__.py`

```py
# 导入分布式训练所需的模块和函数
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.loss import loss_parallel
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)

# 定义 __all__ 列表，指定了当前模块中公开的所有名称
__all__ = [
    "ColwiseParallel",        # 列并行处理器，用于指定列级别的并行处理
    "ParallelStyle",          # 并行风格类，定义了并行模块的风格
    "PrepareModuleInput",     # 准备模块输入的函数，用于模块输入的预处理
    "PrepareModuleOutput",    # 准备模块输出的函数，用于模块输出的后处理
    "RowwiseParallel",        # 行并行处理器，用于指定行级别的并行处理
    "SequenceParallel",       # 序列并行处理器，用于指定序列级别的并行处理
    "parallelize_module",     # 并行化模块的函数，用于分布式训练中的模块并行
    "loss_parallel",          # 并行损失函数，用于在分布式设置中进行损失的并行计算
]
```