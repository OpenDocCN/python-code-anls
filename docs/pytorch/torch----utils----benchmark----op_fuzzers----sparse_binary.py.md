# `.\pytorch\torch\utils\benchmark\op_fuzzers\sparse_binary.py`

```
# 添加类型检查忽略声明，允许未类型化的定义
mypy: allow-untyped-defs

# 导入 numpy 库并重命名为 np
import numpy as np

# 导入 torch 库
import torch

# 从 torch.utils.benchmark 模块中导入 Fuzzer, FuzzedParameter, ParameterAlias, FuzzedSparseTensor 类
from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedSparseTensor

# 定义最小维度大小常量
_MIN_DIM_SIZE = 16

# 定义最大维度大小常量
_MAX_DIM_SIZE = 16 * 1024 ** 2

# 使用列表生成器创建 2 的幂的大小元组，范围从 log2(_MIN_DIM_SIZE) 到 log2(_MAX_DIM_SIZE)
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),
    int(np.log2(_MAX_DIM_SIZE)) + 1,
))


class BinaryOpSparseFuzzer(Fuzzer):
```