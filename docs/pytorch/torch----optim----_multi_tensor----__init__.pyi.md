# `.\pytorch\torch\optim\_multi_tensor\__init__.pyi`

```py
# 从 functools 模块导入 partial 函数，用于创建偏函数
from functools import partial

# 从 torch 模块导入优化器类 optim

# 创建 Adam 优化器的偏函数，foreach=True 表示支持并行优化
Adam = partial(optim.Adam, foreach=True)

# 创建 AdamW 优化器的偏函数，foreach=True 表示支持并行优化
AdamW = partial(optim.AdamW, foreach=True)

# 创建 NAdam 优化器的偏函数，foreach=True 表示支持并行优化
NAdam = partial(optim.NAdam, foreach=True)

# 创建 SGD 优化器的偏函数，foreach=True 表示支持并行优化
SGD = partial(optim.SGD, foreach=True)

# 创建 RAdam 优化器的偏函数，foreach=True 表示支持并行优化
RAdam = partial(optim.RAdam, foreach=True)

# 创建 RMSprop 优化器的偏函数，foreach=True 表示支持并行优化
RMSprop = partial(optim.RMSprop, foreach=True)

# 创建 Rprop 优化器的偏函数，foreach=True 表示支持并行优化
Rprop = partial(optim.Rprop, foreach=True)

# 创建 ASGD 优化器的偏函数，foreach=True 表示支持并行优化
ASGD = partial(optim.ASGD, foreach=True)

# 创建 Adamax 优化器的偏函数，foreach=True 表示支持并行优化
Adamax = partial(optim.Adamax, foreach=True)

# 创建 Adadelta 优化器的偏函数，foreach=True 表示支持并行优化
Adadelta = partial(optim.Adadelta, foreach=True)

# 创建 Adagrad 优化器的偏函数，foreach=True 表示支持并行优化
Adagrad = partial(optim.Adagrad, foreach=True)
```