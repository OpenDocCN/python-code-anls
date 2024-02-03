# `bitsandbytes\bitsandbytes\optim\__init__.py`

```py
# 导入 COMPILED_WITH_CUDA 变量，用于检查是否使用了 CUDA 编译
from bitsandbytes.cextension import COMPILED_WITH_CUDA

# 导入 Adagrad 相关类
from .adagrad import Adagrad, Adagrad8bit, Adagrad32bit

# 导入 Adam 相关类
from .adam import Adam, Adam8bit, Adam32bit, PagedAdam, PagedAdam8bit, PagedAdam32bit

# 导入 AdamW 相关类
from .adamw import (
    AdamW,
    AdamW8bit,
    AdamW32bit,
    PagedAdamW,
    PagedAdamW8bit,
    PagedAdamW32bit,
)

# 导入 LAMB 相关类
from .lamb import LAMB, LAMB8bit, LAMB32bit

# 导入 LARS 相关类
from .lars import LARS, LARS8bit, LARS32bit, PytorchLARS

# 导入 Lion 相关类
from .lion import Lion, Lion8bit, Lion32bit, PagedLion, PagedLion8bit, PagedLion32bit

# 导入 GlobalOptimManager 类
from .optimizer import GlobalOptimManager

# 导入 RMSprop 相关类
from .rmsprop import RMSprop, RMSprop8bit, RMSprop32bit

# 导入 SGD 相关类
from .sgd import SGD, SGD8bit, SGD32bit
```