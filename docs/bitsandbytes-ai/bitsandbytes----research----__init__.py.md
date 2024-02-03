# `bitsandbytes\bitsandbytes\research\__init__.py`

```py
# 从当前目录中导入 nn 模块
from . import nn
# 从 autograd._functions 模块中导入以下函数
from .autograd._functions import (
    # 导入 matmul_fp8_global 函数
    matmul_fp8_global,
    # 导入 matmul_fp8_mixed 函数
    matmul_fp8_mixed,
    # 导入 switchback_bnb 函数
    switchback_bnb,
)
```