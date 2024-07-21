# `.\pytorch\torch\nn\intrinsic\quantized\__init__.py`

```py
# 导入所有模块，包括当前目录下的 modules 模块
from .modules import *  # noqa: F403

# 为了确保客户可以使用下面列出的模块，而不需要直接导入它们
import torch.nn.intrinsic.quantized.dynamic

# 声明一个列表，包含了可以通过 `from <module> import *` 方式导入的符号名
__all__ = [
    'BNReLU2d',     # 二维批标准化和ReLU激活函数
    'BNReLU3d',     # 三维批标准化和ReLU激活函数
    'ConvReLU1d',   # 一维卷积和ReLU激活函数
    'ConvReLU2d',   # 二维卷积和ReLU激活函数
    'ConvReLU3d',   # 三维卷积和ReLU激活函数
    'LinearReLU',   # 线性层和ReLU激活函数组合
]
```