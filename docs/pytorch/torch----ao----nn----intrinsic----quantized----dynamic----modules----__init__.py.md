# `.\pytorch\torch\ao\nn\intrinsic\quantized\dynamic\modules\__init__.py`

```
# 导入 torch 库，用于神经网络和张量操作
import torch
# 从当前目录下的 linear_relu 模块中导入 LinearReLU 类
from .linear_relu import LinearReLU

# 定义一个列表 __all__，指定了在该模块中公开的接口名称
__all__ = [
    'LinearReLU',  # 将 LinearReLU 类添加到 __all__ 列表中
]
```