# `.\pytorch\torch\nn\intrinsic\quantized\modules\conv_relu.py`

```
# 导入 torch 中量化的 1 维、2 维和 3 维卷积加ReLU模块
from torch.ao.nn.intrinsic.quantized import ConvReLU1d
from torch.ao.nn.intrinsic.quantized import ConvReLU2d
from torch.ao.nn.intrinsic.quantized import ConvReLU3d

# 定义一个公开的列表，包含了本模块中需要公开的量化卷积加ReLU模块名字
__all__ = [
    'ConvReLU1d',  # 将 ConvReLU1d 添加到公开列表中
    'ConvReLU2d',  # 将 ConvReLU2d 添加到公开列表中
    'ConvReLU3d',  # 将 ConvReLU3d 添加到公开列表中
]
```