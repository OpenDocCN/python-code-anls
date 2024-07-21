# `.\pytorch\torch\nn\intrinsic\quantized\modules\bn_relu.py`

```
# 导入 torch 模块中的量化推理操作：二维批归一化和激活函数合并操作
from torch.ao.nn.intrinsic.quantized import BNReLU2d
# 导入 torch 模块中的量化推理操作：三维批归一化和激活函数合并操作
from torch.ao.nn.intrinsic.quantized import BNReLU3d

# 定义一个列表，包含本模块中公开的量化操作类名称，用于模块导入时的限定性导入
__all__ = [
    'BNReLU2d',  # 二维批归一化和激活函数合并操作
    'BNReLU3d',  # 三维批归一化和激活函数合并操作
]
```