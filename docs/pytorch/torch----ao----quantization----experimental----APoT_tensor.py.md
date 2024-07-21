# `.\pytorch\torch\ao\quantization\experimental\APoT_tensor.py`

```
# 引入 torch 库，用于张量操作
# 引入 APoTQuantizer 类，来自 torch.ao.quantization.experimental.quantizer 模块，用于进行 APoT 量化
import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# 用于存储 APoT 量化后的张量的类
class TensorAPoT:
    # 量化器对象，类型为 APoTQuantizer
    quantizer: APoTQuantizer
    # 存储量化后的张量数据，类型为 torch.Tensor
    data: torch.Tensor

    # 初始化方法，接收 APoTQuantizer 类型的 quantizer 和 torch.Tensor 类型的 apot_data 参数
    def __init__(self, quantizer: APoTQuantizer, apot_data: torch.Tensor):
        # 将传入的 quantizer 参数赋值给对象的 quantizer 属性
        self.quantizer = quantizer
        # 将传入的 apot_data 参数赋值给对象的 data 属性
        self.data = apot_data

    # 返回对象存储的量化后的整数表示的张量数据
    def int_repr(self):
        return self.data
```