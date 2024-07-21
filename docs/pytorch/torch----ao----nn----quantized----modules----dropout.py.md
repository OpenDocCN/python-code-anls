# `.\pytorch\torch\ao\nn\quantized\modules\dropout.py`

```
# mypy: allow-untyped-defs
# 导入PyTorch库
import torch

# 定义在模块中公开的类名列表
__all__ = ['Dropout']

# 定义一个名为Dropout的类，继承自torch.nn.Dropout类
class Dropout(torch.nn.Dropout):
    r"""This is the quantized equivalent of :class:`~torch.nn.Dropout`.
        And this is a placeholder to enable models where fp32 tensors
        had dropout to work with quantized tensors in train and eval mode.

    Args:
        p: probability of an element to be zeroed
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    # 前向传播方法，直接返回输入的数据
    def forward(self, input):
        return input

    # 获取类的名称的内部方法
    def _get_name(self):
        return 'QuantizedDropout'

    # 从一个浮点数模型转换为量化Dropout的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return cls(mod.p, mod.inplace)

    # 从参考模型转换为量化Dropout的类方法
    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.p, mod.inplace)
```