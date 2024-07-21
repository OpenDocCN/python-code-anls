# `.\pytorch\torch\ao\nn\intrinsic\qat\modules\linear_relu.py`

```py
# mypy: allow-untyped-defs
# 引入PyTorch相关模块
import torch
import torch.ao.nn.qat as nnqat  # 导入量化感知训练相关模块
import torch.ao.nn.intrinsic as nni  # 导入内置加速模块
import torch.nn.functional as F  # 导入PyTorch中的函数式接口模块

# 定义一个继承自量化感知训练模块和融合模块的线性ReLU模块
class LinearReLU(nnqat.Linear, nni._FusedModule):
    r"""
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.ao.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.qat.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearReLU  # type: ignore[assignment]

    # 初始化函数，定义模块的输入维度、输出维度、是否包含偏置项和量化配置
    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None):
        super().__init__(in_features, out_features, bias, qconfig)

    # 前向传播函数，对输入数据进行线性变换后应用ReLU激活函数
    def forward(self, input):
        return F.relu(F.linear(input, self.weight_fake_quant(self.weight), self.bias))

    # 类方法，从浮点模型转换得到量化感知训练模型
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant)

    # 将当前模型转换为浮点模型
    def to_float(self):
        linear = torch.nn.Linear(self.in_features, self.out_features, self.bias is not None)
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        relu = torch.nn.ReLU()
        return torch.ao.nn.intrinsic.LinearReLU(linear, relu)
```