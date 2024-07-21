# `.\pytorch\torch\ao\nn\qat\modules\linear.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入特定于加速器的线性模块
from torch.ao.nn.intrinsic import LinearReLU
# 导入参数化工具函数
from torch.nn.utils.parametrize import (
    is_parametrized,
    type_before_parametrizations,
    transfer_parametrizations_and_params,
)

# 声明公开的模块列表
__all__ = [
    "Linear"
]

# 自定义线性模块，支持量化感知训练
class Linear(nn.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    # 定义浮点数模块作为参考
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None, device=None, dtype=None) -> None:
        # 设置工厂参数字典
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类构造函数初始化线性模块
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        # 断言确保必须提供 qconfig 参数
        assert qconfig, 'qconfig must be provided for QAT module'
        # 设置量化配置
        self.qconfig = qconfig
        # 根据工厂参数创建权重的假量化模块
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    # 前向传播函数
    def forward(self, input):
        # 使用假量化后的权重进行线性变换，并加上偏置
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)

    # 从浮点数模块创建量化感知模块的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module or qparams_dict
            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        # 断言确保输入模块是指定的浮点数模块类型
        assert type_before_parametrizations(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        # 断言确保浮点数模块有 qconfig 定义
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        # 断言确保浮点数模块的 qconfig 是有效的
        assert mod.qconfig, "Input float module must have a valid qconfig"
        
        # 如果输入模块是 LinearReLU 类型，则使用其第一个模块
        if type_before_parametrizations(mod) == LinearReLU:
            mod = mod[0]

        # 获取模块的量化配置
        qconfig = mod.qconfig
        # 创建新的量化感知线性模块
        qat_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, qconfig=qconfig)

        # 如果模块的权重参数被参数化了，则进行参数转移
        if is_parametrized(mod, "weight"):
            transfer_parametrizations_and_params(mod, qat_linear, "weight")
        else:
            qat_linear.weight = mod.weight

        # 如果模块的偏置参数被参数化了，则进行参数转移
        if is_parametrized(mod, "bias"):
            transfer_parametrizations_and_params(mod, qat_linear, "bias")
        else:
            qat_linear.bias = mod.bias

        return qat_linear

    # 将当前量化感知模块转换为浮点数模块的方法
    def to_float(self):
        # 创建一个新的普通线性模块
        linear = torch.nn.Linear(self.in_features, self.out_features, self.bias is not None)
        # 将当前模块的权重参数设置为新模块的参数
        linear.weight = torch.nn.Parameter(self.weight.detach())
        # 如果有偏置参数，则将当前模块的偏置参数设置为新模块的参数
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        # 将当前模块的训练状态应用于新模块
        linear.train(self.training)
        return linear
```