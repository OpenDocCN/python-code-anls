# `.\pytorch\torch\ao\nn\intrinsic\quantized\modules\bn_relu.py`

```py
# mypy: allow-untyped-defs

# 导入必要的 PyTorch 模块
import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq

# 定义公开的类名列表，这些类名可以被外部导入
__all__ = [
    "BNReLU2d",
    "BNReLU3d"
]

# 定义 BNReLU2d 类，继承自 nnq.BatchNorm2d
class BNReLU2d(nnq.BatchNorm2d):
    r"""
    A BNReLU2d module is a fused module of BatchNorm2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm2d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm2d

    """
    # 浮点数模块的参考类
    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU2d

    # 初始化函数，接受多个参数，包括 num_features, eps, momentum, device, dtype
    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        super().__init__(num_features, eps=eps, momentum=momentum, device=device, dtype=dtype)

    # 前向传播函数，处理输入 input
    def forward(self, input):
        # 检查输入的形状是否为四维 `(N, C, H, W)`
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        # 调用量化操作的批量归一化与ReLU激活函数
        return torch.ops.quantized.batch_norm2d_relu(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)

    # 返回类名的字符串形式
    def _get_name(self):
        return 'QuantizedBNReLU2d'

    # 从浮点数模块创建量化模块的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # TODO: Add qat support for BNReLU2d
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    # 从参考模块创建量化模块的类方法
    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)

# 定义 BNReLU3d 类，继承自 nnq.BatchNorm3d
class BNReLU3d(nnq.BatchNorm3d):
    r"""
    A BNReLU3d module is a fused module of BatchNorm3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm3d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm3d

    """
    # 浮点数模块的参考类
    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU3d

    # 初始化函数，接受多个参数，包括 num_features, eps, momentum, device, dtype
    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        super().__init__(num_features, eps=eps, momentum=momentum, device=device, dtype=dtype)

    # 前向传播函数，处理输入 input
    def forward(self, input):
        # 检查输入的形状是否为五维 `(N, C, D, H, W)`
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        # 调用量化操作的批量归一化与ReLU激活函数
        return torch.ops.quantized.batch_norm3d_relu(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)

    # 返回类名的字符串形式
    def _get_name(self):
        return 'QuantizedBNReLU3d'

    # 从浮点数模块创建量化模块的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # TODO: Add qat support for BNReLU3d
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    # 从参考模块创建量化模块的类方法
    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)
```