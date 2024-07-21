# `.\pytorch\torch\ao\nn\intrinsic\quantized\modules\conv_relu.py`

```
# mypy: allow-untyped-defs

# 引入 PyTorch 相关模块
import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq

# 从 torch.nn.utils 中导入函数 fuse_conv_bn_weights
from torch.nn.utils import fuse_conv_bn_weights

# 定义模块公开接口
__all__ = [
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
]

# 导入 nnq 模块中的 _reverse_repeat_padding 函数
_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding

# TODO: factor out the common parts to ConvNd
# ConvReLU1d 类，继承自 nnq.Conv1d
class ConvReLU1d(nnq.Conv1d):
    r"""
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv1d

    """
    # _FLOAT_MODULE 类型为 torch.ao.nn.intrinsic.ConvReLU1d，忽略类型检查
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU1d  # type: ignore[assignment]

    # 初始化方法
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

    # 前向传播方法
    def forward(self, input):
        # 临时使用 len(shape) 代替 ndim，因为 JIT 存在问题
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        # 如果 padding_mode 不是 'zeros'，则进行填充处理
        if self.padding_mode != 'zeros':
            # Conv1d 中的 padding 存储为 (p, p)，需要转换为 (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 调用 torch.ops.quantized.conv1d_relu 进行量化卷积和ReLU操作
        return torch.ops.quantized.conv1d_relu(
            input, self._packed_params, self.scale, self.zero_point)

    # 返回模块名称的方法
    def _get_name(self):
        return 'QuantizedConvReLU1d'

    # 从 float 模型转换为量化模型的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) == torch.ao.nn.intrinsic.qat.ConvBnReLU1d:
            assert mod.bn.running_var is not None and mod.bn.running_mean is not None
            # 调用 fuse_conv_bn_weights 函数进行权重和偏置的融合
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
        return super().from_float(mod, use_precomputed_fake_quant)

    # 从参考量化模型创建量化模型的类方法
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        assert type(ref_qconv) != torch.ao.nn.intrinsic.ConvBnReLU1d, \
            "BatchNorm1d should be fused into Conv1d before converting to reference module"
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)

# ConvReLU2d 类，继承自 nnq.Conv2d
class ConvReLU2d(nnq.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    # 将 torch.ao.nn.intrinsic.ConvReLU2d 赋值给 _FLOAT_MODULE 变量，忽略类型检查
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU2d  # type: ignore[assignment]

    # 初始化方法，设置卷积层参数及相关选项
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        # 调用父类的初始化方法，设置卷积层的基本参数
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

    # 前向传播方法，执行量化卷积并整合ReLU操作
    def forward(self, input):
        # 由于 JIT 问题，暂时使用 len(shape) 替代 ndim
        # https://github.com/pytorch/pytorch/issues/23890
        # 检查输入张量的形状是否为 `(N, C, H, W)`
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        # 如果 padding_mode 不是 'zeros'，则执行反向填充操作
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 执行量化卷积并ReLU操作
        return torch.ops.quantized.conv2d_relu(
            input, self._packed_params, self.scale, self.zero_point)

    # 返回当前模块的名称
    def _get_name(self):
        return 'QuantizedConvReLU2d'

    # 类方法，从浮点模型转换为量化模型
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 如果输入模型是 torch.ao.nn.intrinsic.qat.ConvBnReLU2d 类型，则进行权重融合操作
        if type(mod) == torch.ao.nn.intrinsic.qat.ConvBnReLU2d:
            assert mod.bn.running_var is not None and mod.bn.running_mean is not None
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                mod.bn.eps, mod.bn.weight, mod.bn.bias)
        # 调用父类方法，从浮点模型创建量化模型
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    # 类方法，从参考量化卷积模型创建新的量化卷积ReLU模型
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        # 断言输入模型不是 torch.ao.nn.intrinsic.ConvBnReLU2d 类型，否则报错
        assert type(ref_qconv) != torch.ao.nn.intrinsic.ConvBnReLU2d, \
            "BatchNorm2d should be fused into Conv2d before converting to reference module"
        # 调用父类方法，从参考量化卷积模型创建新的量化卷积ReLU模型
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
class ConvReLU3d(nnq.Conv3d):
    r"""
    A ConvReLU3d module is a fused module of Conv3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv3d`.

    Attributes: Same as torch.ao.nn.quantized.Conv3d

    """
    # 定义_FLOAT_MODULE作为torch.ao.nn.intrinsic.ConvReLU3d的别名，类型为忽略赋值类型的注释
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU3d  # type: ignore[assignment]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        # 断言padding_mode不是'reflect'，因为Conv3d不支持反射填充
        assert padding_mode != 'reflect', "Conv3d does not support reflection padding"
        # 调用父类构造函数，初始化Conv3d部分的参数
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

    def forward(self, input):
        # 由于JIT问题，暂时使用len(shape)代替ndim
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        # 如果padding_mode不是'zeros'，则反转并重复填充，然后使用指定的填充模式pad输入
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 调用torch.ops.quantized.conv3d_relu执行量化的Conv3d和ReLU操作
        return torch.ops.quantized.conv3d_relu(
            input, self._packed_params, self.scale, self.zero_point)

    def _get_name(self):
        # 返回该类的名称字符串'QuantizedConvReLU3d'
        return 'QuantizedConvReLU3d'

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 如果mod是torch.ao.nn.intrinsic.qat.ConvBnReLU3d类型，则融合Conv和BatchNorm参数
        if type(mod) == torch.ao.nn.intrinsic.qat.ConvBnReLU3d:
            assert mod.bn.running_var is not None and mod.bn.running_mean is not None
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight,
                mod.bias,
                mod.bn.running_mean,
                mod.bn.running_var,
                mod.bn.eps,
                mod.bn.weight,
                mod.bn.bias,
            )
        # 调用父类的from_float方法，将浮点数模型转换为量化模型
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        # 断言ref_qconv不是torch.ao.nn.intrinsic.ConvBnReLU3d类型，因为在转换为参考模块之前应该融合BatchNorm3d
        assert type(ref_qconv) != torch.ao.nn.intrinsic.ConvBnReLU3d, \
            "BatchNorm3d should be fused into Conv3d before converting to reference module"
        # 调用父类的from_reference方法，根据参考量化Conv3d创建新的量化ConvReLU3d对象
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
```