# `.\pytorch\torch\ao\nn\intrinsic\quantized\modules\conv_add.py`

```
# mypy: allow-untyped-defs
# 导入 PyTorch 库
import torch
# 导入 PyTorch AO（AI优化）的神经网络内部加速模块
import torch.ao.nn.intrinsic
# 导入 PyTorch AO 的量化训练加速模块
import torch.ao.nn.intrinsic.qat
# 导入 PyTorch 的函数模块
import torch.nn.functional as F
# 导入 PyTorch AO 的量化模块
import torch.ao.nn.quantized as nnq

# 从 nnq 模块中导入 _reverse_repeat_padding 函数
_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding

# 定义 ConvAdd2d 类，继承自 nnq.Conv2d
class ConvAdd2d(nnq.Conv2d):
    r"""
    A ConvAdd2d module is a fused module of Conv2d and Add

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    # 类属性，指定浮点数模块为 torch.ao.nn.intrinsic.ConvAdd2d，忽略类型检查
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAdd2d  # type: ignore[assignment]

    # 构造方法，初始化 ConvAdd2d 类的实例
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        # 调用父类构造方法，初始化 nnq.Conv2d 的实例
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

    # 前向传播方法定义
    def forward(self, input, extra_input):
        # 检查输入张量的形状是否为 (N, C, H, W)
        # 由于 JIT 问题，临时使用 len(shape) 代替 ndim
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        # 如果填充模式不是 'zeros'，则计算反转的重复填充
        _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
        # 对输入张量进行填充，使用反转的重复填充和指定的填充模式
        input = F.pad(input, _reversed_padding_repeated_twice,
                      mode=self.padding_mode)
        # 调用量化的 conv2d_add 操作，执行量化卷积加操作
        return torch.ops.quantized.conv2d_add(
            input, extra_input, self._packed_params, self.scale, self.zero_point)

    # 获取类名的辅助方法
    def _get_name(self):
        return 'QuantizedConvAdd2d'

    # 从浮点数模块转换方法，返回基于浮点数模块的量化卷积加模块
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    # 从参考量化卷积模块转换方法，返回基于参考量化卷积的量化卷积加模块
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)

# 定义 ConvAddReLU2d 类，继承自 nnq.Conv2d
class ConvAddReLU2d(nnq.Conv2d):
    r"""
    A ConvAddReLU2d module is a fused module of Conv2d, Add and Relu

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """
    # 类属性，指定浮点数模块为 torch.ao.nn.intrinsic.ConvAddReLU2d，忽略类型检查
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAddReLU2d  # type: ignore[assignment]

    # 构造方法，初始化 ConvAddReLU2d 类的实例
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        # 调用父类构造方法，初始化 nnq.Conv2d 的实例
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)
    # 对输入和额外输入执行前向传播操作
    def forward(self, input, extra_input):
        # 由于 JIT 问题，暂时使用 len(shape) 而不是 ndim
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            # 如果输入的形状不是 `(N, C, H, W)`，则抛出数值错误
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != 'zeros':
            # 如果填充模式不是 'zeros'，则根据填充方式进行填充
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 调用量化卷积加ReLU操作，返回结果
        return torch.ops.quantized.conv2d_add_relu(
            input, extra_input, self._packed_params, self.scale, self.zero_point)

    # 返回当前类的名称
    def _get_name(self):
        return 'QuantizedConvAddReLU2d'

    # 从浮点模型创建量化模型
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    # 从参考量化卷积模型创建当前模型
    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
```