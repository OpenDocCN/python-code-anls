# `.\pytorch\torch\ao\nn\quantized\modules\__init__.py`

```py
# 添加类型提示，允许未经类型定义的函数
mypy: allow-untyped-defs

# 导入 PyTorch 库
import torch

# 量化模块使用 `torch.nn` 和 `torch.ao.nn.quantizable` 包。
# 然而，`quantizable` 包使用“延迟导入”以避免循环依赖。
# 因此，我们需要在这里导入它，以确保在模块中使用之前已解析。
import torch.ao.nn.quantizable

# 从 PyTorch 中导入 MaxPool2d 池化模块
from torch.nn.modules.pooling import MaxPool2d

# 从本地模块中导入各种激活函数和组件
from .activation import ReLU6, Hardswish, ELU, LeakyReLU, Sigmoid, Softmax, MultiheadAttention, PReLU
from .dropout import Dropout
from .batchnorm import BatchNorm2d, BatchNorm3d
from .normalization import LayerNorm, GroupNorm, InstanceNorm1d, \
    InstanceNorm2d, InstanceNorm3d
from .conv import Conv1d, Conv2d, Conv3d
from .conv import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from .linear import Linear
from .embedding_ops import Embedding, EmbeddingBag
from .rnn import LSTM

# 从功能模块中导入特定的功能类
from .functional_modules import FloatFunctional, FXFloatFunctional, QFunctional

# 导出给外部的符号列表
__all__ = [
    'BatchNorm2d',
    'BatchNorm3d',
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
    'DeQuantize',  # 这个符号未在之前的代码中定义
    'ELU',
    'Embedding',
    'EmbeddingBag',
    'GroupNorm',
    'Hardswish',
    'InstanceNorm1d',
    'InstanceNorm2d',
    'InstanceNorm3d',
    'LayerNorm',
    'LeakyReLU',
    'Linear',
    'LSTM',
    'MultiheadAttention',
    'Quantize',  # 这个符号未在之前的代码中定义
    'ReLU6',
    'Sigmoid',
    'Softmax',
    'Dropout',
    'PReLU',
    # 封装模块
    'FloatFunctional',
    'FXFloatFunctional',
    'QFunctional',
]

# 定义一个量化模块的类
class Quantize(torch.nn.Module):
    r"""Quantizes an incoming tensor

    Args:
     `scale`: 输出量化张量的比例
     `zero_point`: 输出量化张量的零点
     `dtype`: 输出量化张量的数据类型
     `factory_kwargs`: 用于配置内部缓冲区初始化的 kwargs 字典。
         当前支持 `device` 和 `dtype`。例如，`factory_kwargs={'device': 'cuda', 'dtype': torch.float64}`
         将在当前 CUDA 设备上将内部缓冲区初始化为 `torch.float64` 类型。
         注意，`dtype` 仅适用于浮点型缓冲区。

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> qt = qm(t)
        >>> print(qt)
        tensor([[ 1., -1.],
                [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    # 初始化方法，设置量化参数和数据类型
    def __init__(self, scale, zero_point, dtype, factory_kwargs=None):
        # 调用静态方法，处理工厂关键字参数，确保符合torch.nn工厂参数的格式
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        # 调用父类初始化方法
        super().__init__()
        # 注册缓冲区，存储量化的比例因子（scale），使用torch.tensor创建张量
        self.register_buffer('scale', torch.tensor([scale], **factory_kwargs))
        # 注册缓冲区，存储量化的零点（zero_point），使用torch.tensor创建张量，指定数据类型为torch.long
        self.register_buffer('zero_point',
                             torch.tensor([zero_point], dtype=torch.long,
                                          **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        # 存储数据类型
        self.dtype = dtype

    # 前向传播方法，对输入张量X进行量化
    def forward(self, X):
        # 使用torch.quantize_per_tensor函数对张量X进行量化
        return torch.quantize_per_tensor(X, float(self.scale),
                                         int(self.zero_point), self.dtype)

    # 静态方法，从浮点模型中构建量化模型
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # 断言模型对象mod具有'activation_post_process'属性
        assert hasattr(mod, 'activation_post_process')
        # 调用'activation_post_process'属性的方法，计算量化参数（scale和zero_point）
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 返回一个Quantize对象，使用浮点数形式的scale和长整型形式的zero_point，指定dtype为'activation_post_process'的dtype
        return Quantize(scale.float().item(), zero_point.long().item(), mod.activation_post_process.dtype)

    # 返回额外的表示信息，描述量化模型的scale、zero_point和dtype
    def extra_repr(self):
        return f'scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}'
# 定义一个继承自torch.nn.Module的类，用于反量化输入的张量
class DeQuantize(torch.nn.Module):
    r"""Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """

    # 前向传播函数，接受量化的输入张量Xq，并返回其反量化后的张量
    def forward(self, Xq):
        return Xq.dequantize()

    # 从浮点数模型mod创建一个DeQuantize实例的静态方法
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return DeQuantize()
```