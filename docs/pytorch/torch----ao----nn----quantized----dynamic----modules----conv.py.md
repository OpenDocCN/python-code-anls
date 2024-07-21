# `.\pytorch\torch\ao\nn\quantized\dynamic\modules\conv.py`

```py
# mypy: allow-untyped-defs
r"""Dynamically quantized convolution modules."""

import torch  # 导入 PyTorch 模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 函数操作模块

from torch import Tensor  # 导入 Tensor 类型
from torch._ops import ops  # 导入 ops 操作模块
from torch.nn.common_types import _size_1_t  # 导入 _size_1_t 类型
from torch.nn.modules.utils import _single, _pair, _triple  # 导入尺寸处理工具函数
from torch.ao.nn.quantized.modules.conv import _reverse_repeat_padding  # 导入反向重复填充函数
import torch.ao.nn.quantized as nnq  # 导入动态量化模块
import warnings  # 导入警告模块

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']

# 定义 Conv1d 类，继承自 nnq.Conv1d，用于动态量化的卷积模块
class Conv1d(nnq.Conv1d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d` and :class:`~torch.ao.nn.quantized.dynamic.Conv1d`.

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.quantized.dynamic.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> output = m(input)

    """

    _FLOAT_MODULE = nn.Conv1d  # 类属性，指向原始的 nn.Conv1d 类
    _NNIQAT_CONV_BN_MODULE = None  # type: ignore[assignment]，类型标注：忽略赋值检查
    _NNI_CONV_RELU_MODULE = None  # type: ignore[assignment]，类型标注：忽略赋值检查

    # 初始化方法，定义了动态量化 Conv1d 的参数及其警告信息
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: _size_1_t = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 reduce_range=True):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _single(kernel_size)  # 处理 kernel_size 的尺寸为单元素元组
        stride = _single(stride)  # 处理 stride 的尺寸为单元素元组
        padding = padding if isinstance(padding, str) else _single(padding)  # 处理 padding 的尺寸为单元素元组或字符串
        dilation = _single(dilation)  # 处理 dilation 的尺寸为单元素元组

        # 调用父类的初始化方法，初始化动态量化 Conv1d 的参数
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, **factory_kwargs)

    # 返回当前类的名称字符串
    def _get_name(self):
        return 'DynamicQuantizedConv1d'
    # 对输入进行前向传播操作，接受一个三维张量作为输入，支持减少范围的选项
    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # 由于即时编译问题，临时使用 len(shape) 替代 ndim
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            # 如果输入张量的形状不是 (N, C, L)，则抛出 ValueError 异常
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != 'zeros':
            # Conv1d 中的 padding 存储形式为 (p, p)，需要获取 (p,) 形式
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            # 使用指定的 padding 方式对输入进行填充
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 调用量化的动态 1D 卷积操作，传入填充后的输入和预先打包的参数
        return ops.quantized.conv1d_dynamic(input, self._packed_params, reduce_range)
# 定义一个继承自`nnq.Conv2d`的类，实现动态量化的二维卷积模块，输入和输出均为浮点数张量。

# 类属性包括：
# - weight (Tensor): 从可学习权重参数中提取的打包张量。
# - scale (Tensor): 输出的缩放标量。
# - zero_point (Tensor): 输出的零点标量。

# 其他属性继承自`torch.nn.Conv2d`。

class Conv2d(nnq.Conv2d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d` and :class:`~torch.ao.nn.quantized.dynamic.Conv2d` and

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.dynamic.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    """
    _FLOAT_MODULE = nn.Conv2d
    _NNIQAT_CONV_BN_MODULE = None  # type: ignore[assignment]
    _NNI_CONV_RELU_MODULE = None  # type: ignore[assignment]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        # 发出警告，指出当前实现的模块数值精度较低，不建议使用
        warnings.warn(
            f"The current implementation of the {self._get_name()} module "
            "has poor numerical accuracy and its use is not recommended"
        )
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)  # 处理卷积核大小为元组或整数的情况
        stride = _pair(stride)  # 处理步长为元组或整数的情况
        padding = _pair(padding)  # 处理填充为元组或整数的情况
        dilation = _pair(dilation)  # 处理扩展为元组或整数的情况

        # 调用父类的构造方法初始化
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'DynamicQuantizedConv2d'

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # 临时使用 len(shape) 替代 ndim，因为存在 JIT 问题
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != 'zeros':
            # 计算反向重复填充模式下的填充量，并对输入进行填充
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 调用动态量化卷积的操作函数，返回输出张量
        return ops.quantized.conv2d_dynamic(
            input, self._packed_params, reduce_range)



# 定义一个继承自`nnq.Conv3d`的类，实现动态量化的三维卷积模块，输入和输出均为浮点数张量。

# 类的详细信息、参数和实现细节可参考相关文档。

class Conv3d(nnq.Conv3d):
    r"""A dynamically quantized conv module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    # `_FLOAT_MODULE`指向`nn.Conv3d`类，用于非量化版本的卷积操作
    _FLOAT_MODULE = nn.Conv3d
    # `_NNIQAT_CONV_BN_MODULE`和`_NNI_CONV_RELU_MODULE`暂未定义，用于量化相关的卷积操作和ReLU组合
    _NNIQAT_CONV_BN_MODULE = None  # type: ignore[assignment]
    _NNI_CONV_RELU_MODULE = None  # type: ignore[assignment]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        # 发出警告，提醒用户当前的量化动态卷积实现精度较差，不建议使用
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        # 断言：不支持反射填充模式
        assert padding_mode != 'reflect', "Conv3d does not support reflection padding"
        # 设定工厂参数，用于创建相应的张量
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 将核大小、步幅、填充和扩张参数转换为三元组形式
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        # 调用父类的初始化方法，完成卷积层的初始化
        super()._init(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        # 返回当前类的名称字符串
        return 'DynamicQuantizedConv3d'

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # 检查输入张量的形状是否为`(N, C, D, H, W)`
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        # 如果填充模式不是`zeros`，根据填充方式对输入进行填充
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 调用量化动态卷积操作，传入输入张量、打包参数、是否减少范围的标志
        return ops.quantized.conv3d_dynamic(
            input, self._packed_params, reduce_range)
class ConvTranspose1d(nnq.ConvTranspose1d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose1d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nndq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nndq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nndq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nndq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        # 用于记录设备和数据类型的工厂参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类初始化方法，传入所有必要参数
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding,
            groups, bias, dilation, padding_mode, **factory_kwargs)

    def _get_name(self):
        # 返回当前类的名称字符串
        return 'DynamicQuantizedConvTranspose1d'

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # 由于 JIT 问题，暂时使用 len(shape) 代替 ndim
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            # 如果输入张量的形状不是 (N, C, L)，则抛出值错误
            raise ValueError("Input shape must be `(N, C, L)`!")
        # 调用自定义的量化反卷积运算符进行前向传播
        return torch.ops.quantized.conv_transpose1d_dynamic(
            input, self._packed_params, reduce_range)


class ConvTranspose2d(nnq.ConvTranspose2d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv2d`
    """
    Attributes:
        weight (Tensor):     从可学习权重参数中生成的打包张量。
        scale (Tensor):      输出缩放的标量
        zero_point (Tensor): 输出零点的标量
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )
        # 准备工厂参数字典，用于构造父类实例
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类构造函数，初始化动态量化的转置卷积层
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding,
            groups, bias, dilation, padding_mode, **factory_kwargs)

    def _get_name(self):
        # 返回当前层的名称字符串
        return 'DynamicQuantizedConvTranspose2d'

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # 临时使用 len(shape) 替代 ndim，因为存在 JIT 问题
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            # 如果输入形状不是 (N, C, H, W)，则抛出异常
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        # 调用量化操作库中的动态转置卷积操作
        return ops.quantized.conv_transpose2d_dynamic(
            input, self._packed_params, reduce_range)
class ConvTranspose3d(nnq.ConvTranspose3d):
    r"""A dynamically quantized transposed convolution module with floating point tensors as inputs and outputs.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.dynamic.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose3d  # 将 nn.ConvTranspose3d 赋给 _FLOAT_MODULE，用于非量化操作时的模块

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        warnings.warn(
            f"The current implementation of the {self._get_name()} module has poor numerical accuracy and its use is not recommended"  # noqa: B950
        )  # 发出警告，提示当前模块实现数值精度较差，不建议使用

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding,
            groups, bias, dilation, padding_mode, **factory_kwargs)  # 调用父类的构造方法初始化模块

    def _get_name(self):
        return 'DynamicQuantizedConvTranspose3d'  # 返回当前模块的名称

    def forward(self, input: Tensor, reduce_range: bool = True) -> Tensor:
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:  # 检查输入张量的形状是否为 (N, C, T, H, W)，如果不是则抛出 ValueError
            raise ValueError("Input shape must be `(N, C, T, H, W)`!")
        
        return ops.quantized.conv_transpose3d_dynamic(  # 调用 quantized 操作进行动态量化转置卷积操作
            input, self._packed_params, reduce_range)
```