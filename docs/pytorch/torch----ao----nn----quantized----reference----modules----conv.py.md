# `.\pytorch\torch\ao\nn\quantized\reference\modules\conv.py`

```
# mypy: allow-untyped-defs
# 引入PyTorch库
import torch
# 引入神经网络模块
import torch.nn as nn
# 引入PyTorch中的函数操作模块
import torch.nn.functional as F
# 引入类型提示相关库
from typing import Optional, Dict, Any, List
# 引入与大小相关的通用类型定义
from torch.nn.common_types import _size_1_t
# 从当前目录下的utils文件中引入ReferenceQuantizedModule类
from .utils import ReferenceQuantizedModule

# 导出的类名列表
__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']

# 定义_ConvNd类，继承自torch.nn.modules.conv._ConvNd和ReferenceQuantizedModule类
class _ConvNd(torch.nn.modules.conv._ConvNd, ReferenceQuantizedModule):
    """ A reference version of nn.quantized.Conv2d
        we will not pack the parameters in this module, since weight packing is an
        optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
        this is useful when user want to use this module in other backends like Glow.
    """
    # 类型注解，bias属性为可选的torch.Tensor类型
    __annotations__ = {"bias": Optional[torch.Tensor]}
    # 类变量，表示这是一个参考实现
    _IS_REFERENCE = True

    # 静态方法，用于从浮点数模型float_conv转换得到量化参考模型
    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        # 创建一个_ConvNd类的实例qref_conv
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams)
        # 将浮点数模型的权重复制给量化参考模型
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        # 如果浮点数模型有偏置，则将其复制给量化参考模型
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        # 返回量化参考模型
        return qref_conv

# Conv1d类，继承自_ConvNd和nn.Conv1d
class Conv1d(_ConvNd, nn.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: _size_1_t = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):
        # 调用nn.Conv1d的初始化方法，初始化卷积层参数
        nn.Conv1d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        # 调用父类_ConvNd的初始化权重量化参数方法
        self._init_weight_qparams(weight_qparams, device)
    # 定义前向传播方法，接受输入张量 x，并返回处理后的张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv1d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """
        # 获取量化和反量化后的权重
        weight_quant_dequant = self.get_weight()
        # 执行一维卷积操作，使用量化后的权重和给定的偏置，以及其他参数
        result = F.conv1d(
            x, weight_quant_dequant, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        # 返回卷积操作的结果张量
        return result

    # 返回当前类的名称作为字符串
    def _get_name(self):
        return "QuantizedConv1d(Reference)"

    # 从浮点数卷积模型创建一个量化卷积模型的类方法
    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        return _ConvNd.from_float(cls, float_conv, weight_qparams)
class Conv2d(_ConvNd, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):
        # 调用父类 nn.Conv2d 的构造函数，初始化二维卷积层
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        # 使用给定的量化参数初始化权重量化参数
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv2d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
        # 获取量化后的权重和反量化函数
        weight_quant_dequant = self.get_weight()
        # 调用 F.conv2d 执行二维卷积操作
        result = F.conv2d(
            x, weight_quant_dequant, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        return result

    def _get_name(self):
        # 返回当前类的名称作为字符串
        return "QuantizedConv2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        # 调用 _ConvNd 类的 from_float 方法，将浮点数卷积转换为当前类的实例
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class Conv3d(_ConvNd, nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros",
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):
        # 调用父类 nn.Conv3d 的构造函数，初始化三维卷积层
        nn.Conv3d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        # 使用给定的量化参数初始化权重量化参数
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv3d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """
        # 获取量化后的权重和反量化函数
        weight_quant_dequant = self.get_weight()
        # 调用 F.conv3d 执行三维卷积操作
        result = F.conv3d(
            x, weight_quant_dequant, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        return result

    def _get_name(self):
        # 返回当前类的名称作为字符串
        return "QuantizedConv3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        # 调用 _ConvNd 类的 from_float 方法，将浮点数卷积转换为当前类的实例
        return _ConvNd.from_float(cls, float_conv, weight_qparams)
    """ A reference version of nn.quantized.ConvTranspose2d
        we will not pack the parameters in this module, since weight packing is an
        optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
        this is useful when user want to use this module in other backends like Glow.
    """
    @staticmethod
    # 静态方法：从一个浮点数版本的 Conv 对象构造出一个量化 ConvTranspose2d 对象的参考版本
    def from_float(cls, float_conv, weight_qparams):
        # 使用类的构造函数初始化量化 ConvTranspose2d 对象
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.output_padding,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams)
        # 将浮点数版本的权重参数转换为量化 ConvTranspose2d 对象的参数
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        # 如果浮点数版本有偏置项，则将其也转换为量化 ConvTranspose2d 对象的参数
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        # 返回构造好的量化 ConvTranspose2d 对象的参考版本
        return qref_conv
class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: _size_1_t = 0,
                 output_padding: _size_1_t = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: _size_1_t = 1,
                 padding_mode: str = "zeros",
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):
        # 调用父类 nn.ConvTranspose1d 的初始化方法，设置卷积转置1维的参数
        nn.ConvTranspose1d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
            groups, bias, dilation, padding_mode, device, dtype)
        # 初始化权重量化参数
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose1d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """

        # 断言 padding 是元组类型
        assert isinstance(self.padding, tuple)
        # 计算输出填充大小，在 TorchScript 中无法用 List 替换 Tuple 或 Sequence
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]

        # 获取权重并进行量化-去量化处理
        weight_quant_dequant = self.get_weight()
        # 执行卷积转置1维操作
        result = F.conv_transpose1d(
            x, weight_quant_dequant, self.bias, self.stride,
            self.padding, output_padding, self.groups, self.dilation)
        return result

    def _get_name(self):
        return "QuantizedConvTranspose1d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        # 从浮点数模型创建量化卷积转置1维对象
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)


class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0,
                 groups=1, bias=True, dilation=1,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):

        # 调用父类 nn.ConvTranspose2d 的初始化方法，设置卷积转置2维的参数
        nn.ConvTranspose2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
            groups, bias, dilation, padding_mode, device, dtype)
        # 初始化权重量化参数
        self._init_weight_qparams(weight_qparams, device)
    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        """
        前向传播函数，用于执行量化转置卷积操作。

        Args:
            x (torch.Tensor): 输入张量
            output_size (Optional[List[int]], optional): 输出尺寸，默认为None

        Returns:
            torch.Tensor: 经过量化转置卷积后的张量
        """
        assert isinstance(self.padding, tuple)
        # 在 "_output_padding" 中不能将 List 替换为 Tuple 或 Sequence，
        # 因为 TorchScript 不支持 `Sequence[T]` 或 `Tuple[T, ...]`。

        # 计算输出填充
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]

        # 获取权重的量化和反量化版本
        weight_quant_dequant = self.get_weight()

        # 执行转置卷积操作
        result = F.conv_transpose2d(
            x, weight_quant_dequant, self.bias, self.stride,
            self.padding, output_padding, self.groups, self.dilation)

        return result

    def _get_name(self):
        """
        返回模块的名称字符串。
        """
        return "QuantizedConvTranspose2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        """
        从浮点数模型创建量化转置卷积层。

        Args:
            float_conv: 浮点数卷积层
            weight_qparams: 权重量化参数

        Returns:
            QuantizedConvTranspose2d: 量化转置卷积层实例
        """
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)
class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0,
                 groups=1, bias=True, dilation=1,
                 padding_mode="zeros",
                 device=None,
                 dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None):
        # 调用父类 nn.ConvTranspose3d 的初始化方法，设置卷积转置层的参数
        nn.ConvTranspose3d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
            groups, bias, dilation, padding_mode, device, dtype)
        # 初始化权重量化参数
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose3d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """

        # 断言 padding 是一个元组
        assert isinstance(self.padding, tuple)
        
        # 计算输出填充 output_padding，根据输入、输出大小、步长、填充、卷积核大小和扩张
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]

        # 获取量化后的权重和去量化操作
        weight_quant_dequant = self.get_weight()
        
        # 执行卷积转置操作，使用量化后的权重，返回结果
        result = F.conv_transpose3d(
            x, weight_quant_dequant, self.bias, self.stride,
            self.padding, output_padding, self.groups, self.dilation)
        
        return result

    def _get_name(self):
        # 返回当前类的名称字符串，用于标识量化卷积转置层
        return "QuantizedConvTranspose3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        # 从浮点模型创建量化卷积转置层对象
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)
```