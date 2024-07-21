# `.\pytorch\torch\ao\nn\quantized\modules\conv.py`

```
# mypy: allow-untyped-defs
r"""Quantized convolution modules."""

# 引入必要的模块和库
from typing import Optional, List, TypeVar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat

# 引入OPS和常用类型定义
from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.utils import fuse_conv_bn_weights

# 引入自定义的工具函数和类
from .utils import _quantize_weight, WeightedQuantizedModule

# 导出的类名列表
__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']

# 支持的填充模式集合
_SUPPORTED_PADDING = {
    'zeros',
    'reflect'
}

# 辅助函数，用于将填充列表反向重复两次
def _reverse_repeat_padding(padding: List[int]) -> List[int]:
    _reversed_padding_repeated_twice: List[int] = []
    N = len(padding)
    for idx in range(N):
        for _ in range(2):
            _reversed_padding_repeated_twice.append(padding[N - idx - 1])
    return _reversed_padding_repeated_twice


# ConvNd类，继承自WeightedQuantizedModule类
class _ConvNd(WeightedQuantizedModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        # 所有子类都有相同的初始化方法签名 - 参见PR #49702s
        raise NotImplementedError
    # 初始化函数，设置卷积层的各项参数和属性
    def _init(self, in_channels, out_channels, kernel_size, stride,
              padding, dilation,
              transposed, output_padding,
              groups, bias,
              padding_mode='zeros',
              device=None,
              dtype=None) -> None:
        # 创建一个包含设备和数据类型的关键字参数字典
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的初始化方法
        super().__init__()

        # 检查输入通道数是否可以被分组数整除，若不能则引发异常
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        # 检查输出通道数是否可以被分组数整除，若不能则引发异常
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        # 设置对象的属性值
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        # 检查填充模式是否为支持的模式之一，若不是则引发异常
        if padding_mode not in _SUPPORTED_PADDING:
            raise ValueError(f"'padding_mode' {padding_mode} is not supported by quantized convolution")
        self.padding_mode = padding_mode

        # 如果是转置卷积，设置权重形状为[in_channels, out_channels // groups]
        # 否则设置为[out_channels, in_channels // groups]
        if self.transposed:
            weight_shape = [in_channels, out_channels // self.groups]
        else:
            weight_shape = [out_channels, in_channels // self.groups]
        
        # 创建一个空的量化权重张量，使用给定的工厂关键字参数
        qweight = torch._empty_affine_quantized(
            weight_shape + list(kernel_size),
            scale=1, zero_point=0, dtype=torch.qint8,
            **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})
        
        # 如果存在偏置，则创建一个浮点类型的偏置张量；否则偏置为None
        bias_float = (
            torch.zeros(out_channels, dtype=torch.float,
                        **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}) if bias else None)
        
        # 调用设置权重和偏置的方法
        self.set_weight_bias(qweight, bias_float)
        # 设置比例尺度为1.0
        self.scale = 1.0
        # 设置零点为0
        self.zero_point = 0

    # 设置权重和偏置的方法，需要在子类中实现
    def set_weight_bias(self, qweight, bias_float):
        raise NotImplementedError

    # 获取偏置的方法，需要在子类中实现
    def bias(self):
        raise NotImplementedError

    # 获取权重和偏置的方法，需要在子类中实现
    def _weight_bias(self):
        raise NotImplementedError

    # 返回卷积层的额外描述信息，包括各种参数和属性
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, scale={scale}, zero_point={zero_point}')
        # 如果存在填充，将填充信息添加到描述字符串中
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        # 如果存在膨胀，将膨胀信息添加到描述字符串中
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        # 如果存在输出填充，将输出填充信息添加到描述字符串中
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        # 如果分组数不为1，将分组数添加到描述字符串中
        if self.groups != 1:
            s += ', groups={groups}'
        # 如果没有偏置，将偏置信息添加到描述字符串中
        if self.bias() is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    # ===== 序列化方法 =====
    # 这里的特殊考虑是，我们必须将权重解压成常规的QTensor形式进行序列化。压缩的权重不应该
    # （此处内容未完整给出，请自行填写）
    # 在状态字典中保存当前对象的状态，继承自父类的方法
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # 调用内部方法获取权重和偏置
        (w, b) = self._weight_bias()
        # 将权重、偏置、缩放因子和零点偏移保存到状态字典中
        destination[prefix + 'weight'] = w
        destination[prefix + 'bias'] = b
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    @torch.jit.export
    # 获取对象的状态，用于序列化
    def __getstate__(self):
        # 调用内部方法获取权重和偏置
        (w, b) = self._weight_bias()
        # 返回对象的状态元组，包括各种属性和权重、偏置、缩放因子、零点偏移等
        return (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.transposed,
            self.output_padding,
            self.groups,
            self.padding_mode,
            w,
            b,
            self.scale,
            self.zero_point,
            self.training
        )

    # ===== 反序列化方法 =====
    # 反序列化方法的对应方法，将序列化的QTensor权重打包成FBGEMM操作所需的打包格式
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 设置权重和偏置
        self.set_weight_bias(
            state_dict[prefix + 'weight'], state_dict[prefix + 'bias'])
        # 移除已经处理过的权重和偏置键
        state_dict.pop(prefix + 'weight')
        state_dict.pop(prefix + 'bias')
        # 设置缩放因子和零点偏移
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')
        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')
        # 调用父类的加载方法，不检查额外的键
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys,
            unexpected_keys, error_msgs)

    @torch.jit.export
    # 设置对象的状态，用于反序列化
    def __setstate__(self, state):
        # 恢复对象的各种属性
        self.in_channels = state[0]
        self.out_channels = state[1]
        self.kernel_size = state[2]
        self.stride = state[3]
        self.padding = state[4]
        self.dilation = state[5]
        self.transposed = state[6]
        self.output_padding = state[7]
        self.groups = state[8]
        self.padding_mode = state[9]
        # 设置权重和偏置
        self.set_weight_bias(state[10], state[11])
        # 设置缩放因子和零点偏移
        self.scale = state[12]
        self.zero_point = state[13]
        # 设置训练标志
        self.training = state[14]

    # 深拷贝方法，用于创建当前对象的副本
    def __deepcopy__(self, memo):
        # 创建当前对象的新实例
        new_instance = type(self).__new__(type(self))
        # 初始化新实例
        torch.nn.Module.__init__(new_instance)
        # 获取当前对象的状态
        state = self.__getstate__()
        # 设置新实例的状态
        new_instance.__setstate__(state)
        # 返回新实例
        return new_instance
    def __copy__(self):
        # 使用 __deepcopy__ 方法创建一个新对象，不传入任何参数
        return self.__deepcopy__({})

    @classmethod
    def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
        r"""Creates a qconv object and returns it.
        """
        # 如果 weight_post_process 未定义，则使用 mod.qconfig.weight() 来初始化它
        if weight_post_process is None:
            weight_post_process = mod.qconfig.weight()
        # 对权重进行量化观察
        weight_post_process(mod.weight)
        assert weight_post_process.dtype == torch.qint8, \
            'Weight observer must have a dtype of qint8'
        # 将权重转换为量化后的权重 qweight
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # 创建一个 qconv 对象，调用的是派生类中的 __init__ 方法而不是 _ConvNd 中的方法
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    mod.stride, mod.padding, mod.dilation, mod.groups,
                    mod.bias is not None, mod.padding_mode)
        # 设置权重和偏置
        qconv.set_weight_bias(qweight, mod.bias)
        # 如果 activation_post_process 为 None 或者其 dtype 为 torch.float，则返回 qconv
        if activation_post_process is None or activation_post_process.dtype == torch.float:
            return qconv  # 动态量化不需要 scale/zero_point
        else:
            # 计算激活量化参数的 scale 和 zero_point
            act_scale, act_zp = activation_post_process.calculate_qparams()
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 如果模型具有 "weight_fake_quant" 属性
        if hasattr(mod, "weight_fake_quant"):
            # 如果 mod 的类型为 cls._NNIQAT_CONV_BN_MODULE
            if type(mod) == cls._NNIQAT_CONV_BN_MODULE:
                # 合并卷积和批量归一化的权重和偏置
                mod.weight, mod.bias = fuse_conv_bn_weights(
                    mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                    mod.bn.eps, mod.bn.weight, mod.bn.bias)
            # 确保输入 QAT 模块已经附加了观察器
            assert hasattr(mod, "activation_post_process"), \
                "Input QAT module must have observer attached"
            # 使用 mod.weight_fake_quant 作为权重观察器
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            # 确保 mod 的类型为 cls._FLOAT_MODULE
            assert type(mod) == cls._FLOAT_MODULE, \
                " nnq." + cls.__name__ + ".from_float only works for " + \
                cls._FLOAT_MODULE.__name__ + " but got:" + str(type(mod))
            # 确保输入的浮点模块已经定义了 qconfig
            assert hasattr(mod, "qconfig"), \
                "Input float module must have qconfig defined."
            # 如果 mod 没有激活后处理过程，则 activation_post_process 为 None
            activation_post_process = None if not hasattr(
                mod, "activation_post_process") else mod.activation_post_process
            # 如果 mod 的类型为 [cls._NNI_CONV_RELU_MODULE, cls._NNI_CONV_ADD_MODULE, cls._NNI_CONV_ADD_RELU_MODULE] 中的一种，则使用其第一个元素
            if type(mod) in [cls._NNI_CONV_RELU_MODULE, cls._NNI_CONV_ADD_MODULE, cls._NNI_CONV_ADD_RELU_MODULE]:
                mod = mod[0]
            # 使用 mod.qconfig.weight() 作为权重观察器
            weight_post_process = mod.qconfig.weight()
        # 调用 get_qconv 方法来获取量化后的卷积对象
        return cls.get_qconv(mod, activation_post_process, weight_post_process)

    @classmethod
    # 从参考量化模块创建一个量化模块，使用(fbgemm/qnnpack)量化引擎
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        # 创建一个新的量化卷积模块，从参考量化模块(ref_qconv)中继承参数
        qconv = cls(
            ref_qconv.in_channels,            # 输入通道数
            ref_qconv.out_channels,           # 输出通道数
            ref_qconv.kernel_size,            # 卷积核大小
            ref_qconv.stride,                 # 卷积步长
            ref_qconv.padding,                # 卷积填充
            ref_qconv.dilation,               # 卷积膨胀率
            ref_qconv.groups,                 # 卷积分组数
            ref_qconv.bias is not None,       # 是否有偏置
            ref_qconv.padding_mode,           # 填充模式
            device=ref_qconv.weight.device,   # 设备类型
            dtype=ref_qconv.weight.dtype)     # 数据类型
        # 获取参考量化模块的量化权重
        qweight = ref_qconv.get_quantized_weight()
        # 设置量化卷积模块的权重和偏置
        qconv.set_weight_bias(qweight, ref_qconv.bias)
        # 设置量化卷积模块的输出缩放因子
        qconv.scale = float(output_scale)
        # 设置量化卷积模块的输出零点
        qconv.zero_point = int(output_zero_point)
        # 返回创建的量化卷积模块
        return qconv
class Conv1d(_ConvNd):
    r"""Applies a 1D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
        ...                                     dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE = nn.Conv1d
    _NNIQAT_CONV_BN_MODULE = nniqat.ConvBn1d
    _NNI_CONV_RELU_MODULE = nni.ConvReLU1d
    _NNI_CONV_ADD_MODULE: None = None
    _NNI_CONV_ADD_RELU_MODULE: None = None

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
                 dtype=None):
        # 定义一个字典，包含设备和数据类型的参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 将 kernel_size, stride, padding, dilation 转换为一维元组
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        # 如果 padding 是字符串，则保持原样，否则转换为一维元组
        padding = padding if isinstance(padding, str) else _single(padding)
        dilation = _single(dilation)

        # 对于 _ConvNd 的子类，需要调用 _init 而不是 __init__。参见 PR #49702 上的讨论
        super()._init(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        # 返回该类的名称字符串
        return 'QuantizedConv1d'

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        if self.padding_mode == 'zeros':
            # 使用 quantized.conv1d_prepack 函数预打包权重和偏置参数
            self._packed_params = torch.ops.quantized.conv1d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups)
        else:
            # 使用 quantized.conv1d_prepack 函数预打包权重和偏置参数（使用默认零填充）
            self._packed_params = torch.ops.quantized.conv1d_prepack(
                w, b, self.stride, _pair(0), self.dilation,
                self.groups)

    def _weight_bias(self):
        # 使用 quantized.conv1d_unpack 函数解包打包的参数，并返回权重 w 和偏置 b
        w, b = torch.ops.quantized.conv1d_unpack(self._packed_params)
        return w, b
    # 返回权重参数元组的第一个元素，即卷积层的权重
    def weight(self):
        return self._weight_bias()[0]

    # 返回偏置参数元组的第二个元素，即卷积层的偏置
    def bias(self):
        return self._weight_bias()[1]

    # 对输入进行前向传播计算
    def forward(self, input):
        # 临时使用 len(shape) 替代 ndim，因为存在 JIT 问题
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != 'zeros':
            # Conv1d 中的 padding 存储形式为 (p, p)，需转换为 (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 执行量化卷积操作，返回结果
        return ops.quantized.conv1d(input, self._packed_params, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""从浮点数模型或 qparams_dict 创建一个量化模块。

        Args:
            mod (Module): 一个浮点数模型，可以是由 torch.ao.quantization 工具产生的或用户提供的
            use_precomputed_fake_quant (bool, optional): 是否使用预先计算的伪量化参数。默认为 False
        """
        return _ConvNd.from_float(cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE = nn.Conv2d
    _NNIQAT_CONV_BN_MODULE = nniqat.ConvBn2d
    _NNI_CONV_RELU_MODULE = nni.ConvReLU2d
    _NNI_CONV_ADD_MODULE = nni.ConvAdd2d
    _NNI_CONV_ADD_RELU_MODULE = nni.ConvAddReLU2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)  # 将 kernel_size 转换为二元组
        stride = _pair(stride)            # 将 stride 转换为二元组
        padding = _pair(padding)          # 将 padding 转换为二元组
        dilation = _pair(dilation)        # 将 dilation 转换为二元组
        # 子类 _ConvNd 的构造函数必须调用 _init 而不是 __init__。参见 PR #49702 上的讨论
        super()._init(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedConv2d'  # 返回当前类的名称字符串

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        if self.padding_mode == 'zeros':
            # 使用 quantized 操作预打包权重和偏置，基于 'zeros' 填充模式
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups)
        else:
            # 使用 quantized 操作预打包权重和偏置，当填充模式不是 'zeros' 时使用零填充
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                w, b, self.stride, _pair(0), self.dilation, self.groups)

    def _weight_bias(self):
        return self._packed_params.unpack()  # 返回解包后的权重和偏置信息
    # 返回权重的值，调用 _weight_bias 方法并获取其返回值的第一个元素
    def weight(self):
        return self._weight_bias()[0]

    # 返回偏置的值，调用 _weight_bias 方法并获取其返回值的第二个元素
    def bias(self):
        return self._weight_bias()[1]

    # 对输入数据进行前向传播计算
    def forward(self, input):
        # 临时使用 len(shape) 替代 ndim，因为存在 JIT 问题
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        # 如果填充模式不是 'zeros'，根据当前填充参数和模式进行填充操作
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        # 调用 ops.quantized.conv2d 进行量化卷积操作，使用封装好的参数进行计算
        return ops.quantized.conv2d(
            input, self._packed_params, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        # 调用 _ConvNd 的 from_float 方法，从浮点数模块或者 qparams_dict 创建一个量化模块
        return _ConvNd.from_float(cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
class Conv3d(_ConvNd):
    r"""Applies a 3D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """
    _FLOAT_MODULE = nn.Conv3d
    _NNIQAT_CONV_BN_MODULE = nniqat.ConvBn3d
    _NNI_CONV_RELU_MODULE = nni.ConvReLU3d
    _NNI_CONV_ADD_MODULE: None = None
    _NNI_CONV_ADD_RELU_MODULE: None = None

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        assert padding_mode != 'reflect', "Conv3d does not support reflection padding"
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Convert kernel_size, stride, padding, dilation to tuples if they are not
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        # Call _init method of super class _ConvNd to initialize the convolutional layer
        # with quantized parameters
        super()._init(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedConv3d'

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        # Prepacks the weight and bias tensors for quantized convolution operation
        if self.padding_mode == 'zeros':
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups)
        else:
            # If padding_mode is not 'zeros', use zero padding for prepacking
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                w, b, self.stride, _triple(0), self.dilation, self.groups)
    # 返回打包的参数 (weight, bias) 元组
    def _weight_bias(self):
        return self._packed_params.unpack()

    # 返回量化卷积层的权重
    def weight(self):
        return self._weight_bias()[0]

    # 返回量化卷积层的偏置
    def bias(self):
        return self._weight_bias()[1]

    # 对输入进行前向传播计算
    def forward(self, input):
        # 由于即时编译问题，暂时使用 len(shape) 替代 ndim
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        
        # 如果填充模式不是 'zeros'，则调整输入的填充方式
        if self.padding_mode != 'zeros':
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(input, _reversed_padding_repeated_twice,
                          mode=self.padding_mode)
        
        # 执行量化卷积操作，使用打包的参数、量化比例和零点
        return ops.quantized.conv3d(
            input, self._packed_params, self.scale, self.zero_point)

    @classmethod
    # 从浮点数模型中创建量化模块或 qparams_dict
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
# === Transposed Convolutions ===

# 定义一个类型变量 MOD，限定为 nn.modules.conv._ConvNd 或其子类
MOD = TypeVar('MOD', bound=nn.modules.conv._ConvNd)

# 定义 _ConvTransposeNd 类，继承自 _ConvNd
class _ConvTransposeNd(_ConvNd):

    # 类属性，指定用于处理浮点数的模块类型
    _FLOAT_MODULE = MOD

    # 初始化方法
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, device=None, dtype=None):
        # 如果填充模式不是 'zeros'，则抛出 ValueError 异常
        if padding_mode != 'zeros':
            raise ValueError(f'Only "zeros" padding mode is supported for {self.__class__.__name__}')
        
        # 创建工厂参数字典
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 调用父类的 _init 方法进行初始化，而不是直接调用 __init__ 方法
        # 参见 PR #49702 上的讨论
        super()._init(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode, **factory_kwargs)

    # 定义一个静态方法 _input_padding，计算输入填充
    def _input_padding(self, kernel_size: List[int], dilation: List[int], padding: List[int]) -> List[int]:
        res = torch.jit.annotate(List[int], [])
        for kdx in range(len(kernel_size)):
            pad = (dilation[kdx] * (kernel_size[kdx] - 1) - padding[kdx])
            res.append(pad)
        return res

    # 类方法 from_float，从浮点数模块创建一个量化模块
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Creates a quantized module from a float module or qparams_dict.
        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        # 派生类需要重写 cls._FLOAT_MODULE 属性
        msg = ' nnq.' + cls.__name__ + '.from_float only works for ' + \
              cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        
        # 确保 mod 是 cls._FLOAT_MODULE 类型的对象
        assert type(mod) == cls._FLOAT_MODULE, msg
        assert hasattr(mod, 'qconfig'), \
            'Input float module must have qconfig defined.'
        
        # 获取权重后处理器，并对权重进行处理
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        
        # 确保权重后处理器的 dtype 是 torch.qint8
        assert weight_post_process.dtype == torch.qint8, \
            'Weight observer must have a dtype of qint8'
        
        # 对权重进行量化
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        
        # 使用派生类的 __init__ 方法创建 qconv 对象，而不是 _ConvTransposeNd 的方法
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    mod.stride, mod.padding, mod.output_padding, mod.groups,
                    mod.bias is not None, mod.dilation, mod.padding_mode)
        
        # 设置权重和偏置
        qconv.set_weight_bias(qweight, mod.bias)
        
        # 如果模块没有 activation_post_process 或者其 dtype 是 torch.float，则返回 qconv
        if not hasattr(mod, "activation_post_process") or mod.activation_post_process.dtype == torch.float:
            return qconv  # 动态量化不需要 scale/zero_point
        else:
            # 计算激活量化参数
            act_scale, act_zp = mod.activation_post_process.calculate_qparams()
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    # 静态方法
    @staticmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconvt (Module): a reference quantized module, either produced by torch.ao.quantization
                                 utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        # 使用给定的类方法创建一个量化模块，基于参考量化模块的属性和参数
        qconv = cls(
            ref_qconvt.in_channels,
            ref_qconvt.out_channels,
            ref_qconvt.kernel_size,  # type: ignore[arg-type]
            ref_qconvt.stride,  # type: ignore[arg-type]
            ref_qconvt.padding,  # type: ignore[arg-type]
            ref_qconvt.output_padding,  # type: ignore[arg-type]
            ref_qconvt.groups,
            ref_qconvt.bias is not None,  # type: ignore[arg-type]
            ref_qconvt.dilation,  # type: ignore[arg-type]
            ref_qconvt.padding_mode,
            device=ref_qconvt.weight.device,
            dtype=ref_qconvt.weight.dtype)
        # 获取参考量化模块的量化权重
        qweight = ref_qconvt.get_quantized_weight()
        # 设置量化卷积模块的权重和偏置
        qconv.set_weight_bias(qweight, ref_qconvt.bias)
        # 设置量化卷积模块的输出比例因子
        qconv.scale = float(output_scale)
        # 设置量化卷积模块的输出零点
        qconv.zero_point = int(output_zero_point)
        # 返回创建的量化卷积模块
        return qconv
class ConvTranspose1d(_ConvTransposeNd):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    .. note:: Currently only the QNNPACK engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'qnnpack'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose1d  # 指定非量化版本的模块为 nn.ConvTranspose1d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _single(kernel_size)  # 转换 kernel_size 成一维元组
        stride = _single(stride)  # 转换 stride 成一维元组
        padding = _single(padding)  # 转换 padding 成一维元组
        dilation = _single(dilation)  # 转换 dilation 成一维元组
        output_padding = _single(output_padding)  # 转换 output_padding 成一维元组

        # 调用父类构造函数初始化对象
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedConvTranspose1d'  # 返回该类的名称
    # 设置量化卷积转置1维层的权重和偏置参数
    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        # 使用 Torch 的量化操作函数进行权重和偏置参数的打包
        self._packed_params = torch.ops.quantized.conv_transpose1d_prepack(
            w, b, self.stride, self.padding, self.output_padding, self.dilation,
            self.groups)

    # 获取量化卷积转置1维层的权重和偏置参数
    def _weight_bias(self):
        # 使用 Torch 的量化操作函数进行权重和偏置参数的解包
        w, b = torch.ops.quantized.conv_transpose1d_unpack(self._packed_params)
        return w, b

    # 返回量化卷积转置1维层的权重
    def weight(self):
        # 获取权重和偏置参数中的权重部分
        (w, _) = self._weight_bias()
        return w

    # 返回量化卷积转置1维层的偏置
    def bias(self):
        # 获取权重和偏置参数中的偏置部分
        (_, b) = self._weight_bias()
        return b

    # 定义量化卷积转置1维层的前向传播过程
    def forward(self, input):
        # 检查输入张量的形状是否为 (N, C, L)，由于 JIT 存在问题，使用 len(shape) 代替 ndim
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        # 执行量化卷积转置1维的前向传播计算
        return torch.ops.quantized.conv_transpose1d(
            input, self._packed_params, self.scale, self.zero_point)

    # 从参考量化卷积转置Nd层创建新的实例
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):
        return _ConvTransposeNd.from_reference(cls, ref_qconvt, output_scale, output_zero_point)
class ConvTranspose2d(_ConvTransposeNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # QNNPACK or FBGEMM as backend
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> # With square kernels and equal stride
        >>> import torch.ao.nn.quantized as nnq
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose2d  # 设置_Float_Module为nn.ConvTranspose2d，用于指定浮点数模块

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)  # 将kernel_size转换为tuple
        stride = _pair(stride)  # 将stride转换为tuple
        padding = _pair(padding)  # 将padding转换为tuple
        dilation = _pair(dilation)  # 将dilation转换为tuple
        output_padding = _pair(output_padding)  # 将output_padding转换为tuple

        super().__init__(  # 调用父类的构造方法来初始化对象
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedConvTranspose2d'  # 返回该类的名称作为字符串

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._packed_params = torch.ops.quantized.conv_transpose2d_prepack(
            w, b, self.stride, self.padding, self.output_padding, self.dilation,
            self.groups)  # 使用torch.ops.quantized.conv_transpose2d_prepack方法设置权重和偏置参数
    # 解压缩量化卷积操作的参数，获取权重和偏置
    def _weight_bias(self):
        w, b = torch.ops.quantized.conv2d_unpack(self._packed_params)
        return w, b

    # 返回卷积层的权重
    def weight(self):
        (w, _) = self._weight_bias()
        return w

    # 返回卷积层的偏置
    def bias(self):
        (_, b) = self._weight_bias()
        return b

    # 执行量化卷积转置操作
    def forward(self, input):
        # 由于 JIT 存在问题，暂时使用 len(input.shape) 替代 input.ndim
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        # 调用量化卷积转置操作
        return ops.quantized.conv_transpose2d(
            input, self._packed_params, self.scale, self.zero_point)

    @classmethod
    # 从参考量化卷积转置操作中构造新的实例
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):
        return _ConvTransposeNd.from_reference(cls, ref_qconvt, output_scale, output_zero_point)
class ConvTranspose3d(_ConvTransposeNd):
    r"""Applies a 3D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    .. note:: Currently only the FBGEMM engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'fbgemm'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = 'fbgemm'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> input = torch.randn(20, 16, 50, 100, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """

    _FLOAT_MODULE = nn.ConvTranspose3d

    # 初始化方法，定义了类的属性和初始参数
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 将 kernel_size、stride、padding 和 dilation 转化为三元组
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)

        # 调用父类 _ConvTransposeNd 的初始化方法
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)

    # 返回类的名称作为字符串
    def _get_name(self):
        return 'QuantizedConvTranspose3d'
    # 设置量化卷积转置层的权重和偏置参数，并使用预打包函数进行封装
    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._packed_params = torch.ops.quantized.conv_transpose3d_prepack(
            w, b, self.stride, self.padding, self.output_padding, self.dilation,
            self.groups)

    # 解压缩并返回量化卷积层转置的权重和偏置参数
    def _weight_bias(self):
        w, b = torch.ops.quantized.conv3d_unpack(self._packed_params)
        return w, b

    # 返回量化卷积转置层的权重
    def weight(self):
        (w, _) = self._weight_bias()
        return w

    # 返回量化卷积转置层的偏置
    def bias(self):
        (_, b) = self._weight_bias()
        return b

    # 前向传播函数，执行量化卷积转置操作
    def forward(self, input):
        # 临时使用 len(shape) 替代 ndim，因为 JIT 存在问题
        # 参考：https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, T, H, W)`!")
        # 调用量化卷积转置函数进行前向传播
        return ops.quantized.conv_transpose3d(
            input, self._packed_params, self.scale, self.zero_point)

    # 类方法，从参考的量化卷积转置层创建新的实例
    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):
        return _ConvTransposeNd.from_reference(cls, ref_qconvt, output_scale, output_zero_point)
```