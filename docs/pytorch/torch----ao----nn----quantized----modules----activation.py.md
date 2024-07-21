# `.\pytorch\torch\ao\nn\quantized\modules\activation.py`

```
# mypy: allow-untyped-defs
# 导入torch模块
import torch
# 导入warn函数从warnings模块
from warnings import warn
# 定义导出的类列表
__all__ = [
    "ReLU6",
    "Hardswish",
    "ELU",
    "LeakyReLU",
    "Sigmoid",
    "Softmax",
    "MultiheadAttention",
    "PReLU"
]

# 定义ReLU6类，继承自torch.nn.ReLU
class ReLU6(torch.nn.ReLU):
    r"""Applies the element-wise function:

    :math:`\text{ReLU6}(x) = \min(\max(x_0, x), q(6))`, where :math:`x_0` is the
    zero_point, and :math:`q(6)` is the quantized representation of number 6.

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.quantized.ReLU6()
        >>> input = torch.randn(2)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    
    # 定义初始化方法
    def __init__(self, inplace=False):
        # 调用父类的初始化方法
        super().__init__(inplace)
        # 设置是否原地操作的标志位
        self.inplace = inplace

    # 前向传播方法，接受input作为参数
    def forward(self, input):
        # 调用torch.ops.quantized.relu6函数进行ReLU6激活函数的计算
        return torch.ops.quantized.relu6(input, self.inplace)

    # 获取类名的私有方法
    def _get_name(self):
        return 'QuantizedReLU6'

    # 从浮点模型转换为量化模型的静态方法
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return ReLU6(mod.inplace)

# 定义Hardswish类，继承自torch.nn.Hardswish
class Hardswish(torch.nn.Hardswish):
    r"""This is the quantized version of :class:`~torch.nn.Hardswish`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    
    # 初始化方法，接受scale, zero_point, device=None, dtype=None作为参数
    def __init__(self, scale, zero_point, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的初始化方法
        super().__init__()
        # 使用torch.tensor创建并注册缓冲区scale和zero_point
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    # 前向传播方法，接受input作为参数
    def forward(self, input):
        # 调用torch.ops.quantized.hardswish函数进行Hardswish激活函数的计算
        return torch.ops.quantized.hardswish(input, self.scale, self.zero_point)

    # 获取类名的私有方法
    def _get_name(self):
        return 'QuantizedHardswish'

    # 从浮点模型转换为量化模型的静态方法
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # 计算scale和zero_point
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 返回Hardswish类的实例
        return Hardswish(float(scale), int(zero_point))

    # 根据参考模型和提供的scale和zero_point创建实例的类方法
    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point))

# 定义ELU类，继承自torch.nn.ELU
class ELU(torch.nn.ELU):
    r"""This is the quantized equivalent of :class:`~torch.nn.ELU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """
    
    # 初始化方法，接受scale, zero_point, alpha=1.作为参数
    def __init__(self, scale, zero_point, alpha=1.):
        # 调用父类的初始化方法
        super().__init__(alpha)
        # 设置scale和zero_point属性
        self.scale = scale
        self.zero_point = zero_point

    # 前向传播方法，接受input作为参数
    def forward(self, input):
        # 调用torch.ao.nn.quantized.functional.elu函数进行ELU激活函数的计算
        return torch.ao.nn.quantized.functional.elu(
            input, self.scale, self.zero_point, self.alpha)
    # 返回固定字符串 'QuantizedELU'，用作对象的名称
    def _get_name(self):
        return 'QuantizedELU'

    # 从浮点数模型创建 QuantizedELU 对象的静态方法
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # 获取激活后处理的量化参数：缩放因子和零点
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 使用获取的参数创建并返回一个 ELU 对象
        return ELU(float(scale), int(zero_point), mod.alpha)

    # 从参考模型创建 QuantizedELU 对象的类方法
    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # 使用给定的缩放因子、零点和 alpha 值创建并返回一个 ELU 对象
        return cls(float(scale), int(zero_point), mod.alpha)
class LeakyReLU(torch.nn.LeakyReLU):
    r"""This is the quantized equivalent of :class:`~torch.nn.LeakyReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
    """
    
    def __init__(self, scale: float, zero_point: int, negative_slope: float = 1e-2,
                 inplace: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类构造函数初始化 LeakyReLU 激活函数的负斜率
        super().__init__(negative_slope, inplace)
        # 创建并注册缓冲区，存储量化的 scale 和 zero_point
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # 调用量化的 LeakyReLU 操作
        return torch.ops.quantized.leaky_relu(
            input, self.negative_slope, self.inplace, self.scale, self.zero_point)

    def _get_name(self):
        # 返回当前类的名称
        return 'QuantizedLeakyReLU'

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 从浮点模型转换得到 scale 和 zero_point，创建一个新的 QuantizedLeakyReLU 实例
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # 根据给定的 scale 和 zero_point 创建一个新的 QuantizedLeakyReLU 实例
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)


class Sigmoid(torch.nn.Sigmoid):
    r"""This is the quantized equivalent of :class:`~torch.nn.Sigmoid`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, output_scale: float, output_zero_point: int):
        # 调用父类构造函数初始化 Sigmoid 激活函数
        super().__init__()
        # 存储量化的 scale 和 zero_point
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point

    def forward(self, input):
        # 调用量化的 Sigmoid 操作
        return torch.ops.quantized.sigmoid(input, self.output_scale, self.output_zero_point)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 从浮点模型转换得到 scale 和 zero_point，创建一个新的 QuantizedSigmoid 实例
        output_scale, output_zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(output_scale), int(output_zero_point))


class Softmax(torch.nn.Softmax):
    r"""This is the quantized version of :class:`~torch.nn.Softmax`.

    Args:
        dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, dim=None, scale=1.0, zero_point=0):
        # 调用父类构造函数初始化 Softmax
        super().__init__()
        # 存储量化的 dim, scale 和 zero_point
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point
    def forward(self, input):
        dim = self.dim  # 将对象属性self.dim赋值给局部变量dim
        if dim is None:  # 如果dim为None，则执行以下代码块
            stacklevel = 3  # 设置stacklevel为3，用于日志栈追溯
            # 注意：在_get_softmax_dim上添加mypy忽略，看起来比将_get_softmax_dim作为官方API更好。
            # 调用torch.nn.functional._get_softmax_dim函数确定softmax操作的维度
            dim = torch.nn.functional._get_softmax_dim(  # type: ignore[attr-defined]
                "softmax", input.dim(), stacklevel)
        # 调用torch.ops.quantized.softmax函数执行量化softmax操作，使用指定的维度、缩放因子和零点偏移
        return torch.ops.quantized.softmax(
            input, dim, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedSoftmax'  # 返回字符串'QuantizedSoftmax'

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # 计算模型的量化参数scale和zero_point
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 根据浮点数模型创建Softmax对象，指定维度、缩放因子和零点偏移
        return Softmax(mod.dim, float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # 根据参考模型创建Softmax对象，指定维度、缩放因子和零点偏移
        return cls(mod.dim, float(scale), int(zero_point))
class MultiheadAttention(torch.ao.nn.quantizable.MultiheadAttention):
    _FLOAT_MODULE = torch.ao.nn.quantizable.MultiheadAttention

    # 返回当前类的名称字符串
    def _get_name(self):
        return "QuantizedMultiheadAttention"

    @classmethod
    def from_float(cls, other):
        # 整体流程为 float -> observed -> quantized
        # 本方法仅实现 observed -> quantized 部分
        raise NotImplementedError("It looks like you are trying to convert a "
                                  "non-observed MHA module. Please, see "
                                  "the examples on quantizable MHAs.")

    @classmethod
    def from_observed(cls, other):
        # 将给定的其他模块转换为量化形式
        converted = torch.ao.quantization.convert(other, mapping=None,
                                                  inplace=False,
                                                  remove_qconfig=True,
                                                  convert_custom_config_dict=None)
        converted.__class__ = cls
        
        # 移除 bias_k 和 bias_v 的参数，以便量化它们
        # TODO: 这可能会导致精度损失。量化 cat 操作使用第一个元素的尺度和零点，
        #       这可能会损失 bias_k 和 bias_v 的精度（它们与 k/v 首先 cat 在一起）。
        if converted.bias_k is not None:
            bias_k = converted._parameters.pop('bias_k')
            sc, zp = torch._choose_qparams_per_tensor(bias_k,
                                                      reduce_range=False)
            bias_k = torch.quantize_per_tensor(bias_k, sc, zp, torch.quint8)
            setattr(converted, 'bias_k', bias_k)  # noqa: B010

        if converted.bias_v is not None:
            bias_v = converted._parameters.pop('bias_v')
            sc, zp = torch._choose_qparams_per_tensor(bias_k,  # type: ignore[possibly-undefined]
                                                      reduce_range=False)
            bias_v = torch.quantize_per_tensor(bias_v, sc, zp, torch.quint8)
            setattr(converted, 'bias_v', bias_v)  # noqa: B010

        # 删除不再需要的参数和权重
        del converted.in_proj_weight
        del converted.in_proj_bias

        # 返回转换后的量化模块
        return converted


class PReLU(torch.nn.Module):
    r"""This is the quantized equivalent of :class:`~torch.nn.PReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        num_parameters: number of parameters: 1, or the number of channels at input. Default: 1
    """
    
    # 初始化 PReLU 类的实例
    def __init__(self, output_scale: float, output_zero_point: int,
                 num_parameters: int = 1) -> None:
        super().__init__()
        self.num_parameters = num_parameters  # 记录参数数量
        self.scale = output_scale  # 记录输出张量的量化尺度
        self.zero_point = output_zero_point  # 记录输出张量的量化零点
        w = torch.randn(num_parameters, dtype=torch.float)  # 生成随机权重 w
        qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.quint8)  # 将权重 w 进行量化
        self.set_weight(qw)  # 设置量化后的权重
    # 设置权重的方法，将传入的张量 w 赋值给对象的 weight 属性
    def set_weight(self, w: torch.Tensor) -> None:
        self.weight = w

    # 前向传播方法，调用 quantized.prelu 操作进行量化 PReLU 激活函数的前向传播
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.prelu(input, self.weight, self.scale, self.zero_point)

    # 返回当前类的名称 '_get_name'，用于识别量化 PReLU 类
    def _get_name(self):
        return 'QuantizedPReLU'

    # 类方法，从浮点数模型转换得到量化 PReLU 模型
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 计算量化参数 scale 和 zero_point
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        # 创建一个量化 PReLU 对象
        qprelu = cls(float(scale), int(zero_point), mod.num_parameters)
        # 将浮点权重转换为 float 类型
        float_wt = mod.weight.float()
        # 创建权重的量化观察器
        observer = mod.qconfig.weight()
        observer(float_wt)
        # 检查观察器的数据类型是否为 quint8，如果不是则发出警告
        if observer.dtype != torch.quint8:
            warn(
                f"PReLU's weight observer should have dtype quint8 but got {observer.dtype}"
            )
        # 计算权重的量化参数 wt_scale 和 wt_zp
        wt_scale, wt_zp = observer.calculate_qparams()
        # 使用量化参数对权重进行量化
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.quint8)
        # 将量化后的权重设置到量化 PReLU 对象中
        qprelu.set_weight(qweight)
        return qprelu

    # 类方法，从参考模型和给定的 scale 和 zero_point 创建量化 PReLU 模型
    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # 创建一个量化 PReLU 对象
        qprelu = cls(float(scale), int(zero_point), mod.num_parameters)
        # 将浮点权重转换为 float 类型
        float_wt = mod.weight.float()
        # 创建权重的量化观察器
        observer = mod.qconfig.weight()
        observer(float_wt)
        # 检查观察器的数据类型是否为 quint8，如果不是则发出警告
        if observer.dtype != torch.quint8:
            warn(
                f"PReLU's weight observer should have dtype quint8 but got {observer.dtype}"
            )
        # 计算权重的量化参数 wt_scale 和 wt_zp
        wt_scale, wt_zp = observer.calculate_qparams()
        # 使用量化参数对权重进行量化
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.quint8)
        # 将量化后的权重设置到量化 PReLU 对象中
        qprelu.set_weight(qweight)
        return qprelu
```