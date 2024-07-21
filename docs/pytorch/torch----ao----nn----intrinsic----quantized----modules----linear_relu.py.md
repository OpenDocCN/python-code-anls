# `.\pytorch\torch\ao\nn\intrinsic\quantized\modules\linear_relu.py`

```
# 导入torch和量化后的神经网络模块
import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic as nni
from torch.ao.nn.quantized.modules.utils import _quantize_weight

# 暴露的类列表，包括以下三个类
__all__ = [
    "LinearReLU",
    "LinearLeakyReLU",
    "LinearTanh",
]

# LinearReLU类继承自nnq.Linear，结合了线性和ReLU激活函数的功能
class LinearReLU(nnq.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    # 使用nni.LinearReLU作为_Float_MODULE，类型忽略赋值检查
    _FLOAT_MODULE = nni.LinearReLU  # type: ignore[assignment]

    # 初始化方法，定义输入特征数、输出特征数、是否包含偏置项和数据类型
    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    # 前向传播方法，应用量化的线性ReLU操作
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_relu(
            x, self._packed_params._packed_params, self.scale, self.zero_point)

    # 获取模块的名称
    def _get_name(self):
        return 'QuantizedLinearReLU'

    # 从浮点模型转换为量化模型的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant)

    # 从参考的线性ReLU模型创建量化模型的类方法
    @classmethod
    def from_reference(cls, ref_linear_relu, output_scale, output_zero_point):
        return super().from_reference(ref_linear_relu[0], output_scale, output_zero_point)

# LinearLeakyReLU类继承自nnq.Linear，结合了线性和LeakyReLU激活函数的功能
class LinearLeakyReLU(nnq.Linear):
    r"""
    For onednn backend only
    A LinearLeakyReLU module fused from Linear and LeakyReLU modules
    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.
    Attributes:
        Same as torch.ao.nn.quantized.Linear
        + negative_slope
    Examples::
        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearLeakyReLU(20, 30, 0.01)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    # 使用nni.LinearLeakyReLU作为_Float_MODULE，类型忽略赋值检查
    _FLOAT_MODULE = nni.LinearLeakyReLU  # type: ignore[assignment]

    # 初始化方法，定义输入特征数、输出特征数、负斜率、是否包含偏置项和数据类型
    def __init__(self, in_features, out_features, negative_slope, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)
        self.negative_slope = negative_slope

    # 前向传播方法，应用量化的线性LeakyReLU操作
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_leaky_relu(
            x, self._packed_params._packed_params, self.scale, self.zero_point, self.negative_slope)

    # 获取模块的名称
    def _get_name(self):
        return 'QuantizedLinearLeakyReLU'

    # 从浮点模型转换为量化模型的类方法
    @classmethod
    # 根据浮点模块创建量化线性LeakyReLU模块
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 断言输入的模块是LinearLeakyReLU类型
        assert type(mod) == nni.LinearLeakyReLU, 'Input float module should be LinearLeakyReLU'
        # 断言输入的浮点模块必须有qconfig属性定义
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        
        # 获取激活后处理器
        activation_post_process = mod.activation_post_process
        # 获取LeakyReLU层
        leaky_relu = mod[1]
        # 获取线性层
        mod = mod[0]
        
        # 获取权重后处理器并对模块的权重进行处理
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        # 获取权重的数据类型
        dtype = weight_post_process.dtype
        
        # 计算激活量化参数
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        # 断言权重观察器必须是torch.qint8类型
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        
        # 对模块的权重进行量化
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        
        # 创建量化的线性LeakyReLU模块
        qlinear_leaky_relu = cls(
            mod.in_features,
            mod.out_features,
            leaky_relu.negative_slope,
            dtype=dtype)
        
        # 设置量化后的权重和偏置
        qlinear_leaky_relu.set_weight_bias(qweight, mod.bias)
        # 设置量化后的缩放因子和零点
        qlinear_leaky_relu.scale = float(act_scale)
        qlinear_leaky_relu.zero_point = int(act_zp)
        
        # 返回量化后的线性LeakyReLU模块
        return qlinear_leaky_relu

    # 根据参考模块创建量化的线性LeakyReLU模块
    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        # 获取参考模块中的线性层和LeakyReLU层
        linear = ref_mod[0]
        leaky_relu = ref_mod[1]
        
        # 创建量化的线性LeakyReLU模块
        qlinear_leaky_relu = cls(
            linear.in_features,
            linear.out_features,
            leaky_relu.negative_slope)
        
        # 获取线性层的量化权重
        qweight = linear.get_quantized_weight()
        
        # 设置量化后的权重和偏置
        qlinear_leaky_relu.set_weight_bias(qweight, linear.bias)
        # 设置量化后的缩放因子和零点
        qlinear_leaky_relu.scale = float(output_scale)
        qlinear_leaky_relu.zero_point = int(output_zero_point)
        
        # 返回量化后的线性LeakyReLU模块
        return qlinear_leaky_relu
class LinearTanh(nnq.Linear):
    r"""
    A LinearTanh module fused from Linear and Tanh modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearTanh(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearTanh  # type: ignore[assignment]

    # 初始化函数，定义了 LinearTanh 类的属性和行为
    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    # 前向传播函数，实现了量化的线性和双曲正切的结合
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_tanh(
            x, self._packed_params._packed_params, self.scale, self.zero_point)

    # 返回类的名称
    def _get_name(self):
        return 'QuantizedLinearTanh'

    # 从浮点模型转换成量化模型的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 确保输入的浮点模型是 LinearTanh 类型
        assert type(mod) == nni.LinearTanh, 'Input float module should be LinearTanh'
        # 确保输入的浮点模型有 qconfig 属性
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        # 获取激活后处理器
        activation_post_process = mod.activation_post_process
        mod = mod[0]
        # 获取权重后处理器
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        # 计算激活的量化参数
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        # 确保权重观察器的数据类型是 torch.qint8
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        # 对权重进行量化
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # 创建并初始化量化的 LinearTanh 类
        qlinear_tanh = cls(
            mod.in_features,
            mod.out_features,
            dtype=dtype)
        qlinear_tanh.set_weight_bias(qweight, mod.bias)
        qlinear_tanh.scale = float(act_scale)
        qlinear_tanh.zero_point = int(act_zp)
        return qlinear_tanh

    # 从参考模型转换成量化模型的类方法
    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        # 获取参考模型的线性层
        linear = ref_mod[0]
        # 创建并初始化量化的 LinearTanh 类
        qlinear_tanh = cls(
            linear.in_features,
            linear.out_features)
        qweight = linear.get_quantized_weight()
        qlinear_tanh.set_weight_bias(qweight, linear.bias)
        qlinear_tanh.scale = float(output_scale)
        qlinear_tanh.zero_point = int(output_zero_point)
        return qlinear_tanh
```