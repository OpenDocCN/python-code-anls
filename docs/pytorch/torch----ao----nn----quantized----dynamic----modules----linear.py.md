# `.\pytorch\torch\ao\nn\quantized\dynamic\modules\linear.py`

```py
# mypy: allow-untyped-defs
# 导入PyTorch库
import torch
# 导入量化的神经网络模块
import torch.ao.nn.quantized as nnq
# 导入权重量化工具函数
from torch.ao.nn.quantized.modules.utils import _quantize_weight
# 导入内部量化模块
import torch.ao.nn.intrinsic as nni

# 模块公开的内容，只包括"Linear"
__all__ = [
    "Linear",
]

# 定义一个动态量化线性模块，输入输出是浮点数张量
class Linear(nnq.Linear):
    r"""
    A dynamic quantized linear module with floating point tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module which are of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable floating point bias of the module of shape
                       :math:`(\text{out\_features})`. If :attr:`bias` is ``True``,
                       the values are initialized to zero.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.quantized.dynamic.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    # 版本号，用于区分本类和父类nnq.Linear的版本
    _version = 4

    def __init__(self, in_features, out_features, bias_=True, dtype=torch.qint8):
        # 调用父类构造函数初始化模块
        super().__init__(in_features, out_features, bias_, dtype=dtype)
        # 设置版本号为4
        self.version = 4

    def forward(self, x):
        # 处理self.bias为None的情况
        if self._packed_params.dtype == torch.qint8:
            # 根据dtype调用不同的量化动态线性运算符
            if self.version is None or self.version < 4:
                Y = torch.ops.quantized.linear_dynamic(
                    x, self._packed_params._packed_params)
            else:
                Y = torch.ops.quantized.linear_dynamic(
                    x, self._packed_params._packed_params, reduce_range=True)
        elif self._packed_params.dtype == torch.float16:
            # 调用FP16版本的量化动态线性运算符
            Y = torch.ops.quantized.linear_dynamic_fp16(
                x, self._packed_params._packed_params)
        else:
            # 抛出不支持的dtype错误
            raise RuntimeError('Unsupported dtype on dynamic quantized linear!')
        # 将输出张量转换为输入张量的数据类型并返回
        return Y.to(x.dtype)

    def _get_name(self):
        # 返回模块的名称
        return 'DynamicQuantizedLinear'

    def extra_repr(self):
        # 返回模块的额外表示字符串，包括输入特征数、输出特征数和dtype
        extra_repr_str = f'in_features={self.in_features}, out_features={self.out_features}, dtype={self._packed_params.dtype}'
        if self._packed_params.dtype == torch.qint8:
            # 如果dtype是qint8，附加qscheme信息到额外表示字符串
            extra_repr_str += f', qscheme={self.weight().qscheme()}'
        return extra_repr_str
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 从本地元数据中获取版本号，如果不存在则为 None
        version = local_metadata.get('version', None)
        # 将获取到的版本号赋值给当前对象的版本属性
        self.version = version
        # 调用父类的 _load_from_state_dict 方法来加载模型状态字典
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a dynamic quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
        """
        # 支持转换的浮点模块类型列表
        float_modules = [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
                         torch.ao.nn.intrinsic.modules.fused.LinearReLU, torch.ao.nn.qat.dynamic.Linear]

        # 断言输入的模块类型在支持转换的浮点模块类型列表中
        assert type(mod) in float_modules, \
            'nn.quantized.dynamic.Linear.from_float only works for one of' + \
            str([float_mod.__name__ for float_mod in float_modules])
        # 断言输入的浮点模块有 qconfig 属性
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        # 如果输入模块类型是 nni.LinearReLU，则使用其第一个元素
        if type(mod) == nni.LinearReLU:
            mod = mod[0]
        # 如果模块的 qconfig 不为 None，并且其 weight 属性有 qconfig 定义，则使用模块的 weight 观察器
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            # 当导入 qconfig 会产生循环导入问题时，延迟导入的解决方法
            from torch.ao.quantization.qconfig import default_dynamic_qconfig
            weight_observer = default_dynamic_qconfig.weight()
        # 获取 weight 观察器的数据类型
        dtype = weight_observer.dtype
        # 断言数据类型为 qint8 或 float16，这是动态量化线性模块支持的唯一数据类型
        assert dtype in [torch.qint8, torch.float16], "The only supported dtypes for " \
            f"dynamic quantized linear are qint8 and float16 got: {dtype}"
        # 将 weight 观察器应用于模块的权重
        weight_observer(mod.weight)
        # 根据数据类型选择相应的量化权重处理方法
        if dtype == torch.qint8:
            qweight = _quantize_weight(mod.weight.float(), weight_observer)
        elif dtype == torch.float16:
            qweight = mod.weight.float()
        else:
            raise RuntimeError('Unsupported dtype specified for dynamic quantized Linear!')
        # 创建一个动态量化线性模块实例
        qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)
        # 设置动态量化线性模块的权重和偏置
        qlinear.set_weight_bias(qweight, mod.bias)
        # 返回创建的动态量化线性模块实例
        return qlinear
    # 从参考的量化模块创建一个动态量化模块 (fbgemm/qnnpack)
    def from_reference(cls, ref_qlinear):
        """ Create a (fbgemm/qnnpack) dynamic quantized module from a reference quantized
        module
        Args:
            ref_qlinear (Module): a reference quantized  module, either produced by
            torch.ao.quantization functions or provided by the user
        """
        # 使用类方法 cls 创建一个新的量化线性模块，传入输入特征数、输出特征数和权重数据类型
        qlinear = cls(ref_qlinear.in_features, ref_qlinear.out_features, dtype=ref_qlinear.weight_dtype)
        # 获取参考量化模块的量化权重
        qweight = ref_qlinear.get_quantized_weight()
        # 获取参考量化模块的偏置
        bias = ref_qlinear.bias
        # 设置新创建的量化模块的权重和偏置
        qlinear.set_weight_bias(qweight, bias)
        # 返回创建的动态量化模块
        return qlinear
```