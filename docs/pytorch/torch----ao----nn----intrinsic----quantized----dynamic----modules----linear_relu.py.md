# `.\pytorch\torch\ao\nn\intrinsic\quantized\dynamic\modules\linear_relu.py`

```py
# mypy: allow-untyped-defs
# 引入 torch 库
import torch
# 引入动态量化的相关模块
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.intrinsic as nni

# 定义公开的类名列表
__all__ = [
    "LinearReLU"
]

# 定义 LinearReLU 类，继承自 nnqd.Linear
class LinearReLU(nnqd.Linear):
    """
    A LinearReLU module fused from Linear and ReLU modules that can be used
    for dynamic quantization.
    Supports both, FP16 and INT8 quantization.

    We adopt the same interface as :class:`torch.ao.nn.quantized.dynamic.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.dynamic.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.quantized.dynamic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # 定义 FLOAT_MODULE 类变量，类型为 nni.LinearReLU，忽略类型检查
    _FLOAT_MODULE = nni.LinearReLU  # type: ignore[assignment]

    # 初始化方法，接收输入特征数、输出特征数、是否包含偏置和数据类型
    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    # 前向传播方法，接收输入张量 x，返回输出张量 Y
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 根据参数类型选择量化线性整流操作函数
        if self._packed_params.dtype == torch.qint8:
            # 如果是 qint8 类型的参数，使用量化线性整流动态操作函数
            # TODO 检查是否应该在此处默认设置 reduce_range = True
            Y = torch.ops.quantized.linear_relu_dynamic(
                x, self._packed_params._packed_params, reduce_range=True)
        elif self._packed_params.dtype == torch.float16:
            # 如果是 float16 类型的参数，使用量化线性整流动态操作函数（FP16 版本）
            Y = torch.ops.quantized.linear_relu_dynamic_fp16(
                x, self._packed_params._packed_params)
        else:
            # 抛出运行时异常，指示不支持的量化类型
            raise RuntimeError('Unsupported dtype on dynamic quantized linear relu!')
        # 将输出张量 Y 转换为输入张量 x 的数据类型，并返回
        return Y.to(x.dtype)

    # 获取类的名称方法，返回字符串 'DynamicQuantizedLinearReLU'
    def _get_name(self):
        return 'DynamicQuantizedLinearReLU'

    # 从浮点模型转换成量化模型的类方法，接收浮点模型 mod 和是否使用预先计算的伪量化标志
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    # 从参考量化线性整流模型转换成当前模型的类方法，接收参考量化线性整流模型 ref_qlinear_relu
    @classmethod
    def from_reference(cls, ref_qlinear_relu):
        return super().from_reference(ref_qlinear_relu[0])
```