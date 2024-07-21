# `.\pytorch\torch\ao\nn\quantized\reference\modules\linear.py`

```py
# mypy: allow-untyped-defs
# 引入PyTorch库
import torch
# 引入神经网络模块
import torch.nn as nn
# 引入PyTorch函数库
import torch.nn.functional as F
# 引入类型提示
from typing import Optional, Dict, Any
# 从当前目录的utils模块中引入ReferenceQuantizedModule工具类
from .utils import ReferenceQuantizedModule

# 定义模块的公开接口列表
__all__ = ['Linear']

# 定义线性层类，继承自nn.Linear和ReferenceQuantizedModule
class Linear(nn.Linear, ReferenceQuantizedModule):
    """ A reference quantized linear module that fits into the FX
    Graph Mode Quantization workflow
    activation will be floating point Tensor, we will store floating
    point weight as well in the module, but in forward we'll quantize
    and dequantize the weight before running the floating point functional
    linear operator.
    """
    # 声明此模块为参考量化模块
    _IS_REFERENCE = True

    # 初始化函数，接受输入特征数、输出特征数、是否有偏置、设备和数据类型等参数
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias_: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            weight_qparams: Optional[Dict[str, Any]] = None):
        # 调用父类的初始化方法，设置输入特征数、输出特征数、是否有偏置、设备和数据类型
        super().__init__(in_features, out_features, bias_, device, dtype)
        # 调用内部方法初始化权重量化参数
        self._init_weight_qparams(weight_qparams, device)

    # 获取模块名称的方法
    def _get_name(self):
        return "QuantizedLinear(Reference)"

    # 前向传播方法，接受输入张量x，并返回输出张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.linear ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.linear --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized linear
        """
        # 获取量化后的权重
        weight_quant_dequant = self.get_weight()
        # 使用PyTorch函数库中的线性操作对输入x和权重进行线性计算，并加上偏置
        result = F.linear(x, weight_quant_dequant, self.bias)
        # 返回计算结果
        return result

    # 类方法，从浮点数模型中创建参考量化线性层
    @classmethod
    def from_float(cls, float_linear, weight_qparams):
        # 创建QuantizedLinear(Reference)对象，使用与浮点数模型相同的参数
        qref_linear = Linear(
            float_linear.in_features, float_linear.out_features,
            float_linear.bias is not None, device=float_linear.weight.device,
            dtype=float_linear.weight.dtype, weight_qparams=weight_qparams)
        # 将浮点数模型的权重复制到量化模型中
        qref_linear.weight = torch.nn.Parameter(float_linear.weight.detach())
        # 如果浮点数模型有偏置，也将其复制到量化模型中
        if float_linear.bias is not None:
            qref_linear.bias = torch.nn.Parameter(float_linear.bias.detach())
        # 返回创建的量化线性层对象
        return qref_linear
```