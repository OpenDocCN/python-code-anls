# `.\pytorch\torch\ao\quantization\experimental\quantizer.py`

```py
# mypy: allow-untyped-defs
# 导入PyTorch库及相关模块
import torch
from torch import Tensor
import numpy as np
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float, quant_dequant_util

# APoTQuantizer类用于存储APoT量化器，并实现量化和反量化操作
class APoTQuantizer:
    alpha: torch.Tensor
    gamma: torch.Tensor
    quantization_levels: torch.Tensor
    level_indices: torch.Tensor

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        quantization_levels: torch.Tensor,
        level_indices: torch.Tensor) -> None:
        # 初始化APoTQuantizer实例的属性
        self.alpha = alpha
        self.gamma = gamma
        self.quantization_levels = quantization_levels
        self.level_indices = level_indices

    r""" Quantizes fp Tensor to integer APoT representation.
    根据指定的APoT非均匀观察器的qparams，将fp张量量化为整数APoT表示。
    该方法遵循APoT论文中概述的方法：https://arxiv.org/pdf/1909.13144.pdf。
    Args:
        tensor2quantize: fp张量
    Returns:
        result: tensor2quantize的APoT张量表示
    """
    def quantize(self, tensor2quantize: Tensor):
        result = torch.tensor([])

        # 对tensor2quantize的元素应用float_to_apot映射
        tensor2quantize = tensor2quantize.detach().apply_(lambda x: float_to_apot(x,
                                                                                  self.quantization_levels,
                                                                                  self.level_indices,
                                                                                  self.alpha))

        # 转换为dtype的APoT整数表示
        tensor2quantize = tensor2quantize.int()

        from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT

        result = TensorAPoT(self, tensor2quantize)  # type: ignore[assignment]

        return result

    r""" Dequantizes integer Tensor to floating point (fp) representation
    基于来自指定APoT非均匀观察器的计算量化级别，将整数张量反量化为浮点（fp）表示。
    该方法遵循APoT论文中概述的方法：https://arxiv.org/pdf/1909.13144.pdf。
    Args:
        tensor2quantize: fp张量
    Returns:
        result: 输入张量的fp降低精度表示
    """
    def dequantize(self, apot_tensor) -> Tensor:
        orig_size = apot_tensor.data.size()
        apot_tensor_data = apot_tensor.data.flatten()

        print(apot_tensor_data)

        # 对apot_tensor_data的元素应用apot_to_float映射
        result_temp = np.empty(shape=apot_tensor_data.size())
        for i in range(len(apot_tensor_data)):
            new_ele = apot_to_float(apot_tensor_data[i], self.quantization_levels, self.level_indices)
            result_temp[i] = new_ele

        result = torch.from_numpy(result_temp).reshape(orig_size)

        return result

    r""" Returns result of quantize -> dequantize on a fp Tensor (reduced precision)
    返回fp张量（降低精度）上量化->反量化操作的结果
    ```
    # 定义一个类方法用于量化和反量化操作，基于指定的非均匀观察者的量化级别。
    # 这个方法遵循了APoT论文中描述的方法：https://arxiv.org/pdf/1909.13144.pdf。
    class Quantizer:
        def quant_dequant(self, tensor2quantize: Tensor) -> Tensor:
            # 将量化级别转换为列表
            levels_lst = list(self.quantization_levels)
            
            # 对输入的张量应用量化和反量化的工具函数，返回结果张量
            result = tensor2quantize.apply_(lambda x: quant_dequant_util(x, levels_lst))  # type: ignore[call-arg]
            
            return result
    
        def q_apot_alpha(self) -> float:
            # 如果调用了这个方法，表示子类需要实现这个函数，否则会抛出未实现的错误
            raise NotImplementedError
r""" Global method to create quantizer and call quantizer quantize_APoT
    Args:
        tensor2quantize: fp Tensor to quantize
        alpha: Tensor qparam alpha (clipping level)
        gamma: Tensor qparam gamma (scale factor for quantization levels)
        quantization_levels: Tensor with fp quantization levels
        level_indices: Tensor with integer quantization level indices
    Returns:
        result: ApoT Tensor representation of tensor2quantize
"""
# 定义一个全局方法用于创建量化器并调用其 quantize_APoT 方法，将输入张量 tensor2quantize 进行量化
def quantize_APoT(tensor2quantize: Tensor, alpha: Tensor, gamma: Tensor, quantization_levels: Tensor, level_indices: Tensor):
    # 创建一个 APoTQuantizer 实例，使用给定的 alpha、gamma、quantization_levels 和 level_indices
    quantizer = APoTQuantizer(alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
    # 调用 quantize 方法对 tensor2quantize 进行量化，并返回结果
    result = quantizer.quantize(tensor2quantize)
    return result

r""" Global method to create quantizer and call quantizer dequantize_APoT
    Args:
        apot_tensor: APoT Tensor to dequantize
    Returns:
        result: fp Tensor dequantized from apot_tensor
"""
# 定义一个全局方法用于创建量化器并调用其 dequantize_APoT 方法，对给定的 APoT 张量 apot_tensor 进行反量化
def dequantize_APoT(apot_tensor) -> Tensor:
    # 获取 apot_tensor 对应的量化器
    quantizer = apot_tensor.quantizer
    # 调用 quantizer 的 dequantize 方法对 apot_tensor 进行反量化，并返回结果
    result = quantizer.dequantize(apot_tensor)
    return result

r""" Global method to create quantizer and call quantizer quant_dequant
    Args:
        tensor2quantize: fp Tensor to quantize
        alpha: Tensor qparam alpha (clipping level)
        gamma: Tensor qparam gamma (scale factor for quantization levels)
        quantization_levels: Tensor with fp quantization levels
        level_indices: Tensor with integer quantization level indices
    Returns:
        result: fp reduced precision Tensor from tensor2quantize
"""
# 定义一个全局方法用于创建量化器并调用其 quant_dequant 方法，对给定的 fp 张量 tensor2quantize 进行量化再反量化
def quant_dequant_APoT(tensor2quantize: Tensor,
                       alpha: Tensor,
                       gamma: Tensor,
                       quantization_levels: Tensor,
                       level_indices: Tensor) -> Tensor:
    # 创建一个 APoTQuantizer 实例，使用给定的 alpha、gamma、quantization_levels 和 level_indices
    quantizer = APoTQuantizer(alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
    # 调用 quantizer 的 quant_dequant 方法对 tensor2quantize 进行量化再反量化，并返回结果
    result = quantizer.quant_dequant(tensor2quantize)
    return result
```