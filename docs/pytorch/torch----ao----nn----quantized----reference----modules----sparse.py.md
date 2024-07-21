# `.\pytorch\torch\ao\nn\quantized\reference\modules\sparse.py`

```
# mypy: allow-untyped-defs
# 引入PyTorch中用于定义神经网络的模块和函数
import torch.nn as nn
import torch.nn.functional as F
# 引入Tensor数据类型
from torch import Tensor
# 引入自定义的ReferenceQuantizedModule工具类
from .utils import ReferenceQuantizedModule
# 引入类型提示工具
from typing import Optional, Dict, Any

# 定义导出的模块列表
__all__ = ['Embedding', 'EmbeddingBag']

# 定义量化的Embedding类，继承自nn.Embedding和ReferenceQuantizedModule类
class Embedding(nn.Embedding, ReferenceQuantizedModule):
    """ A reference quantized Embedding module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 device=None, dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None) -> None:
        # 调用父类的构造函数，初始化Embedding模块
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight, device, dtype)
        # 初始化权重量化参数
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        # 返回模块名称的方法
        return "QuantizedEmbedding(Reference)"

    def forward(self, input: Tensor) -> Tensor:
        # 获取量化和反量化后的权重
        weight_quant_dequant = self.get_weight()
        # 调用PyTorch中的embedding函数，进行嵌入操作
        return F.embedding(
            input, weight_quant_dequant, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    @classmethod
    def from_float(cls, mod, weight_qparams):
        # 根据浮点数模型创建量化的Embedding模块的类方法
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            mod.weight,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams)

# 定义量化的EmbeddingBag类，继承自nn.EmbeddingBag和ReferenceQuantizedModule类
class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):
    """ A reference quantized EmbeddingBag module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """
    # 初始化函数，用于初始化量化嵌入层对象
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 max_norm: Optional[float] = None, norm_type: float = 2., 
                 scale_grad_by_freq: bool = False, mode: str = 'mean', 
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 include_last_offset: bool = False, padding_idx: Optional[int] = None,
                 device=None, dtype=None,
                 weight_qparams: Optional[Dict[str, Any]] = None) -> None:
        # 调用父类的初始化函数，传递所有参数
        super().__init__(num_embeddings, embedding_dim, max_norm, norm_type,
                         scale_grad_by_freq, mode, sparse, _weight, include_last_offset,
                         padding_idx, device, dtype)
        # 调用私有方法 _init_weight_qparams 初始化权重量化参数
        self._init_weight_qparams(weight_qparams, device)

    # 返回嵌入层对象的名称字符串
    def _get_name(self):
        return "QuantizedEmbedding(Reference)"

    # 前向传播函数，接收输入张量 input，偏移量 offsets（可选），样本权重 per_sample_weights（可选），返回张量
    def forward(self, input: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None) -> Tensor:
        # 获取权重量化后的张量
        weight_quant_dequant = self.get_weight()
        # 使用 F.embedding_bag 进行嵌入操作，传递所有相关参数
        return F.embedding_bag(input, weight_quant_dequant, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights, self.include_last_offset,
                               self.padding_idx)

    # 类方法，从浮点模型 mod 转换成量化嵌入层对象，传递权重量化参数 weight_qparams
    @classmethod
    def from_float(cls, mod, weight_qparams, use_precomputed_fake_quant=False):
        return cls(
            mod.num_embeddings,         # 继承自浮点模型的 num_embeddings
            mod.embedding_dim,          # 继承自浮点模型的 embedding_dim
            mod.max_norm,               # 继承自浮点模型的 max_norm
            mod.norm_type,              # 继承自浮点模型的 norm_type
            mod.scale_grad_by_freq,     # 继承自浮点模型的 scale_grad_by_freq
            mod.mode,                   # 继承自浮点模型的 mode
            mod.sparse,                 # 继承自浮点模型的 sparse
            mod.weight,                 # 继承自浮点模型的 weight
            mod.include_last_offset,    # 继承自浮点模型的 include_last_offset
            mod.padding_idx,            # 继承自浮点模型的 padding_idx
            mod.weight.device,          # 继承自浮点模型的 device
            mod.weight.dtype,           # 继承自浮点模型的 dtype
            weight_qparams              # 传入的权重量化参数
        )
```