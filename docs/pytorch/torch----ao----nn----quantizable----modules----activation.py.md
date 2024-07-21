# `.\pytorch\torch\ao\nn\quantizable\modules\activation.py`

```
# mypy: allow-untyped-defs
import torch  # 导入 PyTorch 库
import torch.jit  # 避免循环导入需要导入的模块
from torch import nn  # 导入神经网络模块
import torch.nn.functional as nnF  # 导入神经网络函数模块

from torch import Tensor  # 导入 Tensor 数据类型
from typing import Optional, Tuple  # 导入类型提示需要的模块

import warnings  # 导入警告模块

__all__ = [
    "MultiheadAttention"
]

class MultiheadAttention(nn.MultiheadAttention):
    _FLOAT_MODULE = nn.MultiheadAttention

    r"""Quantizable implementation of the MultiheadAttention.

    Note::
        Please, refer to :class:`~torch.nn.MultiheadAttention` for more
        information

    Allows the model to jointly attend to information from different
    representation subspaces.
    See reference: Attention Is All You Need

    The original MHA module is not quantizable.
    This reimplements it by explicitly instantiating the linear layers.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> multihead_attn = nnqa.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    Note::
        Please, follow the quantization flow to convert the quantizable MHA.
    """
    __constants__ = ['batch_first']  # 定义常量列表包含 'batch_first' 字符串
    # 初始化方法，设置多头注意力的参数和模型配置
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0., bias: bool = True,
                 add_bias_kv: bool = False, add_zero_attn: bool = False,
                 kdim: Optional[int] = None, vdim: Optional[int] = None, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        # 设置工厂参数字典，用于后续组件的设备和数据类型配置
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的初始化方法，传递多头注意力的相关参数和工厂参数
        super().__init__(embed_dim, num_heads, dropout,
                         bias, add_bias_kv,
                         add_zero_attn, kdim, vdim, batch_first,
                         **factory_kwargs)
        # 初始化线性层，用于计算查询向量（Q），键向量（K），值向量（V），和输出的投影
        self.linear_Q = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.linear_K = nn.Linear(self.kdim, self.embed_dim, bias=bias, **factory_kwargs)
        self.linear_V = nn.Linear(self.vdim, self.embed_dim, bias=bias, **factory_kwargs)
        # 初始化输出投影层，用于将注意力输出投影回原始维度
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)  # type: ignore[assignment]

        # Functionals
        # 初始化量化功能模块，用于量化乘积的结果
        self.q_scaling_product = torch.ao.nn.quantized.FloatFunctional()
        # 注意：在顶部导入 torch.ao.nn.quantized 可能会导致循环导入

        # Quant/Dequant
        # 初始化量化/反量化模块，用于注意力输出和注意力权重的量化和反量化
        self.quant_attn_output = torch.ao.quantization.QuantStub()
        self.quant_attn_output_weights = torch.ao.quantization.QuantStub()
        self.dequant_q = torch.ao.quantization.DeQuantStub()
        self.dequant_k = torch.ao.quantization.DeQuantStub()
        self.dequant_v = torch.ao.quantization.DeQuantStub()

    # 获取类名称的私有方法
    def _get_name(self):
        return 'QuantizableMultiheadAttention'

    # 类方法修饰符，用于 Torch 的 JIT 编译时标记为未使用
    @classmethod
    @torch.jit.unused
    # 将量化的多头注意力机制（MHA）转换回浮点数表示的工具函数

    def dequantize(self):
        r"""Utility to convert the quantized MHA back to float.

        The motivation for this is that it is not trivial to convert the weights
        from the format that is used in the quantized version back to the
        float.
        """

        # 创建一个新的浮点数表示的多头注意力机制对象
        fp = self._FLOAT_MODULE(self.embed_dim, self.num_heads, self.dropout,
                                (self.linear_Q._weight_bias()[1] is not None),
                                (self.bias_k is not None),
                                self.add_zero_attn, self.kdim, self.vdim, self.batch_first)

        # 确保新对象与当前对象的一些属性一致
        assert fp._qkv_same_embed_dim == self._qkv_same_embed_dim

        # 如果存在偏置项，将其量化解码后作为新对象的参数
        if self.bias_k is not None:
            fp.bias_k = nn.Parameter(self.bias_k.dequantize())
        if self.bias_v is not None:
            fp.bias_v = nn.Parameter(self.bias_v.dequantize())

        # 设置线性层的权重
        w, b = self.out_proj._weight_bias()  # type: ignore[operator, has-type]
        fp.out_proj.weight = nn.Parameter(w.dequantize())
        if b is not None:
            fp.out_proj.bias = nn.Parameter(b)

        # 解码并设置线性层 Q、K、V 的权重
        wQ, bQ = self.linear_Q._weight_bias()  # type: ignore[operator]
        wQ = wQ.dequantize()
        wK, bK = self.linear_K._weight_bias()  # type: ignore[operator]
        wK = wK.dequantize()
        wV, bV = self.linear_V._weight_bias()  # type: ignore[operator]
        wV = wV.dequantize()

        # 根据是否共享嵌入维度，设置不同的输入投影权重和偏置
        if fp._qkv_same_embed_dim:
            # 使用不同的参数
            _start = 0
            _end = _start + fp.embed_dim
            fp.in_proj_weight[_start:_end, :] = wQ
            if fp.in_proj_bias is not None:
                assert all(bQ == 0)
                fp.in_proj_bias[_start:_end] = bQ

            _start = _end
            _end = _start + fp.embed_dim
            fp.in_proj_weight[_start:_end, :] = wK
            if fp.in_proj_bias is not None:
                assert all(bK == 0)
                fp.in_proj_bias[_start:_end] = bK

            _start = _end
            fp.in_proj_weight[_start:, :] = wV
            if fp.in_proj_bias is not None:
                assert all(bV == 0)
                fp.in_proj_bias[_start:] = bV
        else:
            # 使用分开的参数
            fp.q_proj_weight = nn.Parameter(wQ)
            fp.k_proj_weight = nn.Parameter(wK)
            fp.v_proj_weight = nn.Parameter(wV)
            if fp.in_proj_bias is None:
                self.linear_Q.bias = None
                self.linear_K.bias = None
                self.linear_V.bias = None
            else:
                fp.in_proj_bias[0:fp.embed_dim] = bQ
                fp.in_proj_bias[fp.embed_dim:(fp.embed_dim * 2)] = bK
                fp.in_proj_bias[(fp.embed_dim * 2):] = bV

        # 返回转换后的浮点数表示的多头注意力机制对象
        return fp

    @classmethod
    # 定义一个类方法 `from_observed`，用于从观测值转换为量化值
    def from_observed(cls, other):
        # 整个流程是从浮点数 -> 观测值 -> 量化值
        # 这个类只处理从浮点数 -> 观测值的转换
        # 参考 nn.quantized.MultiheadAttention 查看更多信息
        raise NotImplementedError("It looks like you are trying to prepare an "
                                  "MHA module. Please, see "
                                  "the examples on quantizable MHAs.")

    # 定义前向传播方法 `forward`
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True,
                is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        注意::
            请参考 :func:`~torch.nn.MultiheadAttention.forward` 获取更多信息

        Args:
            query, key, value: 将查询(query)和一组键-值对映射到输出
                更多细节请参见 "Attention Is All You Need"
            key_padding_mask: 如果提供，则在注意力计算时忽略键中指定的填充元素。
                给定一个二进制掩码，如果值为 True，则在注意力层中对应的值将被忽略。
            need_weights: 输出注意力权重
            attn_mask: 2D 或 3D 掩码，防止注意力机制关注特定位置。
                2D 掩码将广播到所有批次，而 3D 掩码允许为每个批次的条目指定不同的掩码。
        """
    """
            Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
              Default: ``False``.
            - average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
              heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
              effect when ``need_weights=True.``. Default: True (i.e. average weights across heads)
    
            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - attn_output_weights: If ``average_attn_weights=True``, returns attention weights averaged
              across heads of shape :math:`(N, L, S)`, where N is the batch size, L is the target sequence length,
              S is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(N, num_heads, L, S)`.
            """
            return self._forward_impl(query, key, value, key_padding_mask,
                                      need_weights, attn_mask, average_attn_weights,
                                      is_causal)
```