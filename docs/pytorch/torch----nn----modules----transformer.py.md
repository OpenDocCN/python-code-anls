# `.\pytorch\torch\nn\modules\transformer.py`

```
# 指定允许未类型化的函数定义（用于类型检查）
# 导入必要的库和模块
import copy  # 导入copy模块，用于对象的深拷贝操作
import warnings  # 导入warnings模块，用于警告处理
from typing import Any, Callable, Optional, Union  # 导入类型注解相关的模块

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数模块，用于神经网络操作
from torch import Tensor  # 导入PyTorch的张量类型
from torch.nn.init import xavier_uniform_  # 导入PyTorch的参数初始化函数

from .activation import MultiheadAttention  # 导入自定义模块中的多头注意力机制
from .container import ModuleList  # 导入自定义模块中的模块列表容器
from .dropout import Dropout  # 导入自定义模块中的Dropout层
from .linear import Linear  # 导入自定义模块中的线性层
from .module import Module  # 导入自定义模块基类
from .normalization import LayerNorm  # 导入自定义模块中的层归一化

# 定义导出的模块和类名列表
__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
]


def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    # 如果设备未指定，则默认为CPU
    if device is None:
        device = torch.device("cpu")
    # 如果数据类型未指定，则默认为float32
    if dtype is None:
        dtype = torch.float32
    # 生成一个方形的因果屏蔽矩阵，用于序列
    # 被屏蔽的位置填充为float('-inf')，未屏蔽的位置填充为float(0.0)
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    # 如果输入张量是嵌套的，则返回None
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        # 如果张量维度为2，则表示未批量处理：S, E
        if len(src_size) == 2:
            return src_size[0]  # 返回序列长度S
        else:
            # 如果张量维度为3，则表示批量处理：B, S, E（如果batch_first=True）或者S, B, E（如果batch_first=False）
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]  # 返回序列长度S或者B，根据batch_first决定
            

class Transformer(Module):
    r"""A transformer model.

    User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.
    Args:
        d_model: 编码器/解码器输入中期望的特征数（默认为512）。
        nhead: 多头注意力模型中的头数（默认为8）。
        num_encoder_layers: 编码器中的子编码器层数（默认为6）。
        num_decoder_layers: 解码器中的子解码器层数（默认为6）。
        dim_feedforward: 前馈网络模型的维度（默认为2048）。
        dropout: dropout 值（默认为0.1）。
        activation: 编码器/解码器中间层的激活函数，可以是字符串（"relu"或"gelu"）或一元可调用对象。默认为relu。
        custom_encoder: 自定义编码器（默认为None）。
        custom_decoder: 自定义解码器（默认为None）。
        layer_norm_eps: 层归一化组件中的 eps 值（默认为1e-5）。
        batch_first: 如果为``True``，则输入和输出张量为(batch, seq, feature)。默认为``False``（seq, batch, feature）。
        norm_first: 如果为``True``，编码器和解码器层将在其他注意力和前馈操作之前执行 LayerNorm，否则在之后执行。默认为``False``（之后执行）。
        bias: 如果设置为``False``，则``Linear``和``LayerNorm``层将不学习加性偏置。默认为``True``。
        device: 指定设备（默认为None）。
        dtype: 指定数据类型（默认为None）。

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: 可以在 https://github.com/pytorch/examples/tree/master/word_language_model 查看如何应用 nn.Transformer 模块进行单词语言模型的完整示例。
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            encoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            decoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        src_is_causal: Optional[bool] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ):
        # 此方法定义了 Transformer 模型的前向传播逻辑，处理输入和掩码，生成输出
        pass

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        r"""Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        # 使用辅助函数生成一个方形的因果关系掩码，用于序列数据
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        # 对 Transformer 模型中的参数进行初始化，使用 Xavier uniform 分布初始化权重
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
# TransformerEncoder 类是一堆 N 个编码器层组成的堆栈。
# 用户可以使用相应的参数构建 BERT 模型（参考 https://arxiv.org/abs/1810.04805）。

class TransformerEncoder(Module):
    # __constants__ 是一个类属性，定义为一个包含字符串 "norm" 的常量列表
    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 记录 API 使用情况，使用一次日志记录函数
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        # 复制编码器层对象，创建编码器层列表
        self.layers = _get_clones(encoder_layer, num_layers)
        # 记录编码器层数量
        self.num_layers = num_layers
        # 记录归一化（规范化）层对象
        self.norm = norm
        # 记录是否启用嵌套张量
        self.enable_nested_tensor = enable_nested_tensor
        # 控制是否使用嵌套张量
        self.use_nested_tensor = enable_nested_tensor
        # 记录掩码检查对象
        self.mask_check = mask_check

        # 检查编码器层对象的类型及属性，记录可能的性能优化限制
        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ""
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first:
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn.batch_first was not True"
                + "(use batch_first for better inference performance)"
            )
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
            )
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.activation_relu_or_gelu was not True"
            )
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps):
            why_not_sparsity_fast_path = (
                f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
            )
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        # 如果启用嵌套张量并且存在性能限制的情况，发出警告并禁用嵌套张量
        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(
                f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}"
            )
            self.use_nested_tensor = False

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
# TransformerDecoder 类定义，表示一个由多个解码器层堆叠而成的 Transformer 解码器
class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
            解码器层对象，必须是 TransformerDecoderLayer 类的一个实例。
        num_layers: the number of sub-decoder-layers in the decoder (required).
            解码器中子解码器层的数量，必须指定。
        norm: the layer normalization component (optional).
            可选的层归一化组件。

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    # 定义常量列表，指明该类的常量属性，这里包含了 "norm"
    __constants__ = ["norm"]

    # 初始化方法，设置 TransformerDecoder 对象的初始状态
    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 记录 API 使用日志
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        # 克隆并存储多个解码器层对象
        self.layers = _get_clones(decoder_layer, num_layers)
        # 存储解码器层的数量
        self.num_layers = num_layers
        # 存储归一化组件
        self.norm = norm

    # 前向传播方法，定义数据在解码器中的传播过程
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: Optional[bool] = None,
            memory_is_causal: bool = False,
        ) -> Tensor:
            r"""Pass the inputs (and mask) through the decoder layer in turn.
    
            Args:
                tgt: the sequence to the decoder (required).
                memory: the sequence from the last layer of the encoder (required).
                tgt_mask: the mask for the tgt sequence (optional).
                memory_mask: the mask for the memory sequence (optional).
                tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                memory_key_padding_mask: the mask for the memory keys per batch (optional).
                tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                    Default: ``None``; try to detect a causal mask.
                    Warning:
                    ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                    the causal mask. Providing incorrect hints can result in
                    incorrect execution, including forward and backward
                    compatibility.
                memory_is_causal: If specified, applies a causal mask as
                    ``memory mask``.
                    Default: ``False``.
                    Warning:
                    ``memory_is_causal`` provides a hint that
                    ``memory_mask`` is the causal mask. Providing incorrect
                    hints can result in incorrect execution, including
                    forward and backward compatibility.
    
            Shape:
                see the docs in :class:`~torch.nn.Transformer`.
            """
            # 初始化输出为目标序列
            output = tgt
    
            # 获取目标序列的长度
            seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
            
            # 检测是否需要应用因果掩码到目标序列
            tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
    
            # 逐层通过解码器层处理输出
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_is_causal=tgt_is_causal,
                    memory_is_causal=memory_is_causal,
                )
    
            # 如果有归一化层，则对输出进行归一化处理
            if self.norm is not None:
                output = self.norm(output)
    
            # 返回处理后的输出
            return output
# 定义 TransformerEncoderLayer 类，继承自 Module 类
class TransformerEncoderLayer(Module):
    # TransformerEncoderLayer 是由自注意力机制和前馈神经网络组成的标准编码器层
    # 该标准编码器层基于论文 "Attention Is All You Need"
    # Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    # Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    # Neural Information Processing Systems, pages 6000-6010. 用户可以在应用过程中修改或实现不同的方式。

    # TransformerEncoderLayer 能够处理传统的 torch.tensor 输入，也可以处理 Nested Tensor 输入。
    # 派生类应当能够同样接受这两种输入格式。
    # （目前 Nested Tensor 处于原型状态，TransformerEncoderLayer 并不支持所有输入组合。）

    # 如果你正在实现一个自定义层，可以将其派生自 Module 或 TransformerEncoderLayer 类。
    # 如果你的自定义层支持 torch.Tensor 和 Nested Tensor 输入，将其实现为 TransformerEncoderLayer 的派生类。
    # 如果你的自定义层只支持 torch.Tensor 输入，则将其实现为 Module 的派生类。

    # 参数：
    #   d_model: 输入特征的数量（必填）
    #   nhead: 多头注意力模型中的头数（必填）
    #   dim_feedforward: 前馈网络模型的维度（默认为 2048）
    #   dropout: dropout 值（默认为 0.1）
    #   activation: 中间层的激活函数，可以是字符串 ("relu" 或 "gelu") 或一元可调用对象（默认为 relu）
    #   layer_norm_eps: 层归一化组件中的 eps 值（默认为 1e-5）
    #   batch_first: 如果为 True，则输入和输出张量提供为 (batch, seq, feature)。默认为 False（seq, batch, feature）
    #   norm_first: 如果为 True，则在注意力和前馈操作之前进行层归一化。否则在操作之后进行。默认为 False（在操作之后）
    #   bias: 如果设置为 False，则 Linear 和 LayerNorm 层将不会学习一个附加的偏置。默认为 True。

    # 示例：
    #   >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    #   >>> src = torch.rand(10, 32, 512)
    #   >>> out = encoder_layer(src)
    #
    # 另一种方式，当 batch_first 为 True 时：
    #   >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
    #   >>> src = torch.rand(32, 10, 512)
    #   >>> out = encoder_layer(src)
    """
    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
          自动求导已禁用（使用 ``torch.inference_mode`` 或 ``torch.no_grad``）或者没有张量参数设置了 ``requires_grad``

        - training is disabled (using ``.eval()``)
          训练模式已禁用（使用 ``.eval()``）

        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
          batch_first 设置为 ``True`` 并且输入是批处理的（即 ``src.dim() == 3``）

        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
          激活函数为 ``"relu"``, ``"gelu"``, ``torch.functional.relu``, 或 ``torch.functional.gelu`` 中的一种

        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
          最多只能传递 ``src_mask`` 和 ``src_key_padding_mask`` 中的一个

        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
          如果 ``src`` 是 `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_，则不能传递 ``src_mask`` 和 ``src_key_padding_mask``

        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)
          两个 ``LayerNorm`` 实例具有一致的 ``eps`` 值（除非调用者在未修改另一个的情况下手动修改了其中一个）

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
        
        如果使用了优化的实现，则可以传递 `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ 以更有效地表示填充，
        而不是使用填充掩码。在这种情况下，将返回一个 `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_，
        并且可以期望与填充部分的比例成正比的额外加速。

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
    ) -> None:
        # 定义一个空函数，无返回值
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的构造函数
        super().__init__()
        # 创建多头注意力机制
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        # 实现前馈神经网络模型
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        # 实现第一个层标准化
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # 实现第二个层标准化
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # 用于激活函数的传统字符串支持。
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # 在TorchScript中无法测试self.activation是否在forward()中，
        # 因此存储一些关于它的信息。
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        # 调用父类的__setstate__函数
        super().__setstate__(state)
        # 如果当前对象没有activation属性，则将其设置为F.relu
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ):
        # self-attention block
        def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
        ) -> Tensor:
            # 使用self-attention机制处理输入x
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
            # 对处理后的结果应用dropout操作
            return self.dropout1(x)

        # feed forward block
        def _ff_block(self, x: Tensor) -> Tensor:
            # 前馈神经网络块的实现
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            # 对处理后的结果应用dropout操作
            return self.dropout2(x)
# TransformerDecoderLayer 类定义，表示一个 Transformer 解码器的层次结构
class TransformerDecoderLayer(Module):
    # TransformerDecoderLayer 类的文档字符串，描述了其结构和基于论文 "Attention Is All You Need" 的设计原理
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    # 类的常量声明
    __constants__ = ["norm_first"]

    # 初始化方法
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        # 调用父类的初始化方法
        super(TransformerDecoderLayer, self).__init__()
        # 设置类的属性
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first

        self._reset_parameters()

    # 重置参数的私有方法
    def _reset_parameters(self):
        # 初始化线性层的参数
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    # 前向传播方法
    def forward(self, tgt: Tensor, memory: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # 如果 batch_first 为 False，则交换 tgt 和 memory 的维度
        if not self.batch_first:
            tgt, memory = tgt.transpose(0, 1), memory.transpose(0, 1)

        # 根据 norm_first 的值选择是否在 self attention 之前应用 LayerNorm
        if self.norm_first:
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                                     key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
        else:
            tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                     key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(self.norm1(tgt2))

        # 应用 LayerNorm 和 Multihead Attention
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)

        # 应用 LayerNorm、Feedforward 和激活函数
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # 如果 batch_first 为 False，则将 tgt 的维度还原
        if not self.batch_first:
            tgt = tgt.transpose(0, 1)
        return tgt
    ) -> None:
        # 定义用于实例化子模块的参数字典
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类构造函数初始化对象
        super().__init__()
        # 创建 self-attention 模块
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # 创建 multihead attention 模块
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # 创建第一个线性层，用于 feedforward 模型
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        # 创建 dropout 模块
        self.dropout = Dropout(dropout)
        # 创建第二个线性层，用于 feedforward 模型
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        # 设置是否在层归一化操作中先进行归一化
        self.norm_first = norm_first
        # 创建第一个归一化层
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # 创建第二个归一化层
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # 创建第三个归一化层
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # 创建第一个 dropout 模块
        self.dropout1 = Dropout(dropout)
        # 创建第二个 dropout 模块
        self.dropout2 = Dropout(dropout)
        # 创建第三个 dropout 模块
        self.dropout3 = Dropout(dropout)

        # 处理激活函数的遗留字符串支持
        if isinstance(activation, str):
            # 获取对应的激活函数
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        # 若状态中未包含激活函数，则默认为 ReLU
        if "activation" not in state:
            state["activation"] = F.relu
        # 调用父类的 __setstate__ 方法
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        # 如果设置了 norm_first 标志，先对输入应用 LayerNorm，并传递给 self-attention block 和 multihead attention block
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            # 否则，先传递给 self-attention block，然后应用 LayerNorm，并传递给 multihead attention block 和 feed-forward block
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        # 调用 self-attention 层进行计算，传入 x 作为 query、key、value，并应用对应的掩码
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        # 对输出应用 dropout1
        return self.dropout1(x)

    # multihead attention block
    # 定义多头注意力机制块函数，处理输入 x 和记忆 mem，并可选地使用注意力掩码和键填充掩码
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        # 调用多头注意力机制模块，处理输入 x 和记忆 mem，返回处理后的输出
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        # 对输出结果进行 dropout 处理
        return self.dropout2(x)

    # 前馈神经网络块
    def _ff_block(self, x: Tensor) -> Tensor:
        # 前馈神经网络块的实现：先经过第一层线性变换、激活函数、dropout，再经过第二层线性变换
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # 对输出结果进行 dropout 处理
        return self.dropout3(x)
# 创建一个函数，用于复制给定模块 N 次，并返回一个包含这些复制模块的 ModuleList 对象
def _get_clones(module, N):
    # FIXME: copy.deepcopy() 在 nn.module 上未定义
    return ModuleList([copy.deepcopy(module) for i in range(N)])


# 创建一个函数，根据激活函数名称返回对应的激活函数
def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    # 如果激活函数名称不是 'relu' 或 'gelu'，抛出运行时错误
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


# 创建一个函数，判断给定的注意力掩码是否是因果的
def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # 如果 is_causal 是 True，则认为是因果的
    make_causal = is_causal is True

    # 如果 is_causal 是 None 并且 mask 不为 None，则进行进一步判断
    if is_causal is None and mask is not None:
        # 如果 size 不为 None，则使用指定大小生成一个方形的因果 mask
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # 检查 mask 是否与生成的因果 mask 相等
        # 不使用 torch.equal 以处理批处理 mask 的情况，通过广播进行比较
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal
```