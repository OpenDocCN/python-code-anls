# `.\models\marian\modeling_marian.py`

```
# coding=utf-8
# 版权 2021 年 Marian Team 作者和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“现状”提供软件
# 分发，无论是明示的还是暗示的，但是没有任何担保或条件
# 特定用途的适用性，包括但不限于对适销性和特定用途的适用性的隐含保证。
"""从 Marian C++ 仓库移植的 PyTorch MarianMTModel 模型。"""


import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_marian import MarianConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MarianConfig"
_CHECKPOINT_FOR_DOC = "Helsinki-NLP/opus-mt-en-de"


MARIAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Helsinki-NLP/opus-mt-en-de",
    # 查看所有 Marian 模型请访问 https://huggingface.co/models?filter=marian
]


# 从 transformers.models.bart.modeling_bart.shift_tokens_right 复制而来
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的 token 向右移动一位。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id 必须被定义。")
    # 将标签中可能存在的 -100 值替换为 `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class MarianSinusoidalPositionalEmbedding(nn.Embedding):
    """此模块生成任意长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        初始化权重矩阵，类似于 XLM 的 create_sinusoidal_embeddings 函数，但特征没有交错。
        余弦特征位于向量的后半部分。[dim // 2:]
        """
        n_pos, dim = out.shape
        # 创建位置编码矩阵，使用 numpy 数组生成，用于 Transformer 的位置编码
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 设置 requires_grad 属性为 False，以避免在 PyTorch 1.8+ 版本中出现错误
        out.requires_grad = False
        # 计算中间分隔点位置，处理偶数和奇数维度的情况
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 将正弦值填充到权重矩阵的前半部分
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        # 将余弦值填充到权重矩阵的后半部分
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        # 将权重矩阵从计算图中分离出来，不再进行梯度计算
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """
        前向传播函数，用于计算位置编码的张量。

        `input_ids_shape` 应该是 [bsz x seqlen] 的形状。
        """
        bsz, seq_len = input_ids_shape[:2]
        # 根据序列长度和过去键值对长度计算位置编码的位置张量
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的 forward 方法，返回位置编码的张量
        return super().forward(positions)
# Copied from transformers.models.marian.modeling_marian.MarianAttention with Marian->Marian
class MarianAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[MarianConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设置注意力机制的嵌入维度
        self.num_heads = num_heads  # 设置注意力头的数量
        self.dropout = dropout  # 设置dropout比率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        self.config = config  # 存储Marian配置信息

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于注意力权重计算
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否为因果注意力

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 创建键的投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 创建值的投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 创建查询的投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 创建输出的投影层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 前向传播函数，实现注意力机制的计算
        ...



# Copied from transformers.models.marian.modeling_marian.MarianEncoderLayer with Marian->Marian, BART->MARIAN
class MarianEncoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 设置编码器层的嵌入维度

        self.self_attn = MARIAN_ATTENTION_CLASSES[config._attn_implementation](  # 创建自注意力层
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 创建自注意力层的LayerNorm层
        self.dropout = config.dropout  # 设置dropout比率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout比率
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终的LayerNorm层

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ):
        # 前向传播函数，实现编码器层的计算
        ...
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states  # 保存原始输入以便后续的残差连接
        hidden_states, attn_weights, _ = self.self_attn(  # 使用自注意力机制进行计算
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 在计算后的隐藏状态上应用 dropout
        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 对残差连接后的隐藏状态进行 layer normalization

        residual = hidden_states  # 保存当前层处理后的结果以便后续的残差连接
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 使用激活函数处理线性转换
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 应用 dropout
        hidden_states = self.fc2(hidden_states)  # 进行第二个线性转换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 再次应用 dropout
        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.final_layer_norm(hidden_states)  # 最终的 layer normalization

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)  # 处理潜在的浮点数溢出或 NaN 值

        outputs = (hidden_states,)  # 输出最终的隐藏状态作为主要结果

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则将其添加到输出中

        return outputs  # 返回输出结果
# 定义一个字典，将字符串 "eager" 映射到类 MarianAttention
MARIAN_ATTENTION_CLASSES = {"eager": MarianAttention}


# 从 transformers.models.bart.modeling_bart.BartDecoderLayer 复制而来，将 Bart 替换为 Marian，BART 替换为 MARIAN
class MarianDecoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        # 设置嵌入维度为配置中的 d_model
        self.embed_dim = config.d_model

        # 初始化自注意力机制，根据配置选择不同的实现类
        self.self_attn = MARIAN_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )

        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置激活函数为配置中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的 dropout 概率
        self.activation_dropout = config.activation_dropout

        # 初始化自注意力机制的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化编码器注意力机制，根据配置选择不同的实现类
        self.encoder_attn = MARIAN_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 初始化编码器注意力机制的 LayerNorm
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 定义第一个线性层，将嵌入维度映射到配置中指定的解码器前馈神经网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 定义第二个线性层，将解码器前馈神经网络维度映射回嵌入维度
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        # 初始化最终层的 LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        ):
        # 此处需要补充具体的前向传播逻辑，但根据代码结构，主要负责模型层的连接与数据流动
        pass


# 定义 MarianPreTrainedModel 类，继承自 PreTrainedModel
class MarianPreTrainedModel(PreTrainedModel):
    # 设置配置类为 MarianConfig
    config_class = MarianConfig
    # 设置基础模型前缀为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Embedding, MarianSinusoidalPositionalEmbedding]):
        # 初始化权重函数，根据模块类型选择不同的初始化方法
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            # 对线性层进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                # 如果有偏置，则将偏置初始化为零
                module.bias.data.zero_()
        elif isinstance(module, MarianSinusoidalPositionalEmbedding):
            # 对 MarianSinusoidalPositionalEmbedding 类型不进行任何操作
            pass
        elif isinstance(module, nn.Embedding):
            # 对嵌入层进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # 如果有填充索引，则将填充索引的权重初始化为零
                module.weight.data[module.padding_idx].zero_()

    @property
    # 定义一个方法用于生成虚拟输入数据
    def dummy_inputs(self):
        # 获取配置中的填充标记 ID
        pad_token = self.config.pad_token_id
        # 创建包含两个示例序列的张量，设备为当前对象的设备
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典，包括注意力掩码、输入序列和解码器输入序列
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 使用填充标记 ID 生成注意力掩码
            "input_ids": input_ids,  # 将输入序列添加到字典中
            "decoder_input_ids": input_ids,  # 将解码器输入序列添加到字典中，与输入序列相同
        }
        # 返回生成的虚拟输入字典
        return dummy_inputs
MARIAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MarianConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MARIAN_GENERATION_EXAMPLE = r"""
    Pytorch version of marian-nmt's transformer.h (c++). Designed for the OPUS-NMT translation checkpoints. Available
    models are listed [here](https://huggingface.co/models?search=Helsinki-NLP).

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, MarianMTModel

    >>> src = "fr"  # source language
    >>> trg = "en"  # target language

    >>> model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    >>> model = MarianMTModel.from_pretrained(model_name)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)

    >>> sample_text = "où est l'arrêt de bus ?"
    >>> batch = tokenizer([sample_text], return_tensors="pt")

    >>> generated_ids = model.generate(**batch)
    >>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    "Where's the bus stop?"
    ```
"""

MARIAN_INPUTS_DOCSTRING = r"""
"""


class MarianEncoder(MarianPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MarianEncoderLayer`].

    Args:
        config: MarianConfig
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        embed_tokens (nn.Embedding): output embedding
            A PyTorch embedding layer representing the output embeddings of the model.
    """
    # 初始化函数，接受一个配置对象 config 和一个可选的嵌入词向量 embed_tokens
    def __init__(self, config: MarianConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置 dropout 概率为配置中的值
        self.dropout = config.dropout
        # 设置 encoder 层级的 dropout 概率为配置中的值
        self.layerdrop = config.encoder_layerdrop

        # 从配置中获取词嵌入的维度
        embed_dim = config.d_model
        # 获取填充索引
        self.padding_idx = config.pad_token_id
        # 获取最大源序列位置
        self.max_source_positions = config.max_position_embeddings
        # 如果配置中设置了缩放嵌入，则设置嵌入缩放因子为 sqrt(embed_dim)，否则为 1.0
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 如果传入了 embed_tokens，则使用传入的词嵌入，否则创建一个新的 nn.Embedding 对象
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 使用 MarianSinusoidalPositionalEmbedding 类创建位置嵌入对象，设置最大位置和嵌入维度
        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings, embed_dim, self.padding_idx
        )
        
        # 创建一个包含多个 MarianEncoderLayer 的列表，数量为配置中指定的编码器层数
        self.layers = nn.ModuleList([MarianEncoderLayer(config) for _ in range(config.encoder_layers)])

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

        # 调用初始化后的后处理方法
        self.post_init()

    # 返回当前模型的词嵌入对象
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置当前模型的词嵌入对象为给定的值
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数，接受多个参数，处理输入序列以生成输出
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的词 id 序列，默认为 None
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码，默认为 None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，默认为 None
class MarianDecoder(MarianPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MarianDecoderLayer`]

    Args:
        config: MarianConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout                  # 初始化 dropout 概率
        self.layerdrop = config.decoder_layerdrop      # 初始化层间 dropout 概率
        self.padding_idx = config.pad_token_id         # 初始化填充 token 的索引
        self.max_target_positions = config.max_position_embeddings  # 最大目标位置
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 嵌入缩放系数

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens          # 如果提供了嵌入 tokens，则使用提供的
        else:
            self.embed_tokens = nn.Embedding(config.decoder_vocab_size, config.d_model, self.padding_idx)
                                                    # 否则创建一个新的嵌入 tokens 对象

        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.d_model, self.padding_idx
        )                                           # 使用正弦位置编码初始化位置嵌入对象

        self.layers = nn.ModuleList([MarianDecoderLayer(config) for _ in range(config.decoder_layers)])
                                                    # 创建多层解码器层列表

        self.gradient_checkpointing = False          # 梯度检查点，默认为 False

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens                     # 返回输入嵌入 tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value                    # 设置输入嵌入 tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass for the MarianDecoder module.

        Args:
            input_ids (torch.LongTensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask for masking out padded tokens
            encoder_hidden_states (torch.FloatTensor): Hidden states from the encoder
            encoder_attention_mask (torch.LongTensor): Attention mask for encoder's hidden states
            head_mask (torch.Tensor): Mask for heads in the self-attention layers
            cross_attn_head_mask (torch.Tensor): Mask for heads in the cross-attention layers
            past_key_values (Tuple[Tuple[torch.FloatTensor]]): Cached key-value pairs for fast decoding
            inputs_embeds (torch.FloatTensor): Optional tensor of embedded inputs
            use_cache (bool): Whether to use cached key-value pairs
            output_attentions (bool): Whether to output attentions
            output_hidden_states (bool): Whether to output hidden states
            return_dict (bool): Whether to return a dictionary as output

        Returns:
            Various outputs depending on the configuration (return_dict or not)
        """
        # Forward pass logic will be implemented here in subsequent code
    def __init__(self, config: MarianConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size

        # We always use self.shared for token embeddings to ensure compatibility with all marian models
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        if self.config.share_encoder_decoder_embeddings:
            # If embeddings are shared between encoder and decoder, use the same instance
            encoder_embed_tokens = decoder_embed_tokens = self.shared
        else:
            # If embeddings are not shared, create separate instances for encoder and decoder
            # to ensure they are not tied.
            encoder_embed_tokens = copy.deepcopy(self.shared)
            decoder_embed_tokens = copy.deepcopy(self.shared)
            self.shared = None  # Reset self.shared to None for separate embeddings

        # Initialize encoder and decoder with respective embeddings
        self.encoder = MarianEncoder(config, encoder_embed_tokens)
        self.decoder = MarianDecoder(config, decoder_embed_tokens)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # Returns the shared embeddings if they are shared, otherwise returns encoder embeddings
        return self.get_encoder().get_input_embeddings()

    def set_input_embeddings(self, value):
        if self.config.share_encoder_decoder_embeddings:
            # If embeddings are shared, set the shared instance for both encoder and decoder
            self.shared = value
            self.encoder.embed_tokens = self.shared
            self.decoder.embed_tokens = self.shared
        else:
            # If embeddings are not shared, only set encoder embeddings
            self.encoder.embed_tokens = value

    def get_decoder_input_embeddings(self):
        if self.config.share_encoder_decoder_embeddings:
            # Error if decoder embeddings are accessed when they are shared with encoder
            raise ValueError(
                "`get_decoder_input_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
                "is `True`. Please use `get_input_embeddings` instead."
            )
        # Return decoder embeddings (should not be reached if embeddings are shared)
        return self.get_decoder().get_input_embeddings()

    def set_decoder_input_embeddings(self, value):
        if self.config.share_encoder_decoder_embeddings:
            # Error if trying to set decoder embeddings when they are shared with encoder
            raise ValueError(
                "`config.share_encoder_decoder_embeddings` is set to `True` meaning the decoder input embeddings "
                "are shared with the encoder. In order to set the decoder input embeddings, you should simply set "
                "the encoder input embeddings by calling `set_input_embeddings` with the appropriate embeddings."
            )
        # Set decoder embeddings (should not be reached if embeddings are shared)
        self.decoder.embed_tokens = value

    def get_encoder(self):
        # Returns the encoder instance
        return self.encoder

    def get_decoder(self):
        # Returns the decoder instance
        return self.decoder
    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    # 使用指定的文档字符串装饰器来添加输入参数的描述信息，此处为Marian模型的输入说明文档字符串
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    # 使用指定的文档字符串装饰器来替换返回值的描述信息，指定输出类型为Seq2SeqModelOutput，并使用_CONFIG_FOR_DOC类的配置信息

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        # 输入序列的token IDs，数据类型为torch的LongTensor，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 注意力掩码，用于指示模型在哪些位置上需要注意力，数据类型为可选的torch.Tensor，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的输入token IDs，数据类型为可选的torch.LongTensor，默认为None
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # 解码器的注意力掩码，数据类型为可选的torch.Tensor，默认为None
        head_mask: Optional[torch.Tensor] = None,
        # 头部掩码，用于控制层间的连接，数据类型为可选的torch.Tensor，默认为None
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，用于控制解码器层间的连接，数据类型为可选的torch.Tensor，默认为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 跨注意力头部掩码，用于跨模块（encoder-decoder）的头部连接，数据类型为可选的torch.Tensor，默认为None
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        # 编码器的输出，数据类型为可选的Union类型（包括元组或BaseModelOutput），默认为None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 过去的键值对，用于缓存解码器的过去状态信息，数据类型为可选的元组类型（包含元组的torch.FloatTensor），默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 输入嵌入向量，数据类型为可选的torch.FloatTensor，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器输入嵌入向量，数据类型为可选的torch.FloatTensor，默认为None
        use_cache: Optional[bool] = None,
        # 是否使用缓存，用于控制是否缓存中间状态以加快推理速度，数据类型为可选的布尔值，默认为None
        output_attentions: Optional[bool] = None,
        # 是否输出注意力权重，数据类型为可选的布尔值，默认为None
        output_hidden_states: Optional[bool] = None,
        # 是否输出隐藏状态，数据类型为可选的布尔值，默认为None
        return_dict: Optional[bool] = None,
        # 是否返回字典格式的输出，数据类型为可选的布尔值，默认为None
@add_start_docstrings(
    "The Marian Model with a language modeling head. Can be used for summarization.", MARIAN_START_DOCSTRING
)
# 定义了一个继承自MarianPreTrainedModel的类MarianMTModel，用于Marian模型的语言建模任务和摘要生成任务
class MarianMTModel(MarianPreTrainedModel):
    # 指定基础模型的前缀
    base_model_prefix = "model"
    # 在加载过程中忽略的键列表
    _keys_to_ignore_on_load_missing = [
        "final_logits_bias",
        "encoder.embed_positions.weight",
        "decoder.embed_positions.weight",
    ]
    # 在保存过程中忽略的键列表
    _keys_to_ignore_on_save = ["model.encoder.embed_positions.weight", "model.decoder.embed_positions.weight"]
    # 共享权重的键列表
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化函数，接受一个MarianConfig类型的配置对象
    def __init__(self, config: MarianConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个MarianModel类型的模型实例
        self.model = MarianModel(config)

        # 根据配置决定目标词汇表大小，用于创建final_logits_bias缓冲区
        target_vocab_size = config.vocab_size if config.share_encoder_decoder_embeddings else config.decoder_vocab_size
        # 注册一个缓冲区final_logits_bias，全零初始化，形状为(1, target_vocab_size)
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        # 创建一个线性层lm_head，用于生成模型的最终输出，输入维度为config.d_model，输出维度为target_vocab_size，无偏置
        self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)

        # 执行初始化权重和应用最终处理的函数
        self.post_init()

    # 获取编码器的方法
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器的方法
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整token嵌入大小的方法
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类方法resize_token_embeddings进行token嵌入大小的调整
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 如果共享编码器和解码器嵌入，则调整final_logits_bias的大小
        if self.config.share_encoder_decoder_embeddings:
            self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # 内部方法，用于调整token嵌入大小
    def _resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of=None) -> nn.Embedding:
        # 获取当前输入嵌入
        old_embeddings = self.get_input_embeddings()
        # 调用内部方法_get_resized_embeddings进行嵌入的调整
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        # 设置调整后的输入嵌入
        self.set_input_embeddings(new_embeddings)

        # 更新config.decoder_vocab_size，如果嵌入被绑定
        new_num_tokens = new_embeddings.weight.shape[0]
        if self.config.share_encoder_decoder_embeddings:
            self.config.decoder_vocab_size = new_num_tokens

        # 如果单词嵌入未绑定，则确保lm head也被调整大小
        if (
            self.config.share_encoder_decoder_embeddings
            and self.get_output_embeddings() is not None
            and not self.config.tie_word_embeddings
        ):
            # 获取当前输出嵌入
            old_lm_head = self.get_output_embeddings()
            # 调用内部方法_get_resized_lm_head进行lm head的调整
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            # 设置调整后的输出嵌入（lm head）
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()
    # 调整解码器的 token embeddings 大小
    def resize_decoder_token_embeddings(self, new_num_tokens):
        # 如果配置中指定共享编码器和解码器的 embeddings，则抛出数值错误
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError(
                "`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
                "is `True`. Please use `resize_token_embeddings` instead."
            )

        # 获取当前解码器的输入 embeddings
        old_embeddings = self.model.get_decoder_input_embeddings()
        # 根据新的 token 数量调整 embeddings 大小
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        # 设置调整后的 embeddings 到模型中
        self.model.set_decoder_input_embeddings(new_embeddings)

        # 如果输出 embeddings 存在且不与输入 embeddings 绑定，确保语言模型头也被调整大小
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            # 获取当前语言模型头
            old_lm_head = self.get_output_embeddings()
            # 根据新的 token 数量调整语言模型头的大小
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            # 设置调整后的语言模型头
            self.set_output_embeddings(new_lm_head)

        # 获取调整后的解码器输入 embeddings
        model_embeds = self.model.get_decoder_input_embeddings()

        # 如果新的 token 数量为 None，则返回当前的 embeddings
        if new_num_tokens is None:
            return model_embeds

        # 更新基础模型和当前模型配置中的解码器词汇表大小
        self.config.decoder_vocab_size = new_num_tokens

        # 如果需要，重新绑定权重
        self.tie_weights()

        # 调整最终 logits 偏置
        self._resize_final_logits_bias(new_num_tokens)

        # 返回调整后的解码器输入 embeddings
        return model_embeds

    # 调整最终 logits 的偏置
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取当前 logits 偏置的旧 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于旧的 token 数量，则截取当前偏置
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 否则，创建额外的偏置，并将其与当前偏置连接起来
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 将调整后的偏置注册为模型的缓冲区
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出 embeddings（语言模型头）
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出 embeddings（语言模型头）
    def set_output_embeddings(self, new_embeddings: nn.Embedding):
        self.lm_head = new_embeddings
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        # 获取输出嵌入层
        output_embeddings = self.get_output_embeddings()
        # 检查是否存在输出嵌入层，并且是否允许共享参数
        if output_embeddings is not None and getattr(self.config, "tie_word_embeddings", True):
            # 获取解码器的输入嵌入层（如果嵌入层被共享，返回共享的嵌入层；否则返回解码器的embed_tokens）
            word_embeddings = self.get_decoder().get_input_embeddings()
            # 调用函数来共享或克隆权重
            self._tie_or_clone_weights(output_embeddings, word_embeddings)

        # 如果模型是编码-解码结构并且配置允许编码器和解码器共享权重
        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            # 如果对象有基础模型前缀，则将当前实例设置为基础模型的实例
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            # 调用函数来共享编码器和解码器的权重
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        # 遍历模型的所有模块
        for module in self.modules():
            # 如果模块有 `_tie_weights` 方法，则调用该方法
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MARIAN_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Seq2SeqLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            `Seq2SeqLMOutput`: A class representing the outputs of the Seq2Seq language model.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If labels are provided, adjust `use_cache` and prepare `decoder_input_ids` if necessary
        if labels is not None:
            # Issue a warning and set `use_cache` to False if `labels` are provided
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            
            # If `decoder_input_ids` and `decoder_inputs_embeds` are not provided, prepare `decoder_input_ids`
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # Shift the labels to the right to align with decoder inputs
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Pass the inputs to the underlying model for computation
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Compute the logits from the language model head and add bias
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # Compute the masked language modeling loss if `labels` are provided
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))

        # Prepare the output based on `return_dict` setting
        if not return_dict:
            # Return a tuple with logits and additional outputs if `return_dict` is False
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return a `Seq2SeqLMOutput` object with specified attributes if `return_dict` is True
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        **kwargs,
    ) -> Dict:
        """
        Prepare inputs for text generation.

        Args:
            decoder_input_ids: Input IDs for decoder.
            past_key_values: Tuple of past key and value tensors.
            attention_mask: Mask to avoid attention on padding tokens.
            head_mask: Mask to nullify selected heads of the attention modules.
            decoder_head_mask: Mask to nullify selected heads of the decoder self-attention modules.
            cross_attn_head_mask: Mask to nullify selected heads of the cross-attention modules.
            use_cache: Flag to control whether to use caching.
            encoder_outputs: Output tensors from the encoder.

        Returns:
            Dictionary containing prepared inputs.
        """

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        """
        Shift labels to the right to prep inputs for decoder.

        Args:
            labels: Tensor of labels.

        Returns:
            Tensor of shifted labels.
        """
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder past key and value tensors based on beam index.

        Args:
            past_key_values: Tuple of past key and value tensors.
            beam_idx: Tensor containing indices to reorder with.

        Returns:
            Reordered tuple of past key and value tensors.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 从 transformers.models.bart.modeling_bart.BartDecoderWrapper 复制并将 Bart 改为 Marian 的类定义
class MarianDecoderWrapper(MarianPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在使用因果语言模型与 EncoderDecoderModel 框架组合时正确加载预训练的检查点。
    """

    def __init__(self, config):
        # 调用父类的构造函数，传入配置信息
        super().__init__(config)
        # 初始化 MarianDecoder 实例
        self.decoder = MarianDecoder(config)

    def forward(self, *args, **kwargs):
        # 将前向计算委托给 self.decoder 对象
        return self.decoder(*args, **kwargs)


# 从 transformers.models.bart.modeling_bart.BartForCausalLM 复制并将 Bart 改为 Marian，facebook/bart-base->Helsinki-NLP/opus-mt-fr-en
class MarianForCausalLM(MarianPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深度复制配置信息
        config = copy.deepcopy(config)
        # 标记为解码器
        config.is_decoder = True
        # 标记为非编码-解码器结构
        config.is_encoder_decoder = False
        # 调用父类的构造函数，传入配置信息
        super().__init__(config)
        # 初始化 MarianDecoderWrapper 实例
        self.model = MarianDecoderWrapper(config)

        # 初始化线性层，连接解码器隐藏状态到词汇表大小，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型的输入嵌入层
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置模型的输入嵌入层
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回模型的输出嵌入层（lm_head）
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置模型的输出嵌入层（lm_head）
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置模型的解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 返回模型的解码器
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播函数，接受多个输入参数并返回结果
        ...

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 为生成准备输入数据的函数，接受多个输入参数并返回结果
        ...
    ):
        # 如果模型用作编码器-解码器模型中的解码器，解码器注意力掩码是即时创建的
        if attention_mask is None:
            # 如果注意力掩码为空，则创建一个形状与输入相同的全1张量作为注意力掩码
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去键值对中第一个层的过去长度
            past_length = past_key_values[0][0].shape[2]

            # 有些生成方法已经只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                # 如果输入的长度大于过去长度，则计算要移除的前缀长度
                remove_prefix_length = past_length
            else:
                # 否则，默认行为是保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 截取输入序列，保留从 remove_prefix_length 开始到末尾的部分
            input_ids = input_ids[:, remove_prefix_length:]
        # 第一步，decoder_cached_states 是空的
        return {
            "input_ids": input_ids,  # encoder_outputs 已定义。不再需要 input_ids
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 重新排序过去的缓存，根据 beam_idx 对每一层的过去状态进行索引选择
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```