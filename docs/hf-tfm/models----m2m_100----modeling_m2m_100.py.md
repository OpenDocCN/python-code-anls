# `.\models\m2m_100\modeling_m2m_100.py`

```py
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch M2M100 model."""

import math  # 导入数学库
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数相关的模块
from ...integrations.deepspeed import is_deepspeed_zero3_enabled  # 导入DeepSpeed相关的模块
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入处理注意力掩码相关的函数
from ...modeling_outputs import (  # 导入模型输出相关的类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型相关的工具函数
from ...utils import (  # 导入工具函数
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_m2m_100 import M2M100Config  # 导入M2M100模型的配置

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "M2M100Config"  # 文档中使用的配置名称
_CHECKPOINT_FOR_DOC = "facebook/m2m100_418M"  # 文档中使用的检查点名称

M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST = [  # M2M100预训练模型的存档列表
    "facebook/m2m100_418M",
    # 查看所有M2M100模型 https://huggingface.co/models?filter=m2m_100
]


# 从transformers.models.bart.modeling_bart.shift_tokens_right复制而来
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的token向右移动一位。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)  # 创建一个与输入形状相同的零张量
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # 将输入的除了第一个位置的所有token向右移动一位
    shifted_input_ids[:, 0] = decoder_start_token_id  # 将第一个位置设置为decoder的起始token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 用pad_token_id替换标签中可能存在的-100值
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    根据输入的input_ids生成位置id，非padding符号用它们的位置数字表示。位置数字从padding_idx+1开始，padding符号被忽略。
    这是从fairseq的`utils.make_positions`修改而来的。
    """
    # 这里的类型转换和转换非常平衡，既适用于ONNX导出，也适用于XLA。
    mask = input_ids.ne(padding_idx).int()  # 创建一个掩码，指示哪些位置是非padding的
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask  # 生成递增的位置id，并应用掩码
    # 返回一个张量，其中包含 incremental_indices 张量的长整型值与 padding_idx 的加法结果
    return incremental_indices.long() + padding_idx
class M2M100SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2  # 定义偏移量为2，用于创建位置嵌入
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.padding_idx = padding_idx  # 可选的填充索引
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)  # 调用make_weights方法创建权重

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)  # 调用get_embedding方法获取嵌入权重
        if hasattr(self, "weights"):
            # 在forward方法中，将权重转换为参数的正确dtype和device
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.register_buffer("weights", emb_weights, persistent=False)  # 将权重注册为缓冲区，非持久性注册

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2  # 嵌入维度的一半
        emb = math.log(10000) / (half_dim - 1)  # 计算基于半嵌入维度的对数间隔
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)  # 计算指数衰减的正弦周期
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)  # 创建位置嵌入
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)  # 组合sin和cos，形成嵌入
        if embedding_dim % 2 == 1:
            # 若嵌入维度为奇数，进行零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0  # 将填充索引位置的嵌入置零

        return emb.to(torch.get_default_dtype())  # 返回默认dtype的嵌入张量

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # 从输入的token ids创建位置ids，任何填充的token保持填充状态
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # 如果需要扩展嵌入
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 选择对应位置ids的嵌入并返回，同时进行分离计算图
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    # 根据输入的嵌入向量生成位置编码标识符
    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor  # 输入的嵌入向量，形状为 [batch_size, sequence_length, embedding_size]

        Returns: torch.Tensor  # 返回形状与输入相同的位置编码标识符张量
        """
        input_shape = inputs_embeds.size()[:-1]  # 获取输入张量的形状，不包括最后一维（通常是嵌入维度）
        sequence_length = input_shape[1]  # 获取序列长度，即第二个维度的大小

        # 生成从 self.padding_idx + 1 到 sequence_length + self.padding_idx + 1 的序列，作为位置编码标识符
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置编码标识符张量进行扩展，使其形状与输入张量相同，并确保内存布局连续
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->M2M100
class M2M100Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[M2M100Config] = None,
    ):
        super().__init__()
        # 初始化函数，设置注意力模型的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 确保 embed_dim 必须被 num_heads 整除，否则抛出错误
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性变换层，用于计算查询、键、值和输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入张量重塑成期望的形状，用于多头注意力的计算
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
        # 前向传播函数，执行注意力计算和线性变换
        # hidden_states: 输入的隐藏状态张量
        # key_value_states: 键值对状态张量，可选
        # past_key_value: 过去的键值对张量，可选
        # attention_mask: 注意力掩码张量，可选
        # layer_head_mask: 层头掩码张量，可选
        # output_attentions: 是否输出注意力张量，布尔值

# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->M2M100, MBART->M2M100
class M2M100EncoderLayer(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()
        # 初始化函数，设置编码器层的参数
        self.embed_dim = config.d_model

        # 自注意力层
        self.self_attn = M2M100_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 前馈神经网络的两个线性层
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)

        # 最终层的 LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        # 前向传播函数，执行编码器层的计算
        # hidden_states: 输入的隐藏状态张量
        # attention_mask: 注意力掩码张量
        # layer_head_mask: 层头掩码张量
        # output_attentions: 是否输出注意力张量，布尔值
    ) -> torch.Tensor:
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
        # 记录输入的原始状态，用于残差连接
        residual = hidden_states
        # 对输入的 hidden_states 进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用 self-attention 层处理输入，返回处理后的 hidden_states、注意力权重 attn_weights，以及可能的 attentions
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对 self-attention 处理后的 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与处理后的 hidden_states 相加，实现残差连接
        hidden_states = residual + hidden_states

        # 记录输入的原始状态，用于残差连接
        residual = hidden_states
        # 对处理后的 hidden_states 再进行 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 经过第一个全连接层 fc1，并使用激活函数 activation_fn
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对 fc1 输出的 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 经过第二个全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 对 fc2 输出的 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与处理后的 hidden_states 相加，实现残差连接
        hidden_states = residual + hidden_states

        # 如果 hidden_states 的数据类型为 torch.float16，并且包含无穷大或 NaN 值，则进行值的修正
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构造输出元组，包含处理后的 hidden_states
        outputs = (hidden_states,)

        # 如果需要返回 attentions tensors，则将 attn_weights 加入输出元组
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出元组
        return outputs
# 定义一个字典，映射关系为字符串"eager"到类M2M100Attention
M2M100_ATTENTION_CLASSES = {"eager": M2M100Attention}

# 从transformers.models.mbart.modeling_mbart.MBartDecoderLayer复制并修改为使用M2M100，替换MBart为M2M100
class M2M100DecoderLayer(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()
        self.embed_dim = config.d_model

        # 初始化自注意力层，使用配置中指定的注意力实现方法
        self.self_attn = M2M100_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数根据配置选择
        self.activation_dropout = config.activation_dropout

        # 对自注意力层的输出进行LayerNorm归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化编码器注意力层，使用配置中指定的注意力实现方法
        self.encoder_attn = M2M100_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 对编码器注意力层的输出进行LayerNorm归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 第一个全连接层，线性变换到decoder_ffn_dim维度
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)

        # 第二个全连接层，线性变换回embed_dim维度
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        # 最终输出层的LayerNorm归一化
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
    pass

# M2M100PreTrainedModel继承自PreTrainedModel，设置相关类属性和方法
class M2M100PreTrainedModel(PreTrainedModel):
    config_class = M2M100Config  # 指定配置类为M2M100Config
    base_model_prefix = "model"  # 基础模型前缀为"model"
    supports_gradient_checkpointing = True  # 支持梯度检查点

    # 不需要分割的模块名称列表，排除"M2M100Attention"
    _no_split_modules = ["M2M100Attention"]

    # 初始化模型权重
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# M2M_100_START_DOCSTRING是一个字符串，包含了关于M2M100PreTrainedModel的文档字符串模板
M2M_100_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
"""
    # 作为普通的 PyTorch 模块使用，并参考 PyTorch 文档以获取有关一般使用和行为的所有信息。

    Parameters:
        config ([`M2M100Config`]):
            模型配置类，包含模型的所有参数。使用配置文件初始化时不会加载与模型相关的权重，只加载配置信息。
            可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型的权重。
"""

M2M_100_GENERATION_EXAMPLE = r"""
    Translation example:

    ```
    >>> from transformers import AutoTokenizer, M2M100ForConditionalGeneration

    >>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

    >>> text_to_translate = "Life is like a box of chocolates"
    >>> model_inputs = tokenizer(text_to_translate, return_tensors="pt")

    >>> # translate to French
    >>> gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("fr"))
    >>> print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
    ```
"""

M2M_100_INPUTS_DOCSTRING = r"""
"""


class M2M100Encoder(M2M100PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`M2M100EncoderLayer`].

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # Embedding layer for tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # Positional embedding for token positions
        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )

        # List of encoder layers
        self.layers = nn.ModuleList([M2M100EncoderLayer(config) for _ in range(config.encoder_layers)])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Gradient checkpointing disabled by default
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Implementation of forward pass for encoder
        pass


class M2M100Decoder(M2M100PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`M2M100DecoderLayer`]

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化方法，接收配置参数和可选的嵌入层
    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法，传递配置参数
        super().__init__(config)
        
        # 设置对象的属性，从配置中获取各种参数
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 创建嵌入层对象，vocab_size表示词汇表大小，d_model表示嵌入维度，padding_idx表示填充标识
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 如果提供了外部的嵌入层，将其权重复制到当前的嵌入层
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 创建位置编码对象，使用正弦函数生成位置编码
        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )

        # 创建解码器层的列表，每个解码器层具有相同的配置参数
        self.layers = nn.ModuleList([M2M100DecoderLayer(config) for _ in range(config.decoder_layers)])
        
        # 创建层归一化对象，对隐藏层进行归一化处理
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 是否启用梯度检查点，初始化为False
        self.gradient_checkpointing = False

        # 执行初始化后的处理操作，可能包括权重初始化和其他的后续处理
        self.post_init()

    # 前向传播方法，接收多个输入参数，实现模型的数据流向
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器添加文档字符串到模型类，描述该类的基本信息和用途
@add_start_docstrings(
    "The bare M2M100 Model outputting raw hidden-states without any specific head on top.",
    M2M_100_START_DOCSTRING,
)
# 定义 M2M100Model 类，继承自 M2M100PreTrainedModel 类
class M2M100Model(M2M100PreTrainedModel):
    # 定义用于共享权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化函数，接受一个 M2M100Config 对象作为参数
    def __init__(self, config: M2M100Config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 从配置中获取填充索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建一个共享的嵌入层，用于编码器和解码器
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 初始化编码器和解码器
        self.encoder = M2M100Encoder(config, self.shared)
        self.decoder = M2M100Decoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.shared = value
        # 更新编码器和解码器的嵌入层
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 实现权重绑定的方法
    def _tie_weights(self):
        # 如果配置中指定了词嵌入层共享
        if self.config.tie_word_embeddings:
            # 将编码器和解码器的词嵌入层绑定或克隆为共享的嵌入层
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器的方法
    def get_encoder(self):
        return self.encoder

    # 获取解码器的方法
    def get_decoder(self):
        return self.decoder

    # 前向传播方法，接受多个输入参数，并返回模型的输出
    @add_start_docstrings_to_model_forward(M2M_100_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数参数的详细描述在文档字符串中给出
    ):
        # 函数主体实现模型的前向传播逻辑，具体细节可以参考函数内部实现
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        # 设置输出注意力权重，如果未指定则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，如果未指定则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否使用缓存，如果未指定则使用配置中的默认设置
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 设置是否返回字典形式的输出，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果没有提供编码器的输出，则调用编码器来生成编码器的输出
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # 如果用户传入的是一个元组形式的编码器输出，在 return_dict=True 时将其包装为 BaseModelOutput 类型
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # 解码器的输出包括 (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不是以字典形式返回结果，则将解码器和编码器输出组合起来返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 以 Seq2SeqModelOutput 类型返回结果，包括解码器和编码器的相关隐藏状态和注意力权重
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 为 M2M100ForConditionalGeneration 类添加文档字符串，描述其作为带有语言建模头的 M2M100 模型，用于摘要生成
@add_start_docstrings(
    "The M2M100 Model with a language modeling head. Can be used for summarization.", M2M_100_START_DOCSTRING
)
class M2M100ForConditionalGeneration(M2M100PreTrainedModel):
    # 指定模型中用于连接的前缀
    base_model_prefix = "model"
    # 定义共享权重的键列表，这些权重被绑定在一起
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化函数，接受 M2M100Config 类型的配置对象作为参数
    def __init__(self, config: M2M100Config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 创建一个 M2M100Model 模型实例，并赋值给 self.model
        self.model = M2M100Model(config)
        # 创建一个线性层 lm_head，用于语言建模任务的最终处理，将输入维度设为 config.d_model，输出维度设为 self.model.shared.num_embeddings，不带偏置
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 执行额外的初始化操作和最终处理
        self.post_init()

    # 返回模型中的编码器部分
    def get_encoder(self):
        return self.model.get_encoder()

    # 返回模型中的解码器部分
    def get_decoder(self):
        return self.model.get_decoder()

    # 返回 lm_head 层，用于输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置 lm_head 层的新嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 前向传播函数，接受多个输入参数，包括输入的 ID、注意力掩码、解码器输入的 ID 等
    # 使用装饰器添加文档字符串，描述了输入参数和输出类型
    # 使用装饰器替换返回值的文档字符串为 Seq2SeqLMOutput 类型，并指定相关配置类
    # 添加末尾的文档字符串，展示了 M2M-100 生成任务的示例
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Returns a tuple containing either torch.Tensor or Seq2SeqLMOutput.

        """
        # Determine whether to use the provided return_dict or the default from configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            # If decoder_input_ids is not provided, shift labels to the right for decoder input
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Forward pass through the model with specified inputs and optional arguments
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
        
        # Generate logits from the language model head
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            # Move labels tensor to the same device as lm_logits for proper loss computation
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            # Compute masked language modeling loss using CrossEntropyLoss
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # Return output as tuple if return_dict is False
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return Seq2SeqLMOutput object containing relevant outputs if return_dict is True
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
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
        # 如果使用了过去的键值（past_key_values），则计算过去的长度
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能已经仅传递了最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：仅保留最后一个输入 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 修剪 decoder_input_ids，去除前面不需要的部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回一个字典，包含不同的模型输入和掩码
        return {
            "input_ids": None,  # encoder_outputs 已定义，input_ids 不再需要
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能是为了调试目的）
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 重新排序过去的键值，根据 beam_idx 进行重新排列
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```