# `.\models\speech_to_text\modeling_speech_to_text.py`

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
""" PyTorch Speech2Text model."""

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_speech_to_text import Speech2TextConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Speech2TextConfig"


SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/s2t-small-librispeech-asr",
    # See all Speech2Text models at https://huggingface.co/models?filter=speech_to_text
]


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个新的和input_ids相同形状的张量，用于存放右移后的结果
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将input_ids向右移动一位，即第一列为decoder_start_token_id，后续列为input_ids的前n-1列
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 如果有-100值，将其替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Conv1dSubsampler(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """
    def __init__(self, config):
        super(Conv1dSubsampler, self).__init__()
        self.config = config
        self.num_layers = config.num_conv_layers  # 设置卷积层的数量
        self.in_channels = config.input_feat_per_channel * config.input_channels  # 输入通道数
        self.mid_channels = config.conv_channels  # 中间层的通道数
        self.out_channels = config.d_model  # 输出通道数，这里是模型的维度
        self.kernel_sizes = config.conv_kernel_sizes  # 卷积核的大小列表

        self.conv_layers = nn.ModuleList(
            # 创建卷积层的 ModuleList，每层使用不同的卷积核大小和通道数设置
            nn.Conv1d(
                self.in_channels if i == 0 else self.mid_channels // 2,  # 输入通道数的设置
                self.mid_channels if i < self.num_layers - 1 else self.out_channels * 2,  # 输出通道数的设置
                kernel_size=k,  # 当前卷积层的卷积核大小
                stride=2,  # 步长设置为2
                padding=k // 2,  # 根据卷积核大小设置填充
            )
            for i, k in enumerate(self.kernel_sizes)  # 遍历卷积核大小列表
        )

    def forward(self, input_features):
        hidden_states = input_features.transpose(1, 2).contiguous()  # 转置输入特征以适应卷积操作 -> B x (C x D) x T
        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)  # 应用当前卷积层
            hidden_states = nn.functional.glu(hidden_states, dim=1)  # 应用 GLU 激活函数
        hidden_states = hidden_states.transpose(1, 2).contiguous()  # 再次转置以恢复原始维度 -> T x B x (C x D)
        return hidden_states
class Speech2TextSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2  # 偏移量，用于处理位置编码时的偏移
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.padding_idx = padding_idx  # 可选的填充索引，用于指定填充位置
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 在前向传播中将权重转换为正确的数据类型和设备
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)  # 将嵌入权重设置为模块的参数
        self.weights.requires_grad = False  # 设置权重不需要梯度
        self.weights.detach_()  # 分离权重，使其不参与反向传播

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        构建正弦位置编码。这与tensor2tensor中的实现匹配，但与《Attention Is All You Need》第3.5节中的描述略有不同。
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)  # 计算正弦周期的频率
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)  # 计算正弦位置编码
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)  # 计算位置编码矩阵
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)  # 合并正弦和余弦编码
        if embedding_dim % 2 == 1:
            # 如果嵌入维度为奇数，则在末尾添加零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0  # 将填充索引处的位置编码设置为零向量
        return emb.to(torch.get_default_dtype())  # 返回默认数据类型的位置编码

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # 从输入的token ids创建位置 ids，保持填充的位置不变
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # 如果需要，扩展嵌入
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        # 从输入的 token ids 创建位置 ids
        raise NotImplementedError
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: torch.Tensor representing input tensor with token IDs
            padding_idx: int, index of padding token in input_ids
            past_key_values_length: int, length of past key values to be considered
        Returns:
            torch.Tensor representing the tensor with position indices
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        # Create a mask where non-padding elements are marked as 1 and padding elements as 0
        mask = input_ids.ne(padding_idx).int()
        # Compute cumulative sum of the mask along the second dimension (sequence length),
        # and adjust for past key values length. Type conversion ensures compatibility with mask.
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        # Convert the indices to long type and add padding_idx to non-padding positions
        return incremental_indices.long() + padding_idx
# 从transformers.models.bart.modeling_bart.BartAttention复制的代码，将Bart替换为Speech2Text
class Speech2TextAttention(nn.Module):
    """来自“Attention Is All You Need”论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Speech2TextConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设置注意力机制的嵌入维度
        self.num_heads = num_heads  # 设置注意力头的数量
        self.dropout = dropout  # 设置dropout率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        self.config = config  # 保存配置信息

        # 确保embed_dim能够被num_heads整除，否则抛出异常
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能够被num_heads整除 (当前 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于缩放注意力分数
        self.is_decoder = is_decoder  # 标识是否是解码器注意力
        self.is_causal = is_causal  # 标识是否是因果注意力

        # 初始化线性层，用于对key、value和query进行投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """将张量形状重新排列为(bsz, num_heads, seq_len, head_dim)，并转置前两个维度"""
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
        """前向传播函数，实现注意力机制的计算"""
        pass  # 这里只是占位符，实际上需要根据具体实现补充内容


# 从transformers.models.mbart.modeling_mbart.MBartEncoderLayer复制的代码，将MBart替换为Speech2Text, MBART替换为SPEECH_TO_TEXT
class Speech2TextEncoderLayer(nn.Module):
    def __init__(self, config: Speech2TextConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 设置编码器层的嵌入维度

        # 初始化自注意力层
        self.self_attn = SPEECH_TO_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 使用LayerNorm对自注意力层的输出进行归一化
        self.dropout = config.dropout  # 设置dropout率
        self.activation_fn = ACT2FN[config.activation_function]  # 获取激活函数
        self.activation_dropout = config.activation_dropout  # 设置激活函数的dropout率
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 使用LayerNorm对最终输出进行归一化
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
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
        # 保存输入的残差连接，以备后续使用
        residual = hidden_states
        # 对输入的 hidden_states 进行 Layer Normalization 处理
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用 self attention 模块处理 normalized 后的 hidden_states，获取输出、注意力权重和注意力分布
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对输出的 hidden_states 进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 恢复残差连接
        hidden_states = residual + hidden_states

        # 再次保存输入的残差连接
        residual = hidden_states
        # 对上一步的输出进行 Layer Normalization 处理
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数和全连接层 fc1 处理 normalized 后的 hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对输出的 hidden_states 进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用全连接层 fc2 处理 dropout 后的 hidden_states
        hidden_states = self.fc2(hidden_states)
        # 对输出的 hidden_states 进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 恢复残差连接
        hidden_states = residual + hidden_states

        # 如果 hidden_states 的数据类型是 torch.float16，并且包含 inf 或 nan 的情况
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 对 hidden_states 进行值的 clamp 处理，避免出现超出浮点数范围的情况
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组，包含最终的 hidden_states
        outputs = (hidden_states,)

        # 如果需要输出 attentions，将 attentions 加入到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出元组
        return outputs
# 从transformers.models.mbart.modeling_mbart.MBartDecoderLayer复制代码，将MBart替换为Speech2Text，MBART替换为SPEECH_TO_TEXT
class Speech2TextDecoderLayer(nn.Module):
    def __init__(self, config: Speech2TextConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 获取配置中的模型维度大小

        # 初始化自注意力机制
        self.self_attn = SPEECH_TO_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,  # 传入嵌入维度
            num_heads=config.decoder_attention_heads,  # 解码器注意力头数
            dropout=config.attention_dropout,  # 注意力机制的dropout
            is_decoder=True,  # 标识为解码器自注意力
            is_causal=True,  # 使用因果注意力
            config=config,  # 传入配置对象
        )
        self.dropout = config.dropout  # 配置中的dropout比率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout比率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层的LayerNorm

        # 初始化编码器注意力机制
        self.encoder_attn = SPEECH_TO_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,  # 传入嵌入维度
            config.decoder_attention_heads,  # 解码器注意力头数
            dropout=config.attention_dropout,  # 注意力机制的dropout
            is_decoder=True,  # 标识为解码器自注意力
            config=config,  # 传入配置对象
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 编码器注意力层的LayerNorm

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # 第一个线性层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # 第二个线性层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终的LayerNorm

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



# Speech2TextPreTrainedModel类，继承自PreTrainedModel类
class Speech2TextPreTrainedModel(PreTrainedModel):
    config_class = Speech2TextConfig  # 配置类为Speech2TextConfig
    base_model_prefix = "model"  # 基础模型前缀为"model"
    main_input_name = "input_features"  # 主要输入名称为"input_features"
    supports_gradient_checkpointing = True  # 支持梯度检查点

    def _init_weights(self, module):
        std = self.config.init_std  # 初始化标准差
        if isinstance(module, (nn.Linear, nn.Conv1d)):  # 如果模块是线性层或一维卷积层
            module.weight.data.normal_(mean=0.0, std=std)  # 权重初始化为正态分布
            if module.bias is not None:  # 如果有偏置项
                module.bias.data.zero_()  # 偏置初始化为零
        elif isinstance(module, nn.Embedding):  # 如果模块是嵌入层
            module.weight.data.normal_(mean=0.0, std=std)  # 权重初始化为正态分布
            if module.padding_idx is not None:  # 如果有填充索引
                module.weight.data[module.padding_idx].zero_()  # 填充索引位置的权重初始化为零

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        计算卷积层的输出长度
        """
        for i in range(self.config.num_conv_layers):  # 遍历卷积层数量
            input_lengths = (input_lengths - 1) // 2 + 1  # 根据卷积层的计算方式更新输入长度

        return input_lengths  # 返回更新后的输入长度
    # 定义一个方法来获取特征向量的注意力掩码
    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        # 如果注意力掩码的维度大于2，说明生成了一个3D的注意力掩码，
        # 这里将其转换为2D的形式
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        # 根据注意力掩码的和，获取下采样后的长度列表
        subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        bsz = attention_mask.size()[0]

        # 创建一个与注意力掩码相同形状的全零张量，用于存储生成的注意力掩码
        attention_mask = torch.zeros(
            (bsz, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 将生成的注意力掩码中对应于输出长度索引之前的所有位置设置为1，
        # 这确保了这些位置的数值被全部关注到
        attention_mask[(torch.arange(bsz, device=attention_mask.device), subsampled_lengths - 1)] = 1

        # 反转张量并在最后一个维度上累加，然后再次反转，将其转换为整数类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).long()

        # 返回生成的注意力掩码
        return attention_mask
# 定义文档字符串，说明继承自 `PreTrainedModel` 的模型基类的用途和特性
SPEECH_TO_TEXT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Speech2TextConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义空的输入文档字符串，此处留空，没有具体的输入参数描述
SPEECH_TO_TEXT_INPUTS_DOCSTRING = r"""
"""


class Speech2TextEncoder(Speech2TextPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Speech2TextEncoderLayer`].

    Args:
        config: Speech2TextConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: Speech2TextConfig):
        super().__init__(config)

        # 从配置中初始化模型属性
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 初始化卷积子采样器
        self.conv = Conv1dSubsampler(config)

        # 初始化声学-文本正弦位置嵌入
        self.embed_positions = Speech2TextSinusoidalPositionalEmbedding(
            self.max_source_positions,
            embed_dim,
            self.padding_idx,
        )

        # 创建多层 Transformer 编码器层
        self.layers = nn.ModuleList([Speech2TextEncoderLayer(config) for _ in range(config.encoder_layers)])
        
        # 初始化层归一化模块
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 禁用梯度检查点
        self.gradient_checkpointing = False
        
        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):  
    pass  # 此处为 forward 方法声明，但未提供具体实现，因此暂时没有更多的内容需要注释
        

class Speech2TextDecoder(Speech2TextPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`Speech2TextDecoderLayer`]

    Args:
        config: Speech2TextConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化方法，接受一个 Speech2TextConfig 类型的参数 config
    def __init__(self, config: Speech2TextConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置dropout参数
        self.dropout = config.dropout
        # 设置decoder层的layerdrop参数
        self.layerdrop = config.decoder_layerdrop
        # 设置padding的索引号
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_target_positions
        # 如果配置中指定了scale_embedding，则设置embed_scale为d_model的平方根，否则为1.0
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 创建一个词嵌入层，vocab_size为词汇表大小，d_model为词嵌入维度，padding_idx为填充的索引号
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 创建一个音频转文本的正弦位置编码层，max_target_positions为最大目标位置，d_model为模型维度，padding_idx为填充的索引号
        self.embed_positions = Speech2TextSinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
            self.padding_idx,
        )

        # 创建一个由多个音频转文本解码器层组成的列表，每层的配置都来自config
        self.layers = nn.ModuleList([Speech2TextDecoderLayer(config) for _ in range(config.decoder_layers)])

        # 创建一个LayerNorm层，对输入进行归一化处理，维度为d_model
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点参数为False
        self.gradient_checkpointing = False
        
        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入词嵌入层的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入词嵌入层的方法
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播方法，接受多个参数用于处理音频转文本的过程
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
@add_start_docstrings(
    "The bare Speech2Text Model outputting raw hidden-states without any specific head on top.",
    SPEECH_TO_TEXT_START_DOCSTRING,
)
# 定义了一个不带特定顶部头部的语音到文本模型
class Speech2TextModel(Speech2TextPreTrainedModel):
    def __init__(self, config: Speech2TextConfig):
        super().__init__(config)

        # 初始化编码器和解码器
        self.encoder = Speech2TextEncoder(config)
        self.decoder = Speech2TextDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回解码器的嵌入层
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置解码器的嵌入层
        self.decoder.embed_tokens = value

    def get_encoder(self):
        # 返回编码器
        return self.encoder

    def get_decoder(self):
        # 返回解码器
        return self.decoder

    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 重写了父类的前向传播方法，增加了文档字符串和输出的替换
    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    "The Speech2Text Model with a language modeling head. Can be used for summarization.",
    SPEECH_TO_TEXT_START_DOCSTRING,
)
# 定义了一个带有语言建模头部的语音到文本模型，可以用于摘要生成
class Speech2TextForConditionalGeneration(Speech2TextPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Speech2TextConfig):
        super().__init__(config)

        # 初始化基础语音到文本模型和语言建模头部
        self.model = Speech2TextModel(config)
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_encoder(self):
        # 返回基础模型的编码器
        return self.model.get_encoder()

    def get_decoder(self):
        # 返回基础模型的解码器
        return self.model.get_decoder()

    def get_output_embeddings(self):
        # 返回语言建模头部的输出嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置语言建模头部的输出嵌入层
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，接收多个输入参数，并可选地返回不同的输出
    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 如果使用了过去的关键值（用于缓存），则截断decoder_input_ids，只保留最后一个位置的输入
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回包含各种输入参数的字典，用于生成器的输入准备
        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 改为False以避免缓存（可能用于调试）
        }

    @staticmethod
    # 重新排序缓存中的过去关键值，以适应beam搜索中的索引重排
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重排序后的过去关键值
        reordered_past = ()
        # 遍历每个层的过去关键值
        for layer_past in past_key_values:
            # 将每个层的过去状态按beam_idx重排，并转移到对应设备
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重排后的过去关键值
        return reordered_past
```