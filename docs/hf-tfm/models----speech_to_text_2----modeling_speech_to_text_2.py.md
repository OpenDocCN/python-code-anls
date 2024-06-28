# `.\models\speech_to_text_2\modeling_speech_to_text_2.py`

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
""" PyTorch Speech2Text2 model."""


import copy
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging, replace_return_docstrings
from .configuration_speech_to_text_2 import Speech2Text2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Speech2Text2Config"
_CHECKPOINT_FOR_DOC = "facebook/s2t-wav2vec2-large-en-de"


SPEECH_TO_TEXT_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/s2t-wav2vec2-large-en-de",
    # See all Speech2Text2 models at https://huggingface.co/models?filter=speech2text2
]


# Copied from transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding with Speech2Text->Speech2Text2
class Speech2Text2SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # 调用make_weights方法初始化位置编码权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用get_embedding方法生成位置编码权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 如果已经有了weights属性，在forward方法中将权重转换为正确的数据类型和设备
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 将生成的位置编码权重设置为模型的可学习参数，并且不需要梯度
        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        # 计算嵌入向量的一半维度
        half_dim = embedding_dim // 2
        # 计算公式中的常数
        emb = math.log(10000) / (half_dim - 1)
        # 计算指数部分
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        # 计算正弦和余弦位置编码
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果嵌入维度是奇数，则进行零填充
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果指定了填充索引，则将填充位置的嵌入向量设为零向量
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        # 将嵌入向量转换为默认的 torch 数据类型并返回
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # 从输入的 token ids 中创建位置 ids，保持填充的 token 仍然填充
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # 如果超出当前权重矩阵的最大位置，则扩展权重矩阵
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 根据位置 ids 选择对应的权重，重塑为 (batch_size, seq_len, -1) 的形状并返回
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # 使用 input_ids 中非填充符号的位置数替换为它们的位置数字，位置数字从 padding_idx+1 开始
        mask = input_ids.ne(padding_idx).int()
        # 累积计算位置编号，加上过去键值长度，并保持输入张量的形状
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx
# 从transformers.models.bart.modeling_bart.BartAttention复制代码，将Bart->Speech2Text2
class Speech2Text2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Speech2Text2Config] = None,
    ):
        super().__init__()
        # 初始化注意力机制的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查embed_dim是否能够被num_heads整除
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
        # 重新形状张量，以便进行多头注意力计算
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
        # 实现注意力机制的前向传播
        # hidden_states: 输入的隐藏状态张量
        # key_value_states: 键-值状态张量（可选）
        # past_key_value: 过去的键-值对（可选）
        # attention_mask: 注意力掩码（可选）
        # layer_head_mask: 层头掩码（可选）
        # output_attentions: 是否输出注意力权重（布尔值）
        pass  # 这里应该有具体的实现，但是在这个例子中被省略了

class Speech2Text2DecoderLayer(nn.Module):
    def __init__(self, config: Speech2Text2Config):
        super().__init__()
        self.embed_dim = config.d_model

        # 初始化解码器层的参数
        self.self_attn = Speech2Text2Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 层归一化层，用于自注意力
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 如果配置为解码器，还需要初始化编码器-解码器注意力层
        if config.is_decoder:
            self.encoder_attn = Speech2Text2Attention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 线性变换层，用于前馈神经网络部分
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    # 定义模型的前向传播方法，用于生成模型的输出
    def forward(
        self,
        # 输入参数：当前层的隐藏状态，类型为 torch.Tensor
        hidden_states: torch.Tensor,
        # 输入参数：注意力遮罩，可选的 torch.Tensor 类型，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 输入参数：编码器的隐藏状态，可选的 torch.Tensor 类型，默认为 None
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 输入参数：编码器的注意力遮罩，可选的 torch.Tensor 类型，默认为 None
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 输入参数：层级头部掩码，可选的 torch.Tensor 类型，默认为 None
        layer_head_mask: Optional[torch.Tensor] = None,
        # 输入参数：跨注意力层级头部掩码，可选的 torch.Tensor 类型，默认为 None
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        # 输入参数：过去的键值对，可选的 torch.Tensor 类型的元组，默认为 None
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 输入参数：是否输出注意力权重，可选的布尔值，默认为 False
        output_attentions: Optional[bool] = False,
        # 输入参数：是否使用缓存，可选的布尔值，默认为 True
        use_cache: Optional[bool] = True,
# 定义一个自定义的解码器类 Speech2Text2Decoder，继承自 Speech2Text2PreTrainedModel
class Speech2Text2Decoder(Speech2Text2PreTrainedModel):
    """
    Transformer 解码器，由 config.decoder_layers 层组成。每一层是一个 Speech2Text2DecoderLayer 类的实例。

    Args:
        config: Speech2Text2Config，模型的配置对象
        embed_tokens (nn.Embedding): 输出的嵌入层对象
    """

    # 初始化方法，接受一个配置对象 config
    def __init__(self, config: Speech2Text2Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置层级下降的概率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充标记的索引
        self.padding_idx = config.pad_token_id
        # 设置目标序列的最大位置
        self.max_target_positions = config.max_target_positions
        # 根据配置中的 scale_embedding 决定是否对嵌入进行缩放
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 定义词嵌入层，输入参数为词汇表大小、嵌入维度、填充标记的索引
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 定义位置编码器，使用 Speech2Text2SinusoidalPositionalEmbedding 类创建
        self.embed_positions = Speech2Text2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
            self.padding_idx,
        )

        # 定义多层解码器，使用 nn.ModuleList 存储多个 Speech2Text2DecoderLayer 层
        self.layers = nn.ModuleList([Speech2Text2DecoderLayer(config) for _ in range(config.decoder_layers)])

        # 是否启用梯度检查点
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入的嵌入层对象
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入的嵌入层对象
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    # 定义一个前向传播函数，用于模型推理阶段
    def forward(
        self,
        input_ids=None,  # 输入的token IDs
        attention_mask=None,  # 注意力掩码，指示哪些位置是padding的
        encoder_hidden_states=None,  # 编码器的隐藏状态
        encoder_attention_mask=None,  # 编码器的注意力掩码
        head_mask=None,  # 多头注意力的掩码
        cross_attn_head_mask=None,  # 跨注意力头的掩码
        past_key_values=None,  # 过去的键值对（用于循环生成）
        inputs_embeds=None,  # 输入的嵌入表示
        use_cache=None,  # 是否使用缓存（用于加速推理）
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出隐藏状态
        return_dict=None,  # 是否以字典形式返回结果
# 使用装饰器添加文档字符串，描述了 Speech2Text2Model 的用途和结构特性
@add_start_docstrings(
    "The Speech2Text2 Model with a language modeling head. Can be used for summarization.",
    SPEECH_TO_TEXT_2_START_DOCSTRING,
)
# 定义 Speech2Text2DecoderWrapper 类，继承自 Speech2Text2PreTrainedModel 类
class Speech2Text2DecoderWrapper(Speech2Text2PreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    # 初始化方法，接收配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 Speech2Text2Decoder 对象并赋值给 self.decoder
        self.decoder = Speech2Text2Decoder(config)

    # 前向传播方法，将参数传递给 self.decoder 的前向传播方法
    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


# 使用装饰器添加文档字符串，描述了 Speech2Text2Decoder 的用途和结构特性
@add_start_docstrings(
    "The Speech2Text2 Decoder with a language modeling head. Can be used as the decoder part of"
    " [`EncoderDecoderModel`] and [`SpeechEncoderDecoder`].",
    SPEECH_TO_TEXT_2_START_DOCSTRING,
)
# 定义 Speech2Text2ForCausalLM 类，继承自 Speech2Text2PreTrainedModel 类
class Speech2Text2ForCausalLM(Speech2Text2PreTrainedModel):
    # 类变量，定义了绑定权重的键名列表
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接收配置对象 config
    def __init__(self, config):
        # 深拷贝配置对象 config
        config = copy.deepcopy(config)
        # 设置 config 的 is_decoder 属性为 True，is_encoder_decoder 属性为 False
        config.is_decoder = True
        config.is_encoder_decoder = False
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 Speech2Text2DecoderWrapper 对象并赋值给 self.model
        self.model = Speech2Text2DecoderWrapper(config)

        # 创建线性层 lm_head，连接 config 的隐藏大小和词汇表大小，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用自定义的初始化方法，初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层的方法，返回 self.model.decoder 的 embed_tokens 属性
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层的方法，将 value 赋值给 self.model.decoder 的 embed_tokens 属性
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出嵌入层的方法，返回 self.lm_head 属性
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层的方法，将 new_embeddings 赋值给 self.lm_head 属性
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器的方法，将 decoder 赋值给 self.model.decoder
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器的方法，返回 self.model.decoder
    def get_decoder(self):
        return self.model.decoder

    # 用于替换返回值文档字符串的装饰器，输出类型为 CausalLMOutputWithCrossAttentions，配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，接收多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 省略前向传播的具体实现

    # 为生成准备输入的方法，接收输入的多个参数，并省略其具体实现
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 省略输入准备的具体实现
    ):
        # 如果模型作为编码器-解码器模型的解码器使用，那么注意力遮罩将动态创建
        if attention_mask is None:
            # 如果注意力遮罩为None，则创建一个全为1的注意力遮罩，形状与input_ids相同
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去键值对的长度（过去的状态）
            past_length = past_key_values[0][0].shape[2]

            # 某些生成方法可能只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                # 如果输入ID的长度大于过去的长度，移除前缀长度为过去的长度
                remove_prefix_length = past_length
            else:
                # 否则，默认旧的行为：保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 更新输入ID，仅保留后缀部分以便进行生成
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回一个包含各种生成所需信息的字典
        return {
            "input_ids": input_ids,  # 编码器输出已定义，不需要input_ids
            "attention_mask": attention_mask,  # 注意力遮罩
            "past_key_values": past_key_values,  # 过去的键值对
            "use_cache": use_cache,  # 是否使用缓存
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序过去的状态，根据beam_idx重排过去的键值对
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                # 对每一层的过去状态，根据beam_idx在设备上重新选择索引
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```