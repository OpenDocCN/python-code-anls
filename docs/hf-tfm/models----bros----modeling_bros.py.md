# `.\models\bros\modeling_bros.py`

```py
# coding=utf-8
# Copyright 2023-present NAVER Corp, The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
""" PyTorch Bros model."""

# Import necessary libraries
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# Importing specific components from Hugging Face's library
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# Import Bros configuration from local file
from .configuration_bros import BrosConfig

# Get logger for logging messages
logger = logging.get_logger(__name__)

# Constant variables for documentation and model checkpoints
_CHECKPOINT_FOR_DOC = "jinho8345/bros-base-uncased"
_CONFIG_FOR_DOC = "BrosConfig"

# List of pretrained model archives
BROS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "jinho8345/bros-base-uncased",
    "jinho8345/bros-large-uncased",
    # See all Bros models at https://huggingface.co/models?filter=bros
]

# Start documentation string for Bros model
BROS_START_DOCSTRING = r"""
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BrosConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Input documentation string placeholder
BROS_INPUTS_DOCSTRING = r"""
"""


@dataclass
class BrosSpadeOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    
    This class inherits from `ModelOutput` in Hugging Face's library and serves as a base for outputs
    from token classification models specific to the Bros model.
    
    Attributes:
        Inherits attributes from `ModelOutput`.
    """
    # 定义函数的参数及其类型注释，loss 是一个可选的浮点张量，表示分类损失
    # initial_token_logits 是一个张量，形状为 (batch_size, sequence_length, config.num_labels)，表示实体初始标记的分类分数（SoftMax 之前）
    # subsequent_token_logits 是一个张量，形状为 (batch_size, sequence_length, sequence_length+1)，表示实体序列标记的分类分数（SoftMax 之前）
    # hidden_states 是一个可选的张量元组，当传入参数 output_hidden_states=True 或者配置参数 config.output_hidden_states=True 时返回，包含每层模型输出的隐藏状态
    # attentions 是一个可选的张量元组，当传入参数 output_attentions=True 或者配置参数 config.output_attentions=True 时返回，包含每层模型输出的注意力权重
    """

    # loss 表示分类损失，默认为 None
    loss: Optional[torch.FloatTensor] = None
    # initial_token_logits 表示实体初始标记的分类分数，默认为 None
    initial_token_logits: torch.FloatTensor = None
    # subsequent_token_logits 表示实体序列标记的分类分数，默认为 None
    subsequent_token_logits: torch.FloatTensor = None
    # hidden_states 表示模型每层的隐藏状态的元组，默认为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions 表示模型每层的注意力权重的元组，默认为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
class BrosPositionalEmbedding1D(nn.Module):
    # 引用：https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L15
    # 一维位置编码的模块定义

    def __init__(self, config):
        super(BrosPositionalEmbedding1D, self).__init__()
        # 初始化函数，接收配置参数 config

        self.dim_bbox_sinusoid_emb_1d = config.dim_bbox_sinusoid_emb_1d
        # 从配置中获取一维位置编码的维度大小

        # 计算正弦函数的频率逆数，用于位置编码
        inv_freq = 1 / (
            10000 ** (torch.arange(0.0, self.dim_bbox_sinusoid_emb_1d, 2.0) / self.dim_bbox_sinusoid_emb_1d)
        )
        # 将频率逆数作为缓冲区注册到模块中
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，输入位置序列，返回位置编码张量

        seq_size = pos_seq.size()
        b1, b2, b3 = seq_size
        # 获取位置序列的大小

        sinusoid_inp = pos_seq.view(b1, b2, b3, 1) * self.inv_freq.view(1, 1, 1, self.dim_bbox_sinusoid_emb_1d // 2)
        # 计算正弦输入，使用位置序列乘以频率逆数的张量，并广播到合适的形状

        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        # 将正弦和余弦结果连接在一起，得到最终的位置编码张量

        return pos_emb


class BrosPositionalEmbedding2D(nn.Module):
    # 二维位置编码的模块定义

    def __init__(self, config):
        super(BrosPositionalEmbedding2D, self).__init__()
        # 初始化函数，接收配置参数 config

        self.dim_bbox = config.dim_bbox
        # 从配置中获取边界框维度的大小

        # 创建一维位置编码模块实例，用于X和Y方向
        self.x_pos_emb = BrosPositionalEmbedding1D(config)
        self.y_pos_emb = BrosPositionalEmbedding1D(config)

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，输入边界框张量，返回位置编码后的张量

        stack = []
        # 初始化一个空列表，用于存储位置编码的结果

        for i in range(self.dim_bbox):
            # 遍历边界框维度

            if i % 2 == 0:
                stack.append(self.x_pos_emb(bbox[..., i]))
                # 如果是偶数索引，使用X方向的位置编码模块
            else:
                stack.append(self.y_pos_emb(bbox[..., i]))
                # 如果是奇数索引，使用Y方向的位置编码模块

        bbox_pos_emb = torch.cat(stack, dim=-1)
        # 将所有位置编码结果连接在一起，形成最终的边界框位置编码张量

        return bbox_pos_emb


class BrosBboxEmbeddings(nn.Module):
    # 边界框嵌入的模块定义

    def __init__(self, config):
        super(BrosBboxEmbeddings, self).__init__()
        # 初始化函数，接收配置参数 config

        self.bbox_sinusoid_emb = BrosPositionalEmbedding2D(config)
        # 创建二维位置编码模块实例

        self.bbox_projection = nn.Linear(config.dim_bbox_sinusoid_emb_2d, config.dim_bbox_projection, bias=False)
        # 创建线性层，用于将二维位置编码映射到边界框投影维度

    def forward(self, bbox: torch.Tensor):
        # 前向传播函数，输入边界框张量，返回映射后的边界框嵌入张量

        bbox_t = bbox.transpose(0, 1)
        # 转置边界框张量，使得第一维度和第二维度交换

        bbox_pos = bbox_t[None, :, :, :] - bbox_t[:, None, :, :]
        # 计算边界框的位置关系张量，使用广播来扩展维度

        bbox_pos_emb = self.bbox_sinusoid_emb(bbox_pos)
        # 使用二维位置编码模块对位置关系张量进行编码

        bbox_pos_emb = self.bbox_projection(bbox_pos_emb)
        # 使用线性层对位置编码结果进行投影映射

        return bbox_pos_emb


class BrosTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 文本嵌入的模块定义
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 创建词嵌入层，用于将词的索引映射成词向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，用于将位置索引映射成位置向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建token类型嵌入层，用于将token类型索引映射成token类型向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建LayerNorm层，用于对隐藏状态的归一化处理
        # 参数名不符合 snake-case 命名规范，是为了兼容 TensorFlow 的模型变量名
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建dropout层，用于在训练时进行随机失活处理
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置id (1, len position emb) 在序列化时是连续存储的，并且会被导出
        # 根据配置添加绝对或相对的位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册一个持久的缓冲区 position_ids ，存储连续的位置id
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # 注册一个非持久的缓冲区 token_type_ids ，存储所有位置的token类型id是0
        self.register_buffer(
            "token_type_ids",
            torch.zeros(
                self.position_ids.size(),
                dtype=torch.long,
                device=self.position_ids.device,
            ),
            persistent=False,
        )

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的词的索引
        token_type_ids: Optional[torch.Tensor] = None,  # token的类型id
        position_ids: Optional[torch.Tensor] = None,  # 位置id
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的词的向量
        past_key_values_length: int = 0,  # 之前的键值对的长度
    ) -> torch.Tensor:  # 返回值是张量
        # 如果有输入的词的索引，获取其形状，否则获取输入的词向量的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]  # 序列长度

        # 如果没有指定位置id，将位置id设置为连续的一段
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果没有指定token类型id，根据情况获取token类型id的值
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果没有指定输入的词向量，获取输入词的索引对应的词向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据token类型id获取token类型的嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算总的嵌入向量，包括词向量、token类型嵌入、位置嵌入
        embeddings = inputs_embeds + token_type_embeddings
        # 如果使用绝对位置嵌入，计算并加上位置嵌入向量
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对总的嵌入向量进行LayerNorm处理
        embeddings = self.LayerNorm(embeddings)
        # 对处理后的嵌入向量进行随机失活处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入向量
        return embeddings
class BrosSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏大小是否能被注意力头数整除，同时没有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果位置嵌入类型是相对键或相对键查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 调整形状以便计算注意力分数
    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[torch.Tensor] = False,
    ):
        # 此处是模型的前向传播函数，实现自注意力机制和额外的逻辑
        pass  # 这里可以根据具体实现添加详细的功能注释


# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制，将 Bert 改为 Bros
class BrosSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性层
        hidden_states = self.dense(hidden_states)
        # Dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BrosAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建自注意力和输出对象
        self.self = BrosSelfAttention(config)
        self.output = BrosSelfOutput(config)
        self.pruned_heads = set()  # 用于存储被修剪的注意力头部集合
    # 对 self 对象的 heads 进行修剪操作
    def prune_heads(self, heads):
        # 如果 heads 列表为空，则直接返回，不进行操作
        if len(heads) == 0:
            return
        
        # 调用 find_pruneable_heads_and_indices 函数查找可修剪的 heads 和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # 修剪 self.query 线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        # 修剪 self.key 线性层
        self.self.key = prune_linear_layer(self.self.key, index)
        # 修剪 self.value 线性层
        self.self.value = prune_linear_layer(self.self.value, index)
        # 修剪 self.output.dense 线性层，dim=1 表示在第一个维度上进行修剪
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录被修剪的 heads
        # 减去被修剪的 heads 的数量，更新注意力头的数量
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        # 计算所有注意力头的新尺寸
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        # 将被修剪的 heads 添加到 pruned_heads 集合中
        self.pruned_heads = self.pruned_heads.union(heads)

    # 定义 forward 方法，实现模型的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用 self.self 方法进行自注意力机制计算
        self_outputs = self.self(
            hidden_states=hidden_states,
            bbox_pos_emb=bbox_pos_emb,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        # 将 self_outputs[0] 与 hidden_states 作为输入，调用 self.output 方法
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力信息，则将 attentions 添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出注意力信息，则添加到 outputs 中
        return outputs
# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制而来，修改为 BrosIntermediate
class BrosIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征维度 config.hidden_size 转换为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数 ACT2FN[config.hidden_act] 或者直接使用给定的激活函数 config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过线性层变换
        hidden_states = self.dense(hidden_states)
        # 经过中间激活函数变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义 BrosOutput 类，继承自 nn.Module
class BrosOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征维度 config.intermediate_size 转换为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm 归一化层，对隐藏状态进行归一化，eps 是归一化过程中的小数值稳定项
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，以 config.hidden_dropout_prob 概率丢弃隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过线性层变换
        hidden_states = self.dense(hidden_states)
        # 经过 Dropout 层处理
        hidden_states = self.dropout(hidden_states)
        # 将输入张量 input_tensor 和处理后的 hidden_states 相加，并经过 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义 BrosLayer 类，继承自 nn.Module
class BrosLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置用于 feed forward 的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度，用于注意力计算
        self.seq_len_dim = 1
        # BrosAttention 类的实例，用于处理注意力
        self.attention = BrosAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力但不是解码器，则抛出异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise Exception(f"{self} should be used as a decoder model if cross attention is added")
            # 否则，创建 BrosAttention 类的实例，用于交叉注意力
            self.crossattention = BrosAttention(config)
        # BrosIntermediate 类的实例，用于处理中间层
        self.intermediate = BrosIntermediate(config)
        # BrosOutput 类的实例，用于处理输出层
        self.output = BrosOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,

        # 前向传播函数定义，接受多个输入参数，返回处理后的隐藏状态张量
        # hidden_states: 输入的隐藏状态张量
        # bbox_pos_emb: 边界框位置嵌入张量
        # attention_mask: 注意力掩码张量，可选
        # head_mask: 头部掩码张量，可选
        # encoder_hidden_states: 编码器隐藏状态张量，可选
        # encoder_attention_mask: 编码器注意力掩码张量，可选
        # past_key_value: 过去的键值对元组，可选
        # output_attentions: 是否输出注意力张量，默认为 False
    ) -> Tuple[torch.Tensor]:
        # 如果有缓存的过去的键/值对，则取前两个（用于自注意力）
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行自注意力计算，传入隐藏状态、边界框位置嵌入、注意力掩码、头部掩码等参数
        self_attention_outputs = self.attention(
            hidden_states,
            bbox_pos_emb=bbox_pos_emb,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，则最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 输出中除了最后一个元素（自注意力缓存），其余都作为输出
            outputs = self_attention_outputs[1:-1]
            # 当前的键/值对是最后一个元素
            present_key_value = self_attention_outputs[-1]
        else:
            # 输出中包括自注意力的权重
            outputs = self_attention_outputs[1:]

        # 跨注意力的当前键/值对初始化为None
        cross_attn_present_key_value = None
        # 如果是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果self对象具有crossattention属性，抛出异常
            if hasattr(self, "crossattention"):
                raise Exception(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # 如果有缓存的过去的键/值对，则取后两个（用于跨注意力）
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行跨注意力计算，传入自注意力输出、注意力掩码、头部掩码、编码器隐藏状态等参数
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取跨注意力的输出
            attention_output = cross_attention_outputs[0]
            # 将跨注意力的权重添加到输出中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将跨注意力的当前键/值对添加到现有的键/值对中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用分块机制到前向传播的输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        # 将层输出添加到输出元组中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回所有输出
        return outputs

    # 定义前馈网络的分块函数，接收注意力输出并返回层输出
    def feed_forward_chunk(self, attention_output):
        # 执行中间层计算
        intermediate_output = self.intermediate(attention_output)
        # 执行输出层计算，传入中间输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output
# 定义一个用于编码的自定义 PyTorch 模块，继承自 nn.Module
class BrosEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化模块的配置参数
        self.config = config
        # 创建多个 BrosLayer 模块组成的列表，数量由配置参数决定
        self.layer = nn.ModuleList([BrosLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
# 以下是 BrosPooler 类定义，用于池化模型隐藏状态
class BrosPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将隐藏状态的大小转换为配置参数中的隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 使用双曲正切函数作为激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 简单地使用第一个标记对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BrosRelationExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化关系抽取器模块的配置参数
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size
        self.head_hidden_size = config.hidden_size
        self.classifier_dropout_prob = config.classifier_dropout_prob

        # 使用指定的 dropout 概率创建一个 dropout 层
        self.drop = nn.Dropout(self.classifier_dropout_prob)
        # 使用线性层定义查询（query）操作，将骨干隐藏状态大小映射到关系头大小的多个关系
        self.query = nn.Linear(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)

        # 使用线性层定义键（key）操作，将骨干隐藏状态大小映射到关系头大小的多个关系
        self.key = nn.Linear(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)

        # 定义一个虚拟节点，通过 nn.Parameter 创建，值为全零向量
        self.dummy_node = nn.Parameter(torch.zeros(1, self.backbone_hidden_size))

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        # 对查询层进行查询操作，并应用 dropout
        query_layer = self.query(self.drop(query_layer))

        # 创建一个虚拟向量，将其添加到键层中
        dummy_vec = self.dummy_node.unsqueeze(0).repeat(1, key_layer.size(1), 1)
        key_layer = torch.cat([key_layer, dummy_vec], axis=0)
        
        # 对键层进行键操作，并应用 dropout
        key_layer = self.key(self.drop(key_layer))

        # 重新调整查询层和键层的形状以适应多头关系的表示
        query_layer = query_layer.view(
            query_layer.size(0), query_layer.size(1), self.n_relations, self.head_hidden_size
        )
        key_layer = key_layer.view(key_layer.size(0), key_layer.size(1), self.n_relations, self.head_hidden_size)

        # 计算查询层和键层之间的关系分数，采用矩阵乘法进行计算
        relation_score = torch.matmul(
            query_layer.permute(2, 1, 0, 3), key_layer.permute(2, 1, 3, 0)
        )  # 相当于 torch.einsum("ibnd,jbnd->nbij", (query_layer, key_layer))

        return relation_score
class BrosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 BrosConfig 作为配置类
    config_class = BrosConfig
    # 基础模型的名称前缀
    base_model_prefix = "bros"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重
            # 与 TF 版本稍有不同，TF 使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则初始化为零向量
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了 padding_idx，则将对应位置的权重初始化为零向量
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，初始化偏置为零向量，初始化权重为全1向量
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    "The bare Bros Model transformer outputting raw hidden-states without any specific head on top.",
    BROS_START_DOCSTRING,
)
class BrosModel(BrosPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        # 初始化 BrosModel 类
        self.config = config

        # 初始化文本嵌入层、边界框嵌入层和编码器
        self.embeddings = BrosTextEmbeddings(config)
        self.bbox_embeddings = BrosBboxEmbeddings(config)
        self.encoder = BrosEncoder(config)

        # 如果需要添加池化层，则初始化池化层
        self.pooler = BrosPooler(config) if add_pooling_layer else None

        # 初始化模型权重
        self.init_weights()

    def get_input_embeddings(self):
        # 返回文本嵌入层的权重
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置文本嵌入层的权重
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 对模型的注意力头进行剪枝
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `forward`，用于执行模型的前向传播操作，通常在神经网络模型中使用
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可以是一个 PyTorch Tensor，默认为 None
        bbox: Optional[torch.Tensor] = None,  # bounding box 数据，用于图像处理或对象识别任务，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，指定模型注意力的作用范围，默认为 None
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，用于处理多句子任务时区分不同句子，默认为 None
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs，指定输入 token 的位置信息，默认为 None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，用于屏蔽某些注意力头，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量，用于直接输入嵌入向量而不是 token IDs，默认为 None
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，默认为 None
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码，默认为 None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对，用于存储过去的注意力信息，默认为 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，用于存储中间计算结果以加速反向传播，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出，默认为 None
# 为 BrosForTokenClassification 类添加文档字符串，描述其作为 Bros 模型的一个带有标记分类头的子类，用于命名实体识别（NER）等任务
@add_start_docstrings(
    """
    Bros Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BROS_START_DOCSTRING,
)
class BrosForTokenClassification(BrosPreTrainedModel):
    # 在加载时忽略的键列表，遇到未预期的 "pooler" 键时不加载
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象 config
        super().__init__(config)
        # 初始化模型的标签数量
        self.num_labels = config.num_labels

        # 初始化 BrosModel，传入配置对象 config
        self.bros = BrosModel(config)
        
        # 根据配置设置分类器的 dropout 概率，若配置对象中存在 "classifier_dropout" 属性则使用其值，否则使用隐藏层 dropout 的概率
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )
        # 定义一个 dropout 层，用于分类器
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 定义一个线性层，将隐藏状态映射到标签数量的输出空间
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化模型权重
        self.init_weights()

    # 为 forward 方法添加文档字符串，描述输入参数和输出类型，参照 BROS_INPUTS_DOCSTRING 的格式
    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回值的文档字符串，指定输出类型为 TokenClassifierOutput，配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox_first_token_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        Token classification model's forward method.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            bbox (torch.Tensor): Bounding box coordinates for tokens.
            attention_mask (torch.Tensor, optional): Mask for attention mechanism.
            token_type_ids (torch.Tensor, optional): Type IDs for tokens.
            position_ids (torch.Tensor, optional): Positional embeddings.
            head_mask (torch.Tensor, optional): Mask for attention heads.
            inputs_embeds (torch.Tensor, optional): Embedded inputs.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return as a dictionary.

        Returns:
            Union[Tuple[torch.Tensor], TokenClassifierOutput]: Model outputs.

        Examples:

        ```
        >>> import torch
        >>> from transformers import BrosProcessor, BrosForTokenClassification

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(**encoding)
        ```
        """

        # Determine whether to use the return dictionary format or not
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs to the model's token classification method
        outputs = self.bros(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the sequence output from the model's outputs
        sequence_output = outputs[0]

        # Apply dropout to the sequence output
        sequence_output = self.dropout(sequence_output)

        # Pass the sequence output through the classifier
        logits = self.classifier(sequence_output)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if bbox_first_token_mask is not None:
                bbox_first_token_mask = bbox_first_token_mask.view(-1)
                loss = loss_fct(
                    logits.view(-1, self.num_labels)[bbox_first_token_mask], labels.view(-1)[bbox_first_token_mask]
                )
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # If return_dict is False, prepare the output as a tuple
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, prepare the output as a TokenClassifierOutput object
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 用于标注任务的 Bros 模型，其在隐藏状态输出之上添加了一个标记分类头部。
# initial_token_classifier 用于预测每个实体的第一个标记，subsequent_token_classifier 用于预测实体内部的后续标记。
# 与 BrosForTokenClassification 相比，这个模型对序列化错误更加健壮，因为它从一个标记预测下一个标记。

@add_start_docstrings(
    """
    Bros Model with a token classification head on top (a entity_linker layer on top of the hidden-states output) e.g.
    for Entity-Linking. The entity_linker is used to predict intra-entity links (one entity to another entity).
    """,
    BROS_START_DOCSTRING,
)
class BrosSpadeELForTokenClassification(BrosPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size

        self.bros = BrosModel(config)
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )

        # Initial token classification for Entity Linking
        self.initial_token_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        # Entity linker for Entity Linking
        self.entity_linker = BrosRelationExtractor(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BrosSpadeOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox_first_token_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        initial_token_labels: Optional[torch.Tensor] = None,
        subsequent_token_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，将配置对象传递给父类
        super().__init__(config)
        # 将配置对象保存在实例中
        self.config = config
        # 从配置对象中获取标签数目并保存在实例中
        self.num_labels = config.num_labels
        # 从配置对象中获取关系数目并保存在实例中
        self.n_relations = config.n_relations
        # 从配置对象中获取隐藏层大小并保存在实例中
        self.backbone_hidden_size = config.hidden_size

        # 创建 BrosModel 的实例并保存在实例变量中
        self.bros = BrosModel(config)
        
        # 检查配置对象中是否有 classifier_dropout 属性，如果有则使用其值，否则使用 hidden_dropout_prob 的值（此行代码存在问题，可能需要修正）
        (config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob)

        # 创建 BrosRelationExtractor 的实例并保存在实例变量中
        self.entity_linker = BrosRelationExtractor(config)

        # 调用模型的初始化权重方法
        self.init_weights()

    # 前向传播方法，接受多个输入参数，并按照给定的格式返回结果
    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox_first_token_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        返回值的类型标注，可以是一个包含 torch.Tensor 的元组或者 TokenClassifierOutput 对象。

        Returns:
            返回模型预测的输出结果。

        Examples:
        示例代码展示了如何使用该方法进行预测和处理输出结果。

        ```
        >>> import torch
        >>> from transformers import BrosProcessor, BrosSpadeELForTokenClassification

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosSpadeELForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(**encoding)
        ```"""

        # 检查是否使用用户指定的 return_dict，若未指定则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的主要推断方法，传入各种输入参数
        outputs = self.bros(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取模型输出中的最后一个隐藏状态，并转置以便进行后续处理
        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.transpose(0, 1).contiguous()

        # 使用实体链接器对最后一个隐藏状态进行实体链接，得到最终的预测 logits
        logits = self.entity_linker(last_hidden_states, last_hidden_states).squeeze(0)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            # 获取批处理大小和最大序列长度
            batch_size, max_seq_length = attention_mask.shape
            device = attention_mask.device

            # 创建自我注意力掩码，用于过滤掉当前标记的自注意力
            self_token_mask = torch.eye(max_seq_length, max_seq_length + 1).to(device).bool()

            # 创建 bbox_first_token_mask，用于标记第一个标记是否包含 bbox
            mask = bbox_first_token_mask.view(-1)
            bbox_first_token_mask = torch.cat(
                [
                    ~bbox_first_token_mask,
                    torch.zeros([batch_size, 1], dtype=torch.bool).to(device),
                ],
                axis=1,
            )
            # 使用最小的浮点数填充 logits，以便在损失计算中忽略这些位置
            logits = logits.masked_fill(bbox_first_token_mask[:, None, :], torch.finfo(logits.dtype).min)
            logits = logits.masked_fill(self_token_mask[None, :, :], torch.finfo(logits.dtype).min)

            # 计算损失值
            loss = loss_fct(logits.view(-1, max_seq_length + 1)[mask], labels.view(-1)[mask])

        # 如果不需要返回字典形式的输出，则返回一个元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则返回 TokenClassifierOutput 对象，其中包含损失值、预测 logits、隐藏状态和注意力值
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```