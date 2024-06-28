# `.\models\splinter\modeling_splinter.py`

```
# coding=utf-8
# Copyright 2021 Tel AViv University, AllenAI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Splinter model."""

# 导入必要的库
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入相关的模块和类
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, ModelOutput, QuestionAnsweringModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_splinter import SplinterConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 文档中使用的检查点和配置名称
_CHECKPOINT_FOR_DOC = "tau/splinter-base"
_CONFIG_FOR_DOC = "SplinterConfig"

# Splinter 预训练模型的存档列表
SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tau/splinter-base",
    "tau/splinter-base-qass",
    "tau/splinter-large",
    "tau/splinter-large-qass",
    # 查看所有 Splinter 模型 https://huggingface.co/models?filter=splinter
]

# SplinterEmbeddings 类，用于构建来自单词、位置和令牌类型嵌入的嵌入
class SplinterEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 单词嵌入层，根据配置创建一个单词嵌入层，将词汇表大小、隐藏大小和填充标记索引作为参数
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，根据配置创建一个位置嵌入层，将最大位置嵌入大小和隐藏大小作为参数
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 令牌类型嵌入层，根据配置创建一个令牌类型嵌入层，将类型词汇表大小和隐藏大小作为参数
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 层，用于归一化隐藏状态
        # self.LayerNorm 名称没有使用蛇形命名，以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 丢弃层，用于在训练过程中随机丢弃一些隐藏单元，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) 在序列化时是连续的内存块，并且在导出时被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # position_embedding_type，位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
    # 定义一个前向传播函数，用于处理输入的张量数据和可选的参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID张量，默认为None
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型ID张量，默认为None
        position_ids: Optional[torch.LongTensor] = None,  # 位置ID张量，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入张量，默认为None
        past_key_values_length: Optional[int] = 0,  # 过去的关键值长度，默认为0
    ) -> Tuple:  # 函数返回一个元组

        if input_ids is not None:
            input_shape = input_ids.size()  # 获取输入token ID张量的形状
        else:
            input_shape = inputs_embeds.size()[:-1]  # 获取输入嵌入张量的形状，不包括最后一维度

        seq_length = input_shape[1]  # 获取序列长度

        if position_ids is None:
            # 如果位置ID张量为None，则从self.position_ids中获取对应长度的位置ID张量
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            # 如果token类型ID张量为None，则创建与input_shape相同形状的全零张量
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果输入的嵌入张量为None，则使用self.word_embeddings根据input_ids创建嵌入张量
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # 根据token类型ID获取token类型嵌入张量

        embeddings = inputs_embeds + token_type_embeddings  # 将输入嵌入张量和token类型嵌入张量相加

        if self.position_embedding_type == "absolute":
            # 如果位置嵌入类型是"absolute"，则根据位置ID获取位置嵌入张量并加到embeddings中
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)  # 对嵌入张量进行Layer Normalization
        embeddings = self.dropout(embeddings)  # 对嵌入张量进行Dropout操作
        return embeddings  # 返回处理后的嵌入张量作为输出
# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制的代码，将 Bert 替换为 Splinter
class SplinterSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否是注意力头数的整数倍，如果不是且没有嵌入大小属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为相对键或相对键查询，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器的标志
        self.is_decoder = config.is_decoder

    # 将输入张量重塑为注意力分数的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制的代码，将 Bert 替换为 Splinter
class SplinterSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层（dense）
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过 dropout
        hidden_states = self.dropout(hidden_states)
        # 加上输入张量并通过 LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Splinter
class SplinterAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 SplinterSelfAttention 层，传入配置和位置嵌入类型
        self.self = SplinterSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 SplinterSelfOutput 层，传入配置
        self.output = SplinterSelfOutput(config)
        # 初始化一个空集合，用于存储被修剪的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数，找到可修剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被修剪的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用 self.self 的前向传播方法，处理注意力相关计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用 self.output 的前向传播方法，处理注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果有输出注意力，将其添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Splinter
class SplinterIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层，将隐藏大小映射到中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层变换隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Splinter
class SplinterOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 直接继承 BERT 输出层的初始化方式，这里省略不写
        pass  # Initialization identical to BERT Output layer, omitted here
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，将输入大小设为config中的intermediate_size，输出大小设为config中的hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，对隐藏状态的维度为config中的hidden_size进行归一化，设定epsilon为config中的layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，以config中的hidden_dropout_prob作为dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受两个张量作为输入，返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对经过全连接层后的隐藏状态进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将dropout后的隐藏状态与输入张量相加，然后通过LayerNorm层进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回归一化后的隐藏状态作为输出
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制并修改为Splinter
class SplinterLayer(nn.Module):
    # 初始化方法，设置SplinterLayer的属性
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中用于分块的大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设为1
        self.seq_len_dim = 1
        # 创建SplinterAttention对象，并将其赋值给self.attention
        self.attention = SplinterAttention(config)
        # 是否作为解码器模型
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨注意力
        if self.add_cross_attention:
            # 如果不是解码器模型，则抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建带有绝对位置嵌入类型的SplinterAttention对象，并将其赋值给self.crossattention
            self.crossattention = SplinterAttention(config, position_embedding_type="absolute")
        # 创建SplinterIntermediate对象，并将其赋值给self.intermediate
        self.intermediate = SplinterIntermediate(config)
        # 创建SplinterOutput对象，并将其赋值给self.output

        self.output = SplinterOutput(config)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果过去的键/值缓存不为空，则获取解码器单向自注意力的缓存键/值元组在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力层处理隐藏状态，得到自注意力的输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 输出中去除最后一个元素，因为它是自注意力缓存的元组
            outputs = self_attention_outputs[1:-1]
            # 获取当前的键/值缓存
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，输出中包含自注意力权重
            outputs = self_attention_outputs[1:]  # 如果我们输出注意力权重，则添加自注意力
        

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # 如果传入了编码器隐藏状态，但未实例化交叉注意力层，则引发值错误
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值缓存不为空，则获取交叉注意力的缓存键/值元组在位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力层处理自注意力输出和编码器隐藏状态，得到交叉注意力的输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力层的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力权重添加到输出中
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果我们输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到当前的键/值缓存中的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用到前向分块处理中
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将前向分块处理的输出添加到输出元组中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值作为最后的输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回最终的输出元组
        return outputs

    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间输出和注意力输出，得到层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制代码，并将Bert->Splinter修改为SplinterEncoder类
class SplinterEncoder(nn.Module):
    # 初始化方法，接受一个config对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的config对象保存到实例变量self.config中
        self.config = config
        # 创建一个包含多个SplinterLayer实例的ModuleList，数量由config.num_hidden_layers决定
        self.layer = nn.ModuleList([SplinterLayer(config) for _ in range(config.num_hidden_layers)])
        # 设定梯度检查点标志，默认为False
        self.gradient_checkpointing = False

    # 前向传播方法，接收多个输入参数和一些可选参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不需要输出隐藏状态，则初始化为空元组；否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化为空元组；否则为None
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出交叉注意力权重或者配置不支持，则初始化为空元组；否则为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果开启梯度检查点且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果同时使用缓存，则发出警告并设置use_cache为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果需要使用缓存，则初始化next_decoder_cache为空元组；否则为None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态加入all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取过去的键值对，用于解码器
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则直接调用解码器层的前向传播函数
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前层的隐藏状态为解码器层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的缓存信息加入next_decoder_cache
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的自注意力权重加入all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置支持添加交叉注意力，则将当前层的交叉注意力权重加入all_cross_attentions
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终隐藏状态加入all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出结果，则返回一个元组
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则返回一个BaseModelOutputWithPastAndCrossAttentions对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    """
    SPLINTER_INPUTS_DOCSTRING定义了一个用于SPLINTER模型输入说明文档的字符串常量。
    """
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        
        attention_mask (`torch.FloatTensor` of shape `{0}`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        
        token_type_ids (`torch.LongTensor` of shape `{0}`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            
            [What are token type IDs?](../glossary#token-type-ids)
        
        position_ids (`torch.LongTensor` of shape `{0}`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            
            [What are position IDs?](../glossary#position-ids)
        
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare Splinter Model transformer outputting raw hidden-states without any specific head on top.",
    SPLINTER_START_DOCSTRING,
)
class SplinterModel(SplinterPreTrainedModel):
    """
    The model is an encoder (with only self-attention) following the architecture described in [Attention is all you
    need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize embeddings and encoder layers
        self.embeddings = SplinterEmbeddings(config)
        self.encoder = SplinterEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # Iterate over layers and prune specified heads in attention layers
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SPLINTER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the SplinterModel. See superclass for more details.
        """
        # Implement the forward pass using the specified inputs and return model outputs
        # including attentions, hidden states, and past key values if applicable
        ...
    """
    实现了基于问题感知的跨度选择（QASS）头部，参考Splinter的论文描述。
    
    """
    
    # 初始化函数，设置QASS模型的各个层
    def __init__(self, config):
        super().__init__()
    
        # 定义转换层，将输入特征转换为问题起始和结束的表示
        self.query_start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.query_end_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        
        # 定义转换层，将输入特征转换为序列起始和结束的表示
        self.start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.end_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
    
        # 定义分类器层，用于预测起始位置和结束位置的概率分布
        self.start_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.end_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    
    # 前向传播函数，接收输入和位置信息，返回起始位置和结束位置的预测 logits
    def forward(self, inputs, positions):
        _, _, dim = inputs.size()
        index = positions.unsqueeze(-1).repeat(1, 1, dim)  # 创建位置索引张量 [batch_size, num_positions, dim]
        gathered_reps = torch.gather(inputs, dim=1, index=index)  # 根据位置索引收集输入特征表示 [batch_size, num_positions, dim]
    
        # 对问题起始和结束的特征表示进行变换
        query_start_reps = self.query_start_transform(gathered_reps)  # [batch_size, num_positions, dim]
        query_end_reps = self.query_end_transform(gathered_reps)  # [batch_size, num_positions, dim]
        
        # 对序列起始和结束的特征表示进行变换
        start_reps = self.start_transform(inputs)  # [batch_size, seq_length, dim]
        end_reps = self.end_transform(inputs)  # [batch_size, seq_length, dim]
    
        # 使用分类器预测起始位置的 logits
        hidden_states = self.start_classifier(query_start_reps)  # [batch_size, num_positions, dim]
        start_reps = start_reps.permute(0, 2, 1)  # 调整维度顺序为 [batch_size, dim, seq_length]
        start_logits = torch.matmul(hidden_states, start_reps)  # 计算起始位置的 logits
    
        # 使用分类器预测结束位置的 logits
        hidden_states = self.end_classifier(query_end_reps)  # [batch_size, num_positions, dim]
        end_reps = end_reps.permute(0, 2, 1)  # 调整维度顺序为 [batch_size, dim, seq_length]
        end_logits = torch.matmul(hidden_states, end_reps)  # 计算结束位置的 logits
    
        # 返回起始位置和结束位置的预测 logits
        return start_logits, end_logits
# 使用装饰器添加文档字符串到类定义，描述了这个类在提取式问答任务（如SQuAD）中的作用，特别是在隐藏状态输出之上有一个线性层来计算“起始位置logits”和“结束位置logits”。
@add_start_docstrings(
    """
    Splinter Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    SPLINTER_START_DOCSTRING,  # 添加了预定义的起始文档字符串
)

# 声明一个新的类，继承自SplinterPreTrainedModel，用于问题回答的Splinter模型
class SplinterForQuestionAnswering(SplinterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化Splinter模型
        self.splinter = SplinterModel(config)
        # 初始化用于问题感知的跨度选择头部
        self.splinter_qass = QuestionAwareSpanSelectionHead(config)
        # 设置问题的token ID
        self.question_token_id = config.question_token_id

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加文档字符串到模型前向传播函数，描述了输入参数及其作用
    @add_start_docstrings_to_model_forward(SPLINTER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码示例文档字符串，包括检查点、输出类型和配置类信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播函数定义，处理各种输入参数并返回输出结果
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        question_positions: Optional[torch.LongTensor] = None,
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when start and end positions are provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, num_questions, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, num_questions, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # Optional: A scalar tensor representing the total loss for span extraction tasks.
    loss: Optional[torch.FloatTensor] = None
    # Optional: Tensor containing scores for the start positions of spans in each question.
    start_logits: torch.FloatTensor = None
    # Optional: Tensor containing scores for the end positions of spans in each question.
    end_logits: torch.FloatTensor = None
    # Optional: Tuple of tensors representing hidden states of the model at each layer and embeddings.
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # Optional: Tuple of tensors representing attention weights for each layer's self-attention mechanism.
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个 Splinter 模型，用于预训练中的重复跨度选择任务。与 QA 任务不同的是，这里没有问题，而是多个问题标记，
# 这些标记替换了重复跨度的出现。
@add_start_docstrings(
    """
    Splinter Model for the recurring span selection task as done during the pretraining. The difference to the QA task
    is that we do not have a question, but multiple question tokens that replace the occurrences of recurring spans
    instead.
    """,
    SPLINTER_START_DOCSTRING,
)
class SplinterForPreTraining(SplinterPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        # 初始化 Splinter 模型和问题感知的跨度选择头部
        self.splinter = SplinterModel(config)
        self.splinter_qass = QuestionAwareSpanSelectionHead(config)
        self.question_token_id = config.question_token_id
        
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        SPLINTER_INPUTS_DOCSTRING.format("batch_size, num_questions, sequence_length")
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        question_positions: Optional[torch.LongTensor] = None,
    ):
        # 模型的前向传播方法，接收多个输入参数，并输出模型的预测或特征表示

    def _prepare_question_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 准备问题位置的方法，根据输入的 input_ids 返回一个 torch.Tensor，表示问题在输入中的位置
        
        # 在 input_ids 中寻找与 self.config.question_token_id 相等的位置
        rows, flat_positions = torch.where(input_ids == self.config.question_token_id)
        
        # 统计每行中的问题数量
        num_questions = torch.bincount(rows)
        
        # 创建一个形状为 (batch_size, 最大问题数量) 的 positions 张量，并用 pad_token_id 初始化
        positions = torch.full(
            (input_ids.size(0), num_questions.max()),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        
        # 根据 flat_positions 将问题位置填充到 positions 张量中
        cols = torch.cat([torch.arange(n) for n in num_questions])
        positions[rows, cols] = flat_positions
        
        # 返回填充好的 positions 张量
        return positions
```