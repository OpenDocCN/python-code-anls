# `.\models\megatron_bert\modeling_megatron_bert.py`

```py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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
""" PyTorch MegatronBERT model."""


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_megatron_bert import MegatronBertConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MegatronBertConfig"
_CHECKPOINT_FOR_DOC = "nvidia/megatron-bert-cased-345m"

MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/megatron-bert-cased-345m",
    # See all MegatronBERT models at https://huggingface.co/models?filter=megatron_bert
]


def load_tf_weights_in_megatron_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow checkpoint 文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    # 初始化变量名称列表和数组
    names = []
    arrays = []
    # 遍历初始化变量的列表，每个变量由名称和形状组成
    for name, shape in init_vars:
        # 记录日志，显示正在加载的 TensorFlow 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用 TensorFlow API 加载指定路径下的变量数据
        array = tf.train.load_variable(tf_path, name)
        # 将变量名称添加到名称列表
        names.append(name)
        # 将加载的变量数据添加到数组列表
        arrays.append(array)

    # 遍历名称列表和数组列表，分别为权重变量名和对应的数据数组
    for name, array in zip(names, arrays):
        # 将变量名称按 '/' 分割为列表
        name = name.split("/")
        
        # 检查名称列表中是否包含特定的变量名，如果包含则跳过当前循环
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 记录日志，显示跳过加载的变量名称
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        
        # 初始化指针为模型对象
        pointer = model
        
        # 遍历变量名列表
        for m_name in name:
            # 如果变量名符合特定格式，按指定规则分割名称
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            
            # 根据名称的第一个部分选择不同的操作
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                # 尝试获取指定名称的属性，如果失败则记录日志并跳过当前循环
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            
            # 如果名称列表长度大于等于2，表示有额外的数字部分
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        
        # 检查变量名的结尾是否为 "_embeddings"
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            # 如果变量名为 "kernel"，将数组转置
            array = np.transpose(array)
        
        # 检查指针和数组的形状是否匹配，如果不匹配则引发 ValueError
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        
        # 记录日志，显示正在初始化的 PyTorch 权重的名称
        logger.info("Initialize PyTorch weight {}".format(name))
        # 将数组转换为 PyTorch 张量，并赋值给指针的数据部分
        pointer.data = torch.from_numpy(array)

    # 返回经过初始化后的模型对象
    return model
class MegatronBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 定义词嵌入层，将词索引映射为隐藏表示向量，支持填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 定义位置嵌入层，将位置索引映射为隐藏表示向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 定义标记类型嵌入层，将标记类型索引映射为隐藏表示向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        # 在 Megatron 中，LayerNorm 在第一个 dropout 之后应用。
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 注册一个缓冲区变量，存储从 0 到 config.max_position_embeddings-1 的位置索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # 如果未提供位置索引，则使用预先注册的位置 ids，从 past_key_values_length 到 seq_length + past_key_values_length
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            # 如果未提供标记类型索引，则使用全零的张量，形状与输入一致
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果未提供输入嵌入，通过词嵌入层获得嵌入表示
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取标记类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和标记类型嵌入相加作为最终的嵌入表示
        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            # 如果位置嵌入类型为绝对位置编码，则添加绝对位置嵌入
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Megatron BERT 将 LayerNorm 移动到 dropout 后面（以及每个层中）。
        # embeddings = self.LayerNorm(embeddings)
        # 应用 dropout
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->MegatronBert
class MegatronBertSelfAttention(nn.Module):
    # 初始化函数，用于初始化一个注意力机制模型实例
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 检查隐藏层大小是否能够整除注意力头的数量，如果不能整除且没有嵌入大小参数，则引发数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头的数量和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout层，用于注意力概率的随机丢弃
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入的类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对位置嵌入，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 标记当前模型是否为解码器
        self.is_decoder = config.is_decoder
# 基于 transformers.models.bert.modeling_bert.BertSelfOutput。将 LayerNorm 移到下面的 MegatronBertAttention 中。
class MegatronBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化一个 dropout 层，使用 config.hidden_dropout_prob 的概率进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 应用全连接层
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 返回残差连接后的结果
        return residual + hidden_states


# 基于 transformers.models.bert.modeling_bert.BertAttention。添加了 LayerNorm。
class MegatronBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 LayerNorm 层，输入维度为 config.hidden_size，epsilon 参数为 config.layer_norm_eps
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 MegatronBertSelfAttention 层
        self.self = MegatronBertSelfAttention(config)
        # 初始化 MegatronBertSelfOutput 层
        self.output = MegatronBertSelfOutput(config)
        # 初始化一个集合，用于存储被修剪的注意力头的索引
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可修剪的头部和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的头部
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
        # 对输入的 hidden_states 应用 LayerNorm 层
        ln_outputs = self.ln(hidden_states)
        # 将经过 LayerNorm 处理后的输出传递给 MegatronBertSelfAttention 层处理
        self_outputs = self.self(
            ln_outputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将 MegatronBertSelfOutput 处理后的结果与原始 hidden_states 相加得到 attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力矩阵，则在 outputs 中添加它们
        outputs = (attention_output,) + self_outputs[1:]
        # 返回最终的输出元组
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制，修改为 Bert->MegatronBert
class MegatronBertIntermediate(nn.Module):
    # 初始化方法，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # 判断 config.hidden_act 是否是字符串类型
        if isinstance(config.hidden_act, str):
            # 如果是字符串，则从预定义的映射 ACT2FN 中获取对应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 如果不是字符串，则直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受一个 torch.Tensor 类型的 hidden_states，返回一个 torch.Tensor 类型的结果
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对输入的 hidden_states 进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的 hidden_states 应用激活函数 intermediate_act_fn
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回激活后的结果
        return hidden_states
# 基于 transformers.models.bert.modeling_bert.BertOutput。将 LayerNorm 移动到下面的 MegatronBertLayer 中。
class MegatronBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入的特征维度转换为隐藏层的维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 添加一个用于随机失活的层，以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行随机失活处理
        hidden_states = self.dropout(hidden_states)
        # 返回经过线性变换和随机失活处理后的隐藏状态与输入张量的和
        return input_tensor + hidden_states


# 基于 transformers.models.bert.modeling_bert.BertLayer。添加了 LayerNorm。
class MegatronBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置用于前向传播分块处理的大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度索引
        self.seq_len_dim = 1
        # 创建注意力层对象
        self.attention = MegatronBertAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨注意力，需要作为解码器模型使用
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建跨注意力层对象
            self.crossattention = MegatronBertAttention(config)
        # 使用 LayerNorm 对隐藏状态进行归一化处理
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Bert 中间层对象
        self.intermediate = MegatronBertIntermediate(config)
        # 创建 Bert 输出层对象
        self.output = MegatronBertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        ):
        # 在前向传播中，依次通过注意力层、LayerNorm、中间层和输出层处理隐藏状态
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果过去的键/值对不为空，则获取自注意力部分的前两个位置的缓存
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力模型处理隐藏状态，得到自注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 提取自注意力输出的主要输出部分
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 排除最后一个元素，因为它是缓存的结构
            outputs = self_attention_outputs[1:-1]
            # 获取当前注意力的键/值对
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，输出包括自注意力权重
            outputs = self_attention_outputs[1:]  # 添加自注意力权重
          
        # 交叉注意力的当前键/值对默认为空
        cross_attn_present_key_value = None
        # 如果是解码器且存在编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 检查是否存在交叉注意力层
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力的过去键/值对在过去键/值对元组的第三和第四位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力模型处理自注意力输出和编码器的相关信息
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 提取交叉注意力的主要输出部分
            attention_output = cross_attention_outputs[0]
            # 添加交叉注意力权重到输出列表中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力缓存添加到当前键/值对的第三和第四位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用于前馈网络的函数，并根据需要分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将层输出添加到输出元组中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值对作为最后的输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 对注意力输出进行层归一化处理
        ln_output = self.ln(attention_output)
        # 进行前馈网络的中间层处理
        intermediate_output = self.intermediate(ln_output)
        # 输出最终的层输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 定义 MegatronBertEncoder 类，继承自 nn.Module
class MegatronBertEncoder(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 将 config 存储在实例中
        self.config = config
        # 创建一个包含多个 MegatronBertLayer 实例的列表，列表长度为 config.num_hidden_layers
        self.layer = nn.ModuleList([MegatronBertLayer(config) for _ in range(config.num_hidden_layers)])

        # 最终的层归一化层。我们删除了第一个 LN，将 LN 移动到每个隐藏层以及此层
        # 这只是最终的 LN（Transformer 的 BERT 附加到每个隐藏层）。
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    # 前向传播方法
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
# 从 transformers.models.bert.modeling_bert.BertPooler 复制，将 Bert->MegatronBert
class MegatronBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 Tanh 激活函数
        self.activation = nn.Tanh()

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地提取与第一个令牌对应的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        # 经过线性层
        pooled_output = self.dense(first_token_tensor)
        # 经过激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从 transformers.models.bert.modeling_bert.BertPredictionHeadTransform 复制，将 Bert->MegatronBert
class MegatronBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果 config.hidden_act 是字符串，则使用 ACT2FN 字典中对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建一个层归一化层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层
        hidden_states = self.dense(hidden_states)
        # 经过激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 经过层归一化层
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertLMPredictionHead 复制，将 Bert->MegatronBert
class MegatronBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MegatronBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入的权重相同，但每个标记都有一个仅用于输出的偏置项。
        # 创建一个线性层，用于将隐藏状态映射到词汇表大小的输出空间，没有偏置项。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 创建一个可学习的偏置参数，大小与词汇表大小相同。
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接，以便偏置项能够正确地随 `resize_token_embeddings` 被调整大小。
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 使用预定义的变换对隐藏状态进行转换
        hidden_states = self.transform(hidden_states)
        # 将转换后的隐藏状态输入到线性层中，得到输出
        hidden_states = self.decoder(hidden_states)
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertOnlyMLMHead 复制而来，将 Bert 替换为 MegatronBert
class MegatronBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 MegatronBertLMPredictionHead 初始化预测模块
        self.predictions = MegatronBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 基于序列输出计算预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从 transformers.models.bert.modeling_bert.BertOnlyNSPHead 复制而来，将 Bert 替换为 MegatronBert
class MegatronBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层初始化序列关系预测模块
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        # 基于池化输出计算序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# 从 transformers.models.bert.modeling_bert.BertPreTrainingHeads 复制而来，将 Bert 替换为 MegatronBert
class MegatronBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 MegatronBertLMPredictionHead 初始化预测模块
        self.predictions = MegatronBertLMPredictionHead(config)
        # 使用线性层初始化序列关系预测模块
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 基于序列输出计算预测分数
        prediction_scores = self.predictions(sequence_output)
        # 基于池化输出计算序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# MegatronBertPreTrainedModel 类，为预训练模型提供权重初始化和简单的预训练模型加载接口
class MegatronBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 MegatronBertConfig 进行配置
    config_class = MegatronBertConfig
    # 使用 load_tf_weights_in_megatron_bert 进行加载 TensorFlow 权重
    load_tf_weights = load_tf_weights_in_megatron_bert
    # 设置基础模型前缀为 "bert"
    base_model_prefix = "bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用正态分布初始化权重，与 TensorFlow 版本稍有不同，后者使用截断正态分布
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 的偏置项初始化为零，权重初始化为一
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 线性层的偏置项初始化为零
            module.bias.data.zero_()


@dataclass
# 从 transformers.models.bert.modeling_bert.BertForPreTrainingOutput 复制而来，将 Bert 替换为 MegatronBert
class MegatronBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`MegatronBertForPreTraining`].
    """
    # 可选参数：如果提供了 `labels`，则返回损失值，类型为 `torch.FloatTensor`，形状为 `(1,)`
    loss: Optional[torch.FloatTensor] = None
    # 语言建模头部的预测得分，形状为 `(batch_size, sequence_length, config.vocab_size)`
    prediction_logits: torch.FloatTensor = None
    # 下一个序列预测（分类）头部的预测得分，形状为 `(batch_size, 2)`
    seq_relationship_logits: torch.FloatTensor = None
    # 可选参数：如果 `output_hidden_states=True` 或 `config.output_hidden_states=True`，返回模型每层的隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选参数：如果 `output_attentions=True` 或 `config.output_attentions=True`，返回注意力权重
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# MEGATRON_BERT_START_DOCSTRING 变量包含了关于 Megatron-BERT 模型的文档字符串，描述了其继承自 PreTrainedModel 的特性，
# 并提供了关于如何使用这个模型的基本信息，包括参数配置和模型行为的说明。

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MegatronBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.



# MEGATRON_BERT_INPUTS_DOCSTRING 变量当前为空字符串，应该用来描述 Megatron-BERT 模型的输入说明文档。

MEGATRON_BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 `AutoTokenizer` 获得这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，避免在填充标记索引上执行注意力操作。遮罩值在 `[0, 1]` 范围内选择：
            # - 1 表示不遮罩的标记，
            # - 0 表示遮罩的标记。
            # [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引在 `[0, 1]` 中选择：
            # - 0 对应 *句子 A* 的标记，
            # - 1 对应 *句子 B* 的标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围是 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的特定头部置零的遮罩。遮罩值在 `[0, 1]` 范围内选择：
            # - 1 表示头部未被遮罩，
            # - 0 表示头部被遮罩。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，可以直接传递嵌入表示而不是 `input_ids`。如果想更精细地控制如何将 `input_ids` 索引转换为关联向量，这将很有用。
            # 这对于比模型内部嵌入查找矩阵更有控制的情况很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
@add_start_docstrings(
    "The bare MegatronBert Model transformer outputting raw hidden-states without any specific head on top.",
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertModel(MegatronBertPreTrainedModel):
    """
    MegatronBertModel类继承自MegatronBertPreTrainedModel，代表一个裸的MegatronBert模型，输出没有特定头部的原始隐藏状态。

    这个模型可以作为编码器（只有自注意力）或解码器使用。当作为解码器时，在自注意力层之间会添加一个交叉注意力层，遵循[Attention is
    all you need](https://arxiv.org/abs/1706.03762)中描述的架构，作者包括Ashish Vaswani、Noam Shazeer、Niki Parmar、
    Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Lukasz Kaiser和Illia Polosukhin。

    要作为解码器使用，需要用`is_decoder`参数设置为`True`来初始化模型配置。要在Seq2Seq模型中使用，需要用`is_decoder`和
    `add_cross_attention`参数都设置为`True`来初始化；此时前向传播期望一个`encoder_hidden_states`作为输入。
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化嵌入层和编码器
        self.embeddings = MegatronBertEmbeddings(config)
        self.encoder = MegatronBertEncoder(config)

        # 如果add_pooling_layer为True，初始化池化层
        self.pooler = MegatronBertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型中的注意力头。heads_to_prune: {layer_num: 要在该层剪枝的头列表} 参见基类PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
MegatronBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
`next sentence prediction (classification)` head.
"""
# 声明一个 MegatronBertForPreTraining 类，继承自 MegatronBertPreTrainedModel 类
class MegatronBertForPreTraining(MegatronBertPreTrainedModel):
    # 定义一个列表，包含了与权重绑定相关的键值
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化函数，接受配置对象 config 和一个布尔型参数 add_binary_head
    def __init__(self, config, add_binary_head=True):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建一个 MegatronBertModel 对象
        self.bert = MegatronBertModel(config)
        # 创建一个 MegatronBertPreTrainingHeads 对象
        self.cls = MegatronBertPreTrainingHeads(config)

        # 调用对象的后初始化方法
        self.post_init()

    # 获取输出嵌入的方法
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的方法，接受一个新的嵌入张量 new_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数，具体功能请参考文档中的说明
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MegatronBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        next_sentence_label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 函数的具体实现由装饰器提供，用于替换文档字符串和返回值的描述

"""
MegatronBert Model with a `language modeling` head on top for CLM fine-tuning.
"""
# 声明一个 MegatronBertForCausalLM 类，继承自 MegatronBertPreTrainedModel 类
class MegatronBertForCausalLM(MegatronBertPreTrainedModel):
    # 定义一个列表，包含了与权重绑定相关的键值
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化函数，接受配置对象 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置对象不是解码器，发出警告信息
        if not config.is_decoder:
            logger.warning("If you want to use `MegatronBertForCausalLM` as a standalone, add `is_decoder=True.`")

        # 创建一个 MegatronBertModel 对象，关闭额外的池化层
        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        # 创建一个 MegatronBertOnlyMLMHead 对象
        self.cls = MegatronBertOnlyMLMHead(config)

        # 调用对象的后初始化方法
        self.post_init()

    # 获取输出嵌入的方法
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的方法，接受一个新的嵌入张量 new_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数，具体功能请参考文档中的说明
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此方法用于模型的前向传播，接收多个输入参数并返回模型输出
        # 可选参数中包括输入张量、注意力掩码、token类型ID、位置ID等
        # 返回包括预测标签、隐藏状态等，具体返回方式由return_dict参数控制
        pass

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 准备生成过程中的输入，根据输入ID、过去键值等准备生成所需的输入格式
        input_shape = input_ids.shape

        # 如果没有给定注意力掩码，则创建全1的注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了过去的键值（past_key_values），则调整输入ID，移除前缀部分
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：仅保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含输入ID、注意力掩码和过去键值的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        # 重新排序缓存中的过去键值，以适应beam搜索的顺序
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每一层的过去状态进行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 使用装饰器为 MegatronBertForMaskedLM 类添加文档字符串，描述其作为 MegatronBert 模型并带有语言建模头部的特性
@add_start_docstrings("""MegatronBert Model with a `language modeling` head on top.""", MEGATRON_BERT_START_DOCSTRING)
class MegatronBertForMaskedLM(MegatronBertPreTrainedModel):
    # 定义权重共享的关键字列表
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置标记为解码器，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `MegatronBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 MegatronBertModel 实例，禁用池化层
        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        # 创建 MegatronBertOnlyMLMHead 实例
        self.cls = MegatronBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输出嵌入层的方法
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层的方法，接受新的嵌入层作为参数
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接受多个输入和控制参数
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 初始化返回字典，如果未提供则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用BERT模型进行前向传播
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从BERT模型输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入分类器，得到预测得分
        prediction_scores = self.cls(sequence_output)

        # 初始化masked_lm_loss为None
        masked_lm_loss = None

        # 如果提供了标签，则计算masked language modeling损失
        if labels is not None:
            # 使用交叉熵损失函数，忽略标签为-100的token（padding token）
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果return_dict为False，则按非字典方式返回结果
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果return_dict为True，则按MaskedLMOutput对象方式返回结果
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        # 获取输入的形状信息和有效的批量大小
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # 如果未定义pad_token_id，则无法进行生成，抛出异常
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        # 在attention_mask末尾添加一个虚拟token
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)

        # 创建一个全是pad_token_id的虚拟token张量
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )

        # 将虚拟token添加到输入ids的末尾
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回生成模型需要的输入字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 使用装饰器添加文档字符串，描述了 MegatronBertForNextSentencePrediction 类的作用及其顶部的文档信息
@add_start_docstrings(
    """MegatronBert Model with a `next sentence prediction (classification)` head on top.""",
    MEGATRON_BERT_START_DOCSTRING,
)
# 定义 MegatronBertForNextSentencePrediction 类，继承自 MegatronBertPreTrainedModel
class MegatronBertForNextSentencePrediction(MegatronBertPreTrainedModel):
    
    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 MegatronBertModel 实例，并赋值给 self.bert
        self.bert = MegatronBertModel(config)
        
        # 创建 MegatronBertOnlyNSPHead 实例，并赋值给 self.cls
        self.cls = MegatronBertOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加文档字符串，描述了 forward 方法的输入参数及其作用
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回文档字符串，指定输出类型为 NextSentencePredictorOutput，配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，接受多个输入参数和 **kwargs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:
            
            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
    
        Returns:
            Depending on `return_dict`:
            - If `return_dict=False` (default): returns a tuple with `seq_relationship_scores` followed by `outputs[2:]`.
            - If `return_dict=True`: returns a `NextSentencePredictorOutput` containing loss, logits, hidden states, and attentions.
    
        Example:
        ```
        >>> from transformers import AutoTokenizer, MegatronBertForNextSentencePrediction
        >>> import torch
    
        >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/megatron-bert-cased-345m")
        >>> model = MegatronBertForNextSentencePrediction.from_pretrained("nvidia/megatron-bert-cased-345m")
    
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
    
        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
    
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # Pass input tensors through the BERT model to get outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # Get the pooled output from BERT's outputs
        pooled_output = outputs[1]
    
        # Predict next sentence relationship using a classifier layer
        seq_relationship_scores = self.cls(pooled_output)
    
        next_sentence_loss = None
        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))
    
        # Return outputs based on `return_dict` flag
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output
    
        # Return a `NextSentencePredictorOutput` object if `return_dict=True`
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    MegatronBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertForSequenceClassification(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 Bert 模型和相关组件
        self.bert = MegatronBertModel(config)
        # Dropout 层，用于减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器，线性层，将 BERT 输出映射到标签空间
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行后续处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播函数，处理输入并生成模型输出。

        Args:
            input_ids (Optional[torch.LongTensor], optional): 输入的 token IDs. Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): 注意力掩码，指示哪些元素是填充的. Defaults to None.
            token_type_ids (Optional[torch.LongTensor], optional): token 类型 IDs，区分 segment A 和 segment B. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): token 的位置 IDs. Defaults to None.
            head_mask (Optional[torch.FloatTensor], optional): 多头注意力机制的掩码. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): 嵌入式表示的输入. Defaults to None.
            labels (Optional[torch.LongTensor], optional): 标签，用于计算损失. Defaults to None.
            output_attentions (Optional[bool], optional): 是否返回注意力权重. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否返回所有隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否以字典形式返回输出. Defaults to None.

        Returns:
            SequenceClassifierOutput: 包含模型输出和损失的对象
        """
        # BERT 模型的 forward 方法，处理输入并生成模型输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]  # 取出池化输出，通常用于分类任务

        pooled_output = self.dropout(pooled_output)  # 应用 dropout 防止过拟合
        logits = self.classifier(pooled_output)  # 使用线性分类器映射到标签空间

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化返回字典，如果未提供则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给BERT模型进行处理，并获取其输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从BERT模型的输出中获取汇聚的输出表示
        pooled_output = outputs[1]

        # 对汇聚的输出表示进行dropout操作
        pooled_output = self.dropout(pooled_output)

        # 将dropout后的输出传递给分类器，得到预测的logits
        logits = self.classifier(pooled_output)

        # 初始化损失为None
        loss = None

        # 如果提供了标签，则计算相应的损失
        if labels is not None:
            # 如果问题类型未定义，则根据标签类型和类数自动推断问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择合适的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典，则按照非字典返回格式组织输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则创建SequenceClassifierOutput对象，并返回
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义 MegatronBertForMultipleChoice 类，继承自 MegatronBertPreTrainedModel，用于多项选择任务的 Megatron-BERT 模型
@add_start_docstrings(
    """
    MegatronBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output
    and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertForMultipleChoice(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Megatron-BERT 模型
        self.bert = MegatronBertModel(config)
        # Dropout 层，用于随机断开神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器，线性层，将 BERT 隐藏层的输出映射到一个值，用于多项选择的分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接收多个输入和控制参数，返回模型输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数参数说明文档
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据需要确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入张量的选择数
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重新调整输入张量的形状，将其视为二维张量
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果存在输入嵌入，则将其视为三维张量
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用BERT模型处理输入
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取池化后的输出
        pooled_output = outputs[1]

        # 对池化后的输出进行dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器预测logits
        logits = self.classifier(pooled_output)
        # 调整logits的形状以匹配选择数
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果有提供标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不要求返回字典格式的输出，则按元组格式返回结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典格式的输出，则创建MultipleChoiceModelOutput对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
MegatronBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.
"""
@add_start_docstrings(
    """
    MegatronBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertForTokenClassification(MegatronBertPreTrainedModel):
    def __init__(self, config):
        """
        Initialize the MegatronBertForTokenClassification model.

        Args:
            config (MegatronBertConfig): Configuration object specifying the model architecture and hyperparameters.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize the MegatronBertModel with pooling layer excluded
        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        """
        Forward pass of the MegatronBertForTokenClassification model.

        Args:
            input_ids (torch.LongTensor, optional): Tensor of shape `(batch_size, sequence_length)` containing input token IDs.
            attention_mask (torch.FloatTensor, optional): Tensor of shape `(batch_size, sequence_length)` containing attention masks.
            token_type_ids (torch.LongTensor, optional): Tensor of shape `(batch_size, sequence_length)` containing token type IDs.
            position_ids (torch.LongTensor, optional): Tensor of shape `(batch_size, sequence_length)` containing position IDs.
            head_mask (torch.FloatTensor, optional): Tensor of shape `(batch_size, sequence_length)` containing attention head masks.
            inputs_embeds (torch.FloatTensor, optional): Tensor of shape `(batch_size, sequence_length, hidden_size)` containing precomputed embeddings.
            labels (torch.LongTensor, optional): Tensor of shape `(batch_size, sequence_length)` containing labels for computing token classification loss.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return outputs as a dictionary.

        Returns:
            Union[Tuple, TokenClassifierOutput]: Depending on `return_dict`, either a tuple or a `TokenClassifierOutput` object.

        Notes:
            - Labels should be in the range `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform the forward pass through MegatronBertModel
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Apply dropout on the output of the BERT model
        sequence_output = self.dropout(sequence_output)
        
        # Pass the modified output through the classifier layer
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Compute the token classification loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # Prepare output tuple if return_dict is False
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Return TokenClassifierOutput object if return_dict is True
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用自定义的文档字符串描述 MegatronBertForQuestionAnswering 类，它是基于 Megatron-BERT 模型的抽取式问答任务模型，
# 在隐藏状态输出的基础上加上线性层，用于计算 `span start logits` 和 `span end logits`。
@add_start_docstrings(
    """
    MegatronBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertForQuestionAnswering(MegatronBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数目
        self.num_labels = config.num_labels

        # 初始化 Megatron-BERT 模型，不添加池化层
        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        # QA 输出层，线性层，输入为隐藏状态大小，输出为类别数目
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用自定义的文档字符串描述 forward 方法的输入参数和功能
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用代码示例的文档字符串描述 forward 方法的返回值类型和相关配置
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 如果 return_dict 未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 BERT 模型进行前向传播
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 BERT 输出的序列表示
        sequence_output = outputs[0]

        # 将序列表示传递给 QA 输出层，得到起始位置和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # 压缩维度并确保连续存储
        end_logits = end_logits.squeeze(-1).contiguous()  # 压缩维度并确保连续存储

        total_loss = None
        # 如果提供了起始和结束位置，则计算损失
        if start_positions is not None and end_positions is not None:
            # 如果是多 GPU 情况下，需要添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入范围的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数计算起始和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典格式的输出，则按原样返回结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]  # 加入额外的输出
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QuestionAnsweringModelOutput 格式的结果
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```