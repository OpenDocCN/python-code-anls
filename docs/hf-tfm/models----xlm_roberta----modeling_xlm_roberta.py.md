# `.\models\xlm_roberta\modeling_xlm_roberta.py`

```
# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch XLM-RoBERTa model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_xlm_roberta import XLMRobertaConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "FacebookAI/xlm-roberta-base"
_CONFIG_FOR_DOC = "XLMRobertaConfig"

# XLM-RoBERTa 模型的预训练模型存档列表
XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",
    "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
    "FacebookAI/xlm-roberta-large-finetuned-conll03-german",
    # 查看所有 XLM-RoBERTa 模型：https://huggingface.co/models?filter=xlm-roberta
]


# 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 复制而来，用于 XLM-RoBERTa
class XLMRobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # 从 transformers.models.bert.modeling_bert.BertEmbeddings.__init__ 复制而来
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层，将词汇表大小、隐藏层大小和填充索引作为参数
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，将最大位置嵌入大小和隐藏层大小作为参数
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建token类型嵌入层，将类型词汇表大小和隐藏层大小作为参数
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用给定的隐藏层大小和epsilon值创建LayerNorm层，以与TensorFlow模型变量名保持一致
        # 可以加载任何TensorFlow检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建dropout层，使用给定的隐藏层dropout概率作为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 设置位置嵌入类型，默认为"absolute"，从配置中获取
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区，创建位置ID张量，长度为最大位置嵌入大小，并在序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区，创建token类型ID张量，形状与位置ID相同，数据类型为长整型
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 设置填充索引为配置文件中的填充token ID
        self.padding_idx = config.pad_token_id
        # 重新创建位置嵌入层，使用最大位置嵌入大小、隐藏层大小和填充索引作为参数
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
    ):
        # 如果没有指定位置编码，但指定了输入的 token ids
        if position_ids is None:
            # 根据输入的 token ids 创建位置编码。任何填充的 token 保持填充状态。
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
        else:
            # 如果未指定输入的 token ids，则根据输入的嵌入张量创建位置编码
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果指定了输入的 token ids
        if input_ids is not None:
            # 获取输入张量的形状
            input_shape = input_ids.size()
        else:
            # 获取输入嵌入张量的形状，不包括最后一个维度（通常是 batch 维度）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，即输入的 token 序列的第二个维度大小
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，该缓冲区全部为零。这通常在自动生成时发生，
        # 注册的缓冲区有助于在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 获取模型中已注册的 token_type_ids 缓冲区，并截取到与序列长度相匹配的部分
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                # 扩展缓冲区的 token_type_ids 到与输入形状相匹配的大小
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果模型中没有注册 token_type_ids 缓冲区，则创建一个全零的张量
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果没有提供 inputs_embeds 参数，则通过输入的 token ids 获取对应的嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 根据 token_type_ids 获取对应的 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和 token 类型嵌入相加作为总的嵌入
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入类型是 "absolute"，则获取对应的位置嵌入并添加到总的嵌入中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 应用 LayerNorm 对嵌入进行归一化
        embeddings = self.LayerNorm(embeddings)

        # 对归一化后的嵌入进行 dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入张量作为结果
        return embeddings

    # 从输入嵌入张量中创建位置编码
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入张量的形状，不包括最后一个维度（通常是 batch 维度）
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度，即输入嵌入张量的第二个维度大小
        sequence_length = input_shape[1]

        # 根据输入嵌入张量的大小创建顺序的位置编码
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置编码张量扩展为与输入形状相匹配的大小
        return position_ids.unsqueeze(0).expand(input_shape)
# 从 transformers.models.roberta.modeling_roberta.RobertaSelfAttention 复制并将 Roberta 替换为 XLMRoberta
class XLMRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，如果不能且 config 没有 embedding_size 属性，则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建线性层用于计算查询、键和值
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建 dropout 层，用于注意力概率的 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为相对位置编码之一，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder

    # 转置输入张量以适应注意力得分计算的形状
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



# 从 transformers.models.roberta.modeling_roberta.RobertaSelfOutput 复制并将 Roberta 替换为 XLMRoberta
class XLMRobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建全连接层，用于对隐藏状态进行线性变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建 LayerNorm 层，用于对变换后的状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层，用于对归一化后的状态进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的状态进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 后的状态与输入张量进行残差连接，并经过 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 从 transformers.models.roberta.modeling_roberta.RobertaAttention 复制并修改为 XLMRobertaAttention
class XLMRobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 self 注意力机制，使用给定的配置和位置嵌入类型
        self.self = XLMRobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化输出层，用于处理 self 注意力机制的输出
        self.output = XLMRobertaSelfOutput(config)
        # 存储需要剪枝的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的注意力头
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
        # 进行 self 注意力机制的前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用输出层处理 self 注意力机制的输出和原始输入的隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果有需要，添加注意力输出到结果中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要，将注意力加入到输出中
        return outputs


# 从 transformers.models.roberta.modeling_roberta.RobertaIntermediate 复制并修改为 XLMRobertaIntermediate
class XLMRobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，将隐藏状态的大小映射为中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 中间激活函数，根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过线性层映射到中间大小
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理映射后的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.roberta.modeling_roberta.RobertaOutput 复制并修改为 XLMRobertaOutput
class XLMRobertaOutput(nn.Module):
    # 初始化函数，用于初始化对象的各个成员变量
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，对隐藏状态进行归一化，归一化的特征数为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，以config.hidden_dropout_prob的概率将输入置为0，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，处理输入的隐藏状态和输入张量，返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过线性层进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行Dropout操作，以减少过拟合风险
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的隐藏状态与输入张量相加，并通过LayerNorm进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量作为输出
        return hidden_states
```python`
# 从 transformers.models.roberta.modeling_roberta.RobertaLayer 复制并修改，适配 Roberta 到 XLMRoberta
class XLMRobertaLayer(nn.Module):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 feed-forward 的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度，通常为 1
        self.seq_len_dim = 1
        # 初始化 XLMRobertaAttention 层
        self.attention = XLMRobertaAttention(config)
        # 设置是否为解码器
        self.is_decoder = config.is_decoder
        # 设置是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力
        if self.add_cross_attention:
            # 确保如果添加了交叉注意力，则模型必须是解码器模型
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化交叉注意力层
            self.crossattention = XLMRobertaAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = XLMRobertaIntermediate(config)
        # 初始化输出层
        self.output = XLMRobertaOutput(config)

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
        # 定义函数的输入和输出类型，这里函数接收一个参数并返回一个元组，其中元素类型为 torch.Tensor

        # 如果过去的键/值对不为空，则取前两个作为自注意力机制的过去键/值对，否则为 None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        # 使用 self.attention 方法进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，则最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，还需添加自注意力

        cross_attn_present_key_value = None
        # 如果是解码器且有编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果未定义交叉注意力层，抛出 ValueError
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值对不为空，则取倒数两个作为交叉注意力机制的过去键/值对，否则为 None
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            
            # 使用 self.crossattention 方法进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力计算的输出
            attention_output = cross_attention_outputs[0]
            # 添加交叉注意力计算的输出到 outputs 中，如果输出注意力权重
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力缓存添加到当前键/值对的末尾位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对 attention_output 应用分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 将 attention_output 应用于 feed_forward_chunk 的中间层和输出层，返回层输出
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从 transformers.models.roberta.modeling_roberta.RobertaEncoder 复制而来，将 Roberta 替换为 XLMRoberta
class XLMRobertaEncoder(nn.Module):
    # 初始化函数，接收一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 将配置对象保存在实例中
        self.config = config
        # 创建一个层列表，其中每个层都是 XLMRobertaLayer 类的实例，数量等于配置中指定的隐藏层数
        self.layer = nn.ModuleList([XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点设为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接收多个输入参数
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
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

初始化变量 `all_hidden_states`, `all_self_attentions`, `all_cross_attentions`，根据 `output_hidden_states` 和 `output_attentions` 的布尔值确定是否创建空元组或赋值为 `None`。


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

如果启用了梯度检查点并且处于训练模式，且 `use_cache=True`，则发出警告并将 `use_cache` 设为 `False`，因为这两者不兼容。


        next_decoder_cache = () if use_cache else None

根据 `use_cache` 的布尔值确定是否创建空元组 `next_decoder_cache` 或赋值为 `None`。


        for i, layer_module in enumerate(self.layer):

遍历 `self.layer` 中的每个层模块，`i` 是索引，`layer_module` 是当前层模块的引用。


            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

如果 `output_hidden_states` 为真，则将当前 `hidden_states` 添加到 `all_hidden_states` 中。


            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

根据索引 `i` 获取 `head_mask` 和 `past_key_values` 中的对应项，若它们不为 `None`。


            if self.gradient_checkpointing and self.training:
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
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

根据是否启用梯度检查点和是否在训练模式下，调用相应的层模块函数 `_gradient_checkpointing_func` 或 `layer_module`，并将结果赋给 `layer_outputs`。


            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

更新 `hidden_states`，如果 `use_cache` 为真，则将当前层的缓存信息添加到 `next_decoder_cache` 中。如果 `output_attentions` 为真，更新 `all_self_attentions` 和 `all_cross_attentions`，根据是否添加跨层注意力信息。


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

如果 `output_hidden_states` 为真，则将最终的 `hidden_states` 添加到 `all_hidden_states` 中。


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

如果不返回字典形式的输出，返回一个元组，包含所有非空的输出变量。


        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

以字典形式返回模型的输出，使用 `BaseModelOutputWithPastAndCrossAttentions` 类，包含 `hidden_states`、`next_decoder_cache`、`all_hidden_states`、`all_self_attentions` 和 `all_cross_attentions`。
# 从 transformers.models.roberta.modeling_roberta.RobertaPooler 复制并修改为 XLMRobertaPooler 类
class XLMRobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出大小都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个 token 的隐藏状态来进行池化
        first_token_tensor = hidden_states[:, 0]
        # 将池化后的输出通过全连接层处理
        pooled_output = self.dense(first_token_tensor)
        # 对全连接层输出应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output


# 从 transformers.models.roberta.modeling_roberta.RobertaPreTrainedModel 复制并修改为 XLMRobertaPreTrainedModel 类
class XLMRobertaPreTrainedModel(PreTrainedModel):
    """
    处理权重初始化以及预训练模型下载和加载的抽象类。
    """

    # 使用 XLMRobertaConfig 进行配置
    config_class = XLMRobertaConfig
    # 基础模型前缀为 "roberta"
    base_model_prefix = "roberta"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块列表
    _no_split_modules = ["XLMRobertaEmbeddings", "XLMRobertaSelfAttention"]

    # 从 transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights 复制并修改为 _init_weights 方法
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，标准差为 config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置，则初始化为零向量
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，标准差为 config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果存在 padding_idx，则对应的权重初始化为零向量
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的偏置为零向量，权重为全一向量
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# XLM_ROBERTA_START_DOCSTRING 从原文档复制
XLM_ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`XLMRobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# XLM_ROBERTA_INPUTS_DOCSTRING 从原文档复制
XLM_ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。

            # 可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 获取详细信息。

            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于避免在填充标记索引上执行注意力操作。遮罩的取值为 `[0, 1]`：

            # - 1 表示**未遮罩**的标记，
            # - 0 表示**遮罩**的标记。

            # [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引取值为 `[0, 1]`：

            # - 0 对应于 *句子 A* 的标记，
            # - 1 对应于 *句子 B* 的标记。

            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列中每个标记的位置索引，在位置嵌入中使用。索引范围为 `[0, config.max_position_embeddings - 1]`。

            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 遮罩，用于将自注意力模块的某些头部置为零。遮罩的取值为 `[0, 1]`：

            # - 1 表示**未遮罩**的头部，
            # - 0 表示**遮罩**的头部。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选参数，可以直接传递嵌入表示，而不是 `input_ids`。如果希望更好地控制如何将 `input_ids` 索引转换为相关向量，这将非常有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 以获取更多详细信息。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回张量中的 `hidden_states` 以获取更多详细信息。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta.RobertaModel中复制过来，将Roberta替换为XLMRoberta，ROBERTA替换为XLM_ROBERTA
class XLMRobertaModel(XLMRobertaPreTrainedModel):
    """
    该模型可以作为编码器（仅自注意力）或解码器使用，后者在自注意力层之间添加了一个交叉注意力层，遵循Ashish Vaswani等人在《Attention is all you need》中描述的架构。

    若要作为解码器使用，需要使用配置中的`is_decoder`参数初始化为True。
    若要在Seq2Seq模型中使用，需要将`is_decoder`和`add_cross_attention`参数都设置为True；在前向传递中，需要将`encoder_hidden_states`作为输入。

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    # 从transformers.models.bert.modeling_bert.BertModel.__init__中复制过来，将Bert替换为XLMRoberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)

        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        对模型的注意力头进行修剪。heads_to_prune: 字典，格式为{层号: 要在该层中修剪的头列表}，参见基类PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从transformers.models.bert.modeling_bert.BertModel.forward中复制过来
    # 定义一个方法 forward，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，类型为可选的 PyTorch Tensor
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指示哪些元素是 padding，类型为可选的 PyTorch Tensor
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，如用于区分两个句子，类型为可选的 PyTorch Tensor
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs，标识输入 tokens 的位置信息，类型为可选的 PyTorch Tensor
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，用于指定每个注意力头是否执行，类型为可选的 PyTorch Tensor
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，代替 input_ids 的嵌入输入，类型为可选的 PyTorch Tensor
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，用于传递给注意力层，类型为可选的 PyTorch Tensor
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码，用于指示哪些编码器隐藏状态应该被忽略，类型为可选的 PyTorch Tensor
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对，用于在解码器自回归生成时保存历史状态，类型为可选的列表，元素为 PyTorch Tensor
        use_cache: Optional[bool] = None,  # 是否使用缓存加速解码器自回归生成，类型为可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出所有隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出，类型为可选的布尔值
@add_start_docstrings(
    "XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.",
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM 复制过来，并将 Roberta 改为 XLMRoberta，ROBERTA 改为 XLM_ROBERTA
class XLMRobertaForCausalLM(XLMRobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # 如果配置不是解码器，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `XLMRobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化 XLM-RoBERTa 模型和语言建模头部
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回语言建模头部的解码器权重
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置语言建模头部的解码器权重为新的嵌入
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 准备生成过程中的输入，接受输入的ID，过去的键值（用于存储中间状态），注意力掩码以及模型关键字参数
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入ID的形状信息
        input_shape = input_ids.shape
        # 如果未提供注意力掩码，则创建一个全为1的张量，形状与输入ID相同
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果提供了过去的键值（用于缓存中间状态）
        if past_key_values is not None:
            # 获取过去键值中每层的长度（通常对应解码器状态）
            past_length = past_key_values[0][0].shape[2]

            # 如果输入ID的长度大于过去的长度，说明需要裁剪输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length  # 裁剪长度设为过去长度
            else:
                # 否则，默认行为：仅保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 从输入ID中裁剪掉指定长度的前缀部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回一个包含输入ID、注意力掩码和过去键值的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存数据，根据给定的beam索引
    def _reorder_cache(self, past_key_values, beam_idx):
        # 创建一个空的元组用于存储重新排序后的过去键值
        reordered_past = ()
        # 遍历每层的过去键值
        for layer_past in past_key_values:
            # 对每个过去状态根据beam索引进行重新排序，并添加到重新排序后的过去状态元组中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去状态元组
        return reordered_past
# 为 XLM-RoBERTa 模型添加文档字符串，指明其是在语言建模任务上使用的模型
@add_start_docstrings(
    """XLM-RoBERTa Model with a `language modeling` head on top.""",
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_roberta.RobertaForMaskedLM 复制代码，并将 Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForMaskedLM(XLMRobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 初始化函数，接受一个配置对象 config
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中设置为 decoder，则发出警告信息，建议设置为 bi-directional self-attention
        if config.is_decoder:
            logger.warning(
                "If you want to use `XLMRobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 XLM-RoBERTa 模型，设置不添加 pooling 层
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        # 创建 XLM-RoBERTa 的语言建模头部
        self.lm_head = XLMRobertaLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入的函数
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入的函数
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数并返回输出结果
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
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
        # 输入参数文档字符串，描述了输入的形状
        # 示例代码文档字符串，展示了如何使用该函数的例子，包括模型的检查点、输出类型、配置类等信息
        # 这里没有函数体，因为该函数定义被截断，后续代码未展示
        ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 根据是否有 return_dict 参数决定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoBERTa 模型进行前向传播
        outputs = self.roberta(
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
        # 获取 RoBERTa 输出的序列输出
        sequence_output = outputs[0]
        # 使用语言模型头部对序列输出进行预测
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 将标签移到正确的设备以支持模型并行计算
            labels = labels.to(prediction_scores.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算掩码语言模型损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不返回字典格式的输出，则返回额外的输出参数
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 MaskedLMOutput 对象，包含损失、预测结果、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Copied from transformers.models.roberta.modeling_roberta.RobertaLMHead
class XLMRobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # Linear layer for transforming hidden states to vocab size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization for stabilizing learning
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Linear layer for decoding to vocabulary size
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # Bias parameter for the decoder
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # Project features to hidden size using dense layer
        x = self.dense(features)
        # Apply GELU activation function
        x = gelu(x)
        # Normalize using layer normalization
        x = self.layer_norm(x)

        # Project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # Tie weights to prevent disconnection (TPU or bias resizing)
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForSequenceClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Number of labels in the classification task
        self.num_labels = config.num_labels
        self.config = config

        # XLM-RoBERTa model without pooling layer
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        # Classification head for XLM-RoBERTa
        self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否使用返回字典，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoBERTa 模型进行前向传播
        outputs = self.roberta(
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
        # 获取模型输出的序列表示
        sequence_output = outputs[0]
        # 将序列表示传递给分类器以获取 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将标签移到正确的设备上，以便启用模型的并行计算
            labels = labels.to(logits.device)
            # 如果未指定问题类型，则根据标签类型和数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
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

        # 如果不需要返回字典，则返回分类器的输出
        if not return_dict:
            output = (logits,) + outputs[2:]  # 保留 logits 和额外的输出（如隐藏状态）
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
`
"""
XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.
"""

# Copied from transformers.models.roberta.modeling_roberta.RobertaForMultipleChoice with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForMultipleChoice(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize XLM-RoBERTa model based on provided configuration
        self.roberta = XLMRobertaModel(config)
        # Dropout layer with dropout probability from configuration
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Linear layer for classification with input size as hidden size from configuration, output size 1
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and perform final processing after model setup
        self.post_init()

    @add_start_docstrings_to_model_forward(
        XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # Forward method for XLM-RoBERTa multiple choice model
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 return_dict 参数决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算 num_choices，如果 input_ids 存在则取其第二维的大小，否则取 inputs_embeds 的第二维大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入数据展平，以便于传入模型
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将展平后的输入传入 RoBERTa 模型
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取池化后的输出
        pooled_output = outputs[1]

        # 应用 dropout 层
        pooled_output = self.dropout(pooled_output)
        # 通过分类器获取 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重塑为 (batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # 将 labels 移动到正确的设备上以支持模型并行处理
            labels = labels.to(reshaped_logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            # 如果不使用返回字典，则返回一个元组
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典，则返回 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.

This class inherits from XLMRobertaPreTrainedModel and adds a token classification layer on top of XLM-RoBERTa.
"""
@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# Copied from transformers.models.roberta.modeling_roberta.RobertaForTokenClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForTokenClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        """
        Initializes the XLMRobertaForTokenClassification model.

        Args:
            config (XLMRobertaConfig): Configuration class instance defining the model architecture and parameters.
        """
        super().__init__(config)
        # Number of output labels for token classification
        self.num_labels = config.num_labels

        # XLM-RoBERTa model without pooling layer
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        
        # Dropout layer with specified dropout rate
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Linear layer for token classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and perform final model setup
        self.post_init()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
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
        Performs forward pass of the XLMRobertaForTokenClassification model.

        Args:
            input_ids (torch.LongTensor, optional): Tokenized input IDs.
            attention_mask (torch.FloatTensor, optional): Attention mask to avoid performing attention on padding tokens.
            token_type_ids (torch.LongTensor, optional): Segment token indices to differentiate segments in sequence pairs.
            position_ids (torch.LongTensor, optional): Indices of positions of each input sequence tokens in the position embeddings.
            head_mask (torch.FloatTensor, optional): Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (torch.FloatTensor, optional): Optionally, the embeddings of the inputs instead of input_ids.
            labels (torch.LongTensor, optional): Labels for computing the token classification loss.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dict instead of a tuple of outputs.

        Returns:
            TokenClassifierOutput: Token classification output consisting of logits and optionally hidden states, attentions, and loss.
        """
        # Implementation details are copied from transformers.
        pass
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否返回字典类型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoBERTa 模型进行前向传播
        outputs = self.roberta(
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

        # 获取模型输出的序列特征向量
        sequence_output = outputs[0]

        # 对序列特征向量应用 dropout 操作
        sequence_output = self.dropout(sequence_output)
        
        # 将处理后的特征向量输入分类器，得到分类 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            # 将标签移到正确的设备以实现模型并行计算
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典类型的输出，则重新构造输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从 transformers.models.roberta.modeling_roberta.RobertaClassificationHead 复制而来，将 Roberta 替换为 XLMRoberta
class XLMRobertaClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 定义一个全连接层
        # 根据配置选择分类器的 dropout，如果未指定则使用隐藏层的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 定义一个 dropout 层
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)  # 输出层，映射到标签数量的维度

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # 取 <s> 标记（等同于 [CLS]）
        x = self.dropout(x)  # 应用 dropout
        x = self.dense(x)  # 应用全连接层
        x = torch.tanh(x)  # 应用 tanh 激活函数
        x = self.dropout(x)  # 再次应用 dropout
        x = self.out_proj(x)  # 应用输出投影层
        return x


@add_start_docstrings(
    """
    用于抽取式问答任务（如 SQuAD）的 XLM-RoBERTa 模型，顶部有一个跨度分类头部，线性层在隐藏状态输出的顶部，
    计算 'span start logits' 和 'span end logits'。
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering 复制而来，将 Roberta 替换为 XLMRoberta，ROBERTA 替换为 XLM_ROBERTA
class XLMRobertaForQuestionAnswering(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)  # XLM-RoBERTa 模型，不包含池化层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 线性层，映射到标签数量的维度

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="deepset/roberta-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
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
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
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
        # Determine whether to use the provided return_dict or default to the model's configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through the RoBERTa model and retrieve outputs
        outputs = self.roberta(
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

        # Extract the sequence output from RoBERTa's output
        sequence_output = outputs[0]

        # Compute logits for the Question Answering task
        logits = self.qa_outputs(sequence_output)
        
        # Split logits into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # Squeeze unnecessary dimensions from logits
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Ensure start_positions and end_positions are properly shaped
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # Define the ignored index and clamp positions within valid range
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Compute CrossEntropyLoss for start and end positions
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # If return_dict is False, return a tuple containing the loss and other outputs
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # If return_dict is True, return a QuestionAnsweringModelOutput object
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从输入的 input_ids 中创建位置标识符，替换非填充符号为它们的位置编号。位置编号从 padding_idx+1 开始，填充符号被忽略。
# 这是从 fairseq 的 `utils.make_positions` 修改而来。

# 定义一个函数，接受输入参数 input_ids（输入的索引序列）、padding_idx（填充符号的索引）、past_key_values_length（过去键值的长度，默认为0）
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 创建一个布尔掩码，标记哪些位置不是填充符号，将非填充位置标记为1，填充位置标记为0
    mask = input_ids.ne(padding_idx).int()
    # 计算累积的索引，用于替换非填充符号的位置。加上 past_key_values_length 是为了处理过去键值的长度。
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将累积的索引转换为长整型，并加上 padding_idx，得到最终的位置标识符
    return incremental_indices.long() + padding_idx
```