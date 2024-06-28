# `.\models\bert_generation\modeling_bert_generation.py`

```
# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model specific for generation."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_bert_generation import BertGenerationConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 模型文档中的预定义变量
_CHECKPOINT_FOR_DOC = "google/bert_for_seq_generation_L-24_bbc_encoder"
_CONFIG_FOR_DOC = "BertGenerationConfig"


# 从transformers.models.bert.modeling_bert.BertSelfOutput复制并更改为BertGeneration
class BertGenerationSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接变换
        hidden_states = self.dense(hidden_states)
        # 随机失活
        hidden_states = self.dropout(hidden_states)
        # 残差连接和Layer normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertSelfAttention复制并更改为BertGeneration
class BertGenerationSelfAttention(nn.Module):
    # 省略了构造函数
    pass
    # 初始化函数，接受配置参数和可能的位置嵌入类型
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 检查隐藏层大小是否可以被注意力头数整除，同时检查是否存在嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不满足条件，抛出数值错误异常
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层，用于注意力概率的 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对位置嵌入，则初始化距离嵌入的 Embedding 层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 标记是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量重塑为注意力分数的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 计算新的张量形状，将注意力头维度放到第二维，头大小维度放到最后一维
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重塑张量形状
        x = x.view(new_x_shape)
        # 调换维度，以便计算注意力分数
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接受隐藏状态和各种可选的参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->BertGeneration
class BertGenerationAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层，使用BertGenerationSelfAttention类
        self.self = BertGenerationSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化输出层，使用BertGenerationSelfOutput类
        self.output = BertGenerationSelfOutput(config)
        # 存储需要被剪枝的注意力头索引的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用帮助函数找到可剪枝的注意力头和其对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的注意力头
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
        # 前向传播函数，调用自注意力层和输出层
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力层的输出作为参数传给输出层
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则添加到outputs中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到outputs中
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->BertGeneration
class BertGenerationIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义线性层，将隐藏状态映射到中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层进行映射
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->BertGeneration
class BertGenerationOutput(nn.Module):
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，将输入特征大小设为 config.intermediate_size，输出特征大小设为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 Layer Normalization 层，对输入的隐藏状态进行归一化处理，归一化的维度为 config.hidden_size，设置 epsilon 为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于随机将输入张量中的元素设置为零，以防止过拟合，丢弃率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了数据从输入到输出的流程
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行 Dropout 处理，以减少过拟合风险
        hidden_states = self.dropout(hidden_states)
        # 将 Dropout 后的隐藏状态与输入张量进行加法操作，并对结果进行 Layer Normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制而来，将Bert改为BertGeneration
class BertGenerationLayer(nn.Module):
    # 初始化方法，接收一个config对象作为参数
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设为1
        self.seq_len_dim = 1
        # 创建BertGenerationAttention对象
        self.attention = BertGenerationAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力，确保作为解码器模型使用
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建具有绝对位置嵌入类型的BertGenerationAttention对象
            self.crossattention = BertGenerationAttention(config, position_embedding_type="absolute")
        # 创建BertGenerationIntermediate对象
        self.intermediate = BertGenerationIntermediate(config)
        # 创建BertGenerationOutput对象
        self.output = BertGenerationOutput(config)

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
        # 定义方法 feed_forward_chunk，它接收 attention_output 作为输入并返回处理后的层输出
        def feed_forward_chunk(self, attention_output):
            # 使用 self.intermediate 对 attention_output 进行中间层处理
            intermediate_output = self.intermediate(attention_output)
            # 使用 self.output 对中间层输出和 attention_output 进行最终层处理，得到最终层输出
            layer_output = self.output(intermediate_output, attention_output)
            # 返回最终层输出作为 feed_forward_chunk 方法的输出
            return layer_output

        # 如果 past_key_value 不为 None，则将其前两个元素作为 self_attn_past_key_value，否则设为 None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        # 使用 self.attention 方法处理 hidden_states，根据给定的参数生成 self_attention_outputs
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        
        # 从 self_attention_outputs 中获取 self-attention 输出
        attention_output = self_attention_outputs[0]

        # 如果当前对象是解码器（decoder），则 self_attention_outputs 的最后一个元素为 self-attn cache 元组
        if self.is_decoder:
            # 从 self_attention_outputs 中排除最后一个元素，其余元素存入 outputs
            outputs = self_attention_outputs[1:-1]
            # 将 self_attention_outputs 的最后一个元素作为 present_key_value
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，则 outputs 包括除了第一个元素之外的所有 self_attention_outputs
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加 self attentions
       
        # 初始化交叉注意力的 present_key_value 为 None
        cross_attn_present_key_value = None
        
        # 如果是解码器且 encoder_hidden_states 不为 None
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果当前对象没有 crossattention 属性，则抛出异常
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果 past_key_value 不为 None，则取其倒数两个元素作为 cross_attn_past_key_value，否则设为 None
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            
            # 使用 self.crossattention 方法处理 attention_output，生成 cross_attention_outputs
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            
            # 从 cross_attention_outputs 中获取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            
            # 将 cross_attention_outputs 的除了第一个和最后一个元素之外的所有元素添加到 outputs 中
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加 cross attentions
            
            # 将 cross_attention_outputs 的最后一个元素作为 cross_attn_present_key_value
            cross_attn_present_key_value = cross_attention_outputs[-1]
            
            # 将 present_key_value 和 cross_attn_present_key_value 相加，更新 present_key_value
            present_key_value = present_key_value + cross_attn_present_key_value
        
        # 对 attention_output 应用分块处理策略，得到层输出 layer_output
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        
        # 将 layer_output 添加到 outputs 的开头
        outputs = (layer_output,) + outputs
        
        # 如果是解码器，则将 attn key/values 作为输出的最后一个元素添加到 outputs 中
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        
        # 返回最终的 outputs
        return outputs
# 从transformers.models.bert.modeling_bert.BertEncoder复制代码，并将Bert->BertGeneration
class BertEncoder(nn.Module):
    # 初始化方法，接受一个config对象作为参数
    def __init__(self, config):
        super().__init__()
        # 将传入的config对象保存到实例变量中
        self.config = config
        # 创建一个由多个BertGenerationLayer组成的层列表，列表长度为config.num_hidden_layers
        self.layer = nn.ModuleList([BertGenerationLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为False
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
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不需要输出隐藏状态，则初始化为空元组；否则设为 None，准备存储每层的隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化为空元组；否则设为 None，准备存储每层的自注意力权重
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出交叉注意力权重或没有配置交叉注意力，则初始化为空元组；否则设为 None，准备存储每层的交叉注意力权重
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点且在训练模式下
        if self.gradient_checkpointing and self.training:
            # 如果同时设置了使用缓存，则给出警告并强制设置 `use_cache=False`
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果需要使用缓存，则初始化为空元组；否则设为 None，准备存储下一个解码器缓存
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果提供了头部掩码，则获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果提供了过去的键值对，则获取当前层的过去键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且在训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数对当前层进行调用，计算当前层的输出
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
                # 否则直接调用当前层，计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的缓存添加到下一个解码器缓存的元组中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到所有自注意力权重的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置了添加交叉注意力，则将当前层的交叉注意力权重添加到所有交叉注意力权重的元组中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典格式的结果，则将各项结果组成元组返回
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
        # 否则，将结果封装成 BaseModelOutputWithPastAndCrossAttentions 对象返回
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
def load_tf_weights_in_bert_generation(
    model, tf_hub_path, model_class, is_encoder_named_decoder=False, is_encoder=False
):
    try:
        # 尝试导入必要的库
        import numpy as np
        import tensorflow.compat.v1 as tf
        import tensorflow_hub as hub
        import tensorflow_text  # noqa: F401

        # 禁用 TensorFlow 的即时执行模式
        tf.disable_eager_execution()
    except ImportError:
        # 如果导入失败，记录错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 使用 TensorFlow Hub 加载模型
    tf_model = hub.Module(tf_hub_path)
    # 初始化 TensorFlow 的全局变量
    init = tf.global_variables_initializer()

class BertGenerationEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        # 初始化词嵌入层和位置嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # LayerNorm 保持与 TensorFlow 模型变量名一致，以便加载任何 TensorFlow 的检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册位置编码张量，用于处理序列位置信息
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 如果位置编码未提供，则使用预定义的位置编码张量
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未提供输入嵌入，则使用输入的词嵌入进行计算
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 计算位置嵌入
        position_embeddings = self.position_embeddings(position_ids)

        # 将词嵌入和位置嵌入相加
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertGenerationPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # BertGenerationPreTrainedModel 的配置类
    config_class = BertGenerationConfig
    # 基础模型前缀
    base_model_prefix = "bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层（全连接层）
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为模型配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为模型配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了填充索引，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
BERT_GENERATION_START_DOCSTRING = r"""
    
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertGenerationConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_GENERATION_INPUTS_DOCSTRING = r"""
    
    Inputs:
        input_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.__call__` and :meth:`transformers.PreTrainedTokenizer.encode` for
            details.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            Defaults to ``None``.
        decoder_input_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, optional):
            Provide for generation tasks to guide decoding. Indices can be obtained using
            :class:`~transformers.BertTokenizer`. See :meth:`transformers.PreTrainedTokenizer.__call__` and
            :meth:`transformers.PreTrainedTokenizer.encode` for details.
        decoder_attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, optional):
            Mask to avoid performing attention on padding token indices for the decoder input. Mask values selected in
            ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            Defaults to ``None``.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, optional):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            Defaults to ``None``.
        inputs_embeds (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, optional):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert tokens to embeddings before feeding them into
            the model.
        decoder_inputs_embeds (:obj:`torch.Tensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`,
            optional):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert tokens to embeddings before
            feeding them into the model.
        labels (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (padding value).
            Tokens with labels set to -100 are ignored (masked), otherwise compute loss.

    Returns:
        :obj:`torch.Tensor`: Returns a tuple comprising various elements depending on the configuration. The first
        element is the final hidden states from the model, which can be used for further downstream tasks such as
        classification, regression, or sequence generation.

    Example::

        from transformers import BertTokenizer, BertGenerationModel

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertGenerationModel.from_pretrained('bert-base-uncased')

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        outputs = model(**inputs)

        logits = outputs.logits
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列的token索引在词汇表中的位置。
            # 可以使用[`AutoTokenizer`]获取这些索引。参见[`PreTrainedTokenizer.__call__`]和[`PreTrainedTokenizer.encode`]获取更多细节。
            # [什么是输入ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于避免在填充token索引上执行注意力计算。
            # 遮罩值在 `[0, 1]` 之间：
            # - 1 表示token没有被遮罩，
            # - 0 表示token被遮罩。
            # [什么是注意力遮罩？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列token在位置嵌入中的位置索引。
            # 索引范围在 `[0, config.max_position_embeddings - 1]` 之间。
            # [什么是位置ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中选择的注意力头的遮罩。
            # 遮罩值在 `[0, 1]` 之间：
            # - 1 表示注意力头没有被遮罩，
            # - 0 表示注意力头被遮罩。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选参数，可以直接传入嵌入表示，而不是传递`input_ids`。
            # 如果您希望对如何将`input_ids`索引转换为相关向量有更多控制权，这将非常有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。更多细节请参见返回的张量中的`attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。更多细节请参见返回的张量中的`hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通的元组。
    """
    # 给BertGenerationEncoder类添加文档字符串，描述其作为无特定头部的原始隐藏状态输出的BertGeneration模型转换器
    @add_start_docstrings(
        "The bare BertGeneration model transformer outputting raw hidden-states without any specific head on top.",
        BERT_GENERATION_START_DOCSTRING,
    )
    """
    """
    # BertGenerationEncoder类，用于BertGenerationPreTrainedModel的扩展
    class BertGenerationEncoder(BertGenerationPreTrainedModel):
    
        """
        # BertGenerationEncoder类的初始化函数，初始化模型配置
        def __init__(self, config):
            # 调用父类的初始化函数
            super().__init__(config)
            # 将配置保存到对象中
            self.config = config
    
            # 初始化嵌入层
            self.embeddings = BertGenerationEmbeddings(config)
            # 初始化编码器层
            self.encoder = BertEncoder(config)
    
            # 调用后处理函数，初始化权重并进行最终处理
            self.post_init()
    
        # 获取输入嵌入层的函数
        def get_input_embeddings(self):
            return self.embeddings.word_embeddings
    
        # 设置输入嵌入层的函数
        def set_input_embeddings(self, value):
            self.embeddings.word_embeddings = value
    
        # 剪枝模型中的注意力头部的函数
        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)
    
    """
    """
        # 为模型正向传播函数添加文档字符串
        @add_start_docstrings_to_model_forward(BERT_GENERATION_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=BaseModelOutputWithPastAndCrossAttentions,
            config_class=_CONFIG_FOR_DOC,
        )
    ```
    # 前向传播函数，用于模型的前向推理过程
    def forward(
        self,
        # 输入的 token IDs 张量，可以是可选的
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码张量，用于指示哪些 token 是有效的，可以是可选的
        attention_mask: Optional[torch.Tensor] = None,
        # 位置 IDs 张量，用于指定每个 token 的位置信息，可以是可选的
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码张量，用于屏蔽某些注意力头部的输出，可以是可选的
        head_mask: Optional[torch.Tensor] = None,
        # 输入嵌入张量，代替输入 token IDs 进行输入，可以是可选的
        inputs_embeds: Optional[torch.Tensor] = None,
        # 编码器隐藏状态张量，可以是可选的
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器注意力掩码张量，用于指示编码器哪些 token 是有效的，可以是可选的
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 过去的键-值元组，用于存储前一个时间步的键-值，可以是可选的
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 是否使用缓存，控制是否使用过去的键-值缓存结果，可以是可选的
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重张量，可以是可选的
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态张量，可以是可选的
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典对象作为前向传播的输出，可以是可选的
        return_dict: Optional[bool] = None,
class BertGenerationOnlyLMHead(nn.Module):
    # 定义一个类，用于BERT生成模型的语言建模头部
    def __init__(self, config):
        super().__init__()
        # 初始化方法，继承父类构造函数
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 创建一个线性层，用于生成输出词汇的logits
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 创建一个可学习的偏置参数
        self.decoder.bias = self.bias
        # 将偏置参数赋给线性层的偏置

    def forward(self, hidden_states):
        # 前向传播方法
        logits = self.decoder(hidden_states)
        # 计算输出的logits
        return logits

    def _tie_weights(self):
        # 方法用于绑定两个权重（如果它们在TPU上断开连接或者偏置被调整大小时）
        self.bias = self.decoder.bias


@add_start_docstrings(
    """BertGeneration Model with a `language modeling` head on top for CLM fine-tuning.""",
    BERT_GENERATION_START_DOCSTRING,
)
class BertGenerationDecoder(BertGenerationPreTrainedModel):
    # BertGeneration解码器类，继承自BertGeneration预训练模型

    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertGenerationDecoder` as a standalone, add `is_decoder=True.`")
        # 如果不是解码器模式，发出警告信息

        self.bert = BertGenerationEncoder(config)
        # 创建一个BertGeneration编码器对象
        self.lm_head = BertGenerationOnlyLMHead(config)
        # 创建一个BERT生成模型的语言建模头部对象

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 获取输出词嵌入
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置输出词嵌入
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_GENERATION_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 前向传播方法，支持多种参数输入
    # 准备生成过程中的输入数据
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入张量的形状信息
        input_shape = input_ids.shape
        
        # 如果未提供注意力掩码，则创建一个全1的张量作为注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用过去的键值对，则根据情况裁剪输入的输入ID
        if past_key_values is not None:
            # 获取过去键值对中第一个元素的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入ID的长度大于过去的长度，则移除前缀部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个输入ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回一个包含准备好的输入数据的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存中的过去键值对，以适应束搜索的索引
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 将每个过去状态按照给定的束搜索索引重新排序，并组成元组
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对元组
        return reordered_past
```