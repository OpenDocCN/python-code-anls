# `.\models\ernie\modeling_ernie.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""PyTorch ERNIE model."""


import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入来自HuggingFace库的模块和类
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
from .configuration_ernie import ErnieConfig

# 获取logger对象用于记录日志
logger = logging.get_logger(__name__)

# 以下两行定义了文档中用到的一些模型和配置信息
_CHECKPOINT_FOR_DOC = "nghuyong/ernie-1.0-base-zh"
_CONFIG_FOR_DOC = "ErnieConfig"

# 预训练模型的存档列表
ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nghuyong/ernie-1.0-base-zh",
    "nghuyong/ernie-2.0-base-en",
    "nghuyong/ernie-2.0-large-en",
    "nghuyong/ernie-3.0-base-zh",
    "nghuyong/ernie-3.0-medium-zh",
    "nghuyong/ernie-3.0-mini-zh",
    "nghuyong/ernie-3.0-micro-zh",
    "nghuyong/ernie-3.0-nano-zh",
    "nghuyong/ernie-gram-zh",
    "nghuyong/ernie-health-zh",
    # 查看所有 ERNIE 模型：https://huggingface.co/models?filter=ernie
]

# ErnieEmbeddings 类的定义，用于构建来自词嵌入、位置嵌入和标记类型嵌入的嵌入层
class ErnieEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 初始化函数，用于初始化模型参数和配置
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义词嵌入层，根据配置参数设置词表大小、隐藏层大小和填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 定义位置嵌入层，根据配置参数设置最大位置嵌入数和隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 定义token类型嵌入层，根据配置参数设置token类型词表大小和隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # 如果配置中使用任务ID，定义任务类型嵌入层，根据配置参数设置任务类型词表大小和隐藏层大小
        if config.use_task_id:
            self.task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)

        # self.LayerNorm 没有使用蛇形命名法以保持与 TensorFlow 模型变量名的一致性，并能够加载任何 TensorFlow 检查点文件
        # 定义Layer Normalization层，根据配置参数设置隐藏层大小和epsilon值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义Dropout层，根据配置参数设置隐藏层的dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义位置嵌入类型，根据配置参数获取绝对位置编码类型或其他类型
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置ID张量，用于序列化时持久化存储，长度为最大位置嵌入数
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册token类型ID张量，用于序列化时持久化存储，形状与位置ID相同，类型为长整型
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播函数，接受多个输入参数，返回模型的输出结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果给定了 input_ids，则获取其形状作为 input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，从 inputs_embeds 获取形状，排除最后一个维度（通常是 batch 维度）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则从预设的 position_ids 中切片出相应长度的部分
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置 token_type_ids 为注册的缓冲区，默认为全零，当其自动生成时有效，用于在模型追踪过程中解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为 None，则从 word_embeddings 中获取对应 input_ids 的嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 获取 token_type_ids 对应的 token type embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入的嵌入向量与 token type embeddings 相加得到最终的 embeddings
        embeddings = inputs_embeds + token_type_embeddings

        # 如果使用绝对位置编码，则将位置编码 embeddings 加到当前 embeddings 中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 如果模型使用 task_id，则将 task_type_ids 加入 embeddings
        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings += task_type_embeddings

        # 对 embeddings 进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回最终的 embeddings
        return embeddings
# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Ernie
class ErnieSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否能被注意力头数整除，否则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果使用相对位置编码，创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否作为解码器使用
        self.is_decoder = config.is_decoder

    # 调整形状以便进行注意力计算
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
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Ernie
class ErnieSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 层归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 密集层计算
        hidden_states = self.dense(hidden_states)
        # dropout 计算
        hidden_states = self.dropout(hidden_states)
        # 层归一化计算并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Ernie
# 定义一个名为 ErnieAttention 的自定义神经网络模块，继承自 nn.Module 类
class ErnieAttention(nn.Module):
    # 初始化函数，接受配置参数 config 和位置嵌入类型 position_embedding_type
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建 ErnieSelfAttention 层，并赋值给 self.self 属性
        self.self = ErnieSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 ErnieSelfOutput 层，并赋值给 self.output 属性
        self.output = ErnieSelfOutput(config)
        # 初始化一个空集合，用于存储被剪枝的注意力头信息
        self.pruned_heads = set()

    # 定义一个方法，用于剪枝注意力头
    def prune_heads(self, heads):
        # 如果 heads 列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数，找到可以剪枝的头部和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层中的权重
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头部信息
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数，接受多个输入张量并返回一个张量元组
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
        # 调用 self.self 的前向传播，获取自注意力输出
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力输出和输入 hidden_states 传入 self.output 层，获取注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 构建输出元组，包含注意力输出和可能的注意力权重
        outputs = (attention_output,) + self_outputs[1:]  # 如果有的话，加入注意力权重
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并改为 Ernie
# 定义一个名为 ErnieIntermediate 的神经网络模块，继承自 nn.Module 类
class ErnieIntermediate(nn.Module):
    # 初始化函数，接受配置参数 config
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征大小 config.hidden_size 映射到 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串类型，则使用 ACT2FN 字典映射的激活函数，否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接受输入张量 hidden_states 并返回张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量经过线性层 dense，得到中间状态 hidden_states
        hidden_states = self.dense(hidden_states)
        # 将中间状态 hidden_states 经过激活函数 intermediate_act_fn
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回激活后的中间状态 hidden_states
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并改为 Ernie
# 定义一个名为 ErnieOutput 的神经网络模块，继承自 nn.Module 类
class ErnieOutput(nn.Module):
    # 初始化方法，用于创建一个新的对象实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对输入进行归一化处理，设置epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于在训练过程中随机置零输入张量的部分元素，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，定义了如何从输入计算输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入全连接层，进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行 Dropout 操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 对加上输入张量的结果进行 LayerNorm 归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的张量作为输出
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertLayer 复制并修改为 ErnieLayer
class ErnieLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播的块大小（feed forward chunk size）
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设为1
        self.seq_len_dim = 1
        # 初始化 Ernie 注意力层
        self.attention = ErnieAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力（cross attention）
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨注意力，检查是否作为解码器使用，否则引发异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化 Ernie 跨注意力层，使用绝对位置嵌入
            self.crossattention = ErnieAttention(config, position_embedding_type="absolute")
        # 初始化 Ernie 中间层
        self.intermediate = ErnieIntermediate(config)
        # 初始化 Ernie 输出层
        self.output = ErnieOutput(config)

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
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用过去的键/值缓存（如果存在）的前两个位置来初始化自注意力机制的过去键/值
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 执行自注意力机制
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则添加自注意力
                                                  
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值元组位于过去键/值元组的第3、4个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行交叉注意力机制
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果需要输出注意力权重，则添加交叉注意力

            # 将交叉注意力的缓存添加到现在键/值元组的第3、4个位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用于前向传播的分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 将注意力输出应用于前向传播的分块处理
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制并修改为ErnieEncoder
class ErnieEncoder(nn.Module):
    # 初始化方法，接受一个config对象作为参数
    def __init__(self, config):
        super().__init__()
        # 将config对象保存到实例的self.config属性中
        self.config = config
        # 创建一个包含多个ErnieLayer对象的列表，列表长度为config.num_hidden_layers
        self.layer = nn.ModuleList([ErnieLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 前向传播方法，接受多个输入参数
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
        # 如果不需要输出隐藏状态，设置一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，设置一个空元组
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出交叉注意力权重或者配置不支持，设置一个空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点并且在训练阶段，检查是否与使用缓存参数冲突，如有冲突则警告并强制关闭使用缓存
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果不使用缓存，初始化一个空元组以保存下一个解码器缓存
        next_decoder_cache = () if use_cache else None
        # 遍历所有层次的解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，将当前隐藏状态加入到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果没有指定头部掩码，则设为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对，如果没有指定过去键值对，则设为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并且在训练阶段，使用梯度检查点函数来计算当前层的输出
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
                # 否则，直接调用当前层模块来计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，将当前层的缓存信息加入到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，将当前层的自注意力权重加入到所有自注意力权重元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置支持添加交叉注意力，将当前层的交叉注意力权重加入到所有交叉注意力权重元组中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，将最终的隐藏状态加入到所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，将需要返回的各项整合成一个元组并返回
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
        # 否则，返回一个带有过去键值对和交叉注意力的基础模型输出对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Ernie
class ErniePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取出隐藏状态中的第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态输入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数处理全连接层的输出
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Ernie
class ErniePredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # LayerNorm 层，输入维度为 config.hidden_size，epsilon 为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入隐藏状态经过全连接层
        hidden_states = self.dense(hidden_states)
        # 使用指定的激活函数处理全连接层的输出
        hidden_states = self.transform_act_fn(hidden_states)
        # 输入 LayerNorm 层处理后返回
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Ernie
class ErnieLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 声明一个 ErniePredictionHeadTransform 对象，用于转换隐藏状态
        self.transform = ErniePredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个 token 都有一个独立的输出偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 声明一个 bias 参数，用于输出层每个 token 的偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要建立 decoder.bias 与 self.bias 之间的关联，以便在调整 token embeddings 时正确调整偏置
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 隐藏状态经过 transform 转换
        hidden_states = self.transform(hidden_states)
        # 转换后的隐藏状态经过线性层，输出预测分数
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Ernie
class ErnieOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 声明一个 ErnieLMPredictionHead 对象，用于预测 MLM 的结果
        self.predictions = ErnieLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 序列的输出经过预测层，得到预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->Ernie
class ErnieOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 一个线性层，用于预测 NSP（Next Sentence Prediction）
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    # 定义一个方法 `forward`，用于执行前向传播
    def forward(self, pooled_output):
        # 调用 `seq_relationship` 方法计算序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回计算得到的序列关系分数
        return seq_relationship_score
# 从 transformers.models.bert.modeling_bert.BertPreTrainingHeads 复制的代码，将 Bert 替换为 Ernie
class ErniePreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 ErnieLMPredictionHead 对象，用于预测下一个词的概率分布
        self.predictions = ErnieLMPredictionHead(config)
        # 创建线性层，用于预测两个句子之间的关系
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 调用 predictions 对象进行预测，生成预测分数
        prediction_scores = self.predictions(sequence_output)
        # 使用 seq_relationship 层预测句子之间的关系得分
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回预测的语言模型分数和句子关系分数
        return prediction_scores, seq_relationship_score


class ErniePreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和预训练模型下载加载的抽象类。
    """

    config_class = ErnieConfig
    base_model_prefix = "ernie"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层的权重，标准差为 config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，标准差为 config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有 padding_idx，则将对应的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 层的偏置初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
# 从 transformers.models.bert.modeling_bert.BertForPreTrainingOutput 复制的代码，将 Bert 替换为 Ernie
class ErnieForPreTrainingOutput(ModelOutput):
    """
    [`ErnieForPreTraining`] 的输出类型。
    """
    # 定义函数的参数说明和类型注解
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            # 如果提供了 `labels` 参数，则返回的可选参数，表示总损失，包括掩码语言建模损失和下一个序列预测（分类）损失。
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            # 语言建模头部的预测分数（softmax之前的每个词汇标记的分数）。
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            # 下一个序列预测（分类）头部的预测分数（softmax之前的True/False连续性得分）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 模型在每层输出后的隐藏状态，以及初始嵌入输出的元组。
            # 如果传递了 `output_hidden_states=True` 或者 `config.output_hidden_states=True`，则返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            # 自注意力头部中的注意力权重，经过注意力softmax后的权重，用于计算加权平均值。
            # 如果传递了 `output_attentions=True` 或者 `config.output_attentions=True`，则返回。
    """

    # 损失值，类型为可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 语言建模头部的预测分数，张量形状为 `(batch_size, sequence_length, config.vocab_size)`
    prediction_logits: torch.FloatTensor = None
    # 下一个序列预测头部的预测分数，张量形状为 `(batch_size, 2)`
    seq_relationship_logits: torch.FloatTensor = None
    # 隐藏状态，类型为可选的浮点张量元组，形状为 `(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，类型为可选的浮点张量元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[torch.FloatTensor]] = None
"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ErnieConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""



"""
"""
"""

@add_start_docstrings(
    "The bare Ernie Model transformer outputting raw hidden-states without any specific head on top.",
    ERNIE_START_DOCSTRING,
)
"""
# 定义 ErnieModel 类，继承自 ErniePreTrainedModel
class ErnieModel(ErniePreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """



    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Ernie
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = ErnieEmbeddings(config)  # 初始化 ErnieEmbeddings，用于处理输入的词嵌入
        self.encoder = ErnieEncoder(config)  # 初始化 ErnieEncoder，用于进行编码器的编码

        self.pooler = ErniePooler(config) if add_pooling_layer else None  # 如果 add_pooling_layer 为真，初始化 ErniePooler，用于池化层处理

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和最终处理



    # Copied from transformers.models.bert.modeling_bert.BertModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings  # 返回输入嵌入的词嵌入



    # Copied from transformers.models.bert.modeling_bert.BertModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value  # 设置输入嵌入的词嵌入为给定的值



    # Copied from transformers.models.bert.modeling_bert.BertModel._prune_heads
    # 定义一个方法 `_prune_heads`，用于修剪模型中的注意力头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 字典中的每个层及对应要修剪的注意力头部列表
        for layer, heads in heads_to_prune.items():
            # 在模型的编码器（encoder）中定位到指定层的注意力（attention）对象，并执行修剪操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 声明一个前向传播方法 `forward`，并应用装饰器添加文档字符串和代码示例文档字符串
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
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
@add_start_docstrings(
    """
    Ernie Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    ERNIE_START_DOCSTRING,
)
class ErnieForPreTraining(ErniePreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 从 transformers.models.bert.modeling_bert.BertForPreTraining.__init__ 复制而来，将 Bert 替换为 Ernie，bert 替换为 ernie
    def __init__(self, config):
        super().__init__(config)

        self.ernie = ErnieModel(config)
        self.cls = ErniePreTrainingHeads(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 从 transformers.models.bert.modeling_bert.BertForPreTraining.get_output_embeddings 复制而来
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 从 transformers.models.bert.modeling_bert.BertForPreTraining.set_output_embeddings 复制而来
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=ErnieForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，接受多个输入参数，执行 Ernie 模型的预测任务。

        Args:
            input_ids (Optional[torch.Tensor], optional): 输入 token IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码，指示哪些元素是填充项. Defaults to None.
            token_type_ids (Optional[torch.Tensor], optional): token 类型 IDs，用于区分句子 A 和句子 B. Defaults to None.
            task_type_ids (Optional[torch.Tensor], optional): 任务类型 IDs，用于特定任务的区分. Defaults to None.
            position_ids (Optional[torch.Tensor], optional): 位置 IDs，指示每个 token 的位置. Defaults to None.
            head_mask (Optional[torch.Tensor], optional): 头部掩码，用于指定哪些注意力头应该被屏蔽. Defaults to None.
            inputs_embeds (Optional[torch.Tensor], optional): 直接输入的嵌入表示. Defaults to None.
            labels (Optional[torch.Tensor], optional): 模型的标签，用于 MLM 损失计算. Defaults to None.
            next_sentence_label (Optional[torch.Tensor], optional): 下一个句子预测的标签. Defaults to None.
            output_attentions (Optional[bool], optional): 是否输出注意力权重. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否返回字典格式的输出. Defaults to None.

        Returns:
            ErnieForPreTrainingOutput or torch.Tensor: 根据 return_dict 决定返回 ErnieForPreTrainingOutput 对象或直接的张量输出.
        """
        # 实现具体的前向传播逻辑，包括输入处理、模型计算和输出处理
        pass


@add_start_docstrings(
    """Ernie Model with a `language modeling` head on top for CLM fine-tuning.""", ERNIE_START_DOCSTRING
)
class ErnieForCausalLM(ErniePreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 从 transformers.models.bert.modeling_bert.BertLMHeadModel.__init__ 复制而来，将 BertLMHeadModel->ErnieForCausalLM, Bert->Ernie, bert->ernie
    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `ErnieForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 Ernie 模型和仅含 MLM 头部的头部
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        self.cls = ErnieOnlyMLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 从 transformers.models.bert.modeling_bert.BertLMHeadModel.get_output_embeddings 复制而来
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        # 将预测层的解码器替换为新的嵌入层
        self.cls.predictions.decoder = new_embeddings
    
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建全为1的掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
    
        # 如果使用了过去的键值对，根据需要截取输入的decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
    
            # 一些生成方法已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1
    
            input_ids = input_ids[:, remove_prefix_length:]
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
    
    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel._reorder_cache
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        # 重新排序过去的键值对，以匹配新的beam索引
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
@add_start_docstrings("""Ernie Model with a `language modeling` head on top.""", ERNIE_START_DOCSTRING)
class ErnieForMaskedLM(ErniePreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 从transformers.models.bert.modeling_bert.BertForMaskedLM.__init__复制而来，将Bert->Ernie，bert->ernie
    def __init__(self, config):
        super().__init__(config)

        # 如果配置为decoder，发出警告，因为ErnieForMaskedLM需要使用双向self-attention，所以要求config.is_decoder=False
        if config.is_decoder:
            logger.warning(
                "If you want to use `ErnieForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化Ernie模型，不添加池化层
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        # 初始化仅包含MLM头部的ErnieOnlyMLMHead
        self.cls = ErnieOnlyMLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 从transformers.models.bert.modeling_bert.BertForMaskedLM.get_output_embeddings复制而来
    def get_output_embeddings(self):
        # 返回MLM头部的预测解码器
        return self.cls.predictions.decoder

    # 从transformers.models.bert.modeling_bert.BertForMaskedLM.set_output_embeddings复制而来
    def set_output_embeddings(self, new_embeddings):
        # 设置MLM头部的预测解码器为新的嵌入层
        self.cls.predictions.decoder = new_embeddings

    # 使用add_start_docstrings_to_model_forward装饰器添加文档字符串到forward方法
    # 使用add_code_sample_docstrings添加代码示例和期望输出的文档字符串
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        # 如果 return_dict 不为 None，则使用给定的值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ERNIE 模型进行前向传播
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 通过分类器获取预测得分
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果提供了标签，则计算 masked language modeling 的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数，-100 代表填充标记
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 从 transformers.models.bert.modeling_bert.BertForMaskedLM.prepare_inputs_for_generation 复制而来
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # 添加一个虚拟的 token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        # 扩展 attention_mask，在最后添加一个全零的列
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建一个全是 PAD token 的虚拟 token
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        # 在 input_ids 后面添加虚拟 token
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回输入字典，包括修改后的 input_ids 和 attention_mask
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 定义 ErnieForNextSentencePrediction 类，它在 ERNIE 模型的基础上添加了一个“下一个句子预测（分类）”的头部。
@add_start_docstrings(
    """Ernie Model with a `next sentence prediction (classification)` head on top.""",
    ERNIE_START_DOCSTRING,
)
class ErnieForNextSentencePrediction(ErniePreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForNextSentencePrediction.__init__ 复制而来，将其中的 Bert 改为 Ernie，bert 改为 ernie
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Ernie 模型
        self.ernie = ErnieModel(config)
        # 初始化仅包含 NSP 头部的 ErnieOnlyNSPHead
        self.cls = ErnieOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 向前传播函数，接受多个输入参数并返回一个输出结果，使用了 add_start_docstrings_to_model_forward 和 replace_return_docstrings 进行文档字符串的注释和替换
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
):
        ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:
            Tuple containing either logits or a full NextSentencePredictorOutput if configured.

        Example:

        ```
        >>> from transformers import AutoTokenizer, ErnieForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
        >>> model = ErnieForNextSentencePrediction.from_pretrained("nghuyong/ernie-1.0-base-zh")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            # 如果传入了过时的参数 `next_sentence_label`，发出警告并使用 `labels` 替代
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        # 确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ERNIE 模型进行前向传播
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 ERNIE 模型输出中提取池化后的输出
        pooled_output = outputs[1]

        # 使用分类器对池化输出进行预测下一个句子关系的分数
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            # 如果提供了标签，计算下一个句子预测的损失
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            # 如果不返回字典，则按照旧版格式构造输出
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 返回包含损失、分数、隐藏状态和注意力权重的 NextSentencePredictorOutput 对象
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Ernie Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ERNIE_START_DOCSTRING,
)
class ErnieForSequenceClassification(ErniePreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ 复制并修改为 Ernie 模型的序列分类/回归头部
    def __init__(self, config):
        super().__init__(config)
        # 初始化时设置标签数量和配置
        self.num_labels = config.num_labels
        self.config = config

        # 使用 ErnieModel 初始化 Ernie 模型
        self.ernie = ErnieModel(config)
        # 根据配置设置分类器的丢弃率，如果未指定，则使用隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用丢弃率初始化 Dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 设置线性分类器层，输入大小为隐藏层大小，输出大小为标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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
        # 初始化返回字典，根据是否已定义确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用ERNIE模型进行前向传播，获取输出
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从ERNIE模型的输出中获取池化后的表示
        pooled_output = outputs[1]

        # 应用Dropout层到池化后的表示
        pooled_output = self.dropout(pooled_output)
        
        # 通过分类器获取预测的逻辑回归
        logits = self.classifier(pooled_output)

        # 初始化损失为None
        loss = None

        # 如果提供了标签，则计算损失
        if labels is not None:
            # 如果问题类型未定义，则根据标签的数据类型和类别数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择损失函数并计算损失
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

        # 如果不使用返回字典，则返回输出和损失
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用返回字典对象封装损失和模型输出
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
Ernie Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.
"""

# 继承自 ErniePreTrainedModel 的 ErnieForMultipleChoice 类，用于多项选择任务的 Ernie 模型
class ErnieForMultipleChoice(ErniePreTrainedModel):
    
    # 从 transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__ 复制而来，将其中的 Bert 替换为 Ernie
    def __init__(self, config):
        super().__init__(config)
        
        # 初始化 ErnieModel
        self.ernie = ErnieModel(config)
        
        # 分类器的 dropout 率，默认使用 config 中的 hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 线性分类器，将隐藏状态大小（hidden_size）映射到1，用于多项选择任务
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加输入文档字符串和示例代码文档字符串到模型前向传播方法
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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
        # 根据函数声明，接受输入并返回包含损失或输出的元组或多选模型输出对象
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定 num_choices，如果没有提供 input_ids，则从 inputs_embeds 计算
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 根据是否为 None，重新形状化输入张量
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 ERNIE 模型，获取输出
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 ERNIE 输出中获取汇聚后的输出
        pooled_output = outputs[1]

        # 对汇聚后的输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器计算 logits
        logits = self.classifier(pooled_output)
        # 重新形状化 logits 以匹配 num_choices
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不要求返回字典形式的输出，构建输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选模型输出对象，包括损失、logits、隐藏状态和注意力权重
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
Ernie Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
# 导入所需的库
@add_start_docstrings(
    """
    添加一个头部的令牌分类器（在隐藏状态输出的顶部添加一个线性层），例如用于命名实体识别（NER）任务。
    """,
    ERNIE_START_DOCSTRING,
)
# 定义 ErnieForTokenClassification 类，继承自 ErniePreTrainedModel
class ErnieForTokenClassification(ErniePreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ 复制而来，将 Bert 替换为 Ernie
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)
        # 设置标签数目
        self.num_labels = config.num_labels

        # 创建 Ernie 模型，不添加池化层
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        
        # 根据配置设置分类器的 dropout，如果未指定则使用隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建 Dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性层作为分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行后续处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数说明文档字符串

        batch_size, sequence_length
        """
        # 确保返回的字典选项
        if return_dict is None:
            return_dict = self.config.use_return_dict

        # 执行 Ernie 模型的前向传播，获取输出
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 若标签存在，将输出传递给分类器进行分类
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # 若返回字典，将 logits 加入到输出中
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((logits,) + outputs[2:]) if return_dict else output

        # 创建命名元组并返回
        return TokenClassifierOutput(
            loss=None if labels is None else self.compute_loss(logits, labels),
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ERNIE 模型进行前向传播
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 使用分类器对处理后的序列输出进行分类预测
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果存在标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用返回字典，则构造输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典，则构造 TokenClassifierOutput 对象返回
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Ernie Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ERNIE_START_DOCSTRING,
)
class ErnieForQuestionAnswering(ErniePreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ 复制过来，将其中的 Bert 修改为 Ernie
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 设置分类标签的数量
        self.num_labels = config.num_labels

        # 创建 Ernie 模型，不添加池化层
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        # 创建一个线性层，用于输出答案起始位置和结束位置的 logit
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
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
        # Determine if return_dict should be set to self.config.use_return_dict if not provided
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform forward pass through the ERNIE model
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the sequence output from the model outputs
        sequence_output = outputs[0]

        # Compute logits for the question answering task
        logits = self.qa_outputs(sequence_output)

        # Split logits into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze unnecessary dimensions and ensure contiguous memory layout
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If start_positions or end_positions have extra dimensions, squeeze them
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # Clamp positions to avoid out-of-bound errors
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Define CrossEntropyLoss with ignored index
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            # Compute start and end loss
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # Calculate total loss as the average of start and end loss
            total_loss = (start_loss + end_loss) / 2

        # If return_dict is False, return outputs in a tuple
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # If return_dict is True, return structured QuestionAnsweringModelOutput
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```