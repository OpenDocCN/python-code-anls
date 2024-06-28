# `.\models\visual_bert\modeling_visual_bert.py`

```
# coding=utf-8
# Copyright 2021 The UCLA NLP Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch VisualBERT model."""

# 导入需要的库和模块
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax

# 导入自定义模块和函数
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
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
from .configuration_visual_bert import VisualBertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档字符串中使用的配置和检查点名称
_CONFIG_FOR_DOC = "VisualBertConfig"
_CHECKPOINT_FOR_DOC = "uclanlp/visualbert-vqa-coco-pre"

# 预训练模型存档列表
VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uclanlp/visualbert-vqa",
    "uclanlp/visualbert-vqa-pre",
    "uclanlp/visualbert-vqa-coco-pre",
    "uclanlp/visualbert-vcr",
    "uclanlp/visualbert-vcr-pre",
    "uclanlp/visualbert-vcr-coco-pre",
    "uclanlp/visualbert-nlvr2",
    "uclanlp/visualbert-nlvr2-pre",
    "uclanlp/visualbert-nlvr2-coco-pre",
    # See all VisualBERT models at https://huggingface.co/models?filter=visual_bert
]

# VisualBertEmbeddings 类定义，构建来自单词、位置、标记类型嵌入和视觉嵌入的嵌入
class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()

        # 创建词嵌入层，用于将词索引映射到隐藏表示向量，参数包括词汇表大小、隐藏大小，以及填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # 创建位置嵌入层，用于将位置索引映射到隐藏表示向量，参数包括最大位置数和隐藏大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 创建类型嵌入层，用于将类型索引（如token类型）映射到隐藏表示向量，参数包括类型词汇表大小和隐藏大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建 LayerNorm 层，用于标准化隐藏表示向量，参数包括隐藏大小和epsilon值
        # 注：变量名不使用蛇形命名以匹配 TensorFlow 模型变量名，方便加载 TensorFlow 的检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建 Dropout 层，用于在训练过程中随机失活一部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建并注册位置索引张量，用于将位置信息导入模型
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # 对于视觉特征处理
        # 创建视觉token类型嵌入层，用于映射视觉token类型索引到隐藏表示向量
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建视觉位置嵌入层，用于映射视觉位置索引到隐藏表示向量
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 如果需要特殊的视觉初始化
        if config.special_visual_initialize:
            # 使用语言模型的token类型嵌入初始化视觉token类型嵌入
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            # 使用语言模型的位置嵌入初始化视觉位置嵌入
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        # 创建视觉投影层，用于将视觉嵌入维度投影到隐藏表示向量维度
        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)
class VisualBertSelfAttention(nn.Module):
    # 定义 VisualBertSelfAttention 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化函数，接收一个配置对象 config
        super().__init__()
        # 要求隐藏大小 config.hidden_size 必须能被注意力头数 config.num_attention_heads 整除，同时不应具有 embedding_size 属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 对输入张量 x 进行维度重排，以便进行注意力分数计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 生成混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 通过 transpose_for_scores 函数，生成键和值的张量
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 通过 transpose_for_scores 函数，生成查询的张量
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算注意力分数，使用点积操作在 "查询" 和 "键" 之间
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 将注意力分数除以 sqrt(注意力头大小)，以进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果存在注意力遮罩，则将其应用于注意力分数
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 应用 dropout 操作到注意力概率上
        attention_probs = self.dropout(attention_probs)

        # 如果存在头遮罩，则将其应用于注意力概率
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，将注意力概率与值层进行加权求和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 重新排列上下文层的维度，使其返回到初始的 hidden_states 的维度
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 返回计算结果，如果需要输出注意力权重，则一并返回
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->VisualBert
class VisualBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征的大小映射为输出特征的大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，以一定的概率进行随机失活，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行特征变换
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 使用 LayerNorm 进行归一化，并将输入张量与变换后的隐藏状态相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建自注意力层对象
        self.self = VisualBertSelfAttention(config)
        # 创建输出层对象
        self.output = VisualBertSelfOutput(config)
        # 用于记录被剪枝的注意力头集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头，并返回头部索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 使用自注意力层处理隐藏状态
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 将自注意力层的输出传递给输出层处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力矩阵，则将其添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->VisualBert
class VisualBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征的大小映射为中间特征的大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行特征变换
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->VisualBert
# 定义 VisualBertOutput 类，继承自 nn.Module
class VisualBertOutput(nn.Module):
    # 初始化函数，接收 config 参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建 LayerNorm 层，对隐藏状态进行归一化，eps 参数为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，用于随机屏蔽神经元，防止过拟合，dropout 概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收 hidden_states 和 input_tensor 两个张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用 Dropout 层
        hidden_states = self.dropout(hidden_states)
        # 对处理后的隐藏状态应用 LayerNorm 层，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义 VisualBertLayer 类，继承自 nn.Module
class VisualBertLayer(nn.Module):
    # 初始化函数，接收 config 参数
    def __init__(self, config):
        super().__init__()
        # 设定 feed forward 过程中的 chunk 大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 创建 VisualBertAttention 对象
        self.attention = VisualBertAttention(config)
        # 创建 VisualBertIntermediate 对象
        self.intermediate = VisualBertIntermediate(config)
        # 创建 VisualBertOutput 对象
        self.output = VisualBertOutput(config)

    # 前向传播函数，接收 hidden_states、attention_mask、head_mask 等参数，返回处理后的输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 调用 self.attention 的前向传播函数，处理注意力机制
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取注意力输出
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，则将其添加到 outputs 中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 将注意力输出应用于前向传播的分块处理函数，处理后的结果作为层的输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将层输出添加到 outputs 中
        outputs = (layer_output,) + outputs

        return outputs

    # feed forward 的分块处理函数，接收 attention_output 作为输入，返回处理后的层输出
    def feed_forward_chunk(self, attention_output):
        # 调用 self.intermediate 处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 调用 self.output 处理 intermediate_output 和 attention_output，返回层输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# 定义 VisualBertEncoder 类，继承自 nn.Module
class VisualBertEncoder(nn.Module):
    # 初始化函数，接收 config 参数
    def __init__(self, config):
        super().__init__()
        # 将 config 存储在 self.config 中
        self.config = config
        # 创建 nn.ModuleList 存储多个 VisualBertLayer 层，层数为 config.num_hidden_layers
        self.layer = nn.ModuleList([VisualBertLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认禁用梯度检查点
        self.gradient_checkpointing = False

    # 前向传播函数，接收 hidden_states、attention_mask、head_mask 等参数，返回处理后的输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
        # 遍历每个层进行前向传播，处理 hidden_states，返回输出
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

        return hidden_states
        ):
        # 如果不输出隐藏状态，则将 all_hidden_states 初始化为空元组；否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则将 all_self_attentions 初始化为空元组；否则设为 None
        all_self_attentions = () if output_attentions else None

        # 遍历模型的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码（如果有的话）
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点且在训练模式下，使用梯度检查点函数进行前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层进行前向传播
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的注意力权重加入 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则最后将最终隐藏状态加入 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则将需要返回的值打包为元组返回
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        # 否则，以 BaseModelOutput 类型返回结果
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->VisualBert
class VisualBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取隐藏状态的第一个 token 对应的隐藏状态作为汇总输出
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态输入全连接层得到汇总输出
        pooled_output = self.dense(first_token_tensor)
        # 对汇总输出应用激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->VisualBert
class VisualBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 应用 LayerNorm 对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入隐藏状态到全连接层
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 对结果应用 LayerNorm
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->VisualBert
class VisualBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化预测头部变换层
        self.transform = VisualBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个 token 都有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 为每个 token 添加输出偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要在 `resize_token_embeddings` 时链接这两个变量，以便正确调整偏置
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 输入隐藏状态到变换层
        hidden_states = self.transform(hidden_states)
        # 将变换后的隐藏状态输入到解码器中
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->VisualBert
class VisualBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化预测头部
        self.predictions = VisualBertLMPredictionHead(config)
        # 序列关系分类任务的线性层
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 对序列输出进行预测
        prediction_scores = self.predictions(sequence_output)
        # 对序列关系进行分类
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class VisualBertPreTrainedModel(PreTrainedModel):
    """
    VisualBert 的预训练模型基类
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VisualBertConfig
    base_model_prefix = "visual_bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层或者嵌入层，使用正态分布初始化权重
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 与 TensorFlow 版本稍有不同，PyTorch 使用 normal_ 方法初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        # 如果模块是 LayerNorm 层，初始化偏置为零，权重为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 如果模块是线性层且有偏置项，初始化偏置为零
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    """
    Output type of `VisualBertForPreTraining`.

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the sentence-image prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the sentence-image prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    # 创建一个字符串，描述 VisualBert 模型的基本信息，没有任何特定的输出头部
    "The bare VisualBert Model transformer outputting raw hidden-states without any specific head on top.",
    # 使用 VISUAL_BERT_START_DOCSTRING 常量来继续完整的文档字符串
    VISUAL_BERT_START_DOCSTRING,
)

class VisualBertModel(VisualBertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Initialize VisualBert model components
        self.embeddings = VisualBertEmbeddings(config)  # VisualBert embeddings module
        self.encoder = VisualBertEncoder(config)        # VisualBert encoder module

        # Optionally add a pooling layer based on add_pooling_layer flag
        self.pooler = VisualBertPooler(config) if add_pooling_layer else None

        # Determine if to bypass the transformer and add an additional layer
        self.bypass_transformer = config.bypass_transformer
        if self.bypass_transformer:
            self.additional_layer = VisualBertLayer(config)  # Additional layer for bypassing transformer

        # Initialize weights and perform final processing
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
        # Iterate over layers and prune specified attention heads
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the VisualBert model.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs. Defaults to None.
            attention_mask (torch.LongTensor, optional): Mask to avoid performing attention on padding token indices.
                Defaults to None.
            token_type_ids (torch.LongTensor, optional): Segment token indices to differentiate image and text tokens.
                Defaults to None.
            position_ids (torch.LongTensor, optional): Indices of each input token in its position. Defaults to None.
            head_mask (torch.LongTensor, optional): Mask to nullify selected heads of the self-attention modules.
                Defaults to None.
            inputs_embeds (torch.FloatTensor, optional): Embedded input tokens. Defaults to None.
            visual_embeds (torch.FloatTensor, optional): Embedded visual features. Defaults to None.
            visual_attention_mask (torch.LongTensor, optional): Mask for visual features to avoid attending on padding.
                Defaults to None.
            visual_token_type_ids (torch.LongTensor, optional): Segment token indices for visual inputs. Defaults to None.
            image_text_alignment (torch.LongTensor, optional): Alignment between image and text tokens. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions weights. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary. Defaults to None.

        Returns:
            BaseModelOutputWithPooling: Output of the VisualBert model with optional pooling.
        """
        # Implementation of forward pass for VisualBert model
        pass

@add_start_docstrings(
    """
    VisualBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `sentence-image prediction (classification)` head.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForPreTraining(VisualBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传递配置对象
        super().__init__(config)

        # 创建 VisualBertModel 实例，传递配置对象作为参数
        self.visual_bert = VisualBertModel(config)
        # 创建 VisualBertPreTrainingHeads 实例，传递配置对象作为参数
        self.cls = VisualBertPreTrainingHeads(config)

        # 调用 post_init 方法，用于初始化权重并进行最终处理
        self.post_init()

    # 返回预测的输出嵌入层解码器
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层解码器的新值
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接受多个输入参数，包括输入的序列、注意力掩码、类型 ID 等
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=VisualBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        sentence_image_labels: Optional[torch.LongTensor] = None,
@add_start_docstrings(
    """
    VisualBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for VCR tasks.
    """,
    VISUAL_BERT_START_DOCSTRING,
)



class VisualBertForMultipleChoice(VisualBertPreTrainedModel):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 VisualBertModel 模型对象
        self.visual_bert = VisualBertModel(config)
        # Dropout 层，使用配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 全连接层，将隐藏层输出映射到 1 维，用于多选题任务
        self.cls = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，接收多个输入参数和一些可选参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,



@add_start_docstrings(
    """
    VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled
    output) for VQA.
    """,
    VISUAL_BERT_START_DOCSTRING,
)



class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 标签数量等于配置中的标签数
        self.num_labels = config.num_labels

        # 创建 VisualBertModel 模型对象
        self.visual_bert = VisualBertModel(config)
        # Dropout 层，使用配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 全连接层，将隐藏层输出映射到标签数维度，用于问答任务
        self.cls = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
        # 定义前向传播方法，接受多种输入参数，都是可选的Tensor类型
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # 输入的token IDs，用于模型输入序列的表示
        attention_mask: Optional[torch.LongTensor] = None,
        # 注意力掩码，指示哪些元素在计算注意力时被忽略
        token_type_ids: Optional[torch.LongTensor] = None,
        # 用于区分不同句子或段落的类型IDs
        position_ids: Optional[torch.LongTensor] = None,
        # 位置IDs，指示每个token在序列中的位置
        head_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，用于指定哪些注意力头部被屏蔽
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 输入的嵌入表示，替代input_ids的嵌入
        visual_embeds: Optional[torch.FloatTensor] = None,
        # 可视化输入的嵌入表示，例如图像的嵌入
        visual_attention_mask: Optional[torch.LongTensor] = None,
        # 可视化输入的注意力掩码
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        # 可视化输入的类型IDs
        image_text_alignment: Optional[torch.LongTensor] = None,
        # 图像与文本对齐信息
        output_attentions: Optional[bool] = None,
        # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,
        # 是否输出隐藏状态
        return_dict: Optional[bool] = None,
        # 是否返回字典形式的输出
        labels: Optional[torch.LongTensor] = None,
        # 标签，用于模型的监督学习训练
@add_start_docstrings(
    """
    VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled
    output) for Visual Reasoning e.g. for NLVR task.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForVisualReasoning(VisualBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize the VisualBertModel with the provided configuration
        self.visual_bert = VisualBertModel(config)
        
        # Dropout layer with dropout probability as specified in the configuration
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Linear layer for classification, mapping hidden_size to num_labels
        self.cls = nn.Linear(config.hidden_size, config.num_labels)  # 2

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
):
    class VisualBertRegionToPhraseAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            if config.hidden_size % config.num_attention_heads != 0:
                raise ValueError(
                    f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                    f"heads ({config.num_attention_heads})"
                )
            # Number of attention heads is set to 1 for this module
            self.num_attention_heads = 1  # config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # Linear transformations for query, key, and value vectors
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

            # Dropout layer for attention probabilities
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        def transpose_for_scores(self, x):
            # Reshape and permute dimensions for multi-head attention
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
    # 将注意力掩码转换为与查询张量相同的数据类型
    attention_mask = attention_mask.to(query.dtype)
    # 在张量的维度上扩展注意力掩码，以便与后续张量计算兼容
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    # 使用注意力掩码的逆来填充注意力分数张量，以便在softmax操作中不被考虑
    attention_mask = (1.0 - attention_mask) * torch.finfo(query.dtype).min

    # 使用查询网络层处理查询张量，生成混合查询层
    mixed_query_layer = self.query(query)
    # 使用键网络层处理键张量，生成混合键层
    mixed_key_layer = self.key(key)

    # 将混合查询层转置以便进行注意力得分计算
    query_layer = self.transpose_for_scores(mixed_query_layer)
    # 将混合键层转置以便进行注意力得分计算
    key_layer = self.transpose_for_scores(mixed_key_layer)

    # 计算注意力得分矩阵，使用查询层与键层的乘积
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    # 对注意力得分进行缩放，以减少梯度消失问题
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # 将注意力掩码添加到注意力得分中，屏蔽无效位置的注意力
    attention_scores = attention_scores + attention_mask

    # 去除不必要的维度，使得注意力得分张量的形状与预期一致
    attention_scores = attention_scores.squeeze(1)
    # 返回最终的注意力得分张量作为前向传播的输出
    return attention_scores
# 使用装饰器为该类添加文档字符串，描述其作为 VisualBert 模型的特性和用途，特别是在 Flickr30 Entities 任务中的应用。
@add_start_docstrings(
    """
    VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment
    e.g. for Flickr30 Entities task.
    """,
    VISUAL_BERT_START_DOCSTRING,  # 引用预定义的 VisualBert 文档字符串模板的一部分
)
# 定义 VisualBertForRegionToPhraseAlignment 类，继承自 VisualBertPreTrainedModel
class VisualBertForRegionToPhraseAlignment(VisualBertPreTrainedModel):
    # 定义 _tied_weights_keys 列表，用于记录需要绑定权重的键名
    _tied_weights_keys = ["cls.predictions.decoder.bias"]

    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 VisualBertModel 实例，传入 config 参数
        self.visual_bert = VisualBertModel(config)
        # 创建一个 dropout 层，使用配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建 VisualBertPreTrainingHeads 实例，用于预测任务
        self.cls = VisualBertPreTrainingHeads(config)
        # 创建 VisualBertRegionToPhraseAttention 实例，处理区域到短语的注意力对齐
        self.attention = VisualBertRegionToPhraseAttention(config)

        # 调用类的后初始化方法，可能用于权重初始化和最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述其输入参数和返回类型
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义 forward 方法，处理模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        region_to_phrase_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
```