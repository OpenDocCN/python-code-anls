# `.\models\lilt\modeling_lilt.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LiLT model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_lilt import LiltConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LiltConfig"

LILT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "SCUT-DLVCLab/lilt-roberta-en-base",
    # See all LiLT models at https://huggingface.co/models?filter=lilt
]


class LiltTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化词嵌入层，用于将输入词编号转换为向量表示
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，用于表示词的位置信息
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化类型嵌入层，用于表示输入的类型信息
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm 使用 nn.LayerNorm 进行层归一化，保持和 TensorFlow 模型变量名一致以便加载 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 使用 dropout 进行随机失活，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 注册一个持久化的 buffer，用于存储位置 ID，这些位置 ID 在序列化时会被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        # 设置填充标记的索引
        self.padding_idx = config.pad_token_id
        # 初始化位置嵌入层，用于表示词的位置信息，带有填充标记的索引设置
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # 如果位置 ids 为空且输入 ids 不为空，则从输入 token ids 创建位置 ids。任何填充的 token 保持填充状态。
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                # 如果位置 ids 为空且输入 ids 也为空，则从输入嵌入创建位置 ids。
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            # 如果输入 ids 不为空，获取其形状
            input_shape = input_ids.size()
        else:
            # 否则，获取输入嵌入的形状，去掉最后一个维度（即序列长度）
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            # 如果 token 类型 ids 为空，则创建全零的 token 类型 ids，与输入形状相同
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果输入嵌入为空，则通过输入 ids 获取单词嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入与 token 类型嵌入相加，得到整体嵌入
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            # 如果位置嵌入类型为 "absolute"，则添加绝对位置嵌入
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 应用 LayerNorm 对 embeddings 进行归一化
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回 embeddings 和 position_ids
        return embeddings, position_ids

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Args:
        非填充符号替换为它们的位置编号。位置编号从 padding_idx+1 开始。忽略填充符号。这是从 fairseq 的 `utils.make_positions` 修改而来。
            input_ids: torch.Tensor
            padding_idx: int
        Returns: torch.Tensor
        """
        # 创建一个 mask，标记出非填充符号位置
        mask = input_ids.ne(padding_idx).int()
        # 使用累加的方式生成位置 ids，确保在填充符号处保持填充状态
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        Args:
        我们直接提供嵌入。无法推断哪些是填充符号，因此只生成顺序的位置 ids。
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状，去掉最后一个维度得到序列长度
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序的位置 ids，从 padding_idx+1 开始到 sequence_length + padding_idx+1
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
class LiltLayoutEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 将隐藏大小除以6，因为有6种不同的布局嵌入：
        # 左侧位置、上侧位置、右侧位置、下侧位置、高度、宽度
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)

        self.padding_idx = config.pad_token_id
        # 使用config中的参数初始化嵌入层，设置padding_idx为padding标记的ID
        self.box_position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size // config.channel_shrink_ratio,
            padding_idx=self.padding_idx,
        )
        # 线性层，将隐藏大小映射到更小的尺寸，用于嵌入向量的线性变换
        self.box_linear_embeddings = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size // config.channel_shrink_ratio
        )
        # LayerNorm 层，用于归一化输入向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size // config.channel_shrink_ratio, eps=config.layer_norm_eps)
        # Dropout 层，用于随机丢弃输入向量的一部分，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, bbox=None, position_ids=None):
        try:
            # 从bbox中提取左侧、上侧、右侧、下侧位置的嵌入向量
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            # 抛出异常，如果bbox的坐标值不在0-1000范围内
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        # 计算高度和宽度的嵌入向量
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        # 拼接左侧、上侧、右侧、下侧、高度、宽度的嵌入向量
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        # 对拼接的嵌入向量进行线性变换
        spatial_position_embeddings = self.box_linear_embeddings(spatial_position_embeddings)
        # 获取位置ID对应的位置嵌入向量
        box_position_embeddings = self.box_position_embeddings(position_ids)

        # 将位置嵌入向量加到拼接的嵌入向量上
        spatial_position_embeddings = spatial_position_embeddings + box_position_embeddings

        # 对加和后的嵌入向量进行LayerNorm归一化
        spatial_position_embeddings = self.LayerNorm(spatial_position_embeddings)
        # 对归一化后的嵌入向量进行Dropout操作
        spatial_position_embeddings = self.dropout(spatial_position_embeddings)

        # 返回最终的空间位置嵌入向量
        return spatial_position_embeddings
    # 初始化函数，接收配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类初始化方法
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，同时隐藏层大小不具有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 抛出数值错误，显示隐藏层大小不能被注意力头数整除
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义布局查询、键、值的线性层
        self.layout_query = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )
        self.layout_key = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )
        self.layout_value = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )

        # 定义 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，如果未提供则默认为绝对位置
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置通道缩减比例
        self.channel_shrink_ratio = config.channel_shrink_ratio
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class LiltSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，将输入的hidden_size维度映射到hidden_size维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm层，用于对输入进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout层，用于在训练过程中随机将一部分输入置为0，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层操作，将hidden_states映射到相同维度
        hidden_states = self.dense(hidden_states)
        # Dropout操作，随机置0
        hidden_states = self.dropout(hidden_states)
        # LayerNorm操作，对映射后的结果进行归一化处理并与原始输入相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LiltAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # LiltSelfAttention模块，用于计算注意力机制
        self.self = LiltSelfAttention(config, position_embedding_type=position_embedding_type)
        # LiltSelfOutput模块，用于处理自注意力的输出
        self.output = LiltSelfOutput(config)
        # 用于存储被剪枝的注意力头索引
        self.pruned_heads = set()

        # 保存原始的hidden_size，并根据channel_shrink_ratio调整hidden_size大小
        ori_hidden_size = config.hidden_size
        config.hidden_size = config.hidden_size // config.channel_shrink_ratio
        # 用于处理布局输入的LiltSelfOutput模块
        self.layout_output = LiltSelfOutput(config)
        config.hidden_size = ori_hidden_size

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        # 如果没有要剪枝的头部，则直接返回
        if len(heads) == 0:
            return
        # 调用帮助函数找到可剪枝的头部和对应索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对线性层进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头部索引
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用self模块的forward方法，计算自注意力机制
        self_outputs = self.self(
            hidden_states,
            layout_inputs,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 对自注意力的输出进行处理，传入self.output模块
        attention_output = self.output(self_outputs[0][0], hidden_states)
        # 对布局注意力的输出进行处理，传入self.layout_output模块
        layout_attention_output = self.layout_output(self_outputs[0][1], layout_inputs)
        # 如果有需要，则添加注意力输出到结果中
        outputs = ((attention_output, layout_attention_output),) + self_outputs[1:]
        return outputs
# 定义 LiltLayer 类，继承自 nn.Module，表示一个自定义的神经网络层
class LiltLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 LiltLayer 类，设置一些基本属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 设定前向传播的块大小
        self.seq_len_dim = 1  # 序列长度维度为 1
        self.attention = LiltAttention(config)  # 初始化注意力层对象
        self.intermediate = LiltIntermediate(config)  # 初始化中间层对象
        self.output = LiltOutput(config)  # 初始化输出层对象

        # 保存原始的隐藏大小和中间大小
        ori_hidden_size = config.hidden_size
        ori_intermediate_size = config.intermediate_size

        # 根据配置调整隐藏大小和中间大小
        config.hidden_size = config.hidden_size // config.channel_shrink_ratio
        config.intermediate_size = config.intermediate_size // config.channel_shrink_ratio

        # 创建新的中间层和输出层对象，用于布局处理
        self.layout_intermediate = LiltIntermediate(config)
        self.layout_output = LiltOutput(config)

        # 恢复原始的隐藏大小和中间大小
        config.hidden_size = ori_hidden_size
        config.intermediate_size = ori_intermediate_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        ):
        # 前向传播方法定义，接收隐藏状态、布局输入以及可选的注意力掩码和头部掩码
    # 定义函数，接受多个参数并返回一个元组，包含一个 torch.Tensor 对象
    ) -> Tuple[torch.Tensor]:
        # 调用 self.attention 方法，传入多个参数
        # hidden_states: 隐藏状态
        # layout_inputs: 布局输入
        # attention_mask: 注意力掩码
        # head_mask: 头部掩码
        # output_attentions: 是否输出注意力权重，默认为 False
        self_attention_outputs = self.attention(
            hidden_states,
            layout_inputs,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 从 self_attention_outputs 中获取第一个元素的第一个元素，即 self attention 的输出
        attention_output = self_attention_outputs[0][0]
        # 从 self_attention_outputs 中获取第一个元素的第二个元素，即 layout attention 的输出
        layout_attention_output = self_attention_outputs[0][1]

        # 如果输出注意力权重，则将 self attentions 添加到输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 调用 apply_chunking_to_forward 函数，对 self.feed_forward_chunk 进行分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 调用 apply_chunking_to_forward 函数，对 self.layout_feed_forward_chunk 进行分块处理
        layout_layer_output = apply_chunking_to_forward(
            self.layout_feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, layout_attention_output
        )
        # 将处理后的输出添加到 outputs 元组中
        outputs = ((layer_output, layout_layer_output),) + outputs

        # 返回最终的输出元组
        return outputs

    # 从 transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk 复制过来的函数
    def feed_forward_chunk(self, attention_output):
        # 调用 self.intermediate 方法，对 attention_output 进行处理
        intermediate_output = self.intermediate(attention_output)
        # 调用 self.output 方法，对 intermediate_output 和 attention_output 进行处理，得到最终的输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理后的输出
        return layer_output

    # 定义的函数，处理 layout attention 的输出
    def layout_feed_forward_chunk(self, attention_output):
        # 调用 self.layout_intermediate 方法，对 attention_output 进行处理
        intermediate_output = self.layout_intermediate(attention_output)
        # 调用 self.layout_output 方法，对 intermediate_output 和 attention_output 进行处理，得到最终的输出
        layer_output = self.layout_output(intermediate_output, attention_output)
        # 返回处理后的输出
        return layer_output
# 声明一个名为 LiltEncoder 的类，继承自 nn.Module
class LiltEncoder(nn.Module):
    # 初始化函数，接受一个配置参数 config
    # 从 transformers.models.bert.modeling_bert.BertEncoder.__init__ 复制而来，将 Bert 替换为 Lilt
    def __init__(self, config):
        super().__init__()
        # 将配置参数保存到实例变量中
        self.config = config
        # 创建一个 nn.ModuleList，其中包含 config.num_hidden_layers 个 LiltLayer 实例
        self.layer = nn.ModuleList([LiltLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点功能为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接受多个输入参数和可选的返回类型注解
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        layout_inputs: torch.Tensor,  # 布局输入张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力掩码张量，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 可选的头部掩码张量，默认为 None
        output_attentions: Optional[bool] = False,  # 是否输出注意力张量的开关，默认为 False
        output_hidden_states: Optional[bool] = False,  # 是否输出所有隐藏状态的开关，默认为 False
        return_dict: Optional[bool] = True,  # 是否返回字典形式的结果，默认为 True
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:  # 返回类型为元组或 BaseModelOutput 类型

        # 如果需要输出隐藏状态，则初始化空的所有隐藏状态元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力，则初始化空的所有自注意力元组
        all_self_attentions = () if output_attentions else None

        # 遍历 self.layer 中的每个层次模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果梯度检查点开启且处于训练状态，则使用 _gradient_checkpointing_func 函数来计算层输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layout_inputs,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则，直接调用层模块的 __call__ 方法计算层输出
                layer_outputs = layer_module(
                    hidden_states,
                    layout_inputs,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            # 更新隐藏状态和布局输入为当前层的输出结果的第一个元素和第二个元素
            hidden_states = layer_outputs[0][0]
            layout_inputs = layer_outputs[0][1]

            # 如果需要输出注意力，则将当前层的注意力张量添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则返回非 None 的所有值的元组
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
        # 否则，返回一个 BaseModelOutput 类型的对象，包含最终的隐藏状态、所有隐藏状态和所有自注意力
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# 从 transformers.models.bert.modeling_bert.BertPooler 复制而来的 LiltPooler 类
class LiltPooler(nn.Module):
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 Tanh 激活函数实例
        self.activation = nn.Tanh()
    # 定义一个前向传播方法，接收隐藏状态作为输入，并返回转换后的张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过选择第一个标记对应的隐藏状态来“汇聚”模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态输入全连接层，进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 对线性变换后的结果应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回经过汇聚和激活处理后的输出张量
        return pooled_output
class LiltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 LiltConfig 作为配置类
    config_class = LiltConfig
    # 模型的前缀名称
    base_model_prefix = "lilt"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不进行模块分割的模块列表
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重，均值为0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，均值为0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了填充索引，将对应索引的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，初始化偏置为零，初始化权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


LILT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LiltConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LILT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare LiLT Model transformer outputting raw hidden-states without any specific head on top.",
    LILT_START_DOCSTRING,
)
class LiltModel(LiltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        # 设置模型配置
        self.config = config

        # 初始化嵌入层、布局嵌入和编码器
        self.embeddings = LiltTextEmbeddings(config)
        self.layout_embeddings = LiltLayoutEmbeddings(config)
        self.encoder = LiltEncoder(config)

        # 如果需要，添加池化层
        self.pooler = LiltPooler(config) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入层的权重
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的权重
        self.embeddings.word_embeddings = value
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 字典中的每个元素，其中 key 是层号，value 是需要剪枝的头部列表
        for layer, heads in heads_to_prune.items():
            # 在编码器的指定层中的注意力模型中执行剪枝操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        """
        Performs forward pass of the model. Args:
            input_ids (Optional[torch.Tensor], optional): Input tensors for the model.
            bbox (Optional[torch.Tensor], optional): Bounding box tensors.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensors.
            token_type_ids (Optional[torch.Tensor], optional): Token type ID tensors.
            position_ids (Optional[torch.Tensor], optional): Position ID tensors.
            head_mask (Optional[torch.Tensor], optional): Head mask tensors.
            inputs_embeds (Optional[torch.Tensor], optional): Embedded input tensors.
            output_attentions (Optional[bool], optional): Whether to output attentions.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states.
            return_dict (Optional[bool], optional): Whether to return as dictionary.
        """
        # 实现模型的前向传播，接收和处理各种输入张量
        # 具体参数作用见函数说明文档和相关注释
        pass
@add_start_docstrings(
    """
    LiLT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    LILT_START_DOCSTRING,
)
class LiltForSequenceClassification(LiltPreTrainedModel):
    # 从transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.__init__复制而来，将Roberta替换为Lilt，roberta替换为lilt
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 初始化分类标签数
        self.config = config

        self.lilt = LiltModel(config, add_pooling_layer=False)  # 初始化Lilt模型，不添加池化层
        self.classifier = LiltClassificationHead(config)  # 初始化分类头部

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """
    Lilt Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    LILT_START_DOCSTRING,
)
class LiltForTokenClassification(LiltPreTrainedModel):
    # 从transformers.models.roberta.modeling_roberta.RobertaForTokenClassification.__init__复制而来，将Roberta替换为Lilt，roberta替换为lilt
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 初始化分类标签数

        self.lilt = LiltModel(config, add_pooling_layer=False)  # 初始化Lilt模型，不添加池化层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 初始化Dropout层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 初始化线性分类器层

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID序列，可以为空
        bbox: Optional[torch.LongTensor] = None,  # 包围框信息的张量，可以为空
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码张量，可以为空
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型ID张量，可以为空
        position_ids: Optional[torch.LongTensor] = None,  # 位置ID张量，可以为空
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码张量，可以为空
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入张量，可以为空
        labels: Optional[torch.LongTensor] = None,  # 用于计算标记分类损失的标签张量，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力张量的标志，可以为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的标志，可以为空
        return_dict: Optional[bool] = None,  # 是否返回字典类型的输出，可以为空
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        返回Lilt模型的前向传播结果。

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算标记分类损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。

        Returns:
            如果 `return_dict=False`：
                返回一个包含 `(logits, hidden_states, attentions)` 的元组，其中 `logits` 是预测的分类结果张量。
                如果 `loss` 不为空，则还包含 `loss`。

            如果 `return_dict=True`：
                返回一个 `TokenClassifierOutput` 对象，包含 `loss`、`logits`、`hidden_states` 和 `attentions` 属性。

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
        >>> model = AutoModelForTokenClassification.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = tokenizer(words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> predicted_class_indices = outputs.logits.argmax(-1)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 确定是否使用模型配置中的返回字典选项

        outputs = self.lilt(
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

        sequence_output = outputs[0]  # 获取模型输出的序列输出

        sequence_output = self.dropout(sequence_output)  # 对序列输出应用dropout操作
        logits = self.classifier(sequence_output)  # 使用分类器对序列输出进行分类

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行计算
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算交叉熵损失

        if not return_dict:
            output = (logits,) + outputs[2:]  # 构建输出元组
            return ((loss,) + output) if loss is not None else output  # 如果有损失则返回损失和输出，否则只返回输出

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  # 返回TokenClassifierOutput对象，包含损失、logits、隐藏状态和注意力
# 从 transformers.models.roberta.modeling_roberta.RobertaClassificationHead 复制代码，并将 Roberta 替换为 Lilt
class LiltClassificationHead(nn.Module):
    """用于句子级分类任务的头部模块。"""

    def __init__(self, config):
        super().__init__()
        # 全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的 dropout 概率，默认为 config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 最终的输出全连接层，输出维度是 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取序列中第一个 token 的隐藏状态作为特征
        x = features[:, 0, :]  # 相当于取 <s> token (等同于 [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    在 Lilt 模型顶部添加用于提取式问答任务的 span 分类头部（在隐藏状态输出之上的线性层，计算 `span start logits` 和 `span end logits`）。
    """,
    LILT_START_DOCSTRING,
)
class LiltForQuestionAnswering(LiltPreTrainedModel):
    # 从 transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.__init__ 复制代码，并将 Roberta 替换为 Lilt, roberta 替换为 lilt
    def __init__(self, config):
        super().__init__(config)
        # 设置模型的标签数目
        self.num_labels = config.num_labels

        # 使用 LiltModel 初始化，不添加汇聚层
        self.lilt = LiltModel(config, add_pooling_layer=False)
        # 输出层，线性层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
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
```