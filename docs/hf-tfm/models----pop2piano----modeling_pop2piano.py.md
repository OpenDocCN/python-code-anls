# `.\models\pop2piano\modeling_pop2piano.py`

```py
# coding=utf-8
# Copyright 2023 The Pop2Piano Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Pop2Piano model."""

import copy
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.generation import GenerationConfig

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from .configuration_pop2piano import Pop2PianoConfig

logger = logging.get_logger(__name__)

_load_pop2piano_layer_norm = True

try:
    from apex.normalization import FusedRMSNorm

    _load_pop2piano_layer_norm = False

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of Pop2PianoLayerNorm")
except ImportError:
    # using the normal Pop2PianoLayerNorm
    pass
except Exception:
    logger.warning("Discovered apex but it failed to load, falling back to Pop2PianoLayerNorm")
    pass


_CONFIG_FOR_DOC = "Pop2PianoConfig"
_CHECKPOINT_FOR_DOC = "sweetcocoa/pop2piano"

POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sweetcocoa/pop2piano",
    # See all Pop2Piano models at https://huggingface.co/models?filter=pop2piano
]


POP2PIANO_INPUTS_DOCSTRING = r"""
"""


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Pop2Piano
class Pop2PianoLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the Pop2Piano style. No bias and no subtraction of mean.
        """
        super().__init__()
        # Initialize the weight parameter with ones (no bias)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Set the epsilon value for numerical stability in variance calculation
        self.variance_epsilon = eps
    # 定义一个前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # Pop2Piano 使用一种只进行缩放而不进行偏移的层归一化，也称为均方根层归一化
        # 参考论文 https://arxiv.org/abs/1910.07467 ，因此方差是在没有均值和偏差的情况下计算的。
        # 另外，我们希望确保对半精度输入的累积是在 fp32 中完成的。

        # 计算隐藏状态的方差，将隐藏状态转换为 torch.float32 类型，然后平方并在最后一个维度上取平均值
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 使用归一化的方差对隐藏状态进行归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型是 torch.float16 或 torch.bfloat16，则将隐藏状态转换为相应的半精度类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回加权的隐藏状态
        return self.weight * hidden_states
# 如果 `_load_pop2piano_layer_norm` 为假，将 `Pop2PianoLayerNorm` 设置为 `FusedRMSNorm` 类。
if not _load_pop2piano_layer_norm:
    Pop2PianoLayerNorm = FusedRMSNorm  # noqa

# 将 `Pop2PianoLayerNorm` 添加到 `ALL_LAYERNORM_LAYERS` 列表中
ALL_LAYERNORM_LAYERS.append(Pop2PianoLayerNorm)


# 从 `transformers.models.t5.modeling_t5.T5DenseActDense` 复制，并修改为 `Pop2PianoDenseActDense`，同时将 `T5` 修改为 `Pop2Piano`，`t5` 修改为 `pop2piano`
class Pop2PianoDenseActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig):
        super().__init__()
        # 初始化线性层 `wi`，输入维度为 `config.d_model`，输出维度为 `config.d_ff`，无偏置
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化线性层 `wo`，输入维度为 `config.d_ff`，输出维度为 `config.d_model`，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 初始化丢弃层，使用 `config.dropout_rate` 的丢弃率
        self.dropout = nn.Dropout(config.dropout_rate)
        # 选择激活函数，根据配置选择 `ACT2FN` 中对应的函数
        self.act = ACT2FN[config.dense_act_fn]

    # 前向传播函数，接收 `hidden_states` 作为输入
    def forward(self, hidden_states):
        # 输入 `hidden_states` 到 `wi` 线性层，得到输出 `hidden_states`
        hidden_states = self.wi(hidden_states)
        # 对 `hidden_states` 应用激活函数 `act`
        hidden_states = self.act(hidden_states)
        # 对 `hidden_states` 应用丢弃层
        hidden_states = self.dropout(hidden_states)
        # 如果 `self.wo.weight` 是 `torch.Tensor` 类型，并且 `hidden_states` 的数据类型与 `self.wo.weight` 的数据类型不同，且 `self.wo.weight` 的数据类型不是 `torch.int8`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将 `hidden_states` 转换为 `self.wo.weight` 的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # 输入 `hidden_states` 到 `wo` 线性层，得到最终输出 `hidden_states`
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 从 `transformers.models.t5.modeling_t5.T5DenseGatedActDense` 复制，并修改为 `Pop2PianoDenseGatedActDense`，同时将 `T5` 修改为 `Pop2Piano`
class Pop2PianoDenseGatedActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig):
        super().__init__()
        # 初始化两个线性层 `wi_0` 和 `wi_1`，输入维度为 `config.d_model`，输出维度为 `config.d_ff`，无偏置
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化线性层 `wo`，输入维度为 `config.d_ff`，输出维度为 `config.d_model`，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 初始化丢弃层，使用 `config.dropout_rate` 的丢弃率
        self.dropout = nn.Dropout(config.dropout_rate)
        # 选择激活函数，根据配置选择 `ACT2FN` 中对应的函数
        self.act = ACT2FN[config.dense_act_fn]

    # 前向传播函数，接收 `hidden_states` 作为输入
    def forward(self, hidden_states):
        # 将 `hidden_states` 输入到 `wi_0` 线性层，应用激活函数后得到 `hidden_gelu`
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 将 `hidden_states` 输入到 `wi_1` 线性层，得到 `hidden_linear`
        hidden_linear = self.wi_1(hidden_states)
        # 将 `hidden_gelu` 与 `hidden_linear` 相乘得到 `hidden_states`
        hidden_states = hidden_gelu * hidden_linear
        # 对 `hidden_states` 应用丢弃层
        hidden_states = self.dropout(hidden_states)

        # 若要使得 8 位量化适用于 google/flan-t5-xxl，保持 `self.wo` 为 `float32`
        # 参见 https://github.com/huggingface/transformers/issues/20287
        # 确保权重不是 `int8` 类型，以防用户强制设置 `_keep_in_fp32_modules` 为 `None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将 `hidden_states` 转换为 `self.wo.weight` 的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 输入 `hidden_states` 到 `wo` 线性层，得到最终输出 `hidden_states`
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 从 `transformers.models.t5.modeling_t5.T5LayerFF` 复制，并修改为 `Pop2PianoLayerFF`，同时将 `T5` 修改为 `Pop2Piano`
class Pop2PianoLayerFF(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config: Pop2PianoConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置中的是否启用门控激活函数的标志，选择不同的神经网络层结构
        if config.is_gated_act:
            self.DenseReluDense = Pop2PianoDenseGatedActDense(config)
        else:
            self.DenseReluDense = Pop2PianoDenseActDense(config)

        # 初始化层归一化对象，设置归一化的维度和 epsilon 值
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化 dropout 层，设置 dropout 的丢弃率
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播方法，接受隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states):
        # 对输入的隐藏状态进行层归一化处理
        forwarded_states = self.layer_norm(hidden_states)
        # 将归一化后的状态输入到 DenseReluDense 网络中进行处理
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 将原始隐藏状态与 dropout 处理后的输出相加，得到最终的隐藏状态
        hidden_states = hidden_states + self.dropout(forwarded_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.t5.modeling_t5.T5Attention中复制而来，用于Pop2Piano模型的注意力机制实现
class Pop2PianoAttention(nn.Module):
    def __init__(self, config: Pop2PianoConfig, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder  # 标记是否为解码器
        self.has_relative_attention_bias = has_relative_attention_bias  # 是否包含相对注意力偏置
        self.relative_attention_num_buckets = config.relative_attention_num_buckets  # 相对注意力偏置的桶数
        self.relative_attention_max_distance = config.relative_attention_max_distance  # 相对注意力的最大距离
        self.d_model = config.d_model  # 模型的维度
        self.key_value_proj_dim = config.d_kv  # 键值投影的维度
        self.n_heads = config.num_heads  # 注意力头的数量
        self.dropout = config.dropout_rate  # Dropout率
        self.inner_dim = self.n_heads * self.key_value_proj_dim  # 内部维度，注意力头乘以投影维度

        # 使用线性层定义查询(q), 键(k), 值(v)和输出(o)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            # 如果需要相对注意力偏置，使用Embedding层来存储偏置信息
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()  # 初始化被修剪的注意力头集合为空
        self.gradient_checkpointing = False  # 梯度检查点标志设置为False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可修剪的注意力头和它们的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 修剪线性层
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - 相对位置的整数张量
            bidirectional: a boolean - 是否是双向注意力
            num_buckets: an integer - 桶的数量
            max_distance: an integer - 最大距离限制

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
            返回一个形状与relative_position相同的张量，包含在区间[0, num_buckets)内的int32值
        """
        relative_buckets = 0  # 初始化相对位置桶号为0

        if bidirectional:
            num_buckets //= 2  # 如果是双向注意力，桶的数量减半
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            # 如果相对位置大于0，则加上一半的桶数作为桶偏移量
            relative_position = torch.abs(relative_position)  # 取相对位置的绝对值
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
            # 如果是单向注意力，将相对位置限制为非正数

        # 现在相对位置范围为[0, inf)

        # 小部分的桶用于准确的位置增量
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半桶用于位置对数级别增大，直到max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        # 根据is_small条件选择桶号，累加到相对位置桶号中

        return relative_buckets  # 返回相对位置桶号的张量
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备未指定，使用 self.relative_attention_bias 的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建一个形状为 (query_length, 1) 的张量，表示查询序列的位置
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建一个形状为 (1, key_length) 的张量，表示记忆序列的位置
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置偏差，形状为 (query_length, key_length)
        relative_position = memory_position - context_position
        # 将相对位置映射到桶中，返回形状为 (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用 self.relative_attention_bias 对相对位置桶进行加权，形状变为 (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 将结果进行维度变换，形状变为 (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # 返回最终的相对位置偏差张量
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->Pop2Piano,t5->pop2piano
class Pop2PianoLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化自注意力层，使用 Pop2PianoAttention 模块
        self.SelfAttention = Pop2PianoAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 初始化层归一化模块，使用 Pop2PianoLayerNorm 进行归一化
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化 dropout 模块，丢弃率为 config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 对输入的 hidden_states 进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用 SelfAttention 进行自注意力计算
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将原始 hidden_states 和 dropout 后的 attention_output 相加作为最终的输出
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，包含更新后的 hidden_states 和可能的 attention 输出
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->Pop2Piano,t5->pop2piano
class Pop2PianoLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化编码解码注意力层，使用 Pop2PianoAttention 模块
        self.EncDecAttention = Pop2PianoAttention(config, has_relative_attention_bias=False)
        # 初始化层归一化模块，使用 Pop2PianoLayerNorm 进行归一化
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化 dropout 模块，丢弃率为 config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # 对输入的 hidden_states 进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用 EncDecAttention 进行编码解码注意力计算
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 将原始 hidden_states 和 dropout 后的 attention_output 相加作为最终的输出
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，包含更新后的 layer_output 和可能的 attention 输出
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5Block with T5->Pop2Piano,t5->pop2piano
class Pop2PianoBlock(nn.Module):
    # 初始化方法，接受配置参数和是否包含相对注意力偏置的标志
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置设置当前模块是否为解码器
        self.is_decoder = config.is_decoder
        # 创建一个空的模块列表，用于存储不同层的模块
        self.layer = nn.ModuleList()
        # 向模块列表中添加一个自注意力层，使用Pop2PianoLayerSelfAttention类初始化
        self.layer.append(Pop2PianoLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        # 如果当前模块是解码器，向模块列表中添加一个交叉注意力层
        if self.is_decoder:
            self.layer.append(Pop2PianoLayerCrossAttention(config))

        # 向模块列表中添加一个Feed Forward层
        self.layer.append(Pop2PianoLayerFF(config))

    # 前向传播方法，接收多个参数来执行前向计算
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
# 定义一个继承自PreTrainedModel的抽象类，用于处理权重初始化和预训练模型的下载与加载接口
class Pop2PianoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为Pop2PianoConfig
    config_class = Pop2PianoConfig
    # 基础模型的前缀，用于命名
    base_model_prefix = "transformer"
    # 不支持模型并行化
    is_parallelizable = False
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = ["Pop2PianoBlock"]
    # 需要保持在fp32精度的模块列表
    _keep_in_fp32_modules = ["wo"]

    # 将输入的ids向右移动一位的方法
    def _shift_right(self, input_ids):
        # 获取解码器起始标记id
        decoder_start_token_id = self.config.decoder_start_token_id
        # 获取填充标记id
        pad_token_id = self.config.pad_token_id

        # 如果解码器起始标记id未定义，则抛出数值错误
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In Pop2Piano it is usually set to the pad_token_id."
            )

        # 将输入向右移动一位
        if is_torch_fx_proxy(input_ids):
            # 对于torch.fx代理，不支持原生的项目分配
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 如果填充标记id未定义，则抛出数值错误
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 将标签中可能存在的-100值替换为填充标记id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


# 定义一个继承自Pop2PianoPreTrainedModel的类Pop2PianoStack
class Pop2PianoStack(Pop2PianoPreTrainedModel):
    # 从transformers.models.t5.modeling_t5.T5Stack.__init__中复制而来，修改为Pop2PianoStack
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        # 嵌入标记，可以是None
        self.embed_tokens = embed_tokens
        # 是否是解码器
        self.is_decoder = config.is_decoder

        # 使用列表推导式创建模块列表block，每个Pop2PianoBlock都有一个相对注意偏置
        self.block = nn.ModuleList(
            [Pop2PianoBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        # 最终层的LayerNorm
        self.final_layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化权重并应用最终处理
        self.post_init()
        # 模型并行化，默认为False
        self.model_parallel = False
        # 设备映射，默认为None
        self.device_map = None
        # 梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 从transformers.models.t5.modeling_t5.T5Stack.get_input_embeddings中复制而来
    def get_input_embeddings(self):
        return self.embed_tokens

    # 从transformers.models.t5.modeling_t5.T5Stack.set_input_embeddings中复制而来
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
    # 定义模型的前向传播方法，接收多个输入参数
    def forward(
        self,
        input_ids=None,  # 输入的 token IDs
        attention_mask=None,  # 自注意力机制的掩码，指示哪些 token 应该被忽略
        encoder_hidden_states=None,  # 编码器的隐藏状态（通常用于 Transformer 架构）
        encoder_attention_mask=None,  # 编码器的注意力掩码（如果有的话）
        inputs_embeds=None,  # 输入的嵌入表示（如不直接传入 token IDs 而是其它形式的输入）
        head_mask=None,  # 头部掩码，用于遮蔽特定的注意力头
        cross_attn_head_mask=None,  # 跨注意力头的掩码
        past_key_values=None,  # 过去的键值对，用于支持自回归生成
        use_cache=None,  # 是否使用缓存加速
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出隐藏状态
        return_dict=None,  # 是否返回一个字典作为输出
class Pop2PianoConcatEmbeddingToMel(nn.Module):
    """Embedding Matrix for `composer` tokens."""

    def __init__(self, config):
        super().__init__()
        # 使用 nn.Embedding 创建一个嵌入矩阵，用于存储 `composer` tokens 的嵌入向量
        self.embedding = nn.Embedding(num_embeddings=config.composer_vocab_size, embedding_dim=config.d_model)

    def forward(self, feature, index_value, embedding_offset):
        # 根据给定的偏移量调整索引值
        index_shifted = index_value - embedding_offset
        # 通过嵌入层获取对应的 `composer` tokens 的嵌入向量，并添加一个维度
        composer_embedding = self.embedding(index_shifted).unsqueeze(1)
        # 将 composer_embedding 和输入特征 feature 在维度 1 进行连接
        inputs_embeds = torch.cat([composer_embedding, feature], dim=1)
        return inputs_embeds


Pop2Piano_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Pop2PianoConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings("""Pop2Piano Model with a `language modeling` head on top.""", Pop2Piano_START_DOCSTRING)
class Pop2PianoForConditionalGeneration(Pop2PianoPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: Pop2PianoConfig):
        super().__init__(config)
        self.config = config
        self.model_dim = config.d_model

        # 创建一个共享的嵌入层，用于模型的输入和输出
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 创建一个 Pop2PianoConcatEmbeddingToMel 类的实例，用于处理 composer tokens 的嵌入
        self.mel_conditioner = Pop2PianoConcatEmbeddingToMel(config)

        # 初始化编码器和解码器的配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        # 创建编码器堆栈
        self.encoder = Pop2PianoStack(encoder_config, self.shared)

        # 初始化解码器的配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        # 创建解码器堆栈
        self.decoder = Pop2PianoStack(decoder_config, self.shared)

        # 创建语言模型头部，用于输出预测的下一个 token
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        # 更新共享嵌入层的嵌入向量
        self.shared = new_embeddings
        # 更新编码器和解码器的输入嵌入层
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
    # 设置新的输出嵌入层，用于语言模型的生成
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取当前的输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 获取编码器（encoder）模型
    def get_encoder(self):
        return self.encoder

    # 获取解码器（decoder）模型
    def get_decoder(self):
        return self.decoder

    # 获取 Mel conditioner 输出，用于在生成模型中控制 MIDI token 的类型
    def get_mel_conditioner_outputs(
        self,
        input_features: torch.FloatTensor,
        composer: str,
        generation_config: GenerationConfig,
        attention_mask: torch.FloatTensor = None,
    ):
        """
        This method is used to concatenate mel conditioner tokens at the front of the input_features in order to
        control the type of MIDI token generated by the model.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                input features extracted from the feature extractor.
            composer (`str`):
                composer token which determines the type of MIDI tokens to be generated.
            generation_config (`~generation.GenerationConfig`):
                The generation is used to get the composer-feature_token pair.
            attention_mask (`torch.FloatTensor`, *optional*):
                For batched generation, input_features are padded to have the same shape across all examples.
                `attention_mask` helps determine which areas were padded and which were not:
                - 1 for tokens that are **not padded**,
                - 0 for tokens that are **padded**.
        """
        # 获取 composer 对应的 feature_token 值
        composer_to_feature_token = generation_config.composer_to_feature_token
        # 如果 composer 不在 composer_to_feature_token 的键中，抛出 ValueError
        if composer not in composer_to_feature_token.keys():
            raise ValueError(
                f"Please choose a composer from {list(composer_to_feature_token.keys())}. Composer received - {composer}"
            )
        # 获取 composer 对应的值，并将其转换为 torch.Tensor
        composer_value = composer_to_feature_token[composer]
        composer_value = torch.tensor(composer_value, device=self.device)
        # 将 composer_value 在 batch 维度上重复，以便与 input_features 对齐
        composer_value = composer_value.repeat(input_features.shape[0])

        # 获取最小的 embedding offset
        embedding_offset = min(composer_to_feature_token.values())

        # 调用 self.mel_conditioner 方法，添加 composer_value 到 input_features 的前部
        input_features = self.mel_conditioner(
            feature=input_features,
            index_value=composer_value,
            embedding_offset=embedding_offset,
        )
        # 如果存在 attention_mask，则根据其值对 input_features 进行调整
        if attention_mask is not None:
            input_features[~attention_mask[:, 0].bool()] = 0.0

            # 由于 self.mel_conditioner 在 inputs_embeds 前添加了一个新数组，需要对 attention_mask 做同样处理以保持形状一致
            attention_mask = torch.cat([attention_mask[:, 0].view(-1, 1), attention_mask], dim=1)
            return input_features, attention_mask

        # 如果 attention_mask 为 None，则返回调整后的 input_features 和 None
        return input_features, None

    # 添加文档字符串到模型的前向方法，用于描述 POP2PIANO_INPUTS_DOCSTRING
    # 替换返回文档字符串，输出类型为 Seq2SeqLMOutput，配置类为 _CONFIG_FOR_DOC
    @torch.no_grad()
    def generate(
        self,
        input_features,
        attention_mask=None,
        composer="composer1",
        generation_config=None,
        **kwargs,
    ):
        # 生成器函数，用于生成模型的输出结果
        # input_features: 输入特征
        # attention_mask: 注意力掩码，控制哪些位置的输入会被模型关注
        # composer: 生成器的名称，默认为"composer1"
        # generation_config: 生成配置，控制生成过程的参数配置
        # **kwargs: 其他可选参数

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 为生成过程准备模型输入
        # input_ids: 输入的token IDs
        # past_key_values: 过去的键值，用于生成过程中的速度和效率
        # attention_mask: 注意力掩码，控制哪些位置的输入会被模型关注
        # head_mask: 头部掩码，控制哪些注意力头部会被激活
        # decoder_head_mask: 解码器头部掩码，控制解码器的注意力头部
        # cross_attn_head_mask: 跨注意力头部掩码，控制跨注意力模块的头部
        # use_cache: 是否使用缓存，提高生成效率
        # encoder_outputs: 编码器的输出，用于解码过程
        # **kwargs: 其他可选参数

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 根据标签准备解码器输入的token IDs
        # labels: 模型的目标标签
        return self._shift_right(labels)
        # 调用内部函数_shift_right，将标签向右移动，用作解码器的输入
    # 重新排序缓存中的过去键值，根据给定的 beam_idx
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果过去的键值未包含在输出中
        if past_key_values is None:
            # 提示用户设置 `use_cache=True` 来加快解码速度
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            # 返回原始的过去键值
            return past_key_values

        # 初始化重新排序后的解码器过去状态的元组
        reordered_decoder_past = ()
        
        # 遍历每一层的过去状态
        for layer_past_states in past_key_values:
            # 初始化当前层重新排序后的过去状态的元组
            reordered_layer_past_states = ()
            
            # 遍历当前层的每一个过去状态
            for layer_past_state in layer_past_states:
                # 根据给定的 beam_idx 选择正确的批次索引，以匹配过去状态的设备
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            # 检查重新排序后的第一个过去状态的形状与原始的第一个过去状态的形状是否匹配
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            
            # 检查重新排序后的过去状态的长度是否与原始过去状态的长度匹配
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            # 将当前层重新排序后的过去状态添加到解码器过去状态元组中
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)

        # 返回重新排序后的解码器过去状态
        return reordered_decoder_past
```