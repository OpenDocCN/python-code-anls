# `.\models\gpt2\modeling_gpt2.py`

```py
# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""PyTorch OpenAI GPT-2 model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gpt2 import GPT2Config

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai-community/gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "distilbert/distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    # Collect names and arrays of weights and biases
    names = []
    arrays = []
    for name, shape in init_vars:
        # 记录日志，显示正在加载的 TensorFlow 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用 TensorFlow 提供的方法加载变量的值
        array = tf.train.load_variable(tf_path, name)
        # 将加载的变量名添加到列表中
        names.append(name)
        # 将加载的变量值添加到列表中，并将其压缩（去除多余的维度）
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        # 跳过变量名中的 "model/" 部分
        name = name[6:]
        # 根据 "/" 分割变量名
        name = name.split("/")
        # 初始化指针为模型对象
        pointer = model
        # 遍历分割后的变量名
        for m_name in name:
            # 如果变量名匹配字母+数字的模式
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                # 使用数字分割变量名
                scope_names = re.split(r"(\d+)", m_name)
            else:
                # 否则，直接使用变量名
                scope_names = [m_name]
            # 根据变量名的首字符选择操作的属性
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            # 如果变量名有多个部分，则根据数字选择指定的属性
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            # 检查指针的形状与加载的数组的形状是否匹配
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            # 如果形状不匹配，将错误信息添加到异常中并抛出异常
            e.args += (pointer.shape, array.shape)
            raise
        # 记录日志，显示正在初始化的 PyTorch 权重的名称
        logger.info(f"Initialize PyTorch weight {name}")
        # 将加载的数组转换为 PyTorch 张量，并赋值给指针的数据
        pointer.data = torch.from_numpy(array)
    # 返回更新后的模型对象
    return model
# 定义一个名为 GPT2Attention 的类，继承自 nn.Module
class GPT2Attention(nn.Module):
    # 初始化方法，接受 config、is_cross_attention 和 layer_idx 三个参数
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        # 调用父类的初始化方法
        super().__init__()

        # 获取最大位置嵌入数，并将其设为缓冲区 "bias"
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            # 创建一个下三角形矩阵，并转换为布尔型张量
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,  # 非持久化缓冲区
        )
        # 设置 "masked_bias" 缓冲区，用于掩码操作
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        # 获取隐藏大小
        self.embed_dim = config.hidden_size
        # 获取注意力头数和头维度
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        # 如果 embed_dim 不能被 num_heads 整除，抛出异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 是否缩放注意力权重
        self.scale_attn_weights = config.scale_attn_weights
        # 是否为交叉注意力
        self.is_cross_attention = is_cross_attention

        # 层级注意力权重缩放、重新排序和向上转型
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # 如果是交叉注意力，则创建 c_attn 和 q_attn 两个卷积层
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            # 否则创建 c_attn 卷积层
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        # 创建 c_proj 卷积层
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        # 创建注意力的 dropout 层和残差的 dropout 层
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # 初始化剪枝的注意力头集合
        self.pruned_heads = set()

    # 剪枝注意力头的方法
    def prune_heads(self, heads):
        # 如果没有要剪枝的头部，直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可剪枝的头部及其索引
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # 剪枝 conv1d 层
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # 更新超参数
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 计算注意力权重，query 和 key 的点积
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            # 如果需要缩放注意力权重，按照数值开方缩放
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # 根据层索引按照逆序缩放注意力权重
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # 如果不是跨注意力，实施因果遮罩
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # 创建遮罩值的张量，确保与注意力权重张量类型和设备一致
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            # 根据因果遮罩条件调整注意力权重
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # 应用外部注意力遮罩
            attn_weights = attn_weights + attention_mask

        # 使用 softmax 函数归一化注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 将注意力权重转换回 value 张量的数据类型（如果需要）
        attn_weights = attn_weights.type(value.dtype)
        # 对注意力权重应用 dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对注意力头部进行遮罩处理
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    # 对给定的 query, key, value 张量进行上转型和重新排序的注意力计算
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 使用 `torch.baddbmm` 进行计算（在 Megatron-LM 中效率更高，带有 alpha 参数用于缩放）
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # 预先分配用于 `baddbmm` 的 attn_weights 张量
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # 计算缩放因子
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # 上转型（关闭自动转型）和重新排序（将 K 转置并展平）
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # 如果不是跨注意力层，则实现因果遮罩
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # 需要将 mask_value 转换为张量，以匹配 attn_weights 的数据类型和设备
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # 应用注意力遮罩
            attn_weights = attn_weights + attention_mask

        # 对 attn_weights 进行 softmax 归一化
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 如果 attn_weights 的数据类型不是 torch.float32，则抛出运行时错误
        # 将 attn_weights 转换回 value 的数据类型（如果在混合精度中）
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)

        # 对 attn_weights 应用注意力丢弃
        attn_weights = self.attn_dropout(attn_weights)

        # 如果存在 head_mask，则对注意力权重应用头部掩码
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算最终的注意力输出
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    # 将张量按照指定的头数和注意力头大小进行分割
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # 调整张量维度顺序，将头数维度和注意力头尺寸维度合并到隐藏大小维度中
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                # 如果作为跨注意力使用，必须定义权重 `q_attn`。
                # 实例化类时，请确保使用 `GPT2Attention(..., is_cross_attention=True)`。
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            # 使用自注意力机制生成查询
            query = self.q_attn(hidden_states)
            # 使用编码器的自注意力机制生成键和值，并按照 split_size 在维度2上拆分
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # 使用自注意力机制生成查询、键和值，并按照 split_size 在维度2上拆分
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # 将查询、键和值分割成多头注意力
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            # 如果存在过去的键和值，则将当前的键和值与过去的连接
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            # 如果需要缓存，将当前的键和值保存在 present 变量中
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            # 如果需要重新排序和升级注意力，则调用对应方法
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            # 否则直接调用注意力计算方法
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并多头注意力的输出
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # 使用投影层映射到特征空间
        attn_output = self.c_proj(attn_output)
        # 应用残差连接的 dropout
        attn_output = self.resid_dropout(attn_output)

        # 准备输出，包括注意力输出和可能的缓存
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出中

        return outputs  # 返回注意力输出和可能的缓存信息
# 定义一个 GPT2MLP 类，继承自 nn.Module 类
class GPT2MLP(nn.Module):
    # 初始化方法，接受中间层大小和配置对象作为参数
    def __init__(self, intermediate_size, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置对象中获取隐藏层大小作为嵌入维度
        embed_dim = config.hidden_size
        # 创建一个一维卷积层，输出大小为 intermediate_size，输入大小为 embed_dim
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        # 创建一个一维卷积层，输出大小为 embed_dim，输入大小为 intermediate_size
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        # 获取激活函数名称对应的激活函数，并赋值给 self.act
        self.act = ACT2FN[config.activation_function]
        # 创建一个以 config.resid_pdrop 为概率的 Dropout 层
        self.dropout = nn.Dropout(config.resid_pdrop)

    # 前向传播方法，接受隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 使用 self.c_fc 对隐藏状态进行一维卷积
        hidden_states = self.c_fc(hidden_states)
        # 使用 self.act 对卷积结果进行激活函数处理
        hidden_states = self.act(hidden_states)
        # 使用 self.c_proj 对激活后的结果进行一维卷积
        hidden_states = self.c_proj(hidden_states)
        # 使用 self.dropout 对卷积结果进行 Dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个 GPT2Block 类，继承自 nn.Module 类
class GPT2Block(nn.Module):
    # 初始化方法，接受配置对象和可选的层索引作为参数
    def __init__(self, config, layer_idx=None):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置对象中获取隐藏层大小
        hidden_size = config.hidden_size
        # 如果配置对象中指定了内部维度，则使用该值，否则使用默认值 4 * hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        # 创建一个 LayerNorm 层，对隐藏状态进行归一化，epsilon 参数由配置对象提供
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 创建一个 GPT2Attention 层，用于注意力机制处理
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化，epsilon 参数由配置对象提供
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 如果配置中指定需要添加交叉注意力机制，则创建相应的交叉注意力层和归一化层
        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 创建一个 GPT2MLP 类型的多层感知机层
        self.mlp = GPT2MLP(inner_dim, config)

    # 前向传播方法，接受多个可选的参数，包括隐藏状态、过去的层状态、注意力掩码等，返回处理后的隐藏状态
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # 保留残差连接的输入隐藏状态
        residual = hidden_states
        # Layer normalization 层，用于规范化输入隐藏状态
        hidden_states = self.ln_1(hidden_states)
        # 使用注意力机制进行计算
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 提取注意力输出中的主要输出
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        # 保留其它输出（如果有的话）
        outputs = attn_outputs[1:]
        # 残差连接，将注意力输出与输入相加
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # 如果存在编码器的隐藏状态，则添加一个自注意力块用于交叉注意力
            if not hasattr(self, "crossattention"):
                # 如果没有定义交叉注意力层，抛出错误
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 保留残差连接的输入隐藏状态
            residual = hidden_states
            # Layer normalization 层，用于规范化输入隐藏状态
            hidden_states = self.ln_cross_attn(hidden_states)
            # 使用交叉注意力机制进行计算
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # 提取交叉注意力输出中的主要输出
            attn_output = cross_attn_outputs[0]
            # 残差连接，将交叉注意力输出与之前的隐藏状态相加
            hidden_states = residual + attn_output
            # 将交叉注意力的其它输出（如果有的话）添加到总输出中
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        # 保留残差连接的输入隐藏状态
        residual = hidden_states
        # Layer normalization 层，用于规范化输入隐藏状态
        hidden_states = self.ln_2(hidden_states)
        # 使用前馈神经网络（MLP）进行计算
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 残差连接，将前馈网络输出与输入隐藏状态相加
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            # 如果使用缓存，则将当前隐藏状态和其它输出作为结果返回
            outputs = (hidden_states,) + outputs
        else:
            # 如果不使用缓存，则将当前隐藏状态和除了第一个元素以外的其它输出返回
            outputs = (hidden_states,) + outputs[1:]

        # 返回计算结果，包括隐藏状态、注意力（如果有的话）、交叉注意力（如果有的话）
        return outputs  # hidden_states, present, (attentions, cross_attentions)
class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 GPT2Config 类作为默认配置类
    config_class = GPT2Config
    # 使用 load_tf_weights_in_gpt2 函数来加载 TensorFlow 权重
    load_tf_weights = load_tf_weights_in_gpt2
    # 在模型中 Transformer 部分的名称前缀
    base_model_prefix = "transformer"
    # 模型是否支持并行化计算
    is_parallelizable = True
    # 模型是否支持梯度检查点（gradient checkpointing）
    supports_gradient_checkpointing = True
    # 不需要进行参数分割的模块名称列表
    _no_split_modules = ["GPT2Block"]
    # 在设备上跳过指定的键（key）设备放置（device placement）
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # 使用正态分布初始化权重，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果指定了填充索引，则将填充索引位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零，权重初始化为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 根据 OpenAI GPT-2 论文中的方案重新初始化选定的权重：
        #   > 使用一个修改后的初始化方法，考虑模型深度的累积在残差路径上的影响。在初始化时，将残差层的权重按照
        #   > 1/√N 的因子进行缩放，其中 N 是残差层的数量。
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # 参考（Megatron-LM）：https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # 特殊的缩放初始化 --> 每个 Transformer 块有两个 Layer Norms
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    """
    # 定义可选的语言建模损失，类型为 torch.FloatTensor，形状为 (1,)
    loss: Optional[torch.FloatTensor] = None
    # 定义可选的多项选择分类损失，类型为 torch.FloatTensor，形状为 (1,)
    mc_loss: Optional[torch.FloatTensor] = None
    # 定义预测的语言建模头部得分，类型为 torch.FloatTensor，形状为 (batch_size, num_choices, sequence_length, config.vocab_size)
    # 表示每个词汇标记的预测分数，SoftMax 之前的值
    logits: torch.FloatTensor = None
    # 定义预测的多项选择分类头部得分，类型为 torch.FloatTensor，形状为 (batch_size, num_choices)
    # 表示每个选择的预测分数，SoftMax 之前的值
    mc_logits: torch.FloatTensor = None
    # 定义预先计算的键值对 (past_key_values)，类型为 Tuple[Tuple[torch.FloatTensor]]
    # 长度为 config.n_layers，包含每个层的键和值张量，形状为 (batch_size, num_heads, sequence_length, embed_size_per_head)
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 定义模型的隐藏状态 (hidden_states)，类型为 tuple(torch.FloatTensor)
    # 包含每个层输出的隐藏状态，形状为 (batch_size, sequence_length, hidden_size)，包括初始嵌入的输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义注意力权重 (attentions)，类型为 tuple(torch.FloatTensor)
    # 包含每个层的自注意力权重，形状为 (batch_size, num_heads, sequence_length, sequence_length)
    # 用于计算自注意力头部的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# GPT2_START_DOCSTRING 是一个包含模型文档字符串的原始字符串常量，描述了 GPT-2 模型的继承关系和使用说明。
GPT2_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPT2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# GPT2_INPUTS_DOCSTRING 是一个空的原始字符串常量，可能用于描述模型的输入参数文档字符串，但在当前代码中未进行定义。
GPT2_INPUTS_DOCSTRING = r"""
"""

# PARALLELIZE_DOCSTRING 是一个包含并行化模型方法文档字符串的原始字符串常量，描述了其实验性质以及如何使用。
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - openai-community/gpt2: 12
                - openai-community/gpt2-medium: 24
                - openai-community/gpt2-large: 36
                - openai-community/gpt2-xl: 48

    Example:

    ```
    # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""

# DEPARALLELIZE_DOCSTRING 是一个包含取消并行化模型方法文档字符串的原始字符串常量，描述了如何将模型从并行状态移到 CPU 上。
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```
    # On a 4 GPU machine with openai-community/gpt2-large:
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    # 将模型从并行模式转换为单机模式，将模型放回CPU并通过调用torch.cuda.empty_cache()清理内存
    model.deparallelize()
"""
@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size  # 从配置中获取隐藏层大小作为嵌入维度

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)  # 创建词嵌入层
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)  # 创建位置嵌入层

        self.drop = nn.Dropout(config.embd_pdrop)  # 创建dropout层
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])  # 创建GPT2Block的模块列表，用于堆叠层

        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)  # 创建LayerNorm层用于最终处理

        # Model parallel
        self.model_parallel = False  # 模型并行标志位初始化为False
        self.device_map = None  # 设备映射初始化为None
        self.gradient_checkpointing = False  # 梯度检查点初始化为False

        # Initialize weights and apply final processing
        self.post_init()  # 调用后处理函数完成权重初始化和最终处理

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 检查device_map的有效性
        warnings.warn(
            "`GPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        # 如果device_map为None，则使用默认的均衡设备映射
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        # 断言设备映射的有效性
        assert_device_map(self.device_map, len(self.h))
        # 将模型设为模型并行模式
        self.model_parallel = True
        # 确定第一个和最后一个设备
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # 将词嵌入层和位置嵌入层加载到第一个设备上
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # 加载各个块到对应的设备
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # 将ln_f加载到最后一个设备上
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，提示函数即将被移除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 将模型并行模式设置为False
        self.model_parallel = False
        # 将设备映射设置为None
        self.device_map = None
        # 将第一个和最后一个设备设置为cpu
        self.first_device = "cpu"
        self.last_device = "cpu"
        # 将词嵌入层和位置嵌入层加载到cpu
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        # 将所有块加载到cpu
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        # 将ln_f加载到cpu，并清空cuda缓存
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()  # 清空CUDA缓存

    def get_input_embeddings(self):
        return self.wte  # 返回词嵌入层
    # 设置新的输入嵌入（词向量）到模型中
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 剪枝模型中的注意力头部
    # heads_to_prune: 需要剪枝的头部字典 {层号: 需要在该层剪枝的头部列表}
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            # 在指定层的注意力模块中剪枝特定的注意力头部
            self.h[layer].attn.prune_heads(heads)

    # 前向传播函数，处理模型的输入和参数，并返回输出
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器添加文档字符串，描述该类是基于GPT2模型的语言建模头部模型
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    # 定义权重共享的键名列表
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化函数，接收一个配置参数config对象
    def __init__(self, config):
        # 调用父类构造函数，初始化模型
        super().__init__(config)
        # 创建GPT2模型的实例并赋值给self.transformer
        self.transformer = GPT2Model(config)
        # 创建一个线性层作为语言建模的头部，输入维度为config.n_embd，输出维度为config.vocab_size，无偏置
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 模型并行处理的标志和设备映射初始化为None
        self.model_parallel = False
        self.device_map = None

        # 调用后处理函数，初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器添加文档字符串，描述该方法用于模型并行化
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，表明该方法将在未来版本中删除，建议使用from_pretrained函数加载模型
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        # 根据设备映射或默认情况下创建设备映射
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 断言设备映射的正确性
        assert_device_map(self.device_map, len(self.transformer.h))
        # 在模型上应用并行化，使用设备映射
        self.transformer.parallelize(self.device_map)
        # 将lm_head移动到第一个设备
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        # 设置模型并行标志为True
        self.model_parallel = True

    # 使用装饰器添加文档字符串，描述该方法用于取消模型的并行化
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，表明该方法将在未来版本中删除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 取消模型的并行化
        self.transformer.deparallelize()
        # 将模型和lm_head移动到CPU上
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        # 设置模型并行标志为False
        self.model_parallel = False
        # 清空CUDA缓存
        torch.cuda.empty_cache()

    # 返回lm_head作为输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 从 kwargs 中获取 token_type_ids，如果不存在则设为 None
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # 如果 past_key_values 存在，则根据其信息决定是否跳过部分输入 ID
        if past_key_values:
            # 获取 past_key_values 中的长度信息，通常是上一次生成的序列长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入 ID 的长度大于 past_length，则移除前缀长度为 past_length
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认行为是只保留最后一个输入 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 更新输入 ID，移除前缀部分
            input_ids = input_ids[:, remove_prefix_length:]
            
            # 如果 token_type_ids 存在，则同步更新其长度
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # 从 kwargs 中获取 attention_mask 和 position_ids
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果 attention_mask 存在而 position_ids 不存在，则创建新的 position_ids
        if attention_mask is not None and position_ids is None:
            # 在批量生成时动态创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            # 如果 past_key_values 存在，则根据输入 ID 的长度截取 position_ids
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # 如果传入 inputs_embeds，且 past_key_values 不存在，则只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典，包括 past_key_values、use_cache、position_ids、attention_mask、token_type_ids
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,


        # 此方法定义了模型的前向传播逻辑，接收多个可选的输入参数
        self,
        # input_ids：输入的token IDs，可以是LongTensor类型，可选
        input_ids: Optional[torch.LongTensor] = None,
        # past_key_values：用于保存过去的键值状态，可选，类型为Tuple[Tuple[torch.Tensor]]
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        # attention_mask：注意力遮罩，指定哪些token需要被attention，可选，类型为FloatTensor
        attention_mask: Optional[torch.FloatTensor] = None,
        # token_type_ids：用于区分不同句子的token类型ID，可选，类型为LongTensor
        token_type_ids: Optional[torch.LongTensor] = None,
        # position_ids：位置ID，标识token在输入序列中的位置，可选，类型为LongTensor
        position_ids: Optional[torch.LongTensor] = None,
        # head_mask：用于指定哪些注意力头是激活的，可选，类型为FloatTensor
        head_mask: Optional[torch.FloatTensor] = None,
        # inputs_embeds：用于直接传入嵌入向量而不是token IDs，可选，类型为FloatTensor
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # encoder_hidden_states：编码器的隐藏状态，可选，类型为Tensor
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask：编码器的注意力遮罩，可选，类型为FloatTensor
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # labels：用于训练时指定的标签，可选，类型为LongTensor
        labels: Optional[torch.LongTensor] = None,
        # use_cache：是否使用缓存，可选，类型为bool
        use_cache: Optional[bool] = None,
        # output_attentions：是否输出注意力权重，可选，类型为bool
        output_attentions: Optional[bool] = None,
        # output_hidden_states：是否输出隐藏状态，可选，类型为bool
        output_hidden_states: Optional[bool] = None,
        # return_dict：是否返回结果字典形式的输出，可选，类型为bool
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 如果未指定return_dict，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer处理输入数据，获取transformer的输出
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从transformer输出中获取隐藏状态
        hidden_states = transformer_outputs[0]

        # 如果启用模型并行化，则设置隐藏状态的设备
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # 生成语言模型的预测输出
        lm_logits = self.lm_head(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行化
            labels = labels.to(lm_logits.device)
            # 将logits向左偏移一个位置，以便预测下一个标记
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平标记
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 如果不使用return_dict选项，则返回输出元组
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用return_dict选项返回带有交叉注意力的CausalLMOutputWithCrossAttentions对象
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 返回类型声明为一个元组，元组中包含多个元组，每个元组中包含 torch.Tensor 对象
        return tuple(
            # 外层元组的每个元素是一个通过索引操作重新排序后的 past_state 的元组
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            # 对于 past_key_values 中的每个 layer_past 元素，重新排序其中的 past_state 对象
            for layer_past in past_key_values
        )
# 使用装饰器为类添加起始文档字符串，描述其作为 GPT2 模型变换器的功能，包括语言建模和多选分类头部
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    # 定义需要权重共享的键列表
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接收配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将标签数目设置为 1
        config.num_labels = 1
        # 创建 GPT2Model 的实例并赋值给 self.transformer
        self.transformer = GPT2Model(config)
        # 创建 nn.Linear 实例作为语言模型头部 self.lm_head，连接数为 config.n_embd 到 config.vocab_size，无偏置
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 创建 SequenceSummary 的实例作为多选头 self.multiple_choice_head
        self.multiple_choice_head = SequenceSummary(config)

        # Model parallel
        # 初始化 model parallel 和 device_map 参数
        self.model_parallel = False
        self.device_map = None

        # 调用自定义的初始化方法，应用最终处理
        self.post_init()

    # 使用装饰器为方法添加并行化的起始文档字符串
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，提示该方法即将在 Transformers v5 中移除，并提供替代方法
        warnings.warn(
            "`GPT2DoubleHeadsModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should"
            " load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your"
            " own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'transformer.h.0': 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        # 根据设备映射创建 device_map
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 验证设备映射的正确性
        assert_device_map(self.device_map, len(self.transformer.h))
        # 在 transformer 上应用并行化，根据 device_map
        self.transformer.parallelize(self.device_map)
        # 将 lm_head 和 multiple_choice_head 移至 transformer 的第一个设备
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.multiple_choice_head = self.multiple_choice_head.to(self.transformer.first_device)
        # 设置 model_parallel 为 True
        self.model_parallel = True

    # 使用装饰器为方法添加取消并行化的起始文档字符串
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，提示该方法即将在 Transformers v5 中移除，并提供替代方法
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 取消 transformer 的并行化
        self.transformer.deparallelize()
        # 将 transformer、lm_head、multiple_choice_head 移至 CPU
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.multiple_choice_head = self.multiple_choice_head.to("cpu")
        # 设置 model_parallel 为 False，并清空 CUDA 缓存
        self.model_parallel = False
        torch.cuda.empty_cache()

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # 获取额外参数中的 token_type_ids
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # 如果存在过去的键值，则需要根据 past_key_values 进行处理
        if past_key_values:
            # 获取过去的长度，通常是通过 past_key_values 的第一个元素获取
            past_length = past_key_values[0][0].shape[2]

            # 如果输入的 input_ids 的长度大于过去的长度，截取掉前面的部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个输入的 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            # 如果存在 token_type_ids，则也需要相应地调整它的长度
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # 获取 attention_mask 和 position_ids
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果 attention_mask 存在而 position_ids 不存在，则需要动态生成 position_ids
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在 past_key_values，则只保留与 input_ids 相关的部分 position_ids
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # 返回准备好的输入参数字典
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # 前向传播函数，接收多种输入参数并输出模型的返回结果
        pass

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 返回一个元组，其中每个元素都是一个元组，每个元组包含经过重新排序后的每个层的过去键-值缓存
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            # 对每个层的过去键-值缓存执行重新排序
            for layer_past in past_key_values
        )
"""
GPT2 模型转换器，顶部带有序列分类头（线性层）。

[`GPT2ForSequenceClassification`] 使用最后一个令牌进行分类，与其他因果模型（例如 GPT-1）类似。

由于它在最后一个令牌上执行分类，因此需要知道最后一个令牌的位置。如果在配置中定义了 `pad_token_id`，它会在每行中找到不是填充令牌的最后一个令牌。如果没有定义 `pad_token_id`，它会简单地取批次中每行的最后一个值。当传递 `inputs_embeds` 而不是 `input_ids` 时，由于无法猜测填充令牌，它会执行相同的操作（取批次中每行的最后一个值）。
"""
@add_start_docstrings(
    """
    GPT2 模型，顶部带有标记分类头（即隐藏状态输出的线性层），例如用于命名实体识别（NER）任务。
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForTokenClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 设置模型的标签数量
        self.num_labels = config.num_labels

        # 使用给定的配置创建 GPT2Model 实例作为模型的转换器
        self.transformer = GPT2Model(config)
        
        # 确定分类器的 dropout 率
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        
        # 根据确定的 dropout 率创建 Dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 创建线性层作为分类器，连接 GPT2 模型的隐藏层到分类标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化模型并设置模型并行计算相关的变量
        self.model_parallel = False
        self.device_map = None

        # 调用额外的初始化函数
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # 设置代码示例文档字符串和格式化
    # fmt: off
    @add_code_sample_docstrings(
        checkpoint="brad1141/gpt2-finetuned-comp2",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=0.25,
        expected_output=[
            "Lead",
            "Lead",
            "Lead",
            "Position",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
        ],
    )
    # fmt: on
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 已定义，则保持其值不变；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Transformer 模型进行处理
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 Transformer 模型的输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        # 对隐藏状态应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 使用分类器获取 logits
        logits = self.classifier(hidden_states)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 将标签移动到 logits 的设备上
            labels = labels.to(logits.device)
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典，则返回 logits 和其他 Transformer 模型输出的组合
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 添加文档字符串以描述 GPT-2 模型在抽取式问答任务（如 SQuAD）上的应用，包括在隐藏状态输出之上的线性层用于计算“span start logits”和“span end logits”。
@add_start_docstrings(
    """
    The GPT-2 Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 设定标签数目
        self.num_labels = config.num_labels
        # 使用给定的配置初始化 GPT-2 模型的主体部分
        self.transformer = GPT2Model(config)
        # 定义一个线性层，用于输出问题回答的起始位置和结束位置的logits
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Model parallel
        # 模型并行设为False
        self.model_parallel = False
        # 设备映射为空
        self.device_map = None

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加文档字符串以描述模型的前向传播过程，包括输入的详细说明
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串，指定检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_CHECKPOINT_FOR_DOC,
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
        # Determine whether to use the provided return_dict or the default from configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through the transformer model with specified arguments
        outputs = self.transformer(
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

        # Extract sequence output from the model outputs
        sequence_output = outputs[0]

        # Compute logits for question answering from sequence output
        logits = self.qa_outputs(sequence_output)
        
        # Split logits into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # Squeeze and make contiguous the logits tensors
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Ensure start_positions and end_positions have correct dimensions and device
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            
            # Clamp positions to ignore indices outside the model input
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Define loss function and compute start and end losses
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # Return outputs without dictionary format if return_dict is False
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # Return outputs in QuestionAnsweringModelOutput format
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```