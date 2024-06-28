# `.\models\roformer\modeling_roformer.py`

```
# coding=utf-8
# 设定文件编码为 UTF-8

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# 版权声明：2021 年由 HuggingFace Inc. 团队保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证授权（“许可证”）;

# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本：

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则根据许可证分发的软件都是基于“原样”分发的，无论是明示还是隐含的任何形式的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证以了解特定语言下的权限和限制

""" PyTorch RoFormer model."""
# PyTorch RoFormer 模型

import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_roformer import RoFormerConfig

# 获取 logger 实例，用于记录日志信息
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "junnyu/roformer_chinese_base"
_CONFIG_FOR_DOC = "RoFormerConfig"

ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "junnyu/roformer_chinese_small",
    "junnyu/roformer_chinese_base",
    "junnyu/roformer_chinese_char_small",
    "junnyu/roformer_chinese_char_base",
    "junnyu/roformer_small_discriminator",
    "junnyu/roformer_small_generator",
    # See all RoFormer models at https://huggingface.co/models?filter=roformer
]

# 从 transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding 复制到 RoFormerSinusoidalPositionalEmbedding
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    # 该模块生成任意长度的正弦位置嵌入

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        # 调用父类 nn.Embedding 的初始化方法，设定位置数量和嵌入维度

        self.weight = self._init_weight(self.weight)
        # 初始化权重矩阵，调用 _init_weight 方法

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        初始化权重矩阵，类似于 XLM 的 create_sinusoidal_embeddings 函数，但特征未交错。
        余弦特征位于向量的后半部分。[dim // 2:]
        """
        n_pos, dim = out.shape
        # 创建位置编码矩阵，使用正弦和余弦函数
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # 提前设置为False，以避免在 pytorch-1.8+ 中的一个错误
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 将正弦编码部分赋值给 out 的前半部分
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        # 将余弦编码部分赋值给 out 的后半部分
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()  # 分离出该张量
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` 期望为 [bsz x seqlen]。"""
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置索引张量，从 past_key_values_length 到 past_key_values_length + seq_len
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)
def load_tf_weights_in_roformer(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块用于字符串匹配

        import numpy as np  # 导入 NumPy 库用于数值计算
        import tensorflow as tf  # 导入 TensorFlow 库用于加载 TensorFlow 模型权重
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取 TensorFlow checkpoint 文件的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志，显示正在转换的 TensorFlow checkpoint 路径

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)  # 获取 TensorFlow checkpoint 中的变量列表
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志，显示正在加载的 TensorFlow 权重和形状
        array = tf.train.load_variable(tf_path, name)  # 加载 TensorFlow checkpoint 中的变量数据
        names.append(name.replace("bert", "roformer"))  # 将变量名中的 "bert" 替换为 "roformer"
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        
        # adam_v 和 adam_m 是 AdamWeightDecayOptimizer 中用于计算 m 和 v 的变量，对于使用预训练模型不需要
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")  # 记录日志，跳过不需要加载的变量名
            continue
        
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]

            # 根据变量名的开头选择 PyTorch 模型中的指针位置
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")  # 记录日志，跳过找不到的变量名
                    continue

            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)  # 对于 "kernel" 类型的变量，转置数组

        try:
            if not pointer.shape == array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise

        logger.info(f"Initialize PyTorch weight {name}")  # 记录日志，显示正在初始化的 PyTorch 权重名
        pointer.data = torch.from_numpy(array)  # 将 NumPy 数组转换为 PyTorch 张量赋值给指针

    return model  # 返回加载完成的 PyTorch 模型
# 定义 RoFormerSelfAttention 类，用于实现自注意力机制
class RoFormerSelfAttention(nn.Module):
    # 初始化函数，设置模型参数和层
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否可以被注意力头数整除，或者是否具有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不能整除，抛出数值错误
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

        # 设置是否为解码器，以及旋转值的配置
        self.is_decoder = config.is_decoder
        self.rotary_value = config.rotary_value

    # 转置操作，将输入张量重塑为多头注意力的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，实现自注意力机制的计算
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 返回隐藏状态的形状，用于后续的嵌入操作
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 如果没有给定嵌入向量，则使用词嵌入层获取
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 如果没有给定 token_type_ids，则创建全零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        # 根据 token_type_ids 获取 token 类型的嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算最终的嵌入向量，将词嵌入和 token 类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings

        # 对嵌入向量进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入向量进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回最终的嵌入向量
        return embeddings
    # 应用旋转位置嵌入到查询、键、值张量中
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # 分割正弦和余弦位置编码
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # 创建重复的正弦位置编码张量，用于与查询、键、值张量相乘
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # 创建重复的余弦位置编码张量，用于与查询、键、值张量相乘
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # 旋转查询张量的一半元素并与余弦、正弦位置编码相乘
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # 旋转键张量的一半元素并与余弦、正弦位置编码相乘
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # 如果存在值张量，旋转其一半元素并与余弦、正弦位置编码相乘
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        # 如果不存在值张量，返回旋转后的查询和键张量
        return query_layer, key_layer
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->RoFormer
class RoFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义全连接层，将输入维度转换为相同的隐藏层维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，用于归一化隐藏层状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层
        hidden_states = self.dense(hidden_states)
        # Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 操作，残差连接并归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RoFormerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # RoFormerSelfAttention 实例
        self.self = RoFormerSelfAttention(config)
        # RoFormerSelfOutput 实例
        self.output = RoFormerSelfOutput(config)
        # 初始化一个集合，用于记录被修剪的注意力头
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        # 如果没有需要修剪的头，则直接返回
        if len(heads) == 0:
            return
        # 调用工具函数找到可以修剪的注意力头和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录修剪的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # End Copy
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # RoFormerSelfAttention 的前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # RoFormerSelfOutput 的前向传播，传入注意力输出和隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，将它们添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出注意力权重，则添加它们
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->RoFormer
class RoFormerIntermediate(nn.Module):
    # 初始化方法，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化方法，确保正确初始化
        super().__init__()
        # 创建一个全连接层，输入大小为config中的隐藏层大小，输出大小为config中的中间层大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # 检查config中的隐藏层激活函数，如果是字符串，则使用预定义的激活函数映射表ACT2FN获取对应的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则，直接使用config中提供的激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，定义了数据在模型中的传递过程
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理输入的隐藏状态数据，得到处理后的结果
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理全连接层输出的数据，得到最终的隐藏状态表示
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态数据
        return hidden_states
# 定义 RoFormer 模型的输出层，继承自 nn.Module 类
class RoFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将中间大小的特征转换为隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm 层，用于对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于在训练过程中随机断开神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过线性层进行转换
        hidden_states = self.dense(hidden_states)
        # 对转换后的隐藏状态应用 Dropout
        hidden_states = self.dropout(hidden_states)
        # 对 Dropout 后的隐藏状态进行 LayerNorm 处理，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义 RoFormer 模型的一个层，继承自 nn.Module 类
class RoFormerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置 feed forward 阶段的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度设置为 1
        self.seq_len_dim = 1
        # RoFormer 层的注意力机制，使用 RoFormerAttention 类定义
        self.attention = RoFormerAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        # 如果添加跨注意力机制，需要检查是否作为解码器使用，否则抛出错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 添加跨注意力机制的注意力层
            self.crossattention = RoFormerAttention(config)
        # RoFormer 层的中间层，使用 RoFormerIntermediate 类定义
        self.intermediate = RoFormerIntermediate(config)
        # RoFormer 层的输出层，使用 RoFormerOutput 类定义
        self.output = RoFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        # 如果有缓存的键/值对，则取其中的前两个作为自注意力的过去键/值对
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        # 使用自注意力层处理隐藏状态，生成自注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]  # 获取自注意力输出
        
        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 提取除最后一个之外的所有输出
            present_key_value = self_attention_outputs[-1]  # 最后一个是当前键/值对
        else:
            outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则添加自注意力输出
        
        cross_attn_present_key_value = None
        
        # 如果是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                    "layers by setting `config.add_cross_attention=True`"
                )
            
            # 如果有缓存的键/值对，则取其中的后两个作为跨注意力的过去键/值对
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            
            # 使用跨注意力层处理自注意力输出和编码器隐藏状态，生成跨注意力输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                sinusoidal_pos,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]  # 获取跨注意力输出
            outputs = outputs + cross_attention_outputs[1:-1]  # 添加跨注意力的输出
            
            # 将跨注意力缓存添加到当前键/值对的第三和第四位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        
        # 将注意力输出应用到前馈网络块中，生成层输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs  # 将层输出和之前的输出合并
        
        # 如果是解码器，将注意力键/值作为最后的输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        
        return outputs  # 返回所有输出

    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出，生成中间层输出
        intermediate_output = self.intermediate(attention_output)
        
        # 使用输出层处理中间层输出和注意力输出，生成层输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output  # 返回层输出
class RoFormerEncoder(nn.Module):
    # RoFormer 编码器模型
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化 RoFormer 的位置编码器
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads
        )
        # 初始化 RoFormer 的每一层
        self.layer = nn.ModuleList([RoFormerLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # RoFormer 编码器的前向传播函数
        # （具体实现根据参数进行变化，返回结果为一个字典或者多个张量）
        pass  # 实际前向传播代码未提供，暂时使用 pass 占位


class RoFormerPredictionHeadTransform(nn.Module):
    # RoFormer 预测头部变换模块
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # RoFormer 预测头部变换模块的前向传播
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RoFormerLMPredictionHead(nn.Module):
    # RoFormer 语言模型预测头部模块
    def __init__(self, config):
        super().__init__()
        self.transform = RoFormerPredictionHeadTransform(config)
        # 输出权重与输入嵌入相同，但每个标记有一个输出偏置
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 需要一个链接以便在 `resize_token_embeddings` 时正确调整偏置
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # RoFormer 语言模型预测头部模块的前向传播
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOnlyMLMHead 复制并修改为 RoFormer
class RoFormerOnlyMLMHead(nn.Module):
    # 仅包含 RoFormer 语言模型头部的模块
    def __init__(self, config):
        super().__init__()
        self.predictions = RoFormerLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # RoFormer 仅语言模型头部的前向传播
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RoFormerPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化、下载和加载预训练模型的简单接口。
    """

    config_class = RoFormerConfig
    load_tf_weights = load_tf_weights_in_roformer
    base_model_prefix = "roformer"
    # 定义一个类变量，指示是否支持梯度检查点
    supports_gradient_checkpointing = True
    
    # 初始化模型权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重数据，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 RoFormerSinusoidalPositionalEmbedding 类的实例
        elif isinstance(module, RoFormerSinusoidalPositionalEmbedding):
            # 对于这种情况，不做任何初始化操作
            pass
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重数据，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了填充索引，则将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为1
            module.weight.data.fill_(1.0)
ROFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RoFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
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
    """
    Add model-specific documentation to the given function.

    Args:
        **kwargs: Keyword arguments forwarded to the function.

    This decorator helps in adding standardized documentation string (`ROFORMER_START_DOCSTRING`) to functions.
    """
)
    # 定义 RoFormer 模型的基础类，输出未经特定头部处理的原始隐藏状态
    "The bare RoFormer Model transformer outputting raw hidden-states without any specific head on top.",
    # 引入 RoFormer 的文档字符串开头
    ROFORMER_START_DOCSTRING,
# RoFormer 模型类，继承自 RoFormerPreTrainedModel 类
class RoFormerModel(RoFormerPreTrainedModel):
    """
    
    模型可以作为编码器（仅自注意力）或解码器运行，当作解码器时，在自注意力层之间添加了交叉注意力层，
    遵循 [Attention is all you need](https://arxiv.org/abs/1706.03762) 中描述的架构，作者为 Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin.
    
    若要作为解码器运行，模型需要用配置中的 `is_decoder` 参数初始化为 `True`。
    若要在 Seq2Seq 模型中使用，模型需要用 `is_decoder` 参数和 `add_cross_attention` 参数同时初始化为 `True`；
    此时预期在前向传播中输入 `encoder_hidden_states`。
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 使用配置初始化 RoFormerEmbeddings 对象
        self.embeddings = RoFormerEmbeddings(config)

        # 如果嵌入大小与隐藏大小不同，初始化线性层以映射嵌入大小到隐藏大小
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        # 初始化 RoFormerEncoder 层
        self.encoder = RoFormerEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层的 word_embeddings 属性
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置嵌入层的 word_embeddings 属性为给定值
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型中的注意力头。heads_to_prune: {层编号: 要在该层中剪枝的头列表} 参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 对每一层，调用 RoFormerEncoder 中相应的注意力层对象的 prune_heads 方法
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化类变量，包含需要共享权重的键列表
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传递配置对象
        super().__init__(config)

        # 如果配置指定为解码器，则发出警告，建议设定 `config.is_decoder=False`，以支持双向自注意力
        if config.is_decoder:
            logger.warning(
                "If you want to use `RoFormerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 RoFormerModel 的实例，使用给定的配置对象
        self.roformer = RoFormerModel(config)
        # 创建 RoFormerOnlyMLMHead 的实例，使用给定的配置对象
        self.cls = RoFormerOnlyMLMHead(config)

        # 调用后续初始化方法，用于初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入的方法，返回 MLM 头部的解码器
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的方法，接受一个新的嵌入参数，并将其分配给 MLM 头部的解码器
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 正向传播方法，接受多个输入参数并返回结果
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # 设置返回字典，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RoFormer 模型进行前向传播
        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 RoFormer 输出中获取序列输出
        sequence_output = outputs[0]
        
        # 通过分类层获取预测分数
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 定义交叉熵损失函数，用于计算 Masked Language Modeling 损失
            loss_fct = CrossEntropyLoss()  # -100 索引表示填充标记
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不需要返回字典，则按照元组方式返回结果
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果需要返回字典，则构建 MaskedLMOutput 对象并返回
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # 添加一个虚拟的 token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        # 扩展注意力掩码，在最后加入全零列
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建一个全为填充 token id 的张量，并拼接到输入 ids 后面
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回输入字典，用于生成过程
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 为 RoFormer 模型添加文档字符串，用于指明其作为语言建模模型进行 Causal LM 微调
@add_start_docstrings(
    """RoFormer Model with a `language modeling` head on top for CLM fine-tuning.""", ROFORMER_START_DOCSTRING
)
class RoFormerForCausalLM(RoFormerPreTrainedModel):
    # 定义绑定权重的关键键名列表，用于权重共享
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 如果配置中标志不是解码器，则发出警告信息
        if not config.is_decoder:
            logger.warning("If you want to use `RoFormerForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 RoFormer 模型和仅包含 MLM 头部的对象
        self.roformer = RoFormerModel(config)
        self.cls = RoFormerOnlyMLMHead(config)

        # 执行初始化权重和应用最终处理
        self.post_init()

    # 获取输出嵌入的方法，返回预测的解码器
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的方法，用新的嵌入替换预测的解码器
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 重写的前向传播方法，包含详细的输入和输出文档字符串
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 准备用于生成的输入，处理输入形状和 attention_mask
        input_shape = input_ids.shape

        # 如果 attention_mask 为空，则创建全为 1 的新 attention_mask
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果 past_key_values 不为空，则根据其值截取 input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：仅保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回处理后的输入字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新排序缓存数据，以适应新的束搜索索引顺序
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空的元组，用于存储重新排序后的过去状态
        reordered_past = ()
        # 遍历每一层的过去状态
        for layer_past in past_key_values:
            # 对每层的过去状态的前两个元素（通常是隐藏状态和注意力权重）按照束搜索索引进行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                # 将未重新排序的其余部分直接添加到元组中
                + layer_past[2:],
            )
        # 返回重新排序后的完整过去状态元组
        return reordered_past
class RoFormerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出大小都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义一个全连接层，将隐藏状态映射到类别数量（config.num_labels）
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        # 取序列中的第一个特征向量作为表示句子的向量（等同于 [CLS] 标记）
        x = features[:, 0, :]
        # 对输入进行 dropout 处理
        x = self.dropout(x)
        # 通过全连接层 dense 进行线性变换
        x = self.dense(x)
        # 使用配置中指定的激活函数 ACT2FN[self.config.hidden_act] 对 x 进行激活
        x = ACT2FN[self.config.hidden_act](x)
        # 再次对输出进行 dropout 处理
        x = self.dropout(x)
        # 通过全连接层 out_proj 进行线性变换，得到最终的分类输出
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    RoFormer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
# RoFormer 用于序列分类任务的模型，顶部有一个用于分类或回归的线性层（在池化输出之上）
class RoFormerForSequenceClassification(RoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # RoFormer 模型的主体部分
        self.roformer = RoFormerModel(config)
        # 分类器部分，用于序列分类任务
        self.classifier = RoFormerClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多种输入并返回模型输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 `return_dict` 不为 None，则使用传入的值；否则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 RoFormer 模型，获取输出
        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 RoFormer 模型输出的序列输出
        sequence_output = outputs[0]
        
        # 将序列输出传递给分类器，获取分类 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None

        # 如果存在标签，则计算损失
        if labels is not None:
            # 如果问题类型未定义，则根据标签类型和类别数量进行自动推断问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签回归任务，计算平均平方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归任务，计算平均平方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 `return_dict` 为 False，则返回 logits 和可能的额外输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 `return_dict` 为 True，则返回包含损失、logits、隐藏状态和注意力的 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 RoFormer 模型进行多选题分类任务，顶部有一个线性层（线性层放在汇总输出之上，并带有 softmax），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    RoFormer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
class RoFormerForMultipleChoice(RoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 RoFormer 模型
        self.roformer = RoFormerModel(config)
        # 序列汇总层
        self.sequence_summary = SequenceSummary(config)
        # 分类器，线性层，输入大小为隐藏状态的大小，输出为1维
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        ROFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接收多个输入和可选的标签，返回一个包含输出的命名元组或字典
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        ) -> Union[MultipleChoiceModelOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据是否返回字典类型决定返回值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选项数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入的 `input_ids` 重新形状为二维张量，每行包含一个批次的输入序列
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将注意力掩码 `attention_mask` 重新形状为二维张量
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将标记类型 `token_type_ids` 重新形状为二维张量
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        # 将嵌入向量 `inputs_embeds` 重新形状为三维张量，每行包含一个批次的嵌入向量序列
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用 RoFormer 模型处理输入数据，获取输出结果
        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行汇总
        pooled_output = self.sequence_summary(sequence_output)
        # 使用分类器获取 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重新形状为二维张量
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不要求返回字典类型，构建输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典类型，构建 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    RoFormer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROFORMER_START_DOCSTRING,
)



class RoFormerForTokenClassification(RoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 RoFormer 模型
        self.roformer = RoFormerModel(config)
        # Dropout 层用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器线性层，将隐藏状态映射到类别数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # RoFormer 模型的前向传播
        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 应用 dropout
        sequence_output = self.dropout(sequence_output)
        # 将序列输出映射到类别空间
        logits = self.classifier(sequence_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用 return_dict，则返回元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



@add_start_docstrings(
    """
    RoFormer Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,


    # RoFormer模型，顶部带有用于类似SQuAD的抽取式问答任务的跨度分类头部（在隐藏状态输出之上的线性层，用于计算`span start logits`和`span end logits`）。
    # 此处是一个文档字符串或注释，描述了RoFormer模型及其在问答任务中的应用。
    ROFORMER_START_DOCSTRING,
)
# 结束 RoFormerForQuestionAnswering 类的定义，此处的 ")" 是类定义的结束符号
class RoFormerForQuestionAnswering(RoFormerPreTrainedModel):
    # RoFormerForQuestionAnswering 类的初始化函数，继承自 RoFormerPreTrainedModel 类
    def __init__(self, config):
        # 调用父类 RoFormerPreTrainedModel 的初始化方法
        super().__init__(config)

        # 设置分类标签数量为 2
        config.num_labels = 2
        # 将分类标签数量保存在 self.num_labels 中
        self.num_labels = config.num_labels

        # 初始化 RoFormer 模型，使用给定的配置参数
        self.roformer = RoFormerModel(config)
        # 创建一个线性层，用于答案预测，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        # 调用自定义的 post_init 方法
        self.post_init()

    # 为 forward 方法添加模型输入的文档字符串，描述了输入参数的含义
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 为 forward 方法添加代码示例的文档字符串，展示了如何调用模型以及其返回的结果类型
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播方法，定义了模型的输入和输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        # 如果 return_dict 为 None，则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RoFormer 模型进行推理
        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 RoFormer 输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入问答输出层，得到开始和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 的维度大于 1，则进行压缩
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 对超出模型输入范围的 start/end positions 进行截断
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略 ignored_index 处的预测
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典形式的输出，则返回一个元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QuestionAnsweringModelOutput 类型的对象，包含损失、logits、隐藏状态和注意力
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```