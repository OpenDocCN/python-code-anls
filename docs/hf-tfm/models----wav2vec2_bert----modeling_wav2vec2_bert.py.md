# `.\models\wav2vec2_bert\modeling_wav2vec2_bert.py`

```
# coding=utf-8
# Copyright 2024 The Seamless Authors and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Wav2Vec2-BERT model."""

import math  # 导入数学函数库
import warnings  # 导入警告处理模块
from typing import Optional, Tuple, Union  # 导入类型提示模块

import numpy as np  # 导入数值计算库numpy
import torch  # 导入深度学习框架PyTorch
import torch.utils.checkpoint  # 导入PyTorch的checkpoint工具
from torch import nn  # 导入PyTorch的神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...integrations.deepspeed import is_deepspeed_zero3_enabled  # 导入DeepSpeed集成模块
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask  # 导入注意力掩码工具函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型工具函数
from ...utils import (  # 导入工具函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_peft_available,
    logging,
)
from .configuration_wav2vec2_bert import Wav2Vec2BertConfig  # 导入Wav2Vec2-BERT配置

logger = logging.get_logger(__name__)  # 获取日志记录器


_HIDDEN_STATES_START_POSITION = 2  # 隐藏状态的起始位置索引

# General docstring
_CONFIG_FOR_DOC = "Wav2Vec2BertConfig"  # 文档中的配置信息

# Base docstring
_BASE_CHECKPOINT_FOR_DOC = "facebook/w2v-bert-2.0"  # 基础检查点的文档字符串
_PRETRAINED_CHECKPOINT_FOR_DOC = "hf-audio/wav2vec2-bert-CV16-en"  # 预训练检查点的文档字符串
_EXPECTED_OUTPUT_SHAPE = [1, 146, 1024]  # 预期输出的形状

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'mr quilter is the apostle of the middle classes and we are glad to welcome his gospel'"  # CTC任务的预期输出示例
_CTC_EXPECTED_LOSS = 17.04  # CTC任务的预期损失

WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # Wav2Vec2-BERT模型的预训练模型列表
    "facebook/w2v-bert-2.0",
    # See all Wav2Vec2-BERT models at https://huggingface.co/models?filter=wav2vec2-bert
]


# Copied from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2._compute_new_attention_mask
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.
    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`torch.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`
    Returns:
        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`
    """
    # 获取隐藏状态张量的形状信息，并分别赋给 batch_size 和 mask_seq_len
    batch_size, mask_seq_len = hidden_states.shape[:2]
    
    # 在当前设备上创建一个张量，包含从 0 到 mask_seq_len-1 的整数序列，并扩展为二维的 batch_size 行，mask_seq_len 列的张量
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)
    
    # 创建一个布尔掩码张量，其中元素为 True 表示相应位置的索引大于等于 seq_lens 中对应的值，否则为 False
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
    
    # 创建一个与 hidden_states 张量相同形状的新张量，所有元素初始化为 1
    mask = hidden_states.new_ones((batch_size, mask_seq_len))
    
    # 使用布尔掩码 bool_mask 将 mask 中对应位置的元素置为 0
    mask = mask.masked_fill(bool_mask, 0)
    
    # 返回生成的 mask 张量，用于在序列中标记不需要处理的位置
    return mask
# 从 transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices 复制而来的函数
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算给定形状的随机掩码区间。用于实现 ASR 中的 SpecAugment 数据增强方法。
    注意：此方法未经优化以在 TPU 上运行，应作为训练过程中的预处理步骤在 CPU 上运行。

    Args:
        shape: 要计算掩码的形状。应为一个大小为 2 的元组，第一个元素是批量大小，第二个元素是要跨越的轴的长度。
        mask_prob: 将被掩盖的整个轴的百分比（介于 0 和 1 之间）。由 `mask_prob*shape[1]/mask_length` 计算生成长度为 `mask_length` 的独立掩码区间的数量。
                  由于重叠，`mask_prob` 是一个上限，实际百分比会较小。
        mask_length: 掩码的大小
        min_masks: 最小掩码数量
        attention_mask: （右填充的）注意力掩码，独立地缩短每个批处理维度的特征轴。

    Returns:
        np.ndarray: 一个布尔类型的数组，表示掩码位置的二维数组。
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` 必须大于 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` 必须小于 `sequence_length`，但得到 `mask_length`: {mask_length}"
            f" 和 `sequence_length`: {sequence_length}`"
        )

    # epsilon 用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """给定输入长度，计算应掩盖多少个区间"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保掩盖的区间数量 <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保 num_masked span 也 <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批处理中的掩盖区间数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment 掩码初始化
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    # 计算在序列长度内的最大掩盖区间数
    max_num_masked_span = compute_num_masked_span(sequence_length)
    # 如果最大被遮蔽跨度为0，则直接返回原始的spec_aug_mask
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历每个输入的长度
    for input_length in input_lengths:
        # 计算当前输入的被遮蔽跨度数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要遮蔽的索引位置
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 如果没有选中任何索引，则说明所有的输入长度小于mask_length，此时使用最后一个位置作为虚拟遮蔽索引
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只会发生在input_length严格小于sequence_length的情况下，
            # 最后一个token必须是填充token，可以用作虚拟的遮蔽ID
            dummy_mask_idx = sequence_length - 1
        else:
            # 否则使用选中的第一个索引作为虚拟遮蔽索引
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟遮蔽索引添加到spec_aug_mask_idx中，以确保所有批次的维度相同
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        # 将当前批次的遮蔽索引添加到列表中
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将遮蔽索引列表转换为NumPy数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮蔽索引扩展为遮蔽跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    # 将遮蔽索引重塑为(batch_size, max_num_masked_span * mask_length)
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 为开始索引添加偏移量，以确保索引现在创建一个跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保遮蔽索引不超过sequence_length - 1
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将遮蔽索引应用到spec_aug_mask中
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回处理后的spec_aug_mask
    return spec_aug_mask
# Copied from transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices
# 定义函数 `_sample_negative_indices`，用于从特征向量中采样负向量索引
def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    从特征向量中随机采样 `num_negatives` 个向量索引。
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    # 生成正向量本身的索引，并将其重复 `num_negatives` 次
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    # 从同一话语中获取 `num_negatives` 个随机向量索引
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    # Convert mask_time_indices to boolean if provided, otherwise create a boolean mask of all True
    # 如果提供了 mask_time_indices，则将其转换为布尔型，否则创建一个全为 True 的布尔掩码
    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    # Iterate over batches
    # 遍历每个批次
    for batch_idx in range(batch_size):
        # Determine the upper bound for valid indices based on mask_time_indices
        # 基于 mask_time_indices 确定有效索引的上界
        high = mask_time_indices[batch_idx].sum() - 1
        # Get mapped masked indices from sequence_length_range based on mask_time_indices
        # 根据 mask_time_indices 从 sequence_length_range 中获取映射后的掩码索引
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        # Create a matrix of feature indices broadcasting to shape (high + 1, num_negatives)
        # 创建一个广播到形状 (high + 1, num_negatives) 的特征索引矩阵
        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        # Sample `num_negatives` indices randomly within range (0, high)
        # 在范围 (0, high) 内随机采样 `num_negatives` 个索引
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # Avoid sampling the same positive vector, but maintain uniform distribution
        # 避免采样相同的正向量，但保持均匀分布
        sampled_indices[sampled_indices >= feature_indices] += 1

        # Remap to actual indices
        # 将采样后的索引重新映射到实际索引
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # Correct for batch size
        # 校正批次大小
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    # Return sampled negative indices
    # 返回采样的负向量索引
    return sampled_negative_indices


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRotaryPositionalEmbedding with Wav2Vec2Conformer->Wav2Vec2Bert
# 定义类 `Wav2Vec2BertRotaryPositionalEmbedding`，实现旋转位置嵌入
class Wav2Vec2BertRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    旋转位置嵌入模块
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, config):
        # 初始化方法
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base

        # Compute inverse frequencies for rotary positional embeddings
        # 计算旋转位置嵌入的反向频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        # Register inverse frequencies as a buffer, not trainable
        # 将反向频率注册为缓冲区，不参与训练
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None
    def forward(self, hidden_states):
        # 获取隐藏状态的序列长度
        sequence_length = hidden_states.shape[1]

        # 如果序列长度与缓存的序列长度相同且已缓存的旋转位置嵌入不为空，则直接返回缓存的旋转位置嵌入
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        # 更新缓存的序列长度为当前序列长度
        self.cached_sequence_length = sequence_length
        # 使用时间戳创建频率矩阵，将时间戳转换为与 inv_freq 相同的数据类型
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        # 构建嵌入向量，将频率矩阵按最后一个维度连接起来
        embeddings = torch.cat((freqs, freqs), dim=-1)

        # 计算嵌入向量的余弦和正弦值
        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # 将计算得到的嵌入向量转换为与隐藏状态输入相同的数据类型
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        # 返回缓存的旋转位置嵌入
        return self.cached_rotary_positional_embedding
# 从transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRelPositionalEmbedding复制的代码，
# 将Wav2Vec2Conformer改为Wav2Vec2Bert
class Wav2Vec2BertRelPositionalEmbedding(nn.Module):
    """相对位置编码模块。"""

    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_source_positions  # 设置最大长度为配置中的源位置最大数
        self.d_model = config.hidden_size  # 设置模型的隐藏层大小为配置中的隐藏大小
        self.pe = None  # 初始化位置编码为None
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))  # 调用extend_pe方法扩展位置编码

    def extend_pe(self, x):
        # 重置位置编码
        if self.pe is not None:
            # self.pe包含正负两部分
            # self.pe的长度为2 * 输入长度 - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # 假设`i`是查询向量的位置，`j`是键向量的位置。当键位于左侧时（i>j），使用正的相对位置，否则使用负的相对位置。
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.int64).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.int64).float() * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # 反转正索引的顺序，并连接正负索引。这用于支持位移技巧，参见https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        self.extend_pe(hidden_states)
        start_idx = self.pe.size(1) // 2 - hidden_states.size(1) + 1
        end_idx = self.pe.size(1) // 2 + hidden_states.size(1)
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


class Wav2Vec2BertFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)
    # 定义一个方法 forward，接收隐藏状态作为输入参数
    def forward(self, hidden_states):
        # 对隐藏状态进行层归一化处理，用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 对归一化后的隐藏状态进行投影
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态和归一化前的隐藏状态
        return hidden_states, norm_hidden_states
class Wav2Vec2BertFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, hidden_size=None):
        super().__init__()
        act_fn = act_fn if act_fn is not None else config.hidden_act  # 设置激活函数，如果未提供则使用配置中的默认值
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size  # 设置隐藏层大小，如果未提供则使用配置中的默认值
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)  # 中间层使用激活dropout

        self.intermediate_dense = nn.Linear(hidden_size, config.intermediate_size)  # 中间层的全连接层
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn  # 中间层的激活函数

        self.output_dense = nn.Linear(config.intermediate_size, hidden_size)  # 输出层的全连接层
        self.output_dropout = nn.Dropout(config.hidden_dropout)  # 输出层使用隐藏dropout

    # 从transformers库中的wav2vec2模型中的Wav2Vec2FeedForward类复制而来的forward方法
    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)  # 中间层的全连接操作
        hidden_states = self.intermediate_act_fn(hidden_states)  # 中间层的激活函数操作
        hidden_states = self.intermediate_dropout(hidden_states)  # 中间层的dropout操作

        hidden_states = self.output_dense(hidden_states)  # 输出层的全连接操作
        hidden_states = self.output_dropout(hidden_states)  # 输出层的dropout操作
        return hidden_states


class Wav2Vec2BertConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 层归一化操作
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )  # 第一个1x1卷积层

        self.glu = nn.GLU(dim=1)  # GLU激活函数，应用在第一个卷积层的输出上
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )  # 深度卷积层，使用组卷积来处理每个通道独立的操作

        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 深度卷积层后的层归一化
        self.activation = ACT2FN[config.hidden_act]  # 使用指定的激活函数
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )  # 第二个1x1卷积层

        self.dropout = nn.Dropout(config.conformer_conv_dropout)  # 卷积模块的dropout操作
    # 对输入的 hidden_states 进行层归一化处理
    hidden_states = self.layer_norm(hidden_states)

    # 如果传入了 attention_mask，确保在深度卷积中不泄露填充位置的信息
    if attention_mask is not None:
        # 将 attention_mask 转换成布尔张量，并在未填充位置上用 0 替换
        hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

    # 交换 hidden_states 张量的第一维（batch 维）和第二维（时间步维）的顺序
    hidden_states = hidden_states.transpose(1, 2)

    # 应用 GLU 机制，通过 pointwise_conv1 进行卷积操作
    # 结果张量维度变为 (batch, channel, dim)
    hidden_states = self.pointwise_conv1(hidden_states)

    # 经过 GLU 激活函数处理，输出维度为 (batch, channel, dim)
    hidden_states = self.glu(hidden_states)

    # 对 hidden_states 序列进行左侧填充，以适应因果卷积的需要
    hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

    # 应用一维深度卷积操作，处理输入序列
    hidden_states = self.depthwise_conv(hidden_states)

    # 对深度卷积后的 hidden_states 进行层归一化，然后恢复原始维度顺序
    hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    # 应用激活函数处理 hidden_states
    hidden_states = self.activation(hidden_states)

    # 通过 pointwise_conv2 进行卷积操作
    hidden_states = self.pointwise_conv2(hidden_states)

    # 对输出进行 dropout 处理，以防止过拟合
    hidden_states = self.dropout(hidden_states)

    # 最后再次交换 hidden_states 张量的第一维和第二维的顺序，返回结果
    hidden_states = hidden_states.transpose(1, 2)
    return hidden_states
    """Construct an Wav2Vec2BertSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """
    # 定义 Wav2Vec2BertSelfAttention 类，用于构建自注意力机制模块，支持旋转或相对位置编码的增强功能

    def __init__(self, config, is_adapter_attention=False):
        super().__init__()
        # 调用父类构造函数进行初始化

        hidden_size = config.hidden_size if not is_adapter_attention else config.output_hidden_size
        # 根据是否适配器注意力选择隐藏层大小或输出隐藏层大小

        self.head_size = hidden_size // config.num_attention_heads
        # 计算每个注意力头的大小
        self.num_heads = config.num_attention_heads
        # 设置注意力头的数量
        self.position_embeddings_type = config.position_embeddings_type if not is_adapter_attention else None
        # 根据是否适配器注意力选择位置编码类型或设为None

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        # Query 线性变换层
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        # Key 线性变换层
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        # Value 线性变换层
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        # 输出线性变换层

        self.dropout = nn.Dropout(p=config.attention_dropout)
        # Dropout 层，用于注意力计算时的随机失活

        if self.position_embeddings_type == "relative":
            # 如果位置编码类型为 "relative"
            self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=False)
            # 用于位置编码的线性变换层
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            # 用于矩阵 c 的可学习偏置参数
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            # 用于矩阵 d 的可学习偏置参数

        if self.position_embeddings_type == "relative_key":
            # 如果位置编码类型为 "relative_key"
            self.left_max_position_embeddings = config.left_max_position_embeddings
            # 左侧最大位置编码的数量
            self.right_max_position_embeddings = config.right_max_position_embeddings
            # 右侧最大位置编码的数量
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            # 总位置数量
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)
            # 距离编码的嵌入层，根据位置数量和头大小初始化

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 模型的前向传播函数，接收输入张量和可选的注意力掩码、相对位置编码张量以及输出注意力权重的标志

        # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_rotary_embedding
        # 从 transformers 库中的另一个模块复制的部分，用于应用旋转嵌入的函数
    # 对输入的隐藏状态应用旋转嵌入
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        # 获取批次大小、序列长度和隐藏单元大小
        batch_size, sequence_length, hidden_size = hidden_states.size()
        
        # 将隐藏状态重新形状为(batch_size, sequence_length, num_heads, head_size)
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        # 从相对位置嵌入中提取余弦和正弦值
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # 将隐藏状态进行转置
        hidden_states = hidden_states.transpose(0, 1)
        
        # 分割旋转的状态，分别处理前半部分和后半部分
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        
        # 拼接旋转后的状态，按照最后一个维度进行拼接
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        
        # 应用旋转嵌入到隐藏状态中
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        
        # 恢复隐藏状态的转置
        hidden_states = hidden_states.transpose(0, 1)

        # 将隐藏状态重新形状为(batch_size, sequence_length, num_heads * head_size)
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        # 返回处理后的隐藏状态
        return hidden_states

    # 从transformers库中的wav2vec2_conformer模型复制的代码，用于应用相对位置嵌入
    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        # 1. project positional embeddings
        # 将相对位置嵌入投影到新的空间
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        # 重新组织张量形状以适应多头注意力机制的计算需求
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        # 调整张量的维度顺序以进行多头注意力计算
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

        # 2. Add bias to query
        # 将偏置项添加到查询向量中，以引入位置信息
        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        # 3. attention score: first compute matrix a and matrix c
        # 根据文献中描述的方法计算注意力矩阵 A 和 C
        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        # 4. then compute matrix b and matrix d
        # 根据文献中描述的方法计算注意力矩阵 B 和 D
        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # 5. shift matrix b and matrix d
        # 在注意力矩阵 B 上进行零填充和移位操作
        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.size(-1) // 2 + 1]

        # 6. sum matrices
        # 将计算得到的注意力矩阵 A+C 和修正后的注意力矩阵 B 加总并除以缩放因子
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores
class Wav2Vec2BertEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn1 = Wav2Vec2BertFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = Wav2Vec2BertSelfAttention(config)

        # Conformer Convolution
        self.conv_module = Wav2Vec2BertConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.ffn2 = Wav2Vec2BertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)  # Layer normalization on the input
        hidden_states = self.ffn1(hidden_states)  # First feed-forward neural network transformation
        hidden_states = hidden_states * 0.5 + residual  # Residual connection and scaling

        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)  # Layer normalization on the output of FFN1
        hidden_states, attn_weights = self.self_attn(  # Self-attention mechanism
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)  # Dropout applied to self-attention output
        hidden_states = hidden_states + residual  # Residual connection after self-attention

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)  # Conformer convolution operation
        hidden_states = residual + hidden_states  # Residual connection after convolutional module

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)  # Layer normalization on the output of the convolution
        hidden_states = self.ffn2(hidden_states)  # Second feed-forward neural network transformation
        hidden_states = hidden_states * 0.5 + residual  # Residual connection and scaling
        hidden_states = self.final_layer_norm(hidden_states)  # Final layer normalization

        return hidden_states, attn_weights  # Return final hidden states and attention weights
    # 初始化函数，用于创建一个新的神经网络模型实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置信息保存在对象实例中
        self.config = config

        # 根据配置文件中的位置嵌入类型，选择不同的位置嵌入方法
        if config.position_embeddings_type == "relative":
            # 如果位置嵌入类型为相对位置嵌入，则使用Wav2Vec2BertRelPositionalEmbedding类
            self.embed_positions = Wav2Vec2BertRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            # 如果位置嵌入类型为旋转位置嵌入，则使用Wav2Vec2BertRotaryPositionalEmbedding类
            self.embed_positions = Wav2Vec2BertRotaryPositionalEmbedding(config)
        else:
            # 如果未指定有效的位置嵌入类型，则将位置嵌入设为None
            self.embed_positions = None

        # 定义一个用于随机失活的Dropout层，根据配置中的隐藏层失活率来设置
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建一个神经网络层的列表，其中每一层都是Wav2Vec2BertEncoderLayer的实例，数量由配置文件中的隐藏层数决定
        self.layers = nn.ModuleList([Wav2Vec2BertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认情况下，梯度检查点设置为False
        self.gradient_checkpointing = False

    # 前向传播函数，接收输入状态并进行模型前向计算
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
class Wav2Vec2BertAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        # feature dim might need to be down-projected
        # 如果配置中输出的隐藏大小与隐藏大小不同，可能需要降维
        if config.output_hidden_size != config.hidden_size:
            # 创建线性层，用于将隐藏状态降维到输出的隐藏大小
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # 创建 LayerNorm 层，用于归一化降维后的隐藏状态
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size, eps=config.layer_norm_eps)
        else:
            self.proj = self.proj_layer_norm = None
        # 创建多个 Wav2Vec2BertAdapterLayer 层组成的列表
        self.layers = nn.ModuleList(Wav2Vec2BertAdapterLayer(config) for _ in range(config.num_adapter_layers))
        # 设置层间隔率
        self.layerdrop = config.layerdrop

        # 获取适配器卷积的核大小和步长
        self.kernel_size = config.adapter_kernel_size
        self.stride = config.adapter_stride

    def _compute_sub_sample_lengths_from_attention_mask(self, seq_lens):
        # 如果序列长度为空，则返回空
        if seq_lens is None:
            return seq_lens
        # 计算填充长度
        pad = self.kernel_size // 2
        # 计算子采样长度
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1
        return seq_lens.floor()

    def forward(self, hidden_states, attention_mask=None):
        # 如果需要降维隐藏状态
        if self.proj is not None and self.proj_layer_norm is not None:
            # 降维隐藏状态
            hidden_states = self.proj(hidden_states)
            # 对降维后的隐藏状态进行 LayerNorm 归一化
            hidden_states = self.proj_layer_norm(hidden_states)

        # 初始化子采样长度为 None
        sub_sampled_lengths = None
        # 如果存在注意力遮罩
        if attention_mask is not None:
            # 计算子采样长度
            sub_sampled_lengths = (attention_mask.size(1) - (1 - attention_mask.int()).sum(1)).to(hidden_states.device)

        # 遍历每个适配器层
        for layer in self.layers:
            # 随机生成一个 layerdrop 概率值
            layerdrop_prob = torch.rand([])
            # 根据注意力遮罩计算子采样长度
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(sub_sampled_lengths)
            # 如果处于推理阶段或者未丢弃该层
            if not self.training or (layerdrop_prob > self.layerdrop):
                # 将隐藏状态传递给适配器层处理
                hidden_states = layer(
                    hidden_states, attention_mask=attention_mask, sub_sampled_lengths=sub_sampled_lengths
                )

        # 返回处理后的隐藏状态
        return hidden_states


class Wav2Vec2BertAdapterLayer(nn.Module):
    # 待实现
    def __init__(self, config):
        super().__init__()
        embed_dim = config.output_hidden_size  # 从配置中获取嵌入维度大小
        dropout = config.conformer_conv_dropout  # 从配置中获取卷积层的dropout率

        self.kernel_size = config.adapter_kernel_size  # 从配置中获取卷积核大小
        self.stride = config.adapter_stride  # 从配置中获取卷积的步长

        # 1. residual convolution
        self.residual_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 使用 LayerNorm 对残差连接后的特征进行归一化
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 定义一个一维卷积层，用于残差连接

        self.activation = nn.GLU(dim=1)
        # 定义一个门控线性单元（GLU），应用于卷积输出

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 使用 LayerNorm 对自注意力层的输出进行归一化
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 定义一个一维卷积层，用于自注意力机制

        self.self_attn = Wav2Vec2BertSelfAttention(config, is_adapter_attention=True)
        # 创建一个自定义的自注意力层实例

        self.self_attn_dropout = nn.Dropout(dropout)
        # 定义一个dropout层，用于自注意力的输出

        # Feed-forward
        self.ffn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 使用 LayerNorm 对前馈网络层的输出进行归一化
        self.ffn = Wav2Vec2BertFeedForward(config, act_fn=config.adapter_act, hidden_size=embed_dim)
        # 创建一个自定义的前馈网络层实例，用于特征转换和映射
    ):
        # 计算残差连接的归一化
        residual = self.residual_layer_norm(hidden_states)

        # 对残差进行池化，以匹配多头注意力输出的序列长度。
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        # 对自注意力层的隐藏状态进行归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 在输入多头注意力层之前进行池化。
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        # 如果存在注意力掩码，进行相应的计算
        if attention_mask is not None:
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # 剩余的计算步骤与普通的Transformer编码器层相同。
        hidden_states, attn_weigths = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 更新残差
        residual = hidden_states

        # 应用前馈网络层的归一化
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states
# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerPreTrainedModel
# with Wav2Vec2Conformer->Wav2Vec2Bert, wav2vec2_conformer->wav2vec2_bert, input_values->input_features

class Wav2Vec2BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2BertConfig  # 指定配置类为Wav2Vec2BertConfig
    base_model_prefix = "wav2vec2_bert"  # 模型的基本前缀名
    main_input_name = "input_features"  # 主要输入名称为input_features
    supports_gradient_checkpointing = True  # 支持梯度检查点

    # Ignore copy
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Wav2Vec2BertSelfAttention):
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)  # 使用Xavier初始化pos_bias_u
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)  # 使用Xavier初始化pos_bias_v
        elif isinstance(module, Wav2Vec2BertFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)  # 均匀分布初始化projection.weight
            nn.init.uniform_(module.projection.bias, a=-k, b=k)  # 均匀分布初始化projection.bias
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 正态分布初始化weight
            if module.bias is not None:
                module.bias.data.zero_()  # 将bias初始化为零
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()  # 将bias初始化为零
            module.weight.data.fill_(1.0)  # 将weight初始化为1.0
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)  # 使用Kaiming正态分布初始化weight
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)  # 均匀分布初始化bias

    # Ignore copy
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride, padding):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length + 2 * padding - kernel_size, stride, rounding_mode="floor") + 1

        if add_adapter:
            padding = self.config.adapter_kernel_size // 2
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(
                    input_lengths, self.config.adapter_kernel_size, self.config.adapter_stride, padding
                )

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
    ):
        # 计算没有填充的部分的长度，即 attention_mask.sum(-1)，但不能原地操作以便在推断模式下运行。
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 根据非填充长度获取特征提取器的输出长度，并根据需要添加适配器
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        # 获取批处理大小
        batch_size = attention_mask.shape[0]

        # 创建一个全零的注意力掩码张量，形状为 (batch_size, feature_vector_length)，类型与设备与 attention_mask 一致
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 确保在输出长度索引之前的所有位置都被注意到
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1

        # 将注意力掩码张量沿着最后一个维度翻转，然后累积求和，并再次翻转，最后将其转换为布尔类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        # 返回处理后的注意力掩码张量
        return attention_mask
# 定义 Wav2Vec2BertModel 的文档字符串，描述了该模型的基本信息和引用的论文
WAV2VEC2_BERT_START_DOCSTRING = r"""
    Wav2Vec2Bert was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
    Auli.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) sub-class. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`Wav2Vec2BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 Wav2Vec2BertModel 的输入文档字符串，描述了模型的输入参数及其含义
WAV2VEC2_BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_features`, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2BertProcessor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用 @add_start_docstrings 装饰器添加了额外的文档字符串，描述了 Wav2Vec2BertModel 的基本信息和配置参数
@add_start_docstrings(
    "The bare Wav2Vec2Bert Model transformer outputting raw hidden-states without any specific head on top.",
    WAV2VEC2_BERT_START_DOCSTRING,
)
    # 初始化函数，接收一个 Wav2Vec2BertConfig 类型的参数 config
    def __init__(self, config: Wav2Vec2BertConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将传入的 config 参数保存到 self.config 中
        self.config = config
        # 创建一个 Wav2Vec2BertFeatureProjection 对象并保存到 self.feature_projection 中
        self.feature_projection = Wav2Vec2BertFeatureProjection(config)

        # 如果 config 中 mask_time_prob 大于 0.0 或者 config 中 mask_feature_prob 大于 0.0，
        # 则需要创建一个 nn.Parameter 类型的张量 self.masked_spec_embed
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            # 用均匀分布初始化 self.masked_spec_embed
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 创建一个 Wav2Vec2BertEncoder 对象并保存到 self.encoder 中
        self.encoder = Wav2Vec2BertEncoder(config)

        # 如果 config 中 add_adapter 为 True，则创建一个 Wav2Vec2BertAdapter 对象并保存到 self.adapter 中
        # 否则 self.adapter 为 None
        self.adapter = Wav2Vec2BertAdapter(config) if config.add_adapter else None

        # 如果 config 中 use_intermediate_ffn_before_adapter 为 True，
        # 则创建一个 Wav2Vec2BertFeedForward 对象并保存到 self.intermediate_ffn 中
        # 激活函数为 "relu"
        self.intermediate_ffn = None
        if config.use_intermediate_ffn_before_adapter:
            self.intermediate_ffn = Wav2Vec2BertFeedForward(config, act_fn="relu")

        # 调用类的 post_init 方法，用于初始化权重和应用最终处理
        self.post_init()
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # compute mask indices for time axis if not provided
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    @add_start_docstrings_to_model_forward(WAV2VEC2_BERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_PRETRAINED_CHECKPOINT_FOR_DOC,
        output_type=Wav2Vec2BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 如果未指定是否输出注意力权重，则使用模型配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定是否输出隐藏状态，则使用模型配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定是否返回字典形式的输出，则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入特征映射到特征投影空间
        hidden_states, extract_features = self.feature_projection(input_features)
        # 根据给定的时间索引和注意力掩码对隐藏状态进行掩码操作
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 编码器处理隐藏状态，并返回编码器的输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的隐藏状态作为下一步的处理对象
        hidden_states = encoder_outputs[0]

        # 如果存在中间的Feed Forward Network，则对隐藏状态进行扩展处理
        if self.intermediate_ffn:
            expanded_hidden_states = self.intermediate_ffn(hidden_states)
            hidden_states = hidden_states + 0.5 * expanded_hidden_states

        # 如果存在适配器，则使用适配器处理隐藏状态
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        # 如果不要求返回字典形式的输出，则返回一个元组，包括隐藏状态、提取的特征以及可能的额外输出
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回 Wav2Vec2BaseModelOutput 类的对象，包括最终的隐藏状态、提取的特征、编码器的隐藏状态和注意力权重
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 为 Wav2Vec2BertForCTC 类添加文档字符串，描述其作为 Connectionist Temporal Classification (CTC) 语言建模头部的 Wav2Vec2Bert 模型。
@add_start_docstrings(
    """Wav2Vec2Bert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV2VEC2_BERT_START_DOCSTRING,
)
class Wav2Vec2BertForCTC(Wav2Vec2BertPreTrainedModel):
    # 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForCTC.__init__ 复制过来，将 Wav2Vec2Conformer 类重命名为 Wav2Vec2Bert，WAV2VEC2_CONFORMER 重命名为 WAV2VEC2_BERT，wav2vec2_conformer 重命名为 wav2vec2_bert。
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        # 使用 Wav2Vec2BertModel 创建 wav2vec2_bert 模型
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言
        self.target_lang = target_lang

        # 如果配置中未定义语言模型头部的词汇表大小，则抛出值错误异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2BertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        # 根据配置信息初始化线性层 lm_head
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加文档字符串到模型前向方法，描述 WAV2VEC2_BERT 输入的格式和用途
    @add_start_docstrings_to_model_forward(WAV2VEC2_BERT_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串，指定检查点、输出类型、配置类、预期输出和预期损失
    @add_code_sample_docstrings(
        checkpoint=_PRETRAINED_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        # 输入特征：可选的 torch.Tensor 类型
        # 注意力掩码：可选的 torch.Tensor 类型，默认为 None
        # 输出注意力：可选的布尔类型，默认为 None
        # 输出隐藏状态：可选的布尔类型，默认为 None
        # 返回字典：可选的布尔类型，默认为 None
        # 标签：可选的 torch.Tensor 类型，默认为 None
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # Determine whether to use return_dict based on provided argument or configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input features through wav2vec2_bert model
        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Retrieve hidden states from the model output and apply dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Generate logits using the language model head
        logits = self.lm_head(hidden_states)

        # Initialize loss variable
        loss = None
        if labels is not None:
            # Check if any label value exceeds the vocabulary size
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # Calculate input_lengths based on attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(input_features.shape[:2], device=input_features.device, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum([-1])).to(torch.long)

            # Mask out invalid labels and compute target_lengths
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # Compute log probabilities using log_softmax and transpose for CTC loss computation
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # Disable cudnn to ensure reproducibility in loss calculation
            with torch.backends.cudnn.flags(enabled=False):
                # Compute CTC loss using log_probs and other parameters
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        # If return_dict is False, return a tuple with logits and other outputs
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, return CausalLMOutput object with necessary attributes
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
"""
Wav2Vec2Bert Model with a sequence classification head on top (a linear layer over the pooled output) for
tasks like SUPERB Keyword Spotting.
"""
@add_start_docstrings(
    """
    Wav2Vec2Bert Model with a sequence classification head on top (a linear layer over the pooled output) for
    tasks like SUPERB Keyword Spotting.
    """,
    WAV2VEC2_BERT_START_DOCSTRING,
)
class Wav2Vec2BertForSequenceClassification(Wav2Vec2BertPreTrainedModel):
    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.__init__ with Wav2Vec2->Wav2Vec2Bert,wav2vec2->wav2vec2_bert
    def __init__(self, config):
        super().__init__(config)

        # Check if adapter usage is enabled; raise error if so
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2Bert adapters (config.add_adapter=True)"
            )
        
        # Initialize the Wav2Vec2BertModel
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        
        # Calculate the number of layers including input embeddings
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        
        # Initialize layer weights if weighted layer sum is enabled
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Project pooled output to a specified size
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        
        # Final linear layer for classification
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # Freeze parameters of the Wav2Vec2BertModel
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV2VEC2_BERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_BASE_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward with Wav2Vec2->Wav2Vec2Bert,wav2vec2->wav2vec2_bert,WAV_2_VEC_2->WAV2VEC2_BERT, input_values->input_features
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        ):
        # Implement the forward pass for Wav2Vec2BertForSequenceClassification
        pass
        ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回的字典对象，如果未指定则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏层状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用wav2vec2_bert模型，传入输入特征和其他参数，获取输出结果
        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 根据配置决定是否使用加权层求和
        if self.config.use_weighted_layer_sum:
            # 提取隐藏状态，并按照层权重进行加权求和
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 直接使用第一个输出作为隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态投影到特征空间
        hidden_states = self.projector(hidden_states)

        # 如果没有提供注意力掩码，则使用平均值作为汇聚输出
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            # 根据注意力掩码生成特征向量的掩码，并根据掩码对隐藏状态进行填充
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            # 按照掩码进行加权求和，得到汇聚输出
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 对汇聚输出进行分类
        logits = self.classifier(pooled_output)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 如果不使用返回字典，则按照旧版的输出格式返回结果
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 使用新版的SequenceClassifierOutput格式返回结果
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 用于音频帧分类任务的 Wav2Vec2Bert 模型，其在顶部带有帧分类头部。
@add_start_docstrings(
    """
    Wav2Vec2Bert Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    WAV2VEC2_BERT_START_DOCSTRING,
)
class Wav2Vec2BertForAudioFrameClassification(Wav2Vec2BertPreTrainedModel):
    # 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForAudioFrameClassification.__init__ 复制，并将 Wav2Vec2Conformer 替换为 Wav2Vec2Bert，WAV2VEC2_CONFORMER 替换为 WAV2VEC2_BERT，wav2vec2_conformer 替换为 wav2vec2_bert
    def __init__(self, config):
        super().__init__(config)

        # 检查是否存在适配器并且配置要求使用适配器，如果是则引发值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2Bert adapters (config.add_adapter=True)"
            )
        
        # 创建 Wav2Vec2BertModel 实例并赋值给 self.wav2vec2_bert
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        
        # 计算变换层的数量（变换器层 + 输入嵌入层），如果配置要求使用加权层求和，则初始化权重
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 创建分类器线性层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 保存标签数量到 self.num_labels
        self.num_labels = config.num_labels

        # 初始化模型权重
        self.init_weights()

    # 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForAudioFrameClassification.freeze_base_model 复制，并将 wav2vec2_conformer 替换为 wav2vec2_bert
    def freeze_base_model(self):
        """
        调用此函数将禁用基础模型的梯度计算，使其参数在训练过程中不会被更新。仅分类头部将被更新。
        """
        # 遍历 self.wav2vec2_bert 的所有参数，并设置 requires_grad=False 来禁用梯度计算
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV2VEC2_BERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_BASE_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForAudioFrameClassification.forward 复制，并将 wav2vec2_conformer 替换为 wav2vec2_bert，input_values 替换为 input_features
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # 函数前向传播，接受输入特征 input_features（可选的 torch.Tensor）
        # attention_mask（可选的 torch.Tensor）用于指定哪些元素需要注意，labels（可选的 torch.Tensor）用于指定预测的标签
        # output_attentions（可选的 bool）指示是否返回注意力权重，output_hidden_states（可选的 bool）指示是否返回隐藏状态
        # return_dict（可选的 bool）指示是否返回字典格式的输出
        
        # 使用 self.wav2vec2_bert 进行前向传播，将输入特征 input_features 作为输入
        # 返回的结果为 TokenClassifierOutput 类型的输出
        
        # 具体使用示例请参考代码库中的模型检查点 _BASE_CHECKPOINT_FOR_DOC
        # 返回结果类型为 TokenClassifierOutput，配置类为 _CONFIG_FOR_DOC，处理的模态为音频

        pass


这里只是注释代码，没有实际的代码内容需要输出。
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 根据需要设置是否返回字典类型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置选择是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用wav2vec2_bert模型进行前向传播
        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置为使用加权层求和，则对隐藏状态进行加权求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态列表的起始位置
            hidden_states = torch.stack(hidden_states, dim=1)  # 在指定维度上堆叠隐藏状态
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行softmax归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 按照权重求和隐藏状态
        else:
            hidden_states = outputs[0]  # 否则直接使用第一个输出作为隐藏状态

        logits = self.classifier(hidden_states)  # 使用分类器对隐藏状态进行分类

        loss = None
        if labels is not None:
            # 如果提供了标签，计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        if not return_dict:
            # 如果不需要返回字典类型的输出，则返回分类器的logits和可能的隐藏状态列表
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        # 否则返回一个TokenClassifierOutput对象，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义 AMSoftmaxLoss 类，用于实现 AM-Softmax 损失函数
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        # 设置 AM-Softmax 的参数：缩放因子和边界值
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        # 使用随机初始化的权重作为模型参数，需计算梯度
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        # 使用交叉熵损失作为损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        # 将标签展平以便与预测结果匹配
        labels = labels.flatten()
        # 对权重进行 L2 归一化
        weight = nn.functional.normalize(self.weight, dim=0)
        # 对输入的隐藏状态进行 L2 归一化
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        # 计算余弦相似度
        cos_theta = torch.mm(hidden_states, weight)
        # 计算 AM-Softmax 中的 psi 值
        psi = cos_theta - self.margin

        # 将标签转换为独热编码
        onehot = nn.functional.one_hot(labels, self.num_labels)
        # 计算最终的预测 logits
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        # 计算损失
        loss = self.loss(logits, labels)

        return loss


# 定义 TDNNLayer 类，实现时间延迟神经网络中的一层
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 从配置中获取输入和输出维度，以及卷积核大小和扩张率
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        # 使用线性层作为卷积核
        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        # 激活函数为 ReLU
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 检查是否可用 peft 库，并警告用户
        if is_peft_available():
            from peft.tuners.lora import LoraLayer

            if isinstance(self.kernel, LoraLayer):
                warnings.warn(
                    "Detected LoRA on TDNNLayer. LoRA weights won't be applied due to optimization. "
                    "You should exclude TDNNLayer from LoRA's target modules.",
                )

        # 调整输入张量的维度顺序以进行卷积计算
        hidden_states = hidden_states.transpose(1, 2)
        # 调整卷积核的形状以匹配卷积函数的要求
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        # 使用函数式 API 执行一维卷积操作
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        # 再次调整张量的维度顺序以还原原始形状
        hidden_states = hidden_states.transpose(1, 2)

        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    用于 Speaker Verification 等任务的 Wav2Vec2Bert 模型，顶部带有 XVector 特征提取头。
    """,
    WAV2VEC2_BERT_START_DOCSTRING,
)
# 定义 Wav2Vec2BertForXVector 类，继承自 Wav2Vec2BertPreTrainedModel 类
class Wav2Vec2BertForXVector(Wav2Vec2BertPreTrainedModel):
    # 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForXVector.__init__ 复制，替换相关字符串
    # 初始化函数，接受配置参数并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 创建Wav2Vec2BertModel模型实例
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        
        # 计算层数，包括Transformer层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        
        # 如果配置中使用加权层求和，则初始化层权重为均匀分布
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 创建线性层，用于将隐藏状态映射到TDNN输入维度
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 创建TDNN层列表
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 创建特征提取器线性层，将TDNN输出映射到x-vector输出维度
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        
        # 创建分类器线性层，将x-vector输出映射到类别数目维度
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 创建AMSoftmax损失函数实例，用于训练中的目标函数
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 初始化模型权重
        self.init_weights()

    # 冻结基础模型，使得在训练过程中不更新其参数
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    # 计算TDNN层的输出长度
    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        # 遍历每个TDNN层的卷积核大小，更新输入长度
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    @add_start_docstrings_to_model_forward(WAV2VEC2_BERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_BASE_CHECKPOINT_FOR_DOC,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 定义前向传播函数，将wav2vec2_bert改名为wav2vec2_bert，input_values改名为input_features
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用wav2vec2_bert模型进行前向传播
        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用加权层求和，则进行加权求和操作
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则，直接使用输出的第一个隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态投影到指定维度
        hidden_states = self.projector(hidden_states)

        # 对每一层的TDNN进行前向传播
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # 统计汇聚操作
        if attention_mask is None:
            # 如果没有注意力掩码，则计算整体平均值和标准差
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            # 如果有注意力掩码，则根据掩码计算每层的长度
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            # 对每一层进行统计汇聚
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        # 将均值和标准差拼接在一起作为统计汇聚结果
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        # 使用特征提取器处理统计汇聚的结果
        output_embeddings = self.feature_extractor(statistic_pooling)
        # 使用分类器得到最终的logits
        logits = self.classifier(output_embeddings)

        # 计算损失值（如果有标签的话）
        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        # 根据是否使用返回字典决定返回的内容
        if not return_dict:
            # 如果不使用返回字典，则返回一个元组，包含logits、output_embeddings和所有隐藏状态
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典，则创建XVectorOutput对象并返回
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```