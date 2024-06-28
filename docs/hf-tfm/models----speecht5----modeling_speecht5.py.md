# `.\models\speecht5\modeling_speecht5.py`

```py
# coding=utf-8
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch SpeechT5 model."""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSpectrogramOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 1

# General docstring
_CONFIG_FOR_DOC = "SpeechT5Config"


SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/speecht5_asr",
    "microsoft/speecht5_tts",
    "microsoft/speecht5_vc",
    # See all SpeechT5 models at https://huggingface.co/models?filter=speecht5
]


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def shift_spectrograms_right(input_values: torch.Tensor, reduction_factor: int = 1):
    """
    Shift input spectrograms one timestep to the right. Also applies the reduction factor to the sequence length.
    """
    # thin out frames for reduction factor
    if reduction_factor > 1:
        input_values = input_values[:, reduction_factor - 1 :: reduction_factor]

    # Initialize a tensor of zeros with the same shape as input_values
    shifted_input_values = input_values.new_zeros(input_values.shape)
    # 将输入数据的每一行的除第一列外的所有列向右移动一个位置，使用输入数据的副本进行操作
    shifted_input_values[:, 1:] = input_values[:, :-1].clone()
    
    # 将输入数据中可能存在的标签为 -100 的数值替换为零
    shifted_input_values.masked_fill_(shifted_input_values == -100.0, 0.0)
    
    # 返回经过处理后的移位后的输入数据
    return shifted_input_values
# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob: The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                   independently generated mask spans of length `mask_length` is computed by
                   `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                   actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape  # 解构元组，获取批量大小和序列长度

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")  # 如果 mask_length 不大于 1，则抛出错误

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )  # 如果 mask_length 大于序列长度，则抛出错误

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()  # 生成一个随机数 epsilon，用于概率取整

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)  # 计算应该遮罩的 span 数量
        num_masked_span = max(num_masked_span, min_masks)  # 确保遮罩的 span 数量不低于 min_masks

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length  # 确保遮罩的 span 数量不超过序列长度

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)  # 确保遮罩的 span 数量不超过 input_length - (mask_length - 1)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()  # 如果 attention_mask 不为空，计算每个样本的有效长度
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]  # 否则默认每个样本的有效长度为 sequence_length
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 创建一个全零的布尔类型数组，用于存放遮罩

    spec_aug_mask_idxs = []  # 初始化存放遮罩索引的列表

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算序列长度内的最大遮罩 span 数量
    # 如果最大可屏蔽跨度为0，则直接返回特定的屏蔽掩码
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历每个输入序列的长度
    for input_length in input_lengths:
        # 计算当前输入序列的屏蔽跨度数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要屏蔽的索引位置
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 如果没有选择到任何索引，说明所有位置都被占用，使用最后一个位置作为虚拟屏蔽索引
        if len(spec_aug_mask_idx) == 0:
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 扩展屏蔽索引数组，以确保长度达到最大可屏蔽跨度
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将屏蔽索引数组转换为 numpy 数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将屏蔽索引扩展为屏蔽跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量以形成完整的屏蔽跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保屏蔽的索引不超过序列长度
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 在指定位置上进行屏蔽操作，将屏蔽结果存储在 spec_aug_mask 中
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回最终的特殊增强屏蔽结果
    return spec_aug_mask
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer 复制而来，将 Wav2Vec2 替换为 SpeechT5
class SpeechT5NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为前一层的卷积维度（如果有的话），否则为 1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个 1D 卷积层，输入维度为 in_conv_dim，输出维度为 out_conv_dim
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 使用配置中的卷积核大小
            stride=config.conv_stride[layer_id],      # 使用配置中的步幅大小
            bias=config.conv_bias,                    # 是否使用配置中的偏置
        )
        # 使用配置中指定的激活函数名，从全局的 ACT2FN 字典中获取相应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将隐藏状态传递给卷积层进行计算
        hidden_states = self.conv(hidden_states)
        # 将卷积层的输出应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer 复制而来，将 Wav2Vec2 替换为 SpeechT5
class SpeechT5LayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为前一层的卷积维度（如果有的话），否则为 1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个 1D 卷积层，输入维度为 in_conv_dim，输出维度为 out_conv_dim
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 使用配置中的卷积核大小
            stride=config.conv_stride[layer_id],      # 使用配置中的步幅大小
            bias=config.conv_bias,                    # 是否使用配置中的偏置
        )
        # 创建一个 LayerNorm 层，对输出的卷积结果进行归一化处理
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 使用配置中指定的激活函数名，从全局的 ACT2FN 字典中获取相应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将隐藏状态传递给卷积层进行计算
        hidden_states = self.conv(hidden_states)

        # 对卷积结果进行维度转换，调整至合适的形状
        hidden_states = hidden_states.transpose(-2, -1)
        # 将转置后的结果输入到 LayerNorm 层中进行归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 再次将维度转换回原始形状
        hidden_states = hidden_states.transpose(-2, -1)

        # 最后应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer 复制而来，将 Wav2Vec2 替换为 SpeechT5
class SpeechT5GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为前一层的卷积维度（如果有的话），否则为 1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个 1D 卷积层，输入维度为 in_conv_dim，输出维度为 out_conv_dim
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 使用配置中的卷积核大小
            stride=config.conv_stride[layer_id],      # 使用配置中的步幅大小
            bias=config.conv_bias,                    # 是否使用配置中的偏置
        )
        # 使用配置中指定的激活函数名，从全局的 ACT2FN 字典中获取相应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个 GroupNorm 层，对输出的卷积结果进行分组归一化处理
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 将隐藏状态传递给卷积层进行计算
        hidden_states = self.conv(hidden_states)
        # 将卷积结果输入到 GroupNorm 层中进行分组归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 最后应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states
# 将 Speech2TextSinusoidalPositionalEmbedding 类替换为 SpeechT5SinusoidalPositionalEmbedding，适用于 SpeechT5 模型
class SpeechT5SinusoidalPositionalEmbedding(nn.Module):
    """本模块生成任意长度的正弦余弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 初始化模块，设置未使用的偏移量、嵌入维度、填充索引
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # 预生成权重矩阵
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # 自定义权重矩阵生成方法，确保正确类型和设备的参数
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 根据数理规则生成预设的权重矩阵
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 调整权重矩阵设备与类型，并更新参数
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False  # 确保权重矩阵不可训练
        self.weights.detach_()

    # 根据数学原理生成权重矩阵，定义句法和辅助常量使用
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        创造正弦余弦嵌入式权重矩阵，遵循文档中归纳的公式，生成为一种特定应用的基础矢量组。
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果 Weight 维度不兼容，进行必要的补充以避免错误
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.to(torch.get_default_dtype())

    # 无梯度权重获得器函数，实际赋予输入序列对应位置的权重嵌入，且优化使用管理
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()    
        # 创建从输入令牌 ID 转换得到的唯一位置索引列表
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )
        # 扩展权重矩阵，以覆盖可能的序列长度变化情况
        max_pos = self.padding_idx + 1 + seq_len     
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 从位置索引矩阵中按索引选择权重矩阵元素，构建输出 Tensor
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    # 创建携带历史关键值转换序列长度以适应当前输入 ID 序列的附加位置索引
    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        # 接受含填充和不需要的技术标识符的线性 ID 列表，并返回在上下文考虑中的增强系列长度
        # 分发到与输入相同设备的自定义模块结果
        """
        将非填充符号替换为它们的位置编号。位置编号从 padding_idx+1 开始。填充符号被忽略。这是基于fairseq的`utils.make_positions`修改的版本。

        Args:
            x: torch.Tensor 输入张量
        Returns:
            torch.Tensor 包含位置编号的张量
        """
        # 下面的类型转换操作被精心设计，以便同时适用于ONNX导出和XLA。
        # 创建一个张量，其中非填充位置为1，填充位置为0
        mask = input_ids.ne(padding_idx).int()
        # 计算每个位置的累积索引，并加上过去的键值长度
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        # 将计算得到的位置索引转换为长整型，并加上填充索引，以得到最终的位置编号张量
        return incremental_indices.long() + padding_idx
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding 复制代码，并将 Wav2Vec2 替换为 SpeechT5
class SpeechT5PositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个一维卷积层，用于位置编码
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 初始化权重归一化函数
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 如果启用了 DeepSpeed zero3 模式
        if is_deepspeed_zero3_enabled():
            import deepspeed
            # 使用 GatheredParameters 从第0个修饰器秩（modifier_rank=0）收集卷积层的权重
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            # 注册卷积层权重的外部参数
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 对卷积层进行权重归一化
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 初始化同填充层
        self.padding = SpeechT5SamePadLayer(config.num_conv_pos_embeddings)
        # 初始化激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将输入的 hidden_states 调换维度，使得通道维度处于第二个位置
        hidden_states = hidden_states.transpose(1, 2)

        # 通过卷积层进行位置编码计算
        hidden_states = self.conv(hidden_states)
        # 进行同填充处理
        hidden_states = self.padding(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)

        # 调换维度，使通道维度回到最后一个位置
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class SpeechT5ScaledPositionalEncoding(nn.Module):
    """
    Scaled positional encoding, see §3.2 in https://arxiv.org/abs/1809.08895
    """

    def __init__(self, dropout, dim, max_len=5000):
        # 初始化位置编码矩阵 pe
        pe = torch.zeros(max_len, dim)
        # 生成位置索引
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算位置编码的除数项
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.int64).float() * -(math.log(10000.0) / dim)))
        # 计算 sin 和 cos 的位置编码
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # 增加一维作为 batch 维度
        pe = pe.unsqueeze(0)
        super().__init__()
        # 将 pe 注册为 buffer，非持久性
        self.register_buffer("pe", pe, persistent=False)
        # 初始化 dropout 层
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        # 初始化缩放参数 alpha
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, emb):
        # 将输入的 emb 与位置编码相加，并乘以缩放参数 alpha
        emb = emb + self.alpha * self.pe[:, : emb.size(1)]
        # 应用 dropout
        emb = self.dropout(emb)
        return emb


class SpeechT5RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_length=1000):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        # 初始化相对位置编码层
        self.pe_k = torch.nn.Embedding(2 * max_length, dim)
    # 定义前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 获取序列长度
        seq_len = hidden_states.shape[1]
        # 创建一个从 0 到 seq_len-1 的整数序列，并将其转换为长整型张量，移动到与 hidden_states 相同的设备上
        pos_seq = torch.arange(0, seq_len).long().to(hidden_states.device)
        # 将 pos_seq 转置并计算每对位置之间的差值，形成位置编码矩阵
        pos_seq = pos_seq[:, None] - pos_seq[None, :]

        # 将位置编码矩阵中小于 -self.max_length 的值设为 -self.max_length
        pos_seq[pos_seq < -self.max_length] = -self.max_length
        # 将位置编码矩阵中大于等于 self.max_length 的值设为 self.max_length - 1
        pos_seq[pos_seq >= self.max_length] = self.max_length - 1
        # 将位置编码矩阵中所有值加上 self.max_length，保证所有值非负
        pos_seq = pos_seq + self.max_length

        # 使用位置编码矩阵作为索引，调用位置编码器的 pe_k 方法进行位置编码
        return self.pe_k(pos_seq)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer 复制代码，将 Wav2Vec2 替换为 SpeechT5
class SpeechT5SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 如果 num_conv_pos_embeddings 是偶数，则 num_pad_remove 为 1，否则为 0
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果 num_pad_remove 大于 0，则从 hidden_states 的末尾移除对应数量的填充
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        # 返回处理后的 hidden_states
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder 复制代码，将 Wav2Vec2 替换为 SpeechT5
class SpeechT5FeatureEncoder(nn.Module):
    """从原始音频波形构建特征"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            # 如果特征提取使用 group normalization，则创建相应的卷积层列表
            conv_layers = [SpeechT5GroupNormConvLayer(config, layer_id=0)] + [
                SpeechT5NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果特征提取使用 layer normalization，则创建相应的卷积层列表
            conv_layers = [
                SpeechT5LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 抛出值错误，提示 config.feat_extract_norm 的值必须是 'group' 或 'layer'
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        # 将卷积层列表转换为 ModuleList
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        # 冻结模型参数，设置 _requires_grad 为 False
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        # 将输入的音频值扩展一个维度
        hidden_states = input_values[:, None]

        # 如果模型正在训练并且需要梯度，设置 hidden_states 的梯度追踪为 True
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层，依次对 hidden_states 进行卷积操作
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 如果使用梯度检查点和需要梯度且在训练中，则调用梯度检查点函数对 conv_layer 进行处理
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则，直接调用 conv_layer 对 hidden_states 进行卷积操作
                hidden_states = conv_layer(hidden_states)

        # 返回处理后的 hidden_states
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection 复制代码，将 Wav2Vec2 替换为 SpeechT5
class SpeechT5FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 layer normalization 初始化 layer_norm，设置 eps 为 config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 使用线性层初始化 projection，将卷积的最后一维映射到 hidden_size
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 使用 dropout 初始化 dropout，设置 dropout 概率为 config.feat_proj_dropout
        self.dropout = nn.Dropout(config.feat_proj_dropout)
    # 定义一个前向传播方法，用于处理隐藏状态
    def forward(self, hidden_states):
        # 对隐藏状态进行层归一化处理，用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态投影到新的空间
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态应用丢弃（dropout）操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态以及归一化前的隐藏状态
        return hidden_states, norm_hidden_states
# 定义一个名为 SpeechT5SpeechEncoderPrenet 的类，继承自 nn.Module
class SpeechT5SpeechEncoderPrenet(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()
        # 将传入的 config 参数保存到实例的 config 属性中
        self.config = config
        # 创建一个 SpeechT5FeatureEncoder 类的实例，并保存到 feature_encoder 属性中
        self.feature_encoder = SpeechT5FeatureEncoder(config)
        # 创建一个 SpeechT5FeatureProjection 类的实例，并保存到 feature_projection 属性中
        self.feature_projection = SpeechT5FeatureProjection(config)

        # 只有当 config 中 mask_time_prob 大于 0.0 或者 config 中 mask_feature_prob 大于 0.0 时，才需要创建 masked_spec_embed 属性
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            # 创建一个随机初始化的可学习参数（nn.Parameter），并保存到 masked_spec_embed 属性中
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 创建一个 SpeechT5PositionalConvEmbedding 类的实例，并保存到 pos_conv_embed 属性中
        self.pos_conv_embed = SpeechT5PositionalConvEmbedding(config)
        # 创建一个 SpeechT5SinusoidalPositionalEmbedding 类的实例，并保存到 pos_sinusoidal_embed 属性中
        self.pos_sinusoidal_embed = SpeechT5SinusoidalPositionalEmbedding(
            config.max_speech_positions + config.pad_token_id + 1,
            config.hidden_size,
            config.pad_token_id,
        )

    # 冻结 feature_encoder 中的参数
    def freeze_feature_encoder(self):
        self.feature_encoder._freeze_parameters()

    # 前向传播方法
    def forward(
        self,
        input_values: torch.Tensor,  # 输入的特征张量
        attention_mask: Optional[torch.LongTensor] = None,  # 可选的注意力掩码张量
        mask_time_indices: Optional[torch.FloatTensor] = None,  # 可选的时间掩码张量
    ):
        # 使用 feature_encoder 对输入的特征进行编码
        extract_features = self.feature_encoder(input_values)
        # 将编码后的特征张量进行维度转置
        extract_features = extract_features.transpose(1, 2)

        # 如果 attention_mask 不为 None，则计算与特征向量对应的减少的 attention_mask
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1],  # 提取特征后的特征长度
                attention_mask,
            )

        # 使用 feature_projection 对特征进行投影，并获取隐藏状态和投影后的特征
        hidden_states, extract_features = self.feature_projection(extract_features)
        
        # 对隐藏状态进行掩码处理，根据传入的 mask_time_indices 和 attention_mask
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 使用 pos_conv_embed 对隐藏状态进行位置卷积嵌入
        positional_conv_embedding = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + positional_conv_embedding

        # 如果 attention_mask 不为 None，则创建一个 padding_mask 张量，用于表示 padding 的位置
        if attention_mask is not None:
            padding_mask = attention_mask.ne(1).long()
        else:
            # 否则创建一个全零张量，形状与隐藏状态的前两个维度相同，用于表示没有 padding 的位置
            padding_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)

        # 使用 pos_sinusoidal_embed 对隐藏状态进行位置正弦嵌入
        positional_sinusoidal_embeddings = self.pos_sinusoidal_embed(padding_mask)
        hidden_states = hidden_states + positional_sinusoidal_embeddings

        # 返回隐藏状态和注意力掩码（如果有的话）
        return hidden_states, attention_mask

    # 从 transformers.models.unispeech.modeling_unispeech.UniSpeechPreTrainedModel._get_feature_vector_attention_mask 复制的方法
    # 计算非填充部分的长度，即每个序列的有效长度
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
    # 根据有效长度计算输出长度，并转换为长整型
    output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)
    # 获取批量大小
    batch_size = attention_mask.shape[0]

    # 创建一个全零的注意力掩码张量，形状为(batch_size, feature_vector_length)，类型和设备与原始注意力掩码相同
    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )
    # 将每个序列的有效长度对应位置设为1，表示需要关注的部分
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
    # 反转注意力掩码张量，进行累积和布尔化处理，以确保输出长度位置之前的所有值都被关注
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    # 返回处理后的注意力掩码张量
    return attention_mask

```  
    # 从UniSpeechPreTrainedModel中复制的函数，用于计算卷积层的输出长度
    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        计算卷积层的输出长度
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的1D卷积层输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历配置中的卷积核大小和步长，依次计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

```  
    # 从Wav2Vec2Model中复制的函数，用于对隐藏状态进行掩码处理
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        # 检查配置中是否允许应用 SpecAugment，如果不允许，则直接返回隐藏状态
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        # 获取隐藏状态的尺寸信息：批大小、序列长度和隐藏大小
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # 如果提供了 mask_time_indices，则沿时间轴应用 SpecAugment
            # 使用给定的 mask_time_indices 对隐藏状态进行 SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 如果未提供 mask_time_indices，则根据配置和训练状态生成并应用时间轴上的 SpecAugment
            # 根据配置生成时间轴上的掩码索引
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
            # 如果配置允许并且在训练模式下，生成并应用特征轴上的 SpecAugment
            # 根据配置生成特征轴上的掩码索引
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            # 将特征轴上的掩码扩展到整个序列上，并将相应位置的隐藏状态置零
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        # 返回经过 SpecAugment 处理后的隐藏状态
        return hidden_states
class SpeechT5SpeechDecoderPrenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 定义多层线性层组成的神经网络
        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    config.num_mel_bins if i == 0 else config.speech_decoder_prenet_units,
                    config.speech_decoder_prenet_units,
                )
                for i in range(config.speech_decoder_prenet_layers)
            ]
        )

        # 定义最终的线性层
        self.final_layer = nn.Linear(config.speech_decoder_prenet_units, config.hidden_size)
        
        # 编码位置信息的模块
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            config.positional_dropout,
            config.hidden_size,
            config.max_speech_positions,
        )
        
        # 处理说话者嵌入的线性层
        self.speaker_embeds_layer = nn.Linear(config.speaker_embedding_dim + config.hidden_size, config.hidden_size)

    # 实现一致性 dropout 的方法
    def _consistent_dropout(self, inputs_embeds, p):
        mask = torch.bernoulli(inputs_embeds[0], p=p)
        all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
        return torch.where(all_masks == 1, inputs_embeds, 0) * 1 / (1 - p)

    # 前向传播方法
    def forward(
        self,
        input_values: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
    ):
        # 在评估时也始终应用 dropout，参见 https://arxiv.org/abs/1712.05884 §2.2。

        inputs_embeds = input_values
        
        # 对每一层线性层应用 ReLU 激活函数和一致性 dropout
        for layer in self.layers:
            inputs_embeds = nn.functional.relu(layer(inputs_embeds))
            inputs_embeds = self._consistent_dropout(inputs_embeds, self.config.speech_decoder_prenet_dropout)

        # 应用最终的线性层
        inputs_embeds = self.final_layer(inputs_embeds)
        
        # 编码位置信息
        inputs_embeds = self.encode_positions(inputs_embeds)

        # 如果提供了说话者嵌入，则将其归一化并拼接到输入中，然后再应用 ReLU 激活函数
        if speaker_embeddings is not None:
            speaker_embeddings = nn.functional.normalize(speaker_embeddings)
            speaker_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, inputs_embeds.size(1), -1)
            inputs_embeds = torch.cat([inputs_embeds, speaker_embeddings], dim=-1)
            inputs_embeds = nn.functional.relu(self.speaker_embeds_layer(inputs_embeds))

        return inputs_embeds
    # 初始化函数，用于构造一个卷积层和相关的批归一化、激活函数、dropout等组件
    def __init__(self, config, layer_id=0):
        # 调用父类的初始化函数
        super().__init__()

        # 根据给定的层编号确定输入卷积层的维度
        if layer_id == 0:
            in_conv_dim = config.num_mel_bins
        else:
            in_conv_dim = config.speech_decoder_postnet_units

        # 根据给定的层编号确定输出卷积层的维度
        if layer_id == config.speech_decoder_postnet_layers - 1:
            out_conv_dim = config.num_mel_bins
        else:
            out_conv_dim = config.speech_decoder_postnet_units

        # 创建一个一维卷积层对象
        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.speech_decoder_postnet_kernel,
            stride=1,
            padding=(config.speech_decoder_postnet_kernel - 1) // 2,
            bias=False,
        )
        # 创建一个一维批归一化层对象
        self.batch_norm = nn.BatchNorm1d(out_conv_dim)

        # 根据层编号选择是否使用双曲正切作为激活函数
        if layer_id < config.speech_decoder_postnet_layers - 1:
            self.activation = nn.Tanh()
        else:
            self.activation = None

        # 创建一个dropout层对象，用于在训练过程中随机置零输入张量的部分元素
        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

    # 前向传播函数，接收输入的隐藏状态张量，经过卷积、批归一化、激活和dropout处理后返回处理后的隐藏状态张量
    def forward(self, hidden_states):
        # 使用卷积层处理输入张量
        hidden_states = self.conv(hidden_states)
        # 使用批归一化层处理卷积后的张量
        hidden_states = self.batch_norm(hidden_states)
        # 如果存在激活函数，则应用激活函数到批归一化后的张量
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        # 应用dropout到处理后的张量
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的张量作为本层的输出
        return hidden_states
class SpeechT5SpeechDecoderPostnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 初始化模型配置

        # 定义输出特征和概率的线性层
        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)
        self.prob_out = nn.Linear(config.hidden_size, config.reduction_factor)

        # 创建一系列的语音解码后处理层
        self.layers = nn.ModuleList(
            [SpeechT5BatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)]
        )

    def forward(self, hidden_states: torch.Tensor):
        # 计算特征输出，将其形状变换为(batch_size, seq_len, num_mel_bins)
        outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, self.config.num_mel_bins)
        # 经过后处理网络处理特征输出
        outputs_after_postnet = self.postnet(outputs_before_postnet)
        # 计算最终的分类概率
        logits = self.prob_out(hidden_states).view(hidden_states.size(0), -1)
        return outputs_before_postnet, outputs_after_postnet, logits

    def postnet(self, hidden_states: torch.Tensor):
        # 转置隐藏状态以适应后处理层的输入格式
        layer_output = hidden_states.transpose(1, 2)
        # 通过每个后处理层处理隐藏状态
        for layer in self.layers:
            layer_output = layer(layer_output)
        # 将处理后的输出转置回原始形状，并与原始隐藏状态相加
        return hidden_states + layer_output.transpose(1, 2)


class SpeechT5TextEncoderPrenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 词嵌入层，用于将词汇索引映射为隐藏表示
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 编码位置信息的扩展
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            config.positional_dropout,
            config.hidden_size,
            config.max_text_positions,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids: torch.Tensor):
        # 获取输入词嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        # 添加位置编码并返回编码结果
        inputs_embeds = self.encode_positions(inputs_embeds)
        return inputs_embeds


class SpeechT5TextDecoderPrenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 随机失活层，用于减少过拟合
        self.dropout = nn.Dropout(config.positional_dropout)
        # 嵌入缩放因子，用于调整嵌入的标度
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 词嵌入层，将词汇索引映射为隐藏表示
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 嵌入位置信息的正弦位置编码
        self.embed_positions = SpeechT5SinusoidalPositionalEmbedding(
            config.max_text_positions + config.pad_token_id + 1,
            config.hidden_size,
            config.pad_token_id,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        ):
        # 如果给定了输入的 token IDs
        if input_ids is not None:
            # 获取输入 token IDs 的形状
            input_shape = input_ids.size()
            # 将输入 token IDs 展平成二维张量，保留最后一个维度不变
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            # 如果没有提供输入 token IDs，则抛出数值错误
            raise ValueError("You have to specify `decoder_input_ids`")

        # 如果提供了过去的键值对，则计算过去的键的长度
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # 使用 embed_positions 方法计算位置嵌入
        positions = self.embed_positions(input_ids, past_key_values_length)

        # 使用 embed_tokens 方法获取输入 token IDs 的嵌入表示，并乘以 embed_scale
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # 将位置嵌入加到输入的嵌入表示中
        inputs_embeds += positions
        # 对输入的嵌入表示进行 dropout
        inputs_embeds = self.dropout(inputs_embeds)

        # 返回处理后的输入嵌入表示和注意力掩码
        return inputs_embeds, attention_mask
class SpeechT5TextDecoderPostnet(nn.Module):
    # 定义一个用于T5文本解码的后处理网络模块
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 线性层，将隐藏状态映射到词汇表大小的输出空间，没有偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        # 前向传播函数，接收隐藏状态并返回线性层的输出
        return self.lm_head(hidden_states)

    def get_output_embeddings(self):
        # 返回当前的输出嵌入层（即lm_head）
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出嵌入层
        self.lm_head = new_embeddings


class SpeechT5Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with relative position bias (see
    https://aclanthology.org/N18-2074.pdf)
    """
    # 定义T5模型中的注意力机制模块，支持相对位置偏置
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化查询、键、值的线性投影层和输出投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入张量重塑为适合多头注意力计算的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 实现注意力模块的前向传播
        # 接收隐藏状态、键值状态、注意力掩码等，返回加权后的输出及可能的注意力分布
        raise NotImplementedError


class SpeechT5FeedForward(nn.Module):
    # 定义一个T5模型中的前馈神经网络模块
    def __init__(self, config, intermediate_size):
        super().__init__()
        # 使用配置中的激活函数的丢弃层
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 中间层的线性层和激活函数
        self.intermediate_dense = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 输出层的线性层和丢弃层
        self.output_dense = nn.Linear(intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)
    # 定义一个前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 通过中间的稠密层处理隐藏状态
        hidden_states = self.intermediate_dense(hidden_states)
        # 应用中间激活函数到处理后的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对处理后的隐藏状态进行中间的丢弃操作

        # 通过输出稠密层处理隐藏状态
        hidden_states = self.output_dense(hidden_states)
        # 对处理后的隐藏状态进行输出丢弃操作
        hidden_states = self.output_dropout(hidden_states)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
class SpeechT5EncoderLayer(nn.Module):
    def __init__(self, config: SpeechT5Config):
        super().__init__()
        # 初始化自注意力层，使用配置中的隐藏层大小、注意力头数和注意力丢弃率，作为编码器
        self.attention = SpeechT5Attention(
            embed_dim=config.hidden_size,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 定义丢弃层，用于隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 定义层归一化，使用配置中的隐藏层大小和层归一化 epsilon 值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义前馈网络，使用配置和编码器的 FFN 维度
        self.feed_forward = SpeechT5FeedForward(config, config.encoder_ffn_dim)
        # 定义最终层归一化，使用配置中的隐藏层大小和层归一化 epsilon 值
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                输入层的隐藏状态，形状为 `(batch, seq_len, hidden_size)`
            attention_mask (`torch.FloatTensor`):
                注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`，其中填充元素用极大负值表示
            layer_head_mask (`torch.FloatTensor`):
                给定层中注意力头的掩码，形状为 `(config.encoder_attention_heads,)`
            position_bias (`torch.FloatTensor`):
                相对位置嵌入，形状为 `(seq_len, seq_len, hidden_size // encoder_attention_heads)`
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 以获取更多细节。
        """
        # 保存残差连接
        residual = hidden_states
        # 使用自注意力层处理隐藏状态，返回处理后的隐藏状态、注意力权重以及（如果输出）所有层的注意力
        hidden_states, attn_weights, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )

        # 应用丢弃层
        hidden_states = self.dropout(hidden_states)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 应用层归一化
        hidden_states = self.layer_norm(hidden_states)
        # 使用前馈网络处理隐藏状态
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SpeechT5DecoderLayer(nn.Module):
    # 这里需要继续实现 SpeechT5DecoderLayer 类的定义，但不在此处添加注释
    # 初始化函数，用于创建一个 SpeechT5DecoderLayer 对象
    def __init__(self, config: SpeechT5Config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建自注意力机制对象 SpeechT5Attention
        self.self_attn = SpeechT5Attention(
            embed_dim=config.hidden_size,                # 设置注意力机制的输入维度
            num_heads=config.decoder_attention_heads,    # 设置注意力头的数量
            dropout=config.attention_dropout,            # 设置注意力机制的dropout率
            is_decoder=True,                             # 声明这是一个decoder层的自注意力机制
        )
        # 创建dropout层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建LayerNorm层，用于自注意力机制后的归一化
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建编码器注意力机制对象 SpeechT5Attention
        self.encoder_attn = SpeechT5Attention(
            config.hidden_size,                           # 设置编码器注意力机制的输入维度
            config.decoder_attention_heads,               # 设置注意力头的数量
            dropout=config.attention_dropout,             # 设置注意力机制的dropout率
            is_decoder=True,                              # 声明这是一个decoder层的编码器注意力机制
        )
        # 创建LayerNorm层，用于编码器注意力机制后的归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建前向传播（Feed Forward）层对象 SpeechT5FeedForward
        self.feed_forward = SpeechT5FeedForward(config, config.decoder_ffn_dim)
        # 创建LayerNorm层，用于前向传播层后的归一化
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定该类使用的配置类
    config_class = SpeechT5Config
    # 指定基础模型前缀名称
    base_model_prefix = "speecht5"
    # 主要输入名称
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是 SpeechT5PositionalConvEmbedding 类型
        if isinstance(module, SpeechT5PositionalConvEmbedding):
            # 使用正态分布初始化卷积层权重
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积层偏置初始化为常数0
            nn.init.constant_(module.conv.bias, 0)
        # 如果模块是 SpeechT5FeatureProjection 类型
        elif isinstance(module, SpeechT5FeatureProjection):
            # 计算初始化范围
            k = math.sqrt(1 / module.projection.in_features)
            # 使用均匀分布初始化投影层权重和偏置
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果模块是 nn.Linear 类型
        elif isinstance(module, nn.Linear):
            # 使用正态分布初始化全连接层权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，将偏置初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 nn.LayerNorm 或 nn.GroupNorm 类型
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将归一化层的偏置初始化为0
            module.bias.data.zero_()
            # 将归一化层的权重初始化为1
            module.weight.data.fill_(1.0)
        # 如果模块是 nn.Conv1d 类型
        elif isinstance(module, nn.Conv1d):
            # 使用 Kaiming 正态分布初始化卷积层权重
            nn.init.kaiming_normal_(module.weight)
            # 如果有偏置，使用均匀分布初始化卷积层偏置
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将填充索引对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    # 定义神经网络模型的前向传播函数
    def forward(
        self,
        # 输入隐藏状态，通常是一个张量
        hidden_states: torch.FloatTensor,
        # 注意力掩码，用于指定哪些位置的输入应该被忽略
        attention_mask: Optional[torch.Tensor] = None,
        # 头部掩码，用于指定哪些注意力头部应该被忽略
        head_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出所有隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回一个字典格式的结果
        return_dict: Optional[bool] = None,
class SpeechT5EncoderWithSpeechPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Encoder that applies SpeechT5SpeechEncoderPrenet to convert the audio waveform data to
    hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        # 实例化一个 SpeechT5SpeechEncoderPrenet 对象，用于处理音频波形数据
        self.prenet = SpeechT5SpeechEncoderPrenet(config)
        # 实例化一个 SpeechT5Encoder 对象，用于编码处理后的特征
        self.wrapped_encoder = SpeechT5Encoder(config)

        # 调用后初始化方法，用于初始化权重和进行最终处理
        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 使用 SpeechT5SpeechEncoderPrenet 处理输入的音频数据，生成隐藏状态和注意力掩码
        hidden_states, attention_mask = self.prenet(input_values, attention_mask)

        # 将处理后的特征传递给 wrapped_encoder 进行进一步编码
        outputs = self.wrapped_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class SpeechT5EncoderWithTextPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Encoder that applies SpeechT5TextEncoderPrenet to convert the input_ids to hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        # 实例化一个 SpeechT5TextEncoderPrenet 对象，用于处理输入的文本特征
        self.prenet = SpeechT5TextEncoderPrenet(config)
        # 实例化一个 SpeechT5Encoder 对象，用于编码处理后的特征
        self.wrapped_encoder = SpeechT5Encoder(config)

        # 调用后初始化方法，用于初始化权重和进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取预处理器（prenet）的输入嵌入
        return self.prenet.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 设置预处理器（prenet）的输入嵌入
        self.prenet.set_input_embeddings(value)

    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 使用 SpeechT5TextEncoderPrenet 处理输入的文本数据，生成隐藏状态
        hidden_states = self.prenet(input_values)

        # 将处理后的特征传递给 wrapped_encoder 进行进一步编码
        outputs = self.wrapped_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class SpeechT5EncoderWithoutPrenet(SpeechT5PreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when used in combination with
    [`SpeechT5Model`].
    """
    # 初始化方法，接收一个 SpeechT5Config 类型的配置参数
    def __init__(self, config: SpeechT5Config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 创建一个 SpeechT5Encoder 的实例，并将其赋值给 self.wrapped_encoder
        self.wrapped_encoder = SpeechT5Encoder(config)

        # 执行后续的初始化操作和权重设置
        self.post_init()

    # 前向传播方法，接收多个输入参数并返回一个联合类型的值
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 调用 self.wrapped_encoder 的前向传播方法，传递所有输入参数，并返回其输出
        return self.wrapped_encoder(
            hidden_states=input_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
class SpeechT5Decoder(SpeechT5PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SpeechT5DecoderLayer`].
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.layerdrop = config.decoder_layerdrop  # 设置层间丢弃率，从配置中获取

        # 创建多个解码层，并用列表包装成模块列表
        self.layers = nn.ModuleList([SpeechT5DecoderLayer(config) for _ in range(config.decoder_layers)])

        self.gradient_checkpointing = False  # 梯度检查点设置为假

        # 初始化权重并进行最终处理
        self.post_init()

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



class SpeechT5DecoderWithSpeechPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Decoder that applies SpeechT5SpeechDecoderPrenet to convert log-mel filterbanks to hidden
    features.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        # 初始化语音预网络
        self.prenet = SpeechT5SpeechDecoderPrenet(config)
        # 创建一个SpeechT5Decoder对象作为包装解码器
        self.wrapped_decoder = SpeechT5Decoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeddings: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数的返回类型，表示返回的结果是一个元组或者BaseModelOutputWithPastAndCrossAttentions类的实例
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # 使用prenet方法处理输入数值和说话者嵌入，生成解码器的隐藏状态
        decoder_hidden_states = self.prenet(input_values, speaker_embeddings)

        # 调用wrapped_decoder进行解码器的正向传播
        outputs = self.wrapped_decoder(
            # 传递解码器的隐藏状态
            hidden_states=decoder_hidden_states,
            # 传递注意力掩码
            attention_mask=attention_mask,
            # 传递编码器的隐藏状态
            encoder_hidden_states=encoder_hidden_states,
            # 传递编码器的注意力掩码
            encoder_attention_mask=encoder_attention_mask,
            # 传递头部掩码
            head_mask=head_mask,
            # 传递交叉注意力头部掩码
            cross_attn_head_mask=cross_attn_head_mask,
            # 传递过去的键值对
            past_key_values=past_key_values,
            # 指定是否使用缓存
            use_cache=use_cache,
            # 指定是否输出注意力权重
            output_attentions=output_attentions,
            # 指定是否输出隐藏状态
            output_hidden_states=output_hidden_states,
            # 指定是否返回字典格式的输出
            return_dict=return_dict,
        )

        # 返回wrapped_decoder的输出结果
        return outputs
# 继承自 SpeechT5PreTrainedModel 的 SpeechT5DecoderWithTextPrenet 类，包装了 SpeechT5Decoder，并应用 SpeechT5TextDecoderPrenet 将输入标记转换为隐藏特征。
class SpeechT5DecoderWithTextPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Decoder that applies SpeechT5TextDecoderPrenet to convert input tokens to hidden features.
    """

    # 初始化方法，接收一个 SpeechT5Config 类型的参数 config
    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        # 创建 SpeechT5TextDecoderPrenet 实例并赋值给 self.prenet 属性
        self.prenet = SpeechT5TextDecoderPrenet(config)
        # 创建 SpeechT5Decoder 实例并赋值给 self.wrapped_decoder 属性
        self.wrapped_decoder = SpeechT5Decoder(config)

        # 调用自定义方法 post_init，用于初始化权重和应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法，委托给 self.prenet 的 get_input_embeddings 方法
    def get_input_embeddings(self):
        return self.prenet.get_input_embeddings()

    # 设置输入嵌入层的方法，委托给 self.prenet 的 set_input_embeddings 方法
    def set_input_embeddings(self, value):
        self.prenet.set_input_embeddings(value)

    # 前向传播方法，接收多个输入参数并返回模型输出
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # 使用 self.prenet 处理输入值和注意力掩码，返回解码器隐藏状态和注意力掩码
        decoder_hidden_states, attention_mask = self.prenet(input_values, attention_mask, past_key_values)

        # 调用 self.wrapped_decoder 的 forward 方法，传入解码器隐藏状态和注意力掩码等参数，获取模型输出
        outputs = self.wrapped_decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回模型输出
        return outputs


# 继承自 SpeechT5PreTrainedModel 的 SpeechT5DecoderWithoutPrenet 类，作为辅助类在与 SpeechT5Model 结合使用时正确加载预训练检查点。
class SpeechT5DecoderWithoutPrenet(SpeechT5PreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when used in combination with
    [`SpeechT5Model`].
    """

    # 初始化方法，接收一个 SpeechT5Config 类型的参数 config
    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        # 创建 SpeechT5Decoder 实例并赋值给 self.wrapped_decoder 属性
        self.wrapped_decoder = SpeechT5Decoder(config)

        # 调用自定义方法 post_init，用于初始化权重和应用最终处理
        self.post_init()
    # 定义一个方法 forward，用于模型的前向传播
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,  # 输入值，可以是浮点型张量，可选
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码，可以是长整型张量，可选
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器隐藏状态，可以是浮点型张量，可选
        encoder_attention_mask: Optional[torch.LongTensor] = None,  # 编码器注意力掩码，可以是长整型张量，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可以是张量，可选
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码，可以是张量，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对列表，可以是浮点型张量列表，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以是布尔值，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可以是布尔值，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以是布尔值，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式结果，可以是布尔值，可选
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # 调用 wrapped_decoder 对象的方法，进行解码器的包装前向传播
        outputs = self.wrapped_decoder(
            hidden_states=input_values,  # 输入值作为隐藏状态传入
            attention_mask=attention_mask,  # 注意力掩码传入
            encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态传入
            encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码传入
            head_mask=head_mask,  # 头部掩码传入
            cross_attn_head_mask=cross_attn_head_mask,  # 交叉注意力头部掩码传入
            past_key_values=past_key_values,  # 过去的键值对列表传入
            use_cache=use_cache,  # 是否使用缓存传入
            output_attentions=output_attentions,  # 是否输出注意力传入
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态传入
            return_dict=return_dict,  # 是否返回字典格式结果传入
        )
        # 返回 wrapped_decoder 方法的输出结果
        return outputs
class SpeechT5GuidedMultiheadAttentionLoss(nn.Module):
    """
    Guided attention loss from the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional
    Networks with Guided Attention](https://arxiv.org/abs/1710.08969), adapted for multi-head attention.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.sigma = config.guided_attention_loss_sigma  # 初始化 sigma 参数
        self.scale = config.guided_attention_loss_scale  # 初始化 scale 参数

    def forward(
        self, attentions: torch.FloatTensor, input_masks: torch.BoolTensor, output_masks: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute the attention loss.

        Args:
            attentions (`torch.FloatTensor` of shape `(batch_size, layers * heads, output_sequence_length, input_sequence_length)`):
                Batch of multi-head attention weights
            input_masks (`torch.BoolTensor` of shape `(batch_size, input_sequence_length)`):
                Input attention mask as booleans.
            output_masks (`torch.BoolTensor` of shape `(batch_size, output_sequence_length)`):
                Target attention mask as booleans.

        Returns:
            `torch.Tensor` with the loss value
        """
        guided_attn_masks = self._make_guided_attention_masks(input_masks, output_masks, attentions.device)
        masks = output_masks.unsqueeze(-1) & input_masks.unsqueeze(-2)  # 创建掩码，用于选择有效的注意力权重
        masks = masks.to(attentions.device).unsqueeze(1)  # 将掩码移到与注意力权重相同的设备上，并扩展维度以匹配注意力权重的形状

        losses = guided_attn_masks * attentions  # 计算引导注意力损失
        loss = torch.mean(losses.masked_select(masks))  # 使用掩码选择有效区域内的损失值，并计算平均损失
        return self.scale * loss  # 返回经过缩放的损失值

    def _make_guided_attention_masks(self, input_masks, output_masks, device):
        input_lengths = input_masks.sum(-1)  # 计算输入掩码的有效长度
        output_lengths = output_masks.sum(-1)  # 计算输出掩码的有效长度

        guided_attn_masks = torch.zeros((len(input_masks), output_masks.shape[1], input_masks.shape[1]), device=device)  # 初始化引导注意力掩码

        for idx, (ilen, olen) in enumerate(zip(input_lengths, output_lengths)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma, device)  # 生成并填充每个样本的引导注意力掩码

        return guided_attn_masks.unsqueeze(1)  # 扩展维度以匹配注意力权重的形状

    @staticmethod
    def _make_guided_attention_mask(input_length, output_length, sigma, device):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(input_length, device=device),  # 创建输入序列长度的网格
            torch.arange(output_length, device=device),  # 创建输出序列长度的网格
            indexing="xy",
        )
        grid_x = grid_x.float() / output_length  # 标准化输出网格
        grid_y = grid_y.float() / input_length  # 标准化输入网格
        return 1.0 - torch.exp(-((grid_y - grid_x) ** 2) / (2 * (sigma**2)))  # 生成引导注意力掩码
    # 初始化方法，接受一个 SpeechT5Config 类型的参数 config
    def __init__(self, config: SpeechT5Config):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置文件设置是否使用引导注意力损失
        self.use_guided_attention_loss = config.use_guided_attention_loss
        # 设置引导注意力损失的头数
        self.guided_attention_loss_num_heads = config.guided_attention_loss_num_heads
        # 设置减少因子
        self.reduction_factor = config.reduction_factor

        # 定义 L1 损失函数
        self.l1_criterion = L1Loss()
        # 定义带权重的二元交叉熵损失函数
        self.bce_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))

        # 如果使用引导注意力损失，则初始化 SpeechT5GuidedMultiheadAttentionLoss 类
        if self.use_guided_attention_loss:
            self.attn_criterion = SpeechT5GuidedMultiheadAttentionLoss(config)

    # 前向传播方法，接受多个张量作为输入并返回一个张量
    def forward(
        self,
        attention_mask: torch.LongTensor,
        outputs_before_postnet: torch.FloatTensor,
        outputs_after_postnet: torch.FloatTensor,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        cross_attentions: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 创建一个填充掩码，用于过滤掉填充部分
        padding_mask = labels != -100.0

        # 根据填充掩码选择有效的标签数据
        labels = labels.masked_select(padding_mask)
        outputs_before_postnet = outputs_before_postnet.masked_select(padding_mask)
        outputs_after_postnet = outputs_after_postnet.masked_select(padding_mask)

        # 计算声谱图损失，包括前后处理网络输出的 L1 损失
        l1_loss = self.l1_criterion(outputs_after_postnet, labels) + self.l1_criterion(outputs_before_postnet, labels)

        # 根据填充掩码构建停止标签
        masks = padding_mask[:, :, 0]
        stop_labels = torch.cat([~masks * 1.0, torch.ones(masks.size(0), 1).to(masks.device)], dim=1)
        stop_labels = stop_labels[:, 1:].masked_select(masks)
        logits = logits.masked_select(masks)

        # 计算停止令牌损失，使用带权重的二元交叉熵损失函数
        bce_loss = self.bce_criterion(logits, stop_labels)

        # 组合所有损失
        loss = l1_loss + bce_loss

        # 如果使用引导注意力损失，则计算该损失
        if self.use_guided_attention_loss:
            # 将所有交叉注意力的头拼接在一起
            attn = torch.cat([x[:, : self.guided_attention_loss_num_heads] for x in cross_attentions], dim=1)
            # 获取输入和输出的掩码
            input_masks = attention_mask == 1
            output_masks = padding_mask[:, :, 0]
            # 如果有减少因子，则按照减少因子对输出掩码进行调整
            if self.reduction_factor > 1:
                output_masks = output_masks[:, self.reduction_factor - 1 :: self.reduction_factor]
            # 计算引导注意力损失
            attn_loss = self.attn_criterion(attn, input_masks, output_masks)
            # 将引导注意力损失添加到总损失中
            loss += attn_loss

        # 返回最终的损失张量
        return loss
# 定义字符串常量，用于描述 SpeechT5Model 类的文档字符串起始部分，包含继承关系及通用模型方法说明
SPEECHT5_BASE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        encoder ([`SpeechT5EncoderWithSpeechPrenet`] or [`SpeechT5EncoderWithTextPrenet`] or `None`):
            The Transformer encoder module that applies the appropiate speech or text encoder prenet. If `None`,
            [`SpeechT5EncoderWithoutPrenet`] will be used and the `input_values` are assumed to be hidden states.
        decoder ([`SpeechT5DecoderWithSpeechPrenet`] or [`SpeechT5DecoderWithTextPrenet`] or `None`):
            The Transformer decoder module that applies the appropiate speech or text decoder prenet. If `None`,
            [`SpeechT5DecoderWithoutPrenet`] will be used and the `decoder_input_values` are assumed to be hidden
            states.
"""

# 定义字符串常量，用于描述 SpeechT5Model 类的文档字符串起始部分，缩短版，不包含特定的 pre- 或 post-nets 描述
SPEECHT5_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义字符串常量，用于描述 SpeechT5Model 类的输入文档字符串，目前为空字符串
SPEECHT5_INPUTS_DOCSTRING = r"""
"""

# 使用装饰器 @add_start_docstrings 添加文档字符串到 SpeechT5Model 类，描述其作为裸型（bare）Encoder-Decoder 模型输出原始隐藏状态的特性
@add_start_docstrings(
    "The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.",
    SPEECHT5_BASE_START_DOCSTRING,
)
class SpeechT5Model(SpeechT5PreTrainedModel):
    def __init__(
        self,
        config: SpeechT5Config,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ):
        # 调用父类的初始化方法，并传入配置参数
        super().__init__(config)
        # 存储传入的配置参数
        self.config = config
        # 如果未提供编码器，则使用默认的 SpeechT5EncoderWithoutPrenet
        self.encoder = SpeechT5EncoderWithoutPrenet(config) if encoder is None else encoder
        # 如果未提供解码器，则使用默认的 SpeechT5DecoderWithoutPrenet
        self.decoder = SpeechT5DecoderWithoutPrenet(config) if decoder is None else decoder

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 如果编码器是 SpeechT5EncoderWithTextPrenet 类型，则获取其输入嵌入
        if isinstance(self.encoder, SpeechT5EncoderWithTextPrenet):
            return self.encoder.get_input_embeddings()
        # 如果解码器是 SpeechT5DecoderWithTextPrenet 类型，则获取其输入嵌入
        if isinstance(self.decoder, SpeechT5DecoderWithTextPrenet):
            return self.decoder.get_input_embeddings()
        # 否则返回空
        return None

    def set_input_embeddings(self, value):
        # 如果编码器是 SpeechT5EncoderWithTextPrenet 类型，则设置其输入嵌入
        if isinstance(self.encoder, SpeechT5EncoderWithTextPrenet):
            self.encoder.set_input_embeddings(value)
        # 如果解码器是 SpeechT5DecoderWithTextPrenet 类型，则设置其输入嵌入
        if isinstance(self.decoder, SpeechT5DecoderWithTextPrenet):
            self.decoder.set_input_embeddings(value)

    def get_encoder(self):
        # 返回编码器对象
        return self.encoder

    def get_decoder(self):
        # 返回解码器对象
        return self.decoder

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 如果编码器是 SpeechT5EncoderWithSpeechPrenet 类型，则冻结其特征编码器，使其梯度不计算
        if isinstance(self.encoder, SpeechT5EncoderWithSpeechPrenet):
            self.encoder.prenet.freeze_feature_encoder()

    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器添加文档字符串，描述了SpeechT5模型的功能和起始文档字符串
@add_start_docstrings(
    """SpeechT5 Model with a speech encoder and a text decoder.""",
    SPEECHT5_START_DOCSTRING,
)
# SpeechT5ForSpeechToText类继承自SpeechT5PreTrainedModel类
class SpeechT5ForSpeechToText(SpeechT5PreTrainedModel):
    # 定义了与文本解码器的权重绑定的关键字列表
    _tied_weights_keys = ["text_decoder_postnet.lm_head.weight"]

    # 初始化方法，接受一个SpeechT5Config类型的参数config
    def __init__(self, config: SpeechT5Config):
        # 调用父类SpeechT5PreTrainedModel的初始化方法
        super().__init__(config)

        # 如果配置中未指定词汇表大小，则引发值错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that does not define the"
                " vocabulary size of the language model head. Please instantiate the model as follows:"
                " `SpeechT5ForSpeechToText.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of"
                " your model's configuration."
            )

        # 创建语音编码器和文本解码器实例
        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        text_decoder = SpeechT5DecoderWithTextPrenet(config)
        # 创建SpeechT5Model实例，整合语音编码器和文本解码器
        self.speecht5 = SpeechT5Model(config, speech_encoder, text_decoder)

        # 创建文本解码器后处理模块的实例
        self.text_decoder_postnet = SpeechT5TextDecoderPostnet(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回语音T5模型中的编码器
    def get_encoder(self):
        return self.speecht5.get_encoder()

    # 返回语音T5模型中的解码器
    def get_decoder(self):
        return self.speecht5.get_decoder()

    # 冻结特征编码器，禁用其梯度计算，以确保在训练过程中不更新其参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    # 获取文本解码器后处理模块的输出嵌入
    def get_output_embeddings(self):
        return self.text_decoder_postnet.get_output_embeddings()

    # 设置文本解码器后处理模块的输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.text_decoder_postnet.set_output_embeddings(new_embeddings)

    # 重写forward方法，接受多个参数来进行前向推断
    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    # 为生成准备输入的方法，用于生成器模型
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值（past_key_values），则裁剪decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经仅传递最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认旧行为：仅保留最后一个ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 裁剪decoder_input_ids，保留remove_prefix_length之后的部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回包含准备好的输入的字典
        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 改变这个以避免缓存（可能用于调试）
        }

    # 静态方法：重新排序缓存中的过去键值，用于束搜索时的重排
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 遍历每一层的过去键值
        for layer_past in past_key_values:
            # 重新排序过去的状态，根据束搜索的索引
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重排后的过去键值
        return reordered_past
# 定义一个生成语音的函数，接受以下参数：
# - model: SpeechT5PreTrainedModel，用于生成语音的预训练模型
# - input_values: torch.FloatTensor，输入的语音数据
# - speaker_embeddings: Optional[torch.FloatTensor]，说话者嵌入（可选）
# - attention_mask: Optional[torch.LongTensor]，注意力掩码（可选）
# - threshold: float，阈值，默认为0.5
# - minlenratio: float，最小长度比率，默认为0.0
# - maxlenratio: float，最大长度比率，默认为20.0
# - vocoder: Optional[nn.Module]，声码器模块（可选）
# - output_cross_attentions: bool，是否输出交叉注意力（默认为False）
# - return_output_lengths: bool，是否返回输出长度（默认为False）
# 返回值为Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]，输出的语音数据或包含语音数据和长度的元组

if speaker_embeddings is None:
    # 如果未提供说话者嵌入，抛出数值错误并提供解决方法的文本信息
    raise ValueError(
        """`speaker_embeddings` must be specified. For example, you can use a speaker embeddings by following
                the code snippet provided in this link:
                https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors
                """
    )

if attention_mask is None:
    # 如果未提供注意力掩码，根据输入数据中的填充标记生成注意力掩码
    encoder_attention_mask = 1 - (input_values == model.config.pad_token_id).int()
else:
    # 否则，使用给定的注意力掩码
    encoder_attention_mask = attention_mask

bsz = input_values.size(0)  # 计算批次大小

# 使用模型的编码器对输入进行编码
encoder_out = model.speecht5.encoder(
    input_values=input_values,
    attention_mask=encoder_attention_mask,
    return_dict=True,
)

encoder_last_hidden_state = encoder_out.last_hidden_state  # 获取编码器的最后隐藏状态

# 如果模型的编码器是 SpeechT5EncoderWithSpeechPrenet 类型，对注意力掩码进行降采样处理
if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
    encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
        encoder_out[0].shape[1], encoder_attention_mask
    )

# 根据最大和最小长度比率以及模型的减少因子计算输出的最大和最小长度
maxlen = int(encoder_last_hidden_state.size(1) * maxlenratio / model.config.reduction_factor)
minlen = int(encoder_last_hidden_state.size(1) * minlenratio / model.config.reduction_factor)

# 初始化输出序列，以一个全部为零的梅尔频谱开始
output_sequence = encoder_last_hidden_state.new_zeros(bsz, 1, model.config.num_mel_bins)

spectrogram = []  # 初始化频谱列表
cross_attentions = []  # 初始化交叉注意力列表
past_key_values = None  # 初始化过去的键值对
idx = 0  # 初始化索引
result_spectrogram = {}  # 初始化结果频谱字典
    while True:
        idx += 1

        # 在整个输出序列上运行解码器的预处理网络。
        decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)
        # 在预处理网络输出的最后一个元素上运行解码器层。
        decoder_out = model.speecht5.decoder.wrapped_decoder(
            hidden_states=decoder_hidden_states[:, -1:],
            attention_mask=None,
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_cross_attentions,
            return_dict=True,
        )

        # 如果需要输出跨注意力，将其收集起来。
        if output_cross_attentions:
            cross_attentions.append(torch.cat(decoder_out.cross_attentions, dim=0))

        # 获取解码器的最后隐藏状态，并将其作为下一步预测的输入。
        last_decoder_output = decoder_out.last_hidden_state.squeeze(1)
        past_key_values = decoder_out.past_key_values

        # 预测当前步骤的新的梅尔频谱。
        spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
        spectrum = spectrum.view(bsz, model.config.reduction_factor, model.config.num_mel_bins)
        spectrogram.append(spectrum)

        # 将新的梅尔频谱扩展到输出序列中。
        new_spectrogram = spectrum[:, -1, :].view(bsz, 1, model.config.num_mel_bins)
        output_sequence = torch.cat((output_sequence, new_spectrogram), dim=1)

        # 预测这是停止标记的概率。
        prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))

        # 如果仍未达到最小长度要求，继续生成。
        if idx < minlen:
            continue
        else:
            # 如果生成循环次数小于最大长度，检查满足概率阈值的批次中的序列。
            # 否则，假设所有序列都满足阈值，并为批次中的其他频谱填充。
            if idx < maxlen:
                meet_thresholds = torch.sum(prob, dim=-1) >= threshold
                meet_indexes = torch.where(meet_thresholds)[0].tolist()
            else:
                meet_indexes = range(len(prob))
            meet_indexes = [i for i in meet_indexes if i not in result_spectrogram]

            # 如果满足阈值的序列数大于零，则处理这些序列的频谱。
            if len(meet_indexes) > 0:
                spectrograms = torch.stack(spectrogram)
                spectrograms = spectrograms.transpose(0, 1).flatten(1, 2)
                spectrograms = model.speech_decoder_postnet.postnet(spectrograms)
                for meet_index in meet_indexes:
                    result_spectrogram[meet_index] = spectrograms[meet_index]

            # 如果已经收集到足够的结果频谱，则停止生成。
            if len(result_spectrogram) >= bsz:
                break

    # 将结果频谱收集到列表中。
    spectrograms = [result_spectrogram[i] for i in range(len(result_spectrogram))]
    `
    # 如果不需要返回输出长度信息
    if not return_output_lengths:
        # 如果 batch size 为 1，则直接取第一个 spectrogram；否则对 spectrograms 进行批次填充
        spectrogram = spectrograms[0] if bsz == 1 else torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        # 如果有 vocoder，则对 spectrogram 进行合成处理；否则直接使用 spectrogram
        if vocoder is not None:
            outputs = vocoder(spectrogram)
        else:
            outputs = spectrogram
        # 如果需要输出交叉注意力，将交叉注意力拼接起来
        if output_cross_attentions:
            cross_attentions = torch.cat(cross_attentions, dim=2)
            # 如果 batch size 大于 1，则重塑交叉注意力的形状
            if bsz > 1:
                cross_attentions = cross_attentions.view(
                    bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
                )
            # 将输出结果设置为 spectrogram 和 cross_attentions 的元组
            outputs = (outputs, cross_attentions)
    else:
        # 如果需要返回输出长度信息
        # 计算每个 spectrogram 的长度并存储在 spectrogram_lengths 中
        spectrogram_lengths = []
        for i in range(bsz):
            spectrogram_lengths.append(spectrograms[i].size(0))
        # 如果没有 vocoder，则对 spectrograms 进行批次填充并返回 spectrograms 和 lengths 的元组
        if vocoder is None:
            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
            outputs = (spectrograms, spectrogram_lengths)
        else:
            # 否则，对 spectrograms 进行批次填充并使用 vocoder 处理生成 waveforms
            waveforms = vocoder(spectrograms)
            # 计算每个 waveform 的长度并存储在 waveform_lengths 中
            waveform_lengths = [int(waveforms.size(1) / max(spectrogram_lengths)) * i for i in spectrogram_lengths]
            # 返回 waveforms 和 waveform_lengths 的元组作为输出结果
            outputs = (waveforms, waveform_lengths)
        # 如果需要输出交叉注意力，将交叉注意力拼接起来并加入到输出结果中
        if output_cross_attentions:
            cross_attentions = torch.cat(cross_attentions, dim=2)
            cross_attentions = cross_attentions.view(
                bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
            )
            outputs = (*outputs, cross_attentions)
    # 返回最终的输出结果
    return outputs
# 为 SpeechT5ForTextToSpeech 类添加文档字符串，描述其是一个带有文本编码器和语音解码器的 SpeechT5 模型
@add_start_docstrings(
    """SpeechT5 Model with a text encoder and a speech decoder.""",
    SPEECHT5_START_DOCSTRING,
)
class SpeechT5ForTextToSpeech(SpeechT5PreTrainedModel):
    # 主要输入名称为 input_ids
    main_input_name = "input_ids"

    # 初始化函数，接受一个 SpeechT5Config 对象作为参数
    def __init__(self, config: SpeechT5Config):
        super().__init__(config)

        # 如果配置中的词汇表大小为 None，则抛出错误，要求配置中定义语言模型头的词汇表大小
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that does not define the"
                " vocabulary size of the language model head. Please instantiate the model as follows:"
                " `SpeechT5ForTextToSpeech.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of"
                " your model's configuration."
            )

        # 创建文本编码器和语音解码器
        text_encoder = SpeechT5EncoderWithTextPrenet(config)
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        # 使用创建的编码器和解码器创建 SpeechT5Model 对象
        self.speecht5 = SpeechT5Model(config, text_encoder, speech_decoder)

        # 创建语音解码器的后处理网络 SpeechT5SpeechDecoderPostnet
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回文本编码器
    def get_encoder(self):
        return self.speecht5.get_encoder()

    # 返回语音解码器
    def get_decoder(self):
        return self.speecht5.get_decoder()

    # 重写 forward 函数，接受多个输入参数，用于执行前向传播操作
    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSpectrogramOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.Tensor] = None,
    ):
        pass

    # 生成函数，用于生成输出的语音波形
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: Optional[nn.Module] = None,
        output_cross_attentions: bool = False,
        return_output_lengths: bool = False,
        **kwargs,
    ):
        pass
    # 定义生成语音的方法，用于生成基于输入 ID 的语音
    def generate_speech(
        self,
        input_ids: torch.LongTensor,  # 输入的文本 ID，用于生成语音内容
        speaker_embeddings: Optional[torch.FloatTensor] = None,  # 说话者的嵌入向量，可选
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码，可选，用于指定哪些位置要被关注
        threshold: float = 0.5,  # 阈值，控制生成语音的质量
        minlenratio: float = 0.0,  # 最小长度比率，生成语音的最小长度与输入文本长度的比率
        maxlenratio: float = 20.0,  # 最大长度比率，生成语音的最大长度与输入文本长度的比率
        vocoder: Optional[nn.Module] = None,  # 语音合成器模型，可选
        output_cross_attentions: bool = False,  # 是否输出交叉注意力
        return_output_lengths: bool = False,  # 是否返回输出长度信息
@add_start_docstrings(
    """SpeechT5 Model with a speech encoder and a speech decoder.""",
    SPEECHT5_START_DOCSTRING,
)
class SpeechT5ForSpeechToSpeech(SpeechT5PreTrainedModel):
    """
    SpeechT5ForSpeechToSpeech is a specialized model for speech-to-speech tasks, incorporating
    both an encoder and a decoder for processing speech data.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)

        # Initialize the speech encoder with a prenet specific to SpeechT5
        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        
        # Initialize the speech decoder with a prenet specific to SpeechT5
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        
        # Combine the encoder and decoder into the SpeechT5Model
        self.speecht5 = SpeechT5Model(config, speech_encoder, speech_decoder)

        # Initialize the postnet for the speech decoder
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)

        # Perform post-initialization tasks
        self.post_init()

    def get_encoder(self):
        """
        Returns the speech encoder from the SpeechT5 model.
        """
        return self.speecht5.get_encoder()

    def get_decoder(self):
        """
        Returns the speech decoder from the SpeechT5 model.
        """
        return self.speecht5.get_decoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will freeze the gradient computation for the feature encoder,
        preventing its parameters from being updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSpectrogramOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the SpeechT5 model for speech-to-speech conversion tasks.
        """
        # Implementation details handled internally by SpeechT5Model

    @torch.no_grad()
    def generate_speech(
        self,
        input_values: torch.FloatTensor,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: Optional[nn.Module] = None,
        output_cross_attentions: bool = False,
        return_output_lengths: bool = False,
    ):
        """
        Generates speech output based on input values, optionally using speaker embeddings.
        """
        # Implementation details for generating speech output


注释：
这段代码定义了一个名为SpeechT5ForSpeechToSpeech的Python类，用于处理语音到语音的转换任务。该类继承自SpeechT5PreTrainedModel，并组合了一个自定义的语音编码器、解码器及其他相关组件，实现了前向传播方法和生成语音的方法。
    # 这个模型也是一个 PyTorch 的 `torch.nn.Module` 子类。
    # 可以像普通的 PyTorch 模块一样使用，并且关于一般使用和行为的所有问题，请参考 PyTorch 的文档。

    # 参数:
    #     config ([`SpeechT5HifiGanConfig`]):
    #         模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型关联的权重，只会加载配置。
    #         若要加载模型权重，请查看 [`~PreTrainedModel.from_pretrained`] 方法。
# 定义一个自定义的残差块类，继承自 nn.Module
class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # 创建多个卷积层，每个卷积层具有不同的扩张率
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        
        # 创建另一组卷积层，每个卷积层的扩张率为 1
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    # 计算卷积的填充数，确保输出与输入大小相同
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 对所有卷积层应用权重归一化
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除所有卷积层的权重归一化
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    # 前向传播函数，依次通过多个卷积层和激活函数，实现残差连接
    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual  # 残差连接
        return hidden_states


# 将 HiFi-GAN 的 vocoder 模型定义为 SpeechT5HifiGan 类，它继承自 PreTrainedModel
@add_start_docstrings(
    """HiFi-GAN vocoder.""",  # 添加了关于 HiFi-GAN 的文档字符串
    HIFIGAN_START_DOCSTRING,  # 引用了 HIFIGAN_START_DOCSTRING 的文档字符串（可能是预定义的常量或函数）
)
class SpeechT5HifiGan(PreTrainedModel):
    config_class = SpeechT5HifiGanConfig  # 指定配置类为 SpeechT5HifiGanConfig
    main_input_name = "spectrogram"  # 指定主要输入名称为 "spectrogram"
    # 初始化函数，接受一个名为config的SpeechT5HifiGanConfig对象作为参数
    def __init__(self, config: SpeechT5HifiGanConfig):
        # 调用父类的初始化函数，传递config作为参数
        super().__init__(config)
        # 计算并记录resblock_kernel_sizes列表的长度，即卷积块的数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        # 计算并记录upsample_rates列表的长度，即上采样率的数量
        self.num_upsamples = len(config.upsample_rates)
        # 创建一个一维卷积层，输入维度为config.model_in_dim，输出通道数为config.upsample_initial_channel，卷积核大小为7，步长为1，填充为3
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # 创建一个空的模块列表upsampler，用于存储上采样卷积层
        self.upsampler = nn.ModuleList()
        # 遍历upsample_rates和upsample_kernel_sizes的元素，逐个创建反卷积层并添加到upsampler中
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        # 创建一个空的模块列表resblocks，用于存储残差块
        self.resblocks = nn.ModuleList()
        # 根据upsampler的长度，遍历每个上采样层并添加对应数量的残差块
        for i in range(len(self.upsampler)):
            # 计算当前层的通道数
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            # 遍历resblock_kernel_sizes和resblock_dilation_sizes的元素，逐个创建残差块并添加到resblocks中
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 创建最终的一维卷积层，输入通道数为channels，输出通道数为1，卷积核大小为7，步长为1，填充为3
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        # 注册缓冲区mean，用于存储均值张量，初始化为与输入维度一致的全零张量
        self.register_buffer("mean", torch.zeros(config.model_in_dim))
        # 注册缓冲区scale，用于存储标准差张量，初始化为与输入维度一致的全一张量
        self.register_buffer("scale", torch.ones(config.model_in_dim))

        # 调用post_init函数，用于初始化权重并进行最终处理
        self.post_init()

    # 权重初始化函数，初始化线性层和一维卷积层的权重
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果module是nn.Linear或nn.Conv1d类型的实例
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正态分布随机初始化权重，均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()

    # 应用权重归一化，对conv_pre、upsampler中的每个层和resblocks中的每个块应用权重归一化
    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)

    # 移除权重归一化，对conv_pre、upsampler中的每个层和resblocks中的每个块移除权重归一化
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)
    def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        if self.config.normalize_before:
            # 如果配置要求，在前处理时对输入的频谱图进行标准化
            spectrogram = (spectrogram - self.mean) / self.scale

        # 检查输入的频谱图是否是批处理的
        is_batched = spectrogram.dim() == 3
        if not is_batched:
            # 如果输入的频谱图未经批处理，则添加批处理维度
            spectrogram = spectrogram.unsqueeze(0)

        # 将频谱图的通道维和时间步维进行转置，以便卷积层处理
        hidden_states = spectrogram.transpose(2, 1)

        # 在前处理卷积层中应用卷积操作
        hidden_states = self.conv_pre(hidden_states)

        # 循环进行上采样操作
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            # 应用残差块
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        # 在后处理卷积层中应用卷积操作和激活函数
        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        if not is_batched:
            # 如果输入未经批处理，则去除批处理维度，并将张量展平成音频波形
            waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)
        else:
            # 如果输入经过批处理，则去除时间步维，因为此时时间步维已经折叠成一个
            waveform = hidden_states.squeeze(1)

        return waveform
```