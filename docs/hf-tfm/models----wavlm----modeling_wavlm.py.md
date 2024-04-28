# `.\transformers\models\wavlm\modeling_wavlm.py`

```py
# 该代码实现了一个 WavLM 模型，WavLM 是一个基于 Transformer 的语音模型，由微软研究院和 Hugging Face 团队开发。这个模型可以用于语音识别、语音分类等任务。

# 这是该模型的 PyTorch 实现，代码中包含了一些公共函数和类定义。

# 首先导入所需的库和模块
# coding=utf-8
# Copyright 2021 The Fairseq Authors, Microsoft Research, and The HuggingFace Inc. team. All rights reserved.
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

import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_wavlm import WavLMConfig


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2

# General docstring
_CONFIG_FOR_DOC = "WavLMConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'mister quilter is the aposle of the middle classes and we are glad to welcome his gospel'"
_CTC_EXPECTED_LOSS = 12.51

# Frame class docstring
_FRAME_CLASS_CHECKPOINT = "microsoft/wavlm-base-plus-sd"
_FRAME_EXPECTED_OUTPUT = [0, 0]

# Speaker Verification docstring
_XVECTOR_CHECKPOINT = "microsoft/wavlm-base-plus-sv"
_XVECTOR_EXPECTED_OUTPUT = 0.97

WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/wavlm-base",
    "microsoft/wavlm-base-plus",
    "microsoft/wavlm-large",
    # See all WavLM models at https://huggingface.co/models?filter=wavlm
]

# 定义了一个计算随机 mask 掩码的函数
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
    """
    # ...
    # 定义一个函数，用于计算输入长度生成多少个需要被遮罩的跨度
    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 根据输入长度计算应该遮罩多少个跨度
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保遮罩的跨度数量不超过序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保遮罩的跨度数量不超过输入长度-(mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 如果输入的mask_length小于1，则报错
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    # 如果输入的mask_length大于序列长度，则报错
    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # 用于概率舍入
    epsilon = np.random.rand(1).item()

    # 计算batch中需要遮罩的跨度数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # 定义一个全为False的布尔数组，用于填充遮罩
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    # 计算最大遮罩跨度数量
    max_num_masked_span = compute_num_masked_span(sequence_length)

    # 如果最大遮罩跨度数量为0，则直接返回全为False的布尔数组
    if max_num_masked_span == 0:
        return spec_aug_mask
    # 针对每个输入长度进行操作
    for input_length in input_lengths:
        # 计算当前输入长度下需要遮掩的 span 数量
        num_masked_span = compute_num_masked_span(input_length)
    
        # 获取需要遮掩的随机索引
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )
    
        # 选取第一个采样的索引作为 dummy 索引来填充向量
        # 确保所有批次的尺寸相同,因为概率舍入
        # 选取第一个采样只是将这些向量填充了两次
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只有在 `input_length` 严格小于 `sequence_length` 时才会发生
            # 最后一个token必须是填充token,我们可以用它作为 dummy 遮掩id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]
    
        # 将实际的遮掩索引和 dummy 索引拼接起来
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)
    
    # 将 spec_aug_mask_idxs 转换为 numpy 数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)
    
    # 将遮掩索引扩展为遮掩 span
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)
    
    # 给遮掩起始索引加上偏移量,形成遮掩 span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
    
    # 确保遮掩索引不超过序列长度
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1
    
    # 将遮掩索引散布到 spec_aug_mask 中
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)
    
    return spec_aug_mask
# 定义 WavLMNoLayerNormConvLayer 类，继承自 nn.Module
class WavLMNoLayerNormConvLayer(nn.Module):
    # 初始化函数，接受 config 和 layer_id 两个参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为 config.conv_dim[layer_id - 1] 或者 1（如果 layer_id <= 0）
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为 config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建卷积层，指定输入维度、输出维度、卷积核大小、步长和是否使用偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数为 config.feat_extract_activation 对应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数，接受 hidden_states 作为输入
    def forward(self, hidden_states):
        # 对输入进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的结果使用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的结果
        return hidden_states


# 定义 WavLMLayerNormConvLayer 类，继承自 nn.Module
class WavLMLayerNormConvLayer(nn.Module):
    # 初始化函数，接受 config 和 layer_id 两个参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为 config.conv_dim[layer_id - 1] 或者 1（如果 layer_id <= 0）
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为 config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建卷积层，指定输入维度、输出维度、卷积核大小、步长和是否使用偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建 LayerNorm 层，指定特征维度和是否使用可学习的缩放参数和平移参数
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 设置激活函数为 config.feat_extract_activation 对应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数，接受 hidden_states 作为输入
    def forward(self, hidden_states):
        # 对输入进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 将卷积后的结果维度进行变换
        hidden_states = hidden_states.transpose(-2, -1)
        # 对变换后的结果进行 LayerNorm 操作
        hidden_states = self.layer_norm(hidden_states)
        # 恢复维度
        hidden_states = hidden_states.transpose(-2, -1)
        # 对处理后的结果使用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的结果
        return hidden_states


# 定义 WavLMGroupNormConvLayer 类，继承自 nn.Module
class WavLMGroupNormConvLayer(nn.Module):
    # 初始化函数，接受 config 和 layer_id 两个参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为 config.conv_dim[layer_id - 1] 或者 1（如果 layer_id <= 0）
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为 config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建卷积层，指定输入维度、输出维度、卷积核大小、步长和是否使用偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数为 config.feat_extract_activation 对应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
        # 创建 GroupNorm 层，指定组数量和通道数量，并指定是否使用可学习的缩放参数和平移参数
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    # 前向传播函数，接受 hidden_states 作为输入
    def forward(self, hidden_states):
        # 对输入进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的结果使用 GroupNorm 操作
        hidden_states = self.layer_norm(hidden_states)
        # 对处理后的结果使用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的结果
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding复制并改名为WavLMPositionalConvEmbedding
class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个一维卷积层，用于位置编码的计算
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        # 如果存在nn.utils.parametrizations中的weight_norm，则将weight_norm赋值为它
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 如果启用了deepspeed zero3，则使用深度增强模块
        if is_deepspeed_zero3_enabled():
            import deepspeed
            # 使用deepspeed.zero.GatheredParameters来处理权重共享
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 否则，应用权重共享
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建一个WavLMSamePadLayer实例
        self.padding = WavLMSamePadLayer(config.num_conv_pos_embeddings)
        # 获取激活函数类型
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数
    def forward(self, hidden_states):
        # 将输入的hidden_states张量的维度进行转置
        hidden_states = hidden_states.transpose(1, 2)

        # 对转置后的hidden_states进行卷积计算
        hidden_states = self.conv(hidden_states)
        # 使用padding层进行padding操作
        hidden_states = self.padding(hidden_states)
        # 对经过padding操作后的hidden_states使用激活函数进行激活
        hidden_states = self.activation(hidden_states)

        # 再次对hidden_states张量的维度进行转置，恢复原先的维度
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer复制并改名为WavLMSamePadLayer
class WavLMSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 计算需要移除的填充数
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    # 前向传播函数
    def forward(self, hidden_states):
        # 如果需要移除填充，则对hidden_states进行截取操作
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder复制并改名为WavLMFeatureEncoder
class WavLMFeatureEncoder(nn.Module):
    """从原始音频波形中构造特征"""
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 根据 config.feat_extract_norm 的值来构建 conv_layers
        if config.feat_extract_norm == "group":
            # 第一个卷积层使用 WavLMGroupNormConvLayer，其他层使用 WavLMNoLayerNormConvLayer
            conv_layers = [WavLMGroupNormConvLayer(config, layer_id=0)] + [
                WavLMNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 所有卷积层使用 WavLMLayerNormConvLayer
            conv_layers = [WavLMLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            # 如果 config.feat_extract_norm 不符合要求，则抛出异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        # 将构建好的卷积层列表封装成 nn.ModuleList
        self.conv_layers = nn.ModuleList(conv_layers)
        # 禁用梯度检查点
        self.gradient_checkpointing = False
        # 标记模型参数可训练
        self._requires_grad = True

    # 冻结模型参数，使其不可训练
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    # 前向传播方法
    def forward(self, input_values):
        # 将输入数据增加一个维度，变成 [batch_size, 1, ...]
        hidden_states = input_values[:, None]

        # 确保在训练时 hidden_states 需要梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 依次通过各个卷积层
        for conv_layer in self.conv_layers:
            # 如果启用了梯度检查点且正在训练
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 使用梯度检查点机制计算卷积层输出
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 直接计算卷积层输出
                hidden_states = conv_layer(hidden_states)

        # 返回最终的特征表示
        return hidden_states
# 创建一个WavLMFeatureExtractor类，继承自WavLMFeatureEncoder
class WavLMFeatureExtractor(WavLMFeatureEncoder):
    def __init__(self, config):
        # 调用父类WavLMFeatureEncoder的初始化方法
        super().__init__(config)
        # 发出警告，提示该类已被弃用，并将在Transformers v5中被移除
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection类复制，并将Wav2Vec2更改为WavLM
class WavLMFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建层归一化对象
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 创建线性变换对象，配置输入和输出维度
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 创建丢弃层对象，配置丢弃概率
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 需要非投影隐藏状态用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states
# 创建WavLMAttention类，继承自nn.Module
class WavLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        num_buckets: int = 320,
        max_distance: int = 800,
        has_relative_position_bias: bool = True,
    ):
        super().__init__()
        # 初始化注意力机制参数
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
        # 创建线性投影对象
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # 初始化相对位置编码参数
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # 创建GRU相对位置编码的常量和线性层
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        if has_relative_position_bias:
            # 如果存在相对位置偏置，创建嵌入层
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        index=0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        # 获取隐藏状态的 batch size, 目标长度和隐藏状态的维度
        bsz, tgt_len, _ = hidden_states.size()

        # 如果位置偏置为空，则创建位置偏置
        if position_bias is None:
            # 计算位置偏置
            position_bias = self.compute_bias(tgt_len, tgt_len)
            # 重复位置偏置以匹配 batch size，然后展开成 (batch size * self.num_heads, 目标长度, 目标长度) 的形状
            position_bias = (
                position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, tgt_len)
            )

        # 计算相对位置偏置：
        # 1) 重新塑造隐藏状态
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)

        # 2) 投影隐藏状态
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        # 重新塑造投影后的隐藏状态，sum(-1) 相当于对最后一个维度求和
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)

        # 3) 从投影后的隐藏状态计算位置偏置的门控
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0

        # 4) 将门控应用于位置偏置，计算门控位置偏置
        gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((-1, tgt_len, tgt_len))

        # 使用多头自注意力机制计算注意力输出和注意力权重
        attn_output, attn_weights = self.torch_multi_head_self_attention(
            hidden_states, attention_mask, gated_position_bias, output_attentions
        )

        return attn_output, attn_weights, position_bias

    def torch_multi_head_self_attention(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Union[torch.LongTensor, torch.BoolTensor],
        gated_position_bias: torch.FloatTensor,
        output_attentions: bool,
    ) -> (torch.FloatTensor, torch.FloatTensor):
        """simple wrapper around torch's multi_head_attention_forward function"""
        # self-attention assumes q = k = v
        # 将隐藏状态进行转置，作为查询、键、值
        query = key = value = hidden_states.transpose(0, 1)
        # 如果有注意力掩码，则生成键值掩码
        key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None

        # disable bias and add_zero_attn
        bias_k = bias_v = None
        add_zero_attn = False

        # PyTorch 1.3.0 has F.multi_head_attention_forward defined
        # so no problem with backwards compatibility
        # 调用 PyTorch 中的 multi_head_attention_forward 函数进行多头注意力计算
        attn_output, attn_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k,
            bias_v,
            add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            output_attentions,
            gated_position_bias,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )

        # [Seq_Len, Batch Size, ...] -> [Batch Size, Seq_Len, ...]
        # 将输出的注意力张量转置，调整其形状为[Batch Size, Seq_Len, ...]
        attn_output = attn_output.transpose(0, 1)

        if attn_weights is not None:
            # IMPORTANT: Attention weights are averaged weights
            # here which should not be the case. This is an open issue
            # on PyTorch: https://github.com/pytorch/pytorch/issues/32590
            # 注意：注意力权重应该是未经平均的权重，但在这里进行了平均。
            # 这是 PyTorch 上的一个已知问题。
            attn_weights = attn_weights[:, None].broadcast_to(
                attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
            )

        return attn_output, attn_weights

    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor:
        # 生成相对位置编码
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        # 获取相对位置编码的值
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values
    # 计算相对位置的桶索引
    def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
        # 桶的数量为 self.num_buckets 的一半
        num_buckets = self.num_buckets // 2

        # 计算相对位置是否大于 0，如果是则乘以 num_buckets，得到相对桶索引
        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        # 取相对位置的绝对值
        relative_positions = torch.abs(relative_positions)

        # 计算小于 max_exact 的相对位置
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        # 如果相对位置大于 max_exact，计算对应的桶索引
        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
        relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
        relative_position_if_large = (max_exact + relative_positions_if_large).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # 根据相对位置大小选择相应的桶索引
        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        # 返回相对位置的桶索引
        return relative_buckets
# 定义 WavLMFeedForward 类，继承自 nn.Module 类，用于处理神经网络的前馈部分
class WavLMFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化中间层的 dropout 操作
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 初始化中间层的全连接神经网络
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            # 如果隐藏激活函数为字符串形式，则查找对应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 初始化输出层的全连接神经网络
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播函数
    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# 定义 WavLMEncoderLayer 类，继承自 nn.Module 类，用于处理神经网络的编码层
class WavLMEncoderLayer(nn.Module):
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        # 初始化注意力机制对象
        self.attention = WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数
    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)

        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 定义 WavLMEncoderLayerStableLayerNorm 类
class WavLMEncoderLayerStableLayerNorm(nn.Module):
    # 初始化函数，接受一个 WavLMConfig 类型的参数和一个布尔类型的参数，设置了注意力、Dropout 和 LayerNorm 层
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        # 初始化注意力层对象
        self.attention = WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化 LayerNorm 层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化前馈神经网络层对象
        self.feed_forward = WavLMFeedForward(config)
        # 初始化最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受隐藏层状态、注意力 mask、位置偏置（可选）、是否输出注意力权重的标志，返回处理后的隐藏层状态和位置偏置
    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        # 保存注意力前的隐藏层状态
        attn_residual = hidden_states
        # 对隐藏层状态进行 LayerNorm 处理
        hidden_states = self.layer_norm(hidden_states)
        # 经过注意力层处理，得到处理后的隐藏层状态、注意力权重和位置偏置
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        # 对处理后的隐藏层状态进行 Dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 加上保存的注意力前的隐藏层状态，得到最终的隐藏层状态
        hidden_states = attn_residual + hidden_states
        # 经过 LayerNorm 处理后的隐藏层状态再经过前馈神经网络处理
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 输出结果包括隐藏层状态和位置偏置
        outputs = (hidden_states, position_bias)

        # 如果需要输出注意力权重，则加上注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
class WavLMEncoder(nn.Module):
    # WavLMEncoder类的构造函数，接受一个config参数
    def __init__(self, config):
        # 调用nn.Module的构造函数
        super().__init__()
        # 将传入的config赋值给类的config属性
        self.config = config
        # 创建一个WavLMPositionalConvEmbedding实例，并将其赋值给pos_conv_embed属性
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        # 创建一个LayerNorm实例，并将其赋值给layer_norm属性
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout实例，并将其赋值给dropout属性
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建一个ModuleList实例，并将其中包含的WavLMEncoderLayer实例赋值给layers属性
        self.layers = nn.ModuleList(
            [WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers)]
        )
        # 将gradient_checkpointing设置为False
        self.gradient_checkpointing = False

    # WavLMEncoder类的前向传播函数，输入参数为hidden_states，以及一些可选参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    # 这个函数是处理模型的隐藏状态和注意力输出的
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
        # 如果不需要输出隐藏状态和注意力权重就初始化为空
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
    
        # 如果存在注意力遮罩，将被遮罩的token的隐藏状态设置为0
        if attention_mask is not None:
            hidden_states[~attention_mask] = 0.0
    
        # 将位置嵌入添加到隐藏状态中
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
    
        # 判断是否启用了deepspeed zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None
    
        # 遍历每一层
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 以一定概率跳过当前层（layerdrop）
            dropout_probability = torch.rand([])
            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果没有跳过当前层或启用了deepspeed zero3
                if self.gradient_checkpointing and self.training:
                    # 使用梯度检查点执行前向计算
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_bias,
                        output_attentions,
                    )
                else:
                    # 正常执行前向计算
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_bias=position_bias,
                        output_attentions=output_attentions,
                        index=i,
                    )
                hidden_states, position_bias = layer_outputs[:2]
            else:
                layer_outputs = (None, None)
    
            # 如果需要输出注意力权重，将当前注意力权重添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
    
        # 将最后一层的隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 根据输出要求返回对应的结果
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 定义 WavLMEncoderStableLayerNorm 类，继承自 nn.Module
class WavLMEncoderStableLayerNorm(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()  # 调用父类的初始化函数
        self.config = config  # 保存 config 参数
        # 创建 WavLMPositionalConvEmbedding 对象
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        # 创建 nn.LayerNorm 对象，用于层标准化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 nn.Dropout 对象，用于 Dropout 操作
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建 nn.ModuleList 对象，用于保存多个 WavLMEncoderLayerStableLayerNorm 对象
        self.layers = nn.ModuleList(
            [
                WavLMEncoderLayerStableLayerNorm(config, has_relative_position_bias=(i == 0))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False  # 默认关闭梯度检查点

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
            # 初始化隐藏状态和自注意力的输出
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None

            # 如果存在注意力掩码，则确保填充标记不被注意
            if attention_mask is not None:
                hidden_states[~attention_mask] = 0

            # 计算位置嵌入并与隐藏状态相加
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.dropout(hidden_states)

            # 检查是否启用了深度速度Zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
            position_bias = None

            # 遍历所有层
            for i, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 添加LayerDrop
                dropout_probability = torch.rand([])

                skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 使用梯度检查点和训练模式来执行层
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                            position_bias,
                            output_attentions,
                        )
                    else:
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            position_bias=position_bias,
                        )
                    hidden_states, position_bias = layer_outputs[:2]

                if skip_the_layer:
                    layer_outputs = (None, None)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[2],)

            hidden_states = self.layer_norm(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                # 如果不返回字典，则返回具体值
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 返回包含最终隐藏状态、隐藏状态历史和自注意力的字典结果
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
            )
# 基于 Gumbel-Softmax 的向量量化模块
class WavLMGumbelVectorQuantizer(nn.Module):
    """
    使用 Gumbel-Softmax 进行向量量化。参见 [CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) 获取更多信息。
    """

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置代码本分组数和每组的代码本向量数
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group

        # 检查代码本向量维度是否能被分组数整除
        if config.codevector_dim % self.num_groups != 0:
            # 如果不能整除，抛出异常
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible"
                f" by `config.num_codevector_groups` {self.num_groups} "
                "for concatenation."
            )

        # 初始化代码本变量（代码本词）
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        # 初始化线性投影层，将输入映射到代码本索引
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # 可以在训练过程中衰减的温度参数
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs):
        # 计算概率分布的困惑度
        marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity
    def forward(self, hidden_states):
        # 获取隐藏状态的维度
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 投影到编码向量维度
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 使用Gumbel Softmax方法以可微分的方式对隐藏状态进行采样，得到编码向量的概率分布
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True)
            codevector_probs = codevector_probs.type_as(hidden_states)

            # 计算困惑度
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            # 使用非可微分的方式对隐藏状态进行argmax操作，得到硬编码向量分布（one hot）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs)

        # 将编码向量概率分布重新转换为原始形状
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        
        # 使用概率分布检索编码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity
# 定义一个名为 WavLMAdapter 的类，用于转换 Wav2Vec2 模型为 WavLM 模型
class WavLMAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 如果输出隐藏层大小不等于隐藏层大小，则需要进行降维投影
        if config.output_hidden_size != config.hidden_size:
            # 创建线性投影层，将隐藏层大小调整为输出隐藏层大小
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # 创建 LayerNorm 层，用于对投影后的向量进行归一化
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            # 如果输出隐藏层大小等于隐藏层大小，则不需要进行降维投影
            self.proj = self.proj_layer_norm = None

        # 创建 WavLMAdapterLayer 的 ModuleList，用于处理输入数据
        self.layers = nn.ModuleList(WavLMAdapterLayer(config) for _ in range(config.num_adapter_layers))
        # 设置层丢弃率
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # 如果需要，对隐藏状态进行投影
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 将隐藏状态转置以适应卷积层的输入格式
        hidden_states = hidden_states.transpose(1, 2)

        # 遍历每个适配器层
        for layer in self.layers:
            # 计算层丢弃率
            layerdrop_prob = np.random.random()
            # 如果处于推理阶段或者层丢弃率低于阈值，则执行适配器层
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        # 再次转置隐藏状态以恢复原始形状
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 定义一个名为 WavLMAdapterLayer 的类，用于处理输入数据
class WavLMAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一维卷积层，用于特征转换
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states):
        # 通过卷积层进行特征转换
        hidden_states = self.conv(hidden_states)
        # 使用 GLU 激活函数
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


# 定义一个名为 WavLMPreTrainedModel 的类，用于处理预训练模型的初始化和加载
class WavLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # WavLMConfig 类
    config_class = WavLMConfig
    # 模型名称前缀
    base_model_prefix = "wavlm"
    # 主要输入名称
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化模型的权重
    def _init_weights(self, module):
        # 对于 WavLMGumbelVectorQuantizer 类型的模块，进行特殊的初始化
        if isinstance(module, WavLMGumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        # 对于 WavLMPositionalConvEmbedding 类型的模块，进行特殊的初始化
        elif isinstance(module, WavLMPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        # 对于 WavLMFeatureProjection 类型的模块，进行特殊的初始化
        elif isinstance(module, WavLMFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 对于 nn.Linear 类型的模块，进行初始化
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 对于 nn.LayerNorm 或 nn.GroupNorm 类型的模块，初始化偏置为零，权重填充为1.0
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 对于 nn.Conv1d 类型的模块，使用 kaiming_normal_ 算法进行初始化
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            # 如果有偏置项，进行初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # 计算卷积层的输出长度
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层的输出长度公式，来源于https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历卷积核大小和步长，计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要增加 adapter 层，再次计算输出长度
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    # 获取特征向量的注意力屏蔽
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
        # 计算去除填充的有效长度以便后续使用，沿着最后一个维度求累积和
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 根据非填充的长度获取特征提取器的输出长度，如果需要添加适配器则加上
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        # 获取 batch 大小
        batch_size = attention_mask.shape[0]

        # 初始化一个全零的注意力掩码，形状为(batch_size, feature_vector_length)，数据类型和设备与 attention_mask 一致
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # 确保输出长度之前的所有值都被注意到
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # 翻转掩码张量，进行累积和操作，再次翻转，并转换为布尔类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        # 返回注意力掩码
        return attention_mask
# WAVLM_START_DOCSTRING 用于定义 WavLM 模型的文档字符串，介绍了该模型的来源和继承关系
WAVLM_START_DOCSTRING = r"""
    WavLM was proposed in [WavLM: Unified Speech Representation Learning with Labeled and Unlabeled
    Data](https://arxiv.org/abs/2110.13900) by Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo
    Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian,
    Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`WavLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# WAVLM_INPUTS_DOCSTRING 用于定义 WavLM 模型的输入文档字符串
WAVLM_INPUTS_DOCSTRING = r"""
    # 这些是函数的参数说明:
    # input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
    # 输入的原始语音波形数据, 可以是从 .flac 或 .wav 音频文件加载得到的 List[float] 或 numpy.ndarray 数据, 需要使用 AutoProcessor 进行填充和转换为 torch.FloatTensor 类型
    # attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    # 注意力掩码, 用于避免在填充标记上进行卷积和注意力操作, 1 表示未被掩码, 0 表示被掩码
    # 注意: attention_mask 只应在相应的处理器有 config.return_attention_mask == True 时传入, 否则会导致性能下降
    # output_attentions (`bool`, *optional*):
    # 是否返回所有注意力层的注意力张量
    # output_hidden_states (`bool`, *optional*):
    # 是否返回所有层的隐藏状态
    # return_dict (`bool`, *optional*):
    # 是否返回 ~utils.ModelOutput 而不是普通元组
"""
@add_start_docstrings(
    "The bare WavLM Model transformer outputting raw hidden-states without any specific head on top.",
    WAVLM_START_DOCSTRING,
)
# 基于 WavLMPreTrainedModel 创建 WavLMModel 类
class WavLMModel(WavLMPreTrainedModel):
    def __init__(self, config: WavLMConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = WavLMFeatureEncoder(config)  # 创建特征提取器
        self.feature_projection = WavLMFeatureProjection(config)  # 创建特征投影器
 
        # 只有当 mask 时间概率大于 0.0 或者特征概率大于 0.0 时，模型才需要掩码向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())  # 创建掩码语谱图嵌入向量

        if config.do_stable_layer_norm:
            self.encoder = WavLMEncoderStableLayerNorm(config)  # 创建稳定层归一化编码器
        else:
            self.encoder = WavLMEncoder(config)  # 创建编码器

        self.adapter = WavLMAdapter(config) if config.add_adapter else None  # 如果设置了适配器，创建适配器

        # 初始化权重并应用最终处理
        self.post_init()

    def freeze_feature_extractor(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会被更新。
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会被更新。
        """
        self.feature_extractor._freeze_parameters()  # 冻结特征编码器的参数

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        # 检查配置中的apply_spec_augment选项是否为True，如果不是则直接返回隐藏状态
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        # 计算batch size、序列长度和隐藏层大小
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            # 使用给定的mask_time_indices沿时间轴应用SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 计算mask_time_indices并根据条件应用SpecAugment
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
            # 计算mask_feature_indices并应用SpecAugment沿特征轴
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

    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Wav2Vec2BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数，输入为input_values（音频特征向量）和 attention_mask（用于掩盖注意力的掩码），输出为元组或Wav2Vec2BaseModelOutput
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 如果output_attentions不为空则为其赋值，否则使用self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states不为空则为其赋值，否则使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict不为空则为其赋值，否则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用特征提取器提取特征
        extract_features = self.feature_extractor(input_values)
        # 转置特征矩阵
        extract_features = extract_features.transpose(1, 2)

        # 如果attention_mask不为空
        if attention_mask is not None:
            # 计算对应于特征向量的缩减注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 使用特征投影层对提取的特征进行投影
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 屏蔽隐藏状态
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 使用编码器输出隐藏状态和注意力
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果适配器不为空，使用适配器
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 如果return_dict为False，则返回隐藏状态、提取的特征以及编码器输出的其他值
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回Wav2Vec2BaseModelOutput对象，包括最后的隐藏状态、提取的特征、编码器输出的隐藏状态和注意力
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 为Connectionist Temporal Classification (CTC)顶部的一个`语言建模`头部的WavLM模型添加起始文档字符串
@add_start_docstrings(
    """WavLM Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAVLM_START_DOCSTRING,
)
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC复制，将Wav2Vec2->WavLM，wav2vec2->wavlm，WAV_2_VEC_2->WAVLM
class WavLMForCTC(WavLMPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        # 初始化函数，继承父类WavLMPreTrainedModel的初始化
        super().__init__(config)

        # 创建WavLMModel对象
        self.wavlm = WavLMModel(config)
        # 添加dropout层，设置概率为config.final_dropout
        self.dropout = nn.Dropout(config.final_dropout)

        # 目标语言，默认为None
        self.target_lang = target_lang

        # 如果config.vocab_size为None，抛出错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `WavLMForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        # 设置输出隐藏层的大小，并创建线性层lm_head
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        """
        此方法重写了[`~PreTrainedModel.tie_weights`]，以便在传递`target_lang=...`到`from_pretrained(...)`时可以正确加载适配器权重。

        用户不应调用此方法，而且可能在将来会更改。
        """

        # 注意，`tie_weights`通常用于绑定输入和输出嵌入权重。该方法被重新用于正确加载WavLM的适配器层，以便我们不必引入新的API到[`PreTrainedModel`]。虽然有些诡计，但是WavLM从不需要绑定输入和输出嵌入，所以在这里重新用这个函数是可以的。
        target_lang = self.target_lang

        # 如果target_lang不为None且self.config.adapter_attn_dim为None，则抛出值错误
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果target_lang为None且self.config.adapter_attn_dim不为None，则打印日志
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果target_lang不为None，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 冻结特征提取器的梯度计算，使其在训练过程中不被更新
    def freeze_feature_extractor(self):
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 冻结特征编码器的梯度计算，使其在训练过程中不被更新
    def freeze_feature_encoder(self):
        self.wavlm.feature_extractor._freeze_parameters()

    # 冻结基础模型的梯度计算，使其在训练过程中不被更新，只有分类头会被更新
    def freeze_base_model(self):
        for param in self.wavlm.parameters():
            param.requires_grad = False

    # 前向传播函数，接收输入的数值、注意力掩码、是否输出注意力权重、是否输出隐藏状态、是否返回字典、标签作为输入
    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # 确保 return_dict 存在，若为 None，则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 wavlm 模型对输入进行处理，获得输出
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 hidden_states 并对其进行 dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 使用 lm_head 模型生成 logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 从 attention_mask 获取 loss 的输入长度
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充的标记为 -100，当不被关注时
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss 不支持 fp16，将 logits 转为概率并进行转置
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 关闭 cudnn 加速，计算 ctc_loss
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            # 如果不返回字典，则输出 logits ���其他输出信息
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回 CausalLMOutput 类型的输出
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 使用预训练的 WAVLM 模型添加一个用于序列分类任务的线性层
@add_start_docstrings(
    """
    WavLM Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """,
    WAVLM_START_DOCSTRING,
)
class WavLMForSequenceClassification(WavLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 检查配置中是否有 "add_adapter" 属性并且为True，如果是，则抛出异常
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of WavLM adapters (config.add_adapter=True)"
            )
        # 创建 WAVLM 模型对象
        self.wavlm = WavLMModel(config)
        # 计算隐藏层数量
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        # 如果配置中开启了加权层求和，创建权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建线性投影层
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建分类器线性层
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征提取器，禁用梯度计算以防止参数在训练期间更新
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 冻结特征编码器，禁用梯度计算以防止参数在训练期间更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wavlm.feature_extractor._freeze_parameters()

    # 冻结基础模型，禁用梯度计算以防止参数在训练期间更新，仅分类头会被更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False

    # 添加文档字符串注释和代码示例注释
    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward 复制而来，修改了 Wav2Vec2 为 WavLM，wav2vec2 为 wavlm
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入值，可选的张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选的张量，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典，可选的布尔值，默认为 None
        labels: Optional[torch.Tensor] = None,  # 标签，可选的张量，默认为 None
    ) -> Union[Tuple, SequenceClassifierOutput]:  # 函数返回值的类型，可能是元组或 SequenceClassifierOutput 类型
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果 return_dict 不为 None，则使用它；否则使用 self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states  # 如果 self.config.use_weighted_layer_sum 为 True，则将 output_hidden_states 设为 True
    
        # 调用 wavlm 模型进行前向传播
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        if self.config.use_weighted_layer_sum:  # 如果配置中使用了加权层求和
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态的起始位置
            hidden_states = torch.stack(hidden_states, dim=1)  # 在维度 1 上堆叠隐藏状态
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行 softmax 归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 使用归一化的层权重对隐藏状态进行加权求和
        else:
            hidden_states = outputs[0]  # 否则直接使用第一个输出作为隐藏状态
    
        hidden_states = self.projector(hidden_states)  # 使用投影器对隐藏状态进行投影
        if attention_mask is None:  # 如果没有提供注意力掩码
            pooled_output = hidden_states.mean(dim=1)  # 对隐藏状态进行平均池化
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)  # 获取特征向量的注意力掩码
            hidden_states[~padding_mask] = 0.0  # 将填充部分的隐藏状态置零
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)  # 对填充部分进行掩码后的池化
    
        logits = self.classifier(pooled_output)  # 使用分类器对池化后的输出进行分类
    
        loss = None  # 初始化损失为 None
        if labels is not None:  # 如果提供了标签
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))  # 计算损失
    
        if not return_dict:  # 如果不返回字典
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 构建输出元组
            return ((loss,) + output) if loss is not None else output  # 如果有损失，则将损失与输出元组合并返回，否则直接返回输出元组
    
        return SequenceClassifierOutput(  # 返回 SequenceClassifierOutput 对象
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 add_start_docstrings 装饰器添加模型的文档字符串
# 这个模型是 WavLM 模型，具有用于说话者分离等任务的帧分类头部
# 在 WAVLM_START_DOCSTRING 常量之上添加
@add_start_docstrings(
    """
    WavLM Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    WAVLM_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification 复制得到，将其中的 Wav2Vec2->WavLM, wav2vec2->wavlm, WAV_2_VEC_2->WAVLM 替换
# WavLMForAudioFrameClassification 类派生自 WavLMPreTrainedModel 类
class WavLMForAudioFrameClassification(WavLMPreTrainedModel):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置对象有 add_adapter 属性并且配置为 True，抛出 ValueError 异常
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of WavLM adapters (config.add_adapter=True)"
            )
        # 使用配置对象创建 WavLMModel 实例
        self.wavlm = WavLMModel(config)
        # 计算 transformer 层的数量加 1（input embeddings），如果配置中使用 weighted layer sum，则创建一个层数量大小的张量来保存权重值
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建一个线性层，用于分类任务，其输入大小为配置中的 hidden_size，输出大小为配置中的 num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 保存标签数量
        self.num_labels = config.num_labels

        # 初始化网络权重
        self.init_weights()

    # 冻结特征提取器的参数，不会在训练中更新
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 冻结特征编码器的参数，不会在训练中更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wavlm.feature_extractor._freeze_parameters()

    # 冻结基础模型的参数，不会在训练中更新，只有分类头部会更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False

    # 前向传播函数，接受输入值、注意力掩码、标签等参数，并返回模型的输出
    # 使用 add_start_docstrings_to_model_forward 添加模型前向传播的文档字符串
    # 使用 add_code_sample_docstrings 添加模型前向传播的代码示例文档字符串
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 此函数定义了一个模型的前向传播过程
    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        """
        # 该函数接受以下参数:
        # - input_values: 输入值
        # - attention_mask: 注意力掩码, 用于指定需要关注的部分
        # - labels: 标签, 用于计算分类/回归损失
        # - output_attentions: 是否输出注意力权重
        # - output_hidden_states: 是否输出隐藏状态
        # - return_dict: 是否返回字典格式的输出
    
        # 如果 return_dict 为 None, 则使用 config.use_return_dict 作为默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 如果启用了加权层求和, 则输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
    
        # 通过 wavlm 模型获得输出
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 如果启用了加权层求和, 则计算加权平均的隐藏状态
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        # 否则直接使用第一个隐藏状态
        else:
            hidden_states = outputs[0]
    
        # 通过分类器获得逻辑值
        logits = self.classifier(hidden_states)
    
        # 如果提供了标签, 则计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))
    
        # 如果不使用字典格式的输出, 则返回逻辑值和其他输出
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output
    
        # 否则返回 TokenClassifierOutput, 包含损失, 逻辑值, 隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从 transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss 复制而来的类 AMSoftmaxLoss，用于AM Softmax损失计算
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss


# 从 transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer 复制而来的类 TDNNLayer，用于时延神经网络（TDNN）操作
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(1)
        # 将hidden_states展开成卷积特征，用于下一步操作
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states


# 添加了有关 Speaker Verification 任务的 XVector 特征提取头部的 WavLM 模型
@add_start_docstrings(
    """
    WavLM Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    WAVLM_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector 复制而来的类 WavLMForXVector，用于Speaker Verification任务的WavLM模型
class WavLMForXVector(WavLMPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化对象
        super().__init__(config)

        # 初始化 WavLMModel 对象
        self.wavlm = WavLMModel(config)
        # 计算变换器层 + 输入嵌入层的总层数
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        # 如果使用加权层求和，初始化层权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 初始化投影层，将隐藏状态映射到 TDNN 输入维度
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 初始化 TDNN 层
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 初始化特征提取器，将 TDNN 层输出映射到 x-vector 维度
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        # 初始化分类器，将 x-vector 映射到标签数维度
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 初始化损失函数，AMSoftmaxLoss
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 初始化权重
        self.init_weights()

    # 冻结特征提取器的梯度计算
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 冻结特征编码器的梯度计算
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wavlm.feature_extractor._freeze_parameters()

    # 冻结基础模型的梯度计算
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 将基础模型的参数梯度计算关闭
        for param in self.wavlm.parameters():
            param.requires_grad = False

    # 获取 TDNN 层输出长度
    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层的输出长度计算公式参考自 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        # 计算每个 TDNN 层的输出长度
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    # 添加文档字符串注释和示例代码注释
    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_XVECTOR_CHECKPOINT,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_XVECTOR_EXPECTED_OUTPUT,
    )
    # forward方法用于执行模型的前向传播
    def forward(
        # 输入值，类型为torch.Tensor，可选参数
        input_values: Optional[torch.Tensor],
        # 注意力掩码，类型为torch.Tensor，可选参数，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，类型为bool，可选参数，默认为None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为bool，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,
        # 返回类型，类型为bool，可选参数，默认为None
        return_dict: Optional[bool] = None,
        # 标签值，类型为torch.Tensor，可选参数，默认为None
        labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果return_dict为None，则使用self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        # 如果self.config.use_weighted_layer_sum为True，则设置为True，否则使用output_hidden_states

        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 使用输入数值、关注力掩码、输出注意力、输出隐藏状态、返回值形式来调用self.wavlm函数

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            # 如果self.config.use_weighted_layer_sum为True，则获取输出的隐藏状态并按指定维度叠加
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 对层权重进行softmax处理
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            # 将隐藏状态与归一化权重相乘然后按指定维度求和
        else:
            hidden_states = outputs[0]
            # 否则获取输出的第一个元素作为隐藏状态

        hidden_states = self.projector(hidden_states)
        # 使用self.projector函数对隐藏状态进行处理

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)
            # 对每个tdnn_layer层进行处理

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
            # 如果关注力掩码为None，则对隐藏状态进行均值和标准差计算
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            # 否则调用_get_feat_extract_output_lengths和_get_tdnn_output_lengths函数来获取特征提取器输出长度和tdnn输出长度
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
                # 遍历计算各个长度的均值和标准差
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
            # 将均值和标准差堆叠
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)
        # 将均值和标准差连接在一起

        output_embeddings = self.feature_extractor(statistic_pooling)
        # 使用self.feature_extractor函数对统计汇总进行特征提取
        logits = self.classifier(output_embeddings)
        # 使用self.classifier函数对输出嵌入进行分类

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)
            # 如果存在标签，则使用self.objective计算损失

        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
            # 如果不需要返回字典，则返回输出元组和隐藏状态列表
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回XVectorOutput对象，包括损失、logits、嵌入、隐藏状态和关注力
```