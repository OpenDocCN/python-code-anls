# `.\transformers\models\sew\modeling_sew.py`

```
# coding=utf-8
# Copyright 2021 ASAPP Inc. and the HuggingFace Inc. team. All rights reserved.
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
# 该代码文件定义了 PyTorch 实现的 SEW (Speech Encoder with Transformers) 模型

import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sew import SEWConfig


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 1

# General docstring
_CONFIG_FOR_DOC = "SEWConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "asapp/sew-tiny-100k-ft-ls100h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 512]

# CTC docstring
_CTC_EXPECTED_OUTPUT = (
    "'MISTER QUILTER IS THE APPOSTILE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPOLLE'"
)
_CTC_EXPECTED_LOSS = 0.42

# Audio class docstring
_SEQ_CLASS_CHECKPOINT = "anton-l/sew-mid-100k-ft-keyword-spotting"
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 9.52

SEW_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "asapp/sew-tiny-100k",
    "asapp/sew-small-100k",
    "asapp/sew-mid-100k",
    # See all SEW models at https://huggingface.co/models?filter=sew
]


# 根据给定的形状、掩码概率、掩码长度等参数计算随机遮蔽区域的索引
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算随机遮蔽区域的索引。用于实现 [SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)。
    注意, 这个方法并不适合在 TPU 上运行, 应该在 CPU 上作为预处理过程的一部分运行。
    """
    Args:
        shape: 要计算掩码的形状。这应该是一个大小为2的元组，其中第一个元素是批大小，第二个元素是要跨越的轴的长度。
        mask_prob: 将被掩盖的整个轴的百分比（介于0和1之间）。通过`mask_prob*shape[1]/mask_length`计算生成的长度为`mask_length`的独立生成掩码跨度的数量。由于重叠，`mask_prob`是一个上限，实际百分比将更小。
        mask_length: 掩码的大小
        min_masks: 掩码跨度的最小数量
        attention_mask: 一个（右填充）注意力掩码，独立缩短每个批维度的特征轴。
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length`必须大于0。")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length`必须小于`sequence_length`，但是得到`mask_length`：{mask_length}"
            f"和`sequence_length`：{sequence_length}`"
        )

    # epsilon用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """给定输入长度，计算应掩盖多少跨度"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保掩码跨度数<=序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保num_masked span也<= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批次中掩盖的跨度数
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment掩码填充
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask
```  
    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算该输入的被遮罩的跨度数量
        num_masked_span = compute_num_masked_span(input_length)

        # 获取随机索引以进行遮罩
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个采样索引作为填充向量的虚拟索引，以确保由于概率舍入而使所有批次具有相同的维度
        # 选择第一个样本只是两次填充这些向量。
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只会发生在`input_length`严格小于`sequence_length`的情况下，
            # 此时最后一个标记必须是填充标记，我们可以将其用作虚拟遮罩ID
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮罩索引扩展为遮罩跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量到起始索引，使索引现在创建一个跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保我们不能有大于sequence_length的索引
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将索引散布到遮罩
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer复制代码，并将Wav2Vec2更改为SEW
class SEWNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，指定输入和输出维度、卷积核大小、步长和是否包含偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 激活函数为config中指定的特征提取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的结果应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer复制代码，并将Wav2Vec2更改为SEW
class SEWLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，指定输入和输出维度、卷积核大小、步长和是否包含偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一个LayerNorm层，指定归一化的维度和是否包含可学习的仿射变换
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 激活函数为config中指定的特征提取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        # 转置hidden_states的倒数第二维和倒数第一维
        hidden_states = hidden_states.transpose(-2, -1)
        # 对转置后的hidden_states应用LayerNorm
        hidden_states = self.layer_norm(hidden_states)
        # 再次转置hidden_states的倒数第二维和倒数第一维
        hidden_states = hidden_states.transpose(-2, -1)

        # 对LayerNorm后的结果应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer复制代码，并将Wav2Vec2更改为SEW
class SEWGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，指定输入和输出维度、卷积核大小、步长和是否包含偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 激活函数为config中指定的特征提取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个GroupNorm层，指定组数和通道数，并包含可学习的仿射变换
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的结果应用GroupNorm
        hidden_states = self.layer_norm(hidden_states)
        # 对GroupNorm后的结果应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


class SEWPositionalConvEmbedding(nn.Module):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个一维卷积层，设置输入和输出通道数、卷积核大小、填充、分组和步长
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
            stride=config.squeeze_factor,
        )

        # 如果使用了深度加速库的零3功能
        if is_deepspeed_zero3_enabled():
            # 导入深度加速库
            import deepspeed
            # 使用 GatheredParameters 将权重参数聚合到第0个修饰器等级
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                # 对卷积层进行权重归一化
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            # 注册卷积层的权重参数到外部参数
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 对卷积层进行权重归一化
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        # 创建一个填充层
        self.padding = SEWSamePadLayer(config.num_conv_pos_embeddings)
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数
    def forward(self, hidden_states):
        # 经过卷积层
        hidden_states = self.conv(hidden_states)
        # 经过填充层
        hidden_states = self.padding(hidden_states)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer复制代码，并将Wav2Vec2更改为SEW
class SEWSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 如果卷积位置嵌入数量为偶数，则num_pad_remove为1，否则为0
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果num_pad_remove大于0，则移除hidden_states的最后self.num_pad_remove个元素
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        return hidden_states


class SEWUpsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将隐藏状态投影到更大的维度
        self.projection = nn.Linear(config.hidden_size, config.hidden_size * config.squeeze_factor)
        # 激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
        # 压缩因子
        self.squeeze_factor = config.squeeze_factor

    def forward(self, hidden_states):
        # 对隐藏状态进行投影
        hidden_states = self.projection(hidden_states)
        # 使用激活函数
        hidden_states = self.activation(hidden_states)

        if self.squeeze_factor > 1:
            # 将嵌入通道转换为序列长度
            bsz, src_len, src_embed_dim = hidden_states.size()
            tgt_len = src_len * self.squeeze_factor
            tgt_embed_dim = src_embed_dim // self.squeeze_factor
            hidden_states = hidden_states.reshape(bsz, src_len, self.squeeze_factor, tgt_embed_dim)
            hidden_states = hidden_states.reshape(bsz, tgt_len, tgt_embed_dim)

        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder复制代码，并将Wav2Vec2更改为SEW
class SEWFeatureEncoder(nn.Module):
    """从原始音频波形构建特征"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            # 如果特征提取规范为"group"，则使用SEWGroupNormConvLayer作为第一层，后续使用SEWNoLayerNormConvLayer
            conv_layers = [SEWGroupNormConvLayer(config, layer_id=0)] + [
                SEWNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果特征提取规范为"layer"，则使用SEWLayerNormConvLayer
            conv_layers = [SEWLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            # 抛出异常，特征提取规范必须是'group'或'layer'
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        # 冻结参数，使其不可训练
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False
    # 前向传播函数，接收输入数值
    def forward(self, input_values):
        # 将输入数值转换为二维张量
        hidden_states = input_values[:, None]

        # 如果需要梯度且处于训练模式，则设置 hidden_states 需要梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历卷积层列表
        for conv_layer in self.conv_layers:
            # 如果需要梯度、开启梯度检查点且处于训练模式，则使用梯度检查点函数进行计算
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则直接调用卷积层进行计算
                hidden_states = conv_layer(hidden_states)

        # 返回最终的隐藏状态
        return hidden_states
# 定义 SEWFeatureExtractor 类，继承自 SEWFeatureEncoder 类
class SEWFeatureExtractor(SEWFeatureEncoder):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 发出警告，提示该类已被弃用，将在 Transformers v5 中移除，建议使用父类的名称
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 从 transformers.models.bart.modeling_bart.BartAttention 复制并修改为 SEWAttention
class SEWAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化方法，接受多个参数
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[SEWConfig] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化各种属性
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查 embed_dim 是否能被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化 Linear 层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 定义一个辅助方法 _shape
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward 复制并修改为 SEWFeedForward
class SEWFeedForward(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 Dropout 层
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 初始化 Linear 层和激活函数
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 初始化 Linear 层和 Dropout 层
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)
    # 前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用中间密集层对隐藏状态进行处理
        hidden_states = self.intermediate_dense(hidden_states)
        # 使用中间激活函数对处理后的隐藏状态进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 使用中间丢弃层对激活后的隐藏状态进行丢弃

        hidden_states = self.intermediate_dropout(hidden_states)
        # 使用输出密集层对丢弃后的隐藏状态进行处理
        hidden_states = self.output_dense(hidden_states)
        # 使用输出丢弃层对处理后的隐藏状态进行丢弃
        hidden_states = self.output_dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer复制代码，并将Wav2Vec2->SEW
class SEWEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化SEWAttention模块
        self.attention = SEWAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 初始化Dropout模块
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化LayerNorm模块
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化SEWFeedForward模块
        self.feed_forward = SEWFeedForward(config)
        # 初始化LayerNorm模块
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 保存注意力机制之前的隐藏状态
        attn_residual = hidden_states
        # 使用注意力机制处理隐藏状态
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 对处理后的隐藏状态进行Dropout
        hidden_states = self.dropout(hidden_states)
        # 将注意力机制之前的隐藏状态与处理后的隐藏状态相加
        hidden_states = attn_residual + hidden_states

        # 对相加后的隐藏状态进行LayerNorm
        hidden_states = self.layer_norm(hidden_states)
        # 将LayerNorm后的隐藏状态与FeedForward模块处理后的结果相加
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 对相加后的隐藏状态再进行LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SEWEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化SEWPositionalConvEmbedding模块
        self.pos_conv_embed = SEWPositionalConvEmbedding(config)
        # 初始化AvgPool1d模块
        self.pool = nn.AvgPool1d(config.squeeze_factor, config.squeeze_factor)
        # 初始化LayerNorm模块
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化Dropout模块
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化SEWEncoderLayer模块列表
        self.layers = nn.ModuleList([SEWEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化SEWUpsampling模块
        self.upsample = SEWUpsampling(config)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
# SEWPreTrainedModel类继承自PreTrainedModel类
class SEWPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # SEWConfig类作为配置类
    config_class = SEWConfig
    # 模型前缀为"sew"
    base_model_prefix = "sew"
    # 主输入名称为"input_values"
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是 SEWPositionalConvEmbedding 类型的模块
        if isinstance(module, SEWPositionalConvEmbedding):
            # 使用正态分布初始化卷积层的权重
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积层的偏置初始化为0
            nn.init.constant_(module.conv.bias, 0)
        # 如果是 nn.Linear 类型的模块
        elif isinstance(module, nn.Linear):
            # 使用正态分布初始化全连接层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果是 nn.LayerNorm 或 nn.GroupNorm 类型的模块
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将偏置初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
        # 如果是 nn.Conv1d 类型的模块
        elif isinstance(module, nn.Conv1d):
            # 如果启用了深度速度 zero3
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # 如果模块有 weight_v 和 weight_g 属性
                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    # 使用 kaiming_normal 初始化权重
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    # 使用 kaiming_normal 初始化权重
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                # 使用 kaiming_normal 初始化权重
                nn.init.kaiming_normal_(module.weight.data)

        # 如果是 nn.Linear 或 nn.Conv1d 类型的模块且有偏置
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            # 将偏置初始化为0
            module.bias.data.zero_()

    # 获取特征提取层的输出长度
    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        # 计算卷积层的输出长度
        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的 1D 卷积层输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历卷积核大小和步长
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            # 计算输出长度
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    # 获取特征向量的注意力掩码，根据特征向量长度和注意力掩码
    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # 根据注意力掩码的总和计算输出长度，并转换为长整型
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        # 获取批处理大小
        batch_size = attention_mask.shape[0]

        # 创建与注意力掩码相同形状的全零张量
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # 确保输出长度之前的所有值都被关注
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # 翻转张量，累积求和，再次翻转，并转换为布尔型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        # 返回注意力掩码
        return attention_mask
# SEW 模型的文档字符串，包含了模型的介绍、作者信息和参数说明
SEW_START_DOCSTRING = r"""
    SEW was proposed in [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech
    Recognition](https://arxiv.org/abs/2109.06870) by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger,
    Yoav Artzi.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SEWConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# SEW 模型的输入文档字符串，包含了输入参数的说明
SEW_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.
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

# 添加 SEW 模型的文档字符串和起始注释
@add_start_docstrings(
    "The bare SEW Model transformer outputting raw hidden-states without any specific head on top.",
    SEW_START_DOCSTRING,
)
# 定义 SEWModel 类，继承自 SEWPreTrainedModel
class SEWModel(SEWPreTrainedModel):
    # 初始化方法，接受一个 SEWConfig 实例作为参数
    def __init__(self, config: SEWConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存传入的配置信息
        self.config = config
        # 根据配置信息创建 SEWFeatureEncoder 实例，用于提取特征
        self.feature_extractor = SEWFeatureEncoder(config)
        # 创建具有指定参数的 LayerNorm 层，用于特征提取后的标准化
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
    
        # 检查是否需要特征投影到隐藏层的大小
        self.project_features = config.conv_dim[-1] != config.hidden_size
        if self.project_features:
            # 如果需要特征投影，则创建一个线性层
            self.feature_projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 创建 dropout 层，用于特征投影
        self.feature_dropout = nn.Dropout(config.feat_proj_dropout)
    
        # 如果配置了时间或特征掩码的概率，则创建一个参数张量用于掩码
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
    
        # 创建 SEWEncoder 实例，用于编码器部分
        self.encoder = SEWEncoder(config)
    
        # 初始化权重并应用最终处理
        self.post_init()
    
    # 从 Transformers 中复制的方法，用于对隐藏状态进行掩码处理
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
        # 如果配置中指定不应用 SpecAugment，则直接返回隐藏状态
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        # 生成索引并沿着时间轴应用 SpecAugment
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            # 使用给定的 mask_time_indices 沿时间轴应用 SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 如果训练模式下且配置中指定了 mask_time_prob，则生成 mask_time_indices 并应用 SpecAugment
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
            # 如果训练模式下且配置中指定了 mask_feature_prob，则生成 mask_feature_indices 并沿特征轴应用 SpecAugment
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

    @add_start_docstrings_to_model_forward(SEW_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
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
    # 定义函数签名，声明函数的输入参数和返回类型
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果输出注意力权重参数为None，则使用配置文件中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态参数为None，则使用配置文件中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典参数为None，则使用配置文件中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征向量
        extract_features = self.feature_extractor(input_values)
        # 转置特征向量的维度
        extract_features = extract_features.transpose(1, 2)
        # 对特征向量执行层归一化
        extract_features = self.layer_norm(extract_features)

        # 如果需要对特征向量进行投影
        if self.project_features:
            extract_features = self.feature_projection(extract_features)
        # 对特征向量应用特征dropout
        hidden_states = self.feature_dropout(extract_features)

        # 如果有注意力遮罩
        if attention_mask is not None:
            # 计算减少的与特征向量对应的注意力遮罩
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        # 对隐藏状态应用掩码
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # 编码器的正向传播
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果不返回字典，则返回隐藏状态和额外输出
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        # 返回BaseModelOutput对象，包含最后的隐藏状态、隐藏状态和注意力矩阵
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 给 SEW 模型添加一个 connectionist temporal classification (CTC) 的语言模型头
# 在 SEW_START_DOCSTRING 前添加注释
@add_start_docstrings(
    """SEW Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    SEW_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC 复制而来，将 Wav2Vec2 改为 SEW，wav2vec2 改为 sew，WAV_2_VEC_2 改为 SEW
class SEWForCTC(SEWPreTrainedModel):
    # 初始化方法，参数包括配置(config)和目标语言(target_lang)
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        # 初始化 SEWModel
        self.sew = SEWModel(config)
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言
        self.target_lang = target_lang

        # 如果配置中的词汇量大小未定义，则抛出数值错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `SEWForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        # 根据配置初始化输出隐藏层大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 初始化语言模型头
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 绑定权重的方法，用于正确加载适配器权重
    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # 请注意，`tie_weights` 通常用于绑定输入和输出嵌入权重。
        # 该方法被重新用于正确加载 SEW 的适配器层，以便无需为
        # [`PreTrainedModel`] 引入新 API。虽然略有 hacky，
        # SEW 从不必须绑定输入和输出嵌入，因此在这里重新用该函数是可以的。
        target_lang = self.target_lang

        # 如果目标语言不为 None，并且未定义 config.adapter_attn_dim，则抛出数值错误
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果目标语言为 None，并且 config.adapter_attn_dim 存在，则记录警告信息
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果目标语言不为 None，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 冻结特征提取器，禁用特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_extractor(self):
        # 发出警告，提醒使用者该方法即将被删除，建议使用等效的 freeze_feature_encoder 方法代替
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 freeze_feature_encoder 方法
        self.freeze_feature_encoder()

    # 冻结特征编码器，禁用特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_encoder(self):
        # 调用 _freeze_parameters 方法，禁用特征提取器的梯度计算
        self.sew.feature_extractor._freeze_parameters()

    # 冻结基础模型，禁用基础模型的梯度计算，使其参数在训练期间不会更新，只有分类头会被更新
    def freeze_base_model(self):
        # 遍历模型的参数，将其梯度计算设置为 False
        for param in self.sew.parameters():
            param.requires_grad = False

    # 重写了父类的 forward 方法，添加了输入和输出的文档字符串，并调用了装饰器函数
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
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional`):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.sew(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

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
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 引入函数和类的文档字符串，描述了 SEW 模型用于序列分类任务的概况
# 这是一个 SEW 模型，其顶部有一个序列分类头部（一个线性层在池化输出之上），用于诸如 SUPERB 关键词识别等任务
class SEWForSequenceClassification(SEWPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 检查是否存在 config.add_adapter 属性并且为 True，如果是则引发 ValueError
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of SEW adapters (config.add_adapter=True)"
            )
        
        # 创建 SEWModel 对象
        self.sew = SEWModel(config)
        # 计算变换器层的数量加上输入嵌入层的数量
        num_layers = config.num_hidden_layers + 1
        # 如果 config.use_weighted_layer_sum 为 True，则初始化一个参数为变换器层数量的张量，并且权重和为1
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建一个线性层，将隐藏状态映射到分类器的投影空间
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建一个线性层，将分类器投影到类别数量上
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 冻结特征提取器，不再计算其梯度
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        # 发出警告，表明此方法已经弃用，建议使用 `freeze_feature_encoder` 方法
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 `freeze_feature_encoder` 方法
        self.freeze_feature_encoder()

    # 冻结特征编码器，不再计算其梯度
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数，不再计算其梯度
        self.sew.feature_extractor._freeze_parameters()

    # 冻结基础模型，不再计算其梯度
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 将基础模型的参数的 requires_grad 属性设置为 False，以防止其在训练期间更新
        for param in self.sew.parameters():
            param.requires_grad = False

    # 重写了父类的 forward 方法，并添加了输入和输出的文档字符串，以及样例代码的文档字符串
    # 详细信息可参考 SEW_INPUTS_DOCSTRING、_SEQ_CLASS_CHECKPOINT、SequenceClassifierOutput、_CONFIG_FOR_DOC、_SEQ_CLASS_EXPECTED_OUTPUT、_SEQ_CLASS_EXPECTED_LOSS
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入值，可以是张量或空值
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可以是张量或空值，默认为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为空
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，默认为空
        labels: Optional[torch.Tensor] = None,  # 标签，用于计算分类/回归损失，默认为空
    ) -> Union[Tuple, SequenceClassifierOutput]:  # 返回值类型，可以是元组或SequenceClassifierOutput对象

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 确定是否使用返回字典
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states  # 根据配置确定是否输出加权隐藏状态

        outputs = self.sew(  # 调用self.sew方法，得到输出
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:  # 如果配置使用加权层求和
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态列表
            hidden_states = torch.stack(hidden_states, dim=1)  # 沿指定维度拼接隐藏状态张量
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 计算层权重的softmax值
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 加权求和隐藏状态
        else:
            hidden_states = outputs[0]  # 否则直接取第一个输出作为隐藏状态

        hidden_states = self.projector(hidden_states)  # 使用投影器对隐藏状态进行处理
        if attention_mask is None:  # 如果没有注意力遮罩
            pooled_output = hidden_states.mean(dim=1)  # 对隐藏状态进行平均池化
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)  # 获取特征向量注意力遮罩
            hidden_states[~padding_mask] = 0.0  # 将不相关部分的隐藏状态置零
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)  # 求和并归一化

        logits = self.classifier(pooled_output)  # 使用分类器生成logits

        loss = None  # 初始化损失为None
        if labels is not None:  # 如果存在标签
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))  # 计算损失

        if not return_dict:  # 如果不返回字典形式的输出
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 组合输出元组
            return ((loss,) + output) if loss is not None else output  # 如果存在损失则加入输出

        return SequenceClassifierOutput(  # 返回SequenceClassifierOutput对象
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```