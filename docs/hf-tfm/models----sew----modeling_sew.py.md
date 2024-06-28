# `.\models\sew\modeling_sew.py`

```py
# coding=utf-8
# 版权所有 2021 年 ASAPP 公司和 HuggingFace 公司团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证的条款，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch SEW 模型。"""

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

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 1

# 用于文档的通用字符串
_CONFIG_FOR_DOC = "SEWConfig"

# 用于基本文档的检查点字符串
_CHECKPOINT_FOR_DOC = "asapp/sew-tiny-100k-ft-ls100h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 512]

# CTC（Connectionist Temporal Classification）文档字符串
_CTC_EXPECTED_OUTPUT = (
    "'MISTER QUILTER IS THE APPOSTILE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPOLLE'"
)
_CTC_EXPECTED_LOSS = 0.42

# 音频分类文档字符串
_SEQ_CLASS_CHECKPOINT = "anton-l/sew-mid-100k-ft-keyword-spotting"
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 9.52

SEW_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "asapp/sew-tiny-100k",
    "asapp/sew-small-100k",
    "asapp/sew-mid-100k",
    # 查看所有 SEW 模型 https://huggingface.co/models?filter=sew
]

# 从 transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices 复制而来
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算给定形状的随机掩码间隔。用于实现 ASR 的 [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779)。请注意，此方法未经优化，应在 CPU 上作为训练预处理的一部分运行，而不是在 TPU 上运行。

    """
    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    # 解包形状参数
    batch_size, sequence_length = shape

    # 检查 mask_length 是否合法
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    # 检查 mask_length 是否小于 sequence_length
    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon 用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 计算应该屏蔽的 span 的数量
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        # 确保不低于最小屏蔽数
        num_masked_span = max(num_masked_span, min_masks)

        # 确保 num_masked_span 不超过 sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保 num_masked span 不超过 input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算每个 batch 中的屏蔽 span 的数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # 创建用于 SpecAugment 的屏蔽 mask
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    # 计算最大可能的屏蔽 span 数量
    max_num_masked_span = compute_num_masked_span(sequence_length)

    # 如果最大屏蔽 span 数量为 0，则直接返回空的 spec_aug_mask
    if max_num_masked_span == 0:
        return spec_aug_mask
    # 遍历输入长度列表中的每个长度
    for input_length in input_lengths:
        # 计算当前输入长度下的需要屏蔽的片段数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要屏蔽的索引
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个样本索引作为填充向量的虚拟索引，确保所有批次具有相同的维度
        # 这是由于概率舍入导致的维度问题的解决方案
        if len(spec_aug_mask_idx) == 0:
            # 只有在 `input_length` 严格小于 `sequence_length` 时才会发生这种情况，
            # 此时最后一个标记必须是填充标记，可以用作虚拟屏蔽标识符
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟屏蔽索引扩展到匹配的数量，并添加到屏蔽索引列表中
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将屏蔽索引列表转换为 numpy 数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将屏蔽索引扩展为屏蔽段
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量到起始索引，以确保索引现在创建一个完整的屏蔽段
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保屏蔽索引不会超过序列长度
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 使用屏蔽索引在 spec_aug_mask 上进行填充操作
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回填充后的 spec_aug_mask
    return spec_aug_mask
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->SEW
class SEWNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为上一层的卷积维度或者默认为1（如果是第一层）
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个卷积层，指定输入和输出维度，卷积核大小，步长和是否有偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数为预定义的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 执行卷积操作
        hidden_states = self.conv(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->SEW
class SEWLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为上一层的卷积维度或者默认为1（如果是第一层）
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个卷积层，指定输入和输出维度，卷积核大小，步长和是否有偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一个LayerNorm层，对输出进行标准化，并可选地进行仿射变换
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 设置激活函数为预定义的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 执行卷积操作
        hidden_states = self.conv(hidden_states)

        # 将卷积输出的维度转置以便进行LayerNorm操作
        hidden_states = hidden_states.transpose(-2, -1)
        # 应用LayerNorm进行标准化
        hidden_states = self.layer_norm(hidden_states)
        # 再次将维度转置回来
        hidden_states = hidden_states.transpose(-2, -1)

        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->SEW
class SEWGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为上一层的卷积维度或者默认为1（如果是第一层）
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个卷积层，指定输入和输出维度，卷积核大小，步长和是否有偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数为预定义的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个GroupNorm层，指定组数和通道数，对输出进行标准化
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 执行卷积操作
        hidden_states = self.conv(hidden_states)
        # 应用GroupNorm进行标准化
        hidden_states = self.layer_norm(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


class SEWPositionalConvEmbedding(nn.Module):
    # 在此处继续实现其他功能
    pass
    # 初始化函数，用于初始化类的实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个一维卷积层对象
        self.conv = nn.Conv1d(
            config.hidden_size,  # 输入通道数（隐藏大小）
            config.hidden_size,  # 输出通道数（隐藏大小，保持不变）
            kernel_size=config.num_conv_pos_embeddings,  # 卷积核大小
            padding=config.num_conv_pos_embeddings // 2,  # 填充大小
            groups=config.num_conv_pos_embedding_groups,  # 分组卷积的组数
            stride=config.squeeze_factor,  # 步长
        )

        # 如果启用了Deepspeed的zero3模式
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # 使用Deepspeed的分布式参数收集功能
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                # 对卷积层进行权重归一化，并命名为"weight"，dim=2表示在输出通道维度上进行归一化
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            # 注册卷积层权重的外部参数
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 对卷积层进行权重归一化，并命名为"weight"，dim=2表示在输出通道维度上进行归一化
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        # 创建一个与卷积层同样大小的填充层对象
        self.padding = SEWSamePadLayer(config.num_conv_pos_embeddings)
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数，定义了数据如何通过网络层流动
    def forward(self, hidden_states):
        # 经过一维卷积层
        hidden_states = self.conv(hidden_states)
        # 经过填充层
        hidden_states = self.padding(hidden_states)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer 复制代码，并将 Wav2Vec2 更改为 SEW
class SEWSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 根据卷积位置嵌入的数量确定是否需要移除一个填充元素
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            # 如果需要移除填充元素，则截取掉隐藏状态的末尾
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder 复制代码，并将 Wav2Vec2 更改为 SEW
class SEWFeatureEncoder(nn.Module):
    """从原始音频波形中构造特征"""

    def __init__(self, config):
        super().__init__()

        # 根据配置选择不同的特征提取归一化方式
        if config.feat_extract_norm == "group":
            # 如果是组归一化，则使用 SEWGroupNormConvLayer 作为第一层，其余层为 SEWNoLayerNormConvLayer
            conv_layers = [SEWGroupNormConvLayer(config, layer_id=0)] + [
                SEWNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果是层归一化，则所有层均使用 SEWLayerNormConvLayer
            conv_layers = [SEWLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            # 如果归一化方式不是 'group' 或 'layer'，则抛出异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )

        # 将所有的卷积层组成一个模块列表
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        # 冻结模型的所有参数，使其不可训练
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False
    def forward(self, input_values):
        # 将输入的张量扩展维度，增加一个维度，用于卷积操作
        hidden_states = input_values[:, None]

        # 如果需要计算梯度并且处于训练模式，则将 hidden_states 设置为需要梯度计算
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有的卷积层进行前向传播
        for conv_layer in self.conv_layers:
            # 如果需要计算梯度、启用了梯度检查点功能并且处于训练模式，则使用梯度检查点函数进行前向传播
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则直接调用卷积层进行前向传播计算
                hidden_states = conv_layer(hidden_states)

        # 返回最终的隐藏状态张量
        return hidden_states
class SEWFeatureExtractor(SEWFeatureEncoder):
    # SEWFeatureExtractor 类继承自 SEWFeatureEncoder 类

    def __init__(self, config):
        # 初始化函数，接受一个配置参数 config
        super().__init__(config)
        # 调用父类 SEWFeatureEncoder 的初始化方法

        # 发出警告，提示 SEWFeatureExtractor 类已过时，建议使用 SEWFeatureEncoder 类
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 从 transformers.models.bart.modeling_bart.BartAttention 复制并修改为 SEWAttention
class SEWAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        # 初始化函数，定义注意力机制的参数
        super().__init__()
        self.embed_dim = embed_dim  # 注意力机制的输入维度
        self.num_heads = num_heads  # 注意力头的数量
        self.dropout = dropout  # Dropout 概率
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.config = config  # SEW 的配置对象

        # 检查 embed_dim 是否可以被 num_heads 整除，否则抛出错误
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否是因果的

        # 线性变换层，用于计算 Q、K、V 矩阵
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入张量 tensor 重新形状为 (bsz, seq_len, num_heads, head_dim) 并转置
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 前向传播函数，接受多个输入参数并进行注意力计算
        pass  # 此处未实现具体功能，需要根据具体的注意力机制进行实现


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward 复制并修改为 SEWFeedForward
class SEWFeedForward(nn.Module):
    def __init__(self, config):
        # 初始化函数，接受一个配置参数 config
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 中间层线性变换，用于激活函数前的线性变换
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 输出层线性变换，用于最终输出
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)
    # 定义前向传播函数，接受隐藏状态作为输入参数
    def forward(self, hidden_states):
        # 使用中间层的稠密层对隐藏状态进行变换
        hidden_states = self.intermediate_dense(hidden_states)
        # 对变换后的隐藏状态应用中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对应用激活函数后的隐藏状态进行中间层的dropout操作
    
        # 使用输出层的稠密层对经过中间层处理后的隐藏状态再次进行变换
        hidden_states = self.output_dense(hidden_states)
        # 对输出层变换后的隐藏状态进行输出层的dropout操作
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer复制而来，将Wav2Vec2替换为SEW
class SEWEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化注意力机制，使用SEWAttention类
        self.attention = SEWAttention(
            embed_dim=config.hidden_size,  # 设置嵌入维度为隐藏大小
            num_heads=config.num_attention_heads,  # 设置注意力头数
            dropout=config.attention_dropout,  # 设置注意力机制的dropout率
            is_decoder=False,
        )
        # 随机失活层，使用隐藏dropout率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 层归一化，使用隐藏大小和层归一化epsilon值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 前馈神经网络，使用SEWFeedForward类
        self.feed_forward = SEWFeedForward(config)
        # 最终层归一化，使用隐藏大小和层归一化epsilon值
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 注意力残差连接
        attn_residual = hidden_states
        # 执行注意力计算
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 应用dropout
        hidden_states = self.dropout(hidden_states)
        # 添加注意力残差到隐藏状态
        hidden_states = attn_residual + hidden_states

        # 应用层归一化
        hidden_states = self.layer_norm(hidden_states)
        # 添加前馈神经网络输出到隐藏状态
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SEWEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # SEW位置卷积嵌入
        self.pos_conv_embed = SEWPositionalConvEmbedding(config)
        # 平均池化层，使用squeeze因子配置
        self.pool = nn.AvgPool1d(config.squeeze_factor, config.squeeze_factor)
        # 层归一化，使用隐藏大小和层归一化epsilon值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 随机失活层，使用隐藏dropout率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # SEW编码器层列表，根据隐藏层数配置
        self.layers = nn.ModuleList([SEWEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # SEW上采样
        self.upsample = SEWUpsampling(config)
        # 梯度检查点，默认为关闭
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
):
    pass  # 此处省略forward方法实现，仅给出了方法签名

class SEWPreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和预训练模型下载加载的抽象类。
    """

    config_class = SEWConfig  # SEW模型的配置类
    base_model_prefix = "sew"  # 基础模型前缀
    main_input_name = "input_values"  # 主输入名称
    supports_gradient_checkpointing = True  # 支持梯度检查点
    def _init_weights(self, module):
        """Initialize the weights"""  # 定义一个初始化权重的函数，参数是一个神经网络模块
        if isinstance(module, SEWPositionalConvEmbedding):
            # 对 SEWPositionalConvEmbedding 类型的模块，使用正态分布初始化卷积层的权重，均值为 0，标准差为根据核大小和输入通道数计算的值
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积层的偏置初始化为 0
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            # 对线性层，使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 对 LayerNorm 和 GroupNorm 模块，偏置初始化为 0，权重初始化为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            if is_deepspeed_zero3_enabled():
                import deepspeed  # 导入 deepspeed 库
                # 如果启用了 DeepSpeed 的 Zero-3，且卷积层有 weight_v 和 weight_g 属性，使用 GatheredParameters 初始化权重
                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                # 否则使用 He 初始化方法，适用于 ReLU 激活函数
                nn.init.kaiming_normal_(module.weight.data)

        # 对于 Linear 和 Conv1d 类型的模块，且具有偏置的，偏置初始化为 0
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """  # 定义一个计算卷积层输出长度的函数，输入是一个长度张量或整数
        def _conv_out_length(input_length, kernel_size, stride):
            # 计算 1D 卷积层输出长度的公式，取自 PyTorch 文档
            # input_length 是输入长度，kernel_size 是卷积核大小，stride 是步幅
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历配置中的卷积核大小和步幅，依次计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 返回计算得到的输出长度
        return input_lengths
    # 定义一个方法用于生成特征向量的注意力掩码
    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # 计算输出长度，即根据注意力掩码每个样本的有效长度来确定输出的长度
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        # 获取批次大小
        batch_size = attention_mask.shape[0]

        # 初始化一个全零的注意力掩码张量，形状为(batch_size, feature_vector_length)
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        
        # 将输出长度前的所有位置设为1，以确保在这些位置之前的所有值都被注意到
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        
        # 将注意力掩码张量沿最后一个维度翻转，累加并再次翻转，并转换为布尔类型，以生成正确的注意力掩码
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        
        # 返回生成的注意力掩码张量
        return attention_mask
"""
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
@add_start_docstrings(
    "The bare SEW Model transformer outputting raw hidden-states without any specific head on top.",
    SEW_START_DOCSTRING,
)
class SEWModel(SEWPreTrainedModel):
    # 初始化函数，接收一个SEWConfig类型的配置对象作为参数
    def __init__(self, config: SEWConfig):
        # 调用父类的初始化方法，传入配置对象作为参数
        super().__init__(config)
        # 将配置对象保存到实例变量self.config中
        self.config = config
        # 使用配置对象创建SEWFeatureEncoder类型的特征提取器实例，并保存到self.feature_extractor中
        self.feature_extractor = SEWFeatureEncoder(config)
        # 创建一个LayerNorm层，用于归一化最后一个卷积层的输出，参数为config.conv_dim[-1]，eps为config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)

        # 判断是否需要对特征进行投影
        self.project_features = config.conv_dim[-1] != config.hidden_size
        if self.project_features:
            # 如果需要投影，创建一个Linear层，将config.conv_dim[-1]维的特征投影到config.hidden_size维
            self.feature_projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 创建一个Dropout层，用于特征投影后的dropout，dropout率为config.feat_proj_dropout
        self.feature_dropout = nn.Dropout(config.feat_proj_dropout)

        # 如果配置中mask_time_prob或mask_feature_prob大于0，创建一个参数化的特征嵌入张量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 使用配置对象创建SEWEncoder类型的编码器实例，并保存到self.encoder中
        self.encoder = SEWEncoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states复制而来
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
            # apply SpecAugment along time axis using computed mask indices
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
            # expand feature mask indices to match the shape of hidden_states and apply
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states
        ) -> Union[Tuple, BaseModelOutput]:
        # 如果未提供输出注意力机制，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未提供输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供返回字典选项，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取输入特征
        extract_features = self.feature_extractor(input_values)
        # 调整特征的维度顺序
        extract_features = extract_features.transpose(1, 2)
        # 应用层归一化到特征
        extract_features = self.layer_norm(extract_features)

        # 如果需要对特征进行投影
        if self.project_features:
            extract_features = self.feature_projection(extract_features)
        # 使用特征丢弃层处理特征
        hidden_states = self.feature_dropout(extract_features)

        # 如果提供了注意力掩码
        if attention_mask is not None:
            # 计算与特征向量对应的减少注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        # 对隐藏状态进行掩码处理，根据时间索引进行掩码
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # 编码器处理隐藏状态
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 取编码器输出的第一个元素作为隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 返回元组形式的结果，包含隐藏状态和额外的编码器输出
            return (hidden_states,) + encoder_outputs[1:]

        # 返回基础模型输出对象，包含最后的隐藏状态、所有隐藏状态和注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """SEW Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    SEW_START_DOCSTRING,
)
# 使用`add_start_docstrings`装饰器添加模型文档字符串，描述SEW模型具有一个在顶部的语言建模头用于连接主义时间分类(CTC)。
# SEW_START_DOCSTRING为预定义的模型开始文档字符串。

# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC类复制而来，修改Wav2Vec2为SEW，wav2vec2为sew，WAV_2_VEC_2为SEW
class SEWForCTC(SEWPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        # 初始化SEW模型
        self.sew = SEWModel(config)
        # 使用config中的final_dropout创建一个dropout层
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        # 如果config中未定义语言模型头的词汇量大小，则抛出异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `SEWForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        # 根据config的设置选择输出隐藏大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 创建一个线性层作为语言模型头，连接隐藏大小和词汇量大小
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        """
        重写`~PreTrainedModel.tie_weights`方法，以便在通过`from_pretrained(...)`传递`target_lang=...`时正确加载适配器权重。

        该方法不应由用户调用，并且可能在未来发生更改。
        """

        # 注意，`tie_weights`通常用于绑定输入和输出嵌入权重。在这里重新用于SEW，以便我们在SEW加载适配器层时不必引入新的API。
        # 虽然有些巧妙，SEW永远不必绑定输入和输出嵌入，因此在这里重新使用该函数是可以的。
        target_lang = self.target_lang

        # 如果target_lang不为None，并且config中未定义adapter_attn_dim，则抛出异常
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果target_lang为None，并且config中定义了adapter_attn_dim，则记录信息提示，默认将target_lang设置为'eng'
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果target_lang不为None，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不会更新
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 发出警告，提醒方法`freeze_feature_extractor`已过时，并将在 Transformers v5 中移除。请使用等效的`freeze_feature_encoder`方法。
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用等效的`freeze_feature_encoder`方法，冻结特征编码器的参数
        self.freeze_feature_encoder()

    # 调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不会更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数
        self.sew.feature_extractor._freeze_parameters()

    # 调用此函数将禁用基础模型的梯度计算，使其参数在训练过程中不会更新。只有分类头部会更新。
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历self.sew对象的参数，并设置requires_grad为False，以禁用梯度计算
        for param in self.sew.parameters():
            param.requires_grad = False

    # 使用add_start_docstrings_to_model_forward和add_code_sample_docstrings为模型的forward方法添加文档字符串
    @add_start_docstrings_to_model_forward(SEW_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    # 定义模型的forward方法，用于执行前向传播
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入值，类型为可选的torch.Tensor
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，类型为可选的torch.Tensor，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为可选的bool，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的bool，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，类型为可选的bool，默认为None
        labels: Optional[torch.Tensor] = None,  # 标签，类型为可选的torch.Tensor，默认为None
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # Decide whether to return a dictionary or not based on the provided argument or configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform sequence to sequence processing using the model's encoder-decoder structure
        outputs = self.sew(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states from the model's output and apply dropout regularization
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Generate logits from the processed hidden states using the language model head
        logits = self.lm_head(hidden_states)

        # Initialize loss as None; compute CTC (Connectionist Temporal Classification) loss if labels are provided
        loss = None
        if labels is not None:
            # Validate label values to ensure they are within the vocabulary size
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # Calculate input lengths based on the attention mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # Determine which labels are valid and compute target lengths
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # Apply log softmax to logits and transpose dimensions for CTC loss computation
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # Compute CTC loss with adjustments for padding and configuration settings
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

        # If return_dict is False, format the output tuple accordingly
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, construct and return a CausalLMOutput object with relevant attributes
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# SEW 模型，顶部带有一个序列分类头（在汇总输出之上的线性层），用于诸如 SUPERB 关键词识别等任务。
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification 复制而来，将 Wav2Vec2 改为 SEW，wav2vec2 改为 sew，WAV_2_VEC_2 改为 SEW。
class SEWForSequenceClassification(SEWPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 如果配置允许添加适配器且为真，则引发值错误，因为序列分类不支持使用 SEW 适配器（config.add_adapter=True）。
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of SEW adapters (config.add_adapter=True)"
            )
        
        # 创建 SEW 模型对象
        self.sew = SEWModel(config)
        
        # 计算层数：变压器层数 + 输入嵌入层
        num_layers = config.num_hidden_layers + 1
        
        # 如果配置使用加权层求和，则初始化层权重参数为均匀值
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 线性投影层，将隐藏状态大小映射到分类器投影大小
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        
        # 分类器层，将分类器投影大小映射到类别数量
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征提取器，不再计算特征编码器的梯度，使其在训练期间不更新
    def freeze_feature_extractor(self):
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 冻结特征编码器，不再计算特征编码器的梯度，使其在训练期间不更新
    def freeze_feature_encoder(self):
        self.sew.feature_extractor._freeze_parameters()

    # 冻结基础模型，不再计算基础模型的梯度，使其在训练期间不更新，仅更新分类头
    def freeze_base_model(self):
        for param in self.sew.parameters():
            param.requires_grad = False

    # 在模型前向方法上添加文档字符串注释，详细描述输入和输出
    @add_start_docstrings_to_model_forward(SEW_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_SEQ_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 设置是否返回字典形式的输出，默认从模型配置中获取
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 根据配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用序列编码器模块进行前向传播
        outputs = self.sew(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置要求使用加权层求和，则进行相应的处理
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # 将处理后的隐藏状态传递给投影层进行处理
        hidden_states = self.projector(hidden_states)
        
        # 如果没有提供注意力掩码，则对隐藏状态进行均值池化
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            # 根据注意力掩码生成填充掩码，并对隐藏状态进行相应的处理
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 对池化后的输出应用分类器以获得最终的分类预测
        logits = self.classifier(pooled_output)

        # 如果提供了标签，则计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 根据是否返回字典形式的输出进行结果的返回
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器的输出，包括损失、预测 logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```