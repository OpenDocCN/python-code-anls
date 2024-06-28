# `.\models\wavlm\modeling_wavlm.py`

```py
# coding=utf-8
# 版权声明
# 版权所有（c）2021年 Fairseq作者，Microsoft Research和HuggingFace Inc.团队。保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）许可;
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据“许可证”分发的软件
# 以“原样”分发，无任何明示或暗示的担保或条件。
# 有关更多详细信息，请参阅许可证。

""" PyTorch WavLM模型。"""

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
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_peft_available,
    logging,
)
from .configuration_wavlm import WavLMConfig

logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 2

# 用于文档的配置信息
_CONFIG_FOR_DOC = "WavLMConfig"

# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC（Connectionist Temporal Classification）的预期输出文本和损失值
_CTC_EXPECTED_OUTPUT = "'mister quilter is the aposle of the middle classes and we are glad to welcome his gospel'"
_CTC_EXPECTED_LOSS = 12.51

# Frame类的检查点和预期输出
_FRAME_CLASS_CHECKPOINT = "microsoft/wavlm-base-plus-sd"
_FRAME_EXPECTED_OUTPUT = [0, 0]

# Speaker Verification（说话者验证）的检查点和预期输出
_XVECTOR_CHECKPOINT = "microsoft/wavlm-base-plus-sv"
_XVECTOR_EXPECTED_OUTPUT = 0.97

# WavLM预训练模型的存档列表
WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/wavlm-base",
    "microsoft/wavlm-base-plus",
    "microsoft/wavlm-large",
    # 查看所有WavLM模型：https://huggingface.co/models?filter=wavlm
]

# 从transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices复制的函数
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算给定形状的随机掩码跨度。用于实现《SpecAugment: 一种用于ASR的简单数据增强方法》。
    请注意，此方法未经优化以在TPU上运行，应作为训练期间的预处理步骤在CPU上运行。
    """
    pass  # 此处为函数体的占位符，未实际执行任何操作
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
    # 解包 shape 参数
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

    # epsilon 用于概率性取整
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 计算应该遮罩的 span 数量
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        # 确保遮罩的 span 数量不小于 min_masks
        num_masked_span = max(num_masked_span, min_masks)

        # 确保遮罩的 span 数量不超过 sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保遮罩的 span 数量不超过 input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算每个 batch 中的实际长度列表
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # 创建一个全零的布尔掩码数组
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    # 计算最大允许的遮罩 span 数量
    max_num_masked_span = compute_num_masked_span(sequence_length)

    # 如果最大允许的遮罩 span 数量为 0，则直接返回全零的掩码数组
    if max_num_masked_span == 0:
        return spec_aug_mask
    # 遍历输入长度列表中的每个长度
    for input_length in input_lengths:
        # 计算当前输入长度下要生成的被遮盖区间数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要遮盖的起始索引
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个被抽样的索引作为填充向量的虚拟索引，确保所有批次具有相同的维度
        # 这是由于概率舍入而产生的
        if len(spec_aug_mask_idx) == 0:
            # 如果没有选择到任何索引，说明输入长度严格小于序列长度，此时最后一个标记应该是填充标记
            # 我们可以使用它作为虚拟遮盖标识符的索引
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟索引添加到遮盖索引数组中，以确保所有批次的数组长度相同
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将遮盖索引列表转换为 NumPy 数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮盖索引扩展为遮盖区间
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    # 将数组形状重新调整为批次大小乘以最大遮盖区间数乘以遮盖长度
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 对起始索引添加偏移量，以便索引现在表示一个区间
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不会超过序列长度
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 使用散点方法将索引应用到遮盖向量中
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回生成的遮盖向量
    return spec_aug_mask
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->WavLM
class WavLMNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 根据给定层编号确定输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，根据配置设定卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 使用预定义的激活函数对卷积层的输出进行激活
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将输入的隐藏状态应用到卷积层上
        hidden_states = self.conv(hidden_states)
        # 应用激活函数到卷积层的输出
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->WavLM
class WavLMLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 根据给定层编号确定输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，根据配置设定卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一个 LayerNorm 层，对卷积层的输出进行归一化处理
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 使用预定义的激活函数对卷积层的输出进行激活
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将输入的隐藏状态应用到卷积层上
        hidden_states = self.conv(hidden_states)

        # 将卷积层输出的维度转置，以便对 LayerNorm 进行处理
        hidden_states = hidden_states.transpose(-2, -1)
        # 应用 LayerNorm 对卷积层输出进行归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 再次转置回原始维度，并将处理后的结果返回
        hidden_states = hidden_states.transpose(-2, -1)

        # 应用激活函数到处理后的卷积层输出
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->WavLM
class WavLMGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 根据给定层编号确定输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，根据配置设定卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 使用预定义的激活函数对卷积层的输出进行激活
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个 GroupNorm 层，对卷积层的输出进行分组归一化处理
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 将输入的隐藏状态应用到卷积层上
        hidden_states = self.conv(hidden_states)
        # 应用 GroupNorm 对卷积层输出进行分组归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 应用激活函数到处理后的卷积层输出
        hidden_states = self.activation(hidden_states)
        return hidden_states
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding 复制而来，改名为 WavLMPositionalConvEmbedding
class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个 1D 卷积层，用于位置编码
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 设置权重归一化函数
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 如果使用了 deepspeed 的 zero3 加速，对卷积层进行特殊处理
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # 在 zero3 加速模式下，使用 GatheredParameters 对象管理权重
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            # 注册外部参数以进行 zero3 加速管理
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 普通情况下，对卷积层应用权重归一化
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建一个用于同步填充的对象
        self.padding = WavLMSamePadLayer(config.num_conv_pos_embeddings)
        # 选择激活函数，根据配置中的 feat_extract_activation 选择对应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将输入的 hidden_states 调整维度，转换为 Conv1d 的输入格式
        hidden_states = hidden_states.transpose(1, 2)

        # 应用卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积结果进行同步填充
        hidden_states = self.padding(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)

        # 调整输出维度，返回结果
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer 复制而来，改名为 WavLMSamePadLayer
class WavLMSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 根据 num_conv_pos_embeddings 的奇偶性确定需要移除的填充数
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果需要移除填充，则按照设定的数量截取隐藏状态
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder 复制而来，改名为 WavLMFeatureEncoder
class WavLMFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()

        # 根据配置文件中的特征提取归一化方式选择不同的卷积层列表
        if config.feat_extract_norm == "group":
            # 如果归一化方式是"group"，则创建包含组归一化的第一个卷积层和其余的无归一化卷积层
            conv_layers = [WavLMGroupNormConvLayer(config, layer_id=0)] + [
                WavLMNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果归一化方式是"layer"，则创建全部使用层归一化的卷积层列表
            conv_layers = [WavLMLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            # 如果归一化方式既不是"group"也不是"layer"，则抛出值错误
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )

        # 将卷积层列表转换为 nn.ModuleList 类型，使其成为 nn.Module 的一部分
        self.conv_layers = nn.ModuleList(conv_layers)

        # 设置梯度检查点为关闭状态
        self.gradient_checkpointing = False

        # 设置需要梯度计算为 True
        self._requires_grad = True

    # 冻结模型参数的方法
    def _freeze_parameters(self):
        # 遍历所有模型参数，并设置其 requires_grad 属性为 False
        for param in self.parameters():
            param.requires_grad = False
        # 将模型的 _requires_grad 属性设置为 False，表示模型参数已冻结
        self._requires_grad = False

    # 前向传播方法，接受输入值 input_values 作为参数
    def forward(self, input_values):
        # 将输入值增加一个维度，用于后续卷积操作
        hidden_states = input_values[:, None]

        # 如果模型需要梯度计算并且处于训练模式，则设置 hidden_states 的 requires_grad 为 True
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层，并应用它们到 hidden_states 上
        for conv_layer in self.conv_layers:
            # 如果模型需要梯度计算并且开启了梯度检查点并且处于训练模式，则使用梯度检查点函数处理 hidden_states
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则直接调用当前卷积层处理 hidden_states
                hidden_states = conv_layer(hidden_states)

        # 返回最终的 hidden_states，经过所有卷积层处理后的结果
        return hidden_states
class WavLMFeatureExtractor(WavLMFeatureEncoder):
    # 继承自WavLMFeatureEncoder的WavLMFeatureExtractor类的初始化方法
    def __init__(self, config):
        # 调用父类WavLMFeatureEncoder的初始化方法
        super().__init__(config)
        # 发出警告，提示该类已被弃用，并建议使用Transformers v5中的基类
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection复制并修改为WavLM
class WavLMFeatureProjection(nn.Module):
    # WavLMFeatureProjection类，继承自nn.Module
    def __init__(self, config):
        # 初始化方法
        super().__init__()
        # 使用LayerNorm进行层归一化，eps参数为配置文件中的layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 使用Linear进行特征投影，将卷积维度投影到隐藏大小，config.hidden_size为配置文件中的隐藏大小
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 使用Dropout进行特征投影的dropout，概率为config.feat_proj_dropout
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 执行前向传播
        # 对隐藏状态进行LayerNorm归一化处理
        norm_hidden_states = self.layer_norm(hidden_states)
        # 对归一化后的隐藏状态进行投影
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的结果应用Dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class WavLMAttention(nn.Module):
    """基于'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        num_buckets: int = 320,
        max_distance: int = 800,
        has_relative_position_bias: bool = True,
    ):
        # 初始化方法
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

        # 线性变换层，用于计算Q、K、V和输出的线性投影
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # GRU相对位置编码的常数项和线性变换
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        if has_relative_position_bias:
            # 如果启用相对位置偏置，则使用Embedding层
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        index=0,
        # 定义前向传播方法，接受隐藏状态、注意力掩码、位置偏置等参数
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Attention layer with relative attention"""
        # 获取输入张量的维度信息
        bsz, tgt_len, _ = hidden_states.size()

        # 如果位置偏置为None，则计算位置偏置
        if position_bias is None:
            # 计算位置偏置
            position_bias = self.compute_bias(tgt_len, tgt_len)
            # 扩展位置偏置以适应多头注意力的形状要求
            position_bias = (
                position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, tgt_len)
            )

        # 计算相对位置偏置:
        # 1) 重塑隐藏状态张量，以便将多头注意力的头部维度放在中间
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)

        # 2) 投影隐藏状态以计算相对位置偏置
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        # 将投影后的张量重塑，并对最后一个维度求和
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)

        # 3) 从投影后的隐藏状态计算位置偏置的门控值
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0

        # 4) 将门控值应用于位置偏置，计算门控位置偏置
        gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((-1, tgt_len, tgt_len))

        # 调用多头自注意力函数进行注意力计算
        attn_output, attn_weights = self.torch_multi_head_self_attention(
            hidden_states, attention_mask, gated_position_bias, output_attentions
        )

        # 返回注意力计算结果、注意力权重和位置偏置
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
        query = key = value = hidden_states.transpose(0, 1)
        # 根据注意力掩码创建键掩码，若没有注意力掩码则为None
        key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None

        # disable bias and add_zero_attn
        bias_k = bias_v = None
        add_zero_attn = False

        # PyTorch 1.3.0 has F.multi_head_attention_forward defined
        # so no problem with backwards compatibility
        # 使用 F.multi_head_attention_forward 函数进行多头注意力计算
        attn_output, attn_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            # 将三个投影的偏置连接起来作为参数传入
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
        # 调整注意力输出的维度顺序
        attn_output = attn_output.transpose(0, 1)

        if attn_weights is not None:
            # IMPORTANT: Attention weights are averaged weights
            # here which should not be the case. This is an open issue
            # on PyTorch: https://github.com/pytorch/pytorch/issues/32590
            # 对注意力权重进行处理，这里的平均权重处理可能不是理想的情况
            attn_weights = attn_weights[:, None].broadcast_to(
                attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
            )

        return attn_output, attn_weights

    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor:
        # 生成相对位置编码
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        # 使用 _relative_positions_bucket 方法将相对位置映射到桶中
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        # 将映射后的相对位置桶转移到与相对位置嵌入张量相同的设备上
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        # 获取相对位置嵌入的值并进行维度变换
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values
    # 定义一个方法，用于将相对位置转换成相对桶索引
    def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
        # 桶的数量，除以2后取整
        num_buckets = self.num_buckets // 2

        # 将相对位置是否大于0的结果转换成long类型，并乘以桶数量
        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        # 取相对位置的绝对值
        relative_positions = torch.abs(relative_positions)

        # 定义最大的精确桶数量
        max_exact = num_buckets // 2
        # 判断相对位置是否小于最大精确值
        is_small = relative_positions < max_exact

        # 如果相对位置较大，计算相对位置的大桶索引
        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
        relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
        relative_position_if_large = (max_exact + relative_positions_if_large).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # 根据 is_small 条件选择相对位置或者大桶索引，加到 relative_buckets 中
        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        # 返回相对桶索引
        return relative_buckets
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward复制而来，将Wav2Vec2替换为WavLM
class WavLMFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 创建一个线性层，将输入大小为config.hidden_size映射到config.intermediate_size
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # 根据配置选择激活函数，如果配置中指定的是字符串，使用ACT2FN字典中对应的函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 创建一个线性层，将config.intermediate_size映射回config.hidden_size
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        # 进行中间线性层的映射和激活函数处理
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        # 进行最终线性层的映射和dropout处理
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class WavLMEncoderLayer(nn.Module):
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        
        # 创建WavLMAttention层，初始化时设置了多种参数，包括注意力头数、dropout等
        self.attention = WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        
        # 创建dropout层
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # 创建LayerNorm层，用于规范化隐藏状态
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 创建WavLMFeedForward层，用于处理隐藏状态
        self.feed_forward = WavLMFeedForward(config)
        
        # 创建最终的LayerNorm层，用于规范化输出的隐藏状态
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        # 将注意力层之前的隐藏状态保存下来，用于后续的残差连接
        attn_residual = hidden_states
        
        # 使用注意力层处理隐藏状态，获取处理后的隐藏状态、注意力权重以及位置偏置
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        
        # 对处理后的隐藏状态应用dropout
        hidden_states = self.dropout(hidden_states)
        
        # 添加残差连接
        hidden_states = attn_residual + hidden_states
        
        # 对添加了注意力之后的隐藏状态进行LayerNorm规范化
        hidden_states = self.layer_norm(hidden_states)

        # 使用前馈网络处理规范化后的隐藏状态
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        
        # 对前馈网络处理后的隐藏状态再次进行LayerNorm规范化
        hidden_states = self.final_layer_norm(hidden_states)

        # 准备输出结果，包括隐藏状态和位置偏置
        outputs = (hidden_states, position_bias)

        # 如果需要输出注意力权重，则在输出结果中添加注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    # 初始化函数，用于创建一个新的WavLM模型层
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = True):
        # 调用父类初始化函数
        super().__init__()
        # 初始化注意力层，传入配置参数
        self.attention = WavLMAttention(
            embed_dim=config.hidden_size,                      # 隐藏层大小
            num_heads=config.num_attention_heads,              # 注意力头数
            dropout=config.attention_dropout,                  # 注意力层的dropout率
            num_buckets=config.num_buckets,                    # 桶的数量（用于相对位置编码）
            max_distance=config.max_bucket_distance,           # 最大桶距离（用于相对位置编码）
            has_relative_position_bias=has_relative_position_bias,  # 是否包含相对位置偏置
        )
        # 初始化dropout层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化Layer Normalization层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化前馈神经网络层
        self.feed_forward = WavLMFeedForward(config)
        # 初始化最终的Layer Normalization层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受隐藏状态作为输入，执行模型的前向计算
    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        # 保存注意力机制前的残差连接
        attn_residual = hidden_states
        # 应用Layer Normalization层
        hidden_states = self.layer_norm(hidden_states)
        # 调用注意力层的前向传播计算
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        # 应用dropout层
        hidden_states = self.dropout(hidden_states)
        # 执行残差连接
        hidden_states = attn_residual + hidden_states
        # 应用最终的Layer Normalization层
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 输出包括最终隐藏状态和位置偏置
        outputs = (hidden_states, position_bias)

        # 如果需要输出注意力权重，添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回所有输出
        return outputs
# 定义一个用于处理音频数据的编码器模型，继承自 nn.Module 类
class WavLMEncoder(nn.Module):
    # 初始化方法，接收一个配置参数 config
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 将配置参数保存到实例变量中
        self.config = config
        # 初始化位置卷积嵌入层对象，用于处理位置信息的嵌入
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        # 初始化 LayerNorm 层，用于标准化隐藏状态向量
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，用于在训练过程中进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 使用 nn.ModuleList 初始化一个包含多个 WavLMEncoderLayer 的列表
        # 每个 WavLMEncoderLayer 对象都基于相同的 config 参数，并根据其在列表中的位置决定是否使用相对位置偏置
        self.layers = nn.ModuleList(
            [WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers)]
        )
        # 初始化梯度检查点标记，默认为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收隐藏状态、注意力掩码等参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            
            # 如果存在 attention_mask，则将未填充的 token 对应的 hidden_states 置为 0
            if attention_mask is not None:
                hidden_states[~attention_mask] = 0.0
            
            # 计算位置嵌入并与 hidden_states 相加
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            # Layer normalization
            hidden_states = self.layer_norm(hidden_states)
            # Dropout
            hidden_states = self.dropout(hidden_states)
            
            # 检查是否启用了 DeepSpeed Zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
            position_bias = None
            
            # 遍历每个 Transformer 层
            for i, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                
                # 添加 LayerDrop 功能，控制层的随机丢弃
                dropout_probability = torch.rand([])
                
                # 根据 LayerDrop 的概率决定是否跳过当前层
                skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 如果启用了梯度检查点且在训练阶段，则使用梯度检查点函数
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                            position_bias,
                            output_attentions,
                        )
                    else:
                        # 否则直接调用 Transformer 层
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_bias=position_bias,
                            output_attentions=output_attentions,
                            index=i,
                        )
    
                    # 更新 hidden_states 和 position_bias
                    hidden_states, position_bias = layer_outputs[:2]
    
                # 如果跳过了当前层，则设置 layer_outputs 为 None
                if skip_the_layer:
                    layer_outputs = (None, None)
    
                # 如果需要输出注意力矩阵，则将当前层的注意力矩阵添加到 all_self_attentions 中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[2],)
    
            # 如果需要输出隐藏状态，则将最终的 hidden_states 添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 根据 return_dict 的设置返回相应的结果
            if not return_dict:
                # 如果不需要返回字典形式的输出，则返回元组
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            else:
                # 否则以 BaseModelOutput 形式返回结果
                return BaseModelOutput(
                    last_hidden_state=hidden_states,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attentions,
                )
# 定义一个稳定的层归一化的编码器类，继承自 nn.Module
class WavLMEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()  # 调用父类的初始化方法
        self.config = config  # 存储传入的配置信息

        # 初始化位置卷积嵌入层，使用给定的配置信息
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)

        # 初始化层归一化层，指定隐藏层大小和 epsilon 值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化 dropout 层，设定丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout)

        # 使用列表推导式初始化编码器层列表，每层调用 WavLMEncoderLayerStableLayerNorm 类
        # 对于第一层（i == 0），设定相对位置偏置参数为 True
        self.layers = nn.ModuleList(
            [
                WavLMEncoderLayerStableLayerNorm(config, has_relative_position_bias=(i == 0))
                for i in range(config.num_hidden_layers)
            ]
        )

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 定义前向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        # 参数列表包括隐藏状态、注意力掩码、是否输出注意力权重、是否输出隐藏状态、是否返回字典形式结果等
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # 确保填充的标记不参与注意力计算
            hidden_states[~attention_mask] = 0

        # 使用位置卷积嵌入层处理位置信息
        position_embeddings = self.pos_conv_embed(hidden_states)
        # 将位置嵌入的结果加到隐藏状态上
        hidden_states = hidden_states + position_embeddings
        # 对隐藏状态进行dropout
        hidden_states = self.dropout(hidden_states)

        # 检查是否启用了 DeepSpeed zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None

        # 迭代处理每个层
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                # 如果需要输出隐藏状态，将当前隐藏状态添加到所有隐藏状态元组中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop（参见 https://arxiv.org/abs/1909.11556 进行描述）
            dropout_probability = torch.rand([])

            # 根据 LayerDrop 的概率决定是否跳过当前层
            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 在 DeepSpeed zero3 情况下，所有 GPU 必须同步运行
                # 如果启用了梯度检查点且处于训练阶段，使用梯度检查点函数处理当前层的调用
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_bias,
                        output_attentions,
                    )
                else:
                    # 否则直接调用当前层处理隐藏状态
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        position_bias=position_bias,
                    )
                # 更新隐藏状态和位置偏置
                hidden_states, position_bias = layer_outputs[:2]

            # 如果跳过当前层，设置层输出为 None
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果需要输出自注意力权重，将当前层的自注意力权重添加到所有自注意力元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        # 对最终的隐藏状态进行 LayerNorm 处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果需要输出隐藏状态，将最终的隐藏状态添加到所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则根据需求返回相应的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回以 BaseModelOutput 形式封装的结果
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
    """
    使用 Gumbel softmax 进行向量量化。参见[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf)获取更多信息。
    """

    def __init__(self, config):
        super().__init__()
        self.num_groups = config.num_codevector_groups  # 设置编码向量组数
        self.num_vars = config.num_codevectors_per_group  # 每组编码向量的数量

        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible"
                f" by `config.num_codevector_groups` {self.num_groups} "
                "for concatenation."
            )

        # 存储码本变量（码字）
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)  # 权重投影层

        # 可以在训练中进行衰减
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs):
        """
        计算困惑度函数。
        Args:
            probs (torch.Tensor): 概率分布张量

        Returns:
            torch.Tensor: 计算得到的困惑度值
        """
        marginal_probs = probs.mean(dim=0)  # 计算边际概率
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()  # 计算困惑度
        return perplexity
    def forward(self, hidden_states):
        # 获取输入张量的批大小、序列长度和隐藏单元大小
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到代码向量维度
        hidden_states = self.weight_proj(hidden_states)
        # 将张量形状重新视图为(batch_size * sequence_length * num_groups, -1)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 使用Gumbel Softmax采样代码向量的概率，以可区分的方式
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True)
            codevector_probs = codevector_probs.type_as(hidden_states)

            # 计算困惑度
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            # 在非可区分的方式下取argmax
            # 计算硬代码向量分布（one hot）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            # 计算困惑度
            perplexity = self._compute_perplexity(codevector_probs)

        # 将codevector_probs形状重新视图为(batch_size * sequence_length, -1)
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率检索代码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        # 返回最终的codevectors和困惑度
        return codevectors, perplexity
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Adapter with Wav2Vec2->WavLM
class WavLMAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 如果输出的隐藏层大小与配置中的隐藏层大小不同，可能需要进行降维投影
        if config.output_hidden_size != config.hidden_size:
            # 创建一个线性投影层，将隐藏状态大小从隐藏层大小投影到输出隐藏层大小
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # 创建一个LayerNorm层，用于投影后的隐藏状态的归一化
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        # 创建一系列适配器层，并存储在模块列表中
        self.layers = nn.ModuleList(WavLMAdapterLayer(config) for _ in range(config.num_adapter_layers))
        # 设置层丢弃率
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # 如果存在投影层和LayerNorm层，则对隐藏状态进行投影
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 转置隐藏状态的维度，将第1和第2维互换位置
        hidden_states = hidden_states.transpose(1, 2)

        # 对每个适配器层进行迭代计算
        for layer in self.layers:
            # 随机生成一个丢弃概率
            layerdrop_prob = np.random.random()
            # 如果处于评估模式或者随机生成的概率大于层丢弃率，则应用该适配器层
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        # 再次转置隐藏状态的维度，将第1和第2维互换位置
        hidden_states = hidden_states.transpose(1, 2)
        # 返回最终的隐藏状态
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AdapterLayer with Wav2Vec2->WavLM
class WavLMAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个一维卷积层，用于适配器
        self.conv = nn.Conv1d(
            config.output_hidden_size,        # 输入通道数为输出隐藏层大小
            2 * config.output_hidden_size,    # 输出通道数为2倍的输出隐藏层大小
            config.adapter_kernel_size,       # 卷积核大小由配置定义
            stride=config.adapter_stride,     # 卷积步长由配置定义
            padding=1,                        # 填充为1
        )

    def forward(self, hidden_states):
        # 将隐藏状态输入卷积层进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 使用门控线性单元(Gated Linear Unit, GLU)激活函数进行非线性变换
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        # 返回经过卷积和GLU激活函数处理后的隐藏状态
        return hidden_states


class WavLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为WavLMConfig
    config_class = WavLMConfig
    # 基础模型前缀为"wavlm"
    base_model_prefix = "wavlm"
    # 主输入名称为"input_values"
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是 WavLMGumbelVectorQuantizer 类型，使用特殊的初始化方法
        if isinstance(module, WavLMGumbelVectorQuantizer):
            # 初始化权重矩阵的权重数据为标准正态分布
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            # 将偏置数据初始化为零
            module.weight_proj.bias.data.zero_()
            # 使用均匀分布初始化编码向量
            nn.init.uniform_(module.codevectors)
        # 如果模块是 WavLMPositionalConvEmbedding 类型，使用特定的正态分布初始化
        elif isinstance(module, WavLMPositionalConvEmbedding):
            # 使用正态分布初始化卷积核权重数据
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积层的偏置初始化为常数0
            nn.init.constant_(module.conv.bias, 0)
        # 如果模块是 WavLMFeatureProjection 类型，使用均匀分布初始化投影权重和偏置
        elif isinstance(module, WavLMFeatureProjection):
            # 计算均匀分布的上下限
            k = math.sqrt(1 / module.projection.in_features)
            # 使用均匀分布初始化投影层的权重
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            # 使用均匀分布初始化投影层的偏置
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果模块是 nn.Linear 类型，使用正态分布初始化权重，同时将偏置初始化为零
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 nn.LayerNorm 或 nn.GroupNorm 类型，将偏置初始化为零，权重初始化为1
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果模块是 nn.Conv1d 类型，使用 Kaiming 正态分布初始化权重
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                # 计算均匀分布的上下限
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                # 使用均匀分布初始化卷积层的偏置
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """
        # 如果未指定 add_adapter，则使用配置中的默认值
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 根据 PyTorch 文档计算一维卷积层的输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 根据配置中的卷积核大小和步长计算每个卷积层的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要添加适配器，根据配置中的适配器层数计算适配器的输出长度
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        """Compute attention mask for feature vectors"""
        # 此方法计算用于特征向量的注意力掩码，输入参数包括特征向量的长度和注意力掩码张量
        # 计算非填充部分的长度，相当于 attention_mask.sum(-1)，但不进行原地操作以便在推断模式下运行
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 根据非填充长度获取特征提取器的输出长度，可以选择添加适配器
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        # 获取批次大小
        batch_size = attention_mask.shape[0]

        # 创建一个全零的注意力掩码张量，形状为 (batch_size, feature_vector_length)，与输入的注意力掩码相同的数据类型和设备
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 设置输出长度前的所有位置为 1，确保这些位置上的值被完全注意到
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1

        # 将注意力掩码进行翻转，累积求和，并再次翻转，最终转换为布尔类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        # 返回最终的注意力掩码张量
        return attention_mask
# WAVLM_START_DOCSTRING 变量，包含了关于 WavLM 模型的详细介绍和引用的论文信息
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
# WAVLM_INPUTS_DOCSTRING 变量，此处还未添加具体的文档字符串内容
WAVLM_INPUTS_DOCSTRING = r"""
"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            # 输入的原始语音波形的浮点值。可以通过加载 `.flac` 或 `.wav` 音频文件得到一个 `List[float]` 或 `numpy.ndarray` 类型的数组。
            # 使用 `AutoProcessor` 进行填充并转换为 `torch.FloatTensor` 类型的张量。详见 [`Wav2Vec2Processor.__call__`]。

        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于避免在填充标记索引上执行卷积和注意力操作。遮罩中的值选择在 `[0, 1]` 范围内：

            # - 1 表示**未遮罩**的标记，
            # - 0 表示**已遮罩**的标记。

            # [什么是注意力遮罩?](../glossary#attention-mask)

            # <Tip warning={true}>
            # 如果相应的处理器具有 `config.return_attention_mask == True`，则应传递 `attention_mask`。对于所有处理器的配置中，`config.return_attention_mask == False` 的模型，在进行批处理推断时应避免传递 `attention_mask` 以避免性能下降。对于这些模型，`input_values` 应仅填充为 0 并传递而不传递 `attention_mask`。请注意，这些模型根据 `input_values` 是否填充会得到略有不同的结果。
            # </Tip>

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多细节，请参阅返回的张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多细节，请参阅返回的张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
"""
@add_start_docstrings(
    "The bare WavLM Model transformer outputting raw hidden-states without any specific head on top.",
    WAVLM_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model 复制而来，将 Wav2Vec2Model 改为 WavLMModel，wav2vec2 改为 wavlm，WAV_2_VEC_2 改为 WAVLM，WavLMBaseModelOutput 改为 Wav2Vec2BaseModelOutput
class WavLMModel(WavLMPreTrainedModel):
    def __init__(self, config: WavLMConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = WavLMFeatureEncoder(config)  # 初始化特征提取器
        self.feature_projection = WavLMFeatureProjection(config)  # 初始化特征投影器

        # 如果配置中的 mask_time_prob 大于 0.0 或者 mask_feature_prob 大于 0.0，则模型需要掩码向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())  # 初始化掩码特征向量

        # 根据配置选择稳定层归一化编码器或一般编码器
        if config.do_stable_layer_norm:
            self.encoder = WavLMEncoderStableLayerNorm(config)  # 初始化稳定层归一化编码器
        else:
            self.encoder = WavLMEncoder(config)  # 初始化一般编码器

        self.adapter = WavLMAdapter(config) if config.add_adapter else None  # 根据配置选择是否添加适配器

        # 初始化权重并应用最终处理
        self.post_init()

    def freeze_feature_extractor(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其在训练过程中不会更新其参数。
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其在训练过程中不会更新其参数。
        """
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
"""
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        # 检查配置中的 `apply_spec_augment` 是否为 True，如果不是，则直接返回隐藏状态
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            # 如果给定了 mask_time_indices，则使用这些索引应用 SpecAugment 到时间轴上的隐藏状态
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 根据配置中的概率生成 mask_time_indices，并应用 SpecAugment 到时间轴上的隐藏状态
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
            # 根据配置中的概率生成 mask_feature_indices，并应用 SpecAugment 到特征轴上的隐藏状态
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
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 如果输出注意力值未指定，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取输入特征向量
        extract_features = self.feature_extractor(input_values)
        # 调整特征向量的维度顺序
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # 计算与特征向量对应的减少的注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 对特征向量进行特征投影
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 根据给定的时间索引和注意力掩码屏蔽隐藏状态
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 使用编码器处理隐藏状态和注意力掩码
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果存在适配器模块，应用适配器
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 如果不要求返回字典形式的输出，返回一个元组
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 否则，返回一个 Wav2Vec2BaseModelOutput 对象
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """WavLM Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAVLM_START_DOCSTRING,
)
# 使用装饰器 `add_start_docstrings` 添加模型的文档字符串，描述了该模型的用途和特性
# 从 `transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC` 复制而来，修改为 `WavLMForCTC`，并进行了相应的符号和名称替换
class WavLMForCTC(WavLMPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类的初始化方法，传入配置信息
        super().__init__(config)

        # 初始化 WavLM 模型
        self.wavlm = WavLMModel(config)
        # 添加一个 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言属性
        self.target_lang = target_lang

        # 检查配置中是否定义了词汇表大小，如果没有则引发错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `WavLMForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        # 根据配置决定输出隐藏层大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 添加一个线性层作为语言模型的输出层
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并进行最终处理
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """
        
        # 覆盖 `PreTrainedModel.tie_weights` 方法，以便在传递 `target_lang=...` 给 `from_pretrained(...)` 时能正确加载适配器权重

        # 注意，通常 `tie_weights` 用于绑定输入和输出嵌入权重。在这里重新用于正确加载 WavLM 的适配器层，以避免为 `PreTrainedModel` 引入新的 API。
        # 虽然有些许 hacky，但是 WavLM 永远不必绑定输入和输出嵌入，因此在这里重新用这个函数是可以接受的。

        # 获取目标语言
        target_lang = self.target_lang

        # 如果 `target_lang` 不为 `None`，且 `config.adapter_attn_dim` 未定义，则引发错误
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果 `target_lang` 为 `None`，但 `config.adapter_attn_dim` 已定义，则记录信息提示默认设置为 'eng'
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果 `target_lang` 不为 `None`，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新。
    def freeze_feature_extractor(self):
        # 发出警告信息，提醒方法 `freeze_feature_extractor` 将在 Transformers v5 中删除，
        # 建议使用等效的 `freeze_feature_encoder` 方法。
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 `freeze_feature_encoder` 方法冻结特征编码器的参数。
        self.freeze_feature_encoder()

    # 调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新。
    def freeze_feature_encoder(self):
        # 调用特征编码器内部的方法 `_freeze_parameters`，冻结其参数。
        self.wavlm.feature_extractor._freeze_parameters()

    # 调用此函数将禁用基础模型的梯度计算，使其参数在训练期间不会更新，仅更新分类头部。
    def freeze_base_model(self):
        # 遍历语音语言模型 `wavlm` 的所有参数，并将其 `requires_grad` 属性设为 False。
        for param in self.wavlm.parameters():
            param.requires_grad = False

    # 重写了 `forward` 方法，并应用了两个装饰器 `add_start_docstrings_to_model_forward` 和 `add_code_sample_docstrings`。
    # 这些装饰器用于向 `forward` 方法添加文档字符串，提供了模型输入、输出和示例代码的描述。
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
        # 设置返回字典，如果未提供，则使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用wavlm模型，传入输入值和额外的参数，并获取输出
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态，并应用dropout进行正则化
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 将隐藏状态输入到语言模型头部以获取预测的logits
        logits = self.lm_head(hidden_states)

        # 初始化损失为None
        loss = None
        if labels is not None:
            # 检查标签是否超出词汇表大小，如果是则引发值错误
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 根据注意力掩码获取输入长度
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充的标记用-100表示未被关注时
            # 创建标签掩码以指示有效的标签位置和计算目标长度
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # 对logits进行log_softmax处理，并进行维度变换
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 使用ctc_loss计算损失，确保不启用fp16计算
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

        # 如果不需要返回字典，则构建输出元组
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回CausalLMOutput对象，封装损失、logits、隐藏状态和注意力张量
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 定义一个带有顶部序列分类头部的 WavLM 模型，用于类似 SUPERB 关键词检测任务的应用
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

        # 如果配置允许使用适配器且配置为真，则引发值错误，因为序列分类不支持 WavLM 适配器
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of WavLM adapters (config.add_adapter=True)"
            )
        
        # 初始化 WavLM 模型
        self.wavlm = WavLMModel(config)
        
        # 计算层数，包括变换器层和输入嵌入
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        
        # 如果配置使用加权层求和，则初始化层权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 用于投影的线性层，将隐藏大小投影到分类器投影大小
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        
        # 分类器线性层，将分类器投影大小映射到类别数
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.freeze_feature_extractor 复制而来
    def freeze_feature_extractor(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新。
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.freeze_feature_encoder 复制而来
    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新。
        """
        self.wavlm.feature_extractor._freeze_parameters()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.freeze_base_model 复制而来
    def freeze_base_model(self):
        """
        调用此函数将禁用基础模型的梯度计算，使其参数在训练期间不会更新。只有分类头部将会更新。
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward复制过来，替换Wav2Vec2为WavLM，wav2vec2为wavlm
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

        # 确定是否返回字典格式的输出，若未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用wavlm模型进行正向传播
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置中指定使用加权层求和，则对隐藏状态进行加权求和操作
        if self.config.use_weighted_layer_sum:
            # 获取隐藏状态
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 在指定维度上堆叠隐藏状态
            hidden_states = torch.stack(hidden_states, dim=1)
            # 计算加权层的softmax权重
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 对隐藏状态进行加权求和操作
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用第一个输出作为隐藏状态
            hidden_states = outputs[0]

        # 将加权求和后的隐藏状态投影到目标维度
        hidden_states = self.projector(hidden_states)
        
        # 如果没有指定attention_mask，则将隐藏状态进行均值池化操作
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            # 否则，根据给定的attention_mask计算padding_mask
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            # 将非padding位置的隐藏状态设置为0
            hidden_states[~padding_mask] = 0.0
            # 对padding后的隐藏状态进行求和并除以padding_mask的求和得到均值池化的结果
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 将池化后的输出传入分类器得到logits
        logits = self.classifier(pooled_output)

        # 初始化损失为None
        loss = None
        # 如果给定了labels，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 如果不返回字典格式的输出，则按顺序返回logits和隐藏状态列表
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典格式的输出，则使用SequenceClassifierOutput对象包装结果并返回
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 用于在顶部添加模型文档字符串，描述该模型是在音频帧分类任务上带有分类头的WavLM模型
@add_start_docstrings(
    """
    WavLM Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    WAVLM_START_DOCSTRING,
)
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification复制而来，将Wav2Vec2->WavLM，wav2vec2->wavlm，WAV_2_VEC_2->WAVLM
class WavLMForAudioFrameClassification(WavLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中有add_adapter属性且为True，则引发值错误，因为音频帧分类不支持使用WavLM适配器
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of WavLM adapters (config.add_adapter=True)"
            )
        
        # 初始化WavLM模型
        self.wavlm = WavLMModel(config)
        
        # 计算层数，包括变压器层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        
        # 如果配置中使用加权层求和，则初始化层权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 分类器层，将隐藏状态大小映射到类标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        # 初始化模型权重
        self.init_weights()

    # Deprecated警告，已弃用freeze_feature_extractor方法，请使用freeze_feature_encoder代替
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

    # 冻结特征编码器，禁止特征编码器参数的梯度计算，使其在训练过程中不会更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wavlm.feature_extractor._freeze_parameters()

    # 冻结基础模型，禁止基础模型参数的梯度计算，使其在训练过程中不会更新，仅分类头会更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False

    # 为模型前向传播方法添加模型输入的文档字符串，引用WAVLM_INPUTS_DOCSTRING，并提供代码示例文档字符串
    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_FRAME_CLASS_CHECKPOINT,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_FRAME_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 确定是否使用返回字典，如果未指定则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用语言模型的前向传播，获取模型的输出结果
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置指定使用加权层求和，则处理隐藏状态
        if self.config.use_weighted_layer_sum:
            # 获取模型输出中的隐藏状态
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 将隐藏状态堆叠在一起
            hidden_states = torch.stack(hidden_states, dim=1)
            # 计算加权层的权重并进行softmax归一化
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 对隐藏状态进行加权求和
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用模型输出的第一个隐藏状态
            hidden_states = outputs[0]

        # 使用分类器对隐藏状态进行分类预测
        logits = self.classifier(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        # 如果不要求返回字典，则返回分类器的输出和隐藏状态
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        # 否则返回一个TokenClassifierOutput对象，包括损失、预测的logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从 transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss 复制而来，定义了一个 AMSoftmaxLoss 类
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale  # 缩放参数，用于调整余弦相似度的范围
        self.margin = margin  # 间隔参数，用于调整余弦相似度与边界的距离
        self.num_labels = num_labels  # 标签类别数量
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)  # 损失函数使用的权重参数
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数

    def forward(self, hidden_states, labels):
        labels = labels.flatten()  # 将标签展平为一维张量
        weight = nn.functional.normalize(self.weight, dim=0)  # 对权重进行 L2 归一化
        hidden_states = nn.functional.normalize(hidden_states, dim=1)  # 对隐藏状态进行 L2 归一化
        cos_theta = torch.mm(hidden_states, weight)  # 计算余弦相似度
        psi = cos_theta - self.margin  # 计算带有间隔的余弦相似度

        onehot = nn.functional.one_hot(labels, self.num_labels)  # 将标签转换为独热编码
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)  # 根据标签和间隔调整后的余弦相似度计算最终的 logits
        loss = self.loss(logits, labels)  # 计算损失值

        return loss


# 从 transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer 复制而来，定义了一个 TDNNLayer 类
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]  # 输入维度
        self.out_conv_dim = config.tdnn_dim[layer_id]  # 输出维度
        self.kernel_size = config.tdnn_kernel[layer_id]  # 卷积核大小
        self.dilation = config.tdnn_dilation[layer_id]  # 膨胀率

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)  # 线性层作为卷积核
        self.activation = nn.ReLU()  # ReLU 激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_peft_available():  # 检查是否可用 peft 库
            from peft.tuners.lora import LoraLayer  # 导入 LoraLayer

            if isinstance(self.kernel, LoraLayer):  # 如果 kernel 是 LoraLayer 类型
                warnings.warn(
                    "Detected LoRA on TDNNLayer. LoRA weights won't be applied due to optimization. "
                    "You should exclude TDNNLayer from LoRA's target modules.",
                )

        # 为了向后兼容性，保留 nn.Linear，但调用 F.conv1d 以提高速度
        hidden_states = hidden_states.transpose(1, 2)  # 转置隐藏状态的维度
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)  # 调整权重的维度
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)  # 使用 conv1d 进行卷积操作
        hidden_states = hidden_states.transpose(1, 2)  # 再次转置隐藏状态的维度

        hidden_states = self.activation(hidden_states)  # 应用 ReLU 激活函数
        return hidden_states


@add_start_docstrings(
    """
    WavLM Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    WAVLM_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector 复制而来，定义了一个 WavLMForXVector 类，用于 XVector 特征提取
class WavLMForXVector(WavLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)  # 调用父类的初始化方法，传递配置参数给父类

        self.wavlm = WavLMModel(config)  # 创建一个语音语言模型对象
        num_layers = config.num_hidden_layers + 1  # 计算层的数量：变换器层 + 输入嵌入层
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)  # 如果配置启用了加权层求和，则创建一个权重参数
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])  # 创建一个线性层投影器

        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]  # 创建一系列TDNN层
        self.tdnn = nn.ModuleList(tdnn_layers)  # 将TDNN层存储在模块列表中

        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)  # 创建特征提取器的线性层
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)  # 创建分类器的线性层

        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)  # 创建AMSoftmax损失函数对象

        self.init_weights()  # 初始化模型权重

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
        self.freeze_feature_encoder()  # 警告已弃用此方法，建议使用等效的 `freeze_feature_encoder` 方法

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wavlm.feature_extractor._freeze_parameters()  # 冻结特征编码器的参数，禁用其在训练期间的梯度计算

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False  # 禁用基础模型的梯度计算，使其参数在训练期间不会更新。仅更新分类头部的参数。

    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1  # 计算1D卷积层的输出长度公式

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)  # 遍历TDNN内核大小，计算输入长度的输出长度

        return input_lengths

    @add_start_docstrings_to_model_forward(WAVLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_XVECTOR_CHECKPOINT,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_XVECTOR_EXPECTED_OUTPUT,
    )
    # 定义一个方法 `forward`，用于模型前向传播
    def forward(
        # 输入值，可以是一个 PyTorch 张量，可选参数
        self,
        input_values: Optional[torch.Tensor],
        # 注意力掩码，用于指定模型注意力分布，可选参数
        attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力分布，可选参数，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回一个字典作为输出，可选参数，默认为 None
        return_dict: Optional[bool] = None,
        # 标签数据，可选参数，用于某些任务的监督学习
        labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 根据返回参数设置是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏层状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用 WAVLM 模型进行语音识别任务的计算
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置中使用加权层求和，则对隐藏状态进行加权求和操作
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # 对隐藏状态进行投影
        hidden_states = self.projector(hidden_states)

        # 通过一系列 TDNN 层处理隐藏状态特征
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # 统计池化操作
        if attention_mask is None:
            # 如果没有注意力掩码，则对隐藏状态在第一维上进行均值和标准差计算
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            # 根据注意力掩码计算特征提取器输出的长度
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                # 对每个序列进行统计池化操作
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        # 通过特征提取器得到最终的输出特征向量
        output_embeddings = self.feature_extractor(statistic_pooling)
        # 使用分类器进行最终的分类预测
        logits = self.classifier(output_embeddings)

        # 计算损失
        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        # 根据返回参数决定返回值的组织方式
        if not return_dict:
            # 如果不使用返回字典，则返回元组形式的结果
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 使用自定义的输出类构造返回结果
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```