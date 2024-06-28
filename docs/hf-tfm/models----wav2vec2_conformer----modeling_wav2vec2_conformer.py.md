# `.\models\wav2vec2_conformer\modeling_wav2vec2_conformer.py`

```
# 设置编码格式为 UTF-8
# 版权声明，包括 Fairseq 作者和 HuggingFace Inc. 团队
#
# 根据 Apache 许可证版本 2.0 使用本文件，详见许可证
#
# 如果不是依照许可证的规定使用本文件，不得使用
#
# 详细许可证信息请访问 http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“按原样”提供，无任何明示或暗示的保证或条件
# 包括但不限于适销性或特定用途适用性的保证或条件。详见许可证。
""" PyTorch Wav2Vec2-Conformer model."""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入自定义的激活函数映射 ACT2FN
from ...activations import ACT2FN
# 导入 DeepSpeed 集成检测功能
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
# 导入模型输出类
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel
# 导入工具函数
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_peft_available,
    logging,
    replace_return_docstrings,
)
# 导入 Wav2Vec2Conformer 的配置类
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态起始位置的全局常量
_HIDDEN_STATES_START_POSITION = 2

# 用于文档的配置名称
_CONFIG_FOR_DOC = "Wav2Vec2ConformerConfig"

# 用于文档的检查点名称
_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-conformer-rope-large-960h-ft"

# 预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 292, 1024]

# CTC（Connectionist Temporal Classification）的预期输出示例
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 64.21

# 预训练模型存档列表
WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-conformer-rel-pos-large",
    # 更多 Wav2Vec2Conformer 模型请查看 https://huggingface.co/models?filter=wav2vec2-conformer
]

@dataclass
# 基于 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput
# 的输出类型定义，用于支持潜在的隐藏状态和注意力
class Wav2Vec2ConformerForPreTrainingOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ConformerForPreTraining`], with potential hidden states and attentions.
    """
    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """

    # 总体损失，包括对比损失和多样性损失，用于分类任务
    loss: Optional[torch.FloatTensor] = None
    # 模型隐藏状态投影到 config.proj_codevector_dim 维度，用于预测掩码后的量化状态
    projected_states: torch.FloatTensor = None
    # 量化提取的特征向量投影到 config.proj_codevector_dim 维度，作为对比损失的正样本向量
    projected_quantized_states: torch.FloatTensor = None
    # 模型每一层的隐藏状态，以及初始嵌入输出的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 模型每一层的注意力权重的元组，用于计算自注意力头中的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 对比损失 L_m，参考论文中的定义
    contrastive_loss: Optional[torch.FloatTensor] = None
    # 多样性损失 L_d，参考论文中的定义
    diversity_loss: Optional[torch.FloatTensor] = None
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
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape  # 获取批次大小和序列长度

    if mask_length < 1:  # 如果 mask_length 小于 1，抛出数值错误
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:  # 如果 mask_length 大于序列长度，抛出数值错误
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon 用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 计算应该被掩盖的 span 的数量
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保 num_masked_span 不超过 sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保 num_masked span 不超过 input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算每个批次中的被掩盖的 span 的数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()  # 如果 attention_mask 不为 None，则计算每个批次的实际长度
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]  # 否则，所有批次长度都为 sequence_length
    )

    # 创建一个全零的布尔数组，用于表示 SpecAugment 掩盖
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []  # 存储 SpecAugment 掩盖的索引

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算可以掩盖的最大 span 数量
    # 如果最大的被遮蔽片段数为0，则直接返回特定的遮蔽掩码
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算当前输入长度下的被遮蔽片段数目
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要遮蔽的索引位置
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个被抽样的索引作为填充向量的虚拟索引，以确保所有批次的维度相同（由于概率舍入）
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只可能发生在 `input_length` 严格小于 `sequence_length` 的情况下，
            # 此时最后一个令牌必须是填充令牌，我们可以使用其作为虚拟遮蔽标识符
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟遮蔽索引添加到被抽样索引中，使得总长度等于 `max_num_masked_span`
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将列表转换为 NumPy 数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮蔽索引扩展为遮蔽片段
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量到起始索引，以便索引现在创建一个片段
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不能大于 `sequence_length - 1`
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 在遮蔽掩码上散布索引以进行遮蔽
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回经过特定数据增强处理后的遮蔽掩码
    return spec_aug_mask
# Copied from transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices
def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    # 获取批量大小和序列长度
    batch_size, sequence_length = features_shape

    # 生成正向向量本身的索引，并将它们重复 `num_negatives` 次
    sequence_length_range = np.arange(sequence_length)

    # 从同一句话中获取 `num_negatives` 个随机向量索引
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    # 如果给定了 mask_time_indices，则将其转换为布尔型数组；否则创建一个全部为 True 的数组
    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    # 遍历每个批次中的索引
    for batch_idx in range(batch_size):
        # 计算非零元素的数量
        high = mask_time_indices[batch_idx].sum() - 1
        # 获取映射后的屏蔽索引
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        # 广播以重复索引
        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        # 从 0 到 high 之间随机选择向量索引
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # 避免采样相同的正向向量，同时保持分布均匀
        sampled_indices[sampled_indices >= feature_indices] += 1

        # 将采样的负向索引重新映射到实际索引
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # 对批次大小进行修正
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->Wav2Vec2Conformer
class Wav2Vec2ConformerNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 获取输入卷积维度和输出卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 获取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 执行一维卷积操作
        hidden_states = self.conv(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->Wav2Vec2Conformer
class Wav2Vec2ConformerLayerNormConvLayer(nn.Module):
    # 初始化函数，用于设置卷积神经网络的一维卷积层及其参数
    def __init__(self, config, layer_id=0):
        # 调用父类的初始化方法
        super().__init__()
        
        # 根据给定的配置获取输入卷积维度，如果layer_id大于0则取上一层的卷积维度，否则默认为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 获取当前层的输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一维卷积层，指定输入和输出的卷积维度，以及内核大小、步长和是否使用偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        
        # 创建一维卷积层后的层归一化层，指定归一化的维度和是否启用元素级别的仿射变换
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        
        # 根据配置选择激活函数，并将其赋值给activation变量
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数，接受输入的hidden_states进行前向计算，并返回计算后的结果
    def forward(self, hidden_states):
        # 应用一维卷积操作到输入的hidden_states
        hidden_states = self.conv(hidden_states)

        # 将hidden_states的维度进行转置，交换倒数第二和倒数第一维度
        hidden_states = hidden_states.transpose(-2, -1)
        
        # 对转置后的hidden_states进行层归一化操作
        hidden_states = self.layer_norm(hidden_states)
        
        # 再次将hidden_states的维度进行转置，恢复到初始维度
        hidden_states = hidden_states.transpose(-2, -1)
        
        # 应用激活函数activation到归一化后的hidden_states
        hidden_states = self.activation(hidden_states)
        
        # 返回处理后的hidden_states作为前向传播的输出
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer复制而来，将Wav2Vec2改为Wav2Vec2Conformer
class Wav2Vec2ConformerGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果layer_id大于0，则设置输入卷积维度为config.conv_dim[layer_id - 1]，否则为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个1维卷积层，输入维度为self.in_conv_dim，输出维度为self.out_conv_dim
        # 使用config.conv_kernel[layer_id]作为卷积核大小，config.conv_stride[layer_id]作为步长，config.conv_bias作为偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数为ACT2FN[config.feat_extract_activation]
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个Group Normalization层，num_groups和num_channels都为self.out_conv_dim，启用仿射变换
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 对输入的hidden_states进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的hidden_states进行Group Normalization
        hidden_states = self.layer_norm(hidden_states)
        # 对Group Normalization后的hidden_states应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding复制而来，将Wav2Vec2改为Wav2Vec2Conformer
class Wav2Vec2ConformerPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个1维卷积层，输入和输出维度都为config.hidden_size
        # 使用config.num_conv_pos_embeddings作为卷积核大小，config.num_conv_pos_embeddings // 2作为填充，groups为config.num_conv_pos_embedding_groups
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 设置权重归一化方式为nn.utils.weight_norm
        weight_norm = nn.utils.weight_norm
        # 如果存在nn.utils.parametrizations.weight_norm，则使用该权重归一化方式
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 如果启用了deepspeed的zero3，则使用zero.GatheredParameters对self.conv.weight进行处理
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 否则，对self.conv进行权重归一化
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建一个Wav2Vec2ConformerSamePadLayer实例，使用config.num_conv_pos_embeddings作为参数
        self.padding = Wav2Vec2ConformerSamePadLayer(config.num_conv_pos_embeddings)
        # 设置激活函数为ACT2FN[config.feat_extract_activation]
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将输入的hidden_states进行维度转换，将第1和第2个维度互换
        hidden_states = hidden_states.transpose(1, 2)

        # 对转置后的hidden_states进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的hidden_states进行填充操作
        hidden_states = self.padding(hidden_states)
        # 对填充后的hidden_states应用激活函数
        hidden_states = self.activation(hidden_states)

        # 再次将hidden_states的第1和第2个维度互换回来
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2ConformerRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """
    def __init__(self, config):
        super().__init__()
        # 计算每个注意头的隐藏大小
        dim = config.hidden_size // config.num_attention_heads
        # 旋转嵌入的基数
        base = config.rotary_embedding_base

        # 计算频率的倒数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        # 将频率的倒数注册为缓冲区
        self.register_buffer("inv_freq", inv_freq)
        # 初始化缓存的序列长度和旋转位置嵌入
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states):
        # 获取输入隐藏状态的序列长度
        sequence_length = hidden_states.shape[1]

        # 如果缓存的序列长度与当前序列长度相同，并且缓存的旋转位置嵌入不为空，则直接返回缓存的旋转位置嵌入
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        # 更新缓存的序列长度
        self.cached_sequence_length = sequence_length

        # 计算时间戳，使用与 inv_freq 常量相同的数据类型
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        # 计算频率
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        # 创建嵌入向量，包括 cos 和 sin 部分
        embeddings = torch.cat((freqs, freqs), dim=-1)

        # 计算 cos 和 sin 的嵌入向量
        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # 将计算得到的嵌入向量转换为与隐藏状态输入相同的数据类型
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        # 返回缓存的旋转位置嵌入
        return self.cached_rotary_positional_embedding
class Wav2Vec2ConformerRelPositionalEmbedding(nn.Module):
    """Relative positional encoding module."""

    def __init__(self, config):
        super().__init__()
        # 最大位置编码长度，从配置中获取
        self.max_len = config.max_source_positions
        # 模型隐藏层大小，从配置中获取
        self.d_model = config.hidden_size
        # 位置编码张量，初始为空
        self.pe = None
        # 初始化位置编码
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        # 重置位置编码
        if self.pe is not None:
            # self.pe 包含正负两部分
            # self.pe 的长度为 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    # 调整 self.pe 的数据类型和设备
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # 创建正向和负向的位置编码
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

        # 反转正向索引的顺序并连接正向和负向索引，支持偏移技巧
        # 参考 https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        # 扩展位置编码
        self.extend_pe(hidden_states)
        start_idx = self.pe.size(1) // 2 - hidden_states.size(1) + 1
        end_idx = self.pe.size(1) // 2 + hidden_states.size(1)
        # 获取相对位置编码
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer 复制，将 Wav2Vec2 改为 Wav2Vec2Conformer
class Wav2Vec2ConformerSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 如果卷积位置编码数是偶数，则需要移除一个填充
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            # 移除最后一维的填充
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder复制到Wav2Vec2->Wav2Vec2Conformer
class Wav2Vec2ConformerFeatureEncoder(nn.Module):
    """从原始音频波形构建特征"""

    def __init__(self, config):
        super().__init__()

        # 根据配置选择特征提取层的归一化方式
        if config.feat_extract_norm == "group":
            # 如果使用组归一化，第一层使用Wav2Vec2ConformerGroupNormConvLayer，其后的层使用Wav2Vec2ConformerNoLayerNormConvLayer
            conv_layers = [Wav2Vec2ConformerGroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2ConformerNoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果使用层归一化，使用Wav2Vec2ConformerLayerNormConvLayer
            conv_layers = [
                Wav2Vec2ConformerLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 抛出异常，如果配置中的特征提取归一化不在支持的选项中
            raise ValueError(
                f"`config.feat_extract_norm` 是 {config.feat_extract_norm}，但必须是 ['group', 'layer'] 中的一种"
            )
        
        # 将卷积层列表转换为模块列表
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False  # 是否启用梯度检查点
        self._requires_grad = True  # 是否需要梯度更新

    def _freeze_parameters(self):
        # 冻结所有参数，不进行梯度更新
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]  # 增加维度以匹配期望的输入形状

        # 如果需要梯度更新并且处于训练模式，则确保隐藏状态需要梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 如果启用梯度检查点并且需要梯度更新，则使用梯度检查点函数优化卷积层的计算
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则直接通过卷积层计算隐藏状态
                hidden_states = conv_layer(hidden_states)

        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection复制到Wav2Vec2->Wav2Vec2Conformer
class Wav2Vec2ConformerFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用层归一化对隐藏状态进行归一化
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 使用线性映射对隐藏状态进行投影到指定的隐藏维度
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 使用Dropout对投影结果进行随机失活
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 对隐藏状态进行归一化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态进行线性投影
        hidden_states = self.projection(norm_hidden_states)
        # 对投影结果进行随机失活
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward复制到Wav2Vec2->Wav2Vec2Conformer
class Wav2Vec2ConformerFeedForward(nn.Module):
    # 初始化函数，用于创建一个新的神经网络层对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        
        # 创建中间层的dropout层，根据配置中的激活函数的dropout值
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 创建中间层的全连接层，将输入大小调整为配置中的中间层大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # 根据配置中的激活函数，选择相应的激活函数，并赋值给self.intermediate_act_fn
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
        # 创建输出层的全连接层，将中间层大小调整为配置中的隐藏层大小
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # 创建输出层的dropout层，根据配置中的隐藏层的dropout值
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播函数，定义了数据在神经网络中的流动方向
    def forward(self, hidden_states):
        # 中间层全连接层的前向传播，对输入的hidden_states进行线性变换
        hidden_states = self.intermediate_dense(hidden_states)
        
        # 中间层的激活函数的前向传播，应用于线性变换后的结果
        hidden_states = self.intermediate_act_fn(hidden_states)
        
        # 中间层的dropout层的前向传播，对激活函数的输出进行随机置零
        hidden_states = self.intermediate_dropout(hidden_states)

        # 输出层全连接层的前向传播，对中间层输出进行线性变换
        hidden_states = self.output_dense(hidden_states)
        
        # 输出层的dropout层的前向传播，对线性变换后的结果进行随机置零
        hidden_states = self.output_dropout(hidden_states)
        
        # 返回最终的神经网络层的输出结果
        return hidden_states
# 定义一个用于 Conformer 模块的卷积块
class Wav2Vec2ConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        super().__init__()
        # 检查是否满足 'SAME' 填充条件，深度可分离卷积核大小应为奇数
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        
        # Layer normalization 层，对隐藏状态进行归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 第一个点卷积层，将隐藏大小映射到两倍的隐藏大小
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        
        # GLU（门控线性单元）激活函数，用于特征维度的门控
        self.glu = nn.GLU(dim=1)
        
        # 深度可分离卷积层，用于捕获局部依赖关系
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=(config.conv_depthwise_kernel_size - 1) // 2,
            groups=config.hidden_size,
            bias=False,
        )
        
        # 批标准化层，用于加速收敛和稳定训练过程
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        
        # 激活函数，根据配置选择的激活函数类型
        self.activation = ACT2FN[config.hidden_act]
        
        # 第二个点卷积层，将隐藏大小映射回原始大小
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        
        # Dropout 层，用于随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.conformer_conv_dropout)

    def forward(self, hidden_states):
        # 对隐藏状态进行层归一化
        hidden_states = self.layer_norm(hidden_states)
        
        # 交换时间维度和特征维度，使得特征维度在最后一维
        hidden_states = hidden_states.transpose(1, 2)

        # GLU 机制，将特征维度分成两部分并应用门控
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        # 1D 深度可分离卷积，捕获局部依赖关系
        hidden_states = self.depthwise_conv(hidden_states)
        
        # 批标准化，加速收敛和稳定训练过程
        hidden_states = self.batch_norm(hidden_states)
        
        # 激活函数，根据配置选择的激活函数类型
        hidden_states = self.activation(hidden_states)

        # 第二个点卷积层，将隐藏大小映射回原始大小
        hidden_states = self.pointwise_conv2(hidden_states)
        
        # Dropout 层，随机丢弃部分神经元，防止过拟合
        hidden_states = self.dropout(hidden_states)
        
        # 恢复时间维度和特征维度的交换，使得特征维度在第二维
        hidden_states = hidden_states.transpose(1, 2)
        
        # 返回处理后的隐藏状态
        return hidden_states


class Wav2Vec2ConformerSelfAttention(nn.Module):
    """Construct an Wav2Vec2ConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """
    # 初始化函数，用于初始化一个多头注意力层对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 计算每个注意力头的大小
        self.head_size = config.hidden_size // config.num_attention_heads
        # 设置注意力头的数量
        self.num_heads = config.num_attention_heads
        # 设置位置编码的类型（绝对或相对）
        self.position_embeddings_type = config.position_embeddings_type

        # 初始化用于查询的线性层
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化用于键的线性层
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化用于值的线性层
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化输出的线性层
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        # 初始化用于dropout的层
        self.dropout = nn.Dropout(p=config.attention_dropout)

        # 如果位置编码类型为"relative"
        if self.position_embeddings_type == "relative":
            # 初始化用于位置编码的线性层
            self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            # 初始化用于矩阵c和矩阵d的可学习偏置
            # 参考文献 https://arxiv.org/abs/1901.02860 第3.3节
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    # 前向传播函数，处理输入的隐藏状态和其他可选参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 定义函数的输入和输出类型，返回一个元组，包含三个元素：
        # 1. torch.Tensor：经过注意力机制处理后的隐藏状态
        # 2. Optional[torch.Tensor]：可能为 None 的注意力概率分布
        # 3. Optional[Tuple[torch.Tensor]]：可能为 None 的额外张量元组

        # 获取隐藏状态的批量大小、序列长度和隐藏单元大小
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # 将 query/key 状态与 value 状态分开处理
        query_key_states = hidden_states
        value_states = hidden_states

        # 如果采用旋转型位置编码
        if self.position_embeddings_type == "rotary":
            # 检查相对位置编码是否已定义，如果未定义则抛出错误
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            # 对 query_key_states 应用旋转型位置编码
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

        # 投影 query_key_states 和 value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # 将维度重新排列为 (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 如果采用相对位置编码
        if self.position_embeddings_type == "relative":
            # 检查相对位置编码是否已定义，如果未定义则抛出错误
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'relative'"
                )
            # 应用相对位置编码到 qk 分数，参考 Transformer_XL: https://arxiv.org/abs/1901.02860
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            # 根据经典方法计算注意力分数
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        # 如果存在注意力掩码，则应用到注意力分数上
        if attention_mask is not None:
            scores = scores + attention_mask

        # 计算注意力概率分布，维度为 (batch, head, time1, time2)
        probs = torch.softmax(scores, dim=-1)
        # 对注意力概率分布应用 dropout
        probs = self.dropout(probs)

        # 计算加权后的 value，维度为 (batch, head, time1, d_k)
        hidden_states = torch.matmul(probs, value)

        # 将维度重新排列为 (batch, time1, hidden_size)，并应用输出线性层
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)

        # 返回处理后的隐藏状态和可能的注意力概率分布
        return hidden_states, probs
    # 对输入的隐藏状态应用旋转嵌入和相对位置嵌入
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        # 获取批量大小、序列长度和隐藏层大小
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # 将隐藏状态重塑为(batch_size, sequence_length, num_heads, head_size)
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        # 获取相对位置嵌入的余弦部分和正弦部分
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # 旋转隐藏状态和旋转部分
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        # 应用旋转嵌入公式
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        # 将隐藏状态重塑为(batch_size, sequence_length, num_heads * head_size)
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        return hidden_states
    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        # 1. project positional embeddings
        # 将位置嵌入投影
        # => (batch, head, 2*time1-1, d_k)
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

        # 2. Add bias to query
        # 给查询添加偏置
        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        # 3. attention score: first compute matrix a and matrix c
        # 计算注意力分数：首先计算矩阵 a 和矩阵 c
        # 如 https://arxiv.org/abs/1901.02860 第 3.3 节所述
        # => (batch, head, time1, time2)
        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        # 4. then compute matrix b and matrix d
        # 然后计算矩阵 b 和矩阵 d
        # => (batch, head, time1, 2*time1-1)
        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # 5. shift matrix b and matrix d
        # 移位矩阵 b 和矩阵 d
        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.size(-1) // 2 + 1]

        # 6. sum matrices
        # 求和矩阵
        # => (batch, head, time1, time2)
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores
class Wav2Vec2ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = Wav2Vec2ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = Wav2Vec2ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = Wav2Vec2ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = Wav2Vec2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weights


注释：

# 定义一个名为Wav2Vec2ConformerEncoderLayer的类，用于实现Conformer结构，参考自https://arxiv.org/abs/2005.08100
class Wav2Vec2ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size  # 从配置中获取隐藏大小
        dropout = config.attention_dropout  # 从配置中获取注意力丢弃率

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)  # Layer normalization层
        self.ffn1 = Wav2Vec2ConformerFeedForward(config)  # 第一个前向传播网络

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)  # Layer normalization层
        self.self_attn_dropout = nn.Dropout(dropout)  # Dropout层
        self.self_attn = Wav2Vec2ConformerSelfAttention(config)  # 自注意力层

        # Conformer Convolution
        self.conv_module = Wav2Vec2ConformerConvolutionModule(config)  # Conformer卷积模块

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)  # Layer normalization层
        self.ffn2 = Wav2Vec2ConformerFeedForward(config)  # 第二个前向传播网络
        self.final_layer_norm = nn.LayerNorm(embed_dim)  # 最终的Layer normalization层

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        hidden_states = hidden_states  # 输入隐藏状态

        # 1. Feed-Forward 1 layer
        residual = hidden_states  # 残差连接
        hidden_states = self.ffn1_layer_norm(hidden_states)  # Layer normalization
        hidden_states = self.ffn1(hidden_states)  # 第一个前向传播网络
        hidden_states = hidden_states * 0.5 + residual  # 残差连接加权和
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)  # Layer normalization
        hidden_states, attn_weights = self.self_attn(  # 自注意力计算
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)  # Dropout层
        hidden_states = hidden_states + residual  # 残差连接加和

        # 3. Convolutional Layer
        residual = hidden_states  # 残差连接
        hidden_states = self.conv_module(hidden_states)  # Conformer卷积模块应用
        hidden_states = residual + hidden_states  # 残差连接加和

        # 4. Feed-Forward 2 Layer
        residual = hidden_states  # 残差连接
        hidden_states = self.ffn2_layer_norm(hidden_states)  # Layer normalization
        hidden_states = self.ffn2(hidden_states)  # 第二个前向传播网络
        hidden_states = hidden_states * 0.5 + residual  # 残差连接加权和
        hidden_states = self.final_layer_norm(hidden_states)  # 最终的Layer normalization

        return hidden_states, attn_weights  # 返回隐藏状态和注意力权重
    # 初始化方法，接收配置参数并调用父类初始化方法
    def __init__(self, config):
        super().__init__()
        # 将配置参数存储在实例变量中
        self.config = config

        # 根据配置中的位置嵌入类型选择不同的位置嵌入方式
        if config.position_embeddings_type == "relative":
            # 如果位置嵌入类型为"relative"，则使用相对位置嵌入方式初始化位置嵌入对象
            self.embed_positions = Wav2Vec2ConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            # 如果位置嵌入类型为"rotary"，则使用旋转位置嵌入方式初始化位置嵌入对象
            self.embed_positions = Wav2Vec2ConformerRotaryPositionalEmbedding(config)
        else:
            # 如果位置嵌入类型不在预期的"relative"或"rotary"中，将位置嵌入对象设置为None
            self.embed_positions = None

        # 初始化位置卷积嵌入对象
        self.pos_conv_embed = Wav2Vec2ConformerPositionalConvEmbedding(config)
        # 初始化层归一化对象，设置归一化大小和epsilon
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化dropout对象，设置隐藏层dropout比例
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化多层编码器层的列表，每层使用相同的配置参数
        self.layers = nn.ModuleList([Wav2Vec2ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为False
        self.gradient_checkpointing = False
    ):
        # 初始化隐藏状态和自注意力列表，根据是否需要输出隐藏状态和注意力矩阵做判断
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 如果有注意力掩码，则将未注意到的位置的隐藏状态置为0
        if attention_mask is not None:
            hidden_states[~attention_mask] = 0.0

            # 扩展注意力掩码维度，确保其与隐藏状态的数据类型匹配
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 对隐藏状态进行dropout处理
        hidden_states = self.dropout(hidden_states)

        # 如果存在位置嵌入，则计算相对位置嵌入
        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        # 检查是否启用了DeepSpeed Zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 遍历每个层进行处理
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加LayerDrop机制，决定是否跳过当前层
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False

            # 如果不跳过当前层或者启用了DeepSpeed Zero3
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果启用了梯度检查点功能且在训练模式下，则使用梯度检查点函数
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        relative_position_embeddings,
                        output_attentions,
                    )
                else:
                    # 否则直接调用当前层的forward方法
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        relative_position_embeddings=relative_position_embeddings,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            # 如果跳过当前层，则将输出设置为None
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果需要输出注意力矩阵，则将当前层的自注意力矩阵加入到列表中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对最终的隐藏状态进行LayerNorm处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到列表中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据是否需要以字典形式返回结果，决定返回哪些值
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GumbelVectorQuantizer复制到Wav2Vec2->Wav2Vec2Conformer
class Wav2Vec2ConformerGumbelVectorQuantizer(nn.Module):
    """
    使用Gumbel softmax进行向量量化。详见[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf)获取更多信息。
    """

    def __init__(self, config):
        super().__init__()
        self.num_groups = config.num_codevector_groups  # 设置编码向量组的数量
        self.num_vars = config.num_codevectors_per_group  # 设置每组编码向量的数量

        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # 为编码簇变量（码本）预留存储空间
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        # 权重投影层，用于线性映射
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # 可以在训练过程中逐渐减小的温度参数
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        """
        计算困惑度的静态方法。
        
        Args:
            probs (Tensor): 概率分布张量
            mask (Tensor, optional): 掩码张量，用于指示哪些位置应计入计算
        
        Returns:
            Tensor: 计算得到的困惑度
        """
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity
    # 定义前向传播方法，接受隐藏状态和时间掩码索引作为输入
    def forward(self, hidden_states, mask_time_indices=None):
        # 获取批量大小、序列长度和隐藏大小
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 投影到代码向量维度
        hidden_states = self.weight_proj(hidden_states)
        # 重新调整张量形状以便后续处理
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 使用 Gumbel Softmax 方法对隐藏状态进行采样，以获取代码向量概率
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # 计算困惑度
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # 在非可微方式下，取隐藏状态的最大值索引，计算硬代码向量分布（one hot）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        # 将代码向量概率重新调整张量形状以便后续处理
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率值检索代码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        # 将结果重新调整张量形状以便后续处理
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        # 返回代码向量和困惑度
        return codevectors, perplexity
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Adapter复制并修改为Wav2Vec2Conformer
class Wav2Vec2ConformerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 如果输出的隐藏大小不等于隐藏大小，则可能需要降维特征维度
        if config.output_hidden_size != config.hidden_size:
            # 线性投影层，将隐藏状态投影到输出的隐藏大小
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # LayerNorm层，用于归一化投影后的隐藏状态
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        # 使用Wav2Vec2ConformerAdapterLayer创建一组适配器层
        self.layers = nn.ModuleList(Wav2Vec2ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers))
        # LayerDrop的概率
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # 如果需要，对隐藏状态进行降维投影
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 将维度1和2进行转置，适配器层通常操作的维度顺序
        hidden_states = hidden_states.transpose(1, 2)

        # 对每一层适配器进行迭代
        for layer in self.layers:
            layerdrop_prob = np.random.random()
            # 如果不在训练阶段或者随机数大于LayerDrop概率，则跳过当前层
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        # 再次转置维度1和2，返回最终的隐藏状态
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AdapterLayer复制并修改为Wav2Vec2ConformerAdapterLayer
class Wav2Vec2ConformerAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 一维卷积层，用于适配器层的特征提取和变换
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states):
        # 对隐藏状态进行一维卷积操作
        hidden_states = self.conv(hidden_states)
        # 使用门控线性单元（GLU）激活函数进行特征变换
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


class Wav2Vec2ConformerPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化、下载和加载预训练模型的简单接口。
    """

    # 对应的配置类
    config_class = Wav2Vec2ConformerConfig
    # 基础模型的前缀
    base_model_prefix = "wav2vec2_conformer"
    # 主输入名称
    main_input_name = "input_values"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是 Wav2Vec2ConformerForPreTraining 类型，则初始化其两个线性层的参数
        if isinstance(module, Wav2Vec2ConformerForPreTraining):
            module.project_hid.reset_parameters()  # 重置隐藏层投影的参数
            module.project_q.reset_parameters()  # 重置查询投影的参数
            module.project_hid._is_hf_initialized = True  # 设置隐藏层投影已经初始化标志
            module.project_q._is_hf_initialized = True  # 设置查询投影已经初始化标志
        # 如果 module 是 Wav2Vec2ConformerGumbelVectorQuantizer 类型，则特殊初始化参数
        elif isinstance(module, Wav2Vec2ConformerGumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)  # 使用正态分布初始化权重
            module.weight_proj.bias.data.zero_()  # 将偏置初始化为零
            nn.init.uniform_(module.codevectors)  # 使用均匀分布初始化 codevectors
        # 如果 module 是 Wav2Vec2ConformerSelfAttention 类型，则根据属性初始化参数
        elif isinstance(module, Wav2Vec2ConformerSelfAttention):
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)  # 使用 Xavier 均匀分布初始化 pos_bias_u
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)  # 使用 Xavier 均匀分布初始化 pos_bias_v
        # 如果 module 是 Wav2Vec2ConformerPositionalConvEmbedding 类型，则使用正态分布初始化参数
        elif isinstance(module, Wav2Vec2ConformerPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,  # 卷积层权重初始化为正态分布
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)  # 卷积层偏置初始化为零
        # 如果 module 是 Wav2Vec2ConformerFeatureProjection 类型，则使用均匀分布初始化参数
        elif isinstance(module, Wav2Vec2ConformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)  # 投影层权重均匀初始化
            nn.init.uniform_(module.projection.bias, a=-k, b=k)  # 投影层偏置均匀初始化
        # 如果 module 是 nn.Linear 类型，则使用正态分布初始化权重，同时初始化偏置为零
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.LayerNorm 或 nn.GroupNorm 类型，则初始化偏置为零，权重为1
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果 module 是 nn.Conv1d 类型，则使用 Kaiming 正态分布初始化权重，初始化偏置为特定均匀分布
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)  # 使用 Kaiming 正态分布初始化卷积层权重
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)  # 使用均匀分布初始化卷积层偏置
    ):
        """
        计算卷积层的输出长度
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的一维卷积层输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            # 计算每个卷积核对应的输出长度
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            # 如果需要添加适配器层，则对每层适配器使用特定的卷积核大小和步长计算输出长度
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # 计算非填充部分的长度，即注意力掩码中每个序列的实际长度之和
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 获取特征向量提取器的输出长度
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        # 创建一个注意力掩码张量，用于控制哪些部分需要进行注意力
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # 确保在输出长度之前的所有位置都被注意到
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # 对注意力掩码进行翻转和累加操作，确保在输出长度之前的所有位置都被置为True
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
# WAV2VEC2_CONFORMER_START_DOCSTRING 是一个长字符串，用于存储 Wav2Vec2Conformer 模型的文档字符串。
WAV2VEC2_CONFORMER_START_DOCSTRING = r"""
    Wav2Vec2Conformer was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
    Auli.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) sub-class. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`Wav2Vec2ConformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# WAV2VEC2_CONFORMER_INPUTS_DOCSTRING 是另一个长字符串，用于存储 Wav2Vec2Conformer 模型输入的文档字符串。
WAV2VEC2_CONFORMER_INPUTS_DOCSTRING = r"""
    This docstring should detail the expected inputs of the Wav2Vec2Conformer model.
    It typically includes information on the type and shape of input tensors required
    for the model's forward pass, along with any additional context or constraints
    on the input data.

    Example:
        Inputs:
            - input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input tokens in the vocabulary.

    Note:
        This docstring should be completed to provide comprehensive guidance on how to
        format and prepare inputs for the model.
"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            # 输入的原始语音波形的浮点值。可以通过加载 `.flac` 或 `.wav` 音频文件并将其转换成 `List[float]` 或 `numpy.ndarray` 类型的数组获得。使用 `soundfile` 库 (`pip install soundfile`)。
            # 使用 [`AutoProcessor`] 进行填充和转换，生成 `torch.FloatTensor` 类型的张量 `input_values`。详见 [`Wav2Vec2Processor.__call__`]。
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于在填充标记索引上避免进行卷积和注意力操作。
            # 遮罩值选择在 `[0, 1]`：

            # - 1 表示**未遮罩**的标记，
            # - 0 表示**已遮罩**的标记。

            # [什么是注意力遮罩？](../glossary#attention-mask)

            <Tip warning={true}>
            # 只有在相应的处理器具有 `config.return_attention_mask == True` 时才应传递 `attention_mask`。对于所有处理器具有 `config.return_attention_mask == False` 的模型，如 [wav2vec2-conformer-rel-pos-large](https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large)，应避免传递 `attention_mask` 以避免在进行批量推理时性能下降。对于这些模型，`input_values` 应简单地填充为 0 并传递，而不使用 `attention_mask`。请注意，这些模型在 `input_values` 是否填充会稍有不同的结果。
            </Tip>

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回张量中的 `attentions` 以获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回张量中的 `hidden_states` 以获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
@add_start_docstrings(
    "The bare Wav2Vec2Conformer Model transformer outputting raw hidden-states without any specific head on top.",
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
class Wav2Vec2ConformerModel(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config: Wav2Vec2ConformerConfig):
        super().__init__(config)
        self.config = config
        # 初始化特征提取器
        self.feature_extractor = Wav2Vec2ConformerFeatureEncoder(config)
        # 初始化特征投影层
        self.feature_projection = Wav2Vec2ConformerFeatureProjection(config)

        # 如果配置中的掩码概率大于0.0，则需要初始化掩码向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 初始化编码器
        self.encoder = Wav2Vec2ConformerEncoder(config)

        # 如果配置要求添加适配器，则初始化适配器
        self.adapter = Wav2Vec2ConformerAdapter(config) if config.add_adapter else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model.freeze_feature_encoder复制而来
    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新。
        """
        self.feature_extractor._freeze_parameters()

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

        # `config.apply_spec_augment` 可以设置为 False 来禁用掩蔽
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # 根据是否提供了 `mask_time_indices`，选择是否沿时间轴应用 SpecAugment
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # 使用给定的 `mask_time_indices` 沿时间轴应用 SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 如果 `mask_time_indices` 未提供且训练模式下配置允许，则生成新的 `mask_time_indices` 并应用 SpecAugment
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
            # 如果训练模式下配置允许，则生成新的 `mask_feature_indices` 并沿特征轴应用 SpecAugment
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

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Wav2Vec2BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model.forward 复制而来，将 wav2vec2 改为 wav2vec2_conformer
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 确定是否输出注意力权重，如果未指定则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态，如果未指定则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典形式的输出，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征向量
        extract_features = self.feature_extractor(input_values)
        # 转置特征向量，调整维度顺序
        extract_features = extract_features.transpose(1, 2)

        # 如果存在注意力掩码，计算对应于特征向量的降维注意力掩码
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 特征投影
        hidden_states, extract_features = self.feature_projection(extract_features)
        
        # 对隐藏状态进行遮罩处理
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 编码器处理
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的隐藏状态输出
        hidden_states = encoder_outputs[0]

        # 如果存在适配器，应用适配器到隐藏状态上
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 如果不要求以字典形式返回结果，则返回元组形式的输出
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 以 Wav2Vec2BaseModelOutput 类型返回结果，包括最后的隐藏状态、提取的特征、编码器的隐藏状态、注意力权重
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """Wav2Vec2Conformer Model with a quantizer and `VQ` head on top.""", WAV2VEC2_CONFORMER_START_DOCSTRING
)
class Wav2Vec2ConformerForPreTraining(Wav2Vec2ConformerPreTrainedModel):
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.__init__ 复制而来，将类名和部分参数改为适应 Wav2Vec2Conformer 模型
    def __init__(self, config: Wav2Vec2ConformerConfig):
        super().__init__(config)
        # 初始化 Wav2Vec2ConformerModel 模型
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        # 定义特征量化器的 dropout 层
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        # 初始化 Wav2Vec2ConformerGumbelVectorQuantizer 量化器
        self.quantizer = Wav2Vec2ConformerGumbelVectorQuantizer(config)

        # 定义线性层用于投影隐藏状态到编码向量维度
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        # 定义线性层用于投影量化码向量维度
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.set_gumbel_temperature 复制而来
    def set_gumbel_temperature(self, temperature: int):
        """
        设置 Gumbel softmax 的温度值为给定值。仅在训练时需要。
        """
        self.quantizer.temperature = temperature

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.freeze_feature_encoder 复制而来，将函数名和部分参数改为适应 Wav2Vec2Conformer 模型
    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，以便在训练过程中不更新其参数。
        """
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    @staticmethod
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.compute_contrastive_logits 复制而来
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 0.1,
    ):
        """
        基于余弦相似度作为距离度量计算对比损失的 logits，计算方式为 `[positive_feature, negative_features]` 和 `[predicted_features]` 的相似度。
        可以应用温度参数调整。
        """
        # 将目标特征和负样本特征拼接在一起
        target_features = torch.cat([target_features, negative_features], dim=0)

        # 计算余弦相似度
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # 应用温度参数
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Wav2Vec2ConformerForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.forward方法复制而来，做了如下替换：
    # - Wav2Vec2 替换为 Wav2Vec2Conformer
    # - wav2vec2 替换为 wav2vec2_conformer
    # - wav2vec2_conformer-base 替换为 wav2vec2-conformer-rel-pos-large
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加模型文档字符串，描述了这是一个带有语言建模头部的 Wav2Vec2Conformer 模型，用于CTC（连接主义时间分类）任务。
@add_start_docstrings(
    """Wav2Vec2Conformer Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
class Wav2Vec2ConformerForCTC(Wav2Vec2ConformerPreTrainedModel):
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.__init__ 复制而来，将 Wav2Vec2 替换为 Wav2Vec2Conformer，wav2vec2 替换为 wav2vec2_conformer
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        # 创建 Wav2Vec2ConformerModel 模型
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        # 使用配置中指定的最终 dropout 率创建 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言
        self.target_lang = target_lang

        # 如果配置中未定义语言模型头部的词汇表大小，则抛出错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ConformerForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        # 根据配置设置输出隐藏层大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 创建线性层作为语言模型头部
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并进行最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.freeze_feature_encoder 复制而来，将 wav2vec2 替换为 wav2vec2_conformer
    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不会更新。
        """
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    # 从 add_start_docstrings_to_model_forward 和 add_code_sample_docstrings 复制而来，将 Wav2Vec2 替换为 Wav2Vec2Conformer，wav2vec2 替换为 wav2vec2_conformer
    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward 复制而来，将 Wav2Vec2 替换为 Wav2Vec2Conformer，wav2vec2 替换为 wav2vec2_conformer
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        ):
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 初始化 return_dict 变量，如果 return_dict 参数为 None，则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 wav2vec2_conformer 模型处理输入数据
        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态，并应用 dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 将隐藏状态通过 lm_head 网络得到预测的 logits
        logits = self.lm_head(hidden_states)

        # 初始化损失变量
        loss = None
        if labels is not None:
            # 检查标签值是否在合法范围内
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 根据注意力掩码获取输入长度
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充的标记为 -100，在计算损失时忽略这些标记
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # 计算 log-probabilities 并进行格式转换
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 使用 CTC 损失函数计算损失
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

        # 如果不要求返回字典，则返回 logits 和可能的其他输出状态
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回 CausalLMOutput 对象，其中包括损失、logits、隐藏状态和注意力权重
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 使用自定义的文档字符串装饰器为模型类添加描述和文档
@add_start_docstrings(
    """
    Wav2Vec2Conformer 模型，在顶部添加了一个序列分类头（一个线性层，用于池化输出），用于诸如 SUPERB 关键词检测之类的任务。
    """,
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
class Wav2Vec2ConformerForSequenceClassification(Wav2Vec2ConformerPreTrainedModel):
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.__init__ 复制而来，将 Wav2Vec2->Wav2Vec2Conformer，wav2vec2->wav2vec2_conformer
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中包含 add_adapter，并且其值为 True，则引发异常
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2Conformer adapters (config.add_adapter=True)"
            )
        
        # 创建 Wav2Vec2ConformerModel 对象
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        
        # 计算层数，包括 Transformer 层和输入嵌入层
        num_layers = config.num_hidden_layers + 1
        # 如果配置中设置了 use_weighted_layer_sum，则初始化层权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 定义分类器的线性投影层
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 定义分类器的线性分类层
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.freeze_feature_encoder 复制而来，将 wav2vec2->wav2vec2_conformer
    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不会更新。
        """
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    # 冻结基础模型，禁用基础模型的梯度计算，使其参数在训练过程中不会更新，只有分类头会被更新。
    def freeze_base_model(self):
        """
        调用此函数将禁用基础模型的梯度计算，使其参数在训练过程中不会更新。只有分类头会被更新。
        """
        for param in self.wav2vec2_conformer.parameters():
            param.requires_grad = False

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward 复制而来，将 Wav2Vec2->Wav2Vec2Conformer，wav2vec2->wav2vec2_conformer，WAV_2_VEC_2->WAV2VEC2_CONFORMER
    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        # 此处省略了函数定义的其他输入参数
        ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 确保返回字典不为空，使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果配置中使用加权层求和，则设置输出隐藏状态为真
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用 wav2vec2_conformer 模型进行处理
        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置中使用加权层求和，则根据指定位置获取隐藏状态并加权求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用第一个输出作为隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态投影到指定的维度
        hidden_states = self.projector(hidden_states)

        # 如果没有注意力掩码，则计算平均池化输出
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            # 否则根据注意力掩码生成填充掩码，并对隐藏状态进行掩码处理
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            # 计算池化输出，除以掩码元素数量以得到平均值
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 使用分类器生成 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果存在标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 如果不使用返回字典，则组装输出并返回
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回序列分类器输出对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Wav2Vec2Conformer Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
class Wav2Vec2ConformerForAudioFrameClassification(Wav2Vec2ConformerPreTrainedModel):
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.__init__ 复制而来，将 Wav2Vec2 替换为 Wav2Vec2Conformer，wav2vec2 替换为 wav2vec2_conformer，WAV_2_VEC_2 替换为 WAV2VEC2_CONFORMER
    def __init__(self, config):
        super().__init__(config)

        # 检查配置中是否有 add_adapter 属性且为 True，若是则抛出异常，因为音频帧分类不支持使用 Wav2Vec2Conformer 适配器
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2Conformer adapters (config.add_adapter=True)"
            )
        
        # 初始化 Wav2Vec2Conformer 模型
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        
        # 计算层数，包括 Transformer 层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  
        
        # 如果配置中使用加权层求和，则初始化层权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 初始化分类器，用于最终的帧分类任务
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        # 初始化模型权重
        self.init_weights()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.freeze_feature_encoder 复制而来，将 wav2vec2 替换为 wav2vec2_conformer
    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其在训练过程中不会更新参数。
        """
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.freeze_base_model 复制而来，将 wav2vec2 替换为 wav2vec2_conformer
    def freeze_base_model(self):
        """
        调用此函数将禁用基础模型的梯度计算，使其参数在训练过程中不会更新。只有分类头将会更新。
        """
        for param in self.wav2vec2_conformer.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.forward 复制而来，将 wav2vec2 替换为 wav2vec2_conformer
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        此函数实现模型的前向传播逻辑，接受输入值、注意力掩码等参数，并返回模型输出结果。

        Args:
            input_values (Optional[torch.Tensor]): 输入值张量。
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码张量，默认为 None。
            labels (Optional[torch.Tensor], optional): 标签张量，默认为 None。
            output_attentions (Optional[bool], optional): 是否输出注意力，默认为 None。
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态，默认为 None。
            return_dict (Optional[bool], optional): 是否返回字典格式的输出，默认为 None。
            **kwargs: 其他参数。

        Returns:
            模型输出结果。
        """
        # 在这里实现具体的前向传播逻辑
        pass
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化是否返回字典的标志，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏层状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用wav2vec2_conformer模型进行推理
        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置指定使用加权层求和机制
        if self.config.use_weighted_layer_sum:
            # 提取隐藏状态，并堆叠为张量
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            # 计算归一化的权重并应用到隐藏状态上
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用第一个输出作为隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态传入分类器得到logits
        logits = self.classifier(hidden_states)

        # 初始化损失值
        loss = None
        # 如果存在标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 对logits进行reshape并计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        # 如果不要求返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        # 返回TokenClassifierOutput对象，包含损失、logits、隐藏状态和注意力信息
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale  # 缩放因子，用于放大余弦相似度的值
        self.margin = margin  # 间隔参数，用于增加类别间的距离
        self.num_labels = num_labels  # 类别数目
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)  # 分类权重矩阵，随机初始化
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数，用于计算分类损失

    def forward(self, hidden_states, labels):
        labels = labels.flatten()  # 将标签展平，以便与预测结果匹配
        weight = nn.functional.normalize(self.weight, dim=0)  # 对权重进行L2归一化，保证数值稳定性
        hidden_states = nn.functional.normalize(hidden_states, dim=1)  # 对隐藏状态进行L2归一化，保证数值稳定性
        cos_theta = torch.mm(hidden_states, weight)  # 计算余弦相似度
        psi = cos_theta - self.margin  # 计算带有间隔参数的余弦相似度

        onehot = nn.functional.one_hot(labels, self.num_labels)  # 将标签转换为one-hot编码
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)  # 计算缩放后的预测分数
        loss = self.loss(logits, labels)  # 计算最终的损失值

        return loss


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]  # 输入维度
        self.out_conv_dim = config.tdnn_dim[layer_id]  # 输出维度
        self.kernel_size = config.tdnn_kernel[layer_id]  # 卷积核大小
        self.dilation = config.tdnn_dilation[layer_id]  # 膨胀率

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)  # 线性层，用于卷积
        self.activation = nn.ReLU()  # ReLU激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_peft_available():  # 检查是否存在PEFT库
            from peft.tuners.lora import LoraLayer

            if isinstance(self.kernel, LoraLayer):
                warnings.warn(
                    "Detected LoRA on TDNNLayer. LoRA weights won't be applied due to optimization. "
                    "You should exclude TDNNLayer from LoRA's target modules.",
                )

        # for backward compatibility, we keep nn.Linear but call F.conv1d for speed up
        hidden_states = hidden_states.transpose(1, 2)  # 转置隐藏状态的维度
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)  # 调整卷积核的形状
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)  # 一维卷积操作
        hidden_states = hidden_states.transpose(1, 2)  # 恢复隐藏状态的维度

        hidden_states = self.activation(hidden_states)  # 应用ReLU激活函数
        return hidden_states


@add_start_docstrings(
    """
    Wav2Vec2Conformer Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
class Wav2Vec2ConformerForXVector(Wav2Vec2ConformerPreTrainedModel):
    pass  # 没有额外的代码，只是为了提供类文档字符串的类定义
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)  # 初始化一个Wav2Vec2ConformerModel对象并赋值给self.wav2vec2_conformer
        num_layers = config.num_hidden_layers + 1  # 计算transformer层数加上输入嵌入层的总层数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)  # 如果配置使用加权层求和，则初始化权重参数
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])  # 初始化一个线性层self.projector

        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]  # 根据配置中tdnn_dim的定义创建TDNNLayer对象列表
        self.tdnn = nn.ModuleList(tdnn_layers)  # 将TDNNLayer对象列表封装为nn.ModuleList赋值给self.tdnn

        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)  # 初始化一个线性层self.feature_extractor
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)  # 初始化一个线性层self.classifier

        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)  # 初始化一个AMSoftmaxLoss对象self.objective

        self.init_weights()  # 调用init_weights方法进行初始化参数设置

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector.freeze_feature_encoder复制，并替换wav2vec2为wav2vec2_conformer
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()  # 调用Wav2Vec2ConformerModel中feature_extractor的_freeze_parameters方法冻结特征编码器参数

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector.freeze_base_model复制，并替换wav2vec2为wav2vec2_conformer
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_conformer.parameters():  # 遍历Wav2Vec2ConformerModel的所有参数
            param.requires_grad = False  # 将参数的梯度计算设置为False，即不更新这些参数的梯度

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector._get_tdnn_output_lengths复制，并替换wav2vec2为wav2vec2_conformer
    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.tdnn_kernel:  # 遍历配置中定义的TDNN核大小
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)  # 计算每个TDNN层的输出长度

        return input_lengths

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector.forward复制，并替换Wav2Vec2为Wav2Vec2Conformer,wav2vec2为wav2vec2_conformer,WAV_2_VEC_2为WAV2VEC2_CONFORMER
    # 定义模型的前向传播方法
    def forward(
        self,
        # 输入的张量值，可以为 None
        input_values: Optional[torch.Tensor],
        # 注意力掩码，可以为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出，默认为 None
        return_dict: Optional[bool] = None,
        # 标签数据的张量，可以为 None
        labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化 return_dict，如果未指定则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据 self.config.use_weighted_layer_sum 设置 output_hidden_states
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用 wav2vec2_conformer 模型，传入参数并获取输出
        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用加权层求和，则对隐藏状态进行加权求和操作
        if self.config.use_weighted_layer_sum:
            # 从 outputs 中获取隐藏状态
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 在第二维度上堆叠隐藏状态
            hidden_states = torch.stack(hidden_states, dim=1)
            # 计算归一化的权重
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 使用权重加权求和隐藏状态
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用 outputs 的第一个元素作为隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态投影到指定维度
        hidden_states = self.projector(hidden_states)

        # 通过循环对隐藏状态进行时间延迟神经网络层的前向传播
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # 统计池化操作
        if attention_mask is None:
            # 如果没有给定 attention_mask，则计算全局平均值和标准差
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            # 否则根据 attention_mask 计算特征提取器的输出长度和 TDNN 层的输出长度
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            # 遍历计算每个 TDNN 层的平均值和标准差
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        # 拼接平均特征和标准差特征
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        # 通过特征提取器提取统计池化特征的表示
        output_embeddings = self.feature_extractor(statistic_pooling)
        # 通过分类器生成 logits
        logits = self.classifier(output_embeddings)

        # 初始化 loss
        loss = None
        # 如果提供了标签，则计算损失值
        if labels is not None:
            loss = self.objective(logits, labels)

        # 如果 return_dict 为 False，则返回 logits、output_embeddings 和隐藏状态
        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 否则返回 XVectorOutput 对象，包含 loss、logits、embeddings、隐藏状态和注意力
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```