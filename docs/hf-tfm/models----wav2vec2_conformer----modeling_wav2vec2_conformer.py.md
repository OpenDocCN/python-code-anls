# `.\transformers\models\wav2vec2_conformer\modeling_wav2vec2_conformer.py`

```
# 设置文件编码格式为 utf-8
# 版权声明
# 使用 Apache License, Version 2.0 许可
# 查看许可证详情，请访问 http://www.apache.org/licenses/LICENSE-2.0
# 软件按 "AS IS" 基础分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取详细语言解释和限制
# PyTorch Wav2Vec2-Conformer 模型

# 导入必要的库和模块
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入相关对象和类型
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
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 Wav2Vec2ConformerConfig 配置
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig

# 配置日志记录器
logger = logging.get_logger(__name__)

# 定义隐藏状态起始位置
_HIDDEN_STATES_START_POSITION = 2

# 常用文档字符串
_CONFIG_FOR_DOC = "Wav2Vec2ConformerConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-conformer-rope-large-960h-ft"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 1024]

# CTC 文档字符串
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 64.21

# 预训练模型库列表
WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-conformer-rel-pos-large",
    # 查看所有 Wav2Vec2Conformer 模型，请访问 https://huggingface.co/models?filter=wav2vec2-conformer
]

# 数据类，用于定义 Wav2Vec2ConformerForPreTrainingOutput 的输出类型
@dataclass
class Wav2Vec2ConformerForPreTrainingOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ConformerForPreTraining`], with potential hidden states and attentions.
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

    # 定义各参数的类型和是否可选
    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None
# 从transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices中复制过来的函数，用于计算随机的掩码范围
def _compute_mask_indices(
    shape: Tuple[int, int],  # 定义传入参数的形状，一个大小为2的元组，包含批量大小和要计算掩码的轴的长度
    mask_prob: float,  # 掩码的概率，即整个轴的百分比（在0和1之间），将被掩盖，用来计算长度为mask_length的独立生成的掩码范围数量
    mask_length: int,  # 掩码的长度
    attention_mask: Optional[torch.LongTensor] = None,  # (右填充的) 注意力掩码，可以独立地缩短每个批量维度的特征轴
    min_masks: int = 0,  # 掩码的最小数量
) -> np.ndarray:  # 返回一个numpy数组
    """
    计算给定形状的随机掩码范围。用于实现SpecAugment：用于ASR的简单数据增强方法。请注意，此方法未经过优化，应作为训练过程中预处理的一部分在CPU上运行。

    Args:
        shape: 需要计算掩码的形状。这应该是大小为2的元组，其中第一个元素是批量大小，第二个元素是要跨度的轴的长度。
        mask_prob: 整个轴的百分比（在0和1之间）将被掩盖，掩码长度为mask_length的独立生成的掩码范围数量由mask_prob*shape[1]/mask_length计算得出。
                  注意，由于重叠，mask_prob是一个上限，实际百分比会更小。
        mask_length: 掩码的长度
        min_masks: 掩码的最小数量
        attention_mask: (右填充的) 注意力掩码，可以独立地缩短每个批量维度的特征轴。
    """
    batch_size, sequence_length = shape  # 将shape的两个值分别赋给批量大小和序列长度

    if mask_length < 1:  # 如果掩码长度小于1，则抛出异常
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:  # 如果掩码长度大于序列长度，则抛出异常
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon用于概率舍入
    epsilon = np.random.rand(1).item()  # 生成一个随机数，并取出其中的元素值

    def compute_num_masked_span(input_length):
        """给定输入长度，计算应掩盖多少个范围"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)  # 计算应该掩盖多少个范围
        num_masked_span = max(num_masked_span, min_masks)  # 取最大值，确保不小于最小掩码数量

        # 确保掩盖的范围数量不大于序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保掩盖的范围数量不大于输入长度 - (掩码长度 - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span  # 返回掩盖的范围数量

    # 计算批量中掩盖的范围数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()  # 如果有attention_mask，则计算每个批量维度的特征轴的输入长度，然后将结果转换为列表
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]  # 如果没有attention_mask，则将批量大小个长度为序列长度的列表作为输入长度
    )

    # SpecAugment掩码初始化
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 初始化一个布尔类型的二维数组
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算掩盖的最大范围数量
    # 如果最大被遮蔽跨度数为0，则返回spectral augmentation mask
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算该输入的被遮蔽跨度数
        num_masked_span = compute_num_masked_span(input_length)

        # 获取随机索引以进行遮蔽
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择作为填充向量的虚拟索引，确保由于概率舍入而对所有批次都具有相同的维度
        # 仅选择第一个样本将两次填充这些向量
        if len(spec_aug_mask_idx) == 0:
            # 当`input_length`严格小于`sequence_length`时，这种情况只能发生
            # 最后一个标记必须是一个填充标记，我们可以使用它作为虚拟遮蔽ID
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 连接虚拟索引以及将其余部分填充为虚拟索引的最大数
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将遮蔽索引扩展为遮蔽跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量以使起始索引成为一个跨度索引
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保我们的索引不能大于sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 分散索引来进行遮蔽
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
# 从transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices中复制过来的函数
# 从特征向量中随机采样出num_negatives个向量
def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    # 计算features_shape的形状，得到batch_size和sequence_length
    batch_size, sequence_length = features_shape

    # 生成正向量自身的索引，将其重复`num_negatives`次
    sequence_length_range = np.arange(sequence_length)

    # 从同一个utterance中随机采样出`num_negatives`个向量的索引
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    # 如果mask_time_indices不为None，则按照mask_time_indices中的索引进行掩码
    # 如果mask_time_indices为None，则掩码全为真
    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    # 遍历每个batch
    for batch_idx in range(batch_size):
        # 计算mask_time_indices[batch_idx]中为真的元素数目减1
        high = mask_time_indices[batch_idx].sum() - 1
        # 得到被掩码的索引
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        # 生成高度为high+1，宽度为num_negatives的feature_indices数组
        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        # 从feature_indices中随机采样，生成高度为high+1，宽度为num_negatives的随机数矩阵sampled_indices
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # 避免采样到相同的正向量，但保持分布均匀
        sampled_indices[sampled_indices >= feature_indices] += 1

        # 映射到实际的索引
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # 将batch size纠正回去
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer中复制过来的类，将Wav2Vec2换成Wav2Vec2Conformer
class Wav2Vec2ConformerNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果layer_id大于0，则设置输入维度为config.conv_dim[layer_id-1]，否则输入维度为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出维度为config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 使用nn.Conv1d进行卷积，设置输入维度为self.in_conv_dim，输出维度为self.out_conv_dim
        # 设置卷积核大小为config.conv_kernel[layer_id]，步长为config.conv_stride[layer_id]，偏置为config.conv_bias
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数为config.feat_extract_activation所对应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将输入hidden_states进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 使用激活函数对hidden_states进行激活
        hidden_states = self.activation(hidden_states)
        # 返回结果
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer中复制过来的类，将Wav2Vec2换成Wav2Vec2Conformer
class Wav2Vec2ConformerLayerNormConvLayer(nn.Module):
    # 初始化卷积层对象，设置输入维度、输出维度、卷积核大小和步长
    def __init__(self, config, layer_id=0):
        # 调用父类的初始化方法
        super().__init__()
        # 根据给定的层级 ID 选择输入维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 根据给定的层级 ID 选择输出维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层对象
        self.conv = nn.Conv1d(
            self.in_conv_dim,  # 输入维度
            self.out_conv_dim,  # 输出维度
            kernel_size=config.conv_kernel[layer_id],  # 卷积核大小
            stride=config.conv_stride[layer_id],  # 步长
            bias=config.conv_bias,  # 是否包含偏置
        )
        # 创建一个 LayerNorm 层对象，用于对输出进行标准化
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播方法
    def forward(self, hidden_states):
        # 对输入进行卷积操作
        hidden_states = self.conv(hidden_states)

        # 将输出的维度转换，方便进行 LayerNorm 操作
        hidden_states = hidden_states.transpose(-2, -1)
        # 对输出进行 LayerNorm 操作
        hidden_states = self.layer_norm(hidden_states)
        # 再次转换维度
        hidden_states = hidden_states.transpose(-2, -1)

        # 对输出进行激活函数操作
        hidden_states = self.activation(hidden_states)
        # 返回处理后的结果
        return hidden_states
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->Wav2Vec2Conformer
# 创建Wav2Vec2ConformerGroupNormConvLayer类，继承自nn.Module
class Wav2Vec2ConformerGroupNormConvLayer(nn.Module):
    # 初始化函数，接受config和layer_id两个参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 获取输入通道数
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 获取输出通道数
        self.out_conv_dim = config.conv_dim[layer_id]
        
        # 创建一维卷积层，卷积核大小为config.conv_kernel[layer_id]，步长为config.conv_stride[layer_id]，偏置为config.conv_bias
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 获取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
        
        # 创建归一化层，分组数为self.out_conv_dim，通道数为self.out_conv_dim，仅对仿射变换中的偏置和缩放进行训练
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    # 前向传播函数，接受hidden_states作为输入
    def forward(self, hidden_states):
        # 经过卷积层
        hidden_states = self.conv(hidden_states)
        # 经过归一化层
        hidden_states = self.layer_norm(hidden_states)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states

# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->Wav2Vec2Conformer
# 创建Wav2Vec2ConformerPositionalConvEmbedding类，继承自nn.Module
class Wav2Vec2ConformerPositionalConvEmbedding(nn.Module):
    # 初始化函数，接受config作为参数
    def __init__(self, config):
        super().__init__()
        # 创建一维卷积层，输入通道数和输出通道数都为config.hidden_size，卷积核大小为config.num_conv_pos_embeddings，padding为config.num_conv_pos_embeddings // 2，分组数为config.num_conv_pos_embedding_groups
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        
        # 通过WeightNorm对卷积层进行权重归一化
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)
        
        # 创建Wav2Vec2ConformerSamePadLayer实例
        self.padding = Wav2Vec2ConformerSamePadLayer(config.num_conv_pos_embeddings)
        # 获取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数，接受hidden_states作为输入
    def forward(self, hidden_states):
        # 调整hidden_states的维度
        hidden_states = hidden_states.transpose(1, 2)
        
        # 经过卷积层
        hidden_states = self.conv(hidden_states)
        # 经过padding
        hidden_states = self.padding(hidden_states)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)

        # 调整hidden_states的维度
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

# 创建Wav2Vec2ConformerRotaryPositionalEmbedding类，继承自nn.Module
class Wav2Vec2ConformerRotaryPositionalEmbedding(nn.Module):
    # Rotary positional embedding方法
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """
    pass  # 空实现，功能未提供详细注释信息
    # 初始化函数，接收配置参数并调用父类的初始化方法
    def __init__(self, config):
        super().__init__()
        # 根据配置计算每个注意力头的隐藏大小
        dim = config.hidden_size // config.num_attention_heads
        # 获取旋转嵌入的基数
        base = config.rotary_embedding_base
        # 计算频率的倒数
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # 将频率的倒数注册为缓冲区
        self.register_buffer("inv_freq", inv_freq)
        # 初始化缓存的序列长度和旋转位置嵌入为 None
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    # 前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 获取隐藏状态的序列长度
        sequence_length = hidden_states.shape[1]

        # 如果序列长度与缓存的序列长度相同且已缓存的旋转位置嵌入不为 None，则直接返回缓存的旋转位置嵌入
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        # 更新缓存的序列长度
        self.cached_sequence_length = sequence_length
        # 在频率常数的数据类型中计算时间戳
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        # 计算频率矩阵
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        # 拼接正弦和余弦的嵌入矩阵
        embeddings = torch.cat((freqs, freqs), dim=-1)

        # 计算余弦嵌入
        cos_embeddings = embeddings.cos()[:, None, None, :]
        # 计算正弦嵌入
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # 将计算得到的嵌入转换为和隐藏状态输入相同的数据类型
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        return self.cached_rotary_positional_embedding
class Wav2Vec2ConformerRelPositionalEmbedding(nn.Module):
    """相对位置编码模块。"""

    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_source_positions
        self.d_model = config.hidden_size
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        # 重置位置编码
        if self.pe is not None:
            # self.pe 包含正负两部分
            # self.pe 的长度为 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # 假设`i`是查询向量的位置，`j`是键向量的位置。当键在左侧时（i>j）使用正相对位置，反之使用负相对位置。
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # 反转正索引的顺序，并连接正负索引。这用于支持类似于 https://arxiv.org/abs/1901.02860 中的位移技巧
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


class Wav2Vec2ConformerSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states
class Wav2Vec2ConformerFeatureEncoder(nn.Module):
    """从原始音频波形中构建特征"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2ConformerGroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2ConformerNoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Wav2Vec2ConformerLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # 确保需要梯度的 hidden_states 用于梯度检查点
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection 复制并修改为 Wav2Vec2ConformerFeatureProjection
class Wav2Vec2ConformerFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 需要非投影的 hidden states 用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward 复制并修改为 Wav2Vec2ConformerFeedForward
class Wav2Vec2ConformerFeedForward(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个中间层的 Dropout 层，根据配置中的激活函数的 Dropout
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 创建一个全连接层，输入大小为隐藏层大小，输出大小为中间层大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中的隐藏层激活函数标识为字符串，则使用 ACT2FN 字典中对应值作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用配置中的隐藏层激活函数
            self.intermediate_act_fn = config.hidden_act

        # 创建一个输出全连接层，输入大小为中间层大小，输出大小为隐藏层大小
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个输出层的 Dropout 层，根据配置中的隐藏层的 Dropout
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播函数，输入为隐藏层的状态
    def forward(self, hidden_states):
        # 经过中间全连接层
        hidden_states = self.intermediate_dense(hidden_states)
        # 经过中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 经过中间层的 Dropout
        hidden_states = self.intermediate_dropout(hidden_states)

        # 经过输出全连接层
        hidden_states = self.output_dense(hidden_states)
        # 经过输出层的 Dropout
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
class Wav2Vec2ConformerConvolutionModule(nn.Module):
    """定义在conformer块中使用的卷积块"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            # 检查config.conv_depthwise_kernel_size是否是奇数，因为对于'SAME'填充来说应该是奇数
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        # pointwise卷积层1
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # GLU激活函数
        self.glu = nn.GLU(dim=1)
        # 深度卷积层
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=(config.conv_depthwise_kernel_size - 1) // 2,
            groups=config.hidden_size,
            bias=False,
        )
        # 批归一化层
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        # 激活函数
        self.activation = ACT2FN[config.hidden_act]
        # pointwise卷积层2
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # dropout层
        self.dropout = nn.Dropout(config.conformer_conv_dropout)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        # 交换时间维度和特征维度
        hidden_states = hidden_states.transpose(1, 2)

        # GLU机制
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # 1D深度卷积
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2ConformerSelfAttention(nn.Module):
    """构建一个Wav2Vec2ConformerSelfAttention对象。
    可以加入旋转或相对位置嵌入来增强功能。
    """
    # 初始化函数，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
    
        # 计算每个头部的大小
        self.head_size = config.hidden_size // config.num_attention_heads
        # 头部的数量
        self.num_heads = config.num_attention_heads
        # 位置编码类型
        self.position_embeddings_type = config.position_embeddings_type
    
        # 创建线性层，分别用于查询、键、值和输出
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
    
        # 创建丢弃层，用于注意力机制的丢弃
        self.dropout = nn.Dropout(p=config.attention_dropout)
    
        # 如果位置编码类型是相对位置编码
        if self.position_embeddings_type == "relative":
            # 用于位置编码的线性变换
            self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            # 学习的偏置，用于矩阵c和矩阵d
            # 参考论文https://arxiv.org/abs/1901.02860 第3.3节
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
    
    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 设置函数的输入类型和返回类型，这里返回三个值的元组

        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # 获取隐藏状态的批大小、序列长度和隐藏层大小

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states
        # 确保查询/键状态可以不等于值状态

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)
        # 如果位置嵌入类型为“rotary”，则确保相对位置嵌入不为空。然后应用旋转嵌入

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)
        # 投影查询/键状态和值状态

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # 转置查询、键和值，改变其形状

        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type =="
                    " 'relative'"
                )
            # apply relative_position_embeddings to qk scores
            # as proposed in Transformer_XL: https://arxiv.org/abs/1901.02860
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        # 如果位置嵌入类型是“relative”，则确保相对位置嵌入不为空。然后应用相对位置嵌入到qk得分中；否则，计算q和k的乘积，并除以头部尺寸的平方根。

        # apply attention_mask if necessary
        if attention_mask is not None:
            scores = scores + attention_mask
        # 必要时应用注意力掩码

        # => (batch, head, time1, time2)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        # 计算softmax概率，应用dropout

        # => (batch, head, time1, d_k)
        hidden_states = torch.matmul(probs, value)
        # 计算加权和，得到注意力输出

        # => (batch, time1, hidden_size)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)
        # 转置隐藏状态，重塑形状，然后将其传递到线性输出层

        return hidden_states, probs
        # 返回隐藏状态和概率
    # 在 hidden_states 上应用旋转嵌入，并考虑相对位置嵌入
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        # 获取隐藏状态的形状信息
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # 将 hidden_states 重新调整形状
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        # 提取相对位置嵌入的 cos 和 sin 值
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # 用旋转嵌入旋转 hidden_states
        hidden_states = hidden_states.transpose(0, 1)
        # 拆分 hidden_states 中的前半部分和后半部分
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        # 拼接旋转后的前半部分和后半部分
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        # 使用 cos 和 sin 进行线性组合
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        # 将 hidden_states 重新调整形状，恢复原始形状
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        return hidden_states
    # 对相对位置嵌入进行应用
    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        # 1. 投影位置嵌入
        # => (batch, head, 2*time1-1, d_k)
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

        # 2. 为查询添加偏置
        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        # 3. 注意力分数：首先计算矩阵 a 和矩阵 c
        # 如 https://arxiv.org/abs/1901.02860 第3.3节描述
        # => (batch, head, time1, time2)
        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        # 4. 然后计算矩阵 b 和矩阵 d
        # => (batch, head, time1, 2*time1-1)
        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # 5. 移动矩阵 b 和矩阵 d
        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.size(-1) // 2 + 1]

        # 6. 求和矩阵
        # => (batch, head, time1, time2)
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores
# 定义一个名为Wav2Vec2ConformerEncoderLayer的类，继承自nn.Module
class Wav2Vec2ConformerEncoderLayer(nn.Module):
    # 在初始化函数中设置了一些模型参数
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # 创建Feed-forward 1层的LayerNorm和FeedForward对象
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = Wav2Vec2ConformerFeedForward(config)

        # 创建Self-Attention层的LayerNorm和Dropout，并定义Self-Attention层对象
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = Wav2Vec2ConformerSelfAttention(config)

        # 创建Conformer Convolution模块对象
        self.conv_module = Wav2Vec2ConformerConvolutionModule(config)

        # 创建Feed-forward 2层的LayerNorm和FeedForward对象，以及最后的LayerNorm对象
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = Wav2Vec2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 将输入的hidden_states进行赋值
        hidden_states = hidden_states

        # 1. Feed-Forward 1层
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention层
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weigts = self.self_attn(
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

        # 4. Feed-Forward 2层
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        # 返回结果hidden_states和attn_weigts
        return hidden_states, attn_weigts

# 定义一个名为Wav2Vec2ConformerEncoder的类，继承自nn.Module
class Wav2Vec2ConformerEncoder(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 将配置参数保存在对象中
        self.config = config
    
        # 根据配置参数中的位置嵌入类型，选择不同的位置嵌入方式
        if config.position_embeddings_type == "relative":
            self.embed_positions = Wav2Vec2ConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = Wav2Vec2ConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None
    
        # 创建位置卷积嵌入
        self.pos_conv_embed = Wav2Vec2ConformerPositionalConvEmbedding(config)
        # 创建 LayerNorm 层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建由多个 Conformer 编码层组成的列表
        self.layers = nn.ModuleList([Wav2Vec2ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点开关
        self.gradient_checkpointing = False
    
    # 前向传播函数
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask=None,  # 注意力掩码
        output_attentions=False,  # 是否输出注意力权重
        output_hidden_states=False,  # 是否输出隐藏状态
        return_dict=True,  # 返回结果方式，默认为字典
        ):
            # 初始化所有隐藏状态为元组，如果不需要输出隐藏状态则为 None
            all_hidden_states = () if output_hidden_states else None
            # 初始化所有注意力权重为元组，如果不需要输出注意力权重则为 None
            all_self_attentions = () if output_attentions else None

            if attention_mask is not None:
                # 确保填充的标记输出为 0
                hidden_states[~attention_mask] = 0.0

                # 扩展注意力遮罩
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

            # 在隐藏状态上应用丢弃操作
            hidden_states = self.dropout(hidden_states)

            if self.embed_positions is not None:
                # 计算相对位置嵌入
                relative_position_embeddings = self.embed_positions(hidden_states)
            else:
                relative_position_embeddings = None

            # 检查是否启用了 DeepSpeed Zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

            # 遍历所有层
            for i, layer in enumerate(self.layers):
                if output_hidden_states:
                    # 如果需要输出隐藏状态，则将当前隐藏状态添加到列表中
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 添加 LayerDrop（参考 https://arxiv.org/abs/1909.11556）
                dropout_probability = torch.rand([])

                # 判断是否跳过当前层
                skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    if self.gradient_checkpointing and self.training:
                        # 使用渐变检查点功能运行当前层
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                            relative_position_embeddings,
                            output_attentions,
                        )
                    else:
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            relative_position_embeddings=relative_position_embeddings,
                            output_attentions=output_attentions,
                        )
                    hidden_states = layer_outputs[0]

                if skip_the_layer:
                    layer_outputs = (None, None)

                if output_attentions:
                    # 如果需要输出注意力权重，则将当前注意力权重添加到列表中
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 应用层归一化
            hidden_states = self.layer_norm(hidden_states)
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到列表中
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                # 如果不需要返回字典格式的输出，则返回元组格式
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 返回包含指定格式的 BaseModelOutput 对象
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GumbelVectorQuantizer复制，并将Wav2Vec2改为Wav2Vec2Conformer
class Wav2Vec2ConformerGumbelVectorQuantizer(nn.Module):
    """
    使用 Gumbel softmax 进行向量量化。更多信息请参阅《CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX》
    """

    def __init__(self, config):
        super().__init__()
        # 设置量化的组数
        self.num_groups = config.num_codevector_groups
        # 设置每组的量化变量数
        self.num_vars = config.num_codevectors_per_group

        # 如果 codevector_dim 不能被 num_groups 整除，抛出 ValueError
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # 存储码书变量（码字）的存储空间
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        # 权重投影
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # 可以进行训练过程中的衰减
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        # 如果给定了掩码
        if mask is not None:
            # 将掩码扩展以匹配概率的形状
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            # 使用掩码对概率进行替换
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            # 计算边际概率
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            # 计算概率的均值
            marginal_probs = probs.mean(dim=0)

        # 计算困惑度
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity
    # 定义一个前向传播函数，接受隐藏状态和可选的时间索引掩码
    def forward(self, hidden_states, mask_time_indices=None):
        # 获取隐藏状态张量的形状信息
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到码向量维度
        hidden_states = self.weight_proj(hidden_states)
        # 重新塑造隐藏状态张量的形状
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 在可微分的方式中，通过 Gumbel 分布对隐藏状态进行采样得到码向量概率分布
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # 计算困惑度
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # 以非可微分的方式取码向量概率分布的 argmax
            # 计算硬码向量分布（one-hot）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        # 重新塑造码向量概率分布张量的形状
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率分布检索码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        # 返回码向量和困惑度
        return codevectors, perplexity
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Adapter with Wav2Vec2->Wav2Vec2Conformer
# 定义一个名为 Wav2Vec2ConformerAdapter 的类，继承自 nn.Module
class Wav2Vec2ConformerAdapter(nn.Module):
    # 定义初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 如果 config 中的 output_hidden_size 不等于 hidden_size，则需要进行降维处理
        if config.output_hidden_size != config.hidden_size:
            # 创建一个线性层，用于降维
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # 创建一个 LayerNorm 层，用于处理降维后的数据
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            # 否则将降维相关属性设为 None
            self.proj = self.proj_layer_norm = None

        # 创建一个包含多个 Wav2Vec2ConformerAdapterLayer 实例的 ModuleList
        self.layers = nn.ModuleList(Wav2Vec2ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers))
        # 设置 layerdrop 参数
        self.layerdrop = config.layerdrop

    # 定义前向传播方法，接受 hidden_states 参数
    def forward(self, hidden_states):
        # 如果需要降维，则进行降维操作
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 将 hidden_states 调换维度顺序
        hidden_states = hidden_states.transpose(1, 2)

        # 遍历每个层并应用 layerdrop 操作
        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        # 再次将 hidden_states 调换维度顺序
        hidden_states = hidden_states.transpose(1, 2)
        # 返回处理后的 hidden_states
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AdapterLayer with Wav2Vec2->Wav2Vec2Conformer
# 定义一个名为 Wav2Vec2ConformerAdapterLayer 的类，继承自 nn.Module
class Wav2Vec2ConformerAdapterLayer(nn.Module):
    # 定义初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个卷积层，用于特征提取
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    # 定义前向传播方法，接受 hidden_states 参数
    def forward(self, hidden_states):
        # 将 hidden_states 通过卷积层处理
        hidden_states = self.conv(hidden_states)
        # 对处理后的 hidden_states 应用激活函数
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个名为 Wav2Vec2ConformerPreTrainedModel 的类，继承自 PreTrainedModel
class Wav2Vec2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置 config_class 属性为 Wav2Vec2ConformerConfig
    config_class = Wav2Vec2ConformerConfig
    # 设置 base_model_prefix 属性为 "wav2vec2_conformer"
    base_model_prefix = "wav2vec2_conformer"
    # 设置 main_input_name 属性为 "input_values"
    main_input_name = "input_values"
    # 设置 supports_gradient_checkpointing 属性为 True
    supports_gradient_checkpointing = True
    # 初始化权重的函数，针对不同类型的模块进行不同的权重初始化操作
    def _init_weights(self, module):
        """Initialize the weights"""
        # 对于 Wav2Vec2ConformerForPreTraining 类型的模块，需要初始化最后两个线性层
        if isinstance(module, Wav2Vec2ConformerForPreTraining):
            # 重置隐藏层和查询层的参数
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()
            # 标记隐藏层和查询层已经初始化
            module.project_hid._is_hf_initialized = True
            module.project_q._is_hf_initialized = True
        # 对于 Wav2Vec2ConformerGumbelVectorQuantizer 类型的模块，需要特殊初始化
        elif isinstance(module, Wav2Vec2ConformerGumbelVectorQuantizer):
            # 使用正态分布初始化权重和偏置
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            # 使用均匀分布初始化 codevectors
            nn.init.uniform_(module.codevectors)
        # 对于 Wav2Vec2ConformerSelfAttention 类型的模块
        elif isinstance(module, Wav2Vec2ConformerSelfAttention):
            # 如果有位置偏置 pos_bias_u，使用 xavier_uniform_ 方法初始化
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            # 如果有位置偏置 pos_bias_v，使用 xavier_uniform_ 方法初始化
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        # 对于 Wav2Vec2ConformerPositionalConvEmbedding 类型的模块
        elif isinstance(module, Wav2Vec2ConformerPositionalConvEmbedding):
            # 使用正态分布初始化卷积层的权重和偏置
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        # 对于 Wav2Vec2ConformerFeatureProjection 类型的模块
        elif isinstance(module, Wav2Vec2ConformerFeatureProjection):
            # 计算初始化范围
            k = math.sqrt(1 / module.projection.in_features)
            # 使用均匀分布初始化投影层的权重和偏置
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 对于 nn.Linear 类型的模块
        elif isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            # 如果存在偏置，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 对于 nn.LayerNorm 或 nn.GroupNorm 类型的模块
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为 1.0
            module.weight.data.fill_(1.0)
        # 对于 nn.Conv1d 类型的模块
        elif isinstance(module, nn.Conv1d):
            # 使用 kaiming_normal_ 方法初始化权重
            nn.init.kaiming_normal_(module.weight)

            # 如果存在偏置，使用均匀分布初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # 获取特征提取器的输出长度
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    # 计算卷积层的输出长度
    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.Tensor, add_adapter=None
    ):
        # 如果没有指定 add_adapter，则使用配置中的 add_adapter
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        # 定义计算卷积层输出长度的函数
        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 中拿到的1D卷积层输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历卷积核大小和步长，计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果有添加适配器网络层
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    # 获取特征向量注意力遮罩
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # 计算非填充长度，等效于 attention_mask.sum(-1)，但不是原地操作以便在推断模式下运行
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 获取特征提取输出长度
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        # 创建全零的注意力遮罩
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # 确保输出长度之前的值都被关注到
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # 翻转、累积、再翻转注意力遮罩，确保输出长度之后的值都不被关注到
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
```  
# WAV2VEC2_CONFORMER_START_DOCSTRING 对象的文档字符串，描述了 Wav2Vec2Conformer 模型的相关信息和用法
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

# WAV2VEC2_CONFORMER_INPUTS_DOCSTRING 对象的文档字符串，暂时为空
WAV2VEC2_CONFORMER_INPUTS_DOCSTRING = r"""
    # 定义输入值参数
    Args:
        # 输入原始语音波形的浮点张量, 形状为 (batch_size, sequence_length)
        # 数值可以通过将 .flac 或 .wav 音频文件加载到 List[float] 或 numpy.ndarray 中获得
        # 使用 AutoProcessor 进行填充和转换为 torch.FloatTensor 类型的张量
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.
        # 注意力掩码张量, 形状为 (batch_size, sequence_length), 可选
        # 用于避免在填充令牌上执行卷积和注意力操作
        # 值为 0 表示被遮掩, 1 表示未被遮掩
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0, 
            1]`:
    
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
            [What are attention masks?](../glossary#attention-mask)
    
            <Tip warning={true}>
    
            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [wav2vec2-conformer-rel-pos-large](https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large),
            `attention_mask` should **not** be passed to avoid degraded performance when doing batched inference. For
            such models `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware
            that these models also yield slightly different results depending on whether `input_values` is padded or
            not.
    
            </Tip>
    
        # 是否返回所有注意力层的注意力张量, 可选
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        # 是否返回所有层的隐藏状态, 可选
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # 是否返回 ModelOutput 对象, 而不是普通元组, 可选
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
定义一个 Wav2Vec2ConformerModel 类，继承自 Wav2Vec2ConformerPreTrainedModel 类
并且添加了一些文档字符串说明。
"""
@add_start_docstrings(
    "The bare Wav2Vec2Conformer Model transformer outputting raw hidden-states without any specific head on top.",
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
class Wav2Vec2ConformerModel(Wav2Vec2ConformerPreTrainedModel):
    def __init__(self, config: Wav2Vec2ConformerConfig):
        """
        初始化函数，接受一个配置对象作为参数。

        参数:
            config (Wav2Vec2ConformerConfig): 配置对象

        返回:
            None
        """
        super().__init__(config)
        self.config = config
        # 实例化特征提取器
        self.feature_extractor = Wav2Vec2ConformerFeatureEncoder(config)
        # 实例化特征投影层
        self.feature_projection = Wav2Vec2ConformerFeatureProjection(config)

        # 如果配置中的 mask_prob 大于 0.0，则模型需要掩码向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            # 初始化掩码向量参数
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 实例化编码器
        self.encoder = Wav2Vec2ConformerEncoder(config)

        # 如果配置中包含 adapter，则实例化适配器
        self.adapter = Wav2Vec2ConformerAdapter(config) if config.add_adapter else None

        # 初始化权重并进行最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model.freeze_feature_encoder 复制而来
    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，以便其参数在训练期间不会更新。
        """
        # 冻结特征编码器的参数
        self.feature_extractor._freeze_parameters()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states 复制而来
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    # 设置函数注释和描述
    def forward(
        self,
        # 输入的特征值，可以是 None 或者 torch.Tensor 类型
        input_values: Optional[torch.Tensor],
        # 注意力遮罩，可以是 None 或者 torch.Tensor 类型
        attention_mask: Optional[torch.Tensor] = None,
        # 沿时间轴的遮罩索引，可以是 None 或者 torch.FloatTensor 类型
        mask_time_indices: Optional[torch.FloatTensor] = None,
        # 是否返回注意力权重，可以是 None 或者 bool 类型
        output_attentions: Optional[bool] = None,
        # 是否返回隐藏层状态，可以是 None 或者 bool 类型
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出，可以是 None 或者 bool 类型
        return_dict: Optional[bool] = None,
        # 如果未指定输出注意力的话，设定为配置中的输出注意力
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态的话，设定为配置中的输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典的话，设定为使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取输入值的特征
        extract_features = self.feature_extractor(input_values)
        # 转置特征以便后续处理
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # 计算对应于特征向量的缩减注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 进行特征映射
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 对隐藏状态进行处理
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 进行编码器的前向传播
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        # 如果存在适配器，对隐藏状态进行处理
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 如果不使用返回字典，则返回隐藏状态、提取特征和编码器输出的其余部分
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回 Wav2Vec2BaseModelOutput 对象，包括隐藏状态、提取特征、隐藏状态和注意力
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 给 Wav2Vec2ConformerForPreTraining 类添加文档字符串，描述了使用量化器和 VQ 头的 Wav2Vec2Conformer 模型
@add_start_docstrings(
    """Wav2Vec2Conformer Model with a quantizer and `VQ` head on top.""", WAV2VEC2_CONFORMER_START_DOCSTRING
)
class Wav2Vec2ConformerForPreTraining(Wav2Vec2ConformerPreTrainedModel):
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.__init__ 复制而来，将参数 Wav2Vec2 修改为 Wav2Vec2Conformer，将 wav2vec2 修改为 wav2vec2_conformer
    def __init__(self, config: Wav2Vec2ConformerConfig):
        super().__init__(config)
        # 实例化 Wav2Vec2ConformerModel，并赋给 wav2vec2_conformer 属性
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        # 添加特征 dropout 的模块
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        # 实例化 Wav2Vec2ConformerGumbelVectorQuantizer，并赋给 quantizer 属性
        self.quantizer = Wav2Vec2ConformerGumbelVectorQuantizer(config)

        # 实例化线性层，并赋给 project_hid 和 project_q 属性
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.set_gumbel_temperature 复制而来
    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        # 设置 Gumbel 温度
        self.quantizer.temperature = temperature

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.freeze_feature_encoder 复制而来，将 wav2vec2 修改为 wav2vec2_conformer
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器，使其在训练过程中参数不会被更新
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
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        # 将目标特征和负特征拼接在一起
        target_features = torch.cat([target_features, negative_features], dim=0)

        # 计算余弦相似度作为距离度量，并将结果类型转换为目标特征的类型
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # 应用温度
        logits = logits / temperature
        return logits

    # 为 model_forward 方法添加文档字符串
    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    # 替换返回结果的文档字符串
    @replace_return_docstrings(output_type=Wav2Vec2ConformerForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining.forward复制而来，该函数已被修改为Wav2Vec2Conformer。
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入值，类型为torch.Tensor，可选参数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，类型为torch.Tensor，可选参数，默认为None
        mask_time_indices: Optional[torch.BoolTensor] = None,  # 时间索引掩码，类型为torch.BoolTensor，可选参数，默认为None
        sampled_negative_indices: Optional[torch.BoolTensor] = None,  # 负采样的索引，类型为torch.BoolTensor，可选参数，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为bool，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为bool，可选参数，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，类型为bool，可选参数，默认为None
# 引入必要的库
@add_start_docstrings(
    """Wav2Vec2Conformer Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
# 定义一个继承自Wav2Vec2ConformerPreTrainedModel的类Wav2Vec2ConformerForCTC
class Wav2Vec2ConformerForCTC(Wav2Vec2ConformerPreTrainedModel):
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.__init__复制代码，并修改相关命名
    # 初始化函数，接受配置config和可选的目标语言参数target_lang
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建Wav2Vec2ConformerModel对象
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        # 创建一个丢弃层
        self.dropout = nn.Dropout(config.final_dropout)

        # 存储目标语言参数
        self.target_lang = target_lang

        # 检查配置中是否定义了词汇表大小，如果没有则抛出错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ConformerForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        # 根据配置定义LM头部的全连接层
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.freeze_feature_encoder复制代码，并修改相关命名
    # 冻结特征编码器，禁用梯度计算，使其参数在训练期间不会更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward复制代码，并修改相关命名
    # 前向传播函数
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

        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)  # 对隐藏状态进行dropout处理

        logits = self.lm_head(hidden_states)  # 使用lm_head生成logits

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
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 如果不需要返回字典，则只返回logits和隐藏状态
            return ((loss,) + output) if loss is not None else output  # 如果存在loss，则将loss和output组合返回，否则返回output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions  # 返回结果作为CausalLMOutput实例
        )
@add_start_docstrings(
    """
    Wav2Vec2Conformer Model with a sequence classification head on top (a linear layer over the pooled output) for
    tasks like SUPERB Keyword Spotting.
    """,
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
class Wav2Vec2ConformerForSequenceClassification(Wav2Vec2ConformerPreTrainedModel):
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.__init__中复制而来，把Wav2Vec2->Wav2Vec2Conformer,wav2vec2->wav2vec2_conformer
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2Conformer adapters (config.add_adapter=True)"
            )
        # 创建Wav2Vec2ConformerModel对象
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        # 计算transformer层的数量+1（transformer层+输入嵌入）
        num_layers = config.num_hidden_layers + 1
        # 如果config中使用了加权层求和的方式
        if config.use_weighted_layer_sum:
            # 初始化层权重
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建一个线性层，输入维度为config.hidden_size，输出维度为config.classifier_proj_size
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建一个线性层，输入维度为config.classifier_proj_size，输出维度为config.num_labels
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.freeze_feature_encoder中复制而来，将wav2vec2->wav2vec2_conformer
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数，使其在训练过程中不更新
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    # 冻结基础模型的参数，使其在训练过程中不更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_conformer.parameters():
            param.requires_grad = False

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward中复制而来，把Wav2Vec2->Wav2Vec2Conformer,wav2vec2->wav2vec2_conformer,WAV_2_VEC_2->WAV2VEC2_CONFORMER
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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 检查是否需要返回字典形式的输出，如果未指定则根据配置决定是否使用字典形式返回
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果配置指定使用加权层求和，则设置输出隐藏状态为True
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 对输入进行wav2vec2 Conformer模型的前向传播
        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置指定使用加权层求和
        if self.config.use_weighted_layer_sum:
            # 获取隐藏状态
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 将隐藏状态堆叠起来形成新的张量
            hidden_states = torch.stack(hidden_states, dim=1)
            # 对层权重进行softmax归一化
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 使用归一化的权重对隐藏状态进行加权求和
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 如果不使用加权层求和，则直接使用第一个输出的隐藏状态
            hidden_states = outputs[0]

        # 通过投影层对隐藏状态进行投影
        hidden_states = self.projector(hidden_states)
        # 如果没有提供注意力掩码
        if attention_mask is None:
            # 对隐藏状态进行平均池化
            pooled_output = hidden_states.mean(dim=1)
        else:
            # 获取特征向量注意力掩码
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            # 将不相关位置的隐藏状态置零
            hidden_states[~padding_mask] = 0.0
            # 计算池化输出，同时考虑了注意力掩码
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 通过分类器获取logits
        logits = self.classifier(pooled_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 如果不返回字典形式的输出
        if not return_dict:
            # 将输出构建为元组
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典形式的输出
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 导入必要的库
@add_start_docstrings(
    """
    Wav2Vec2Conformer Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
# 定义一个新的类Wav2Vec2ConformerForAudioFrameClassification，继承自Wav2Vec2ConformerPreTrainedModel
class Wav2Vec2ConformerForAudioFrameClassification(Wav2Vec2ConformerPreTrainedModel):
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.__init__复制而来，将Wav2Vec2改为Wav2Vec2Conformer，wav2vec2改为wav2vec2_conformer，WAV_2_VEC_2改为WAV2VEC2_CONFORMER
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 检查是否支持使用Wav2Vec2Conformer适配器
        if hasattr(config, "add_adapter") and config.add_adapter:
            # 如果配置中设置了add_adapter为True，则抛出ValueError异常
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2Conformer adapters (config.add_adapter=True)"
            )
        # 创建Wav2Vec2Conformer模型
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        # 计算层数，包括transformer层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  
        # 如果配置中设置了use_weighted_layer_sum为True
        if config.use_weighted_layer_sum:
            # 初始化层权重参数为均匀分布
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建用于分类的线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 存储标签数量
        self.num_labels = config.num_labels

        # 初始化模型参数
        self.init_weights()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.freeze_feature_encoder复制而来，将wav2vec2改为wav2vec2_conformer
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数，使其在训练过程中不更新
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.freeze_base_model复制而来，将wav2vec2改为wav2vec2_conformer
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 冻结基础模型的参数，使其在训练过程中不更新，只有分类头会更新
        for param in self.wav2vec2_conformer.parameters():
            param.requires_grad = False

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.forward复制而来，将wav2vec2改为wav2vec2_conformer
    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未提供则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果使用加权层求和，则输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 将输入传递给wav2vec2_conformer模型
        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用加权层求和
        if self.config.use_weighted_layer_sum:
            # 从输出中获取隐藏状态
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 将隐藏状态堆叠起来
            hidden_states = torch.stack(hidden_states, dim=1)
            # 计算规范化权重
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 对隐藏状态进行加权求和
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用输出中的第一个元素作为隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态传递给分类器，得到logits
        logits = self.classifier(hidden_states)

        # 如果提供了标签
        loss = None
        if labels is not None:
            # 定义损失函数
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        # 如果不需要返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        # 否则返回TokenClassifierOutput对象，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# AMSoftmaxLoss 是一种用于多分类任务的损失函数
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        # 初始化 AMSoftmaxLoss 类
        super(AMSoftmaxLoss, self).__init__()
        # 设置 scale 和 margin 参数
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        # 创建一个可学习的权重矩阵
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        # 使用交叉熵损失作为基本损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        # 将标签展平为1维
        labels = labels.flatten()
        # 对权重矩阵进行归一化
        weight = nn.functional.normalize(self.weight, dim=0)
        # 对输入特征进行归一化
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        # 计算余弦相似度
        cos_theta = torch.mm(hidden_states, weight)
        # 根据 margin 调整余弦相似度
        psi = cos_theta - self.margin
        # 创建独热编码标签
        onehot = nn.functional.one_hot(labels, self.num_labels)
        # 根据标签将余弦相似度进行缩放
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        # 计算损失
        loss = self.loss(logits, labels)

        return loss


# TDNNLayer 是一个时间延迟神经网络层
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        # 初始化 TDNNLayer 类
        super().__init__()
        # 根据配置信息设置输入和输出通道数以及卷积核大小和膨胀率
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        # 创建线性层和激活函数
        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # 在通道维度上添加一个维度
        hidden_states = hidden_states.unsqueeze(1)
        # 使用 nn.functional.unfold 进行时间延迟操作
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        # 交换通道和时间维度
        hidden_states = hidden_states.transpose(1, 2)
        # 应用线性层
        hidden_states = self.kernel(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Wav2Vec2ConformerForXVector 是一个预训练模型
@add_start_docstrings(
    """
    Wav2Vec2Conformer Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    WAV2VEC2_CONFORMER_START_DOCSTRING,
)
class Wav2Vec2ConformerForXVector(Wav2Vec2ConformerPreTrainedModel):
    pass
    # 初始化函数，接收配置参数并调用父类初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 创建 Wav2Vec2ConformerModel 模型
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        # 计算总层数（transformer 层 + 输入嵌入层）
        num_layers = config.num_hidden_layers + 1
        # 如果配置了使用加权层求和，则创建可学习的层权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建线性投影层，用于将隐藏状态映射到 TDNN 层的输入维度
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 创建 TDNN 层列表
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 创建特征提取器，用于从 TDNN 层的输出提取特征
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        # 创建分类器，用于将特征映射到标签空间
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 创建 AMSoftmaxLoss 损失函数，用于计算特征向量和标签之间的距离
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 初始化权重
        self.init_weights()

    # 冻结特征编码器参数，使其在训练过程中不更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    # 冻结基础模型参数，使其在训练过程中不更新，只更新分类头部
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_conformer.parameters():
            param.requires_grad = False

    # 计算 TDNN 层输出长度
    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    # 将模型的前向传播方法进行装饰，添加文档字符串和代码示例
    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 覆盖父类的前向传播方法
    def forward(self,):
    # 定义模型前向传播过程
    def forward(
        # 输入张量
        self,
        input_values: Optional[torch.Tensor],
        # 注意力掩码张量，用于忽略填充部分
        attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出所有隐藏层输出
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典类型的输出
        return_dict: Optional[bool] = None,
        # 标签张量
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        
        outputs = self.wav2vec2_conformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Check if weighted layer sum is used, then compute the hidden states with weighted sum
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        # If not using weighted layer sum, use the first element of the outputs as hidden states
        else:
            hidden_states = outputs[0]

        # Project the hidden states
        hidden_states = self.projector(hidden_states)

        # Apply TDNN layers to the hidden states
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        if attention_mask is None:
            # If no attention mask is provided, compute mean and standard deviation of hidden states
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            # If attention mask is provided, compute mean and standard deviation based on output and TDNN lengths
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        # Compute output embeddings
        output_embeddings = self.feature_extractor(statistic_pooling)

        # Compute logits using the output embeddings
        logits = self.classifier(output_embeddings)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        # Return the appropriate output based on the return_dict parameter
        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
        else:
            return XVectorOutput(
                loss=loss,
                logits=logits,
                embeddings=output_embeddings,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
```