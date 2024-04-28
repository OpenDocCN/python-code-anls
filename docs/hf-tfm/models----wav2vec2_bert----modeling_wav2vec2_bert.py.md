# `.\transformers\models\wav2vec2_bert\modeling_wav2vec2_bert.py`

```
# 声明编码格式为 utf-8
# 版权声明
# 根据 Apache 2.0 版权协议的规定，在遵守协议的前提下，不得使用此文件
# 可以获得协议的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据本许可协议分发的软件
# 根据“原样”基础分发，没有任何担保或条件
# 样本授权规定特定语言的授权和限制
""" PyTorch Wav2Vec2-BERT model."""

# 导入所需的库
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
# 导入自定义的函数和类
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
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
    logging,
)
from .configuration_wav2vec2_bert import Wav2Vec2BertConfig

# 获取/logger/对象
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 2

# 通用文档字符串
_CONFIG_FOR_DOC = "Wav2Vec2BertConfig"

# 基础检查点的文档字符串
_BASE_CHECKPOINT_FOR_DOC = "facebook/w2v-bert-2.0"
# 预训练检查点的文档字符串
_PRETRAINED_CHECKPOINT_FOR_DOC = "hf-audio/wav2vec2-bert-CV16-en"
# 预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 146, 1024]

# CTC的文档字符串
_CTC_EXPECTED_OUTPUT = "'mr quilter is the apostle of the middle classes and we are glad to welcome his gospel'"
_CTC_EXPECTED_LOSS = 17.04

# 预训练模型的存档列表
WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/w2v-bert-2.0",
    # 查看所有 Wav2Vec2-BERT 模型：https://huggingface.co/models?filter=wav2vec2-bert
]

# 从transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2._compute_new_attention_mask复制过来的函数
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    计算一个注意力掩码，形式为`(batch, seq_len)`，其中每个批次元素的注意力停止在 `seq_lens` 中相应的元素处
    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            要掩码的序列，其中 `*` 是任意数量的特定于序列的维度，包括没有
        seq_lens (`torch.Tensor` of shape `(batch)`:
            每个元素表示 `hidden_states` 中相同索引处的序列的长度
    Returns:
        `torch.FloatTensor`: 形状为 `(batch, seq_len)` 的浮点注意力掩码
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]
    # 生成一个索引序列，范围是mask_seq_len，设备是seq_lens的设备，并扩展成batch_size行，-1列的张量
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    # 生成一个布尔值的掩码，判断indices中的值是否大于等于seq_lens的每个元素，形状是(batch_size, mask_seq_len)
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    # 使用hidden_states的维度创建一个全为1的掩码，形状是(batch_size, mask_seq_len)
    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    # 使用bool_mask将mask中对应位置的值置为0，形状不变
    mask = mask.masked_fill(bool_mask, 0)

    # 返回生成的掩码
    return mask
# 从transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices中复制而来的函数
def _compute_mask_indices(
    shape: Tuple[int, int],  # 输入形状的元组，包含批量大小和序列长度
    mask_prob: float,  # 将被掩盖的轴的百分比，取值范围为0到1
    mask_length: int,  # 掩码的长度
    attention_mask: Optional[torch.LongTensor] = None,  # 可选参数，右填充的注意力掩码，独立缩短每个批次维度的特征轴
    min_masks: int = 0,  # 最小掩码数量
) -> np.ndarray:
    """
    计算给定形状的随机掩码范围。用于实现[SpecAugment: A Simple Data Augmentation Method for ASR]。
    注意，此方法未经过优化，应在CPU上作为训练期间的预处理的一部分运行，而不是在TPU上运行。

    Args:
        shape: 要计算掩码的形状。应该是大小为2的元组，第一个元素是批量大小，第二个元素是要跨度的轴的长度。
        mask_prob: 将被掩盖的整个轴的百分比（介于0到1之间），其长度为`mask_length`的独立生成的掩码范围的数量由`mask_prob*shape[1]/mask_length`计算得到。
                   由于重叠的存在，`mask_prob`是一个上限，实际百分比将更小。
        mask_length: 掩码的长度
        min_masks: 最小的掩码数量
        attention_mask: 一个（右填充的）注意力掩码，它独立缩短每个批次维度的特征轴。
    """
    batch_size, sequence_length = shape  # 解包形状元组，获取批次大小和序列长度

    if mask_length < 1:  # 如果掩码长度小于1，则引发值错误
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:  # 如果掩码长度大于序列长度，则引发值错误
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """给定输入长度，计算应掩盖多少个范围"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)  # 计算应掩盖的范围数量
        num_masked_span = max(num_masked_span, min_masks)  # 确保掩盖的范围数量不小于最小掩码数

        # 确保掩盖范围数量不超过序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保掩盖范围数量不超过输入长度 - (掩码长度 - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批量中的掩码范围数
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()  # 如果attention_mask不为None，则计算其和，并转换为列表
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]  # 否则使用序列长度填充列表
    )

    # 用于填充SpecAugment掩码的掩码
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 创建与输入形状相同的全零数组
    spec_aug_mask_idxs = []  # 用于存储SpecAugment掩码的索引

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算掩盖范围的最大数量
    # 如果最大遮蔽跨度数为零，直接返回特定的增强遮蔽
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算该输入的遮蔽跨度数
        num_masked_span = compute_num_masked_span(input_length)

        # 获取随机索引以进行遮蔽
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个样本索引作为虚拟索引来填充向量，确保由于概率取整导致所有批次具有相同的维度
        # 选择第一个样本只是让这些向量填充两次。
        if len(spec_aug_mask_idx) == 0:
            # 只有在`input_length`严格小于`sequence_length`时才会出现这种情况
            # 最后一个标记必须是填充标记，我们可以将其用作虚拟遮蔽 ID
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟遮蔽索引连接到遮蔽索引中，并用虚拟索引填充
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将遮蔽索引转换为数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮蔽的索引扩展为遮蔽的跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 将起始索引添加偏移量，以便现在索引创建一个跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不会大于sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 散布索引进行遮蔽
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回特定的增强遮蔽
    return spec_aug_mask
# 从给定的特征向量中采样 num_negatives 个负样本向量
def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    # 获取批大小和序列长度
    batch_size, sequence_length = features_shape

    # 生成正向向量索引，并重复 num_negatives 次
    sequence_length_range = np.arange(sequence_length)

    # 从同一个样本中随机采样 num_negatives 个负样本向量的索引
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    # 如果提供了 mask_time_indices，则使用它来过滤有效的时间步索引
    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        # 获取当前批次中有效时间步的数量
        high = mask_time_indices[batch_idx].sum() - 1
        # 获取有效时间步的原始索引
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        # 生成随机的负样本索引，但避免采样到正向向量本身
        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        sampled_indices[sampled_indices >= feature_indices] += 1

        # 将随机索引映射回原始索引空间
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # 根据批大小进行索引偏移
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


# 实现 Wav2Vec2 Bert 模型的旋转式位置编码
class Wav2Vec2BertRotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embedding
    Reference: https://blog.eleuther.ai/rotary-embeddings/
    Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, config):
        super().__init__()
        # 计算旋转式位置编码的维度
        dim = config.hidden_size // config.num_attention_heads
        # 设置旋转式位置编码的基数
        base = config.rotary_embedding_base

        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # 注册逆频率参数
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None
    # 定义一个方法，用于计算前向传播时的旋转位置嵌入
    def forward(self, hidden_states):
        # 获取隐藏状态的序列长度
        sequence_length = hidden_states.shape[1]
        
        # 如果序列长度与缓存的序列长度相同且存在缓存的旋转位置嵌入，则直接返回缓存的旋转位置嵌入
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding
        
        # 缓存当前的序列长度
        self.cached_sequence_length = sequence_length
        
        # 以inv_freq常量的数据类型计算嵌入
        # 生成一个时间戳序列，数据类型与inv_freq相同
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        # 计算频率矩阵
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        # 将频率矩阵拼接成嵌入矩阵
        embeddings = torch.cat((freqs, freqs), dim=-1)
        
        # 计算嵌入矩阵的余弦值
        cos_embeddings = embeddings.cos()[:, None, None, :]
        # 计算嵌入矩阵的正弦值
        sin_embeddings = embeddings.sin()[:, None, None, :]
        
        # 将计算得到的嵌入值转换成和隐藏状态输入数据类型相同的数据类型
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        # 返回计算得到的旋转位置嵌入
        return self.cached_rotary_positional_embedding
# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRelPositionalEmbedding with Wav2Vec2Conformer->Wav2Vec2Bert
# 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRelPositionalEmbedding 复制并修改为 Wav2Vec2Conformer->Wav2Vec2Bert

class Wav2Vec2BertRelPositionalEmbedding(nn.Module):
    """Relative positional encoding module."""
    # 相对位置编码模块

    def __init__(self, config):
        super().__init__()
        # 初始化函数
        self.max_len = config.max_source_positions
        # 最大序列长度为配置文件中设置的源位置的最大长度
        self.d_model = config.hidden_size
        # 模型维度为配置文件中的隐藏层大小
        self.pe = None
        # 初始化位置编码为空
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))
        # 扩展位置编码

    def extend_pe(self, x):
        # 重新设定位置编码
        # 重置位置编码
        if self.pe is not None:
            # 如果位置编码不为空
            # self.pe包含正负两部分，self.pe的长度为2 * 输入长度 - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # 假设`i`是查询向量的位置，`j`是键向量的位置。当键位于左侧（i>j）时，我们使用正的相对位置，
        # 否则使用负的相对位置。
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

        # 反转正索引的顺序并连接正负索引。 这用于支持位移技巧
        # https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        # 前向传播
        self.extend_pe(hidden_states)
        start_idx = self.pe.size(1) // 2 - hidden_states.size(1) + 1
        end_idx = self.pe.size(1) // 2 + hidden_states.size(1)
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


class Wav2Vec2BertFeatureProjection(nn.Module):
    # Wav2Vec2BertFeatureProjection类
    def __init__(self, config):
        # 初始化函数
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        # 声明层归一化
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        # 声明线性变换
        self.dropout = nn.Dropout(config.feat_proj_dropout)
        # 添加丢弃层
    # 定义一个方法，用于前向传播计算
    def forward(self, hidden_states):
        # 对隐藏状态进行层标准化处理，用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 将标准化后的隐藏状态投影到指定维度
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态进行随机失活处理
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态和标准化后的隐藏状态
        return hidden_states, norm_hidden_states
class Wav2Vec2BertFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, hidden_size=None):
        super().__init__()
        # 如果未提供激活函数，则使用配置中的隐藏层激活函数
        act_fn = act_fn if act_fn is not None else config.hidden_act
        # 如果未提供隐藏层大小，则使用配置中的隐藏层大小
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        # 创建一个中间层的 Dropout 层
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 创建一个线性变换层，用于中间层的全连接
        self.intermediate_dense = nn.Linear(hidden_size, config.intermediate_size)
        # 根据提供的激活函数字符串或函数，转换为激活函数
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn

        # 创建一个输出层的全连接层
        self.output_dense = nn.Linear(config.intermediate_size, hidden_size)
        # 创建一个输出层的 Dropout 层
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward.forward 复制而来
    def forward(self, hidden_states):
        # 经过中间层全连接变换
        hidden_states = self.intermediate_dense(hidden_states)
        # 经过中间层激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 经过中间层的 Dropout
        hidden_states = self.intermediate_dropout(hidden_states)

        # 经过输出层的全连接变换
        hidden_states = self.output_dense(hidden_states)
        # 经过输出层的 Dropout
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2BertConvolutionModule(nn.Module):
    """Conformer 块中使用的卷积块"""

    def __init__(self, config):
        super().__init__()
        # 如果深度可分离卷积的内核大小减 1 为奇数，则抛出错误
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        # 创建一个 LayerNorm 层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个一维卷积层，执行 pointwise 卷积
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # 创建 GLU 激活函数
        self.glu = nn.GLU(dim=1)
        # 创建一个一维深度可分离卷积层
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )

        # 创建一个 LayerNorm 层
        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 根据配置中的隐藏层激活函数字符串或函数，创建激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 创建第二个一维 pointwise 卷积层
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # 创建卷积块的 Dropout 层
        self.dropout = nn.Dropout(config.conformer_conv_dropout)
    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.layer_norm(hidden_states) 
        # 对输入的hidden_states进行layer normalization，使其归一化
        
        # 如果attention_mask不为空
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            # 使用attention_mask对hidden_states进行填充，将attention_mask中为False的位置用0进行填充

        hidden_states = hidden_states.transpose(1, 2)
        # 将hidden_states的第1维和第2维交换位置，即将hidden_states的时间维度与特征维度交换

        hidden_states = self.pointwise_conv1(hidden_states)
        # 进行pointwise卷积操作

        hidden_states = self.glu(hidden_states)
        # 应用GLU机制，将输入hidden_states进行门控线性单元操作

        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))
        # 对hidden_states进行padding，用零填充，以防止因为因果卷积产生的边界问题

        hidden_states = self.depthwise_conv(hidden_states)
        # 进行一维深度卷积操作

        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        # 对hidden_states进行layer normalization，归一化深度卷积后的结果

        hidden_states = self.activation(hidden_states)
        # 使用激活函数对hidden_states进行激活操作

        hidden_states = self.pointwise_conv2(hidden_states)
        # 进行pointwise卷积操作

        hidden_states = self.dropout(hidden_states)
        # 对hidden_states进行dropout操作

        hidden_states = hidden_states.transpose(1, 2)
        # 将hidden_states的第1维和第2维交换位置

        return hidden_states
class Wav2Vec2BertSelfAttention(nn.Module):
    """Construct an Wav2Vec2BertSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """

    def __init__(self, config, is_adapter_attention=False):
        # 初始化方法，构造 Wav2Vec2BertSelfAttention 对象
        super().__init__()
        hidden_size = config.hidden_size if not is_adapter_attention else config.output_hidden_size

        self.head_size = hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.position_embeddings_type = config.position_embeddings_type if not is_adapter_attention else None

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=config.attention_dropout)

        if self.position_embeddings_type == "relative":
            # 如果位置嵌入类型是相对位置嵌入
            # 位置编码的线性变换
            self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=False)
            # 用于矩阵 c 和矩阵 d 的学习偏置，详见论文 https://arxiv.org/abs/1901.02860 第 3.3 节
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

        if self.position_embeddings_type == "relative_key":
            # 如果位置嵌入类型是相对键位置嵌入
            self.left_max_position_embeddings = config.left_max_position_embeddings
            self.right_max_position_embeddings = config.right_max_position_embeddings
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_rotary_embedding
    # 应用旋转嵌入到隐藏状态上
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        # 获取批次大小、序列长度和隐藏状态尺寸
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # 将隐藏状态重塑为(批次大小, 序列长度, 注意力头数, 每个头的大小)
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)
    
        # 从相对位置嵌入中获取余弦和正弦
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]
    
        # 旋转隐藏状态
        hidden_states = hidden_states.transpose(0, 1)  # 将序列维度移到前面
        rotated_states_begin = hidden_states[..., : self.head_size // 2]  # 获取前半部分
        rotated_states_end = hidden_states[..., self.head_size // 2 :]  # 获取后半部分
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)  # 连接并翻转后半部分
        hidden_states = (hidden_states * cos) + (rotated_states * sin)  # 将旋转状态与余弦和正弦结合
        hidden_states = hidden_states.transpose(0, 1)  # 将序列维度移回去
    
        # 将隐藏状态重塑为(批次大小, 序列长度, 所有头的大小)
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)
    
        return hidden_states
    
    # 从transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention复制的_apply_relative_embeddings方法
    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        # 1. project positional embeddings
        # 将相对位置编码进行投影
        # => (batch, head, 2*time1-1, d_k)
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

        # 2. Add bias to query
        # 为查询添加偏置
        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        # 3. attention score: first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # 计算注意力分数：首先计算矩阵 a 和矩阵 c
        # 如 https://arxiv.org/abs/1901.02860 第 3.3 节所述
        # => (batch, head, time1, time2)
        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        # 4. then compute matrix b and matrix d
        # 然后计算矩阵 b 和矩阵 d
        # => (batch, head, time1, 2*time1-1)
        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # 5. shift matrix b and matrix d
        # 移动矩阵 b 和矩阵 d
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
# 定义一个 Wav2Vec2BertEncoderLayer 类，基于论文 https://arxiv.org/abs/2005.08100 实现
class Wav2Vec2BertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size  # 获取配置中的隐藏层大小
        dropout = config.attention_dropout  # 获取配置中的注意力丢弃率

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 初始化第一个前馈层的 LayerNorm
        self.ffn1 = Wav2Vec2BertFeedForward(config)  # 初始化第一个前馈层

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 初始化自注意力层的 LayerNorm
        self.self_attn_dropout = nn.Dropout(dropout)  # 初始化自注意力层的丢弃层
        self.self_attn = Wav2Vec2BertSelfAttention(config)  # 初始化自注意力层

        # Conformer Convolution
        self.conv_module = Wav2Vec2BertConvolutionModule(config)  # 初始化 Conformer 卷积模块

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 初始化第二个前馈层的 LayerNorm
        self.ffn2 = Wav2Vec2BertFeedForward(config)  # 初始化第二个前馈层
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 初始化最终的 LayerNorm

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states  # 设置 hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states  # 记录残差连接
        hidden_states = self.ffn1_layer_norm(hidden_states)  # LayerNorm
        hidden_states = self.ffn1(hidden_states)  # Feed-forward 1
        hidden_states = hidden_states * 0.5 + residual  # 残差连接
        residual = hidden_states  # 更新残差连接

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)  # LayerNorm
        hidden_states, attn_weigts = self.self_attn(  # Self-Attention
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)  # 自注意力层的丢弃
        hidden_states = hidden_states + residual  # 残差连接

        # 3. Convolutional Layer
        residual = hidden_states  # 记录残差连接
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)  # Conformer 卷积模块
        hidden_states = residual + hidden_states  # 残差连接

        # 4. Feed-Forward 2 Layer
        residual = hidden_states  # 记录残差连接
        hidden_states = self.ffn2_layer_norm(hidden_states)  # LayerNorm
        hidden_states = self.ffn2(hidden_states)  # Feed-forward 2
        hidden_states = hidden_states * 0.5 + residual  # 残差连接
        hidden_states = self.final_layer_norm(hidden_states)  # 最终的 LayerNorm

        return hidden_states, attn_weigts  # 返回 hidden_states 和自注意力权重
    # 初始化方法，接受配置参数并调用父类初始化方法
    def __init__(self, config):
        super().__init__()
        # 保存配置参数
        self.config = config

        # 根据配置参数中的位置嵌入类型进行不同的初始化
        if config.position_embeddings_type == "relative":
            # 如果是相对位置嵌入，使用Wav2Vec2BertRelPositionalEmbedding类进行初始化
            self.embed_positions = Wav2Vec2BertRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            # 如果是旋转位置嵌入，使用Wav2Vec2BertRotaryPositionalEmbedding类进行初始化
            self.embed_positions = Wav2Vec2BertRotaryPositionalEmbedding(config)
        else:
            # 如果没有指定位置嵌入类型，设置为None
            self.embed_positions = None

        # 初始化丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化编码器层列表，根据config.num_hidden_layers重复创建Wav2Vec2BertEncoderLayer
        self.layers = nn.ModuleList([Wav2Vec2BertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 渐变检查点为假
        self.gradient_checkpointing = False

    # 前向传播方法，接受隐藏状态等参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
class Wav2Vec2BertAdapter(nn.Module):
    # 初始化函数，用于定义模型结构
    def __init__(self, config):
        super().__init__()
        # 如果输出隐藏层大小与隐藏层大小不同，则需要进行投影
        if config.output_hidden_size != config.hidden_size:
            # 线性投影层，用于将隐藏状态维度投影到指定大小
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # 投影后的层归一化，用于规范化投影后的隐藏状态
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size, eps=config.layer_norm_eps)
        else:
            self.proj = self.proj_layer_norm = None
        # 适配器层列表，用于堆叠适配器层
        self.layers = nn.ModuleList(Wav2Vec2BertAdapterLayer(config) for _ in range(config.num_adapter_layers))
        # LayerDrop概率，用于控制层级丢弃的概率
        self.layerdrop = config.layerdrop

        # 适配器卷积核大小
        self.kernel_size = config.adapter_kernel_size
        # 适配器卷积步长
        self.stride = config.adapter_stride

    # 根据注意力掩码计算子采样长度
    def _compute_sub_sample_lengths_from_attention_mask(self, seq_lens):
        if seq_lens is None:
            return seq_lens
        # 计算填充长度
        pad = self.kernel_size // 2
        # 计算子采样长度
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1
        return seq_lens.floor()

    # 前向传播函数
    def forward(self, hidden_states, attention_mask=None):
        # 如果需要，进行隐藏状态的投影
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 计算子采样长度
        sub_sampled_lengths = None
        if attention_mask is not None:
            # 计算子采样长度
            sub_sampled_lengths = (attention_mask.size(1) - (1 - attention_mask.int()).sum(1)).to(hidden_states.device)

        # 循环遍历适配器层
        for layer in self.layers:
            # 随机生成LayerDrop概率
            layerdrop_prob = torch.rand([])
            # 根据注意力掩码计算子采样长度
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(sub_sampled_lengths)
            # 如果不是训练阶段或通过LayerDrop概率
            if not self.training or (layerdrop_prob > self.layerdrop):
                # 应用适配器层
                hidden_states = layer(
                    hidden_states, attention_mask=attention_mask, sub_sampled_lengths=sub_sampled_lengths
                )

        return hidden_states


class Wav2Vec2BertAdapterLayer(nn.Module):
    # Wav2Vec2BertAdapterLayer类定义
    # 初始化模型，传入配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 获取嵌入维度和dropout参数
        embed_dim = config.output_hidden_size
        dropout = config.conformer_conv_dropout

        # 从配置参数获取适配器的卷积核大小和步幅
        self.kernel_size = config.adapter_kernel_size
        self.stride = config.adapter_stride

        # 1. 残差卷积层
        # 层归一化
        self.residual_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 1维卷积层
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.activation = nn.GLU(dim=1)

        # 自注意力机制
        # 层归一化
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 1维卷积层
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 自注意力层
        self.self_attn = Wav2Vec2BertSelfAttention(config, is_adapter_attention=True)
        # dropout层
        self.self_attn_dropout = nn.Dropout(dropout)

        # 前馈神经网络
        # 层归一化
        self.ffn_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 前馈神经网络层
        self.ffn = Wav2Vec2BertFeedForward(config, act_fn=config.adapter_act, hidden_size=embed_dim)

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        sub_sampled_lengths: Optional[torch.Tensor] = None,
```.
    ):
        # 对残差进行 Layer Normalization
        residual = self.residual_layer_norm(hidden_states)

        # 对残差进行池化，以匹配多头注意力输出的序列长度
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        # 对自注意力输出进行 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 在输入多头注意力层之前进行池化
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        # 如果存在注意力掩码
        if attention_mask is not None:
            # 计算新的注意力掩码
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            # 准备 4 维注意力掩码
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # 其余计算与普通 Transformer 编码器层相同
        # 传入自注意力，返回自注意力的权重
        hidden_states, attn_weigths = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        # 添加残差连接
        hidden_states = hidden_states + residual

        # 更新残差
        residual = hidden_states

        # 对输出进行 Layer Normalization
        hidden_states = self.ffn_layer_norm(hidden_states)
        # 经过前馈神经网络，并添加残差连接
        hidden_states = self.ffn(hidden_states) + residual

        # 返回编码器层的输出
        return hidden_states
# 从transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerPreTrainedModel复制而来，将Wav2Vec2Conformer->Wav2Vec2Bert，wav2vec2_conformer->wav2vec2_bert，input_values->input_features
class Wav2Vec2BertPreTrainedModel(PreTrainedModel):
    """
    用于处理权重初始化和一个简单接口用于下载和加载预训练模型的抽象类。
    """

    config_class = Wav2Vec2BertConfig
    base_model_prefix = "wav2vec2_bert"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True

    # 忽略复制
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, Wav2Vec2BertSelfAttention):
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, Wav2Vec2BertFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # 忽略复制
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        计算卷积层的输出长度
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride, padding):
            # 1D卷积层输出长度公式来自 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
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
        # 计算非填充部分的长度，对attention_mask在最后一个维度上进行累加
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 获取特征提取的输出长度，根据是否添加适配器进行适配
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        # 获取 batch 大小
        batch_size = attention_mask.shape[0]

        # 生成与特征向量长度相同的全零注意力掩码
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        
        # 确保输出长度之前的所有值都被关注
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        
        # 翻转 attention_mask，对最后一个维度进行累加，再次翻转，并转换为 bool 类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        
        # 返回处理后的注意力掩码
        return attention_mask
# WAV2VEC2_BERT_START_DOCSTRING 是模型说明文档的字符串常量
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

# WAV2VEC2_BERT_INPUTS_DOCSTRING 是输入参数的说明文档的字符串常量
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

# 为 Wav2Vec2BertModel 类添加注释
@add_start_docstrings(
    "The bare Wav2Vec2Bert Model transformer outputting raw hidden-states without any specific head on top.",
    WAV2VEC2_BERT_START_DOCSTRING,
)
class Wav2Vec2BertModel(Wav2Vec2BertPreTrainedModel):
    # 初始化方法，接受一个Wav2Vec2BertConfig类型的参数
    def __init__(self, config: Wav2Vec2BertConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置参数
        self.config = config
        # 创建Wav2Vec2BertFeatureProjection对象
        self.feature_projection = Wav2Vec2BertFeatureProjection(config)

        # 如果mask_prob大于0.0，模型需要掩盖向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            # 创建一个随机初始化的可学习参数
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 创建Wav2Vec2BertEncoder对象
        self.encoder = Wav2Vec2BertEncoder(config)

        # 如果配置中包含adapter，则创建Wav2Vec2BertAdapter对象，否则为None
        self.adapter = Wav2Vec2BertAdapter(config) if config.add_adapter else None

        # 如果使用adapter之前需要中间FFN，则创建Wav2Vec2BertFeedForward对象
        self.intermediate_ffn = None
        if config.use_intermediate_ffn_before_adapter:
            self.intermediate_ffn = Wav2Vec2BertFeedForward(config, act_fn="relu")

        # 初始化权重并进行最终处理
        self.post_init()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states中复制的方法
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        如果配置中设置了 apply_spec_augment 为 False，则不进行掩码处理，直接返回隐藏状态
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        获取隐藏状态的 batch_size, sequence_length, 和 hidden_size
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            如果给定了 mask_time_indices，则根据其值对时间轴进行掩码处理
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # compute mask_time_indices if not provided and apply SpecAugment along time axis
            如果未提供 mask_time_indices 并且处于训练状态，则根据配置生成 mask_time_indices 并对时间轴进行掩码处理
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
            如果配置中设置 mask_feature_prob 大于 0 并且处于训练状态，则生成 mask_feature_indices 并对特征轴进行掩码处理
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
    定义 forward 函数，并添加模型相关的文档字符串和代码示例相关的文档字符串
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数，接受输入并返回结果
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 如果输出注意力为空，则使用配置中的输出注意力
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态为空，则使用配置中的输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典为空，则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 根据输入特征和提取特征，获取隐藏状态和提取特征
        hidden_states, extract_features = self.feature_projection(input_features)
        # 使用掩码隐藏状态
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 使用编码器获取编码器输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果存在中间前馈神经网络，则对隐藏状态进行处理
        if self.intermediate_ffn:
            expanded_hidden_states = self.intermediate_ffn(hidden_states)
            hidden_states = hidden_states + 0.5 * expanded_hidden_states

        # 如果存在适配器，则对隐藏状态进行处理
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        # 如果不返回字典，则返回元组和编码器输出的其他部分
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回基础模型输出
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器添加模型的起始文档字符串，并引用预定义的文档字符串
class Wav2Vec2BertForCTC(Wav2Vec2BertPreTrainedModel):
    # 根据传入的配置和目标语言，初始化 Wav2Vec2BertForCTC 模型
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 Wav2Vec2BertModel 模型实例
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        # 添加一个 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言
        self.target_lang = target_lang

        # 如果配置中未定义词汇量的大小，则抛出错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2BertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        # 根据需要的输出隐藏大小来初始化 LM head 线性层
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加模型的前向传播文档字符串，并引用预定义的文档字符串和代码示例
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
````
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 检查是否应该返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用wav2vec2_bert模型，传入输入特征，注意力掩码等参数，获取输出
        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态应用dropout
        hidden_states = self.dropout(hidden_states)

        # 使用语言模型头部预测下一个标记的对数概率
        logits = self.lm_head(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果存在标签
        if labels is not None:
            # 检查标签是否超出了词汇表大小
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 从注意力掩码中检索损失的输入长度
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(input_features.shape[:2], device=input_features.device, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum([-1])).to(torch.long)

            # 假设填充标记用-100填充，未被关注到时
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # CTC损失不支持fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 禁用cudnn加速以确保稳定的CTC损失计算
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

        # 如果不返回字典形式的输出
        if not return_dict:
            # 组合输出结果
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            # 返回损失和输出结果的元组，如果损失不为None
            return ((loss,) + output) if loss is not None else output

        # 返回字典形式的输出结果
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 此类是基于 Wav2Vec2Bert 模型的序列分类模型
@add_start_docstrings(
    """
    Wav2Vec2Bert Model with a sequence classification head on top (a linear layer over the pooled output) for
    tasks like SUPERB Keyword Spotting.
    """,
    WAV2VEC2_BERT_START_DOCSTRING,
)
class Wav2Vec2BertForSequenceClassification(Wav2Vec2BertPreTrainedModel):
    # 从父类复制初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 如果配置要求添加 adapter，则抛出错误，因为序列分类不支持使用 Wav2Vec2Bert 的 adapter
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2Bert adapters (config.add_adapter=True)"
            )
        # 初始化 Wav2Vec2Bert 模型
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        # 计算 transformer 层数加上输入嵌入的总层数
        num_layers = config.num_hidden_layers + 1
        # 如果使用加权层求和，则创建权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建一个投影层，将隐藏层大小映射到分类器输入大小
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建分类器线性层
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结基础模型，使其参数不会在训练过程中更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    # 模型前向传播
    @add_start_docstrings_to_model_forward(WAV2VEC2_BERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_BASE_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从父类复制前向传播方法
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 根据 return_dict 是否为 None 来确定是否返回一个字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果使用加权层求和，则将输出的 hidden_states 置为 True
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 将输入特征和其他参数传递给 wav2vec2_bert 模型进行前向推理
        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 根据是否使用加权层求和来决定下一步处理
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # 将 hidden_states 投影到新的维度上
        hidden_states = self.projector(hidden_states)
        # 根据是否存在 attention_mask 进行不同的操作
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 通过分类器获得预测值
        logits = self.classifier(pooled_output)

        # 如果存在 labels，则计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 根据 return_dict 来返回不同的结果
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回结果
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 这个类是 Wav2Vec2Bert 模型的一个子类，用于音频帧分类任务
@add_start_docstrings(
    """
    Wav2Vec2Bert Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    WAV2VEC2_BERT_START_DOCSTRING,
)
class Wav2Vec2BertForAudioFrameClassification(Wav2Vec2BertPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中设置了使用 adapter，则报错，因为音频帧分类不支持使用 Wav2Vec2Bert adapter
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2Bert adapters (config.add_adapter=True)"
            )
        # 创建 Wav2Vec2Bert 模型
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        # 计算需要的权重层数
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        # 如果使用加权层求和，则创建权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建分类器层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        self.init_weights()

    # 冻结 Wav2Vec2Bert 模型的参数，使其在训练中不更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV2VEC2_BERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_BASE_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 前向传播函数
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 若return_dict非空则使用，否则使用self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        # 若self.config.use_weighted_layer_sum为真则output_hidden_states为真

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取模型的输出结果

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            # 使用加权求和计算隐藏状态
        else:
            hidden_states = outputs[0]
            # 获取输出的第一个元素作为隐藏状态

        logits = self.classifier(hidden_states)
        # 通过分类器得到logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))
            # 如果labels不为空，则计算损失函数

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output
            # 如果return_dict为空，则返回logits和outputs中除第一个元素外的所有元素

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回TokenClassifierOutput对象，包含损失值、logits、hidden_states和attentions
# 从 transformers.models.wav2vec2.modeling_wav2vec2 模块中复制的 AMSoftmaxLoss 类
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        # 初始化父类 nn.Module
        super(AMSoftmaxLoss, self).__init__()
        # 设置放大因子和边界值
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        # 创建待训练的权重参数
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        # 创建交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        # 将标签展平为一维
        labels = labels.flatten()
        # 对权重进行 L2 归一化
        weight = nn.functional.normalize(self.weight, dim=0)
        # 对输入特征进行 L2 归一化
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        # 计算余弦相似度
        cos_theta = torch.mm(hidden_states, weight)
        # 减去边界值得到 psi
        psi = cos_theta - self.margin

        # 创建独热编码标签
        onehot = nn.functional.one_hot(labels, self.num_labels)
        # 根据独热编码计算加权后的 logits
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        # 计算交叉熵损失
        loss = self.loss(logits, labels)

        return loss


# 从 transformers.models.wav2vec2.modeling_wav2vec2 模块中复制的 TDNNLayer 类
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        # 初始化父类 nn.Module
        super().__init__()
        # 根据配置信息初始化输入、输出通道数和卷积核大小
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        # 创建线性层和激活函数
        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # 增加一个维度以满足卷积输入格式
        hidden_states = hidden_states.unsqueeze(1)
        # 使用 unfold 操作进行时间维度的卷积
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        # 将通道维度移到第二维
        hidden_states = hidden_states.transpose(1, 2)
        # 使用线性层进行特征转换
        hidden_states = self.kernel(hidden_states)

        # 应用 ReLU 激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer 模块中复制的 Wav2Vec2BertForXVector 类
@add_start_docstrings(
    """
    Wav2Vec2Bert Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    WAV2VEC2_BERT_START_DOCSTRING,
)
class Wav2Vec2BertForXVector(Wav2Vec2BertPreTrainedModel):
    pass
    # 初始化函数，接受配置参数并进行初始化
    def __init__(self, config):
        # 调用父类的初始化函数，传入配置参数
        super().__init__(config)

        # 创建 Wav2Vec2BertModel 模型
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        # 计算层数：transformer 层 + 输入嵌入层
        num_layers = config.num_hidden_layers + 1
        # 如果配置中使用加权层求和
        if config.use_weighted_layer_sum:
            # 创建可训练的层权重参数，初始化为均匀分布
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建线性投影层，将隐藏状态投影到指定维度
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 创建 TDNN 模块列表
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 创建特征提取器，将 TDNN 输出拼接并投影到指定维度
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        # 创建分类器，将特征向量映射到标签空间
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 创建 AMSoftmax 损失函数，用于训练时的监督信号
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 初始化模型参数
        self.init_weights()

    # 冻结基础模型的参数，仅允许分类头的参数进行梯度更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历基础模型的参数，并将其梯度计算设置为 False
        for param in self.wav2vec2_bert.parameters():
            param.requires_grad = False

    # 计算 TDNN 层的输出长度
    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        # 计算一维卷积层的输出长度
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层的输出长度计算公式，来源于 PyTorch 文档
            return (input_length - kernel_size) // stride + 1

        # 遍历 TDNN 层的卷积核大小，并更新输入长度
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        # 返回计算后的输出长度
        return input_lengths

    # 模型的前向传播函数
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
        # 定义函数签名以及注释说明

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置返回字典为传入值或者根据配置设定的返回字典
        
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        # 如果配置中使用加权层求和则设置输出的隐藏状态为True，否则为output_hidden_states
        
        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用wav2vec2_bert模型进行前向传播，得到输出
        
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 如果配置中使用加权层求和，则设置隐藏状态为输出的指定位置的值
            hidden_states = torch.stack(hidden_states, dim=1)
            # 将隐藏状态进行堆叠
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 使用softmax函数计算归一化的权重
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            # 计算加权层求和后的隐藏状态
        else:
            hidden_states = outputs[0]
            # 否则将隐藏状态设置为输出的第一个元素

        hidden_states = self.projector(hidden_states)
        # 对隐藏状态进行投影
        
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)
            # 对每个时间延迟神经网络层进行操作

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
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
        # 对隐藏状态进行统计池化得到 mean_features 和 std_features

        output_embeddings = self.feature_extractor(statistic_pooling)
        # 使用特征提取器对统计池化后的值进行处理
        logits = self.classifier(output_embeddings)
        # 使用分类器对处理后的值进行分类

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)
            # 如果存在标签，则计算损失

        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
            # 如果不返回字典，返回指定的元组
        
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回XVectorOutput实例
```