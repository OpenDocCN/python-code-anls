# `.\models\data2vec\modeling_data2vec_audio.py`

```
# 设置文件编码为UTF-8
# 版权声明
# 根据Apache许可2.0版进行许可
# 除非法律要求或书面同意，否则不得使用此文件
# 您可以在以下网址获得许可证的副本
#  http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件按"原样"分发
# 没有任何种类的明示或暗示的保证或条件
# 有关特定语言的特定语言约束和限制，请参阅许可证
""" PyTorch Data2VecAudio model."""

# 导入必要的库
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
# 导入外部库

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
from .configuration_data2vec_audio import Data2VecAudioConfig
# 导入所需的模块

logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 2

# 一般文档字符串
_CONFIG_FOR_DOC = "Data2VecAudioConfig"
# 基本文档字符串
_CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC文档字符串
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 66.95

# Data2VecAudio的预训练模型存档列表
DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-audio-base",
    "facebook/data2vec-audio-base-10m",
    "facebook/data2vec-audio-base-100h",
    "facebook/data2vec-audio-base-960h",
    # 查看所有Data2VecAudio模型，访问https://huggingface.co/models?filter=data2vec-audio
]

# 从transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices中复制
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算给定形状的随机蒙版间隔。用于实现[SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)。
    请注意，此方法未经优化以在TPU上运行，并应在训练期间的预处理过程中在CPU上运行。
    Args:
        shape: 用于计算掩码的形状。应为大小为 2 的元组，其中第一个元素是批量大小，第二个元素是要跨越的轴的长度。
        mask_prob: 整个轴的百分比（介于 0 和 1 之间），将被掩盖。掩码长度为 `mask_length` 的独立生成的掩码跨度数量由 `mask_prob*shape[1]/mask_length` 计算。由于重叠，`mask_prob` 是一个上限，实际百分比将更小。
        mask_length: 掩码的大小
        min_masks: 掩盖的最小跨度数
        attention_mask: （右填充）注意力掩码，独立缩短每个批次维度的特征轴。
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` 必须大于 0。")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` 必须小于 `sequence_length`，但得到了 `mask_length`: {mask_length}"
            f" 和 `sequence_length`: {sequence_length}`"
        )

    # epsilon 用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """给定输入长度，计算应掩盖多少个跨度"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保 num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保 num_masked span 也 <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批次中掩码跨度的数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask
    for input_length in input_lengths:
        # 遍历输入长度列表

        # 计算当前输入的遮罩段数
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要遮罩的索引
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个作为填充向量的虚拟索引，确保所有批次的维度相同
        # 由于概率舍入的原因
        # 选择第一个样本只是为了两次填充那些向量
        if len(spec_aug_mask_idx) == 0:
            # 只有在`input_length`严格小于`sequence_length`时才会出现这种情况
            # 此时最后一个标记必须是填充标记，我们可以使用作为虚拟掩码id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 扩展遮罩索引以形成遮罩段
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 给起始索引添加偏移量，以确保索引现在构成一个段
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保我们不能有大于sequence_length的索引
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将索引分散到遮罩
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
class Data2VecAudioConvLayer(nn.Module):
    # Data2VecAudioConvLayer类的构造函数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果layer_id大于0，则in_conv_dim为config.conv_dim[layer_id - 1]，否则为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # out_conv_dim为config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建1维卷积层，参数为输入维度、输出维度、卷积核大小、步长和是否包含偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建LayerNorm层，参数为输出维度和是否应用可学习的缩放和偏置
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 获取激活函数的引用
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数
    def forward(self, hidden_states):
        # 将输入通过卷积层
        hidden_states = self.conv(hidden_states)

        # 交换维度，从[B, C, L]变为[B, L, C]
        hidden_states = hidden_states.transpose(-2, -1)
        # 对交换后的维度进行LayerNorm
        hidden_states = self.layer_norm(hidden_states)
        # 再次交换维度，恢复到原来的维度顺序[B, C, L]
        hidden_states = hidden_states.transpose(-2, -1)

        # 对输出进行激活函数处理
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers库中拷贝的代码，稍作修改
class Data2VecAudioPadLayer(nn.Module):
    # Data2VecAudioPadLayer类的构造函数
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 如果num_conv_pos_embeddings是偶数，num_pad_remove为1，否则为0
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    # 前向传播函数
    def forward(self, hidden_states):
        # 如果num_pad_remove大于0，则截取掉最后self.num_pad_remove个位置
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class Data2VecAudioPositionalConvLayer(nn.Module):
    # Data2VecAudioPositionalConvLayer类的构造函数
    def __init__(self, config):
        super().__init__()
        # 创建1维卷积层，参数为输入维度、输出维度、卷积核大小、填充大小和分组数
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_pos_kernel_size,
            padding=config.conv_pos_kernel_size // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 创建Data2VecAudioPadLayer实例
        self.padding = Data2VecAudioPadLayer(config.conv_pos_kernel_size)
        # 获取激活函数的引用
        self.activation = ACT2FN[config.feat_extract_activation]
        # 创建LayerNorm层，参数为隐藏状态维度和是否应用可学习的缩放和偏置
        self.layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

    # 前向传播函数
    def forward(self, hidden_states):
        # 将隐藏状态通过卷积层
        hidden_states = self.conv(hidden_states)
        # 对卷积结果进行填充
        hidden_states = self.padding(hidden_states)

        # 交换维度，从[B, C, L]变为[B, L, C]
        hidden_states = hidden_states.transpose(1, 2)
        # 对交换后的维度进行LayerNorm
        hidden_states = self.layer_norm(hidden_states)
        # 再次交换维度，恢复到原来的维度顺序[B, C, L]
        hidden_states = hidden_states.transpose(1, 2)
        # 对输出进行激活函数处理
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Data2VecAudioPositionalConvEmbedding(nn.Module):
    # Data2VecAudioPositionalConvEmbedding类的构造函数
    def __init__(self, config):
        super().__init__()
        # 创建多个Data2VecAudioPositionalConvLayer层组成的模块列表
        self.layers = nn.ModuleList(
            [Data2VecAudioPositionalConvLayer(config) for _ in range(config.num_conv_pos_embeddings)]
        )
    # 定义一个前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 转置隐藏状态张量，将第1和第2个维度交换位置
        hidden_states = hidden_states.transpose(1, 2)
        # 遍历每个层，并对隐藏状态进行处理
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # 再次转置隐藏状态张量，将第1和第2个维度交换回原始位置
        hidden_states = hidden_states.transpose(1, 2)
        # 返回处理后的隐藏状态
        return hidden_states
class Data2VecAudioFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [Data2VecAudioConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        )
        self.gradient_checkpointing = False  # 设置默认为不使用梯度检查点
        self._requires_grad = True  # 默认需要计算梯度

    # 冻结模型参数
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False  # 更新需要计算梯度的标志

    # 前向传播
    def forward(self, input_values):
        hidden_states = input_values[:, None]

        if self._requires_grad and self.training:  # 如果需要计算梯度且处于训练状态
            hidden_states.requires_grad = True  # 设置隐藏状态需要计算梯度

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:  # 如果需要计算梯度且启用梯度检查点且处于训练状态
                hidden_states = self._gradient_checkpointing_func(  # 使用梯度检查点函数计算隐藏状态
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)  # 计算隐藏状态经过卷积层后的结果

        return hidden_states


class Data2VecAudioFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)  # 初始化LayerNorm层
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)  # 初始化线性投影层
        self.dropout = nn.Dropout(config.feat_proj_dropout)  # 初始化丢弃层

    def forward(self, hidden_states):
        norm_hidden_states = self.layer_norm(hidden_states)  # 对隐藏状态进行LayerNorm
        hidden_states = self.projection(norm_hidden_states)  # 进行线性投影
        hidden_states = self.dropout(hidden_states)  # 进行丢弃
        return hidden_states, norm_hidden_states  # 返回投影后的隐藏状态和未投影的隐藏状态


class Data2VecAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Data2VecAudioConfig] = None,
        # 初始化模型参数
        super().__init__()
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置dropout概率
        self.dropout = dropout
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads
        # 设置模型配置
        self.config = config

        # 检查嵌入维度是否可以被注意力头数量整除
        if (self.head_dim * num_heads) != self.embed_dim:
            # 抛出数值错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 设置缩放系数
        self.scaling = self.head_dim**-0.5
        # 设置模型是否为解码器
        self.is_decoder = is_decoder
        # 设置模型是否为可导致的
        self.is_causal = is_causal

        # 初始化键、值、查询和输出的线性变换
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 重塑张量形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 正向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# 定义一个新的神经网络模块，用于 Data2VecAudio 模型的前馈网络部分
class Data2VecAudioFeedForward(nn.Module):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__()
        # 初始化中间层的 Dropout 层
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 定义中间层的全连接层，将输入维度转换为中间维度
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 定义输出层的全连接层，将中间维度转换为隐藏维度
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化输出层的 Dropout 层
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播函数
    def forward(self, hidden_states):
        # 中间层的全连接运算
        hidden_states = self.intermediate_dense(hidden_states)
        # 中间层的激活函数运算
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 中间层的 Dropout 操作
        hidden_states = self.intermediate_dropout(hidden_states)

        # 输出层的全连接运算
        hidden_states = self.output_dense(hidden_states)
        # 输出层的 Dropout 操作
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# 定义一个新的神经网络模块，用于 Data2VecAudio 模型的编码器层
class Data2VecAudioEncoderLayer(nn.Module):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__()
        # 定义注意力机制
        self.attention = Data2VecAudioAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化 Layer Normalization 层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义前馈网络层
        self.feed_forward = Data2VecAudioFeedForward(config)
        # 初始化最终 Layer Normalization 层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 保存注意力机制之前的隐藏状态，以便后续计算残差连接
        attn_residual = hidden_states
        # 注意力机制的前向传播
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 计算残差连接
        hidden_states = attn_residual + hidden_states

        # Layer Normalization 操作
        hidden_states = self.layer_norm(hidden_states)
        # 前馈网络的前向传播
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终的 Layer Normalization 操作
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果需要输出注意力权重，则将其包含在输出中
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 定义一个新的神经网络模块，用于 Data2VecAudio 模型的编码器
class Data2VecAudioEncoder(nn.Module):
    # 定义一个类，该类作为Data2VecAudioEncoder的子类，表示一个语音编码器
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 存储配置信息
        self.config = config
        # 创建一个Data2VecAudioPositionalConvEmbedding对象，用于对位置信息进行卷积嵌入
        self.pos_conv_embed = Data2VecAudioPositionalConvEmbedding(config)
        # 创建一个LayerNorm对象，用于层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout对象，用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建一个ModuleList对象，该对象包含多个Data2VecAudioEncoderLayer对象，用于构建语音编码器的多层
        self.layers = nn.ModuleList([Data2VecAudioEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点，默认值为False
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # TODO: 添加此处forward方法的注释
    # 如果输出所有隐藏状态，则初始化一个空元组，否则初始化为None
    all_hidden_states = () if output_hidden_states else None
    
    # 如果输出所有自注意力，则初始化一个空元组，否则初始化为None
    all_self_attentions = () if output_attentions else None
    
    # 如果有注意力掩码
    if attention_mask is not None:
        # 确保填充的标记输出为0
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        hidden_states[~expand_attention_mask] = 0
        
        # 扩展注意力掩码的形状
        attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        )

    # 使用位置卷积嵌入层对隐藏状态进行位置编码
    position_embeddings = self.pos_conv_embed(hidden_states)
    
    # 将位置编码加到隐藏状态上
    hidden_states = hidden_states + position_embeddings
    
    # 对隐藏状态进行层归一化
    hidden_states = self.layer_norm(hidden_states)
    
    # 对隐藏状态进行Dropout
    hidden_states = self.dropout(hidden_states)

    # 检查是否启用了deepspeed_zero3
    deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

    # 遍历所有层
    for layer in self.layers:
        # 如果输出所有隐藏状态，则将当前隐藏状态加入到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 添加LayerDrop
        dropout_probability = torch.rand([])
        skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False

        # 如果不跳过该层或启用了deepspeed_zero3
        if not skip_the_layer or deepspeed_zero3_is_enabled:
            # 如果启用了渐变检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用渐变检查点函数进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # 使用当前层进行前向传播
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                
            # 更新隐藏状态
            hidden_states = layer_outputs[0]

        # 如果跳过该层，则layer_outputs为(None, None)
        if skip_the_layer:
            layer_outputs = (None, None)

        # 如果输出所有自注意力，则将当前自注意力加入到all_self_attentions中
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    # 如果输出所有隐藏状态，则将最后一个隐藏状态加入到all_hidden_states中
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # 如果不返回字典形式的结果
    if not return_dict:
        # 返回包含非None值的元组
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    
    # 返回字典形式的结果
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
# 定义 Data2VecAudioAdapter 类，继承自 nn.Module
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Adapter 复制过来，将 Wav2Vec2 替换为 Data2VecAudio
class Data2VecAudioAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 如果输出的隐藏层大小与隐藏层大小不一致，可能需要进行降维投影
        if config.output_hidden_size != config.hidden_size:
            # 创建线性投影层
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # 创建投影层的 LayerNorm
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        # 创建 Data2VecAudioAdapterLayer 的模块列表
        self.layers = nn.ModuleList(Data2VecAudioAdapterLayer(config) for _ in range(config.num_adapter_layers))
        # 设置 layerdrop
        self.layerdrop = config.layerdrop

    # 前向传播函数
    def forward(self, hidden_states):
        # 如果需要进行投影
        if self.proj is not None and self.proj_layer_norm is not None:
            # 对 hidden_states 进行投影
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 将 hidden_states 转置
        hidden_states = hidden_states.transpose(1, 2)

        # 遍历各个层
        for layer in self.layers:
            # 计算是否进行 dropout
            layerdrop_prob = np.random.random()
            # 如果不是训练状态或者不进行 dropout，则执行该层
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        # 再次将 hidden_states 转置
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 定义 Data2VecAudioAdapterLayer 类，继承自 nn.Module
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AdapterLayer 复制过来，将 Wav2Vec2 替换为 Data2VecAudio
class Data2VecAudioAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建卷积层
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    # 前向传播函数
    def forward(self, hidden_states):
        # 对 hidden_states 进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对 hidden_states 进行 Gated Linear Unit (GLU) 操作
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


# 定义 Data2VecAudioPreTrainedModel 类，继承自 PreTrainedModel
# 一个处理权重初始化以及下载和加载预训练模型的抽象类
class Data2VecAudioPreTrainedModel(PreTrainedModel):
    config_class = Data2VecAudioConfig  # 设置配置类
    base_model_prefix = "data2vec_audio"  # 设置基础模型前缀
    main_input_name = "input_values"  # 设置主要输入名称
    supports_gradient_checkpointing = True  # 支持梯度检查点
    # 初始化模型参数的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是 Data2VecAudioFeatureProjection 类型
        if isinstance(module, Data2VecAudioFeatureProjection):
            # 计算初始化权重的标准差
            k = math.sqrt(1 / module.projection.in_features)
            # 使用均匀分布初始化投影层的权重和偏置
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果模块是 Data2VecAudioPositionalConvLayer 类型
        elif isinstance(module, Data2VecAudioPositionalConvLayer):
            # 将卷积层的偏置初始化为零
            nn.init.constant_(module.conv.bias, 0)
        # 如果模块是线性层
        elif isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            # 如果有偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 LayerNorm 或 GroupNorm 类型
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 如果有偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
            # 如果有权重，则将权重初始化为1
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        # 如果模块是一维卷积层
        elif isinstance(module, nn.Conv1d):
            # 使用 Kaiming 正态分布初始化权重
            nn.init.kaiming_normal_(module.weight)

            # 如果有偏置，则使用均匀分布初始化偏置
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # 从输入长度计算特征提取层的输出长度的函数
    # 复制自 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PreTrainedModel._get_feat_extract_output_lengths
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 PyTorch 文档中获取一维卷积层的输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 对于每个卷积核大小和步长的组合
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            # 计算卷积层的输出长度
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果启用了适配器
        if add_adapter:
            # 对于每个适配器层
            for _ in range(self.config.num_adapter_layers):
                # 计算适配器层的输出长度
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    # 获取特征向量注意力掩码的函数
    # 复制自 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PreTrainedModel._get_feature_vector_attention_mask
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    # 计算非填充部分的长度，相当于 attention_mask.sum(-1)，但不是原地操作以便在推理模式下运行
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

    # 获取特征提取器的输出长度，根据是否添加适配器
    output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
    output_lengths = output_lengths.to(torch.long)

    # 获取批量大小
    batch_size = attention_mask.shape[0]

    # 创建与注意力掩码形状相同的全零张量
    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )

    # 确保在输出长度索引之前的所有值都被注意到
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
    
    # 将张量按最后一个维度翻转，然后累积求和，再翻转回来，并转换为布尔类型
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    
    # 返回注意力掩码
    return attention_mask
DATA2VEC_AUDIO_START_DOCSTRING = r"""
    Data2VecAudio was proposed in [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and
    Language](https://arxiv.org/pdf/2202.03555) by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu and
    Michael Auli.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Data2VecAudioConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Data2VecAudio 的文档字符串，提供了模型的说明和使用方法，包括来源、继承关系、参数等
DATA2VEC_AUDIO_INPUTS_DOCSTRING = r"""
# 未提供注释
"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should be passed if the corresponding processor has `config.return_attention_mask ==
            True`, which is the case for all pre-trained Data2Vec Audio models. Be aware that that even with
            `attention_mask`, zero-padded inputs will have slightly different outputs compared to non-padded inputs
            because there are more than one convolutional layer in the positional encodings. For a more detailed
            explanation, see [here](https://github.com/huggingface/transformers/issues/25621#issuecomment-1713759349).

            </Tip>

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
定义 Data2VecAudioModel 类，继承自 Data2VecAudioPreTrainedModel 类
@param config: Data2VecAudioConfig 类的实例，包含模型的配置信息
"""
class Data2VecAudioModel(Data2VecAudioPreTrainedModel):
    def __init__(self, config: Data2VecAudioConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 将传入的配置信息保存到模型的属性中
        self.config = config
        # 创建 Data2VecAudioFeatureEncoder 对象，用于提取音频特征
        self.feature_extractor = Data2VecAudioFeatureEncoder(config)
        # 创建 Data2VecAudioFeatureProjection 对象，用于特征投影
        self.feature_projection = Data2VecAudioFeatureProjection(config)

        # 如果 mask 时间的概率大于 0.0 或者 mask 特征的概率大于 0.0，则模型需要 masking 向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            # 创建一个随机初始化的可学习参数，用于 masking
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 创建 Data2VecAudioEncoder 对象，用于编码音频特征
        self.encoder = Data2VecAudioEncoder(config)

        # 如果配置中需要添加 adapter，则创建 Data2VecAudioAdapter 对象
        self.adapter = Data2VecAudioAdapter(config) if config.add_adapter else None

        # 初始化模型权重并进行最终处理
        self.post_init()

    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新。
        """
        # 调用特征编码器对象的方法，冻结其参数
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    # 根据 SpecAugment 方法对提取的特征进行时间轴和特征轴的遮蔽
    def mask_time_and_feature(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> torch.Tensor:
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
    
        # `config.apply_spec_augment` can set masking to False
        # 如果配置中设置 apply_spec_augment 为 False，则直接返回隐藏层状态 hidden_states
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states
    
        # generate indices & apply SpecAugment along time axis
        # 生成所需的索引，并且沿时间轴应用 SpecAugment
        batch_size, sequence_length, hidden_size = hidden_states.size()
    
        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            # 使用给定的 mask_time_indices 遮蔽隐藏层 hidden_states 里的数据，进行 SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 如果 mask_time_indices 为 None，并且在训练模式下，按照一定概率生成 mask_time_indices，并对隐藏层 hidden_states 进行 SpecAugment
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
            # 如果在训练模式下，按照一定概率生成 mask_feature_indices，并对隐藏层 hidden_states 进行 SpecAugment
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0
    
        # 返回处理后的隐藏层状态 hidden_states
        return hidden_states
    
    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Wav2Vec2BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 重写父类 BaseWav2Vec2Model 的 forward 方法
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Performs the forward pass of Wav2Vec2Model. Input values can be obtained from Wav2Vec2Processor.
        """
    
        # 根据输入参数和配置信息，执行前向传播
    # 定义函数，参数包括input_values、attention_mask、output_attentions、output_hidden_states、return_dict
    # 返回值包括Tuple类型或Wav2Vec2BaseModelOutput类型的对象
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 如果output_attentions不为空，则使用output_attentions，否则使用self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states不为空，则使用output_hidden_states，否则使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict不为空，则使用return_dict，否则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征，返回的结果是shape转置后的extract_features
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        # 如果attention_mask不为空，则根据extract_features的shape来计算reduced attention_mask
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 将extract_features传入feature_projection进行特征投影
        # 返回hidden_states和extract_features
        hidden_states, extract_features = self.feature_projection(extract_features)

        # 对hidden_states进行掩码操作
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 将hidden_states传入encoder进行编码，获取encoder_outputs
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从encoder_outputs中取出hidden_states
        hidden_states = encoder_outputs[0]

        # 如果存在adapter，则对hidden_states进行转换
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 如果return_dict为False，返回包含hidden_states、extract_features和encoder_outputs[1:]的tuple
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 如果return_dict为True，返回Wav2Vec2BaseModelOutput对��，包含last_hidden_state、extract_features、encoder_outputs的hidden_states和attentions
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 给 Data2VecAudioForCTC 类添加注释：使用上方的语言模型（Language Modeling）头，并进行 CTC（Connectionist Temporal Classification）任务
@add_start_docstrings(
    """Data2VecAudio Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    DATA2VEC_AUDIO_START_DOCSTRING,
)

# Data2VecAudioForCTC 类继承自 Data2VecAudioPreTrainedModel 类
class Data2VecAudioForCTC(Data2VecAudioPreTrainedModel):

    # 构造函数，接收一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)

        # 创建一个 Data2VecAudioModel 对象
        self.data2vec_audio = Data2VecAudioModel(config)
        # 创建一个 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 如果配置对象没有定义语言模型头的词汇表大小，则抛出异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        # 初始化输出层的线性变换
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重，进行最后的处理
        self.post_init()

    # 冻结特征提取器（feature extractor）的梯度计算
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

    # 冻结特征提取器（feature encoder）的梯度计算    
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()

    # 重载的 forward 方法
    # 参数和返回值请参考 add_start_docstrings_to_model_forward 和 add_code_sample_docstrings 的注释
    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward 复制而来，将wav2vec2替换为data2vec_audio
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

        outputs = self.data2vec_audio(
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
        # 检查是否传入了标签，如果有则计算损失
        if labels is not None:
            # 检查标签是否超出词汇表的大小
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 从注意力遮罩中提取输入长度
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充的标记用-100填充，当不被注意时
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # CTC损失不支持fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 禁用cudnn标志并计算损失
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

        # 如果不需返回字典，则返回损失和输出
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回损失、logits、隐藏层和注意力
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 在Data2VecAudio模型中添加顶部的序列分类头，用于像SUPERB Keyword Spotting这样的任务
@add_start_docstrings(
    """
    Data2VecAudio Model with a sequence classification head on top (a linear layer over the pooled output) for tasks
    like SUPERB Keyword Spotting.
    """,
    DATA2VEC_AUDIO_START_DOCSTRING,
)
class Data2VecAudioForSequenceClassification(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中存在"add_adapter"属性且为True，则抛出值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Data2VecAudio adapters (config.add_adapter=True)"
            )
        # 初始化Data2VecAudioModel
        self.data2vec_audio = Data2VecAudioModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        # 如果配置中使用加权层求和，则初始化权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 初始化投影层和分类器
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征提取器的梯度
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

    # 冻结特征编码器的梯度
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()

    # 冻结基础模型的梯度，只更新分类头
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False

    # 添加开始注释到模型前向方法
    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    # 添加代码示例注释
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward 复制到data2vec_audio
        # 前向传播函数，用于执行模型的前向传播操作
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
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
                用于计算序列分类/回归损失的标签。指数应在`[0，...，config.num_labels - 1]`范围内。如果`config.num_labels == 1`，则计算回归损失（均方误差损失），如果`config.num_labels > 1`，则计算分类损失（交叉熵损失）。
            """
    
            # 如果return_dict不为None，则使用传入的return_dict值，否则使用self.config.use_return_dict
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # 如果self.config.use_weighted_layer_sum为True，则output_hidden_states设为True，否则保持原值
            output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
    
            # 通过调用data2vec_audio方法执行特征提取
            outputs = self.data2vec_audio(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    
            # 根据模型配置选择不同的计算方式
            if self.config.use_weighted_layer_sum:
                hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
                hidden_states = torch.stack(hidden_states, dim=1)
                norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
                hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            else:
                hidden_states = outputs[0]
    
            # 将隐藏状态传入投影层
            hidden_states = self.projector(hidden_states)
            # 如果attention_mask为None，则计算隐藏状态的平均值作为池化输出，否则根据attention_mask进行填充掩码计算池化输出
            if attention_mask is None:
                pooled_output = hidden_states.mean(dim=1)
            else:
                padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
                hidden_states[~padding_mask] = 0.0
                pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
    
            # 通过分类器得到logits
            logits = self.classifier(pooled_output)
    
            # 初始化损失为None
            loss = None
            # 如果存在标签，则计算损失
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
    
            # 根据return_dict决定返回结果的格式
            if not return_dict:
                output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
                return ((loss,) + output) if loss is not None else output
    
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        """
        Data2VecAudio Model with a frame classification head on top for tasks like Speaker Diarization.
        """
        # 在 Data2VecAudioForAudioFrameClassification 类中定义了一个带有帧分类头的模型，用于说话人分离等任务
        @add_start_docstrings(
            "Data2VecAudio Model with a frame classification head on top for tasks like Speaker Diarization.",
            DATA2VEC_AUDIO_START_DOCSTRING,
        )
        class Data2VecAudioForAudioFrameClassification(Data2VecAudioPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                
                # 如果配置中存在 add_adapter 属性并且为真，则抛出异常
                if hasattr(config, "add_adapter") and config.add_adapter:
                    raise ValueError(
                        "Audio frame classification does not support the use of Data2VecAudio adapters"
                        " (config.add_adapter=True)"
                    )
                # 创建 Data2VecAudioModel 模型
                self.data2vec_audio = Data2VecAudioModel(config)
                # 定义层数
                num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
                # 如果配置中使用加权求和，则创建层权重参数
                if config.use_weighted_layer_sum:
                    self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
                # 创建分类器
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
                # 定义标签数
                self.num_labels = config.num_labels
                
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
                self.data2vec_audio.feature_extractor._freeze_parameters()

            # 冻结基础模型的梯度计算
            def freeze_base_model(self):
                """
                Calling this function will disable the gradient computation for the base model so that its parameters will not
                be updated during training. Only the classification head will be updated.
                """
                for param in self.data2vec_audio.parameters():
                    param.requires_grad = False

            @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
            @add_code_sample_docstrings(
                checkpoint=_CHECKPOINT_FOR_DOC,
                output_type=TokenClassifierOutput,
                config_class=_CONFIG_FOR_DOC,
                modality="audio",
            )
            # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.forward 复制而来的代码
            # 进行模型的前向传播
            # 包含输入值、注意力掩码、标签、输出注意力、输出隐藏状态和返回字典等参数
            def forward(
                self,
                input_values: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
        ```
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 确定是否返回字典格式的输出，若未指定则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 若使用加权层求和，需要将隐藏状态输出设置为 True
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用 data2vec_audio 方法对音频数据进行向量化处理
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 若使用加权层求和
        if self.config.use_weighted_layer_sum:
            # 提取模型输出的隐藏状态
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 将隐藏状态堆叠起来，以便进行加权求和
            hidden_states = torch.stack(hidden_states, dim=1)
            # 对层权重进行 softmax 归一化
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 对隐藏状态进行加权求和
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 若不使用加权层求和，则直接使用模型输出的第一个隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态输入分类器，得到 logits
        logits = self.classifier(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        # 若不返回字典格式的输出
        if not return_dict:
            # 组装输出结果
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        # 返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```  
# 从transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss复制而来的类
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale  # 缩放参数
        self.margin = margin  # 边界参数
        self.num_labels = num_labels  # 标签数量
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)  # 权重参数
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, hidden_states, labels):
        labels = labels.flatten()  # 将标签展平
        weight = nn.functional.normalize(self.weight, dim=0)  # 标准化权重
        hidden_states = nn.functional.normalize(hidden_states, dim=1)  # 标准化隐藏状态
        cos_theta = torch.mm(hidden_states, weight)  # 计算余弦相似度
        psi = cos_theta - self.margin  # 计算 psi 值

        onehot = nn.functional.one_hot(labels, self.num_labels)  # 将标签编码为 one-hot 向量
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)  # 计算最终的 logits 值
        loss = self.loss(logits, labels)  # 计算损失值

        return loss
# 从transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer复制而来的类
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]  # 输入卷积维度
        self.out_conv_dim = config.tdnn_dim[layer_id]  # 输出卷积维度
        self.kernel_size = config.tdnn_kernel[layer_id]  # 卷积核大小
        self.dilation = config.tdnn_dilation[layer_id]  # 膨胀系数

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)  # 线性层
        self.activation = nn.ReLU()  # ReLU激活函数

    def forward(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(1)  # 添加维度
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),  # 卷积核大小和输入维度
            stride=(1, self.in_conv_dim),  # 步长
            dilation=(self.dilation, 1),  # 膨胀系数
        )
        hidden_states = hidden_states.transpose(1, 2)  # 转置
        hidden_states = self.kernel(hidden_states)  # 卷积操作

        hidden_states = self.activation(hidden_states)  # 激活函数
        return hidden_states

@add_start_docstrings(
    """
    Data2VecAudio Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    DATA2VEC_AUDIO_START_DOCSTRING,  # 添加文档字符串
)
class Data2VecAudioForXVector(Data2VecAudioPreTrainedModel):  # 为 XVector 特征提取头的 Data2VecAudio 模型
    def __init__(self, config):
        super().__init__(config)

        # 初始化音频数据到向量的模型
        self.data2vec_audio = Data2VecAudioModel(config)
        # 计算 Transformer 层数加上输入嵌入层的数量
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        # 如果配置中使用加权层求和
        if config.use_weighted_layer_sum:
            # 初始化层权重为均匀分布的参数
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 初始化线性投影层
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 初始化一系列 TDNN 层
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 初始化特征提取器
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        # 初始化分类器
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 初始化 AMSoftmax 损失函数
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 初始化权重
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 引发警告，说明该方法已被弃用，建议使用相应的新方法
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用对应的新方法
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数，使其在训练过程中不会更新
        self.data2vec_audio.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 冻结基础模型的参数，使其在训练过程中不会更新，只有分类头会被更新
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层输出长度的计算公式，参考自 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        # 计算 TDNN 层的输出长度
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector.forward 复制过来，将 wav2vec2 改为 data2vec_audio
    # 定义一个名为forward的方法，用于进行前向传播
    def forward(
        # 输入数值，类型为torch张量，可选参数
        input_values: Optional[torch.Tensor],
        # 注意力掩码，类型为torch张量，可选参数，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 输出注意力权重，类型为布尔值，可选参数，默认为None
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态，类型为布尔值，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,
        # 返回字典，类型为布尔值，可选参数，默认为None
        return_dict: Optional[bool] = None,
        # 标签，类型为torch张量，可选参数，默认为None
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 如果 return_dict 为 None，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 self.config.use_weighted_layer_sum 为真，则设置 output_hidden_states 为 True，否则保持原值
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 使用 data2vec_audio 方法处理输入数据
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用权重层求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接取第一个输出
            hidden_states = outputs[0]

        # 通过 projector 进行投影
        hidden_states = self.projector(hidden_states)

        # 遍历 tdnn 层
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # 统计池化
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

        # 通过 feature_extractor 处理统计池化结果，再通过 classifier 进行分类
        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        # 如果存在 labels，计算损失值
        if labels is not None:
            loss = self.objective(logits, labels)

        # 如果不需要返回字典形式的结果
        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回 XVectorOutput 对象
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```