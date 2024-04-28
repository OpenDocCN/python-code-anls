# `.\transformers\models\unispeech\modeling_unispeech.py`

```py
# 设置编码格式为 utf-8
# 版权声明信息
# 版权声明信息来源于 The Fairseq Authors 和 HuggingFace Inc. 团队，保留所有权利。
# 根据 Apache 许可证 2.0 版本（“许可证”）获得授权
# 除非符合许可证要求或书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据“按原样”分发，不提供任何形式的明示或默示保证或条件
# 请查看许可证以了解权限和限制
""" PyTorch UniSpeech model."""

# 导入必要的库
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入 HuggingFace 中的相关模块
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput, Wav2Vec2BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_unispeech import UniSpeechConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态开始的位置（索引）
_HIDDEN_STATES_START_POSITION = 2

# 通用文档字符串
_CONFIG_FOR_DOC = "UniSpeechConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "patrickvonplaten/unispeech-large-1500h-cv-timit"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 1024]

# CTC（Connectionist Temporal Classification）文档字符串
_CTC_EXPECTED_OUTPUT = "'mister quilter is the apposl of the midle classes and weare glad to welcom his gosepl'"
_CTC_EXPECTED_LOSS = 17.17

# UniSpeech 预训练模型存档列表
UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/unispeech-large-1500h-cv",
    "microsoft/unispeech-large-multi-lingual-1500h-cv",
    # 查看所有 UniSpeech 模型：https://huggingface.co/models?filter=unispeech
]


# 数据类，定义 UniSpeech 预训练模型输出
@dataclass
class UniSpeechForPreTrainingOutput(ModelOutput):
    """
    Output type of [`UniSpeechForPreTrainingOutput`], with potential hidden states and attentions.
    Args:
        # 损失函数，包括对比损失（L_m）和多样性损失（L_d），参考论文链接
        loss (*optional*, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`):

        # 模型隐藏状态投影到`config.proj_codevector_dim`维度，用于预测掩码后的投影量化状态
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):

        # 量化提取特征向量，表示对比损失的正类目标向量
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):

        # 模型隐藏状态，包括嵌入输出和每一层的输出
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):

        # 注意力权重，用于计算自注意力头中的加权平均值
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):

    """
    
    # 损失
    loss: Optional[torch.FloatTensor] = None
    # 模型隐藏状态
    projected_states: torch.FloatTensor = None
    # 量化提取特征向量
    projected_quantized_states: torch.FloatTensor = None
    # 马氏距离
    codevector_perplexity: torch.FloatTensor = None
    # 模型隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 从 transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices 复制而来，用于计算随机的遮罩范围，实现了 SpecAugment 方法
def _compute_mask_indices(
    shape: Tuple[int, int],  # 接收一个元组形式的参数 shape，包含两个整数，表示批量大小和序列长度
    mask_prob: float,  # 表示要遮罩的轴上的百分比，取值范围在 0 到 1 之间
    mask_length: int,  # 遮罩的长度
    attention_mask: Optional[torch.LongTensor] = None,  # 一个可选的注意力遮罩，用于独立缩短每个批量维度的特征轴，右填充
    min_masks: int = 0,  # 最小的遮罩数量，默认为 0
) -> np.ndarray:  # 返回一个 NumPy 数组，用于表示遮罩的索引

    """
    计算给定形状的随机遮罩范围。用于实现 SpecAugment 方法。
    注意：该方法未经过优化，应在 CPU 上作为训练期间的预处理的一部分运行，不适用于 TPU。

    Args:
        shape: 要计算遮罩的形状。应为大小为 2 的元组，第一个元素是批量大小，第二个元素是要遮罩的轴的长度。
        mask_prob: 要遮罩的整个轴的百分比（0 到 1 之间）。通过 `mask_prob*shape[1]/mask_length` 计算生成长度为 `mask_length` 的独立遮罩范围的数量。
                   由于重叠，`mask_prob` 是一个上限，实际百分比会更小。
        mask_length: 遮罩的长度
        min_masks: 最小遮罩数量
        attention_mask: 一个（右填充的）注意力遮罩，用于独立缩短每个批次维度的特征轴。

    Returns:
        一个 NumPy 数组，表示遮罩的索引
    """

    batch_size, sequence_length = shape  # 解包 shape 参数，得到批量大小和序列长度

    if mask_length < 1:  # 如果遮罩长度小于 1，则抛出 ValueError 异常
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:  # 如果遮罩长度大于序列长度，则抛出 ValueError 异常
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon 用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """给定输入长度，计算应该遮罩的跨度数量"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)  # 计算应遮罩的跨度数量
        num_masked_span = max(num_masked_span, min_masks)  # 取遮罩数量和最小遮罩数量的较大值

        # 确保遮罩的数量不超过序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保遮罩数量不超过输入长度减去 (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批次中的遮罩跨度数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()  # 计算注意力遮罩的和，并将结果转换为列表
        if attention_mask is not None  # 如果注意力遮罩不为 None
        else [sequence_length for _ in range(batch_size)]  # 否则，生成长度为批次大小的序列长度列表
    )

    # 用于 SpecAugment 的遮罩
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 创建一个全零的布尔数组，用于表示遮罩
    spec_aug_mask_idxs = []  # 用于存储遮罩的索引列表

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算序列长度中最大的遮罩跨度数量
    # 如果最大被遮蔽跨度为0，则直接返回特殊的增强遮蔽
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算该输入的被遮蔽跨度的数量
        num_masked_span = compute_num_masked_span(input_length)

        # 获取随机的索引来进行遮蔽
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个被抽样的索引作为填充向量的虚拟索引
        # 以确保由于概率舍入而确保所有批次具有相同的维度
        # 选择第一个样本只会让这些向量填充两次
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只会在 `input_length` 严格小于 `sequence_length` 的情况下发生
            # 此时最后一个标记必须是填充标记，我们可以将其用作虚拟遮罩id
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

    # 添加偏移量以使起始索引现在创建一个跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不会大于 sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将索引散布为遮罩
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
# 创建一个自定义的卷积层类，用于构建无层规范化的卷积神经网络层
class UniSpeechNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1  # 定义输入卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]  # 定义输出卷积维度

        self.conv = nn.Conv1d(  # 创建一个一维卷积函数实例
            self.in_conv_dim,  # 输入维度
            self.out_conv_dim,  # 输出维度
            kernel_size=config.conv_kernel[layer_id],  # 卷积核大小
            stride=config.conv_stride[layer_id],  # 步幅
            bias=config.conv_bias,  # 是否使用偏置
        )
        self.activation = ACT2FN[config.feat_extract_activation]  # 设置激活函数

    # 前向传播函数
    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)  # 卷积操作
        hidden_states = self.activation(hidden_states)  # 激活函数
        return hidden_states  # 返回处理后的隐藏状态


# 创建一个自定义的卷积层类，用于构建带层规范化的卷积神经网络层
class UniSpeechLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1  # 定义输入卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]  # 定义输出卷积维度

        self.conv = nn.Conv1d(  # 创建一个一维卷积函数实例
            self.in_conv_dim,  # 输入维度
            self.out_conv_dim,  # 输出维度
            kernel_size=config.conv_kernel[layer_id],  # 卷积核大小
            stride=config.conv_stride[layer_id],  # 步幅
            bias=config.conv_bias,  # 是否使用偏置
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)  # 创建层规范化实例
        self.activation = ACT2FN[config.feat_extract_activation]  # 设置激活函数

    # 前向传播函数
    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)  # 卷积操作
        hidden_states = hidden_states.transpose(-2, -1)  # 转置操作
        hidden_states = self.layer_norm(hidden_states)  # 层规范化
        hidden_states = hidden_states.transpose(-2, -1)  # 再次转置操作
        hidden_states = self.activation(hidden_states)  # 激活函数
        return hidden_states  # 返回处理后的隐藏状态


# 创建一个自定义的卷积层类，用于构建带组规范化的卷积神经网络层
class UniSpeechGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1  # 定义输入卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]  # 定义输出卷积维度

        self.conv = nn.Conv1d(  # 创建一个一维卷积函数实例
            self.in_conv_dim,  # 输入维度
            self.out_conv_dim,  # 输出维度
            kernel_size=config.conv_kernel[layer_id],  # 卷积核大小
            stride=config.conv_stride[layer_id],  # 步幅
            bias=config.conv_bias,  # 是否使用偏置
        )
        self.activation = ACT2FN[config.feat_extract_activation]  # 设置激活函数
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)  # 创建组规范化实例

  # 前向传播函数
    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)  # 卷积操作
        hidden_states = self.layer_norm(hidden_states)  # 组规范化
        hidden_states = self.activation(hidden_states)  # 激活函数
        return hidden_states  # 返回处理后的隐藏状态
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding复制而来，将Wav2Vec2替换为UniSpeech
class UniSpeechPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个 1D 卷积层
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

        if is_deepspeed_zero3_enabled():
            import deepspeed
            # 使用DeepSpeed的分布式训练功能，将卷积层参数聚合到rank=0的modifier上
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 对卷积层进行权重归一化
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建一个与卷积层相同的填充函数
        self.padding = UniSpeechSamePadLayer(config.num_conv_pos_embeddings)
        # 设置激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 调整hidden_states的维度，将第一维和第二维交换位置
        hidden_states = hidden_states.transpose(1, 2)

        # 经过卷积层处理
        hidden_states = self.conv(hidden_states)
        # 经过填充层处理
        hidden_states = self.padding(hidden_states)
        # 经过激活函数处理
        hidden_states = self.activation(hidden_states)

        # 恢复hidden_states的维度，将第一维和第二维再次交换位置
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer复制而来，将Wav2Vec2替换为UniSpeech
class UniSpeechSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 根据卷积的位置嵌入数量确定需要移除的填充数量
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 根据需要移除的填充数量对hidden_states进行截取
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder复制而来，将Wav2Vec2替换为UniSpeech
class UniSpeechFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    # 初始化函数，接受一个配置参数，并调用父类的初始化函数
    def __init__(self, config):
        super().__init__()
    
        # 根据配置参数中的特征提取归一化方式选择不同的卷积层组合
        if config.feat_extract_norm == "group":
            conv_layers = [UniSpeechGroupNormConvLayer(config, layer_id=0)] + [
                UniSpeechNoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                UniSpeechLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 如果特征提取归一化方式不是 "group" 或 "layer"，则抛出数值错误
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        # 将卷积层组合封装成模块列表
        self.conv_layers = nn.ModuleList(conv_layers)
        # 是否进行梯度检查点
        self.gradient_checkpointing = False
        # 参数是否需要梯度
        self._requires_grad = True
    
    # 冻结参数函数，将所有参数设置为不需要梯度
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False
    
    # 前向传播函数，接受输入数值，进行卷积操作并返回结果
    def forward(self, input_values):
        # 将输入值添加一个维度，用于卷积操作
        hidden_states = input_values[:, None]
    
        # 如果参数需要梯度并且处于训练状态，则将隐藏状态设置为需要梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True
    
        # 遍历卷积层进行卷积操作
        for conv_layer in self.conv_layers:
            # 如果参数需要梯度，且进行梯度检查点，且处于训练状态，则采用梯度检查点函数
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)
    
        # 返回最终的隐藏状态
        return hidden_states
# 定义一个名为 UniSpeechFeatureExtractor 的类，继承自 UniSpeechFeatureEncoder
class UniSpeechFeatureExtractor(UniSpeechFeatureEncoder):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 发出一个警告，提醒用户该类即将被弃用，并在 Transformers v5 版本中移除，建议使用父类代替
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection 复制并修改为 UniSpeechFeatureProjection
class UniSpeechFeatureProjection(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化层归一化对象，输入维度为 config.conv_dim[-1]，epsilon 为 config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 初始化线性映射对象，输入维度为 config.conv_dim[-1]，输出维度为 config.hidden_size
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 初始化 Dropout 对象，丢弃概率为 config.feat_proj_dropout
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    # 前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行层归一化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态进行线性映射
        hidden_states = self.projection(norm_hidden_states)
        # 对线性映射后的隐藏状态进行 Dropout
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态和归一化后的隐藏状态
        return hidden_states, norm_hidden_states


# 从 transformers.models.bart.modeling_bart.BartAttention 复制并修改为 UniSpeechAttention
class UniSpeechAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化函数，接受多个参数，包括嵌入维度、头数、dropout、是否为解码器、是否有偏置、是否因果、配置对象等
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[UniSpeechConfig] = None,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化各个参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查头数是否合法
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 设置缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化键、值、查询、输出的线性映射对象
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 将张量重塑为适合多头注意力的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，接受隐藏状态等多个参数作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward复制代码并更名为UniSpeechFeedForward
class UniSpeechFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 定义中间层全连接层，将隐藏层大小转换为中间层大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            # 如果隐藏激活函数是字符串，则使用预定义的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 定义输出层全连接层，将中间层大小转换为隐藏层大小
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)  # 中间层全连接操作
        hidden_states = self.intermediate_act_fn(hidden_states)  # 应用中间层激活函数
        hidden_states = self.intermediate_dropout(hidden_states)  # 中间层的 dropout

        hidden_states = self.output_dense(hidden_states)  # 输出层全连接操作
        hidden_states = self.output_dropout(hidden_states)  # 输出层 dropout
        return hidden_states  # 返回隐藏状态


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer复制代码并更名为UniSpeechEncoderLayer
class UniSpeechEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 attention、dropout、LayerNorm 和 feed forward 层
        self.attention = UniSpeechAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = UniSpeechFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states  # 备份隐藏状态用于后续加法残差连接
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )  # 计算注意力，获取注意力权重
        hidden_states = self.dropout(hidden_states)  # 对隐藏状态进行 dropout
        hidden_states = attn_residual + hidden_states  # 执行加法残差连接

        hidden_states = self.layer_norm(hidden_states)  # LayerNorm 层
        hidden_states = hidden_states + self.feed_forward(hidden_states)  # 加上 feed forward 层的输出
        hidden_states = self.final_layer_norm(hidden_states)  # 最终的 LayerNorm

        outputs = (hidden_states,)  # 结果元组

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则加入结果元组

        return outputs  # 返回结果


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AttnAdapterLayer复制代码并更名为UniSpeechAttnAdapterLayer
class UniSpeechAttnAdapterLayer(nn.Module):
    # 初始化适配器模块，使用3D张量权重作为参数，不使用ModuleList以提高训练吞吐量
    def __init__(self, config):
        # 调用父类构造函数进行初始化
        super().__init__()
        # 从配置中获取适配器的注意力维度和隐藏层维度
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size

        # 使用LayerNorm对隐藏状态进行规范化
        self.norm = nn.LayerNorm(self.hidden_dim)
        # 创建一个全连接层，将隐藏状态映射到适配器输入维度
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        # 激活函数使用ReLU
        self.act_fn = nn.ReLU()
        # 创建一个全连接层，将适配器输入维度映射回隐藏层维度
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    # 前向传播函数，接受隐藏状态作为输入，返回经过适配器模块处理后的隐藏状态
    def forward(self, hidden_states: torch.FloatTensor):
        # 对输入的隐藏状态进行规范化
        hidden_states = self.norm(hidden_states)

        # 将规范化后的隐藏状态通过第一个全连接层
        hidden_states = self.linear_1(hidden_states)
        # 经过激活函数处理
        hidden_states = self.act_fn(hidden_states)
        # 将结果通过第二个全连接层
        hidden_states = self.linear_2(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个名为UniSpeechEncoderLayerStableLayerNorm的类，继承自nn.Module
class UniSpeechEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化Self-attention层
        self.attention = UniSpeechAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 初始化Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化LayerNorm层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化Feed-Forward层
        self.feed_forward = UniSpeechFeedForward(config)
        # 初始化最终的LayerNorm层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果配置中存在adapter_attn_dim，则初始化AttnAdapterLayer，否则设为None
        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = UniSpeechAttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 保存Self-attention前的hidden_states作为残差连接的值
        attn_residual = hidden_states
        # 对输入进行LayerNorm
        hidden_states = self.layer_norm(hidden_states)
        # 进行Self-attention计算
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 对Self-attention计算结果进行Dropout
        hidden_states = self.dropout(hidden_states)
        # 将残差连接的值与Self-attention计算结果相加
        hidden_states = attn_residual + hidden_states
        # 对结果进行Feed-Forward计算和LayerNorm
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 如果存在adapter_layer，则对结果进行适配层的计算
        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        outputs = (hidden_states,)

        # 如果需要输出attention权重，则将attention权重添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 定义一个名为UniSpeechEncoder的类，继承自nn.Module
class UniSpeechEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化位置卷积嵌入层
        self.pos_conv_embed = UniSpeechPositionalConvEmbedding(config)
        # 初始化LayerNorm层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化多个UniSpeechEncoderLayer层
        self.layers = nn.ModuleList([UniSpeechEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 如果未设置输出隐藏状态，则初始化为一个空元组
    all_hidden_states = () if output_hidden_states else None
    # 如果未设置输出注意力，则初始化为一个空元组
    all_self_attentions = () if output_attentions else None

    # 如果有设置注意力遮罩
    if attention_mask is not None:
        # 确保填充的标记输出为0
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        hidden_states[~expand_attention_mask] = 0

        # 扩展注意力遮罩
        attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        )

    # 计算位置嵌入
    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # 判断是否启用deepspeed zero3
    deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

    # 遍历每个层
    for layer in self.layers:
        # 如果设置了输出隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 添加LayerDrop（参考https://arxiv.org/abs/1909.11556）
        dropout_probability = torch.rand([])

        skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
        if not skip_the_layer or deepspeed_zero3_is_enabled:
            # 在deepspeed zero3下所有GPU必须同步运行
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
            hidden_states = layer_outputs[0]

        if skip_the_layer:
            layer_outputs = (None, None)

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    # 如果设置了输出隐藏状态
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # 如果没有设置返回字典，则返回相关元组
    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    # 返回BaseModelOutput类型的对象
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderStableLayerNorm复制代码，将Wav2Vec2替换为UniSpeech
class UniSpeechEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化位置卷积嵌入层
        self.pos_conv_embed = UniSpeechPositionalConvEmbedding(config)
        # 初始化层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化dropout层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化编码器层列表
        self.layers = nn.ModuleList(
            [UniSpeechEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        # 梯度检查点设为False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        # 如果未设置输出隐藏状态，则将 all_hidden_states 设置为一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果未设置输出注意力，则将 all_self_attentions 设置为空元组
        all_self_attentions = () if output_attentions else None

        # 如果注意力掩码不为空
        if attention_mask is not None:
            # 确保填充的标记不被注意到
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # 扩展注意力掩码
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 使用位置卷积嵌入层处理隐藏状态
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # 对隐藏状态进行丢弃
        hidden_states = self.dropout(hidden_states)

        # 检查是否启用了deepspeed_zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 对每个层进行循环处理
        for layer in self.layers:
            # 如果设置了输出隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加LayerDrop (参见https://arxiv.org/abs/1909.11556)
            dropout_probability = torch.rand([])

            # 如果训练并且随机数小于配置的dropout概率，则跳过该层
            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False

            # 如果不跳过该层或者启用了deepspeed_zero3
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果启用了梯度检查点且在训练中
                if self.gradient_checkpointing and self.training:
                    # 使用梯度检查点函数来运行层
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    # 运行层
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            # 如果跳过该层，则层的输���设置为None
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果设置了输出注意力
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对隐藏状态进行layer_norm处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果设置了输出隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果
        if not return_dict:
            # 返回结果的元组
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class UniSpeechGumbelVectorQuantizer(nn.Module):
    """
    使用 Gumbel Softmax 进行矢量量化。详见 https://arxiv.org/pdf/1611.01144.pdf。
    """

    def __init__(self, config):
        super().__init__()
        # 确定码本的组数
        self.num_groups = config.num_codevector_groups
        # 确定每组码本的个数
        self.num_vars = config.num_codevectors_per_group

        # 检查码本维度是否能被组数整除
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} 必须能被 `config.num_codevector_groups`"
                f" {self.num_groups} 整除，以便进行连接"
            )

        # 用于存储码本的参数（码字）
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        # 权重投影层
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # 温度参数，可用于训练过程中的衰减
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs):
        # 计算边际概率
        marginal_probs = probs.mean(dim=0)
        # 计算困惑度
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity
    # 定义前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 获取输入数据的形状信息
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 投影到编码器向量维度
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 通过古贝尔分布采样编码向量的概率，以可微分的方式
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # 计算困惑度
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            # 非可微分的方式取最大值
            # 计算硬编码向量分布（独热编码）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率检索编码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity
class UniSpeechPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置默认的配置类为UniSpeechConfig
    config_class = UniSpeechConfig
    # 模型的前缀，用于指定模型中的主要组件
    base_model_prefix = "unispeech"
    # 主输入名称，用于指定模型的主要输入
    main_input_name = "input_values"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是UniSpeechGumbelVectorQuantizer类型，则采用特殊的初始化方法
        if isinstance(module, UniSpeechGumbelVectorQuantizer):
            # 使用正态分布初始化权重
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            # 将偏置初始化为零
            module.weight_proj.bias.data.zero_()
            # 对codevectors进行均匀分布初始化
            nn.init.uniform_(module.codevectors)
        # 如果模块是UniSpeechPositionalConvEmbedding类型，则采用特殊的初始化方法
        elif isinstance(module, UniSpeechPositionalConvEmbedding):
            # 使用正态分布初始化卷积层的权重
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积层的偏置初始化为零
            nn.init.constant_(module.conv.bias, 0)
        # 如果模块是UniSpeechFeatureProjection类型，则采用特殊的初始化方法
        elif isinstance(module, UniSpeechFeatureProjection):
            # 计算初始化权重的范围
            k = math.sqrt(1 / module.projection.in_features)
            # 使用均匀分布初始化投影层的权重
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            # 使用均匀分布初始化投影层的偏置
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果模块是nn.Linear类型，则采用正态分布初始化权重
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            # 如果有偏置，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是nn.LayerNorm或nn.GroupNorm类型，则将偏置初始化为零，将权重初始化为1
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果模块是nn.Conv1d类型，则采用Kaiming正态分布初始化权重
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            # 如果有偏置，则计算初始化的范围，并使用均匀分布初始化偏置
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html获取的一维卷积层的输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 计算卷积层的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    # 获取特征向量的注意力掩码
    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # 计算非填充长度，即每个样本的有效长度
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        # 通过有效长度获取特征提取器的输出长度，并转换为整数类型
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)
        # 获取输入的 batch 大小
        batch_size = attention_mask.shape[0]

        # 创建一个与输入相同大小的全 0 注意力掩码
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # 将 output_lengths - 1 处的位置标记为 1，表示需要关注的位置
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # 反转 attention_mask，并对每行进行累加，再次反转 attention_mask，并全部转换为布尔类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        # 返回处理后的 attention_mask
        return attention_mask
# 定义 UniSpeech 模型的文档字符串，包括了该模型的提出论文链接、继承关系以及参数说明
UNISPEECH_START_DOCSTRING = r"""
    UniSpeech was proposed in [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled
    Data](https://arxiv.org/abs/2101.07597) by Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei,
    Michael Zeng, Xuedong Huang.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`UniSpeechConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 UniSpeech 模型输入的文档字符串
UNISPEECH_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            输入的原始语音波形的浮点值。可以通过加载 `.flac` 或 `.wav` 音频文件并转换为 `List[float]` 或 `numpy.ndarray` 类型的数组来获得值。
            使用 `AutoProcessor` 将数组准备成 `input_values` 张量的方法，请参见 [`Wav2Vec2Processor.__call__`]。

        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            遮罩，用于避免在填充记号索引上执行卷积和注意力操作。遮罩值取 `[0, 1]`：

            - 1 表示**未遮罩的**记号，
            - 0 表示**遮罩的**记号。

            [注意力遮罩是什么？](../glossary#attention-mask)

            <Tip warning={true}>

            如果相应的处理器具有 `config.return_attention_mask == True`，则应传递 `attention_mask`。
            对于处理器的 `config.return_attention_mask == False` 的所有模型，在执行批处理推断时传递 `attention_mask` 会导致性能下降。
            对于这些模型，应使用 0 填充 `input_values`，然后在不传递 `attention_mask` 的情况下传递。
            请注意，这些模型的结果在 `input_values` 填充与否时略有不同。

            </Tip>

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`]，而不是普通的元组。
# 定义一个UniSpeechModel类，继承自UniSpeechPreTrainedModel类
# 给UniSpeech Model添加文档字符串，描述其输出原始隐藏状态而不带特定的顶层头部
# 使用UniSpeech_START_DOCSTRING作为起始文档字符串的一部分
class UniSpeechModel(UniSpeechPreTrainedModel):
    # 初始化UniSpeechModel类
    def __init__(self, config: UniSpeechConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置config属性为传递进来的config
        self.config = config
        # 创建UniSpeechFeatureEncoder对象并赋值给feature_extractor属性
        self.feature_extractor = UniSpeechFeatureEncoder(config)
        # 创建UniSpeechFeatureProjection对象并赋值给feature_projection属性
        self.feature_projection = UniSpeechFeatureProjection(config)

        # 如果mask_time_prob或mask_feature_prob大于0，则初始化masked_spec_embed属性
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 根据config中的do_stable_layer_norm属性选择使用UniSpeechEncoderStableLayerNorm类或UniSpeechEncoder类，并将其赋值给encoder属性
        if config.do_stable_layer_norm:
            self.encoder = UniSpeechEncoderStableLayerNorm(config)
        else:
            self.encoder = UniSpeechEncoder(config)

        # 执行初始化权重和应用最终处理的方法
        self.post_init()

    # 定义_mask_hidden_states方法，用来处理隐藏状态数据
    # 方法接受输入的隐藏状态、时间索引的mask和注意力mask
    # 方法将对hidden_states进行处理并返回处理后的结果
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    # Defines a function to apply SpecAugment along time and feature axis on extracted features
    def apply_spec_augment(
        self,
        hidden_states: torch.Tensor,
        mask_time_indices: Optional[torch.Tensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
    
        # Check if SpecAugment should be applied based on config
        if not getattr(self.config, "apply_spec_augment", True):
            # If apply_spec_augment is False in the config, return the hidden states unchanged
            return hidden_states
    
        # Get dimensions of the hidden states tensor
        batch_size, sequence_length, hidden_size = hidden_states.size()
    
        # Apply masking along time axis based on given mask_time_indices
        if mask_time_indices is not None:
            # Apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        # If mask_time_indices is not provided, generate mask indices based on config settings
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
    
        # Apply masking along feature axis based on config settings
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
            # Mask features along the feature axis
            hidden_states[mask_feature_indices] = 0
    
        # Return the masked hidden states
        return hidden_states
    # 定义方法，用于推理（inference），接收输入并返回模型输出
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 如果输出注意力（attentions）参数为非空，则使用该参数，否则使用模型配置中的输出注意力设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态（hidden_states）参数为非空，则使用该参数，否则使用模型配置中的输出隐藏状态设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典（return_dict）参数为非空，则使用该参数，否则使用模型配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征（feature_extractor）从输入值（input_values）中提取特征
        extract_features = self.feature_extractor(input_values)
        # 调整提取的特征的维度顺序，将时间维度与特征维度进行交换
        extract_features = extract_features.transpose(1, 2)

        # 如果存在注意力掩码（attention_mask）
        if attention_mask is not None:
            # 计算对应于特征向量的减少注意力掩码（attention_mask）
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        # 将特征投影到隐藏状态空间
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 对隐藏状态进行遮蔽，根据时间索引（mask_time_indices）和注意力掩码（attention_mask）
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 使用编码器（encoder）对隐藏状态进行编码
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果不返回字典
        if not return_dict:
            # 返回元组，包含隐藏状态、提取的特征以及编码器输出的其它部分
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回包含最后隐藏状态、提取的特征、编码器隐藏状态以及注意力的模型输出
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 导入必要的库和装饰器
@add_start_docstrings(
    """UniSpeech Model with a vector-quantization module and ctc loss for pre-training.""", UNISPEECH_START_DOCSTRING
)
class UniSpeechForPreTraining(UniSpeechPreTrainedModel):
    def __init__(self, config: UniSpeechConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化 UniSpeechModel 实例
        self.unispeech = UniSpeechModel(config)
        # 初始化特征降维的 Dropout 层
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        # 初始化矢量量化器
        self.quantizer = UniSpeechGumbelVectorQuantizer(config)
        # 初始化量化后特征的线性映射层
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        # 初始化量化后特征的线性映射层
        self.project_hid = nn.Linear(config.proj_codevector_dim, config.hidden_size)

        # 初始化 CTC 分类器的线性层
        self.ctc_proj = nn.Linear(config.hidden_size, config.num_ctc_classes)
        # 初始化最终的 Dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 执行模型初始化后处理
        self.post_init()

    # 设置 Gumbel softmax 温度
    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    # 冻结特征提取器参数
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

    # 冻结特征提取器参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech.feature_extractor._freeze_parameters()

    # 计算对比损失的逻辑值
    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        # 将正样本和负样本特征连接起来
        target_features = torch.cat([target_features, negative_features], dim=0)

        # 计算预测特征与目标特征的余弦相似度
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1)
        logits = logits.type_as(target_features)

        # 应用温度
        logits = logits / temperature
        return logits
    # 前向传播函数，用于模型推理
    def forward(
        # 输入数值，类型为 torch.Tensor
        input_values: Optional[torch.Tensor],
        # 注意力的 mask，类型为 torch.Tensor，可选参数
        attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力信息，类型为 bool，可选参数
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为 bool，可选参数
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式结果，类型为 bool，可选参数
        return_dict: Optional[bool] = None,
# 使用 `add_start_docstrings` 装饰器为类添加文档字符串
# 创建 UniSpeechForCTC 类，是 UniSpeechPreTrainedModel 的子类
# config: UniSpeech 配置对象，target_lang: 目标语言的语言 id，可选参数
class UniSpeechForCTC(UniSpeechPreTrainedModel):
    # 初始化方法
    # config: UniSpeech 配置对象，target_lang: 目标语言的语言 id，可选参数
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类构造方法
        super().__init__(config)
        # 创建 UniSpeechModel 对象
        self.unispeech = UniSpeechModel(config)
        # 创建丢弃层对象
        self.dropout = nn.Dropout(config.final_dropout)
        # 设置目标语言
        self.target_lang = target_lang
        # 如果配置对象中未定义语言模型头的词汇量大小则抛出异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `UniSpeechForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        # 根据配置对象中是否定义添加适配器层标识来确定输出隐藏层大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 创建线性层，用于语言建模头
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        # 初始化权重并应用最终处理
        self.post_init()

    # 绑定权重的方法
    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """
        # 重写 `tie_weights` 方法，以便在传递 `target_lang=...` 给 `from_pretrained(...)` 时正确加载适配器权重

        # 获取目标语言
        target_lang = self.target_lang
        # 如果目标语言不为空且配置对象中适配器注意力维度未定义，则抛出异常
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果目标语言为空且配置对象中适配器注意力维度未定义则打印提示信息
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果目标语言不为空则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 冻结特征提取器，禁止在训练过程中更新特征编码器的参数
    def freeze_feature_extractor(self):
        # 发出警告，提示方法`freeze_feature_extractor`即将被弃用，建议使用相应的`freeze_feature_encoder`方法代替
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用`freeze_feature_encoder`方法
        self.freeze_feature_encoder()

    # 冻结特征编码器，禁止在训练过程中更新特征编码器的参数
    def freeze_feature_encoder(self):
        # 调用内部特征提取器对象的`_freeze_parameters`方法，禁止参数更新
        self.unispeech.feature_extractor._freeze_parameters()

    # 冻结基础模型，禁止在训练过程中更新基础模型的参数，只允许更新分类头部参数
    def freeze_base_model(self):
        # 遍历UniSpeech模型的参数，并将其`requires_grad`属性设置为False
        for param in self.unispeech.parameters():
            param.requires_grad = False

    # 为`forward`方法添加模型输入的文档字符串和示例代码的文档字符串
    @add_start_docstrings_to_model_forward(UNISPEECH_INPUTS_DOCSTRING)
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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional`):
            Connectionist Temporal Classification (CTC)的标签。`target_length` 必须小于等于输出logits的序列长度。
            索引在`[-100, 0, ..., config.vocab_size - 1]`范围内。所有值设为`-100`的标签将被忽略（掩码），
            损失只计算在`[0, ..., config.vocab_size - 1]`范围内的标签。
        """

        # 根据需要选择是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取unispeech模型的输出
        outputs = self.unispeech(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 对隐藏状态进行dropout处理
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 基于隐藏状态生成logits
        logits = self.lm_head(hidden_states)

        loss = None
        # 如果有标签
        if labels is not None:
            # 确保标签值在vocab_size范围内，否则引发异常
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 从attention_mask中获取输入长度
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充标记是-100，当不参与注意力时
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss不支持fp16，将logits转换为概率log_softmax
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                # 计算CTC损失
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        # 如果不需要字典形式的输出
        if not return_dict:
            # 返回输出
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典形式的输出
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 使用装饰器添加模型文档字符串
@add_start_docstrings(
    """
    UniSpeech Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """,
    UNISPEECH_START_DOCSTRING,
)
# 定义 UniSpeechForSequenceClassification 类，继承自 UniSpeechPreTrainedModel
class UniSpeechForSequenceClassification(UniSpeechPreTrainedModel):
    # 初始化函数，参数为配置对象 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置对象中存在 add_adapter 属性并且为 True，则抛出数值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of UniSpeech adapters (config.add_adapter=True)"
            )
        # 创建 UniSpeechModel 对象，传入配置对象 config
        self.unispeech = UniSpeechModel(config)
        # 计算隐藏层个数加一（transformer 层 + 输入嵌入层）
        num_layers = config.num_hidden_layers + 1
        # 如果配置对象中 use_weighted_layer_sum 属性为 True
        if config.use_weighted_layer_sum:
            # 初始化一个参数，用于计算加权层和
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建一个线性层，用于投影输出到分类器投影大小
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建一个线性层，用于分类
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义 freeze_feature_extractor 函数
    # 调用此函数将禁用特征编码器的梯度计算，以便在训练期间不会更新其参数
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

    # 定义 freeze_feature_encoder 函数
    # 调用此函数将禁用特征编码器的梯度计算，以便在训练期间不会更新其参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech.feature_extractor._freeze_parameters()

    # 定义 freeze_base_model 函数
    # 调用此函数将禁用基础模型的梯度计算，以便在训练期间不会更新其参数。只有分类头将被更新。
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.unispeech.parameters():
            param.requires_grad = False

    # 使用装饰器添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(UNISPEECH_INPUTS_DOCSTRING)
    # 使用装饰器添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从transformers.models.unispeech.modeling_unispeech.UniSpeechForSequenceClassification.forward中复制过来，修改了Wav2Vec2为UniSpeech，wav2vec2为unispeech
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入的语音特征张量，可选参数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量，可选参数，默认为None
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重张量，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否返回所有隐藏状态张量，可选参数，默认为None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出，可选参数，默认为None
        labels: Optional[torch.Tensor] = None,  # 用于计算序列分类/回归损失的标签张量，可选参数，默认为None
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            用于计算序列分类/回归损失的标签。索引应在`[0, ..., config.num_labels - 1]`之间。如果`config.num_labels == 1`，则计算回归损失（均方损失）；
            如果`config.num_labels > 1`，则计算分类损失（交叉熵损失）。
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果return_dict不为None，则使用指定值，否则使用模型配置中的设置
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states  # 如果模型配置中启用了加权层求和，则始终返回隐藏状态张量

        outputs = self.unispeech(  # 调用UniSpeech模型的forward方法
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:  # 如果模型配置中启用了加权层求和
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态张量列表
            hidden_states = torch.stack(hidden_states, dim=1)  # 在指定维度上堆叠隐藏状态张量
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行softmax归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 加权求和隐藏状态张量
        else:
            hidden_states = outputs[0]  # 获取隐藏状态张量

        hidden_states = self.projector(hidden_states)  # 使用投影层投影隐藏状态张量
        if attention_mask is None:  # 如果没有提供注意力遮罩
            pooled_output = hidden_states.mean(dim=1)  # 对隐藏状态进行平均池化
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)  # 获取特征向量的注意力遮罩
            hidden_states[~padding_mask] = 0.0  # 将非填充位置的隐藏状态置零
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)  # 对隐藏状态进行池化，考虑填充位置

        logits = self.classifier(pooled_output)  # 使用分类器获取logits

        loss = None  # 初始化损失值为None
        if labels is not None:  # 如果提供了标签
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))  # 计算损失值

        if not return_dict:  # 如果不要求以字典形式返回结果
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 组装输出元组
            return ((loss,) + output) if loss is not None else output  # 返回结果元组，包括损失值和其他输出
       
        # 以字典形式返回结果
        return SequenceClassifierOutput(
            loss=loss,  # 损失值
            logits=logits,  # logits
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力权重
        )
```