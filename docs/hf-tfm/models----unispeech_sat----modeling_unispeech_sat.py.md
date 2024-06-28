# `.\models\unispeech_sat\modeling_unispeech_sat.py`

```py
# 指定编码方式为 UTF-8，确保脚本可以正确处理 Unicode 字符
# 版权声明，版权归 Fairseq 作者和 HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证版本 2.0 许可使用本文件，除非符合许可证的规定，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于"原样"提供的，不提供任何明示或暗示的担保或条件
# 有关特定语言的详细信息，请参阅许可证
""" PyTorch UniSpeechSat model."""

import math  # 导入数学模块
import warnings  # 导入警告模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 检查点工具
from torch import nn  # 导入 PyTorch 神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数
from ...integrations.deepspeed import is_deepspeed_zero3_enabled  # 导入深度加速相关函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型类
from ...utils import (  # 导入实用函数
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_peft_available,
    logging,
    replace_return_docstrings,
)
from .configuration_unispeech_sat import UniSpeechSatConfig  # 导入 UniSpeechSat 配置类


logger = logging.get_logger(__name__)  # 获取记录器

_HIDDEN_STATES_START_POSITION = 2  # 隐藏状态的起始位置

# 通用文档字符串
_CONFIG_FOR_DOC = "UniSpeechSatConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "microsoft/unispeech-sat-base-100h-libri-ft"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC（连续文本转录）文档字符串
_CTC_EXPECTED_OUTPUT = "'MISTER QUILDER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 39.88

# 帧级分类文档字符串
_FRAME_CLASS_CHECKPOINT = "microsoft/unispeech-sat-base-plus-sd"
_FRAME_EXPECTED_OUTPUT = [0, 0]

# 说话人验证文档字符串
_XVECTOR_CHECKPOINT = "microsoft/unispeech-sat-base-plus-sv"
_XVECTOR_EXPECTED_OUTPUT = 0.97

UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # 查看所有 UniSpeechSat 模型的详细信息，请访问 https://huggingface.co/models?filter=unispeech_sat
]


@dataclass
class UniSpeechSatForPreTrainingOutput(ModelOutput):
    """
    [`UniSpeechSatForPreTrainingOutput`] 的输出类型，包括潜在的隐藏状态和注意力。
    """
    # 定义函数的参数和返回值的类型注解，以及可选的描述信息
    Args:
        loss (*optional*, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`):
            在训练模式下返回的总损失，包括对比损失（L_m）和多样性损失（L_d），参考官方论文[https://arxiv.org/pdf/2006.11477.pdf]中的分类损失。
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            模型隐藏状态投影到 `config.proj_codevector_dim` 维度，可用于预测掩码后的量化投影状态。
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            量化提取的特征向量投影到 `config.proj_codevector_dim` 维度，代表对比损失的正样本向量。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `torch.FloatTensor` 类型的张量（一个用于嵌入层输出，每层一个用于层输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            每层模型的隐藏状态以及初始嵌入层输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含 `torch.FloatTensor` 类型的张量（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力机制 softmax 后的注意力权重，用于计算自注意力头部的加权平均值。
    """

    # 定义可选的变量，用于存储不同类型的模型输出
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices

def _compute_mask_indices(
    shape: Tuple[int, int],                            # 定义函数 _compute_mask_indices 的参数 shape，是一个二元组，表示输入的形状
    mask_prob: float,                                  # 概率参数，确定要屏蔽的轴的百分比
    mask_length: int,                                  # 屏蔽长度
    attention_mask: Optional[torch.LongTensor] = None,  # 可选的注意力掩码，用于在每个批次维度上独立地缩短特征轴
    min_masks: int = 0,                                # 最小屏蔽数量
) -> np.ndarray:                                       # 返回一个 NumPy 数组，表示生成的屏蔽索引

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

    batch_size, sequence_length = shape               # 解包形状参数，分别得到批次大小和序列长度

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")  # 如果 mask_length 小于 1，抛出值错误异常

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )                                               # 如果 mask_length 大于 sequence_length，抛出值错误异常

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()                # 生成一个随机数作为 epsilon 用于概率舍入

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)  # 计算应屏蔽的 span 数量
        num_masked_span = max(num_masked_span, min_masks)  # 确保屏蔽的 span 数量不小于 min_masks

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length  # 确保屏蔽的 span 数量不超过 sequence_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)  # 确保屏蔽的 span 数量不超过 input_length - (mask_length - 1)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()      # 如果 attention_mask 存在，计算每个批次维度的特征轴的长度并转换为列表
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]  # 否则，默认为每个批次维度都是 sequence_length
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 创建一个全为 False 的布尔数组，形状为 (batch_size, sequence_length)
    spec_aug_mask_idxs = []                         # 初始化用于存储屏蔽索引的列表

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算序列长度中的最大屏蔽 span 数量
    # 如果最大被遮蔽跨度为0，则直接返回特定的遮蔽掩码
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 对每个输入长度进行循环处理
    for input_length in input_lengths:
        # 计算当前输入的被遮蔽跨度的数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要遮蔽的索引位置
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 如果没有选择任何索引，则说明输入长度严格小于序列长度，
        # 此时最后一个标记必须是填充标记，我们将其作为虚拟遮蔽标识符
        if len(spec_aug_mask_idx) == 0:
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟遮蔽标识符添加到遮蔽索引列表中，以保证所有批次具有相同的维度
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将遮蔽索引列表转换为NumPy数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮蔽索引扩展为遮蔽跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量以创建遮蔽跨度的起始索引
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不超过序列长度
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 在遮蔽掩码中根据索引设置遮蔽
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回生成的遮蔽掩码
    return spec_aug_mask
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer 复制而来，修改为 UniSpeechSatNoLayerNormConvLayer
class UniSpeechSatNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1  # 获取输入通道维度
        self.out_conv_dim = config.conv_dim[layer_id]  # 获取输出通道维度

        # 定义一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 卷积核大小
            stride=config.conv_stride[layer_id],      # 卷积步长
            bias=config.conv_bias,                    # 是否使用偏置
        )
        self.activation = ACT2FN[config.feat_extract_activation]  # 激活函数

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)  # 执行卷积操作
        hidden_states = self.activation(hidden_states)  # 应用激活函数
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer 复制而来，修改为 UniSpeechSatLayerNormConvLayer
class UniSpeechSatLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1  # 获取输入通道维度
        self.out_conv_dim = config.conv_dim[layer_id]  # 获取输出通道维度

        # 定义一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 卷积核大小
            stride=config.conv_stride[layer_id],      # 卷积步长
            bias=config.conv_bias,                    # 是否使用偏置
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)  # 层归一化
        self.activation = ACT2FN[config.feat_extract_activation]  # 激活函数

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)  # 执行卷积操作

        hidden_states = hidden_states.transpose(-2, -1)  # 转置操作
        hidden_states = self.layer_norm(hidden_states)  # 执行层归一化
        hidden_states = hidden_states.transpose(-2, -1)  # 转置操作

        hidden_states = self.activation(hidden_states)  # 应用激活函数
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer 复制而来，修改为 UniSpeechSatGroupNormConvLayer
class UniSpeechSatGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1  # 获取输入通道维度
        self.out_conv_dim = config.conv_dim[layer_id]  # 获取输出通道维度

        # 定义一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 卷积核大小
            stride=config.conv_stride[layer_id],      # 卷积步长
            bias=config.conv_bias,                    # 是否使用偏置
        )
        self.activation = ACT2FN[config.feat_extract_activation]  # 激活函数

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)  # 分组归一化

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)  # 执行卷积操作
        hidden_states = self.layer_norm(hidden_states)  # 执行分组归一化
        hidden_states = self.activation(hidden_states)  # 应用激活函数
        return hidden_states
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding 复制并修改为 UniSpeechSatPositionalConvEmbedding
class UniSpeechSatPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 创建一个一维卷积层，用于位置编码
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        
        # 设置权重归一化方法为 weight_norm
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        
        # 如果启用了 deepspeed zero3 加速，使用 gathered parameter 和 weight normalization
        if is_deepspeed_zero3_enabled():
            import deepspeed
            
            # 使用 deepspeed 的 gathered parameter 来管理权重
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 否则使用常规的 weight normalization
            self.conv = weight_norm(self.conv, name="weight", dim=2)
        
        # 创建一个用于填充的层，用于处理卷积后的输出
        self.padding = UniSpeechSatSamePadLayer(config.num_conv_pos_embeddings)
        
        # 激活函数由配置中的 feat_extract_activation 决定
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 调整输入张量的维度顺序以适应卷积层的要求
        hidden_states = hidden_states.transpose(1, 2)
        
        # 通过卷积层进行位置编码
        hidden_states = self.conv(hidden_states)
        
        # 使用填充层处理卷积后的张量
        hidden_states = self.padding(hidden_states)
        
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        
        # 调整输出张量的维度顺序，返回最终的隐藏状态张量
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer 复制并修改为 UniSpeechSatSamePadLayer
class UniSpeechSatSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        
        # 根据 num_conv_pos_embeddings 的奇偶性确定要移除的填充数目
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果需要移除填充，则截取相应长度的张量
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder 复制并修改为 UniSpeechSatFeatureEncoder
class UniSpeechSatFeatureEncoder(nn.Module):
    """从原始音频波形构建特征"""
    # 初始化函数，接受一个配置参数对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 根据配置参数选择不同的特征提取层规范化方式
        if config.feat_extract_norm == "group":
            # 如果配置为 "group"，创建一个包含 GroupNorm 的卷积层列表
            conv_layers = [UniSpeechSatGroupNormConvLayer(config, layer_id=0)] + [
                UniSpeechSatNoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果配置为 "layer"，创建一个包含 LayerNorm 的卷积层列表
            conv_layers = [
                UniSpeechSatLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 如果配置不是预期的 "group" 或 "layer"，抛出数值错误异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        
        # 将卷积层列表转换为模块列表，并存储在 self.conv_layers 中
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
        
        # 初始化需要梯度计算标志为 True
        self._requires_grad = True

    # 冻结模型参数的函数
    def _freeze_parameters(self):
        # 遍历模型的所有参数，并将其 requires_grad 属性设为 False
        for param in self.parameters():
            param.requires_grad = False
        
        # 将模型的自定义需要梯度计算标志设为 False
        self._requires_grad = False

    # 前向传播函数
    def forward(self, input_values):
        # 将输入值的维度扩展为 (batch_size, 1, ...)，用于卷积层的输入
        hidden_states = input_values[:, None]

        # 如果模型需要梯度计算且处于训练状态，确保 hidden_states 需要梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层，对 hidden_states 进行卷积操作
        for conv_layer in self.conv_layers:
            # 如果模型需要梯度计算且启用了梯度检查点功能且处于训练状态
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数来执行卷积操作
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,  # 调用卷积层的 __call__ 方法
                    hidden_states,  # 当前的 hidden_states
                )
            else:
                # 直接调用卷积层进行前向传播计算
                hidden_states = conv_layer(hidden_states)

        # 返回最终的隐藏状态结果
        return hidden_states
class UniSpeechSatFeatureExtractor(UniSpeechSatFeatureEncoder):
    # 继承自UniSpeechSatFeatureEncoder类，用于提取特征
    def __init__(self, config):
        # 调用父类构造函数初始化
        super().__init__(config)
        # 发出警告，提醒该类已被弃用，将在Transformers v5中移除，建议使用父类UniSpeechSatFeatureEncoder
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection with Wav2Vec2->UniSpeechSat
class UniSpeechSatFeatureProjection(nn.Module):
    # 用于特征投影的类，继承自nn.Module
    def __init__(self, config):
        # 初始化函数
        super().__init__()
        # LayerNorm层，用于归一化最后一个卷积维度的特征
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 线性映射层，将卷积维度映射到隐藏层维度
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # Dropout层，用于特征投影的dropout操作
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 对未投影的隐藏状态执行LayerNorm操作
        norm_hidden_states = self.layer_norm(hidden_states)
        # 执行特征投影，将归一化后的隐藏状态映射到隐藏层维度
        hidden_states = self.projection(norm_hidden_states)
        # 应用dropout操作
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->UniSpeechSat
class UniSpeechSatAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[UniSpeechSatConfig] = None,
    ):
        # 初始化函数
        super().__init__()
        # 设置注意力机制的维度和头数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 设置dropout概率
        self.dropout = dropout
        # 计算每个头的维度
        self.head_dim = embed_dim // num_heads
        # 存储配置
        self.config = config

        # 检查embed_dim必须能被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 设置缩放因子
        self.scaling = self.head_dim**-0.5
        # 设置是否为解码器注意力
        self.is_decoder = is_decoder
        # 设置是否为因果注意力
        self.is_causal = is_causal

        # 线性映射层，用于查询、键、值的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 输出映射层，用于最终输出的线性映射
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 对张量进行形状变换，用于多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->UniSpeechSat
class UniSpeechSatFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用给定的激活函数的 dropout
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 中间层的全连接层，输入维度是 hidden_size，输出维度是 intermediate_size
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 输出层的全连接层，输入维度是 intermediate_size，输出维度是 hidden_size
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 输出层的 dropout
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        # 中间层的全连接操作
        hidden_states = self.intermediate_dense(hidden_states)
        # 中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 中间层的 dropout
        hidden_states = self.intermediate_dropout(hidden_states)

        # 输出层的全连接操作
        hidden_states = self.output_dense(hidden_states)
        # 输出层的 dropout
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer with Wav2Vec2->UniSpeechSat
class UniSpeechSatEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # UniSpeechSatEncoderLayer 中使用的自定义注意力层
        self.attention = UniSpeechSatAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # Encoder 层的 dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # Layer normalization 层，输入维度是 hidden_size
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # FeedForward 层，使用 UniSpeechSatFeedForward 初始化
        self.feed_forward = UniSpeechSatFeedForward(config)
        # 最终的 layer normalization 层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 注意力层前的残差连接
        attn_residual = hidden_states
        # 调用注意力层的前向传播
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 对注意力输出进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 残差连接加上注意力输出
        hidden_states = attn_residual + hidden_states

        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)
        # 加上 FeedForward 层的输出
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终的 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AttnAdapterLayer with Wav2Vec2->UniSpeechSat
class UniSpeechSatAttnAdapterLayer(nn.Module):
    # 这里省略部分代码
    def __init__(self, config):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 从配置中获取适配器注意力维度和隐藏层大小
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size

        # 初始化层归一化层
        self.norm = nn.LayerNorm(self.hidden_dim)
        # 初始化线性层1，将隐藏状态映射到适配器注意力维度
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        # 初始化激活函数ReLU
        self.act_fn = nn.ReLU()
        # 初始化线性层2，将适配器注意力维度映射回隐藏层大小
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, hidden_states: torch.FloatTensor):
        # 应用层归一化到隐藏状态
        hidden_states = self.norm(hidden_states)

        # 应用线性层1
        hidden_states = self.linear_1(hidden_states)
        # 应用ReLU激活函数
        hidden_states = self.act_fn(hidden_states)
        # 应用线性层2
        hidden_states = self.linear_2(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayerStableLayerNorm 复制并修改为 UniSpeechSatEncoderLayerStableLayerNorm 类
class UniSpeechSatEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化注意力机制，使用 UniSpeechSatAttention 类
        self.attention = UniSpeechSatAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化层归一化层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化前馈神经网络层，使用 UniSpeechSatFeedForward 类
        self.feed_forward = UniSpeechSatFeedForward(config)
        # 初始化最终层归一化层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果配置中定义了 adapter_attn_dim 属性，则初始化适配器层
        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = UniSpeechSatAttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 复制隐藏状态以用于注意力残差连接
        attn_residual = hidden_states
        # 应用层归一化到隐藏状态
        hidden_states = self.layer_norm(hidden_states)
        # 使用注意力层计算新的隐藏状态、注意力权重，并可能返回注意力权重
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 应用 dropout 到隐藏状态
        hidden_states = self.dropout(hidden_states)
        # 添加注意力残差到新的隐藏状态
        hidden_states = attn_residual + hidden_states
        # 添加前馈神经网络到最终归一化的隐藏状态
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 如果存在适配器层，则将其应用到隐藏状态
        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将其添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出元组
        return outputs


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder 复制并修改为 UniSpeechSatEncoder 类
class UniSpeechSatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 存储配置
        self.config = config
        # 初始化位置卷积嵌入层，使用 UniSpeechSatPositionalConvEmbedding 类
        self.pos_conv_embed = UniSpeechSatPositionalConvEmbedding(config)
        # 初始化层归一化层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化编码器层列表，每层使用 UniSpeechSatEncoderLayer 类
        self.layers = nn.ModuleList([UniSpeechSatEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
            # 初始化隐藏状态和自注意力向量的存储，根据需要选择是否输出
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None

            # 如果存在注意力掩码，则确保填充的令牌输出为0
            if attention_mask is not None:
                expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
                hidden_states[~expand_attention_mask] = 0

                # 扩展注意力掩码以匹配模型输出维度
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

            # 检查是否启用了DeepSpeed zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

            # 遍历每一层
            for layer in self.layers:
                if output_hidden_states:
                    # 如果需要输出隐藏状态，则记录当前层的隐藏状态
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 应用LayerDrop技术（参见论文https://arxiv.org/abs/1909.11556）
                dropout_probability = torch.rand([])
                skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False

                # 如果不跳过当前层或者启用了DeepSpeed zero3
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 使用梯度检查点函数来计算当前层的输出（仅在训练时）
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

                # 如果跳过当前层，则输出设为None
                if skip_the_layer:
                    layer_outputs = (None, None)

                # 如果需要输出自注意力向量，则记录当前层的自注意力向量
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 如果需要输出隐藏状态，则记录最终的隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 根据return_dict的值返回模型输出
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderStableLayerNorm 复制而来，修改为 UniSpeechSatEncoderStableLayerNorm
class UniSpeechSatEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 初始化模型配置信息
        self.pos_conv_embed = UniSpeechSatPositionalConvEmbedding(config)  # 使用 UniSpeechSatPositionalConvEmbedding 初始化位置卷积嵌入层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 初始化层归一化层
        self.dropout = nn.Dropout(config.hidden_dropout)  # 初始化 dropout 层，用于随机失活
        # 使用 UniSpeechSatEncoderLayerStableLayerNorm 复制 config.num_hidden_layers 次，形成编码器层列表
        self.layers = nn.ModuleList(
            [UniSpeechSatEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False  # 初始化梯度检查点标志为 False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # 确保不对填充的 token 进行注意力计算
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # 扩展 attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 使用位置卷积嵌入层对隐藏状态进行处理
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # 检查是否启用了 DeepSpeed Zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 遍历每一层进行 Transformer 的前向传播
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop（参考 https://arxiv.org/abs/1909.11556）
            dropout_probability = torch.rand([])

            # 根据 LayerDrop 的概率决定是否跳过当前层
            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果启用了梯度检查点和处于训练模式，则使用梯度检查点技术来计算层的输出
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    # 否则直接调用层的 __call__ 方法计算输出
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            # 如果跳过当前层，则输出设为 None
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果需要输出自注意力机制的结果，则记录每层的注意力矩阵
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对最终的隐藏状态进行 LayerNorm 处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果需要输出所有隐藏状态，则记录最终的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据返回值的类型，返回不同的结果格式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 定义一个使用 Gumbel softmax 进行向量量化的类，详见[CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf)
class UniSpeechSatGumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See [CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, config):
        super().__init__()
        self.num_groups = config.num_codevector_groups  # 从配置中获取码矢组数
        self.num_vars = config.num_codevectors_per_group  # 从配置中获取每组的码矢数

        # 确保码矢的维度能够被码矢组数整除
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible by `config.num_codevector_groups`"
                f" {self.num_groups} for concatenation"
            )

        # 存储码矢变量（码字）的容器
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        # 权重投影层，将隐藏状态投影到码矢变量空间
        self.weight_proj = nn.Linear(config.hidden_size, self.num_groups * self.num_vars)

        # 训练中可以衰减的温度参数
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        # 计算概率分布的复杂度（perplexity）
        marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity
    def forward(self, hidden_states):
        # 获取输入张量的批大小、序列长度和隐藏大小
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到编码向量维度
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 在可微分的方式中使用 Gumbel Softmax 对隐藏状态进行采样，生成编码向量的概率分布
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # 计算 perplexity（复杂度指数）
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            # 在非可微分的方式中取最大值，生成硬编码向量分布（one-hot）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs)

        # 将编码向量的概率分布重新调整形状，用于检索编码向量
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率分布检索编码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        # 返回编码向量和 perplexity
        return codevectors, perplexity
````
# 定义 UniSpeechSat 预训练模型的抽象基类，用于处理权重初始化和下载/加载预训练模型的简单接口
class UniSpeechSatPreTrainedModel(PreTrainedModel):
    """
    """
    # 配置类，用于加载模型的配置信息
    config_class = UniSpeechSatConfig
    # 模型前缀，用于关键参数的命名
    base_model_prefix = "unispeech_sat"
    # 主要输入名称，用于模型输入的指定
    main_input_name = "input_values"
    # 支持梯度检查点的保存和恢复功能
    supports_gradient_checkpointing = True

    # 初始化权重的方法
    def _init_weights(self, module):
        """初始化模型权重"""
        # Gumbel Softmax 在初始化权重时需要特殊处理
        if isinstance(module, UniSpeechSatGumbelVectorQuantizer):
            # 初始化权重和偏置，确保分布满足模型需求
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            # 使用特定规则进行向量量化维的数量初始化
            nn.init.uniform_(module.codevectors)
        # 卷积嵌入位置组件的初始化
        elif isinstance(module, UniSpeechSatPositionalConvEmbedding):
            # 正态分布初始化卷积权重，确保权重满足模型结构要求
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 初始化偏置项为0
            nn.init.constant_(module.conv.bias, 0)
        # 特定特征处理的投影组件的初始化
        elif isinstance(module, UniSpeechSatFeatureProjection):
            # 分布均匀初始化权重，确保范围符合模型设计
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            # 同样初始化偏置项，采用均匀分布
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 计算softmax线性层的权重时需特定处理
        elif isinstance(module, nn.Linear):
            # 正态分布初始化权重，初始化范围可根据config参数自定义
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 偏置项设为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 对于标准化操作，初始化各项为0和1，分别对应偏置和权重的情况
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 对于一维卷积层，使用高斯分布进行权重初始化
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(
                # 1D卷积层的权重初始化通常使用kaiming方法，保证非线性组件的有效激活
                module.weight
            )
            # 初始化偏置项时，可根据特定规则（通常是与权重初始化一致或特殊化处理）进行
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # 计算卷积层输出长度的方法
    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        根据输入长度计算卷积层的输出长度
        """
        # 定义用于输出长度计算的函数，以下是关键公式的应用示例。
        def _conv_out_length(input_length, kernel_size, stride):
            # 计算1D卷积层输出长度的方法，通常用于响应某类特定结构（如神经网络）输入输出长度变换的需要。
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历模型配置中的核参数和步长，逐个计算同时考虑到多组参数时的输出长度。
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 返回最终的输出长度信息，这通常用于指导后续层的结构或绑定输入与输出长度关系的简化。
        return input_lengths
    # 计算每个样本非填充部分的长度，即每个样本中有效部分的长度
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

    # 根据非填充长度计算特征提取器的输出长度，转换为长整型
    output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)
    
    # 获取当前批次的大小
    batch_size = attention_mask.shape[0]

    # 初始化一个全零张量，用于构建注意力掩码
    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )

    # 将每个样本中特定索引位置设置为 1，确保在输出长度之前的所有值都被注意到
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
    
    # 反转张量并在最后一个维度上累加，然后再次反转，最终转换为布尔类型
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

    # 返回生成的注意力掩码张量
    return attention_mask
# UniSpeechSat 的开始文档字符串，提供了关于该模型的介绍和参考文献链接
UNISPEECH_SAT_START_DOCSTRING = r"""
    UniSpeechSat was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
    Auli.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`UniSpeechSatConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# UniSpeechSat 的输入文档字符串，通常包含在函数或方法的开头，描述了输入参数和期望的格式
UNISPEECH_SAT_INPUTS_DOCSTRING = r"""
    Describe the inputs of the UniSpeechSat model here, including their types and expected formats.
    This docstring should detail what inputs are required for the model to operate correctly.
    Include any specifics regarding input preprocessing or normalization if applicable.
"""
    # 定义函数的输入参数及其类型说明
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
    
            <Tip warning={true}>
    
            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [microsoft/unispeech-sat-base-100h-libri-ft](https://huggingface.co/microsoft/unispeech-sat-base-100h-libri-ft),
            `attention_mask` should **not** be passed to avoid degraded performance when doing batched inference. For
            such models `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware
            that these models also yield slightly different results depending on whether `input_values` is padded or
            not.
    
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
The bare UniSpeechSat Model transformer outputting raw hidden-states without any specific head on top.
Raw hidden-states指不带特定输出头部的UniSpeechSat模型的原始隐藏状态输出。

@add_start_docstrings(
    "The bare UniSpeechSat Model transformer outputting raw hidden-states without any specific head on top.",
    UNISPEECH_SAT_START_DOCSTRING,
)
用于为UniSpeechSat模型添加文档字符串装饰器，描述其作为一个原始隐藏状态输出的模型。

class UniSpeechSatModel(UniSpeechSatPreTrainedModel):
定义UniSpeechSat模型类，继承自UniSpeechSatPreTrainedModel。

    def __init__(self, config: UniSpeechSatConfig):
        初始化方法，接收UniSpeechSatConfig对象作为参数。

        super().__init__(config)
        调用父类的初始化方法，传入配置参数config。

        self.config = config
        将配置参数保存在self.config中。

        self.feature_extractor = UniSpeechSatFeatureEncoder(config)
        创建UniSpeechSatFeatureEncoder对象，用于特征提取。

        self.feature_projection = UniSpeechSatFeatureProjection(config)
        创建UniSpeechSatFeatureProjection对象，用于特征投影。

        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        创建形状为(config.hidden_size,)的可学习参数张量，用于掩码语谱图。

        if config.do_stable_layer_norm:
            根据配置参数判断是否启用稳定层归一化。
            self.encoder = UniSpeechSatEncoderStableLayerNorm(config)
            如果启用，创建UniSpeechSatEncoderStableLayerNorm对象作为编码器。
        else:
            否则，创建普通的UniSpeechSatEncoder对象作为编码器。
            self.encoder = UniSpeechSatEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()
        调用post_init方法，用于初始化权重和应用最终处理。

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states
    定义_mask_hidden_states方法，从wav2vec2模型中复制而来，用于掩码隐藏状态。

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        接收类型为torch.FloatTensor的隐藏状态张量作为输入参数。

        mask_time_indices: Optional[torch.FloatTensor] = None,
        可选参数，类型为torch.FloatTensor，用于掩码时间索引。

        attention_mask: Optional[torch.LongTensor] = None,
        可选参数，类型为torch.LongTensor，用于注意力掩码。
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
            # expand feature mask indices to match the dimensions of hidden_states
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states
        ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 如果未指定output_attentions参数，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取输入特征向量
        extract_features = self.feature_extractor(input_values)
        # 调整特征向量的维度顺序
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # 计算与特征向量对应的减少的attention_mask
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        # 特征投影
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 对隐藏状态进行掩码处理
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 编码器前向传播
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果不使用return_dict格式，则返回元组形式的结果
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 使用return_dict格式返回结果
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器添加文档字符串，描述 UniSpeechSat 模型带有量化器和顶部的 `VQ` 头
@add_start_docstrings("""UniSpeechSat Model with a quantizer and `VQ` head on top.""", UNISPEECH_SAT_START_DOCSTRING)
class UniSpeechSatForPreTraining(UniSpeechSatPreTrainedModel):
    def __init__(self, config: UniSpeechSatConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        
        # 初始化 UniSpeechSat 模型
        self.unispeech_sat = UniSpeechSatModel(config)
        
        # 创建用于特征量化的 dropout 层
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        # 初始化量化器，使用 Gumbel softmax 方法
        self.quantizer = UniSpeechSatGumbelVectorQuantizer(config)
        
        # 线性变换，将 codevector_dim 维度投影到 proj_codevector_dim 维度
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        
        # 线性变换，将 hidden_size 维度投影到 proj_codevector_dim 维度
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)

        # 初始化 dropout 层，用于最终处理
        self.dropout = nn.Dropout(config.final_dropout)

        # 线性变换，将 hidden_size 维度投影到 codevector_dim 维度，用于说话人投影
        self.speaker_proj = nn.Linear(config.hidden_size, config.codevector_dim)
        
        # 初始化标签嵌入参数，维度为 num_clusters x codevector_dim，并将其初始化为零
        self.label_embeddings_concat = nn.Parameter(torch.FloatTensor(config.num_clusters, config.codevector_dim))
        self.label_embeddings_concat.data.zero_()

        # 初始化 LayerNorm 层，用于特征提取器的输出，设定 epsilon 为 layer_norm_eps
        self.layer_norm_for_extract = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 如果设置了 do_stable_layer_norm，则固定 LayerNorm 的参数不更新
        if self.config.do_stable_layer_norm:
            self.layer_norm_for_extract.requires_grad = False

        # 调用初始化函数 post_init()，用于初始化权重和应用最终处理
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        设置 Gumbel softmax 温度为指定值。仅在训练时使用。
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其在训练期间不更新参数。
        """
        # 引发警告，表明此方法将在 Transformers v5 中移除，建议使用 freeze_feature_encoder 方法代替
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其在训练期间不更新参数。
        """
        # 冻结 wav2vec2 模型的特征提取器参数
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 1,
        ):
        """
        计算对比损失的逻辑 logits。输入目标特征、负样本特征和预测特征，以及温度参数。
        """
    ):
        """
        计算对比损失的对数概率，使用余弦相似度作为距离度量，比较 `[positive_feature, negative_features]` 和 `[predicted_features]`。
        此外，可以应用温度参数调节。

        Args:
            target_features (torch.Tensor): 包含正负样本特征的张量。
            negative_features (torch.Tensor): 负样本特征的张量。
            predicted_features (torch.Tensor): 预测特征的张量。
            temperature (float): 温度参数，用于调节对数概率的尺度。

        Returns:
            torch.Tensor: 经过温度调节后的对数概率张量。
        """
        
        # 将目标特征和负样本特征连接起来
        target_features = torch.cat([target_features, negative_features], dim=0)

        # 计算预测特征与目标特征的余弦相似度
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1)
        # 将 logits 转换为与目标特征相同的数据类型
        logits = logits.type_as(target_features)

        # 应用温度参数进行尺度调整
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=UniSpeechSatForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, UniSpeechSatForPreTrainingOutput]:
        r"""
        Returns:

        Example:

        ```
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, UniSpeechSatForPreTraining
        >>> from transformers.models.unispeech_sat.modeling_unispeech_sat import _compute_mask_indices

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base")
        >>> model = UniSpeechSatForPreTraining.from_pretrained("microsoft/unispeech-sat-base")
        >>> # TODO: Add full pretraining example
        ```"""

        # Determine if the function should return a dictionary format as specified by the `return_dict` parameter
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform UniSpeechSat model inference
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        transformer_features = outputs[0]  # Extract the transformer features from the model outputs

        # Quantize all extracted features (unmasked) and apply dropout
        extract_features = self.dropout_features(outputs[1])

        # Placeholder variables for future implementations
        logits = extract_features
        loss = quantized_features = codevector_perplexity = None

        # Below are commented-out sections which may be used for future logic implementation:
        # layer normalization (has no effect when `config.do_stable_layer_norm == False`)
        #        extract_features = self.layer_norm_for_extract(extract_features)
        #        quantized_features, codevector_perplexity = self.quantizer(extract_features)
        #
        # project quantized features twice
        #        quantized_features = self.project_q(quantized_features)
        #        quantized_features = self.project_hid(quantized_features)
        #
        #        loss = None
        #        logits = quantized_features

        # If return_dict is False, construct tuple output without loss
        if not return_dict:
            if loss is not None:
                return (loss, logits, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (logits, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        # If return_dict is True, construct UniSpeechSatForPreTrainingOutput object with specified attributes
        return UniSpeechSatForPreTrainingOutput(
            loss=loss,
            logits=logits,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """UniSpeechSat Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    UNISPEECH_SAT_START_DOCSTRING,
    """
        target_lang (`str`, *optional*):
            Language id of adapter weights. Adapter weights are stored in the format adapter.<lang>.safetensors or
            adapter.<lang>.bin. Only relevant when using an instance of [`UniSpeechSatForCTC`] with adapters. Uses
            'eng' by default.
    """,
)
# 定义了一个新的类 UniSpeechSatForCTC，继承自 UniSpeechSatPreTrainedModel
# 该类用于基于 CTC 的语言建模任务，结合了 UniSpeechSat 模型和一个线性输出层（lm_head）
class UniSpeechSatForCTC(UniSpeechSatPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 UniSpeechSat 模型和一个 dropout 层
        self.unispeech_sat = UniSpeechSatModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言（适配器权重的语言标识）
        self.target_lang = target_lang

        # 检查配置中是否定义了词汇表大小，如果未定义则抛出异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `UniSpeechSatForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        # 根据配置设置线性输出层的输入和输出大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并进行后续处理
        self.post_init()

    def tie_weights(self):
        """
        重写 [`~PreTrainedModel.tie_weights`] 方法，以便在 `from_pretrained(...)` 中传递 `target_lang=...` 时正确加载适配器权重。

        用户不应调用此方法，因为未来可能会更改。

        该方法通常用于绑定输入和输出嵌入权重。在这里，我们重新利用它来正确加载 UniSpeechSat 的适配器层，避免引入新的 `PreTrainedModel` API。
        虽然有点 hacky，但 UniSpeechSat 永远不必绑定输入和输出嵌入，因此在这里重新用这个函数是可以的。
        """
        
        # 获取目标语言
        target_lang = self.target_lang

        # 如果 target_lang 不为 None，且配置中未定义 adapter_attn_dim，则抛出 ValueError
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果 target_lang 为 None，且配置中定义了 adapter_attn_dim，则记录警告信息
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果 target_lang 不为 None，则加载相应的适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_extractor(self):
        # 发出警告，提示该函数将在 Transformers v5 中删除，建议使用等效的 freeze_feature_encoder 方法
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 freeze_feature_encoder 方法来冻结特征编码器的参数
        self.freeze_feature_encoder()

    # 调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_encoder(self):
        # 调用内部的 _freeze_parameters 方法来冻结特征编码器的参数
        self.unispeech_sat.feature_extractor._freeze_parameters()

    # 调用此函数将禁用基础模型的梯度计算，使其参数在训练期间不会更新，只有分类头部会被更新
    def freeze_base_model(self):
        # 遍历 unispeech_sat 模型的所有参数，并将其 requires_grad 属性设为 False，从而禁用梯度计算
        for param in self.unispeech_sat.parameters():
            param.requires_grad = False

    # 将该函数装饰为模型的前向传播方法，并添加相应的文档字符串注释
    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
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

        # Determine if return_dict is provided; if not, use the model's default setting
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input_values through unispeech_sat model, with optional outputs controlled by parameters
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Retrieve the hidden states from the outputs and apply dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Compute logits from the language model head based on the processed hidden states
        logits = self.lm_head(hidden_states)

        # Initialize loss as None
        loss = None
        if labels is not None:
            # Check if any label index exceeds the vocabulary size; raise ValueError if so
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # Calculate input_lengths based on attention_mask if provided; otherwise assume all inputs are attended
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # Create a mask to ignore padded tokens and compute target_lengths
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # Apply log softmax to logits and transpose for CTC loss calculation
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # Disable cudnn optimization flags for CTC loss calculation
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

        # If return_dict is False, prepare output as a tuple of logits and optional hidden states
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, return CausalLMOutput object containing loss, logits, hidden states, and attentions
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 用于处理UniSpeechSat模型的序列分类任务的模型定义，添加了一个线性层在池化输出之上作为分类头部。

@add_start_docstrings(
    """
    UniSpeechSat Model with a sequence classification head on top (a linear layer over the pooled output) for tasks
    like SUPERB Keyword Spotting.
    """,
    UNISPEECH_SAT_START_DOCSTRING,
)
class UniSpeechSatForSequenceClassification(UniSpeechSatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 检查配置是否支持适配器，如果支持则引发错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of UniSpeechSat adapters (config.add_adapter=True)"
            )
        
        # 初始化UniSpeechSat模型
        self.unispeech_sat = UniSpeechSatModel(config)
        
        # 确定层数（变压器层 + 输入嵌入）
        num_layers = config.num_hidden_layers + 1
        
        # 如果配置使用加权层求和，则初始化层权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 定义投影层，将隐藏状态投影到分类器投影尺寸
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        
        # 定义分类器层，将投影后的特征映射到类别数量
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification中复制过来的方法，冻结特征提取器
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

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification中复制过来的方法，冻结特征编码器
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech_sat.feature_extractor._freeze_parameters()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification中复制过来的方法，冻结基础模型
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.unispeech_sat.parameters():
            param.requires_grad = False

    # 根据UNISPEECH_SAT_INPUTS_DOCSTRING和其他参数添加文档字符串和代码示例文档字符串
    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从 transformers.models.unispeech_sat.modeling_unispeech_sat.UniSpeechSat.forward 复制而来，将 Wav2Vec2 修改为 UniSpeechSat，将 wav2vec2 修改为 unispeech_sat
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
    
        # 确定是否返回字典格式的输出，默认为 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据 self.config.use_weighted_layer_sum 确定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
    
        # 将输入 input_values 传递给 UniSpeechSat 模型进行前向传播
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 如果使用加权层求和，则计算加权隐藏状态
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态起始位置的输出
            hidden_states = torch.stack(hidden_states, dim=1)  # 在第1维度上堆叠隐藏状态
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行 softmax 归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 加权求和隐藏状态
        else:
            hidden_states = outputs[0]  # 否则直接使用第一个输出作为隐藏状态
    
        hidden_states = self.projector(hidden_states)  # 投影隐藏状态
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)  # 如果没有注意力掩码，则对隐藏状态进行均值池化
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0  # 使用特征向量注意力掩码设置隐藏状态为0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)  # 汇总隐藏状态
    
        logits = self.classifier(pooled_output)  # 使用分类器生成 logits
    
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))  # 计算交叉熵损失
    
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 如果不返回字典，则输出 logits 和隐藏状态起始位置后的输出
            return ((loss,) + output) if loss is not None else output
    
        # 返回 SequenceClassifierOutput 对象，包括 loss、logits、隐藏状态和注意力
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    UniSpeech-SAT Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    UNISPEECH_SAT_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification 复制过来，将 Wav2Vec2 替换为 UniSpeechSat，将 wav2vec2 替换为 unispeech_sat，将 WAV_2_VEC_2 替换为 UNISPEECH_SAT
class UniSpeechSatForAudioFrameClassification(UniSpeechSatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中存在 add_adapter 属性且为 True，则抛出 ValueError
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of UniSpeechSat adapters (config.add_adapter=True)"
            )
        
        # 初始化 UniSpeechSatModel，并将其赋值给 self.unispeech_sat
        self.unispeech_sat = UniSpeechSatModel(config)
        
        # 计算 transformer 层数加上输入嵌入层的总数
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        
        # 如果配置中使用加权层求和，则初始化权重参数为均匀分布
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 初始化分类器线性层，输入大小为 config.hidden_size，输出大小为 config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 将 config.num_labels 赋值给 self.num_labels
        self.num_labels = config.num_labels

        # 初始化模型权重
        self.init_weights()

    # 将 freeze_feature_extractor 方法标记为过时，未来将在 Transformers v5 中移除，请改用 freeze_feature_encoder 方法
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

    # 禁用特征编码器的梯度计算，使其在训练过程中不会更新参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech_sat.feature_extractor._freeze_parameters()

    # 禁用基础模型的梯度计算，使其在训练过程中不会更新参数，只会更新分类头
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.unispeech_sat.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_FRAME_CLASS_CHECKPOINT,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_FRAME_EXPECTED_OUTPUT,
    )
    # 前向传播函数，接受输入值 input_values、注意力掩码 attention_mask、标签 labels，输出是否返回注意力、隐藏状态、返回字典等
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # 如果输入值不为空，则调用 UniSpeechSatModel 的前向传播函数，传入相应参数
        self.unispeech_sat
        #
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 确定是否返回字典格式的输出结果，如果未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏状态，如果使用加权层求和，则输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用UniSpeech模型，传入输入值和其他参数，并返回输出结果
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置中使用加权层求和，则计算加权后的隐藏状态表示
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 提取隐藏状态的起始位置
            hidden_states = torch.stack(hidden_states, dim=1)  # 在指定维度上堆叠张量
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行softmax归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 加权求和隐藏状态表示
        else:
            hidden_states = outputs[0]  # 否则直接使用第一个输出作为隐藏状态表示

        logits = self.classifier(hidden_states)  # 使用分类器对隐藏状态进行分类

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            # 计算损失，将logits展平为(batch_size, num_labels)，并计算预测标签的交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 构建输出元组
            return output

        # 返回TokenClassifierOutput对象，包括损失、logits、隐藏状态和注意力分布等输出结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss
# AMSoftmaxLoss 类定义，用于实现 AM Softmax 损失函数
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        # 定义可学习的权重参数，用于计算损失
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        # 使用交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        # 对权重参数进行归一化处理
        weight = nn.functional.normalize(self.weight, dim=0)
        # 对输入的隐藏状态进行归一化处理
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        # 计算余弦相似度
        cos_theta = torch.mm(hidden_states, weight)
        # 计算 AM Softmax 损失中的 psi 值
        psi = cos_theta - self.margin

        # 将标签转换为 one-hot 编码
        onehot = nn.functional.one_hot(labels, self.num_labels)
        # 计算最终的预测 logits
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        # 计算损失值
        loss = self.loss(logits, labels)

        return loss


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer
# TDNNLayer 类定义，用于实现时间延迟神经网络层
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 初始化 TDNN 层的参数
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        # 定义线性层作为卷积核
        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        # 定义激活函数为 ReLU
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 检查是否有 peft 可用，如果有，则导入 LoraLayer
        if is_peft_available():
            from peft.tuners.lora import LoraLayer

            # 如果卷积核是 LoraLayer 类型，则发出警告，因为不会应用 LoRA 权重
            if isinstance(self.kernel, LoraLayer):
                warnings.warn(
                    "Detected LoRA on TDNNLayer. LoRA weights won't be applied due to optimization. "
                    "You should exclude TDNNLayer from LoRA's target modules.",
                )

        # 将输入的隐藏状态转置，为了与 conv1d 函数兼容
        hidden_states = hidden_states.transpose(1, 2)
        # 将线性层的权重重塑为卷积核的形状，并进行转置
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        # 使用 conv1d 函数进行卷积操作
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        # 再次将隐藏状态转置回原来的形状
        hidden_states = hidden_states.transpose(1, 2)

        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    UniSpeech-SAT Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    UNISPEECH_SAT_START_DOCSTRING,
)
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector with Wav2Vec2->UniSpeechSat, wav2vec2->unispeech_sat, WAV_2_VEC_2->UNISPEECH_SAT
# UniSpeechSatForXVector 类定义，基于 UniSpeech-SAT 模型，添加了 XVector 特征提取头部，用于说话人验证等任务
class UniSpeechSatForXVector(UniSpeechSatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.unispeech_sat = UniSpeechSatModel(config)  # 初始化UniSpeechSatModel模型
        num_layers = config.num_hidden_layers + 1  # 计算层数，包括transformer层和输入嵌入层
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)  # 如果配置使用加权层求和，则初始化权重参数
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])  # 初始化线性投影层

        # 创建TDNN层列表
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)  # 将TDNN层列表封装成ModuleList

        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)  # 初始化特征提取器的线性层
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)  # 初始化分类器的线性层

        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)  # 初始化AMSoftmax损失函数

        self.init_weights()  # 调用初始化权重方法

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
        self.freeze_feature_encoder()  # 调用freeze_feature_encoder方法冻结特征编码器的参数更新

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech_sat.feature_extractor._freeze_parameters()  # 冻结特征提取器的参数更新

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.unispeech_sat.parameters():  # 遍历UniSpeechSatModel的所有参数
            param.requires_grad = False  # 将参数的梯度计算设为False，不更新这些参数的梯度

    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1  # 计算1D卷积层的输出长度公式

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)  # 循环计算TDNN层的输出长度

        return input_lengths

    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_XVECTOR_CHECKPOINT,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_XVECTOR_EXPECTED_OUTPUT,
    )
    # 定义前向传播方法，用于模型推断阶段
    def forward(
        self,
        # 输入值，类型为可选的 PyTorch 张量
        input_values: Optional[torch.Tensor],
        # 注意力掩码，类型为可选的 PyTorch 张量，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，类型为可选的布尔值，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典类型的输出，类型为可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,
        # 标签，类型为可选的 PyTorch 张量，默认为 None
        labels: Optional[torch.Tensor] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化返回字典，如果未提供则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用unispeech_sat模型，传入输入值和其他参数，获取输出
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置中指定使用加权层求和，则计算加权后的隐藏状态
        if self.config.use_weighted_layer_sum:
            # 从输出中提取隐藏状态的起始位置索引
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 将所有隐藏状态堆叠在一起
            hidden_states = torch.stack(hidden_states, dim=1)
            # 计算层权重的softmax值
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 使用权重加权求和隐藏状态
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用输出的第一个元素作为隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态投影到新的空间
        hidden_states = self.projector(hidden_states)

        # 对每个TDNN层进行循环处理隐藏状态
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # 统计池化操作
        if attention_mask is None:
            # 如果没有注意力掩码，则对整个隐藏状态进行平均和标准差计算
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            # 否则，根据注意力掩码计算特征提取的输出长度
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            # 对每个TDNN层的输出长度进行循环处理
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            # 将计算得到的均值和标准差特征堆叠起来
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        # 将均值和标准差特征拼接在一起作为统计池化结果
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        # 使用特征提取器对统计池化结果进行处理
        output_embeddings = self.feature_extractor(statistic_pooling)
        # 使用分类器对处理后的特征进行分类得到logits
        logits = self.classifier(output_embeddings)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            loss = self.objective(logits, labels)

        # 如果不需要返回字典，则直接返回输出元组
        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则创建XVectorOutput对象并返回
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```