# `.\models\unispeech\modeling_unispeech.py`

```py
# 设置文件编码为UTF-8
# 版权声明
# 2021年由Fairseq作者和HuggingFace团队保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）许可;
# 除非符合许可证，否则不得使用此文件。
# 您可以获取许可证的副本，详见
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何形式的担保或条件，明示或暗示。
# 有关更多详细信息，请参阅许可证。
""" PyTorch UniSpeech模型。"""

import math  # 导入数学模块
import warnings  # 导入警告模块
from dataclasses import dataclass  # 导入dataclass用于数据类
from typing import Optional, Tuple, Union  # 导入类型提示相关

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint模块
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数
from ...integrations.deepspeed import is_deepspeed_zero3_enabled  # 导入DeepSpeed集成函数
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型工具类
from ...utils import (  # 导入通用实用程序功能
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_unispeech import UniSpeechConfig  # 导入UniSpeech模型配置

logger = logging.get_logger(__name__)  # 获取日志记录器


_HIDDEN_STATES_START_POSITION = 2  # 隐藏状态的起始位置索引

# 通用文档字符串
_CONFIG_FOR_DOC = "UniSpeechConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "patrickvonplaten/unispeech-large-1500h-cv-timit"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 1024]

# CTC（连接时序分类）文档字符串
_CTC_EXPECTED_OUTPUT = "'mister quilter is the apposl of the midle classes and weare glad to welcom his gosepl'"
_CTC_EXPECTED_LOSS = 17.17

UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/unispeech-large-1500h-cv",
    "microsoft/unispeech-large-multi-lingual-1500h-cv",
    # 查看所有UniSpeech模型：https://huggingface.co/models?filter=unispeech
]


@dataclass
class UniSpeechForPreTrainingOutput(ModelOutput):
    """
    [`UniSpeechForPreTrainingOutput`]的输出类型，包含潜在的隐藏状态和注意力。
    """
    # 定义函数的参数及其类型注解
    Args:
        loss (*optional*, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`):
            训练模式下返回的损失，是对比损失（L_m）和多样性损失（L_d）的总和，详见官方论文[https://arxiv.org/pdf/2006.11477.pdf]。
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            模型隐藏状态投影到 `config.proj_codevector_dim` 维度后的结果，可以用来预测掩码后的量化投影状态。
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            量化提取的特征向量投影到 `config.proj_codevector_dim` 维度后的结果，代表对比损失的正向目标向量。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含模型每层输出的隐藏状态的元组，每个元素为 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`。
            在 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
    
            包括每层输出以及初始嵌入输出的隐藏状态。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含自注意力每层注意力权重的元组，每个元素为 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
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
    batch_size, sequence_length = shape  # 解包形状元组为批次大小和序列长度

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")  # 抛出值错误异常，如果掩码长度小于1

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )  # 抛出值错误异常，如果掩码长度大于序列长度

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()  # 生成一个随机浮点数作为 epsilon，用于概率舍入

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)  # 计算应该掩盖的跨度数量
        num_masked_span = max(num_masked_span, min_masks)  # 取较大值，确保掩盖的跨度数量不小于最小要求

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length  # 确保掩盖的跨度数量不超过序列长度

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)  # 确保掩盖的跨度数量不超过输入长度减去 (掩码长度 - 1)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )  # 计算批次中每个序列的输入长度，如果有注意力掩码则使用其求和，否则使用序列长度

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 创建一个布尔类型的全零数组，用于存储 SpecAugment 掩码
    spec_aug_mask_idxs = []  # 创建一个空列表，用于存储 SpecAugment 掩码的索引

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算应该掩盖的最大跨度数量
    # 如果最大的被屏蔽段数为0，则直接返回原始的spec_aug_mask
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算当前输入的被屏蔽段数
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要屏蔽的索引位置
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个样本索引作为填充向量的虚拟索引，确保所有批次的维度相同，由于概率性舍入
        # 选择第一个样本只是为了使那些向量填充两次。
        if len(spec_aug_mask_idx) == 0:
            # 如果长度为0，表示input_length严格小于sequence_length，最后一个标记应该是填充标记，我们可以使用它作为虚拟屏蔽id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将dummy_mask_idx重复填充到数组，以保证数组长度为max_num_masked_span
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将列表转换为numpy数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将屏蔽索引扩展为屏蔽段
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    # 将多维数组展平为一维数组
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量以创建屏蔽段的起始索引
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不超过sequence_length - 1
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将屏蔽标记散布到数组中的索引位置
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回生成的spec_aug_mask
    return spec_aug_mask
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->UniSpeech
class UniSpeechNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 从配置中获取输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，指定输入、输出维度、卷积核大小、步长和是否有偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行一维卷积
        hidden_states = self.conv(hidden_states)
        # 应用预先选择的激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->UniSpeech
class UniSpeechLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 从配置中获取输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，指定输入、输出维度、卷积核大小、步长和是否有偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一个层归一化层，归一化输出特征向量并保留可学习的仿射变换
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行一维卷积
        hidden_states = self.conv(hidden_states)

        # 将卷积输出的维度转置，为了适应层归一化的输入要求
        hidden_states = hidden_states.transpose(-2, -1)
        # 应用层归一化到转置后的隐藏状态
        hidden_states = self.layer_norm(hidden_states)
        # 再次转置以恢复原始维度
        hidden_states = hidden_states.transpose(-2, -1)

        # 应用预先选择的激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->UniSpeech
class UniSpeechGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 从配置中获取输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，指定输入、输出维度、卷积核大小、步长和是否有偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个分组归一化层，根据输出卷积维度进行分组归一化
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行一维卷积
        hidden_states = self.conv(hidden_states)
        # 应用分组归一化到卷积输出
        hidden_states = self.layer_norm(hidden_states)
        # 应用预先选择的激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding 复制代码，并将 Wav2Vec2 替换为 UniSpeech
class UniSpeechPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个 1 维卷积层，用于位置编码
        self.conv = nn.Conv1d(
            config.hidden_size,  # 输入通道数和输出通道数都是 hidden_size
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,  # 卷积核大小为 num_conv_pos_embeddings
            padding=config.num_conv_pos_embeddings // 2,  # 填充大小为卷积核大小的一半
            groups=config.num_conv_pos_embedding_groups,  # 分组卷积的组数
        )

        # 使用权重归一化操作，如果支持深度加速的 zero3 功能，则进行相关处理
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        if is_deepspeed_zero3_enabled():
            import deepspeed

            # 使用 deepspeed.zero.GatheredParameters 确保权重的全局收集和归一化处理
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 否则，直接对卷积层的权重进行归一化处理
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建用于对输入进行填充的 UniSpeechSamePadLayer 层
        self.padding = UniSpeechSamePadLayer(config.num_conv_pos_embeddings)
        # 选择激活函数，根据 config 中的 feat_extract_activation 选择对应的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将输入张量进行维度转换，调整为 Conv1d 的输入格式
        hidden_states = hidden_states.transpose(1, 2)

        # 经过卷积层处理
        hidden_states = self.conv(hidden_states)
        # 经过填充层处理
        hidden_states = self.padding(hidden_states)
        # 经过激活函数处理
        hidden_states = self.activation(hidden_states)

        # 最后再次进行维度转换，调整回原始输入的格式
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer 复制代码，并将 Wav2Vec2 替换为 UniSpeech
class UniSpeechSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 根据 num_conv_pos_embeddings 的奇偶性决定是否移除最后一列填充
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果需要移除填充，则进行相应的操作
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder 复制代码，并将 Wav2Vec2 替换为 UniSpeech
class UniSpeechFeatureEncoder(nn.Module):
    """从原始音频波形构造特征"""

    # 该类尚未实现，只是提供了一个文档字符串说明其作用
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 根据配置中的特征提取规范选择不同的卷积层列表
        if config.feat_extract_norm == "group":
            # 如果特征提取规范是"group"，则创建使用组归一化的卷积层列表
            conv_layers = [UniSpeechGroupNormConvLayer(config, layer_id=0)] + [
                UniSpeechNoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果特征提取规范是"layer"，则创建使用层归一化的卷积层列表
            conv_layers = [
                UniSpeechLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 如果特征提取规范既不是"group"也不是"layer"，则抛出值错误异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        
        # 将卷积层列表转换为nn.ModuleList并赋值给对象的conv_layers属性
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # 设置梯度检查点标志为False
        self.gradient_checkpointing = False
        
        # 设置_requires_grad属性为True
        self._requires_grad = True

    # 冻结参数的私有方法
    def _freeze_parameters(self):
        # 遍历所有参数，并将其requires_grad属性设为False
        for param in self.parameters():
            param.requires_grad = False
        
        # 将对象的_requires_grad属性设为False
        self._requires_grad = False

    # 前向传播函数，接受输入值作为参数
    def forward(self, input_values):
        # 将输入值的维度扩展为二维，保留第一维
        hidden_states = input_values[:, None]

        # 如果_requires_grad为True并且当前处于训练模式
        if self._requires_grad and self.training:
            # 设置hidden_states需要计算梯度，以便于梯度检查点
            hidden_states.requires_grad = True

        # 遍历所有卷积层进行前向传播
        for conv_layer in self.conv_layers:
            # 如果_requires_grad为True、gradient_checkpointing为True，并且当前处于训练模式
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数对当前卷积层进行前向传播
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则，直接对当前卷积层进行前向传播
                hidden_states = conv_layer(hidden_states)

        # 返回最终的隐藏状态
        return hidden_states
class UniSpeechFeatureExtractor(UniSpeechFeatureEncoder):
    # UniSpeechFeatureExtractor 类继承自 UniSpeechFeatureEncoder 类
    def __init__(self, config):
        # 初始化函数，调用父类 UniSpeechFeatureEncoder 的初始化方法
        super().__init__(config)
        # 发出警告信息，提示该类已被弃用，并在 Transformers v5 中将被移除，建议使用其父类名称替代
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection 复制而来，将 Wav2Vec2 替换为 UniSpeech
class UniSpeechFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # LayerNorm 对象，对最后一个卷积维度进行归一化，使用的 epsilon 为 config 中的 layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 线性映射，将最后一个卷积维度映射到隐藏大小，config.hidden_size 为隐藏层大小
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 随机失活层，使用的丢弃率为 config 中的 feat_proj_dropout
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 对隐藏状态进行 LayerNorm 归一化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态进行线性映射投影到隐藏大小
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态进行随机失活
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# 从 transformers.models.bart.modeling_bart.BartAttention 复制而来，将 Bart 替换为 UniSpeech
class UniSpeechAttention(nn.Module):
    """来自 'Attention Is All You Need' 论文的多头注意力"""

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
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，根据头维度进行初始化
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性映射对象，用于处理 key、value、query 和输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 对张量进行形状变换，用于注意力计算
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
        # 前向传播函数，执行注意力计算和映射
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->UniSpeech
class UniSpeechFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义中间层的 Dropout 操作，使用配置中的激活函数的 dropout 率
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 定义中间层的全连接层，输入大小为隐藏大小，输出大小为中间大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置中的激活函数类型选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 定义输出层的全连接层，输入大小为中间大小，输出大小为隐藏大小
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义输出层的 Dropout 操作，使用配置中的隐藏层 dropout 率
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        # 中间层的全连接操作
        hidden_states = self.intermediate_dense(hidden_states)
        # 应用中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 应用中间层的 Dropout 操作
        hidden_states = self.intermediate_dropout(hidden_states)

        # 输出层的全连接操作
        hidden_states = self.output_dense(hidden_states)
        # 应用输出层的 Dropout 操作
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer with Wav2Vec2->UniSpeech
class UniSpeechEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义自注意力层，使用 UniSpeechAttention 模块，设置 embed_dim 和 num_heads，关闭解码器模式
        self.attention = UniSpeechAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 定义 Dropout 操作
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 定义层归一化操作，设置输入大小为隐藏大小，epsilon 为配置中的 layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义前馈网络层，使用 UniSpeechFeedForward 模块
        self.feed_forward = UniSpeechFeedForward(config)
        # 定义最终的层归一化操作，设置输入大小为隐藏大小，epsilon 为配置中的 layer_norm_eps
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 复制注意力前的隐藏状态
        attn_residual = hidden_states
        # 进行自注意力计算，获取输出隐藏状态、注意力权重和其他信息（根据输出_attentions 参数）
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 应用 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 添加自注意力前的隐藏状态，形成残差连接
        hidden_states = attn_residual + hidden_states

        # 应用层归一化操作
        hidden_states = self.layer_norm(hidden_states)
        # 应用前馈网络层
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终的层归一化操作
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AttnAdapterLayer with Wav2Vec2->UniSpeech
class UniSpeechAttnAdapterLayer(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置输入维度为配置对象中的适配器注意力维度
        self.input_dim = config.adapter_attn_dim
        # 设置隐藏维度为配置对象中的隐藏大小
        self.hidden_dim = config.hidden_size

        # 初始化 LayerNorm 层，标准化隐藏状态
        self.norm = nn.LayerNorm(self.hidden_dim)
        # 第一个线性层，将隐藏状态映射到适配器注意力维度
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        # 激活函数 ReLU
        self.act_fn = nn.ReLU()
        # 第二个线性层，将适配器注意力维度映射回隐藏状态维度
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    # 前向传播方法，接收一个形状为 [batch_size, seq_length, hidden_size] 的张量作为输入
    def forward(self, hidden_states: torch.FloatTensor):
        # 对输入的隐藏状态进行 LayerNorm 处理
        hidden_states = self.norm(hidden_states)

        # 经过第一个线性层的变换
        hidden_states = self.linear_1(hidden_states)
        # 应用 ReLU 激活函数
        hidden_states = self.act_fn(hidden_states)
        # 经过第二个线性层的变换
        hidden_states = self.linear_2(hidden_states)

        # 返回变换后的隐藏状态张量
        return hidden_states
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayerStableLayerNorm 复制并修改为 UniSpeechEncoderLayerStableLayerNorm
class UniSpeechEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力模块 UniSpeechAttention
        self.attention = UniSpeechAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 初始化 Dropout 模块
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化层归一化模块
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化前馈神经网络模块 UniSpeechFeedForward
        self.feed_forward = UniSpeechFeedForward(config)
        # 初始化最终层归一化模块
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果配置中存在适配器注意力维度，则初始化适配器层 UniSpeechAttnAdapterLayer；否则设为 None
        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = UniSpeechAttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 保留注意力残差连接
        attn_residual = hidden_states
        # 应用层归一化到隐藏状态
        hidden_states = self.layer_norm(hidden_states)
        # 使用自注意力模块进行注意力计算，并返回注意力权重（如果设置输出注意力的话）
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 应用 Dropout 到注意力输出上
        hidden_states = self.dropout(hidden_states)
        # 添加注意力残差到处理后的隐藏状态上
        hidden_states = attn_residual + hidden_states
        # 应用前馈神经网络，并在最终层归一化后加到处理后的隐藏状态上
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 如果适配器层不为 None，则将适配器层应用到处理后的隐藏状态上
        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        # 返回处理后的隐藏状态，可能包含注意力权重（取决于输出设置）
        outputs = (hidden_states,)

        # 如果设置输出注意力，将注意力权重添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder 复制并修改为 UniSpeechEncoder
class UniSpeechEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化位置卷积嵌入模块 UniSpeechPositionalConvEmbedding
        self.pos_conv_embed = UniSpeechPositionalConvEmbedding(config)
        # 初始化层归一化模块
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 模块
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化编码器层模块列表，数量为配置中指定的隐藏层数
        self.layers = nn.ModuleList([UniSpeechEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志设为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None

            if attention_mask is not None:
                # 确保填充的标记输出为0
                expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
                hidden_states[~expand_attention_mask] = 0

                # 扩展 attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

            # 通过位置卷积嵌入层处理位置嵌入
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 添加 LayerDrop（参见 https://arxiv.org/abs/1909.11556 进行描述）
                dropout_probability = torch.rand([])

                skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 在深度速度（deepspeed）zero3下，所有GPU必须同步运行
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

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderStableLayerNorm 复制而来，将 Wav2Vec2 替换为 UniSpeech
class UniSpeechEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化 UniSpeechPositionalConvEmbedding 对象，用于位置编码和卷积嵌入
        self.pos_conv_embed = UniSpeechPositionalConvEmbedding(config)
        # 初始化 LayerNorm 层，用于层归一化，eps 参数为配置中的层归一化 epsilon 值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，用于随机失活，丢弃率为配置中的隐藏层丢弃率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 使用列表推导式创建多个 UniSpeechEncoderLayerStableLayerNorm 层，层数为配置中的隐藏层数量
        self.layers = nn.ModuleList(
            [UniSpeechEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        # 是否启用梯度检查点，默认为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    # 如果设置了输出所有隐藏状态，则初始化一个空元组；否则设为 None
    all_hidden_states = () if output_hidden_states else None
    # 如果设置了输出所有自注意力头，则初始化一个空元组；否则设为 None
    all_self_attentions = () if output_attentions else None
    
    # 如果存在注意力遮罩，则扩展注意力遮罩以确保填充的令牌不被关注
    if attention_mask is not None:
        # 将注意力遮罩扩展为与隐藏状态的最后一个维度相同
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        # 将不需要关注的位置的隐藏状态置为0
        hidden_states[~expand_attention_mask] = 0
    
        # 扩展注意力遮罩，将其用于层间关注权重
        attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        )
    
    # 通过位置卷积嵌入层处理隐藏状态
    position_embeddings = self.pos_conv_embed(hidden_states)
    # 将位置嵌入加到隐藏状态上
    hidden_states = hidden_states + position_embeddings
    # 对隐藏状态应用丢弃（Dropout）
    hidden_states = self.dropout(hidden_states)
    
    # 检查是否启用了 Deepspeed Zero3
    deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
    
    # 遍历每一个层
    for layer in self.layers:
        # 如果要输出所有隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 添加层丢弃（LayerDrop）机制，根据配置随机决定是否跳过该层
        dropout_probability = torch.rand([])
        skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
    
        # 如果不跳过该层或者启用了 Deepspeed Zero3，则执行该层的前向传播
        if not skip_the_layer or deepspeed_zero3_is_enabled:
            # 如果启用了梯度检查点且正在训练，则使用梯度检查点函数执行该层的前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用层对象进行前向传播
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
            hidden_states = layer_outputs[0]
    
        # 如果跳过了该层，则输出设置为 None
        if skip_the_layer:
            layer_outputs = (None, None)
    
        # 如果要输出所有自注意力头，则将当前层的自注意力权重添加到 all_self_attentions 中
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
    # 对最终的隐藏状态应用层归一化
    hidden_states = self.layer_norm(hidden_states)
    
    # 如果要输出所有隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    
    # 如果不返回字典形式的输出，则返回元组形式的结果
    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    
    # 返回基础模型输出对象，包含最终的隐藏状态、所有隐藏状态和所有自注意力权重
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
# 定义一个名为 UniSpeechGumbelVectorQuantizer 的类，继承自 nn.Module，用于实现使用 Gumbel Softmax 进行向量量化的功能。
"""
Vector quantization using gumbel softmax. See [CATEGORICAL REPARAMETERIZATION WITH
GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
"""
class UniSpeechGumbelVectorQuantizer(nn.Module):

    def __init__(self, config):
        # 调用父类构造方法进行初始化
        super().__init__()
        # 设置向量量化的组数和每组的变量数目
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group

        # 检查配置中的 codevector_dim 是否可以被 num_groups 整除
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible by `config.num_codevector_groups`"
                f" {self.num_groups} for concatenation"
            )

        # 创建一个可训练的参数，用于存储码书（codebook）变量（码字）
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        
        # 定义一个线性层，用于将卷积的最后一个维度映射到 num_groups * num_vars 的大小
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # 设定温度参数，用于 Gumbel Softmax 分布
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs):
        # 计算概率分布的困惑度
        marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity
    def forward(self, hidden_states):
        # 获取输入张量的批大小、序列长度和隐藏大小
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到代码向量维度
        hidden_states = self.weight_proj(hidden_states)
        # 将投影后的张量形状转换为(batch_size * sequence_length * self.num_groups, -1)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 使用 gumbel_softmax 方法在可微分的方式中采样代码向量概率
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # 计算困惑度
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            # 非训练状态下，以非可微分方式取 argmax
            # 计算硬代码向量分布（one hot 编码）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs)

        # 将代码向量概率张量重新调整为(batch_size * sequence_length, -1)的形状
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率从代码向量集中检索代码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        # 将每个组中的代码向量求和，形状为(batch_size, sequence_length, -1)
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        # 返回结果：代码向量和困惑度
        return codevectors, perplexity
class UniSpeechPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为UniSpeechConfig
    config_class = UniSpeechConfig
    # 模型的基础名称前缀
    base_model_prefix = "unispeech"
    # 主要输入名称
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果module是UniSpeechGumbelVectorQuantizer类型，初始化其权重
        if isinstance(module, UniSpeechGumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        # 如果module是UniSpeechPositionalConvEmbedding类型，初始化其权重
        elif isinstance(module, UniSpeechPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        # 如果module是UniSpeechFeatureProjection类型，初始化其权重
        elif isinstance(module, UniSpeechFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果module是nn.Linear类型，初始化其权重和偏置
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        # 如果module是nn.LayerNorm或nn.GroupNorm类型，初始化其偏置和权重
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果module是nn.Conv1d类型，使用Kaiming正态分布初始化其权重
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                # 计算适当的均匀分布边界并初始化偏置
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html获取的一维卷积层输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 根据配置中的卷积核大小和步长计算卷积层的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    # 计算非填充部分的长度，即每个样本中非零元素的累积和的最后一个值
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

    # 根据非填充长度计算输出长度，转换为长整型并移至对应设备
    output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)

    # 获取批次大小
    batch_size = attention_mask.shape[0]

    # 创建一个与注意力掩码相同大小的零张量，但使用指定的数据类型和设备
    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )

    # 将输出长度对应位置的值设为1，确保在这些位置之前的所有值都受到注意
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1

    # 反转张量并对每行进行累积求和，然后再次反转，最后转换为布尔类型
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

    # 返回处理后的注意力掩码张量
    return attention_mask
# UNISPEECH_START_DOCSTRING 是一个长字符串，包含了有关 UniSpeech 模型的详细介绍和引用的论文信息
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

# UNISPEECH_INPUTS_DOCSTRING 是一个空字符串，用于承载有关 UniSpeech 模型输入的文档字符串信息
UNISPEECH_INPUTS_DOCSTRING = r"""
"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            # 输入的原始语音波形的浮点数值。可以通过加载 `.flac` 或 `.wav` 音频文件并转换成类型为 `List[float]` 或 `numpy.ndarray` 的数组来获得这些值，例如可以使用 `soundfile` 库 (`pip install soundfile`)。使用 [`AutoProcessor`] 进行填充和转换成 `torch.FloatTensor` 类型的张量。详见 [`Wav2Vec2Processor.__call__`] 的详细信息。
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于避免在填充标记索引上执行卷积和注意力操作。遮罩中的值选自 `[0, 1]`：

            # - 1 表示**未屏蔽**的标记，
            # - 0 表示**已屏蔽**的标记。

            # [什么是注意力遮罩？](../glossary#attention-mask)

            <Tip warning={true}>
            如果对应的处理器具有 `config.return_attention_mask == True`，则应传递 `attention_mask`。对于所有处理器具有 `config.return_attention_mask == False` 的模型，在进行批处理推理时，应**不要**传递 `attention_mask`，以避免性能下降。对于这些模型，`input_values` 应简单地填充为 0 并在不传递 `attention_mask` 的情况下传递。请注意，这些模型的结果也会根据 `input_values` 是否填充而略有不同。
            </Tip>

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
@add_start_docstrings(
    "The bare UniSpeech Model transformer outputting raw hidden-states without any specific head on top.",
    UNISPEECH_START_DOCSTRING,
)
"""
# 使用装饰器添加文档字符串，描述这是一个裸的 UniSpeech 模型，输出未经特定顶部头处理的原始隐藏状态。

class UniSpeechModel(UniSpeechPreTrainedModel):
    def __init__(self, config: UniSpeechConfig):
        super().__init__(config)
        self.config = config
        # 初始化特征提取器和特征投影器
        self.feature_extractor = UniSpeechFeatureEncoder(config)
        self.feature_projection = UniSpeechFeatureProjection(config)

        # 如果配置中包含时间或特征掩码概率，则初始化掩码的频谱嵌入
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 根据配置选择稳定层归一化编码器或一般编码器
        if config.do_stable_layer_norm:
            self.encoder = UniSpeechEncoderStableLayerNorm(config)
        else:
            self.encoder = UniSpeechEncoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states 复制而来
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
            # Compute mask indices for time axis based on configuration parameters
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            # Convert mask indices to boolean tensor on the same device as hidden_states
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            # Apply SpecAugment along time axis using masked_spec_embed
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            # Convert mask indices to boolean tensor on the same device as hidden_states
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            # Expand feature mask indices to match the shape of hidden_states
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            # Apply SpecAugment along feature axis by setting masked values to 0
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    @add_start_docstrings_to_model_forward(UNISPEECH_INPUTS_DOCSTRING)
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
    # 定义函数的返回类型为 Tuple 或 Wav2Vec2BaseModelOutput
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 如果未显式指定输出注意力权重，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未显式指定输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未显式指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征向量
        extract_features = self.feature_extractor(input_values)
        # 调整特征向量的维度顺序
        extract_features = extract_features.transpose(1, 2)

        # 如果存在注意力掩码，则计算对应于特征向量的减少的注意力掩码
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        # 使用特征投影层对特征向量进行投影
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 根据时间索引掩码和注意力掩码对隐藏状态进行掩码处理
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 使用编码器处理隐藏状态和注意力掩码等参数
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果不需要返回字典，则返回元组形式的输出
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 如果需要返回字典，则使用 Wav2Vec2BaseModelOutput 类包装输出
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """UniSpeech Model with a vector-quantization module and ctc loss for pre-training.""", UNISPEECH_START_DOCSTRING
)
class UniSpeechForPreTraining(UniSpeechPreTrainedModel):
    def __init__(self, config: UniSpeechConfig):
        super().__init__(config)
        # 初始化 UniSpeech 模型
        self.unispeech = UniSpeechModel(config)
        # 特征量化器的 dropout
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        # 初始化量化器
        self.quantizer = UniSpeechGumbelVectorQuantizer(config)
        # 将编码向量维度映射到投影编码向量维度
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        # 将投影编码向量维度映射到隐藏层大小
        self.project_hid = nn.Linear(config.proj_codevector_dim, config.hidden_size)

        # CTC 层，将隐藏状态映射到 CTC 类别数
        self.ctc_proj = nn.Linear(config.hidden_size, config.num_ctc_classes)
        # 最终 dropout
        self.dropout = nn.Dropout(config.final_dropout)

        # 初始化权重并应用最终处理
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        # 设置 Gumbel softmax 温度值，仅在训练时需要
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        # 弃用警告，禁止特征提取器的梯度计算，使其参数在训练期间不更新
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 禁止特征编码器的梯度计算，使其参数在训练期间不更新
        self.unispeech.feature_extractor._freeze_parameters()

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
        # 将目标特征和负样本特征连接在一起
        target_features = torch.cat([target_features, negative_features], dim=0)

        # 计算余弦相似度作为距离度量的对比损失的 logits
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1)
        logits = logits.type_as(target_features)

        # 应用温度
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(UNISPEECH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=UniSpeechForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播函数，接受输入值、注意力掩码、输出注意力权重、输出隐藏状态、是否返回字典格式结果等参数
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入值，类型为可选的 PyTorch 张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，类型为可选的 PyTorch 张量，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式结果，类型为可选的布尔值，默认为 None
@add_start_docstrings(
    """UniSpeech Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    UNISPEECH_START_DOCSTRING,
    """
        target_lang (`str`, *optional*):
            Language id of adapter weights. Adapter weights are stored in the format adapter.<lang>.safetensors or
            adapter.<lang>.bin. Only relevant when using an instance of [`UniSpeechForCTC`] with adapters. Uses 'eng'
            by default.
    """,
)
# 定义了一个新的类UniSpeechForCTC，继承自UniSpeechPreTrainedModel类，用于基于CTC进行语言建模
# UniSpeechForCTC类扩展了UniSpeech模型，并添加了用于CTC的语言建模头部

class UniSpeechForCTC(UniSpeechPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        # 初始化UniSpeech模型
        self.unispeech = UniSpeechModel(config)
        # 添加一个dropout层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言，默认为'eng'
        self.target_lang = target_lang

        # 检查配置是否定义了语言模型头的词汇表大小
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `UniSpeechForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        # 根据配置设置输出隐藏层的大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 添加一个线性层，用于语言建模头部
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # 重写`~PreTrainedModel.tie_weights`方法，以便在通过`from_pretrained(...)`传递`target_lang=...`时能够正确加载适配器权重
        # 这个方法不应该由用户调用，并且可能在将来被更改

        target_lang = self.target_lang

        # 如果定义了target_lang但未定义config.adapter_attn_dim，则抛出值错误异常
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果未定义target_lang但定义了config.adapter_attn_dim，则记录默认情况下target_lang被设置为'eng'
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果定义了target_lang，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 警告用户方法已过时，并将在 Transformers v5 版本中移除。建议使用 `freeze_feature_encoder` 方法代替。
    def freeze_feature_extractor(self):
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 `freeze_feature_encoder` 方法来冻结特征编码器，禁止其在训练期间更新参数。
        self.freeze_feature_encoder()

    # 禁止特征编码器的梯度计算，防止其在训练期间更新参数。
    def freeze_feature_encoder(self):
        self.unispeech.feature_extractor._freeze_parameters()

    # 禁止基础模型的梯度计算，使其参数在训练期间不会被更新。只有分类头会被更新。
    def freeze_base_model(self):
        for param in self.unispeech.parameters():
            param.requires_grad = False

    # 添加模型前向传播方法的文档字符串，包括输入、输出等信息。
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
        ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 默认情况下，如果未提供 return_dict，则使用模型配置中的 use_return_dict 设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 unispeech 模型生成输出
        outputs = self.unispeech(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态
        hidden_states = outputs[0]
        # 对隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states)

        # 使用语言模型头部生成 logits
        logits = self.lm_head(hidden_states)

        # 初始化 loss
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 检查标签是否超出词汇表大小
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 从 attention_mask 中获取输入长度
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充的标记用 -100 填充，当没有被注意到时
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # 使用 log_softmax 转换 logits，以备 ctc_loss 使用
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 关闭 cudnn 加速，因为 ctc_loss 不支持 fp16
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

        # 如果不要求返回字典形式的输出
        if not return_dict:
            # 组装输出元组
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回 CausalLMOutput 对象，包含 loss、logits、隐藏状态和注意力权重
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
@add_start_docstrings(
    """
    UniSpeech Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """,
    UNISPEECH_START_DOCSTRING,
)
class UniSpeechForSequenceClassification(UniSpeechPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of UniSpeech adapters (config.add_adapter=True)"
            )
        self.unispeech = UniSpeechModel(config)  # 初始化UniSpeech模型
        num_layers = config.num_hidden_layers + 1  # 计算层数：transformer层 + 输入嵌入层
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)  # 初始化层权重
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)  # 线性投影层
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)  # 分类器线性层

        # Initialize weights and apply final processing
        self.post_init()  # 执行后初始化操作

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.freeze_feature_extractor
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
        self.freeze_feature_encoder()  # 冻结特征编码器的参数

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.freeze_feature_encoder with wav2vec2->unispeech
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech.feature_extractor._freeze_parameters()  # 冻结特征编码器的参数

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.freeze_base_model with wav2vec2->unispeech
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.unispeech.parameters():
            param.requires_grad = False  # 冻结UniSpeech模型的所有参数

    @add_start_docstrings_to_model_forward(UNISPEECH_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从transformers.models.unispeech.modeling_unispeech.UniSpeech.forward复制，将Wav2Vec2->UniSpeech，wav2vec2->unispeech
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入值，可以是张量的可选类型
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可以是张量的可选类型，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以是布尔值的可选类型，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以是布尔值的可选类型，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可以是布尔值的可选类型，默认为None
        labels: Optional[torch.Tensor] = None,  # 标签，用于计算序列分类/回归损失的张量，可选类型，默认为None
    ) -> Union[Tuple, SequenceClassifierOutput]:  # 返回值类型注释，可以是元组或SequenceClassifierOutput

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 确定是否使用返回字典，如果未提供，则使用配置中的默认值
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states  # 根据配置决定是否输出加权层的隐藏状态

        outputs = self.unispeech(  # 使用UniSpeech模型进行前向传播
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:  # 如果配置指定使用加权层求和
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态的起始位置
            hidden_states = torch.stack(hidden_states, dim=1)  # 在维度1上堆叠隐藏状态
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行softmax归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 加权求和隐藏状态
        else:
            hidden_states = outputs[0]  # 否则直接获取第一个输出作为隐藏状态

        hidden_states = self.projector(hidden_states)  # 将隐藏状态投影到指定维度

        if attention_mask is None:  # 如果没有提供注意力掩码
            pooled_output = hidden_states.mean(dim=1)  # 对隐藏状态进行平均池化
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)  # 获取特征向量注意力掩码
            hidden_states[~padding_mask] = 0.0  # 将非填充部分的隐藏状态置为0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)  # 使用注意力掩码进行池化

        logits = self.classifier(pooled_output)  # 使用分类器预测逻辑回归

        loss = None  # 初始化损失为None
        if labels is not None:  # 如果提供了标签
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))  # 计算损失

        if not return_dict:  # 如果不要求返回字典形式的输出
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 组装输出元组
            return ((loss,) + output) if loss is not None else output  # 返回带有损失的输出元组或仅输出元组

        return SequenceClassifierOutput(  # 返回SequenceClassifierOutput对象，包含损失、逻辑回归、隐藏状态和注意力权重
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```