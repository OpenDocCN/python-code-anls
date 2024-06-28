# `.\models\data2vec\modeling_data2vec_audio.py`

```py
# 设置代码文件的编码格式为UTF-8

# 引入必要的依赖库和模块
# 版权声明和许可协议
# 本代码基于Apache License, Version 2.0发布

""" PyTorch Data2VecAudio model. """

# 引入必要的库和模块
import math  # 数学计算库
import warnings  # 警告处理库
from typing import Optional, Tuple, Union  # 类型提示模块

import numpy as np  # 数组处理库
import torch  # PyTorch深度学习库
import torch.utils.checkpoint  # PyTorch中的checkpoint功能
from torch import nn  # PyTorch中的神经网络模块
from torch.nn import CrossEntropyLoss  # 交叉熵损失函数

# 引入Hugging Face自定义的模块和类
from ...activations import ACT2FN  # 激活函数
from ...integrations.deepspeed import is_deepspeed_zero3_enabled  # DeepSpeed集成模块
from ...modeling_outputs import (
    BaseModelOutput,  # 基础模型输出
    CausalLMOutput,  # 因果语言模型输出
    SequenceClassifierOutput,  # 序列分类器输出
    TokenClassifierOutput,  # 标记分类器输出
    Wav2Vec2BaseModelOutput,  # Wav2Vec2基础模型输出
    XVectorOutput,  # X向量输出
)
from ...modeling_utils import PreTrainedModel  # 预训练模型基类
from ...utils import (
    add_code_sample_docstrings,  # 添加代码示例文档字符串
    add_start_docstrings,  # 添加起始文档字符串
    add_start_docstrings_to_model_forward,  # 添加模型前向传播的起始文档字符串
    is_peft_available,  # 是否有PEFT可用
    logging,  # 日志记录
)
from .configuration_data2vec_audio import Data2VecAudioConfig  # Data2VecAudio配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


_HIDDEN_STATES_START_POSITION = 2  # 隐藏状态的起始位置

# 用于文档的配置信息
_CONFIG_FOR_DOC = "Data2VecAudioConfig"

# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]  # 预期的输出形状

# CTC（连接时序分类）的文档信息
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 66.95

# 预训练模型存档列表
DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-audio-base",
    "facebook/data2vec-audio-base-10m",
    "facebook/data2vec-audio-base-100h",
    "facebook/data2vec-audio-base-960h",
    # 更多模型详见 https://huggingface.co/models?filter=data2vec-audio
]


# 从transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices复制而来
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算给定形状的随机掩码间隔。用于实现ASR的数据增强方法[SpecAugment](https://arxiv.org/abs/1904.08779)。
    注意：此方法不适合在TPU上运行，应该在训练期间作为预处理步骤在CPU上运行。
    """
    # 省略部分代码，未完整展示
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
    # 解包 shape 元组，得到 batch_size 和 sequence_length
    batch_size, sequence_length = shape

    # 检查 mask_length 是否合法（大于0）
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
        # 计算应该被mask的span的数量
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        # 确保 num_masked_span 不小于 min_masks
        num_masked_span = max(num_masked_span, min_masks)

        # 确保 num_masked_span * mask_length 不大于 sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保 num_masked_span 不大于 input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算每个样本的输入长度
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # 创建一个全为 False 的布尔数组作为 spec_aug_mask 的初始值
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    # 保存 spec_aug_mask 的索引
    spec_aug_mask_idxs = []

    # 计算最大允许的 masked span 数量
    max_num_masked_span = compute_num_masked_span(sequence_length)

    # 如果 max_num_masked_span 为 0，则直接返回空的 spec_aug_mask
    if max_num_masked_span == 0:
        return spec_aug_mask


这段代码主要是用于生成一种名为 SpecAugment 的遮罩技术，用于语音识别和其他序列数据处理中。
    # 对于每个输入长度计算需要掩码的片段数量
    for input_length in input_lengths:
        
        # 计算当前输入长度下的掩码片段数目
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要掩码的索引
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个样本索引作为填充向量的虚拟索引，以确保所有批次具有相同的维度，因为概率舍入
        # 选择第一个样本索引是为了简化向量的填充操作
        if len(spec_aug_mask_idx) == 0:
            # 当 `input_length` 严格小于 `sequence_length` 时会出现这种情况，
            # 此时最后一个标记应该是填充标记，可以用作虚拟掩码 ID
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟掩码索引添加到掩码索引数组中，确保数组长度达到 `max_num_masked_span`
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将列表转换为 NumPy 数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将掩码索引扩展为掩码片段
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量到起始索引，使索引创建一个掩码片段
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不超过 `sequence_length - 1`
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 使用索引散布掩码
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回最终的掩码张量
    return spec_aug_mask
# 定义一个用于处理音频转换的卷积层的类，继承自 nn.Module
class Data2VecAudioConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 从配置中获取输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，指定输入和输出维度、卷积核大小、步长和是否使用偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一个用于层归一化的 LayerNorm 层，参数为输出卷积维度，启用元素级仿射变换
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 获取指定的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数，接收隐藏状态并返回转换后的隐藏状态
    def forward(self, hidden_states):
        # 对输入的隐藏状态进行一维卷积处理
        hidden_states = self.conv(hidden_states)

        # 将卷积输出的维度进行转置，交换倒数第二和倒数第一维度
        hidden_states = hidden_states.transpose(-2, -1)
        # 对转置后的隐藏状态进行层归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 再次对隐藏状态进行维度转置，还原初始维度排列
        hidden_states = hidden_states.transpose(-2, -1)
        # 使用预定义的激活函数处理隐藏状态并返回
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer 复制而来，修改类名为 Data2VecAudioPadLayer
class Data2VecAudioPadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 根据给定的卷积位置嵌入数量计算需要移除的填充数量
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    # 前向传播函数，根据需要移除的填充数量截断隐藏状态的最后一维
    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 定义一个用于位置卷积的类，继承自 nn.Module
class Data2VecAudioPositionalConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个一维卷积层，指定输入和输出维度相同，卷积核大小、填充方式和卷积组数
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_pos_kernel_size,
            padding=config.conv_pos_kernel_size // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 创建一个用于处理填充的 Data2VecAudioPadLayer 类的实例
        self.padding = Data2VecAudioPadLayer(config.conv_pos_kernel_size)
        # 获取指定的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
        # 创建一个 LayerNorm 层，用于层归一化，参数为隐藏大小，禁用元素级仿射变换
        self.layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

    # 前向传播函数，接收隐藏状态并返回转换后的隐藏状态
    def forward(self, hidden_states):
        # 对输入的隐藏状态进行一维卷积处理
        hidden_states = self.conv(hidden_states)
        # 使用 padding 层处理卷积输出，截断最后一维的填充
        hidden_states = self.padding(hidden_states)

        # 将隐藏状态的维度进行转置，交换第一和第二维度
        hidden_states = hidden_states.transpose(1, 2)
        # 对转置后的隐藏状态进行层归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 再次对隐藏状态进行维度转置，还原初始维度排列
        hidden_states = hidden_states.transpose(1, 2)
        # 使用预定义的激活函数处理隐藏状态并返回
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 定义一个用于位置卷积嵌入的类，继承自 nn.Module
class Data2VecAudioPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用列表推导式创建包含多个 Data2VecAudioPositionalConvLayer 实例的 ModuleList
        self.layers = nn.ModuleList(
            [Data2VecAudioPositionalConvLayer(config) for _ in range(config.num_conv_pos_embeddings)]
        )
    # 定义一个方法，用于前向传播神经网络模型中的隐藏状态
    def forward(self, hidden_states):
        # 转置隐藏状态张量的第一维和第二维，以便适配网络层期望的输入格式
        hidden_states = hidden_states.transpose(1, 2)
        # 依次通过每一层网络层处理隐藏状态张量
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # 再次转置隐藏状态张量的第一维和第二维，使其恢复原始输入的维度
        hidden_states = hidden_states.transpose(1, 2)
        # 返回经过所有网络层处理后的隐藏状态张量
        return hidden_states
class Data2VecAudioFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()
        # 初始化卷积层列表，每层都是 Data2VecAudioConvLayer 类的实例，根据配置创建对应数量的层
        self.conv_layers = nn.ModuleList(
            [Data2VecAudioConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        )
        # 梯度检查点功能默认关闭
        self.gradient_checkpointing = False
        # 默认需要计算梯度
        self._requires_grad = True

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder._freeze_parameters 复制而来
    def _freeze_parameters(self):
        # 将所有参数的梯度计算关闭
        for param in self.parameters():
            param.requires_grad = False
        # 同时将类属性 _requires_grad 设置为 False
        self._requires_grad = False

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder.forward 复制而来
    def forward(self, input_values):
        # 将输入的波形数据增加一个维度，以符合模型的输入要求
        hidden_states = input_values[:, None]

        # 如果当前模型需要计算梯度并且处于训练状态，则设置 hidden_states 变量需要计算梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层，逐层计算特征表示
        for conv_layer in self.conv_layers:
            # 如果开启了梯度检查点并且模型在训练阶段，则使用梯度检查点函数来计算卷积层的输出
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        # 返回最终的隐藏状态表示
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection 复制而来，修改了类名和部分参数
class Data2VecAudioFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 层归一化层，用于标准化卷积层的输出
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 线性投影层，将卷积层的输出映射到隐藏状态的维度上
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # Dropout 层，用于随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 对隐藏状态进行层归一化处理
        norm_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态进行线性投影，映射到目标维度
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态进行 Dropout 处理，随机丢弃部分神经元
        hidden_states = self.dropout(hidden_states)
        # 返回投影后的隐藏状态和未投影的归一化隐藏状态，用于量化操作
        return hidden_states, norm_hidden_states


# 从 transformers.models.bart.modeling_bart.BartAttention 复制而来，修改了类名和部分参数
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
    ):
        # 调用父类初始化方法
        super().__init__()
        # 设置注意力机制的嵌入维度
        self.embed_dim = embed_dim
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置dropout率
        self.dropout = dropout
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads
        # 保存配置信息
        self.config = config

        # 检查嵌入维度是否能被注意力头的数量整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放系数，用于缩放注意力得分
        self.scaling = self.head_dim**-0.5
        # 是否为解码器
        self.is_decoder = is_decoder
        # 是否使用因果（causal）注意力
        self.is_causal = is_causal

        # 初始化键（key）的线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化值（value）的线性投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化查询（query）的线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化输出的线性投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新形状张量，以便为多头注意力做准备
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->Data2VecAudio
# 定义一个名为Data2VecAudioFeedForward的神经网络模块，继承自nn.Module
class Data2VecAudioFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 中间层的dropout操作，使用配置中的激活函数的dropout率
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 中间层的全连接层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择或者初始化激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 输出层的全连接层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 输出层的dropout操作，使用配置中的dropout率
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    # 定义前向传播函数，接受隐藏状态hidden_states作为输入
    def forward(self, hidden_states):
        # 中间层的全连接操作
        hidden_states = self.intermediate_dense(hidden_states)
        # 中间层的激活函数操作
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 中间层的dropout操作
        hidden_states = self.intermediate_dropout(hidden_states)

        # 输出层的全连接操作
        hidden_states = self.output_dense(hidden_states)
        # 输出层的dropout操作
        hidden_states = self.output_dropout(hidden_states)
        # 返回处理后的隐藏状态作为输出
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer with Wav2Vec2->Data2VecAudio
# 定义一个名为Data2VecAudioEncoderLayer的神经网络模块，继承自nn.Module
class Data2VecAudioEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用Data2VecAudioAttention作为注意力机制
        self.attention = Data2VecAudioAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # dropout操作，使用配置中的隐藏层dropout率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # LayerNorm操作，输入维度为config.hidden_size，epsilon值为config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 使用Data2VecAudioFeedForward作为前馈神经网络
        self.feed_forward = Data2VecAudioFeedForward(config)
        # 最终的LayerNorm操作，输入维度为config.hidden_size，epsilon值为config.layer_norm_eps
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 定义前向传播函数，接受隐藏状态hidden_states、注意力掩码attention_mask（可选）、是否输出注意力权重output_attentions（可选）
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 复制隐藏状态用于后续加法残差连接
        attn_residual = hidden_states
        # 使用注意力机制处理隐藏状态
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # dropout操作
        hidden_states = self.dropout(hidden_states)
        # 加法残差连接
        hidden_states = attn_residual + hidden_states

        # LayerNorm操作
        hidden_states = self.layer_norm(hidden_states)
        # 前馈神经网络操作
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终LayerNorm操作
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        # 如果输出注意力权重，将注意力权重加入输出元组
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出元组
        return outputs


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder with Wav2Vec2->Data2VecAudio
# 定义一个名为Data2VecAudioEncoder的神经网络模块，继承自nn.Module
class Data2VecAudioEncoder(nn.Module):
    # 初始化方法，接受一个配置参数对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置对象保存到实例变量中
        self.config = config
        # 创建一个位置卷积嵌入对象，并保存到实例变量中
        self.pos_conv_embed = Data2VecAudioPositionalConvEmbedding(config)
        # 创建一个 LayerNorm 层，并设置其参数
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，并设置其参数
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建一个由多个音频编码器层组成的模块列表，数量由配置中的 num_hidden_layers 决定
        self.layers = nn.ModuleList([Data2VecAudioEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接受输入的隐藏状态张量和其他可选参数
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
                # 确保填充的 token 输出为 0
                expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
                hidden_states[~expand_attention_mask] = 0

                # 扩展 attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

            # 计算位置嵌入
            position_embeddings = self.pos_conv_embed(hidden_states)
            # 将位置嵌入加到隐藏状态上
            hidden_states = hidden_states + position_embeddings
            # Layer Normalization
            hidden_states = self.layer_norm(hidden_states)
            # Dropout
            hidden_states = self.dropout(hidden_states)

            # 检查是否启用了 DeepSpeed Zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

            # 遍历每个 Transformer 层
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 添加 LayerDrop（参见 https://arxiv.org/abs/1909.11556 进行描述）
                dropout_probability = torch.rand([])

                # 判断是否跳过当前层
                skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 如果启用了梯度检查点和处于训练阶段，则调用梯度检查点函数
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                            output_attentions,
                        )
                    else:
                        # 否则直接调用 Transformer 层
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

            # 根据 return_dict 决定返回的数据结构
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Adapter复制而来，将Wav2Vec2改为Data2VecAudio
class Data2VecAudioAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 如果配置中的输出隐藏大小不等于隐藏大小，则可能需要降维特征维度
        if config.output_hidden_size != config.hidden_size:
            # 创建线性投影层，将隐藏状态维度降至输出隐藏大小
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # 创建LayerNorm层，用于规范投影后的隐藏状态
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        # 创建一个包含多个Data2VecAudioAdapterLayer模块的层列表
        self.layers = nn.ModuleList(Data2VecAudioAdapterLayer(config) for _ in range(config.num_adapter_layers))
        # 设置层的随机丢弃率
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # 如果存在投影层和LayerNorm层，则将隐藏状态投影至输出隐藏大小并规范化
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 调换维度顺序，将时间维度置于第二位
        hidden_states = hidden_states.transpose(1, 2)

        # 对每一层进行循环处理
        for layer in self.layers:
            # 计算当前层是否被丢弃的概率
            layerdrop_prob = np.random.random()
            # 如果非训练状态或者未丢弃该层，则通过当前层处理隐藏状态
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        # 恢复原始维度顺序，将时间维度放回第三位
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AdapterLayer复制而来，将Wav2Vec2改为Data2VecAudio
class Data2VecAudioAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个一维卷积层，用于处理隐藏状态
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states):
        # 将隐藏状态通过一维卷积层处理
        hidden_states = self.conv(hidden_states)
        # 对卷积结果进行门控线性单元（GLU）操作
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


class Data2VecAudioPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    # 使用Data2VecAudioConfig作为配置类
    config_class = Data2VecAudioConfig
    # 模型的基本名称前缀
    base_model_prefix = "data2vec_audio"
    # 主要输入名称
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化模型权重的方法，根据模块类型不同采取不同的初始化方式
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是 Data2VecAudioFeatureProjection 类型
        if isinstance(module, Data2VecAudioFeatureProjection):
            # 计算均匀分布的上下界 k
            k = math.sqrt(1 / module.projection.in_features)
            # 对投影层的权重进行均匀初始化
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            # 对投影层的偏置进行均匀初始化
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果模块是 Data2VecAudioPositionalConvLayer 类型
        elif isinstance(module, Data2VecAudioPositionalConvLayer):
            # 对卷积层的偏置进行常数初始化（设置为0）
            nn.init.constant_(module.conv.bias, 0)
        # 如果模块是 nn.Linear 类型
        elif isinstance(module, nn.Linear):
            # 对线性层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 nn.LayerNorm 或 nn.GroupNorm 类型
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 如果有偏置项，则将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
            # 如果有权重项，则将权重项初始化为全1
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        # 如果模块是 nn.Conv1d 类型
        elif isinstance(module, nn.Conv1d):
            # 对卷积层的权重进行 Kaiming 正态分布初始化
            nn.init.kaiming_normal_(module.weight)
            # 如果有偏置项，则计算均匀分布的上下界 k 并进行均匀初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # 从 Wav2Vec2PreTrainedModel 类的方法 _get_feat_extract_output_lengths 复制而来
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        # 根据需要是否添加适配器的标志
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        # 定义计算卷积层输出长度的内部函数
        def _conv_out_length(input_length, kernel_size, stride):
            # 使用 PyTorch 文档中描述的公式计算 1D 卷积层输出长度
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 根据配置中的卷积核大小和步长循环计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要添加适配器，根据配置中的适配器层数循环计算输出长度
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    # 从 Wav2Vec2PreTrainedModel 类的方法 _get_feature_vector_attention_mask 复制而来
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        """
        Computes the attention mask for the feature vector
        """
        # 计算未填充部分的长度，即 attention_mask.sum(-1)，但不是原地操作以便在推理模式下运行。
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 根据非填充长度获取特征提取器的输出长度，可选择添加适配器
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        # 获取批次大小
        batch_size = attention_mask.shape[0]

        # 重新初始化 attention_mask，确保所有值在输出长度之前都是被注意的
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 在输出长度的索引之前的所有位置设置为1，以确保这些位置被注意到
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1

        # 反转 attention_mask，累积求和，再次反转，并转换为布尔类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        # 返回处理后的 attention_mask
        return attention_mask
# DATA2VEC_AUDIO_START_DOCSTRING 的值是一个多行字符串，用于说明 Data2VecAudio 模型的背景和基本信息
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

# DATA2VEC_AUDIO_INPUTS_DOCSTRING 的值暂时为空字符串，用于定义 Data2VecAudio 模型的输入文档字符串
DATA2VEC_AUDIO_INPUTS_DOCSTRING = r"""
"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            # 输入的原始语音波形的浮点值。可以通过加载 *.flac* 或 *.wav* 音频文件得到值数组，类型为 *List[float]* 或 *numpy.ndarray*，
            # 例如通过 soundfile 库（*pip install soundfile*）实现。使用 [`AutoProcessor`] 进行填充并转换为类型为 *torch.FloatTensor* 的张量，
            # 详细信息请参阅 [`Wav2Vec2Processor.__call__`]。

        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免对填充令牌索引执行卷积和注意力的掩码。掩码的值为 `[0, 1]`：
            # 
            # - 1 表示**未掩码**的令牌，
            # - 0 表示**已掩码**的令牌。
            # 
            # [什么是注意力掩码？](../glossary#attention-mask)
            # 
            # <Tip warning={true}>
            # `attention_mask` 应该在相应的处理器具有 `config.return_attention_mask == True` 的情况下传递，这对所有预训练 Data2Vec Audio 模型都是成立的。
            # 请注意，即使有 `attention_mask`，零填充的输入与非填充的输入将会有略微不同的输出，因为在位置编码中有多个卷积层。有关更详细的解释，请参见
            # [这里](https://github.com/huggingface/transformers/issues/25621#issuecomment-1713759349)。
            # </Tip>

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详细信息请参见返回张量下的 `attentions`。
        
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详细信息请参见返回张量下的 `hidden_states`。
        
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
`
"""
注释：

@add_start_docstrings(
    "The bare Data2VecAudio Model transformer outputting raw hidden-states without any specific head on top.",
    DATA2VEC_AUDIO_START_DOCSTRING,
)
# 定义 Data2VecAudioModel 类，继承自 Data2VecAudioPreTrainedModel 类，表示数据向量音频模型
class Data2VecAudioModel(Data2VecAudioPreTrainedModel):
    def __init__(self, config: Data2VecAudioConfig):
        # 调用父类构造函数，初始化模型配置
        super().__init__(config)
        # 保存模型配置
        self.config = config
        # 创建音频特征提取器对象
        self.feature_extractor = Data2VecAudioFeatureEncoder(config)
        # 创建音频特征投影对象
        self.feature_projection = Data2VecAudioFeatureProjection(config)

        # 如果配置中的掩码时间概率大于 0 或者掩码特征概率大于 0，则需要掩码特征嵌入
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            # 初始化掩码特征嵌入参数
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 创建音频编码器对象
        self.encoder = Data2VecAudioEncoder(config)

        # 如果配置中包含适配器，则创建适配器对象；否则适配器对象为 None
        self.adapter = Data2VecAudioAdapter(config) if config.add_adapter else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征编码器函数，禁止特征编码器的梯度计算，以防止其在训练过程中更新参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    # 掩码隐藏状态函数，用于掩码隐藏状态的处理
    def _mask_hidden_states
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
            # Calculate mask indices for time axis based on configuration parameters
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
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
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
        # 如果未指定output_attentions，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取输入特征向量
        extract_features = self.feature_extractor(input_values)
        # 调整特征向量的维度顺序
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # 计算适应特征向量的减少后的attention_mask
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 对特征向量进行特征投影
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 在计算中对隐藏状态进行屏蔽处理
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 编码器的输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果存在适配器，应用适配器
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 如果返回字典，则创建Wav2Vec2BaseModelOutput对象并返回
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    """
    Data2VecAudio Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

    Inherited documentation and configuration are added from `DATA2VEC_AUDIO_START_DOCSTRING`.

    Args:
        config (:class:`~transformers.Data2VecAudioConfig`):
            The model configuration class that specifies the model's architecture and parameters.

    Attributes:
        data2vec_audio (:class:`~transformers.Data2VecAudioModel`):
            The base Data2VecAudioModel instance.
        dropout (:obj:`torch.nn.Dropout`):
            Dropout module with a dropout probability as specified in `config.final_dropout`.
        lm_head (:obj:`torch.nn.Linear`):
            The linear layer for the language modeling head with output size `config.vocab_size`.

    Raises:
        ValueError: If `config.vocab_size` is not defined in the model configuration.

    Notes:
        This class extends `Data2VecAudioPreTrainedModel` and adds a language modeling head on top for CTC.
    """
    def __init__(self, config):
        super().__init__(config)

        # Initialize base Data2VecAudioModel and dropout layer
        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        # Check if vocab_size is defined in the configuration
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        # Determine the output size of the linear layer based on model configuration
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.

        Deprecated:
            The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5.
            Please use the equivalent `freeze_feature_encoder` method instead.
        """
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
        self.data2vec_audio.feature_extractor._freeze_parameters()

    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward with wav2vec2->data2vec_audio
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,

        """
        Forward pass of the Data2VecAudioForCTC model.

        Args:
            input_values (torch.Tensor, optional):
                Input tensor of shape `(batch_size, sequence_length, feature_dim)` containing audio features.
            attention_mask (torch.Tensor, optional):
                Mask to avoid performing attention on padding tokens.
            output_attentions (bool, optional):
                Whether to return attentions weights of all attention layers.
            output_hidden_states (bool, optional):
                Whether to return hidden states of all layers.
            return_dict (bool, optional):
                Whether to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            labels (torch.Tensor, optional):
                Labels for computing the CTC loss.

        Returns:
            :class:`~transformers.modeling_outputs.CausalLMOutput`: A :class:`~transformers.modeling_outputs.CausalLMOutput` containing:
                - loss (`torch.FloatTensor`, optional):
                    CTC loss if :obj:`labels` is provided.
                - logits (`torch.FloatTensor`):
                    The logits output tensor of the language modeling head.

        Examples:
            For examples on usage, please see the documentation and code samples provided.

        Warnings:
            This method is adapted from `transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward`
            with modifications for Data2VecAudio models.
        """
        ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # Decide whether to return results as a dictionary based on the provided argument or configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Convert input audio features into vector representations
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states and apply dropout regularization
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Generate logits from the language model head
        logits = self.lm_head(hidden_states)

        # Initialize loss as None
        loss = None
        if labels is not None:
            # Check if any label index exceeds the vocabulary size, which is invalid
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # Retrieve the lengths of input features using attention mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # Identify valid labels by creating a mask and calculate their lengths
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # Apply log softmax to logits and transpose for CTC loss calculation
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # Disable CuDNN optimizations to ensure reproducibility in loss calculation
            with torch.backends.cudnn.flags(enabled=False):
                # Compute CTC loss using provided parameters
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        # If return_dict is False, return outputs as a tuple
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, return outputs wrapped in CausalLMOutput
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
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

        # 检查配置中是否存在并且允许使用适配器，如果是，则引发值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Data2VecAudio adapters (config.add_adapter=True)"
            )

        # 初始化 Data2VecAudioModel 模型
        self.data2vec_audio = Data2VecAudioModel(config)
        
        # 计算层数：transformer 层数加上输入嵌入层
        num_layers = config.num_hidden_layers + 1  

        # 如果配置中使用加权层求和，则初始化层权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # 线性投影层，将隐藏状态映射到分类器投影尺寸
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)

        # 分类器层，将投影后的特征映射到类别数量
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并进行后处理
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        # 发出警告，方法即将弃用，请改用 `freeze_feature_encoder`
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 `freeze_feature_encoder` 方法冻结特征编码器的参数
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数，禁止其梯度计算
        self.data2vec_audio.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 冻结基础模型的参数，禁止其梯度计算，只有分类头部会被更新
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward 复制，并将 wav2vec2 改为 data2vec_audio
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入的张量数据，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选
        labels: Optional[torch.Tensor] = None,  # 分类/回归任务的标签张量，可选
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 决定是否返回字典格式的输出，根据传入的参数或配置决定
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states  # 如果配置指定使用加权层求和，则强制输出隐藏状态

        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取输出中的隐藏状态
            hidden_states = torch.stack(hidden_states, dim=1)  # 在指定维度上堆叠隐藏状态张量
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行softmax归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 使用加权求和隐藏状态
        else:
            hidden_states = outputs[0]  # 获取普通的隐藏状态输出

        hidden_states = self.projector(hidden_states)  # 投影隐藏状态

        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)  # 如果没有注意力掩码，则对隐藏状态进行平均池化
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)  # 根据注意力掩码生成填充掩码
            hidden_states[~padding_mask] = 0.0  # 使用填充掩码将非填充位置置零
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)  # 对填充位置求和并进行平均池化

        logits = self.classifier(pooled_output)  # 使用分类器得到logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))  # 计算损失

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 如果不返回字典，则组合输出
            return ((loss,) + output) if loss is not None else output  # 返回包含损失的输出或者普通输出

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  # 返回字典格式的分类器输出
@add_start_docstrings(
    """
    Data2VecAudio Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    DATA2VEC_AUDIO_START_DOCSTRING,
)
class Data2VecAudioForAudioFrameClassification(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Data2VecAudio adapters"
                " (config.add_adapter=True)"
            )
        
        # 初始化 Data2VecAudioModel，并设置层数
        self.data2vec_audio = Data2VecAudioModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        
        # 如果设置了使用加权层求和，则初始化层权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 初始化分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        # 初始化模型权重
        self.init_weights()

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
        # 调用 freeze_feature_encoder 方法来冻结特征编码器的参数
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
        # 冻结基础模型的参数，使其在训练过程中不会更新，只有分类头会更新
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification.forward 复制，将 wav2vec2 替换为 data2vec_audio
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化是否返回字典，默认为配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据权重层求和的配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用data2vec_audio方法处理音频输入，获取输出
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置为使用加权层求和，则对隐藏状态进行加权求和处理
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态的起始位置
            hidden_states = torch.stack(hidden_states, dim=1)  # 在指定维度上堆叠隐藏状态张量
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行softmax归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 加权求和隐藏状态
        else:
            hidden_states = outputs[0]  # 否则直接使用第一个输出作为隐藏状态

        logits = self.classifier(hidden_states)  # 使用分类器对隐藏状态进行分类预测

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            # 计算分类器的损失，labels需要转换成合适的形状和类型
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        # 如果不返回字典形式的输出，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 组合输出结果
            return output

        # 返回TokenClassifierOutput对象，包含损失、logits、隐藏状态和注意力
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Copied from transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss
# 定义了一个 AMSoftmaxLoss 类，用于计算带有 AM-Softmax 的损失函数
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale  # 缩放因子，用于调整角度余弦值的范围
        self.margin = margin  # AM-Softmax 中的 margin 参数
        self.num_labels = num_labels  # 标签的数量
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)  # 权重矩阵，初始化为随机值
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数

    def forward(self, hidden_states, labels):
        labels = labels.flatten()  # 将标签展平，以便计算损失
        weight = nn.functional.normalize(self.weight, dim=0)  # 对权重进行 L2 归一化
        hidden_states = nn.functional.normalize(hidden_states, dim=1)  # 对隐藏状态进行 L2 归一化
        cos_theta = torch.mm(hidden_states, weight)  # 计算余弦相似度
        psi = cos_theta - self.margin  # 计算带有 margin 的调整值

        onehot = nn.functional.one_hot(labels, self.num_labels)  # 将标签转为 one-hot 格式
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)  # 根据标签应用 AM-Softmax 运算得到最终 logits
        loss = self.loss(logits, labels)  # 计算损失

        return loss


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer
# 定义了 TDNNLayer 类，用于实现时延神经网络层
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]  # 输入通道维度
        self.out_conv_dim = config.tdnn_dim[layer_id]  # 输出通道维度
        self.kernel_size = config.tdnn_kernel[layer_id]  # 卷积核大小
        self.dilation = config.tdnn_dilation[layer_id]  # 空洞卷积的扩展率

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)  # 使用线性层作为卷积核
        self.activation = nn.ReLU()  # 激活函数为 ReLU

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_peft_available():
            from peft.tuners.lora import LoraLayer

            if isinstance(self.kernel, LoraLayer):
                warnings.warn(
                    "Detected LoRA on TDNNLayer. LoRA weights won't be applied due to optimization. "
                    "You should exclude TDNNLayer from LoRA's target modules.",
                )

        # for backward compatibility, we keep nn.Linear but call F.conv1d for speed up
        hidden_states = hidden_states.transpose(1, 2)  # 转置张量以适应卷积操作的输入要求
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)  # 调整权重形状
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)  # 执行卷积操作
        hidden_states = hidden_states.transpose(1, 2)  # 还原张量形状

        hidden_states = self.activation(hidden_states)  # 应用 ReLU 激活函数
        return hidden_states


@add_start_docstrings(
    """
    Data2VecAudio Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    DATA2VEC_AUDIO_START_DOCSTRING,
)
# 定义了 Data2VecAudioForXVector 类，扩展自 Data2VecAudioPreTrainedModel，用于 XVector 特征提取
class Data2VecAudioForXVector(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法，传递配置参数
        super().__init__(config)

        # 创建音频数据转换模型对象
        self.data2vec_audio = Data2VecAudioModel(config)
        
        # 计算层数，包括Transformer层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        
        # 如果配置指定使用加权层求和，则初始化层权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 创建线性映射层，将隐藏状态映射到TDNN的输入维度
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 创建多个TDNN层的列表
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 创建特征提取器的线性层，用于生成x-vector的输出维度
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        
        # 创建分类器的线性层，将x-vector映射到分类数目的维度
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 初始化AMSoftmax损失函数，使用指定的输出维度和类别数目
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 执行权重初始化的函数
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 发出警告，表明此函数将被弃用，建议使用freeze_feature_encoder代替
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用freeze_feature_encoder方法，冻结特征编码器的参数
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数，使其在训练期间不会更新梯度
        self.data2vec_audio.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历数据转换模型的所有参数，并将requires_grad设置为False，以禁用它们的梯度计算
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从PyTorch文档中获取的1D卷积层输出长度的计算公式
            return (input_length - kernel_size) // stride + 1

        # 计算每个TDNN层的输出长度
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
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector.forward复制而来，修改为使用data2vec_audio
    # 定义一个前向传播方法，用于模型推断或训练时的正向处理
    def forward(
        self,
        # 输入数据张量，可以为空
        input_values: Optional[torch.Tensor],
        # 注意力掩码张量，用于指定模型关注的部分，可以为空
        attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，可选参数，默认为空
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选参数，默认为空
        output_hidden_states: Optional[bool] = None,
        # 是否返回结果字典形式，可选参数，默认为空
        return_dict: Optional[bool] = None,
        # 标签数据张量，用于训练时指定真实标签，可以为空
        labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        # 初始化 return_dict，如果未提供则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果配置中指定了使用加权层求和的隐藏状态，则设置 output_hidden_states 为 True
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        
        # 将输入数据通过 data2vec_audio 方法处理，获取模型的输出
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 如果配置中使用了加权层求和的隐藏状态，对隐藏状态进行加权求和操作
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接取模型输出的第一个元素作为隐藏状态
            hidden_states = outputs[0]
        
        # 将隐藏状态通过 projector 方法进行投影处理
        hidden_states = self.projector(hidden_states)
        
        # 对每个 TDNN 层依次处理隐藏状态
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)
        
        # 统计池化操作
        if attention_mask is None:
            # 如果没有提供 attention_mask，则对隐藏状态在第一维度（batch 维度）上进行均值和标准差计算
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            # 根据 attention_mask 计算特征提取的输出长度和 TDNN 层的输出长度
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            # 对每个序列长度进行遍历，计算每个序列的均值和标准差
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        # 将均值和标准差拼接在一起作为统计池化的结果
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)
        
        # 将统计池化的结果通过 feature_extractor 进行特征提取
        output_embeddings = self.feature_extractor(statistic_pooling)
        
        # 将特征提取的结果通过 classifier 进行分类器预测得到 logits
        logits = self.classifier(output_embeddings)
        
        # 初始化损失为 None
        loss = None
        # 如果提供了 labels，则计算损失
        if labels is not None:
            loss = self.objective(logits, labels)
        
        # 如果不要求返回字典形式的输出，则直接返回元组形式的输出
        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
        
        # 如果要求返回字典形式的输出，则构建 XVectorOutput 对象并返回
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```