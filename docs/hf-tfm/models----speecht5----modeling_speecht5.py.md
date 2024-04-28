# `.\transformers\models\speecht5\modeling_speecht5.py`

```
# 设定编码格式为 UTF-8

# 版权声明，声明代码的版权归于 Fairseq 作者、微软研究院以及 HuggingFace 公司团队所有
# 根据 Apache 许可证 2.0 版本（"许可证"）授权，除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，不提供任何形式的担保或条件，无论是明示的还是暗示的。
# 有关许可证的详细信息，请参阅许可证。

""" PyTorch SpeechT5 model."""

# 导入所需库
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

# 从 HuggingFace 库中导入各种函数和类
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSpectrogramOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 隐藏状态起始位置索引
_HIDDEN_STATES_START_POSITION = 1

# 通用文档字符串，用于指向 SpeechT5Config 配置
_CONFIG_FOR_DOC = "SpeechT5Config"

# 预训练的 SpeechT5 模型列表
SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/speecht5_asr",
    "microsoft/speecht5_tts",
    "microsoft/speecht5_vc",
    # 在 https://huggingface.co/models?filter=speecht5 查看所有 SpeechT5 模型
]

# 从 transformers.models.bart.modeling_bart 中复制的函数，用于将输入的 token 向右移动一位
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个与输入形状相同的新张量，全部填充为零
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入的 token 向右移动一位
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将第一个 token 设置为 decoder_start_token_id
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果 pad_token_id 未定义，则抛出错误
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# 将输入的声谱图向右移动一步，同时将序列长度按照给定的 reduction_factor 进行压缩
def shift_spectrograms_right(input_values: torch.Tensor, reduction_factor: int = 1):
    """
    Shift input spectrograms one timestep to the right. Also applies the reduction factor to the sequence length.
    """
    # 如果 reduction_factor 大于 1，则对帧进行抽稀
    if reduction_factor > 1:
        input_values = input_values[:, reduction_factor - 1 :: reduction_factor]

    # 创建一个与输入形状相同的新张量，全部填充为零
    shifted_input_values = input_values.new_zeros(input_values.shape)
    # 将输入值的第二列到最后一列替换为输入值的第一列到倒数第二列的克隆
    shifted_input_values[:, 1:] = input_values[:, :-1].clone()

    # 将标签中可能存在的值为 -100 的位置替换为零
    shifted_input_values.masked_fill_(shifted_input_values == -100.0, 0.0)

    # 返回替换后的输入值
    return shifted_input_values
# 从transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices中复制代码

# 定义一个函数来计算给定形状的随机mask区间。用于实现[SpecAugment: A Simple Data Augmentation Method for ASR]中的数据增强方法。
# 注意这个方法没有被优化用于在TPU上运行，应该在CPU上作为训练过程中的预处理步骤来运行。

# shape: 要计算mask的形状，是一个大小为2的元组，第一个元素是batch大小，第二个元素是要跨越的轴的长度。
# mask_prob: 整个轴的百分比（在0和1之间）将被mask。长度为`mask_length`的独立生成的mask区间的数量由`mask_prob*shape[1]/mask_length`来计算。
# mask_length: mask的大小
# min_masks: 被mask的最小区间数
# attention_mask: 一个（右填充的）注意力mask，可以独立地缩短每个batch维度的特征轴。

def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:

    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 给定输入长度，计算应该被mask的区间数量
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保被mask的span数量 <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保被mask的span数量 <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算batch中被mask的区间数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    # 用于填充的SpecAugment mask
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    # 计算在batch中被mask的区间的最大数量
    max_num_masked_span = compute_num_masked_span(sequence_length)
    # 如果最大遮罩跨度为0，则直接返回原始的规范化增强掩码
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历每个输入序列的长度
    for input_length in input_lengths:
        # 计算当前输入序列的遮罩跨度数量
        num_masked_span = compute_num_masked_span(input_length)

        # 获取随机的索引来生成遮罩
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个样本的索引作为填充向量的虚拟索引，以确保由于概率舍入导致所有批次具有相同的维度
        # 选择第一个样本只是将这些向量填充两次。
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只会发生在 `input_length` 严格小于 `sequence_length` 的情况下，
            # 此时最后一个标记必须是填充标记，我们可以将其用作虚拟遮罩标识符
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟索引添加到遮罩索引中，以填充到最大遮罩跨度数量
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将遮罩索引转换为 NumPy 数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮罩索引扩展为遮罩跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 将偏移添加到起始索引，以便索引现在形成一个跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保我们不能有大于序列长度的索引
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将遮罩索引应用到规范化增强掩码
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回规范化增强掩码
    return spec_aug_mask
# 定义 SpeechT5NoLayerNormConvLayer 类，继承自 nn.Module
class SpeechT5NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 获取输入卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 获取输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]
        # 创建卷积层对象
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 指定激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 定义前向传播方法
    def forward(self, hidden_states):
        # 卷积操作
        hidden_states = self.conv(hidden_states)
        # 激活函数激活
        hidden_states = self.activation(hidden_states)
        # 返回结果
        return hidden_states


# 定义 SpeechT5LayerNormConvLayer 类，继承自 nn.Module
class SpeechT5LayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 获取输入卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 获取输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]
        # 创建卷积层对象
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一维 LayerNorm 层
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 指定激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 定义前向传播方法
    def forward(self, hidden_states):
        # 卷积操作
        hidden_states = self.conv(hidden_states)
        # 将通道维度移到最后一个维度
        hidden_states = hidden_states.transpose(-2, -1)
        # LayerNorm 操作
        hidden_states = self.layer_norm(hidden_states)
        # 再次将通道维度移到前面
        hidden_states = hidden_states.transpose(-2, -1)
        # 激活函数激活
        hidden_states = self.activation(hidden_states)
        # 返回结果
        return hidden_states


# 定义 SpeechT5GroupNormConvLayer 类，继承自 nn.Module
class SpeechT5GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 获取输入卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 获取输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]
        # 创建卷积层对象
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 指定激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
        # 创建 GroupNorm 层
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    # 定义前向传播方法
    def forward(self, hidden_states):
        # 卷积操作
        hidden_states = self.conv(hidden_states)
        # GroupNorm 操作
        hidden_states = self.layer_norm(hidden_states)
        # 激活函数激活
        hidden_states = self.activation(hidden_states)
        # 返回结果
        return hidden_states
        # 调用父类构造函数
        super().__init__()
        # 初始化偏移量为2，用于位置编码
        self.offset = 2
        # 设置嵌入维度
        self.embedding_dim = embedding_dim
        # 设置填充索引（可选）
        self.padding_idx = padding_idx
        # 调用make_weights方法创建权重参数
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用get_embedding方法生成嵌入权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        # 如果已经存在权重参数，则在前向传播时将权重放在正确的数据类型和设备上
        if hasattr(self, "weights"):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 创建权重参数并设置为不可训练
        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算位置编码的频率
        emb = math.log(10000) / (half_dim - 1)
        # 计算正弦部分的位置编码
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        # 计算位置编码矩阵
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        # 拼接正弦和余弦部分，并调整形状
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果embedding_dim为奇数，则补零
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果存在填充索引，则将对应行置零
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        # 获取输入张量的形状
        bsz, seq_len = input_ids.size()
        # 根据输入的令牌 ID 创建位置 ID。任何填充的令牌保持填充状态
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # 如果超出已有嵌入权重的范围，则重新生成嵌入权重
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 根据位置 ID 选择相应的权重，并调整形状
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
        """
        将非填充符号替换为它们的位置数字。位置数字从 padding_idx+1 开始。填充符号将被忽略。这是根据 fairseq 的 `utils.make_positions` 修改的。

        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # 在这里进行的一系列类型转换和转换的操作被精心平衡，既适用于 ONNX 导出又适用于 XLA。
        # 创建一个掩码，指示非填充位置为1，填充位置为0
        mask = input_ids.ne(padding_idx).int()
        # 计算增量索引，注意加上过去的键值对长度，然后乘以掩码确保只对非填充位置进行操作
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        # 返回增量索引，加上填充索引，确保填充位置保持不变
        return incremental_indices.long() + padding_idx
# 声明一个自定义的模块 SpeechT5PositionalConvEmbedding，继承自 nn.Module
class SpeechT5PositionalConvEmbedding(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 创建一个 1D 卷积层
        self.conv = nn.Conv1d(
            config.hidden_size,  # 输入通道数
            config.hidden_size,  # 输出通道数
            kernel_size=config.num_conv_pos_embeddings,  # 卷积核大小
            padding=config.num_conv_pos_embeddings // 2,  # 填充大小
            groups=config.num_conv_pos_embedding_groups,  # 分组卷积参数
        )

        # 权重归一化
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 深度规模优化
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 等效填充层
        self.padding = SpeechT5SamePadLayer(config.num_conv_pos_embeddings)
        # 获取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数
    def forward(self, hidden_states):
        # 维度转换，交换维度 1 和 2
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)  # 卷积操作
        hidden_states = self.padding(hidden_states)  # 填充
        hidden_states = self.activation(hidden_states)  # 激活函数
        hidden_states = hidden_states.transpose(1, 2)  # 再次维度转换
        return hidden_states  # 返回结果


# 声明一个自定义的模块 SpeechT5ScaledPositionalEncoding，继承自 nn.Module
class SpeechT5ScaledPositionalEncoding(nn.Module):
    # 初始化函数
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe, persistent=False)  # 注册缓冲区
        self.dropout = nn.Dropout(p=dropout)  # Dropout 层
        self.dim = dim  # 编码维度
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))  # 可训练参数 alpha

    # 前向传播函数
    def forward(self, emb):
        emb = emb + self.alpha * self.pe[:, : emb.size(1)]  # 增加位置编码
        emb = self.dropout(emb)  # Dropout 操作
        return emb  # 返回结果


# 声明一个自定义的模块 SpeechT5RelativePositionalEncoding，继承自 nn.Module
class SpeechT5RelativePositionalEncoding(torch.nn.Module):
    # 初始化函数
    def __init__(self, dim, max_length=1000):
        super().__init__()
        self.dim = dim  # 编码维度
        self.max_length = max_length  # 最大长度
        self.pe_k = torch.nn.Embedding(2 * max_length, dim)  # 嵌入层
    # 定义一个方法，用于在 Transformer 模型中进行位置编码的计算，输入隐藏状态
    def forward(self, hidden_states):
        # 获取输入隐藏状态的序列长度
        seq_len = hidden_states.shape[1]
        # 生成一个序列长度的张量，表示位置编码
        pos_seq = torch.arange(0, seq_len).long().to(hidden_states.device)
        # 将位置编码转换为一个二维矩阵，行表示位置，列表示相对位置偏移
        pos_seq = pos_seq[:, None] - pos_seq[None, :]

        # 截断位置编码中大于最大长度的部分
        pos_seq[pos_seq < -self.max_length] = -self.max_length
        # 截断位置编码中小于负最大长度的部分
        pos_seq[pos_seq >= self.max_length] = self.max_length - 1
        # 将位置编码整体偏移使得所有值都在 [0, 2 * max_length) 的范围内
        pos_seq = pos_seq + self.max_length

        # 将位置编码传递给位置编码层，返回位置编码结果
        return self.pe_k(pos_seq)
# 创建一个名为 SpeechT5SamePadLayer 的类，继承自 nn.Module
class SpeechT5SamePadLayer(nn.Module):
    # 初始化方法，接受 num_conv_pos_embeddings 作为参数
    def __init__(self, num_conv_pos_embeddings):
        # 调用父类的初始化方法
        super().__init__()
        # 如果 num_conv_pos_embeddings 除以 2 的余数为 0，设置 num_pad_remove 为 1，否则为 0
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    # 前向传播方法，接受 hidden_states 作为参数
    def forward(self, hidden_states):
        # 如果 num_pad_remove 大于 0，从 hidden_states 中移除最后 num_pad_remove 个元素
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        # 返回处理后的 hidden_states
        return hidden_states


# 创建一个名为 SpeechT5FeatureEncoder 的类，继承自 nn.Module
class SpeechT5FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    # 初始化方法，接受 config 作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 根据 config 中的 feat_extract_norm 属性选择不同的卷积层配置
        if config.feat_extract_norm == "group":
            conv_layers = [SpeechT5GroupNormConvLayer(config, layer_id=0)] + [
                SpeechT5NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                SpeechT5LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 如果 feat_extract_norm 不是 'group' 或 'layer'，抛出数值错误异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        # 创建 nn.ModuleList 对象并保存卷积层配置
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    # 冻结参数方法
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    # 前向传播方法，接受 input_values 作为参数
    def forward(self, input_values):
        # 将 input_values 变形为二维张量
        hidden_states = input_values[:, None]

        # 如果_requires_grad为True且处于训练状态，将hidden_states设置为需要梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历卷积层并应用每一层
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        # 返回处理后的 hidden_states
        return hidden_states


# 创建一个名为 SpeechT5FeatureProjection 的类，继承自 nn.Module
class SpeechT5FeatureProjection(nn.Module):
    # 初始化方法，接受 config 作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建 LayerNorm 层，用于归一化最后一个维度
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 创建线性变换层，将卷积后的特征映射到隐藏层维度
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.feat_proj_dropout)
    # 定义一个方法用于前向传播，接收隐藏状态作为输入参数
    def forward(self, hidden_states):
        # 对隐藏状态进行层归一化，非投影的隐藏状态用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 通过投影层对归一化后的隐藏状态进行投影
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态和归一化的隐藏状态
        return hidden_states, norm_hidden_states
# 定义一个名为SpeechT5SpeechEncoderPrenet的神经网络模型类
class SpeechT5SpeechEncoderPrenet(nn.Module):
    # 初始化函数，设置网络的配置信息，并创建相关的子模块
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_encoder = SpeechT5FeatureEncoder(config)
        self.feature_projection = SpeechT5FeatureProjection(config)

        # 如果mask_time_prob大于0.0或者mask_feature_prob大于0.0，则模型需要masking向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.pos_conv_embed = SpeechT5PositionalConvEmbedding(config)
        self.pos_sinusoidal_embed = SpeechT5SinusoidalPositionalEmbedding(
            config.max_speech_positions + config.pad_token_id + 1,
            config.hidden_size,
            config.pad_token_id,
        )

    # 冻结特征编码器的参数
    def freeze_feature_encoder(self):
        self.feature_encoder._freeze_parameters()

    # 前向传播函数，接收输入和注意力掩码，返回隐藏状态和注意力掩码
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
    ):
        # 提取特征并转置
        extract_features = self.feature_encoder(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # 计算对应于特征向量的减少注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1],
                attention_mask,
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        positional_conv_embedding = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + positional_conv_embedding

        if attention_mask is not None:
            padding_mask = attention_mask.ne(1).long()
        else:
            padding_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)

        positional_sinusoidal_embeddings = self.pos_sinusoidal_embed(padding_mask)
        hidden_states = hidden_states + positional_sinusoidal_embeddings

        return hidden_states, attention_mask

    # 从transformers.models.unispeech.modeling_unispeech.UniSpeechPreTrainedModel._get_feature_vector_attention_mask中复制的函数
    # 获取特征向量的注意力掩码，根据给定的特征向量长度和注意力掩码
    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # 有效地计算 attention_mask 在最后一个维度上的累积和，但不是原位操作，以便能够在推理模式下运行
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        # 获取特征提取输出长度，并转换为整型数据类型
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)
        batch_size = attention_mask.shape[0]

        # 创建与特征向量长度相同的全零注意力掩码
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # 这两个操作确保输出长度之前的所有值都受到了注意
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # 翻转注意力掩码，计算累积和，再翻转回去，最后转换为布尔型数据类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    # 从 transformers.models.unispeech.modeling_unispeech.UniSpeechPreTrainedModel._get_feat_extract_output_lengths 复制而来
    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        计算卷积层的输出长度
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的一维卷积层输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

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
        # `config.apply_spec_augment` 可以将 masking 设为 False
        if not getattr(self.config, "apply_spec_augment", True):
            # 如果不需要应用 SpecAugment，则直接返回原始 hidden_states
            return hidden_states

        # 生成索引并沿时间轴应用 SpecAugment
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # 使用给定的 mask_time_indices 沿时间轴应用 SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 生成 mask_time_indices，并根据 mask_time_prob 和 mask_time_length 随机选择部分索引
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            # 将生成的 mask_time_indices 转换为 torch.Tensor，并根据 hidden_states 的设备和 dtype 进行设定
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            # 使用 masked_spec_embed 将隐藏状态中的对应索引进行遮蔽
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # 生成索引并沿特征轴应用 SpecAugment
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            # 将生成的 mask_feature_indices 转换为 torch.Tensor，并根据 hidden_states 的设备和 dtype 进行设定
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            # 将 mask_feature_indices 中的每个位置向量沿 sequence_length 展开后，与 hidden_states 进行 element-wise 乘法遮蔽
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        # 返回 hidden_states
        return hidden_states
# 定义一个名为 SpeechT5SpeechDecoderPrenet 的类，继承自 nn.Module
class SpeechT5SpeechDecoderPrenet(nn.Module):
    def __init__(self, config):
        # 初始化函数，接收一个 config 参数
        super().__init__()
        # 调用父类的初始化函数
        self.config = config
        # 保存传入的 config 参数

        # 创建神经网络层列表
        self.layers = nn.ModuleList(
            # 使用列表推导式创建 nn.Linear 图层的列表
            [
                nn.Linear(
                    config.num_mel_bins if i == 0 else config.speech_decoder_prenet_units,
                    config.speech_decoder_prenet_units,
                )
                for i in range(config.speech_decoder_prenet_layers)
            ]
        )

        # 创建最终的线性层
        self.final_layer = nn.Linear(config.speech_decoder_prenet_units, config.hidden_size)
        # 创建编码位置的对象
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            config.positional_dropout,
            config.hidden_size,
            config.max_speech_positions,
        )
        # 创建说话者嵌入层
        self.speaker_embeds_layer = nn.Linear(config.speaker_embedding_dim + config.hidden_size, config.hidden_size)

    # 定义一个内部方法 _consistent_dropout
    def _consistent_dropout(self, inputs_embeds, p):
        # 应用一致性的 dropout 操作
        mask = torch.bernoulli(inputs_embeds[0], p=p)
        all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
        return torch.where(all_masks == 1, inputs_embeds, 0) * 1 / (1 - p)

    # 定义前向传播函数
    def forward(
        self,
        input_values: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
    ):
        # 强制应用 dropout，即使在评估时也是如此

        inputs_embeds = input_values
        # 对每一层进行操作
        for layer in self.layers:
            # 应用 ReLU 激活函数
            inputs_embeds = nn.functional.relu(layer(inputs_embeds))
            # 应用一致的 dropout 操作
            inputs_embeds = self._consistent_dropout(inputs_embeds, self.config.speech_decoder_prenet_dropout)

        # 运行最终的线性层
        inputs_embeds = self.final_layer(inputs_embeds)
        # 运行编码位置
        inputs_embeds = self.encode_positions(inputs_embeds)

        # 如果存在说话者嵌入，则进行特定操作
        if speaker_embeddings is not None:
            # 对说话者嵌入进行归一化
            speaker_embeddings = nn.functional.normalize(speaker_embeddings)
            # 设定说话者嵌入与输入嵌入维度相同
            speaker_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, inputs_embeds.size(1), -1)
            # 将说话者嵌入与输入嵌入连接起来
            inputs_embeds = torch.cat([inputs_embeds, speaker_embeddings], dim=-1)
            # 运行说话者嵌入层
            inputs_embeds = nn.functional.relu(self.speaker_embeds_layer(inputs_embeds))

        return inputs_embeds
    # 初始化函数，接受配置和层编号作为参数
    def __init__(self, config, layer_id=0):
        # 继承父类初始化方法
        super().__init__()
    
        # 如果层编号为0，输入卷积维度为配置中的num_mel_bins
        if layer_id == 0:
            in_conv_dim = config.num_mel_bins
        else:
            in_conv_dim = config.speech_decoder_postnet_units
    
        # 如果层编号为配置中的speech_decoder_postnet_layers - 1，输出卷积维度为配置中的num_mel_bins
        # 否则输出卷积维度为配置中的speech_decoder_postnet_units
        if layer_id == config.speech_decoder_postnet_layers - 1:
            out_conv_dim = config.num_mel_bins
        else:
            out_conv_dim = config.speech_decoder_postnet_units
    
        # 创建一维卷积层
        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.speech_decoder_postnet_kernel,
            stride=1,
            padding=(config.speech_decoder_postnet_kernel - 1) // 2,
            bias=False,
        )
        # 创建一维批量归一化层
        self.batch_norm = nn.BatchNorm1d(out_conv_dim)
    
        # 如果层编号小于配置中的speech_decoder_postnet_layers - 1，激活函数为双曲正切函数
        # 否则激活函数为空
        if layer_id < config.speech_decoder_postnet_layers - 1:
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
        # 创建一维Dropout层，丢弃概率为配置中的speech_decoder_postnet_dropout
        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)
    
    # 正向传播函数
    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)  # 卷积操作
        hidden_states = self.batch_norm(hidden_states)  # 批量归一化操作
        if self.activation is not None:  # 如果激活函数不为空
            hidden_states = self.activation(hidden_states)  # 使用激活函数
        hidden_states = self.dropout(hidden_states)  # Dropout操作
        return hidden_states  # 返回结果
# 定义一个名为 SpeechT5SpeechDecoderPostnet 的类，继承自 nn.Module
class SpeechT5SpeechDecoderPostnet(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将参数 config 保存在 self.config 中
        self.config = config

        # 定义一个全连接层，将隐藏状态映射到输出特征向量
        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)
        # 定义一个全连接层，将隐藏状态映射到输出概率
        self.prob_out = nn.Linear(config.hidden_size, config.reduction_factor)

        # 定义一个包含多层 SpeechT5BatchNormConvLayer 的神经网络层
        self.layers = nn.ModuleList(
            [SpeechT5BatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)]
        )

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states: torch.Tensor):
        # 将隐藏状态映射到输出特征向量并重新组织张量形状
        outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, self.config.num_mel_bins)
        # 经过 postnet 处理得到最终输出特征向量
        outputs_after_postnet = self.postnet(outputs_before_postnet)
        # 将隐藏状态映射到输出概率，并重新组织张量形状
        logits = self.prob_out(hidden_states).view(hidden_states.size(0), -1)
        # 返回输出特征向量、经过 postnet 处理的输出特征向量和输出概率
        return outputs_before_postnet, outputs_after_postnet, logits

    # postnet 处理方法，接受隐藏状态作为输入
    def postnet(self, hidden_states: torch.Tensor):
        # 将隐藏状态的维度进行转置
        layer_output = hidden_states.transpose(1, 2)
        # 遍历多层神经网络层，并对隐藏状态进行处理
        for layer in self.layers:
            layer_output = layer(layer_output)
        # 将原始隐藏状态与处理后的隐藏状态相加，并将维度进行转置
        return hidden_states + layer_output.transpose(1, 2)


# 定义一个名为 SpeechT5TextEncoderPrenet 的类，继承自 nn.Module
class SpeechT5TextEncoderPrenet(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将参数 config 保存在 self.config 中
        self.config = config
        # 定义一个嵌入层，将输入的单词 ID 转换为隐藏状态
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 定义一个位置编码层，用于添加位置信息到输入隐藏状态中
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            config.positional_dropout,
            config.hidden_size,
            config.max_text_positions,
        )

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播方法，接受输入的单词 ID 作为输入
    def forward(self, input_ids: torch.Tensor):
        # 将输入单词 ID 转换为隐藏状态
        inputs_embeds = self.embed_tokens(input_ids)
        # 添加位置编码到输入隐藏状态中
        inputs_embeds = self.encode_positions(inputs_embeds)
        # 返回处理后的隐藏状态
        return inputs_embeds


# 定义一个名为 SpeechT5TextDecoderPrenet 的类，继承自 nn.Module
class SpeechT5TextDecoderPrenet(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将参数 config 保存在 self.config 中
        self.config = config
        # 定义一个丢弃层，用于进行随机丢弃一部分隐藏状态
        self.dropout = nn.Dropout(config.positional_dropout)
        # 根据配置决定是否对嵌入进行缩放
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        # 定义一个嵌入层，将输入的单词 ID 转换为隐藏状态
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 定义一个正弦位置编码层，用于添加位置信息到输入隐藏状态中
        self.embed_positions = SpeechT5SinusoidalPositionalEmbedding(
            config.max_text_positions + config.pad_token_id + 1,
            config.hidden_size,
            config.pad_token_id,
        )

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播方法，接受输入的单词 ID、注意力遮罩和过去键值对作为输入
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        ):
            # 检查是否提供了input_ids
            if input_ids is not None:
                # 获取input_ids的形状
                input_shape = input_ids.size()
                # 重新排列input_ids的维度
                input_ids = input_ids.view(-1, input_shape[-1])
            else:
                # 如果没有提供input_ids，则抛出数值错误
                raise ValueError("You have to specify `decoder_input_ids`")

            # 如果过去的键值存在，则获取其长度，否则长度为0
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            # 通过embed_positions函数获取位置编码信息
            positions = self.embed_positions(input_ids, past_key_values_length)

            # 使用embed_tokens函数获取输入的词嵌入，再乘以嵌入缩放因子
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            # 添加位置编码信息到输入的词嵌入中
            inputs_embeds += positions
            # 对输入的词嵌入进行dropout操作
            inputs_embeds = self.dropout(inputs_embeds)

            # 返回处理后的输入嵌入信息和注意力掩码
            return inputs_embeds, attention_mask
# 定义一个名为SpeechT5TextDecoderPostnet的类
class SpeechT5TextDecoderPostnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # 前向传播函数，接受隐藏状态张量作为输入，返回线性映射后的结果
    def forward(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states)

    # 获取输出嵌入（embedding）的函数
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入（embedding）的函数
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


# 定义一个名为SpeechT5Attention的类
class SpeechT5Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with relative position bias (see
    https://aclanthology.org/N18-2074.pdf)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # 如果embed_dim不被num_heads整除，引发错误
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 重新形状张量的函数，返回由batch_size、seq_len和head数决定的张量形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,

# 定义一个名为SpeechT5FeedForward的类
class SpeechT5FeedForward(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)
    # 定义一个前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 通过intermediate_dense层处理隐藏状态
        hidden_states = self.intermediate_dense(hidden_states)
        # 通过intermediate_act_fn激活函数处理隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 通过intermediate_dropout层处理隐藏状态
        hidden_states = self.intermediate_dropout(hidden_states)

        # 通过output_dense层处理隐藏状态
        hidden_states = self.output_dense(hidden_states)
        # 通过output_dropout层处理隐藏状态
        hidden_states = self.output_dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
class SpeechT5EncoderLayer(nn.Module):
    def __init__(self, config: SpeechT5Config):
        super().__init__()
        # 初始化自注意力机制层
        self.attention = SpeechT5Attention(
            embed_dim=config.hidden_size,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 定义丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 定义层归一化层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义前馈神经网络层
        self.feed_forward = SpeechT5FeedForward(config, config.encoder_ffn_dim)
        # 定义最终的层归一化层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            position_bias (`torch.FloatTensor`):
                relative position embeddings of size `(seq_len, seq_len, hidden_size // encoder_attention_heads)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存残差连接
        residual = hidden_states
        # 调用自注意力机制层
        hidden_states, attn_weights, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        # 使用丢弃层
        hidden_states = self.dropout(hidden_states)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 使用层归一化层
        hidden_states = self.layer_norm(hidden_states)
        # 使用前馈神经网络层
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 使用最终的层归一化层
        hidden_states = self.final_layer_norm(hidden_states)
        # 输出结果
        outputs = (hidden_states,)
        # 如果需要输出注意力权重，则加入输出中
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class SpeechT5DecoderLayer(nn.Module):
    # 初始化函数，接受一个SpeechT5Config对象作为参数
    def __init__(self, config: SpeechT5Config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化自注意力机制，使用SpeechT5Attention类
        self.self_attn = SpeechT5Attention(
            embed_dim=config.hidden_size,  # 设置注意力机制中的嵌入维度
            num_heads=config.decoder_attention_heads,  # 设置注意力头的数量
            dropout=config.attention_dropout,  # 设置注意力机制的dropout率
            is_decoder=True,  # 表明这是一个解码器的注意力机制
        )
        # 初始化dropout层，使用nn.Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化自注意力层的LayerNorm层，使用nn.LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化编码器-解码器注意力机制，使用SpeechT5Attention类
        self.encoder_attn = SpeechT5Attention(
            config.hidden_size,  # 设置注意力机制中的隐藏大小
            config.decoder_attention_heads,  # 设置注意力头的数量
            dropout=config.attention_dropout,  # 设置注意力机制的dropout率
            is_decoder=True,  # 表明这是一个解码器的注意力机制
        )
        # 初始化编码器-解码器注意力层的LayerNorm层，使用nn.LayerNorm
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化前馈神经网络，使用SpeechT5FeedForward类
        self.feed_forward = SpeechT5FeedForward(config, config.decoder_ffn_dim)
        # 初始化最终的LayerNorm层，使用nn.LayerNorm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受一系列张量作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态张量，可选
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码张量，可选
        layer_head_mask: Optional[torch.Tensor] = None,  # 层级头掩码张量，可选
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力层级头掩码张量，可选
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 过去的键值对元组，可选
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，可选，默认为False
        use_cache: Optional[bool] = True,  # 是否使用缓存，可选，默认为True
```  
class SpeechT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为SpeechT5Config
    config_class = SpeechT5Config
    # 设置基础模型前缀为"speecht5"
    base_model_prefix = "speecht5"
    # 设置主要输入名称为"input_values"
    main_input_name = "input_values"
    # 启用梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果module是SpeechT5PositionalConvEmbedding类型
        if isinstance(module, SpeechT5PositionalConvEmbedding):
            # 初始化module的卷积权重
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 初始化module的卷积偏置
            nn.init.constant_(module.conv.bias, 0)
        # 如果module是SpeechT5FeatureProjection类型
        elif isinstance(module, SpeechT5FeatureProjection):
            # 计算k值
            k = math.sqrt(1 / module.projection.in_features)
            # 初始化module的投影权重
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            # 初始化module的投影偏置
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果module是nn.Linear类型
        elif isinstance(module, nn.Linear):
            # 初始化module的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，初始化module的偏置
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果module是nn.LayerNorm 或 nn.GroupNorm类型
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 初始化module的偏置为0
            module.bias.data.zero_()
            # 初始化module的权重为1
            module.weight.data.fill_(1.0)
        # 如果module是nn.Conv1d类型
        elif isinstance(module, nn.Conv1d):
            # 用Kaiming正态分布初始化module的权重
            nn.init.kaiming_normal_(module.weight)
            # 如果有偏置，初始化module的偏置
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        # 如果module是nn.Embedding类型
        elif isinstance(module, nn.Embedding):
            # 初始化module的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，初始化���充索引位置的权重为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SpeechT5Encoder(SpeechT5PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* layers. Each layer is a [`SpeechT5EncoderLayer`].
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        # 初始化LayerNorm层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 设置层丢弃概率
        self.layerdrop = config.encoder_layerdrop

        # 创建config.encoder_layers数量的编码器层
        self.layers = nn.ModuleList([SpeechT5EncoderLayer(config) for _ in range(config.encoder_layers)])

        # 初始化相对位置编码
        self.embed_positions = SpeechT5RelativePositionalEncoding(
            config.hidden_size // config.encoder_attention_heads, config.encoder_max_relative_position
        )

        # 禁用梯度检查点
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()
    # 实现 Transformer 模型的前向传播函数，接受输入的隐藏状态和注意力掩码等参数
    def forward(
        self,
        # 输入的隐藏状态，类型为 torch.FloatTensor
        hidden_states: torch.FloatTensor,
        # 注意力掩码，可选参数，默认为 None，类型为 torch.Tensor
        attention_mask: Optional[torch.Tensor] = None,
        # 头部掩码，可选参数，默认为 None，类型为 torch.Tensor
        head_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，可选参数，默认为 None，类型为 bool
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选参数，默认为 None，类型为 bool
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出，可选参数，默认为 None，类型为 bool
        return_dict: Optional[bool] = None,
# 定义一个名为SpeechT5EncoderWithSpeechPrenet的类，继承自SpeechT5PreTrainedModel类
class SpeechT5EncoderWithSpeechPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Encoder that applies SpeechT5SpeechEncoderPrenet to convert the audio waveform data to
    hidden features.
    """

    # 初始化方法，接收config参数
    def __init__(self, config: SpeechT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建SpeechT5SpeechEncoderPrenet实例并赋值给self.prenet
        self.prenet = SpeechT5SpeechEncoderPrenet(config)
        # 创建SpeechT5Encoder实例并赋值给self.wrapped_encoder
        self.wrapped_encoder = SpeechT5Encoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 使用self.prenet处理input_values和attention_mask，得到hidden_states和attention_mask
        hidden_states, attention_mask = self.prenet(input_values, attention_mask)

        # 调用self.wrapped_encoder的前向传播方法
        outputs = self.wrapped_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回输出结果
        return outputs


# 定义一个名为SpeechT5EncoderWithTextPrenet的类，继承自SpeechT5PreTrainedModel类
class SpeechT5EncoderWithTextPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Encoder that applies SpeechT5TextEncoderPrenet to convert the input_ids to hidden features.
    """

    # 初始化方法，接收config参数
    def __init__(self, config: SpeechT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建SpeechT5TextEncoderPrenet实例并赋值给self.prenet
        self.prenet = SpeechT5TextEncoderPrenet(config)
        # 创建SpeechT5Encoder实例并赋值给self.wrapped_encoder
        self.wrapped_encoder = SpeechT5Encoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        # 返回self.prenet的输入嵌入
        return self.prenet.get_input_embeddings()

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        # 设置self.prenet的输入嵌入为value
        self.prenet.set_input_embeddings(value)

    # 前向传播方法
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 使用self.prenet处理input_values，得到hidden_states
        hidden_states = self.prenet(input_values)

        # 调用self.wrapped_encoder的前向传播方法
        outputs = self.wrapped_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回输出结果
        return outputs


# 定义一个名为SpeechT5EncoderWithoutPrenet的类，继承自SpeechT5PreTrainedModel类
class SpeechT5EncoderWithoutPrenet(SpeechT5PreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when used in combination with
    [`SpeechT5Model`].
    """
    # 初始化方法，接收一个SpeechT5Config对象作为参数
    def __init__(self, config: SpeechT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个SpeechT5Encoder对象，并将其赋值给类属性wrapped_encoder
        self.wrapped_encoder = SpeechT5Encoder(config)

        # 调用post_init方法，用于初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 调用wrapped_encoder对象的前向传播方法，并返回结果
        return self.wrapped_encoder(
            hidden_states=input_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
class SpeechT5Decoder(SpeechT5PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SpeechT5DecoderLayer`]
    """

    def __init__(self, config: SpeechT5Config):
        # 调用父类的构造函数，初始化配置信息
        super().__init__(config)
        # 设置层之间的随机丢弃率
        self.layerdrop = config.decoder_layerdrop
        # 创建多个解码器层，根据配置文件中的 decoder_layers 参数
        self.layers = nn.ModuleList([SpeechT5DecoderLayer(config) for _ in range(config.decoder_layers)])
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 接收输入参数，并进行前向传播
        # 参数包括隐藏状态、注意力掩码、编码器隐藏状态、编码器注意力掩码等



class SpeechT5DecoderWithSpeechPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Decoder that applies SpeechT5SpeechDecoderPrenet to convert log-mel filterbanks to hidden
    features.
    """

    def __init__(self, config: SpeechT5Config):
        # 调用父类的构造函数，初始化配置信息
        super().__init__(config)
        # 创建 SpeechT5SpeechDecoderPrenet 对象
        self.prenet = SpeechT5SpeechDecoderPrenet(config)
        # 创建 SpeechT5Decoder 对象
        self.wrapped_decoder = SpeechT5Decoder(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeddings: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 接收输入参数，并进行前向传播
        # 参数包括输入数值、注意力掩码、编码器隐藏状态、编码器注意力掩码、说话者嵌入等
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # 使用输入值和说话者嵌入作为输入，通过预网络处理得到解码器的隐藏状态
        decoder_hidden_states = self.prenet(input_values, speaker_embeddings)

        # 调用包装后的解码器模型，传入相应参数，并接收输出
        outputs = self.wrapped_decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回解码器模型的输出
        return outputs
class SpeechT5DecoderWithTextPrenet(SpeechT5PreTrainedModel):
    """
    Wrapper around SpeechT5Decoder that applies SpeechT5TextDecoderPrenet to convert input tokens to hidden features.
    """

    def __init__(self, config: SpeechT5Config):
        # 调用父类构造函数，初始化模型配置
        super().__init__(config)
        # 初始化文本预处理器，用于将输入标记转换为隐藏特征
        self.prenet = SpeechT5TextDecoderPrenet(config)
        # 初始化包装的解码器
        self.wrapped_decoder = SpeechT5Decoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回文本预处理器的输入嵌入
        return self.prenet.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 设置文本预处理器的输入嵌入
        self.prenet.set_input_embeddings(value)

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # 使用文本预处理器处理输入值，得到解码器的隐藏状态和注意力掩码
        decoder_hidden_states, attention_mask = self.prenet(input_values, attention_mask, past_key_values)

        # 调用包装的解码器进行解码
        outputs = self.wrapped_decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class SpeechT5DecoderWithoutPrenet(SpeechT5PreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when used in combination with
    [`SpeechT5Model`].
    """

    def __init__(self, config: SpeechT5Config):
        # 调用父类构造函数，初始化模型配置
        super().__init__(config)
        # 初始化包装的解码器
        self.wrapped_decoder = SpeechT5Decoder(config)

        # 初始化权重并应用最终处理
        self.post_init()
    # 前向传播函数，接收多个输入参数，并返回模型输出结果
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,  # 输入值，可以为None
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码，可以为None
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器隐藏状态，可以为None
        encoder_attention_mask: Optional[torch.LongTensor] = None,  # 编码器注意力掩码，可以为None
        head_mask: Optional[torch.Tensor] = None,  # 注意头掩码，可以为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头掩码，可以为None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值，可以为None
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可以为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为None
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果，可以为None
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:  # 返回值类型为元组或包含过去和跨注意力的基础模型输出
        # 调用包装的解码器进行处理，传入相应参数
        outputs = self.wrapped_decoder(
            hidden_states=input_values,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 返回处理结果
        return outputs
# 定义一个名为 SpeechT5GuidedMultiheadAttentionLoss 的类，继承自 nn.Module
class SpeechT5GuidedMultiheadAttentionLoss(nn.Module):
    """
    Guided attention loss from the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional
    Networks with Guided Attention](https://arxiv.org/abs/1710.08969), adapted for multi-head attention.
    """

    # 类的初始化方法
    def __init__(self, config: SpeechT5Config):
        super().__init__()
        # 初始化参数 sigma 和 scale，分别用于计算 guided attention loss
        self.sigma = config.guided_attention_loss_sigma
        self.scale = config.guided_attention_loss_scale

    # 前向传播方法
    def forward(
        self, attentions: torch.FloatTensor, input_masks: torch.BoolTensor, output_masks: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute the attention loss.

        Args:
            attentions (`torch.FloatTensor` of shape `(batch_size, layers * heads, output_sequence_length, input_sequence_length)`):
                Batch of multi-head attention weights
            input_masks (`torch.BoolTensor` of shape `(batch_size, input_sequence_length)`):
                Input attention mask as booleans.
            output_masks (`torch.BoolTensor` of shape `(batch_size, output_sequence_length)`):
                Target attention mask as booleans.

        Returns:
            `torch.Tensor` with the loss value
        """
        # 根据输入、输出层的 attention mask 以及多头注意力权重计算 guided attention masks
        guided_attn_masks = self._make_guided_attention_masks(input_masks, output_masks, attentions.device)
        # 将输入、输出层的 attention masks 转换为张量
        masks = output_masks.unsqueeze(-1) & input_masks.unsqueeze(-2)
        masks = masks.to(attentions.device).unsqueeze(1)

        # 根据 guided attention masks 和 attentions 计算损失
        losses = guided_attn_masks * attentions
        loss = torch.mean(losses.masked_select(masks))
        return self.scale * loss

    # 构造 guided attention masks 的内部方法
    def _make_guided_attention_masks(self, input_masks, output_masks, device):
        input_lengths = input_masks.sum(-1)
        output_lengths = output_masks.sum(-1)

        # 创建一个全零张量作为 guided attention masks 的初始值
        guided_attn_masks = torch.zeros((len(input_masks), output_masks.shape[1], input_masks.shape[1]), device=device)

        for idx, (ilen, olen) in enumerate(zip(input_lengths, output_lengths)):
            # 根据输入、输出层的长度和 sigma 构造 guided attention mask
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma, device)

        return guided_attn_masks.unsqueeze(1)

    # 构造 guided attention mask 的静态方法
    @staticmethod
    def _make_guided_attention_mask(input_length, output_length, sigma, device):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(input_length, device=device),
            torch.arange(output_length, device=device),
            indexing="xy",
        )
        grid_x = grid_x.float() / output_length
        grid_y = grid_y.float() / input_length
        # 根据公式计算 guided attention mask
        return 1.0 - torch.exp(-((grid_y - grid_x) ** 2) / (2 * (sigma**2)))


class SpeechT5SpectrogramLoss(nn.Module):
    """
    Loss computation used by SpeechT5ForTextToSpeech.
    """
    # 初始化函数，接受一个配置参数，并调用父类的初始化函数
    def __init__(self, config: SpeechT5Config):
        super().__init__()
        # 设置是否使用引导注意力损失的标志
        self.use_guided_attention_loss = config.use_guided_attention_loss
        # 设置引导注意力损失的头数
        self.guided_attention_loss_num_heads = config.guided_attention_loss_num_heads
        # 设置减少因子
        self.reduction_factor = config.reduction_factor

        # 初始化 L1 损失函数
        self.l1_criterion = L1Loss()
        # 初始化二元交叉熵损失函数
        self.bce_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))

        # 如果使用引导注意力损失，则初始化 SpeechT5GuidedMultiheadAttentionLoss 类
        if self.use_guided_attention_loss:
            self.attn_criterion = SpeechT5GuidedMultiheadAttentionLoss(config)

    # 前向传播函数，接受多个张量参数，返回一个张量
    def forward(
        self,
        attention_mask: torch.LongTensor,
        outputs_before_postnet: torch.FloatTensor,
        outputs_after_postnet: torch.FloatTensor,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        cross_attentions: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 根据标签的填充情况生成填充蒙版
        padding_mask = labels != -100.0

        # 根据填充蒙版过滤标签、前后处理前输出
        labels = labels.masked_select(padding_mask)
        outputs_before_postnet = outputs_before_postnet.masked_select(padding_mask)
        outputs_after_postnet = outputs_after_postnet.masked_select(padding_mask)

        # 计算 L1 损失（平均绝对误差损失）
        l1_loss = self.l1_criterion(outputs_after_postnet, labels) + self.l1_criterion(outputs_before_postnet, labels)

        # 构建停止标签
        masks = padding_mask[:, :, 0]
        stop_labels = torch.cat([~masks * 1.0, torch.ones(masks.size(0), 1).to(masks.device)], dim=1)
        stop_labels = stop_labels[:, 1:].masked_select(masks)
        logits = logits.masked_select(masks)

        # 计算二元交叉熵损失
        bce_loss = self.bce_criterion(logits, stop_labels)

        # 组合 L1 损失和二元交叉熵损失
        loss = l1_loss + bce_loss

        # 如果使用引导注意力损失，则计算引导注意力损失
        if self.use_guided_attention_loss:
            attn = torch.cat([x[:, : self.guided_attention_loss_num_heads] for x in cross_attentions], dim=1)
            input_masks = attention_mask == 1
            output_masks = padding_mask[:, :, 0]
            if self.reduction_factor > 1:
                output_masks = output_masks[:, self.reduction_factor - 1 :: self.reduction_factor]
            attn_loss = self.attn_criterion(attn, input_masks, output_masks)
            # 将引导注意力损失加到总损失上
            loss += attn_loss

        return loss
# 定义一个字符串，用于存储 SpeechT5 模型的基本描述和文档信息
SPEECHT5_BASE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        encoder ([`SpeechT5EncoderWithSpeechPrenet`] or [`SpeechT5EncoderWithTextPrenet`] or `None`):
            The Transformer encoder module that applies the appropiate speech or text encoder prenet. If `None`,
            [`SpeechT5EncoderWithoutPrenet`] will be used and the `input_values` are assumed to be hidden states.
        decoder ([`SpeechT5DecoderWithSpeechPrenet`] or [`SpeechT5DecoderWithTextPrenet`] or `None`):
            The Transformer decoder module that applies the appropiate speech or text decoder prenet. If `None`,
            [`SpeechT5DecoderWithoutPrenet`] will be used and the `decoder_input_values` are assumed to be hidden
            states.
"""

# 定义一个字符串，用于存储 SpeechT5 模型的基本描述和文档信息
SPEECHT5_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SpeechT5Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个字符串，用于存储 SpeechT5 模型的输入描述和文档信息
SPEECHT5_INPUTS_DOCSTRING = r"""
"""

# 装饰器，用于向 SpeechT5Model 添加描述信息
@add_start_docstrings(
    "The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.",
    SPEECHT5_BASE_START_DOCSTRING,
)
# 定义 SpeechT5Model 类，继承自 SpeechT5PreTrainedModel
class SpeechT5Model(SpeechT5PreTrainedModel):
    # 初始化方法
    def __init__(
        self,
        config: SpeechT5Config,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        # 调用父类初始化函数，并传入配置参数
        super().__init__(config)
        # 将配置参数存储到类属性中
        self.config = config
        # 如果给定了编码器，则使用给定的编码器，否则创建一个新的SpeechT5EncoderWithoutPrenet对象
        self.encoder = SpeechT5EncoderWithoutPrenet(config) if encoder is None else encoder
        # 如果给定了解码器，则使用给定的解码器，否则创建一个新的SpeechT5DecoderWithoutPrenet对象
        self.decoder = SpeechT5DecoderWithoutPrenet(config) if decoder is None else decoder

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        # 如果编码器是SpeechT5EncoderWithTextPrenet的实例，则返回其输入嵌入
        if isinstance(self.encoder, SpeechT5EncoderWithTextPrenet):
            return self.encoder.get_input_embeddings()
        # 如果解码器是SpeechT5DecoderWithTextPrenet的实例，则返回其输入嵌入
        if isinstance(self.decoder, SpeechT5DecoderWithTextPrenet):
            return self.decoder.get_input_embeddings()
        # 否则返回空值
        return None

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        # 如果编码器是SpeechT5EncoderWithTextPrenet的实例，则设置其输入嵌入
        if isinstance(self.encoder, SpeechT5EncoderWithTextPrenet):
            self.encoder.set_input_embeddings(value)
        # 如果解码器是SpeechT5DecoderWithTextPrenet的实例，则设置其输入嵌入
        if isinstance(self.decoder, SpeechT5DecoderWithTextPrenet):
            self.decoder.set_input_embeddings(value)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 冻结特征编码器的参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 如果编码器是SpeechT5EncoderWithSpeechPrenet的实例，则冻结特征编码器的参数
        if isinstance(self.encoder, SpeechT5EncoderWithSpeechPrenet):
            self.encoder.prenet.freeze_feature_encoder()

    # 前向传播函数，根据给定参数计算模型的前向传播
    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)  # 添加模型前向传播的描述注释
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 为SpeechT5ForSpeechToText类添加文档字符串
@add_start_docstrings(
    """SpeechT5 Model with a speech encoder and a text decoder.""",
    SPEECHT5_START_DOCSTRING,
)
class SpeechT5ForSpeechToText(SpeechT5PreTrainedModel):
    _tied_weights_keys = ["text_decoder_postnet.lm_head.weight"]

    # 初始化函数，接受一个SpeechT5Config对象作为参数
    def __init__(self, config: SpeechT5Config):
        # 调用父类SpeechT5PreTrainedModel的初始化函数
        super().__init__(config)

        # 如果配置中未定义词汇表大小，则引发值错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that does not define the"
                " vocabulary size of the language model head. Please instantiate the model as follows:"
                " `SpeechT5ForSpeechToText.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of"
                " your model's configuration."
            )

        # 创建语音编码器和文本解码器对象
        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        text_decoder = SpeechT5DecoderWithTextPrenet(config)
        # 创建SpeechT5Model对象，传入语音编码器和文本解码器
        self.speecht5 = SpeechT5Model(config, speech_encoder, text_decoder)

        # 创建文本解码器后处理对象
        self.text_decoder_postnet = SpeechT5TextDecoderPostnet(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器对象
    def get_encoder(self):
        return self.speecht5.get_encoder()

    # 获取解码器对象
    def get_decoder(self):
        return self.speecht5.get_decoder()

    # 冻结特征编码器的梯度计算
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.text_decoder_postnet.get_output_embeddings()

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.text_decoder_postnet.set_output_embeddings(new_embeddings)

    # 前向传播函数
    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        ):
    # 为生成准备输入，包含解码器的输入 ID、过去的键值、注意力掩码、头部掩码、解码器头部掩码、交叉注意头部掩码、使用缓存等
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值
        if past_key_values is not None:
            # 获取过去的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后的输入 ID
            if decoder_input_ids.shape[1] > past_length:
                # 根据过去的长度截断解码器的输入
                remove_prefix_length = past_length
            else:
                # 默认行为：只保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 更新解码器的输入 ID
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回包含各种输入准备参数的字典
        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项可以避免缓存（可能是为了调试）
        }

    # 重新排序缓存
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序过去的键值
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                # 对每个层的过去状态根据 beam_idx 进行索引选择
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 生成语音的函数，接受模型、输入值和其他可选参数，返回生成的声音数据
def _generate_speech(
    model: SpeechT5PreTrainedModel,  # 期望的模型类型
    input_values: torch.FloatTensor,  # 输入值的张量类型
    speaker_embeddings: Optional[torch.FloatTensor] = None,  # 可选的说话者嵌入向量
    attention_mask: Optional[torch.LongTensor] = None,  # 可选的注意力掩码
    threshold: float = 0.5,  # 阈值
    minlenratio: float = 0.0,  # 最小长度比例
    maxlenratio: float = 20.0,  # 最大长度比例
    vocoder: Optional[nn.Module] = None,  # 可选的声码器
    output_cross_attentions: bool = False,  # 是否输出交叉注意力
    return_output_lengths: bool = False,  # 是否返回输出长度
) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:  # 返回值可能是张量或张量组成的元组
    if speaker_embeddings is None:
        raise ValueError(
            """`speaker_embeddings` must be specified. For example, you can use a speaker embeddings by following
                    the code snippet provided in this link:
                    https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors
                    """
        )

    if attention_mask is None:
        # 计算输入值的注意力掩码
        encoder_attention_mask = 1 - (input_values == model.config.pad_token_id).int()
    else:
        encoder_attention_mask = attention_mask

    bsz = input_values.size(0)  # 获取输入值的批量大小

    # 使用模型的编码器对输入进行编码，获取编码器输出
    encoder_out = model.speecht5.encoder(
        input_values=input_values,
        attention_mask=encoder_attention_mask,
        return_dict=True,
    )

    encoder_last_hidden_state = encoder_out.last_hidden_state  # 获取编码器的最后隐藏状态

    # 下采样编码器的注意力掩码
    if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
        encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
            encoder_out[0].shape[1], encoder_attention_mask
        )

    maxlen = int(encoder_last_hidden_state.size(1) * maxlenratio / model.config.reduction_factor)  # 计算最大长度
    minlen = int(encoder_last_hidden_state.size(1) * minlenratio / model.config.reduction_factor)  # 计算最小长度

    # 以全零的梅尔频谱开始输出序列
    output_sequence = encoder_last_hidden_state.new_zeros(bsz, 1, model.config.num_mel_bins)

    spectrogram = []  # 初始化谱图列表
    cross_attentions = []  # 初始化交叉注意力列表
    past_key_values = None  # 初始化过去的键值对
    idx = 0  # 初始化索引
    result_spectrogram = {}  # 初始化结果谱图字典
    while True:
        idx += 1  # 索引加一，表示处理下一个步骤

        # 在整个输出序列上运行解码器的预处理网络
        decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)
        # 在预处理网络输出的最后一个元素上运行解码器层
        decoder_out = model.speecht5.decoder.wrapped_decoder(
            hidden_states=decoder_hidden_states[:, -1:],  # 传入最后一个时间步的隐藏状态
            attention_mask=None,  # 注意力掩码为空，表示没有额外的注意力限制
            encoder_hidden_states=encoder_last_hidden_state,  # 编码器的最后隐藏状态
            encoder_attention_mask=encoder_attention_mask,  # 编码器的注意力掩码
            past_key_values=past_key_values,  # 过去的键值对，用于缓存
            use_cache=True,  # 使用缓存，以加速解码
            output_attentions=output_cross_attentions,  # 输出注意力权重
            return_dict=True,  # 返回字典形式的结果
        )

        if output_cross_attentions:
            cross_attentions.append(torch.cat(decoder_out.cross_attentions, dim=0))  # 收集交叉注意力权重

        last_decoder_output = decoder_out.last_hidden_state.squeeze(1)  # 获取最后一个解码器的输出
        past_key_values = decoder_out.past_key_values  # 更新过去的键值对

        # 预测当前步骤的新的梅尔频谱
        spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
        spectrum = spectrum.view(bsz, model.config.reduction_factor, model.config.num_mel_bins)  # 调整形状
        spectrogram.append(spectrum)  # 将梅尔频谱添加到列表中

        # 将输出序列扩展到新的梅尔频谱
        new_spectrogram = spectrum[:, -1, :].view(bsz, 1, model.config.num_mel_bins)
        output_sequence = torch.cat((output_sequence, new_spectrogram), dim=1)  # 将新的梅尔频谱连接到输出序列的末尾
        # 预测这是否是停止令牌的概率
        prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))

        if idx < minlen:  # 如果索引小于最小长度，继续下一个循环
            continue
        else:
            # 如果生成循环少于最大长度，检查批次中已经满足概率阈值的样本。
            # 否则，假定所有样本都满足阈值，并填充批次中的其他梅尔频谱。
            if idx < maxlen:
                meet_thresholds = torch.sum(prob, dim=-1) >= threshold  # 检查概率是否达到阈值
                meet_indexes = torch.where(meet_thresholds)[0].tolist()  # 获取满足阈值的样本索引列表
            else:
                meet_indexes = range(len(prob))  # 否则，假定所有样本都满足阈值
            meet_indexes = [i for i in meet_indexes if i not in result_spectrogram]  # 过滤已经处理过的索引
            if len(meet_indexes) > 0:
                # 对满足阈值的样本应用后处理网络，并填充结果梅尔频谱
                spectrograms = torch.stack(spectrogram)
                spectrograms = spectrograms.transpose(0, 1).flatten(1, 2)
                spectrograms = model.speech_decoder_postnet.postnet(spectrograms)
                for meet_index in meet_indexes:
                    result_spectrogram[meet_index] = spectrograms[meet_index]
            if len(result_spectrogram) >= bsz:  # 如果结果梅尔频谱的数量达到批次大小，跳出循环
                break
    spectrograms = [result_spectrogram[i] for i in range(len(result_spectrogram))]  # 将结果梅尔频谱转换为列表形式
    # 如果不需要返回输出长度
    if not return_output_lengths:
        # 如果 batch size 为 1，则直接选择第一个 spectrogram，否则进行 pad 操作
        spectrogram = spectrograms[0] if bsz == 1 else torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        # 如果 vocoder 存在，则使用 vocoder 处理 spectrogram，否则直接使用 spectrogram
        if vocoder is not None:
            outputs = vocoder(spectrogram)
        else:
            outputs = spectrogram
        # 如果需要输出交叉注意力的结果，则拼接多个 cross_attentions
        if output_cross_attentions:
            cross_attentions = torch.cat(cross_attentions, dim=2)
            # 如果 batch size 大于 1，则重新调整 cross_attentions 的形状
            if bsz > 1:
                cross_attentions = cross_attentions.view(
                    bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
                )
            # 输出结果为 outputs 和 cross_attentions
            outputs = (outputs, cross_attentions)
    # 如果需要返回输出长度
    else:
        # 计算每个 spectrogram 的长度并存储在列表中
        spectrogram_lengths = []
        for i in range(bsz):
            spectrogram_lengths.append(spectrograms[i].size(0))
        # 如果 vocoder 不存在，则对 spectrograms 进行 pad，并且输出结果包括 spectrograms 和 spectrogram_lengths
        if vocoder is None:
            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
            outputs = (spectrograms, spectrogram_lengths)
        # 如果 vocoder 存在
        else:
            waveforms = []
            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
            # 使用 vocoder 处理 spectrograms，并计算每个 waveform 的长度
            waveforms = vocoder(spectrograms)
            waveform_lengths = [int(waveforms.size(1) / max(spectrogram_lengths)) * i for i in spectrogram_lengths]
            # 输出结果为 waveforms 和 waveform_lengths
            outputs = (waveforms, waveform_lengths)
        # 如果需要输出交叉注意力的结果，则拼接多个 cross_attentions
        if output_cross_attentions:
            cross_attentions = torch.cat(cross_attentions, dim=2)
            cross_attentions = cross_attentions.view(
                bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
            )
            # 输出结果为 outputs、cross_attentions
            outputs = (*outputs, cross_attentions)
    # 返回最终的 outputs
    return outputs
# 定义了一个名为SpeechT5ForTextToSpeech的类，继承自SpeechT5PreTrainedModel类
@add_start_docstrings(
    """SpeechT5 Model with a text encoder and a speech decoder.""",
    SPEECHT5_START_DOCSTRING,
)
class SpeechT5ForTextToSpeech(SpeechT5PreTrainedModel):
    # 类属性，表示主要输入的名称为input_ids
    main_input_name = "input_ids"

    # 初始化函数，接受一个SpeechT5Config类型的config对象作为参数
    def __init__(self, config: SpeechT5Config):
        # 调用父类SpeechT5PreTrainedModel的初始化函数
        super().__init__(config)

        # 如果config对象中的vocab_size属性为None，则抛出ValueError异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that does not define the"
                " vocabulary size of the language model head. Please instantiate the model as follows:"
                " `SpeechT5ForTextToSpeech.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of"
                " your model's configuration."
            )

        # 创建一个SpeechT5EncoderWithTextPrenet类型的对象，并赋值给text_encoder变量
        text_encoder = SpeechT5EncoderWithTextPrenet(config)
        # 创建一个SpeechT5DecoderWithSpeechPrenet类型的对象，并赋值给speech_decoder变量
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        # 创建一个SpeechT5Model类型的对象，并赋值给self.speecht5变量
        self.speecht5 = SpeechT5Model(config, text_encoder, speech_decoder)

        # 创建一个SpeechT5SpeechDecoderPostnet类型的对象，并赋值给self.speech_decoder_postnet变量
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)

        # 调用post_init方法，完成权重的初始化和最终处理
        self.post_init()

    # 返回text encoder部分的模型
    def get_encoder(self):
        return self.speecht5.get_encoder()

    # 返回speech decoder部分的模型
    def get_decoder(self):
        return self.speecht5.get_decoder()

    # 重写forward方法，接受多个参数，包括输入、注意力掩码、解码器输入值等，返回模型的输出
    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSpectrogramOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.Tensor] = None,
    ):
        # 在没有梯度的情况下，调用forward方法
        @torch.no_grad()
        def generate(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            speaker_embeddings: Optional[torch.FloatTensor] = None,
            threshold: float = 0.5,
            minlenratio: float = 0.0,
            maxlenratio: float = 20.0,
            vocoder: Optional[nn.Module] = None,
            output_cross_attentions: bool = False,
            return_output_lengths: bool = False,
            **kwargs,
        ):
            # 在没有梯度的情况下，调用generate方法
            @torch.no_grad()
    # 生成语音内容的方法
    def generate_speech(
        # 输入的 ID 序列，类型为长整型张量
        self,
        input_ids: torch.LongTensor,
        # 说话者的嵌入表示，可选的浮点张量，默认为 None
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        # 注意力遮罩，可选的长整型张量，默认为 None
        attention_mask: Optional[torch.LongTensor] = None,
        # 阈值，类型为浮点数，默认为 0.5
        threshold: float = 0.5,
        # 最小长度比例，类型为浮点数，默认为 0.0
        minlenratio: float = 0.0,
        # 最大长度比例，类型为浮点数，默认为 20.0
        maxlenratio: float = 20.0,
        # 语音合成器，可选的神经网络模块，默认为 None
        vocoder: Optional[nn.Module] = None,
        # 输出交叉注意力信息的标志，默认为 False
        output_cross_attentions: bool = False,
        # 返回输出长度的标志，默认为 False
        return_output_lengths: bool = False,
# 为 SpeechT5ForSpeechToSpeech 类添加文档字符串，描述其含义
@add_start_docstrings(
    """SpeechT5 Model with a speech encoder and a speech decoder.""",
    SPEECHT5_START_DOCSTRING,
)
# 定义类 SpeechT5ForSpeechToSpeech，继承自 SpeechT5PreTrainedModel
class SpeechT5ForSpeechToSpeech(SpeechT5PreTrainedModel):
    # 初始化函数，接受一个 SpeechT5Config 类型的参数 config
    def __init__(self, config: SpeechT5Config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建一个 speech encoder
        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        # 创建一个 speech decoder
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        # 创建一个 SpeechT5Model 对象，包含 encoder 和 decoder
        self.speecht5 = SpeechT5Model(config, speech_encoder, speech_decoder)

        # 创建一个 speech decoder postnet
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)

        # 初始化权重并进行最终处理
        # 调用 post_init 函数进行初始化
        self.post_init()

    # 获取 encoder
    def get_encoder(self):
        return self.speecht5.get_encoder()

    # 获取 decoder
    def get_decoder(self):
        return self.speecht5.get_decoder()

    # 冻结 feature encoder
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSpectrogramOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # 输入数据
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.Tensor] = None,
        
    # 生成语音的函数
    @torch.no_grad()
    def generate_speech(
        self,
        # 输入数据
        input_values: torch.FloatTensor,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        # 阈值
        threshold: float = 0.5,
        # 最小长度比例
        minlenratio: float = 0.0,
        # 最大长度比例
        maxlenratio: float = 20.0,
        # 声码器
        vocoder: Optional[nn.Module] = None,
        # 输出交叉注意力
        output_cross_attentions: bool = False,
        # 返回输出长度
        return_output_lengths: bool = False,
)
# HIFIGAN_START_DOCSTRING 的文档字符串
HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    # 这个模型也是一个 PyTorch 的 torch.nn.Module 子类
    # 可以像常规的 PyTorch 模块一样使用，并参考 PyTorch 文档了解一切与一般使用和行为相关的事项

    # 参数:
    #     config ([`SpeechT5HifiGanConfig`]):
    #         模型配置类，包含模型的所有参数。使用配置文件初始化不会加载模型相关的权重，只会加载配置信息。
    #         参考 `~PreTrainedModel.from_pretrained` 方法加载模型权重。
# 定义一个名为 HifiGanResidualBlock 的类，继承自 nn.Module
class HifiGanResidualBlock(nn.Module):
    # 构造函数，初始化残差块
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # 创建包含多个卷积层的列表，每个卷积层的参数都是不同的
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        # 创建包含多个卷积层的列表，每个卷积层的参数都是相同的
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    # 计算卷积层的填充值
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 对残差块中的卷积层应用权重归一化
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除残差块中的卷积层的权重归一化
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    # 前向传播函数，对输入的音频数据执行残差块的卷积操作
    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


# SpeechT5HifiGan 类，继承自 PreTrainedModel 类
@add_start_docstrings(
    """HiFi-GAN vocoder.""",
    HIFIGAN_START_DOCSTRING,
)
class SpeechT5HifiGan(PreTrainedModel):
    # 使用 SpeechT5HifiGanConfig 配置类进行配置
    config_class = SpeechT5HifiGanConfig
    # 主要输入名称为 "spectrogram"
    main_input_name = "spectrogram"
    def __init__(self, config: SpeechT5HifiGanConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 计算残差块的数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        # 计算上采样层的数量
        self.num_upsamples = len(config.upsample_rates)
        # 创建一个1维卷积层作为模型的初始层
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # 创建上采样模块列表
        self.upsampler = nn.ModuleList()
        # 遍历上采样率和卷积核大小的组合
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            # 添加一个转置卷积层到上采样模块列表
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        # 创建残差块模块列表
        self.resblocks = nn.ModuleList()
        # 遍历上采样模块列表中的每个上采样层
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            # 遍历残差块的卷积核大小和膨胀率
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                # 添加一个HiFi-GAN残差块到残差块模块列表
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 创建一个1维卷积层作为模型的最终层
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        # 注册缓冲区用于存储平均值和尺度
        self.register_buffer("mean", torch.zeros(config.model_in_dim))
        self.register_buffer("scale", torch.ones(config.model_in_dim))

        # 初始化权重并应用最终处理
        self.post_init()

    def _init_weights(self, module):
        """初始化权重。"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 将偏置初始化为零
                module.bias.data.zero_()

    def apply_weight_norm(self):
        # 对预卷积层应用权重归一化
        nn.utils.weight_norm(self.conv_pre)
        # 对上采样层列表中的每一层应用权重归一化
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        # 对残差块列表中的每一块应用权重归一化
        for layer in self.resblocks:
            layer.apply_weight_norm()
        # 对后卷积层应用权重归一化
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        # 移除预卷积层的权重归一化
        nn.utils.remove_weight_norm(self.conv_pre)
        # 移除上采样层列表中的每一层的权重归一化
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        # 移除残差块列表中的每一块的权重归一化
        for layer in self.resblocks:
            layer.remove_weight_norm()
        # 移除后卷积层的权重归一化
        nn.utils.remove_weight_norm(self.conv_post)
        def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
            r"""
            Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
            of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
            waveform.

            Args:
                spectrogram (`torch.FloatTensor`):
                    Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                    config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

            Returns:
                `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
                shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
            """
            # 如果需要在进行转换前进行规范化
            if self.config.normalize_before:
                # 对log-mel频谱进行规范化，减去均值self.mean并除以标准差self.scale
                spectrogram = (spectrogram - self.mean) / self.scale

            # 判断输入频谱是否为batched
            is_batched = spectrogram.dim() == 3
            if not is_batched:
                # 如果不是batched，则在第0维度增加一维度，使其成为batched
                spectrogram = spectrogram.unsqueeze(0)

            # 将频谱转置，将维度2和维度1互换
            hidden_states = spectrogram.transpose(2, 1)

            # 应用conv_pre卷积层
            hidden_states = self.conv_pre(hidden_states)
            # 循环进行上采样
            for i in range(self.num_upsamples):
                hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
                hidden_states = self.upsampler[i](hidden_states)

                # 应用残差块
                res_state = self.resblocks[i * self.num_kernels](hidden_states)
                for j in range(1, self.num_kernels):
                    res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
                hidden_states = res_state / self.num_kernels

            hidden_states = nn.functional.leaky_relu(hidden_states)
            # 应用conv_post卷积层
            hidden_states = self.conv_post(hidden_states)
            # 使用tanh函数激活隐藏状态
            hidden_states = torch.tanh(hidden_states)

            if not is_batched:
                # 移除批量维度并压缩张量为1维音频波形
                waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)
            else:
                # 移除序列长度维度，因为压缩到1维
                waveform = hidden_states.squeeze(1)

            return waveform
```