# `.\transformers\models\seamless_m4t\modeling_seamless_m4t.py`

```
# 设置文件编码为utf-8
# 版权信息，版权归The HuggingFace Inc.团队所有
# 根据Apache许可证2.0，只有遵守许可证规定才能使用该文件
# 许可证详细信息可在http://www.apache.org/licenses/LICENSE-2.0获得
# 未经适用法律要求或书面同意，根据许可证分发的软件是基于"AS IS"基础上分发的，不带任何担保或条件，无论是明示或默示的
# 请查看许可证以获取具体语言的权限和限制
""" PyTorch SeamlessM4T model."""

# 引入所需模块
import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

# 引入其他库函数
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 引入SeamlessM4T的配置
from .configuration_seamless_m4t import SeamlessM4TConfig

# 获取logger对象
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/hf-seamless-m4t-medium"
_CONFIG_FOR_DOC = "SeamlessM4TConfig"

# 可下载的预训练模型列表
SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/hf-seamless-m4t-medium",
    # 可在https://huggingface.co/models?filter=seamless_m4t 查看所有SeamlessM4T模型
]

SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_hifigan": "https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/config.json",
}

# 定义生成输出类
@dataclass
class SeamlessM4TGenerationOutput(ModelOutput):
    """
    Class defining the generated outputs from [`SeamlessM4TModel`], [`SeamlessM4TForTextToText`],
    [`SeamlessM4TForTextToSpeech`], [`SeamlessM4TForSpeechToSpeech`] and [`SeamlessM4TForTextToSpeech`].
    # 定义函数参数，表示预测模型输出的音频波形
    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
        # 可选参数，表示每个 `waveform` 批次元素的样本长度
        waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
            The length in samples of each element in the `waveform` batch.
        # 可选参数，表示生成的翻译序列，是文本到文本或语音到文本模型的输出
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The generated translated sequences. This is the output of the text-to-text or the speech-to-text models.
            The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished
            early due to the `eos_token_id`.
        # 可选参数，表示生成的单元序列，是文本到单元模型的输出
        unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            The generated translated unit sequences. This is the output of the text-to-units model. The second
            dimension (unit_sequence_length) is either equal to `t2u_max_length` or shorter if all batches finished
            early due to the `t2u_eos_token_id`.
    """

    # 初始化波形变量，可为空
    waveform: Optional[torch.FloatTensor] = None
    # 初始化波形长度变量，可为空
    waveform_lengths: Optional[torch.IntTensor] = None
    # 初始化文本到文本或语音到文本模型生成的序列变量，可为空
    sequences: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化文本到单元模型生成的序列变量，可为空
    unit_sequences: Optional[Tuple[torch.FloatTensor]] = None
# SeamlessM4T 模型文档字符串的起始部分，提供了关于该模型的一般说明和参数信息
SEAMLESS_M4T_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# SeamlessM4T 模型文档字符串的输入部分的第一部分，描述了输入参数中的 input_ids
SEAMLESS_M4T_INPUTS_DOCSTRING_FIRST_PART = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returned by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
    """

# SeamlessM4T 模型文档字符串的输入部分的文本部分，描述了输入参数中的 input_ids
SEAMLESS_M4T_INPUTS_DOCSTRING_TEXT_PART = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """

# SeamlessM4T 模型文档字符串的输入部分的语音部分，描述了输入参数中的 input_features
SEAMLESS_M4T_INPUTS_DOCSTRING_SPEECH_PART = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returned by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
        """

# 构建模型输入文档字符串，将不同部分拼接在一起
M4T_MODEL_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_FIRST_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART

# 文本输入文档字符串，描述了文本输入参数
M4T_TEXT_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_TEXT_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART

# 语音输入文档字符串，描述了语音输入参数
M4T_SPEECH_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_SPEECH_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART


############ UTILS ################


# 从输入的 input_ids 创建位置信息，替换非填充符号为它们的位置数字
# 位置数字从填充索引 padding_idx+1 开始，填充符号为 padding_idx
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    # 这一系列的类型转换和强制转换是经过精心平衡的，旨在同时适用于 ONNX 导出和 XLA。
    # 创建一个掩码，标记非填充索引的位置为1，填充索引的位置为0
    mask = input_ids.ne(padding_idx).int()
    # 计算累积和，每个位置的值为之前所有非填充位置的数量，乘以掩码确保填充位置保持0
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将结果转换为长整型，并加上填充索引，以便正确处理填充位置
    return incremental_indices.long() + padding_idx
# 将输入 id 向右移动一个位置
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 创建一个新的张量与输入 id 具有相同的形状
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入 id 向右移动一个位置，第一个元素设为 decoder_start_token_id
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果 pad_token_id 未定义，抛出异常
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将 shifted_input_ids 中值为 -100 的元素替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    # 返回修改后的 shifted_input_ids
    return shifted_input_ids


# 计算新的注意力掩码
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    # 获取 batch 大小和序列长度
    batch_size, mask_seq_len = hidden_states.shape[:2]

    # 创建一个从 0 到 mask_seq_len-1 的张量，并扩展到 batch 大小
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    # 根据 seq_lens 创建一个布尔掩码，表示哪些位置需要被屏蔽
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    # 创建一个全 1 的注意力掩码张量
    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    # 将需要被屏蔽的位置设为 0
    mask = mask.masked_fill(bool_mask, 0)

    # 返回注意力掩码张量
    return mask


# 格式化语音生成参数
def format_speech_generation_kwargs(kwargs):
    # 初始化用于文本生成和语音生成的参数字典
    kwargs_text = {}
    kwargs_speech = {}

    # 根据参数名称将参数分配到相应的字典中
    # 没有前缀的参数会传递给两个模型
    # 有 text_ 前缀的参数会传递给文本生成模型
    # 有 speech_ 前缀的参数会传递给语音生成模型
    # 遍历关键字参数字典中的键值对
    for key, value in kwargs.items():
        # 如果键以"text_"开头
        if key.startswith("text_"):
            # 截取"text_"后面的部分作为新的键
            key = key[len("text_") :]
            # 将对应的值存入kwargs_text字典
            kwargs_text[key] = value
        # 如果键以"speech_"开头
        elif key.startswith("speech_"):
            # 截取"speech_"后面的部分作为新的键
            key = key[len("speech_") :]
            # 将对应的值存入kwargs_speech字典
            kwargs_speech[key] = value
        else:
            # 如果键不以"text_"或"speech_"开头
            # 如果键不在kwargs_text中，将其添加到kwargs_text中
            if key not in kwargs_text:
                kwargs_text[key] = value
            # 如果键不在kwargs_speech中，将其添加到kwargs_speech中
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    # 返回两个字典作为结果
    return kwargs_text, kwargs_speech
############ SPEECH ENCODER related code ################

# 定义一个自定义的位置卷积嵌入模块，用于语音编码器
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding复制而来，将Wav2Vec2改为SeamlessM4TConformer，feat_extract_activation改为speech_encoder_hidden_act
class SeamlessM4TConformerPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个 1D 卷积层，用于位置嵌入
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 使用权重归一化
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        if is_deepspeed_zero3_enabled():  # 检查是否启用了深度速度优化
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):  # 收集权重参数
                self.conv = weight_norm(self.conv, name="weight", dim=2)  # 对卷积层进行权重归一化
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)  # 注册额外的参数
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)  # 注册额外的参数
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)  # 对卷积层进行权重归一化

        # 创建一个填充层
        self.padding = SeamlessM4TConformerSamePadLayer(config.num_conv_pos_embeddings)
        # 激活函数
        self.activation = ACT2FN[config.speech_encoder_hidden_act]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)  # 调整张量形状

        hidden_states = self.conv(hidden_states)  # 进行卷积操作
        hidden_states = self.padding(hidden_states)  # 进行填充操作
        hidden_states = self.activation(hidden_states)  # 应用激活函数

        hidden_states = hidden_states.transpose(1, 2)  # 调整张量形状
        return hidden_states


# 从transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRotaryPositionalEmbedding复制而来，将Wav2Vec2改为SeamlessM4T，num_attention_heads改为speech_encoder_attention_heads
class SeamlessM4TConformerRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.speech_encoder_attention_heads  # 计算维度
        base = config.rotary_embedding_base  # 基础值

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # 计算频率
        self.register_buffer("inv_freq", inv_freq)  # 注册缓冲区
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None
    # 定义前向传播方法，用于计算旋转位置编码的嵌入
    def forward(self, hidden_states):
        # 获取隐藏状态的序列长度
        sequence_length = hidden_states.shape[1]

        # 如果序列长度与缓存的序列长度相同，并且已经有缓存的旋转位置编码，则直接返回缓存的旋转位置编码
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        # 更新缓存的序列长度为当前序列长度
        self.cached_sequence_length = sequence_length
        # 生成时间戳序列，数据类型与inv_freq常数相同
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        # 计算频率矩阵
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        # 将频率矩阵拼接成嵌入矩阵
        embeddings = torch.cat((freqs, freqs), dim=-1)

        # 计算嵌入矩阵的余弦值
        cos_embeddings = embeddings.cos()[:, None, None, :]
        # 计算嵌入矩阵的正弦值
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # 将计算得到的余弦值和正弦值嵌入堆叠在一起，作为旋转位置编码的嵌入
        # 并将其数据类型转换为隐藏状态输入的数据类型
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        # 返回旋转位置编码的嵌入
        return self.cached_rotary_positional_embedding
# 实现相对位置编码的PyTorch模块
class SeamlessM4TConformerRelPositionalEmbedding(nn.Module):
    """相对位置编码模块。"""

    def __init__(self, config):
        super().__init__()
        # 最大位置长度
        self.max_len = config.max_source_positions
        # 隐藏层大小
        self.d_model = config.hidden_size
        # 位置编码张量
        self.pe = None
        # 扩展位置编码
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        # 如果已经有位置编码张量，且长度足够
        if self.pe is not None:
            # self.pe包含正负两部分
            # self.pe的长度是2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                # 如果设备或数据类型不匹配，转换之
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        # 假设 `i` 是查询向量的位置， `j` 是键向量的位置
        # 当keys在左侧时(i>j)使用正相对位置，否则(i<j)使用负相对位置
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

        # 反转正索引的顺序，并连接正负索引
        # 这用于支持位移技巧
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        self.extend_pe(hidden_states)
        # 获取相对位置编码的起止索引
        start_idx = self.pe.size(1) // 2 - hidden_states.size(1) + 1
        end_idx = self.pe.size(1) // 2 + hidden_states.size(1)
        # 获取相对位置编码
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


# 实现同样填充的PyTorch模块
class SeamlessM4TConformerSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 确定需要删除的填充数量
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0
    # 定义前向传播函数
    def forward(self, hidden_states):
        # 如果需要移除填充，则截取隐藏状态的前面部分
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        # 返回处理后的隐藏状态
        return hidden_states
# 这个模块实现了一个特征映射层，用于将输入的隐藏状态映射到模型的隐藏层大小
class SeamlessM4TConformerFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 LayerNorm 对输入的隐藏状态进行归一化
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        # 使用线性层将归一化后的隐藏状态映射到模型的隐藏层大小
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        # 在映射后的隐藏状态上应用 Dropout 以防止过拟合
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行归一化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态映射到模型的隐藏层大小
        hidden_states = self.projection(norm_hidden_states)
        # 在映射后的隐藏状态上应用 Dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 这个模块实现了一个前馈网络层，用于对输入的隐藏状态进行非线性变换
class SeamlessM4TConformerFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, dropout=None):
        super().__init__()
        # 如果没有提供 dropout 和 activation function，则使用配置中的默认值
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act

        # 在中间层应用 Dropout 以防止过拟合
        self.intermediate_dropout = nn.Dropout(dropout)
        # 使用线性层将输入的隐藏状态映射到中间层大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        # 对中间层的输出应用指定的激活函数
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn

        # 使用线性层将中间层的输出映射回模型的隐藏层大小
        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        # 在输出层应用 Dropout 以防止过拟合
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        # 将输入的隐藏状态映射到中间层大小，并应用激活函数和 Dropout
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        # 将中间层的输出映射回模型的隐藏层大小，并应用 Dropout
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# 这个模块实现了一个卷积块，用于在 Conformer 块中使用
class SeamlessM4TConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""
    # 初始化函数，接受配置参数并进行初始化
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 检查深度可分离卷积核大小是否为奇数，若不是则引发 ValueError 异常
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        # 初始化层归一化模块
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        # 初始化第一个点卷积层
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # 初始化 GLU 激活函数
        self.glu = nn.GLU(dim=1)
        # 初始化深度可分离卷积层
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding="same",
            groups=config.hidden_size,
            bias=False,
        )
        # 初始化批归一化模块
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        # 初始化激活函数
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        # 初始化第二个点卷积层
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    # 前向传播函数，接受隐藏状态和注意力掩码作为输入
    def forward(self, hidden_states, attention_mask=None):
        # 对隐藏状态进行层归一化处理
        hidden_states = self.layer_norm(hidden_states)

        # 确保在深度卷积中不会泄露填充位置，将填充位置置零
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # 交换时间维度和特征维度
        hidden_states = hidden_states.transpose(1, 2)

        # GLU 机制
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # 进行 1D 深度可分离卷积
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        # 使用第二个点卷积层
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        # 返回处理后的隐藏状态
        return hidden_states
class SeamlessM4TConformerSelfAttention(nn.Module):
    """Construct a SeamlessM4TConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """

    def __init__(self, config, use_position_embeddings=True):
        super().__init__()

        # 计算每个头的大小
        self.head_size = config.hidden_size // config.speech_encoder_attention_heads
        # 注意力头的数量
        self.num_heads = config.speech_encoder_attention_heads
        # 如果使用位置编码，则设置位置编码的类型，否则为None
        self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None

        # Query, Key, Value 和输出的线性映射
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        # 用于在自注意力计算中的Dropout
        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

        # 如果使用相对位置编码，则初始化相应的参数
        if self.position_embeddings_type == "relative":
            # 位置编码的线性转换
            self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            # 这两个可学习的偏置用于矩阵c和矩阵d，如https://arxiv.org/abs/1901.02860 第3.3节所述
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    # 从transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention.forward复制而来
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # 多头注意力机制的实现
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 获取 hidden_states 的batch_size、序列长度和隐藏层大小
        batch_size, sequence_length, hidden_size = hidden_states.size()
    
        # 确保 query/key 状态可以与 value 状态不同
        query_key_states = hidden_states
        value_states = hidden_states
    
        # 如果使用了 rotary 位置编码
        if self.position_embeddings_type == "rotary":
            # 如果没有提供 relative_position_embeddings，则引发错误
            if relative_position_embeddings is None:
                raise ValueError("`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'")
            # 应用 rotary 位置编码
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)
    
        # 将 query_key_states 和 value_states 进行线性变换
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)
    
        # 将 query、key 和 value 调整为适合注意力机制的格式
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
    
        # 如果使用了相对位置编码
        if self.position_embeddings_type == "relative":
            # 如果没有提供 relative_position_embeddings，则引发错误
            if relative_position_embeddings is None:
                raise ValueError("`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'relative'")
            # 将相对位置编码应用到 query-key 得分中
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            # 计算 query-key 得分
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
    
        # 如果提供了注意力掩码，则将其应用到得分中
        if attention_mask is not None:
            scores = scores + attention_mask
    
        # 应用 softmax 并进行dropout
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
    
        # 将 probs 和 value 相乘得到输出
        hidden_states = torch.matmul(probs, value)
    
        # 将输出调整为最终形状
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)
    
        return hidden_states, probs
    
    
    这是一个多头注意力机制的实现。主要包括以下步骤:
    
    1. 获取 `hidden_states` 的 `batch_size`、`sequence_length` 和 `hidden_size`。
    2. 确保 `query/key` 状态可以与 `value` 状态不同。
    3. 如果使用了 `rotary` 位置编码，应用 `rotary` 位置编码。
    4. 对 `query_key_states` 和 `value_states` 进行线性变换。
    5. 将 `query`、`key` 和 `value` 调整为适合注意力机制的格式。
    6. 如果使用了相对位置编码，应用相对位置编码到 `query-key` 得分中。
    7. 计算 `query-key` 得分。
    8. 如果提供了注意力掩码，将其应用到得分中。
    9. 应用 `softmax` 并进行 `dropout`。
    10. 将 `probs` 和 `value` 相乘得到输出。
    11. 将输出调整为最终形状。
    # 将相对位置编码应用于隐藏状态
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        # 获取隐藏状态的批大小、序列长度和隐藏大小
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # 将隐藏状态重塑为(batch_size, sequence_length, num_heads, head_size)的形状
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)
    
        # 从相对位置编码中获取cosine和sine值
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]
    
        # 旋转隐藏状态
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)
    
        # 将隐藏状态重塑回原始形状
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)
    
        return hidden_states
    
    # 从transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention复制的_apply_relative_embeddings
    # 1. 对相对位置编码进行投影
    # => (batch, head, 2*time1-1, d_k)
    proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
    proj_relative_position_embeddings = proj_relative_position_embeddings.view(
        relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
    )
    proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
    proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

    # 2. 给查询添加偏置
    # => (batch, head, time1, d_k)
    query = query.transpose(1, 2)
    q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
    q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

    # 3. 注意力分数：首先计算矩阵a和矩阵c
    # 如https://arxiv.org/abs/1901.02860 第3.3节所述
    # => (batch, head, time1, time2)
    scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

    # 4. 然后计算矩阵b和矩阵d
    # => (batch, head, time1, 2*time1-1)
    scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

    # 5. 移动矩阵b和矩阵d
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
# SeamlessM4TConformerEncoderLayer 类定义了一个 Conformer 编码器层
class SeamlessM4TConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    # 初始化函数，接受配置参数 config
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size # 获取隐藏层大小
        dropout = config.speech_encoder_dropout # 获取语音编码器dropout率

        # 1. Feed-forward 1 层
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim) # 添加 Layer Normalization 层
        self.ffn1 = SeamlessM4TConformerFeedForward(config) # 创建前馈层

        # 2. 自注意力层
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim) # 添加 Layer Normalization 层
        self.self_attn_dropout = nn.Dropout(dropout) # 添加 Dropout 层
        self.self_attn = SeamlessM4TConformerSelfAttention(config) # 创建自注意力层

        # 3. Conformer 卷积层
        self.conv_module = SeamlessM4TConformerConvolutionModule(config) # 创建 Conformer 卷积层

        # 4. Feed-forward 2 层
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim) # 添加 Layer Normalization 层
        self.ffn2 = SeamlessM4TConformerFeedForward(config) # 创建第二个前馈层
        self.final_layer_norm = nn.LayerNorm(embed_dim) # 最后添加一个 Layer Normalization 层

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states # 输入隐藏状态

        # 1. Feed-Forward 1 层
        residual = hidden_states # 保留残差连接
        hidden_states = self.ffn1_layer_norm(hidden_states) # 进行 Layer Normalization
        hidden_states = self.ffn1(hidden_states) # 通过前馈层
        hidden_states = hidden_states * 0.5 + residual # 残差连接

        # 2. 自注意力层
        residual = hidden_states # 保留残差连接
        hidden_states = self.self_attn_layer_norm(hidden_states) # 进行 Layer Normalization
        hidden_states, attn_weigts = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        ) # 通过自注意力层
        hidden_states = self.self_attn_dropout(hidden_states) # 应用 Dropout
        hidden_states = hidden_states + residual # 残差连接

        # 3. Conformer 卷积层
        residual = hidden_states # 保留残差连接
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask) # 通过 Conformer 卷积层
        hidden_states = residual + hidden_states # 残差连接

        # 4. Feed-Forward 2 层
        residual = hidden_states # 保留残差连接
        hidden_states = self.ffn2_layer_norm(hidden_states) # 进行 Layer Normalization
        hidden_states = self.ffn2(hidden_states) # 通过第二个前馈层
        hidden_states = hidden_states * 0.5 + residual # 残差连接
        hidden_states = self.final_layer_norm(hidden_states) # 最后进行一次 Layer Normalization

        return hidden_states, attn_weigts # 返回输出和注意力权重
    # 初始化方法，传入配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置参数
        self.config = config

        # 根据配置参数中的位置嵌入类型选择相应的位置嵌入方式
        if config.position_embeddings_type == "relative":
            self.embed_positions = SeamlessM4TConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = SeamlessM4TConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        # 初始化丢弃层
        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        
        # 构建多层编码器层
        self.layers = nn.ModuleList(
            [SeamlessM4TConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        # 初始化 LayerNorm 层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
class SeamlessM4TConformerAdapterLayer(nn.Module):
    # 初始化方法，接受一个 config 对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 获取嵌入维度和适配器的丢弃率
        embed_dim = config.hidden_size
        dropout = config.adaptor_dropout

        # 设置卷积核大小和步长
        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride

        # 1. 残差卷积
        # 对嵌入维度进行层归一化
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        # 创建一维卷积层，输入维度为嵌入维度，输出维度为 2 * 嵌入维度，设置卷积核大小、步长和填充
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 创建激活函数
        self.activation = nn.GLU(dim=1)

        # 自注意力
        # 对嵌入维度进行层归一化
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        # 创建一维卷积层，输入维度为嵌入维度，输出维度为 2 * 嵌入维度，设置卷积核大小、步长和填充
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 创建自注意力层对象
        self.self_attn = SeamlessM4TConformerSelfAttention(config, use_position_embeddings=False)
        # 创建 dropout 层
        self.self_attn_dropout = nn.Dropout(dropout)

        # 前馈神经网络
        # 对嵌入维度进行层归一化
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        # 创建前馈神经网络对象
        self.ffn = SeamlessM4TConformerFeedForward(config, act_fn="relu", dropout=dropout)

    # 从注意力掩码计算子采样长度
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        # 计算填充大小
        pad = self.kernel_size // 2
        # 计算序列长度
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        # 使用公式计算子采样长度
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.floor()

    # 正向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 对残差进行 layer normalization
        residual = self.residual_layer_norm(hidden_states)

        # 将残差应用池化以匹配多头注意力输出的序列长度。
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        # 通过残差卷积层
        residual = self.residual_conv(residual)
        # 应用激活函数
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        # 对自注意力层输出进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 在馈送到多头注意力层之前应用池化。
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        # 通过自注意力卷积层
        hidden_states = self.self_attn_conv(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        if attention_mask is not None:
            # 从注意力遮罩中计算子采样长度
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                hidden_states.device
            )
            # 计算新的注意力遮罩
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            # 准备4D注意力遮罩
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # 剩余计算与普通 Transformer 编码器层相同。
        # self-attention 操作
        hidden_states, attn_weigths = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # self-attention dropout
        hidden_states = self.self_attn_dropout(hidden_states)
        # 加上残差
        hidden_states = hidden_states + residual

        # 更新残差
        residual = hidden_states

        # 对经过 self-attention 的隐藏状态进行 layer normalization
        hidden_states = self.ffn_layer_norm(hidden_states)
        # 前馈神经网络层
        hidden_states = self.ffn(hidden_states) + residual

        # 返回最终隐藏状态
        return hidden_states
# 创建一个适配器模型类，继承自 nn.Module
class SeamlessM4TConformerAdapter(nn.Module):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()

        # 创建一个 nn.ModuleList，包含多个适配器层，数量由 config.num_adapter_layers 决定
        self.layers = nn.ModuleList(SeamlessM4TConformerAdapterLayer(config) for _ in range(config.num_adapter_layers))

    # 前向传播函数
    def forward(self, hidden_states, attention_mask):
        # 对隐藏状态进行降维处理（如果需要）

        # 遍历适配器层
        for layer in self.layers:
            # 将隐藏状态传入每一层适配器层
            hidden_states = layer(hidden_states, attention_mask)

        # 返回处理后的隐藏状态
        return hidden_states


############ TEXT / UNITS related code ################


# 从 transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding 复制过来的代码
# 创建一个产生正弦位置嵌入的模块
class SeamlessM4TSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    # 初始化函数
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()
        # 定义偏移量
        self.offset = 2
        # 嵌入维度
        self.embedding_dim = embedding_dim
        # 填充索引
        self.padding_idx = padding_idx
        # 生成权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # 生成权重的函数
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 获得嵌入权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        # 如果已经有权重了，在前向传播时将权重放在正确的 dtype 和 device 上
        if hasattr(self, "weights"):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 注册权重缓冲区
        self.register_buffer("weights", emb_weights, persistent=False)

    # 获得嵌入的静态方法
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算嵌入参数
        emb = math.log(10000) / (half_dim - 1)
        # 计算正弦参数
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        # 计算嵌入权重
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        # 计算正弦和余弦值并拼接起来
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果嵌入维度是奇数，填充零向量
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果有填充索引，将对应行置零
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        # 返回嵌入权重
        return emb.to(torch.get_default_dtype())

    # 前向传播函数
    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    # 检查是否存在输入的ids，如果存在则获取其大小
    if input_ids is not None:
        bsz, seq_len = input_ids.size()
        # 从输入的token ids创建位置id，任何填充的token仍然保持填充
        position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )
    else:
        bsz, seq_len = inputs_embeds.size()[:-1]
        # 从inputs_embeds创建位置id
        position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)
    
    # 如果需要，扩展嵌入
    max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
    if max_pos > self.weights.size(0):
        # 创建新的权重张量，使其具有足够的长度
        self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)
    
    # 通过位置id选择权重，返回权重的形状
    return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    
    # 从inputs_embeds直接提供嵌入。无法推断哪些是填充的，因此只需生成顺序位置id
    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
    
        # 生成顺序位置id
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
class SeamlessM4TAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 初始化函数，初始化注意力模块参数
    # 参数:
    #   embed_dim: 输入嵌入维度
    #   num_heads: 头的数量
    #   dropout: Dropout概率
    #   is_decoder: 是否为解码器
    #   bias: 是否使用偏置
    #   is_causal: 是否是因果模式
    #   config: 可选的SeamlessM4T配置参数
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[SeamlessM4TConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查嵌入维度是否能够被头数量整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性层，用于对输入进行投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 对输入张量进行形状变换
    # 参数:
    #   tensor: 输入张量
    #   seq_len: 序列长度
    #   bsz: batch size
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    # 参数:
    #   hidden_states: 输入张量
    #   encoder_hidden_states: 编码器隐藏状态，可选
    #   past_key_value: 过去的键值对，可选
    #   attention_mask: 注意力掩码，可选
    #   output_attentions: 是否输出注意力权重，可选
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # 定义一个类，初始化函数，包括配置参数、编码器的前馈神经网络维度和注意力头数
    def __init__(self, config: SeamlessM4TConfig, encoder_ffn_dim=None, encoder_attention_heads=None):
        # 调用父类的初始化函数
        super().__init__()
        # 如果未提供编码器的前馈神经网络维度，则使用配置参数中的值
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        # 如果未提供编码器的注意力头数，则使用配置参数中的值
        encoder_attention_heads = (
            config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        )
    
        # 设置嵌入维度为隐藏状态的维度
        self.embed_dim = config.hidden_size
        # 创建自注意力机制
        self.self_attn = SeamlessM4TAttention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 设置注意力机制的dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        # 设置自注意力机制的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
    
        # 创建前馈神经网络
        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)
    
        # 设置前馈神经网络的 LayerNorm
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        # 设置前馈神经网络的dropout
        self.ffn_dropout = nn.Dropout(config.activation_dropout)
    
    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                输入到层中的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                大小为 `(batch, 1, tgt_len, src_len)` 的注意力掩码，其中填充元素由非常大的负值指示。
        """
        # 保存残差连接
        residual = hidden_states
        # 对输入进行 LayerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用自注意力机制计算结果
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 对注意力结果进行dropout
        hidden_states = self.attn_dropout(hidden_states)
        # 叠加残差连接
        hidden_states = residual + hidden_states
    
        # 保存残差连接
        residual = hidden_states
    
        # 对输入进行 LayerNorm
        hidden_states = self.ffn_layer_norm(hidden_states)
    
        # 使用前馈神经网络计算结果
        hidden_states = self.ffn(hidden_states)
        # 对前馈神经网络结果进行dropout
        hidden_states = self.ffn_dropout(hidden_states)
    
        # 叠加残差连接
        hidden_states = residual + hidden_states
    
        # 输出结果
        outputs = (hidden_states,)
    
        # 如果需要输出注意力权重信息，则将其加入输出中
        if output_attentions:
            outputs += (attn_weights,)
    
        return outputs
class SeamlessM4TDecoderLayer(nn.Module):
    def __init__(self, config: SeamlessM4TConfig, decoder_ffn_dim=None, decoder_attention_heads=None):
        super().__init__()
        # 若未传入 decoder_ffn_dim，则使用 config 中的 decoder_ffn_dim 值
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        # 若未传入 decoder_attention_heads，则使用 config 中的 decoder_attention_heads 值
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )

        # 配置实例变量
        self.embed_dim = config.hidden_size
        # 初始化 self_attn 模块：SeamlessM4TAttention 注意力机制
        self.self_attn = SeamlessM4TAttention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 配置实例变量
        self.dropout = config.dropout
        # 激活函数，根据配置中的信息选择相应的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 配置实例变量
        self.attn_dropout = nn.Dropout(config.dropout)

        # 初始化 self attention 层规范化模块
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化 cross attention 模块：SeamlessM4TAttention 注意力机制
        self.cross_attention = SeamlessM4TAttention(
            self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        # 初始化 cross attention 层规范化模块
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化 ffn 模块：SeamlessM4TFeedForwardNetwork 前馈神经网络
        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)

        # 初始化 ffn 层规范化模块
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        # 初始化 ffn dropout 模块
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # 省略若干代码...

############ SUB-MODELS related code ################


class SeamlessM4TPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SeamlessM4TConfig
    base_model_prefix = "seamless_m4t"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SeamlessM4TEncoderLayer", "SeamlessM4TDecoderLayer", "SeamlessM4TConformerEncoderLayer"]


注释：
    def _init_weights(self, module):
        """初始化模型的权重"""
        # 从配置中获取初始化器的范围
        std = self.config.initializer_range
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是自注意力层
        elif isinstance(module, SeamlessM4TConformerSelfAttention):
            # 如果存在位置偏置项，使用均匀分布进行初始化
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        # 如果是位置卷积嵌入层
        elif isinstance(module, SeamlessM4TConformerPositionalConvEmbedding):
            # 使用正态分布初始化卷积核权重
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积偏置项初始化为零
            nn.init.constant_(module.conv.bias, 0)
        # 如果是特征投影层
        elif isinstance(module, SeamlessM4TConformerFeatureProjection):
            # 使用均匀分布初始化权重和偏置项
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果是 LayerNorm 或者 GroupNorm 层
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将偏置项初始化为零，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是一维卷积层
        elif isinstance(module, nn.Conv1d):
            # 使用 Kaiming 正态分布初始化权重
            nn.init.kaiming_normal_(module.weight)
            # 如果存在偏置项，使用均匀分布初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        """从注意力掩码计算子采样长度"""
        # 获取适配器的卷积核大小和步长
        kernel_size, stride = self.config.adaptor_kernel_size, self.config.adaptor_stride
        # 计算填充大小
        pad = kernel_size // 2
        # 计算每个样本的序列长度
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        # 计算子采样后的序列长度
        seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

        return seq_lens.floor()

    def compute_last_hidden_states_per_sample(
        self,
        hidden_states: Tuple[Tuple[torch.Tensor]],
        beam_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the last hidden states.

        Parameters:
            hidden_states (`Tuple[Tuple[torch.Tensor]]`):
                The generated hidden states. Tuple (one element for each generated token) of tuples (one element for
                each layer of the decoder) of torch.FloatTensor of shape (batch_size*num_beams*num_return_sequences,
                generated_length, hidden_size).
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
                generate-time.

        Return:
            `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length, hidden_size)`
            containing
                the last hidden states.
        """
        # 1. First, let's compute last_hidden_states from hidden_states.
        # For each generation step, takes the hidden state from the last layer.
        # shape: (batch_size*vocab_size*num_return_sequences, # generation_steps, hidden_dim)
        last_hidden_states = torch.concat([hidden_states[-1] for hidden_states in hidden_states], dim=1)

        # 2. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected.
        # In that case, return directly last_hidden_states
        if beam_indices is None:
            return last_hidden_states

        # 3. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 4. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards anyways
        beam_indices[beam_indices_mask] = 0

        # 5. expand beam_indices to last_hidden_states dim
        beam_indices = beam_indices.unsqueeze(-1)
        beam_indices = beam_indices.expand(-1, -1, last_hidden_states.shape[-1])

        # 6. select the right candidate for each beam
        # in other words, new_last_hidden_states[i,j,k] = last_hidden_states[beam_indices[i,j,k], j, k] for all i, j, k
        last_hidden_states = torch.gather(last_hidden_states, 0, beam_indices)

        return last_hidden_states
# 使用装饰器添加文档字符串，描述了该类的功能以及其包含的各个部分的组成
@add_start_docstrings(
    """Transformer speech encoder consisting of *config.speech_encoder_layers* conformer self attention layers.
    Each layer is a [`SeamlessM4TConformerEncoderLayer`].""",
    SEAMLESS_M4T_START_DOCSTRING,
)
# 定义了一个名为 SeamlessM4TSpeechEncoder 的类，继承自 SeamlessM4TPreTrainedModel
class SeamlessM4TSpeechEncoder(SeamlessM4TPreTrainedModel):
    # 定义了一个类变量 main_input_name，表示主要输入的名称为 "input_features"
    main_input_name = "input_features"

    # 定义了类的初始化方法，接受一个名为 config 的 SeamlessM4TConfig 类型参数
    def __init__(self, config: SeamlessM4TConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 SeamlessM4TConformerFeatureProjection 对象，用于特征投影
        self.feature_projection = SeamlessM4TConformerFeatureProjection(config)
        # 创建一个 SeamlessM4TConformerEncoder 对象，用于编码器部分
        self.encoder = SeamlessM4TConformerEncoder(config)
        # 创建一个 SeamlessM4TConformerFeedForward 对象，用于中间的全连接层
        self.intermediate_ffn = SeamlessM4TConformerFeedForward(config, act_fn="relu", dropout=0.0)
        # 如果配置中添加了 adapter，则创建一个 SeamlessM4TConformerAdapter 对象，否则为 None
        self.adapter = SeamlessM4TConformerAdapter(config) if config.add_adapter else None
        # 创建一个 nn.LayerNorm 对象，用于内部层的归一化
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义了类的前向传播方法
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 设置输出注意力的标志，如果未指定则使用配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的标志，如果未指定则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的标志，如果未指定则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果输入特征为 None，则抛出 ValueError 异常
        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4TSpeechEncoder.forward`.
                Make sure one of them is not `None`."""
            )

        # 将输入特征进行特征投影，得到隐藏状态
        hidden_states = self.feature_projection(input_features)

        # 将隐藏状态传递给编码器，并获取编码器的输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从编码器的输出中获取隐藏状态
        hidden_states = encoder_outputs[0]

        # 对隐藏状态进行扩展，并应用中间的全连接层
        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

        # 如果存在适配器，则对隐藏状态进行适配
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        # 对隐藏状态进行内部层的归一化
        hidden_states = self.inner_layer_norm(hidden_states)

        # 如果不需要返回字典，则返回元组形式的结果
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        # 如果需要返回字典，则构造 Wav2Vec2BaseModelOutput 对象并返回
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 基于 MBart 和 NllbMoe 进行启发
@add_start_docstrings(
    # 创建一个字符串，描述Transformer编码器由config.encoder_layers个自注意力层组成的结构。每一层都是一个SeamlessM4TEncoderLayer。
    "Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`SeamlessM4TEncoderLayer`].",
    # 插入SEAMLESS_M4T_START_DOCSTRING的值
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional`):
            Input embedding
        is_t2u_encoder (`bool`, *optional*, defaults to `False`):
            indicates if it belongs to the text-to-units model, in which case it won't have input embeddings
    """,
# 定义 SeamlessM4TEncoder 类，继承自 SeamlessM4TPreTrainedModel 类
class SeamlessM4TEncoder(SeamlessM4TPreTrainedModel):
    # 初始化方法，接受配置参数 config、嵌入标记 embed_tokens 和是否为 t2u 编码器的标志 is_t2u_encoder
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        is_t2u_encoder: bool = False,
    ):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置类属性
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_dim = config.hidden_size

        self.is_t2u_encoder = is_t2u_encoder
        self.max_source_positions = config.max_position_embeddings

        # 如果不是 t2u 编码器
        if not self.is_t2u_encoder:
            # 根据配置设置嵌入比例
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            # 创建嵌入标记对象
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

            # 如果传入了嵌入标记对象，则使用传入的权重
            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight

            # 创建嵌入位置对象
            self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
            )

        # 创建编码器层列表
        layers = []
        for _ in range(config.encoder_layers):
            layers.append(
                SeamlessM4TEncoderLayer(
                    config,
                    encoder_attention_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                )
            )

        # 将编码器层列表转换为模块列表
        self.layers = nn.ModuleList(layers)

        # 创建层归一化对象
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受输入 id、注意力掩码、输入嵌入、输出注意力、输出隐藏状态、返回字典等参数
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,



# 添加文档字符串，描述 Transformer 解码器由 config.decoder_layers 层组成，每层是一个 SeamlessM4TDecoderLayer
# 并包含一些额外的文档信息
@add_start_docstrings(
    "Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SeamlessM4TDecoderLayer`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
    """,
)
# 定义 SeamlessM4TDecoder 类，继承自 SeamlessM4TPreTrainedModel 类
class SeamlessM4TDecoder(SeamlessM4TPreTrainedModel):
    # 初始化方法，接受配置参数 config 和嵌入标记 embed_tokens
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        ):
            # 调用父类的构造函数，初始化配置参数
            super().__init__(config)
            # 设置dropout参数
            self.dropout = config.dropout
            # 设置decoder_layerdrop参数
            self.layerdrop = config.decoder_layerdrop
            # 设置padding_idx参数
            self.padding_idx = config.pad_token_id
            # 设置vocab_size参数
            self.vocab_size = config.vocab_size
            # 设置max_target_positions参数
            self.max_target_positions = config.max_position_embeddings
            # 设置embed_scale参数
            self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

            if embed_tokens is not None:
                # 如果embed_tokens已定义，则使用其形状
                self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
                self.embed_tokens.weight = embed_tokens.weight
            else:
                # 否则使用默认的Embedding
                self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

            # 初始化位置编码
            self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(
                self.max_target_positions,
                config.hidden_size,
                padding_idx=self.padding_idx,
            )

            layers = []
            # 根据decoder_layers参数循环创建decoder层
            for _ in range(config.decoder_layers):
                layers.append(
                    SeamlessM4TDecoderLayer(
                        config,
                        decoder_attention_heads=config.decoder_attention_heads,
                        decoder_ffn_dim=config.decoder_ffn_dim,
                    )
                )
            self.layers = nn.ModuleList(layers)
            # 初始化LayerNorm
            self.layer_norm = nn.LayerNorm(config.hidden_size)

            self.gradient_checkpointing = False
            # 初始化权重并应用最终处理
            self.post_init()

        def get_input_embeddings(self):
            return self.embed_tokens

        def set_input_embeddings(self, value):
            self.embed_tokens = value

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
# 导入装饰器，用于添加文档字符串到类的开始位置
@add_start_docstrings(
    # 添加类文档字符串，描述此类是一个 Transformer 的文本到单元编码-解码器，
    # 其中编码器是一个无嵌入的 SeamlessM4TEncoder，解码器是一个 SeamlessM4TDecoder。
    "Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4TEncoder`] without embeddings and the decoder is a [`SeamlessM4TDecoder`].",
    # 添加额外的类文档字符串，从 SEAMLESS_M4T_START_DOCSTRING 变量中获取
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
# 定义 SeamlessM4TTextToUnitModel 类，继承自 SeamlessM4TPreTrainedModel
class SeamlessM4TTextToUnitModel(SeamlessM4TPreTrainedModel):
    # 定义类的初始化方法
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建编码器对象，使用 SeamlessM4TEncoder 类，is_t2u_encoder 参数设置为 True
        self.encoder = SeamlessM4TEncoder(config, is_t2u_encoder=True)
        # 创建解码器对象，使用 SeamlessM4TDecoder 类，embed_tokens_decoder 参数指定解码器的输入嵌入
        self.decoder = SeamlessM4TDecoder(config, embed_tokens_decoder)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 设置输出注意力权重，默认为配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否使用缓存，默认为配置中的值
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 设置是否返回字典，默认为配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果没有编码器输出，则调用编码器进行编码
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # 如果用户传递了元组作为编码器输出，并且返回字典为True，则将其包装在BaseModelOutput中
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # 解码器输出包括(dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不返回字典，则返回解码器输出和编码器输出
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回Seq2SeqModelOutput对象，包括解码器和编码器的相关输出
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 导入必要的库
@add_start_docstrings(
    "Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4TTextToUnit`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
# 定义一个新的类，继承自SeamlessM4TPreTrainedModel
class SeamlessM4TTextToUnitForConditionalGeneration(SeamlessM4TPreTrainedModel):
    # 在加载模型时忽略的键列表
    _keys_to_ignore_on_load_missing = [
        "vocoder",
        "speech_encoder",
        "text_encoder",
        "text_decoder",
    ]
    # 共享权重的键列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化函数
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        # 深拷贝配置
        config = copy.deepcopy(config)
        # 更新配置
        for param, val in config.to_dict().items():
            if param.startswith("t2u_"):
                config.__setattr__(param[4:], val)
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建SeamlessM4TTextToUnitModel模型
        self.model = SeamlessM4TTextToUnitModel(config, embed_tokens_decoder)

        # 创建线性层作为语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.t2u_vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输入��入层
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 前向传播函数
    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 定义函数返回类型为 Seq2SeqLMOutput 或者包含 torch.FloatTensor 的元组
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果提供了标签，则设置 use_cache 为 False
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # 如果未提供解码器输入，根据标签生成解码器输入
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id
                )

        # 调用模型进行前向传播
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取语言模型的输出
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不需要返回字典形式的结果
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 Seq2SeqLMOutput 对象
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    # 为生成准备输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值，截取解码器输入的最后一个 token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回包含准备生成所需输入的字典
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
    # 从标签中准备解码器的输入 ID，将标签向右移动一个位置，用解码器的起始标记填充
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id)

    # 重新排序缓存中的键值对，根据 beam_idx 重新排列
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # 绑定权重，如果配置中设置了 tie_word_embeddings 为 True，则绑定输出和输入的嵌入权重
    def _tie_weights(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
############ VOCODER related code ################

# HIFIGAN_START_DOCSTRING 是一个文档字符串，用于描述 HifiGan 模型的继承关系和参数说明
HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SeamlessM4TConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 HifiGanResidualBlock 类，继承自 nn.Module
# 该类实现了残差块的功能
class HifiGanResidualBlock(nn.Module):
    # 初始化函数，定义残差块的结构
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # 定义多个卷积层，用于残差块的计算
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

    # 计算卷积层的 padding 大小
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 对卷积层应用权重归一化
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除卷积层的权重归一化
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    # 前向传播函数，实现残差块的计算
    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states
class SeamlessM4TVariancePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        embed_dim = config.unit_embed_dim
        kernel_size = config.variance_predictor_kernel_size
        var_pred_dropout = config.var_pred_dropout

        # 创建第一个卷积层，用于预测方差
        self.conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.activation_fuction = nn.ReLU()  # 激活函数为ReLU
        self.ln1 = nn.LayerNorm(embed_dim)  # LayerNorm层
        self.dropout_module = nn.Dropout(p=var_pred_dropout)  # Dropout层
        # 创建第二个卷积层，用于预测方差
        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=1,
        )
        self.ln2 = nn.LayerNorm(embed_dim)  # LayerNorm层
        self.proj = nn.Linear(embed_dim, 1)  # 线性变换层

    def forward(self, hidden_states: Tensor) -> Tensor:
        # 输入: B x T x C; 输出: B x T
        hidden_states = self.conv1(hidden_states.transpose(1, 2))  # 第一个卷积层
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)  # 激活函数
        hidden_states = self.dropout_module(self.ln1(hidden_states))  # Dropout和LayerNorm
        hidden_states = self.conv2(hidden_states.transpose(1, 2))  # 第二个卷积层
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)  # 激活函数
        hidden_states = self.dropout_module(self.ln2(hidden_states))  # Dropout和LayerNorm
        return self.proj(hidden_states).squeeze(dim=2)  # 线性变换并压缩维度


class SeamlessM4THifiGan(nn.Module):
    def __init__(self, config: SeamlessM4TConfig):
        super().__init__()
        model_in_dim = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        # 创建预处理卷积层
        self.conv_pre = nn.Conv1d(
            model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # 创建上采样器
        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        # 创建残差块
        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 创建后处理卷积层
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)
    # 定义一个方法，将对数梅尔频谱转换为语音波形。传入一个对数梅尔频谱的批量返回一个语音波形的批量。传入一个单个的未批量化的对数梅尔频谱返回一个单个的未批量化的语音波形。
    def forward(self, input_embeds: torch.FloatTensor) -> torch.FloatTensor:
        # 使用预处理卷积层处理输入的嵌入向量
        hidden_states = self.conv_pre(input_embeds)
        # 循环进行上采样操作
        for i in range(self.num_upsamples):
            # 使用LeakyReLU激活函数处理隐藏状态
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 使用上采样器进行上采样操作
            hidden_states = self.upsampler[i](hidden_states)

            # 使用残差块处理隐藏状态
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        # 使用LeakyReLU激活函数处理隐藏状态
        hidden_states = nn.functional.leaky_relu(hidden_states)
        # 使用后处理卷积层处理隐藏状态
        hidden_states = self.conv_post(hidden_states)
        # 使用双曲正切函数处理隐藏状态
        hidden_states = torch.tanh(hidden_states)

        # 去除序列长度维度，因为它会被压缩为1
        waveform = hidden_states.squeeze(1)

        # 返回语音波形
        return waveform
# 添加起始文档字符串，描述了 HiFi-GAN 声码器的代码，包括了一个链接到 GitHub 仓库的说明
@add_start_docstrings(
    """Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).""",
    HIFIGAN_START_DOCSTRING,
)
# 定义 SeamlessM4TCodeHifiGan 类，继承自 PreTrainedModel
class SeamlessM4TCodeHifiGan(PreTrainedModel):
    # 配置类为 SeamlessM4TConfig
    config_class = SeamlessM4TConfig
    # 主输入名称为 "input_embeds"
    main_input_name = "input_embeds"
    # 不需要拆分的模块列表为空
    _no_split_modules = []

    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置填充标记的 ID
        self.pad_token_id = config.t2u_pad_token_id
        # 创建 SeamlessM4TVariancePredictor 对象
        self.dur_predictor = SeamlessM4TVariancePredictor(config)

        # 创建单元嵌入层
        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        # 创建说话者嵌入层
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        # 创建语言嵌入层
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)

        # 创建 HiFi-GAN 模型
        self.hifi_gan = SeamlessM4THifiGan(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取持续时间输出长度的方法
    def _get_dur_output_lengths(self, input_ids, dur_out):
        """
        Computes the output length after the duration layer.
        """
        # 计算单元长度，排除填充标记
        unit_lengths = (input_ids != self.pad_token_id).sum(1)

        # 处理没有填充或填充过多的边缘情况
        unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)

        # 计算累积持续时间输出
        cumulative_dur_out = torch.cumsum(dur_out, dim=1)
        # 获取单元长度对应的持续时间输出长度
        unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()

        return unit_lengths
    def _get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        计算 hifigan 卷积层的输出长度
        """

        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 上得到的 1D 卷积层输出长度公式
            return (
                torch.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode="floor") + 1
            )

        def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            # 计算转置卷积层输出长度的函数
            return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

        # 计算 conv_pre 的输出长度
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        # 计算 upsampler 的输出长度
        for i, (upsample_rate, kernel_size) in enumerate(
            zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)
        ):
            input_lengths = _transpose_conv_out_length(
                input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2
            )

        # 计算 resblock 的输出长度
        for i in range(len(self.config.upsample_rates)):
            for kernel_size, dilation in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                for dil in dilation:
                    input_lengths = _conv_out_length(
                        input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil
                    )

                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

        # 计算 conv_post 的输出长度
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        return input_lengths

    def forward(
        self, input_ids: torch.LongTensor, spkr_id: torch.Tensor, lang_id: torch.Tensor
    # 定义函数，接受输入的 input_ids，并返回隐藏状态和长度
    def forward(self, input_ids: torch.Tensor, spkr_id: int, tgt_lang: str) -> Tuple[torch.Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
    
                Indices can be obtained using [`SeamlessM4TTextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            spkr_id (`int`, *optional`):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional`):
                The language id to use as target language for translation.
        """
        # 对输入的 input_ids 进行单位嵌入并转置
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)
        # 根据 spkr_id 提取说话者的嵌入并转置
        spkr = self.speaker_embedding(spkr_id).transpose(1, 2)
        # 根据 lang_id 提取语言的嵌入并转置
        lang = self.language_embedding(lang_id).transpose(1, 2)
    
        # 使用 dur_predictor 对 hidden_states 进行预测
        log_dur_pred = self.dur_predictor(hidden_states.transpose(1, 2))
        # 对预测结果进行处理
        dur_out = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1)
        
        # 如果 hidden_states 的大小为 1，使用 repeat_interleave 函数进行扩展
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            # 如果是 batched 的样本，需要对每个样本进行插补，并且需要填充(padding)，可能会丢失并行性
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            # 使用循环对隐藏状态进行扩展和重新排列
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=-1).transpose(0, 1)
                for (hidden_state, duration) in zip(hidden_states, dur_out)
            ]
            # 对 hidden_states 进行填充并进行转置
            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)
    
        # 对 spkr 进行扩展
        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        # 对 lang 进行扩展
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        # 将 lang、hidden_states 和 spkr 进行连接
        hidden_states = torch.cat([lang, hidden_states, spkr], dim=1)
    
        # 对 hidden_states 使用 hifi_gan 进行处理
        hidden_states = self.hifi_gan(hidden_states)
    
        # 获取 unit_lengths 和 lengths 的值并返回
        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        lengths = self._get_output_hifigan_lengths(unit_lengths)
    
        return hidden_states, lengths
    
    # 初始化权重的函数
    def _init_weights(self, module):
        """Initialize the weights."""
        # 若 module 是 Linear、Conv1d 或 ConvTranspose1d 类型，对权重进行初始化
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置(bias)，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 若 module 是 Embedding 类型，对权重进行初始化
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有 padding_idx，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    # 对模型中的各个层应用权重归一化
    def apply_weight_norm(self):
        # 对预处理卷积层应用权重归一化
        nn.utils.weight_norm(self.hifi_gan.conv_pre)
        # 对上采样器中的每一层应用权重归一化
        for layer in self.hifi_gan.upsampler:
            nn.utils.weight_norm(layer)
        # 对残差模块列表中的每个残差块应用权重归一化
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        # 对后处理卷积层应用权重归一化
        nn.utils.weight_norm(self.hifi_gan.conv_post)

    # 移除模型中各层的权重归一化
    def remove_weight_norm(self):
        # 移除预处理卷积层的权重归一化
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        # 对上采样器中的每一层移除权重归一化
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        # 对残差模块列表中的每个残差块移除权重归一化
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        # 移除后处理卷积层的权重归一化
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)
############ WHOLE MODEL related code ################

# 定义了一个基于 SeamlessM4TPreTrainedModel 的文本到文本转换器模型
@add_start_docstrings(
    "The text-to-text SeamlessM4T Model transformer which can be used for T2TT.",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TForTextToText(SeamlessM4TPreTrainedModel):
    # 在模型加载时忽略的键列表
    _keys_to_ignore_on_load_missing = ["speech_encoder", "t2u_model", "vocoder"]
    # 主要输入名称
    main_input_name = "input_ids"

    # 需要共享权重的键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化方法
    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        # 共享层，将词汇映射到隐藏状态
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 文本编码器
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        # 文本解码器
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        # 语言模型头，用于预测下一个词的概率分布
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.text_encoder

    # 获取解码器
    def get_decoder(self):
        return self.text_decoder

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 绑定权重方法
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 前向传播方法
    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        ...

    # 生成方法
    def generate(
        self,
        input_ids=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    # 为生成准备输入的方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值，就截取decoder_input_ids的最后一个元素
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs已经定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    # 重新排序缓存
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的跨注意力状态不需要重新排序 -> 它们始终保持不变
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
# 使用装饰器为该类添加文档字符串，描述该模型可以用于语音到文本转换
# SEAMLESS_M4T_START_DOCSTRING 应该是一个定义好的文档字符串变量
class SeamlessM4TForSpeechToText(SeamlessM4TPreTrainedModel):
    # 在加载时忽略的键的列表
    _keys_to_ignore_on_load_missing = ["text_decoder", "t2u_model", "vocoder"]
    # 主要输入的名称
    main_input_name = "input_features"

    # 被绑定权重的键的列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化方法，接受配置对象作为参数
    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        # 创建共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 创建语音编码器
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        # 创建文本解码器
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        # 创建语言模型头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回编码器对象
    def get_encoder(self):
        return self.speech_encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.text_decoder

    # 返回输出嵌入层对象
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入层对象
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 返回输入嵌入层对象
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置新的输入嵌入层对象
    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 正向传播方法，接受多种输入参数
    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    def generate(
        self,
        input_features=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
        # 检查是否使用了 past 参数，如果使用则截断 decoder_input_ids
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，包含各个参数
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序 past_key_values，根据 beam_idx
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的跨注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
# 添加注释以说明类的用途和特性
@add_start_docstrings(
    "The text-to-speech SeamlessM4T Model transformer which can be used for T2ST.",
    SEAMLESS_M4T_START_DOCSTRING,
)
# 定义一个类 SeamlessM4TForTextToSpeech，继承自 SeamlessM4TPreTrainedModel
class SeamlessM4TForTextToSpeech(SeamlessM4TPreTrainedModel):
    # 定义需要在加载时忽略的键
    _keys_to_ignore_on_load_missing = ["speech_encoder"]
    # 定义主要输入的名称
    main_input_name = "input_ids"

    # 定义需要共享权重的键
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化函数，接受一个 config 对象
    def __init__(self, config: SeamlessM4TConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建一个共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 创建文本编码器和解码器
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        # 创建一个线性层作为语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用后续初始化函数
        self.post_init()

        # 创建一个 SeamlessM4TTextToUnitForConditionalGeneration 模型实例
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        # 创建一个 SeamlessM4TCodeHifiGan 模型实例
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    # 返回文本编码器
    def get_encoder(self):
        return self.text_encoder

    # 返回文本解码器
    def get_decoder(self):
        return self.text_decoder

    # 返回输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 定义前向传播函数，接受多种输入和返回多个选项
    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 禁止使用 torch 的梯度计算
    @torch.no_grad()
    # 生成器函数，用于生成模型的输出
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        spkr_id: Optional[int] = 0,
        **kwargs,
    ):  
    # 为生成准备输入的函数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值，对解码器输入进行切割
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回包含准备好的输入数据的字典
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    # 重新排序缓存的函数
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终是相同的
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        # 返回重新排序后的缓存
        return reordered_past
# 添加文档字符串，描述该模型的用途以及相关说明
@add_start_docstrings(
    "The speech-to-speech SeamlessM4T Model transformer which can be used for S2ST.",
    SEAMLESS_M4T_START_DOCSTRING,
)
# 创建 SeamlessM4TForSpeechToSpeech 类，继承自 SeamlessM4TPreTrainedModel
class SeamlessM4TForSpeechToSpeech(SeamlessM4TPreTrainedModel):
    # 在加载过程中忽略的键
    _keys_to_ignore_on_load_missing = ["text_encoder"]
    # 主要输入名称
    main_input_name = "input_features"

    # 绑定权重的键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建共享层
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 创建语音编码器
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        # 创建文本解码器
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        # 创建语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

        # 创建文本到单元的模型
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        # 创建声码器
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    # 获取编码器
    def get_encoder(self):
        return self.speech_encoder

    # 获取解码器
    def get_decoder(self):
        return self.text_decoder

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 前向传播方法，并添加文档字符串
    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    # 无梯度的生成方法
    @torch.no_grad()
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        spkr_id: Optional[int] = 0,
        **kwargs,
    # 静态方法
    @staticmethod
    # 重新排列缓存的键值对，以适应beam搜索
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排列过去的键值对，用于beam搜索
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 对于交叉注意力状态的缓存，不需要重新排列，它们始终保持不变
            reordered_past += (
                # 使用beam索引重新排列过去状态
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        # 返回重新排列后的过去状态
        return reordered_past

    # 为生成准备输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值对，则截取decoder_input_ids
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回输入的准备结果
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
# 添加文档字符串到模型的起始位置，并提供一些默认值
@add_start_docstrings(
    "The original SeamlessM4T Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        current_modality (`str`, *optional*, defaults to `"text"`):
            Default modality. Used to initialize the model.
    """,
)
class SeamlessM4TModel(SeamlessM4TPreTrainedModel):
    # 初始化需要共享的权重
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config, current_modality="text"):
        # 调用父类的初始化方法
        super().__init__(config)

        # 用词汇表大小、隐藏层大小和填充标记初始化共享权重
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 初始化文本编码器、语音编码器、文本解码器、语言模型头
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

        # 设置当前的模态
        self.current_modality = current_modality
        if current_modality == "speech":
            self.main_input_name = "input_features"

        # 这些模型已经在它们的初始化中调用了post_init
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    # 设置模态
    def set_modality(self, modality="text"):
        if modality == "text":
            self.main_input_name = "input_ids"
            self.current_modality = "text"
        elif modality == "speech":
            self.main_input_name = "input_features"
            self.current_modality = "speech"
        else:
            raise ValueError(f"`modality={modality}` is not a valid modality. It must be `text` or `speech`.")

    # 获取编码器
    def get_encoder(self):
        if self.current_modality == "text":
            return self.text_encoder
        else:
            return self.speech_encoder

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 添加文档字符串到模型前向传播的位置
    @add_start_docstrings_to_model_forward(M4T_MODEL_INPUTS_DOCSTRING)
    # 前向传播方法，用于模型的前向推断
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入序列的 token ID
        input_features: Optional[torch.FloatTensor] = None,  # 输入序列的特征向量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指示输入序列中的填充部分
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入序列的 token ID
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器输出
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入向量
        labels: Optional[torch.LongTensor] = None,  # 标签序列
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        **kwargs,  # 其他参数
    @torch.no_grad()  # 声明下面的 generate 方法为不计算梯度的操作
    # 生成方法，用于生成模型的输出
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入序列的 token ID
        input_features: Optional[torch.Tensor] = None,  # 输入序列的特征向量
        return_intermediate_token_ids: Optional[bool] = None,  # 是否返回中间 token ID
        tgt_lang: Optional[str] = None,  # 目标语言
        spkr_id: Optional[int] = 0,  # 说话者 ID
        generate_speech: Optional[bool] = True,  # 是否生成语音
        **kwargs,  # 其他参数
    # 为生成准备输入的方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的 token ID
        past_key_values=None,  # 过去的键值对
        attention_mask=None,  # 注意力掩码
        use_cache=None,  # 是否使用缓存
        encoder_outputs=None,  # 编码器输出
        **kwargs,  # 其他参数
    ):
        # 如果使用过去的键值对，裁剪解码器输入的 token ID
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回准备好的输入
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键值对
            "decoder_input_ids": decoder_input_ids,  # 解码器输入的 token ID
            "attention_mask": attention_mask,  # 注意力掩码
            "use_cache": use_cache,  # 是否使用缓存
        }

    @staticmethod
    # 重新排序缓存方法
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序过去的键值对
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的跨注意力状态不需要重新排序，它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
```