# `.\models\seamless_m4t\modeling_seamless_m4t.py`

```
# 导入必要的库和模块
import copy  # 导入copy模块，用于复制对象
import math  # 导入math模块，用于数学运算
from dataclasses import dataclass  # 导入dataclass类装饰器，用于定义数据类
from typing import Optional, Tuple, Union  # 导入类型提示相关的类和函数

import torch  # 导入PyTorch库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint工具
from torch import Tensor, nn  # 从torch中导入Tensor和nn模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数CrossEntropyLoss

from ...activations import ACT2FN  # 导入激活函数映射ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled  # 导入深速模块，检查是否启用深速
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
# 从模型注意力掩码工具中导入相关函数
from ...modeling_outputs import (
    BaseModelOutput,  # 导入基础模型输出类
    BaseModelOutputWithPastAndCrossAttentions,  # 导入包含过去和交叉注意力的基础模型输出类
    Seq2SeqLMOutput,  # 导入序列到序列语言模型输出类
    Seq2SeqModelOutput,  # 导入序列到序列模型输出类
    Wav2Vec2BaseModelOutput,  # 导入Wav2Vec2基础模型输出类
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类PreTrainedModel
from ...utils import (
    ModelOutput,  # 导入模型输出基类ModelOutput
    add_start_docstrings,  # 导入添加起始文档字符串函数
    add_start_docstrings_to_model_forward,  # 导入为模型forward方法添加起始文档字符串函数
    logging,  # 导入日志记录模块logging
)
from .configuration_seamless_m4t import SeamlessM4TConfig  # 从当前目录导入SeamlessM4TConfig配置类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置常量
_CHECKPOINT_FOR_DOC = "facebook/hf-seamless-m4t-medium"
_CONFIG_FOR_DOC = "SeamlessM4TConfig"

# 预训练模型存档列表
SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/hf-seamless-m4t-medium",  # Facebook发布的SeamlessM4T中等模型
    # 查看所有SeamlessM4T模型：https://huggingface.co/models?filter=seamless_m4t
]

# SpeechT5预训练HiFiGAN配置映射
SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_hifigan": "https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/config.json",
}

@dataclass
class SeamlessM4TGenerationOutput(ModelOutput):
    """
    定义来自[`SeamlessM4TModel`], [`SeamlessM4TForTextToText`],
    [`SeamlessM4TForTextToSpeech`], [`SeamlessM4TForSpeechToSpeech`] 和 [`SeamlessM4TForTextToSpeech`] 的生成输出类。
    继承自ModelOutput类。
    """
    # 定义函数参数和它们的类型注释，这些参数用于接收模型生成的结果
    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            模型预测的最终音频波形。
        waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
            `waveform` 批次中每个元素的样本长度，可选参数。
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            生成的翻译序列。这是文本到文本或语音到文本模型的输出。
            第二维 (sequence_length) 要么等于 `max_length`，要么由于 `eos_token_id` 导致所有批次提前结束而较短。
        unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            生成的单元序列。这是文本到单元模型的输出。
            第二维 (unit_sequence_length) 要么等于 `t2u_max_length`，要么由于 `t2u_eos_token_id` 导致所有批次提前结束而较短。
# 定义一个文档字符串，描述此模型是 PyTorch 的 torch.nn.Module 子类，详细说明如何使用和相关参数配置。
SEAMLESS_M4T_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4TConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义文档字符串的一部分，描述模型输入的第一部分，包括输入的 token 索引和输入的音频特征。
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

# 定义文档字符串的一部分，描述模型输入的文本部分，包括输入的 token 索引。
SEAMLESS_M4T_INPUTS_DOCSTRING_TEXT_PART = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """

# 定义文档字符串的一部分，描述模型输入的语音部分，包括输入的音频特征。
SEAMLESS_M4T_INPUTS_DOCSTRING_SPEECH_PART = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returned by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
        """

# 合并文档字符串的各部分，形成描述模型输入的完整文档字符串。
M4T_MODEL_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_FIRST_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART

M4T_TEXT_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_TEXT_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART

M4T_SPEECH_INPUTS_DOCSTRING = SEAMLESS_M4T_INPUTS_DOCSTRING_SPEECH_PART + SEAMLESS_M4T_INPUTS_DOCSTRING_LAST_PART


############ UTILS ################

# 从输入的 token 索引创建位置 ID 的函数，来自于 transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    """
    # 创建一个掩码，用来标识输入张量中不等于填充索引的位置
    mask = input_ids.ne(padding_idx).int()
    # 根据掩码计算累积的索引值，并在累积的基础上加上过去键值长度，然后再乘以掩码
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将累积的索引转换为长整型，并加上填充索引，得到最终的增量索引
    return incremental_indices.long() + padding_idx
# 从transformers.models.bart.modeling_bart.shift_tokens_right复制而来
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的token向右移动一位。
    """
    # 创建一个与input_ids形状相同的全零张量
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将input_ids的每一行向左移动一位赋值给shifted_input_ids的对应行
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将decoder_start_token_id赋值给shifted_input_ids的每一行的第一个元素
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果pad_token_id为None，则抛出异常
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将shifted_input_ids中可能的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    计算形式为`(batch, seq_len)`的注意力掩码，其中每个批次中的每个元素的注意力在对应的`seq_lens`停止。

    Args:
        hidden_states (`torch.FloatTensor`，形状为`(batch, seq_len, *)`)：
            要掩码的序列，其中`*`是任意数量的特定于序列的维度，包括没有。
        seq_lens (`torch.Tensor`，形状为`(batch)`)：
            每个元素表示`hidden_states`中相同索引处的序列的长度

    Returns:
        `torch.FloatTensor`：形状为`(batch, seq_len)`的浮点数注意力掩码
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    # 在设备上创建一个范围为mask_seq_len的张量，与seq_lens设备扩展
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    # 创建一个布尔掩码，其中indices大于或等于seq_lens相应位置的元素为True
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    # 创建一个形状为(batch_size, mask_seq_len)的全1张量
    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    # 将bool_mask中的True位置替换为0
    mask = mask.masked_fill(bool_mask, 0)

    return mask


def format_speech_generation_kwargs(kwargs):
    """
    为生成语音的SeamlessM4T模型格式化kwargs，将kwargs分配给文本生成或语音生成模型。

    Args:
        kwargs (`dict`)`:
             关键字参数有两种类型：

                - 没有前缀时，它们将作为每个子模型的`generate`方法的`**kwargs`输入，除了`decoder_input_ids`只会传递给文本组件。
                - 带有*text_*或*speech_*前缀时，它们将作为文本模型和语音模型的`generate`方法的输入。优先级高于没有前缀的关键字。

                这意味着您可以为一种生成策略指定一种生成，而另一种不指定。
    """
    # 为文本生成和语音生成模型分配kwargs
    kwargs_text = {}
    kwargs_speech = {}
    # 遍历关键字参数字典的每一对键值对
    for key, value in kwargs.items():
        # 如果键以"text_"开头
        if key.startswith("text_"):
            # 截取"text_"后的部分作为新的键
            key = key[len("text_") :]
            # 将对应的值存入kwargs_text字典中
            kwargs_text[key] = value
        # 如果键以"speech_"开头
        elif key.startswith("speech_"):
            # 截取"speech_"后的部分作为新的键
            key = key[len("speech_") :]
            # 将对应的值存入kwargs_speech字典中
            kwargs_speech[key] = value
        else:
            # 如果键不以"text_"或"speech_"开头
            # 如果键不存在于kwargs_text中，将其键值对存入kwargs_text中
            if key not in kwargs_text:
                kwargs_text[key] = value
            # 如果键不存在于kwargs_speech中，将其键值对存入kwargs_speech中
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    # 返回处理后的kwargs_text和kwargs_speech字典
    return kwargs_text, kwargs_speech
############ SPEECH ENCODER related code ################

# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding复制而来，修改为SeamlessM4TConformerPositionalConvEmbedding，feat_extract_activation改为speech_encoder_hidden_act
class SeamlessM4TConformerPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个1维卷积层，用于位置编码
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 初始化权重归一化函数
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 如果启用了deepspeed的zero3模式，则使用gathered parameters来管理权重
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建一个与卷积核尺寸相匹配的填充层
        self.padding = SeamlessM4TConformerSamePadLayer(config.num_conv_pos_embeddings)
        # 根据配置中的激活函数名称选择对应的激活函数
        self.activation = ACT2FN[config.speech_encoder_hidden_act]

    def forward(self, hidden_states):
        # 调整输入的hidden_states维度，将时间维度放到第二维度
        hidden_states = hidden_states.transpose(1, 2)

        # 通过卷积层进行位置编码
        hidden_states = self.conv(hidden_states)
        # 对编码后的序列进行填充处理
        hidden_states = self.padding(hidden_states)
        # 应用预先选择的激活函数
        hidden_states = self.activation(hidden_states)

        # 将处理后的hidden_states恢复原来的维度顺序
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRotaryPositionalEmbedding复制而来，修改为SeamlessM4TConformerRotaryPositionalEmbedding，num_attention_heads改为speech_encoder_attention_heads
class SeamlessM4TConformerRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.speech_encoder_attention_heads
        base = config.rotary_embedding_base

        # 计算旋转位置编码的逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None
    # 定义前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 获取隐藏状态张量的序列长度
        sequence_length = hidden_states.shape[1]

        # 如果序列长度与缓存的序列长度相同且已缓存的旋转位置嵌入不为None，则直接返回缓存的旋转位置嵌入
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        # 更新缓存的序列长度为当前序列长度
        self.cached_sequence_length = sequence_length

        # 使用inv_freq常量的dtype计算时间戳
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        # 计算频率矩阵，使用torch.einsum执行张量乘法
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        # 将cos和sin嵌入拼接在一起
        embeddings = torch.cat((freqs, freqs), dim=-1)

        # 计算cos和sin的嵌入
        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]

        # 将计算得到的嵌入堆叠为旋转位置嵌入，并使用隐藏状态的dtype
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)

        # 返回计算得到的旋转位置嵌入
        return self.cached_rotary_positional_embedding
# 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRelPositionalEmbedding 复制到 SeamlessM4TConformerRelPositionalEmbedding，并将 Wav2Vec2 替换为 SeamlessM4T
class SeamlessM4TConformerRelPositionalEmbedding(nn.Module):
    """相对位置编码模块。"""

    def __init__(self, config):
        super().__init__()
        # 最大位置编码长度
        self.max_len = config.max_source_positions
        # 隐藏层大小
        self.d_model = config.hidden_size
        # 位置编码（初始化为 None）
        self.pe = None
        # 扩展位置编码
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        # 重置位置编码
        if self.pe is not None:
            # self.pe 包含正负两部分
            # self.pe 的长度为 2 * 输入长度 - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # 创建正位置编码和负位置编码
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

        # 反转正位置编码的顺序并拼接正负编码
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        # 扩展位置编码以匹配输入张量长度
        self.extend_pe(hidden_states)
        start_idx = self.pe.size(1) // 2 - hidden_states.size(1) + 1
        end_idx = self.pe.size(1) // 2 + hidden_states.size(1)
        # 提取相对位置编码
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


# 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSamePadLayer 复制到 SeamlessM4TConformerSamePadLayer，并将 Wav2Vec2 替换为 SeamlessM4T
class SeamlessM4TConformerSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 如果卷积位置编码数为偶数，则移除一个填充元素
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0
    # 定义一个方法 `forward`，用于前向传播计算
    def forward(self, hidden_states):
        # 检查是否需要移除末尾的填充元素
        if self.num_pad_remove > 0:
            # 如果需要移除填充元素，截取隐藏状态中的有效部分
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        # 返回处理后的隐藏状态
        return hidden_states
# 在 SeamlessM4TConformerFeatureProjection 类中定义初始化方法
class SeamlessM4TConformerFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 LayerNorm 层，用于归一化输入维度的隐藏状态
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        # 初始化线性投影层，将输入维度投影到隐藏层维度
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        # 初始化 Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    # 前向传播函数，接收隐藏状态作为输入，经过归一化、投影和随机失活后输出
    def forward(self, hidden_states):
        # 对未投影的隐藏状态进行归一化，用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的隐藏状态投影到目标维度
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态进行随机失活处理
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 在 SeamlessM4TConformerFeedForward 类中定义初始化方法
class SeamlessM4TConformerFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, dropout=None):
        super().__init__()
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act

        # 初始化中间层的 Dropout 层，用于随机失活
        self.intermediate_dropout = nn.Dropout(dropout)
        # 初始化中间层的线性层，将隐藏状态投影到中间层的大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        # 初始化中间层的激活函数，将其设为配置中指定的函数或字符串对应的函数
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn

        # 初始化输出层的线性层，将中间层的大小投影回隐藏状态的大小
        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        # 初始化输出层的 Dropout 层，用于随机失活
        self.output_dropout = nn.Dropout(dropout)

    # 前向传播函数，接收隐藏状态作为输入，经过中间层和输出层处理后输出
    def forward(self, hidden_states):
        # 将隐藏状态传入中间层的线性层进行处理
        hidden_states = self.intermediate_dense(hidden_states)
        # 将中间层处理后的结果经过激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对中间层的输出进行随机失活处理
        hidden_states = self.intermediate_dropout(hidden_states)

        # 将中间层处理后的结果传入输出层的线性层进行处理
        hidden_states = self.output_dense(hidden_states)
        # 对输出层的结果进行随机失活处理
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# 在 SeamlessM4TConformerConvolutionModule 类中定义空的文档字符串作为说明
class SeamlessM4TConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""
    def __init__(self, config):
        super().__init__()
        # 检查深度卷积核大小是否为奇数，以确保 'SAME' 填充有效
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        
        # 初始化层归一化模块
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 第一个逐点卷积层，输出通道数为输入通道数的两倍
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
        
        # 深度可分离卷积层
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding="same",
            groups=config.hidden_size,
            bias=False,
        )
        
        # 批归一化层
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        
        # 激活函数
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        
        # 第二个逐点卷积层
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        
        # Dropout层
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states, attention_mask=None):
        # 应用层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 确保在深度卷积中不泄露填充位置信息，将填充位置置为0
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # 交换时间维度和特征维度
        hidden_states = hidden_states.transpose(1, 2)

        # GLU机制，将通道维度扩展为原来的两倍
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        # 1维深度可分离卷积
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        # 第二个逐点卷积层
        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 再次交换时间维度和特征维度
        hidden_states = hidden_states.transpose(1, 2)
        
        return hidden_states
        # 构造 SeamlessM4TConformerSelfAttention 类的对象，可增强旋转或相对位置编码
        """Construct a SeamlessM4TConformerSelfAttention object.
        Can be enhanced with rotary or relative position embeddings.
        """

        # 初始化函数，设置对象的参数和层
        def __init__(self, config, use_position_embeddings=True):
            super().__init__()

            # 根据配置计算每个头部的大小
            self.head_size = config.hidden_size // config.speech_encoder_attention_heads
            # 设置注意力头部的数量
            self.num_heads = config.speech_encoder_attention_heads
            # 如果使用位置编码则设置位置编码的类型
            self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None

            # 初始化线性层，用于查询、键、值和输出
            self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

            # 初始化 dropout 层
            self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

            # 如果位置编码类型是 "relative"，则需要额外的位置编码处理
            if self.position_embeddings_type == "relative":
                # 初始化用于位置编码的线性层
                self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                # 初始化用于矩阵 c 和矩阵 d 的可学习偏置，详见论文 https://arxiv.org/abs/1901.02860 第 3.3 节
                self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
                self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

        # 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention.forward 复制而来
        # 定义前向传播函数，接收隐藏状态、注意力掩码、相对位置编码和是否输出注意力权重
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            relative_position_embeddings: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()
        
        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states
        
        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            # 如果位置编码类型是"rotary"，则将rotary位置编码应用于查询/键状态
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)
        
        # project query_key_states and value_states
        # 将query_key_states和value_states投影
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)
        
        # => (batch, head, time1, d_k)
        # 调整维度顺序以适应注意力计算要求
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'relative'"
                )
            # 如果位置编码类型是"relative"，则将相对位置编码应用于查询和键的分数
            # 基于Transformer_XL的建议：https://arxiv.org/abs/1901.02860
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            # 默认情况下，计算注意力分数
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        # apply attention_mask if necessary
        # 如果存在attention_mask，则应用它
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # => (batch, head, time1, time2)
        # 计算注意力权重
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        # => (batch, head, time1, d_k)
        # 计算加权值
        hidden_states = torch.matmul(probs, value)
        
        # => (batch, time1, hidden_size)
        # 调整维度以便输出
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)
        
        # 返回最终输出
        return hidden_states, probs
    # 定义一个方法来应用旋转嵌入到隐藏状态中，这里使用相对位置嵌入作为输入
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        # 获取隐藏状态的维度信息：批量大小、序列长度、隐藏大小
        batch_size, sequence_length, hidden_size = hidden_states.size()
        # 将隐藏状态重塑为多头和头部大小的形状
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        # 从相对位置嵌入中获取余弦值和正弦值
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # 使用旋转嵌入来旋转隐藏状态
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        # 将隐藏状态重新展平为(batch_size, sequence_length, num_heads * head_size)的形状
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        # 返回应用旋转嵌入后的隐藏状态
        return hidden_states

    # 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_relative_embeddings 处复制过来的注释
    `
        def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
            # 1. project positional embeddings
            # 将相对位置嵌入通过线性层进行投影，形状变为(batch, head, 2*time1-1, d_k)
            proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
            proj_relative_position_embeddings = proj_relative_position_embeddings.view(
                relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
            )
            proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
            proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)
    
            # 2. Add bias to query
            # 将查询向量的维度调整后，添加位置偏置，形状变为(batch, head, time1, d_k)
            query = query.transpose(1, 2)
            q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
            q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)
    
            # 3. attention score: first compute matrix a and matrix c
            # 计算矩阵a和矩阵c，按照 https://arxiv.org/abs/1901.02860 第三节的描述进行计算，形状为(batch, head, time1, time2)
            scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))
    
            # 4. then compute matrix b and matrix d
            # 计算矩阵b和矩阵d，形状为(batch, head, time1, 2*time1-1)
            scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)
    
            # 5. shift matrix b and matrix d
            # 创建一个零填充张量，使scores_bd的维度变为(batch, head, time1, 2*time1)
            zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
            scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
            scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
            scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
            scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
            scores_bd = scores_bd[:, :, :, : scores_bd.size(-1) // 2 + 1]
    
            # 6. sum matrices
            # 将矩阵a和矩阵b相加，并除以sqrt(head_size)，得到最终的注意力得分，形状为(batch, head, time1, time2)
            scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)
    
            return scores
# Conformer 编码器层，基于 https://arxiv.org/abs/2005.08100 提出的 Conformer 结构
class SeamlessM4TConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    # 从 transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ 复制而来，做了以下替换：Wav2Vec2->SeamlessM4T, attention_dropout->speech_encoder_dropout, torch.nn->nn
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size  # 从配置中获取隐藏层大小作为嵌入维度
        dropout = config.speech_encoder_dropout  # 从配置中获取dropout率

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)  # Layer normalization 层
        self.ffn1 = SeamlessM4TConformerFeedForward(config)  # 第一个前馈网络

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)  # Layer normalization 层
        self.self_attn_dropout = nn.Dropout(dropout)  # Dropout 层
        self.self_attn = SeamlessM4TConformerSelfAttention(config)  # Conformer 自注意力机制

        # Conformer Convolution
        self.conv_module = SeamlessM4TConformerConvolutionModule(config)  # Conformer 卷积模块

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)  # Layer normalization 层
        self.ffn2 = SeamlessM4TConformerFeedForward(config)  # 第二个前馈网络
        self.final_layer_norm = nn.LayerNorm(embed_dim)  # 最终的 Layer normalization 层

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states  # 输入的隐藏状态

        # 1. Feed-Forward 1 层
        residual = hidden_states  # 残差连接
        hidden_states = self.ffn1_layer_norm(hidden_states)  # Layer normalization
        hidden_states = self.ffn1(hidden_states)  # 第一个前馈网络
        hidden_states = hidden_states * 0.5 + residual  # 残差连接和缩放

        residual = hidden_states  # 更新残差连接

        # 2. Self-Attention 层
        hidden_states = self.self_attn_layer_norm(hidden_states)  # Layer normalization
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )  # Conformer 自注意力机制
        hidden_states = self.self_attn_dropout(hidden_states)  # Dropout
        hidden_states = hidden_states + residual  # 残差连接

        # 3. 卷积层
        residual = hidden_states  # 残差连接
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)  # Conformer 卷积模块
        hidden_states = residual + hidden_states  # 残差连接

        # 4. Feed-Forward 2 层
        residual = hidden_states  # 残差连接
        hidden_states = self.ffn2_layer_norm(hidden_states)  # Layer normalization
        hidden_states = self.ffn2(hidden_states)  # 第二个前馈网络
        hidden_states = hidden_states * 0.5 + residual  # 残差连接和缩放
        hidden_states = self.final_layer_norm(hidden_states)  # 最终的 Layer normalization

        return hidden_states, attn_weights  # 返回隐藏状态和注意力权重
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置对象保存到实例变量中
        self.config = config

        # 根据配置中的位置嵌入类型选择不同的位置嵌入方式
        if config.position_embeddings_type == "relative":
            # 如果位置嵌入类型为"relative"，则使用 SeamlessM4TConformerRelPositionalEmbedding 初始化位置嵌入
            self.embed_positions = SeamlessM4TConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            # 如果位置嵌入类型为"rotary"，则使用 SeamlessM4TConformerRotaryPositionalEmbedding 初始化位置嵌入
            self.embed_positions = SeamlessM4TConformerRotaryPositionalEmbedding(config)
        else:
            # 否则，位置嵌入为 None
            self.embed_positions = None

        # 初始化 dropout 层，根据配置中的 dropout 率
        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        
        # 使用 nn.ModuleList 初始化一个编码层列表，包含 config.speech_encoder_layers 个 SeamlessM4TConformerEncoderLayer 层
        self.layers = nn.ModuleList(
            [SeamlessM4TConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        # 初始化 LayerNorm 层，用于规范化隐藏状态，设置 epsilon 为 config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化梯度检查点，默认为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接受隐藏状态和可选的注意力掩码等参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
# 定义一个名为 SeamlessM4TConformerAdapterLayer 的类，继承自 nn.Module
class SeamlessM4TConformerAdapterLayer(nn.Module):
    # 初始化方法，接收一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置中获取隐藏层大小作为嵌入维度
        embed_dim = config.hidden_size
        # 从配置中获取适配器的 dropout 率
        dropout = config.adaptor_dropout

        # 定义卷积核大小和步幅
        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride

        # 1. 残差卷积层
        # 使用 LayerNorm 对嵌入维度进行归一化
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        # 定义一个 1 维卷积层，输入维度为 embed_dim，输出维度为 2 * embed_dim
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 定义激活函数 GLU
        self.activation = nn.GLU(dim=1)

        # Self-Attention
        # 使用 LayerNorm 对嵌入维度进行归一化
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        # 定义一个 1 维卷积层，输入维度为 embed_dim，输出维度为 2 * embed_dim
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 定义自注意力层
        self.self_attn = SeamlessM4TConformerSelfAttention(config, use_position_embeddings=False)
        # 定义 dropout 层
        self.self_attn_dropout = nn.Dropout(dropout)

        # Feed-forward
        # 使用 LayerNorm 对嵌入维度进行归一化
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        # 定义一个前馈网络层
        self.ffn = SeamlessM4TConformerFeedForward(config, act_fn="relu", dropout=dropout)

    # 计算从注意力掩码中计算子采样长度的私有方法
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        # 计算填充量
        pad = self.kernel_size // 2
        # 计算每个序列的长度，减去不关注部分后的有效长度
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
        # 根据卷积步幅计算子采样长度
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1
        # 向下取整并返回子采样长度
        return seq_lens.floor()

    # 前向传播方法，接收隐藏状态、注意力掩码和是否输出注意力作为参数
    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 对输入的 hidden_states 应用残差层归一化
        residual = self.residual_layer_norm(hidden_states)

        # 对残差进行池化操作，使其与多头注意力输出的序列长度相匹配。
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        # 对输入的 hidden_states 应用自注意力层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 在输入多头注意力层之前应用池化操作。
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        # 如果有注意力掩码，则计算子采样长度并准备新的注意力掩码
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                hidden_states.device
            )
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # 剩余的计算步骤与普通的Transformer编码器层完全相同。
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 将残差更新为当前隐藏状态
        residual = hidden_states

        # 对隐藏状态应用前馈神经网络层归一化
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states
class SeamlessM4TConformerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 使用列表推导式创建包含多个 SeamlessM4TConformerAdapterLayer 的 ModuleList
        self.layers = nn.ModuleList(SeamlessM4TConformerAdapterLayer(config) for _ in range(config.num_adapter_layers))

    def forward(self, hidden_states, attention_mask):
        # 如果需要，对 hidden_states 进行下投影处理

        # 遍历每个适配器层，依次对 hidden_states 和 attention_mask 进行处理
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # 返回处理后的 hidden_states
        return hidden_states


############ TEXT / UNITS related code ################


# 从 transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding 复制而来
class SeamlessM4TSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # 调用 make_weights 方法创建权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 获取 sinusoidal embeddings 的权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 如果已经存在 weights 属性，在 forward 方法中将其转换为正确的 dtype 和 device
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 注册权重为 buffer，非持久性
        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        构建 sinusoidal embeddings，与 tensor2tensor 中的实现相匹配，但与 "Attention Is All You Need" 第 3.5 节中的描述略有不同。
        """
        half_dim = embedding_dim // 2
        # 计算 sinusoidal embeddings 的权重
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # 如果 embedding_dim 是奇数，则进行零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            # 如果存在 padding_idx，则将对应位置的 embeddings 设为零向量
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
        ):
        # 如果输入的 token ids 不为 None，则获取 batch size 和 sequence length
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # 根据输入的 token ids 创建位置 ids。任何填充的 token 仍然保持填充状态。
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            # 否则，获取输入嵌入张量的 batch size 和 sequence length（除最后一个维度外的所有维度）
            bsz, seq_len = inputs_embeds.size()[:-1]
            # 根据输入的嵌入张量创建位置 ids
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # 如果需要，扩展嵌入张量
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            # 调整权重张量的大小，以适应更大的位置数
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 根据位置 ids 从权重张量中选择对应的嵌入向量，并调整形状为 (batch size, sequence length, embedding dim)
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入张量的形状（除最后一个维度外的所有维度）
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成连续的位置 ids，从 padding_idx + 1 到 sequence_length + padding_idx + 1
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置 ids 扩展到与输入形状相同，并确保连续性，然后加上 past_key_values_length
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
class SeamlessM4TAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.bart.modeling_bart.BartAttention.__init__ with Bart->SeamlessM4T
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
        self.embed_dim = embed_dim  # 设置注意力机制的嵌入维度
        self.num_heads = num_heads  # 设置注意力头的数量
        self.dropout = dropout  # 设置dropout概率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子的计算
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否是因果（causal）的注意力

        # 初始化线性变换层，用于对输入进行键、值和查询的线性投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 对输入的张量进行形状重塑，以便进行多头注意力的并行计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 执行注意力机制的前向传播
        # hidden_states: 输入的隐藏状态张量
        # encoder_hidden_states: 编码器的隐藏状态张量（可选）
        # past_key_value: 缓存的键值对（可选）
        # attention_mask: 注意力掩码（可选）
        # output_attentions: 是否输出注意力权重（可选）



# Copied from transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeDenseActDense with NllbMoe->SeamlessM4T,DenseActDense->FeedForwardNetwork, d_model->hidden_size
class SeamlessM4TFeedForwardNetwork(nn.Module):
    def __init__(self, config: SeamlessM4TConfig, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, ffn_dim)  # 第一个全连接层，将隐藏状态映射到FFN维度
        self.fc2 = nn.Linear(ffn_dim, config.hidden_size)  # 第二个全连接层，将FFN维度映射回隐藏状态维度
        self.dropout = nn.Dropout(config.activation_dropout)  # dropout层
        self.act = ACT2FN[config.activation_function]  # 激活函数选择

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)  # 第一个全连接层的前向传播
        hidden_states = self.act(hidden_states)  # 应用激活函数
        hidden_states = self.dropout(hidden_states)  # 应用dropout
        if (
            isinstance(self.fc2.weight, torch.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8)
        ):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)  # 转换张量类型以匹配第二个全连接层
        hidden_states = self.fc2(hidden_states)  # 第二个全连接层的前向传播
        return hidden_states
    # 初始化函数，接受配置信息和可选的编码器参数维度和注意力头数
    def __init__(self, config: SeamlessM4TConfig, encoder_ffn_dim=None, encoder_attention_heads=None):
        # 调用父类的初始化方法
        super().__init__()

        # 如果未提供编码器前馈网络维度，则使用配置中的值
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        # 如果未提供编码器注意力头数，则使用配置中的值
        encoder_attention_heads = (
            config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        )

        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 初始化自注意力层
        self.self_attn = SeamlessM4TAttention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 初始化注意力的 dropout 层
        self.attn_dropout = nn.Dropout(config.dropout)
        # 初始化自注意力层的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化前馈网络层
        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)
        # 初始化前馈网络的 LayerNorm
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        # 初始化前馈网络的 dropout 层
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
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
        """
        # 保留残差连接
        residual = hidden_states
        # 对输入的 hidden_states 进行 LayerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用自注意力层进行计算，并返回注意力权重
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 对自注意力层的输出应用 dropout
        hidden_states = self.attn_dropout(hidden_states)
        # 加上残差连接
        hidden_states = residual + hidden_states

        # 保留残差连接
        residual = hidden_states

        # 对前馈网络输入进行 LayerNorm
        hidden_states = self.ffn_layer_norm(hidden_states)
        # 使用前馈网络进行计算
        hidden_states = self.ffn(hidden_states)
        # 对前馈网络的输出应用 dropout
        hidden_states = self.ffn_dropout(hidden_states)

        # 加上残差连接
        hidden_states = residual + hidden_states

        # 输出结果作为元组的第一个元素
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将它们作为元组的第二个元素添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义 SeamlessM4TDecoderLayer 类，继承自 nn.Module，用于实现 SeamlessM4T 模型的解码器层
class SeamlessM4TDecoderLayer(nn.Module):
    def __init__(self, config: SeamlessM4TConfig, decoder_ffn_dim=None, decoder_attention_heads=None):
        super().__init__()
        # 如果未提供 decoder_ffn_dim 参数，则使用 config 中的值
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        # 如果未提供 decoder_attention_heads 参数，则使用 config 中的值
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )

        # 设置嵌入维度为 config 中的隐藏大小
        self.embed_dim = config.hidden_size
        # 创建自注意力层，用于解码器自身的注意力机制
        self.self_attn = SeamlessM4TAttention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 设置丢弃率
        self.dropout = config.dropout
        # 激活函数根据配置文件中的激活函数选择
        self.activation_fn = ACT2FN[config.activation_function]
        # 注意力机制的丢弃层
        self.attn_dropout = nn.Dropout(config.dropout)

        # 创建自注意力层规范化层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建交叉注意力层，用于解码器与编码器之间的信息交换
        self.cross_attention = SeamlessM4TAttention(
            self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        # 创建交叉注意力层规范化层
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建前馈神经网络层
        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)

        # 创建前馈神经网络层规范化层
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        # 前馈神经网络的丢弃层
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    # 定义前向传播函数
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
        # 此处是模型的具体前向传播逻辑，涉及输入的张量、注意力掩码等
        pass  # 在实现具体逻辑时填充



# 定义 SeamlessM4TPreTrainedModel 类，继承自 PreTrainedModel，用作处理权重初始化和预训练模型下载加载的抽象类
class SeamlessM4TPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为 SeamlessM4TConfig
    config_class = SeamlessM4TConfig
    # 基础模型前缀为 "seamless_m4t"
    base_model_prefix = "seamless_m4t"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不拆分的模块列表，用于梯度检查点
    _no_split_modules = ["SeamlessM4TEncoderLayer", "SeamlessM4TDecoderLayer", "SeamlessM4TConformerEncoderLayer"]
    def _init_weights(self, module):
        """初始化模型权重"""
        # 从配置中获取初始化范围的标准差
        std = self.config.initializer_range
        
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果定义了padding_idx，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        # 如果是自定义的Self-Attention层
        elif isinstance(module, SeamlessM4TConformerSelfAttention):
            # 如果有位置偏置u，使用均匀分布的Xavier初始化
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            # 如果有位置偏置v，使用均匀分布的Xavier初始化
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        
        # 如果是自定义的Positional Convolution Embedding层
        elif isinstance(module, SeamlessM4TConformerPositionalConvEmbedding):
            # 使用正态分布初始化卷积层的权重，均值为0，标准差根据卷积核大小和输入通道数确定
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积层的偏置初始化为零
            nn.init.constant_(module.conv.bias, 0)
        
        # 如果是自定义的Feature Projection层
        elif isinstance(module, SeamlessM4TConformerFeatureProjection):
            # 计算均匀分布的边界k
            k = math.sqrt(1 / module.projection.in_features)
            # 使用均匀分布初始化投影层的权重和偏置
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        
        # 如果是LayerNorm或者GroupNorm层
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1.0
            module.weight.data.fill_(1.0)
        
        # 如果是一维卷积层
        elif isinstance(module, nn.Conv1d):
            # 使用Kaiming正态分布初始化卷积层的权重
            nn.init.kaiming_normal_(module.weight)
            # 如果存在偏置项，使用均匀分布初始化其值，范围根据组数、输入通道数和卷积核大小确定
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
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
        # For each generation step, concatenate the hidden states from the last layer.
        # shape: (batch_size*num_return_sequences, sequence_length, hidden_size)
        last_hidden_states = torch.concat([hidden_states[-1] for hidden_states in hidden_states], dim=1)

        # 2. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach where the first (and only) beam is always selected.
        # In that case, return directly last_hidden_states.
        if beam_indices is None:
            return last_hidden_states

        # 3. Cut beam_indices to the longest beam length.
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 4. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards anyways.
        beam_indices[beam_indices_mask] = 0

        # 5. Expand beam_indices to match the dimensions of last_hidden_states.
        # beam_indices shape: (batch_size*num_return_sequences, max_beam_length, hidden_size)
        beam_indices = beam_indices.unsqueeze(-1)
        beam_indices = beam_indices.expand(-1, -1, last_hidden_states.shape[-1])

        # 6. Select the right candidate for each beam.
        # In other words, new_last_hidden_states[i,j,k] = last_hidden_states[beam_indices[i,j,k], j, k] for all i, j, k.
        last_hidden_states = torch.gather(last_hidden_states, 0, beam_indices)

        return last_hidden_states
# 使用装饰器为类添加文档字符串，描述该类是一个 Transformer 语音编码器，由 config.speech_encoder_layers 个 Conformer 自注意力层组成。
# 每一层是一个 `SeamlessM4TConformerEncoderLayer`。

# 引入 SEAMLESS_M4T_START_DOCSTRING 定义的文档字符串作为类文档的一部分
class SeamlessM4TSpeechEncoder(SeamlessM4TPreTrainedModel):
    # 主要输入特征的名称
    main_input_name = "input_features"

    # 初始化方法，接受一个 SeamlessM4TConfig 类型的 config 参数
    def __init__(self, config: SeamlessM4TConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 特征投影层，使用 SeamlessM4TConformerFeatureProjection 类根据 config 初始化
        self.feature_projection = SeamlessM4TConformerFeatureProjection(config)
        
        # 编码器层，使用 SeamlessM4TConformerEncoder 类根据 config 初始化
        self.encoder = SeamlessM4TConformerEncoder(config)
        
        # 中间的前馈神经网络层，使用 SeamlessM4TConformerFeedForward 类根据 config 初始化，
        # 激活函数为 "relu"，dropout 率为 0.0
        self.intermediate_ffn = SeamlessM4TConformerFeedForward(config, act_fn="relu", dropout=0.0)
        
        # 如果 config 中指定要添加适配器，则使用 SeamlessM4TConformerAdapter 类根据 config 初始化适配器，否则为 None
        self.adapter = SeamlessM4TConformerAdapter(config) if config.add_adapter else None
        
        # 内部的 LayerNorm 层，对隐藏状态进行标准化，隐藏单元数目为 config.hidden_size
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_features: Optional[torch.Tensor],  # 输入特征，类型为 torch.Tensor，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，类型为 torch.Tensor，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为 bool，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为 bool，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果，类型为 bool，可选
        **kwargs,  # 其他关键字参数
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:  # 返回值类型为 Tuple 或 Wav2Vec2BaseModelOutput 类型
        # 如果 input_features 为 None，则抛出 ValueError 异常
        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4TSpeechEncoder.forward`.
                Make sure one of them is not `None`."""
            )

        # 使用特征投影层对输入特征进行处理，得到隐藏状态
        hidden_states = self.feature_projection(input_features)

        # 使用编码器层对隐藏状态进行编码，得到编码器的输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从编码器输出中获取隐藏状态
        hidden_states = encoder_outputs[0]

        # 使用中间的前馈神经网络层对隐藏状态进行扩展处理
        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        
        # 将扩展后的隐藏状态与原始隐藏状态加权相加
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

        # 如果存在适配器，则使用适配器对隐藏状态进行处理
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        # 对隐藏状态进行内部的 LayerNorm 处理
        hidden_states = self.inner_layer_norm(hidden_states)

        # 如果 return_dict 为 False，则返回一个元组，包含隐藏状态和 encoder_outputs 的其它部分
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回一个 Wav2Vec2BaseModelOutput 对象，
        # 包含最终的隐藏状态、编码器的隐藏状态和注意力权重
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`SeamlessM4TEncoderLayer`].
    """
    """
    SEAMLESS_M4T_START_DOCSTRING,
    """
    """
    embed_tokens (`nn.Embedding`, *optional*):
        Input embedding
    is_t2u_encoder (`bool`, *optional*, defaults to `False`):
        indicates if it belongs to the text-to-units model, in which case it won't have input embeddings
    """
@add_start_docstrings(
    "Transformer encoder consisting of *config.encoder_layers* layers. Each layer is a [`SeamlessM4TEncoderLayer`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
    """,
)
class SeamlessM4TEncoder(SeamlessM4TPreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        is_t2u_encoder: bool = False,
    ):
        super().__init__(config)

        self.dropout = config.dropout  # 设置dropout概率
        self.layerdrop = config.encoder_layerdrop  # 获取encoder层的layerdrop参数
        self.padding_idx = config.pad_token_id  # 获取pad token的索引
        embed_dim = config.hidden_size  # 获取嵌入维度大小

        self.is_t2u_encoder = is_t2u_encoder  # 判断是否为t2u编码器
        self.max_source_positions = config.max_position_embeddings  # 获取最大位置嵌入数

        if not self.is_t2u_encoder:
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 如果设置了scale_embedding，则设置嵌入缩放因子

            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)  # 创建词嵌入层

            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight  # 如果提供了外部的embed_tokens，则使用其权重

            self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
            )  # 创建无缝M4T正弦位置嵌入层

        layers = []
        for _ in range(config.encoder_layers):
            layers.append(
                SeamlessM4TEncoderLayer(
                    config,
                    encoder_attention_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                )
            )  # 创建指定数量的编码器层

        self.layers = nn.ModuleList(layers)  # 将编码器层列表转换为模块列表

        self.layer_norm = nn.LayerNorm(config.hidden_size)  # 创建层归一化层

        self.gradient_checkpointing = False  # 设置梯度检查点为False，表示不使用梯度检查点
        # 初始化权重并应用最终处理
        self.post_init()
    ):
        # 调用父类的初始化方法，传入配置信息
        super().__init__(config)
        # 从配置中获取丢弃率，解码器层丢弃率，填充标记索引，词汇表大小，最大目标位置和嵌入缩放因子
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # 如果提供了 embed_tokens，则使用其形状来定义嵌入层
            self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight
        else:
            # 否则，创建一个新的嵌入层，使用配置中的隐藏大小和词汇表大小
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        # 创建位置嵌入层，使用 SeamlessM4TSinusoidalPositionalEmbedding 类来处理
        self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        layers = []
        # 根据配置的解码器层数，循环创建解码器层，并添加到 layers 列表中
        for _ in range(config.decoder_layers):
            layers.append(
                SeamlessM4TDecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )
        # 将所有解码器层作为一个 nn.ModuleList
        self.layers = nn.ModuleList(layers)
        # 初始化层归一化，使用配置中的隐藏大小
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层对象 embed_tokens
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # 设置嵌入层 embed_tokens 的值为指定的 value
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
# 使用装饰器为类添加文档字符串，描述了该模型的基本结构和组件
@add_start_docstrings(
    "Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4TEncoder`] without embeddings and the decoder is a [`SeamlessM4TDecoder`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
# 定义了一个名为 SeamlessM4TTextToUnitModel 的新模型类，继承自 SeamlessM4TPreTrainedModel
class SeamlessM4TTextToUnitModel(SeamlessM4TPreTrainedModel):
    
    # 初始化方法，接受配置参数 config 和可选的解码器嵌入 embed_tokens_decoder
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        # 调用父类构造方法，传入配置参数 config
        super().__init__(config)

        # 创建编码器，使用 SeamlessM4TEncoder 类构造，标志 is_t2u_encoder 设置为 True
        self.encoder = SeamlessM4TEncoder(config, is_t2u_encoder=True)
        
        # 创建解码器，使用 SeamlessM4TDecoder 类构造，传入解码器嵌入 embed_tokens_decoder
        self.decoder = SeamlessM4TDecoder(config, embed_tokens_decoder)

        # 初始化模型权重并应用最终处理
        self.post_init()

    # 前向传播方法定义，接受多个输入参数和可选的输出配置
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
        # 其余参数用于控制模型的具体行为，例如缓存、注意力、隐藏状态和返回格式等
    # 定义一个方法，接受输入参数并返回一个元组或者Seq2SeqModelOutput类型的对象
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        # 如果output_attentions参数不为空，则使用该参数，否则使用配置中的output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states参数不为空，则使用该参数，否则使用配置中的output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果use_cache参数不为空，则使用该参数，否则使用配置中的use_cache
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果return_dict参数不为空，则使用该参数，否则使用配置中的use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 如果encoder_outputs为空，则调用self.encoder进行编码
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # 如果return_dict为True且encoder_outputs不是BaseModelOutput类型的实例，则封装为BaseModelOutput对象
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
    
        # decoder_outputs包含(dec_features, past_key_value, dec_hidden, dec_attn)
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
    
        # 如果return_dict为False，则返回decoder_outputs和encoder_outputs的组合
        if not return_dict:
            return decoder_outputs + encoder_outputs
    
        # 如果return_dict为True，则返回Seq2SeqModelOutput对象，其中包括decoder和encoder的相关输出
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
# 定义一个类，继承自SeamlessM4TPreTrainedModel，实现了文本到单元的编码-解码器模型，带有语言模型头部。
# 基础的编码-解码模型是`SeamlessM4TTextToUnit`。
@add_start_docstrings(
    "Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4TTextToUnit`].",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
class SeamlessM4TTextToUnitForConditionalGeneration(SeamlessM4TPreTrainedModel):
    # 在加载模型时要忽略的键列表
    _keys_to_ignore_on_load_missing = [
        "vocoder",
        "speech_encoder",
        "text_encoder",
        "text_decoder",
    ]
    # 共享权重的键列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(
        self,
        config: SeamlessM4TConfig,  # 传入的配置对象
        embed_tokens_decoder: Optional[nn.Embedding] = None,  # 解码器的嵌入层，可选
    ):
        # 深拷贝配置对象
        config = copy.deepcopy(config)
        # 针对配置对象的参数进行更新，主要用于bos_token_id等
        for param, val in config.to_dict().items():
            if param.startswith("t2u_"):
                config.__setattr__(param[4:], val)
        # 调用父类构造函数，初始化模型
        super().__init__(config)

        # 创建SeamlessM4TTextToUnitModel对象作为模型成员
        self.model = SeamlessM4TTextToUnitModel(config, embed_tokens_decoder)

        # 初始化语言模型头部，线性层
        self.lm_head = nn.Linear(config.hidden_size, config.t2u_vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器部分
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器部分
    def get_decoder(self):
        return self.model.decoder

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输入嵌入层（解码器的嵌入层）
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层（解码器的嵌入层）
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 前向传播函数，接收多个输入和输出参数，执行编码-解码过程
    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入token ID
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器的输出
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 用于存储过去的键-值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器的输入嵌入表示
        labels: Optional[torch.LongTensor] = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
        # ...
    # 定义函数 prepare_inputs_for_generation，准备生成模型输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的 token IDs
        past_key_values=None,  # 用于缓存的先前键值对
        attention_mask=None,  # 注意力掩码，指示哪些位置的 token 应被忽略
        use_cache=None,  # 是否使用缓存
        encoder_outputs=None,  # 编码器的输出，作为生成输入的一部分
        **kwargs,  # 其它关键字参数，这里未使用
    ):
        # 如果已经提供了 past_key_values，则截取 decoder_input_ids 的最后一个 token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个包含生成所需输入的字典
        return {
            "input_ids": None,  # 由于定义了 encoder_outputs，因此不需要 input_ids
            "encoder_outputs": encoder_outputs,  # 编码器的输出作为输入的一部分
            "past_key_values": past_key_values,  # 用于缓存的先前键值对
            "decoder_input_ids": decoder_input_ids,  # 解码器的输入 token IDs
            "attention_mask": attention_mask,  # 注意力掩码，指示哪些 token 应被忽略
            "use_cache": use_cache,  # 是否使用缓存
        }
    # 根据标签准备解码器的输入标识符列表，将标签向右移动一位，并在开头添加解码器起始标记
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id)

    @staticmethod
    # 重新排序缓存中的过去键值，根据beam_idx重新排列每个层级的过去键值
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终保持不变
            reordered_past += (
                # 选择对应beam_idx的过去状态的第0和第1元素，并保留其余的不变
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # 绑定模型的权重，如果配置中允许词嵌入绑定，则将输出词嵌入与输入词嵌入绑定
    def _tie_weights(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
############ VOCODER related code ################

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

# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
# 定义了一个 HifiGanResidualBlock 类，继承自 nn.Module
class HifiGanResidualBlock(nn.Module):
    # 初始化方法，接收 channels（通道数）、kernel_size（卷积核大小）、dilation（扩张率列表）、leaky_relu_slope（LeakyReLU 的负斜率）等参数
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope  # 初始化 LeakyReLU 的负斜率

        # 定义多个卷积层组成的 ModuleList，每个卷积层具有不同的 dilation（扩张率）
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
        
        # 定义另一个 ModuleList，每个卷积层具有相同的 dilation=1（即不扩张）
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

    # 计算卷积层的 padding 数量
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 应用权重归一化到所有的 convs1 和 convs2 卷积层
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除所有 convs1 和 convs2 卷积层的权重归一化
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    # 前向传播方法，接收 hidden_states 作为输入
    def forward(self, hidden_states):
        # 遍历 convs1 和 convs2 的每对卷积层
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states  # 保存残差
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # LeakyReLU 激活函数
            hidden_states = conv1(hidden_states)  # 第一层卷积
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # LeakyReLU 激活函数
            hidden_states = conv2(hidden_states)  # 第二层卷积
            hidden_states = hidden_states + residual  # 加上残差
        return hidden_states  # 返回最终的 hidden_states
# 定义一个名为 SeamlessM4TVariancePredictor 的 PyTorch 模型类
class SeamlessM4TVariancePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 从配置中获取嵌入维度
        embed_dim = config.unit_embed_dim
        # 获取方差预测器的卷积核大小
        kernel_size = config.variance_predictor_kernel_size
        # 获取方差预测器的dropout率
        var_pred_dropout = config.var_pred_dropout

        # 第一个卷积层，输入和输出维度都是 embed_dim，卷积核大小为 kernel_size
        self.conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        # 激活函数使用 ReLU
        self.activation_fuction = nn.ReLU()
        # LayerNorm 层，标准化 embed_dim 维度的数据
        self.ln1 = nn.LayerNorm(embed_dim)
        # dropout 模块，应用于变量预测器
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        # 第二个卷积层，输入和输出维度都是 embed_dim，卷积核大小为 kernel_size
        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=1,
        )
        # 第二个 LayerNorm 层，标准化 embed_dim 维度的数据
        self.ln2 = nn.LayerNorm(embed_dim)
        # 线性投影层，将 embed_dim 维度的数据投影到维度为 1 的输出
        self.proj = nn.Linear(embed_dim, 1)

    # 前向传播方法，接受输入 hidden_states，并返回预测的 Tensor 结果
    def forward(self, hidden_states: Tensor) -> Tensor:
        # 将输入 hidden_states 进行维度转换后通过第一个卷积层
        hidden_states = self.conv1(hidden_states.transpose(1, 2))
        # 使用 ReLU 激活函数后再次进行维度转换
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        # 应用 LayerNorm 和 dropout
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        # 通过第二个卷积层
        hidden_states = self.conv2(hidden_states.transpose(1, 2))
        # 再次使用 ReLU 激活函数并进行维度转换
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        # 再次应用 LayerNorm 和 dropout
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        # 最终通过线性投影层并在维度2上挤压，得到最终的预测结果
        return self.proj(hidden_states).squeeze(dim=2)


# 定义一个名为 SeamlessM4THifiGan 的 PyTorch 模型类
class SeamlessM4THifiGan(nn.Module):
    def __init__(self, config: SeamlessM4TConfig):
        super().__init__()
        # 计算输入模型的总维度，包括单位嵌入维度、语言嵌入维度和说话人嵌入维度
        model_in_dim = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim
        # 从配置中获取泄漏的 ReLU 斜率
        self.leaky_relu_slope = config.leaky_relu_slope
        # 获取残差块的数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        # 获取上采样率的数量
        self.num_upsamples = len(config.upsample_rates)
        # 第一个卷积层，输入维度为 model_in_dim，输出维度为 config.upsample_initial_channel
        self.conv_pre = nn.Conv1d(
            model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # 上采样器模块列表
        self.upsampler = nn.ModuleList()
        # 遍历配置中的上采样率和卷积核大小，创建相应的转置卷积层
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

        # 残差块模块列表
        self.resblocks = nn.ModuleList()
        # 遍历创建残差块，根据配置中的残差块大小、扩张率和泄漏的 ReLU 斜率创建
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 第二个卷积层，输入维度为 channels，输出维度为 1
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)
    def forward(self, input_embeds: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                model_in_dim)`, or un-batched and of shape `(sequence_length, model_in_dim)`. Note that `model_in_dim`
                is the sum of `config.unit_embed_dim`, `config.lang_embed_dim` and `config.spkr_embed_dim`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """

        # Apply initial convolutional layers to the input log-mel spectrogram
        hidden_states = self.conv_pre(input_embeds)
        
        # Upsample the features using a series of upsample blocks
        for i in range(self.num_upsamples):
            # Apply leaky ReLU activation function to the hidden states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # Upsample the hidden states using the specified upsampler
            hidden_states = self.upsampler[i](hidden_states)

            # Apply residual blocks to the upsampled features
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                # Aggregate outputs of all residual blocks within the current layer
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            # Average the aggregated state across all kernels
            hidden_states = res_state / self.num_kernels

        # Apply leaky ReLU activation function to the final hidden states
        hidden_states = nn.functional.leaky_relu(hidden_states)
        # Apply post-convolutional layers to the processed features
        hidden_states = self.conv_post(hidden_states)
        # Apply hyperbolic tangent activation function to constrain values to range [-1, 1]
        hidden_states = torch.tanh(hidden_states)

        # Squeeze out the sequence-length dimension since it collapses to 1
        waveform = hidden_states.squeeze(1)

        return waveform
# 添加文档字符串，描述该类是实现了 HiFi-GAN 语音合成器，参考自 https://github.com/facebookresearch/speech-resynthesis
@add_start_docstrings(
    """Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).""",
    HIFIGAN_START_DOCSTRING,
)
# 定义 SeamlessM4TCodeHifiGan 类，继承自 PreTrainedModel
class SeamlessM4TCodeHifiGan(PreTrainedModel):
    # 引用配置类 SeamlessM4TConfig
    config_class = SeamlessM4TConfig
    # 主输入名称为 "input_embeds"
    main_input_name = "input_embeds"
    # 用于标记不分割的模块列表为空
    _no_split_modules = []

    # 类的初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置填充标记 ID
        self.pad_token_id = config.t2u_pad_token_id
        # 创建持续时间预测器对象
        self.dur_predictor = SeamlessM4TVariancePredictor(config)

        # 创建单位嵌入层对象，用于单位（例如音素）的嵌入表示
        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        # 创建说话者嵌入层对象，用于说话者的嵌入表示
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        # 创建语言嵌入层对象，用于语言的嵌入表示
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)

        # 创建 HiFi-GAN 模型对象
        self.hifi_gan = SeamlessM4THifiGan(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 计算持续时间预测层输出后的输出长度
    def _get_dur_output_lengths(self, input_ids, dur_out):
        """
        Computes the output length after the duration layer.
        """
        # 计算输入中非填充标记的单位长度
        unit_lengths = (input_ids != self.pad_token_id).sum(1)

        # 处理没有填充或填充过多的边界情况
        unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)

        # 计算累积持续时间输出
        cumulative_dur_out = torch.cumsum(dur_out, dim=1)
        # 根据单位长度索引获取相应的持续时间输出，并压缩维度
        unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()

        # 返回单位长度
        return unit_lengths
    # 计算 Hifigan 卷积层的输出长度
    def _get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the hifigan convolutional layers
        """

        # 计算一维卷积层输出长度的函数，参考自 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            return (
                torch.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode="floor") + 1
            )

        # 计算转置卷积层输出长度的函数
        def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

        # conv_pre
        # 计算 conv_pre 层的输出长度，使用 _conv_out_length 函数
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        # upsampler
        # 遍历 upsampler 配置，计算每层的输出长度，使用 _transpose_conv_out_length 函数
        for i, (upsample_rate, kernel_size) in enumerate(
            zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)
        ):
            input_lengths = _transpose_conv_out_length(
                input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2
            )

        # resblock
        # 遍历 resblock 的配置，计算每个卷积块的输出长度
        for i in range(len(self.config.upsample_rates)):
            for kernel_size, dilation in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                for dil in dilation:
                    input_lengths = _conv_out_length(
                        input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil
                    )

                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

        # conv_post
        # 计算 conv_post 层的输出长度，使用 _conv_out_length 函数
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        # 返回最终计算得到的输入长度
        return input_lengths

    def forward(
        self, input_ids: torch.LongTensor, spkr_id: torch.Tensor, lang_id: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            spkr_id (`int`, *optional*):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional*):
                The language id to use as target language for translation.
        """
        # 将输入的序列 token 索引转换为单位嵌入后，进行维度转置
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)
        # 根据说话者 ID 获取说话者嵌入后，进行维度转置
        spkr = self.speaker_embedding(spkr_id).transpose(1, 2)
        # 根据目标语言 ID 获取语言嵌入后，进行维度转置
        lang = self.language_embedding(lang_id).transpose(1, 2)

        # 使用持续时间预测器对隐藏状态进行预测，然后通过指数函数和修剪获取持续时间输出
        log_dur_pred = self.dur_predictor(hidden_states.transpose(1, 2))
        dur_out = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1)
        # B x C x T
        # 如果 batch_size 为 1，则直接重复隐藏状态
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            # 如果 batch_size 大于 1 并且处于训练模式，则发出警告，因为样本在前向传播过程中被交错处理
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            # 对每个样本进行交错处理并填充，使之具有相同长度的序列
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=-1).transpose(0, 1)
                for (hidden_state, duration) in zip(hidden_states, dur_out)
            ]

            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)

        # 重复说话者和语言嵌入，使其与隐藏状态具有相同的时间维度
        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        # 将语言嵌入、隐藏状态和说话者嵌入连接起来
        hidden_states = torch.cat([lang, hidden_states, spkr], dim=1)

        # 将连接后的向量传递给 Hifi-GAN 模型
        hidden_states = self.hifi_gan(hidden_states)

        # 获取单位持续时间的输出长度
        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        # 根据单位持续时间的输出长度计算 Hifi-GAN 的输出长度
        lengths = self._get_output_hifigan_lengths(unit_lengths)

        return hidden_states, lengths

    def _init_weights(self, module):
        """Initialize the weights."""
        # 初始化线性层、一维卷积和一维转置卷积层的权重
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 初始化嵌入层的权重
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    # 应用权重归一化到预处理卷积层
    def apply_weight_norm(self):
        nn.utils.weight_norm(self.hifi_gan.conv_pre)
        # 对上采样器中的每一层应用权重归一化
        for layer in self.hifi_gan.upsampler:
            nn.utils.weight_norm(layer)
        # 对残差块列表中的每个块应用权重归一化
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        # 应用权重归一化到后处理卷积层
        nn.utils.weight_norm(self.hifi_gan.conv_post)

    # 移除预处理卷积层的权重归一化
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        # 对上采样器中的每一层移除权重归一化
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        # 对残差块列表中的每个块移除权重归一化
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        # 移除后处理卷积层的权重归一化
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)
############ WHOLE MODEL related code ################

# 定义一个基于文本到文本转换的 SeamlessM4T 模型的类
@add_start_docstrings(
    "The text-to-text SeamlessM4T Model transformer which can be used for T2TT.",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TForTextToText(SeamlessM4TPreTrainedModel):
    # 加载时需要忽略的关键字列表
    _keys_to_ignore_on_load_missing = ["speech_encoder", "t2u_model", "vocoder"]
    # 主输入名称
    main_input_name = "input_ids"

    # 被绑定权重的关键字列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化方法，接收一个 SeamlessM4TConfig 类型的参数配置
    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        # 共享的词嵌入层，使用给定的词汇大小、隐藏大小和填充标记ID来初始化
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 文本编码器和文本解码器，均基于给定的配置和共享的词嵌入层来初始化
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)

        # 语言模型头部线性层，将隐藏状态映射到词汇表大小的输出
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回文本编码器
    def get_encoder(self):
        return self.text_encoder

    # 返回文本解码器
    def get_decoder(self):
        return self.text_decoder

    # 返回输出词嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出词嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 返回输入词嵌入
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置新的输入词嵌入，同时更新相关共享的词嵌入层
    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 绑定权重，如果配置允许的话，将文本编码器、文本解码器和语言模型头部绑定到共享的词嵌入层上
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 前向传播函数，接收多个输入和可选的参数，执行模型的前向计算
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
        pass  # 实际前向传播逻辑未在此处实现，而是留给具体的模型实现

    # 生成方法，接收多个生成相关的输入参数，执行生成过程
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
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值（past_key_values），则只保留decoder_input_ids的最后一个标记
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，准备用于生成的输入
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 对于每一层的过去键值，重新排序它们以匹配beam_idx指定的顺序
        for layer_past in past_key_values:
            # 缓存的跨注意力状态不需要重新排序 -> 它们始终保持不变
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        # 返回重新排序后的过去键值
        return reordered_past
@add_start_docstrings(
    "The speech-to-text SeamlessM4T Model transformer which can be used for S2TT.",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TForSpeechToText(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["text_decoder", "t2u_model", "vocoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        # Initialize shared embeddings for text decoding
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        
        # Initialize speech encoder specific to SeamlessM4T model
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        
        # Initialize text decoder specific to SeamlessM4T model, using shared embeddings
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        
        # Final layer for language modeling prediction
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.speech_encoder

    def get_decoder(self):
        return self.text_decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    def _tie_weights(self):
        # Tie weights if specified in the configuration
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

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
    ):
        """
        Forward pass of the SeamlessM4T model for speech-to-text tasks.
        """

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
    ):
        """
        Generate sequences based on input features and configuration.
        """

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Prepare inputs for sequence generation based on given parameters.
        """
    ):  # 定义一个静态方法，接收参数 past_key_values 和 beam_idx
        # 如果 past_key_values 不为 None，则截取 decoder_input_ids 的最后一个输入
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个包含以下键值对的字典
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,  # 返回 encoder 的输出
            "past_key_values": past_key_values,  # 返回过去的 key-value
            "decoder_input_ids": decoder_input_ids,  # 返回 decoder 的输入 ids
            "attention_mask": attention_mask,  # 返回注意力掩码
            "use_cache": use_cache,  # 返回是否使用缓存的标志
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()  # 初始化一个空元组来存储重新排序后的过去状态

        # 遍历 past_key_values 中的每一层的过去状态
        for layer_past in past_key_values:
            # 对于每个 past_state，使用 beam_idx 进行索引选择并添加到 reordered_past 中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )

        # 返回重新排序后的过去状态
        return reordered_past
# 添加文档字符串注释，描述这个类是用于文本到语音的转换的 SeamlessM4T 模型
@add_start_docstrings(
    "The text-to-speech SeamlessM4T Model transformer which can be used for T2ST.",
    SEAMLESS_M4T_START_DOCSTRING,
)
class SeamlessM4TForTextToSpeech(SeamlessM4TPreTrainedModel):
    # 在模型加载时忽略的键列表，这里忽略了 "speech_encoder"
    _keys_to_ignore_on_load_missing = ["speech_encoder"]
    # 主输入名称为 "input_ids"
    main_input_name = "input_ids"

    # 绑定权重的关键字列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化函数，接受一个 SeamlessM4TConfig 类型的配置参数
    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)

        # 创建一个共享的嵌入层，用于处理输入的文本
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 创建文本编码器和解码器实例
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        # 创建语言模型头部线性层，用于生成预测输出
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

        # 创建文本到单元模型和声码器模型实例
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    # 返回文本编码器实例
    def get_encoder(self):
        return self.text_encoder

    # 返回文本解码器实例
    def get_decoder(self):
        return self.text_decoder

    # 返回语言模型头部线性层实例
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 返回文本解码器的嵌入层
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置新的输入嵌入层
    def set_input_embeddings(self, value):
        # 同时更新文本编码器、文本解码器和共享的嵌入层
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 绑定权重函数，如果配置要求，将文本编码器、解码器和语言模型头部的权重绑定到共享嵌入层上
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 前向传播函数，接受多种输入参数用于文本到语音的转换任务
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
    ):
        # 在前向传播中不计算梯度
        @torch.no_grad()
    # 定义一个生成函数，用于生成文本或模型预测的输入
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        spkr_id: Optional[int] = 0,
        **kwargs,
    ):
        # 如果使用了过去的键值（past_key_values），则截取decoder_input_ids的最后一个token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，包含生成过程所需的所有输入
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    # 静态方法：重新排序缓存（past_key_values），用于beam搜索
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空元组，用于存储重新排序后的过去键值
        reordered_past = ()
        # 遍历每一层的过去键值
        for layer_past in past_key_values:
            # 对于每个层的前两个past_state进行按照beam索引重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        # 返回重新排序后的过去键值
        return reordered_past
# 用装饰器添加模型文档字符串，描述了该模型用于语音到语音转换任务的用途
@add_start_docstrings(
    "The speech-to-speech SeamlessM4T Model transformer which can be used for S2ST.",
    SEAMLESS_M4T_START_DOCSTRING,
)
# 声明 SeamlessM4TForSpeechToSpeech 类，继承自 SeamlessM4TPreTrainedModel 类
class SeamlessM4TForSpeechToSpeech(SeamlessM4TPreTrainedModel):
    # 在模型加载时忽略的键列表
    _keys_to_ignore_on_load_missing = ["text_encoder"]
    # 主输入名称
    main_input_name = "input_features"

    # 被绑定权重的键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 共享的嵌入层，使用 nn.Embedding 创建，包含词汇表大小、隐藏层大小和填充标记ID
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 语音编码器，使用 SeamlessM4TSpeechEncoder 类创建
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        # 文本解码器，使用 SeamlessM4TDecoder 类创建，传入共享的嵌入层
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        # 语言模型头部，使用 nn.Linear 创建，连接隐藏层到词汇表大小，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

        # 文本到单元的条件生成模型，使用 SeamlessM4TTextToUnitForConditionalGeneration 类创建
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        # 高保真生成模块，使用 SeamlessM4TCodeHifiGan 类创建
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    # 返回语音编码器对象
    def get_encoder(self):
        return self.speech_encoder

    # 返回文本解码器对象
    def get_decoder(self):
        return self.text_decoder

    # 返回语言模型头部对象
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 返回输入嵌入层对象
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置新的输入嵌入层
    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    # 如果配置允许，则绑定权重，共享词嵌入和语言模型头部的权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 使用装饰器添加模型前向方法的文档字符串，描述了输入参数和输出
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
    ):
        pass  # 实际实现被省略

    # 使用 torch.no_grad 装饰器，指定生成方法为无梯度运行
    @torch.no_grad()
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        spkr_id: Optional[int] = 0,
        **kwargs,
    ):
        pass  # 实际实现被省略

    # 静态方法声明省略
    # 重新排列缓存中的过去键值，以适应新的束搜索索引顺序
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 对于每一层的过去键值
        for layer_past in past_key_values:
            # 对于编码-解码注意力的缓存状态，无需重新排序，因为它们始终相同
            reordered_past += (
                # 对每个过去状态按照新的束搜索索引顺序进行选择
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # 准备生成模型输入的函数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值，截取decoder_input_ids的最后一个标记
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
@add_start_docstrings(
    "The original SeamlessM4T Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).",
    SEAMLESS_M4T_START_DOCSTRING,
    """
        current_modality (`str`, *optional*, defaults to `"text"`):
            Default modality. Used to initialize the model.
    """,
)
class SeamlessM4TModel(SeamlessM4TPreTrainedModel):
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config, current_modality="text"):
        super().__init__(config)

        # Initialize shared embeddings for both encoder and decoder
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # Initialize encoders and decoders based on the specified modality
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Set the default modality for the model
        self.current_modality = current_modality
        if current_modality == "speech":
            self.main_input_name = "input_features"

        # Initialize additional models for different modalities
        # These models have their own initialization procedures
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def set_modality(self, modality="text"):
        # Set the modality of the model (either "text" or "speech")
        if modality == "text":
            self.main_input_name = "input_ids"
            self.current_modality = "text"
        elif modality == "speech":
            self.main_input_name = "input_features"
            self.current_modality = "speech"
        else:
            raise ValueError(f"`modality={modality}` is not a valid modality. It must be `text` or `speech`.")

    def get_encoder(self):
        # Return the encoder based on the current modality
        if self.current_modality == "text":
            return self.text_encoder
        else:
            return self.speech_encoder

    def get_output_embeddings(self):
        # Return the output embeddings used for LM head
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # Set new embeddings for the LM head
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        # Return the input embeddings used in the decoder
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        # Set new input embeddings for the encoder and decoder
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        # Tie weights for word embeddings if specified in the config
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_MODEL_INPUTS_DOCSTRING)
    @torch.no_grad()
    # 声明这是一个不需要计算梯度的函数装饰器，使用了PyTorch的torch.no_grad()装饰器

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
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
    # 神经网络的前向传播函数，接受多种输入参数和选项

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        spkr_id: Optional[int] = 0,
        generate_speech: Optional[bool] = True,
        **kwargs,
    ):
    # 生成函数，生成模型的输出，可以控制生成过程的多个选项

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            # 如果使用了缓存的过去键值，则截取decoder_input_ids的最后一个标记
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            # input_ids不需要，因为encoder_outputs已经定义了
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
        # 为生成过程准备输入的函数，返回一个包含准备好的输入信息的字典

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            # 缓存的跨注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
        # 重新排序缓存的函数，以匹配指定的beam索引
```