# `.\models\seamless_m4t_v2\modeling_seamless_m4t_v2.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明，声明了版权信息和许可证信息
# 如果没有特定许可证要求，你不能使用这个文件
# 可以在以下链接获取 Apache License, Version 2.0 的详细信息：
# http://www.apache.org/licenses/LICENSE-2.0
#
# 根据适用法律的要求或书面同意，本软件是基于“原样”提供的，
# 没有任何形式的担保或条件，包括但不限于适销性保证、特定用途的适用性或非侵权性的条件。
# 查看许可证以获取具体的语言权限
""" PyTorch SeamlessM4Tv2 model."""

# 导入需要的库和模块
import copy  # 导入 copy 模块，用于对象的深拷贝操作
import math  # 导入 math 模块，提供标准数学函数库
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from typing import Optional, Tuple, Union  # 导入类型提示所需的类型定义

import torch  # 导入 PyTorch 深度学习库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块，用于实现内存优化的检查点
from torch import Tensor, nn  # 从 torch 模块导入 Tensor 类和 nn 模块
from torch.nn import CrossEntropyLoss  # 从 torch.nn 模块导入 CrossEntropyLoss 类

# 从其他模块导入所需的功能和类
from ...activations import ACT2FN  # 从 activations 模块导入 ACT2FN，激活函数的映射字典
from ...deepspeed import is_deepspeed_zero3_enabled  # 从 deepspeed 模块导入 is_deepspeed_zero3_enabled 函数，用于检查是否启用了 DeepSpeed Zero3
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 从 modeling_attn_mask_utils 模块导入相关的函数，用于准备注意力掩码
from ...modeling_outputs import (  # 从 modeling_outputs 模块导入多个输出类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 从 modeling_utils 模块导入 PreTrainedModel 类
from ...utils import (  # 从 utils 模块导入多个实用函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config  # 从当前目录下的 configuration_seamless_m4t_v2 模块导入 SeamlessM4Tv2Config 类

# 获取 logger 实例，用于记录日志信息
logger = logging.get_logger(__name__)

# 以下两个变量用于文档生成时的占位符，实际内容可以略过
_CHECKPOINT_FOR_DOC = ""
_CONFIG_FOR_DOC = "SeamlessM4Tv2Config"

# 预训练模型的存档列表，列出了预训练模型的名称
SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/seamless-m4t-v2-large",
    # 更多预训练模型可以在 https://huggingface.co/models?filter=seamless_m4t_v2 查看
]

# 预训练模型的配置映射，给出了 SpeechT5 模型的配置信息
SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_hifigan": "https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/config.json",
}

@dataclass
# 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TGenerationOutput 复制而来，仅将 SeamlessM4T 改为 SeamlessM4Tv2
class SeamlessM4Tv2GenerationOutput(ModelOutput):
    """
    定义从 `SeamlessM4Tv2Model`, `SeamlessM4Tv2ForTextToText`,
    `SeamlessM4Tv2ForTextToSpeech`, `SeamlessM4Tv2ForSpeechToSpeech` 和 `SeamlessM4Tv2ForTextToSpeech` 生成的输出类。
    """
    # 定`
    # 定义了四个可选参数，用于接收模型预测的音频波形和文本或单元序列的输出
    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            模型预测的最终音频波形。
        waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
            每个 `waveform` 批次中每个元素的样本长度，可选参数。
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            生成的文本或语音译文序列。这是文本到文本或语音到文本模型的输出。
            第二维 (sequence_length) 可能等于 `max_length`，或者如果所有批次因 `eos_token_id` 提前结束而更短。
        unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            生成的单元序列翻译。这是文本到单元模型的输出。
            第二维 (unit_sequence_length) 可能等于 `t2u_max_length`，或者如果所有批次因 `t2u_eos_token_id` 提前结束而更短。
    """
    
    waveform: Optional[torch.FloatTensor] = None
    waveform_lengths: Optional[torch.IntTensor] = None
    sequences: Optional[Tuple[torch.FloatTensor]] = None
    unit_sequences: Optional[Tuple[torch.FloatTensor]] = None
    
    
    **Q1:** How are the `waveform`, `sequences`, and `unit_sequences` typically processed after being returned from the model?  
    **Q2:** What are some scenarios where the `waveform_lengths` parameter might be crucial in practical applications?  
    **Q3:** Can you explain the significance of `max_length` and `t2u_max_length` in the context of sequence generation with these models?
# 使用 `dataclass` 装饰器定义一个数据类，用于存储 [`SeamlessM4Tv2TextToUnitDecoder`] 的输出
@dataclass
class SeamlessM4Tv2TextToUnitDecoderOutput(ModelOutput):
    """
    Class defining the outputs from [`SeamlessM4Tv2TextToUnitDecoder`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
            for *masked*
    """

    # 定义属性 `last_hidden_state`，类型为 `torch.FloatTensor`，存储最后一层模型的隐藏状态序列
    last_hidden_state: torch.FloatTensor = None
    # 定义属性 `hidden_states`，类型为 `Optional[Tuple[torch.FloatTensor]]`，存储每层模型的隐藏状态序列的元组（可选）
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义属性 `attentions`，类型为 `Optional[Tuple[torch.FloatTensor]]`，存储每层注意力权重的元组（可选）
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义属性 `padding_mask`，类型为 `Optional[torch.Tensor]`，指示哪些输入由于填充而应被忽略的张量（可选）
    padding_mask: Optional[torch.Tensor] = None


# 使用 `dataclass` 装饰器定义一个数据类，用于存储 [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] 和 [`SeamlessM4Tv2TextToUnitModel`] 的输出
@dataclass
class SeamlessM4Tv2TextToUnitOutput(ModelOutput):
    """
    Class defining the outputs from [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] and
    [`SeamlessM4Tv2TextToUnitModel`].
    """
    # 最后一个隐藏状态，是解码器模型最后一层的输出隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 填充遮罩，指示哪些输入由于填充而应被忽略，元素为1表示*未遮蔽*，为0表示*遮蔽*
    padding_mask: Optional[torch.Tensor] = None
    # 定义可选的解码器隐藏状态的元组，初始值为None，类型为torch.FloatTensor
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的解码器注意力权重的元组，初始值为None，类型为torch.FloatTensor
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的编码器最后隐藏状态，初始值为None，类型为torch.FloatTensor
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义可选的编码器隐藏状态的元组，初始值为None，类型为torch.FloatTensor
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的编码器注意力权重的元组，初始值为None，类型为torch.FloatTensor
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的损失值，初始值为None，类型为torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
# 定义一个模型文档字符串，说明这是一个 PyTorch 的子类模块，可以像常规的 PyTorch 模块一样使用，详细信息请参考 PyTorch 文档。
SEAMLESS_M4T_V2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4Tv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个文档字符串，描述多模态输入的说明，用于接收文本和音频特征输入。
SEAMLESS_M4T_V2_MULTIMODAL_INPUTS_DOCSTRING = r"""
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

# 定义一个文本输入的说明文档字符串，描述文本输入的参数。
M4T_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """

# 定义一个语音输入的说明文档字符串，描述语音特征输入的参数。
M4T_SPEECH_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returned by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
        """

# 定义一个文本到单位（units）输入的说明文档字符串，暂时为空白。
M4T_TEXT_TO_UNITS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入序列的token索引，在词汇表中进行编码
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        char_input_ids (`torch.LongTensor` of shape `(batch_size, char_sequence_length)`):
            # 字符索引，字符和索引的对应关系在生成配置的char_to_id字典中
            Character indices. The correspondence between characters and indices can be found in `char_to_id`, a
            dictionary in the generation configuration.
        char_count_per_id (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 每个输入id中字符的数量
                Number of characters per input id.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 注意力掩码，避免在填充token索引上执行注意力
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            # 编码器输出，包括最后一个隐藏层的隐藏状态
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*):
            # 如果不想传递`input_ids`，可以直接传递嵌入表示
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 用于计算MLM损失的标签，索引应在`[-100, 0, ..., config.vocab_size]`范围内
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            # 是否返回`~utils.ModelOutput`而不是普通的元组
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# 从输入的 `input_ids` 中创建位置编码，替换非填充符号为它们的位置编号。位置编号从 `padding_idx + 1` 开始，忽略填充符号。
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 创建一个掩码，标记非填充位置为1，其余为0
    mask = input_ids.ne(padding_idx).int()
    # 计算累积的位置索引，加上 `past_key_values_length` 并乘以掩码，以忽略填充部分
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 返回加上填充索引的位置编码
    return incremental_indices.long() + padding_idx


# 将输入的 `input_ids` 向右移动一个token
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 创建一个新的张量，与 `input_ids` 形状相同
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将 `input_ids` 向右移动一位，左边填充 `decoder_start_token_id`
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果 `pad_token_id` 为None，则引发值错误
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的 -100 值替换为 `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# 计算形如 `(batch, seq_len)` 的注意力掩码，对于每个批次中的元素，在对应的 `seq_lens` 处停止
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    # 获取隐藏状态的批次大小和序列长度
    batch_size, mask_seq_len = hidden_states.shape[:2]

    # 创建索引张量，形状为 `(mask_seq_len)`，并在设备上扩展至匹配 `seq_lens`
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    # 创建布尔掩码，指示每个批次中的每个位置是否应该被屏蔽
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    # 创建全1张量的掩码，并根据布尔掩码进行填充
    mask = hidden_states.new_ones((batch_size, mask_seq_len))
    mask = mask.masked_fill(bool_mask, 0)

    return mask


# 格式化适用于生成语音的 `SeamlessM4Tv2` 模型的参数
def format_speech_generation_kwargs(kwargs):
    # 将参数适配到文本生成或语音生成模型中
    # 此处还有一些未注释的代码，但由于示例要求每行都要注释，因此省略未注释的部分
    # 定义一个函数，接受关键字参数kwargs作为输入
    def attribute_kwargs(kwargs):
        # 初始化两个空字典，用于存储文本模型和语音模型的关键字参数
        kwargs_text = {}
        kwargs_speech = {}
        
        # 遍历kwargs字典中的每一对键值对
        for key, value in kwargs.items():
            # 如果键以"text_"开头
            if key.startswith("text_"):
                # 截取"text_"后面的部分作为新的键，并将对应的值存入kwargs_text字典
                key = key[len("text_") :]
                kwargs_text[key] = value
            # 如果键以"speech_"开头
            elif key.startswith("speech_"):
                # 截取"speech_"后面的部分作为新的键，并将对应的值存入kwargs_speech字典
                key = key[len("speech_") :]
                kwargs_speech[key] = value
            else:
                # 如果键不以"text_"或"speech_"开头，则根据键的值更新kwargs_text和kwargs_speech字典
                # 如果该键已经在某个模型的配置中设置了特定的值，则不进行覆盖
                if key not in kwargs_text:
                    kwargs_text[key] = value
                if key not in kwargs_speech:
                    kwargs_speech[key] = value
        
        # 返回两个字典，分别包含文本模型和语音模型的关键字参数
        return kwargs_text, kwargs_speech
############ SPEECH ENCODER related code ################

# 用于将输入特征进行归一化、投影和丢弃部分节点的模块
class SeamlessM4Tv2ConformerFeatureProjection(nn.Module):
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeatureProjection.__init__复制而来
    def __init__(self, config):
        super().__init__()
        # 归一化层，将输入特征进行均值和方差的归一化
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        # 线性投影层，将归一化后的特征投影到更高维度的隐藏层空间
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        # Dropout层，随机丢弃一部分神经元，用于减少过拟合
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states):
        # 对于量化，需要保留非投影的隐藏状态
        norm_hidden_states = self.layer_norm(hidden_states.to(self.layer_norm.weight.dtype))
        # 对归一化后的隐藏状态进行投影
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态进行dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeedForward复制而来，修改为SeamlessM4Tv2
class SeamlessM4Tv2ConformerFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, dropout=None):
        super().__init__()
        # 如果未指定dropout，则使用配置中的值
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        # 如果未指定激活函数，则使用配置中的值
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act

        # 中间层的dropout，用于在全连接层前丢弃部分神经元
        self.intermediate_dropout = nn.Dropout(dropout)
        # 中间层的全连接层，将隐藏状态映射到中间尺寸
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        # 中间层的激活函数，根据配置选择激活函数类型
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn

        # 输出层的全连接层，将中间尺寸的隐藏状态映射回原始隐藏层大小
        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        # 输出层的dropout，用于在输出层前丢弃部分神经元
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        # 中间层的全连接映射和激活函数处理
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        # 输出层的全连接映射和dropout处理
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SeamlessM4Tv2ConformerConvolutionModule(nn.Module):
    """在Conformer块中使用的卷积模块。使用类似于`https://doi.org/10.48550/arxiv.1609.03499`中描述的因果深度卷积。"""
    # 这里只是定义了一个类，并没有实现具体的初始化和前向传播方法，需要在其他地方继续完善
    # 初始化函数，接收一个配置对象 `config`
    def __init__(self, config):
        # 调用父类构造函数初始化
        super().__init__()
        # 检查深度可分离卷积核大小是否为奇数，否则抛出数值错误异常
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        # 初始化 Layer Normalization 层，对输入进行归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        # 初始化第一个逐点卷积层，将隐藏大小转换为两倍隐藏大小
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # 初始化 GLU 激活机制，沿第一个维度进行 Gated Linear Unit 操作
        self.glu = nn.GLU(dim=1)
        # 初始化深度可分离卷积层，使用给定的深度可分离卷积核大小、隐藏大小和组数
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        # 初始化深度可分离卷积层后的 Layer Normalization 层
        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size)
        # 根据配置中的激活函数名称选择相应的激活函数
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        # 初始化第二个逐点卷积层，将隐藏大小转换为给定的隐藏大小
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # 初始化 dropout 层，使用给定的丢弃率
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    # 前向传播函数，接收隐藏状态 `hidden_states` 和注意力掩码 `attention_mask`
    def forward(self, hidden_states, attention_mask=None):
        # 对隐藏状态进行 Layer Normalization
        hidden_states = self.layer_norm(hidden_states)

        # 如果有注意力掩码，则在需要的位置进行填充 0 操作
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # 交换时间维度和特征维度
        hidden_states = hidden_states.transpose(1, 2)

        # 执行 GLU 机制，将第一个逐点卷积层的输出通过 GLU 激活
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        # 由于因果卷积的需要，在序列的左侧进行填充
        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # 执行一维深度可分离卷积
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.activation(hidden_states)

        # 执行第二个逐点卷积层
        hidden_states = self.pointwise_conv2(hidden_states)
        # 应用 dropout 操作
        hidden_states = self.dropout(hidden_states)

        # 交换时间维度和特征维度，返回结果
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
# 定义一个 SeamlessM4Tv2ConformerSelfAttention 类，用于实现自注意力机制。
# 可以增强相对位置编码的功能。

def __init__(self, config, use_position_embeddings=True):
    # 调用父类构造函数初始化模块
    super().__init__()

    # 计算每个注意力头的大小
    self.head_size = config.hidden_size // config.speech_encoder_attention_heads
    # 设置注意力头的数量
    self.num_heads = config.speech_encoder_attention_heads
    # 如果使用位置编码，则设置位置编码的类型
    self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None

    # 初始化线性层，用于q、k、v的变换
    self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
    self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
    self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
    self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

    # 初始化dropout层
    self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

    # 如果使用相对键位置编码
    if self.position_embeddings_type == "relative_key":
        # 设置左侧和右侧最大位置编码
        self.left_max_position_embeddings = config.left_max_position_embeddings
        self.right_max_position_embeddings = config.right_max_position_embeddings
        # 计算位置编码的总数
        num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
        # 初始化位置编码的Embedding层
        self.distance_embedding = nn.Embedding(num_positions, self.head_size)

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 定义函数签名和返回类型，返回注意力输出和可能的注意力权重和注意力矩阵
        # self-attention 机制
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # 确保查询/键状态可以与值状态不同
        query_key_states = hidden_states
        value_states = hidden_states

        # 投影查询/键状态和值状态
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # 调整维度顺序使其符合注意力计算要求
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            # 如果使用相对位置编码
            query_length, key_length = query.shape[2], key.shape[2]

            # 构造相对位置信息
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

            # 获取位置嵌入并处理类型兼容性
            positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 兼容性

            # 计算相对位置注意力权重
            relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
            attn_weights = attn_weights + (relative_position_attn_weights / math.sqrt(self.head_size))

        # 如果有注意力掩码，应用掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 使用 softmax 归一化注意力权重
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # 应用 dropout
        attn_weights = self.dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value)

        # 调整维度顺序以便输出
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        # 应用最终的线性变换
        attn_output = self.linear_out(attn_output)

        # 如果不需要输出注意力权重，则设为 None
        if not output_attentions:
            attn_weights = None

        # 返回注意力输出和注意力权重
        return attn_output, attn_weights
class SeamlessM4Tv2ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ with Wav2Vec2->SeamlessM4Tv2, attention_dropout->speech_encoder_dropout, torch.nn->nn
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = SeamlessM4Tv2ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = SeamlessM4Tv2ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = SeamlessM4Tv2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)  # Layer normalization on input
        hidden_states = self.ffn1(hidden_states)  # Apply first feed-forward network
        hidden_states = hidden_states * 0.5 + residual  # Add residual connection

        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)  # Layer normalization on input
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )  # Apply self-attention mechanism
        hidden_states = self.self_attn_dropout(hidden_states)  # Apply dropout
        hidden_states = hidden_states + residual  # Add residual connection

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)  # Apply conformer convolution
        hidden_states = residual + hidden_states  # Add residual connection

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)  # Layer normalization on input
        hidden_states = self.ffn2(hidden_states)  # Apply second feed-forward network
        hidden_states = hidden_states * 0.5 + residual  # Add residual connection
        hidden_states = self.final_layer_norm(hidden_states)  # Layer normalization on output

        return hidden_states, attn_weights
    # 初始化方法，接受配置对象作为参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 将配置对象保存到实例变量中
        self.config = config

        # 定义 dropout 层，使用配置中的 dropout 率
        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        
        # 创建多层 Conformer 编码器，根据配置中的层数来创建
        self.layers = nn.ModuleList(
            [SeamlessM4Tv2ConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        # 定义层归一化层，使用配置中的隐藏层大小和层归一化 epsilon
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    # 实现分块注意力的方法
    def _apply_chunk_attention(self, attention_mask, hidden_states):
        """
        创建分块注意力掩码。它创建一个掩码以防止跨块的注意力，确保每个位置只注意其自身块内的位置。
        如果在配置中指定了左侧块重叠 (`speech_encoder_chunk_size`)，则相应调整注意力掩码，
        以允许每个位置还注意前 `speech_encoder_chunk_size - 1` 个块。
        """
        # 获取序列长度
        sequence_len = hidden_states.shape[1]

        # 根据序列长度生成块索引
        chunk_indices = torch.arange(sequence_len, device=hidden_states.device)
        chunk_indices = torch.div(chunk_indices, self.config.speech_encoder_chunk_size).long()

        # 计算每个块的起始索引
        start_indices = torch.full_like(chunk_indices, 0)
        if self.config.speech_encoder_left_chunk_num >= 0:
            start_indices = (chunk_indices - self.config.speech_encoder_left_chunk_num).clamp_(min=0)
            start_indices = start_indices * self.config.speech_encoder_chunk_size
        
        # 扩展起始索引以匹配序列长度
        start_indices = start_indices.unsqueeze(1).expand(-1, sequence_len)

        # 计算每个块的结束索引
        end_indices = ((chunk_indices + 1) * self.config.speech_encoder_chunk_size).clamp_(max=sequence_len)
        end_indices = end_indices.unsqueeze(1).expand(-1, sequence_len)

        # 创建索引张量
        indices = torch.arange(sequence_len, device=hidden_states.device).unsqueeze(0).expand(sequence_len, -1)

        # 创建块掩码，以防止跨块的注意力
        chunk_mask = (indices < start_indices) | (indices >= end_indices)
        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)

        # 如果存在注意力掩码，则将块掩码与之结合
        attention_mask = chunk_mask if attention_mask is None else (attention_mask.bool() | chunk_mask)
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        return attention_mask

    # 前向传播方法，接受隐藏状态、注意力掩码等参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    # 初始化空元组，根据模型设置决定是否保存隐藏状态和注意力分布
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
    
            # 复制注意力掩码以备后用
            conv_attention_mask = attention_mask
            if attention_mask is not None:
                # 确保填充的标记输出为0
                hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
                # 扩展注意力掩码以适应模型结构
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )
    
            # 如果模型配置中定义了语音编码器的分块大小，则应用分块注意力
            if self.config.speech_encoder_chunk_size is not None:
                attention_mask = self._apply_chunk_attention(attention_mask, hidden_states)
    
            # 如果存在注意力掩码，则使用浮点数的最小值进行扩展
            if attention_mask is not None:
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
    
            # 应用丢弃层，随机丢弃部分神经元
            hidden_states = self.dropout(hidden_states)
    
            # 检查是否启用了深速（deepspeed）的Zero3模式
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
    
            # 遍历每个层进行前向传播
            for i, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
    
                # 添加层丢弃（LayerDrop）功能，根据训练状态和配置决定是否跳过该层
                dropout_probability = torch.rand([])
                skip_the_layer = (
                    True if self.training and (dropout_probability < self.config.speech_encoder_layerdrop) else False
                )
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 如果启用梯度检查点且处于训练阶段，则使用梯度检查点函数加速计算
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                        )
                    else:
                        # 否则，直接调用层对象进行计算
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            conv_attention_mask=conv_attention_mask,
                        )
                    hidden_states = layer_outputs[0]
    
                if skip_the_layer:
                    layer_outputs = (None, None)
    
                # 如果需要输出注意力分布，则收集每层的自注意力分布
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
            # 应用层归一化（Layer Normalization）到最终隐藏状态
            hidden_states = self.layer_norm(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 根据返回格式要求，返回模型的输出
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapterLayer with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.adaptor_dropout

        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride

        # 1. residual convolution
        # 对输入进行残差层归一化
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        # 创建残差卷积层，输入维度为embed_dim，输出维度为2 * embed_dim
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 激活函数使用门控线性单元（GLU）
        self.activation = nn.GLU(dim=1)

        # Self-Attention
        # 对自注意力层进行归一化
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        # 创建自注意力卷积层，输入维度为embed_dim，输出维度为2 * embed_dim
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 创建自注意力机制对象，关闭位置嵌入
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config, use_position_embeddings=False)
        # 自注意力层使用dropout进行正则化
        self.self_attn_dropout = nn.Dropout(dropout)

        # Feed-forward
        # 对前馈网络层进行归一化
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        # 创建前馈网络对象，激活函数为ReLU，使用dropout进行正则化
        self.ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=dropout)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        # 计算从注意力掩码中得到的子采样长度
        pad = self.kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.floor()

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # Apply layer normalization to the hidden states using residual connections
        residual = self.residual_layer_norm(hidden_states)

        # Apply 1D convolution to the residual, changing its shape to (batch, feature_dim, seq_len)
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # Change the shape back to (batch, seq_len, feature_dim)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        # Apply layer normalization to the hidden states before self-attention
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Apply 1D convolution to the hidden states, changing its shape to (batch, feature_dim, seq_len)
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # Change the shape back to (batch, seq_len, feature_dim)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        # Compute attention mask adjustments if attention_mask is provided
        if attention_mask is not None:
            # Calculate sub-sampled lengths from the attention mask
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                hidden_states.device
            )
            # Compute new attention mask based on sub-sampled lengths
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            # Prepare the attention mask for 4D usage
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # Perform self-attention mechanism similar to a vanilla Transformer encoder layer
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual  # Add residual connection

        # Update residual to the current hidden states
        residual = hidden_states

        # Apply layer normalization to the updated hidden states
        hidden_states = self.ffn_layer_norm(hidden_states)
        # Apply feed-forward network to the normalized hidden states and add residual connection
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states
# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapter with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 使用 config 中的参数创建多层适配器层
        self.layers = nn.ModuleList(
            SeamlessM4Tv2ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers)
        )

    def forward(self, hidden_states, attention_mask):
        # down project hidden_states if necessary
        # 遍历每一层适配器层并应用到 hidden_states 上
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


############ TEXT / UNITS related code ################


# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding
class SeamlessM4Tv2SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # 调用 make_weights 方法初始化位置嵌入权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用 get_embedding 方法生成 sinusoidal embeddings，并根据条件设置权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 在 forward 方法中将权重转换为正确的数据类型和设备类型
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 将生成的权重注册为 buffer，不进行梯度计算
        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        # 计算 sinusoidal embeddings 的频率
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # 如果 embedding_dim 是奇数，则在最后一列添加零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            # 如果指定了 padding_idx，则将对应位置的 embeddings 设置为零向量
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
        ):
        # 如果输入的标识符不为空
        if input_ids is not None:
            # 获取输入标识符的批量大小和序列长度
            bsz, seq_len = input_ids.size()
            # 根据输入的标识符创建位置标识符。任何填充的标识符仍保持填充状态。
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            # 否则，获取输入嵌入的最后两个维度的大小，即批量大小和序列长度
            bsz, seq_len = inputs_embeds.size()[:-1]
            # 根据输入的嵌入和过去键值长度创建位置标识符
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # 如果需要扩展嵌入
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            # 根据需要扩展权重矩阵
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 返回根据位置标识符索引后的权重张量，形状重塑为[bsz, seq_len, embedding_dim]，并且返回的张量是不可变的
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状，不包括最后一个维度
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 创建顺序的位置标识符，从self.padding_idx + 1开始到sequence_length + self.padding_idx + 1结束
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置标识符增加一个维度，并扩展为与输入形状相同，然后确保连续性，并加上过去键值长度
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
class SeamlessM4Tv2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 从 transformers.models.bart.modeling_bart.BartAttention.__init__ 复制而来，将 Bart 替换为 SeamlessM4Tv2
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[SeamlessM4Tv2Config] = None,
    ):
        super().__init__()
        # 设置模块中的属性值
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 头的数量
        self.dropout = dropout  # dropout 比例
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.config = config  # 可选配置对象

        # 检查 embed_dim 是否能被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否使用因果注意力

        # 初始化四个线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # K 投影
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Q 投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出投影层

    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        # 重塑张量形状以适应多头注意力的计算
        new_projection_shape = projection.size()[:-1] + (self.num_heads, self.head_dim)
        # 将头放到第二个位置 (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 前向传播函数，接收输入张量并进行注意力计算
        pass  # 此处省略前向传播的具体实现

class SeamlessM4Tv2FeedForwardNetwork(nn.Module):
    def __init__(self, config: SeamlessM4Tv2Config, ffn_dim: int):
        super().__init__()
        # 初始化前馈神经网络的层
        self.fc1 = nn.Linear(config.hidden_size, ffn_dim)  # 第一个线性层
        self.fc2 = nn.Linear(ffn_dim, config.hidden_size)  # 第二个线性层
        self.dropout = nn.Dropout(config.activation_dropout)  # dropout 层
        self.act = ACT2FN[config.activation_function]  # 激活函数选择
    # 定义一个前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用全连接层 fc1 对隐藏状态进行线性变换
        hidden_states = self.fc1(hidden_states)
        # 对变换后的隐藏状态应用激活函数 act
        hidden_states = self.act(hidden_states)
        # 对激活后的隐藏状态应用 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 检查 fc2 的权重是否是 torch.Tensor 类型，并且确保隐藏状态的数据类型与 fc2 的权重不同，
        # 且 fc2 的权重数据类型不是 torch.int8 或 torch.uint8
        if (
            isinstance(self.fc2.weight, torch.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8)
        ):
            # 将隐藏状态转换为与 fc2 的权重相同的数据类型
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        # 使用全连接层 fc2 对处理后的隐藏状态再进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TEncoderLayer复制代码，并将SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2EncoderLayer(nn.Module):
    def __init__(self, config: SeamlessM4Tv2Config, encoder_ffn_dim=None, encoder_attention_heads=None):
        super().__init__()
        # 如果未提供encoder_ffn_dim，则使用config中的值
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        # 如果未提供encoder_attention_heads，则使用config中的值
        encoder_attention_heads = (
            config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        # 创建self attention层，使用SeamlessM4Tv2Attention
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # dropout层，用于self attention
        self.attn_dropout = nn.Dropout(config.dropout)
        # Layer normalization层，用于self attention输出
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建feedforward network层
        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)
        # Layer normalization层，用于feedforward network输出
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        # dropout层，用于feedforward network
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

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
        # Layer normalization层，用于self attention输入
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 执行self attention操作
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 应用dropout到self attention输出
        hidden_states = self.attn_dropout(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states

        # 保留残差连接
        residual = hidden_states

        # Layer normalization层，用于feedforward network输入
        hidden_states = self.ffn_layer_norm(hidden_states)

        # 执行feedforward network操作
        hidden_states = self.ffn(hidden_states)
        # 应用dropout到feedforward network输出
        hidden_states = self.ffn_dropout(hidden_states)

        # 残差连接
        hidden_states = residual + hidden_states

        # 返回结果
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出attention权重，则添加到输出中

        return outputs


# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoderLayer复制代码，并将SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2DecoderLayer(nn.Module):
    def __init__(self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None):
        super().__init__()
        # 如果未提供 decoder_ffn_dim 参数，则使用 config 中的值
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        # 如果未提供 decoder_attention_heads 参数，则使用 config 中的值
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )

        # 设置嵌入维度为 config 中的 hidden_size
        self.embed_dim = config.hidden_size
        # 初始化自注意力层，作为解码器的一部分
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 激活函数使用配置中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置注意力 dropout
        self.attn_dropout = nn.Dropout(config.dropout)

        # 初始化自注意力层后的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化交叉注意力层，作为解码器的一部分
        self.cross_attention = SeamlessM4Tv2Attention(
            self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        # 初始化交叉注意力层后的 LayerNorm
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化前馈神经网络
        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)

        # 初始化前馈神经网络后的 LayerNorm
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        # 设置前馈神经网络的 dropout
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
# 定义一个名为 SeamlessM4Tv2TextToUnitDecoderLayer 的自定义神经网络层，继承自 nn.Module
class SeamlessM4Tv2TextToUnitDecoderLayer(nn.Module):
    # 初始化函数，接收配置 config 对象和可选的解码器前馈维度和注意力头数参数
    def __init__(self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None):
        # 调用父类初始化函数
        super().__init__()
        # 如果 decoder_ffn_dim 为 None，则使用 config 中的 decoder_ffn_dim；否则使用传入的 decoder_ffn_dim
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        # 如果 decoder_attention_heads 为 None，则使用 config 中的 decoder_attention_heads；否则使用传入的 decoder_attention_heads
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )
        # 设置 dropout 属性为 config 中的 dropout
        self.dropout = config.dropout
        # 设置 embed_dim 属性为 config 中的 hidden_size
        self.embed_dim = config.hidden_size

        # 创建一个自注意力机制层，用于处理解码器的自注意力
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 创建自注意力层的 LayerNorm 归一化层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建第一个卷积层，输入和输出通道数都为 embed_dim，卷积核大小为 7，步长为 1，padding 方式为 "same"
        self.conv1 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=7, stride=1, padding="same")
        # 设置激活函数为 config 中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 创建第二个卷积层，输入和输出通道数为 embed_dim，卷积核大小为 7，步长为 1，padding 方式为 "same"
        self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=7, stride=1, padding="same")

        # 创建卷积层的 LayerNorm 归一化层
        self.conv_layer_norm = nn.LayerNorm(config.hidden_size)
        # 创建卷积层的 Dropout 层，使用初始化时指定的 dropout
        self.conv_dropout = nn.Dropout(self.dropout)

    # 定义前向传播函数，接收隐藏状态 hidden_states 和可选的注意力掩码参数，返回处理后的输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                输入到该层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                注意力掩码张量，形状为 `(batch, 1, tgt_len, src_len)`，其中填充元素由非常大的负值表示。
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                表示哪些输入由于填充而应被忽略，其中元素为 1 表示*未被掩码*，为 0 表示*被掩码*
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多细节。
        """
        residual = hidden_states

        # Self Attention
        # 使用 self_attn 层进行自注意力计算
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Conv
        residual = hidden_states

        # Apply padding mask to avoid leaking padded positions in the convolution layer
        # 应用填充掩码以避免在卷积层中泄露填充位置
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv1(hidden_states.transpose(1, 2)).transpose(1, 2)

        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.conv2(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = self.conv_dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.conv_layer_norm(hidden_states)

        outputs = (hidden_states, present_key_value)

        if output_attentions:
            outputs += self_attn_weights

        return outputs
# 定义了一个名为 SeamlessM4Tv2PreTrainedModel 的类，继承自 PreTrainedModel 抽象类，
# 用于处理权重初始化以及预训练模型下载和加载的简单接口。

class SeamlessM4Tv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 SeamlessM4Tv2Config
    config_class = SeamlessM4Tv2Config
    # 模型基础名称前缀
    base_model_prefix = "seamless_m4t_v2"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表
    _no_split_modules = [
        "SeamlessM4Tv2EncoderLayer",
        "SeamlessM4Tv2DecoderLayer",
        "SeamlessM4Tv2ConformerEncoderLayer",
        "SeamlessM4Tv2TextToUnitDecoderLayer",
    ]

    # 初始化权重函数
    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，将对应权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 Conformer 自注意力层
        elif isinstance(module, SeamlessM4Tv2ConformerSelfAttention):
            # 如果存在位置偏置，使用 Xavier 初始化
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        # 如果是 Conformer 特征投影层
        elif isinstance(module, SeamlessM4Tv2ConformerFeatureProjection):
            # 计算均匀分布的范围 k
            k = math.sqrt(1 / module.projection.in_features)
            # 初始化权重和偏置为均匀分布
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果是 LayerNorm 或者 GroupNorm 层
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 初始化偏置为零，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是 1 维卷积或者转置卷积层
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            # 使用 Kaiming 正态分布初始化权重
            nn.init.kaiming_normal_(module.weight)
            # 如果存在偏置，使用均匀分布初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # 从注意力掩码计算子采样长度的函数
    # 复制自 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TPreTrainedModel._compute_sub_sample_lengths_from_attention_mask
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        # 获取适配器的卷积核大小和步长
        kernel_size, stride = self.config.adaptor_kernel_size, self.config.adaptor_stride
        # 计算填充大小
        pad = kernel_size // 2
        # 计算序列长度，去除注意力掩码的填充部分
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        # 计算子采样长度
        seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

        return seq_lens.floor()
    # 根据输入的token id列表，返回每个token id对应的文本子词串
    def _indices_to_subwords(self, input_ids):
        """
        Returns the corresponding text string for each input id.
        返回每个输入id对应的文本字符串。
        """
        # 检查是否存在self.generation_config中的'id_to_text'键，该键将token id映射到子词。如果不存在，则抛出异常。
        if not hasattr(self.generation_config, "id_to_text"):
            raise ValueError(
                """This model generation config doesn't have a `id_to_text` key which maps
                token ids to subwords. Make sure to load the right generation config."""
            )
        
        # 获取输入张量的批大小和序列长度
        batch_size, sequence_len = input_ids.shape

        # 初始化一个空列表，用于存储每个批次的子词列表
        subwords_batch = []
        
        # 遍历每个批次中的样本
        for batch_id in range(batch_size):
            # 初始化一个空列表，用于存储当前样本的子词列表
            subwords = []
            
            # 遍历当前样本的序列长度
            for i in range(sequence_len):
                # 根据输入的token id从self.generation_config.id_to_text中获取对应的子词串
                subword = self.generation_config.id_to_text.get(str(input_ids[batch_id, i].item()))
                # 将获取的子词串转换为字符串并添加到当前样本的子词列表中
                subwords.append(str(subword))
            
            # 将当前样本的子词列表添加到批次子词列表中
            subwords_batch.append(subwords)
        
        # 返回包含每个输入id对应的文本子词串的批次列表
        return subwords_batch

    # 计算子词列表中各子词的字符长度
    def _count_character_length_in_subword(
        self,
        input_ids,
        subwords_batch,
        merge_space_with_prev_subword=False,
        pad_token_id=0,
        unk_token_id=1,
        space="▁",
    ):
    def _get_char_input_ids(self, input_ids, subwords_batch, char_count_per_id, pad_token_id=0, unk_token_id=1):
        """
        Returns the corresponding character input id for each character of `subwords_batch`.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            char_count_per_id (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Number of characters per input id.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
        Returns:
            `torch.Tensor`: Tensor of shape `(batch_size, char_sequence_length)` containing the id of each character.
        """
        # 检查生成配置中是否存在 `char_to_id` 键，该键将字符映射到字符 id
        if not hasattr(self.generation_config, "char_to_id"):
            raise ValueError(
                """This model generation config doesn't have a `char_to_id` key which maps
                characters to character ids. Make sure to load the right generation config."""
            )

        # 获取输入的 batch 大小
        batch_size = input_ids.shape[0]
        # 计算每个子词的最大字符长度，并将其转换为整数
        max_len = int(char_count_per_id.sum(1).max().item())

        # 创建一个新的零填充张量来存储字符序列，形状为 `(batch_size, max_len)`
        char_seqs = input_ids.new_zeros((batch_size, max_len)).fill_(pad_token_id)

        # 计算每个样本中非填充字符的子词长度
        subword_lens = input_ids.ne(pad_token_id).sum(1)

        # 遍历每个样本
        for batch_id in range(batch_size):
            total = 0
            # 获取当前样本中的非填充子词索引和对应的子词字符串
            subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][: subword_lens[batch_id]]
            # 遍历每个子词索引和对应的子词字符串
            for subword_idx, subword in zip(subword_indices, subwords):
                if subword_idx == unk_token_id:
                    # 如果子词索引为未知 token 的索引，则将字符 id 设置为未知 token 的 id
                    char_ids = [unk_token_id]
                else:
                    # 否则，根据生成配置中的 `char_to_id` 映射获取每个字符的字符 id
                    char_ids = [self.generation_config.char_to_id.get(ch, unk_token_id) for ch in list(subword)]
                # 计算当前子词的字符长度
                char_seq_len = len(char_ids)
                # 将字符 id 转换为张量，并将其复制到 `char_seqs` 中的适当位置
                char_seqs[batch_id, total : total + char_seq_len] = torch.tensor(char_ids).to(char_seqs)
                total += char_seq_len
        # 返回字符 id 张量
        return char_seqs
    def _hard_upsample(self, hidden_states, durations):
        """
        Repeats the time dimension of each sample in the batch based on the corresponding duration.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, *)`, *optional*):
                The sequence to repeat, where `*` is any number of sequence-specific dimensions including none.
            durations (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates how many times to repeat time segments.
        """
        # 如果只有一个样本，使用 `torch.repeat_interleave` 来重复时间维度
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, durations.view(-1), dim=1)
        else:
            # 如果是批量样本，且处于训练模式，警告可能因为样本交错而导致并行性下降
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning_once(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            # 对于每个样本，根据对应的 duration 重复隐藏状态，形成列表
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=0)
                for (hidden_state, duration) in zip(hidden_states, durations)
            ]
            # 对重复后的序列进行填充，确保批量操作时序列长度一致
            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)

        # 返回处理后的隐藏状态
        return hidden_states
@add_start_docstrings(
    """Transformer speech encoder consisting of *config.speech_encoder_layers* conformer self attention layers.
    Each layer is a [`SeamlessM4Tv2ConformerEncoderLayer`].""",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
# 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TSpeechEncoder 复制过来，将 SeamlessM4T 替换为 SeamlessM4Tv2
class SeamlessM4Tv2SpeechEncoder(SeamlessM4Tv2PreTrainedModel):
    main_input_name = "input_features"

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)

        # 特征投影层，将输入特征投影到指定维度
        self.feature_projection = SeamlessM4Tv2ConformerFeatureProjection(config)
        # Conformer 编码器层
        self.encoder = SeamlessM4Tv2ConformerEncoder(config)
        # 中间的前馈神经网络层
        self.intermediate_ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=0.0)
        # 适配器层，可选添加
        self.adapter = SeamlessM4Tv2ConformerAdapter(config) if config.add_adapter else None
        # 内部层归一化
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4Tv2SpeechEncoder.forward`.
                Make sure one of them is not `None`."""
            )

        # 使用特征投影层处理输入特征
        hidden_states = self.feature_projection(input_features)

        # 调用编码器层处理特征向量序列
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的隐藏状态
        hidden_states = encoder_outputs[0]

        # 使用中间的前馈神经网络层处理隐藏状态
        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

        # 如果适配器存在，则应用适配器层处理隐藏状态
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        # 应用内部层归一化到隐藏状态
        hidden_states = self.inner_layer_norm(hidden_states)

        # 如果不使用 return_dict，则返回元组形式的输出
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        # 使用 Wav2Vec2BaseModelOutput 格式返回结果，包括最终隐藏状态、隐藏状态列表和注意力权重列表
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 添加了文档字符串，描述了该函数作为一个Transformer编码器的初始化函数，详细描述了各个参数的含义和用法
@add_start_docstrings(
    "Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`SeamlessM4Tv2EncoderLayer`].",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
        is_t2u_encoder (`bool`, *optional*, defaults to `False`):
            indicates if it belongs to the text-to-units model, in which case it won't have input embeddings
    """,
)
# 在原来的SeamlessM4T模型的基础上创建了SeamlessM4Tv2Encoder，继承自SeamlessM4Tv2PreTrainedModel
class SeamlessM4Tv2Encoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
        is_t2u_encoder: bool = False,
    ):
        super().__init__(config)

        # 从配置中获取参数并初始化
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_dim = config.hidden_size

        self.is_t2u_encoder = is_t2u_encoder
        self.max_source_positions = config.max_position_embeddings

        # 如果不是文本到单元（text-to-units）编码器，初始化嵌入
        if not self.is_t2u_encoder:
            # 根据配置选择是否对嵌入进行缩放
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            # 初始化嵌入层
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

            # 如果提供了额外的嵌入，使用提供的权重
            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight

            # 初始化位置嵌入
            self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
            )

        # 初始化编码器层列表
        layers = []
        for _ in range(config.encoder_layers):
            layers.append(
                SeamlessM4Tv2EncoderLayer(
                    config,
                    encoder_attention_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                )
            )

        self.layers = nn.ModuleList(layers)

        # 初始化层归一化层
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 是否开启梯度检查点
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数并返回相关输出
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
):
# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoder复制而来，将SeamlessM4T改为SeamlessM4Tv2
class SeamlessM4Tv2Decoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        super().__init__(config)
        self.dropout = config.dropout  # 设置dropout比率
        self.layerdrop = config.decoder_layerdrop  # 设置层级dropout比率
        self.padding_idx = config.pad_token_id  # 设置填充索引
        self.vocab_size = config.vocab_size  # 设置词汇表大小
        self.max_target_positions = config.max_position_embeddings  # 设置最大目标位置
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0  # 设置嵌入缩放因子

        if embed_tokens is not None:
            # 如果定义了embed_tokens，则使用其形状
            self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight  # 使用给定的权重初始化嵌入层权重
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)  # 创建新的嵌入层

        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )  # 初始化位置嵌入层

        layers = []
        for _ in range(config.decoder_layers):
            layers.append(
                SeamlessM4Tv2DecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )  # 创建decoder层，并添加到layers列表中
        self.layers = nn.ModuleList(layers)  # 转换为模块列表
        self.layer_norm = nn.LayerNorm(config.hidden_size)  # 应用层归一化

        self.gradient_checkpointing = False  # 设置梯度检查点为False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回输入嵌入层

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入嵌入层为给定值

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
    ):
        """
        Transformer解码器，包含config.decoder_layers层。每层是一个`SeamlessM4Tv2DecoderLayer`。

        Args:
            input_ids (torch.LongTensor, optional): 输入的token ids
            attention_mask (torch.Tensor, optional): 注意力遮罩
            encoder_hidden_states (torch.FloatTensor, optional): 编码器的隐藏状态
            encoder_attention_mask (torch.LongTensor, optional): 编码器的注意力遮罩
            past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]], optional): 缓存的键值对
            inputs_embeds (torch.FloatTensor, optional): 输入的嵌入
            use_cache (bool, optional): 是否使用缓存
            output_attentions (bool, optional): 是否输出注意力权重
            output_hidden_states (bool, optional): 是否输出隐藏状态
            return_dict (bool, optional): 是否返回字典

        Returns:
            Union[dict, Tuple]: 根据return_dict的设置返回字典或元组
        """
        pass  # 前向传播暂未实现

@add_start_docstrings(
    "Transformer解码器，包含*config.decoder_layers*层。每层是一个[`SeamlessM4Tv2DecoderLayer`]。",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional*):
            输入嵌入
    """,
)
class SeamlessM4Tv2TextToUnitDecoder(SeamlessM4Tv2PreTrainedModel):
    pass  # 未实现
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        # 调用父类构造函数，初始化模型配置
        super().__init__(config)
        # 从配置中获取并设置模型参数
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        # 根据配置决定是否对嵌入进行缩放
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # 如果传入了外部的嵌入层，使用其形状和权重初始化自身的嵌入层
            self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight
        else:
            # 否则，使用配置中的词汇表大小和隐藏层大小初始化嵌入层
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        # 初始化字符级别的嵌入层和位置编码嵌入
        self.embed_char = nn.Embedding(config.char_vocab_size, config.hidden_size)
        self.embed_char_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        # 初始化位置编码的权重参数
        self.pos_emb_alpha_char = nn.Parameter(torch.ones(1))
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        # 初始化持续时间预测器模型
        self.duration_predictor = SeamlessM4Tv2VariancePredictor(
            config.variance_predictor_embed_dim,
            config.variance_predictor_hidden_dim,
            config.variance_predictor_kernel_size,
            config.variance_pred_dropout,
        )

        # 初始化位置编码嵌入层
        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        # 根据配置中的解码器层数，循环添加解码器层到模型中
        layers = []
        for _ in range(config.decoder_layers):
            layers.append(
                SeamlessM4Tv2TextToUnitDecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )
        self.layers = nn.ModuleList(layers)
        # 初始化层归一化层
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 关闭梯度检查点功能
        self.gradient_checkpointing = False
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入层对象
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的权重
        self.embed_tokens = value

    def forward(
        self,
        char_input_ids: torch.LongTensor = None,
        char_count_per_id: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 前向传播函数的参数，包括字符输入 ID、每个字符 ID 的计数、编码器隐藏状态、是否输出注意力、是否输出隐藏状态、是否返回字典形式结果
# 添加函数说明文档的装饰器，用于该类的文档注释
@add_start_docstrings(
    "Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4Tv2Encoder`] without embeddings and the decoder is a [`SeamlessM4Tv2TextToUnitDecoder`].",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
# 定义 SeamlessM4Tv2TextToUnitModel 类，继承自 SeamlessM4Tv2PreTrainedModel
class SeamlessM4Tv2TextToUnitModel(SeamlessM4Tv2PreTrainedModel):
    
    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitModel.__init__ 复制而来，修改了类名和部分注释
    def __init__(
        self,
        config: SeamlessM4Tv2Config,  # 接收一个 SeamlessM4Tv2Config 类型的配置参数
        embed_tokens_decoder: Optional[nn.Embedding] = None,  # 可选的解码器嵌入层
    ):
        # 调用父类的构造函数，初始化模型配置
        super().__init__(config)

        # 初始化编码器，使用 SeamlessM4Tv2Encoder 类，并指定为文本到单元的编码器
        self.encoder = SeamlessM4Tv2Encoder(config, is_t2u_encoder=True)
        
        # 初始化解码器，使用 SeamlessM4Tv2TextToUnitDecoder 类，并传入解码器嵌入层
        self.decoder = SeamlessM4Tv2TextToUnitDecoder(config, embed_tokens_decoder)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法定义，接收多个输入参数和选项，返回预测输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，可选
        char_input_ids: torch.LongTensor = None,  # 字符级输入的 token IDs
        char_count_per_id: torch.LongTensor = None,  # 每个 token 的字符数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器的输出，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选
        # 注意：这里截断了原始代码，未包含完整的 forward 方法内容
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        # 如果指定了 output_attentions 参数，则使用指定的值，否则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果指定了 output_hidden_states 参数，则使用指定的值，否则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果指定了 return_dict 参数，则使用指定的值，否则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 encoder_outputs 为 None，则调用 self.encoder 进行编码
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # 如果 return_dict=True 并且 encoder_outputs 不是 BaseModelOutput 类型的实例，则封装成 BaseModelOutput
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # 调用 self.decoder 进行解码，生成 decoder_outputs
        decoder_outputs = self.decoder(
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            encoder_hidden_states=encoder_outputs[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果 return_dict=False，则返回 decoder_outputs 和 encoder_outputs 的元组
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 如果 return_dict=True，则将 decoder_outputs 和 encoder_outputs 封装成 SeamlessM4Tv2TextToUnitOutput 类型返回
        return SeamlessM4Tv2TextToUnitOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            padding_mask=decoder_outputs.padding_mask,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
# 将注释添加到类定义之前，描述了该类的作用和基础模型
@add_start_docstrings(
    "Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4Tv2TextToUnitModel`].",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
class SeamlessM4Tv2TextToUnitForConditionalGeneration(SeamlessM4Tv2PreTrainedModel):
    # 被忽略的键列表，用于在加载时处理缺失情况
    _keys_to_ignore_on_load_missing = [
        "vocoder",
        "speech_encoder",
        "text_encoder",
        "text_decoder",
    ]
    # 被绑定权重的键列表，这些权重在模型训练中会被共享
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    # 从SeamlessM4TTextToUnitForConditionalGeneration.__init__方法中复制而来，主要目的是初始化模型参数
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        # 深拷贝配置，主要用于处理诸如bos_token_id等特殊参数
        config = copy.deepcopy(config)
        for param, val in config.to_dict().items():
            if param.startswith("t2u_"):
                config.__setattr__(param[4:], val)
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)

        # 创建基础的SeamlessM4Tv2TextToUnitModel模型
        self.model = SeamlessM4Tv2TextToUnitModel(config, embed_tokens_decoder)

        # 创建语言模型头部，一个线性层将隐藏状态映射到t2u_vocab_size大小的输出空间，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.t2u_vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 从SeamlessM4TTextToUnitForConditionalGeneration.get_encoder方法中复制而来，返回模型的编码器部分
    def get_encoder(self):
        return self.model.encoder

    # 从SeamlessM4TTextToUnitForConditionalGeneration.get_decoder方法中复制而来，返回模型的解码器部分
    def get_decoder(self):
        return self.model.decoder

    # 从SeamlessM4TTextToUnitForConditionalGeneration.get_output_embeddings方法中复制而来，返回模型的语言模型头部
    def get_output_embeddings(self):
        return self.lm_head

    # 从SeamlessM4TTextToUnitForConditionalGeneration.set_output_embeddings方法中复制而来，设置模型的语言模型头部
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从SeamlessM4TTextToUnitForConditionalGeneration.get_input_embeddings方法中复制而来，返回模型的输入嵌入层
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 从SeamlessM4TTextToUnitForConditionalGeneration.set_input_embeddings方法中复制而来，设置模型的输入嵌入层
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 从add_start_docstrings_to_model_forward函数中添加了注释，用于详细描述模型前向传播时的输入参数和文档字符串
    @add_start_docstrings_to_model_forward(M4T_TEXT_TO_UNITS_INPUTS_DOCSTRING)
    # 定义 forward 方法，用于模型的前向传播
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs，数据类型为 LongTensor
        char_input_ids: torch.LongTensor = None,  # 字符级别的 token IDs，数据类型为 LongTensor
        char_count_per_id: torch.LongTensor = None,  # 每个 token 的字符数，数据类型为 LongTensor
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选的 Tensor 类型
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器的输出，可选的元组类型
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，可选的 FloatTensor 类型
        labels: Optional[torch.LongTensor] = None,  # 标签，可选的 LongTensor 类型
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔类型
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔类型
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选的布尔类型
        **kwargs,  # 其他参数
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        # 确定是否使用配置中指定的返回字典格式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型进行前向传播，传入各种参数
        outputs = self.model(
            input_ids,
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取语言模型的 logits
        lm_logits = self.lm_head(outputs[0])

        # 计算掩码语言建模的损失
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不使用字典格式返回结果，则以元组的形式返回输出
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 使用自定义的输出类封装结果，以字典格式返回
        return SeamlessM4Tv2TextToUnitOutput(
            last_hidden_state=lm_logits,
            padding_mask=outputs.padding_mask,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            loss=masked_lm_loss,
        )

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration._tie_weights 复制过来的方法
    # 用于将输出层权重绑定到输入层权重或克隆输入层权重
    def _tie_weights(self) -> None:
        # 检查配置中是否指定绑定词嵌入层
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            # 如果有输出嵌入层，则执行权重绑定操作
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
        config ([`SeamlessM4Tv2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个残差块类，继承自 nn.Module
# 该类的实例化对象可以用作 PyTorch 模型的一部分，详细使用方法请参考 PyTorch 文档
# 参数包括 channels（通道数）、kernel_size（卷积核大小，默认为3）、dilation（膨胀率，是一个元组，默认为(1, 3, 5)）、leaky_relu_slope（泄漏整流的斜率，默认为0.1）
class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # 初始化第一个卷积层的列表，每个卷积层使用不同的膨胀率
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

        # 初始化第二个卷积层的列表，每个卷积层的膨胀率都为1（无膨胀）
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

    # 计算给定卷积核大小和膨胀率的填充大小
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 对所有卷积层应用权重归一化
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除所有卷积层的权重归一化
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)
    # 定义前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 遍历两个卷积层的元组，每次迭代使用一对卷积层
        for conv1, conv2 in zip(self.convs1, self.convs2):
            # 保留残差连接之前的隐藏状态，用于后续加法操作
            residual = hidden_states
            # 应用带泄漏整流线性单元（Leaky ReLU）激活函数到隐藏状态
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 第一层卷积操作
            hidden_states = conv1(hidden_states)
            # 再次应用带泄漏整流线性单元（Leaky ReLU）激活函数到隐藏状态
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 第二层卷积操作
            hidden_states = conv2(hidden_states)
            # 将当前隐藏状态与之前保留的残差连接进行加法操作，实现残差连接
            hidden_states = hidden_states + residual
        # 返回经过所有卷积层和残差连接后的隐藏状态
        return hidden_states
# 定义名为 SeamlessM4Tv2VariancePredictor 的神经网络模型类，继承自 nn.Module
class SeamlessM4Tv2VariancePredictor(nn.Module):
    # 初始化函数，定义模型的结构和参数
    def __init__(self, embed_dim, hidden_dim, kernel_size, var_pred_dropout):
        super().__init__()

        # 第一个一维卷积层，输入维度为 embed_dim，输出维度为 hidden_dim
        # 使用指定的卷积核大小 kernel_size 和填充方式 "same"
        self.conv1 = nn.Conv1d(
            embed_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding="same",
        )

        # 激活函数ReLU
        self.activation_fuction = nn.ReLU()

        # 第一个 LayerNorm 层，归一化 hidden_dim 维度的输入
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Dropout 模块，以概率 var_pred_dropout 随机置零输入张量的元素
        self.dropout_module = nn.Dropout(p=var_pred_dropout)

        # 第二个一维卷积层，输入维度和输出维度均为 hidden_dim
        # 使用指定的卷积核大小 kernel_size 和填充方式 "same"
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding="same",
        )

        # 第二个 LayerNorm 层，归一化 hidden_dim 维度的输入
        self.ln2 = nn.LayerNorm(hidden_dim)

        # 线性变换层，将 hidden_dim 维度映射到 1 维度
        self.proj = nn.Linear(hidden_dim, 1)

    # 前向传播函数，定义数据在模型中的流动过程
    def forward(self, hidden_states: Tensor, padding_mask: Tensor = None) -> Tensor:
        # Input: B x T x C; Output: B x T
        # 如果 padding_mask 不为空，则对 hidden_states 进行 mask 操作，将未被掩盖的位置置零
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

        # 将 hidden_states 沿着第二个维度进行转置，然后通过第一个卷积层 conv1
        hidden_states = self.conv1(hidden_states.transpose(1, 2))

        # 经过激活函数ReLU，并再次沿第二个维度进行转置
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)

        # 通过 LayerNorm 层 ln1，并应用 Dropout 模块
        hidden_states = self.dropout_module(self.ln1(hidden_states))

        # 如果 padding_mask 不为空，则再次对 hidden_states 进行 mask 操作
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

        # 经过第二个卷积层 conv2
        hidden_states = self.conv2(hidden_states.transpose(1, 2))

        # 经过激活函数ReLU，并再次沿第二个维度进行转置
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)

        # 通过 LayerNorm 层 ln2，并应用 Dropout 模块
        hidden_states = self.dropout_module(self.ln2(hidden_states))

        # 最终通过线性变换层 proj，将 hidden_states 映射到最终输出的张量
        return self.proj(hidden_states).squeeze(dim=2)


# 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4THifiGan 复制而来，修改为 SeamlessM4Tv2HifiGan
class SeamlessM4Tv2HifiGan(nn.Module):
    # 初始化函数，接受一个 SeamlessM4Tv2Config 类型的参数 config
    def __init__(self, config: SeamlessM4Tv2Config):
        # 调用父类的初始化方法
        super().__init__()
        # 计算输入模型的维度，由配置对象 config 中的三个维度相加得到
        model_in_dim = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim
        # 设置 Leaky ReLU 激活函数的斜率
        self.leaky_relu_slope = config.leaky_relu_slope
        # 计算 Residual Block 的数量，由配置对象 config 中的 resblock_kernel_sizes 列表的长度确定
        self.num_kernels = len(config.resblock_kernel_sizes)
        # 计算 Upsample 的数量，由配置对象 config 中的 upsample_rates 列表的长度确定
        self.num_upsamples = len(config.upsample_rates)
        
        # 创建一个一维卷积层，用于预处理输入数据
        self.conv_pre = nn.Conv1d(
            model_in_dim,                                     # 输入通道数，等于 model_in_dim
            config.upsample_initial_channel,                   # 输出通道数，由配置对象 config 决定
            kernel_size=7,                                     # 卷积核大小
            stride=1,                                          # 步长
            padding=3,                                         # 填充大小
        )

        # 创建一个空的模块列表，用于存储 Upsample 层
        self.upsampler = nn.ModuleList()
        # 遍历配置对象 config 中的 upsample_rates 和 upsample_kernel_sizes 列表
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            # 向 upsampler 中添加一个一维转置卷积层
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),    # 输入通道数，随着循环次数递减
                    config.upsample_initial_channel // (2 ** (i + 1)),  # 输出通道数，递减
                    kernel_size=kernel_size,                     # 卷积核大小
                    stride=upsample_rate,                        # 步长
                    padding=(kernel_size - upsample_rate) // 2,  # 填充大小
                )
            )

        # 创建一个空的模块列表，用于存储 Residual Block 层
        self.resblocks = nn.ModuleList()
        # 遍历 upsample 层数量的次数
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))  # 计算通道数
            # 遍历配置对象 config 中的 resblock_kernel_sizes 和 resblock_dilation_sizes 列表
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                # 向 resblocks 中添加一个 HifiGanResidualBlock 残差块
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 创建一个一维卷积层，用于后处理输出数据
        self.conv_post = nn.Conv1d(
            channels,                                           # 输入通道数，等于最后一个循环中计算的 channels
            1,                                                  # 输出通道数，固定为 1
            kernel_size=7,                                      # 卷积核大小
            stride=1,                                           # 步长
            padding=3,                                          # 填充大小
        )
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

        # Apply convolutional layers to the input spectrogram
        hidden_states = self.conv_pre(input_embeds)

        # Upsample the feature maps using a series of upsampling layers
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            # Apply residual blocks to enhance feature representation
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        # Apply activation function and final convolutional layer
        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        # Squeeze out the sequence-length dimension to obtain the waveform
        waveform = hidden_states.squeeze(1)

        return waveform
# 在此类中定义了一个特定的HiFi-GAN vocoder，该类继承自PreTrainedModel，用于语音再合成。
@add_start_docstrings(
    """Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).""",
    HIFIGAN_START_DOCSTRING,
)
class SeamlessM4Tv2CodeHifiGan(PreTrainedModel):
    # 使用SeamlessM4Tv2Config作为配置类
    config_class = SeamlessM4Tv2Config
    # 主输入的名称为"input_embeds"
    main_input_name = "input_embeds"
    # 不需要拆分的模块为空列表
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)

        # 设置填充标记的ID，从配置中获取
        self.pad_token_id = config.t2u_pad_token_id
        # 单位嵌入的维度，从配置中获取
        embed_dim = config.unit_embed_dim
        # 方差预测器的卷积核大小，从配置中获取
        kernel_size = config.variance_predictor_kernel_size
        # 方差预测器的dropout率，从配置中获取
        var_pred_dropout = config.var_pred_dropout
        # 创建方差预测器对象
        self.dur_predictor = SeamlessM4Tv2VariancePredictor(embed_dim, embed_dim, kernel_size, var_pred_dropout)

        # 单位嵌入层，使用单位HiFi-GAN词汇表的大小和嵌入维度来初始化
        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        # 说话人嵌入层，使用声码器的说话人数量和嵌入维度来初始化
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        # 语言嵌入层，使用声码器的语言数量和嵌入维度来初始化
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)

        # 初始化HiFi-GAN模型，使用给定的配置对象
        self.hifi_gan = SeamlessM4Tv2HifiGan(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan中复制而来，计算持续时间预测输出后的长度
    def _get_dur_output_lengths(self, input_ids, dur_out):
        """
        Computes the output length after the duration layer.
        """
        # 计算每个样本中非填充标记的数量，即单位的长度
        unit_lengths = (input_ids != self.pad_token_id).sum(1)

        # 处理边界情况，确保长度在合理范围内
        unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)

        # 计算累积的持续时间输出，并根据单位长度获取相应位置的值
        cumulative_dur_out = torch.cumsum(dur_out, dim=1)
        unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()

        return unit_lengths

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan中复制而来，计算HiFi-GAN输出的长度
    def _get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the hifigan convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的一维卷积层输出长度公式
            return (
                torch.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode="floor") + 1
            )

        def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            # 计算转置卷积层输出长度的函数
            return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

        # conv_pre 部分，计算输入长度经过卷积层后的输出长度
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        # upsampler 部分，遍历配置中的上采样率和卷积核大小，计算转置卷积层的输出长度
        for i, (upsample_rate, kernel_size) in enumerate(
            zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)
        ):
            input_lengths = _transpose_conv_out_length(
                input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2
            )

        # resblock 部分，遍历配置中的卷积核大小和膨胀系数，计算卷积层的输出长度
        for i in range(len(self.config.upsample_rates)):
            for kernel_size, dilation in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                for dil in dilation:
                    input_lengths = _conv_out_length(
                        input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil
                    )

                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

        # conv_post 部分，再次应用一维卷积层计算最终输出长度
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        return input_lengths

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan.forward 复制，修改了模型名字和语音者ID
    def forward(
        self, input_ids: torch.LongTensor, speaker_id: torch.Tensor, lang_id: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                
                Indices can be obtained using [`SeamlessM4Tv2TextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            speaker_id (`int`, *optional*):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional*):
                The language id to use as target language for translation.
        """
        # 将输入的 token 索引转换为单位嵌入向量并进行维度转置，得到形状为 (batch_size, C, sequence_length) 的隐藏状态
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)
        # 根据说话者 ID 获取说话者的嵌入向量并进行维度转置，形状为 (1, C, 1)
        spkr = self.speaker_embedding(speaker_id).transpose(1, 2)
        # 根据语言 ID 获取语言的嵌入向量并进行维度转置，形状为 (1, C, 1)
        lang = self.language_embedding(lang_id).transpose(1, 2)

        # 使用隐藏状态作为输入，通过时长预测器获取对数时长预测值
        log_dur_pred = self.dur_predictor(hidden_states.transpose(1, 2))
        # 将对数时长预测值转换为正数，再四舍五入为最接近的整数，然后进行截断，形状为 (batch_size, 1, T)
        dur_out = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1)
        
        # 如果 batch_size 为 1，将隐藏状态重复多次，使得每个 token 的长度与 dur_out 中的持续时间匹配
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            # 如果存在多个样本且处于训练模式下，发出警告说明因为数据交错而失去并行性
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            # 对于每个样本，根据对应的持续时间将隐藏状态插值并进行转置
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=-1).transpose(0, 1)
                for (hidden_state, duration) in zip(hidden_states, dur_out)
            ]
            # 将插值后的隐藏状态序列进行填充以保持批处理的顺序，并进行维度转置，形状为 (batch_size, C, T)
            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)

        # 将说话者和语言的嵌入向量根据长度进行重复，使得它们与隐藏状态的长度相匹配
        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        # 将语言、隐藏状态和说话者的嵌入向量连接起来，形状为 (batch_size, C + 2 * C + 1, T)
        hidden_states = torch.cat([lang, hidden_states, spkr], dim=1)

        # 使用 Hifi-GAN 模型处理连接后的隐藏状态，得到输出
        hidden_states = self.hifi_gan(hidden_states)

        # 根据输入 token 的持续时间和持续长度获取单元的长度
        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        # 根据单元的长度获取 Hifi-GAN 模型的输出长度
        lengths = self._get_output_hifigan_lengths(unit_lengths)

        return hidden_states, lengths

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan._init_weights 复制而来
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果 module 是线性层、一维卷积层或一维转置卷积层，则初始化权重为正态分布
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是嵌入层，则初始化权重为正态分布
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    # 为当前类定义一个方法，用于应用权重归一化
    def apply_weight_norm(self):
        # 对 self.hifi_gan.conv_pre 应用 PyTorch 中的权重归一化
        nn.utils.weight_norm(self.hifi_gan.conv_pre)
        # 对 self.hifi_gan.upsampler 中的每一层应用权重归一化
        for layer in self.hifi_gan.upsampler:
            nn.utils.weight_norm(layer)
        # 对 self.hifi_gan.resblocks 中的每一个残差块应用其定义的权重归一化方法
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        # 对 self.hifi_gan.conv_post 应用 PyTorch 中的权重归一化
        nn.utils.weight_norm(self.hifi_gan.conv_post)
    
    # 为当前类定义一个方法，用于移除权重归一化
    def remove_weight_norm(self):
        # 对 self.hifi_gan.conv_pre 移除 PyTorch 中的权重归一化
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        # 对 self.hifi_gan.upsampler 中的每一层移除权重归一化
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        # 对 self.hifi_gan.resblocks 中的每一个残差块移除其定义的权重归一化方法
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        # 对 self.hifi_gan.conv_post 移除 PyTorch 中的权重归一化
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)
############ WHOLE MODEL related code ################

# 引入文本到文本转换的SeamlessM4Tv2模型转换器，用于T2TT任务
# SEAMLESS_M4T_V2_START_DOCSTRING为其文档字符串的一部分
@add_start_docstrings(
    "The text-to-text SeamlessM4Tv2 Model transformer which can be used for T2TT.",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToText复制而来，对应部分名称被改为SeamlessM4Tv2
# 还有SeamlessM4TTokenizer->SeamlessM4Tv2Tokenizer, SeamlessM4TProcessor->SeamlessM4TProcessor
class SeamlessM4Tv2ForTextToText(SeamlessM4Tv2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["speech_encoder", "t2u_model", "vocoder"]
    main_input_name = "input_ids"

    # 转为SeamlessM4Tv2模型的权重绑定键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)

        # 共享的嵌入层，使用config中定义的词汇量大小、隐藏大小和填充标记ID
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 文本编码器和文本解码器，使用SeamlessM4Tv2Encoder和SeamlessM4Tv2Decoder初始化
        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)

        # 语言模型头部，线性层，输入维度为隐藏大小，输出维度为词汇量大小，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取文本编码器
    def get_encoder(self):
        return self.text_encoder

    # 获取文本解码器
    def get_decoder(self):
        return self.text_decoder

    # 获取输出的嵌入层（语言模型头部）
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出的嵌入层（语言模型头部）
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输入的嵌入层（用于文本解码器）
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置输入的嵌入层（文本编码器、文本解码器、共享嵌入层都使用同一个嵌入）
    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 如果配置允许，则绑定权重（文本编码器的词嵌入、文本解码器的词嵌入、语言模型头部的权重）
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 前向传播函数，接受一系列输入参数，详细说明见M4T_TEXT_INPUTS_DOCSTRING
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
    ):
        # 生成文本的方法，接受多个参数用于控制生成过程和配置
        def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
        ):
            # 如果使用了过去的键值（past_key_values），则截断decoder_input_ids
            if past_key_values is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

            # 返回包含生成所需输入的字典
            return {
                "input_ids": None,  # encoder_outputs 已定义，因此不需要input_ids
                "encoder_outputs": encoder_outputs,
                "past_key_values": past_key_values,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
            }

        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            reordered_past = ()
            # 对每层的过去键值进行重新排序
            for layer_past in past_key_values:
                # 对于缓存的跨注意力状态，不需要重新排序，因为它们始终相同
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
                )
            # 返回重新排序后的过去键值
            return reordered_past
# 定义一个带有文档字符串的类，用于将语音转文本，基于 SeamlessM4Tv2 模型的变形器
@add_start_docstrings(
    "The speech-to-text SeamlessM4Tv2 Model transformer which can be used for S2TT.",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
class SeamlessM4Tv2ForSpeechToText(SeamlessM4Tv2PreTrainedModel):
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_missing = ["text_decoder", "t2u_model", "vocoder"]
    # 主要输入名称
    main_input_name = "input_features"

    # 被绑定权重的键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.__init__ 复制而来，将 SeamlessM4T 替换为 SeamlessM4Tv2
    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)

        # 共享的嵌入层，用于词汇表的嵌入
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 语音编码器，基于 SeamlessM4Tv2SpeechEncoder 类
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        # 文本解码器，基于 SeamlessM4Tv2Decoder 类，共享嵌入层
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        # 语言模型头部，线性层，用于输出词汇表的大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_encoder 复制而来
    def get_encoder(self):
        # 返回语音编码器
        return self.speech_encoder

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_decoder 复制而来
    def get_decoder(self):
        # 返回文本解码器
        return self.text_decoder

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_output_embeddings 复制而来
    def get_output_embeddings(self):
        # 返回语言模型头部，用于输出词汇表的大小
        return self.lm_head

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.set_output_embeddings 复制而来
    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的新嵌入层
        self.lm_head = new_embeddings

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_input_embeddings 复制而来
    def get_input_embeddings(self):
        # 返回文本解码器的嵌入层
        return self.text_decoder.embed_tokens

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.set_input_embeddings 复制而来
    def set_input_embeddings(self, value):
        # 设置文本解码器的嵌入层
        self.text_decoder.embed_tokens = value

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText._tie_weights 复制而来
    def _tie_weights(self):
        # 如果配置要求词嵌入层绑定
        if self.config.tie_word_embeddings:
            # 绑定文本解码器的嵌入层和共享的嵌入层
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            # 绑定语言模型头部和共享的嵌入层
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.forward 复制而来
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        input_features: torch.LongTensor = None,  # 输入特征，类型为长整型张量，可选参数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选的张量类型
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的标识符，可选的长整型张量
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器注意力掩码，可选的长整型张量
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器输出，可选的浮点型张量元组
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，可选的浮点型张量元组
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入，可选的浮点型张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入，可选的浮点型张量
        labels: Optional[torch.LongTensor] = None,  # 标签，可选的长整型张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果，可选的布尔值
        **kwargs,  # 其他关键字参数
    ):
    
    # 从`transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.generate`复制而来的方法
    def generate(
        self,
        input_features=None,  # 输入特征，未指定类型
        tgt_lang=None,  # 目标语言，未指定类型
        generation_config=None,  # 生成配置，未指定类型
        logits_processor=None,  # 对数处理器，未指定类型
        stopping_criteria=None,  # 停止条件，未指定类型
        prefix_allowed_tokens_fn=None,  # 前缀允许标记函数，未指定类型
        synced_gpus=False,  # 是否同步 GPU，布尔类型，默认为 False
        **kwargs,  # 其他关键字参数
    ):
    
    # 从`transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.prepare_inputs_for_generation`复制而来的方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入标识符
        past_key_values=None,  # 过去的键值对，可选参数
        attention_mask=None,  # 注意力掩码，可选参数
        use_cache=None,  # 是否使用缓存，可选参数
        encoder_outputs=None,  # 编码器输出，可选参数
        **kwargs,  # 其他关键字参数
    ):
        # 如果使用过去的键值对，则仅保留最后一个解码器输入标识符
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
    
        # 返回一个字典，包含生成所需的输入信息
        return {
            "input_ids": None,  # 不需要输入标识符，因为已经定义了 encoder_outputs
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键值对
            "decoder_input_ids": decoder_input_ids,  # 解码器输入标识符
            "attention_mask": attention_mask,  # 注意力掩码
            "use_cache": use_cache,  # 是否使用缓存
        }
    
    @staticmethod
    # 从`transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText._reorder_cache`复制而来的静态方法
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序，它们总是相同的
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
# 将类标记为用于文本到语音转换的 SeamlessM4Tv2 模型转换器，并添加相关文档字符串
@add_start_docstrings(
    "The text-to-speech SeamlessM4Tv2 Model transformer which can be used for T2ST.",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
class SeamlessM4Tv2ForTextToSpeech(SeamlessM4Tv2PreTrainedModel):
    # 在模型加载时应忽略的键列表
    _keys_to_ignore_on_load_missing = ["speech_encoder"]
    # 主输入名称
    main_input_name = "input_ids"

    # 被绑定权重的键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.__init__ 复制而来，用于初始化模型
    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)

        # 共享的嵌入层，用于词汇表大小、隐藏大小和填充标记ID的嵌入
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 文本编码器和解码器
        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        # 语言模型的头部线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

        # 文本到单元的模型和语音合成器
        self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4Tv2CodeHifiGan(config)

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.get_encoder 复制而来，返回文本编码器
    def get_encoder(self):
        return self.text_encoder

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.get_decoder 复制而来，返回文本解码器
    def get_decoder(self):
        return self.text_decoder

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.get_output_embeddings 复制而来，返回输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.set_output_embeddings 复制而来，设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.get_input_embeddings 复制而来，返回输入的嵌入层
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.set_input_embeddings 复制而来，设置输入的嵌入层
    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech._tie_weights 复制而来，用于绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 将 M4T_TEXT_INPUTS_DOCSTRING 添加到模型的前向传播函数
    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.forward with SeamlessM4T->SeamlessM4Tv2
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
        # 前向传播函数，接收多个输入参数用于模型推断
        @torch.no_grad()
        def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            return_intermediate_token_ids: Optional[bool] = None,
            tgt_lang: Optional[str] = None,
            speaker_id: Optional[int] = 0,
            **kwargs,
        ):
            # 生成函数，生成目标语言的文本输出
            pass

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 为生成准备输入的函数，根据需要裁剪 decoder_input_ids 如果使用了过去的 key values
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个包含生成所需输入的字典
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        # 静态方法，重新排序缓存中的过去 key values
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每个层级的过去 key values 执行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
# 使用装饰器添加模型文档字符串，描述了这个模型可以用于语音到语音的转换任务
@add_start_docstrings(
    "The speech-to-speech SeamlessM4Tv2 Model transformer which can be used for S2ST.",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
class SeamlessM4Tv2ForSpeechToSpeech(SeamlessM4Tv2PreTrainedModel):
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_missing = ["text_encoder"]
    # 主要输入名称
    main_input_name = "input_features"

    # 要绑定权重的关键字列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 从SeamlessM4TForSpeechToSpeech.__init__复制过来的初始化方法，但使用了SeamlessM4Tv2
    def __init__(self, config):
        super().__init__(config)

        # 共享的嵌入层，用于文本解码和语音编码
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 语音编码器
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        # 文本解码器
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

        # 文本到单元的转换模型
        self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)
        # 高保真生成模块
        self.vocoder = SeamlessM4Tv2CodeHifiGan(config)

    # 从SeamlessM4TForSpeechToSpeech.get_encoder复制过来的方法，返回语音编码器
    def get_encoder(self):
        return self.speech_encoder

    # 从SeamlessM4TForSpeechToSpeech.get_decoder复制过来的方法，返回文本解码器
    def get_decoder(self):
        return self.text_decoder

    # 从SeamlessM4TForSpeechToSpeech.get_output_embeddings复制过来的方法，返回语言模型头
    def get_output_embeddings(self):
        return self.lm_head

    # 从SeamlessM4TForSpeechToSpeech.set_output_embeddings复制过来的方法，设置新的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从SeamlessM4TForSpeechToSpeech.get_input_embeddings复制过来的方法，返回文本解码器的嵌入层
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 从SeamlessM4TForSpeechToSpeech.set_input_embeddings复制过来的方法，设置新的输入嵌入层
    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    # 从SeamlessM4TForSpeechToSpeech._tie_weights复制过来的方法，根据配置绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 使用装饰器添加模型前向传播方法的文档字符串，描述了输入和输出的详细信息
    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    # 从SeamlessM4TForSpeechToSpeech.forward复制过来的方法，但使用了SeamlessM4Tv2
    # 定义一个方法用于模型的前向传播，接受多个输入参数
    def forward(
        self,
        input_features: torch.LongTensor = None,  # 输入特征张量，类型为长整型Tensor，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选参数，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的标识符张量，可选参数，默认为None
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器注意力掩码张量，可选参数，默认为None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器输出的元组张量，可选参数，默认为None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键-值对的元组张量，可选参数，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入张量，可选参数，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入张量，可选参数，默认为None
        labels: Optional[torch.LongTensor] = None,  # 标签张量，可选参数，默认为None
        use_cache: Optional[bool] = None,  # 是否使用缓存的布尔值，可选参数，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力张量的布尔值，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的布尔值，可选参数，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典的布尔值，可选参数，默认为None
        **kwargs,  # 其他未命名参数
    ):
        pass  # 此处未实现具体的功能，仅为声明方法的占位符

    @torch.no_grad()
    # 定义一个生成方法，不进行梯度计算
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,  # 输入特征张量，可选参数，默认为None
        return_intermediate_token_ids: Optional[bool] = None,  # 是否返回中间令牌ID的布尔值，可选参数，默认为None
        tgt_lang: Optional[str] = None,  # 目标语言的字符串，可选参数，默认为None
        speaker_id: Optional[int] = 0,  # 说话者ID的整数，可选参数，默认为0
        **kwargs,  # 其他未命名参数
    ):
        pass  # 此处未实现具体的功能，仅为声明方法的占位符

    @staticmethod
    # 静态方法：重新排序缓存中的过去键-值对
    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech._reorder_cache 复制而来
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去键-值对元组
        reordered_past = ()
        # 遍历每个层的过去键-值对
        for layer_past in past_key_values:
            # 对于非交叉注意力状态，不需要重新排序，保持不变
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.prepare_inputs_for_generation 复制而来
    # 准备生成时的输入参数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的标识符张量
        past_key_values=None,  # 过去的键-值对，可选参数，默认为None
        attention_mask=None,  # 注意力掩码张量，可选参数，默认为None
        use_cache=None,  # 是否使用缓存的布尔值，可选参数，默认为None
        encoder_outputs=None,  # 编码器输出的张量，可选参数，默认为None
        **kwargs,  # 其他未命名参数
    ):
        # 如果使用了过去的键-值对，截断解码器的输入标识符
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，包含生成所需的输入参数
        return {
            "input_ids": None,  # 输入标识符不需要，因为已经定义了 encoder_outputs
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键-值对
            "decoder_input_ids": decoder_input_ids,  # 解码器输入的标识符
            "attention_mask": attention_mask,  # 注意力掩码
            "use_cache": use_cache,  # 是否使用缓存
        }
@add_start_docstrings(
    "The original SeamlessM4Tv2 Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        current_modality (`str`, *optional*, defaults to `"text"`):
            Default modality. Used only to initialize the model. It can be set to `"text"` or `"speech"`.
            This will be updated automatically according to the modality passed to the forward and generate passes (`input_ids` for text and `input_features` for audio).
    """,
)
class SeamlessM4Tv2Model(SeamlessM4Tv2PreTrainedModel):
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.__init__ with SeamlessM4T->SeamlessM4Tv2
    def __init__(self, config, current_modality="text"):
        super().__init__(config)

        # Initialize shared embeddings based on vocabulary size and hidden size
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # Initialize text encoder, speech encoder, text decoder, and language model head
        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Set default modality
        self.current_modality = current_modality
        if current_modality == "speech":
            self.main_input_name = "input_features"

        # Initialize text-to-unit model and vocoder
        # These models already call post_init in their initialization
        self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4Tv2CodeHifiGan(config)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.set_modality
    def set_modality(self, modality="text"):
        # Set main input name and current modality based on input modality
        if modality == "text":
            self.main_input_name = "input_ids"
            self.current_modality = "text"
        elif modality == "speech":
            self.main_input_name = "input_features"
            self.current_modality = "speech"
        else:
            raise ValueError(f"`modality={modality}` is not a valid modality. It must be `text` or `speech`.")

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.get_encoder
    def get_encoder(self):
        # Return the appropriate encoder based on current modality
        if self.current_modality == "text":
            return self.text_encoder
        else:
            return self.speech_encoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.get_output_embeddings
    def get_output_embeddings(self):
        # Return the language model head (output embeddings)
        return self.lm_head

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        # Set new embeddings for the language model head
        self.lm_head = new_embeddings
    # 从模型中获取输入嵌入层的引用
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置输入嵌入层的值，同时设置编码器和解码器的嵌入层以及共享的值
    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 根据配置决定是否绑定词嵌入层的权重，用于语言模型头部
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_MODEL_INPUTS_DOCSTRING)
    # 模型的前向传播函数，根据输入参数执行模型计算
    # 与原始模型名不同，此处使用了"SeamlessM4Tv2"
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
        pass

    @torch.no_grad()
    # 根据生成任务准备模型的输入，用于生成阶段
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: Optional[int] = 0,
        generate_speech: Optional[bool] = True,
        **kwargs,
    ):
        pass

    # 准备生成阶段的输入，包括解码器的输入ID、过去的关键值、注意力掩码、是否使用缓存以及编码器输出等
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        pass
        # 如果使用了过去的键值对，根据past_key_values调整decoder_input_ids的形状，仅保留最后一个位置的token输入
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个包含多个项的字典，表示模型的输出
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键值对，用于缓存
            "decoder_input_ids": decoder_input_ids,  # 解码器的输入ids
            "attention_mask": attention_mask,  # 注意力掩码
            "use_cache": use_cache,  # 是否使用缓存
        }

    @staticmethod
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel._reorder_cache复制而来
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()  # 初始化一个空元组用于存储重新排序后的过去的键值对

        # 遍历过去的键值对中的每一层(layer_past)
        for layer_past in past_key_values:
            # 对于每一层的前两个状态，根据beam_idx重新排序，保留其余部分不变
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )

        # 返回重新排序后的过去的键值对
        return reordered_past
```