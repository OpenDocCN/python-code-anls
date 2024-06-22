# `.\transformers\models\seamless_m4t_v2\modeling_seamless_m4t_v2.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 使用此文件，详细信息请参考 License
# 网址：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，只能在符合许可证条件的情况下使用本文件
# 在“原样”基础下分发软件，不提供任何担保或条件，无论是明示或暗示的
# 详细信息请查看给定的许可证
""" PyTorch SeamlessM4Tv2 模型."""

# 导入模块
import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

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
# 导入 SeamlessM4Tv2Config 配置
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config

# 获取 Logger 对象
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = ""
_CONFIG_FOR_DOC = "SeamlessM4Tv2Config"

# 预训练模型列表
SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/seamless-m4t-v2-large",
    # 查看所有 SeamlessM4T-v2 模型网址：https://huggingface.co/models?filter=seamless_m4t_v2
]

# 预训练 HiFi-GAN 的 SpeechT5 配置映射
SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_hifigan": "https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/config.json",
}

# 创建 SeamlessM4Tv2GenerationOutput 类
@dataclass
# 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TGenerationOutput 复制到 SeamlessM4Tv2GenerationOutput，
# 用于 SeamlessM4Tv2Model、SeamlessM4Tv2ForTextToText、SeamlessM4Tv2ForTextToSpeech、SeamlessM4Tv2ForSpeechToSpeech 和 SeamlessM4Tv2ForTextToSpeech 生成输出
class SeamlessM4Tv2GenerationOutput(ModelOutput):
    """
    Class defining the generated outputs from [`SeamlessM4Tv2Model`], [`SeamlessM4Tv2ForTextToText`],
    [`SeamlessM4Tv2ForTextToSpeech`], [`SeamlessM4Tv2ForSpeechToSpeech`] and [`SeamlessM4Tv2ForTextToSpeech`].
    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
            模型预测的最终音频波形。
        waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
            The length in samples of each element in the `waveform` batch.
            `waveform` 批中每个元素的样本长度。
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The generated translated sequences. This is the output of the text-to-text or the speech-to-text models.
            The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished
            early due to the `eos_token_id`.
            生成的翻译序列。这是文本到文本或语音到文本模型的输出。
            第二维度 (sequence_length) 要么等于 `max_length`，要么较短，如果所有批次都因 `eos_token_id` 提前结束。
        unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            The generated translated unit sequences. This is the output of the text-to-units model. The second
            dimension (unit_sequence_length) is either equal to `t2u_max_length` or shorter if all batches finished
            early due to the `t2u_eos_token_id`.
            生成的单元序列翻译。这是文本到单元模型的输出。
            第二维度 (unit_sequence_length) 要么等于 `t2u_max_length`，要么较短，如果所有批次都因 `t2u_eos_token_id` 提前结束。
    """

    waveform: Optional[torch.FloatTensor] = None
    waveform_lengths: Optional[torch.IntTensor] = None
    sequences: Optional[Tuple[torch.FloatTensor]] = None
    unit_sequences: Optional[Tuple[torch.FloatTensor]] = None
# 定义用于 SeamlessM4Tv2TextToUnitDecoder 的输出数据类
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

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    padding_mask: Optional[torch.Tensor] = None


# 定义用于 SeamlessM4Tv2TextToUnitForConditionalGeneration 和 SeamlessM4Tv2TextToUnitModel 的输出数据类
@dataclass
class SeamlessM4Tv2TextToUnitOutput(ModelOutput):
    """
        Class defining the outputs from [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] and
        [`SeamlessM4Tv2TextToUnitModel`].
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
            for *masked*
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
    """

    # 最后一个隐藏状态，默认为 None
    last_hidden_state: torch.FloatTensor = None
    # 填充掩码，默认为 None
    padding_mask: Optional[torch.Tensor] = None
    # 定义可选类型的变量，用于存储解码器隐藏状态、注意力权重、编码器最后隐藏状态、编码器隐藏状态和注意力权重以及损失值
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None
# 模型的文档字符串，描述了此模型是一个 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类
SEAMLESS_M4T_V2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~SeamlessM4Tv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 描述多模态输入的文档字符串，包括参数input_ids和input_features
SEAMLESS_M4T_V2_MULTIMODAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
    """

# 描述文本输入的文档字符串，包括参数input_ids
M4T_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """

# 描述语音输入的文档字符串，包括参数input_features
M4T_SPEECH_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
            Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
            [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
        """

# 模型输入的文档字符串，包括多模态输入的描述
M4T_MODEL_INPUTS_DOCSTRING = SEAMLESS_M4T_V2_MULTIMODAL_INPUTS_DOCSTRING + SEAMLESS_M4T_V2_END_INPUTS_DOCSTRING

# 文本输入的文档字符串，包括多模态输入和结束输入的描述
M4T_TEXT_INPUTS_DOCSTRING = M4T_TEXT_INPUTS_DOCSTRING + SEAMLESS_M4T_V2_END_INPUTS_DOCSTRING

# 语音输入的文档字符串，包括多模态输入和结束输入的描述
M4T_SPEECH_INPUTS_DOCSTRING = M4T_SPEECH_INPUTS_DOCSTRING + SEAMLESS_M4T_V2_END_INPUTS_DOCSTRING

# 文本到单元输入的文档字符串为空
M4T_TEXT_TO_UNITS_INPUTS_DOCSTRING = r"""
    # 输入序列标记的索引。如果不传入则为可选参数
    # 可以使用`SeamlessM4TTokenizer`或`SeamlessM4TProcessor`获取索引。详见`PreTrainedTokenizer.encode`和`PreTrainedTokenizer.__call__`
    # 什么是输入标识？(../glossary#input-ids)
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        
        # 字符索引。字符和索引的对应关系可以在`char_to_id`中找到，`char_to_id`是生成配置中的一个字典
        char_input_ids (`torch.LongTensor` of shape `(batch_size, char_sequence_length)`):
        
        # 每个输入标识的字符数
        char_count_per_id (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        
        # 避免在填充标记索引上执行注意力的蒙版。蒙版值取在`[0, 1]`范围内：
        # - 对于**未蒙版**的标记为1
        # - 对于**蒙版**的标记为0
        # 什么是注意力蒙版？(../glossary#attention-mask)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
        
        # 编码器输出
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
        
        # 输入嵌入
        inputs_embeds (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*):
        
        # 用于计算掩码语言建模损失的标签。索引应在`[-100, 0, ..., config.vocab_size]`范围内
        # 索引设置为`-100`的标记将被忽略(蒙版)，损失仅计算标签为`[0, ..., config.vocab_size]`的标记
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        
        # 是否返回所有注意力层的注意力张量
        output_attentions (`bool`, *optional*):
        
        # 是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
        
        # 是否返回一个[`~utils.ModelOutput`]，而不是普通的元组
        return_dict (`bool`, *optional*):
# 从输入的input_ids中创建位置id，用于代替非padding符号，位置号从padding_idx+1开始，padding符号被忽略
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 使用input_ids中不等于padding_idx的元素创建一个mask
    mask = input_ids.ne(padding_idx).int()
    # 计算每个位置的累计值，并加上past_key_values_length，然后乘以mask
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

# 将输入的input_ids向右移动一个token
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 创建一个与input_ids相同形状的tensor并填充为0
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将input_ids向右移动一位并赋值给创建的tensor
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将decoder_start_token_id赋值给新的tensor的第一个位置
    shifted_input_ids[:, 0] = decoder_start_token_id
    # 如果pad_token_id为None，则抛出ValueError异常
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

# 计算新的注意力掩码，为每个元素的位置计算一个注意力掩码，使得注意力停在对应的seq_lens位置
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    # 获取hidden_states的形状信息
    batch_size, mask_seq_len = hidden_states.shape[:2]
    # 创建一个indices从0到mask_seq_len的tensor
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)
    # 创建一个布尔掩码，如果indices大于等于seq_lens，则为True
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
    # 创建一个形状和hidden_states一样的全一的掩码tensor
    mask = hidden_states.new_ones((batch_size, mask_seq_len))
    # 将mask中的bool_mask位置上的元素设为0
    mask = mask.masked_fill(bool_mask, 0)
    return mask

# 为生成语音的SeamlessM4Tv2模型格式化kwargs，将kwargs属性分配给文本生成模型或语音生成模型
def format_speech_generation_kwargs(kwargs):
    pass  # 由于该函数的实现没在代码中给出，因此无法添加具体的注释
    Args:
        kwargs (`dict`):
             Keyword arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                except for `decoder_input_ids` which will only be passed through the text components.
                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                text model and speech model respectively. It has the priority over the keywords without a prefix.

                This means you can, for example, specify a generation strategy for one generation but not for the
                other.

参数：
-   kwargs（`dict`）:
    -   关键字参数有两种类型：
        -   如果没有前缀，则作为每个子模型的`generate`方法的`**kwargs`输入，除了`decoder_input_ids`，它只会通过文本组件传递。
        -   如果具有 *text_* 或 *speech_* 前缀，则会分别作为文本模型和语音模型的`generate`方法的输入。它优先于没有前缀的关键字。

        这意味着你可以为一个生成过程指定一个生成策略，但不指定另一个生成过程的策略。


    # attribute kwargs to models
    kwargs_text = {}
    kwargs_speech = {}
    for key, value in kwargs.items():
        if key.startswith("text_"):
            key = key[len("text_") :]
            kwargs_text[key] = value
        elif key.startswith("speech_"):
            key = key[len("speech_") :]
            kwargs_speech[key] = value
        else:
            # If the key is already in a specific config, then it's been set with a
            # submodules specific value and we don't override
            if key not in kwargs_text:
                kwargs_text[key] = value
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    return kwargs_text, kwargs_speech

将参数映射到模型：
-   创建空字典`kwargs_text`和`kwargs_speech`用于存储映射后的参数
-   遍历参数字典`kwargs`中的键值对
-   如果键以"text_"开头，则将键名截掉"text_"前缀，并将对应值存储到`kwargs_text`中
-   如果键以"speech_"开头，则将键名截掉"speech_"前缀，并将对应值存储到`kwargs_speech`中
-   否则，如果键不在`kwargs_text`中，则将键值存储到`kwargs_text`中
-   同样地，如果键不在`kwargs_speech`中，则将键值存储到`kwargs_speech`中
-   返回映射后的参数字典`kwargs_text`和`kwargs_speech`
# 定义一个名为SeamlessM4Tv2ConformerFeatureProjection的类，继承自nn.Module
class SeamlessM4Tv2ConformerFeatureProjection(nn.Module):
    # 初始化方法，接受一个config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化层归一化模块，使用config中的feature_projection_input_dim作为输入维度，config中的layer_norm_eps作为eps
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        # 初始化线性映射模块，将输入维度为config.feature_projection_input_dim映射到config中的hidden_size
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        # 初始化dropout模块，使用config中的speech_encoder_dropout作为dropout比例
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    # 前向传播方法，接受一个hidden_states参数
    def forward(self, hidden_states):
        # 对非映射的隐藏状态进行层归一化，转换为层归一化模块的数据类型，并赋值给norm_hidden_states
        norm_hidden_states = self.layer_norm(hidden_states.to(self.layer_norm.weight.dtype))
        # 对层归一化后的隐藏状态进行线性映射
        hidden_states = self.projection(norm_hidden_states)
        # 对线性映射后的隐藏状态进行dropout
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states

# 定义一个名为SeamlessM4Tv2ConformerFeedForward的类，继承自nn.Module
class SeamlessM4Tv2ConformerFeedForward(nn.Module):
    # 初始化方法，接受一个config参数、一个act_fn参数和一个dropout参数
    def __init__(self, config, act_fn=None, dropout=None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果没有传入dropout参数，则使用config中的speech_encoder_dropout作为dropout比例
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        # 如果没有传入act_fn参数，则使用config中的speech_encoder_hidden_act作为激活函数
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act
        # 初始化中间层dropout模块，使用传入的或config中的dropout比例
        self.intermediate_dropout = nn.Dropout(dropout)
        # 初始化中间层线性映射模块，将输入维度为config.hidden_size映射到config.speech_encoder_intermediate_size
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        # 初始化中间层激活函数，并根据act_fn的类型选择对应的激活函数
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn
        # 初始化输出层线性映射模块，将输入维度为config.speech_encoder_intermediate_size映射到config.hidden_size
        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        # 初始化输出层dropout模块，使用传入的或config中的dropout比例
        self.output_dropout = nn.Dropout(dropout)

    # 前向传播方法，接受一个hidden_states参数
    def forward(self, hidden_states):
        # 对输入的隐藏状态进行中间层线性映射
        hidden_states = self.intermediate_dense(hidden_states)
        # 对中间层线性映射后的隐藏状态进行中间层激活函数的计算
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对中间层激活函数计算后的隐藏状态进行中间层dropout
        hidden_states = self.intermediate_dropout(hidden_states)
        # 对中间层处理后的隐藏状态进行输出层线性映射
        hidden_states = self.output_dense(hidden_states)
        # 对输出层线性映射后的隐藏状态进行输出层dropout
        hidden_states = self.output_dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states

# 定义一个名为SeamlessM4Tv2ConformerConvolutionModule的类，继承自nn.Module
class SeamlessM4Tv2ConformerConvolutionModule(nn.Module):
    # 描述信息，未定义初始化方法和前向传播方法，需要补充完整
    """Convolution block used in the conformer block. Uses a causal depthwise convolution similar to that
    described in Section 2.1 of `https://doi.org/10.48550/arxiv.1609.03499"""
    # 初始化函数，传入配置参数
    def __init__(self, config):
        super().__init__()
        # 检查深度可分离卷积的核尺寸是否为奇数，以保证 'SAME' 填充
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        # 创建 LayerNorm 层
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        # 创建 1D 卷积层（1x1 卷积）
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
        # 创建深度可分离卷积层
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        # 创建深度卷积后的 LayerNorm 层
        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size)
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        # 创建第二个 1D 卷积层（1x1 卷积）
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    # 前向传播函数
    def forward(self, hidden_states, attention_mask=None):
        # 对输入进行 LayerNorm 处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果存在注意力掩码，则用0来填充相应位置，避免信息泄漏
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # 交换时间维度和特征维度
        hidden_states = hidden_states.transpose(1, 2)

        # 执行 GLU 机制
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        # 为了因果卷积，对序列左边进行填充
        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # 执行深度可分离卷积
        hidden_states = self.depthwise_conv(hidden_states)
        # 对深度卷积结果进行 LayerNorm 处理
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        # 执行 Dropout
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
# 定义一个SeamlessM4Tv2ConformerSelfAttention类，用于构建自注意力模型
# 可以通过相对位置嵌入进行增强
class SeamlessM4Tv2ConformerSelfAttention(nn.Module):
    """Construct a SeamlessM4Tv2ConformerSelfAttention object.
    Can be enhanced with relative position embeddings.
    """

    def __init__(self, config, use_position_embeddings=True):
        # 初始化函数
        super().__init__()

        # 设置头大小为隐藏层大小除以注意力头数
        self.head_size = config.hidden_size // config.speech_encoder_attention_heads
        # 设置注意力头数
        self.num_heads = config.speech_encoder_attention_heads
        # 如果使用位置嵌入，则设置位置嵌入类型
        self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None

        # 定义线性变换层，用于计算查询、键、值和输出
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        # 定义dropout层，用于进行随机失活
        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

        # 如果位置嵌入类型为"relative_key"，则初始化相对位置嵌入
        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = config.left_max_position_embeddings
            self.right_max_position_embeddings = config.right_max_position_embeddings
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 定义函数签名，输入隐藏状态张量，输出注意力输出张量、注意力权重张量（可选）、注意力头张量元组（可选）

        # 获取隐藏状态张量的批大小、序列长度和隐藏大小
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # 确保查询/键状态可以不等于值状态
        query_key_states = hidden_states
        value_states = hidden_states

        # 将查询/键状态和值状态进行投影
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # 转置查询、键和值张量的维度
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        # 如果位置嵌入类型是"relative_key"，则使用相对位置信息调整注意力权重
        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

            positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

            relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
            attn_weights = attn_weights + (relative_position_attn_weights / math.sqrt(self.head_size))

        # 如果存在注意力遮罩，则将其应用于注意力权重
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 对注意力权重进行 softmax 归一化
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value)

        # 重塑注意力输出的形状
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.linear_out(attn_output)

        # 如果不输出注意力权重，则将其设为 None
        if not output_attentions:
            attn_weights = None

        # 返回注意力输出、注意力权重（可选）和注意力头张量元组（可选）
        return attn_output, attn_weights
class SeamlessM4Tv2ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    # 用于初始化 Conformer 编码器层的类
    # 从transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ 复制而来，将Wav2Vec2->SeamlessM4Tv2, attention_dropout->speech_encoder_dropout, torch.nn->nn
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout

        # Feed-forward 1
        # 第一个前馈网络层，用于学习特征表示
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = SeamlessM4Tv2ConformerFeedForward(config)

        # Self-Attention
        # 自注意力机制层
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config)

        # Conformer Convolution
        # Conformer卷积模块
        self.conv_module = SeamlessM4Tv2ConformerConvolutionModule(config)

        # Feed-forward 2
        # 第二个前馈网络层，用于学习特征表示
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
        # 前馈网络1层
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        # 自注意力机制层
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        # 卷积层
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        # 前馈网络2层
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weights


class SeamlessM4Tv2ConformerEncoder(nn.Module):
    # 初始化函数，接受配置参数，并调用父类的初始化方法
    def __init__(self, config):
        super().__init__()
        # 保存配置参数到实例变量
        self.config = config

        # 创建一个dropout层，用于随机丢弃输入张量中的部分元素
        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        # 创建一个包含多个SeamlessM4Tv2ConformerEncoderLayer层的模块列表
        self.layers = nn.ModuleList(
            [SeamlessM4Tv2ConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        # 创建一个LayerNorm层，用于对输入张量进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 设置梯度检查点为False
        self.gradient_checkpointing = False

    # 将注意力机制应用于分块的隐藏状态张量
    def _apply_chunk_attention(self, attention_mask, hidden_states):
        """
        Creates a chunk attention mask. It creates a mask to prevent attention across chunks, ensuring that each
        position attends only to positions within its own chunk. If a left chunk overlap is specified
        (`speech_encoder_chunk_size` in the configuration), the attention mask is adjusted accordingly to allow each
        position to also attends the `speech_encoder_chunk_size - 1` previous chunks.
        """
        # 获取隐藏状态张量的序列长度
        sequence_len = hidden_states.shape[1]

        # 根据序列长度创建分块索引
        chunk_indices = torch.arange(sequence_len, device=hidden_states.device)
        chunk_indices = torch.div(chunk_indices, self.config.speech_encoder_chunk_size).long()

        # 设置起始索引为0，根据左侧分块重叠数量调整起始索引
        start_indices = torch.full_like(chunk_indices, 0)
        if self.config.speech_encoder_left_chunk_num >= 0:
            start_indices = (chunk_indices - self.config.speech_encoder_left_chunk_num).clamp_(min=0)
            start_indices = start_indices * self.config.speech_encoder_chunk_size
            start_indices = start_indices
        start_indices = start_indices.unsqueeze(1).expand(-1, sequence_len)

        # 根据分块索引计算结束索引
        end_indices = ((chunk_indices + 1) * self.config.speech_encoder_chunk_size).clamp_(max=sequence_len)
        end_indices = end_indices.unsqueeze(1).expand(-1, sequence_len)

        # 创建索引张量
        indices = torch.arange(sequence_len, device=hidden_states.device).unsqueeze(0).expand(sequence_len, -1)

        # 创建分块掩码，用于防止跨分块的注意力
        chunk_mask = (indices < start_indices) | (indices >= end_indices)
        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)

        # 将注意力掩码与输入的注意力掩码进行合并
        attention_mask = chunk_mask if attention_mask is None else (attention_mask.bool() | chunk_mask)
        # 将注意力掩码转换为与隐藏状态张量相同的数据类型
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        return attention_mask

    # 前向传播函数，接受隐藏状态张量等输入，执行模型的前向计算
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ): 
        # 如果不需要输出隐藏状态，则将 all_hidden_states 设为一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则将 all_self_attentions 设为一个空元组
        all_self_attentions = () if output_attentions else None

        # 将 attention_mask 复制给 conv_attention_mask
        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # 确保填充的标记输出 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            # 扩展 attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 如果 speech_encoder_chunk_size 存在，则调用 _apply_chunk_attention 对 attention_mask 进行处理
        if self.config.speech_encoder_chunk_size is not None:
            attention_mask = self._apply_chunk_attention(attention_mask, hidden_states)

        # 将 attention_mask 扩展至与 hidden_states 相同类型
        if attention_mask is not None:
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min

        # 对 hidden_states 进行 dropout 处理
        hidden_states = self.dropout(hidden_states)

        # 判断是否启用了 deepspeed zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 遍历每个层并进行处理
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则将 hidden_states 加入 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop
            dropout_probability = torch.rand([])
            skip_the_layer = (
                True if self.training and (dropout_probability < self.config.speech_encoder_layerdrop) else False
            )
            # 若 skip_the_layer 为 False 或者 deepspeed_zero3_is_enabled 为 True，则执行下面的操作
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果启用了梯度检查点，且当前处于训练模式，则调用 _gradient_checkpointing_func 处理
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        conv_attention_mask=conv_attention_mask,
                    )
                hidden_states = layer_outputs[0]

            # 如果 skip_the_layer 为 True，则将 layer_outputs 设为 (None, None)
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果需要输出注意力权重，则将当前层的注意力权重加入 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对 hidden_states 进行 layer normalization 处理
        hidden_states = self.layer_norm(hidden_states)
        # 如果需要输出隐藏状态，则将 hidden_states 加入 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 的值返回不同的结果
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
```  
# 从transformer.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapterLayer复制代码，并将SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapterLayer(nn.Module):
    # 初始化函数，接受一个config参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 从config中获取隐藏层大小
        embed_dim = config.hidden_size
        # 从config中获取adaptor_dropout
        dropout = config.adaptor_dropout

        # 从config中获取adaptor_kernel_size和adaptor_stride
        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride

        # 1. residual convolution
        # 初始化残差层规范化层
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        # 初始化残差卷积层
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 初始化激活函数
        self.activation = nn.GLU(dim=1)

        # Self-Attention
        # 初始化自注意力层规范化层
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        # 初始化自注意力卷积层
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        # 初始化自注意力模块
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config, use_position_embeddings=False)
        # 初始化自注意力的dropout层
        self.self_attn_dropout = nn.Dropout(dropout)

        # Feed-forward
        # 初始化前馈层规范化层
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        # 初始化前馈网络模块
        self.ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=dropout)

    # 根据注意力掩码计算子采样长度
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        pad = self.kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        # 根据公式计算子采样长度
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.floor()

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
```py`
        ):
        # 对隐藏状态进行残差层归一化
        residual = self.residual_layer_norm(hidden_states)

        # 对残差进行池化，以匹配多头注意力输出的序列长度
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        # 对隐藏状态进行自注意力层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 在输入多头注意力层之前应用池化
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        if attention_mask is not None:
            # 从注意力掩码计算子采样长度
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                hidden_states.device
            )
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # 其余计算与普通Transformer编码器层相同
        hidden_states, attn_weigths = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states
# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapter复制代码，并将SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 创建一个包含config.num_adapter_layers个SeamlessM4Tv2ConformerAdapterLayer对象的列表
        self.layers = nn.ModuleList(
            SeamlessM4Tv2ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers)
        )

    def forward(self, hidden_states, attention_mask):
        # 如果需要，对hidden_states进行下投影

        # 遍历每个层，对hidden_states进行处理
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


############ TEXT / UNITS related code ################


# 从transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding复制代码
class SeamlessM4Tv2SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # 调用make_weights方法创建权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 获取嵌入权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 在forward方法中将权重放在正确的dtype和设备上
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 注册权重为buffer
        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        # 如果输入的 token ids 不为空
        if input_ids is not None:
            # 获取输入的 batch size 和序列长度
            bsz, seq_len = input_ids.size()
            # 根据输入的 token ids 创建位置 ids。任何填充的 token 仍然保持填充状态。
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            # 获取输入的嵌入张量的 batch size 和序列长度
            bsz, seq_len = inputs_embeds.size()[:-1]
            # 从输入的嵌入张量创建位置 ids
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # 如果需要扩展嵌入张量
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            # 创建新的权重张量，以扩展到 max_pos + self.offset
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 选择对应位置的权重，形成最终的嵌入张量，并将其分离出来
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入张量的形状
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成连续的位置 ids
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 扩展位置 ids，使其与输入嵌入张量的形状一致，并添加 past_key_values_length
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
class SeamlessM4Tv2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 从transformers.models.bart.modeling_bart.BartAttention.__init__复制而来，将Bart->SeamlessM4Tv2
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
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, self.head_dim)
        # 将头部移动到第二个位置 (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        
class SeamlessM4Tv2FeedForwardNetwork(nn.Module):
    def __init__(self, config: SeamlessM4Tv2Config, ffn_dim: int):
        super().__init__()
        # 初始化全连接层
        self.fc1 = nn.Linear(config.hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.activation_dropout)
        self.act = ACT2FN[config.activation_function]
    # 前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用全连接层 fc1 对隐藏状态进行线性变换
        hidden_states = self.fc1(hidden_states)
        # 对线性变换后的隐藏状态应用激活函数
        hidden_states = self.act(hidden_states)
        # 对激活后的隐藏状态应用 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 检查 fc2 的权重是否为张量类型，以及隐藏状态的数据类型是否与 fc2 的权重数据类型不同，并且 fc2 的权重数据类型不是 torch.int8 或 torch.uint8
        if (
            isinstance(self.fc2.weight, torch.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8)
        ):
            # 将隐藏状态转换为与 fc2 的权重相同的数据类型
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        # 使用全连接层 fc2 对隐藏状态进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TEncoderLayer复制代码，并将SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2EncoderLayer(nn.Module):
    def __init__(self, config: SeamlessM4Tv2Config, encoder_ffn_dim=None, encoder_attention_heads=None):
        super().__init__()
        # 如果未提供encoder_ffn_dim，则使用config中的encoder_ffn_dim
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        # 如果未提供encoder_attention_heads，则使用config中的encoder_attention_heads
        encoder_attention_heads = (
            config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        # 创建SeamlessM4Tv2Attention对象
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建SeamlessM4Tv2FeedForwardNetwork对象
        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)

        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
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
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 进行自注意力计算
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoderLayer复制代码，并将SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2DecoderLayer(nn.Module):
    # 初始化方法，接受配置参数和解码器的维度和注意力头数
    def __init__(self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果未提供解码器的维度，则使用配置参数中的解码器维度
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        # 如果未提供解码器的注意力头数，则使用配置参数中的注意力头数
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )

        # 设置嵌入维度为隐藏状态的维度
        self.embed_dim = config.hidden_size
        # 创建自注意力层
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 设置丢弃率
        self.dropout = config.dropout
        # 设置激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置注意力丢弃层
        self.attn_dropout = nn.Dropout(config.dropout)

        # 设置自注意力层的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建交叉注意力层
        self.cross_attention = SeamlessM4Tv2Attention(
            self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        # 设置交叉注意力层的 LayerNorm
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建前馈神经网络
        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)

        # 设置前馈神经网络的 LayerNorm
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        # 设置前馈神经网络的丢弃层
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
# 定义一个名为SeamlessM4Tv2TextToUnitDecoderLayer的神经网络模块类
class SeamlessM4Tv2TextToUnitDecoderLayer(nn.Module):
    # 初始化函数，接受配置参数config、解码器前馈维度decoder_ffn_dim和解码器注意力头数decoder_attention_heads
    def __init__(self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None):
        # 调用父类的初始化函数
        super().__init__()
        # 如果decoder_ffn_dim为None，则使用config中的decoder_ffn_dim，否则使用传入的decoder_ffn_dim
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        # 如果decoder_attention_heads为None，则使用config中的decoder_attention_heads，否则使用传入的decoder_attention_heads
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )
        # 将config中的dropout赋值给self.dropout
        self.dropout = config.dropout
        # 将config中的hidden_size赋值给self.embed_dim
        self.embed_dim = config.hidden_size

        # 创建一个自注意力层对象，传入embed_dim、decoder_attention_heads、config中的attention_dropout和is_decoder=True
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 创建一个LayerNorm对象，对self.embed_dim进行LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建一个一维卷积层对象，输入和输出维度都是self.embed_dim，卷积核大小为7，步长为1，填充方式为"same"
        self.conv1 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=7, stride=1, padding="same")
        # 根据配置中的激活函数选择对应的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 创建第二个一维卷积层对象，输入和输出维度都是self.embed_dim，卷积核大小为7，步长为1，填充方式为"same"
        self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=7, stride=1, padding="same")

        # 创建一个LayerNorm对象，对config.hidden_size进行LayerNorm
        self.conv_layer_norm = nn.LayerNorm(config.hidden_size)
        # 创建一个Dropout对象，丢弃概率为self.dropout
        self.conv_dropout = nn.Dropout(self.dropout)

    # 前向传播函数，接受hidden_states、attention_mask、padding_mask和output_attentions等参数
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
                输入到层的形状为`(batch, seq_len, embed_dim)`的张量
            attention_mask (`torch.FloatTensor`):
                大小为`(batch, 1, tgt_len, src_len)`的注意力掩码，其中填充元素由非常大的负值表示
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                指示由于填充而应被忽略的输入，其中元素为1表示*未被掩盖*，为0表示*被掩盖*
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的`attentions`
        """
        residual = hidden_states

        # 自注意力
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 卷积
        residual = hidden_states

        # 应用填充掩码以避免在卷积层中泄漏填充位置
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
# 定义一个继承自PreTrainedModel的抽象类，用于处理权重初始化和预训练模型的下载和加载
class SeamlessM4Tv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类
    config_class = SeamlessM4Tv2Config
    # 基础模型前缀
    base_model_prefix = "seamless_m4t_v2"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = [
        "SeamlessM4Tv2EncoderLayer",
        "SeamlessM4Tv2DecoderLayer",
        "SeamlessM4Tv2ConformerEncoderLayer",
        "SeamlessM4Tv2TextToUnitDecoderLayer",
    ]

    # 初始化权重的方法
    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SeamlessM4Tv2ConformerSelfAttention):
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, SeamlessM4Tv2ConformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # 从注意力掩码计算子采样长度的方法
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TPreTrainedModel._compute_sub_sample_lengths_from_attention_mask复制而来
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        kernel_size, stride = self.config.adaptor_kernel_size, self.config.adaptor_stride
        pad = kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

        return seq_lens.floor()
    # 将输入的 id 转换为对应的文本字符串
    def _indices_to_subwords(self, input_ids):
        """
        Returns the corresponding text string for each input id.
        """
        # 检查生成配置中是否存在 id_to_text 键，用于将 token id 映射到子词
        if not hasattr(self.generation_config, "id_to_text"):
            raise ValueError(
                """This model generation config doesn't have a `id_to_text` key which maps
                token ids to subwords. Make sure to load the right generation config."""
            )
        # 获取输入 id 的批量大小和序列长度
        batch_size, sequence_len = input_ids.shape

        # 存储子词的列表
        subwords_batch = []
        # 遍历每个批次
        for batch_id in range(batch_size):
            subwords = []
            # 遍历每个序列位置
            for i in range(sequence_len):
                # 获取对应 id 的子词
                subword = self.generation_config.id_to_text.get(str(input_ids[batch_id, i].item()))
                subwords.append(str(subword))
            subwords_batch.append(subwords)
        return subwords_batch

    # 计算子词中字符的长度
    def _count_character_length_in_subword(
        self,
        input_ids,
        subwords_batch,
        merge_space_with_prev_subword=False,
        pad_token_id=0,
        unk_token_id=1,
        space="▁",
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
        # 检查是否存在字符到字符id的映射，如果不存在则抛出异常
        if not hasattr(self.generation_config, "char_to_id"):
            raise ValueError(
                """This model generation config doesn't have a `char_to_id` key which maps
                characters to character ids. Make sure to load the right generation config."""
            )

        # 获取输入的批次大小和最大长度
        batch_size = input_ids.shape[0]
        max_len = int(char_count_per_id.sum(1).max().item())

        # 创建一个全零张量用于存储字符id序列
        char_seqs = input_ids.new_zeros((batch_size, max_len)).fill_(pad_token_id)

        # 计算每个子词的长度
        subword_lens = input_ids.ne(pad_token_id).sum(1)

        # 遍历每个批次
        for batch_id in range(batch_size):
            total = 0
            # 获取当前批次的子词索引和子词列表
            subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][: subword_lens[batch_id]]
            # 遍历每个子词及其索引
            for subword_idx, subword in zip(subword_indices, subwords):
                if subword_idx == unk_token_id:
                    char_ids = [unk_token_id]
                else:
                    # 获取与子词对应的字符标记索引
                    char_ids = [self.generation_config.char_to_id.get(ch, unk_token_id) for ch in list(subword)]
                char_seq_len = len(char_ids)
                # 将字符id转换为张量并填充到字符序列张量中
                char_seqs[batch_id, total : total + char_seq_len] = torch.tensor(char_ids).to(char_seqs)
                total += char_seq_len
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
            # 如果隐藏状态的批量大小为1，则使用torch.repeat_interleave函数重复隐藏状态的时间维度
            if hidden_states.size(0) == 1:
                hidden_states = torch.repeat_interleave(hidden_states, durations.view(-1), dim=1)
            else:
                # 如果是批量样本，需要对每个样本进行交错排列，并进行填充 -> 会导致并行性丢失
                if hidden_states.shape[0] > 1 and self.training:
                    logger.warning_once(
                        """`self.training=True` and you use batching. You lose parallelism during the hifigan
                                   forward pass because the samples are interleaved."""
                    )
                # 对隐藏状态和持续时间进行交错排列，然后对隐藏状态进行填充
                hidden_states = [
                    torch.repeat_interleave(hidden_state, duration, dim=0)
                    for (hidden_state, duration) in zip(hidden_states, durations)
                ]

                hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)

            return hidden_states
# 导入模块
@add_start_docstrings(
    """Transformer speech encoder consisting of *config.speech_encoder_layers* conformer self attention layers.
    Each layer is a [`SeamlessM4Tv2ConformerEncoderLayer`].""",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
# 定义新类 SeamlessM4Tv2SpeechEncoder，继承自 SeamlessM4Tv2PreTrainedModel 类
class SeamlessM4Tv2SpeechEncoder(SeamlessM4Tv2PreTrainedModel):
    # 主要输入名称
    main_input_name = "input_features"

    # 初始化方法
    def __init__(self, config: SeamlessM4Tv2Config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 特征投影层
        self.feature_projection = SeamlessM4Tv2ConformerFeatureProjection(config)
        # 编码器层
        self.encoder = SeamlessM4Tv2ConformerEncoder(config)
        # 中间的前馈神经网络
        self.intermediate_ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=0.0)
        # 适配器
        self.adapter = SeamlessM4Tv2ConformerAdapter(config) if config.add_adapter else None
        # 内部层归一化
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 确定是否输出注意力信息
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态信息
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果输入特征为空，则引发错误
        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4Tv2SpeechEncoder.forward`.
                Make sure one of them is not `None`."""
            )

        # 特征投影
        hidden_states = self.feature_projection(input_features)

        # 编码器
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 更新隐藏状态
        hidden_states = encoder_outputs[0]

        # 扩展隐藏状态
        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

        # 如果存在适配器，则应用适配器
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        # 内部层归一化
        hidden_states = self.inner_layer_norm(hidden_states)

        # 如果不返回字典，则返回元组形式的结果
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        # 返回字典形式的结果
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 根据 MBart 和 NllbMoe 的启发
# 添加起始文档字符串
@add_start_docstrings(
    "Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`SeamlessM4Tv2EncoderLayer`].",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional`):
            输入嵌入
        is_t2u_encoder (`bool`, *optional*，默认为 `False`):
            指示是否属于文本到单元模型，若是，它将没有输入嵌入
    """,
)
# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TEncoder复制到SeamlessM4Tv2
class SeamlessM4Tv2Encoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
        is_t2u_encoder: bool = False,
    ):
        # 调用父类初始化函数
        super().__init__(config)

        # 初始化一些参数
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_dim = config.hidden_size

        self.is_t2u_encoder = is_t2u_encoder
        self.max_source_positions = config.max_position_embeddings

        # 如果不是t2u_encoder，则进行下列操作
        if not self.is_t2u_encoder:
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight

            self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
            )

        layers = []
        # 根据encoder_layers个数，初始化多个SeamlessM4Tv2EncoderLayer，并添加到layers列表中
        for _ in range(config.encoder_layers):
            layers.append(
                SeamlessM4Tv2EncoderLayer(
                    config,
                    encoder_attention_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                )
            )

        self.layers = nn.ModuleList(layers)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
        # 添加起始文档字符串
@add_start_docstrings(
    "Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SeamlessM4Tv2DecoderLayer`].",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional`):
            输入嵌入
    """,
)
# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoder中复制代码，并将其命名为SeamlessM4Tv2Decoder
class SeamlessM4Tv2Decoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        # 调用父类构造函数
        super().__init__(config)
        # 初始化dropout
        self.dropout = config.dropout
        # 初始化layerdrop
        self.layerdrop = config.decoder_layerdrop
        # 初始化padding_idx
        self.padding_idx = config.pad_token_id
        # 初始化vocab_size
        self.vocab_size = config.vocab_size
        # 初始化max_target_positions
        self.max_target_positions = config.max_position_embeddings
        # 根据配置初始化embed_scale
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # 如果embed_tokens被定义，使用其形状
            self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight
        else:
            # 否则初始化embed_tokens
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        # 初始化embed_positions
        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        layers = []
        for _ in range(config.decoder_layers):
            layers.append(
                # 初始化config.decoder_layers个SeamlessM4Tv2DecoderLayer
                SeamlessM4Tv2DecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )
        # 初始化layers为ModuleList
        self.layers = nn.ModuleList(layers)
        # 初始化layer_norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回embed_tokens
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # 设置embed_tokens为value
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
    ):
        pass

# 添加说明文档字符串
@add_start_docstrings(
    "Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SeamlessM4Tv2DecoderLayer`].",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
    """,
)
class SeamlessM4Tv2TextToUnitDecoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        # 调用父类的构造函数初始化对象
        super().__init__(config)
        # 设置对象的属性
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 如果提供了 embed_tokens，则使用其形状来初始化 embed_tokens 属性
        if embed_tokens is not None:
            self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight
        else:
            # 否则，使用 config 中的参数初始化 embed_tokens 属性
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        # 初始化字符级别的嵌入层
        self.embed_char = nn.Embedding(config.char_vocab_size, config.hidden_size)
        # 初始化字符级别的位置编码
        self.embed_char_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        # 初始化位置嵌入层的权重
        self.pos_emb_alpha_char = nn.Parameter(torch.ones(1))
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        # 初始化持续时间预测器
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

        # 初始化解码器层
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

        # 设置是否使用渐变检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数
    def forward(
        self,
        char_input_ids: torch.LongTensor = None,
        char_count_per_id: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加模型文档字符串，并设置初始参数
@add_start_docstrings(
    "Transformer bare text-to-unit encoder-decoder. The encoder is a [`SeamlessM4Tv2Encoder`] without embeddings and the decoder is a [`SeamlessM4Tv2TextToUnitDecoder`].",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
class SeamlessM4Tv2TextToUnitModel(SeamlessM4Tv2PreTrainedModel):
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitModel.__init__中复制，并进行修改
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        super().__init__(config)

        # 初始化编码器为SeamlessM4Tv2Encoder对象，解码器为SeamlessM4Tv2TextToUnitDecoder对象
        self.encoder = SeamlessM4Tv2Encoder(config, is_t2u_encoder=True)
        self.decoder = SeamlessM4Tv2TextToUnitDecoder(config, embed_tokens_decoder)

        # 初始化权重并应用最终处理
        self.post_init()

    # 向前传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        char_input_ids: torch.LongTensor = None,
        char_count_per_id: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义一个方法，其输入参数是输入张量、是否输出注意力、是否输出隐藏状态、是否使用返回字典
    # 方法的返回类型是一个元组，包含一个张量或者 Seq2SeqModelOutput 对象
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        # 如果未指定是否输出注意力，则使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定是否输出隐藏状态，则使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定是否返回字典，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果没有传入编码器输出，则调用编码器进行编码
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # 如果 return_dict=True 且编码器输出不是 BaseModelOutput 对象，则将其包装在 BaseModelOutput 中
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # 调用解码器进行解码，返回解码器的输出
        decoder_outputs = self.decoder(
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            encoder_hidden_states=encoder_outputs[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不返回字典，则直接返回解码器输出和编码器输出
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 如果返回字典，则构建 SeamlessM4Tv2TextToUnitOutput 对象，并返回
        return SeamlessM4Tv2TextToUnitOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            padding_mask=decoder_outputs.padding_mask,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
```  
# 导入必要的库和模块
@add_start_docstrings(
    "Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a [`SeamlessM4Tv2TextToUnitModel`].",
    SEAMLESS_M4T_V2_START_DOCSTRING,
    """
        embed_tokens_decoder (`nn.Embedding`, *optional*): input embedding of the decoder.
    """,
)
# 定义一个新的类继承自SeamlessM4Tv2PreTrainedModel类
class SeamlessM4Tv2TextToUnitForConditionalGeneration(SeamlessM4Tv2PreTrainedModel):
    # 定义属性_keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_missing = [
        "vocoder",
        "speech_encoder",
        "text_encoder",
        "text_decoder",
    ]
    # 定义属性_tied_weights_keys
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    # 定义初始化方法，传入config和embed_tokens_decoder参数
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        # 深拷贝config，用于修改bos_token_id等参数
        config = copy.deepcopy(config)
        # 遍历config的每一个属性
        for param, val in config.to_dict().items():
            # 如果属性名以"t2u_"开头
            if param.startswith("t2u_"):
                # 修改config对应属性的值
                config.__setattr__(param[4:], val)
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建SeamlessM4Tv2TextToUnitModel对象
        self.model = SeamlessM4Tv2TextToUnitModel(config, embed_tokens_decoder)

        # 创建nn.Linear对象用于LM头部
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

    # 设置输出嵌入层新的嵌入层对象
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层新的嵌入层对象
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 将M4T_TEXT_TO_UNITS_INPUTS_DOCSTRING添加到model_forward
    @add_start_docstrings_to_model_forward(M4T_TEXT_TO_UNITS_INPUTS_DOCSTRING)
    # 对输入的文本进行前向传播，生成预测结果
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的文本序列，数据类型为 LongTensor，默认为 None
        char_input_ids: torch.LongTensor = None,  # 字符级别的输入文本序列，数据类型为 LongTensor，默认为 None
        char_count_per_id: torch.LongTensor = None,  # 每个输入文本序列对应的字符数量，数据类型为 LongTensor，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，数据类型为 torch.Tensor，可选参数，默认为 None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器输出，数据类型为元组，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，数据类型为 torch.FloatTensor，默认为 None
        labels: Optional[torch.LongTensor] = None,  # 标签，数据类型为 LongTensor，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，数据类型为 bool，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，数据类型为 bool，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，数据类型为 bool，默认为 None
        **kwargs,  # 其他参数，以字典形式传入
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:  # 返回值的类型注释
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果 return_dict 为 None，则使用 self.config.use_return_dict

        # 使用模型进行前向传播
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
        # 获取语言模型预测值
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典形式的结果
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回字典格式的结果
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

    # 从预训练模型中复制函数 _tie_weights
    def _tie_weights(self) -> None:
        # 如果配置参数中设置了 tie_word_embeddings 为 True
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:  # 如果存在输出嵌入表示
                # 将输出嵌入表示与输入嵌入表示进行权重共享
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
# VOCODER 相关代码

# HIFIGAN_START_DOCSTRING 是关于 HifiGanResidualBlock 类的文档字符串的起始部分，提供了模型的一般信息和参数说明
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

# 从 speecht5 模型中复制的 HifiGanResidualBlock 类
class HifiGanResidualBlock(nn.Module):
    # HifiGanResidualBlock 类的构造函数
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        # 设置 LeakyReLU 斜率
        self.leaky_relu_slope = leaky_relu_slope

        # 定义一系列卷积层，用于处理输入数据的不同尺度
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
        # 定义一系列卷积层，用于处理输入数据的相同尺度
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

    # 计算卷积层的填充量
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
    # 定义前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 对于每一对卷积层 conv1 和 conv2，在两个列表 self.convs1 和 self.convs2 中进行迭代
        for conv1, conv2 in zip(self.convs1, self.convs2):
            # 保存残差连接的原始隐藏状态
            residual = hidden_states
            # 使用 LeakyReLU 激活函数对隐藏状态进行非线性变换
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 第一个卷积层 conv1 进行卷积操作
            hidden_states = conv1(hidden_states)
            # 再次使用 LeakyReLU 激活函数对隐藏状态进行非线性变换
            hidden
# 定义一个名为SeamlessM4Tv2VariancePredictor的类，继承自nn.Module
class SeamlessM4Tv2VariancePredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, kernel_size, var_pred_dropout):
        super().__init__()

        # 定义一个卷积层，输入维度为embed_dim，输出维度为hidden_dim，卷积核大小为kernel_size，padding方式为"same"
        self.conv1 = nn.Conv1d(
            embed_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding="same",
        )
        # 定义激活函数为ReLU
        self.activation_fuction = nn.ReLU()
        # 定义LayerNorm层，输入维度为hidden_dim
        self.ln1 = nn.LayerNorm(hidden_dim)
        # 定义Dropout层，概率为var_pred_dropout
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        # 定义第二个卷积层，输入维度为hidden_dim，输出维度为hidden_dim，卷积核大小为kernel_size，padding方式为"same"
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding="same",
        )
        # 定义第二个LayerNorm层，输入维度为hidden_dim
        self.ln2 = nn.LayerNorm(hidden_dim)
        # 定义全连接层，输入维度为hidden_dim，输出维度为1
        self.proj = nn.Linear(hidden_dim, 1)

    # 前向传播函数，接受hidden_states和padding_mask作为输入，并返回Tensor
    def forward(self, hidden_states: Tensor, padding_mask: Tensor = None) -> Tensor:
        # Input: B x T x C; Output: B x T
        # 如果存在padding_mask，则将hidden_states中与padding_mask对应位置为False的元素置为0
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv1(hidden_states.transpose(1, 2))  # 将hidden_states进行卷积操作
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)  # 使用激活函数并转置
        hidden_states = self.dropout_module(self.ln1(hidden_states))  # 对结果进行LayerNorm和Dropout操作
        # 如果存在padding_mask，则将hidden_states中与padding_mask对应位置为False的元素置为0
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv2(hidden_states.transpose(1, 2))  # 将hidden_states进行第二次卷积操作
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)  # 使用激活函数并转置
        hidden_states = self.dropout_module(self.ln2(hidden_states))  # 对结果进行第二次LayerNorm和Dropout操作
        return self.proj(hidden_states).squeeze(dim=2)  # 返回全连接层的结果并在维度2上进行压缩

# 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4THifiGan复制得到SeamlessM4Tv2HifiGan类
class SeamlessM4Tv2HifiGan(nn.Module):
    # 初始化函数，接受一个 SeamlessM4Tv2Config 类型的参数 config
    def __init__(self, config: SeamlessM4Tv2Config):
        # 调用父类的初始化函数
        super().__init__()
        # 计算模型输入的维度，包括单位嵌入维度、语言嵌入维度和说话者嵌入维度
        model_in_dim = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim
        # 保存 LeakyReLU 函数的斜率
        self.leaky_relu_slope = config.leaky_relu_slope
        # 记录残差块（Residual Block）的数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        # 记录上采样层的数量
        self.num_upsamples = len(config.upsample_rates)
        # 创建第一个卷积层，作为网络的前置层
        self.conv_pre = nn.Conv1d(
            # 输入通道数为模型输入维度
            model_in_dim,
            # 输出通道数为配置中定义的上采样初始通道数
            config.upsample_initial_channel,
            # 卷积核大小为7
            kernel_size=7,
            # 步长为1
            stride=1,
            # 填充为3
            padding=3,
        )

        # 创建上采样层的 ModuleList，用于存储多个上采样层
        self.upsampler = nn.ModuleList()
        # 遍历配置中的上采样率和卷积核大小，并创建对应的上采样层
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    # 输入通道数为上一个上采样层的输出通道数，初始为上采样初始通道数
                    config.upsample_initial_channel // (2**i),
                    # 输出通道数为上一个上采样层的输出通道数的一半
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    # 卷积核大小为配置中定义的上采样卷积核大小
                    kernel_size=kernel_size,
                    # 上采样率为配置中定义的上采样率
                    stride=upsample_rate,
                    # 填充为卷积核大小减去上采样率的一半
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        # 创建残差块的 ModuleList，用于存储多个残差块
        self.resblocks = nn.ModuleList()
        # 遍历所有的上采样层，并为每个上采样层创建相应数量的残差块
        for i in range(len(self.upsampler)):
            # 计算当前残差块的输入通道数，为上一个上采样层的输出通道数的一半
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            # 遍历配置中定义的残差块卷积核大小和膨胀率，并创建相应的残差块
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                # 创建一个残差块，并添加到残差块的 ModuleList 中
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 创建网络的后置卷积层，用于将特征映射转换为音频信号
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)
```py  
    # 将 log-mel 频谱图转换为语音波形
    def forward(self, input_embeds: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        将 log-mel 频谱图转换为语音波形，支持批量操作。输入一个 log-mel 频谱图会返回一个语音波形。
        输入批量的 log-mel 频谱图会返回一批量的语音波形。
    
        Args:
            spectrogram (`torch.FloatTensor`):
                包含 log-mel 频谱图的张量。可以是批量的，其形状为 `(batch_size, sequence_length, model_in_dim)`，
                或者单个未批量化的，形状为 `(sequence_length, model_in_dim)`。
                其中 `model_in_dim` 是 `config.unit_embed_dim`, `config.lang_embed_dim` 和 `config.spkr_embed_dim` 之和。
    
        Returns:
            `torch.FloatTensor`: 包含语音波形的张量。如果输入的频谱图是批量的，其形状将是 `(batch_size, num_frames,)`。
            如果是单个未批量化的，其形状将是 `(num_frames,)`。
        """
    
        # 通过卷积前处理模块处理输入嵌入
        hidden_states = self.conv_pre(input_embeds)
        
        # 遍历上采样阶段
        for i in range(self.num_upsamples):
            # 应用 Leaky ReLU 激活函数，提供一些负值处理
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            
            # 上采样处理，将频谱图转换为更高的分辨率
            hidden_states = self.upsampler[i](hidden_states)
    
            # 初始化跳跃连接的残差状态
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            
            # 通过多层残差块进行卷积操作
            for j in range(1, self.num_kernels):
                # 将残差块的输出累加到残差状态中
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            
            # 将残差状态除以核数以平均结果
            hidden_states = res_state / self.num_kernels
    
        # 应用 Leaky ReLU 激活函数
        hidden_states = nn.functional.leaky_relu(hidden_states)
        
        # 最终的卷积后处理模块，得到输出状态
        hidden_states = self.conv_post(hidden_states)
        
        # 通过 tanh 激活函数将输出限制在 [-1, 1] 之间
        hidden_states = torch.tanh(hidden_states)
    
        # 移除序列长度维度，因为这会变成 1
        waveform = hidden_states.squeeze(1)
    
        # 返回最终的语音波形
        return waveform
# 为 `SeamlessM4Tv2CodeHifiGan` 类添加文档描述，继承自 `PreTrainedModel`
@add_start_docstrings(
    """Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).""",
    HIFIGAN_START_DOCSTRING,
)
class SeamlessM4Tv2CodeHifiGan(PreTrainedModel):
    # 设置配置类和主要输入名称，并初始化不分割模块的空列表
    config_class = SeamlessM4Tv2Config
    main_input_name = "input_embeds"
    _no_split_modules = []

    # 构造函数
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)

        # 从配置中读取填充符号的标识符
        self.pad_token_id = config.t2u_pad_token_id
        # 获取嵌入维度
        embed_dim = config.unit_embed_dim
        # 获取预测器的卷积核大小
        kernel_size = config.variance_predictor_kernel_size
        # 获取预测器的 dropout 率
        var_pred_dropout = config.var_pred_dropout
        # 初始化持续时间预测器
        self.dur_predictor = SeamlessM4Tv2VariancePredictor(embed_dim, embed_dim, kernel_size, var_pred_dropout)

        # 初始化用于单元、发音人、和语言的嵌入层
        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)

        # 初始化 HiFi-GAN 模块
        self.hifi_gan = SeamlessM4Tv2HifiGan(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 计算经过持续时间预测层后的输出长度
    def _get_dur_output_lengths(self, input_ids, dur_out):
        """
        Computes the output length after the duration layer.
        """
        # 计算非填充符号的数量
        unit_lengths = (input_ids != self.pad_token_id).sum(1)

        # 确保长度在有效范围内，避免边界情况
        unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)

        # 计算累计的持续时间输出
        cumulative_dur_out = torch.cumsum(dur_out, dim=1)
        # 根据累计持续时间和计算出的单元长度获取最终长度
        unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()

        # 返回最终的输出长度
        return unit_lengths

    # 获取 HiFi-GAN 输出的长度
    def _get_output_hifigan_lengths
    # 计算 HifiGAN 卷积层的输出长度的函数
    def _get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the hifigan convolutional layers
        """

        # 计算 1D 卷积层的输出长度，采用了PyTorch文档中的公式
        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            # 1D 卷积层的输出长度公式来自于 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (
                torch.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode="floor") + 1
            )

        # 计算转置卷积层的输出长度
        def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
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

        # 返回输入长度
        return input_lengths

    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan.forward 拷贝而来，将 SeamlessM4T 替换为 SeamlessM4Tv2，将 spkr_id 替换为 speaker_id
    def forward(
        self, input_ids: torch.LongTensor, speaker_id: torch.Tensor, lang_id: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4Tv2TextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            speaker_id (`int`, *optional`):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional`):
                The language id to use as target language for translation.
        """
        # 使用unit_embedding对input_ids进行嵌入并转置，得到形状为(batch_size, sequence_length, embedding_dim)的hidden_states
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)
        # 使用speaker_embedding对speaker_id进行嵌入并转置，得到形状为(1, speaker_embedding_dim, sequence_length)的spkr
        spkr = self.speaker_embedding(speaker_id).transpose(1, 2)
        # 使用language_embedding对lang_id进行嵌入并转置，得到形状为(1, language_embedding_dim, sequence_length)的lang
        lang = self.language_embedding(lang_id).transpose(1, 2)

        # 使用dur_predictor对hidden_states进行转置再预测时长log_dur_pred
        log_dur_pred = self.dur_predictor(hidden_states.transpose(1, 2))
        # 对log_dur_pred进行softmax操作，并取整，然后限制在范围内，获得对应时长dur_out
        dur_out = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1)
        
        # B x C x T
        # 如果hidden_states的样本数为1，则对hidden_states进行重复以匹配dur_out的时长
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            # 如果样本数大于1且处于训练状态，则给出警告信息
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            # 对每个样本的hidden_states和对应的duration进行重复以匹配长度，并转置得到一致的形状
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=-1).transpose(0, 1)
                for (hidden_state, duration) in zip(hidden_states, dur_out)
            ]
            # 对重复后的hidden_states进行填充处理，保证batch间不同长度的一致性，并转置得到统一的形状
            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)

        # 对spkr进行重复以匹配hidden_states的长度
        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        # 对lang进行重复以匹配hidden_states的长度
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        # 将lang、hidden_states和spkr拼接在一起
        hidden_states = torch.cat([lang, hidden_states, spkr], dim=1)

        # 使用hifi_gan进行特征转换，得到对应的hidden_states
        hidden_states = self.hifi_gan(hidden_states)

        # 计算unit_lengths和lengths
        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        lengths = self._get_output_hifigan_lengths(unit_lengths)

        return hidden_states, lengths

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan._init_weights
    def _init_weights(self, module):
        """Initialize the weights."""
        # 初始化线性、1维卷积、1维转置卷积的权重和偏置
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 初始化嵌入层的权重和偏置
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    # 应用权重归一化到预处理卷积层
    def apply_weight_norm(self):
        nn.utils.weight_norm(self.hifi_gan.conv_pre)
        # 对上采样层中的每一层应用权重归一化
        for layer in self.hifi_gan.upsampler:
            nn.utils.weight_norm(layer)
        # 对残差块中的每一层应用权重归一化
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        # 应用权重归一化到后处理卷积层
        nn.utils.weight_norm(self.hifi_gan.conv_post)

    # 移除预处理卷积层的权重归一化
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        # 移除上采样层中每一层的权重归一化
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        # 移除残差块中每一层的权重归一化
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        # 移除后处理卷积层的权重归一化
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)
# 定义一个文本至文本的 SeamlessM4Tv2 模型转换器，可用于 T2TT
# 该类继承自 SeamlessM4Tv2PreTrainedModel 类
class SeamlessM4Tv2ForTextToText(SeamlessM4Tv2PreTrainedModel):
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_missing = ["speech_encoder", "t2u_model", "vocoder"]
    # 主输入名称
    main_input_name = "input_ids"

    # 将权重绑定的关键键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化函数
    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)

        # 公共嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 文本编码器和解码器
        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.text_encoder

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
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

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
        **kwargs,
    # 生成文本的方法，接受输入id、目标语言、生成配置、logits处理器、停止条件、前缀允许的token函数和其他参数
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
        # 如果使用了过去的键值，就截取decoder_input_ids
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 准备生成的输入，返回一个字典类型的数据
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    # 重新排序缓存的方法，接受过去的键值和beam索引
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重排序的过去键值
        reordered_past = ()
        # 遍历每层的过去键值
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们总是相同的
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        # 返回重新排序后的过去键值
        return reordered_past
# 导入相关库
@add_start_docstrings(
    "The speech-to-text SeamlessM4Tv2 Model transformer which can be used for S2TT.",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
# 定义 SeamlessM4Tv2ForSpeechToText 类，继承 SeamlessM4Tv2PreTrainedModel 类
class SeamlessM4Tv2ForSpeechToText(SeamlessM4Tv2PreTrainedModel):
    # 在加载模型时忽略的键列表
    _keys_to_ignore_on_load_missing = ["text_decoder", "t2u_model", "vocoder"]
    # 主输入名称
    main_input_name = "input_features"

    # 需要共享权重的键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 初始化方法，接收一个 SeamlessM4Tv2Config 类型的对象
    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)

        # 创建一个共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 创建一个 SeamlessM4Tv2SpeechEncoder 类实例
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        # 创建一个 SeamlessM4Tv2Decoder 类实例
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        # 创建一个线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重和应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.speech_encoder

    # 获取解码器
    def get_decoder(self):
        return self.text_decoder

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输入的嵌入层
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 设置输入的嵌入层
    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    # 绑定权重
    def _tie_weights(self):
        # 如果需要绑定词嵌入层的权重
        if self.config.tie_word_embeddings:
            # 绑定或克隆权重
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    # 这里需要继续添加模型前向方法的实现
    # 定义了一个名为 forward 的方法，用于模型的前向传播
    def forward(
        self,
        input_features: torch.LongTensor = None,  # 输入特征，类型为 LongTensor，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，类型为可选的 Tensor，默认为 None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的标识符，类型为可选的 LongTensor，默认为 None
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码，类型为可选的 LongTensor，默认为 None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器的输出，类型为可选的元组，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 上下文键值对，类型为可选的元组，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，类型为可选的 FloatTensor，默认为 None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入表示，类型为可选的 FloatTensor，默认为 None
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为可选的 LongTensor，默认为 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为可选的布尔值，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，类型为可选的布尔值，默认为 None
        **kwargs,
    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.generate 复制而来
    def generate(
        self,
        input_features=None,  # 输入特征，默认为 None
        tgt_lang=None,  # 目标语言，默认为 None
        generation_config=None,  # 生成配置，默认为 None
        logits_processor=None,  # 对 logits 进行处理，默认为 None
        stopping_criteria=None,  # 停止条件，默认为 None
        prefix_allowed_tokens_fn=None,  # 前缀允许的标记函数，默认为 None
        synced_gpus=False,  # 是否同步 GPU，默认为 False
        **kwargs,
    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.prepare_inputs_for_generation 复制而来
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的标识符
        past_key_values=None,  # 上下文键值对，默认为 None
        attention_mask=None,  # 注意力掩码，默认为 None
        use_cache=None,  # 是否使用缓存，默认为 None
        encoder_outputs=None,  # 编码器输出，默认为 None
        **kwargs,
    ):
        # 如果使用了过去的键值对，则截断解码器输入的标识符
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，包含了生成时所需的输入
        return {
            "input_ids": None,  # encoder_outputs 已经定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键值对
            "decoder_input_ids": decoder_input_ids,  # 解码器输入的标识符
            "attention_mask": attention_mask,  # 注意力掩码
            "use_cache": use_cache,  # 是否使用缓存
        }

    @staticmethod
    # 从 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText._reorder_cache 复制而来
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()  # 初始化重新排序的过去键值对
        for layer_past in past_key_values:
            # 对每一层的过去键值对重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past  # 返回重新排序后的过去键值对
# SeamlessM4Tv2ForTextToSpeech类


@add_start_docstrings("The text-to-speech SeamlessM4Tv2 Model transformer which can be used for T2ST.", SEAMLESS_M4T_V2_START_DOCSTRING)


该类是SeamlessM4Tv2用于文本到语音（Text-to-Speech）转换的模型，可以用于T2ST（Text-to-Speech Training）。使用add_start_docstrings方法添加了注释。


class SeamlessM4Tv2ForTextToSpeech(SeamlessM4Tv2PreTrainedModel):


继承自SeamlessM4Tv2PreTrainedModel类。SeamlessM4Tv2ForTextToSpeech类是SeamlessM4Tv2模型的子类，用于文本到语音转换。


_keys_to_ignore_on_load_missing = ["speech_encoder"]


定义了一个成员变量_keys_to_ignore_on_load_missing，并初始化为包含字符串"speech_encoder"的列表。


main_input_name = "input_ids"


定义了一个成员变量main_input_name，并初始化为字符串"input_ids"。


_tied_weights_keys = [
    "lm_head.weight",
    "text_encoder.embed_tokens.weight",
    "text_decoder.embed_tokens.weight",
]


定义了一个成员变量_tied_weights_keys，并初始化为包含三个字符串的列表。这些字符串表示权重矩阵的名称。


def __init__(self, config: SeamlessM4Tv2Config):


构造函数，接收一个类型为SeamlessM4Tv2Config的参数config。


super().__init__(config)


调用父类SeamlessM4Tv2PreTrainedModel的构造函数。


self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)


创建一个Embedding层，将config.vocab_size、config.hidden_size和config.pad_token_id作为参数传递。


self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)


创建一个SeamlessM4Tv2Encoder对象，将config和self.shared作为参数传递。


self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)


创建一个SeamlessM4Tv2Decoder对象，将config和self.shared作为参数传递。


self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


创建一个Linear层，将config.hidden_size和config.vocab_size作为参数传递。


self.post_init()


调用post_init()方法进行初始化。


self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)


创建一个SeamlessM4Tv2TextToUnitForConditionalGeneration对象，将config作为参数传递。


self.vocoder = SeamlessM4Tv2CodeHifiGan(config)


创建一个SeamlessM4Tv2CodeHifiGan对象，将config作为参数传递。


def get_encoder(self):


定义了一个get_encoder方法。


return self.text_encoder


返回text_encoder成员变量。


def get_decoder(self):


定义了一个get_decoder方法。


return self.text_decoder


返回text_decoder成员变量。


def get_output_embeddings(self):


定义了一个get_output_embeddings方法。


return self.lm_head


返回lm_head成员变量。


def set_output_embeddings(self, new_embeddings):


定义了一个set_output_embeddings方法。


self.lm_head = new_embeddings


将new_embeddings赋值给lm_head成员变量。


def get_input_embeddings(self):


定义了一个get_input_embeddings方法。


return self.text_decoder.embed_tokens


返回text_decoder.embed_tokens成员变量。


def set_input_embeddings(self, value):


定义了一个set_input_embeddings方法。


self.text_encoder.embed_tokens = value
self.text_decoder.embed_tokens = value
self.shared = value


将value分别赋值给text_encoder.embed_tokens、text_decoder.embed_tokens和shared成员变量。


def _tie_weights(self):


定义了一个_tie_weights方法。


if self.config.tie_word_embeddings:


如果config.tie_word_embeddings为真。


self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
self._tie_or_clone_weights(self.lm_head, self.shared)


调用_tie_or_clone_weights方法，将text_encoder.embed_tokens、text_decoder.embed_tokens和lm_head与shared关联起来。


@add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)


使用add_start_docstrings_to_model_forward方法添加注释。
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.forward中复制而来，将SeamlessM4T->SeamlessM4Tv2
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ID张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 可选的解码器token ID张量，默认为None
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 可选的解码器注意力掩码张量，默认为None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 可选的编码器输出元组，默认为None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 可选的过去的键-值元组，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的输入嵌入张量，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的解码器输入嵌入张量，默认为None
        labels: Optional[torch.LongTensor] = None,  # 可选的标签张量，默认为None
        use_cache: Optional[bool] = None,  # 可选的使用缓存布尔值，默认为None
        output_attentions: Optional[bool] = None,  # 可选的输出注意力布尔值，默认为None
        output_hidden_states: Optional[bool] = None,  # 可选的输出隐藏状态布尔值，默认为None
        return_dict: Optional[bool] = None,  # 可选的返回字典布尔值，默认为None
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 可选的输入token ID张量，默认为None
        return_intermediate_token_ids: Optional[bool] = None,  # 可选的返回中间token ID布尔值，默认为None
        tgt_lang: Optional[str] = None,  # 可选的目标语言字符串，默认为None
        speaker_id: Optional[int] = 0,  # 可选的说话者ID整数，默认为0
        **kwargs,  # 接受任意其他关键字参数
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.prepare_inputs_for_generation中复制而来
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器token ID张量
        past_key_values=None,  # 过去的键-值元组，默认为None
        attention_mask=None,  # 注意力掩码，默认为None
        use_cache=None,  # 使用缓存布尔值，默认为None
        encoder_outputs=None,  # 编码器输出，默认为None
        **kwargs,  # 接受任意其他关键字参数
    ):
        # 如果使用了过去的键-值，对解码器token ID进行裁剪
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键-值
            "decoder_input_ids": decoder_input_ids,  # 解码器token ID
            "attention_mask": attention_mask,  # 注意力掩码
            "use_cache": use_cache,  # 使用缓存
        }

    @staticmethod
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech._reorder_cache中复制而来
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 对于每一层的过去的键-值，对其进行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
# 使用特定的文档字符串为SeamlessM4Tv2ForSpeechToSpeech类添加描述和注释
@add_start_docstrings(
    "The speech-to-speech SeamlessM4Tv2 Model transformer which can be used for S2ST.",
    SEAMLESS_M4T_V2_START_DOCSTRING,
)
class SeamlessM4Tv2ForSpeechToSpeech(SeamlessM4Tv2PreTrainedModel):
    # 在加载丢失的键时要忽略的列表
    _keys_to_ignore_on_load_missing = ["text_encoder"]
    # 主输入名称
    main_input_name = "input_features"

    # 要连接权重的键列表
    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.__init__复制过来，将SeamlessM4T替换为SeamlessM4Tv2
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建共享层
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 创建语音编码器
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        # 创建文本解码器
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        # 创建线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

        # 创建文本到单元模型
        self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)
        # 创建音码器
        self.vocoder = SeamlessM4Tv2CodeHifiGan(config)

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.get_encoder复制而来
    def get_encoder(self):
        return self.speech_encoder

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.get_decoder复制而来
    def get_decoder(self):
        return self.text_decoder

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.get_output_embeddings复制而来
    def get_output_embeddings(self):
        return self.lm_head

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.set_output_embeddings复制而来
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.get_input_embeddings复制而来
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.set_input_embeddings复制而来
    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech._tie_weights复制而来
    def _tie_weights(self):
        # 如果配置要求词嵌入权重相连，则连接或克隆权重
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # 使用特定的文档字符串为model_forward添加描述和注释
    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.forward复制而来，将SeamlessM4T替换为SeamlessM4Tv2
    # 定义一个方法，用于模型的前向传播
    def forward(
        self,
        input_features: torch.LongTensor = None,  # 输入的特征向量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的token ID，默认为None
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码，默认为None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器输出，默认为None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入向量，默认为None
        labels: Optional[torch.LongTensor] = None,  # 标签，默认为None
        use_cache: Optional[bool] = None,  # 是否使用缓存，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
        **kwargs,  # 其他关键字参数
    @torch.no_grad()  # 声明下面的方法不需要梯度
    def generate(  # 定义一个生成方法
        self,
        input_features: Optional[torch.Tensor] = None,  # 输入的特征向量，默认为None
        return_intermediate_token_ids: Optional[bool] = None,  # 是否返回中间token ID，默认为None
        tgt_lang: Optional[str] = None,  # 目标语言，默认为None
        speaker_id: Optional[int] = 0,  # 说话者ID，默认为0
        **kwargs,  # 其他关键字参数
    @staticmethod  # 静态方法装饰器
    # 从模型中复制_reorder_cache方法，用于重新排序缓存
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()  # 初始化重新排序的过去键值对
        for layer_past in past_key_values:  # 遍历过去的键值对
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )  # 将重新排序后的过去状态添加到重新排序过的过去键值对中
        return reordered_past  # 返回重新排序过的过去键值对

    # 从模型中复制prepare_inputs_for_generation方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的token ID
        past_key_values=None,  # 过去的键值对，默认为None
        attention_mask=None,  # 注意力掩码，默认为None
        use_cache=None,  # 是否使用缓存，默认为None
        encoder_outputs=None,  # 编码器输出，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 如果使用了过去的键值对，则截取解码器输入的token ID
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键值对
            "decoder_input_ids": decoder_input_ids,  # 解码器输入的token ID
            "attention_mask": attention_mask,  # 注意力掩码
            "use_cache": use_cache,  # 是否使用缓存
        }  # 返回准备好的输入字典
# 添加模型说明文档，并设置默认的模态参数
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

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.__init__复制而来，用于初始化SeamlessM4Tv2模型
    def __init__(self, config, current_modality="text"):
        
        # 调用父类构造函数
        super().__init__(config)

        # 共享的嵌入层，用于共享模型训练中的嵌入矩阵
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # 文本编码器
        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)
        # 语音编码器
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        # 文本解码器
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        # 线性映射层，用于将模型输出映射到词汇表的维度上
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        # 调用后处理函数
        self.post_init()

        # 设置当前模态，默认为"text"
        self.current_modality = current_modality
        if current_modality == "speech":
            self.main_input_name = "input_features"

        # 这些模型在初始化时已经调用了post_init
        # 根据文本生成单元生成文本
        self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)
        # HiFi-GAN模块，用于声音合成
        self.vocoder = SeamlessM4Tv2CodeHifiGan(config)

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.set_modality复制而来，设置模态
    def set_modality(self, modality="text"):
        if modality == "text":
            self.main_input_name = "input_ids"
            self.current_modality = "text"
        elif modality == "speech":
            self.main_input_name = "input_features"
            self.current_modality = "speech"
        else:
            raise ValueError(f"`modality={modality}` is not a valid modality. It must be `text` or `speech`.")

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.get_encoder复制而来，获取编码器
    def get_encoder(self):
        if self.current_modality == "text":
            return self.text_encoder
        else:
            return self.speech_encoder

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.get_output_embeddings复制而来，获取输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.set_output_embeddings复制而来，设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.get_input_embeddings复制而来
    # 返回文本解码器的嵌入层（embed_tokens）
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens
    
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.set_input_embeddings复制而来
    # 设置模型的输入嵌入层（embed_tokens）
    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value
    
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel._tie_weights复制而来
    # 将权重进行绑定，使得所有相关的嵌入层和lm_head共享权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)
    
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.forward复制而来，将SeamlessM4T换成SeamlessM4Tv2
    # 模型的前向传播，接受多种输入参数，返回模型的输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token序列的ID
        input_features: Optional[torch.FloatTensor] = None,  # 输入的特征矩阵
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码部分的token序列的ID
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码部分的注意力掩码
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器的输出
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 循环解码器的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入向量
        labels: Optional[torch.LongTensor] = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否使用字典作为返回类型
        **kwargs,
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token序列的ID
        input_features: Optional[torch.Tensor] = None,  # 输入的特征矩阵
        return_intermediate_token_ids: Optional[bool] = None,  # 是否返回生成过程中的中间token序列的ID
        tgt_lang: Optional[str] = None,  # 目标语言
        speaker_id: Optional[int] = 0,  # 说话者的ID
        generate_speech: Optional[bool] = True,  # 是否生成语音
        **kwargs,
    # 从transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.prepare_inputs_for_generation复制而来
    # 为生成准备输入参数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的token序列的ID
        past_key_values=None,  # 循环解码器的键值对
        attention_mask=None,  # 注意力掩码
        use_cache=None,  # 是否使用缓存
        encoder_outputs=None,  # 编码器的输出
        **kwargs,
    # 如果 past_key_values 存在，则只保留 decoder_input_ids 的最后一个标记
    # 否则返回完整的 decoder_input_ids
    ):
        # 如果 past_key_values 不为空，截断 decoder_input_ids，只保留最后一列
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回包含解码所需信息的字典
        return {
            "input_ids": None,  # 因为 encoder_outputs 已定义，所以不需要 input_ids
            "encoder_outputs": encoder_outputs,  # 编码器输出数据
            "past_key_values": past_key_values,  # 过去的键值对，用于加速解码
            "decoder_input_ids": decoder_input_ids,  # 解码器的输入 ID
            "attention_mask": attention_mask,  # 用于解码器的注意力掩码
            "use_cache": use_cache,  # 是否使用缓存
        }

    # 定义一个静态方法，用于重新排序缓存
    @staticmethod
    # 复制自 transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel._reorder_cache
    # 这个方法用于在梁搜索(beam search)中重新排序缓存
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空的元组，用于存储重新排序后的缓存数据
        reordered_past = ()
        # 遍历过去每一层的缓存数据
        for layer_past in past_key_values:
            # 对缓存的交叉注意力状态无需重新排序，因为它们总是相同的
            # 对于每一层，重新排序第一和第二个元素，保留后续元素
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        # 返回重新排序后的缓存数据
        return reordered_past
```