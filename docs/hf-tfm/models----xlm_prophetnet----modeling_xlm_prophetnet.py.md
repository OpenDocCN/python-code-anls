# `.\models\xlm_prophetnet\modeling_xlm_prophetnet.py`

```py
# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch XLM-ProphetNet model."""

# 引入必要的库和模块
import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# 引入 PyTorch 库
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm

# 引入激活函数映射和模型输出
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 引入 XLM-ProphetNet 的配置文件
from .configuration_xlm_prophetnet import XLMProphetNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "XLMProphetNetConfig"

# XLM-ProphetNet 的预训练模型列表
XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/xprophetnet-large-wiki100-cased",
    # 查看所有 XLMProphetNet 模型的链接
    # 在 https://huggingface.co/models?filter=xprophetnet
]

# 从 src.transformers.models.prophetnet.modeling_prophetnet.PROPHETNET_START_DOCSTRING 复制的文档字符串，
# 将 ProphetNetConfig 替换为 XLMProphetNetConfig
XLM_PROPHETNET_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
    from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
    file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`XLMProphetNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 从 src.transformers.models.prophetnet.modeling_prophetnet.PROPHETNET_INPUTS_DOCSTRING 复制的文档字符串，
# 将 ProphetNet 替换为 XLMProphetNet
XLM_PROPHETNET_INPUTS_DOCSTRING = r"""
"""
Copied from src.transformers.models.prophetnet.modeling_prophetnet.PROPHETNET_STANDALONE_INPUTS_DOCSTRING with ProphetNet->XLMProphetNet
"""
XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.prophetnet.modeling_prophetnet.softmax
def softmax(hidden_state, dim, onnx_trace=False):
    """
    Applies softmax function along a specific dimension of the input tensor.

    Args:
        hidden_state (torch.Tensor): Input tensor to apply softmax.
        dim (int): Dimension along which softmax will be computed.
        onnx_trace (bool, optional): Whether to trace the operation for ONNX compatibility.

    Returns:
        torch.Tensor: Tensor after applying softmax along the specified dimension.
    """
    if onnx_trace:
        return nn.functional.softmax(hidden_state.float(), dim=dim)
    else:
        return nn.functional.softmax(hidden_state, dim=dim, dtype=torch.float32)


# Copied from transformers.models.prophetnet.modeling_prophetnet.ngram_attention_bias
def ngram_attention_bias(sequence_length, ngram, device, dtype):
    """
    Compute n-gram attention bias tensor for ProphetNet.

    Args:
        sequence_length (int): Length of the input sequence.
        ngram (int): Size of the n-gram.
        device (torch.device): Device on which to allocate the tensors.
        dtype (torch.dtype): Data type of the tensors.

    Returns:
        torch.Tensor: N-gram attention bias tensor of shape (ngram, sequence_length, 2 * sequence_length).
    """
    left_block = (
        torch.ones((ngram, sequence_length, sequence_length), device=device, dtype=dtype) * torch.finfo(dtype).min
    )
    right_block = left_block.detach().clone()
    # create bias
    for stream_idx in range(ngram):
        right_block[stream_idx].fill_diagonal_(0, wrap=False)
        left_block[stream_idx].triu_(-stream_idx + 1)

    left_block[:, :, 0] = 0
    return torch.cat([left_block, right_block], dim=2)
# 计算相对位置桶的函数，用于指定数量的桶、最大距离和相对位置列表
def compute_relative_buckets(num_buckets, max_distance, relative_positions, is_bidirectional=False):
    """
    This function computes individual parts of the relative position buckets. For more detail, see paper.
    """
    # 反转相对位置，用负数表示
    inv_relative_positions = -relative_positions
    # 初始化相对位置桶
    rel_positions_bucket = 0

    # 如果是双向的相对位置计算
    if is_bidirectional:
        # 将桶的数量减半
        num_buckets = num_buckets // 2
        # 根据负相对位置是否小于零，确定其所属桶的索引
        rel_positions_bucket = (
            rel_positions_bucket
            + torch.lt(inv_relative_positions, torch.zeros_like(inv_relative_positions)).int() * num_buckets
        )
        # 取相对位置的绝对值
        inv_relative_positions = torch.abs(inv_relative_positions)
    else:
        # 将负相对位置限制在非负数范围内
        inv_relative_positions = torch.max(inv_relative_positions, torch.zeros_like(inv_relative_positions))

    # 计算精确的最大值
    max_exact = num_buckets // 2
    # 判断是否是小距离的情况
    is_small = torch.lt(inv_relative_positions, max_exact)
    # 如果是大距离，使用对数函数计算其桶索引
    val_if_large = max_exact + torch.log(inv_relative_positions.float() / max_exact) / math.log(
        max_distance / max_exact
    ) * (num_buckets - max_exact)
    # 限制桶索引在合理范围内
    val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1)).int()
    # 根据距离大小判断采用小距离还是大距离计算的结果
    rel_positions_bucket = rel_positions_bucket + torch.where(is_small, inv_relative_positions.int(), val_if_large)
    # 返回相对位置桶
    return rel_positions_bucket


# 从transformers.models.prophetnet.modeling_prophetnet.compute_all_stream_relative_buckets复制而来
# 计算所有流的相对位置桶
def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids):
    """
    This function computes both main and predict relative position buckets. For more detail, see paper.
    """
    # 主流相对位置
    main_stream_relative_positions = position_ids.unsqueeze(1).repeat(1, position_ids.size(-1), 1)
    main_stream_relative_positions = main_stream_relative_positions - position_ids.unsqueeze(-1)

    # 预测流相对位置
    predicting_stream_relative_positions = torch.cat((position_ids - 1, position_ids), dim=-1).unsqueeze(1)
    predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, position_ids.size(-1), 1)
    predicting_stream_relative_positions = predicting_stream_relative_positions - position_ids.unsqueeze(-1)

    # 获取主要和预测位置桶
    main_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, main_stream_relative_positions, is_bidirectional=False
    )
    predict_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, predicting_stream_relative_positions, is_bidirectional=False
    )
    # 返回主流和预测流的相对位置桶
    return main_relative_position_buckets, predict_relative_position_buckets


# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput中复制而来，
# 用于XLMProphetNet的序列到序列语言模型输出
@dataclass
class XLMProphetNetSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_ngram: Optional[torch.FloatTensor] = None
    # 定义了多个可选类型的 Torch 张量元组变量，用于存储模型解码器的各种状态和注意力机制的输出
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义了一个属性方法，用于获取解码器的交叉注意力机制，同时发出未来移除警告
    @property
    def decoder_cross_attentions(self):
        warnings.warn(
            "`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        return self.cross_attentions
@dataclass
# 定义 XLMProphetNetSeq2SeqModelOutput 类，继承自 ModelOutput，用于存储编码器模型的输出结果，包含预先计算的隐藏状态以加速顺序解码。
class XLMProphetNetSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.
    """

    # 最后一个隐藏状态，类型为 torch.FloatTensor
    last_hidden_state: torch.FloatTensor
    # 可选项，最后一个 n-gram 隐藏状态，类型为 torch.FloatTensor
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    # 可选项，过去的键/值对，用于加速顺序解码，类型为 Tuple[torch.FloatTensor]
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，解码器的隐藏状态序列，类型为 Tuple[torch.FloatTensor]
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，解码器的 n-gram 隐藏状态序列，类型为 Tuple[torch.FloatTensor]
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，解码器的注意力权重序列，类型为 Tuple[torch.FloatTensor]
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，解码器的 n-gram 注意力权重序列，类型为 Tuple[torch.FloatTensor]]
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，交叉注意力权重序列，类型为 Tuple[torch.FloatTensor]
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，编码器的最后一个隐藏状态，类型为 torch.FloatTensor
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 可选项，编码器的隐藏状态序列，类型为 Tuple[torch.FloatTensor]]
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，编码器的注意力权重序列，类型为 Tuple[torch.FloatTensor]]
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def decoder_cross_attentions(self):
        # 发出警告，提示 `decoder_cross_attentions` 将被移除，请使用 `cross_attentions` 替代
        warnings.warn(
            "`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        # 返回交叉注意力权重序列 cross_attentions
        return self.cross_attentions


@dataclass
# 定义 XLMProphetNetDecoderModelOutput 类，继承自 ModelOutput，用于存储解码器模型的输出结果。
class XLMProphetNetDecoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """

    # 最后一个隐藏状态，类型为 torch.FloatTensor
    last_hidden_state: torch.FloatTensor
    # 可选项，最后一个 n-gram 隐藏状态，类型为 torch.FloatTensor
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    # 可选项，过去的键/值对，用于加速顺序解码，类型为 Tuple[torch.FloatTensor]
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，隐藏状态序列，类型为 Tuple[torch.FloatTensor]
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，n-gram 隐藏状态序列，类型为 Tuple[torch.FloatTensor]
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，注意力权重序列，类型为 Tuple[torch.FloatTensor]
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，n-gram 注意力权重序列，类型为 Tuple[torch.FloatTensor]
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，交叉注意力权重序列，类型为 Tuple[torch.FloatTensor]
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# 定义 XLMProphetNetDecoderLMOutput 类，继承自 ModelOutput，用于存储解码器语言模型的输出结果。
class XLMProphetNetDecoderLMOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """

    # 可选项，损失值，类型为 torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    # 预测的 logits，类型为 torch.FloatTensor
    logits: torch.FloatTensor = None
    # 可选项，预测的 n-gram logits，类型为 torch.FloatTensor
    logits_ngram: Optional[torch.FloatTensor] = None
    # 可选项，过去的键/值对，用于加速顺序解码，类型为 Tuple[torch.FloatTensor]
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，隐藏状态序列，类型为 Tuple[torch.FloatTensor]
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，n-gram 隐藏状态序列，类型为 Tuple[torch.FloatTensor]
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，注意力权重序列，类型为 Tuple[torch.FloatTensor]
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 可选项，n-gram 注意力权重序列，类型为 Tuple[torch.FloatTensor]
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetPreTrainedModel复制而来，将ProphetNet替换为XLMProphetNet
class XLMProphetNetPreTrainedModel(PreTrainedModel):
    # 配置类为XLMProphetNetConfig
    config_class = XLMProphetNetConfig
    # 基础模型前缀为"prophetnet"
    base_model_prefix = "prophetnet"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重的函数，根据不同类型的module设置不同的初始化方式
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            # 如果存在padding_idx，则将其对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 将输入向右移动的函数，用于decoder端的输入准备
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        # 断言确保decoder_start_token_id已定义，通常设置为pad_token_id
        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In XLMProphetNet it is usually set to the"
            " pad_token_id. See XLMProphetNet docs for more information"
        )

        # 将输入向右移动一位
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # 断言确保pad_token_id已定义，用于替换labels中可能存在的-100值
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        # 断言确保shifted_input_ids中所有值都为非负数
        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetPositionalEmbeddings复制而来，将ProphetNet替换为XLMProphetNet
class XLMProphetNetPositionalEmbeddings(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """

    def __init__(self, config: XLMProphetNetConfig) -> None:
        # 最大长度为config中的max_position_embeddings
        self.max_length = config.max_position_embeddings
        super().__init__(config.max_position_embeddings, config.hidden_size, config.pad_token_id)
    # 定义一个方法 forward，用于模型的前向传播
    def forward(self, inputs_shape, device, attention_mask=None, past_key_values=None, position_ids=None):
        # 断言语句，确保 position_ids 为 None 或者 self.padding_idx 未设置
        assert (position_ids is None) or (
            self.padding_idx is None
        ), "If position_ids is pre-computed then padding_idx should not be set."

        # 如果 position_ids 为 None
        if position_ids is None:
            # 如果 past_key_values 不为 None，则在解码单步时 position_ids 对每个 token 都相同
            if past_key_values is not None:
                # 获取过去键值中的输入 token 数量
                prev_num_input_ids = past_key_values[0][0].shape[2]
                # 计算新的输入 token 数量
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                # 计算新的 position_ids，并将其设为 padding_idx 加上 num_input_ids
                position_ids = torch.ones((1, 1), dtype=torch.long, device=device) * (
                    int(self.padding_idx + num_input_ids)
                )
            else:
                # 如果 attention_mask 为 None，则初始化 attention_mask 为全 1 的张量
                if attention_mask is None:
                    attention_mask = torch.ones(inputs_shape, dtype=torch.long, device=device)

                # 从 input_ids / attention_mask 中获取 position_ids
                position_ids = (
                    torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
                ).long() + self.padding_idx

                # 确保 position_ids 不超过 max_length - 1
                position_ids = position_ids.clamp(0, self.max_length - 1)

        # 调用父类的 forward 方法，并返回其结果以及计算得到的 position_ids
        return super().forward(position_ids), position_ids

    # 定义一个私有方法 _forward，用于调用父类的 forward 方法
    def _forward(self, position_ids):
        return super().forward(position_ids)
# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetAttention with ProphetNet->XLMProphetNet
class XLMProphetNetAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: XLMProphetNetConfig,
        num_attn_heads: int,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.attention_dropout = config.attention_dropout  # 设置注意力(dropout)的概率
        self.dropout = config.dropout  # 设置全连接层(dropout)的概率
        self.num_attn_heads = num_attn_heads  # 设置注意力头的数量
        self.head_dim = hidden_size // num_attn_heads  # 计算每个注意力头的维度

        assert self.head_dim * num_attn_heads == hidden_size, (
            "`config.hidden_size` must be divisible by `config.num_encoder_attention_heads` and"
            " `config.num_decoder_attention_heads`"
        )

        self.key_proj = nn.Linear(hidden_size, hidden_size)  # 创建线性层，用于计算键的投影
        self.value_proj = nn.Linear(hidden_size, hidden_size)  # 创建线性层，用于计算值的投影
        self.query_proj = nn.Linear(hidden_size, hidden_size)  # 创建线性层，用于计算查询的投影

        self.out_proj = nn.Linear(hidden_size, hidden_size)  # 创建线性层，用于输出投影

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()  # 重新形状张量，以便进行多头注意力计算

    def forward(
        self,
        hidden_states,
        key_value_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,



# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetFeedForward with ProphetNet->XLMProphetNet
class XLMProphetNetFeedForward(nn.Module):
    """
    This is the residual two feed-forward layer block based on the original Transformer implementation.
    """

    def __init__(self, config: XLMProphetNetConfig, ffn_dim: int):
        super().__init__()
        self.activation_fn = ACT2FN[config.activation_function]  # 设置激活函数
        self.intermediate = nn.Linear(config.hidden_size, ffn_dim)  # 创建线性层，用于中间变换
        self.output = nn.Linear(ffn_dim, config.hidden_size)  # 创建线性层，用于输出变换
        self.activation_dropout = config.activation_dropout  # 设置激活(dropout)的概率
        self.dropout = config.dropout  # 设置全连接层(dropout)的概率

    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)  # 中间变换
        hidden_states = self.activation_fn(hidden_states)  # 激活函数处理

        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 激活(dropout)
        hidden_states = self.output(hidden_states)  # 输出变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 全连接层(dropout)
        return hidden_states



# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetNgramSelfAttention with ProphetNet->XLMProphetNet
class XLMProphetNetNgramSelfAttention(nn.Module):
    # 初始化方法，接受一个配置对象 config：XLMProphetNetConfig
    def __init__(self, config: XLMProphetNetConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置隐藏层大小为 config 中的 hidden_size
        self.hidden_size = config.hidden_size

        # 设置桶的数量为 config 中的 num_buckets
        self.num_buckets = config.num_buckets
        # 设置相对最大距离为 config 中的 relative_max_distance
        self.relative_max_distance = config.relative_max_distance
        # 设置注意力头的数量为 config 中的 num_decoder_attention_heads
        self.num_attn_heads = config.num_decoder_attention_heads
        # 设置全连接层的 dropout 率为 config 中的 dropout
        self.dropout = config.dropout
        # 设置注意力机制的 dropout 率为 config 中的 attention_dropout
        self.attention_dropout = config.attention_dropout
        # 设置每个注意力头的维度为 hidden_size / num_attn_heads
        self.head_dim = config.hidden_size // self.num_attn_heads
        # 设置 ngram 参数为 config 中的 ngram

        # 断言条件：确保 hidden_size 能够被 num_attn_heads 整除
        assert (
            self.head_dim * self.num_attn_heads == config.hidden_size
        ), "config.hidden_size must be divisible by num_attn_heads"
        
        # key, value, query 的投影层
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # 输出投影层
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # 相对位置编码嵌入层
        self.relative_pos_embeddings = nn.Linear(config.hidden_size, self.num_buckets * self.num_attn_heads)

        # 用于 ONNX 运行时的标志，默认为 False
        self.onnx_trace = False

    # 将张量 tensor 重新整形为 (batch_size, seq_len, num_attn_heads, head_dim)，并进行转置和连续性处理
    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()

    # 准备模型用于 ONNX 导出时设置 onnx_trace 标志为 True
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask=None,
        layer_head_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
    ):
    # 获取主要相对位置编码嵌入
    def get_main_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
        # input hidden_states [batch_size, sequence_length, hidden_size]
        # input attn_weights [batch_size, num_heads, sequence_length, sequence_length]
        # input position_ids [batch_size, sequence_length] or [1,1]
        batch_size, num_attn_heads, tgt_len, src_len = attn_weights.shape
        # 将注意力权重张量重新调整形状为 [batch_size, num_heads, tgt_len, src_len]
        attn_weights = attn_weights.view(batch_size, num_attn_heads, tgt_len, src_len)
        
        # 如果未提供主要相对位置桶，则计算它们
        if main_relative_position_buckets is None:
            batch_size, sequence_length = hidden_states.shape[:2]
            # 生成相对位置张量，维度为 [batch_size, sequence_length, sequence_length+1]
            relative_positions = (
                torch.arange(1, attn_weights.shape[-1] + 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )
            # 计算相对位置差，并减去位置 ID，形成相对位置差矩阵
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            # 计算主要相对位置桶，用于后续的注意力计算
            main_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        # 计算相对位置编码张量，形状为 [batch_size, sequence_length, num_buckets * num_heads]
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
        # 调整相对位置编码张量的形状为 [batch_size, sequence_length, num_buckets, num_heads]
        rel_pos_embeddings = rel_pos_embeddings.view(
            rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads)
        )
        # 将维度重新排列为 [batch_size, num_heads, sequence_length, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 3, 1, 2)
        # 调整形状为 [batch_size, num_heads, sequence_length, num_buckets * 1]
        rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:3] + (-1,))

        # 将主要相对位置桶扩展到所有头部，形状为 [batch_size * num_heads * sequence_length, sequence_length]
        main_relative_position_buckets = main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
        # 调整形状为 [batch_size * num_heads * sequence_length, sequence_length]，并转换为长整型
        main_relative_position_buckets = main_relative_position_buckets.view(
            -1, main_relative_position_buckets.shape[-1]
        ).long()
        # 调整相对位置编码张量的形状，以匹配相应的主要相对位置桶
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))

        # 使用索引从相对位置编码张量中聚合主要相对位置桶对应的编码
        main_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=1, index=main_relative_position_buckets)
        # 调整形状为 [batch_size, num_heads, tgt_len, num_buckets]
        main_relative_pos_embeddings = main_relative_pos_embeddings.view(batch_size, num_attn_heads, tgt_len, -1)
        # 返回主要相对位置编码张量
        return main_relative_pos_embeddings
    # 定义函数 predict_relative_position_embeddings，接受多个输入参数
    def predict_relative_position_embeddings(
        hidden_states, attn_weights, position_ids, predict_relative_position_buckets=None
    ):
        # 获取 hidden_states 的 batch_size 和 sequence_length 维度大小
        # hidden_states 的形状为 [batch_size, sequence_length, ngram, hidden_size]
        batch_size, sequence_length = hidden_states.shape[0:2]
    
        # 如果 predict_relative_position_buckets 为 None，则计算相对位置
        if predict_relative_position_buckets is None:
            # 获取 attn_weights 的 key_sequence_length 维度大小
            key_sequence_length = attn_weights.shape[-1]
            # 断言检查 position_ids 是否正确，应为 1 2 3 4 5 ... (key_sequence_length - 1)
            assert (
                position_ids[0][0] == key_sequence_length - 1
            ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"
            
            # 创建相对位置张量 relative_positions
            relative_positions = (
                torch.arange(0, key_sequence_length)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )
            
            # 计算相对位置偏移量
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            
            # 使用 compute_relative_buckets 计算预测相对位置的桶
            predict_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )
    
        # 将 hidden_states 的 ngram 维度与 sequence_length 维度互换位置
        hidden_states = hidden_states.transpose(1, 2)
        
        # 计算相对位置嵌入 rel_pos_embeddings
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
    
        # 调整 rel_pos_embeddings 的形状为 [batch_size, ngram, sequence_length, num_buckets, num_heads]
        rel_pos_embeddings = rel_pos_embeddings.view(
            hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads)
        )
        
        # 将 rel_pos_embeddings 的维度顺序重新排列为 [batch_size, ngram, sequence_length, num_heads, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 2, 1, 4, 3)
        
        # 将 rel_pos_embeddings 展开为二维张量 [batch_size * ngram * sequence_length * num_heads, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, self.num_buckets)
        
        # 将 predict_relative_position_buckets 在第 0 维度上增加一个维度
        predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0)
        
        # 在第 0 维度上重复 predict_relative_position_buckets self.ngram 次，
        # 在第 1 维度上重复 batch_size 次，第 2 维度上重复 num_attn_heads 次，最后一维度不变
        predict_relative_position_buckets = predict_relative_position_buckets.repeat(
            self.ngram, 1, self.num_attn_heads, 1
        )
        
        # 将 predict_relative_position_buckets 重塑为二维张量 [ngram * batch_size * num_heads * sequence_length, -1]
        predict_relative_position_buckets = predict_relative_position_buckets.view(
            -1, predict_relative_position_buckets.size(-1)
        ).long()
    
        # 使用 torch.gather 根据 predict_relative_position_buckets 从 rel_pos_embeddings 中获取预测的相对位置嵌入
        predict_relative_pos_embeddings = torch.gather(
            rel_pos_embeddings, dim=1, index=predict_relative_position_buckets
        )
    
        # 将预测的相对位置嵌入 predict_relative_pos_embeddings 重新调整为形状 [batch_size, gram, num_heads, sequence_length, -1]
        predict_relative_pos_embeddings = predict_relative_pos_embeddings.view(
            batch_size, self.ngram, self.num_attn_heads, sequence_length, -1
        )
    
        # 返回预测的相对位置嵌入 predict_relative_pos_embeddings
        return predict_relative_pos_embeddings
# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetEncoderLayer with ProphetNet->XLMProphetNet, Prophetnet->XLMProphetnet
class XLMProphetNetEncoderLayer(nn.Module):
    """
    Encoder block for XLMProphetnet
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        # 1st residual block
        # 定义自注意力机制层，使用XLMProphetNetAttention模块，配置头数为config.num_encoder_attention_heads
        self.self_attn = XLMProphetNetAttention(config, config.num_encoder_attention_heads)
        # 定义Layer Normalization层，用于自注意力输出的归一化
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 2nd residual block
        # 定义前馈神经网络层，使用XLMProphetNetFeedForward模块，配置隐藏层大小为config.encoder_ffn_dim
        self.feed_forward = XLMProphetNetFeedForward(config, config.encoder_ffn_dim)
        # 定义Layer Normalization层，用于前馈神经网络输出的归一化
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = False,
    ):
        # 1st residual block
        # 执行自注意力机制，获取注意力输出、注意力权重和无用信息，更新隐藏状态
        attention_output, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对注意力输出和原始隐藏状态进行残差连接后，再进行Layer Normalization
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)

        # 2nd residual block
        # 执行前馈神经网络，得到前馈网络的输出
        feed_forward_output = self.feed_forward(hidden_states)
        # 对前馈网络的输出和原始隐藏状态进行残差连接后，再进行Layer Normalization
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        # 组装输出结果
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出结果中

        return outputs


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLayer with Prophetnet->XLMProphetnet, ProphetNet->XLMProphetNet
class XLMProphetNetDecoderLayer(nn.Module):
    """
    Decoder block for XLMProphetnet
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        # 1st residual block
        # 定义N-gram自注意力机制层，使用XLMProphetNetNgramSelfAttention模块
        self.self_attn = XLMProphetNetNgramSelfAttention(config)
        # 定义Layer Normalization层，用于自注意力输出的归一化
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 2nd residual block
        # 如果配置了交叉注意力，定义交叉注意力机制层，使用XLMProphetNetAttention模块，配置头数为config.num_decoder_attention_heads
        if config.add_cross_attention:
            self.cross_attn = XLMProphetNetAttention(config, config.num_decoder_attention_heads)
            # 定义Layer Normalization层，用于交叉注意力输出的归一化
            self.cross_attn_layer_norm = LayerNorm(config.hidden_size)

        # 3rd residual block
        # 定义前馈神经网络层，使用XLMProphetNetFeedForward模块，配置隐藏层大小为config.decoder_ffn_dim
        self.feed_forward = XLMProphetNetFeedForward(config, config.decoder_ffn_dim)
        # 定义Layer Normalization层，用于前馈神经网络输出的归一化
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attn_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ):
        # 1st residual block
        # 执行N-gram自注意力机制，更新隐藏状态
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            extended_predict_attention_mask=extended_predict_attention_mask,
            main_relative_position_buckets=main_relative_position_buckets,
            predict_relative_position_buckets=predict_relative_position_buckets,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 对注意力输出和原始隐藏状态进行残差连接后，再进行Layer Normalization
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)

        # 2nd residual block
        if config.add_cross_attention:
            # 执行交叉注意力机制，获取注意力输出，更新隐藏状态
            cross_attention_output = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=encoder_attn_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                encoder_hidden_states=encoder_hidden_states,
                output_attentions=output_attentions,
            )
            # 对交叉注意力输出和原始隐藏状态进行残差连接后，再进行Layer Normalization
            hidden_states = self.cross_attn_layer_norm(cross_attention_output + hidden_states)

        # 3rd residual block
        # 执行前馈神经网络，得到前馈网络的输出
        feed_forward_output = self.feed_forward(hidden_states)
        # 对前馈网络的输出和原始隐藏状态进行残差连接后，再进行Layer Normalization
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        # 组装输出结果
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_output[-1],)  # 如果需要输出注意力权重，则添加到输出结果中

        return outputs
        ):
            # 1st residual block
            # 如果过去的键/值对存在，则从中获取自注意力缓存的键/值对的前两个位置，否则设为 None
            self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
            # 使用自注意力模型处理隐藏状态，生成 ngram_attention_output 是自注意力输出，self_attn_weights 是自注意力权重，self_attn_weights_ngram 是 ngram 注意力权重，present_key_value 是当前的键/值对
            ngram_attention_output, self_attn_weights, self_attn_weights_ngram, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=self_attn_past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                extended_predict_attention_mask=extended_predict_attention_mask,
                main_relative_position_buckets=main_relative_position_buckets,
                predict_relative_position_buckets=predict_relative_position_buckets,
                position_ids=position_ids,
            )
            # 将自注意力输出与原始隐藏状态相加，并进行 Layer Normalization
            hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

            # 如果过去的键/值对存在，则从中获取交叉注意力缓存的键/值对的后两个位置，否则设为 None
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attn_weights = None
            if encoder_hidden_states is not None:
                # 2nd residual block
                # 如果编码器的隐藏状态存在，则使用交叉注意力模型处理隐藏状态与编码器的键/值状态，生成 attention_output 是交叉注意力输出，cross_attn_weights 是交叉注意力权重，cross_attn_present_key_value 是当前的键/值对
                attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attn_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
                # 将交叉注意力输出与原始隐藏状态相加，并进行 Layer Normalization
                hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)

                # 将交叉注意力的键/值对添加到 present_key_value 中的后两个位置
                present_key_value = present_key_value + cross_attn_present_key_value

            # 3rd residual block
            # 使用前馈神经网络处理隐藏状态，生成 feed_forward_output 是前馈神经网络输出
            feed_forward_output = self.feed_forward(hidden_states)
            # 将前馈神经网络的输出与原始隐藏状态相加，并进行 Layer Normalization
            hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

            # 将最终的隐藏状态作为输出
            outputs = (hidden_states,)

            # 如果需要输出注意力权重，则将自注意力和交叉注意力权重添加到输出中
            if output_attentions:
                outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)

            # 如果需要使用缓存，则将当前的键/值对添加到输出中
            if use_cache:
                outputs += (present_key_value,)

            # 返回最终的输出元组
            return outputs
# 添加起始文档字符串，描述 XLMProphetNetModel 的独立编码器部分
@add_start_docstrings(
    "The standalone encoder part of the XLMProphetNetModel.",
    XLM_PROPHETNET_START_DOCSTRING,
)
# 从 transformers.models.prophetnet.modeling_prophetnet.ProphetNetEncoder 复制而来，做了如下更改：microsoft/prophetnet-large-uncased->patrickvonplaten/xprophetnet-large-uncased-standalone, ProphetNet->XLMProphetNet, PROPHETNET->XLM_PROPHETNET
class XLMProphetNetEncoder(XLMProphetNetPreTrainedModel):
    r"""
    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`XLMProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    """

    def __init__(self, config: XLMProphetNetConfig, word_embeddings: nn.Embedding = None):
        super().__init__(config)

        # 初始化词嵌入，如果未提供则随机初始化，并设置填充索引
        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )
        # 初始化位置嵌入
        self.position_embeddings = XLMProphetNetPositionalEmbeddings(config)
        # 初始化嵌入层的 LayerNorm
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        # 创建编码器层列表，每层都是 XLMProphetNetEncoderLayer 类的实例
        self.layers = nn.ModuleList([XLMProphetNetEncoderLayer(config) for _ in range(config.num_encoder_layers)])

        # 是否使用梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入的词嵌入
        return self.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入的词嵌入
        self.word_embeddings = value

    # 添加起始文档字符串到模型的 forward 方法，提供 XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING 描述
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 使用给定的配置和可选的词嵌入初始化模型
    def __init__(self, config: XLMProphetNetConfig, word_embeddings: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 从配置中获取参数并设置为对象属性
        self.ngram = config.ngram  # ngram 参数
        self.num_buckets = config.num_buckets  # 桶的数量
        self.relative_max_distance = config.relative_max_distance  # 相对最大距离
        self.dropout = config.dropout  # dropout 比率
        self.max_target_positions = config.max_position_embeddings  # 最大目标位置数

        # 如果提供了词嵌入，则使用提供的；否则创建一个新的词嵌入对象
        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )
        # 创建位置嵌入对象
        self.position_embeddings = XLMProphetNetPositionalEmbeddings(config)

        # 创建 ngram 嵌入对象
        self.ngram_embeddings = nn.Embedding(self.ngram, config.hidden_size, None)
        # 创建多个解码层，并组成一个模块列表
        self.layers = nn.ModuleList([XLMProphetNetDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        # 创建用于层归一化的对象
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
        # 执行初始化权重和最终处理步骤
        self.post_init()

    # 返回模型的输入词嵌入
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置模型的输入词嵌入
    def set_input_embeddings(self, value):
        self.word_embeddings = value

    # 前向传播函数，具有详细的文档字符串和输出文档的替换
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XLMProphetNetDecoderModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 计算带缓冲的相对桶
    def compute_buffered_relative_buckets(self, position_ids):
        # 获取批次大小和序列长度
        batch_size, sequence_length = position_ids.shape

        # 创建从1到self.max_target_positions的整数序列，并移到设备上
        position_ids = torch.arange(1, self.max_target_positions).to(position_ids.device).repeat(batch_size, 1)
        
        # 计算主相对桶和预测相对桶
        main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(
            self.num_buckets, self.relative_max_distance, position_ids
        )

        # 缓冲主相对桶
        main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(batch_size, 1, 1)
        
        # 缓冲预测相对桶，包括当前目标位置和扩展的序列长度部分
        predict_relative_buckets = torch.cat(
            [
                predict_relative_buckets[:, :sequence_length, :sequence_length],
                predict_relative_buckets[
                    :, :sequence_length, self.max_target_positions : self.max_target_positions + sequence_length
                ],
            ],
            2,
        ).repeat(batch_size, 1, 1)

        # 返回主相对桶和预测相对桶
        return main_relative_buckets, predict_relative_buckets

    # 准备注意力掩码
    def prepare_attention_mask(self, hidden_states, attention_mask):
        # 获取批次大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]

        # 获取因果掩码，用最小值填充
        causal_mask = torch.full(
            (seq_length, seq_length),
            torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        causal_mask = torch.triu(causal_mask, 1)  # 取上三角部分作为因果掩码

        # 扩展因果掩码以适应批次和注意力头数
        extended_causal_mask = causal_mask[:seq_length, :seq_length][None, None, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + causal_mask.shape
        )

        # 添加常规注意力掩码
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask

        # 将注意力掩码转换为hidden_states的dtype并返回
        return extended_attention_mask.to(hidden_states.dtype)
    # 定义一个方法，准备预测用的注意力掩码
    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        # 获取批次大小和序列长度
        batch_size, seq_length = hidden_states.shape[:2]

        # 获取预测用因果掩码
        predict_causal_mask = ngram_attention_bias(
            self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype
        )
        # 将因果掩码按照特定规则连接起来，以适应预测流的需要
        predict_causal_mask = torch.cat(
            [
                predict_causal_mask[:, :seq_length, :seq_length],
                predict_causal_mask[
                    :, :seq_length, self.max_target_positions : self.max_target_positions + seq_length
                ],
            ],
            dim=-1,
        )
        # 扩展因果掩码以适应批次和注意力头数目
        extended_predict_causal_mask = predict_causal_mask[None, None, :, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + predict_causal_mask.shape
        )

        # 添加常规注意力掩码（如果有）
        if attention_mask is not None:
            # 根据注意力掩码生成扩展的注意力掩码，负无穷处保持不变
            extended_attention_mask = (1.0 - attention_mask[:, None, None, None, :]) * torch.finfo(self.dtype).min
            extended_attention_mask = extended_attention_mask.expand(
                (batch_size, self.config.num_decoder_attention_heads, self.ngram, seq_length, seq_length)
            )
            # 预测流的注意力掩码应始终为0，将其连接到扩展的注意力掩码中
            extended_attention_mask = torch.cat(
                [extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1
            )
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            extended_predict_attention_mask = extended_predict_causal_mask

        # 返回最终的扩展预测注意力掩码，转换为隐藏状态的数据类型
        return extended_predict_attention_mask.to(hidden_states.dtype)
# 为 XLMProphetNetModel 类添加文档字符串，描述其作为 XLMProphetNetPreTrainedModel 的子类，以及模型输出原始隐藏状态的特性
@add_start_docstrings(
    "The bare XLMProphetNet Model outputting raw hidden-states without any specific head on top.",
    XLM_PROPHETNET_START_DOCSTRING,
)
# 从 transformers.models.prophetnet.modeling_prophetnet.ProphetNetModel 复制并修改的 XLMProphetNetModel 类
# 原始模型地址由 microsoft/prophetnet-large-uncased 更改为 patrickvonplaten/xprophetnet-large-uncased-standalone，
# 类名由 ProphetNetModel 更改为 XLMProphetNetModel，相关常量和字符串也做相应的修改
class XLMProphetNetModel(XLMProphetNetPreTrainedModel):
    # 指定了 encoder 和 decoder 共享权重的键名列表
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight"]

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__(config)
        # 初始化词嵌入层，使用配置中的词汇大小、隐藏层大小和填充标识符
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # 复制配置以初始化编码器和解码器，确保配置的一致性和独立性
        encoder_config = copy.deepcopy(config)
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        # 初始化编码器，传入编码器配置和词嵌入层
        self.encoder = XLMProphetNetEncoder(encoder_config, self.word_embeddings)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        # 初始化解码器，传入解码器配置和词嵌入层
        self.decoder = XLMProphetNetDecoder(decoder_config, self.word_embeddings)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入词嵌入层
    def set_input_embeddings(self, value):
        self.word_embeddings = value
        self.encoder.word_embeddings = self.word_embeddings
        self.decoder.word_embeddings = self.word_embeddings

    # 绑定编码器和解码器的词嵌入权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.word_embeddings, self.word_embeddings)
            self._tie_or_clone_weights(self.decoder.word_embeddings, self.word_embeddings)

    # 获取编码器对象
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象
    def get_decoder(self):
        return self.decoder

    # 前向传播函数，接受多个输入参数并返回模型输出
    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XLMProphetNetSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    "The XLMProphetNet Model with a language modeling head. Can be used for sequence generation tasks.",

# 定义了一个字符串，描述了带有语言建模头部的 XLMProphetNet 模型，适用于序列生成任务。

    XLM_PROPHETNET_START_DOCSTRING,

# 引用了常量 XLM_PROPHETNET_START_DOCSTRING，可能是用于生成模型文档字符串的起始标记。
# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetForConditionalGeneration with microsoft/prophetnet-large-uncased->patrickvonplaten/xprophetnet-large-uncased-standalone, ProphetNet->XLMProphetNet, PROPHETNET->XLM_PROPHETNET
class XLMProphetNetForConditionalGeneration(XLMProphetNetPreTrainedModel):
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight", "lm_head.weight"]

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__(config)
        self.prophetnet = XLMProphetNetModel(config)  # 初始化ProphetNet模型
        self.padding_idx = config.pad_token_id  # 设置填充索引
        self.disable_ngram_loss = config.disable_ngram_loss  # 禁用N-gram损失

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 初始化线性层用于语言建模头

        # Initialize weights and apply final processing
        self.post_init()  # 调用后续初始化方法

    def get_output_embeddings(self):
        return self.lm_head  # 返回语言建模头的权重

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings  # 设置新的语言建模头权重

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.word_embeddings, self.lm_head)  # 如果需要，则绑定或克隆词嵌入的权重到语言建模头

    def get_input_embeddings(self):
        return self.prophetnet.word_embeddings  # 返回ProphetNet模型的词嵌入层

    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XLMProphetNetSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        """
        此方法实现了XLMProphetNetForConditionalGeneration的前向传播逻辑，接受多个输入参数，并返回模型输出。
        """
        # 实现详细的前向传播逻辑...
    # 计算损失函数，用于模型训练过程中的损失计算
    def _compute_loss(self, logits, labels, ignore_index=-100):
        # 创建与labels相同维度的零张量，用于存储扩展后的目标标签
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)

        # 根据配置参数ngram扩展目标标签，用于计算ngram损失
        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        # 调整logits的维度顺序以便计算损失
        logits = logits.transpose(0, 1).contiguous()
        # 对logits进行log_softmax操作，用于计算负对数似然损失
        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )

        # 计算负对数似然损失
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        # 如果配置中的平滑因子eps大于0，则进行标签平滑处理
        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            eps_i = self.config.eps / lprobs.size(-1)
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        # 返回计算得到的损失
        return loss

    # 生成过程中准备输入，返回用于生成的输入参数字典
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
        # 断言encoder_outputs不为None，确保生成过程中有编码器输出
        assert encoder_outputs is not None, "`encoder_outputs` have to be passed for generation."

        # 如果有过去的键值，将decoder_input_ids限制为最后一个token
        if past_key_values:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        # 返回生成过程所需的参数字典
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    # 根据标签准备解码器输入ids，用于解码器生成过程
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    # 重新排序缓存数据，用于生成过程中的beam搜索
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每一层的缓存数据进行重新排序，以便与beam搜索结果匹配
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past

    # 获取编码器模型
    def get_encoder(self):
        return self.prophetnet.encoder

    # 获取解码器模型
    def get_decoder(self):
        return self.prophetnet.decoder
@add_start_docstrings(
    "The standalone decoder part of the XLMProphetNetModel with a lm head on top. The model can be used for causal"
    " language modeling.",
    XLM_PROPHETNET_START_DOCSTRING,
)
# 定义 XLMProphetNetForCausalLM 类，继承自 XLMProphetNetPreTrainedModel
# 这个类是 XLMProphetNet 模型的独立解码器部分，顶部带有语言建模头
# 可用于因果语言建模。

class XLMProphetNetForCausalLM(XLMProphetNetPreTrainedModel):
    # 静态成员变量，用于指定需要共享权重的层
    _tied_weights_keys = [
        "prophetnet.word_embeddings.weight",
        "prophetnet.decoder.word_embeddings.weight",
        "lm_head.weight",
    ]

    def __init__(self, config: XLMProphetNetConfig):
        # 设置用于条件语言建模的配置
        config = copy.deepcopy(config)
        config.is_decoder = True  # 设置为解码器
        config.is_encoder_decoder = False  # 不是编码解码模型
        super().__init__(config)  # 调用父类构造函数，初始化配置
        self.prophetnet = XLMProphetNetDecoderWrapper(config)  # 初始化 XLMProphetNetDecoderWrapper

        self.padding_idx = config.pad_token_id  # 设置填充符索引
        self.disable_ngram_loss = config.disable_ngram_loss  # 是否禁用 ngram 损失

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 初始化语言建模头，线性层映射到词汇表大小，无偏置

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入层，即 prophetnet 解码器的词嵌入层
        return self.prophetnet.decoder.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层
        self.prophetnet.decoder.word_embeddings = value

    def get_output_embeddings(self):
        # 返回输出嵌入层，即语言建模头
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入层
        self.lm_head = new_embeddings

    def _tie_weights(self):
        # 如果配置要求共享词嵌入权重，则共享解码器词嵌入层和语言建模头的权重
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.decoder.word_embeddings, self.lm_head)

    def set_decoder(self, decoder):
        # 设置解码器
        self.prophetnet.decoder = decoder

    def get_decoder(self):
        # 获取解码器
        return self.prophetnet.decoder

    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XLMProphetNetDecoderLMOutput, config_class=_CONFIG_FOR_DOC)
    # 重写 forward 方法，添加模型输入的文档字符串和输出的类型说明
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 模型前向传播方法，包含多个输入和控制参数

        # 输出是否返回字典格式结果
        return_dict: Optional[bool] = None,


注释：
    # 定义一个方法用于计算损失函数，接收模型预测的logits、真实标签、以及一个忽略索引值（默认为-100）
    def _compute_loss(self, logits, labels, ignore_index=-100):
        # 创建一个与labels相同数据类型和形状的全零张量，填充值为ignore_index，形状为(self.config.ngram, labels.size(0), labels.size(1))
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(ignore_index)

        # 循环创建ngram个维度的标签张量，用于损失计算
        for i in range(self.config.ngram):
            # 如果i大于0并且self.disable_ngram_loss为True，则退出循环
            if i > 0 and self.disable_ngram_loss:
                break
            # 将labels复制到第i维的标签张量中
            expend_targets[i, :, :] = labels

        # 转置logits张量，使其形状变为(序列长度, 批次大小, 类别数)，并保证内存连续性
        logits = logits.transpose(0, 1).contiguous()
        # 对logits进行log_softmax操作，计算对数概率，dim=-1表示沿着最后一个维度进行softmax操作，dtype=torch.float32指定数据类型
        lprobs = nn.functional.log_softmax(
            logits.view(-1, logits.size(-1)),  # 将logits视图展平为二维张量
            dim=-1,
            dtype=torch.float32,
        )

        # 计算负对数似然损失，将lprobs视图展平为一维张量，expend_targets也展平为一维张量，reduction="mean"表示计算均值
        loss = nn.functional.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        # 如果配置参数self.config.eps大于0.0
        if self.config.eps > 0.0:
            # 计算平滑损失，对lprobs在最后一个维度求和并保持维度不变
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            # 获取非遮蔽标记的令牌，即expend_targets不等于ignore_index的元素视图
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            # 根据非遮蔽标记的令牌，重新计算smooth_loss的均值
            smooth_loss = smooth_loss[non_masked_tokens].mean()

            # 计算eps_i，即self.config.eps除以lprobs的最后一个维度的长度
            eps_i = self.config.eps / lprobs.size(-1)
            # 计算最终损失，结合平滑损失和eps_i的影响
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        # 返回计算得到的损失值
        return loss

    # 定义一个方法，准备生成过程中的输入参数，接收input_ids等参数及其它关键字参数
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # 如果注意力掩码为None，则创建一个与input_ids形状相同的全1张量作为注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # 如果past_key_values不为None，则只保留input_ids的最后一个时间步作为输入
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 返回准备好的输入参数字典，包括input_ids、attention_mask、head_mask、past_key_values和use_cache
        # input_ids不需要在这里定义，因为encoder_outputs已经定义了
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    # 从transformers.models.bart.modeling_bart.BartForCausalLM._reorder_cache中复制而来的方法
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序过的过去键值对的元组
        reordered_past = ()
        # 遍历过去的每一层键值对
        for layer_past in past_key_values:
            # 对每个过去状态，根据beam_idx重新排序，并放置到reordered_past中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对
        return reordered_past
# 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderWrapper复制而来，将ProphetNet->XLMProphetNet，prophetnet->XLMProphetNet
class XLMProphetNetDecoderWrapper(XLMProphetNetPreTrainedModel):
    """
    这是一个包装类，使得[`XLMProphetNetForCausalLM`]能够从预训练的XLMProphetNet类正确加载。
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__(config)

        # 初始化词嵌入层，使用给定的词汇表大小、隐藏大小和填充标记ID
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化解码器，传入配置和词嵌入层
        self.decoder = XLMProphetNetDecoder(config, word_embeddings=self.word_embeddings)

        # 初始化权重并应用最终处理
        self.post_init()

    def _tie_weights(self):
        # 将词嵌入层的权重与解码器的输入嵌入层权重绑定
        self._tie_or_clone_weights(self.word_embeddings, self.decoder.get_input_embeddings())

    def forward(self, *args, **kwargs):
        # 前向传播，调用解码器的前向方法
        return self.decoder(*args, **kwargs)
```