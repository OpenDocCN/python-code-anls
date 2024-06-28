# `.\models\blenderbot_small\modeling_blenderbot_small.py`

```py
# coding=utf-8
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BlenderbotSmall model."""


import copy
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_blenderbot_small import BlenderbotSmallConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BlenderbotSmallConfig"


BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/blenderbot_small-90M",
    # See all BlenderbotSmall models at https://huggingface.co/models?filter=blenderbot_small
]


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个新的张量，形状与输入相同，用于存储右移后的输入ids
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入ids的内容向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将decoder起始token id放到每个序列的开头
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果pad_token_id未定义，抛出异常
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.blenderbot.modeling_blenderbot.BlenderbotLearnedPositionalEmbedding with Blenderbot->BlenderbotSmall
class BlenderbotSmallLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """
        `input_ids_shape` is expected to be [bsz x seqlen].
        Forward pass of the model.
        """
        # 从输入的 `input_ids_shape` 中提取 batch size (`bsz`) 和 sequence length (`seq_len`)
        bsz, seq_len = input_ids_shape[:2]
        
        # 根据 `past_key_values_length` 和当前 `seq_len` 创建一个序列，表示位置编码的位置
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        
        # 调用父类的 `forward` 方法，传入位置编码的序列 `positions`，并返回结果
        return super().forward(positions)
# 从transformers.models.bart.modeling_bart.BartAttention复制过来，将Bart替换为BlenderbotSmall
class BlenderbotSmallAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BlenderbotSmallConfig] = None,
    ):
        super().__init__()
        # 初始化模型参数
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头的数量
        self.dropout = dropout  # dropout概率
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.config = config  # BlenderbotSmall配置对象

        # 检查embed_dim是否能被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除 (当前 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否使用因果注意力

        # 线性变换层，用于生成查询、键、值和输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重塑为适合多头注意力的形状
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
        # 实现前向传播逻辑
        pass  # 由于这只是模型定义，前向传播逻辑尚未实现


# 从transformers.models.bart.modeling_bart.BartEncoderLayer复制过来，将Bart替换为BlenderbotSmall，BART替换为BLENDERBOT_SMALL
class BlenderbotSmallEncoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 嵌入维度，从配置中获取

        # 自注意力层及其归一化层
        self.self_attn = BLENDERBOT_SMALL_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout  # dropout概率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout概率
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终的归一化层
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        # 输入的隐藏状态，形状为 `(batch, seq_len, embed_dim)`
        hidden_states: torch.FloatTensor,
        # 注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`，用极大负值表示填充元素
        attention_mask: torch.FloatTensor,
        # 给定层中的注意力头部掩码，形状为 `(encoder_attention_heads,)`
        layer_head_mask: torch.FloatTensor,
        # 是否输出所有注意力层的注意力张量，默认为 `False`
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 将输入的隐藏状态作为残差连接的起点
        residual = hidden_states
        # 调用自注意力机制 `self_attn` 进行处理，获取处理后的隐藏状态、注意力权重及可能的所有注意力层输出
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对处理后的隐藏状态进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与处理后的隐藏状态相加，形成新的隐藏状态
        hidden_states = residual + hidden_states
        # 对新的隐藏状态进行自注意力层的归一化处理
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 将当前隐藏状态作为下一层的残差连接起点
        residual = hidden_states
        # 经过激活函数后的处理
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对处理后的隐藏状态进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 经过第二个线性层 `fc2` 处理
        hidden_states = self.fc2(hidden_states)
        # 对处理后的隐藏状态进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与处理后的隐藏状态相加，形成新的隐藏状态
        hidden_states = residual + hidden_states
        # 对新的隐藏状态进行最终层归一化处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果隐藏状态的数据类型为 `torch.float16` 并且存在无穷大或 NaN 值
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 对隐藏状态进行截断处理，确保数值范围在可接受的范围内
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 输出结果为包含新的隐藏状态的元组
        outputs = (hidden_states,)

        # 如果需要输出所有注意力层的注意力权重
        if output_attentions:
            # 将注意力权重也添加到输出结果中
            outputs += (attn_weights,)

        # 返回输出结果
        return outputs
# 定义一个字典，用于存储针对 BlenderbotSmall 的注意力机制类的映射关系，目前只包含 "eager" 类型
BLENDERBOT_SMALL_ATTENTION_CLASSES = {
    "eager": BlenderbotSmallAttention,
}

# 从 transformers.models.bart.modeling_bart.BartDecoderLayer 复制而来，将 Bart 替换为 BlenderbotSmall，BART 替换为 BLENDERBOT_SMALL
class BlenderbotSmallDecoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 从配置中获取嵌入维度

        # 初始化自注意力层，根据配置选择具体的注意力类，并设置相关参数
        self.self_attn = BLENDERBOT_SMALL_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout  # 设置丢弃率
        self.activation_fn = ACT2FN[config.activation_function]  # 获取激活函数
        self.activation_dropout = config.activation_dropout  # 设置激活函数的丢弃率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化自注意力层的 LayerNorm

        # 初始化编码器注意力层，同样根据配置选择注意力类，并设置参数
        self.encoder_attn = BLENDERBOT_SMALL_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化编码器注意力层的 LayerNorm

        # 初始化第一个全连接层和第二个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化最终的 LayerNorm

    # 前向传播函数，定义模型的计算流程，接受一系列输入参数并返回输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    # 定义一个方法，用于生成模型输入的虚拟数据
    def dummy_inputs(self):
        # 获取配置中的填充标记 ID
        pad_token = self.config.pad_token_id
        # 创建一个张量作为模型输入的示例，包含两个样本的输入序列
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入数据字典，包括注意力掩码和输入 ID
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 生成对应的注意力掩码，标记填充位置为 False
            "input_ids": input_ids,  # 将创建的输入 ID 添加到字典中
            "decoder_input_ids": input_ids,  # 将输入 ID 作为解码器的输入 ID，这里简单地复用编码器的输入
        }
        # 返回包含虚拟输入数据的字典
        return dummy_inputs
# 定义 BlenderbotSmallStartDocstring 常量，包含模型文档字符串，描述模型继承自 PreTrainedModel 类。
BLENDERBOT_SMALL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BlenderbotSmallConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 BlenderbotSmallGenerationExample 常量，包含对话示例的文档字符串，展示如何使用模型进行对话生成。
BLENDERBOT_SMALL_GENERATION_EXAMPLE = r"""
    Conversation example:

    ```
    >>> from transformers import AutoTokenizer, BlenderbotSmallForConditionalGeneration

    >>> mname = "facebook/blenderbot_small-90M"
    >>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
    >>> tokenizer = AutoTokenizer.from_pretrained(mname)
    >>> UTTERANCE = "My friends are cool but they eat too many carbs."
    >>> print("Human: ", UTTERANCE)
    Human:  My friends are cool but they eat too many carbs.

    >>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
    >>> reply_ids = model.generate(**inputs)
    >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
    Bot:  what kind of carbs do they eat? i don't know much about carbs.

    >>> REPLY = "I'm not sure"
    >>> print("Human: ", REPLY)
    Human: I'm not sure

    >>> NEXT_UTTERANCE = (
    ...     "My friends are cool but they eat too many carbs.__end__ __start__what kind of carbs do they eat? "
    ...     "i don't know much about carbs__end__ "
    ...     "__start__ I'm not sure."
    ... )
    >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
    >>> next_reply_ids = model.generate(**inputs)
    >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
    Bot:  they eat a lot of carbs. carbs are high in fat, protein, and fats.
    ```
"""

# 定义 BlenderbotSmallInputsDocstring 常量，目前为空字符串，用于描述模型输入的文档字符串。
BLENDERBOT_SMALL_INPUTS_DOCSTRING = r"""
"""


class BlenderbotSmallEncoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BlenderbotSmallEncoderLayer`].

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        
        # 从配置中获取dropout和encoder层的dropout比例
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        
        # 设置embedding维度和padding索引
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        # 如果配置中设置了scale_embedding，则使用sqrt(embed_dim)作为embedding的缩放因子，否则为1.0
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        
        # 如果提供了embed_tokens，则直接使用，否则创建一个新的embedding
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        
        # 创建学习后的位置embedding
        self.embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # 创建encoder层的ModuleList，包含多个BlenderbotSmallEncoderLayer实例
        self.layers = nn.ModuleList([BlenderbotSmallEncoderLayer(config) for _ in range(config.encoder_layers)])
        # 对embedding层进行LayerNorm处理
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        
        # 设置梯度检查点为False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
# 定义了一个继承自BlenderbotSmallPreTrainedModel的Transformer解码器类，包含多个BlenderbotSmallDecoderLayer层
class BlenderbotSmallDecoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer解码器，由config.decoder_layers个BlenderbotSmallDecoderLayer层组成。

    Args:
        config: BlenderbotSmallConfig的实例，包含模型配置信息
        embed_tokens (nn.Embedding): 输出的嵌入层
    """

    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout  # 从配置中获取dropout率
        self.layerdrop = config.decoder_layerdrop  # 从配置中获取层间dropout率
        self.padding_idx = config.pad_token_id  # 从配置中获取填充token的索引
        self.max_target_positions = config.max_position_embeddings  # 从配置中获取最大目标位置数
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 根据配置决定是否对嵌入进行缩放

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens  # 如果提供了embed_tokens，则使用提供的嵌入层
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)  # 否则创建新的嵌入层

        self.embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )  # 学习得到的位置嵌入层

        self.layers = nn.ModuleList([BlenderbotSmallDecoderLayer(config) for _ in range(config.decoder_layers)])  # 创建多个解码层
        self.layernorm_embedding = nn.LayerNorm(config.d_model)  # 嵌入层的LayerNorm

        self.gradient_checkpointing = False  # 是否使用梯度检查点（暂未启用）

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens  # 获取输入的嵌入层

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入的嵌入层

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 省略了具体的前向传播逻辑，在实际代码中应该包含完整的Transformer解码器的前向传播过程
        pass
    # 返回模型的解码器
    def get_decoder(self):
        return self.decoder

    # 应用装饰器，将 BLENDERBOT_SMALL_INPUTS_DOCSTRING 添加到模型前向传播方法的文档字符串中
    # 使用 replace_return_docstrings 函数，将输出类型设为 Seq2SeqModelOutput，并替换配置类为 _CONFIG_FOR_DOC
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 输入的注意力掩码
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的 token IDs
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力机制的掩码
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器的多头注意力机制掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力机制的掩码
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,  # 编码器的输出
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 用于存储过去的键值对
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入输入
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器嵌入输入
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出结果
# 添加文档字符串描述 BlenderbotSmallForConditionalGeneration 类，它是带有语言建模头部的 BlenderbotSmall 模型，可用于摘要生成。
@add_start_docstrings(
    "The BlenderbotSmall Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class BlenderbotSmallForConditionalGeneration(BlenderbotSmallPreTrainedModel):
    # 设置基础模型前缀为 "model"
    base_model_prefix = "model"
    # 在加载时忽略的关键字列表，缺失时的处理方式
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 指定需要共享权重的键名列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接受 BlenderbotSmallConfig 类型的参数 config
    def __init__(self, config: BlenderbotSmallConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 使用给定的 config 创建 BlenderbotSmallModel 实例，并赋值给 self.model
        self.model = BlenderbotSmallModel(config)
        # 注册一个用于偏置的缓冲张量，形状为 (1, self.model.shared.num_embeddings)，初始化为零
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建一个线性层 lm_head，用于生成最终的输出
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回当前模型的编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 返回当前模型的解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 的大小，返回调整后的新的嵌入层
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的 resize_token_embeddings 方法，返回新的嵌入层
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整 final_logits_bias 的大小以匹配新的嵌入层
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整 final_logits_bias 的大小以匹配新的 token 数量
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取当前 final_logits_bias 的旧 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新 token 数量小于等于旧 token 数量，则直接截取现有的部分作为新的偏置
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 否则，创建额外的零偏置，拼接在现有偏置后面，以扩展偏置大小
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册调整后的 final_logits_bias 为新的偏置
        self.register_buffer("final_logits_bias", new_bias)

    # 返回语言建模头部 lm_head
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入到 lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 将文档字符串添加到模型前向方法，描述输入格式和返回结果
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    # 替换返回值文档字符串为 Seq2SeqLMOutput 类型，使用 _CONFIG_FOR_DOC 配置类
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加结束文档字符串 BLENDERBOT_SMALL_GENERATION_EXAMPLE
    @add_end_docstrings(BLENDERBOT_SMALL_GENERATION_EXAMPLE)
    # 定义一个前向传播函数，用于执行模型的前向推理过程
    def forward(
        self,
        # 输入序列的标识符，通常是一个长整型张量
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，用于指示模型在哪些位置需要进行注意力计算
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器的输入序列标识符，可选参数
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力掩码，指示解码器哪些位置需要注意力计算
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，用于屏蔽特定注意力头部的计算
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，用于解码器屏蔽特定注意力头部的计算
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，用于屏蔽编码器和解码器之间的交叉注意力头部的计算
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，可以是元组或基本模型输出的联合类型
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        # 过去的键值对，用于存储过去计算的注意力权重
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入嵌入，用于直接提供输入的嵌入表示
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器输入嵌入，用于直接提供解码器输入的嵌入表示
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，通常是一个长整型张量，用于模型的监督训练
        labels: Optional[torch.LongTensor] = None,
        # 是否使用缓存，用于控制是否返回缓存项
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重信息
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态信息
        output_hidden_states: Optional[bool] = None,
        # 是否以返回字典形式输出
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Tuple of masked language modeling loss and model outputs if not in `return_dict` mode,
            otherwise a `Seq2SeqLMOutput` containing various model outputs.
        """
        # Determine whether to use the provided `return_dict` or the default from `self.config`
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If labels are provided, adjust `use_cache` and set `decoder_input_ids` if not already provided
        if labels is not None:
            if use_cache:
                # Warn about the deprecated use of `use_cache` when `labels` are provided
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # Shift the `labels` to the right to align with decoder input format
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Pass the input arguments to the underlying model for computation
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Compute language modeling logits and add bias for final logits
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        # If labels are provided, compute the masked language modeling loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Return the appropriate output based on `return_dict` mode
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return a structured `Seq2SeqLMOutput` containing relevant model outputs
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
        # 如果使用了过去的键值（即past_key_values不为None），则根据过去的长度修剪decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值中的第一个元素的长度（过去长度）
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能已经只传递最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                # 如果decoder_input_ids的长度大于过去长度，则移除前缀长度为过去长度
                remove_prefix_length = past_length
            else:
                # 默认的旧行为：仅保留最后一个ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 修剪decoder_input_ids，仅保留从remove_prefix_length到结尾的部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回包含生成所需输入的字典
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能用于调试）
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去键值
        reordered_past = ()
        # 对每一层的过去键值进行重新排序
        for layer_past in past_key_values:
            # 对于每个过去状态，按照beam_idx重新排序（转换为相同设备）
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        # 返回重新排序后的过去键值
        return reordered_past
# 从transformers.models.bart.modeling_bart.BartDecoderWrapper复制并修改为BlenderbotSmallDecoderWrapper
class BlenderbotSmallDecoderWrapper(BlenderbotSmallPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在因果语言模型与EncoderDecoderModel框架结合使用时正确加载预训练检查点。
    """

    def __init__(self, config):
        super().__init__(config)
        # 创建BlenderbotSmallDecoder对象作为该类的decoder属性
        self.decoder = BlenderbotSmallDecoder(config)

    def forward(self, *args, **kwargs):
        # 调用self.decoder的forward方法，将所有参数传递给decoder对象
        return self.decoder(*args, **kwargs)


# 从transformers.models.bart.modeling_bart.BartForCausalLM复制并修改为BlenderbotSmallForCausalLM
class BlenderbotSmallForCausalLM(BlenderbotSmallPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深拷贝配置对象，设置is_decoder为True，is_encoder_decoder为False，并调用父类的初始化方法
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 创建BlenderbotSmallDecoderWrapper对象作为该类的model属性
        self.model = BlenderbotSmallDecoderWrapper(config)

        # 创建一个线性层作为lm_head属性，输出尺寸为config.vocab_size，输入尺寸为config.hidden_size，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回self.model.decoder的embed_tokens属性作为输入嵌入层
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置self.model.decoder的embed_tokens属性为给定的value
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回self.lm_head作为输出嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置self.lm_head为给定的new_embeddings
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置self.model.decoder为给定的decoder对象
        self.model.decoder = decoder

    def get_decoder(self):
        # 返回self.model.decoder属性
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        这个方法定义了模型的前向传播逻辑，支持各种可选参数。
        """
        # 实现在BlenderbotSmallDecoderWrapper对象上的前向传播，将所有参数传递给decoder对象
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          encoder_hidden_states=encoder_hidden_states,
                          encoder_attention_mask=encoder_attention_mask, head_mask=head_mask,
                          cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values,
                          inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                          output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                          return_dict=return_dict)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        """
        准备生成过程的输入，支持各种可选参数。
        """
        # 实现在BlenderbotSmallDecoderWrapper对象上的准备生成输入的逻辑，传递所有参数给decoder对象
        raise NotImplementedError
    ):
        # 如果模型在编码器-解码器模型中作为解码器使用，那么解码器的注意力遮罩将即时创建
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认为旧行为：只保留最终ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        # 第一步，解码器缓存状态为空
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```