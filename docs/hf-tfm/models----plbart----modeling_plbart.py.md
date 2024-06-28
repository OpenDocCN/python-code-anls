# `.\models\plbart\modeling_plbart.py`

```
# coding=utf-8
# Copyright 2022, UCLA NLP, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch PLBART model."""
import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_plbart import PLBartConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "uclanlp/plbart-base"
_CONFIG_FOR_DOC = "PLBartConfig"

PLBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uclanlp/plbart-base",
    "uclanlp/plbart-cs-java",
    "uclanlp/plbart-multi_task-all",
    # See all PLBART models at https://huggingface.co/models?filter=plbart
]


# Copied from transformers.models.mbart.modeling_mbart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    # 复制输入的 input_ids 张量
    prev_output_tokens = input_ids.clone()

    # 如果 pad_token_id 为 None，则抛出值错误
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    
    # 将标签中可能存在的 -100 值替换为 pad_token_id
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    # 计算每个样本中非 pad_token_id 的最后一个 token 的位置索引
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    
    # 获取 decoder_start_tokens
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    
    # 将 input_ids 向右移动一个 token，并用 decoder_start_tokens 包装最后一个非 pad token
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens
# 从transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding复制过来，改为使用PLBart
class PLBartLearnedPositionalEmbedding(nn.Embedding):
    """
    这个模块学习位置编码，最大长度固定。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # 对于PLBart，如果指定了padding_idx，则将embedding id偏移2，并相应调整num_embeddings。其他模型没有这个hack。
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids'的形状预期为[bsz x seqlen]。"""

        bsz, seq_len = input_ids.shape[:2]
        # 根据设备类型和过去键值对的长度，创建位置张量
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


# 从transformers.models.bart.modeling_bart.BartAttention复制过来，改为使用PLBart
class PLBartAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力模块"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PLBartConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除 (得到 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性层，用于查询（q_proj）、键（k_proj）、值（v_proj）和输出（out_proj）的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新塑造张量形状以便多头注意力计算
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
        # 省略了具体的前向传播过程
        pass


# 从transformers.models.bart.modeling_bart.BartEncoderLayer复制过来，改为使用PLBart，BART->PLBART
class PLBartEncoderLayer(nn.Module):
    # 初始化函数，用于创建一个新的编码器层对象
    def __init__(self, config: PLBartConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为配置文件中的模型维度
        self.embed_dim = config.d_model

        # 初始化自注意力机制，根据配置选择不同的实现类
        self.self_attn = PLBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )

        # 初始化自注意力层的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 设置丢弃率为配置文件中定义的丢弃率
        self.dropout = config.dropout

        # 根据配置选择激活函数
        self.activation_fn = ACT2FN[config.activation_function]

        # 设置激活函数的丢弃率
        self.activation_dropout = config.activation_dropout

        # 第一个全连接层，将嵌入维度映射到编码器中的FFN维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)

        # 第二个全连接层，将FFN维度映射回嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)

        # 最终的 LayerNorm 层，用于标准化最终输出
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
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
        # 拷贝输入的隐藏状态作为残差连接的基础
        residual = hidden_states
        # 使用自注意力机制处理隐藏状态，获取处理后的隐藏状态、注意力权重及额外输出
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对处理后的隐藏状态应用丢弃机制
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接：将原始隐藏状态与处理后的隐藏状态相加
        hidden_states = residual + hidden_states
        # 对残差连接后的隐藏状态进行层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 再次使用残差连接的方法处理隐藏状态
        residual = hidden_states
        # 对处理后的隐藏状态应用激活函数和线性变换 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对处理后的隐藏状态应用激活函数的丢弃机制
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 应用第二个线性变换 fc2
        hidden_states = self.fc2(hidden_states)
        # 对处理后的隐藏状态应用丢弃机制
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接：将原始隐藏状态与处理后的隐藏状态相加
        hidden_states = residual + hidden_states
        # 对残差连接后的隐藏状态进行层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果隐藏状态的数据类型为 float16 并且存在无穷大或 NaN 值，则进行值的截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终输出元组
        return outputs
# TODO: Implement attention with SDPA for PLBart.
# 定义了一个字典，用于存储不同实现方式的注意力机制类，"eager"对应的实现类为PLBartAttention
PLBART_ATTENTION_CLASSES = {"eager": PLBartAttention}


# Copied from transformers.models.bart.modeling_bart.BartDecoderLayer with Bart->PLBart, BART->PLBART
# 定义了PLBart解码器的一个层，继承自nn.Module
class PLBartDecoderLayer(nn.Module):
    def __init__(self, config: PLBartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 使用配置中的注意力实现类创建自注意力层
        self.self_attn = PLBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 对自注意力输出进行层归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 使用配置中的注意力实现类创建编码器注意力层
        self.encoder_attn = PLBART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 对编码器注意力输出进行层归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 第一个全连接层和第二个全连接层，用于多头注意力的前馈神经网络
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        # 对最终输出进行层归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，定义了层的计算逻辑
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
    ):
        # 省略了具体的前向传播逻辑，这里应包含对输入数据的处理和层之间的连接


# Copied from transformers.models.bart.modeling_bart.BartClassificationHead with Bart->PLBart
# 用于句子级分类任务的头部
class PLBartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        # 全连接层，将输入维度转换为内部维度
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 输出投影层，将内部维度转换为类别数量的维度
        self.out_proj = nn.Linear(inner_dim, num_classes)

    # 前向传播函数，定义了层的计算逻辑
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# 定义了PLBart预训练模型的基类，继承自PreTrainedModel
class PLBartPreTrainedModel(PreTrainedModel):
    config_class = PLBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表，这些模块在初始化权重时不会进行处理
    _no_split_modules = ["PLBartDecoderLayer", "PLBartEncoderLayer"]
    
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        # 从配置中获取初始化的标准差
        std = self.config.init_std
        # 如果当前模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果模块有偏置项，则将偏置项初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果当前模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果嵌入层有填充索引，则将填充索引位置的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
PLBART_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PLBartConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PLBART_GENERATION_EXAMPLE = r"""
    Mask-filling example:

    ```python
    >>> from transformers import AutoTokenizer, PLBartForConditionalGeneration

    >>> model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-base")
    >>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")

    >>> # en_XX is the language symbol id <LID> for English
    >>> TXT = "<s> Is 0 the <mask> Fibonacci number ? </s> en_XX"
    >>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="pt").input_ids

    >>> logits = model(input_ids).logits
    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    >>> probs = logits[0, masked_index].softmax(dim=0)
    >>> values, predictions = probs.topk(5)

    >>> tokenizer.decode(predictions).split()
    ['first', 'same', 'highest', 'result', 'number']
    ```
"""

PLBART_INPUTS_DOCSTRING = r"""
"""


# Copied from transformers.models.bart.modeling_bart.BartEncoder with Bart->PLBart
class PLBartEncoder(PLBartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PLBartEncoderLayer`].

    Args:
        config: PLBartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化函数，用于初始化模型的各个组件
    def __init__(self, config: PLBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)

        # 从配置中获取 dropout 的设置
        self.dropout = config.dropout
        # 从配置中获取 encoder_layerdrop 的设置
        self.layerdrop = config.encoder_layerdrop

        # 从配置中获取嵌入维度
        embed_dim = config.d_model
        # 从配置中获取填充标记的索引
        self.padding_idx = config.pad_token_id
        # 从配置中获取最大源序列长度
        self.max_source_positions = config.max_position_embeddings
        # 根据配置决定是否对嵌入进行缩放
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 创建词嵌入层，vocab_size 是词汇表大小，embed_dim 是词嵌入维度，padding_idx 是填充标记的索引
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 如果传入了预训练的 embed_tokens，则使用传入的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 创建学习位置编码的对象，max_position_embeddings 是最大位置编码的长度，embed_dim 是嵌入维度
        self.embed_positions = PLBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )

        # 创建多层编码器，每层使用相同的配置参数
        self.layers = nn.ModuleList([PLBartEncoderLayer(config) for _ in range(config.encoder_layers)])

        # 根据配置判断是否使用特定的注意力实现方法
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 创建嵌入层的 LayerNorm 层，用于归一化嵌入层输出
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回嵌入层的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置嵌入层的方法，接受一个新的嵌入层作为参数并赋值给当前嵌入层
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数，接受多个输入参数并返回模型输出
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 从transformers.models.bart.modeling_bart.BartDecoder复制的代码，修改为PLBartDecoder类
class PLBartDecoder(PLBartPreTrainedModel):
    """
    Transformer解码器，由config.decoder_layers层组成。每一层是一个[`PLBartDecoderLayer`]

    Args:
        config: PLBartConfig
        embed_tokens (nn.Embedding): 输出的嵌入层
    """

    def __init__(self, config: PLBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout  # 获取config中的dropout值
        self.layerdrop = config.decoder_layerdrop  # 获取config中的decoder_layerdrop值
        self.padding_idx = config.pad_token_id  # 获取config中的pad_token_id值
        self.max_target_positions = config.max_position_embeddings  # 获取config中的max_position_embeddings值
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 根据config中的scale_embedding决定是否使用嵌入缩放

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)  # 创建词嵌入层

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight  # 如果提供了embed_tokens，则使用提供的权重初始化嵌入层

        self.embed_positions = PLBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )  # 创建位置编码层

        self.layers = nn.ModuleList([PLBartDecoderLayer(config) for _ in range(config.decoder_layers)])  # 创建多层解码器层

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"  # 检查是否使用FlashAttention 2.0实现
        self._use_sdpa = config._attn_implementation == "sdpa"  # 检查是否使用Scaled Dot-Product Attention (SDPA)

        self.layernorm_embedding = nn.LayerNorm(config.d_model)  # 创建层归一化层

        self.gradient_checkpointing = False  # 是否启用梯度检查点

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回输入的嵌入层

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入的嵌入层为给定的值

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def __init__(self, config: PLBartConfig):
        # 调用父类的构造函数，传入配置对象
        super().__init__(config)

        # 从配置对象中获取填充标记索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建一个共享的词嵌入层，将词汇表映射到模型维度，并使用填充标记索引进行填充
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器，共享词嵌入层
        self.encoder = PLBartEncoder(config, self.shared)
        self.decoder = PLBartDecoder(config, self.shared)

        # 初始化模型权重
        self.init_weights()

    def get_input_embeddings(self):
        # 返回共享的输入词嵌入层
        return self.shared

    def set_input_embeddings(self, value):
        # 设置新的输入词嵌入层，并更新编码器和解码器的词嵌入层
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def _tie_weights(self):
        # 如果配置要求共享词嵌入层，则绑定编码器和解码器的词嵌入层权重
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        # 返回编码器
        return self.encoder

    def get_decoder(self):
        # 返回解码器
        return self.decoder

    @add_start_docstrings_to_model_forward(PLBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用自定义的文档字符串修饰符添加模型的开始文档字符串，描述了 PLBART 模型带有语言建模头部，适用于代码到文本、文本到代码和代码到代码的任务。
# 这里引用了 PLBART_START_DOCSTRING 中定义的常量。
@add_start_docstrings(
    "The PLBART Model with a language modeling head. Can be used for code-to-text, text-to-code and code-to-code.",
    PLBART_START_DOCSTRING,
)
# 定义 PLBartForConditionalGeneration 类，继承自 PLBartPreTrainedModel
class PLBartForConditionalGeneration(PLBartPreTrainedModel):
    # 指定基础模型的前缀为 "model"
    base_model_prefix = "model"
    # 在加载过程中忽略的键名列表
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 被绑定权重的键名列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接受一个 PLBartConfig 类型的参数 config
    def __init__(self, config: PLBartConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 PLBartModel 实例并赋值给 self.model
        self.model = PLBartModel(config)
        # 注册一个缓冲区 final_logits_bias，用全零张量填充，形状为 (1, self.model.shared.num_embeddings)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建一个线性层 lm_head，输入维度为 config.d_model，输出维度为 self.model.shared.num_embeddings，没有偏置
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 调用初始化权重的方法
        self.init_weights()

    # 获取编码器部分的方法
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器部分的方法
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 大小的方法，返回新的嵌入层
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的 resize_token_embeddings 方法
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调用 _resize_final_logits_bias 方法，调整 final_logits_bias 的大小以匹配新的 token embeddings
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        # 返回新的嵌入层
        return new_embeddings

    # 调整 final_logits_bias 大小的私有方法，没有返回值
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取旧的 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于旧的 token 数量，则截取 final_logits_bias
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        # 否则，创建额外的零偏置，扩展 final_logits_bias
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册一个缓冲区 final_logits_bias，更新为新的偏置
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出嵌入层 lm_head 的方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层 lm_head 的方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 使用自定义的文档字符串修饰符添加到模型的前向方法，描述了 PLBART_INPUTS_DOCSTRING 定义的输入文档字符串
    # 以及返回类型为 Seq2SeqLMOutput，使用 _CONFIG_FOR_DOC 指定的配置类，并附加 PLBART_GENERATION_EXAMPLE 的结尾文档字符串
    @add_start_docstrings_to_model_forward(PLBART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(PLBART_GENERATION_EXAMPLE)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        # 输入序列的标识符，类型为可选的长整型张量
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力遮罩，类型为可选的长整型张量
        attention_mask: Optional[torch.LongTensor] = None,
        # 解码器的输入序列标识符，类型为可选的长整型张量
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力遮罩，类型为可选的张量
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # 头部遮罩，类型为可选的张量
        head_mask: Optional[torch.Tensor] = None,
        # 解码器的头部遮罩，类型为可选的长整型张量
        decoder_head_mask: Optional[torch.LongTensor] = None,
        # 交叉注意力头部遮罩，类型为可选的张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出的列表，类型为可选的浮点数张量列表
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去的键值对，类型为可选的浮点数张量列表
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入的嵌入向量，类型为可选的浮点数张量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器输入的嵌入向量，类型为可选的浮点数张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，类型为可选的张量
        labels: Optional[torch.Tensor] = None,
        # 是否使用缓存，类型为可选的布尔值
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，类型为可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的结果，类型为可选的布尔值
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Depending on `return_dict`:
            - if `False` (default), returns a tuple with `lm_logits` followed by various model outputs.
            - if `True`, returns a `Seq2SeqLMOutput` object containing loss, logits, and other outputs.

        """
        # Determine whether to use `return_dict` from self.config or override with provided value
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Adjust `decoder_input_ids` if not provided, using shifted `labels` for autoregressive decoding
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        # Forward pass through the model with specified inputs and optional masks and caches
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

        # Compute logits for the language model head and apply a bias if provided
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        # Compute masked language modeling loss if `labels` are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Prepare output depending on whether `return_dict` is `False` or `True`
        if not return_dict:
            # Return a tuple with `lm_logits` followed by other model outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        else:
            # Return a `Seq2SeqLMOutput` object with loss, logits, and various model outputs
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
        decoder_input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        **kwargs,  # TODO: Check if this is needed. It is unused?
    ) -> Dict[str, Any]:
        # 如果使用了过去的键值（past_key_values），则根据其长度裁剪decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认旧行为：只保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回一个字典，包含用于生成的输入参数
        return {
            "input_ids": None,  # encoder_outputs 已经定义。input_ids 不需要
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能用于调试目的）
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签右移一个位置，用于解码器输入
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态无需重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 给 PLBartForSequenceClassification 类添加文档字符串，描述其作为 PLBart 模型的序列分类器及其顶部的线性层用途
@add_start_docstrings(
    """
    PLBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for code
    classification.
    """,
    PLBART_START_DOCSTRING,
)
class PLBartForSequenceClassification(PLBartPreTrainedModel):
    # 定义权重绑定的键列表，用于共享编码和解码器的嵌入层权重
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: PLBartConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 创建 PLBartModel 实例作为主模型
        self.model = PLBartModel(config)
        # 创建 PLBartClassificationHead 实例作为分类器的头部
        self.classification_head = PLBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并进行最终处理
        self.post_init()

    # 给 forward 方法添加文档字符串，描述其输入和输出，引用输入文档和代码示例文档
    @add_start_docstrings_to_model_forward(PLBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.bart.modeling_bart.BartForSequenceClassification.forward 复制而来
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 实现 PLBartForSequenceClassification 的前向传播逻辑，详细参数请参见 transformers 文档
        pass  # Placeholder for actual implementation

# 从 transformers.models.bart.modeling_bart.BartDecoderWrapper 复制而来，修改 Bart->PLBart
class PLBartDecoderWrapper(PLBartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 PLBartDecoder 实例作为解码器
        self.decoder = PLBartDecoder(config)

    def forward(self, *args, **kwargs):
        # 调用 PLBartDecoder 的 forward 方法进行前向传播
        return self.decoder(*args, **kwargs)


# 从 transformers.models.bart.modeling_bart.BartForCausalLM 复制而来，修改 Bart->PLBart，facebook/bart-base->uclanlp/plbart-base
class PLBartForCausalLM(PLBartPreTrainedModel):
    # 定义权重绑定的键列表，用于共享语言模型头部的权重
    _tied_weights_keys = ["lm_head.weight"]
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 深拷贝配置对象，以免修改原始配置
        config = copy.deepcopy(config)
        # 设置标志位表明当前实例是解码器
        config.is_decoder = True
        # 设置标志位表明当前实例不是编码器解码器结构
        config.is_encoder_decoder = False
        # 调用父类初始化方法，传入深拷贝后的配置对象
        super().__init__(config)
        # 使用配置对象初始化 PLBartDecoderWrapper 模型
        self.model = PLBartDecoderWrapper(config)

        # 初始化语言模型头部，使用线性层将隐藏状态映射到词汇表大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层对象
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层对象
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出嵌入层对象
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层对象
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器对象
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器对象
    def get_decoder(self):
        return self.model.decoder

    # 前向传播方法，接收多种输入参数，使用装饰器替换返回值注释
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
        # 准备用于生成的输入数据，处理输入的各种条件参数
        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
        ):
            # 如果没有给定注意力掩码，则创建一个全为1的注意力掩码
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape)

            # 如果有过去的键值对，计算过去长度，并截取相应的输入ID
            if past_key_values:
                past_length = past_key_values[0][0].shape[2]

                # 一些生成方法已经只传递了最后一个输入ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认行为：保留最后一个输入ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]

            # 返回准备好的生成输入数据字典
            return {
                "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
    # 定义函数 _reorder_cache，用于重新排序缓存中的过去键值
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去键值元组
        reordered_past = ()
        # 遍历每个层级的过去键值
        for layer_past in past_key_values:
            # 对每个层级的过去状态进行索引选择，根据 beam_idx 重新排序，并转移到与 past_state 相同的设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值元组
        return reordered_past
```