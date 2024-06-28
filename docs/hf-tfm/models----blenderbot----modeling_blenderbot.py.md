# `.\models\blenderbot\modeling_blenderbot.py`

```
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
""" PyTorch Blenderbot model."""

# 导入必要的库和模块
import copy  # 导入深拷贝函数
import math  # 导入数学函数
import os  # 导入操作系统相关功能
import warnings  # 导入警告模块
from typing import List, Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入PyTorch库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint功能
from torch import nn  # 导入神经网络相关模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

# 导入Hugging Face自定义的模块和函数
from ...activations import ACT2FN  # 导入激活函数
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入注意力掩码相关函数
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import (  # 导入工具函数
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..blenderbot_small import BlenderbotSmallForConditionalGeneration, BlenderbotSmallModel  # 导入小型Blenderbot模型
from .configuration_blenderbot import BlenderbotConfig  # 导入Blenderbot配置类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中使用的配置和检查点名称
_CONFIG_FOR_DOC = "BlenderbotConfig"
_CHECKPOINT_FOR_DOC = "facebook/blenderbot-400M-distill"

# 预训练的Blenderbot模型存档列表
BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/blenderbot-3B",
    # See all Blenderbot models at https://huggingface.co/models?filter=blenderbot
]

# 从transformers.models.bart.modeling_bart中复制的函数，将输入的ids向右移动一个token
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BlenderbotLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """
        `input_ids_shape` is expected to be [bsz x seqlen].
        """
        # 从输入的 input_ids_shape 中获取 batch size (bsz) 和序列长度 (seq_len)
        bsz, seq_len = input_ids_shape[:2]
        
        # 根据 past_key_values_length 和当前序列长度 seq_len 创建位置编码
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        
        # 调用父类的 forward 方法，并传入位置编码 positions
        return super().forward(positions)
# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Blenderbot
class BlenderbotAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BlenderbotConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设置注意力机制的嵌入维度
        self.num_heads = num_heads  # 设置多头注意力机制的头数
        self.dropout = dropout  # 设置dropout概率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        self.config = config  # 配置参数对象

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放系数
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否使用因果注意力机制

        # 线性变换层，用于将输入投影到不同的空间
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
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
        BLENDERBOT_ATTENTION_CLASSES = {"eager": BlenderbotAttention}
        # 省略了具体的前向传播逻辑，需根据实际需要填充
        pass


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Blenderbot, MBART->BLENDERBOT
class BlenderbotEncoderLayer(nn.Module):
    def __init__(self, config: BlenderbotConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 设置编码器层的嵌入维度

        # 创建自注意力层对象，选择不同的注意力实现（根据配置中的_attn_implementation字段）
        self.self_attn = BLENDERBOT_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层的LayerNorm
        self.dropout = config.dropout  # 设置dropout概率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout概率
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个线性层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个线性层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终的LayerNorm
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
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
        # 保存输入状态作为残差连接的一部分
        residual = hidden_states
        # 对输入状态进行 Layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用自注意力机制处理隐藏状态，得到处理后的隐藏状态、注意力权重以及（如果有的话）额外的注意力信息
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对处理后的隐藏状态进行 Dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接到处理后的隐藏状态上
        hidden_states = residual + hidden_states

        # 保存上一步操作后的隐藏状态作为残差连接的一部分
        residual = hidden_states
        # 对处理后的隐藏状态再次进行 Layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数到全连接层 fc1 上
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对处理后的隐藏状态进行 Dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 通过全连接层 fc2 得到最终的隐藏状态表示
        hidden_states = self.fc2(hidden_states)
        # 对最终隐藏状态再次进行 Dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接到处理后的隐藏状态上
        hidden_states = residual + hidden_states

        # 如果隐藏状态的数据类型是 torch.float16，并且包含无穷大或 NaN 值，则进行值的截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组，包含最终的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重也添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出元组
        return outputs
# 从transformers.models.mbart.modeling_mbart.MBartDecoderLayer复制并修改为BlenderbotDecoderLayer，同时将MBart->Blenderbot, MBART->BLENDERBOT
class BlenderbotDecoderLayer(nn.Module):
    def __init__(self, config: BlenderbotConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 设置嵌入维度为配置中的模型维度

        # 创建自注意力机制，根据配置选择不同的实现类
        self.self_attn = BLENDERBOT_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout  # 设置dropout概率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数根据配置选择
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout概率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 对自注意力输出进行LayerNorm

        # 创建编码器-解码器注意力机制，根据配置选择不同的实现类
        self.encoder_attn = BLENDERBOT_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 对编码器-解码器注意力输出进行LayerNorm

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # 第一个线性层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # 第二个线性层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终输出进行LayerNorm

    # 前向传播函数，接受一系列输入和掩码，执行解码器层的计算
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
    # 此模型继承自 `PreTrainedModel`。请查看超类文档，了解库实现的所有通用方法，如下载或保存模型、调整输入嵌入大小、剪枝头等。

    # 此模型也是 PyTorch 的 `torch.nn.Module` 子类。您可以像使用常规 PyTorch 模块一样使用它，并参考 PyTorch 文档处理一般用法和行为相关事宜。

    # 参数：
    #   config ([`BlenderbotConfig`]):
    #       模型配置类，包含模型的所有参数。使用配置文件初始化不会加载模型的权重，仅加载配置。查看 `~PreTrainedModel.from_pretrained` 方法以加载模型权重。
"""

BLENDERBOT_GENERATION_EXAMPLE = r"""
    Conversation example:

    ```python
    >>> from transformers import AutoTokenizer, BlenderbotForConditionalGeneration

    >>> mname = "facebook/blenderbot-400M-distill"
    >>> model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    >>> tokenizer = AutoTokenizer.from_pretrained(mname)
    >>> UTTERANCE = "My friends are cool but they eat too many carbs."
    >>> print("Human: ", UTTERANCE)
    Human:  My friends are cool but they eat too many carbs.

    >>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
    >>> reply_ids = model.generate(**inputs)
    >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
    Bot: That's unfortunate. Are they trying to lose weight or are they just trying to be healthier?

    >>> REPLY = "I'm not sure"
    >>> print("Human: ", REPLY)
    Human: I'm not sure

    >>> NEXT_UTTERANCE = (
    ...     "My friends are cool but they eat too many carbs.</s> <s>That's unfortunate. "
    ...     "Are they trying to lose weight or are they just trying to be healthier?</s> "
    ...     "<s> I'm not sure."
    ... )
    >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
    >>> next_reply_ids = model.generate(**inputs)
    >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
    Bot:   I see. Well, it's good that they're trying to change their eating habits.
    ```
"""

BLENDERBOT_INPUTS_DOCSTRING = r"""
"""


class BlenderbotEncoder(BlenderbotPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BlenderbotEncoderLayer`].

    Args:
        config: BlenderbotConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置 dropout 和 layerdrop 参数
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 设置 embed_dim 为模型的维度大小，获取 padding_idx 和 max_source_positions
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 如果传入了 embed_tokens，则使用传入的，否则创建新的 Embedding 层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 创建学习得到的位置编码
        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )

        # 创建 encoder 层列表，根据配置文件中的 encoder_layers 数量
        self.layers = nn.ModuleList([BlenderbotEncoderLayer(config) for _ in range(config.encoder_layers)])

        # 创建 LayerNorm 层，输入维度为 config.d_model
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 是否开启渐变检查点，默认为 False
        self.gradient_checkpointing = False

        # 初始化权重并进行最终处理
        self.post_init()
    # 定义模型的前向传播方法，处理输入数据并生成输出结果
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BlenderbotDecoderLayer`]

    Args:
        config: BlenderbotConfig
        embed_tokens (nn.Embedding): output embedding
    """
    
    # BlenderbotDecoder 类，继承自 BlenderbotPreTrainedModel
    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置层间 dropout 概率
        self.layerdrop = config.decoder_layerdrop
        # 获取填充 token 的索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置数
        self.max_target_positions = config.max_position_embeddings
        # 设置嵌入向量的缩放因子
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        
        # 如果提供了嵌入 token，则使用提供的；否则创建新的嵌入层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        
        # 创建学习的位置嵌入层
        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        
        # 创建多个解码层，并放入 ModuleList 中
        self.layers = nn.ModuleList([BlenderbotDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建层归一化层
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 是否启用渐变检查点
        self.gradient_checkpointing = False
        
        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数
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
    # 类方法：从预训练模型加载模型实例
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # 如果预训练模型名或路径为特定字符串 "facebook/blenderbot-90M"
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            # 发出警告，提示该检查点已被弃用，并建议使用相同功能的新检查点
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical"
                " checkpoint `facebook/small_blenderbot-90M` with"
                " `BlenderbotSmallModel.from_pretrained('facebook/small_blenderbot-90M')` instead.",
                FutureWarning,
            )
            # 返回从预训练模型加载的小型 Blenderbot 模型实例
            return BlenderbotSmallModel.from_pretrained(pretrained_model_name_or_path)

        # 否则调用父类的方法加载预训练模型实例
        return super(BlenderbotModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    # 返回输入嵌入层对象
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层对象
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 获取编码器对象
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象
    def get_decoder(self):
        return self.decoder

    # 前向传播方法，添加了模型输入的文档字符串和返回值的替换说明
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加模型文档字符串，指定Blenderbot模型带有语言建模头部，可用于摘要生成。
@add_start_docstrings(
    "The Blenderbot Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_START_DOCSTRING
)
# 定义BlenderbotForConditionalGeneration类，继承自BlenderbotPreTrainedModel类。
class BlenderbotForConditionalGeneration(BlenderbotPreTrainedModel):
    # 指定基础模型前缀为"model"
    base_model_prefix = "model"
    # 指定在加载过程中忽略的键名列表，缺失时的处理
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 指定需要绑定权重的键名列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接受一个BlenderbotConfig类型的配置对象config
    def __init__(self, config: BlenderbotConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建BlenderbotModel实例，存储于self.model属性中
        self.model = BlenderbotModel(config)
        # 注册一个张量缓冲区"final_logits_bias"，值为全零张量，形状为(1, self.model.shared.num_embeddings)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建一个线性层self.lm_head，输入特征数为config.d_model，输出特征数为self.model.shared.num_embeddings，无偏置
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 执行初始化权重和最终处理
        self.post_init()

    # 类方法，从预训练模型加载模型实例
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # 如果预训练模型名称或路径为"facebook/blenderbot-90M"，发出警告，并返回BlenderbotSmallForConditionalGeneration类的预训练实例
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical"
                " checkpoint `facebook/small_blenderbot-90M` with"
                " `BlenderbotSmallForConditionalGeneration.from_pretrained('facebook/small_blenderbot-90M')` instead.",
                FutureWarning,
            )
            return BlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)

        # 否则，调用父类的from_pretrained方法，返回预训练模型实例
        return super(BlenderbotForConditionalGeneration, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    # 获取编码器的方法，返回self.model的get_encoder()方法结果
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器的方法，返回self.model的get_decoder()方法结果
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整token嵌入大小的方法，接受新的token数量new_num_tokens和可选参数pad_to_multiple_of
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的resize_token_embeddings方法，返回新的嵌入层new_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调用自身的_resize_final_logits_bias方法，调整final_logits_bias张量大小
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        # 返回新的嵌入层new_embeddings
        return new_embeddings

    # 调整final_logits_bias张量大小的方法，接受新的token数量new_num_tokens
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取旧的token数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的token数量小于等于旧的token数量
        if new_num_tokens <= old_num_tokens:
            # 截取final_logits_bias张量，使其列数等于new_num_tokens
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 创建一个额外的零张量，形状为(1, new_num_tokens - old_num_tokens)，设备与final_logits_bias相同
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            # 拼接final_logits_bias和extra_bias，扩展final_logits_bias张量
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册新的final_logits_bias张量
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出嵌入层self.lm_head的方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层self.lm_head的方法，接受新的嵌入层new_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 添加模型前向方法的文档字符串，应用输入文档字符串装饰器
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    # 替换返回值文档字符串，指定输出类型为Seq2SeqLMOutput，应用配置类_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加末尾文档字符串BLENDERBOT_GENERATION_EXAMPLE
    @add_end_docstrings(BLENDERBOT_GENERATION_EXAMPLE)
    # 正向传播方法，用于模型的前向推断过程，接受多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，可以为空
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可以为空
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入 token IDs，可以为空
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力遮罩，可以为空
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力机制的掩码，可以为空
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器多头注意力机制的掩码，可以为空
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力的多头掩码，可以为空
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,  # 编码器的输出，可以为空
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对，可以为空
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量，可以为空
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入向量，可以为空
        labels: Optional[torch.LongTensor] = None,  # 标签，可以为空
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为空
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，可以为空
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            A tuple containing either masked language modeling loss and model outputs or just model outputs.

        """
        # Determine whether to use the provided return_dict or the default from configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            # Issue a warning if use_cache is True when labels are provided, then set use_cache to False
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            
            # If decoder_input_ids and decoder_inputs_embeds are not provided, shift labels to the right for decoder input
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Pass the inputs to the model for computation
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
        
        # Calculate logits for language modeling head and add bias
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # Compute masked language modeling loss if labels are provided
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # Return output as a tuple if return_dict is False
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return Seq2SeqLMOutput if return_dict is True, containing relevant outputs
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
        # 如果使用了过去的键值（past_key_values），则修剪decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 某些生成方法可能已经只传递了最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：保留仅最后一个ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 修剪decoder_input_ids，去掉前面部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的输入字典
        return {
            "input_ids": None,  # encoder_outputs已定义。input_ids不需要
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 将此项更改以避免缓存（可能是为了调试）
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态无需重新排序 -> 它们始终保持不变
            reordered_past += (
                # 对每一层的过去状态执行重新排序，按beam_idx的顺序重新选择
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],  # 剩余部分保持不变
            )
        # 返回重新排序后的过去键值
        return reordered_past
# 从 `transformers.models.bart.modeling_bart.BartDecoderWrapper` 复制并修改为 Blenderbot 的解码器包装器
class BlenderbotDecoderWrapper(BlenderbotPreTrainedModel):
    """
    这个包装器类是一个辅助类，用于在将因果语言模型与 [`EncoderDecoderModel`] 框架结合使用时正确加载预训练检查点。
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化 Blenderbot 解码器
        self.decoder = BlenderbotDecoder(config)

    def forward(self, *args, **kwargs):
        # 前向传播到 Blenderbot 解码器
        return self.decoder(*args, **kwargs)


# 从 `transformers.models.bart.modeling_bart.BartForCausalLM` 复制并修改为 Blenderbot 的因果语言模型
# 使用 Bart -> Blenderbot, facebook/bart-base -> facebook/blenderbot-400M-distill
class BlenderbotForCausalLM(BlenderbotPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深度拷贝配置
        config = copy.deepcopy(config)
        # 设定为解码器
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        
        # 使用 BlenderbotDecoderWrapper 初始化模型
        self.model = BlenderbotDecoderWrapper(config)

        # 定义 LM 头部线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回解码器的嵌入标记
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置解码器的嵌入标记
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回 LM 头部
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 获取解码器
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
        # 前向传播，支持因果语言模型的生成
        ...

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 为生成准备输入
        ...
        # 如果模型作为编码器-解码器模型的解码器使用，会动态创建解码器注意力遮罩
        if attention_mask is None:
            # 如果未提供注意力遮罩，则创建一个与输入形状相同的全1张量
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                # 如果输入 ID 的长度大于过去长度，则移除前缀长度为过去长度
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 裁剪输入 ID，仅保留后缀部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回一个字典，包含输入 ID、注意力遮罩、过去键值、是否使用缓存的信息
        return {
            "input_ids": input_ids,  # 编码器输出已定义，不需要输入 ID
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每一层的过去状态进行重新排序，根据 beam_idx 重新排列
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```