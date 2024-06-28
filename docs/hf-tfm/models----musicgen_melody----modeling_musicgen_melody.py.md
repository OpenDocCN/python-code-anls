# `.\models\musicgen_melody\modeling_musicgen_melody.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明，版权归 Meta AI 和 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件。除非符合许可证的规定，否则不得使用本文件。
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于"原样"的基础分发的，不提供任何明示或暗示的担保或条件。
# 请参阅许可证以获取有关详细信息。
""" PyTorch Musicgen Melody model."""

# 导入所需的模块和库
import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# 导入自定义的模块和函数
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForTextEncoding
from .configuration_musicgen_melody import MusicgenMelodyConfig, MusicgenMelodyDecoderConfig

# 如果是类型检查阶段，导入 BaseStreamer 类
if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 配置文件和检查点的文档常量
_CONFIG_FOR_DOC = "MusicgenMelodyConfig"
_CHECKPOINT_FOR_DOC = "facebook/musicgen-melody"

# 预训练模型的存档列表
MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/musicgen-melody",
    # 更多 Musicgen Melody 模型请查看 https://huggingface.co/models?filter=musicgen_melody
]

# 定义一个数据类，用于 Musicgen Melody 模型的输出，包含过去状态的基类
@dataclass
class MusicgenMelodyOutputWithPast(ModelOutput):
    """
    Base class for Musicgen Melody autoregressive outputs.
    
    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            语言建模损失（在提供 `labels` 时返回）。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头的预测分数（SoftMax 之前的每个词汇标记的分数）。
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, 当传递 `use_cache=True` 或 `config.use_cache=True` 时返回):
            长度为 `config.n_layers` 的 `tuple(torch.FloatTensor)` 的元组，每个元组包含 2 个张量，形状为
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`。

            包含预计算的隐藏状态（在自注意力块中的键和值），可用于加速顺序解码。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 的元组（如果模型有嵌入层则包含嵌入层的输出 + 每层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            每层模型的隐藏状态以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            `torch.FloatTensor` 的元组（每个层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            自注意力头中注意力 softmax 后的注意力权重，用于计算自注意力头中加权平均值。
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            条件隐藏状态序列，表示文本编码器输出和音频编码器输出的投影连接。
            作为条件信号使用。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[torch.FloatTensor] = None
# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个和 input_ids 形状相同的全零张量
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将 input_ids 向右移动一位，赋值给 shifted_input_ids
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    # 将 decoder_start_token_id 放置在 shifted_input_ids 的第一列
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将 shifted_input_ids 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenSinusoidalPositionalEmbedding with Musicgen->MusicgenMelody
class MusicgenMelodySinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 根据给定的 num_positions 和 embedding_dim 创建权重矩阵
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        # 获取 sinusoidal 位置编码的权重矩阵
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            # 在 forward 方法中，将权重矩阵转换为参数的正确 dtype 和 device
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 将权重矩阵转换为 nn.Parameter，并设置为不可训练状态
        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # 对于奇数 embedding_dim，补充一个零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    # Ignore copy
    # 前向传播函数，接受嵌入输入和过去键值长度作为参数
    def forward(self, inputs_embeds: torch.Tensor, past_key_values_length: int = 0):
        # 获取输入嵌入的批大小、序列长度和嵌入维度
        bsz, seq_len, _ = inputs_embeds.size()
        
        # 根据输入的令牌 id 创建位置 id
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(inputs_embeds.device)
        
        # 如果序列长度大于权重张量的大小，扩展权重张量
        if seq_len > self.weights.size(0):
            self.make_weights(seq_len + self.offset, self.embedding_dim)
        
        # 根据位置 id 从权重张量中选择对应位置的嵌入，然后分离（detach）出来
        return self.weights.index_select(0, position_ids.view(-1)).detach()
# 从transformers.models.bart.modeling_bart.BartAttention复制的代码，修改为MusicgenMelodyAttention
class MusicgenMelodyAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[MusicgenMelodyConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设置注意力机制的嵌入维度
        self.num_heads = num_heads  # 头数，即多头注意力中的注意力头的数量
        self.dropout = dropout  # dropout率，用于避免过拟合
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度
        self.config = config  # 可选的配置对象

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于缩放注意力分数
        self.is_decoder = is_decoder  # 是否为解码器层
        self.is_causal = is_causal  # 是否是因果（自回归）注意力

        # 初始化线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 键的投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 值的投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 查询的投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出的投影层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """重塑张量形状以适应多头注意力的输入要求"""
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
        """
        执行注意力机制的前向传播
        Args:
            hidden_states: 输入的隐藏状态张量
            key_value_states: 可选的键值状态张量（用于encoder-decoder注意力）
            past_key_value: 可选的过去的键值对（用于加速Transformer解码器的计算）
            attention_mask: 可选的注意力掩码张量
            layer_head_mask: 可选的层级头掩码张量（用于控制每个头的选择性）
            output_attentions: 是否输出注意力权重

        Returns:
            tuple:
                - attention_output: 经过注意力机制后的输出张量
                - attention_weights: 注意力权重（如果output_attentions为True时）
        """
        # 省略前向传播的具体实现，主要处理输入和输出的形状变换以及线性变换
        pass

class MusicgenMelodyDecoderLayer(nn.Module):
    def __init__(self, config: MusicgenMelodyDecoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 设置解码器层的隐藏大小

        # 创建自注意力层
        self.self_attn = MusicgenMelodyAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
        )

        self.dropout = config.dropout  # dropout率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层的LayerNorm

        # 两个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=False)  # 第一个全连接层
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=False)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终输出的LayerNorm
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size `(attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states  # 保存输入状态作为残差连接的基准

        hidden_states = self.self_attn_layer_norm(hidden_states)  # 对输入状态进行 layer normalization

        # Self Attention
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 获取自注意力机制中的过去键/值投影状态，如果有的话

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 应用自注意力机制，得到新的隐藏状态、注意力权重和当前键/值投影状态

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 对隐藏状态应用 dropout

        hidden_states = residual + hidden_states  # 残差连接

        # Fully Connected
        residual = hidden_states  # 保存残差连接前的状态

        hidden_states = self.final_layer_norm(hidden_states)  # 对最终的隐藏状态进行 layer normalization

        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 应用激活函数和第一个全连接层
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)  # 第二个全连接层
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 输出为最终的隐藏状态

        if output_attentions:
            outputs += (self_attn_weights,)  # 如果需要输出注意力权重，则添加到输出中

        if use_cache:
            outputs += (present_key_value,)  # 如果需要使用缓存，则添加当前的键/值投影状态到输出中

        return outputs  # 返回所有输出
# 从 transformers.models.musicgen.modeling_musicgen.MusicgenPreTrainedModel 复制代码并将 Musicgen 替换为 MusicgenMelody
class MusicgenMelodyPreTrainedModel(PreTrainedModel):
    """
    用于处理权重初始化、下载和加载预训练模型的抽象类。

    Attributes:
        config_class: 与该模型相关的配置类 MusicgenMelodyDecoderConfig
        base_model_prefix: 模型的基础名称前缀为 "model"
        supports_gradient_checkpointing: 支持梯度检查点
        _no_split_modules: 不需要拆分的模块列表，包括 "MusicgenMelodyDecoderLayer" 和 "MusicgenMelodyAttention"
    """

    def _init_weights(self, module):
        """
        初始化给定模块的权重。

        Args:
            module: 要初始化权重的模块

        Notes:
            根据模块类型不同，使用配置的初始化因子初始化权重和偏置。
        """
        std = self.config.initializer_factor
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Musicgen_Melody_START_DOCSTRING 是 Musicgen Melody 模型的文档字符串
MUSICGEN_MELODY_START_DOCSTRING = r"""

    Musicgen Melody 模型是由 Jade Copet 等人在 [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) 中提出的。该模型是一个仅解码器的 Transformer，用于条件音乐生成。

    该模型继承自 [`PreTrainedModel`]。查阅超类文档以了解库为所有模型实现的通用方法（如下载或保存模型、调整输入嵌入、剪枝头等）。

    该模型也是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。可以像常规的 PyTorch 模块一样使用，并参考 PyTorch 文档了解所有与一般用法和行为相关的事项。

    Parameters:
        config ([`MusicgenMelodyConfig`]): 包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

# MUSICGEN_MELODY_INPUTS_DOCSTRING 是 Musicgen Melody 模型输入的文档字符串
MUSICGEN_MELODY_INPUTS_DOCSTRING = r"""
"""

# MUSICGEN_MELODY_DECODER_INPUTS_DOCSTRING 是 Musicgen Melody 解码器输入的文档字符串
MUSICGEN_MELODY_DECODER_INPUTS_DOCSTRING = r"""
"""


# 从 transformers.models.musicgen.modeling_musicgen.MusicgenDecoder 复制代码并将 MUSICGEN->MUSICGEN_MELODY,Musicgen->MusicgenMelody
class MusicgenMelodyDecoder(MusicgenMelodyPreTrainedModel):
    """
    Transformer 解码器，由 *config.num_hidden_layers* 层组成。每一层都是一个 [`MusicgenMelodyDecoderLayer`]
    """
    # 初始化函数，用于初始化模型参数及配置
    def __init__(self, config: MusicgenMelodyDecoderConfig):
        # 调用父类的初始化函数，传入配置对象
        super().__init__(config)
        # 设置模型的dropout比例
        self.dropout = config.dropout
        # 设置模型的层级dropout比例
        self.layerdrop = config.layerdrop
        # 设置模型的最大目标位置
        self.max_target_positions = config.max_position_embeddings
        # 设置模型的隐藏层维度
        self.d_model = config.hidden_size
        # 设置模型的码书数量
        self.num_codebooks = config.num_codebooks
        # 如果配置中指定了缩放embedding，则计算embedding的缩放因子，否则为1.0
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 计算embedding的维度，词汇表大小加1
        embed_dim = config.vocab_size + 1
        # 使用nn.ModuleList创建多个嵌入层对象，数量为码书数量
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(embed_dim, config.hidden_size) for _ in range(config.num_codebooks)]
        )

        # 使用自定义的Sinusoidal位置嵌入类创建位置嵌入对象
        self.embed_positions = MusicgenMelodySinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        # 使用nn.ModuleList创建多个解码层对象，数量为隐藏层数量
        self.layers = nn.ModuleList([MusicgenMelodyDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建层归一化对象，用于层次之间的正则化
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 是否启用梯度检查点，默认为False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层对象的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层对象的方法
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 重写的前向传播函数，处理解码器的输入数据
    @add_start_docstrings_to_model_forward(MUSICGEN_MELODY_DECODER_INPUTS_DOCSTRING)
    # 忽略复制操作的说明
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加文档字符串，描述该模型是MusicgenMelody的解码器模型，输出原始的隐藏状态，没有特定的顶部头部。
# 使用MUSICGEN_MELODY_START_DOCSTRING中定义的文档字符串。
@add_start_docstrings(
    "The bare MusicgenMelody decoder model outputting raw hidden-states without any specific head on top.",
    MUSICGEN_MELODY_START_DOCSTRING,
)
# 从transformers.models.musicgen.modeling_musicgen.MusicgenModel复制代码，将MUSICGEN->MUSICGEN_MELODY，Musicgen->MusicgenMelody
class MusicgenMelodyModel(MusicgenMelodyPreTrainedModel):
    def __init__(self, config: MusicgenMelodyDecoderConfig):
        super().__init__(config)
        # 初始化解码器，并传入配置参数
        self.decoder = MusicgenMelodyDecoder(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 返回解码器的嵌入层
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置解码器的嵌入层
    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 添加文档字符串到模型的forward方法，使用MUSICGEN_MELODY_DECODER_INPUTS_DOCSTRING中定义的文档字符串
    # 忽略复制操作
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 如果没有显式提供输出注意力权重，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有显式提供输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有显式提供是否使用缓存，则使用配置中的默认值
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果没有显式提供是否返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 解码器的输出包括 (解码特征, 过去键值, 解码隐藏状态, 解码注意力权重)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不要求返回字典，则直接返回解码器的输出
        if not return_dict:
            return decoder_outputs

        # 返回包含过去键值的基本模型输出对象，包括最终隐藏状态、过去键值、隐藏状态列表和注意力权重列表
        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )
# 添加模型文档字符串，描述该类为带语言建模头部的Musicgen Melody解码器模型
@add_start_docstrings(
    "The Musicgen Melody decoder model with a language modelling head on top.",
    MUSICGEN_MELODY_START_DOCSTRING,
)
# 从transformers.models.musicgen.modeling_musicgen.MusicgenForCausalLM复制过来，将MUSICGEN->MUSICGEN_MELODY,Musicgen->MusicgenMelody,MusicGen->Musicgen Melody
class MusicgenMelodyForCausalLM(MusicgenMelodyPreTrainedModel):
    def __init__(self, config: MusicgenMelodyDecoderConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)

        # 创建Musicgen Melody模型实例
        self.model = MusicgenMelodyModel(config)

        # 设置编码簿数量
        self.num_codebooks = config.num_codebooks
        # 创建线性层列表作为语言建模头部，每个编码簿对应一个线性层
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层（embed_tokens）的方法
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层（embed_tokens）的方法
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出嵌入层列表（lm_heads）的方法
    def get_output_embeddings(self):
        return self.lm_heads

    # 设置输出嵌入层列表（lm_heads）的方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_heads = new_embeddings

    # 设置解码器（decoder）的方法
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器（decoder）的方法
    def get_decoder(self):
        return self.model.decoder

    # 添加模型前向方法的文档字符串，包括Musicgen Melody解码器输入的详细描述
    # 并使用装饰器replace_return_docstrings替换返回值的文档字符串为MusicgenMelodyOutputWithPast类型的描述
    @add_start_docstrings_to_model_forward(MUSICGEN_MELODY_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MusicgenMelodyOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 忽略复制
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MusicgenMelodyOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        Returns:
            Tuple or MusicgenMelodyOutputWithPast: Depending on `return_dict`, returns either a tuple or an instance
            of `MusicgenMelodyOutputWithPast`.
        """

        # Determine whether to use the provided `return_dict` or default from configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the model with specified inputs and configuration
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states from model outputs
        hidden_states = outputs[0]

        # Generate logits for language modeling heads based on hidden states
        lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)

        # Placeholder for loss; training for MusicgenMelody is not implemented
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented for MusicgenMelody.")

        # Reshape logits for further processing
        # (bsz, num_codebooks, seq_len, vocab_size) -> (bsz * num_codebooks, seq_len, vocab_size)
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])

        # If `return_dict` is False, return output as a tuple with optional loss
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # If `return_dict` is True, return structured output using `MusicgenMelodyOutputWithPast`
        return MusicgenMelodyOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Ignore copy
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        past_key_values=None,
        use_cache=True,
        delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,
        ):
        """
        Prepare inputs for the generation process, tailored for Music generation tasks.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The input token IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding tokens.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Hidden states from the encoder.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid attending to encoder padding tokens.
            head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask for the attention heads.
            past_key_values (tuple of `torch.Tensor` of shape `(batch_size, num_heads, past_sequence_length, hidden_size)`):
                Cached key and value states for fast decoding.
            use_cache (bool, *optional*):
                Whether to use caching mechanism for fast decoding.
            delay_pattern_mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_codebooks, vocab_size)`, *optional*):
                Mask indicating patterns of delay for pattern-based music generation.
            guidance_scale (float, *optional*):
                Scaling factor for guidance during generation.

        Returns:
            dict: Dictionary containing prepared inputs for the generation process.
        """
    ):
        # 如果延迟模式掩码为 None，则调用 build_delay_pattern_mask 方法生成
        if delay_pattern_mask is None:
            input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
                input_ids,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # 应用延迟模式掩码到输入的 token ids
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)

        # 如果有指导尺度并且大于1，则进行以下操作
        if guidance_scale is not None and guidance_scale > 1:
            # 对于无分类器指导的情况，需要在批次维度上复制解码器参数（将在采样前拆分）
            input_ids = input_ids.repeat((2, 1))
            # 如果存在注意力掩码，则也进行相同的复制操作
            if attention_mask is not None:
                attention_mask = attention_mask.repeat((2, 1))

            # 如果存在编码器隐藏状态，则在批次维度上进行拼接，用零填充
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.concatenate(
                    [encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=0
                )

            # 如果存在编码器注意力掩码，则在批次维度上进行拼接，用零填充
            if encoder_attention_mask is not None:
                encoder_attention_mask = torch.concatenate(
                    [encoder_attention_mask, torch.zeros_like(encoder_attention_mask)], dim=0
                )

        # 如果过去的关键值不为 None，则仅保留输入的最后一个 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

            # 在第一代步骤中，仅使用条件信号，但保留注意力掩码
            encoder_hidden_states = None

        # 返回生成方法的结果字典
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask."""
        # 获取输入 token ids 的序列长度
        seq_len = input_ids.shape[-1]
        # 裁剪 decoder_pad_token_mask 到与输入序列长度相同的维度
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        # 根据 mask 中值为 -1 的位置，保留输入 token ids 的预测，其他位置用 mask 中的值替换
        input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
# 添加文档字符串到类 `MusicgenMelodyForConditionalGeneration`，描述了该模型的组成和用途
@add_start_docstrings(
    "The composite Musicgen Melody model with a text and audio conditional models, a MusicgenMelody decoder and an audio encoder, "
    "for music generation tasks with one or both of text and audio prompts.",
    MUSICGEN_MELODY_START_DOCSTRING,
    """
        text_encoder (`Optional[PreTrainedModel]`, *optional*): Text encoder.
        audio_encoder (`Optional[PreTrainedModel]`, *optional*): Audio code decoder.
        decoder (`Optional[MusicgenMelodyForCausalLM]`, *optional*): MusicGen Melody decoder used to generate audio codes.
    """
)

class MusicgenMelodyForConditionalGeneration(PreTrainedModel):
    # 指定配置类为 `MusicgenMelodyConfig`
    config_class = MusicgenMelodyConfig
    # 主要输入名称为 `input_ids`
    main_input_name = "input_ids"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: MusicgenMelodyConfig = None,
        text_encoder: Optional[PreTrainedModel] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[MusicgenMelodyForCausalLM] = None,
    ):
        if config is None and None in (text_encoder, audio_encoder, decoder):
            raise ValueError(
                "Either a configuration has to be provided, or all three of text encoder, audio encoder and Musicgen Melody decoder."
            )
        # 如果配置为 None 并且 text_encoder、audio_encoder、decoder 中有任何一个为 None，则抛出 ValueError
        if config is None:
            # 如果配置为 None，则从子模型配置中创建 MusicgenMelodyConfig 对象
            config = MusicgenMelodyConfig.from_sub_models_config(
                text_encoder.config, audio_encoder.config, decoder.config
            )
        else:
            # 如果配置不为 None，则检查配置是否为 self.config_class 类型，否则抛出 ValueError
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # 使用给定配置初始化父类
        super().__init__(config)

        # 如果 text_encoder 为 None，则从配置中创建 AutoModelForTextEncoding 对象
        if text_encoder is None:
            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder)

        # 如果 audio_encoder 为 None，则从配置中创建 AutoModel 对象
        if audio_encoder is None:
            audio_encoder = AutoModel.from_config(config.audio_encoder)

        # 如果 decoder 为 None，则使用 MusicgenMelodyForCausalLM 类创建 decoder 对象
        if decoder is None:
            decoder = MusicgenMelodyForCausalLM(config.decoder)

        # 分别将 text_encoder、audio_encoder、decoder 赋值给对象属性
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder

        # 确保各模型配置指向共享配置，以保持配置更新同步
        self.text_encoder.config = self.config.text_encoder
        self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config.decoder

        # 如果 text_encoder 输出的 embeddings 不为 None，则抛出 ValueError
        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.text_encoder} should not have a LM Head. Please use a model without and LM Head"
            )

        # 如果 text_encoder 和 decoder 的隐藏层大小不一致，则初始化线性层进行投影
        if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size:
            self.enc_to_dec_proj = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)

        # 如果音频编码器提取色度后的输出维度与 decoder 的隐藏层大小不一致，则初始化线性层进行投影
        if self.config.num_chroma != self.decoder.config.hidden_size:
            self.audio_enc_to_dec_proj = nn.Linear(self.config.num_chroma, self.decoder.config.hidden_size)

        # 初始化后处理函数，包括初始化投影层的权重，并根据需要将 text_encoder 和 decoder 的权重进行绑定
        self.post_init()

    def _init_weights(self, module):
        # MusicgenMelodyForConditionalGeneration 由已初始化的 PreTrainedModels 组成
        # 投影层仍需初始化
        std = self.decoder.config.initializer_factor
        # 如果 module 是 nn.Linear 类型，则初始化其权重
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
    # 绑定权重函数，用于绑定文本编码器和解码器
    def tie_weights(self):
        # 如果需要绑定文本编码器和解码器
        if self.config.tie_encoder_decoder:
            # 获取解码器基础模型的前缀
            decoder_base_model_prefix = self.decoder.base_model_prefix
            # 绑定文本编码器和解码器基础模型的权重
            self._tie_encoder_decoder_weights(
                self.text_encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    # 获取文本编码器
    def get_text_encoder(self):
        return self.text_encoder

    # 获取编码器
    def get_encoder(self):
        # 获取文本编码器以计算生成时的条件隐藏状态
        return self.get_text_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    # 从预训练的子模型创建实例
    @classmethod
    def from_sub_models_pretrained(
        cls,
        text_encoder_pretrained_model_name_or_path: str = None,
        audio_encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    
    # 前向传播函数，接收一系列输入和参数
    @add_start_docstrings_to_model_forward(MUSICGEN_MELODY_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MusicgenMelodyOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    
    # 为生成准备输入的函数，接收一系列输入和参数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        encoder_hidden_states=None,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        use_cache=None,
        decoder_delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,
    ):
        if decoder_delay_pattern_mask is None:
            # 如果延迟模式掩码为None，则调用self.decoder.build_delay_pattern_mask生成新的延迟模式掩码
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                decoder_input_ids,
                self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # 应用延迟模式掩码到decoder_input_ids上
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)

        if guidance_scale is not None and guidance_scale > 1:
            # 对于分类器无关的指导，我们需要将decoder参数在批次维度上复制（在采样之前会分割这些参数）
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 某些生成方法已经只传递了最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认旧的行为：保留仅最终ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 截取decoder_input_ids以去除前缀长度
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            # 我们只想在第一个生成步骤使用条件信号，但保留注意力掩码
            encoder_hidden_states = None
            # 我们也必须更新注意力掩码

        return {
            "input_ids": None,  # encoder_hidden_states已定义。input_ids不需要
            "encoder_hidden_states": encoder_hidden_states,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_head_mask": decoder_head_mask,
            "use_cache": use_cache,
        }

    # 从transformers.models.musicgen.modeling_musicgen.MusicgenForConditionalGeneration._prepare_decoder_input_ids_for_generation复制而来
    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """为使用编码器-解码器模型生成准备 `decoder_input_ids`"""

        # 1. 检查用户是否手动定义了 `decoder_input_ids`。为了方便输入命名，我们也允许用户在 `input_ids` 下传递它，如果编码器不将其用作主要输入。
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. 编码器-解码器模型期望 `decoder_input_ids` 以特殊令牌开头。让我们确保这一点。
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = (
            torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device)
            * decoder_start_token_id
        )

        # 如果没有用户输入 -> 使用 decoder_start_token_id 作为 decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start

        # 如果有用户输入但不以 decoder_start_token_id 开头 -> 在前面添加 decoder_start_token_id（并调整 decoder_attention_mask 如果提供）
        elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def _prepare_encoder_hidden_states_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
        guidance_scale: Optional[float] = None,
    ):
        """为生成准备编码器隐藏状态的参数"""

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        """根据标签准备解码器的输入ids"""
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        """调整标记嵌入大小的方法，通过 EncoderDecoderModel 直接不支持。请使用包装对象的相应方法（model.encoder.resize_token_embeddings(...) 或 model.decoder.resize_token_embeddings(...)）"""
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )
    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        # 如果已经提供了输入张量，则直接返回
        if inputs is not None:
            return inputs

        # 如果未提供输入张量但未定义起始标记 ID，则抛出数值错误异常
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # 如果 `model_kwargs` 中包含张量，则从中推断批次大小
        # 这对基于解码器的语言模型的软提示或多模态实现非常有帮助
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        # 返回形状为 (batch_size, 1) 的全一张量，乘以起始标记 ID，数据类型为长整型
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        """Generates sequences using the model."""
        # 实现生成序列的方法，这里不作注释

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        model_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Updates model keyword arguments for generation."""
        # 更新模型生成过程中的关键字参数

        # 更新 past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # 如果输出对象有状态信息，则更新状态
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # 更新 token_type_ids，添加最后一个值
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # 更新解码器注意力掩码
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

        # 返回更新后的模型关键字参数字典
        return model_kwargs
```