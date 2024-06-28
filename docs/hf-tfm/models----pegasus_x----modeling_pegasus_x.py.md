# `.\models\pegasus_x\modeling_pegasus_x.py`

```
# coding=utf-8
# 版权所有 2022 年，Google 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch PEGASUS-X 模型。"""

import dataclasses  # 导入 dataclasses 模块，用于支持数据类
import math  # 导入 math 模块，提供数学函数
from typing import Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 numpy 库
import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 从 torch 中导入 nn 模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入注意力掩码处理工具函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
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
from .configuration_pegasus_x import PegasusXConfig  # 导入 PEGASUS-X 的配置文件

logger = logging.get_logger(__name__)  # 获取 logger 实例

_CHECKPOINT_FOR_DOC = "google/pegasus-x-base"  # 预训练模型的检查点名称
_CONFIG_FOR_DOC = "PegasusXConfig"  # 用于文档的配置文件名称

PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 支持的 PEGASUS-X 预训练模型列表
    "google/pegasus-x-base",
    "google/pegasus-x-large",
    # 查看所有 PEGASUS 模型，请访问 https://huggingface.co/models?filter=pegasus-x
]


@dataclasses.dataclass
class DimensionInfo:
    """维度信息的包装器。"""

    batch_size: int  # 批量大小
    seq_len: int  # 标记长度
    block_size: int  # 块大小
    num_heads: int  # 头的数量
    hidden_dim: int  # 隐藏单元维度
    dim_per_head: int  # 每个头的维度
    num_blocks: int  # 块的数量
    global_len: int  # 全局长度
    padded_seq_len: int  # 填充后的标记序列长度

    # 注意：与原始 Flax 实现相比，在编码器层的开始处，我们将标记表示填充到块大小的倍数，因此始终 T=P。


# 从 transformers.models.bart.modeling_bart.shift_tokens_right 复制过来
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的标记向右移动一个位置。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)  # 创建与 input_ids 形状相同的全零张量
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # 将 input_ids 向右移动一位
    shifted_input_ids[:, 0] = decoder_start_token_id  # 将起始位置的标记设为 decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id 必须定义。")
    # 用 pad_token_id 替换标签中可能存在的 -100 值
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    # 返回变换后的输入标识符列表
    return shifted_input_ids
class PegasusXSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, embed_dim, max_scale: int = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim  # 设置嵌入维度
        self.max_scale = max_scale  # 最大缩放系数

    @torch.no_grad()
    def forward(self, input_embeds: torch.Tensor, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        batch_size, seq_len = input_embeds.shape[:2]  # 获取输入张量的批量大小和序列长度
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=input_embeds.device
        )[:, None]  # 创建位置张量，从past_key_values_length到past_key_values_length + seq_len
        pe = torch.zeros((seq_len, self.embed_dim), device=input_embeds.device, dtype=input_embeds.dtype)  # 初始化位置编码张量
        half_d_feature = self.embed_dim // 2  # 特征维度的一半
        div_term = torch.exp(
            torch.arange(half_d_feature, device=input_embeds.device, dtype=torch.int64).type_as(input_embeds)
            * -(np.log(float(self.max_scale)) / (half_d_feature - 1))
        )  # 计算分割项，用于计算正弦和余弦值
        pe[:, :half_d_feature] = torch.sin(positions * div_term)  # 计算正弦位置编码
        pe[:, half_d_feature:] = torch.cos(positions * div_term)  # 计算余弦位置编码
        return pe[None].expand(batch_size, -1, -1)  # 返回位置编码张量，扩展为与输入张量相同的形状


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->PegasusX
class PegasusXAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PegasusXConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设置嵌入维度
        self.num_heads = num_heads  # 头数
        self.dropout = dropout  # dropout率
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )  # 检查嵌入维度是否能被头数整除

        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否因果

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # K矩阵的投影
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # V矩阵的投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Q矩阵的投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出矩阵的投影

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义一个方法用于前向传播计算
    def forward(
        self,
        # 输入参数：当前隐藏状态，作为Transformer模型的输入
        hidden_states: torch.Tensor,
        # 输入参数：键-值状态，用于注意力机制的计算，可选
        key_value_states: Optional[torch.Tensor] = None,
        # 输入参数：过去的键-值状态元组，用于Transformer解码器，可选
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 输入参数：注意力掩码，指定哪些位置需要注意，可选
        attention_mask: Optional[torch.Tensor] = None,
        # 输入参数：层级头部掩码，控制层级上的注意力头部，可选
        layer_head_mask: Optional[torch.Tensor] = None,
        # 输入参数：是否输出注意力信息，默认为False
        output_attentions: bool = False,
    # 定义了一个名为 PegasusXGlobalLocalAttention 的类，继承自 nn.Module 类。
    """Global + Local attention. For use with Encoder only."""
    # 此类实现了全局和局部注意力机制，仅用于编码器。

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
    ):
        # 初始化函数，设置类的参数和模块。
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # 检查 embed_dim 是否能被 num_heads 整除，如果不能，抛出 ValueError。
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 线性变换层，用于投影查询、键、值以及输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新整形张量的形状，以适应多头注意力的计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 前向传播函数，实现注意力机制的计算
        ...

    def compute_global_attention_representations(
        self, global_q, global_k, global_v, local_k, local_v, mask, dim: DimensionInfo
    ):
        # 计算全局注意力表示的函数，输入包括全局查询、全局键值对、局部键值对、掩码和维度信息
        ...
    ):
        """Compute attention representations for global tokens.

        Global tokens will attend to both global tokens as well as all input sequence tokens. Because the input
        sequence tokens are arranged in blocks for local attention, we unblock them and compute attention.

        Args:
            global_q (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                query vectors from global tokens
            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                key vectors from global tokens
            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                value vectors from global tokens
            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                key vectors from local tokens
            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                value vectors from local tokens
            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask
            dim (DimensionInfo): DimensionInfo wrapper for dimensions

        Returns:
            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size
        """
        # Concatenate global and local key vectors along the sequence dimension
        # Shape: [batch_size, num_heads, global_len+padded_seq_len, dim_per_head]
        global_and_local_k = torch.cat([global_k, local_k], dim=2)
        
        # Concatenate global and local value vectors along the sequence dimension
        # Shape: [batch_size, num_heads, global_len+padded_seq_len, dim_per_head]
        global_and_local_v = torch.cat([global_v, local_v], dim=2)

        # Extend the mask to cover both global and local tokens
        # Shape: [batch_size, global_len+padded_seq_len]
        extended_mask = nn.functional.pad(mask, pad=(dim.global_len, 0), value=0)

        # Compute attention weights between global query and concatenated global/local key vectors
        # Shape: [batch_size, num_heads, global_len, global_len+padded_seq_len]
        attn_weights = torch.einsum("BHGF,BHXF->BHGX", global_q, global_and_local_k)
        attn_weights = attn_weights + extended_mask[:, None, None, :]  # Add extended mask
        
        # Apply softmax to compute attention probabilities and apply dropout
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        # Compute attention output using attention probabilities and concatenated global/local value vectors
        # Shape: [batch_size, num_heads, global_len, dim_per_head]
        attn_output = torch.einsum("BHGX,BHXF->BHGF", attn_probs, global_and_local_v)
        return attn_output, attn_probs
# 定义 PegasusXEncoderLayer 类，继承自 nn.Module，用于实现 Pegasus X 模型的编码器层
class PegasusXEncoderLayer(nn.Module):
    # 初始化方法，接受两个参数：stagger_blocks_this_layer 表示是否在此层中交错块，config 表示配置信息对象 PegasusXConfig
    def __init__(self, stagger_blocks_this_layer: bool, config: PegasusXConfig):
        super().__init__()
        # 设置编码器层的 embed_dim 属性为配置中的 d_model
        self.embed_dim = config.d_model
        # 使用 PegasusXGlobalLocalAttention 创建自注意力机制对象 self.self_attn
        self.self_attn = PegasusXGlobalLocalAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            block_size=config.block_size,
            dropout=config.attention_dropout,
        )
        # 初始化自注意力层归一化层 self.self_attn_layer_norm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化全局自注意力层归一化层 self.global_self_attn_layer_norm
        self.global_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置激活函数为配置中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数 dropout 概率
        self.activation_dropout = config.activation_dropout
        # 初始化全连接层 fc1，输入维度为 embed_dim，输出维度为配置中的 encoder_ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 初始化全连接层 fc2，输入维度为配置中的 encoder_ffn_dim，输出维度为 embed_dim
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 初始化最终的归一化层 self.final_layer_norm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置是否在此层中交错块的标志
        self.stagger_blocks_this_layer = stagger_blocks_this_layer
        # 设置块大小为配置中的 block_size
        self.block_size = config.block_size

    # 前向传播方法，接受多个参数：hidden_states 表示输入的隐藏状态张量，global_hidden_states 表示全局隐藏状态张量，
    # attention_mask 表示注意力掩码张量，output_attentions 表示是否输出注意力信息，默认为 False
    def forward(
        self,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        # 留待具体实现前向传播逻辑

    # 类方法，用于在本地 tokens 上填充隐藏状态和注意力掩码
    @classmethod
    def pad_local_tokens(cls, hidden_states, attention_mask, block_size):
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # 计算需要填充的大小
        pad_size = block_size // 2
        # 获取张量数据类型的最小值
        mask_min_value = torch.finfo(hidden_states.dtype).min
        # 对隐藏状态进行填充，只在序列长度维度上进行填充
        padded_hidden_states = torch.nn.functional.pad(
            hidden_states,
            pad=(0, 0, pad_size, pad_size),
        )
        # 对注意力掩码进行填充，只在序列长度维度上进行填充，并设置填充值为 mask_min_value
        padded_mask = torch.nn.functional.pad(
            attention_mask,
            pad=(pad_size, pad_size),
            value=mask_min_value,
        )
        return padded_hidden_states, padded_mask

    # 类方法，用于在本地 tokens 上取消填充隐藏状态
    @classmethod
    def unpad_local_tokens(cls, padded_hidden_states, block_size):
        # padded_hidden_states: [batch_size, padded seq_len, hidden_dim]
        # 计算填充的大小
        pad_size = block_size // 2
        # 返回去除填充后的隐藏状态，仅保留有效序列长度的部分
        return padded_hidden_states[:, pad_size:-pad_size, :]


class PegasusXDecoderLayer(nn.Module):
    # 留待后续实现
    # 初始化函数，用于初始化一个 PegasusXDecoderLayer 对象
    def __init__(self, config: PegasusXConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model

        # 创建自注意力机制对象，用于解码器的自注意力
        self.self_attn = PegasusXAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
        )

        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置激活函数为配置中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的 dropout 概率
        self.activation_dropout = config.activation_dropout

        # 创建自注意力层的 LayerNorm 层，用于归一化输入
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建编码器-解码器注意力机制对象，用于解码器的编码器-解码器注意力
        self.encoder_attn = PegasusXAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
        )

        # 创建编码器-解码器注意力层的 LayerNorm 层，用于归一化输入
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建第一个全连接层，将解码器的嵌入维度映射到配置中指定的前馈神经网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 创建第二个全连接层，将前馈神经网络的维度映射回解码器的嵌入维度
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        # 创建最终的 LayerNorm 层，用于归一化输出
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
class PegasusXPreTrainedModel(PreTrainedModel):
    # 设置配置类为 PegasusXConfig
    config_class = PegasusXConfig
    # 模型前缀为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不拆分的模块列表，使用正则表达式指定
    _no_split_modules = [r"PegasusXEncoderLayer", r"PegasusXDecoderLayer"]

    def _init_weights(self, module):
        # 初始化权重函数，使用配置中的初始标准差
        std = self.config.init_std
        # 如果是线性层，初始化权重为正态分布，偏置为零
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层，初始化权重为正态分布
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


PEGASUS_X_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PegasusXConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PEGASUS_X_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, PegasusXForConditionalGeneration

    >>> model = PegasusXForConditionalGeneration.from_pretrained("google/pegasus-x-base")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")

    >>> ARTICLE_TO_SUMMARIZE = (
    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    ... )
    >>> inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"])
    >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "California's largest electricity provider has turned off power to hundreds of thousands of customers."
    ```
"""

PEGASUS_X_INPUTS_DOCSTRING = r"""
"""


class PegasusXEncoder(PegasusXPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PegasusXEncoderLayer`].

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类构造函数初始化模型
        super().__init__(config)

        # 从配置中获取模型的dropout率和encoder层的layerdrop率
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 获取词嵌入的维度，并设置最大源序列位置和词嵌入的缩放因子
        embed_dim = config.d_model
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 如果提供了外部的嵌入词向量，则使用它；否则初始化一个词嵌入层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim)

        # 初始化全局嵌入层和位置嵌入层
        self.embed_global = nn.Embedding(config.num_global_tokens, embed_dim)
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(embed_dim)

        # 初始化编码器层的模块列表，根据配置可能会交错局部块
        self.layers = nn.ModuleList(
            [
                PegasusXEncoderLayer(
                    stagger_blocks_this_layer=i % 2 == 1 and config.stagger_local_blocks, config=config
                )
                for i in range(config.encoder_layers)
            ]
        )

        # 初始化层归一化模块
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点标志
        self.gradient_checkpointing = False

        # 调用后续初始化函数，包括权重初始化和最终处理
        self.post_init()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        # 记录日志，设置新的最大位置嵌入数量
        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        # 根据新的配置重新初始化位置嵌入层
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(self.config.d_model)
        self.embed_positions.to(self.device)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        # 返回当前位置嵌入层的引用
        return self.embed_positions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
class PegasusXDecoder(PegasusXPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`PegasusDecoderLayer`]

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout  # 设置 dropout 概率
        self.layerdrop = config.decoder_layerdrop  # 设置层级 dropout 概率
        self.max_target_positions = config.max_position_embeddings  # 最大目标位置
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 嵌入尺度，如果配置中指定要缩放则开平方根

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens  # 如果提供了嵌入词汇表，则使用它
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)  # 否则创建一个新的嵌入层

        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(config.d_model)  # 初始化位置编码
        self.layers = nn.ModuleList([PegasusXDecoderLayer(config) for _ in range(config.decoder_layers)])  # 创建多个解码层
        self.layer_norm = nn.LayerNorm(config.d_model)  # 层归一化

        self.gradient_checkpointing = False  # 梯度检查点默认关闭
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回嵌入词汇表

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置新的嵌入词汇表

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,



@add_start_docstrings(
    "The bare PEGASUS-X Model outputting raw hidden-states without any specific head on top.",
    PEGASUS_X_START_DOCSTRING,
)
class PegasusXModel(PegasusXPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: PegasusXConfig):
        super().__init__(config)

        vocab_size = config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model)  # 创建共享的嵌入层

        self.encoder = PegasusXEncoder(config, self.shared)  # 初始化编码器
        self.decoder = PegasusXDecoder(config, self.shared)  # 初始化解码器

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.shared  # 返回共享的嵌入层

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared  # 设置编码器的嵌入词汇表
        self.decoder.embed_tokens = self.shared  # 设置解码器的嵌入词汇表

    def get_encoder(self):
        return self.encoder  # 返回编码器实例

    def get_decoder(self):
        return self.decoder  # 返回解码器实例
    # 调整模型的位置嵌入矩阵大小，如果 `new_num_position_embeddings` 不等于 `config.max_position_embeddings`。
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        # 将模型配置中的 `max_position_embeddings` 设置为新的位置嵌入数量
        self.config.max_position_embeddings = new_num_position_embeddings
        # 调整编码器的位置嵌入矩阵大小
        self.encoder.resize_position_embeddings(new_num_position_embeddings)
        # 调整解码器的位置嵌入矩阵大小
        self.decoder.resize_position_embeddings(new_num_position_embeddings)

    # 返回编码器和解码器的位置嵌入矩阵
    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        return (self.encoder.get_position_embeddings(), self.decoder.get_position_embeddings())

    # 前向传播函数，通过添加 `add_start_docstrings_to_model_forward` 和 `replace_return_docstrings` 来注释函数用途和返回类型
    @add_start_docstrings_to_model_forward(PEGASUS_X_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器为 PegasusXForConditionalGeneration 类添加文档字符串，说明其用途和示例
@add_start_docstrings("The PEGASUS-X for conditional generation (e.g. summarization).", PEGASUS_X_START_DOCSTRING)
# 声明 PegasusXForConditionalGeneration 类，继承自 PegasusXPreTrainedModel
class PegasusXForConditionalGeneration(PegasusXPreTrainedModel):
    # 定义模型的参数前缀
    base_model_prefix = "model"
    # 定义共享权重的关键字列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接收 PegasusXConfig 类型的配置参数
    def __init__(self, config: PegasusXConfig):
        super().__init__(config)
        # 使用给定配置初始化 PegasusXModel 实例
        self.model = PegasusXModel(config)
        # 定义线性层 lm_head，用于生成输出的逻辑
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 调用后续初始化方法
        self.post_init()

    # 返回 encoder 的方法
    def get_encoder(self):
        return self.model.get_encoder()

    # 返回 decoder 的方法
    def get_decoder(self):
        return self.model.get_decoder()

    # 返回 lm_head，用于输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 调整位置嵌入的方法，根据新的位置嵌入数量调整模型的配置和编码器、解码器的位置嵌入
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        # 更新模型配置的最大位置嵌入数量
        self.config.max_position_embeddings = new_num_position_embeddings
        # 调整编码器和解码器的位置嵌入
        self.model.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    # 返回编码器和解码器的位置嵌入矩阵
    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.model.encoder.get_position_embeddings(), self.model.decoder.get_position_embeddings())

    # 使用装饰器为 model_forward 方法添加文档字符串，详细说明输入和输出的文档
    @add_start_docstrings_to_model_forward(PEGASUS_X_INPUTS_DOCSTRING)
    # 替换输出的文档字符串为 Seq2SeqLMOutput 类型和配置类相关的描述
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加 PEGASUS_X_GENERATION_EXAMPLE 的结束文档字符串
    @add_end_docstrings(PEGASUS_X_GENERATION_EXAMPLE)
    # 定义模型的前向传播函数，接受多个输入参数，所有参数都是可选的
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入序列的 token IDs，类型为可选的 PyTorch 张量
        attention_mask: Optional[torch.Tensor] = None,  # 输入序列的注意力掩码，类型为可选的 PyTorch 张量
        decoder_input_ids: Optional[torch.Tensor] = None,  # 解码器输入序列的 token IDs，类型为可选的 PyTorch 张量
        decoder_attention_mask: Optional[torch.Tensor] = None,  # 解码器输入序列的注意力掩码，类型为可选的 PyTorch 张量
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,  # 编码器的输出，类型为可选的 PyTorch 浮点张量元组
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,  # 用于存储解码器过去键值的元组，类型为可选的 PyTorch 浮点张量
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入序列的嵌入表示，类型为可选的 PyTorch 张量
        decoder_inputs_embeds: Optional[torch.Tensor] = None,  # 解码器输入序列的嵌入表示，类型为可选的 PyTorch 张量
        labels: Optional[torch.Tensor] = None,  # 模型的标签，类型为可选的 PyTorch 张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，类型为可选的布尔值
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Depending on `return_dict`:
            - if `return_dict` is `False`: returns a tuple comprising `lm_logits` and additional outputs from the model.
            - if `return_dict` is `True`: returns a `Seq2SeqLMOutput` object containing loss, logits, and other model outputs.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            # If `labels` are provided, adjust `use_cache` and prepare `decoder_input_ids` if necessary.
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # Shift labels to the right to create `decoder_input_ids` for training.
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Pass inputs to the model for computation.
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
        # Compute logits from the model's output.
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            # Compute masked language modeling loss if `labels` are provided.
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # If `return_dict` is `False`, format output as a tuple.
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # If `return_dict` is `True`, format output using `Seq2SeqLMOutput`.
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
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值对，则根据过去的长度调整 decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值对的第一个元素的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能已经只传递了最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                # 如果 decoder_input_ids 的长度大于过去的长度，只保留后面的部分
                remove_prefix_length = past_length
            else:
                # 默认行为：只保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 调整 decoder_input_ids，去除前缀部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回一个包含各种信息的字典
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能是为了调试目的）
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 调用 shift_tokens_right 函数，根据标签生成 decoder_input_ids
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态无需重新排序 -> 它们始终相同
            # 重新排序每一层的过去状态，以便按照 beam_idx 的顺序重新排列
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 从 transformers.models.bart.modeling_bart.BartDecoderWrapper 复制并修改为 PegasusXDecoderWrapper
# 这个类是一个辅助类，用于在使用因果语言模型与 EncoderDecoderModel 框架结合时，正确加载预训练检查点。

class PegasusXDecoderWrapper(PegasusXPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在因果语言模型与 EncoderDecoderModel 框架结合时正确加载预训练检查点。
    """

    def __init__(self, config):
        # 调用父类构造函数初始化对象
        super().__init__(config)
        # 创建 PegasusXDecoder 对象作为该类的一个属性
        self.decoder = PegasusXDecoder(config)

    def forward(self, *args, **kwargs):
        # 将前向传播调用委托给 self.decoder 对象
        return self.decoder(*args, **kwargs)
```