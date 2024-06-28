# `.\models\pegasus\modeling_pegasus.py`

```py
# 设置文件编码格式为 UTF-8
# 版权声明，指出版权归 Google 和 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何形式的明示或暗示保证，包括但不限于适销性或特定用途适用性的保证
# 有关详细信息，请参阅许可证

""" PyTorch PEGASUS model."""

import copy  # 导入深拷贝函数
import math  # 导入数学库中的数学函数
from typing import List, Optional, Tuple, Union  # 导入类型提示支持的数据结构

import numpy as np  # 导入 numpy 库
import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 检查点工具
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入注意力掩码工具函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型工具函数
from ...utils import (  # 导入通用工具函数
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_pegasus import PegasusConfig  # 导入 Pegasus 配置文件

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "google/pegasus-large"  # 用于文档的检查点模型名称
_CONFIG_FOR_DOC = "PegasusConfig"  # 用于文档的配置文件名称

PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型的存档列表
    "google/pegasus-large",
    # 可以在 https://huggingface.co/models?filter=pegasus 查看所有 PEGASUS 模型
]


# 从 transformers.models.bart.modeling_bart.shift_tokens_right 复制过来
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的 token 向右移动一位。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)  # 创建与 input_ids 形状相同的零张量
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # 将 input_ids 向右移动一位
    shifted_input_ids[:, 0] = decoder_start_token_id  # 在首位插入 decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将 labels 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# 从 transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding 复制过来，并将 Marian 改为 Pegasus
class PegasusSinusoidalPositionalEmbedding(nn.Embedding):
    """该模块生成任意长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)  # 调用父类的初始化方法
        self.weight = self._init_weight(self.weight)  # 初始化权重

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Initialize positional embeddings for transformer model.

        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape  # 获取输出张量的形状，n_pos 表示位置数，dim 表示维度
        # 创建位置编码矩阵，用于表示不同位置的嵌入向量
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # 设置张量为不需要梯度，以避免在 PyTorch 1.8+ 版本中出现错误
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1  # 计算用于分隔 sin 和 cos 的索引位置
        # 将 sin 和 cos 值填充到输出张量的不同部分
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()  # 分离张量，防止在后续计算中被修改
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """
        Perform forward pass of the transformer model.

        `input_ids_shape` is expected to be [bsz x seqlen].
        """
        bsz, seq_len = input_ids_shape[:2]  # 解析输入张量的大小，bsz 表示批量大小，seq_len 表示序列长度
        # 生成位置编码张量，表示每个位置的索引，加上 past_key_values_length 以适应历史键值长度
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)
# 从transformers.models.bart.modeling_bart.BartAttention复制并将Bart->Pegasus
class PegasusAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PegasusConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 初始化嵌入维度
        self.num_heads = num_heads  # 初始化注意头数
        self.dropout = dropout  # 初始化dropout率
        self.head_dim = embed_dim // num_heads  # 每个注意头的维度
        self.config = config  # Pegasus的配置对象

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除 (得到的 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器的标志
        self.is_causal = is_causal  # 是否为因果注意力的标志

        # 初始化线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新整形张量以适应多头注意力的结构
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
        # PegasusAttention的前向传播函数
        pass  # 函数体未提供完整，暂时无内容

# 从transformers.models.mbart.modeling_mbart.MBartEncoderLayer复制并将MBart->Pegasus, MBART->PEGASUS
class PegasusEncoderLayer(nn.Module):
    def __init__(self, config: PegasusConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 初始化嵌入维度

        # 使用配置中的注意力实现类构建自注意力层
        self.self_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层的LayerNorm
        self.dropout = config.dropout  # dropout率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout率
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个前馈网络层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个前馈网络层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终的LayerNorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        # PegasusEncoderLayer的前向传播函数
        pass  # 函数体未提供完整，暂时无内容
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
        # 保留输入的残差连接
        residual = hidden_states
        # 对输入的 hidden_states 进行 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用 self-attention 机制计算新的 hidden_states，并返回 attention 权重和额外信息
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对计算后的 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与新计算得到的 hidden_states 相加，形成新的 hidden_states
        hidden_states = residual + hidden_states

        # 再次保留输入的残差连接
        residual = hidden_states
        # 对更新后的 hidden_states 进行 Layer Normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数和线性变换 fc1 到 hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对经过 fc1 的 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 经过第二个线性变换 fc2
        hidden_states = self.fc2(hidden_states)
        # 对 fc2 输出的 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与新计算得到的 hidden_states 相加，形成最终的 hidden_states
        hidden_states = residual + hidden_states

        # 如果 hidden_states 的数据类型是 torch.float16，并且包含无穷大或 NaN 值
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 对 hidden_states 进行截断操作，避免超出数据类型的范围
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 将最终的 hidden_states 存入 outputs
        outputs = (hidden_states,)

        # 如果需要输出 attentions，将 attentions 加入到 outputs 中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回 outputs
        return outputs
# 从transformers.models.mbart.modeling_mbart.MBartDecoderLayer复制过来，将MBart替换为Pegasus，MBART替换为PEGASUS
class PegasusDecoderLayer(nn.Module):
    def __init__(self, config: PegasusConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 设置嵌入维度为配置中的模型维度大小

        # 初始化自注意力层，根据配置选择实现方式，设定头数、dropout等参数
        self.self_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout  # 设置dropout率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数设定为配置中指定的函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化自注意力层的LayerNorm

        # 初始化编码器-解码器注意力层，根据配置选择实现方式，设定头数、dropout等参数
        self.encoder_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化编码器-解码器注意力层的LayerNorm

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # 第一个线性层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # 第二个线性层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终输出的LayerNorm

    # 前向传播函数定义
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



# PegasusPreTrainedModel类定义，继承自PreTrainedModel
class PegasusPreTrainedModel(PreTrainedModel):
    config_class = PegasusConfig  # 指定配置类为PegasusConfig
    base_model_prefix = "model"  # 基础模型前缀设定为"model"
    supports_gradient_checkpointing = True  # 支持梯度检查点

    # 初始化权重函数
    def _init_weights(self, module):
        std = self.config.init_std  # 初始化标准差设定为配置中的初始标准差
        if isinstance(module, nn.Linear):  # 如果是线性层
            module.weight.data.normal_(mean=0.0, std=std)  # 权重初始化为正态分布
            if module.bias is not None:
                module.bias.data.zero_()  # 如果有偏置项，初始化为0
        elif isinstance(module, PegasusSinusoidalPositionalEmbedding):
            pass  # 如果是PegasusSinusoidalPositionalEmbedding类型，则不进行任何初始化操作
        elif isinstance(module, nn.Embedding):  # 如果是嵌入层
            module.weight.data.normal_(mean=0.0, std=std)  # 权重初始化为正态分布
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果有padding_idx，对应位置初始化为0



# PEGASUS_START_DOCSTRING文档字符串定义
PEGASUS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.

"""
    # 使用它作为普通的 PyTorch 模块，并参考 PyTorch 文档以了解所有与一般用法和行为相关的事项。

    Parameters:
        config ([`PegasusConfig`]):
            模型配置类，包含模型的所有参数。使用配置文件初始化不会加载模型的权重，只加载配置信息。查看
            [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

PEGASUS_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```
    >>> from transformers import AutoTokenizer, PegasusForConditionalGeneration

    >>> model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

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

PEGASUS_INPUTS_DOCSTRING = r"""
"""


class PegasusEncoder(PegasusPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PegasusEncoderLayer`].

    Args:
        config: PegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # Initialize embedding tokens with padding index if provided, otherwise default
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # Initialize sinusoidal positional embeddings
        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )

        # Create encoder layers based on config
        self.layers = nn.ModuleList([PegasusEncoderLayer(config) for _ in range(config.encoder_layers)])

        # Layer normalization for encoder output
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
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
        # 记录日志，显示设置最大位置编码数
        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        # 更新模型配置中的最大位置编码数
        self.config.max_position_embeddings = new_num_position_embeddings

        # 创建新的位置编码嵌入对象，根据新的最大位置编码数和模型维度创建
        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            self.config.max_position_embeddings,
            self.config.d_model,
            self.padding_idx,
        )
        # 将位置编码嵌入对象移到指定设备（通常是 GPU）
        self.embed_positions.to(self.device)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        # 返回当前位置编码嵌入对象
        return self.embed_positions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
class PegasusDecoder(PegasusPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`PegasusDecoderLayer`]

    Args:
        config: PegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout  # 从配置中获取 dropout 概率
        self.layerdrop = config.decoder_layerdrop  # 从配置中获取层级丢弃率
        self.padding_idx = config.pad_token_id  # 从配置中获取填充符索引
        self.max_target_positions = config.max_position_embeddings  # 从配置中获取最大目标位置数
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 根据配置设置嵌入缩放因子

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens  # 如果提供了嵌入词表，直接使用，否则创建新的 nn.Embedding
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 使用 PegasusSinusoidalPositionalEmbedding 类创建位置嵌入
        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )

        # 使用 PegasusDecoderLayer 类创建多层解码器层
        self.layers = nn.ModuleList([PegasusDecoderLayer(config) for _ in range(config.decoder_layers)])

        # 使用 nn.LayerNorm 创建层归一化模块
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False  # 初始化梯度检查点为 False

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回输入嵌入

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入嵌入

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
        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        # 根据新的位置嵌入数量重新设置位置嵌入矩阵
        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            self.config.max_position_embeddings,
            self.config.d_model,
            self.padding_idx,
        )
        self.embed_positions.to(self.device)  # 将位置嵌入移到指定设备上

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        return self.embed_positions  # 返回位置嵌入矩阵
    # 定义神经网络模型的前向传播方法，接受多个输入参数用于模型推理
    def forward(
        self,
        input_ids=None,                    # 输入的 token IDs，用于模型输入
        attention_mask=None,               # 注意力遮罩，指示哪些位置是padding或特殊token
        encoder_hidden_states=None,        # 编码器的隐藏状态，用于某些模型（如BERT）
        encoder_attention_mask=None,       # 编码器的注意力遮罩，指示编码器输入中padding位置
        head_mask=None,                    # 多头注意力机制中的头部掩码，控制哪些头部被屏蔽
        cross_attn_head_mask=None,         # 跨注意力头的掩码，用于控制哪些跨注意力头被屏蔽
        past_key_values=None,              # 用于存储过去的键值对，提高解码效率
        inputs_embeds=None,                # 直接提供的嵌入表示，而不是通过输入ID计算得到
        use_cache=None,                    # 控制是否使用缓存加速解码
        output_attentions=None,            # 控制是否输出注意力权重
        output_hidden_states=None,         # 控制是否输出所有隐藏状态
        return_dict=None,                  # 控制是否以字典形式返回结果
# 使用 `add_start_docstrings` 装饰器为 PegasusModel 类添加文档字符串，描述它是一个不带特定头部的 PEGASUS 模型的裸输出版本。
# 引用 PEGASUS_START_DOCSTRING，补充完整文档字符串内容。

class PegasusModel(PegasusPreTrainedModel):
    # 定义了共享权重的键列表，这些权重在编码器和解码器的嵌入层之间共享
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: PegasusConfig):
        super().__init__(config)

        # 初始化填充索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建一个共享的嵌入层对象，用于词嵌入
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器对象，并传入共享的嵌入层对象
        self.encoder = PegasusEncoder(config, self.shared)
        self.decoder = PegasusDecoder(config, self.shared)

        # 初始化模型权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回共享的嵌入层对象，用于模型输入的词嵌入
        return self.shared

    def set_input_embeddings(self, value):
        # 设置新的共享嵌入层对象，并更新编码器和解码器的嵌入层
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        # 返回编码器对象
        return self.encoder

    def get_decoder(self):
        # 返回解码器对象
        return self.decoder

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        调整模型的位置嵌入矩阵大小，如果 `new_num_position_embeddings != config.max_position_embeddings`。

        参数:
            new_num_position_embeddings (`int`):
                新的位置嵌入数量。如果位置嵌入是学习的，则增加大小将在末尾添加新初始化的向量，
                减小大小将从末尾删除向量。如果位置嵌入不是学习的（如正弦位置嵌入），增加大小将
                在末尾添加正确的向量，减小大小将从末尾删除向量。
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        # 调整编码器和解码器的位置嵌入大小
        self.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        返回位置嵌入矩阵
        """
        return (self.encoder.get_position_embeddings(), self.decoder.get_position_embeddings())

    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义名为 forward 的方法，用于进行模型的前向传播
    def forward(
        # 输入序列的标识符张量，可选参数，默认为 None
        input_ids: Optional[torch.Tensor] = None,
        # 注意力遮罩张量，可选参数，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器输入序列的标识符张量，可选参数，默认为 None
        decoder_input_ids: Optional[torch.Tensor] = None,
        # 解码器输入的注意力遮罩张量，可选参数，默认为 None
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # 头遮罩张量，可选参数，默认为 None
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头遮罩张量，可选参数，默认为 None
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头遮罩张量，可选参数，默认为 None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出的元组张量，可选参数，默认为 None
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        # 过去键值的元组张量，可选参数，默认为 None
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        # 输入的嵌入张量，可选参数，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器输入的嵌入张量，可选参数，默认为 None
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        # 是否使用缓存，布尔型可选参数，默认为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力，布尔型可选参数，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，布尔型可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，布尔型可选参数，默认为 None
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    "The PEGASUS Model with a language modeling head. Can be used for summarization.", PEGASUS_START_DOCSTRING
)
class PegasusForConditionalGeneration(PegasusPreTrainedModel):
    # 在模型类上添加文档字符串，说明它是带有语言建模头部的PEGASUS模型，可用于摘要生成

    base_model_prefix = "model"
    # 定义基础模型的前缀为 "model"

    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 定义在加载模型时忽略的键列表，此处为 ["final_logits_bias"]

    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    # 定义权重共享的键列表，包括编码器和解码器的嵌入权重以及语言模型头部的权重

    def __init__(self, config: PegasusConfig):
        super().__init__(config)
        # 调用父类的初始化方法

        self.model = PegasusModel(config)
        # 实例化一个PEGASUS模型，使用给定的配置参数

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 注册一个缓冲区（buffer），用零填充的张量，用于模型的最终对数偏置

        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # 初始化语言建模头部，使用线性层将输入特征维度映射到词汇表大小的输出维度，无偏置项

        # Initialize weights and apply final processing
        self.post_init()
        # 调用后处理函数，初始化权重并进行最终处理

    def get_encoder(self):
        # 返回模型的编码器
        return self.model.get_encoder()

    def get_decoder(self):
        # 返回模型的解码器
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调整词嵌入矩阵的大小，继承自父类的方法

        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调用父类的方法，获取调整后的新词嵌入矩阵

        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        # 调整最终对数偏置的大小以匹配新的词嵌入矩阵大小

        return new_embeddings
        # 返回调整后的新词嵌入矩阵

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 调整最终对数偏置的大小，私有方法

        old_num_tokens = self.final_logits_bias.shape[-1]
        # 获取当前最终对数偏置的维度大小

        if new_num_tokens <= old_num_tokens:
            # 如果新的词嵌入数小于等于当前对数偏置数

            new_bias = self.final_logits_bias[:, :new_num_tokens]
            # 截取当前最终对数偏置，以匹配新的词嵌入数
        else:
            # 如果新的词嵌入数大于当前对数偏置数

            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            # 创建额外的对数偏置，用零填充

            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
            # 将当前对数偏置与额外对数偏置连接起来，以匹配新的词嵌入数

        self.register_buffer("final_logits_bias", new_bias)
        # 注册调整后的最终对数偏置

    def get_output_embeddings(self):
        # 返回语言建模头部
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置语言建模头部的新词嵌入
        self.lm_head = new_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        调整模型的位置嵌入矩阵，如果 `new_num_position_embeddings != config.max_position_embeddings`。

        Arguments:
            new_num_position_embeddings (`int`):
                新的位置嵌入数量。如果位置嵌入是可学习的，增加大小将在末尾添加新的初始化向量，减少大小将从末尾移除向量。
                如果位置嵌入不可学习（如正弦位置嵌入），增加大小将根据位置编码算法在末尾添加正确的向量，减少大小将从末尾移除向量。
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        # 设置配置文件中的最大位置嵌入数量

        self.model.encoder.resize_position_embeddings(new_num_position_embeddings)
        # 调整模型编码器的位置嵌入矩阵大小

        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)
        # 调整模型解码器的位置嵌入矩阵大小
    # 定义一个方法，返回编码器和解码器的位置嵌入矩阵
    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        # 调用模型对象的编码器和解码器的位置嵌入矩阵方法，返回二元组
        return (self.model.encoder.get_position_embeddings(), self.model.decoder.get_position_embeddings())

    # 将模型前向方法添加文档字符串，使用 PEGASUS_INPUTS_DOCSTRING 描述输入
    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    # 替换返回值文档字符串为 Seq2SeqLMOutput 类型，并使用 _CONFIG_FOR_DOC 配置类
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 在前向方法末尾添加 PEGASUS_GENERATION_EXAMPLE 文档字符串
    @add_end_docstrings(PEGASUS_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        encoder_outputs: Optional[ModelOutput] = None,
        decoder_attention_mask: torch.LongTensor = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
    
    
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
    
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    
        if labels is not None:
    
    
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
    
    
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
    
    
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
    
    
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    
    
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    
    
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
    
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
        # 如果使用了过去的键值（past_key_values），则截断decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 某些生成方法已经只传递了最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留旧的行为：只保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的输入信息作为字典
        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 将此项更改以避免缓存（可能用于调试）
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签右移一位以作为解码器的输入
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态无需重新排序 -> 它们始终保持不变
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        # 返回重新排序后的过去键值
        return reordered_past
# 从transformers.models.bart.modeling_bart.BartDecoderWrapper复制并修改为PegasusDecoderWrapper
class PegasusDecoderWrapper(PegasusPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在因果语言模型与EncoderDecoderModel框架结合使用时正确加载预训练检查点。
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化Pegasus解码器
        self.decoder = PegasusDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class PegasusForCausalLM(PegasusPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 使用PegasusDecoderWrapper来构建模型
        self.model = PegasusDecoderWrapper(config)

        # 定义LM头部线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入层
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 获取输出嵌入层（LM头部）
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入层（LM头部）
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 获取解码器
        return self.model.decoder

    def get_position_embeddings(self) -> nn.Embedding:
        """
        返回位置嵌入矩阵
        """
        return self.model.decoder.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        如果`new_num_position_embeddings != config.max_position_embeddings`，则调整模型的位置嵌入矩阵大小。

        参数:
            new_num_position_embeddings (`int`):
                新的位置嵌入数量。如果位置嵌入是学习的，则增加大小将在末尾添加新初始化的向量，而减小大小将从末尾删除向量。
                如果位置嵌入不是学习的（如正弦位置嵌入），增加大小将按照位置编码算法在末尾添加正确的向量，而减小大小将从末尾删除向量。
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.bart.modeling_bart.BartForCausalLM.forward复制并修改为Pegasus，facebook/bart-base->google/pegasus-large
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ID序列，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选参数，默认为None
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器的隐藏状态，可选参数，默认为None
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器的注意力遮罩，可选参数，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩，可选参数，默认为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头部遮罩，可选参数，默认为None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对，列表，可选参数，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，可选参数，默认为None
        labels: Optional[torch.LongTensor] = None,  # 标签，可选参数，默认为None
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选参数，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选参数，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典格式结果，可选参数，默认为None
    ):
        # 定义一个静态方法 `prepare_inputs_for_generation`，用于生成输入准备
        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
        ):
            # 如果没有给定注意力遮罩，则创建一个全为1的注意力遮罩，形状与输入的token ID序列相同
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape)

            # 如果有过去的键值对传入
            if past_key_values:
                past_length = past_key_values[0][0].shape[2]

                # 一些生成方法可能已经只传入了最后一个输入ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认行为：保留最后一个输入ID
                    remove_prefix_length = input_ids.shape[1] - 1

                # 移除前缀长度部分的输入ID序列
                input_ids = input_ids[:, remove_prefix_length:]

            # 返回一个字典，包含处理后的输入参数
            return {
                "input_ids": input_ids,  # 输入的token ID序列，不需要这个参数
                "attention_mask": attention_mask,  # 注意力遮罩
                "past_key_values": past_key_values,  # 过去的键值对
                "use_cache": use_cache,  # 是否使用缓存
            }

        # 静态方法 `_reorder_cache`，用于重新排序缓存的过去的键值对
        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            reordered_past = ()
            # 对每一层的过去状态重新排序
            for layer_past in past_key_values:
                reordered_past += (
                    # 对于每个过去状态，在给定的beam索引上进行索引选择
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            return reordered_past
```