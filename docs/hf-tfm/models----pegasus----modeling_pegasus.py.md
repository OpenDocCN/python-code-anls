# `.\transformers\models\pegasus\modeling_pegasus.py`

```py
# 导入所需的模块和函数
import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
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
from .configuration_pegasus import PegasusConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一些常量
_CHECKPOINT_FOR_DOC = "google/pegasus-large"
_CONFIG_FOR_DOC = "PegasusConfig"

# 列出所有预训练的 PEGASUS 模型
PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/pegasus-large",
    # See all PEGASUS models at https://huggingface.co/models?filter=pegasus
]

# 从 transformers.models.bart.modeling_bart 复制 shift_tokens_right 函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    将输入ID向右移动一个token。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将可能出现的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# 从 transformers.models.marian.modeling_marian 复制 PegasusSinusoidalPositionalEmbedding 类
class PegasusSinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.
    这个模块生成任意长度的正弦波位置嵌入。
    """

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    # 初始化权重，用于位置编码，与 XLM create_sinusoidal_embeddings 类似，但特征没有交错。
    # 余弦特征位于向量的第二半部分。[dim // 2:]
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        # 获取输出张量的形状信息
        n_pos, dim = out.shape
        # 创建位置编码矩阵，使用 numpy 数组
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 设置 requires_grad 为 False，以避免在 pytorch-1.8+ 中出现错误
        out.requires_grad = False
        # 计算标记位置，这里使用整除来检查奇偶性
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 将正弦部分赋值给out的前半部分
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        # 将余弦部分赋值给out的后半部分
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        # 分离出out参数，不计算梯度
        out.detach_()
        # 返回结果
        return out

    # 定义前向传播函数，接收输入的形状信息和过往键值对长度
    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 获取批次大小和序列长度
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置索引数组，为当前位置添加偏移量，使用张量
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的forward方法，传入位置信息的张量
        return super().forward(positions)
# 这是一个多头注意力层的实现，用于 Pegasus 模型
class PegasusAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        # 设置注意力层的基本参数，包括词嵌入维度、头数、dropout等
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查输入参数是否正确
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 定义用于计算Query、Key、Value的线性层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 对 Q、K、V 进行reshape，使其可以和多头注意力计算相匹配
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
PEGASUS_ATTENTION_CLASSES = {"eager": PegasusAttention}


# 这是 Pegasus 编码器层的实现
class PegasusEncoderLayer(nn.Module):
    def __init__(self, config: PegasusConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 创建多头注意力层
        self.self_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    # 定义函数，输入为hidden_states，输出为torch.Tensor类型
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.FloatTensor,
                layer_head_mask: torch.FloatTensor,
                output_attentions: bool) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            # 输入的隐藏状态，形状为`(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            # 注意力掩码，形状为`(batch, 1, tgt_len, src_len)`，通过负值表示填充元素
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            # 给定层中注意力头的掩码，大小为`(encoder_attention_heads,)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            # 是否返回所有注意力层的注意力张量，默认为假。更多细节参见返回张量中的`attentions`
        """

        residual = hidden_states
        # 将输入状态标准化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 使用self_attn方法进行注意力计算
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 在隐藏状态上执行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states

        residual = hidden_states
        # 将输入状态标准化
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数进行全连接层计算
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 在隐藏状态上执行激活函数的dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 通过全连接层计算隐藏状态
        hidden_states = self.fc2(hidden_states)
        # 在隐藏状态上执行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 如果隐藏状态的数据类型为torch.float16，且存在无穷大或NaN值
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 对隐藏状态进行值截断
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        # 如果设置了output_attentions为真
        if output_attentions:
            # 将注意力权重添加到输出中
            outputs += (attn_weights,)

        return outputs
        # 返回outputs
# 这个类是 PegasusDecoderLayer 类，继承自 nn.Module，负责实现 Pegasus 模型中解码器层的功能
class PegasusDecoderLayer(nn.Module):
    def __init__(self, config: PegasusConfig):
        # 调用父类的构造函数
        super().__init__()
        # 设置 embed_dim 属性为 config.d_model
        self.embed_dim = config.d_model

        # 创建自注意力模块，使用 PEGASUS_ATTENTION_CLASSES 指定的注意力实现类
        self.self_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 设置 dropout 属性为 config.dropout
        self.dropout = config.dropout
        # 根据 config.activation_function 获取激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置 activation_dropout 属性为 config.activation_dropout
        self.activation_dropout = config.activation_dropout

        # 创建自注意力层标准化模块
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建编码器注意力模块，使用 PEGASUS_ATTENTION_CLASSES 指定的注意力实现类
        self.encoder_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 创建编码器注意力层标准化模块
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建两个全连接层用于前馈网络
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 创建最终层标准化模块
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 定义前向传播方法
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
        # 在此处实现前向传播逻辑

# 这是 PegasusPreTrainedModel 类，继承自 PreTrainedModel，是 Pegasus 预训练模型的基类
class PegasusPreTrainedModel(PreTrainedModel):
    # 设置配置类为 PegasusConfig
    config_class = PegasusConfig
    # 设置模型前缀为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重的方法
    def _init_weights(self, module):
        # 获取初始化标准差
        std = self.config.init_std
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 用正态分布初始化权重，标准差为 std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏差项，将其初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 PegasusSinusoidalPositionalEmbedding 类型
        elif isinstance(module, PegasusSinusoidalPositionalEmbedding):
            # 不做任何操作
            pass
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 用正态分布初始化权重，标准差为 std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，将其对应的权重初始化为 0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# 这是 PEGASUS_START_DOCSTRING 变量，包含了 Pegasus 模型的文档字符串
PEGASUS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    # 使用此类作为常规的 PyTorch 模块，并参考 PyTorch 文档以了解所有与一般使用和行为相关的事项。
    # 参数说明：
    #     config ([`PegasusConfig`]):
    #         包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只加载配置信息。
    #         可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""
PEGASUS_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
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
    ```py
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

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([PegasusEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        调整模型的位置嵌入矩阵大小，如果 `new_num_position_embeddings != config.max_position_embeddings`。

        参数:
            new_num_position_embeddings (`int`):
                新的位置嵌入数量。如果位置嵌入是可学习的，增加大小将在末尾添加新初始化的向量，而减小大小将从末尾删除向量。
                如果位置嵌入不是可学习的（例如，正弦位置嵌入），增加大小将按照位置编码算法在末尾添加正确的向量，而减小
                大小将从末尾删除向量。
        """
        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        # 设置模型的最大位置嵌入数量
        self.config.max_position_embeddings = new_num_position_embeddings

        # 根据新的最大位置嵌入数量创建正弦位置嵌入对象
        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            self.config.max_position_embeddings,
            self.config.d_model,
            self.padding_idx,
        )
        # 将位置嵌入对象移到指定的设备上
        self.embed_positions.to(self.device)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        返回位置嵌入矩阵
        """
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
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        
        # If embed_tokens is provided, set it as the embed_tokens attribute else create a new nn.Embedding
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # Create sinusoidal positional embeddings
        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        
        # Create a list of decoder layers based on config.decoder_layers
        self.layers = nn.ModuleList([PegasusDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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

        # Update the embed_positions with new_num_position_embeddings
        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            self.config.max_position_embeddings,
            self.config.d_model,
            self.padding_idx,
        )
        self.embed_positions.to(self.device)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        return self.embed_positions
    # 实现 Transformer 模型的前向传播过程
    def forward(
        # 输入 token 的 IDs
        input_ids=None,
        # 注意力遮罩，指示哪些位置需要注意哪些位置不需要
        attention_mask=None,
        # 编码器隐藏状态，用于跨层注意力机制
        encoder_hidden_states=None,
        # 编码器注意力遮罩，指示哪些编码器位置需要注意哪些位置不需要
        encoder_attention_mask=None,
        # 头部遮罩，用于控制注意力头部的屏蔽
        head_mask=None,
        # 交叉注意力头部的遮罩，用于跨层和跨模态注意力
        cross_attn_head_mask=None,
        # 过去的键值对，用于允许每个时间步重用上一层的键值对
        past_key_values=None,
        # 输入的嵌入向量，如果提供，则忽略 input_ids
        inputs_embeds=None,
        # 是否使用缓存，用于存储中间结果，以便下一次调用时重用
        use_cache=None,
        # 是否输出注意力权重
        output_attentions=None,
        # 是否输出隐藏状态
        output_hidden_states=None,
        # 是否返回字典形式的输出
        return_dict=None,
```  
# 将PEGASUS模型定义为输出原始隐藏状态，没有特定的顶部头部
# 继承PegasusPreTrainedModel类
class PegasusModel(PegasusPreTrainedModel):
    # 绑定权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化方法
    def __init__(self, config: PegasusConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 获取填充token的索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享的词嵌入层
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 初始化编码器和解码器
        self.encoder = PegasusEncoder(config, self.shared)
        self.decoder = PegasusDecoder(config, self.shared)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入的嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入的嵌入
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 调整位置嵌入
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
        # 设置最大位置嵌入数
        self.config.max_position_embeddings = new_num_position_embeddings
        self.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.decoder.resize_position_embeddings(new_num_position_embeddings)

    # 获取位置嵌入
    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.encoder.get_position_embeddings(), self.decoder.get_position_embeddings())

    # 将文档字符串添加到模型前向函数
    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个前向传播函数，接受多个输入参数并返回相关输出
    def forward(
        # 输入序列的id张量，默认为空值
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码张量，默认为空值
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器输入序列的id张量，默认为空值
        decoder_input_ids: Optional[torch.Tensor] = None,
        # 解码器注意力掩码张量，默认为空值
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # 头掩码张量，用于屏蔽特定头，默认为空值
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头掩码张量，默认为空值
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头掩码张量，默认为空值
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出的元组张量，默认为空值
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        # 过去的键-值张量元组，默认为空值
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        # 输入的嵌入张量，默认为空值
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器输入的嵌入张量，默认为空值
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        # 是否使用缓存，默认为空值
        use_cache: Optional[bool] = None,
        # 是否输出注意力，默认为空值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为空值
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，默认为空值
        return_dict: Optional[bool] = None,
# 导入必要的模块和函数
@add_start_docstrings(
    "The PEGASUS Model with a language modeling head. Can be used for summarization.", PEGASUS_START_DOCSTRING
)
# 定义 PegasusForConditionalGeneration 类，继承自 PegasusPreTrainedModel 类
class PegasusForConditionalGeneration(PegasusPreTrainedModel):
    # 指定基础模型参数的前缀
    base_model_prefix = "model"
    # 在加载缺失键时忽略的键列表
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 需要共享权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法
    def __init__(self, config: PegasusConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 PegasusModel 对象并赋值给 self.model
        self.model = PegasusModel(config)
        # 创建 final_logits_bias 属性并初始化为零向量
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建 lm_head 属性并初始化为一个线性层
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 的大小
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的方法调整 token embeddings 的大小，并得到新的 embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整 final_logits_bias 的大小以匹配新的 embeddings 大小
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        # 返回新的 embeddings
        return new_embeddings

    # 调整 final_logits_bias 的大小
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取原始的 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于原始的 token 数量
        if new_num_tokens <= old_num_tokens:
            # 则截取 final_logits_bias 的子集
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 否则创建额外的偏置并将其连接到 final_logits_bias 的末尾
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 更新 final_logits_bias
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出 embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 调整位置 embeddings 的大小
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
        # 更新模型配置中的最大位置 embeddings 数量
        self.config.max_position_embeddings = new_num_position_embeddings
        # 调整编码器的位置 embeddings 大小
        self.model.encoder.resize_position_embeddings(new_num_position_embeddings)
        # 调整解码器的位置 embeddings 大小
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)
    # 获取位置嵌入矩阵
    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        # 返回编码器和解码器的位置嵌入
        return (self.model.encoder.get_position_embeddings(), self.model.decoder.get_position_embeddings())
    
    # 在模型前向传播上添加开始和结束文档字符串
    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(PEGASUS_GENERATION_EXAMPLE)
    def forward(
        self,
        # 输入 ID 序列
        input_ids: Optional[torch.Tensor] = None,
        # 输入的注意力掩码
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器输入 ID 序列
        decoder_input_ids: Optional[torch.Tensor] = None,
        # 解码器注意力掩码
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # 编码器注意力掩码
        head_mask: Optional[torch.Tensor] = None,
        # 解码器注意力掩码
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        # 过去的关键值对
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        # 输入嵌入
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器输入嵌入
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        # 标签
        labels: Optional[torch.Tensor] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典
        return_dict: Optional[bool] = None,
    ):
        ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """

        # 设置返回的字典，如果未提供则使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                # 如果提供了标签，则将`use_cache`参数设置为`False`
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # 如果未提供解码器输入标识或输入嵌入，则根据标签创建解码器输入标识
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # 向模型传递参数并获取输出
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
        
        # 计算语言建模的logits和最终logits的偏置
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # 计算掩码语言建模的损失
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不返回字典，则返回相应的输出
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回带有loss、logits和其他相关信息的字典
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
    # 准备用于生成的输入数据，包括解码器输入、过去键值、注意力掩码、头部掩码等
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
        # 如果使用了过去的键值，根据情况截取解码器输入的部分
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果解码器输入的长度超过了过去键值的长度，则截取掉前面多余的部分
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认只保留最后一个输入 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回包含各种输入数据的字典
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # 根据标签准备解码器的输入 ID
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 重新排序缓存中的数据，以适应 Beam Search 等情况
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 交叉注意力状态无需重新排序 -> 它们始终保持不变
            reordered_past += (
                # 重新排序过去键值
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],  # 保持不变
            )
        return reordered_past
# 从transformers.models.bart.modeling_bart.BartDecoderWrapper复制，将Bart更改为Pegasus
class PegasusDecoderWrapper(PegasusPreTrainedModel):
    """
    此包装类是一个辅助类，用于在因果语言模型与[`EncoderDecoderModel`]框架结合使用时正确加载预训练检查点。
    """

    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 创建PegasusDecoder对象
        self.decoder = PegasusDecoder(config)

    def forward(self, *args, **kwargs):
        # 调用PegasusDecoder的forward方法
        return self.decoder(*args, **kwargs)


class PegasusForCausalLM(PegasusPreTrainedModel):
    # 被绑定权重的键列表
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深拷贝配置
        config = copy.deepcopy(config)
        # 将解码器标记为True，编码器解码器标记为False
        config.is_decoder = True
        config.is_encoder_decoder = False
        # 调用父类初始化方法
        super().__init__(config)
        # 创建PegasusDecoderWrapper对象
        self.model = PegasusDecoderWrapper(config)

        # 线性层，用于生成预测的下一个token
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型的输入嵌入
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置模型的输入嵌入
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回模型的输出嵌入
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置模型的输出嵌入
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 返回解码器
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
                新的位置嵌入数量。如果位置嵌入是可学习的，增加大小将在末尾添加新初始化的向量，而减小大小将从末尾删除向量。
                如果位置嵌入不是可学习的（如正弦位置嵌入），增加大小将按照位置编码算法在末尾添加正确的向量，而减小大小将从末尾删除向量。
        """
        # 更新配置中的最大位置嵌入数
        self.config.max_position_embeddings = new_num_position_embeddings
        # 调整位置嵌入矩阵大小
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.bart.modeling_bart.BartForCausalLM.forward复制，将Bart更改为Pegasus，facebook/bart-base更改为google/pegasus-large
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入数据的 token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩模
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器的隐藏状态
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器的注意力掩模
        head_mask: Optional[torch.Tensor] = None,  # 头掩模
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头掩模
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        labels: Optional[torch.LongTensor] = None,  # 标签数据
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    # 准备生成的输入数据
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 如果没有给定注意力掩模，则创建一个全 1 的注意力掩模
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        # 第一步时，解码器缓存状态为空
        return {
            "input_ids": input_ids,  # 编码器输出已经被定义，不需要输入 input_ids
            "attention_mask": attention_mask,  # 注意力掩模
            "past_key_values": past_key_values,  # 过去的键值对
            "use_cache": use_cache,  # 是否使用缓存
        }

    @staticmethod
    # 重新排序缓存
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```