# `.\transformers\models\m2m_100\modeling_m2m_100.py`

```
# 设置编码格式为utf-8
# 版权声明
# 根据 Apache License, Version 2.0 许可证规定
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则在“AS IS”基础上分发的软件
# 没有任何种类的保证或条件，无论是明示的还是暗示的
# 对于特定语言支持权限和限制，请查看许可证
""" PyTorch M2M100 model."""
# 导入所需的模块
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
# 导入所需的模块和函数
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
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
from .configuration_m2m_100 import M2M100Config

# 记录日志
logger = logging.get_logger(__name__)

# 文档中的配置和检查点信息
_CONFIG_FOR_DOC = "M2M100Config"
_CHECKPOINT_FOR_DOC = "facebook/m2m100_418M"


M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/m2m100_418M",
    # 查看所有 M2M100 模型，请访问 https://huggingface.co/models?filter=m2m_100
]


# 从transformers.models.bart.modeling_bart.shift_tokens_right中复制函数
# 将输入id向右移动一个标记
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的-100值替换为`pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# 从输入id中创建位置id
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length = 0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    # 仔细平衡这里的一系列转换和类型��换，既适用于ONNX导出又适用于XLA
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim = 1).type_as(mask) + past_key_values_length) * mask
    # 将增量索引转换为长整型，并加上填充索引
    return incremental_indices.long() + padding_idx
class M2M100SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 初始化函数，设置模块参数
        super().__init__()
        # 设置偏移量
        self.offset = 2
        # 设置嵌入维度
        self.embedding_dim = embedding_dim
        # 设置填充索引
        self.padding_idx = padding_idx
        # 调用make_weights方法创建权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 创建权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 如果已经存在权重，将新权重转换为相同的dtype和device
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 将权重注册为模块的缓冲区
        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算频率
        emb = math.log(10000) / (half_dim - 1)
        # 计算正弦和余弦部分的嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # 如果嵌入维度是奇数，进行零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            # 如果有填充索引，将对应行的嵌入设置为零向量
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        if input_ids is not None:
            # 如果输入id不为None，获取批次大小和序列长度
            bsz, seq_len = input_ids.size()
            # 从输入标记id创建位置id，任何填充的标记仍然保持填充状态
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            # 如果输入id为None，获取批次大小和序列长度
            bsz, seq_len = inputs_embeds.size()[:-1]
            # 从输入嵌入创建位置id
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # 如果需要扩展嵌入
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            # 创建新的权重
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 返回嵌入张量
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    # 从给定的输入嵌入中创建位置 ID，因为直接提供了嵌入，无法推断哪些是填充的，因此生成顺序位置 ID
    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成位置 ID，从填充索引加1开始到序列长度加上填充索引加1为止
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置 ID 的维度扩展为输入嵌入的形状，并保持连续
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
# 从transformers.models.bart.modeling_bart.BartAttention复制而来，将Bart替换为M2M100
class M2M100Attention(nn.Module):
    """从论文'Attention Is All You Need'中的多头注意力机制派生的类"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[M2M100Config] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 确保embed_dim必须能被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能够被num_heads整除 (当前 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性变换层，用于计算Q、K、V及输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重塑成(batch_size, seq_len, num_heads, head_dim)的形状，并进行维度转置
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,



# 从transformers.models.mbart.modeling_mbart.MBartEncoderLayer复制而来，将MBart替换为M2M100, MBART替换为M2M100
class M2M100EncoderLayer(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()
        self.embed_dim = config.d_model

        # 使用M2M100配置初始化自注意力层和LayerNorm
        self.self_attn = M2M100_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 线性变换层和LayerNorm，用于前馈神经网络部分
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

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
        # Save a copy of the input hidden_states for later residual connection
        residual = hidden_states
        # Apply layer normalization to the input hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Perform self-attention mechanism on the input hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # Apply dropout with a specified dropout rate to the hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # Add the residual connection to the processed hidden_states
        hidden_states = residual + hidden_states

        # Save a copy of the intermediate hidden_states for another residual connection
        residual = hidden_states
        # Apply layer normalization to the intermediate hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        # Apply an activation function to hidden_states followed by a linear transformation
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # Apply dropout with a specified activation dropout rate to hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # Apply another linear transformation to hidden_states
        hidden_states = self.fc2(hidden_states)
        # Apply dropout with a specified dropout rate to hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # Add the second residual connection to the processed hidden_states
        hidden_states = residual + hidden_states

        # Check for NaN or infinite values in hidden_states if dtype is torch.float16
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # Clamp the values of hidden_states to prevent overflow or undefined results
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Store the final processed hidden_states in outputs
        outputs = (hidden_states,)

        # Append attention weights to outputs if output_attentions is True
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义一个全局变量 M2M100_ATTENTION_CLASSES，存储了 M2M100Attention 的字典
M2M100_ATTENTION_CLASSES = {"eager": M2M100Attention}

# 定义一个 M2M100DecoderLayer 类，继承自 nn.Module
# 该类用于 M2M100 解码器的一个层，初始化时需要传入一个 M2M100Config 参数
class M2M100DecoderLayer(nn.Module):
    # 定义初始化函数
    def __init__(self, config: M2M100Config):
        # 调用父类初始化方法
        super().__init__()
        # 设置embed_dim为d_model配置值
        self.embed_dim = config.d_model
        # 初始化self_attn属性，使用M2M100_ATTENTION_CLASSES中的配置参数
        self.self_attn = M2M100_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 设置self.dropout为config中的dropout值
        self.dropout = config.dropout
        # 设置activation_fn为ACT2FN中的配置参数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置activation_dropout为config的activation_dropout值
        self.activation_dropout = config.activation_dropout
        # 初始化self_attn_layer_norm，设置为nn.LayerNorm类型，传入参数为embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化encoder_attn，使用M2M100_ATTENTION_CLASSES中的配置参数
        self.encoder_attn = M2M100_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 初始化encoder_attn_layer_norm，设置为nn.LayerNorm类型，传入参数为embed_dim
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化fc1，设置为nn.Linear类型，传入参数为embed_dim, config.decoder_ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 初始化fc2，设置为nn.Linear类型，传入参数为config.decoder_ffn_dim, embed_dim
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 初始化final_layer_norm，设置为nn.LayerNorm类型，传入参数为embed_dim
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
# 定义M2M100PreTrainedModel类，继承自PreTrainedModel类
class M2M100PreTrainedModel(PreTrainedModel):
    # 设置config_class属性为M2M100Config
    config_class = M2M100Config
    # 设置base_model_prefix属性为"model"
    base_model_prefix = "model"
    # 设置supports_gradient_checkpointing属性为True
    supports_gradient_checkpointing = True
    # 设置_no_split_modules属性为["M2M100Attention"]
    _no_split_modules = ["M2M100Attention"]

    # 定义初始化权重方法
    def _init_weights(self, module):
        # 设置std为self.config.init_std
        std = self.config.init_std
        # 如果module是nn.Linear类型
        if isinstance(module, nn.Linear):
            # 设置module.weight数据为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果module.bias不为空，设置module.bias数据为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果module是nn.Embedding类型
        elif isinstance(module, nn.Embedding):
            # 设置module.weight数据为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果module.padding_idx不为空，将对应位置的数据设置为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


M2M_100_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    # 使用此类像普通的 PyTorch 模型一样，并参考 PyTorch 文档了解与一般使用和行为相关的所有事项。
    # 参数：
    #     config ([`M2M100Config`]):
    #         包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只加载配置。
    #         请查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

M2M_100_GENERATION_EXAMPLE = r"""
    Translation example:

    ```python
    >>> from transformers import AutoTokenizer, M2M100ForConditionalGeneration

    >>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

    >>> text_to_translate = "Life is like a box of chocolates"
    >>> model_inputs = tokenizer(text_to_translate, return_tensors="pt")

    >>> # translate to French
    >>> gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("fr"))
    >>> print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
    ```
"""

M2M_100_INPUTS_DOCSTRING = r"""
"""


class M2M100Encoder(M2M100PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`M2M100EncoderLayer`].

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([M2M100EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class M2M100Decoder(M2M100PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`M2M100DecoderLayer`]

    Args:
        config: M2M100Config
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化类实例，接受配置和嵌入词元作为参数
    def __init__(self, config: M2M100Config, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类属性
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 创建词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 如果传入了额外的嵌入词元，使用它来初始化词嵌入层的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 创建 sinusoidal 位置编码层
        self.embed_positions = M2M100SinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        
        # 创建解码器层列表
        self.layers = nn.ModuleList([M2M100DecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建层归一化层
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 初始化渐变检查点标志
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器添加文档字符串到 M2M100Model 类，描述该模型输出原始隐藏状态，不包含特定的头结构
# 并引入 M2M_100_START_DOCSTRING 文档字符串
@add_start_docstrings(
    "The bare M2M100 Model outputting raw hidden-states without any specific head on top.",
    M2M_100_START_DOCSTRING,
)
# 定义 M2M100Model 类，继承自 M2M100PreTrainedModel
class M2M100Model(M2M100PreTrainedModel):
    # 定义 _tied_weights_keys 成员变量，表示需要绑定权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 定义初始化方法，传入配置参数 config：M2M100Config
    def __init__(self, config: M2M100Config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 获取配置参数中的 padding_idx 和 vocab_size 并赋值给变量
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享的嵌入层，vocab_size 为词汇表大小，config.d_model 为模型维度，padding_idx 为填充索引
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器，传入配置参数和共享嵌入层
        self.encoder = M2M100Encoder(config, self.shared)
        self.decoder = M2M100Decoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.shared

    # 定义设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 定义绑定权重的方法
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 定义获取编码器的方法
    def get_encoder(self):
        return self.encoder

    # 定义获取解码器的方法
    def get_decoder(self):
        return self.decoder

    # 使用装饰器添加文档字符串到 forward 方法，描述其输入和输出
    @add_start_docstrings_to_model_forward(M2M_100_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播方法 forward，接受多个输入参数并返回输出结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_attentions 为 None，则使用配置中的 output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_hidden_states 为 None，则使用配置中的 output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果 use_cache 为 None，则使用配置中的 use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict

        if encoder_outputs is None:
            # 如果 encoder_outputs 为 None，则通过 encoder 输入获取 encoder_outputs
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            # 如果 return_dict 为 True，且 encoder_outputs 不是 BaseModelOutput 类型，则将其封装成 BaseModelOutput 类型

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取解码器的输出

        if not return_dict:
            # 如果 return_dict 为 False，则返回 decoder_outputs 和 encoder_outputs 的组合
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        # 返回 Seq2SeqModelOutput 类型，包含解码器和编码器输出
# M2M100ForConditionalGeneration 是一个继承自 M2M100PreTrainedModel 的模型类
# 它实现了一个能够进行条件生成的 M2M100 模型
@add_start_docstrings(
    "The M2M100 Model with a language modeling head. Can be used for summarization.", M2M_100_START_DOCSTRING
)
class M2M100ForConditionalGeneration(M2M100PreTrainedModel):
    # 模型的基本前缀
    base_model_prefix = "model"
    # 需要绑定权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: M2M100Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 M2M100 模型实例
        self.model = M2M100Model(config)
        # 创建一个线性层用于生成语言模型的输出
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取模型的编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取模型的解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 获取输出嵌入矩阵
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入矩阵
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 定义模型的前向传播方法
    @add_start_docstrings_to_model_forward(M2M_100_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(M2M_100_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
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
        # 这是 M2M100ForConditionalGeneration 模型的前向传播方法
        # 它接受各种输入参数,并返回一个 Seq2SeqLMOutput 对象
        # 这个方法主要是调用了 self.model 的前向传播方法,并对输出进行了进一步处理
        pass
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        # 根据参数 return_dict 是否为 None 选择是否使用配置参数中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                # 对标签进行右移以作为 decoder_input_ids
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # 通过模型生成输出
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
        # 获取语言模型的 logits
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用 PyTorch 张量
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            # 计算 masked_lm_loss
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            # 根据情况返回输出
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回模型输出结果对象
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
        # 如果传入了过去的关键值（past_key_values），则计算过去的长度
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 如果当前输入的decoder_input_ids长度比过去的长度长
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则默认移除前缀长度为decoder_input_ids的长度减1
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 截取decoder_input_ids以去除前缀部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回包含相关信息的字典
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 修改此处以避免缓存（可能用于调试）
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排列缓存信息中的过去关键值
        reordered_past = ()
        for layer_past in past_key_values:
            # 使用beam_idx重新排序每个layer的过去状态
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```