# `.\transformers\models\plbart\modeling_plbart.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 UCLA NLP、Facebook AI Research Team 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关特定语言的权限和限制
""" PyTorch PLBART 模型。"""
# 导入所需的库和模块
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "uclanlp/plbart-base"
_CONFIG_FOR_DOC = "PLBartConfig"

# 预训练 PLBART 模型存档列表
PLBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uclanlp/plbart-base",
    "uclanlp/plbart-cs-java",
    "uclanlp/plbart-multi_task-all",
    # 查看所有 PLBART 模型：https://huggingface.co/models?filter=plbart
]

# 从 transformers.models.mbart.modeling_mbart.shift_tokens_right 复制的函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    将输入的标识向右移动一个标识，并包装最后一个非填充标识（<LID> 标识）。注意，与其他类似 Bart 模型不同，MBart 没有单个 `decoder_start_token_id`。
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 用 `pad_token_id` 替换标签中可能存在的 -100 值
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens
# 从transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding复制而来，将Bart->PLBart
class PLBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # PLBart被设置为如果指定了padding_idx，则通过2偏移嵌入id，并相应调整num_embeddings。其他模型没有这个hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


# 从transformers.models.bart.modeling_bart.BartAttention复制而来，将Bart->PLBart
class PLBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

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
# 从transformers.models.bart.modeling_bart.BartEncoderLayer复制而来，将Bart->PLBart, BART->PLBART
class PLBartEncoderLayer(nn.Module):
    # 初始化函数，接受一个 PLBartConfig 对象作为参数
    def __init__(self, config: PLBartConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model

        # 创建自注意力层对象，根据配置中的实现方式选择不同的实现类
        self.self_attn = PLBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        # 创建自注意力层的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的 dropout 概率
        self.activation_dropout = config.activation_dropout
        # 创建全连接层 fc1，输入维度为嵌入维度，输出维度为配置中的编码器前馈网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 创建全连接层 fc2，输入维度为配置中的编码器前馈网络维度，输出维度为嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 创建最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，接受隐藏状态、注意力掩码、层头掩码和是否输出注意力权重作为参数
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
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
        # 保存输入 hidden_states 作为残差连接的基准
        residual = hidden_states
        # 使用 self_attn 层处理 hidden_states，得到输出 hidden_states 和注意力权重 attn_weights
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接的结果与当前 hidden_states 相加
        hidden_states = residual + hidden_states
        # 对相加后的 hidden_states 进行 LayerNorm 处理
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存当前 hidden_states 作为残差连接的基准
        residual = hidden_states
        # 使用激活函数 activation_fn 处理 hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对处理后的 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用第二个全连接层 fc2 处理 hidden_states
        hidden_states = self.fc2(hidden_states)
        # 对处理后的 hidden_states 进行 dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接的结果与当前 hidden_states 相加
        hidden_states = residual + hidden_states
        # 对相加后的 hidden_states 进行 LayerNorm 处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果 hidden_states 的数据类型为 torch.float16，并且包含无穷大或 NaN 值，则进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组，包含处理后的 hidden_states
        outputs = (hidden_states,)

        # 如果需要返回 attentions，则将 attn_weights 加入输出元组
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义 PLBart 的注意力类别字典，包含 "eager" 对应 PLBartAttention 类
PLBART_ATTENTION_CLASSES = {"eager": PLBartAttention}

# 从 transformers.models.bart.modeling_bart.BartDecoderLayer 复制代码，并将 Bart->PLBart, BART->PLBART
class PLBartDecoderLayer(nn.Module):
    def __init__(self, config: PLBartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 初始化自注意力层
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

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化编码器注意力层
        self.encoder_attn = PLBART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

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
# 从 transformers.models.bart.modeling_bart.BartClassificationHead 复制代码，并将 Bart->PLBart
class PLBartClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

# 定义 PLBartPreTrainedModel 类，继承自 PreTrainedModel
class PLBartPreTrainedModel(PreTrainedModel):
    config_class = PLBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # 定义不需要拆分的模块列表
    _no_split_modules = ["PLBartDecoderLayer", "PLBartEncoderLayer"]

    # 初始化模型权重
    def _init_weights(self, module):
        # 获取初始化标准差
        std = self.config.init_std
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，将其对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# PLBART_START_DOCSTRING 是一个包含模型文档字符串的原始字符串，用于描述 PLBart 模型的继承关系和参数说明
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

# PLBART_GENERATION_EXAMPLE 是一个包含模型生成示例的原始字符串，展示了如何使用 PLBart 进行遮蔽填充
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
    ```py
"""

# PLBART_INPUTS_DOCSTRING 是一个空字符串，用于描述 PLBart 模型的输入
PLBART_INPUTS_DOCSTRING = r"""
"""


# 以下是从 transformers.models.bart.modeling_bart.BartEncoder 复制并修改为 PLBartEncoder 的类定义
# PLBartEncoder 是一个 Transformer 编码器，由 config.encoder_layers 个自注意力层组成，每个层都是 PLBartEncoderLayer
class PLBartEncoder(PLBartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PLBartEncoderLayer`].

    Args:
        config: PLBartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化编码器类，接受配置和嵌入标记作为参数
    def __init__(self, config: PLBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置丢弃率和编码器层丢弃率
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 获取嵌入维度
        embed_dim = config.d_model
        # 获取填充索引
        self.padding_idx = config.pad_token_id
        # 获取最大源位置
        self.max_source_positions = config.max_position_embeddings
        # 设置嵌入缩放
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 初始化嵌入标记
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 如果传入了嵌入标记，则使用传入的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 初始化位置嵌入
        self.embed_positions = PLBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # 初始化编码器层列表
        self.layers = nn.ModuleList([PLBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        # 根据配置选择是否使用 Flash Attention 2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 根据配置选择是否使用 SDPA
        self._use_sdpa = config._attn_implementation == "sdpa"
        # 初始化嵌入层的 LayerNorm
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # 关闭梯度检查点
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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 从transformers.models.bart.modeling_bart.BartDecoder复制代码，并将Bart->PLBart
class PLBartDecoder(PLBartPreTrainedModel):
    """
    由*config.decoder_layers*层组成的Transformer解码器。每一层都是一个[`PLBartDecoderLayer`]

    Args:
        config: PLBartConfig
        embed_tokens (nn.Embedding): 输出嵌入
    """

    def __init__(self, config: PLBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = PLBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([PLBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
@add_start_docstrings(
    "The bare PLBART Model outputting raw hidden-states without any specific head on top.",
    PLBART_START_DOCSTRING,
)
class PLBartModel(PLBartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    # 初始化方法，接受一个 PLBartConfig 对象作为参数
    def __init__(self, config: PLBartConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 从配置对象中获取填充符索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建一个共享的嵌入层，用于编码器和解码器
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器对象，传入配置对象和共享的嵌入层
        self.encoder = PLBartEncoder(config, self.shared)
        # 创建解码器对象，传入配置对象和共享的嵌入层
        self.decoder = PLBartDecoder(config, self.shared)

        # 初始化模型权重
        self.init_weights()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.shared = value
        # 更新编码器的嵌入层
        self.encoder.embed_tokens = self.shared
        # 更新解码器的嵌入层
        self.decoder.embed_tokens = self.shared

    # 绑定权重，如果配置中设置了词嵌入层绑定
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            # 绑定编码器和解码器的嵌入层权重
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器对象
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象
    def get_decoder(self):
        return self.decoder

    # 前向传播方法
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
# 引入 add_start_docstrings 装饰器，用于添加模型的文档字符串
@add_start_docstrings(
    "The PLBART Model with a language modeling head. Can be used for code-to-text, text-to-code and code-to-code.",
    PLBART_START_DOCSTRING,
)
# 定义 PLBartForConditionalGeneration 类，继承自 PLBartPreTrainedModel
class PLBartForConditionalGeneration(PLBartPreTrainedModel):
    # 定义基本模型前缀
    base_model_prefix = "model"
    # 在加载时忽略的键列表，缺失时的处理
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 绑定权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法
    def __init__(self, config: PLBartConfig):
        # 调用父类初始化方法
        super().__init__(config)
        # 创建 PLBartModel 对象
        self.model = PLBartModel(config)
        # 注册缓冲区，用于存储最终对数偏差
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 线性层，用于生成模型的输出
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化模型权重
        self.init_weights()

    # 获取编码器方法
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器方法
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整令牌嵌入大小的方法
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的 resize_token_embeddings 方法，得到新的嵌入层
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整最终对数偏差的大小以匹配新的嵌入层
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整最终对数偏差大小的内部方法
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取旧的令牌数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的令牌数量小于等于旧的令牌数量
        if new_num_tokens <= old_num_tokens:
            # 截取对应的最终对数偏差
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 创建额外的对数偏差，用零填充
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            # 拼接新的最终对数偏差
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册调整后的最终对数偏差
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出嵌入层方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 添加模型前向方法的文档字符串
    @add_start_docstrings_to_model_forward(PLBART_INPUTS_DOCSTRING)
    # 替换返回值文档字符串，指定输出类型和配置类
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向方法的结尾文档字符串
    @add_end_docstrings(PLBART_GENERATION_EXAMPLE)
    # 此方法用于模型的前向传播
    def forward(
        self,
        # 输入序列的token IDs，默认为None
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力遮罩，用于指示哪些token需要被注意，默认为None
        attention_mask: Optional[torch.LongTensor] = None,
        # 解码器的输入序列的token IDs，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力遮罩，用于指示哪些token需要被注意，默认为None
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # 多头注意力的遮罩，默认为None
        head_mask: Optional[torch.Tensor] = None,
        # 解码器的多头注意力的遮罩，默认为None
        decoder_head_mask: Optional[torch.LongTensor] = None,
        # 交叉注意力的多头注意力遮罩，默认为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器的输出，默认为None
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 用于存储历史键值对的列表，默认为None
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入的嵌入表示，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器输入的嵌入表示，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，默认为None
        labels: Optional[torch.Tensor] = None,
        # 是否使用缓存，默认为None
        use_cache: Optional[bool] = None,
        # 是否输出注意力，默认为None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果，默认为None
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        # 确定是否返回字典类型的结果，如果未指定则根据配置决定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果有提供标签
        if labels is not None:
            # 如果没有提供解码器的输入 id 并且没有提供解码器的嵌入向量，则通过将标签右移一位来生成解码器的输入 id
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        # 将输入传递给模型进行前向传播
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
        # 使用语言模型头部获取预测的 logits
        lm_logits = self.lm_head(outputs[0])
        # 将预测的 logits 与最终的 logits 偏置相加
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        # 如果提供了标签，则计算掩蔽语言模型损失
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算掩蔽语言模型损失
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典类型的结果
        if not return_dict:
            # 整合输出
            output = (lm_logits,) + outputs[1:]
            # 返回输出，包括掩蔽语言模型损失（如果存在）
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 Seq2SeqLMOutput 类型的结果，包括掩蔽语言模型损失（如果存在）和其他相关输出
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
    # 为生成过程准备输入数据
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
        **kwargs,  # TODO: 检查是否需要此项。目前未使用？
    ) -> Dict[str, Any]:
        # 如果使用了过去的键值（past_key_values不为None），则截断decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            
            # 一些生成方法已经只传递最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认使用旧的行为：仅保留最后一个ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs已定义，input_ids不需要
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（假定是为了调试）
        }

    # 从标签准备解码器输入ID
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)

    # 重新排序缓存
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 引入 add_start_docstrings 装饰器，添加了关于 PLBart 序列分类模型的文档字符串
# 在顶部加了一个序列分类器，其结构为在 pooled 输出之上的线性层，例如用于代码分类
@add_start_docstrings(
    """
    PLBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for code
    classification.
    """,
    PLBART_START_DOCSTRING,
)
# 定义了 PLBartForSequenceClassification 类，继承自 PLBartPreTrainedModel
class PLBartForSequenceClassification(PLBartPreTrainedModel):
    # 定义了共享权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化函数，接收一个 PLBartConfig 实例作为参数
    def __init__(self, config: PLBartConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, **kwargs)
        # 创建一个 PLBartModel 实例
        self.model = PLBartModel(config)
        # 创建一个 PLBartClassificationHead 实例，用于序列分类
        self.classification_head = PLBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 装饰器添加了模型 forward 方法的文档字符串
    # 使用 add_code_sample_docstrings 装饰器添加了代码示例的文档字符串
    # 从 transformers.models.bart.modeling_bart.BartForSequenceClassification.forward 复制的方法
    @add_start_docstrings_to_model_forward(PLBART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义了模型的 forward 方法
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
# 从 transformers.models.bart.modeling_bart.BartDecoderWrapper 复制的类，将 Bart 替换为 PLBart
class PLBartDecoderWrapper(PLBartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    # 初始化函数，接收一个 PLBartConfig 实例作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建一个 PLBartDecoder 实例
        self.decoder = PLBartDecoder(config)

    # 定义了 forward 方法，传递参数到 PLBartDecoder 的 forward 方法
    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


# 从 transformers.models.bart.modeling_bart.BartForCausalLM 复制的类，将 Bart 替换为 PLBart，facebook/bart-base 替换为 uclanlp/plbart-base
class PLBartForCausalLM(PLBartPreTrainedModel):
    # 定义了共享权重的键列表
    _tied_weights_keys = ["lm_head.weight"]
    # 初始化方法，传入配置并通过深拷贝创建新配置对象
    def __init__(self, config):
        # 使用深拷贝创建配置对象
        config = copy.deepcopy(config)
        # 设置为解码器
        config.is_decoder = True
        # 设置为非编码器-解码器模型
        config.is_encoder_decoder = False
        # 调用父类初始化方法
        super().__init__(config)
        # 创建 PLBartDecoderWrapper 模型
        self.model = PLBartDecoderWrapper(config)

        # 创建线性层，用于生成语言模型输出
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入词嵌入
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出词嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出词嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 前向传播方法，接受一系列输入参数并返回输出
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
    
    # 为生成准备输入数据的方法
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 如果注意力掩码为空，则创建全为1的注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # 如果有过去键值存在，则根据过去键值调整输入 ID
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # 有些生成方法可能只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最终一个输入 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 调整输入 ID
            input_ids = input_ids[:, remove_prefix_length:]
        # 第一步，解码器缓存状态为空
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
        # 重新排列缓存中的过去键和值
        def _reorder_cache(past_key_values, beam_idx):
            # 初始化重新排列后的过去信息
            reordered_past = ()
            # 遍历每一层的过去信息
            for layer_past in past_key_values:
                # 通过beam_idx重新排列每一层的过去信息，并组成新的过去信息元组
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            # 返回重新排列后的过去信息
            return reordered_past
```