# `.\transformers\models\marian\modeling_marian.py`

```py
# 指定文件编码为 UTF-8
# 版权声明，版权归 The Marian Team 作者和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除了适用法律要求或书面同意外，根据许可证分发的软件基于“原样”分发，
# 没有任何形式的保证或条件，明示或暗示
# 有关授权特定语言，参见许可证，限制模型应用的语言和约束

# 从 PyTorch Marian C++ 仓库传输到 PyTorch MarianMTModel 模型

import copy
import math
from typing import Dict, List, Optional, Tuple, Union

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
from .configuration_marian import MarianConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点
_CONFIG_FOR_DOC = "MarianConfig"
_CHECKPOINT_FOR_DOC = "Helsinki-NLP/opus-mt-en-de"

# 预训练模型列表
MARIAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Helsinki-NLP/opus-mt-en-de",
    # 可查看所有 Marian 模型：https://huggingface.co/models?filter=marian
]

# 从 transformers.models.bart.modeling_bart.shift_tokens_right 复制的函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的 token 右移一个位置。
    """
    # 创建与输入形状相同的一个全零张量
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将原始输入向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将 decoder 的起始 token 放在首位
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果 pad_token_id 为 None，抛出异常
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将 labels 中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class MarianSinusoidalPositionalEmbedding(nn.Embedding):
    """该模块生成任何长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        # 初始化权重参数
        self.weight = self._init_weight(self.weight)

    @staticmethod
    # 初始化位置编码权重（Position Encoding Weight）
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        该函数与 XLM 中的 create_sinusoidal_embeddings 功能相同，但特征不会交错排列。
        余弦特征位于向量的后半部分 [dim // 2:]。
        """
        # 获取输出张量的形状
        n_pos, dim = out.shape
        # 创建位置编码矩阵
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 设置输出张量不需要梯度计算
        out.requires_grad = False
        # 计算正弦和余弦特征的分界点
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 将正弦特征赋值到输出张量的前半部分
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        # 将余弦特征赋值到输出张量的后半部分
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        # 分离输出张量，使之不需要计算梯度
        out.detach_()
        return out
    
    # 前向传播函数
    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 获取批大小和序列长度
        bsz, seq_len = input_ids_shape[:2]
        # 根据 past_key_values_length 计算当前位置
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的前向传播函数
        return super().forward(positions)
# 从transformers.models.bart.modeling_bart.BartAttention复制而来，将Bart替换为Marian
class MarianAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[MarianConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除（得到`embed_dim`：{self.embed_dim}"
                f"和`num_heads`：{num_heads}）."
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
# 从transformers.models.bart.modeling_bart.BartEncoderLayer复制而来，将Bart替换为Marian，BART替换为MARIAN
class MarianEncoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = MARIAN_ATTENTION_CLASSES[config._attn_implementation](
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
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    # 定义一个方法，输入参数为hidden_states, attention_mask, layer_head_mask, output_attentions，返回一个元组
    def forward(
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            # hidden_states: 输入层的张量，形状为(batch, seq_len, embed_dim)
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            # attention_mask: 注意力掩码的张量，大小为(batch, 1, tgt_len, src_len)，其中填充元素由非常大的负值表示 
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            # layer_head_mask: 给定层中注意力头的掩码张量，大小为(encoder_attention_heads,)
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            # output_attentions: 是否返回所有注意力层的注意力张量。有关更多详情，请参见返回张量中的“attentions”
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存hidden_states，用于最后与变换后的hidden_states相加
        residual = hidden_states
        # 调用self_attn方法，传入参数hidden_states, attention_mask, layer_head_mask, output_attentions
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对hidden_states进行dropout变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将变换后的hidden_states与保存的residual相加
        hidden_states = residual + hidden_states
        # 对得到的hidden_states进行层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存hidden_states，用于最后与变换后的hidden_states相加
        residual = hidden_states
        # 对hidden_states进行激活函数变换
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对得到的hidden_states进行dropout变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 再次对hidden_states进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 对得到的hidden_states进行dropout变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将变换后的hidden_states与保存的residual相加
        hidden_states = residual + hidden_states
        # 对得到的hidden_states进行层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果hidden_states的数据类型为torch.float16且包含无穷大或NaN值
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 对hidden_states进行截断，防止出现无穷大或NaN值
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 将变换后的hidden_states保存在元组outputs中
        outputs = (hidden_states,)

        # 如果output_attentions为真，则将attn_weights也保存在元组outputs中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回outputs
        return outputs
MARIAN_ATTENTION_CLASSES = {"eager": MarianAttention}

# 从transformers.models.bart.modeling_bart.BartDecoderLayer中复制，将Bart->Marian, BART->MARIAN
class MarianDecoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 创建自注意力机制对象
        self.self_attn = MARIAN_ATTENTION_CLASSES[config._attn_implementation](
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

        # 创建自注意力机制的LayerNorm对象
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建编码器-解码器注意力机制对象
        self.encoder_attn = MARIAN_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        
        # 创建编码器-解码器注意力机制的LayerNorm对象
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建全连接层对象
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        
        # 创建最终的LayerNorm对象
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
class MarianPreTrainedModel(PreTrainedModel):
    config_class = MarianConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    # 初始化模型权重
    def _init_weights(self, module: Union[nn.Linear, nn.Embedding, MarianSinusoidalPositionalEmbedding]):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            # 初始化线性层权重和偏置
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, MarianSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层权重
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    # 定义一个方法，用于生成模型的虚拟输入数据
    def dummy_inputs(self):
        # 获取模型配置中的填充标记 ID
        pad_token = self.config.pad_token_id
        # 创建包含两个示例输入序列的张量，每个序列由整数表示
        # 第一个序列：[0, 6, 10, 4, 2]
        # 第二个序列：[0, 8, 12, 2, pad_token]
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典，包括注意力遮罩、输入序列和解码器输入序列
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 生成注意力遮罩，标记填充部分
            "input_ids": input_ids,  # 输入序列
            "decoder_input_ids": input_ids,  # 解码器输入序列与输入序列相同
        }
        # 返回虚拟输入字典
        return dummy_inputs
# 定义 MARIAN_START_DOCSTRING 变量，包含模型的基本信息和参数说明
MARIAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MarianConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 MARIAN_GENERATION_EXAMPLE 变量，包含模型的示例和用法说明
MARIAN_GENERATION_EXAMPLE = r"""
    Pytorch version of marian-nmt's transformer.h (c++). Designed for the OPUS-NMT translation checkpoints. Available
    models are listed [here](https://huggingface.co/models?search=Helsinki-NLP).

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, MarianMTModel

    >>> src = "fr"  # source language
    >>> trg = "en"  # target language

    >>> model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    >>> model = MarianMTModel.from_pretrained(model_name)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)

    >>> sample_text = "où est l'arrêt de bus ?"
    >>> batch = tokenizer([sample_text], return_tensors="pt")

    >>> generated_ids = model.generate(**batch)
    >>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    "Where's the bus stop?"
    ```py
"""

# 定义 MARIAN_INPUTS_DOCSTRING 变量，暂时没有内容，可以用于添加模型输入说明
MARIAN_INPUTS_DOCSTRING = r"""
"""

# 定义 MarianEncoder 类，继承自 MarianPreTrainedModel
class MarianEncoder(MarianPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MarianEncoderLayer`].

    Args:
        config: MarianConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化Encoder对象，传入配置信息和嵌入的token（如果有）
    def __init__(self, config: MarianConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的构造函数
        super().__init__(config)

        # 设置dropout和layerdrop参数
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 获取嵌入维度，填充索引，最大源序列长度，以及嵌入缩放因子
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 如果传入了嵌入token，则使用传入的；否则，创建一个新的嵌入token
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 创建位置编码对象
        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings, embed_dim, self.padding_idx
        )
        # 创建多个Encoder层
        self.layers = nn.ModuleList([MarianEncoderLayer(config) for _ in range(config.encoder_layers)])

        # 设置梯度检查点为False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数，接收输入id，注意力掩码，头部掩码，嵌入输入，注意力权重输出，隐藏状态输出，返回字典标志等参数
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class MarianDecoder(MarianPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MarianDecoderLayer`]

    Args:
        config: MarianConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 继承父类构造函数，并初始化一些属性
        super().__init__(config)
        # 初始化一些参数
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 如果提供了输出嵌入（output embedding），则使用提供的；否则初始化一个嵌入
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.decoder_vocab_size, config.d_model, self.padding_idx)

        # 初始化嵌入位置
        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.d_model, self.padding_idx
        )
        # 创建解码器层
        self.layers = nn.ModuleList([MarianDecoderLayer(config) for _ in range(config.decoder_layers)])

        # 设置是否启用渐变检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入
        self.embed_tokens = value

    def forward(
        # 定义前向传播函数的输入参数
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    "The bare Marian Model outputting raw hidden-states without any specific head on top.", MARIAN_START_DOCSTRING
)
class MarianModel(MarianPreTrainedModel):
    # 定义一个类属性
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self, config: MarianConfig):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)

        # 获取配置中的填充索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size

        # 创建词嵌入层，用于表示词的向量
        # 使用共享的词嵌入层以确保与所有Marian模型的兼容性
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        if self.config.share_encoder_decoder_embeddings:
            # 如果共享编码器和解码器的词嵌入层，则设置相同的词嵌入层
            encoder_embed_tokens = decoder_embed_tokens = self.shared
        else:
            # 如果词嵌入层不共享，则为编码器和解码器分别创建词嵌入层
            # 以确保它们不相互关联
            encoder_embed_tokens = copy.deepcopy(self.shared)
            decoder_embed_tokens = copy.deepcopy(self.shared)
            self.shared = None

        # 创建Marian模型的编码器和解码器
        self.encoder = MarianEncoder(config, encoder_embed_tokens)
        self.decoder = MarianDecoder(config, decoder_embed_tokens)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 如果词嵌入层是共享的，则返回共享的词嵌入层，否则返回特定于编码器的词嵌入层
        return self.get_encoder().get_input_embeddings()

    def set_input_embeddings(self, value):
        if self.config.share_encoder_decoder_embeddings:
            # 如果词嵌入层是共享的，则设置共享的词嵌入层并将其分配给编码器和解码器
            self.shared = value
            self.encoder.embed_tokens = self.shared
            self.decoder.embed_tokens = self.shared
        else:  # 如果词嵌入层不是共享的，只设置编码器的词嵌入层
            self.encoder.embed_tokens = value

    def get_decoder_input_embeddings(self):
        if self.config.share_encoder_decoder_embeddings:
            # 如果词嵌入层是共享的，则禁止调用此方法，提示用户使用get_input_embeddings代替
            raise ValueError(
                "`get_decoder_input_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
                "is `True`. Please use `get_input_embeddings` instead."
            )
        return self.get_decoder().get_input_embeddings()

    def set_decoder_input_embeddings(self, value):
        if self.config.share_encoder_decoder_embeddings:
            # 如果词嵌入层是共享的，则禁止调用此方法，提示用户直接设置编码器的词嵌入层
            raise ValueError(
                "`config.share_encoder_decoder_embeddings` is set to `True` meaning the decoder input embeddings "
                "are shared with the encoder. In order to set the decoder input embeddings, you should simply set "
                "the encoder input embeddings by calling `set_input_embeddings` with the appropriate embeddings."
            )
        # 如果词嵌入层不是共享的，则设置解码器的词嵌入层
        self.decoder.embed_tokens = value

    def get_encoder(self):
        # 返回Marian模型的编码器
        return self.encoder

    def get_decoder(self):
        # 返回Marian模型的解码器
        return self.decoder
    # 调整解码器的词嵌入，并返回调整后的嵌入层
    def resize_decoder_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        # 如果配置中指定共享编码器和解码器的嵌入层，则抛出错误
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError(
                "`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
                "is `True`. Please use `resize_token_embeddings` instead."
            )

        # 获取原始的解码器输入嵌入层
        old_embeddings = self.get_decoder_input_embeddings()
        # 调整原始嵌入层大小为指定大小
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        # 设置调整后的嵌入层为解码器的输入嵌入层
        self.set_decoder_input_embeddings(new_embeddings)

        # 获取解码器的输入嵌入层
        model_embeds = self.get_decoder_input_embeddings()

        if new_num_tokens is None:
            # 如果没有指定新的标记数量，则返回模型嵌入层
            return model_embeds

        # 更新基础模型和当前模型配置的解码器词汇量大小
        self.config.decoder_vocab_size = new_num_tokens

        # 如果需要，重新绑定权重
        self.tie_weights()

        return model_embeds

    # 调用模型的前向传播函数
    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用给定的注释初始化MarianMTModel类，可以用于文本摘要任务
@add_start_docstrings(
    "The Marian Model with a language modeling head. Can be used for summarization.", MARIAN_START_DOCSTRING
)
class MarianMTModel(MarianPreTrainedModel):
    # 指定基础模型前缀
    base_model_prefix = "model"
    # 加载时忽略的键列表
    _keys_to_ignore_on_load_missing = [
        "final_logits_bias",
        "encoder.embed_positions.weight",
        "decoder.embed_positions.weight",
    ]
    # 保存时忽略的键列表
    _keys_to_ignore_on_save = ["model.encoder.embed_positions.weight", "model.decoder.embed_positions.weight"]
    # 用于共享权重的键列表
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: MarianConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建MarianModel模型
        self.model = MarianModel(config)

        # 计算目标词汇表大小
        target_vocab_size = config.vocab_size if config.share_encoder_decoder_embeddings else config.decoder_vocab_size
        # 注册用于偏置的缓冲区
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        # 创建线性层用于语言模型头部
        self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整token嵌入的大小
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 如果共享编码器和解码器的嵌入，则调整最终处理的偏置
        if self.config.share_encoder_decoder_embeddings:
            self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # 调整token嵌入的大小
    def _resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of=None) -> nn.Embedding:
        # 获取旧的嵌入层
        old_embeddings = self.get_input_embeddings()
        # 调整嵌入层大小
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        # 设置新的嵌入层
        self.set_input_embeddings(new_embeddings)

        # 获取新的嵌入层的token数量
        new_num_tokens = new_embeddings.weight.shape[0]
        # 如果共享编码器和解码器的嵌入，则更新decoder_vocab_size
        if self.config.share_encoder_decoder_embeddings:
            self.config.decoder_vocab_size = new_num_tokens

        # 如果单词嵌入不是共享的，则确保lm head也被调整大小
        if (
            self.config.share_encoder_decoder_embeddings
            and self.get_output_embeddings() is not None
            and not self.config.tie_word_embeddings
        ):
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()
    # 调整解码器的 token embeddings 的大小
    def resize_decoder_token_embeddings(self, new_num_tokens):
        # 如果配置中指定了共享编码器-解码器 embeddings，则抛出错误
        if self.config.share_encoder_decoder_embeddings:
            raise ValueError(
                "`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
                "is `True`. Please use `resize_token_embeddings` instead."
            )

        # 获取旧的解码器输入 embeddings
        old_embeddings = self.model.get_decoder_input_embeddings()
        # 获取调整大小后的新 embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        # 设置解码器输入 embeddings
        self.model.set_decoder_input_embeddings(new_embeddings)

        # 如果词嵌入没有被绑定，确保语言模型头也被调整大小
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            # 获取旧的语言模型头
            old_lm_head = self.get_output_embeddings()
            # 获取调整大小后的新语言模型头
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            # 设置新的语言模型头
            self.set_output_embeddings(new_lm_head)

        # 获取模型 embeddings
        model_embeds = self.model.get_decoder_input_embeddings()

        # 如果没有提供新的 token 数量，则返回模型 embeddings
        if new_num_tokens is None:
            return model_embeds

        # 更新基础模型和当前模型的配置
        self.config.decoder_vocab_size = new_num_tokens

        # 如果需要的话重新绑定权重
        self.tie_weights()

        # 调整最终 logits 的偏置
        self._resize_final_logits_bias(new_num_tokens)

        # 返回模型 embeddings
        return model_embeds

    # 调整最终 logits 的偏置
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取旧的 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于旧的 token 数量，则截取偏置
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 否则，在最后增加额外的偏置
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册调整后的偏置
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出 embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出 embeddings
    def set_output_embeddings(self, new_embeddings: nn.Embedding):
        self.lm_head = new_embeddings
    # 将输入嵌入和输出嵌入之间的权重进行绑定或克隆

    # 获取输出嵌入层
    output_embeddings = self.get_output_embeddings()
    # 检查输出嵌入层是否存在，并且是否需要共享权重
    if output_embeddings is not None and getattr(self.config, "tie_word_embeddings", True):
        # 获取输入嵌入层，共享嵌入层或者获取解码器嵌入
        word_embeddings = self.get_decoder().get_input_embeddings()
        # 绑定或克隆权重
        self._tie_or_clone_weights(output_embeddings, word_embeddings)

    # 如果模型是编码器-解码器结构，并且设置了绑定编码器解码器权重的标志
    if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
        # 如果模型有基础模型前缀，则设置当前模型为基础模型
        if hasattr(self, self.base_model_prefix):
            self = getattr(self, self.base_model_prefix)
        # 绑定编码器解码器权重
        self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

    # 遍历模型的不同层级模块
    for module in self.modules():
        # 如果模块具有 "_tie_weights" 方法，则调用该方法
        if hasattr(module, "_tie_weights"):
            module._tie_weights()

# 定义 forward 方法
    # 根据输入和decoder的输入等参数，执行模型的前向传播
    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MARIAN_GENERATION_EXAMPLE)
    def forward(
        self,
        # 输入序列的 token ID
        input_ids: torch.LongTensor = None,
        # 注意力掩码
        attention_mask: Optional[torch.Tensor] = None,
        # decoder的输入 token ID
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # decoder的注意力掩码
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # 头部掩码
        head_mask: Optional[torch.Tensor] = None,
        # decoder的头部掩码
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        # 过去的键值对
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 输入嵌入向量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # decoder输入嵌入向量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签序列
        labels: Optional[torch.LongTensor] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 输出注意力分布
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 返回字典类型结果
        return_dict: Optional[bool] = None,
    ) -> Seq2SeqLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        # 确定是否返回输出字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 处理标签信息
        if labels is not None:
            # 如果使用缓存，则提示警告，并更改为不使用缓存
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # 若解码器输入 ID 和解码器嵌入都未提供，则使用标签右移生成解码器输入 ID
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # 调用模型进行前向传播
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
        # 计算语言模型输出并加上最终 logits 偏置
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # 配置交叉熵损失函数并计算损失
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))

        # 如果不返回输出字典，则组合输出并返回
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 Seq2SeqLMOutput 对象
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
        decoder_input_ids: torch.LongTensor,  # 准备用于生成的输入数据，包括解码器输入的token ID
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，用于生成过程中记忆上下文
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指示哪些token需要被注意
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，用于屏蔽某些头部的注意力
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器头部掩码，跟head_mask类似
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码，用于交叉注意力机制
        use_cache: Optional[bool] = None,  # 是否使用缓存，用于调试
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,  # 编码器的输出，用于生成过程中
        **kwargs,  # 其他参数
    ) -> Dict:  # 返回一个字典
        # cut decoder_input_ids if past is used 如果使用了过去的键值对，就需要切割解码器的输入token ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:  # 如果解码器输入的token ID长度大于过去的长度
                remove_prefix_length = past_length  # 需要移除前缀长度
            else:
                # Default to old behavior: keep only final ID 保持原始行为：只保留最后一个token
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]  # 切割解码器的输入token ID

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed 输入ID为空，因为encoder_outputs已经定义了
            "encoder_outputs": encoder_outputs,  # 编码器的输出
            "past_key_values": past_key_values,  # 过去的键值对
            "decoder_input_ids": decoder_input_ids,  # 解码器的输入token ID
            "attention_mask": attention_mask,  # 注意力掩码
            "head_mask": head_mask,  # 头部掩码
            "decoder_head_mask": decoder_head_mask,  # 解码器头部掩码
            "cross_attn_head_mask": cross_attn_head_mask,  # 交叉注意力头部掩码
            "use_cache": use_cache,  # 是否使用缓存（用于调试）
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)  # 根据标签准备解码器输入token ID

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):  # 重新排序缓存中的past_key_values
        reordered_past = ()  # 初始化重新排序后的past_key_values
        for layer_past in past_key_values:  # 遍历每一个过去的键值对
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],  # 重新排序过去的状态信息
            )
        return reordered_past  # 返回重新排序后的past_key_values
# 根据transformers.models.bart.modeling_bart.BartDecoderWrapper复制并修改为MarianDecoderWrapper，用作Marian模型的解码器包装器

class MarianDecoderWrapper(MarianPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在因果语言模型与[`EncoderDecoderModel`]框架结合使用时正确加载预训练检查点。
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化Marian解码器
        self.decoder = MarianDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


# 从transformers.models.bart.modeling_bart.BartForCausalLM复制并修改为MarianForCausalLM，将Bart改为Marian，facebook/bart-base改为Helsinki-NLP/opus-mt-fr-en
class MarianForCausalLM(MarianPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深拷贝配置
        config = copy.deepcopy(config)
        # 设定为解码器
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 初始化Marian解码器包装器
        self.model = MarianDecoderWrapper(config)

        # 初始化线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
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
    # 为生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 如果模型作为编码器在编码器-解码器模型中被使用，解码器注意力掩码将动态创建
        if attention_mask is None:
            # 如果注意力掩码为空，使用与输入相同形状的全为1的张量作为注意力掩码
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：只保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 仅保留ID之后的部分作为新的输入ID
            input_ids = input_ids[:, remove_prefix_length:]
        # 第一步，解码器缓存状态为空
        return {
            "input_ids": input_ids,  # encoder_outputs被定义。不需要input_ids
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 对过去键值对进行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```