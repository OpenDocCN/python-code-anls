# `.\transformers\models\musicgen\modeling_musicgen.py`

```py
# 设置编码为 UTF-8
# 版权声明及许可证信息
# 版权归 Meta AI 和 The HuggingFace Inc. 团队所有，保留所有权利。
# 根据 Apache 许可证 2.0 版（“许可证”）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“原样”基础提供的，不提供任何形式的担保或条件，
# 无论是明示的还是暗示的。
# 有关详细信息，请参阅许可证。

"""PyTorch Musicgen model."""
# 导入所需模块
import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# 导入生成配置相关的模块
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    ModelOutput,
    Seq2SeqLMOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig

# 如果是类型检查的情况下，导入 BaseStreamer 模块
if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中使用的配置信息
_CONFIG_FOR_DOC = "MusicgenConfig"
_CHECKPOINT_FOR_DOC = "facebook/musicgen-small"

# Musicgen 预训练模型存档列表
MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/musicgen-small",
    # 查看所有 Musicgen 模型：https://huggingface.co/models?filter=musicgen
]

# 定义用于无条件输入的 Musicgen 模型输出类
@dataclass
class MusicgenUnconditionalInput(ModelOutput):
    """
    # 定义函数的参数列表，包括文本编码器的输出、注意力掩码和引导比例
    Args:
        encoder_outputs  (`Tuple[torch.FloatTensor]` of length 1, with tensor shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the text encoder model.
        attention_mask (`torch.LongTensor`)  of shape `(batch_size, sequence_length)`, *optional*):
            Encoder attention mask to avoid performing attention on padding token indices. Mask values selected in `[0,
            1]`: 1 for tokens that are **not masked**, 0 for tokens that are **masked**.
        guidance_scale (`float`, *optional*):
            Guidance scale for classifier free guidance, setting the balance between the conditional logits (predicted
            from the prompts) and the unconditional logits (predicted without prompts).
    """
    
    # 初始化函数内部变量 encoder_outputs，attention_mask 和 guidance_scale，默认值为 None
    encoder_outputs: Tuple[torch.FloatTensor] = None
    attention_mask: torch.LongTensor = None
    guidance_scale: float = None
# 从transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right复制过来的函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的token向右移动一个位置。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入的token向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    # 设置第一个位置token为decoder_start_token_id
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将labels中可能存在的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class MusicgenSinusoidalPositionalEmbedding(nn.Module):
    """该模块生成任意长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            # 在前向传播中，将权重放置在param的正确dtype和device上
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """
        构建正弦嵌入。这与tensor2tensor中的实现相匹配，但与"Attention Is All You Need"第3.5章节的描述略有不同。
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # 零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    # 定义一个前向传播函数，接受输入的 token ids 和已经处理的键值对长度
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        # 获取输入的批次大小、码书数量和序列长度
        bsz, codebooks, seq_len = input_ids.size()
        # 从输入的 token ids 创建位置 ids
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(input_ids.device)
        # 如果序列长度超过了权重矩阵的容量，需要扩展权重
        if seq_len > self.weights.size(0):
            self.make_weights(seq_len + self.offset, self.embedding_dim)
        # 返回根据位置 ids 选择权重矩阵后的结果，并且detach以避免计算梯度
        return self.weights.index_select(0, position_ids.view(-1)).detach()
# 从transformers.models.bart.modeling_bart.BartAttention中拷贝代码，并将类名从BartAttention改为MusicgenAttention
class MusicgenAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[MusicgenConfig] = None,
    ):
        super().__init__()
        # 嵌入维度
        self.embed_dim = embed_dim
        # 注意力头的数量
        self.num_heads = num_heads
        # dropout率
        self.dropout = dropout
        # 头维度等于嵌入维度除以注意力头的数量
        self.head_dim = embed_dim // num_heads
        # 配置
        self.config = config

        # 如果头维度乘以注意力头的数量不等于嵌入维度，抛出异常
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5
        # 是否为解码器层
        self.is_decoder = is_decoder
        # 是否是因果关系
        self.is_causal = is_causal

        # 分别定义k、v、q、out的全连接层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 将张量形状转换为(bsz, num_heads, seq_len, head_dim)的格式，并交换维度1和2
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



class MusicgenDecoderLayer(nn.Module):
    def __init__(self, config: MusicgenDecoderConfig):
        super().__init__()
        # 嵌入维度
        self.embed_dim = config.hidden_size

        # 创建自注意力层
        self.self_attn = MusicgenAttention(
            embed_dim=self.embed_dim, # 嵌入维度
            num_heads=config.num_attention_heads, # 注意力头数量
            dropout=config.attention_dropout, # dropout率
            is_decoder=True, # 是否为解码器层
            bias=False, # 是否使用偏置
        )
        # dropout率
        self.dropout = config.dropout
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 激活函数的dropout率
        self.activation_dropout = config.activation_dropout

        # 创建自注意力层的层归一化层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建编码器注意力层
        self.encoder_attn = MusicgenAttention(
            self.embed_dim,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
        )
        # 创建编码器注意力层的层归一化层
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建前馈神经网络的全连接层1
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=False)
        # 创建前馈神经网络的全连接层2
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=False)
        # 创建最后的层归一化层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    # MBartDecoderLayer 类的前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 自注意力机制的掩码张量，默认为None
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态张量，默认为None
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码张量，默认为None
        layer_head_mask: Optional[torch.Tensor] = None,  # 层头掩码张量，默认为None
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,  # 跨层注意力机制的层头掩码张量，默认为None
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 存储过去的键值对的元组，默认为None
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为False
        use_cache: Optional[bool] = True,  # 是否使用缓存，默认为True
class MusicgenPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权ts�initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 MusicgenDecoderConfig 作为配置类
    config_class = MusicgenDecoderConfig
    # 模型的前缀名称
    base_model_prefix = "model"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表
    _no_split_modules = ["MusicgenDecoderLayer", "MusicgenAttention"]

    def _init_weights(self, module):
        # 初始化权重函数
        std = self.config.initializer_factor
        # 如果是线性层或一维卷积层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正ta�normal_(mean=0.0��重
            module.weight.data.normal_(mean=0.0, std=��化为零      # 如果存在le�bias is��ot�始化为             module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，则将对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Musicgen模型的起始文档字符串
MUSICGEN_START_DOCSTRING = r"""

    Musicgen模型由Jade Copet、Felix Kreuk、Itai Gat、Tal Remez、David Kant、Gabriel Synnaeve、Yossi Adi、Alexandre Défossez
    在[Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)中提出。它是一个在条件音乐生成任务上训练的编码器解码器transformer模型。

    该模型继承自[`PreTrainedModel`]。查看超类文档以获取库实现的所有通用方法（如下载或保存、调整输入嵌入、修剪头ments for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MusicgenConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MUSICGEN_INPUTS_DOCSTRING = r"""
"""

MUSICGEN_DECODER_INPUTS_DOCSTRING = r"""
"""


class MusicgenDecoder(MusicgenPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MusicgenDecoderLayer`]
    """
    # 初始化函数，接收一个 MusicgenDecoderConfig 对象作为参数
    def __init__(self, config: MusicgenDecoderConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置类属性 dropout 为配置中的 dropout 参数
        self.dropout = config.dropout
        # 设置类属性 layerdrop 为配置中的 layerdrop 参数
        self.layerdrop = config.layerdrop
        # 设置类属性 max_target_positions 为配置中的 max_position_embeddings 参数
        self.max_target_positions = config.max_position_embeddings
        # 设置类属性 d_model 为配置中的 hidden_size 参数
        self.d_model = config.hidden_size
        # 设置类属性 num_codebooks 为配置中的 num_codebooks 参数
        self.num_codebooks = config.num_codebooks
        # 如果配置中 scale_embedding 为 True，则设置类属性 embed_scale 为 hidden_size 的平方根，否则设为 1.0
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 计算 embed_dim 为 vocab_size + 1
        embed_dim = config.vocab_size + 1
        # 创建多个 Embedding 层，存储在 ModuleList 中，每个 Embedding 层的大小为 config.hidden_size
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(embed_dim, config.hidden_size) for _ in range(config.num_codebooks)]
        )

        # 初始化位置编码层，传入最大位置长度和隐藏层大小作为参数
        self.embed_positions = MusicgenSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        # 创建多个 Decoder 层，存储在 ModuleList 中，数量为 config.num_hidden_layers
        self.layers = nn.ModuleList([MusicgenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化 LayerNorm 层，大小为 hidden_size
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 设置梯度检查为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入 embedding 层
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入 embedding 层
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数，接收多个参数，详细参数说明可参考 MUSICGEN_DECODER_INPUTS_DOCSTRING
    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(
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
# 使用 `add_start_docstrings` 装饰器为 MusicgenModel 类添加文档字符串，描述其输出原始隐藏状态的解码器模型，没有特定的输出头
# `MUSICGEN_START_DOCSTRING` 包含的文档字符串
class MusicgenModel(MusicgenPreTrainedModel):
    # 初始化方法，接收 MusicgenDecoderConfig 类型的配置参数
    def __init__(self, config: MusicgenDecoderConfig):
        # 调用父类构造函数
        super().__init__(config)
        # 初始化解码器
        self.decoder = MusicgenDecoder(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 重写 forward 方法，接收一系列输入参数并执行前向传播
    # 添加了文档字符串，描述了 forward 方法接受的各种输入
    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(
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
    # 定义一个方法，接受输入并返回解码器的输出，包括注意力、隐藏状态等
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # 如果未指定输出注意力，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定使用缓存，则使用配置中的默认值
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果未指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 解码器输出包括（解码特征，过去的键值对，解码隐藏状态，解码注意力）
        # 调用解码器的forward方法，传递参数
        decoder_outputs = self.decoder(
            input_ids=input_ids,  # 输入的标识符
            attention_mask=attention_mask,  # 注意力掩码
            encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
            encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
            head_mask=head_mask,  # 头部掩码
            cross_attn_head_mask=cross_attn_head_mask,  # 交叉注意力头部掩码
            past_key_values=past_key_values,  # 过去的键值对
            inputs_embeds=inputs_embeds,  # 输入的嵌入向量
            use_cache=use_cache,  # 是否使用缓存
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典
        )

        # 如果不返回字典，则直接返回解码器的输出
        if not return_dict:
            return decoder_outputs

        # 返回带有过去注意力和交叉注意力的基本模型输出
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_outputs.last_hidden_state,  # 最后隐藏状态
            past_key_values=decoder_outputs.past_key_values,  # 过去的键值对
            hidden_states=decoder_outputs.hidden_states,  # 隐藏状态
            attentions=decoder_outputs.attentions,  # 注意力
            cross_attentions=decoder_outputs.cross_attentions,  # 交叉注意力
        )
# 使用装饰器添加文档字符串，说明 Musicgen 解码模型及其语言建模头
class MusicgenForCausalLM(MusicgenPreTrainedModel):
    def __init__(self, config: MusicgenDecoderConfig):
        # 调用父类构造函数
        super().__init__(config)

        # 创建 Musicgen 模型
        self.model = MusicgenModel(config)

        # 设置编码矢量和词汇量大小，并创建语言建模头
        self.num_codebooks = config.num_codebooks
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_heads

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_heads = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 前向传播方法，使用装饰器添加输入输出文档字符串
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
                Returns:
        """

        # 确定是否从模型返回字典结构的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入序列传入模型进行预测
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型的隐藏状态
        hidden_states = outputs[0]

        # 获取每个头部的预测结果，并在维度1上进行堆叠
        lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 若存在标签，则抛出未实现的错误，因为 Musicgen 的训练未实现
            raise NotImplementedError("Training is not implemented for Musicgen.")

        # 将预测结果的形状从 (bsz, num_codebooks, seq_len, vocab_size) 改为 (bsz * num_codebooks, seq_len, vocab_size)
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])

        if not return_dict:
            # 若不返回字典结构的结果，则返回 (lm_logits, ...) 的元组
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 若返回字典结构的结果，则返回 CausalLMOutputWithCrossAttentions 的对象
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=True,
        delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,

注释：
    ):
        # 如果没有传入延迟模式掩码，则根据输入的input_ids构建延迟模式掩码
        if delay_pattern_mask is None:
            input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
                input_ids,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # 应用延迟模式掩码
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)

        # 如果guidance_scale不为None且大于1，则需要在批次维度上复制解码器参数
        if guidance_scale is not None and guidance_scale > 1:
            # 为了分类器自由的guidance，我们需要在批次维度上复制解码器参数（在抽样之前我们会将其拆分）
            input_ids = input_ids.repeat((2, 1))
            if attention_mask is not None:
                attention_mask = attention_mask.repeat((2, 1))

        # 如果有过去的键值对，则将输入限制在最后一个token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回包含生成所需参数的字典
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "head_mask": head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """将延迟模式掩码应用到解码器的input_ids上，只保留掩码值为-1的预测值，
        其他地方的值按照掩码中的值进行设定。"""
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
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
# 给 MusicgenForConditionalGeneration 类添加文档字符串，描述其作用和特性
@add_start_docstrings(
    "The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder, "
    "for music generation tasks with one or both of text and audio prompts.",
    MUSICGEN_START_DOCSTRING,
)
class MusicgenForConditionalGeneration(PreTrainedModel):
    # 指定配置类为 MusicgenConfig
    config_class = MusicgenConfig
    # 定义模型基类前缀为"encoder_decoder"
    base_model_prefix = "encoder_decoder"
    # 主输入名称为"input_ids"
    main_input_name = "input_ids"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[MusicgenConfig] = None,
        text_encoder: Optional[PreTrainedModel] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[MusicgenForCausalLM] = None,
    # 绑定权重的方法
    def tie_weights(self):
        # 如果需要绑定文本编码器和解码器
        if self.config.tie_encoder_decoder:
            # 获取解码器的基本模型前缀
            decoder_base_model_prefix = self.decoder.base_model_prefix
            # 绑定文本编码器和解码器的权重
            self._tie_encoder_decoder_weights(
                self.text_encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    # 获取音频编码器
    def get_audio_encoder(self):
        return self.audio_encoder

    # 获取文本编码器
    def get_text_encoder(self):
        return self.text_encoder

    # 获取编码器
    def get_encoder(self):
        # 获取文本编码器以计算生成的编码器隐藏状态
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

    # 从预训练模型中加载模型
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        ```py"""

        # 目前不支持快速初始化对于复合模型
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for MusicgenForConditionalGeneration. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        # 返回加载的模型
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    # 从子模型中加载预训练模型
    @classmethod
    def from_sub_models_pretrained(
        cls,
        text_encoder_pretrained_model_name_or_path: str = None,
        audio_encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    # 在模型前向方法中添加文档字符串
    @add_start_docstrings_to_model_forward(MUSICGEN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
```  
    # 定义一个 forward 方法，用于模型前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的词语在词汇表中的索引，可选的长整型张量
        attention_mask: Optional[torch.BoolTensor] = None,  # 注意力遮罩，可选的布尔型张量
        input_values: Optional[torch.FloatTensor] = None,  # 输入的值，可选的浮点型张量
        padding_mask: Optional[torch.BoolTensor] = None,  # 填充遮罩，可选的布尔型张量
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的词语索引，可选的长整型张量
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力遮罩，可选的布尔型张量
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,  # 编码器的输出，可选的包含浮点型张量的元组
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,  # 上下文键值对，元组中包含元组的张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入，可选的浮点型张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入，可选的浮点型张量
        labels: Optional[torch.LongTensor] = None,  # 标签，可选的长整型张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典，可选的布尔值
        **kwargs,  # 其它关键字参数
    # 定义一个 prepare_inputs_for_generation 方法，用于为生成准备输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的词语索引
        past_key_values=None,  # 上下文键值对，可选的参数
        attention_mask=None,  # 注意力遮罩，可选的参数
        head_mask=None,  # 头部遮罩，可选的参数
        decoder_attention_mask=None,  # 解码器注意力遮罩，可选的参数
        decoder_head_mask=None,  # 解码器头部遮罩，可选的参数
        cross_attn_head_mask=None,  # 交叉注意力头部遮罩，可选的参数
        use_cache=None,  # 是否使用缓存，可选的参数
        encoder_outputs=None,  # 编码器的输出，可选的参数
        decoder_delay_pattern_mask=None,  # 解码器的延迟模式遮罩，可选的参数
        guidance_scale=None,  # 引导尺度，可选的参数
        **kwargs,  # 其它关键字参数
        # 如果 decoder_delay_pattern_mask 为 None，则根据生成配置构建解码器延迟模式掩码
        if decoder_delay_pattern_mask is None:
            # 调用解码器的 build_delay_pattern_mask 方法构建延迟模式掩码
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                decoder_input_ids,
                self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # 应用延迟模式掩码
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)

        # 如果 guidance_scale 不为 None 且大于 1
        if guidance_scale is not None and guidance_scale > 1:
            # 对于无分类器指导的情况，我们需要在批处理维度上复制解码器参数（在采样之前将这些参数分割）
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))

        # 如果过去的键值不为 None
        if past_key_values is not None:
            # 获取过去键值中的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留只有最后一个 ID 的旧行为
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 移除前缀，保留有效部分的解码器输入 IDs
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的生成用解码器输入 IDs 及相关参数
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    # 为生成准备解码器输入 IDs
    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""

        # 1. 检查用户是否手动定义了`decoder_input_ids`。为了方便起见，如果编码器不使用它作为主要输入，
        # 我们也允许用户在`input_ids`下传递它。
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # 如果用户在model_kwargs中定义了decoder_input_ids，则将其弹出，并赋值给decoder_input_ids
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            # 如果model_input_name不等于"input_ids"，并且用户在model_kwargs中定义了input_ids，则将其弹出，并赋值给decoder_input_ids
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. 编码器-解码器模型期望`decoder_input_ids`以特殊令牌开头。确保这一点。
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        # 创建一个跟decoder_start_token_id相同值的张量，通过其设备将张量发送给特定设备
        decoder_input_ids_start = (
            torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device)
            * decoder_start_token_id
        )

        # 没有用户输入->使用decoder_start_token_id作为decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start

        # 用户输入但不以decoder_start_token_id开头->在开头添加decoder_start_token_id（如果提供了decoder_attention_mask则相应调整）
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

    def _prepare_text_encoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
        guidance_scale: Optional[float] = None,


注释：
    def prepare_encoder_input_ids(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # 1. 获取文本编码器
        encoder = self.get_text_encoder()
        # 与Accelerate大型模型推断兼容性: 我们需要编码器将输出结果与输入结果放在同一设备上
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. 从模型参数准备编码器参数和编码器关键字参数
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        # 获取编码器的定义参数
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        # 编码器是否接受通配符参数
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            # 不接受，则只将编码器参数与编码器定义参数交集的部分用作编码器关键字参数
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. 确保编码器返回一个`ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        last_hidden_state = encoder(**encoder_kwargs).last_hidden_state

        # 对于自由导向分类器，我们需要在编码器隐藏状态中添加一个"null"输入
        if guidance_scale is not None and guidance_scale > 1:
            last_hidden_state = torch.concatenate([last_hidden_state, torch.zeros_like(last_hidden_state)], dim=0)
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = torch.concatenate(
                    [model_kwargs["attention_mask"], torch.zeros_like(model_kwargs["attention_mask"])], dim=0
                )

        model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=last_hidden_state)

        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签向右移动一个位置，成为解码器的输入
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
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


注释：
    ) -> torch.LongTensor:
        """初始化生成的输入id，如果需要的话。"""
        # 如果已提供输入，则直接返回输入
        if inputs is not None:
            return inputs

        # 如果提供了编码器输出，则创建虚拟的输入id以确保不会被用于编码
        encoder_outputs = model_kwargs.get("encoder_outputs")
        if encoder_outputs is not None:
            # 创建具有值为-100的虚拟输入id，作为一个检查，确保它们不会用于编码
            shape = encoder_outputs[0].size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        # 如果未提供输入id，但也未提供bos_token_id，则抛出异常
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # 如果`model_kwargs`中存在张量，则可以从中推断出批处理大小。这在软提示或构建在解码器模型之上的多模态实现中很有用。
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        # 返回批处理大小的张量，其值全为bos_token_id
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
    # 获取无条件生成所需的输入
    def get_unconditional_inputs(self, num_samples=1):
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.
    
        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.
            max_new_tokens (int, *optional*):
                Number of tokens to generate for each sample. More tokens means longer audio samples, at the expense of
                longer inference (since more audio tokens need to be generated per sample).
    
        Example:
        ```py
        >>> from transformers import MusicgenForConditionalGeneration
    
        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    
        >>> # get the unconditional (or 'null') inputs for the model
        >>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```"""
        # 创建一个 num_samples 行 1 列, 大小为模型文本编码器隐藏层维度的零张量, 用作模型最后一个隐藏状态输入
        last_hidden_state = torch.zeros(
            (num_samples, 1, self.config.text_encoder.hidden_size), device=self.device, dtype=self.dtype
        )
    
        # 创建一个 num_samples 行 1 列的零张量作为注意力掩码
        attention_mask = torch.zeros((num_samples, 1), device=self.device, dtype=torch.long)
    
        # 返回无条件输入对象, 包括最后一个隐藏状态和注意力掩码
        return MusicgenUnconditionalInput(
            encoder_outputs=(last_hidden_state,),
            attention_mask=attention_mask,
            guidance_scale=1.0,
        )
```