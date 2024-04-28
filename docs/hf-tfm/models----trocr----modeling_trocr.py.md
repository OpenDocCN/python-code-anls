# `.\transformers\models\trocr\modeling_trocr.py`

```
# 设置编码格式
# 版权声明
# 根据Apache许可证2.0版许可，除非符合许可证，否则您不能使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得依照本许可证分发软件
# 软件以"原样"分布，不附带任何担保或条件，无论是明示或暗示的
# 请参阅许可证以获取特定语言规定权限和限制
""" PyTorch TrOCR解码器模型（基于RoBERTa）"""


import copy
import math
from typing import Optional, Tuple, Union  # 引入相关模块和类型

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging, replace_return_docstrings
from .configuration_trocr import TrOCRConfig  # 从配置文件中引入TrOCRConfig

# 获取logger对象用于记录日志
logger = logging.get_logger(__name__)

# 以下为在文档中使用的变量声明
_CONFIG_FOR_DOC = "TrOCRConfig"
_CHECKPOINT_FOR_DOC = "microsoft/trocr-base-handwritten"

# TrOCR预训练模型的存档列表
TROCR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/trocr-base-handwritten",
    # 查看所有TrOCR模型：https://huggingface.co/models?filter=trocr
]

# 从transformers.models.bart.modeling_bart中直接复制TrOCRLearnedPositionalEmbedding类，将Bart->TrOCR
class TrOCRLearnedPositionalEmbedding(nn.Embedding):
    """
    此模块学习最大尺寸的位置嵌入。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # TrOCR模型设置成如果指定了padding_idx，则通过2偏移嵌入ids，并相应地调整num_embeddings
        # 其他模型没有这个修改
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids'的形状应为[bsz x seqlen]。"""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


class TrOCRSinusoidalPositionalEmbedding(nn.Module):
    """该模块生成任意长度的正弦位置嵌入。"""
    # 初始化方法，接受位置数量、嵌入维度和填充索引作为参数
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 设置偏移量为2
        self.offset = 2
        # 设置嵌入维度
        self.embedding_dim = embedding_dim
        # 设置填充索引
        self.padding_idx = padding_idx
        # 获取嵌入权重
        self.weights = self.get_embedding(num_positions, embedding_dim, padding_idx)
        # 注册缓冲区
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    # 获取嵌入
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算指数
        emb = math.log(10000) / (half_dim - 1)
        # 计算指数的指数
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        # 构建嵌入
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # 如果嵌入维度是奇数，则用零填充
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            # 如果填充索引不为空，则将对应位置置为0
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    # 前向传播方法
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        # 获取输入的批量大小和序列长度
        bsz, seq_len = input_ids.size()
        # 从输入的 token id 创建位置 id，任何填充的 token 仍然保持填充状态
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # 如果需要，扩展嵌入
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # 如有必要，重新计算/扩展嵌入
            self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        # 获取嵌入
        x = self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

        return x

    # 从输��的 token id 创建位置 id
    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        """
        # 仅保留非填充符号
        mask = input_ids.ne(padding_idx).int()
        # 计算位置 id
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx
    # 定义一个名为 TrOCRAttention 的类，继承自 nn.Module 类
    """Multi-headed attention from 'Attention Is All You Need' paper."""
    # 类的描述信息，说明类的作用和来源

    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        kdim: int = None,
        vdim: int = None,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_cross_attention: bool = False,
    ):
        # 类的初始化方法，接收一些参数
        super().__init__()
        # 调用父类的初始化方法
        self.embed_dim = embed_dim
        # 初始化 embed_dim 属性
        self.kdim = kdim if kdim is not None else embed_dim
        # 初始化 kdim 属性，如果 kdim 为 None，则使用 embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # 初始化 vdim 属性，如果 vdim 为 None，则使用 embed_dim
        self.num_heads = num_heads
        # 初始化 num_heads 属性
        self.dropout = dropout
        # 初始化 dropout 属性
        self.head_dim = embed_dim // num_heads
        # 计算每个头的维度
        if not (self.head_dim * num_heads == self.embed_dim):
            # 如果 embed_dim 不能整除 num_heads，抛出数值错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        # 计算缩放因子
        self.is_decoder = is_decoder
        # 初始化是否为解码器属性

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        # 初始化 k 投影层
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        # 初始化 v 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化 q 投影层

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化输出投影层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 定义一个方法用于重新整形张量
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # 返回重新整形后的张量

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
class TrOCRDecoderLayer(nn.Module):
        # 定义一个前向传播方法，接收一系列参数
    # 初始化函数，接受一个 TrOCRConfig 对象作为参数
    def __init__(self, config: TrOCRConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为配置中的隐藏层大小
        self.embed_dim = config.hidden_size

        # 初始化自注意力机制
        self.self_attn = TrOCRAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置激活函数为配置中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的 dropout 概率
        self.activation_dropout = config.activation_dropout

        # 初始化自注意力机制的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 如果是解码器模型
        if config.is_decoder:
            # 初始化编码器注意力机制
            self.encoder_attn = TrOCRAttention(
                config,
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                kdim=config.cross_attention_hidden_size,
                vdim=config.cross_attention_hidden_size,
                dropout=config.attention_dropout,
                is_decoder=True,
                is_cross_attention=True,
            )
            # 初始化编码器注意力机制的 LayerNorm 层
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化第一个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 初始化第二个全连接层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 初始化最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数
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
# TrOCRPreTrainedModel 是一个预训练模型的基类
class TrOCRPreTrainedModel(PreTrainedModel):
    # 配置类为 TrOCRConfig
    config_class = TrOCRConfig
    # 基础模型前缀为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模块权重
    def _init_weights(self, module):
        # 从标准差为 self.config.init_std 的正态分布初始化权重
        std = self.config.init_std
        # 如果是 Linear 或 Conv1d 层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置项，则将其初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 Embedding 层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，则将其对应的权重初始化为 0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# TrOCR 模型的文档字符串
TROCR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TrOCRConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# TrOCR 解码器模型
class TrOCRDecoder(TrOCRPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TrOCRDecoderLayer`]

    Args:
        config: TrOCRConfig
    """

    def __init__(self, config: TrOCRConfig):
        # 继承 TrOCRPreTrainedModel 的初始化
        super().__init__(config)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置 decoder layer 的 dropout 概率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充 token 的 ID
        self.padding_idx = config.pad_token_id
        # 根据 config.scale_embedding 决定是否对 embedding 进行缩放
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 创建 token embedding 层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # 根据 config.use_learned_position_embeddings 决定使用学习的还是正弦位置编码
        if config.use_learned_position_embeddings:
            self.embed_positions = TrOCRLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_positions = TrOCRSinusoidalPositionalEmbedding(
                config.max_position_embeddings + self.padding_idx + 1,
                config.hidden_size,
                self.padding_idx,
            )

        # 根据 config.layernorm_embedding 决定是否使用 LayerNorm 层
        if config.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm(config.hidden_size)
        else:
            self.layernorm_embedding = None

        # 创建 decoder 层
        self.layers = nn.ModuleList([TrOCRDecoderLayer(config) for _ in range(config.decoder_layers)])

        # 是否启用梯度检查点
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()
    # 获取输入的嵌入向量
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入的嵌入向量
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 神经网络的前向传播函数，接收一系列输入参数
    def forward(
        self,
        input_ids=None,  # 输入的 token 序列
        attention_mask=None,  # 注意力遮挡 mask
        encoder_hidden_states=None,  # 编码器的隐藏状态
        encoder_attention_mask=None,  # 编码器的注意力 mask
        head_mask=None,  # 多头注意力 mask
        cross_attn_head_mask=None,  # 跨注意力多头 mask
        past_key_values=None,  # 用于存储中间状态的变量
        inputs_embeds=None,  # 输入的嵌入向量
        use_cache=None,  # 是否使用缓存
        output_attentions=None,  # 输出注意力权重
        output_hidden_states=None,  # 输出隐藏状态
        return_dict=None,  # 是否返回结果字典
# 使用装饰器添加起始文档字符串，说明该类作为带语言建模头的 TrOCR 模型，能够用于摘要生成
@add_start_docstrings(
    "The TrOCR Model with a language modeling head. Can be used for summarization.",
    TROCR_START_DOCSTRING,
)
# 定义 TrOCRDecoderWrapper 类，继承自 TrOCRPreTrainedModel 类
class TrOCRDecoderWrapper(TrOCRPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 TrOCRDecoder 对象
        self.decoder = TrOCRDecoder(config)

    # 前向传播方法，用于调用 decoder 的前向传播方法
    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

# 使用装饰器添加起始文档字符串，说明该类作为带语言建模头的 TrOCR 解码器，能够用作 [`EncoderDecoderModel`] 和 [`VisionEncoderDecoder`] 的解码器部分
@add_start_docstrings(
    "The TrOCR Decoder with a language modeling head. Can be used as the decoder part of [`EncoderDecoderModel`] and"
    " [`VisionEncoderDecoder`].",
    TROCR_START_DOCSTRING,
)
# 定义 TrOCRForCausalLM 类，继承自 TrOCRPreTrainedModel 类
class TrOCRForCausalLM(TrOCRPreTrainedModel):
    # 定义一个类变量 _tied_weights_keys
    _tied_weights_keys = ["output_projection.weight"]

    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 复制配置参数，避免修改原始配置
        config = copy.deepcopy(config)
        # 设置配置参数的解码器属性
        config.is_decoder = True
        config.is_encoder_decoder = False
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 TrOCRDecoderWrapper 对象
        self.model = TrOCRDecoderWrapper(config)

        # 创建一个线性层，用于输出
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        return self.output_projection

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 前向传播方法，接受多个输入参数
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
    # 为生成准备输入的方法，接受多个输入参数
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    对定义TransformerDecoderLayer类的函数之后，开始定义函数decode方法
    def decode(self, input_ids=None, past_key_values=None, attention_mask=None, use_cache=False):
        # 如果作为编码器-解码器模型中的解码器使用，则动态创建解码器注意力遮罩
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去密钥数的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经仅传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留只有最后一个ID的旧行为
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        # 第一步，解码器缓存状态是空的
        return {
            "input_ids": input_ids,  # encoder_outputs已经定义，不需要input_ids
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    定义内部函数_reorder_cache，用来重排缓存past_key_values
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        遍历过去的每一层密钥值
        for layer_past in past_key_values:
            为每层过去密钥值创建一个元组
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        返回重排后的past_key_values
        return reordered_past
```