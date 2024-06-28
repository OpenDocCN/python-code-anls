# `.\models\trocr\modeling_trocr.py`

```py
# 定义了一个 Python 脚本的版权声明和编码声明
# 该模块实现了一个基于 RoBERTa 的 PyTorch TrOCR 解码器模型

import copy  # 导入 copy 模块，用于复制对象
import math  # 导入 math 模块，提供数学函数
from typing import Optional, Tuple, Union  # 导入类型提示工具

import torch  # 导入 PyTorch 库
from torch import nn  # 导入 PyTorch 中的神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 从内部模块导入激活函数映射
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入处理注意力掩码的函数
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions  # 导入模型输出相关类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import add_start_docstrings, logging, replace_return_docstrings  # 导入工具函数和日志记录器
from .configuration_trocr import TrOCRConfig  # 导入 TrOCR 模型的配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

_CONFIG_FOR_DOC = "TrOCRConfig"  # 文档中使用的 TrOCR 配置类名
_CHECKPOINT_FOR_DOC = "microsoft/trocr-base-handwritten"  # 文档中使用的 TrOCR 预训练模型地址

# 预训练模型的列表，包含了所有可用的 TrOCR 预训练模型地址
TROCR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/trocr-base-handwritten",
    # 更多预训练模型地址可以在 https://huggingface.co/models?filter=trocr 查看
]

# 从 transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding 复制代码，并修改为 TrOCR
class TrOCRLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # TrOCR 特有的设置，如果指定了 padding_idx 则需要将 embedding ids 偏移 2，并相应地调整 num_embeddings
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        # 根据序列长度和过去键值对长度创建位置张量
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


class TrOCRSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        # 设置起始位置偏移量为2
        self.offset = 2
        # 设置嵌入维度
        self.embedding_dim = embedding_dim
        # 设置填充索引（如果有的话）
        self.padding_idx = padding_idx
        # 调用get_embedding方法获取嵌入权重
        self.weights = self.get_embedding(num_positions, embedding_dim, padding_idx)
        # 注册一个用于存储浮点张量的缓冲区
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        构建正弦嵌入。这与tensor2tensor中的实现相匹配，但与《Attention Is All You Need》第3.5节中的描述略有不同。
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算正弦波的周期
        emb = math.log(10000) / (half_dim - 1)
        # 计算正弦和余弦的权重
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果embedding_dim为奇数，填充一个零向量
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果有填充索引，将其对应行置零
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # 根据输入的token id创建位置id。任何填充的token保持填充状态。
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # 如果权重为None或者最大位置超过了当前权重的大小，则重新计算/扩展权重
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # 如有需要，重新计算/扩展嵌入权重
            self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        # 根据位置id选择相应的权重，形成输出张量x
        x = self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

        return x

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        将非填充符号替换为它们的位置号码。位置号码从padding_idx+1开始。忽略填充符号。这是从fairseq的`utils.make_positions`修改而来。
        """
        # 这里的类型转换和转换序列被精心平衡，既能与ONNX导出一起工作，也能与XLA一起工作。
        # 创建一个mask，用于标记非填充符号
        mask = input_ids.ne(padding_idx).int()
        # 生成增量索引，考虑过去的键值长度，并根据mask调整
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx
class TrOCRAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

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
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if not (self.head_dim * num_heads == self.embed_dim):
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于调整注意力分数的大小

        self.is_decoder = is_decoder

        # 下面开始定义用于注意力计算的线性变换层
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)  # K 矩阵的线性投影
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)  # V 矩阵的线性投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Q 矩阵的线性投影

        # 最后的输出投影层，用于将注意力输出映射回原始的 embed_dim 维度
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入的 tensor 重塑为适合多头注意力的形状
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
        # 注意力层的前向传播函数，接受多个输入和参数进行注意力计算和输出
    # 初始化方法，接受一个TrOCRConfig类型的配置参数
    def __init__(self, config: TrOCRConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为配置参数中的隐藏大小
        self.embed_dim = config.hidden_size

        # 创建自注意力机制对象
        self.self_attn = TrOCRAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 设置dropout比例
        self.dropout = config.dropout
        # 设置激活函数为配置参数中指定的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的dropout比例
        self.activation_dropout = config.activation_dropout

        # 对自注意力输出进行LayerNorm归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 如果配置中指定为decoder模式，则创建编码器注意力机制对象
        if config.is_decoder:
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
            # 对编码器注意力输出进行LayerNorm归一化
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 全连接层1，将嵌入维度映射到配置参数中的解码器FFN维度
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 全连接层2，将解码器FFN维度映射回嵌入维度
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 最终输出的LayerNorm归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
class TrOCRPreTrainedModel(PreTrainedModel):
    # 指定该类的配置类为TrOCRConfig
    config_class = TrOCRConfig
    # 模型中基础模型的前缀名称为"model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        # 从配置中获取初始化的标准差
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 如果是线性层或者一维卷积层，使用正态分布初始化权重，偏置初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，如果有padding_idx，则将对应索引的权重初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


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


class TrOCRDecoder(TrOCRPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TrOCRDecoderLayer`]

    Args:
        config: TrOCRConfig
    """

    def __init__(self, config: TrOCRConfig):
        super().__init__(config)
        # 从配置中获取参数
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        # 如果配置中指定了缩放嵌入，则计算嵌入缩放因子为隐藏大小的平方根，否则为1.0
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 创建词嵌入层，vocab_size为词汇表大小，hidden_size为隐藏大小，padding_idx为填充索引
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # 根据配置选择学习的位置编码还是正弦位置编码
        if config.use_learned_position_embeddings:
            self.embed_positions = TrOCRLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_positions = TrOCRSinusoidalPositionalEmbedding(
                config.max_position_embeddings + self.padding_idx + 1,
                config.hidden_size,
                self.padding_idx,
            )

        # 根据配置选择是否使用层归一化
        if config.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm(config.hidden_size)
        else:
            self.layernorm_embedding = None

        # 创建多层Transformer解码器层列表
        self.layers = nn.ModuleList([TrOCRDecoderLayer(config) for _ in range(config.decoder_layers)])

        # 默认关闭梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并进行最终处理
        self.post_init()
    # 返回输入的嵌入向量
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入的嵌入向量
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 模型的前向传播函数，接受多个参数用于 Transformer 模型的输入和控制行为
    def forward(
        self,
        input_ids=None,  # 输入的 token IDs
        attention_mask=None,  # 注意力掩码，指示哪些 token 是真实的输入还是填充
        encoder_hidden_states=None,  # 编码器的隐藏状态（用于 encoder-decoder 模型）
        encoder_attention_mask=None,  # 编码器的注意力掩码（用于 encoder-decoder 模型）
        head_mask=None,  # 多头注意力的头部掩码，用于控制哪些头部参与注意力计算
        cross_attn_head_mask=None,  # 跨注意力的头部掩码，用于 encoder-decoder 模型
        past_key_values=None,  # 用于存储过去的键值对，以便于循环生成器等场景
        inputs_embeds=None,  # 直接输入的嵌入向量（替代 input_ids）
        use_cache=None,  # 是否使用缓存
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出所有隐藏状态
        return_dict=None,  # 是否返回字典格式的输出
@add_start_docstrings(
    "The TrOCR Model with a language modeling head. Can be used for summarization.",
    TROCR_START_DOCSTRING,
)
class TrOCRDecoderWrapper(TrOCRPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = TrOCRDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


@add_start_docstrings(
    "The TrOCR Decoder with a language modeling head. Can be used as the decoder part of [`EncoderDecoderModel`] and"
    " [`VisionEncoderDecoder`].",
    TROCR_START_DOCSTRING,
)
class TrOCRForCausalLM(TrOCRPreTrainedModel):
    _tied_weights_keys = ["output_projection.weight"]

    def __init__(self, config):
        # 深度复制配置，标记为解码器，不作为编码器-解码器模型
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 创建 TrOCRDecoderWrapper 实例
        self.model = TrOCRDecoderWrapper(config)

        # 创建线性层用于输出投影，无偏置
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型中的嵌入层
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置模型的输入嵌入层
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回输出投影层
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出投影层
        self.output_projection = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 获取解码器
        return self.model.decoder

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
    ):
        # 此处包含模型前向传播逻辑，详细的参数说明可以参考函数定义的文档字符串

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 为生成准备输入，具体实现逻辑可能包括处理输入参数和缓存的键值对
    ):
        # 如果模型作为编码器-解码器模型的解码器使用，则动态创建解码器注意力遮罩
        if attention_mask is None:
            # 如果注意力遮罩为 None，则创建一个与输入长度相同的全 1 矩阵作为注意力遮罩
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 某些生成方法可能只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                # 如果输入长度大于过去长度，则设置移除前缀的长度为过去长度
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 移除前缀，保留后缀部分作为新的 input_ids
            input_ids = input_ids[:, remove_prefix_length:]
        # 第一步，decoder_cached_states 是空的
        return {
            "input_ids": input_ids,  # encoder_outputs 已经定义，input_ids 不再需要
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 根据 beam_idx 重新排序每层的过去状态
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```