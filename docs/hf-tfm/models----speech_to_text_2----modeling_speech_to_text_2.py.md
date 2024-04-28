# `.\transformers\models\speech_to_text_2\modeling_speech_to_text_2.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据Apache License, Version 2.0，除非符合许可证的要求或经书面同意，否则不得使用此文件
# 你可以在http://www.apache.org/licenses/LICENSE-2.0获取许可证的副本
# 除非法律要求或经合同书面同意，否则基于"AS IS"的基础分发软件，无论有无任何种类的保证或条件
# 有关具体语言规则的权限以及许可证下的限制，请参阅许可证
""" PyTorch Speech2Text2 模型。"""


导入所需的库或模块
import copy
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

从Hugging Face相关模块导入的函数
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging, replace_return_docstrings
从.configuration_speech_to_text_2模块导入Speech2Text2Config类
from .configuration_speech_to_text_2 import Speech2Text2Config


获取logger以记录日志
logger = logging.get_logger(__name__)

_DEFINE = "Speech2Text2Config"
_CHECKPOINT_FOR_DOC = "facebook/s2t-wav2vec2-large-en-de"


SPEECH_TO_TEXT_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/s2t-wav2vec2-large-en-de",
    # 参考https://huggingface.co/models?filter=speech2text2 查看所有Speech2Text2模型
]


# 从transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding 模块中复制代码，并将Speech2Text改为Speech2Text2
class Speech2Text2SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    # 初始化函数，设置模块参数
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # 创建权重函数
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 获取位置嵌入
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        # 如果已经有weights参数，将权重放到正确定的dtype和设备上
        if hasattr(self, "weights"):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 设置权重参数，并设置为不需要梯度
        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        构建正弦嵌入。这与 tensor2tensor 中的实现匹配，但与 "Attention Is All You Need" 第3.5节中的描述略有不同。
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算嵌入值
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果嵌入维度为奇数，进行零填充
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果指定了填充索引，将该索引对应的嵌入值设为0
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # 从输入的token id创建位置 id。任何填充的 token 仍然保持填充
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # 如果需要，扩展嵌入
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 选择适当位置的嵌入，将结果 reshape 后返回
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        用位置数字替换非填充符号。位置编号从 padding_idx+1 开始。填充符号将被忽略。这是从 fairseq 的 `utils.make_positions` 修改的。
        
        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # 仅保留非填充符号的 mask
        mask = input_ids.ne(padding_idx).int()
        # 生成增量索引，考虑已处理的 key values 长度
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        # 返回位置 id，加上填充索引
        return incremental_indices.long() + padding_idx
# 从transformers.models.bart.modeling_bart.BartAttention中复制了代码，并将类名由BartAttention改为Speech2Text2Attention
class Speech2Text2Attention(nn.Module):
    """来自“Attention Is All You Need”论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Speech2Text2Config] = None,
    ):
        super().__init__()
        # 初始化注意力模型的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查embed_dim是否可以被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除 (得到 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads})."
            )
        # 缩放系数，用于注意力分数的缩放
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性映射，用于计算查询、键和值的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 将张量重新排列以适应多头注意力机制
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，用于计算注意力
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
class Speech2Text2DecoderLayer(nn.Module):
    def __init__(self, config: Speech2Text2Config):
        super().__init__()
        # 获取模型配置中的维度参数
        self.embed_dim = config.d_model

        # 初始化自注意力层
        self.self_attn = Speech2Text2Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 自注意力层的Layer Normalization
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 如果是解码器，初始化编码器-解码器注意力层
        if config.is_decoder:
            self.encoder_attn = Speech2Text2Attention(
                self.embed_dim,
                config.decoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 第一个全连接层和第二个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 最终的Layer Normalization
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    # 前向传播函数，用于模型推理过程中计算隐藏状态的前向传播
    def forward(
        self,
        # 当前层的隐藏状态，即模型当前处理的输入
        hidden_states: torch.Tensor,
        # 注意力掩码，用于指定哪些位置需要被注意
        attention_mask: Optional[torch.Tensor] = None,
        # 编码器隐藏状态，用于进行编码器-解码器注意力
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器的注意力掩码
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 层级头掩码，用于控制每个头的行为
        layer_head_mask: Optional[torch.Tensor] = None,
        # 跨层级注意力头的掩码
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        # 需要保留的键值对，用于长序列解码
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 是否输出注意力矩阵
        output_attentions: Optional[bool] = False,
        # 是否使用缓存，用于更快的推理
        use_cache: Optional[bool] = True,
# 定义一个名为 Speech2Text2PreTrainedModel 的类，它继承自 PreTrainedModel
class Speech2Text2PreTrainedModel(PreTrainedModel):
    # 指定配置类为 Speech2Text2Config
    config_class = Speech2Text2Config
    # 指定基础模型的前缀为 "model"
    base_model_prefix = "model"
    # 声明该模型支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重的方法
    def _init_weights(self, module):
        # 从配置中获取初始化的标准差
        std = self.config.init_std
        # 如果模块是线性层或者一维卷积层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 对权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 对权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，则将填充索引位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# Speech2Text2PreTrainedModel 类的文档字符串
SPEECH_TO_TEXT_2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Speech2Text2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个名为 Speech2Text2Decoder 的类，它继承自 Speech2Text2PreTrainedModel
class Speech2Text2Decoder(Speech2Text2PreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`Speech2Text2DecoderLayer`]

    Args:
        config: Speech2Text2Config
        embed_tokens (nn.Embedding): output embedding
    """

    # 初始化方法
    def __init__(self, config: Speech2Text2Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置层间隔 dropout 概率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_target_positions
        # 设置嵌入尺度
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 初始化输出嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 初始化位置嵌入
        self.embed_positions = Speech2Text2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
            self.padding_idx,
        )

        # 创建多层解码器
        self.layers = nn.ModuleList([Speech2Text2DecoderLayer(config) for _ in range(config.decoder_layers)])

        # 是否使用梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    # 定义模型的前向传播函数，接收一系列输入参数
    def forward(
        self,
        input_ids=None,  # 输入序列的token IDs
        attention_mask=None,  # 表示哪些token需要被attention（mask掉的token为0，未mask掉的为1）
        encoder_hidden_states=None,  # 编码器的隐藏状态
        encoder_attention_mask=None,  # 编码器的attention mask
        head_mask=None,  # 多头注意力的mask
        cross_attn_head_mask=None,  # 用于跨注意力头的mask
        past_key_values=None,  # 用于缓存的键值对
        inputs_embeds=None,  # 输入的嵌入表示
        use_cache=None,  # 是否使用缓存
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出所有层的隐藏状态
        return_dict=None,  # 是否返回字典格式的输出
# 使用add_start_docstrings装饰器添加模型文档字符串注释
@add_start_docstrings(
    "The Speech2Text2 Model with a language modeling head. Can be used for summarization.",
    SPEECH_TO_TEXT_2_START_DOCSTRING,
)
# 定义一个名为Speech2Text2DecoderWrapper的类，它是Speech2Text2PreTrainedModel的子类
class Speech2Text2DecoderWrapper(Speech2Text2PreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    # 初始化方法，接受config参数
    def __init__(self, config):
        # 调用父类Speech2Text2PreTrainedModel的初始化方法
        super().__init__(config)
        # 创建一个名为decoder的Speech2Text2Decoder实例
        self.decoder = Speech2Text2Decoder(config)

    # 前向传播方法，接受任意数量的位置参数和关键字参数
    def forward(self, *args, **kwargs):
        # 调用decoder的前向传播方法，并返回其结果
        return self.decoder(*args, **kwargs)


# 使用add_start_docstrings装饰器添加模型文档字符串注释
@add_start_docstrings(
    "The Speech2Text2 Decoder with a language modeling head. Can be used as the decoder part of"
    " [`EncoderDecoderModel`] and [`SpeechEncoderDecoder`].",
    SPEECH_TO_TEXT_2_START_DOCSTRING,
)
# 定义一个名为Speech2Text2ForCausalLM的类，它是Speech2Text2PreTrainedModel的子类
class Speech2Text2ForCausalLM(Speech2Text2PreTrainedModel):
    # 定义一个名为_tied_weights_keys的类属性，值为["lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受config参数
    def __init__(self, config):
        # 复制config对象
        config = copy.deepcopy(config)
        # 设置config的is_decoder属性为True，is_encoder_decoder属性为False
        config.is_decoder = True
        config.is_encoder_decoder = False
        # 调用父类Speech2Text2PreTrainedModel的初始化方法
        super().__init__(config)
        # 创建一个名为model的Speech2Text2DecoderWrapper实例
        self.model = Speech2Text2DecoderWrapper(config)

        # 创建一个全连接层，输入大小为config.hidden_size，输出大小为config.vocab_size，不使用偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        # 调用post_init方法
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 前向传播方法，接受一系列输入参数，返回CausalLMOutputWithCrossAttentions类型的输出
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 准备生成输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    # 如果模型被用作编码器-解码器模型中的解码器，那么解码器的注意力掩码将动态创建
    if attention_mask is None:
        # 创建一个与输入形状相同的全1张量作为注意力掩码
        attention_mask = input_ids.new_ones(input_ids.shape)

    if past_key_values:
        # 获取过去键值对的长度
        past_length = past_key_values[0][0].shape[2]

        # 一些生成方法可能只传递最后一个输入 ID
        if input_ids.shape[1] > past_length:
            # 如果输入 ID 的长度大于过去的长度，保留过去长度个字符之后的部分
            remove_prefix_length = past_length
        else:
            # 默认行为：仅保留最后一个输入 ID
            remove_prefix_length = input_ids.shape[1] - 1

        # 更新输入 ID，去掉前缀
        input_ids = input_ids[:, remove_prefix_length:]
    # 第一步，解码器缓存状态为空
    return {
        "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
    }

@staticmethod
def _reorder_cache(past_key_values, beam_idx):
    # 重新排序过去的键值对，根据 beam_idx（束搜索的索引）
    reordered_past = ()
    # 遍历过去的键值对
    for layer_past in past_key_values:
        # 为每一层重新排序过去的状态
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past
```