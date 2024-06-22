# `.\transformers\models\blenderbot\modeling_blenderbot.py`

```py
# 指定文件编码为 UTF-8
# 版权声明和许可协议
# 版权所有（c）2021 Facebook, Inc. 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）许可;
# 除非符合许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发的软件
# 在无任何形式的明示或暗示担保的情况下发布。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch Blenderbot 模型。"""


# 导入必要的库和模块
import copy  # 用于复制对象
import math  # 用于数学运算
import os  # 用于处理操作系统相关的功能
import warnings  # 用于警告处理
from typing import List, Optional, Tuple, Union  # 用于类型提示

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 用于 PyTorch 模型的检查点
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import CrossEntropyLoss  # 交叉熵损失函数

# 导入自定义的模块和函数
from ...activations import ACT2FN  # 激活函数映射表
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 注意力掩码相关函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import (  # 导入工具函数
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,  # 日志记录
    replace_return_docstrings,
)
from ..blenderbot_small import BlenderbotSmallForConditionalGeneration, BlenderbotSmallModel  # 导入小型 Blenderbot 模型
from .configuration_blenderbot import BlenderbotConfig  # 导入 Blenderbot 配置类


logger = logging.get_logger(__name__)  # 获取日志记录器

_CONFIG_FOR_DOC = "BlenderbotConfig"  # 配置文件的名称
_CHECKPOINT_FOR_DOC = "facebook/blenderbot-400M-distill"  # 预训练模型的检查点路径


BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # Blenderbot 预训练模型的存档列表
    "facebook/blenderbot-3B",
    # 查看所有 Blenderbot 模型的存档列表：https://huggingface.co/models?filter=blenderbot
]


# 从 transformers.models.bart.modeling_bart.shift_tokens_right 复制的函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的标识向右移动一个标识位。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)  # 创建和 input_ids 形状相同的零张量
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # 将 input_ids 向右移动一位
    shifted_input_ids[:, 0] = decoder_start_token_id  # 将第一个标识替换为 decoder 的起始标识

    if pad_token_id is None:  # 如果 pad_token_id 未定义
        raise ValueError("self.model.config.pad_token_id has to be defined.")  # 报错：必须定义 pad_token_id
    # 将标签中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids  # 返回向右移动后的标识


class BlenderbotLearnedPositionalEmbedding(nn.Embedding):
    """
    该模块学习固定最大大小的位置嵌入。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)  # 调用父类的构造方法初始化嵌入层
    # 定义前向传播方法，用于位置编码
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 从输入的形状中获取批量大小和序列长度
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置编码的索引，从过去键值对长度到过去键值对长度加上序列长度
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的前向传播方法，传入位置编码的索引，返回位置编码张量
        return super().forward(positions)
# 从transformers.models.bart.modeling_bart.BartAttention复制并修改为BlenderbotAttention类
class BlenderbotAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BlenderbotConfig] = None,
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
BLENDERBOT_ATTENTION_CLASSES = {"eager": BlenderbotAttention}

# 从transformers.models.mbart.modeling_mbart.MBartEncoderLayer复制并修改为BlenderbotEncoderLayer类
class BlenderbotEncoderLayer(nn.Module):
    def __init__(self, config: BlenderbotConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BLENDERBOT_ATTENTION_CLASSES[config._attn_implementation](
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
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量，形状为(batch, seq_len, embed_dim)
        attention_mask: torch.Tensor,  # 注意力掩码张量，形状为(batch, 1, tgt_len, src_len)，其中填充元素由极大负值指示
        layer_head_mask: torch.Tensor,  # 给定层中注意力头的掩码张量，形状为(encoder_attention_heads,)
        output_attentions: bool = False,  # 是否返回所有注意力层的注意力张量，默认为False
    ) -> torch.Tensor:  # 返回张量的类型提示
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
        residual = hidden_states  # 将输入的隐藏状态张量保存为残差连接的一部分
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 对输入的隐藏状态进行自注意力层的归一化处理
        hidden_states, attn_weights, _ = self.self_attn(  # 使用自注意力层处理隐藏状态，返回处理后的隐藏状态、注意力权重和额外信息
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对隐藏状态进行dropout操作
        hidden_states = residual + hidden_states  # 使用残差连接将dropout后的隐藏状态与原始输入相加

        residual = hidden_states  # 将上一步得到的隐藏状态保存为残差连接的一部分
        hidden_states = self.final_layer_norm(hidden_states)  # 对隐藏状态进行最终层的归一化处理
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 对归一化后的隐藏状态进行全连接层和激活函数处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 对全连接层输出进行dropout操作
        hidden_states = self.fc2(hidden_states)  # 再次对全连接层输出进行处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对最终输出进行dropout操作
        hidden_states = residual + hidden_states  # 使用残差连接将dropout后的隐藏状态与之前的残差相加

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):  # 如果隐藏状态的数据类型为torch.float16并且存在inf或nan的元素
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000  # 计算截断值
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)  # 对隐藏状态进行截断处理

        outputs = (hidden_states,)  # 将最终的隐藏状态作为输出

        if output_attentions:  # 如果需要输出注意力权重
            outputs += (attn_weights,)  # 将注意力权重添加到输出中

        return outputs  # 返回输出
# 从transformers.models.mbart.modeling_mbart中拷贝了BlenderbotDecoderLayer类，将MBart->Blenderbot，MBART->BLENDERBOT
class BlenderbotDecoderLayer(nn.Module):
    def __init__(self, config: BlenderbotConfig):
        super().__init__()
        # 初始化层的维度为config中的d_model值
        self.embed_dim = config.d_model

        # 创建自注意力层对象，用于解码器内部的自注意力机制
        self.self_attn = BLENDERBOT_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        # 激活函数使用配置中的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 对自注意力层的输出进行LayerNorm归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建编码器注意力层对象，用于解码器与编码器之间的注意力机制
        self.encoder_attn = BLENDERBOT_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 对编码器注意力层的输出进行LayerNorm归一化
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 第一个全连接层，用于FFN网络
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 第二个全连接层，用于FFN网络
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 对FFN网络的输出进行LayerNorm归一化
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
class BlenderbotPreTrainedModel(PreTrainedModel):
    # BlenderbotPreTrainedModel类的配置类为BlenderbotConfig
    config_class = BlenderbotConfig
    # 模型的基础名称前缀为"model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型的权重
    def _init_weights(self, module):
        std = self.config.init_std
        # 如果是线性层，初始化权重为正态分布，均值为0，标准差为config中的init_std值
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置项，将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层，初始化权重为正态分布，均值为0，标准差为config中的init_std值
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有padding_idx，将其对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 返回模型的虚拟输入，用于测试模型
    @property
    def dummy_inputs(self):
        # 获取填充标记
        pad_token = self.config.pad_token_id
        # 创建输入张量
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构造虚拟输入字典
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs


BLENDERBOT_START_DOCSTRING = r"""
    # 这个模型继承自 PreTrainedModel。查看超类文档以了解库实现的通用方法（例如下载或保存、调整输入嵌入、修剪头等）。
    
    # 这个模型还是一个 PyTorch torch.nn.Module 子类。将其用作常规的 PyTorch Module，并参考 PyTorch 文档了解所有与一般用法和行为相关的事项。
    
    # 参数：
    #     config ([BlenderbotConfig]):
    #         包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 ~PreTrainedModel.from_pretrained 方法以加载模型权重。
"""

BLENDERBOT_GENERATION_EXAMPLE = r"""
    Conversation example:

    ```python
    >>> from transformers import AutoTokenizer, BlenderbotForConditionalGeneration

    >>> mname = "facebook/blenderbot-400M-distill"
    >>> model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    >>> tokenizer = AutoTokenizer.from_pretrained(mname)
    >>> UTTERANCE = "My friends are cool but they eat too many carbs."
    >>> print("Human: ", UTTERANCE)
    Human:  My friends are cool but they eat too many carbs.

    >>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
    >>> reply_ids = model.generate(**inputs)
    >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
    Bot: That's unfortunate. Are they trying to lose weight or are they just trying to be healthier?

    >>> REPLY = "I'm not sure"
    >>> print("Human: ", REPLY)
    Human: I'm not sure

    >>> NEXT_UTTERANCE = (
    ...     "My friends are cool but they eat too many carbs.</s> <s>That's unfortunate. "
    ...     "Are they trying to lose weight or are they just trying to be healthier?</s> "
    ...     "<s> I'm not sure."
    ... )
    >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
    >>> next_reply_ids = model.generate(**inputs)
    >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
    Bot:   I see. Well, it's good that they're trying to change their eating habits.
    ```py
"""

BLENDERBOT_INPUTS_DOCSTRING = r"""
"""


class BlenderbotEncoder(BlenderbotPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BlenderbotEncoderLayer`].

    Args:
        config: BlenderbotConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        # 初始化一个指定配置的BlenderbotEncoder实例
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 设置嵌入维度
        embed_dim = config.d_model
        # 设置填充token的索引
        self.padding_idx = config.pad_token_id
        # 设置最大源位置
        self.max_source_positions = config.max_position_embeddings
        # 设置嵌入尺度
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 如果有嵌入标记，则使用给定的；否则，初始化一个新的
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 初始化位置编码
        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # 初始化编码器层
        self.layers = nn.ModuleList([BlenderbotEncoderLayer(config) for _ in range(config.encoder_layers)])
        # 初始化层归一化
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 是否使用梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()
    # 定义一个前向传播函数，接受多个参数
    def forward(
        # 输入的 token IDs
        input_ids=None,
        # 注意力掩码
        attention_mask=None,
        # 头部掩码
        head_mask=None,
        # 输入的嵌入向量
        inputs_embeds=None,
        # 是否输出注意力权重
        output_attentions=None,
        # 是否输出隐藏状态
        output_hidden_states=None,
        # 是否返回字典形式的结果
        return_dict=None,
class BlenderbotDecoder(BlenderbotPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BlenderbotDecoderLayer`]

    Args:
        config: BlenderbotConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置层级丢弃概率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_position_embeddings
        # 设置嵌入缩放因子
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 如果传入了嵌入令牌，则使用传入的，否则创建一个新的嵌入令牌
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 创建学习位置嵌入
        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # 创建多个解码器层
        self.layers = nn.ModuleList([BlenderbotDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建层归一化
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,



@add_start_docstrings(
    "The bare Blenderbot Model outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_START_DOCSTRING,
)
class BlenderbotModel(BlenderbotPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: BlenderbotConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享的嵌入层
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器
        self.encoder = BlenderbotEncoder(config, self.shared)
        self.decoder = BlenderbotDecoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    @classmethod
    # 根据预训练模型名称或路径创建模型实例
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # 如果预训练模型名称或路径为"facebook/blenderbot-90M"
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            # 发出警告，提示该检查点已被弃用，建议使用"facebook/small_blenderbot-90M"检查点
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical"
                " checkpoint `facebook/small_blenderbot-90M` with"
                " `BlenderbotSmallModel.from_pretrained('facebook/small_blenderbot-90M')` instead.",
                FutureWarning,
            )
            # 返回使用"facebook/blenderbot-90M"的BlenderbotSmallModel实例
            return BlenderbotSmallModel.from_pretrained(pretrained_model_name_or_path)

        # 调用父类的from_pretrained方法创建模型实例
        return super(BlenderbotModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
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

    # 前向传播函数，接收多个输入参数
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用给定的初始文档字符串和 Blenderbot 配置创建 Blenderbot 有条件生成模型类
@add_start_docstrings(
    "The Blenderbot Model with a language modeling head. Can be used for summarization.", BLENDERBOT_START_DOCSTRING
)
class BlenderbotForConditionalGeneration(BlenderbotPreTrainedModel):
    # 基础模型前缀为 "model"
    base_model_prefix = "model"
    # 加载缺失键时要忽略的键列表
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 权重绑定的键列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化函数，接受 Blenderbot 配置
    def __init__(self, config: BlenderbotConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 Blenderbot 模型对象
        self.model = BlenderbotModel(config)
        # 注册 final_logits_bias 缓冲区
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建线性层 lm_head
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从预训练模型加载模型
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # 如果预训练模型为 "facebook/blenderbot-90M"，发出警告并返回相应模型
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical"
                " checkpoint `facebook/small_blenderbot-90M` with"
                " `BlenderbotSmallForConditionalGeneration.from_pretrained('facebook/small_blenderbot-90M')` instead.",
                FutureWarning,
            )
            return BlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)

        # 从预训练模型加载模型
        return super(BlenderbotForConditionalGeneration, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token 嵌入
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调整 token 嵌入
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整 final_logits_bias 的大小以匹配新的嵌入维度
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整 final_logits_bias 的大小以匹配新的 token 数量
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于旧的 token 数量
        if new_num_tokens <= old_num_tokens:
            # 截取现有的偏置
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 创建额外的偏置
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            # 拼接偏置
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册调整后的 final_logits_bias
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 将模型前向传播函数的输入文档字符串添加到模型前向传播函数
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    # 替换返回值文档字符串
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向传播函数的结束文档字符串
    @add_end_docstrings(BLENDERBOT_GENERATION_EXAMPLE)
    # 定义了模型的前向传播方法，用于执行输入数据的前向计算，并返回计算结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入序列的token ID，类型为可选的长整型张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，类型为可选的张量，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入序列的token ID，类型为可选的长整型张量，默认为None
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器注意力遮罩，类型为可选的长整型张量，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力机制的头部遮罩，类型为可选的张量，默认为None
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器多头注意力机制的头部遮罩，类型为可选的张量，默认为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力机制的头部遮罩，类型为可选的张量，默认为None
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,  # 编码器输出结果，类型为可选的元组或基础模型输出对象，默认为None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 用于解码器的历史键值对，类型为可选的浮点数张量列表，默认为None
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入向量，类型为可选的张量，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入向量，类型为可选的浮点数张量，默认为None
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为可选的长整型张量，默认为None
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为可选的布尔值，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为可选的布尔值，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值，默认为None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，类型为可选的布尔值，默认为None
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Return a dictionary if return_dict is True, else return a tuple consisting of a torch.FloatTensor and Seq2SeqLMOutput.

        """
        # Determine whether to use the return_dict option from the configuration or the provided argument
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If labels are provided, handle cache usage and decoder input preparation
        if labels is not None:
            # Warn and set use_cache to False if labels are provided
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # If decoder inputs are not provided, shift labels for decoder input preparation
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Pass input data and related parameters to the model for forward pass
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
        
        # Compute logits for language modeling
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        # If labels are provided, compute masked language modeling loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # If return_dict is False, return output as tuple
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # If return_dict is True, construct Seq2SeqLMOutput object and return
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
```  
    # 准备生成的输入参数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器的输入 ID
        past_key_values=None,  # 过去的键值（用于缓存）
        attention_mask=None,  # 注意力掩码
        head_mask=None,  # 头部掩码
        decoder_head_mask=None,  # 解码器头部掩码
        cross_attn_head_mask=None,  # 交叉注意力头部掩码
        use_cache=None,  # 是否使用缓存
        encoder_outputs=None,  # 编码器的输出
        **kwargs,  # 其它参数
    ):
        # 如果使用了过去的键值，则截断 decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认为旧行为：仅保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回生成所需的参数字典
        return {
            "input_ids": None,  # encoder_outputs 已定义。input_ids 不需要
            "encoder_outputs": encoder_outputs,  # 编码器的输出
            "past_key_values": past_key_values,  # 过去的键值
            "decoder_input_ids": decoder_input_ids,  # 解码器的输入 ID
            "attention_mask": attention_mask,  # 注意力掩码
            "head_mask": head_mask,  # 头部掩码
            "decoder_head_mask": decoder_head_mask,  # 解码器头部掩码
            "cross_attn_head_mask": cross_attn_head_mask,  # 交叉注意力头部掩码
            "use_cache": use_cache,  # 更改此参数以避免缓存（可能用于调试）
        }

    # 重新排序缓存
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 对于每一层的过去键值
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终相同
            # 通过使用 beam_idx 对 past_state 进行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],  # 保持未更改的部分
            )
        # 返回重新排序后的过去键值
        return reordered_past
# 从transformers.models.bart.modeling_bart.BartDecoderWrapper复制并修改为BlenderbotDecoderWrapper
class BlenderbotDecoderWrapper(BlenderbotPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在因果语言模型与EncoderDecoderModel框架结合使用时正确加载预训练检查点。
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化BlenderbotDecoder对象
        self.decoder = BlenderbotDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


# 从transformers.models.bart.modeling_bart.BartForCausalLM复制并修改为BlenderbotForCausalLM，将Bart->Blenderbot，facebook/bart-base->facebook/blenderbot-400M-distill
class BlenderbotForCausalLM(BlenderbotPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 初始化BlenderbotDecoderWrapper对象
        self.model = BlenderbotDecoderWrapper(config)

        # 初始化线性层，用于LM头
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
            # 如果模型作为编码器-解码器模型中的解码器使用，则动态创建解码器注意力掩码
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape)

            if past_key_values:
                past_length = past_key_values[0][0].shape[2]

                # 一些生成方法已经只传递最后一个输入 ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认为旧行为：仅保留最后一个 ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]
            # 第一步，decoder_cached_states 为空
            return {
                "input_ids": input_ids,  # encoder_outputs 已定义。不需要 input_ids
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }

        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            reordered_past = ()
            for layer_past in past_key_values:
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            return reordered_past
```