# `.\transformers\models\blenderbot_small\modeling_blenderbot_small.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Facebook, Inc. 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用本文件
# 除非符合许可证要求或书面同意，否则不得使用本文件
# 您可以在以下网址获得许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件基于“原样”提供，
# 没有任何明示或暗示的保证或条件
# 请参阅许可证了解特定语言的权限和限制
""" PyTorch BlenderbotSmall model."""


# 导入所需库和模块
import copy  # 复制对象的库
import math  # 数学运算库
from typing import List, Optional, Tuple, Union  # 类型提示的库

import torch  # PyTorch 库
import torch.utils.checkpoint  # PyTorch 的检查点库
from torch import nn  # PyTorch 神经网络库
from torch.nn import CrossEntropyLoss  # 交叉熵损失函数

# 导入相关工具函数和模块
from ...activations import ACT2FN  # 激活函数映射
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 准备注意力遮罩的工具函数
from ...modeling_outputs import (  # 模型输出相关类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 预训练模型基类
from ...utils import (  # 工具函数
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_blenderbot_small import BlenderbotSmallConfig  # BlenderbotSmall 模型配置类


logger = logging.get_logger(__name__)  # 获取日志记录器


_CONFIG_FOR_DOC = "BlenderbotSmallConfig"  # 用于文档的配置类名称


BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST = [  # BlenderbotSmall 预训练模型列表
    "facebook/blenderbot_small-90M",
    # 查看所有 BlenderbotSmall 模型的列表请访问：https://huggingface.co/models?filter=blenderbot_small
]


# 从 transformers.models.bart.modeling_bart 中复制的函数，用于向右移动输入标记
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)  # 创建与输入形状相同的零张量
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # 将输入标记向右移动一个位置
    shifted_input_ids[:, 0] = decoder_start_token_id  # 将起始标记设置为解码器起始标记

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")  # 如果 pad_token_id 未定义，则引发 ValueError
    # 将标签中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids  # 返回向右移动的输入标记


# 从 transformers.models.blenderbot.modeling_blenderbot 中复制的类，用于学习位置嵌入
class BlenderbotSmallLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)  # 调用父类的初始化方法
    # 定义了一个名为 forward 的方法，用于计算位置编码
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 从输入的形状中获取 batch size 和 sequence length
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置编码的索引，范围是 [past_key_values_length, past_key_values_length + seq_len)，类型为 long
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的 forward 方法计算位置编码
        return super().forward(positions)
# 从transformers.models.bart.modeling_bart.BartAttention复制而来，将Bart->BlenderbotSmall
class BlenderbotSmallAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BlenderbotSmallConfig] = None,
    ):
        super().__init__()
        # 初始化注意力层的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 确保embed_dim能够被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能够被num_heads整除 (得到 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads})."
            )
        # 缩放因子，用于归一化注意力分数
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化投影矩阵
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 重塑输入张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# 从transformers.models.bart.modeling_bart.BartEncoderLayer复制而来，将Bart->BlenderbotSmall，BART->BLENDERBOT_SMALL
class BlenderbotSmallEncoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        # 获取配置中的模型维度
        self.embed_dim = config.d_model

        # 初始化自注意力层
        self.self_attn = BLENDERBOT_SMALL_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        # 自注意力层的Layer Norm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        # 前馈神经网络的第一层和第二层
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 最终的Layer Norm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    # 定义了一个方法，用于前向传播
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入层的输入，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): 注意力掩码，大小为 `(batch, 1, tgt_len, src_len)`，
                其中填充元素由非常大的负值指示。
            layer_head_mask (`torch.FloatTensor`): 给定层中注意力头的掩码，大小为 `(encoder_attention_heads,)`。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。更多细节请参阅返回的张量中的 `attentions`。
        """
        # 保存残差连接
        residual = hidden_states
        # 使用自注意力层计算输出
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对输出应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 对输出应用自注意力层的 LayerNormalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存残差连接
        residual = hidden_states
        # 使用激活函数和全连接层 fc1 计算输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对输出应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用全连接层 fc2 计算输出
        hidden_states = self.fc2(hidden_states)
        # 对输出应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 对输出应用最终的 LayerNormalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果输出类型为 torch.float16 且存在无穷大或 NaN 值，则进行修剪
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出
        return outputs
# 将 BlenderbotSmallAttention 类添加到 BLENDERBOT_SMALL_ATTENTION_CLASSES 字典中
BLENDERBOT_SMALL_ATTENTION_CLASSES = {
    "eager": BlenderbotSmallAttention,
}

# 从 transformers.models.bart.modeling_bart.BartDecoderLayer 复制并修改为 BlenderbotSmallDecoderLayer
# 将 Bart 改为 BlenderbotSmall，BART 改为 BLENDERBOT_SMALL
class BlenderbotSmallDecoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__()
        # 获取模型的嵌入维度
        self.embed_dim = config.d_model

        # 使用配置中指定的注意力实现类创建自注意力机制
        self.self_attn = BLENDERBOT_SMALL_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 获取激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的 dropout 概率
        self.activation_dropout = config.activation_dropout

        # 创建自注意力机制的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 使用配置中指定的注意力实现类创建编码器-解码器注意力机制
        self.encoder_attn = BLENDERBOT_SMALL_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 创建编码器-解码器注意力机制的 LayerNorm 层
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 创建全连接层的第一个线性变换
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 创建全连接层的第二个线性变换
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 创建最终的 LayerNorm 层
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
    ):
        # 此处应该有模型的前向传播逻辑，但由于代码截断，无法提供完整的注释
        pass

# BlenderbotSmallPreTrainedModel 类的定义
class BlenderbotSmallPreTrainedModel(PreTrainedModel):
    # BlenderbotSmallConfig 类作为配置类
    config_class = BlenderbotSmallConfig
    # 模型的基础名称前缀
    base_model_prefix = "model"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重
    def _init_weights(self, module):
        # 从配置中获取初始化的标准差
        std = self.config.init_std
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，将其权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 属性装饰器
    @property
    # 生成虚拟输入数据的方法
    def dummy_inputs(self):
        # 获取配置中的填充标记ID
        pad_token = self.config.pad_token_id
        # 创建输入ID张量，其中包含两个子张量，每个子张量代表一个句子的输入ID序列
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典，包含注意力掩码、输入ID以及解码器输入ID
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 根据填充标记ID生成注意力掩码张量
            "input_ids": input_ids,  # 将输入ID张量加入字典
            "decoder_input_ids": input_ids,  # 将输入ID张量作为解码器的输入ID加入字典
        }
        # 返回虚拟输入字典
        return dummy_inputs
BLENDERBOT_SMALL_START_DOCSTRING = r"""
    # 这个模型继承自`PreTrainedModel`。查看超类文档以获取库实现的通用方法（如下载或保存、调整输入嵌入、剪枝头等）。

    # 这个模型也是一个 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。
    # 将其用作常规的 PyTorch 模块，并参考 PyTorch 文档以获取所有与一般用法和行为相关的事项。

    # 参数:
    #     config ([`BlenderbotSmallConfig`]):
    #         具有模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，仅加载配置。查看
    #         [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

BLENDERBOT_SMALL_GENERATION_EXAMPLE = r"""
    # 对话示例:

    ```python
    >>> from transformers import AutoTokenizer, BlenderbotSmallForConditionalGeneration

    >>> mname = "facebook/blenderbot_small-90M"
    >>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
    >>> tokenizer = AutoTokenizer.from_pretrained(mname)
    >>> UTTERANCE = "My friends are cool but they eat too many carbs."
    >>> print("Human: ", UTTERANCE)
    # 人类:  我的朋友很酷，但他们吃的碳水化合物太多了。

    >>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
    >>> reply_ids = model.generate(**inputs)
    >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
    # 机器人: 他们吃什么样的碳水化合物？我不太了解碳水化合物。

    >>> REPLY = "I'm not sure"
    >>> print("Human: ", REPLY)
    # 人类: 我不确定

    >>> NEXT_UTTERANCE = (
    ...     "My friends are cool but they eat too many carbs.__end__ __start__what kind of carbs do they eat? "
    ...     "i don't know much about carbs__end__ "
    ...     "__start__ I'm not sure."
    ... )
    >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
    >>> next_reply_ids = model.generate(**inputs)
    >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
    # 机器人: 他们吃了很多碳水化合物。碳水化合物含有大量脂肪、蛋白质和脂肪。
    ```py
"""

BLENDERBOT_SMALL_INPUTS_DOCSTRING = r"""
"""


class BlenderbotSmallEncoder(BlenderbotSmallPreTrainedModel):
    """
    # 由*config.encoder_layers*个自注意力层组成的Transformer编码器。每一层都是一个[`BlenderbotSmallEncoderLayer`]。

    # 参数:
    #     config: BlenderbotSmallConfig
    #     embed_tokens (nn.Embedding): 输出嵌入
    """
    # 初始化方法，接受一个 BlenderbotSmallConfig 类型的参数 config 和一个可选的 nn.Embedding 类型的参数 embed_tokens
    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置 dropout 概率为 config 中的配置
        self.dropout = config.dropout
        # 设置 encoder layer dropout 概率为 config 中的配置
        self.layerdrop = config.encoder_layerdrop

        # 获取 embedding 维度
        embed_dim = config.d_model
        # 获取 padding 的索引
        self.padding_idx = config.pad_token_id
        # 获取最大源序列长度
        self.max_source_positions = config.max_position_embeddings
        # 如果配置了 scale embedding，则设置 embed_scale 为 embedding 维度的平方根，否则为 1.0
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 如果传入了 embed_tokens，则使用传入的 embed_tokens，否则创建一个新的 nn.Embedding
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 创建位置 embedding 对象
        self.embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # 创建多个编码器层，并组成一个 ModuleList
        self.layers = nn.ModuleList([BlenderbotSmallEncoderLayer(config) for _ in range(config.encoder_layers)])
        # 创建一个 LayerNorm 对象，对 embedding 进行归一化处理
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # 是否使用梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
class BlenderbotSmallDecoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BlenderbotSmallDecoderLayer`]

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置层间 dropout 概率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_position_embeddings
        # 设置嵌入缩放因子
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 如果提供了嵌入令牌，则使用提供的嵌入令牌，否则创建一个新的嵌入令牌
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 创建位置嵌入
        self.embed_positions = BlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # 创建多个解码层
        self.layers = nn.ModuleList([BlenderbotSmallDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建层归一化
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点
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
    "The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class BlenderbotSmallModel(BlenderbotSmallPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: BlenderbotSmallConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享的嵌入层
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器
        self.encoder = BlenderbotSmallEncoder(config, self.shared)
        self.decoder = BlenderbotSmallDecoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder
    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 在模型的前向传播过程中，接收一系列输入并生成输出
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    # 替换返回值文档字符串，指定输出类型和配置类
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为长整型张量，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的 token IDs，可选
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器头部掩码，可选
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码，可选
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,  # 编码器输出，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值，可选
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，可选
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入表示，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，可选
# 添加模型文档字符串，描述 BlenderbotSmall 模型及其用途
@add_start_docstrings(
    "The BlenderbotSmall Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
# 定义 BlenderbotSmallForConditionalGeneration 类，继承自 BlenderbotSmallPreTrainedModel
class BlenderbotSmallForConditionalGeneration(BlenderbotSmallPreTrainedModel):
    # 模型参数前缀
    base_model_prefix = "model"
    # 加载时忽略的键
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 共享权重的键
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接受 BlenderbotSmallConfig 类型的参数
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        # 创建 BlenderbotSmallModel 对象
        self.model = BlenderbotSmallModel(config)
        # 注册 final_logits_bias 缓冲区
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建线性层 lm_head
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 大小
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整 final_logits_bias 大小
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出 embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 添加模型前向方法的文档字符串
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    # 替��返回值文档字符串
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向方法结束文档字符串
    @add_end_docstrings(BLENDERBOT_SMALL_GENERATION_EXAMPLE)
    # 定义一个方法用于模型的前向传播
    def forward(
        # 输入的 token IDs，类型为 LongTensor，可选参数，默认为 None
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，类型为 Tensor，可选参数，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器的输入 token IDs，类型为 LongTensor，可选参数，默认为 None
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力掩码，类型为 LongTensor，可选参数，默认为 None
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，类型为 Tensor，可选参数，默认为 None
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，类型为 Tensor，可选参数，默认为 None
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，类型为 Tensor，可选参数，默认为 None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，类型为 Tuple 或 BaseModelOutput，可选参数，默认为 None
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        # 过去的键值对，类型为 List[torch.FloatTensor]，可选参数，默认为 None
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入的嵌入向量，类型为 Tensor，可选参数，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器输入的嵌入向量，类型为 torch.FloatTensor，可选参数，默认为 None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，类型为 LongTensor，可选参数，默认为 None
        labels: Optional[torch.LongTensor] = None,
        # 是否使用缓存，类型为 bool，可选参数，默认为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，类型为 bool，可选参数，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为 bool，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，类型为 bool，可选参数，默认为 None
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        # 设置返回字典，如果未提供则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果提供了标签
        if labels is not None:
            # 如果使用缓存，则更改警告信息
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            # 设置使用缓存为 False
            use_cache = False
            # 如果解码器输入 ID 和解码器输入嵌入均未提供
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # 将标签向右移动一个位置，用于解码器输入
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # 使用模型进行前向传播
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
        # 计算语言模型的 logits
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        # 如果提供了标签
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算掩码语言建模损失
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典
        if not return_dict:
            # 组装输出
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
    # 为生成准备输入数据，用于生成下一个词的序列
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
        # 如果过去键值存在，则截断decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认为旧的行为：仅保留最后一个ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回包含生成所需输入的字典
        return {
            "input_ids": None,  # encoder_outputs已经定义。不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 将此更改为避免缓存（推测是为了调试）
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 对过去键值进行重新排序
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不必重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 从transformers.models.bart.modeling_bart.BartDecoderWrapper复制代码，将Bart->BlenderbotSmall
# 定义了一个BlenderbotSmallDecoderWrapper类，用于正确加载预训练检查点，当因果语言模型与EncoderDecoderModel框架结合使用时。
class BlenderbotSmallDecoderWrapper(BlenderbotSmallPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 实例化一个BlenderbotSmallDecoder对象
        self.decoder = BlenderbotSmallDecoder(config)

    # 前向传播函数
    def forward(self, *args, **kwargs):
        # 调用BlenderbotSmallDecoder对象的前向传播函数
        return self.decoder(*args, **kwargs)


# 从transformers.models.bart.modeling_bart.BartForCausalLM复制代码，将Bart->BlenderbotSmall, facebook/bart-base->facebook/blenderbot_small-90M
class BlenderbotSmallForCausalLM(BlenderbotSmallPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化函数
    def __init__(self, config):
        # 深度复制配置
        config = copy.deepcopy(config)
        # 设置is_decoder为True
        config.is_decoder = True
        # 设置is_encoder_decoder为False
        config.is_encoder_decoder = False
        # 调用父类的初始化函数
        super().__init__(config)
        # 实例化一个BlenderbotSmallDecoderWrapper对象
        self.model = BlenderbotSmallDecoderWrapper(config)

        # 实例化一个线性层作为语言模型头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 前向传播函数
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
    # 为生成准备输入的函数
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