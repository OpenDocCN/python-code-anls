# `.\transformers\models\pegasus_x\modeling_pegasus_x.py`

```
# 设定文件编码为 UTF-8
# 版权声明和许可证信息
# 这个模型是 PyTorch 下的 PEGASUS-X 模型

import dataclasses  # 引入 dataclasses 模块，用于创建不可变数据类
import math  # 引入 math 模块，提供数学运算函数
from typing import Optional, Tuple, Union  # 引入类型提示相关的模块

import numpy as np  # 引入 numpy 库，用于数组操作
import torch  # 引入 torch 库
import torch.utils.checkpoint  # 引入 torch.utils.checkpoint 模块，用于进行内存检查
from torch import nn  # 从 torch 中引入 nn 模块，用于定义神经网络
from torch.nn import CrossEntropyLoss  # 引入交叉熵损失函数

# 从相关模块中引入所需函数和类
from ...activations import ACT2FN  
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel  # 引入模型相关的实用工具
from ...utils import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 从 PEGASUS-X 配置文件中引入配置类
from .configuration_pegasus_x import PegasusXConfig  


logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "google/pegasus-x-base"  # 用于文档的检查点
_CONFIG_FOR_DOC = "PegasusXConfig"  # 用于文档的配置信息

# PEGASUS-X 预训练模型的存档列表
PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/pegasus-x-base",
    "google/pegasus-x-large",
    # See all PEGASUS models at https://huggingface.co/models?filter=pegasus-x
]


@dataclasses.dataclass  # 使用 dataclasses 装饰器创建数据类
class DimensionInfo:
    """维度信息的包装类。"""

    batch_size: int  # 批大小
    seq_len: int  # 序列长度
    block_size: int  # 块大小
    num_heads: int  # 头数
    hidden_dim: int  # 隐藏层维度
    dim_per_head: int  # 每个头的维度
    num_blocks: int  # 块数量
    global_len: int  # 全局长度
    padded_seq_len: int  # 填充后的序列长度

    # 注意：与原始的 Flax 实现相比，我们将在编码器层开始时将令牌表示填充到块大小的倍数，因此 T=P 总是成立。

# 从 transformers.models.bart.modeling_bart.shift_tokens_right 复制的函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的 token 向右移动一个位置。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)  # 创建与 input_ids 形状相同的全零张量
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # 将 input_ids 向右移动一位
    shifted_input_ids[:, 0] = decoder_start_token_id  # 将 decoder_start_token_id 放置在首位

    if pad_token_id is None:  # 如果 pad_token_id 未定义
        raise ValueError("self.model.config.pad_token_id has to be defined.")  # 抛出错误：self.model.config.pad_token_id 必须定义
    # 将标签中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
```  
    # 返回经过移位后的输入 ID
    return shifted_input_ids
# 定义一个可产生任意长度的正弦波位置编码的模块
class PegasusXSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    # 初始化函数，设定嵌入维度和最大比例尺
    def __init__(self, embed_dim, max_scale: int = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_scale = max_scale

    # 前向传播函数，输入为输入嵌入和过去的 key-value 长度
    @torch.no_grad()
    def forward(self, input_embeds: torch.Tensor, past_key_values_length: int = 0) -> torch.Tensor:
        """input_ids_shape is expected to be [bsz x seqlen]."""
        # 获取批次大小和序列长度
        batch_size, seq_len = input_embeds.shape[:2]
        # 计算位置索引
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=input_embeds.device
        )[:, None]
        # 创建位置编码张量
        pe = torch.zeros((seq_len, self.embed_dim), device=input_embeds.device, dtype=input_embeds.dtype)
        # 计算正弦波的一半维度
        half_d_feature = self.embed_dim // 2
        # 计算指数项系数
        div_term = torch.exp(
            torch.arange(half_d_feature, device=input_embeds.device, dtype=input_embeds.dtype)
            * -(np.log(float(self.max_scale)) / (half_d_feature - 1))
        )
        # 计算正弦和余弦分量
        pe[:, :half_d_feature] = torch.sin(positions * div_term)
        pe[:, half_d_feature:] = torch.cos(positions * div_term)
        # 扩展到批次维度
        return pe[None].expand(batch_size, -1, -1)


# 复制 transformers.models.bart.modeling_bart.BartAttention 模块
class PegasusXAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化函数，设定注意力机制的参数
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PegasusXConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查嵌入维度是否能被头数整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 定义线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 重塑张量以适应多头注意力
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义一个前向传播函数，接收隐藏状态、键值状态、过去的键值、注意力掩码、层头掩码和是否输出注意力矩阵的参数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态张量
        key_value_states: Optional[torch.Tensor] = None,  # 键值状态张量，可选参数
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 过去的键值，可选参数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选参数
        layer_head_mask: Optional[torch.Tensor] = None,  # 层头掩码，可选参数
        output_attentions: bool = False,  # 是否输出注意力矩阵，默认为False
class PegasusXGlobalLocalAttention(nn.Module):
    """Global + Local attention. For use with Encoder only."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
    ):
        super().__init__()
        # 初始化 PegasusXGlobalLocalAttention 类的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # 如果 embed_dim 不能被 num_heads 整除，则引发 ValueError
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 线性变换层，用于查询、键和值的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入张量重塑为(batch_size, seq_len, num_heads, head_dim)，并转置以匹配注意力计算的维度顺序
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # PegasusXGlobalLocalAttention 类的前向传播方法
        # 接受token_hidden_states（本地信息）和global_hidden_states（全局信息）
        # 可能还包括注意力掩码和是否输出注意力分数的选项
        # 这里应该还有一些代码，但它在注释之外
        pass

    def compute_global_attention_representations(
        self, global_q, global_k, global_v, local_k, local_v, mask, dim: DimensionInfo
    ):
        # 计算全局注意力表示的方法
        # 通过全局查询、键和值以及局部键和值计算全局注意力
        # 可能还包括掩码和维度信息的处理
        # 这里应该还有一些代码，但它在注释之外
        pass
    # 计算全局 token 的注意力表示
    def compute_global_attention_representations(
        self,
        global_q, # 全局 token 的查询向量
        global_k, # 全局 token 的键向量
        global_v, # 全局 token 的值向量
        local_k, # 本地 token 的键向量
        local_v, # 本地 token 的值向量
        mask, # 注意力掩码
        dim # 维度信息
    ):
        # 拼接全局和本地的键向量
        global_and_local_k = torch.cat([global_k, local_k], dim=2)
        # 拼接全局和本地的值向量
        global_and_local_v = torch.cat([global_v, local_v], dim=2)
    
        # 对注意力掩码进行填充
        extended_mask = nn.functional.pad(mask, pad=(dim.global_len, 0), value=0)
    
        # 计算全局 token 对所有 token 的注意力权重
        attn_weights = torch.einsum("BHGF,BHXF->BHGX", global_q, global_and_local_k)
        attn_weights = attn_weights + extended_mask[:, None, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    
        # 计算全局 token 的注意力输出
        attn_output = torch.einsum("BHGX,BHXF->BHGF", attn_probs, global_and_local_v)
        return attn_output, attn_probs
    
    # 计算本地 token 的注意力表示
    def compute_local_attention_representations(
        self,
        global_k, # 全局 token 的键向量
        global_v, # 全局 token 的值向量
        local_q, # 本地 token 的查询向量
        local_k, # 本地 token 的键向量
        local_v, # 本地 token 的值向量
        mask, # 注意力掩码
        dim # 维度信息
    ):
class PegasusXEncoderLayer(nn.Module):
    # 定义 PegasusXEncoderLayer 类
    def __init__(self, stagger_blocks_this_layer: bool, config: PegasusXConfig):
        # 初始化函数
        super().__init__()
        # 调用父类的初始化方法
        self.embed_dim = config.d_model
        # 设置嵌入维度为配置中的模型维度
        self.self_attn = PegasusXGlobalLocalAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            block_size=config.block_size,
            dropout=config.attention_dropout,
        )
        # 初始化全局-本地自注意力模块
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化自注意力层归一化
        self.global_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化全局自注意力层归一化
        self.dropout = config.dropout
        # 设置 dropout 概率为配置中的 dropout
        self.activation_fn = ACT2FN[config.activation_function]
        # 激活函数为配置中指定的激活函数
        self.activation_dropout = config.activation_dropout
        # 激活函数的 dropout 概率为配置中的 activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 初始化第一个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 初始化第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化最终层归一化
        self.stagger_blocks_this_layer = stagger_blocks_this_layer
        # 设置本层是否交错块的标志
        self.block_size = config.block_size
        # 设置块大小为配置中的块大小

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        # 正向传播函数
        pass
        # 略过，待实现

    @classmethod
    def pad_local_tokens(cls, hidden_states, attention_mask, block_size):
        # 类方法：填充本地标记
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # 输入隐藏状态的形状
        pad_size = block_size // 2
        # 计算填充大小
        mask_min_value = torch.finfo(hidden_states.dtype).min
        # 获取掩码的最小值
        padded_hidden_states = torch.nn.functional.pad(
            hidden_states,
            pad=(0, 0, pad_size, pad_size),
        )
        # 填充隐藏状态
        padded_mask = torch.nn.functional.pad(
            attention_mask,
            pad=(pad_size, pad_size),
            value=mask_min_value,
        )
        # 填充掩码
        return padded_hidden_states, padded_mask
        # 返回填充后的隐藏状态和掩码

    @classmethod
    def unpad_local_tokens(cls, padded_hidden_states, block_size):
        # 类方法：取消填充本地标记
        # padded_hidden_states: [batch_size, padded seq_len, hidden_dim]
        # 填充后的隐藏状态的形状
        pad_size = block_size // 2
        # 计算填充大小
        return padded_hidden_states[:, pad_size:-pad_size, :]
        # 返回取消填充后的隐藏状态


class PegasusXDecoderLayer(nn.Module):
    # 定义 PegasusXDecoderLayer 类
    # 初始化 PegasusXDecoderLayer 类的实例，传入配置信息
    def __init__(self, config: PegasusXConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model

        # 初始化自注意力层
        self.self_attn = PegasusXAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
        )
        # 设置dropout率
        self.dropout = config.dropout
        # 获取激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的dropout率
        self.activation_dropout = config.activation_dropout

        # 初始化自注意力层的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化编码器注意力层
        self.encoder_attn = PegasusXAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
        )
        # 初始化编码器注意力层的 LayerNorm
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化第一个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 初始化第二个全连接层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 初始化最终的 LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
# 这个类继承自 PreTrainedModel，是 PEGASUS-X 预训练模型的基类
class PegasusXPreTrainedModel(PreTrainedModel):
    # 指定配置类为 PegasusXConfig
    config_class = PegasusXConfig
    # 指定基础模型前缀为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不可拆分的模块
    _no_split_modules = [r"PegasusXEncoderLayer", r"PegasusXDecoderLayer"]

    # 初始化权重的方法
    def _init_weights(self, module):
        # 获取初始化标准差
        std = self.config.init_std
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置项，则将其初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)


# PEGASUS-X 模型的文档字符串
PEGASUS_X_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PegasusXConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# PEGASUS-X 生成示例的文档字符串
PEGASUS_X_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, PegasusXForConditionalGeneration

    >>> model = PegasusXForConditionalGeneration.from_pretrained("google/pegasus-x-base")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")

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
    ```
"""

# PEGASUS-X 输入文档字符串
PEGASUS_X_INPUTS_DOCSTRING = r"""
"""


# PEGASUS-X 编码器类
class PegasusXEncoder(PegasusXPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PegasusXEncoderLayer`].

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化 Encoder 类
    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        self.dropout = config.dropout  # 设置 dropout
        self.layerdrop = config.encoder_layerdrop  # 设置层丢弃率

        embed_dim = config.d_model  # 获取嵌入维度
        self.max_source_positions = config.max_position_embeddings  # 获取最大源位置
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 计算嵌入缩放因子

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens  # 如果传入了嵌入矩阵，使用传入的嵌入矩阵
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim)  # 否则，创建一个嵌入矩阵

        self.embed_global = nn.Embedding(config.num_global_tokens, embed_dim)  # 创建全局嵌入矩阵
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(embed_dim)  # 创建位置编码器
        # 创建编码器层列表
        self.layers = nn.ModuleList(
            [
                PegasusXEncoderLayer(
                    stagger_blocks_this_layer=i % 2 == 1 and config.stagger_local_blocks, config=config
                )
                for i in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)  # 创建层归一化模块

        self.gradient_checkpointing = False  # 梯度检查点初始化为 False
        # 初始化权重并应用最终处理
        self.post_init()

    # 调整位置嵌入矩阵大小
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

        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(self.config.d_model)  # 更新位置编码器
        self.embed_positions.to(self.device)  # 将位置���码器移动到设备上

    # 获取位置嵌入矩阵
    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        return self.embed_positions

    # 前向传播方法
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
class PegasusXDecoder(PegasusXPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`PegasusDecoderLayer`]

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 初始化 PegasusXDecoder 类
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            # 创建默认的嵌入层
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # 创建位置编码
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(config.d_model)
        self.layers = nn.ModuleList([PegasusXDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

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
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,



@add_start_docstrings(
    "The bare PEGASUS-X Model outputting raw hidden-states without any specific head on top.",
    PEGASUS_X_START_DOCSTRING,
)
class PegasusXModel(PegasusXPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: PegasusXConfig):
        # 初始化 PegasusXModel 类
        super().__init__(config)

        vocab_size = config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model)

        # 创建 PegasusXEncoder 和 PegasusXDecoder
        self.encoder = PegasusXEncoder(config, self.shared)
        self.decoder = PegasusXDecoder(config, self.shared)

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

    def get_decoder(self):
        return self.decoder
    # 调整模型的位置嵌入矩阵当 `new_num_position_embeddings != config.max_position_embeddings` 时
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
        # 更新模型配置中的最大位置嵌入数量
        self.config.max_position_embeddings = new_num_position_embeddings
        # 调整编码器的位置嵌入
        self.encoder.resize_position_embeddings(new_num_position_embeddings)
        # 调整解码器的位置嵌入
        self.decoder.resize_position_embeddings(new_num_position_embeddings)

    # 返回位置嵌入矩阵
    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.encoder.get_position_embeddings(), self.decoder.get_position_embeddings())

    # 重写 forward 方法，传入各种输入参数，返回 Seq2SeqModelOutput 类型的结果
    @add_start_docstrings_to_model_forward(PEGASUS_X_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ...
        # (此处省略部分输入参数，具体内容不是重点，故省略)
        # ...
@add_start_docstrings("The PEGASUS-X for conditional generation (e.g. summarization).", PEGASUS_X_START_DOCSTRING)
# 使用给定的文档字符串将注释添加到 PEGASUS-X 类上，以便提供关于其用途的说明
class PegasusXForConditionalGeneration(PegasusXPreTrainedModel):
    # 设置基础模型前缀为 "model"
    base_model_prefix = "model"
    # 定义共享权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: PegasusXConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化 PEGASUS-X 模型
        self.model = PegasusXModel(config)
        # 初始化语言模型头部
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_encoder(self):
        # 返回模型的编码器
        return self.model.get_encoder()

    def get_decoder(self):
        # 返回模型的解码器
        return self.model.get_decoder()

    def get_output_embeddings(self):
        # 返回语言模型头部的输出嵌入
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的输出嵌入
        self.lm_head = new_embeddings

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
        # 调整模型的位置嵌入矩阵
        self.config.max_position_embeddings = new_num_position_embeddings
        # 调整编码器的位置嵌入
        self.model.encoder.resize_position_embeddings(new_num_position_embeddings)
        # 调整解码器的位置嵌入
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        # 返回位置嵌入矩阵
        return (self.model.encoder.get_position_embeddings(), self.model.decoder.get_position_embeddings())

    @add_start_docstrings_to_model_forward(PEGASUS_X_INPUTS_DOCSTRING)
    # 使用给定的文档字符串将注释添加到模型正向传播方法上
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 使用给定的文档字符串将注释添加到返回值的说明上
    @add_end_docstrings(PEGASUS_X_GENERATION_EXAMPLE)
    # 使用给定的文档字符串将注释添加到模型正向传播方法的结尾处
    # 定义 Transformer 模型的前向传播函数
    def forward(
        # 输入序列的 token IDs，可选的 Torch 张量，默认为 None
        input_ids: Optional[torch.Tensor] = None,
        # 注意力遮罩，可选的 Torch 张量，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器输入序列的 token IDs，可选的 Torch 张量，默认为 None
        decoder_input_ids: Optional[torch.Tensor] = None,
        # 解码器的注意力遮罩，可选的 Torch 张量，默认为 None
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # 编码器的输出，可选的 Torch 浮点数元组，默认为 None
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        # 过去的键值对，可选的 Torch 浮点数元组，默认为 None
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        # 输入嵌入，可选的 Torch 张量，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器输入嵌入，可选的 Torch 张量，默认为 None
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        # 标签，可选的 Torch 张量，默认为 None
        labels: Optional[torch.Tensor] = None,
        # 是否使用缓存，可选的布尔值，默认为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选的布尔值，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出，可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        # 定义函数的输入参数和返回值类型，此处返回值类型可以是元组或者Seq2SeqLMOutput类型

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果return_dict已经被定义，则使用已定义的值，否则使用self.config.use_return_dict的值

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # 当labels已经被定义时，强制将use_cache设置为False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
            # 当decoder_input_ids和decoder_inputs_embeds都没有定义时，通过shift_tokens_right函数将labels右移一位，并赋给decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用self.model函数，传入各种参数，获取outputs

        lm_logits = self.lm_head(outputs[0])
        # 将outputs的第一个元素通过self.lm_head函数得到lm_logits

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        # 当labels已经被定义时，将lm_logits通过CrossEntropyLoss函数得到masked_lm_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # 当return_dict为False时，将lm_logits和outputs的后续元素组成output返回

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
        # 当return_dict为True时，将masked_lm_loss、lm_logits和outputs的各种属性组成Seq2SeqLMOutput对象返回

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
        # 定义函数prepare_inputs_for_generation的输入参数
    ):
        # 如果过去的键值存在，则裁剪 decoder_input_ids
        if past_key_values is not None:
            # 获取过去的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                # 要移除的前缀长度为过去的长度
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 裁剪 decoder_input_ids
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs 已定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能是为了调试）
        }

    # 从标签准备解码器的输入 ids
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签向右移动一位，填充用 pad_token_id，起始符号使用 decoder_start_token_id
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    # 重新排序缓存
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                # 对过去的状态执行重新排序
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 从transformers.models.bart.modeling_bart.BartDecoderWrapper复制代码，将Bart替换为PegasusX
class PegasusXDecoderWrapper(PegasusXPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在使用因果语言模型与EncoderDecoderModel框架结合时正确加载预训练的检查点。
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化PegasusXDecoder实例
        self.decoder = PegasusXDecoder(config)

    def forward(self, *args, **kwargs):
        # 调用PegasusXDecoder的forward方法
        return self.decoder(*args, **kwargs)
```