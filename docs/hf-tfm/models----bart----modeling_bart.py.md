# `.\transformers\models\bart\modeling_bart.py`

```
# 设置编码格式为 UTF-8
# 版权声明，使用 Apache License 2.0 进行许可，保留所有权利
# 导入所需库和模块
import copy  # 导入深拷贝模块
import math  # 导入数学函数模块
import warnings  # 导入警告模块
from typing import List, Optional, Tuple, Union  # 导入类型提示模块

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
import torch.utils.checkpoint  # 导入 PyTorch 中的检查点模块
from torch import nn  # 导入 PyTorch 中的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入 PyTorch 中的损失函数

from ...activations import ACT2FN  # 从激活函数模块导入激活函数
from ...modeling_attn_mask_utils import (  # 导入注意力掩码工具模块
    _prepare_4d_attention_mask,  # 用于准备四维注意力掩码的函数
    _prepare_4d_attention_mask_for_sdpa,  # 用于准备用于 SDPA 的四维注意力掩码的函数
    _prepare_4d_causal_attention_mask,  # 用于准备四维因果注意力掩码的函数
    _prepare_4d_causal_attention_mask_for_sdpa,  # 用于准备用于 SDPA 的四维因果注意力掩码的函数
)
from ...modeling_outputs import (  # 导入模型输出模块
    BaseModelOutput,  # 基础模型输出类
    BaseModelOutputWithPastAndCrossAttentions,  # 带有过去和交叉注意力的基础模型输出类
    CausalLMOutputWithCrossAttentions,  # 带有交叉注意力的因果语言建模输出类
    Seq2SeqLMOutput,  # 序列到序列语言建模输出类
    Seq2SeqModelOutput,  # 序列到序列模型输出类
    Seq2SeqQuestionAnsweringModelOutput,  # 序列到序列问答模型输出类
    Seq2SeqSequenceClassifierOutput,  # 序列到序列序列分类器模型输出类
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型工具类
from ...utils import (  # 导入实用函数模块
    add_code_sample_docstrings,  # 添加代码示例文档字符串的函数
    add_end_docstrings,  # 添加结尾文档字符串的函数
    add_start_docstrings,  # 添加起始文档字符串的函数
    add_start_docstrings_to_model_forward,  # 添加起始文档字符串到模型前向函数的函数
    is_flash_attn_2_available,  # 判断是否可用闪存注意力机制的函数
    is_flash_attn_greater_or_equal_2_10,  # 判断闪存注意力机制版本是否大于或等于 2.10 的函数
    logging,  # 日志记录模块
    replace_return_docstrings,  # 替换返回文档字符串的函数
)

if is_flash_attn_2_available():  # 如果可用闪存注意力机制
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # 导入闪存注意力机制相关函数
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # 导入 BERT 填充相关函数

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "facebook/bart-base"  # 用于文档的检查点
_CONFIG_FOR_DOC = "BartConfig"  # 用于文档的配置

_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]  # 期望的输出形状

_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "valhalla/bart-large-sst2"  # 序列分类任务的检查点
_SEQ_CLASS_EXPECTED_LOSS = 0.0  # 预期的损失值
_SEQ_CLASS_EXPECTED_OUTPUT = "'POSITIVE'"  # 预期的输出

_CHECKPOINT_FOR_QA = "valhalla/bart-large-finetuned-squadv1"  # 问答任务的检查点
_QA_EXPECTED_LOSS = 0.59  # 预期的损失值
_QA_EXPECTED_OUTPUT = "' nice puppet'"  # 预期的输出

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [  # BART 预训练模型存档列表
    "facebook/bart-large",  # Facebook BART 大型模型
    # 查看所有 BART 模型，请访问 https://huggingface.co/models?filter=bart
]


def _get_unpad_data(attention_mask):  # 获取取消填充数据的函数
    # 计算批次中的序列长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到非填充部分的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 找到批次中的最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
``` 
    # 计算序列长度的累积和，并在结果中填充指定值
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # 返回三个值：indices，cu_seqlens，max_seqlen_in_batch
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的标记向右移动一个标记位。
    """
    # 创建一个和输入标记形状相同的零张量
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入标记的除了第一个位置外的所有位置向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将第一个位置设置为解码器起始标记的标记ID
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        # 如果未定义pad_token_id，则引发值错误
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    该模块学习位置嵌入，最大大小固定。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart被设置为如果指定了padding_idx，则将嵌入id偏移2，并相应调整num_embeddings。其他模型没有此技巧
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' 的形状预期为 [bsz x seqlen]。"""

        bsz, seq_len = input_ids.shape[:2]
        # 创建一个张量，表示序列的位置编码
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """来自 'Attention Is All You Need' 论文的多头注意力"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BartConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            # 如果 embed_dim 不是 num_heads 的倍数，则引发值错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重新形状，以适应多头自注意力的计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    _mask: Optional[torch.Tensor] = None,
            # 输入参数 layer_head_mask: Optional[torch.Tensor]，表示层头掩码，可选
            layer_head_mask: Optional[torch.Tensor] = None,
            # 输入参数 output_attentions: bool = False，表示是否输出注意力信息，默认为 False
            output_attentions: bool = False,
class BartFlashAttention2(BartAttention):
    """
    Bart flash attention module. This module inherits from `BartAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 检查 Flash Attention 版本是否大于等于 2.1，以确定是否需要使用底部右对齐的掩码
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # 重塑张量形状的辅助函数
    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
        """
        如果输入的隐藏状态中包含至少一个填充令牌，则调用 Flash Attention 的前向方法，
        首先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

        参数:
            query_states (`torch.Tensor`):
                传递给 Flash Attention API 的输入查询状态
            key_states (`torch.Tensor`):
                传递给 Flash Attention API 的输入键状态
            value_states (`torch.Tensor`):
                传递给 Flash Attention API 的输入值状态
            attention_mask (`torch.Tensor`):
                填充掩码 - 对应于大小为 `(batch_size, seq_len)` 的张量，其中 0 表示填充令牌的位置，1 表示非填充令牌的位置。
            dropout (`int`, *optional*):
                注意力丢弃率
            softmax_scale (`float`, *optional*):
                在应用 softmax 之前的 QK^T 缩放。默认为 1 / sqrt(head_dim)
        """
        如果不使用顶部左侧掩码，则 `causal` 设为 `self.is_causal`。否则：
        # TODO: 一旦 Flash Attention for RoCm 升级到 2.1，就删除 `query_length != 1` 检查。有关详细信息，请参见 LlamaFlashAttention2 __init__ 中的注释。
        `causal` 设为 `self.is_causal and query_length != 1`

        # 序列中至少包含一个填充令牌
        如果 attention_mask 不为 None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        否则:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        返回 attn_output

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制过来的
```  
    # 对输入进行处理，根据注意力掩码去除填充部分的数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取批次大小、键值序列长度、键值头数和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 重新索引键值层，去除填充部分的数据
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 重新索引值层，去除填充部分的数据
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 根据查询长度对查询层进行处理
        if query_length == kv_seq_len:
            # 如果查询长度等于键值序列长度，重新索引查询层，去除填充部分的数据
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果查询长度为1，对查询层进行处理，去除填充部分的数据
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，根据注意力掩码去除填充部分的数据
            # 这里的 attention_mask[:, -query_length:] 会假设左填充
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的输入
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义 BartSdpaAttention 类，继承自 BartAttention
class BartSdpaAttention(BartAttention):
    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        key_value_states: Optional[torch.Tensor] = None,  # 键值状态张量，可选
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 过去的键值对，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量，可选
        layer_head_mask: Optional[torch.Tensor] = None,  # 层头遮罩张量，可选
        output_attentions: bool = False,  # 是否输出注意力权重，缺省为 False
BART_ATTENTION_CLASSES = {
    "eager": BartAttention,  # "eager" 类型的注意力使用 BartAttention 类
    "sdpa": BartSdpaAttention,  # "sdpa" 类型的注意力使用 BartSdpaAttention 类
    "flash_attention_2": BartFlashAttention2,  # "flash_attention_2" 类型的注意力使用 BartFlashAttention2 类
}


# 定义 BartEncoderLayer 类，继承自 nn.Module
class BartEncoderLayer(nn.Module):
    # 初始化函数
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 获取配置中的嵌入维度

        # 初始化自注意力层
        self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,  # 嵌入维度
            num_heads=config.encoder_attention_heads,  # 编码器注意力头数
            dropout=config.attention_dropout,  # 注意力层的 dropout
            config=config,  # 配置对象
        )
        # 自注意力层的 LayerNormalization
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout  # dropout 概率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的 dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终层的 LayerNormalization

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.FloatTensor,  # 输入的隐藏状态张量
        attention_mask: torch.FloatTensor,  # 注意力遮罩张量
        layer_head_mask: torch.FloatTensor,  # 层头遮罩张量
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，缺省为 False
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): 大小为 `(batch, 1, tgt_len, src_len)` 的注意力遮罩，
                其中填充元素由非常大的负值表示。
            layer_head_mask (`torch.FloatTensor`): 给定层中注意力头的遮罩，大小为 `(encoder_attention_heads,)`。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多细节。
        """
        # 保存残差连接，用于最后加和
        residual = hidden_states
        # 通过自注意力层进行计算
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 使用 dropout 进行正则化
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 使用层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存残差连接，用于最后加和
        residual = hidden_states
        # 通过全连接层 1 进行计算
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用 dropout 进行正则化
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 通过全连接层 2 进行计算
        hidden_states = self.fc2(hidden_states)
        # 使用 dropout 进行正则化
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 使用层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果张量类型为 torch.float16 并且存在无穷大或 NaN 值
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 将值限制在范围内
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 保存输出张量
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出
        return outputs
```  
class BartDecoderLayer(nn.Module):
    # BART 解码器层的定义
    def __init__(self, config: BartConfig):
        super().__init__()
        # 获取 BART 模型配置中的嵌入维度
        self.embed_dim = config.d_model

        # 定义自注意力层，根据配置选择不同的注意力实现
        self.self_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 定义 dropout 层
        self.dropout = config.dropout
        # 定义激活函数，根据配置选择不同的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 定义激活函数的 dropout 层
        self.activation_dropout = config.activation_dropout

        # 定义自注意力层的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 定义编码器-解码器注意力层，根据配置选择不同的注意力实现
        self.encoder_attn = BART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 定义编码器-解码器注意力层的 LayerNorm 层
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 定义全连接层1
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 定义全连接层2
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 定义最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # BART 解码器层的前向传播
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
class BartClassificationHead(nn.Module):
    # 用于句子级分类任务的头部定义
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        # 定义全连接层
        self.dense = nn.Linear(input_dim, inner_dim)
        # 定义 dropout 层
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 定义输出投影层
        self.out_proj = nn.Linear(inner_dim, num_classes)

    # 头部的前向传播
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 通过全连接层
        hidden_states = self.dense(hidden_states)
        # 使用 tanh 激活函数
        hidden_states = torch.tanh(hidden_states)
        # 再次进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 通过输出投影层得到最终结果
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartPreTrainedModel(PreTrainedModel):
    # BART 预训练模型的基类定义
    config_class = BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # 加载时忽略的键
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    # 不需要分割的模块
    _no_split_modules = [r"BartEncoderLayer", r"BartDecoderLayer"]
    # 跳过键的设备放置
    _skip_keys_device_placement = "past_key_values"
    # 是否支持闪存注意力2
    _supports_flash_attn_2 = True
    # 是否支持 SDPA
    _supports_sdpa = True
    # 初始化模型参数的权重
    def _init_weights(self, module):
        # 获取初始化标准差
        std = self.config.init_std
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 返回虚拟输入，用于模型推理
    @property
    def dummy_inputs(self):
        # 获取填充标记
        pad_token = self.config.pad_token_id
        # 创建虚拟输入张量
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        # 返回虚拟输入字典
        return dummy_inputs
class PretrainedBartModel(BartPreTrainedModel):
    def __init_subclass__(self):
        # 发出警告，表明`PretrainedBartModel`类已被弃用，请使用`BartPreTrainedModel`代替
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


class BartPretrainedModel(BartPreTrainedModel):
    def __init_subclass__(self):
        # 发出警告，表明`PretrainedBartModel`类已被弃用，请使用`BartPreTrainedModel`代替
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPreTrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BartConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, BartForConditionalGeneration

    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    >>> ARTICLE_TO_SUMMARIZE = (
    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    ... )
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
    ```

    Mask filling example:

    ```python
    >>> from transformers import AutoTokenizer, BartForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    >>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    >>> TXT = "My friends are <mask> but they eat too many carbs."
    >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    >>> logits = model(input_ids).logits
    # 找到输入标识符中被掩盖标记的索引，此处使用了 PyTorch 张量的方法
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # 对模型输出的对应位置的逻辑回归值进行 softmax 处理，得到概率分布
    probs = logits[0, masked_index].softmax(dim=0)
    # 取得概率分布中最大的五个值及其对应的索引，即对应的预测结果
    values, predictions = probs.topk(5)
    
    # 将预测结果使用分词器转换为对应的文本单词
    tokenizer.decode(predictions).split()
    # 返回预测结果的文本形式的列表
    ['not', 'good', 'healthy', 'great', 'very']
"""

BART_INPUTS_DOCSTRING = r"""
"""


class BartEncoder(BartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout  # 使用配置中的 dropout 概率
        self.layerdrop = config.encoder_layerdrop  # 使用配置中的 encoder layer dropout 概率

        embed_dim = config.d_model  # 嵌入维度取自配置
        self.padding_idx = config.pad_token_id  # 填充标记的索引取自配置
        self.max_source_positions = config.max_position_embeddings  # 最大源位置取自配置
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 嵌入尺度为嵌入维度的平方根（如果配置为 True）

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)  # 嵌入标记的嵌入层

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight  # 如果提供了嵌入标记，则使用提供的权重

        self.embed_positions = BartLearnedPositionalEmbedding(  # 学习的位置嵌入
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])  # 编码器层的模块列表，每层为 BartEncoderLayer
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"  # 是否使用 Flash Attention 2 实现
        self._use_sdpa = config._attn_implementation == "sdpa"  # 是否使用 SDPA 实现
        self.layernorm_embedding = nn.LayerNorm(embed_dim)  # 嵌入的 LayerNorm 层

        self.gradient_checkpointing = False  # 梯度检查点（用于动态图计算）默认关闭
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens  # 获取输入嵌入层

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入嵌入层

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class BartDecoder(BartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    # 初始化方法，接受一个BartConfig对象和一个可选的嵌入令牌（nn.Embedding类型）参数
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)
        # 从配置中获取一些参数并赋值给相应的属性
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 创建嵌入令牌（Embedding）层，其大小为（vocab_size, d_model），并指定填充索引
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 如果传入了embed_tokens参数，则使用传入的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 创建BartLearnedPositionalEmbedding对象，用于位置编码
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # 创建一组BartDecoderLayer对象，数量为decoder_layers
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 根据配置确定是否使用FlashAttentionV2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # 根据配置确定是否使用SDPA（Self-Distillation Pretraining Attention）
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 对嵌入向量进行LayerNorm
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 梯度检查点，用于在训练过程中减少内存占用
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播方法
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
# 导入必要的库和模块
@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",  # 添加模型文档字符串
    BART_START_DOCSTRING,  # 添加 BART 模型的文档字符串
)
class BartModel(BartPreTrainedModel):  # 定义 BartModel 类，继承自 BartPreTrainedModel 类
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]  # 定义共享权重的键列表

    def __init__(self, config: BartConfig):  # 初始化函数，接受 BartConfig 对象作为参数
        super().__init__(config)  # 调用父类的初始化函数

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size  # 获取填充索引和词汇表大小
        # 创建共享的嵌入层，用于输入和输出的嵌入
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器对象
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    def _tie_weights(self):  # 函数用于绑定权重
        if self.config.tie_word_embeddings:  # 如果需要绑定词嵌入权重
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)  # 绑定或克隆编码器的嵌入权重
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)  # 绑定或克隆解码器的嵌入权重

    def get_input_embeddings(self):  # 获取输入嵌入
        return self.shared

    def set_input_embeddings(self, value):  # 设置输入嵌入
        self.shared = value  # 设置共享嵌入为给定值
        self.encoder.embed_tokens = self.shared  # 设置编码器的嵌入层为共享嵌入
        self.decoder.embed_tokens = self.shared  # 设置解码器的嵌入层为共享嵌入

    def get_encoder(self):  # 获取编码器对象
        return self.encoder

    def get_decoder(self):  # 获取解码器对象
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)  # 添加模型前向传播的文档字符串
    @add_code_sample_docstrings(  # 添加代码示例的文档字符串
        checkpoint=_CHECKPOINT_FOR_DOC,  # 添加检查点路径
        output_type=Seq2SeqModelOutput,  # 添加输出类型
        config_class=_CONFIG_FOR_DOC,  # 添加配置类
        expected_output=_EXPECTED_OUTPUT_SHAPE,  # 添加预期输出形状
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入 token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入 token ID
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器注意力遮罩
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器头部遮罩
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部遮罩
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器输出
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.",  # 添加模型文档字符串
    BART_START_DOCSTRING  # 添加 BART 模型的文档字符串
)
class BartForConditionalGeneration(BartPreTrainedModel):  # 定义 BartForConditionalGeneration 类，继承自 BartPreTrainedModel 类
    base_model_prefix = "model"  # 基础模型前缀
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]  # 定义共享权重的键列表
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]  # 在加载时忽略的键列表
    def __init__(self, config: BartConfig):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 创建一个 BART 模型对象
        self.model = BartModel(config)
        # 初始化一个与模型共享的全局注意力头的偏置
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 初始化线性层，用于将模型输出映射到词汇表的维度
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_encoder(self):
        # 获取 BART 模型的编码器
        return self.model.get_encoder()

    def get_decoder(self):
        # 获取 BART 模型的解码器
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调整模型的词嵌入矩阵大小，返回新的词嵌入对象
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整最终输出偏置向量的大小以匹配新的词嵌入大小
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取原始词汇表大小
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的词汇表大小小于等于原始词汇表大小
        if new_num_tokens <= old_num_tokens:
            # 则截取偏置向量，保留前 new_num_tokens 个元素
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 否则，在偏置向量末尾添加额外的零向量，以扩展其大小
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 更新最终输出偏置向量
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        # 获取语言模型头部的线性层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的线性层
        self.lm_head = new_embeddings

    # 覆盖父类的 forward 方法，添加文档字符串和返回值注释
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        # 设置返回字典，如果未提供则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            # 如果提供了标签，则将 use_cache 设置为 False，并发出警告
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # 如果未提供解码器输入，根据标签创建解码器输入
            if decoder_input_ids is None and decoder_inputs_embeds is None:
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

        # 获取语言模型的输出
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            # 将标签转移到与 lm_logits 相同的设备上，并计算损失
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不返回字典，则返回 lm_logits 和其他输出
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
    # 为生成准备输入的函数，对应于生成器模型的一个方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果过去的键值已经存在，则截断decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果decoder_input_ids的长度大于past_length
            if decoder_input_ids.shape[1] > past_length:
                # 移除前缀长度为past_length
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 截断decoder_input_ids
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的输入的字典
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此项以避免缓存（可能是为了调试）
        }

    # 从标签中准备解码器输入ID的函数
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签向右移动一位，用于解码器的输入
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 重新排序缓存的静态方法
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序过去的键值
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的跨注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                # 对每个层的过去状态按beam_idx重新排序
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                # 保持其他不变
                + layer_past[2:],
            )
        return reordered_past
# 使用装饰器添加文档字符串，描述带有顶部序列分类/头部的 Bart 模型，例如用于 GLUE 任务
@add_start_docstrings(
    """
    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    BART_START_DOCSTRING,
)
class BartForSequenceClassification(BartPreTrainedModel):
    # 在这里定义了一些被绑定权重的键名，这些键将在模型参数共享时使用
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        # 初始化 BartModel
        self.model = BartModel(config)
        # 初始化分类头
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加文档字符串，描述模型的前向传播过程，输入和输出
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
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



# 使用装饰器添加文档字符串，描述带有顶部抽取式问答任务的 BART 模型
@add_start_docstrings(
    """
    BART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BART_START_DOCSTRING,
)
class BartForQuestionAnswering(BartPreTrainedModel):
    # 在这里定义了一些被绑定权重的键名，这些键将在模型参数共享时使用
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 将分类数目设为 2，用于问答任务中的二分类
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 初始化 BartModel
        self.model = BartModel(config)
        # 初始化问答任务输出层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加文档字符串，描述模型的前向传播过程，输入和输出
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串，用于自动生成文档
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,  # QA 检查点
        output_type=Seq2SeqQuestionAnsweringModelOutput,  # 输出类型为 Seq2SeqQuestionAnsweringModelOutput
        config_class=_CONFIG_FOR_DOC,  # 配置类
        expected_loss=_QA_EXPECTED_LOSS,  # 预期损失
        expected_output=_QA_EXPECTED_OUTPUT,  # 预期输出
    )
    # 前向传播函数，接收多个输入参数
    def forward(
        self,
        input_ids: torch.Tensor = None,  # 输入的 token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入 token IDs
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器头部掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器输出
        start_positions: Optional[torch.LongTensor] = None,  # 起始位置
        end_positions: Optional[torch.LongTensor] = None,  # 结束位置
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
class BartDecoderWrapper(BartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        # 创建一个 BartDecoder 实例作为该类的属性
        self.decoder = BartDecoder(config)

    def forward(self, *args, **kwargs):
        # 调用 BartDecoder 的 forward 方法
        return self.decoder(*args, **kwargs)


@add_start_docstrings(
    """
    BART decoder with with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    BART_START_DOCSTRING,
)
class BartForCausalLM(BartPreTrainedModel):
    # 定义一个类变量，表示权重被绑定的键名
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        # 设置配置参数为解码器模式，并关闭编码器-解码器模式
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 创建 BartDecoderWrapper 实例
        self.model = BartDecoderWrapper(config)

        # 创建线性层用于语言模型的头部，权重与输入嵌入层绑定
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入层
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 获取输出嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入层
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 获取解码器
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
    ):
        """
        Forward pass for the BartForCausalLM model.

        Args:
            input_ids: Optionally, a torch.LongTensor of shape [batch_size, sequence_length] containing the input
                token indices. During generation, the model operates in auto-regressive mode and generates one token
                at a time. More details in :obj:`BartModel.prepare_inputs_for_generation`.
            attention_mask: Optionally, a torch.Tensor of shape [batch_size, sequence_length] with indices selected
                in [0, 1], where 1 indicates tokens that are **not masked**, 0 indicates **masked** tokens. More
                details in :obj:`BartModel.prepare_inputs_for_generation`.
            encoder_hidden_states: Optionally, a torch.FloatTensor of shape [batch_size, sequence_length, hidden_size]
                containing the hidden states of the encoder. Used in the cross-attention layer.
            encoder_attention_mask: Optionally, a torch.FloatTensor of shape [batch_size, sequence_length] with indices
                selected in [0, 1], where 1 indicates **valid** positions in the attention layers and 0 indicates
                **masked** positions. More details in :obj:`BartModel.prepare_inputs_for_generation`.
            head_mask: Optionally, a torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices
                selected in [0, 1], where 1 indicates the head is **not masked**, 0 indicates the head is **masked**.
            cross_attn_head_mask: Optionally, a torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices
                selected in [0, 1], where 1 indicates the head is **not masked**, 0 indicates the head is **masked**.
            past_key_values: Optionally, a list of torch.FloatTensor containing cached past key values of the
                self-attention layers. Can be used to speed up decoding.
            inputs_embeds: Optionally, a torch.FloatTensor of shape [batch_size, sequence_length, hidden_size]
                containing the embedded input.
            labels: Optionally, a torch.LongTensor of shape [batch_size, sequence_length] containing the labels.
                If provided, the model will calculate the loss and return it.
            use_cache: bool, optional, default to `None`. If `True`, past key values are used to speed up decoding.
            output_attentions: bool, optional, defaults to `None`. Whether or not to return the attentions tensors of
                all attention layers. See :obj:`BartModel` for more information.
            output_hidden_states: bool, optional, defaults to `None`. Whether or not to return the hidden states of
                all layers. See :obj:`BartModel` for more information.
            return_dict: bool, optional, defaults to `None`. Whether or not to return a :class:`~transformers.file_utils.ModelOutput`
                instead of a plain tuple. See :obj:`BartModel` for more information.

        Returns:
            :class:`~transformers.modeling_outputs.CausalLMOutputWithCrossAttentions` or :class:`torch.Tensor` or
            :class:`~transformers.file_utils.ModelOutput`: A dictionary of outputs containing the generated tokens and
            the attentions if `output_attentions=True` is passed or the `config.return_dict=True`. If `labels` is
            not `None` the dictionary also contains the loss. Otherwise, a tensor of shape `(batch_size, sequence_length, config.vocab_size)`.
        """
        # 调用 BartDecoderWrapper 的 forward 方法
        return self.model(input_ids, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states,
                          encoder_attention_mask=encoder_attention_mask, head_mask=head_mask,
                          cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values,
                          inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                          output_attentions=output_attentions, output_hidden_states=output_hidden
        # 如果模型用作编码器-解码器模型中的解码器，则动态创建解码器注意力掩码
        if attention_mask is None:
            # 如果注意力掩码为空，则创建一个全为1的张量，形状与输入ID相同
            attention_mask = input_ids.new_ones(input_ids.shape)

        # 如果有过去的键值，则进行以下处理
        if past_key_values:
            # 获取过去的键值中每层的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经仅传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                # 如果输入ID的长度大于过去的长度，则去除前缀长度为过去的长度
                remove_prefix_length = past_length
            else:
                # 否则，默认为旧的行为：仅保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 去除前缀长度，保留后缀以供下一步使用
            input_ids = input_ids[:, remove_prefix_length:]

        # 在第一步中，解码器缓存状态为空
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 对于过去的键值中的每一层，按照beam_idx重新排序
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                # 将每一层的过去状态按照beam_idx重新排序，并添加到重新排序后的过去状态中
            )
        return reordered_past
```