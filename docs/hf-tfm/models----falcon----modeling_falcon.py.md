# `.\models\falcon\modeling_falcon.py`

```py
# 设置文件编码为UTF-8
# 版权声明
# 许可证声明

"""PyTorch Falcon model."""

# 引入需要的库
import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

# 引入自定义的工具类和模型输出
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_2_0
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from .configuration_falcon import FalconConfig

# 检查是否在类型检查模式下
if TYPE_CHECKING:
    from ...configuration_utils import PretrainedConfig

# 如果支持Flash attention 2，则进行相关引入
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取日志记录器
logger = logging.get_logger(__name__)

# Falcon预训练模型的存档列表
FALCON_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tiiuae/falcon-40b",
    "tiiuae/falcon-40b-instruct",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-rw-7b",
    "tiiuae/falcon-rw-1b",
]
_CHECKPOINT_FOR_DOC = "Rocketknight1/falcon-rw-1b"
_CONFIG_FOR_DOC = "FalconConfig"


# 注意：不幸的是，在训练过程中我们没有融合矩阵乘和偏置，这意味着在操作之间有一个额外的bfloat16量化。
# 为了不降低我们HF-port的质量，我们在最终模型中保留了这些特性。
class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 执行矩阵乘法
        hidden_states = input @ self.weight.T
        if self.bias is None:
            return hidden_states
        # 添加偏置
        return hidden_states + self.bias


# 从transformers.models.llama.modeling_llama.rotate_half复制过来
def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    # 使用切片操作获取 x 的后一半数据（最后一个维度），":" 表示从头至尾
    x2 = x[..., x.shape[-1] // 2 :]
    # 使用 torch.cat() 沿指定维度拼接张量，将 -x2 和 x1 拼接在一起，dim=-1 表示在最后一个维度上进行拼接
    return torch.cat((-x2, x1), dim=-1)
# 从transformers.models.llama.modeling_llama.apply_rotary_pos_emb中复制的函数，用于应用Rotary Position Embedding到查询和键张量中
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Args:
        q (`torch.Tensor`): 查询张量
        k (`torch.Tensor`): 键张量
        cos (`torch.Tensor`): Rotary嵌入的余弦部分
        sin (`torch.Tensor`): Rotary嵌入的正弦部分
        position_ids (`torch.Tensor`): 对应于查询和键张量的令牌位置索引。例如，当使用KV-cache时，可以用它传递偏移的位置id
        unsqueeze_dim (`int`, *可选*, 默认为1): 指定沿哪个维度展开cos[position_ids]和sin[position_ids]，以便它们可以正确广播到q和k的维度。
    Returns:
        `tuple(torch.Tensor)`: 旋转使用Rotary Position Embedding后的查询和键张量
    """

    # 在position_ids处对cos张量进行展开操作
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # 在position_ids处对sin张量进行展开操作
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 计算并返回旋转后的查询张量
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 计算并返回旋转后的键张量
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# 从transformers.models.llama.modeling_llama._get_unpad_data中复制的函数，用于获取未填充的数据
def _get_unpad_data(attention_mask):
    # 计算批次中各序列的长度之和
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 获取attention_mask中值不为零的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中最大的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积的序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 从transformers.models.llama.modeling_llama.LlamaRotaryEmbedding中复制的类，将Llama替换为Falcon
class FalconRotaryEmbedding(nn.Module):
    # 初始化函数，设置注意力机制需要的参数
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类初始化函数
        super().__init__()
    
        # 初始化对象的维度、最大位置编码长度和基数
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率的倒数，用于位置编码
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 将频率的倒数注册为缓冲，不参与反向传播
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
        # 生成余弦和正弦值的缓存，为了让 torch.jit.trace 工作
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    
    # 设置余弦和正弦缓存
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 记录当前缓存的序列长度
        self.max_seq_len_cached = seq_len
        # 在指定设备上生成序列 t
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
    
        # 计算频率矩阵
        freqs = torch.outer(t, self.inv_freq)
        # 拼接频率值，生成位置编码矩阵
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将余弦缓存注册为缓冲，不参与反向传播
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 将正弦缓存注册为缓冲，不参与反向传播
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    # 前向传播函数
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果指定的序列长度大于当前缓存的最大序列长度，则重新设置余弦和正弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    
        # 返回指定长度的余弦和正弦缓存片段
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding复制并将Llama->Falcon
class FalconLinearScalingRotaryEmbedding(FalconRotaryEmbedding):
    """使用线性缩放扩展的FalconRotaryEmbedding。由Reddit用户/u/kaiokendev提供"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        调用父类的构造函数并传入参数(dim, max_position_embeddings, base, device)
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        使用torch.arange创建张量t，长度为max_seq_len_cached
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # 与论文中不同，但它使用不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        将emb的余弦和正弦值注册为缓存
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding复制并将Llama->Falcon
class FalconDynamicNTKScalingRotaryEmbedding(FalconRotaryEmbedding):
    """使用动态NTK缩放扩展的FalconRotaryEmbedding。由Reddit用户/u/bloc97和/u/emozilla提供"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        调用父类的构造函数并传入参数(dim, max_position_embeddings, base, device)
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            计算base的新值
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            计算inv_freq的新值
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            将inv_freq注册为缓存
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        使用torch.arange创建张量t，长度为max_seq_len_cached
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 与论文中不同，但它使用不同的排列方式以获得相同的计算��果
        emb = torch.cat((freqs, freqs), dim=-1)
        将emb的余弦和正弦值注册为缓存
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def _prepare_4d_attention_mask(mask: torch.Tensor, past_key_values_length: int) -> torch.BoolTensor:
    """
    将注意力掩码从`[batch_size, seq_length]`扩展为`[batch_size, 1, seq_length, seq_length + past_length]`。
    """
    获取mask的形状信息
    batch_size, total_length = mask.shape
    根据past_key_values_length是否为None，计算seq_length的值
    seq_length = total_length - past_key_values_length if past_key_values_length is not None else total_length
    # 创建一个扩展的掩码矩阵，用于屏蔽指定位置的元素
    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    # 将扩展的掩码矩阵在指定维度上进行扩展，以匹配给定的形状
    return expanded_mask.expand(batch_size, 1, seq_length, total_length)
# 建立一个alibi张量，用于注意力偏置
def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    # 获取注意力掩码张量的批量大小和序列长度
    batch_size, seq_length = attention_mask.shape
    # 找到最接近的2的幂
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    # 创建基础张量
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    # 创建幂张量
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    # 计算斜率张量
    slopes = torch.pow(base, powers)

    # 如果最接近的2的幂不等于头的数量
    if closest_power_of_2 != num_heads:
        # 创建额外的基础张量
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        # 计算剩余头的数量
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        # 创建额外的幂张量
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        # 将额外的斜率连接到原斜率张量上
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # 注意：alibi将被添加到将应用于查询键产品的注意力偏置，因此alibi的形状必须是(batch_size, num_heads, query_length, key_length)
    # 在这里我们设定(batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # 然后将正确广播查询长度维度
    # 这与T5的相对位置偏差几乎完全相同
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    # 计算alibi张量
    alibi = slopes[..., None].bfloat16() * arange_tensor
    # 重塑alibi张量并转换为指定数据类型
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


# 从transformers.models.bloom.modeling_bloom.dropout_add中复制的函数
def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    # 对输入张量进行dropout操作
    out = F.dropout(x, p=prob, training=training)
    # 将残差张量添加到dropout后的结果中
    out = residual + out
    return out


class FalconAttention(nn.Module):
    # 初始化方法，接受一个 FalconConfig 类型的参数作为配置
    def __init__(self, config: FalconConfig):
        # 调用父类的初始化方法
        super().__init__()

        # 将配置信息保存在对象中
        self.config = config
        # 从配置信息中获取隐藏层大小
        self.hidden_size = config.hidden_size
        # 从配置信息中获取注意力头的数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_heads
        # 设置分割大小为隐藏层大小
        self.split_size = self.hidden_size
        # 从配置信息中获取隐藏层的丢弃率
        self.hidden_dropout = config.hidden_dropout
        # 从配置信息中获取位置嵌入的最大长度
        self.max_position_embeddings = config.max_position_embeddings
        # 从配置信息中获取绳索的角度
        self.rope_theta = config.rope_theta
        # 设置是否是因果关系
        self.is_causal = True
        # 根据配置信息判断是否使用 SDPA（Structured Dot-Product Attention）
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 如果每个注意力头的维度乘以注意力头的数量不等于隐藏层大小，则抛出数值错误
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 如果配置信息中设置了旋转参数，调用初始化绳索的方法
        if config.rotary:
            self._init_rope()

        # 计算注意力的缩放因子
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        # 设置注意力的缩放参数 beta
        self.beta = self.inv_norm_factor

        # 根据配置信息确定查询-键-值的输出维度
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        
        # 初始化查询-键-值层，将隐藏层映射到 qkv_out_dim 维度
        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        # 保存配置信息中的新解码器架构标志和多查询标志
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        # 初始化全连接层，将隐藏层映射到隐藏层
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        # 初始化注意力丢弃层
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        # 如果使用新解码器架构或者不使用多查询，则设置键-值头的数量为配置信息中的值，否则为 1
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1

    # 从 transformers.models.llama.modeling_llama.LlamaAttention._init_rope 复制过来的注释，这里将 Llama 替换为 Falcon
    # 初始化 RoPE（Rotary Position Embedding），用于处理位置编码
    def _init_rope(self):
        # 如果没有指定 RoPE 的缩放参数
        if self.config.rope_scaling is None:
            # 使用默认的 RoPE 类（FalconRotaryEmbedding）进行初始化
            self.rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 否则，获取 RoPE 的缩放类型和因子
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            # 根据不同的缩放类型选择不同的 RoPE 类
            if scaling_type == "linear":
                # 线性缩放类型的 RoPE 类
                self.rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                # 动态缩放类型的 RoPE 类
                self.rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 抛出错误，表示未知的 RoPE 缩放类型
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 将融合的查询、键、值张量分割成各自的部分
    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        # 如果使用了新的解码器架构
        if self.new_decoder_architecture:
            # 获取输入张量的形状信息
            batch, seq_len, _ = fused_qkv.shape
            # 将融合的查询、键、值张量重塑成新的形状
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)

            # 将各部分张量扁平化并返回
            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        # 如果不使用新的解码器架构且不是多查询模式
        elif not self.multi_query:
            # 获取输入张量的形状信息
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            # 将融合的查询、键、值张量重塑成新的形状
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        # 如果使用了多查询模式
        else:
            # 获取输入张量的形状信息
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            # 将融合的查询、键、值张量重塑成新的形状
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    # 从 transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads 复制而来
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        # 获取输入张量的形状信息：batch_size * num_heads, seq_length, head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        # 计算得到 batch_size
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # 将输入张量重新构造为：batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        # 转置操作，调整张量的维度顺序
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        # 将张量重新构造为所需形状
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
class FalconFlashAttention2(FalconAttention):
    """
    Falcon flash attention module. This module inherits from `FalconAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 设置一个属性，用于检测 Flash Attention 是否使用顶部左侧对齐的掩码
        # 如果 Flash Attention 版本小于 2.1，则使用顶部左侧对齐的掩码
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    # 结束函数定义
    ):
        """
        调用 Flash Attention 的前向方法 - 如果输入的隐藏状态包含至少一个填充标记，则先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

        Args:
            query_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入查询状态
            key_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入键状态
            value_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入值状态
            attention_mask (`torch.Tensor`):
                填充遮罩 - 对应大小为`(batch_size, seq_len)`的张量，其中 0 表示填充标记的位置，1 表示非填充标记的位置。
            dropout (`int`, *optional*):
                注意力丢弃率
            softmax_scale (`float`, *optional*):
                在应用 softmax 前的 QK^T 缩放。默认为 1 / sqrt(head_dim)
        """
        # 如果 Flash Attention 不使用左上角的遮罩，则使用自回归模式(is_causal)
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 删除 `query_length != 1` 检查，一旦 Flash Attention for RoCm 升级到 2.1 版本，详细信息，请参见 LlamaFlashAttention2 __init__ 中的注释。
            causal = self.is_causal and query_length != 1

        # 包含至少一个填充标记在序列中
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # 取消填充输入
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # 计算注意力输出（未填充）
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

            # 填充最终的注意力输出
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # 计算注意力输出
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # 返回注意力输出
        return attn_output

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制
    # 定义一个私有方法，用于处理输入数据，包括查询层、键层、值层、注意力掩码和查询长度
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取批次大小、键值序列长度、键值头数和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据索引重新组织键层数据
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 根据索引重新组织值层数据
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据查询长度情况处理查询层数据
        if query_length == kv_seq_len:
            # 根据索引重新组织查询层数据
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 处理特殊情况，查询长度为1
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个内存复制，这非常糟糕。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 处理一般情况，根据查询长度计算未填充的输入
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的输入数据
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义一个名为FalconMLP的类，继承自nn.Module类
class FalconMLP(nn.Module):
    # 初始化方法，接受一个FalconConfig类型的config参数
    def __init__(self, config: FalconConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 从config参数中获取hidden_size值
        hidden_size = config.hidden_size

        # 创建一个全连接层，输入维度为hidden_size，输出维度为4 * hidden_size
        self.dense_h_to_4h = FalconLinear(hidden_size, 4 * hidden_size, bias=config.bias)
        # 创建一个GELU激活函数实例
        self.act = nn.GELU()
        # 创建一个全连接层，输入维度为4 * hidden_size，输出维度为hidden_size
        self.dense_4h_to_h = FalconLinear(4 * hidden_size, hidden_size, bias=config.bias)
        # 从config参数中获取hidden_dropout值
        self.hidden_dropout = config.hidden_dropout

    # 前向传播方法，接受一个torch.Tensor类型的x参数，返回一个torch.Tensor类型的值
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入x应用GELU激活函数和全连接层dense_h_to_4h
        x = self.act(self.dense_h_to_4h(x))
        # 对上一步的结果应用全连接层dense_4h_to_h
        x = self.dense_4h_to_h(x)
        # 返回最终结果
        return x

# 定义一个名为FalconDecoderLayer的类，继承自nn.Module类
class FalconDecoderLayer(nn.Module):
    # 初始化方法，接受一个FalconConfig类型的config参数
    def __init__(self, config: FalconConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 从config参数中获取hidden_size值
        hidden_size = config.hidden_size
        # 从config参数中获取num_attention_heads值
        self.num_heads = config.num_attention_heads

        # 根据config的_attn_implementation字段选择不同的FalconAttention类，并使用config初始化它
        self.self_attention = FALCON_ATTENTION_CLASSES[config._attn_implementation](config)
        # 创建一个FalconMLP实例，使用config初始化它
        self.mlp = FalconMLP(config)
        # 从config参数中获取hidden_dropout值
        self.hidden_dropout = config.hidden_dropout
        # 保存config参数
        self.config = config

        # 如果config的new_decoder_architecture字段为True
        if config.new_decoder_architecture:
            # 创建一个LN层，设置输入维度为hidden_size
            self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            # 创建一个LN层，设置输入维度为hidden_size
            self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 如果config的new_decoder_architecture字段为False
        else:
            # 创建一个LN层，设置输入维度为hidden_size
            self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            # 如果config的parallel_attn字段为False
            if not config.parallel_attn:
                # 创建一个LN层，设置输入维度为hidden_size
                self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    # 前向传播方法，接受多个参数，返回一个torch.Tensor类型的值
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
        ):
            # 如果在参数中传入了"padding_mask"，发出警告，提示该参数在 v4.37 版本将被移除，建议使用"attention_mask"代替
            if "padding_mask" in kwargs:
                warnings.warn(
                    "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
                )

            # 保存原始输入作为残差连接的一部分
            residual = hidden_states

            # 如果使用了新的解码器架构，对注意力和 MLP 层进行独立的 LayerNormalization
            if self.config.new_decoder_architecture:
                attention_layernorm_out = self.ln_attn(hidden_states)
                mlp_layernorm_out = self.ln_mlp(hidden_states)
            else:
                # 否则只对输入进行 LayerNormalization
                attention_layernorm_out = self.input_layernorm(hidden_states)

            # 自注意力层
            attn_outputs = self.self_attention(
                attention_layernorm_out,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                alibi=alibi,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            # 获取自注意力层的输出
            attention_output = attn_outputs[0]

            # 如果没有使用新的解码器架构，进行残差连接和 dropout
            if not self.config.new_decoder_architecture:
                if self.config.parallel_attn:
                    # 如果并行处理注意力，直接将注意力输出作为 MLP 层的输入
                    mlp_layernorm_out = attention_layernorm_out
                else:
                    # 否则，进行残差连接和 dropout，并对输出进行 LayerNormalization
                    residual = dropout_add(
                        attention_output, residual, self.config.attention_dropout, training=self.training
                    )
                    mlp_layernorm_out = self.post_attention_layernorm(residual)

            # 获取注意力层的输出
            outputs = attn_outputs[1:]

            # MLP 层
            mlp_output = self.mlp(mlp_layernorm_out)

            # 如果使用了新的解码器架构或并行处理注意力，将自注意力层的输出加到 MLP 层的输出上
            if self.config.new_decoder_architecture or self.config.parallel_attn:
                mlp_output += attention_output

            # 将 MLP 层的输出和残差连接的原始输入进行 dropout 和相加
            output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

            # 如果使用缓存机制，将输出中添加中间结果，否则剔除第一个元素，因为它是隐藏状态
            if use_cache:
                outputs = (output,) + outputs
            else:
                outputs = (output,) + outputs[1:]

            # 返回输出，包括隐藏状态、缓存和注意力分布
            return outputs  # hidden_states, present, attentions
# Falcon 模型的文档字符串，包括模型的继承关系和使用说明
FALCON_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FalconConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Falcon 模型的输入文档字符串
FALCON_INPUTS_DOCSTRING = r"""
"""

# Falcon 预训练模型的类，处理权重初始化、预训练模型的下载和加载等
class FalconPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Falcon 模型配置类
    config_class = FalconConfig
    # 基础模型前缀
    base_model_prefix = "transformer"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 需要单独初始化而不拆分的模块
    _no_split_modules = ["FalconDecoderLayer"]
    # 是否支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 是否支持 SDPA（Sparse Dynamic Parameter Allocation）
    _supports_sdpa = True

    # 初始化方法，调用父类的初始化方法
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    # 初始化模型的权重
    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear) or isinstance(module, FalconLinear):
            # 对线性层的权重进行初始化
            # 与 TF 版本稍有不同，使用正态分布来初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            # 对 LayerNorm 层的权重初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # 从 transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa 调用的方法
    @classmethod
        # 检查并启用SDPA（使用BetterTransformer或PyTorch版本大于等于2.0时）
        def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> "PretrainedConfig":
            # 注意：Falcon从PyTorch 2.0开始支持SDPA，保持向后兼容性（自动在torch>=2.0时使用SDPA）。
            # 如果只进行硬检查
            if hard_check_only:
                # 如果PyTorch版本小于2.0，则抛出ImportError异常
                if not is_torch_greater_or_equal_than_2_0:
                    raise ImportError("PyTorch SDPA requirements in Transformers are not met. Please install torch>=2.0.")

            # 如果PyTorch版本小于2.0，则返回配置
            if not is_torch_greater_or_equal_than_2_0:
                return config

            # 获取类属性use_bettertransformer，如果是True，则返回配置
            _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
            if _is_bettertransformer:
                return config

            # 如果不是硬检查，则设置配置的注意力实现为"sdpa"
            if not hard_check_only:
                config._attn_implementation = "sdpa"
            # 返回配置
            return config
# 添加开始文档字符串作为类的注释，描述了该类的作用
@add_start_docstrings(
    "The bare Falcon Model transformer outputting raw hidden-states without any specific head on top.",
    FALCON_START_DOCSTRING,
)
# 定义 FalconModel 类，继承自 FalconPreTrainedModel
class FalconModel(FalconPreTrainedModel):
    # 初始化方法，接受一个 FalconConfig 类型的参数
    def __init__(self, config: FalconConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化配置参数
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi

        # 创建词嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # 创建一系列 Transformer blocks
        self.h = nn.ModuleList([FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 创建最终的层归一化层
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入的嵌入层
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置新的输入嵌入层
    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    # 前向传播方法，接受多个参数，包括输入的 id 等
    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加开始文档字符串作为类的注释，描述了该类的作用
@add_start_docstrings(
    "The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).",
    FALCON_START_DOCSTRING,
)
# FalconForCausalLM 继承自 FalconPreTrainedModel
class FalconForCausalLM(FalconPreTrainedModel):
    # tied_weights_keys 是类属性，指定了被绑定权重的键名
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受一个 FalconConfig 类型的参数
    def __init__(self, config: FalconConfig):
        # 调用父类初始化方法
        super().__init__(config)
        # 创建 FalconModel 实例
        self.transformer = FalconModel(config)
        # 创建一个线性层作为语言建模的头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入层
    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings
    # 为生成准备输入
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # 如果过去的键值不为空
        if past_key_values is not None:
            # 计算过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能已经只传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:
                # 要删除的前缀长度为过去键值的长度
                remove_prefix_length = past_length
            else:
                # 默认行为：仅保留最终 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 从输入 ID 中删除前缀
            input_ids = input_ids[:, remove_prefix_length:]

        # 注意：带有 alibi 的 Falcon 版本不使用 position_ids。它与 RoPE 一起使用。
        if not self.transformer.use_alibi and attention_mask is not None and position_ids is None:
            # 为批量生成动态创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                # 如果有过去键值，将 position_ids 裁剪到相应的长度
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 返回准备好的输入字典
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    # 前向传播方法
    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```  
    def forward(
        self, input_ids, attention_mask=None, past_key_values=None, position_ids=None, head_mask=None,
        inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size]`. All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`.
        """

        # 确定是否需要返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer网络进行前向传播，获取transformer_outputs
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # 通过lm_head获取语言模型的logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # 将logits和labels进行格式转换，以便计算损失
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            # 如果不返回字典形式的结果，则返回元组形式的输出结果
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有额外交叉注意力的输出结果
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        # Re-order the `past_key_values` cache based on `beam_idx` to match with the correct beam_idx at every generation step
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past
        )
        # Return the re-ordered `past_key_values` cache
        return reordered_past
# 在上方加入文档字符串，描述 Falcon 模型，带有一个线性层的序列分类头部
# 该模型使用最后一个令牌进行分类，需要知道最后一个令牌的位置
# 如果配置中定义了填充标记（pad_token_id），则找到每行中不是填充标记的最后一个令牌
# 如果没有定义填充标记，则简单地取每行中的最后一个值
# 对于使用 inputs_embeds 而不是 input_ids 的情况，也将采取同样的方法
@add_start_docstrings(
    """
    The Falcon Model transformer with a sequence classification head on top (linear layer).

    [`FalconForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    FALCON_START_DOCSTRING,
)

# 定义 FalconForSequenceClassification 类
class FalconForSequenceClassification(FalconPreTrainedModel):
    # 初始化方法
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = FalconModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 在 forward 方法上方添加文档字符串，包含关于输入和代码示例的说明
    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 forward 方法，包含多种参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 在上方加入文档字符串，描述 Falcon 模型，带有一个线性层的标记分类头部
# 用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    Falcon Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FALCON_START_DOCSTRING,
)

# 定义 FalconForTokenClassification 类
class FalconForTokenClassification(FalconPreTrainedModel):
    # 初始化方法
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = FalconModel(config)
        # 根据配置中的参数初始化分类器的 dropout
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()
    # 将模型的输入相关文档字符串添加到模型的前向方法
    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    # 添加代码样本相关的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型前向方法，接受输入参数并返回输出结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token IDs
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,  # 用于生成过去的key-value对
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入输入
        labels: Optional[torch.Tensor] = None,  # 用于计算分类/回归损失的标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回结果字典
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:  # 返回类型提示

        # 如果未提供return_dict，则根据配置确定是否返回结果字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer处理输入并返回transformer的输出
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取transformer输出中的隐藏状态
        hidden_states = transformer_outputs[0]
        # 对隐藏状态应用dropout
        hidden_states = self.dropout(hidden_states)
        # 将隐藏状态输入分类器得到logits
        logits = self.classifier(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 获取标签的形状
            batch_size, seq_length = labels.shape
            # 初始化交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        # 如果不需要返回结果字典
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回Token分类器的输出对象
        return TokenClassifierOutput(
            loss=loss,  # 返回损失
            logits=logits,  # 返回logits
            hidden_states=transformer_outputs.hidden_states,  # 返回隐藏状态
            attentions=transformer_outputs.attentions,  # 返回注意力权重
        )
# 使用 add_start_docstrings 函数添加模型文档字符串，描述模型及其用途
"""
The Falcon Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 继承自 FalconPreTrainedModel 的 FalconForQuestionAnswering 类
class FalconForQuestionAnswering(FalconPreTrainedModel):
    # 初始化函数，接受配置参数并调用父类的初始化函数
    def __init__(self, config):
        super().__init__(config)
        # 使用配置创建 FalconModel 对象
        self.transformer = FalconModel(config)
        # 创建线性层，用于计算 span start logits 和 span end logits
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 函数添加模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    # 模型前向传播方法，接受多个输入参数并返回模型输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(self,
            # 定义方法 forward，接受输入参数以及返回输出结果
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, QuestionAnsweringModelOutput]:
            # 定义 forward 方法的输入参数和返回类型，其中输入参数和返回结果为 torch 张量或 QuestionAnsweringModelOutput 类型对象
    
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # 用给定值或者模型配置中的 use_return_dict 值初始化 return_dict
    
            outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # 将输入参数传递给 transformer 层，得到 transformer 的输出结果
    
            sequence_output = outputs[0]
            # 提取输出结果中的第一个元素作为序列输出
    
            logits = self.qa_outputs(sequence_output)
            # 将序列输出传递给 qa_outputs，得到预测的 logits
    
            start_logits, end_logits = logits.split(1, dim=-1)
            # 将 logits 按照最后一个维度分割为 start_logits 和 end_logits
    
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            # 对 start_logits 和 end_logits 进行压缩维度并保持连续性
    
            total_loss = None
            # 初始化 total_loss 为 None
    
            if start_positions is not None and end_positions is not None:
                # 如果 start_positions 和 end_positions 都不为 None
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # 压缩多余维度
    
                ignored_index = start_logits.size(1)
                # 忽略索引为 start_logits 的第二个维度的大小
    
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)
                # 将 start_positions 和 end_positions 限制在有效范围内
    
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                # 使用 CrossEntropyLoss 计算损失，忽略给定索引
    
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                # 计算 start_loss 和 end_loss
    
                total_loss = (start_loss + end_loss) / 2
                # 计算总损失为 start_loss 和 end_loss 的平均值
    
            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output
            # 如果不返回字典，将输出结果组合成元组返回
    
            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            # 返回 QuestionAnsweringModelOutput 对象，包含损失、start_logits、end_logits、hidden_states 和 attentions
    ```py 
```