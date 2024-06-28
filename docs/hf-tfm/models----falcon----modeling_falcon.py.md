# `.\models\falcon\modeling_falcon.py`

```
# 指定编码格式为 UTF-8

# 版权声明和许可协议，此处声明代码版权归 Falcon 作者及 HuggingFace Inc. 团队所有，保留所有权利
# 根据 Apache 许可证 2.0 版本发布，除非符合许可协议，否则不得使用此文件
# 可以从以下网址获取许可协议的副本：http://www.apache.org/licenses/LICENSE-2.0
# 如果适用法律要求或书面同意，软件将按“原样”分发，没有任何形式的担保或条件
"""PyTorch Falcon model."""

import math  # 导入 math 库提供数学函数
import warnings  # 导入 warnings 库用于警告处理
from typing import TYPE_CHECKING, Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 功能
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss  # 导入损失函数
from torch.nn import functional as F  # 导入 PyTorch 的函数式接口，并重命名为 F

# 导入工具函数和类
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
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import is_torch_greater_or_equal_than_2_0  # 导入 PyTorch 版本判断工具函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

# 导入 FalconConfig 配置类
from .configuration_falcon import FalconConfig


if TYPE_CHECKING:
    from ...configuration_utils import PretrainedConfig

# 如果 flash_attn 2.x 可用，则导入相应的函数和模块
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取 logger 实例
logger = logging.get_logger(__name__)

# Falcon 模型的预训练模型存档列表
FALCON_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tiiuae/falcon-40b",
    "tiiuae/falcon-40b-instruct",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-rw-7b",
    "tiiuae/falcon-rw-1b",
]
# 用于文档的检查点名
_CHECKPOINT_FOR_DOC = "Rocketknight1/falcon-rw-1b"
# 用于文档的配置名
_CONFIG_FOR_DOC = "FalconConfig"


# 注意：在训练期间，我们未融合矩阵乘法和偏置项，这意味着操作之间需要一个额外的量化步骤到 bfloat16。
# 为了不降低 HF 代码的质量，我们在最终模型中保留了这些特征。
class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 执行线性变换
        hidden_states = input @ self.weight.T
        # 如果没有偏置项，则直接返回变换后的结果
        if self.bias is None:
            return hidden_states
        # 否则，将偏置项加到变换后的结果中并返回
        return hidden_states + self.bias


# 从 transformers.models.llama.modeling_llama 中复制的函数，用于旋转输入张量一半隐藏维度的值
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    # 使用切片操作从张量 x 中取出从中间到末尾的所有维度，保持其他维度不变
    x2 = x[..., x.shape[-1] // 2 :]
    # 使用 torch.cat 函数沿着最后一个维度将 x2 和 x1 张量连接起来
    return torch.cat((-x2, x1), dim=-1)
# 从transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb复制而来
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """对查询张量和键张量应用旋转位置嵌入。

    Args:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.Tensor`):
            对应于查询和键张量的标记位置索引。例如，当使用KV缓存时，可以传递偏移的位置ID。
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            'unsqueeze_dim' 参数指定沿其进行展开的维度，以便将 cos[position_ids] 和 sin[position_ids] 正确广播到 q 和 k 的维度。
            例如，注意 cos[position_ids] 和 sin[position_ids] 的形状为 [batch_size, seq_len, head_dim]。然后，
            如果 q 和 k 的形状为 [batch_size, heads, seq_len, head_dim]，设置 unsqueeze_dim=1 使得 cos[position_ids] 和 sin[position_ids]
            可以广播到 q 和 k 的形状。类似地，如果 q 和 k 的形状为 [batch_size, seq_len, heads, head_dim]，则设置 unsqueeze_dim=2。

    Returns:
        `tuple(torch.Tensor)`: 包含使用旋转位置嵌入旋转后的查询和键张量。
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# 从transformers.models.llama.modeling_llama._get_unpad_data复制而来
def _get_unpad_data(attention_mask):
    """获取未填充数据。

    Args:
        attention_mask (`torch.Tensor`): 注意力掩码张量。

    Returns:
        `tuple`: 包含以下三个元素的元组:
            - `torch.Tensor`: 指示非填充位置索引的张量。
            - `torch.Tensor`: 指示累积序列长度的张量，用于填充。
            - `int`: 批次中最大序列长度。
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 从transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding复制而来，更改为FalconRotaryEmbedding
class FalconRotaryEmbedding(nn.Module):
    # 初始化函数，设置模型参数和缓存
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类初始化方法
        super().__init__()

        # 设置模型维度
        self.dim = dim
        # 设置最大位置编码长度，默认为2048
        self.max_position_embeddings = max_position_embeddings
        # 设置基础频率，默认为10000
        self.base = base

        # 计算频率的倒数，用于位置编码
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # 将频率的倒数注册为模型的缓存，不持久化
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 调用内部方法设置余弦和正弦缓存，以便 `torch.jit.trace` 方法能正常工作
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 内部方法，设置余弦和正弦缓存
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 记录当前缓存的最大序列长度
        self.max_seq_len_cached = seq_len
        # 创建序列长度张量 t，并转换为与 inv_freq 相同类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率张量与位置张量的外积
        freqs = torch.outer(t, self.inv_freq)
        # 按最后一个维度拼接频率张量，构成位置编码矩阵
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将余弦值缓存注册为模型的缓存，不持久化，并转换为指定数据类型
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 将正弦值缓存注册为模型的缓存，不持久化，并转换为指定数据类型
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 前向传播方法
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]

        # 如果指定了新的序列长度超过当前缓存的最大序列长度
        if seq_len > self.max_seq_len_cached:
            # 重新设置余弦和正弦缓存
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回当前缓存中的余弦和正弦值，截取到指定的序列长度，转换为输入张量 x 的数据类型
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding复制并修改为Falcon
# TODO @joao: 经过静态缓存后不再从LLama复制，修复我（复制 -> Copied）
class FalconLinearScalingRotaryEmbedding(FalconRotaryEmbedding):
    """使用线性缩放扩展的FalconRotaryEmbedding。由Reddit用户/u/kaiokendev贡献"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # 创建一个整数序列t，其长度为max_seq_len_cached，使用给定的设备和数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        # 将整数序列t除以scaling_factor，以得到频率
        t = t / self.scaling_factor

        # 计算频率矩阵，使用torch.outer计算外积
        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但使用不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将计算得到的cosine和sine值注册为缓冲区，使用给定的数据类型，并标记为非持久化
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding复制并修改为Falcon
# TODO @joao: 经过静态缓存后不再从LLama复制，修复我（复制 -> Copied）
class FalconDynamicNTKScalingRotaryEmbedding(FalconRotaryEmbedding):
    """使用动态NTK缩放扩展的FalconRotaryEmbedding。由Reddit用户/u/bloc97和/u/emozilla贡献"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        # 如果序列长度超过最大位置嵌入数，计算基础频率
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 创建一个整数序列t，其长度为max_seq_len_cached，使用给定的设备和数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率矩阵，使用torch.outer计算外积
        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但使用不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将计算得到的cosine和sine值注册为缓冲区，使用给定的数据类型，并标记为非持久化
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    # 从注意力掩码构建Alibi张量，指定批次大小和序列长度
    batch_size, seq_length = attention_mask.shape
    # 计算最接近的2的幂次方，小于等于给定数值num_heads的最大整数幂次方
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    # 计算基数，用于生成注意力偏置（attention bias）
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    # 生成幂次序列，从1开始到最接近的2的幂次方（包含）
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    # 计算斜率，即基数的各幂次方
    slopes = torch.pow(base, powers)

    # 如果最接近的2的幂次方不等于num_heads，则需要额外计算
    if closest_power_of_2 != num_heads:
        # 计算额外基数
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        # 计算剩余头数
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        # 计算额外幂次序列，步长为2
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        # 将额外的斜率拼接到原始斜率序列中
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # 创建alibi张量，用于相对位置偏置（relative position bias），其形状为(batch_size, num_heads, query_length, key_length)
    # 在此设置为(batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)，query_length维度将会正确广播
    # 这与T5模型中的相对位置偏置基本相同，参见：https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None].bfloat16() * arange_tensor
    # 重新形状化alibi张量，并转换为指定的dtype
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
# 从transformers.models.bloom.modeling_bloom.dropout_add复制而来，定义了一个dropout add函数
def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor 输入张量
        residual (`torch.tensor`, *required*):
            residual tensor 剩余张量
        prob (`float`, *required*):
            dropout probability dropout概率
        training (`bool`, *required*):
            training mode 训练模式
    """
    # 对输入张量x应用dropout操作，根据training参数决定是否使用
    out = F.dropout(x, p=prob, training=training)
    # 将dropout后的结果与剩余张量residual相加
    out = residual + out
    # 返回结果张量out
    return out


class FalconAttention(nn.Module):
    def __init__(self, config: FalconConfig):
        super().__init__()

        # 初始化FalconAttention类的配置参数
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 检查hidden_size是否能被num_heads整除，若不能则抛出错误
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 如果配置为使用rotary，则初始化rope
        if config.rotary:
            self._init_rope()

        # Layer-wise attention scaling，初始化注意力权重的缩放因子
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        
        # 根据配置选择不同的输出维度qkv_out_dim
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        
        # 初始化query_key_value线性层，用于计算查询、键、值
        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        
        # 初始化输出的线性层dense
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        
        # 初始化注意力的dropout层
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # 根据配置初始化num_kv_heads
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1

    # 从transformers.models.llama.modeling_llama.LlamaAttention._init_rope复制而来，用于初始化rope
    # 初始化 RoPE（Rotary Positional Embedding），根据配置设置不同的缩放方式
    def _init_rope(self):
        # 如果配置中没有指定 RoPE 缩放方式，则使用默认的 FalconRotaryEmbedding
        if self.config.rope_scaling is None:
            self.rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 否则根据配置选择不同的 RoPE 缩放类型
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                # 使用线性缩放方式的 RoPE
                self.rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                # 使用动态 NTK 缩放方式的 RoPE
                self.rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 如果缩放类型未知，则抛出 ValueError 异常
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 将融合后的查询/键/值张量拆分为多个头部，根据模型的不同架构选择不同的拆分方式
    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            # 如果是新的解码器架构，则按照指定方式拆分
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            # 如果不是多查询模式，则按照普通的拆分方式
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            # 否则按照另一种特定的拆分方式处理
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    # 从 transformers 库中复制的函数，用于合并注意力机制中的多个头部
    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # 获取输入张量的形状信息：batch_size * num_heads, seq_length, head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        # 计算真实的 batch_size，即 batch_size * num_heads 的商，表示真正的 batch 数量
        batch_size = batch_size_and_num_heads // self.num_heads

        # 将输入张量重新视图化以分解批次大小
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # 通过维度置换将头部维度与序列长度维度交换位置
        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # 将 num_heads 和 head_dim 合并到一个维度中
        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
# 定义了一个名为 FalconFlashAttention2 的类，继承自 FalconAttention 类。
# Falcon flash attention 模块，其权重未被修改。唯一需要更改的是前向传播，在这里需要正确调用 flash attention 的公共 API，并处理输入中可能存在的填充标记。

# 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
# FalconFlashAttention2 类的构造函数，调用父类构造函数初始化。
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # TODO: 一旦 RoCm 的 Flash Attention 升级到 2.1 版本，此处应该移除。
    # flash_attn<2.1 生成左上角对齐的因果掩码，而此处需要的是右下角对齐，默认 flash_attn>=2.1 才支持此特性。该属性用于处理这两个版本之间的差异。
    # 参考：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0
    # 注意，在 flash_attn<2.1 版本中，除非 q_seqlen == 1，否则使用 q_seqlen != k_seqlen 会产生错误的掩码（左上角对齐）。
    self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
# FalconFlashAttention2 类的私有方法 _flash_attention_forward，执行 flash attention 的前向传播。
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Determine if causal masking is required based on the current configuration
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Temporary condition until Flash Attention for RoCm is updated
            causal = self.is_causal and query_length != 1

        # Check if there are padding tokens in the input sequences
        if attention_mask is not None:
            # Get the batch size
            batch_size = query_states.shape[0]
            # Unpad the input sequences based on the attention mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract sequence lengths after unpadding
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            # Extract maximum sequence lengths in the current batch
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Compute attention scores for unpad input using variable-length Flash Attention
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

            # Pad the attention scores to match the original input length
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Compute attention scores using standard Flash Attention
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # Return the computed attention scores
        return attn_output
    # 在 `_upad_input` 方法中，根据给定的注意力掩码和查询长度处理输入数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取批次大小、键值序列长度、头数以及头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 通过索引重组键层数据，以适应未填充的序列
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 通过索引重组值层数据，以适应未填充的序列
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据查询长度处理查询层数据
        if query_length == kv_seq_len:
            # 若查询长度等于键值序列长度，直接重组查询层数据
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 若查询长度为1，进行相关处理
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个 memcpy，这是非常糟糕的。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 处理左填充的情况，使用 -query_length: 切片
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的查询层、键层、值层、查询索引、当前序列长度元组和最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义一个名为 FalconMLP 的神经网络模块
class FalconMLP(nn.Module):
    # 初始化方法，接受一个 FalconConfig 类型的参数 config
    def __init__(self, config: FalconConfig):
        super().__init__()
        # 从配置中获取隐藏层大小
        hidden_size = config.hidden_size

        # 定义全连接层，将隐藏层映射到4倍隐藏层大小，带有可选的偏置
        self.dense_h_to_4h = FalconLinear(hidden_size, 4 * hidden_size, bias=config.bias)
        # 使用 GELU 激活函数
        self.act = nn.GELU()
        # 定义全连接层，将4倍隐藏层大小映射回隐藏层大小，带有可选的偏置
        self.dense_4h_to_h = FalconLinear(4 * hidden_size, hidden_size, bias=config.bias)
        # 获取隐藏层的 dropout 概率
        self.hidden_dropout = config.hidden_dropout

    # 前向传播方法，接受输入张量 x，返回输出张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 GELU 激活函数的全连接层操作
        x = self.act(self.dense_h_to_4h(x))
        # 第二个全连接层操作
        x = self.dense_4h_to_h(x)
        return x


# FalconAttention 类的字典映射，根据配置不同选择不同的注意力机制类
FALCON_ATTENTION_CLASSES = {
    "eager": FalconAttention,
    "sdpa": FalconAttention,  # FalconAttention 原始实现同时包含有/无 SDPA 的前向传播
    "flash_attention_2": FalconFlashAttention2,
}


# FalconDecoderLayer 类定义
class FalconDecoderLayer(nn.Module):
    # 初始化方法，接受一个 FalconConfig 类型的参数 config
    def __init__(self, config: FalconConfig):
        super().__init__()
        # 从配置中获取隐藏层大小
        hidden_size = config.hidden_size
        # 获取注意力头的数量
        self.num_heads = config.num_attention_heads

        # 根据配置选择适当的注意力机制类初始化 self_attention 属性
        self.self_attention = FALCON_ATTENTION_CLASSES[config._attn_implementation](config)
        # 初始化 MLP 层
        self.mlp = FalconMLP(config)
        # 获取隐藏层的 dropout 概率
        self.hidden_dropout = config.hidden_dropout
        # 保存配置
        self.config = config

        # 根据配置选择不同的解码器架构
        if config.new_decoder_architecture:
            # 在 self-attention 前进行层归一化
            self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            # 在 MLP 前进行层归一化
            self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            # 输入层的层归一化
            self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            # 如果不使用并行注意力，则在注意力后进行层归一化
            if not config.parallel_attn:
                self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    # 前向传播方法
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
            # 检查是否传入了 "padding_mask" 参数，如果是则发出警告，因为在 v4.37 版本中将移除，请使用 `attention_mask` 替代。
            if "padding_mask" in kwargs:
                warnings.warn(
                    "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
                )

        # 保存输入的隐藏状态，以备后续计算中使用
        residual = hidden_states

        # 根据配置选择不同的层归一化方法
        if self.config.new_decoder_architecture:
            # 使用新的解码器架构，应用注意力层和MLP层的归一化
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            # 使用旧的解码器架构，只应用输入层的归一化
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # 自注意力机制
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

        attention_output = attn_outputs[0]

        # 如果不使用新的解码器架构，则进行残差连接和Dropout操作
        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                # 并行注意力模式，直接使用注意力层的归一化输出
                mlp_layernorm_out = attention_layernorm_out
            else:
                # 非并行注意力模式，进行残差连接和Dropout操作
                residual = dropout_add(
                    attention_output, residual, self.config.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        # 提取自注意力机制的输出
        outputs = attn_outputs[1:]

        # MLP层的前向传播
        mlp_output = self.mlp(mlp_layernorm_out)

        # 如果使用新的解码器架构或并行注意力模式，则将自注意力输出与MLP输出相加
        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        # 最终的输出，进行Dropout和残差连接
        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        # 如果使用缓存，输出包括隐藏状态、present以及注意力信息；否则，不包括隐藏状态
        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        # 返回模型的输出，包括隐藏状态、present以及注意力信息
        return outputs  # hidden_states, present, attentions
# FalconPreTrainedModel 类的文档字符串，描述了该类继承自 PreTrainedModel，介绍了其方法和通用功能。
# 包含了关于如何使用该模型以及其参数配置的信息。
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

# FalconPreTrainedModel 类的输入文档字符串，目前为空。
FALCON_INPUTS_DOCSTRING = r"""
"""


class FalconPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # FalconPreTrainedModel 类的配置类，指定为 FalconConfig。
    config_class = FalconConfig
    # 模型基础前缀，用于模型的标识。
    base_model_prefix = "transformer"
    # 是否支持梯度检查点。
    supports_gradient_checkpointing = True
    # 不进行分割的模块列表。
    _no_split_modules = ["FalconDecoderLayer"]
    # 是否支持闪光注意力版本2。
    _supports_flash_attn_2 = True
    # 是否支持自发对齐。
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        # 初始化模型权重的函数。
        if isinstance(module, nn.Linear) or isinstance(module, FalconLinear):
            # 对于线性层和特定的自定义线性层 FalconLinear，使用正态分布初始化权重。
            # 与 TensorFlow 版本稍有不同，后者使用截断正态分布进行初始化。
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将其初始化为零。
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，使用正态分布初始化权重。
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了填充索引，则将该索引处的权重初始化为零。
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            # 对于 LayerNorm 层，将偏置初始化为零，将权重初始化为1.0。
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # 从 transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa 适配而来的方法。
    @classmethod
    # 检查并启用 SDPA（Scaled Dot-Product Attention）的设置，可能会修改配置并返回更新后的配置对象。
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> "PretrainedConfig":
        # 注意：自 PyTorch 2.0 起，Falcon 支持 SDPA。为了向后兼容性，保持这样的设定（torch>=2.0 自动使用 SDPA）。
        # 如果只进行严格检查，且当前的 torch 版本不符合要求，则抛出 ImportError 异常。
        if hard_check_only:
            if not is_torch_greater_or_equal_than_2_0:
                raise ImportError("PyTorch SDPA requirements in Transformers are not met. Please install torch>=2.0.")

        # 如果当前 torch 版本不符合要求，则直接返回原始配置对象。
        if not is_torch_greater_or_equal_than_2_0:
            return config

        # 检查是否使用了 BetterTransformer，如果是，则直接返回原始配置对象。
        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
        if _is_bettertransformer:
            return config

        # 如果不是严格检查模式，将注意力机制实现设为 "sdpa"。
        if not hard_check_only:
            config._attn_implementation = "sdpa"
        # 返回更新后的配置对象。
        return config
# 使用装饰器添加文档字符串，描述这是一个不带特定头部的 Falcon 模型变换器的类
@add_start_docstrings(
    "The bare Falcon Model transformer outputting raw hidden-states without any specific head on top.",
    FALCON_START_DOCSTRING,
)
# FalconModel 类，继承自 FalconPreTrainedModel
class FalconModel(FalconPreTrainedModel):
    # 初始化方法，接受一个 FalconConfig 类型的参数 config
    def __init__(self, config: FalconConfig):
        # 调用父类 FalconPreTrainedModel 的初始化方法
        super().__init__(config)

        # 设置模型的嵌入维度和注意力头数
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi

        # 嵌入层 + LayerNorm 嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer 块
        self.h = nn.ModuleList([FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        # 最终的 Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 梯度检查点设置为 False
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入嵌入的方法，接受一个新的嵌入张量作为参数
    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    # 使用装饰器添加文档字符串，描述这是一个 Falcon 模型变换器的前向方法，包含输入参数的详细说明
    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向方法，接受多个输入参数并返回多个输出
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



# 使用装饰器添加文档字符串，描述这是一个带有语言建模头的 Falcon 模型变换器的类
@add_start_docstrings(
    "The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).",
    FALCON_START_DOCSTRING,
)
# FalconForCausalLM 类，继承自 FalconPreTrainedModel
class FalconForCausalLM(FalconPreTrainedModel):
    # 静态变量，指示与输入嵌入权重相关联的键名列表
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受一个 FalconConfig 类型的参数 config
    def __init__(self, config: FalconConfig):
        # 调用父类 FalconPreTrainedModel 的初始化方法
        super().__init__(config)
        
        # 创建一个 FalconModel 实例，并传入配置 config
        self.transformer = FalconModel(config)
        
        # 创建一个线性层用于语言建模的头部，输入维度为 config.hidden_size，输出维度为 config.vocab_size，没有偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入的方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入的方法，接受一个新的嵌入张量作为参数
    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings
    # 准备生成的输入参数，返回一个字典
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            # 获取过去键值张量的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:
                # 如果输入的长度大于过去的长度，则移除前缀长度为过去的长度
                remove_prefix_length = past_length
            else:
                # 否则默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 更新输入的 ID，移除前缀部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 注意：Falcon 版本中带有 alibi 的情况下不使用 position_ids。它在 RoPE 中使用。
        if not self.transformer.use_alibi and attention_mask is not None and position_ids is None:
            # 为批量生成创建即时的 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                # 如果有过去的键值，只保留与输入长度相匹配的 position_ids
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 返回包含所有生成输入的字典
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    # 将模型前向传播方法装饰为添加文档字符串和代码示例文档字符串
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
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        
        # 如果未指定返回字典，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 使用 Transformer 模型处理输入数据，获取模型的输出
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
        # 获取 Transformer 输出中的隐藏状态
        hidden_states = transformer_outputs[0]

        # 使用语言模型头部生成预测的 logits
        lm_logits = self.lm_head(hidden_states)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将 logits 向左移动一个位置，以便预测下一个 token
            shift_logits = lm_logits[..., :-1, :].contiguous()
            # 将 labels 向右移动一个位置，与 shift_logits 对齐
            shift_labels = labels[..., 1:].contiguous()
            # 获取 batch_size, seq_length 和 vocab_size 的大小
            batch_size, seq_length, vocab_size = shift_logits.shape
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        # 如果不要求返回字典，则返回模型输出的元组形式
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # 如果需要返回字典形式的输出，则创建 CausalLMOutputWithCrossAttentions 对象
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """

        # 获取在所有需要索引的设备上的 `beam_idx` 的副本。
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        # 对 `past` 进行重新排序，以便与每个生成步骤中正确的 `beam_idx` 匹配。
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in past
        )
        # 返回重新排序后的 `past`。
        return reordered_past
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
class FalconForSequenceClassification(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = FalconModel(config)  # 初始化 FalconModel，并传入配置信息
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)  # 生成线性层，用于分类

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和最终处理操作


@add_start_docstrings(
    """
    Falcon Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FALCON_START_DOCSTRING,
)
class FalconForTokenClassification(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.num_labels = config.num_labels  # 根据配置设置标签数量

        self.transformer = FalconModel(config)  # 初始化 FalconModel，并传入配置信息

        # 设置分类器的 dropout 概率
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)  # 创建 dropout 层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 创建线性层，用于分类

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和最终处理操作
    # 将模型前向传播方法添加文档字符串，用于文档化模型输入参数和示例代码
    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法，接受多个输入参数并返回分类器输出或损失
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 如果没有显式指定 return_dict，则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 transformer 处理输入数据，得到变换器的输出结果
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

        # 从变换器的输出中获取隐藏状态并应用 dropout
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        
        # 使用分类器模型对隐藏状态进行分类预测
        logits = self.classifier(hidden_states)

        # 初始化损失为 None
        loss = None
        # 如果有提供标签信息，则计算损失值
        if labels is not None:
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        # 如果 return_dict 为 False，则组装输出为元组
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
@add_start_docstrings(
    """
    The Falcon Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FALCON_START_DOCSTRING,
)
class FalconForQuestionAnswering(FalconPreTrainedModel):
    """
    Falcon model for question answering tasks, extending FalconPreTrainedModel.

    Inherits from FalconPreTrainedModel and implements a transformer with a span classification head
    for tasks such as SQuAD. It includes linear layers to compute `span start logits` and `span end logits`.
    """

    def __init__(self, config):
        """
        Initializes the FalconForQuestionAnswering model.

        Args:
            config (FalconConfig): Configuration object specifying the model architecture and parameters.
        """
        super().__init__(config)
        # Initialize the FalconModel with the provided configuration
        self.transformer = FalconModel(config)
        # Linear layer for predicting start and end positions in the span
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and perform additional post-initialization processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FALCON_INPUTS_DOCSTRING)
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
    ):
        """
        Defines the forward pass for FalconForQuestionAnswering.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs.
            attention_mask (torch.FloatTensor, optional): Mask to avoid performing attention on padding tokens.
            head_mask (torch.FloatTensor, optional): Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (torch.FloatTensor, optional): Embedded input tokens.
            start_positions (torch.LongTensor, optional): Index of the start position for the answer span.
            end_positions (torch.LongTensor, optional): Index of the end position for the answer span.
            output_attentions (bool, optional): Whether to output attentions weights.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary as the output.

        Returns:
            FalconForQuestionAnsweringOutput: Output object containing the logits for start and end positions of the span.
        """
        # Forward pass through the FalconModel transformer
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Compute logits for start and end positions using the qa_outputs linear layer
        logits = self.qa_outputs(outputs[0])

        # Return FalconForQuestionAnsweringOutput containing the logits
        return FalconForQuestionAnsweringOutput(
            loss=None if start_positions is None or end_positions is None else self.loss(logits, start_positions, end_positions),
            start_logits=logits[:, :, 0] if logits.shape[:2] == 3 else None,
            end_logits=logits[:, :, 1] if logits.shape[:2] == 3 else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # Decide whether to use the return_dict based on input or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs to the transformer model and retrieve outputs
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the sequence output from the transformer outputs
        sequence_output = outputs[0]

        # Get logits for question answering from the sequence output
        logits = self.qa_outputs(sequence_output)

        # Split logits into start and end logits for the answer span
        start_logits, end_logits = logits.split(1, dim=-1)

        # Squeeze the logits tensors to remove unnecessary dimensions
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Handle multi-GPU training by squeezing additional dimensions if present
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # Clamp positions to prevent them from exceeding sequence length
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Compute the CrossEntropyLoss for start and end positions
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # Calculate total loss as the average of start and end losses
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # Return outputs without loss if return_dict is False
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # Return structured output using QuestionAnsweringModelOutput if return_dict is True
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```