# `.\models\qwen2\modeling_qwen2.py`

```py
# coding=utf-8
# 版权所有 2024 年 Qwen 团队，阿里巴巴集团和 HuggingFace Inc. 团队。保留所有权利。
#
# 本代码基于 EleutherAI 的 GPT-NeoX 库和此库中的 GPT-NeoX 和 OPT 实现进行了修改，以适应与 Meta AI 团队训练的模型相比的轻微架构差异。
#
# 根据 Apache 许可证版本 2.0 许可，除非符合许可要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发本软件，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch Qwen2 模型。"""
import inspect  # 导入 inspect 模块，用于获取对象的信息
import math  # 导入 math 模块，提供数学函数
import warnings  # 导入 warnings 模块，用于警告控制
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 模块
import torch.nn.functional as F  # 导入 PyTorch 的函数模块
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数相关模块

from ...activations import ACT2FN  # 导入激活函数映射
from ...cache_utils import Cache, DynamicCache  # 导入缓存相关的工具函数
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa  # 导入注意力掩码相关的函数
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast  # 导入模型输出类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型的基类
from ...utils import (  # 导入工具函数
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_qwen2 import Qwen2Config  # 导入 Qwen2 配置类


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # 如果可用，导入 flash_attn 的函数
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa：导入 Bert 相关的填充函数

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"  # 文档用的模型检查点名称
_CONFIG_FOR_DOC = "Qwen2Config"  # 文档用的配置名称

QWEN2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Qwen/Qwen2-7B-beta",
    # 查看所有 Qwen2 模型请访问 https://huggingface.co/models?filter=qwen2
]


# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)  # 计算每个样本中非填充部分的序列长度总和
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()  # 获取非填充位置的索引
    max_seqlen_in_batch = seqlens_in_batch.max().item()  # 找出批次中最大的序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))  # 计算累积序列长度
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 从 transformers.models.llama.modeling_llama.LlamaRMSNorm 复制并更名为 Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化权重参数为全部为1的张量
        self.variance_epsilon = eps  # 设置方差的epsilon值

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype  # 记录输入张量的数据类型
        hidden_states = hidden_states.to(torch.float32)  # 将输入张量转换为float32类型
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # 计算张量的方差
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # 根据方差和epsilon进行归一化
        return self.weight * hidden_states.to(input_dtype)  # 返回归一化后的张量乘以权重参数


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim  # 设置维度参数
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入长度
        self.base = base  # 设置基础值
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册频率逆数的缓冲张量

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len  # 记录缓存的最大序列长度
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)  # 计算频率张量
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)  # 拼接cos和sin的嵌入张量
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)  # 注册cos缓存张量
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)  # 注册sin缓存张量

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),  # 返回cos缓存张量
            self.sin_cached[:seq_len].to(dtype=x.dtype),  # 返回sin缓存张量
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # 取输入张量的前一半维度
    x2 = x[..., x.shape[-1] // 2 :]  # 取输入张量的后一半维度
    return torch.cat((-x2, x1), dim=-1)  # 返回将输入张量的后一半维度与前一半维度拼接后的张量


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    Args:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.Tensor`):
            查询和键张量对应的位置索引。例如，当使用 KV 缓存时，可以传递偏移的位置 id。
        unsqueeze_dim (`int`, *optional*, 默认为 1):
            'unsqueeze_dim' 参数指定沿其展开 cos[position_ids] 和 sin[position_ids] 的维度，以便它们可以正确地广播到 q 和 k 的维度。
            例如，cos[position_ids] 和 sin[position_ids] 的形状为 [batch_size, seq_len, head_dim]。
            如果 q 和 k 的形状为 [batch_size, heads, seq_len, head_dim]，则设置 unsqueeze_dim=1 使得 cos[position_ids] 和 sin[position_ids] 可以广播到 q 和 k 的形状。
            类似地，如果 q 和 k 的形状为 [batch_size, seq_len, heads, head_dim]，则设置 unsqueeze_dim=2。

    Returns:
        `tuple(torch.Tensor)`: 包含使用旋转位置嵌入旋转后的查询和键张量的元组。
"""
    # 按照位置索引从 cos 中选择并展开维度
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # 按照位置索引从 sin 中选择并展开维度
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 计算旋转后的查询嵌入
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 计算旋转后的键嵌入
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 返回旋转后的查询和键张量
    return q_embed, k_embed
# 从 transformers.models.mistral.modeling_mistral.MistralMLP 复制并修改为 Qwen2MLP
class Qwen2MLP(nn.Module):
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 将配置对象保存在实例中
        self.config = config
        # 从配置中获取隐藏层大小和中间层大小
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # 创建一个线性层，用于门控投影，输入维度是隐藏层大小，输出维度是中间层大小，无偏置
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建一个线性层，用于上游投影，输入维度是隐藏层大小，输出维度是中间层大小，无偏置
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建一个线性层，用于下游投影，输入维度是中间层大小，输出维度是隐藏层大小，无偏置
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 根据配置中的激活函数名，选择相应的激活函数，并保存在实例中
        self.act_fn = ACT2FN[config.hidden_act]

    # 前向传播方法，接收输入张量 x
    def forward(self, x):
        # 对输入张量进行门控投影，然后应用激活函数，再乘以上游投影结果，最后下游投影得到输出
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 从 transformers.models.llama.modeling_llama.repeat_kv 复制过来
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这相当于 torch.repeat_interleave(x, dim=1, repeats=n_rep)。将隐藏状态从 (batch,
    num_key_value_heads, seqlen, head_dim) 扩展为 (batch, num_attention_heads, seqlen, head_dim)
    """
    # 获取隐藏状态张量的形状信息
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # 如果重复次数 n_rep 为 1，直接返回原始隐藏状态张量
    if n_rep == 1:
        return hidden_states
    # 在第二维度上扩展隐藏状态张量，重复 n_rep 次
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 重新整形扩展后的张量，将第二和第三维度合并为新的第二维度
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """
    从 'Attention Is All You Need' 论文中的多头注意力机制修改而来。修改为使用滑动窗口注意力：Longformer
    和 "Generating Long Sequences with Sparse Transformers"。
    """
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config  # 设置实例的配置参数对象
        self.layer_idx = layer_idx  # 设置实例的层索引，可选

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size  # 从配置中获取隐藏层大小
        self.num_heads = config.num_attention_heads  # 从配置中获取注意力头的数量
        self.head_dim = self.hidden_size // self.num_heads  # 计算每个注意力头的维度
        self.num_key_value_heads = config.num_key_value_heads  # 从配置中获取键值头的数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 计算键值头的组数
        self.max_position_embeddings = config.max_position_embeddings  # 从配置中获取最大位置嵌入数
        self.rope_theta = config.rope_theta  # 从配置中获取绳索旋转角度
        self.is_causal = True  # 设置实例是否因果
        self.attention_dropout = config.attention_dropout  # 从配置中获取注意力丢弃率

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        # 初始化查询投影层线性变换
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # 初始化键投影层线性变换
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # 初始化值投影层线性变换
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # 初始化输出投影层线性变换，没有偏置项

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        # 初始化旋转嵌入层对象，用于处理注意力旋转操作

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
# Qwen2FlashAttention2 类，继承自 Qwen2Attention，实现了 Qwen2 闪存注意力模块。
# 该模块主要的改动在于前向传播过程中需要正确调用闪存注意力的公共 API，并处理可能包含的填充标记。
# 另外，对于滑动窗口注意力，仅应用于底部 config.max_window_layers 层。

# 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # TODO: 当 Flash Attention 版本升级到 2.1 后应移除此部分。
    # flash_attn<2.1 生成左上对齐的因果蒙版，而需要的是右下对齐，默认在 flash_attn>=2.1 中已实现。该属性用于处理此差异。
    # 参考: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0。
    # 注意，对于 flash_attn<2.1，当 q_seqlen != k_seqlen（除了 q_seqlen == 1 的情况）会产生错误的蒙版（左上）。
    self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    # Qwen2FlashAttention2 类的前向传播函数
    # hidden_states: 输入的隐藏状态张量
    # attention_mask: 可选的注意力蒙版张量
    # position_ids: 可选的位置 ID 张量
    # past_key_value: 可选的缓存键值对
    # output_attentions: 是否输出注意力权重
    # use_cache: 是否使用缓存
    # **kwargs: 其他关键字参数

def _flash_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    use_sliding_windows=False,
):
    # Qwen2FlashAttention2 类的闪存注意力前向传播函数
    # query_states: 查询状态
    # key_states: 键状态
    # value_states: 值状态
    # attention_mask: 注意力蒙版
    # query_length: 查询长度
    # dropout: dropout 比率，默认为 0.0
    # softmax_scale: softmax 缩放参数，可选
    # use_sliding_windows: 是否使用滑动窗口， 默认为 False

# 从 transformers.models.mistral.modeling_mistral.MistralFlashAttention2._upad_input 复制而来
    # 定义一个方法 _upad_input，接受查询层、键层、值层、注意力掩码和查询长度作为输入参数
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取批处理大小、键值序列长度、头数、头维度
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 如果键值序列长度不等于注意力掩码的最后一个维度长度
        if kv_seq_len != attention_mask.shape[-1]:
            # 调整注意力掩码，使其匹配键值序列的长度
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        # 从注意力掩码中获取未填充数据的索引、当前序列长度、批处理中最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        # 根据获取的索引重新排序键层和值层
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        # 如果查询长度等于键值序列长度
        if query_length == kv_seq_len:
            # 重新排序查询层
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果查询长度为1
        elif query_length == 1:
            # 将查询层缩减为一维
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个内存复制操作，效率不高。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 根据查询长度调整注意力掩码，获取未填充的输入
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回重新排序后的查询层、键层、值层，以及查询层索引、当前序列长度元组和最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# Copied from transformers.models.mistral.modeling_mistral.MistralSdpaAttention with Mistral->Qwen2
class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        """
        Override of forward method from Qwen2Attention to adapt to SDPA API.

        Parameters:
        - hidden_states (torch.Tensor): Input tensor to the attention module.
        - attention_mask (Optional[torch.Tensor]): Mask tensor indicating which elements should be attended to.
        - position_ids (Optional[torch.LongTensor]): Tensor containing positional ids.
        - past_key_value (Optional[Cache]): Cached key value pairs from previous computations.
        - output_attentions (bool): Whether to output attention weights.
        - use_cache (bool): Whether to use cached key value pairs for future computations.

        Returns:
        - Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: Tuple containing:
            - torch.Tensor: Output tensor from the attention module.
            - Optional[torch.Tensor]: Attention weights if `output_attentions` is `True`.
            - Optional[Tuple[torch.Tensor]]: Cached key value pairs if `use_cache` is `True`.
        """
        raise NotImplementedError

QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        # Initialize self attention mechanism based on configuration
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # Initialize MLP layer
        self.mlp = Qwen2MLP(config)

        # Layer normalization for input to the layer
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Layer normalization after attention mechanism
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Forward pass for Qwen2 decoder layer.

        Parameters:
        - hidden_states (torch.Tensor): Input tensor to the decoder layer.
        - attention_mask (Optional[torch.Tensor]): Mask tensor indicating which elements should be attended to.
        - position_ids (Optional[torch.LongTensor]): Tensor containing positional ids.
        - past_key_value (Optional[Tuple[torch.Tensor]]): Cached key value pairs from previous computations.
        - output_attentions (Optional[bool]): Whether to output attention weights.
        - use_cache (Optional[bool]): Whether to use cached key value pairs for future computations.
        - **kwargs: Additional keyword arguments for future expansion.

        Returns:
        - Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: Tuple containing:
            - torch.Tensor: Output tensor from the decoder layer.
            - Optional[torch.Tensor]: Attention weights if `output_attentions` is `True`.
            - Optional[Tuple[torch.Tensor]]: Cached key value pairs if `use_cache` is `True`.
        """
        raise NotImplementedError
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 如果传入了 `padding_mask` 参数，则发出警告，提示该参数在 v4.37 版本中将被移除
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入层的输入，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *可选*): 注意力掩码，形状为 `(batch, sequence_length)`，其中填充元素为 0
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。查看返回的张量中 `attentions` 以获取更多细节。
            use_cache (`bool`, *可选*):
                如果设置为 `True`，将返回 `past_key_values` 键值状态，可用于加速解码（参见 `past_key_values`）。
            past_key_value (`Tuple(torch.FloatTensor)`, *可选*): 缓存的过去键和值投影状态
        """

        # 记录输入的残差连接
        residual = hidden_states

        # 应用输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力机制
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # 将残差连接应用到自注意力输出上
        hidden_states = residual + hidden_states

        # 全连接层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 构建输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存，则添加当前的键值状态到输出中
        if use_cache:
            outputs += (present_key_value,)

        return outputs
# QWEN2_START_DOCSTRING 是一个原始字符串文档，描述了该模型的继承和基本使用说明
QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 使用装饰器 @add_start_docstrings 添加文档注释，说明该类是基于 Qwen2PreTrainedModel 的一个裸模型
@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    # 设置模型的配置类和模型名称前缀
    config_class = Qwen2Config
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不拆分的模块列表
    _no_split_modules = ["Qwen2DecoderLayer"]
    # 跳过设备放置的键
    _skip_keys_device_placement = "past_key_values"
    # 支持快闪注意力机制 2
    _supports_flash_attn_2 = True
    # 支持自我注意力分配
    _supports_sdpa = True
    # 支持缓存类
    _supports_cache_class = True

    # 初始化权重函数，根据配置的初始化范围初始化线性层和嵌入层
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# QWEN2_INPUTS_DOCSTRING 是一个原始字符串文档，目前为空
QWEN2_INPUTS_DOCSTRING = r"""
"""

# 使用装饰器 @add_start_docstrings 添加文档注释，说明该类是 Qwen2PreTrainedModel 的一个具体实现，用于 Transformer 解码器
@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    # 初始化方法，接受一个 Qwen2Config 类型的参数 config
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id  # 设置填充索引
        self.vocab_size = config.vocab_size  # 设置词汇表大小

        # 创建词嵌入层，使用 config 中的参数进行初始化
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 创建多层解码器层的列表，每层都是 Qwen2DecoderLayer 类的实例
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation  # 设置注意力实现方式
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 设置 RMS 归一化器

        self.gradient_checkpointing = False  # 设置是否使用梯度检查点
        # 初始化权重并应用最终处理
        self.post_init()

    # 返回词嵌入层对象
    def get_input_embeddings(self):
        return self.embed_tokens
    # 定义一个方法，用于设置输入的嵌入向量
    def set_input_embeddings(self, value):
        # 将输入的嵌入向量赋给对象的embed_tokens属性
        self.embed_tokens = value

    # 使用装饰器将下面的方法添加文档字符串，文档字符串内容在外部定义为QWEN2_INPUTS_DOCSTRING
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    # 定义前向传播方法，接受多个输入参数
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ids，数据类型为torch中的LongTensor
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力遮罩张量，数据类型为torch.Tensor
        position_ids: Optional[torch.LongTensor] = None,  # 可选的位置ids张量，数据类型为torch中的LongTensor
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 可选的过去的键值张量列表，数据类型为包含torch中的FloatTensor的列表
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的输入嵌入张量，数据类型为torch.FloatTensor
        use_cache: Optional[bool] = None,  # 是否使用缓存的标志，数据类型为bool型，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重的标志，数据类型为bool型，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的标志，数据类型为bool型，可选
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，数据类型为bool型，可选
class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 使用配置对象初始化 Qwen2Model 模型
        self.model = Qwen2Model(config)
        # 设置词汇表大小
        self.vocab_size = config.vocab_size
        # 使用线性层初始化 lm_head，连接隐藏状态和词汇表大小，不带偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型中的 embed_tokens 属性，用作输入嵌入层
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        # 设置模型中的 embed_tokens 属性为给定的 value，用作输入嵌入层
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        # 返回模型的输出嵌入层 lm_head
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置模型的输出嵌入层 lm_head 为给定的 new_embeddings
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置模型的 decoder 属性为给定的 decoder
        self.model = decoder

    def get_decoder(self):
        # 返回模型的 decoder 属性
        return self.model

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        实现模型的前向传播逻辑，支持文档化字符串和返回字符串替换。
        """
        # 具体的前向传播逻辑在模型的实现中处理，这里只是声明
        pass

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        准备用于生成的输入，根据需要定制输入。
        """
        # 具体的输入准备逻辑在模型的实现中处理，这里只是声明
        pass
        # 如果传入的 past_key_values 不为空，则进行处理
        if past_key_values is not None:
            # 如果 past_key_values 是 Cache 类型的实例
            if isinstance(past_key_values, Cache):
                # 获取缓存的序列长度、已看到的 token 数量和最大缓存长度
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                # 否则从 past_key_values 中获取缓存长度和已看到的 token 数量
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # 仅保留未处理的 token：
            # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，则说明部分输入仅作为缓存传递
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果 past_length 小于 input_ids 的长度，则说明 input_ids 包含所有输入 token，可以根据 past_length 舍弃部分 input_ids
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则 (past_length >= input_ids.shape[1])，假设 input_ids 仅包含未处理的 token

            # 如果将超过最大缓存长度，需要裁剪输入 attention_mask
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        # 获取 kwargs 中的 position_ids，如果 attention_mask 不为空且 position_ids 为空，则动态生成 position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入 inputs_embeds，则仅在第一代生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典，包括 position_ids、past_key_values、use_cache 和 attention_mask 等
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    # 定义一个函数 _reorder_cache，用于重新排序缓存中的过去键值对
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空的重新排序后的过去键值对元组
        reordered_past = ()
        # 遍历过去键值对列表中的每一层
        for layer_past in past_key_values:
            # 对每一层的过去状态，按照给定的 beam_idx 重新排序，并转移到相同的设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对元组
        return reordered_past
"""
Qwen2模型变换器，顶部带有序列分类头（线性层）。

[`Qwen2ForSequenceClassification`] 使用最后一个标记进行分类，类似其他因果模型（例如GPT-2）的做法。

由于它在最后一个标记上进行分类，因此需要知道最后一个标记的位置。如果配置中定义了 `pad_token_id`，它会找到每行中不是填充标记的最后一个标记。如果没有定义 `pad_token_id`，则简单地取每个批次行中的最后一个值。当传递 `inputs_embeds` 而不是 `input_ids` 时，由于无法猜测填充标记，它也会采取同样的做法（取每个批次行中的最后一个值）。
"""
@add_start_docstrings(
    """
    `forward` 方法用于执行前向传播。

    参数:
    - `input_ids` (torch.LongTensor, optional): 输入的token ID序列.
    - `attention_mask` (torch.Tensor, optional): 注意力遮罩，指示哪些元素是填充的.
    - `position_ids` (torch.LongTensor, optional): 指示每个token的位置ID.
    - `past_key_values` (List[torch.FloatTensor], optional): 过去的键值对，用于缓存计算.
    - `inputs_embeds` (torch.FloatTensor, optional): 替代 `input_ids` 的嵌入表示.
    - `labels` (torch.LongTensor, optional): 分类标签.
    - `use_cache` (bool, optional): 是否使用缓存.
    - `output_attentions` (bool, optional): 是否输出注意力权重.
    - `output_hidden_states` (bool, optional): 是否输出隐藏状态.
    - `return_dict` (bool, optional): 是否返回字典格式的输出.

    返回:
    - 根据模型配置返回不同的输出，包括分类结果、注意力权重等.
    """,
    QWEN2_FORWARD_DOCSTRING,
)
class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        返回模型中的输入嵌入层。
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        设置模型的输入嵌入层。
        """
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Qwen2模型的前向传播方法。
        """
```