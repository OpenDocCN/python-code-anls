# `.\models\mixtral\modeling_mixtral.py`

```py
# 设置编码格式为 UTF-8
# 版权声明和许可信息，基于 Apache License, Version 2.0
# 本代码基于 EleutherAI 的 GPT-NeoX 库，包括 GPT-NeoX 和 OPT 实现的修改，以适应与 Meta AI 团队训练的模型的架构差异。
# 导入 Python 标准库中的模块和函数
""" PyTorch Mixtral model."""

# 导入 inspect 模块，用于获取对象的信息
import inspect
# 导入 math 模块，提供数学函数
import math
# 导入 warnings 模块，用于发出警告信息
import warnings
# 导入类型提示相关模块
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 模块
import torch
# 导入 PyTorch 中的函数库和功能模块
import torch.nn.functional as F
import torch.utils.checkpoint
# 导入 PyTorch 中的 nn 模块
from torch import nn
# 导入 PyTorch 中的损失函数
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入混合精度训练相关模块
from ...activations import ACT2FN
# 导入缓存相关模块
from ...cache_utils import Cache, DynamicCache
# 导入模型中的注意力掩码工具函数
from ...modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
# 导入模型输出相关类
from ...modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
# 导入模型工具函数
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 实用工具函数
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
# 导入工具函数
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
# 导入导入相关工具函数
from ...utils.import_utils import is_torch_fx_available
# 导入 Mixtral 模型配置类
from .configuration_mixtral import MixtralConfig

# 检查是否支持 Flash Attention 2 版本，根据情况导入相应的模块和函数
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    # 检查 Flash Attention 函数是否支持窗口大小参数
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# 如果支持 Torch FX，将 _prepare_4d_causal_attention_mask 函数包装为 FX 图中的叶节点函数
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 文档配置信息
_CONFIG_FOR_DOC = "MixtralConfig"


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    """
    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    # 如果 gate_logits 为空或者不是元组，则返回 0
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    # 如果 gate_logits 是元组，则计算设备并将各层的门控 logits 拼接起来
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    # 计算 routing weights，即经过 softmax 处理后的权重
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    # 根据 routing weights 获取 top_k 个专家的索引
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    # 创建一个 one-hot 编码的专家 mask
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # 如果没有 attention_mask，则计算每个专家被路由到的 token 的百分比
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # 计算路由到每个专家的平均概率
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # 创建专家注意力 mask，用于处理 padding token
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # 计算每个专家被路由到的 token 的百分比，考虑了 attention_mask
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # 创建路由概率专家注意力 mask，用于处理 padding token
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # 计算路由到每个专家的平均概率，考虑了 attention_mask
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )
    # 计算每个专家的损失乘以路由器概率，并对所有专家求和得到总损失
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    # 将总损失乘以专家的数量，得到最终的整体损失
    return overall_loss * num_experts
# Copied from transformers.models.llama.modeling_llama._get_unpad_data
# 计算非填充数据的索引、累计序列长度和批次中最大序列长度
def _get_unpad_data(attention_mask):
    # 计算每个批次中的序列长度总和
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到所有非填充位置的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中最大的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累计序列长度并进行填充
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mixtral
# MixtralRMSNorm类，用于模仿T5LayerNorm，实现均值归一化
class MixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # 初始化权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 设置方差的小值 epsilon
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 获取输入的数据类型
        input_dtype = hidden_states.dtype
        # 将输入转换为 float32 类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算输入张量的方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 应用均值归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回加权后的归一化结果
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Mixtral
# MixtralRotaryEmbedding类，用于生成旋转嵌入矩阵，实现Self-Attention操作
class MixtralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 初始化维度、最大位置嵌入和基数
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率的倒数，用于生成正弦和余弦值
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # 将频率作为缓冲区注册，以便后续使用
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建旋转嵌入的正弦和余弦缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置缓存的最大序列长度
        self.max_seq_len_cached = seq_len
        # 生成等间距的整数张量
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算正弦和余弦值的缓存
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果当前序列长度超过缓存的最大序列长度，重新设置正弦和余弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回旋转嵌入的正弦和余弦值
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
# 实现输入张量的上下半部分交换
def rotate_half(x):
    # 对输入张量的一半隐藏维度进行旋转操作
    """Rotates half the hidden dims of the input."""
    
    # 将输入张量 x 的前半部分进行切片，保留其隐藏维度的前一半数据
    x1 = x[..., : x.shape[-1] // 2]
    
    # 将输入张量 x 的后半部分进行切片，保留其隐藏维度的后一半数据
    x2 = x[..., x.shape[-1] // 2 :]
    
    # 将 x 的后半部分取负值，并与 x 的前半部分连接在一起，以实现旋转操作
    return torch.cat((-x2, x1), dim=-1)
# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Unsqueezes cos and sin tensors along unsqueeze_dim to match dimensions of q and k
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # Apply rotary position embedding to q and k tensors
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # Extract dimensions from hidden_states tensor
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # If n_rep is 1, return the original hidden_states tensor
    if n_rep == 1:
        return hidden_states
    # Expand hidden_states tensor to repeat along the specified dimension
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # Reshape expanded tensor to the desired shape
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Mixtral
class MixtralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    # 初始化函数，接受配置参数和可选的层索引
    def __init__(self, config: MixtralConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 保存传入的配置参数
        self.config = config
        # 保存传入的层索引
        self.layer_idx = layer_idx
        # 如果未提供层索引，发出警告，因为在使用缓存时可能会导致前向调用中的错误
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 从配置中获取隐藏层大小
        self.hidden_size = config.hidden_size
        # 从配置中获取注意力头数
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_heads
        # 从配置中获取键值头数
        self.num_key_value_heads = config.num_key_value_heads
        # 计算每组键值头的数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # 从配置中获取最大位置嵌入数
        self.max_position_embeddings = config.max_position_embeddings
        # 从配置中获取旋转嵌入的基础值
        self.rope_theta = config.rope_theta
        # 设定是否是因果注意力
        self.is_causal = True
        # 从配置中获取注意力丢弃率
        self.attention_dropout = config.attention_dropout

        # 检查隐藏层大小是否能被注意力头数整除，否则抛出数值错误
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 初始化查询投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # 初始化键投影层
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 初始化值投影层
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 初始化输出投影层
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # 初始化旋转嵌入层
        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    # 根据给定的张量形状，调整其形状以适应注意力头数和头维度的结构
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，接收隐藏状态、注意力掩码、位置ID、过去的键值对缓存等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
# 从 `transformers.models.mistral.modeling_mistral.MistralFlashAttention2` 复制的 `MixtralFlashAttention2` 类，将 Mistral 更名为 Mixtral
class MixtralFlashAttention2(MixtralAttention):
    """
    Mixtral flash attention module. This module inherits from `MixtralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # 从 `transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__` 复制的构造函数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: 在 Flash Attention for RoCm 版本升级到 2.1 之后应该移除这段注释。
        # flash_attn<2.1 生成左上对齐的因果蒙版，而这里需要右下对齐的默认效果。此属性用于处理这种差异。参考：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0。
        # 注意，对于 flash_attn<2.1，除了 q_seqlen == 1 的情况外，使用 q_seqlen != k_seqlen 会产生错误的蒙版（左上对齐）。
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
        """
        Override of the forward method to integrate Mixtral flash attention with handling of padding tokens.
        """
        # 真正的前向传播方法，集成了 Mixtral flash attention 并处理填充标记
        pass

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
    # 定义一个方法 `_upad_input`，该方法接受多个输入参数：query_layer, key_layer, value_layer, attention_mask, query_length
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取 key_layer 的形状信息，分别为 batch_size, kv_seq_len, num_heads, head_dim
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 如果 kv_seq_len 不等于 attention_mask 的最后一个维度长度，需要重新创建 padding mask
        if kv_seq_len != attention_mask.shape[-1]:
            # 获取 attention_mask 的最后一个维度长度
            attention_mask_num_tokens = attention_mask.shape[-1]
            # 更新 attention_mask，保留 kv_seq_len 长度的部分
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        # 调用 _get_unpad_data 函数，获取解压后的数据 indices_k, cu_seqlens_k, max_seqlen_in_batch_k
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        # 通过索引操作，对 key_layer 进行重新组织，形状变为 (batch_size * kv_seq_len, num_heads, head_dim)
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        # 对 value_layer 进行类似的重新组织
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        # 根据 query_length 的不同情况进行不同的处理
        if query_length == kv_seq_len:
            # 如果 query_length 等于 kv_seq_len，则对 query_layer 进行索引操作
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果 query_length 等于 1，则将 query_layer 的形状调整，并生成相应的索引和长度信息
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个 memcpy 操作，非常不好。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，根据 -query_length: 切片假设左填充，更新 attention_mask，并调用 unpad_input 函数处理 query_layer
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的结果，包括 query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k)
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 从`transformers.models.mistral.modeling_mistral.MistralSdpaAttention`复制而来，将"Mistral"改为"Mixtral"
class MixtralSdpaAttention(MixtralAttention):
    """
    使用`torch.nn.functional.scaled_dot_product_attention`的Mixtral注意力模块。此模块继承自`MixtralAttention`，
    其权重保持不变。唯一的更改在于前向传递，以适应SDPA API。
    """

    # 从MixtralAttention.forward进行调整
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        pass  # 这里的实际实现在SDPA API中进行了调整，但在注释中未提供具体的实现细节

# 定义了Mixtral注意力类别的映射字典
MIXTRAL_ATTENTION_CLASSES = {
    "eager": MixtralAttention,
    "flash_attention_2": MixtralFlashAttention2,
    "sdpa": MixtralSdpaAttention,  # 将sdpa映射到MixtralSdpaAttention类
}


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        # 线性层定义
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        # 激活函数从ACT2FN字典中选择
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # 前向传递计算
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


# MixtralBLockSparseTop2MLP被废弃，用MixtralBlockSparseTop2MLP代替，发出一次警告
class MixtralBLockSparseTop2MLP(MixtralBlockSparseTop2MLP):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "MixtralBLockSparseTop2MLP is deprecated by MixtralBlockSparseTop2MLP and will be removed in v4.40."
        )
        super().__init__(*args, **kwargs)


class MixtralSparseMoeBlock(nn.Module):
    """
    这个实现严格等同于标准的MoE，具有全容量（没有丢弃标记的令牌）。它更快，因为它将MoE操作
    形式化为块稀疏操作，以适应对专家的不平衡分配，而标准MoE要么（1）丢弃标记，以降低性能，要么（2）
    将容量因子设置为专家数量，从而浪费填充的计算和内存。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        # gating机制的线性层
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # 创建MixtralBlockSparseTop2MLP模块列表，用于每个专家
        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        接收隐藏状态作为输入，返回处理后的隐藏状态和路由器的logits值。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态张量，形状为(batch_size, sequence_length, hidden_dim)

        Returns:
            torch.Tensor: 处理后的最终隐藏状态张量，形状为(batch_size, sequence_length, hidden_dim)
            torch.Tensor: 路由器的logits张量，形状为(batch * sequence_length, n_experts)
        """

        # 获取输入张量的维度信息
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # 将输入的三维张量重塑为二维张量，以便进行路由器的计算
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 使用路由器模型计算路由器的logits
        router_logits = self.gate(hidden_states)

        # 使用softmax函数对logits进行归一化处理，得到路由权重
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 从每个路由权重中选择top-k的值，并重新归一化
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # 将归一化后的路由权重转换为输入张量的数据类型
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 初始化一个全零张量，用于存储最终的隐藏状态
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 使用one-hot编码创建选定专家的专家掩码
        # 这将用于轻松地索引哪个专家将被调用
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # 遍历模型中所有可用的专家，并在每个专家上执行计算
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # 将top_x张量转换为Python列表，以便在PyTorch中更快地索引
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # 根据索引从隐藏状态中获取正确的隐藏状态，并计算当前专家的隐藏状态
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # 使用index_add_方法将当前专家的隐藏状态加到最终隐藏状态中
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # 将最终隐藏状态张量重塑回原始形状(batch_size, sequence_length, hidden_dim)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        # 返回最终的隐藏状态张量和路由器的logits张量
        return final_hidden_states, router_logits
# 定义 MixtralDecoderLayer 类，继承自 nn.Module，用于实现 Mixtral 模型的解码器层
class MixtralDecoderLayer(nn.Module):
    # 初始化方法，接受 MixtralConfig 和层索引作为参数
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        # 设置隐藏层大小
        self.hidden_size = config.hidden_size

        # 初始化自注意力机制，根据配置选择不同的注意力实现类进行初始化
        self.self_attn = MIXTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # 初始化块稀疏多路注意力模块
        self.block_sparse_moe = MixtralSparseMoeBlock(config)

        # 初始化输入层归一化模块
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 初始化注意力后归一化模块
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 前向传播方法，接受隐藏状态、注意力掩码、位置 ID、过去的键值对等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            # 如果传入了 `padding_mask` 参数，发出警告提示
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *可选*): 注意力掩码张量，形状为 `(batch, sequence_length)`，其中填充元素为0
            past_key_value (`Tuple(torch.FloatTensor)`, *可选*): 缓存的过去键值投影状态
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 了解更多细节。
            output_router_logits (`bool`, *可选*):
                是否返回所有路由器的logits。这对计算路由器损失很有用，在推理时不应返回。
            use_cache (`bool`, *可选*):
                如果设置为 `True`，则返回 `past_key_values` 键值状态，可用于加速解码 (参见 `past_key_values`).
        """

        residual = hidden_states

        # 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力层
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 全连接层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
# MIXTRAL_START_DOCSTRING 是一个多行原始字符串，用于描述 MixtralPreTrainedModel 类的文档字符串。
# 它包含了关于模型继承自 PreTrainedModel 的信息，以及如何使用 PyTorch 的说明和参数列表。
MIXTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MixtralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# add_start_docstrings 是一个装饰器，用于为 MixtralPreTrainedModel 类添加文档字符串。
# 第一个参数是描述该模型输出原始隐藏状态的概述性文本。
# 第二个参数是 MIXTRAL_START_DOCSTRING，用于详细描述该类的配置和参数信息。
@add_start_docstrings(
    "The bare Mixtral Model outputting raw hidden-states without any specific head on top.",
    MIXTRAL_START_DOCSTRING,
)
# MixtralPreTrainedModel 类继承自 PreTrainedModel，用于 Mixtral 模型的预训练和初始化。
class MixtralPreTrainedModel(PreTrainedModel):
    # 配置类，指定了 Mixtral 模型的配置信息。
    config_class = MixtralConfig
    # 基础模型的前缀，通常用于命名前缀。
    base_model_prefix = "model"
    # 是否支持梯度检查点。
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表。
    _no_split_modules = ["MixtralDecoderLayer"]
    # 跳过键的设备放置。
    _skip_keys_device_placement = "past_key_values"
    # 是否支持 Flash Attention 2。
    _supports_flash_attn_2 = True
    # 是否支持 SDPA（Scaled Dot-Product Attention）。
    _supports_sdpa = True
    # 是否支持缓存类。
    _supports_cache_class = True

    # 初始化权重的函数。
    def _init_weights(self, module):
        std = self.config.initializer_range
        # 如果是线性层，初始化权重和偏置。
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层，初始化权重。
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# MIXTRAL_INPUTS_DOCSTRING 是一个未填充的多行原始字符串，可能用于描述 MixtralModel 类的输入信息。
MIXTRAL_INPUTS_DOCSTRING = r"""
"""


# add_start_docstrings 是一个装饰器，用于为 MixtralModel 类添加文档字符串。
# 第一个参数是描述该模型输出原始隐藏状态的概述性文本。
# 第二个参数是 MIXTRAL_START_DOCSTRING，用于详细描述该类的配置和参数信息。
@add_start_docstrings(
    "The bare Mixtral Model outputting raw hidden-states without any specific head on top.",
    MIXTRAL_START_DOCSTRING,
)
# MixtralModel 类继承自 MixtralPreTrainedModel，代表了 Mixtral 模型的具体实现。
class MixtralModel(MixtralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]

    Args:
        config: MixtralConfig
    """
    # 初始化函数，接受一个 MixtralConfig 类型的参数 config
    def __init__(self, config: MixtralConfig):
        # 调用父类的初始化函数，传入 config 参数
        super().__init__(config)
        # 设置 padding_idx 属性为 config 的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置 vocab_size 属性为 config 的 vocab_size
        self.vocab_size = config.vocab_size

        # 创建一个嵌入层对象 embed_tokens，用于将输入的 token 转换为向量表示
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # 创建一个由多个 MixtralDecoderLayer 组成的层列表，每层通过不同的 layer_idx 构建
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 设置 _attn_implementation 属性为 config 的 _attn_implementation
        self._attn_implementation = config._attn_implementation
        
        # 创建一个 MixtralRMSNorm 对象 norm，用于进行归一化处理
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
        
        # 调用 post_init 函数，完成权重初始化和最终处理
        self.post_init()

    # 返回 embed_tokens 属性，即输入嵌入层对象
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置 embed_tokens 属性为 value
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 忽略复制操作，用于 forward 函数的装饰器
    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# MixtralForCausalLM 类，继承自 MixtralPreTrainedModel 类，用于混合专家模型的因果语言建模任务

class MixtralForCausalLM(MixtralPreTrainedModel):
    # 定义被绑定权重的键值，用于共享权重
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象 config
        super().__init__(config)
        # 初始化 MixtralModel 模型，根据传入的配置对象 config
        self.model = MixtralModel(config)
        # 设置词汇表大小为配置对象中的词汇表大小
        self.vocab_size = config.vocab_size
        # 初始化 lm_head，使用线性层将隐藏状态映射到词汇表大小，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 设置路由辅助损失系数为配置对象中的路由辅助损失系数
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # 设置本地专家的数量为配置对象中的本地专家数量
        self.num_experts = config.num_local_experts
        # 设置每个令牌的专家数量为配置对象中的每个令牌专家数量
        self.num_experts_per_tok = config.num_experts_per_tok
        # 调用后处理初始化方法，用于初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层，返回 MixtralModel 模型的嵌入 tokens
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入层的值
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 获取输出嵌入层，返回 lm_head 线性层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层的值
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器，用于设置 MixtralModel 模型的 decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取解码器，返回当前 MixtralModel 模型
    def get_decoder(self):
        return self.model

    # 前向传播函数，接受多种输入参数，返回 MoeCausalLMOutputWithPast 类型的输出
    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播函数，详细参数含义见上方修饰器的文档注释
        # 本函数无具体实现，仅用于说明接口，实际实现需在派生类中完成
        pass

    # 为生成准备输入的函数
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        **kwargs,
    ):
        # 为生成任务准备输入的函数，详细参数含义见上方函数签名
        # 本函数无具体实现，仅用于说明接口，实际实现需在派生类中完成
        pass
        # Omit tokens covered by past_key_values
        # 如果 past_key_values 不为空，则跳过已被处理的 token

        if past_key_values is not None:
            # Check if past_key_values is an instance of Cache
            # 检查 past_key_values 是否为 Cache 类的实例
            if isinstance(past_key_values, Cache):
                # Get sequence length from past_key_values
                # 从 past_key_values 中获取序列长度
                cache_length = past_key_values.get_seq_length()
                # Get seen tokens count from past_key_values
                # 从 past_key_values 中获取已看到的 token 数量
                past_length = past_key_values.seen_tokens
                # Get maximum cache length from past_key_values
                # 从 past_key_values 中获取最大缓存长度
                max_cache_length = past_key_values.get_max_length()
            else:
                # Assume past_key_values is a tuple and get dimensions from it
                # 假设 past_key_values 是一个元组，并从中获取维度信息
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 保留未处理的 token：

            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            # 如果 attention_mask 的长度超过 input_ids 的长度，则说明部分输入作为缓存的一部分传递（例如将 input_embeds 作为输入）

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            # 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有的输入 token。根据 past_length 可以丢弃 input_ids 的部分 token。

            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            # 否则（past_length >= input_ids.shape[1]），假设 input_ids 只包含未处理的 token。

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            # 如果即将超出最大缓存长度，我们需要裁剪输入的 attention mask。

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        # Get position_ids from kwargs if not provided
        # 如果 attention_mask 不为空且 position_ids 为空，则动态创建 position_ids 以进行批量生成

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # 如果传递了 `inputs_embeds`，我们只想在第一代步骤中使用它们

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Update model_inputs with various parameters
        # 使用各种参数更新 model_inputs

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
            }
        )
        # Return the constructed model_inputs dictionary
        # 返回构建的 model_inputs 字典

        return model_inputs
    # 定义一个函数 `_reorder_cache`，用于重新排序缓存数据
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空的元组，用于存储重新排序后的缓存数据
        reordered_past = ()
        # 遍历每层的缓存数据
        for layer_past in past_key_values:
            # 对每层的缓存数据进行重新排序，并将结果作为元组加入到 `reordered_past` 中
            reordered_past += (
                # 对每个 `past_state` 根据 `beam_idx` 进行索引选择，并放到对应设备上
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的缓存数据
        return reordered_past
"""
The Mixtral Model transformer with a sequence classification head on top (linear layer).

[`MixtralForSequenceClassification`] uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a
`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
each row of the batch).
"""
# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->Mixtral, LLAMA->MIXTRAL
class MixtralForSequenceClassification(MixtralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MixtralModel(config)  # 初始化 Mixtral 模型
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)  # 线性层用于分类得分

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens  # 返回输入嵌入的模型

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value  # 设置输入嵌入的模型

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
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
        Forward pass for MixtralForSequenceClassification.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Mask to avoid performing attention on padding tokens.
            position_ids (torch.LongTensor, optional): IDs to mark each token's position.
            past_key_values (List[torch.FloatTensor], optional): Cached key/value states for faster decoding.
            inputs_embeds (torch.FloatTensor, optional): Precomputed embeddings for the input tokens.
            labels (torch.LongTensor, optional): Labels for computing the sequence classification loss.
            use_cache (bool, optional): Whether or not to use cached key/value states.
            output_attentions (bool, optional): Whether or not to output attentions weights.
            output_hidden_states (bool, optional): Whether or not to output hidden states.
            return_dict (bool, optional): Whether or not to return a dictionary as the output.

        Returns:
            Depending on `return_dict`, either a model output dictionary or a tuple of logits and loss.

        Notes:
            This method defines how inputs are processed through the Mixtral model for sequence classification.
        """
        # 实现 MixtralForSequenceClassification 的前向传播
        # 具体实现根据参数的不同选择执行不同的操作，最终返回结果
        pass
```