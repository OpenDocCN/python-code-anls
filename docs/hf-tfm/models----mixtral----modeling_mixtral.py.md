# `.\transformers\models\mixtral\modeling_mixtral.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 基于 EleutherAI 的 GPT-NeoX 库和此库中的 OPT 实现对代码进行修改以适应 GPT-NeoX 和 Meta AI 团队训练模型的轻微架构差异
# 根据 Apache 许可证版本 2.0 进行许可
# 只有在遵守许可证的情况下才能使用此文件。您可以在以下位置获取许可证的副本
# 如果不是适用法律法规或已达成书面协议，则按“原样”分发软件
# 不提供任何明示或暗示的担保或条件。请参考许可证，了解特定语言下的许可条件和限制
""" PyTorch Mixtral model."""

# 导入所需的模块
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torch_fx_available
from .configuration_mixtral import MixtralConfig

# 判断是否引入了flash_attn版本2
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    # 判断是否flash_attn_func支持window_size参数
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# 在FX图中，该函数是一个叶子函数。
# 这意味着该功能不会被追踪，仅会出现在图中作为一个节点。
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    # 包装_prepare_4d_causal_attention_mask函数，使其成为FX图中的叶子节点
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


# 获取记录器对象
logger = logging.get_logger(__name__)

# 用于文档的配置字符串
_CONFIG_FOR_DOC = "MixtralConfig"

# 定义辅助的负载平衡损失函数
def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    # 这个函数实现了 Switch Transformer 论文中提出的损失函数(方程 (4) - (6))
    # 它的目的是惩罚专家之间路由过于不均衡的情况
    def compute_aux_loss(gate_logits, num_experts):
        # 如果 gate_logits 为 None 或不是元组，则返回 0 作为损失
        if gate_logits is None or not isinstance(gate_logits, tuple):
            return 0
        
        # 如果 gate_logits 是元组,则将其连接为一个张量
        if isinstance(gate_logits, tuple):
            compute_device = gate_logits[0].device
            concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
        
        # 对连接后的 gate_logits 应用 softmax 得到路由权重
        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
        
        # 选择前 top_k 个专家
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        
        # 创建一个 one-hot 编码的专家掩码张量
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
        
        # 计算每个专家被路由的token比例
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        
        # 计算路由到这些专家的平均概率
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
        
        # 计算最终的辅助损失
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss * num_experts
# 从transformers.models.llama.modeling_llama._get_unpad_data复制而来
def _get_unpad_data(attention_mask):
    # 计算每个样本中实际的序列长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 获取attention_mask中为1的位置索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批中的最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积的序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 从transformers.models.llama.modeling_llama.LlamaRMSNorm复制而来，将Llama替换为Mixtral
class MixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MixtralRMSNorm等同于T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 记录输入的数据类型
        input_dtype = hidden_states.dtype
        # 将隐藏状态转换为32位浮点数
        hidden_states = hidden_states.to(torch.float32)
        # 计算方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 根据方差进行归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# 从transformers.models.llama.modeling_llama.LlamaRotaryEmbedding复制而来，将Llama替换为Mixtral
class MixtralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了使torch.jit.trace工作，在这里构建
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 与论文中不同，但使用不同的置换以获得相同的计算
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# 从transformers.models.llama.modeling_llama.rotate_half复制而来
def rotate_half(x):
    # 旋转输入数据的一半隐藏维度
    """Rotates half the hidden dims of the input."""
    # 将输入张量按照最后一个维度的一半分割成两部分，前半部分为 x1
    x1 = x[..., : x.shape[-1] // 2]
    # 后半部分为 x2
    x2 = x[..., x.shape[-1] // 2 :]
    # 将 x2 取负值，然后与 x1 进行拼接，形成新的张量
    return torch.cat((-x2, x1), dim=-1)
# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# 应用旋转位置嵌入到查询和键张量

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`): The position indices of the tokens corresponding to the query and key tensors.
        unsqueeze_dim (`int`, *optional*, defaults to 1): The dimension along which to unsqueeze cos[position_ids] and sin[position_ids]
            so that they can be properly broadcasted to the dimensions of q and k.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # 在指定维度上对 cos[position_ids] 进行扩展以匹配 q 和 k 的维度
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # 在指定维度上对 sin[position_ids] 进行扩展以匹配 q 和 k 的维度
    q_embed = (q * cos) + (rotate_half(q) * sin)     # 用旋转位置嵌入对查询张量进行旋转
    k_embed = (k * cos) + (rotate_half(k) * sin)     # 用旋转位置嵌入对键张量进行旋转
    return q_embed, k_embed  # 返回旋转后的查询和键张量


# Copied from transformers.models.llama.modeling_llama.repeat_kv
# 重复键值张量

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)  # 返回重复后的键值张量


# Copied from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Mixtral
# 多头注意力模型

class MixtralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    # 初始化方法，接受一个 MixtralConfig 类型的参数 config 和一个可选的整数类型的参数 layer_idx
    def __init__(self, config: MixtralConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 config 参数赋值给 self.config
        self.config = config
        # 将传入的 layer_idx 参数赋值给 self.layer_idx
        self.layer_idx = layer_idx
        # 如果 layer_idx 为 None，则输出警告信息
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 设置隐藏层大小为 config 中的 hidden_size
        self.hidden_size = config.hidden_size
        # 设置注意力头数为 config 中的 num_attention_heads
        self.num_heads = config.num_attention_heads
        # 设置每个注意力头的维度为隐藏层大小除以注意力头数
        self.head_dim = self.hidden_size // self.num_heads
        # 设置键值头数为 config 中的 num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        # 设置键值组数为注意力头数除以键值头数
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # 设置最大位置嵌入长度为 config 中的 max_position_embeddings
        self.max_position_embeddings = config.max_position_embeddings
        # 设置绳索参数为 config 中的 rope_theta
        self.rope_theta = config.rope_theta
        # 设置是否因果为 True
        self.is_causal = True
        # 设置注意力丢弃率为 config 中的 attention_dropout
        self.attention_dropout = config.attention_dropout

        # 如果隐藏层大小不能被注意力头数整除，则抛出异常
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 使用 nn.Linear 创建 Q、K、V、O 四个投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # 使用 MixtralRotaryEmbedding 创建旋转��入
        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    # 重新塑形张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
# 从 transformers.models.mistral.modeling_mistral.MistralFlashAttention2 复制而来，修改 Mistral 为 Mixtral
class MixtralFlashAttention2(MixtralAttention):
    """
    Mixtral 闪烁注意力模块。该模块继承自 `MixtralAttention`，因为模块的权重保持不变。唯一需要更改的是前向传播，在这里需要正确调用闪烁注意力的公共 API，并处理输入中可能包含的填充标记。
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: 一旦 RoCm 的 Flash Attention 升级到 2.1，则应该删除此处的内容。
        # flash_attn<2.1 生成左上角对齐的因果掩码，而这里需要的是默认情况下为右下角对齐，这在 flash_attn>=2.1 中已经成为默认值。此属性用于处理这种差异。参考: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # 请注意，使用 flash_attn<2.1，当 q_seqlen != k_seqlen（除非 q_seqlen == 1 的特殊情况），会产生错误的掩码（左上角）。
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
    # 用于处理输入数据，进行对齐和剔除 padding，并返回处理后的数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取输入数据的维度信息
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 在第一次迭代中，需要正确重新创建填充mask，通过对其进行适当的切片
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        # 获取去除padding后的数据索引、去除padding后的序列长度、批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        # 基于去除padding后的索引，由于重新确定了维度，需要重新索引key_layer和value_layer
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            # 如果query_length等于kv_seq_len，则也需要根据索引重新调整query_layer
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果query_length等于1，则需要处理逻辑来重新定义query_layer、索引indices_q和cu_seqlens_q
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个内存拷贝，非常糟糕。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 对于其它情形，需要对attention_mask进行处理，调用unpad_input函数来处理query_layer和获取相应的索引信息
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的query_layer、key_layer、value_layer、indices_q、cu_seqlens信息、max_seqlen_in_batch信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 从 transformers.models.llama.modeling_llama.LlamaSdpaAttention 复制代码并将 Llama 替换为 Mixtral
class MixtralSdpaAttention(MixtralAttention):
    """
    使用 torch.nn.functional.scaled_dot_product_attention 的 Mixtral 注意力模块。该模块继承自 `MixtralAttention`，因为模块的权重保持不变。唯一的更改在于前向传播，以适应 SDPA API。
    """

    # 从 MixtralAttention.forward 进行改编
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
MIXTRAL_ATTENTION_CLASSES = {
    "eager": MixtralAttention,
    "flash_attention_2": MixtralFlashAttention2,
    "sdpa": MixtralSdpaAttention,
}


class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        # 第一个全连接层，输入维度为隐藏层维度，输出维度为前馈神经网络维度，无偏置
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        # 第二个全连接层，输入维度为前馈神经网络维度，输出维度为隐藏层维度，无偏置
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        # 第三个全连接层，输入维度为隐藏层维度，输出维度为前馈神经网络维度，无偏置
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        # 激活函数，根据配置选择对应的激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # 计算前馈神经网络的输出
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
    """
    该实现与标准的 MoE 完全等效（没有丢弃的令牌）。它更快，因为它将 MoE 操作阐述为块稀疏操作，以适应将令牌分配给专家的不平衡情况，而标准 MoE 要么
    (1) 以性能降低为代价丢弃令牌，要么 (2) 将容量因子设置为专家数，因此在填充上浪费计算和内存。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # 门控模块，输入维度为隐藏层维度，输出维度为本地专家数量，无偏置
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # 专家模块列表，包含 num_experts 个 MixtralBLockSparseTop2MLP 模块
        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)])
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """定义了模型的前向传播方法，接受隐藏状态张量作为输入，返回处理后的隐藏状态张量及路由器输出张量"""
        # 获取隐藏状态张量的形状信息：批大小、序列长度、隐藏维度
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 将隐藏状态张量重塑为二维张量，形状为（批大小 * 序列长度，隐藏维度）
        hidden_states = hidden_states.view(-1, hidden_dim)
        # 使用门控层计算路由器输出张量
        router_logits = self.gate(hidden_states)

        # 对路由器输出进行 softmax 操作，计算路由权重，dim=1 表示对每个样本的输出进行 softmax
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # 从路由权重中选取前 k 个最大值及其对应的索引，dim=-1 表示在最后一个维度上进行操作
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # 归一化路由权重，保证其和为 1
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # 将路由权重转换回与隐藏状态张量相同的数据类型
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 创建一个与隐藏状态张量形状相同的全零张量，用于存储最终的隐藏状态
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 使用 one-hot 编码创建专家掩码，用于索引将被调用的专家
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # 遍历模型中的所有可用专家，并在每个专家上执行计算
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # 在 torch 中，使用列表进行索引比使用 torch 张量更快
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # 索引正确的隐藏状态并计算当前专家的隐藏状态
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # 但是 `index_add_` 只支持使用 torch 张量进行索引，因此这里使用 `top_x` 张量
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # 将最终隐藏状态重塑为原来的形状（批大小、序列长度、隐藏维度）
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # 返回最终隐藏状态及路由器输出张量
        return final_hidden_states, router_logits
# 定义一个混合解码器层，继承自 nn.Module
class MixtralDecoderLayer(nn.Module):
    # 构造函数，接收配置和层的索引
    def __init__(self, config: MixtralConfig, layer_idx: int):
        # 调用父类的构造函数
        super().__init__()
        # 设置隐藏层大小，来自配置
        self.hidden_size = config.hidden_size

        # 初始化自注意力层，根据配置中的注意力实现类型
        self.self_attn = MIXTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # 初始化稀疏多专家（Mixture of Experts, MoE）块
        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        # 输入层前的归一化层
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 注意力层后的归一化层
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor, # 输入的隐藏状态
        attention_mask: Optional[torch.Tensor] = None, # 可选的注意力掩码
        position_ids: Optional[torch.LongTensor] = None, # 可选的位置标识
        past_key_value: Optional[Tuple[torch.Tensor]] = None, # 可选的前一个键值对
        output_attentions: Optional[bool] = False, # 是否输出注意力权重
        output_router_logits: Optional[bool] = False, # 是否输出路由器逻辑
        use_cache: Optional[bool] = False, # 是否使用缓存
        **kwargs, # 捕获其他关键字参数
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 如果在参数中存在"padding_mask"，则发出警告
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的“attentions”。
            output_router_logits (`bool`, *optional*):
                是否返回所有路由器的logits。这对于计算路由器损失很有用，在推断期间不应返回。
            use_cache (`bool`, *optional*):
                如果设置为`True`，则返回`past_key_values`键值状态，并且可以用来加速解码（查看`past_key_values`）。
        """

        # 保存输入hidden_states到residual
        residual = hidden_states

        # 对输入hidden_states进行LayerNorm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 添加残差连接
        hidden_states = residual + hidden_states

        # Fully Connected
        # 保存当前hidden_states到residual
        residual = hidden_states
        # 对hidden_states进行LayerNorm
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 使用block_sparse_moe模块对hidden_states进行处理
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 构建输出
        outputs = (hidden_states,)

        # 如果需要输出attentions，添加self_attn_weights
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用cache，添加present_key_value
        if use_cache:
            outputs += (present_key_value,)

        # 如果需要输出router的logits，添加router_logits
        if output_router_logits:
            outputs += (router_logits,)

        # 返回outputs
        return outputs
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

# 添加 Mixtral Model 的文档字符串
@add_start_docstrings(
    "The bare Mixtral Model outputting raw hidden-states without any specific head on top.",
    MIXTRAL_START_DOCSTRING,
)
# 从MistralPreTrainedModel类中复制到MixtralPreTrainedModel类
# Mistral->Mixtral
class MixtralPreTrainedModel(PreTrainedModel):
    config_class = MixtralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MixtralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    # 初始化模型权重
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


MIXTRAL_INPUTS_DOCSTRING = r"""
"""

# 添加 Mixtral Model 输入的文档字符串
@add_start_docstrings(
    "The bare Mixtral Model outputting raw hidden-states without any specific head on top.",
    MIXTRAL_START_DOCSTRING,
)
# 从MistralModel类中复制到MixtralModel类
# Mistral->Mixtral,Mistral->Mixtral
class MixtralModel(MixtralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]

    Args:
        config: MixtralConfig
    """
    # 初始化 MixtralDecoder 类
    def __init__(self, config: MixtralConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置填充 token 的 ID
        self.padding_idx = config.pad_token_id
        # 设置词表大小
        self.vocab_size = config.vocab_size
    
        # 创建嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 创建多个 MixtralDecoderLayer 组成的列表
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 设置注意力机制的实现方式
        self._attn_implementation = config._attn_implementation
        # 创建 MixtralRMSNorm 层
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
        # 是否开启梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并执行最终处理
        self.post_init()
    
    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens
    
    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    # 前向传播方法
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
    ):
        # 该方法的功能是执行 MixtralDecoder 的前向传播计算
# 这是一个 MixtralForCausalLM 类，继承自 MixtralPreTrainedModel
class MixtralForCausalLM(MixtralPreTrainedModel):
    # 定义一个需要绑定权重的键列表
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化 MixtralModel 实例
        self.model = MixtralModel(config)
        # 获取词表大小
        self.vocab_size = config.vocab_size
        # 定义一个线性层用于语言模型头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 获取 Router 辅助损失系数
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # 获取本地专家的数量
        self.num_experts = config.num_local_experts
        # 获取每个token可使用的专家数量
        self.num_experts_per_tok = config.num_experts_per_tok
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入embedding
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入embedding
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 获取输出embedding
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出embedding
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model

    # 定义前向传播方法
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
        # 该方法的具体实现在此处
        pass

    # 定义准备输入的方法
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 该方法的具体实现在此处
        pass
        ):  
            # 如果存在 past_key_values，则省略已处理过的标记
            if past_key_values is not None:
                # 如果 past_key_values 是 Cache 类型，则获取长度和已看到的标记数
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                    max_cache_length = past_key_values.get_max_length()
                else:
                    # 否则，默认长度和已看到的标记数为第一项的第三维长度
                    cache_length = past_length = past_key_values[0][0].shape[2]
                    max_cache_length = None

                # 保留未处理的标记:
                # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，则说明部分输入完全作为缓存的一部分进行传递
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                # 2 - 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有输入标记。我们可以根据 past_length 丢弃 input_ids。
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                # 3 - 否则（past_length >= input_ids.shape[1]），假设 input_ids 仅包含未处理的标记。

                # 如果我们即将超出最大缓存长度，则需要裁剪输入注意力掩码。
                if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
                ):
                    attention_mask = attention_mask[:, -max_cache_length:]

            position_ids = kwargs.get("position_ids", None)
            if attention_mask is not None and position_ids is None:
                # 为批量生成动态创建 position_ids
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            # 如果传递了 `inputs_embeds`，则只在第一代步中使用它们
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                }
            )
            return model_inputs

        @staticmethod
    # 重新排序缓存中的过去键值对
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空元组，用于存储重新排序后的过去状态
        reordered_past = ()
        # 遍历每个层级的过去状态
        for layer_past in past_key_values:
            # 对于每个层级的过去状态，按照给定的beam_idx重新排序，并将重新排序后的结果组成元组添加到reordered_past中
            reordered_past += (
                # 通过索引选择张量中的特定项，并转移到与past_state相同的设备上
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去状态元组
        return reordered_past
# 添加模型文档字符串，描述Mixtral模型在顺序分类任务上的使用情况
# 该模型使用最后一个标记进行分类，类似于其他因果模型（例如GPT-2）的做法
# 如果配置中定义了'pad_token_id'，则找到每一行中不是填充标记的最后一个标记进行分类
# 如果没有定义'pad_token_id'，则直接取每一行中的最后一个值作为分类标记
# 当传入'inputs_embeds'而非'input_ids'时，无法猜测填充标记，因此也取每一行中的最后一个值作为分类标记
"""
The Mixtral Model transformer with a sequence classification head on top (linear layer).
[`MixtralForSequenceClassification`] uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do.
Since it does classification on the last token, it requires to know the position of the last token. If a
`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
each row of the batch).
""",
MIXTRAL_START_DOCSTRING,
)

# 从transformers库中引入MixtralForSequenceClassification类，并进行文本替换
# 该类继承自MixtralPreTrainedModel类
class MixtralForSequenceClassification(MixtralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MixtralModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 重写forward方法，实现模型的前向传播过程
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
```