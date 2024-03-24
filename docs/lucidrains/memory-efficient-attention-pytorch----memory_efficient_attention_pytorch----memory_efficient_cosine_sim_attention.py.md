# `.\lucidrains\memory-efficient-attention-pytorch\memory_efficient_attention_pytorch\memory_efficient_cosine_sim_attention.py`

```py
import math
import torch
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange

# helper functions

# 检查变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 对输入张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# regular attention

# 普通的注意力机制
def attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    **kwargs
):
    # 计算查询、键之间的相似度
    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    # 添加注意力偏置
    if exists(attn_bias):
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    # 处理掩码
    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, mask_value)

    # 处理因果关系
    if causal:
        i, j = sim.shape[-2:]
        mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

    # 计算注意力权重
    attn = sim.softmax(dim = -1)

    # 计算输出
    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out

# memory efficient attention

# 汇总查询、键、值的函数
def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[-2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    exp_weight = weight.exp()
    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value

# 使用 checkpoint 优化的汇总查询、键、值的函数
checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

# 数值不稳定的内存高效注意力机制
def numerically_unstable_memory_efficient_attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8
):
    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # 将所有输入分块

    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    if exists(attn_bias):
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim = -2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim = -1), attn_bias_chunks))

    # 循环遍历所有块并累积

    out = []
    # 遍历查询块列表，获取索引和查询块
    for q_index, q_chunk in enumerate(q_chunks):
        # 计算查询块的起始索引
        q_start_index = q_index * q_bucket_size
        # 初始化期望权重列表和加权值列表
        exp_weights = []
        weighted_values = []

        # 遍历键值块、值块和掩码块的元组列表，获取索引和对应的块
        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            # 计算键块的起始索引
            k_start_index = k_index * k_bucket_size

            # 如果是因果的且键块的起始索引大于查询块的起始索引加上查询块的长度减1，则跳过当前循环
            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                continue

            # 如果存在注意力偏置，则获取当前查询块和键块对应的注意力偏置
            attn_bias_chunk = attn_bias_chunks[q_index][k_index] if exists(attn_bias) else None

            # 调用summarize_qkv_fn函数，计算期望权重和加权值
            exp_weight_chunk, weighted_value_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                attn_bias_chunk,
                causal,
                (q_start_index, k_start_index)
            )

            # 将计算得到的期望权重和加权值添加到对应列表中
            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)

        # 计算所有加权值的总和
        all_values = sum(weighted_values)
        # 计算所有期望权重的总和
        all_weights = sum(exp_weights)

        # 对所有加权值进行归一化处理
        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        # 将归一化后的值添加到输出列表中
        out.append(normalized_values)

    # 沿着指定维度连接输出列表中的张量，形成最终输出结果
    return torch.cat(out, dim=-2)
# 主要类定义

class CosineSimAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False,
        memory_efficient = False,
        q_bucket_size = 512,
        k_bucket_size = 1024
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal

        inner_dim = heads * dim_head

        # 初始化缩放参数
        scale_init_value = -math.log(math.log2(seq_len ** 2 - seq_len))
        self.scale = nn.Parameter(torch.full((1, heads, 1, 1), scale_init_value))

        # 线性变换层，将输入维度映射到内部维度
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # 内存高效注意力相关参数
        # 可在前向传播中覆盖
        self.memory_efficient = memory_efficient
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        memory_efficient = None,
        q_bucket_size = None,
        k_bucket_size = None,
    ):
        memory_efficient = default(memory_efficient, self.memory_efficient)
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        h = self.heads
        context = default(context, x)

        # 对输入进行线性变换得到查询、键、值
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        # 重排维度以适应多头注意力计算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 对查询、键进行 L2 归一化
        q, k = map(l2norm, (q, k))

        # 缩放查询
        q = q * self.scale.exp()

        # 根据内存高效标志选择注意力函数
        attn_fn = attention if not memory_efficient else numerically_unstable_memory_efficient_attention

        # 计算注意力得到输出
        out = attn_fn(q, k, v, mask = mask, attn_bias = attn_bias, causal = self.causal, q_bucket_size = q_bucket_size, k_bucket_size = k_bucket_size)

        # 重排维度以还原原始形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```