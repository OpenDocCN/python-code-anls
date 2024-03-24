# `.\lucidrains\memory-efficient-attention-pytorch\memory_efficient_attention_pytorch\memory_efficient_attention.py`

```py
import torch
from functools import partial
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from einops import rearrange

# 导入所需的库

def exists(val):
    return val is not None

# 检查值是否存在的辅助函数

def default(val, d):
    return val if exists(val) else d

# 如果值存在则返回该值，否则返回默认值的辅助函数

# regular attention

def attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    **kwargs
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # 缩放查询向量

    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    # 计算注意力分数

    if exists(attn_bias):
        sim = sim + attn_bias

    # 添加注意力偏置

    mask_value = -torch.finfo(sim.dtype).max

    # 计算掩码值

    if exists(mask):
        if mask.ndim == 2:
            mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, mask_value)

    # 应用掩码

    if causal:
        i, j = sim.shape[-2:]
        mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

    # 应用因果掩码

    sim = sim - sim.amax(dim = -1, keepdim = True).detach()
    attn = sim.softmax(dim = -1)

    # 计算注意力权重

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out

    # 计算输出

# memory efficient attention

def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices, dropout):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[-2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    # 计算权重

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    # 添加注意力偏置

    mask_value = -torch.finfo(weight.dtype).max

    # 计算掩码值

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    # 应用掩码

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    # 应用因果掩码

    weight_max = weight.amax(dim = -1, keepdim = True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()

    exp_weight = F.dropout(exp_weight, p = dropout)

    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

# 创建检查点函数

def memory_efficient_attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8,
    dropout = 0.,
    training = False
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # 缩放查询向量

    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # 根据是否需要反向传播选择函数

    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    if exists(attn_bias):
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim = -2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim = -1), attn_bias_chunks))

    # 将输入分块

    out = []

    # 初始化输出列表
    # 遍历查询块列表，获取索引和查询块
    for q_index, q_chunk in enumerate(q_chunks):
        # 初始化空列表，用于存储期望权重、加权值和权重最大值
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        # 遍历键值块、值块和掩码块的元组列表
        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            # 计算查询块和键块的起始索引
            q_start_index = q_index * q_bucket_size
            k_start_index = k_index * k_bucket_size

            # 如果是因果的且键块的起始索引大于查询块的结束索引，则跳过当前循环
            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                continue

            # 如果存在注意力偏置，则获取当前注意力偏置块
            attn_bias_chunk = attn_bias_chunks[q_index][k_index] if exists(attn_bias) else None

            # 调用 summarize_qkv_fn 函数，计算期望权重、加权值和权重最大值
            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                attn_bias_chunk,
                causal,
                (q_start_index, k_start_index),
                dropout if training else 0.
            )

            # 将计算得到的结果添加到对应的列表中
            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        # 将权重最大值堆叠在一起
        weight_maxes = torch.stack(weight_maxes, dim=-1)

        # 将加权值堆叠在一起
        weighted_values = torch.stack(weighted_values, dim=-1)
        # 将期望权重堆叠在一起
        exp_weights = torch.stack(exp_weights, dim=-1)

        # 计算全局最大值
        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        # 计算重新归一化因子
        renorm_factor = (weight_maxes - global_max).exp().detach()

        # 期望权重乘以重新归一化因子
        exp_weights = exp_weights * renorm_factor
        # 加权值乘以重新排列的重新归一化因子
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        # 对所有加权值进行求和
        all_values = weighted_values.sum(dim=-1)
        # 对所有期望权重进行求和
        all_weights = exp_weights.sum(dim=-1)

        # 对归一化���的值进行计算
        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        # 将归一化后的值添加到输出列表中
        out.append(normalized_values)

    # 沿着指定维度连接输出列表中的张量
    return torch.cat(out, dim=-2)
# 主要的注意力机制类

class Attention(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 输入维度
        heads = 8,  # 头数，默认为8
        dim_head = 64,  # 每个头的维度，默认为64
        dropout = 0.,  # 丢弃概率，默认为0
        causal = False,  # 是否使用因果注意力，默认为False
        memory_efficient = False,  # 是否使用内存高效的注意力，默认为False
        q_bucket_size = 512,  # 查询桶大小，默认为512
        k_bucket_size = 1024  # 键值桶大小，默认为1024
    ):
        super().__init__()
        self.heads = heads  # 头数
        self.causal = causal  # 是否因果
        self.dropout = dropout  # 丢弃概率
        inner_dim = heads * dim_head  # 内部维度为头数乘以每个头的维度

        self.to_q = nn.Linear(dim, inner_dim, bias = False)  # 输入到查询的线性层
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)  # 输入到键值的线性层
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 输出的线性层

        # 内存高效注意力相关参数
        # 可在前向传播中覆盖
        self.memory_efficient = memory_efficient  # 是否内存高效
        self.q_bucket_size = q_bucket_size  # 查询桶大小
        self.k_bucket_size = k_bucket_size  # 键值桶大小

    # 前向传播函数
    def forward(
        self,
        x,  # 输入张量
        context = None,  # 上下文，默认为None
        mask = None,  # 掩码，默认为None
        attn_bias = None,  # 注意力偏置，默认为None
        memory_efficient = None,  # 是否内存高效，默认为None
        q_bucket_size = None,  # 查询桶大小，默认为None
        k_bucket_size = None,  # 键值桶大小，默认为None
    ):
        memory_efficient = default(memory_efficient, self.memory_efficient)  # 使用默认值或者自定义值
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)  # 使用默认值或者自定义值
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)  # 使用默认值或者自定义值

        h = self.heads  # 头数
        context = default(context, x)  # 上下文，默认为输入张量

        q = self.to_q(x)  # 查询张量
        k, v = self.to_kv(context).chunk(2, dim = -1)  # 键值张量拆分为k和v

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))  # 重排张量形状

        attn_fn = attention if not memory_efficient else memory_efficient_attention  # 根据内存高效性选择不同的注意力函数

        out = attn_fn(q, k, v, mask = mask, attn_bias = attn_bias, causal = self.causal, q_bucket_size = q_bucket_size, 
                    k_bucket_size = k_bucket_size, dropout = self.dropout, training = self.training)  # 注意力计算

        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排输出形状
        return self.to_out(out)  # 输出结果
```