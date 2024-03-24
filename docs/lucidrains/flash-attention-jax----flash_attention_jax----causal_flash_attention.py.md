# `.\lucidrains\flash-attention-jax\flash_attention_jax\causal_flash_attention.py`

```py
# 导入所需的库
import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit
from jax.numpy import einsum
from einops import rearrange

# 定义常量
EPSILON = 1e-10
MASK_VALUE = -1e10
Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024

# 定义 flash attention 函数
def _query_chunk_flash_attention(q_range_chunk, k_range, q, k, v):
    # 获取输入张量的形状信息
    q_len, k_len, bh, dim, v_dim = q.shape[0], *k.shape, v.shape[-1]
    scale = 1 / jnp.sqrt(dim)
    q_scaled  = q * scale

    # 定义内部函数用于处理数据块
    def chunk_scanner(carries, _):
        key_chunk_idx, out, row_sum, row_max = carries
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        # 切片获取 k 和 v 的数据块
        k_chunk = lax.dynamic_slice(k, (key_chunk_idx, 0, 0), slice_sizes=(k_chunk_sizes, bh, dim))
        v_chunk = lax.dynamic_slice(v, (key_chunk_idx, 0, 0), slice_sizes=(k_chunk_sizes, bh, v_dim))

        k_range_chunk = lax.dynamic_slice(k_range, (0, key_chunk_idx), slice_sizes=(1, k_chunk_sizes))

        # 创建因果 mask
        causal_mask = q_range_chunk < k_range_chunk

        # 计算注意力权重
        attn_weights = einsum('i ... d, j ... d -> i ... j', q_scaled, k_chunk)

        causal_mask = rearrange(causal_mask, 'i j -> i 1 j')
        attn_weights = jnp.where(causal_mask, MASK_VALUE, attn_weights)

        block_row_max = jnp.max(attn_weights, axis = -1, keepdims = True)

        exp_weights = jnp.exp(attn_weights - block_row_max)

        exp_weights = jnp.where(causal_mask, 0., exp_weights)

        block_row_sum = jnp.sum(exp_weights, axis = -1, keepdims = True) + EPSILON

        exp_values = einsum('i ... j, j ... d -> i ... d', exp_weights, v_chunk)

        new_row_max = jnp.maximum(block_row_max, row_max)

        exp_row_max_diff = jnp.exp(row_max - new_row_max)
        exp_block_row_max_diff = jnp.exp(block_row_max - new_row_max)

        new_row_sum = exp_row_max_diff * row_sum + exp_block_row_max_diff * block_row_sum

        out = (row_sum / new_row_sum) * exp_row_max_diff * out + \
              (exp_block_row_max_diff / new_row_sum) * exp_values

        return (key_chunk_idx + k_chunk_sizes, out, new_row_sum, new_row_max), None

    # 初始化输出张量
    out = jnp.zeros((q_len, bh, dim))
    row_sum = jnp.zeros((q_len, bh, 1))
    row_max = jnp.ones((q_len, bh, 1)) * -1e6

    # 扫描数据块并处理
    (_, out, row_sum, row_max), _ = lax.scan(chunk_scanner, init = (0, out, row_sum, row_max), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    out = out.reshape(q_len, bh, v_dim)
    row_sum = row_sum.reshape(q_len, bh)
    row_max = row_max.reshape(q_len, bh)

    return out, row_sum, row_max

# 定义因果 flash attention 函数
def _causal_flash_attention(q, k, v):
    batch, heads, q_len, dim, k_len, v_dim = *q.shape, *v.shape[-2:]

    bh = batch * heads

    q, k, v = map(lambda t: rearrange(t, 'b h n d -> n (b h) d'), (q, k, v))

    q_range = jnp.arange(q_len).reshape(q_len, 1) + (k_len - q_len)
    k_range = jnp.arange(k_len).reshape(1, k_len)

    # 定义内部函数用于处理数据块
    def chunk_scanner(chunk_idx, _):
        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, dim))
        q_range_chunk = lax.dynamic_slice(q_range, (chunk_idx, 0), slice_sizes = (chunk_sizes, 1))

        return (chunk_idx + chunk_sizes, _query_chunk_flash_attention(q_range_chunk, k_range, q_chunk, k, v))

    _, (out, row_sum, row_max) = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    out = out.reshape(q_len, bh, v_dim)
    row_sum = row_sum.reshape(q_len, bh)
    row_max = row_max.reshape(q_len, bh)

    out = rearrange(out, 'n (b h) d -> b h n d', b = batch)
    return out, (row_sum, row_max)

# 定义自定义 VJP 和 JIT 编译的因果 flash attention 函数
@custom_vjp
@jit
def causal_flash_attention(q, k, v):
  out, _ = _causal_flash_attention(q, k, v)
  return out

# JIT 编译的 flash attention 前向传播函数
@jit
def flash_attention_forward(q, k, v):
    out, (row_sum, row_max) = _causal_flash_attention(q, k, v)
    return out, (q, k, v, out, row_sum, row_max)

# 定义用于反向传播的内部函数
def _query_chunk_flash_attention_backward(query_range_chunk, key_range, q, k, v, o, do, l, m):
    q_len, bh, dim, k_len, _, v_dim = *q.shape, *v.shape
    # 计算缩放因子，用于缩放查询向量
    scale = 1 / jnp.sqrt(dim)
    # 对查询向量进行缩放
    q_scaled = q * scale

    # 定义一个函数，用于处理每个块的计算
    def chunk_scanner(carries, _):
        key_chunk_idx, dq = carries
        # 确定当前块的大小
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        # 从键和值中提取当前块的数据
        k_chunk = lax.dynamic_slice(k, (key_chunk_idx, 0, 0), slice_sizes=(k_chunk_sizes, bh, dim))
        v_chunk = lax.dynamic_slice(v, (key_chunk_idx, 0, 0), slice_sizes=(k_chunk_sizes, bh, v_dim))

        # 从键范围中提取当前块的数据
        key_range_chunk = lax.dynamic_slice(key_range, (0, key_chunk_idx), slice_sizes=(1, k_chunk_sizes))

        # 创建因果掩码，用于屏蔽未来信息
        causal_mask = query_range_chunk < key_range_chunk

        # 计算注意力权重
        attn_weights = einsum('i ... d, j ... d -> i ... j', q_scaled, k_chunk)

        # 将因果掩码应用到注意力权重中
        causal_mask = rearrange(causal_mask, 'i j -> i 1 j')
        attn_weights = jnp.where(causal_mask, MASK_VALUE, attn_weights)

        # 计算指数化的注意力权重
        exp_attn_weights = jnp.exp(attn_weights - m)

        # 将因果掩码应用到指数化的注意力权重中
        exp_attn_weights = jnp.where(causal_mask, 0., exp_attn_weights)

        # 计算归一化的注意力权重
        p = exp_attn_weights / l

        # 计算值向量的加权和
        dv_chunk = einsum('i ... j, i ... d -> j ... d', p, do)
        dp = einsum('i ... d, j ... d -> i ... j', do, v_chunk)

        # 计算 D 和 ds
        D = jnp.sum(do * o, axis = -1, keepdims = True)
        ds = p * scale * (dp - D)

        # 计算查询向量的梯度
        dq_chunk = einsum('i ... j, j ... d -> i ... d', ds, k_chunk)
        dk_chunk = einsum('i ... j, i ... d -> j ... d', ds, q)

        return (key_chunk_idx + k_chunk_sizes, dq + dq_chunk), (dk_chunk, dv_chunk)

    # 初始化查询向量的梯度
    dq = jnp.zeros_like(q)

    # 执行块扫描操作，计算查询向量、键向量和值向量的梯度
    (_, dq), (dk, dv) = lax.scan(chunk_scanner, init = (0, dq), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    # 重塑查询向量、键向量和值向量的梯度
    dq = dq.reshape(q_len, bh, dim)
    dk = dk.reshape(k_len, bh, v_dim)
    dv = dv.reshape(k_len, bh, v_dim)

    # 返回查询向量、键向量和值向量的梯度
    return dq, dk, dv
# 使用 JIT 编译器对函数进行优化
@jit
# 定义反向传播函数 flash_attention_backward，接受 res 和 do 两个参数
def flash_attention_backward(res, do):
    # 解包 res 中的变量 q, k, v, o, l, m
    q, k, v, o, l, m = res

    # 获取 q, k, v 的形状信息
    batch, heads, q_len, dim, k_len, v_dim = *q.shape, *v.shape[-2:]

    # 计算 batch * heads
    bh = batch * heads

    # 重塑 m 和 l 的形状
    m = m.reshape(q_len, bh, 1)
    l = l.reshape(q_len, bh, 1)

    # 重塑 q, k, v, o, do 的形状
    q, k, v, o, do = map(lambda t: rearrange(t, 'b h n d -> n (b h) d'), (q, k, v, o, do))

    # 创建与 k 形状相同的全零数组 dk 和 dv
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    # 创建 q_len 的范围数组
    q_range = jnp.arange(q_len).reshape(q_len, 1) + (k_len - q_len)
    k_range = jnp.arange(k_len).reshape(1, k_len)

    # 定义 chunk_scanner 函数
    def chunk_scanner(carries, _):
        chunk_idx, dk, dv = carries

        # 计算 chunk_sizes
        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        # 切片获取 q_chunk 和 q_range_chunk
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, q.shape[-1]))
        q_range_chunk = lax.dynamic_slice(q_range, (chunk_idx, 0), slice_sizes = (chunk_sizes, 1))

        # 切片获取 m_chunk, l_chunk, o_chunk, do_chunk
        m_chunk = lax.dynamic_slice(m, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, 1))
        l_chunk = lax.dynamic_slice(l, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, 1))
        o_chunk = lax.dynamic_slice(o, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, o.shape[-1]))
        do_chunk = lax.dynamic_slice(do, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, do.shape[-1]))

        # 调用 _query_chunk_flash_attention_backward 函数处理 chunk 数据
        dq_chunk, dk_chunk, dv_chunk = _query_chunk_flash_attention_backward(q_range_chunk, k_range, q_chunk, k, v, o_chunk, do_chunk, l_chunk, m_chunk)
        return (chunk_idx + chunk_sizes, dk + dk_chunk, dv + dv_chunk), dq_chunk

    # 使用 lax.scan 函数对 chunk_scanner 进行迭代计算
    (_, dk, dv), dq = lax.scan(chunk_scanner, init = (0, dk, dv), xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    # 重塑 dq 的形状
    dq = dq.reshape(q_len, bh, dim)

    # 重塑 dq, dk, dv 的形状
    dq, dk, dv = map(lambda t: rearrange(t, 'n (b h) d -> b h n d', b = batch), (dq, dk, dv))

    # 返回 dq, dk, dv
    return dq, dk, dv

# 定义 causal_flash_attention 的导数函数
causal_flash_attention.defvjp(flash_attention_forward, flash_attention_backward)
```