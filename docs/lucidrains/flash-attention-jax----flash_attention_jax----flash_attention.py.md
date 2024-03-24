# `.\lucidrains\flash-attention-jax\flash_attention_jax\flash_attention.py`

```py
# 导入数学库和 JAX 库
import math
import jax
# 导入 partial 函数
from functools import partial
# 从 JAX 库中导入 nn、custom_vjp、numpy、lax、jit 模块
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit
# 从 JAX 的 numpy 模块中导入 einsum 函数
from jax.numpy import einsum
# 从 einops 库中导入 rearrange 函数

from einops import rearrange

# 常量定义

# 定义 EPSILON 常量
EPSILON = 1e-10
# 定义 MASK_VALUE 常量
MASK_VALUE = -1e10

# 定义 Q_CHUNK_SIZE 常量
Q_CHUNK_SIZE = 1024
# 定义 K_CHUNK_SIZE 常量
K_CHUNK_SIZE = 1024

# 闪电注意力

# 定义 _query_chunk_flash_attention 函数
def _query_chunk_flash_attention(chunk_idx, q, k, v, key_mask):
    # 获取 q 的长度、batch 大小、头数、维度、k 的长度和 v 的维度
    q_len, batch, heads, dim, k_len, v_dim = *q.shape, k.shape[0], v.shape[-1]
    # 计算缩放因子
    scale = 1 / jnp.sqrt(dim)
    # 对 q 进行缩放
    q_scaled = q * scale

    # 定义 chunk_scanner 函数
    def chunk_scanner(carries, _):
        chunk_idx, out, row_sum, row_max = carries
        # 计算 k_chunk_sizes
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        # 切片获取 k_chunk 和 v_chunk
        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0, 0, 0), slice_sizes=(k_chunk_sizes, batch, heads, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0, 0, 0), slice_sizes=(k_chunk_sizes, batch, heads, v_dim))
        key_mask_chunk = lax.dynamic_slice(key_mask, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, batch))

        # 计算注意力权重
        attn_weights = einsum('i ... d, j ... d -> i ... j', q_scaled, k_chunk)

        key_mask_chunk = rearrange(key_mask_chunk, 'j b -> 1 b 1 j')
        attn_weights = jnp.where(key_mask_chunk, attn_weights, MASK_VALUE)

        block_row_max = jnp.max(attn_weights, axis=-1, keepdims=True)

        new_row_max = jnp.maximum(block_row_max, row_max)
        exp_weights = jnp.exp(attn_weights - new_row_max)

        exp_weights = jnp.where(key_mask_chunk, exp_weights, 0.)
        block_row_sum = jnp.sum(exp_weights, axis=-1, keepdims=True) + EPSILON

        exp_values = einsum('i ... j, j ... d -> i ... d', exp_weights, v_chunk)

        exp_row_max_diff = jnp.exp(row_max - new_row_max)

        new_row_sum = exp_row_max_diff * row_sum + block_row_sum

        out = (row_sum / new_row_sum) * exp_row_max_diff * out + \
              (1. / new_row_sum) * exp_values

        return (chunk_idx + k_chunk_sizes, out, new_row_sum, new_row_max), None

    # 初始化 out、row_sum、row_max
    out = jnp.zeros((q_len, batch, heads, dim))
    row_sum = jnp.zeros((q_len, batch, heads, 1))
    row_max = jnp.ones((q_len, batch, heads, 1)) * -1e6

    # 扫描 chunk_scanner 函数
    (_, out, row_sum, row_max), _ = lax.scan(chunk_scanner, init=(0, out, row_sum, row_max), xs=None, length=math.ceil(k_len / K_CHUNK_SIZE))

    row_sum = rearrange(row_sum, 'n ... 1 -> n ...')
    row_max = rearrange(row_max, 'n ... 1 -> n ...')

    lse = jnp.log(row_sum) + row_max

    return out, lse

# 定义 _flash_attention 函数
def _flash_attention(q, k, v, key_mask):
    # 获取 batch、heads、q_len、dim、v_dim
    batch, heads, q_len, dim, v_dim = *q.shape, v.shape[-1]

    # 定义 chunk_scanner 函数
    def chunk_scanner(chunk_idx, _):
        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0, 0, 0), slice_sizes=(chunk_sizes, batch, heads, dim))

        return (chunk_idx + chunk_sizes, _query_chunk_flash_attention(chunk_idx, q_chunk, k, v, key_mask))

    # 重排 q、k、v 和 key_mask
    q, k, v = map(lambda t: rearrange(t, 'b h n d -> n b h d'), (q, k, v))
    key_mask = rearrange(key_mask, 'b j -> j b')

    _, (out, lse) = lax.scan(chunk_scanner, init=0, xs=None, length=math.ceil(q_len / Q_CHUNK_SIZE))

    out = rearrange(out, 'c n b h d -> b h (c n) d')
    lse = rearrange(lse, 'c n b h -> b h (c n)')

    return out, lse

# 定义 flash_attention 函数
@custom_vjp
@jit
def flash_attention(q, k, v, key_mask):
    out, _ = _flash_attention(q, k, v, key_mask)
    return out

# 定义 flash_attention_forward 函数
@jit
def flash_attention_forward(q, k, v, key_mask):
    out, lse = _flash_attention(q, k, v, key_mask)
    return out, (q, k, v, key_mask, out, lse)

# 定义 _query_chunk_flash_attention_backward 函数
def _query_chunk_flash_attention_backward(q, k, v, key_mask, o, do, lse):
    q_len, batch, heads, dim, k_len, v_dim = *q.shape, v.shape[0], v.shape[-1]

    scale = 1 / jnp.sqrt(dim)
    q_scaled = q * scale
    # 定义一个函数用于扫描数据块，处理注意力机制中的计算
    def chunk_scanner(carries, _):
        # 从参数中获取数据块索引和数据块
        chunk_idx, dq = carries
        # 计算数据块的大小，取最小值
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        # 从输入的k中切片出当前数据块的部分
        k_chunk = lax.dynamic_slice(k, (chunk_idx, batch, heads, 0), slice_sizes=(k_chunk_sizes, batch, heads, dim))
        # 从输入的v中切片出当前数据块的部分
        v_chunk = lax.dynamic_slice(v, (chunk_idx, batch, heads, 0), slice_sizes=(k_chunk_sizes, batch, heads, v_dim))
        # 从输入的key_mask中切片出当前数据块的部分
        key_mask_chunk = lax.dynamic_slice(key_mask, (chunk_idx, batch), slice_sizes=(k_chunk_sizes, batch))

        # 计算注意力权重
        attn_weights = einsum('i ... d, j ... d -> i ... j', q_scaled, k_chunk)

        # 计算softmax
        p = jnp.exp(attn_weights - lse)

        # 根据key_mask对softmax结果进行掩码处理
        key_mask_chunk = rearrange(key_mask_chunk, 'j b -> 1 b 1 j')
        p = jnp.where(key_mask_chunk, p, 0.)        

        # 计算值向量的加权和
        dv_chunk = einsum('i ... j, i ... d -> j ... d', p, do)
        # 计算梯度
        dp = einsum('i ... d, j ... d -> i ... j', do, v_chunk)

        # 计算D
        D = jnp.sum(do * o, axis = -1, keepdims = True)
        # 计算梯度
        ds = p * scale * (dp - D)

        # 计算查询向量的梯度
        dq_chunk = einsum('i ... j, j ... d -> i ... d', ds, k_chunk)
        # 计算键向量的梯度
        dk_chunk = einsum('i ... j, i ... d -> j ... d', ds, q)

        # 返回更新后的数据块索引和梯度
        return (chunk_idx + k_chunk_sizes, dq + dq_chunk), (dk_chunk, dv_chunk)

    # 初始化查询向量的梯度
    dq = jnp.zeros_like(q)

    # 使用scan函数对数据块进行扫描，计算梯度
    (_, dq), (dk, dv) = lax.scan(chunk_scanner, init = (0, dq), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    # 重组键向量的梯度
    dk = rearrange(dk, 'c n ... -> (c n) ...')
    # 重组值向量的梯度
    dv = rearrange(dv, 'c n ... -> (c n) ...')
    # 返回查询向量、键向量和值向量的梯度
    return dq, dk, dv
# 使用 JIT 编译器对函数进行即时编译，提高性能
@jit
# 定义反向传播函数，接收前向传播的结果和梯度
def flash_attention_backward(res, do):
    # 解包前向传播结果
    q, k, v, key_mask, o, lse = res

    # 获取输入张量的形状信息
    batch, heads, q_len, dim = q.shape

    # 重新排列张量的维度顺序
    lse = rearrange(lse, 'b h n -> n b h 1')

    q, k, v, o, do = map(lambda t: rearrange(t, 'b h n d -> n b h d'), (q, k, v, o, do))
    key_mask = rearrange(key_mask, 'b j -> j b')

    # 创建与 k 和 v 形状相同的零张量
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    # 定义用于扫描每个块的函数
    def chunk_scanner(carries, _):
        chunk_idx, dk, dv = carries

        # 定义每个块的大小
        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        # 切片获取每个块的输入张量
        q_chunk = lax.dynamic_slice(q, (chunk_idx, batch, heads, 0), slice_sizes = (chunk_sizes, batch, heads, q.shape[-1]))
        lse_chunk = lax.dynamic_slice(lse, (chunk_idx, batch, heads, 0), slice_sizes = (chunk_sizes, batch, heads, 1))
        o_chunk = lax.dynamic_slice(o, (chunk_idx, batch, heads, 0), slice_sizes = (chunk_sizes, batch, heads, o.shape[-1]))
        do_chunk = lax.dynamic_slice(do, (chunk_idx, batch, heads, 0), slice_sizes = (chunk_sizes, batch, heads, do.shape[-1]))

        # 调用子函数计算每个块的梯度
        dq_chunk, dk_chunk, dv_chunk = _query_chunk_flash_attention_backward(q_chunk, k, v, key_mask, o_chunk, do_chunk, lse_chunk)
        return (chunk_idx + chunk_sizes, dk + dk_chunk, dv + dv_chunk), dq_chunk

    # 使用 lax.scan 函数对每个块进行扫描
    (_, dk, dv), dq = lax.scan(chunk_scanner, init = (0, dk, dv), xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    # 重新排列梯度张量的维度顺序
    dq = rearrange(dq, 'c n b h d -> b h (c n) d')
    dk, dv = map(lambda t: rearrange(t, 'n b h d -> b h n d'), (dk, dv))

    # 返回计算得到的梯度
    return dq, dk, dv, None

# 将反向传播函数注册到前向传播函数上
flash_attention.defvjp(flash_attention_forward, flash_attention_backward)
```