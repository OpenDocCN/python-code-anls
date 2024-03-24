# `.\lucidrains\flash-attention-jax\flash_attention_jax\rabe_attention.py`

```py
# 导入数学库和部分函数
import math
from functools import partial

# 导入 JAX 库
import jax
from jax import lax, numpy as jnp, jit

# 定义常量
HIGHEST_PRECISION = jax.lax.Precision.HIGHEST

# 使用 partial 函数创建一个新的函数 einsum，指定精度为 HIGHEST_PRECISION
einsum = partial(jnp.einsum, precision = HIGHEST_PRECISION)

# 定义函数 _query_chunk_attention，实现分块注意力机制
def _query_chunk_attention(q, k, v, k_chunk_size = 4096):
    # 获取输入张量的维度信息
    q_len, k_len, dim, v_dim = q.shape[-2], *k.shape, v.shape[-1]

    # 确定 k_chunk_size 的大小
    k_chunk_size = min(k_chunk_size, k_len)
    # 对查询张量 q 进行缩放
    q = q / jnp.sqrt(dim)

    # 定义一个内部函数 summarize_chunk，用于计算每个块的注意力权重和值
    @partial(jax.checkpoint, prevent_cse = False)
    def summarize_chunk(q, k, v):
        # 计算注意力权重
        attn_weights = einsum('qd, kd -> qk', q, k)
        # 计算最大分数
        max_score = jnp.max(attn_weights, axis = -1, keepdims = True)
        max_score = jax.lax.stop_gradient(max_score)
        # 计算指数权重和值
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = einsum('vf, qv -> qf', v, exp_weights)
        return (exp_values, exp_weights.sum(axis = -1), max_score.reshape((q_len,)))

    # 定义一个函数 chunk_scanner，用于遍历块并计算注意力权重和值
    def chunk_scanner(chunk_idx):
        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0), slice_sizes=(k_chunk_size, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0), slice_sizes=(k_chunk_size, v_dim))
        return summarize_chunk(q, k_chunk, v_chunk)

    # 使用 map 函数并行处理所有块
    chunk_values, chunk_weights, chunk_max = jax.lax.map(chunk_scanner, xs = jnp.arange(0, k_len, k_chunk_size))
    global_max = jnp.max(chunk_max, axis = 0, keepdims = True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    # 汇总所有块的值和权重
    all_values = chunk_values.sum(axis = 0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis = 0)
    return all_values / all_weights

# 使用 JIT 编译函数 rabe_attention，实现基于块的自注意力机制
@jit
def rabe_attention(q, k, v, q_chunk_size = 1024, k_chunk_size = 4096):
    # 获取输入张量的维度信息
    q_len, dim, v_dim = *q.shape, v.shape[-1]

    # 定义函数 chunk_scanner，用于遍历查询张量的块并计算注意力权重和值
    def chunk_scanner(chunk_idx, _):
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (min(q_chunk_size, q_len), dim))
        return (chunk_idx + q_chunk_size, _query_chunk_attention(q_chunk, k, v, k_chunk_size = k_chunk_size))

    # 使用 scan 函数并行处理所有查询张量的块
    _, res = jax.lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / q_chunk_size))
    return res.reshape(q_len, v_dim)
```