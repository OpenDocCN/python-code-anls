# `.\lucidrains\flash-attention-jax\flash_attention_jax\cosine_sim_flash_attention.py`

```py
# 导入数学库和 JAX 库，以及部分函数
import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit

# 常量定义

EPSILON = 1e-10
MASK_VALUE = -1e10

Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024
COSINE_SIM_SCALE = 10 # 这可能需要是 log(序列长度) 的函数，但在我的测试中，16 对于 2048 和 4096 是足够的

# 闪电注意力

def _query_chunk_flash_attention(chunk_idx, q, k, v, key_mask):
    q_len, k_len, dim, v_dim = q.shape[-2], *k.shape, v.shape[-1]

    def chunk_scanner(carries, _):
        chunk_idx, out, row_sum = carries
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, v_dim))
        key_mask_chunk = lax.dynamic_slice(key_mask, (chunk_idx,), slice_sizes=(k_chunk_sizes,))

        attn_weights = (q @ k_chunk.transpose() * COSINE_SIM_SCALE) - COSINE_SIM_SCALE  # 这个输出范围为 [-2 * scale, 0]，行和现在受到键/值序列长度的限制 - 如果您希望定制归一化常数（在极端序列长度的情况下），也可以进一步移动这个值

        attn_weights = jnp.where(key_mask_chunk, attn_weights, MASK_VALUE)

        exp_weights = jnp.exp(attn_weights)
        exp_weights = jnp.where(key_mask_chunk, exp_weights, 0.)

        block_row_sum = jnp.sum(exp_weights, axis = -1, keepdims = True)

        exp_values = exp_weights @ v_chunk

        chunk_out = exp_values / k_len

        return (chunk_idx + k_chunk_sizes, out + chunk_out, row_sum + block_row_sum), None

    out = jnp.zeros((q_len, dim))
    row_sum = jnp.zeros((q_len, 1))

    (_, out, row_sum), _ = lax.scan(chunk_scanner, init = (0, out, row_sum), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    out = out * (k_len / (row_sum + EPSILON)) # 在获取所有正确的行和之后重新归一化

    out = out.reshape(q_len, v_dim)
    row_sum = row_sum.reshape(q_len)

    return out, row_sum

@jit
def l2norm(t):
    return t / (jnp.linalg.norm(t) + EPSILON)

@jit
def cosine_sim_flash_attention(q, k, v, key_mask):
    q, k = map(l2norm, (q, k))
    return cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask)

def _cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask):
    q_len, dim, v_dim = *q.shape, v.shape[-1]

    def chunk_scanner(chunk_idx, _):
        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (chunk_sizes, dim))

        return (chunk_idx + chunk_sizes, _query_chunk_flash_attention(chunk_idx, q_chunk, k, v, key_mask))

    _, (out, row_sum) = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    out = out.reshape(q_len, v_dim)
    row_sum = row_sum.reshape(q_len)

    return out, (row_sum,)

@custom_vjp
def cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask):
  out, _ = _cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask)
  return out

@jit
def flash_attention_forward(q, k, v, key_mask):
    out, (row_sum,) = _cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask)
    return out, (q, k, v, key_mask, out, row_sum)

def _query_chunk_flash_attention_backward(q, k, v, key_mask,o, do, l):
    q_len, dim, k_len, v_dim = *q.shape, *v.shape
    # 定义一个函数，用于扫描处理输入数据的分块
    def chunk_scanner(carries, _):
        # 从输入参数中获取当前处理的分块索引和数据
        chunk_idx, dq = carries
        # 计算当前分块的大小，取最小值
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        # 从输入数据中切片出当前处理的键值对应的分块数据
        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, v_dim))
        key_mask_chunk = lax.dynamic_slice(key_mask, (chunk_idx,), slice_sizes=(k_chunk_sizes,))

        # 计算注意力权重
        attn_weights = q @ k_chunk.transpose() * COSINE_SIM_SCALE - COSINE_SIM_SCALE

        # 计算指数化的注意力权重
        exp_attn_weights = jnp.exp(attn_weights)

        # 将注意力权重应用于键掩码
        exp_attn_weights = jnp.where(key_mask_chunk, exp_attn_weights, 0.)

        # 计算注意力概率
        p = exp_attn_weights / (l + EPSILON)

        # 计算值的梯度
        dv_chunk = p.transpose() @ do
        dp = do @ v_chunk.transpose()

        # 计算 D 值
        D = jnp.sum(do * o, axis=-1, keepdims=True)
        # 计算 s 值
        ds = p * COSINE_SIM_SCALE * (dp - D)

        # 计算查询的梯度
        dq_chunk = ds @ k_chunk
        # 计算键的梯度
        dk_chunk = ds.transpose() @ q

        # 返回更新后的分块索引和数据
        return (chunk_idx + k_chunk_sizes, dq + dq_chunk), (dk_chunk, dv_chunk)

    # 初始化 dq
    dq = jnp.zeros_like(q)

    # 使用 chunk_scanner 函数扫描处理输入数据的分块
    (_, dq), (dk, dv) = lax.scan(chunk_scanner, init=(0, dq), xs=None, length=math.ceil(k_len / K_CHUNK_SIZE))

    # 重新调整 dq、dk、dv 的形状
    dq = dq.reshape(q_len, dim)
    dk = dk.reshape(k_len, v_dim)
    dv = dv.reshape(k_len, v_dim)

    # 返回更新后的 dq、dk、dv
    return dq, dk, dv
# 使用 JIT 编译器对函数进行即时编译，提高性能
@jit
# 定义反向传播函数，接收前向传播的结果和梯度
def flash_attention_backward(res, do):
    # 解包前向传播结果
    q, k, v, key_mask, o, l = res

    # 获取查询向量的长度和维度
    q_len, dim = q.shape

    # 创建和 k, v 相同形状的零矩阵
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    # 重塑 l 的形状为 (q_len, 1)
    l = l.reshape(q_len, 1)

    # 定义一个函数用于扫描数据块
    def chunk_scanner(carries, _):
        chunk_idx, dk, dv = carries

        # 设置数据块的大小，不超过 Q_CHUNK_SIZE
        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        # 切片获取查询向量的数据块
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes=(chunk_sizes, q.shape[-1]))
        l_chunk = lax.dynamic_slice(l, (chunk_idx, 0), slice_sizes=(chunk_sizes, 1))
        o_chunk = lax.dynamic_slice(o, (chunk_idx, 0), slice_sizes=(chunk_sizes, o.shape[-1]))
        do_chunk = lax.dynamic_slice(do, (chunk_idx, 0), slice_sizes=(chunk_sizes, do.shape[-1]))

        # 调用子函数计算梯度
        dq_chunk, dk_chunk, dv_chunk = _query_chunk_flash_attention_backward(q_chunk, k, v, key_mask, o_chunk, do_chunk, l_chunk)
        return (chunk_idx + chunk_sizes, dk + dk_chunk, dv + dv_chunk), dq_chunk

    # 使用 lax.scan 函数扫描数据块
    (_, dk, dv), dq = lax.scan(chunk_scanner, init=(0, dk, dv), xs=None, length=math.ceil(q_len / Q_CHUNK_SIZE))

    # 重塑 dq 的形状为 (q_len, dim)
    dq = dq.reshape(q_len, dim)

    # 返回 dq, dk, dv 和 None
    return dq, dk, dv, None

# 定义反向传播函数的导数
cosine_sim_flash_attention_after_l2norm.defvjp(flash_attention_forward, flash_attention_backward)
```