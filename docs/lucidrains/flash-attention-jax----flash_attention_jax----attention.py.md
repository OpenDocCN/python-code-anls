# `.\lucidrains\flash-attention-jax\flash_attention_jax\attention.py`

```py
# 导入需要的库
import jax
from jax import nn
from jax import jit, numpy as jnp
from jax.numpy import einsum

# 导入重塑函数
from einops import rearrange

# 定义常量
EPSILON = 1e-10
MASK_VALUE = -1e10
COSINE_SIM_SCALE = 10

# 定义注意力机制函数
@jit
def attention(q, k, v, key_mask):
    # 获取维度和 k 的长度
    dim, k_len = q.shape[-1], k.shape[-2]
    scale = 1 / jnp.sqrt(dim)

    # 对查询进行缩放
    q = q * scale
    # 计算查询和键之间的相似度
    sim = einsum('... i d, ... j d -> ... i j', q, k)

    # 对键进行掩码处理
    key_mask = rearrange(key_mask, 'b j -> b 1 1 j')
    sim = jnp.where(key_mask, sim, MASK_VALUE)

    # 计算注意力权重并返回加权后的值
    attn = nn.softmax(sim, axis = -1)
    return attn @ v

# 定义因果注意力机制函数
@jit
def causal_attention(q, k, v):
    q_len, dim, k_len = *q.shape[-2:], k.shape[-2]
    scale = 1 / jnp.sqrt(dim)

    # 对查询进行缩放
    q = q * scale
    # 计算查询和键之间的相似度
    sim = einsum('... i d, ... j d -> ... i j', q, k)

    # 创建因果掩码
    causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k_len - q_len + 1)
    sim = jnp.where(causal_mask, MASK_VALUE, sim)

    # 计算注意力权重并返回加权后的值
    attn = nn.softmax(sim, axis = -1)
    return einsum('... i j, ... j d -> ... i d', attn, v)

# 定义余弦相似度注意力机制函数
@jit
def l2norm(t):
    return t / (jnp.linalg.norm(t) + EPSILON)

@jit
def cosine_sim_attention(q, k, v, key_mask):
    dim, k_len = q.shape[-1], k.shape[-2]
    # 对查询和键进行 L2 归一化
    q, k = map(l2norm, (q, k))

    # 计算余弦相似度
    sim = einsum('... i d, ... j d -> ... i j', q, k) * COSINE_SIM_SCALE

    # 对键进行掩码处理
    key_mask = rearrange(key_mask, 'b j -> b 1 1 j')
    sim = jnp.where(key_mask, sim, MASK_VALUE)

    # 计算注意力权重并返回加权后的值
    attn = nn.softmax(sim, axis = -1)
    return einsum('... i j, ... j d -> ... i d', attn, v)
```