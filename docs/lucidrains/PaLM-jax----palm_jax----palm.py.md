# `.\lucidrains\PaLM-jax\palm_jax\palm.py`

```py
# 导入所需的模块和库
from typing import List, Tuple

import numpy as onp
from jax import random, nn, lax, jit, numpy as np
from jax.numpy import einsum

from equinox import Module, static_field
from einops import rearrange, repeat

# bias-less layernorm

class LayerNorm(Module):
    gamma: np.ndarray
    eps: float = static_field()

    def __init__(self, dim, eps = 1e-5):
        # 初始化 LayerNorm 类，设置 gamma 和 eps 属性
        self.gamma = np.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        # 计算均值和均方差
        mean = np.mean(x, axis = -1, keepdims = True)
        mean_of_squares = np.mean(np.square(x), axis = -1, keepdims = True)
        variance = mean_of_squares - np.square(mean)
        inv = lax.rsqrt(variance + self.eps)
        # 返回 LayerNorm 结果
        return inv * (x - mean) * self.gamma

# Rotary embedding

def fixed_pos_embedding(inv_freq, seq):
    # 生成固定位置嵌入的正弦和余弦值
    sinusoid_inp = einsum('i , j -> i j', np.arange(seq), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, '... d -> ... (d r)', r = 2)
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)

def rotate_every_two(x):
    # 将输入张量中的每两个元素进行旋转
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x[..., 0], x[..., 1]
    x = np.stack((-x2, x1), axis = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    # 应用旋转位置嵌入
    return (x * cos) + (rotate_every_two(x) * sin)

# attention - multi-query, one-headed key / values variant
# feedforward - Shazeer's SwiGLU variant

class ParallelTransformerBlock(Module):
    norm: Module
    wi: np.ndarray
    attn_wo: np.ndarray
    ff_wo: np.ndarray

    heads: int = static_field()
    fused_dims: Tuple[int] = static_field()
    scale: float = static_field()
    mask_value: float = static_field()

    def __init__(
        self,
        dim,
        dim_head,
        heads,
        key,
        ff_mult = 4,
        mask_value = -1e10
    ):
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.norm = LayerNorm(dim)
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, ff_inner_dim, ff_inner_dim)

        self.wi = random.normal(key, (dim, sum(self.fused_dims)))
        self.attn_wo = random.normal(key, (attn_inner_dim, dim))
        self.ff_wo = random.normal(key, (ff_inner_dim, dim))

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.mask_value = mask_value

    def __call__(self, x, *, pos_emb, causal_mask):
        n, split_indices = x.shape[-2], onp.cumsum(self.fused_dims[:-1])

        x = self.norm(x)

        # fused attention and feedforward projections

        q, k, v, ff, ff_gate = np.split(x @ self.wi, split_indices, axis = -1)

        # split out heads

        q = rearrange(q, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q *= self.scale

        # apply rotary embeddings

        q, k = map(lambda t: apply_rotary_pos_emb(t, pos_emb), (q, k))

        # sim

        sim = einsum('... h i d, ... j d -> ... h i j', q, k)

        # causal mask

        sim = np.where(causal_mask, sim, self.mask_value)

        # attention

        attn = nn.softmax(sim, axis = -1)

        # aggregate values

        out = einsum('... h i j, ... j d -> ... h i d', attn, v)

        # merge heads

        out = rearrange(out, '... h n d -> ... n (h d)')

        # feedforward out

        attn_out = out @ self.attn_wo

        ff_out = (ff * nn.swish(ff_gate)) @ self.ff_wo

        # combine heads out

        return attn_out + ff_out

# main class

class PaLM(Module):
    embedding: np.ndarray
    norm: Module
    layers: List[List[Module]]
    inv_freq: onp.ndarray = static_field()

    def __init__(
        self,
        *,
        num_tokens,
        dim,
        dim_head,
        depth,
        heads,
        key,
        ff_mult = 4
    # 初始化 Transformer 模型的参数
    ):
        # 使用正态分布随机初始化嵌入矩阵，乘以0.02缩放
        self.embedding = random.normal(key, (num_tokens, dim)) * 0.02
        # 计算位置编码的倒数频率
        self.inv_freq = 1.0 / (10000 ** (np.arange(0, dim_head, 2) / dim_head))

        # 创建 Transformer 模型的多个层
        self.layers = [ParallelTransformerBlock(dim = dim, dim_head = dim_head, heads = heads, ff_mult = ff_mult, key = key) for _ in range(depth)]
        # 初始化 LayerNorm 层
        self.norm = LayerNorm(dim)

    # 定义 JIT 编译的调用函数
    @jit
    def __call__(self, x):
        # 获取输入张量 x 的最后一个维度大小
        n = x.shape[-1]
        # 使用嵌入矩阵将输入 x 映射到嵌入空间
        x = self.embedding[x]

        # 生成固定的位置编码
        rotary_emb = fixed_pos_embedding(self.inv_freq, n)
        # 生成因果掩码，下三角矩阵
        causal_mask = np.tril(np.ones((n, n)))

        # 遍历 Transformer 模型的每个层进行前向传播
        for block in self.layers:
            # 调用每个层的前向传播函数，更新输入 x
            x = block(x, pos_emb = rotary_emb, causal_mask = causal_mask) + x

        # 对输出 x 进行 LayerNorm 处理
        x = self.norm(x)
        # 返回最终输出，执行嵌入矩阵的转置乘积
        return x @ self.embedding.transpose()
```