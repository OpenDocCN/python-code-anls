# `.\lucidrains\mlp-gpt-jax\mlp_gpt_jax\mlp_gpt_jax.py`

```py
# 导入必要的库
from functools import partial

import jax
from jax import random
from jax import nn
import jax.numpy as np

import haiku as hk
from haiku import initializers
from einops import rearrange

# 常量定义
EPS = 1e-3
ATTN_MASK_VALUE = -1e10

# 定义 LayerNorm 函数
LayerNorm = partial(hk.LayerNorm, create_scale = True, create_offset = True, axis = -1)

# 定义 exists 函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义 Attention 类
class Attention(hk.Module):
    def __init__(
        self,
        *,
        dim_out,
        dim_head
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.to_qkv = hk.Linear(dim_head * 3)
        self.to_out = hk.Linear(dim_out)

    def __call__(self, x):
        n = x.shape[0]

        qkv = self.to_qkv(x)
        q, k, v = np.split(qkv, 3, axis = -1)
        sim = np.einsum('i d, j d -> i j', q, k) * self.scale

        mask = np.triu(np.ones((n, n), dtype = bool), 1)
        sim = np.where(mask, ATTN_MASK_VALUE, sim)

        attn = nn.softmax(sim, axis = -1)
        out = np.einsum('i j, j d -> i d', attn, v)
        return self.to_out(out)

# 定义 SGU 类
class SGU(hk.Module):
    def __init__(
        self,
        *,
        dim,
        dim_out,
        seq_len
    ):
        super().__init__()
        self.seq_len = seq_len
        self.norm = LayerNorm()
        self.proj_out = hk.Linear(dim_out)

    def __call__(self, x, gate_res = None):
        n = self.seq_len
        x, gate = np.split(x, 2, axis = -1)

        gate = self.norm(gate)

        init_scale = EPS / n
        init_eps = initializers.RandomUniform(minval = -init_scale, maxval = init_scale)

        weights = hk.get_parameter('spatial_weights', shape = (n, n), init = init_eps)
        biases = hk.get_parameter('spatial_biases', shape = (n, 1), init = np.ones)

        mask = np.tril(np.ones((n, n)))
        weights = weights * mask

        gate = np.einsum('n d, m n -> m d', gate, weights)
        gate += biases

        if exists(gate_res):
            gate += gate_res

        x = x * gate
        return self.proj_out(x)

# 定义 gMLP 类
class gMLP(hk.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        name,
        attn_dim = None
    ):
        super().__init__(name = name)
        self.attn = Attention(dim_head = attn_dim, dim_out = dim_ff // 2) if exists(attn_dim) else None
        self.norm = LayerNorm()
        self.proj_in = hk.Linear(dim_ff)
        self.sgu = SGU(dim = dim_ff, dim_out = dim_ff // 2, seq_len = seq_len)
        self.proj_out = hk.Linear(dim)

    def __call__(self, x):
        x = self.norm(x)
        gate_res = self.attn(x) if exists(self.attn) else None

        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.sgu(x, gate_res)
        x = self.proj_out(x)
        return x

# 定义 MaybeExecute 类
class MaybeExecute(hk.Module):
    def __init__(
        self,
        *,
        prob_execute,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.prob_execute = prob_execute

    def __call__(self, x):
        key = hk.next_rng_key()
        p = random.bernoulli(key, p = self.prob_execute)
        out = self.fn(x) * p + 0 * (1 - p)
        return out / self.prob_execute

# 定义 MLPGpt 类
class MLPGpt(hk.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        depth,
        heads = 1,
        ff_mult = 4,
        attn_dim = None,
        clamp_gate = True,
        layer_survival_prob = 1.
    ):
        super().__init__()
        self.embed = hk.Embed(num_tokens, dim)

        gmlps = [gMLP(dim = dim, dim_ff = dim * ff_mult, seq_len = seq_len, name = f'gmlp{i}', attn_dim = attn_dim) for i in range(depth)]
        self.layers = [MaybeExecute(prob_execute = layer_survival_prob, fn = gmlp) for gmlp in gmlps]

        self.to_logits = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_tokens)
        ])
    # 定义一个类的调用方法，接受输入 x
    def __call__(self, x):
        # 将输入 x 嵌入到模型中
        x = self.embed(x)

        # 遍历模型中的每一层，并对输入 x 进行处理
        for layer in self.layers:
            x += layer(x)

        # 将处理后的结果转换为 logits
        return self.to_logits(x)
# 定义一个装饰器函数，用于将 MLPGpt 模型转换为可训练的函数
def TransformedMLPGpt(**kwargs):
    # 定义一个内部函数，使用 hk.transform 装饰器将其转换为可训练函数
    def inner(seq):
        # 调用 MLPGpt 模型，并传入参数 kwargs，对输入序列进行处理
        return MLPGpt(**kwargs)(seq)
    # 返回内部函数
    return inner
```