# `.\lucidrains\progen\progen_transformer\progen.py`

```py
# 导入必要的库
from functools import partial

import jax
from jax import random
from jax import nn
from jax.lax import stop_gradient
import jax.numpy as np
import jmp

import haiku as hk
from haiku import initializers
from einops import rearrange, repeat

from progen_transformer.utils import exists

# 定义常量

ATTN_MASK_VALUE = -1e10

# 定义辅助函数

# 部分应用 LayerNorm 函数，创建 LayerNorm 实例
LayerNorm = partial(hk.LayerNorm, create_scale = True, create_offset = False, axis = -1)

# 生成固定位置的嵌入
def fixed_pos_embedding(seq, dim):
    # 计算频率
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    # 生成正弦和余弦输入
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(seq), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, "b n -> b (n r)", r = 2)[None, :, :]
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)

# 将每两个元素进行旋转
def rotate_every_two(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x[..., 0], x[..., 1]
    x = np.stack((-x2, x1), axis = -1)
    return rearrange(x, "... d r -> ... (d r)")

# 应用旋转位置嵌入
def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    rot_dim = sin.shape[-1]
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x = (x * cos) + (rotate_every_two(x) * sin)
    return np.concatenate((x, x_pass), axis = -1)

# 移动令牌
def shift_tokens(x):
    x_shift, x_pass = np.array_split(x, 2, axis = -1)
    x_shift = np.pad(x_shift, ((1, 0), (0, 0)), mode = 'constant')[:-1]
    return np.concatenate((x_shift, x_pass), axis = -1)

# 定义类

# 局部注意力机制
class LocalAttention(hk.Module):
    def __init__(
        self,
        *,
        name,
        dim,
        window_size,
        heads = 8,
        dim_head = 64,
        shift_tokens = True
    ):
        super().__init__(name = name)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.norm = LayerNorm()
        self.shift_tokens = shift_tokens

        self.to_qkv = hk.Linear(inner_dim * 3, with_bias = False)
        self.to_out = hk.Linear(dim)

    def __call__(self, x, *, pos_emb):
        x = self.norm(x)

        if self.shift_tokens:
            x = shift_tokens(x)

        n, h, wsz = x.shape[0], self.heads, self.window_size
        assert (n % wsz) == 0, 'sequence length must be divisible by the window size'
        window = n // wsz

        qkv = self.to_qkv(x)
        q, k, v = np.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h = h), (q, k, v))

        q, k, v = map(lambda t: apply_rotary_pos_emb(t, pos_emb), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'h (w n) d -> h w n d', w = window), (q, k, v))

        k, v = map(lambda t: np.pad(t, ((0, 0), (1, 0), (0, 0), (0, 0)), constant_values = 0.), (k ,v))
        k, v = map(lambda t: np.concatenate((t[:, :-1], t[:, 1:]), axis = 2), (k, v))

        sim = np.einsum('h w i d, h w j d -> h w i j', q, k) * self.scale

        mask = np.tril(np.ones((wsz, wsz * 2)), wsz)
        sim = np.where(mask, sim, ATTN_MASK_VALUE)

        sim = sim - stop_gradient(np.amax(sim, axis = -1, keepdims = True))
        attn = nn.softmax(sim, axis = -1)

        out = np.einsum('h w i j, h w j d -> h w i d', attn, v)
        out = rearrange(out, 'h w n d -> (w n) (h d)')
        return self.to_out(out)

# 前馈神经网络
class FeedForward(hk.Module):
    def __init__(
        self,
        *,
        name,
        dim,
        ff_mult = 4,
        glu = False,
        seq_len = None,
        spatial_gate = False,
        shift_tokens = True
    ):
        super().__init__(name = name)
        assert not (glu and spatial_gate), 'glu and sgu cannot be turned on at the same time'
        hidden_dim = dim * ff_mult
        hidden_dim *= (1 if not glu else 2)

        self.norm = LayerNorm()
        self.shift_tokens = shift_tokens

        self.proj_in = hk.Linear(hidden_dim)
        self.proj_out = hk.Linear(dim)

        self.glu = glu
        self.sgu = SGU(dim = hidden_dim, dim_out = hidden_dim // 2, seq_len = seq_len) if spatial_gate else None
    # 定义一个类的调用方法，接受输入 x
    def __call__(self, x):
        # 对输入 x 进行归一化处理
        x = self.norm(x)

        # 如果需要进行移位操作
        if self.shift_tokens:
            # 对 x 进行移位操作
            x = shift_tokens(x)

        # 对 x 进行投影操作
        x = self.proj_in(x)

        # 如果使用门控线性单元（GLU）
        if self.glu:
            # 将 x 拆分成两部分，分别为 x 和门控信号 gate
            x, gate = np.split(x, 2, axis=-1)
            # 对 x 进行门控线性单元激活函数处理
            x *= nn.gelu(gate)
        else:
            # 对 x 进行门控线性单元激活函数处理
            x = nn.gelu(x)

        # 如果存在自定义的门控单元（SGU）
        if exists(self.sgu):
            # 对 x 进行自定义门控单元处理
            x = self.sgu(x)

        # 对 x 进行输出投影操作
        x = self.proj_out(x)
        # 返回处理后的 x
        return x
# 定义 SGU 类，继承自 hk.Module
class SGU(hk.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,
        dim_out,
        seq_len,
        eps = 1e-3
    ):
        super().__init__()
        self.eps = eps
        self.seq_len = seq_len
        self.norm = LayerNorm()
        self.proj_out = hk.Linear(dim_out)

    # 调用函数
    def __call__(self, x):
        n = self.seq_len
        # 将输入 x 沿着最后一个轴分割成两部分
        x, gate = np.split(x, 2, axis = -1)

        # 对 gate 进行归一化
        gate = self.norm(gate)

        # 初始化缩放值
        init_scale = self.eps / n
        # 初始化随机均匀分布
        init_eps = initializers.RandomUniform(minval = -init_scale, maxval = init_scale)

        # 获取参数 weights 和 biases
        weights = hk.get_parameter('spatial_weights', shape = (n, n), init = init_eps)
        biases = hk.get_parameter('spatial_biases', shape = (n, 1), init = np.ones)

        # 生成一个下三角矩阵 mask
        mask = np.tril(np.ones((n, n)))
        weights = weights * mask

        # 使用矩阵乘法计算 gate
        gate = np.einsum('n d, m n -> m d', gate, weights)
        gate += biases

        # 对输入 x 进行门控
        x = x * gate
        return self.proj_out(x)

# 定义 ProGenBase 类，继承自 hk.Module
class ProGenBase(hk.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        depth,
        window_size = 256,
        global_mlp_depth = 2,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        ff_glu = True,
        attn_dim = None,
        clamp_gate = True,
        shift_tokens = True
    ):
        super().__init__()
        self.dim_head = dim_head
        self.embed = hk.Embed(num_tokens, dim)

        self.layers = []
        # 循环创建 depth 个层
        for i in range(depth):
            use_gmlp = (depth - i) <= global_mlp_depth
            use_ff_glu = not use_gmlp and ff_glu

            # 添加 LocalAttention 和 FeedForward 层到 layers 列表
            self.layers.append([
                LocalAttention(name = f'attn{i}', dim = dim, window_size = window_size, heads = heads, dim_head = dim_head, shift_tokens = shift_tokens),
                FeedForward(name = f'ff{i}', dim = dim, ff_mult = ff_mult, seq_len = seq_len, spatial_gate = use_gmlp, glu = use_ff_glu, shift_tokens = shift_tokens)
            ])

        # 定义输出层
        self.to_logits = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_tokens)
        ])

    # 调用函数
    def __call__(self, x):
        n = x.shape[0]
        x = self.embed(x)
        rotary_emb = fixed_pos_embedding(n, self.dim_head)

        # 循环遍历每个层并进行操作
        for attn, ff in self.layers:
            x += attn(x, pos_emb = rotary_emb)
            x += ff(x)

        return self.to_logits(x)

# 定义 ProGen 函数
def ProGen(mixed_precision = False, mixed_precision_policy = dict(params = 'float32', compute = 'float16', output = 'float32'), **kwargs):
    # 使用 hk.transform 装饰器
    @hk.transform
    def inner(seq):
        if mixed_precision:
            serialized_policy = ','.join([f'{k}={v}' for k, v in mixed_precision_policy.items()])
            policy = jmp.get_policy(serialized_policy)
            hk.mixed_precision.set_policy(ProGenBase, policy)
        return ProGenBase(**kwargs)(seq)
    return inner
```