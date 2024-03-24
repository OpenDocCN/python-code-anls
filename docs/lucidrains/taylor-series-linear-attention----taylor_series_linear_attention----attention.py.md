# `.\lucidrains\taylor-series-linear-attention\taylor_series_linear_attention\attention.py`

```py
# 导入必要的库
import importlib
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from typing import Optional
from torchtyping import TensorType

from rotary_embedding_torch import RotaryEmbedding

# 定义常量

# 命名元组，用于存储缓存信息
Cache = namedtuple('Cache', [
    'seq_len',
    'last_token',
    'kv_cumsum',
    'k_cumsum'
])

# 定义函数

# 判断变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 对张量进行循环移位操作
def shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value = 0.)
    return torch.cat((t, t_shift), dim = -1)

# 预标准化

# RMS 标准化模块
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.gamma * F.normalize(x, dim = -1) * self.scale

# 使用二阶泰勒展开计算指数函数
def second_taylor_expansion(x: Tensor):
    dtype, device, dim = x.dtype, x.device, x.shape[-1]

    x, ps = pack([x], '* d')

    lead_dims = x.shape[0]

    # exp(qk) = 1 + qk + (qk)^2 / 2

    x0 = x.new_ones((lead_dims,))
    x1 = x
    x2 = einsum('... i, ... j -> ... i j', x, x) * (0.5 ** 0.5)

    # 连接 - 维度 D 现在变成 (1 + D + D ^2)
    # 在论文中，他们必须大幅减少注意力头维度才能使其工作

    out, _ = pack([x0, x1, x2], 'b *')
    out, = unpack(out, ps, '* d')
    return out

# 主类

# 泰勒级数线性注意力模块
class TaylorSeriesLinearAttn(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 16,
        heads = 8,
        causal = False,
        one_headed_kv = False,
        rotary_emb = False,
        combine_heads = True,
        gate_value_heads = False,
        prenorm = False,
        shift_tokens = False,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.shift_tokens = shift_tokens
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.heads = heads
        self.dim_hidden = dim_inner

        self.causal = causal
        self.causal_linear_attn_fn = None

        if causal:
            if not exists(importlib.util.find_spec('fast_transformers')):
                print('pytorch-fast-transformers must be installed. `pip install pytorch-fast-transformers` first')
                exit()

            from fast_transformers.causal_product import CausalDotProduct
            self.causal_linear_attn_fn = CausalDotProduct.apply

        kv_heads = heads if not one_headed_kv else 1
        dim_kv_inner = dim_head * (heads if not one_headed_kv else 1)

        self.rotary_emb = RotaryEmbedding(dim_head) if rotary_emb else None

        self.one_headed_kv = one_headed_kv

        # 查询投影层
        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_inner, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # 键值投影层
        self.to_kv = nn.Sequential(
            nn.Linear(dim, dim_kv_inner * 2, bias = False),
            Rearrange('b n (kv h d) -> kv b h n d', kv = 2, h = kv_heads)
        )

        # 值门控层
        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        ) if gate_value_heads else None

        # 合并注意力头
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.to_out = nn.Identity()

        if combine_heads:
            # 输出层
            self.to_out = nn.Sequential(
                nn.Linear(dim_inner, dim, bias = False),
                nn.Dropout(dropout)
            )
    # 定义一个方法用于前向传播
    def forward(
        # 输入参数 x，类型为张量，形状为 ['batch', 'seq', 'dim']，数据类型为 float
        x: TensorType['batch', 'seq', 'dim', float],
        # 可选参数 mask，类型为张量，形状为 ['batch', 'seq']，数据类型为 bool，默认为 None
        mask: Optional[TensorType['batch', 'seq', bool]] = None,
        # 可选参数 context，类型为张量，形状为 ['batch', 'target_seq', 'dim']，数据类型为 float，默认为 None
        context: Optional[TensorType['batch', 'target_seq', 'dim', float]] = None,
        # 参数 eps，数据类型为 float，默认值为 1e-5
        eps: float = 1e-5,
        # 可选参数 cache，类型为 Cache 对象，默认为 None
        cache: Optional[Cache] = None,
        # 参数 return_cache，数据类型为 bool，默认值为 False
        return_cache = False
# 适用于图像和视频的通道优先的Taylor Series线性注意力机制模块
class ChannelFirstTaylorSeriesLinearAttn(Module):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()
        # 初始化Taylor Series线性注意力机制
        self.attn = TaylorSeriesLinearAttn(*args, **kwargs)

    def forward(
        self,
        x: Tensor
    ):
        # 将输入张量重新排列为'通道优先'的形式
        x = rearrange(x, 'b c ... -> b ... c')
        # 打包输入张量，将通道维度视为单个维度
        x, ps = pack([x], 'b * c')

        # 使用Taylor Series线性注意力机制处理输入张量
        out = self.attn(x)

        # 解包处理后的张量，恢复原始形状
        out, = unpack(out, ps, 'b * c')
        # 将输出张量重新排列为原始形状
        return rearrange(out, 'b ... c -> b c ...')
```