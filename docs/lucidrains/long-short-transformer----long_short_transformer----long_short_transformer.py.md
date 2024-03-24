# `.\lucidrains\long-short-transformer\long_short_transformer\long_short_transformer.py`

```py
# 从 math 模块中导入 gcd（最大公约数）和 ceil（向上取整）函数
from math import gcd, ceil
# 导入 functools 模块
import functools

# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn（神经网络）和 einsum（张量乘法）模块
from torch import nn, einsum
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F

# 导入 rotary_embedding_torch 模块中的 RotaryEmbedding 和 apply_rotary_emb 函数
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# 导入 einops 模块中的 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 定义函数 exists，判断值是否存在
def exists(val):
    return val is not None

# 定义函数 default，如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义函数 lcm，计算多个数的最小公倍数
def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

# 定义函数 pad_to_multiple，将张量的长度填充到指定的倍数
def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    if m.is_integer():
        return tensor

    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)

# 定义函数 look_around，根据给定的向前和向后偏移量，在张量周围填充指定值
def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

# 定义类 PreNorm，实现预层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 定义类 FeedForward，实现前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# 定义类 LongShortAttention，实现长短注意力机制
class LongShortAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = True,
        window_size = 128,
        pos_emb = None,
        segment_size = 16,
        r = 1,
        dropout = 0.
    ):
        super().__init__()
        assert not (causal and r >= segment_size), 'r should be less than segment size, if autoregressive'

        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal

        self.window_size = window_size
        self.segment_size = segment_size
        self.pad_to_multiple = window_size if not causal else lcm(window_size, segment_size)

        self.to_dynamic_proj = nn.Linear(dim_head, r, bias = False)
        self.local_norm = nn.LayerNorm(dim_head)
        self.global_norm = nn.LayerNorm(dim_head)

        self.pos_emb = default(pos_emb, RotaryEmbedding(dim_head))

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

# 定义主类 LongShortTransformer，实现长短变换器
class LongShortTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        window_size = 128,
        causal = True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        segment_size = None,
        r = None,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):  
        # 调用父类的构造函数
        super().__init__()
        # 设置最大序列长度
        self.max_seq_len = max_seq_len

        # 创建 token embedding 层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建旋转嵌入层
        pos_emb = RotaryEmbedding(dim_head)

        # 处理自回归默认变量的方式不同
        # 具体来说，segments 仅在自回归情况下使用
        # r 在非自回归情况下是投影的 r << n，在自回归情况下是每个段的投影 r
        # 是的，这很令人困惑，我知道

        # 设置 segment_size 默认值
        segment_size = default(segment_size, 16 if causal else None)
        # 设置 r 默认值
        r = default(r, 1 if causal else 128)

        # 创建多层神经网络
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 每层包含一个注意力机制和一个前馈神经网络
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LongShortAttention(dim = dim, heads = heads, dim_head = dim_head, window_size = window_size, causal = causal, pos_emb = pos_emb, segment_size = segment_size, r = r, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        # 创建输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        # 对输入进行 token embedding
        x = self.token_emb(x)

        # 遍历每一层的注意力机制和前馈神经网络
        for attn, ff in self.layers:
            # 注意力机制
            x = attn(x, mask = mask) + x
            # 前馈神经网络
            x = ff(x) + x

        # 输出结果
        return self.to_logits(x)
```