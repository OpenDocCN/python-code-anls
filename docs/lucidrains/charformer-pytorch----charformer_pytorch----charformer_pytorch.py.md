# `.\lucidrains\charformer-pytorch\charformer_pytorch\charformer_pytorch.py`

```py
# 导入 math 模块
import math
# 从 math 模块中导入 gcd 函数
from math import gcd
# 导入 functools 模块
import functools
# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn, F, einsum
import torch.nn.functional as F
from torch import nn, einsum
# 从 einops 模块中导入 rearrange, reduce, repeat
from einops import rearrange, reduce, repeat
# 从 einops.layers.torch 模块中导入 Rearrange
from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 计算多个数的最小公倍数
def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

# 计算带有掩码的张量的均值
def masked_mean(tensor, mask, dim = -1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim)
    mean = tensor.sum(dim = dim) / total_el.clamp(min = 1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean

# 计算下一个可被整除的长度
def next_divisible_length(seqlen, multiple):
    return math.ceil(seqlen / multiple) * multiple

# 将张量填充到指定的倍数
def pad_to_multiple(tensor, multiple, *, seq_dim, dim = -1, value = 0.):
    seqlen = tensor.shape[seq_dim]
    length = next_divisible_length(seqlen, multiple)
    if length == seqlen:
        return tensor
    remainder = length - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# 辅助类

# 填充层
class Pad(nn.Module):
    def __init__(self, padding, value = 0.):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, value = self.value)

# 深度卷积层
class DepthwiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, groups = dim_in)
        self.proj_out = nn.Conv1d(dim_out, dim_out, 1)

    def forward(self, x):
        x = self.conv(x)
        return self.proj_out(x)

# 主类

class GBST(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_block_size = None,
        blocks = None,
        downsample_factor = 4,
        score_consensus_attn = True
    ):
        super().__init__()
        assert exists(max_block_size) ^ exists(blocks), 'either max_block_size or blocks are given on initialization'
        self.token_emb = nn.Embedding(num_tokens, dim)

        if exists(blocks):
            assert isinstance(blocks, tuple), 'blocks must be a tuple of block sizes'
            self.blocks = tuple(map(lambda el: el if isinstance(el, tuple) else (el, 0), blocks))
            assert all([(offset < block_size) for block_size, offset in self.blocks]), 'offset must be always smaller than the block size'

            max_block_size = max(list(map(lambda t: t[0], self.blocks)))
        else:
            self.blocks = tuple(map(lambda el: (el, 0), range(1, max_block_size + 1)))

        self.pos_conv = nn.Sequential(
            Pad((0, 0, 0, max_block_size - 1)),
            Rearrange('b n d -> b d n'),
            DepthwiseConv1d(dim, dim, kernel_size = max_block_size),
            Rearrange('b d n -> b n d')
        )

        self.score_fn = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... () -> ...')
        )

        self.score_consensus_attn = score_consensus_attn

        assert downsample_factor <= max_block_size, 'final downsample factor should be less than the maximum block size'

        self.block_pad_multiple = lcm(*[block_size for block_size, _ in self.blocks])
        self.downsample_factor = downsample_factor
```