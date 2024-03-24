# `.\lucidrains\AoA-pytorch\aoa_pytorch\aoa_pytorch.py`

```
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# 定义一个函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义一个函数，如果变量存在则返回该变量，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义一个名为AttentionOnAttention的类，继承自nn.Module
class AttentionOnAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        aoa_dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 定义线性层，用于将输入转换为查询向量
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 定义线性层，用于将输入转换为键值对
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        # 定义dropout层
        self.dropout = nn.Dropout(dropout)

        # 定义Attention on Attention模块
        self.aoa = nn.Sequential(
            nn.Linear(2 * inner_dim, 2 * dim),
            nn.GLU(),
            nn.Dropout(aoa_dropout)
        )

    # 前向传播函数
    def forward(self, x, context = None):
        h = self.heads

        # 将输入x转换为查询向量
        q_ = self.to_q(x)

        # 如果存在上下文信息，则使用上下文信息作为键值对，否则使用输入x作为键值对
        context = default(context, x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        # 将查询向量、键向量和值向量按照头数拆分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q_, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 计算注意力权重
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # 加权平均值
        attn_out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并头部
        out = rearrange(attn_out, 'b h n d -> b n (h d)', h = h)

        # Attention on Attention模块
        out = self.aoa(torch.cat((out, q_), dim = -1))
        return out
```