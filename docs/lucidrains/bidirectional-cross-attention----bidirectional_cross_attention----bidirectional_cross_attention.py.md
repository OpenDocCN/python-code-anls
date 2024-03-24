# `.\lucidrains\bidirectional-cross-attention\bidirectional_cross_attention\bidirectional_cross_attention.py`

```
import torch
from torch import nn
from einops import rearrange
from torch import einsum

# 检查变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 双向交叉注意力机制模块
class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.,
        talking_heads = False,
        prenorm = False,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        # 初始化层归一化模块
        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        # 初始化 dropout 模块
        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        # 初始化线性变换模块
        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        # 初始化 talking heads 模块
        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # 获取共享的查询/键和值用于序列和上下文
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # 分割头部
        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        # 计算相似度
        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        # 如果提供了相对位置偏置
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # 掩码
        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # 获取序列长度和上下文长度维度的注意力
        # 共享相似度矩阵
        attn = sim.softmax(dim = -1)
        context_attn = sim.softmax(dim = -2)

        # dropout
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # 源序列聚合上下文的值，上下文聚合源序列的值
        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # 合并头部并组合输出
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out
```