# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\attention.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 einops 库中导入 rearrange 函数
from einops import rearrange
# 从 torch 库中导入 einsum 函数
from torch import einsum
# 从 bidirectional_cross_attention 模块中导入 BidirectionalCrossAttention 类

# 定义函数，判断变量是否存在
def exists(val):
    return val is not None

# 定义函数，返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义前馈神经网络类
def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# 自注意力机制类
class SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        mask = None,
    ):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 自注意力块类
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dropout = 0.,
        ff_mult = 4,
        **kwargs
    ):
        super().__init__()
        self.attn = SelfAttention(dim = dim, dropout = dropout, **kwargs)
        self.ff = FeedForward(dim = dim, mult = ff_mult, dropout = dropout)

    def forward(self, x, mask = None):
        x = self.attn(x, mask = mask) + x
        x = self.ff(x) + x
        return x

# 双向交叉注意力类
class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.
    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None
    ):
        h = self.heads

        x = self.norm(x)
        context = self.context_norm(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(context_mask):
            mask_value = -torch.finfo(sim.dtype).max
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class JointCrossAttentionBlock(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        dim,  # 维度
        context_dim = None,  # 上下文维度，默认为None
        ff_mult = 4,  # FeedForward模块的倍数，默认为4
        dropout = 0.,  # dropout概率，默认为0
        **kwargs  # 其他参数
    ):
        super().__init__()  # 调用父类的初始化函数
        context_dim = default(context_dim, dim)  # 如果上下文维度为None，则设置为维度值

        # 创建双向交叉注意力模块
        self.attn = BidirectionalCrossAttention(dim = dim, context_dim = context_dim, dropout = dropout, prenorm = True, **kwargs)
        # 创建FeedForward模块
        self.ff = FeedForward(dim, mult = ff_mult, dropout = dropout)
        # 创建上下文的FeedForward模块
        self.context_ff = FeedForward(context_dim, mult = ff_mult, dropout = dropout)

    # 前向传播函数
    def forward(
        self,
        x,  # 输入数据
        context,  # 上下文数据
        mask = None,  # 掩码，默认为None
        context_mask = None  # 上下文掩码，默认为None
    ):
        # 使用注意力模块处理输入数据和上下文数据
        attn_out, context_attn_out = self.attn(x, context, mask = mask, context_mask = context_mask)

        # 更新输入数据
        x = x + attn_out
        # 更新上下文数据
        context = context + context_attn_out

        # 使用FeedForward模块处理输入数据
        x = self.ff(x) + x
        # 使用上下文的FeedForward模块处理上下文数据
        context = self.context_ff(context) + context

        # 返回更新后的输入数据和上下文数据
        return x, context
```