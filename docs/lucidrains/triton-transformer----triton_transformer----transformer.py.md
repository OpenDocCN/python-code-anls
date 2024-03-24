# `.\lucidrains\triton-transformer\triton_transformer\transformer.py`

```
# 导入必要的库
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# 导入自定义的模块
from triton_transformer.layernorm import layernorm
from triton_transformer.softmax import softmax
from triton_transformer.cross_entropy import cross_entropy_fn
from triton_transformer.bmm import fused_relu_squared
from triton_transformer.dropout import dropout_fn
from triton_transformer.utils import exists, default

# 定义类

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, use_triton = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.use_triton = use_triton

    def forward(self, x, **kwargs):
        use_triton = kwargs.get('use_triton', self.use_triton)
        normed = layernorm(x, self.norm.weight, use_triton = use_triton)
        return self.fn(normed, **kwargs) + x

# 辅助类

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None, use_triton = None):
        use_triton = default(use_triton, self.use_triton)
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b i d, b j d -> b i j', q, k)

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(mask, mask_value)

        attn = softmax(sim, causal = self.causal, use_triton = use_triton)
        attn = dropout_fn(attn, self.dropout, use_triton = use_triton)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return dropout_fn(out, self.dropout, use_triton = use_triton)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        use_triton = False
    ):
        super().__init__()
        self.use_triton = use_triton
        inner_dim = dim * mult
        self.dropout = dropout
        self.proj_in_weight = nn.Parameter(torch.randn(dim, inner_dim))
        self.proj_out = nn.Linear(inner_dim, dim)

    def forward(self, x, use_triton = None):
        use_triton = default(use_triton, self.use_triton)

        x = fused_relu_squared(x, self.proj_in_weight, use_triton = use_triton)
        x = dropout_fn(x, self.dropout, use_triton = use_triton)

        x = self.proj_out(x)
        return x

# 主类

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        ff_dropout = 0.,
        ff_mult = 4,
        attn_dropout = 0.,
        use_triton = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化最大序列长度
        self.max_seq_len = max_seq_len
        # 创建 token embedding 层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建位置 embedding 层
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 初始化层列表
        self.layers = nn.ModuleList([])
        # 创建部分预归一化残差块
        wrapper = partial(PreNormResidual, dim)

        # 循环创建指定深度的注意力和前馈网络层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                wrapper(Attention(dim, heads = heads, dim_head = dim_head, causal = causal, dropout = attn_dropout, use_triton = use_triton)),
                wrapper(FeedForward(dim, dropout = ff_dropout, mult = ff_mult, use_triton = use_triton))
            ]))

        # 创建层归一化层
        self.norm = nn.LayerNorm(dim)
        # 创建输出层
        self.to_logits = nn.Linear(dim, num_tokens)

        # 创建掩码

        self.use_triton = use_triton
        self.causal = causal
        # 根据是否自回归创建掩码
        mask = torch.ones(max_seq_len, max_seq_len, dtype = torch.bool).triu(1) if causal else None
        self.register_buffer('mask', mask, persistent = False)

    def forward(
        self,
        x,
        mask = None,
        *,
        labels = None,
        use_triton = None
    ):
        # 设置使用 Triton 加速的标志
        use_triton = default(use_triton, self.use_triton)
        # 获取序列长度和设备信息
        n, device = x.shape[1], x.device

        # 嵌入 token 并添加位置嵌入

        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        # 生成掩码，取决于是否自回归

        assert not (self.causal and exists(mask)), 'mask is not needed during autoregressive mode'

        if self.causal and not use_triton:
            mask = self.mask[:n, :n]
            mask = rearrange(mask, 'i j -> () i j')
        elif not self.causal and exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = ~mask

        # 通过层

        for attn, ff in self.layers:
            x = attn(x, mask = mask, use_triton = use_triton)
            x = ff(x, use_triton = use_triton)

        # 进行层归一化
        x = layernorm(x, self.norm.weight, use_triton = use_triton, stable = True)
        # 计算 logits
        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        # 计算损失
        loss = cross_entropy_fn(logits, labels, ignore_index = 0, use_triton = use_triton)
        return loss
```