# `.\lucidrains\all-normalization-transformer\all_normalization_transformer\all_normalization_transformer.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F
# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 定义累积均值函数
def cum_mean(t):
    # 获取张量的设备信息
    device = t.device
    # 创建一个从 1 到张量最后一个维度大小的张量
    running_num = torch.arange(t.shape[-1], device=t.device) + 1
    # 返回累积和除以运行次数的结果
    return t.cumsum(dim=-1) / running_num

# 定义归一化函数
def normalize(t, eps=1e-8):
    # 减去均值
    t -= t.mean(dim=-1, keepdim=True)
    # 计算标准差
    s = (t ** 2).mean(dim=-1, keepdim=True)
    # 返回归一化结果
    return t * torch.rsqrt(s + eps)

# 定义因果归一化函数
def causal_normalize(t, eps=1e-8):
    # 减去因果均值
    t -= cum_mean(t).diagonal(dim1=-2, dim2=-1)[..., None]
    # 计算因果标准差
    s = cum_mean(t ** 2).diagonal(dim1=-2, dim2=-1)[..., None]
    # 返回因果归一化结果
    return t * torch.rsqrt(s + eps)

# 定义残差模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# 定义后归一化模块
class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.fn(x)
        return self.norm(x)

# 定义前归一化模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# 定义前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        return self.net(x)

# 定义注意力模块
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, causal = False, shared_kv = False):
        super().__init__()
        self.causal = causal
        self.heads = heads
        self.scale = dim ** -0.5
        self.shared_kv = shared_kv
        self.num_qkv = 3 if not shared_kv else 2

        self.to_qkv = nn.Linear(dim, dim * self.num_qkv, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.norm_g = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.norm_b = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(self, x):
        b, n, _, h, device = *x.shape, self.heads, x.device
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = self.num_qkv, h = h)

        if self.shared_kv:
            q, k = qkv
            v = k
        else:
            q, k, v = qkv

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if self.causal:
            mask = torch.ones(n, n, device = device).triu_(1).bool()
            dots.masked_fill_(mask, 0.)

        normalize_fn = causal_normalize if self.causal else normalize
        normed_attn = normalize_fn(dots)
        attn = normed_attn * self.norm_g + self.norm_b

        if self.causal:
            attn.masked_fill_(mask, 0.)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

# 定义变压器模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads = 8, causal = False, only_norm = False, shared_kv = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PostNorm(dim, Attention(dim, heads, causal = causal, shared_kv = shared_kv))),
                Residual(PreNorm(dim, FeedForward(dim))) if not only_norm else nn.Identity(),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# 定义变压器语言模型模块
class TransformerLM(nn.Module):
    def __init__(self, *, num_tokens, dim, depth, max_seq_len, heads = 8, causal = False, only_norm = False, shared_kv = False):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.transformer = Transformer(dim, depth, heads, causal = causal, only_norm = only_norm, shared_kv = shared_kv)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, **kwargs):
        _, n = x.shape
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device=x.device))
        x = self.transformer(x)
        x = self.to_logits(x)
        return x
```