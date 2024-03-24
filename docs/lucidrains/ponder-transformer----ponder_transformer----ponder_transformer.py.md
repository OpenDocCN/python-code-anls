# `.\lucidrains\ponder-transformer\ponder_transformer\ponder_transformer.py`

```py
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# 常量

ABS_MAX_STEPS = 100

# 辅助函数

def exists(val):
    return val is not None

# 类

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        n, h, device = x.shape[1], self.heads, x.device
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            sim = sim.masked_fill(mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = device).triu(j - i + 1).bool()
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# pondering 类和辅助函数

def pad_to(t, padding, dim = -1, value = 0.):
    if dim > 0:
        dim = dim - t.ndim
    zeroes = -dim - 1
    return F.pad(t, (*((0, 0) * zeroes), *padding), value = value)

def safe_cumprod(t, eps = 1e-10, dim = -1):
    t = torch.clip(t, min = eps, max = 1.)
    return torch.exp(torch.cumsum(torch.log(t), dim = dim))

def exclusive_cumprod(t, dim = -1):
    cum_prod = safe_cumprod(t, dim = dim)
    return pad_to(cum_prod, (1, -1), value = 1., dim = dim)

def calc_geometric(l, dim = -1):
    return exclusive_cumprod(1 - l, dim = dim) * l

# 主类

class Block(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        ff_mult = 4
    ):
        super().__init__()
        self.causal = causal
        self.attn = PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal))
        self.ff = PreNorm(dim, FeedForward(dim = dim, mult = ff_mult))

        self.to_halt_logits = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... () -> ...')
        )

    def forward(self, x, mask = None):
        x = self.attn(x, mask = mask) + x
        x = self.ff(x) + x

        if self.causal:
            denom = torch.arange(x.shape[-2], device = x.device)
            denom = rearrange(denom, 'n -> () n ()')
            halt_input = x.cumsum(dim = 1) / (denom + 1)
        else:
            halt_input = x.mean(dim = 1)

        halt_logits = self.to_halt_logits(halt_input)

        return x, halt_logits

class PonderTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        causal = True,
        dim_head = 64,
        heads = 8,
        ponder_kl_div_loss_weight = 0.01,
        ponder_lambda_p = 0.2,
        ponder_epsilon = 0.05,
        eps = 1e-20
        ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化epsilon值
        self.eps = eps
        # 初始化causal值
        self.causal = causal
        # 初始化序列长度为最大序列长度
        self.seq_len = max_seq_len
        # 创建token嵌入层，将token映射到指定维度
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建位置嵌入层，将位置映射到指定维度
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 计算最大步数

        # 计算停止概率的阈值
        thres = 1 - ponder_epsilon
        # 计算几何级数停止概率
        halt_probs = calc_geometric(torch.full((ABS_MAX_STEPS,), ponder_lambda_p))
        # 计算停止概率的累积和
        cum_halt_probs = halt_probs.cumsum(dim = 0)
        # 训练最大步数为满足停止概率小于阈值的步数
        self.train_max_steps = (cum_halt_probs < thres).sum().item()

        # 初始化ponder_lambda_p值
        self.ponder_lambda_p = ponder_lambda_p
        # 初始化ponder_kl_div_loss_weight值
        self.ponder_kl_div_loss_weight = ponder_kl_div_loss_weight

        # pondering block

        # 创建Block模块
        self.block = Block(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            causal = causal
        )

        # 隐藏状态到'Y' - 输出

        # 创建输出层，包括LayerNorm和线性层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )
```