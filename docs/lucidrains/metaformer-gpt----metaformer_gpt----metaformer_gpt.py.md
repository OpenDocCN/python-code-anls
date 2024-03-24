# `.\lucidrains\metaformer-gpt\metaformer_gpt\metaformer_gpt.py`

```py
import torch
from torch import nn, einsum
from einops import rearrange, repeat

from scipy.fftpack import next_fast_len

# 辅助函数

def cummean(x, *, dim):
    # 计算累积均值
    numer = x.cumsum(dim = dim)
    denom = torch.arange(x.shape[1], device = x.device) + 1
    return numer / rearrange(denom, '... -> ... 1')

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # 使用傅立叶技巧进行 O(N log(N)) 1维卷积

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    # 对输入信号和权重进行傅立叶变换
    f_x = torch.fft.rfft(x, n = fast_len, dim = dim)
    f_weight = torch.fft.rfft(weights, n = fast_len, dim = weight_dim)

    # 计算频域乘积
    f_v_weight = f_x * rearrange(f_weight.conj(), '... -> ... 1')
    out = torch.fft.irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    # 选择输出的部分
    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# 类

class MeanCenteringPool(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)
        x = cummean(x, dim = 1) - x
        return self.proj(x)

class MultiheadExponentialTimeDecay(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.randn(heads))

        self.project_in = nn.Linear(dim, inner_dim, bias = False)
        self.project_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        b, n, d, h, device = *x.shape, self.heads, x.device

        x = self.norm(x)

        # 线性投影

        x = self.project_in(x)

        # 分割头部

        x = rearrange(x, 'b n (h d) -> b h n d', h = h)

        # 准备指数 alpha

        alpha = self.alpha.sigmoid()
        alpha = rearrange(alpha, 'h -> h 1')

        # 计算权重

        arange = torch.arange(n, device = device)
        weights = alpha * (1 - alpha) ** torch.flip(arange, dims = (0,))
        output = conv1d_fft(x, weights)

        # 合并头部

        output = rearrange(output, 'b h n d -> b n (h d)')
        return self.project_out(output)

def FeedForward(dim, mult = 4):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.Linear(hidden_dim, dim, bias = False)
    )

class MetaformerGPT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 16,
        dim_head = 32,
        max_seq_len = 2048,
        ff_mult = 4
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadExponentialTimeDecay(dim, heads = heads, dim_head = dim_head),
                MeanCenteringPool(dim),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device))

        for mh_esa, pool, ff in self.layers:
            x = mh_esa(x) + x
            x = pool(x) + x
            x = ff(x) + x

        return self.to_logits(x)
```