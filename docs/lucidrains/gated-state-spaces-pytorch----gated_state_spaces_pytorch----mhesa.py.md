# `.\lucidrains\gated-state-spaces-pytorch\gated_state_spaces_pytorch\mhesa.py`

```
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft
from einops import rearrange
from scipy.fftpack import next_fast_len

# functions

# 检查值是否存在
def exists(val):
    return val is not None

# 在张量中添加指定数量的维度
def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

# 使用傅立叶技巧进行 O(N log(N)) 的一维卷积
def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = torch.fft.rfft(x, n = fast_len, dim = dim)
    f_weight = torch.fft.rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)
    out = torch.fft.irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# classes

# MHESA 模块
class MHESA(nn.Module):
    """ used for time-series in ETSFormer https://arxiv.org/abs/2202.01381 """

    def __init__(
        self,
        *,
        dim,
        heads,
        reverse_seq = False
    ):
        super().__init__()
        assert (dim % heads) == 0
        self.reverse_seq = reverse_seq

        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.alphas = nn.Parameter(torch.randn(heads))
        self.dampen_factors = nn.Parameter(torch.randn(heads))

        # params D

        self.param_D = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        """
        einstein notation:
        b - batch
        h - heads
        l - sequence length
        d - dimension
        """

        if self.reverse_seq:
            x = torch.flip(x, dims = (1,))

        device, seq_len = x.device, x.shape[1]
        u = self.norm(x)

        # learned weighted residual

        residual = u * self.param_D

        # weights derived from alphas (learned exponential smoothing decay rate)

        alphas = self.alphas.sigmoid()
        dampen_factors = self.dampen_factors.sigmoid()

        reversed_powers = torch.arange(seq_len - 1, -1, -1, device = device)
        K = alphas * (((1 - alphas) * dampen_factors) ** rearrange(reversed_powers, '... l -> ... l 1'))

        # conv1d fft O(nlog(n))

        u = rearrange(u, '... (h d) -> ... h d', h = self.heads)

        out = conv1d_fft(u, K, dim = -3, weight_dim = -2)

        out = rearrange(out, '... h d -> ... (h d)')

        out = out + residual

        if self.reverse_seq:
            out = torch.flip(out, dims = (1,))

        return out

# GatedMHESA 模块
class GatedMHESA(nn.Module):
    """ Pseudocode 3.2 """
    """ except state spaces replaced with multi-head exponential smoothing with learned alpha """
    """ used for time-series in ETSFormer https://arxiv.org/abs/2202.01381 """

    def __init__(
        self,
        *,
        dim,    
        heads = 8,
        dim_mhesa = 512,
        dim_expansion_factor = 4,
    ):
        super().__init__()
        assert (dim_mhesa % heads) == 0

        self.norm = nn.LayerNorm(dim)

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dim_mhesa, bias = False), nn.GELU())

        self.mhesa = MHESA(dim = dim_mhesa, heads = heads)

        self.to_gate = nn.Linear(dim_mhesa, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.mhesa(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        return out + residual

# Gated Dsconv LM

class GatedExponentialSmoothingLM(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        num_tokens,  # 标记的数量
        dim,  # 向量维度
        depth,  # 模型深度
        heads = 8,  # 多头注意力机制的头数
        dim_mhesa = 512,  # MHESA 模块的维度
        dim_expansion_factor = 4,  # 扩展因子
    ):
        super().__init__()
        # 创建标记嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 创建多个 GatedMHESA 层
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                GatedMHESA(
                    dim = dim,
                    heads = heads,
                    dim_mhesa = dim_mhesa,
                    dim_expansion_factor = dim_expansion_factor
                )
            )

        # 创建输出层
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    # 前向传播函数
    def forward(self, x, labels = None):
        # 对输入进行标记嵌入
        x = self.token_emb(x)

        # 遍历多个 GatedMHESA 层
        for mhesa in self.layers:
            x = mhesa(x)

        # 将结果传入输出层
        logits = self.to_logits(x)

        # 如果没有标签，则直接返回结果
        if not exists(labels):
            return logits

        # 重新排列 logits 的维度
        logits = rearrange(logits, 'b n c -> b c n')
        # 计算交叉熵损失
        return F.cross_entropy(logits, labels)
```