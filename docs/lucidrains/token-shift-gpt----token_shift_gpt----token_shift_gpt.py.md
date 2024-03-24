# `.\lucidrains\token-shift-gpt\token_shift_gpt\token_shift_gpt.py`

```
# 从 math 模块中导入 log2 和 ceil 函数
# 从 torch 模块中导入 nn, einsum 和 nn.functional 模块
from math import log2, ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

# 从 einops 模块中导入 rearrange 函数
from einops import rearrange

# 定义一个辅助函数，用于检查值是否存在
def exists(val):
    return val is not None

# 定义一个函数，用于在指定维度上对输入进行平移
def shift(x, amt, dim = -1):
    return F.pad(x, (*((0, 0) * (-dim - 1)), amt, -amt), value = 0.)

# 定义一个函数，用于在 tokens 上进行平移
def shift_tokens(x, amt, eps = 1e-5):
    n, device = x.shape[1], x.device

    # 计算累积和
    cumsum = x.cumsum(dim = 1)
    *x, x_pass = x.chunk(amt + 1, dim = -1)
    *x_cumsum, _ = cumsum.chunk(amt + 1, dim = -1)

    # 计算平移量
    amts = 2 ** torch.arange(amt)
    amts = amts.tolist()

    shifts = []
    denom = torch.arange(n, device = device)

    for x_chunk, x_cumsum_chunk, amt in zip(x, x_cumsum, amts):
        # 计算平移后的值
        shifted_chunk = shift(x_cumsum_chunk, amt, dim = -2) - shift(x_cumsum_chunk, 2 * amt, dim = -2)
        shifted_denom = shift(denom, amt, dim = -1) - shift(denom, 2 * amt, dim = -1)
        shifted_denom = rearrange(shifted_denom, 'n -> () n ()')
        normed_shifted_x = shifted_chunk /  (shifted_denom + eps)
        shifts.append(normed_shifted_x)

    return torch.cat((*shifts, x_pass), dim = -1)

# 定义一个函数，用于计算折扣累积和
def discounted_cumsum(t, gamma):
    try:
        from torch_discounted_cumsum import discounted_cumsum_left
    except ImportError:
        print('unable to import torch_discounted_cumsum - please run `pip install torch-discounted-cumsum`')

    b, n, d = t.shape
    t = rearrange(t, 'b n d -> (b d) n')
    t = discounted_cumsum_left(t, gamma)
    t = rearrange(t, '(b d) n -> b n d', b = b)
    return t

# 定义一个残差模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# 定义一个前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_seq_len,
        num_shifts,
        mult = 4,
        eps = 1e-3,
        use_discounted_cumsum = False,
        discount_gamma = 0.9
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.project_in = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU()
        )

        self.num_shifts = num_shifts
        hidden_dim = dim * mult // 2

        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.to_gate = nn.Linear(hidden_dim, hidden_dim)

        nn.init.constant_(self.to_gate.weight, eps)
        nn.init.constant_(self.to_gate.bias, 1.)

        self.project_out = nn.Linear(hidden_dim, dim)

        # 用于使用折扣累积和方法

        self.use_discounted_cumsum = use_discounted_cumsum
        self.discount_gamma = discount_gamma

    def forward(self, x):
        x = self.norm(x)

        x = self.project_in(x)

        x, gate = x.chunk(2, dim = -1)

        gate = self.gate_norm(gate)

        if self.use_discounted_cumsum:
            gate = shift(gate, 1, dim = -2)
            gate = discounted_cumsum(gate, self.discount_gamma)
        else:
            gate = shift_tokens(gate, self.num_shifts)

        x = x * self.to_gate(gate)
        return self.project_out(x)

# 定义一个 TokenShiftGPT 模块
class TokenShiftGPT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        depth,
        ff_mult = 4,
        use_discounted_cumsum = False,
        discount_gamma = 0.9
    ):
        super().__init__()
        self.seq_len = max_seq_len
        num_shifts = ceil(log2(max_seq_len)) - 1

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.net = nn.Sequential(
            *[Residual(FeedForward(dim = dim, num_shifts = num_shifts, mult = ff_mult, max_seq_len = max_seq_len, use_discounted_cumsum = use_discounted_cumsum, discount_gamma = discount_gamma)) for _ in range(depth)],
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )
    # 定义一个前向传播函数，接收输入 x
    def forward(self, x):
        # 对输入 x 进行 token embedding
        x = self.token_emb(x)
        # 生成位置编码，长度为 x 的第二维度，设备为 x 所在的设备
        pos_emb = self.pos_emb(torch.arange(x.shape[1], device = x.device))
        # 将位置编码与 token embedding 相加，并重新排列维度
        x = x + rearrange(pos_emb, 'n d -> () n d')
        # 将处理后的输入 x 输入到神经网络中进行计算
        return self.net(x)
```