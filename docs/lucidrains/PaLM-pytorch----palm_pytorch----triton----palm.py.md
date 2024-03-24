# `.\lucidrains\PaLM-pytorch\palm_pytorch\triton\palm.py`

```
# 导入所需的库
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

# 导入自定义的模块
from palm_pytorch.triton.softmax import causal_softmax
from palm_pytorch.triton.layernorm import layernorm_without_bias

# normalization

# 定义 LayerNorm 类，用于实现 Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return layernorm_without_bias(x, x.shape[-1:], self.gamma)


# residual

# 定义 Residual 类，用于实现残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# rotary positional embedding

# 定义 RotaryEmbedding 类，用于实现旋转位置嵌入
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


# 定义旋转操作函数
def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


# 应用旋转位置嵌入到输入张量
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# feedforward

# 定义 SwiGLU 类，用于实现 Swish-Gated Linear Unit
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

# 定义 ParallelTransformerBlock 类，实现并行的 Transformer 模块
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.ff_out = nn.Sequential(SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False)

        # for caching of rotary embeddings

        self.register_buffer("pos_emb", None, persistent=False)

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # attention

        attn = causal_softmax(sim)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


# transformer

# 定义 PaLM 函数，用于实现 Parallel Transformer
def PaLM(*, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4):
    # 创建一个神经网络模型，包括嵌入层、多个平行Transformer块、LayerNorm层和线性层
    net = nn.Sequential(
        # 创建一个嵌入层，将输入的标记转换为指定维度的向量
        nn.Embedding(num_tokens, dim),
        # 使用循环创建指定数量的平行Transformer块，并将它们作为残差连接添加到Sequential中
        *[
            Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult))
            for _ in range(depth)
        ],
        # 添加LayerNorm层，对模型的输出进行归一化处理
        LayerNorm(dim),
        # 添加线性层，将模型的输出映射为标记的数量
        nn.Linear(dim, num_tokens, bias=False)
    )

    # 将最后一个线性层的权重设置为嵌入层的权重
    net[-1].weight = net[0].weight

    # 对嵌入层的权重进行正态分布初始化，标准差为0.02
    nn.init.normal_(net[0].weight, std=0.02)
    # 返回创建的神经网络模型
    return net
```