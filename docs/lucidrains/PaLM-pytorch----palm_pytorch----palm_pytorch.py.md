# `.\lucidrains\PaLM-pytorch\palm_pytorch\palm_pytorch.py`

```
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 einops 库中导入 rearrange 函数
from einops import rearrange
# 从 torch 库中导入 einsum 和 nn 模块
from torch import einsum, nn

# normalization
# they use layernorm without bias, something that pytorch does not offer

# 定义 LayerNorm 类，继承自 nn.Module
class LayerNorm(nn.Module):
    # 初始化函数
    def __init__(self, dim):
        super().__init__()
        # 创建可学习参数 gamma
        self.gamma = nn.Parameter(torch.ones(dim))
        # 创建 buffer beta
        self.register_buffer("beta", torch.zeros(dim))

    # 前向传播函数
    def forward(self, x):
        # 使用 F.layer_norm 进行层归一化
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual

# 定义 Residual 类，继承自 nn.Module
class Residual(nn.Module):
    # 初始化函数
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    # 前向传播函数
    def forward(self, x):
        # 返回残差连接结果
        return self.fn(x) + x

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

# 定义 RotaryEmbedding 类，继承自 nn.Module
class RotaryEmbedding(nn.Module):
    # 初始化函数
    def __init__(self, dim):
        super().__init__()
        # 计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 创建 buffer inv_freq
        self.register_buffer("inv_freq", inv_freq)

    # 前向传播函数
    def forward(self, max_seq_len, *, device):
        # 生成序列
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        # 计算频率
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        # 拼接频率
        return torch.cat((freqs, freqs), dim=-1)

# 旋转位置嵌入
def rotate_half(x):
    # 重新排列张量维度
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    # 拆分张量
    x1, x2 = x.unbind(dim=-2)
    # 拼接张量
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    # 计算旋转位置嵌入
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

# 定义 SwiGLU 类，继承自 nn.Module
class SwiGLU(nn.Module):
    # 前向传播函数
    def forward(self, x):
        # 拆分张量
        x, gate = x.chunk(2, dim=-1)
        # 使用 SiLU 激活函数
        return F.silu(gate) * x

# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

# 定义 ParallelTransformerBlock 类，继承自 nn.Module
class ParallelTransformerBlock(nn.Module):
    # 初始化函数
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        # 归一化层
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    # 获取掩码
    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    # 获取旋转嵌入
    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取输入张量 x 的形状信息
        n, device, h = x.shape[1], x.device, self.heads

        # 对输入张量 x 进行 LayerNorm 处理
        x = self.norm(x)

        # 使用融合的注意力和前馈神经网络投影层对输入张量 x 进行投影
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # 将投影后的张量按照指定维度进行分割，用于多头注意力
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # 获取旋转位置嵌入
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # 缩放
        q = q * self.scale

        # 计算相似度
        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # 获取因果掩码
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力权重计算
        attn = sim.softmax(dim=-1)

        # 聚合值
        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # 合并多头
        out = rearrange(out, "b h n d -> b n (h d)")
        # 返回注意力输出和前馈网络输出的和
        return self.attn_out(out) + self.ff_out(ff)
# 定义一个函数PaLM，用于创建一个Parallel Transformer模型
def PaLM(*, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4):
    # 创建一个神经网络模型，包括嵌入层、多个ParallelTransformerBlock、LayerNorm层和线性层
    net = nn.Sequential(
        nn.Embedding(num_tokens, dim),  # 创建一个嵌入层，将输入的token映射到指定维度的向量
        *[
            Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult))
            for _ in range(depth)  # 创建指定数量的ParallelTransformerBlock，并将其作为Residual块添加到模型中
        ],
        LayerNorm(dim),  # 添加LayerNorm层，用于归一化模型输出
        nn.Linear(dim, num_tokens, bias=False)  # 添加线性层，将模型输出映射到指定数量的token
    )

    # 将嵌入层的权重赋值给线性层的权重，实现权重共享
    net[-1].weight = net[0].weight

    # 对嵌入层的权重进行正态分布初始化
    nn.init.normal_(net[0].weight, std=0.02)
    
    # 返回创建的神经网络模型
    return net
```