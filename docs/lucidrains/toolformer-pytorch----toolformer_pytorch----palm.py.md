# `.\lucidrains\toolformer-pytorch\toolformer_pytorch\palm.py`

```
import torch
from torch import nn, einsum
from einops import rearrange

from x_clip.tokenizer import tokenizer

# 导入所需的库

# helpers

# 定义一个辅助函数，用于检查值是否存在
def exists(val):
    return val is not None

# normalization

# 定义一个 RMS 归一化层
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

# 定义一个旋转位置嵌入层
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

# 旋转半个位置
def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# all we need

# 定义并行 Transformer 块
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = RMSNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        # 注册缓存的因果掩码和旋转嵌入
        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    # 获取因果掩码
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
# Transformer 类定义
class Transformer(nn.Module):
    # 初始化函数，接受维度、深度、头数、头维度和前馈网络倍数作为参数
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        ff_mult = 4,
    ):
        super().__init__()
        # 初始化一个空的模块列表
        self.layers = nn.ModuleList([])

        # 循环创建指定深度的 ParallelTransformerBlock，并添加到模块列表中
        for _ in range(depth):
            self.layers.append(
                ParallelTransformerBlock(dim, dim_head, heads, ff_mult), 
            )

    # 前向传播函数
    def forward(self, x):
        # 遍历模块列表中的每个块，对输入进行变换并加上原始输入
        for block in self.layers:
            x = block(x) + x
        return x


# PaLM 类定义
class PaLM(nn.Module):
    # 初始化函数，接受维度、深度、标记数、头维度、头数和前馈网络倍数作为参数
    def __init__(
        self, 
        dim, 
        depth, 
        num_tokens=tokenizer.vocab_size,
        dim_head=64, 
        heads=8, 
        ff_mult=4,
    ):
        super().__init__()
        # 创建一个嵌入层，将标记映射到指定维度
        self.emb = nn.Embedding(num_tokens, dim)

        # 创建一个 Transformer 模型
        self.transformer = Transformer(dim, depth, heads, dim_head, ff_mult)

        # 创建一个输出层，包括 RMSNorm 层和线性层
        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    # 前向传播函数
    def forward(self, x):
        # 将输入通过嵌入层映射到指定维度
        x = self.emb(x)
        # 将映射后的输入通过 Transformer 模型进行变换
        x = self.transformer(x)
        # 将变换后的结果通过输出层得到最终的 logits
        return self.to_logits(x)

# 主函数入口
if __name__ == "__main__":
    # 创建一个 PaLM 模型实例
    palm = PaLM(
        num_tokens = 20000,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
    )

    # 生成一个随机的标记序列
    tokens = torch.randint(0, 20000, (1, 512))
    # 将标记序列输入到 PaLM 模型中得到 logits
    logits = palm(tokens)
    # 打印 logits 的形状
    print(logits.shape)
```