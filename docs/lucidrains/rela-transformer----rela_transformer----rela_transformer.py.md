# `.\lucidrains\rela-transformer\rela_transformer\rela_transformer.py`

```py
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 torch 中导入 nn、einsum 模块
from torch import nn, einsum
# 从 einops 中导入 rearrange、repeat 函数
from einops import rearrange, repeat

# 定义辅助函数 exists，用于检查值是否存在
def exists(val):
    return val is not None

# 定义 GatedRMSNorm 类，继承自 nn.Module
class GatedRMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-8
    ):
        super().__init__()
        # 初始化缩放因子 scale
        self.scale = dim ** -0.5
        # 初始化 eps
        self.eps = eps
        # 初始化可学习参数 w 和 g
        self.w = nn.Parameter(torch.ones(dim))
        self.g = nn.Parameter(torch.ones(dim))

    # 前向传播函数
    def forward(self, x):
        # 计算输入 x 的 L2 范数，并进行缩放
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        # 对输入 x 进行归一化处理
        normed_x = x / norm.clamp(min = self.eps) * self.g
        # 返回经过门控的 RMS 归一化结果
        return normed_x * (x * self.w).sigmoid()

# 定义 FeedForward 函数，返回一个包含线性层和 GELU 激活函数的序列
def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

# 定义 ReLA 类，继承自 nn.Module
class ReLA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        causal = True,
        dim_head = 64,
        heads = 8,
        num_memory_kv = 0,
        relu_squared = False
    ):
        super().__init__()
        # 初始化头数和内部维度
        self.heads = heads
        inner_dim = dim_head * heads
        # 初始化缩放因子 scale
        self.scale = dim_head ** -0.5
        # 初始化是否是因果关系
        self.causal = causal
        # 初始化是否对激活函数进行平方操作
        self.relu_squared = relu_squared
        # 初始化 RMS 归一化层
        self.norm = GatedRMSNorm(dim)

        # 初始化 q、k、v 的线性层
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 初始化记忆键值对
        self.mem_k = nn.Parameter(torch.randn(num_memory_kv, inner_dim))
        self.mem_v = nn.Parameter(torch.randn(num_memory_kv, inner_dim))

        # 初始化值的 RMS 归一化层和输出层
        self.norm_values = GatedRMSNorm(dim_head)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    # 前向传播函数
    def forward(self, x, mask = None):
        # 获取输入 x 的批量大小和设备信息
        b, device = x.shape[0], x.device
        # 对输入 x 进行 RMS 归一化处理
        x = self.norm(x)
        h = self.heads

        # 将输入 x 经过 qkv 线性层并分块
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 将记忆键值对进行扩展并拼接到 k、v 中
        mem_k, mem_v = map(lambda t: repeat(t, 'n d -> b n d', b = b), (self.mem_k, self.mem_v))
        k = torch.cat((mem_k, k), dim = 1)
        v = torch.cat((mem_v, v), dim = 1)

        # 重排 q、k、v 的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 对 q 进行缩放
        q = q * self.scale
        # 计算注意力分数
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 计算注意力值，并进行 ReLU 激活
        attn = F.relu(sim)

        # 如果设置了 relu_squared 标志，则对注意力值进行平方操作
        if self.relu_squared:
            attn = attn ** 2

        # 如果存在 mask，则进行 mask 操作
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            attn = attn.masked_fill(~mask, 0.)

        # 如果是因果关系，进行因果 mask 操作
        if self.causal:
            i, j = attn.shape[-2:]
            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            attn = attn.masked_fill(causal_mask, 0.)

        # 计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = self.norm_values(out)

        # 重排输出维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义 ReLATransformer 类，继承自 nn.Module
class ReLATransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        causal = True,
        heads = 8,
        dim_head = 64,
        num_memory_kv = 0,
        no_ff = False,
        ff_mult = 4,
        relu_squared = False
    ):
        super().__init__()
        # 初始化最大序列长度、token 词嵌入和位置嵌入
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 初始化层列表
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ReLA(dim = dim, relu_squared = relu_squared, heads = heads, dim_head = dim_head, num_memory_kv = num_memory_kv, causal = causal),
                FeedForward(dim = dim, mult = ff_mult) if not no_ff else None
            ]))

        # 初始化输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )
    # 定义前向传播函数，接受输入张量 x 和掩码 mask，默认为 None
    def forward(self, x, mask = None):
        # 获取输入张量 x 的维度 n 和设备信息
        n, device = x.shape[1], x.device
        # 对输入张量 x 进行 token embedding
        x = self.token_emb(x)
        # 根据输入张量 x 的长度 n，生成位置编码 pos_emb
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        # 将位置编码与 token embedding 相加
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        # 遍历每个注意力层和前馈层
        for attn, ff in self.layers:
            # 使用注意力层处理输入张量 x，并将结果与原始输入相加
            x = attn(x, mask = mask) + x

            # 如果前馈层存在
            if exists(ff):
                # 使用前馈层处理输入张量 x，并将结果与原始输入相加
                x = ff(x) + x

        # 将处理后的张量 x 转换为最终的输出 logits
        return self.to_logits(x)
```