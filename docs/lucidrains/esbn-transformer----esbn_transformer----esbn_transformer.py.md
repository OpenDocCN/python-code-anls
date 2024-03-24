# `.\lucidrains\esbn-transformer\esbn_transformer\esbn_transformer.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 从 einops 库中导入 rearrange、repeat、reduce 函数
from einops import rearrange, repeat, reduce

# 从 einops.layers.torch 库中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 返回指定数据类型的最大负值的函数
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# 对所有张量进行重排列的函数
def rearrange_all(tensors, *args, **kwargs):
    return map(lambda t: rearrange(t, *args, **kwargs), tensors)

# 前馈网络

# 分组层归一化类
class GroupLayerNorm(nn.Module):
    def __init__(self, dim, groups = 1, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.groups = groups
        self.g = nn.Parameter(torch.ones(1, groups, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, groups, dim, 1))

    def forward(self, x):
        x = rearrange(x, 'b (g d) n -> b g d n', g = self.groups)
        std = torch.var(x, dim = 2, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 2, keepdim = True)
        out = (x - mean) / (std + self.eps) * self.g + self.b
        return rearrange(out, 'b g d n -> b (g d) n')

# 预归一化类
class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn,
        groups = 1
    ):
        super().__init__()
        self.norm = GroupLayerNorm(dim, groups = groups)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 前馈网络类
class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        groups = 1
    ):
        super().__init__()
        input_dim = dim * groups
        hidden_dim = dim * mult * groups

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, groups = groups),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 1, groups = groups)
        )

    def forward(self, x):
        return self.net(x)

# 注意力机制类
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        groups = 1
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.groups = groups
        self.heads = heads
        self.causal = causal
        input_dim = dim * groups
        inner_dim = dim_head * heads * groups

        self.to_q = nn.Conv1d(input_dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv1d(input_dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, input_dim, 1)

    def forward(self, x, mask = None, context = None):
        n, device, h, g, causal = x.shape[2], x.device, self.heads, self.groups, self.causal
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = 1))
        q, k, v = rearrange_all((q, k, v), 'b (g h d) n -> (b g h) n d', g = g, h = h)

        q = q * self.scale

        sim = einsum('b i d, b j d -> b i j', q, k)

        if g > 1:
            # 在存在符号的情况下，允许网络使用来自感官侧的注意力矩阵绑定符号
            sim = rearrange(sim, '(b g h) i j -> b g h i j', g = g, h = h)
            sim = sim.cumsum(dim = 1)
            sim = rearrange(sim, 'b g h i j -> (b g h) i j')

        if exists(mask):
            mask = repeat(mask, 'b n -> (b g h) n', h = h, g = g)
            mask = rearrange(mask, 'b n -> b n ()') * rearrange(mask, 'b n -> b () n')
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = torch.ones((n, n), device = device).triu(1).bool()
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b g h) n d -> b (g h d) n', h = h, g = g)
        return self.to_out(out)

# Transformer 块类
class TransformerBlock(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        dim,  # 输入维度
        causal = False,  # 是否使用因果注意力
        dim_head = 64,  # 注意力头的维度
        heads = 8,  # 注意力头的数量
        ff_mult = 4,  # FeedForward 层的倍数
        groups = 1  # 分组数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化注意力层，包括预处理和注意力计算
        self.attn = PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, groups = groups), groups = groups)
        # 初始化前馈神经网络层，包括预处理和前馈计算
        self.ff = PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, groups = groups), groups = groups)
    
    # 前向传播函数
    def forward(self, x, mask = None):
        # 使用注意力层处理输入数据，并将结果与输入相加
        x = self.attn(x, mask = mask) + x
        # 使用前馈神经网络层处理数据，并将结果与输入相加
        x = self.ff(x) + x
        # 返回处理后的数据
        return x
# 主类定义

class EsbnTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 维度
        depth,  # 深度
        num_tokens,  # 令牌数量
        max_seq_len,  # 最大序列长度
        causal = False,  # 是否因果
        dim_head = 64,  # 头部维度
        heads = 8,  # 头部数量
        ff_mult = 4  # FeedForward 层倍增因子
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)  # 令牌嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)  # 位置嵌入层

        self.layers = nn.ModuleList([])
        self.pre_transformer_block = TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads)  # 前置 Transformer 块

        self.symbols = nn.Parameter(torch.randn(max_seq_len, dim))  # 符号参数

        for _ in range(depth):
            self.layers.append(TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads, groups = 2))  # 添加 Transformer 块到层列表

        self.post_transformer_block = TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads,)  # 后置 Transformer 块

        self.to_logits = nn.Sequential(
            Rearrange('b d n -> b n d'),  # 重新排列张量维度
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_tokens)  # 线性层
        )

    def forward(self, x, mask = None):
        b, n, d, device = *x.shape, self.dim, x.device
        x = self.token_emb(x)  # 通过令牌嵌入层获取输入张量的嵌入表示

        pos_emb = self.pos_emb(torch.arange(n, device = device))  # 获取位置嵌入
        pos_emb = rearrange(pos_emb, 'n d -> () n d')  # 重新排列位置嵌入张量维度

        x = x + pos_emb  # 将位置嵌入加到输入张量上
        x = rearrange(x, 'b n d -> b d n')  # 重新排列张量维度

        x = self.pre_transformer_block(x, mask = mask)  # 前置 Transformer 块处理输入张量

        x = rearrange(x, 'b d n -> b () d n')  # 重新排列张量维度
        symbols = self.symbols[:, :n]  # 获取符号参数

        symbols = repeat(symbols, 'n d -> b () d n', b = b)  # 重复符号参数以匹配输入张量维度
        x = torch.cat((x, symbols), dim = 1)  # 拼接张量
        x = rearrange(x, 'b ... n -> b (...) n')  # 重新排列张量维度

        for block in self.layers:
            x = block(x, mask = mask)  # 遍历并应用每个 Transformer 块

        x = rearrange(x, 'b (s d) n -> b s d n', s = 2)  # 重新��列张量维度
        x = x[:, 1]  # 获取特定索引的张量

        x = self.post_transformer_block(x, mask = mask)  # 后置 Transformer 块处理张量
        return self.to_logits(x)  # 返回处理后的张量
```