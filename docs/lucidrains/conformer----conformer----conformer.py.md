# `.\lucidrains\conformer\conformer\conformer.py`

```
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# 导入所需的库

# helper functions

# 定义辅助函数

def exists(val):
    return val is not None

# 检查值是否存在的函数

def default(val, d):
    return val if exists(val) else d

# 如果值存在则返回该值，否则返回默认值的函数

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# 计算卷积核大小的 padding 值的函数

# helper classes

# 定义辅助类

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

# Swish 激活函数类的定义

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

# GLU 激活函数类的定义

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# 深度卷积类的定义

# attention, feedforward, and conv module

# 注意力、前馈和卷积模块的定义

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

# 缩放类的定义

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 预归一化类的定义

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

# 注意力机制类的定义

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):  # 定义神经网络模型的初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.net = nn.Sequential(  # 创建一个包含多个神经网络层的序列容器
            nn.Linear(dim, dim * mult),  # 添加线性层，输入维度为dim，输出维度为dim * mult
            Swish(),  # 使用Swish激活函数
            nn.Dropout(dropout),  # 添加Dropout层，以减少过拟合
            nn.Linear(dim * mult, dim),  # 添加线性层，输入维度为dim * mult，输出维度为dim
            nn.Dropout(dropout)  # 再次添加Dropout层
        )

    def forward(self, x):  # 定义神经网络模型的前向传播方法
        return self.net(x)  # 返回神经网络模型对输入x的输出结果
# 定义一个 ConformerConvModule 类，继承自 nn.Module
class ConformerConvModule(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 计算内部维度
        inner_dim = dim * expansion_factor
        # 计算填充大小
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        # 定义网络结构
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # LayerNorm 层
            Rearrange('b n c -> b c n'),  # 重新排列维度
            nn.Conv1d(dim, inner_dim * 2, 1),  # 一维卷积层
            GLU(dim=1),  # GLU 激活函数
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),  # 深度卷积层
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),  # BatchNorm1d 层或 Identity 层
            Swish(),  # Swish 激活函数
            nn.Conv1d(inner_dim, dim, 1),  # 一维卷积层
            Rearrange('b c n -> b n c'),  # 重新排列维度
            nn.Dropout(dropout)  # Dropout 层
        )

    # 前向传播方法
    def forward(self, x):
        return self.net(x)

# 定义一个 ConformerBlock 类，继承自 nn.Module
class ConformerBlock(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 定义网络结构
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)  # FeedForward 层
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)  # Attention 层
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)  # ConformerConvModule 层
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)  # FeedForward 层

        self.attn = PreNorm(dim, self.attn)  # PreNorm 层
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))  # Scale 层
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))  # Scale 层

        self.post_norm = nn.LayerNorm(dim)  # LayerNorm 层

    # 前向传播方法
    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

# 定义一个 Conformer 类，继承自 nn.Module
class Conformer(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        # 调用父类的初始化方法
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        # 循环创建 ConformerBlock 层，并添加到 layers 中
        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal
            ))

    # 前向传播方法
    def forward(self, x):
        # 遍历 layers 中的每个 ConformerBlock 层，并进行前向传播
        for block in self.layers:
            x = block(x)

        return x
```