# `.\lucidrains\uniformer-pytorch\uniformer_pytorch\uniformer_pytorch.py`

```
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Reduce

# helpers

# 检查值是否存在的辅助函数
def exists(val):
    return val is not None

# classes

# LayerNorm 类
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        # 计算标准差
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        # 计算均值
        mean = torch.mean(x, dim = 1, keepdim = True)
        # LayerNorm 操作
        return (x - mean) / (std + self.eps) * self.g + self.b

# FeedForward 函数
def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv3d(dim * mult, dim, 1)
    )

# MHRAs (multi-head relation aggregators)

# LocalMHRA 类
class LocalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        # 使用 BatchNorm3d 代替 LayerNorm
        self.norm = nn.BatchNorm3d(dim)

        # 仅使用值，因为注意力矩阵由卷积处理
        self.to_v = nn.Conv3d(dim, inner_dim, 1, bias = False)

        # 通过相对位置聚合
        self.rel_pos = nn.Conv3d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)

        # 合并所有头部的输出
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        b, c, *_, h = *x.shape, self.heads

        # 转换为值
        v = self.to_v(x)

        # 分割头部
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h)

        # 通过相对位置聚合
        out = self.rel_pos(v)

        # 合并头部
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)
        return self.to_out(out)

# GlobalMHRA 类
class GlobalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        shape, h = x.shape, self.heads

        x = rearrange(x, 'b c ... -> b c (...)')

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 注意力
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h = h)

        out = self.to_out(out)
        return out.view(*shape)

# Transformer 类
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mhsa_type = 'g',
        local_aggr_kernel = 5,
        dim_head = 64,
        ff_mult = 4,
        ff_dropout = 0.,
        attn_dropout = 0.
    # 调用父类的构造函数初始化对象
    ):
        super().__init__()

        # 初始化一个空的神经网络模块列表
        self.layers = nn.ModuleList([])

        # 循环创建指定数量的层
        for _ in range(depth):
            # 根据不同的注意力类型创建不同的注意力模块
            if mhsa_type == 'l':
                attn = LocalMHRA(dim, heads = heads, dim_head = dim_head, local_aggr_kernel = local_aggr_kernel)
            elif mhsa_type == 'g':
                attn = GlobalMHRA(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            # 将卷积层、注意力层和前馈网络层组成一个模块列表，并添加到神经网络模块列表中
            self.layers.append(nn.ModuleList([
                nn.Conv3d(dim, dim, 3, padding = 1),
                attn,
                FeedForward(dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    # 前向传播函数
    def forward(self, x):
        # 遍历每个层，依次进行前向传播
        for dpe, attn, ff in self.layers:
            # 执行卷积层、注意力层和前馈网络层的操作，并将结果与输入相加
            x = dpe(x) + x
            x = attn(x) + x
            x = ff(x) + x

        # 返回最终的输出结果
        return x
# 主类定义
class Uniformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        num_classes,  # 类别数量
        dims = (64, 128, 256, 512),  # 不同层的维度
        depths = (3, 4, 8, 3),  # 不同层的深度
        mhsa_types = ('l', 'l', 'g', 'g'),  # 多头自注意力类型
        local_aggr_kernel = 5,  # 局部聚合核大小
        channels = 3,  # 输入通道数
        ff_mult = 4,  # FeedForward 层的倍数
        dim_head = 64,  # 头部维度
        ff_dropout = 0.,  # FeedForward 层的 dropout
        attn_dropout = 0.  # 注意力层的 dropout
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        # 将输入视频转换为 tokens
        self.to_tokens = nn.Conv3d(channels, init_dim, (3, 4, 4), stride = (2, 4, 4), padding = (1, 0, 0))

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))
        mhsa_types = tuple(map(lambda t: t.lower(), mhsa_types))

        self.stages = nn.ModuleList([])

        # 遍历不同层的深度和多头自注意力类型
        for ind, (depth, mhsa_type) in enumerate(zip(depths, mhsa_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            # 添加 Transformer 层和下采样层到 stages
            self.stages.append(nn.ModuleList([
                Transformer(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mhsa_type = mhsa_type,
                    ff_mult = ff_mult,
                    ff_dropout = ff_dropout,
                    attn_dropout = attn_dropout
                ),
                nn.Sequential(
                    nn.Conv3d(stage_dim, dims[ind + 1], (1, 2, 2), stride = (1, 2, 2)),
                    LayerNorm(dims[ind + 1]),
                ) if not is_last else None
            ]))

        # 输出层
        self.to_logits = nn.Sequential(
            Reduce('b c t h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )

    # 前向传播函数
    def forward(self, video):
        x = self.to_tokens(video)

        # 遍历不同层的 Transformer 和下采样层
        for transformer, conv in self.stages:
            x = transformer(x)

            if exists(conv):
                x = conv(x)

        return self.to_logits(x)
```