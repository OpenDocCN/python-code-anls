# `.\lucidrains\vit-pytorch\vit_pytorch\max_vit.py`

```
# 导入必要的库
from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# 辅助函数

# 检查变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将变量转换为元组，如果不是元组则重复多次
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 辅助类

# 残差连接
class Residual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# MBConv

# Squeeze-and-Excitation 模块
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

# MBConv 残差块
class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

# 随机丢弃采样
class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

# MBConv 构建函数
def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# 注意力相关类

# 注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言维度应该能够被每个头的维度整除
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        # 计算头的数量
        self.heads = dim // dim_head
        # 缩放因子
        self.scale = dim_head ** -0.5

        # LayerNorm 层
        self.norm = nn.LayerNorm(dim)
        # 线性变换，将输入维度转换为查询、键、值的维度
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        # 注意力机制
        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),  # Softmax 激活函数
            nn.Dropout(dropout)  # Dropout 层
        )

        # 输出层
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),  # 线性变换
            nn.Dropout(dropout)  # Dropout 层
        )

        # 相对位置偏置

        # Embedding 层，用于存储相对位置偏置
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        # 计算相对位置偏置
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        # 注册缓冲区，存储相对位置索引
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        # 获取输入张量的形状信息
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # LayerNorm 层
        x = self.norm(x)

        # 展开张量
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # 为查询、键、值进行投影
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 分割头
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 缩放
        q = q * self.scale

        # 计算相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 添加相对位置偏置
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # 注意力机制
        attn = self.attend(sim)

        # 聚合
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并头
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # 合并头的输出
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)
# 定义一个名为 MaxViT 的神经网络模型类，继承自 nn.Module
class MaxViT(nn.Module):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 断言 depth 是一个元组，用于指定每个阶段的 transformer 块数量
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # 卷积 stem

        # 如果未指定 dim_conv_stem，则设为 dim
        dim_conv_stem = default(dim_conv_stem, dim)

        # 定义卷积 stem
        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # 变量

        # 计算阶段数量
        num_stages = len(depth)

        # 计算每个阶段的维度
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        # 初始化 layers 为一个空的 nn.ModuleList
        self.layers = nn.ModuleList([])

        # 为了高效的块状 - 网格状注意力，设置窗口大小的简写
        w = window_size

        # 遍历每个阶段
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                # 定义一个块
                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # 块状注意力
                    Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # 网格状注���力
                    Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                # 将块添加到 layers 中
                self.layers.append(block)

        # MLP 头部

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    # 前向传播函数
    def forward(self, x):
        # 经过卷积 stem
        x = self.conv_stem(x)

        # 遍历每个阶段的块
        for stage in self.layers:
            x = stage(x)

        # 经过 MLP 头部
        return self.mlp_head(x)
```