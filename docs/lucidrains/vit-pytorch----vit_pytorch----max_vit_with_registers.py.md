# `.\lucidrains\vit-pytorch\vit_pytorch\max_vit_with_registers.py`

```
# 导入必要的库
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# 辅助函数

# 检查变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将单个元素打包成指定模式的数据
def pack_one(x, pattern):
    return pack([x], pattern)

# 将数据解包成单个元素
def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# 将变量转换为元组，如果不是元组则重复多次
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length

# 辅助类

# 定义前馈神经网络结构
def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(dim * mult)
    return Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    )

# MBConv

# 定义Squeeze-and-Excitation模块
class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

# 定义MBConv残差模块
class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

# 定义DropSample模块
class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

# 定义MBConv模块
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

    net = Sequential(
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

# 定义注意力机制模块
class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        num_registers = 1
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言寄存器数量大于0
        assert num_registers > 0
        # 断言维度应该可以被每个头的维度整除
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        # 计算头的数量
        self.heads = dim // dim_head
        # 缩放因子
        self.scale = dim_head ** -0.5

        # LayerNorm层
        self.norm = nn.LayerNorm(dim)
        # 线性变换层，将输入维度转换为3倍的维度，用于计算Q、K、V
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        # 注意力机制
        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),  # Softmax激活函数
            nn.Dropout(dropout)  # Dropout层
        )

        # 输出层
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),  # 线性变换层
            nn.Dropout(dropout)  # Dropout层
        )

        # 相对位置偏差

        # 计算相对位置偏差的数量
        num_rel_pos_bias = (2 * window_size - 1) ** 2

        # Embedding层，用于存储相对位置偏差
        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        # 生成相对位置偏差的索引
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        # 对相对位置偏差索引进行填充
        rel_pos_indices = F.pad(rel_pos_indices, (num_registers, 0, num_registers, 0), value = num_rel_pos_bias)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        # 获取设备信息、头的数量、相对位置偏差索引
        device, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        # LayerNorm层
        x = self.norm(x)

        # 为查询、键、值进行投影
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 分割头
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 缩放
        q = q * self.scale

        # 计算相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 添加位置偏差
        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # 注意力机制
        attn = self.attend(sim)

        # 聚合
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并头部输出
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class MaxViT(Module):
    # 定义一个名为 MaxViT 的类，继承自 Module 类
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
        channels = 3,
        num_register_tokens = 4
    ):
        # 初始化函数，接受一系列参数
        super().__init__()
        # 调用父类的初始化函数

        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'
        assert num_register_tokens > 0
        # 断言语句，确保 depth 是元组类型，num_register_tokens 大于 0

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)
        # 如果 dim_conv_stem 为 None，则设置为 dim

        self.conv_stem = Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )
        # 创建一个包含两个卷积层的 Sequential 对象，作为卷积部分的网络结构

        # variables

        num_stages = len(depth)
        # 计算 depth 的长度，作为阶段数

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        # 计算每个阶段的维度

        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        # 将维度组成成对

        self.layers = nn.ModuleList([])
        # 创建一个空的 ModuleList 对象用于存储网络层

        # window size

        self.window_size = window_size
        # 设置窗口大小

        self.register_tokens = nn.ParameterList([])
        # 创建一个空的 ParameterList 对象用于存储注册令牌

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            # 遍历每个阶段
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim
                # 判断是否为当前阶段的第一个块

                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample = is_first,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                )
                # 创建一个 MBConv 对象

                block_attn = Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)
                block_ff = FeedForward(dim = layer_dim, dropout = dropout)
                # 创建注意力和前馈网络对象

                grid_attn = Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)
                grid_ff = FeedForward(dim = layer_dim, dropout = dropout)
                # 创建注意力和前馈网络对象

                register_tokens = nn.Parameter(torch.randn(num_register_tokens, layer_dim))
                # 创建一个随机初始化的注册令牌

                self.layers.append(ModuleList([
                    conv,
                    ModuleList([block_attn, block_ff]),
                    ModuleList([grid_attn, grid_ff])
                ]))
                # 将卷积层、注意力和前馈网络组成的模块列表添加到网络层中

                self.register_tokens.append(register_tokens)
                # 将注册令牌添加到注册令牌列表中

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )
        # 创建一个线性层用于分类
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的批量大小 b 和窗口大小 w
        b, w = x.shape[0], self.window_size

        # 对输入张量 x 进行卷积操作
        x = self.conv_stem(x)

        # 遍历每个层的操作，包括卷积、注意力机制和前馈网络
        for (conv, (block_attn, block_ff), (grid_attn, grid_ff)), register_tokens in zip(self.layers, self.register_tokens):
            # 对输入张量 x 进行卷积操作
            x = conv(x)

            # block-like attention

            # 对输入张量 x 进行重新排列操作，将其转换为多维矩阵
            x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)

            # 准备注册令牌

            # 将注册令牌进行重复操作，以匹配输入张量 x 的形状
            r = repeat(register_tokens, 'n d -> b x y n d', b = b, x = x.shape[1],y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            # 对输入张量 x 进行块状注意力操作，并与原始输入相加
            x = block_attn(x) + x
            # 对输入张量 x 进行块状前馈网络操作，并与原始输入相加
            x = block_ff(x) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')

            r = unpack_one(r, register_batch_ps, '* n d')

            # grid-like attention

            # 对输入张量 x 进行重新排列操作，将其转换为多维矩阵
            x = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)

            # 准备注册令牌

            # 对注册令牌进行降维操作，计算均值
            r = reduce(r, 'b x y n d -> b n d', 'mean')
            r = repeat(r, 'b n d -> b x y n d', x = x.shape[1], y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            # 对输入张量 x 进行网格状注意力操作，并与原始输入相加
            x = grid_attn(x) + x

            r, x = unpack(x, register_ps, 'b * d')

            # 对输入张量 x 进行网格状前馈网络操作，并与��始输入相加
            x = grid_ff(x) + x

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (w1 x) (w2 y)')

        # 返回经过 MLP 头部处理后的结果
        return self.mlp_head(x)
```