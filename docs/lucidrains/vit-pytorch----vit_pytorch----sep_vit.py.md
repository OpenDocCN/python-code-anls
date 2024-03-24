# `.\lucidrains\vit-pytorch\vit_pytorch\sep_vit.py`

```
# 导入必要的库
from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# 辅助函数

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 辅助类

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class OverlappingPatchEmbed(nn.Module):
    def __init__(self, dim_in, dim_out, stride = 2):
        super().__init__()
        kernel_size = stride * 2 - 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride = stride, padding = padding)

    def forward(self, x):
        return self.conv(x)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# 前馈网络

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 注意力机制

class DSSA(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.norm = ChanLayerNorm(dim)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)

        # 窗口标记

        self.window_tokens = nn.Parameter(torch.randn(dim))

        # 窗口标记的预处理和非线性变换
        # 然后将窗口标记投影到查询和键

        self.window_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h = heads),
        )

        # 窗口注意力

        self.window_attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        """
        einstein notation

        b - batch
        c - channels
        w1 - window size (height)
        w2 - also window size (width)
        i - sequence dimension (source)
        j - sequence dimension (target dimension to be reduced)
        h - heads
        x - height of feature map divided by window size
        y - width of feature map divided by window size
        """

        # 获取输入张量的形状信息
        batch, height, width, heads, wsz = x.shape[0], *x.shape[-2:], self.heads, self.window_size
        # 检查高度和宽度是否可以被窗口大小整除
        assert (height % wsz) == 0 and (width % wsz) == 0, f'height {height} and width {width} must be divisible by window size {wsz}'
        # 计算窗口数量
        num_windows = (height // wsz) * (width // wsz)

        # 对输入张量进行归一化处理
        x = self.norm(x)

        # 将窗口折叠进行“深度”注意力 - 不确定为什么它被命名为深度，当它只是“窗口化”注意力时
        x = rearrange(x, 'b c (h w1) (w w2) -> (b h w) c (w1 w2)', w1 = wsz, w2 = wsz)

        # 添加窗口标记
        w = repeat(self.window_tokens, 'c -> b c 1', b = x.shape[0])
        x = torch.cat((w, x), dim = -1)

        # 为查询、键、值进行投影
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # 分离头部
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # 缩放
        q = q * self.scale

        # 相似度
        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # 注意力
        attn = self.attend(dots)

        # 聚合值
        out = torch.matmul(attn, v)

        # 分离窗口标记和窗口化特征图
        window_tokens, windowed_fmaps = out[:, :, 0], out[:, :, 1:]

        # 如果只有一个窗口，则提前返回
        if num_windows == 1:
            fmap = rearrange(windowed_fmaps, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
            return self.to_out(fmap)

        # 执行点对点注意力，这是论文中的主要创新
        window_tokens = rearrange(window_tokens, '(b x y) h d -> b h (x y) d', x = height // wsz, y = width // wsz)
        windowed_fmaps = rearrange(windowed_fmaps, '(b x y) h n d -> b h (x y) n d', x = height // wsz, y = width // wsz)

        # 窗口化查询和键（在进行预归一化激活之前）
        w_q, w_k = self.window_tokens_to_qk(window_tokens).chunk(2, dim = -1)

        # 缩放
        w_q = w_q * self.scale

        # 相似度
        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)

        w_attn = self.window_attend(w_dots)

        # 聚合来自“深度”注意力步骤的特征图（论文中最有趣的部分，我以前没有见过）
        aggregated_windowed_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, windowed_fmaps)

        # 折叠回窗口，然后组合头部以进行聚合
        fmap = rearrange(aggregated_windowed_fmap, 'b h (x y) (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
        return self.to_out(fmap)
class Transformer(nn.Module):
    # 定义 Transformer 类，继承自 nn.Module
    def __init__(
        self,
        dim,
        depth,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        dropout = 0.,
        norm_output = True
    ):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            # 遍历深度次数
            self.layers.append(nn.ModuleList([
                DSSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout),
            ]))
            # 在 layers 中添加 DSSA 和 FeedForward 模块

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()
        # 如果 norm_output 为 True，则使用 ChanLayerNorm，否则使用 nn.Identity

    def forward(self, x):
        # 前向传播函数
        for attn, ff in self.layers:
            # 遍历 layers 中的模块
            x = attn(x) + x
            # 对输入 x 进行注意力操作
            x = ff(x) + x
            # 对输入 x 进行前馈操作

        return self.norm(x)
        # 返回经过规范化的结果

class SepViT(nn.Module):
    # 定义 SepViT 类，继承自 nn.Module
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        window_size = 7,
        dim_head = 32,
        ff_mult = 4,
        channels = 3,
        dropout = 0.
    ):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'
        # 断言 depth 是元组类型，用于指示每个阶段的 transformer 块数量

        num_stages = len(depth)
        # 获取深度的长度

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (channels, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        # 计算每个阶段的维度

        strides = (4, *((2,) * (num_stages - 1)))
        # 定义步长

        hyperparams_per_stage = [heads, window_size]
        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))
        # 处理每个阶段的超参数

        self.layers = nn.ModuleList([])

        for ind, ((layer_dim_in, layer_dim), layer_depth, layer_stride, layer_heads, layer_window_size) in enumerate(zip(dim_pairs, depth, strides, *hyperparams_per_stage)):
            # 遍历每个阶段的参数
            is_last = ind == (num_stages - 1)

            self.layers.append(nn.ModuleList([
                OverlappingPatchEmbed(layer_dim_in, layer_dim, stride = layer_stride),
                PEG(layer_dim),
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_mult = ff_mult, dropout = dropout, norm_output = not is_last),
            ]))
            # 在 layers 中添加 OverlappingPatchEmbed、PEG 和 Transformer 模块

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )
        # 定义 MLP 头部模块

    def forward(self, x):
        # 前向传播函数
        for ope, peg, transformer in self.layers:
            # 遍历 layers 中的模块
            x = ope(x)
            # 对输入 x 进行 OverlappingPatchEmbed 操作
            x = peg(x)
            # 对输入 x 进行 PEG 操作
            x = transformer(x)
            # 对输入 x 进行 Transformer 操作

        return self.mlp_head(x)
        # 返回经过 MLP 头部处理的结果
```