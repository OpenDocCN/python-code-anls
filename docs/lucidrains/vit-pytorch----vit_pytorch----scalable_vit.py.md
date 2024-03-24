# `.\lucidrains\vit-pytorch\vit_pytorch\scalable_vit.py`

```
# 导入必要的库
from functools import partial
import torch
from torch import nn

# 导入 einops 库中的函数和层
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将输入转换为元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 将输入转换为指定长度的元组
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 辅助类

# 通道层归一化
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

# 下采样
class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)

# 位置编码器
class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor = 4, dropout = 0.):
        super().__init__()
        inner_dim = dim * expansion_factor
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

# 可扩展的自注意力机制
class ScalableSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_key = 32,
        dim_value = 32,
        dropout = 0.,
        reduction_factor = 1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = ChanLayerNorm(dim)
        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, reduction_factor, stride = reduction_factor, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, reduction_factor, stride = reduction_factor, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        height, width, heads = *x.shape[-2:], self.heads

        x = self.norm(x)

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # 分割头部

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # 相似度

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 注意力权重

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 聚合数值

        out = torch.matmul(attn, v)

        # 合并头部

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = height, y = width)
        return self.to_out(out)

# 交互式窗口化自注意力机制
class InteractiveWindowedSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        heads = 8,
        dim_key = 32,
        dim_value = 32,
        dropout = 0.
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化头数和缩放因子
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.window_size = window_size
        # 初始化注意力机制和dropout层
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # 初始化通道层归一化和局部交互模块
        self.norm = ChanLayerNorm(dim)
        self.local_interactive_module = nn.Conv2d(dim_value * heads, dim_value * heads, 3, padding = 1)

        # 初始化转换层，将输入转换为查询、键和值
        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, 1, bias = False)

        # 初始化输出层，包括卷积层和dropout层
        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 获取输入张量的高度、宽度、头数和窗口大小
        height, width, heads, wsz = *x.shape[-2:], self.heads, self.window_size

        # 对输入张量进行归一化
        x = self.norm(x)

        # 计算窗口的高度和宽度
        wsz_h, wsz_w = default(wsz, height), default(wsz, width)
        assert (height % wsz_h) == 0 and (width % wsz_w) == 0, f'height ({height}) or width ({width}) of feature map is not divisible by the window size ({wsz_h}, {wsz_w})'

        # 将输入张量转换为查询、键和值
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # 获取局部交互模块的输出
        local_out = self.local_interactive_module(v)

        # 将查询、键和值分割成窗口（并拆分出头部）以进行有效的自注意力计算
        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d', h = heads, w1 = wsz_h, w2 = wsz_w), (q, k, v))

        # 计算相似度
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 注意力计算
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 聚合值
        out = torch.matmul(attn, v)

        # 将窗口重新整形为完整的特征图（并合并头部）
        out = rearrange(out, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz_h, y = width // wsz_w, w1 = wsz_h, w2 = wsz_w)

        # 添加局部交互模块的输出
        out = out + local_out

        return self.to_out(out)
class Transformer(nn.Module):
    # 定义 Transformer 类，继承自 nn.Module
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        ff_expansion_factor = 4,
        dropout = 0.,
        ssa_dim_key = 32,
        ssa_dim_value = 32,
        ssa_reduction_factor = 1,
        iwsa_dim_key = 32,
        iwsa_dim_value = 32,
        iwsa_window_size = None,
        norm_output = True
    ):
        # 初始化函数
        super().__init__()
        # 初始化 nn.ModuleList 用于存储 Transformer 层
        self.layers = nn.ModuleList([])
        # 循环创建 Transformer 层
        for ind in range(depth):
            # 判断是否为第一层
            is_first = ind == 0

            # 添加 Transformer 层的组件到 layers 中
            self.layers.append(nn.ModuleList([
                ScalableSelfAttention(dim, heads = heads, dim_key = ssa_dim_key, dim_value = ssa_dim_value, reduction_factor = ssa_reduction_factor, dropout = dropout),
                FeedForward(dim, expansion_factor = ff_expansion_factor, dropout = dropout),
                PEG(dim) if is_first else None,
                FeedForward(dim, expansion_factor = ff_expansion_factor, dropout = dropout),
                InteractiveWindowedSelfAttention(dim, heads = heads, dim_key = iwsa_dim_key, dim_value = iwsa_dim_value, window_size = iwsa_window_size, dropout = dropout)
            ]))

        # 初始化最后的归一化层
        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    # 前向传播函数
    def forward(self, x):
        # 遍历 Transformer 层
        for ssa, ff1, peg, iwsa, ff2 in self.layers:
            # Self-Attention 操作
            x = ssa(x) + x
            # FeedForward 操作
            x = ff1(x) + x

            # 如果存在 PEG 操作，则执行
            if exists(peg):
                x = peg(x)

            # Interactive Windowed Self-Attention 操作
            x = iwsa(x) + x
            # 再次 FeedForward 操作
            x = ff2(x) + x

        # 返回归一化后的结果
        return self.norm(x)

class ScalableViT(nn.Module):
    # 定义 ScalableViT 类，继承自 nn.Module
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        reduction_factor,
        window_size = None,
        iwsa_dim_key = 32,
        iwsa_dim_value = 32,
        ssa_dim_key = 32,
        ssa_dim_value = 32,
        ff_expansion_factor = 4,
        channels = 3,
        dropout = 0.
    ):
        # 初始化函数
        super().__init__()
        # 将图像转换为补丁
        self.to_patches = nn.Conv2d(channels, dim, 7, stride = 4, padding = 3)

        # 断言 depth 为元组，表示每个阶段的 Transformer 块数量
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # 计算每个阶段的维度
        num_stages = len(depth)
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))

        # 定义每个阶段的超参数
        hyperparams_per_stage = [
            heads,
            ssa_dim_key,
            ssa_dim_value,
            reduction_factor,
            iwsa_dim_key,
            iwsa_dim_value,
            window_size,
        ]

        # 将超参数转换为每个阶段的形式
        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        # 初始化 Transformer 层
        self.layers = nn.ModuleList([])

        # 遍历每个阶段的维度和超参数
        for ind, (layer_dim, layer_depth, layer_heads, layer_ssa_dim_key, layer_ssa_dim_value, layer_ssa_reduction_factor, layer_iwsa_dim_key, layer_iwsa_dim_value, layer_window_size) in enumerate(zip(dims, depth, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            # 添加 Transformer 层和下采样层到 layers 中
            self.layers.append(nn.ModuleList([
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_expansion_factor = ff_expansion_factor, dropout = dropout, ssa_dim_key = layer_ssa_dim_key, ssa_dim_value = layer_ssa_dim_value, ssa_reduction_factor = layer_ssa_reduction_factor, iwsa_dim_key = layer_iwsa_dim_key, iwsa_dim_value = layer_iwsa_dim_value, iwsa_window_size = layer_window_size, norm_output = not is_last),
                Downsample(layer_dim, layer_dim * 2) if not is_last else None
            ]))

        # MLP 头部
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        # 将图像转换为补丁
        x = self.to_patches(img)

        # 遍历每个 Transformer 层
        for transformer, downsample in self.layers:
            x = transformer(x)

            # 如果存在下采样层，则执行
            if exists(downsample):
                x = downsample(x)

        # 返回 MLP 头部的结果
        return self.mlp_head(x)
```