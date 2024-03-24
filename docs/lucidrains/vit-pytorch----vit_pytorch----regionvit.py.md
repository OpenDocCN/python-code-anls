# `.\lucidrains\vit-pytorch\vit_pytorch\regionvit.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 einops 库中导入 rearrange 函数和 Rearrange、Reduce 类
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
# 从 torch 库中导入 nn.functional 模块，并重命名为 F

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将变量转换为元组，如果不是元组则重复 length 次
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 判断一个数是否可以被另一个数整除
def divisible_by(val, d):
    return (val % d) == 0

# 辅助类

# 下采样类
class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)

# PEG 类
class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# transformer 类

# 前馈网络
def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim, 1)
    )

# 注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, rel_pos_bias = None):
        h = self.heads

        # prenorm

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add relative positional bias for local tokens

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# R2LTransformer 类
class R2LTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        window_size,
        depth = 4,
        heads = 4,
        dim_head = 32,
        attn_dropout = 0.,
        ff_dropout = 0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.window_size = window_size
        rel_positions = 2 * window_size - 1
        self.local_rel_pos_bias = nn.Embedding(rel_positions ** 2, heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout)
            ]))
    # 定义一个前向传播函数，接受本地 tokens 和区域 tokens 作为输入
    def forward(self, local_tokens, region_tokens):
        # 获取本地 tokens 的设备信息
        device = local_tokens.device
        # 获取本地 tokens 和区域 tokens 的高度和宽度
        lh, lw = local_tokens.shape[-2:]
        rh, rw = region_tokens.shape[-2:]
        # 计算窗口大小
        window_size_h, window_size_w = lh // rh, lw // rw

        # 重排本地 tokens 和区域 tokens 的维度
        local_tokens = rearrange(local_tokens, 'b c h w -> b (h w) c')
        region_tokens = rearrange(region_tokens, 'b c h w -> b (h w) c')

        # 计算本地相对位置偏差
        h_range = torch.arange(window_size_h, device = device)
        w_range = torch.arange(window_size_w, device = device)
        grid_x, grid_y = torch.meshgrid(h_range, w_range, indexing = 'ij')
        grid = torch.stack((grid_x, grid_y))
        grid = rearrange(grid, 'c h w -> c (h w)')
        grid = (grid[:, :, None] - grid[:, None, :]) + (self.window_size - 1)
        bias_indices = (grid * torch.tensor([1, self.window_size * 2 - 1], device = device)[:, None, None]).sum(dim = 0)
        rel_pos_bias = self.local_rel_pos_bias(bias_indices)
        rel_pos_bias = rearrange(rel_pos_bias, 'i j h -> () h i j')
        rel_pos_bias = F.pad(rel_pos_bias, (1, 0, 1, 0), value = 0)

        # 遍历 r2l transformer 层
        for attn, ff in self.layers:
            # 对区域 tokens 进行自注意力操作
            region_tokens = attn(region_tokens) + region_tokens

            # 将区域 tokens 连接到本地 tokens
            local_tokens = rearrange(local_tokens, 'b (h w) d -> b h w d', h = lh)
            local_tokens = rearrange(local_tokens, 'b (h p1) (w p2) d -> (b h w) (p1 p2) d', p1 = window_size_h, p2 = window_size_w)
            region_tokens = rearrange(region_tokens, 'b n d -> (b n) () d')

            # 对本地 tokens 进行自注意力操作，同时考虑区域 tokens
            region_and_local_tokens = torch.cat((region_tokens, local_tokens), dim = 1)
            region_and_local_tokens = attn(region_and_local_tokens, rel_pos_bias = rel_pos_bias) + region_and_local_tokens

            # 前馈神经网络
            region_and_local_tokens = ff(region_and_local_tokens) + region_and_local_tokens

            # 分离本地和区域 tokens
            region_tokens, local_tokens = region_and_local_tokens[:, :1], region_and_local_tokens[:, 1:]
            local_tokens = rearrange(local_tokens, '(b h w) (p1 p2) d -> b (h p1 w p2) d', h = lh // window_size_h, w = lw // window_size_w, p1 = window_size_h)
            region_tokens = rearrange(region_tokens, '(b n) () d -> b n d', n = rh * rw)

        # 重排本地 tokens 和区域 tokens 的维度
        local_tokens = rearrange(local_tokens, 'b (h w) c -> b c h w', h = lh, w = lw)
        region_tokens = rearrange(region_tokens, 'b (h w) c -> b c h w', h = rh, w = rw)
        # 返回本地 tokens 和区域 tokens
        return local_tokens, region_tokens
# 定义一个名为 RegionViT 的类，继承自 nn.Module
class RegionViT(nn.Module):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        *,
        dim = (64, 128, 256, 512),  # 定义维度的元组
        depth = (2, 2, 8, 2),  # 定义深度的元组
        window_size = 7,  # 定义窗口大小
        num_classes = 1000,  # 定义类别数量
        tokenize_local_3_conv = False,  # 是否使用局部 3 卷积
        local_patch_size = 4,  # 定义局部补丁大小
        use_peg = False,  # 是否使用 PEG
        attn_dropout = 0.,  # 注意力机制的 dropout
        ff_dropout = 0.,  # 前馈神经网络的 dropout
        channels = 3,  # 通道数
    ):
        super().__init__()  # 调用父类的初始化函数
        dim = cast_tuple(dim, 4)  # 将维度转换为元组
        depth = cast_tuple(depth, 4)  # 将深度转换为元组
        assert len(dim) == 4, 'dim needs to be a single value or a tuple of length 4'  # 断言维度长度为 4
        assert len(depth) == 4, 'depth needs to be a single value or a tuple of length 4'  # 断言深度长度为 4

        self.local_patch_size = local_patch_size  # 设置局部补丁大小

        region_patch_size = local_patch_size * window_size  # 计算区域补丁大小
        self.region_patch_size = local_patch_size * window_size  # 设置区域补丁大小

        init_dim, *_, last_dim = dim  # 解构维度元组

        # 定义局部和区域编码器

        if tokenize_local_3_conv:
            self.local_encoder = nn.Sequential(
                nn.Conv2d(3, init_dim, 3, 2, 1),
                nn.LayerNorm(init_dim),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, 3, 2, 1),
                nn.LayerNorm(init_dim),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, 3, 1, 1)
            )
        else:
            self.local_encoder = nn.Conv2d(3, init_dim, 8, 4, 3)

        self.region_encoder = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = region_patch_size, p2 = region_patch_size),
            nn.Conv2d((region_patch_size ** 2) * channels, init_dim, 1)
        )

        # 定义层

        current_dim = init_dim  # 初始化当前维度
        self.layers = nn.ModuleList([])  # 初始化层列表

        for ind, dim, num_layers in zip(range(4), dim, depth):
            not_first = ind != 0  # 判断是否为第一层
            need_downsample = not_first  # 是否需要下采样
            need_peg = not_first and use_peg  # 是否需要 PEG

            self.layers.append(nn.ModuleList([
                Downsample(current_dim, dim) if need_downsample else nn.Identity(),  # 如果需要下采样则使用 Downsample，否则使用恒等映射
                PEG(dim) if need_peg else nn.Identity(),  # 如果需要 PEG 则使用 PEG，否则使用恒等映射
                R2LTransformer(dim, depth = num_layers, window_size = window_size, attn_dropout = attn_dropout, ff_dropout = ff_dropout)  # 使用 R2LTransformer
            ]))

            current_dim = dim  # 更新当前维度

        # 定义最终的 logits

        self.to_logits = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),  # 对特征进行降维
            nn.LayerNorm(last_dim),  # 对最后一个维度进行 LayerNorm
            nn.Linear(last_dim, num_classes)  # 线性变换得到类别数量
        )

    # 前向传播函数
    def forward(self, x):
        *_, h, w = x.shape  # 获取输入张量的高度和宽度
        assert divisible_by(h, self.region_patch_size) and divisible_by(w, self.region_patch_size), 'height and width must be divisible by region patch size'  # 断言高度和宽度必须能被区域补丁大小整除
        assert divisible_by(h, self.local_patch_size) and divisible_by(w, self.local_patch_size), 'height and width must be divisible by local patch size'  # 断言高度和宽度必须能被局部补丁大小整除

        local_tokens = self.local_encoder(x)  # 使用局部编码器对输入进行编码
        region_tokens = self.region_encoder(x)  # 使用区域编码器对输入进行编码

        for down, peg, transformer in self.layers:  # 遍历层列表
            local_tokens, region_tokens = down(local_tokens), down(region_tokens)  # 对局部和区域 tokens 进行下采样
            local_tokens = peg(local_tokens)  # 使用 PEG 对局部 tokens 进行处理
            local_tokens, region_tokens = transformer(local_tokens, region_tokens)  # 使用 transformer 对局部和区域 tokens 进行处理

        return self.to_logits(region_tokens)  # 返回最终的 logits
```