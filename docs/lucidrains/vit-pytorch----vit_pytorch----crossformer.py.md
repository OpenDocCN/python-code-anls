# `.\lucidrains\vit-pytorch\vit_pytorch\crossformer.py`

```py
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

# 辅助函数

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 交叉嵌入层

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_sizes,
        stride = 2
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # 计算每个尺度的维度
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        # 对输入进行卷积操作，并将结果拼接在一起
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

# 动态位置偏置

def DynamicPositionBias(dim):
    return nn.Sequential(
        nn.Linear(2, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, 1),
        Rearrange('... () -> ...')
    )

# transformer 类

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        attn_type,
        window_size,
        dim_head = 32,
        dropout = 0.
    ):
        super().__init__()
        assert attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

        # 位置

        self.dpb = DynamicPositionBias(dim // 4)

        # 计算和存储用于检索偏置的索引

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 解构 x 的形状，获取高度、宽度、头数、窗口大小和设备信息
        *_, height, width, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device

        # 对输入进行预处理
        x = self.norm(x)

        # 根据不同的注意力类型重新排列输入，以便进行短距离或长距离注意力
        if self.attn_type == 'short':
            x = rearrange(x, 'b d (h s1) (w s2) -> (b h w) d s1 s2', s1 = wsz, s2 = wsz)
        elif self.attn_type == 'long':
            x = rearrange(x, 'b d (l1 h) (l2 w) -> (b h w) d l1 l2', l1 = wsz, l2 = wsz)

        # 将输入转换为查询、键、值
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # 将查询、键、值按头数进行分割
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))
        q = q * self.scale

        # 计算注意力矩阵
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 添加动态位置偏置
        pos = torch.arange(-wsz, wsz + 1, device = device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        rel_pos = rearrange(rel_pos, 'c i j -> (i j) c')
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]
        sim = sim + rel_pos_bias

        # 注意力权重归一化
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # 合并头部
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = wsz, y = wsz)
        out = self.to_out(out)

        # 根据不同的注意力类型重新排列输出
        if self.attn_type == 'short':
            out = rearrange(out, '(b h w) d s1 s2 -> b d (h s1) (w s2)', h = height // wsz, w = width // wsz)
        elif self.attn_type == 'long':
            out = rearrange(out, '(b h w) d l1 l2 -> b d (l1 h) (l2 w)', h = height // wsz, w = width // wsz)

        return out
# 定义一个名为 Transformer 的神经网络模块
class Transformer(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        *,
        local_window_size,
        global_window_size,
        depth = 4,
        dim_head = 32,
        attn_dropout = 0.,
        ff_dropout = 0.,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化一个空的神经网络模块列表
        self.layers = nn.ModuleList([])

        # 循环创建指定深度的神经网络层
        for _ in range(depth):
            # 每层包含两个注意力机制和两个前馈神经网络
            self.layers.append(nn.ModuleList([
                Attention(dim, attn_type = 'short', window_size = local_window_size, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
                Attention(dim, attn_type = 'long', window_size = global_window_size, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout)
            ]))

    # 前向传播函数
    def forward(self, x):
        # 遍历每一层的注意力机制和前馈神经网络
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            # 执行短程注意力机制和前馈神经网络
            x = short_attn(x) + x
            x = short_ff(x) + x
            # 执行长程注意力机制和前馈神经网络
            x = long_attn(x) + x
            x = long_ff(x) + x

        # 返回处理后的数据
        return x

# 定义一个名为 CrossFormer 的神经网络模块
class CrossFormer(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        *,
        dim = (64, 128, 256, 512),
        depth = (2, 2, 8, 2),
        global_window_size = (8, 4, 2, 1),
        local_window_size = 7,
        cross_embed_kernel_sizes = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides = (4, 2, 2, 2),
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        channels = 3
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 将参数转换为元组形式
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        # 断言确保参数长度为4
        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        # 定义维度相关变量
        last_dim = dim[-1]
        dims = [channels, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # 初始化一个空的神经网络模块列表
        self.layers = nn.ModuleList([])

        # 循环创建交叉嵌入层和 Transformer 层
        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):
            self.layers.append(nn.ModuleList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride = cel_stride),
                Transformer(dim_out, local_window_size = local_wsz, global_window_size = global_wsz, depth = layers, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
            ]))

        # 定义最终的逻辑层
        self.to_logits = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    # 前向传播函数
    def forward(self, x):
        # 遍历每一层的交叉嵌入层和 Transformer 层
        for cel, transformer in self.layers:
            # 执行交叉嵌入层
            x = cel(x)
            # 执行 Transformer 层
            x = transformer(x)

        # 返回最终的逻辑结果
        return self.to_logits(x)
```