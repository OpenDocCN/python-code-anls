# `.\lucidrains\uformer-pytorch\uformer_pytorch\uformer_pytorch.py`

```py
# 导入 math 模块
import math
# 从 math 模块导入 log, pi, sqrt 函数
from math import log, pi, sqrt
# 从 functools 模块导入 partial 函数
from functools import partial

# 导入 torch 模块
import torch
# 从 torch 模块导入 nn, einsum 函数
from torch import nn, einsum
# 从 torch.nn 模块导入 functional 模块
import torch.nn.functional as F

# 导入 einops 模块中的 rearrange, repeat 函数
from einops import rearrange, repeat

# 定义常量 List 为 nn.ModuleList 类
List = nn.ModuleList

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 将变量转换为元组的函数
def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

# 位置嵌入

# 应用旋转位置嵌入的函数
def apply_rotary_emb(q, k, pos_emb):
    sin, cos = pos_emb
    dim_rotary = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

# 每两个元素旋转的函数
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

# 轴向旋转嵌入类
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    def forward(self, x):
        device, dtype, h, w = x.device, x.dtype, *x.shape[-2:]

        seq_x = torch.linspace(-1., 1., steps = h, device = device)
        seq_x = seq_x.unsqueeze(-1)

        seq_y = torch.linspace(-1., 1., steps = w, device = device)
        seq_y = seq_y.unsqueeze(-1)

        scales = self.scales[(*((None,) * (len(seq_x.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        scales = self.scales[(*((None,) * (len(seq_y.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        seq_x = seq_x * scales * pi
        seq_y = seq_y * scales * pi

        x_sinu = repeat(seq_x, 'i d -> i j d', j = w)
        y_sinu = repeat(seq_y, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> i j d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'i j d -> () i j (d r)', r = 2), (sin, cos))
        return sin, cos

# 时间正弦位置嵌入类
class TimeSinuPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb)
        emb = einsum('i, j -> i  j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb

# 辅助类

# 层归一化类
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

# 预归一化��
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 注意力类
class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, window_size = 16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
    # 定义前向传播函数，接受输入 x，跳跃连接 skip，默认时间嵌入 time_emb 和位置嵌入 pos_emb
    def forward(self, x, skip = None, time_emb = None, pos_emb = None):
        # 获取头数 h，窗口大小 w，输入张量的批量大小 b
        h, w, b = self.heads, self.window_size, x.shape[0]

        # 如果时间嵌入存在，则将其重排维度并与输入相加
        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            x = x + time_emb

        # 将输入 x 转换为查询向量 q
        q = self.to_q(x)

        # 将键值对输入设置为 x
        kv_input = x

        # 如果跳跃连接存在，则将其与键值对输入连接在一起
        if exists(skip):
            kv_input = torch.cat((kv_input, skip), dim = 0)

        # 将键值对输入转换为键 k 和值 v，并按维度进行分块
        k, v = self.to_kv(kv_input).chunk(2, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) x y c', h = h), (q, k, v))

        # 如果位置嵌入存在，则应用旋转位置嵌入到查询 q 和键 k 上
        if exists(pos_emb):
            q, k = apply_rotary_emb(q, k, pos_emb)

        # 重排查询 q、键 k 和值 v 的维度
        q, k, v = map(lambda t: rearrange(t, 'b (x w1) (y w2) c -> (b x y) (w1 w2) c', w1 = w, w2 = w), (q, k, v))

        # 如果跳跃连接存在，则对键 k 和值 v 进行维度重排
        if exists(skip):
            k, v = map(lambda t: rearrange(t, '(r b) n d -> b (r n) d', r = 2), (k, v))

        # 计算注意力相似度矩阵
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # 对相似度矩阵进行 softmax 操作得到注意力权重
        attn = sim.softmax(dim = -1)
        # 根据注意力权重计算输出
        out = einsum('b i j, b j d -> b i d', attn, v)

        # 重排输出的维度
        out = rearrange(out, '(b h x y) (w1 w2) c -> b (h c) (x w1) (y w2)', b = b, h = h, y = x.shape[-1] // w, w1 = w, w2 = w)
        # 将输出传递给输出层并返回结果
        return self.to_out(out)
# 定义一个前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        hidden_dim = dim * mult
        # 输入投影层，将输入维度转换为隐藏维度
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)
        # 输出投影层，包含卷积、GELU激活函数和再次卷积
        self.project_out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x, time_emb = None):
        # 对输入进行投影
        x = self.project_in(x)
        # 如果存在时间嵌入，则将其重排并加到输入上
        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            x = x + time_emb
        # 返回经过输出投影层的结果
        return self.project_out(x)

# 定义一个块模块
class Block(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        window_size = 16,
        time_emb_dim = None,
        rotary_emb = True
    ):
        super().__init__()
        self.attn_time_emb = None
        self.ff_time_emb = None
        # 如果存在时间嵌入维度，则创建注意力和前馈的时间嵌入
        if exists(time_emb_dim):
            self.attn_time_emb = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            self.ff_time_emb = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim * ff_mult))

        # 如果使用轴向旋转嵌入，则创建位置嵌入
        self.pos_emb = AxialRotaryEmbedding(dim_head) if rotary_emb else None

        # 创建多个块层
        self.layers = List([])
        for _ in range(depth):
            self.layers.append(List([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, window_size = window_size)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult))
            ]))

    def forward(self, x, skip = None, time = None):
        attn_time_emb = None
        ff_time_emb = None
        # 如果存在时间信息，则计算注意力和前馈的时间嵌入
        if exists(time):
            assert exists(self.attn_time_emb) and exists(self.ff_time_emb), 'time_emb_dim must be given on init if you are conditioning based on time'
            attn_time_emb = self.attn_time_emb(time)
            ff_time_emb = self.ff_time_emb(time)

        pos_emb = None
        # 如果存在位置嵌入，则计算位置嵌入
        if exists(self.pos_emb):
            pos_emb = self.pos_emb(x)

        # 遍历每个块层，进行注意力和前馈操作
        for attn, ff in self.layers:
            x = attn(x, skip = skip, time_emb = attn_time_emb, pos_emb = pos_emb) + x
            x = ff(x, time_emb = ff_time_emb) + x
        # 返回处理后的结果
        return x

# 定义一个 Uformer 模块
class Uformer(nn.Module):
    def __init__(
        self,
        dim = 64,
        channels = 3,
        stages = 4,
        num_blocks = 2,
        dim_head = 64,
        window_size = 16,
        heads = 8,
        ff_mult = 4,
        time_emb = False,
        input_channels = None,
        output_channels = None
    ):
        # 调用父类的构造函数
        super().__init__()
        # 设置输入通道数为默认值或者与输出通道数相同
        input_channels = default(input_channels, channels)
        output_channels = default(output_channels, channels)

        self.to_time_emb = None
        time_emb_dim = None

        # 如果需要时间嵌入
        if time_emb:
            time_emb_dim = dim
            # 创建时间嵌入层
            self.to_time_emb = nn.Sequential(
                TimeSinuPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )

        # 输入通道到维度转换
        self.project_in = nn.Sequential(
            nn.Conv2d(input_channels, dim, 3, padding = 1),
            nn.GELU()
        )

        # 维度到输出通道转换
        self.project_out = nn.Sequential(
            nn.Conv2d(dim, output_channels, 3, padding = 1),
        )

        # 下采样和上采样列表
        self.downs = List([])
        self.ups = List([])

        # 将参数转换为指定深度的元组
        heads, window_size, dim_head, num_blocks = map(partial(cast_tuple, depth = stages), (heads, window_size, dim_head, num_blocks))

        # 遍历各个阶段
        for ind, heads, window_size, dim_head, num_blocks in zip(range(stages), heads, window_size, dim_head, num_blocks):
            is_last = ind == (stages - 1)

            # 添加下采样模块
            self.downs.append(List([
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim),
                nn.Conv2d(dim, dim * 2, 4, stride = 2, padding = 1)
            ]))

            # 添加上采样模块
            self.ups.append(List([
                nn.ConvTranspose2d(dim * 2, dim, 2, stride = 2),
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim)
            ]))

            dim *= 2

            # 如果是最后一个阶段，设置中间模块
            if is_last:
                self.mid = Block(dim = dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim)

    # 前向传播函数
    def forward(
        self,
        x,
        time = None
    ):
        # 如果存在时间信息
        if exists(time):
            assert exists(self.to_time_emb), 'time_emb must be set to true to condition on time'
            time = time.to(x)
            time = self.to_time_emb(time)

        # 输入数据通过输入通道转换
        x = self.project_in(x)

        skips = []
        # 对下采样模块进行迭代
        for block, downsample in self.downs:
            x = block(x, time = time)
            skips.append(x)
            x = downsample(x)

        # 中间模块
        x = self.mid(x, time = time)

        # 对上采样模块进行迭代
        for (upsample, block), skip in zip(reversed(self.ups), reversed(skips)):
            x = upsample(x)
            x = block(x, skip = skip, time = time)

        # 输出数据通过输出通道转换
        x = self.project_out(x)
        return x
```