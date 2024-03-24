# `.\lucidrains\lumiere-pytorch\lumiere_pytorch\mp_lumiere.py`

```py
# 导入所需的库
from math import sqrt
from functools import partial
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
# 导入 beartype 库，用于类型注解
from beartype import beartype
from beartype.typing import List, Tuple, Optional
# 导入 einops 库，用于操作张量
from einops import rearrange, pack, unpack, repeat
# 导入 lumiere 库中的函数
from lumiere_pytorch.lumiere import (
    image_or_video_to_time,
    handle_maybe_channel_last,
    Lumiere
)

# 定义一些辅助函数

# 判断变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 将张量打包成指定模式的形状
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包后的张量解包成指定模式的形状
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 压缩字典中值不存在的键值对
def compact_values(d: dict):
    return {k: v for k, v in d.items() if exists(v)}

# 计算 L2 范数
def l2norm(t, dim = -1, eps = 1e-12):
    return F.normalize(t, dim = dim, eps = eps)

# 对权重进行归一化处理
def normalize_weight(weight, eps = 1e-4):
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

# 在一维上进行插值
def interpolate_1d(x, length, mode = 'bilinear'):
    x = rearrange(x, 'b c t -> b c t 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    return rearrange(x, 'b c t 1 -> b c t')

# MP 激活函数
class MPSiLU(Module):
    def forward(self, x):
        return F.silu(x) / 0.596

# 增益 - 层缩放
class Gain(Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain

# MP 线性层
class Linear(Module):
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)

# 强制权重归一化的卷积层和线性层
class Conv2d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 2

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.conv2d(x, weight, padding = 'same')

class Conv1d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        init_dirac = False
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in, kernel_size)
        self.weight = nn.Parameter(weight)

        if init_dirac:
            nn.init.dirac_(self.weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.conv1d(x, weight, padding = 'same')

# 像素归一化层
class PixelNorm(Module):
    # 初始化函数，设置维度和epsilon值
    def __init__(self, dim, eps = 1e-4):
        # 调用父类的初始化函数
        super().__init__()
        # 设置像素规范化的高epsilon值
        self.dim = dim
        self.eps = eps

    # 前向传播函数
    def forward(self, x):
        # 获取维度
        dim = self.dim
        # 返回经过L2范数规范化后的结果乘以维度的平方根
        return l2norm(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])
# 定义一个类，实现magnitude preserving sum的功能
# t的值根据经验设定为0.3，用于encoder/decoder/attention residuals和embedding
class MPAdd(Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    # 实现前向传播功能
    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den

# 定义一个类，实现mp attention的功能
class MPAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        mp_add_t = 0.3,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.pixel_norm = PixelNorm(dim = -1)

        self.dropout = nn.Dropout(dropout)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Linear(dim, hidden_dim * 3)
        self.to_out = Linear(hidden_dim, dim)

        self.mp_add = MPAdd(t = mp_add_t)

    # 实现前向传播功能
    def forward(self, x):
        res, b = x, x.shape[0]

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q, k, v = map(self.pixel_norm, (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return self.mp_add(out, res)

# 定义一个类，实现时间维度的下采样
class MPTemporalDownsample(Module):
    def __init__(
        self,
        dim,
        channel_last = False,
        time_dim = None
    ):
        super().__init__()
        self.time_dim = time_dim
        self.channel_last = channel_last
        self.conv = Conv1d(dim, dim, 3, init_dirac = True)

    # 实现前向传播功能
    @handle_maybe_channel_last
    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        t = x.shape[-1]
        assert t > 1, 'time dimension must be greater than 1 to be compressed'

        x = interpolate_1d(x, t // 2)
        return self.conv(x)

# 定义一个类，实现时间维度的上采样
class MPTemporalUpsample(Module):
    def __init__(
        self,
        dim,
        channel_last = False,
        time_dim = None
    ):
        super().__init__()
        self.time_dim = time_dim
        self.channel_last = channel_last
        self.conv = Conv1d(dim, dim, 3, init_dirac = True)

    # 实现前向传播功能
    @handle_maybe_channel_last
    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        t = x.shape[-1]
        x = interpolate_1d(x, t * 2)
        return self.conv(x)

# 定义一个类，实现MP卷积膨胀块的功能
class MPConvolutionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        conv2d_kernel_size = 3,
        conv1d_kernel_size = 3,
        channel_last = False,
        time_dim = None,
        mp_add_t = 0.3,
        dropout = 0.
    ):
        super().__init__()
        self.time_dim = time_dim
        self.channel_last = channel_last

        self.spatial_conv = nn.Sequential(
            Conv2d(dim, dim, conv2d_kernel_size, 3),
            MPSiLU()
        )

        self.temporal_conv = nn.Sequential(
            Conv1d(dim, dim, conv1d_kernel_size, 3),
            MPSiLU(),
            nn.Dropout(dropout)
        )

        self.proj_out = nn.Sequential(
            Conv1d(dim, dim, 1),
            Gain()
        )

        self.residual_mp_add = MPAdd(t = mp_add_t)

    # 实现前向传播功能
    @handle_maybe_channel_last
    def forward(
        self,
        x,
        batch_size = None
        ):
        # 将输入赋值给残差变量
        residual = x

        # 判断输入是否为视频，判断输入的维度是否为5
        is_video = x.ndim == 5

        # 如果是视频
        if is_video:
            # 获取批量大小
            batch_size = x.shape[0]
            # 重新排列输入数据的维度
            x = rearrange(x, 'b c t h w -> (b t) c h w')

        # 对输入进行空间卷积
        x = self.spatial_conv(x)

        # 重新排列参数
        rearrange_kwargs = compact_values(dict(b = batch_size, t = self.time_dim))

        # 断言重新排列参数的长度大于0
        assert len(rearrange_kwargs) > 0, 'either batch_size is passed in on forward, or time_dim is set on init'
        # 重新排列输入数据的维度
        x = rearrange(x, '(b t) c h w -> b h w c t', **rearrange_kwargs)

        # 打包输入数据
        x, ps = pack_one(x, '* c t')

        # 对输入进行时间卷积
        x = self.temporal_conv(x)
        # 对输入进行投影输出
        x = self.proj_out(x)

        # 解包输入数据
        x = unpack_one(x, ps, '* c t')

        # 如果是视频
        if is_video:
            # 重新排列输入数据的维度
            x = rearrange(x, 'b h w c t -> b c t h w')
        else:
            # 重新排列输入数据的维度
            x = rearrange(x, 'b h w c t -> (b t) c h w')

        # 返回残差模块添加后的结果
        return self.residual_mp_add(x, residual)
# 定义一个多头注意力膨胀块类，继承自 Module 类
class MPAttentionInflationBlock(Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 维度
        depth = 1,  # 层数，默认为1
        time_dim = None,  # 时间维度，默认为None
        channel_last = False,  # 是否通道在最后，默认为False
        mp_add_t = 0.3,  # MP 添加时间，默认为0.3
        dropout = 0.,  # 丢弃率，默认为0
        **attn_kwargs  # 其他注意力参数
    ):
        super().__init__()

        self.time_dim = time_dim  # 初始化时间维度
        self.channel_last = channel_last  # 初始化通道在最后

        self.temporal_attns = ModuleList([])  # 初始化时间注意力模块列表

        # 循环创建指定层数的多头注意力模块
        for _ in range(depth):
            attn = MPAttention(
                dim = dim,
                dropout = dropout,
                **attn_kwargs
            )

            self.temporal_attns.append(attn)  # 将创建的多头注意力模块添加到列表中

        # 定义输出投影层
        self.proj_out = nn.Sequential(
            Linear(dim, dim),  # 线性层
            Gain()  # 增益层
        )

        # 定义残差 MP 添加层
        self.residual_mp_add = MPAdd(t = mp_add_t)

    # 前向传播函数
    @handle_maybe_channel_last
    def forward(
        self,
        x,  # 输入张量
        batch_size = None  # 批量大小，默认为None
    ):
        is_video = x.ndim == 5  # 判断是否为视频数据
        assert is_video ^ (exists(batch_size) or exists(self.time_dim)), 'either a tensor of shape (batch, channels, time, height, width) is passed in, or (batch * time, channels, height, width) along with `batch_size`'

        if self.channel_last:
            x = rearrange(x, 'b ... c -> b c ...')  # 重新排列张量维度

        if is_video:
            batch_size = x.shape[0]  # 获取批量大小
            x = rearrange(x, 'b c t h w -> b h w t c')  # 重新排列张量维度
        else:
            assert exists(batch_size) or exists(self.time_dim)  # 断言批量大小或时间维度存在

            rearrange_kwargs = dict(b = batch_size, t = self.time_dim)
            x = rearrange(x, '(b t) c h w -> b h w t c', **compact_values(rearrange_kwargs))  # 重新排列张量维度

        x, ps = pack_one(x, '* t c')  # 打包张量

        residual = x  # 保存残差

        # 遍历时间注意力模块列表
        for attn in self.temporal_attns:
            x = attn(x)  # 多头注意��操作

        x = self.proj_out(x)  # 投影输出

        x = self.residual_mp_add(x, residual)  # 残差 MP 添加

        x = unpack_one(x, ps, '* t c')  # 解包张量

        if is_video:
            x = rearrange(x, 'b h w t c -> b c t h w')  # 重新排列张量维度
        else:
            x = rearrange(x, 'b h w t c -> (b t) c h w')  # 重新排列张量维度

        if self.channel_last:
            x = rearrange(x, 'b c ... -> b ... c')  # 重新排列张量维度

        return x  # 返回结果张量

# MPLumiere 是 Lumiere 的一个部分，包含四个 MP 时间模块
MPLumiere = partial(
    Lumiere,
    conv_klass = MPConvolutionInflationBlock,  # 卷积类
    attn_klass = MPAttentionInflationBlock,  # 注意力类
    downsample_klass = MPTemporalDownsample,  # 下采样类
    upsample_klass = MPTemporalUpsample  # 上采样类
)
```