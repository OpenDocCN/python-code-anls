# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\karras_unet_3d.py`

```
"""
the magnitude-preserving unet proposed in https://arxiv.org/abs/2312.02696 by Karras et al.
"""

import math
from math import sqrt, ceil
from functools import partial
from typing import Optional, Union, Tuple

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack

from denoising_diffusion_pytorch.attend import Attend

# helpers functions

# 检查变量是否存在
def exists(x):
    return x is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 逻辑异或操作
def xnor(x, y):
    return not (x ^ y)

# 在数组末尾添加元素
def append(arr, el):
    arr.append(el)

# 在数组开头添加元素
def prepend(arr, el):
    arr.insert(0, el)

# 将张量打包成指定模式的形状
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包后的张量解包成原始形状
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 将输入转换为元组
def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

# 判断一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 在论文中，他们使用 eps 1e-4 作为像素归一化的值

# 计算 L2 范数
def l2norm(t, dim = -1, eps = 1e-12):
    return F.normalize(t, dim = dim, eps = eps)

# mp activations
# section 2.5

# MPSiLU 激活函数
class MPSiLU(Module):
    def forward(self, x):
        return F.silu(x) / 0.596

# gain - layer scaling

# 增益层
class Gain(Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain

# magnitude preserving concat
# equation (103) - default to 0.5, which they recommended

# 保持幅度的拼接层
class MPCat(Module):
    def __init__(self, t = 0.5, dim = -1):
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]

        C = sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))

        a = a * (1. - t) / sqrt(Na)
        b = b * t / sqrt(Nb)

        return C * torch.cat((a, b), dim = dim)

# magnitude preserving sum
# equation (88)
# empirically, they found t=0.3 for encoder / decoder / attention residuals
# and for embedding, t=0.5

# 保持幅度的求和层
class MPAdd(Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den

# pixelnorm
# equation (30)

# 像素归一化层
class PixelNorm(Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        # 论文中像素归一化的高 epsilon 值
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return l2norm(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

# forced weight normed conv3d and linear
# algorithm 1 in paper

# 归一化权重的 Conv3d 和 Linear 层
def normalize_weight(weight, eps = 1e-4):
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

# 3D 卷积层
class Conv3d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 3
        self.concat_ones_to_input = concat_ones_to_input
    # 定义前向传播函数，接受输入 x
    def forward(self, x):

        # 如果处于训练模式
        if self.training:
            # 在不计算梯度的情况下，对权重进行归一化处理
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                # 将归一化后的权重复制给当前权重
                self.weight.copy_(normed_weight)

        # 对权重进行归一化处理，并除以输入特征的平方根
        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)

        # 如果需要将输入与全为1的张量进行拼接
        if self.concat_ones_to_input:
            # 在输入张量的最后一维度上填充1
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 1, 0), value = 1.)

        # 返回经过卷积操作后的结果
        return F.conv3d(x, weight, padding='same')
# 定义一个线性层模块，包含输入维度、输出维度和一个小的常数 eps
class Linear(Module):
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        # 用随机数初始化权重矩阵
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    # 前向传播函数
    def forward(self, x):
        # 如果处于训练状态
        if self.training:
            # 使用 torch.no_grad() 上下文管理器，不计算梯度
            with torch.no_grad():
                # 对权重进行归一化处理
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                # 将归一化后的权重复制给原始权重
                self.weight.copy_(normed_weight)

        # 对权重进行归一化处理，并除以输入维度的平方根
        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        # 返回线性变换后的结果
        return F.linear(x, weight)

# MP Fourier Embedding 模块

class MPFourierEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        # 断言维度必须是2的倍数
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        # 初始化权重参数，不需要计算梯度
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    # 前向传播函数
    def forward(self, x):
        # 对输入进行维度重排
        x = rearrange(x, 'b -> b 1')
        # 计算频率
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # 返回正弦和余弦函数的拼接结果，并乘以根号2
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2)

# 构建基本模块

class Encoder(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        factorize_space_time_attn = False,
        downsample = False,
        downsample_config: Tuple[bool, bool, bool] = (True, True, True)
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.downsample = downsample
        self.downsample_config = downsample_config

        self.downsample_conv = None

        curr_dim = dim
        # 如果需要下采样
        if downsample:
            # 使用 1x1 卷积进行下采样
            self.downsample_conv = Conv3d(curr_dim, dim_out, 1)
            curr_dim = dim_out

        # 像素归一化
        self.pixel_norm = PixelNorm(dim = 1)

        self.to_emb = None
        # 如果存在嵌入维度
        if exists(emb_dim):
            # 构建嵌入层
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        # 第一个基本模块
        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv3d(curr_dim, dim_out, 3)
        )

        # 第二个基本模块
        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv3d(dim_out, dim_out, 3)
        )

        # MPAdd 模块
        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        self.factorized_attn = factorize_space_time_attn

        # 如果有注意力机制
        if has_attn:
            attn_kwargs = dict(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

            # 如果需要分解空间和时间的注意力机制
            if factorize_space_time_attn:
                self.attn = nn.ModuleList([
                    Attention(**attn_kwargs, only_space = True),
                    Attention(**attn_kwargs, only_time = True),
                ])
            else:
                self.attn = Attention(**attn_kwargs)

    # 前向传播函数
    def forward(
        self,
        x,
        emb = None
        ):
        # 如果存在下采样参数
        if self.downsample:
            # 获取输入张量的时间、高度、宽度
            t, h, w = x.shape[-3:]
            # 根据下采样配置计算缩放因子
            resize_factors = tuple((2 if downsample else 1) for downsample in self.downsample_config)
            # 计算插值后的形状
            interpolate_shape = tuple(shape // factor for shape, factor in zip((t, h, w), resize_factors))

            # 对输入张量进行三线性插值
            x = F.interpolate(x, interpolate_shape, mode='trilinear')
            # 使用下采样卷积层处理插值后的张量
            x = self.downsample_conv(x)

        # 对输入张量进行像素归一化
        x = self.pixel_norm(x)

        # 复制输入张量
        res = x.clone()

        # 使用第一个残差块处理输入张量
        x = self.block1(x)

        # 如果存在嵌入向量
        if exists(emb):
            # 计算缩放因子
            scale = self.to_emb(emb) + 1
            # 对输入张量进行缩放
            x = x * rearrange(scale, 'b c -> b c 1 1 1')

        # 使用第二个残差块处理输入张量
        x = self.block2(x)

        # 将残差块的输出与之前复制的张量相加
        x = self.res_mp_add(x, res)

        # 如果存在注意力机制
        if exists(self.attn):
            # 如果使用分解的注意力机制
            if self.factorized_attn:
                # 获取空间注意力和时间注意力
                attn_space, attn_time = self.attn
                # 先对空间进行注意力处理
                x = attn_space(x)
                # 再对时间进行注意力处理
                x = attn_time(x)

            else:
                # 使用整体的注意力机制处理输入张量
                x = self.attn(x)

        # 返回处理后的张量
        return x
# 定义一个名为 Decoder 的类，继承自 Module 类
class Decoder(Module):
    # 初始化方法
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        factorize_space_time_attn = False,
        upsample = False,
        upsample_config: Tuple[bool, bool, bool] = (True, True, True)
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果未指定 dim_out，则设为 dim
        dim_out = default(dim_out, dim)

        # 设置是否需要上采样和上采样配置
        self.upsample = upsample
        self.upsample_config = upsample_config

        # 如果不需要上采样，则需要跳跃连接
        self.needs_skip = not upsample

        # 如果存在 emb_dim，则创建线性层和增益层
        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        # 第一个块包含 MPSiLU 和 3D 卷积层
        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv3d(dim, dim_out, 3)
        )

        # 第二个块包含 MPSiLU、Dropout 和 3D 卷积层
        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv3d(dim_out, dim_out, 3)
        )

        # 如果输入维度不等于输出维度，则使用 1x1 卷积层进行维度匹配
        self.res_conv = Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # 创建 MPAdd 模块
        self.res_mp_add = MPAdd(t = mp_add_t)

        # 初始化注意力机制相关参数
        self.attn = None
        self.factorized_attn = factorize_space_time_attn

        # 如果需要注意力机制
        if has_attn:
            attn_kwargs = dict(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

            # 如果需要分解空间和时间的注意力机制
            if factorize_space_time_attn:
                self.attn = nn.ModuleList([
                    Attention(**attn_kwargs, only_space = True),
                    Attention(**attn_kwargs, only_time = True),
                ])
            else:
                self.attn = Attention(**attn_kwargs)

    # 前向传播方法
    def forward(
        self,
        x,
        emb = None
    ):
        # 如果需要上采样
        if self.upsample:
            t, h, w = x.shape[-3:]
            resize_factors = tuple((2 if upsample else 1) for upsample in self.upsample_config)
            interpolate_shape = tuple(shape * factor for shape, factor in zip((t, h, w), resize_factors))

            x = F.interpolate(x, interpolate_shape, mode = 'trilinear')

        # 计算残差连接
        res = self.res_conv(x)

        # 第一个块的操作
        x = self.block1(x)

        # 如果存在 emb，则进行缩放
        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1 1')

        # 第二个块的操作
        x = self.block2(x)

        # 计算残差连接的 MPAdd
        x = self.res_mp_add(x, res)

        # 如果存在注意力机制
        if exists(self.attn):
            # 如果使用分解的注意力机制
            if self.factorized_attn:
                attn_space, attn_time = self.attn
                x = attn_space(x)
                x = attn_time(x)

            else:
                x = self.attn(x)

        return x

# 定义名为 Attention 的类，继承自 Module 类
class Attention(Module):
    # 初始化方法
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        flash = False,
        mp_add_t = 0.3,
        only_space = False,
        only_time = False
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 确保只有空间或时间中的一个为 True
        assert (int(only_space) + int(only_time)) <= 1

        # 设置头数和隐藏维度
        self.heads = heads
        hidden_dim = dim_head * heads

        # 像素归一化
        self.pixel_norm = PixelNorm(dim = -1)

        # 注意力机制
        self.attend = Attend(flash = flash)

        # 记忆键值对
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv3d(dim, hidden_dim * 3, 1)
        self.to_out = Conv3d(hidden_dim, dim, 1)

        # MPAdd 模块
        self.mp_add = MPAdd(t = mp_add_t)

        # 是否只考虑空间或时间
        self.only_space = only_space
        self.only_time = only_time
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 保存输入 x 的原始形状
        res, orig_shape = x, x.shape
        b, c, t, h, w = orig_shape

        # 将输入 x 转换为查询、键、值
        qkv = self.to_qkv(x)

        # 根据 self.only_space 和 self.only_time 进行不同的重排操作
        if self.only_space:
            qkv = rearrange(qkv, 'b c t x y -> (b t) c x y')
        elif self.only_time:
            qkv = rearrange(qkv, 'b c t x y -> (b x y) c t')

        # 将查询、键、值分成三部分
        qkv = qkv.chunk(3, dim = 1)

        # 重排查询、键、值的形状
        q, k, v = map(lambda t: rearrange(t, 'b (h c) ... -> b h (...) c', h = self.heads), qkv)

        # 复制记忆键值对
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = k.shape[0]), self.mem_kv)

        # 拼接键和值
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        # 对查询、键、值进行像素归一化
        q, k, v = map(self.pixel_norm, (q, k, v))

        # 进行注意力计算
        out = self.attend(q, k, v)

        # 重排输出形状
        out = rearrange(out, 'b h n d -> b (h d) n')

        # 根据 self.only_space 和 self.only_time 进行不同的重排操作
        if self.only_space:
            out = rearrange(out, '(b t) c n -> b c (t n)', t = t)
        elif self.only_time:
            out = rearrange(out, '(b x y) c n -> b c (n x y)', x = h, y = w)

        # 恢复输出形状
        out = out.reshape(orig_shape)

        # 将输出转换为最终输出
        out = self.to_out(out)

        # 将最终输出与输入相加并返回
        return self.mp_add(out, res)
# 定义了一个名为KarrasUnet3D的类，代表Karras提出的3D U-Net模型
# 该模型没有偏置，没有组归一化，使用保持幅度的操作

class KarrasUnet3D(Module):
    """
    根据图21的配置G进行设计
    """

    def __init__(
        self,
        *,
        image_size,              # 图像大小
        frames,                  # 帧数
        dim = 192,               # 维度
        dim_max = 768,           # 通道数将在每次下采样时翻倍，并限制在这个值
        num_classes = None,      # 类别数，在论文中为一个流行的基准测试使用了1000个类别
        channels = 4,            # 为什么是4个通道，可能是指alpha通道？
        num_downsamples = 3,     # 下采样次数
        num_blocks_per_stage: Union[int, Tuple[int, ...]] = 4,  # 每个阶段的块数
        downsample_types: Optional[Tuple[str, ...]] = None,     # 下采样类型
        attn_res = (16, 8),      # 注意力机制的分辨率
        fourier_dim = 16,        # 傅立叶维度
        attn_dim_head = 64,      # 注意力机制的头数
        attn_flash = False,      # 是否使用闪光注意力
        mp_cat_t = 0.5,          # MP Cat阈值
        mp_add_emb_t = 0.5,      # MP Add Emb阈值
        attn_res_mp_add_t = 0.3, # 注意力机制MP Add阈值
        resnet_mp_add_t = 0.3,   # ResNet MP Add阈值
        dropout = 0.1,           # 丢弃率
        self_condition = False,  # 是否自我条件
        factorize_space_time_attn = False  # 是否分解空间时间注意力
    @property
    def downsample_factor(self):
        return 2 ** self.num_downsamples

    def forward(
        self,
        x,
        time,
        self_cond = None,
        class_labels = None
    ):
        # 验证图像形状

        assert x.shape[1:] == (self.channels, self.frames, self.image_size, self.image_size)

        # 自我条件

        if self.self_condition:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim = 1)
        else:
            assert not exists(self_cond)

        # 时间条件

        time_emb = self.to_time_emb(time)

        # 类别条件

        assert xnor(exists(class_labels), self.needs_class_labels)

        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)

            assert class_labels.shape[-1] == self.num_classes
            class_labels = class_labels.float() * sqrt(self.num_classes)

            class_emb = self.to_class_emb(class_labels)

            time_emb = self.add_class_emb(time_emb, class_emb)

        # 最终的MP-SiLU用于嵌入

        emb = self.emb_activation(time_emb)

        # 跳跃连接

        skips = []

        # 输入块

        x = self.input_block(x)

        skips.append(x)

        # 下采样

        for encoder in self.downs:
            x = encoder(x, emb = emb)
            skips.append(x)

        # 中间

        for decoder in self.mids:
            x = decoder(x, emb = emb)

        # 上采样

        for decoder in self.ups:
            if decoder.needs_skip:
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)

            x = decoder(x, emb = emb)

        # 输出块

        return self.output_block(x)

# 改进的MP Transformer

class MPFeedForward(Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        mp_add_t = 0.3
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            PixelNorm(dim = 1),
            Conv3d(dim, dim_inner, 1),
            MPSiLU(),
            Conv3d(dim_inner, dim, 1)
        )

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        res = x
        out = self.net(x)
        return self.mp_add(out, res)

class MPImageTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_mem_kv = 4,
        ff_mult = 4,
        attn_flash = False,
        residual_mp_add_t = 0.3
    # 定义一个继承自 nn.Module 的 Transformer 类
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化一个空的 ModuleList 用于存储 Transformer 的层
        self.layers = ModuleList([])

        # 根据指定的深度循环创建 Transformer 的每一层
        for _ in range(depth):
            # 在 layers 中添加一个包含 Attention 和 MPFeedForward 两个模块的 ModuleList
            self.layers.append(ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, num_mem_kv = num_mem_kv, flash = attn_flash, mp_add_t = residual_mp_add_t),
                MPFeedForward(dim = dim, mult = ff_mult, mp_add_t = residual_mp_add_t)
            ]))

    # 定义 Transformer 类的前向传播函数
    def forward(self, x):

        # 遍历 Transformer 的每一层，依次进行 Attention 和 FeedForward 操作
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        # 返回处理后的结果
        return x
# 如果当前脚本作为主程序运行
if __name__ == '__main__':

    # 创建一个 KarrasUnet3D 的实例
    unet = KarrasUnet3D(
        frames = 32,  # 视频帧数
        image_size = 64,  # 图像大小
        dim = 8,  # 维度
        dim_max = 768,  # 最大维度
        num_downsamples = 6,  # 下采样次数
        num_blocks_per_stage = (4, 3, 2, 2, 2, 2),  # 每个阶段的块数
        downsample_types = (
            'image',  # 图像下采样类型
            'frame',  # 帧下采样类型
            'image',  # 图像下采样类型
            'frame',  # 帧下采样类型
            'image',  # 图像下采样类型
            'frame',  # 帧下采样类型
        ),
        attn_dim_head = 8,  # 注意力机制的头数
        num_classes = 1000,  # 类别数
        factorize_space_time_attn = True  # 是否在空间和时间上分别进行注意力操作
    )

    # 创建一个形状为 (2, 4, 32, 64, 64) 的随机张量作为视频输入
    video = torch.randn(2, 4, 32, 64, 64)

    # 使用 unet 对视频进行去噪处理
    denoised_video = unet(
        video,  # 输入视频
        time = torch.ones(2,),  # 时间信息
        class_labels = torch.randint(0, 1000, (2,))  # 类别标签
    )
```