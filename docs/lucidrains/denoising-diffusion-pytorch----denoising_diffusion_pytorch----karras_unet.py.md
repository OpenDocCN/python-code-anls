# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\karras_unet.py`

```py
"""
the magnitude-preserving unet proposed in https://arxiv.org/abs/2312.02696 by Karras et al.
"""

import math
from math import sqrt, ceil
from functools import partial

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

# 返回默认值
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

# 将张量打包成指定模式
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包的张量解包成指定模式
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 将输入转换为元组
def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

# 判断是否可以整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

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
        # high epsilon for the pixel norm in the paper
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return l2norm(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

# forced weight normed conv2d and linear
# algorithm 1 in paper

# 规范化权重
def normalize_weight(weight, eps = 1e-4):
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

# 卷积层
class Conv2d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 2
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

        # 如果需要将输入特征的维度扩展为与权重相同
        if self.concat_ones_to_input:
            # 在输入特征的高度维度上添加一个维度，值为1
            x = F.pad(x, (0, 0, 0, 0, 1, 0), value = 1.)

        # 返回经过卷积操作后的结果
        return F.conv2d(x, weight, padding='same')
class Linear(Module):
    # 定义一个线性层模块，包含输入维度、输出维度和一个小的常数 eps
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        # 用随机数初始化权重
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

# mp fourier embeds

class MPFourierEmbedding(Module):
    # 定义一个多项式傅里叶嵌入模块，包含维度信息
    def __init__(self, dim):
        super().__init__()
        # 断言维度能被 2 整除
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        # 初始化权重参数，不需要梯度
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    # 前向传播函数
    def forward(self, x):
        # 对输入进行重新排列
        x = rearrange(x, 'b -> b 1')
        # 计算频率
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # 返回正弦和余弦函数的拼接结果
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2)

# building block modules

class Encoder(Module):
    # 定义一个编码器模块，包含多个参数和子模块
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
        downsample = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.downsample = downsample
        self.downsample_conv = None

        curr_dim = dim
        # 如果需要下采样，添加一个卷积层
        if downsample:
            self.downsample_conv = Conv2d(curr_dim, dim_out, 1)
            curr_dim = dim_out

        # ��素归一化
        self.pixel_norm = PixelNorm(dim = 1)

        self.to_emb = None
        # 如果存在嵌入维度，添加线性层和增益操作
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        # 第一个块
        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(curr_dim, dim_out, 3)
        )

        # 第二个块
        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        # MPAdd 操作
        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        # 如果有注意力机制，添加注意力模块
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    # 前向传播函数
    def forward(
        self,
        x,
        emb = None
    ):
        # 如果需要下采样，进行插值操作和卷积
        if self.downsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h // 2, w // 2), mode = 'bilinear')
            x = self.downsample_conv(x)

        # 像素归一化
        x = self.pixel_norm(x)

        res = x.clone()

        x = self.block1(x)

        # 如果存在嵌入信息，进行缩放操作
        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        # 如果存在注意力模块，应用注意力机制
        if exists(self.attn):
            x = self.attn(x)

        return x

class Decoder(Module):
    # 定义一个解码器模块，包含多个参数和子模块
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
        upsample = False
    # 初始化函数，继承父类的初始化方法
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果输出维度未指定，则使用输入维度作为输出维度
        dim_out = default(dim_out, dim)

        # 设置上采样标志
        self.upsample = upsample
        # 判断是否需要跳跃连接
        self.needs_skip = not upsample

        # 初始化嵌入层
        self.to_emb = None
        # 如果嵌入维度存在，则创建嵌入层
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        # 第一个块
        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(dim, dim_out, 3)
        )

        # 第二个块
        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        # 残差连接的卷积层
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # 残差连接的加法操作
        self.res_mp_add = MPAdd(t = mp_add_t)

        # 注意力机制
        self.attn = None
        # 如果需要注意力机制
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    # 前向传播函数
    def forward(
        self,
        x,
        emb = None
    ):
        # 如果需要上采样
        if self.upsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h * 2, w * 2), mode = 'bilinear')

        # 计算残差连接
        res = self.res_conv(x)

        # 第一个块的操作
        x = self.block1(x)

        # 如果嵌入存在，则对输入进行缩放
        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        # 第二个块的操作
        x = self.block2(x)

        # 残差连接的加法操作
        x = self.res_mp_add(x, res)

        # 如果存在注意力机制，则应用注意力机制
        if exists(self.attn):
            x = self.attn(x)

        # 返回结果
        return x
# 定义注意力机制模块
class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        flash = False,
        mp_add_t = 0.3
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        # 像素归一化
        self.pixel_norm = PixelNorm(dim = -1)

        # 注意力机制
        self.attend = Attend(flash = flash)

        # 存储键值对的参数
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        # 将输入转换为查询、键、值
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1)
        # 输出转换
        self.to_out = Conv2d(hidden_dim, dim, 1)

        # 多路加法
        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        res, b, c, h, w = x, *x.shape

        # 将输入转换为查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        # 重复存储的键值对
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        # 对查询、键、值进行像素归一化
        q, k, v = map(self.pixel_norm, (q, k, v))

        # 注意力机制
        out = self.attend(q, k, v)

        # 重排输出形状
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        return self.mp_add(out, res)

# Karras 提出的 Unet 模型
# 无偏置、无组归一化、保持幅度的操作

class KarrasUnet(Module):
    """
    根据图 21 配置 G
    """

    def __init__(
        self,
        *,
        image_size,
        dim = 192,
        dim_max = 768,            # 通道数每次下采样会翻倍，最大值为此值
        num_classes = None,       # 论文中为了一个流行的基准测试，使用 1000 个类别
        channels = 4,             # 论文中为 4 个通道，可能是指 alpha 通道？
        num_downsamples = 3,
        num_blocks_per_stage = 4,
        attn_res = (16, 8),
        fourier_dim = 16,
        attn_dim_head = 64,
        attn_flash = False,
        mp_cat_t = 0.5,
        mp_add_emb_t = 0.5,
        attn_res_mp_add_t = 0.3,
        resnet_mp_add_t = 0.3,
        dropout = 0.1,
        self_condition = False
    ):
        # 调用父类的构造函数
        super().__init__()

        # 设置 self_condition 属性
        self.self_condition = self_condition

        # 确定维度

        # 设置通道数和图像大小
        self.channels = channels
        self.image_size = image_size
        input_channels = channels * (2 if self_condition else 1)

        # 输入和输出块

        # 创建输入块
        self.input_block = Conv2d(input_channels, dim, 3, concat_ones_to_input = True)

        # 创建输出块
        self.output_block = nn.Sequential(
            Conv2d(dim, channels, 3),
            Gain()
        )

        # 时间嵌入

        # 设置嵌入维度
        emb_dim = dim * 4

        # 创建时间嵌入
        self.to_time_emb = nn.Sequential(
            MPFourierEmbedding(fourier_dim),
            Linear(fourier_dim, emb_dim)
        )

        # 类别嵌入

        # 检查是否需要类别标签
        self.needs_class_labels = exists(num_classes)
        self.num_classes = num_classes

        if self.needs_class_labels:
            # 创建类别嵌入
            self.to_class_emb = Linear(num_classes, 4 * dim)
            self.add_class_emb = MPAdd(t = mp_add_emb_t)

        # 最终嵌入激活函数

        # 设置嵌入激活函数
        self.emb_activation = MPSiLU()

        # 下采样数量

        # 设置下采样数量
        self.num_downsamples = num_downsamples

        # 注意力

        # 设置注意力的分辨率
        attn_res = set(cast_tuple(attn_res))

        # ResNet 块

        # 设置 ResNet 块的参数
        block_kwargs = dict(
            dropout = dropout,
            emb_dim = emb_dim,
            attn_dim_head = attn_dim_head,
            attn_res_mp_add_t = attn_res_mp_add_t,
            attn_flash = attn_flash
        )

        # UNet 编码器和解码器

        # 初始化编码器和解码器列表
        self.downs = ModuleList([])
        self.ups = ModuleList([])

        curr_dim = dim
        curr_res = image_size

        # 处理初始输入块和前三个编码器块的跳跃连接
        self.skip_mp_cat = MPCat(t = mp_cat_t, dim = 1)

        prepend(self.ups, Decoder(dim * 2, dim, **block_kwargs))

        assert num_blocks_per_stage >= 1

        for _ in range(num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)

            append(self.downs, enc)
            prepend(self.ups, dec)

        # 阶段

        for _ in range(self.num_downsamples):
            dim_out = min(dim_max, curr_dim * 2)
            upsample = Decoder(dim_out, curr_dim, has_attn = curr_res in attn_res, upsample = True, **block_kwargs)

            curr_res //= 2
            has_attn = curr_res in attn_res

            downsample = Encoder(curr_dim, dim_out, downsample = True, has_attn = has_attn, **block_kwargs)

            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(self.ups, Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs))

            for _ in range(num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn = has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs)

                append(self.downs, enc)
                prepend(self.ups, dec)

            curr_dim = dim_out

        # 处理两个中间解码器

        mid_has_attn = curr_res in attn_res

        self.mids = ModuleList([
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
        ])

        self.out_dim = channels

    @property
    def downsample_factor(self):
        # 返回下采样因子
        return 2 ** self.num_downsamples

    def forward(
        self,
        x,
        time,
        self_cond = None,
        class_labels = None
    ):
        # 验证图像形状是否符合预期

        assert x.shape[1:] == (self.channels, self.image_size, self.image_size)

        # 自身条件

        if self.self_condition:
            # 如果存在自身条件，则将其与输入数据拼接在一起
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim = 1)
        else:
            # 确保不存在自身条件
            assert not exists(self_cond)

        # 时间条件

        time_emb = self.to_time_emb(time)

        # 类别条件

        assert xnor(exists(class_labels), self.needs_class_labels)

        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                # 将类别标签转换为 one-hot 编码
                class_labels = F.one_hot(class_labels, self.num_classes)

            assert class_labels.shape[-1] == self.num_classes
            # 将类别标签转换为浮点数并乘以根号下类别数
            class_labels = class_labels.float() * sqrt(self.num_classes)

            class_emb = self.to_class_emb(class_labels)

            # 将类别嵌入加入到时间嵌入中
            time_emb = self.add_class_emb(time_emb, class_emb)

        # 最终的 mp-silu 嵌入

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

        # 中间层

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
# 定义 MPFeedForward 类，用于实现多头感知器前馈网络
class MPFeedForward(Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 输入维度
        mult = 4,  # 内部维度倍数，默认为4
        mp_add_t = 0.3  # MPAdd 参数，默认为0.3
    ):
        super().__init__()
        dim_inner = int(dim * mult)  # 计算内部维度
        self.net = nn.Sequential(  # 定义网络结构
            PixelNorm(dim = 1),  # 像素归一化
            Conv2d(dim, dim_inner, 1),  # 1x1 卷积
            MPSiLU(),  # MPSiLU激活函数
            Conv2d(dim_inner, dim, 1)  # 1x1 卷积
        )

        self.mp_add = MPAdd(t = mp_add_t)  # 初始化 MPAdd 操作

    # 前向传播函数
    def forward(self, x):
        res = x  # 保存输入
        out = self.net(x)  # 网络前向传播
        return self.mp_add(out, res)  # 返回 MPAdd 操作结果

# 定义 MPImageTransformer 类，用于实现多头图像变换器
class MPImageTransformer(Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 输入维度
        depth,  # 深度
        dim_head = 64,  # 头维度，默认为64
        heads = 8,  # 头数，默认为8
        num_mem_kv = 4,  # 记忆键值对数，默认为4
        ff_mult = 4,  # 前馈网络内部维度倍数，默认为4
        attn_flash = False,  # 是否使用闪回，默认为False
        residual_mp_add_t = 0.3  # 残差 MPAdd 参数，默认为0.3
    ):
        super().__init__()
        self.layers = ModuleList([])  # 初始化层列表

        for _ in range(depth):  # 根据深度循环添加层
            self.layers.append(ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, num_mem_kv = num_mem_kv, flash = attn_flash, mp_add_t = residual_mp_add_t),  # 添加注意力层
                MPFeedForward(dim = dim, mult = ff_mult, mp_add_t = residual_mp_add_t)  # 添加前馈网络层
            ]))

    # 前向传播函数
    def forward(self, x):
        for attn, ff in self.layers:  # 遍历层列表
            x = attn(x)  # 注意力层前向传播
            x = ff(x)  # 前馈网络层前向传播

        return x  # 返回结果

# 定义 InvSqrtDecayLRSched 函数，用于实现反平方根衰减学习率调度
def InvSqrtDecayLRSched(
    optimizer,  # 优化器
    t_ref = 70000,  # 参考时间，默认为70000
    sigma_ref = 0.01  # 参考 Sigma，默认为0.01
):
    """
    refer to equation 67 and Table1
    """
    def inv_sqrt_decay_fn(t: int):  # 定义反平方根衰减函数
        return sigma_ref / sqrt(max(t / t_ref, 1.))  # 返回学习率

    return LambdaLR(optimizer, lr_lambda = inv_sqrt_decay_fn)  # 返回学习率调度器

# 示例
if __name__ == '__main__':
    # 创建 KarrasUnet 实例
    unet = KarrasUnet(
        image_size = 64,
        dim = 192,
        dim_max = 768,
        num_classes = 1000,
    )

    images = torch.randn(2, 4, 64, 64)  # 创建随机输入图像

    # 输入图像进行去噪处理
    denoised_images = unet(
        images,
        time = torch.ones(2,),  # 时间参数
        class_labels = torch.randint(0, 1000, (2,))  # 类别标签
    )

    assert denoised_images.shape == images.shape  # 断言输出形状与输入形状相同
```