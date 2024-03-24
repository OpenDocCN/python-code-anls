# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\karras_unet_1d.py`

```
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

# 将元素转换为元组
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

# 在一维上插值
def interpolate_1d(x, length, mode = 'bilinear'):
    x = rearrange(x, 'b c t -> b c t 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    return rearrange(x, 'b c t 1 -> b c t')

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

# 像素范数层
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

# 一维卷积层
class Conv1d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        init_dirac = False,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size)
        self.weight = nn.Parameter(weight)

        if init_dirac:
            nn.init.dirac_(self.weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size
        self.concat_ones_to_input = concat_ones_to_input
    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 如果处于训练模式
        if self.training:
            # 在不计算梯度的情况下，对权重进行归一化处理
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                # 将归一化后的权重复制给当前权重
                self.weight.copy_(normed_weight)

        # 对权重进行归一化处理，并除以输入特征数的平方根
        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)

        # 如果需要将输入的维度扩展为包含全为1的维度
        if self.concat_ones_to_input:
            # 在输入 x 上进行填充，使得维度增加一维，填充值为1
            x = F.pad(x, (0, 0, 1, 0), value = 1.)

        # 返回一维卷积操作的结果，使用权重 weight 进行卷积，padding 为 'same'
        return F.conv1d(x, weight, padding = 'same')
# 定义线性层模块，继承自 Module 类
class Linear(Module):
    # 初始化函数，接受输入维度、输出维度和 eps 参数
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        # 调用父类的初始化函数
        super().__init__()
        # 生成随机权重矩阵
        weight = torch.randn(dim_out, dim_in)
        # 将权重矩阵设置为可训练参数
        self.weight = nn.Parameter(weight)
        # 设置 eps 属性
        self.eps = eps
        # 记录输入维度
        self.fan_in = dim_in

    # 前向传播函数
    def forward(self, x):
        # 如果处于训练模式
        if self.training:
            # 使用 torch.no_grad() 上下文管理器，不计算梯度
            with torch.no_grad():
                # 对权重矩阵进行归一化处理
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                # 将归一化后的权重矩阵复制给 self.weight
                self.weight.copy_(normed_weight)

        # 对权重矩阵进行归一化处理，并除以输入维度的平方根
        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        # 返回线性变换后的结果
        return F.linear(x, weight)

# MP Fourier Embedding 模块

class MPFourierEmbedding(Module):
    # 初始化函数，接受维度参数
    def __init__(self, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 断言维度能被 2 整除
        assert divisible_by(dim, 2)
        # 计算维度的一半
        half_dim = dim // 2
        # 初始化权重参数，不需要梯度
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    # 前向传播函数
    def forward(self, x):
        # 对输入进行维度重排，增加一个维度
        x = rearrange(x, 'b -> b 1')
        # 计算频率
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # 返回正弦和余弦函数的拼接结果，乘以根号2
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2)

# 构建基础模块

class Encoder(Module):
    # 初始化函数，接受维度、输出维度等参数
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
        # 调用父类的初始化函数
        super().__init__()
        # 如果未指定输出维度，则设为输入维度
        dim_out = default(dim_out, dim)

        # 是否下���样
        self.downsample = downsample
        self.downsample_conv = None

        curr_dim = dim
        # 如果下采样为真
        if downsample:
            # 初始化下采样卷积层
            self.downsample_conv = Conv1d(curr_dim, dim_out, 1)
            curr_dim = dim_out

        # 像素归一化
        self.pixel_norm = PixelNorm(dim = 1)

        self.to_emb = None
        # 如果存在嵌入维度
        if exists(emb_dim):
            # 初始化嵌入层
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        # 第一个块
        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv1d(curr_dim, dim_out, 3)
        )

        # 第二个块
        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv1d(dim_out, dim_out, 3)
        )

        # MPAdd 模块
        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        # 如果有注意力机制
        if has_attn:
            # 初始化注意力层
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
        # 如果下采样为真
        if self.downsample:
            # 对输入进行一维插值，减半长度
            x = interpolate_1d(x, x.shape[-1] // 2, mode = 'bilinear')
            x = self.downsample_conv(x)

        # 对输入进行像素归一化
        x = self.pixel_norm(x)

        # 复制输入作为残差
        res = x.clone()

        # 第一个块的前向传播
        x = self.block1(x)

        # 如果存在嵌入
        if exists(emb):
            # 计算缩放因子
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1')

        # 第二个块的前向传播
        x = self.block2(x)

        # MPAdd 模块的前向传播
        x = self.res_mp_add(x, res)

        # 如果存在注意力层
        if exists(self.attn):
            x = self.attn(x)

        # 返回结果
        return x

# 解码器模块

class Decoder(Module):
    # 初始化函数，接受维度、输出维度等参数
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
            Conv1d(dim, dim_out, 3)
        )

        # 第二个块
        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv1d(dim_out, dim_out, 3)
        )

        # 残差连接的卷积层
        self.res_conv = Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

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
            # 对输入进行一维插值上采样
            x = interpolate_1d(x, x.shape[-1] * 2, mode = 'bilinear')

        # 计算残差连接
        res = self.res_conv(x)

        # 第一个块的操作
        x = self.block1(x)

        # 如果嵌入存在
        if exists(emb):
            # 计算缩放因子
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1')

        # 第二个块的操作
        x = self.block2(x)

        # 执行残差连接的加法操作
        x = self.res_mp_add(x, res)

        # 如果存在注意力机制
        if exists(self.attn):
            # 执行注意力机制操作
            x = self.attn(x)

        # 返回结果
        return x
# 定义一个注意力机制的类，继承自 Module 类
class Attention(Module):
    # 初始化函数，设置注意力机制的参数
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        flash = False,
        mp_add_t = 0.3
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置头数和隐藏维度
        self.heads = heads
        hidden_dim = dim_head * heads

        # 像素归一化
        self.pixel_norm = PixelNorm(dim = -1)

        # 注意力机制
        self.attend = Attend(flash = flash)

        # 记忆键值对
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv1d(dim, hidden_dim * 3, 1)
        self.to_out = Conv1d(hidden_dim, dim, 1)

        # 多路加法
        self.mp_add = MPAdd(t = mp_add_t)

    # 前向传播函数
    def forward(self, x):
        res, b, c, n = x, *x.shape

        # 将输入数据转换为查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h n c', h = self.heads), qkv)

        # 扩展记忆键值对
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        # 对查询、键、值进行像素归一化
        q, k, v = map(self.pixel_norm, (q, k, v))

        # 进行注意力计算
        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)

        return self.mp_add(out, res)

# 定义一个基于 Karras 提出的 Unet 的 1D 版本
class KarrasUnet1D(Module):
    """
    going by figure 21. config G
    """

    # 初始化函数，设置 Unet 的参数
    def __init__(
        self,
        *,
        seq_len,
        dim = 192,
        dim_max = 768,            
        num_classes = None,       
        channels = 4,             
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
    # 初始化函数，继承父类的初始化方法
    ):
        super().__init__()

        # 设置 self_condition 属性
        self.self_condition = self_condition

        # 确定维度

        # 设置通道数和序列长度
        self.channels = channels
        self.seq_len = seq_len
        # 计算输入通道数
        input_channels = channels * (2 if self_condition else 1)

        # 输入和输出块

        # 创建输入块
        self.input_block = Conv1d(input_channels, dim, 3, concat_ones_to_input = True)

        # 创建输出块
        self.output_block = nn.Sequential(
            Conv1d(dim, channels, 3),
            Gain()
        )

        # 时间嵌入

        # 设置嵌入维度
        emb_dim = dim * 4

        # 创建时间嵌入层
        self.to_time_emb = nn.Sequential(
            MPFourierEmbedding(fourier_dim),
            Linear(fourier_dim, emb_dim)
        )

        # 类别嵌入

        # 判断是否需要类别标签
        self.needs_class_labels = exists(num_classes)
        self.num_classes = num_classes

        # 如果需要类别标签
        if self.needs_class_labels:
            # 创建类别嵌入层
            self.to_class_emb = Linear(num_classes, 4 * dim)
            self.add_class_emb = MPAdd(t = mp_add_emb_t)

        # 最终嵌入激活函数

        self.emb_activation = MPSiLU()

        # 下采样数量

        self.num_downsamples = num_downsamples

        # 注意力

        attn_res = set(cast_tuple(attn_res))

        # ResNet 块

        block_kwargs = dict(
            dropout = dropout,
            emb_dim = emb_dim,
            attn_dim_head = attn_dim_head,
            attn_res_mp_add_t = attn_res_mp_add_t,
            attn_flash = attn_flash
        )

        # UNet 编码器和解码器

        self.downs = ModuleList([])
        self.ups = ModuleList([])

        curr_dim = dim
        curr_res = seq_len

        self.skip_mp_cat = MPCat(t = mp_cat_t, dim = 1)

        # 处理初始输入块和前三个编码器块的跳跃连接

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
        return 2 ** self.num_downsamples

    def forward(
        self,
        x,
        time,
        self_cond = None,
        class_labels = None
    ):
        # 验证图像形状是否符合要求

        assert x.shape[1:] == (self.channels, self.seq_len)

        # 自身条件

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

        # 最终的 mp-silu 用于嵌入

        emb = self.emb_activation(time_emb)

        # 跳过连接

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
# 定义一个 MPFeedForward 类，用于实现多头感知器前馈网络
class MPFeedForward(Module):
    # 初始化函数，接收参数 dim（维度）、mult（倍数，默认为4）、mp_add_t（MPAdd 参数，默认为0.3）
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        mp_add_t = 0.3
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算内部维度
        dim_inner = int(dim * mult)
        # 定义网络结构
        self.net = nn.Sequential(
            PixelNorm(dim = 1),  # 对输入进行像素归一化
            Conv2d(dim, dim_inner, 1),  # 1x1 卷积层
            MPSiLU(),  # MP SiLU 激活函数
            Conv2d(dim_inner, dim, 1)  # 1x1 卷积层
        )

        # 初始化 MPAdd 模块
        self.mp_add = MPAdd(t = mp_add_t)

    # 前向传播函数
    def forward(self, x):
        res = x
        out = self.net(x)  # 网络前向传播
        return self.mp_add(out, res)  # 返回 MPAdd 模块的输出结果

# 定义一个 MPImageTransformer 类，用于实现多头图像变换器
class MPImageTransformer(Module):
    # 初始化函数，接收参数 dim（维度）、depth（深度）、dim_head（头部维度，默认为64）、heads（头数，默认为8）、num_mem_kv（记忆键值对数，默认为4）、ff_mult（前馈网络倍数，默认为4）、attn_flash（是否使用闪回，默认为False）、residual_mp_add_t（MPAdd 参数，默认为0.3）
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
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化网络层列表
        self.layers = ModuleList([])

        # 根据深度循环添加注意力和前馈网络层
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, num_mem_kv = num_mem_kv, flash = attn_flash, mp_add_t = residual_mp_add_t),  # 添加注意力层
                MPFeedForward(dim = dim, mult = ff_mult, mp_add_t = residual_mp_add_t)  # 添加前馈网络层
            ]))

    # 前向传播函数
    def forward(self, x):
        # 遍历网络层列表
        for attn, ff in self.layers:
            x = attn(x)  # 注意力层前向传播
            x = ff(x)  # 前馈网络层前向传播

        return x  # 返回输出结果

# 示例代码
if __name__ == '__main__':
    # 创建 KarrasUnet1D 实例
    unet = KarrasUnet1D(
        seq_len = 64,
        dim = 192,
        dim_max = 768,
        num_classes = 1000,
    )

    # 生成随机输入图像
    images = torch.randn(2, 4, 64)

    # 使用 unet 进行图像去噪
    denoised_images = unet(
        images,
        time = torch.ones(2,),
        class_labels = torch.randint(0, 1000, (2,))
    )

    # 断言去噪后的图像形状与原始图像形状相同
    assert denoised_images.shape == images.shape
```